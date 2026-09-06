import Foundation
import Metal

// Compute over dictionary-encoded arrays *without decoding them*.
//
// A dictionary column is int32 codes plus a (usually much smaller) values array. Every function here
// works on whichever of the two is cheaper:
//
// | function | what runs |
// |---|---|
// | `compare` (scalar) | the dictionary is compared once (`values.length` elements), and the boolean result is gathered by the codes |
// | `filter` / `take` / `slice` | the codes only; the values array is shared with the input, not copied (`AnyMetalArray.filter` / `take` / `slice`) |
// | `unique` / `value_counts` | `unique` / `value_counts` over the codes, then one `take` of the values |
// | `group_by` | the codes *are* the dense keys the existing `GroupBy` wants: no hashing, no decode |
// | `dictionaryEncoded()` | dictionary encoding for primitive, temporal, boolean (GPU) and utf8 / binary (host hash map) columns |
//
// The saving is the point: comparing a 5-value dictionary over a 300k-row column touches 5 elements plus
// one gather, instead of 300k comparisons.

extension AnyMetalArray {

    // MARK: - compare on the dictionary

    /// Arrow `equal` / `less` / ... between a dictionary column and a scalar, evaluated on the dictionary.
    ///
    /// The dictionary (not the column) is compared, and the resulting booleans are gathered by the codes,
    /// so the cost is `values.length` comparisons plus one gather. A null code, or a null dictionary
    /// entry, yields a null result — the same answer `decode()` then `compare` would give.
    public func dictionaryCompare<T: ArrowPrimitive>(_ op: CompareOp, _ scalar: T) throws -> MetalBooleanArray {
        guard case .dictionary(let codes, let values) = self else {
            throw ArrowMetalError.unsupportedType("dictionaryCompare needs a dictionary array, got \(arrowFormat)")
        }
        guard let typed: MetalArray<T> = dictionaryValues(values) else {
            throw ArrowMetalError.unsupportedType("dictionary values are \(values.arrowFormat), not \(T.arrowFormat)")
        }
        return try DictionaryCompute.gather(try typed.compare(op, scalar), by: codes)
    }

    /// Equality (and inequality) between a utf8 or binary dictionary column and a scalar string.
    /// Ordering comparisons are not defined for strings in this package, so only `.eq` and `.ne` are accepted.
    public func dictionaryCompare(_ op: CompareOp, _ scalar: String) throws -> MetalBooleanArray {
        guard case .dictionary(let codes, let values) = self else {
            throw ArrowMetalError.unsupportedType("dictionaryCompare needs a dictionary array, got \(arrowFormat)")
        }
        guard case .string(let s) = values else {
            if case .binary(let b) = values {
                let equal = try b.equals(scalar)
                return try DictionaryCompute.gather(op == .eq ? equal : try equal.not(), by: codes)
            }
            throw ArrowMetalError.unsupportedType("dictionary values are \(values.arrowFormat), not utf8")
        }
        guard op == .eq || op == .ne else {
            throw ArrowMetalError.unsupportedType("string dictionaries support == and != only, not \(op.rawValue)")
        }
        let equal = try s.equals(scalar)
        return try DictionaryCompute.gather(op == .eq ? equal : try equal.not(), by: codes)
    }

    // MARK: - unique / value_counts on the codes

    /// Arrow `unique` over a dictionary column: the distinct values actually used, in dictionary order.
    /// Runs `unique()` over the codes (typically a tiny array) and gathers the values once.
    public func dictionaryUnique() throws -> AnyMetalArray {
        guard case .dictionary(let codes, let values) = self else {
            throw ArrowMetalError.unsupportedType("dictionaryUnique needs a dictionary array, got \(arrowFormat)")
        }
        return try values.take(try codes.unique())
    }

    /// Arrow `value_counts` over a dictionary column: the distinct values used and how many rows carry each.
    /// Runs `value_counts` over the codes, so the values array is read only for the final gather.
    public func dictionaryValueCounts() throws -> (values: AnyMetalArray, counts: MetalArray<Int64>) {
        guard case .dictionary(let codes, let values) = self else {
            throw ArrowMetalError.unsupportedType("dictionaryValueCounts needs a dictionary array, got \(arrowFormat)")
        }
        let (usedCodes, counts) = try codes.valueCounts()
        return (try values.take(usedCodes), counts)
    }

    // MARK: - group_by on the codes

    /// A `GroupBy` over a dictionary column's codes plus the values each group belongs to.
    ///
    /// The codes are already the dense keys `GroupBy` wants, so this is free: no hashing, no decode.
    /// Group `k` belongs to `values[k]`; groups for dictionary entries no row uses come back empty.
    public func dictionaryGroupBy() throws -> (GroupBy<Int32>, values: AnyMetalArray) {
        guard case .dictionary(let codes, let values) = self else {
            throw ArrowMetalError.unsupportedType("dictionaryGroupBy needs a dictionary array, got \(arrowFormat)")
        }
        return (try GroupBy(keys: codes, keyCount: Swift.max(values.length, 1)), values)
    }

    // MARK: - dictionary_encode

    /// Arrow `dictionary_encode`: turns any supported column into `.dictionary(codes:values:)`.
    ///
    /// Primitive, temporal and boolean columns are encoded on the GPU (sort, mark runs, scan — the
    /// `unique()` pipeline); utf8 and binary columns use the existing host hash map. A column that is
    /// already dictionary encoded is returned unchanged.
    public func dictionaryEncoded() throws -> AnyMetalArray {
        switch self {
        case .dictionary: return self
        case .int8(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .int8(r.unique))
        case .uint8(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .uint8(r.unique))
        case .int16(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .int16(r.unique))
        case .uint16(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .uint16(r.unique))
        case .int32(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .int32(r.unique))
        case .uint32(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .uint32(r.unique))
        case .int64(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .int64(r.unique))
        case .uint64(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .uint64(r.unique))
        case .float32(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .float32(r.unique))
        case .float64(let a): let r = try a.dictionaryEncode(); return .dictionary(codes: r.codes, values: .float64(r.unique))
        case .boolean(let a):
            let r = try a.toUInt8Array().dictionaryEncode()
            return .dictionary(codes: r.codes, values: .boolean(try MetalBooleanArray.fromUInt8Array(r.unique)))
        case .string(let s):
            let r = try s.dictionaryEncode()
            return .dictionary(codes: r.codes, values: .string(r.unique))
        case .binary(let s):
            let r = try s.dictionaryEncode()
            return .dictionary(codes: r.codes, values: .binary(markBinary(r.unique)))
        case .temporal(let t):
            switch t.storage {
            case .int32(let a):
                let r = try a.dictionaryEncode()
                return .dictionary(codes: r.codes, values: .temporal(try MetalTemporalArray(type: t.type, r.unique)))
            case .int64(let a):
                let r = try a.dictionaryEncode()
                return .dictionary(codes: r.codes, values: .temporal(try MetalTemporalArray(type: t.type, r.unique)))
            }
        case .runEndEncoded:
            return try runEndDecode().dictionaryEncoded()
        case .extended(let e): return try e.storage.dictionaryEncoded()
        case .decimal, .list, .structure, .map, .union, .null, .float16, .smallDecimal, .interval, .fixedBinary:
            throw ArrowMetalError.unsupportedType("dictionary encoding of \(arrowFormat) is not implemented")
        }
    }
}

/// Turns an `AnyMetalArray` of dictionary values into a typed array when the element type matches.
/// Temporal values are unwrapped to their storage integers, which is what a comparison needs.
private func dictionaryValues<T: ArrowPrimitive>(_ a: AnyMetalArray) -> MetalArray<T>? {
    switch a {
    case .int8(let x): return x as? MetalArray<T>
    case .uint8(let x): return x as? MetalArray<T>
    case .int16(let x): return x as? MetalArray<T>
    case .uint16(let x): return x as? MetalArray<T>
    case .int32(let x): return x as? MetalArray<T>
    case .uint32(let x): return x as? MetalArray<T>
    case .int64(let x): return x as? MetalArray<T>
    case .uint64(let x): return x as? MetalArray<T>
    case .float32(let x): return x as? MetalArray<T>
    case .float64(let x): return x as? MetalArray<T>
    case .temporal(let t):
        switch t.storage {
        case .int32(let x): return x as? MetalArray<T>
        case .int64(let x): return x as? MetalArray<T>
        }
    default: return nil
    }
}

enum DictionaryCompute {
    /// Maps a per-dictionary-entry boolean result onto the rows through the codes.
    ///
    /// The booleans become one byte per entry (GPU), the codes gather those bytes (GPU `take`, which
    /// carries both the entry's and the code's null-ness), and the bytes are packed back into a bitmap.
    static func gather(_ result: MetalBooleanArray, by codes: MetalArray<Int32>) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(try result.toUInt8Array().take(codes))
    }
}
