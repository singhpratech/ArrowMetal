import Foundation

/// Arrow `pivot_wider`: a scalar aggregate that folds a key column and a value column into one struct
/// row with a named field per expected key.
///
/// This one runs on the **host**. It is a single pass over the key column deciding, per row, which of
/// `keyNames` that row belongs to — a string comparison or an integer comparison against a list that
/// is typically a handful of entries long — and then one `take` of a single row per field on the GPU.
/// There is no parallel work worth a kernel here: the output is one row wide however long the input
/// is, and the answer is decided by the first non-null value found for each key.
public enum PivotWider {

    /// What to do with a pivot key that is not in `keyNames`.
    public enum UnexpectedKey: Sendable {
        /// Arrow's default: rows with an unexpected key contribute nothing.
        case ignore
        /// Raise, as Arrow's `"raise"` does.
        case raise
    }

    /// Arrow `pivot_wider(pivot_keys, pivot_values, key_names)`.
    ///
    /// The result is a one-row struct whose fields are `keyNames`, in that order, each of the value
    /// column's type. A key that never appears (or appears only with a null value) gives a null
    /// field. A key that carries more than one **non-null** value raises, as Arrow does.
    ///
    /// The key column may be `utf8`, `binary` or any integer type; `keyNames` are matched as written
    /// for a string key column and parsed as integers for an integer one, which is Arrow's "the key
    /// names are cast to the pivot key column type" rule.
    public static func pivot(keys: AnyMetalArray, values: AnyMetalArray, keyNames: [String],
                             unexpectedKey: UnexpectedKey = .ignore) throws -> MetalStructArray {
        let n = keys.length
        guard values.length == n else { throw ArrowMetalError.lengthMismatch(n, values.length) }
        guard Set(keyNames).count == keyNames.count else {
            throw ArrowMetalError.invalidArrowArray("pivot_wider key_names must be unique")
        }
        let fieldOf = try keyResolver(keys, keyNames: keyNames)
        var source = [Int32?](repeating: nil, count: keyNames.count)
        for i in 0..<n {
            guard let field = try fieldOf(i) else { continue }
            guard field >= 0 else {
                if case .raise = unexpectedKey {
                    throw ArrowMetalError.invalidArrowArray("pivot_wider: unexpected pivot key at row \(i)")
                }
                continue
            }
            guard try valueIsValid(values, i) else { continue }
            if source[field] != nil {
                throw ArrowMetalError.invalidArrowArray("pivot_wider: duplicate non-null value for key '\(keyNames[field])'")
            }
            source[field] = Int32(i)
        }
        let ctx = keys.context
        let children = try source.map { row -> AnyMetalArray in
            try values.take(try MetalArray<Int32>([row], context: ctx))
        }
        return try MetalStructArray(names: keyNames, children: children, context: ctx)
    }

    /// A closure mapping a row to its field index, `-1` for a key that is not in `keyNames`, and nil
    /// for a null key (which contributes nothing whatever `unexpectedKey` says).
    private static func keyResolver(_ keys: AnyMetalArray, keyNames: [String]) throws -> (Int) throws -> Int? {
        var byName: [String: Int] = [:]
        for (i, k) in keyNames.enumerated() { byName[k] = i }
        switch keys {
        case .string(let s), .binary(let s):
            return { i in s[i].flatMap { byName[$0] } ?? (s.isValid(i) ? -1 : nil) }
        case .int8(let a): return integerResolver(a, keyNames)
        case .uint8(let a): return integerResolver(a, keyNames)
        case .int16(let a): return integerResolver(a, keyNames)
        case .uint16(let a): return integerResolver(a, keyNames)
        case .int32(let a): return integerResolver(a, keyNames)
        case .uint32(let a): return integerResolver(a, keyNames)
        case .int64(let a): return integerResolver(a, keyNames)
        case .uint64(let a): return integerResolver(a, keyNames)
        case .dictionary(let codes, let dictValues):
            // The key of a row is the dictionary value its code names; resolve the values once.
            let inner = try keyResolver(dictValues, keyNames: keyNames)
            return { i in try codes[i].flatMap { try inner(Int($0)) } }
        case .extended(let e): return try keyResolver(e.storage, keyNames: keyNames)
        case .float32, .float64, .boolean, .temporal, .decimal, .list, .structure, .map, .union, .runEndEncoded,
             .null, .float16, .smallDecimal, .interval, .fixedBinary:
            throw ArrowMetalError.unsupportedType("pivot_wider needs a string, binary or integer key column, got \(keys.arrowFormat)")
        }
    }

    /// Integer keys match by their canonical decimal rendering, so every width — int8 through
    /// uint64 — compares exactly, with no detour through `Double`.
    private static func integerResolver<T: ArrowPrimitive>(_ a: MetalArray<T>, _ keyNames: [String]) -> (Int) -> Int? {
        var byValue: [String: Int] = [:]
        for (i, k) in keyNames.enumerated() {
            if let v = Int64(k) { byValue[String(v)] = i } else if let u = UInt64(k) { byValue[String(u)] = i }
        }
        return { i in a[i].map { byValue["\($0)"] ?? -1 } }
    }

    private static func valueIsValid(_ values: AnyMetalArray, _ i: Int) throws -> Bool {
        switch values {
        case .int8(let a): return a.isValid(i)
        case .uint8(let a): return a.isValid(i)
        case .int16(let a): return a.isValid(i)
        case .uint16(let a): return a.isValid(i)
        case .int32(let a): return a.isValid(i)
        case .uint32(let a): return a.isValid(i)
        case .int64(let a): return a.isValid(i)
        case .uint64(let a): return a.isValid(i)
        case .float32(let a): return a.isValid(i)
        case .float64(let a): return a.isValid(i)
        case .boolean(let a): return a.isValid(i)
        case .string(let a), .binary(let a): return a.isValid(i)
        case .temporal(let a): return a.isValid(i)
        case .decimal(let a): return a.isValid(i)
        case .dictionary(let codes, _): return codes.isValid(i)
        case .list(let a): return a.isValid(i)
        case .structure(let a): return a.isValid(i)
        case .map(let a): return a.isValid(i)
        case .union:
            throw ArrowMetalError.unsupportedType("pivot_wider: a union value column has no validity bitmap")
        case .runEndEncoded:
            throw ArrowMetalError.unsupportedType("pivot_wider: decode the run-end encoded value column first")
        case .null: return false
        case .float16(let a): return a.isValid(i)
        case .smallDecimal(let a): return a.isValid(i)
        case .interval(let a): return a.isValid(i)
        case .fixedBinary(let a): return a.isValid(i)
        case .extended(let e): return try valueIsValid(e.storage, i)
        }
    }
}

extension AnyMetalArray {
    /// The `MetalContext` this column's buffers live in.
    public var context: MetalContext {
        switch self {
        case .int8(let a): return a.context
        case .uint8(let a): return a.context
        case .int16(let a): return a.context
        case .uint16(let a): return a.context
        case .int32(let a): return a.context
        case .uint32(let a): return a.context
        case .int64(let a): return a.context
        case .uint64(let a): return a.context
        case .float32(let a): return a.context
        case .float64(let a): return a.context
        case .boolean(let a): return a.context
        case .string(let a): return a.context
        case .temporal(let a): return a.context
        case .binary(let a): return a.context
        case .decimal(let a): return a.context
        case .dictionary(let codes, _): return codes.context
        case .list(let a): return a.context
        case .structure(let a): return a.context
        case .map(let a): return a.context
        case .union(let a): return a.context
        case .runEndEncoded(let runEnds, _): return runEnds.context
        case .null(let a): return a.context
        case .float16(let a): return a.context
        case .smallDecimal(let a): return a.context
        case .interval(let a): return a.context
        case .fixedBinary(let a): return a.context
        case .extended(let e): return e.storage.context
        }
    }
}
