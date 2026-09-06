import Foundation

// Concatenation of Metal-resident columns.
//
// Every other operator in the engine is a GPU kernel; this one is a memcpy. The buffers are unified
// memory, so appending B's bytes after A's is a `memcpy` at ~100 GB/s with no transfer and no kernel
// launch, and concatenation is always a *materialising* boundary anyway (union, the right tail of a
// full outer join, the two sides of a join key before densification). Doing it with a kernel would
// need a launch, a barrier and a second buffer for the bitmap merge, and would not go faster.
//
// Validity bitmaps are the one part that is not a straight memcpy: Arrow packs 8 rows per byte in LSB
// order, so a run that starts at a bit offset which is not a multiple of 8 has to be shifted. The
// byte-aligned case (the common one, because the first input starts at 0) is a memcpy too.

/// Sets `count` bits starting at `dstOffset` in `dst` from `src` (nil means "all valid").
func copyValidityBits(dst: UnsafeMutablePointer<UInt8>, dstOffset: Int,
                      src: UnsafePointer<UInt8>?, count: Int) {
    guard count > 0 else { return }
    guard let src else {
        for i in 0..<count { Bitmap.set(dst, dstOffset + i) }
        return
    }
    if dstOffset % 8 == 0 {
        let whole = count / 8
        if whole > 0 { memcpy(dst + dstOffset / 8, src, whole) }
        for i in (whole * 8)..<count where Bitmap.isSet(src, i) { Bitmap.set(dst, dstOffset + i) }
        return
    }
    for i in 0..<count where Bitmap.isSet(src, i) { Bitmap.set(dst, dstOffset + i) }
}

/// Concatenates arrays of one Arrow type into a single column, in the order given.
///
/// Supported: every fixed-width primitive, boolean, utf8/binary, temporal and dictionary-encoded
/// columns whose dictionaries are identical (otherwise the codes are decoded first). Nested types are
/// rejected with an error naming the format.
public func concatMetalArrays(_ arrays: [AnyMetalArray]) throws -> AnyMetalArray {
    guard let head = arrays.first else {
        throw ArrowMetalError.invalidArrowArray("concat needs at least one array")
    }
    if arrays.count == 1 { return head }
    let ctx = head.metalContext
    try ctx.flush()                       // the inputs must be materialised before the CPU reads them

    func primitive<T: ArrowPrimitive>(_ pick: (AnyMetalArray) -> MetalArray<T>?) throws -> AnyMetalArray {
        var parts: [MetalArray<T>] = []
        for a in arrays {
            guard let p = pick(a) else {
                throw ArrowMetalError.unsupportedType("concat: mixed column types (\(head.arrowFormat) and \(a.arrowFormat))")
            }
            parts.append(p)
        }
        let total = parts.reduce(0) { $0 + $1.length }
        let anyNulls = parts.contains { $0.validity != nil }
        let values = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * T.byteWidth, 1), zeroed: false, context: ctx)
        let validity = anyNulls ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx) : nil
        var row = 0, nulls = 0
        let vp = values.mutableContents
        for p in parts {
            if p.length > 0 {
                memcpy(vp.advanced(by: row * T.byteWidth), p.values.contents, p.length * T.byteWidth)
            }
            if let bm = validity {
                copyValidityBits(dst: bm.mutableTyped(UInt8.self), dstOffset: row,
                                 src: p.validity.map { $0.typed(UInt8.self) }, count: p.length)
            }
            nulls += p.nullCount
            row += p.length
        }
        return arrowMetalWrap(MetalArray<T>(length: total, nullCount: nulls, validity: validity,
                                            values: values, context: ctx))
    }

    switch head {
    case .int8: return try primitive { $0.typedPart(Int8.self) }
    case .int16: return try primitive { $0.typedPart(Int16.self) }
    case .int32: return try primitive { $0.typedPart(Int32.self) }
    case .int64: return try primitive { $0.typedPart(Int64.self) }
    case .uint8: return try primitive { $0.typedPart(UInt8.self) }
    case .uint16: return try primitive { $0.typedPart(UInt16.self) }
    case .uint32: return try primitive { $0.typedPart(UInt32.self) }
    case .uint64: return try primitive { $0.typedPart(UInt64.self) }
    case .float32: return try primitive { $0.typedPart(Float.self) }
    case .float64: return try primitive { $0.typedPart(Double.self) }

    case .boolean:
        var parts: [MetalBooleanArray] = []
        for a in arrays {
            guard case .boolean(let b) = a else {
                throw ArrowMetalError.unsupportedType("concat: mixed column types (boolean and \(a.arrowFormat))")
            }
            parts.append(b)
        }
        let total = parts.reduce(0) { $0 + $1.length }
        let anyNulls = parts.contains { $0.validity != nil }
        let values = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx)
        let validity = anyNulls ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx) : nil
        var row = 0, nulls = 0
        for p in parts {
            copyValidityBits(dst: values.mutableTyped(UInt8.self), dstOffset: row,
                             src: p.length > 0 ? p.values.typed(UInt8.self) : nil, count: p.length)
            if let bm = validity {
                copyValidityBits(dst: bm.mutableTyped(UInt8.self), dstOffset: row,
                                 src: p.validity.map { $0.typed(UInt8.self) }, count: p.length)
            }
            nulls += p.nullCount
            row += p.length
        }
        return .boolean(MetalBooleanArray(length: total, nullCount: nulls, validity: validity, values: values, context: ctx))

    case .string, .binary:
        var parts: [MetalStringArray] = []
        var binary = false
        for a in arrays {
            switch a {
            case .string(let s): parts.append(s)
            case .binary(let s): parts.append(s); binary = true
            default: throw ArrowMetalError.unsupportedType("concat: mixed column types (utf8 and \(a.arrowFormat))")
            }
        }
        let total = parts.reduce(0) { $0 + $1.length }
        let totalBytes = parts.reduce(0) { $0 + $1.totalBytes }
        let anyNulls = parts.contains { $0.validity != nil }
        let offs = try MetalArrowBuffer.allocate(byteCount: (total + 1) * 4, zeroed: false, context: ctx)
        let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalBytes, 1), zeroed: false, context: ctx)
        let validity = anyNulls ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx) : nil
        let op = offs.mutableTyped(Int32.self)
        var row = 0, byte = 0, nulls = 0
        for p in parts {
            let po = p.offsets.typed(Int32.self)
            for i in 0..<p.length { op[row + i] = Int32(byte) + po[i] - po[0] }
            let n = p.totalBytes - Int(po[0])
            if n > 0 { memcpy(data.mutableContents.advanced(by: byte), p.data.contents.advanced(by: Int(po[0])), n) }
            if let bm = validity {
                copyValidityBits(dst: bm.mutableTyped(UInt8.self), dstOffset: row,
                                 src: p.validity.map { $0.typed(UInt8.self) }, count: p.length)
            }
            nulls += p.nullCount
            row += p.length
            byte += n
        }
        op[total] = Int32(byte)
        let s = MetalStringArray(length: total, nullCount: nulls, validity: validity, offsets: offs, data: data, context: ctx)
        s.isBinary = binary
        return binary ? .binary(s) : .string(s)

    case .temporal(let t):
        var storages: [AnyMetalArray] = []
        for a in arrays {
            guard case .temporal(let x) = a, x.type == t.type else {
                throw ArrowMetalError.unsupportedType("concat: mixed temporal types (\(head.arrowFormat) and \(a.arrowFormat))")
            }
            switch x.storage {
            case .int32(let v): storages.append(.int32(v))
            case .int64(let v): storages.append(.int64(v))
            }
        }
        let merged = try concatMetalArrays(storages)
        switch merged {
        case .int32(let v): return .temporal(try MetalTemporalArray(type: t.type, v))
        case .int64(let v): return .temporal(try MetalTemporalArray(type: t.type, v))
        default: throw ArrowMetalError.unsupportedType("concat: temporal storage")
        }

    case .dictionary:
        // Dictionaries need not agree between the parts, so decode and concatenate the values.
        return try concatMetalArrays(arrays.map { a -> AnyMetalArray in
            if case .dictionary = a { return (try? a.dictionaryDecoded()) ?? a }
            return a
        })

    case .extended(let e):
        return try concatMetalArrays(arrays.map { a -> AnyMetalArray in
            if case .extended(let x) = a { return x.storage }
            return a
        }).markingExtension(like: e)

    default:
        throw ArrowMetalError.unsupportedType("concat of \(head.arrowFormat) columns is not implemented")
    }
}

extension AnyMetalArray {
    /// The concrete `MetalArray<T>` inside, when this case carries one of that exact type.
    func typedPart<T: ArrowPrimitive>(_: T.Type) -> MetalArray<T>? {
        switch self {
        case .int8(let a): return a as? MetalArray<T>
        case .int16(let a): return a as? MetalArray<T>
        case .int32(let a): return a as? MetalArray<T>
        case .int64(let a): return a as? MetalArray<T>
        case .uint8(let a): return a as? MetalArray<T>
        case .uint16(let a): return a as? MetalArray<T>
        case .uint32(let a): return a as? MetalArray<T>
        case .uint64(let a): return a as? MetalArray<T>
        case .float32(let a): return a as? MetalArray<T>
        case .float64(let a): return a as? MetalArray<T>
        default: return nil
        }
    }

    func dictionaryDecoded() throws -> AnyMetalArray {
        guard case .dictionary(let codes, let values) = self else { return self }
        return try values.take(codes)
    }

    func markingExtension(like e: MetalExtensionArray) -> AnyMetalArray {
        .extended(MetalExtensionArray(storage: self, name: e.name, metadata: e.metadata,
                                      otherMetadata: e.otherMetadata))
    }
}

/// Concatenates record batches with identical schemas (Polars `concat`, SQL `UNION ALL`).
public func concatRecordBatches(_ batches: [MetalRecordBatch]) throws -> MetalRecordBatch {
    guard let head = batches.first else {
        throw ArrowMetalError.invalidArrowArray("union needs at least one input")
    }
    if batches.count == 1 { return head }
    for b in batches.dropFirst() where b.names != head.names {
        throw ArrowMetalError.invalidArrowArray("union: column names differ (\(head.names) vs \(b.names))")
    }
    var cols: [AnyMetalArray] = []
    for i in head.names.indices {
        cols.append(try concatMetalArrays(batches.map { $0.columns[i] }))
    }
    return try MetalRecordBatch(names: head.names, columns: cols)
}
