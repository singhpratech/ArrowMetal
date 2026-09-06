import Foundation
import Metal

// Host-side helpers the streaming merge stages need: concatenating Metal-resident columns without a
// round trip through Arrow IPC, and a small type-erased value used as a hash key (the streaming
// group-by's global table) and as the comparison key of the external sort's k-way merge.
//
// Everything here runs on the CPU on purpose. The GPU produces *small* per-batch results — one row
// per group, k rows for a top-k, a sketch — and the merge of those results is a few thousand rows
// per batch, so a host merge costs microseconds and keeps the GPU free for the next batch. The one
// place a host loop touches every row is the external sort's k-way merge, which is documented as a
// CPU merge in docs/STREAMING.md.

/// A single Arrow value, type erased, ordered and hashable.
///
/// Used as the key of the streaming group-by's global table and as the sort key of the external
/// sort's merge. `null` sorts last, matching `argsort`'s placement of nulls.
public enum StreamValue: Hashable, Comparable, @unchecked Sendable {
    case null
    case int(Int64)
    case uint(UInt64)
    case double(Double)
    case bool(Bool)
    case string(String)
    case bytes([UInt8])

    /// A Double view of a numeric value, for aggregate merges that widen.
    public var asDouble: Double? {
        switch self {
        case .int(let v): return Double(v)
        case .uint(let v): return Double(v)
        case .double(let v): return v
        case .bool(let v): return v ? 1 : 0
        default: return nil
        }
    }

    public var isNull: Bool { if case .null = self { return true } else { return false } }

    private var rank: Int {
        switch self {
        case .int, .uint, .double, .bool: return 0
        case .string, .bytes: return 1
        case .null: return 2      // nulls last
        }
    }

    public static func < (a: StreamValue, b: StreamValue) -> Bool { less(a, b) }

    /// Total order used by the streaming merges (a named function, so the `<` of `Expr` and
    /// `MetalArray` cannot be picked by overload resolution in a generic context).
    public static func less(_ a: StreamValue, _ b: StreamValue) -> Bool {
        if a.rank != b.rank { return a.rank < b.rank }
        switch (a, b) {
        case (.null, .null): return false
        case (.string(let x), .string(let y)): return x < y
        case (.bytes(let x), .bytes(let y)): return x.lexicographicallyPrecedes(y)
        case (.string(let x), .bytes(let y)): return Array(x.utf8).lexicographicallyPrecedes(y)
        case (.bytes(let x), .string(let y)): return x.lexicographicallyPrecedes(Array(y.utf8))
        case (.int(let x), .int(let y)): return x < y  // Int64
        case (.uint(let x), .uint(let y)): return x < y
        case (.bool(let x), .bool(let y)): return !x && y
        default:
            // Mixed numeric shapes (an int column merged against a float one) compare as doubles.
            return (a.asDouble ?? 0) < (b.asDouble ?? 0)
        }
    }
}

// MARK: - Reading a Metal column out to host values

extension AnyMetalArray {
    /// Copies this column out to host values. Only used on the *small* results the GPU hands the merge
    /// stage (group keys, top-k rows, sort-run rows), never on a whole streamed batch.
    public func streamValues() throws -> [StreamValue] {
        switch self {
        case .int8(let a): return a.toArray().map { $0.map { .int(Int64($0)) } ?? .null }
        case .int16(let a): return a.toArray().map { $0.map { .int(Int64($0)) } ?? .null }
        case .int32(let a): return a.toArray().map { $0.map { .int(Int64($0)) } ?? .null }
        case .int64(let a): return a.toArray().map { $0.map { .int($0) } ?? .null }
        case .uint8(let a): return a.toArray().map { $0.map { .uint(UInt64($0)) } ?? .null }
        case .uint16(let a): return a.toArray().map { $0.map { .uint(UInt64($0)) } ?? .null }
        case .uint32(let a): return a.toArray().map { $0.map { .uint(UInt64($0)) } ?? .null }
        case .uint64(let a): return a.toArray().map { $0.map { .uint($0) } ?? .null }
        case .float32(let a): return a.toArray().map { $0.map { .double(Double($0)) } ?? .null }
        case .float64(let a): return a.toArray().map { $0.map { .double($0) } ?? .null }
        case .boolean(let a): return a.toArray().map { $0.map { .bool($0) } ?? .null }
        case .string(let a): return a.toArray().map { $0.map { .string($0) } ?? .null }
        case .binary(let a):
            return (0..<a.length).map { i -> StreamValue in
                guard a.isValid(i) else { return .null }
                let o = a.offsets.typed(Int32.self), d = a.data.typed(UInt8.self)
                return .bytes(Array(UnsafeBufferPointer(start: d + Int(o[i]), count: Int(o[i + 1] - o[i]))))
            }
        case .temporal(let a): return a.toArray().map { $0.map { .int($0) } ?? .null }
        case .dictionary: return try dictionaryDecoded().streamValues()
        default:
            throw ArrowMetalError.unsupportedType("streaming merge does not support \(arrowFormat) key/value columns")
        }
    }

    /// A dictionary column decoded to its value type; every other column unchanged.
    func dictionaryDecoded() throws -> AnyMetalArray { try decode() }

    /// Rebuilds a column of this column's own type from host values.
    public func rebuild(_ vals: [StreamValue], context: MetalContext? = nil) throws -> AnyMetalArray {
        let ctx = context ?? anyContext
        func ints() -> [Int64?] { vals.map { v in
            switch v { case .int(let x): return x; case .uint(let x): return Int64(bitPattern: x)
            case .double(let x): return Int64(x); case .bool(let b): return b ? 1 : 0; default: return nil } } }
        func uints() -> [UInt64?] { vals.map { v in
            switch v { case .uint(let x): return x; case .int(let x): return UInt64(bitPattern: x)
            case .double(let x): return UInt64(x); case .bool(let b): return b ? 1 : 0; default: return nil } } }
        func dbls() -> [Double?] { vals.map { $0.isNull ? nil : $0.asDouble } }
        func strs() -> [String?] { vals.map { v in
            switch v { case .string(let s): return s; case .bytes(let b): return String(decoding: b, as: UTF8.self)
            case .int(let x): return String(x); case .double(let x): return String(x); default: return nil } } }
        switch self {
        case .int8: return .int8(try MetalArray<Int8>(ints().map { $0.map { Int8(truncatingIfNeeded: $0) } }, context: ctx))
        case .int16: return .int16(try MetalArray<Int16>(ints().map { $0.map { Int16(truncatingIfNeeded: $0) } }, context: ctx))
        case .int32: return .int32(try MetalArray<Int32>(ints().map { $0.map { Int32(truncatingIfNeeded: $0) } }, context: ctx))
        case .int64: return .int64(try MetalArray<Int64>(ints(), context: ctx))
        case .uint8: return .uint8(try MetalArray<UInt8>(uints().map { $0.map { UInt8(truncatingIfNeeded: $0) } }, context: ctx))
        case .uint16: return .uint16(try MetalArray<UInt16>(uints().map { $0.map { UInt16(truncatingIfNeeded: $0) } }, context: ctx))
        case .uint32: return .uint32(try MetalArray<UInt32>(uints().map { $0.map { UInt32(truncatingIfNeeded: $0) } }, context: ctx))
        case .uint64: return .uint64(try MetalArray<UInt64>(uints(), context: ctx))
        case .float32: return .float32(try MetalArray<Float>(dbls().map { $0.map { Float($0) } }, context: ctx))
        case .float64: return .float64(try MetalArray<Double>(dbls(), context: ctx))
        case .boolean:
            let bits = vals.map { v -> Bool? in if case .bool(let b) = v { return b } else if let d = v.asDouble { return d != 0 } else { return nil } }
            return .boolean(try makeBooleanArray(bits, context: ctx))
        case .string: return .string(try MetalStringArray(strs(), context: ctx))
        case .binary:
            let a = try MetalStringArray(strs(), context: ctx)
            a.isBinary = true
            return .binary(a)
        case .temporal(let t): return .temporal(try MetalTemporalArray(type: t.type, ints(), context: ctx))
        case .dictionary: return try dictionaryDecoded().rebuild(vals, context: ctx)
        default:
            throw ArrowMetalError.unsupportedType("streaming merge cannot rebuild a \(arrowFormat) column")
        }
    }
}

/// A nullable boolean array (`MetalBooleanArray` only ships a non-null `[Bool]` initialiser).
public func makeBooleanArray(_ vals: [Bool?], context: MetalContext = .shared) throws -> MetalBooleanArray {
    let n = vals.count
    let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context)
    let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: context)
    let vp = vb.mutableTyped(UInt8.self), bp = bm.mutableTyped(UInt8.self)
    var nulls = 0
    for (i, v) in vals.enumerated() {
        if let v { Bitmap.set(bp, i); if v { Bitmap.set(vp, i) } } else { nulls += 1 }
    }
    return MetalBooleanArray(length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm, values: vb, context: context)
}

// MARK: - Concatenating Metal columns

/// Concatenates equal-typed Metal columns into one, by `memcpy` inside unified memory.
///
/// No GPU work and no Arrow round trip: the values buffers are copied back to back and the validity
/// bitmaps re-packed bit by bit. Used for the top-k candidate merge, the broadcast join's build side
/// and the external sort's output assembly.
public func concatColumns(_ cols: [AnyMetalArray]) throws -> AnyMetalArray {
    guard let head = cols.first else { throw ArrowMetalError.invalidArrowArray("concat of no columns") }
    if cols.count == 1 { return head }
    switch head {
    case .int8: return .int8(try concatPrimitive(cols.map { try pick($0, "int8") { if case .int8(let a) = $0 { return a }; return nil } }))
    case .int16: return .int16(try concatPrimitive(cols.map { try pick($0, "int16") { if case .int16(let a) = $0 { return a }; return nil } }))
    case .int32: return .int32(try concatPrimitive(cols.map { try pick($0, "int32") { if case .int32(let a) = $0 { return a }; return nil } }))
    case .int64: return .int64(try concatPrimitive(cols.map { try pick($0, "int64") { if case .int64(let a) = $0 { return a }; return nil } }))
    case .uint8: return .uint8(try concatPrimitive(cols.map { try pick($0, "uint8") { if case .uint8(let a) = $0 { return a }; return nil } }))
    case .uint16: return .uint16(try concatPrimitive(cols.map { try pick($0, "uint16") { if case .uint16(let a) = $0 { return a }; return nil } }))
    case .uint32: return .uint32(try concatPrimitive(cols.map { try pick($0, "uint32") { if case .uint32(let a) = $0 { return a }; return nil } }))
    case .uint64: return .uint64(try concatPrimitive(cols.map { try pick($0, "uint64") { if case .uint64(let a) = $0 { return a }; return nil } }))
    case .float32: return .float32(try concatPrimitive(cols.map { try pick($0, "float32") { if case .float32(let a) = $0 { return a }; return nil } }))
    case .float64: return .float64(try concatPrimitive(cols.map { try pick($0, "float64") { if case .float64(let a) = $0 { return a }; return nil } }))
    case .boolean: return .boolean(try concatBoolean(cols.map { try pick($0, "bool") { if case .boolean(let a) = $0 { return a }; return nil } }))
    case .string: return .string(try concatStrings(cols.map { try pick($0, "utf8") { if case .string(let a) = $0 { return a }; return nil } }, binary: false))
    case .binary: return .binary(try concatStrings(cols.map { try pick($0, "binary") { if case .binary(let a) = $0 { return a }; return nil } }, binary: true))
    case .temporal(let ht):
        let tt = ht.type
        var i32: [MetalArray<Int32>] = [], i64: [MetalArray<Int64>] = []
        for c in cols {
            guard case .temporal(let a) = c, a.type == tt else {
                throw ArrowMetalError.unsupportedType("concat of mixed temporal types")
            }
            if let x = a.asInt32 { i32.append(x) } else if let x = a.asInt64 { i64.append(x) }
        }
        if !i32.isEmpty { return .temporal(try MetalTemporalArray(type: tt, try concatPrimitive(i32))) }
        return .temporal(try MetalTemporalArray(type: tt, try concatPrimitive(i64)))
    case .dictionary:
        return try concatColumns(try cols.map { try $0.dictionaryDecoded() })
    default:
        throw ArrowMetalError.unsupportedType("streaming concat does not support \(head.arrowFormat)")
    }
}

/// Concatenates record batches that share a schema.
public func concatBatches(_ batches: [MetalRecordBatch]) throws -> MetalRecordBatch {
    guard let head = batches.first else { throw ArrowMetalError.invalidArrowArray("concat of no batches") }
    if batches.count == 1 { return head }
    let nonEmpty = batches.filter { $0.length > 0 }
    if nonEmpty.isEmpty { return head }
    if nonEmpty.count == 1 { return nonEmpty[0] }
    var cols: [AnyMetalArray] = []
    for i in 0..<head.columnCount {
        cols.append(try concatColumns(nonEmpty.map { $0.columns[i] }))
    }
    return try MetalRecordBatch(names: head.names, columns: cols)
}

private func pick<T>(_ c: AnyMetalArray, _ what: String, _ f: (AnyMetalArray) -> T?) throws -> T {
    guard let v = f(c) else { throw ArrowMetalError.unsupportedType("concat expected \(what), got \(c.arrowFormat)") }
    return v
}

func concatPrimitive<T: ArrowPrimitive>(_ arrays: [MetalArray<T>]) throws -> MetalArray<T> {
    guard let head = arrays.first else { throw ArrowMetalError.invalidArrowArray("concat of no arrays") }
    if arrays.count == 1 { return head }
    let ctx = head.context
    let total = arrays.reduce(0) { $0 + $1.length }
    let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * T.byteWidth, 1), zeroed: false, context: ctx)
    var row = 0
    for a in arrays {
        let n = a.length
        if n > 0 { memcpy(out.mutableContents + row * T.byteWidth, a.values.contents, n * T.byteWidth) }
        row += n
    }
    let (validity, nulls) = try concatValidity(arrays.map { ($0.validity, $0.length) }, total: total, context: ctx)
    return MetalArray<T>(length: total, nullCount: nulls, validity: validity, values: out, context: ctx)
}

private func concatBoolean(_ arrays: [MetalBooleanArray]) throws -> MetalBooleanArray {
    let ctx = arrays[0].context
    let total = arrays.reduce(0) { $0 + $1.length }
    let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx)
    let p = out.mutableTyped(UInt8.self)
    var row = 0
    for a in arrays {
        let src = a.values.typed(UInt8.self)
        for i in 0..<a.length where Bitmap.isSet(src, i) { Bitmap.set(p, row + i) }
        row += a.length
    }
    let (validity, nulls) = try concatValidity(arrays.map { ($0.validity, $0.length) }, total: total, context: ctx)
    return MetalBooleanArray(length: total, nullCount: nulls, validity: validity, values: out, context: ctx)
}

private func concatStrings(_ arrays: [MetalStringArray], binary: Bool) throws -> MetalStringArray {
    let ctx = arrays[0].context
    let total = arrays.reduce(0) { $0 + $1.length }
    let totalBytes = arrays.reduce(0) { $0 + $1.totalBytes }
    let off = try MetalArrowBuffer.allocate(byteCount: (total + 1) * 4, zeroed: false, context: ctx)
    let dat = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalBytes, 1), zeroed: false, context: ctx)
    let op = off.mutableTyped(Int32.self)
    var row = 0, pos = 0
    for a in arrays {
        let ao = a.offsets.typed(Int32.self)
        let base = Int(ao[0])
        let bytes = a.totalBytes - base
        if bytes > 0 { memcpy(dat.mutableContents + pos, a.data.contents + base, bytes) }
        for i in 0..<a.length { op[row + i] = Int32(pos + Int(ao[i]) - base) }
        pos += bytes
        row += a.length
    }
    op[total] = Int32(pos)
    let (validity, nulls) = try concatValidity(arrays.map { ($0.validity, $0.length) }, total: total, context: ctx)
    let s = MetalStringArray(length: total, nullCount: nulls, validity: validity, offsets: off, data: dat, context: ctx)
    s.isBinary = binary
    return s
}

/// Packs the validity bitmaps of several arrays into one. Returns nil when no input had nulls.
private func concatValidity(_ parts: [(MetalArrowBuffer?, Int)], total: Int,
                            context: MetalContext) throws -> (MetalArrowBuffer?, Int) {
    guard parts.contains(where: { $0.0 != nil }) else { return (nil, 0) }
    let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: context)
    let p = bm.mutableTyped(UInt8.self)
    var row = 0, nulls = 0
    for (v, n) in parts {
        if let v {
            let vp = v.typed(UInt8.self)
            for i in 0..<n {
                if Bitmap.isSet(vp, i) { Bitmap.set(p, row + i) } else { nulls += 1 }
            }
        } else {
            for i in 0..<n { Bitmap.set(p, row + i) }
        }
        row += n
    }
    return (bm, nulls)
}
