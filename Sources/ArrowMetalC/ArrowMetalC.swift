import Foundation
import CArrowABI
import ArrowMetal

// C ABI over ArrowMetal. Handles are retained boxes; errors are reported through a thread-local string.

final class Box { let a: AnyMetalArray; init(_ a: AnyMetalArray) { self.a = a } }

private let errorKey = "ArrowMetalC.lastError"
private func setError(_ e: Error) { Thread.current.threadDictionary[errorKey] = "\(e)" }
private func lastErrorPtr() -> UnsafePointer<CChar>? {
    let s = (Thread.current.threadDictionary[errorKey] as? String) ?? ""
    // Keep a stable C string per thread.
    if let old = Thread.current.threadDictionary["ArrowMetalC.lastErrorC"] as? UnsafeMutablePointer<CChar> { free(old) }
    let c = strdup(s)!
    Thread.current.threadDictionary["ArrowMetalC.lastErrorC"] = c
    return UnsafePointer(c)
}
private let versionC = strdup("0.1.0")!
private let deviceNameC = strdup(MetalContext.shared.device.name)!
private var formatCache: [String: UnsafeMutablePointer<CChar>] = [:]
private let formatLock = NSLock()

@inline(__always) private func handle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func emit(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
    return 0
}
private func run(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do { return emit(try body(), out) } catch { setError(error); return 1 }
}

@_cdecl("am_version") public func am_version() -> UnsafePointer<CChar>? { UnsafePointer(versionC) }
@_cdecl("am_device_name") public func am_device_name() -> UnsafePointer<CChar>? { UnsafePointer(deviceNameC) }
@_cdecl("am_last_error") public func am_last_error() -> UnsafePointer<CChar>? { lastErrorPtr() }

@_cdecl("am_import")
public func am_import(_ schema: UnsafePointer<ArrowSchema>?, _ array: UnsafeMutablePointer<ArrowArray>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let schema, let array else { return 2 }
    return run(out) { try importArrowArray(schema: schema, array: array).array }
}
@_cdecl("am_import_device")
public func am_import_device(_ schema: UnsafePointer<ArrowSchema>?, _ array: UnsafeMutablePointer<ArrowDeviceArray>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let schema, let array else { return 2 }
    return run(out) { try importArrowDeviceArray(schema: schema, array: array).array }
}
@_cdecl("am_export")
public func am_export(_ a: OpaquePointer?, _ schema: UnsafeMutablePointer<ArrowSchema>?, _ array: UnsafeMutablePointer<ArrowArray>?) -> Int32 {
    guard let x = handle(a), let array else { return 2 }
    if let schema { x.exportArrowSchema(into: schema) }
    x.exportArrowArray(into: array)
    return 0
}
@_cdecl("am_export_device")
public func am_export_device(_ a: OpaquePointer?, _ schema: UnsafeMutablePointer<ArrowSchema>?, _ array: UnsafeMutablePointer<ArrowDeviceArray>?) -> Int32 {
    guard let x = handle(a), let array else { return 2 }
    if let schema { x.exportArrowSchema(into: schema) }
    x.exportArrowDeviceArray(into: array)
    return 0
}
@_cdecl("am_release") public func am_release(_ a: OpaquePointer?) {
    guard let a else { return }
    Unmanaged<Box>.fromOpaque(UnsafeRawPointer(a)).release()
}
@_cdecl("am_length") public func am_length(_ a: OpaquePointer?) -> Int64 { Int64(handle(a)?.length ?? -1) }
@_cdecl("am_null_count") public func am_null_count(_ a: OpaquePointer?) -> Int64 { Int64(handle(a)?.nullCount ?? -1) }
@_cdecl("am_format") public func am_format(_ a: OpaquePointer?) -> UnsafePointer<CChar>? {
    guard let x = handle(a) else { return nil }
    formatLock.lock(); defer { formatLock.unlock() }
    if let c = formatCache[x.arrowFormat] { return UnsafePointer(c) }
    let c = strdup(x.arrowFormat)!
    formatCache[x.arrowFormat] = c
    return UnsafePointer(c)
}

// MARK: - Generic dispatch helpers

private func withPrimitive<R>(_ a: AnyMetalArray, _ body: (any PrimitiveOps) throws -> R) throws -> R {
    switch a {
    case .int8(let x): return try body(x)
    case .uint8(let x): return try body(x)
    case .int16(let x): return try body(x)
    case .uint16(let x): return try body(x)
    case .int32(let x): return try body(x)
    case .uint32(let x): return try body(x)
    case .int64(let x): return try body(x)
    case .uint64(let x): return try body(x)
    case .float32(let x): return try body(x)
    case .float64(let x): return try body(x)
    case .boolean: throw ArrowMetalError.unsupportedType("operation needs a primitive array, got boolean")
    case .string: throw ArrowMetalError.unsupportedType("operation needs a primitive array, got utf8")
    case .temporal, .binary, .dictionary, .list, .structure, .map, .union:
        throw ArrowMetalError.unsupportedType("operation needs a primitive array, got \(a.arrowFormat)")
    }
}

/// Type-erased operations on `MetalArray<T>` so the C layer does not repeat the 10-way switch per function.
protocol PrimitiveOps {
    func erased() -> AnyMetalArray
    func reduce(_ op: Int32) throws -> (Int64, Double, Int32, Bool)
    func compareScalar(_ op: CompareOp, _ scalar: UnsafeRawPointer) throws -> MetalBooleanArray
    func compareArray(_ op: CompareOp, _ other: AnyMetalArray) throws -> MetalBooleanArray
    func arithScalar(_ op: ArithmeticOp, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray
    func arithArray(_ op: ArithmeticOp, _ other: AnyMetalArray) throws -> AnyMetalArray
    func filterWhere(_ op: CompareOp, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray
    func castTo(_ format: String) throws -> AnyMetalArray
    func argsort(_ descending: Bool) throws -> MetalArray<Int32>
    func topK(_ k: Int, _ largest: Bool) throws -> MetalArray<Int32>
}

extension MetalArray: PrimitiveOps {
    func erased() -> AnyMetalArray { wrap(self) }
    func reduce(_ op: Int32) throws -> (Int64, Double, Int32, Bool) {
        switch op {
        case 0:
            guard let s = try sum() else { return (0, 0, 0, true) }
            switch s { case .int(let v): return (v, 0, 0, false); case .uint(let v): return (Int64(bitPattern: v), 0, 1, false); case .float(let v): return (0, v, 2, false) }
        case 1, 2:
            guard let v = op == 1 ? try min() : try max() else { return (0, 0, 0, true) }
            if T.isFloatingPoint { return (0, v.asDouble, 2, false) }
            if T.minValue < 0 as T { return (v.asInt64, 0, 0, false) }
            return (Int64(bitPattern: v.asUInt64), 0, 1, false)
        case 3:
            guard let m = try mean() else { return (0, 0, 2, true) }
            return (0, m, 2, false)
        default: throw ArrowMetalError.invalidArrowArray("unknown reduce op \(op)")
        }
    }
    private func scalarValue(_ p: UnsafeRawPointer) -> T { p.loadUnaligned(as: T.self) }
    private func same(_ other: AnyMetalArray) throws -> MetalArray<T> {
        guard let o = unwrap(other, T.self) else { throw ArrowMetalError.unsupportedType("array types differ: \(T.arrowFormat) vs \(other.arrowFormat)") }
        return o
    }
    func compareScalar(_ op: CompareOp, _ scalar: UnsafeRawPointer) throws -> MetalBooleanArray { try compare(op, scalarValue(scalar)) }
    func compareArray(_ op: CompareOp, _ other: AnyMetalArray) throws -> MetalBooleanArray { try compare(op, try same(other)) }
    func arithScalar(_ op: ArithmeticOp, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray { wrap(try arithmetic(op, scalarValue(scalar))) }
    func arithArray(_ op: ArithmeticOp, _ other: AnyMetalArray) throws -> AnyMetalArray { wrap(try arithmetic(op, try same(other))) }
    func filterWhere(_ op: CompareOp, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray { wrap(try filter(where: op, scalarValue(scalar))) }
    func argsort(_ descending: Bool) throws -> MetalArray<Int32> { try argsort(descending: descending) }
    func topK(_ k: Int, _ largest: Bool) throws -> MetalArray<Int32> { try topK(k, largest: largest) }
    func castTo(_ format: String) throws -> AnyMetalArray {
        switch format {
        case "c": return .int8(try cast(to: Int8.self))
        case "C": return .uint8(try cast(to: UInt8.self))
        case "s": return .int16(try cast(to: Int16.self))
        case "S": return .uint16(try cast(to: UInt16.self))
        case "i": return .int32(try cast(to: Int32.self))
        case "I": return .uint32(try cast(to: UInt32.self))
        case "l": return .int64(try cast(to: Int64.self))
        case "L": return .uint64(try cast(to: UInt64.self))
        case "f": return .float32(try cast(to: Float.self))
        case "g": return .float64(try cast(to: Double.self))
        default: throw ArrowMetalError.unsupportedType("cast target \(format)")
        }
    }
}

func wrap<T: ArrowPrimitive>(_ a: MetalArray<T>) -> AnyMetalArray {
    switch a {
    case let x as MetalArray<Int8>: return .int8(x)
    case let x as MetalArray<UInt8>: return .uint8(x)
    case let x as MetalArray<Int16>: return .int16(x)
    case let x as MetalArray<UInt16>: return .uint16(x)
    case let x as MetalArray<Int32>: return .int32(x)
    case let x as MetalArray<UInt32>: return .uint32(x)
    case let x as MetalArray<Int64>: return .int64(x)
    case let x as MetalArray<UInt64>: return .uint64(x)
    case let x as MetalArray<Float>: return .float32(x)
    case let x as MetalArray<Double>: return .float64(x)
    default: fatalError("unreachable")
    }
}
func unwrap<T: ArrowPrimitive>(_ a: AnyMetalArray, _: T.Type) -> MetalArray<T>? {
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
    case .boolean, .string, .temporal, .binary, .dictionary, .list, .structure, .map, .union: return nil
    }
}
private func cmpOp(_ op: Int32) throws -> CompareOp {
    guard op >= 0, op < CompareOp.allCases.count else { throw ArrowMetalError.invalidArrowArray("bad compare op \(op)") }
    return CompareOp.allCases[Int(op)]
}
private func arithOp(_ op: Int32) throws -> ArithmeticOp {
    guard op >= 0, op < ArithmeticOp.allCases.count else { throw ArrowMetalError.invalidArrowArray("bad arithmetic op \(op)") }
    return ArithmeticOp.allCases[Int(op)]
}
private func boolean(_ a: AnyMetalArray) throws -> MetalBooleanArray {
    guard case .boolean(let b) = a else { throw ArrowMetalError.unsupportedType("expected a boolean array") }
    return b
}

// MARK: - Exported operations

@_cdecl("am_reduce")
public func am_reduce(_ a: OpaquePointer?, _ op: Int32, _ outI: UnsafeMutablePointer<Int64>?, _ outF: UnsafeMutablePointer<Double>?,
                      _ outKind: UnsafeMutablePointer<Int32>?, _ isNull: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    do {
        let (i, f, kind, null) = try withPrimitive(x) { try $0.reduce(op) }
        outI?.pointee = i; outF?.pointee = f; outKind?.pointee = kind; isNull?.pointee = null ? 1 : 0
        return 0
    } catch { setError(error); return 1 }
}
@_cdecl("am_compare_scalar")
public func am_compare_scalar(_ a: OpaquePointer?, _ op: Int32, _ scalar: UnsafeRawPointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let scalar else { return 2 }
    return run(out) { .boolean(try withPrimitive(x) { try $0.compareScalar(try cmpOp(op), scalar) }) }
}
@_cdecl("am_compare_array")
public func am_compare_array(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let y = handle(b) else { return 2 }
    return run(out) { .boolean(try withPrimitive(x) { try $0.compareArray(try cmpOp(op), y) }) }
}
@_cdecl("am_arith_scalar")
public func am_arith_scalar(_ a: OpaquePointer?, _ op: Int32, _ scalar: UnsafeRawPointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let scalar else { return 2 }
    return run(out) { try withPrimitive(x) { try $0.arithScalar(try arithOp(op), scalar) } }
}
@_cdecl("am_arith_array")
public func am_arith_array(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let y = handle(b) else { return 2 }
    return run(out) { try withPrimitive(x) { try $0.arithArray(try arithOp(op), y) } }
}
@_cdecl("am_cast")
public func am_cast(_ a: OpaquePointer?, _ format: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let format else { return 2 }
    let f = String(cString: format)
    return run(out) { try withPrimitive(x) { try $0.castTo(f) } }
}
@_cdecl("am_bool_and")
public func am_bool_and(_ a: OpaquePointer?, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let y = handle(b) else { return 2 }
    return run(out) { .boolean(try boolean(x).and(try boolean(y))) }
}
@_cdecl("am_bool_or")
public func am_bool_or(_ a: OpaquePointer?, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let y = handle(b) else { return 2 }
    return run(out) { .boolean(try boolean(x).or(try boolean(y))) }
}
@_cdecl("am_bool_not")
public func am_bool_not(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { .boolean(try boolean(x).not()) }
}
@_cdecl("am_filter")
public func am_filter(_ a: OpaquePointer?, _ mask: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let m = handle(mask) else { return 2 }
    return run(out) { try x.filter(try boolean(m)) }
}
@_cdecl("am_filter_where")
public func am_filter_where(_ a: OpaquePointer?, _ op: Int32, _ scalar: UnsafeRawPointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let scalar else { return 2 }
    return run(out) { try withPrimitive(x) { try $0.filterWhere(try cmpOp(op), scalar) } }
}
@_cdecl("am_take")
public func am_take(_ a: OpaquePointer?, _ indices: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let i = handle(indices) else { return 2 }
    return run(out) {
        switch i {
        case .int32(let idx): return try x.take(idx)
        case .int64(let idx): return try x.take(idx)
        case .uint32(let idx): return try x.take(idx)
        default: throw ArrowMetalError.unsupportedType("take indices must be int32, int64 or uint32")
        }
    }
}
@_cdecl("am_slice")
public func am_slice(_ a: OpaquePointer?, _ offset: Int64, _ length: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    guard offset >= 0, length >= 0, Int(offset + length) <= x.length else { setError(ArrowMetalError.invalidArrowArray("slice out of range")); return 1 }
    return run(out) { try x.slice(offset: Int(offset), length: Int(length)) }
}
/// Indices that sort the array (stable, nulls last). Output is int32.
@_cdecl("am_argsort")
public func am_argsort(_ a: OpaquePointer?, _ descending: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { .int32(try withPrimitive(x) { try $0.argsort(descending != 0) }) }
}
/// Sorted copy of the array (stable, nulls last). Same element type as the input.
@_cdecl("am_sort")
public func am_sort(_ a: OpaquePointer?, _ descending: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try x.take(try withPrimitive(x) { try $0.argsort(descending != 0) }) }
}
/// Indices of the k largest (or smallest) values, in sorted order. Output is int32.
@_cdecl("am_top_k")
public func am_top_k(_ a: OpaquePointer?, _ k: Int64, _ largest: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    guard k >= 0 else { setError(ArrowMetalError.invalidArrowArray("top_k needs k >= 0")); return 1 }
    return run(out) { .int32(try withPrimitive(x) { try $0.topK(Int(k), largest != 0) }) }
}
@_cdecl("am_group_by")
public func am_group_by(_ keys: OpaquePointer?, _ keyCount: Int64, _ agg: Int32, _ values: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let k = handle(keys) else { return 2 }
    let v = handle(values)
    return run(out) {
        func go<K: ArrowIndex>(_ ka: MetalArray<K>) throws -> AnyMetalArray {
            let gb = try ka.groupBy(keyCount: Int(keyCount))
            if agg == 1 { return .int64(try gb.count()) }
            guard let v else { throw ArrowMetalError.invalidArrowArray("values required for this aggregation") }
            switch (agg, v) {
            case (0, .float32(let f)): return .float64(try gb.sumFloat(f))
            case (0, .int8(let x)): return .int64(try gb.sum(x))
            case (0, .uint8(let x)): return .int64(try gb.sum(x))
            case (0, .int16(let x)): return .int64(try gb.sum(x))
            case (0, .uint16(let x)): return .int64(try gb.sum(x))
            case (0, .int32(let x)): return .int64(try gb.sum(x))
            case (0, .uint32(let x)): return .int64(try gb.sum(x))
            case (0, .int64(let x)): return .int64(try gb.sum(x))
            case (0, .uint64(let x)): return .int64(try gb.sum(x))
            case (2, _): return try withPrimitive(v) { p in try minMaxErased(gb, p.erased(), isMin: true) }
            case (3, _): return try withPrimitive(v) { p in try minMaxErased(gb, p.erased(), isMin: false) }
            case (4, .int8(let x)): return .float64(try gb.mean(x))
            case (4, .uint8(let x)): return .float64(try gb.mean(x))
            case (4, .int16(let x)): return .float64(try gb.mean(x))
            case (4, .uint16(let x)): return .float64(try gb.mean(x))
            case (4, .int32(let x)): return .float64(try gb.mean(x))
            case (4, .uint32(let x)): return .float64(try gb.mean(x))
            case (4, .int64(let x)): return .float64(try gb.mean(x))
            case (4, .uint64(let x)): return .float64(try gb.mean(x))
            case (5, _): return .int64(try withPrimitive(v) { p in try countErased(gb, p.erased()) })
            default: throw ArrowMetalError.unsupportedType("group-by agg \(agg) on \(v.arrowFormat)")
            }
        }
        switch k {
        case .int32(let ka): return try go(ka)
        case .int64(let ka): return try go(ka)
        case .uint32(let ka): return try go(ka)
        default: throw ArrowMetalError.unsupportedType("group-by keys must be int32, int64 or uint32")
        }
    }
}

private func minMaxErased<K: ArrowIndex>(_ gb: GroupBy<K>, _ v: AnyMetalArray, isMin: Bool) throws -> AnyMetalArray {
    switch v {
    case .int8(let x): return .int8(isMin ? try gb.min(x) : try gb.max(x))
    case .uint8(let x): return .uint8(isMin ? try gb.min(x) : try gb.max(x))
    case .int16(let x): return .int16(isMin ? try gb.min(x) : try gb.max(x))
    case .uint16(let x): return .uint16(isMin ? try gb.min(x) : try gb.max(x))
    case .int32(let x): return .int32(isMin ? try gb.min(x) : try gb.max(x))
    case .uint32(let x): return .uint32(isMin ? try gb.min(x) : try gb.max(x))
    case .float32(let x): return .float32(isMin ? try gb.min(x) : try gb.max(x))
    default: throw ArrowMetalError.unsupportedType("group-by min/max needs a 32-bit or narrower type")
    }
}
private func countErased<K: ArrowIndex>(_ gb: GroupBy<K>, _ v: AnyMetalArray) throws -> MetalArray<Int64> {
    switch v {
    case .int8(let x): return try gb.count(x)
    case .uint8(let x): return try gb.count(x)
    case .int16(let x): return try gb.count(x)
    case .uint16(let x): return try gb.count(x)
    case .int32(let x): return try gb.count(x)
    case .uint32(let x): return try gb.count(x)
    case .int64(let x): return try gb.count(x)
    case .uint64(let x): return try gb.count(x)
    case .float32(let x): return try gb.count(x)
    case .float64(let x): return try gb.count(x)
    case .boolean, .string, .temporal, .binary, .dictionary, .list, .structure, .map, .union:
        throw ArrowMetalError.unsupportedType("count over non-numeric values")
    }
}

// MARK: - Strings

private func string(_ a: AnyMetalArray) throws -> MetalStringArray {
    guard case .string(let s) = a else { throw ArrowMetalError.unsupportedType("expected a utf8 array") }
    return s
}
/// kind: 0 byte length, 1 char length, 2 hash32
@_cdecl("am_str_unary")
public func am_str_unary(_ a: OpaquePointer?, _ kind: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) {
        let s = try string(x)
        switch kind {
        case 0: return .int32(try s.byteLength())
        case 1: return .int32(try s.charLength())
        case 2: return .uint32(try s.hash32())
        default: throw ArrowMetalError.invalidArrowArray("bad string op \(kind)")
        }
    }
}
/// pred: 0 equals, 1 starts_with, 2 ends_with, 3 contains. `pattern` is UTF-8 bytes of `len`.
@_cdecl("am_str_match")
public func am_str_match(_ a: OpaquePointer?, _ pred: Int32, _ pattern: UnsafePointer<UInt8>?, _ len: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let pattern else { return 2 }
    let pat = String(decoding: UnsafeBufferPointer(start: pattern, count: Int(len)), as: UTF8.self)
    return run(out) {
        guard let p = MetalStringArray.Predicate(rawValue: Int(pred)) else { throw ArrowMetalError.invalidArrowArray("bad predicate") }
        return .boolean(try string(x).matches(p, pat))
    }
}
@_cdecl("am_str_equals_array")
public func am_str_equals_array(_ a: OpaquePointer?, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a), let y = handle(b) else { return 2 }
    return run(out) { .boolean(try string(x).equals(try string(y))) }
}
/// Dictionary-encodes: `codes` receives int32 codes, `unique` the unique strings.
@_cdecl("am_str_dictionary_encode")
public func am_str_dictionary_encode(_ a: OpaquePointer?, _ codes: UnsafeMutablePointer<OpaquePointer?>?, _ unique: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    do {
        let (c, u) = try string(x).dictionaryEncode()
        _ = emit(.int32(c), codes); _ = emit(.string(u), unique)
        return 0
    } catch { setError(error); return 1 }
}

// MARK: - Batching

/// Opens a batch on the calling thread: every call until am_batch_end appends to one command buffer.
@_cdecl("am_batch_begin") public func am_batch_begin() -> Int32 {
    do { try MetalContext.shared.beginBatch(); return 0 } catch { setError(error); return 1 }
}
/// Runs the batch and closes it. Deferred errors (e.g. take out of range) are reported here.
@_cdecl("am_batch_end") public func am_batch_end() -> Int32 {
    do { try MetalContext.shared.endBatch(); return 0 } catch { setError(error); return 1 }
}
