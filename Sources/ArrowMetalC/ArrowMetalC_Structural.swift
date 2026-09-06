import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the structural, conditional and set-lookup functions: am_is_null, am_is_valid,
// am_fill_null, am_drop_null, am_if_else, am_coalesce, am_is_in, am_index_in, am_and_kleene,
// am_or_kleene. Handles and error reporting follow ArrowMetalC.swift exactly: the same retained
// `Box` and the same thread-local error slot, so am_last_error() reports failures from here too.

private let structuralErrorKey = "ArrowMetalC.lastError"
private func stSetError(_ e: Error) { Thread.current.threadDictionary[structuralErrorKey] = "\(e)" }

@inline(__always) private func stHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func stEmit(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
    return 0
}
private func stRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do { return stEmit(try body(), out) } catch { stSetError(error); return 1 }
}
private func stBoolean(_ a: AnyMetalArray) throws -> MetalBooleanArray {
    guard case .boolean(let b) = a else { throw ArrowMetalError.unsupportedType("expected a boolean array") }
    return b
}

// MARK: - Type-erased structural operations

/// The structural functions on `MetalArray<T>`, erased so the C layer does not repeat a ten-way
/// switch per entry point.
protocol StructuralCOps {
    func stIsNull() throws -> MetalBooleanArray
    func stIsValid() throws -> MetalBooleanArray
    func stFillNull(_ scalar: UnsafeRawPointer) throws -> AnyMetalArray
    func stDropNull() throws -> AnyMetalArray
    func stIfElse(_ cond: MetalBooleanArray, _ right: AnyMetalArray) throws -> AnyMetalArray
    func stCoalesce(_ rest: [AnyMetalArray]) throws -> AnyMetalArray
    func stIsIn(_ set: AnyMetalArray) throws -> MetalBooleanArray
    func stIndexIn(_ set: AnyMetalArray) throws -> MetalArray<Int32>
}

extension MetalArray: StructuralCOps {
    private func sameType(_ other: AnyMetalArray) throws -> MetalArray<T> {
        guard let o = unwrap(other, T.self) else {
            throw ArrowMetalError.unsupportedType("array types differ: \(T.arrowFormat) vs \(other.arrowFormat)")
        }
        return o
    }
    func stIsNull() throws -> MetalBooleanArray { try isNull() }
    func stIsValid() throws -> MetalBooleanArray { try isValid() }
    func stFillNull(_ scalar: UnsafeRawPointer) throws -> AnyMetalArray {
        wrap(try fillingNull(scalar.loadUnaligned(as: T.self)))
    }
    func stDropNull() throws -> AnyMetalArray { wrap(try dropNull()) }
    func stIfElse(_ cond: MetalBooleanArray, _ right: AnyMetalArray) throws -> AnyMetalArray {
        wrap(try MetalArray<T>.ifElse(cond, self, try sameType(right)))
    }
    func stCoalesce(_ rest: [AnyMetalArray]) throws -> AnyMetalArray {
        wrap(try MetalArray<T>.coalesce([self] + (try rest.map { try sameType($0) })))
    }
    func stIsIn(_ set: AnyMetalArray) throws -> MetalBooleanArray { try isIn(try sameType(set)) }
    func stIndexIn(_ set: AnyMetalArray) throws -> MetalArray<Int32> { try indexIn(try sameType(set)) }
}

private func withStructural<R>(_ a: AnyMetalArray, _ body: (any StructuralCOps) throws -> R) throws -> R {
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
    case .string, .binary: throw ArrowMetalError.unsupportedType("operation needs a primitive array, got \(a.arrowFormat)")
    case .temporal(let t):
        // Temporal columns are integers underneath; run the structural op on the storage.
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .dictionary: throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    }
}

// MARK: - Exported operations

/// Arrow `is_null`: a boolean array, true where the input is null. Never has nulls itself.
/// Works on primitive and boolean arrays.
@_cdecl("am_is_null")
public func am_is_null(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a) else { return 2 }
    return stRun(out) {
        if case .boolean(let b) = x { return .boolean(try b.isNull()) }
        return .boolean(try withStructural(x) { try $0.stIsNull() })
    }
}

/// Arrow `is_valid`: the complement of `am_is_null`.
@_cdecl("am_is_valid")
public func am_is_valid(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a) else { return 2 }
    return stRun(out) {
        if case .boolean(let b) = x { return .boolean(try b.isValid()) }
        return .boolean(try withStructural(x) { try $0.stIsValid() })
    }
}

/// Arrow `fill_null`: nulls become `scalar`. `scalar` points at a value of the array's element type;
/// for a boolean array it points at one byte, non-zero meaning true.
@_cdecl("am_fill_null")
public func am_fill_null(_ a: OpaquePointer?, _ scalar: UnsafeRawPointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a), let scalar else { return 2 }
    return stRun(out) {
        if case .boolean(let b) = x { return .boolean(try b.fillingNull(scalar.loadUnaligned(as: UInt8.self) != 0)) }
        return try withStructural(x) { try $0.stFillNull(scalar) }
    }
}

/// Arrow `drop_null`: the non-null elements, in order.
@_cdecl("am_drop_null")
public func am_drop_null(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a) else { return 2 }
    return stRun(out) {
        if case .boolean(let b) = x { return .boolean(try b.dropNull()) }
        return try withStructural(x) { try $0.stDropNull() }
    }
}

/// Arrow `if_else`: `cond ? left : right`. `cond` must be a boolean array; `left` and `right` must
/// have the same type as each other and the same length as `cond`. A null condition yields null.
@_cdecl("am_if_else")
public func am_if_else(_ cond: OpaquePointer?, _ left: OpaquePointer?, _ right: OpaquePointer?,
                       _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let c = stHandle(cond), let l = stHandle(left), let r = stHandle(right) else { return 2 }
    return stRun(out) {
        let condition = try stBoolean(c)
        if case .boolean(let lb) = l { return .boolean(try MetalBooleanArray.ifElse(condition, lb, try stBoolean(r))) }
        return try withStructural(l) { try $0.stIfElse(condition, r) }
    }
}

/// Arrow `coalesce`: the first non-null value across `count` arrays of one type and length.
@_cdecl("am_coalesce")
public func am_coalesce(_ arrays: UnsafePointer<OpaquePointer?>?, _ count: Int64,
                        _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let arrays, count > 0 else { return 2 }
    var inputs: [AnyMetalArray] = []
    inputs.reserveCapacity(Int(count))
    for i in 0..<Int(count) {
        guard let h = stHandle(arrays[i]) else { return 2 }
        inputs.append(h)
    }
    return stRun(out) { try withStructural(inputs[0]) { try $0.stCoalesce(Array(inputs.dropFirst())) } }
}

/// Arrow `is_in`: a boolean array, true where the element appears among the non-null values of
/// `set_array`. Nulls in the set are ignored and a null element is not in the set, so the result
/// never has nulls.
@_cdecl("am_is_in")
public func am_is_in(_ a: OpaquePointer?, _ setArray: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a), let s = stHandle(setArray) else { return 2 }
    return stRun(out) { .boolean(try withStructural(x) { try $0.stIsIn(s) }) }
}

/// Arrow `index_in`: int32 index into `set_array` of the first occurrence of each element, null
/// where the element is null or absent from the set.
@_cdecl("am_index_in")
public func am_index_in(_ a: OpaquePointer?, _ setArray: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a), let s = stHandle(setArray) else { return 2 }
    return stRun(out) { .int32(try withStructural(x) { try $0.stIndexIn(s) }) }
}

/// Arrow `and_kleene`: three-valued AND over two boolean arrays (`false AND null` is `false`).
@_cdecl("am_and_kleene")
public func am_and_kleene(_ a: OpaquePointer?, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a), let y = stHandle(b) else { return 2 }
    return stRun(out) { .boolean(try stBoolean(x).andKleene(try stBoolean(y))) }
}

/// Arrow `or_kleene`: three-valued OR over two boolean arrays (`true OR null` is `true`).
@_cdecl("am_or_kleene")
public func am_or_kleene(_ a: OpaquePointer?, _ b: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = stHandle(a), let y = stHandle(b) else { return 2 }
    return stRun(out) { .boolean(try stBoolean(x).orKleene(try stBoolean(y))) }
}
