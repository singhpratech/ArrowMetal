import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the trigonometric, logical, float-classification, conditional and hashing kernels:
// am_trig, am_logical, am_float_class, am_fill_null_direction, am_case_when, am_choose,
// am_replace_with_mask, am_indices_nonzero and am_hash64. Handles and error reporting follow
// ArrowMetalC.swift exactly — the same retained `Box` and the same thread-local error slot, so
// am_last_error() reports failures from here too.
//
// The op tables below are the contract; they are repeated in include/arrowmetal.h and must not be
// reordered.
//
//   am_trig         0 sin  1 cos  2 tan  3 asin  4 acos  5 atan
//                   6 sinh 7 cosh 8 tanh 9 asinh 10 acosh 11 atanh
//                   12 atan2 (needs b)
//                   13 sin_checked  14 cos_checked  15 tan_checked  16 asin_checked
//                   17 acos_checked 18 acosh_checked 19 atanh_checked
//   am_logical      0 xor  1 and_not  2 and_not_kleene
//   am_float_class  0 is_nan  1 is_finite  2 is_inf
//
// Ops 0-11 are `TrigOp.allCases` in order and 13-19 are `TrigCheckedOp.allCases` in order; the
// switches below rely on that and nothing else does.

private let condErrorKey = "ArrowMetalC.lastError"
private func cnSetError(_ e: Error) { Thread.current.threadDictionary[condErrorKey] = "\(e)" }

@inline(__always) private func cnHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func cnRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch {
        cnSetError(error)
        return 1
    }
}
private func cnBoolean(_ a: AnyMetalArray) throws -> MetalBooleanArray {
    guard case .boolean(let b) = a else { throw ArrowMetalError.unsupportedType("expected a boolean array, got \(a.arrowFormat)") }
    return b
}
private func cnCollect(_ p: UnsafePointer<OpaquePointer?>?, _ count: Int64) throws -> [AnyMetalArray] {
    guard let p, count > 0 else { throw ArrowMetalError.invalidArrowArray("expected at least one array") }
    return try (0..<Int(count)).map {
        guard let h = cnHandle(p[$0]) else { throw ArrowMetalError.invalidArrowArray("null array handle at \($0)") }
        return h
    }
}

// MARK: - Type-erased operations

/// The new functions on `MetalArray<T>`, erased so the C layer does not repeat a ten-way switch per
/// entry point.
protocol ConditionalCOps {
    func cnTrig(_ op: Int32, _ b: AnyMetalArray?) throws -> AnyMetalArray
    func cnFloatClass(_ op: Int32) throws -> MetalBooleanArray
    func cnFillNull(forward: Bool) throws -> AnyMetalArray
    func cnCaseWhen(_ conds: [MetalBooleanArray], _ values: [AnyMetalArray], _ def: AnyMetalArray?) throws -> AnyMetalArray
    func cnChoose(_ indices: AnyMetalArray, _ values: [AnyMetalArray]) throws -> AnyMetalArray
    func cnReplaceWithMask(_ mask: MetalBooleanArray, _ replacements: AnyMetalArray) throws -> AnyMetalArray
    func cnIndicesNonzero() throws -> MetalArray<UInt64>
    func cnHash64() throws -> MetalArray<UInt64>
}

extension MetalArray: ConditionalCOps {
    private func sameType(_ other: AnyMetalArray) throws -> MetalArray<T> {
        guard let o = unwrap(other, T.self) else {
            throw ArrowMetalError.unsupportedType("array types differ: \(T.arrowFormat) vs \(other.arrowFormat)")
        }
        return o
    }

    func cnTrig(_ op: Int32, _ b: AnyMetalArray?) throws -> AnyMetalArray {
        let unary = TrigOp.allCases.count                  // 12
        switch Int(op) {
        case 0..<unary:
            return wrap(try trig(TrigOp.allCases[Int(op)]))
        case unary:                                        // 12: atan2
            guard let b else { throw ArrowMetalError.invalidArrowArray("atan2 needs a second array") }
            return wrap(try atan2(try sameType(b)))
        case (unary + 1)..<(unary + 1 + TrigCheckedOp.allCases.count):
            return wrap(try trigChecked(TrigCheckedOp.allCases[Int(op) - unary - 1]))
        default:
            throw ArrowMetalError.invalidArrowArray("unknown trig op \(op)")
        }
    }

    func cnFloatClass(_ op: Int32) throws -> MetalBooleanArray {
        guard op >= 0, Int(op) < FloatClassOp.allCases.count else {
            throw ArrowMetalError.invalidArrowArray("unknown float class op \(op)")
        }
        return try floatClass(FloatClassOp.allCases[Int(op)])
    }

    func cnFillNull(forward: Bool) throws -> AnyMetalArray {
        wrap(forward ? try fillNullForward() : try fillNullBackward())
    }

    func cnCaseWhen(_ conds: [MetalBooleanArray], _ values: [AnyMetalArray], _ def: AnyMetalArray?) throws -> AnyMetalArray {
        wrap(try MetalArray<T>.caseWhen(conds: conds,
                                        values: try values.map { try sameType($0) },
                                        else: try def.map { try sameType($0) }))
    }

    func cnChoose(_ indices: AnyMetalArray, _ values: [AnyMetalArray]) throws -> AnyMetalArray {
        let cols = try values.map { try sameType($0) }
        switch indices {
        case .int32(let i): return wrap(try MetalArray<T>.choose(i, cols))
        case .int64(let i): return wrap(try MetalArray<T>.choose(i, cols))
        case .uint32(let i): return wrap(try MetalArray<T>.choose(i, cols))
        default: throw ArrowMetalError.unsupportedType("choose needs int32, int64 or uint32 indices, got \(indices.arrowFormat)")
        }
    }

    func cnReplaceWithMask(_ mask: MetalBooleanArray, _ replacements: AnyMetalArray) throws -> AnyMetalArray {
        wrap(try replaceWithMask(mask, try sameType(replacements)))
    }

    func cnIndicesNonzero() throws -> MetalArray<UInt64> { try indicesNonzero() }
    func cnHash64() throws -> MetalArray<UInt64> { try hash64() }
}

/// The ten-way switch, written once. Every case the `AnyMetalArray` enum has today is listed, so a
/// new case in the enum is a compile error here rather than a silent fall-through.
private func withConditional<R>(_ a: AnyMetalArray, _ body: (any ConditionalCOps) throws -> R) throws -> R {
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
        // Temporal columns are integers underneath; run the op on the storage.
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .dictionary: throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    case .decimal(let d): throw ArrowMetalError.unsupportedType("\(d.type) columns use am_decimal_op, not this entry point")
    case .list, .structure, .map, .union:
        throw ArrowMetalError.unsupportedType("operation needs a primitive array, got \(a.arrowFormat)")
    }
}

// MARK: - Exported operations

/// One trigonometric, inverse-trigonometric or hyperbolic function. See the op table at the top of
/// this file and in arrowmetal.h. Float columns only; `b` is the second argument of `atan2` (op 12)
/// and must be NULL for every other op. The `_checked` ops (13-19) raise on a domain violation.
@_cdecl("am_trig")
public func am_trig(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?,
                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a) else { return 2 }
    let y = b.flatMap { cnHandle($0) }
    if b != nil && y == nil { return 2 }
    return cnRun(out) { try withConditional(x) { try $0.cnTrig(op, y) } }
}

/// Arrow `xor` / `and_not` / `and_not_kleene` over two boolean arrays.
@_cdecl("am_logical")
public func am_logical(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?,
                       _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a), let y = cnHandle(b) else { return 2 }
    return cnRun(out) {
        guard op >= 0, Int(op) < LogicalExtraOp.allCases.count else {
            throw ArrowMetalError.invalidArrowArray("unknown logical op \(op)")
        }
        return .boolean(try cnBoolean(x).logicalExtra(LogicalExtraOp.allCases[Int(op)], try cnBoolean(y)))
    }
}

/// Arrow `is_nan` / `is_finite` / `is_inf`. Defined on every numeric type; a null element gives a
/// null result, as in Arrow.
@_cdecl("am_float_class")
public func am_float_class(_ a: OpaquePointer?, _ op: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a) else { return 2 }
    return cnRun(out) { .boolean(try withConditional(x) { try $0.cnFloatClass(op) }) }
}

/// Arrow `fill_null_forward` (`forward` non-zero) or `fill_null_backward`. Nulls with no non-null
/// element on the chosen side stay null.
@_cdecl("am_fill_null_direction")
public func am_fill_null_direction(_ a: OpaquePointer?, _ forward: Int32,
                                   _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a) else { return 2 }
    return cnRun(out) {
        if case .boolean(let b) = x {
            return .boolean(forward != 0 ? try b.fillNullForward() : try b.fillNullBackward())
        }
        return try withConditional(x) { try $0.cnFillNull(forward: forward != 0) }
    }
}

/// Arrow `case_when`: `count` boolean conditions and `count` value columns of one type and length,
/// plus an optional default. The value of the first true condition wins; a **null condition counts
/// as false**, as in Arrow. With no default, a row that matches nothing is null.
@_cdecl("am_case_when")
public func am_case_when(_ conds: UnsafePointer<OpaquePointer?>?, _ values: UnsafePointer<OpaquePointer?>?,
                         _ count: Int64, _ elseOrNull: OpaquePointer?,
                         _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    return cnRun(out) {
        let cs = try cnCollect(conds, count).map { try cnBoolean($0) }
        let vs = try cnCollect(values, count)
        let def = elseOrNull.flatMap { cnHandle($0) }
        return try withConditional(vs[0]) { try $0.cnCaseWhen(cs, vs, def) }
    }
}

/// Arrow `choose`: `values[indices[i]][i]`. Indices are int32, int64 or uint32; a null index gives a
/// null output and an out-of-range index is an error.
@_cdecl("am_choose")
public func am_choose(_ indices: OpaquePointer?, _ values: UnsafePointer<OpaquePointer?>?, _ count: Int64,
                      _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let idx = cnHandle(indices) else { return 2 }
    return cnRun(out) {
        let vs = try cnCollect(values, count)
        return try withConditional(vs[0]) { try $0.cnChoose(idx, vs) }
    }
}

/// Arrow `replace_with_mask`: rows where `mask` is true take the next value from `replacements`, in
/// order; rows where the mask is null become null; the rest keep their own value. `replacements`
/// must hold at least as many elements as the mask has valid trues.
@_cdecl("am_replace_with_mask")
public func am_replace_with_mask(_ a: OpaquePointer?, _ mask: OpaquePointer?, _ replacements: OpaquePointer?,
                                 _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a), let m = cnHandle(mask), let r = cnHandle(replacements) else { return 2 }
    return cnRun(out) {
        let bm = try cnBoolean(m)
        if case .boolean(let xb) = x { return .boolean(try xb.replaceWithMask(bm, try cnBoolean(r))) }
        return try withConditional(x) { try $0.cnReplaceWithMask(bm, r) }
    }
}

/// Arrow `indices_nonzero`: the uint64 row numbers where the value is valid and not zero.
@_cdecl("am_indices_nonzero")
public func am_indices_nonzero(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a) else { return 2 }
    return cnRun(out) {
        if case .boolean(let b) = x { return .uint64(try b.indicesNonzero()) }
        return .uint64(try withConditional(x) { try $0.cnIndicesNonzero() })
    }
}

/// A 64-bit hash of every element (MurmurHash3's finaliser over the normalised value; see
/// Kernels/Hash64.swift). Nulls hash to 0 and stay null.
@_cdecl("am_hash64")
public func am_hash64(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = cnHandle(a) else { return 2 }
    return cnRun(out) {
        if case .boolean(let b) = x { return .uint64(try b.hash64()) }
        return .uint64(try withConditional(x) { try $0.cnHash64() })
    }
}
