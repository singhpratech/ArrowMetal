import Foundation
import CArrowABI
import ArrowMetal

// C ABI for Arrow's checked (overflow-raising) arithmetic and for the remaining element-wise math.
// The op numbering here is the contract; it is repeated in include/arrowmetal.h and must not be
// reordered.
//
//   am_unary_checked      0 negate_checked  1 abs_checked  2 sqrt_checked
//                         3 ln_checked      4 log10_checked 5 log2_checked  6 log1p_checked
//   am_binary_checked     0 add_checked     1 subtract_checked  2 multiply_checked  3 divide_checked
//                         4 power_checked   5 shift_left_checked 6 shift_right_checked 7 logb_checked
//   am_cumulative_checked 0 cumulative_sum_checked  1 cumulative_prod_checked
//                         2 pairwise_diff_checked (p1 = period)
//   am_math_extra         0 expm1  1 log1p  2 logb  3 hypot
//                         4 round (p1 = mode | ndigits << 8)  5 round_to_multiple (p1 = mode)
//                         6 round_binary (p1 = mode, b = int32 ndigits column)
//
// A failing element comes back as return code 1 with am_last_error() reading, for example,
// "add_checked: overflow at index 4097" or "divide_checked: divide by zero at index 0".

private let checkedErrorKey = "ArrowMetalC.lastError"
private func setError(_ e: Error) { Thread.current.threadDictionary[checkedErrorKey] = "\(e)" }

@inline(__always) private func handle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func run(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch {
        setError(error)
        return 1
    }
}

/// Type-erased entry points, so the C layer writes the exhaustive switch over `AnyMetalArray` once.
protocol CheckedOps {
    func amUnaryChecked(_ op: Int32) throws -> AnyMetalArray
    func amBinaryCheckedScalar(_ op: Int32, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray
    func amBinaryCheckedArray(_ op: Int32, _ other: AnyMetalArray) throws -> AnyMetalArray
    func amCumulativeChecked(_ op: Int32, _ p1: Int64) throws -> AnyMetalArray
    func amMathExtra(_ op: Int32, _ other: AnyMetalArray?, _ scalar: UnsafeRawPointer?, _ p1: Int64) throws -> AnyMetalArray
}

extension MetalArray: CheckedOps {
    func amUnaryChecked(_ op: Int32) throws -> AnyMetalArray {
        switch op {
        case 0: return wrap(try negateChecked())
        case 1: return wrap(try absChecked())
        case 2: return wrap(try sqrtChecked())
        case 3: return wrap(try lnChecked())
        case 4: return wrap(try log10Checked())
        case 5: return wrap(try log2Checked())
        case 6: return wrap(try log1pChecked())
        default: throw ArrowMetalError.invalidArrowArray("unknown checked unary op \(op)")
        }
    }

    func amBinaryCheckedScalar(_ op: Int32, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray {
        let s = scalar.loadUnaligned(as: T.self)
        switch op {
        case 0: return wrap(try addChecked(s))
        case 1: return wrap(try subtractChecked(s))
        case 2: return wrap(try multiplyChecked(s))
        case 3: return wrap(try divideChecked(s))
        case 4: return wrap(try powerChecked(s))
        case 5: return wrap(try shiftLeftChecked(s))
        case 6: return wrap(try shiftRightChecked(s))
        case 7: return wrap(try logbChecked(s))
        default: throw ArrowMetalError.invalidArrowArray("unknown checked binary op \(op)")
        }
    }

    func amBinaryCheckedArray(_ op: Int32, _ other: AnyMetalArray) throws -> AnyMetalArray {
        guard let o = unwrap(other, T.self) else {
            throw ArrowMetalError.unsupportedType("array types differ: \(T.arrowFormat) vs \(other.arrowFormat)")
        }
        switch op {
        case 0: return wrap(try addChecked(o))
        case 1: return wrap(try subtractChecked(o))
        case 2: return wrap(try multiplyChecked(o))
        case 3: return wrap(try divideChecked(o))
        case 4: return wrap(try powerChecked(o))
        case 5: return wrap(try shiftLeftChecked(o))
        case 6: return wrap(try shiftRightChecked(o))
        case 7: return wrap(try logbChecked(o))
        default: throw ArrowMetalError.invalidArrowArray("unknown checked binary op \(op)")
        }
    }

    func amCumulativeChecked(_ op: Int32, _ p1: Int64) throws -> AnyMetalArray {
        switch op {
        case 0: return wrap(try cumulativeSumChecked())
        case 1: return wrap(try cumulativeProdChecked())
        case 2: return wrap(try pairwiseDiffChecked(period: Int(clamping: p1)))
        default: throw ArrowMetalError.invalidArrowArray("unknown checked cumulative op \(op)")
        }
    }

    func amMathExtra(_ op: Int32, _ other: AnyMetalArray?, _ scalar: UnsafeRawPointer?, _ p1: Int64) throws -> AnyMetalArray {
        /// The round mode lives in `p1`'s low byte for every rounding op; `ndigits` is the rest.
        func mode() throws -> RoundMode {
            guard let m = RoundMode(rawValue: Int(p1 & 0xFF)) else {
                throw ArrowMetalError.invalidArrowArray("unknown round mode \(p1 & 0xFF)")
            }
            return m
        }
        func typedOther() throws -> MetalArray<T> {
            guard let o = other, let x = unwrap(o, T.self) else {
                throw ArrowMetalError.unsupportedType("this op needs a second column of the same type")
            }
            return x
        }
        switch op {
        case 0: return wrap(try expm1())
        case 1: return wrap(try log1p())
        case 2:
            if other != nil { return wrap(try logb(try typedOther())) }
            guard let scalar else { throw ArrowMetalError.invalidArrowArray("logb needs a base column or scalar") }
            return wrap(try logb(scalar.loadUnaligned(as: T.self)))
        case 3:
            if other != nil { return wrap(try hypot(try typedOther())) }
            guard let scalar else { throw ArrowMetalError.invalidArrowArray("hypot needs a second column or scalar") }
            return wrap(try hypot(scalar.loadUnaligned(as: T.self)))
        case 4: return wrap(try round(ndigits: Int(p1 >> 8), mode: try mode()))
        case 5:
            guard let scalar else { throw ArrowMetalError.invalidArrowArray("round_to_multiple needs a multiple") }
            return wrap(try roundToMultiple(scalar.loadUnaligned(as: T.self), mode: try mode()))
        case 6:
            guard let o = other, let nd = unwrap(o, Int32.self) else {
                throw ArrowMetalError.unsupportedType("round_binary needs an int32 ndigits column")
            }
            return wrap(try roundBinary(nd, mode: try mode()))
        default: throw ArrowMetalError.invalidArrowArray("unknown math_extra op \(op)")
        }
    }
}

/// Every case the enum has, so a new Arrow type cannot silently fall through this entry point.
private func withChecked<R>(_ a: AnyMetalArray, _ body: (any CheckedOps) throws -> R) throws -> R {
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
    case .boolean: throw ArrowMetalError.unsupportedType("checked arithmetic needs a primitive array, got boolean")
    case .string, .binary: throw ArrowMetalError.unsupportedType("checked arithmetic needs a primitive array, got \(a.arrowFormat)")
    case .temporal(let t):
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .dictionary: throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    case .decimal(let d): throw ArrowMetalError.unsupportedType("\(d.type) columns use am_decimal_op, not this entry point")
    case .list, .structure, .map, .union:
        throw ArrowMetalError.unsupportedType("checked arithmetic needs a primitive array, got \(a.arrowFormat)")
    case .runEndEncoded: throw ArrowMetalError.unsupportedType("decode the run-end encoded array first")
    case .float16(let x): return try body(try x.toFloat32())
    case .extended(let e): return try withChecked(e.storage, body)
    case .null, .smallDecimal, .interval, .fixedBinary:
        throw ArrowMetalError.unsupportedType("checked arithmetic needs a primitive array, got \(a.arrowFormat)")
    }
}

// MARK: - Exported operations

/// One checked unary op. See the op table at the top of this file and in arrowmetal.h.
@_cdecl("am_unary_checked")
public func am_unary_checked(_ a: OpaquePointer?, _ op: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try withChecked(x) { try $0.amUnaryChecked(op) } }
}

/// One checked binary op. Pass either `b` (array form) or `scalar` (scalar form), not both.
@_cdecl("am_binary_checked")
public func am_binary_checked(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ scalar: UnsafeRawPointer?,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    if let b {
        guard let y = handle(b) else { return 2 }
        return run(out) { try withChecked(x) { try $0.amBinaryCheckedArray(op, y) } }
    }
    guard let scalar else { return 2 }
    return run(out) { try withChecked(x) { try $0.amBinaryCheckedScalar(op, scalar) } }
}

/// Checked cumulative sum / product, and checked pairwise difference (`p1` is its period).
@_cdecl("am_cumulative_checked")
public func am_cumulative_checked(_ a: OpaquePointer?, _ op: Int32, _ p1: Int64,
                                  _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try withChecked(x) { try $0.amCumulativeChecked(op, p1) } }
}

/// `expm1`, `log1p`, `logb`, `hypot` and the rounding family. See the op table for what `b`, `scalar`
/// and `p1` mean for each op.
@_cdecl("am_math_extra")
public func am_math_extra(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ scalar: UnsafeRawPointer?,
                          _ p1: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    let other = b.flatMap { handle($0) }
    if b != nil && other == nil { return 2 }
    return run(out) { try withChecked(x) { try $0.amMathExtra(op, other, scalar, p1) } }
}
