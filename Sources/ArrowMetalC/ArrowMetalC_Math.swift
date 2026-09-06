import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the bit-wise, element-wise math and cumulative kernels. The op numbering here is the
// contract; it is repeated in include/arrowmetal.h and must not be reordered.
//
//   am_unary       0 negate  1 abs  2 sign  3 sqrt  4 exp  5 ln  6 log10  7 log2
//                  8 floor   9 ceil 10 round 11 trunc 12 bit_wise_not
//   am_binary      0 bit_wise_and  1 bit_wise_or  2 bit_wise_xor  3 shift_left  4 shift_right
//                  5 modulo  6 power  7 min_element_wise  8 max_element_wise
//   am_cumulative  0 cumulative_sum  1 cumulative_min  2 cumulative_max
//
// The first eleven unary ops are `UnaryMathOp.allCases` in order, and ops 0-4 of am_binary are
// `BitwiseOp.allCases` in order; the switches below rely on that and nothing else does.

private let mathErrorKey = "ArrowMetalC.lastError"
private func setError(_ e: Error) { Thread.current.threadDictionary[mathErrorKey] = "\(e)" }

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

/// Type-erased entry points for the new kernels, so the C layer writes the ten-way switch once.
protocol MathOps {
    func amUnary(_ op: Int32) throws -> AnyMetalArray
    func amBinaryScalar(_ op: Int32, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray
    func amBinaryArray(_ op: Int32, _ other: AnyMetalArray) throws -> AnyMetalArray
    func amCumulative(_ op: Int32) throws -> AnyMetalArray
}

extension MetalArray: MathOps {
    func amUnary(_ op: Int32) throws -> AnyMetalArray {
        if op == 12 { return wrap(try bitwiseNot()) }
        guard op >= 0, Int(op) < UnaryMathOp.allCases.count else {
            throw ArrowMetalError.invalidArrowArray("unknown unary op \(op)")
        }
        return wrap(try unaryMath(UnaryMathOp.allCases[Int(op)]))
    }

    func amBinaryScalar(_ op: Int32, _ scalar: UnsafeRawPointer) throws -> AnyMetalArray {
        let s = scalar.loadUnaligned(as: T.self)
        switch op {
        case 0...4: return wrap(try bitwise(BitwiseOp.allCases[Int(op)], s))
        case 5: return wrap(try modulo(s))
        case 6: return wrap(try power(s))
        case 7, 8: throw ArrowMetalError.unsupportedType("min/max_element_wise takes two arrays, not a scalar")
        default: throw ArrowMetalError.invalidArrowArray("unknown binary op \(op)")
        }
    }

    func amBinaryArray(_ op: Int32, _ other: AnyMetalArray) throws -> AnyMetalArray {
        guard let o = unwrap(other, T.self) else {
            throw ArrowMetalError.unsupportedType("array types differ: \(T.arrowFormat) vs \(other.arrowFormat)")
        }
        switch op {
        case 0...4: return wrap(try bitwise(BitwiseOp.allCases[Int(op)], o))
        case 5: return wrap(try modulo(o))
        case 6: return wrap(try power(o))
        case 7: return wrap(try minElementWise(o))
        case 8: return wrap(try maxElementWise(o))
        default: throw ArrowMetalError.invalidArrowArray("unknown binary op \(op)")
        }
    }

    func amCumulative(_ op: Int32) throws -> AnyMetalArray {
        guard op >= 0, Int(op) < CumulativeOp.allCases.count else {
            throw ArrowMetalError.invalidArrowArray("unknown cumulative op \(op)")
        }
        return wrap(try cumulative(CumulativeOp.allCases[Int(op)]))
    }
}

private func withMath<R>(_ a: AnyMetalArray, _ body: (any MathOps) throws -> R) throws -> R {
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
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .dictionary: throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    case .runEndEncoded: throw ArrowMetalError.unsupportedType("decode the run-end encoded array first")
    }
}

// MARK: - Exported operations

/// One element-wise unary op. See the op table at the top of this file and in arrowmetal.h.
@_cdecl("am_unary")
public func am_unary(_ a: OpaquePointer?, _ op: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try withMath(x) { try $0.amUnary(op) } }
}

/// One element-wise binary op. Pass either `b` (array form) or `scalar` (scalar form), not both.
/// `min_element_wise` and `max_element_wise` have no scalar form.
@_cdecl("am_binary")
public func am_binary(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ scalar: UnsafeRawPointer?,
                      _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    if let b {
        guard let y = handle(b) else { return 2 }
        return run(out) { try withMath(x) { try $0.amBinaryArray(op, y) } }
    }
    guard let scalar else { return 2 }
    return run(out) { try withMath(x) { try $0.amBinaryScalar(op, scalar) } }
}

/// Cumulative sum / min / max. Output null where input null; the running value carries across nulls.
@_cdecl("am_cumulative")
public func am_cumulative(_ a: OpaquePointer?, _ op: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try withMath(x) { try $0.amCumulative(op) } }
}
