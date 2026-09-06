import Foundation
import CArrowABI
import ArrowMetal

// C ABI for Arrow decimal columns (`decimal128` = "d:p,s", `decimal256` = "d:p,s,256"). One entry point with
// an op table; the table itself is documented in include/arrowmetal.h. `am_format` already reports the
// format string through AnyMetalArray.arrowFormat, so a decimal handle round-trips through am_export.

private let errorKey = "ArrowMetalC.lastError"
private func fail(_ e: Error) -> Int32 { Thread.current.threadDictionary[errorKey] = "\(e)"; return 1 }

@inline(__always) private func array(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func produce(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>) {
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
}

/// Reads a 16-byte little-endian two's-complement scalar.
private func decimalScalar(_ p: UnsafeRawPointer) -> ArrowDecimal128 {
    ArrowDecimal128(lo: p.loadUnaligned(fromByteOffset: 0, as: UInt64.self),
                    hi: p.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
}

private func decimalArray(_ a: AnyMetalArray, _ op: Int32) throws -> MetalDecimalArray {
    guard case .decimal(let d) = a else {
        throw ArrowMetalError.unsupportedType("am_decimal_op \(op) needs a decimal array, got \(a.arrowFormat)")
    }
    return d
}

/// One entry point for every decimal operation. See the op table in include/arrowmetal.h.
///
/// Ops 0-5 compare (against `b` or the 16-byte little-endian `scalar`), 6-8 add / subtract / multiply,
/// 9-11 negate / abs / sign, 12-15 rescale with a rounding mode to the target scale `p1`, 16-17 the casts,
/// 18-20 the reductions (returned as a length-1 array so the 128-bit result survives the C ABI).
@_cdecl("am_decimal_op")
public func am_decimal_op(_ a: OpaquePointer?, _ op: Int32, _ b: OpaquePointer?, _ scalar: UnsafeRawPointer?,
                          _ p1: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do {
        let other = array(b)
        switch op {
        case 0...5:
            let d = try decimalArray(x, op)
            let cmp = CompareOp.allCases[Int(op)]
            if let o = other {
                produce(.boolean(try d.compare(cmp, try decimalArray(o, op))), out)
            } else if let s = scalar {
                produce(.boolean(try d.compare(cmp, decimalScalar(s))), out)
            } else {
                throw ArrowMetalError.invalidArrowArray("am_decimal_op \(op) needs either b or scalar")
            }
        case 6, 7:
            let d = try decimalArray(x, op)
            if let o = other {
                let y = try decimalArray(o, op)
                produce(.decimal(op == 6 ? try d.adding(y) : try d.subtracting(y)), out)
            } else if let s = scalar {
                let v = decimalScalar(s)
                produce(.decimal(op == 6 ? try d.adding(v) : try d.subtracting(v)), out)
            } else {
                throw ArrowMetalError.invalidArrowArray("am_decimal_op \(op) needs either b or scalar")
            }
        case 8:
            let d = try decimalArray(x, op)
            if let o = other {
                produce(.decimal(try d.multiplied(by: try decimalArray(o, op))), out)
            } else if let s = scalar {
                produce(.decimal(try d.multiplied(by: s.loadUnaligned(as: Int64.self))), out)
            } else {
                throw ArrowMetalError.invalidArrowArray("am_decimal_op 8 needs either b or an int64 scalar")
            }
        case 9: produce(.decimal(try decimalArray(x, op).negated()), out)
        case 10: produce(.decimal(try decimalArray(x, op).absoluteValue()), out)
        case 11: produce(.int32(try decimalArray(x, op).sign()), out)
        case 12...15:
            let mode = DecimalRoundMode(rawValue: Int(op) - 12)!
            produce(.decimal(try decimalArray(x, op).rescaled(to: Int(p1), mode: mode)), out)
        case 16: produce(.float64(try decimalArray(x, op).toFloat64()), out)
        case 17:
            let t = try ArrowDecimalType(precision: 38, scale: Int(p1), bitWidth: 128)
            switch x {
            case .float64(let f): produce(.decimal(try MetalDecimalArray.fromFloat64(f, type: t)), out)
            case .int64(let i): produce(.decimal(try MetalDecimalArray.fromInt64(i, type: t)), out)
            default: throw ArrowMetalError.unsupportedType("am_decimal_op 17 casts a float64 or int64 column, got \(x.arrowFormat)")
            }
        case 18, 19, 20:
            let d = try decimalArray(x, op)
            let v = op == 18 ? try d.sum() : (op == 19 ? try d.min() : try d.max())
            produce(.decimal(try MetalDecimalArray(type: d.type, [v], context: d.context)), out)
        default:
            throw ArrowMetalError.invalidArrowArray("unknown am_decimal_op op \(op)")
        }
        return 0
    } catch { return fail(error) }
}
