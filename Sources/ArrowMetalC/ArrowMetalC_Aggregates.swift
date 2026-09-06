import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the statistical aggregates and for run-end encoding. The op table lives in
// include/arrowmetal.h and is the contract; the switch below is its implementation.

private let errorKey = "ArrowMetalC.lastError"
private func fail(_ e: Error) -> Int32 { Thread.current.threadDictionary[errorKey] = "\(e)"; return 1 }

@inline(__always) private func array(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func produce(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>) {
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
}

/// Saturating conversion for the `p1` argument of ops that take a numeric value.
private func clampToInt64(_ d: Double) -> Int64 {
    if d.isNaN { return 0 }
    if d <= -9.223372036854775e18 { return .min }
    if d >= 9.223372036854775e18 { return .max }
    return Int64(d)
}

/// One aggregate result: `kind` 0 = int64, 1 = uint64 (in the int64 slot), 2 = double.
private struct AggregateResult {
    var int64: Int64 = 0
    var double: Double = 0
    var kind: Int32 = 0
    var isNull: Bool = false
    static let null = AggregateResult(isNull: true)
    static func i(_ v: Int64) -> AggregateResult { AggregateResult(int64: v, kind: 0) }
    static func u(_ v: UInt64) -> AggregateResult { AggregateResult(int64: Int64(bitPattern: v), kind: 1) }
    static func d(_ v: Double) -> AggregateResult { AggregateResult(double: v, kind: 2) }
    static func sum(_ s: SumResult) -> AggregateResult {
        switch s {
        case .int(let v): return .i(v)
        case .uint(let v): return .u(v)
        case .float(let v): return .d(v)
        }
    }
}

/// The aggregates `am_reduce_ex` exposes, type-erased over the element type.
private protocol ExtendedAggregates {
    func aggregate(_ op: Int32, _ p1: Double) throws -> AggregateResult
}

extension MetalArray: ExtendedAggregates {
    /// A scalar of this element type from the `p1` argument (used by `index`).
    /// A Float64 column takes `p1` exactly; an integer column rounds it (values beyond 2^53 cannot be
    /// expressed by a double argument, which the header documents).
    private func scalar(_ p1: Double) -> T {
        if let exact = p1 as? T { return exact }
        if let integer = T.self as? any FixedWidthInteger.Type {
            return integer.init(truncatingIfNeeded: clampToInt64(p1.rounded())) as! T
        }
        return T(Float(p1))
    }

    /// One element as a result, keeping the column's own kind (signed, unsigned or floating point).
    private func result(_ v: T?) -> AggregateResult {
        guard let v else { return .null }
        if T.isFloatingPoint { return .d(v.asDouble) }
        if T.minValue < 0 as T { return .i(v.asInt64) }
        return .u(v.asUInt64)
    }

    fileprivate func aggregate(_ op: Int32, _ p1: Double) throws -> AggregateResult {
        switch op {
        case 0: return try product().map { AggregateResult.sum($0) } ?? .null
        case 1: return try variance(ddof: 0).map { AggregateResult.d($0) } ?? .null
        case 2: return try variance(ddof: 1).map { AggregateResult.d($0) } ?? .null
        case 3: return try stddev(ddof: 0).map { AggregateResult.d($0) } ?? .null
        case 4: return try stddev(ddof: 1).map { AggregateResult.d($0) } ?? .null
        case 5: return try quantile(p1).map { AggregateResult.d($0) } ?? .null
        case 6: return try approximateMedian().map { AggregateResult.d($0) } ?? .null
        case 7: return result(try mode()?.value)
        case 8: return .i(Int64(try countDistinct()))
        case 9: return result(try first())
        case 10: return result(try last())
        case 11: return .i(try index(of: scalar(p1)))
        case 12, 13: throw ArrowMetalError.unsupportedType("any / all need a boolean array, got \(T.arrowFormat)")
        case 14: return result(try minMax()?.min)
        case 15: return result(try minMax()?.max)
        case 16: return try mode().map { AggregateResult.i($0.count) } ?? .null
        default: throw ArrowMetalError.invalidArrowArray("unknown am_reduce_ex op \(op)")
        }
    }
}

private func aggregates(_ a: AnyMetalArray) throws -> any ExtendedAggregates {
    switch a {
    case .int8(let x): return x
    case .uint8(let x): return x
    case .int16(let x): return x
    case .uint16(let x): return x
    case .int32(let x): return x
    case .uint32(let x): return x
    case .int64(let x): return x
    case .uint64(let x): return x
    case .float32(let x): return x
    case .float64(let x): return x
    case .temporal(let t):
        // Temporal aggregates run on the storage integers, which is what the values are.
        switch t.storage { case .int32(let x): return x; case .int64(let x): return x }
    default:
        throw ArrowMetalError.unsupportedType("am_reduce_ex needs a primitive, temporal or boolean array, got \(a.arrowFormat)")
    }
}

/// Statistical and positional aggregates. See the op table in include/arrowmetal.h.
@_cdecl("am_reduce_ex")
public func am_reduce_ex(_ a: OpaquePointer?, _ op: Int32, _ p1: Double,
                         _ outI: UnsafeMutablePointer<Int64>?, _ outF: UnsafeMutablePointer<Double>?,
                         _ outKind: UnsafeMutablePointer<Int32>?, _ isNull: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let x = array(a) else { return 2 }
    do {
        let r: AggregateResult
        if case .boolean(let b) = x {
            switch op {
            case 12: r = .i(try b.anyTrue() ? 1 : 0)
            case 13: r = .i(try b.allTrue() ? 1 : 0)
            case 8:
                // A boolean column has at most two distinct values; the counts settle which.
                let (trueCount, validCount) = try b.trueAndValidCounts()
                r = .i(validCount == 0 ? 0 : ((trueCount == 0 || trueCount == validCount) ? 1 : 2))
            default: throw ArrowMetalError.unsupportedType("am_reduce_ex op \(op) is not defined for a boolean array")
            }
        } else {
            r = try aggregates(x).aggregate(op, p1)
        }
        outI?.pointee = r.int64
        outF?.pointee = r.double
        outKind?.pointee = r.kind
        isNull?.pointee = r.isNull ? 1 : 0
        return 0
    } catch { return fail(error) }
}

/// Expands a run-end encoded array into a flat one (GPU binary search plus a gather).
@_cdecl("am_run_end_decode")
public func am_run_end_decode(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do { produce(try x.runEndDecode(), out); return 0 } catch { return fail(error) }
}

/// Run-end encodes a primitive, boolean or temporal array (GPU boundary marks plus a scan).
@_cdecl("am_run_end_encode")
public func am_run_end_encode(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do { produce(try x.runEndEncode(), out); return 0 } catch { return fail(error) }
}
