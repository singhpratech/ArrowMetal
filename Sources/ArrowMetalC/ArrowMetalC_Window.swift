import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the window, shift, pairwise, cumulative-product/mean and rolling-window kernels, plus the
// multi-column sort. The op numbering here is the contract; it is repeated in include/arrowmetal.h and
// must not be reordered.
//
//   am_window   0 row_number      1 rank            2 dense_rank     3 percent_rank   4 cume_dist
//               5 shift           6 pairwise_diff   7 cumulative_prod                 8 cumulative_mean
//               9 rolling_sum    10 rolling_min    11 rolling_max   12 rolling_mean
//
// p1/p2 and the scalar carry the arguments each op needs; anything an op does not use is ignored.

private let errorKey = "ArrowMetalC.lastError"
private func setError(_ e: Error) { Thread.current.threadDictionary[errorKey] = "\(e)" }

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

/// Type-erased entry point for the window kernels, so the C layer writes the ten-way switch once.
protocol WindowOps {
    func amWindow(_ op: Int32, _ p1: Int64, _ p2: Int64, _ scalar: UnsafeRawPointer?) throws -> AnyMetalArray
}

extension MetalArray: WindowOps {
    func amWindow(_ op: Int32, _ p1: Int64, _ p2: Int64, _ scalar: UnsafeRawPointer?) throws -> AnyMetalArray {
        /// `p2 <= 0` means "no minPeriods given", which the Swift API spells as nil (the whole window).
        func minPeriods() -> Int? { p2 > 0 ? Int(p2) : nil }
        func window() throws -> Int {
            guard let w = Int(exactly: p1) else { throw ArrowMetalError.invalidArrowArray("window \(p1) is out of range") }
            return w
        }
        switch op {
        case 0: return .int32(try rowNumber())
        case 1: return .int32(try rank())
        case 2: return .int32(try denseRank())
        case 3: return .float64(try percentRank())
        case 4: return .float64(try cumeDist())
        case 5: return wrap(try shift(by: Int(p1), fill: scalar.map { $0.loadUnaligned(as: T.self) }))
        case 6: return wrap(try pairwiseDiff(period: Int(p1)))
        case 7: return wrap(try cumulativeProd())
        case 8: return .float64(try cumulativeMean())
        case 9: return wrap(try rollingSum(window: try window(), minPeriods: minPeriods()))
        case 10: return wrap(try rollingMin(window: try window(), minPeriods: minPeriods()))
        case 11: return wrap(try rollingMax(window: try window(), minPeriods: minPeriods()))
        case 12: return .float64(try rollingMean(window: try window(), minPeriods: minPeriods()))
        default: throw ArrowMetalError.invalidArrowArray("unknown window op \(op)")
        }
    }
}

private func withWindow<R>(_ a: AnyMetalArray, _ body: (any WindowOps) throws -> R) throws -> R {
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
    case .boolean: throw ArrowMetalError.unsupportedType("window functions need a primitive array, got boolean")
    case .string, .binary, .decimal, .list, .structure, .map, .union:
        throw ArrowMetalError.unsupportedType("window functions need a primitive array, got \(a.arrowFormat)")
    case .temporal(let t):
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .dictionary: throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    }
}

// MARK: - Exported operations

/// One window, shift, pairwise, cumulative or rolling op. See the op table at the top of this file and
/// in arrowmetal.h for what `p1`, `p2` and `scalar_or_null` mean for each op.
@_cdecl("am_window")
public func am_window(_ a: OpaquePointer?, _ op: Int32, _ p1: Int64, _ p2: Int64,
                      _ scalar: UnsafeRawPointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = handle(a) else { return 2 }
    return run(out) { try withWindow(x) { try $0.amWindow(op, p1, p2, scalar) } }
}

/// Multi-column (lexicographic) sort: int32 indices that order the rows by every column in turn, the
/// first being the most significant. `descending` has one entry per column (it may be NULL for all
/// ascending). Stable, and nulls come last in every key whichever direction that key is sorted in.
@_cdecl("am_lexsort")
public func am_lexsort(_ columns: UnsafeMutablePointer<OpaquePointer?>?, _ descending: UnsafePointer<Int32>?,
                       _ count: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let columns, count > 0 else { return 2 }
    var cols: [AnyMetalArray] = []
    var desc: [Bool] = []
    for i in 0..<Int(count) {
        guard let c = handle(columns[i]) else { return 2 }
        cols.append(c)
        desc.append(descending.map { $0[i] != 0 } ?? false)
    }
    return run(out) { .int32(try lexsortIndices(cols, descending: desc)) }
}
