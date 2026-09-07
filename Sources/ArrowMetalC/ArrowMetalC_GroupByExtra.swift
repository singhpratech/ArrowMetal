import Foundation
import CArrowABI
import ArrowMetal

// C ABI for group-by over arbitrary key columns and for the grouped aggregates that need it.
//
// `am_group_by` (in ArrowMetalC.swift) stays exactly as it was: dense integer keys, five aggregates,
// one call. This file adds the other shape — hand it the key columns themselves and it maps them to
// dense ids on the GPU, keeps the mapping in a handle, and answers many aggregates against it:
//
//     am_group_by_keys(cols, 2, &gb);
//     am_group_by_keys_result(gb, 0, &region);      // the key values, one row per group
//     am_group_agg_ex(gb, revenue, AM_GAGG_SUM, 0, &sums);
//     am_group_by_release(gb);
//
// The op table lives in include/arrowmetal.h and is the contract; the switch below implements it.

private let errorKey = "ArrowMetalC.lastError"
private func fail(_ e: Error) -> Int32 { Thread.current.threadDictionary[errorKey] = "\(e)"; return 1 }

@inline(__always) private func array(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func produce(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>) {
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
}

/// A retained `GroupByKeys`, handed out as an opaque `am_groupby*`.
final class GroupByBox {
    let keys: GroupByKeys
    init(_ k: GroupByKeys) { self.keys = k }
}

@inline(__always) private func groupBy(_ p: OpaquePointer?) -> GroupByKeys? {
    guard let p else { return nil }
    return Unmanaged<GroupByBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().keys
}

/// Maps one or more key columns to dense group ids on the GPU and returns a handle.
/// Release it with `am_group_by_release`.
@_cdecl("am_group_by_keys")
public func am_group_by_keys(_ columns: UnsafePointer<OpaquePointer?>?, _ count: Int64,
                             _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let columns, let out, count > 0 else { return 2 }
    do {
        var cols: [AnyMetalArray] = []
        for i in 0..<Int(count) {
            guard let c = array(columns[i]) else { return 2 }
            cols.append(c)
        }
        let gbk = try GroupByKeys(columns: cols)
        out.pointee = OpaquePointer(Unmanaged.passRetained(GroupByBox(gbk)).toOpaque())
        return 0
    } catch { return fail(error) }
}

/// Number of groups the mapping produced.
@_cdecl("am_group_by_group_count")
public func am_group_by_group_count(_ gb: OpaquePointer?) -> Int64 { Int64(groupBy(gb)?.groupCount ?? -1) }

/// The i-th key column, one row per group, in group order and with the input column's Arrow type.
@_cdecl("am_group_by_keys_result")
public func am_group_by_keys_result(_ gb: OpaquePointer?, _ i: Int64,
                                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let g = groupBy(gb), let out else { return 2 }
    do {
        let cols = try g.groupKeys()
        guard i >= 0, Int(i) < cols.count else {
            throw ArrowMetalError.invalidArrowArray("key column \(i) of \(cols.count)")
        }
        produce(cols[Int(i)], out)
        return 0
    } catch { return fail(error) }
}

/// The dense group id of every row (int32, never null) — the input to the dense `am_group_by`.
@_cdecl("am_group_by_ids")
public func am_group_by_ids(_ gb: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let g = groupBy(gb), let out else { return 2 }
    produce(.int32(g.ids), out)
    return 0
}

@_cdecl("am_group_by_release")
public func am_group_by_release(_ gb: OpaquePointer?) {
    guard let gb else { return }
    Unmanaged<GroupByBox>.fromOpaque(UnsafeRawPointer(gb)).release()
}

/// Grouped aggregates against a key mapping. See the op table in include/arrowmetal.h.
@_cdecl("am_group_agg_ex")
public func am_group_agg_ex(_ gb: OpaquePointer?, _ values: OpaquePointer?, _ op: Int32, _ p1: Double,
                            _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let g = groupBy(gb), let out else { return 2 }
    do {
        let result = try groupedAggregate(g, values.flatMap(array), op, p1)
        produce(try g.trimExported(result), out)
        return 0
    } catch { return fail(error) }
}

/// Arrow `hash_pivot_wider` over a utf8 pivot-key column: one struct field per name.
@_cdecl("am_group_pivot_wider")
public func am_group_pivot_wider(_ gb: OpaquePointer?, _ pivotKeys: OpaquePointer?, _ values: OpaquePointer?,
                                 _ names: UnsafePointer<UnsafePointer<CChar>?>?, _ nameCount: Int64,
                                 _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let g = groupBy(gb), let pivot = array(pivotKeys), let v = array(values), let names, let out else { return 2 }
    do {
        guard case .string(let p) = pivot else {
            throw ArrowMetalError.unsupportedType("pivot_wider needs a utf8 pivot-key column, got \(pivot.arrowFormat)")
        }
        var labels: [String] = []
        for i in 0..<Int(nameCount) {
            guard let c = names[i] else { return 2 }
            labels.append(String(cString: c))
        }
        let s = try withGroupedValues(v) { try $0.pivot(g.groupBy, p, labels) }
        produce(.structure(s), out)
        return 0
    } catch { return fail(error) }
}

// MARK: - the op table

private func groupedAggregate(_ g: GroupByKeys, _ values: AnyMetalArray?, _ op: Int32, _ p1: Double) throws -> AnyMetalArray {
    let gb = g.groupBy
    // Ops that need no value column.
    switch op {
    case 1: return .int64(try gb.countAll())
    default: break
    }
    guard let values else { throw ArrowMetalError.invalidArrowArray("am_group_agg_ex op \(op) needs a value column") }
    if case .boolean(let b) = values {
        switch op {
        case 14: return .boolean(try gb.any(b))
        case 15: return .boolean(try gb.all(b))
        case 2: return .int64(try gb.count(try b.toUInt8Array()))
        default: throw ArrowMetalError.unsupportedType("am_group_agg_ex op \(op) is not defined for a boolean column")
        }
    }
    return try withGroupedValues(values) { try $0.apply(gb, op, p1) }
}

/// Type-erased grouped aggregates, so the C layer does not repeat the ten-way switch per op.
private protocol GroupedValueOps {
    func apply<K: ArrowIndex>(_ gb: GroupBy<K>, _ op: Int32, _ p1: Double) throws -> AnyMetalArray
    func pivot<K: ArrowIndex>(_ gb: GroupBy<K>, _ keys: MetalStringArray, _ names: [String]) throws -> MetalStructArray
    func scalarExtra(_ op: Int32, _ p1: Double) throws -> Double?
}

extension MetalArray: GroupedValueOps {
    fileprivate func apply<K: ArrowIndex>(_ gb: GroupBy<K>, _ op: Int32, _ p1: Double) throws -> AnyMetalArray {
        switch op {
        case 0: return try summed(gb)
        case 2: return .int64(try gb.countValid(self))
        case 3: return .float64(try meaned(gb))
        case 4: return anyArray(try gb.minMax(self).min)
        case 5: return anyArray(try gb.minMax(self).max)
        case 6: return .structure(try gb.minMaxStruct(self))
        case 7: return anyArray(try gb.first(self))
        case 8: return anyArray(try gb.last(self))
        case 9: return .structure(try gb.firstLast(self))
        case 10: return anyArray(try gb.one(self))
        case 11: return .list(try gb.list(self))
        case 12: return .list(try gb.distinct(self))
        case 13: return .int64(try gb.countDistinct(self))
        case 16: return try producted(gb)
        // The grouped variance now runs in binary64 on the values as they are, Float64 included.
        case 17: return .float64(try gb.variance(self, ddof: 0))
        case 18: return .float64(try gb.variance(self, ddof: 1))
        case 19: return .float64(try gb.stddev(self, ddof: 0))
        case 20: return .float64(try gb.stddev(self, ddof: 1))
        case 21: return .float64(try gb.approximateMedian(self))
        case 22: return .float64(try gb.quantile(self, p1))
        case 23: return .float64(try gb.skew(self))
        case 24: return .float64(try gb.kurtosis(self))
        case 25: return .float64(try gb.tdigest(self, p1))
        default: throw ArrowMetalError.invalidArrowArray("unknown am_group_agg_ex op \(op)")
        }
    }

    fileprivate func pivot<K: ArrowIndex>(_ gb: GroupBy<K>, _ keys: MetalStringArray, _ names: [String]) throws -> MetalStructArray {
        try gb.pivotWider(pivotKeys: keys, values: self, names: names)
    }

    /// `hash_sum`, keeping Arrow's output type per input type.
    private func summed<K: ArrowIndex>(_ gb: GroupBy<K>) throws -> AnyMetalArray {
        if let f = self as? MetalArray<Float> { return .float64(try gb.sumFloatAsDouble(f)) }
        if let d = self as? MetalArray<Double> { return .float64(try gb.sumDouble(d)) }
        if let u = self as? MetalArray<UInt64> { return .uint64(try gb.sumUnsigned(u)) }
        return .int64(try sumInteger(gb))
    }

    private func meaned<K: ArrowIndex>(_ gb: GroupBy<K>) throws -> MetalArray<Double> {
        if let f = self as? MetalArray<Float> { return try gb.meanFloat(f) }
        if let d = self as? MetalArray<Double> { return try gb.meanDouble(d) }
        return try gb.meanErasedInteger(self)
    }

    private func producted<K: ArrowIndex>(_ gb: GroupBy<K>) throws -> AnyMetalArray {
        if T.isFloatingPoint { return .float64(try gb.productFloat(self)) }
        return .int64(try gb.productIntErased(self))
    }

    private func sumInteger<K: ArrowIndex>(_ gb: GroupBy<K>) throws -> MetalArray<Int64> {
        guard !T.isFloatingPoint else { throw ArrowMetalError.unsupportedType("integer sum over \(T.arrowFormat)") }
        return try gb.sumErasedInteger(self)
    }

    private func anyArray(_ a: MetalArray<T>) -> AnyMetalArray { wrap(a) }

    /// The column as something the grouped variance accepts: itself, unless it is Float64, which is
    /// narrowed to Float32 because the deviations are formed in Float32 anyway.
    private func narrowed() throws -> any NarrowedStatistics {
        if let d = self as? MetalArray<Double> { return try d.cast(to: Float.self) }
        return self
    }

    fileprivate func scalarExtra(_ op: Int32, _ p1: Double) throws -> Double? {
        switch op {
        case 0: return try skew()
        case 1: return try kurtosis()
        case 2: return try tdigest(p1)
        case 3: return try skew(biased: false)
        case 4: return try kurtosis(biased: false)
        default: throw ArrowMetalError.invalidArrowArray("unknown am_reduce_ex2 op \(op)")
        }
    }
}

private func withGroupedValues<R>(_ a: AnyMetalArray, _ body: (any GroupedValueOps) throws -> R) throws -> R {
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
    case .temporal(let t):
        switch t.storage { case .int32(let x): return try body(x); case .int64(let x): return try body(x) }
    case .float16(let x): return try body(try x.toFloat32())
    case .extended(let e): return try withGroupedValues(e.storage, body)
    case .boolean, .string, .binary, .dictionary, .decimal, .list, .structure, .map, .union, .runEndEncoded,
         .null, .smallDecimal, .interval, .fixedBinary:
        throw ArrowMetalError.unsupportedType("grouped aggregate over \(a.arrowFormat)")
    }
}

// MARK: - scalar skew / kurtosis / tdigest

/// The scalar aggregates `am_reduce_ex` does not cover: 0 skew, 1 kurtosis, 2 tdigest(p1),
/// 3 skew (sample-corrected), 4 kurtosis (sample-corrected). Writes one double.
@_cdecl("am_reduce_ex2")
public func am_reduce_ex2(_ a: OpaquePointer?, _ op: Int32, _ p1: Double,
                          _ outF: UnsafeMutablePointer<Double>?, _ isNull: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let x = array(a) else { return 2 }
    do {
        let v = try withGroupedValues(x) { try $0.scalarExtra(op, p1) }
        outF?.pointee = v ?? 0
        isNull?.pointee = v == nil ? 1 : 0
        return 0
    } catch { return fail(error) }
}




/// `hash_variance` / `hash_stddev` over a column already narrowed to something the kernel accepts.
private protocol NarrowedStatistics {
    func variance<K: ArrowIndex>(_ gb: GroupBy<K>, ddof: Int) throws -> MetalArray<Double>
    func stddev<K: ArrowIndex>(_ gb: GroupBy<K>, ddof: Int) throws -> MetalArray<Double>
}

extension MetalArray: NarrowedStatistics {
    fileprivate func variance<K: ArrowIndex>(_ gb: GroupBy<K>, ddof: Int) throws -> MetalArray<Double> {
        try gb.variance(self, ddof: ddof)
    }
    fileprivate func stddev<K: ArrowIndex>(_ gb: GroupBy<K>, ddof: Int) throws -> MetalArray<Double> {
        try gb.stddev(self, ddof: ddof)
    }
}
