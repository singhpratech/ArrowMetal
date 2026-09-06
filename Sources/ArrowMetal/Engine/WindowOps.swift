import Foundation

// Window functions over partitions.
//
// The shape is always the same, and it is the shape a GPU likes: **sort once, work in sorted order,
// scatter back**.
//
//  1. `lexsort` the rows by `(partition keys..., order keys...)`. Every partition is now one contiguous
//     run and inside it the rows are in the window's order. That is one radix sort per key column.
//  2. Compute the answer as a function of a row's *position* in that order, its partition's first
//     position and its tie group's first position — all three of which are a `GroupBy` minimum over the
//     row positions plus a `take`, kernels that already exist.
//  3. `take` the result through the inverse permutation, which is `argsort` of the permutation, so the
//     output is aligned to the caller's rows rather than to the sorted ones.
//
// Ranking, `lag` and `lead` fall out of step 2 with no new kernel at all: the index a row wants is
// `if_else(position - k >= partition_start, position - k, null)`, one fused expression, and `take` of a
// null index gives a null. The running and rolling functions (`cum_sum`, `rolling_*`) are the exception:
// their existing kernels are already exactly right per partition, so each partition's contiguous slice
// is handed to them and the results are concatenated. That is one dispatch per partition, which is the
// right trade below a few thousand partitions and the documented limit above it.
//
// Nulls follow SQL `ORDER BY x NULLS LAST`, which is what `lexsort` does in both directions: a null key
// sorts after every value and all nulls of one key form a single tie group.
enum WindowOps {

    /// Above this many partitions the per-partition path (cum_sum and the rolling functions) refuses to
    /// run, because it would issue one dispatch per partition.
    static let maxPartitionsForSlicedPath = 8192

    static func apply(_ input: MetalRecordBatch, _ specs: [WindowSpec]) throws -> MetalRecordBatch {
        var out = input
        for spec in specs { out = try one(out, spec) }
        return out
    }

    private static func one(_ input: MetalRecordBatch, _ spec: WindowSpec) throws -> MetalRecordBatch {
        let n = input.length
        let ctx = input.columns.first?.metalContext ?? .shared
        guard n > 0 else {
            let field = try PlanSchema.of(input).windowField(spec)
            return try adding(input, spec.name, try Executor.emptyColumn(field.exprType ?? .float64, ctx))
        }

        // A whole-partition aggregate needs no order and no permutation.
        if case .partitionAggregate(let op, let valueColumn) = spec.function {
            let keys = try spec.partitionBy.map { try columnNamed($0, input) }
            guard !keys.isEmpty else {
                let r = try input.query(ExprQuery(terminal: .aggregate([ExprAggregate(op, .column(valueColumn), name: spec.name)])))
                let scalar = try Executor.scalarColumn(r.scalars[0], as: nil, context: ctx)
                return try adding(input, spec.name, try broadcast(scalar, to: n, ctx))
            }
            let gk = try GroupByKeys(columns: keys)
            try ctx.flush(reopen: true)
            let agg = try gk.trimExported(try Executor.aggregateOne(gk, ExprAggregate(op, .column(valueColumn), name: spec.name),
                                                                    over: input, ctx))
            return try adding(input, spec.name, try agg.take(gk.ids))
        }

        // 1. Sort by (partition, order).
        var sortCols: [AnyMetalArray] = [], sortDesc: [Bool] = []
        for c in spec.partitionBy { sortCols.append(try columnNamed(c, input)); sortDesc.append(false) }
        for k in spec.orderBy { sortCols.append(try columnNamed(k.column, input)); sortDesc.append(k.descending) }
        let positions = try GroupByKeys.rowIndices(n, ctx)
        let perm: MetalArray<Int32>
        if sortCols.isEmpty { perm = positions } else { perm = try lexsortIndices(sortCols, descending: sortDesc) }
        let inverse = sortCols.isEmpty ? positions : try perm.argsort()

        // 2. Partition ids in sorted order, and each row's partition start.
        //
        // With no partitioning the whole input is one partition, so every row's start is 0 and its end
        // is n - 1; those constant columns are only materialised if the chosen function actually reads
        // them, because at 50M rows filling one costs more than the window function does.
        var startPerRow: MetalArray<Int32>! = nil
        var endPerRow: MetalArray<Int32>! = nil
        var sortedPartitionIds: MetalArray<Int32>? = nil
        var partitionCount = 1
        if !spec.partitionBy.isEmpty {
            let gk = try GroupByKeys(columns: try spec.partitionBy.map { try columnNamed($0, input) })
            try ctx.flush(reopen: true)
            partitionCount = Swift.max(gk.groupCount, 1)
            let sortedIds = try gk.ids.take(perm)
            let gb = try GroupBy(keys: sortedIds, keyCount: partitionCount)
            let starts = try gb.min64(positions)
            let ends = try gb.max64(positions)
            startPerRow = try starts.take(sortedIds)
            endPerRow = try ends.take(sortedIds)
            sortedPartitionIds = sortedIds
        }
        func starts() throws -> MetalArray<Int32> {
            if startPerRow == nil { startPerRow = try constant(0, n, ctx) }
            return startPerRow
        }
        func ends() throws -> MetalArray<Int32> {
            if endPerRow == nil { endPerRow = try constant(Int32(n - 1), n, ctx) }
            return endPerRow
        }

        let result: AnyMetalArray
        switch spec.function {
        case .rowNumber:
            result = .int32(try offsetFrom(positions, try starts(), ctx))

        case .rank, .denseRank:
            let tieStart = try tieGroupStarts(input, spec, perm, sortedPartitionIds, positions, partitionCount, ctx)
            if case .rank = spec.function {
                result = .int32(try offsetFrom(tieStart.starts, try starts(), ctx))
            } else {
                result = .int32(try denseRanks(tieStart.starts, positions, try starts(), ctx))
            }

        case .lag(let c, let k), .lead(let c, let k):
            var delta = k
            if case .lead = spec.function { delta = -k }
            let idx = try neighbourIndex(positions, try starts(), try ends(), delta: delta, ctx)
            result = try columnNamed(c, input).take(perm).take(idx)

        case .cumSum(let c), .rollingSum(let c, _), .rollingMean(let c, _),
             .rollingMin(let c, _), .rollingMax(let c, _):
            guard partitionCount <= maxPartitionsForSlicedPath else {
                throw ArrowMetalError.unsupportedType(
                    "window \(spec.function.label) over \(partitionCount) partitions: the running and rolling "
                    + "functions run one dispatch per partition and are capped at \(maxPartitionsForSlicedPath)")
            }
            let sorted = try columnNamed(c, input).take(perm)
            result = try slicedPerPartition(sorted, spec.function, try starts(), try ends(),
                                            partitionCount: partitionCount, n: n, ctx)

        case .partitionAggregate:
            throw ArrowMetalError.unsupportedType("unreachable: partition aggregate is handled above")
        }

        // 3. Back to the caller's row order.
        return try adding(input, spec.name, sortCols.isEmpty ? result : try result.take(inverse))
    }

    // MARK: - Pieces

    private static func columnNamed(_ n: String, _ b: MetalRecordBatch) throws -> AnyMetalArray {
        guard let c = b[n] else { throw ArrowMetalError.invalidArrowArray("window: no column named \(n)") }
        return c
    }

    private static func adding(_ b: MetalRecordBatch, _ name: String, _ col: AnyMetalArray) throws -> MetalRecordBatch {
        var names = b.names, cols = b.columns
        if let i = names.firstIndex(of: name) { cols[i] = col } else { names.append(name); cols.append(col) }
        return try MetalRecordBatch(names: names, columns: cols)
    }

    private static func constant(_ v: Int32, _ n: Int, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        try MetalArray<Int32>([Int32](repeating: v, count: n), context: ctx)
    }

    private static func broadcast(_ scalar: AnyMetalArray, to n: Int, _ ctx: MetalContext) throws -> AnyMetalArray {
        try scalar.take(try constant(0, n, ctx))
    }

    /// `a - b + 1` as int32, in one fused kernel.
    private static func offsetFrom(_ a: MetalArray<Int32>, _ b: MetalArray<Int32>, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        let r = try runExprQuery(query().project([("r", (col("a") - col("b") + 1).cast(to: .int32))]),
                                 names: ["a", "b"], columns: [.int32(a), .int32(b)], context: ctx)
        guard case .int32(let out) = r.columns[0] else {
            throw ArrowMetalError.invalidArrowArray("window: rank did not come back as int32")
        }
        return out
    }

    /// `if_else(pos - delta >= start && pos - delta <= end, pos - delta, null)`: the index a `lag` or
    /// `lead` wants, null where it would leave its partition.
    private static func neighbourIndex(_ pos: MetalArray<Int32>, _ start: MetalArray<Int32>,
                                       _ end: MetalArray<Int32>, delta: Int, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        let target = col("p") - delta
        let e = Expr.ifElse((target >= col("s")) && (target <= col("e")), target.cast(to: .int32), nullLit(.int32))
        let r = try runExprQuery(query().project([("i", e)]),
                                 names: ["p", "s", "e"], columns: [.int32(pos), .int32(start), .int32(end)], context: ctx)
        guard case .int32(let out) = r.columns[0] else {
            throw ArrowMetalError.invalidArrowArray("window: neighbour index did not come back as int32")
        }
        return out
    }

    private struct TieGroups { var starts: MetalArray<Int32> }

    /// The first sorted position of each row's tie group: rows agreeing on the partition and every
    /// order key are one group, which is exactly what `GroupByKeys` over those columns computes.
    private static func tieGroupStarts(_ input: MetalRecordBatch, _ spec: WindowSpec,
                                       _ perm: MetalArray<Int32>, _ sortedPartitionIds: MetalArray<Int32>?,
                                       _ positions: MetalArray<Int32>, _ partitionCount: Int,
                                       _ ctx: MetalContext) throws -> TieGroups {
        var cols: [AnyMetalArray] = []
        if let p = sortedPartitionIds { cols.append(.int32(p)) }
        for k in spec.orderBy { cols.append(try columnNamed(k.column, input).take(perm)) }
        guard !cols.isEmpty else {
            // No order at all: every row of a partition ties, so the tie start is the partition start.
            return TieGroups(starts: try constant(0, positions.length, ctx))
        }
        let gk = try GroupByKeys(columns: cols)
        try ctx.flush(reopen: true)
        let gb = try GroupBy(keys: gk.ids, keyCount: Swift.max(gk.groupCount, 1))
        let mins = try gb.min64(positions)
        return TieGroups(starts: try mins.take(gk.ids))
    }

    /// `DENSE_RANK`: the number of distinct tie groups seen so far inside the partition.
    ///
    /// A row that starts a tie group marks a 1; the inclusive running sum of those marks counts groups
    /// across the whole sorted array, and subtracting the count at the partition's first row (where the
    /// mark is always 1) turns the global count into a per-partition one.
    private static func denseRanks(_ tieStart: MetalArray<Int32>, _ positions: MetalArray<Int32>,
                                   _ partitionStart: MetalArray<Int32>, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        let marks = try runExprQuery(
            query().project([("m", Expr.ifElse(col("t") == col("p"), .typedInt(1, .int32), .typedInt(0, .int32)))]),
            names: ["t", "p"], columns: [.int32(tieStart), .int32(positions)], context: ctx)
        guard case .int32(let m) = marks.columns[0] else {
            throw ArrowMetalError.invalidArrowArray("window: dense_rank marks were not int32")
        }
        let running = try m.cumulative(.sum)
        let base = try running.take(partitionStart)
        let r = try runExprQuery(query().project([("r", (col("g") - col("b") + 1).cast(to: .int32))]),
                                 names: ["g", "b"], columns: [.int32(running), .int32(base)], context: ctx)
        guard case .int32(let out) = r.columns[0] else {
            throw ArrowMetalError.invalidArrowArray("window: dense_rank did not come back as int32")
        }
        return out
    }

    /// The running and rolling functions, one contiguous partition at a time.
    private static func slicedPerPartition(_ sorted: AnyMetalArray, _ fn: WindowFunction,
                                           _ startPerRow: MetalArray<Int32>, _ endPerRow: MetalArray<Int32>,
                                           partitionCount: Int, n: Int, _ ctx: MetalContext) throws -> AnyMetalArray {
        try ctx.flush(reopen: true)
        // Partition boundaries in sorted order, read once.
        var bounds: [(Int, Int)] = []
        let sp = startPerRow.values.typed(Int32.self)
        let ep = endPerRow.values.typed(Int32.self)
        var i = 0
        while i < n {
            let lo = Int(sp[i]), hi = Int(ep[i])
            bounds.append((lo, hi - lo + 1))
            i = hi + 1
        }
        var parts: [AnyMetalArray] = []
        for (offset, length) in bounds {
            let slice = try sorted.slice(offset: offset, length: length)
            parts.append(try runningOne(slice, fn))
        }
        return parts.count == 1 ? parts[0] : try concatMetalArrays(parts)
    }

    private static func runningOne(_ a: AnyMetalArray, _ fn: WindowFunction) throws -> AnyMetalArray {
        func apply<T: ArrowPrimitive>(_ v: MetalArray<T>, _ wrap: (MetalArray<T>) -> AnyMetalArray) throws -> AnyMetalArray {
            switch fn {
            case .cumSum: return wrap(try v.cumulative(.sum))
            case .rollingSum(_, let w): return wrap(try v.rollingSum(window: w))
            case .rollingMin(_, let w): return wrap(try v.rollingMin(window: w))
            case .rollingMax(_, let w): return wrap(try v.rollingMax(window: w))
            case .rollingMean(_, let w): return .float64(try v.rollingMean(window: w))
            default: throw ArrowMetalError.unsupportedType("window: \(fn.label) is not a running function")
            }
        }
        switch a {
        case .int8(let v): return try apply(v) { .int8($0) }
        case .int16(let v): return try apply(v) { .int16($0) }
        case .int32(let v): return try apply(v) { .int32($0) }
        case .int64(let v): return try apply(v) { .int64($0) }
        case .uint8(let v): return try apply(v) { .uint8($0) }
        case .uint16(let v): return try apply(v) { .uint16($0) }
        case .uint32(let v): return try apply(v) { .uint32($0) }
        case .uint64(let v): return try apply(v) { .uint64($0) }
        case .float32(let v): return try apply(v) { .float32($0) }
        case .float64(let v): return try apply(v) { .float64($0) }
        default:
            throw ArrowMetalError.unsupportedType("window \(fn.label) over a \(a.arrowFormat) column")
        }
    }
}
