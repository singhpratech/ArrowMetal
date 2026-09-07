import Foundation
import Metal

/// Group-by over **arbitrary** key columns: Arrow's `hash_*` aggregation without the caller having to
/// dictionary-encode first.
///
/// `GroupBy` on its own takes dense integer keys `0 ..< keyCount`. This type is the stage in front of
/// it: it turns any key column — sparse or negative integers, floats, booleans, temporal values, utf8 or
/// binary, dictionary-encoded columns, decimals — and any combination of up to a few such columns into
/// exactly those dense ids, and remembers enough to hand the key values back per group.
///
/// ## The mapping
///
/// There are two ways a column becomes dense ids, and the cheaper one is tried first.
///
/// **The range path**, for an integer, boolean, temporal or dictionary column whose values span a small
/// enough range: one GPU pass marks which values in `[min, max]` occur, a GPU scan turns the marks into
/// ranks, and a second pass reads each row's rank. No sort at all — three linear passes over the rows
/// plus a scan over the range. It is taken when `max - min + 1` is at most 2^24 and at most `max(2^16,
/// 4 * rows)`, which is what keeps the scan cheaper than the sort it replaces. This is the common case
/// for real key columns (categories, ids, dates, dictionary codes) and it is roughly **20x** faster than
/// the sort at 50 million rows and a thousand distinct keys.
///
/// **The hash-table path**, for utf8 and binary columns: `Kernels/StringHashTable.swift` hashes every
/// row to 64 bits, inserts it into an open-addressing table in device memory (equality decided by
/// comparing the bytes, so a hash collision can never merge two strings), ranks the occupied slots and
/// relabels them into first-seen order. Its cost scales with the *distinct* count rather than the row
/// count — 20 ms at 50 million rows and a thousand keys, where the sort it replaced took 200 ms.
///
/// **The sort path**, for everything else: `dictionaryEncode()` — normalise floats, argsort, mark run
/// boundaries by comparing adjacent sorted values, prefix-scan the marks into ranks, scatter the ranks
/// back to the rows. That is four GPU passes and one radix sort, and it is the same code `unique()` and
/// `value_counts()` run on. Floats, decimals and wide-range integers land here.
///
/// Several columns are folded **pairwise**: dense ids `a` (cardinality `Ka`) and `b` (cardinality `Kb`)
/// combine into the int64 key `a * Kb + b`, which is injective, and that key is re-encoded to squeeze it
/// back to `0 ..< K`. Re-encoding after every fold keeps the cardinality bounded by the row count
/// instead of multiplying out, so four columns never overflow. Better still, the composite's range is
/// known to be `Ka * Kb` without looking at it, so the re-encode usually takes the range path too and
/// the whole fold never sorts anything.
///
/// The alternative — hashing the per-column ids into one 64-bit key and resolving collisions — was
/// measured against this, with both composites re-encoded by the same sort so the only difference is
/// how the key is formed. At 50 million rows and ~100k groups the injective combine came to **160.0 ms**
/// and the 64-bit hash to **163.9 ms**, and the hash figure does not yet include the verification pass a
/// hash needs and an injective map does not. Same cost, more work — and only the injective key has a
/// range known in advance, which is what lets the fold skip the sort altogether.
///
/// Nulls follow Arrow: a null key is not skipped, it forms its own group. `gk_densify` gives every null
/// row the dedicated id `uniqueCount`, so the dense ids the aggregates see have no nulls at all and
/// every group `0 ..< groupCount` has at least one row.
///
/// ## Group order
///
/// Groups come out in a deterministic order — ascending by key for numeric, boolean, temporal and
/// decimal columns (nulls last), first-seen order for utf8 and binary ones, and lexicographic in the
/// column order for several columns. That is **not** pyarrow's order, which is first-seen for every
/// type. Label the rows with `groupKeys()` and sort both sides before comparing.
public final class GroupByKeys {
    /// The key columns as they were handed in, in the same order.
    public let columns: [AnyMetalArray]
    /// Dense group id of every row, `0 ..< groupCount`, never null.
    public let ids: MetalArray<Int32>
    /// Number of distinct key combinations present in the rows. Zero for an empty input.
    public let groupCount: Int
    /// Number of rows the keys came from.
    public let rows: Int
    public let context: MetalContext

    private var cachedRepRows: MetalArray<Int32>?
    private var cachedKeys: [AnyMetalArray]?
    private var cachedSegments: GroupSegments?

    /// The dense-key `GroupBy` every aggregate runs on. Its `keyCount` is `max(groupCount, 1)`, because
    /// `GroupBy` refuses a zero key count; the extra group of an empty input has no rows.
    public let groupBy: GroupBy<Int32>

    /// Maps `columns` to dense ids on the GPU. At least one column is required and every column must
    /// have the same length.
    public init(columns: [AnyMetalArray]) throws {
        guard let head = columns.first else {
            throw ArrowMetalError.invalidArrowArray("group-by needs at least one key column")
        }
        let n = head.length
        for c in columns.dropFirst() where c.length != n {
            throw ArrowMetalError.lengthMismatch(n, c.length)
        }
        try Dispatch.checkLength(n)
        let ctx = try GroupByKeys.contextOf(head)
        // Several integer key columns whose ranges multiply out small enough are packed into one key in
        // a single pass instead of folded pairwise (`Kernels/GroupByKeysDense.swift`). The ids it hands
        // back are the fold's own, value for value; it is the same key order arrived at with one range
        // encoding instead of three.
        if let fused = try GroupByKeysDense.ids(columns, ctx) {
            self.columns = columns
            self.ids = fused.0
            self.groupCount = fused.1
            self.rows = n
            self.context = ctx
            self.groupBy = try GroupBy(keys: fused.0, keyCount: Swift.max(fused.1, 1))
            return
        }
        var (ids, K) = try GroupByKeys.denseIds(head, ctx)
        for c in columns.dropFirst() {
            let (ids2, K2) = try GroupByKeys.denseIds(c, ctx)
            guard K == 0 || K2 == 0 || K <= Int.max / K2 else {
                throw ArrowMetalError.invalidArrowArray(
                    "multi-column group-by overflowed the 64-bit composite key (\(K) x \(K2) distinct)")
            }
            let composite = try GroupByKeys.combine(ids, ids2, cardinality: K2, ctx)
            (ids, K) = try GroupByKeys.encodeComposite(composite, span: K * K2, ctx)
        }
        self.columns = columns
        self.ids = ids
        self.groupCount = K
        self.rows = n
        self.context = ctx
        self.groupBy = try GroupBy(keys: ids, keyCount: Swift.max(K, 1))
    }

    /// The distinct key values, one row per group, in group order: `groupKeys()[j][g]` is the value of
    /// key column `j` for group `g`, with the same Arrow type as the input column (nulls included).
    ///
    /// GPU: a group-by minimum over the row indices picks one representative row per group, then each
    /// key column is gathered with `take`.
    public func groupKeys() throws -> [AnyMetalArray] {
        if let c = cachedKeys { return c }
        let rep = try representativeRows()
        let out = try columns.map { try $0.take(rep) }
        cachedKeys = out
        return out
    }

    /// Row index of one representative row per group (the lowest row index in the group).
    public func representativeRows() throws -> MetalArray<Int32> {
        if let c = cachedRepRows { return c }
        guard groupCount > 0 else {
            let empty = try MetalArray<Int32>([Int32](), context: context)
            cachedRepRows = empty
            return empty
        }
        let mins = try groupBy.min(try GroupByKeys.rowIndices(rows, context))
        let rep = groupCount == mins.length ? mins : try mins.slice(offset: 0, length: groupCount)
        // Every group has at least one row, so no representative is null; drop the all-ones bitmap.
        let clean = MetalArray<Int32>(length: rep.length, nullCount: 0, validity: nil,
                                      values: rep.values, context: context)
        cachedRepRows = clean
        return clean
    }

    /// The sorted key order shared by the segmented aggregates, built once and reused.
    func segments() throws -> GroupSegments {
        if let s = cachedSegments { return s }
        let s = try groupBy.segments()
        cachedSegments = s
        return s
    }

    /// Trims an aggregate result to `groupCount` rows (the padded key count of an empty input).
    public func trimExported(_ a: AnyMetalArray) throws -> AnyMetalArray { try trim(a) }

    /// Trims an aggregate result to `groupCount` rows (the padded key count of an empty input).
    func trim(_ a: AnyMetalArray) throws -> AnyMetalArray {
        a.length == groupCount ? a : try a.slice(offset: 0, length: groupCount)
    }

    // MARK: - One column to dense ids

    /// Dense ids and their cardinality for one key column. Null rows take the last id.
    public static func denseIds(_ column: AnyMetalArray, _ ctx: MetalContext) throws -> (MetalArray<Int32>, Int) {
        // The range path first: no sort at all when the values span a small enough range.
        if let fast = try rangeIds(column, ctx) { return fast }
        switch column {
        case .int8(let a): return try encode(a, ctx)
        case .uint8(let a): return try encode(a, ctx)
        case .int16(let a): return try encode(a, ctx)
        case .uint16(let a): return try encode(a, ctx)
        case .int32(let a): return try encode(a, ctx)
        case .uint32(let a): return try encode(a, ctx)
        case .int64(let a): return try encode(a, ctx)
        case .uint64(let a): return try encode(a, ctx)
        case .float32(let a): return try encode(a, ctx)
        case .float64(let a): return try encode(a, ctx)
        case .boolean(let a): return try encode(try a.toUInt8Array(), ctx)
        case .string(let a), .binary(let a):
            // The hash table writes the null group's id itself, so there is no `densify` pass here.
            if MetalStringArray.prefersHashTable(rows: a.length) { return try a.hashTableDenseIds() }
            let (codes, unique) = try a.dictionaryEncodeSorted()
            return try densify(codes, uniqueCount: unique.length, nullCount: a.nullCount, ctx)
        case .temporal(let t):
            switch t.storage {
            case .int32(let a): return try encode(a, ctx)
            case .int64(let a): return try encode(a, ctx)
            }
        case .dictionary(let codes, _):
            // The codes are already small dense integers; re-encoding them is the cheapest radix sort
            // available and drops dictionary entries no row uses, which is what Arrow's grouping does.
            return try encode(codes, ctx)
        case .decimal(let d):
            // A decimal is folded limb by limb: two rows are in the same group exactly when every
            // 64-bit limb matches, so the pairwise fold over the limbs is the grouping.
            var ids: MetalArray<Int32>? = nil
            var K = 0
            for limb in 0..<d.type.limbCount {
                let column = try limbColumn(d, limb: limb, ctx)
                // Back through denseIds so a limb with a narrow range (the usual case for the high
                // limb, which is all zeros, and often for the low one too) takes the range path.
                let (limbIds, limbK) = try denseIds(.uint64(column), ctx)
                if let previous = ids {
                    guard K == 0 || limbK == 0 || K <= Int.max / limbK else {
                        throw ArrowMetalError.invalidArrowArray("decimal group-by overflowed the composite key")
                    }
                    let composite = try combine(previous, limbIds, cardinality: limbK, ctx)
                    (ids, K) = try encodeComposite(composite, span: K * limbK, ctx)
                } else {
                    ids = limbIds; K = limbK
                }
            }
            guard let ids else { throw ArrowMetalError.unsupportedType("decimal with no limbs") }
            return (ids, K)
        case .runEndEncoded:
            return try denseIds(try column.runEndDecode(), ctx)
        case .list:
            throw ArrowMetalError.unsupportedType("group-by over a list key column")
        case .structure:
            throw ArrowMetalError.unsupportedType("group-by over a struct key column (pass its fields as separate key columns)")
        case .map:
            throw ArrowMetalError.unsupportedType("group-by over a map key column")
        case .union:
            throw ArrowMetalError.unsupportedType("group-by over a union key column")
        case .extended(let e): return try denseIds(e.storage, ctx)
        case .float16(let a): return try denseIds(.float32(try a.toFloat32()), ctx)
        case .null, .smallDecimal, .interval, .fixedBinary:
            throw ArrowMetalError.unsupportedType("group-by over a \(column.arrowFormat) key column is not implemented")
        }
    }


    // MARK: - The range fast path

    /// Range of an integer-like column as int64, or nil when the column is not integer-like, is empty,
    /// or holds unsigned values above `Int64.max` (where the subtraction below would wrap).
    static func integerRange(_ column: AnyMetalArray) throws -> (lo: Int64, hi: Int64)? {
        func of<T: ArrowPrimitive>(_ a: MetalArray<T>) throws -> (Int64, Int64)? {
            guard !T.isFloatingPoint, let mm = try a.minMax() else { return nil }
            if T.minValue < 0 as T { return (mm.min.asInt64, mm.max.asInt64) }
            let hi = mm.max.asUInt64
            guard hi <= UInt64(Int64.max) else { return nil }
            return (Int64(mm.min.asUInt64), Int64(hi))
        }
        switch column {
        case .int8(let a): return try of(a)
        case .uint8(let a): return try of(a)
        case .int16(let a): return try of(a)
        case .uint16(let a): return try of(a)
        case .int32(let a): return try of(a)
        case .uint32(let a): return try of(a)
        case .int64(let a): return try of(a)
        case .uint64(let a): return try of(a)
        case .float32, .float64, .boolean, .string, .binary, .temporal, .decimal,
             .dictionary, .list, .structure, .map, .union, .runEndEncoded,
             .null, .float16, .smallDecimal, .interval, .fixedBinary, .extended:
            return nil
        }
    }

    /// Whether a value range of `span` values is worth scanning instead of sorting `rows` rows.
    ///
    /// The scan costs O(span) and the sort O(rows log rows) with a large constant, so the crossover is
    /// generous: anything up to four values of range per row, capped at 2^24 so the occupancy array
    /// stays under 64 MB.
    static func rangeIsWorthIt(span: Int, rows: Int) -> Bool {
        span > 0 && span <= 1 << 24 && span <= Swift.max(1 << 16, 4 * rows)
    }

    /// Dense ids for an int64 column whose values are known to lie in `[lo, lo + span)`.
    ///
    /// Marks the occupied values, scans the marks, and reads each row's rank out of the scan. Returns
    /// the ids and their cardinality; null rows take the dedicated last id.
    static func rangeEncode(_ a: MetalArray<Int64>, lo: Int64, span: Int, _ ctx: MetalContext)
        throws -> (MetalArray<Int32>, Int) {
        let n = a.length
        let occBuf = try MetalArrowBuffer.allocate(byteCount: span * 4, context: ctx)
        let vv = a.validity ?? a.values
        let occupy = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_occupy",
                                      cacheKey: "groupbykeys/gk_occupy")
        try ctx.run { enc in
            enc.setComputePipelineState(occupy)
            enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
            Dispatch.setUInt(enc, a.validity == nil ? 0 : 1, index: 2)
            Dispatch.setScalar(enc, lo, index: 3)
            Dispatch.setLength(enc, n, nil, index: 4)
            enc.setBuffer(occBuf.mtl, offset: 0, index: 5)
            Dispatch.dispatch1D(enc, occupy, count: n)
        }
        ctx.retainUntilFlush(a)
        let occ = MetalArray<Int32>(length: span, nullCount: 0, validity: nil, values: occBuf, context: ctx)
        let cum = try occ.cumulativeSum()
        try ctx.syncPoint()
        let distinct = withExtendedLifetime(cum) { Int(cum.valuePointer[span - 1]) }
        let cardinality = distinct + (a.nullCount > 0 ? 1 : 0)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: ctx)
        if n > 0 {
            let rank = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_rank",
                                        cacheKey: "groupbykeys/gk_rank")
            try ctx.run { enc in
                enc.setComputePipelineState(rank)
                enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
                enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
                Dispatch.setUInt(enc, a.validity == nil ? 0 : 1, index: 2)
                Dispatch.setScalar(enc, lo, index: 3)
                Dispatch.setUInt(enc, distinct, index: 4)
                enc.setBuffer(cum.values.mtl, offset: cum.values.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                Dispatch.dispatch1D(enc, rank, count: n)
            }
            ctx.retainUntilFlush(a); ctx.retainUntilFlush(cum)
            try ctx.syncPoint()
        }
        return (MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx), cardinality)
    }

    /// The range path for one key column, or nil when the column does not qualify.
    static func rangeIds(_ column: AnyMetalArray, _ ctx: MetalContext) throws -> (MetalArray<Int32>, Int)? {
        let n = column.length
        guard n > 0 else { return nil }
        // Boolean, temporal and dictionary columns are integer columns underneath; unwrap them first.
        let integers: AnyMetalArray
        switch column {
        case .boolean(let b): integers = .uint8(try b.toUInt8Array())
        case .temporal(let t):
            switch t.storage { case .int32(let a): integers = .int32(a); case .int64(let a): integers = .int64(a) }
        case .dictionary(let codes, _): integers = .int32(codes)
        default: integers = column
        }
        guard let (lo, hi) = try integerRange(integers) else { return nil }
        let span = hi &- lo
        guard span >= 0, span < Int64(Int.max) - 1, rangeIsWorthIt(span: Int(span) + 1, rows: n) else { return nil }
        let as64 = try castToInt64(integers)
        return try rangeEncode(as64, lo: lo, span: Int(span) + 1, ctx)
    }

    /// An integer column as int64, without a copy when it already is one.
    private static func castToInt64(_ column: AnyMetalArray) throws -> MetalArray<Int64> {
        switch column {
        case .int8(let a): return try a.cast(to: Int64.self)
        case .uint8(let a): return try a.cast(to: Int64.self)
        case .int16(let a): return try a.cast(to: Int64.self)
        case .uint16(let a): return try a.cast(to: Int64.self)
        case .int32(let a): return try a.cast(to: Int64.self)
        case .uint32(let a): return try a.cast(to: Int64.self)
        case .int64(let a): return a
        case .uint64(let a): return try a.cast(to: Int64.self)
        default: throw ArrowMetalError.unsupportedType("not an integer column: \(column.arrowFormat)")
        }
    }

    /// Dense ids for a folded composite key, which is non-null and known to lie in `[0, span)`. The
    /// range path applies whenever that span is small enough; otherwise the composite is sorted.
    static func encodeComposite(_ composite: MetalArray<Int64>, span: Int, _ ctx: MetalContext)
        throws -> (MetalArray<Int32>, Int) {
        if composite.length > 0, rangeIsWorthIt(span: span, rows: composite.length) {
            return try rangeEncode(composite, lo: 0, span: span, ctx)
        }
        let (codes, unique) = try composite.dictionaryEncode()
        return (codes, unique.length)
    }

    /// `dictionaryEncode` plus `gk_densify`: dense ids with nulls given their own group.
    private static func encode<T: ArrowPrimitive>(_ a: MetalArray<T>, _ ctx: MetalContext) throws -> (MetalArray<Int32>, Int) {
        let (codes, unique) = try a.dictionaryEncode()
        return try densify(codes, uniqueCount: unique.length, nullCount: a.nullCount, ctx)
    }

    /// Replaces the null codes with the dedicated null group id. The result has no validity bitmap.
    private static func densify(_ codes: MetalArray<Int32>, uniqueCount: Int, nullCount: Int,
                                _ ctx: MetalContext) throws -> (MetalArray<Int32>, Int) {
        let n = codes.length
        let cardinality = uniqueCount + (nullCount > 0 ? 1 : 0)
        guard n > 0 else { return (try MetalArray<Int32>([Int32](), context: ctx), cardinality) }
        guard nullCount > 0 else {
            // No nulls: the codes are already the ids. Drop the (all-ones or absent) bitmap.
            return (MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: codes.values, context: ctx),
                    cardinality)
        }
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_densify", cacheKey: "groupbykeys/gk_densify")
        let v = codes.validity ?? codes.values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(codes.values.mtl, offset: codes.values.offset, index: 0)
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setUInt(enc, codes.validity == nil ? 0 : 1, index: 2)
            Dispatch.setUInt(enc, uniqueCount, index: 3)
            Dispatch.setLength(enc, n, nil, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(codes)
        try ctx.syncPoint()
        return (MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx), cardinality)
    }

    /// `a * cardinality + b` as an int64 column (no nulls, since both inputs are dense ids).
    public static func combine(_ a: MetalArray<Int32>, _ b: MetalArray<Int32>, cardinality: Int,
                        _ ctx: MetalContext) throws -> MetalArray<Int64> {
        let n = a.length
        guard n > 0 else { return try MetalArray<Int64>([Int64](), context: ctx) }
        let out = try MetalArrowBuffer.allocate(byteCount: n * 8, zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_combine", cacheKey: "groupbykeys/gk_combine")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
            enc.setBuffer(b.values.mtl, offset: b.values.offset, index: 1)
            Dispatch.setUInt(enc, Swift.max(cardinality, 1), index: 2)
            Dispatch.setLength(enc, n, nil, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(a); ctx.retainUntilFlush(b)
        try ctx.syncPoint()
        return MetalArray<Int64>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// `0 ..< n` as an int32 column, written by the GPU (a 50M-row host loop is not free).
    public static func rowIndices(_ n: Int, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        guard n > 0 else { return try MetalArray<Int32>([Int32](), context: ctx) }
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_iota", cacheKey: "groupbykeys/gk_iota")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            Dispatch.setLength(enc, n, nil, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        try ctx.syncPoint()
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// One 64-bit limb of a decimal column as a uint64 array carrying the decimal's validity.
    private static func limbColumn(_ d: MetalDecimalArray, limb: Int, _ ctx: MetalContext) throws -> MetalArray<UInt64> {
        let n = d.length
        guard n > 0 else { return try MetalArray<UInt64>([UInt64](), context: ctx) }
        let out = try MetalArrowBuffer.allocate(byteCount: n * 8, zeroed: false, context: ctx)
        let pso = try ctx.pipeline(source: GroupByKeysSource.source, function: "gk_limb", cacheKey: "groupbykeys/gk_limb")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(d.values.mtl, offset: d.values.offset, index: 0)
            Dispatch.setUInt(enc, d.type.limbCount, index: 1)
            Dispatch.setUInt(enc, limb, index: 2)
            Dispatch.setLength(enc, n, nil, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        try ctx.syncPoint()
        return MetalArray<UInt64>(length: n, nullCount: d.nullCount, validity: d.validity, values: out, context: ctx)
    }

    private static func contextOf(_ a: AnyMetalArray) throws -> MetalContext {
        switch a {
        case .int8(let x): return x.context
        case .uint8(let x): return x.context
        case .int16(let x): return x.context
        case .uint16(let x): return x.context
        case .int32(let x): return x.context
        case .uint32(let x): return x.context
        case .int64(let x): return x.context
        case .uint64(let x): return x.context
        case .float32(let x): return x.context
        case .float64(let x): return x.context
        case .boolean(let x): return x.context
        case .string(let x), .binary(let x): return x.context
        case .temporal(let x): return x.context
        case .decimal(let x): return x.context
        case .dictionary(let codes, _): return codes.context
        case .list(let x): return x.context
        case .structure(let x): return x.context
        case .map(let x): return x.context
        case .union(let x): return x.context
        case .runEndEncoded(let runEnds, _): return runEnds.context
        case .null(let x): return x.context
        case .float16(let x): return x.context
        case .smallDecimal(let x): return x.context
        case .interval(let x): return x.context
        case .fixedBinary(let x): return x.context
        case .extended(let x): return try contextOf(x.storage)
        }
    }
}

extension GroupBy where K == Int32 {
    /// Group-by over arbitrary key columns: `GroupBy(keys: [region, year])`.
    ///
    /// The keys are mapped to dense ids on the GPU (see `GroupByKeys`), so any supported key type works
    /// and nulls form their own group. The unique key values per group are **not** kept by this form —
    /// build a `GroupByKeys` directly when you want `groupKeys()`.
    ///
    /// `denseKeyCount` keeps the old fast path: pass it when the single key column already holds dense
    /// integers `0 ..< denseKeyCount`, and no mapping stage runs at all.
    public init(keys columns: [AnyMetalArray], denseKeyCount: Int? = nil) throws {
        if let denseKeyCount {
            guard columns.count == 1 else {
                throw ArrowMetalError.invalidArrowArray("denseKeyCount takes exactly one key column, got \(columns.count)")
            }
            guard case .int32(let k) = columns[0] else {
                throw ArrowMetalError.unsupportedType("denseKeyCount needs int32 keys, got \(columns[0].arrowFormat)")
            }
            try self.init(keys: k, keyCount: denseKeyCount)
            return
        }
        let mapped = try GroupByKeys(columns: columns)
        try self.init(keys: mapped.ids, keyCount: Swift.max(mapped.groupCount, 1))
    }
}

extension AnyMetalArray {
    /// Arrow `hash_*` grouping over this column: dense group ids plus the key values per group.
    public func groupByKeys() throws -> GroupByKeys { try GroupByKeys(columns: [self]) }
}
