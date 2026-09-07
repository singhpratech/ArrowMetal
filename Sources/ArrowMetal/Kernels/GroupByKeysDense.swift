import Foundation
import Metal

// The fused dense path for several **integer** key columns.
//
// `GroupByKeys` folds key columns pairwise: each column becomes dense ids of its own, the ids are
// combined into an int64 composite `a * Kb + b`, and the composite is re-encoded back to `0 ..< K`.
// When every column is an integer with a small range, that fold does the same work three times over —
// two range encodings (mark, scan, rank) plus a third over an int64 composite it had to materialise —
// where one pass over the rows could have produced the composite directly.
//
// This file is that one pass. When every key column is integer-like and the product of the column
// ranges is small enough to scan, each row's key becomes
//
//     packed = Σ_j (k_j - min_j) * stride_j,     stride_j = Π_{m > j} span_m
//
// with a null in column j taking the slot `span_j - 1` reserved above that column's values. The pack
// kernel marks the occupancy of the packed value in the same pass, one scan ranks the values that
// occur, and a second pass reads each row's rank.
//
// **The ids are the ones the fold produces, value for value.** Both paths label the groups by the
// ascending order of the composite key, and the composite orders lexicographically by (column 0,
// column 1, ...) with nulls last in each column — the pairwise fold because each column's ids are its
// values' ascending ranks, this path because `k_j - min_j` is monotone in `k_j`. Ranking the values
// that actually occur then collapses both to the same dense `0 ..< K`, so `groupCount`, the per-row
// ids and therefore the group order of every aggregate are unchanged.
//
// At 10 million rows and two int32 keys with ~1024 groups this is 6.5 ms -> 3.4 ms, against pyarrow's
// 5.7 ms; the single-int32-key row next to it, which does the same aggregation over one range
// encoding, is 2.6 ms.

/// One integer-like key column, reduced to what the pack kernel needs: the buffer to read, the MSL
/// type to read it as, the validity bitmap (nil when the column has no nulls), and the value range.
struct DenseKeyColumn {
    let values: MetalArrowBuffer
    let validity: MetalArrowBuffer?
    let mslType: String
    let lo: Int64
    /// Number of slots this column occupies in the packed key: `hi - lo + 1`, plus one for the nulls.
    let span: Int
    /// The slot a null row of this column takes: the last one, so nulls sort after every value.
    var nullSlot: Int { span - 1 }
}

/// The `lo`, `stride` and `nullSlot` of up to `GroupByKeysDense.maxColumns` columns. The layout matches
/// `GkPackParams` in the MSL below: four int64s, then four int32s, then four int32s.
struct GkPackParams {
    var lo: (Int64, Int64, Int64, Int64) = (0, 0, 0, 0)
    var stride: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
    var nullSlot: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
}

enum GroupByKeysDense {

    /// How many columns the packed key covers. Past this the pairwise fold takes over; four integer
    /// key columns whose ranges multiply to under 2^24 is already an unusual group-by.
    static let maxColumns = 4

    /// Set `ARROWMETAL_NO_DENSE_KEYS=1` to force every multi-column group-by back onto the pairwise
    /// fold. Nothing in the library needs it — both paths hand back the same ids — but it is how the
    /// equivalence tests compare them, and how the before/after benchmark numbers were measured.
    static let disabled = ProcessInfo.processInfo.environment["ARROWMETAL_NO_DENSE_KEYS"] != nil

    /// Whether a packed range of `span` values is worth scanning for `rows` rows.
    ///
    /// Tighter than `GroupByKeys.rangeIsWorthIt`, deliberately. The product of the column ranges can be
    /// far larger than the number of key combinations that actually occur — two columns of 4096 sparse
    /// values each multiply out to 2^24 whatever their cardinality — and the pairwise fold, which ranks
    /// each column before combining, scans only the cardinalities. Requiring the product to stay inside
    /// the row count keeps the occupancy scan below one pass over the rows, which is the cost this path
    /// is removing several of.
    static func worthIt(span: Int, rows: Int) -> Bool {
        span > 0 && span <= 1 << 24 && span <= Swift.max(1 << 16, rows)
    }

    /// Dense ids for several key columns in one pass, or nil when the columns do not qualify: too many
    /// of them, one that is not integer-like, an empty input, or a packed range too large to scan.
    static func ids(_ columns: [AnyMetalArray], _ ctx: MetalContext) throws -> (MetalArray<Int32>, Int)? {
        let n = columns.first?.length ?? 0
        guard !disabled, n > 0, columns.count >= 2, columns.count <= maxColumns else { return nil }
        var specs: [DenseKeyColumn] = []
        for c in columns {
            guard let s = try column(c, ctx) else { return nil }
            specs.append(s)
        }
        // Strides right to left, so column 0 is the most significant: the fold's own key order.
        var strides = [Int](repeating: 1, count: specs.count)
        var total = 1
        for j in stride(from: specs.count - 1, through: 0, by: -1) {
            strides[j] = total
            guard total <= Int.max / specs[j].span else { return nil }
            total *= specs[j].span
            guard total <= 1 << 24 else { return nil }
        }
        guard worthIt(span: total, rows: n) else { return nil }

        var params = GkPackParams()
        withUnsafeMutableBytes(of: &params.lo) { p in
            let t = p.bindMemory(to: Int64.self)
            for (j, s) in specs.enumerated() { t[j] = s.lo }
        }
        withUnsafeMutableBytes(of: &params.stride) { p in
            let t = p.bindMemory(to: Int32.self)
            for j in 0..<specs.count { t[j] = Int32(strides[j]) }
        }
        withUnsafeMutableBytes(of: &params.nullSlot) { p in
            let t = p.bindMemory(to: Int32.self)
            for (j, s) in specs.enumerated() { t[j] = Int32(s.nullSlot) }
        }

        let packed = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let occBuf = try MetalArrowBuffer.allocate(byteCount: total * 4, context: ctx)
        let shape = specs.map { (msl: $0.mslType, hasValidity: $0.validity != nil) }
        let key = "groupbykeysdense/pack/" + shape.map { "\($0.msl)\($0.hasValidity ? "v" : "")" }.joined(separator: "_")
        let packPSO = try ctx.pipeline(source: packSource(shape), function: "gkp_pack", cacheKey: key)
        let bound = params
        try ctx.run { enc in
            enc.setComputePipelineState(packPSO)
            for (j, s) in specs.enumerated() {
                enc.setBuffer(s.values.mtl, offset: s.values.offset, index: 2 * j)
                let v = s.validity ?? s.values
                enc.setBuffer(v.mtl, offset: v.offset, index: 2 * j + 1)
            }
            let base = 2 * specs.count
            withUnsafeBytes(of: bound) { enc.setBytes($0.baseAddress!, length: $0.count, index: base) }
            Dispatch.setLength(enc, n, nil, index: base + 1)
            enc.setBuffer(packed.mtl, offset: packed.offset, index: base + 2)
            enc.setBuffer(occBuf.mtl, offset: occBuf.offset, index: base + 3)
            Dispatch.dispatch1D(enc, packPSO, count: n)
        }
        for s in specs {
            ctx.retainUntilFlush(s.values)
            if let v = s.validity { ctx.retainUntilFlush(v) }
        }

        let occ = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: occBuf, context: ctx)
        let cum = try occ.cumulativeSum()
        try ctx.syncPoint()
        let cardinality = withExtendedLifetime(cum) { Int(cum.valuePointer[total - 1]) }

        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let rankPSO = try ctx.pipeline(source: packSource(shape), function: "gkp_rank",
                                       cacheKey: key + "/rank")
        try ctx.run { enc in
            enc.setComputePipelineState(rankPSO)
            enc.setBuffer(packed.mtl, offset: packed.offset, index: 0)
            enc.setBuffer(cum.values.mtl, offset: cum.values.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, rankPSO, count: (n + 3) / 4)
        }
        ctx.retainUntilFlush(packed); ctx.retainUntilFlush(cum)
        try ctx.syncPoint()
        return (MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx),
                cardinality)
    }

    /// One key column reduced to a `DenseKeyColumn`, or nil when it is not an integer-like column with
    /// a known range. Booleans, temporal values and dictionary codes are integers underneath and are
    /// unwrapped here, exactly as `GroupByKeys.rangeIds` unwraps them.
    static func column(_ c: AnyMetalArray, _ ctx: MetalContext) throws -> DenseKeyColumn? {
        let integers: AnyMetalArray
        switch c {
        case .boolean(let b): integers = .uint8(try b.toUInt8Array())
        case .temporal(let t):
            switch t.storage { case .int32(let a): integers = .int32(a); case .int64(let a): integers = .int64(a) }
        case .dictionary(let codes, _): integers = .int32(codes)
        default: integers = c
        }
        func of<T: ArrowPrimitive>(_ a: MetalArray<T>) throws -> DenseKeyColumn? {
            guard !T.isFloatingPoint, let mm = try a.minMax() else { return nil }
            let lo: Int64, hi: Int64
            if T.minValue < 0 as T {
                (lo, hi) = (mm.min.asInt64, mm.max.asInt64)
            } else {
                let top = mm.max.asUInt64
                guard top <= UInt64(Int64.max) else { return nil }
                (lo, hi) = (Int64(mm.min.asUInt64), Int64(top))
            }
            let range = hi &- lo
            guard range >= 0, range < Int64(Int.max) - 2 else { return nil }
            // One slot per value, plus one for the nulls: the null slot is above every value, which is
            // where the pairwise fold's `densify` puts the null group too.
            let span = Int(range) + 1 + (a.nullCount > 0 ? 1 : 0)
            return DenseKeyColumn(values: a.values, validity: a.nullCount > 0 ? a.validity : nil,
                                  mslType: T.mslType, lo: lo, span: span)
        }
        switch integers {
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

    /// MSL for one shape of packed key: the column types and which of them carry a validity bitmap.
    /// Both kernels live in one source so a shape compiles once.
    static func packSource(_ shape: [(msl: String, hasValidity: Bool)]) -> String {
        var args: [String] = []
        for (j, s) in shape.enumerated() {
            args.append("device const \(s.msl)* c\(j) [[buffer(\(2 * j))]]")
            args.append("device const uchar* v\(j) [[buffer(\(2 * j + 1))]]")
        }
        let base = 2 * shape.count
        args.append("constant GkPackParams& prm [[buffer(\(base))]]")
        args.append("device const uint* nPtr [[buffer(\(base + 1))]]")
        args.append("device int* packed [[buffer(\(base + 2))]]")
        args.append("device int* occ [[buffer(\(base + 3))]]")
        args.append("uint i [[thread_position_in_grid]]")
        var body = ""
        for (j, s) in shape.enumerated() {
            let value = "(int)((long)c\(j)[i] - prm.lo[\(j)])"
            let slot = s.hasValidity ? "(bit_get(v\(j), i) ? \(value) : prm.nullSlot[\(j)])" : value
            body += "        key += \(slot) * prm.stride[\(j)];\n"
        }
        return KernelSource.prelude + """

        struct GkPackParams { long lo[4]; int stride[4]; int nullSlot[4]; };

        // One row per thread: the packed key, and the mark that says the key occurs. Two threads that
        // land on the same key write the same 1, so the unsynchronised store is as safe as `gk_occupy`'s.
        kernel void gkp_pack(\(args.joined(separator: ",\n                             "))) {
            if (i >= *nPtr) return;
            int key = 0;
        \(body)    packed[i] = key;
            occ[key] = 1;
        }

        // `cum` is the INCLUSIVE scan of the marks, so `cum[key] - 1` is the dense id of that key among
        // the ones that occur. Four rows per thread, vector in and vector out; the gathered scan reads
        // are random either way. The tail past the last whole group of four is scalar.
        kernel void gkp_rank(device const int* packed [[buffer(0)]], device const int* cum [[buffer(1)]],
                             device const uint* nPtr [[buffer(2)]], device int* out [[buffer(3)]],
                             uint t [[thread_position_in_grid]]) {
            uint n = *nPtr;
            uint i = t * 4u;
            if (i + 4u <= n) {
                int4 k = *(device const int4*)(packed + i);
                *(device int4*)(out + i) = int4(cum[k.x], cum[k.y], cum[k.z], cum[k.w]) - 1;
            } else {
                for (; i < n; i++) out[i] = cum[packed[i]] - 1;
            }
        }
        """
    }
}
