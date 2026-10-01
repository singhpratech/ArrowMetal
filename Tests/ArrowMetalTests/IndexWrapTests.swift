import XCTest
import Metal
@testable import ArrowMetal

/// Kernels whose 32-bit index arithmetic reached 2^32 inside an array of at most 2^32 - 1 elements.
///
/// A grid-stride loop `i += gridSize` cannot step past `n` once `n > 2^32 - gridSize`: the index wraps to
/// a small value that is again below `n`, and the loop never ends. A block end `start + chunk` wraps on the
/// last block and that block reads nothing. A word count `(n + 31) / 32` is 0 for `n > 2^32 - 32`. Block
/// loops now walk by offset inside the block, grid-stride and segment loops stop at `n`, and the word count
/// is `(n >> 5) + carry` (docs/FINDINGS.md round 18).
///
/// With `ARROWMETAL_BIG_TESTS=1` every fixed family runs once over 2^32 - 1 elements against a host
/// reference. The inputs are periodic: one period of memory (1 GiB per column) mapped back to back up to
/// the array's length, so a 32 GiB Float64 column needs 1 GiB of memory. Outputs are real memory; the
/// largest (an Int32 per row, 16 GiB) is the group order's and the partition's.
final class IndexWrapTests: XCTestCase {

    /// 2^32 - 1, or `ARROWMETAL_WRAP_ROWS` to try the same checks at a smaller size.
    static let n = ProcessInfo.processInfo.environment["ARROWMETAL_WRAP_ROWS"].flatMap { Int($0) } ?? (1 << 32) - 1

    private func requireBig() throws {
        try requireRealGPU()
        guard ProcessInfo.processInfo.environment["ARROWMETAL_BIG_TESTS"] == "1" else {
            throw XCTSkip("set ARROWMETAL_BIG_TESTS=1 to run the kernels over 2^32 - 1 elements")
        }
    }

    // MARK: - periodic inputs

    /// `copies` mappings of one anonymous region back to back: `period` bytes of memory seen as
    /// `period * copies` bytes, every copy the same pages.
    final class PeriodicMemory: @unchecked Sendable {
        let base: UnsafeMutableRawPointer
        let length: Int
        let period: Int

        init(period: Int, covering bytes: Int) throws {
            let page = Int(vm_page_size)
            precondition(period % page == 0, "the period must be whole pages")
            let copies = (bytes + period - 1) / period
            let total = copies * period
            var src: mach_vm_address_t = 0
            guard mach_vm_allocate(mach_task_self_, &src, mach_vm_size_t(period), VM_FLAGS_ANYWHERE) == KERN_SUCCESS else {
                throw XCTSkip("mach_vm_allocate of \(period) bytes failed")
            }
            var dst: mach_vm_address_t = 0
            guard mach_vm_allocate(mach_task_self_, &dst, mach_vm_size_t(total), VM_FLAGS_ANYWHERE) == KERN_SUCCESS else {
                mach_vm_deallocate(mach_task_self_, src, mach_vm_size_t(period))
                throw XCTSkip("mach_vm_allocate of \(total) bytes failed")
            }
            // Touch the source so the region exists before it is shared.
            memset(UnsafeMutableRawPointer(bitPattern: UInt(src))!, 0, period)
            for c in 0..<copies {
                var at = dst + mach_vm_address_t(c * period)
                var cur: vm_prot_t = 0, maxp: vm_prot_t = 0
                let kr = mach_vm_remap(mach_task_self_, &at, mach_vm_size_t(period), 0,
                                       VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self_, src, 0,
                                       &cur, &maxp, VM_INHERIT_NONE)
                guard kr == KERN_SUCCESS else {
                    mach_vm_deallocate(mach_task_self_, dst, mach_vm_size_t(total))
                    mach_vm_deallocate(mach_task_self_, src, mach_vm_size_t(period))
                    throw XCTSkip("mach_vm_remap failed (\(kr))")
                }
            }
            mach_vm_deallocate(mach_task_self_, src, mach_vm_size_t(period))
            base = UnsafeMutableRawPointer(bitPattern: UInt(dst))!
            length = total
            self.period = period
        }

        deinit { mach_vm_deallocate(mach_task_self_, mach_vm_address_t(UInt(bitPattern: base)), mach_vm_size_t(length)) }

        func buffer(_ ctx: MetalContext, byteCount: Int) throws -> MetalArrowBuffer {
            guard let b = ctx.device.makeBuffer(bytesNoCopy: base, length: length, options: [.storageModeShared],
                                                deallocator: nil) else {
                throw XCTSkip("makeBuffer(bytesNoCopy:) refused \(length) bytes")
            }
            return MetalArrowBuffer(mtl: b, byteCount: byteCount, keepAlive: self)
        }
    }

    /// A column of `n` elements of `T` whose element `i` is `f(i % periodElements)`.
    static func periodic<T: ArrowPrimitive>(_: T.Type, n: Int = n, periodBytes: Int = 1 << 30,
                                            validity: MetalArrowBuffer? = nil, nullCount: Int = 0,
                                            _ f: @escaping @Sendable (Int) -> T) throws -> (MetalArray<T>, Int) {
        let ctx = MetalContext.shared
        let pe = periodBytes / MemoryLayout<T>.stride
        let mem = try PeriodicMemory(period: periodBytes, covering: n * MemoryLayout<T>.stride + 16_384)
        let p = mem.base.bindMemory(to: T.self, capacity: pe)
        let pp = UnsafeSendable(p)
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (pe / 64), hi = c == 63 ? pe : (c + 1) * (pe / 64)
            for i in lo..<hi { pp.value[i] = f(i) }
        }
        let buf = try mem.buffer(ctx, byteCount: n * MemoryLayout<T>.stride)
        return (MetalArray<T>(length: n, nullCount: nullCount, validity: validity, values: buf, context: ctx), pe)
    }

    /// A bitmap of `n` bits whose bit `i` is `f(i % periodBits)`.
    static func periodicBits(n: Int = n, periodBits: Int, _ f: @escaping @Sendable (Int) -> Bool) throws -> MetalArrowBuffer {
        let periodBytes = periodBits / 8
        let mem = try PeriodicMemory(period: periodBytes, covering: (n + 7) / 8 + 16_384)
        let p = UnsafeSendable(mem.base.bindMemory(to: UInt8.self, capacity: periodBytes))
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (periodBytes / 64), hi = c == 63 ? periodBytes : (c + 1) * (periodBytes / 64)
            for b in lo..<hi {
                var byte: UInt8 = 0
                for k in 0..<8 where f(b * 8 + k) { byte |= 1 << k }
                p.value[b] = byte
            }
        }
        return try mem.buffer(MetalContext.shared, byteCount: (n + 7) / 8)
    }

    /// Sums `f` over `0..<count` in parallel.
    static func hostSum<A: AdditiveArithmetic & Sendable>(_ count: Int, _ zero: A, _ f: @escaping @Sendable (Int) -> A) -> A {
        let parts = Partials(zero, 64)
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (count / 64), hi = c == 63 ? count : (c + 1) * (count / 64)
            var acc = zero
            for i in lo..<hi { acc += f(i) }
            parts.set(c, acc)
        }
        return parts.values.reduce(zero, +)
    }

    /// Element patterns: hash-like, so neighbouring rows differ and every key appears in every stretch. Both
    /// repeat every `P` rows, the period of a 1 GiB column of 8-byte values, so every column of every
    /// width repeats together every `P` rows.
    static let P = 1 << 27
    static func key8(_ i: Int) -> UInt8 { UInt8(truncatingIfNeeded: (UInt32(truncatingIfNeeded: i & (P - 1)) &* 2_654_435_761) >> 24) }
    static func val8(_ i: Int) -> UInt8 { UInt8(truncatingIfNeeded: (UInt32(truncatingIfNeeded: i & (P - 1)) &* 0x9E37_79B1) >> 13) }

    /// Per-key statistics of a value over rows `0..<count`, keys `key8`.
    struct GroupStats {
        var cnt = [Int](repeating: 0, count: 256), sum = [Int](repeating: 0, count: 256)
        var sum2 = [Int](repeating: 0, count: 256), sum3 = [Int](repeating: 0, count: 256)
        var min = [Int](repeating: Int.max, count: 256), max = [Int](repeating: Int.min, count: 256)
        mutating func merge(_ o: GroupStats, times q: Int = 1) {
            for k in 0..<256 {
                cnt[k] += q * o.cnt[k]; sum[k] += q * o.sum[k]; sum2[k] += q * o.sum2[k]; sum3[k] += q * o.sum3[k]
                if q > 0 { min[k] = Swift.min(min[k], o.min[k]); max[k] = Swift.max(max[k], o.max[k]) }
            }
        }
    }

    static func groupStats(_ count: Int, _ value: @escaping @Sendable (Int) -> Int) -> GroupStats {
        let lock = NSLock()
        var all = GroupStats()
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (count / 64), hi = c == 63 ? count : (c + 1) * (count / 64)
            var g = GroupStats()
            for i in lo..<hi {
                let k = Int(key8(i)), v = value(i)
                g.cnt[k] += 1; g.sum[k] += v; g.sum2[k] += v * v; g.sum3[k] += v * v * v
                if v < g.min[k] { g.min[k] = v }
                if v > g.max[k] { g.max[k] = v }
            }
            lock.lock(); all.merge(g); lock.unlock()
        }
        return all
    }

    /// Group statistics over `n` rows whose keys and values both repeat every `period` rows.
    static func groupStatsPeriodic(n: Int, period: Int, _ value: @escaping @Sendable (Int) -> Int) -> GroupStats {
        var out = GroupStats()
        out.merge(groupStats(period, value), times: n / period)
        out.merge(groupStats(n % period, value))
        return out
    }

    // MARK: - whole-column reductions and aggregates

    /// `sum`, `min`, `max` (`reduce_*`, a vector loop and a tail), `product`, `min_max` and `variance`
    /// (`agg_*`) and `skew` (`gx_moment34`), without and with a validity bitmap: grid-stride loops over
    /// 2^19 threads, which never ended for `n > 2^32 - 2^19` (and from `n > 2^32 - 2^21` for the vector loop).
    func testReductionsAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let (a, pe) = try Self.periodic(UInt8.self, Self.val8)
        let q = n / pe, r = n % pe
        func periodicSum(_ f: @escaping @Sendable (Int) -> Int) -> Int { q * Self.hostSum(pe, 0, f) + Self.hostSum(r, 0, f) }
        let total = periodicSum { Int(Self.val8($0)) }
        let start = Date()
        XCTAssertEqual(try a.sum(), .uint(UInt64(total)), "sum")
        XCTAssertEqual(try a.min(), 0)
        XCTAssertEqual(try a.max(), 255)
        let mm = try a.minMax()
        XCTAssertEqual(mm?.min, 0); XCTAssertEqual(mm?.max, 255)
        let mean = Double(total) / Double(n)
        let ss = periodicSum { let d = Double(Self.val8($0)) - 127.5; return Int(d * d * 4) }   // exact: (2v - 255)^2
        let varRef = Double(ss) / 4 / Double(n) - (mean - 127.5) * (mean - 127.5)
        let v = try XCTUnwrap(try a.variance())
        XCTAssertEqual(v, varRef, accuracy: varRef * 1e-6, "variance")
        XCTAssertNotNil(try a.skew(), "skew")
        print("IndexWrapTests: sum, min, max, min_max, variance, skew over \(n) elements, \(String(format: "%.2f", Date().timeIntervalSince(start))) s")

        // Odd values, so the wrapping product never collapses to zero and a row read twice changes it.
        let (odd, _) = try Self.periodic(UInt8.self) { Self.val8($0) | 1 }
        func periodicProduct() -> UInt64 {
            let parts = Partials(UInt64(1), 64)
            func prod(_ count: Int) -> UInt64 {
                DispatchQueue.concurrentPerform(iterations: 64) { c in
                    let lo = c * (count / 64), hi = c == 63 ? count : (c + 1) * (count / 64)
                    var acc: UInt64 = 1
                    for i in lo..<hi { acc &*= UInt64(Self.val8(i) | 1) }
                    parts.set(c, acc)
                }
                return parts.values.reduce(1, &*)
            }
            let whole = prod(pe), rest = prod(r)
            var acc: UInt64 = 1
            for _ in 0..<q { acc &*= whole }
            return acc &* rest
        }
        XCTAssertEqual(try odd.product(), .uint(periodicProduct()), "product")

        // A validity bitmap with its own period of the same 2^30 rows: every 7th row null.
        let bits = try Self.periodicBits(periodBits: pe) { $0 % 7 != 3 }
        let validRows = q * Self.hostSum(pe, 0) { $0 % 7 != 3 ? 1 : 0 } + Self.hostSum(r, 0) { $0 % 7 != 3 ? 1 : 0 }
        let nv = MetalArray<UInt8>(length: n, nullCount: n - validRows, validity: bits, values: a.values, context: a.context)
        let validTotal = periodicSum { $0 % 7 != 3 ? Int(Self.val8($0)) : 0 }
        XCTAssertEqual(try nv.sum(), .uint(UInt64(validTotal)), "sum with nulls")
        XCTAssertNotNil(try nv.variance(), "variance with nulls")
    }

    /// The fused expression reduce (`am_reduce`) counts its 32-row words as `(n + 31) / 32`, which was 0
    /// for `n > 2^32 - 32`.
    func testExpressionReduceAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let (a, pe) = try Self.periodic(UInt8.self, Self.val8)
        let q = n / pe, r = n % pe
        let total = q * Self.hostSum(pe, 0) { Int(Self.val8($0)) } + Self.hostSum(r, 0) { Int(Self.val8($0)) }
        let big = q * Self.hostSum(pe, 0) { Self.val8($0) > 200 ? 1 : 0 } + Self.hostSum(r, 0) { Self.val8($0) > 200 ? 1 : 0 }
        let b = try MetalRecordBatch(names: ["v"], columns: [.uint8(a)])
        let out = try LazyFrame(PlanSource(name: "t", batch: b))
            .aggregate([ExprAggregate(.sum, col("v").cast(to: .int64), name: "s"),
                        ExprAggregate(.count, col("v"), name: "c"),
                        ExprAggregate(.sum, (col("v") > 200).cast(to: .int64), name: "big")]).collect()
        guard case .int64(let s)? = out["s"], case .int64(let c)? = out["c"], case .int64(let g)? = out["big"] else {
            return XCTFail("aggregate columns: \(out.names)")
        }
        XCTAssertEqual(s.toArray().first ?? nil, Int64(total), "sum")
        XCTAssertEqual(c.toArray().first ?? nil, Int64(n), "count")
        XCTAssertEqual(g.toArray().first ?? nil, Int64(big), "sum of a predicate")
    }

    /// Bitmap word kernels (`bitmap_*`, `lx_xor`, the Kleene forms, `st_fill_words`) and the filter's block
    /// scan compare a word index with `(n + 31) / 32`, which was 0 for `n > 2^32 - 32`: no word was written.
    func testBitmapsAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let pb = 1 << 30
        let x = MetalBooleanArray(length: n, nullCount: 0, validity: nil,
                                  values: try Self.periodicBits(periodBits: pb) { Self.val8($0) & 1 == 1 }, context: .shared)
        let y = MetalBooleanArray(length: n, nullCount: 0, validity: nil,
                                  values: try Self.periodicBits(periodBits: pb) { Self.key8($0) & 2 == 2 }, context: .shared)
        func check(_ got: MetalBooleanArray, _ what: String, _ f: @escaping @Sendable (Bool, Bool) -> Bool) {
            let w = UnsafeSendable(got.values.typed(UInt8.self))
            let bad = FoldCheckCounter()
            DispatchQueue.concurrentPerform(iterations: 64) { c in
                let lo = c * (n / 64), hi = c == 63 ? n : (c + 1) * (n / 64)
                var local = 0
                for i in lo..<hi {
                    let j = i % pb
                    let want = f(Self.val8(j) & 1 == 1, Self.key8(j) & 2 == 2)
                    if ((w.value[i >> 3] >> (i & 7)) & 1 == 1) != want { local += 1 }
                }
                bad.add(local)
            }
            XCTAssertEqual(bad.value, 0, "\(what): wrong bits out of \(n)")
        }
        check(try x.and(y), "and") { $0 && $1 }
        check(try x.or(y), "or") { $0 || $1 }
        check(try x.xor(y), "xor") { $0 != $1 }
        check(try x.not(), "not") { a, _ in !a }

        // The filter's block scan: keep the rows whose value is odd.
        let (a, pe) = try Self.periodic(UInt8.self, Self.val8)
        let kept = try a.filter(x)
        let q = n / pe, r = n % pe
        let want = q * Self.hostSum(pe, 0) { Int(Self.val8($0) & 1) } + Self.hostSum(r, 0) { Int(Self.val8($0) & 1) }
        XCTAssertEqual(kept.length, want, "filtered rows")
        let last = (0..<n).reversed().first { Self.val8($0 % pe) & 1 == 1 }!
        XCTAssertEqual(kept.valuePointer[kept.length - 1], Self.val8(last % pe), "last kept row")
    }

    // MARK: - group-by

    /// The atomic group-by (`GroupBySource`) and the per-key extrema (`GroupByExtremaSource`) split the
    /// rows into at most 1,024 or 4,096 blocks of `chunk` rows; the last block's end `start + chunk` wrapped
    /// from `n = 2^32 - 1,023` and that block counted nothing.
    func testGroupByAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let (keys, _) = try Self.periodic(UInt32.self) { UInt32(Self.key8($0)) }
        let (vals, _) = try Self.periodic(UInt8.self, Self.val8)
        let ref = Self.groupStatsPeriodic(n: n, period: Self.P) { Int(Self.val8($0)) }
        let g = try keys.groupBy(keyCount: 256)
        XCTAssertEqual(try g.count().toArray().map { Int($0 ?? -1) }, ref.cnt, "count")
        XCTAssertEqual(try g.sum(vals).toArray().map { Int($0 ?? -1) }, ref.sum, "sum")
        XCTAssertEqual(try g.min(vals).toArray().map { Int($0 ?? 0) }, ref.min, "min")
        XCTAssertEqual(try g.max(vals).toArray().map { Int($0 ?? 0) }, ref.max, "max")
        let ex = try g.minMax(vals)
        XCTAssertEqual(ex.min.toArray().map { Int($0 ?? 0) }, ref.min, "extrema min")
        XCTAssertEqual(ex.max.toArray().map { Int($0 ?? 0) }, ref.max, "extrema max")
    }

    /// The exact Float64 group sums (`GroupSumExactSource`), the same blocks over a 32 GiB column.
    func testExactGroupSumAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let (keys, _) = try Self.periodic(UInt32.self) { UInt32(Self.key8($0)) }
        let (vals, _) = try Self.periodic(Double.self) { Double(Self.val8($0)) }
        let ref = Self.groupStatsPeriodic(n: n, period: Self.P) { Int(Self.val8($0)) }
        let want = ref.sum.map(Double.init)      // integers below 2^53: exact in any order
        let g = try keys.groupBy(keyCount: 256)
        XCTAssertEqual(try g.sumDouble(vals).toArray().map { $0 ?? -1 }, want, "exact sum")
    }

    /// The group order (`GroupOrderSource`, the chunked counting sort over 2,048 sub-blocks) and the
    /// per-group loops that walk a segment `t += 256` up to its end (`AggregatesExtraSource`,
    /// `GroupMomentsSource`, `SegmentedSource`): a segment ending past 2^32 - 256 never ended.
    func testGroupSegmentsAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let (keys, _) = try Self.periodic(UInt32.self) { UInt32(Self.key8($0)) }
        let (vals, _) = try Self.periodic(UInt8.self, Self.val8)
        let ref = Self.groupStatsPeriodic(n: n, period: Self.P) { Int(Self.val8($0)) }
        let g = try keys.groupBy(keyCount: 256)
        let start = Date()
        let seg = try g.segments()
        print("IndexWrapTests: group order over \(n) rows, \(String(format: "%.2f", Date().timeIntervalSince(start))) s")
        // Every row lands once, in its key's segment, in row order within the segment.
        let ord = UnsafeSendable(seg.ord.valuePointer)
        let st = seg.segStart.typed(UInt32.self), en = seg.segEnd.typed(UInt32.self)
        XCTAssertEqual((0..<256).map { Int(en[$0]) - Int(st[$0]) }, ref.cnt, "segment sizes")
        let bad = FoldCheckCounter()
        let bounds = (0..<256).map { (Int(st[$0]), Int(en[$0])) }
        DispatchQueue.concurrentPerform(iterations: 256) { k in
            var local = 0, prev = -1
            for t in bounds[k].0..<bounds[k].1 {
                let row = Int(ord.value[t])
                if Int(Self.key8(row)) != k || row <= prev { local += 1 }
                prev = row
            }
            bad.add(local)
        }
        XCTAssertEqual(bad.value, 0, "rows out of place in the group order")

        // `gx_seg_minmax`, `gx_seg_moment34` (AggregatesExtraSource) and `gm_*` (GroupMomentsSource).
        let mm = try g.minMaxSegmented(vals, segments: seg)
        XCTAssertEqual(mm.min.toArray().map { $0.map(Int.init) ?? -1 }, ref.min, "segmented min")
        XCTAssertEqual(mm.max.toArray().map { $0.map(Int.init) ?? -1 }, ref.max, "segmented max")
        let variance = try g.varianceDouble(vals, ddof: 0).toArray()
        let skew = try g.skew(vals, segments: seg).toArray()
        for k in 0..<256 {
            let c = Double(ref.cnt[k]), m1 = Double(ref.sum[k]) / c
            let m2 = Double(ref.sum2[k]) / c - m1 * m1
            let m3 = Double(ref.sum3[k]) / c - 3 * m1 * Double(ref.sum2[k]) / c + 2 * m1 * m1 * m1
            XCTAssertEqual(variance[k] ?? -1, m2, accuracy: m2 * 1e-9, "variance of key \(k)")
            XCTAssertEqual(skew[k] ?? 99, m3 / (m2 * m2.squareRoot()), accuracy: 1e-6, "skew of key \(k)")
        }
        // `seg_reduce` (SegmentedSource) over a 32 GiB Int64 column.
        let (wide, _) = try Self.periodic(Int64.self) { Int64(Self.val8($0)) }
        XCTAssertEqual(try g.min64(wide, segments: seg).toArray().map { Int($0 ?? -1) }, ref.min, "segmented min64")
        XCTAssertEqual(try g.max64(wide, segments: seg).toArray().map { Int($0 ?? -1) }, ref.max, "segmented max64")
    }

    // MARK: - selection

    /// Top-k (`RadixSelectSource`, which `topK` takes at this size, and `TopKSource` called directly) and
    /// the partition (`PartitionNth`) over 2^32 - 1 rows of which only the last 2^22 are valid, so the
    /// answer lives in the last blocks: the blocks whose end wrapped.
    func testSelectionAtTheLimit() throws {
        try requireBig()
        let n = Self.n, tail = 1 << 22, from = n - tail
        let ctx = MetalContext.shared
        let bits = try MetalArrowBuffer.allocate(byteCount: (n + 7) / 8 + 64, zeroed: true, context: ctx)
        let bp = bits.mutableTyped(UInt8.self)
        for i in from..<n { bp[i >> 3] |= 1 << (i & 7) }
        let (raw, pe) = try Self.periodic(UInt32.self) { UInt32(Self.val8($0)) }
        let a = MetalArray<UInt32>(length: n, nullCount: from, validity: bits, values: raw.values, context: ctx)
        // The reference: the valid rows by (value desc, row asc).
        let rows = (from..<n).sorted { (Self.val8($0 % pe), -$0) > (Self.val8($1 % pe), -$1) }
        for k in [10, 5_000] {
            let got = try a.topK(k, largest: true).toRawArray().map { Int($0) }
            XCTAssertEqual(got, Array(rows.prefix(k)), "top \(k)")
        }
        let tk = try XCTUnwrap(try a.topKSelect(10, largest: true)).toRawArray().map { Int($0) }
        XCTAssertEqual(tk, Array(rows.prefix(10)), "top 10 through the per-block selection")
        let small = (from..<n).sorted { (Self.val8($0 % pe), $0) < (Self.val8($1 % pe), $1) }
        for k in [10, 5_000] {
            let got = try a.topK(k, largest: false).toRawArray().map { Int($0) }
            XCTAssertEqual(got, Array(small.prefix(k)), "bottom \(k)")
        }

        // The partition's select (`pn_hist`, a grid-stride loop over 2^18 threads) and its count and
        // scatter (blocks of up to 2^25 rows) over a 16 GiB key column, straight on the keys.
        let (keys, kpe) = try Self.periodic(UInt32.self) { UInt32(Self.val8($0)) << 8 | UInt32(Self.key8($0)) }
        let rank = n / 2
        let owner = NSObject()
        let pivot = try PartitionNth.select(keys, rank: rank, candidates: n, validity: nil, nanKey: nil, owner: owner)
        // The reference: how many keys are below the pivot.
        func below(_ count: Int) -> Int { Self.hostSum(count, 0) { (UInt32(Self.val8($0)) << 8 | UInt32(Self.key8($0))) < pivot ? 1 : 0 } }
        let less = (n / kpe) * below(kpe) + below(n % kpe)
        let lessEq = less + (n / kpe) * Self.hostSum(kpe, 0) { (UInt32(Self.val8($0)) << 8 | UInt32(Self.key8($0))) == pivot ? 1 : 0 }
            + Self.hostSum(n % kpe, 0) { (UInt32(Self.val8($0)) << 8 | UInt32(Self.key8($0))) == pivot ? 1 : 0 }
        XCTAssertTrue(less <= rank && rank < lessEq, "the select's key is the rank-th: \(less) <= \(rank) < \(lessEq)")
        let part = try PartitionNth.partition(keys, validity: nil, threshold: pivot, nanKey: nil, nullPlacement: .atEnd, owner: owner)
        let out = UnsafeSendable(part.valuePointer)
        let wrong = FoldCheckCounter()
        let seen = FoldCheckCounter()
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (n / 64), hi = c == 63 ? n : (c + 1) * (n / 64)
            var local = 0, sum = 0, prev = -1
            for t in lo..<hi {
                let row = Int(out.value[t])
                let key = UInt32(Self.val8(row % kpe)) << 8 | UInt32(Self.key8(row % kpe))
                let bucket = t < less ? 0 : (t < lessEq ? 1 : 2)
                if (bucket == 0 && key >= pivot) || (bucket == 1 && key != pivot) || (bucket == 2 && key <= pivot) { local += 1 }
                // Stable: row numbers rise inside a bucket.
                if t != lo && t != less && t != lessEq && row <= prev { local += 1 }
                prev = row
                sum &+= row
            }
            wrong.add(local); seen.add(sum)
        }
        XCTAssertEqual(wrong.value, 0, "rows in the wrong bucket or out of order")
        XCTAssertEqual(seen.value, n % 2 == 0 ? n / 2 * (n - 1) : (n - 1) / 2 * n, "every row exactly once (sum of the row numbers)")
    }

    // MARK: - the sort's block histogram

    /// `radix_histogram` (`SortSource`) over blocks of up to 2^25 rows, straight on the GPU over a periodic
    /// 16 GiB key column: the last block's end wrapped from `n = 2^32 - 32,767`.
    func testSortHistogramAtTheLimit() throws {
        try requireBig()
        let n = Self.n
        let ctx = MetalContext.shared
        let (keys, pe) = try Self.periodic(UInt32.self) { UInt32(Self.val8($0)) }
        var e = Swift.max(4096, ((n + 127) / 128 + 255) / 256 * 256)
        while e > 256 && (n + e - 1) / e < 64 { e >>= 1 }
        let blocks = (n + e - 1) / e
        let pso = try Dispatch.pipeline(ctx, family: "radix", source: SortSource.source(K: "uint"),
                                        function: "radix_histogram", type: "uint")
        let counts = try MetalArrowBuffer.allocate(byteCount: 256 * blocks * 4, zeroed: true, context: ctx)
        let spans = try MetalArrowBuffer.allocate(byteCount: blocks * 8, zeroed: true, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(keys.values.mtl, offset: keys.values.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.setUInt(enc, 0, index: 2)
            Dispatch.setUInt(enc, e, index: 3)
            Dispatch.setUInt(enc, blocks, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            enc.setBuffer(spans.mtl, offset: 0, index: 6)
            enc.setBuffer(spans.mtl, offset: blocks * 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try ctx.syncPoint()
        let c = counts.typed(UInt32.self)
        var got = [Int](repeating: 0, count: 256)
        for d in 0..<256 { for b in 0..<blocks { got[d] += Int(c[d * blocks + b]) } }
        let q = n / pe, r = n % pe
        var want = [Int](repeating: 0, count: 256)
        let whole = Self.countByValue(pe), rest = Self.countByValue(r)
        for d in 0..<256 { want[d] = q * whole[d] + rest[d] }
        XCTAssertEqual(got, want, "digit counts")
        XCTAssertEqual((0..<256).reduce(0) { $0 + Int(c[$1 * blocks + blocks - 1]) }, n - (blocks - 1) * e, "the last block's rows")
    }

    /// Rows per value of `val8` over `0..<count`.
    static func countByValue(_ count: Int) -> [Int] {
        let lock = NSLock()
        var cnt = [Int](repeating: 0, count: 256)
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (count / 64), hi = c == 63 ? count : (c + 1) * (count / 64)
            var lc = [Int](repeating: 0, count: 256)
            for i in lo..<hi { lc[Int(val8(i))] += 1 }
            lock.lock(); for k in 0..<256 { cnt[k] += lc[k] }; lock.unlock()
        }
        return cnt
    }
}

/// A pointer the concurrent fill and check loops share.
struct UnsafeSendable<P>: @unchecked Sendable {
    let value: P
    init(_ v: P) { value = v }
}

/// Per-chunk partial results of a concurrent loop.
final class Partials<A>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: [A]
    init(_ zero: A, _ count: Int) { v = [A](repeating: zero, count: count) }
    func set(_ i: Int, _ x: A) { lock.lock(); v[i] = x; lock.unlock() }
    var values: [A] { lock.lock(); defer { lock.unlock() }; return v }
}
