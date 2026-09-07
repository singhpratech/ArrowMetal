import Foundation
import Metal

extension MetalArray {
    /// What one run of the GPU radix sort produced.
    ///
    /// The rows are arranged as the caller's `nullPlacement` asks — `[values][NaNs][nulls]` at the end,
    /// `[nulls][NaNs][values]` at the start — and the three block boundaries are known on the host
    /// whenever the run was analysed (see `argsort`). `keys` holds the sorted keys of the value block
    /// alone: key `j` belongs to output position `valueStart + j`.
    struct SortRun {
        /// Row numbers in the caller's arrangement. The null and NaN blocks are always the partition's
        /// own and always right; the value block is only in sorted order when the sort carried the
        /// payload through its passes, which `orderIsSorted` says.
        var order: MetalArrowBuffer?
        var orderIsSorted: Bool
        /// Sorted keys of the value block.
        var keys: MetalArrowBuffer
        /// Whether the block boundaries and `flags` below were read back (only when `analyse` ran).
        var measured: Bool
        var valueStart: Int
        var valueCount: Int
        var nullStart: Int
        var nullCount: Int
        /// `KEY_FLAG_NEGZERO` / `KEY_FLAG_NAN`: whether the column holds a value the key map is not
        /// injective on. Zero for every integer type.
        var flags: UInt32
    }

    /// The order-preserving key of a NaN, which every NaN shares (`SortSource.key_from_f32/f64`).
    static func nanKey(wide: Bool, descending: Bool) -> UInt64 {
        if descending { return wide ? UInt64.max : UInt64(UInt32.max) }
        return wide ? 0xFFF0_0000_0000_0001 : 0xFF80_0001
    }

    /// The order-preserving key of ±0.0, which -0.0 and +0.0 share.
    static func zeroKey(wide: Bool, descending: Bool) -> UInt64 {
        let k: UInt64 = wide ? 0x8000_0000_0000_0000 : 0x8000_0000
        return descending ? (wide ? ~k : UInt64(UInt32(truncatingIfNeeded: ~k))) : k
    }

    /// Arrow `array_sort_indices`: indices that sort the values, stable.
    ///
    /// `descending` picks the direction and `nullPlacement` decides whether the null rows sit after every
    /// value (Arrow's default) or before every one of them; the two are independent, exactly as in Arrow,
    /// so nulls stay at the chosen end in both directions. NaN sorts after +inf (total order). Runs an
    /// LSD radix sort on the GPU over 8-bit digits: four passes for a 32-bit key, eight for a 64-bit
    /// one, minus any pass whose digit is the same in every row, which is an identity permutation and
    /// is dropped (see below).
    public func argsort(descending: Bool = false,
                        nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Int32> {
        try Dispatch.checkLength(length)
        if length == 0 { return try MetalArray<Int32>([Int32](), context: context) }
        let run = try radixSortRun(descending: descending, nullPlacement: nullPlacement, wantOrder: true)
        return MetalArray<Int32>(length: length, nullCount: 0, validity: nil, values: run.order!, context: context)
    }

    /// Sorted copy (nulls at whichever end `nullPlacement` names, `atEnd` by default).
    ///
    /// Not `take(argsort())` any more when the key map can be undone. The radix sort already holds the
    /// sorted *keys*, and the map from a value to its key is a bijection except on -0.0 (which shares
    /// +0.0's key, so that the two tie) and on NaN (every NaN payload shares one key). A column with
    /// neither — which the key kernel reports, having read the values anyway — therefore needs no gather
    /// at all: inverting the sorted keys is a sequential read and a sequential write where the gather was
    /// a random 8-byte one, and the sort can drop the row-number payload it only carried for the gather,
    /// which takes each of its passes from 24 bytes an element to 16.
    ///
    /// When the column *does* hold one of them the answer is still built from the keys, and only the
    /// output positions the map cannot reconstruct — the run of zeros, the run of NaNs, the null block —
    /// are copied through the sorted row numbers. Those runs are contiguous (they share a key) and are
    /// found by a binary search over the sorted keys. If between them they cover more than half the
    /// output, the whole thing is gathered instead, which is what `take` did.
    public func sorted(descending: Bool = false,
                       nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<T> {
        if let out = try sortedByInvertingKeys(descending: descending, nullPlacement: nullPlacement) { return out }
        return try take(try argsort(descending: descending, nullPlacement: nullPlacement))
    }

    /// Indices of the k smallest (or largest) values, in the same order `argsort` would put them.
    ///
    /// Three implementations, all producing the same total order (key, then row index):
    ///
    /// - **Radix select** (`Kernels/RadixSelect.swift`) for any k: a digit histogram finds the bin holding
    ///   the k-th key, then one compaction pass keeps only the rows that can still be in the answer. Two
    ///   passes over the column, and a sort of roughly `k + n/256` rows.
    /// - **Per-threadgroup selection** (`Kernels/TopK.swift`) for k up to 1024 on small inputs: one dispatch,
    ///   each threadgroup keeping the best k of its own block. It has no mid-flight readback, so it wins
    ///   where latency rather than bandwidth decides.
    /// - The **full argsort** for everything else: types with no key mapping, k close to n, and the case
    ///   where fewer than k rows are non-null, which needs the null rows placed.
    public func topK(_ k: Int, largest: Bool = true) throws -> MetalArray<Int32> {
        guard k > 0 else { return try MetalArray<Int32>([Int32](), context: context) }
        if TopK.preferRadixSelect(n: length, k: k) {
            if let r = try topKRadixSelect(k, largest: largest) { return r }
            if let s = try topKSelect(k, largest: largest) { return s }
        } else {
            if let s = try topKSelect(k, largest: largest) { return s }
            if let r = try topKRadixSelect(k, largest: largest) { return r }
        }
        let idx = try argsort(descending: largest)
        return try idx.slice(offset: 0, length: Swift.min(k, idx.length))
    }

    // MARK: - the sort itself

    /// Which inverse the `unkey` kernel has to apply, or nil for a type with no direct key map (the
    /// narrow integers, which widen to int32 and so cannot be read back as themselves).
    private static var unkeyMode: UInt32? {
        switch T.self {
        case is Int32.Type, is Int64.Type: return 0
        case is UInt32.Type, is UInt64.Type: return 1
        case is Float.Type, is Double.Type: return 2
        default: return nil
        }
    }

    /// Runs the radix sort. `wantOrder` asks for the row numbers; without it the sort carries no payload
    /// unless the values themselves cannot be rebuilt from the keys.
    ///
    /// The null rows never enter the sort. A stable three-way partition (`part_count`/`part_scan`/
    /// `part_scatter`) puts them, the NaNs when `.atStart` wants them separated, and the values into
    /// their blocks first, compacting the value block's keys as it goes; the sort then runs over the
    /// value block alone. That is both the correct null order for free — the partition is stable, so the
    /// nulls keep their input order, which is what Arrow asks for — and one fewer pass of work per radix
    /// round. It replaces a host-side pass over the whole sorted index array, which cost three times the
    /// sort itself on a 10% null column.
    func radixSortRun(descending: Bool, nullPlacement: NullPlacement, wantOrder: Bool,
                      keysWanted: Bool = false) throws -> SortRun {
        let ctx = context
        let n = length
        let wide = T.byteWidth == 8
        let keyType = wide ? "ulong" : "uint"
        let bits = SortSource.digitBits
        let radix = 1 << bits
        let src = SortSource.source(K: keyType)
        func p(_ f: String) throws -> MTLComputePipelineState {
            try ctx.pipeline(source: src, function: f, cacheKey: "sort/\(keyType)/\(bits)/\(f)")
        }
        // Narrow types widen to 32-bit keys; the mapping kernel expects the source width, so cast first.
        let mapFn: String
        let source: MetalArrowBuffer
        var tmpKeep: MetalArray<Int32>? = nil
        switch T.self {
        case is Int32.Type: mapFn = "key_from_i32"; source = values
        case is UInt32.Type: mapFn = "key_from_u32"; source = values
        case is Float.Type: mapFn = "key_from_f32"; source = values
        case is Int64.Type: mapFn = "key_from_i64"; source = values
        case is UInt64.Type: mapFn = "key_from_u64"; source = values
        case is Double.Type: mapFn = "key_from_f64"; source = values
        default:
            let widened = try cast(to: Int32.self); tmpKeep = widened; mapFn = "key_from_i32"; source = widened.values
        }
        let kb = wide ? 8 : 4
        // A NaN is placed with the nulls, not with the values, which is Arrow's rule. `.atEnd` needs no
        // separate block for it — the keys already leave every NaN at the tail of the value block, in
        // both directions — so only `.atStart` on a float column asks the partition for a NaN bucket.
        let separateNaN = T.isFloatingPoint && nullPlacement == .atStart
        // Read once, here: on a slice at an offset that is not a multiple of 32 these normalise the
        // buffers, which runs a kernel, and that must not happen with an encoder already open.
        let bitmap = validity
        let usePartition = bitmap != nil || separateNaN
        let partitionBitmap = bitmap ?? values
        let nanK = Self.nanKey(wide: wide, descending: descending)

        // `mapKeys` is what the key map writes; when the partition runs it compacts the value block's
        // keys into `partKeys` and the sort ping-pongs those with `keysB` instead.
        let mapKeys = try MetalArrowBuffer.allocate(byteCount: n * kb, zeroed: false, context: ctx)
        var keysB = try MetalArrowBuffer.allocate(byteCount: n * kb, zeroed: false, context: ctx)
        // Each block reads and writes `radix` entries of the counts table on every pass, at a stride of
        // `blocks`, and the scan below is a *single* threadgroup over `radix * blocks` entries, so both
        // fixed costs fall as blocks get bigger and fewer; against that, too few blocks leaves GPU cores
        // idle. The scatter is memory bound, which pushes the balance a long way towards fewer: at 50M
        // int64 rows argsort measures 37.7 ms at 128 blocks, 49.0 ms at 256 and 55.8 ms at 2048.
        // Small inputs keep the old rule instead — a 20k-element argsort (the size top-k's final
        // ordering lands on) must not run on five threadgroups — so the block halves until there are at
        // least 64 of them, and never holds fewer than 4096 elements.
        func blockPlan(_ rows: Int) -> (elemsPerBlock: Int, blocks: Int) {
            var e = Swift.max(4096, ((rows + 127) / 128 + 255) / 256 * 256)
            while e > 256 && (rows + e - 1) / e < 64 { e >>= 1 }
            return (e, Swift.max(1, (rows + e - 1) / e))
        }
        let (elemsPerBlock, blocks) = blockPlan(n)
        // The passes are re-planned around a shorter value block when the partition takes rows out, and
        // a shorter block can want *more* threadgroups, not fewer (the rule halves the block until there
        // are at least 64 of them). `blockPlan` never asks for more than 128 whatever the row count —
        // 128 exactly, at 520,193 rows — so the tables are sized for that, but only when a re-plan is
        // possible at all; the assignment below falls back to this layout if one ever wanted more.
        let tableBlocks = usePartition ? Swift.max(blocks, 128) : blocks
        let counts = try MetalArrowBuffer.allocate(byteCount: radix * tableBlocks * 4, zeroed: false, context: ctx)
        let spanOr = try MetalArrowBuffer.allocate(byteCount: tableBlocks * kb, zeroed: false, context: ctx)
        let spanAnd = try MetalArrowBuffer.allocate(byteCount: tableBlocks * kb, zeroed: false, context: ctx)
        // Only `sorted()` on a float column has any use for the -0.0 / NaN report, and the kernel does
        // not touch the pointer when `wantFlags` is 0 — but it gets a scratch buffer of its own either
        // way, so that no kernel ever holds two differently-typed device pointers into one allocation.
        let wantFlags = keysWanted && T.isFloatingPoint
        let flagBuf = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: ctx)
        // The partition's own tables: three counters per block and the three bucket totals, which are
        // what the sort's element count comes from when the nulls have been taken out.
        let partCounts = usePartition ? try MetalArrowBuffer.allocate(byteCount: 3 * blocks * 4, zeroed: false, context: ctx) : nil
        let sizes = usePartition ? try MetalArrowBuffer.allocate(byteCount: 3 * 4, zeroed: true, context: ctx) : nil
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let mapPSO = try p(mapFn), histPSO = try p("radix_histogram"), scanPSO = try p("radix_scan")
        let passes = ((wide ? 64 : 32) + bits - 1) / bits
        let blockGrid = MTLSize(width: blocks, height: 1, depth: 1)
        // Element count for every kernel below the partition: the value bucket's size, which lives on the
        // GPU (`sizes[0]`) so that no readback stands between the partition and the passes.
        let countBuf = sizes

        func setKey(_ enc: MTLComputeCommandEncoder, _ v: UInt64, index: Int) {
            if wide { var x = v; enc.setBytes(&x, length: 8, index: index) }
            else { var x = UInt32(truncatingIfNeeded: v); enc.setBytes(&x, length: 4, index: index) }
        }
        func bindPartition(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(mapKeys.mtl, offset: mapKeys.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(partitionBitmap.mtl, offset: partitionBitmap.offset, index: 2)
            Dispatch.setUInt(enc, bitmap != nil ? 1 : 0, index: 3)
            setKey(enc, nanK, index: 4)
            Dispatch.setUInt(enc, separateNaN ? 1 : 0, index: 5)
            // Buckets, category-indexed at four bits each: value, NaN, null.
            Dispatch.setUInt(enc, 0x210, index: 6)
            Dispatch.setUInt(enc, elemsPerBlock, index: 7)
            Dispatch.setUInt(enc, blocks, index: 8)
        }

        func encodeMap(_ enc: MTLComputeCommandEncoder) {
            enc.setComputePipelineState(mapPSO)
            enc.setBuffer(source.mtl, offset: source.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(mapKeys.mtl, offset: mapKeys.offset, index: 2)
            Dispatch.setUInt(enc, descending ? 1 : 0, index: 3)
            enc.setBuffer(flagBuf.mtl, offset: flagBuf.offset, index: 4)
            Dispatch.setUInt(enc, wantFlags ? 1 : 0, index: 5)
            Dispatch.dispatch1D(enc, mapPSO, count: n)
            enc.memoryBarrier(scope: .buffers)
        }
        /// The row numbers the payload starts from, when the partition has not already written them.
        func encodeIota(_ enc: MTLComputeCommandEncoder, _ vals: MetalArrowBuffer) throws {
            let iotaPSO = try p("iota_u32")
            enc.setComputePipelineState(iotaPSO)
            enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.dispatch1D(enc, iotaPSO, count: n)
            enc.memoryBarrier(scope: .buffers)
        }
        /// The stable three-way partition, when there is anything to take out of the sort.
        func encodePartition(_ enc: MTLComputeCommandEncoder, orderA: MetalArrowBuffer,
                             orderB: MetalArrowBuffer, keysOut: MetalArrowBuffer) throws {
            guard let partCounts, let sizes else { return }
            let countPSO = try p("part_count"), pscanPSO = try p("part_scan"), scatPSO = try p("part_scatter")
            enc.setComputePipelineState(countPSO)
            bindPartition(enc)
            enc.setBuffer(partCounts.mtl, offset: partCounts.offset, index: 9)
            enc.dispatchThreadgroups(blockGrid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pscanPSO)
            enc.setBuffer(partCounts.mtl, offset: partCounts.offset, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(sizes.mtl, offset: sizes.offset, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatPSO)
            bindPartition(enc)
            enc.setBuffer(partCounts.mtl, offset: partCounts.offset, index: 9)
            Dispatch.setUInt(enc, 0, index: 10)                       // the value bucket is slot 0
            enc.setBuffer(orderA.mtl, offset: orderA.offset, index: 11)
            enc.setBuffer(orderB.mtl, offset: orderB.offset, index: 12)
            enc.setBuffer(keysOut.mtl, offset: keysOut.offset, index: 13)
            enc.dispatchThreadgroups(blockGrid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
        }
        func encodeHistogram(_ enc: MTLComputeCommandEncoder, keys: MetalArrowBuffer, shift: Int,
                             elemsPerBlock: Int, blocks: Int) {
            enc.setComputePipelineState(histPSO)
            enc.setBuffer(keys.mtl, offset: keys.offset, index: 0)
            Dispatch.setLength(enc, n, countBuf, index: 1)
            Dispatch.setUInt(enc, shift, index: 2)
            Dispatch.setUInt(enc, elemsPerBlock, index: 3)
            Dispatch.setUInt(enc, blocks, index: 4)
            enc.setBuffer(counts.mtl, offset: counts.offset, index: 5)
            enc.setBuffer(spanOr.mtl, offset: spanOr.offset, index: 6)
            enc.setBuffer(spanAnd.mtl, offset: spanAnd.offset, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
        }

        // The keys the sort runs on: the compacted value block when the partition ran, the mapped column
        // itself otherwise. Both ping-pong through `keysB`.
        let partKeys = usePartition ? try MetalArrowBuffer.allocate(byteCount: n * kb, zeroed: false, context: ctx) : nil
        var keysA = partKeys ?? mapKeys

        // A pass whose digit is the same in every row is the identity permutation — the sort is stable,
        // so equal digits keep their order — and can be dropped entirely. The first histogram reports
        // the bitwise OR and AND of the keys along with its counts, and `or ^ and` is exactly the set of
        // bits that differ somewhere in the column, so one readback names every skippable pass at once.
        // A narrow range (sorted-ish data, small integers, a column of one repeated value) can lose most
        // of the passes this way. The readback costs a command-buffer boundary, so inputs small enough
        // for that to matter keep the whole sort in one buffer and run every pass.
        var activePasses = Array(0..<passes)
        let analyse = n >= 1 << 18
        var flags: UInt32 = 0
        var measured = false
        var mValue = n, mNull = 0, mNaN = 0

        // Order buffers are needed by the partition (it is what holds the null and NaN blocks) and by any
        // caller that asked for the permutation. A `sorted()` that turns out to need them after all —
        // because the column holds a -0.0 or a NaN, which the map is not injective on — allocates them
        // below, once the flags have been read.
        var orderA: MetalArrowBuffer? = nil, orderB: MetalArrowBuffer? = nil
        if wantOrder || usePartition {
            orderA = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
            orderB = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        }

        try withExtendedLifetime(tmpKeep) {
            if analyse {
                try ctx.run { enc in
                    encodeMap(enc)
                    if usePartition { try encodePartition(enc, orderA: orderA!, orderB: orderB!, keysOut: partKeys!) }
                    // A caller that asked for the permutation is certainly carrying the payload, so its
                    // row numbers go in this command buffer rather than waiting for the next one.
                    if wantOrder && !usePartition { try encodeIota(enc, orderA!) }
                    encodeHistogram(enc, keys: keysA, shift: 0, elemsPerBlock: elemsPerBlock, blocks: blocks)
                }
                try ctx.syncPoint()
                measured = true
                if wantFlags { flags = withExtendedLifetime(flagBuf) { flagBuf.typed(UInt32.self)[0] } }
                if let sizes {
                    withExtendedLifetime(sizes) {
                        let s = sizes.typed(UInt32.self)
                        mValue = Int(s[0]); mNaN = Int(s[1]); mNull = Int(s[2])
                    }
                }
                var differing: UInt64 = 0
                withExtendedLifetime((spanOr, spanAnd)) {
                    var o: UInt64 = 0, a: UInt64 = .max
                    if wide {
                        let po = spanOr.typed(UInt64.self), pa = spanAnd.typed(UInt64.self)
                        for b in 0..<blocks { o |= po[b]; a &= pa[b] }
                    } else {
                        let po = spanOr.typed(UInt32.self), pa = spanAnd.typed(UInt32.self)
                        for b in 0..<blocks { o |= UInt64(po[b]); a &= UInt64(pa[b]) }
                    }
                    differing = o ^ a
                }
                let mask = UInt64(radix - 1)
                activePasses = (0..<passes).filter { (differing >> UInt64($0 * bits)) & mask != 0 }
            }
        }

        // The payload rides along only when someone will read it: the permutation itself, or a `sorted()`
        // whose value block holds a run the key map cannot be inverted over.
        let carryPayload = wantOrder || (keysWanted && flags != 0)
        if carryPayload && orderA == nil {
            orderA = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
            orderB = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        }
        var valsA = orderA, valsB = orderB
        var resultOrder = orderA
        // The passes run over the value block, not the column: with the nulls taken out, blocks sized
        // for the whole column would leave that fraction of the GPU idle (at 50% nulls, half of it).
        // What it costs is the pass-0 histogram, which can then no longer be the analysis's own.
        var (passElems, passBlocks) = measured && mValue < n ? blockPlan(mValue) : (elemsPerBlock, blocks)
        if passBlocks > tableBlocks { (passElems, passBlocks) = (elemsPerBlock, blocks) }
        let sameLayout = passBlocks == blocks && passElems == elemsPerBlock
        let passGrid = MTLSize(width: passBlocks, height: 1, depth: 1)
        let needIota = carryPayload && !usePartition && !(analyse && wantOrder)
        let needArrange = usePartition && nullPlacement == .atStart
        // A column whose every digit is constant — one repeated value, or one the analysis found nothing
        // varying in — has no pass left to run, and then this second command buffer would be empty. It
        // costs 60-140 µs to commit one, so it is not committed.
        let anyWork = !analyse || needIota || !activePasses.isEmpty || needArrange
        try withExtendedLifetime(tmpKeep) {
            if !anyWork { return }
            try ctx.run { enc in
                if !analyse {
                    encodeMap(enc)
                    if usePartition { try encodePartition(enc, orderA: orderA!, orderB: orderB!, keysOut: partKeys!) }
                }
                if needIota { try encodeIota(enc, valsA!) }
                let scatPSO = activePasses.isEmpty ? nil : try p(carryPayload ? "radix_scatter" : "radix_scatter_nk")
                for (i, pass) in activePasses.enumerated() {
                    // The histogram of digit 0 is already in `counts` when the analysis ran it and pass 0
                    // survived; every other pass needs its own, over the keys the previous pass produced.
                    if !(analyse && sameLayout && i == 0 && pass == 0) {
                        encodeHistogram(enc, keys: keysA, shift: pass * bits,
                                        elemsPerBlock: passElems, blocks: passBlocks)
                    }
                    enc.setComputePipelineState(scanPSO)
                    enc.setBuffer(counts.mtl, offset: counts.offset, index: 0)
                    Dispatch.setUInt(enc, radix * passBlocks, index: 1)
                    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
                    enc.memoryBarrier(scope: .buffers)
                    enc.setComputePipelineState(scatPSO!)
                    enc.setBuffer(keysA.mtl, offset: keysA.offset, index: 0)
                    enc.setBuffer((valsA ?? keysA).mtl, offset: (valsA ?? keysA).offset, index: 1)
                    Dispatch.setLength(enc, n, countBuf, index: 2)
                    Dispatch.setUInt(enc, pass * bits, index: 3)
                    Dispatch.setUInt(enc, passElems, index: 4)
                    Dispatch.setUInt(enc, passBlocks, index: 5)
                    enc.setBuffer(counts.mtl, offset: counts.offset, index: 6)
                    enc.setBuffer(keysB.mtl, offset: keysB.offset, index: 7)
                    enc.setBuffer((valsB ?? keysB).mtl, offset: (valsB ?? keysB).offset, index: 8)
                    enc.dispatchThreadgroups(passGrid, threadsPerThreadgroup: tg)
                    enc.memoryBarrier(scope: .buffers)
                    swap(&keysA, &keysB)
                    if carryPayload { swap(&valsA, &valsB); resultOrder = valsA }
                }
                // `.atStart` is the one arrangement the partition cannot write directly: it lays the
                // blocks out value-first so the sort always runs on `[0, valueCount)`, and this moves the
                // three blocks into the order the caller asked for.
                if needArrange, let ord = resultOrder {
                    let arrangePSO = try p("part_arrange")
                    let dst = ord === orderA! ? orderB! : orderA!
                    enc.setComputePipelineState(arrangePSO)
                    enc.setBuffer(ord.mtl, offset: ord.offset, index: 0)
                    enc.setBuffer(sizes!.mtl, offset: sizes!.offset, index: 1)
                    Dispatch.setLength(enc, n, nil, index: 2)
                    enc.setBuffer(dst.mtl, offset: dst.offset, index: 3)
                    Dispatch.dispatch1D(enc, arrangePSO, count: n)
                    enc.memoryBarrier(scope: .buffers)
                    resultOrder = dst
                }
            }
            ctx.retainUntilFlush(self)
        }
        try ctx.syncPoint()

        let valueStart = nullPlacement == .atStart ? mNull + mNaN : 0
        let nullStart = nullPlacement == .atStart ? 0 : mValue + mNaN
        return SortRun(order: resultOrder, orderIsSorted: carryPayload, keys: keysA, measured: measured,
                       valueStart: valueStart, valueCount: mValue, nullStart: nullStart, nullCount: mNull,
                       flags: flags)
    }

    // MARK: - sorted values without the gather

    /// `sorted()` built out of the sorted keys, or nil when this column cannot take that path.
    private func sortedByInvertingKeys(descending: Bool, nullPlacement: NullPlacement) throws -> MetalArray<T>? {
        let n = length
        // Below the analysis threshold the flags are never read back, so the path cannot know whether the
        // map is invertible; a gather of that many rows costs less than the command buffer it would take
        // to find out.
        guard n >= 1 << 18, let mode = Self.unkeyMode else { return nil }
        try Dispatch.checkLength(n)
        let ctx = context
        let wide = T.byteWidth == 8
        let keyType = wide ? "ulong" : "uint"
        let src = SortSource.source(K: keyType)
        func p(_ f: String) throws -> MTLComputePipelineState {
            try ctx.pipeline(source: src, function: f, cacheKey: "sort/\(keyType)/\(SortSource.digitBits)/\(f)")
        }
        let run = try radixSortRun(descending: descending, nullPlacement: nullPlacement,
                                   wantOrder: false, keysWanted: true)
        guard run.measured else { return nil }

        // Output positions the inverse map cannot produce: the null block, and the runs of values that
        // share a key. Both ends of each run come from a binary search over the sorted keys.
        var ranges: [(Int, Int)] = []
        if run.nullCount > 0 { ranges.append((run.nullStart, run.nullStart + run.nullCount)) }
        // The NaN block `.atStart` separated out sits outside the key array and is always gathered.
        let nanBlock = n - run.valueCount - run.nullCount
        if nanBlock > 0 {
            let s = nullPlacement == .atStart ? run.nullCount : run.valueCount
            ranges.append((s, s + nanBlock))
        }
        if run.flags != 0 {
            let keys = run.keys
            func bound(_ target: UInt64, upper: Bool) -> Int {
                var lo = 0, hi = run.valueCount
                if wide {
                    let k = keys.typed(UInt64.self)
                    while lo < hi { let m = (lo + hi) / 2; if upper ? (k[m] <= target) : (k[m] < target) { lo = m + 1 } else { hi = m } }
                } else {
                    let k = keys.typed(UInt32.self)
                    let t = UInt32(truncatingIfNeeded: target)
                    while lo < hi { let m = (lo + hi) / 2; if upper ? (k[m] <= t) : (k[m] < t) { lo = m + 1 } else { hi = m } }
                }
                return lo
            }
            func addRun(_ key: UInt64) {
                let lo = bound(key, upper: false), hi = bound(key, upper: true)
                if hi > lo { ranges.append((run.valueStart + lo, run.valueStart + hi)) }
            }
            withExtendedLifetime(keys) {
                if run.flags & 1 != 0 { addRun(Self.zeroKey(wide: wide, descending: descending)) }
                if run.flags & 2 != 0 { addRun(Self.nanKey(wide: wide, descending: descending)) }
            }
        }
        // More than half the output through the fix-up is more work than gathering all of it, which is
        // what `take` used to do; the sort has already carried the row numbers, so do exactly that.
        let fixed = ranges.reduce(0) { $0 + ($1.1 - $1.0) }
        var invert = true
        // Only when the row numbers over the value block are the sorted ones: without the payload the
        // partition's are still in input order there, and only the null and NaN blocks may be read.
        if fixed * 2 > n && run.orderIsSorted { ranges = [(0, n)]; invert = false }
        if !ranges.isEmpty && run.order == nil { return nil }

        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let sourceValues = values
        try ctx.run { enc in
            if invert {
                let unkeyPSO = try p("unkey")
                enc.setComputePipelineState(unkeyPSO)
                enc.setBuffer(run.keys.mtl, offset: run.keys.offset, index: 0)
                Dispatch.setUInt(enc, run.valueCount, index: 1)
                Dispatch.setUInt(enc, descending ? 1 : 0, index: 2)
                Dispatch.setUInt(enc, Int(mode), index: 3)
                Dispatch.setUInt(enc, run.valueStart, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, unkeyPSO, count: Swift.max(run.valueCount, 1))
                enc.memoryBarrier(scope: .buffers)
            }
            if !ranges.isEmpty {
                let gatherPSO = try p("gather_range")
                let ord = run.order!
                enc.setComputePipelineState(gatherPSO)
                enc.setBuffer(sourceValues.mtl, offset: sourceValues.offset, index: 0)
                enc.setBuffer(ord.mtl, offset: ord.offset, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                for (lo, hi) in ranges {
                    Dispatch.setUInt(enc, lo, index: 2)
                    Dispatch.setUInt(enc, hi, index: 3)
                    Dispatch.dispatch1D(enc, gatherPSO, count: hi - lo)
                    enc.memoryBarrier(scope: .buffers)
                }
            }
        }
        ctx.retainUntilFlush(self)
        try ctx.syncPoint()

        // The validity bitmap is one run of set bits and one of clear: the null block is contiguous.
        var outValidity: MetalArrowBuffer? = nil
        if validity != nil {
            let bm = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: true, context: ctx)
            let p = bm.mutableTyped(UInt8.self)
            let lo = run.nullStart == 0 ? run.nullCount : 0
            let hi = run.nullStart == 0 ? n : run.nullStart
            if hi > lo {
                let firstFull = (lo + 7) / 8, lastFull = hi / 8
                if lastFull > firstFull { memset(p + firstFull, 0xFF, lastFull - firstFull) }
                for i in lo..<Swift.min(hi, firstFull * 8) { Bitmap.set(p, i) }
                for i in Swift.max(lo, lastFull * 8)..<hi { Bitmap.set(p, i) }
            }
            outValidity = bm
        }
        return MetalArray<T>(length: n, nullCount: run.nullCount, validity: outValidity, values: out, context: ctx)
    }
}

extension MetalRecordBatch {
    /// Sorts every column by one column (stable, nulls last).
    public func sorted(by column: String, descending: Bool = false) throws -> MetalRecordBatch {
        guard let c = self[column] else { throw ArrowMetalError.invalidArrowArray("no column named \(column)") }
        let idx: MetalArray<Int32>
        switch c {
        case .int8(let a): idx = try a.argsort(descending: descending)
        case .uint8(let a): idx = try a.argsort(descending: descending)
        case .int16(let a): idx = try a.argsort(descending: descending)
        case .uint16(let a): idx = try a.argsort(descending: descending)
        case .int32(let a): idx = try a.argsort(descending: descending)
        case .uint32(let a): idx = try a.argsort(descending: descending)
        case .int64(let a): idx = try a.argsort(descending: descending)
        case .uint64(let a): idx = try a.argsort(descending: descending)
        case .float32(let a): idx = try a.argsort(descending: descending)
        case .float64(let a): idx = try a.argsort(descending: descending)
        default: throw ArrowMetalError.unsupportedType("sort by \(c.arrowFormat)")
        }
        return try take(idx)
    }
}
