import Foundation
import Metal

extension MetalArray {
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
        let ctx = context
        let n = length
        if n == 0 { return try MetalArray<Int32>([Int32](), context: ctx) }
        let wide = T.byteWidth == 8
        let keyType = wide ? "ulong" : "uint"
        let bits = SortSource.digitBits
        let radix = 1 << bits
        let src = SortSource.source(K: keyType)
        func p(_ f: String) throws -> MTLComputePipelineState { try ctx.pipeline(source: src, function: f, cacheKey: "sort/\(keyType)/\(bits)/\(f)") }
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
        _ = tmpKeep
        let kb = wide ? 8 : 4
        var keysA = try MetalArrowBuffer.allocate(byteCount: n * kb, zeroed: false, context: ctx)
        var keysB = try MetalArrowBuffer.allocate(byteCount: n * kb, zeroed: false, context: ctx)
        var valsA = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        var valsB = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        // Each block reads and writes `radix` entries of the counts table on every pass, at a stride of
        // `blocks`, and the scan below is a *single* threadgroup over `radix * blocks` entries, so both
        // fixed costs fall as blocks get bigger and fewer; against that, too few blocks leaves GPU cores
        // idle. The scatter is memory bound, which pushes the balance a long way towards fewer: at 50M
        // int64 rows argsort measures 37.7 ms at 128 blocks, 49.0 ms at 256 and 55.8 ms at 2048.
        // Small inputs keep the old rule instead — a 20k-element argsort (the size top-k's final
        // ordering lands on) must not run on five threadgroups — so the block halves until there are at
        // least 64 of them, and never holds fewer than 4096 elements.
        var elemsPerBlock = Swift.max(4096, ((n + 127) / 128 + 255) / 256 * 256)
        while elemsPerBlock > 256 && (n + elemsPerBlock - 1) / elemsPerBlock < 64 { elemsPerBlock >>= 1 }
        let blocks = (n + elemsPerBlock - 1) / elemsPerBlock
        let counts = try MetalArrowBuffer.allocate(byteCount: radix * blocks * 4, zeroed: false, context: ctx)
        let spanOr = try MetalArrowBuffer.allocate(byteCount: blocks * kb, zeroed: false, context: ctx)
        let spanAnd = try MetalArrowBuffer.allocate(byteCount: blocks * kb, zeroed: false, context: ctx)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let mapPSO = try p(mapFn), iotaPSO = try p("iota_u32"), histPSO = try p("radix_histogram"), scanPSO = try p("radix_scan"), scatPSO = try p("radix_scatter")
        let passes = ((wide ? 64 : 32) + bits - 1) / bits
        let blockGrid = MTLSize(width: blocks, height: 1, depth: 1)

        func encodeMap(_ enc: MTLComputeCommandEncoder) {
            enc.setComputePipelineState(mapPSO)
            enc.setBuffer(source.mtl, offset: source.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(keysA.mtl, offset: 0, index: 2)
            Dispatch.setUInt(enc, descending ? 1 : 0, index: 3)
            Dispatch.dispatch1D(enc, mapPSO, count: n)
            enc.setComputePipelineState(iotaPSO)
            enc.setBuffer(valsA.mtl, offset: 0, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.dispatch1D(enc, iotaPSO, count: n)
            enc.memoryBarrier(scope: .buffers)
        }
        func encodeHistogram(_ enc: MTLComputeCommandEncoder, shift: Int) {
            enc.setComputePipelineState(histPSO)
            enc.setBuffer(keysA.mtl, offset: 0, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.setUInt(enc, shift, index: 2)
            Dispatch.setUInt(enc, elemsPerBlock, index: 3)
            Dispatch.setUInt(enc, blocks, index: 4)
            enc.setBuffer(counts.mtl, offset: 0, index: 5)
            enc.setBuffer(spanOr.mtl, offset: 0, index: 6)
            enc.setBuffer(spanAnd.mtl, offset: 0, index: 7)
            enc.dispatchThreadgroups(blockGrid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
        }
        func encodeScanAndScatter(_ enc: MTLComputeCommandEncoder, shift: Int) {
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(counts.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, radix * blocks, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatPSO)
            enc.setBuffer(keysA.mtl, offset: 0, index: 0)
            enc.setBuffer(valsA.mtl, offset: 0, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, shift, index: 3)
            Dispatch.setUInt(enc, elemsPerBlock, index: 4)
            Dispatch.setUInt(enc, blocks, index: 5)
            enc.setBuffer(counts.mtl, offset: 0, index: 6)
            enc.setBuffer(keysB.mtl, offset: 0, index: 7)
            enc.setBuffer(valsB.mtl, offset: 0, index: 8)
            enc.dispatchThreadgroups(blockGrid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            swap(&keysA, &keysB); swap(&valsA, &valsB)
        }

        // A pass whose digit is the same in every row is the identity permutation — the sort is stable,
        // so equal digits keep their order — and can be dropped entirely. The first histogram reports
        // the bitwise OR and AND of the keys along with its counts, and `or ^ and` is exactly the set of
        // bits that differ somewhere in the column, so one readback names every skippable pass at once.
        // A narrow range (sorted-ish data, small integers, a column of one repeated value) can lose most
        // of the passes this way. The readback costs a command-buffer boundary, so inputs small enough
        // for that to matter keep the whole sort in one buffer and run every pass.
        var activePasses = Array(0..<passes)
        let analyse = n >= 1 << 18
        if analyse {
            try ctx.run { enc in
                encodeMap(enc)
                encodeHistogram(enc, shift: 0)
            }
            try ctx.syncPoint()
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
        try ctx.run { enc in
            if !analyse { encodeMap(enc) }
            for (i, pass) in activePasses.enumerated() {
                // The histogram of digit 0 is already in `counts` when the analysis ran it and pass 0
                // survived; every other pass needs its own, over the keys the previous pass produced.
                if !(analyse && i == 0 && pass == 0) { encodeHistogram(enc, shift: pass * bits) }
                encodeScanAndScatter(enc, shift: pass * bits)
            }
        }
        try ctx.syncPoint()
        // valsA holds the sorted original indices (uint32). The GPU pass leaves the nulls wherever the
        // key map put them and every NaN in one block after +inf; a host-side stable partition of the
        // index array then moves both to the end the caller asked for. That partition is O(n) against
        // the sort's several passes, and it is the same pass the nulls-last path has always run.
        //
        // NaN travels with the nulls, not with the values, which is Arrow's rule: a NaN is "greater
        // than any value" in the same sense a null is, so `at_start` moves both to the front — nulls
        // first, then the NaNs, then the values. `at_end` needs no move at all, because the radix keys
        // already leave them in exactly that order.
        let idx = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: valsA, context: ctx)
        let nanCount = nullPlacement == .atStart && T.isFloatingPoint ? nanRowCount() : 0
        guard validity != nil || nanCount > 0 else { return idx }
        let bm = validity?.typed(UInt8.self)
        let values = T.isFloatingPoint && nanCount > 0 ? valuePointer : nil

        func bucket(_ row: Int32) -> Int {
            if let bm, !Bitmap.isSet(bm, Int(row)) { return 0 }        // null
            if let values, values[Int(row)] != values[Int(row)] { return 1 }   // NaN
            return 2                                                   // an ordinary value
        }

        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let src2 = valsA.typed(Int32.self), dst = out.mutableTyped(Int32.self)
        // Nulls keep their *input* order among themselves, which is not the order the bytes under the
        // validity bitmap happened to sort into.
        var nulls: [Int32] = []
        nulls.reserveCapacity(nullCount)
        for i in 0..<n { let j = src2[i]; if bucket(j) == 0 { nulls.append(j) } }
        nulls.sort()
        var nans: [Int32] = []
        if nanCount > 0 {
            nans.reserveCapacity(nanCount)
            for i in 0..<n { let j = src2[i]; if bucket(j) == 1 { nans.append(j) } }
        }
        let front = nullPlacement == .atStart ? nulls.count + nans.count : 0
        var k = front
        for i in 0..<n { let j = src2[i]; if bucket(j) == 2 { dst[k] = j; k += 1 } }
        k = nullPlacement == .atStart ? 0 : k
        if nullPlacement == .atStart {
            for j in nulls { dst[k] = j; k += 1 }
            for j in nans { dst[k] = j; k += 1 }
        } else {
            for j in nulls { dst[k] = j; k += 1 }
        }
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// How many rows carry a NaN. Only the `at_start` float path asks, and only then does it pay for
    /// the scan.
    private func nanRowCount() -> Int {
        guard T.isFloatingPoint else { return 0 }
        let p = valuePointer
        var count = 0
        if let bm = validity?.typed(UInt8.self) {
            for i in 0..<length where Bitmap.isSet(bm, i) && p[i] != p[i] { count += 1 }
        } else {
            for i in 0..<length where p[i] != p[i] { count += 1 }
        }
        return count
    }

    /// Sorted copy (nulls at whichever end `nullPlacement` names, `atEnd` by default).
    public func sorted(descending: Bool = false,
                       nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<T> {
        try take(try argsort(descending: descending, nullPlacement: nullPlacement))
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
