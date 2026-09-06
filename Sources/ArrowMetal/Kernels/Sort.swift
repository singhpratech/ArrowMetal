import Foundation
import Metal

extension MetalArray {
    /// Arrow `array_sort_indices`: indices that sort the values, stable.
    ///
    /// `descending` picks the direction and `nullPlacement` decides whether the null rows sit after every
    /// value (Arrow's default) or before every one of them; the two are independent, exactly as in Arrow,
    /// so nulls stay at the chosen end in both directions. NaN sorts after +inf (total order). Runs an
    /// LSD radix sort on the GPU (4 passes for 32-bit, 8 for 64-bit).
    public func argsort(descending: Bool = false,
                        nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Int32> {
        try Dispatch.checkLength(length)
        let ctx = context
        let n = length
        if n == 0 { return try MetalArray<Int32>([Int32](), context: ctx) }
        let wide = T.byteWidth == 8
        let keyType = wide ? "ulong" : "uint"
        let src = SortSource.source(K: keyType)
        func p(_ f: String) throws -> MTLComputePipelineState { try ctx.pipeline(source: src, function: f, cacheKey: "sort/\(keyType)/\(f)") }
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
        let elemsPerBlock = 4096
        let blocks = (n + elemsPerBlock - 1) / elemsPerBlock
        let counts = try MetalArrowBuffer.allocate(byteCount: 256 * blocks * 4, zeroed: false, context: ctx)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let mapPSO = try p(mapFn), iotaPSO = try p("iota_u32"), histPSO = try p("radix_histogram"), scanPSO = try p("radix_scan"), scatPSO = try p("radix_scatter")
        let passes = wide ? 8 : 4
        try ctx.run { enc in
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
            for pass in 0..<passes {
                let shift = pass * 8
                enc.setComputePipelineState(histPSO)
                enc.setBuffer(keysA.mtl, offset: 0, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, shift, index: 2)
                Dispatch.setUInt(enc, elemsPerBlock, index: 3)
                Dispatch.setUInt(enc, blocks, index: 4)
                enc.setBuffer(counts.mtl, offset: 0, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(scanPSO)
                enc.setBuffer(counts.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, 256 * blocks, index: 1)
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
                enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                swap(&keysA, &keysB); swap(&valsA, &valsB)
            }
        }
        try ctx.syncPoint()
        // valsA holds the sorted original indices (uint32). Nulls: move them last (stable) on the CPU side of the
        // index array, which is cheap relative to the sort; descending reverses the non-null prefix.
        let idx = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: valsA, context: ctx)
        guard let bm = validity?.typed(UInt8.self) else { return idx }
        // A stable partition of the index array moves the nulls to whichever end the caller asked for,
        // keeping their original row order among themselves (Arrow's stable order for nulls is the input
        // order, not the order of the bytes that happen to sit under the validity bitmap). This is the
        // same host-side pass the nulls-last path has always run; `at_start` only changes where the two
        // blocks are written.
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let src2 = valsA.typed(Int32.self), dst = out.mutableTyped(Int32.self)
        var nulls: [Int32] = []
        nulls.reserveCapacity(nullCount)
        for i in 0..<n { let j = src2[i]; if !Bitmap.isSet(bm, Int(j)) { nulls.append(j) } }
        nulls.sort()
        var k = nullPlacement == .atStart ? nulls.count : 0
        for i in 0..<n { let j = src2[i]; if Bitmap.isSet(bm, Int(j)) { dst[k] = j; k += 1 } }
        k = nullPlacement == .atStart ? 0 : k
        for j in nulls { dst[k] = j; k += 1 }
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Sorted copy (nulls at whichever end `nullPlacement` names, `atEnd` by default).
    public func sorted(descending: Bool = false,
                       nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<T> {
        try take(try argsort(descending: descending, nullPlacement: nullPlacement))
    }

    /// Indices of the k smallest (or largest) values, in the same order `argsort` would put them.
    ///
    /// For k up to 1024 this is a partial selection (`Kernels/TopK.swift`): each threadgroup keeps only
    /// the best k of its own block, and one small radix sort orders the surviving candidates — about one
    /// read per row instead of the eight radix passes a full sort costs. Larger k, types the selection
    /// kernel does not map, and the case where fewer than k rows are non-null fall back to the sort.
    public func topK(_ k: Int, largest: Bool = true) throws -> MetalArray<Int32> {
        guard k > 0 else { return try MetalArray<Int32>([Int32](), context: context) }
        if let selected = try topKSelect(k, largest: largest) { return selected }
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
