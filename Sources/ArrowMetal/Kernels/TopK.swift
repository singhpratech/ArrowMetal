import Foundation
import Metal

/// GPU top-k selection: the k best rows without sorting the other n - k.
///
/// A full argsort of 50M Int64 costs eight radix passes over the whole array. Top-k needs far less:
/// each threadgroup keeps only the best k of its own block, and the union of those per-block winners is
/// guaranteed to contain the global winners, so the final ordering only has to sort `blocks * k`
/// candidates with the existing radix sort. See `TopKSource` for the per-block selection.
///
/// The ordering is the same total order `argsort` uses — the order-preserving key of the value, ties
/// broken by row index — so `topK(k)` returns exactly the first k indices `argsort` would.
enum TopK {
    /// Value types the selection kernel maps to a sort key directly. Everything else keeps the sort path.
    static func keyKind<T: ArrowPrimitive>(_: T.Type) -> (kind: String, valueType: String, keyType: String)? {
        switch T.self {
        case is Int32.Type: return ("i32", "int", "uint")
        case is UInt32.Type: return ("u32", "uint", "uint")
        case is Float.Type: return ("f32", "float", "uint")
        case is Int64.Type: return ("i64", "long", "ulong")
        case is UInt64.Type: return ("u64", "ulong", "ulong")
        case is Double.Type: return ("f64", "ulong", "ulong")
        default: return nil
        }
    }

    /// Number of selection blocks, and the elements each one scans.
    ///
    /// Enough blocks to fill the GPU, few enough that `blocks * k` candidates stay cheap to sort, and
    /// never so many that a block holds fewer than k rows (a short block would have to pad).
    static func plan(n: Int, k: Int) -> (blocks: Int, elemsPerBlock: Int) {
        let candidateBudget = 1 << 19
        var blocks = Swift.max(1, Swift.min(2048, (n + 32_767) / 32_768))
        blocks = Swift.max(1, Swift.min(blocks, candidateBudget / k))
        blocks = Swift.max(1, Swift.min(blocks, n / k))
        return (blocks, (n + blocks - 1) / blocks)
    }

    /// Capacity of the per-threadgroup buffer: room for the k survivors plus one full chunk of arrivals,
    /// rounded up to a power of two for the bitonic network.
    static func capacity(k: Int) -> Int {
        var c = 1
        while c < k + Dispatch.threadgroupSize { c <<= 1 }
        return c
    }
}

extension MetalArray {
    /// Row indices of the k best rows, selected per threadgroup and ordered by one small radix sort.
    /// Returns nil when this array's type or shape is better served by the full sort.
    func topKSelect(_ k: Int, largest: Bool) throws -> MetalArray<Int32>? {
        guard k > 0, k <= 1024, let kind = TopK.keyKind(T.self) else { return nil }
        let n = length
        // Fewer valid rows than k means the answer has to reach into the null rows, which the selection
        // kernel skips; the sort path already places them, so let it.
        guard n >= k, length - nullCount >= k else { return nil }
        try Dispatch.checkLength(n)
        let ctx = context
        let (blocks, elemsPerBlock) = TopK.plan(n: n, k: k)
        let cap = TopK.capacity(k: k)
        let wide = kind.keyType == "ulong"
        let src = TopKSource.source(kind: kind.kind, V: kind.valueType, K: kind.keyType)
        let pso = try Dispatch.pipeline(ctx, family: "topk", source: src, function: "topk_select", type: kind.kind)
        let m = blocks * k
        let candKeys = try MetalArrowBuffer.allocate(byteCount: m * (wide ? 8 : 4), zeroed: false, context: ctx)
        let candIdx = try MetalArrowBuffer.allocate(byteCount: m * 4, zeroed: false, context: ctx)
        let vv = validity ?? values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            Dispatch.setUInt(enc, largest ? 1 : 0, index: 4)
            Dispatch.setUInt(enc, k, index: 5)
            Dispatch.setUInt(enc, cap, index: 6)
            Dispatch.setUInt(enc, elemsPerBlock, index: 7)
            enc.setBuffer(candKeys.mtl, offset: candKeys.offset, index: 8)
            enc.setBuffer(candIdx.mtl, offset: candIdx.offset, index: 9)
            enc.setThreadgroupMemoryLength(roundUp(cap * (wide ? 8 : 4), to: 16), index: 0)
            enc.setThreadgroupMemoryLength(roundUp(cap * 4, to: 16), index: 1)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        ctx.retainUntilFlush(self)

        // Each block writes its k candidates already ascending in (key, row) order, and block b covers
        // lower row indices than block b + 1. So candidate position is itself the tie-break the total
        // order wants, and one *stable* radix sort of the keys is the whole final ordering.
        let idx = MetalArray<UInt32>(length: m, nullCount: 0, validity: nil, values: candIdx, context: ctx)
        let ord: MetalArray<Int32>
        if wide {
            ord = try MetalArray<UInt64>(length: m, nullCount: 0, validity: nil, values: candKeys, context: ctx).argsort()
        } else {
            ord = try MetalArray<UInt32>(length: m, nullCount: 0, validity: nil, values: candKeys, context: ctx).argsort()
        }
        // A block with fewer than k selectable rows (a short tail block, or one that is all nulls) pads
        // with a sentinel row index. Dropping the sentinels keeps the relative order of the real
        // candidates, so the winners are still the first k.
        let gathered = try idx.take(ord)
        let ranked = try gathered.filter(try gathered.compare(.ne, UInt32.max))
        guard ranked.length >= k else { return nil }
        return try ranked.slice(offset: 0, length: k).cast(to: Int32.self)
    }
}
