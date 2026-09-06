import Foundation
import Metal

/// Arrow's cumulative (running) functions.
public enum CumulativeOp: String, CaseIterable, Sendable {
    case sum, min, max
}

extension MetalArray {
    /// Arrow `cumulative_sum` / `cumulative_min` / `cumulative_max` over this column.
    ///
    /// Nulls are skipped in Arrow's sense: the output is null exactly where the input is, and the
    /// running value carries across the gap unchanged. The result shares the input's validity bitmap
    /// zero-copy.
    ///
    /// Integer sums wrap, like the rest of the unchecked arithmetic here. `float32` and `float64` sums
    /// run a two-level scan, which reassociates the additions, so the last few ulp can differ from a
    /// strictly sequential sum; min and max are exact on every type. `float64` accumulates through the
    /// software binary64 adder in `DoubleMath`, and its min/max order values on their bit patterns —
    /// NaN is skipped, as it is by the `min`/`max` reductions.
    ///
    /// The scan needs an exact element count, so a pending (batched) input is materialised first.
    public func cumulative(_ op: CumulativeOp) throws -> MetalArray<T> {
        let n = length                       // resolves a pending length; the totals pass needs the real count
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * T.byteWidth, zeroed: false, context: ctx)
        guard n > 0 else { return MetalArray<T>(length: 0, nullCount: 0, validity: nil, values: out, context: ctx) }

        let blocks = (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
        let totals = try MetalArrowBuffer.allocate(byteCount: blocks * T.byteWidth, zeroed: false, context: ctx)
        let (src, cacheType) = Self.cumulativeSource
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "cumulative", source: src, function: f, type: cacheType)
        }
        let blockPSO = try pso("cum_block_\(op.rawValue)")
        let totalsPSO = try pso("cum_totals_\(op.rawValue)")
        let addPSO = try pso("cum_add_\(op.rawValue)")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        let hasV = validity != nil
        try ctx.run { enc in
            enc.setComputePipelineState(blockPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            let v = validity ?? values
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, hasV ? 1 : 0, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 5)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(totalsPSO)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(addPSO)
            enc.setBuffer(out.mtl, offset: out.offset, index: 0)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
        ctx.retainUntilFlush(self)
        ctx.retainUntilFlush(totals)
        return MetalArray<T>(length: n, nullCount: _nullCount, validity: validity, values: out, context: ctx)
    }

    public func cumulativeSum() throws -> MetalArray<T> { try cumulative(.sum) }
    public func cumulativeMin() throws -> MetalArray<T> { try cumulative(.min) }
    public func cumulativeMax() throws -> MetalArray<T> { try cumulative(.max) }

    /// Generated scan source for this element type, plus its pipeline cache key.
    static var cumulativeSource: (source: String, type: String) {
        let ids = MathTypes.identities(T.self)
        if T.self == Double.self {
            // Values are raw binary64 patterns; adds go through the software adder, order through `d_key`.
            let lt = "d_key((long)a) < d_key((long)b)"
            let ops: [CumulativeSource.Op] = [
                (name: "sum", identity: ids.sum, body: "d_add(a, b)"),
                (name: "min", identity: ids.min, body: "d_is_nan(a) ? b : (d_is_nan(b) ? a : ((\(lt)) ? a : b))"),
                (name: "max", identity: ids.max, body: "d_is_nan(a) ? b : (d_is_nan(b) ? a : ((\(lt)) ? b : a))"),
            ]
            return (CumulativeSource.source(V: "ulong", extraPrelude: DoubleMath.msl, ops: ops), "double")
        }
        let t = T.mslType, u = MathTypes.unsigned(t)
        let ops: [CumulativeSource.Op]
        if T.isFloatingPoint {
            ops = [
                (name: "sum", identity: ids.sum, body: "a + b"),
                (name: "min", identity: ids.min, body: "isnan(a) ? b : (isnan(b) ? a : (a < b ? a : b))"),
                (name: "max", identity: ids.max, body: "isnan(a) ? b : (isnan(b) ? a : (a > b ? a : b))"),
            ]
        } else {
            ops = [
                (name: "sum", identity: ids.sum, body: "(\(t))((\(u))a + (\(u))b)"),
                (name: "min", identity: ids.min, body: "a < b ? a : b"),
                (name: "max", identity: ids.max, body: "a > b ? a : b"),
            ]
        }
        return (CumulativeSource.source(V: t, extraPrelude: "", ops: ops), t)
    }
}
