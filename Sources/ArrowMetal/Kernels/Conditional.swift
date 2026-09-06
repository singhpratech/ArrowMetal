import Foundation
import Metal

/// Arrow's remaining conditional and selection functions: `fill_null_forward`, `fill_null_backward`,
/// `case_when`, `choose`, `replace_with_mask` and `indices_nonzero`.
///
/// Everything here runs on the GPU. The two null fills and `replace_with_mask` are scans and get
/// their own kernels (`ConditionalSource.swift`); `case_when` and `choose` are expressed as folds of
/// the existing `if_else` kernel, one GPU pass per branch, which is the same shape `coalesce` uses
/// and needs no new kernel at all; `indices_nonzero` is `iota` put through the existing stream
/// compaction.
enum Conditional {
    /// A two-level inclusive scan over a `uint` buffer, reusing the scan `CumulativeSource`
    /// generates: block scan, exclusive scan of the block totals, fold the block prefix back in.
    /// `op` is `"max"` (the null fills) or `"sum"` (`replace_with_mask`).
    static let scanSource = CumulativeSource.source(V: "uint", extraPrelude: "", ops: [
        (name: "sum", identity: "(uint)0", body: "a + b"),
        (name: "max", identity: "(uint)0", body: "a > b ? a : b"),
    ])

    static func scanUInt(_ ctx: MetalContext, seed: MetalArrowBuffer, count n: Int, op: String) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: ctx)
        guard n > 0 else { return out }
        let blocks = (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
        let totals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        func pso(_ f: String) throws -> MTLComputePipelineState {
            try Dispatch.pipeline(ctx, family: "condscan", source: scanSource, function: f, type: "uint")
        }
        let blockPSO = try pso("cum_block_\(op)")
        let totalsPSO = try pso("cum_totals_\(op)")
        let addPSO = try pso("cum_add_\(op)")
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(blockPSO)
            enc.setBuffer(seed.mtl, offset: seed.offset, index: 0)
            enc.setBuffer(seed.mtl, offset: seed.offset, index: 1)          // unused: hasValidity is 0
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, 0, index: 3)
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
        ctx.retainUntilFlush(totals)
        return out
    }

    static func pipeline(_ ctx: MetalContext, _ source: String, _ function: String, type: String) throws -> MTLComputePipelineState {
        try Dispatch.pipeline(ctx, family: "conditional", source: source, function: function, type: type)
    }
}

// MARK: - fill_null_forward / fill_null_backward

extension MetalArray {
    /// Arrow `fill_null_forward`: every null takes the value of the nearest non-null element before
    /// it. Leading nulls (with no non-null element before them) stay null, as in Arrow.
    ///
    /// One GPU max-scan over "index of the last valid row so far", then a gather. An array with no
    /// validity bitmap is returned unchanged.
    public func fillNullForward() throws -> MetalArray<T> { try fillNullDirected(backward: false) }

    /// Arrow `fill_null_backward`: every null takes the value of the nearest non-null element after
    /// it; trailing nulls stay null.
    public func fillNullBackward() throws -> MetalArray<T> { try fillNullDirected(backward: true) }

    private func fillNullDirected(backward: Bool) throws -> MetalArray<T> {
        guard let vld = validity else { return self }
        let n = length                       // the scan needs the exact count
        try Dispatch.checkLength(n)
        let ctx = context
        guard n > 0 else { return self }
        let seed = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let seedPSO = try Conditional.pipeline(ctx, ConditionalSource.common, "cn_fill_seed", type: "common")
        try ctx.run { enc in
            enc.setComputePipelineState(seedPSO)
            enc.setBuffer(vld.mtl, offset: vld.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            Dispatch.setUInt(enc, backward ? 1 : 0, index: 2)
            enc.setBuffer(seed.mtl, offset: seed.offset, index: 3)
            Dispatch.dispatch1D(enc, seedPSO, count: n)
        }
        let scan = try Conditional.scanUInt(ctx, seed: seed, count: n, op: "max")

        let outValues = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: n, zeroed: false, context: ctx)
        let mslT = Dispatch.moveType(T.self)
        let gatherPSO = try Conditional.pipeline(ctx, ConditionalSource.moves(T: mslT), "cn_fill_gather", type: mslT)
        try ctx.run { enc in
            enc.setComputePipelineState(gatherPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(scan.mtl, offset: scan.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, backward ? 1 : 0, index: 3)
            enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 4)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 5)
            Dispatch.dispatch1D(enc, gatherPSO, count: n)
        }
        ctx.retainUntilFlush(self); ctx.retainUntilFlush(seed); ctx.retainUntilFlush(scan); ctx.retainUntilFlush(validBytes)
        let outValidity = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n)
        let res = MetalArray<T>(length: n, nullCount: 0, validity: outValidity, values: outValues, context: ctx)
        res.recomputeNullCount()
        return res
    }
}

extension MetalBooleanArray {
    /// Arrow `fill_null_forward` on a boolean column (unpack to bytes, fill, repack).
    public func fillNullForward() throws -> MetalBooleanArray {
        guard validity != nil else { return self }
        return try MetalBooleanArray.fromUInt8Array(try toUInt8Array().fillNullForward())
    }
    /// Arrow `fill_null_backward` on a boolean column.
    public func fillNullBackward() throws -> MetalBooleanArray {
        guard validity != nil else { return self }
        return try MetalBooleanArray.fromUInt8Array(try toUInt8Array().fillNullBackward())
    }
}

// MARK: - case_when / choose

extension MetalArray {
    /// An all-null column of `length` elements, the starting accumulator for `case_when` and
    /// `choose` when no branch matches.
    static func allNull(length: Int, context: MetalContext) throws -> MetalArray<T> {
        let a = try MetalArray<T>.allocate(length: length, withValidity: true, context: context)
        a.nullCount = length
        return a
    }

    /// Arrow `case_when`: the value of the first branch whose condition is true, or `defaultValue`
    /// (null when there is none).
    ///
    /// Arrow's `case_when` takes the conditions as a `struct` of booleans; here they are a plain
    /// array, one per branch. **A null condition counts as false**, which is what Arrow does — the
    /// row falls through to the next condition rather than becoming null. A null in the chosen
    /// branch's values does make the output null.
    ///
    /// Implemented as a right-to-left fold of the GPU `if_else` kernel: `k` branches cost `k`
    /// passes, and no branch's values are read for a row that does not select it.
    public static func caseWhen(conds: [MetalBooleanArray], values: [MetalArray<T>],
                                else defaultValue: MetalArray<T>? = nil) throws -> MetalArray<T> {
        guard conds.count == values.count else {
            throw ArrowMetalError.invalidArrowArray("case_when has \(conds.count) conditions for \(values.count) value columns")
        }
        guard let n = conds.first?.length ?? values.first?.length ?? defaultValue?.length else {
            throw ArrowMetalError.invalidArrowArray("case_when needs at least one condition or a default")
        }
        let ctx = conds.first?.context ?? values.first?.context ?? defaultValue!.context
        for c in conds where c.length != n { throw ArrowMetalError.lengthMismatch(n, c.length) }
        for v in values where v.length != n { throw ArrowMetalError.lengthMismatch(n, v.length) }
        if let d = defaultValue, d.length != n { throw ArrowMetalError.lengthMismatch(n, d.length) }

        var acc = try defaultValue ?? MetalArray<T>.allNull(length: n, context: ctx)
        for i in stride(from: conds.count - 1, through: 0, by: -1) {
            // Arrow reads a null condition as false, so fill it before the three-valued if_else.
            let c = try conds[i].fillingNull(false)
            acc = try MetalArray<T>.ifElse(c, values[i], acc)
        }
        return acc
    }

    /// Arrow `choose`: `values[indices[i]][i]`, element-wise. A null index gives a null output; an
    /// index outside `[0, values.count)` is an error, as in Arrow.
    ///
    /// The range check is one GPU min and one GPU max over the index column (both skip nulls); the
    /// selection itself is a fold of `if_else`, one pass per candidate column.
    public static func choose<I: ArrowPrimitive>(_ indices: MetalArray<I>, _ values: [MetalArray<T>]) throws -> MetalArray<T> {
        guard !values.isEmpty else { throw ArrowMetalError.invalidArrowArray("choose needs at least one value column") }
        let n = indices.length
        for v in values where v.length != n { throw ArrowMetalError.lengthMismatch(n, v.length) }
        if let lo = try indices.min(), lo.asInt64 < 0 {
            throw ArrowMetalError.invalidArrowArray("choose: index \(lo.asInt64) out of range [0, \(values.count))")
        }
        if let hi = try indices.max(), hi.asInt64 >= Int64(values.count) {
            throw ArrowMetalError.invalidArrowArray("choose: index \(hi.asInt64) out of range [0, \(values.count))")
        }
        var acc = try MetalArray<T>.allNull(length: n, context: indices.context)
        for j in stride(from: values.count - 1, through: 0, by: -1) {
            guard let scalar = I(exactly: j) else {
                throw ArrowMetalError.invalidArrowArray("choose: \(values.count) columns do not fit the index type \(I.arrowFormat)")
            }
            acc = try MetalArray<T>.ifElse(try indices.compare(.eq, scalar), values[j], acc)
        }
        return acc
    }
}

// MARK: - replace_with_mask

extension MetalArray {
    /// Arrow `replace_with_mask`: rows where `mask` is true take the next value from `replacements`,
    /// in order; rows where the mask is null become null; every other row keeps its own value.
    ///
    /// `replacements` must hold at least as many elements as the mask has valid `true`s (Arrow
    /// raises when it holds fewer, and ignores the surplus when it holds more). One GPU sum-scan of
    /// the mask gives each selected row its position in `replacements`, then one gather.
    public func replaceWithMask(_ mask: MetalBooleanArray, _ replacements: MetalArray<T>) throws -> MetalArray<T> {
        let n = length
        guard mask.length == n else { throw ArrowMetalError.lengthMismatch(n, mask.length) }
        try Dispatch.checkLength(n)
        let ctx = context
        guard n > 0 else { return self }

        let seed = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let seedPSO = try Conditional.pipeline(ctx, ConditionalSource.common, "cn_mask_seed", type: "common")
        let maskValid = mask.validity ?? mask.values
        try ctx.run { enc in
            enc.setComputePipelineState(seedPSO)
            enc.setBuffer(mask.values.mtl, offset: mask.values.offset, index: 0)
            enc.setBuffer(maskValid.mtl, offset: maskValid.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            Dispatch.setUInt(enc, mask.validity == nil ? 0 : 1, index: 3)
            enc.setBuffer(seed.mtl, offset: seed.offset, index: 4)
            Dispatch.dispatch1D(enc, seedPSO, count: n)
        }
        let scan = try Conditional.scanUInt(ctx, seed: seed, count: n, op: "sum")
        try ctx.syncPoint()          // the scan total is read on the CPU next
        let needed = Int(withExtendedLifetime(scan) { scan.typed(UInt32.self)[n - 1] })
        guard replacements.length >= needed else {
            throw ArrowMetalError.invalidArrowArray(
                "replace_with_mask: the mask selects \(needed) rows but replacements has \(replacements.length)")
        }

        let outValues = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: n, zeroed: false, context: ctx)
        let mslT = Dispatch.moveType(T.self)
        let pso = try Conditional.pipeline(ctx, ConditionalSource.moves(T: mslT), "cn_replace", type: mslT)
        let flags = (mask.validity == nil ? 0 : 1) | (validity == nil ? 0 : 2) | (replacements.validity == nil ? 0 : 4)
        let vv = validity ?? values, rv = replacements.validity ?? replacements.values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
            enc.setBuffer(mask.values.mtl, offset: mask.values.offset, index: 2)
            enc.setBuffer(maskValid.mtl, offset: maskValid.offset, index: 3)
            enc.setBuffer(replacements.values.mtl, offset: replacements.values.offset, index: 4)
            enc.setBuffer(rv.mtl, offset: rv.offset, index: 5)
            enc.setBuffer(scan.mtl, offset: scan.offset, index: 6)
            Dispatch.setLength(enc, n, nil, index: 7)
            Dispatch.setUInt(enc, flags, index: 8)
            enc.setBuffer(outValues.mtl, offset: outValues.offset, index: 9)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 10)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(self); ctx.retainUntilFlush(mask); ctx.retainUntilFlush(replacements)
        ctx.retainUntilFlush(seed); ctx.retainUntilFlush(scan); ctx.retainUntilFlush(validBytes)
        let outValidity = try BitmapOps.packBits(ctx, bytes: validBytes, bits: n)
        let res = MetalArray<T>(length: n, nullCount: 0, validity: outValidity, values: outValues, context: ctx)
        res.recomputeNullCount()
        return res
    }
}

extension MetalBooleanArray {
    /// Arrow `replace_with_mask` on a boolean column.
    public func replaceWithMask(_ mask: MetalBooleanArray, _ replacements: MetalBooleanArray) throws -> MetalBooleanArray {
        try MetalBooleanArray.fromUInt8Array(
            try toUInt8Array().replaceWithMask(mask, try replacements.toUInt8Array()))
    }
}

// MARK: - indices_nonzero

extension MetalArray {
    /// Arrow `indices_nonzero`: the **uint64** row numbers where the value is valid and not zero,
    /// in order. The result never has nulls.
    ///
    /// `-0.0` counts as zero and every NaN counts as non-zero, which is what the IEEE `!= 0` test
    /// this uses gives, and what Arrow does.
    public func indicesNonzero() throws -> MetalArray<UInt64> {
        let mask = try compare(.ne, 0)
        return try MetalArray<UInt64>.iota64(length, context: context).filter(mask)
    }

    /// `0, 1, 2, ...` as uint64, on the GPU.
    static func iota64(_ n: Int, context ctx: MetalContext) throws -> MetalArray<UInt64> {
        let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try Conditional.pipeline(ctx, ConditionalSource.common, "cn_iota64", type: "common")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                Dispatch.setLength(enc, n, nil, index: 0)
                enc.setBuffer(buf.mtl, offset: buf.offset, index: 1)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<UInt64>(length: n, nullCount: 0, validity: nil, values: buf, context: ctx)
    }
}

extension MetalBooleanArray {
    /// Arrow `indices_nonzero` on a boolean column: the row numbers of the valid `true`s.
    public func indicesNonzero() throws -> MetalArray<UInt64> {
        try MetalArray<UInt64>.iota64(length, context: context).filter(self)
    }
}
