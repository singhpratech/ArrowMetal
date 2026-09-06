import Foundation
import Metal

/// Which rows a join keeps.
public enum JoinKind: String, Sendable, CaseIterable {
    /// Only left rows that match at least one build row.
    case inner
    /// Every left row: unmatched ones come back once, with a null right index.
    case left
}

/// Key types the GPU hash join accepts (Arrow `int32` / `int64`).
public protocol ArrowJoinKey: ArrowPrimitive {}
extension Int32: ArrowJoinKey {}
extension Int64: ArrowJoinKey {}

/// GPU hash join over equal keys. Returns the index pairs of every match: `leftIndices[i]` is a row of
/// `left` and `rightIndices[i]` a row of `right` with the same key. Duplicate keys on either side produce
/// every combination (many-to-many). Null keys never match; with `.left`, a left row with no match appears
/// once with a null right index. The pair order is unspecified.
///
/// The right side is the build side: an open-addressing table of `2^ceil(log2(2 * right.length))` slots in
/// device memory, with duplicate keys chained per bucket. The left side probes it twice (count, exclusive
/// scan on the GPU, write), so the output is written without atomics and every left row keeps its pairs
/// together.
public func hashJoin<K: ArrowJoinKey>(left: MetalArray<K>, right: MetalArray<K>,
                                      kind: JoinKind) throws -> (leftIndices: MetalArray<Int32>, rightIndices: MetalArray<Int32>) {
    let ctx = left.context
    guard ctx === right.context else { throw ArrowMetalError.invalidArrowArray("join: both sides must share one MetalContext") }
    let nL = left.length, nR = right.length
    try Dispatch.checkLength(nL)
    try Dispatch.checkLength(nR)
    guard nL <= Int(Int32.max), nR <= Int(Int32.max) else {
        throw ArrowMetalError.invalidArrowArray("join: arrays above 2^31 rows are not supported")
    }
    if nL == 0 {
        return (try MetalArray<Int32>([Int32](), context: ctx), try MetalArray<Int32>([Int32](), context: ctx))
    }

    // Table size: the next power of two at least twice the build rows, so a free slot always exists.
    var tableSize = 1
    while tableSize < 2 * Swift.max(1, nR) { tableSize <<= 1 }
    let mask = tableSize - 1

    let src = JoinSource.source(KT: K.mslType)
    func pso(_ fn: String) throws -> MTLComputePipelineState {
        try Dispatch.pipeline(ctx, family: "join", source: src, function: fn, type: K.mslType)
    }
    let clearPSO = try pso("hj_clear"), buildPSO = try pso("hj_build"), countPSO = try pso("hj_probe_count")
    let scanBlockPSO = try pso("hj_scan_block"), scanTotalsPSO = try pso("hj_scan_totals"), scanAddPSO = try pso("hj_scan_add")
    let writePSO = try pso("hj_probe_write")

    let slots = try MetalArrowBuffer.allocate(byteCount: tableSize * 4, zeroed: false, context: ctx)
    let next = try MetalArrowBuffer.allocate(byteCount: Swift.max(nR, 1) * 4, zeroed: false, context: ctx)
    let counts = try MetalArrowBuffer.allocate(byteCount: nL * 4, zeroed: false, context: ctx)
    let offsets = try MetalArrowBuffer.allocate(byteCount: nL * 4, zeroed: false, context: ctx)
    let blocks = (nL + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
    let blockTotals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
    let grand = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
    let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
    let isLeft = kind == .left ? 1 : 0
    let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)

    /// Binds the probe arguments shared by both probe passes (buffers 0...8).
    func bindProbe(_ enc: MTLComputeCommandEncoder) {
        enc.setBuffer(left.values.mtl, offset: left.values.offset, index: 0)
        let lv = left.validity ?? left.values
        enc.setBuffer(lv.mtl, offset: lv.offset, index: 1)
        Dispatch.setUInt(enc, left.validity == nil ? 0 : 1, index: 2)
        Dispatch.setLength(enc, nL, nil, index: 3)
        Dispatch.setUInt(enc, mask, index: 4)
        enc.setBuffer(right.values.mtl, offset: right.values.offset, index: 5)
        enc.setBuffer(slots.mtl, offset: 0, index: 6)
        enc.setBuffer(next.mtl, offset: 0, index: 7)
        Dispatch.setUInt(enc, isLeft, index: 8)
    }

    // Build the table, count matches per left row, and scan the counts into offsets.
    try ctx.run { enc in
        enc.setComputePipelineState(clearPSO)
        enc.setBuffer(slots.mtl, offset: 0, index: 0)
        Dispatch.setLength(enc, tableSize, nil, index: 1)
        Dispatch.dispatch1D(enc, clearPSO, count: tableSize)
        enc.memoryBarrier(scope: .buffers)
        if nR > 0 {
            enc.setComputePipelineState(buildPSO)
            enc.setBuffer(right.values.mtl, offset: right.values.offset, index: 0)
            let rv = right.validity ?? right.values
            enc.setBuffer(rv.mtl, offset: rv.offset, index: 1)
            Dispatch.setUInt(enc, right.validity == nil ? 0 : 1, index: 2)
            Dispatch.setLength(enc, nR, nil, index: 3)
            Dispatch.setUInt(enc, mask, index: 4)
            enc.setBuffer(slots.mtl, offset: 0, index: 5)
            enc.setBuffer(next.mtl, offset: 0, index: 6)
            enc.setBuffer(errorFlag.mtl, offset: 0, index: 7)
            Dispatch.dispatch1D(enc, buildPSO, count: nR)
            enc.memoryBarrier(scope: .buffers)
        }
        enc.setComputePipelineState(countPSO)
        bindProbe(enc)
        enc.setBuffer(counts.mtl, offset: 0, index: 9)
        Dispatch.dispatch1D(enc, countPSO, count: nL)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scanBlockPSO)
        enc.setBuffer(counts.mtl, offset: 0, index: 0)
        Dispatch.setLength(enc, nL, nil, index: 1)
        enc.setBuffer(offsets.mtl, offset: 0, index: 2)
        enc.setBuffer(blockTotals.mtl, offset: 0, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scanTotalsPSO)
        enc.setBuffer(blockTotals.mtl, offset: 0, index: 0)
        Dispatch.setUInt(enc, blocks, index: 1)
        enc.setBuffer(grand.mtl, offset: 0, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scanAddPSO)
        enc.setBuffer(offsets.mtl, offset: 0, index: 0)
        enc.setBuffer(blockTotals.mtl, offset: 0, index: 1)
        Dispatch.setLength(enc, nL, nil, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
    }
    try ctx.syncPoint()      // the output size is decided by the GPU and sizes the next dispatch
    let total = withExtendedLifetime(grand) { Int(grand.typed(UInt32.self)[0]) }
    if withExtendedLifetime(errorFlag, { errorFlag.typed(UInt32.self)[0] }) != 0 {
        throw ArrowMetalError.invalidArrowArray("join: hash table insertion failed")
    }
    guard total <= Int(Int32.max) else {
        throw ArrowMetalError.invalidArrowArray("join: \(total) matches exceed the 2^31 row limit")
    }

    let outLeft = try MetalArrowBuffer.allocate(byteCount: total * 4, zeroed: false, context: ctx)
    let outRight = try MetalArrowBuffer.allocate(byteCount: total * 4, zeroed: false, context: ctx)
    // Only a left join can produce nulls; an inner join binds the (unused) 4-byte flag buffer instead.
    let rightValidBytes = kind == .left ? try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx) : errorFlag
    if total > 0 {
        try ctx.run { enc in
            enc.setComputePipelineState(writePSO)
            bindProbe(enc)
            enc.setBuffer(offsets.mtl, offset: 0, index: 9)
            enc.setBuffer(outLeft.mtl, offset: 0, index: 10)
            enc.setBuffer(outRight.mtl, offset: 0, index: 11)
            enc.setBuffer(rightValidBytes.mtl, offset: 0, index: 12)
            Dispatch.setUInt(enc, kind == .left ? 1 : 0, index: 13)
            Dispatch.dispatch1D(enc, writePSO, count: nL)
        }
    }
    for b in [slots, next, counts, offsets, blockTotals, grand, errorFlag] { ctx.retainUntilFlush(b) }
    ctx.retainUntilFlush(left); ctx.retainUntilFlush(right)

    var rightValidity: MetalArrowBuffer? = nil
    if kind == .left && total > 0 {
        ctx.retainUntilFlush(rightValidBytes)
        rightValidity = try BitmapOps.packBits(ctx, bytes: rightValidBytes, bits: total)
    }
    let li = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: outLeft, context: ctx)
    let ri = MetalArray<Int32>(length: total, nullCount: 0, validity: rightValidity, values: outRight, context: ctx)
    ri.recomputeNullCount()
    return (li, ri)
}

extension MetalRecordBatch {
    /// Equi-join with another record batch on one `int32` or `int64` column from each side.
    ///
    /// The result carries every column of this batch followed by every column of `other`, gathered with
    /// `take` from the index pairs `hashJoin` produces. A right column whose name is already taken gets a
    /// `_right` suffix. With `.left`, right columns are null on unmatched left rows.
    public func join(_ other: MetalRecordBatch, on leftKey: String, rightKey: String, kind: JoinKind) throws -> MetalRecordBatch {
        guard let lc = self[leftKey] else { throw ArrowMetalError.invalidArrowArray("no column named \(leftKey)") }
        guard let rc = other[rightKey] else { throw ArrowMetalError.invalidArrowArray("no column named \(rightKey)") }
        let leftIndices: MetalArray<Int32>, rightIndices: MetalArray<Int32>
        switch (lc, rc) {
        case (.int32(let a), .int32(let b)): (leftIndices, rightIndices) = try hashJoin(left: a, right: b, kind: kind)
        case (.int64(let a), .int64(let b)): (leftIndices, rightIndices) = try hashJoin(left: a, right: b, kind: kind)
        default:
            throw ArrowMetalError.unsupportedType("join keys \(lc.arrowFormat) / \(rc.arrowFormat): both must be int32, or both int64")
        }
        var names = self.names
        var cols = try columns.map { try $0.take(leftIndices) }
        for (i, name) in other.names.enumerated() {
            var unique = name
            if names.contains(unique) {
                unique = name + "_right"
                var k = 2
                while names.contains(unique) { unique = "\(name)_right\(k)"; k += 1 }
            }
            names.append(unique)
            cols.append(try other.columns[i].take(rightIndices))
        }
        return try MetalRecordBatch(names: names, columns: cols)
    }
}
