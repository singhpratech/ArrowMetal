import Foundation
import Metal
import CArrowABI

// The nested functions and layouts `Nested.swift` left out:
//
//   * `list_view` ("+vl") and `large_list_view` ("+vL") import, by turning the out-of-order
//     offsets-and-sizes pair into the contiguous int32 offsets `MetalListArray` uses. A view whose rows
//     already sit back to back keeps the producer's child untouched; anything else materialises a
//     contiguous child with one GPU gather. Both export as plain `list` ("+l") — see the note on
//     `MetalListArray`'s exporter.
//   * Arrow `list_parent_indices` and `list_slice` on `MetalListArray`.
//   * Arrow `map_lookup` on `MetalMapArray`, for utf8/binary and integer keys.

// MARK: - list_view / large_list_view

/// True for the two list-view format strings.
func isListViewFormat(_ f: String) -> Bool { f == "+vl" || f == "+vL" }

/// Imports `list_view` / `large_list_view` as a `MetalListArray`.
///
/// The view layout is three buffers — validity, per-row offsets and per-row sizes — and rows may point
/// anywhere in the child, in any order, overlapping. `MetalListArray` needs one monotonic offsets buffer,
/// so the import walks the rows once on the host: when they already lie back to back the offsets are the
/// view's own and the child is shared, and otherwise the row sizes are prefix-summed into fresh offsets
/// and the child is gathered on the GPU into that contiguous order.
func importListViewArray(format fmt: String, schema: UnsafePointer<ArrowSchema>,
                         array: UnsafeMutablePointer<ArrowArray>, context: MetalContext) throws -> ImportResult {
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    guard schema.pointee.n_children == 1, array.pointee.n_children == 1,
          let childSchema = schema.pointee.children?[0], let childArray = array.pointee.children?[0] else {
        throw ArrowMetalError.invalidArrowArray("\(fmt) expects exactly one child")
    }
    let fieldName = childSchema.pointee.name.map { String(cString: $0) } ?? "item"
    // Move the child out first so its lifetime is independent of the parent.
    let movedChild = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
    movedChild.initialize(to: childArray.pointee)
    childArray.pointee.release = nil
    defer { movedChild.deallocate() }
    let childResult = try importArrowArray(schema: childSchema, array: movedChild, context: context)

    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    guard a.n_buffers == 3, a.buffers != nil else {
        throw ArrowMetalError.invalidArrowArray("\(fmt) arrays have three buffers, got \(a.n_buffers)")
    }
    func buffer(_ i: Int) -> UnsafeRawPointer? { a.buffers[i].map { UnsafeRawPointer($0) } }
    var validity: MetalArrowBuffer? = nil
    if let vp = buffer(0) {
        let bytes = Swift.max(Bitmap.byteCount(bits: length), 1)
        let buf = try MetalArrowBuffer.allocate(byteCount: bytes, context: context)
        let sp = vp.assumingMemoryBound(to: UInt8.self)
        let dp = buf.mutableTyped(UInt8.self)
        if offset % 8 == 0 { memcpy(dp, sp + offset / 8, bytes) }
        else { for i in 0..<length where Bitmap.isSet(sp, i + offset) { Bitmap.set(dp, i) } }
        validity = buf
    }
    guard let offPtr = buffer(1), let sizePtr = buffer(2) else {
        throw ArrowMetalError.invalidArrowArray("\(fmt) offsets or sizes buffer is null")
    }
    let large = fmt == "+vL"
    let childLength = childResult.array.length

    // Effective (offset, size) per row: a null row references nothing, whatever its slots say.
    var offs = [Int32](repeating: 0, count: length)
    var sizes = [Int32](repeating: 0, count: length)
    for i in 0..<length {
        let valid = validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true
        guard valid else { continue }
        let o: Int64, s: Int64
        if large {
            o = offPtr.assumingMemoryBound(to: Int64.self)[offset + i]
            s = sizePtr.assumingMemoryBound(to: Int64.self)[offset + i]
        } else {
            o = Int64(offPtr.assumingMemoryBound(to: Int32.self)[offset + i])
            s = Int64(sizePtr.assumingMemoryBound(to: Int32.self)[offset + i])
        }
        guard o >= 0, s >= 0, o + s <= Int64(childLength), o + s <= Int64(Int32.max) else {
            throw ArrowMetalError.invalidArrowArray("\(fmt) row \(i) references \(o)..<\(o + s) of a child with \(childLength) elements")
        }
        offs[i] = Int32(o); sizes[i] = Int32(s)
    }

    var contiguous = true
    var run: Int32 = length > 0 ? offs[0] : 0
    for i in 0..<length {
        if offs[i] != run { contiguous = false; break }
        run += sizes[i]
    }

    let outOffsets = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
    let op = outOffsets.mutableTyped(Int32.self)
    if contiguous {
        var pos: Int32 = length > 0 ? offs[0] : 0
        for i in 0..<length { op[i] = pos; pos += sizes[i] }
        op[length] = pos
        let list = MetalListArray(length: length, nullCount: 0, validity: validity, offsets: outOffsets,
                                  values: childResult.array, kind: .variable, fieldName: fieldName, context: context)
        list.recomputeNullCount()
        return ImportResult(array: .list(list), zeroCopy: false)
    }
    // Non-contiguous: prefix-sum the sizes and gather the child into that order on the GPU.
    var pos: Int32 = 0
    for i in 0..<length { op[i] = pos; pos = pos &+ sizes[i] }
    op[length] = pos
    let total = Int(pos)
    let viewOffsets = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 4, 1), zeroed: false, context: context)
    let viewSizes = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 4, 1), zeroed: false, context: context)
    let vop = viewOffsets.mutableTyped(Int32.self), vsp = viewSizes.mutableTyped(Int32.self)
    for i in 0..<length { vop[i] = offs[i]; vsp[i] = sizes[i] }
    let childIdxBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * 4, 1), zeroed: true, context: context)
    if length > 0 {
        let pso = try context.pipeline(source: NestedExtraSource.lists, function: "list_view_gather",
                                       cacheKey: "nestedextra/list_view_gather")
        try context.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(viewOffsets.mtl, offset: viewOffsets.offset, index: 0)
            enc.setBuffer(viewSizes.mtl, offset: viewSizes.offset, index: 1)
            enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 2)
            Dispatch.setLength(enc, length, nil, index: 3)
            enc.setBuffer(childIdxBuf.mtl, offset: childIdxBuf.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: length)
        }
    }
    let childIdx = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: childIdxBuf, context: context)
    let newChild = try childResult.array.take(childIdx)
    let list = MetalListArray(length: length, nullCount: 0, validity: validity, offsets: outOffsets,
                              values: newChild, kind: .variable, fieldName: fieldName, context: context)
    list.recomputeNullCount()
    return ImportResult(array: .list(list), zeroCopy: false)
}

// MARK: - list_parent_indices / list_slice

extension MetalListArray {
    private func extraPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: NestedExtraSource.lists, function: fn, cacheKey: "nestedextra/\(fn)")
    }

    /// Arrow `list_parent_indices`: for every child element this list references — the same range
    /// `listFlatten()` returns, `offsets[0] ..< offsets[length]` — the index of the row that covers it.
    ///
    /// GPU: one binary search over the offsets per output element, so empty and null rows cost nothing.
    /// The result never has nulls. A child element under a null row is only produced when the producer
    /// left that row's offsets spanning a range, which pyarrow does not do.
    public func listParentIndices() throws -> MetalArray<Int32> {
        let r = childRange
        let m = r.count
        try Dispatch.checkLength(m)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(m * 4, 1), zeroed: true, context: ctx)
        if m > 0 && length > 0 {
            let pso = try extraPipeline("list_parent_indices")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                Dispatch.setUInt(enc, length, index: 1)
                Dispatch.setLength(enc, m, nil, index: 2)
                var base = Int32(r.lowerBound)
                enc.setBytes(&base, length: 4, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: m)
            }
        }
        return MetalArray<Int32>(length: m, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// Arrow `list_parent_indices` in **int64**, which is the width pyarrow returns.
    ///
    /// The same GPU binary search, widened by the GPU cast. `listParentIndices()` keeps the int32 form,
    /// which is what the list offsets themselves are and what every caller inside this package wants.
    public func listParentIndices64() throws -> MetalArray<Int64> {
        try listParentIndices().cast(to: Int64.self)
    }

    /// Arrow `list_slice`: `row[start:stop:step]` for every row.
    ///
    /// `stop == nil` slices to the end of each row. `start` and `step` must be non-negative and `step` at
    /// least 1, as Arrow requires; a row shorter than `start` becomes empty and a null row stays null.
    /// GPU: one kernel computes the new row lengths, the existing scan turns them into offsets, and a
    /// second kernel expands the per-row index ranges for the child's own `take`. The result is always a
    /// variable-length list (`+l`), never a fixed-size one.
    public func listSlice(start: Int, stop: Int? = nil, step: Int = 1) throws -> MetalListArray {
        guard start >= 0 else { throw ArrowMetalError.invalidArrowArray("list_slice start must be >= 0, got \(start)") }
        guard step >= 1 else { throw ArrowMetalError.invalidArrowArray("list_slice step must be >= 1, got \(step)") }
        if let s = stop, s < 0 { throw ArrowMetalError.invalidArrowArray("list_slice stop must be >= 0, got \(s)") }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try extraPipeline("list_slice_lengths")
            let v = validity ?? offsets
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                var s = Int32(clamping: start), e = Int32(stop.map { Int32(clamping: $0) } ?? -1), st = Int32(clamping: step)
                enc.setBytes(&s, length: 4, index: 4)
                enc.setBytes(&e, length: 4, index: 5)
                enc.setBytes(&st, length: 4, index: 6)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let lenArray = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
        let outOffsets = try lenArray.exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let idxBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * 4, 1), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try extraPipeline("list_slice_gather")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                var s = Int32(clamping: start), st = Int32(clamping: step)
                enc.setBytes(&s, length: 4, index: 3)
                enc.setBytes(&st, length: 4, index: 4)
                enc.setBuffer(idxBuf.mtl, offset: idxBuf.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let idx = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: idxBuf, context: ctx)
        let newValues = try values.take(idx)
        let out = MetalListArray(length: n, nullCount: 0, validity: validity, offsets: outOffsets,
                                 values: newValues, kind: .variable, fieldName: fieldName, context: ctx)
        out.recomputeNullCount()
        return out
    }
}

// MARK: - map_lookup

/// Which matching entry Arrow's `map_lookup` returns.
public enum MapLookupOccurrence: Int, Sendable, CaseIterable {
    case first = 0, last = 1, all = 2
}

/// A `map_lookup` key: utf8/binary bytes, or an integer widened to int64.
public enum MapLookupKey: Sendable {
    case bytes([UInt8])
    case integer(Int64)

    public static func string(_ s: String) -> MapLookupKey { .bytes(Array(s.utf8)) }
}

extension MetalMapArray {
    /// Arrow `map_lookup`: the value(s) whose key matches, per row.
    ///
    /// `first` / `last` return an array of the map's item type — null where the row is null or the key is
    /// absent. `all` returns a `list` of the item type, null on the same rows, so an empty list never
    /// stands for "not found" (this is what pyarrow returns too).
    ///
    /// GPU: one kernel scans every row's entry range and reports the first match, the last match and the
    /// match count; `all` then scans the counts into offsets and a second kernel writes each row's
    /// matching entry indices, which the item array's own `take` gathers. Keys must be utf8 / binary or
    /// an integer type (widened to int64 with the existing GPU cast); float and nested keys are rejected.
    public func mapLookup(_ key: MapLookupKey, occurrence: MapLookupOccurrence = .first) throws -> AnyMetalArray {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let keys = self.keys

        // Bind the key column and the pattern in the shape the kernel wants.
        var kind = "int"
        var keyBuf0: MetalArrowBuffer          // utf8 offsets, or the int64 values
        var keyBuf1: MetalArrowBuffer          // utf8 data, or a filler
        var patBuf: MetalArrowBuffer
        var patLen = 0
        switch (keys, key) {
        case (.string(let s), .bytes(let b)), (.binary(let s), .bytes(let b)):
            kind = "str"
            keyBuf0 = s.offsets; keyBuf1 = s.data
            patBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(b.count, 1), context: ctx)
            let p = patBuf.mutableTyped(UInt8.self)
            for (i, x) in b.enumerated() { p[i] = x }
            patLen = b.count
        case (.string, _), (.binary, _):
            throw ArrowMetalError.unsupportedType("this map has utf8 keys; look up a string, not an integer")
        default:
            guard case .integer(let v) = key else {
                throw ArrowMetalError.unsupportedType("this map has \(keys.arrowFormat) keys; look up an integer, not a string")
            }
            let wide = try widenKeysToInt64(keys)
            keyBuf0 = wide.values
            keyBuf1 = wide.values
            patBuf = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)
            patBuf.mutableTyped(Int64.self)[0] = v
            patLen = 1
            // Keep the widened copy alive for as long as the dispatch needs it.
            ctx.retainUntilFlush(wide)
        }

        let source = NestedExtraSource.mapLookup(K: kind)
        let list = entries
        let firstBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), zeroed: true, context: ctx)
        let lastBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), zeroed: true, context: ctx)
        let countBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: source, function: "map_lookup_scan", cacheKey: "map/\(kind)/scan")
            let v = list.validity ?? list.offsets
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(list.offsets.mtl, offset: list.offsets.offset, index: 0)
                enc.setBuffer(keyBuf0.mtl, offset: keyBuf0.offset, index: 1)
                enc.setBuffer(keyBuf1.mtl, offset: keyBuf1.offset, index: 2)
                enc.setBuffer(patBuf.mtl, offset: patBuf.offset, index: 3)
                Dispatch.setUInt(enc, patLen, index: 4)
                enc.setBuffer(v.mtl, offset: v.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                Dispatch.setUInt(enc, list.validity == nil ? 0 : 1, index: 7)
                enc.setBuffer(firstBuf.mtl, offset: firstBuf.offset, index: 8)
                enc.setBuffer(lastBuf.mtl, offset: lastBuf.offset, index: 9)
                enc.setBuffer(countBuf.mtl, offset: countBuf.offset, index: 10)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        ctx.retainUntilFlush(patBuf)
        try ctx.syncPoint()

        let items = self.items
        if occurrence != .all {
            let raw = occurrence == .first ? firstBuf : lastBuf
            // A -1 slot means "no match": make it a null index so `take` produces a null value.
            let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: ctx)
            let rp = raw.mutableTyped(Int32.self), bp = bm.mutableTyped(UInt8.self)
            var nulls = 0
            for i in 0..<n {
                if rp[i] >= 0 { Bitmap.set(bp, i) } else { rp[i] = 0; nulls += 1 }
            }
            let idx = MetalArray<Int32>(length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm,
                                        values: raw, context: ctx)
            return try items.take(idx)
        }

        let counts = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: countBuf, context: ctx)
        let outOffsets = try counts.exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let idxBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * 4, 1), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: source, function: "map_lookup_gather", cacheKey: "map/\(kind)/gather")
            let v = list.validity ?? list.offsets
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(list.offsets.mtl, offset: list.offsets.offset, index: 0)
                enc.setBuffer(keyBuf0.mtl, offset: keyBuf0.offset, index: 1)
                enc.setBuffer(keyBuf1.mtl, offset: keyBuf1.offset, index: 2)
                enc.setBuffer(patBuf.mtl, offset: patBuf.offset, index: 3)
                Dispatch.setUInt(enc, patLen, index: 4)
                enc.setBuffer(v.mtl, offset: v.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                Dispatch.setUInt(enc, list.validity == nil ? 0 : 1, index: 7)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 8)
                enc.setBuffer(idxBuf.mtl, offset: idxBuf.offset, index: 9)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        ctx.retainUntilFlush(patBuf)
        try ctx.syncPoint()
        let idx = MetalArray<Int32>(length: total, nullCount: 0, validity: nil, values: idxBuf, context: ctx)
        let child = try items.take(idx)
        // A row with no match is a null list, matching what `first` / `last` return for the same row.
        let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: ctx)
        let cp = countBuf.typed(Int32.self), bp = bm.mutableTyped(UInt8.self)
        var nulls = 0
        for i in 0..<n { if cp[i] > 0 { Bitmap.set(bp, i) } else { nulls += 1 } }
        let out = MetalListArray(length: n, nullCount: nulls, validity: nulls == 0 ? nil : bm,
                                 offsets: outOffsets, values: child, kind: .variable, fieldName: "item", context: ctx)
        return .list(out)
    }

    /// Widens an integer key column to int64 with the existing GPU cast.
    private func widenKeysToInt64(_ keys: AnyMetalArray) throws -> MetalArray<Int64> {
        switch keys {
        case .int8(let a): return try a.cast(to: Int64.self)
        case .uint8(let a): return try a.cast(to: Int64.self)
        case .int16(let a): return try a.cast(to: Int64.self)
        case .uint16(let a): return try a.cast(to: Int64.self)
        case .int32(let a): return try a.cast(to: Int64.self)
        case .uint32(let a): return try a.cast(to: Int64.self)
        case .int64(let a): return a
        case .uint64(let a): return try a.cast(to: Int64.self)
        case .temporal(let t): return try t.int64Values()
        default:
            throw ArrowMetalError.unsupportedType("map_lookup keys must be utf8, binary or an integer type, got \(keys.arrowFormat)")
        }
    }
}

extension AnyMetalArray {
    /// Arrow `list_parent_indices` on a list or map column.
    public func listParentIndices() throws -> MetalArray<Int32> {
        switch self {
        case .list(let l): return try l.listParentIndices()
        case .map(let m): return try m.entries.listParentIndices()
        default: throw ArrowMetalError.unsupportedType("list_parent_indices needs a list array, got \(arrowFormat)")
        }
    }

    /// The same in int64, which is what pyarrow's `list_parent_indices` returns.
    public func listParentIndices64() throws -> MetalArray<Int64> {
        try listParentIndices().cast(to: Int64.self)
    }
    /// Arrow `list_slice` on a list or map column (a map's entries are sliced as a list).
    public func listSlice(start: Int, stop: Int? = nil, step: Int = 1) throws -> AnyMetalArray {
        switch self {
        case .list(let l): return .list(try l.listSlice(start: start, stop: stop, step: step))
        case .map(let m): return .list(try m.entries.listSlice(start: start, stop: stop, step: step))
        default: throw ArrowMetalError.unsupportedType("list_slice needs a list array, got \(arrowFormat)")
        }
    }
    /// Arrow `map_lookup` on a map column.
    public func mapLookup(_ key: MapLookupKey, occurrence: MapLookupOccurrence = .first) throws -> AnyMetalArray {
        guard case .map(let m) = self else {
            throw ArrowMetalError.unsupportedType("map_lookup needs a map array, got \(arrowFormat)")
        }
        return try m.mapLookup(key, occurrence: occurrence)
    }
}
