import Foundation
import Metal

// The per-column decode: page headers on the host, everything else on the GPU.

/// The decoded buffers of one leaf column, before they are given an Arrow type.
final class ParquetLeafData {
    let leaf: ParquetLeaf
    let file: ParquetFile
    let context: MetalContext
    /// Number of levels decoded (rows, for a flat column).
    let levels: Int
    /// Number of non-null values.
    let nonNull: Int
    /// One byte per level holding the definition level, or nil when the column cannot be null.
    var defLevels: MetalArrowBuffer?
    /// One byte per level holding the repetition level, or nil for a non-repeated column.
    var repLevels: MetalArrowBuffer?
    /// Packed Arrow validity bitmap over the levels, or nil when nothing is null.
    var validity: MetalArrowBuffer?
    var nullCount: Int = 0

    enum Values {
        /// Row-positioned fixed-width values.
        case fixed(MetalArrowBuffer, width: Int)
        /// Row-positioned packed boolean bits.
        case boolean(MetalArrowBuffer)
        /// Row-positioned int32 offsets (levels + 1 entries) and a data buffer.
        case bytes(offsets: MetalArrowBuffer, data: MetalArrowBuffer)
        /// Row-positioned int32 codes plus the merged dictionary.
        case dictionary(codes: MetalArrowBuffer, values: AnyMetalArray)
    }
    var values: Values
    /// Buffers that must outlive this object's own (page descriptors, decompressed page bytes).
    private var keepAlive: [AnyObject] = []
    func retain(_ o: AnyObject) { keepAlive.append(o) }

    init(leaf: ParquetLeaf, file: ParquetFile, context: MetalContext, levels: Int, nonNull: Int, values: Values) {
        self.leaf = leaf
        self.file = file
        self.context = context
        self.levels = levels
        self.nonNull = nonNull
        self.values = values
    }
}

extension ParquetFile {

    // MARK: - Page headers

    /// Walks a column chunk's page headers. Only Thrift headers are read here; no column bytes.
    func pageHeaders(of chunk: ParquetColumnMetadata, rowGroup: Int) throws -> (dict: ParquetRawPage?, data: [ParquetRawPage]) {
        var at = Int(chunk.startOffset)
        // `data_page_offset` and `dictionary_page_offset` are signed i64 in the footer: a corrupt one
        // is negative or past the end, and either way it must not become a read position.
        guard at >= 0, at < fileSize else {
            throw ParquetError.malformed("column chunk starts at \(at), outside the \(fileSize)-byte file")
        }
        let limit = chunk.totalCompressedSize > 0
            ? Swift.min(at + Int(chunk.totalCompressedSize), fileSize) : fileSize
        var seen: Int64 = 0
        var dict: ParquetRawPage? = nil
        var data: [ParquetRawPage] = []
        while seen < chunk.numValues && at < limit {
            var r = ThriftReader(bytes, at: at)
            let h = try ParquetPageHeader.read(&r)
            let body = r.pos
            guard h.compressedSize >= 0, body + Int(h.compressedSize) <= fileSize else {
                throw ParquetError.truncated("page at \(at) claims \(h.compressedSize) bytes")
            }
            // Every one of these is a signed thrift i32 that the decode below narrows to UInt32 to
            // build a page descriptor, and `UInt32(negative)` is a trap, not an error. A page header
            // that is negative anywhere is malformed; say so here, once, rather than on the way in.
            guard h.uncompressedSize >= 0, h.numValues >= 0, h.dictNumValues >= 0,
                  h.defLevelsByteLength >= 0, h.repLevelsByteLength >= 0,
                  Int(h.defLevelsByteLength) + Int(h.repLevelsByteLength) <= Int(h.compressedSize),
                  Int(h.defLevelsByteLength) + Int(h.repLevelsByteLength) <= Int(h.uncompressedSize) else {
                throw ParquetError.malformed(
                    "page at \(at) has a negative or inconsistent header (uncompressed \(h.uncompressedSize), "
                    + "values \(h.numValues), dict values \(h.dictNumValues), levels "
                    + "\(h.repLevelsByteLength)+\(h.defLevelsByteLength) of \(h.compressedSize))")
            }
            let page = ParquetRawPage(header: h, bodyOffset: body, rowGroup: rowGroup)
            switch h.type {
            case .dictionaryPage: dict = page
            case .dataPage, .dataPageV2:
                data.append(page)
                seen += Int64(h.numValues)
            case .indexPage: break
            }
            at = body + Int(h.compressedSize)
        }
        return (dict, data)
    }

    // MARK: - Leaf decode

    func decodeLeaf(_ leaf: ParquetLeaf, rowGroups: [Int], options: ParquetReadOptions,
                    needRepetition: Bool = false) throws -> ParquetLeafData {
        let ctx = context
        let maxDef = leaf.maxDefinition
        let maxRep = leaf.maxRepetition
        // Writers mark every column `optional` whether or not it contains nulls -- pyarrow always does --
        // so a column that is in fact dense would otherwise pay for definition levels, ranks, a scatter
        // and a validity bitmap it does not need. The footer says when that is provably not the case:
        // every selected chunk reporting `null_count == 0` (or every data page being a v2 page with
        // `num_nulls == 0`) means every definition level is the maximum, dense positions and row
        // positions coincide, and the whole level path can be skipped. Set here, used below as
        // `levelDef`; `maxDef` itself still goes to `pq_page_layout`, which must skip the level bytes.
        var provablyDense = maxDef > 0 && maxRep == 0 && !needRepetition && !rowGroups.isEmpty
        if provablyDense {
            for g in rowGroups {
                let rg = metadata.rowGroups[g]
                guard leaf.index < rg.columns.count, rg.columns[leaf.index].meta.statistics?.nullCount == 0 else {
                    provablyDense = false
                    break
                }
            }
        }

        // ---- 1. page headers
        var dataPages: [ParquetRawPage] = []
        var dictPages: [ParquetRawPage] = []
        var codecOf: [ParquetCodec] = []            // per data page
        var dictCodec: [ParquetCodec] = []          // per dictionary page
        var dictBaseOf: [Int: UInt32] = [:]         // row group -> merged dictionary base
        var dictCountOf: [Int: Int] = [:]
        var totalDict = 0
        for g in rowGroups {
            let rg = metadata.rowGroups[g]
            guard leaf.index < rg.columns.count else {
                throw ParquetError.malformed("row group \(g) has \(rg.columns.count) columns, need \(leaf.index + 1)")
            }
            let meta = rg.columns[leaf.index].meta
            if rg.columns[leaf.index].filePath != nil && !(rg.columns[leaf.index].filePath!.isEmpty) {
                throw ParquetError.unsupported("column chunks stored in a separate file")
            }
            let (d, pages) = try pageHeaders(of: meta, rowGroup: g)
            if let d {
                // Dictionary bases are 32-bit in the page descriptor, and every dictionary buffer is
                // sized `totalDict * width`: both need `totalDict` to stay inside UInt32.
                guard totalDict + Int(d.header.dictNumValues) <= Int(UInt32.max) else {
                    throw ParquetError.malformed("dictionary of \(totalDict) + \(d.header.dictNumValues) entries")
                }
                dictBaseOf[g] = UInt32(totalDict)
                dictCountOf[g] = Int(d.header.dictNumValues)
                totalDict += Int(d.header.dictNumValues)
                dictPages.append(d)
                dictCodec.append(meta.codec)
            }
            for p in pages { dataPages.append(p); codecOf.append(meta.codec) }
        }
        guard !dataPages.isEmpty else {
            return try emptyLeaf(leaf, options: options)
        }
        let totalLevels = dataPages.reduce(0) { $0 + Int($1.header.numValues) }
        try Dispatch.checkLength(totalLevels)

        // ---- 2. address every page with 32-bit offsets relative to one binding point
        let page = metalPageSize()
        var minOffset = Int.max, maxEnd = 0
        for p in dataPages + dictPages {
            minOffset = Swift.min(minOffset, p.bodyOffset)
            maxEnd = Swift.max(maxEnd, p.bodyOffset + Int(p.header.compressedSize))
        }
        let srcBase = (minOffset / page) * page
        guard maxEnd - srcBase < Int(UInt32.max) else {
            throw ParquetError.unsupported("a single column chunk spanning more than 4 GiB")
        }

        // ---- 3. page descriptors and decompression
        var infos = [ParquetPageInfo](repeating: ParquetPageInfo(), count: dataPages.count)
        var dictInfos = [ParquetPageInfo](repeating: ParquetPageInfo(), count: dictPages.count)
        let anyCompressed = codecOf.contains { $0 != .uncompressed } || dictCodec.contains { $0 != .uncompressed }

        var levelOffset = 0
        for (i, p) in dataPages.enumerated() {
            var info = ParquetPageInfo()
            info.dataLength = UInt32(p.header.uncompressedSize)
            info.numValues = UInt32(p.header.numValues)
            info.levelOffset = UInt32(levelOffset)
            info.numRows = UInt32(p.header.numRows >= 0 ? p.header.numRows : p.header.numValues)
            info.encoding = UInt32(p.header.encoding.rawValue)
            info.flags = p.header.type == .dataPageV2 ? 1 : 0
            info.dictBase = dictBaseOf[p.rowGroup] ?? 0
            if p.header.type == .dataPageV2 {
                info.repLength = UInt32(p.header.repLevelsByteLength)
                info.defLength = UInt32(p.header.defLevelsByteLength)
            }
            infos[i] = info
            levelOffset += Int(p.header.numValues)
        }
        for (i, p) in dictPages.enumerated() {
            var info = ParquetPageInfo()
            info.dataLength = UInt32(p.header.uncompressedSize)
            info.numValues = UInt32(p.header.dictNumValues)
            dictInfos[i] = info
        }

        // Only this column's byte range is wrapped for the GPU, so a projection never pays for the
        // columns it skips.
        let (mapped, mappedOffset) = try buffer(covering: srcBase..<maxEnd)
        let pageData: MTLBuffer
        let pageDataOffset: Int
        var owned: MetalArrowBuffer? = nil
        if !anyCompressed {
            // The mapped file *is* the page buffer.
            pageData = mapped.mtl
            pageDataOffset = mapped.offset + mappedOffset
            for i in infos.indices { infos[i].dataOffset = UInt32(dataPages[i].bodyOffset - srcBase) }
            for i in dictInfos.indices { dictInfos[i].dataOffset = UInt32(dictPages[i].bodyOffset - srcBase) }
        } else {
            var dst = 0
            var byCodec: [ParquetCodec: [PageBlock]] = [:]
            var copies: [PageBlock] = []
            func stage(_ p: ParquetRawPage, _ codec: ParquetCodec, _ info: inout ParquetPageInfo) {
                let uncompressed = Int(p.header.uncompressedSize)
                info.dataOffset = UInt32(dst)
                let src = UInt32(p.bodyOffset - srcBase)
                let levelBytes = p.header.type == .dataPageV2
                    ? Int(p.header.repLevelsByteLength) + Int(p.header.defLevelsByteLength) : 0
                let compressed = codec == .uncompressed || !p.header.isCompressed
                if compressed {
                    copies.append(PageBlock(srcOffset: src, srcLength: UInt32(p.header.compressedSize),
                                            dstOffset: UInt32(dst), dstLength: UInt32(uncompressed)))
                } else if levelBytes > 0 {
                    // A v2 page keeps its levels uncompressed; only the values are a compressed block.
                    copies.append(PageBlock(srcOffset: src, srcLength: UInt32(levelBytes),
                                            dstOffset: UInt32(dst), dstLength: UInt32(levelBytes)))
                    byCodec[codec, default: []].append(
                        PageBlock(srcOffset: src + UInt32(levelBytes),
                                  srcLength: UInt32(Int(p.header.compressedSize) - levelBytes),
                                  dstOffset: UInt32(dst + levelBytes),
                                  dstLength: UInt32(uncompressed - levelBytes)))
                } else {
                    byCodec[codec, default: []].append(
                        PageBlock(srcOffset: src, srcLength: UInt32(p.header.compressedSize),
                                  dstOffset: UInt32(dst), dstLength: UInt32(uncompressed)))
                }
                dst = roundUp(dst + uncompressed, to: 8)
            }
            for (i, p) in dictPages.enumerated() { stage(p, dictCodec[i], &dictInfos[i]) }
            for (i, p) in dataPages.enumerated() { stage(p, codecOf[i], &infos[i]) }
            let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(dst, 1), zeroed: false, context: ctx)
            if !copies.isEmpty {
                try Decompress.into(ctx, codec: .uncompressed, source: mapped.mtl,
                                    sourceOffset: mapped.offset + mappedOffset, blocks: copies, out: out)
            }
            for (codec, blocks) in byCodec {
                try Decompress.into(ctx, codec: codec, source: mapped.mtl,
                                    sourceOffset: mapped.offset + mappedOffset, blocks: blocks, out: out)
            }
            owned = out
            pageData = out.mtl
            pageDataOffset = out.offset
        }

        // ---- 4. page layout, then levels
        let pagesBuf = try upload(infos, ctx)
        try runLayout(ctx, data: pageData, dataOffset: pageDataOffset, pages: pagesBuf,
                      count: infos.count, maxDef: maxDef, maxRep: maxRep)
        if !dictInfos.isEmpty {
            // Dictionary pages have no levels; their value section is the whole page.
            for i in dictInfos.indices {
                dictInfos[i].valuesOffset = dictInfos[i].dataOffset
                dictInfos[i].valuesLength = dictInfos[i].dataLength
                dictInfos[i].nonNullCount = dictInfos[i].numValues
            }
        }

        // A v2 page states its null count outright, which settles the question even without statistics.
        if maxDef > 0 && maxRep == 0 && !needRepetition && !provablyDense {
            provablyDense = dataPages.allSatisfy { $0.header.type == .dataPageV2 && $0.header.numNulls == 0 }
        }
        let levelDef = provablyDense ? 0 : maxDef

        var defBytes: MetalArrowBuffer? = nil
        var ranks: MetalArrowBuffer? = nil
        var totalNonNull = totalLevels
        if levelDef > 0 {
            let db = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalLevels, 1), zeroed: false, context: ctx)
            let rk = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalLevels * 4, 4), zeroed: false, context: ctx)
            try runLevels(ctx, data: pageData, dataOffset: pageDataOffset, pages: pagesBuf, count: infos.count,
                          bitWidth: bitWidth(of: levelDef), matchLevel: levelDef, which: 0, countSlot: 0,
                          levels: db, ranks: rk)
            totalNonNull = try runPageScan(ctx, pages: pagesBuf, count: infos.count, slot: 0)
            defBytes = db
            ranks = rk
        } else {
            var acc = 0
            for i in infos.indices { infos[i].nonNullCount = infos[i].numValues; infos[i].nonNullOffset = UInt32(acc); acc += Int(infos[i].numValues) }
            // Re-upload the counts, keeping the layout the GPU just computed.
            try patchCounts(ctx, pages: pagesBuf, infos: infos)
        }
        var repBytes: MetalArrowBuffer? = nil
        if needRepetition && maxRep > 0 {
            let rb = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalLevels, 1), zeroed: true, context: ctx)
            let dummy = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
            try runLevels(ctx, data: pageData, dataOffset: pageDataOffset, pages: pagesBuf, count: infos.count,
                          bitWidth: bitWidth(of: maxRep), matchLevel: 0, which: 1, countSlot: 2,
                          levels: rb, ranks: dummy)
            repBytes = rb
        }

        // ---- 5. read the finished descriptors back so pages can be grouped by encoding
        try ctx.syncPoint()
        let finished = download(pagesBuf, count: infos.count)

        // ---- 6. dictionary
        var dictValOffset: MetalArrowBuffer? = nil
        var dictValLength: MetalArrowBuffer? = nil
        var dictFixed: MetalArrowBuffer? = nil
        let width = physicalWidth(leaf)
        if !dictPages.isEmpty {
            if leaf.physical == .byteArray {
                let vo = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalDict * 4, 4), zeroed: true, context: ctx)
                let vl = try MetalArrowBuffer.allocate(byteCount: Swift.max((totalDict + 1) * 4, 8), zeroed: true, context: ctx)
                for (i, p) in dictPages.enumerated() {
                    let base = Int(dictBaseOf[p.rowGroup] ?? 0)
                    try runDictBytesScan(ctx, data: pageData, dataOffset: pageDataOffset,
                                         start: Int(dictInfos[i].valuesOffset),
                                         end: Int(dictInfos[i].valuesOffset) + Int(dictInfos[i].valuesLength),
                                         count: dictCountOf[p.rowGroup] ?? 0, base: base, valOffset: vo, valLength: vl)
                }
                dictValOffset = vo
                dictValLength = vl
            } else {
                let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalDict * width, 1), zeroed: true, context: ctx)
                for (i, p) in dictPages.enumerated() {
                    let base = Int(dictBaseOf[p.rowGroup] ?? 0)
                    let n = (dictCountOf[p.rowGroup] ?? 0) * width
                    if n > 0 {
                        try runDictFixed(ctx, data: pageData, dataOffset: pageDataOffset,
                                         start: Int(dictInfos[i].valuesOffset), nBytes: n,
                                         dstStart: base * width, out: buf)
                    }
                }
                dictFixed = buf
            }
        }

        // ---- 7. values, one kernel per encoding family
        let decoder = ParquetValueDecoder(
            file: self, leaf: leaf, ctx: ctx, pageData: pageData, pageDataOffset: pageDataOffset,
            infos: finished, totalLevels: totalLevels, totalNonNull: totalNonNull,
            defBytes: defBytes, ranks: ranks, maxDef: levelDef, width: width,
            totalDict: totalDict, dictValOffset: dictValOffset, dictValLength: dictValLength,
            dictFixed: dictFixed, dictionaryEncoded: options.dictionaryEncoded)
        let result = try decoder.run()

        // ---- 8. validity
        let data = ParquetLeafData(leaf: leaf, file: self, context: ctx, levels: totalLevels,
                                   nonNull: totalNonNull, values: result)
        data.defLevels = defBytes
        data.repLevels = repBytes
        if levelDef > 0, totalNonNull < totalLevels {
            let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: totalLevels), 4),
                                                   zeroed: true, context: ctx)
            try runLevelsToBitmap(ctx, levels: defBytes!, n: totalLevels, maxDef: levelDef, out: bm)
            data.validity = bm
            data.nullCount = totalLevels - totalNonNull
        }
        // Keep the decompressed page buffer alive for as long as anything might still point into it.
        if let owned { data.retain(owned) }
        data.retain(mapped)
        data.retain(pagesBuf)
        return data
    }

    private func emptyLeaf(_ leaf: ParquetLeaf, options: ParquetReadOptions) throws -> ParquetLeafData {
        let ctx = context
        let width = Swift.max(physicalWidth(leaf), 1)
        if leaf.physical == .byteArray {
            let off = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)
            let dat = try MetalArrowBuffer.allocate(byteCount: 1, context: ctx)
            return ParquetLeafData(leaf: leaf, file: self, context: ctx, levels: 0, nonNull: 0,
                                   values: .bytes(offsets: off, data: dat))
        }
        let buf = try MetalArrowBuffer.allocate(byteCount: width, context: ctx)
        return ParquetLeafData(leaf: leaf, file: self, context: ctx, levels: 0, nonNull: 0, values: .fixed(buf, width: width))
    }

    /// Physical byte width of one value (1 for booleans, which are handled specially).
    func physicalWidth(_ leaf: ParquetLeaf) -> Int {
        switch leaf.physical {
        case .boolean: return 1
        case .int32, .float: return 4
        case .int64, .double: return 8
        case .int96: return 12
        case .byteArray: return 0
        case .fixedLenByteArray: return leaf.typeLength
        }
    }

    func bitWidth(of v: Int) -> Int { v <= 0 ? 0 : (Int.bitWidth - v.leadingZeroBitCount) }
}

// MARK: - Buffers and dispatch helpers

extension ParquetFile {
    func upload(_ infos: [ParquetPageInfo], _ ctx: MetalContext) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(infos.count * MemoryLayout<ParquetPageInfo>.stride, 1),
                                                zeroed: true, context: ctx)
        infos.withUnsafeBytes { memcpy(buf.mutableContents, $0.baseAddress!, $0.count) }
        return buf
    }
    func download(_ buf: MetalArrowBuffer, count: Int) -> [ParquetPageInfo] {
        var out = [ParquetPageInfo](repeating: ParquetPageInfo(), count: count)
        out.withUnsafeMutableBytes { memcpy($0.baseAddress!, buf.contents, count * MemoryLayout<ParquetPageInfo>.stride) }
        return out
    }
    /// Copies the host-known non-null counts into the GPU descriptors without disturbing the layout.
    func patchCounts(_ ctx: MetalContext, pages: MetalArrowBuffer, infos: [ParquetPageInfo]) throws {
        try ctx.syncPoint()
        let p = pages.mutableContents.assumingMemoryBound(to: ParquetPageInfo.self)
        for i in infos.indices {
            p[i].nonNullCount = infos[i].nonNullCount
            p[i].nonNullOffset = infos[i].nonNullOffset
        }
    }

    func pso(_ ctx: MetalContext, _ function: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: ParquetDecodeSource.source, function: function, cacheKey: "parquet/decode/\(function)")
    }

    func runLayout(_ ctx: MetalContext, data: MTLBuffer, dataOffset: Int, pages: MetalArrowBuffer,
                   count: Int, maxDef: Int, maxRep: Int) throws {
        let p = try pso(ctx, "pq_page_layout")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(data, offset: dataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, maxDef, index: 3)
            Dispatch.setUInt(enc, maxRep, index: 4)
            Dispatch.dispatch1D(enc, p, count: count)
        }
    }

    func runLevels(_ ctx: MetalContext, data: MTLBuffer, dataOffset: Int, pages: MetalArrowBuffer, count: Int,
                   bitWidth: Int, matchLevel: Int, which: Int, countSlot: Int,
                   levels: MetalArrowBuffer, ranks: MetalArrowBuffer) throws {
        let p = try pso(ctx, "pq_decode_levels")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(data, offset: dataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, bitWidth, index: 3)
            Dispatch.setUInt(enc, matchLevel, index: 4)
            Dispatch.setUInt(enc, which, index: 5)
            Dispatch.setUInt(enc, countSlot, index: 6)
            enc.setBuffer(levels.mtl, offset: levels.offset, index: 7)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    @discardableResult
    func runPageScan(_ ctx: MetalContext, pages: MetalArrowBuffer, count: Int, slot: Int) throws -> Int {
        let p = try pso(ctx, "pq_page_scan")
        let total = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 0)
            Dispatch.setUInt(enc, count, index: 1)
            Dispatch.setUInt(enc, slot, index: 2)
            enc.setBuffer(total.mtl, offset: total.offset, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        try ctx.syncPoint()
        return withExtendedLifetime(total) { Int(total.typed(UInt32.self)[0]) }
    }

    func runLevelsToBitmap(_ ctx: MetalContext, levels: MetalArrowBuffer, n: Int, maxDef: Int,
                           out: MetalArrowBuffer) throws {
        let p = try pso(ctx, "pq_levels_to_bitmap")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(levels.mtl, offset: levels.offset, index: 0)
            Dispatch.setUInt(enc, n, index: 1)
            Dispatch.setUInt(enc, maxDef, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
            Dispatch.dispatch1D(enc, p, count: (n + 31) / 32)
        }
    }

    func runDictBytesScan(_ ctx: MetalContext, data: MTLBuffer, dataOffset: Int, start: Int, end: Int,
                          count: Int, base: Int, valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        let p = try pso(ctx, "pq_dict_bytes_scan")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(data, offset: dataOffset, index: 0)
            Dispatch.setUInt(enc, start, index: 1)
            Dispatch.setUInt(enc, end, index: 2)
            Dispatch.setUInt(enc, count, index: 3)
            Dispatch.setUInt(enc, base, index: 4)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 5)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 6)
            Dispatch.dispatch1D(enc, p, count: 1)
        }
    }

    func runDictFixed(_ ctx: MetalContext, data: MTLBuffer, dataOffset: Int, start: Int, nBytes: Int,
                      dstStart: Int, out: MetalArrowBuffer) throws {
        let p = try pso(ctx, "pq_dict_fixed")
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(data, offset: dataOffset, index: 0)
            Dispatch.setUInt(enc, start, index: 1)
            Dispatch.setUInt(enc, nBytes, index: 2)
            Dispatch.setUInt(enc, dstStart, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
            Dispatch.dispatch1D(enc, p, count: nBytes)
        }
    }
}

extension ParquetFile {
    /// Exclusive prefix sum over `n` uint32 values. Returns the total.
    @discardableResult
    func scanU32(_ ctx: MetalContext, input: MetalArrowBuffer, output: MetalArrowBuffer, n: Int) throws -> Int {
        guard n > 0 else { return 0 }
        let perBlock = 256 * 4
        let blocks = (n + perBlock - 1) / perBlock
        let sums = try MetalArrowBuffer.allocate(byteCount: Swift.max(blocks * 4, 4), zeroed: true, context: ctx)
        let total = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        let pBlock = try pso(ctx, "pq_scan_block")
        let pSums = try pso(ctx, "pq_scan_sums")
        let pAdd = try pso(ctx, "pq_scan_add")
        try ctx.run { enc in
            enc.setComputePipelineState(pBlock)
            enc.setBuffer(input.mtl, offset: input.offset, index: 0)
            enc.setBuffer(output.mtl, offset: output.offset, index: 1)
            enc.setBuffer(sums.mtl, offset: sums.offset, index: 2)
            Dispatch.setUInt(enc, n, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pSums)
            enc.setBuffer(sums.mtl, offset: sums.offset, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(total.mtl, offset: total.offset, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pAdd)
            enc.setBuffer(output.mtl, offset: output.offset, index: 0)
            enc.setBuffer(sums.mtl, offset: sums.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            Dispatch.dispatch1D(enc, pAdd, count: n)
        }
        try ctx.syncPoint()
        return withExtendedLifetime((sums, total)) { Int(total.typed(UInt32.self)[0]) }
    }
}
