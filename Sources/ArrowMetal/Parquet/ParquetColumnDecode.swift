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
    /// The rows (row group, row-group-coordinate range) the levels cover, in order. For a repeated
    /// column these are rows, not level entries.
    var rowSpans: [(group: Int, rows: Range<Int>)] = []

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

/// A leaf column's pages as a read selects them (`ParquetFile.collectPages`).
struct ParquetLeafPages {
    var dataPages: [ParquetRawPage] = []
    var dictPages: [ParquetRawPage] = []
    var codecOf: [ParquetCodec] = []            // per data page
    var dictCodec: [ParquetCodec] = []          // per dictionary page
    var dictBaseOf: [Int: UInt32] = [:]         // row group -> merged dictionary base
    var dictCountOf: [Int: Int] = [:]
    var totalDict = 0
    var spans: [(group: Int, rows: Range<Int>)] = []
    var skippedPages = 0
}

/// Where a column's pages go in its staging buffer, and the blocks that put them there
/// (`ParquetFile.stagePages`).
struct ParquetStaging {
    var size = 0
    var dataOffsets: [UInt32] = []
    var dictOffsets: [UInt32] = []
    var copies: [PageBlock] = []
    var byCodec: [ParquetCodec: [PageBlock]] = [:]
    var dictByCodec: [ParquetCodec: [PageBlock]] = [:]
}

/// A column's pages already staged by `ParquetFile.stageTogether`: `size` bytes at `base` in `buffer`.
struct ParquetPreStaged {
    let buffer: MetalArrowBuffer
    let base: Int
    let size: Int
    let pages: Int
}

extension ParquetFile {
    /// The most a batch of `stageTogether` stages at once: an eighth of the device's recommended
    /// working set, at most 3 GiB (the decoders' 32-bit offsets allow under 4).
    var maxStagingBytes: Int { Swift.min(Int(context.device.recommendedMaxWorkingSetSize) / 8, 3 << 30) }

    // MARK: - Page headers

    /// A column chunk's page headers: from the handle's cache when an earlier read (or this read's
    /// `prefetchPageHeaders`) parsed them, else parsed now and cached.
    func pageHeaders(of chunk: ParquetColumnMetadata, rowGroup: Int) throws -> (dict: ParquetRawPage?, data: [ParquetRawPage]) {
        let key = ParquetChunkKey(chunk, rowGroup: rowGroup)
        if let hit = cachedPageHeaders(key) { return hit }
        let parsed = try parsePageHeaders(of: chunk, rowGroup: rowGroup)
        cachePageHeaders(key, parsed)
        return parsed
    }

    /// Parses the page headers of every chunk of `leaves` over `rowGroups` that is not cached yet,
    /// spread across cores. Each header sits in its own page of the mapping, so a cold read of a large
    /// file takes a minor fault per data page here; done one column at a time that was 13-14 ms of a
    /// 2 GB, 8-column read. A chunk that fails to parse is left out, and the decode parses it again and
    /// raises exactly as it would have.
    func prefetchPageHeaders(leaves: [ParquetLeaf], rowGroups: [Int]) {
        var todo: [(ParquetColumnMetadata, Int)] = []
        for leaf in leaves {
            for g in rowGroups where g >= 0 && g < metadata.rowGroups.count {
                let rg = metadata.rowGroups[g]
                guard leaf.index < rg.columns.count else { continue }
                let meta = rg.columns[leaf.index].meta
                if cachedPageHeaders(ParquetChunkKey(meta, rowGroup: g)) == nil { todo.append((meta, g)) }
            }
        }
        guard todo.count > 1 else { return }
        DispatchQueue.concurrentPerform(iterations: todo.count) { i in
            if let parsed = try? parsePageHeaders(of: todo[i].0, rowGroup: todo[i].1) {
                cachePageHeaders(ParquetChunkKey(todo[i].0, rowGroup: todo[i].1), parsed)
            }
        }
        ParquetProfile.lap("read.headers \(todo.count) chunks")
    }

    /// Walks a column chunk's page headers. Only Thrift headers are read here; no column bytes.
    func parsePageHeaders(of chunk: ParquetColumnMetadata, rowGroup: Int) throws -> (dict: ParquetRawPage?, data: [ParquetRawPage]) {
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
            try validate(h, at: at, body: body)
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

    /// Rejects a page header that would overflow or trap later on.
    func validate(_ h: ParquetPageHeader, at: Int, body: Int) throws {
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
    }

    // MARK: - Leaf decode

    /// Decodes one leaf column over `rowGroups`. With a `plan`, the decoded and skipped pages are counted
    /// into it, and with `subset` a flat column decodes only the data pages its offset index says overlap
    /// the plan's candidate rows (`ParquetPageIndex.swift`); `rowSpans` then says which rows came back.
    func decodeLeaf(_ leaf: ParquetLeaf, rowGroups: [Int], options: ParquetReadOptions,
                    needRepetition: Bool = false, plan: ParquetReadPlan? = nil,
                    subset: Bool = false, staged: [Int: ParquetPreStaged]? = nil) throws -> ParquetLeafData {
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
        let lp = try collectPages(leaf, rowGroups: rowGroups, plan: plan, subset: subset)
        let dataPages = lp.dataPages, dictPages = lp.dictPages, codecOf = lp.codecOf, dictCodec = lp.dictCodec
        let dictBaseOf = lp.dictBaseOf, dictCountOf = lp.dictCountOf, totalDict = lp.totalDict
        let spans = lp.spans
        plan?.count(decoded: dataPages.count, skipped: lp.skippedPages)
        ParquetProfile.lap("col.headers", sync: ctx)
        guard !dataPages.isEmpty else {
            let empty = try emptyLeaf(leaf, options: options)
            empty.rowSpans = spans
            return empty
        }
        let totalLevels = dataPages.reduce(0) { $0 + Int($1.header.numValues) }
        try Dispatch.checkLength(totalLevels)

        // ---- 2. address every page with 32-bit offsets relative to one binding point. Only this
        // column's page bytes are handed to the GPU, so a projection never pays for the columns it skips.
        let source = try pageSource(covering: (dataPages + dictPages).map {
            $0.bodyOffset..<($0.bodyOffset + Int($0.header.compressedSize))
        })
        func rel(_ p: ParquetRawPage) -> UInt32 { UInt32(source.offset(ofFile: p.bodyOffset)) }

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

        let mapped = source.buffer
        let pageData: MTLBuffer
        let pageDataOffset: Int
        var owned: MetalArrowBuffer? = nil
        if !anyCompressed {
            // The mapped file *is* the page buffer.
            pageData = mapped.mtl
            pageDataOffset = source.bindingOffset
            for i in infos.indices { infos[i].dataOffset = rel(dataPages[i]) }
            for i in dictInfos.indices { dictInfos[i].dataOffset = rel(dictPages[i]) }
        } else {
            let st = stagePages(lp, rel: rel)
            for i in infos.indices { infos[i].dataOffset = st.dataOffsets[i] }
            for i in dictInfos.indices { dictInfos[i].dataOffset = st.dictOffsets[i] }
            if let pre = staged?[leaf.index], pre.size == st.size, pre.pages == dataPages.count + dictPages.count {
                // `stageTogether` already decompressed these pages, with every other column of the read.
                owned = pre.buffer
                pageData = pre.buffer.mtl
                pageDataOffset = pre.buffer.offset + pre.base
            } else {
                ParquetProfile.lap("col.stage")
                let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(st.size, 1), zeroed: false, context: ctx)
                ParquetProfile.lap("col.alloc-pagebuf")
                try runStaging([(st, 0, 0)], source: mapped.mtl, sourceOffset: source.bindingOffset, out: out)
                ParquetProfile.lap("col.decompress \(leaf.name) \(st.byCodec.values.reduce(0) { $0 + $1.count }) pages", sync: ctx)
                owned = out
                pageData = out.mtl
                pageDataOffset = out.offset
            }
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
            // `pq_decode_levels` writes a rank for every level it decodes, so the scratch rank buffer
            // needs a slot per level. (It was once 4 bytes, which is fine up to the 16 KB allocation
            // padding -- 4,096 levels -- and past that wrote over whatever memory followed.)
            let scratchRanks = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalLevels * 4, 4), zeroed: false, context: ctx)
            try runLevels(ctx, data: pageData, dataOffset: pageDataOffset, pages: pagesBuf, count: infos.count,
                          bitWidth: bitWidth(of: maxRep), matchLevel: 0, which: 1, countSlot: 2,
                          levels: rb, ranks: scratchRanks)
            repBytes = rb
        }

        // ---- 5. read the finished descriptors back so pages can be grouped by encoding
        try ctx.syncPoint()
        ParquetProfile.lap("col.layout+levels")
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

        ParquetProfile.lap("col.dict", sync: ctx)
        // ---- 7. values, one kernel per encoding family
        let decoder = ParquetValueDecoder(
            file: self, leaf: leaf, ctx: ctx, pageData: pageData, pageDataOffset: pageDataOffset,
            infos: finished, totalLevels: totalLevels, totalNonNull: totalNonNull,
            defBytes: defBytes, ranks: ranks, maxDef: levelDef, width: width,
            totalDict: totalDict, dictValOffset: dictValOffset, dictValLength: dictValLength,
            dictFixed: dictFixed, dictionaryEncoded: options.dictionaryEncoded)
        let result = try decoder.run()
        ParquetProfile.lap("col.values", sync: ctx)

        // ---- 8. validity
        let data = ParquetLeafData(leaf: leaf, file: self, context: ctx, levels: totalLevels,
                                   nonNull: totalNonNull, values: result)
        data.defLevels = defBytes
        data.repLevels = repBytes
        data.rowSpans = spans
        if levelDef > 0, totalNonNull < totalLevels {
            let bm = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: totalLevels), 4),
                                                   zeroed: true, context: ctx)
            try runLevelsToBitmap(ctx, levels: defBytes!, n: totalLevels, maxDef: levelDef, out: bm)
            data.validity = bm
            data.nullCount = totalLevels - totalNonNull
        }
        ParquetProfile.lap("col.validity", sync: ctx)
        // Keep the decompressed page buffer alive for as long as anything might still point into it.
        if let owned { data.retain(owned) }
        data.retain(mapped)
        data.retain(pagesBuf)
        return data
    }

    /// Step 1 of `decodeLeaf`: the dictionary and data pages of `leaf` over `rowGroups`, from their
    /// headers (and, with `subset` and a `plan`, the offset index). Counts nothing into the plan.
    func collectPages(_ leaf: ParquetLeaf, rowGroups: [Int], plan: ParquetReadPlan?, subset: Bool) throws -> ParquetLeafPages {
        var lp = ParquetLeafPages()
        for g in rowGroups {
            let rg = metadata.rowGroups[g]
            guard leaf.index < rg.columns.count else {
                throw ParquetError.malformed("row group \(g) has \(rg.columns.count) columns, need \(leaf.index + 1)")
            }
            let meta = rg.columns[leaf.index].meta
            if rg.columns[leaf.index].filePath != nil && !(rg.columns[leaf.index].filePath!.isEmpty) {
                throw ParquetError.unsupported("column chunks stored in a separate file")
            }
            let d: ParquetRawPage?
            let pages: [ParquetRawPage]
            if subset, leaf.maxRepetition == 0, let ranges = plan?.ranges[g],
               let found = try indexedPages(of: meta, rowGroup: g, column: leaf.index, ranges: ranges) {
                (d, pages) = (found.dict, found.data)
                lp.spans.append(contentsOf: found.spans.map { (g, $0) })
                lp.skippedPages += found.skipped
            } else {
                (d, pages) = try pageHeaders(of: meta, rowGroup: g)
                lp.spans.append((g, 0..<rowsIn(group: g)))
            }
            if let d {
                // Dictionary bases are 32-bit in the page descriptor, and every dictionary buffer is
                // sized `totalDict * width`: both need `totalDict` to stay inside UInt32.
                guard lp.totalDict + Int(d.header.dictNumValues) <= Int(UInt32.max) else {
                    throw ParquetError.malformed("dictionary of \(lp.totalDict) + \(d.header.dictNumValues) entries")
                }
                lp.dictBaseOf[g] = UInt32(lp.totalDict)
                lp.dictCountOf[g] = Int(d.header.dictNumValues)
                lp.totalDict += Int(d.header.dictNumValues)
                lp.dictPages.append(d)
                lp.dictCodec.append(meta.codec)
            }
            for p in pages { lp.dataPages.append(p); lp.codecOf.append(meta.codec) }
        }
        return lp
    }

    /// Where every page of `lp` goes in a staging buffer (8-byte aligned, dictionary pages first) and
    /// the blocks that put it there: plain copies, dictionary pages (host-decoded for SNAPPY) and the
    /// compressed data pages by codec. `rel` gives a page's offset from the source binding point.
    func stagePages(_ lp: ParquetLeafPages, rel: (ParquetRawPage) -> UInt32) -> ParquetStaging {
        var st = ParquetStaging()
        var dst = 0
        func stage(_ p: ParquetRawPage, _ codec: ParquetCodec, dictionary: Bool) -> UInt32 {
            let uncompressed = Int(p.header.uncompressedSize)
            let at = UInt32(dst)
            let src = rel(p)
            let levelBytes = p.header.type == .dataPageV2
                ? Int(p.header.repLevelsByteLength) + Int(p.header.defLevelsByteLength) : 0
            let compressed = codec == .uncompressed || !p.header.isCompressed
            if compressed {
                st.copies.append(PageBlock(srcOffset: src, srcLength: UInt32(p.header.compressedSize),
                                           dstOffset: UInt32(dst), dstLength: UInt32(uncompressed)))
            } else if levelBytes > 0 {
                // A v2 page keeps its levels uncompressed; only the values are a compressed block.
                st.copies.append(PageBlock(srcOffset: src, srcLength: UInt32(levelBytes),
                                           dstOffset: UInt32(dst), dstLength: UInt32(levelBytes)))
                st.byCodec[codec, default: []].append(
                    PageBlock(srcOffset: src + UInt32(levelBytes),
                              srcLength: UInt32(Int(p.header.compressedSize) - levelBytes),
                              dstOffset: UInt32(dst + levelBytes),
                              dstLength: UInt32(uncompressed - levelBytes)))
            } else if dictionary {
                st.dictByCodec[codec, default: []].append(
                    PageBlock(srcOffset: src, srcLength: UInt32(p.header.compressedSize),
                              dstOffset: UInt32(dst), dstLength: UInt32(uncompressed)))
            } else {
                st.byCodec[codec, default: []].append(
                    PageBlock(srcOffset: src, srcLength: UInt32(p.header.compressedSize),
                              dstOffset: UInt32(dst), dstLength: UInt32(uncompressed)))
            }
            dst = roundUp(dst + uncompressed, to: 8)
            return at
        }
        st.dictOffsets = lp.dictPages.enumerated().map { stage($0.element, lp.dictCodec[$0.offset], dictionary: true) }
        st.dataOffsets = lp.dataPages.enumerated().map { stage($0.element, lp.codecOf[$0.offset], dictionary: false) }
        st.size = dst
        return st
    }

    /// Runs the blocks of several stagings into `out`, each part's source offsets moved by `srcShift`
    /// (to the common binding point `sourceOffset`) and its destinations by `dstShift`: the copies,
    /// then the dictionary pages, then the data pages, one dispatch (or host pass) per codec for all
    /// parts together.
    func runStaging(_ parts: [(staging: ParquetStaging, srcShift: Int, dstShift: Int)], source: MTLBuffer,
                    sourceOffset: Int, out: MetalArrowBuffer) throws {
        func moved(_ bs: [PageBlock], _ s: Int, _ d: Int) -> [PageBlock] {
            (s == 0 && d == 0) ? bs : bs.map {
                PageBlock(srcOffset: UInt32(Int($0.srcOffset) + s), srcLength: $0.srcLength,
                          dstOffset: UInt32(Int($0.dstOffset) + d), dstLength: $0.dstLength)
            }
        }
        var copies: [PageBlock] = []
        var dict: [ParquetCodec: [PageBlock]] = [:]
        var data: [ParquetCodec: [PageBlock]] = [:]
        for p in parts {
            copies += moved(p.staging.copies, p.srcShift, p.dstShift)
            for (c, b) in p.staging.dictByCodec { dict[c, default: []] += moved(b, p.srcShift, p.dstShift) }
            for (c, b) in p.staging.byCodec { data[c, default: []] += moved(b, p.srcShift, p.dstShift) }
        }
        let ctx = context
        if !copies.isEmpty {
            try Decompress.into(ctx, codec: .uncompressed, source: source, sourceOffset: sourceOffset, blocks: copies, out: out)
        }
        for (codec, blocks) in dict {
            try Decompress.into(ctx, codec: codec, source: source, sourceOffset: sourceOffset, blocks: blocks,
                                out: out, preferHost: true)
        }
        for (codec, blocks) in data {
            try Decompress.into(ctx, codec: codec, source: source, sourceOffset: sourceOffset, blocks: blocks, out: out)
        }
    }

    /// Decompresses the pages of several flat columns of one read together, before any of them is
    /// decoded, into one staging buffer per batch; `decodeLeaf` then finds its pages staged.
    ///
    /// A column of token-dense Snappy or LZ4 pages is one GPU dispatch of a few thousand pages, and the
    /// page-per-thread kernel gets only a couple of SIMD groups per core out of that, waiting on memory
    /// most of the time; two or three such columns in one dispatch take about as long as one. So every
    /// GPU-compressed column the read decodes in full is staged here, when their pages come from one
    /// mapping. Batches are capped (`maxStagingBytes`), and anything unusual -- a header that does not
    /// parse, a failed decompression -- drops the whole thing and leaves every column to stage its own
    /// pages as before, so errors come out exactly as they did.
    func stageTogether(leaves: [ParquetLeaf], rowGroups: [Int]) -> [Int: ParquetPreStaged] {
        var items: [(leaf: ParquetLeaf, lp: ParquetLeafPages, source: ParquetPageSource, st: ParquetStaging)] = []
        var seen = Set<Int>()
        for leaf in leaves where leaf.maxRepetition == 0 && seen.insert(leaf.index).inserted {
            guard let lp = try? collectPages(leaf, rowGroups: rowGroups, plan: nil, subset: false) else { return [:] }
            guard !lp.dataPages.isEmpty,
                  lp.codecOf.contains(where: { $0 == .snappy || $0 == .lz4 || $0 == .lz4Raw }) else { continue }
            let ranges = (lp.dataPages + lp.dictPages).map { $0.bodyOffset..<($0.bodyOffset + Int($0.header.compressedSize)) }
            guard let source = try? pageSource(covering: ranges) else { return [:] }
            let st = stagePages(lp, rel: { UInt32(source.offset(ofFile: $0.bodyOffset)) })
            items.append((leaf, lp, source, st))
        }
        guard items.count >= 2, let first = items.first,
              items.allSatisfy({ $0.source.buffer.mtl === first.source.buffer.mtl }) else { return [:] }
        let common = first.source.buffer.offset
        let cap = Swift.min(Int(UInt32.max) - (1 << 20), Swift.max(maxStagingBytes, 1 << 26))
        var staged: [Int: ParquetPreStaged] = [:]
        var i = 0
        while i < items.count {
            // Consecutive columns, page-aligned in the batch, up to the cap (a single larger column
            // stages on its own as before).
            var j = i, total = 0
            while j < items.count, total + roundUp(Swift.max(items[j].st.size, 1), to: metalPageSize()) <= cap {
                total += roundUp(Swift.max(items[j].st.size, 1), to: metalPageSize())
                j += 1
            }
            if j - i < 2 { i = Swift.max(j, i + 1); continue }
            do {
                let arena = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: context)
                var parts: [(staging: ParquetStaging, srcShift: Int, dstShift: Int)] = []
                var base = 0
                var bases: [Int] = []
                for k in i..<j {
                    parts.append((items[k].st, items[k].source.bindingOffset - common, base))
                    bases.append(base)
                    base += roundUp(Swift.max(items[k].st.size, 1), to: metalPageSize())
                }
                try runStaging(parts, source: first.source.buffer.mtl, sourceOffset: common, out: arena)
                for (n, k) in (i..<j).enumerated() {
                    staged[items[k].leaf.index] = ParquetPreStaged(
                        buffer: arena, base: bases[n], size: items[k].st.size,
                        pages: items[k].lp.dataPages.count + items[k].lp.dictPages.count)
                }
            } catch {
                return [:]
            }
            ParquetProfile.lap("read.stage-together \(j - i) columns", sync: context)
            i = j
        }
        return staged
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
