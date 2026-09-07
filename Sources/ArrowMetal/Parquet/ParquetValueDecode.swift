import Foundation
import Metal

// Value decoding: one kernel per encoding family, each writing *dense* values (no gaps for nulls), then
// a single scatter into row positions. A column chunk whose pages mix encodings -- which is what a writer
// produces when a dictionary outgrows its budget and the remaining pages fall back to PLAIN -- runs one
// kernel per encoding over its own slice of the page list and lands in the same output buffers.

struct ParquetValueDecoder {
    let file: ParquetFile
    let leaf: ParquetLeaf
    let ctx: MetalContext
    let pageData: MTLBuffer
    let pageDataOffset: Int
    let infos: [ParquetPageInfo]
    let totalLevels: Int
    let totalNonNull: Int
    let defBytes: MetalArrowBuffer?
    let ranks: MetalArrowBuffer?
    let maxDef: Int
    let width: Int
    let totalDict: Int
    let dictValOffset: MetalArrowBuffer?
    let dictValLength: MetalArrowBuffer?
    let dictFixed: MetalArrowBuffer?
    let dictionaryEncoded: Bool

    var hasDef: Bool { maxDef > 0 }
    var isByteArray: Bool { leaf.physical == .byteArray }
    var isBoolean: Bool { leaf.physical == .boolean }

    /// Page indices grouped by the kernel that decodes them.
    private func groups() throws -> [ParquetEncoding: [Int]] {
        var g: [ParquetEncoding: [Int]] = [:]
        for (i, p) in infos.enumerated() {
            guard let e = ParquetEncoding(rawValue: Int32(p.encoding)) else {
                throw ParquetError.unsupported("encoding \(p.encoding)")
            }
            let key: ParquetEncoding = e == .plainDictionary ? .rleDictionary : e
            g[key, default: []].append(i)
        }
        return g
    }

    func run() throws -> ParquetLeafData.Values {
        let g = try groups()
        let allDictionary = g.count == 1 && g[.rleDictionary] != nil && totalDict > 0

        // Dictionary codes, when any page needs them.
        var codes: MetalArrowBuffer? = nil
        if let dictPages = g[.rleDictionary] {
            let c = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
            let sub = try subset(dictPages)
            try runRLEValues(pages: sub, count: dictPages.count, out: c)
            try runDictRebase(pages: sub, count: dictPages.count, codes: c)
            codes = c
        }

        if allDictionary && dictionaryEncoded {
            let codeArray = try rowPositioned(dense: codes!, width: 4)
            let values = try dictionaryArray()
            return .dictionary(codes: codeArray, values: values)
        }

        if isByteArray {
            return try byteArrayValues(groups: g, codes: codes)
        }
        return try fixedValues(groups: g, codes: codes)
    }

    // MARK: - Fixed width

    private func fixedValues(groups g: [ParquetEncoding: [Int]], codes: MetalArrowBuffer?) throws -> ParquetLeafData.Values {
        let w = Swift.max(width, 1)
        let dense = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * w, w), zeroed: true, context: ctx)
        for (enc, idx) in g {
            let sub = try subset(idx)
            switch enc {
            case .plain:
                if isBoolean { try runPlainBool(pages: sub, count: idx.count, out: dense) }
                else { try runPlainFixed(pages: sub, count: idx.count, width: w, out: dense) }
            case .rle:
                // RLE-encoded booleans: decode to uint32, then narrow this subset to one byte per value.
                let tmp = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
                try runRLEValues(pages: sub, count: idx.count, out: tmp)
                try runNarrowU32ToBytes(input: tmp, pages: sub, count: idx.count, out: dense)
            case .rleDictionary, .plainDictionary:
                guard let codes, let dict = dictFixed else { throw ParquetError.malformed("dictionary page missing") }
                try runDictGatherFixed(dict: dict, codes: codes, pages: sub, count: idx.count,
                                       width: w, dictCount: totalDict, out: dense)
            case .byteStreamSplit:
                try runByteStreamSplit(pages: sub, count: idx.count, width: w, out: dense)
            case .deltaBinaryPacked:
                let start = try startPositions(idx)
                let end = try MetalArrowBuffer.allocate(byteCount: Swift.max(idx.count * 4, 4), zeroed: true, context: ctx)
                try runDeltaBinaryPacked(pages: sub, count: idx.count, width: w, start: start, out: dense, end: end)
            case .bitPacked:
                throw ParquetError.unsupported("the deprecated BIT_PACKED level encoding")
            default:
                throw ParquetError.unsupported("\(enc.name) on \(leaf.physical.name)")
            }
        }
        if isBoolean {
            // Booleans are one byte per value here; pack after they reach their row positions.
            let rowBytes = try rowPositionedBuffer(dense: dense, width: 1)
            let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: totalLevels), 4),
                                                     zeroed: true, context: ctx)
            try runBytesToBitmap(bytes: rowBytes, n: totalLevels, out: bits)
            return .boolean(bits)
        }
        let rows = try rowPositionedBuffer(dense: dense, width: w)
        return .fixed(rows, width: w)
    }

    // MARK: - Byte arrays

    private func byteArrayValues(groups g: [ParquetEncoding: [Int]], codes: MetalArrowBuffer?) throws -> ParquetLeafData.Values {
        // Per dense value: where its bytes are and how many. Sources may be the page buffer or, for the
        // delta byte-array encodings, a dense staging buffer built here.
        let valOffset = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
        let valLength = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
        // Groups whose bytes live somewhere other than the page buffer are gathered separately. The
        // staging buffer is held here, not just its MTLBuffer: dropping the wrapper would return the
        // allocation to the pool while the gather still points into it.
        var staged: [(indices: [Int], source: MetalArrowBuffer)] = []

        for (enc, idx) in g {
            let sub = try subset(idx)
            switch enc {
            case .plain:
                try runPlainBytesScan(pages: sub, count: idx.count, valOffset: valOffset, valLength: valLength)
            case .rleDictionary, .plainDictionary:
                guard let codes, let dOff = dictValOffset, let dLen = dictValLength else {
                    throw ParquetError.malformed("dictionary page missing")
                }
                try runDictBytesMap(codes: codes, dictOffset: dOff, dictLength: dLen, pages: sub,
                                    count: idx.count, dictCount: totalDict,
                                    valOffset: valOffset, valLength: valLength)
            case .deltaLengthByteArray:
                try deltaLengthByteArray(idx, sub, valOffset: valOffset, valLength: valLength)
            case .deltaByteArray:
                let buf = try deltaByteArray(idx, sub, valOffset: valOffset, valLength: valLength)
                staged.append((idx, buf))
            default:
                throw ParquetError.unsupported("\(enc.name) on BYTE_ARRAY")
            }
        }

        // Row lengths -> offsets.
        let rowLen = try MetalArrowBuffer.allocate(byteCount: Swift.max((totalLevels + 1) * 4, 8), zeroed: true, context: ctx)
        let allPages = try subset(Array(infos.indices))
        try runRowLengths(pages: allPages, count: infos.count, valLength: valLength, out: rowLen)
        let offsets = try MetalArrowBuffer.allocate(byteCount: Swift.max((totalLevels + 1) * 4, 8), zeroed: true, context: ctx)
        let totalBytes = try file.scanU32(ctx, input: rowLen, output: offsets, n: totalLevels + 1)
        let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalBytes, 1), zeroed: false, context: ctx)

        var stagedIndices = Set<Int>()
        for s in staged { stagedIndices.formUnion(s.indices) }
        let direct = infos.indices.filter { !stagedIndices.contains($0) }
        if !direct.isEmpty {
            try runGatherBytes(source: pageData, sourceOffset: pageDataOffset, pages: try subset(direct),
                               count: direct.count, valOffset: valOffset, valLength: valLength,
                               offsets: offsets, out: data)
        }
        for s in staged {
            try runGatherBytes(source: s.source.mtl, sourceOffset: s.source.offset, pages: try subset(s.indices),
                               count: s.indices.count, valOffset: valOffset, valLength: valLength,
                               offsets: offsets, out: data)
        }
        withExtendedLifetime(staged) {}
        return .bytes(offsets: offsets, data: data)
    }

    /// DELTA_LENGTH_BYTE_ARRAY: delta-packed lengths, then the bytes.
    private func deltaLengthByteArray(_ idx: [Int], _ sub: MetalArrowBuffer,
                                      valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
        let end = try MetalArrowBuffer.allocate(byteCount: Swift.max(idx.count * 4, 4), zeroed: true, context: ctx)
        let start = try startPositions(idx)
        try runDeltaBinaryPacked(pages: sub, count: idx.count, width: 4, start: start, out: lens, end: end)
        let starts = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalNonNull * 4, 4), zeroed: true, context: ctx)
        _ = try file.scanU32(ctx, input: lens, output: starts, n: totalNonNull)
        try runDeltaLengthsToTable(lengths: lens, denseStart: starts, endPos: end, pages: sub, count: idx.count,
                                  valOffset: valOffset, valLength: valLength)
    }

    /// DELTA_BYTE_ARRAY: delta-packed prefix lengths, delta-packed suffix lengths, then the suffixes.
    /// Values are rebuilt into a dense staging buffer that the row gather then reads from.
    private func deltaByteArray(_ idx: [Int], _ sub: MetalArrowBuffer,
                                valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws -> MetalArrowBuffer {
        let n = Swift.max(totalNonNull, 1)
        let prefix = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: true, context: ctx)
        let suffix = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: true, context: ctx)
        let end1 = try MetalArrowBuffer.allocate(byteCount: Swift.max(idx.count * 4, 4), zeroed: true, context: ctx)
        let end2 = try MetalArrowBuffer.allocate(byteCount: Swift.max(idx.count * 4, 4), zeroed: true, context: ctx)
        let start = try startPositions(idx)
        try runDeltaBinaryPacked(pages: sub, count: idx.count, width: 4, start: start, out: prefix, end: end1)
        try runDeltaBinaryPacked(pages: sub, count: idx.count, width: 4, start: end1, out: suffix, end: end2)
        // Byte positions of the suffixes, then of the reconstructed values.
        let suffixStart = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: true, context: ctx)
        _ = try file.scanU32(ctx, input: suffix, output: suffixStart, n: totalNonNull)
        let total = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: true, context: ctx)
        try runCopyU32(input: prefix, out: total, n: totalNonNull)
        try runAddU32(a: total, b: suffix, n: totalNonNull)
        let denseOffsets = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
        let denseBytes = try file.scanU32(ctx, input: total, output: denseOffsets, n: totalNonNull)
        let staging = try MetalArrowBuffer.allocate(byteCount: Swift.max(denseBytes, 1), zeroed: true, context: ctx)
        try runDeltaByteArray(prefix: prefix, suffix: suffix, suffixStart: suffixStart, endPos: end2,
                              pages: sub, count: idx.count, denseOffsets: denseOffsets, out: staging)
        try runDenseBytesTable(offsets: denseOffsets, lengths: total, n: totalNonNull,
                               valOffset: valOffset, valLength: valLength)
        return staging
    }

    /// Start byte positions for a delta stream: the page's value section.
    private func startPositions(_ idx: [Int]) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: Swift.max(idx.count * 4, 4), zeroed: true, context: ctx)
        let p = buf.mutableTyped(UInt32.self)
        for (k, i) in idx.enumerated() { p[k] = infos[i].valuesOffset }
        return buf
    }

    // MARK: - Dictionary values as an Arrow array

    private func dictionaryArray() throws -> AnyMetalArray {
        if isByteArray {
            guard let dOff = dictValOffset, let dLen = dictValLength else {
                throw ParquetError.malformed("dictionary page missing")
            }
            let offsets = try MetalArrowBuffer.allocate(byteCount: Swift.max((totalDict + 1) * 4, 8), zeroed: true, context: ctx)
            let bytes = try file.scanU32(ctx, input: dLen, output: offsets, n: totalDict + 1)
            let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: ctx)
            try runGatherFlat(source: pageData, sourceOffset: pageDataOffset, valOffset: dOff, valLength: dLen,
                              offsets: offsets, n: totalDict, out: data)
            let arr = MetalStringArray(length: totalDict, nullCount: 0, validity: nil,
                                       offsets: offsets, data: data, context: ctx)
            arr.isBinary = !isUTF8(leaf)
            return arr.isBinary ? .binary(arr) : .string(arr)
        }
        guard let dict = dictFixed else { throw ParquetError.malformed("dictionary page missing") }
        return try ParquetTypeMap.array(leaf: leaf, file: file, values: dict, width: Swift.max(width, 1),
                                        length: totalDict, validity: nil, nullCount: 0, context: ctx)
    }

    func isUTF8(_ leaf: ParquetLeaf) -> Bool {
        switch leaf.logicalType {
        case .string, .json, .enum: return true
        default: return false
        }
    }

    // MARK: - Dense -> rows

    /// Moves a dense buffer into row positions, or returns it unchanged when the column has no nulls.
    private func rowPositionedBuffer(dense: MetalArrowBuffer, width w: Int) throws -> MetalArrowBuffer {
        guard hasDef, let defBytes, let ranks else { return dense }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(totalLevels * w, w), zeroed: true, context: ctx)
        let all = try subset(Array(infos.indices))
        try runScatter(dense: dense, pages: all, count: infos.count, width: w,
                       defLevels: defBytes, ranks: ranks, out: out)
        return out
    }
    private func rowPositioned(dense: MetalArrowBuffer, width w: Int) throws -> MetalArrowBuffer {
        try rowPositionedBuffer(dense: dense, width: w)
    }

    // MARK: - Subsets and dispatch

    func subset(_ idx: [Int]) throws -> MetalArrowBuffer {
        let sub = idx.map { infos[$0] }
        return try file.upload(sub, ctx)
    }

    private func pso(_ f: String) throws -> MTLComputePipelineState { try file.pso(ctx, f) }

    private func perPage(_ function: String, count: Int, _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard count > 0 else { return }
        let p = try pso(function)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            bind(enc)
            enc.dispatchThreadgroups(MTLSize(width: count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
    private func perElement(_ function: String, count: Int, _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard count > 0 else { return }
        let p = try pso(function)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            bind(enc)
            Dispatch.dispatch1D(enc, p, count: count)
        }
    }

    func runPlainFixed(pages: MetalArrowBuffer, count: Int, width w: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_plain_fixed", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
        }
    }
    func runPlainBool(pages: MetalArrowBuffer, count: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_plain_bool", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
        }
    }
    func runByteStreamSplit(pages: MetalArrowBuffer, count: Int, width w: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_byte_stream_split", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            enc.setBuffer(out.mtl, offset: out.offset, index: 4)
        }
    }
    func runRLEValues(pages: MetalArrowBuffer, count: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_decode_rle_values", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
        }
    }
    func runDictRebase(pages: MetalArrowBuffer, count: Int, codes: MetalArrowBuffer) throws {
        try perPage("pq_dict_rebase", count: count) { enc in
            enc.setBuffer(codes.mtl, offset: codes.offset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
        }
    }
    func runDictGatherFixed(dict: MetalArrowBuffer, codes: MetalArrowBuffer, pages: MetalArrowBuffer,
                            count: Int, width w: Int, dictCount: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_dict_gather_fixed", count: count) { enc in
            enc.setBuffer(dict.mtl, offset: dict.offset, index: 0)
            enc.setBuffer(codes.mtl, offset: codes.offset, index: 1)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 2)
            Dispatch.setUInt(enc, count, index: 3)
            Dispatch.setUInt(enc, w, index: 4)
            Dispatch.setUInt(enc, dictCount, index: 5)
            enc.setBuffer(out.mtl, offset: out.offset, index: 6)
        }
    }
    func runScatter(dense: MetalArrowBuffer, pages: MetalArrowBuffer, count: Int, width w: Int,
                    defLevels: MetalArrowBuffer, ranks: MetalArrowBuffer, out: MetalArrowBuffer) throws {
        try perPage("pq_scatter", count: count) { enc in
            enc.setBuffer(dense.mtl, offset: dense.offset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            Dispatch.setUInt(enc, maxDef, index: 4)
            enc.setBuffer(defLevels.mtl, offset: defLevels.offset, index: 5)
            enc.setBuffer(ranks.mtl, offset: ranks.offset, index: 6)
            enc.setBuffer(out.mtl, offset: out.offset, index: 7)
        }
    }
    func runPlainBytesScan(pages: MetalArrowBuffer, count: Int,
                           valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        try perElement("pq_plain_bytes_scan", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 3)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 4)
        }
    }
    func runDictBytesMap(codes: MetalArrowBuffer, dictOffset: MetalArrowBuffer, dictLength: MetalArrowBuffer,
                         pages: MetalArrowBuffer, count: Int, dictCount: Int,
                         valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        try perPage("pq_dict_bytes_map", count: count) { enc in
            enc.setBuffer(codes.mtl, offset: codes.offset, index: 0)
            enc.setBuffer(dictOffset.mtl, offset: dictOffset.offset, index: 1)
            enc.setBuffer(dictLength.mtl, offset: dictLength.offset, index: 2)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 3)
            Dispatch.setUInt(enc, count, index: 4)
            Dispatch.setUInt(enc, dictCount, index: 5)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 6)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 7)
        }
    }
    func runRowLengths(pages: MetalArrowBuffer, count: Int, valLength: MetalArrowBuffer, out: MetalArrowBuffer) throws {
        let dummy = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        try perPage("pq_row_lengths", count: count) { enc in
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 0)
            Dispatch.setUInt(enc, count, index: 1)
            Dispatch.setUInt(enc, hasDef ? 1 : 0, index: 2)
            Dispatch.setUInt(enc, maxDef, index: 3)
            enc.setBuffer((defBytes ?? dummy).mtl, offset: (defBytes ?? dummy).offset, index: 4)
            enc.setBuffer((ranks ?? dummy).mtl, offset: (ranks ?? dummy).offset, index: 5)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 6)
            enc.setBuffer(out.mtl, offset: out.offset, index: 7)
        }
        withExtendedLifetime(dummy) {}
    }
    func runGatherBytes(source: MTLBuffer, sourceOffset: Int, pages: MetalArrowBuffer, count: Int,
                        valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer,
                        offsets: MetalArrowBuffer, out: MetalArrowBuffer) throws {
        let dummy = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        try perPage("pq_gather_bytes", count: count) { enc in
            enc.setBuffer(source, offset: sourceOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, hasDef ? 1 : 0, index: 3)
            Dispatch.setUInt(enc, maxDef, index: 4)
            enc.setBuffer((defBytes ?? dummy).mtl, offset: (defBytes ?? dummy).offset, index: 5)
            enc.setBuffer((ranks ?? dummy).mtl, offset: (ranks ?? dummy).offset, index: 6)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 7)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 8)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 9)
            enc.setBuffer(out.mtl, offset: out.offset, index: 10)
        }
        withExtendedLifetime(dummy) {}
    }
    func runGatherFlat(source: MTLBuffer, sourceOffset: Int, valOffset: MetalArrowBuffer,
                       valLength: MetalArrowBuffer, offsets: MetalArrowBuffer, n: Int, out: MetalArrowBuffer) throws {
        try perElement("pq_gather_flat", count: n) { enc in
            enc.setBuffer(source, offset: sourceOffset, index: 0)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 1)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 2)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
        }
    }
    func runBytesToBitmap(bytes: MetalArrowBuffer, n: Int, out: MetalArrowBuffer) throws {
        try perElement("pq_bytes_to_bitmap", count: (n + 31) / 32) { enc in
            enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 0)
            Dispatch.setUInt(enc, n, index: 1)
            enc.setBuffer(out.mtl, offset: out.offset, index: 2)
        }
    }
    func runNarrowU32ToBytes(input: MetalArrowBuffer, pages: MetalArrowBuffer, count: Int, out: MetalArrowBuffer) throws {
        try perPage("pq_narrow_u32_pages", count: count) { enc in
            enc.setBuffer(input.mtl, offset: input.offset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            enc.setBuffer(out.mtl, offset: out.offset, index: 3)
        }
    }
    func runCopyU32(input: MetalArrowBuffer, out: MetalArrowBuffer, n: Int) throws {
        try perElement("pq_copy_u32", count: n) { enc in
            enc.setBuffer(input.mtl, offset: input.offset, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
        }
    }
    func runAddU32(a: MetalArrowBuffer, b: MetalArrowBuffer, n: Int) throws {
        try perElement("pq_add_u32", count: n) { enc in
            enc.setBuffer(a.mtl, offset: a.offset, index: 0)
            enc.setBuffer(b.mtl, offset: b.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
        }
    }
    func runDeltaBinaryPacked(pages: MetalArrowBuffer, count: Int, width w: Int,
                              start: MetalArrowBuffer, out: MetalArrowBuffer, end: MetalArrowBuffer) throws {
        try perPage("pq_delta_binary_packed", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 1)
            Dispatch.setUInt(enc, count, index: 2)
            Dispatch.setUInt(enc, w, index: 3)
            enc.setBuffer(start.mtl, offset: start.offset, index: 4)
            enc.setBuffer(out.mtl, offset: out.offset, index: 5)
            enc.setBuffer(end.mtl, offset: end.offset, index: 6)
        }
    }
    func runDeltaLengthsToTable(lengths: MetalArrowBuffer, denseStart: MetalArrowBuffer, endPos: MetalArrowBuffer,
                                pages: MetalArrowBuffer, count: Int,
                                valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        try perPage("pq_delta_lengths_to_table", count: count) { enc in
            enc.setBuffer(lengths.mtl, offset: lengths.offset, index: 0)
            enc.setBuffer(denseStart.mtl, offset: denseStart.offset, index: 1)
            enc.setBuffer(endPos.mtl, offset: endPos.offset, index: 2)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 3)
            Dispatch.setUInt(enc, count, index: 4)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 5)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 6)
        }
    }
    func runDeltaByteArray(prefix: MetalArrowBuffer, suffix: MetalArrowBuffer, suffixStart: MetalArrowBuffer,
                           endPos: MetalArrowBuffer, pages: MetalArrowBuffer, count: Int,
                           denseOffsets: MetalArrowBuffer, out: MetalArrowBuffer) throws {
        try perPage("pq_delta_byte_array", count: count) { enc in
            enc.setBuffer(pageData, offset: pageDataOffset, index: 0)
            enc.setBuffer(prefix.mtl, offset: prefix.offset, index: 1)
            enc.setBuffer(suffix.mtl, offset: suffix.offset, index: 2)
            enc.setBuffer(suffixStart.mtl, offset: suffixStart.offset, index: 3)
            enc.setBuffer(endPos.mtl, offset: endPos.offset, index: 4)
            enc.setBuffer(pages.mtl, offset: pages.offset, index: 5)
            Dispatch.setUInt(enc, count, index: 6)
            enc.setBuffer(denseOffsets.mtl, offset: denseOffsets.offset, index: 7)
            enc.setBuffer(out.mtl, offset: out.offset, index: 8)
        }
    }
    func runDenseBytesTable(offsets: MetalArrowBuffer, lengths: MetalArrowBuffer, n: Int,
                            valOffset: MetalArrowBuffer, valLength: MetalArrowBuffer) throws {
        try perElement("pq_dense_bytes_table", count: n) { enc in
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
            enc.setBuffer(lengths.mtl, offset: lengths.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            enc.setBuffer(valOffset.mtl, offset: valOffset.offset, index: 3)
            enc.setBuffer(valLength.mtl, offset: valLength.offset, index: 4)
        }
    }

}
