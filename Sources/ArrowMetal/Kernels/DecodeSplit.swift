import Foundation
import Metal

// CPU + GPU page decompression.
//
// An LZ77 page (Snappy, LZ4) is a serial token stream, so the GPU decodes one page per SIMD group or
// per thread and a dispatch lasts as long as its slowest page. That is fast for pages that are mostly
// long literals (incompressible values move at memory speed) and slow for token-dense pages: one
// 160 KB page of sequential int64 values takes about 15 ms on a SIMD group and 58 ms on a thread, while
// one CPU core decodes it in about 50 µs. So a read splits its pages between the two sides: the
// router below estimates each page's cost on the host and on the GPU from its header (compressed and
// uncompressed length, hence its ratio, before a byte of it is decoded) and hands the host the most
// token-dense pages until the host's share, spread over its cores, takes about as long as the GPU's.
// Both then run at once: the GPU dispatch is committed first, the host decodes its pages across cores
// while it runs, and the read waits once for both before any value kernel reads a page.

/// One page (or a v2 page's value section) to decompress, as the router sees it.
struct DecodeCandidate {
    var codec: ParquetCodec
    var srcLength: Int
    var dstLength: Int
}

/// Who decompresses a page: the host, the GPU's SIMD-group-per-page kernel, or its page-per-thread kernel.
enum DecodeSide: Equatable, Sendable {
    case host, gpuGroup, gpuLane
}

/// A host job: a run of blocks of one codec (`.uncompressed` for a plain copy) the host decodes in order.
struct HostDecodeJob {
    var codec: ParquetCodec
    var blocks: [PageBlock]
    var cost: Double
}

/// Per-page decompression costs, measured on an Apple M4 Max (12 performance and 4 efficiency cores)
/// over the Snappy and LZ4 files of `Benchmarks/parquet_bench.py` and a 10 M-row, 7-column
/// pyarrow-default Snappy file: nanoseconds per output byte, by page class. A page's class is its
/// ratio, uncompressed over compressed: near 1 it is one long literal, from 1.25 up it is token-dense
/// (sequential, low-cardinality or timestamp values), and far above that (runs of a repeated value) it is
/// a few long copies again.
enum DecodeCost {
    /// 0 for a literal page, 1 for a token-dense one, falling again for long-run pages.
    static func density(_ c: DecodeCandidate) -> Double {
        let ratio = Double(c.dstLength) / Double(Swift.max(c.srcLength, 1))
        let rising = Swift.min(Swift.max((ratio - 1.02) / 0.23, 0), 1)
        return ratio > 4 ? rising * 4 / ratio : rising
    }

    // Host, one core, reading a freshly opened file, where the first touch of each page of the mapping
    // is a minor fault. Literal pages are a copy: 1.2 GB of them took 18.5 ms on all cores (5.8 ms
    // from a mapping already touched). Token-dense pages: Snappy 52-78 µs, LZ4 40-44 µs per 160 KB
    // page on one core (2.0-3.0 and 3.6-3.9 GB/s of output), plus the faults.
    static let hostLiteralNsPerByte = 0.18
    static let hostSnappyDenseNsPerByte = 0.42
    static let hostLZ4DenseNsPerByte = 0.37
    static let hostNsPerPage = 1_500.0
    /// ZSTD, GZIP and BROTLI run on the host only; these place their pages in the schedule.
    static let hostZstdNsPerByte = 0.9
    static let hostDeflateNsPerByte = 2.8

    static func hostNs(_ c: DecodeCandidate) -> Double {
        let bytes = Double(c.dstLength)
        switch c.codec {
        case .snappy, .lz4, .lz4Raw:
            let dense = c.codec == .snappy ? hostSnappyDenseNsPerByte : hostLZ4DenseNsPerByte
            let d = density(c)
            return hostNsPerPage + bytes * (hostLiteralNsPerByte + (dense - hostLiteralNsPerByte) * d)
        case .zstd: return hostNsPerPage + bytes * hostZstdNsPerByte
        case .gzip, .brotli: return hostNsPerPage + bytes * hostDeflateNsPerByte
        default: return hostNsPerPage + bytes * hostLiteralNsPerByte
        }
    }

    // GPU, SIMD group per page: a page's own decode time (latency) and what it adds to a full dispatch
    // (throughput). One token-dense 160 KB page takes 12-15 ms on its own (75-95 ns per output byte),
    // and 12,625 of them 132-163 ms (0.10-0.12 ns per byte); while every CPU core decodes too, the
    // GPU's dense pages take longer, and of 90 to 400 ns per byte for the latency here, 130 gave the
    // lowest whole-file read times. 1.2 GB of literal pages take 9.5-10 ms (0.008 ns per byte).
    static let groupLatencyLiteralNsPerByte = 4.0
    static let groupLatencyDenseNsPerByte = 130.0
    static let groupThroughputLiteralNsPerByte = 0.008
    static let groupThroughputDenseNsPerByte = 0.12
    // GPU, page per thread: one thread walks the page, 58 ms for a token-dense 160 KB Snappy page
    // (0.37 µs per output byte) and 39-40 ms for an LZ4 one (0.25 µs), about the same for 32 pages
    // at once as for 12,625 (51-66 ms).
    static let laneLatencySnappyNsPerByte = 370.0
    static let laneLatencyLZ4NsPerByte = 250.0
    static let laneSlots = 16_384.0
    static let gpuLaunchNs = 120_000.0

    static func groupLatency(_ c: DecodeCandidate) -> Double {
        let d = density(c)
        return Double(c.dstLength) * (groupLatencyLiteralNsPerByte + (groupLatencyDenseNsPerByte - groupLatencyLiteralNsPerByte) * d)
    }
    static func groupThroughput(_ c: DecodeCandidate) -> Double {
        let d = density(c)
        return Double(c.dstLength) * (groupThroughputLiteralNsPerByte + (groupThroughputDenseNsPerByte - groupThroughputLiteralNsPerByte) * d)
    }
    static func laneLatency(_ c: DecodeCandidate) -> Double {
        let per = c.codec == .snappy ? laneLatencySnappyNsPerByte : laneLatencyLZ4NsPerByte
        return Double(c.dstLength) * per * Swift.max(density(c), 0.75)
    }

    /// Cores the host decode spreads over, counting an efficiency core as a quarter of a performance
    /// core (16 threads decoded token-dense pages 12.4 times as fast as one on the M4 Max, 12 threads
    /// 11.2 times).
    static let hostWorkers: Double = {
        func sysctlInt(_ name: String) -> Int? {
            var v: Int32 = 0
            var n = MemoryLayout<Int32>.size
            return sysctlbyname(name, &v, &n, nil, 0) == 0 && v > 0 ? Int(v) : nil
        }
        let all = ProcessInfo.processInfo.activeProcessorCount
        if let p = sysctlInt("hw.perflevel0.logicalcpu") {
            let e = sysctlInt("hw.perflevel1.logicalcpu") ?? Swift.max(all - p, 0)
            return Swift.max(1, (Double(p) + Double(e) / 4) * 0.9)
        }
        return Swift.max(1, Double(all) * 0.7)
    }()
}

/// Decides, per page, which side decompresses it (see the file comment).
enum DecodeRouter {
    /// Forces every Snappy and LZ4 page to one side; nil routes by cost. The tests use it to run
    /// every page through each decoder, and `ARROWMETAL_PARQUET_DECODE` (`host`, `gpu`, `lane`,
    /// `auto`) sets it for a process.
    static var forced: DecodeSide? = {
        switch ProcessInfo.processInfo.environment["ARROWMETAL_PARQUET_DECODE"]?.lowercased() {
        case "host": return .host
        case "gpu", "group": return .gpuGroup
        case "lane": return .gpuLane
        default: return nil
        }
    }()

    /// Routes `cands`. A nil entry is a page stored plain, which the GPU copies; ZSTD, GZIP and BROTLI
    /// pages go to the host, and their cost counts against the host's share; Snappy and LZ4 pages go to
    /// whichever side the costs say.
    ///
    /// The Snappy and LZ4 pages are ordered densest first -- the pages the GPU is slowest at next to
    /// the host -- and the host takes the prefix that minimises the later of the two finishing times:
    /// the host's pages spread over its cores (never less than its largest page), against one GPU
    /// dispatch of the rest (its slowest page plus what every page adds; token-dense pages go to the
    /// page-per-thread kernel instead when that is predicted faster). Ties go to the smaller host share.
    static func route(_ cands: [DecodeCandidate?]) -> [DecodeSide] {
        var sides = [DecodeSide](repeating: .host, count: cands.count)
        var hostFixedNs = 0.0, gpuFixedNs = 0.0
        var flexible: [Int] = []
        for (i, c) in cands.enumerated() {
            guard let c else {
                // Plain pages are one more GPU dispatch, at memory speed.
                sides[i] = .gpuGroup
                gpuFixedNs = DecodeCost.gpuLaunchNs
                continue
            }
            if isGPUCodec(c.codec) { flexible.append(i) } else { hostFixedNs += DecodeCost.hostNs(c) }
        }
        if flexible.isEmpty { return sides }
        if let f = forced {
            for i in flexible { sides[i] = f }
            return sides
        }
        let n = flexible.count
        var density = [Double](repeating: 0, count: n)
        for k in 0..<n { density[k] = DecodeCost.density(cands[flexible[k]]!) }
        let order = (0..<n).sorted {
            density[$0] != density[$1] ? density[$0] > density[$1]
                : cands[flexible[$0]]!.dstLength > cands[flexible[$1]]!.dstLength
        }
        let w = DecodeCost.hostWorkers
        // Suffix sums for the GPU side: pages order[k...] go to the GPU.
        var sufLat = [Double](repeating: 0, count: n + 1)         // slowest page, SIMD-group kernel
        var sufThr = [Double](repeating: 0, count: n + 1)         // what every page adds, SIMD-group kernel
        var sufLaneLat = [Double](repeating: 0, count: n + 1)     // slowest dense page, page-per-thread kernel
        var sufLaneCount = [Int](repeating: 0, count: n + 1)
        var sufRestLat = [Double](repeating: 0, count: n + 1)     // the other pages, SIMD-group kernel
        var sufRestThr = [Double](repeating: 0, count: n + 1)
        for k in stride(from: n - 1, through: 0, by: -1) {
            let c = cands[flexible[order[k]]]!
            let lat = DecodeCost.groupLatency(c), thr = DecodeCost.groupThroughput(c)
            sufLat[k] = Swift.max(sufLat[k + 1], lat)
            sufThr[k] = sufThr[k + 1] + thr
            if density[order[k]] >= 0.99 {
                sufLaneLat[k] = Swift.max(sufLaneLat[k + 1], DecodeCost.laneLatency(c))
                sufLaneCount[k] = sufLaneCount[k + 1] + 1
                sufRestLat[k] = sufRestLat[k + 1]; sufRestThr[k] = sufRestThr[k + 1]
            } else {
                sufLaneLat[k] = sufLaneLat[k + 1]; sufLaneCount[k] = sufLaneCount[k + 1]
                sufRestLat[k] = Swift.max(sufRestLat[k + 1], lat)
                sufRestThr[k] = sufRestThr[k + 1] + thr
            }
        }
        func gpuTime(_ k: Int) -> (Double, lane: Bool) {
            guard k < n else { return (gpuFixedNs, false) }
            let group = DecodeCost.gpuLaunchNs + sufLat[k] + sufThr[k]
            if sufLaneCount[k] > 0 {
                let lane = DecodeCost.gpuLaunchNs * 2 + sufRestLat[k] + sufRestThr[k]
                    + sufLaneLat[k] * Swift.max(1, (Double(sufLaneCount[k]) / DecodeCost.laneSlots).rounded(.up))
                if lane < group { return (gpuFixedNs + lane, true) }
            }
            return (gpuFixedNs + group, false)
        }
        var best = (k: 0, t: Double.infinity, lane: false, host: 0.0, gpu: 0.0)
        var hostSum = hostFixedNs, hostMax = 0.0
        for k in 0...n {
            if k > 0 {
                let h = DecodeCost.hostNs(cands[flexible[order[k - 1]]]!)
                hostSum += h
                hostMax = Swift.max(hostMax, h)
            }
            let hostT = hostSum == 0 ? 0 : Swift.max(hostSum / w, hostMax)
            let (g, lane) = gpuTime(k)
            let t = Swift.max(hostT, g)
            if t < best.t * 0.995 { best = (k, t, lane, hostT, g) }
        }
        for (j, o) in order.enumerated() {
            let i = flexible[o]
            if j < best.k { sides[i] = .host }
            else { sides[i] = best.lane && density[o] >= 0.99 ? .gpuLane : .gpuGroup }
        }
        if ParquetProfile.enabled {
            let hostPages = best.k, lanePages = sides.filter { $0 == .gpuLane }.count
            lastSummary = String(format: "host %d pages (%.1f ms predicted), GPU %d pages (%d page per thread, %.1f ms predicted)",
                                 hostPages, best.host / 1e6, n - hostPages, lanePages, best.gpu / 1e6)
        }
        return sides
    }

    /// The last routing decision, for `ARROWMETAL_PARQUET_PROFILE`.
    static var lastSummary = ""

    static func isGPUCodec(_ c: ParquetCodec) -> Bool { c == .snappy || c == .lz4 || c == .lz4Raw }
}

extension Decompress {
    /// Runs one staging: `copies` (plain page copies) and the GPU blocks on the GPU, the host jobs on
    /// the host's cores at the same time, and returns once both are done. Blocks of the two sides never
    /// share a page of `out` (the staging layout keeps the host's pages in their own page-aligned
    /// range), so neither side writes memory the other is writing.
    static func overlapped(_ ctx: MetalContext, source: MTLBuffer, sourceOffset: Int, out: MetalArrowBuffer,
                           copies: [PageBlock], gpu: [(codec: ParquetCodec, group: [PageBlock], lane: [PageBlock])],
                           host: [HostDecodeJob]) throws {
        let anyGPU = !copies.isEmpty || gpu.contains { !$0.group.isEmpty || !$0.lane.isEmpty }
        var pending: [GPUPending] = []
        var committed: MetalContext.Batch? = nil
        var event: (MTLSharedEvent, UInt64)? = nil
        let hadBatch = ctx.isBatching
        if anyGPU {
            if !hadBatch { try ctx.openBatch() }
            do {
                if !copies.isEmpty {
                    try gpuCopy(ctx, source: source, sourceOffset: sourceOffset, blocks: copies, out: out)
                }
                for g in gpu {
                    guard g.codec != .uncompressed else { continue }
                    let function = g.codec == .snappy ? "snappy_decompress" : "lz4_decompress"
                    if !g.group.isEmpty {
                        pending.append(try encodeGPU(ctx, function: function, perThread: false, source: source,
                                                     sourceOffset: sourceOffset, blocks: g.group, out: out, codec: g.codec))
                    }
                    if !g.lane.isEmpty {
                        pending.append(try encodeGPU(ctx, function: function, perThread: true, source: source,
                                                     sourceOffset: sourceOffset, blocks: g.lane, out: out, codec: g.codec))
                    }
                }
            } catch {
                if !hadBatch { try? ctx.flush() }
                throw error
            }
            if host.isEmpty {
                // Nothing to overlap: the caller's batch carries the dispatch.
                if !hadBatch { try ctx.flush() } else { try ctx.syncPoint() }
                try check(pending)
                return
            }
            committed = ctx.detachBatch()
            if let b = committed { event = ctx.commitSignalled(b.commandBuffer) }
            if hadBatch { try ctx.openBatch() }
        }
        var hostError: Error? = nil
        let t0 = ParquetProfile.enabled ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
        if !host.isEmpty {
            do { try runHost(host, source: source, sourceOffset: sourceOffset, out: out) } catch { hostError = error }
        }
        let t1 = ParquetProfile.enabled ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
        if let b = committed {
            ctx.waitSignalled(b.commandBuffer, event)
            try ctx.finishBatch(b)
        }
        if ParquetProfile.enabled {
            let t2 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            lastOverlap = String(format: "host %.1f ms, then GPU %.1f ms more", Double(t1 - t0) / 1e6, Double(t2 - t1) / 1e6)
        }
        if let e = hostError { throw e }
        try check(pending)
    }

    /// How the last overlapped staging went, for `ARROWMETAL_PARQUET_PROFILE`.
    static var lastOverlap = ""

    struct GPUPending {
        let blocks: [PageBlock]
        let status: MetalArrowBuffer
        let desc: MetalArrowBuffer
        let codec: ParquetCodec
    }

    private static func encodeGPU(_ ctx: MetalContext, function: String, perThread: Bool, source: MTLBuffer,
                                  sourceOffset: Int, blocks: [PageBlock], out: MetalArrowBuffer,
                                  codec: ParquetCodec) throws -> GPUPending {
        let desc = try MetalArrowBuffer.allocate(byteCount: blocks.count * MemoryLayout<PageBlock>.stride,
                                                 zeroed: false, context: ctx)
        blocks.withUnsafeBytes { memcpy(desc.mutableContents, $0.baseAddress!, $0.count) }
        let status = try MetalArrowBuffer.allocate(byteCount: blocks.count * 4, context: ctx)
        let name = perThread ? function + "_lane" : function
        let pso = try ctx.pipeline(source: DecompressSource.source, function: name, cacheKey: "parquet/decompress/\(name)")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(source, offset: sourceOffset, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            enc.setBuffer(desc.mtl, offset: desc.offset, index: 2)
            Dispatch.setUInt(enc, blocks.count, index: 3)
            enc.setBuffer(status.mtl, offset: status.offset, index: 4)
            let w = 32
            if perThread {
                // One page per thread, in threadgroups of 32 so the pages spread over every core.
                enc.dispatchThreadgroups(MTLSize(width: (blocks.count + w - 1) / w, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            } else {
                // One SIMD group per page: lane 0 parses the token stream out of an 8 KB
                // threadgroup-memory window while all 32 lanes move the bytes.
                enc.dispatchThreadgroups(MTLSize(width: blocks.count, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            }
        }
        return GPUPending(blocks: blocks, status: status, desc: desc, codec: codec)
    }

    private static func check(_ pending: [GPUPending]) throws {
        for p in pending {
            let st = p.status.typed(UInt32.self)
            for k in p.blocks.indices where st[k] != 0 {
                let why = ["ok", "output overrun", "bad token", "input truncated"][Int(Swift.min(st[k], 3))]
                throw ParquetError.malformed("\(p.codec.name) page \(k) failed to decompress: \(why)")
            }
        }
    }

    /// Decodes one block on the host into `out`; throws unless it fills its slot exactly.
    static func hostDecode(_ codec: ParquetCodec, _ b: PageBlock, src: UnsafeRawPointer, dst: UnsafeMutableRawPointer,
                           zstd: Zstd.Context?) throws {
        let inPtr = src.advanced(by: Int(b.srcOffset)).assumingMemoryBound(to: UInt8.self)
        let outPtr = dst.advanced(by: Int(b.dstOffset)).assumingMemoryBound(to: UInt8.self)
        let produced: Int
        switch codec {
        case .uncompressed:
            memcpy(outPtr, inPtr, Int(Swift.min(b.srcLength, b.dstLength)))
            return
        case .snappy: produced = try SnappyHost.decompress(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
        case .lz4, .lz4Raw: produced = try LZ4Host.decompress(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
        case .zstd: produced = try Zstd.decompress(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength), context: zstd)
        case .gzip: produced = try gunzipBlock(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
        case .brotli: produced = try brotliBlock(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
        case .lzo: throw ParquetError.unsupported("LZO compression")
        }
        guard produced == Int(b.dstLength) else {
            throw ParquetError.malformed("\(codec.name) page: produced \(produced) of \(b.dstLength) bytes")
        }
    }

    /// Runs the host jobs across cores, most expensive first, and throws the first error by job order.
    static func runHost(_ jobs: [HostDecodeJob], source: MTLBuffer, sourceOffset: Int, out: MetalArrowBuffer) throws {
        let src = UnsafeRawPointer(source.contents()).advanced(by: sourceOffset)
        let dst = out.mutableContents
        let order = jobs.indices.sorted { jobs[$0].cost > jobs[$1].cost }
        let lock = NSLock()
        var failure: (Int, Error)? = nil
        let body: (Int) -> Void = { k in
            let j = order[k]
            let job = jobs[j]
            let zctx = job.codec == .zstd ? Zstd.Context() : nil
            do {
                for b in job.blocks { try hostDecode(job.codec, b, src: src, dst: dst, zstd: zctx) }
            } catch {
                lock.lock()
                if failure == nil || failure!.0 > j { failure = (j, error) }
                lock.unlock()
            }
        }
        let total = jobs.reduce(0.0) { $0 + $1.cost }
        if jobs.count >= 2 && total > 200_000 {
            DispatchQueue.concurrentPerform(iterations: jobs.count, execute: body)
        } else {
            for k in jobs.indices { body(k) }
        }
        if let (_, e) = failure { throw e }
    }
}
