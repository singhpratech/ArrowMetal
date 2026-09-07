import Foundation
import Metal
import Compression

/// One compressed block: a byte range of the source and where its plaintext goes.
struct PageBlock {
    var srcOffset: UInt32
    var srcLength: UInt32
    var dstOffset: UInt32
    var dstLength: UInt32
}

/// Block decompression of Parquet pages.
///
/// SNAPPY, LZ4 and LZ4_RAW run entirely on the GPU (`DecompressSource`), one threadgroup per page.
/// UNCOMPRESSED needs no work at all: the mapped file *is* the page buffer, so the reader passes the
/// file's own `MTLBuffer` straight to the decoders.
///
/// ZSTD, GZIP and BROTLI are decompressed on the host, straight into the shared-memory output buffer the
/// GPU decoders will read, with pages spread across cores. GZIP and BROTLI go through Foundation's
/// Compression framework (`COMPRESSION_ZLIB` is raw DEFLATE, so the gzip container is stripped first).
/// The SDK has no `COMPRESSION_ZSTD`, so libzstd is looked up with `dlopen` at first use; when it is not
/// installed, a ZSTD column raises `ParquetError.unsupported` naming the missing library rather than
/// returning wrong data.
enum Decompress {
    /// Decompresses every block into a buffer the caller owns, so several codecs can share one buffer.
    ///
    /// `source` is bound at `sourceOffset`, so `block.srcOffset` is relative to that: a column chunk
    /// anywhere in a multi-gigabyte file still addresses its pages with 32-bit offsets.
    static func into(_ ctx: MetalContext, codec: ParquetCodec, source: MTLBuffer, sourceOffset: Int,
                     blocks: [PageBlock], out: MetalArrowBuffer) throws {
        guard !blocks.isEmpty else { return }
        switch codec {
        case .uncompressed:
            try gpuCopy(ctx, source: source, sourceOffset: sourceOffset, blocks: blocks, out: out)
        case .snappy:
            try gpuDecompress(ctx, function: "snappy_decompress", source: source, sourceOffset: sourceOffset,
                              blocks: blocks, out: out, codec: codec)
        case .lz4, .lz4Raw:
            try gpuDecompress(ctx, function: "lz4_decompress", source: source, sourceOffset: sourceOffset,
                              blocks: blocks, out: out, codec: codec)
        case .gzip, .zstd, .brotli:
            try hostDecompress(codec: codec, source: source, sourceOffset: sourceOffset, blocks: blocks, out: out)
        case .lzo:
            throw ParquetError.unsupported("LZO compression")
        }
    }

    private static func blockBuffer(_ ctx: MetalContext, _ blocks: [PageBlock]) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: blocks.count * MemoryLayout<PageBlock>.stride,
                                                zeroed: false, context: ctx)
        blocks.withUnsafeBytes { memcpy(buf.mutableContents, $0.baseAddress!, $0.count) }
        return buf
    }

    private static func gpuCopy(_ ctx: MetalContext, source: MTLBuffer, sourceOffset: Int,
                                blocks: [PageBlock], out: MetalArrowBuffer) throws {
        let desc = try blockBuffer(ctx, blocks)
        let pso = try ctx.pipeline(source: DecompressSource.source, function: "block_copy", cacheKey: "parquet/decompress/block_copy")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(source, offset: sourceOffset, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            enc.setBuffer(desc.mtl, offset: desc.offset, index: 2)
            Dispatch.setUInt(enc, blocks.count, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks.count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        withExtendedLifetime(desc) {}
    }

    private static func gpuDecompress(_ ctx: MetalContext, function: String, source: MTLBuffer, sourceOffset: Int,
                                      blocks: [PageBlock], out: MetalArrowBuffer, codec: ParquetCodec) throws {
        let desc = try blockBuffer(ctx, blocks)
        let status = try MetalArrowBuffer.allocate(byteCount: blocks.count * 4, context: ctx)
        let pso = try ctx.pipeline(source: DecompressSource.source, function: function, cacheKey: "parquet/decompress/\(function)")
        // One threadgroup per page: thread 0 parses the token stream out of an 8 KB threadgroup-memory
        // window while all 256 threads move the bytes.
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(source, offset: sourceOffset, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            enc.setBuffer(desc.mtl, offset: desc.offset, index: 2)
            Dispatch.setUInt(enc, blocks.count, index: 3)
            enc.setBuffer(status.mtl, offset: status.offset, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: blocks.count, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        try ctx.syncPoint()
        let st = status.typed(UInt32.self)
        for i in 0..<blocks.count where st[i] != 0 {
            let why = ["ok", "output overrun", "bad token", "input truncated"][Int(Swift.min(st[i], 3))]
            throw ParquetError.malformed("\(codec.name) page \(i) failed to decompress: \(why)")
        }
        withExtendedLifetime(desc) {}
        withExtendedLifetime(status) {}
    }

    // MARK: - Host codecs

    private static func hostDecompress(codec: ParquetCodec, source: MTLBuffer, sourceOffset: Int,
                                       blocks: [PageBlock], out: MetalArrowBuffer) throws {
        let src = UnsafeRawPointer(source.contents()).advanced(by: sourceOffset)
        let dst = out.mutableContents
        var failure: Error? = nil
        let lock = NSLock()
        let n = blocks.count
        let work: (Int) -> Void = { i in
            let b = blocks[i]
            do {
                let inPtr = src.advanced(by: Int(b.srcOffset)).assumingMemoryBound(to: UInt8.self)
                let outPtr = dst.advanced(by: Int(b.dstOffset)).assumingMemoryBound(to: UInt8.self)
                let produced: Int
                switch codec {
                case .gzip: produced = try gunzip(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
                case .brotli: produced = try appleDecode(COMPRESSION_BROTLI, inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
                case .zstd: produced = try Zstd.decompress(inPtr, Int(b.srcLength), outPtr, Int(b.dstLength))
                default: produced = 0
                }
                guard produced == Int(b.dstLength) else {
                    throw ParquetError.malformed("\(codec.name) page \(i): produced \(produced) of \(b.dstLength) bytes")
                }
            } catch {
                lock.lock(); if failure == nil { failure = error }; lock.unlock()
            }
        }
        if n >= 4 { DispatchQueue.concurrentPerform(iterations: n, execute: work) }
        else { for i in 0..<n { work(i) } }
        if let f = failure { throw f }
    }

    private static func appleDecode(_ algorithm: compression_algorithm,
                                    _ src: UnsafePointer<UInt8>, _ srcLen: Int,
                                    _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int) throws -> Int {
        let n = compression_decode_buffer(dst, dstLen, src, srcLen, nil, algorithm)
        guard n > 0 || dstLen == 0 else { throw ParquetError.malformed("compression_decode_buffer produced nothing") }
        return n
    }

    /// GZIP (RFC 1952) on top of `COMPRESSION_ZLIB`, which is raw DEFLATE: strip the container.
    private static func gunzip(_ src: UnsafePointer<UInt8>, _ srcLen: Int,
                               _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int) throws -> Int {
        guard srcLen >= 18, src[0] == 0x1F, src[1] == 0x8B, src[2] == 8 else {
            // Not a gzip container; try raw DEFLATE / zlib as written by some producers.
            return try appleDecode(COMPRESSION_ZLIB, src, srcLen, dst, dstLen)
        }
        let flg = src[3]
        var p = 10
        if flg & 0x04 != 0 {                                   // FEXTRA
            guard p + 2 <= srcLen else { throw ParquetError.truncated("gzip FEXTRA") }
            let xlen = Int(src[p]) | (Int(src[p + 1]) << 8)
            p += 2 + xlen
        }
        if flg & 0x08 != 0 { while p < srcLen && src[p] != 0 { p += 1 }; p += 1 }   // FNAME
        if flg & 0x10 != 0 { while p < srcLen && src[p] != 0 { p += 1 }; p += 1 }   // FCOMMENT
        if flg & 0x02 != 0 { p += 2 }                                              // FHCRC
        guard p < srcLen - 8 else { throw ParquetError.truncated("gzip body") }
        return try appleDecode(COMPRESSION_ZLIB, src.advanced(by: p), srcLen - p - 8, dst, dstLen)
    }
}

/// libzstd, loaded lazily with `dlopen` because the macOS SDK exposes no `COMPRESSION_ZSTD`.
enum Zstd {
    typealias DecompressFn = @convention(c) (UnsafeMutableRawPointer?, Int, UnsafeRawPointer?, Int) -> Int
    typealias IsErrorFn = @convention(c) (Int) -> UInt32

    /// Candidate library paths, most specific first. `ARROWMETAL_ZSTD` overrides everything.
    static let candidates: [String] = {
        var c: [String] = []
        if let env = ProcessInfo.processInfo.environment["ARROWMETAL_ZSTD"] { c.append(env) }
        c += ["libzstd.1.dylib", "/opt/homebrew/lib/libzstd.1.dylib", "/opt/homebrew/lib/libzstd.dylib",
              "/usr/local/lib/libzstd.1.dylib", "/usr/local/lib/libzstd.dylib", "/usr/lib/libzstd.1.dylib"]
        return c
    }()

    private static let loaded: (decompress: DecompressFn, isError: IsErrorFn)? = {
        for name in candidates {
            guard let h = dlopen(name, RTLD_LAZY) else { continue }
            guard let d = dlsym(h, "ZSTD_decompress"), let e = dlsym(h, "ZSTD_isError") else { continue }
            return (unsafeBitCast(d, to: DecompressFn.self), unsafeBitCast(e, to: IsErrorFn.self))
        }
        return nil
    }()

    static var isAvailable: Bool { loaded != nil }

    static func decompress(_ src: UnsafePointer<UInt8>, _ srcLen: Int,
                           _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int) throws -> Int {
        guard let fns = loaded else {
            throw ParquetError.unsupported(
                "ZSTD pages need libzstd, which macOS does not ship and this SDK's Compression framework does not "
                + "implement. Install it (brew install zstd) or point ARROWMETAL_ZSTD at libzstd.1.dylib.")
        }
        let n = fns.decompress(dst, dstLen, src, srcLen)
        if fns.isError(n) != 0 { throw ParquetError.malformed("ZSTD_decompress failed") }
        return n
    }
}
