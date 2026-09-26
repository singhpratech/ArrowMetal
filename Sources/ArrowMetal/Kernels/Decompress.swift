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
/// SNAPPY, LZ4 and LZ4_RAW pages are split between the GPU (`DecompressSource`: a SIMD group per page,
/// or a thread per page) and the host (`SnappyHost`, `LZ4Host`), page by page, by `DecodeRouter`, and
/// both sides run at once (`Decompress.overlapped`, in `DecodeSplit.swift`). UNCOMPRESSED needs no work
/// at all: the mapped file *is* the page buffer, so the reader passes the file's own `MTLBuffer`
/// straight to the decoders.
///
/// ZSTD, GZIP and BROTLI are decompressed on the host only, straight into the shared-memory output
/// buffer the GPU decoders will read, in the same schedule. GZIP and BROTLI go through Foundation's
/// Compression framework (`COMPRESSION_ZLIB` is raw DEFLATE, so the gzip container is stripped first).
/// The SDK has no `COMPRESSION_ZSTD`, so libzstd is looked up with `dlopen` at first use; when it is not
/// installed, a ZSTD column raises `ParquetError.unsupported` naming the missing library rather than
/// returning wrong data.
enum Decompress {
    private static func blockBuffer(_ ctx: MetalContext, _ blocks: [PageBlock]) throws -> MetalArrowBuffer {
        let buf = try MetalArrowBuffer.allocate(byteCount: blocks.count * MemoryLayout<PageBlock>.stride,
                                                zeroed: false, context: ctx)
        blocks.withUnsafeBytes { memcpy(buf.mutableContents, $0.baseAddress!, $0.count) }
        return buf
    }

    static func gpuCopy(_ ctx: MetalContext, source: MTLBuffer, sourceOffset: Int,
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

    // MARK: - Host codecs

    static func gunzipBlock(_ src: UnsafePointer<UInt8>, _ srcLen: Int,
                            _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int) throws -> Int {
        try gunzip(src, srcLen, dst, dstLen)
    }
    static func brotliBlock(_ src: UnsafePointer<UInt8>, _ srcLen: Int,
                            _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int) throws -> Int {
        try appleDecode(COMPRESSION_BROTLI, src, srcLen, dst, dstLen)
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
    typealias CreateDCtxFn = @convention(c) () -> OpaquePointer?
    typealias FreeDCtxFn = @convention(c) (OpaquePointer?) -> Int
    typealias DecompressDCtxFn = @convention(c) (OpaquePointer?, UnsafeMutableRawPointer?, Int, UnsafeRawPointer?, Int) -> Int

    /// A reusable decompression context (`ZSTD_DCtx`), freed with the object. Nil inside when this
    /// libzstd lacks the context API, and `decompress` then uses the one-shot call.
    final class Context {
        let dctx: OpaquePointer?
        init?() {
            guard let c = Zstd.contextAPI, let d = c.create() else { return nil }
            dctx = d
        }
        deinit { if let c = Zstd.contextAPI { _ = c.free(dctx) } }
    }

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

    private static let contextAPI: (create: CreateDCtxFn, free: FreeDCtxFn, decompress: DecompressDCtxFn)? = {
        guard loaded != nil else { return nil }
        for name in candidates {
            guard let h = dlopen(name, RTLD_LAZY) else { continue }
            guard let c = dlsym(h, "ZSTD_createDCtx"), let f = dlsym(h, "ZSTD_freeDCtx"),
                  let d = dlsym(h, "ZSTD_decompressDCtx") else { continue }
            return (unsafeBitCast(c, to: CreateDCtxFn.self), unsafeBitCast(f, to: FreeDCtxFn.self),
                    unsafeBitCast(d, to: DecompressDCtxFn.self))
        }
        return nil
    }()

    static var isAvailable: Bool { loaded != nil }

    static func decompress(_ src: UnsafePointer<UInt8>, _ srcLen: Int,
                           _ dst: UnsafeMutablePointer<UInt8>, _ dstLen: Int, context: Context? = nil) throws -> Int {
        guard let fns = loaded else {
            throw ParquetError.unsupported(
                "ZSTD pages need libzstd, which macOS does not ship and this SDK's Compression framework does not "
                + "implement. Install it (brew install zstd) or point ARROWMETAL_ZSTD at libzstd.1.dylib.")
        }
        let n: Int
        if let context, let api = contextAPI { n = api.decompress(context.dctx, dst, dstLen, src, srcLen) }
        else { n = fns.decompress(dst, dstLen, src, srcLen) }
        if fns.isError(n) != 0 { throw ParquetError.malformed("ZSTD_decompress failed") }
        return n
    }
}

/// Copies for the host decoders. Every one of them stays inside `[0, cap)` of the page's own output
/// slot and inside the page's own input: the wide forms are only taken when the whole 16 or 8 bytes
/// they touch are inside.
@inline(__always)
private func copy16(_ d: UnsafeMutablePointer<UInt8>, _ s: UnsafePointer<UInt8>) {
    UnsafeMutableRawPointer(d).storeBytes(of: UnsafeRawPointer(s).loadUnaligned(as: SIMD16<UInt8>.self), as: SIMD16<UInt8>.self)
}
@inline(__always)
private func copy8(_ d: UnsafeMutablePointer<UInt8>, _ s: UnsafePointer<UInt8>) {
    UnsafeMutableRawPointer(d).storeBytes(of: UnsafeRawPointer(s).loadUnaligned(as: UInt64.self), as: UInt64.self)
}

/// A back-reference: `length` bytes from `offset` back, into `dst[op..<op+length]`, where
/// `op + length <= cap` and `0 < offset <= op` have been checked. Byte `k` of the match is byte
/// `k - offset` of the output, so the match repeats a pattern of `offset` bytes and also repeats at `d`,
/// the first multiple of `offset` that is at least 8, from `d - offset` bytes in. With `d` bytes between
/// source and destination an 8-byte step reads only bytes already written, so after the first
/// `d - offset` bytes (none when `offset >= 8`) the match moves in 8-byte steps and a tail of under 8
/// bytes. Nothing is written at or past `cap`.
@inline(__always)
private func backCopy(_ dst: UnsafeMutablePointer<UInt8>, _ op: Int, _ offset: Int, _ length: Int, _ cap: Int) {
    let d = offset >= 8 ? offset : offset * ((8 + offset - 1) / offset)
    let p = Swift.min(length, d - offset)
    for k in 0..<p { dst[op + k] = dst[op - offset + k] }
    var k = p
    if op + length + 8 <= cap {
        // Room for a whole last step inside the slot: the steps may run up to 7 bytes past the match,
        // bytes the next token overwrites.
        while k < length { copy8(dst + (op + k), UnsafePointer(dst + (op + k - d))); k += 8 }
        return
    }
    while k + 8 <= length { copy8(dst + (op + k), UnsafePointer(dst + (op + k - d))); k += 8 }
    while k < length { dst[op + k] = dst[op + k - d]; k += 1 }
}

/// A Snappy block decoder for the host (the format: a varint of the plaintext length, then literal and
/// copy elements). Every read and write is checked against the block's bounds, so a damaged page is an
/// error, never an access outside the page or the output slot.
enum SnappyHost {
    static func decompress(_ src: UnsafePointer<UInt8>, _ n: Int,
                           _ dst: UnsafeMutablePointer<UInt8>, _ cap: Int) throws -> Int {
        func bad(_ why: String) -> ParquetError { ParquetError.malformed("SNAPPY page failed to decompress: \(why)") }
        var ip = 0
        var declared = 0
        var shift = 0
        while true {
            guard ip < n, shift <= 28 else { throw bad("input truncated") }
            let b = Int(src[ip]); ip += 1
            declared |= (b & 0x7F) << shift
            if b < 0x80 { break }
            shift += 7
        }
        guard declared == cap else { throw bad("output overrun") }
        var op = 0
        while ip < n {
            let tag = Int(src[ip]); ip += 1
            var length: Int
            var offset: Int
            switch tag & 3 {
            case 0:
                length = tag >> 2
                if length < 16 && ip + 16 <= n && op + 16 <= cap {
                    // A literal of at most 16 bytes with room on both sides: one wide copy.
                    copy16(dst + op, src + ip)
                    ip += length + 1
                    op += length + 1
                    continue
                }
                if length >= 60 {
                    let extra = length - 59
                    guard ip + extra <= n else { throw bad("input truncated") }
                    length = 0
                    for k in 0..<extra { length |= Int(src[ip + k]) << (8 * k) }
                    ip += extra
                }
                length += 1
                guard length <= n - ip else { throw bad("input truncated") }
                guard length <= cap - op else { throw bad("output overrun") }
                memcpy(dst + op, src + ip, length)
                ip += length
                op += length
                continue
            case 1:
                guard ip < n else { throw bad("input truncated") }
                length = ((tag >> 2) & 7) + 4
                offset = ((tag >> 5) << 8) | Int(src[ip])
                ip += 1
            case 2:
                guard ip + 2 <= n else { throw bad("input truncated") }
                length = (tag >> 2) + 1
                offset = Int(src[ip]) | (Int(src[ip + 1]) << 8)
                ip += 2
            default:
                guard ip + 4 <= n else { throw bad("input truncated") }
                length = (tag >> 2) + 1
                offset = Int(src[ip]) | (Int(src[ip + 1]) << 8) | (Int(src[ip + 2]) << 16) | (Int(src[ip + 3]) << 24)
                ip += 4
            }
            guard offset > 0, offset <= op else { throw bad("bad token") }
            guard length <= cap - op else { throw bad("output overrun") }
            backCopy(dst, op, offset, length, cap)
            op += length
        }
        return op
    }
}

/// An LZ4 block decoder for the host, with the checks, the order of the checks and the outcomes of the
/// GPU kernels (`lz4_decompress`): the Hadoop framing Parquet's legacy LZ4 codec writes (big-endian
/// plaintext and block lengths) is recognised and stripped the same way, and a block must fill its
/// output exactly. Every read and write is checked against the block's bounds.
enum LZ4Host {
    static func decompress(_ src0: UnsafePointer<UInt8>, _ n0: Int,
                           _ dst: UnsafeMutablePointer<UInt8>, _ cap: Int) throws -> Int {
        func bad(_ why: String) -> ParquetError { ParquetError.malformed("LZ4 page failed to decompress: \(why)") }
        var src = src0, n = n0
        if n >= 8 {
            let u = (Int(src[0]) << 24) | (Int(src[1]) << 16) | (Int(src[2]) << 8) | Int(src[3])
            let c = (Int(src[4]) << 24) | (Int(src[5]) << 16) | (Int(src[6]) << 8) | Int(src[7])
            if u == cap && c == n - 8 { src += 8; n -= 8 }
        }
        var ip = 0, op = 0
        while op < cap {
            // A short sequence (literal run under 15 bytes, match under 19) with room for 48 bytes of
            // output and 32 of input: one 16-byte literal copy and the match in 8-byte steps, every
            // bound implied by the room.
            if op + 48 <= cap && ip + 32 <= n {
                let token = Int(src[ip])
                let lit = token >> 4, m = token & 15
                if lit < 15 && m < 15 {
                    let offset = Int(src[ip + 1 + lit]) | (Int(src[ip + 2 + lit]) << 8)
                    guard offset != 0, offset <= op + lit else { throw bad("bad token") }
                    copy16(dst + op, src + (ip + 1))
                    ip += 3 + lit
                    op += lit
                    backCopy(dst, op, offset, m + 4, cap)
                    op += m + 4
                    continue
                }
            }
            guard ip < n else { throw bad("input truncated") }
            let token = Int(src[ip]); ip += 1
            var litLen = token >> 4
            if litLen == 15 {
                var c2 = 255
                while c2 == 255 && ip < n { c2 = Int(src[ip]); ip += 1; litLen += c2 }
            }
            guard litLen <= n - ip, litLen <= cap - op else { throw bad("output overrun") }
            let litSrc = ip
            ip += litLen
            guard ip + 2 <= n else {
                // The last sequence: literals only.
                memcpy(dst + op, src + litSrc, litLen)
                op += litLen
                break
            }
            let offset = Int(src[ip]) | (Int(src[ip + 1]) << 8)
            ip += 2
            var matchLen = token & 15
            if matchLen == 15 {
                var c2 = 255
                while c2 == 255 && ip < n { c2 = Int(src[ip]); ip += 1; matchLen += c2 }
            }
            matchLen += 4
            guard offset != 0, offset <= op + litLen, matchLen <= cap - op - litLen else { throw bad("bad token") }
            if litLen <= 16 && litSrc + 16 <= n && op + 16 <= cap {
                copy16(dst + op, src + litSrc)
            } else {
                memcpy(dst + op, src + litSrc, litLen)
            }
            op += litLen
            backCopy(dst, op, offset, matchLen, cap)
            op += matchLen
        }
        guard op == cap else { throw bad("input truncated") }
        return op
    }
}
