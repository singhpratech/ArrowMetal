import XCTest
import Compression
@testable import ArrowMetal

/// The CPU + GPU page decompression split: the host decoders against a byte-at-a-time reference and
/// under damage (with guard pages on both sides of every block), the router's decisions, and the
/// staging layout that keeps the host's pages and the GPU's in separate pages of memory.
final class DecompressSplitTests: XCTestCase {

    // MARK: - reference decoders, one byte at a time

    static func referenceSnappy(_ s: [UInt8]) -> [UInt8]? {
        var ip = 0, n = 0, shift = 0
        while true {
            guard ip < s.count, shift <= 28 else { return nil }
            let b = Int(s[ip]); ip += 1
            n |= (b & 0x7F) << shift
            if b < 0x80 { break }
            shift += 7
        }
        var out: [UInt8] = []
        while ip < s.count {
            let tag = Int(s[ip]); ip += 1
            var len = 0, off = 0
            switch tag & 3 {
            case 0:
                len = tag >> 2
                if len >= 60 {
                    let extra = len - 59
                    guard ip + extra <= s.count else { return nil }
                    len = 0
                    for k in 0..<extra { len |= Int(s[ip + k]) << (8 * k) }
                    ip += extra
                }
                len += 1
                guard ip + len <= s.count else { return nil }
                out += s[ip..<(ip + len)]; ip += len
                continue
            case 1:
                guard ip < s.count else { return nil }
                len = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | Int(s[ip]); ip += 1
            case 2:
                guard ip + 2 <= s.count else { return nil }
                len = (tag >> 2) + 1; off = Int(s[ip]) | Int(s[ip + 1]) << 8; ip += 2
            default:
                guard ip + 4 <= s.count else { return nil }
                len = (tag >> 2) + 1
                off = Int(s[ip]) | Int(s[ip + 1]) << 8 | Int(s[ip + 2]) << 16 | Int(s[ip + 3]) << 24; ip += 4
            }
            guard off > 0, off <= out.count else { return nil }
            for _ in 0..<len { out.append(out[out.count - off]) }
        }
        return out.count == n ? out : nil
    }

    static func referenceLZ4(_ s: [UInt8], _ n: Int) -> [UInt8]? {
        var ip = 0
        var out: [UInt8] = []
        while out.count < n {
            guard ip < s.count else { return nil }
            let token = Int(s[ip]); ip += 1
            var lit = token >> 4
            if lit == 15 { var c = 255; while c == 255 && ip < s.count { c = Int(s[ip]); ip += 1; lit += c } }
            guard ip + lit <= s.count, out.count + lit <= n else { return nil }
            out += s[ip..<(ip + lit)]; ip += lit
            guard ip + 2 <= s.count else { break }
            let off = Int(s[ip]) | Int(s[ip + 1]) << 8; ip += 2
            var m = token & 15
            if m == 15 { var c = 255; while c == 255 && ip < s.count { c = Int(s[ip]); ip += 1; m += c } }
            m += 4
            guard off > 0, off <= out.count, out.count + m <= n else { return nil }
            for _ in 0..<m { out.append(out[out.count - off]) }
        }
        return out.count == n ? out : nil
    }

    /// Plaintexts that give every kind of token: runs (offsets 1 to 16), sequential and low-cardinality
    /// integers, random bytes, and a long run that ends exactly at the end of the block.
    static func plaintexts() -> [[UInt8]] {
        var rng = SystemRandomNumberGenerator()
        var out: [[UInt8]] = []
        for period in 1...16 {
            let pattern = (0..<period).map { _ in UInt8.random(in: 0...255, using: &rng) }
            out.append((0..<(3000 + period * 37)).map { pattern[$0 % period] })
        }
        out.append((0..<20_000).flatMap { withUnsafeBytes(of: Int64($0)) { Array($0) } })
        out.append((0..<20_000).flatMap { _ in withUnsafeBytes(of: Int64.random(in: 0..<1000, using: &rng)) { Array($0) } })
        out.append((0..<50_000).map { _ in UInt8.random(in: 0...255, using: &rng) })
        out.append((0..<20_000).map { _ in UInt8.random(in: 0...255, using: &rng) } + [UInt8](repeating: 7, count: 70_000))
        out.append((0..<40_000).map { UInt8(($0 / 3) % 5) })
        out.append([1, 2, 3])
        out.append([])
        return out
    }

    static func lz4(_ p: [UInt8]) -> [UInt8] {
        var dst = [UInt8](repeating: 0, count: p.count + p.count / 200 + 64)
        let n = p.isEmpty ? 0 : compression_encode_buffer(&dst, dst.count, p, p.count, nil, COMPRESSION_LZ4_RAW)
        return Array(dst[0..<n])
    }

    /// Runs `body` with `input` ending right before an inaccessible page and an output slot of `cap`
    /// bytes ending right before another, so a read or write past either end is a crash, not a pass.
    static func guarded(_ input: [UInt8], cap: Int,
                        _ body: (UnsafePointer<UInt8>, Int, UnsafeMutablePointer<UInt8>, Int) throws -> Int) -> (Int?, [UInt8]) {
        let page = Int(getpagesize())
        func region(_ bytes: Int) -> (UnsafeMutableRawPointer, Int, UnsafeMutableRawPointer) {
            let data = roundUp(Swift.max(bytes, 1), to: page)
            let total = data + page
            let base = mmap(nil, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)!
            mprotect(base.advanced(by: data), page, PROT_NONE)
            return (base, total, base.advanced(by: data - bytes))
        }
        let (ib, itotal, ip) = region(input.count)
        let (ob, ototal, op) = region(cap)
        defer { munmap(ib, itotal); munmap(ob, ototal) }
        input.withUnsafeBytes { if !$0.isEmpty { memcpy(ip, $0.baseAddress!, $0.count) } }
        let produced = try? body(UnsafePointer(ip.assumingMemoryBound(to: UInt8.self)), input.count,
                                 op.assumingMemoryBound(to: UInt8.self), cap)
        return (produced, Array(UnsafeBufferPointer(start: op.assumingMemoryBound(to: UInt8.self), count: cap)))
    }

    // MARK: - host decoders

    func testHostSnappyMatchesTheReference() {
        for (i, p) in Self.plaintexts().enumerated() {
            let c = Snappy.compress(p)
            XCTAssertEqual(Self.referenceSnappy(c), p, "case \(i): reference")
            let (n, out) = Self.guarded(c, cap: p.count) { try SnappyHost.decompress($0, $1, $2, $3) }
            XCTAssertEqual(n, p.count, "case \(i)")
            XCTAssertEqual(out, p, "case \(i)")
        }
    }

    func testHostLZ4MatchesTheReference() {
        // Apple's encoder declines inputs of a few bytes.
        for (i, p) in Self.plaintexts().enumerated() where p.count > 16 {
            let c = Self.lz4(p)
            XCTAssertFalse(c.isEmpty, "case \(i)")
            XCTAssertEqual(Self.referenceLZ4(c, p.count), p, "case \(i): reference")
            let (n, out) = Self.guarded(c, cap: p.count) { try LZ4Host.decompress($0, $1, $2, $3) }
            XCTAssertEqual(n, p.count, "case \(i)")
            XCTAssertEqual(out, p, "case \(i)")
            // Parquet's legacy LZ4 codec: the same block behind the Hadoop framing.
            var framed: [UInt8] = []
            for v in [UInt32(p.count), UInt32(c.count)] { framed += withUnsafeBytes(of: v.bigEndian) { Array($0) } }
            framed += c
            let (n2, out2) = Self.guarded(framed, cap: p.count) { try LZ4Host.decompress($0, $1, $2, $3) }
            XCTAssertEqual(n2, p.count, "case \(i) framed")
            XCTAssertEqual(out2, p, "case \(i) framed")
        }
    }

    /// Hand-built tokens: overlapping copies at every offset from 1 to 16 and lengths up to 64, the last
    /// one ending exactly at the end of the output, where no step may run past it.
    func testOverlappingCopiesEndingAtTheSlotEnd() {
        for offset in 1...16 {
            for length in [4, 5, 7, 8, 9, 15, 16, 17, 31, 64] {
                let seed = (0..<offset).map { UInt8(($0 * 37 + offset) & 0xFF) }
                let plain = (0..<(offset + length)).map { seed[$0 % offset] }
                var s: [UInt8] = [UInt8(plain.count)]                           // varint (< 128)
                s.append(UInt8((offset - 1) << 2))                               // literal of `offset` bytes
                s += seed
                s += [UInt8(((length - 1) << 2) | 2), UInt8(offset), 0]         // copy with a 2-byte offset
                let (n, out) = Self.guarded(s, cap: plain.count) { try SnappyHost.decompress($0, $1, $2, $3) }
                XCTAssertEqual(n, plain.count, "snappy offset \(offset) length \(length)")
                XCTAssertEqual(out, plain, "snappy offset \(offset) length \(length)")

                var l: [UInt8] = [UInt8(min(offset, 15) << 4 | min(length - 4, 15))]
                if offset >= 15 { l.append(UInt8(offset - 15)) }
                l += seed
                l += [UInt8(offset), 0]
                if length - 4 >= 15 { l.append(UInt8(length - 4 - 15)) }
                l.append(0)                                                      // an empty last sequence
                let (m, out2) = Self.guarded(l, cap: plain.count) { try LZ4Host.decompress($0, $1, $2, $3) }
                XCTAssertEqual(m, plain.count, "lz4 offset \(offset) length \(length)")
                XCTAssertEqual(out2, plain, "lz4 offset \(offset) length \(length)")
            }
        }
    }

    /// Damaged blocks, between guard pages: every decode throws or fills its slot, and never touches a
    /// byte outside its input or its output slot (the guard pages would end the process).
    func testDamagedBlocksStayInsideTheirBounds() {
        var rng = SystemRandomNumberGenerator()
        var raised = 0, total = 0
        for p in Self.plaintexts() where p.count > 16 {
            let s = Snappy.compress(p), l = Self.lz4(p)
            for _ in 0..<40 {
                for (codec, good) in [(ParquetCodec.snappy, s), (.lz4, l)] {
                    var bad = good
                    for _ in 0..<Int.random(in: 1...4, using: &rng) {
                        let at = Int.random(in: 0..<bad.count, using: &rng)
                        bad[at] = bad[at] &+ UInt8.random(in: 1...255, using: &rng)
                    }
                    if Bool.random(using: &rng) { bad = Array(bad.prefix(Int.random(in: 0..<bad.count, using: &rng))) }
                    let cap = Bool.random(using: &rng) ? p.count : Int.random(in: 0...(p.count + 64), using: &rng)
                    let (n, _) = Self.guarded(bad, cap: cap) {
                        codec == .snappy ? try SnappyHost.decompress($0, $1, $2, $3) : try LZ4Host.decompress($0, $1, $2, $3)
                    }
                    total += 1
                    if let n { XCTAssertLessThanOrEqual(n, cap) } else { raised += 1 }
                }
            }
        }
        XCTAssertGreaterThan(raised, total / 4)
    }

    // MARK: - router

    private func dense(_ codec: ParquetCodec = .snappy, _ bytes: Int = 160_000) -> DecodeCandidate {
        DecodeCandidate(codec: codec, srcLength: bytes / 2, dstLength: bytes)
    }
    private func literal(_ bytes: Int = 160_000) -> DecodeCandidate {
        DecodeCandidate(codec: .snappy, srcLength: bytes + 20, dstLength: bytes)
    }

    func testRouterGivesTheHostTokenDensePagesAndTheGPUTheLiteralOnes() {
        let saved = DecodeRouter.forced
        defer { DecodeRouter.forced = saved }
        DecodeRouter.forced = nil
        // The shape of the 50 M-row benchmark file: 7,575 token-dense pages and 7,575 literal ones.
        let cands: [DecodeCandidate?] = (0..<7575).map { _ in dense() } + (0..<7575).map { _ in literal() }
        let sides = DecodeRouter.route(cands)
        let denseHost = sides[0..<7575].filter { $0 == .host }.count
        let literalGPU = sides[7575...].filter { $0 != .host }.count
        XCTAssertGreaterThan(denseHost, 7575 * 8 / 10, "most token-dense pages go to the host")
        XCTAssertEqual(literalGPU, 7575, "every literal page stays on the GPU")
    }

    func testRouterKeepsAFewPagesOnTheHost() {
        let saved = DecodeRouter.forced
        defer { DecodeRouter.forced = saved }
        DecodeRouter.forced = nil
        // One column of a small file: the GPU would decode each page alone.
        XCTAssertEqual(DecodeRouter.route((0..<6).map { _ in dense() }), [DecodeSide](repeating: .host, count: 6))
        // One large dictionary page.
        XCTAssertEqual(DecodeRouter.route([DecodeCandidate(codec: .snappy, srcLength: 600_000, dstLength: 790_000)]), [.host])
    }

    func testRouterSendsHostCodecsToTheHostAndPlainPagesToTheGPU() {
        let saved = DecodeRouter.forced
        defer { DecodeRouter.forced = saved }
        for forced: DecodeSide? in [nil, .gpuGroup, .gpuLane] {
            DecodeRouter.forced = forced
            let sides = DecodeRouter.route([DecodeCandidate(codec: .zstd, srcLength: 100, dstLength: 400), nil,
                                            DecodeCandidate(codec: .gzip, srcLength: 100, dstLength: 400),
                                            DecodeCandidate(codec: .brotli, srcLength: 100, dstLength: 400)])
            XCTAssertEqual(sides, [.host, .gpuGroup, .host, .host], "\(String(describing: forced))")
        }
        DecodeRouter.forced = .gpuLane
        XCTAssertEqual(DecodeRouter.route([dense(), literal(), dense(.lz4)]), [.gpuLane, .gpuLane, .gpuLane])
        DecodeRouter.forced = .host
        XCTAssertEqual(DecodeRouter.route([dense(), literal(), dense(.lz4Raw)]), [.host, .host, .host])
    }

    // MARK: - staging layout

    /// With pages on both sides, no page of memory holds bytes the host writes and bytes the GPU writes,
    /// and every page still has its own slot.
    func testHostAndGPUPagesNeverShareAPageOfMemory() throws {
        let path = ParquetTests.fixtures.appendingPathComponent("nulls__v2_snappy.parquet").path
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("fixture missing") }
        let f = try ParquetFile(path: path)
        let groups = Array(0..<f.rowGroupCount)
        let page = metalPageSize()
        var checked = 0
        for leaf in f.leaves where leaf.maxRepetition == 0 {
            let lp = try f.collectPages(leaf, rowGroups: groups, plan: nil, subset: false)
            let n = lp.dictPages.count + lp.dataPages.count
            guard n >= 2 else { continue }
            for pattern in 0..<3 {
                let sides: [DecodeSide] = (0..<n).map { i in
                    switch pattern { case 0: return i % 2 == 0 ? .host : .gpuGroup
                                     case 1: return i % 3 == 0 ? .gpuLane : .host
                                     default: return i == n - 1 ? .host : .gpuGroup }
                }
                let st = f.stagePages(lp, rel: { UInt32($0.bodyOffset) }, sides: sides)
                let hostBlocks = st.host.map { $0.block }
                let gpuBlocks = st.copies + st.group.values.flatMap { $0 } + st.lane.values.flatMap { $0 }
                func pages(_ bs: [PageBlock]) -> Set<Int> {
                    Set(bs.filter { $0.dstLength > 0 }.flatMap { b in
                        (Int(b.dstOffset) / page)...((Int(b.dstOffset) + Int(b.dstLength) - 1) / page)
                    })
                }
                XCTAssertTrue(pages(hostBlocks).isDisjoint(with: pages(gpuBlocks)), "\(leaf.name) pattern \(pattern)")
                // Every page has its own slot, inside the staging buffer.
                let slots = (st.dictOffsets + st.dataOffsets).map(Int.init)
                XCTAssertEqual(Set(slots).count, slots.count)
                for b in hostBlocks + gpuBlocks { XCTAssertLessThanOrEqual(Int(b.dstOffset) + Int(b.dstLength), st.size) }
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 0)
    }
}
