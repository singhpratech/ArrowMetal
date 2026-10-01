import XCTest
import Metal
@testable import ArrowMetal

/// Row-wise kernels (one thread per element, `Dispatch.dispatch1D` and `Dispatch.dispatchRows`) at and
/// past the 2^32-thread width of one grid dimension.
///
/// A `(groups, 1, 1)` grid of 256-thread threadgroups is 2^32 threads at 2^24 threadgroups, which the GPU
/// wraps (FINDINGS round 13): an array of 2^32 - 255 to 2^32 - 1 elements ran no thread at all. The grid
/// is folded into rows of 65,536 threadgroups there, and every kernel reads its position through the fold.
/// These tests check the rewrite of the kernel text, run row-wise kernels with the fold forced on at
/// 40M elements against the plain grid, and (with `ARROWMETAL_BIG_TESTS=1`, about 9 GB) run a kernel over
/// 2^32 - 1 elements.
final class GridRowFoldTests: XCTestCase {

    private var savedRowFold = Dispatch.rowFoldThreads
    override func setUp() {
        super.setUp()
        savedRowFold = Dispatch.rowFoldThreads
    }
    override func tearDown() {
        Dispatch.rowFoldThreads = savedRowFold
        super.tearDown()
    }

    // MARK: - the kernel text

    static let sample = """
    #include <metal_stdlib>
    using namespace metal;
    #define FOLD_ARGS device uint* out [[buffer(0)]], \\
        constant uint& n [[buffer(1)]], \\
        uint i [[thread_position_in_grid]]
    kernel void fold_index(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                           device uint* tgs [[buffer(2)]],
                           uint i [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                           uint tgid [[threadgroup_position_in_grid]]) {
        if (i < n) out[i] = i;
        if (lid == 0u && i < n) tgs[tgid] = tgid;
    }
    kernel void fold_macro(FOLD_ARGS) { if (i < n) out[i] = i + 1u; }
    kernel void fold_macro2(FOLD_ARGS, device uint* unused [[buffer(2)]]) { if (i < n) out[i] = 2u * i; }
    kernel void fold_mixed(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                           uint lid [[thread_position_in_threadgroup]], uint i [[thread_position_in_grid]],
                           uint tpg [[threads_per_threadgroup]]) {
        if (i < n) out[i] = lid + 1000u * tpg;
    }
    kernel void fold_two_d(device uint* out [[buffer(0)]], uint2 tgid2 [[threadgroup_position_in_grid]]) {
        out[tgid2.x] = tgid2.y;
    }
    """

    func testRewriteOfTheKernelText() {
        let out = Dispatch.foldGridPositions(Self.sample)
        XCTAssertFalse(out.contains("uint i [[thread_position_in_grid]]"))
        XCTAssertFalse(out.contains("uint tgid [[threadgroup_position_in_grid]]"))
        XCTAssertTrue(out.contains("uint2 am_fold_i [[thread_position_in_grid]]"))
        XCTAssertTrue(out.contains("uint2 am_fold_tgid [[threadgroup_position_in_grid]]"))
        // One declaration per kernel body, the macro's in both kernels that name it, none in the define.
        let decl = "uint i = (am_fold_i.y << \(Dispatch.rowFoldShift)u) + am_fold_i.x;"
        XCTAssertEqual(out.components(separatedBy: decl).count - 1, 4)
        XCTAssertEqual(out.components(separatedBy: "uint tgid = (am_fold_tgid.y << 16u) + am_fold_tgid.x;").count - 1, 1)
        // The other scalar position inputs of a kernel turn into uint2 read as .x, so the kind matches.
        XCTAssertTrue(out.contains("uint2 am_fold_lid [[thread_position_in_threadgroup]]"))
        XCTAssertTrue(out.contains("uint lid = am_fold_lid.x;"))
        XCTAssertTrue(out.contains("uint tpg = am_fold_tpg.x;"))
        XCTAssertTrue(out.contains("uint lid [[thread_index_in_threadgroup]]"), "an index input is left alone")
        XCTAssertTrue(out.contains("uint2 tgid2 [[threadgroup_position_in_grid]]"), "a uint2 position is left alone")
        XCTAssertEqual(Dispatch.foldGridPositions("kernel void k() {}"), "kernel void k() {}")
        XCTAssertEqual(Dispatch.rowFoldShift, 24)
    }

    func testGridShapes() {
        Dispatch.rowFoldThreads = 1 << 32
        XCTAssertEqual(Dispatch.rowGrid(threadgroups: (1 << 24) - 1).map { [$0.width, $0.height] }, [(1 << 24) - 1, 1])
        XCTAssertEqual(Dispatch.rowGrid(threadgroups: 1 << 24).map { [$0.width, $0.height] }, [1 << 16, 256])
        XCTAssertNil(Dispatch.rowGrid(threadgroups: (1 << 24) + 1), "more than 2^32 threads cannot be indexed")
        XCTAssertEqual(Dispatch.launchedThreadgroups(1 << 24), 1 << 24, "the fold at the limit launches no spare threadgroup")
        Dispatch.rowFoldThreads = 1 << 20
        XCTAssertEqual(Dispatch.rowGrid(threadgroups: 4095).map { [$0.width, $0.height] }, [4095, 1])
        XCTAssertEqual(Dispatch.rowGrid(threadgroups: 156_251).map { [$0.width, $0.height] }, [1 << 16, 3])
        XCTAssertEqual(Dispatch.launchedThreadgroups(156_251), 3 << 16)
    }

    /// The rewritten sample, compiled and run on a folded grid of three rows: every element and every
    /// threadgroup sees its row-major position, through a plain parameter and through a macro.
    func testFoldedPositionsOnTheGPU() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let n = 40_000_003
        let groups = (n + 255) / 256
        Dispatch.rowFoldThreads = 1 << 20
        let out = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: true, context: ctx)
        let tgs = try MetalArrowBuffer.allocate(byteCount: Dispatch.launchedThreadgroups(groups) * 4, zeroed: true, context: ctx)
        let kernels: [(String, (Int) -> UInt32)] = [
            ("fold_index", { UInt32($0) }), ("fold_macro", { UInt32($0 + 1) }),
            ("fold_macro2", { UInt32(truncatingIfNeeded: 2 * $0) }), ("fold_mixed", { UInt32($0 % 256 + 256_000) }),
        ]
        for (fn, want) in kernels {
            let pso = try ctx.pipeline(source: Self.sample, function: fn, cacheKey: "test/gridfold/\(fn)")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(out.mtl, offset: 0, index: 0)
                Dispatch.setUInt(enc, n, index: 1)
                enc.setBuffer(tgs.mtl, offset: 0, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            try ctx.syncPoint()
            let p = out.typed(UInt32.self)
            var bad = 0
            for i in 0..<n where p[i] != want(i) { bad += 1; if bad < 4 { XCTFail("\(fn) element \(i): \(p[i])") } }
            XCTAssertEqual(bad, 0, fn)
        }
        let t = tgs.typed(UInt32.self)
        XCTAssertTrue((0..<groups).allSatisfy { t[$0] == UInt32($0) }, "threadgroup positions")
    }

    // MARK: - row-wise kernels through the fold

    /// Element-wise kernels, the fused expression kernel and the two-level scans (cumulative sum, forward
    /// fill) with the fold forced on at 40M elements, three rows, against the plain grid.
    func testRowWiseKernelsFoldedMatchPlain() throws {
        try requireRealGPU()
        let n = 40_000_003
        var rng = SystemRandomNumberGenerator()
        let bytes = (0..<n).map { _ in UInt8.random(in: 0...255, using: &rng) }
        let ints = (0..<n).map { _ in Int32.random(in: -1000...1000, using: &rng) }
        let withNulls: [Int32?] = ints.enumerated().map { $0.offset % 5 == 2 ? nil : $0.element }
        let idx = (0..<n).map { _ in Int32.random(in: 0..<Int32(n), using: &rng) }
        let u8 = try MetalArray<UInt8>(bytes), i32 = try MetalArray<Int32>(ints)
        let nul = try MetalArray<Int32>(withNulls), ix = try MetalArray<Int32>(idx)
        let rb = try MetalRecordBatch(names: ["a", "b"], columns: [.int32(i32), .uint8(u8)])

        func results() throws -> [[Int64]] {
            let xor = try u8.bitwise(.xor, 0x5A).toRawArray().map(Int64.init)
            let not = try u8.bitwiseNot().toRawArray().map(Int64.init)
            let add = try i32.add(i32).toRawArray().map(Int64.init)
            let take = try i32.take(ix).toRawArray().map(Int64.init)
            let cum = try i32.cumulativeSum().toRawArray().map(Int64.init)
            let ffill = try nul.fillNullForward().toArray().map { $0.map(Int64.init) ?? Int64.min }
            let q = try rb.query(query().project([("e", col("a") * 3 + col("b").cast(to: .int32)),
                                                  ("s", col("a").cast(to: .float32).signBit)]))
            guard case .int32(let e)? = q["e"], case .boolean(let s)? = q["s"] else { XCTFail("query columns"); return [] }
            let sb = s.toArray().map { Int64($0 == true ? 1 : 0) }
            return [xor, not, add, take, cum, ffill, e.toRawArray().map(Int64.init), sb]
        }
        Dispatch.rowFoldThreads = 1 << 32
        let plain = try results()
        Dispatch.rowFoldThreads = 1 << 20
        let folded = try results()
        let names = ["xor", "not", "add", "take", "cumulative_sum", "fill_null_forward", "expr", "expr signbit"]
        for (k, name) in names.enumerated() {
            XCTAssertEqual(plain[k].count, n, name)
            XCTAssertTrue(plain[k] == folded[k], "\(name): folded grid differs from the plain grid")
        }
        // And against the host, so the plain grid is not just agreeing with itself.
        XCTAssertEqual(plain[0].prefix(1000), bytes.prefix(1000).map { Int64($0 ^ 0x5A) }[...])
        var run: Int64 = 0
        XCTAssertTrue(ints.enumerated().allSatisfy { run += Int64($0.element); return plain[4][$0.offset] == run })
    }

    // MARK: - at the limit

    /// A kernel over 2^32 - 1 one-byte elements: 2^24 threadgroups, the grid that ran no thread before
    /// the fold. About 8.6 GB of buffers, so it runs only with `ARROWMETAL_BIG_TESTS=1`.
    func testKernelOver2To32Minus1Elements() throws {
        try requireRealGPU()
        guard ProcessInfo.processInfo.environment["ARROWMETAL_BIG_TESTS"] == "1" else {
            throw XCTSkip("set ARROWMETAL_BIG_TESTS=1 to run the 2^32 - 1 element kernel (about 8.6 GB)")
        }
        let ctx = MetalContext.shared
        let n = (1 << 32) - 1
        guard ctx.device.maxBufferLength >= n + 16_384 else { throw XCTSkip("maxBufferLength \(ctx.device.maxBufferLength)") }
        let values = try MetalArrowBuffer.allocate(byteCount: n, zeroed: false, context: ctx)
        let p = values.mutableTyped(UInt8.self)
        // A pattern that differs between neighbours and between the start and the end of the array.
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (n / 64), hi = c == 63 ? n : (c + 1) * (n / 64)
            for i in lo..<hi { p[i] = UInt8(truncatingIfNeeded: i &* 31 &+ (i >> 20)) }
        }
        let a = MetalArray<UInt8>(length: n, nullCount: 0, validity: nil, values: values, context: ctx)
        func check(_ r: MetalArray<UInt8>, _ what: String, _ f: @escaping @Sendable (UInt8) -> UInt8) {
            let q = r.valuePointer
            let bad = FoldCheckCounter()
            DispatchQueue.concurrentPerform(iterations: 64) { c in
                let lo = c * (n / 64), hi = c == 63 ? n : (c + 1) * (n / 64)
                var local = 0
                for i in lo..<hi where q[i] != f(p[i]) { local += 1 }
                bad.add(local)
            }
            XCTAssertEqual(bad.value, 0, "\(what): wrong bytes out of \(n)")
            XCTAssertEqual(q[n - 1], f(p[n - 1]), "\(what): last element")
        }
        let start = Date()
        do { let x = try a.bitwise(.xor, 0xA5); check(x, "xor") { $0 ^ 0xA5 } }
        print("GridRowFoldTests: xor over \(n) elements, \(String(format: "%.2f", Date().timeIntervalSince(start))) s with the check")
        do { let nt = try a.bitwiseNot(); check(nt, "not") { ~$0 } }
        // The same kernel on the plain (2^24, 1, 1) grid, for the record: how many bytes it got right.
        Dispatch.rowFoldThreads = Int.max
        let plain = try a.bitwise(.xor, 0xA5)
        let q = plain.valuePointer
        let right = FoldCheckCounter()
        DispatchQueue.concurrentPerform(iterations: 64) { c in
            let lo = c * (n / 64), hi = c == 63 ? n : (c + 1) * (n / 64)
            var local = 0
            for i in lo..<hi where q[i] == p[i] ^ 0xA5 { local += 1 }
            right.add(local)
        }
        print("GridRowFoldTests: plain grid of 2^24 threadgroups: \(right.value) of \(n) bytes right")
    }
}

/// A counter the concurrent check loops add to.
final class FoldCheckCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var v = 0
    func add(_ x: Int) { lock.lock(); v += x; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return v }
}
