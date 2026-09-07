import XCTest
@testable import ArrowMetal

/// The string kernels at the boundaries the brief names, and the batching shapes around them.
/// Every expectation on this page was read off pyarrow 25.0.1 (`pc.match_like`, `pc.split_pattern`,
/// `pc.utf8_upper` / `utf8_lower` / `utf8_swapcase`), which is the oracle these kernels answer to.
final class AdversarialStringAndBatchTests: XCTestCase {

    // MARK: - LIKE

    private static let likeRows = ["", "a", "ab", "abc", "a%b", "a_b", "aXb", "abab", "ba", "\u{00E9}x"]

    private func assertLike(_ pattern: String, _ expected: [Bool],
                            file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalStringArray(Self.likeRows.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.matchLike(pattern).toArray().map { $0 ?? false }, expected,
                       "LIKE \(pattern)", file: file, line: line)
    }

    /// `%%`, a bare `%`, `_` at either end, an empty pattern, a pattern longer than every row, and a
    /// `_` over a two-byte code point (Arrow's `_` is one *character*, not one byte).
    func testLikeAtItsBoundaries() throws {
        try requireRealGPU()
        try assertLike("%%", [true, true, true, true, true, true, true, true, true, true])
        try assertLike("%", [true, true, true, true, true, true, true, true, true, true])
        try assertLike("", [true, false, false, false, false, false, false, false, false, false])
        try assertLike("_", [false, true, false, false, false, false, false, false, false, false])
        try assertLike("_%", [false, true, true, true, true, true, true, true, true, true])
        try assertLike("%_", [false, true, true, true, true, true, true, true, true, true])
        try assertLike("_b", [false, false, true, false, false, false, false, false, false, false])
        try assertLike("a_", [false, false, true, false, false, false, false, false, false, false])
        try assertLike("abcd", [false, false, false, false, false, false, false, false, false, false])
        try assertLike("%a%b%", [false, false, true, true, true, true, true, true, false, false])
        // `__` matches "éx": two code points, four bytes.
        try assertLike("__", [false, false, true, false, false, false, false, false, true, true])
    }

    /// A backslash escapes the following character and is dropped, which is what pyarrow does too:
    /// `pc.match_like(pa.array(["a\\nb", "anb"]), "a\\nb")` is `[False, True]`.
    func testLikeEscapes() throws {
        try requireRealGPU()
        let rows = ["a\\nb", "anb", "a%b", "ab", "a_b", "axb"]
        let a = try MetalStringArray(rows.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.matchLike("a\\nb").toArray().map { $0 ?? false },
                       [false, true, false, false, false, false])
        XCTAssertEqual(try a.matchLike("a\\%b").toArray().map { $0 ?? false },
                       [false, false, true, false, false, false])
        XCTAssertEqual(try a.matchLike("a\\_b").toArray().map { $0 ?? false },
                       [false, false, false, false, true, false])
        XCTAssertEqual(try a.matchLike("a\\\\b").toArray().map { $0 ?? false },
                       [false, false, false, false, false, false])
        XCTAssertEqual(try a.matchLike("%\\%%").toArray().map { $0 ?? false },
                       [false, false, true, false, false, false])
    }

    func testLikeOnNullsAndOnASlice() throws {
        try requireRealGPU()
        let xs: [String?] = ["abc", nil, "xbz", nil, "b"]
        let a = try MetalStringArray(xs, context: .shared)
        XCTAssertEqual(try a.matchLike("%b%").toArray(), [true, nil, true, nil, true])
        for off in [1, 31, 32, 33] {
            let long: [String?] = (0..<200).map { $0 % 7 == 0 ? nil : "row\($0 % 3 == 0 ? "b" : "c")end" }
            let full = try MetalStringArray(long, context: .shared)
            let s = try full.slice(offset: off, length: 100)
            let standalone = try MetalStringArray(Array(long[off..<(off + 100)]), context: .shared)
            XCTAssertEqual(try s.matchLike("%b%").toArray(), try standalone.matchLike("%b%").toArray(),
                           "LIKE on slice offset \(off)")
        }
    }

    // MARK: - split

    private func split(_ rows: [String], _ sep: String, maxSplits: Int = -1,
                       reverse: Bool = false) throws -> [[String]] {
        let a = try MetalStringArray(rows.map { Optional($0) }, context: .shared)
        let (offsets, pieces) = try a.splitPatternPair(sep, maxSplits: maxSplits, reverse: reverse)
        let offs = offsets.toArray().map { Int($0!) }
        let flat = pieces.toArray()
        return (0..<rows.count).map { i in Array(flat[offs[i]..<offs[i + 1]]).map { $0! } }
    }

    /// Consecutive separators, a separator at either end, a row that is only separators, an empty row,
    /// a separator longer than the row, `max_splits` of 0, and a multi-byte (utf8) separator.
    func testSplitAtItsBoundaries() throws {
        try requireRealGPU()
        let rows = ["a,b,,c", ",a,", "abc", "", "a,,,b"]
        XCTAssertEqual(try split(rows, ","),
                       [["a", "b", "", "c"], ["", "a", ""], ["abc"], [""], ["a", "", "", "b"]])
        XCTAssertEqual(try split(rows, ",", maxSplits: 0),
                       [["a,b,,c"], [",a,"], ["abc"], [""], ["a,,,b"]])
        XCTAssertEqual(try split(rows, ",", maxSplits: 1),
                       [["a", "b,,c"], ["", "a,"], ["abc"], [""], ["a", ",,b"]])
        XCTAssertEqual(try split(rows, ",", maxSplits: 2),
                       [["a", "b", ",c"], ["", "a", ""], ["abc"], [""], ["a", "", ",b"]])
        XCTAssertEqual(try split(rows, ",", maxSplits: 1, reverse: true),
                       [["a,b,", "c"], [",a", ""], ["abc"], [""], ["a,,", "b"]])
        XCTAssertEqual(try split(["aXXbXXc", "XXaXX"], "XX"), [["a", "b", "c"], ["", "a", ""]])
        XCTAssertEqual(try split(["a\u{00E9}b", "\u{00E9}\u{00E9}\u{00E9}"], "\u{00E9}"),
                       [["a", "b"], ["", "", "", ""]])
        XCTAssertEqual(try split(["ab"], "abcdef"), [["ab"]], "a separator longer than the row")
        XCTAssertThrowsError(try MetalStringArray(["a"]).splitPattern(""), "an empty separator is rejected")
    }

    // MARK: - case mapping

    /// The rows whose case mapping changes the byte length, which is what the two-pass output sizing
    /// exists for: `ß` (2 bytes) uppercases to `ẞ` (3), `ı` (2) uppercases to `I` (1), and `straße`
    /// (7) to `STRAẞE` (8). pyarrow uses the *simple* mapping here, so `ß` does not become `SS`.
    func testCaseMappingWhereTheOutputLengthChanges() throws {
        try requireRealGPU()
        // Compared as scalar arrays, not as `String`: Swift compares strings under canonical
        // equivalence, so `"\u{00C9}" == "E\u{0301}"` and a normalisation difference would hide.
        func scalars(_ s: String) -> [UInt32] { s.unicodeScalars.map { $0.value } }
        let rows = ["\u{00DF}",          // ß  -> ẞ, 2 bytes becoming 3
                    "\u{0130}",          // İ  -> İ, unchanged
                    "\u{0131}",          // ı  -> I, 2 bytes becoming 1
                    "stra\u{00DF}e",     // 7 bytes becoming 8
                    "\u{1F600}a",        // a 4-byte emoji beside a letter
                    "\u{00E9}",          // é precomposed
                    "e\u{0301}",         // é decomposed: the combining mark must pass through
                    "\u{00B5}",          // µ  -> Μ
                    "\u{01C5}",          // ǅ  titlecase: upper Ǆ, lower ǆ, swapcase unchanged
                    "I", "i", "\u{0178}", "\u{00FF}"]
        let a = try MetalStringArray(rows.map { Optional($0) }, context: .shared)
        XCTAssertEqual(try a.utf8Upper().toArray().map { scalars($0!) },
                       [[0x1E9E], [0x130], [0x49], [0x53, 0x54, 0x52, 0x41, 0x1E9E, 0x45],
                        [0x1F600, 0x41], [0xC9], [0x45, 0x301], [0x39C], [0x1C4],
                        [0x49], [0x49], [0x178], [0x178]])
        XCTAssertEqual(try a.utf8Lower().toArray().map { scalars($0!) },
                       [[0xDF], [0x69], [0x131], [0x73, 0x74, 0x72, 0x61, 0xDF, 0x65],
                        [0x1F600, 0x61], [0xE9], [0x65, 0x301], [0xB5], [0x1C6],
                        [0x69], [0x69], [0xFF], [0xFF]])
        XCTAssertEqual(try a.unicodeSwapcase().toArray().map { scalars($0!) },
                       [[0x1E9E], [0x69], [0x49], [0x53, 0x54, 0x52, 0x41, 0x1E9E, 0x45],
                        [0x1F600, 0x41], [0xC9], [0x45, 0x301], [0x39C], [0x1C5],
                        [0x69], [0x49], [0xFF], [0x178]])
        // The uppercase form really is longer than the input for `ß` and `straße`, so the data buffer
        // has to be sized from a measured pass and not from the input length.
        XCTAssertEqual(rows.map { $0.utf8.count }, [2, 2, 2, 7, 5, 2, 3, 2, 2, 1, 1, 2, 2])
        XCTAssertEqual(try a.utf8Upper().toArray().map { $0!.utf8.count },
                       [3, 2, 1, 8, 5, 2, 3, 2, 2, 1, 1, 2, 2])
    }

    /// Many rows that each grow, so the whole output buffer is bigger than the input's, plus nulls and
    /// an empty row.
    func testCaseMappingGrowsAWholeColumn() throws {
        try requireRealGPU()
        var rows: [String?] = []
        for i in 0..<3000 { rows.append(i % 13 == 0 ? nil : "stra\u{00DF}e-\(i)") }
        rows.append("")
        let a = try MetalStringArray(rows, context: .shared)
        let up = try a.utf8Upper().toArray()
        XCTAssertEqual(up.count, rows.count)
        for (i, r) in rows.enumerated() {
            if let r { XCTAssertEqual(up[i], r.replacingOccurrences(of: "stra\u{00DF}e", with: "STRA\u{1E9E}E").uppercased()) }
            else { XCTAssertNil(up[i]) }
        }
    }

    // MARK: - batching

    /// A batch whose body throws part way must leave no batch open, must not swallow the error, and
    /// must not leave a later op reading a GPU-side length the failed batch had queued.
    func testBatchThrowingLeavesNoStaleState() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        struct Boom: Error {}
        let xs: [String?] = ["aa", "bb", nil, "cc"]
        let a = try MetalStringArray(xs, context: ctx)
        XCTAssertThrowsError(try ctx.batch { () -> Int in
            _ = try a.utf8Upper()
            _ = try a.matchLike("%a%")
            throw Boom()
        }) { XCTAssertTrue($0 is Boom, "the body's error, not a Metal error") }
        XCTAssertFalse(ctx.isBatching)
        // The next batch sees a clean slate: the lengths it reads are its own.
        let after = try ctx.batch { try a.utf8Upper() }
        XCTAssertEqual(after.toArray(), ["AA", "BB", nil, "CC"])
        XCTAssertEqual(try a.matchLike("%a%").toArray(), [true, false, nil, false])
    }

    /// A result produced inside a batch and read after it: the read must be safe without the caller
    /// synchronising, and it must equal what the unbatched path produces.
    func testResultsProducedInABatchAreSafeToReadAfterIt() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let xs: [String?] = (0..<2000).map { $0 % 9 == 0 ? nil : "row-\($0 % 50)" }
        let a = try MetalStringArray(xs, context: ctx)
        var upper: MetalStringArray! = nil
        var lengths: MetalArray<Int32>! = nil
        var order: MetalArray<Int32>! = nil
        try ctx.batch {
            upper = try a.utf8Upper()
            lengths = try a.byteLength()
            order = try a.argsort()
        }
        XCTAssertEqual(upper.toArray(), xs.map { $0?.uppercased() })
        XCTAssertEqual(lengths.toArray(), xs.map { $0.map { s in Int32(s.utf8.count) } })
        XCTAssertEqual(order.toRawArray(), try a.argsort().toRawArray())
    }

    /// A slice taken of a batched result before the batch is committed still reads the committed data.
    func testSliceOfABatchedResultTakenBeforeTheCommit() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        let vals = (0..<4096).map { Int32(($0 &* 7919) % 3001) }
        let a = try MetalArray<Int32>(vals, context: ctx)
        var sliced: MetalArray<Int32>! = nil
        var full: MetalArray<Int32>! = nil
        try ctx.batch {
            full = try a.argsort()
            sliced = try full.slice(offset: 33, length: 100)   // sliced while the batch is still open
        }
        XCTAssertEqual(sliced.toRawArray(), Array(full.toRawArray()[33..<133]))
    }

    /// Two contexts on two threads, each with its own batch. `currentBatch` is thread-local, so
    /// neither may see the other's encoder, and the pooled buffers one parks must not be handed to the
    /// other while its GPU work is still in flight.
    func testTwoThreadsBatchingConcurrently() throws {
        try requireRealGPU()
        let a = MetalContext.shared
        let b = try MetalContext()
        let vals = (0..<20_000).map { Int64(($0 &* 2654435761) % 65_537) }
        let want = try MetalArray<Int64>(vals, context: a).argsort().toRawArray()
        let group = DispatchGroup()
        let lock = NSLock()
        var failures: [String] = []
        for (i, ctx) in [a, b, a, b].enumerated() {
            DispatchQueue.global().async(group: group) {
                do {
                    for _ in 0..<10 {
                        let arr = try MetalArray<Int64>(vals, context: ctx)
                        let got = try ctx.batch { try arr.argsort() }.toRawArray()
                        if got != want {
                            lock.lock(); failures.append("worker \(i) disagreed"); lock.unlock()
                            return
                        }
                    }
                } catch {
                    lock.lock(); failures.append("worker \(i): \(error)"); lock.unlock()
                }
            }
        }
        group.wait()
        XCTAssertEqual(failures, [])
        XCTAssertFalse(a.isBatching)
    }
}
