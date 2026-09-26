import XCTest
import CArrowABI
@testable import ArrowMetal

/// The `utf8_view` layout (`StringView.swift`): every kernel with a view form must give, on a view
/// column, exactly what it gives on the same strings as offsets + bytes, without converting the column;
/// every other kernel must convert once and still be right.
final class StringViewTests: XCTestCase {
    /// Inline (<= 12 bytes), out-of-line, empty, null, non-ASCII and long rows, several times over so
    /// the out-of-line rows spread across many small data buffers.
    static let base: [String?] = ["apple", "banana", nil, "", "apricot", "cherry pie with cream", "app",
                                  "grape🍇", "banana", "Apple", "exactly12byt", "thirteen byte",
                                  "héllo naïve é", "ÅNGSTRÖM", "Ωμέγα", String(repeating: "x", count: 300),
                                  "customer-1234", nil, "customer-1234", "ß and ſ"]
    static var sample: [String?] { (0..<23).flatMap { k in base.map { s in s.map { k % 2 == 0 ? $0 : $0 + String(k) } } } }

    private func pair() throws -> (view: MetalStringArray, offsets: MetalStringArray) {
        let s = Self.sample
        let v = try MetalStringArray.viewLayout(s, bufferBytes: 100)
        XCTAssertNotNil(v.view)
        XCTAssertGreaterThan(v.view!.dataBuffers.count, 5, "the sample should spread over several data buffers")
        return (v, try MetalStringArray(s))
    }

    private func assertNoConversion<R>(_ what: String, _ v: MetalStringArray, _ body: () throws -> R) rethrows -> R {
        let before = StringViewStorage.conversions.columns
        let r = try body()
        XCTAssertEqual(StringViewStorage.conversions.columns, before, "\(what) converted a view column")
        XCTAssertFalse(v.convertedFromView, "\(what) converted a view column")
        return r
    }

    func testRowsReadBackWithoutConversion() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        assertNoConversion("toArray", v) { XCTAssertEqual(v.toArray(), o.toArray()) }
        XCTAssertEqual(v.nullCount, o.nullCount)
    }

    func testLengthsHashesAndPredicates() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        try assertNoConversion("lengths / predicates", v) {
            XCTAssertEqual(try v.byteLength().toArray(), try o.byteLength().toArray())
            XCTAssertEqual(try v.charLength().toArray(), try o.charLength().toArray())
            XCTAssertEqual(try v.hash32().toArray(), try o.hash32().toArray())
            for p in ["", "a", "an", "customer-1", "thirteen byte", "x", "é", "Ω"] {
                XCTAssertEqual(try v.contains(p).toArray(), try o.contains(p).toArray(), "contains \(p)")
                XCTAssertEqual(try v.startsWith(p).toArray(), try o.startsWith(p).toArray(), "starts_with \(p)")
                XCTAssertEqual(try v.endsWith(p).toArray(), try o.endsWith(p).toArray(), "ends_with \(p)")
                XCTAssertEqual(try v.equals(p).toArray(), try o.equals(p).toArray(), "equals \(p)")
            }
            XCTAssertEqual(try v.equals(o).toArray(), try o.equals(o).toArray())
            XCTAssertEqual(try o.equals(v).toArray(), try o.equals(o).toArray())
            XCTAssertEqual(try v.equals(v).toArray(), try o.equals(o).toArray())
        }
    }

    func testSetLookupBothWays() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        let setStrings: [String?] = ["banana", String(repeating: "x", count: 300), nil, "Ωμέγα", "nope"]
        let setO = try MetalStringArray(setStrings)
        let setV = try MetalStringArray.viewLayout(setStrings, bufferBytes: 8)
        try assertNoConversion("is_in / index_in", v) {
            let want = try o.isIn(setO).toArray()
            XCTAssertEqual(try v.isIn(setO).toArray(), want)
            XCTAssertEqual(try v.isIn(setV).toArray(), want)
            XCTAssertEqual(try o.isIn(setV).toArray(), want)
            let wantIdx = try o.indexIn(setO).toArray()
            XCTAssertEqual(try v.indexIn(setO).toArray(), wantIdx)
            XCTAssertEqual(try v.indexIn(setV).toArray(), wantIdx)
        }
        XCTAssertFalse(setV.convertedFromView)
    }

    func testSortGroupUniqueAndGather() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        try assertNoConversion("sort / group / unique / gather", v) {
            for d in [false, true] {
                XCTAssertEqual(try v.argsort(descending: d).toArray(), try o.argsort(descending: d).toArray())
                XCTAssertEqual(try v.sorted(descending: d).toArray(), try o.sorted(descending: d).toArray())
            }
            let (cv, uv) = try v.dictionaryEncode()
            let (co, uo) = try o.dictionaryEncode()
            XCTAssertEqual(cv.toArray(), co.toArray())
            XCTAssertEqual(uv.toArray(), uo.toArray())
            let (sv, _) = try v.dictionaryEncodeSorted()
            let (so, _) = try o.dictionaryEncodeSorted()
            XCTAssertEqual(sv.toArray(), so.toArray())
            XCTAssertEqual(try v.unique().toArray(), try o.unique().toArray())
            let (gv, kv) = try GroupByKeys.denseIds(.string(v), v.context)
            let (go, ko) = try GroupByKeys.denseIds(.string(o), o.context)
            XCTAssertEqual(kv, ko)
            XCTAssertEqual(gv.toArray(), go.toArray())
            let mask = try v.startsWith("a")
            XCTAssertEqual(try v.filter(mask).toArray(), try o.filter(mask).toArray())
            let idx = try MetalArray<Int32>([5, 0, nil, 15, 19, 2])
            XCTAssertEqual(try v.take(idx).toArray(), try o.take(idx).toArray())
            let sl = try v.slice(offset: 7, length: 30)
            XCTAssertNotNil(sl.view)
            XCTAssertEqual(sl.toArray(), try o.slice(offset: 7, length: 30).toArray())
            XCTAssertEqual(try sl.contains("an").toArray(), try o.slice(offset: 7, length: 30).contains("an").toArray())
        }
    }

    func testCaseTransformsAndTrims() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        try assertNoConversion("case transforms", v) {
            XCTAssertEqual(try v.utf8Upper().toArray(), try o.utf8Upper().toArray())
            XCTAssertEqual(try v.utf8Lower().toArray(), try o.utf8Lower().toArray())
            XCTAssertEqual(try v.unicodeTitle().toArray(), try o.unicodeTitle().toArray())
            XCTAssertEqual(try v.unicodeSwapcase().toArray(), try o.unicodeSwapcase().toArray())
            XCTAssertEqual(try v.asciiUpper().toArray(), try o.asciiUpper().toArray())
            XCTAssertEqual(try v.trim().toArray(), try o.trim().toArray())
            XCTAssertEqual(try v.replaceSubstring("an", with: "AN", maxReplacements: -1).toArray(),
                           try o.replaceSubstring("an", with: "AN", maxReplacements: -1).toArray())
            XCTAssertEqual(try v.reverse().toArray(), try o.reverse().toArray())
            XCTAssertEqual(try v.countSubstring("a").toArray(), try o.countSubstring("a").toArray())
            XCTAssertEqual(try v.findSubstring("byte").toArray(), try o.findSubstring("byte").toArray())
            XCTAssertEqual(try v.classify(.alpha).toArray(), try o.classify(.alpha).toArray())
            XCTAssertEqual(try v.utf8IsAlpha().toArray(), try o.utf8IsAlpha().toArray())
            XCTAssertEqual(try v.asciiTitle().toArray(), try o.asciiTitle().toArray())
        }
    }

    func testKernelWithoutViewFormConvertsOnce() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        let before = StringViewStorage.conversions.columns
        // The byte-counting pads (StringBytes.swift) have no view form.
        XCTAssertEqual(try v.asciiLpad(width: 20, pad: "*").toArray(), try o.asciiLpad(width: 20, pad: "*").toArray())
        XCTAssertTrue(v.convertedFromView)
        XCTAssertEqual(StringViewStorage.conversions.columns, before + 1)
        // The converted form is kept: a second kernel without a view form converts nothing.
        XCTAssertEqual(try v.binaryReverse().toArray(), try o.binaryReverse().toArray())
        XCTAssertEqual(StringViewStorage.conversions.columns, before + 1)
        XCTAssertEqual(v.totalBytes, o.totalBytes)
    }

    func testCDataRoundTripKeepsTheViews() throws {
        try requireRealGPU()
        let (v, o) = try pair()
        var schema = ArrowSchema(), arr = ArrowArray()
        AnyMetalArray.string(v).exportArrowSchema(into: &schema)
        AnyMetalArray.string(v).exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.format), "vu")
        XCTAssertEqual(arr.n_buffers, Int64(3 + v.view!.dataBuffers.count))
        let back = try importArrowArray(schema: &schema, array: &arr)
        if let rel = schema.release { rel(&schema) }
        guard case .string(let w) = back.array else { return XCTFail("expected a string column") }
        XCTAssertNotNil(w.view)
        XCTAssertTrue(back.zeroCopy, "re-importing ArrowMetal's own page-aligned buffers should not copy")
        XCTAssertEqual(w.toArray(), o.toArray())
        XCTAssertEqual(try w.contains("an").toArray(), try o.contains("an").toArray())
    }

    func testMalformedViewsReadAsEmpty() throws {
        try requireRealGPU()
        // A view pointing past its buffer, and one naming a buffer that does not exist: both read as "".
        let v = try MetalStringArray.viewLayout(["thirteen byte", "fourteen bytes"], bufferBytes: 100)
        let p = v.view!.views.mutableContents
        p.storeBytes(of: Int32(1000), toByteOffset: 12, as: Int32.self)        // offset past the buffer
        p.storeBytes(of: Int32(7), toByteOffset: 16 + 8, as: Int32.self)       // buffer 7 of 1
        XCTAssertEqual(try v.byteLength().toArray(), [0, 0])
        XCTAssertEqual(try v.contains("byte").toArray(), [false, false])
        XCTAssertEqual(v.toArray(), ["", ""])
    }
}
