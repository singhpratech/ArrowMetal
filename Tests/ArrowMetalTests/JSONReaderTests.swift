import XCTest
@testable import ArrowMetal

/// The GPU JSON reader (docs/JSON.md). The differential tests against pyarrow live in
/// python/tests/test_json.py; these check the reader's stages and its public API from Swift.
final class JSONReaderTests: XCTestCase {

    func read(_ text: String, _ options: JSONReadOptions = JSONReadOptions()) throws -> JSONTable {
        try JSONReader(string: text).read(options)
    }

    func testFlatRecord() throws {
        try requireRealGPU()
        let t = try read("{\"a\":1,\"b\":\"x\",\"c\":true,\"d\":null,\"e\":1.5}\n")
        XCTAssertEqual(t.names, ["a", "b", "c", "d", "e"])
        XCTAssertEqual(t.rowCount, 1)
        XCTAssertEqual(t["a"]?.asInt64?.toArray(), [1])
        XCTAssertEqual(t["b"]?.asString?.toArray(), ["x"])
        XCTAssertEqual(t["c"]?.asBoolean?.toArray(), [true])
        XCTAssertEqual(t["e"]?.asFloat64?.toArray(), [1.5])
        if case .null(let n)? = t["d"] { XCTAssertEqual(n.length, 1) } else { XCTFail("d should be null-typed") }
    }
}
