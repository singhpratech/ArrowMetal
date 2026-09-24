import XCTest
import Foundation
@testable import ArrowMetal

/// The GPU JSON reader (docs/JSON.md). The differential tests against pyarrow live in
/// python/tests/test_json.py; these check each stage against a CPU reference written here, and the
/// public API from Swift.
final class JSONReaderTests: XCTestCase {

    func read(_ text: String, _ options: JSONReadOptions = JSONReadOptions()) throws -> JSONTable {
        try JSONReader(string: text).read(options)
    }

    func readError(_ text: String, _ options: JSONReadOptions = JSONReadOptions()) -> String? {
        do { _ = try read(text, options); return nil } catch { return "\(error)" }
    }

    // MARK: - stage 1: structure

    /// Exclusive prefix maximum against a CPU loop, across the recursive group levels.
    func testMaxScanMatchesCPU() throws {
        try requireRealGPU()
        let ctx = MetalContext.shared
        var rng = SystemRandomNumberGenerator()
        for n in [1, 255, 256, 257, 70_000, (1 << 18) + 3] {
            let values: [UInt32] = (0..<n).map { _ in UInt64.random(in: 0..<4, using: &rng) == 0 ? UInt32.random(in: 0..<1_000_000, using: &rng) : 0 }
            let buf = try MetalArrowBuffer.allocate(byteCount: n * 4, context: ctx)
            let p = buf.mutableTyped(UInt32.self)
            for i in 0..<n { p[i] = values[i] }
            let out = try JSONKernels.maxScan(ctx, buf, n).typed(UInt32.self)
            var run: UInt32 = 0
            for i in 0..<n {
                XCTAssertEqual(out[i], run, "n=\(n) i=\(i)")
                if out[i] != run { break }
                run = max(run, values[i])
            }
        }
    }

    /// Record boundaries a sequential scanner finds (strings, escapes and depth tracked byte by byte).
    func cpuRecords(_ b: [UInt8]) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var d = 0, ins = false, esc = false, start = -1
        var i = 0
        while i < b.count {
            let c = b[i]
            defer { i += 1 }
            if esc { esc = false; continue }
            if c == 0x5C { esc = true; continue }
            if c == 0x22 { ins.toggle(); continue }
            if ins { continue }
            if d == 0, c == 0x6E, i + 4 <= b.count, Array(b[i..<i + 4]) == Array("null".utf8) { out.append((i, i + 4)); i += 3; continue }
            if c == 0x7B || c == 0x5B { if d == 0 { start = i }; d += 1 }
            if c == 0x7D || c == 0x5D { d -= 1; if d == 0 { out.append((start, i + 1)) } }
        }
        return out
    }

    func testRecordBoundariesMatchCPUScanner() throws {
        try requireRealGPU()
        var rng = SystemRandomNumberGenerator()
        let pieces = ["{\"a\":1}", "{\"s\":\"{[}]\"}", "{\"e\":\"\\\"}\"}", "{\"b\":\"\\\\\"}", "null",
                      "{\"n\":{\"x\":[1,{\"y\":\"]\"}]}}", "{\"k\":\"" + String(repeating: "\\\\", count: 300) + "\"}",
                      "{\"m\":\n1}", "{}", "{\"q\":\"" + String(repeating: "x", count: 250) + "\\\"\"}"]
        for _ in 0..<20 {
            var text = ""
            for _ in 0..<Int.random(in: 1..<200, using: &rng) {
                text += pieces.randomElement(using: &rng)!
                text += ["\n", " ", "", "\r\n", "\t"].randomElement(using: &rng)!
            }
            let bytes = Array(text.utf8)
            let src = try MetalArrowBuffer.allocate(byteCount: bytes.count + 64, zeroed: true)
            for (i, c) in bytes.enumerated() { src.mutableTyped(UInt8.self)[i] = c }
            let recs = try JSONKernels.records(MetalContext.shared, src, n: bytes.count, start: 0)
            XCTAssertNil(recs.topError.map { "\($0)" })
            let want = cpuRecords(bytes)
            XCTAssertEqual(recs.count, want.count)
            let s = recs.start.typed(UInt32.self), e = recs.end.typed(UInt32.self)
            for (k, w) in want.enumerated() where k < recs.count {
                XCTAssertEqual(Int(s[k]), w.0, "start \(k)")
                XCTAssertEqual(Int(e[k]), w.1, "end \(k)")
            }
        }
    }

    func testTopLevelErrorsCarryPositionAndCode() throws {
        try requireRealGPU()
        let cases: [(String, Int, UInt32)] = [
            ("{\"a\":1}\n x", 9, 1),        // Invalid value.
            ("{\"a\":1}}", 7, 15),           // The document is empty.
            ("{} [1]", 3, 16),               // changed from object to array
            ("{}\n\"s\"", 3, 17),            // changed from object to string
            ("{}\n-12", 3, 18),              // changed from object to number
            ("{}\ntrue", 3, 19),             // changed from object to boolean
            ("{}\n\"s", 3, 10),              // the string's own error comes first
        ]
        for (text, pos, code) in cases {
            let bytes = Array(text.utf8)
            let src = try MetalArrowBuffer.allocate(byteCount: bytes.count + 64, zeroed: true)
            for (i, c) in bytes.enumerated() { src.mutableTyped(UInt8.self)[i] = c }
            let recs = try JSONKernels.records(MetalContext.shared, src, n: bytes.count, start: 0)
            XCTAssertEqual(recs.topError?.position, pos, text)
            XCTAssertEqual(recs.topError?.code, code, text)
        }
    }

    // MARK: - stage 2: walk

    func testWalkEmitsKeySpansValueSpansAndKinds() throws {
        try requireRealGPU()
        let text = "{\"i\":-12,\"f\":1.5e3,\"s\":\"a\\nb\",\"k\\u0041\":true,\"n\":null,\"o\":{\"x\":[1]},\"l\":[1,2],\"z\":NaN,\"b\":false,\"big\":99999999999999999999}"
        let bytes = Array(text.utf8)
        let src = try MetalArrowBuffer.allocate(byteCount: bytes.count + 64, zeroed: true)
        for (i, c) in bytes.enumerated() { src.mutableTyped(UInt8.self)[i] = c }
        let ctx = MetalContext.shared
        let recs = try JSONKernels.records(ctx, src, n: bytes.count, start: 0)
        XCTAssertEqual(recs.count, 1)
        let (counts, err) = try JSONKernels.walkCount(ctx, src, n: bytes.count, spanStart: recs.start, spanEnd: recs.end, spans: 1)
        XCTAssertNil(err.map { "\($0)" })
        let level = try JSONKernels.walkEmit(ctx, src, n: bytes.count, spanStart: recs.start, spanEnd: recs.end, spans: 1, counts: counts)
        XCTAssertEqual(level.count, 10)
        func str(_ start: UInt32, _ len: UInt32) -> String { String(decoding: bytes[Int(start)..<Int(start + len)], as: UTF8.self) }
        let want: [(String, String, JSONKind, UInt32)] = [
            ("i", "-12", .int, 0), ("f", "1.5e3", .float, 0), ("s", "\"a\\nb\"", .string, JSONKind.escapeFlag),
            ("k\\u0041", "true", .trueValue, JSONKind.keyEscapeFlag), ("n", "null", .null, 0),
            ("o", "{\"x\":[1]}", .object, 0), ("l", "[1,2]", .array, 0), ("z", "NaN", .float, JSONKind.specialFlag),
            ("b", "false", .falseValue, 0), ("big", "99999999999999999999", .float, 0),
        ]
        for (k, w) in want.enumerated() {
            let e = level.entry(k)
            XCTAssertEqual(e.parent, 0)
            XCTAssertEqual(str(e.keyStart, e.keyLen), w.0)
            XCTAssertEqual(str(e.valStart, e.valLen), w.1)
            XCTAssertEqual(e.kind, w.2, w.0)
            XCTAssertEqual(e.flags & 0xF0, w.3, w.0)
        }
    }

    func testSyntaxErrorsCarryRapidJSONTexts() throws {
        try requireRealGPU()
        let cases: [(String, String)] = [
            ("{\"a\":1\n", "JSON parse error: Missing a comma or '}' after an object member. in row 0"),
            ("{\"a\":\"x\n", "JSON parse error: Invalid encoding in string. in row 0"),
            ("{\"a\":\"\\q\"}\n", "JSON parse error: Invalid escape character in string. in row 0"),
            ("{\"a\":\"\\ud83d\"}\n", "JSON parse error: The surrogate pair in string is invalid. in row 0"),
            ("{\"a\":\"\\u00G9\"}\n", "JSON parse error: Incorrect hex digit after \\u escape in string. in row 0"),
            ("{\"a\":1,}\n", "JSON parse error: Missing a name for object member. in row 0"),
            ("{\"a\" 1}\n", "JSON parse error: Missing a colon after a name of object member. in row 0"),
            ("{\"a\":[1 2]}\n", "JSON parse error: Missing a comma or ']' after an array element. in row 0"),
            ("{\"a\":1.}\n", "JSON parse error: Miss fraction part in number. in row 0"),
            ("{\"a\":1e}\n", "JSON parse error: Miss exponent in number. in row 0"),
            ("{\"a\":1e309}\n", "JSON parse error: Number too big to be stored in double. in row 0"),
            ("{\"a\":tru}\n", "JSON parse error: Invalid value. in row 0"),
            ("{\"a\":\"x", "JSON parse error: Missing a closing quotation mark in string. in row 0"),
            ("{\"a\":1} x\n", "JSON parse error: Invalid value. in row 1"),
            ("{\"a\":1}}\n", "JSON parse error: The document is empty."),
            ("[1]\n", "JSON parse error: Column() changed from object to array in row 0"),
            ("", "Empty JSON file"),
        ]
        for (text, message) in cases { XCTAssertEqual(readError(text), message, text) }
    }

    func testConflictsAndRepeatedKeysCarryPyarrowTexts() throws {
        try requireRealGPU()
        XCTAssertEqual(readError("{\"a\":1}\n{\"a\":\"x\"}\n"), "JSON parse error: Column(/a) changed from number to string in row 1")
        XCTAssertEqual(readError("{\"a\":1,\"a\":2}\n"), "JSON parse error: Column(/a) was specified twice in row 0")
        XCTAssertEqual(readError("{\"s\":{\"x\":1}}\n{\"s\":{\"x\":\"a\"}}\n"), "JSON parse error: Column(/s/x) changed from number to string in row 1")
        XCTAssertEqual(readError("{\"l\":[1]}\n{\"l\":[\"a\"]}\n"), "JSON parse error: Column(/l/[]) changed from number to string in row 1")
        // A conflict earlier in the file than a syntax error is reported first, even in the same record.
        XCTAssertEqual(readError("{\"a\":1}\n{\"a\":\"x\",\"b\":}\n"), "JSON parse error: Column(/a) changed from number to string in row 1")
        XCTAssertEqual(readError("{\"a\":1}\n{\"a\":}\n{\"a\":\"x\"}\n"), "JSON parse error: Invalid value. in row 1")
    }

    func testNestingLimit() throws {
        try requireRealGPU()
        let deep = "{\"a\":" + String(repeating: "[", count: 1100) + String(repeating: "]", count: 1100) + "}\n"
        XCTAssertEqual(readError(deep), "JSON parse error: Nesting deeper than 1024 levels is not supported. in row 0")
        let ok = "{\"a\":" + String(repeating: "[", count: 200) + String(repeating: "]", count: 200) + "}\n"
        XCTAssertNil(readError(ok))
    }

    // MARK: - stage 3: columns

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

    func testMissingKeysFieldOrderAndInference() throws {
        try requireRealGPU()
        let t = try read("{\"a\":1}\n{\"b\":2.5,\"a\":null}\n\n{\"c\":\"q\",\"a\":3}\r\n{\"b\":7}")
        XCTAssertEqual(t.names, ["a", "b", "c"])
        XCTAssertEqual(t.rowCount, 4)
        XCTAssertEqual(t["a"]?.asInt64?.toArray(), [1, nil, 3, nil])
        XCTAssertEqual(t["b"]?.asFloat64?.toArray(), [nil, 2.5, nil, 7])
        XCTAssertEqual(t["c"]?.asString?.toArray(), [nil, nil, "q", nil])
    }

    func testNumbers() throws {
        try requireRealGPU()
        let t = try read("{\"i\":9223372036854775807,\"j\":-9223372036854775808,\"o\":9223372036854775808,\"f\":NaN,\"g\":-Infinity}\n")
        XCTAssertEqual(t["i"]?.asInt64?.toArray(), [Int64.max])
        XCTAssertEqual(t["j"]?.asInt64?.toArray(), [Int64.min])
        XCTAssertEqual(t["o"]?.asFloat64?.toArray(), [9223372036854775808.0])
        XCTAssertTrue(t["f"]?.asFloat64?.toArray()[0]?.isNaN ?? false)
        XCTAssertEqual(t["g"]?.asFloat64?.toArray(), [-Double.infinity])
    }

    func testStringsUnescape() throws {
        try requireRealGPU()
        let t = try read("{\"s\":\"\\u00e9\\ud83d\\ude00\\n\\t\\\"\\\\\\/\\b\\f\\r\"}\n{\"s\":\"plain\"}\n{\"s\":\"\\u0000\"}\n")
        XCTAssertEqual(t["s"]?.asString?.toArray(), ["\u{e9}\u{1F600}\n\t\"\\/\u{8}\u{c}\r", "plain", "\u{0}"])
    }

    /// Days since 1970-01-01 for a proleptic Gregorian date, by counting (the kernel uses a closed form).
    func days(_ y: Int, _ m: Int, _ d: Int) -> Int64 {
        func leap(_ y: Int) -> Bool { (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 }
        let md = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        var n: Int64 = 0
        if y >= 1970 { for yy in 1970..<y { n += leap(yy) ? 366 : 365 } } else { for yy in y..<1970 { n -= leap(yy) ? 366 : 365 } }
        for mm in 1..<m { n += Int64(md[mm - 1] + (mm == 2 && leap(y) ? 1 : 0)) }
        return n + Int64(d - 1)
    }

    func testTimestampInferenceAndISO8601Forms() throws {
        try requireRealGPU()
        let rows: [(String, Int64)] = [
            ("2020-01-01", days(2020, 1, 1) * 86400),
            ("2020-02-29 13", days(2020, 2, 29) * 86400 + 13 * 3600),
            ("1969-12-31T23:59", -60),
            ("1900-03-01T00:00:01", days(1900, 3, 1) * 86400 + 1),
            ("0000-01-01", days(0, 1, 1) * 86400),
            ("9999-12-31 23:59:59", days(9999, 12, 31) * 86400 + 86399),
            ("2020-01-01T00:00:00Z", days(2020, 1, 1) * 86400),
            ("2020-01-01T10:00+02:00", days(2020, 1, 1) * 86400 + 8 * 3600),
            ("2020-01-01T10:00:00-0530", days(2020, 1, 1) * 86400 + 15 * 3600 + 1800),
            ("2020-01-01T10-05", days(2020, 1, 1) * 86400 + 15 * 3600),
        ]
        let text = rows.map { "{\"t\":\"\($0.0)\"}\n" }.joined() + "{\"t\":null}\n"
        let t = try read(text)
        guard case .temporal(let ts)? = t["t"], case .int64(let v) = ts.storage else { return XCTFail("t should be a timestamp") }
        XCTAssertEqual(ts.type, .timestamp(.second, timezone: nil))
        XCTAssertEqual(v.toArray(), rows.map { $0.1 } + [nil])
        // Any value that is not a timestamp keeps the column utf8.
        for bad in ["2020-02-30", "2019-02-29", "2020-01-01 24:00:00", "2020-01-01 00:00:60", "2020-01-01T0000",
                    "2020-01-01Z", "2020-01-01 00:00:00.5", "2020-1-01", " 2020-01-01"] {
            let u = try read("{\"t\":\"2020-01-01\"}\n{\"t\":\"\(bad)\"}\n")
            XCTAssertNotNil(u["t"]?.asString, bad)
        }
    }

    func testExplicitSchema() throws {
        try requireRealGPU()
        let text = "{\"a\":\"x\",\"b\":1,\"c\":true,\"t\":\"2020-01-01 00:00:00.123\"}\n{\"a\":\"y\",\"c\":false,\"d\":1}\n"
        let schema = [JSONField("t", .timestamp(.milli, timezone: "UTC")), JSONField("b", .int8), JSONField("a", .utf8)]
        let infer = try read(text, JSONReadOptions(explicitSchema: schema))
        XCTAssertEqual(infer.names, ["t", "b", "a", "c", "d"])
        XCTAssertEqual(infer["b"]?.length, 2)
        if case .int8(let b)? = infer["b"] { XCTAssertEqual(b.toArray(), [1, nil]) } else { XCTFail("b should be int8") }
        guard case .temporal(let ts)? = infer["t"], case .int64(let tv) = ts.storage else { return XCTFail("t") }
        XCTAssertEqual(ts.type, .timestamp(.milli, timezone: "UTC"))
        XCTAssertEqual(tv.toArray(), [days(2020, 1, 1) * 86_400_000 + 123, nil])
        let ignore = try read(text, JSONReadOptions(explicitSchema: schema, unexpectedFieldBehavior: .ignore))
        XCTAssertEqual(ignore.names, ["t", "b", "a"])
        XCTAssertEqual(readError(text, JSONReadOptions(explicitSchema: schema, unexpectedFieldBehavior: .error)),
                       "JSON parse error: unexpected field")
        XCTAssertEqual(readError("{\"b\":1.5}\n", JSONReadOptions(explicitSchema: [JSONField("b", .int32)])),
                       "Failed to convert JSON to int32, couldn't parse:1.5")
        XCTAssertEqual(readError("{\"b\":1}\n", JSONReadOptions(explicitSchema: [JSONField("b", .utf8)])),
                       "JSON parse error: Column(/b) changed from string to number in row 0")
        XCTAssertEqual(readError("{\"b\":\"x\"}\n", JSONReadOptions(explicitSchema: [JSONField("b", .timestamp(.second, timezone: nil))])),
                       "Failed to convert JSON to timestamp[s], couldn't parse:x")
    }

    func testNestedStructAndList() throws {
        try requireRealGPU()
        let t = try read("{\"s\":{\"x\":1,\"l\":[1,2]},\"l\":[{\"y\":\"a\"},null,{\"y\":null}]}\n{\"s\":null,\"l\":[]}\n{\"l\":null}\n")
        XCTAssertEqual(t.names, ["s", "l"])
        guard case .structure(let s)? = t["s"] else { return XCTFail("s should be a struct") }
        XCTAssertEqual(s.names, ["x", "l"])
        XCTAssertEqual(s.nullCount, 2)
        XCTAssertEqual(s.children[0].asInt64?.toArray()[0], 1)
        guard case .list(let inner) = s.children[1] else { return XCTFail("s.l should be a list") }
        XCTAssertEqual(inner.values.asInt64?.toArray(), [1, 2])
        guard case .list(let l)? = t["l"] else { return XCTFail("l should be a list") }
        XCTAssertEqual(l.length, 3)
        XCTAssertEqual(l.nullCount, 1)
        let off = l.offsets.typed(Int32.self)
        XCTAssertEqual([off[0], off[1], off[2], off[3]], [0, 3, 3, 3])
        guard case .structure(let e) = l.values else { return XCTFail("l's elements should be structs") }
        XCTAssertEqual(e.length, 3)
        XCTAssertEqual(e.nullCount, 1)
        XCTAssertEqual(e.children[0].asString?.toArray(), ["a", nil, nil])
    }

    func testRowsWithoutColumnsAndNullRows() throws {
        try requireRealGPU()
        XCTAssertEqual(try read("{}\n{}\n{ }\n").rowCount, 3)
        let t = try read("{\"a\":1}\nnull\n{\"a\":2}\n")
        XCTAssertEqual(t["a"]?.asInt64?.toArray(), [1, nil, 2])
    }

    /// Random flat files against Foundation's JSON parser, value by value.
    func testRandomFlatFilesAgainstFoundation() throws {
        try requireRealGPU()
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<10 {
            let nrows = Int.random(in: 1..<3000, using: &rng)
            var lines: [String] = []
            for r in 0..<nrows {
                var parts: [String] = []
                if Bool.random(using: &rng) { parts.append("\"i\":\(Int64.random(in: -(1 << 53)...(1 << 53), using: &rng))") }
                if Bool.random(using: &rng) { parts.append("\"d\":\(Double.random(in: -1e6...1e6, using: &rng))") }
                if Bool.random(using: &rng) { parts.append("\"b\":\(Bool.random(using: &rng))") }
                if Bool.random(using: &rng) { parts.append("\"s\":\"v\\u00e9\\n\(r)\"") }
                if Int.random(in: 0..<10, using: &rng) == 0, !parts.contains(where: { $0.hasPrefix("\"i\"") }) {
                    parts.append("\"i\":null")
                }
                parts.shuffle(using: &rng)
                lines.append("{" + parts.joined(separator: ",") + "}")
            }
            let text = lines.joined(separator: "\n") + "\n"
            let t = try read(text)
            let objs = try lines.map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
            XCTAssertEqual(t.rowCount, nrows)
            if let col = t["i"]?.asInt64?.toArray() {
                XCTAssertEqual(col, objs.map { ($0["i"] as? NSNumber)?.int64Value })
            }
            if let col = t["d"]?.asFloat64?.toArray() {
                XCTAssertEqual(col, objs.map { ($0["d"] as? NSNumber)?.doubleValue })
            }
            if let col = t["b"]?.asBoolean?.toArray() {
                XCTAssertEqual(col, objs.map { ($0["b"] as? NSNumber)?.boolValue })
            }
            if let col = t["s"]?.asString?.toArray() {
                XCTAssertEqual(col, objs.map { $0["s"] as? String })
            }
        }
    }

    func testWideFileBuildsEveryColumn() throws {
        try requireRealGPU()
        let fields = 400
        var text = ""
        for r in 0..<50 {
            text += "{" + (0..<fields).map { f in f % 3 == 0 ? "\"f\(f)\":\(r * f)" : (f % 3 == 1 ? "\"f\(f)\":\"s\(r)\"" : "\"f\(f)\":\(r % 2 == 0)") }
                .joined(separator: ",") + "}\n"
        }
        let t = try read(text)
        XCTAssertEqual(t.names.count, fields)
        for f in stride(from: 0, to: fields, by: 37) {
            switch f % 3 {
            case 0: XCTAssertEqual(t["f\(f)"]?.asInt64?.toArray(), (0..<50).map { Int64($0 * f) })
            case 1: XCTAssertEqual(t["f\(f)"]?.asString?.toArray(), (0..<50).map { "s\($0)" })
            default: XCTAssertEqual(t["f\(f)"]?.asBoolean?.toArray(), (0..<50).map { $0 % 2 == 0 })
            }
        }
    }

    /// Tables whose slot matrix passes the budget are built a group of fields at a time; with the
    /// budget forced down to a few rows' worth, every field is its own group.
    func testSlotMatrixGroups() throws {
        try requireRealGPU()
        let saved = JSONColumnBuilder.matrixBudgetBytes
        defer { JSONColumnBuilder.matrixBudgetBytes = saved }
        var text = ""
        for r in 0..<100 { text += "{\"a\":\(r),\"b\":\"s\(r)\",\"a\(r % 3)\":true,\"c\":[\(r)]}\n" }
        let whole = try read(text)
        JSONColumnBuilder.matrixBudgetBytes = 100 * 4
        let grouped = try read(text)
        XCTAssertEqual(grouped.names, whole.names)
        XCTAssertEqual(grouped.names, ["a", "b", "a0", "c", "a1", "a2"])
        XCTAssertEqual(grouped["a"]?.asInt64?.toArray(), (0..<100).map { Int64($0) })
        XCTAssertEqual(grouped["b"]?.asString?.toArray(), (0..<100).map { "s\($0)" })
        XCTAssertEqual(grouped["a1"]?.asBoolean?.toArray(), (0..<100).map { $0 % 3 == 1 ? true : nil })
        XCTAssertEqual(readError("{\"a\":1,\"b\":2,\"a\":3}\n"), "JSON parse error: Column(/a) was specified twice in row 0")
    }

    func testInputsOf4GiBAreRejected() throws {
        XCTAssertNoThrow(try JSONReader.checkSize(Int(UInt32.max) - 4097))
        XCTAssertThrowsError(try JSONReader.checkSize(Int(UInt32.max))) {
            XCTAssertTrue("\($0)".contains("4 GiB or more are not supported"))
        }
    }

    func testReadsFilesAndRejectsEmptyOnes() throws {
        try requireRealGPU()
        let dir = FileManager.default.temporaryDirectory
        let path = dir.appendingPathComponent("am_json_\(UUID().uuidString).jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        try Data("{\"a\":1}\n{\"a\":2}\n".utf8).write(to: URL(fileURLWithPath: path))
        let t = try JSONReader.read(path: path)
        XCTAssertEqual(t["a"]?.asInt64?.toArray(), [1, 2])
        XCTAssertEqual(try t.recordBatch().length, 2)
        let empty = dir.appendingPathComponent("am_json_\(UUID().uuidString).jsonl").path
        defer { try? FileManager.default.removeItem(atPath: empty) }
        try Data().write(to: URL(fileURLWithPath: empty))
        XCTAssertThrowsError(try JSONReader.read(path: empty)) { XCTAssertEqual("\($0)", "Empty JSON file") }
        XCTAssertThrowsError(try JSONReader(path: dir.appendingPathComponent("does-not-exist.jsonl").path))
    }
}
