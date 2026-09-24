import XCTest
@testable import ArrowMetal

/// The GPU CSV reader: the structure scan against an independent CPU parser at many block sizes,
/// inference, conversions and errors. `python/tests/test_csv.py` checks the same reader against
/// `pyarrow.csv.read_csv` itself.
final class CSVReaderTests: XCTestCase {

    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("am-csv-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ bytes: [UInt8], _ name: String = "t.csv") throws -> String {
        let url = dir.appendingPathComponent(name)
        try Data(bytes).write(to: url)
        return url.path
    }

    private func write(_ text: String, _ name: String = "t.csv") throws -> String { try write(Array(text.utf8), name) }

    private func read(_ text: String, _ configure: (inout CSVReadOptions) -> Void = { _ in }) throws -> MetalRecordBatch {
        var o = CSVReadOptions()
        configure(&o)
        return try CSVReader.read(path: try write(text), options: o)
    }

    private func strings(_ a: AnyMetalArray) -> [String?] {
        guard case .string(let s) = a else { XCTFail("not a string column: \(a)"); return [] }
        return s.toArray()
    }

    // MARK: - reference parser

    /// An independent, byte-at-a-time RFC 4180 parser with pyarrow's rules: quotes open only at a
    /// field start, doubled quotes inside quotes, text after a closing quote is literal, `\r`, `\n` and
    /// `\r\n` end records, empty lines are skipped, a BOM is dropped.
    static func referenceParse(_ bytes: [UInt8], delimiter: UInt8 = 0x2C, quote: UInt8? = 0x22) -> [[[UInt8]]] {
        var b = bytes[...]
        if b.starts(with: [0xEF, 0xBB, 0xBF]) { b = b.dropFirst(3) }
        var records: [[[UInt8]]] = []
        var fields: [[UInt8]] = []
        var cur: [UInt8] = []
        var i = b.startIndex
        var atLineStart = true
        while i < b.endIndex {
            let c = b[i]
            if atLineStart && (c == 0x0A || c == 0x0D) { i += 1; continue }
            atLineStart = false
            if let q = quote, c == q, cur.isEmpty {
                // Quoted section.
                i += 1
                while i < b.endIndex {
                    if b[i] == q {
                        if i + 1 < b.endIndex && b[i + 1] == q { cur.append(q); i += 2; continue }
                        i += 1
                        break
                    }
                    cur.append(b[i]); i += 1
                }
                // Literal remainder up to the delimiter / newline.
                while i < b.endIndex, b[i] != delimiter, b[i] != 0x0A, b[i] != 0x0D { cur.append(b[i]); i += 1 }
                continue
            }
            if c == delimiter { fields.append(cur); cur = []; i += 1; continue }
            if c == 0x0A || c == 0x0D {
                fields.append(cur); cur = []
                records.append(fields); fields = []
                atLineStart = true
                i += 1
                continue
            }
            cur.append(c); i += 1
        }
        if !atLineStart { fields.append(cur); records.append(fields) }
        return records
    }

    /// Random CSV bytes with every structural hazard: quoted fields with delimiters, doubled quotes,
    /// `\n`, `\r\n` and `\r` inside them, text after a closing quote, quotes mid-field, empty fields,
    /// empty lines, mixed line endings, optional BOM and final newline.
    static func randomCSV(_ rng: inout CSVFloatParseTests.SplitMix64, rows: Int, cols: Int) -> [UInt8] {
        var out: [UInt8] = []
        if rng.int(4) == 0 { out += [0xEF, 0xBB, 0xBF] }
        let plain = Array("abcxyz0123456789 .-".utf8)
        func field() -> [UInt8] {
            switch rng.int(9) {
            case 0: return []
            case 1: return [0x22, 0x22]                                          // ""
            case 2:
                var f: [UInt8] = [0x22]
                for _ in 0..<rng.int(12) {
                    switch rng.int(8) {
                    case 0: f += [0x22, 0x22]
                    case 1: f.append(0x2C)
                    case 2: f.append(0x0A)
                    case 3: f += [0x0D, 0x0A]
                    case 4: f.append(0x0D)
                    default: f.append(plain[rng.int(plain.count)])
                    }
                }
                f.append(0x22)
                if rng.int(6) == 0 { f += Array("tail".utf8) }                   // "ab"tail
                return f
            case 3: return Array("ab\"c".utf8)                                   // quote mid-field
            default: return (0..<(1 + rng.int(8))).map { _ in plain[rng.int(plain.count)] }
            }
        }
        for r in 0..<rows {
            if rng.int(10) == 0 { out += rng.int(2) == 0 ? [0x0A] : [0x0D, 0x0A] }   // empty line
            var row: [UInt8] = []
            for c in 0..<cols {
                if c > 0 { row.append(0x2C) }
                var f = field()
                // A record whose only field is empty would be an empty line; keep it non-empty.
                if cols == 1 && f.isEmpty { f = [0x78] }
                row += f
            }
            out += row
            if r < rows - 1 || rng.int(2) == 0 {
                switch rng.int(3) {
                case 0: out.append(0x0A)
                case 1: out += [0x0D, 0x0A]
                default: out.append(0x0D)
                }
            }
        }
        return out
    }

    // MARK: - structure

    func testScanMatchesReferenceParserAtManyBlockSizes() throws {
        var rng = CSVFloatParseTests.SplitMix64(state: 0xC5_0001)
        for trial in 0..<24 {
            let cols = 1 + rng.int(6)
            let rows = 1 + rng.int(trial < 12 ? 30 : 600)
            let bytes = Self.randomCSV(&rng, rows: rows, cols: cols)
            let expected = Self.referenceParse(bytes)
            let path = try write(bytes, "r\(trial).csv")
            let runs: [(Int, CSVReadOptions.FileAccess)] = [1, 2, 3, 5, 16, 64, 257, 1024].map { ($0, .read) }
                + [(3, .map), (1024, .map)]
            for (block, access) in runs {
                var o = CSVReadOptions()
                o.autogenerateColumnNames = true
                o.scanBlockBytes = block
                o.fileAccess = access
                o.columnTypes = Dictionary(uniqueKeysWithValues: (0..<cols).map { ("f\($0)", CSVColumnType.binary) })
                let batch = try CSVReader.read(path: path, options: o)
                XCTAssertEqual(batch.columnCount, cols, "trial \(trial) block \(block)")
                XCTAssertEqual(batch.length, expected.count, "trial \(trial) block \(block)")
                for c in 0..<cols {
                    guard case .binary(let s) = batch.columns[c] else { XCTFail("column \(c) not binary"); continue }
                    let off = s.offsets.typed(Int32.self), d = s.data.typed(UInt8.self)
                    for r in 0..<expected.count {
                        let got = Array(UnsafeBufferPointer(start: d + Int(off[r]), count: Int(off[r + 1] - off[r])))
                        if got != expected[r][c] {
                            XCTFail("trial \(trial) block \(block) row \(r) col \(c): \(String(decoding: got, as: UTF8.self)) != \(String(decoding: expected[r][c], as: UTF8.self))")
                            return
                        }
                    }
                }
            }
        }
    }

    func testQuotedNewlinesCRLFAndDoubledQuotes() throws {
        let b = try read("a,b\r\n\"x\r\ny\",\"p\"\"q\"\r\n\"ab\"cd,e\"f\r\n")
        XCTAssertEqual(strings(b.columns[0]), ["x\r\ny", "abcd"])
        XCTAssertEqual(strings(b.columns[1]), ["p\"q", "e\"f"])
    }

    func testMissingFinalNewlineBOMAndEmptyLines() throws {
        let path = try write([0xEF, 0xBB, 0xBF] + Array("a,b\n\n1,2\n\r\n3,4".utf8))
        let b = try CSVReader.read(path: path)
        XCTAssertEqual(b.names, ["a", "b"])
        XCTAssertEqual(b.columns[0].asInt64?.toArray(), [1, 3])
        XCTAssertEqual(b.columns[1].asInt64?.toArray(), [2, 4])
    }

    func testRaggedRowNamesTheRow() throws {
        XCTAssertThrowsError(try read("a,b,c\n\n1,2,3\n4,5\n")) { e in
            XCTAssertEqual("\(e)", "CSV parse error: Row #3: Expected 3 columns, got 2: 4,5")
        }
        XCTAssertThrowsError(try read("x\na,b\n1,2\n3,4,5\n") { $0.skipRows = 1 }) { e in
            XCTAssertEqual("\(e)", "CSV parse error: Row #4: Expected 2 columns, got 3: 3,4,5")
        }
    }

    func testEmptyInputs() throws {
        XCTAssertThrowsError(try read("")) { XCTAssertEqual("\($0)", "Empty CSV file") }
        XCTAssertThrowsError(try read("a,b")) {
            XCTAssertEqual("\($0)", "CSV parse error: Empty CSV file or block: cannot infer number of columns")
        }
        let b = try read("a,b\n")
        XCTAssertEqual(b.names, ["a", "b"])
        if case .null(let n) = b.columns[0] { XCTAssertEqual(n.length, 0) } else { XCTFail("expected null") }
    }

    // MARK: - inference and conversion

    func testInferenceOrder() throws {
        let b = try read("""
        i,f,b,d,t,ts,tsn,tsz,s,n,bi
        1,1.5,true,2020-01-01,12:34:56,2020-01-01 12:34:56,2020-01-01 12:34:56.5,2020-01-01T00:00:00Z,x,,1
        -2,2,False,2021-02-28,01:02,2020-01-02T01:02,2020-01-02,2020-01-01 00:00:00+01:00,NA,NA,true

        """)
        let formats = b.columns.map { $0.arrowFormat }
        XCTAssertEqual(formats, ["l", "g", "b", "tdD", "tts", "tss:", "tsn:", "tss:UTC", "u", "n", "b"])
        XCTAssertEqual(b.columns[0].asInt64?.toArray(), [1, -2])
        XCTAssertEqual(b.columns[1].asFloat64?.toArray(), [1.5, 2])
        XCTAssertEqual(strings(b.columns[8]), ["x", "NA"])
        guard case .temporal(let tsz) = b.columns[7], case .int64(let v) = tsz.storage else { return XCTFail() }
        XCTAssertEqual(v.toArray(), [1_577_836_800, 1_577_833_200])
    }

    func testForcedTypesAndErrors() throws {
        let b = try read("a,b,c\n1,0xFF,2020-01-01 12:34:56.123\n-128,7,2020-01-01\n") {
            $0.columnTypes = ["a": .int8, "b": .uint8, "c": .timestamp(.milli, timezone: nil)]
        }
        XCTAssertEqual(b.columns.map { $0.arrowFormat }, ["c", "C", "tsm:"])
        guard case .int8(let a) = b.columns[0], case .uint8(let u) = b.columns[1] else { return XCTFail() }
        XCTAssertEqual(a.toArray(), [1, -128])
        XCTAssertEqual(u.toArray(), [255, 7])
        XCTAssertThrowsError(try read("a,b\n1,x\n") { $0.columnTypes = ["b": .int64] }) {
            XCTAssertEqual("\($0)", "In CSV column #1: Row #2: CSV conversion error to int64: invalid value 'x'")
        }
        XCTAssertThrowsError(try read("a\n2020-01-01 12:34:56\n") { $0.columnTypes = ["a": .timestamp(.second, timezone: "UTC")] }) {
            XCTAssertTrue("\($0)".contains("expected a zone offset in '2020-01-01 12:34:56'"), "\($0)")
        }
    }

    func testProjectionNeverConvertsOtherColumns() throws {
        // Column b would fail its forced type; projecting it away means it is never converted.
        let b = try read("a,b,c\n1,x,2.5\n") {
            $0.includeColumns = ["c", "a"]
            $0.columnTypes = ["b": .int64]
        }
        XCTAssertEqual(b.names, ["c", "a"])
        XCTAssertThrowsError(try read("a\n1\n") { $0.includeColumns = ["z"] }) {
            XCTAssertEqual("\($0)", "Column 'z' in include_columns does not exist in CSV file")
        }
        let m = try read("a\n1\n") { $0.includeColumns = ["a", "z"]; $0.includeMissingColumns = true }
        XCTAssertEqual(m.columns.map { $0.arrowFormat }, ["l", "n"])
    }

    /// Values the GPU float parser hands to the CPU (more than 19 significant digits on a rounding
    /// boundary), in a column converted by the all-columns pass and in one with a complex field (text
    /// after a closing quote), which is converted by its own kernel.
    func testFloatValuesDecidedOnTheCPU() throws {
        let hard = "9007199254740993.0000000000000000001"
        let arr = try MetalStringArray([hard])
        XCTAssertEqual(try arr.parseFloatGPUCounting(Double.self).hostRows, 1, "the premise: the GPU defers this one")
        let b = try read("a,b\n\(hard),\"9007199254740993\".0000000000000000001\n1.5,2.5\n")
        XCTAssertEqual(b.columns[0].asFloat64?.toArray(), [Double(hard)!, 1.5])
        XCTAssertEqual(b.columns[1].asFloat64?.toArray(), [Double(hard)!, 2.5])
        XCTAssertEqual(Double(hard), 9007199254740994)
    }

    func testFloatColumnMatchesCorrectlyRoundedParse() throws {
        var rng = CSVFloatParseTests.SplitMix64(state: 7)
        let values = CSVFloatParseTests.decimals(&rng, count: 5000)
        let text = "x\n" + values.joined(separator: "\n") + "\n"
        let b = try read(text) { $0.columnTypes = ["x": .float64] }
        guard let col = b.columns[0].asFloat64 else { return XCTFail() }
        for (i, s) in values.enumerated() {
            XCTAssertEqual(col[i]?.bitPattern, Double(s)?.bitPattern, s)
        }
    }

    /// The reader's integer grammar is Arrow's CSV one, not the grammar of Arrow's `cast` that
    /// `MetalStringArray.parse` implements: the cast takes `+3` and rejects spaces and hex, the CSV
    /// reader rejects `+` (the column becomes float64) and takes spaces and `0x`.
    func testIntegerGrammarIsTheCSVOneNotTheCasts() throws {
        let cast = try MetalStringArray(["+3", " 1", "0x1F"]).parse(Int64.self).toArray()
        XCTAssertEqual(cast, [3, nil, nil])
        let b = try read("a,b,c\n+3, 1,0x1F\n")
        XCTAssertEqual(b.columns.map { $0.arrowFormat }, ["g", "l", "l"])
        XCTAssertEqual(b.columns[0].asFloat64?.toArray(), [3.0])
        XCTAssertEqual(b.columns[1].asInt64?.toArray(), [1])
        XCTAssertEqual(b.columns[2].asInt64?.toArray(), [31])
    }

    /// `scanBlockBytes` changes only the speed, for any positive size: one past 32 bits once trapped.
    func testScanBlockBytesOfAnySize() throws {
        let path = try write("a,b\n1,\"x\ny\"\n2,z\n")
        for access in [CSVReadOptions.FileAccess.read, .map] {
            for size in [1, 7, 1 << 20, (1 << 32) - 1, 1 << 32, 1 << 40, Int.max] {
                var o = CSVReadOptions()
                o.scanBlockBytes = size
                o.fileAccess = access
                let b = try CSVReader.read(path: path, options: o)
                XCTAssertEqual(b.columns[0].asInt64?.toArray(), [1, 2], "\(size)")
                XCTAssertEqual(strings(b.columns[1]), ["x\ny", "z"], "\(size)")
            }
        }
    }

    private func tsValues(_ a: AnyMetalArray) -> [Int64?] {
        guard case .temporal(let t) = a, case .int64(let v) = t.storage else { XCTFail("not a timestamp: \(a)"); return [] }
        return v.toArray()
    }

    /// timestamp[ns] holds 1677-09-21 00:12:43.145224192 .. 2262-04-11 23:47:16.854775807, less the
    /// first value (Arrow scales the whole seconds before adding the fraction, and -9223372037 s does not
    /// scale). Outside it a fractional value is not a timestamp[ns], so inference moves on (to string),
    /// and a forced timestamp[ns] column raises; whole seconds stay timestamp[s] whatever the year.
    func testTimestampNanosecondRange() throws {
        let inside = try read("a\n1677-09-21 00:12:44.0\n2262-04-11 23:47:16.854775807\n")
        XCTAssertEqual(inside.columns[0].arrowFormat, "tsn:")
        XCTAssertEqual(tsValues(inside.columns[0]), [-9_223_372_036_000_000_000, Int64.max])
        for text in ["a\n1000-01-01 00:00:00.5\n2020-01-01 00:00:00.5\n", "a\n3000-01-01 00:00:00.5\n",
                     "a\n2262-04-11 23:47:16.854775808\n", "a\n1677-09-21 00:12:43.145224192\n",
                     "a\n1000-01-01 00:00:00.5Z\n", "a\n1677-09-21 00:12:44.0+01:00\n",
                     "a\n1000-01-01 00:00:00\n2020-01-01 00:00:00.5\n"] {
            XCTAssertEqual(try read(text).columns[0].arrowFormat, "u", text)
        }
        let zoned = try read("a\n2262-04-11 22:47:16.854775807-01:00\n1677-09-21 01:12:44.0+01:00\n")
        XCTAssertEqual(zoned.columns[0].arrowFormat, "tsn:UTC")
        XCTAssertEqual(tsValues(zoned.columns[0]), [Int64.max, -9_223_372_036_000_000_000])
        let seconds = try read("a\n1000-01-01 00:00:00\n9999-12-31 23:59:59\n")
        XCTAssertEqual(seconds.columns[0].arrowFormat, "tss:")
        XCTAssertEqual(tsValues(seconds.columns[0]), [-30_610_224_000, 253_402_300_799])

        let ns = CSVColumnType.timestamp(.nano, timezone: nil)
        for v in ["1000-01-01 00:00:00.5", "1000-01-01 00:00:00", "3000-01-01", "2262-04-11 23:47:16.854775808",
                  "1677-09-21 00:12:43.145224192", "1000-01-01 00:00:00Z"] {
            XCTAssertThrowsError(try read("a\n\(v)\n") { $0.columnTypes = ["a": ns] }) {
                XCTAssertEqual("\($0)", "In CSV column #0: Row #2: CSV conversion error to timestamp[ns]: invalid value '\(v)'")
            }
        }
        let forced = try read("a\n1677-09-21 00:12:44\n2262-04-11 23:47:16.854775807\n") { $0.columnTypes = ["a": ns] }
        XCTAssertEqual(tsValues(forced.columns[0]), [-9_223_372_036_000_000_000, Int64.max])
        let us = try read("a\n1000-01-01 00:00:00.5\n") { $0.columnTypes = ["a": .timestamp(.micro, timezone: nil)] }
        XCTAssertEqual(tsValues(us.columns[0]), [-30_610_223_999_500_000])
    }

    /// pyarrow skips `skip_rows_after_names` rows without checking their width and counts an empty line
    /// among them as a row; the rows after them are numbered with the skipped ones counted.
    func testSkipRowsAfterNamesSkipsRaggedRowsAndCountsEmptyLines() throws {
        let b = try read("a,b\n1,2,3\n4,5\n") { $0.skipRowsAfterNames = 1 }
        XCTAssertEqual(b.columns[0].asInt64?.toArray(), [4])
        XCTAssertEqual(b.columns[1].asInt64?.toArray(), [5])
        let q = try read("a,b\n\"1\n,2\",3,4\n4,5\n") { $0.skipRowsAfterNames = 1 }
        XCTAssertEqual(q.columns[0].asInt64?.toArray(), [4])
        let n = try read("1,2,3\n4,5\n") { $0.skipRowsAfterNames = 1; $0.columnNames = ["a", "b"] }
        XCTAssertEqual(n.columns[1].asInt64?.toArray(), [5])
        let e = try read("a,b\n1,2\n\n3,4\n5,6\n") { $0.skipRowsAfterNames = 2 }
        XCTAssertEqual(e.columns[0].asInt64?.toArray(), [3, 5])
        let lead = try read("a,b\n\n\n1,2\n3,4\n") { $0.skipRowsAfterNames = 1 }
        XCTAssertEqual(lead.columns[0].asInt64?.toArray(), [1, 3])
        let all = try read("a,b\n1,2\n\n\n") { $0.skipRowsAfterNames = 3 }
        XCTAssertEqual(all.names, ["a", "b"])
        XCTAssertEqual(all.columns[0].length, 0)
        XCTAssertThrowsError(try read("a,b\n1,2,3\n4,5\n6\n") { $0.skipRowsAfterNames = 1 }) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #4: Expected 2 columns, got 1: 6")
        }
        XCTAssertThrowsError(try read("a,b\r\r1,2,3\r3,4\r5\r") { $0.skipRowsAfterNames = 2 }) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #5: Expected 2 columns, got 1: 5")
        }
        XCTAssertThrowsError(try read("x\n\na,b\n1,2,3\n4,5\n6\n") { $0.skipRows = 2; $0.skipRowsAfterNames = 1 }) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #6: Expected 2 columns, got 1: 6")
        }
        XCTAssertThrowsError(try read("a,b\n\n1,2,3\n3,4\n5,x\n") { $0.skipRowsAfterNames = 2; $0.columnTypes = ["b": .int64] }) {
            XCTAssertEqual("\($0)", "In CSV column #1: Row #5: CSV conversion error to int64: invalid value 'x'")
        }
    }

    /// pyarrow quotes at most 100 bytes of a ragged row; a longer one is cut to 96 bytes and " ...".
    func testRaggedRowTextIsCutAt100Bytes() throws {
        let row100 = String(repeating: "x", count: 98) + ",1"
        XCTAssertThrowsError(try read("a\n1\n\(row100)\n")) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #3: Expected 1 columns, got 2: \(row100)")
        }
        let row101 = String(repeating: "x", count: 99) + ",1"
        XCTAssertThrowsError(try read("a\n1\n\(row101)\n")) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #3: Expected 1 columns, got 2: \(String(repeating: "x", count: 96)) ...")
        }
    }

    /// A ragged row that runs to the end of the file inside an open quote: the message leaves out one
    /// line terminator from the end of the row text, as pyarrow's does (messages probed with pyarrow 25).
    func testRaggedRowInAnOpenQuoteAtEOFDropsOneTerminator() throws {
        let cases: [(String, String)] = [
            ("a,b\n1,2,\"x\n", "got 3: 1,2,\"x"),
            ("a,b\n1,2,\"x\r\n", "got 3: 1,2,\"x"),
            ("a,b\n1,2,\"x\r", "got 3: 1,2,\"x"),
            ("a,b\n1,2,\"x\n\r", "got 3: 1,2,\"x\n"),
            ("a,b\n1,2,\"x\n\n\n", "got 3: 1,2,\"x\n\n"),
            ("a,b\n1,2,\"x\r\n\r\n", "got 3: 1,2,\"x\r\n"),
            ("a,b\n1,2,\"x", "got 3: 1,2,\"x"),
            ("a,b\n\"1\n", "got 1: \"1"),
        ]
        for (text, tail) in cases {
            for block in [1, 3, 4096] {
                XCTAssertThrowsError(try read(text) { $0.scanBlockBytes = block }) {
                    XCTAssertEqual("\($0)", "CSV parse error: Row #2: Expected 2 columns, \(tail)", text.debugDescription)
                }
            }
        }
        // The 100-byte cut applies to the text without that terminator.
        let row = "1,2,\"" + String(repeating: "y", count: 95)
        XCTAssertThrowsError(try read("a,b\n\(row)\r\n")) {
            XCTAssertEqual("\($0)", "CSV parse error: Row #2: Expected 2 columns, got 3: \(row)")
        }
    }

    /// A quote character equal to the delimiter never opens a quote, as in pyarrow.
    func testDelimiterEqualToQuoteChar() throws {
        let b = try read("a\"b\n1\"2\n") { $0.delimiter = UInt8(ascii: "\""); $0.quoteChar = UInt8(ascii: "\"") }
        XCTAssertEqual(b.names, ["a", "b"])
        XCTAssertEqual(b.columns[0].asInt64?.toArray(), [1])
        XCTAssertEqual(b.columns[1].asInt64?.toArray(), [2])
    }

    func testSkipRowsAndNames() throws {
        let b = try read("junk\n\njunk2\na,b\n1,2\n3,4\n") { $0.skipRows = 3; $0.skipRowsAfterNames = 1 }
        XCTAssertEqual(b.names, ["a", "b"])
        XCTAssertEqual(b.columns[0].asInt64?.toArray(), [3])
        let g = try read("1,2\n") { $0.autogenerateColumnNames = true }
        XCTAssertEqual(g.names, ["f0", "f1"])
        let n = try read("1,2\n") { $0.columnNames = ["x", "y"] }
        XCTAssertEqual(n.names, ["x", "y"])
        XCTAssertThrowsError(try read("j\nk") { $0.skipRows = 2 })
    }
}
