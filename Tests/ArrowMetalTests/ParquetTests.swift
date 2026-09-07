import XCTest
@testable import ArrowMetal

/// The Parquet reader: footer parsing, GPU decompression, every encoding, and the Arrow types they map to.
///
/// The fixtures under `Tests/Fixtures` are written by `generate_parquet.py`: each logical dataset appears
/// once per encoding/codec variant. Reading every variant of a dataset and comparing them element for
/// element checks the encodings against each other; `python/tests/test_parquet.py` then checks the whole
/// set against pyarrow's own reader.
final class ParquetTests: XCTestCase {

    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures")

    private func path(_ name: String) throws -> String {
        let p = Self.fixtures.appendingPathComponent(name + ".parquet").path
        guard FileManager.default.fileExists(atPath: p) else {
            throw XCTSkip("fixture \(name).parquet is missing; run Tests/Fixtures/generate_parquet.py")
        }
        return p
    }

    private func open(_ name: String) throws -> ParquetFile {
        try ParquetFile(path: try path(name))
    }

    // MARK: - element-wise comparison

    /// A per-element string for any array this reader produces, so two arrays can be compared exactly.
    static func fingerprint(_ a: AnyMetalArray) -> [String] {
        func nulls(_ validity: MetalArrowBuffer?, _ n: Int) -> [Bool] {
            guard let v = validity else { return [Bool](repeating: false, count: n) }
            let p = v.typed(UInt8.self)
            return (0..<n).map { !Bitmap.isSet(p, $0) }
        }
        switch a {
        case .int8(let x): return primitive(x)
        case .uint8(let x): return primitive(x)
        case .int16(let x): return primitive(x)
        case .uint16(let x): return primitive(x)
        case .int32(let x): return primitive(x)
        case .uint32(let x): return primitive(x)
        case .int64(let x): return primitive(x)
        case .uint64(let x): return primitive(x)
        case .float32(let x): return primitive(x)
        case .float64(let x): return primitive(x)
        case .boolean(let x):
            let bits = x.values.typed(UInt8.self)
            let isNull = nulls(x.validity, x.length)
            return (0..<x.length).map { isNull[$0] ? "null" : (Bitmap.isSet(bits, $0) ? "true" : "false") }
        case .string(let x), .binary(let x):
            let off = x.offsets.typed(Int32.self)
            let data = x.data.typed(UInt8.self)
            let isNull = nulls(x.validity, x.length)
            return (0..<x.length).map { i in
                if isNull[i] { return "null" }
                let lo = Int(off[i]), hi = Int(off[i + 1])
                return String(decoding: UnsafeBufferPointer(start: data + lo, count: hi - lo), as: UTF8.self)
                    + "#\(hi - lo)"
            }
        case .temporal(let t):
            switch t.storage {
            case .int32(let x): return primitive(x)
            case .int64(let x): return primitive(x)
            }
        case .decimal(let d):
            let p = d.values.typed(UInt8.self)
            let isNull = nulls(d.validity, d.length)
            let w = d.type.byteWidth
            return (0..<d.length).map { i in
                if isNull[i] { return "null" }
                return (0..<w).map { String(format: "%02x", p[i * w + (w - 1 - $0)]) }.joined()
            }
        case .fixedBinary(let f):
            let p = f.values.typed(UInt8.self)
            let isNull = nulls(f.validity, f.length)
            return (0..<f.length).map { i in
                if isNull[i] { return "null" }
                return (0..<f.byteWidth).map { String(format: "%02x", p[i * f.byteWidth + $0]) }.joined()
            }
        case .float16(let f): return primitive(f.bits)
        case .dictionary:
            return (try? fingerprint(a.decode())) ?? ["<dictionary decode failed>"]
        case .list(let l):
            let off = l.offsets.typed(Int32.self)
            let child = fingerprint(l.values)
            let isNull = nulls(l.validity, l.length)
            return (0..<l.length).map { i in
                if isNull[i] { return "null" }
                let lo = Int(off[i]), hi = Int(off[i + 1])
                return "[" + (lo..<hi).map { child[$0] }.joined(separator: ",") + "]"
            }
        default:
            return ["<unsupported \(a.arrowFormat)>"]
        }
    }

    private static func primitive<T: ArrowPrimitive>(_ x: MetalArray<T>) -> [String] {
        let p = x.values.typed(T.self)
        let v = x.validity?.typed(UInt8.self)
        return (0..<x.length).map { i in
            if let v, !Bitmap.isSet(v, i) { return "null" }
            return "\(p[i])"
        }
    }

    private func assertEqual(_ a: MetalRecordBatch, _ b: MetalRecordBatch, _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.names, b.names, "\(label): column names", file: file, line: line)
        for (i, name) in a.names.enumerated() where i < b.columns.count {
            let fa = Self.fingerprint(a.columns[i])
            let fb = Self.fingerprint(b.columns[i])
            XCTAssertEqual(fa.count, fb.count, "\(label).\(name): length", file: file, line: line)
            for j in 0..<Swift.min(fa.count, fb.count) where fa[j] != fb[j] {
                XCTFail("\(label).\(name)[\(j)]: \(fa[j]) != \(fb[j])", file: file, line: line)
                break
            }
        }
    }

    // MARK: - metadata

    func testFooter() throws {
        let f = try open("flat__plain_none")
        XCTAssertEqual(f.numRows, 700)
        XCTAssertGreaterThan(f.rowGroupCount, 0)
        XCTAssertTrue(f.columnNames.contains("id"))
        XCTAssertTrue(f.columnNames.contains("dec38"))
        let id = try XCTUnwrap(f.leaf(named: "id"))
        XCTAssertEqual(id.physical, .int64)
        XCTAssertEqual(id.maxDefinition, 1)          // pyarrow writes optional columns
        XCTAssertEqual(id.maxRepetition, 0)
        XCTAssertNotNil(f.metadata.createdBy)
    }

    func testMultipleRowGroups() throws {
        let f = try open("groups__plain_none")
        XCTAssertGreaterThan(f.rowGroupCount, 1)
        XCTAssertEqual(f.metadata.rowGroups.reduce(0) { $0 + $1.numRows }, f.numRows)
    }

    // MARK: - values

    func testKnownValues() throws {
        try requireRealGPU()
        let f = try open("flat__plain_none")
        let batch = try f.read(columns: ["id", "i32", "b", "s"])
        let id = try XCTUnwrap(batch["id"]?.asInt64)
        XCTAssertEqual(id.length, 700)
        XCTAssertEqual(id.nullCount, 0)
        let p = id.values.typed(Int64.self)
        for i in 0..<700 { XCTAssertEqual(p[i], Int64(i) * 3 - 1000, "row \(i)") }
        let s = try XCTUnwrap(batch["s"])
        let strings = Self.fingerprint(s)
        let cats = ["alpha", "beta", "gamma", "delta", "epsilon", "", "a much longer category value here"]
        for i in 0..<700 {
            let want = cats[i % cats.count]
            XCTAssertEqual(strings[i], want + "#\(want.utf8.count)", "row \(i)")
        }
    }

    /// Every encoding and codec variant of a dataset must decode to exactly the same values.
    func testVariantsAgree() throws {
        try requireRealGPU()
        let families: [(String, [String], [String]?)] = [
            ("flat", ["plain_none", "dict_none", "plain_snappy", "dict_snappy", "v2_snappy", "v2_none"], nil),
            ("nulls", ["plain_none", "dict_none", "plain_snappy", "dict_snappy", "v2_snappy", "v2_none"], nil),
            ("floats", ["plain_none", "bss_none", "bss_snappy"], nil),
            ("delta", ["plain_none", "none", "snappy"], nil),
            ("deltanulls", ["plain_none", "none"], nil),
            ("groups", ["plain_none", "dict_snappy"], nil),
            ("strings", ["plain_none", "dict_snappy", "v2_zstd"], nil),
            ("lists", ["plain_none", "dict_snappy"], nil),
            // The same decimals stored as INT32/INT64 and as FIXED_LEN_BYTE_ARRAY must agree.
            ("decint", ["plain_fixed", "plain_none", "dict_snappy"], nil),
        ]
        for (family, variants, cols) in families {
            var reference: MetalRecordBatch? = nil
            for v in variants {
                let name = "\(family)__\(v)"
                let p = Self.fixtures.appendingPathComponent(name + ".parquet").path
                guard FileManager.default.fileExists(atPath: p) else { continue }
                let f = try ParquetFile(path: p)
                let batch: MetalRecordBatch
                do { batch = try f.read(ParquetReadOptions(columns: cols)) }
                catch let e as ParquetError {
                    if case .unsupported(let why) = e, why.contains("libzstd") {
                        continue                                  // documented host fallback
                    }
                    XCTFail("\(name): \(e)")
                    continue
                }
                if let reference { assertEqual(reference, batch, name) } else { reference = batch }
            }
            XCTAssertNotNil(reference, "no variant of \(family) could be read")
        }
    }

    /// Codecs written over the same small table.
    func testCodecs() throws {
        try requireRealGPU()
        let reference = try open("flat__plain_none").read(columns: ["id", "i32", "f32", "f64", "b", "s", "ts_us"])
        for codec in ["gzip", "lz4", "zstd", "brotli"] {
            for variant in ["plain", "v2"] {
                let name = "flat__\(variant)_\(codec)"
                let p = Self.fixtures.appendingPathComponent(name + ".parquet").path
                guard FileManager.default.fileExists(atPath: p) else { continue }
                do {
                    let batch = try ParquetFile(path: p).read()
                    assertEqual(reference, batch, name)
                } catch let e as ParquetError {
                    if case .unsupported(let why) = e, why.contains("libzstd") {
                        print("skipping \(name): \(why)")
                        continue
                    }
                    XCTFail("\(name): \(e)")
                }
            }
        }
    }

    func testNullsAndEdgeShapes() throws {
        try requireRealGPU()
        let n = try open("nulls__plain_none").read()
        for (i, name) in n.names.enumerated() {
            let col = n.columns[i]
            XCTAssertEqual(col.length, 700, name)
            XCTAssertGreaterThan(col.nullCount, 0, "\(name) should have nulls")
        }
        let empty = try open("empty__plain_none").read()
        for c in empty.columns { XCTAssertEqual(c.length, 0) }
        let all = try open("allnull__plain_none").read()
        XCTAssertEqual(all["x"]?.length, 500)
        XCTAssertEqual(all["x"]?.nullCount, 500)
        XCTAssertEqual(all["y"]?.nullCount, 500)
        let one = try open("one__plain_none").read()
        XCTAssertEqual(one["x"]?.length, 1)
        XCTAssertEqual(one["x"]?.asInt64?.values.typed(Int64.self)[0], 1)
    }

    func testTypes() throws {
        try requireRealGPU()
        let b = try open("flat__plain_none").read()
        XCTAssertEqual(b["id"]?.arrowFormat, "l")
        XCTAssertEqual(b["i32"]?.arrowFormat, "i")
        XCTAssertEqual(b["i16"]?.arrowFormat, "s")
        XCTAssertEqual(b["u32"]?.arrowFormat, "I")
        XCTAssertEqual(b["u64"]?.arrowFormat, "L")
        XCTAssertEqual(b["f32"]?.arrowFormat, "f")
        XCTAssertEqual(b["f64"]?.arrowFormat, "g")
        XCTAssertEqual(b["b"]?.arrowFormat, "b")
        XCTAssertEqual(b["s"]?.arrowFormat, "u")
        XCTAssertEqual(b["blob"]?.arrowFormat, "z")
        XCTAssertEqual(b["fx"]?.arrowFormat, "w:4")
        XCTAssertEqual(b["ts_us"]?.arrowFormat, "tsu:")     // pyarrow writes tz-naive timestamps
        XCTAssertEqual(b["ts_ms"]?.arrowFormat, "tsm:")
        XCTAssertEqual(b["ts_ns"]?.arrowFormat, "tsn:")
        XCTAssertEqual(b["date"]?.arrowFormat, "tdD")
        XCTAssertEqual(b["time_ms"]?.arrowFormat, "ttm")
        XCTAssertEqual(b["time_us"]?.arrowFormat, "ttu")
        XCTAssertEqual(b["dec9"]?.arrowFormat, "d:9,2")
        XCTAssertEqual(b["dec18"]?.arrowFormat, "d:18,4")
        XCTAssertEqual(b["dec38"]?.arrowFormat, "d:38,10")
    }

    func testInt96() throws {
        try requireRealGPU()
        let b = try open("int96__plain_none").read()
        let t = try XCTUnwrap(b["t"]?.asTemporal)
        XCTAssertEqual(t.type, .timestamp(.nano, timezone: nil))
        guard case .int64(let v) = t.storage else { return XCTFail("int96 should widen to int64") }
        let p = v.values.typed(Int64.self)
        for i in 0..<200 {
            XCTAssertEqual(p[i], 1_600_000_000_000_000_000 + Int64(i) * 1_000_000_000, "row \(i)")
        }
    }

    func testDictionaryEncodedOutput() throws {
        try requireRealGPU()
        let f = try open("flat__dict_snappy")
        let dict = try f.read(ParquetReadOptions(columns: ["s"], dictionaryEncoded: true))
        guard case .dictionary(let codes, let values)? = dict["s"] else {
            return XCTFail("expected a dictionary-encoded column, got \(String(describing: dict["s"]?.arrowFormat))")
        }
        XCTAssertEqual(codes.length, 700)
        XCTAssertLessThanOrEqual(values.length, 16)
        let plain = try f.read(ParquetReadOptions(columns: ["s"], dictionaryEncoded: false))
        XCTAssertEqual(Self.fingerprint(dict["s"]!), Self.fingerprint(plain["s"]!))
    }

    // MARK: - projection, row groups, statistics

    func testProjection() throws {
        try requireRealGPU()
        let f = try open("flat__plain_snappy")
        let b = try f.read(columns: ["f64", "id"])
        XCTAssertEqual(b.names, ["f64", "id"])
        XCTAssertEqual(b.columnCount, 2)
    }

    func testRowGroupSelection() throws {
        try requireRealGPU()
        let f = try open("groups__plain_none")
        let all = try f.read(columns: ["id"])
        let first = try f.read(ParquetReadOptions(columns: ["id"], rowGroups: [0]))
        XCTAssertEqual(first.length, Int(f.metadata.rowGroups[0].numRows))
        let fa = Self.fingerprint(all["id"]!), ff = Self.fingerprint(first["id"]!)
        XCTAssertEqual(Array(fa.prefix(ff.count)), ff)
        let last = try f.read(ParquetReadOptions(columns: ["id"], rowGroups: [f.rowGroupCount - 1]))
        XCTAssertEqual(Array(fa.suffix(last.length)), Self.fingerprint(last["id"]!))
    }

    func testStatisticsPushdown() throws {
        try requireRealGPU()
        let f = try open("groups__plain_none")
        // `id` runs from -1000 upwards in steps of 3, so a high threshold excludes the early groups.
        let opts = ParquetReadOptions(columns: ["id"], filters: [ParquetFilter(column: "id", op: .gt, value: .int(1000))])
        let kept = try f.selectedRowGroups(opts)
        XCTAssertLessThan(kept.count, f.rowGroupCount)
        let b = try f.read(opts)
        let values = b["id"]!.asInt64!.values.typed(Int64.self)
        // Everything above the threshold must still be present.
        let total = (0..<700).map { Int64($0) * 3 - 1000 }.filter { $0 > 1000 }
        let got = Set((0..<b.length).map { values[$0] })
        for v in total { XCTAssertTrue(got.contains(v), "statistics pushdown dropped \(v)") }
        // An impossible predicate keeps nothing.
        let none = try f.selectedRowGroups(ParquetReadOptions(filters: [ParquetFilter(column: "id", op: .gt, value: .int(1_000_000))]))
        XCTAssertTrue(none.isEmpty)
    }

    // MARK: - lists

    func testLists() throws {
        try requireRealGPU()
        let b = try open("lists__plain_none").read()
        let xs = try XCTUnwrap(b["xs"])
        XCTAssertEqual(xs.length, 400)
        let f = Self.fingerprint(xs)
        for i in 0..<400 {
            if i % 17 == 0 { XCTAssertEqual(f[i], "null", "row \(i)") }
            else if i % 7 == 0 { XCTAssertEqual(f[i], "[]", "row \(i)") }
            else {
                let k = i % 5 + 1
                let want = "[" + (0..<k).map { j -> String in
                    (i + j) % 11 == 0 ? "null" : "\(i * 100 + j)"
                }.joined(separator: ",") + "]"
                XCTAssertEqual(f[i], want, "row \(i)")
            }
        }
    }

    // MARK: - long strings

    func testLongStrings() throws {
        try requireRealGPU()
        let b = try open("strings__plain_none").read()
        let f = Self.fingerprint(b["s"]!)
        XCTAssertEqual(f[5], "null")
        XCTAssertTrue(f[7].hasSuffix("#65535"), "expected a 64 KB value, got \(f[7].suffix(12))")
        XCTAssertEqual(f[0], "#0")
    }
}
