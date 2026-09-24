import XCTest
@testable import ArrowMetal

/// Nested Parquet columns — structs, maps and lists at any depth — reassembled from their leaves.
///
/// The fixtures under `Tests/Fixtures/nested` are written by `generate_parquet_nested.py`: each dataset
/// once per pyarrow encoding/codec/page-version variant (with small pages, so levels cross many page
/// boundaries), once by DuckDB and once by Polars where Polars can express the shape. These tests check
/// known values against the generator's formulas and every writer's file against every other's;
/// `python/tests/test_parquet_nested.py` checks each file against `pyarrow.parquet.read_table`.
final class ParquetNestedTests: XCTestCase {

    static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures").appendingPathComponent("nested")

    static func path(_ name: String) throws -> String {
        let p = fixtures.appendingPathComponent(name + ".parquet").path
        guard FileManager.default.fileExists(atPath: p) else {
            throw XCTSkip("fixture nested/\(name).parquet is missing; run Tests/Fixtures/generate_parquet_nested.py")
        }
        return p
    }

    static func open(_ name: String) throws -> ParquetFile { try ParquetFile(path: try path(name)) }

    static let variants = ["pa_plain_none", "pa_dict_snappy", "pa_v2_snappy", "pa_v2_lz4", "duckdb", "polars"]

    // MARK: - rendering

    /// One Python-like string per row, for any array the reader produces: `{a: 1, b: "x"}`, `[1, null]`,
    /// `{"k": 1}` for a map, `null`. Strings are quoted so an empty string and a null differ.
    static func rows(_ a: AnyMetalArray) -> [String] {
        func isNull(_ v: MetalArrowBuffer?, _ i: Int) -> Bool { v.map { !Bitmap.isSet($0.typed(UInt8.self), i) } ?? false }
        switch a {
        case .structure(let s):
            let kids = s.children.map { rows($0) }
            return (0..<s.length).map { i in
                if isNull(s.validity, i) { return "null" }
                return "{" + s.names.indices.map { "\(s.names[$0]): \(kids[$0][i])" }.joined(separator: ", ") + "}"
            }
        case .list(let l):
            let kid = rows(l.values)
            let off = l.offsets.typed(Int32.self)
            return (0..<l.length).map { i in
                if isNull(l.validity, i) { return "null" }
                return "[" + (Int(off[i])..<Int(off[i + 1])).map { kid[$0] }.joined(separator: ", ") + "]"
            }
        case .map(let m):
            let keys = rows(m.keys), items = rows(m.items)
            let off = m.entries.offsets.typed(Int32.self)
            return (0..<m.length).map { i in
                if isNull(m.entries.validity, i) { return "null" }
                return "{" + (Int(off[i])..<Int(off[i + 1])).map { "\(keys[$0]): \(items[$0])" }
                    .joined(separator: ", ") + "}"
            }
        case .string(let s):
            let off = s.offsets.typed(Int32.self), data = s.data.typed(UInt8.self)
            return (0..<s.length).map { i in
                if isNull(s.validity, i) { return "null" }
                let lo = Int(off[i]), hi = Int(off[i + 1])
                return "\"" + String(decoding: UnsafeBufferPointer(start: data + lo, count: hi - lo), as: UTF8.self) + "\""
            }
        case .dictionary:
            return (try? rows(a.decode())) ?? ["<dictionary decode failed>"]
        case .boolean(let b):
            let bits = b.values.typed(UInt8.self)
            return (0..<b.length).map { isNull(b.validity, $0) ? "null" : (Bitmap.isSet(bits, $0) ? "true" : "false") }
        default:
            return ParquetTests.fingerprint(a)
        }
    }

    // MARK: - known values

    func testStructsKnownValues() throws {
        try requireRealGPU()
        let b = try Self.open("structs__pa_plain_none").read()
        XCTAssertEqual(b.names, ["k", "s", "ss", "sss", "sb"])
        XCTAssertEqual(b["s"]?.arrowFormat, "+s")
        guard case .structure(let s)? = b["s"], case .structure(let ss)? = b["ss"],
              case .structure(let sss)? = b["sss"] else { return XCTFail("expected struct columns") }
        XCTAssertEqual(s.names, ["a", "b", "c", "d"])
        XCTAssertEqual(s.length, 600)
        XCTAssertEqual(s.nullCount, (0..<600).filter { $0 % 11 == 0 }.count)
        XCTAssertEqual(ss.nullCount, (0..<600).filter { $0 % 13 == 0 }.count)
        let sRows = Self.rows(.structure(s)), ssRows = Self.rows(.structure(ss)), sssRows = Self.rows(.structure(sss))
        let bRows = Self.rows(s.children[1]), dRows = Self.rows(s.children[3])
        for i in 0..<600 {
            if i % 11 == 0 {
                XCTAssertEqual(sRows[i], "null", "s[\(i)]")
            } else {
                let want = i % 5 == 1 ? "null" : "\"str-\(i)-\(String(repeating: "y", count: i % 13))\""
                XCTAssertEqual(bRows[i], want, "s[\(i)].b")
                XCTAssertEqual(dRows[i], i % 4 == 0 ? "null" : (i % 3 != 0 ? "true" : "false"), "s[\(i)].d")
            }
            // ss = {inner: {x, y}, z}
            if i % 13 == 0 {
                XCTAssertEqual(ssRows[i], "null", "ss[\(i)]")
            } else {
                let x = i % 8 == 1 ? "null" : "\(i * 1000 - 7)"
                let y = i % 10 == 4 ? "null" : "\"in\(i % 37)\""
                let inner = i % 6 == 5 ? "null" : "{x: \(x), y: \(y)}"
                let z = i % 3 == 2 ? "null" : "\(i)"
                XCTAssertEqual(ssRows[i], "{inner: \(inner), z: \(z)}", "ss[\(i)]")
            }
            // sss = {p: {q: {r}}}, three optional levels deep
            let r = i % 7 == 6 ? "null" : "\((i % 30000) - 15000)"
            let q = i % 5 == 3 ? "null" : "{r: \(r)}"
            let p = i % 9 == 8 ? "null" : "{q: \(q)}"
            XCTAssertEqual(sssRows[i], i % 17 == 0 ? "null" : "{p: \(p)}", "sss[\(i)]")
        }
    }

    /// `ll`'s rows, from the generator's formula.
    static func expectedLL(_ i: Int) -> String {
        if i % 13 == 0 { return "null" }
        if i % 10 == 0 { return "[]" }
        let inner = (0..<(i % 4 + 1)).map { j -> String in
            if (i + j) % 9 == 0 { return "null" }
            if (i + j) % 5 == 0 { return "[]" }
            return "[" + (0..<((i + j) % 4 + 1)).map { t in
                (i + j + t) % 11 == 0 ? "null" : "\(i * 100 + j * 10 + t)"
            }.joined(separator: ", ") + "]"
        }
        return "[" + inner.joined(separator: ", ") + "]"
    }

    func testNestedListsKnownValues() throws {
        try requireRealGPU()
        let b = try Self.open("lists__pa_plain_none").read()
        let ll = Self.rows(try XCTUnwrap(b["ll"]))
        XCTAssertEqual(ll.count, 600)
        for i in 0..<600 { XCTAssertEqual(ll[i], Self.expectedLL(i), "ll[\(i)]") }
        // struct<xs: list<int64>, name: string>
        let sl = Self.rows(try XCTUnwrap(b["sl"]))
        for i in 0..<600 {
            let want: String
            if i % 12 == 0 { want = "null" } else {
                let xs = i % 5 == 0 ? "null" : "[" + (0..<(i % 6)).map { "\(i + $0)" }.joined(separator: ", ") + "]"
                let name = i % 7 == 0 ? "null" : "\"row\(i)\""
                want = "{xs: \(xs), name: \(name)}"
            }
            XCTAssertEqual(sl[i], want, "sl[\(i)]")
        }
        // list<struct<a: int32, b: string>>
        let ls = Self.rows(try XCTUnwrap(b["ls"]))
        for i in 0..<600 {
            let want: String
            if i % 11 == 0 { want = "null" } else {
                want = "[" + (0..<(i % 4)).map { j -> String in
                    (i + j) % 8 == 0 ? "null" : "{a: \(j == 2 ? "null" : "\(i + j)"), b: \"e\(i * j)\"}"
                }.joined(separator: ", ") + "]"
            }
            XCTAssertEqual(ls[i], want, "ls[\(i)]")
        }
        XCTAssertEqual(b["lll"]?.arrowFormat, "+l")
        XCTAssertEqual(b["lm"]?.arrowFormat, "+l")
        guard case .list(let lm)? = b["lm"] else { return XCTFail("lm is not a list") }
        XCTAssertEqual(lm.values.arrowFormat, "+m")
    }

    func testMapsKnownValues() throws {
        try requireRealGPU()
        let b = try Self.open("maps__pa_plain_none").read()
        guard case .map(let m)? = b["m"] else { return XCTFail("m is not a map") }
        XCTAssertEqual(m.length, 600)
        XCTAssertEqual(m.keys.nullCount, 0)
        let rows = Self.rows(.map(m))
        for i in 0..<600 {
            let want: String
            if i % 13 == 0 { want = "null" } else if i % 7 == 0 { want = "{}" } else {
                want = "{" + (0..<(i % 4 + 1)).map { j in
                    "\"key\((i + j) % 50)\": " + ((i + j) % 6 == 0 ? "null" : "\(i * 10 + j)")
                }.joined(separator: ", ") + "}"
            }
            XCTAssertEqual(rows[i], want, "m[\(i)]")
        }
        XCTAssertEqual(b["ms"]?.arrowFormat, "+m")
        XCTAssertEqual(b["ml"]?.arrowFormat, "+m")
    }

    func testRequiredStructMembers() throws {
        try requireRealGPU()
        let b = try Self.open("reqstruct__pa_plain_none").read()
        let rows = Self.rows(try XCTUnwrap(b["rs"]))
        for i in 0..<200 { XCTAssertEqual(rows[i], i % 7 == 0 ? "null" : "{a: \(i), b: \"r\(i)\"}", "rs[\(i)]") }
    }

    // MARK: - every writer and encoding agrees

    func testWritersAndEncodingsAgree() throws {
        try requireRealGPU()
        for base in ["structs", "lists", "maps"] {
            let reference = try Self.open(base + "__pa_plain_none").read()
            var compared = 0
            for v in Self.variants.dropFirst() {
                guard let p = try? Self.path(base + "__" + v) else { continue }
                let other = try ParquetFile(path: p).read()
                XCTAssertEqual(other.names, reference.names, "\(base)__\(v): names")
                // Polars has no map type: it writes a map as a list of key/value structs.
                for (i, name) in reference.names.enumerated() where !(v == "polars" && name == "lm") {
                    let want = Self.rows(reference.columns[i])
                    let got = Self.rows(try XCTUnwrap(other[name]))
                    XCTAssertEqual(got.count, want.count, "\(base)__\(v).\(name): length")
                    if let j = (0..<Swift.min(got.count, want.count)).first(where: { got[$0] != want[$0] }) {
                        XCTFail("\(base)__\(v).\(name)[\(j)]: \(got[j]) != \(want[j])")
                    }
                }
                compared += 1
            }
            XCTAssertGreaterThanOrEqual(compared, 4, "\(base): too few variants present")
        }
    }

    // MARK: - projection, row groups, dictionary output

    func testProjectionAndRowGroups() throws {
        try requireRealGPU()
        let f = try Self.open("maps__pa_dict_snappy")
        XCTAssertGreaterThan(f.rowGroupCount, 1)
        let all = try f.read(columns: ["ml", "k"])
        XCTAssertEqual(all.names, ["ml", "k"])
        let whole = Self.rows(try XCTUnwrap(all["ml"]))
        var start = 0
        for g in 0..<f.rowGroupCount {
            let part = try f.read(ParquetReadOptions(columns: ["ml"], rowGroups: [g]))
            let got = Self.rows(try XCTUnwrap(part["ml"]))
            XCTAssertEqual(got, Array(whole[start..<(start + got.count)]), "row group \(g)")
            start += got.count
        }
        XCTAssertEqual(start, whole.count)
    }

    func testDictionaryEncodedLeavesInsideNestedColumns() throws {
        try requireRealGPU()
        let f = try Self.open("lists__pa_dict_snappy")
        let encoded = try f.read(ParquetReadOptions(columns: ["lls", "ls"], dictionaryEncoded: true))
        let plain = try f.read(ParquetReadOptions(columns: ["lls", "ls"], dictionaryEncoded: false))
        XCTAssertEqual(Self.rows(encoded["lls"]!), Self.rows(plain["lls"]!))
        XCTAssertEqual(Self.rows(encoded["ls"]!), Self.rows(plain["ls"]!))
    }

    func testDottedLeafPathStillReadsFlat() throws {
        try requireRealGPU()
        let f = try Self.open("structs__pa_plain_none")
        let b = try f.read(columns: ["s.b"])
        let rows = Self.rows(try XCTUnwrap(b["s.b"]))
        XCTAssertEqual(rows.count, 600)
        for i in 0..<600 where i % 11 == 0 || i % 5 == 1 { XCTAssertEqual(rows[i], "null", "s.b[\(i)]") }
    }

    func testSchemaLevels() throws {
        let f = try Self.open("lists__pa_plain_none")
        guard let ll = f.fields.first(where: { $0.name == "ll" }), case .list(let inner, _) = ll.kind,
              case .list(let leaf, _) = inner.kind else { return XCTFail("ll is not list<list<...>>") }
        // optional ll (1) / repeated list (2) / optional element (3) / repeated list (4) / optional element (5)
        XCTAssertEqual([ll.definitionLevel, ll.repetitionLevel, ll.slotDefinitionLevel], [1, 0, 0])
        XCTAssertEqual([inner.definitionLevel, inner.repetitionLevel, inner.slotDefinitionLevel], [3, 1, 2])
        XCTAssertEqual([leaf.definitionLevel, leaf.repetitionLevel, leaf.slotDefinitionLevel], [5, 2, 4])
        let m = try Self.open("maps__pa_plain_none").fields.first { $0.name == "m" }
        XCTAssertEqual(m?.isMap, true)
    }
}
