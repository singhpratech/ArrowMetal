import XCTest
@testable import ArrowMetal

/// The `ARROW:schema` key/value metadata: time zones, durations, extension types and field metadata come
/// back, the `null` type reads as `null`, and absent or malformed metadata is ignored.
/// `python/tests/test_parquet_nested.py` checks the same fixtures against `pyarrow.parquet.read_table`.
final class ParquetArrowSchemaTests: XCTestCase {

    private func open(_ name: String) throws -> ParquetFile { try ParquetNestedTests.open(name) }

    private func temporalType(_ a: AnyMetalArray?) -> ArrowTemporalType? { a?.asTemporal?.type }

    func testStoredSchemaDecodes() throws {
        let f = try open("arrowschema__pa_plain_none")
        let stored = try XCTUnwrap(f.arrowSchema)
        XCTAssertEqual(stored.map(\.name), f.columnNames)
        XCTAssertEqual(stored.first { $0.name == "ts_paris" }?.kind, .timestamp(.micro, timezone: "Europe/Paris"))
        XCTAssertEqual(stored.first { $0.name == "dur_s" }?.kind, .duration(.second))
        XCTAssertEqual(stored.first { $0.name == "nothing" }?.kind, .null)
        XCTAssertEqual(stored.first { $0.name == "u" }?.extensionName, "arrow.uuid")
        XCTAssertEqual(stored.first { $0.name == "inner_tz" }?.children.map(\.name), ["t", "d"])
    }

    func testTimeZonesDurationsAndNullComeBack() throws {
        try requireRealGPU()
        let b = try open("arrowschema__pa_plain_none").read()
        XCTAssertEqual(temporalType(b["ts_paris"]), .timestamp(.micro, timezone: "Europe/Paris"))
        XCTAssertEqual(temporalType(b["ts_ny_ms"]), .timestamp(.milli, timezone: "America/New_York"))
        XCTAssertEqual(temporalType(b["ts_off_ns"]), .timestamp(.nano, timezone: "+05:30"))
        XCTAssertEqual(temporalType(b["ts_utc"]), .timestamp(.micro, timezone: "UTC"))
        XCTAssertEqual(temporalType(b["ts_naive"]), .timestamp(.micro, timezone: nil))
        XCTAssertEqual(temporalType(b["dur_s"]), .duration(.second))
        XCTAssertEqual(temporalType(b["dur_us"]), .duration(.micro))
        guard case .null(let n)? = b["nothing"] else { return XCTFail("nothing is \(String(describing: b["nothing"]?.arrowFormat))") }
        XCTAssertEqual(n.length, 300)
        // The instants are untouched: the zone is metadata.
        let paris = try XCTUnwrap(b["ts_paris"]?.asTemporal), utc = try XCTUnwrap(b["ts_utc"]?.asTemporal)
        guard case .int64(let pv) = paris.storage, case .int64(let uv) = utc.storage else { return XCTFail() }
        XCTAssertEqual(ParquetTests.fingerprint(.int64(pv)), ParquetTests.fingerprint(.int64(uv)))
    }

    func testNestedFieldsTakeTheirTypesBack() throws {
        try requireRealGPU()
        let b = try open("arrowschema__pa_plain_none").read()
        guard case .structure(let s)? = b["inner_tz"] else { return XCTFail("inner_tz is not a struct") }
        XCTAssertEqual(temporalType(s.children[0]), .timestamp(.milli, timezone: "Asia/Tokyo"))
        XCTAssertEqual(temporalType(s.children[1]), .duration(.milli))
        guard case .list(let l)? = b["dur_list"] else { return XCTFail("dur_list is not a list") }
        XCTAssertEqual(temporalType(l.values), .duration(.nano))
        guard case .map(let m)? = b["tz_map"] else { return XCTFail("tz_map is not a map") }
        XCTAssertEqual(temporalType(m.items), .timestamp(.micro, timezone: "Australia/Sydney"))
    }

    func testExtensionTypesAndFieldMetadata() throws {
        try requireRealGPU()
        let f = try open("arrowschema__pa_plain_none")
        let b = try f.read()
        XCTAssertEqual(b["u"]?.extensionName, "arrow.uuid")
        XCTAssertEqual(b["u"]?.storageArray.arrowFormat, "w:16")
        XCTAssertEqual(b["label"]?.extensionName, "example.label")
        XCTAssertEqual(b["rat"]?.extensionName, "example.rational")
        XCTAssertEqual(b["rat"]?.extensionMetadata.map { String(decoding: $0, as: UTF8.self) }, "{\"normalised\":true}")
        XCTAssertNil(b["price"]?.extensionName)
        XCTAssertEqual(f.arrowFieldMetadata(column: "price").string("currency"), "EUR")
        XCTAssertEqual(f.arrowFieldMetadata(column: "fid").string("PARQUET:field_id"), "42")
        XCTAssertTrue(f.arrowFieldMetadata(column: "ts_naive").isEmpty)
        XCTAssertEqual(f.arrowSchemaMetadata.string("source"), "generate_parquet_nested.py")
        XCTAssertNil(f.arrowSchemaMetadata.string(ParquetFile.arrowSchemaKey))
    }

    func testDictionaryEncodedColumnsTakeTheZoneOnTheirDictionary() throws {
        try requireRealGPU()
        let b = try open("arrowschema__pa_dict_snappy").read(ParquetReadOptions(columns: ["ts_paris", "dur_s"],
                                                                                 dictionaryEncoded: true))
        guard case .dictionary(_, let values)? = b["ts_paris"] else { return XCTFail("ts_paris is not dictionary encoded") }
        XCTAssertEqual(temporalType(values), .timestamp(.micro, timezone: "Europe/Paris"))
        guard case .dictionary(_, let durations)? = b["dur_s"] else { return XCTFail("dur_s is not dictionary encoded") }
        XCTAssertEqual(temporalType(durations), .duration(.second))
    }

    func testArrowTypesParquetCannotName() throws {
        try requireRealGPU()
        let b = try open("arrowschema__pa_types").read(ParquetReadOptions(dictionaryEncoded: false))
        XCTAssertEqual(b["d32"]?.arrowFormat, "d:7,2,32")
        XCTAssertEqual(b["d64"]?.arrowFormat, "d:15,3,64")
        XCTAssertEqual(b["fsl"]?.arrowFormat, "+w:3")
        guard case .dictionary(_, let values)? = b["cat"] else { return XCTFail("cat is not dictionary encoded") }
        XCTAssertEqual(values.arrowFormat, "u")
        // No view or 64-bit-offset layouts in the engine: these read as their 32-bit twins.
        XCTAssertEqual(b["sv"]?.arrowFormat, "u")
        XCTAssertEqual(b["ls"]?.arrowFormat, "u")
        XCTAssertEqual(b["ll"]?.arrowFormat, "+l")
        XCTAssertEqual(b["lv"]?.arrowFormat, "+l")
        // A fixed-size list with null rows pads each null row with `width` null child slots.
        let f = try open("arrowschema__pa_fslnull").read()
        guard case .list(let l)? = f["fsl"] else { return XCTFail("fsl is not a list") }
        XCTAssertEqual(l.kind, .fixedSize(3))
        XCTAssertEqual(l.values.length, 60 * 3)
        let rows = ParquetNestedTests.rows(.list(l))
        for i in 0..<60 {
            XCTAssertEqual(rows[i], i % 7 == 0 ? "null" : "[\(i), \(-i), \(i % 5 == 0 ? "null" : "\(i * 2)")]", "fsl[\(i)]")
        }
    }

    func testDottedLeafReadsAsTheParquetSchemaSays() throws {
        try requireRealGPU()
        let b = try open("arrowschema__pa_plain_none").read(columns: ["inner_tz.t"])
        XCTAssertEqual(temporalType(b["inner_tz.t"]), .timestamp(.milli, timezone: "UTC"))
    }

    func testAbsentOrMalformedMetadataIsIgnored() throws {
        try requireRealGPU()
        for name in ["arrowschema__pa_nostore", "arrowschema__pa_corrupt", "arrowschema__pa_truncated"] {
            let f = try open(name)
            XCTAssertNil(f.arrowSchema, name)
            let b = try f.read(ParquetReadOptions(dictionaryEncoded: false))
            XCTAssertEqual(temporalType(b["ts_paris"]), .timestamp(.micro, timezone: "UTC"), name)
            XCTAssertEqual(b.length, 300, name)
        }
    }
}
