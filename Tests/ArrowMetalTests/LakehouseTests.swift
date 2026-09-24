import XCTest
@testable import ArrowMetal

/// Delta Lake and Iceberg reads against the fixtures in `Tests/Fixtures/lakehouse`.
///
/// `expected.json` holds, for each read, the rows the reference readers returned (deltalake's
/// `to_pyarrow_table`, pyiceberg's `scan().to_arrow()`; the column-mapping table's reference is the
/// pyarrow data it was written from). `generate_lakehouse.py` writes both the tables and that file.
/// Rows are compared as sorted lists of per-value fingerprints, because neither format promises an order.
final class LakehouseTests: XCTestCase {

    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures").appendingPathComponent("lakehouse")

    private func fixture(_ rel: String) throws -> String {
        let p = Self.root.appendingPathComponent(rel).path
        guard FileManager.default.fileExists(atPath: p) else {
            throw XCTSkip("fixture \(rel) is missing; run Tests/Fixtures/lakehouse/generate_lakehouse.py")
        }
        return p
    }

    // MARK: - fingerprints (must match fingerprint_value in generate_lakehouse.py)

    static func fingerprint(_ a: AnyMetalArray) throws -> [String] {
        let c = try a.decodedIfDictionary()
        func ints<T: ArrowPrimitive & FixedWidthInteger>(_ x: MetalArray<T>) -> [String] {
            x.toArray().map { $0.map { String($0) } ?? "null" }
        }
        switch c {
        case .int8(let x): return ints(x)
        case .int16(let x): return ints(x)
        case .int32(let x): return ints(x)
        case .int64(let x): return ints(x)
        case .float32(let x): return x.toArray().map { $0.map { "\(Double($0))" } ?? "null" }
        case .float64(let x): return x.toArray().map { $0.map { "\($0)" } ?? "null" }
        case .boolean(let x): return x.toArray().map { $0.map { $0 ? "true" : "false" } ?? "null" }
        case .string(let x): return x.toArray().map { $0 ?? "null" }
        case .temporal(let t): return t.toArray().map { $0.map { String($0) } ?? "null" }
        case .decimal(let d): return d.toArray().map { $0.map { $0.description } ?? "null" }
        default: throw XCTSkip("no fingerprint for \(c.arrowFormat)")
        }
    }

    static func rows(_ b: MetalRecordBatch) throws -> [[String]] {
        try b.columns.first?.metalContext.flush()
        let cols = try b.columns.map { try fingerprint($0) }
        guard let n = cols.first?.count else { return [] }
        return (0..<n).map { i in cols.map { $0[i] } }.sorted { $0.lexicographicallyPrecedes($1) }
    }

    static func filters(_ json: [[Any]]) -> [ParquetFilter] {
        json.map { f in
            let op = ParquetFilter.Op(rawValue: f[1] as! String)!
            let v: ParquetFilter.Value
            if let s = f[2] as? String { v = .string(s) }
            else if let n = f[2] as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() { v = .int(n.boolValue ? 1 : 0) }
            else if let n = f[2] as? NSNumber, CFNumberIsFloatType(n) { v = .double(n.doubleValue) }
            else { v = .int((f[2] as! NSNumber).int64Value) }
            return ParquetFilter(column: f[0] as! String, op: op, value: v)
        }
    }

    // MARK: - every read in expected.json

    func testReadsMatchReferenceReaders() throws {
        let path = try fixture("expected.json")
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
        let cases = doc["cases"] as! [[String: Any]]
        XCTAssertGreaterThan(cases.count, 50)
        var failures: [String] = []
        for (i, c) in cases.enumerated() {
            let columns = c["columns"] as? [String]
            let flt = Self.filters(c["filters"] as? [[Any]] ?? [])
            let expected = c["expected"] as! [String: Any]
            let label = "case \(i) \(c["table"]!) version=\(c["version"] ?? "-") snapshot=\(c["snapshot_id"] ?? "-") "
                + "columns=\(columns ?? []) filters=\(c["filters"] ?? [])"
            do {
                let batch: MetalRecordBatch
                if c["kind"] as! String == "delta" {
                    let version = (c["version"] as? NSNumber)?.int64Value
                    batch = try DeltaTable(path: try fixture(c["table"] as! String))
                        .read(version: version, columns: columns, filters: flt)
                } else {
                    let snap = (c["snapshot_id"] as? NSNumber)?.int64Value
                    batch = try IcebergTable(path: try fixture(c["metadata"] as! String))
                        .read(snapshotId: snap, columns: columns, filters: flt)
                }
                let names = expected["names"] as! [String]
                if batch.names != names { failures.append("\(label): names \(batch.names) != \(names)"); continue }
                let want = (expected["rows"] as! [[String]]).sorted { $0.lexicographicallyPrecedes($1) }
                let got = try Self.rows(batch)
                if got != want {
                    let firstDiff = zip(got, want).first { $0 != $1 }
                    failures.append("\(label): \(got.count) rows vs \(want.count) expected; first difference \(String(describing: firstDiff))")
                }
            } catch {
                failures.append("\(label): \(error)")
            }
        }
        XCTAssert(failures.isEmpty, failures.joined(separator: "\n"))
    }

    // MARK: - Delta

    func testDeltaVersionsAndCheckpoint() throws {
        let t = try DeltaTable(path: try fixture("delta/basic"))
        XCTAssertEqual(try t.latestVersion(), 4)
        let s3 = try t.snapshot(version: 3)          // exactly the checkpoint version
        XCTAssertEqual(s3.version, 3)
        XCTAssertEqual(s3.protocolAction.minReaderVersion, 1)
        XCTAssertEqual(s3.schema.map { $0.name }, ["id", "region", "amount", "qty", "flag", "day", "ts", "price", "name"])
        XCTAssertThrowsError(try t.snapshot(version: 5)) { e in
            XCTAssert("\(e)".contains("version 5"), "\(e)")
        }
    }

    /// The same version read from the checkpoint and from a full replay of the JSON log agree.
    func testCheckpointAgreesWithLogReplay() throws {
        let src = try fixture("delta/partitioned")
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("am-lh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try FileManager.default.copyItem(atPath: src, toPath: tmp.path)
        let log = tmp.appendingPathComponent("_delta_log")
        for n in try FileManager.default.contentsOfDirectory(atPath: log.path) where n.contains("checkpoint") {
            try FileManager.default.removeItem(at: log.appendingPathComponent(n))
        }
        for v: Int64 in [2, 3, 4] {
            let a = try Self.rows(try DeltaTable(path: src).read(version: v))
            let b = try Self.rows(try DeltaTable(path: tmp.path).read(version: v))
            XCTAssertEqual(a, b, "version \(v)")
            XCTAssertFalse(a.isEmpty)
        }
    }

    /// With the commits before a checkpoint removed (log cleanup), the checkpoint and later versions
    /// still read and the earlier ones are reported as not reconstructable.
    func testLogCleanup() throws {
        let src = try fixture("delta/basic")
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("am-lh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try FileManager.default.copyItem(atPath: src, toPath: tmp.path)
        for v in 0...2 {
            try FileManager.default.removeItem(at: tmp.appendingPathComponent("_delta_log")
                .appendingPathComponent(String(format: "%020d.json", v)))
        }
        let t = try DeltaTable(path: tmp.path)
        XCTAssertEqual(try Self.rows(t.read()), try Self.rows(DeltaTable(path: src).read()))
        XCTAssertEqual(try Self.rows(t.read(version: 3)), try Self.rows(DeltaTable(path: src).read(version: 3)))
        XCTAssertThrowsError(try t.read(version: 1)) { e in
            XCTAssert("\(e)".contains("cannot be reconstructed"), "\(e)")
        }
    }

    func testDeltaRejectsUnsupportedFeaturesByName() throws {
        let cases = [("delta/unsupported_deletion_vectors", "deletionVectors"),
                     ("delta/unsupported_column_mapping_id", "columnMapping mode 'id'"),
                     ("delta/unsupported_unknown_feature", "someFutureFeature")]
        for (table, feature) in cases {
            XCTAssertThrowsError(try DeltaTable(path: try fixture(table)).read(), table) { e in
                guard case LakehouseError.unsupportedFeature(let m) = e else { return XCTFail("\(table): \(e)") }
                XCTAssert(m.contains(feature), "\(table): \(m)")
            }
        }
        XCTAssertThrowsError(try DeltaTable(path: Self.root.path)) { e in
            XCTAssert("\(e)".contains("_delta_log"), "\(e)")
        }
    }

    func testDeltaPruning() throws {
        let t = try DeltaTable(path: try fixture("delta/partitioned"))
        let all = try t.scan()
        let south = try t.scan(filters: [ParquetFilter(column: "region", op: .eq, value: .string("south"))])
        XCTAssertEqual(south.stats.filesTotal, all.stats.filesTotal)
        XCTAssertGreaterThan(south.stats.filesPrunedByPartition, 0)
        XCTAssertEqual(south.stats.filesPrunedByStatistics, 0)
        // Per-file min/max statistics prune a filter on a data column.
        let b = try DeltaTable(path: try fixture("delta/basic"))
        let high = try b.scan(filters: [ParquetFilter(column: "id", op: .ge, value: .int(50))])
        XCTAssertGreaterThan(high.stats.filesPrunedByStatistics, 0)
        XCTAssertEqual(high.batch.length, 10)
        // The pruned read returns the same rows as filtering everything.
        let none = try b.scan(filters: [ParquetFilter(column: "id", op: .gt, value: .int(1_000))])
        XCTAssertEqual(none.batch.length, 0)
        XCTAssertEqual(none.stats.filesRead, 0)
        XCTAssertEqual(none.batch.names.count, 9)
    }

    func testDeltaColumnMappingUsesPhysicalNames() throws {
        let s = try DeltaTable(path: try fixture("delta/column_mapping")).snapshot()
        XCTAssertEqual(s.columnMappingMode, "name")
        XCTAssertEqual(s.schema.map { $0.name }, ["id", "region", "total"])
        XCTAssertEqual(s.schema.map { $0.physicalName }, ["col-1a", "col-2b", "col-3c"])
        let v0 = try DeltaTable(path: try fixture("delta/column_mapping")).snapshot(version: 0)
        XCTAssertEqual(v0.schema.map { $0.name }, ["id", "region", "amount"])
    }

    func testDeltaUnknownColumnAndBadLiteral() throws {
        let t = try DeltaTable(path: try fixture("delta/basic"))
        XCTAssertThrowsError(try t.read(columns: ["nope"])) { e in XCTAssert("\(e)".contains("nope"), "\(e)") }
        XCTAssertThrowsError(try t.read(filters: [ParquetFilter(column: "id", op: .eq, value: .string("x"))])) { e in
            XCTAssert("\(e)".contains("does not fit column id"), "\(e)")
        }
    }

    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("am-lh-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func writeCommit(_ dir: URL, _ version: Int, _ actions: [[String: Any]]) throws {
        let log = dir.appendingPathComponent("_delta_log")
        try FileManager.default.createDirectory(at: log, withIntermediateDirectories: true)
        let lines = try actions.map { String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self) }
        try (lines.joined(separator: "\n") + "\n").write(to: log.appendingPathComponent(String(format: "%020d.json", version)),
                                                        atomically: true, encoding: .utf8)
    }

    /// Reader protocol 3 with only the features this reader implements reads normally.
    func testDeltaReaderFeaturesItImplements() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let t = tmp.appendingPathComponent("t")
        try FileManager.default.copyItem(atPath: try fixture("delta/basic"), toPath: t.path)
        try writeCommit(t, 5, [["protocol": ["minReaderVersion": 3, "minWriterVersion": 7,
                                             "readerFeatures": ["columnMapping", "timestampNtz", "vacuumProtocolCheck"],
                                             "writerFeatures": ["columnMapping", "timestampNtz"]]]])
        let s = try DeltaTable(path: t.path).snapshot()
        XCTAssertEqual(s.version, 5)
        XCTAssertEqual(s.protocolAction.minReaderVersion, 3)
        XCTAssertEqual(try Self.rows(DeltaTable(path: t.path).read()), try Self.rows(DeltaTable(path: try fixture("delta/basic")).read()))
    }

    func testNestedColumnsAndRemotePaths() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let schema = #"{"type":"struct","fields":[{"name":"id","type":"long","nullable":true,"metadata":{}},"#
            + #"{"name":"s","type":{"type":"struct","fields":[{"name":"a","type":"long","nullable":true,"metadata":{}}]},"nullable":true,"metadata":{}}]}"#
        let meta: [String: Any] = ["id": "x", "format": ["provider": "parquet", "options": [String: String]()],
                                   "schemaString": schema, "partitionColumns": [String](), "configuration": [String: String]()]
        try writeCommit(tmp, 0, [["protocol": ["minReaderVersion": 1, "minWriterVersion": 2]], ["metaData": meta]])
        let t = try DeltaTable(path: tmp.path)
        XCTAssertThrowsError(try t.read()) { e in
            guard case LakehouseError.unsupportedFeature(let m) = e else { return XCTFail("\(e)") }
            XCTAssert(m.contains("column s has nested type struct"), m)
        }
        XCTAssertEqual(try t.read(columns: ["id"]).length, 0)          // no files yet: an empty id column
        try writeCommit(tmp, 1, [["add": ["path": "s3://bucket/part-0.parquet", "partitionValues": [String: String](),
                                          "size": 10, "modificationTime": 0, "dataChange": true]]])
        XCTAssertThrowsError(try t.read(columns: ["id"])) { e in
            XCTAssert("\(e)".contains("s3:// storage"), "\(e)")
        }

        // Iceberg: a table with a struct column and no snapshot yet.
        let ice = tmp.appendingPathComponent("ice/metadata")
        try FileManager.default.createDirectory(at: ice, withIntermediateDirectories: true)
        let im: [String: Any] = [
            "format-version": 2, "table-uuid": "u", "location": "file:///elsewhere/ice", "last-column-id": 3,
            "current-schema-id": 0,
            "schemas": [["type": "struct", "schema-id": 0, "fields": [
                ["id": 1, "name": "id", "required": false, "type": "long"],
                ["id": 2, "name": "s", "required": false,
                 "type": ["type": "struct", "fields": [["id": 3, "name": "a", "required": false, "type": "int"]]]]]]],
            "partition-specs": [["spec-id": 0, "fields": [[String: Any]]()]], "default-spec-id": 0,
            "current-snapshot-id": -1, "snapshots": [[String: Any]]()]
        try JSONSerialization.data(withJSONObject: im).write(to: ice.appendingPathComponent("00000-a.metadata.json"))
        let it = try IcebergTable(path: tmp.appendingPathComponent("ice").path)
        XCTAssertNil(it.currentSnapshotId)
        XCTAssertThrowsError(try it.read()) { e in XCTAssert("\(e)".contains("column s has nested type struct"), "\(e)") }
        let empty = try it.read(columns: ["id"])
        XCTAssertEqual(empty.length, 0)
        XCTAssertEqual(empty.columns[0].arrowFormat, "l")

        // gzip-compressed metadata is refused by name.
        let gz = tmp.appendingPathComponent("gz/metadata")
        try FileManager.default.createDirectory(at: gz, withIntermediateDirectories: true)
        try Data([0x1F, 0x8B]).write(to: gz.appendingPathComponent("00000-a.gz.metadata.json"))
        XCTAssertThrowsError(try IcebergTable(path: tmp.appendingPathComponent("gz").path)) { e in
            XCTAssert("\(e)".contains("gzip-compressed"), "\(e)")
        }
    }

    // MARK: - Iceberg

    func testIcebergMetadataLocation() throws {
        let dir = try fixture("iceberg/v2_partitioned")
        let t = try IcebergTable(path: dir)
        XCTAssertEqual(t.formatVersion, 2)
        XCTAssertNotNil(t.currentSnapshotId)
        XCTAssertEqual(t.location, "iceberg/v2_partitioned")
        XCTAssertEqual(t.currentSchema.fields.map { $0.name },
                       ["id", "region", "total", "qty", "flag", "day", "ts", "price", "name", "note"])
        XCTAssertEqual(t.currentSchema.fields.first { $0.name == "qty" }?.type, .int64)
        // The newest metadata file is picked when no version hint exists.
        let names = try FileManager.default.contentsOfDirectory(atPath: dir + "/metadata")
            .filter { $0.hasSuffix(".metadata.json") }.sorted()
        XCTAssertEqual((t.metadataPath as NSString).lastPathComponent, names.last)

        let v1 = try IcebergTable(path: try fixture("iceberg/v1_plain"))
        XCTAssertEqual(v1.formatVersion, 1)

        // version-hint.text and vN.metadata.json naming (Hadoop tables).
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("am-lh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent("metadata"), withIntermediateDirectories: true)
        let first = dir + "/metadata/" + names.first!
        try FileManager.default.copyItem(atPath: first, toPath: tmp.appendingPathComponent("metadata/v1.metadata.json").path)
        try FileManager.default.copyItem(atPath: t.metadataPath, toPath: tmp.appendingPathComponent("metadata/v2.metadata.json").path)
        XCTAssert(try IcebergTable(path: tmp.path).metadataPath.hasSuffix("v2.metadata.json"))
        try "1\n".write(to: tmp.appendingPathComponent("metadata/version-hint.text"), atomically: true, encoding: .utf8)
        XCTAssert(try IcebergTable(path: tmp.path).metadataPath.hasSuffix("v1.metadata.json"))
    }

    func testIcebergSnapshotsAndSchemas() throws {
        let t = try IcebergTable(path: try fixture("iceberg/v2_partitioned"))
        let first = t.snapshots[0].snapshotId
        XCTAssertEqual(try t.schema(forSnapshot: first).fields.map { $0.name },
                       ["id", "region", "amount", "qty", "flag", "day", "ts", "price", "name"])
        XCTAssertEqual(try t.read(snapshotId: first).length, 30)
        XCTAssertThrowsError(try t.read(snapshotId: 12345)) { e in
            guard case LakehouseError.notFound(let m) = e else { return XCTFail("\(e)") }
            XCTAssert(m.contains("snapshot 12345"), m)
        }
    }

    func testIcebergRejectsDeleteFiles() throws {
        let t = try IcebergTable(path: try fixture("iceberg/position_deletes"))
        XCTAssertThrowsError(try t.read()) { e in
            guard case LakehouseError.unsupportedFeature(let m) = e else { return XCTFail("\(e)") }
            XCTAssert(m.contains("position delete files"), m)
        }
        // The snapshot before the delete file reads.
        XCTAssertEqual(try t.read(snapshotId: t.snapshots[0].snapshotId).length, 10)
    }

    func testIcebergPruning() throws {
        let t = try IcebergTable(path: try fixture("iceberg/v2_partitioned"))
        let south = try t.scan(filters: [ParquetFilter(column: "region", op: .eq, value: .string("south"))])
        XCTAssertGreaterThan(south.stats.filesPrunedByPartition, 0)
        XCTAssertGreaterThan(south.stats.manifestsTotal, 0)

        let tr = try IcebergTable(path: try fixture("iceberg/transforms"))
        // month(day): a February literal prunes every January file by the manifest summaries.
        let feb = try tr.scan(filters: [ParquetFilter(column: "day", op: .ge, value: .string("2024-02-01"))])
        XCTAssertEqual(feb.batch.length, 0)
        XCTAssertEqual(feb.stats.filesRead, 0)
        XCTAssertEqual(feb.stats.manifestsPruned, feb.stats.manifestsTotal)
        // truncate(25) on id: ids above 30 live only in the [25, 50) partitions.
        let hi = try tr.scan(filters: [ParquetFilter(column: "id", op: .gt, value: .int(30))])
        XCTAssertGreaterThan(hi.stats.filesPrunedByPartition, 0)
        XCTAssertEqual(hi.batch.length, 19)

        let byDay = try IcebergTable(path: try fixture("iceberg/by_day"))
        let early = try byDay.scan(filters: [ParquetFilter(column: "ts", op: .lt, value: .string("2024-01-02"))])
        XCTAssertGreaterThan(early.stats.filesPrunedByPartition, 0)
        XCTAssertEqual(early.batch.length, 4)
    }

    func testIcebergTransformProjection() {
        let ts = LakehouseField(name: "ts", type: .timestamp(nanos: false, utc: true), nullable: true, id: 1)
        let day = LakehouseField(name: "d", type: .date, nullable: true, id: 2)
        let s = LakehouseField(name: "s", type: .string, nullable: true, id: 3)
        let micros: Int64 = 86_400_000_000 * 3 + 5          // 1970-01-04 00:00:00.000005
        func pf(_ t: String) -> IcebergPartitionField { IcebergPartitionField(sourceId: 1, fieldId: 1000, name: "p", transform: t) }
        let f = LakeFilter(field: ts, op: .gt, literal: .int(micros))
        XCTAssertEqual(IcebergTable.project(f, pf("day"))?.0, .ge)
        XCTAssertEqual(IcebergTable.project(f, pf("day"))?.1, .int(3))
        XCTAssertEqual(IcebergTable.project(f, pf("hour"))?.1, .int(72))
        XCTAssertNil(IcebergTable.project(f, pf("bucket[4]")))       // bucket partitions are not pruned
        XCTAssertNil(IcebergTable.project(LakeFilter(field: ts, op: .ne, literal: .int(0)), pf("day")))
        let d = LakeFilter(field: day, op: .lt, literal: .int(LakeTime.days(year: 2024, month: 3, day: 15)))
        XCTAssertEqual(IcebergTable.project(d, pf("month"))?.1, .int(54 * 12 + 2))
        XCTAssertEqual(IcebergTable.project(d, pf("year"))?.1, .int(54))
        let neg = LakeFilter(field: ts, op: .eq, literal: .int(-1))
        XCTAssertEqual(IcebergTable.project(neg, pf("day"))?.1, .int(-1))
        let str = LakeFilter(field: s, op: .le, literal: .string("héllo"))
        XCTAssertEqual(IcebergTable.project(str, pf("truncate[2]"))?.1, .string("hé"))
        let i = LakeFilter(field: LakehouseField(name: "i", type: .int64, nullable: true, id: 4), op: .eq, literal: .int(-7))
        XCTAssertEqual(IcebergTable.project(i, pf("truncate[5]"))?.1, .int(-10))
    }

    func testIcebergBoundDecoding() {
        XCTAssertEqual(IcebergTable.decodeBound([0x2A, 0, 0, 0], .int32), .int(42))
        XCTAssertEqual(IcebergTable.decodeBound([0xFF, 0xFF, 0xFF, 0xFF], .int64), .int(-1))     // promoted int bound
        XCTAssertEqual(IcebergTable.decodeBound([0, 0, 0xC0, 0x3F], .float32), .double(1.5))
        XCTAssertEqual(IcebergTable.decodeBound(Array("abc".utf8), .string), .string("abc"))
        XCTAssertEqual(IcebergTable.decodeBound([0xFF, 0x38], .decimal(precision: 10, scale: 2)), .decimal(ArrowDecimal128(-200)))
        XCTAssertEqual(IcebergTable.decodeBound([0x01, 0x00], .decimal(precision: 10, scale: 2)), .decimal(ArrowDecimal128(256)))
    }

    // MARK: - Avro

    func testAvroCRC32AndSnappy() throws {
        XCTAssertEqual(AvroCodecs.crc32(Array("123456789".utf8)), 0xCBF43926)
        // "abcabcabcabcX": a literal "abc", a 1-byte-offset copy of 9 bytes, a literal "X".
        let snappy: [UInt8] = [13, 0x08, 0x61, 0x62, 0x63, 0x15, 0x03, 0x00, 0x58]
        XCTAssertEqual(try AvroCodecs.snappyDecompress(snappy), Array("abcabcabcabcX".utf8))
        XCTAssertThrowsError(try AvroCodecs.snappyDecompress([4, 0x05, 0x09, 0x00]))  // copy before any output
    }

    /// A hand-built object container file with the snappy codec: header, one block, sync marker.
    func testAvroSnappyContainer() throws {
        func zz(_ v: Int64) -> [UInt8] {
            var u = UInt64(bitPattern: (v << 1) ^ (v >> 63))
            var out: [UInt8] = []
            while u >= 0x80 { out.append(UInt8(u & 0x7F) | 0x80); u >>= 7 }
            out.append(UInt8(u))
            return out
        }
        func str(_ s: String) -> [UInt8] { zz(Int64(s.utf8.count)) + Array(s.utf8) }
        let schema = #"{"type":"record","name":"r","fields":[{"name":"a","type":"long"},{"name":"b","type":["null","string"]}]}"#
        let sync = [UInt8](repeating: 0xAB, count: 16)
        var file: [UInt8] = [0x4F, 0x62, 0x6A, 0x01]
        file += zz(2) + str("avro.schema") + str(schema) + str("avro.codec") + str("snappy") + zz(0)
        file += sync
        let records: [UInt8] = zz(7) + zz(1) + str("hi") + zz(-3) + zz(0)
        // Snappy: preamble, one literal holding everything.
        var compressed: [UInt8] = []
        var n = records.count
        while n >= 0x80 { compressed.append(UInt8(n & 0x7F) | 0x80); n >>= 7 }
        compressed.append(UInt8(n))
        compressed.append(UInt8((records.count - 1) << 2))
        compressed += records
        let crc = AvroCodecs.crc32(records)
        let block = compressed + [UInt8(crc >> 24), UInt8((crc >> 16) & 0xFF), UInt8((crc >> 8) & 0xFF), UInt8(crc & 0xFF)]
        file += zz(2) + zz(Int64(block.count)) + block + sync
        let avro = try AvroFile(bytes: file)
        XCTAssertEqual(avro.codec, "snappy")
        XCTAssertEqual(avro.records.count, 2)
        XCTAssertEqual(avro.records[0]["a"], .long(7))
        XCTAssertEqual(avro.records[0]["b"], .string("hi"))
        XCTAssertEqual(avro.records[1]["a"], .long(-3))
        XCTAssertEqual(avro.records[1]["b"], .null)
        // A corrupted checksum is caught.
        var bad = file
        bad[bad.count - 17] ^= 0xFF
        XCTAssertThrowsError(try AvroFile(bytes: bad))
    }

    func testAvroManifestCodecs() throws {
        // The fixtures carry deflate-compressed (v2) and uncompressed (v1) manifests.
        for (table, codec) in [("iceberg/v2_partitioned", "deflate"), ("iceberg/v1_plain", "null")] {
            let dir = try fixture(table + "/metadata")
            let avros = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".avro") }
            XCTAssertFalse(avros.isEmpty)
            for a in avros {
                let f = try AvroFile(path: dir + "/" + a)
                XCTAssertEqual(f.codec, codec, a)
                XCTAssertFalse(f.records.isEmpty, a)
            }
        }
    }

    // MARK: - literals and the row filter

    func testTimeLiterals() {
        XCTAssertEqual(LakeTime.parseDate("2024-01-01"), 19723)
        XCTAssertEqual(LakeTime.parseDate("1969-12-31"), -1)
        XCTAssertEqual(LakeTime.parseTimestamp("2024-01-01 00:00:00", unitsPerSecond: 1_000_000), 1_704_067_200_000_000)
        XCTAssertEqual(LakeTime.parseTimestamp("2024-01-01T01:00:00.5+01:00", unitsPerSecond: 1_000_000), 1_704_067_200_500_000)
        XCTAssertEqual(LakeTime.parseTimestamp("2024-01-01T00:00:00.000000001Z", unitsPerSecond: 1_000_000_000), 1_704_067_200_000_000_001)
        XCTAssertEqual(LakeTime.parseDecimal("-12.3", scale: 2), ArrowDecimal128(-1230))
        XCTAssertNil(LakeTime.parseDecimal("1.234", scale: 2))
        let c = LakeTime.civil(19723)
        XCTAssertEqual(c.year, 2024); XCTAssertEqual(c.month, 1); XCTAssertEqual(c.day, 1)
    }

    func testRowFilterSemantics() throws {
        let a = AnyMetalArray.int32(try MetalArray<Int32>([1, nil, 3, 4, Int32.max]))
        func sel(_ op: CompareOp, _ l: LakeScalar) throws -> [Bool?] {
            try LakeRowFilter.mask(a, op, l, column: "a").toArray()
        }
        XCTAssertEqual(try sel(.gt, .double(2.5)), [false, nil, true, true, true])
        XCTAssertEqual(try sel(.le, .double(3.5)), [true, nil, true, false, false])
        XCTAssertEqual(try sel(.eq, .double(3.5)), [false, nil, false, false, false])
        XCTAssertEqual(try sel(.ne, .double(3.5)), [true, nil, true, true, true])
        XCTAssertEqual(try sel(.lt, .int(1 << 40)), [true, nil, true, true, true])          // above int32
        XCTAssertEqual(try sel(.ge, .int(-(1 << 40))), [true, nil, true, true, true])
        XCTAssertEqual(try sel(.eq, .int(1 << 40)), [false, nil, false, false, false])
        let s = AnyMetalArray.string(try MetalStringArray(["b", "a", nil, "ab", "é"]))
        XCTAssertEqual(try LakeRowFilter.mask(s, .lt, .string("b"), column: "s").toArray(), [false, true, nil, true, false])
        XCTAssertEqual(try LakeRowFilter.mask(s, .ne, .string("a"), column: "s").toArray(), [true, false, nil, true, true])
        let d = AnyMetalArray.float64(try MetalArray<Double>([1.0, .nan, 3.0]))
        XCTAssertEqual(try LakeRowFilter.mask(d, .ne, .double(1.0), column: "d").toArray(), [false, true, true])
        XCTAssertEqual(try LakeRowFilter.mask(d, .gt, .int(0), column: "d").toArray(), [true, false, true])
    }

    func testConstantColumns() throws {
        let s = try LakeColumns.constant(.string, .string("xy"), length: 3, column: "c")
        XCTAssertEqual(s.asString?.toArray(), ["xy", "xy", "xy"])
        let n = try LakeColumns.constant(.int64, nil, length: 2, column: "c")
        XCTAssertEqual(n.asInt64?.toArray(), [nil, nil])
        let d = try LakeColumns.constant(.date, .int(19723), length: 2, column: "c")
        XCTAssertEqual(d.arrowFormat, "tdD")
        XCTAssertEqual(d.asTemporal?.toArray(), [19723, 19723])
        let dec = try LakeColumns.constant(.decimal(precision: 10, scale: 2), .decimal(ArrowDecimal128(-5)), length: 1, column: "c")
        XCTAssertEqual(dec.asDecimal?.toArray(), [ArrowDecimal128(-5)])
        XCTAssertThrowsError(try LakeColumns.constant(.int8, .int(300), length: 1, column: "c"))
    }
}
