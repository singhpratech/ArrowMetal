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

    // MARK: - review findings: literals and metadata that used to trap, and wrong-row cases

    /// Filter literals at the edges of their types are answers or errors, never a trap.
    func testFilterLiteralsAtTypeEdges() throws {
        // Doubles between Int64.max and the old 9.3e18 guard (and their negatives).
        XCTAssertEqual(LakeScalar.compare(.int(.max), .double(9.25e18)), -1)
        XCTAssertEqual(LakeScalar.compare(.int(.min), .double(-9.25e18)), 1)
        XCTAssertEqual(LakeScalar.compare(.int(.max), .double(0x1p63)), -1)
        XCTAssertEqual(LakeScalar.compare(.int(.min), .double(-0x1p63)), 0)
        let a = AnyMetalArray.int64(try MetalArray<Int64>([1, nil, .max, .min]))
        XCTAssertEqual(try LakeRowFilter.mask(a, .gt, .double(9.25e18), column: "a").toArray(), [false, nil, false, false])
        XCTAssertEqual(try LakeRowFilter.mask(a, .lt, .double(9.25e18), column: "a").toArray(), [true, nil, true, true])
        XCTAssertEqual(try LakeRowFilter.mask(a, .ge, .double(-9.25e18), column: "a").toArray(), [true, nil, true, true])
        XCTAssertEqual(try LakeRowFilter.mask(a, .gt, .double(9.2233720368547748e18), column: "a").toArray(), [false, nil, true, false])
        let i32 = AnyMetalArray.int32(try MetalArray<Int32>([1, 2]))
        XCTAssertEqual(try LakeRowFilter.mask(i32, .lt, .double(9.25e18), column: "q").toArray(), [true, true])
        let t = try DeltaTable(path: try fixture("delta/basic"))
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "id", op: .gt, value: .double(9.25e18))]).length, 0)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "id", op: .lt, value: .double(-9.25e18))]).length, 0)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "id", op: .lt, value: .double(9.25e18))]).length,
                       try t.read().length)
        // Huge years, Unicode numerics, too many digits: errors naming the column.
        XCTAssertNil(LakeTime.parseDate("100000000000000000-01-01"))
        XCTAssertNil(LakeTime.parseDate("5000001-01-01"))
        XCTAssertNil(LakeTime.parseDate("२०२४-01-01"))
        XCTAssertNil(LakeTime.parseTimestamp("2024-01-01 99999999999:00:00", unitsPerSecond: 1_000_000))
        XCTAssertNil(LakeTime.parseTimestamp("2024-01-01 00:00:00.１", unitsPerSecond: 1_000_000))
        XCTAssertNil(LakeTime.parseDecimal("½", scale: 2))
        XCTAssertNil(LakeTime.parseDecimal("१", scale: 2))
        XCTAssertNil(LakeTime.parseDecimal(String(repeating: "9", count: 200), scale: 2))
        XCTAssertNil(LakeTime.parseDecimal(String(repeating: "9", count: 37), scale: 2))     // 39 digits at scale 2
        var nines = ArrowDecimal128(0)
        for _ in 0..<38 { nines = nines * ArrowDecimal128(10) + ArrowDecimal128(9) }
        XCTAssertEqual(LakeTime.parseDecimal(String(repeating: "9", count: 36) + ".99", scale: 2), nines)
        XCTAssertEqual(LakeTime.parseDecimal("000000000000000000000000000000000000000001.5", scale: 1), ArrowDecimal128(15))
        for (column, literal) in [("day", "100000000000000000-01-01"), ("price", "½"), ("price", "१"),
                                  ("price", String(repeating: "9", count: 200)), ("ts", "2024-01-01 99999999999:00:00")] {
            XCTAssertThrowsError(try t.read(filters: [ParquetFilter(column: column, op: .gt, value: .string(literal))]), literal) { e in
                XCTAssert("\(e)".contains("does not fit column \(column)"), "\(e)")
            }
        }
        XCTAssertThrowsError(try t.read(filters: [ParquetFilter(column: "day", op: .gt, value: .int(1 << 40))])) { e in
            XCTAssert("\(e)".contains("does not fit column day"), "\(e)")
        }
        // Transform projection with literals at the ends of the range.
        let f = LakehouseField(name: "x", type: .int64, nullable: true, id: 1)
        let tr = IcebergPartitionField(sourceId: 1, fieldId: 1000, name: "x_trunc", transform: "truncate[10]")
        XCTAssertNil(IcebergTable.project(LakeFilter(field: f, op: .lt, literal: .int(.min)), tr))
        XCTAssertEqual(IcebergTable.project(LakeFilter(field: f, op: .lt, literal: .int(.max)), tr)?.1, .int(.max - 7))
        let ts = LakehouseField(name: "t", type: .timestamp(nanos: false, utc: true), nullable: true, id: 2)
        let hour = IcebergPartitionField(sourceId: 2, fieldId: 1001, name: "t_hour", transform: "hour")
        XCTAssertEqual(IcebergTable.project(LakeFilter(field: ts, op: .ge, literal: .int(.min)), hour)?.1,
                       .int(Int64.min / 3_600_000_000 - 1))
        let year = IcebergPartitionField(sourceId: 2, fieldId: 1002, name: "t_year", transform: "year")
        XCTAssertNotNil(IcebergTable.project(LakeFilter(field: ts, op: .ge, literal: .int(.max)), year))
        XCTAssertNotNil(IcebergTable.project(LakeFilter(field: ts, op: .le, literal: .int(.min)), year))
    }

    /// A float32 column compared with a double literal is the exact comparison (pyarrow widens the column).
    func testFloat32ComparesExactly() throws {
        let a = AnyMetalArray.float32(try MetalArray<Float>([0.1, 0.2, nil, .nan]))
        func sel(_ op: CompareOp, _ d: Double) throws -> [Bool?] { try LakeRowFilter.mask(a, op, .double(d), column: "f").toArray() }
        XCTAssertEqual(try sel(.eq, 0.1), [false, false, nil, false])          // 0.1f is 0.10000000149...
        XCTAssertEqual(try sel(.gt, 0.1), [true, true, nil, false])
        XCTAssertEqual(try sel(.le, 0.1), [false, false, nil, false])
        XCTAssertEqual(try sel(.ne, 0.1), [true, true, nil, true])
        XCTAssertEqual(try sel(.eq, Double(Float(0.1))), [true, false, nil, false])
        XCTAssertEqual(try sel(.lt, 1e300), [true, true, nil, false])
        XCTAssertEqual(try sel(.gt, -1e300), [true, true, nil, false])
        // Delta statistics written as the shortest text of the float round back to it.
        XCTAssertEqual(DeltaSnapshot.statScalar(NSNumber(value: 0.1), .float32), .double(Double(Float(0.1))))
    }

    /// Hand-crafted Avro containers that used to trap, recurse or loop are errors.
    func testAvroMalformedContainersAreErrors() throws {
        func zz(_ v: Int64) -> [UInt8] {
            var u = UInt64(bitPattern: (v << 1) ^ (v >> 63))
            var out: [UInt8] = []
            while u >= 0x80 { out.append(UInt8(u & 0x7F) | 0x80); u >>= 7 }
            out.append(UInt8(u))
            return out
        }
        func str(_ s: String) -> [UInt8] { zz(Int64(s.utf8.count)) + Array(s.utf8) }
        let sync = [UInt8](repeating: 0xAB, count: 16)
        func container(_ schema: String, _ block: [UInt8]) -> [UInt8] {
            [0x4F, 0x62, 0x6A, 0x01] + zz(2) + str("avro.schema") + str(schema) + str("avro.codec") + str("null")
                + zz(0) + sync + block + sync
        }
        let rec = #"{"type":"record","name":"r","fields":[{"name":"a","type":"long"}]}"#
        let cases: [(String, [UInt8], String)] = [
            ("block size near Int64.max", container(rec, zz(1) + zz(.max)), "runs past the end"),
            ("self-recursive record", container(#"{"type":"record","name":"r","fields":[{"name":"a","type":"r"}]}"#, zz(1) + zz(1) + [0]),
             "nest deeper"),
            ("huge block count, zero-width schema", container(#""null""#, zz(.max) + zz(0)), "records in 0 bytes"),
            ("array count Int64.min", container(#"{"type":"array","items":"long"}"#,
                                                zz(1) + zz(Int64(zz(.min).count + 1)) + zz(.min) + zz(0)),
             "array block count"),
        ]
        for (name, bytes, message) in cases {
            XCTAssertThrowsError(try AvroFile(bytes: bytes), name) { e in
                XCTAssert("\(e)".contains(message), "\(name): \(e)")
            }
        }
        // A recursive type through a union (a linked list) still decodes.
        let list = #"{"type":"record","name":"node","fields":[{"name":"v","type":"long"},{"name":"next","type":["null","node"]}]}"#
        let node = zz(2) + zz(1) + zz(3) + zz(0)          // v 2, next -> (v 3, next null)
        let f = try AvroFile(bytes: container(list, zz(1) + zz(Int64(node.count)) + node))
        XCTAssertEqual(f.records.count, 1)
        XCTAssertEqual(f.records[0]["next"]?["v"], .long(3))
    }

    /// A table whose data files are the `delta/basic` commit-0 file, partitioned by a string column.
    private func deltaWithPartitions(_ values: [Any], type: String = "string", extra: [String: Any] = [:]) throws -> URL {
        let tmp = try tempDir()
        let src = try fixture("delta/basic") + "/part-00000-adeae1f2-b155-4c85-a2ab-acee5b343388-c000.snappy.parquet"
        let base = try DeltaTable(path: try fixture("delta/basic")).snapshot(version: 0)
        var schema = try JSONSerialization.jsonObject(with: Data(base.metadata.schemaString.utf8)) as! [String: Any]
        var fields = schema["fields"] as! [[String: Any]]
        fields.append(["name": "grp", "type": type, "nullable": true, "metadata": [String: Any]()])
        schema["fields"] = fields
        let schemaString = String(decoding: try JSONSerialization.data(withJSONObject: schema), as: UTF8.self)
        var actions: [[String: Any]] = [
            ["protocol": ["minReaderVersion": 1, "minWriterVersion": 2]],
            ["metaData": ["id": "t", "format": ["provider": "parquet", "options": [String: Any]()],
                          "schemaString": schemaString, "partitionColumns": ["grp"], "configuration": [String: Any]()]],
        ]
        for (i, v) in values.enumerated() {
            let rel = "f\(i).parquet"
            try FileManager.default.copyItem(atPath: src, toPath: tmp.appendingPathComponent(rel).path)
            var add: [String: Any] = ["path": rel, "partitionValues": ["grp": v], "size": 1, "modificationTime": 0, "dataChange": true]
            for (k, x) in extra { add[k] = x }
            actions.append(["add": add])
        }
        try writeCommit(tmp, 0, actions)
        return tmp
    }

    /// The Delta protocol serializes a null partition value as the empty string, for strings too.
    func testDeltaEmptyStringPartitionIsNull() throws {
        XCTAssertNil(try DeltaTable.partitionScalar("", .string, column: "g"))
        XCTAssertNil(try DeltaTable.partitionScalar("", .binary, column: "g"))
        let dir = try deltaWithPartitions(["", "a", NSNull()])
        defer { try? FileManager.default.removeItem(at: dir) }
        let t = try DeltaTable(path: dir.path)
        let all = try t.read(columns: ["id", "grp"])
        XCTAssertEqual(all.length, 90)
        let grp = try XCTUnwrap(all.columns[1].asString).toArray()
        XCTAssertEqual(grp.filter { $0 == nil }.count, 60)
        XCTAssertEqual(grp.filter { $0 == "a" }.count, 30)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "grp", op: .eq, value: .string(""))]).length, 0)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "grp", op: .ne, value: .string("b"))]).length, 30)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "grp", op: .lt, value: .string("b"))]).length, 30)
    }

    /// `partitionValues` that is not an object of strings and nulls is malformed, not "no partitions".
    func testDeltaMalformedPartitionValues() throws {
        for bad in [["x"] as Any, 5 as Any] {
            let dir = try deltaWithPartitions(["a"], extra: ["partitionValues": bad])
            defer { try? FileManager.default.removeItem(at: dir) }
            XCTAssertThrowsError(try DeltaTable(path: dir.path).read()) { e in
                XCTAssert("\(e)".contains("partitionValues of f0.parquet"), "\(e)")
            }
        }
        let dir = try deltaWithPartitions([7])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try DeltaTable(path: dir.path).read()) { e in
            XCTAssert("\(e)".contains("is not a string or null"), "\(e)")
        }
    }

    /// Row-group pruning of a string filter uses the row filter's byte order. `nfd_strings.parquet`
    /// holds decomposed "é" strings, byte-wise below "f" and above it in Swift's `String` order.
    func testStringRowGroupPruningIsByteWise() throws {
        let path = try fixture("parquet/nfd_strings.parquet")
        let file = try ParquetFile(path: path)
        XCTAssertEqual(file.metadata.rowGroups.count, 3)
        let s = LakehouseField(name: "s", type: .string, nullable: true)
        func groups(_ op: CompareOp, _ lit: String) -> [Int]? {
            LakeDataFile.rowGroupPlan(file, [(column: "s", filter: LakeFilter(field: s, op: op, literal: .string(lit)))]).rowGroups
        }
        XCTAssertEqual(groups(.lt, "f"), [0, 1])
        XCTAssertEqual(groups(.gt, "f"), [2])
        XCTAssertEqual(groups(.eq, "e\u{301} 25"), [1])
        XCTAssertEqual(LakeDataFile.rowGroupPlan(file, [(column: "s", filter: LakeFilter(field: s, op: .lt, literal: .string("f")))]).parquet.count, 0)
        // Through a Delta table over the same file: every decomposed row survives `s < "f"`.
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        try FileManager.default.copyItem(atPath: path, toPath: tmp.appendingPathComponent("p.parquet").path)
        let schema = #"{"type":"struct","fields":[{"name":"id","type":"long","nullable":true,"metadata":{}},{"name":"s","type":"string","nullable":true,"metadata":{}}]}"#
        try writeCommit(tmp, 0, [["protocol": ["minReaderVersion": 1, "minWriterVersion": 2]],
                                 ["metaData": ["id": "t", "format": ["provider": "parquet"], "schemaString": schema,
                                               "partitionColumns": [String](), "configuration": [String: Any]()]],
                                 ["add": ["path": "p.parquet", "partitionValues": [String: Any](), "size": 1,
                                          "modificationTime": 0, "dataChange": true]]])
        let t = try DeltaTable(path: tmp.path)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "s", op: .lt, value: .string("f"))]).length, 40)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "s", op: .le, value: .string("f"))]).length, 40)
        XCTAssertEqual(try t.read(filters: [ParquetFilter(column: "s", op: .gt, value: .string("f"))]).length, 20)
    }

    /// Iceberg paths are used as written first: pyiceberg puts `grp=x%3Dy` on disk literally.
    func testIcebergPathsAreNotPercentDecodedFirst() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let t = tmp.appendingPathComponent("t")
        try FileManager.default.copyItem(atPath: try fixture("iceberg/v1_plain"), toPath: t.path)
        let table = try IcebergTable(path: t.path)
        let loc = table.location.hasSuffix("/") ? String(table.location.dropLast()) : table.location
        for name in ["grp=x%3Dy", "grp=%C3%BC", "plain"] {
            let dir = t.appendingPathComponent("data").appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: dir.appendingPathComponent("f.parquet").path, contents: Data([1]))
            XCTAssertEqual(try table.resolve(loc + "/data/" + name + "/f.parquet"),
                           t.path + "/data/" + name + "/f.parquet")
        }
        // A writer that recorded an encoded URI for a decoded directory still resolves.
        let decoded = t.appendingPathComponent("data").appendingPathComponent("a b")
        try FileManager.default.createDirectory(at: decoded, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: decoded.appendingPathComponent("f.parquet").path, contents: Data([1]))
        XCTAssertEqual(try table.resolve(loc + "/data/a%20b/f.parquet"), t.path + "/data/a b/f.parquet")
    }

    /// A NaN partition value is unequal to every literal: kept for `!=`, pruned for the other comparisons.
    func testDeltaNaNPartitionMatchesNotEqual() throws {
        let dir = try deltaWithPartitions(["1.0", "NaN", "2.0"], type: "double")
        defer { try? FileManager.default.removeItem(at: dir) }
        let t = try DeltaTable(path: dir.path)
        func count(_ op: ParquetFilter.Op, _ v: Double) throws -> Int {
            try t.read(filters: [ParquetFilter(column: "grp", op: op, value: .double(v))]).length
        }
        XCTAssertEqual(try count(.ne, 1.0), 60)
        XCTAssertEqual(try count(.eq, 1.0), 30)
        XCTAssertEqual(try count(.lt, 5.0), 60)
        XCTAssertEqual(try count(.ge, 1.0), 60)
        XCTAssertEqual(try count(.gt, 1.0), 30)
        XCTAssertEqual(try count(.ne, .nan), 90)
        XCTAssertEqual(try count(.eq, .nan), 0)
        let scan = try t.scan(filters: [ParquetFilter(column: "grp", op: .ne, value: .double(1.0))])
        XCTAssertEqual(scan.stats.filesPrunedByPartition, 1)
    }

    /// Reader protocol 3 requires `readerFeatures` as a list of strings; anything else is malformed, so
    /// the feature check cannot be skipped by a list written another way.
    func testDeltaReaderFeaturesMustBeAListOfStrings() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let bad: [(Any?, String)] = [("deletionVectors", "is not a list of strings"), ([1, 2], "is not a list of strings"),
                                     (["a": 1], "is not a list of strings"), (nil, "without protocol.readerFeatures"),
                                     (NSNull(), "without protocol.readerFeatures")]
        for (i, (features, message)) in bad.enumerated() {
            let t = tmp.appendingPathComponent("t\(i)")
            try FileManager.default.copyItem(atPath: try fixture("delta/basic"), toPath: t.path)
            var p: [String: Any] = ["minReaderVersion": 3, "minWriterVersion": 7]
            if let features { p["readerFeatures"] = features }
            try writeCommit(t, 5, [["protocol": p]])
            XCTAssertThrowsError(try DeltaTable(path: t.path).read(), "\(String(describing: features))") { e in
                guard case LakehouseError.malformed(let m) = e else { return XCTFail("\(e)") }
                XCTAssert(m.contains(message), m)
            }
        }
        // A reader protocol below 3 needs no list.
        let t = tmp.appendingPathComponent("v1")
        try FileManager.default.copyItem(atPath: try fixture("delta/basic"), toPath: t.path)
        try writeCommit(t, 5, [["protocol": ["minReaderVersion": 1, "minWriterVersion": 2]]])
        XCTAssertEqual(try DeltaTable(path: t.path).read().length, 50)
    }

    /// A snapshot with neither a manifest list nor a manifests list is malformed, not an empty table.
    func testIcebergSnapshotWithoutManifestsIsAnError() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let t = tmp.appendingPathComponent("t")
        try FileManager.default.copyItem(atPath: try fixture("iceberg/v2_partitioned"), toPath: t.path)
        let before = try IcebergTable(path: t.path).read().length
        XCTAssertGreaterThan(before, 0)
        let metaDir = t.appendingPathComponent("metadata")
        let newest = try FileManager.default.contentsOfDirectory(atPath: metaDir.path)
            .filter { $0.hasSuffix(".metadata.json") }.sorted().last!
        let url = metaDir.appendingPathComponent(newest)
        var meta = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        meta["snapshots"] = (meta["snapshots"] as! [[String: Any]]).map { s in
            var s = s
            s["manifest-list"] = nil
            return s
        }
        try JSONSerialization.data(withJSONObject: meta).write(to: url)
        XCTAssertThrowsError(try IcebergTable(path: url.path).read()) { e in
            guard case LakehouseError.malformed(let m) = e else { return XCTFail("\(e)") }
            XCTAssert(m.contains("neither a manifest-list nor manifests") && m.contains(newest), m)
        }
    }

    /// A data file that holds none of the table's columns is not read as rows of nulls: in Delta without
    /// column mapping (where no column can be dropped), and in Iceberg for a file without field ids in a
    /// table without a name mapping (pyiceberg refuses it too).
    func testDataFilesWithoutTheTableColumnsAreErrors() throws {
        let tmp = try tempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }
        let d = tmp.appendingPathComponent("d")
        try FileManager.default.copyItem(atPath: try fixture("delta/basic"), toPath: d.path)
        let fm = FileManager.default
        let checkpoint = try fm.contentsOfDirectory(atPath: d.appendingPathComponent("_delta_log").path)
            .first { $0.hasSuffix(".checkpoint.parquet") }!
        for f in try fm.contentsOfDirectory(atPath: d.path) where f.hasSuffix(".parquet") {
            try fm.removeItem(at: d.appendingPathComponent(f))
            try fm.copyItem(at: d.appendingPathComponent("_delta_log").appendingPathComponent(checkpoint),
                            to: d.appendingPathComponent(f))
        }
        XCTAssertThrowsError(try DeltaTable(path: d.path).read()) { e in
            XCTAssert("\(e)".contains("holds none of the table's columns"), "\(e)")
        }
        let ice = tmp.appendingPathComponent("i")
        try fm.copyItem(atPath: try fixture("iceberg/v2_partitioned"), toPath: ice.path)
        let noIds = try fixture("parquet/nfd_strings.parquet")
        let data = fm.enumerator(atPath: ice.appendingPathComponent("data").path)!.compactMap { $0 as? String }
            .filter { $0.hasSuffix(".parquet") }
        XCTAssertFalse(data.isEmpty)
        for f in data {
            let u = ice.appendingPathComponent("data").appendingPathComponent(f)
            try fm.removeItem(at: u)
            try fm.copyItem(atPath: noIds, toPath: u.path)
        }
        XCTAssertThrowsError(try IcebergTable(path: ice.path).read()) { e in
            XCTAssert("\(e)".contains("has no field ids"), "\(e)")
        }
    }
}
