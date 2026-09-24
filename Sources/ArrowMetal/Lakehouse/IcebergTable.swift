import Foundation

// Apache Iceberg tables.
//
// An Iceberg table is described by a metadata JSON file (`metadata/NNNNN-<uuid>.metadata.json` or
// `metadata/vN.metadata.json`): its schemas (every column carries a field id that never changes, so a
// renamed column keeps its id), its partition specs, and its snapshots. A snapshot points at a manifest
// list (Avro) naming manifests (Avro), and each manifest lists data files with their partition values
// and per-column statistics (value, null and NaN counts, lower and upper bounds), keyed by field id.
//
// This reader parses the metadata on the CPU (the Avro files with `AvroFile`), prunes manifests by the
// partition summaries in the manifest list and data files by their partition values and column bounds,
// then reads the surviving Parquet files with the GPU Parquet reader, matching columns by field id.
//
// Supported: format versions 1 and 2, the current or any listed snapshot, identity / year / month / day
// / hour / truncate partition transforms for pruning (bucket and void partitions are read, just not
// pruned), schema evolution by field id (renamed columns, added columns read as null from older files,
// int -> long and float -> double promotion). Delete files (position and equality deletes, format v2)
// are rejected with an error naming them, as are data files that are not Parquet.

public struct IcebergPartitionField: Sendable, Equatable {
    public let sourceId: Int
    public let fieldId: Int
    public let name: String
    public let transform: String
}

public struct IcebergPartitionSpec: Sendable, Equatable {
    public let specId: Int
    public let fields: [IcebergPartitionField]
}

public struct IcebergSchema: Sendable, Equatable {
    public let schemaId: Int
    public let fields: [LakehouseField]
}

public struct IcebergSnapshot: Sendable, Equatable {
    public let snapshotId: Int64
    public let parentId: Int64?
    public let sequenceNumber: Int64
    public let timestampMs: Int64
    public let manifestList: String?
    /// Format v1 snapshots may list manifests directly instead of through a manifest list.
    public let manifests: [String]?
    public let schemaId: Int?
    public let operation: String?
}

/// One live data file of a snapshot, as its manifest describes it.
public struct IcebergDataFile: Sendable {
    public let path: String
    public let format: String
    public let specId: Int
    public let recordCount: Int64
    public let fileSizeInBytes: Int64
    /// Partition values in the order of the spec's fields (nil for a null value).
    let partition: [LakeScalar?]
    let lowerBounds: [Int: [UInt8]]
    let upperBounds: [Int: [UInt8]]
    let nullCounts: [Int: Int64]
    let valueCounts: [Int: Int64]
    let nanCounts: [Int: Int64]
}

public final class IcebergTable: @unchecked Sendable {
    public let metadataPath: String
    /// The directory the table's paths are resolved against (the parent of `metadata/`).
    public let tableRoot: String
    /// The `location` the metadata records.
    public let location: String
    public let formatVersion: Int
    public let schemas: [IcebergSchema]
    public let currentSchemaId: Int
    public let specs: [IcebergPartitionSpec]
    public let currentSnapshotId: Int64?
    public let snapshots: [IcebergSnapshot]
    public let properties: [String: String]
    public let context: MetalContext

    /// `path` is a `*.metadata.json` file, a table directory (holding `metadata/`), or the metadata
    /// directory itself. For a directory, `version-hint.text` is honoured when present; otherwise the
    /// metadata file with the highest version number is used.
    public init(path: String, context: MetalContext = .shared) throws {
        self.context = context
        let local = try LakePath.local(path)
        let metaFile = try Self.locateMetadata(local)
        metadataPath = metaFile
        let metaDir = (metaFile as NSString).deletingLastPathComponent
        tableRoot = (metaDir as NSString).lastPathComponent == "metadata" ? (metaDir as NSString).deletingLastPathComponent : metaDir
        guard let data = FileManager.default.contents(atPath: metaFile),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LakehouseError.malformed("\(metaFile) is not a JSON object")
        }
        guard let fv = (obj["format-version"] as? NSNumber)?.intValue else {
            throw LakehouseError.malformed("\(metaFile) has no format-version")
        }
        guard fv == 1 || fv == 2 else {
            throw LakehouseError.unsupportedFeature("Iceberg format version \(fv) (\(metaFile)); versions 1 and 2 are read")
        }
        formatVersion = fv
        location = (obj["location"] as? String) ?? tableRoot
        var props: [String: String] = [:]
        if let p = obj["properties"] as? [String: Any] { for (k, v) in p { if let s = v as? String { props[k] = s } } }
        properties = props

        var schemas: [IcebergSchema] = []
        if let list = obj["schemas"] as? [[String: Any]] {
            for s in list { schemas.append(try Self.parseSchema(s, file: metaFile)) }
        }
        if let single = obj["schema"] as? [String: Any] {
            let s = try Self.parseSchema(single, file: metaFile)
            if !schemas.contains(where: { $0.schemaId == s.schemaId }) { schemas.append(s) }
        }
        guard !schemas.isEmpty else { throw LakehouseError.malformed("\(metaFile) has no schema") }
        self.schemas = schemas
        currentSchemaId = (obj["current-schema-id"] as? NSNumber)?.intValue ?? schemas.last!.schemaId

        var specs: [IcebergPartitionSpec] = []
        if let list = obj["partition-specs"] as? [[String: Any]] {
            for s in list {
                specs.append(IcebergPartitionSpec(specId: (s["spec-id"] as? NSNumber)?.intValue ?? 0,
                                                  fields: try Self.parseSpecFields(s["fields"] as? [[String: Any]] ?? [], file: metaFile)))
            }
        } else if let v1 = obj["partition-spec"] as? [[String: Any]] {
            specs.append(IcebergPartitionSpec(specId: 0, fields: try Self.parseSpecFields(v1, file: metaFile)))
        }
        self.specs = specs

        let cur = (obj["current-snapshot-id"] as? NSNumber)?.int64Value
        currentSnapshotId = (cur == nil || cur == -1) ? nil : cur
        var snaps: [IcebergSnapshot] = []
        for s in obj["snapshots"] as? [[String: Any]] ?? [] {
            guard let id = (s["snapshot-id"] as? NSNumber)?.int64Value else {
                throw LakehouseError.malformed("a snapshot in \(metaFile) has no snapshot-id")
            }
            snaps.append(IcebergSnapshot(snapshotId: id, parentId: (s["parent-snapshot-id"] as? NSNumber)?.int64Value,
                                         sequenceNumber: (s["sequence-number"] as? NSNumber)?.int64Value ?? 0,
                                         timestampMs: (s["timestamp-ms"] as? NSNumber)?.int64Value ?? 0,
                                         manifestList: s["manifest-list"] as? String,
                                         manifests: s["manifests"] as? [String],
                                         schemaId: (s["schema-id"] as? NSNumber)?.intValue,
                                         operation: (s["summary"] as? [String: Any])?["operation"] as? String))
        }
        snapshots = snaps
    }

    // MARK: Locating the metadata file

    static func locateMetadata(_ path: String) throws -> String {
        if !LakePath.isDirectory(path) {
            guard LakePath.exists(path) else { throw LakehouseError.notATable("\(path) does not exist") }
            if path.hasSuffix(".gz") || path.hasSuffix(".gz.metadata.json") {
                throw LakehouseError.unsupportedFeature("gzip-compressed metadata file \(path)")
            }
            return path
        }
        let metaDir = LakePath.isDirectory(LakePath.join(path, "metadata")) ? LakePath.join(path, "metadata") : path
        let names = (try? FileManager.default.contentsOfDirectory(atPath: metaDir)) ?? []
        let hint = LakePath.join(metaDir, "version-hint.text")
        if let h = FileManager.default.contents(atPath: hint),
           let n = Int(String(decoding: h, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)) {
            if names.contains("v\(n).metadata.json") { return LakePath.join(metaDir, "v\(n).metadata.json") }
            let prefix = String(format: "%05d-", n)
            if let m = names.first(where: { $0.hasPrefix(prefix) && $0.hasSuffix(".metadata.json") }) {
                return LakePath.join(metaDir, m)
            }
        }
        var best: (Int, String)? = nil
        for n in names where n.hasSuffix(".metadata.json") && !n.hasSuffix(".gz.metadata.json") {
            var v: Int? = nil
            if n.hasPrefix("v") { v = Int(n.dropFirst().prefix { $0.isASCIIDigit }) }
            else if let dash = n.firstIndex(of: "-") { v = Int(n[..<dash]) }
            guard let v else { continue }
            if best == nil || v > best!.0 { best = (v, n) }
        }
        guard let best else {
            if names.contains(where: { $0.hasSuffix(".gz.metadata.json") || $0.hasSuffix(".metadata.json.gz") }) {
                throw LakehouseError.unsupportedFeature("gzip-compressed metadata files in \(metaDir)")
            }
            throw LakehouseError.notATable("\(path) holds no Iceberg *.metadata.json file")
        }
        return LakePath.join(metaDir, best.1)
    }

    // MARK: Schema and specs

    static func parseSchema(_ s: [String: Any], file: String) throws -> IcebergSchema {
        guard let fields = s["fields"] as? [[String: Any]] else {
            throw LakehouseError.malformed("a schema in \(file) has no fields")
        }
        return IcebergSchema(schemaId: (s["schema-id"] as? NSNumber)?.intValue ?? 0, fields: try fields.map { f in
            guard let id = (f["id"] as? NSNumber)?.intValue, let name = f["name"] as? String else {
                throw LakehouseError.malformed("a schema field in \(file) has no id or name")
            }
            return LakehouseField(name: name, type: parseType(f["type"] as Any), nullable: !((f["required"] as? Bool) ?? false), id: id)
        })
    }

    static func parseType(_ t: Any) -> LakehouseType {
        if let s = t as? String {
            switch s {
            case "boolean": return .boolean
            case "int": return .int32
            case "long": return .int64
            case "float": return .float32
            case "double": return .float64
            case "date": return .date
            case "time": return .time
            case "timestamp": return .timestamp(nanos: false, utc: false)
            case "timestamptz": return .timestamp(nanos: false, utc: true)
            case "timestamp_ns": return .timestamp(nanos: true, utc: false)
            case "timestamptz_ns": return .timestamp(nanos: true, utc: true)
            case "string": return .string
            case "uuid": return .uuid
            case "binary": return .binary
            default:
                if s.hasPrefix("decimal("), s.hasSuffix(")") {
                    let inner = s.dropFirst(8).dropLast().split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                    if inner.count == 2, let p = Int(inner[0]), let sc = Int(inner[1]) { return .decimal(precision: p, scale: sc) }
                }
                if s.hasPrefix("fixed["), s.hasSuffix("]"), let n = Int(s.dropFirst(6).dropLast()) { return .fixed(n) }
                return .nested(s)
            }
        }
        if let d = t as? [String: Any], let kind = d["type"] as? String { return .nested(kind) }
        return .nested("unknown")
    }

    static func parseSpecFields(_ list: [[String: Any]], file: String) throws -> [IcebergPartitionField] {
        var out: [IcebergPartitionField] = []
        for (i, f) in list.enumerated() {
            guard let src = (f["source-id"] as? NSNumber)?.intValue, let tr = f["transform"] as? String else {
                throw LakehouseError.malformed("a partition field in \(file) has no source-id or transform")
            }
            out.append(IcebergPartitionField(sourceId: src, fieldId: (f["field-id"] as? NSNumber)?.intValue ?? (1000 + i),
                                             name: (f["name"] as? String) ?? "p\(i)", transform: tr))
        }
        return out
    }

    public var currentSchema: IcebergSchema {
        schemas.first { $0.schemaId == currentSchemaId } ?? schemas.last!
    }

    /// The schema a scan of `snapshotId` projects: the snapshot's own schema when time-travelling to a
    /// named snapshot, the current schema otherwise (as pyiceberg and the Java reference do).
    public func schema(forSnapshot snapshotId: Int64?) throws -> IcebergSchema {
        guard let snapshotId else { return currentSchema }
        let snap = try snapshot(snapshotId)
        if let sid = snap.schemaId, let s = schemas.first(where: { $0.schemaId == sid }) { return s }
        return currentSchema
    }

    public func snapshot(_ id: Int64) throws -> IcebergSnapshot {
        guard let s = snapshots.first(where: { $0.snapshotId == id }) else {
            throw LakehouseError.notFound("snapshot \(id) of the Iceberg table \(metadataPath) (snapshots: "
                                          + snapshots.map { String($0.snapshotId) }.joined(separator: ", ") + ")")
        }
        return s
    }

    // MARK: Paths

    /// A path from the metadata, resolved to a local file. Paths under the table's recorded `location`
    /// are re-rooted at the directory the metadata was actually found in, so a table that was copied or
    /// moved still reads.
    func resolve(_ p: String) throws -> String {
        func strip(_ s: String) -> String {
            if s.hasPrefix("file://") {
                let r = String(s.dropFirst(7))
                return r.hasPrefix("/") ? r : (r.firstIndex(of: "/").map { String(r[$0...]) } ?? r)
            }
            if s.hasPrefix("file:") { return String(s.dropFirst(5)) }
            return s
        }
        var loc = strip(location)
        while loc.hasSuffix("/") && loc.count > 1 { loc.removeLast() }
        let path = strip(p)
        if !loc.isEmpty, path.hasPrefix(loc + "/") {
            // Iceberg paths are locations, not URIs: a writer that escapes a partition value
            // (pyiceberg writes `grp=x%3Dy`) puts that text on disk. The path is used as written; the
            // percent-decoded form is only a fallback for writers that recorded an encoded URI.
            let rel = String(path.dropFirst(loc.count + 1))
            let asWritten = LakePath.join(tableRoot, rel)
            if LakePath.exists(asWritten) { return asWritten }
            if let decoded = rel.removingPercentEncoding, decoded != rel, LakePath.exists(LakePath.join(tableRoot, decoded)) {
                return LakePath.join(tableRoot, decoded)
            }
            return asWritten
        }
        if p.hasPrefix("/") || !p.contains(":") { return p }
        let local = try LakePath.local(p)
        if LakePath.exists(local) { return local }
        let raw = strip(p)
        return LakePath.exists(raw) ? raw : local
    }

    // MARK: Planning

    /// The data files of a snapshot that may hold rows matching every filter, the schema the scan
    /// projects, and how many manifests and files were pruned.
    public func plan(snapshotId: Int64? = nil, filters: [ParquetFilter] = []) throws
        -> (files: [IcebergDataFile], schema: IcebergSchema, stats: LakehouseScanStats) {
        let schema = try schema(forSnapshot: snapshotId)
        let resolved = try lakeResolveFilters(filters, schema: schema.fields, table: metadataPath)
        let (files, stats) = try plan(snapshotId: snapshotId, schema: schema, filters: resolved)
        return (files, schema, stats)
    }

    private struct ManifestRef {
        let path: String
        let specId: Int
        let content: Int
        let partitions: [AvroValue]?
    }

    func plan(snapshotId: Int64?, schema: IcebergSchema, filters: [LakeFilter]) throws
        -> ([IcebergDataFile], LakehouseScanStats) {
        var stats = LakehouseScanStats()
        guard let sid = snapshotId ?? currentSnapshotId else { return ([], stats) }
        let snap = try snapshot(sid)
        var manifests: [ManifestRef] = []
        if let ml = snap.manifestList {
            let file = try resolve(ml)
            let avro: AvroFile
            do { avro = try AvroFile(path: file) } catch { throw LakehouseError.malformed("manifest list \(file): \(error)") }
            for r in avro.records {
                guard let p = r["manifest_path"]?.string else { throw LakehouseError.malformed("manifest list \(file) entry without manifest_path") }
                manifests.append(ManifestRef(path: p, specId: Int(r["partition_spec_id"]?.int64 ?? 0),
                                             content: Int(r["content"]?.int64 ?? 0), partitions: r["partitions"]?.array))
            }
        } else if let list = snap.manifests {
            manifests = list.map { ManifestRef(path: $0, specId: -1, content: 0, partitions: nil) }
        }

        // Delete files change which rows are live; reading around them would return deleted rows.
        for m in manifests where m.content == 1 {
            let entries = try readManifest(m.path, specHint: m.specId >= 0 ? m.specId : nil)
            let live = entries.filter { $0.status != 2 }
            if !live.isEmpty {
                let kinds = Set(live.map { $0.content == 2 ? "equality" : "position" }).sorted()
                throw LakehouseError.unsupportedFeature(
                    "Iceberg \(kinds.joined(separator: " and ")) delete files (snapshot \(sid) of \(metadataPath) has "
                    + "\(live.count) live delete file\(live.count == 1 ? "" : "s"), e.g. \(live[0].file.path))")
            }
        }

        var out: [IcebergDataFile] = []
        for m in manifests where m.content == 0 {
            stats.manifestsTotal += 1
            let entries = try readManifest(m.path, specHint: m.specId >= 0 ? m.specId : nil)
            let specId = m.specId >= 0 ? m.specId : (entries.first?.file.specId ?? 0)
            if let parts = m.partitions, let spec = specs.first(where: { $0.specId == specId }),
               !manifestMayMatch(parts, spec: spec, schema: schema, filters: filters) {
                stats.manifestsPruned += 1
                let skipped = entries.filter { $0.status != 2 && $0.file.content == 0 }.count
                stats.filesTotal += skipped
                stats.filesPrunedByPartition += skipped
                continue
            }
            for e in entries where e.status != 2 {
                if e.file.content != 0 {
                    throw LakehouseError.unsupportedFeature("Iceberg delete file \(e.file.path) listed in a data manifest")
                }
                stats.filesTotal += 1
                let f = e.file
                guard f.format.uppercased() == "PARQUET" else {
                    throw LakehouseError.unsupportedFeature("\(f.format) data file \(f.path); only Parquet data files are read")
                }
                let spec = specs.first { $0.specId == f.specId }
                if let spec, !partitionMayMatch(f.partition, spec: spec, schema: schema, filters: filters) {
                    stats.filesPrunedByPartition += 1
                    continue
                }
                if !boundsMayMatch(f, filters: filters) {
                    stats.filesPrunedByStatistics += 1
                    continue
                }
                out.append(f.asPublic)
            }
        }
        return (out, stats)
    }

    private struct ManifestEntry {
        let status: Int
        let content: Int
        let file: RawDataFile
    }

    private struct RawDataFile {
        let content: Int
        let path: String
        let format: String
        let specId: Int
        let partition: [LakeScalar?]
        let recordCount: Int64
        let size: Int64
        let lower: [Int: [UInt8]], upper: [Int: [UInt8]]
        let nulls: [Int: Int64], values: [Int: Int64], nans: [Int: Int64]

        var asPublic: IcebergDataFile {
            IcebergDataFile(path: path, format: format, specId: specId, recordCount: recordCount,
                            fileSizeInBytes: size, partition: partition, lowerBounds: lower, upperBounds: upper,
                            nullCounts: nulls, valueCounts: values, nanCounts: nans)
        }
    }

    private func readManifest(_ p: String, specHint: Int?) throws -> [ManifestEntry] {
        let file = try resolve(p)
        let avro: AvroFile
        do { avro = try AvroFile(path: file) } catch { throw LakehouseError.malformed("manifest \(file): \(error)") }
        // The spec a manifest was written with is in its header (v2); v1 manifests may omit it.
        let specId = Int(avro.metadata["partition-spec-id"] ?? "") ?? specHint ?? specs.first?.specId ?? 0
        let spec = specs.first { $0.specId == specId }
        let sourceTypes = currentSchemaTypesById()
        return try avro.records.map { r in
            guard let df = r["data_file"]?.record else { throw LakehouseError.malformed("manifest \(file) entry without data_file") }
            guard let path = df["file_path"]?.string else { throw LakehouseError.malformed("manifest \(file) entry without file_path") }
            var partition: [LakeScalar?] = []
            if let spec, let pr = df["partition"]?.record {
                for (i, pf) in spec.fields.enumerated() {
                    let v = pr[pf.name] ?? (i < pr.values.count ? pr.values[i] : .null)
                    partition.append(Self.avroScalar(v, Self.resultType(pf, sourceTypes[pf.sourceId])))
                }
            }
            return ManifestEntry(status: Int(r["status"]?.int64 ?? 1), content: Int(df["content"]?.int64 ?? 0),
                                 file: RawDataFile(content: Int(df["content"]?.int64 ?? 0), path: path,
                                                   format: df["file_format"]?.string ?? "PARQUET", specId: specId,
                                                   partition: partition, recordCount: df["record_count"]?.int64 ?? 0,
                                                   size: df["file_size_in_bytes"]?.int64 ?? 0,
                                                   lower: Self.bytesMap(df["lower_bounds"]), upper: Self.bytesMap(df["upper_bounds"]),
                                                   nulls: Self.countMap(df["null_value_counts"]),
                                                   values: Self.countMap(df["value_counts"]),
                                                   nans: Self.countMap(df["nan_value_counts"])))
        }
    }

    /// Every field type any schema declares, by id (ids are never reused, so this is unambiguous).
    private func currentSchemaTypesById() -> [Int: LakehouseType] {
        var out: [Int: LakehouseType] = [:]
        for s in schemas { for f in s.fields { if let id = f.id { out[id] = f.type } } }
        for f in currentSchema.fields { if let id = f.id { out[id] = f.type } }
        return out
    }

    private static func bytesMap(_ v: AvroValue?) -> [Int: [UInt8]] {
        var out: [Int: [UInt8]] = [:]
        switch v {
        case .array(let items)?:
            for kv in items { if let k = kv["key"]?.int64, let b = kv["value"]?.bytes { out[Int(k)] = b } }
        case .map(let m)?:
            for (k, x) in m { if let i = Int(k), let b = x.bytes { out[i] = b } }
        default: break
        }
        return out
    }

    private static func countMap(_ v: AvroValue?) -> [Int: Int64] {
        var out: [Int: Int64] = [:]
        switch v {
        case .array(let items)?:
            for kv in items { if let k = kv["key"]?.int64, let c = kv["value"]?.int64 { out[Int(k)] = c } }
        case .map(let m)?:
            for (k, x) in m { if let i = Int(k), let c = x.int64 { out[i] = c } }
        default: break
        }
        return out
    }

    // MARK: Transforms and pruning

    /// The type of a partition field's values.
    static func resultType(_ pf: IcebergPartitionField, _ source: LakehouseType?) -> LakehouseType? {
        let t = pf.transform
        if t == "identity" || t.hasPrefix("truncate[") { return source }
        if ["year", "month", "day", "hour"].contains(t) || t.hasPrefix("bucket[") { return .int32 }
        return nil
    }

    /// A partition value from a manifest's partition record.
    static func avroScalar(_ v: AvroValue, _ type: LakehouseType?) -> LakeScalar? {
        switch v {
        case .null: return nil
        case .long(let i): return type == .boolean ? .bool(i != 0) : .int(i)
        case .boolean(let b): return .bool(b)
        case .float(let f): return .double(Double(f))
        case .double(let d): return .double(d)
        case .string(let s): return .string(s)
        case .bytes(let b), .fixed(let b):
            if case .decimal? = type { return .decimal(decimalFromBigEndian(b)) }
            return .bytes(b)
        default: return nil
        }
    }

    /// Iceberg's single-value binary serialization (bounds and partition summaries).
    static func decodeBound(_ b: [UInt8], _ type: LakehouseType?) -> LakeScalar? {
        func le(_ n: Int) -> UInt64? {
            guard b.count >= n else { return nil }
            var x: UInt64 = 0
            for i in 0..<n { x |= UInt64(b[i]) << (8 * UInt64(i)) }
            return x
        }
        switch type {
        case .int8?, .int16?, .int32?, .date?:
            // An int promoted to long keeps 4-byte bounds in old files; an 8-byte bound is a long.
            if b.count == 8 { return le(8).map { .int(Int64(bitPattern: $0)) } }
            return le(4).map { .int(Int64(Int32(bitPattern: UInt32($0)))) }
        case .int64?, .timestamp?, .time?:
            if b.count == 4 { return le(4).map { .int(Int64(Int32(bitPattern: UInt32($0)))) } }
            return le(8).map { .int(Int64(bitPattern: $0)) }
        case .float32?:
            if b.count == 8 { return le(8).map { .double(Double(bitPattern: $0)) } }
            return le(4).map { .double(Double(Float(bitPattern: UInt32($0)))) }
        case .float64?:
            if b.count == 4 { return le(4).map { .double(Double(Float(bitPattern: UInt32($0)))) } }
            return le(8).map { .double(Double(bitPattern: $0)) }
        case .string?: return .string(String(decoding: b, as: UTF8.self))
        case .boolean?: return b.first.map { .bool($0 != 0) }
        case .decimal?: return .decimal(decimalFromBigEndian(b))
        case .binary?, .fixed?, .uuid?: return .bytes(b)
        default: return nil
        }
    }

    static func decimalFromBigEndian(_ b: [UInt8]) -> ArrowDecimal128 {
        guard !b.isEmpty else { return ArrowDecimal128(0) }
        let negative = b[0] & 0x80 != 0
        var bytes = [UInt8](repeating: negative ? 0xFF : 0, count: 16)
        let n = Swift.min(b.count, 16)
        for i in 0..<n { bytes[15 - i] = b[b.count - 1 - i] }
        var hi: UInt64 = 0, lo: UInt64 = 0
        for i in 0..<8 { hi = (hi << 8) | UInt64(bytes[i]) }
        for i in 8..<16 { lo = (lo << 8) | UInt64(bytes[i]) }
        return ArrowDecimal128(lo: lo, hi: hi)
    }

    /// Applies a partition transform to a filter literal and returns the comparison to make against the
    /// partition value, or nil when the transform does not allow pruning for this comparison.
    static func project(_ f: LakeFilter, _ pf: IcebergPartitionField) -> (CompareOp, LakeScalar)? {
        let t = pf.transform
        if t == "identity" { return (f.op, f.literal) }
        // Order-preserving but not injective: a strict comparison becomes an inclusive one, and `!=`
        // says nothing about the partition.
        let op: CompareOp
        switch f.op {
        case .lt, .le: op = .le
        case .gt, .ge: op = .ge
        case .eq: op = .eq
        case .ne: return nil
        }
        let perSecond: Int64
        if case .timestamp(let ns, _) = f.field.type { perSecond = ns ? 1_000_000_000 : 1_000_000 } else { perSecond = 0 }
        // Floor division for b > 0 that cannot overflow (Int64.min included).
        func floorDiv(_ a: Int64, _ b: Int64) -> Int64 {
            let q = a / b
            return (a % b != 0 && a < 0) ? q - 1 : q
        }
        func days() -> Int64? {
            guard case .int(let v) = f.literal else { return nil }
            if f.field.type == .date { return v }
            if perSecond > 0 { return floorDiv(v, 86400 * perSecond) }
            return nil
        }
        switch t {
        case "day":
            return days().map { (op, .int($0)) }
        case "hour":
            guard perSecond > 0, case .int(let v) = f.literal else { return nil }
            return (op, .int(floorDiv(v, 3600 * perSecond)))
        case "year", "month":
            // Beyond a few million years the transform is not worth computing; do not prune.
            guard let d = days(), d.magnitude <= 1 << 31 else { return nil }
            let c = LakeTime.civil(d)
            let y = Int64(c.year - 1970)
            return (op, .int(t == "year" ? y : y * 12 + Int64(c.month - 1)))
        default:
            if t.hasPrefix("truncate["), t.hasSuffix("]"), let w = Int64(t.dropFirst(9).dropLast()), w > 0 {
                switch f.literal {
                case .int(let v) where f.field.type.isInteger:
                    // v - ((v % w) + w) % w, without overflowing near the ends of the Int64 range.
                    let r = v % w
                    let rem = r < 0 ? r + w : r          // r is in (-w, w), so r + w cannot overflow
                    let (t, o) = v.subtractingReportingOverflow(rem)
                    return o ? nil : (op, .int(t))
                case .string(let s):
                    return (op, .string(String(String.UnicodeScalarView(s.unicodeScalars.prefix(Int(w))))))
                default: return nil
                }
            }
            return nil
        }
    }

    private func manifestMayMatch(_ summaries: [AvroValue], spec: IcebergPartitionSpec, schema: IcebergSchema,
                                  filters: [LakeFilter]) -> Bool {
        let types = currentSchemaTypesById()
        for f in filters {
            guard let fid = f.field.id else { continue }
            for (i, pf) in spec.fields.enumerated() where pf.sourceId == fid && i < summaries.count {
                guard let (op, lit) = Self.project(f, pf) else { continue }
                let s = summaries[i]
                let rt = Self.resultType(pf, types[pf.sourceId])
                let lo = s["lower_bound"]?.bytes.flatMap { Self.decodeBound($0, rt) }
                let hi = s["upper_bound"]?.bytes.flatMap { Self.decodeBound($0, rt) }
                if lo == nil && hi == nil {
                    // No bounds: either every value is null (nothing matches) or nothing is known.
                    if s["contains_null"]?.bool == true && s["contains_nan"]?.bool != true { return false }
                    continue
                }
                if (f.field.type == .float32 || f.field.type == .float64) && op == .ne { continue }
                if !lakeRangeMayMatch(op, lo: lo, hi: hi, literal: lit) { return false }
            }
        }
        return true
    }

    private func partitionMayMatch(_ values: [LakeScalar?], spec: IcebergPartitionSpec, schema: IcebergSchema,
                                   filters: [LakeFilter]) -> Bool {
        for f in filters {
            guard let fid = f.field.id else { continue }
            for (i, pf) in spec.fields.enumerated() where pf.sourceId == fid && i < values.count {
                guard let (op, lit) = Self.project(f, pf) else { continue }
                guard let v = values[i] else { return false }          // a null partition value never matches
                if (f.field.type == .float32 || f.field.type == .float64) && op == .ne { continue }
                if !lakeRangeMayMatch(op, lo: v, hi: v, literal: lit) { return false }
            }
        }
        return true
    }

    private func boundsMayMatch(_ f: RawDataFile, filters: [LakeFilter]) -> Bool {
        for flt in filters {
            guard let id = flt.field.id else { continue }
            if let n = f.values[id], let nc = f.nulls[id], n > 0, n == nc { return false }
            let isFloat = flt.field.type == .float32 || flt.field.type == .float64
            if isFloat && flt.op == .ne { continue }
            switch flt.field.type {
            case .binary, .fixed, .uuid, .decimal, .nested: continue
            default: break
            }
            let lo = f.lower[id].flatMap { Self.decodeBound($0, flt.field.type) }
            let hi = f.upper[id].flatMap { Self.decodeBound($0, flt.field.type) }
            if lo == nil && hi == nil { continue }
            if !lakeRangeMayMatch(flt.op, lo: lo, hi: hi, literal: flt.literal) { return false }
        }
        return true
    }

    // MARK: Reading

    /// Reads a snapshot (the current one when nil): `columns` projects by name in the scan's schema (all
    /// columns when nil), and `filters` keeps the rows where every `(column, op, value)` holds. Filters
    /// prune manifests, data files and row groups first and are then applied to the rows, so the result
    /// holds exactly the matching rows.
    public func read(snapshotId: Int64? = nil, columns: [String]? = nil, filters: [ParquetFilter] = []) throws -> MetalRecordBatch {
        try scan(snapshotId: snapshotId, columns: columns, filters: filters).batch
    }

    /// `read`, plus how many manifests and files the filters pruned.
    public func scan(snapshotId: Int64? = nil, columns: [String]? = nil, filters: [ParquetFilter] = []) throws -> LakehouseScan {
        let schema = try schema(forSnapshot: snapshotId)
        let names = columns ?? schema.fields.map { $0.name }
        var fields: [LakehouseField] = try names.map { n in
            guard let f = schema.fields.first(where: { $0.name == n }) else {
                throw LakehouseError.invalidArgument("no column named \(n) in the Iceberg table \(metadataPath) "
                                                     + "(columns: \(schema.fields.map { $0.name }.joined(separator: ", ")))")
            }
            if case .nested(let s) = f.type {
                throw LakehouseError.unsupportedFeature("column \(n) has nested type \(s); nested columns are not read (project other columns)")
            }
            return f
        }
        let keep = fields.count
        let resolved = try lakeResolveFilters(filters, schema: schema.fields, table: metadataPath)
        var rowFilters: [(column: Int, op: CompareOp, literal: LakeScalar)] = []
        for f in resolved {
            var idx = fields.firstIndex(where: { $0.id == f.field.id })
            if idx == nil { fields.append(f.field); idx = fields.count - 1 }
            rowFilters.append((idx!, f.op, f.literal))
        }
        let (files, stats) = try plan(snapshotId: snapshotId, schema: schema, filters: resolved)
        let nameMapping = Self.parseNameMapping(properties["schema.name-mapping.default"])
        let specs = self.specs
        let batches = try LakeDataFile.readAll(count: files.count) { [fields, rowFilters, context] i in
            let df = files[i]
            let local = try self.resolve(df.path)
            let spec = specs.first { $0.specId == df.specId }
            return try LakeDataFile.read(path: local, fields: fields, resolve: { pf in
                let byId = Self.parquetNamesById(pf, nameMapping: nameMapping)
                return fields.map { f -> LakeColumnSource in
                    if let id = f.id, let n = byId[id] { return .parquet(n) }
                    // Not in the file: an identity partition value if there is one, else null.
                    if let spec, let id = f.id,
                       let i = spec.fields.firstIndex(where: { $0.sourceId == id && $0.transform == "identity" }),
                       i < df.partition.count {
                        return .constant(df.partition[i])
                    }
                    return .constant(nil)
                }
            }, rowGroupFilters: { pf in
                let byId = Self.parquetNamesById(pf, nameMapping: nameMapping)
                return resolved.compactMap { f in
                    guard let id = f.field.id, let n = byId[id] else { return nil }
                    return (column: n, filter: f)
                }
            }, rowFilters: rowFilters, context: context)
        }
        return LakehouseScan(batch: try LakeDataFile.finish(batches, fields: fields, keep: keep, context: context),
                             stats: stats)
    }

    /// Top-level Parquet column names by Iceberg field id. Files written without field ids (imported
    /// files) fall back to the table's `schema.name-mapping.default`, when it has one.
    static func parquetNamesById(_ pf: ParquetFile, nameMapping: [String: Int]) -> [Int: String] {
        var out: [Int: String] = [:]
        let els = pf.metadata.schema
        var index = 1
        var anyId = false
        var topLevel: [ParquetSchemaElement] = []
        func skip() {
            guard index < els.count else { return }
            let e = els[index]
            index += 1
            for _ in 0..<Swift.max(e.numChildren, 0) { skip() }
        }
        let rootChildren = els.first?.numChildren ?? 0
        for _ in 0..<Swift.max(rootChildren, 0) {
            guard index < els.count else { break }
            topLevel.append(els[index])
            skip()
        }
        for e in topLevel { if let id = e.fieldID { out[Int(id)] = e.name; anyId = true } }
        if !anyId {
            for e in topLevel { if let id = nameMapping[e.name] { out[id] = e.name } }
        }
        return out
    }

    /// `schema.name-mapping.default`: `[{"field-id": 1, "names": ["id", "old_id"]}, ...]` as name -> id.
    static func parseNameMapping(_ json: String?) -> [String: Int] {
        guard let json, let data = json.data(using: .utf8),
              let list = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [:] }
        var out: [String: Int] = [:]
        for m in list {
            guard let id = (m["field-id"] as? NSNumber)?.intValue else { continue }
            for n in m["names"] as? [String] ?? [] { out[n] = id }
        }
        return out
    }
}
