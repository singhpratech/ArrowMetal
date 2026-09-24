import Foundation

// Delta Lake tables.
//
// A Delta table is a directory of Parquet data files plus `_delta_log/`, an ordered log of commits:
// `00000000000000000000.json`, `...01.json`, ..., each a newline-delimited list of actions (`add` a
// file, `remove` a file, replace the `metaData`, change the `protocol`). Periodic checkpoints
// (`N.checkpoint.parquet`, or `N.checkpoint.<part>.<parts>.parquet`) hold the reconciled state at version
// N, so version V is the latest checkpoint at or below V plus the commits after it.
//
// This reader replays that log on the CPU (checkpoints are read with the GPU Parquet reader), picks the
// files live at the requested version, prunes them by partition values and per-file statistics, and
// reads the survivors with the GPU Parquet reader.
//
// Protocol support: reader versions 1 to 3; column mapping modes `none` and `name`; reader features
// `columnMapping`, `timestampNtz` and `vacuumProtocolCheck`. Everything else a table requires of its
// readers (deletion vectors, column mapping mode `id`, type widening, v2 checkpoints, variant, any
// feature not listed) is rejected with an error naming it, because reading such a table while ignoring
// the feature would return wrong rows.

/// The `protocol` action: what a reader must support.
public struct DeltaProtocol: Sendable, Equatable {
    public var minReaderVersion: Int
    public var minWriterVersion: Int
    public var readerFeatures: [String]?
    public var writerFeatures: [String]?
}

/// The `metaData` action: schema, partitioning and table properties.
public struct DeltaMetadata: Sendable {
    public var id: String
    public var schemaString: String
    public var partitionColumns: [String]
    public var configuration: [String: String]
    public var format: String
}

/// A live data file (an `add` action).
public struct DeltaAddFile: Sendable {
    /// The path as the log records it: relative to the table root and URI-encoded, or an absolute URI.
    public let path: String
    /// Partition values as strings, keyed by the partition column's physical name; nil is a null value.
    public let partitionValues: [String: String?]
    public let size: Int64
    /// The statistics JSON (`numRecords`, `minValues`, `maxValues`, `nullCount`), when written.
    public let stats: String?
    public let hasDeletionVector: Bool
}

/// The reconciled state of a Delta table at one version.
public struct DeltaSnapshot: Sendable {
    public let tablePath: String
    public let version: Int64
    public let protocolAction: DeltaProtocol
    public let metadata: DeltaMetadata
    /// Live data files, in log order.
    public let files: [DeltaAddFile]
    /// The logical schema (column names as queried; `physicalName` is the name in the data files).
    public let schema: [LakehouseField]
    public let columnMappingMode: String
}

/// How a scan went: files considered and files skipped by each kind of pruning.
public struct LakehouseScanStats: Sendable, Equatable {
    public var filesTotal = 0
    public var filesPrunedByPartition = 0
    public var filesPrunedByStatistics = 0
    public var filesRead: Int { filesTotal - filesPrunedByPartition - filesPrunedByStatistics }
    /// Manifests (Iceberg only) considered and skipped by the manifest list's partition summaries.
    public var manifestsTotal = 0
    public var manifestsPruned = 0
    public init() {}
}

/// The rows a table scan produced and how the scan was pruned.
public struct LakehouseScan {
    public let batch: MetalRecordBatch
    public let stats: LakehouseScanStats
}

public final class DeltaTable: @unchecked Sendable {
    public let path: String
    public let logPath: String
    public let context: MetalContext

    /// The reader features this implementation understands.
    public static let supportedReaderFeatures: Set<String> = ["columnMapping", "timestampNtz", "vacuumProtocolCheck"]

    public init(path: String, context: MetalContext = .shared) throws {
        let root = try LakePath.local(path)
        let log = LakePath.join(root, "_delta_log")
        guard LakePath.isDirectory(log) else {
            throw LakehouseError.notATable("\(path) has no _delta_log directory, so it is not a Delta table")
        }
        self.path = root
        self.logPath = log
        self.context = context
    }

    // MARK: Log listing

    private struct LogListing {
        var commits: [Int64: String] = [:]
        /// version -> the file names of one complete classic checkpoint
        var checkpoints: [Int64: [String]] = [:]
        /// versions that only have a v2 (UUID-named) checkpoint
        var v2Checkpoints: Set<Int64> = []
    }

    private func listLog() throws -> LogListing {
        let names: [String]
        do { names = try FileManager.default.contentsOfDirectory(atPath: logPath) }
        catch { throw LakehouseError.notATable("cannot list \(logPath): \(error.localizedDescription)") }
        var out = LogListing()
        var parts: [Int64: [Int: [Int: String]]] = [:]     // version -> total parts -> part -> name
        for n in names {
            guard n.count > 20, let v = Int64(n.prefix(20)), n.prefix(20).allSatisfy(\.isASCIIDigit) else { continue }
            let rest = n.dropFirst(20)
            if rest == ".json" { out.commits[v] = n; continue }
            if rest == ".checkpoint.parquet" { out.checkpoints[v] = [n]; continue }
            if rest.hasPrefix(".checkpoint.") {
                let fields = rest.dropFirst(".checkpoint.".count).split(separator: ".")
                if fields.count == 3, fields[2] == "parquet", let part = Int(fields[0]), let total = Int(fields[1]),
                   fields[0].count == 10, fields[1].count == 10 {
                    parts[v, default: [:]][total, default: [:]][part] = n
                } else if fields.count == 2, fields[1] == "parquet" || fields[1] == "json" {
                    out.v2Checkpoints.insert(v)
                }
            }
        }
        for (v, byTotal) in parts where out.checkpoints[v] == nil {
            for (total, ps) in byTotal where ps.count == total && (1...total).allSatisfy({ ps[$0] != nil }) {
                out.checkpoints[v] = (1...total).map { ps[$0]! }
                break
            }
        }
        return out
    }

    /// The newest version the log describes.
    public func latestVersion() throws -> Int64 {
        let l = try listLog()
        guard let v = (Array(l.commits.keys) + Array(l.checkpoints.keys)).max() else {
            throw LakehouseError.notATable("\(logPath) holds no commits or checkpoints")
        }
        return v
    }

    // MARK: Replay

    /// The table state at `version` (the latest when nil).
    public func snapshot(version: Int64? = nil) throws -> DeltaSnapshot {
        let listing = try listLog()
        guard let latest = (Array(listing.commits.keys) + Array(listing.checkpoints.keys)).max() else {
            throw LakehouseError.notATable("\(logPath) holds no commits or checkpoints")
        }
        let target = version ?? latest
        guard target >= 0, target <= latest else {
            throw LakehouseError.notFound("version \(target) of the Delta table at \(path); the latest version is \(latest)")
        }
        // The newest classic checkpoint at or below the target from which every later commit exists.
        let base = listing.checkpoints.keys.filter { $0 <= target }.sorted(by: >).first { cp in
            cp == target || ((cp + 1)...target).allSatisfy { listing.commits[$0] != nil }
        }
        var start: Int64
        var protocolAction: DeltaProtocol? = nil
        var metadata: DeltaMetadata? = nil
        var files: [String: DeltaAddFile] = [:]
        var order: [String] = []
        if let base {
            let state = try readCheckpoint(listing.checkpoints[base]!)
            protocolAction = state.protocolAction
            metadata = state.metadata
            for f in state.files { if files[f.path] == nil { order.append(f.path) }; files[f.path] = f }
            start = base + 1
        } else {
            if let v2 = listing.v2Checkpoints.filter({ $0 <= target }).max(),
               !(0...target).allSatisfy({ listing.commits[$0] != nil }) {
                throw LakehouseError.unsupportedFeature(
                    "v2Checkpoint (the table at \(path) is only reconstructable from the v2 checkpoint at version \(v2))")
            }
            guard (0...target).allSatisfy({ listing.commits[$0] != nil }) else {
                let missing = (0...target).first { listing.commits[$0] == nil }!
                throw LakehouseError.notFound("version \(target) of the Delta table at \(path) cannot be reconstructed: "
                                              + "commit \(missing) is missing and no checkpoint covers it")
            }
            start = 0
        }
        if start <= target {
            for v in start...target {
                try replayCommit(LakePath.join(logPath, listing.commits[v]!), version: v,
                                 protocolAction: &protocolAction, metadata: &metadata, files: &files, order: &order)
            }
        }
        guard let p = protocolAction else {
            throw LakehouseError.malformed("the Delta log at \(logPath) has no protocol action up to version \(target)")
        }
        guard let m = metadata else {
            throw LakehouseError.malformed("the Delta log at \(logPath) has no metaData action up to version \(target)")
        }
        try Self.checkProtocol(p, table: path)
        let mode = m.configuration["delta.columnMapping.mode"] ?? "none"
        switch mode {
        case "none", "name": break
        case "id": throw LakehouseError.unsupportedFeature("columnMapping mode 'id' (table \(path)); modes 'none' and 'name' are read")
        default: throw LakehouseError.unsupportedFeature("columnMapping mode '\(mode)' (table \(path))")
        }
        guard m.format == "parquet" else {
            throw LakehouseError.unsupportedFeature("data file format '\(m.format)' (table \(path)); only parquet is read")
        }
        let live = order.compactMap { files[$0] }
        if let dv = live.first(where: { $0.hasDeletionVector }) {
            throw LakehouseError.unsupportedFeature("deletionVectors (file \(dv.path) of table \(path) has a deletion vector)")
        }
        let schema = try Self.parseSchema(m.schemaString, mode: mode, table: path)
        return DeltaSnapshot(tablePath: path, version: target, protocolAction: p, metadata: m, files: live,
                             schema: schema, columnMappingMode: mode)
    }

    static func checkProtocol(_ p: DeltaProtocol, table: String) throws {
        guard p.minReaderVersion >= 1 else {
            throw LakehouseError.malformed("minReaderVersion \(p.minReaderVersion) (table \(table))")
        }
        guard p.minReaderVersion <= 3 else {
            throw LakehouseError.unsupportedFeature("reader protocol version \(p.minReaderVersion) (table \(table)); versions 1 to 3 are read")
        }
        if p.minReaderVersion == 3 {
            guard let features = p.readerFeatures else {
                throw LakehouseError.malformed("reader protocol version 3 without protocol.readerFeatures (table \(table)); "
                                               + "version 3 requires the list")
            }
            for f in features where !supportedReaderFeatures.contains(f) {
                throw LakehouseError.unsupportedFeature("\(f) (a reader feature of table \(table); supported: "
                                                        + supportedReaderFeatures.sorted().joined(separator: ", ") + ")")
            }
        }
    }

    private func replayCommit(_ file: String, version: Int64, protocolAction: inout DeltaProtocol?,
                              metadata: inout DeltaMetadata?, files: inout [String: DeltaAddFile],
                              order: inout [String]) throws {
        guard let data = FileManager.default.contents(atPath: file) else {
            throw LakehouseError.malformed("cannot read commit \(file)")
        }
        var lineNo = 0
        for line in data.split(separator: UInt8(ascii: "\n")) {
            lineNo += 1
            if line.allSatisfy({ $0 == 32 || $0 == 13 || $0 == 9 }) { continue }
            guard let obj = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any] else {
                throw LakehouseError.malformed("commit \(file) line \(lineNo) is not a JSON object")
            }
            if let a = obj["add"] as? [String: Any] {
                let f = try Self.addFile(a, where: "commit \(version)")
                if files[f.path] == nil { order.append(f.path) }
                files[f.path] = f
            } else if let r = obj["remove"] as? [String: Any] {
                guard let p = r["path"] as? String else { throw LakehouseError.malformed("remove without a path in commit \(version)") }
                files[p] = nil
            } else if let m = obj["metaData"] as? [String: Any] {
                metadata = try Self.metadataAction(m, where: "commit \(version)")
            } else if let p = obj["protocol"] as? [String: Any] {
                guard let r = (p["minReaderVersion"] as? NSNumber)?.intValue,
                      let w = (p["minWriterVersion"] as? NSNumber)?.intValue else {
                    throw LakehouseError.malformed("protocol action without versions in commit \(version)")
                }
                func featureList(_ key: String) throws -> [String]? {
                    switch p[key] {
                    case nil, is NSNull: return nil
                    case let list as [String]: return list
                    default:
                        throw LakehouseError.malformed("protocol.\(key) in commit \(version) is not a list of strings")
                    }
                }
                protocolAction = DeltaProtocol(minReaderVersion: r, minWriterVersion: w,
                                               readerFeatures: try featureList("readerFeatures"),
                                               writerFeatures: try featureList("writerFeatures"))
            }
        }
    }

    static func addFile(_ a: [String: Any], where loc: String) throws -> DeltaAddFile {
        guard let p = a["path"] as? String else { throw LakehouseError.malformed("add without a path in \(loc)") }
        var pv: [String: String?] = [:]
        switch a["partitionValues"] {
        case nil, is NSNull: break
        case let m as [String: Any]:
            for (k, v) in m {
                switch v {
                case let s as String: pv[k] = .some(s)
                case is NSNull: pv[k] = .some(nil)
                default:
                    throw LakehouseError.malformed("partition value \(v) of \(k) for \(p) in \(loc) is not a string or null")
                }
            }
        default:
            throw LakehouseError.malformed("partitionValues of \(p) in \(loc) is not a JSON object")
        }
        let dv = a["deletionVector"]
        return DeltaAddFile(path: p, partitionValues: pv, size: (a["size"] as? NSNumber)?.int64Value ?? 0,
                            stats: a["stats"] as? String, hasDeletionVector: dv != nil && !(dv is NSNull))
    }

    static func metadataAction(_ m: [String: Any], where loc: String) throws -> DeltaMetadata {
        guard let schema = m["schemaString"] as? String else {
            throw LakehouseError.malformed("metaData without a schemaString in \(loc)")
        }
        var conf: [String: String] = [:]
        if let c = m["configuration"] as? [String: Any] { for (k, v) in c { if let s = v as? String { conf[k] = s } } }
        let format = ((m["format"] as? [String: Any])?["provider"] as? String) ?? "parquet"
        return DeltaMetadata(id: (m["id"] as? String) ?? "", schemaString: schema,
                             partitionColumns: (m["partitionColumns"] as? [String]) ?? [],
                             configuration: conf, format: format)
    }

    private struct CheckpointState {
        var protocolAction: DeltaProtocol?
        var metadata: DeltaMetadata?
        var files: [DeltaAddFile]
    }

    private func readCheckpoint(_ names: [String]) throws -> CheckpointState {
        var state = CheckpointState(protocolAction: nil, metadata: nil, files: [])
        let paths = ["add.path", "add.partitionValues", "add.size", "add.stats", "add.deletionVector.storageType",
                     "metaData.id", "metaData.schemaString", "metaData.partitionColumns", "metaData.configuration",
                     "metaData.format.provider",
                     "protocol.minReaderVersion", "protocol.minWriterVersion", "protocol.readerFeatures",
                     "protocol.writerFeatures"]
        for name in names {
            let full = LakePath.join(logPath, name)
            let file: ParquetFile
            do { file = try ParquetFile(path: full, context: context) }
            catch { throw LakehouseError.malformed("checkpoint \(full): \(error)") }
            let cols = try file.lakeReadCheckpointColumns(paths)
            func strings(_ p: String) -> [String?]? { if case .strings(let s)? = cols[p] { return s }; return nil }
            func ints(_ p: String) -> [Int64?]? { if case .ints(let s)? = cols[p] { return s }; return nil }
            func lists(_ p: String) -> [[String]?]? { if case .stringLists(let s)? = cols[p] { return s }; return nil }
            func maps(_ p: String) -> [[String: String?]?]? { if case .maps(let s)? = cols[p] { return s }; return nil }
            let n = Int(file.numRows)
            let addPath = strings("add.path"), addPV = maps("add.partitionValues"), addSize = ints("add.size")
            let addStats = strings("add.stats"), addDV = strings("add.deletionVector.storageType")
            let mdID = strings("metaData.id"), mdSchema = strings("metaData.schemaString")
            let mdParts = lists("metaData.partitionColumns"), mdConf = maps("metaData.configuration")
            let mdFormat = strings("metaData.format.provider")
            let prR = ints("protocol.minReaderVersion"), prW = ints("protocol.minWriterVersion")
            let prRF = lists("protocol.readerFeatures"), prWF = lists("protocol.writerFeatures")
            for i in 0..<n {
                if let ap = addPath, i < ap.count, let p = ap[i] {
                    var pv: [String: String?] = [:]
                    if let m = addPV, i < m.count, let mm = m[i] { pv = mm }
                    state.files.append(DeltaAddFile(path: p, partitionValues: pv,
                                                    size: addSize.flatMap { i < $0.count ? $0[i] : nil } ?? 0,
                                                    stats: addStats.flatMap { i < $0.count ? $0[i] : nil },
                                                    hasDeletionVector: addDV.flatMap { i < $0.count ? $0[i] : nil } != nil))
                    continue
                }
                if let s = mdSchema, i < s.count, let schema = s[i] {
                    var conf: [String: String] = [:]
                    if let c = mdConf, i < c.count, let cc = c[i] { for (k, v) in cc { if let v { conf[k] = v } } }
                    state.metadata = DeltaMetadata(id: mdID.flatMap { i < $0.count ? $0[i] : nil } ?? "",
                                                   schemaString: schema,
                                                   partitionColumns: mdParts.flatMap { i < $0.count ? $0[i] : nil } ?? [],
                                                   configuration: conf,
                                                   format: mdFormat.flatMap { i < $0.count ? $0[i] : nil } ?? "parquet")
                    continue
                }
                if let r = prR, i < r.count, let rv = r[i] {
                    state.protocolAction = DeltaProtocol(minReaderVersion: Int(rv),
                                                         minWriterVersion: Int(prW.flatMap { i < $0.count ? $0[i] : nil } ?? 0),
                                                         readerFeatures: prRF.flatMap { i < $0.count ? $0[i] : nil },
                                                         writerFeatures: prWF.flatMap { i < $0.count ? $0[i] : nil })
                }
            }
        }
        return state
    }

    // MARK: Schema

    static func parseSchema(_ json: String, mode: String, table: String) throws -> [LakehouseField] {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let fields = obj["fields"] as? [[String: Any]] else {
            throw LakehouseError.malformed("the schemaString of table \(table) is not a struct type")
        }
        return try fields.map { f in
            guard let name = f["name"] as? String else { throw LakehouseError.malformed("schema field without a name (table \(table))") }
            let meta = f["metadata"] as? [String: Any] ?? [:]
            var physical = name
            if mode == "name" {
                guard let p = meta["delta.columnMapping.physicalName"] as? String else {
                    throw LakehouseError.malformed("column \(name) of table \(table) has no delta.columnMapping.physicalName")
                }
                physical = p
            }
            let id = (meta["delta.columnMapping.id"] as? NSNumber)?.intValue
            return LakehouseField(name: name, type: parseType(f["type"] as Any), nullable: (f["nullable"] as? Bool) ?? true,
                                  id: id, physicalName: physical)
        }
    }

    static func parseType(_ t: Any) -> LakehouseType {
        if let s = t as? String {
            switch s {
            case "string": return .string
            case "long": return .int64
            case "integer": return .int32
            case "short": return .int16
            case "byte": return .int8
            case "float": return .float32
            case "double": return .float64
            case "boolean": return .boolean
            case "binary": return .binary
            case "date": return .date
            case "timestamp": return .timestamp(nanos: false, utc: true)
            case "timestamp_ntz": return .timestamp(nanos: false, utc: false)
            default:
                if s.hasPrefix("decimal("), s.hasSuffix(")") {
                    let inner = s.dropFirst(8).dropLast().split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                    if inner.count == 2, let p = Int(inner[0]), let sc = Int(inner[1]) { return .decimal(precision: p, scale: sc) }
                }
                return .nested(s)
            }
        }
        if let d = t as? [String: Any], let kind = d["type"] as? String { return .nested(kind) }
        return .nested("unknown")
    }

    /// A partition value string (Delta's partition value serialization) as a scalar of `type`. The
    /// protocol serializes a null partition value as null or as the empty string, for every type
    /// (strings included), which is how `deltalake` and DuckDB read it.
    static func partitionScalar(_ s: String?, _ type: LakehouseType, column: String) throws -> LakeScalar? {
        guard let s, !s.isEmpty else { return nil }
        func bad() -> LakehouseError { .malformed("partition value \"\(s)\" of column \(column) is not a \(type)") }
        switch type {
        case .string: return .string(s)
        case .int8, .int16, .int32, .int64:
            guard let i = Int64(s) else { throw bad() }
            return .int(i)
        case .float32:
            // The value the column holds is the float nearest the text.
            guard let d = Double(s) else { throw bad() }
            return .double(Double(Float(d)))
        case .float64:
            guard let d = Double(s) else { throw bad() }
            return .double(d)
        case .boolean:
            switch s.lowercased() {
            case "true": return .bool(true)
            case "false": return .bool(false)
            default: throw bad()
            }
        case .date:
            guard let d = LakeTime.parseDate(s) else { throw bad() }
            return .int(d)
        case .timestamp:
            guard let t = LakeTime.parseTimestamp(s, unitsPerSecond: 1_000_000) else { throw bad() }
            return .int(t)
        case .decimal(_, let scale):
            guard let d = LakeTime.parseDecimal(s, scale: scale) else { throw bad() }
            return .decimal(d)
        case .binary: return .bytes(Array(s.utf8))
        default: throw LakehouseError.unsupportedFeature("partition column \(column) of type \(type)")
        }
    }

    // MARK: Reading

    /// Reads the table at `version` (latest when nil): `columns` projects (all columns when nil), and
    /// `filters` keeps the rows where every `(column, op, value)` holds. Filters prune files by partition
    /// values and by the per-file min/max statistics, prune row groups by the Parquet footer statistics,
    /// and are then applied to the rows, so the result holds exactly the matching rows.
    public func read(version: Int64? = nil, columns: [String]? = nil, filters: [ParquetFilter] = []) throws -> MetalRecordBatch {
        try snapshot(version: version).read(columns: columns, filters: filters, context: context)
    }

    /// `read`, plus how many files the filters pruned.
    public func scan(version: Int64? = nil, columns: [String]? = nil, filters: [ParquetFilter] = []) throws -> LakehouseScan {
        try snapshot(version: version).scan(columns: columns, filters: filters, context: context)
    }
}

extension DeltaSnapshot {
    /// The live files that may hold rows matching every filter, and how many were pruned by what.
    public func plan(filters: [ParquetFilter] = []) throws -> (files: [DeltaAddFile], stats: LakehouseScanStats) {
        let resolved = try lakeResolveFilters(filters, schema: schema, table: tablePath)
        return try plan(resolved)
    }

    private var partitionSet: Set<String> { Set(metadata.partitionColumns) }

    func plan(_ filters: [LakeFilter]) throws -> (files: [DeltaAddFile], stats: LakehouseScanStats) {
        var stats = LakehouseScanStats()
        stats.filesTotal = files.count
        var out: [DeltaAddFile] = []
        let parts = partitionSet
        outer: for f in files {
            for flt in filters where parts.contains(flt.field.name) {
                let raw = f.partitionValues[flt.field.physicalName] ?? nil
                let v = try DeltaTable.partitionScalar(raw, flt.field.type, column: flt.field.name)
                // A null partition value matches no comparison. A value that cannot be ordered against
                // the literal (a NaN on either side) is unequal to it, so it matches `!=` only.
                let keep: Bool
                if let v {
                    if let c = LakeScalar.compare(v, flt.literal) { keep = flt.op.holds(c) } else { keep = flt.op == .ne }
                } else {
                    keep = false
                }
                if !keep {
                    stats.filesPrunedByPartition += 1
                    continue outer
                }
            }
            let dataFilters = filters.filter { !parts.contains($0.field.name) }
            if !dataFilters.isEmpty, let s = f.stats, !statsMayMatch(s, dataFilters) {
                stats.filesPrunedByStatistics += 1
                continue
            }
            out.append(f)
        }
        return (out, stats)
    }

    /// Whether a file's statistics allow a row matching every filter.
    private func statsMayMatch(_ json: String, _ filters: [LakeFilter]) -> Bool {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return true }
        let mins = obj["minValues"] as? [String: Any] ?? [:]
        let maxs = obj["maxValues"] as? [String: Any] ?? [:]
        let nulls = obj["nullCount"] as? [String: Any] ?? [:]
        let numRecords = (obj["numRecords"] as? NSNumber)?.int64Value
        for f in filters {
            let key = f.field.physicalName
            if let n = numRecords, let nc = (nulls[key] as? NSNumber)?.int64Value, n == nc {
                return false       // every value is null, and a comparison with null never holds
            }
            let lo = Self.statScalar(mins[key], f.field.type)
            let hi = Self.statScalar(maxs[key], f.field.type)
            if lo == nil && hi == nil { continue }
            if case .double(let d)? = lo, d.isNaN { continue }
            if case .double(let d)? = hi, d.isNaN { continue }
            // A NaN in a float column satisfies `!=`; statistics do not say whether one is present.
            if f.op == .ne, f.field.type == .float32 || f.field.type == .float64 { continue }
            if !lakeRangeMayMatch(f.op, lo: lo, hi: hi, literal: f.literal) { return false }
        }
        return true
    }

    /// A statistics value as a scalar, or nil when it is absent or not safe to prune on: timestamps are
    /// written at millisecond precision and long strings may be truncated; boolean, decimal, binary and
    /// the other types are not used for pruning.
    static func statScalar(_ v: Any?, _ type: LakehouseType) -> LakeScalar? {
        guard let v, !(v is NSNull) else { return nil }
        switch type {
        case .int8, .int16, .int32, .int64:
            guard let n = v as? NSNumber, CFNumberIsFloatType(n) == false else { return nil }
            return .int(n.int64Value)
        case .float32:
            // Writers print a float32 bound either exactly or as the shortest text that rounds back to
            // it; rounding to the nearest float recovers the value either way.
            guard let n = v as? NSNumber else { return nil }
            return .double(Double(Float(n.doubleValue)))
        case .float64:
            guard let n = v as? NSNumber else { return nil }
            return .double(n.doubleValue)
        case .string:
            guard let s = v as? String, s.count < 32 else { return nil }
            return .string(s)
        case .date:
            guard let s = v as? String, let d = LakeTime.parseDate(s) else { return nil }
            return .int(d)
        default:
            return nil
        }
    }

    /// Reads the snapshot's rows. See `DeltaTable.read`.
    public func read(columns: [String]? = nil, filters: [ParquetFilter] = [],
                     context: MetalContext = .shared) throws -> MetalRecordBatch {
        try scan(columns: columns, filters: filters, context: context).batch
    }

    /// `read`, plus how many files the filters pruned.
    public func scan(columns: [String]? = nil, filters: [ParquetFilter] = [],
                     context: MetalContext = .shared) throws -> LakehouseScan {
        let names = columns ?? schema.map { $0.name }
        var fields: [LakehouseField] = try names.map { n in
            guard let f = schema.first(where: { $0.name == n }) else {
                throw LakehouseError.invalidArgument("no column named \(n) in the Delta table at \(tablePath) "
                                                     + "(columns: \(schema.map { $0.name }.joined(separator: ", ")))")
            }
            if case .nested(let s) = f.type {
                throw LakehouseError.unsupportedFeature("column \(n) has nested type \(s); nested columns are not read (project other columns)")
            }
            return f
        }
        let keep = fields.count
        let resolved = try lakeResolveFilters(filters, schema: schema, table: tablePath)
        let parts = partitionSet
        var rowFilters: [(column: Int, op: CompareOp, literal: LakeScalar)] = []
        for f in resolved where !parts.contains(f.field.name) {
            var idx = fields.firstIndex(where: { $0.name == f.field.name })
            if idx == nil { fields.append(f.field); idx = fields.count - 1 }
            rowFilters.append((idx!, f.op, f.literal))
        }
        let rowGroupFilters = resolved.filter { !parts.contains($0.field.name) }.map { (column: $0.field.physicalName, filter: $0) }
        let (selected, stats) = try plan(resolved)
        let root = tablePath
        // Without column mapping a column cannot be dropped, so every data file holds at least one of the
        // table's data columns; a file holding none of them is not a data file of this table.
        let dataColumns = columnMappingMode == "none"
            ? schema.filter { !parts.contains($0.name) }.map { $0.physicalName } : []
        let batches = try LakeDataFile.readAll(count: selected.count) { [fields] i in
            let file = selected[i]
            let local = try LakePath.local(file.path.contains("://") || file.path.hasPrefix("file:") ? file.path
                                           : LakePath.join(root, file.path.removingPercentEncoding ?? file.path))
            return try LakeDataFile.read(path: local, fields: fields, resolve: { pf in
                if !dataColumns.isEmpty, !pf.fields.contains(where: { dataColumns.contains($0.name) }) {
                    throw LakehouseError.malformed("data file \(local) holds none of the table's columns ("
                                                   + dataColumns.joined(separator: ", ") + ")")
                }
                return try fields.map { f -> LakeColumnSource in
                    if parts.contains(f.name) {
                        let raw = file.partitionValues[f.physicalName] ?? nil
                        return .constant(try DeltaTable.partitionScalar(raw, f.type, column: f.name))
                    }
                    return pf.fields.contains(where: { $0.name == f.physicalName }) ? .parquet(f.physicalName) : .constant(nil)
                }
            }, rowGroupFilters: { _ in rowGroupFilters }, rowFilters: rowFilters, context: context)
        }
        return LakehouseScan(batch: try LakeDataFile.finish(batches, fields: fields, keep: keep, context: context),
                             stats: stats)
    }
}

extension ParquetFilter.Op {
    init(_ op: CompareOp) {
        switch op {
        case .eq: self = .eq
        case .ne: self = .ne
        case .lt: self = .lt
        case .le: self = .le
        case .gt: self = .gt
        case .ge: self = .ge
        }
    }
}
