import Foundation
import CArrowABI
import ArrowMetal

// C ABI over the Delta Lake and Apache Iceberg readers. A read returns a batch handle whose columns are
// handed out one at a time as ordinary `am_array` handles, exactly like a Parquet read, plus the scan's
// pruning counters.

private final class LakehouseBatchBox {
    let batch: MetalRecordBatch
    let stats: LakehouseScanStats
    var names: [UnsafeMutablePointer<CChar>] = []
    init(_ s: LakehouseScan) {
        batch = s.batch
        stats = s.stats
        names = s.batch.names.map { strdup($0)! }
    }
    deinit { for n in names { free(n) } }
}

private func lhStoreMessage(_ m: String) {
    Thread.current.threadDictionary["ArrowMetalC.lastError"] = m
}

@discardableResult
private func lhBadArgument(_ function: String, _ detail: String) -> Int32 {
    lhStoreMessage("\(function): \(detail)")
    return 2
}

@inline(__always) private func lhBatch(_ p: OpaquePointer?) -> LakehouseBatchBox? {
    guard let p else { return nil }
    return Unmanaged<LakehouseBatchBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

/// Shared argument handling: a NULL `columns` reads every column, a non-NULL one exactly those.
private func lhColumns(_ function: String, _ columns: UnsafePointer<UnsafePointer<CChar>?>?,
                       _ n: Int64) throws -> [String]? {
    guard let columns else { return nil }
    return try (0..<Int(n)).map { i in
        guard let c = columns[i] else {
            throw LakehouseError.invalidArgument("\(function): `columns`[\(i)] is NULL")
        }
        return String(cString: c)
    }
}

private func lhRead(_ function: String, _ path: UnsafePointer<CChar>?,
                    _ columns: UnsafePointer<UnsafePointer<CChar>?>?, _ nColumns: Int64,
                    _ filters: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?,
                    _ body: (String, [String]?, [ParquetFilter]) throws -> LakehouseScan) -> Int32 {
    guard let path else { return lhBadArgument(function, "`path` is NULL") }
    guard let out else { return lhBadArgument(function, "`out` is NULL") }
    guard nColumns >= 0 else { return lhBadArgument(function, "`n_columns` is \(nColumns), which is negative") }
    guard columns != nil || nColumns == 0 else {
        return lhBadArgument(function, "`columns` is NULL but `n_columns` is \(nColumns)")
    }
    do {
        let cols = try lhColumns(function, columns, nColumns)
        let parsed = try filters.map { try parseFilters(String(cString: $0)) } ?? []
        let scan = try body(String(cString: path), cols, parsed)
        out.pointee = OpaquePointer(Unmanaged.passRetained(LakehouseBatchBox(scan)).toOpaque())
        return 0
    } catch {
        lhStoreMessage("\(function): \(error)")
        return 1
    }
}

/// Reads a Delta Lake table. `version` -1 reads the latest version; any other negative version is an error.
@_cdecl("am_delta_read")
public func am_delta_read(_ path: UnsafePointer<CChar>?, _ version: Int64,
                          _ columns: UnsafePointer<UnsafePointer<CChar>?>?, _ nColumns: Int64,
                          _ filters: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    lhRead("am_delta_read", path, columns, nColumns, filters, out) { p, cols, flt in
        guard version >= -1 else {
            throw LakehouseError.invalidArgument("version \(version); a Delta version is 0 or more, or -1 for the latest")
        }
        return try DeltaTable(path: p).scan(version: version >= 0 ? version : nil, columns: cols, filters: flt)
    }
}

/// The latest version of a Delta Lake table, or -1 (with am_last_error set).
@_cdecl("am_delta_latest_version")
public func am_delta_latest_version(_ path: UnsafePointer<CChar>?) -> Int64 {
    guard let path else { lhBadArgument("am_delta_latest_version", "`path` is NULL"); return -1 }
    do { return try DeltaTable(path: String(cString: path)).latestVersion() }
    catch { lhStoreMessage("am_delta_latest_version: \(error)"); return -1 }
}

/// Reads an Iceberg table from a metadata JSON file or a table directory. With `has_snapshot_id` 0 the
/// current snapshot is read; otherwise `snapshot_id`.
@_cdecl("am_iceberg_read")
public func am_iceberg_read(_ path: UnsafePointer<CChar>?, _ snapshotId: Int64, _ hasSnapshotId: Int32,
                            _ columns: UnsafePointer<UnsafePointer<CChar>?>?, _ nColumns: Int64,
                            _ filters: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    lhRead("am_iceberg_read", path, columns, nColumns, filters, out) { p, cols, flt in
        try IcebergTable(path: p).scan(snapshotId: hasSnapshotId != 0 ? snapshotId : nil, columns: cols, filters: flt)
    }
}

/// The current snapshot id of an Iceberg table into `out`; returns 0 when there is one, 3 when the
/// table has no current snapshot, 1 on error.
@_cdecl("am_iceberg_current_snapshot")
public func am_iceberg_current_snapshot(_ path: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<Int64>?) -> Int32 {
    guard let path else { return lhBadArgument("am_iceberg_current_snapshot", "`path` is NULL") }
    guard let out else { return lhBadArgument("am_iceberg_current_snapshot", "`out` is NULL") }
    do {
        guard let s = try IcebergTable(path: String(cString: path)).currentSnapshotId else { return 3 }
        out.pointee = s
        return 0
    } catch { lhStoreMessage("am_iceberg_current_snapshot: \(error)"); return 1 }
}

@_cdecl("am_lakehouse_batch_columns")
public func am_lakehouse_batch_columns(_ b: OpaquePointer?) -> Int64 {
    guard let box = lhBatch(b) else { lhStoreMessage("am_lakehouse_batch_columns: `b` is NULL (no batch)"); return -1 }
    return Int64(box.batch.columnCount)
}

@_cdecl("am_lakehouse_batch_rows")
public func am_lakehouse_batch_rows(_ b: OpaquePointer?) -> Int64 {
    guard let box = lhBatch(b) else { lhStoreMessage("am_lakehouse_batch_rows: `b` is NULL (no batch)"); return -1 }
    return Int64(box.batch.length)
}

@_cdecl("am_lakehouse_batch_column_name")
public func am_lakehouse_batch_column_name(_ b: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let box = lhBatch(b), i >= 0, Int(i) < box.names.count else { return nil }
    return UnsafePointer(box.names[Int(i)])
}

@_cdecl("am_lakehouse_batch_column")
public func am_lakehouse_batch_column(_ b: OpaquePointer?, _ i: Int64,
                                      _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let box = lhBatch(b) else { return lhBadArgument("am_lakehouse_batch_column", "`b` is NULL (no batch)") }
    guard let out else { return lhBadArgument("am_lakehouse_batch_column", "`out` is NULL") }
    guard i >= 0, Int(i) < box.batch.columnCount else {
        return lhBadArgument("am_lakehouse_batch_column", "column index \(i) is outside 0..<\(box.batch.columnCount)")
    }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(box.batch.columns[Int(i)])).toOpaque())
    return 0
}

/// The scan's pruning counters: `out` receives files total, files pruned by partition, files pruned by
/// statistics, files read, manifests total, manifests pruned (6 values).
@_cdecl("am_lakehouse_batch_stats")
public func am_lakehouse_batch_stats(_ b: OpaquePointer?, _ out: UnsafeMutablePointer<Int64>?) -> Int32 {
    guard let box = lhBatch(b) else { return lhBadArgument("am_lakehouse_batch_stats", "`b` is NULL (no batch)") }
    guard let out else { return lhBadArgument("am_lakehouse_batch_stats", "`out` is NULL") }
    let s = box.stats
    let v = [s.filesTotal, s.filesPrunedByPartition, s.filesPrunedByStatistics, s.filesRead, s.manifestsTotal, s.manifestsPruned]
    for (i, x) in v.enumerated() { out[i] = Int64(x) }
    return 0
}

@_cdecl("am_lakehouse_batch_release")
public func am_lakehouse_batch_release(_ b: OpaquePointer?) {
    guard let b else { return }
    Unmanaged<LakehouseBatchBox>.fromOpaque(UnsafeRawPointer(b)).release()
}
