import Foundation
import CArrowABI
import ArrowMetal

// C ABI over the Parquet reader. A file handle is a retained box; a read returns a batch handle whose
// columns are handed out one at a time as ordinary `am_array` handles, so callers already speaking the
// Arrow C Data Interface need nothing new.

private final class ParquetBox { let f: ParquetFile; init(_ f: ParquetFile) { self.f = f } }
private final class BatchBox {
    let batch: MetalRecordBatch
    var names: [UnsafeMutablePointer<CChar>] = []
    init(_ b: MetalRecordBatch) {
        batch = b
        names = b.names.map { strdup($0)! }
    }
    deinit { for n in names { free(n) } }
}

private let pqErrorKey = "ArrowMetalC.parquetError"
private func pqSetError(_ e: Error) { Thread.current.threadDictionary[pqErrorKey] = "\(e)" }
private func pqStoreError(_ e: Error) {
    // Share the generic error slot so am_last_error() works for Parquet calls too.
    Thread.current.threadDictionary["ArrowMetalC.lastError"] = "\(e)"
    pqSetError(e)
}

@inline(__always) private func pqFile(_ p: OpaquePointer?) -> ParquetFile? {
    guard let p else { return nil }
    return Unmanaged<ParquetBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().f
}
@inline(__always) private func pqBatch(_ p: OpaquePointer?) -> BatchBox? {
    guard let p else { return nil }
    return Unmanaged<BatchBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

/// Cached C strings for schema names and formats (valid for the life of the file handle).
private var pqStrings: [String: UnsafeMutablePointer<CChar>] = [:]
private let pqStringLock = NSLock()
private func pqCString(_ s: String) -> UnsafePointer<CChar> {
    pqStringLock.lock(); defer { pqStringLock.unlock() }
    if let c = pqStrings[s] { return UnsafePointer(c) }
    let c = strdup(s)!
    pqStrings[s] = c
    return UnsafePointer(c)
}

@_cdecl("am_parquet_open")
public func am_parquet_open(_ path: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let path, let out else { return 2 }
    do {
        let f = try ParquetFile(path: String(cString: path))
        out.pointee = OpaquePointer(Unmanaged.passRetained(ParquetBox(f)).toOpaque())
        return 0
    } catch { pqStoreError(error); return 1 }
}

@_cdecl("am_parquet_close")
public func am_parquet_close(_ f: OpaquePointer?) {
    guard let f else { return }
    Unmanaged<ParquetBox>.fromOpaque(UnsafeRawPointer(f)).release()
}

@_cdecl("am_parquet_num_rows")
public func am_parquet_num_rows(_ f: OpaquePointer?) -> Int64 { pqFile(f)?.numRows ?? -1 }

@_cdecl("am_parquet_num_row_groups")
public func am_parquet_num_row_groups(_ f: OpaquePointer?) -> Int64 { Int64(pqFile(f)?.rowGroupCount ?? -1) }

@_cdecl("am_parquet_row_group_rows")
public func am_parquet_row_group_rows(_ f: OpaquePointer?, _ i: Int64) -> Int64 {
    guard let file = pqFile(f), i >= 0, Int(i) < file.metadata.rowGroups.count else { return -1 }
    return file.metadata.rowGroups[Int(i)].numRows
}

@_cdecl("am_parquet_num_columns")
public func am_parquet_num_columns(_ f: OpaquePointer?) -> Int64 { Int64(pqFile(f)?.fields.count ?? -1) }

@_cdecl("am_parquet_column_name")
public func am_parquet_column_name(_ f: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let file = pqFile(f), i >= 0, Int(i) < file.fields.count else { return nil }
    return pqCString(file.fields[Int(i)].name)
}

/// The Parquet physical type and logical annotation of a column, as a short human-readable string
/// ("INT64", "BYTE_ARRAY/STRING", "list<INT64>"). The Arrow type of the *result* comes from am_format.
@_cdecl("am_parquet_column_type")
public func am_parquet_column_type(_ f: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let file = pqFile(f), i >= 0, Int(i) < file.fields.count else { return nil }
    return pqCString(describe(file.fields[Int(i)]))
}

private func describe(_ f: ParquetField) -> String {
    switch f.kind {
    case .leaf(let l):
        var s = l.physical.name
        switch l.logicalType {
        case .none: break
        case .string: s += "/STRING"
        case .decimal(let p, let sc): s += "/DECIMAL(\(p),\(sc))"
        case .date: s += "/DATE"
        case .time(_, let u): s += "/TIME(\(u))"
        case .timestamp(_, let u): s += "/TIMESTAMP(\(u))"
        case .integer(let b, let sg): s += "/INT(\(b),\(sg ? "signed" : "unsigned"))"
        default: s += "/\(l.logicalType)"
        }
        return s
    case .list(let e, _): return "list<\(describe(e))>"
    case .group(let cs): return "struct<\(cs.map(describe).joined(separator: ","))>"
    }
}

@_cdecl("am_parquet_codec")
public func am_parquet_codec(_ f: OpaquePointer?, _ rowGroup: Int64, _ column: Int64) -> UnsafePointer<CChar>? {
    guard let file = pqFile(f), rowGroup >= 0, Int(rowGroup) < file.metadata.rowGroups.count else { return nil }
    let rg = file.metadata.rowGroups[Int(rowGroup)]
    guard column >= 0, Int(column) < rg.columns.count else { return nil }
    return pqCString(rg.columns[Int(column)].meta.codec.name)
}

@_cdecl("am_parquet_encodings")
public func am_parquet_encodings(_ f: OpaquePointer?, _ rowGroup: Int64, _ column: Int64) -> UnsafePointer<CChar>? {
    guard let file = pqFile(f), rowGroup >= 0, Int(rowGroup) < file.metadata.rowGroups.count else { return nil }
    let rg = file.metadata.rowGroups[Int(rowGroup)]
    guard column >= 0, Int(column) < rg.columns.count else { return nil }
    return pqCString(rg.columns[Int(column)].meta.encodings.map { $0.name }.joined(separator: ","))
}

// MARK: - Reading

@_cdecl("am_parquet_read")
public func am_parquet_read(_ f: OpaquePointer?, _ columns: UnsafePointer<UnsafePointer<CChar>?>?,
                            _ nColumns: Int64, _ rowGroup: Int64,
                            _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard rowGroup >= 0 else {
        return am_parquet_read_ex(f, columns, nColumns, nil, 0, nil, 1, out)
    }
    var one = rowGroup
    return withUnsafePointer(to: &one) { p in
        am_parquet_read_ex(f, columns, nColumns, p, 1, nil, 1, out)
    }
}

/// Full read: projection, row-group selection, statistics filters and the dictionary switch.
///
/// `filters` is a semicolon-separated list of `name<op><literal>` with op in `== != < <= > >=`; the
/// literal is an integer, a float, or a double-quoted string. An empty or NULL string means no filter.
@_cdecl("am_parquet_read_ex")
public func am_parquet_read_ex(_ f: OpaquePointer?, _ columns: UnsafePointer<UnsafePointer<CChar>?>?,
                               _ nColumns: Int64, _ rowGroups: UnsafePointer<Int64>?, _ nRowGroups: Int64,
                               _ filters: UnsafePointer<CChar>?, _ dictionary: Int32,
                               _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let file = pqFile(f), let out else { return 2 }
    do {
        var names: [String]? = nil
        if let columns, nColumns > 0 {
            names = (0..<Int(nColumns)).compactMap { columns[$0].map { String(cString: $0) } }
        }
        var groups: [Int]? = nil
        if let rowGroups, nRowGroups > 0 {
            groups = (0..<Int(nRowGroups)).map { Int(rowGroups[$0]) }
        }
        var parsed: [ParquetFilter] = []
        if let filters {
            parsed = try parseFilters(String(cString: filters))
        }
        let opts = ParquetReadOptions(columns: names, rowGroups: groups,
                                      dictionaryEncoded: dictionary != 0, filters: parsed)
        let batch = try file.read(opts)
        out.pointee = OpaquePointer(Unmanaged.passRetained(BatchBox(batch)).toOpaque())
        return 0
    } catch { pqStoreError(error); return 1 }
}

/// Row groups a filter set keeps, without reading anything (for reporting how much was skipped).
@_cdecl("am_parquet_selected_row_groups")
public func am_parquet_selected_row_groups(_ f: OpaquePointer?, _ filters: UnsafePointer<CChar>?,
                                           _ out: UnsafeMutablePointer<Int64>?, _ cap: Int64) -> Int64 {
    guard let file = pqFile(f) else { return -1 }
    do {
        let parsed = filters.map { try? parseFilters(String(cString: $0)) } ?? []
        let opts = ParquetReadOptions(filters: parsed ?? [])
        let groups = try file.selectedRowGroups(opts)
        if let out { for (i, g) in groups.enumerated() where i < Int(cap) { out[i] = Int64(g) } }
        return Int64(groups.count)
    } catch { pqStoreError(error); return -1 }
}

func parseFilters(_ text: String) throws -> [ParquetFilter] {
    var out: [ParquetFilter] = []
    for part in text.split(separator: ";") {
        let s = part.trimmingCharacters(in: .whitespaces)
        if s.isEmpty { continue }
        var op: ParquetFilter.Op? = nil
        var idx: String.Index? = nil
        for candidate in ["==", "!=", "<=", ">=", "<", ">"] {
            if let r = s.range(of: candidate) {
                op = ParquetFilter.Op(rawValue: candidate)
                idx = r.lowerBound
                break
            }
        }
        guard let op, let idx, let r = s.range(of: op.rawValue) else {
            throw ParquetError.malformed("filter \"\(s)\" has no comparison operator")
        }
        let name = String(s[s.startIndex..<idx]).trimmingCharacters(in: .whitespaces)
        let literal = String(s[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        let value: ParquetFilter.Value
        if literal.hasPrefix("\"") && literal.hasSuffix("\"") && literal.count >= 2 {
            value = .string(String(literal.dropFirst().dropLast()))
        } else if let i = Int64(literal) {
            value = .int(i)
        } else if let d = Double(literal) {
            value = .double(d)
        } else {
            value = .string(literal)
        }
        out.append(ParquetFilter(column: name, op: op, value: value))
    }
    return out
}

// MARK: - Batches

@_cdecl("am_parquet_batch_columns")
public func am_parquet_batch_columns(_ b: OpaquePointer?) -> Int64 { Int64(pqBatch(b)?.batch.columnCount ?? -1) }

@_cdecl("am_parquet_batch_rows")
public func am_parquet_batch_rows(_ b: OpaquePointer?) -> Int64 { Int64(pqBatch(b)?.batch.length ?? -1) }

@_cdecl("am_parquet_batch_column_name")
public func am_parquet_batch_column_name(_ b: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let box = pqBatch(b), i >= 0, Int(i) < box.names.count else { return nil }
    return UnsafePointer(box.names[Int(i)])
}

@_cdecl("am_parquet_batch_column")
public func am_parquet_batch_column(_ b: OpaquePointer?, _ i: Int64,
                                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let box = pqBatch(b), let out, i >= 0, Int(i) < box.batch.columnCount else { return 2 }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(box.batch.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_parquet_batch_release")
public func am_parquet_batch_release(_ b: OpaquePointer?) {
    guard let b else { return }
    Unmanaged<BatchBox>.fromOpaque(UnsafeRawPointer(b)).release()
}
