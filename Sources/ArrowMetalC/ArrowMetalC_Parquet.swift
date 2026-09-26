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
private func pqStoreMessage(_ m: String) {
    Thread.current.threadDictionary["ArrowMetalC.lastError"] = m
    Thread.current.threadDictionary[pqErrorKey] = m
}

/// An argument-validation failure. Every non-zero return leaves a message behind, naming the function
/// and the argument at fault, so `am_last_error()` never hands the caller an unrelated earlier failure.
@discardableResult
private func pqBadArgument(_ function: String, _ detail: String) -> Int32 {
    pqStoreMessage("\(function): \(detail)")
    return 2
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
    guard let path else { return pqBadArgument("am_parquet_open", "`path` is NULL") }
    guard let out else { return pqBadArgument("am_parquet_open", "`out` is NULL") }
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

/// The `-1` these accessors return is a failure like any other, so it too leaves a message behind.
private func pqMissingFile(_ function: String) -> Int64 {
    pqStoreMessage("\(function): `f` is NULL (no open file)")
    return -1
}

@_cdecl("am_parquet_num_rows")
public func am_parquet_num_rows(_ f: OpaquePointer?) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_num_rows") }
    return file.numRows
}

@_cdecl("am_parquet_num_row_groups")
public func am_parquet_num_row_groups(_ f: OpaquePointer?) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_num_row_groups") }
    return Int64(file.rowGroupCount)
}

@_cdecl("am_parquet_row_group_rows")
public func am_parquet_row_group_rows(_ f: OpaquePointer?, _ i: Int64) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_row_group_rows") }
    guard i >= 0, Int(i) < file.metadata.rowGroups.count else {
        pqStoreMessage("am_parquet_row_group_rows: row group \(i) is outside 0..<\(file.metadata.rowGroups.count)")
        return -1
    }
    return file.metadata.rowGroups[Int(i)].numRows
}

@_cdecl("am_parquet_num_columns")
public func am_parquet_num_columns(_ f: OpaquePointer?) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_num_columns") }
    return Int64(file.fields.count)
}

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
    guard let file = pqFile(f) else { return pqBadArgument("am_parquet_read_ex", "`f` is NULL (no open file)") }
    guard let out else { return pqBadArgument("am_parquet_read_ex", "`out` is NULL") }
    guard nColumns >= 0 else {
        return pqBadArgument("am_parquet_read_ex", "`n_columns` is \(nColumns), which is negative")
    }
    guard nRowGroups >= 0 else {
        return pqBadArgument("am_parquet_read_ex", "`n_row_groups` is \(nRowGroups), which is negative")
    }
    guard rowGroups != nil || nRowGroups == 0 else {
        return pqBadArgument("am_parquet_read_ex",
                             "`row_groups` is NULL but `n_row_groups` is \(nRowGroups)")
    }
    guard columns != nil || nColumns == 0 else {
        return pqBadArgument("am_parquet_read_ex", "`columns` is NULL but `n_columns` is \(nColumns)")
    }
    do {
        // A NULL `columns` means "every column"; a non-NULL one means "exactly these", including none of
        // them. Folding the two together made `read(columns=[])` silently return the whole table.
        var names: [String]? = nil
        if let columns {
            names = try (0..<Int(nColumns)).map { i in
                guard let n = columns[i] else {
                    throw ParquetError.malformed("am_parquet_read_ex: `columns`[\(i)] is NULL")
                }
                return String(cString: n)
            }
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
    guard let file = pqFile(f) else {
        pqBadArgument("am_parquet_selected_row_groups", "`f` is NULL (no open file)")
        return -1
    }
    guard cap >= 0 else {
        pqBadArgument("am_parquet_selected_row_groups", "`cap` is \(cap), which is negative")
        return -1
    }
    guard out != nil || cap == 0 else {
        pqBadArgument("am_parquet_selected_row_groups", "`out` is NULL but `cap` is \(cap)")
        return -1
    }
    do {
        // A filter string that will not parse is an error here exactly as it is in am_parquet_read_ex;
        // silently dropping it used to report every row group as selected.
        let parsed = try filters.map { try parseFilters(String(cString: $0)) } ?? []
        let opts = ParquetReadOptions(filters: parsed)
        let groups = try file.selectedRowGroups(opts)
        if let out { for (i, g) in groups.enumerated() where i < Int(cap) { out[i] = Int64(g) } }
        return Int64(groups.count)
    } catch { pqStoreError(error); return -1 }
}

func parseFilters(_ text: String) throws -> [ParquetFilter] {
    // `name<op><literal>` items separated by `;`. The operator is the first one after the name, so a
    // literal may hold operator characters; a double-quoted literal may also hold `;`, and inside it
    // `\"` and `\\` stand for a quote and a backslash (any other backslash is kept as it is). An integer
    // above Int64.max is kept exact as an unsigned value.
    var out: [ParquetFilter] = []
    let chars = Array(text)
    var i = 0
    func skipSpaces() { while i < chars.count, chars[i] == " " || chars[i] == "\t" { i += 1 } }
    while i < chars.count {
        let itemStart = i
        // The name: everything up to the first operator character.
        while i < chars.count, !"=!<>;".contains(chars[i]) { i += 1 }
        let name = String(chars[itemStart..<i]).trimmingCharacters(in: .whitespaces)
        if i >= chars.count || chars[i] == ";" {
            if !name.isEmpty { throw ParquetError.malformed("filter \"\(name)\" has no comparison operator") }
            i += 1
            continue
        }
        let two = i + 1 < chars.count ? String(chars[i...(i + 1)]) : ""
        let op: ParquetFilter.Op
        if let o = ParquetFilter.Op(rawValue: two), two.count == 2 { op = o; i += 2 }
        else if let o = ParquetFilter.Op(rawValue: String(chars[i])) { op = o; i += 1 }
        else {
            throw ParquetError.malformed("filter \"\(String(chars[itemStart..<Swift.min(i + 2, chars.count)]))\" has no comparison operator")
        }
        skipSpaces()
        // A quoted string, read to its closing quote. One that is not closed, or is followed by more text
        // before the `;`, reads as it did before escapes existed: the text up to the `;`, quotes stripped.
        var quoted: String? = nil
        if i < chars.count, chars[i] == "\"" {
            var s = ""
            var j = i + 1
            var closed = false
            while j < chars.count {
                let c = chars[j]
                if c == "\\", j + 1 < chars.count, chars[j + 1] == "\"" || chars[j + 1] == "\\" {
                    s.append(chars[j + 1]); j += 2; continue
                }
                if c == "\"" { closed = true; j += 1; break }
                s.append(c); j += 1
            }
            while j < chars.count, chars[j] == " " || chars[j] == "\t" { j += 1 }
            if closed, j >= chars.count || chars[j] == ";" { quoted = s; i = j }
        }
        let value: ParquetFilter.Value
        if let quoted {
            value = .string(quoted)
        } else {
            let litStart = i
            while i < chars.count, chars[i] != ";" { i += 1 }
            let literal = String(chars[litStart..<i]).trimmingCharacters(in: .whitespaces)
            if literal.hasPrefix("\"") && literal.hasSuffix("\"") && literal.count >= 2 {
                value = .string(String(literal.dropFirst().dropLast()))
            } else if let v = Int64(literal) {
                value = .int(v)
            } else if let u = UInt64(literal) {
                value = .uint(u)
            } else if let d = Double(literal) {
                value = .double(d)
            } else {
                value = .string(literal)
            }
        }
        out.append(ParquetFilter(column: name, op: op, value: value))
        i += 1      // past the `;`, if any
    }
    return out
}

// MARK: - Batches

@_cdecl("am_parquet_batch_columns")
public func am_parquet_batch_columns(_ b: OpaquePointer?) -> Int64 {
    guard let box = pqBatch(b) else {
        pqStoreMessage("am_parquet_batch_columns: `b` is NULL (no batch)")
        return -1
    }
    return Int64(box.batch.columnCount)
}

@_cdecl("am_parquet_batch_rows")
public func am_parquet_batch_rows(_ b: OpaquePointer?) -> Int64 {
    guard let box = pqBatch(b) else {
        pqStoreMessage("am_parquet_batch_rows: `b` is NULL (no batch)")
        return -1
    }
    return Int64(box.batch.length)
}

@_cdecl("am_parquet_batch_column_name")
public func am_parquet_batch_column_name(_ b: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let box = pqBatch(b), i >= 0, Int(i) < box.names.count else { return nil }
    return UnsafePointer(box.names[Int(i)])
}

@_cdecl("am_parquet_batch_column")
public func am_parquet_batch_column(_ b: OpaquePointer?, _ i: Int64,
                                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let box = pqBatch(b) else { return pqBadArgument("am_parquet_batch_column", "`b` is NULL (no batch)") }
    guard let out else { return pqBadArgument("am_parquet_batch_column", "`out` is NULL") }
    guard i >= 0, Int(i) < box.batch.columnCount else {
        return pqBadArgument("am_parquet_batch_column",
                             "column index \(i) is outside 0..<\(box.batch.columnCount)")
    }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(box.batch.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_parquet_batch_release")
public func am_parquet_batch_release(_ b: OpaquePointer?) {
    guard let b else { return }
    Unmanaged<BatchBox>.fromOpaque(UnsafeRawPointer(b)).release()
}

// MARK: - ARROW:schema metadata (docs/PARQUET.md, "The stored Arrow schema")

/// Copies a metadata set into `out` as the C Data Interface metadata blob and returns its size; with a
/// NULL `out` or a small `cap` nothing is copied and the size is still returned, so a caller asks twice.
private func pqMetadataBlob(_ m: ArrowSchemaMetadata, _ out: UnsafeMutablePointer<UInt8>?, _ cap: Int64) -> Int64 {
    guard !m.isEmpty else { return 0 }
    let blob = m.encoded()
    if let out, Int(cap) >= blob.count {
        blob.withUnsafeBufferPointer { out.update(from: $0.baseAddress!, count: blob.count) }
    }
    return Int64(blob.count)
}

@_cdecl("am_parquet_field_metadata")
public func am_parquet_field_metadata(_ f: OpaquePointer?, _ column: UnsafePointer<CChar>?,
                                      _ out: UnsafeMutablePointer<UInt8>?, _ cap: Int64) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_field_metadata") }
    guard let column else {
        pqBadArgument("am_parquet_field_metadata", "`column` is NULL")
        return -1
    }
    guard cap >= 0 else {
        pqBadArgument("am_parquet_field_metadata", "`cap` is \(cap), which is negative")
        return -1
    }
    return pqMetadataBlob(file.arrowFieldMetadata(column: String(cString: column)), out, cap)
}

@_cdecl("am_parquet_schema_metadata")
public func am_parquet_schema_metadata(_ f: OpaquePointer?, _ out: UnsafeMutablePointer<UInt8>?,
                                       _ cap: Int64) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_schema_metadata") }
    guard cap >= 0 else {
        pqBadArgument("am_parquet_schema_metadata", "`cap` is \(cap), which is negative")
        return -1
    }
    return pqMetadataBlob(file.arrowSchemaMetadata, out, cap)
}

// MARK: - Page index (docs/PARQUET.md, "Page-level skipping")

@_cdecl("am_parquet_set_page_index")
public func am_parquet_set_page_index(_ f: OpaquePointer?, _ enabled: Int32) -> Int32 {
    guard let file = pqFile(f) else { return pqBadArgument("am_parquet_set_page_index", "`f` is NULL (no open file)") }
    file.usePageIndex = enabled != 0
    return 0
}

@_cdecl("am_parquet_set_bloom_filters")
public func am_parquet_set_bloom_filters(_ f: OpaquePointer?, _ enabled: Int32) -> Int32 {
    guard let file = pqFile(f) else { return pqBadArgument("am_parquet_set_bloom_filters", "`f` is NULL (no open file)") }
    file.useBloomFilters = enabled != 0
    return 0
}

/// `[row groups read, row groups skipped by statistics, row groups skipped by the page index,
/// data pages decoded, data pages skipped, rows, row groups skipped by bloom filters]` of the most recent
/// read; returns 7.
@_cdecl("am_parquet_last_read_stats")
public func am_parquet_last_read_stats(_ f: OpaquePointer?, _ out: UnsafeMutablePointer<Int64>?, _ cap: Int64) -> Int64 {
    guard let file = pqFile(f) else { return pqMissingFile("am_parquet_last_read_stats") }
    guard cap >= 0 else {
        pqBadArgument("am_parquet_last_read_stats", "`cap` is \(cap), which is negative")
        return -1
    }
    guard out != nil || cap == 0 else {
        pqBadArgument("am_parquet_last_read_stats", "`out` is NULL but `cap` is \(cap)")
        return -1
    }
    let s = file.lastReadStatistics
    let values = [s.rowGroupsRead, s.rowGroupsSkippedByStatistics, s.rowGroupsSkippedByPageIndex,
                  s.pagesDecoded, s.pagesSkipped, s.rows, s.rowGroupsSkippedByBloomFilter]
    if let out { for (i, v) in values.enumerated() where i < Int(cap) { out[i] = Int64(v) } }
    return Int64(values.count)
}

/// How many nulls a top-level column holds according to the file's metadata alone: 0 for a `required`
/// column, otherwise the sum of every row group's `null_count` statistic. -1 when that is not known: a
/// row group without the statistic, or a column that is not a top-level leaf (a struct, list or map).
/// -2 on a bad argument, with a message behind it.
@_cdecl("am_parquet_column_null_count")
public func am_parquet_column_null_count(_ f: OpaquePointer?, _ column: UnsafePointer<CChar>?) -> Int64 {
    guard let file = pqFile(f) else {
        pqStoreMessage("am_parquet_column_null_count: `f` is NULL (no open file)")
        return -2
    }
    guard let column else {
        pqBadArgument("am_parquet_column_null_count", "`column` is NULL")
        return -2
    }
    let name = String(cString: column)
    guard let field = file.fields.first(where: { $0.name == name }) else {
        pqBadArgument("am_parquet_column_null_count", "no top-level column named \"\(name)\"")
        return -2
    }
    guard case .leaf(let leaf) = field.kind else { return -1 }
    if !leaf.isNullable { return 0 }
    var total: Int64 = 0
    for rg in file.metadata.rowGroups {
        guard leaf.index < rg.columns.count, let n = rg.columns[leaf.index].meta.statistics?.nullCount,
              n >= 0 else { return -1 }
        total += n
    }
    return total
}
