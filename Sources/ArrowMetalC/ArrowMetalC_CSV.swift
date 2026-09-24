import Foundation
import CArrowABI
import ArrowMetal

// C ABI over the GPU CSV reader (docs/CSV.md). The reader handle owns the path and a copy of
// the options; a read returns a batch handle with the same accessor shape as am_parquet_batch, whose
// columns are handed out as ordinary am_array handles.

/// Mirrors `am_csv_options` in include/arrowmetal.h field for field (every field naturally aligned,
/// so the layout is the same in C and Swift).
struct AMCSVOptions {
    var delimiter: Int32
    var quoteChar: Int32
    var doubleQuote: Int32
    var decimalPoint: Int32
    var skipRows: Int64
    var skipRowsAfterNames: Int64
    var autogenerateColumnNames: Int32
    var includeMissingColumns: Int32
    var columnNames: UnsafePointer<UnsafePointer<CChar>?>?
    var nColumnNames: Int64
    var includeColumns: UnsafePointer<UnsafePointer<CChar>?>?
    var nIncludeColumns: Int64
    var columnTypeNames: UnsafePointer<UnsafePointer<CChar>?>?
    var columnTypeFormats: UnsafePointer<UnsafePointer<CChar>?>?
    var nColumnTypes: Int64
    var nullValues: UnsafePointer<UnsafePointer<CChar>?>?
    var nNullValues: Int64
    var trueValues: UnsafePointer<UnsafePointer<CChar>?>?
    var nTrueValues: Int64
    var falseValues: UnsafePointer<UnsafePointer<CChar>?>?
    var nFalseValues: Int64
    var stringsCanBeNull: Int32
    var quotedStringsCanBeNull: Int32
    var checkUTF8: Int32
    var fileAccess: Int32
    var scanBlockBytes: Int64
}

private final class CSVReaderBox { let r: CSVReader; init(_ r: CSVReader) { self.r = r } }
private final class CSVBatchBox {
    let batch: MetalRecordBatch
    let rows: Int
    /// Each name's UTF-8 bytes plus a terminating NUL; a name may itself contain NUL bytes, so its
    /// length travels separately (am_csv_batch_column_name_length).
    var names: [(ptr: UnsafeMutablePointer<CChar>, length: Int)] = []
    init(_ b: MetalRecordBatch, rows: Int) {
        batch = b
        self.rows = rows
        names = b.names.map { csvCopyBytes($0) }
    }
    deinit { for n in names { free(n.ptr) } }
}

/// A malloc'd copy of `s`'s UTF-8 bytes with a NUL after them, and the byte count without it.
private func csvCopyBytes(_ s: String) -> (ptr: UnsafeMutablePointer<CChar>, length: Int) {
    let u = Array(s.utf8)
    let p = malloc(u.count + 1)!.assumingMemoryBound(to: CChar.self)
    u.withUnsafeBytes { if $0.count > 0 { memcpy(p, $0.baseAddress!, $0.count) } }
    p[u.count] = 0
    return (p, u.count)
}

private func csvStoreMessage(_ m: String) { Thread.current.threadDictionary["ArrowMetalC.lastError"] = m }

@discardableResult
private func csvBadArgument(_ function: String, _ detail: String) -> Int32 {
    csvStoreMessage("\(function): \(detail)")
    return 2
}

@inline(__always) private func csvReader(_ p: OpaquePointer?) -> CSVReader? {
    guard let p else { return nil }
    return Unmanaged<CSVReaderBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().r
}
@inline(__always) private func csvBatch(_ p: OpaquePointer?) -> CSVBatchBox? {
    guard let p else { return nil }
    return Unmanaged<CSVBatchBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

@_cdecl("am_csv_options_init")
public func am_csv_options_init(_ o: UnsafeMutableRawPointer?) {
    guard let o else { return }
    let p = o.assumingMemoryBound(to: AMCSVOptions.self)
    p.pointee = AMCSVOptions(
        delimiter: 0x2C, quoteChar: 0x22, doubleQuote: 1, decimalPoint: 0x2E,
        skipRows: 0, skipRowsAfterNames: 0, autogenerateColumnNames: 0, includeMissingColumns: 0,
        columnNames: nil, nColumnNames: 0, includeColumns: nil, nIncludeColumns: 0,
        columnTypeNames: nil, columnTypeFormats: nil, nColumnTypes: 0,
        nullValues: nil, nNullValues: 0, trueValues: nil, nTrueValues: 0, falseValues: nil, nFalseValues: 0,
        stringsCanBeNull: 0, quotedStringsCanBeNull: 1, checkUTF8: 1, fileAccess: 0, scanBlockBytes: 0)
}

private func strings(_ p: UnsafePointer<UnsafePointer<CChar>?>?, _ n: Int64, _ what: String) throws -> [String]? {
    guard let p else {
        if n != 0 { throw CSVError.invalidOptions("am_csv_open: `\(what)` is NULL but its count is \(n)") }
        return nil
    }
    guard n >= 0 else { throw CSVError.invalidOptions("am_csv_open: the count of `\(what)` is \(n), which is negative") }
    return try (0..<Int(n)).map { i in
        guard let s = p[i] else { throw CSVError.invalidOptions("am_csv_open: `\(what)`[\(i)] is NULL") }
        return String(cString: s)
    }
}

private func byte(_ v: Int32, _ what: String) throws -> UInt8 {
    guard v >= 0 && v <= 127 else { throw CSVError.invalidOptions("am_csv_open: `\(what)` must be an ASCII character, got \(v)") }
    return UInt8(v)
}

func csvOptions(_ c: AMCSVOptions) throws -> CSVReadOptions {
    var o = CSVReadOptions()
    o.delimiter = try byte(c.delimiter, "delimiter")
    o.quoteChar = c.quoteChar < 0 ? nil : try byte(c.quoteChar, "quote_char")
    o.doubleQuote = c.doubleQuote != 0
    o.decimalPoint = try byte(c.decimalPoint, "decimal_point")
    o.skipRows = Int(c.skipRows)
    o.skipRowsAfterNames = Int(c.skipRowsAfterNames)
    o.autogenerateColumnNames = c.autogenerateColumnNames != 0
    o.includeMissingColumns = c.includeMissingColumns != 0
    o.columnNames = try strings(c.columnNames, c.nColumnNames, "column_names")
    o.includeColumns = try strings(c.includeColumns, c.nIncludeColumns, "include_columns")
    let tn = try strings(c.columnTypeNames, c.nColumnTypes, "column_type_names") ?? []
    let tf = try strings(c.columnTypeFormats, c.nColumnTypes, "column_type_formats") ?? []
    guard tn.count == tf.count else { throw CSVError.invalidOptions("am_csv_open: column_type_names and column_type_formats differ in length") }
    for (n, f) in zip(tn, tf) {
        guard let t = CSVColumnType(format: f) else {
            throw CSVError.invalidOptions("am_csv_open: column type \"\(f)\" for column \(n) is not supported (docs/CSV.md lists the types)")
        }
        o.columnTypes[n] = t
    }
    if let v = try strings(c.nullValues, c.nNullValues, "null_values") { o.nullValues = v }
    if let v = try strings(c.trueValues, c.nTrueValues, "true_values") { o.trueValues = v }
    if let v = try strings(c.falseValues, c.nFalseValues, "false_values") { o.falseValues = v }
    o.stringsCanBeNull = c.stringsCanBeNull != 0
    o.quotedStringsCanBeNull = c.quotedStringsCanBeNull != 0
    o.checkUTF8 = c.checkUTF8 != 0
    if c.scanBlockBytes > 0 { o.scanBlockBytes = Int(c.scanBlockBytes) }
    switch c.fileAccess {
    case 0: o.fileAccess = .read
    case 1: o.fileAccess = .map
    default: throw CSVError.invalidOptions("am_csv_open: `file_access` must be 0 (read) or 1 (map), got \(c.fileAccess)")
    }
    return o
}

@_cdecl("am_csv_open")
public func am_csv_open(_ path: UnsafePointer<CChar>?, _ options: UnsafeRawPointer?,
                        _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let path else { return csvBadArgument("am_csv_open", "`path` is NULL") }
    guard let out else { return csvBadArgument("am_csv_open", "`out` is NULL") }
    do {
        let o = try options.map { try csvOptions($0.assumingMemoryBound(to: AMCSVOptions.self).pointee) } ?? CSVReadOptions()
        let r = try CSVReader(path: String(cString: path), options: o)
        out.pointee = OpaquePointer(Unmanaged.passRetained(CSVReaderBox(r)).toOpaque())
        return 0
    } catch { csvStoreMessage("\(error)"); return 1 }
}

@_cdecl("am_csv_close")
public func am_csv_close(_ r: OpaquePointer?) {
    guard let r else { return }
    Unmanaged<CSVReaderBox>.fromOpaque(UnsafeRawPointer(r)).release()
}

@_cdecl("am_csv_read")
public func am_csv_read(_ r: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let reader = csvReader(r) else { return csvBadArgument("am_csv_read", "`r` is NULL (no open reader)") }
    guard let out else { return csvBadArgument("am_csv_read", "`out` is NULL") }
    do {
        let batch = try reader.read()
        out.pointee = OpaquePointer(Unmanaged.passRetained(CSVBatchBox(batch, rows: reader.numRows)).toOpaque())
        return 0
    } catch { csvStoreMessage("\(error)"); return 1 }
}

@_cdecl("am_csv_batch_rows")
public func am_csv_batch_rows(_ b: OpaquePointer?) -> Int64 {
    guard let box = csvBatch(b) else { csvStoreMessage("am_csv_batch_rows: `b` is NULL (no batch)"); return -1 }
    return Int64(box.rows)
}

@_cdecl("am_csv_batch_columns")
public func am_csv_batch_columns(_ b: OpaquePointer?) -> Int64 {
    guard let box = csvBatch(b) else { csvStoreMessage("am_csv_batch_columns: `b` is NULL (no batch)"); return -1 }
    return Int64(box.batch.columnCount)
}

@_cdecl("am_csv_batch_column_name")
public func am_csv_batch_column_name(_ b: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let box = csvBatch(b), i >= 0, Int(i) < box.names.count else { return nil }
    return UnsafePointer(box.names[Int(i)].ptr)
}

@_cdecl("am_csv_batch_column_name_length")
public func am_csv_batch_column_name_length(_ b: OpaquePointer?, _ i: Int64) -> Int64 {
    guard let box = csvBatch(b) else { csvStoreMessage("am_csv_batch_column_name_length: `b` is NULL (no batch)"); return -1 }
    guard i >= 0, Int(i) < box.names.count else {
        csvStoreMessage("am_csv_batch_column_name_length: column index \(i) is outside 0..<\(box.names.count)")
        return -1
    }
    return Int64(box.names[Int(i)].length)
}

/// The same message as am_last_error(), with its byte length, so a message quoting a value that holds
/// a NUL byte arrives whole. The buffer is per thread and valid until the next call on that thread.
@_cdecl("am_csv_last_error")
public func am_csv_last_error(_ length: UnsafeMutablePointer<Int64>?) -> UnsafePointer<CChar>? {
    let key = "ArrowMetalC.csvLastErrorBytes"
    let d = Thread.current.threadDictionary
    if let old = d[key] as? UnsafeMutablePointer<CChar> { free(old) }
    let (p, n) = csvCopyBytes((d["ArrowMetalC.lastError"] as? String) ?? "")
    d[key] = p
    length?.pointee = Int64(n)
    return UnsafePointer(p)
}

@_cdecl("am_csv_batch_column")
public func am_csv_batch_column(_ b: OpaquePointer?, _ i: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let box = csvBatch(b) else { return csvBadArgument("am_csv_batch_column", "`b` is NULL (no batch)") }
    guard let out else { return csvBadArgument("am_csv_batch_column", "`out` is NULL") }
    guard i >= 0, Int(i) < box.batch.columnCount else {
        return csvBadArgument("am_csv_batch_column", "column index \(i) is outside 0..<\(box.batch.columnCount)")
    }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(box.batch.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_csv_batch_release")
public func am_csv_batch_release(_ b: OpaquePointer?) {
    guard let b else { return }
    Unmanaged<CSVBatchBox>.fromOpaque(UnsafeRawPointer(b)).release()
}
