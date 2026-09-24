import Foundation
import CArrowABI
import ArrowMetal

// C ABI over the GPU JSON reader (docs/JSON.md). A file handle is a retained box around a loaded
// `JSONReader`; a read returns a batch handle with the same accessor shape as `am_parquet_batch`, whose
// columns are handed out one at a time as ordinary `am_array` handles.

private final class JSONFileBox { let r: JSONReader; init(_ r: JSONReader) { self.r = r } }
private final class JSONBatchBox {
    let table: JSONTable
    var names: [UnsafeMutablePointer<CChar>] = []
    /// Per column, every field name the column carries (its own, then each struct field of its type in
    /// depth-first order), each as a 4-byte little-endian length and the UTF-8 bytes: a key holding
    /// `\u0000` survives here, where the C string of `names` and the C Data Interface stop at the NUL.
    var nameBlobs: [UnsafeMutableBufferPointer<UInt8>] = []
    init(_ t: JSONTable) {
        table = t
        names = t.names.map { strdup($0)! }
        nameBlobs = zip(t.names, t.columns).map { name, col in
            var blob: [UInt8] = []
            func put(_ s: String) {
                let u = Array(s.utf8)
                withUnsafeBytes(of: UInt32(u.count).littleEndian) { blob += $0 }
                blob += u
            }
            func walk(_ a: AnyMetalArray) {
                switch a {
                case .structure(let s): for (n, c) in zip(s.names, s.children) { put(n); walk(c) }
                case .list(let l): walk(l.values)
                default: break
                }
            }
            put(name)
            walk(col)
            let stored = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: blob.count)
            _ = stored.initialize(from: blob)
            return stored
        }
    }
    deinit {
        for n in names { free(n) }
        for b in nameBlobs { b.deallocate() }
    }
}

private func jsonStore(_ m: String) {
    Thread.current.threadDictionary["ArrowMetalC.lastError"] = m
}

@discardableResult
private func jsonBadArgument(_ function: String, _ detail: String) -> Int32 {
    jsonStore("\(function): \(detail)")
    return 2
}

@inline(__always) private func jsonFile(_ p: OpaquePointer?) -> JSONReader? {
    guard let p else { return nil }
    return Unmanaged<JSONFileBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().r
}
@inline(__always) private func jsonBatch(_ p: OpaquePointer?) -> JSONBatchBox? {
    guard let p else { return nil }
    return Unmanaged<JSONBatchBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

@_cdecl("am_json_open")
public func am_json_open(_ path: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let path else { return jsonBadArgument("am_json_open", "`path` is NULL") }
    guard let out else { return jsonBadArgument("am_json_open", "`out` is NULL") }
    do {
        let r = try JSONReader(path: String(cString: path))
        out.pointee = OpaquePointer(Unmanaged.passRetained(JSONFileBox(r)).toOpaque())
        return 0
    } catch { jsonStore("\(error)"); return 1 }
}

@_cdecl("am_json_open_buffer")
public func am_json_open_buffer(_ data: UnsafeRawPointer?, _ length: Int64,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let out else { return jsonBadArgument("am_json_open_buffer", "`out` is NULL") }
    guard length >= 0 else { return jsonBadArgument("am_json_open_buffer", "`length` is \(length), which is negative") }
    guard data != nil || length == 0 else { return jsonBadArgument("am_json_open_buffer", "`data` is NULL but `length` is \(length)") }
    do {
        let r = try JSONReader(buffer: UnsafeRawBufferPointer(start: data, count: Int(length)))
        out.pointee = OpaquePointer(Unmanaged.passRetained(JSONFileBox(r)).toOpaque())
        return 0
    } catch { jsonStore("\(error)"); return 1 }
}

@_cdecl("am_json_close")
public func am_json_close(_ f: OpaquePointer?) {
    guard let f else { return }
    Unmanaged<JSONFileBox>.fromOpaque(UnsafeRawPointer(f)).release()
}

/// The explicit schema arrives as an Arrow C Data Interface schema of struct type ("+s"): its children
/// are the fields. Types outside what the reader converts to are rejected with the field's path.
private func jsonType(_ s: UnsafePointer<ArrowSchema>, path: String) throws -> JSONType {
    guard let fmtC = s.pointee.format else { throw JSONError.unsupported("explicit_schema: \(path) has no format") }
    let fmt = String(cString: fmtC)
    switch fmt {
    case "n": return .null
    case "b": return .boolean
    case "c": return .int8
    case "s": return .int16
    case "i": return .int32
    case "l": return .int64
    case "C": return .uint8
    case "S": return .uint16
    case "I": return .uint32
    case "L": return .uint64
    case "f": return .float32
    case "g": return .float64
    case "u": return .utf8
    case "+l":
        guard s.pointee.n_children == 1, let c = s.pointee.children?[0] else {
            throw JSONError.unsupported("explicit_schema: \(path) is a list without a child")
        }
        return .list(try jsonType(UnsafePointer(c), path: path + "/[]"))
    case "+s":
        return .structure(try jsonFields(s, path: path))
    default:
        if fmt.hasPrefix("ts") && fmt.count >= 4 {
            let chars = Array(fmt)
            let unit: ArrowTemporalUnit
            switch chars[2] {
            case "s": unit = .second
            case "m": unit = .milli
            case "u": unit = .micro
            case "n": unit = .nano
            default: throw JSONError.unsupported("explicit_schema: \(path) has timestamp format \(fmt)")
            }
            let tz = String(fmt.dropFirst(4))
            return .timestamp(unit, timezone: tz.isEmpty ? nil : tz)
        }
        throw JSONError.unsupported("explicit_schema: \(path) has Arrow format \"\(fmt)\", which the JSON reader "
                                    + "does not convert to (see docs/JSON.md for the supported types)")
    }
}

private func jsonFields(_ s: UnsafePointer<ArrowSchema>, path: String) throws -> [JSONField] {
    var out: [JSONField] = []
    for i in 0..<Int(s.pointee.n_children) {
        guard let c = s.pointee.children?[i] else { continue }
        let name = c.pointee.name.map { String(cString: $0) } ?? ""
        out.append(JSONField(name, try jsonType(UnsafePointer(c), path: "\(path)/\(name)")))
    }
    return out
}

/// Reads every record. `explicit_schema` is NULL to infer every field, or a struct-typed ArrowSchema
/// whose children fix the named fields' types (it is only read; the caller keeps ownership).
/// `unexpected_field_behavior`: 0 infer, 1 ignore, 2 error.
@_cdecl("am_json_read")
public func am_json_read(_ f: OpaquePointer?, _ explicitSchema: UnsafePointer<ArrowSchema>?,
                         _ unexpectedFieldBehavior: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let file = jsonFile(f) else { return jsonBadArgument("am_json_read", "`f` is NULL (no open file)") }
    guard let out else { return jsonBadArgument("am_json_read", "`out` is NULL") }
    let behavior: JSONUnexpectedFieldBehavior
    switch unexpectedFieldBehavior {
    case 0: behavior = .infer
    case 1: behavior = .ignore
    case 2: behavior = .error
    default:
        return jsonBadArgument("am_json_read", "`unexpected_field_behavior` is \(unexpectedFieldBehavior); use 0 infer, 1 ignore or 2 error")
    }
    do {
        var schema: [JSONField]? = nil
        if let explicitSchema {
            guard let fmt = explicitSchema.pointee.format, String(cString: fmt) == "+s" else {
                return jsonBadArgument("am_json_read", "`explicit_schema` must be a struct (\"+s\") schema")
            }
            schema = try jsonFields(explicitSchema, path: "")
        }
        let table = try file.read(JSONReadOptions(explicitSchema: schema, unexpectedFieldBehavior: behavior))
        out.pointee = OpaquePointer(Unmanaged.passRetained(JSONBatchBox(table)).toOpaque())
        return 0
    } catch { jsonStore("\(error)"); return 1 }
}

@_cdecl("am_json_batch_columns")
public func am_json_batch_columns(_ b: OpaquePointer?) -> Int64 {
    guard let box = jsonBatch(b) else { jsonStore("am_json_batch_columns: `b` is NULL (no batch)"); return -1 }
    return Int64(box.table.columns.count)
}

@_cdecl("am_json_batch_rows")
public func am_json_batch_rows(_ b: OpaquePointer?) -> Int64 {
    guard let box = jsonBatch(b) else { jsonStore("am_json_batch_rows: `b` is NULL (no batch)"); return -1 }
    return Int64(box.table.rowCount)
}

@_cdecl("am_json_batch_column_name")
public func am_json_batch_column_name(_ b: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let box = jsonBatch(b), i >= 0, Int(i) < box.names.count else { return nil }
    return UnsafePointer(box.names[Int(i)])
}

/// Every field name of column `i` with its length (see `JSONBatchBox.nameBlobs`); the bytes stay
/// valid until the batch is released. Returns the blob's length, or -1 for a bad handle or index.
@_cdecl("am_json_batch_column_names")
public func am_json_batch_column_names(_ b: OpaquePointer?, _ i: Int64,
                                       _ out: UnsafeMutablePointer<UnsafePointer<UInt8>?>?) -> Int64 {
    guard let box = jsonBatch(b) else { jsonStore("am_json_batch_column_names: `b` is NULL (no batch)"); return -1 }
    guard let out else { jsonStore("am_json_batch_column_names: `out` is NULL"); return -1 }
    guard i >= 0, Int(i) < box.nameBlobs.count else {
        jsonStore("am_json_batch_column_names: column index \(i) is outside 0..<\(box.nameBlobs.count)")
        return -1
    }
    out.pointee = UnsafePointer(box.nameBlobs[Int(i)].baseAddress)
    return Int64(box.nameBlobs[Int(i)].count)
}

@_cdecl("am_json_batch_column")
public func am_json_batch_column(_ b: OpaquePointer?, _ i: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let box = jsonBatch(b) else { return jsonBadArgument("am_json_batch_column", "`b` is NULL (no batch)") }
    guard let out else { return jsonBadArgument("am_json_batch_column", "`out` is NULL") }
    guard i >= 0, Int(i) < box.table.columns.count else {
        return jsonBadArgument("am_json_batch_column", "column index \(i) is outside 0..<\(box.table.columns.count)")
    }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(box.table.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_json_batch_release")
public func am_json_batch_release(_ b: OpaquePointer?) {
    guard let b else { return }
    Unmanaged<JSONBatchBox>.fromOpaque(UnsafeRawPointer(b)).release()
}
