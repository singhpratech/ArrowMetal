import Foundation
import Metal

// Newline-delimited JSON, parsed on the GPU (docs/JSON.md).
//
// `JSONReader` reads a file into Metal shared memory and turns its top-level objects into Arrow
// columns with the semantics of `pyarrow.json.read_json`: the same type inference (null, bool, int64,
// double, timestamp[s], string, struct, list), the same field order (first appearance), the same
// explicit-schema and unexpected-field options, and the same error texts. The structure scan, the
// per-record walk, key matching, string unescaping and timestamp parsing run as Metal kernels
// (`Kernels/JSONSource.swift`); number text is gathered on the GPU and handed to
// `MetalStringArray.parse`.

/// What to do with a field the explicit schema does not name (pyarrow's `unexpected_field_behavior`).
public enum JSONUnexpectedFieldBehavior: String, Sendable {
    /// Infer its type and append it after the schema's fields (the default).
    case infer
    /// Drop it.
    case ignore
    /// Fail with "JSON parse error: unexpected field".
    case error
}

/// A column type the reader can produce or be asked for through an explicit schema.
public indirect enum JSONType: Equatable, Sendable, CustomStringConvertible {
    case null, boolean
    case int8, int16, int32, int64, uint8, uint16, uint32, uint64
    case float32, float64
    case utf8
    case timestamp(ArrowTemporalUnit, timezone: String?)
    case list(JSONType)
    case structure([JSONField])

    /// Arrow's spelling of the type (`DataType::ToString`), as pyarrow prints it in messages.
    public var description: String {
        switch self {
        case .null: return "null"
        case .boolean: return "bool"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .float32: return "float"
        case .float64: return "double"
        case .utf8: return "string"
        case .timestamp(let u, let tz):
            let unit: String
            switch u { case .second: unit = "s"; case .milli: unit = "ms"; case .micro: unit = "us"; case .nano: unit = "ns" }
            return tz.map { "timestamp[\(unit), tz=\($0)]" } ?? "timestamp[\(unit)]"
        case .list(let t): return "list<item: \(t)>"
        case .structure(let fs): return "struct<\(fs.map { "\($0.name): \($0.type)" }.joined(separator: ", "))>"
        }
    }

    /// The JSON value class this type is read from (the names pyarrow uses in "changed from" messages).
    var jsonClass: String {
        switch self {
        case .null: return "null"
        case .boolean: return "boolean"
        case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64, .float32, .float64: return "number"
        case .utf8, .timestamp: return "string"
        case .list: return "array"
        case .structure: return "object"
        }
    }
    var jsonClassID: Int {
        switch jsonClass {
        case "boolean": return 1
        case "number": return 2
        case "string": return 3
        case "object": return 4
        case "array": return 5
        default: return 0
        }
    }
}

/// One named field of an explicit schema or of a struct type.
public struct JSONField: Equatable, Sendable {
    public var name: String
    public var type: JSONType
    public init(_ name: String, _ type: JSONType) { self.name = name; self.type = type }
}

/// Options for one read; the fields mirror `pyarrow.json.ParseOptions`.
public struct JSONReadOptions: Sendable {
    /// Types for the named fields; fields it does not name follow `unexpectedFieldBehavior`.
    public var explicitSchema: [JSONField]?
    public var unexpectedFieldBehavior: JSONUnexpectedFieldBehavior
    public init(explicitSchema: [JSONField]? = nil, unexpectedFieldBehavior: JSONUnexpectedFieldBehavior = .infer) {
        self.explicitSchema = explicitSchema
        self.unexpectedFieldBehavior = unexpectedFieldBehavior
    }
}

/// Why a read failed. The descriptions are pyarrow's texts, so a caller comparing messages across the
/// two readers sees the same string.
public enum JSONError: Error, Equatable, CustomStringConvertible {
    /// A syntax error, a type conflict, a repeated field or an unexpected field.
    case parse(String)
    /// A value that does not convert to its explicit-schema type.
    case conversion(String)
    /// The file has no bytes at all.
    case empty
    case io(String)
    /// A request outside what this reader covers (an explicit-schema type it cannot produce, a file
    /// of 4 GiB or more).
    case unsupported(String)

    public var description: String {
        switch self {
        case .parse(let m): return "JSON parse error: \(m)"
        case .conversion(let m): return m
        case .empty: return "Empty JSON file"
        case .io(let m): return m
        case .unsupported(let m): return m
        }
    }
}

/// The columns of one read. `rowCount` is kept separately so a file of empty objects still reports
/// its rows when it has no columns.
public struct JSONTable {
    public let names: [String]
    public let columns: [AnyMetalArray]
    public let rowCount: Int

    public subscript(name: String) -> AnyMetalArray? {
        names.firstIndex(of: name).map { columns[$0] }
    }

    /// The same columns as a record batch (which cannot carry a row count without a column).
    public func recordBatch() throws -> MetalRecordBatch {
        try MetalRecordBatch(names: names, columns: columns)
    }
}

/// A newline-delimited JSON file, read once into Metal shared memory and parsed on the GPU.
public final class JSONReader: @unchecked Sendable {
    public let path: String?
    public let context: MetalContext
    /// Bytes in the input.
    public let byteCount: Int

    /// The input, followed by at least 64 zero bytes.
    let source: MetalArrowBuffer

    /// Reads the file. Nothing is parsed until `read` is called.
    ///
    /// The bytes are read with parallel `pread` calls into a Metal buffer, since every byte gets parsed.
    public init(path: String, context: MetalContext = .shared) throws {
        self.path = path
        self.context = context
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw JSONError.io("cannot open \(path): \(String(cString: strerror(errno)))") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw JSONError.io("cannot stat \(path)") }
        let size = Int(st.st_size)
        try JSONReader.checkSize(size)
        byteCount = size
        let buf = try MetalArrowBuffer.allocate(byteCount: size + 64, zeroed: false, context: context)
        let dst = buf.mutableContents
        let chunks = Swift.max(1, Swift.min(8, size / (8 << 20)))
        let per = (size + chunks - 1) / chunks
        var failed = [Int32](repeating: 0, count: chunks)
        failed.withUnsafeMutableBufferPointer { fail in
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                var off = c * per
                let end = Swift.min(size, off + per)
                while off < end {
                    let r = pread(fd, dst + off, end - off, off_t(off))
                    if r <= 0 { fail[c] = r < 0 ? errno : EIO; break }
                    off += r
                }
            }
        }
        if let e = failed.first(where: { $0 != 0 }) {
            throw JSONError.io("cannot read \(path): \(String(cString: strerror(e)))")
        }
        memset(dst + size, 0, 64)
        source = buf
    }

    /// Parses bytes already in memory (they are copied into a Metal buffer once).
    public convenience init(bytes: [UInt8], context: MetalContext = .shared) throws {
        try self.init(buffer: UnsafeRawBufferPointer(start: nil, count: 0), context: context, bytes: bytes)
    }

    /// Parses bytes already in memory without an intermediate array (they are copied into a Metal
    /// buffer once). The caller's memory is not referenced after the call.
    public convenience init(buffer: UnsafeRawBufferPointer, context: MetalContext = .shared) throws {
        try self.init(buffer: buffer, context: context, bytes: nil)
    }

    private init(buffer: UnsafeRawBufferPointer, context: MetalContext, bytes: [UInt8]?) throws {
        path = nil
        self.context = context
        let count = bytes?.count ?? buffer.count
        try JSONReader.checkSize(count)
        byteCount = count
        let dst = try MetalArrowBuffer.allocate(byteCount: count + 64, zeroed: false, context: context)
        if let bytes {
            bytes.withUnsafeBytes { if count > 0 { memcpy(dst.mutableContents, $0.baseAddress!, count) } }
        } else if count > 0 {
            memcpy(dst.mutableContents, buffer.baseAddress!, count)
        }
        memset(dst.mutableContents + count, 0, 64)
        source = dst
    }

    public convenience init(string: String, context: MetalContext = .shared) throws {
        try self.init(bytes: Array(string.utf8), context: context)
    }

    /// Byte positions are 32-bit on the GPU.
    static func checkSize(_ size: Int) throws {
        guard size < Int(UInt32.max) - 4096 else {
            throw JSONError.unsupported("JSON input of \(size) bytes: inputs of 4 GiB or more are not supported yet")
        }
    }

    /// Reads a file in one call.
    public static func read(path: String, options: JSONReadOptions = JSONReadOptions(),
                            context: MetalContext = .shared) throws -> JSONTable {
        try JSONReader(path: path, context: context).read(options)
    }

    /// Parses every record and returns the columns.
    public func read(_ options: JSONReadOptions = JSONReadOptions()) throws -> JSONTable {
        guard byteCount > 0 else { throw JSONError.empty }
        let ctx = context
        let n = byteCount
        let host = source.typed(UInt8.self)
        // pyarrow 25.0.1 skips each byte of the UTF-8 BOM (EF BB BF) that is present at the start, in
        // order, so a partial mark (EF BB, EF BF, BB BF or any one of the three) is skipped as well.
        var start = 0
        for b: UInt8 in [0xEF, 0xBB, 0xBF] where start < n && host[start] == b { start += 1 }

        // Stage 1: where the records are.
        let recs = try JSONKernels.records(ctx, source, n: n, start: start)
        // Stage 2: validate every record and count its fields.
        let (counts, walkError) = try JSONKernels.walkCount(ctx, source, n: n, spanStart: recs.start,
                                                            spanEnd: recs.end, spans: recs.count)
        // The first syntax error decides how many records are read. Its record is kept with the children
        // the walk reached, so a type conflict or repeated key before the error is still found and, being
        // earlier in the file, reported first, as a sequential parser would.
        var rows = recs.count
        var syntax: (position: Int, message: String)? = nil
        let recStart = recs.start.typed(UInt32.self)
        if let top = recs.topError {
            var lo = 0, hi = recs.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if Int(recStart[mid]) < top.position { lo = mid + 1 } else { hi = mid }
            }
            rows = lo
            let text = JSONSyntax.message(top.code)
            syntax = (top.position, JSONSyntax.hasRow(top.code) ? "\(text) in row \(lo)" : text)
        }
        if let w = walkError, w.span < rows {
            // A record past the nesting limit is not built: its partial values would nest as deep.
            rows = w.code == 14 ? w.span : w.span + 1
            syntax = (w.position, "\(JSONSyntax.message(w.code)) in row \(w.span)")
        }
        let level = try JSONKernels.walkEmit(ctx, source, n: n, spanStart: recs.start, spanEnd: recs.end,
                                             spans: rows, counts: counts)
        // Stage 3: columns.
        let builder = JSONColumnBuilder(context: ctx, source: source, n: n, host: host,
                                        behavior: options.unexpectedFieldBehavior)
        let (names, columns) = try builder.fields(level: level, rows: rows, schema: options.explicitSchema,
                                                  path: "", rowToRecord: { $0 })
        let firstParse = builder.parseErrors.min(by: { $0.position < $1.position })
        if let e = firstParse, syntax == nil || e.position < syntax!.position { throw JSONError.parse(e.message) }
        if let s = syntax { throw JSONError.parse(s.message) }
        if let c = builder.conversionErrors.min(by: { $0.position < $1.position }) { throw JSONError.conversion(c.message) }
        return JSONTable(names: names, columns: columns, rowCount: rows)
    }
}
