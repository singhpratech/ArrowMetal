import Foundation

// A hand-rolled Thrift *compact protocol* reader, written in the same spirit as `IPC/FlatBuffers.swift`:
// the Parquet footer is the only Thrift this package ever sees, so instead of a code generator and a
// runtime we decode the handful of structs the format defines, straight out of the mapped file bytes.
//
// Compact protocol, in one paragraph. A struct is a sequence of field headers terminated by a 0 byte.
// A field header is one byte: the low nibble is the type, the high nibble is the *delta* from the
// previous field id (0 means "an explicit zig-zag varint id follows"). Booleans are encoded in the type
// nibble itself (1 = true, 2 = false) so they carry no payload. i16/i32/i64 are zig-zag varints, doubles
// are 8 raw little-endian bytes, binary is a varint length plus the bytes. A list header packs the
// element count into the high nibble when it is below 15 (and spills to a varint otherwise) and the
// element type into the low nibble; a map header is a varint size followed by one packed key/value
// type byte. That is the whole format.

/// Errors raised while reading or writing Parquet.
public enum ParquetError: Error, CustomStringConvertible {
    case truncated(String)
    case malformed(String)
    case unsupported(String)
    case notParquet
    case io(String)

    public var description: String {
        switch self {
        case .truncated(let s): return "Truncated Parquet data: \(s)"
        case .malformed(let s): return "Malformed Parquet data: \(s)"
        case .unsupported(let s): return "Unsupported Parquet feature: \(s)"
        case .notParquet: return "Not a Parquet file (no PAR1 magic)"
        case .io(let s): return "Parquet I/O error: \(s)"
        }
    }
}

/// Thrift compact-protocol field types.
enum TCType: UInt8 {
    case stop = 0, boolTrue = 1, boolFalse = 2, byte = 3, i16 = 4, i32 = 5, i64 = 6
    case double = 7, binary = 8, list = 9, set = 10, map = 11, structure = 12, uuid = 13
}

/// A cursor over one Thrift compact-protocol message.
struct ThriftReader {
    let bytes: UnsafeRawBufferPointer
    var pos: Int
    /// Field-id stack: compact protocol encodes ids as deltas within each struct.
    private var lastFieldID: Int16 = 0
    private var idStack: [Int16] = []

    init(_ bytes: UnsafeRawBufferPointer, at pos: Int = 0) {
        self.bytes = bytes
        self.pos = pos
    }

    // MARK: primitives

    @inline(__always) mutating func byte() throws -> UInt8 {
        guard pos < bytes.count else { throw ParquetError.truncated("byte at \(pos) of \(bytes.count)") }
        defer { pos += 1 }
        return bytes[pos]
    }

    /// LEB128 unsigned varint, at most 10 bytes.
    @inline(__always) mutating func varint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        for _ in 0..<10 {
            let b = try byte()
            result |= UInt64(b & 0x7F) &<< shift
            if b & 0x80 == 0 { return result }
            shift += 7
        }
        throw ParquetError.malformed("varint longer than 10 bytes at \(pos)")
    }

    /// A varint used as a count or a byte length. Anything that cannot be a position inside `bytes` is
    /// malformed, and saying so is the whole point: `Int(someUInt64)` *traps* on overflow, so a corrupt
    /// footer claiming a 2^63-byte string would take the process down instead of raising an error.
    @inline(__always) mutating func varintCount(_ what: @autoclosure () -> String) throws -> Int {
        let u = try varint()
        guard u <= UInt64(bytes.count) else {
            throw ParquetError.malformed("\(what()) of \(u) exceeds the \(bytes.count) bytes available")
        }
        return Int(u)
    }

    @inline(__always) mutating func zigzag() throws -> Int64 {
        let u = try varint()
        return Int64(bitPattern: (u >> 1)) ^ -Int64(bitPattern: u & 1)
    }

    mutating func double() throws -> Double {
        guard pos + 8 <= bytes.count else { throw ParquetError.truncated("double at \(pos)") }
        defer { pos += 8 }
        return Double(bitPattern: bytes.loadUnaligned(fromByteOffset: pos, as: UInt64.self))
    }

    /// Binary/string payload, returned as a range into the underlying bytes (no copy).
    mutating func binaryRange() throws -> Range<Int> {
        let at = pos
        let n = try varintCount("binary at \(at)")
        guard pos + n <= bytes.count else { throw ParquetError.truncated("binary of \(n) bytes at \(pos)") }
        defer { pos += n }
        return pos..<(pos + n)
    }

    mutating func binary() throws -> [UInt8] {
        let r = try binaryRange()
        return Array(bytes[r])
    }

    mutating func string() throws -> String {
        let r = try binaryRange()
        return String(decoding: UnsafeRawBufferPointer(rebasing: bytes[r]), as: UTF8.self)
    }

    // MARK: structs

    /// Deepest struct nesting accepted. `parquet.thrift` nests a handful of levels; anything past this
    /// is a crafted footer, and following it would recurse `skip` until the stack runs out.
    static let maxDepth = 200

    mutating func pushStruct() throws {
        guard idStack.count < ThriftReader.maxDepth else {
            throw ParquetError.malformed("thrift struct nesting deeper than \(ThriftReader.maxDepth) at \(pos)")
        }
        idStack.append(lastFieldID)
        lastFieldID = 0
    }
    mutating func popStruct() {
        lastFieldID = idStack.popLast() ?? 0
    }

    /// Next field header, or nil at the struct's STOP byte.
    mutating func nextField() throws -> (id: Int16, type: TCType)? {
        let b = try byte()
        if b == 0 { return nil }
        let typeBits = b & 0x0F
        guard let t = TCType(rawValue: typeBits) else {
            throw ParquetError.malformed("thrift field type \(typeBits) at \(pos - 1)")
        }
        let delta = Int16(b >> 4)
        let id: Int16
        if delta == 0 {
            id = Int16(truncatingIfNeeded: try zigzag())
        } else {
            id = lastFieldID + delta
        }
        lastFieldID = id
        return (id, t)
    }

    /// List header: element count and element type.
    mutating func listHeader() throws -> (count: Int, type: TCType) {
        let b = try byte()
        var n = Int(b >> 4)
        let at = pos
        if n == 15 { n = try varintCount("thrift list at \(at)") }
        guard let t = TCType(rawValue: b & 0x0F) else {
            throw ParquetError.malformed("thrift list element type \(b & 0x0F)")
        }
        guard n <= bytes.count else { throw ParquetError.malformed("thrift list of \(n) elements") }
        return (n, t)
    }

    /// Reads a struct, calling `field` for every field it contains. `field` returns true when it consumed
    /// the value; when it returns false the value is skipped.
    mutating func readStruct(_ field: (inout ThriftReader, Int16, TCType) throws -> Bool) throws {
        try pushStruct()
        while let f = try nextField() {
            if !(try field(&self, f.id, f.type)) { try skip(f.type) }
        }
        popStruct()
    }

    /// Reads a `list<struct>`, calling `element` once per element.
    mutating func readList(_ element: (inout ThriftReader) throws -> Void) throws {
        let h = try listHeader()
        for _ in 0..<h.count {
            if h.type == .structure { try element(&self) } else { try skip(h.type) }
        }
    }

    /// Reads a `list<i32>` (an enum list, in Parquet's schema).
    mutating func readInt32List() throws -> [Int32] {
        let h = try listHeader()
        var out: [Int32] = []
        out.reserveCapacity(h.count)
        for _ in 0..<h.count {
            if h.type == .i32 || h.type == .i64 || h.type == .i16 { out.append(Int32(truncatingIfNeeded: try zigzag())) }
            else { try skip(h.type) }
        }
        return out
    }

    mutating func readStringList() throws -> [String] {
        let h = try listHeader()
        var out: [String] = []
        out.reserveCapacity(h.count)
        for _ in 0..<h.count {
            if h.type == .binary { out.append(try string()) } else { try skip(h.type) }
        }
        return out
    }

    /// Skips a value of `type` without interpreting it.
    mutating func skip(_ type: TCType) throws {
        switch type {
        case .stop, .boolTrue, .boolFalse: return
        case .byte: _ = try byte()
        case .i16, .i32, .i64: _ = try zigzag()
        case .double: _ = try double()
        case .binary, .uuid: _ = try binaryRange()
        case .list, .set:
            let h = try listHeader()
            for _ in 0..<h.count { try skip(h.type) }
        case .map:
            let mapAt = pos
            let n = try varintCount("thrift map at \(mapAt)")
            if n > 0 {
                let kv = try byte()
                guard let kt = TCType(rawValue: kv >> 4), let vt = TCType(rawValue: kv & 0x0F) else {
                    throw ParquetError.malformed("thrift map types")
                }
                for _ in 0..<n { try skip(kt); try skip(vt) }
            }
        case .structure:
            try pushStruct()
            while let f = try nextField() { try skip(f.type) }
            popStruct()
        }
    }

    /// Reads an i16/i32/i64 field as Int.
    @inline(__always) mutating func int() throws -> Int { Int(try zigzag()) }
    @inline(__always) mutating func int64() throws -> Int64 { try zigzag() }
    @inline(__always) mutating func int32() throws -> Int32 { Int32(truncatingIfNeeded: try zigzag()) }
    @inline(__always) mutating func bool(_ t: TCType) throws -> Bool {
        // A boolean field carries its value in the type nibble; a boolean *inside a list* is a byte.
        switch t {
        case .boolTrue: return true
        case .boolFalse: return false
        default: return try byte() != 0
        }
    }
}

// MARK: - Thrift compact writer (used by the Parquet writer)

/// Minimal Thrift compact-protocol writer: the mirror image of `ThriftReader`.
struct ThriftWriter {
    var bytes: [UInt8] = []
    private var lastFieldID: Int16 = 0
    private var idStack: [Int16] = []

    mutating func varint(_ v: UInt64) {
        var x = v
        while true {
            if x < 0x80 { bytes.append(UInt8(x)); return }
            bytes.append(UInt8((x & 0x7F) | 0x80))
            x >>= 7
        }
    }
    mutating func zigzag(_ v: Int64) { varint(UInt64(bitPattern: (v << 1) ^ (v >> 63))) }

    mutating func fieldHeader(_ id: Int16, _ type: TCType) {
        let delta = id - lastFieldID
        if delta > 0 && delta <= 15 {
            bytes.append(UInt8(delta) << 4 | type.rawValue)
        } else {
            bytes.append(type.rawValue)
            zigzag(Int64(id))
        }
        lastFieldID = id
    }

    mutating func int(_ id: Int16, _ v: Int) { fieldHeader(id, .i32); zigzag(Int64(v)) }
    mutating func int64(_ id: Int16, _ v: Int64) { fieldHeader(id, .i64); zigzag(v) }
    mutating func bool(_ id: Int16, _ v: Bool) { fieldHeader(id, v ? .boolTrue : .boolFalse) }
    mutating func string(_ id: Int16, _ s: String) { binary(id, Array(s.utf8)) }
    mutating func binary(_ id: Int16, _ b: [UInt8]) {
        fieldHeader(id, .binary)
        varint(UInt64(b.count))
        bytes.append(contentsOf: b)
    }

    mutating func beginStruct(_ id: Int16) { fieldHeader(id, .structure); idStack.append(lastFieldID); lastFieldID = 0 }
    mutating func endStruct() { bytes.append(0); lastFieldID = idStack.popLast() ?? 0 }
    /// The outermost struct has no field header of its own.
    mutating func beginRoot() { idStack.append(lastFieldID); lastFieldID = 0 }
    mutating func endRoot() { bytes.append(0); lastFieldID = idStack.popLast() ?? 0 }

    mutating func beginList(_ id: Int16, _ type: TCType, count: Int) {
        fieldHeader(id, .list)
        if count < 15 {
            bytes.append(UInt8(count) << 4 | type.rawValue)
        } else {
            bytes.append(0xF0 | type.rawValue)
            varint(UInt64(count))
        }
        // Elements of a list of structs each start a fresh field-id frame.
    }
    /// One element of a `list<struct>`.
    mutating func beginListElement() { idStack.append(lastFieldID); lastFieldID = 0 }
    mutating func endListElement() { bytes.append(0); lastFieldID = idStack.popLast() ?? 0 }

    mutating func int32List(_ id: Int16, _ vs: [Int32]) {
        beginList(id, .i32, count: vs.count)
        for v in vs { zigzag(Int64(v)) }
    }
    mutating func stringList(_ id: Int16, _ vs: [String]) {
        beginList(id, .binary, count: vs.count)
        for v in vs { let b = Array(v.utf8); varint(UInt64(b.count)); bytes.append(contentsOf: b) }
    }
}
