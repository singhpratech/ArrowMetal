import Foundation

/// Errors raised while reading or writing Arrow IPC data.
public enum ArrowIPCError: Error, CustomStringConvertible {
    /// The data ended before a structure that the format requires.
    case truncated(String)
    /// The bytes are structurally invalid (bad offset, impossible length, ...).
    case malformed(String)
    /// Valid Arrow IPC that this implementation does not handle.
    case unsupported(String)
    /// The bytes are not Arrow IPC at all.
    case notArrowIPC

    public var description: String {
        switch self {
        case .truncated(let s): return "Truncated Arrow IPC data: \(s)"
        case .malformed(let s): return "Malformed Arrow IPC data: \(s)"
        case .unsupported(let s): return "Unsupported Arrow IPC feature: \(s)"
        case .notArrowIPC: return "Not Arrow IPC data (no message prefix and no ARROW1 magic)"
        }
    }
}

// MARK: - FlatBuffers reader
//
// Only the subset Arrow's metadata needs: a root table, vtable field lookup, inline scalars,
// strings, vectors of tables and vectors of inline structs. Little-endian only, which is all
// Arrow allows for the metadata anyway.

/// A bounds-checked view over the bytes of one FlatBuffer.
struct FBBuf {
    let bytes: UnsafeRawBufferPointer

    init(_ bytes: UnsafeRawBufferPointer) { self.bytes = bytes }

    /// A sub-range of a larger buffer (one message's metadata inside a file, say).
    init(_ parent: UnsafeRawBufferPointer, from: Int, count: Int) throws {
        guard from >= 0, count >= 0, from <= parent.count, from + count <= parent.count else {
            throw ArrowIPCError.truncated("flatbuffer range \(from)..<\(from + count) of \(parent.count)")
        }
        self.bytes = UnsafeRawBufferPointer(rebasing: parent[from..<(from + count)])
    }

    @inline(__always) func load<T>(_ type: T.Type, at off: Int) throws -> T {
        let size = MemoryLayout<T>.size
        guard off >= 0, off <= bytes.count - size else {
            throw ArrowIPCError.truncated("read \(size) bytes at \(off) of \(bytes.count)")
        }
        return bytes.loadUnaligned(fromByteOffset: off, as: T.self)
    }

    /// Resolves the unsigned offset stored at `pos` (offsets are relative to their own position).
    @inline(__always) func indirect(_ pos: Int) throws -> Int {
        let target = pos + Int(try load(UInt32.self, at: pos))
        guard target >= 0, target < bytes.count else { throw ArrowIPCError.malformed("offset at \(pos) points outside the buffer") }
        return target
    }

    func string(at pos: Int) throws -> String {
        let n = Int(try load(UInt32.self, at: pos))
        guard n >= 0, pos + 4 + n <= bytes.count else { throw ArrowIPCError.truncated("string of \(n) bytes at \(pos)") }
        let raw = UnsafeRawBufferPointer(rebasing: bytes[(pos + 4)..<(pos + 4 + n)])
        return String(decoding: raw, as: UTF8.self)
    }

    /// The buffer's root table.
    func root() throws -> FBTable { try FBTable(self, at: indirect(0)) }
}

/// A FlatBuffers table: a vtable plus inline fields, addressed by field id.
struct FBTable {
    let buf: FBBuf
    let pos: Int
    private let vtable: Int
    private let vtableSize: Int

    init(_ buf: FBBuf, at pos: Int) throws {
        self.buf = buf
        self.pos = pos
        let soffset = Int(try buf.load(Int32.self, at: pos))
        let vt = pos - soffset
        guard vt >= 0, vt + 4 <= buf.bytes.count else { throw ArrowIPCError.malformed("vtable at \(vt) is out of range") }
        self.vtable = vt
        let size = Int(try buf.load(UInt16.self, at: vt))
        guard size >= 4, vt + size <= buf.bytes.count else { throw ArrowIPCError.malformed("vtable size \(size) is out of range") }
        self.vtableSize = size
    }

    /// Absolute position of field `id`, or nil when the writer omitted it (use the schema default).
    func field(_ id: Int) throws -> Int? {
        let entry = 4 + 2 * id
        guard entry + 2 <= vtableSize else { return nil }
        let voffset = Int(try buf.load(UInt16.self, at: vtable + entry))
        return voffset == 0 ? nil : pos + voffset
    }

    func bool(_ id: Int, default d: Bool = false) throws -> Bool {
        guard let p = try field(id) else { return d }
        return try buf.load(UInt8.self, at: p) != 0
    }
    func uint8(_ id: Int, default d: UInt8 = 0) throws -> UInt8 {
        guard let p = try field(id) else { return d }
        return try buf.load(UInt8.self, at: p)
    }
    func int16(_ id: Int, default d: Int16 = 0) throws -> Int16 {
        guard let p = try field(id) else { return d }
        return try buf.load(Int16.self, at: p)
    }
    func int32(_ id: Int, default d: Int32 = 0) throws -> Int32 {
        guard let p = try field(id) else { return d }
        return try buf.load(Int32.self, at: p)
    }
    func int64(_ id: Int, default d: Int64 = 0) throws -> Int64 {
        guard let p = try field(id) else { return d }
        return try buf.load(Int64.self, at: p)
    }
    func string(_ id: Int) throws -> String? {
        guard let p = try field(id) else { return nil }
        return try buf.string(at: buf.indirect(p))
    }
    func table(_ id: Int) throws -> FBTable? {
        guard let p = try field(id) else { return nil }
        return try FBTable(buf, at: buf.indirect(p))
    }
    func vector(_ id: Int) throws -> FBVector? {
        guard let p = try field(id) else { return nil }
        let v = try buf.indirect(p)
        let n = Int(try buf.load(UInt32.self, at: v))
        guard n >= 0 else { throw ArrowIPCError.malformed("negative vector length") }
        return FBVector(buf: buf, start: v + 4, count: n)
    }
}

/// A FlatBuffers vector. Elements are either 4-byte offsets (tables, strings) or inline structs.
struct FBVector {
    let buf: FBBuf
    let start: Int
    let count: Int

    /// Element `i` of a vector of tables.
    func table(_ i: Int) throws -> FBTable {
        guard i >= 0, i < count else { throw ArrowIPCError.malformed("vector index \(i) of \(count)") }
        return try FBTable(buf, at: buf.indirect(start + 4 * i))
    }
    /// Position of element `i` of a vector of inline structs of `stride` bytes.
    func structAt(_ i: Int, stride: Int) throws -> Int {
        guard i >= 0, i < count else { throw ArrowIPCError.malformed("vector index \(i) of \(count)") }
        let p = start + stride * i
        guard p + stride <= buf.bytes.count else { throw ArrowIPCError.truncated("struct element \(i)") }
        return p
    }
    /// Element `i` of a vector of 64-bit integers.
    func int64(_ i: Int) throws -> Int64 {
        guard i >= 0, i < count else { throw ArrowIPCError.malformed("vector index \(i) of \(count)") }
        return try buf.load(Int64.self, at: start + 8 * i)
    }
}

// MARK: - FlatBuffers builder
//
// FlatBuffers are written back to front: every offset is a distance from the end of the buffer,
// so an object can only refer to objects written before it. Tables are therefore built leaves first.

/// A minimal FlatBuffers writer: scalars, strings, tables with vtables, vectors of offsets and structs.
final class FBBuilder {
    private var bytes: [UInt8]
    private var head: Int
    private var minalign = 1
    private var vtable: [Int] = []
    private var objectEnd = 0
    private var building = false

    init(capacity: Int = 4096) {
        bytes = [UInt8](repeating: 0, count: Swift.max(capacity, 64))
        head = bytes.count
    }

    /// Bytes written so far, measured from the end of the buffer. Every offset uses this frame.
    var offset: Int { bytes.count - head }

    private func ensure(_ n: Int) {
        guard head < n else { return }
        let grow = Swift.max(bytes.count, n - head + 64)
        bytes = [UInt8](repeating: 0, count: grow) + bytes
        head += grow
    }

    /// Pads so that a value of `size` bytes followed by `additional` already-planned bytes ends aligned.
    func prep(_ size: Int, _ additional: Int) {
        if size > minalign { minalign = size }
        let alignSize = ((~(offset + additional)) &+ 1) & (size - 1)
        ensure(alignSize + size + additional)
        head -= alignSize
        for i in 0..<alignSize { bytes[head + i] = 0 }
    }

    /// Writes a scalar without aligning first (space must already be reserved by `prep`).
    func place<T>(_ value: T) {
        let size = MemoryLayout<T>.size
        ensure(size)
        head -= size
        withUnsafeBytes(of: value) { src in
            for i in 0..<size { bytes[head + i] = src[i] }
        }
    }

    func pad(_ n: Int) {
        guard n > 0 else { return }
        ensure(n)
        head -= n
        for i in 0..<n { bytes[head + i] = 0 }
    }

    /// Aligns and writes a scalar.
    func add<T>(_ value: T) {
        prep(MemoryLayout<T>.size, 0)
        place(value)
    }

    func placeBytes(_ src: UnsafeRawBufferPointer) {
        ensure(src.count)
        head -= src.count
        for i in 0..<src.count { bytes[head + i] = src[i] }
    }

    // MARK: tables

    func startObject(_ fieldCount: Int) {
        precondition(!building, "nested table construction is not supported; build children first")
        building = true
        vtable = [Int](repeating: 0, count: fieldCount)
        objectEnd = offset
    }

    private func slot(_ id: Int) { vtable[id] = offset }

    /// Adds a scalar field, omitting it when it equals the schema default (as FlatBuffers requires).
    func addScalar<T: Equatable>(id: Int, _ value: T, default d: T) {
        guard value != d else { return }
        add(value)
        slot(id)
    }

    /// Adds a reference to an already-written table, string or vector. `0` means "absent".
    func addOffset(id: Int, _ off: Int) {
        guard off != 0 else { return }
        prependUOffset(off)
        slot(id)
    }

    func prependUOffset(_ off: Int) {
        prep(4, 0)
        place(UInt32(offset + 4 - off))
    }

    func endObject() -> Int {
        precondition(building, "endObject without startObject")
        add(Int32(0))                       // placeholder for the soffset to the vtable
        let objectOffset = offset
        var used = vtable.count
        while used > 0 && vtable[used - 1] == 0 { used -= 1 }
        for i in stride(from: used - 1, through: 0, by: -1) {
            add(UInt16(vtable[i] == 0 ? 0 : objectOffset - vtable[i]))
        }
        add(UInt16(objectOffset - objectEnd))    // inline size of the table
        add(UInt16((used + 2) * 2))              // size of the vtable
        let vtableOffset = offset
        let tableStart = bytes.count - objectOffset
        var soffset = Int32(vtableOffset - objectOffset)
        withUnsafeBytes(of: &soffset) { src in
            for i in 0..<4 { bytes[tableStart + i] = src[i] }
        }
        vtable = []
        building = false
        return objectOffset
    }

    // MARK: vectors and strings

    func startVector(elementSize: Int, count: Int, alignment: Int) {
        prep(4, elementSize * count)
        prep(alignment, elementSize * count)
    }

    func endVector(_ count: Int) -> Int {
        add(UInt32(count))
        return offset
    }

    func createString(_ s: String) -> Int {
        let utf8 = Array(s.utf8)
        prep(4, utf8.count + 1)
        place(UInt8(0))                     // FlatBuffers strings are null terminated
        utf8.withUnsafeBytes { placeBytes($0) }
        return endVector(utf8.count)
    }

    /// A vector of references to already-written tables or strings.
    func createOffsetVector(_ offsets: [Int]) -> Int {
        startVector(elementSize: 4, count: offsets.count, alignment: 4)
        for o in offsets.reversed() { prependUOffset(o) }
        return endVector(offsets.count)
    }

    /// Writes the root offset and returns the finished buffer. `minimumAlignment` also fixes the
    /// total length to a multiple of itself, which is how Arrow keeps message bodies 8-byte aligned.
    func finish(_ root: Int, minimumAlignment: Int = 8) -> [UInt8] {
        if minimumAlignment > minalign { minalign = minimumAlignment }
        prep(minalign, 4)
        prependUOffset(root)
        return Array(bytes[head...])
    }
}

// MARK: - Arrow metadata constants (format/Message.fbs, Schema.fbs, File.fbs)

enum FBMessageHeader: UInt8 {
    case none = 0, schema = 1, dictionaryBatch = 2, recordBatch = 3, tensor = 4, sparseTensor = 5
}

/// The `Type` union discriminant in Schema.fbs.
enum FBTypeKind: UInt8 {
    case none = 0, null = 1, int = 2, floatingPoint = 3, binary = 4, utf8 = 5, bool = 6
    case decimal = 7, date = 8, time = 9, timestamp = 10, interval = 11, list = 12, structKind = 13
    case union = 14, fixedSizeBinary = 15, fixedSizeList = 16, map = 17, duration = 18
    case largeBinary = 19, largeUtf8 = 20, largeList = 21, runEndEncoded = 22
    case binaryView = 23, utf8View = 24, listView = 25, largeListView = 26

    var name: String {
        switch self {
        case .none: return "none"
        case .null: return "null"
        case .int: return "int"
        case .floatingPoint: return "floating point"
        case .binary: return "binary"
        case .utf8: return "utf8"
        case .bool: return "bool"
        case .decimal: return "decimal"
        case .date: return "date"
        case .time: return "time"
        case .timestamp: return "timestamp"
        case .interval: return "interval"
        case .list: return "list"
        case .structKind: return "struct"
        case .union: return "union"
        case .fixedSizeBinary: return "fixed size binary"
        case .fixedSizeList: return "fixed size list"
        case .map: return "map"
        case .duration: return "duration"
        case .largeBinary: return "large binary"
        case .largeUtf8: return "large utf8"
        case .largeList: return "large list"
        case .runEndEncoded: return "run end encoded"
        case .binaryView: return "binary view"
        case .utf8View: return "utf8 view"
        case .listView: return "list view"
        case .largeListView: return "large list view"
        }
    }
}

/// Metadata version V5, the only one this package writes.
let fbMetadataVersionV5: Int16 = 4
/// FlatBuffers `Precision`: HALF, SINGLE, DOUBLE.
let fbPrecisionSingle: Int16 = 1
let fbPrecisionDouble: Int16 = 2
/// `DateUnit`: DAY, MILLISECOND (default MILLISECOND).
let fbDateUnitDay: Int16 = 0
let fbDateUnitMillisecond: Int16 = 1
/// Struct strides in Arrow's metadata: `FieldNode`, `Buffer` and `Block` from File.fbs.
let fbFieldNodeStride = 16
let fbBufferStride = 16
let fbBlockStride = 24
