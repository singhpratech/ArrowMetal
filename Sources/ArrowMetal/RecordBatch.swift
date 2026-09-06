import Foundation
import CArrowABI

extension AnyMetalArray {
    public var nullCount: Int {
        switch self {
        case .int8(let a): return a.nullCount
        case .uint8(let a): return a.nullCount
        case .int16(let a): return a.nullCount
        case .uint16(let a): return a.nullCount
        case .int32(let a): return a.nullCount
        case .uint32(let a): return a.nullCount
        case .int64(let a): return a.nullCount
        case .uint64(let a): return a.nullCount
        case .float32(let a): return a.nullCount
        case .float64(let a): return a.nullCount
        case .boolean(let a): return a.nullCount
        case .string(let a): return a.nullCount
        case .temporal(let a): return a.nullCount
        case .binary(let a): return a.nullCount
        case .decimal(let a): return a.nullCount
        case .dictionary(let codes, _): return codes.nullCount
        }
    }

    /// Applies `filter` to whichever concrete array this is.
    public func filter(_ mask: MetalBooleanArray) throws -> AnyMetalArray {
        switch self {
        case .int8(let a): return .int8(try a.filter(mask))
        case .uint8(let a): return .uint8(try a.filter(mask))
        case .int16(let a): return .int16(try a.filter(mask))
        case .uint16(let a): return .uint16(try a.filter(mask))
        case .int32(let a): return .int32(try a.filter(mask))
        case .uint32(let a): return .uint32(try a.filter(mask))
        case .int64(let a): return .int64(try a.filter(mask))
        case .uint64(let a): return .uint64(try a.filter(mask))
        case .float32(let a): return .float32(try a.filter(mask))
        case .float64(let a): return .float64(try a.filter(mask))
        case .boolean(let a): return .boolean(try a.filter(mask))
        case .string(let a): return .string(try a.filter(mask))
        case .temporal(let a): return .temporal(try a.filter(mask))
        case .binary(let a): return .binary(markBinary(try a.filter(mask)))
        case .decimal(let a): return .decimal(try a.filter(mask))
        case .dictionary(let codes, let values): return .dictionary(codes: try codes.filter(mask), values: values)
        }
    }

    public func take<I: ArrowIndex>(_ idx: MetalArray<I>) throws -> AnyMetalArray {
        switch self {
        case .int8(let a): return .int8(try a.take(idx))
        case .uint8(let a): return .uint8(try a.take(idx))
        case .int16(let a): return .int16(try a.take(idx))
        case .uint16(let a): return .uint16(try a.take(idx))
        case .int32(let a): return .int32(try a.take(idx))
        case .uint32(let a): return .uint32(try a.take(idx))
        case .int64(let a): return .int64(try a.take(idx))
        case .uint64(let a): return .uint64(try a.take(idx))
        case .float32(let a): return .float32(try a.take(idx))
        case .float64(let a): return .float64(try a.take(idx))
        case .boolean(let a): return .boolean(try a.take(idx))
        case .string(let a): return .string(try a.take(idx))
        case .temporal(let a): return .temporal(try a.take(idx))
        case .binary(let a): return .binary(markBinary(try a.take(idx)))
        case .decimal(let a): return .decimal(try a.take(idx))
        case .dictionary(let codes, let values): return .dictionary(codes: try codes.take(idx), values: values)
        }
    }

    public func slice(offset: Int, length: Int) throws -> AnyMetalArray {
        switch self {
        case .int8(let a): return .int8(try a.slice(offset: offset, length: length))
        case .uint8(let a): return .uint8(try a.slice(offset: offset, length: length))
        case .int16(let a): return .int16(try a.slice(offset: offset, length: length))
        case .uint16(let a): return .uint16(try a.slice(offset: offset, length: length))
        case .int32(let a): return .int32(try a.slice(offset: offset, length: length))
        case .uint32(let a): return .uint32(try a.slice(offset: offset, length: length))
        case .int64(let a): return .int64(try a.slice(offset: offset, length: length))
        case .uint64(let a): return .uint64(try a.slice(offset: offset, length: length))
        case .float32(let a): return .float32(try a.slice(offset: offset, length: length))
        case .float64(let a): return .float64(try a.slice(offset: offset, length: length))
        case .boolean(let a): return .boolean(try a.slice(offset: offset, length: length))
        case .string(let a): return .string(try a.take(try MetalArray<Int32>((offset..<(offset + length)).map { Int32($0) }, context: a.context)))
        case .temporal(let a): return .temporal(try a.slice(offset: offset, length: length))
        case .binary(let a):
            return .binary(markBinary(try a.take(try MetalArray<Int32>((offset..<(offset + length)).map { Int32($0) }, context: a.context))))
        case .decimal(let a): return .decimal(try a.slice(offset: offset, length: length))
        case .dictionary(let codes, let values):
            return .dictionary(codes: try codes.slice(offset: offset, length: length), values: values)
        }
    }

    /// Typed accessors (nil when the column has another type).
    public var asInt32: MetalArray<Int32>? { if case .int32(let a) = self { return a } else { return nil } }
    public var asInt64: MetalArray<Int64>? { if case .int64(let a) = self { return a } else { return nil } }
    public var asFloat32: MetalArray<Float>? { if case .float32(let a) = self { return a } else { return nil } }
    public var asFloat64: MetalArray<Double>? { if case .float64(let a) = self { return a } else { return nil } }
    public var asBoolean: MetalBooleanArray? { if case .boolean(let a) = self { return a } else { return nil } }
    public var asString: MetalStringArray? { if case .string(let a) = self { return a } else { return nil } }
    public var asTemporal: MetalTemporalArray? { if case .temporal(let a) = self { return a } else { return nil } }
    public var asBinary: MetalStringArray? { if case .binary(let a) = self { return a } else { return nil } }
    public var asDecimal: MetalDecimalArray? { if case .decimal(let a) = self { return a } else { return nil } }
}

/// A set of equal-length named columns: the Metal-resident equivalent of an Arrow RecordBatch.
public struct MetalRecordBatch {
    public let names: [String]
    public let columns: [AnyMetalArray]
    public var length: Int { columns.first?.length ?? 0 }
    public var columnCount: Int { columns.count }

    public init(names: [String], columns: [AnyMetalArray]) throws {
        guard names.count == columns.count else { throw ArrowMetalError.invalidArrowArray("names/columns count mismatch") }
        if let first = columns.first {
            for c in columns where c.length != first.length { throw ArrowMetalError.lengthMismatch(first.length, c.length) }
        }
        self.names = names
        self.columns = columns
    }

    public subscript(name: String) -> AnyMetalArray? {
        guard let i = names.firstIndex(of: name) else { return nil }
        return columns[i]
    }
    public subscript(index: Int) -> AnyMetalArray { columns[index] }

    /// Filters every column with the same mask.
    public func filter(_ mask: MetalBooleanArray) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: names, columns: columns.map { try $0.filter(mask) })
    }
    public func take<I: ArrowIndex>(_ idx: MetalArray<I>) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: names, columns: columns.map { try $0.take(idx) })
    }
    public func slice(offset: Int, length: Int) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: names, columns: columns.map { try $0.slice(offset: offset, length: length) })
    }
    public func selecting(_ cols: [String]) throws -> MetalRecordBatch {
        var cs: [AnyMetalArray] = []
        for n in cols {
            guard let c = self[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
            cs.append(c)
        }
        return try MetalRecordBatch(names: cols, columns: cs)
    }
}

// MARK: - Struct / stream interop

final class StructExportHolder {
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowArray>?>
    let childStructs: UnsafeMutablePointer<ArrowArray>
    let buffers: UnsafeMutablePointer<UnsafeRawPointer?>
    let n: Int
    init(n: Int) {
        self.n = n
        children = .allocate(capacity: max(n, 1))
        childStructs = .allocate(capacity: max(n, 1))
        childStructs.initialize(repeating: ArrowArray(), count: max(n, 1))
        for i in 0..<n { children[i] = childStructs + i }
        buffers = .allocate(capacity: 1)
        buffers[0] = nil
    }
    deinit {
        for i in 0..<n { if let r = childStructs[i].release { r(childStructs + i) } }
        children.deallocate(); childStructs.deallocate(); buffers.deallocate()
    }
}

final class StructSchemaHolder {
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowSchema>?>
    let childStructs: UnsafeMutablePointer<ArrowSchema>
    let n: Int
    let formatC = strdup("+s")!
    let nameC: UnsafeMutablePointer<CChar>
    init(n: Int, name: String) {
        self.n = n
        nameC = strdup(name)!
        children = .allocate(capacity: max(n, 1))
        childStructs = .allocate(capacity: max(n, 1))
        childStructs.initialize(repeating: ArrowSchema(), count: max(n, 1))
        for i in 0..<n { children[i] = childStructs + i }
    }
    deinit {
        for i in 0..<n { if let r = childStructs[i].release { r(childStructs + i) } }
        children.deallocate(); childStructs.deallocate(); free(formatC); free(nameC)
    }
}

private func releaseStructArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<StructExportHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}
private func releaseStructSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<StructSchemaHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

extension MetalRecordBatch {
    /// Exports as a struct array (`+s`) with one child per column, the C Data Interface's record batch form.
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        let h = StructExportHolder(n: columns.count)
        for (i, c) in columns.enumerated() { c.exportArrowArray(into: h.childStructs + i) }
        out.pointee.length = Int64(length)
        out.pointee.null_count = 0
        out.pointee.offset = 0
        out.pointee.n_buffers = 1
        out.pointee.n_children = Int64(columns.count)
        out.pointee.buffers = UnsafeMutablePointer<UnsafeRawPointer?>(h.buffers)
        out.pointee.children = h.children
        out.pointee.dictionary = nil
        out.pointee.release = releaseStructArray
        out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
    }

    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        let h = StructSchemaHolder(n: columns.count, name: name)
        for (i, c) in columns.enumerated() { c.exportArrowSchema(name: names[i], into: h.childStructs + i) }
        out.pointee.format = UnsafePointer(h.formatC)
        out.pointee.name = UnsafePointer(h.nameC)
        out.pointee.metadata = nil
        out.pointee.flags = 0
        out.pointee.n_children = Int64(columns.count)
        out.pointee.children = h.children
        out.pointee.dictionary = nil
        out.pointee.release = releaseStructSchema
        out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
    }

    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

/// Imports a struct array (`+s`) as a record batch. Children are moved out of the parent; the parent
/// struct is released once every child has been imported. Top-level nulls on the struct are not supported.
public func importArrowRecordBatch(schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowArray>,
                                   context: MetalContext = .shared) throws -> (batch: MetalRecordBatch, zeroCopy: Bool) {
    guard let f = schema.pointee.format, String(cString: f) == "+s" else {
        throw ArrowMetalError.unsupportedType("expected struct (+s) schema for a record batch")
    }
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    let n = Int(schema.pointee.n_children)
    guard Int(array.pointee.n_children) == n else { throw ArrowMetalError.invalidArrowArray("schema/array child count mismatch") }
    guard array.pointee.null_count == 0 || array.pointee.null_count == -1, array.pointee.offset == 0 else {
        throw ArrowMetalError.unsupportedType("struct arrays with nulls or offsets are not supported")
    }
    var names: [String] = [], cols: [AnyMetalArray] = []
    var zc = true
    for i in 0..<n {
        let cs = schema.pointee.children[i]!
        let ca = array.pointee.children[i]!
        names.append(cs.pointee.name.map { String(cString: $0) } ?? "")
        // Move the child out so its lifetime is independent of the parent, then import it.
        let moved = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
        moved.initialize(to: ca.pointee)
        ca.pointee.release = nil
        let r = try importArrowArray(schema: cs, array: moved, context: context)
        moved.deallocate()
        cols.append(r.array)
        zc = zc && r.zeroCopy
    }
    // All children are released (nil), the parent's release now only frees the parent's own resources.
    if let rel = array.pointee.release { rel(array) }
    return (try MetalRecordBatch(names: names, columns: cols), zc)
}

/// Drains an `ArrowArrayStream` into Metal-resident record batches, releasing the stream afterwards.
public func importArrowArrayStream(_ stream: UnsafeMutablePointer<ArrowArrayStream>,
                                   context: MetalContext = .shared) throws -> [MetalRecordBatch] {
    guard stream.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    defer { stream.pointee.release?(stream) }
    var schema = ArrowSchema()
    guard stream.pointee.get_schema(stream, &schema) == 0 else {
        throw ArrowMetalError.invalidArrowArray(stream.pointee.get_last_error(stream).map { String(cString: $0) } ?? "get_schema failed")
    }
    defer { schema.release?(&schema) }
    var batches: [MetalRecordBatch] = []
    while true {
        var arr = ArrowArray()
        guard stream.pointee.get_next(stream, &arr) == 0 else {
            throw ArrowMetalError.invalidArrowArray(stream.pointee.get_last_error(stream).map { String(cString: $0) } ?? "get_next failed")
        }
        if arr.release == nil { break }   // end of stream
        batches.append(try importArrowRecordBatch(schema: &schema, array: &arr, context: context).batch)
    }
    return batches
}
