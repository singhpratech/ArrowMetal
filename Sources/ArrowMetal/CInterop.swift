import Foundation
import Metal
import CArrowABI

/// Type-erased Metal-backed Arrow array produced by the C Data Interface importer.
public enum AnyMetalArray {
    case int8(MetalArray<Int8>), uint8(MetalArray<UInt8>)
    case int16(MetalArray<Int16>), uint16(MetalArray<UInt16>)
    case int32(MetalArray<Int32>), uint32(MetalArray<UInt32>)
    case int64(MetalArray<Int64>), uint64(MetalArray<UInt64>)
    case float32(MetalArray<Float>), float64(MetalArray<Double>)
    case boolean(MetalBooleanArray)
    case string(MetalStringArray)
    /// date / time / timestamp / duration, an integer array plus its temporal type.
    case temporal(MetalTemporalArray)
    /// binary / large_binary: utf8's layout with the bytes left uninterpreted.
    case binary(MetalStringArray)
    /// decimal128 / decimal256: raw 16- or 32-byte two's-complement values plus precision and scale.
    case decimal(MetalDecimalArray)
    /// A dictionary-encoded array: int32 codes into `values`.
    indirect case dictionary(codes: MetalArray<Int32>, values: AnyMetalArray)
    /// list / large_list / fixed_size_list: int32 offsets over a child of any supported type.
    case list(MetalListArray)
    /// struct as a column (not only as the record batch container).
    case structure(MetalStructArray)
    /// map: a list of non-nullable `struct<key, value>` entries.
    case map(MetalMapArray)
    /// dense or sparse union.
    case union(MetalUnionArray)
    /// A run-end encoded array ("+r"): `runEnds[j]` is the exclusive end of run `j`, `values[j]` its value.
    /// Run ends are narrowed to int32 on import, as dictionary indices are.
    indirect case runEndEncoded(runEnds: MetalArray<Int32>, values: AnyMetalArray)
    /// The Arrow `null` type ("n"): a length and no buffers (`TypesExtra.swift`).
    case null(MetalNullArray)
    /// float16 ("e"): binary16 bit patterns; compute goes through float32 (`TypesExtra.swift`).
    case float16(MetalFloat16Array)
    /// decimal32 / decimal64 ("d:p,s,32" / "d:p,s,64"); compute goes through decimal128 (`TypesExtra.swift`).
    case smallDecimal(MetalSmallDecimalArray)
    /// interval[month] / interval[day_time] / interval[month_day_nano] (`TypesExtra.swift`).
    case interval(MetalIntervalArray)
    /// fixed_size_binary ("w:N"): N raw bytes per element (`TypesExtra.swift`).
    case fixedBinary(MetalFixedBinaryArray)
    /// An extension type: a storage array plus `ARROW:extension:name` / `:metadata` (`ExtensionType.swift`).
    case extended(MetalExtensionArray)

    public var length: Int {
        switch self {
        case .int8(let a): return a.length
        case .uint8(let a): return a.length
        case .int16(let a): return a.length
        case .uint16(let a): return a.length
        case .int32(let a): return a.length
        case .uint32(let a): return a.length
        case .int64(let a): return a.length
        case .uint64(let a): return a.length
        case .float32(let a): return a.length
        case .float64(let a): return a.length
        case .boolean(let a): return a.length
        case .string(let a): return a.length
        case .temporal(let a): return a.length
        case .binary(let a): return a.length
        case .decimal(let a): return a.length
        case .dictionary(let codes, _): return codes.length
        case .list(let a): return a.length
        case .structure(let a): return a.length
        case .map(let a): return a.length
        case .union(let a): return a.length
        case .runEndEncoded(let runEnds, _): return runEndLogicalLength(runEnds)
        case .null(let a): return a.length
        case .float16(let a): return a.length
        case .smallDecimal(let a): return a.length
        case .interval(let a): return a.length
        case .fixedBinary(let a): return a.length
        case .extended(let a): return a.length
        }
    }

    public var arrowFormat: String {
        switch self {
        case .int8: return "c"
        case .uint8: return "C"
        case .int16: return "s"
        case .uint16: return "S"
        case .int32: return "i"
        case .uint32: return "I"
        case .int64: return "l"
        case .uint64: return "L"
        case .float32: return "f"
        case .float64: return "g"
        case .boolean: return "b"
        case .string: return "u"
        case .temporal(let a): return a.type.arrowFormat
        case .binary: return "z"
        case .decimal(let a): return a.type.arrowFormat
        // The C Data Interface puts the index type at the top level of a dictionary schema.
        case .dictionary: return "i"
        case .list(let a): return a.arrowFormat
        case .structure: return "+s"
        case .map: return "+m"
        case .union(let a): return a.arrowFormat
        case .runEndEncoded: return "+r"
        case .null: return "n"
        case .float16: return "e"
        case .smallDecimal(let a): return a.type.arrowFormat
        case .interval(let a): return a.unit.arrowFormat
        case .fixedBinary(let a): return a.arrowFormat
        // An extension type has no format of its own; the schema carries the name in its metadata.
        case .extended(let a): return a.arrowFormat
        }
    }
}

/// Result of an import: the array plus whether its buffers were borrowed without copying.
public struct ImportResult {
    public let array: AnyMetalArray
    public let zeroCopy: Bool
}

/// Owns a moved-in `ArrowArray` struct and calls its release callback when deallocated.
final class ImportedCArray {
    var array: ArrowArray
    init(moving src: UnsafeMutablePointer<ArrowArray>) {
        array = src.pointee
        // Per spec the consumer moves the struct and marks the source as released.
        src.pointee.release = nil
    }
    deinit {
        if let rel = array.release { rel(&array) }
    }
}

// MARK: - Import

/// Imports a CPU (or Metal-device) Arrow array through the C Data Interface into Metal shared memory.
///
/// Ownership: this call moves `array` (its `release` is set to nil) and releases it when the returned
/// buffers are freed, exactly as the C Data Interface prescribes.
/// Zero-copy: when each buffer pointer is page aligned and the array has no offset, the bytes are
/// wrapped with `makeBuffer(bytesNoCopy:)`; otherwise they are copied once.
public func importArrowArray(schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowArray>,
                             context: MetalContext = .shared) throws -> ImportResult {
    guard let fmtC = schema.pointee.format else { throw ArrowMetalError.invalidArrowArray("schema.format is null") }
    let fmt = String(cString: fmtC)
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    // An extension type is a storage type plus two metadata keys; the storage goes through this function
    // again with the keys stripped (`ExtensionType.swift`).
    if let info = extensionInfo(schema) {
        return try importExtensionArray(info, schema: schema, array: array, context: context)
    }
    // The schema decides whether an array is dictionary-encoded; the indices are imported inside.
    if schema.pointee.dictionary != nil { return try importDictionaryArray(schema: schema, array: array, context: context) }
    // null, float16, decimal32/64, interval, fixed_size_binary and the list views (`TypesExtra.swift`).
    if let r = try importExtraFormats(format: fmt, schema: schema, array: array, context: context) { return r }
    // Nested types (list, large_list, fixed_size_list, struct, map, union) recurse through this function.
    if isNestedFormat(fmt) { return try importNestedArray(format: fmt, schema: schema, array: array, context: context) }
    // Run-end encoded arrays carry their run ends and values as two children.
    if fmt == "+r" { return try importRunEndArray(schema: schema, array: array, context: context) }
    guard array.pointee.n_children == 0, array.pointee.dictionary == nil else {
        throw ArrowMetalError.unsupportedType("nested/dictionary arrays are not supported (format \(fmt))")
    }
    if fmt == "u" || fmt == "U" { return try importStringArray(large: fmt == "U", array: array, context: context) }
    if fmt == "z" || fmt == "Z" { return try importBinaryArray(large: fmt == "Z", array: array, context: context) }
    if fmt.hasPrefix("d:") { return try importDecimalArray(type: try ArrowDecimalType.parse(fmt), array: array, context: context) }
    if fmt.hasPrefix("t") { return try importTemporalArray(type: try ArrowTemporalType.parse(fmt), array: array, context: context) }
    guard array.pointee.n_buffers == 2, array.pointee.buffers != nil else {
        throw ArrowMetalError.invalidArrowArray("expected 2 buffers for primitive array, got \(array.pointee.n_buffers)")
    }
    // Fast path: an array exported by ArrowMetal in this process. Share the MTLBuffer objects directly.
    if let own = ownedExportBuffers(array) {
        let owner = ImportedCArray(moving: array)
        let a = owner.array
        let arr = try rebuild(format: fmt, offset: Int(a.offset), length: Int(a.length), nullCount: Int(a.null_count),
                              validity: own.validity, values: own.values, context: context)
        return ImportResult(array: arr, zeroCopy: true)
    }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let validityPtr = a.buffers[0].map { UnsafeRawPointer($0) }
    guard let valuesPtr = a.buffers[1].map({ UnsafeRawPointer($0) }) else {
        throw ArrowMetalError.invalidArrowArray("values buffer is null")
    }
    var nullCount = Int(a.null_count)

    if fmt == "b" {
        // A non-zero Arrow offset is carried, not applied: both bitmaps are wrapped whole and the array
        // starts at bit `offset` (see `MetalArray.offset`), so a sliced producer imports zero-copy too.
        let bits = offset + length
        let (vals, zc1) = try bitBuffer(valuesPtr, length: bits, offset: 0, owner: owner, context: context)
        var validity: MetalArrowBuffer? = nil
        var zc2 = true
        if let vp = validityPtr { (validity, zc2) = try bitBuffer(vp, length: bits, offset: 0, owner: owner, context: context) }
        let arr = MetalBooleanArray(offset: offset, length: length, validity: validity, values: vals,
                                    nullCount: nullCount < 0 ? nil : nullCount, context: context)
        return ImportResult(array: .boolean(arr), zeroCopy: zc1 && zc2)
    }

    guard let ty = arrowPrimitiveType(forFormat: fmt) else { throw ArrowMetalError.unsupportedType(fmt) }

    func build<T: ArrowPrimitive>(_: T.Type) throws -> (MetalArray<T>, Bool) {
        // The producer's `offset` is carried on the array rather than applied, so an array sliced by the
        // producer (pyarrow's `Array.slice`, a Polars column view) imports without touching a byte.
        let (vals, zc1) = try MetalArrowBuffer.wrapOrCopy(valuesPtr, byteCount: (offset + length) * T.byteWidth,
                                                          keepAlive: owner, context: context)
        var validity: MetalArrowBuffer? = nil
        var zc2 = true
        if let vp = validityPtr { (validity, zc2) = try bitBuffer(vp, length: offset + length, offset: 0, owner: owner, context: context) }
        let arr = MetalArray<T>(offset: offset, length: length, validity: validity, values: vals,
                                nullCount: nullCount < 0 ? nil : nullCount, context: context)
        if nullCount < 0 { nullCount = arr.nullCount }
        return (arr, zc1 && zc2)
    }

    switch ty {
    case is Int8.Type: let (a, z) = try build(Int8.self); return ImportResult(array: .int8(a), zeroCopy: z)
    case is UInt8.Type: let (a, z) = try build(UInt8.self); return ImportResult(array: .uint8(a), zeroCopy: z)
    case is Int16.Type: let (a, z) = try build(Int16.self); return ImportResult(array: .int16(a), zeroCopy: z)
    case is UInt16.Type: let (a, z) = try build(UInt16.self); return ImportResult(array: .uint16(a), zeroCopy: z)
    case is Int32.Type: let (a, z) = try build(Int32.self); return ImportResult(array: .int32(a), zeroCopy: z)
    case is UInt32.Type: let (a, z) = try build(UInt32.self); return ImportResult(array: .uint32(a), zeroCopy: z)
    case is Int64.Type: let (a, z) = try build(Int64.self); return ImportResult(array: .int64(a), zeroCopy: z)
    case is UInt64.Type: let (a, z) = try build(UInt64.self); return ImportResult(array: .uint64(a), zeroCopy: z)
    case is Float.Type: let (a, z) = try build(Float.self); return ImportResult(array: .float32(a), zeroCopy: z)
    case is Double.Type: let (a, z) = try build(Double.self); return ImportResult(array: .float64(a), zeroCopy: z)
    default: throw ArrowMetalError.unsupportedType(fmt)
    }
}

/// utf8 / large_utf8 import. Zero-copy for utf8 when page aligned and offset 0; large_utf8 offsets are narrowed
/// (one pass) when the data is under 2 GB.
func importStringArray(large: Bool, array: UnsafeMutablePointer<ArrowArray>, context: MetalContext) throws -> ImportResult {
    guard array.pointee.n_buffers == 3, array.pointee.buffers != nil else { throw ArrowMetalError.invalidArrowArray("expected 3 buffers for utf8") }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let validityPtr = a.buffers[0].map { UnsafeRawPointer($0) }
    guard let offPtr = a.buffers[1].map({ UnsafeRawPointer($0) }) else { throw ArrowMetalError.invalidArrowArray("offsets buffer is null") }
    let dataPtr = a.buffers[2].map { UnsafeRawPointer($0) }
    var zc = true
    var validity: MetalArrowBuffer? = nil
    if let vp = validityPtr { let (b, z) = try bitBuffer(vp, length: length, offset: offset, owner: owner, context: context); validity = b; zc = zc && z }
    let offsets: MetalArrowBuffer
    let data: MetalArrowBuffer
    if !large && offset == 0 {
        let (ob, z1) = try MetalArrowBuffer.wrapOrCopy(offPtr, byteCount: (length + 1) * 4, keepAlive: owner, context: context)
        let total = Int(ob.typed(Int32.self)[length])
        let (db, z2) = try MetalArrowBuffer.wrapOrCopy(dataPtr ?? offPtr, byteCount: total, keepAlive: owner, context: context)
        offsets = ob; data = db; zc = zc && z1 && z2
    } else {
        // Materialise: narrow / rebase offsets so the slice starts at 0.
        zc = false
        let ob = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
        let op = ob.mutableTyped(Int32.self)
        var start = 0, end = 0
        if large {
            let src = offPtr.assumingMemoryBound(to: Int64.self)
            start = Int(src[offset]); end = Int(src[offset + length])
            guard end - start < Int(Int32.max) else { throw ArrowMetalError.unsupportedType("large_utf8 over 2 GB") }
            for i in 0...length { op[i] = Int32(Int(src[offset + i]) - start) }
        } else {
            let src = offPtr.assumingMemoryBound(to: Int32.self)
            start = Int(src[offset]); end = Int(src[offset + length])
            for i in 0...length { op[i] = src[offset + i] - Int32(start) }
        }
        let db = try MetalArrowBuffer.allocate(byteCount: end - start, zeroed: false, context: context)
        if end > start, let dp = dataPtr { memcpy(db.mutableContents, dp + start, end - start) }
        offsets = ob; data = db
    }
    let arr = MetalStringArray(length: length, nullCount: 0, validity: validity, offsets: offsets, data: data, context: context)
    let nc = Int(a.null_count)
    if nc < 0 { arr.recomputeNullCount() } else { arr.setNullCount(nc) }
    return ImportResult(array: .string(arr), zeroCopy: zc)
}

/// Imports an `ArrowDeviceArray`. Metal and CPU device types are accepted; both point at unified memory.
public func importArrowDeviceArray(schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowDeviceArray>,
                                   context: MetalContext = .shared) throws -> ImportResult {
    let dt = array.pointee.device_type
    guard dt == ARROW_DEVICE_METAL || dt == ARROW_DEVICE_CPU else {
        throw ArrowMetalError.unsupportedType("device_type \(dt) is not Metal or CPU")
    }
    if let ev = array.pointee.sync_event {
        // Spec: MTLEvent* for Metal. We cannot wait on a bare MTLEvent without a command buffer, so
        // encode an empty command buffer that waits for it. Producers that already synchronised pass NULL.
        let event = Unmanaged<AnyObject>.fromOpaque(ev).takeUnretainedValue()
        if let sharedEvent = event as? MTLSharedEvent {
            // Wait for the most recent signalled value to be reached.
            let cb = context.queue.makeCommandBuffer()!
            cb.encodeWaitForEvent(sharedEvent, value: sharedEvent.signaledValue)
            cb.commit(); cb.waitUntilCompleted()
        }
    }
    return try withUnsafeMutablePointer(to: &array.pointee.array) { inner in
        try importArrowArray(schema: schema, array: inner, context: context)
    }
}

/// Bitmap buffer import: zero-copy when page aligned and no offset, otherwise repacks with the bit offset applied.
private func bitBuffer(_ src: UnsafeRawPointer, length: Int, offset: Int, owner: AnyObject,
                       context: MetalContext) throws -> (MetalArrowBuffer, Bool) {
    let bytes = Bitmap.byteCount(bits: length)
    if offset == 0 {
        return try MetalArrowBuffer.wrapOrCopy(src, byteCount: bytes, keepAlive: owner, context: context)
    }
    let buf = try MetalArrowBuffer.allocate(byteCount: bytes, context: context)
    let sp = src.assumingMemoryBound(to: UInt8.self)
    let dp = buf.mutableTyped(UInt8.self)
    if offset % 8 == 0 {
        memcpy(dp, sp + offset / 8, bytes)
    } else {
        for i in 0..<length where Bitmap.isSet(sp, i + offset) { Bitmap.set(dp, i) }
    }
    return (buf, false)
}

extension MetalArray {
    func setNullCount(_ n: Int) { self.nullCount = n }
}
extension MetalBooleanArray {
    func setNullCount(_ n: Int) { self.nullCount = n }
}
extension MetalStringArray {
    func setNullCount(_ n: Int) { self.nullCount = n }
    public func recomputeNullCount() {
        guard let v = validity else { nullCount = 0; return }
        nullCount = length - Bitmap.popcount(v.typed(UInt8.self), bits: length)
    }
}

// MARK: - Export

/// Holder retained through `private_data` for exported structs.
final class ExportHolder {
    let keep: [AnyObject]
    let buffers: UnsafeMutablePointer<UnsafeRawPointer?>
    let nBuffers: Int
    let formatC: UnsafeMutablePointer<CChar>?
    let nameC: UnsafeMutablePointer<CChar>?
    init(keep: [AnyObject], bufferPtrs: [UnsafeRawPointer?], format: String?, name: String?) {
        self.keep = keep
        nBuffers = bufferPtrs.count
        buffers = .allocate(capacity: max(bufferPtrs.count, 1))
        for (i, p) in bufferPtrs.enumerated() { buffers[i] = p }
        formatC = format.map { strdup($0) }
        nameC = name.map { strdup($0) }
    }
    deinit {
        buffers.deallocate()
        if let f = formatC { free(f) }
        if let n = nameC { free(n) }
    }
}

private func releaseExportedArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<ExportHolder>.fromOpaque(pd).release()
    p.pointee.release = nil
    p.pointee.private_data = nil
}

private func releaseExportedSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<ExportHolder>.fromOpaque(pd).release()
    p.pointee.release = nil
    p.pointee.private_data = nil
}

/// Exports a schema for a primitive, nullable, top-level field.
public func exportArrowSchema(format: String, name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
    let holder = ExportHolder(keep: [], bufferPtrs: [], format: format, name: name)
    out.pointee.format = UnsafePointer(holder.formatC)
    out.pointee.name = UnsafePointer(holder.nameC)
    out.pointee.metadata = nil
    out.pointee.flags = Int64(ARROW_FLAG_NULLABLE)
    out.pointee.n_children = 0
    out.pointee.children = nil
    out.pointee.dictionary = nil
    out.pointee.release = releaseExportedSchema
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}

private func fillExportedArray(length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer,
                               offset: Int = 0, keep: AnyObject, into out: UnsafeMutablePointer<ArrowArray>) {
    var keeps: [AnyObject] = [keep, values]
    if let v = validity { keeps.append(v) }
    let holder = ExportHolder(keep: keeps, bufferPtrs: [validity?.contents, values.contents], format: nil, name: nil)
    out.pointee.length = Int64(length)
    out.pointee.null_count = Int64(nullCount)
    // Arrow's own `offset`: a slice at an offset that is not a multiple of 32 exports the parent's buffers
    // with the offset attached, so export stays zero-copy at every offset.
    out.pointee.offset = Int64(offset)
    out.pointee.n_buffers = 2
    out.pointee.n_children = 0
    out.pointee.buffers = UnsafeMutablePointer<UnsafeRawPointer?>(holder.buffers)
    out.pointee.children = nil
    out.pointee.dictionary = nil
    out.pointee.release = releaseExportedArray
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}

private func fillDevice(_ out: UnsafeMutablePointer<ArrowDeviceArray>) {
    out.pointee.device_type = ARROW_DEVICE_METAL
    out.pointee.device_id = -1
    out.pointee.reserved = (0, 0, 0)
    // All ArrowMetal kernels block until the GPU has finished, so buffers are always ready.
    out.pointee.sync_event = nil
}

extension MetalArray {
    /// Exports through the CPU C Data Interface. Because the buffers are in unified memory this is zero-copy.
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        ensure()
        fillExportedArray(length: length, nullCount: nullCount, validity: rawValidity, values: rawValues,
                          offset: offset, keep: self, into: out)
    }
    /// Exports through the C Device Data Interface with `device_type = ARROW_DEVICE_METAL`. Also zero-copy.
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: T.arrowFormat, name: name, into: out)
    }
}

extension MetalBooleanArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        ensure()
        fillExportedArray(length: length, nullCount: nullCount, validity: rawValidity, values: rawValues,
                          offset: offset, keep: self, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: "b", name: name, into: out)
    }
}

extension MetalStringArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        var keeps: [AnyObject] = [self, offsets, data]
        if let v = validity { keeps.append(v) }
        let holder = ExportHolder(keep: keeps, bufferPtrs: [validity?.contents, offsets.contents, data.contents], format: nil, name: nil)
        out.pointee.length = Int64(length)
        out.pointee.null_count = Int64(nullCount)
        out.pointee.offset = 0
        out.pointee.n_buffers = 3
        out.pointee.n_children = 0
        out.pointee.buffers = UnsafeMutablePointer<UnsafeRawPointer?>(holder.buffers)
        out.pointee.children = nil
        out.pointee.dictionary = nil
        out.pointee.release = releaseExportedArray
        out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: isBinary ? "z" : "u", name: name, into: out)
    }
}

extension AnyMetalArray {
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        switch self {
        case .int8(let a): a.exportArrowArray(into: out)
        case .uint8(let a): a.exportArrowArray(into: out)
        case .int16(let a): a.exportArrowArray(into: out)
        case .uint16(let a): a.exportArrowArray(into: out)
        case .int32(let a): a.exportArrowArray(into: out)
        case .uint32(let a): a.exportArrowArray(into: out)
        case .int64(let a): a.exportArrowArray(into: out)
        case .uint64(let a): a.exportArrowArray(into: out)
        case .float32(let a): a.exportArrowArray(into: out)
        case .float64(let a): a.exportArrowArray(into: out)
        case .boolean(let a): a.exportArrowArray(into: out)
        case .string(let a): a.exportArrowArray(into: out)
        case .temporal(let a): a.exportArrowArray(into: out)
        case .binary(let a): a.exportArrowArray(into: out)
        case .decimal(let a): a.exportArrowArray(into: out)
        case .dictionary(let codes, let values): exportDictionaryArray(codes: codes, values: values, into: out)
        case .list(let a): a.exportArrowArray(into: out)
        case .structure(let a): a.exportArrowArray(into: out)
        case .map(let a): a.exportArrowArray(into: out)
        case .union(let a): a.exportArrowArray(into: out)
        case .runEndEncoded(let runEnds, let values): exportRunEndArray(runEnds: runEnds, values: values, into: out)
        case .null(let a): a.exportArrowArray(into: out)
        case .float16(let a): a.exportArrowArray(into: out)
        case .smallDecimal(let a): a.exportArrowArray(into: out)
        case .interval(let a): a.exportArrowArray(into: out)
        case .fixedBinary(let a): a.exportArrowArray(into: out)
        case .extended(let a): a.exportArrowArray(into: out)
        }
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        // A dictionary schema carries the value type under `dictionary`; everything else is a flat format.
        if case .dictionary(_, let values) = self {
            exportDictionarySchema(values: values, name: name, into: out)
            return
        }
        // An extension schema is the storage schema plus the extension metadata.
        if case .extended(let a) = self { return a.exportArrowSchema(name: name, into: out) }
        // A nested schema carries its children, so each nested array writes its own.
        switch self {
        case .list(let a): return a.exportArrowSchema(name: name, into: out)
        case .structure(let a): return a.exportArrowSchema(name: name, into: out)
        case .map(let a): return a.exportArrowSchema(name: name, into: out)
        case .union(let a): return a.exportArrowSchema(name: name, into: out)
        case .runEndEncoded(_, let values): return exportRunEndSchema(values: values, name: name, into: out)
        default: break
        }
        ArrowMetal.exportArrowSchema(format: arrowFormat, name: name, into: out)
    }
}

/// Returns the `MetalArrowBuffer`s behind an ArrowMetal-exported `ArrowArray`, or nil for foreign arrays.
func ownedExportBuffers(_ array: UnsafePointer<ArrowArray>) -> (validity: MetalArrowBuffer?, values: MetalArrowBuffer)? {
    guard let rel = array.pointee.release, let pd = array.pointee.private_data else { return nil }
    let ours: @convention(c) (UnsafeMutablePointer<ArrowArray>?) -> Void = releaseExportedArray
    guard unsafeBitCast(rel, to: UnsafeRawPointer.self) == unsafeBitCast(ours, to: UnsafeRawPointer.self) else { return nil }
    let holder = Unmanaged<ExportHolder>.fromOpaque(pd).takeUnretainedValue()
    guard holder.nBuffers == 2 else { return nil }
    var validity: MetalArrowBuffer? = nil
    var values: MetalArrowBuffer? = nil
    for k in holder.keep {
        if let b = k as? MetalArrowBuffer {
            if let p = holder.buffers[0], b.contents == p { validity = b }
            if let p = holder.buffers[1], b.contents == p { values = b }
        }
    }
    guard let v = values else { return nil }
    return (validity, v)
}

/// Builds a typed array from already-owned buffers.
private func rebuild(format: String, offset: Int = 0, length: Int, nullCount: Int, validity: MetalArrowBuffer?,
                     values: MetalArrowBuffer, context: MetalContext) throws -> AnyMetalArray {
    func mk<T: ArrowPrimitive>(_: T.Type) -> MetalArray<T> {
        MetalArray<T>(offset: offset, length: length, validity: validity, values: values,
                      nullCount: nullCount < 0 ? nil : nullCount, context: context)
    }
    switch format {
    case "b":
        return .boolean(MetalBooleanArray(offset: offset, length: length, validity: validity, values: values,
                                          nullCount: nullCount < 0 ? nil : nullCount, context: context))
    case "c": return .int8(mk(Int8.self))
    case "C": return .uint8(mk(UInt8.self))
    case "s": return .int16(mk(Int16.self))
    case "S": return .uint16(mk(UInt16.self))
    case "i": return .int32(mk(Int32.self))
    case "I": return .uint32(mk(UInt32.self))
    case "l": return .int64(mk(Int64.self))
    case "L": return .uint64(mk(UInt64.self))
    case "f": return .float32(mk(Float.self))
    case "g": return .float64(mk(Double.self))
    default: throw ArrowMetalError.unsupportedType(format)
    }
}

/// If `array` was exported by ArrowMetal, returns the underlying `MTLBuffer`s (validity, values) so a Metal
/// consumer can bind them directly instead of re-wrapping raw pointers. Returns nil for foreign arrays.
public func metalBuffers(of array: UnsafePointer<ArrowDeviceArray>) -> [MTLBuffer?]? {
    guard let own = withUnsafePointer(to: array.pointee.array, { ownedExportBuffers($0) }) else { return nil }
    return [own.validity?.mtl, own.values.mtl]
}
