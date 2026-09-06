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
    guard array.pointee.n_children == 0, array.pointee.dictionary == nil else {
        throw ArrowMetalError.unsupportedType("nested/dictionary arrays are not supported (format \(fmt))")
    }
    guard array.pointee.n_buffers == 2 else {
        throw ArrowMetalError.invalidArrowArray("expected 2 buffers for primitive array, got \(array.pointee.n_buffers)")
    }
    // Fast path: an array exported by ArrowMetal in this process. Share the MTLBuffer objects directly.
    if let own = ownedExportBuffers(array) {
        let owner = ImportedCArray(moving: array)
        let a = owner.array
        let arr = try rebuild(format: fmt, length: Int(a.length), nullCount: Int(a.null_count),
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
        let (vals, zc1) = try bitBuffer(valuesPtr, length: length, offset: offset, owner: owner, context: context)
        var validity: MetalArrowBuffer? = nil
        var zc2 = true
        if let vp = validityPtr { (validity, zc2) = try bitBuffer(vp, length: length, offset: offset, owner: owner, context: context) }
        let arr = MetalBooleanArray(length: length, nullCount: 0, validity: validity, values: vals, context: context)
        if nullCount < 0 { arr.recomputeNullCount() } else { arr.setNullCount(nullCount) }
        return ImportResult(array: .boolean(arr), zeroCopy: zc1 && zc2)
    }

    guard let ty = arrowPrimitiveType(forFormat: fmt) else { throw ArrowMetalError.unsupportedType(fmt) }

    func build<T: ArrowPrimitive>(_: T.Type) throws -> (MetalArray<T>, Bool) {
        let byteCount = length * T.byteWidth
        let src = valuesPtr.advanced(by: offset * T.byteWidth)
        let (vals, zc1) = try MetalArrowBuffer.wrapOrCopy(src, byteCount: byteCount, keepAlive: owner, context: context)
        var validity: MetalArrowBuffer? = nil
        var zc2 = true
        if let vp = validityPtr { (validity, zc2) = try bitBuffer(vp, length: length, offset: offset, owner: owner, context: context) }
        let arr = MetalArray<T>(length: length, nullCount: 0, validity: validity, values: vals, context: context)
        if nullCount < 0 { arr.recomputeNullCount(); nullCount = arr.nullCount } else { arr.setNullCount(nullCount) }
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
                               keep: AnyObject, into out: UnsafeMutablePointer<ArrowArray>) {
    var keeps: [AnyObject] = [keep, values]
    if let v = validity { keeps.append(v) }
    let holder = ExportHolder(keep: keeps, bufferPtrs: [validity?.contents, values.contents], format: nil, name: nil)
    out.pointee.length = Int64(length)
    out.pointee.null_count = Int64(nullCount)
    out.pointee.offset = 0
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
        fillExportedArray(length: length, nullCount: nullCount, validity: validity, values: values, keep: self, into: out)
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
        fillExportedArray(length: length, nullCount: nullCount, validity: validity, values: values, keep: self, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: "b", name: name, into: out)
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
        }
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        fillDevice(out)
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
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
private func rebuild(format: String, length: Int, nullCount: Int, validity: MetalArrowBuffer?, values: MetalArrowBuffer,
                     context: MetalContext) throws -> AnyMetalArray {
    func mk<T: ArrowPrimitive>(_: T.Type) -> MetalArray<T> {
        let a = MetalArray<T>(length: length, nullCount: 0, validity: validity, values: values, context: context)
        if nullCount < 0 { a.recomputeNullCount() } else { a.setNullCount(nullCount) }
        return a
    }
    switch format {
    case "b":
        let a = MetalBooleanArray(length: length, nullCount: 0, validity: validity, values: values, context: context)
        if nullCount < 0 { a.recomputeNullCount() } else { a.setNullCount(nullCount) }
        return .boolean(a)
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
