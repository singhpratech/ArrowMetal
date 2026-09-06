import Foundation
import CArrowABI

// Arrow dictionary-encoded arrays. The C Data Interface puts the *index* type in `schema.format` and hangs
// the value type off `schema.dictionary` / `array.dictionary`. We keep the two apart — int32 codes plus a
// type-erased value array — so the codes feed `group_by` directly and `decode()` materialises with `take`.

extension AnyMetalArray {
    /// Materialises a dictionary array by gathering its values with `take`. Other arrays are returned as-is.
    public func decode() throws -> AnyMetalArray {
        guard case .dictionary(let codes, let values) = self else { return self }
        return try values.take(codes)
    }

    /// The (codes, values) pair of a dictionary array, or nil.
    public var asDictionary: (codes: MetalArray<Int32>, values: AnyMetalArray)? {
        if case .dictionary(let c, let v) = self { return (c, v) }
        return nil
    }
}

// MARK: - Import

/// Imports a dictionary-encoded array. Indices must be int32 or int64 (int64 is narrowed to int32);
/// the values can be any array this importer understands (utf8, binary, primitive, temporal).
func importDictionaryArray(schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowArray>,
                           context: MetalContext) throws -> ImportResult {
    guard let dictSchema = schema.pointee.dictionary else {
        throw ArrowMetalError.invalidArrowArray("dictionary schema is null")
    }
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    guard let dictArray = array.pointee.dictionary else {
        throw ArrowMetalError.invalidArrowArray("dictionary-encoded array has no dictionary values")
    }
    guard let fmtC = schema.pointee.format else { throw ArrowMetalError.invalidArrowArray("schema.format is null") }
    let idxFormat = String(cString: fmtC)
    guard idxFormat == "i" || idxFormat == "l" else {
        throw ArrowMetalError.unsupportedType("dictionary indices must be int32 or int64, got \(idxFormat)")
    }
    // Move the values out so their lifetime is independent of the indices, exactly as for struct children.
    let moved = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
    moved.initialize(to: dictArray.pointee)
    dictArray.pointee.release = nil
    array.pointee.dictionary = nil
    defer { moved.deallocate() }
    let valuesResult = try importArrowArray(schema: dictSchema, array: moved, context: context)
    // Indices go through the primitive path with a dictionary-free schema.
    let idxFmtC = strdup(idxFormat)!
    defer { free(idxFmtC) }
    let idxSchema = UnsafeMutablePointer<ArrowSchema>.allocate(capacity: 1)
    idxSchema.initialize(to: ArrowSchema())
    defer { idxSchema.deallocate() }
    idxSchema.pointee.format = UnsafePointer(idxFmtC)
    let idxResult = try importArrowArray(schema: idxSchema, array: array, context: context)
    let codes: MetalArray<Int32>
    switch idxResult.array {
    case .int32(let c): codes = c
    case .int64(let c): codes = try c.cast(to: Int32.self)
    default: throw ArrowMetalError.invalidArrowArray("dictionary indices must be int32 or int64")
    }
    return ImportResult(array: .dictionary(codes: codes, values: valuesResult.array),
                        zeroCopy: idxResult.zeroCopy && valuesResult.zeroCopy)
}

// MARK: - Export

/// Owns the two exported child structs of a dictionary schema.
private final class DictSchemaHolder {
    let dict: UnsafeMutablePointer<ArrowSchema>
    let formatC: UnsafeMutablePointer<CChar>
    let nameC: UnsafeMutablePointer<CChar>
    init(format: String, name: String) {
        dict = .allocate(capacity: 1)
        dict.initialize(to: ArrowSchema())
        formatC = strdup(format)!
        nameC = strdup(name)!
    }
    deinit {
        if let r = dict.pointee.release { r(dict) }
        dict.deallocate(); free(formatC); free(nameC)
    }
}

/// Owns the exported indices array (whose buffers the parent points at) and the exported values array.
private final class DictArrayHolder {
    let indices: UnsafeMutablePointer<ArrowArray>
    let dict: UnsafeMutablePointer<ArrowArray>
    init() {
        indices = .allocate(capacity: 1); indices.initialize(to: ArrowArray())
        dict = .allocate(capacity: 1); dict.initialize(to: ArrowArray())
    }
    deinit {
        if let r = indices.pointee.release { r(indices) }
        if let r = dict.pointee.release { r(dict) }
        indices.deallocate(); dict.deallocate()
    }
}

private func releaseDictSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<DictSchemaHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

private func releaseDictArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<DictArrayHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

/// Writes a dictionary schema: the index type at the top level, the value type under `dictionary`.
func exportDictionarySchema(values: AnyMetalArray, name: String, into out: UnsafeMutablePointer<ArrowSchema>) {
    let holder = DictSchemaHolder(format: "i", name: name)
    values.exportArrowSchema(into: holder.dict)
    out.pointee.format = UnsafePointer(holder.formatC)
    out.pointee.name = UnsafePointer(holder.nameC)
    out.pointee.metadata = nil
    out.pointee.flags = Int64(ARROW_FLAG_NULLABLE)
    out.pointee.n_children = 0
    out.pointee.children = nil
    out.pointee.dictionary = holder.dict
    out.pointee.release = releaseDictSchema
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}

/// Writes a dictionary array: the indices' buffers at the top level, the values under `dictionary`.
func exportDictionaryArray(codes: MetalArray<Int32>, values: AnyMetalArray, into out: UnsafeMutablePointer<ArrowArray>) {
    let holder = DictArrayHolder()
    codes.exportArrowArray(into: holder.indices)
    values.exportArrowArray(into: holder.dict)
    out.pointee.length = holder.indices.pointee.length
    out.pointee.null_count = holder.indices.pointee.null_count
    out.pointee.offset = 0
    out.pointee.n_buffers = holder.indices.pointee.n_buffers
    out.pointee.n_children = 0
    out.pointee.buffers = holder.indices.pointee.buffers      // owned by the indices export, kept alive by holder
    out.pointee.children = nil
    out.pointee.dictionary = holder.dict
    out.pointee.release = releaseDictArray
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}
