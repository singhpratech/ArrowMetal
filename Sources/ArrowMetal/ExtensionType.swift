import Foundation
import CArrowABI

// Arrow extension types over the C Data Interface.
//
// An extension type is a storage type plus two well-known schema metadata keys: `ARROW:extension:name`
// (the type's name, e.g. "arrow.uuid") and `ARROW:extension:metadata` (an opaque, possibly empty
// serialisation of the type's parameters). Nothing about the *array* changes — the buffers are the
// storage type's buffers — so ArrowMetal imports the storage array with the existing importer and keeps
// the two strings beside it, and the exporter writes them back so pyarrow re-creates the extension type
// (when it is registered) instead of handing back bare storage.
//
// The metadata blob layout is the one the C Data Interface prescribes, in native endianness:
//
//     int32 n
//     n times: int32 key_length, key bytes, int32 value_length, value bytes
//
// Keys other than the two extension ones are carried through unchanged, so a field that also has, say,
// PARQUET:field_id keeps it across a round trip.

/// Ordered key/value metadata attached to an `ArrowSchema`.
public struct ArrowSchemaMetadata: Sendable, Equatable {
    public var pairs: [Pair]

    public struct Pair: Sendable, Equatable {
        public var key: String
        public var value: [UInt8]
        public init(key: String, value: [UInt8]) { self.key = key; self.value = value }
        public init(key: String, string: String) { self.key = key; self.value = Array(string.utf8) }
        /// The value decoded as UTF-8 (extension metadata is usually text, sometimes empty).
        public var stringValue: String { String(decoding: value, as: UTF8.self) }
    }

    public static let extensionNameKey = "ARROW:extension:name"
    public static let extensionMetadataKey = "ARROW:extension:metadata"

    public init(_ pairs: [Pair] = []) { self.pairs = pairs }
    public var isEmpty: Bool { pairs.isEmpty }
    public subscript(key: String) -> [UInt8]? {
        get { pairs.first { $0.key == key }?.value }
        set {
            if let i = pairs.firstIndex(where: { $0.key == key }) {
                if let newValue { pairs[i].value = newValue } else { pairs.remove(at: i) }
            } else if let newValue {
                pairs.append(Pair(key: key, value: newValue))
            }
        }
    }
    public func string(_ key: String) -> String? { self[key].map { String(decoding: $0, as: UTF8.self) } }

    /// Decodes the C Data Interface metadata blob. Returns an empty set for a null pointer.
    public static func decode(_ p: UnsafePointer<CChar>?) -> ArrowSchemaMetadata {
        guard let p else { return ArrowSchemaMetadata() }
        var cursor = UnsafeRawPointer(p)
        let n = Int(cursor.loadUnaligned(as: Int32.self))
        cursor += 4
        guard n > 0 else { return ArrowSchemaMetadata() }
        var out: [Pair] = []
        out.reserveCapacity(n)
        for _ in 0..<n {
            let kl = Int(cursor.loadUnaligned(as: Int32.self)); cursor += 4
            guard kl >= 0 else { return ArrowSchemaMetadata(out) }
            let key = String(decoding: UnsafeRawBufferPointer(start: cursor, count: kl), as: UTF8.self)
            cursor += kl
            let vl = Int(cursor.loadUnaligned(as: Int32.self)); cursor += 4
            guard vl >= 0 else { return ArrowSchemaMetadata(out) }
            let value = Array(UnsafeRawBufferPointer(start: cursor, count: vl).bindMemory(to: UInt8.self))
            cursor += vl
            out.append(Pair(key: key, value: value))
        }
        return ArrowSchemaMetadata(out)
    }

    /// Encodes the C Data Interface metadata blob (native endian int32 lengths).
    public func encoded() -> [UInt8] {
        var out: [UInt8] = []
        func put(_ v: Int32) { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
        put(Int32(pairs.count))
        for p in pairs {
            let k = Array(p.key.utf8)
            put(Int32(k.count)); out.append(contentsOf: k)
            put(Int32(p.value.count)); out.append(contentsOf: p.value)
        }
        return out
    }
}

/// An Arrow extension array: a storage array plus the extension name and metadata that name the logical
/// type. Every kernel runs on the storage; the two strings ride along and come back out on export.
public final class MetalExtensionArray: @unchecked Sendable {
    /// The array the buffers actually belong to (any supported type, nested included).
    public let storage: AnyMetalArray
    /// `ARROW:extension:name`.
    public let name: String
    /// `ARROW:extension:metadata`, or nil when the schema carried no such key.
    public let metadata: [UInt8]?
    /// Every other metadata key of the field, carried through unchanged.
    public let otherMetadata: ArrowSchemaMetadata

    public init(storage: AnyMetalArray, name: String, metadata: [UInt8]? = nil,
                otherMetadata: ArrowSchemaMetadata = ArrowSchemaMetadata()) {
        self.storage = storage; self.name = name; self.metadata = metadata; self.otherMetadata = otherMetadata
    }

    public var length: Int { storage.length }
    public var nullCount: Int { storage.nullCount }
    /// The *storage* format string; an extension type has no format of its own.
    public var arrowFormat: String { storage.arrowFormat }
    /// `ARROW:extension:metadata` decoded as UTF-8, which is how every built-in extension type spells it.
    public var metadataString: String? { metadata.map { String(decoding: $0, as: UTF8.self) } }

    func rewrapping(_ newStorage: AnyMetalArray) -> MetalExtensionArray {
        MetalExtensionArray(storage: newStorage, name: name, metadata: metadata, otherMetadata: otherMetadata)
    }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalExtensionArray { rewrapping(try storage.filter(mask)) }
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalExtensionArray { rewrapping(try storage.take(indices)) }
    public func slice(offset: Int, length: Int) throws -> MetalExtensionArray {
        rewrapping(try storage.slice(offset: offset, length: length))
    }

    /// The full metadata an exported schema carries: the caller's other keys plus the extension keys.
    public func exportMetadata() -> ArrowSchemaMetadata {
        var m = otherMetadata
        m[ArrowSchemaMetadata.extensionNameKey] = Array(name.utf8)
        if let md = metadata { m[ArrowSchemaMetadata.extensionMetadataKey] = md }
        return m
    }
}

extension AnyMetalArray {
    public var asExtension: MetalExtensionArray? { if case .extended(let a) = self { return a } else { return nil } }
    /// `ARROW:extension:name`, or nil when this is not an extension array.
    public var extensionName: String? { asExtension?.name }
    /// `ARROW:extension:metadata` as bytes, or nil.
    public var extensionMetadata: [UInt8]? { asExtension?.metadata }
    /// The storage array of an extension array; every other array is its own storage.
    public var storageArray: AnyMetalArray { asExtension?.storage ?? self }
    /// Wraps this array as the storage of an extension type.
    public func asExtensionType(name: String, metadata: [UInt8]? = nil) -> AnyMetalArray {
        .extended(MetalExtensionArray(storage: storageArray, name: name, metadata: metadata))
    }
}

// MARK: - Import

/// The extension name and metadata carried by a schema, or nil when it declares no extension type.
func extensionInfo(_ schema: UnsafePointer<ArrowSchema>)
    -> (name: String, metadata: [UInt8]?, other: ArrowSchemaMetadata)? {
    guard schema.pointee.metadata != nil else { return nil }
    var m = ArrowSchemaMetadata.decode(schema.pointee.metadata)
    guard let nameBytes = m[ArrowSchemaMetadata.extensionNameKey] else { return nil }
    let md = m[ArrowSchemaMetadata.extensionMetadataKey]
    m[ArrowSchemaMetadata.extensionNameKey] = nil
    m[ArrowSchemaMetadata.extensionMetadataKey] = nil
    return (String(decoding: nameBytes, as: UTF8.self), md, m)
}

/// Imports an extension array: the storage array through the ordinary importer (with the extension keys
/// stripped so it does not recurse), plus the two strings.
func importExtensionArray(_ info: (name: String, metadata: [UInt8]?, other: ArrowSchemaMetadata),
                          schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowArray>,
                          context: MetalContext) throws -> ImportResult {
    // A shallow copy of the schema without metadata: the copy owns nothing, and the importer never
    // releases a schema, so the original stays intact and the child pointers are still valid.
    var bare = schema.pointee
    bare.metadata = nil
    bare.release = nil
    bare.private_data = nil
    let r = try withUnsafePointer(to: &bare) { try importArrowArray(schema: $0, array: array, context: context) }
    let ext = MetalExtensionArray(storage: r.array, name: info.name, metadata: info.metadata,
                                  otherMetadata: info.other)
    return ImportResult(array: .extended(ext), zeroCopy: r.zeroCopy)
}

// MARK: - Export

/// Owns the inner (storage) schema and the metadata blob of an exported extension schema. The outer
/// struct points into the inner one for `format`, `children` and `dictionary`, so the inner schema is
/// only released when the outer one is.
private final class ExtensionSchemaHolder {
    let inner: UnsafeMutablePointer<ArrowSchema>
    let nameC: UnsafeMutablePointer<CChar>
    let metadataC: UnsafeMutablePointer<CChar>
    init(name: String, metadata: [UInt8]) {
        inner = .allocate(capacity: 1)
        inner.initialize(to: ArrowSchema())
        nameC = strdup(name)!
        metadataC = .allocate(capacity: Swift.max(metadata.count, 1))
        for (i, b) in metadata.enumerated() { metadataC[i] = CChar(bitPattern: b) }
    }
    deinit {
        if let r = inner.pointee.release { r(inner) }
        inner.deallocate()
        free(nameC)
        metadataC.deallocate()
    }
}

private func releaseExtensionSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<ExtensionSchemaHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

extension MetalExtensionArray {
    /// The array is the storage array; nothing about the buffers is extension-specific.
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) { storage.exportArrowArray(into: out) }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        storage.exportArrowDeviceArray(into: out)
    }

    /// Writes the storage schema with the extension metadata attached, so a consumer that knows the type
    /// (pyarrow with the extension registered) reconstructs it and one that does not sees the storage.
    public func exportArrowSchema(name fieldName: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        let h = ExtensionSchemaHolder(name: fieldName, metadata: exportMetadata().encoded())
        storage.exportArrowSchema(name: fieldName, into: h.inner)
        out.pointee.format = h.inner.pointee.format
        out.pointee.name = UnsafePointer(h.nameC)
        out.pointee.metadata = UnsafePointer(h.metadataC)
        out.pointee.flags = h.inner.pointee.flags
        out.pointee.n_children = h.inner.pointee.n_children
        out.pointee.children = h.inner.pointee.children
        out.pointee.dictionary = h.inner.pointee.dictionary
        out.pointee.release = releaseExtensionSchema
        out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
    }
}
