import Foundation
import Metal

// A Parquet file, mapped once and read from the GPU.
//
// The whole file is `mmap`ed, and because `mmap` always hands back a page-aligned address, any range of
// it can be wrapped as an `MTLBuffer` with `makeBuffer(bytesNoCopy:)` — the same trick `MetalArrowBuffer`
// uses for zero-copy Arrow import. A column chunk's pages are then addressable by a compute kernel as
// byte offsets into that buffer, and the CPU never reads a byte of column data: it only parses the
// Thrift footer and the page headers, which are metadata.
//
// Ranges are wrapped lazily, one per column chunk read (see `buffer(covering:)`), and pages are faulted
// in by the kernel that first touches them, so a projection over two of forty columns brings neither the
// other thirty-eight columns' bytes into memory nor their pages into the GPU's page tables.

/// One `mmap`ed range of a file, wrapped as an `MTLBuffer`.
final class MappedRegion: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int
    /// File offset of `base` (always a page multiple).
    let fileOffset: Int

    init(fd: Int32, fileOffset: Int, length: Int) throws {
        precondition(fileOffset % metalPageSize() == 0)
        guard length > 0 else { throw ParquetError.io("empty mapping") }
        // PROT_WRITE with MAP_PRIVATE gives copy-on-write pages: nothing is written back to the file, and
        // Metal is happy to wrap writable memory (a read-only mapping is rejected by some drivers).
        let p = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off_t(fileOffset))
        guard let p, p != MAP_FAILED else { throw ParquetError.io("mmap failed: \(String(cString: strerror(errno)))") }
        self.base = p
        self.length = length
        self.fileOffset = fileOffset
    }

    deinit { munmap(base, length) }

    var raw: UnsafeRawBufferPointer { UnsafeRawBufferPointer(start: base, count: length) }
}

/// An open Parquet file: mapped bytes plus the decoded footer.
public final class ParquetFile: @unchecked Sendable {
    public let path: String
    public let metadata: ParquetFileMetadata
    /// The flattened leaf columns, in the order the column chunks appear in each row group.
    public let leaves: [ParquetLeaf]
    /// The Arrow-level fields this file exposes (a leaf, or a `list<...>` built from one).
    public let fields: [ParquetField]
    public let context: MetalContext

    let fd: Int32
    let fileSize: Int
    private let region: MappedRegion
    /// Metal wrappers over sub-ranges of the mapping, one per column chunk actually read.
    ///
    /// Wrapping bytes with `makeBuffer(bytesNoCopy:)` makes the GPU's page tables cover them, and that
    /// costs time proportional to the range -- roughly 20 ms per gigabyte on an M4 Max. Wrapping the
    /// whole file up front would charge every projection for the columns it does not read, so ranges are
    /// wrapped on demand and cached: reading two columns of a forty-column file maps two column chunks.
    private var wrapped: [Int: MetalArrowBuffer] = [:]
    private let wrapLock = NSLock()

    public var numRows: Int64 { metadata.numRows }
    public var rowGroupCount: Int { metadata.rowGroups.count }

    public init(path: String, context: MetalContext = .shared) throws {
        self.path = path
        self.context = context
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw ParquetError.io("cannot open \(path): \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw ParquetError.io("cannot stat \(path)") }
        let size = Int(st.st_size)
        guard size >= 12 else { close(fd); throw ParquetError.notParquet }
        self.fd = fd
        self.fileSize = size
        do {
            let page = metalPageSize()
            self.region = try MappedRegion(fd: fd, fileOffset: 0, length: roundUp(size, to: page))
        } catch { close(fd); throw error }
        let raw = region.raw
        // PAR1 ... <metadata> <4-byte metadata length> PAR1
        guard raw[0] == 0x50, raw[1] == 0x41, raw[2] == 0x52, raw[3] == 0x31,
              raw[size - 4] == 0x50, raw[size - 3] == 0x41, raw[size - 2] == 0x52, raw[size - 1] == 0x31 else {
            close(fd)
            throw ParquetError.notParquet
        }
        let metaLen = Int(raw.loadUnaligned(fromByteOffset: size - 8, as: UInt32.self))
        guard metaLen > 0, size - 8 - metaLen >= 4 else { close(fd); throw ParquetError.malformed("footer length \(metaLen)") }
        var r = ThriftReader(raw, at: size - 8 - metaLen)
        do {
            self.metadata = try ParquetFileMetadata.read(&r)
        } catch { close(fd); throw error }
        do {
            let built = try ParquetFile.buildSchema(metadata.schema)
            self.leaves = built.leaves
            self.fields = built.fields
        } catch { close(fd); throw error }
    }

    /// A Metal buffer covering `range` of the file, plus the offset of `range.lowerBound` inside it.
    /// Ranges are rounded out to page boundaries (what `makeBuffer(bytesNoCopy:)` requires) and cached.
    func buffer(covering range: Range<Int>) throws -> (buffer: MetalArrowBuffer, offset: Int) {
        let page = metalPageSize()
        let start = (Swift.max(range.lowerBound, 0) / page) * page
        let end = Swift.min(roundUp(Swift.max(range.upperBound, start + 1), to: page), region.length)
        wrapLock.lock()
        // Any cached wrap that already encloses this range serves it, so overlapping column chunks and
        // repeated reads of the same column never wrap the same bytes twice.
        for (s, b) in wrapped where s <= start && s + b.byteCount >= end {
            wrapLock.unlock()
            return (b, range.lowerBound - s)
        }
        wrapLock.unlock()
        let (buf, _) = try MetalArrowBuffer.wrapOrCopy(UnsafeRawPointer(region.base).advanced(by: start),
                                                       byteCount: end - start,
                                                       keepAlive: region, context: context)
        wrapLock.lock()
        wrapped[start] = buf
        wrapLock.unlock()
        return (buf, range.lowerBound - start)
    }

    deinit { close(fd) }

    /// The file bytes, for header parsing only.
    var bytes: UnsafeRawBufferPointer { region.raw }

    /// Names of the Arrow-level fields.
    public var columnNames: [String] { fields.map { $0.name } }

    // MARK: - Schema flattening

    /// Walks the flat `SchemaElement` list into a tree and records, for each leaf column, its
    /// definition and repetition levels — the two numbers Dremel assembly needs.
    static func buildSchema(_ elements: [ParquetSchemaElement]) throws -> (leaves: [ParquetLeaf], fields: [ParquetField]) {
        guard !elements.isEmpty else { throw ParquetError.malformed("empty schema") }
        var index = 1                    // element 0 is the root message
        var leaves: [ParquetLeaf] = []
        var fields: [ParquetField] = []

        /// Recursive descent; returns the field for the subtree rooted at `elements[i]`.
        ///
        /// `num_children` comes straight out of the footer, so it can claim more children than the
        /// schema list holds. Every index is checked here rather than trusting it: an unchecked
        /// `elements[i]` on a crafted file is a bounds trap, which takes the process down.
        func node(_ i: Int, def: Int, rep: Int, path: [String]) throws -> ParquetField {
            guard i >= 0, i < elements.count else {
                throw ParquetError.malformed("schema element \(i) is outside 0..<\(elements.count)")
            }
            let e = elements[i]
            var d = def, rp = rep
            switch e.repetition {
            case .optional: d += 1
            case .repeated: d += 1; rp += 1
            case .required: break
            }
            let myPath = path + [e.name]
            if e.numChildren == 0 {
                guard let t = e.type else { throw ParquetError.malformed("leaf \(e.name) has no physical type") }
                let leaf = ParquetLeaf(index: leaves.count, element: e, physical: t, path: myPath,
                                       maxDefinition: d, maxRepetition: rp)
                leaves.append(leaf)
                return ParquetField(name: e.name, kind: .leaf(leaf), nullable: e.repetition == .optional)
            }
            var children: [ParquetField] = []
            // `num_children` is a signed thrift i32: a negative one would make `0..<n` a range with a
            // reversed bound, which is a trap, not an error.
            for _ in 0..<Swift.max(e.numChildren, 0) {
                let c = index
                index += 1
                children.append(try node(c, def: d, rep: rp, path: myPath))
            }
            // LIST annotation: group { repeated group list { <element> } }
            let isList = e.logicalType == .list || e.convertedType == .list
            if isList, children.count == 1, case .group(let inner) = children[0].kind, inner.count == 1 {
                let repeatedIdx = i + 1     // the `list` repeated group
                let repeated = elements[repeatedIdx]
                var listDef = d
                if repeated.repetition == .repeated { listDef += 1 }
                return ParquetField(name: e.name,
                                    kind: .list(element: inner[0], repeatedDefinition: listDef),
                                    nullable: e.repetition == .optional)
            }
            // A two-level list: group (LIST) { repeated <element> }
            if isList, children.count == 1, elements[i + 1].repetition == .repeated {
                return ParquetField(name: e.name, kind: .list(element: children[0], repeatedDefinition: d + 1),
                                    nullable: e.repetition == .optional)
            }
            return ParquetField(name: e.name, kind: .group(children), nullable: e.repetition == .optional)
        }

        let root = elements[0]
        for _ in 0..<Swift.max(root.numChildren, 0) {
            let c = index
            index += 1
            fields.append(try node(c, def: 0, rep: 0, path: []))
        }
        return (leaves, fields)
    }

    /// The leaf column with this dotted path, or nil.
    public func leaf(named name: String) -> ParquetLeaf? {
        leaves.first { $0.path.joined(separator: ".") == name || $0.path.last == name }
    }
}

/// One leaf (physical) column of the schema.
public struct ParquetLeaf: Sendable {
    /// Position of the column chunk inside each row group.
    public let index: Int
    public let element: ParquetSchemaElement
    public let physical: ParquetPhysicalType
    public let path: [String]
    public let maxDefinition: Int
    public let maxRepetition: Int

    public var name: String { path.last ?? "" }
    public var dottedPath: String { path.joined(separator: ".") }
    public var isNullable: Bool { maxDefinition > 0 }
    public var logicalType: ParquetLogicalType { element.logicalType }
    public var typeLength: Int { element.typeLength }
}

/// An Arrow-level field: either a leaf column or a list over one.
public struct ParquetField: Sendable {
    public indirect enum Kind: Sendable {
        case leaf(ParquetLeaf)
        case list(element: ParquetField, repeatedDefinition: Int)
        case group([ParquetField])
    }
    public let name: String
    public let kind: Kind
    public let nullable: Bool

    /// Every leaf beneath this field.
    public var leaves: [ParquetLeaf] {
        switch kind {
        case .leaf(let l): return [l]
        case .list(let e, _): return e.leaves
        case .group(let cs): return cs.flatMap { $0.leaves }
        }
    }
}
