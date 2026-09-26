import Foundation
import Metal

// A Parquet file, mapped once and read from the GPU.
//
// The whole file is `mmap`ed read-only and shared, and because `mmap` always hands back a page-aligned
// address, any range of it can be wrapped as an `MTLBuffer` with `makeBuffer(bytesNoCopy:)` — the same
// trick `MetalArrowBuffer` uses for zero-copy Arrow import. A column chunk's pages are then addressable by a compute kernel as
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
    /// A read-only `MAP_SHARED` mapping rather than a writable `MAP_PRIVATE` one.
    let isSharedReadOnly: Bool

    /// `sharedReadOnly` maps the file `PROT_READ` / `MAP_SHARED`; otherwise the mapping is
    /// `PROT_READ | PROT_WRITE` / `MAP_PRIVATE`, whose pages are copy-on-write (nothing is ever written
    /// back to the file).
    ///
    /// The difference shows up the first time the GPU reads a wrapped range: Metal makes the whole range
    /// resident then, and on an M4 Max (macOS 26) that took about 75 ms per gigabyte of a private
    /// mapping against about 7 ms per gigabyte of a shared read-only one. The decoders only ever read
    /// the page bytes (every kernel binds them `device const`), so the Parquet reader maps shared and
    /// read-only, and keeps the private mapping as the fallback for a device that will not wrap it.
    init(fd: Int32, fileOffset: Int, length: Int, sharedReadOnly: Bool = false) throws {
        precondition(fileOffset % metalPageSize() == 0)
        guard length > 0 else { throw ParquetError.io("empty mapping") }
        let p = sharedReadOnly
            ? mmap(nil, length, PROT_READ, MAP_SHARED, fd, off_t(fileOffset))
            : mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off_t(fileOffset))
        guard let p, p != MAP_FAILED else { throw ParquetError.io("mmap failed: \(String(cString: strerror(errno)))") }
        self.base = p
        self.length = length
        self.fileOffset = fileOffset
        self.isSharedReadOnly = sharedReadOnly
    }

    deinit { munmap(base, length) }

    var raw: UnsafeRawBufferPointer { UnsafeRawBufferPointer(start: base, count: length) }
}

/// Several page-aligned ranges of a file mapped next to each other in one reserved range of address
/// space, so that one `MTLBuffer` covers exactly them: one column's chunks, without the other columns'
/// chunks that sit between them in the file.
final class MappedView: @unchecked Sendable {
    let base: UnsafeMutableRawPointer
    let length: Int

    /// `intervals` are page-aligned, sorted and disjoint file ranges, mapped read-only and shared.
    init(fd: Int32, intervals: [Range<Int>]) throws {
        let length = intervals.reduce(0) { $0 + $1.count }
        guard length > 0 else { throw ParquetError.io("empty view") }
        guard let p = mmap(nil, length, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0), p != MAP_FAILED else {
            throw ParquetError.io("mmap failed: \(String(cString: strerror(errno)))")
        }
        var at = 0
        for r in intervals {
            let q = mmap(p.advanced(by: at), r.count, PROT_READ, MAP_SHARED | MAP_FIXED, fd, off_t(r.lowerBound))
            guard q == p.advanced(by: at) else {
                munmap(p, length)
                throw ParquetError.io("mmap failed: \(String(cString: strerror(errno)))")
            }
            at += r.count
        }
        self.base = p
        self.length = length
    }

    deinit { munmap(base, length) }
}

/// Where a column's page bytes are for the GPU: a buffer, the offset kernels bind it at, and how a file
/// offset translates into an offset from that binding point.
struct ParquetPageSource {
    let buffer: MetalArrowBuffer
    /// Bind `buffer.mtl` at this offset.
    let bindingOffset: Int
    /// Sorted, disjoint file ranges and where each starts relative to the binding point.
    let intervals: [(file: Range<Int>, at: Int)]
    /// Bytes addressable from the binding point.
    let length: Int

    /// The offset of file byte `x` (which must lie inside one of the intervals) from the binding point.
    func offset(ofFile x: Int) -> Int {
        var lo = 0, hi = intervals.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if intervals[mid].file.lowerBound <= x { lo = mid } else { hi = mid - 1 }
        }
        return intervals[lo].at + (x - intervals[lo].file.lowerBound)
    }
}

/// An open Parquet file: mapped bytes plus the decoded footer.
public final class ParquetFile: @unchecked Sendable {
    public let path: String
    public let metadata: ParquetFileMetadata
    /// The flattened leaf columns, in the order the column chunks appear in each row group.
    public let leaves: [ParquetLeaf]
    /// The Arrow-level fields this file exposes: leaves, lists, maps and structs, at any depth.
    public let fields: [ParquetField]
    public let context: MetalContext
    /// The decoded `ARROW:schema` key/value metadata, when present and well formed (`ParquetArrowSchema.swift`).
    let cachedArrowSchema: [ParquetArrowField]?

    let fd: Int32
    let fileSize: Int
    /// The file, mapped read-only and shared (see `MappedRegion`); the host parses headers from it and
    /// the GPU reads its pages through the wraps below.
    private let region: MappedRegion
    /// A private writable mapping of the same file, made only if Metal refuses to wrap `region`.
    private var privateRegion: MappedRegion?
    /// Map Parquet files read-only and shared. Off, every file is mapped private and writable, as before
    /// (the tests read the fixtures both ways).
    static var mapsSharedReadOnly = true
    /// Metal wrappers over sub-ranges of the mapping, one per column chunk actually read.
    ///
    /// Wrapping bytes with `makeBuffer(bytesNoCopy:)` costs next to nothing by itself, but the first
    /// kernel that reads a wrapped range makes all of it resident for the GPU, and that costs time
    /// proportional to the range (see `MappedRegion`). Wrapping the whole file up front would charge
    /// every projection for the columns it does not read, so ranges are wrapped on demand and cached:
    /// reading two columns of a forty-column file maps two column chunks.
    private var wrapped: [Int: MetalArrowBuffer] = [:]
    /// Views of one column's chunks (`pageSource(covering:)`), keyed by their file intervals, oldest first.
    private var views: [(key: [Int], source: ParquetPageSource)] = []
    private static let maxViews = 64
    private let wrapLock = NSLock()

    /// Use the column index and offset index, when the file has them, to skip the data pages a
    /// statistics filter rules out (`ParquetPageIndex.swift`). On by default; turning it off gives the
    /// row-group-granular read the same filters produce without the index.
    public var usePageIndex: Bool {
        get { statsLock.lock(); defer { statsLock.unlock() }; return _usePageIndex }
        set { statsLock.lock(); _usePageIndex = newValue; statsLock.unlock() }
    }
    /// What the most recent `read` on this handle did with the statistics and the page index.
    public var lastReadStatistics: ParquetReadStatistics {
        statsLock.lock(); defer { statsLock.unlock() }; return _lastReadStatistics
    }
    /// Use the column chunks' bloom filters, when the file has them, to drop the row groups an equality
    /// filter's value is certainly absent from (`ParquetBloomFilter.swift`). On by default.
    public var useBloomFilters: Bool {
        get { statsLock.lock(); defer { statsLock.unlock() }; return _useBloomFilters }
        set { statsLock.lock(); _useBloomFilters = newValue; statsLock.unlock() }
    }
    private var _useBloomFilters = true
    private var _usePageIndex = true
    private var _lastReadStatistics = ParquetReadStatistics()
    private let statsLock = NSLock()
    func recordReadStatistics(_ s: ParquetReadStatistics) {
        statsLock.lock(); _lastReadStatistics = s; statsLock.unlock()
    }

    public var numRows: Int64 { metadata.numRows }
    public var rowGroupCount: Int { metadata.rowGroups.count }

    public init(path: String, context: MetalContext = .shared) throws {
        self.path = path
        self.context = context
        ParquetProfile.start()
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
            // Virtualised GPUs (CI runners) keep the private mapping, which is what they were tested with.
            self.region = try MappedRegion(fd: fd, fileOffset: 0, length: roundUp(size, to: page),
                                           sharedReadOnly: ParquetFile.mapsSharedReadOnly && !context.isVirtualDevice)
        } catch { close(fd); throw error }
        ParquetProfile.lap("open.mmap")
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
        ParquetProfile.lap("open.footer")
        do {
            let built = try ParquetFile.buildSchema(metadata.schema)
            self.leaves = built.leaves
            self.fields = built.fields
        } catch { close(fd); throw error }
        self.cachedArrowSchema = ParquetFile.decodeArrowSchema(metadata.keyValueMetadata)
        ParquetProfile.lap("open.schema")
        ParquetProfile.report("open")
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
        let buf = try wrap(start: start, length: end - start)
        ParquetProfile.lap("wrap \((end - start) >> 20) MB")
        wrapLock.lock()
        wrapped[start] = buf
        wrapLock.unlock()
        return (buf, range.lowerBound - start)
    }

    /// True when the file is mapped read-only and shared (see `MappedRegion`).
    var isMappedShared: Bool { region.isSharedReadOnly }

    /// Number of column views held (`pageSource(covering:)`); the tests use it to see which path ran.
    var viewCount: Int { wrapLock.lock(); defer { wrapLock.unlock() }; return views.count }

    /// True when a cached wrap already encloses `range`.
    func hasWrap(enclosing range: Range<Int>) -> Bool {
        let page = metalPageSize()
        let start = (Swift.max(range.lowerBound, 0) / page) * page
        let end = Swift.min(roundUp(Swift.max(range.upperBound, start + 1), to: page), region.length)
        wrapLock.lock(); defer { wrapLock.unlock() }
        return wrapped.contains { $0.key <= start && $0.key + $0.value.byteCount >= end }
    }

    /// The page bytes of `ranges` (file byte ranges: one column's pages across the row groups a read
    /// selects) for the GPU.
    ///
    /// Column chunks interleave by row group, so one column's chunks are spread over nearly the whole
    /// file even though they are a fraction of its bytes, and the first kernel to read a wrapped range
    /// pays to make all of it resident. So when the ranges are sparse in their span, they are mapped
    /// side by side into a `MappedView` and only they are wrapped; when they are dense, when a cached
    /// wrap already covers them, or when the file is not mapped shared, the span is wrapped as one range
    /// (`buffer(covering:)`).
    func pageSource(covering ranges: [Range<Int>]) throws -> ParquetPageSource {
        let page = metalPageSize()
        var merged: [Range<Int>] = []
        for r in ranges.filter({ !$0.isEmpty }).sorted(by: { $0.lowerBound < $1.lowerBound }) {
            let lo = (Swift.max(r.lowerBound, 0) / page) * page
            let hi = Swift.min(roundUp(r.upperBound, to: page), region.length)
            if let last = merged.last, lo <= last.upperBound {
                merged[merged.count - 1] = last.lowerBound..<Swift.max(last.upperBound, hi)
            } else if lo < hi {
                merged.append(lo..<hi)
            }
        }
        guard let first = merged.first, let last = merged.last else {
            // Nothing but empty pages: any valid range of the mapping will do as the binding point.
            let (b, off) = try buffer(covering: 0..<Swift.min(page, region.length))
            return ParquetPageSource(buffer: b, bindingOffset: b.offset + off, intervals: [(0..<0, 0)], length: 0)
        }
        let span = first.lowerBound..<last.upperBound
        let total = merged.reduce(0) { $0 + $1.count }
        func whole() throws -> ParquetPageSource {
            guard span.count < Int(UInt32.max) else {
                throw ParquetError.unsupported("a single column chunk spanning more than 4 GiB")
            }
            let (b, off) = try buffer(covering: span)
            return ParquetPageSource(buffer: b, bindingOffset: b.offset + off,
                                     intervals: [(span, 0)], length: span.count)
        }
        // Pages are addressed with 32-bit offsets, so a span of 4 GiB or more always takes a view.
        let fits = span.count < Int(UInt32.max)
        if !isMappedShared || (fits && (merged.count == 1 || total * 5 >= span.count * 4 || hasWrap(enclosing: span))) {
            return try whole()
        }
        guard total < Int(UInt32.max) else {
            throw ParquetError.unsupported("a single column's chunks totalling more than 4 GiB")
        }
        let key = merged.flatMap { [$0.lowerBound, $0.upperBound] }
        wrapLock.lock()
        if let i = views.firstIndex(where: { $0.key == key }) {
            let hit = views.remove(at: i)
            views.append(hit)
            wrapLock.unlock()
            return hit.source
        }
        wrapLock.unlock()
        let view = try MappedView(fd: fd, intervals: merged)
        guard let mtl = context.device.makeBuffer(bytesNoCopy: view.base, length: view.length,
                                                  options: [.storageModeShared], deallocator: nil) else {
            return try whole()
        }
        ParquetProfile.lap("view \(view.length >> 20) MB in \(merged.count) ranges")
        var intervals: [(file: Range<Int>, at: Int)] = []
        var at = 0
        for r in merged { intervals.append((r, at)); at += r.count }
        let source = ParquetPageSource(buffer: MetalArrowBuffer(mtl: mtl, byteCount: view.length, keepAlive: view),
                                       bindingOffset: 0, intervals: intervals, length: view.length)
        wrapLock.lock()
        views.append((key, source))
        if views.count > Self.maxViews { views.removeFirst() }
        wrapLock.unlock()
        return source
    }

    /// Wraps `length` bytes of the mapping at `start` (both page multiples, inside the mapping).
    ///
    /// The mapping is this file's own, so it is known to be mapped: no `mincore` probe, which over a
    /// 2 GB range cost about 35 ms -- all of what used to be measured as the cost of wrapping.
    private func wrap(start: Int, length: Int) throws -> MetalArrowBuffer {
        if region.isSharedReadOnly,
           let b = context.device.makeBuffer(bytesNoCopy: region.base.advanced(by: start), length: length,
                                             options: [.storageModeShared], deallocator: nil) {
            return MetalArrowBuffer(mtl: b, byteCount: length, keepAlive: region)
        }
        // A device that will not wrap read-only memory gets the private, writable mapping.
        let r: MappedRegion
        if region.isSharedReadOnly {
            wrapLock.lock(); defer { wrapLock.unlock() }
            if let p = privateRegion { r = p } else {
                r = try MappedRegion(fd: fd, fileOffset: 0, length: region.length)
                privateRegion = r
            }
        } else {
            r = region
        }
        if let b = context.device.makeBuffer(bytesNoCopy: r.base.advanced(by: start), length: length,
                                             options: [.storageModeShared], deallocator: nil) {
            return MetalArrowBuffer(mtl: b, byteCount: length, keepAlive: r)
        }
        return try MetalArrowBuffer.copy(from: r.base.advanced(by: start), byteCount: length, context: context)
    }

    deinit { close(fd) }

    /// The file bytes, for header parsing only.
    var bytes: UnsafeRawBufferPointer { region.raw }

    /// Names of the Arrow-level fields.
    public var columnNames: [String] { fields.map { $0.name } }

    // MARK: - Schema flattening

    /// Walks the flat `SchemaElement` list into a tree and records, for each leaf column, its
    /// definition and repetition levels — the two numbers Dremel assembly needs — and, for every
    /// Arrow-level field, the three levels its own assembly needs (see `ParquetField`).
    static func buildSchema(_ elements: [ParquetSchemaElement]) throws -> (leaves: [ParquetLeaf], fields: [ParquetField]) {
        guard !elements.isEmpty else { throw ParquetError.malformed("empty schema") }
        var index = 1                    // element 0 is the root message
        var leaves: [ParquetLeaf] = []
        var fields: [ParquetField] = []

        /// Recursive descent; returns the field for the subtree rooted at `elements[i]`.
        ///
        /// `def` and `rep` are the parent's definition and repetition levels, and `slotDef` the
        /// definition level of the nearest repeated ancestor (0 when there is none). A `repeated` node
        /// comes back as the *element* of the list it implies: its levels describe one element, and the
        /// caller either uses it as the element of a LIST / MAP wrapper or wraps it in a list itself.
        ///
        /// `num_children` comes straight out of the footer, so it can claim more children than the
        /// schema list holds. Every index is checked here rather than trusting it: an unchecked
        /// `elements[i]` on a crafted file is a bounds trap, which takes the process down.
        func node(_ i: Int, def: Int, rep: Int, slotDef: Int, path: [String]) throws -> ParquetField {
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
            // Every entry of a repeated node's subtree whose definition level reaches `d` is one
            // element of it; below a repeated node that is the slot condition for everything.
            let mySlotDef = e.repetition == .repeated ? d : slotDef
            let myPath = path + [e.name]
            func levelled(_ kind: ParquetField.Kind, nullable: Bool) -> ParquetField {
                var f = ParquetField(name: e.name, kind: kind, nullable: nullable)
                f.definitionLevel = d
                f.repetitionLevel = rp
                f.slotDefinitionLevel = mySlotDef
                f.isRepeated = e.repetition == .repeated
                f.fieldID = e.fieldID
                return f
            }
            if e.numChildren == 0 {
                guard let t = e.type else { throw ParquetError.malformed("leaf \(e.name) has no physical type") }
                // `type_length` is the value width for FIXED_LEN_BYTE_ARRAY and multiplies every
                // buffer size downstream, so a negative or absurd one has to stop here rather than
                // overflow an allocation. Real ones are tiny: 16 for UUID, at most 32 for a decimal.
                if t == .fixedLenByteArray {
                    guard e.typeLength > 0, e.typeLength <= (1 << 20) else {
                        throw ParquetError.malformed(
                            "FIXED_LEN_BYTE_ARRAY column \(e.name) declares a length of \(e.typeLength)")
                    }
                }
                let leaf = ParquetLeaf(index: leaves.count, element: e, physical: t, path: myPath,
                                       maxDefinition: d, maxRepetition: rp)
                leaves.append(leaf)
                return levelled(.leaf(leaf), nullable: e.repetition == .optional)
            }
            var children: [ParquetField] = []
            var childIndices: [Int] = []
            // `num_children` is a signed thrift i32: a negative one would make `0..<n` a range with a
            // reversed bound, which is a trap, not an error.
            for _ in 0..<Swift.max(e.numChildren, 0) {
                let c = index
                index += 1
                children.append(try node(c, def: d, rep: rp, slotDef: mySlotDef, path: myPath))
                childIndices.append(c)
            }
            let nullable = e.repetition == .optional
            let isMap = e.logicalType == .map || e.convertedType == .map || e.convertedType == .mapKeyValue
            let isList = e.logicalType == .list || e.convertedType == .list
            // A list or map wrapper: exactly one child, and that child repeated.
            if isList || isMap, children.count == 1, children[0].isRepeated {
                let repeated = children[0]
                let repeatedName = elements[childIndices[0]].name
                if isMap, case .group(let kv) = repeated.kind, kv.count == 2 {
                    // MAP: group (MAP) { repeated group key_value { key; value } }. The repeated group
                    // is the entries struct itself, never null.
                    var entries = repeated
                    entries.isRepeated = false
                    var f = levelled(.list(element: entries, repeatedDefinition: repeated.definitionLevel),
                                     nullable: nullable)
                    f.isMap = true
                    return f
                }
                // LIST (or a MAP whose entries are not a key/value pair, which Arrow reads as a list).
                // The backward-compatibility rules of the format decide whether the repeated node is the
                // element (two-level) or only wraps it (three-level): a repeated primitive, a repeated
                // group of several fields, and a repeated group named `array` or `<list>_tuple` are the
                // element; otherwise its one child is.
                var element = repeated
                element.isRepeated = false
                if case .group(let inner) = repeated.kind, inner.count == 1,
                   repeatedName != "array", repeatedName != e.name + "_tuple" {
                    element = inner[0]
                }
                return levelled(.list(element: element, repeatedDefinition: repeated.definitionLevel),
                                nullable: nullable)
            }
            // A repeated child that no LIST / MAP annotation claims is a list of required elements.
            let wrapped = children.map { c -> ParquetField in
                c.isRepeated ? ParquetFile.implicitList(c, parentDef: d, parentRep: rp, parentSlotDef: mySlotDef) : c
            }
            return levelled(.group(wrapped), nullable: nullable)
        }

        let root = elements[0]
        for _ in 0..<Swift.max(root.numChildren, 0) {
            let c = index
            index += 1
            let f = try node(c, def: 0, rep: 0, slotDef: 0, path: [])
            fields.append(f.isRepeated ? implicitList(f, parentDef: 0, parentRep: 0, parentSlotDef: 0) : f)
        }
        return (leaves, fields)
    }

    /// The list a bare `repeated` node implies: `repeated int32 x` reads as a non-null
    /// `list<x: int32 not null>`, as Arrow's own reader has it.
    static func implicitList(_ element: ParquetField, parentDef: Int, parentRep: Int, parentSlotDef: Int) -> ParquetField {
        var e = element
        e.isRepeated = false
        var f = ParquetField(name: element.name, kind: .list(element: e, repeatedDefinition: e.definitionLevel),
                             nullable: false)
        f.definitionLevel = parentDef
        f.repetitionLevel = parentRep
        f.slotDefinitionLevel = parentSlotDef
        f.fieldID = element.fieldID
        return f
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

/// An Arrow-level field: a leaf column, a list (or map) over a field, or a struct of fields.
///
/// Every field carries the three levels Dremel assembly needs to find *its* values among the level
/// entries of any leaf beneath it (the same three Arrow's C++ reader keeps in its `LevelInfo`):
///
/// - an entry is a **slot** of this field — one element of the Arrow array being built — when its
///   repetition level is at most `repetitionLevel` and its definition level reaches
///   `slotDefinitionLevel`;
/// - the slot is **non-null** when the definition level reaches `definitionLevel`.
///
/// A top-level field has `repetitionLevel == 0` and `slotDefinitionLevel == 0`, so its slots are the
/// entries that start a row.
public struct ParquetField: Sendable {
    public indirect enum Kind: Sendable {
        case leaf(ParquetLeaf)
        /// A list, or a map when `isMap` is set (the element is then the key/value entries struct).
        case list(element: ParquetField, repeatedDefinition: Int)
        case group([ParquetField])
    }
    public let name: String
    public let kind: Kind
    public let nullable: Bool
    /// Definition level at which this field is present (non-null).
    public internal(set) var definitionLevel: Int = 0
    /// The largest repetition level of an entry that starts a new slot of this field.
    public internal(set) var repetitionLevel: Int = 0
    /// The definition level of the nearest repeated ancestor: an entry below it is a slot of this field
    /// only when that ancestor has an element there.
    public internal(set) var slotDefinitionLevel: Int = 0
    /// A `MAP` column: a list of key/value entries.
    public internal(set) var isMap = false
    /// The Parquet `field_id`, when the writer recorded one.
    public internal(set) var fieldID: Int32? = nil
    /// Set while the schema is being built: this node is `repeated` and stands for a list element.
    var isRepeated = false

    /// Every leaf beneath this field.
    public var leaves: [ParquetLeaf] {
        switch kind {
        case .leaf(let l): return [l]
        case .list(let e, _): return e.leaves
        case .group(let cs): return cs.flatMap { $0.leaves }
        }
    }
}
