import Foundation
import Metal
import CArrowABI

// Arrow nested types: list, large_list, fixed_size_list, struct, map and dense/sparse union.
//
// Every nested array holds its children as `AnyMetalArray`, so nesting is recursive and a child may be
// any type this package supports, another nested array included. Lists always carry int32 offsets in
// Metal shared memory (`large_list` offsets are narrowed on import, exactly as `large_utf8` offsets are,
// and a `fixed_size_list` materialises the offsets its layout implies), which is what lets one set of
// kernels serve all four list-shaped layouts.
//
// The compute surface is `list_value_length`, `list_flatten`, `list_element` and `struct_field`, plus
// `filter` / `take` / `slice` on every nested type. Maps and unions are import/export and selection only.

// MARK: - Shared helpers

enum NestedSupport {
    /// A bitmap with every one of `bits` bits set (the implicit validity of an array without a bitmap).
    static func onesBitmap(_ bits: Int, _ ctx: MetalContext) throws -> MetalArrowBuffer {
        let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: bits), 1), zeroed: false, context: ctx)
        memset(b.mutableContents, 0xFF, b.byteCount)
        return b
    }

    /// Gathers a validity bitmap through `idx`: the result bit is set when the index is valid *and* the
    /// source bit is set, which is the null rule `take` applies to values. Returns nil when neither side
    /// has nulls.
    static func gatherValidity(_ ctx: MetalContext, _ validity: MetalArrowBuffer?, length: Int,
                               idx: MetalArray<Int32>) throws -> MetalArrowBuffer? {
        let n = idx.length
        guard validity != nil || idx.validity != nil, n > 0 else { return nil }
        let bits = try validity ?? onesBitmap(Swift.max(length, 1), ctx)
        let src = MetalBooleanArray(length: length, nullCount: 0, validity: nil, values: bits, context: ctx)
        let gathered = try src.take(idx)
        return try BitmapOps.combineValidity(ctx, gathered.values, gathered.validity, bits: n)
    }

    /// Copies `length` bits starting at `offset` into a fresh bitmap (host loop; nested slices are metadata
    /// work, not kernel work).
    static func sliceBitmap(_ v: MetalArrowBuffer?, offset: Int, length: Int, _ ctx: MetalContext) throws -> MetalArrowBuffer? {
        guard let v, length > 0 else { return nil }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), context: ctx)
        let sp = v.typed(UInt8.self), dp = out.mutableTyped(UInt8.self)
        for i in 0..<length where Bitmap.isSet(sp, offset + i) { Bitmap.set(dp, i) }
        return out
    }

    /// Number of unset bits among the first `length` bits.
    static func nullCount(_ v: MetalArrowBuffer?, length: Int) -> Int {
        guard let v else { return 0 }
        return length - Bitmap.popcount(v.typed(UInt8.self), bits: length)
    }

    /// Builds a validity bitmap from `[Bool]`, or nil when every element is valid.
    static func bitmap(_ valid: [Bool], _ ctx: MetalContext) throws -> MetalArrowBuffer? {
        guard valid.contains(false) else { return nil }
        let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: valid.count), 1), context: ctx)
        let p = b.mutableTyped(UInt8.self)
        for (i, v) in valid.enumerated() where v { Bitmap.set(p, i) }
        return b
    }

    /// 0, 1, ... n - 1 as an int32 array, shifted by `from`.
    static func iota(_ from: Int, _ n: Int, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: ctx)
        let p = out.mutableTyped(Int32.self)
        for i in 0..<n { p[i] = Int32(from + i) }
        return MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: out, context: ctx)
    }
}

// MARK: - List

/// Which of Arrow's three list layouts an array came from.
public enum ArrowListKind: Hashable, Sendable {
    /// `list` ("+l") and `large_list` ("+L"); large offsets are narrowed to int32 on import.
    case variable
    /// `fixed_size_list` ("+w:N"): every row has exactly `N` child elements, so there is no offsets buffer.
    case fixedSize(Int)

    public var arrowFormat: String {
        switch self {
        case .variable: return "+l"
        case .fixedSize(let n): return "+w:\(n)"
        }
    }
    /// The row width of a fixed-size list, nil for a variable-length one.
    public var fixedWidth: Int? { if case .fixedSize(let n) = self { return n } else { return nil } }
}

/// An Arrow list array in Metal shared memory: a validity bitmap, int32 offsets (length + 1) and a child
/// array of any supported type.
///
/// A `fixed_size_list` keeps materialised offsets (`offsets[i] == i * N`) so that every kernel below is
/// layout-independent; the invariant is restored by every operation and checked on export.
public final class MetalListArray: @unchecked Sendable {
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    /// int32, `length + 1` entries. Not necessarily zero based: a list imported from a sliced producer
    /// points into the middle of its child, exactly as Arrow allows.
    public let offsets: MetalArrowBuffer
    /// The child ("values") array. Its length is at least `offsets[length]`.
    public let values: AnyMetalArray
    public let kind: ArrowListKind
    /// Name of the child field in the exported schema (Arrow's convention is "item").
    public let fieldName: String
    public let context: MetalContext

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, offsets: MetalArrowBuffer,
                values: AnyMetalArray, kind: ArrowListKind = .variable, fieldName: String = "item",
                context: MetalContext = .shared) {
        precondition(offsets.byteCount >= (length + 1) * 4)
        self.length = length; self.nullCount = nullCount; self.validity = validity
        self.offsets = offsets; self.values = values; self.kind = kind
        self.fieldName = fieldName; self.context = context
    }

    /// Builds a list array over consecutive slices of `values`: row i covers `counts[i]` child elements,
    /// and a nil count is a null row with no child elements (a fixed-size list keeps its `N` slots).
    public convenience init(counts: [Int?], values: AnyMetalArray, kind: ArrowListKind = .variable,
                            fieldName: String = "item", context: MetalContext = .shared) throws {
        let n = counts.count
        let off = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: context)
        let p = off.mutableTyped(Int32.self)
        var pos = 0
        for (i, c) in counts.enumerated() {
            p[i] = Int32(pos)
            pos += c ?? (kind.fixedWidth ?? 0)
        }
        p[n] = Int32(pos)
        guard pos <= values.length else {
            throw ArrowMetalError.invalidArrowArray("list counts cover \(pos) child elements but the child has \(values.length)")
        }
        if let w = kind.fixedWidth {
            for c in counts where c != nil && c! != w {
                throw ArrowMetalError.invalidArrowArray("fixed_size_list<\(w)> row length must be \(w)")
            }
        }
        let bm = try NestedSupport.bitmap(counts.map { $0 != nil }, context)
        self.init(length: n, nullCount: counts.filter { $0 == nil }.count, validity: bm, offsets: off,
                  values: values, kind: kind, fieldName: fieldName, context: context)
    }

    public var arrowFormat: String { kind.arrowFormat }
    public func isValid(_ i: Int) -> Bool { validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true }
    /// The child range row `i` covers, or nil when the row is null.
    public func valueRange(_ i: Int) -> Range<Int>? {
        guard isValid(i) else { return nil }
        let o = offsets.typed(Int32.self)
        return Int(o[i])..<Int(o[i + 1])
    }
    /// The child range the whole array references, `offsets[0] ..< offsets[length]`.
    public var childRange: Range<Int> {
        let o = offsets.typed(Int32.self)
        return Int(o[0])..<Int(o[length])
    }

    func setNullCount(_ n: Int) { nullCount = n }
    func recomputeNullCount() { nullCount = NestedSupport.nullCount(validity, length: length) }

    private func pso(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: NestedSource.source, function: fn, cacheKey: "nested/\(fn)")
    }

    // MARK: compute

    /// Arrow `list_value_length`: the number of child elements in each row, null where the row is null.
    public func listValueLength() throws -> MetalArray<Int32> {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1) * 4, zeroed: false, context: context)
        if length > 0 {
            // Eight rows per thread whenever the vector load is legal: `offsets.offset` is zero for a
            // whole array and a multiple of four bytes for a slice, so only a slice at a row that is
            // not a multiple of four falls back to the one-row-per-thread kernel. Both write the same
            // bytes; the wide one runs about twice as fast (1.13 ms to 0.55 ms at 10M rows).
            let wide = offsets.offset % 16 == 0 && out.offset % 16 == 0
            let p = try pso(wide ? "list_value_length8" : "list_value_length")
            try context.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                Dispatch.setLength(enc, length, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, p, count: wide ? (length + 7) / 8 : length)
            }
        }
        return MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: out, context: context)
    }

    /// Arrow `list_flatten`: the child array restricted to the range this list references,
    /// `offsets[0] ..< offsets[length]`.
    public func listFlatten() throws -> AnyMetalArray {
        let r = childRange
        if r.lowerBound == 0 && r.count == values.length { return values }
        return try values.slice(offset: r.lowerBound, length: r.count)
    }

    /// Arrow `list_element`: element `index` of every row, null where the row is null or shorter than
    /// `index + 1`. A negative index is null everywhere.
    public func listElement(_ index: Int) throws -> AnyMetalArray {
        let ctx = context
        let n = length
        let idxBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: true, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: true, context: ctx)
        if n > 0 {
            let p = try pso("list_element_index")
            let v = validity ?? offsets
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                var k = Int32(clamping: index)
                enc.setBytes(&k, length: 4, index: 4)
                enc.setBuffer(idxBuf.mtl, offset: idxBuf.offset, index: 5)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 6)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        let bits = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) : nil
        let idx = MetalArray<Int32>(length: n, nullCount: 0, validity: bits, values: idxBuf, context: ctx)
        idx.recomputeNullCount()
        return try values.take(idx)
    }

    // MARK: selection

    /// Arrow `take`. Offsets are recomputed with the GPU scan and the child is gathered by ranges: one
    /// kernel expands the selected rows' source ranges into a flat index array, which the child's own
    /// `take` then gathers, so the child may be of any type, nested types included.
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalListArray {
        let ctx = context
        let idx32: MetalArray<Int32> = try (indices as? MetalArray<Int32>) ?? indices.cast(to: Int32.self)
        let n = idx32.length
        try Dispatch.checkLength(Swift.max(n, length))
        // Row lengths of the result: the source row's length, or the fixed width for a fixed-size list,
        // so that `offsets[i] == i * N` survives a selection that contains nulls.
        let lens = try listValueLength().take(idx32)
        let fill = Int32(kind.fixedWidth ?? 0)
        let lensFilled: MetalArray<Int32> = lens.validity == nil ? lens : try lens.fillNull(fill)
        let outOffsets = try lensFilled.exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let srcFilled: MetalArray<Int32> = idx32.validity == nil ? idx32 : try idx32.fillNull(-1)
        let childIdxBuf = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1) * 4, zeroed: true, context: ctx)
        let childValidBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: true, context: ctx)
        if n > 0 {
            let p = try pso("list_expand_indices")
            try ctx.run { enc in
                enc.setComputePipelineState(p)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(srcFilled.values.mtl, offset: srcFilled.values.offset, index: 1)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBuffer(childIdxBuf.mtl, offset: childIdxBuf.offset, index: 4)
                enc.setBuffer(childValidBytes.mtl, offset: childValidBytes.offset, index: 5)
                Dispatch.dispatch1D(enc, p, count: n)
            }
        }
        var childBits: MetalArrowBuffer? = nil
        if idx32.validity != nil && total > 0 { childBits = try BitmapOps.packBits(ctx, bytes: childValidBytes, bits: total) }
        let childIdx = MetalArray<Int32>(length: total, nullCount: 0, validity: childBits, values: childIdxBuf, context: ctx)
        childIdx.recomputeNullCount()
        let newValues = try values.take(childIdx)
        return MetalListArray(length: n, nullCount: lens.nullCount, validity: lens.validity, offsets: outOffsets,
                              values: newValues, kind: kind, fieldName: fieldName, context: ctx)
    }

    /// Arrow `filter` (null mask entries drop the row, as everywhere else in this package).
    public func filter(_ mask: MetalBooleanArray) throws -> MetalListArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return try take(try MetalArray<Int32>.iota(length, context: context).filter(mask))
    }

    /// Zero-copy for a variable-length list — the offsets are a view into the same buffer, and the child is
    /// shared untouched. A fixed-size list re-gathers so that `offsets[i] == i * N` still holds.
    public func slice(offset: Int, length n: Int) throws -> MetalListArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("list slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        if kind.fixedWidth != nil { return try take(try NestedSupport.iota(offset, n, context)) }
        let view = MetalArrowBuffer(mtl: offsets.mtl, byteCount: (n + 1) * 4,
                                    offset: offsets.offset + offset * 4, keepAlive: offsets)
        let v = try NestedSupport.sliceBitmap(validity, offset: offset, length: n, context)
        let out = MetalListArray(length: n, nullCount: 0, validity: v, offsets: view, values: values,
                                 kind: kind, fieldName: fieldName, context: context)
        out.recomputeNullCount()
        return out
    }
}

// MARK: - Struct

/// An Arrow struct array ("+s") as a *column*: named children of equal length plus its own validity.
///
/// `MetalRecordBatch` is the top-level form of the same layout; this is the nested one, so a struct may
/// be a list's child, a map's entries, or a struct's own child.
public final class MetalStructArray: @unchecked Sendable {
    public let length: Int
    public internal(set) var nullCount: Int
    public let validity: MetalArrowBuffer?
    public let names: [String]
    public let children: [AnyMetalArray]
    public let context: MetalContext

    public init(length: Int, nullCount: Int, validity: MetalArrowBuffer?, names: [String],
                children: [AnyMetalArray], context: MetalContext = .shared) throws {
        guard names.count == children.count else {
            throw ArrowMetalError.invalidArrowArray("struct names/children count mismatch")
        }
        for c in children where c.length < length {
            throw ArrowMetalError.lengthMismatch(length, c.length)
        }
        self.length = length; self.nullCount = nullCount; self.validity = validity
        self.names = names; self.children = children; self.context = context
    }

    /// Builds a struct column from equal-length children, `valid` giving the struct's own null-ness.
    public convenience init(names: [String], children: [AnyMetalArray], valid: [Bool]? = nil,
                            context: MetalContext = .shared) throws {
        let n = children.first?.length ?? valid?.count ?? 0
        let bm = try valid.flatMap { try NestedSupport.bitmap($0, context) }
        try self.init(length: n, nullCount: valid?.filter { !$0 }.count ?? 0, validity: bm,
                      names: names, children: children, context: context)
    }

    public var arrowFormat: String { "+s" }
    public func isValid(_ i: Int) -> Bool { validity.map { Bitmap.isSet($0.typed(UInt8.self), i) } ?? true }
    func recomputeNullCount() { nullCount = NestedSupport.nullCount(validity, length: length) }

    /// Arrow `struct_field`: one child by name, with the struct's own nulls propagated into it — a field of
    /// a null row is null, which is what Arrow's `struct_field` returns. Use `children` for the raw child.
    public func structField(_ name: String) throws -> AnyMetalArray {
        guard let i = names.firstIndex(of: name) else {
            throw ArrowMetalError.invalidArrowArray("struct has no field named \(name)")
        }
        let child = children[i]
        guard let v = validity, nullCount > 0 else {
            return child.length == length ? child : try child.slice(offset: 0, length: length)
        }
        // A gather through a null-carrying iota is the one path that works for every child type,
        // nested children included.
        let base = try NestedSupport.iota(0, length, context)
        let idx = MetalArray<Int32>(length: length, nullCount: nullCount, validity: v, values: base.values,
                                    context: context)
        return try child.take(idx)
    }
    public subscript(name: String) -> AnyMetalArray? {
        names.firstIndex(of: name).map { children[$0] }
    }

    /// Arrow `take`, by delegating to every child and gathering the struct's own validity.
    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalStructArray {
        let idx32: MetalArray<Int32> = try (indices as? MetalArray<Int32>) ?? indices.cast(to: Int32.self)
        let v = try NestedSupport.gatherValidity(context, validity, length: length, idx: idx32)
        let out = try MetalStructArray(length: idx32.length, nullCount: 0, validity: v, names: names,
                                       children: try children.map { try $0.take(idx32) }, context: context)
        out.recomputeNullCount()
        return out
    }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalStructArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return try take(try MetalArray<Int32>.iota(length, context: context).filter(mask))
    }

    public func slice(offset: Int, length n: Int) throws -> MetalStructArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("struct slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        let v = try NestedSupport.sliceBitmap(validity, offset: offset, length: n, context)
        let out = try MetalStructArray(length: n, nullCount: 0, validity: v, names: names,
                                       children: try children.map { try $0.slice(offset: offset, length: n) },
                                       context: context)
        out.recomputeNullCount()
        return out
    }
}

// MARK: - Map

/// An Arrow map array ("+m"): a list of non-nullable `struct<key, value>` entries.
///
/// Import, export and `filter` / `take` / `slice` only — there is no `map_lookup` kernel.
public final class MetalMapArray: @unchecked Sendable {
    /// The underlying list; its child is the `entries` struct.
    public let entries: MetalListArray
    /// Arrow's `keys_sorted` flag, carried through import and export.
    public let keysSorted: Bool

    public init(entries: MetalListArray, keysSorted: Bool = false) throws {
        guard case .structure(let s) = entries.values, s.children.count == 2 else {
            throw ArrowMetalError.invalidArrowArray("a map's child must be a struct with two fields")
        }
        self.entries = entries
        self.keysSorted = keysSorted
    }

    public var length: Int { entries.length }
    public var nullCount: Int { entries.nullCount }
    public var context: MetalContext { entries.context }
    public var arrowFormat: String { "+m" }
    /// The `entries` struct: children `[keys, items]`.
    public var entryStruct: MetalStructArray { if case .structure(let s) = entries.values { return s } else { fatalError("unreachable") } }
    public var keys: AnyMetalArray { entryStruct.children[0] }
    public var items: AnyMetalArray { entryStruct.children[1] }
    public func isValid(_ i: Int) -> Bool { entries.isValid(i) }
    public func valueRange(_ i: Int) -> Range<Int>? { entries.valueRange(i) }

    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalMapArray {
        try MetalMapArray(entries: try entries.take(indices), keysSorted: keysSorted)
    }
    public func filter(_ mask: MetalBooleanArray) throws -> MetalMapArray {
        try MetalMapArray(entries: try entries.filter(mask), keysSorted: keysSorted)
    }
    public func slice(offset: Int, length: Int) throws -> MetalMapArray {
        try MetalMapArray(entries: try entries.slice(offset: offset, length: length), keysSorted: keysSorted)
    }
}

// MARK: - Union

/// An Arrow union array, dense ("+ud:") or sparse ("+us:").
///
/// Import, export and `filter` / `take` / `slice` only: a type-id-dispatched layout defeats the uniform
/// thread model the compute kernels rely on, so no kernel reads a union's values.
public final class MetalUnionArray: @unchecked Sendable {
    public enum Mode: String, Sendable { case dense = "d", sparse = "s" }

    public let mode: Mode
    public let length: Int
    /// One type code per element, selecting the child in `typeCodes`.
    public let typeIds: MetalArray<Int8>
    /// Dense unions only: the element's offset inside the selected child.
    public let offsets: MetalArray<Int32>?
    /// The type codes in the format string, in child order.
    public let typeCodes: [Int8]
    public let names: [String]
    public let children: [AnyMetalArray]
    public let context: MetalContext

    public init(mode: Mode, length: Int, typeIds: MetalArray<Int8>, offsets: MetalArray<Int32>?,
                typeCodes: [Int8], names: [String], children: [AnyMetalArray],
                context: MetalContext = .shared) throws {
        guard typeCodes.count == children.count, names.count == children.count else {
            throw ArrowMetalError.invalidArrowArray("union type codes / names / children count mismatch")
        }
        guard (mode == .dense) == (offsets != nil) else {
            throw ArrowMetalError.invalidArrowArray("a dense union needs an offsets buffer and a sparse union must not have one")
        }
        self.mode = mode; self.length = length; self.typeIds = typeIds; self.offsets = offsets
        self.typeCodes = typeCodes; self.names = names; self.children = children; self.context = context
    }

    /// Unions carry no validity bitmap of their own (Arrow removed it in 1.0); null-ness lives in the children.
    public var nullCount: Int { 0 }
    public var arrowFormat: String { "+u\(mode.rawValue):" + typeCodes.map(String.init).joined(separator: ",") }
    /// The child a given element selects, and the index inside it.
    public func location(_ i: Int) -> (child: Int, index: Int)? {
        let code = typeIds.valuePointer[i]
        guard let c = typeCodes.firstIndex(of: code) else { return nil }
        return (c, mode == .dense ? Int(offsets!.valuePointer[i]) : i)
    }

    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalUnionArray {
        let idx32: MetalArray<Int32> = try (indices as? MetalArray<Int32>) ?? indices.cast(to: Int32.self)
        // A union has no validity, so a null index has nothing to say; it selects the first child's slot 0.
        let ids = try dropNulls(try typeIds.take(idx32), fill: typeCodes.first ?? 0)
        switch mode {
        case .dense:
            let off = try dropNulls(try offsets!.take(idx32), fill: 0)
            return try MetalUnionArray(mode: mode, length: idx32.length, typeIds: ids, offsets: off,
                                       typeCodes: typeCodes, names: names, children: children, context: context)
        case .sparse:
            return try MetalUnionArray(mode: mode, length: idx32.length, typeIds: ids, offsets: nil,
                                       typeCodes: typeCodes, names: names,
                                       children: try children.map { try $0.take(idx32) }, context: context)
        }
    }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalUnionArray {
        guard mask.length == length else { throw ArrowMetalError.lengthMismatch(length, mask.length) }
        return try take(try MetalArray<Int32>.iota(length, context: context).filter(mask))
    }

    public func slice(offset: Int, length n: Int) throws -> MetalUnionArray {
        guard offset >= 0, n >= 0, offset + n <= length else {
            throw ArrowMetalError.invalidArrowArray("union slice \(offset)..<\(offset + n) is out of range (length \(length))")
        }
        return try take(try NestedSupport.iota(offset, n, context))
    }

    private func dropNulls<T: ArrowPrimitive>(_ a: MetalArray<T>, fill: T) throws -> MetalArray<T> {
        a.validity == nil ? a : try a.fillNull(fill)
    }
}

// MARK: - AnyMetalArray accessors

extension AnyMetalArray {
    public var asList: MetalListArray? { if case .list(let a) = self { return a } else { return nil } }
    public var asStruct: MetalStructArray? { if case .structure(let a) = self { return a } else { return nil } }
    public var asMap: MetalMapArray? { if case .map(let a) = self { return a } else { return nil } }
    public var asUnion: MetalUnionArray? { if case .union(let a) = self { return a } else { return nil } }

    /// True for `list` / `large_list` / `fixed_size_list` / `struct` / `map` / `union`.
    public var isNested: Bool {
        switch self {
        case .list, .structure, .map, .union: return true
        default: return false
        }
    }

    /// The child arrays of a nested (or dictionary) array: a list's values, a struct's fields, a map's
    /// entries, a union's variants, a dictionary's values. Empty for a flat array.
    public var children: [AnyMetalArray] {
        switch self {
        case .list(let l): return [l.values]
        case .structure(let s): return s.children
        case .map(let m): return [m.entries.values]
        case .union(let u): return u.children
        case .dictionary(_, let values): return [values]
        default: return []
        }
    }

    /// The child field names matching `children`.
    public var childNames: [String] {
        switch self {
        case .list(let l): return [l.fieldName]
        case .structure(let s): return s.names
        case .map: return ["entries"]
        case .union(let u): return u.names
        case .dictionary: return ["values"]
        default: return []
        }
    }

    /// Arrow `list_value_length` on a list or map column.
    public func listValueLength() throws -> MetalArray<Int32> { try listArray().listValueLength() }
    /// Arrow `list_flatten` on a list or map column.
    public func listFlatten() throws -> AnyMetalArray { try listArray().listFlatten() }
    /// Arrow `list_element` on a list or map column.
    public func listElement(_ index: Int) throws -> AnyMetalArray { try listArray().listElement(index) }
    /// Arrow `struct_field` on a struct column.
    public func structField(_ name: String) throws -> AnyMetalArray {
        guard case .structure(let s) = self else {
            throw ArrowMetalError.unsupportedType("struct_field needs a struct array, got \(arrowFormat)")
        }
        return try s.structField(name)
    }

    private func listArray() throws -> MetalListArray {
        switch self {
        case .list(let l): return l
        case .map(let m): return m.entries
        default: throw ArrowMetalError.unsupportedType("this function needs a list array, got \(arrowFormat)")
        }
    }
}

// MARK: - Import

/// True for the Arrow format strings this file imports.
func isNestedFormat(_ f: String) -> Bool {
    f == "+s" || f == "+l" || f == "+L" || f == "+m" || f.hasPrefix("+w:") || f.hasPrefix("+ud:") || f.hasPrefix("+us:")
}

/// Bitmap import with the array's bit offset applied (zero-copy only at offset 0 and page alignment).
private func nestedBitmap(_ src: UnsafeRawPointer, length: Int, offset: Int, owner: AnyObject,
                          context: MetalContext) throws -> (MetalArrowBuffer, Bool) {
    let bytes = Swift.max(Bitmap.byteCount(bits: length), 1)
    if offset == 0 { return try MetalArrowBuffer.wrapOrCopy(src, byteCount: bytes, keepAlive: owner, context: context) }
    let buf = try MetalArrowBuffer.allocate(byteCount: bytes, context: context)
    let sp = src.assumingMemoryBound(to: UInt8.self)
    let dp = buf.mutableTyped(UInt8.self)
    if offset % 8 == 0 { memcpy(dp, sp + offset / 8, bytes) }
    else { for i in 0..<length where Bitmap.isSet(sp, i + offset) { Bitmap.set(dp, i) } }
    return (buf, false)
}

/// Moves child `i` out of `array` (so its lifetime becomes independent of the parent) and imports it.
private func importNestedChild(_ schema: UnsafePointer<ArrowSchema>, _ array: UnsafeMutablePointer<ArrowArray>,
                               _ i: Int, context: MetalContext) throws -> (name: String, result: ImportResult) {
    guard let cs = schema.pointee.children?[i], let ca = array.pointee.children?[i] else {
        throw ArrowMetalError.invalidArrowArray("nested array is missing child \(i)")
    }
    let name = cs.pointee.name.map { String(cString: $0) } ?? ""
    let moved = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
    moved.initialize(to: ca.pointee)
    ca.pointee.release = nil
    defer { moved.deallocate() }
    return (name, try importArrowArray(schema: cs, array: moved, context: context))
}

/// Imports any of `+s`, `+l`, `+L`, `+w:N`, `+m`, `+ud:` and `+us:`.
///
/// Children are moved out first, then the parent struct is moved into an `ImportedCArray` that keeps the
/// parent's own buffers (validity, offsets, type ids) alive for as long as the Metal array borrows them.
func importNestedArray(format fmt: String, schema: UnsafePointer<ArrowSchema>,
                       array: UnsafeMutablePointer<ArrowArray>, context: MetalContext) throws -> ImportResult {
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    let nChildren = Int(schema.pointee.n_children)
    guard Int(array.pointee.n_children) == nChildren, nChildren > 0,
          schema.pointee.children != nil, array.pointee.children != nil else {
        throw ArrowMetalError.invalidArrowArray("schema/array child count mismatch for \(fmt)")
    }
    let expected = fmt == "+s" ? nChildren : (fmt.hasPrefix("+u") ? nChildren : 1)
    guard nChildren == expected else { throw ArrowMetalError.invalidArrowArray("\(fmt) expects \(expected) children, got \(nChildren)") }

    var names: [String] = [], kids: [AnyMetalArray] = []
    var zc = true
    for i in 0..<nChildren {
        let (name, r) = try importNestedChild(schema, array, i, context: context)
        names.append(name); kids.append(r.array); zc = zc && r.zeroCopy
    }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let declaredNulls = Int(a.null_count)
    let nBuffers = Int(a.n_buffers)

    func buffer(_ i: Int) -> UnsafeRawPointer? { a.buffers == nil ? nil : a.buffers[i].map { UnsafeRawPointer($0) } }

    switch fmt {
    case "+s":
        guard nBuffers == 1 else { throw ArrowMetalError.invalidArrowArray("struct arrays have one buffer, got \(nBuffers)") }
        var validity: MetalArrowBuffer? = nil
        if let vp = buffer(0) {
            let (b, z) = try nestedBitmap(vp, length: length, offset: offset, owner: owner, context: context)
            validity = b; zc = zc && z
        }
        if offset != 0 { kids = try kids.map { try $0.slice(offset: offset, length: length) }; zc = false }
        let s = try MetalStructArray(length: length, nullCount: 0, validity: validity, names: names,
                                     children: kids, context: context)
        if declaredNulls < 0 || validity != nil { s.recomputeNullCount() } else { s.nullCount = declaredNulls }
        return ImportResult(array: .structure(s), zeroCopy: zc)

    case "+l", "+L", "+m":
        guard nBuffers == 2 else { throw ArrowMetalError.invalidArrowArray("\(fmt) arrays have two buffers, got \(nBuffers)") }
        guard let offPtr = buffer(1) else { throw ArrowMetalError.invalidArrowArray("list offsets buffer is null") }
        var validity: MetalArrowBuffer? = nil
        if let vp = buffer(0) {
            let (b, z) = try nestedBitmap(vp, length: length, offset: offset, owner: owner, context: context)
            validity = b; zc = zc && z
        }
        let offsets: MetalArrowBuffer
        if fmt == "+L" {
            // large_list: narrow the 64-bit offsets, as large_utf8 does. Offsets are not rebased, so the
            // child stays shared; the range must fit in int32.
            zc = false
            let ob = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
            let dst = ob.mutableTyped(Int32.self)
            let src = offPtr.assumingMemoryBound(to: Int64.self)
            for i in 0...length {
                let v = src[offset + i]
                guard v <= Int64(Int32.max) else { throw ArrowMetalError.unsupportedType("large_list beyond 2^31 child elements") }
                dst[i] = Int32(v)
            }
            offsets = ob
        } else if offset == 0 {
            let (ob, z) = try MetalArrowBuffer.wrapOrCopy(offPtr, byteCount: (length + 1) * 4, keepAlive: owner, context: context)
            offsets = ob; zc = zc && z
        } else {
            zc = false
            offsets = try MetalArrowBuffer.copy(from: offPtr.advanced(by: offset * 4), byteCount: (length + 1) * 4, context: context)
        }
        let list = MetalListArray(length: length, nullCount: 0, validity: validity, offsets: offsets,
                                  values: kids[0], kind: .variable, fieldName: names[0], context: context)
        if declaredNulls < 0 || validity != nil { list.recomputeNullCount() } else { list.setNullCount(declaredNulls) }
        guard list.childRange.upperBound <= kids[0].length else {
            throw ArrowMetalError.invalidArrowArray("list offsets reach \(list.childRange.upperBound) but the child has \(kids[0].length) elements")
        }
        if fmt == "+m" {
            let sorted = (schema.pointee.flags & Int64(ARROW_FLAG_MAP_KEYS_SORTED)) != 0
            return ImportResult(array: .map(try MetalMapArray(entries: list, keysSorted: sorted)), zeroCopy: zc)
        }
        return ImportResult(array: .list(list), zeroCopy: zc)

    case let f where f.hasPrefix("+w:"):
        guard let width = Int(f.dropFirst(3)), width >= 0 else { throw ArrowMetalError.unsupportedType(f) }
        guard nBuffers == 1 else { throw ArrowMetalError.invalidArrowArray("fixed_size_list arrays have one buffer, got \(nBuffers)") }
        var validity: MetalArrowBuffer? = nil
        if let vp = buffer(0) {
            let (b, z) = try nestedBitmap(vp, length: length, offset: offset, owner: owner, context: context)
            validity = b; zc = zc && z
        }
        var child = kids[0]
        if offset != 0 {
            child = try child.slice(offset: offset * width, length: length * width)
            zc = false
        }
        guard child.length >= length * width else {
            throw ArrowMetalError.invalidArrowArray("fixed_size_list<\(width)> of \(length) rows needs \(length * width) child elements, got \(child.length)")
        }
        let ob = try MetalArrowBuffer.allocate(byteCount: (length + 1) * 4, zeroed: false, context: context)
        let p = ob.mutableTyped(Int32.self)
        for i in 0...length { p[i] = Int32(i * width) }
        let list = MetalListArray(length: length, nullCount: 0, validity: validity, offsets: ob, values: child,
                                  kind: .fixedSize(width), fieldName: names[0], context: context)
        if declaredNulls < 0 || validity != nil { list.recomputeNullCount() } else { list.setNullCount(declaredNulls) }
        return ImportResult(array: .list(list), zeroCopy: zc)

    case let f where f.hasPrefix("+ud:") || f.hasPrefix("+us:"):
        let dense = f.hasPrefix("+ud:")
        let codes = try parseUnionCodes(f)
        guard codes.count == nChildren else {
            throw ArrowMetalError.invalidArrowArray("\(f) declares \(codes.count) type codes but has \(nChildren) children")
        }
        // Arrow 1.0 removed the union validity bitmap; tolerate an older producer that still sends a null one.
        var base = 0
        if nBuffers == (dense ? 3 : 2) {
            guard buffer(0) == nil else { throw ArrowMetalError.unsupportedType("union arrays with a validity bitmap are not supported") }
            base = 1
        } else if nBuffers != (dense ? 2 : 1) {
            throw ArrowMetalError.invalidArrowArray("\(f) arrays have \(dense ? 2 : 1) buffers, got \(nBuffers)")
        }
        guard let idPtr = buffer(base) else { throw ArrowMetalError.invalidArrowArray("union type ids buffer is null") }
        zc = false
        let idBuf = try MetalArrowBuffer.copy(from: idPtr.advanced(by: offset), byteCount: Swift.max(length, 1), context: context)
        let ids = MetalArray<Int8>(length: length, nullCount: 0, validity: nil, values: idBuf, context: context)
        var offs: MetalArray<Int32>? = nil
        if dense {
            guard let op = buffer(base + 1) else { throw ArrowMetalError.invalidArrowArray("dense union offsets buffer is null") }
            let ob = try MetalArrowBuffer.copy(from: op.advanced(by: offset * 4), byteCount: Swift.max(length, 1) * 4, context: context)
            offs = MetalArray<Int32>(length: length, nullCount: 0, validity: nil, values: ob, context: context)
        } else if offset != 0 {
            kids = try kids.map { try $0.slice(offset: offset, length: length) }
        }
        let u = try MetalUnionArray(mode: dense ? .dense : .sparse, length: length, typeIds: ids, offsets: offs,
                                    typeCodes: codes, names: names, children: kids, context: context)
        return ImportResult(array: .union(u), zeroCopy: zc)

    default:
        throw ArrowMetalError.unsupportedType(fmt)
    }
}

/// "+ud:0,1" -> [0, 1].
private func parseUnionCodes(_ format: String) throws -> [Int8] {
    let body = String(format.dropFirst(4))
    if body.isEmpty { return [] }
    var out: [Int8] = []
    for part in body.split(separator: ",") {
        guard let v = Int8(part) else { throw ArrowMetalError.unsupportedType(format) }
        out.append(v)
    }
    return out
}

// MARK: - Export

/// Owns the child schemas and the strings of an exported nested schema.
final class NestedSchemaHolder {
    let formatC: UnsafeMutablePointer<CChar>
    let nameC: UnsafeMutablePointer<CChar>
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowSchema>?>
    let childStructs: UnsafeMutablePointer<ArrowSchema>
    let n: Int
    init(format: String, name: String, n: Int) {
        formatC = strdup(format)!
        nameC = strdup(name)!
        self.n = n
        let cap = Swift.max(n, 1)
        children = .allocate(capacity: cap)
        childStructs = .allocate(capacity: cap)
        childStructs.initialize(repeating: ArrowSchema(), count: cap)
        for i in 0..<n { children[i] = childStructs + i }
    }
    deinit {
        for i in 0..<n { if let r = childStructs[i].release { r(childStructs + i) } }
        children.deallocate(); childStructs.deallocate(); free(formatC); free(nameC)
    }
}

/// Owns the buffer pointer table, the child arrays and the Metal buffers of an exported nested array.
final class NestedArrayHolder {
    let buffers: UnsafeMutablePointer<UnsafeRawPointer?>
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowArray>?>
    let childStructs: UnsafeMutablePointer<ArrowArray>
    let n: Int
    var keep: [AnyObject] = []
    init(bufferPtrs: [UnsafeRawPointer?], n: Int) {
        buffers = .allocate(capacity: Swift.max(bufferPtrs.count, 1))
        for (i, p) in bufferPtrs.enumerated() { buffers[i] = p }
        self.n = n
        let cap = Swift.max(n, 1)
        children = .allocate(capacity: cap)
        childStructs = .allocate(capacity: cap)
        childStructs.initialize(repeating: ArrowArray(), count: cap)
        for i in 0..<n { children[i] = childStructs + i }
    }
    deinit {
        for i in 0..<n { if let r = childStructs[i].release { r(childStructs + i) } }
        buffers.deallocate(); children.deallocate(); childStructs.deallocate()
    }
}

private func releaseNestedSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<NestedSchemaHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}
private func releaseNestedArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<NestedArrayHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

/// Writes a nested schema: `format`, `name` and one child per entry of `children`, each written by the
/// child array's own schema export.
private func writeNestedSchema(format: String, name: String, flags: Int64,
                               children: [(name: String, array: AnyMetalArray)],
                               into out: UnsafeMutablePointer<ArrowSchema>) -> NestedSchemaHolder {
    let h = NestedSchemaHolder(format: format, name: name, n: children.count)
    for (i, c) in children.enumerated() { c.array.exportArrowSchema(name: c.name, into: h.childStructs + i) }
    out.pointee.format = UnsafePointer(h.formatC)
    out.pointee.name = UnsafePointer(h.nameC)
    out.pointee.metadata = nil
    out.pointee.flags = flags
    out.pointee.n_children = Int64(children.count)
    out.pointee.children = children.isEmpty ? nil : h.children
    out.pointee.dictionary = nil
    out.pointee.release = releaseNestedSchema
    out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
    return h
}

/// Writes a nested array: `bufferPtrs` at the top level and one child per entry of `children`.
private func writeNestedArray(length: Int, nullCount: Int, bufferPtrs: [UnsafeRawPointer?],
                              keep: [AnyObject], children: [AnyMetalArray],
                              into out: UnsafeMutablePointer<ArrowArray>) {
    let h = NestedArrayHolder(bufferPtrs: bufferPtrs, n: children.count)
    h.keep = keep
    for (i, c) in children.enumerated() { c.exportArrowArray(into: h.childStructs + i) }
    out.pointee.length = Int64(length)
    out.pointee.null_count = Int64(nullCount)
    out.pointee.offset = 0
    out.pointee.n_buffers = Int64(bufferPtrs.count)
    out.pointee.n_children = Int64(children.count)
    out.pointee.buffers = UnsafeMutablePointer<UnsafeRawPointer?>(h.buffers)
    out.pointee.children = children.isEmpty ? nil : h.children
    out.pointee.dictionary = nil
    out.pointee.release = releaseNestedArray
    out.pointee.private_data = Unmanaged.passRetained(h).toOpaque()
}

extension MetalListArray {
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        _ = writeNestedSchema(format: kind.arrowFormat, name: name, flags: Int64(ARROW_FLAG_NULLABLE),
                              children: [(fieldName, values)], into: out)
    }
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        var keep: [AnyObject] = [self, offsets]
        if let v = validity { keep.append(v) }
        // A fixed_size_list carries no offsets buffer; its offsets are `i * N` by definition.
        let bufs: [UnsafeRawPointer?] = kind.fixedWidth != nil
            ? [validity?.contents]
            : [validity?.contents, offsets.contents]
        writeNestedArray(length: length, nullCount: nullCount, bufferPtrs: bufs, keep: keep,
                         children: [values], into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

extension MetalStructArray {
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        _ = writeNestedSchema(format: "+s", name: name, flags: Int64(ARROW_FLAG_NULLABLE),
                              children: Array(zip(names, children)).map { (name: $0.0, array: $0.1) }, into: out)
    }
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        var keep: [AnyObject] = [self]
        if let v = validity { keep.append(v) }
        writeNestedArray(length: length, nullCount: nullCount, bufferPtrs: [validity?.contents], keep: keep,
                         children: children, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

extension MetalMapArray {
    /// The map schema Arrow requires: a "+m" whose single child is a non-nullable `entries` struct with a
    /// non-nullable `key` field.
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        let flags = Int64(ARROW_FLAG_NULLABLE) | (keysSorted ? Int64(ARROW_FLAG_MAP_KEYS_SORTED) : 0)
        let h = writeNestedSchema(format: "+m", name: name, flags: flags,
                                  children: [("entries", entries.values)], into: out)
        h.childStructs[0].flags = 0                                   // entries: non-nullable
        h.childStructs[0].children?[0]?.pointee.flags = 0             // key: non-nullable
    }
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        entries.exportArrowArray(into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}

extension MetalUnionArray {
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        _ = writeNestedSchema(format: arrowFormat, name: name, flags: Int64(ARROW_FLAG_NULLABLE),
                              children: Array(zip(names, children)).map { (name: $0.0, array: $0.1) }, into: out)
    }
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        var keep: [AnyObject] = [self, typeIds.values]
        var bufs: [UnsafeRawPointer?] = [typeIds.values.contents]
        if let o = offsets { bufs.append(o.values.contents); keep.append(o.values) }
        writeNestedArray(length: length, nullCount: 0, bufferPtrs: bufs, keep: keep, children: children, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        withUnsafeMutablePointer(to: &out.pointee.array) { exportArrowArray(into: $0) }
        out.pointee.device_type = ARROW_DEVICE_METAL
        out.pointee.device_id = -1
        out.pointee.reserved = (0, 0, 0)
        out.pointee.sync_event = nil
    }
}
