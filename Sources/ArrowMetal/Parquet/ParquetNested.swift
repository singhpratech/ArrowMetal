import Foundation
import Metal

// Nested columns: structs, maps and lists at any depth, reassembled from their leaves.
//
// Dremel assembly, field by field. Every leaf beneath a column is decoded once with its definition and
// repetition levels (one byte per level entry, on the GPU), and every Arrow-level field knows three
// levels (`ParquetField`): an entry of a leaf beneath it is a *slot* of the field when its repetition
// level is at most `repetitionLevel` and its definition level reaches `slotDefinitionLevel`, and the
// slot is non-null when the definition level reaches `definitionLevel`. The level entries of any one
// leaf beneath a field describe that field completely, so:
//
//   - a leaf is its decoded array restricted to its own slots (the package's `filter`);
//   - a struct is its children, plus a validity bitmap read off one leaf's entries at the struct's slots;
//   - a list is its child, plus offsets: at each of the list's slots, the number of child slots before it;
//   - a map is a list whose child is the key/value entries struct.
//
// Each of those is `pq_nest_flags` (one pass over the entries), one prefix sum, and `pq_nest_scatter`.
// A one-level `list<primitive>` keeps its original kernel pair (`ParquetList.swift`).

/// Builds the Arrow array of one nested field. Holds the decoded leaves so each is decoded once.
final class ParquetNestedAssembler {
    let file: ParquetFile
    let context: MetalContext
    let rowGroups: [Int]
    let options: ParquetReadOptions
    private var decoded: [Int: ParquetLeafData] = [:]

    init(file: ParquetFile, rowGroups: [Int], options: ParquetReadOptions) {
        self.file = file
        self.context = file.context
        self.rowGroups = rowGroups
        self.options = options
    }

    /// A leaf's decoded buffers with both level streams, decoded on first use.
    func leafData(_ l: ParquetLeaf) throws -> ParquetLeafData {
        if let d = decoded[l.index] { return d }
        let d = try file.decodeLeaf(l, rowGroups: rowGroups, options: options, needRepetition: true)
        decoded[l.index] = d
        return d
    }

    /// The array for a top-level field. Its length must be the row count of the selected row groups.
    func buildTopLevel(_ f: ParquetField) throws -> AnyMetalArray {
        let out = try build(f, column: f.name)
        let rows = rowGroups.reduce(0) { $0 + Int(file.metadata.rowGroups[$1].numRows) }
        guard out.length == rows else {
            throw ParquetError.malformed("column \(f.name) assembles to \(out.length) rows, the row groups hold \(rows)")
        }
        return out
    }

    // MARK: - Slots

    /// Where one field's slots are among a leaf's level entries.
    struct Slots {
        /// Number of slots (the length of the field's array).
        let count: Int
        /// 1 at each entry that is a slot, `entries + 1` words (the last is 0), or nil when every entry is one.
        let flags: MetalArrowBuffer?
        /// The same flags, one byte per entry, for building a filter mask.
        let flagBytes: MetalArrowBuffer?
        /// Exclusive prefix sum of `flags` (`entries + 1` words), or nil when every entry is a slot.
        let index: MetalArrowBuffer?
        /// 1 where the entry's definition level reaches the field's `definitionLevel`.
        let valid: MetalArrowBuffer
    }

    /// Flags the slots of a field among `data`'s entries and numbers them.
    private func slots(of data: ParquetLeafData, maxRep: Int, slotDef: Int, validDef: Int,
                       numbered: Bool = true) throws -> Slots {
        let ctx = context
        let n = data.levels
        let flags = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: ctx)
        let bytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        let valid = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx)
        let dummy = flags
        let def = data.defLevels ?? dummy
        let rep = data.repLevels ?? dummy
        try ParquetTypeMap.run(file, ctx, "pq_nest_flags", count: n + 1) { enc in
            enc.setBuffer(def.mtl, offset: def.offset, index: 0)
            enc.setBuffer(rep.mtl, offset: rep.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            Dispatch.setUInt(enc, data.defLevels == nil ? 0 : 1, index: 3)
            Dispatch.setUInt(enc, data.repLevels == nil ? 0 : 1, index: 4)
            Dispatch.setUInt(enc, maxRep, index: 5)
            Dispatch.setUInt(enc, slotDef, index: 6)
            Dispatch.setUInt(enc, validDef, index: 7)
            enc.setBuffer(flags.mtl, offset: flags.offset, index: 8)
            enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 9)
            enc.setBuffer(valid.mtl, offset: valid.offset, index: 10)
        }
        guard numbered else {
            return Slots(count: -1, flags: flags, flagBytes: bytes, index: nil, valid: valid)
        }
        let index = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: ctx)
        let count = try file.scanU32(ctx, input: flags, output: index, n: n + 1)
        return Slots(count: count, flags: flags, flagBytes: bytes, index: index, valid: valid)
    }

    /// True when every entry of `data` is a slot of a field with these levels: no repeated ancestor,
    /// and no repetition below the field in this leaf.
    private func everyEntryIsSlot(_ data: ParquetLeafData, maxRep: Int, slotDef: Int) -> Bool {
        slotDef == 0 && data.leaf.maxRepetition <= maxRep
    }

    /// Packs one validity byte per slot into a bitmap; nil when nothing is null.
    private func packValidity(_ bytes: MetalArrowBuffer, count: Int) throws -> (MetalArrowBuffer?, Int) {
        guard count > 0 else { return (nil, 0) }
        let ctx = context
        let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: count), 4),
                                                 zeroed: true, context: ctx)
        try ParquetTypeMap.run(file, ctx, "pq_bytes_to_bitmap", count: (count + 31) / 32) { enc in
            enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 0)
            Dispatch.setUInt(enc, count, index: 1)
            enc.setBuffer(bits.mtl, offset: bits.offset, index: 2)
        }
        try ctx.syncPoint()
        let set = Bitmap.popcount(bits.typed(UInt8.self), bits: count)
        return set < count ? (bits, count - set) : (nil, 0)
    }

    /// Validity bytes at a field's slots, compacted out of the per-entry bytes.
    private func slotValidity(_ s: Slots, entries n: Int) throws -> MetalArrowBuffer {
        guard let flags = s.flags, let index = s.index else { return s.valid }
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(s.count, 1), zeroed: true, context: ctx)
        try ParquetTypeMap.run(file, ctx, "pq_nest_scatter", count: n + 1) { enc in
            enc.setBuffer(flags.mtl, offset: flags.offset, index: 0)
            enc.setBuffer(index.mtl, offset: index.offset, index: 1)
            enc.setBuffer(s.valid.mtl, offset: s.valid.offset, index: 2)
            enc.setBuffer(index.mtl, offset: index.offset, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            Dispatch.setUInt(enc, 1, index: 5)
            enc.setBuffer(out.mtl, offset: out.offset, index: 6)
            enc.setBuffer(index.mtl, offset: index.offset, index: 7)
        }
        return out
    }

    // MARK: - Fields

    func build(_ f: ParquetField, column: String) throws -> AnyMetalArray {
        switch f.kind {
        case .leaf(let l):
            let data = try leafData(l)
            let array = try data.arrowArray()
            if everyEntryIsSlot(data, maxRep: f.repetitionLevel, slotDef: f.slotDefinitionLevel) { return array }
            // Below a repeated ancestor: keep only the entries that are elements of this leaf.
            let s = try slots(of: data, maxRep: f.repetitionLevel, slotDef: f.slotDefinitionLevel,
                              validDef: f.definitionLevel, numbered: false)
            let n = data.levels
            guard n > 0 else { return array }
            let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 4),
                                                     zeroed: true, context: context)
            try ParquetTypeMap.run(file, context, "pq_bytes_to_bitmap", count: (n + 31) / 32) { enc in
                enc.setBuffer(s.flagBytes!.mtl, offset: s.flagBytes!.offset, index: 0)
                Dispatch.setUInt(enc, n, index: 1)
                enc.setBuffer(bits.mtl, offset: bits.offset, index: 2)
            }
            let mask = MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: bits, context: context)
            return try array.filter(mask)

        case .group(let children):
            guard let first = f.leaves.first else {
                throw ParquetError.malformed("struct column \(column) has no leaf columns")
            }
            let data = try leafData(first)
            let n = data.levels
            var length = n
            var validity: MetalArrowBuffer? = nil
            var nulls = 0
            if everyEntryIsSlot(data, maxRep: f.repetitionLevel, slotDef: f.slotDefinitionLevel) {
                if f.nullable, data.defLevels != nil {
                    let s = try slots(of: data, maxRep: f.repetitionLevel, slotDef: 0,
                                      validDef: f.definitionLevel, numbered: false)
                    (validity, nulls) = try packValidity(s.valid, count: n)
                }
            } else {
                let s = try slots(of: data, maxRep: f.repetitionLevel, slotDef: f.slotDefinitionLevel,
                                  validDef: f.definitionLevel)
                length = s.count
                if f.nullable, data.defLevels != nil {
                    (validity, nulls) = try packValidity(try slotValidity(s, entries: n), count: length)
                }
            }
            var names: [String] = []
            var arrays: [AnyMetalArray] = []
            for c in children {
                let a = try build(c, column: column + "." + c.name)
                guard a.length == length else {
                    throw ParquetError.malformed(
                        "struct column \(column) has \(length) slots but its field \(c.name) has \(a.length)")
                }
                names.append(c.name)
                arrays.append(a)
            }
            return .structure(try MetalStructArray(length: length, nullCount: nulls, validity: validity,
                                                   names: names, children: arrays, context: context))

        case .list(let element, _):
            guard let first = f.leaves.first else {
                throw ParquetError.malformed("list column \(column) has no leaf columns")
            }
            let data = try leafData(first)
            let n = data.levels
            let ctx = context
            let s = try slots(of: data, maxRep: f.repetitionLevel, slotDef: f.slotDefinitionLevel,
                              validDef: f.definitionLevel)
            let c = try slots(of: data, maxRep: element.repetitionLevel, slotDef: element.slotDefinitionLevel,
                              validDef: element.definitionLevel)
            let child = try build(element, column: column + "." + element.name)
            guard child.length == c.count else {
                throw ParquetError.malformed(
                    "list column \(column) has \(c.count) elements but its child assembles to \(child.length)")
            }
            guard c.count <= Int(Int32.max) else {
                throw ParquetError.unsupported("list column \(column) has \(c.count) elements, past int32 offsets")
            }
            let length = s.count
            let offsets = try MetalArrowBuffer.allocate(byteCount: Swift.max((length + 1) * 4, 8), zeroed: true, context: ctx)
            let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(length, 1), zeroed: true, context: ctx)
            let flags = s.flags!, index = s.index!, childIndex = c.index!
            try ParquetTypeMap.run(file, ctx, "pq_nest_scatter", count: n + 1) { enc in
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 0)
                enc.setBuffer(index.mtl, offset: index.offset, index: 1)
                enc.setBuffer(s.valid.mtl, offset: s.valid.offset, index: 2)
                enc.setBuffer(childIndex.mtl, offset: childIndex.offset, index: 3)
                Dispatch.setUInt(enc, n, index: 4)
                Dispatch.setUInt(enc, 3, index: 5)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 6)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 7)
            }
            var validity: MetalArrowBuffer? = nil
            var nulls = 0
            if f.nullable, data.defLevels != nil {
                (validity, nulls) = try packValidity(validBytes, count: length)
            }
            let list = MetalListArray(length: length, nullCount: nulls, validity: validity, offsets: offsets,
                                      values: child, fieldName: f.isMap ? "entries" : element.name, context: ctx)
            if f.isMap {
                guard case .structure(let entries) = child, entries.children.count == 2 else {
                    throw ParquetError.malformed("map column \(column) has entries that are not a key/value pair")
                }
                guard entries.children[0].nullCount == 0 else {
                    throw ParquetError.malformed("map column \(column) has a null key")
                }
                return .map(try MetalMapArray(entries: list))
            }
            return .list(list)
        }
    }
}
