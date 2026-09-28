import Foundation

/// Multi-column (lexicographic) sorting, built out of the single-key radix argsort.
///
/// The existing `argsort` is stable, which is the whole trick: sorting by the *least* significant key
/// first and then by each more significant one in turn leaves rows ordered by the first key, ties broken
/// by the second, and so on — the LSD radix idea one column up. Each pass reorders the next key column
/// with `take` before sorting it, so every pass sees the keys in the order the previous passes left them.
///
/// Nulls sit at one end of every key, in both directions: `argsort` places them past the values whether
/// the pass is ascending or descending, so a null sorts as "greater than any value" (or, with
/// `nullPlacement: .atStart`, "less than any value") at every level. `nullPlacement` applies to every
/// key, which is how Arrow's `SortOptions.null_placement` works.
///
/// Cost is one argsort and two `take`s per key. For k keys over n rows that is k radix sorts, which is
/// still far cheaper than a comparison sort with a k-way comparator, and it needs no new kernel.
public func lexsortIndices(_ columns: [AnyMetalArray], descending: [Bool] = [],
                           nullPlacement: NullPlacement = .atEnd) throws -> MetalArray<Int32> {
    try lexsortIndices(columns, descending: descending, nullPlacements: [nullPlacement], floatOrders: [])
}

/// `lexsortIndices` with the null placement and the float order chosen per key.
///
/// `nullPlacements` and `floatOrders` each hold one entry per column, one entry for every column, or
/// none for the defaults (`.atEnd`, `.ieee`). Each key's options go into that key's own radix pass —
/// the null block at that key's chosen end, the float key map that key's order asks for — so a key
/// with options costs exactly what a key without them does.
public func lexsortIndices(_ columns: [AnyMetalArray], descending: [Bool] = [],
                           nullPlacements: [NullPlacement], floatOrders: [FloatOrder]) throws -> MetalArray<Int32> {
    guard let first = columns.first else {
        throw ArrowMetalError.invalidArrowArray("lexsort needs at least one column")
    }
    guard descending.isEmpty || descending.count == columns.count else {
        throw ArrowMetalError.invalidArrowArray("descending has \(descending.count) entries for \(columns.count) columns")
    }
    guard nullPlacements.count <= 1 || nullPlacements.count == columns.count else {
        throw ArrowMetalError.invalidArrowArray("null placement has \(nullPlacements.count) entries for \(columns.count) columns")
    }
    guard floatOrders.count <= 1 || floatOrders.count == columns.count else {
        throw ArrowMetalError.invalidArrowArray("float order has \(floatOrders.count) entries for \(columns.count) columns")
    }
    func placement(_ k: Int) -> NullPlacement {
        nullPlacements.isEmpty ? .atEnd : nullPlacements[nullPlacements.count == 1 ? 0 : k]
    }
    func order(_ k: Int) -> FloatOrder { floatOrders.isEmpty ? .ieee : floatOrders[floatOrders.count == 1 ? 0 : k] }
    let n = first.length
    for c in columns where c.length != n { throw ArrowMetalError.lengthMismatch(n, c.length) }
    let ctx = first.metalContext
    if columns.count == 1 {
        return try columns[0].argsortIndices(descending: descending.first ?? false, nullPlacement: placement(0),
                                             floatOrder: order(0))
    }
    guard n > 0 else { return try MetalArray<Int32>([Int32](), context: ctx) }

    var perm: MetalArray<Int32>? = nil                  // nil means "the identity so far"
    for k in columns.indices.reversed() {
        let desc = descending.isEmpty ? false : descending[k]
        // The first pass sees the column as it is; later ones see it in the order the previous passes left.
        let keys = try perm.map { try columns[k].take($0) } ?? columns[k]
        let idx = try keys.argsortIndices(descending: desc, nullPlacement: placement(k), floatOrder: order(k))
        perm = try perm.map { try $0.take(idx) } ?? idx
    }
    return perm!
}

extension AnyMetalArray {
    /// Stable argsort of whichever concrete array this is (`MetalArray.argsort`), with the nulls at
    /// whichever end `nullPlacement` names.
    ///
    /// Booleans go through their unpacked byte form and temporal columns through their integer storage.
    /// utf8 and binary columns sort byte-wise through the prefix radix sort in `Kernels/StringSort.swift`,
    /// so a lexsort may mix them freely with numeric keys. Dictionary-encoded and nested columns have no
    /// order-preserving GPU key yet, so they throw. `floatOrder` applies to Float32, Float64 and float16
    /// columns (see `FloatOrder`) and is ignored by every other type.
    public func argsortIndices(descending: Bool = false,
                               nullPlacement: NullPlacement = .atEnd,
                               floatOrder: FloatOrder = .ieee) throws -> MetalArray<Int32> {
        switch self {
        case .int8(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .uint8(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .int16(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .uint16(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .int32(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .uint32(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .int64(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .uint64(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .float32(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder)
        case .float64(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder)
        case .boolean(let a): return try a.toUInt8Array().argsort(descending: descending, nullPlacement: nullPlacement)
        case .temporal(let t):
            switch t.storage {
            case .int32(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
            case .int64(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
            }
        // utf8 and binary sort byte-wise on the GPU (`Kernels/StringSort.swift`), which is the order
        // Arrow defines for them.
        case .string(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .binary(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
        case .dictionary, .runEndEncoded, .decimal, .list, .structure, .map, .union:
            throw ArrowMetalError.unsupportedType("sort by \(arrowFormat) is not implemented")
        // float16 sorts through the float32 widening; the rest have no order-preserving GPU key.
        case .float16(let a): return try a.toFloat32().argsort(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder)
        case .smallDecimal(let s):
            switch s.storage {
            case .int32(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
            case .int64(let a): return try a.argsort(descending: descending, nullPlacement: nullPlacement)
            }
        case .extended(let a): return try a.storage.argsortIndices(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder)
        case .null, .interval, .fixedBinary:
            throw ArrowMetalError.unsupportedType("sort by \(arrowFormat) is not implemented")
        }
    }

    /// The sorted values of whichever concrete array this is, in `argsortIndices`' order and of the
    /// same type as this one.
    ///
    /// The six types that have a direct order-preserving key take `MetalArray.sorted()`, which rebuilds
    /// the values out of the sort's own keys instead of gathering them through the permutation; every
    /// other type still gathers. A dictionary column orders by the values its codes point at but comes
    /// back a dictionary: the codes are gathered and the value array is left alone, which is what
    /// `take` of `argsort` has always done for it.
    public func sortedValues(descending: Bool = false,
                             nullPlacement: NullPlacement = .atEnd,
                             floatOrder: FloatOrder = .ieee) throws -> AnyMetalArray {
        switch self {
        case .int32(let a): return .int32(try a.sorted(descending: descending, nullPlacement: nullPlacement))
        case .uint32(let a): return .uint32(try a.sorted(descending: descending, nullPlacement: nullPlacement))
        case .int64(let a): return .int64(try a.sorted(descending: descending, nullPlacement: nullPlacement))
        case .uint64(let a): return .uint64(try a.sorted(descending: descending, nullPlacement: nullPlacement))
        case .float32(let a): return .float32(try a.sorted(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder))
        case .float64(let a): return .float64(try a.sorted(descending: descending, nullPlacement: nullPlacement, floatOrder: floatOrder))
        case .dictionary:
            // The codes carry no order of their own, so the permutation comes from the decoded column;
            // the gather then runs on the codes, so the dictionary survives.
            let order = try decodedIfDictionary().argsortIndices(descending: descending,
                                                                nullPlacement: nullPlacement,
                                                                floatOrder: floatOrder)
            return try take(order)
        default:
            return try take(try argsortIndices(descending: descending, nullPlacement: nullPlacement,
                                               floatOrder: floatOrder))
        }
    }

    /// The Metal context whichever concrete array this is lives in.
    var metalContext: MetalContext {
        switch self {
        case .int8(let a): return a.context
        case .uint8(let a): return a.context
        case .int16(let a): return a.context
        case .uint16(let a): return a.context
        case .int32(let a): return a.context
        case .uint32(let a): return a.context
        case .int64(let a): return a.context
        case .uint64(let a): return a.context
        case .float32(let a): return a.context
        case .float64(let a): return a.context
        case .boolean(let a): return a.context
        case .string(let a): return a.context
        case .temporal(let a): return a.context
        case .binary(let a): return a.context
        case .dictionary(let codes, _): return codes.context
        case .runEndEncoded(let runEnds, _): return runEnds.context
        case .decimal(let a): return a.context
        case .list(let a): return a.context
        case .structure(let a): return a.context
        case .map(let a): return a.context
        case .union(let a): return a.context
        case .null(let a): return a.context
        case .float16(let a): return a.context
        case .smallDecimal(let a): return a.context
        case .interval(let a): return a.context
        case .fixedBinary(let a): return a.context
        case .extended(let a): return a.storage.metalContext
        }
    }
}

// `partition_nth_indices` lives in `Kernels/PartitionNth.swift`: a GPU radix select, not a sort.

extension MetalRecordBatch {
    /// Sorts every column by several keys at once, most significant first (stable, nulls at
    /// `nullPlacement`'s end in every key). `batch.sorted(by: [("region", false), ("revenue", true)])`
    /// orders by region ascending and breaks ties by revenue descending.
    public func sorted(by keys: [(column: String, descending: Bool)],
                       nullPlacement: NullPlacement = .atEnd) throws -> MetalRecordBatch {
        guard !keys.isEmpty else { return self }
        var cols: [AnyMetalArray] = []
        var desc: [Bool] = []
        for k in keys {
            guard let c = self[k.column] else { throw ArrowMetalError.invalidArrowArray("no column named \(k.column)") }
            cols.append(c)
            desc.append(k.descending)
        }
        return try take(try lexsortIndices(cols, descending: desc, nullPlacement: nullPlacement))
    }

    /// Sorts every column by several keys, each with its own direction, null placement and float order
    /// (`SortKey`). Keys that leave both options at their defaults sort exactly as `sorted(by:)` does.
    public func sorted(by keys: [SortKey]) throws -> MetalRecordBatch {
        guard !keys.isEmpty else { return self }
        var cols: [AnyMetalArray] = []
        for k in keys {
            guard let c = self[k.column] else { throw ArrowMetalError.invalidArrowArray("no column named \(k.column)") }
            cols.append(c)
        }
        return try take(try lexsortIndices(cols, descending: keys.map(\.descending),
                                           nullPlacements: keys.map { $0.nullsFirst ? .atStart : .atEnd },
                                           floatOrders: keys.map(\.floatOrder)))
    }
}
