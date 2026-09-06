import Foundation

/// Multi-column (lexicographic) sorting, built out of the single-key radix argsort.
///
/// The existing `argsort` is stable, which is the whole trick: sorting by the *least* significant key
/// first and then by each more significant one in turn leaves rows ordered by the first key, ties broken
/// by the second, and so on — the LSD radix idea one column up. Each pass reorders the next key column
/// with `take` before sorting it, so every pass sees the keys in the order the previous passes left them.
///
/// Nulls are last in every key, in both directions: `argsort` places them after the values whether the
/// pass is ascending or descending, so a null sorts as "greater than any value" at every level.
///
/// Cost is one argsort and two `take`s per key. For k keys over n rows that is k radix sorts, which is
/// still far cheaper than a comparison sort with a k-way comparator, and it needs no new kernel.
public func lexsortIndices(_ columns: [AnyMetalArray], descending: [Bool] = []) throws -> MetalArray<Int32> {
    guard let first = columns.first else {
        throw ArrowMetalError.invalidArrowArray("lexsort needs at least one column")
    }
    guard descending.isEmpty || descending.count == columns.count else {
        throw ArrowMetalError.invalidArrowArray("descending has \(descending.count) entries for \(columns.count) columns")
    }
    let n = first.length
    for c in columns where c.length != n { throw ArrowMetalError.lengthMismatch(n, c.length) }
    let ctx = first.metalContext
    if columns.count == 1 { return try columns[0].argsortIndices(descending: descending.first ?? false) }
    guard n > 0 else { return try MetalArray<Int32>([Int32](), context: ctx) }

    var perm: MetalArray<Int32>? = nil                  // nil means "the identity so far"
    for k in columns.indices.reversed() {
        let desc = descending.isEmpty ? false : descending[k]
        // The first pass sees the column as it is; later ones see it in the order the previous passes left.
        let keys = try perm.map { try columns[k].take($0) } ?? columns[k]
        let idx = try keys.argsortIndices(descending: desc)
        perm = try perm.map { try $0.take(idx) } ?? idx
    }
    return perm!
}

extension AnyMetalArray {
    /// Stable argsort of whichever concrete array this is, nulls last (`MetalArray.argsort`).
    ///
    /// Booleans go through their unpacked byte form and temporal columns through their integer storage.
    /// Strings, binary and dictionary-encoded columns have no order-preserving GPU key yet, so they throw.
    public func argsortIndices(descending: Bool = false) throws -> MetalArray<Int32> {
        switch self {
        case .int8(let a): return try a.argsort(descending: descending)
        case .uint8(let a): return try a.argsort(descending: descending)
        case .int16(let a): return try a.argsort(descending: descending)
        case .uint16(let a): return try a.argsort(descending: descending)
        case .int32(let a): return try a.argsort(descending: descending)
        case .uint32(let a): return try a.argsort(descending: descending)
        case .int64(let a): return try a.argsort(descending: descending)
        case .uint64(let a): return try a.argsort(descending: descending)
        case .float32(let a): return try a.argsort(descending: descending)
        case .float64(let a): return try a.argsort(descending: descending)
        case .boolean(let a): return try a.toUInt8Array().argsort(descending: descending)
        case .temporal(let t):
            switch t.storage {
            case .int32(let a): return try a.argsort(descending: descending)
            case .int64(let a): return try a.argsort(descending: descending)
            }
        case .string, .binary, .dictionary, .decimal, .list, .structure, .map, .union:
            throw ArrowMetalError.unsupportedType("sort by \(arrowFormat) is not implemented")
        // float16 sorts through the float32 widening; the rest have no order-preserving GPU key.
        case .float16(let a): return try a.toFloat32().argsort(descending: descending)
        case .smallDecimal(let s):
            switch s.storage {
            case .int32(let a): return try a.argsort(descending: descending)
            case .int64(let a): return try a.argsort(descending: descending)
            }
        case .extended(let a): return try a.storage.argsortIndices(descending: descending)
        case .null, .interval, .fixedBinary:
            throw ArrowMetalError.unsupportedType("sort by \(arrowFormat) is not implemented")
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

extension MetalArray {
    /// Arrow `partition_nth_indices`: indices arranged so that the element at position `n` is the one
    /// that would be there in sorted order, everything before it no greater and everything after it no
    /// smaller.
    ///
    /// Implemented as a full `argsort` for now — a complete order trivially satisfies the partition —
    /// so it costs a sort rather than the O(length) a selection algorithm would. The signature is the
    /// one a real partition would have, so callers do not change when the kernel does.
    public func partitionNthIndices(_ n: Int) throws -> MetalArray<Int32> {
        guard n >= 0, n <= length else {
            throw ArrowMetalError.invalidArrowArray("partition index \(n) is outside 0...\(length)")
        }
        return try argsort()
    }
}

extension MetalRecordBatch {
    /// Sorts every column by several keys at once, most significant first (stable, nulls last in every
    /// key). `batch.sorted(by: [("region", false), ("revenue", true)])` orders by region ascending and
    /// breaks ties by revenue descending.
    public func sorted(by keys: [(column: String, descending: Bool)]) throws -> MetalRecordBatch {
        guard !keys.isEmpty else { return self }
        var cols: [AnyMetalArray] = []
        var desc: [Bool] = []
        for k in keys {
            guard let c = self[k.column] else { throw ArrowMetalError.invalidArrowArray("no column named \(k.column)") }
            cols.append(c)
            desc.append(k.descending)
        }
        return try take(try lexsortIndices(cols, descending: desc))
    }
}
