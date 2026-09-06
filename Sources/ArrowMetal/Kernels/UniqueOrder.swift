import Foundation
import Metal

/// Arrow's order for `unique`, `value_counts` and `dictionary_encode`: **order of first appearance**,
/// on the GPU.
///
/// `Kernels/Unique.swift` produces the distinct values ascending, because it gets them out of a sort.
/// Arrow instead returns them in the order they first show up in the input, which is a different
/// permutation of the same set — so all that is needed is the permutation, and the sorted pass already
/// has everything to compute it:
///
/// 1. `dictionaryEncode()` gives a dense code per row (its rank among the sorted distinct values) and
///    the sorted distinct values themselves. Null rows get a null code.
/// 2. A **group-min of the row index over the codes** (`GroupBy.min`, which skips null keys) gives, for
///    each distinct value, the earliest row that carries it. That is its first appearance.
/// 3. A **stable argsort of those minima** orders the distinct values by first appearance — one radix
///    sort over `unique.count` elements, not over the rows.
/// 4. A `take` moves the values, the counts or the codes into that order.
///
/// So first-appearance order costs one extra group-min over the rows plus a sort and a gather over the
/// *distinct* values, which is normally far smaller. Nothing leaves the GPU.
///
/// ## Nulls
///
/// Arrow's `unique` and `value_counts` **keep** the null: it appears once, in the position of the first
/// null row, and `value_counts` reports how many rows were null. This path reproduces that, by giving
/// the nulls a group of their own whose "first appearance" is the first null row and then gathering the
/// representative values straight out of the input — a `take` of a null row is a null, so the null
/// falls out of the same gather as everything else. `dictionary_encode` drops it, as Arrow does: a null
/// row gets a null code and the dictionary holds only values.
///
/// The `.sorted` order keeps the older behaviour exactly, nulls dropped, and stays the cheaper of the two.
enum UniqueOrderSource {
    /// A straight typed copy, used to grow an array by one element without a host-side memcpy of the
    /// bulk. The extra element is written by the host into the shared buffer afterwards.
    static func source(T: String) -> String { KernelSource.prelude + """

    kernel void uo_copy(device const \(T)* src [[buffer(0)]], device const uint* nPtr [[buffer(1)]],
                        device \(T)* dst [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < *nPtr) dst[i] = src[i];
    }
    """ }
}

enum UniqueOrder {
    /// `a` with `extra` appended. The bulk copy runs on the GPU; only the one new element is written
    /// by the host.
    static func appending<K: ArrowPrimitive>(_ a: MetalArray<K>, _ extra: K) throws -> MetalArray<K> {
        let ctx = a.context, n = a.length
        let out = try MetalArrowBuffer.allocate(byteCount: (n + 1) * K.byteWidth, zeroed: true, context: ctx)
        if n > 0 {
            let src = UniqueOrderSource.source(T: K.mslType)
            let pso = try Dispatch.pipeline(ctx, family: "unique-order", source: src, function: "uo_copy",
                                            type: K.mslType)
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBuffer(out.mtl, offset: out.offset, index: 2)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            try ctx.syncPoint()
        }
        out.mutableTyped(K.self)[n] = extra
        return MetalArray<K>(length: n + 1, nullCount: 0, validity: nil, values: out, context: ctx)
    }

    /// The row index of the first null, or nil when there is none.
    static func firstNullRow(length: Int, nullCount: Int, validity: MetalArrowBuffer?) -> Int? {
        guard nullCount > 0, let validity else { return nil }
        let bm = validity.typed(UInt8.self)
        for i in 0..<length where !Bitmap.isSet(bm, i) { return i }
        return nil
    }
}

extension MetalArray {

    /// The first-appearance permutation and the pieces every caller needs, computed once.
    ///
    /// `order` reorders the *sorted* distinct values into first-appearance order, `rows` names the
    /// original row that represents each of them in that order, `codes` is the per-row sorted code and
    /// `sorted` the sorted distinct values. Returns nil when there is nothing to report at all.
    private struct FirstAppearance {
        let order: MetalArray<Int32>
        let rows: MetalArray<Int32>
        let codes: MetalArray<Int32>
        let sorted: MetalArray<T>
    }

    private func firstAppearance(includeNulls: Bool) throws -> FirstAppearance? {
        let (codes, sortedUnique) = try dictionaryEncode()
        let u = sortedUnique.length
        let nullRow = includeNulls ? UniqueOrder.firstNullRow(length: length, nullCount: nullCount,
                                                             validity: validity) : nil
        guard u > 0 || nullRow != nil else { return nil }
        var firsts: MetalArray<Int32>
        if u > 0 {
            let rows = try MetalArray<Int32>.iota(length, context: context)
            firsts = try (try GroupBy(keys: codes, keyCount: u)).min(rows)
        } else {
            firsts = try MetalArray<Int32>([Int32](), context: context)
        }
        if let nullRow { firsts = try UniqueOrder.appending(firsts, Int32(nullRow)) }
        // The minima are distinct row indices, so the order is total and stability is moot; the radix
        // argsort is used because it is the one that already exists.
        let order = try firsts.argsort()
        return FirstAppearance(order: order, rows: try firsts.take(order), codes: codes, sorted: sortedUnique)
    }

    /// Arrow `unique` with the order spelled out.
    ///
    /// `.sorted` is the ascending pass in `Kernels/Unique.swift`, nulls dropped. `.firstAppearance` is
    /// Arrow's own order and, like Arrow, keeps the null as one entry at the position of the first null
    /// row.
    public func unique(order: ValueOrder) throws -> MetalArray<T> {
        guard order == .firstAppearance else { return try unique() }
        guard let fa = try firstAppearance(includeNulls: true) else {
            return try MetalArray<T>([T](), context: context)
        }
        // Gathering the representatives straight out of the input reproduces the null for free: a `take`
        // of a null row is a null row.
        return try take(fa.rows)
    }

    /// Arrow `value_counts` with the order spelled out. `.firstAppearance` keeps the null group and
    /// reports its row count, as Arrow does.
    public func valueCounts(order: ValueOrder) throws -> (values: MetalArray<T>, counts: MetalArray<Int64>) {
        guard order == .firstAppearance else { return try valueCounts() }
        guard let fa = try firstAppearance(includeNulls: true) else {
            return (try MetalArray<T>([T](), context: context), try MetalArray<Int64>([Int64](), context: context))
        }
        let u = fa.sorted.length
        var counts: MetalArray<Int64>
        if u > 0 {
            counts = try (try GroupBy(keys: fa.codes, keyCount: u)).count()
        } else {
            counts = try MetalArray<Int64>([Int64](), context: context)
        }
        if UniqueOrder.firstNullRow(length: length, nullCount: nullCount, validity: validity) != nil {
            counts = try UniqueOrder.appending(counts, Int64(nullCount))
        }
        return (try take(fa.rows), try counts.take(fa.order))
    }

    /// Arrow `dictionary_encode` with the order spelled out. The dictionary never holds a null in either
    /// order; a null row gets a null code.
    public func dictionaryEncode(order: ValueOrder)
        throws -> (codes: MetalArray<Int32>, unique: MetalArray<T>) {
        guard order == .firstAppearance else { return try dictionaryEncode() }
        guard let fa = try firstAppearance(includeNulls: false) else { return try dictionaryEncode() }
        let dictionary = try fa.sorted.take(fa.order)
        // `order[p]` is the sorted code that belongs at position p, so its inverse maps a sorted code to
        // its first-appearance position — exactly the recode the row codes need.
        let recode = try fa.order.inversePermutation(maxIndex: Int64(fa.order.length - 1))
        return (try recode.take(fa.codes), dictionary)
    }
}

extension MetalStringArray {
    /// Arrow `unique` over utf8, in whichever order.
    ///
    /// The GPU string dictionary (`Kernels/StringDictionary.swift`) already produces its distinct values
    /// in first-seen order, so `.firstAppearance` is simply that result. Nulls are dropped in both
    /// orders, which is the one place this differs from Arrow's `unique` — the string dictionary has no
    /// slot for one.
    public func unique(order: ValueOrder = .firstAppearance) throws -> MetalStringArray {
        let (_, values) = try dictionaryEncode()
        guard order == .sorted else { return values }
        // There is no order-preserving GPU key for utf8, so the (small) distinct set is ordered here.
        let sorted = values.toArray().sorted { ($0 ?? "") < ($1 ?? "") }
        return try MetalStringArray(sorted, context: context)
    }

    /// Arrow `value_counts` over utf8: the distinct strings and how many rows carry each.
    public func valueCounts(order: ValueOrder = .firstAppearance)
        throws -> (values: MetalStringArray, counts: MetalArray<Int64>) {
        let (codes, values) = try dictionaryEncode()
        let u = values.length
        guard u > 0 else { return (values, try MetalArray<Int64>([Int64](), context: context)) }
        let counts = try (try GroupBy(keys: codes, keyCount: u)).count()
        guard order == .sorted else { return (values, counts) }
        // The distinct strings are ordered on the host — no order-preserving GPU key for utf8 — and the
        // counts follow through the existing gather.
        let strings = values.toArray()
        let positions = strings.indices.sorted { (strings[$0] ?? "") < (strings[$1] ?? "") }
        let perm = try MetalArray<Int32>(positions.map { Int32($0) }, context: context)
        return (try values.take(perm), try counts.take(perm))
    }
}
