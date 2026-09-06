import Foundation
import Metal

// The rest of the join matrix, built on the single-key GPU hash join of `Kernels/Join.swift`.
//
// `hashJoin` does inner and left over one int32 or int64 key. Everything else a query engine needs —
// multi-column keys, utf8 keys, right / full outer / semi / anti, and the as-of join — is here.
//
// ## How the keys are made joinable
//
// A join needs an equality that is exact, not probabilistic. Rather than hashing wide or string keys
// into 64 bits and adding a verification pass, the key columns of *both* sides are concatenated and
// handed to `GroupByKeys`, which is exactly the machine that turns arbitrary key columns — utf8,
// floats, temporal values, several columns at once — into dense int32 ids with an injective mapping.
// Two rows get the same id if and only if their keys are equal, so the int32 hash join over those ids
// is the original join, with no collisions to resolve and no verify pass.
//
// It costs one densification pass over `nL + nR` rows. For the common case that needs none — a single
// int32 or int64 key column on both sides — the raw columns go straight to `hashJoin` and nothing is
// densified.
//
// Nulls never match, in either direction and for every join kind, which is Arrow's and Polars' rule
// (SQL's too). `GroupByKeys` gives null keys a group of their own, so the ids alone would match them
// up; the per-side "every key column is valid" bitmap is attached to the id array instead, and
// `hashJoin` skips a null key on both the build and the probe side.
//
// ## Which rows come out, and in what order
//
// | kind | rows | order |
// |---|---|---|
// | inner | every matching pair | probe (left) order; a left row's matches are contiguous |
// | left | inner, plus each unmatched left row once with null right columns | left order |
// | right | inner, plus each unmatched right row once with null left columns | right order |
// | full | left, plus each unmatched right row once with null left columns | left order, then the right tail |
// | semi | each left row that has at least one match, once | left order |
// | anti | each left row with no match, once | left order |
//
// Semi, anti and the right tail of a full outer join all need the same primitive: "which rows of one
// side did the join touch". That is one `jx_scatter_flag` dispatch over the index pairs plus a
// `filter` of the row indices — no second hash table.

/// The join kinds the engine supports, spelled as Polars spells them.
public enum JoinHow: String, Sendable, CaseIterable {
    case inner, left, right, full, semi, anti
}

/// As-of match direction.
public enum AsofStrategy: String, Sendable, CaseIterable {
    /// The last build key at or before the probe key.
    case backward
    /// The first build key at or after the probe key.
    case forward
    /// Whichever of the two is closer; a tie goes to the backward one.
    case nearest

    var code: Int { self == .backward ? 0 : (self == .forward ? 1 : 2) }
}

extension AnyMetalArray {
    /// The validity bitmap of whichever concrete array this is, when it has one.
    var validityBuffer: MetalArrowBuffer? {
        switch self {
        case .int8(let a): return a.validity
        case .uint8(let a): return a.validity
        case .int16(let a): return a.validity
        case .uint16(let a): return a.validity
        case .int32(let a): return a.validity
        case .uint32(let a): return a.validity
        case .int64(let a): return a.validity
        case .uint64(let a): return a.validity
        case .float32(let a): return a.validity
        case .float64(let a): return a.validity
        case .boolean(let a): return a.validity
        case .string(let a), .binary(let a): return a.validity
        case .temporal(let a):
            switch a.storage { case .int32(let v): return v.validity; case .int64(let v): return v.validity }
        case .decimal(let a): return a.validity
        case .dictionary(let codes, _): return codes.validity
        case .extended(let e): return e.storage.validityBuffer
        default: return nil
        }
    }
}

/// A join's index pairs. Either side may carry nulls: a null left index means "no left row"
/// (the right tail of a full outer join), a null right index "no right row".
public struct JoinIndexPairs {
    public var left: MetalArray<Int32>
    public var right: MetalArray<Int32>
    public var length: Int { left.length }
}

public enum JoinExtra {

    // MARK: - Key preparation

    /// Both sides' key columns as one pair of int32 id arrays whose equality is the join's equality.
    ///
    /// Returns nil when the keys already are a single int32 or int64 column on both sides, in which
    /// case the caller joins them directly and pays no densification.
    static func densify(left: [AnyMetalArray], right: [AnyMetalArray],
                        _ ctx: MetalContext) throws -> (MetalArray<Int32>, MetalArray<Int32>) {
        let nL = left.first?.length ?? 0
        let nR = right.first?.length ?? 0
        var combined: [AnyMetalArray] = []
        for j in left.indices { combined.append(try concatMetalArrays([left[j], right[j]])) }
        let gk = try GroupByKeys(columns: combined)
        let ids = gk.ids
        try ctx.flush()

        func side(_ offset: Int, _ n: Int, _ cols: [AnyMetalArray]) throws -> MetalArray<Int32> {
            let values = n == 0
                ? try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
                : MetalArrowBuffer(mtl: ids.values.mtl, byteCount: n * 4,
                                   offset: ids.values.offset + offset * 4, keepAlive: ids.values)
            var validity: MetalArrowBuffer? = nil
            for c in cols where c.nullCount > 0 {
                guard let v = c.validityBuffer else { continue }
                validity = try BitmapOps.combineValidity(ctx, validity, v, bits: n)
            }
            let a = MetalArray<Int32>(length: n, nullCount: 0, validity: validity, values: values, context: ctx)
            a.recomputeNullCount()
            return a
        }
        return (try side(0, nL, left), try side(nL, nR, right))
    }

    /// True when the two sides can go straight into `hashJoin` with no densification.
    static func directKeys(_ l: [AnyMetalArray], _ r: [AnyMetalArray]) -> Bool {
        guard l.count == 1, r.count == 1 else { return false }
        switch (l[0], r[0]) {
        case (.int32, .int32), (.int64, .int64): return true
        default: return false
        }
    }

    static func rawHashJoin(_ l: AnyMetalArray, _ r: AnyMetalArray, kind: JoinKind)
        throws -> (MetalArray<Int32>, MetalArray<Int32>) {
        switch (l, r) {
        case (.int32(let a), .int32(let b)): return try hashJoin(left: a, right: b, kind: kind)
        case (.int64(let a), .int64(let b)): return try hashJoin(left: a, right: b, kind: kind)
        default: throw ArrowMetalError.unsupportedType("join keys \(l.arrowFormat) / \(r.arrowFormat)")
        }
    }

    // MARK: - Match flags

    /// A boolean, one per row of a side, saying whether any index pair referenced that row.
    static func matchedFlags(indices: MetalArray<Int32>, rows: Int, _ ctx: MetalContext) throws -> MetalBooleanArray {
        let bytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(rows, 1), context: ctx)
        let n = indices.length
        if n > 0 && rows > 0 {
            let pso = try Dispatch.pipeline(ctx, family: "joinextra", source: JoinExtraSource.scatter,
                                            function: "jx_scatter_flag", type: "flag")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(indices.values.mtl, offset: indices.values.offset, index: 0)
                let v = indices.validity ?? indices.values
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setUInt(enc, indices.validity == nil ? 0 : 1, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                Dispatch.setUInt(enc, rows, index: 4)
                enc.setBuffer(bytes.mtl, offset: bytes.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(indices)
            ctx.retainUntilFlush(bytes)
        }
        let bits = try BitmapOps.packBits(ctx, bytes: bytes, bits: Swift.max(rows, 1))
        return MetalBooleanArray(length: rows, nullCount: 0, validity: nil, values: bits, context: ctx)
    }

    /// An all-null int32 index column of `n` rows: the "there is no row on that side" placeholder.
    static func nullIndices(_ n: Int, _ ctx: MetalContext) throws -> MetalArray<Int32> {
        let values = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 1), context: ctx)
        let validity = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), context: ctx)
        return MetalArray<Int32>(length: n, nullCount: n, validity: validity, values: values, context: ctx)
    }

    static func concatIndices(_ a: MetalArray<Int32>, _ b: MetalArray<Int32>) throws -> MetalArray<Int32> {
        guard case .int32(let m) = try concatMetalArrays([.int32(a), .int32(b)]) else {
            throw ArrowMetalError.invalidArrowArray("join: index concatenation lost its type")
        }
        return m
    }

    // MARK: - The join itself

    /// Index pairs for any join kind over any number of key columns of any joinable type.
    public static func indices(leftKeys: [AnyMetalArray], rightKeys: [AnyMetalArray],
                               how: JoinHow) throws -> JoinIndexPairs {
        guard !leftKeys.isEmpty, leftKeys.count == rightKeys.count else {
            throw ArrowMetalError.invalidArrowArray("join: \(leftKeys.count) left keys against \(rightKeys.count) right keys")
        }
        let ctx = leftKeys[0].metalContext
        let nL = leftKeys[0].length, nR = rightKeys[0].length
        for c in leftKeys where c.length != nL { throw ArrowMetalError.lengthMismatch(nL, c.length) }
        for c in rightKeys where c.length != nR { throw ArrowMetalError.lengthMismatch(nR, c.length) }

        // Right joins probe the right side, so the pairs come out in right order; the sides are
        // swapped back when the pairs are returned.
        let swapped = (how == .right)
        var lk = leftKeys, rk = rightKeys
        if swapped { swap(&lk, &rk) }

        let (a, b): (AnyMetalArray, AnyMetalArray)
        if directKeys(lk, rk) {
            (a, b) = (lk[0], rk[0])
        } else {
            let (x, y) = try densify(left: lk, right: rk, ctx)
            (a, b) = (.int32(x), .int32(y))
        }

        switch how {
        case .inner:
            let (l, r) = try rawHashJoin(a, b, kind: .inner)
            return JoinIndexPairs(left: l, right: r)
        case .left:
            let (l, r) = try rawHashJoin(a, b, kind: .left)
            return JoinIndexPairs(left: l, right: r)
        case .right:
            let (l, r) = try rawHashJoin(a, b, kind: .left)   // probes the right side
            return JoinIndexPairs(left: r, right: l)
        case .semi, .anti:
            let (l, _) = try rawHashJoin(a, b, kind: .inner)
            let flags = try matchedFlags(indices: l, rows: nL, ctx)
            let mask = how == .semi ? flags : try flags.not()
            let rows = try GroupByKeys.rowIndices(nL, ctx)
            let kept = try rows.filter(mask)
            return JoinIndexPairs(left: kept, right: try nullIndices(kept.length, ctx))
        case .full:
            let (l, r) = try rawHashJoin(a, b, kind: .left)
            let flags = try matchedFlags(indices: r, rows: nR, ctx)
            let unmatched = try GroupByKeys.rowIndices(nR, ctx).filter(try flags.not())
            if unmatched.length == 0 { return JoinIndexPairs(left: l, right: r) }
            let leftAll = try concatIndices(l, try nullIndices(unmatched.length, ctx))
            let rightAll = try concatIndices(r, unmatched)
            return JoinIndexPairs(left: leftAll, right: rightAll)
        }
    }
}

// MARK: - Record batch level

extension MetalRecordBatch {
    /// Suffixes a right-side column name so it does not collide with a left-side one.
    static func uniqueName(_ name: String, taken: [String], suffix: String) -> String {
        guard taken.contains(name) else { return name }
        var candidate = name + suffix
        var k = 2
        while taken.contains(candidate) { candidate = "\(name)\(suffix)\(k)"; k += 1 }
        return candidate
    }

    /// Equi-join over any number of key columns of any joinable type, in every join kind.
    ///
    /// The result carries this batch's columns followed by `other`'s, minus the right key columns when
    /// `coalesceKeys` is set (Polars' behaviour when the key names match on both sides). Semi and anti
    /// joins return only this batch's columns.
    public func joined(_ other: MetalRecordBatch, leftOn: [String], rightOn: [String],
                       how: JoinHow, suffix: String = "_right", coalesceKeys: Bool = true) throws -> MetalRecordBatch {
        guard leftOn.count == rightOn.count, !leftOn.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("join: \(leftOn.count) left keys against \(rightOn.count) right keys")
        }
        var lk: [AnyMetalArray] = [], rk: [AnyMetalArray] = []
        for n in leftOn {
            guard let c = self[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
            lk.append(c)
        }
        for n in rightOn {
            guard let c = other[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
            rk.append(c)
        }
        let pairs = try JoinExtra.indices(leftKeys: lk, rightKeys: rk, how: how)

        if how == .semi || how == .anti {
            return try MetalRecordBatch(names: names, columns: try columns.map { try $0.take(pairs.left) })
        }
        var outNames = names
        var outCols = try columns.map { try $0.take(pairs.left) }
        // A full or right join's left key column is null on rows that only exist on the right; the
        // key value is on the right side, so fill it in from there.
        if how == .full || how == .right {
            for (j, n) in leftOn.enumerated() {
                guard let i = outNames.firstIndex(of: n) else { continue }
                outCols[i] = try coalesceColumns(outCols[i], try rk[j].take(pairs.right))
            }
        }
        for (i, name) in other.names.enumerated() {
            if coalesceKeys, let j = rightOn.firstIndex(of: name), leftOn[j] == name { continue }
            outNames.append(MetalRecordBatch.uniqueName(name, taken: outNames, suffix: suffix))
            outCols.append(try other.columns[i].take(pairs.right))
        }
        return try MetalRecordBatch(names: outNames, columns: outCols)
    }

    /// Single-key convenience.
    public func joined(_ other: MetalRecordBatch, on: String, how: JoinHow) throws -> MetalRecordBatch {
        try joined(other, leftOn: [on], rightOn: [on], how: how)
    }
}

/// `a` where valid, `b` elsewhere — the key column of a full outer join.
func coalesceColumns(_ a: AnyMetalArray, _ b: AnyMetalArray) throws -> AnyMetalArray {
    if a.nullCount == 0 { return a }
    let ctx = a.metalContext
    let valid = try a.isValidMask(ctx)
    switch (a, b) {
    case (.int32(let x), .int32(let y)): return .int32(try valid.ifElse(x, y))
    case (.int64(let x), .int64(let y)): return .int64(try valid.ifElse(x, y))
    case (.int8(let x), .int8(let y)): return .int8(try valid.ifElse(x, y))
    case (.int16(let x), .int16(let y)): return .int16(try valid.ifElse(x, y))
    case (.uint8(let x), .uint8(let y)): return .uint8(try valid.ifElse(x, y))
    case (.uint16(let x), .uint16(let y)): return .uint16(try valid.ifElse(x, y))
    case (.uint32(let x), .uint32(let y)): return .uint32(try valid.ifElse(x, y))
    case (.uint64(let x), .uint64(let y)): return .uint64(try valid.ifElse(x, y))
    case (.float32(let x), .float32(let y)): return .float32(try valid.ifElse(x, y))
    case (.float64(let x), .float64(let y)): return .float64(try valid.ifElse(x, y))
    case (.boolean(let x), .boolean(let y)): return .boolean(try valid.ifElse(x, y))
    default:
        // utf8, temporal and the rest have no elementwise select kernel; pick per row on the CPU with
        // a gather, which is what the outer-join key column of a string join needs.
        return try coalesceByGather(a, b, ctx)
    }
}

extension AnyMetalArray {
    /// A boolean column that is true exactly where this column is valid.
    func isValidMask(_ ctx: MetalContext) throws -> MetalBooleanArray {
        guard let v = validityBuffer, nullCount > 0 else {
            let all = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: length), 1), zeroed: false, context: ctx)
            memset(all.mutableContents, 0xFF, all.byteCount)
            return MetalBooleanArray(length: length, nullCount: 0, validity: nil, values: all, context: ctx)
        }
        return MetalBooleanArray(length: length, nullCount: 0, validity: nil, values: v, context: ctx)
    }
}

/// Row-wise choice between two columns of a type with no elementwise select kernel: build the gather
/// index on the CPU (`i` from `a`, `n + i` from `b`) and take from their concatenation.
private func coalesceByGather(_ a: AnyMetalArray, _ b: AnyMetalArray, _ ctx: MetalContext) throws -> AnyMetalArray {
    try ctx.flush()
    let n = a.length
    let merged = try concatMetalArrays([a, b])
    var idx = [Int32](repeating: 0, count: n)
    if let v = a.validityBuffer, a.nullCount > 0 {
        let p = v.typed(UInt8.self)
        for i in 0..<n { idx[i] = Bitmap.isSet(p, i) ? Int32(i) : Int32(n + i) }
    } else {
        for i in 0..<n { idx[i] = Int32(i) }
    }
    return try merged.take(try MetalArray<Int32>(idx, context: ctx))
}

// MARK: - As-of join

extension MetalRecordBatch {
    /// Arrow / Polars `join_asof`: for every row of this batch, the nearest row of `other` on an
    /// ordered key, optionally within matching partitions (`by`) and within a `tolerance`.
    ///
    /// The key must be an integer or temporal column on both sides (the usual case is a timestamp).
    /// The build side is sorted on the GPU by `(partition, key)`, its null keys dropped, and every
    /// probe row then does two binary searches — one for its partition's range, one for the key —
    /// which is `log2` dependent loads and no hash table.
    ///
    /// Rows of this batch keep their order and all of them come out (an as-of join is a left join):
    /// a row with no match gets nulls for `other`'s columns.
    public func joinedAsof(_ other: MetalRecordBatch, leftOn: String, rightOn: String,
                           by: [String] = [], byRight: [String]? = nil,
                           strategy: AsofStrategy = .backward, tolerance: Int64? = nil,
                           suffix: String = "_right") throws -> MetalRecordBatch {
        guard let lkCol = self[leftOn] else { throw ArrowMetalError.invalidArrowArray("no column named \(leftOn)") }
        guard let rkCol = other[rightOn] else { throw ArrowMetalError.invalidArrowArray("no column named \(rightOn)") }
        let ctx = lkCol.metalContext
        let byR = byRight ?? by
        guard by.count == byR.count else {
            throw ArrowMetalError.invalidArrowArray("join_asof: \(by.count) left `by` columns against \(byR.count) right ones")
        }
        let nL = length, nR = other.length

        let lk = try asofKey(lkCol, leftOn)
        let rkAll = try asofKey(rkCol, rightOn)

        // Partition ids shared by both sides, and the per-side null-key mask.
        var lPart: MetalArray<Int32>? = nil, rPartAll: MetalArray<Int32>? = nil
        if !by.isEmpty {
            var lc: [AnyMetalArray] = [], rc: [AnyMetalArray] = []
            for n in by {
                guard let c = self[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
                lc.append(c)
            }
            for n in byR {
                guard let c = other[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
                rc.append(c)
            }
            let (l, r) = try JoinExtra.densify(left: lc, right: rc, ctx)
            lPart = l; rPartAll = r
        }

        // Drop the build rows whose key (or partition) is null: they can never be the answer.
        let dropping = rkAll.nullCount > 0 || (rPartAll?.nullCount ?? 0) > 0
        let rowsR = try GroupByKeys.rowIndices(nR, ctx)
        var keptRows = rowsR
        if dropping {
            var keepR = try rkAll.isValid()
            if let rp = rPartAll, rp.nullCount > 0 { keepR = try keepR.and(try rp.isValid()) }
            keptRows = try rowsR.filter(keepR)
        }
        let rk = dropping ? try rkAll.take(keptRows) : rkAll
        let rPart = rPartAll.map { p in dropping ? (try? p.take(keptRows)) ?? p : p }

        // Sort the build side by (partition, key).
        var sortCols: [AnyMetalArray] = []
        if let rp = rPart { sortCols.append(.int32(rp)) }
        sortCols.append(.int64(rk))
        let order = try lexsortIndices(sortCols)
        let rkSorted = try rk.take(order)
        let rPartSorted = try rPart.map { try $0.take(order) }
        let buildRows = try keptRows.take(order)          // original row index of each sorted build row
        let m = rkSorted.length

        // Probe.
        let outIdx = try MetalArrowBuffer.allocate(byteCount: Swift.max(nL * 4, 1), zeroed: false, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(nL, 1), context: ctx)
        if nL > 0 {
            let pso = try Dispatch.pipeline(ctx, family: "joinextra", source: JoinExtraSource.asof,
                                            function: "jx_asof_probe", type: "asof")
            let partBuf = rPartSorted?.values ?? outIdx
            let lpBuf = lPart?.values ?? outIdx
            var lValidity = lk.validity
            if let p = lPart, p.validity != nil {
                lValidity = try BitmapOps.combineValidity(ctx, lValidity, p.validity, bits: nL)
            }
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(lk.values.mtl, offset: lk.values.offset, index: 0)
                let v = lValidity ?? lk.values
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setUInt(enc, lValidity == nil ? 0 : 1, index: 2)
                enc.setBuffer(lpBuf.mtl, offset: lpBuf.offset, index: 3)
                Dispatch.setUInt(enc, lPart == nil ? 0 : 1, index: 4)
                Dispatch.setLength(enc, nL, nil, index: 5)
                enc.setBuffer(rkSorted.values.mtl, offset: rkSorted.values.offset, index: 6)
                enc.setBuffer(partBuf.mtl, offset: partBuf.offset, index: 7)
                Dispatch.setUInt(enc, m, index: 8)
                Dispatch.setUInt(enc, strategy.code, index: 9)
                Dispatch.setScalar(enc, tolerance ?? 0, index: 10)
                Dispatch.setUInt(enc, tolerance == nil ? 0 : 1, index: 11)
                enc.setBuffer(outIdx.mtl, offset: outIdx.offset, index: 12)
                enc.setBuffer(outValid.mtl, offset: outValid.offset, index: 13)
                Dispatch.dispatch1D(enc, pso, count: nL)
            }
            let live: [AnyObject] = [lk, rkSorted, buildRows]
            for o in live { ctx.retainUntilFlush(o) }
            ctx.retainUntilFlush(outIdx); ctx.retainUntilFlush(outValid)
        }
        let bits = try BitmapOps.packBits(ctx, bytes: outValid, bits: Swift.max(nL, 1))
        let sortedPick = MetalArray<Int32>(length: nL, nullCount: 0, validity: bits, values: outIdx, context: ctx)
        sortedPick.recomputeNullCount()
        // The kernel answers in sorted-build coordinates; map back to the caller's row numbers.
        let rightIndices = m == 0 ? try JoinExtra.nullIndices(nL, ctx) : try buildRows.take(sortedPick)

        // The right key and the right `by` columns repeat what the left side already carries when the
        // names match, so they are dropped, as Polars' `join_asof` does.
        var dropped = Set<String>()
        if rightOn == leftOn { dropped.insert(rightOn) }
        for (i, n) in byR.enumerated() where n == by[i] { dropped.insert(n) }
        var outNames = names
        var outCols = columns
        for (i, name) in other.names.enumerated() where !dropped.contains(name) {
            outNames.append(MetalRecordBatch.uniqueName(name, taken: outNames, suffix: suffix))
            outCols.append(try other.columns[i].take(rightIndices))
        }
        return try MetalRecordBatch(names: outNames, columns: outCols)
    }

    /// The as-of key as int64: integers widen, temporal columns go through their integer storage.
    private func asofKey(_ c: AnyMetalArray, _ name: String) throws -> MetalArray<Int64> {
        switch c {
        case .int64(let a): return a
        case .int32(let a): return try a.cast(to: Int64.self)
        case .int16(let a): return try a.cast(to: Int64.self)
        case .int8(let a): return try a.cast(to: Int64.self)
        case .uint32(let a): return try a.cast(to: Int64.self)
        case .uint16(let a): return try a.cast(to: Int64.self)
        case .uint8(let a): return try a.cast(to: Int64.self)
        case .temporal(let t):
            switch t.storage {
            case .int64(let a): return a
            case .int32(let a): return try a.cast(to: Int64.self)
            }
        case .extended(let e): return try asofKey(e.storage, name)
        default:
            throw ArrowMetalError.unsupportedType("join_asof key \"\(name)\" is \(c.arrowFormat); it must be an integer or temporal column")
        }
    }
}
