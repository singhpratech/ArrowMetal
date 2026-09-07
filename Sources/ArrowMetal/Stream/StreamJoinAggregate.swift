import Foundation
import Metal

// Fused broadcast join + aggregate.
//
// A `join -> sum` used to be two operators with a whole joined table between them: every probe batch
// ran the GPU hash join, gathered *every* column of both sides into a new record batch, wrote it to a
// sink, and the caller summed the rows it got back. The joined rows are the largest thing in that
// pipeline and none of them is the answer.
//
// Here the probe and the accumulate are **one kernel**. One thread per probe row walks its bucket
// chain in the build table and, for each match, reads the aggregate's value — from the probe row it
// is standing on, or from the build row the chain points at — and folds it into a thread-local
// accumulator. A threadgroup tree reduction turns 256 of those into one partial, and a second small
// kernel folds the partials into a **device-resident** accumulator pair that lives for the whole
// scan. Nothing proportional to the number of matches is ever allocated, nothing is gathered, and
// nothing crosses to the host until `finish()` reads sixteen bytes per aggregate.
//
// The build side's hash table is built **once**, at construction, not once per probe batch as
// `MetalRecordBatch.join` does — it is the same table for every batch of the scan.
//
// The group-by form (`join(...).groupBy(...)`) keeps the join's index pairs, because a per-group
// accumulator needs a hash table probed by *rows*, which the resident group table cannot be. It
// gathers only the key columns and the aggregated value columns for the matched pairs — never the
// joined batch — and hands that pair of columns to the ordinary streaming group-by, whose global
// table is already resident.
//
// Semantics are the inner join's, and match Polars and DuckDB: duplicate build keys multiply rows,
// a null key on either side never matches, and a probe row with no match contributes nothing. A
// **left** join followed by an aggregate is refused with an error rather than answered wrongly.

// MARK: - Generated MSL

/// Metal for the fused probe-and-accumulate kernel and its cross-threadgroup fold.
///
/// The kernel is generated per (key type, aggregate list): the accumulators are unrolled into
/// registers with the right combine for each one, so there is no per-aggregate branch in the loop
/// and no array indirection. Every accumulator is carried as a raw `ulong` — a signed sum, an
/// unsigned sum, or an IEEE-754 binary64 bit pattern that `d_add` folds correctly rounded, since
/// Metal has neither a `double` type nor a 64-bit atomic.
enum JoinAggregateSource {
    /// How an accumulator's 64 bits are read.
    enum Kind: String {
        case int64, uint64, double
    }

    /// One fused accumulator: which side its value comes from, how to combine it, and how to read it.
    struct Slot {
        /// The value column lives on the build side (gathered through the chain pointer).
        var fromBuild: Bool
        /// sum / count / min / max. `mean` arrives here as `sum` and is divided at the end.
        var op: StreamAggregate.Op
        var kind: Kind
        /// MSL element type of the value buffer; "" for `count` (which reads no value).
        var elementType: String
        /// False for `count` — no value buffer is read at all.
        var readsValue: Bool

        /// Part of the pipeline cache key, so a different aggregate list gets a different pipeline.
        var signature: String {
            "\(fromBuild ? "b" : "p")\(op.rawValue)\(kind.rawValue)\(elementType)\(readsValue ? "v" : "")"
        }
    }

    /// Merge of two non-empty accumulators of this slot's shape.
    private static func merge(_ s: Slot, _ a: String, _ b: String) -> String {
        switch s.op {
        case .count:
            return a                                        // the value half is unused
        case .sum, .mean:
            return s.kind == .double ? "d_add(\(a), \(b))" : "(\(a) + \(b))"
        case .min:
            switch s.kind {
            case .int64: return "(((long)\(b) < (long)\(a)) ? \(b) : \(a))"
            case .uint64: return "((\(b) < \(a)) ? \(b) : \(a))"
            case .double: return "((d_key((long)\(b)) < d_key((long)\(a))) ? \(b) : \(a))"
            }
        case .max:
            switch s.kind {
            case .int64: return "(((long)\(b) > (long)\(a)) ? \(b) : \(a))"
            case .uint64: return "((\(b) > \(a)) ? \(b) : \(a))"
            case .double: return "((d_key((long)\(b)) > d_key((long)\(a))) ? \(b) : \(a))"
            }
        default:
            return a
        }
    }

    /// `dst <- dst (+) src`, where an accumulator with a zero count is the identity. Using the count
    /// as the "is empty" flag keeps `min` / `max` free of a per-type sentinel and makes an all-null
    /// input come out as null rather than as an infinity.
    private static func combine(_ s: Slot, dv: String, dc: String, sv: String, sc: String) -> String {
        """
        if (\(sc) != 0) { \(dv) = (\(dc) == 0) ? (\(sv)) : (\(merge(s, dv, sv))); \(dc) += \(sc); }
        """
    }

    /// Reads one value out of the slot's buffer and widens it into the accumulator's 64 bits.
    private static func load(_ s: Slot, index: Int) -> String {
        let v = "v\(index)[idx]"
        switch s.kind {
        case .double: return s.elementType == "float" ? "d_from_float(\(v))" : "(ulong)\(v)"
        case .int64: return "(ulong)(long)\(v)"
        case .uint64: return "(ulong)\(v)"
        }
    }

    /// NaN is skipped by `min` / `max`, as Arrow does; a `sum` lets it propagate through `d_add`.
    private static func include(_ s: Slot) -> String {
        guard s.kind == .double, s.op == .min || s.op == .max else { return "true" }
        return "!d_isnan((long)vv)"
    }

    /// `KT` is the join key type ("int" or "long"); `slots` the fused accumulators, in order.
    static func source(KT: String, slots: [Slot]) -> String {
        let hash = KT == "long"
            ? "uint lo = (uint)k, hi = (uint)(((ulong)k) >> 32); return jfa_mix(jfa_mix(lo) ^ (hi * 0x9E3779B9u));"
            : "return jfa_mix((uint)k);"
        let width = slots.count * 2

        // Per-slot kernel arguments: values, validity, "has validity". Buffers 9, 10, 11, 12, ...
        var args = ""
        for (a, s) in slots.enumerated() {
            let t = s.readsValue ? s.elementType : KT
            args += """

                                       device const \(t)* v\(a) [[buffer(\(9 + 3 * a))]],
                                       device const uchar* m\(a) [[buffer(\(10 + 3 * a))]],
                                       constant uint& hm\(a) [[buffer(\(11 + 3 * a))]],
            """
        }

        // Per-slot declarations, per-match bodies and threadgroup reductions.
        var decls = "", bodies = "", reduces = "", folds = ""
        for (a, s) in slots.enumerated() {
            decls += "    ulong a\(a) = 0ul; ulong c\(a) = 0ul;\n"
            let idx = s.fromBuild ? "bi" : "i"
            let valid = "!(hm\(a) && !bit_get(m\(a), idx))"
            let vSlot = "partials[tgid * \(width)u + \(2 * a)u]"
            let cSlot = "partials[tgid * \(width)u + \(2 * a + 1)u]"
            let pv = "partials[b * \(width)u + \(2 * a)u]"
            let pc = "partials[b * \(width)u + \(2 * a + 1)u]"
            // Reducing the threadgroup and folding a partial in are the same combine, so the tree
            // reduction below is written once and reused by both kernels.
            let tree = combine(s, dv: "sv[lid]", dc: "sc[lid]", sv: "sv[lid + st]", sc: "sc[lid + st]")
            if !s.readsValue {
                bodies += "                { c\(a) += 1ul; }\n"     // count(*): every matched pair
            } else if s.op == .count {
                bodies += "                { uint idx = \(idx); if (\(valid)) c\(a) += 1ul; }\n"
            } else {
                let fold = combine(s, dv: "a\(a)", dc: "c\(a)", sv: "vv", sc: "1ul")
                bodies += """
                                {
                                    uint idx = \(idx);
                                    if (\(valid)) {
                                        ulong vv = \(load(s, index: a));
                                        if (\(include(s))) { \(fold) }
                                    }
                                }

                """
            }
            reduces += """
                threadgroup_barrier(mem_flags::mem_threadgroup);
                sv[lid] = a\(a); sc[lid] = c\(a);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint st = TG / 2u; st > 0u; st >>= 1u) {
                    if (lid < st) { \(tree) }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lid == 0u) { \(vSlot) = sv[0]; \(cSlot) = sc[0]; }

            """
            folds += """
                {
                    ulong a = 0ul, c = 0ul;
                    for (uint b = lid; b < blocks; b += TG) {
                        ulong pv = \(pv), pc = \(pc);
                        \(combine(s, dv: "a", dc: "c", sv: "pv", sc: "pc"))
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    sv[lid] = a; sc[lid] = c;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint st = TG / 2u; st > 0u; st >>= 1u) {
                        if (lid < st) { \(tree) }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (lid == 0u) {
                        ulong gv = state[\(2 * a)u], gc = state[\(2 * a + 1)u];
                        \(combine(s, dv: "gv", dc: "gc", sv: "sv[0]", sc: "sc[0]"))
                        state[\(2 * a)u] = gv; state[\(2 * a + 1)u] = gc;
                    }
                }

            """
        }

        return KernelSource.prelude + DoubleMath.msl + """

        inline uint jfa_mix(uint h) {
            h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h;
        }
        inline uint jfa_hash(\(KT) k) { \(hash) }

        // Head of the chain of build rows whose key is `k` (0 when the key is absent). The table is
        // the one `hj_build` filled: a slot holds `row + 1`, and the slot's key is `bkeys[slot - 1]`.
        inline uint jfa_find(device const \(KT)* bkeys, device const uint* slots, uint mask, \(KT) k) {
            uint s = jfa_hash(k) & mask;
            for (uint p = 0; p <= mask; p++) {
                uint head = slots[s];
                if (head == 0u) return 0u;
                if (bkeys[head - 1u] == k) return head;
                s = (s + 1u) & mask;
            }
            return 0u;
        }

        // One thread per probe row: walk the chain, fold every match into registers, reduce the
        // threadgroup, write one partial per accumulator per threadgroup.
        kernel void jfa_probe(device const \(KT)* lkeys [[buffer(0)]],
                              device const uchar* lvalidity [[buffer(1)]],
                              constant uint& hasValidity [[buffer(2)]],
                              device const uint* nPtr [[buffer(3)]],
                              constant uint& mask [[buffer(4)]],
                              device const \(KT)* bkeys [[buffer(5)]],
                              device const uint* slots [[buffer(6)]],
                              device const uint* next [[buffer(7)]],
                              device ulong* partials [[buffer(8)]],\(args)
                              uint i [[thread_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint tgid [[threadgroup_position_in_grid]]) {
            threadgroup ulong sv[TG];
            threadgroup ulong sc[TG];
            uint n = *nPtr;
        \(decls)
            if (i < n && !(hasValidity && !bit_get(lvalidity, i))) {
                for (uint r = jfa_find(bkeys, slots, mask, lkeys[i]); r != 0u; r = next[r - 1u]) {
                    uint bi = r - 1u;
        \(bodies)        }
            }
        \(reduces)}

        // One threadgroup: folds this batch's partials into the scan-resident accumulators.
        kernel void jfa_fold(device const ulong* partials [[buffer(0)]],
                             constant uint& blocks [[buffer(1)]],
                             device ulong* state [[buffer(2)]],
                             uint lid [[thread_index_in_threadgroup]]) {
            threadgroup ulong sv[TG];
            threadgroup ulong sc[TG];
        \(folds)}
        """
    }
}

// MARK: - The build side's table, built once

/// The broadcast join's build side as a device-resident open-addressing hash table.
///
/// `MetalRecordBatch.join` builds this table per call, which for a streamed join is per *batch*. The
/// build side does not change across a scan, so it is built here exactly once, with the same
/// `hj_clear` / `hj_build` kernels the one-shot join uses, and every probe batch reads it.
final class BroadcastBuildTable {
    /// "int" or "long": the MSL type of the join key on both sides.
    let keyType: String
    let keys: AnyMetalArray
    let keyValues: MetalArrowBuffer
    let slots: MetalArrowBuffer
    let next: MetalArrowBuffer
    let mask: Int
    let rows: Int
    let context: MetalContext

    init(keyColumn: AnyMetalArray, context ctx: MetalContext) throws {
        let values: MetalArrowBuffer, validity: MetalArrowBuffer?, n: Int
        switch keyColumn {
        case .int32(let a): keyType = "int"; values = a.values; validity = a.validity; n = a.length
        case .int64(let a): keyType = "long"; values = a.values; validity = a.validity; n = a.length
        default:
            throw ArrowMetalError.unsupportedType(
                "join key \(keyColumn.arrowFormat): a fused join key must be int32 or int64 on both sides")
        }
        guard n <= Int(Int32.max) else {
            throw ArrowMetalError.invalidArrowArray("join: build sides above 2^31 rows are not supported")
        }
        keys = keyColumn
        keyValues = values
        rows = n
        context = ctx
        var tableSize = 1
        while tableSize < 2 * Swift.max(1, n) { tableSize <<= 1 }
        mask = tableSize - 1
        slots = try MetalArrowBuffer.allocate(byteCount: tableSize * 4, zeroed: false, context: ctx)
        next = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: ctx)
        let errorFlag = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)

        let src = JoinSource.source(KT: keyType)
        let clearPSO = try Dispatch.pipeline(ctx, family: "join", source: src, function: "hj_clear", type: keyType)
        let buildPSO = try Dispatch.pipeline(ctx, family: "join", source: src, function: "hj_build", type: keyType)
        try ctx.run { enc in
            enc.setComputePipelineState(clearPSO)
            enc.setBuffer(slots.mtl, offset: 0, index: 0)
            Dispatch.setLength(enc, tableSize, nil, index: 1)
            Dispatch.dispatch1D(enc, clearPSO, count: tableSize)
            enc.memoryBarrier(scope: .buffers)
            if n > 0 {
                enc.setComputePipelineState(buildPSO)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                let v = validity ?? values
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                Dispatch.setUInt(enc, mask, index: 4)
                enc.setBuffer(slots.mtl, offset: 0, index: 5)
                enc.setBuffer(next.mtl, offset: 0, index: 6)
                enc.setBuffer(errorFlag.mtl, offset: 0, index: 7)
                Dispatch.dispatch1D(enc, buildPSO, count: n)
            }
        }
        try ctx.syncPoint()
        if withExtendedLifetime(errorFlag, { errorFlag.typed(UInt32.self)[0] }) != 0 {
            throw ArrowMetalError.invalidArrowArray("join: hash table insertion failed")
        }
    }

    /// Binds buffers 0...7, the probe arguments every fused kernel shares.
    func bindProbe(_ enc: MTLComputeCommandEncoder, probeValues: MetalArrowBuffer,
                   probeValidity: MetalArrowBuffer?, rows n: Int) {
        enc.setBuffer(probeValues.mtl, offset: probeValues.offset, index: 0)
        let pv = probeValidity ?? probeValues
        enc.setBuffer(pv.mtl, offset: pv.offset, index: 1)
        Dispatch.setUInt(enc, probeValidity == nil ? 0 : 1, index: 2)
        Dispatch.setLength(enc, n, nil, index: 3)
        Dispatch.setUInt(enc, mask, index: 4)
        enc.setBuffer(keyValues.mtl, offset: keyValues.offset, index: 5)
        enc.setBuffer(slots.mtl, offset: 0, index: 6)
        enc.setBuffer(next.mtl, offset: 0, index: 7)
    }
}

// MARK: - Column plumbing

/// A numeric column reduced to what the fused kernel needs to read it.
struct FusedJoinColumn {
    var values: MetalArrowBuffer
    var validity: MetalArrowBuffer?
    var elementType: String
    var kind: JoinAggregateSource.Kind
    var length: Int
    /// Keeps the array alive for as long as its buffers are bound.
    var owner: AnyObject
}

/// The buffers, MSL element type and accumulator width of a numeric column.
///
/// Float64 arrives as `ulong` bit patterns because Metal has no `double`; float32 is widened into the
/// same software binary64 accumulator, so a fused sum of a float32 column is computed in double.
func fusedJoinColumn(_ a: AnyMetalArray, _ name: String) throws -> FusedJoinColumn {
    func unsupported() -> Error {
        ArrowMetalError.unsupportedType("a fused join aggregate over \(name) (\(a.arrowFormat)): "
                                        + "the value column must be an integer or float")
    }
    func of<T: ArrowPrimitive>(_ c: MetalArray<T>, _ t: String,
                               _ k: JoinAggregateSource.Kind) -> FusedJoinColumn {
        FusedJoinColumn(values: c.values, validity: c.validity, elementType: t, kind: k,
                        length: c.length, owner: c)
    }
    switch a {
    case .int8(let c): return of(c, "char", .int64)
    case .int16(let c): return of(c, "short", .int64)
    case .int32(let c): return of(c, "int", .int64)
    case .int64(let c): return of(c, "long", .int64)
    case .uint8(let c): return of(c, "uchar", .uint64)
    case .uint16(let c): return of(c, "ushort", .uint64)
    case .uint32(let c): return of(c, "uint", .uint64)
    case .uint64(let c): return of(c, "ulong", .uint64)
    case .float32(let c): return of(c, "float", .double)
    case .float64(let c): return of(c, "ulong", .double)
    case .boolean: throw unsupported()
    case .string: throw unsupported()
    case .temporal: throw unsupported()
    case .binary: throw unsupported()
    case .decimal: throw unsupported()
    case .dictionary: throw unsupported()
    case .list: throw unsupported()
    case .structure: throw unsupported()
    case .map: throw unsupported()
    case .union: throw unsupported()
    case .runEndEncoded: throw unsupported()
    case .null: throw unsupported()
    case .float16: throw unsupported()
    case .smallDecimal: throw unsupported()
    case .interval: throw unsupported()
    case .fixedBinary: throw unsupported()
    case .extended: throw unsupported()
    }
}

/// Which side of the join a name refers to.
enum JoinSide {
    case probe(String)
    case build(String)
}

/// Resolves a column name against the probe schema first, then the build schema — the same rule the
/// materialising join uses for its output names, including the `_right` suffix a shadowed build
/// column gets there.
func resolveJoinSide(_ name: String, probeNames: [String], buildNames: [String]) throws -> JoinSide {
    if probeNames.contains(name) { return .probe(name) }
    if buildNames.contains(name) { return .build(name) }
    if name.hasSuffix("_right") {
        let base = String(name.dropLast("_right".count))
        if buildNames.contains(base) { return .build(base) }
    }
    throw ArrowMetalError.invalidArrowArray(
        "no column named \(name) on either side of the join (probe: \(probeNames), build: \(buildNames))")
}

/// The probe-side columns a fused plan actually reads: the join key plus whatever sits on the probe
/// side of the aggregates (and, for the grouped form, of the keys). Duplicates removed, order kept.
///
/// A filtered join is otherwise paying to gather every column of every surviving row, when the
/// answer needs two of them — and one of the columns it gathers is often a utf8 one, which is the
/// most expensive gather in the library.
func probeSideColumns(_ probeKey: String, _ sides: [JoinSide]) -> [String] {
    var out = [probeKey]
    for s in sides {
        guard case .probe(let n) = s, !n.isEmpty, !out.contains(n) else { continue }
        out.append(n)
    }
    return out
}

// MARK: - The fused operator

/// Broadcast join followed by a whole-dataset aggregate, fused into one kernel per batch.
///
/// The probe, the gather of the build-side value and the accumulate all happen in `jfa_probe`; the
/// running answer is a pair of 64-bit words per aggregate in device memory that no batch reads back.
/// The merge stage does nothing at all, so `StreamStats.mergeNanos` is zero for this operator.
public final class BroadcastJoinAggregateOperator: StreamOperator {
    public let build: MetalRecordBatch
    public let probeKey: String
    public let buildKey: String
    public let specs: [StreamAggregate]
    public let filter: Expr?
    /// Aggregates a single kernel can carry, bounded by Metal's 31 buffer arguments (3 per aggregate
    /// after the 9 the probe itself needs).
    public static let maxAggregates = 7

    private let context: MetalContext
    private let table: BroadcastBuildTable
    private let state: MetalArrowBuffer
    /// nil until the first batch fixes the probe columns' types.
    private var slots: [JoinAggregateSource.Slot]?
    private var sides: [JoinSide]
    private var buildColumns: [FusedJoinColumn?] = []
    private var probePSO: MTLComputePipelineState?
    private var foldPSO: MTLComputePipelineState?
    /// The probe-side columns the kernel reads: the filter projects down to exactly these.
    private var probeColumns: [String] = []
    private var signature = ""
    private var probeRows = 0
    private var matchedBatches = 0

    public init(build: MetalRecordBatch, probeKey: String, buildKey: String, specs: [StreamAggregate],
                kind: JoinKind = .inner, filter: Expr? = nil, context: MetalContext = .shared) throws {
        guard kind == .inner else {
            throw ArrowMetalError.unsupportedType(
                "a fused join + aggregate is implemented for inner joins only; a left join followed by "
                + "an aggregate would have to count unmatched probe rows and is not implemented. "
                + "Use join(...).sink(...) and aggregate the result, or an inner join.")
        }
        guard !specs.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("a fused join + aggregate needs at least one aggregate")
        }
        guard specs.count <= Self.maxAggregates else {
            throw ArrowMetalError.unsupportedType(
                "a fused join + aggregate takes at most \(Self.maxAggregates) aggregates, got \(specs.count)")
        }
        for s in specs {
            switch s.op {
            case .sum, .count, .min, .max, .mean: break
            case .variance, .stddev, .countDistinctApprox:
                throw ArrowMetalError.unsupportedType(
                    "\(s.op.rawValue) is not implemented in a fused join + aggregate; "
                    + "sum, count, min, max and mean are")
            }
            if s.op != .count && s.column == nil {
                throw ArrowMetalError.invalidArrowArray("\(s.op.rawValue) needs a column")
            }
        }
        self.build = build
        self.probeKey = probeKey
        self.buildKey = buildKey
        self.specs = specs
        self.filter = filter
        self.context = build.firstContext ?? context
        guard let bk = build[buildKey] else {
            throw ArrowMetalError.invalidArrowArray("no column named \(buildKey) on the build side")
        }
        table = try BroadcastBuildTable(keyColumn: bk, context: self.context)
        state = try MetalArrowBuffer.allocate(byteCount: specs.count * 16, zeroed: true, context: self.context)
        sides = []
    }

    /// Fixes the aggregate plan against the first batch's schema and compiles the two kernels.
    private func prepare(_ batch: MetalRecordBatch) throws {
        var newSides: [JoinSide] = []
        var newSlots: [JoinAggregateSource.Slot] = []
        var newBuild: [FusedJoinColumn?] = []
        for s in specs {
            guard let name = s.column else {
                // count(*): no value column, no side.
                newSides.append(.probe(""))
                newBuild.append(nil)
                newSlots.append(.init(fromBuild: false, op: .count, kind: .int64, elementType: "", readsValue: false))
                continue
            }
            let side = try resolveJoinSide(name, probeNames: batch.names, buildNames: build.names)
            let column: AnyMetalArray
            switch side {
            case .probe(let n): column = batch[n]!
            case .build(let n): column = build[n]!
            }
            let fc = try fusedJoinColumn(column, name)
            let isBuild: Bool
            if case .build = side { isBuild = true } else { isBuild = false }
            newSides.append(side)
            newBuild.append(isBuild ? fc : nil)
            // A named column is always read — `count(column)` reads only its validity bit, but the
            // buffer still has to be bound and the kernel still has to know its type.
            newSlots.append(.init(fromBuild: isBuild, op: s.op == .mean ? .sum : s.op, kind: fc.kind,
                                  elementType: fc.elementType, readsValue: true))
        }
        sides = newSides
        slots = newSlots
        buildColumns = newBuild
        // The filter's output is projected down to exactly what the probe kernel reads. A filtered
        // join used to gather *every* column of the surviving rows — including a utf8 one, the most
        // expensive kind of gather there is — and then read two of them.
        probeColumns = probeSideColumns(probeKey, newSides)
        signature = table.keyType + "|" + newSlots.map(\.signature).joined(separator: ",")
        let src = JoinAggregateSource.source(KT: table.keyType, slots: newSlots)
        probePSO = try Dispatch.pipeline(context, family: "joinagg", source: src,
                                         function: "jfa_probe", type: signature)
        foldPSO = try Dispatch.pipeline(context, family: "joinagg", source: src,
                                        function: "jfa_fold", type: signature)
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        // The plan is fixed against the *unfiltered* batch, because the projection the filter writes
        // is decided by which columns the plan turns out to read.
        if slots == nil {
            guard batch.length > 0 else { return nil }
            try prepare(batch)
        }
        var work = batch
        if let f = filter {
            work = try streamFilterProject(batch, filter: f,
                                           projections: probeColumns.map { ($0, Expr.column($0)) },
                                           context: batch.firstContext ?? context)
        }
        let n = work.length
        guard n > 0 else { return nil }
        try Dispatch.checkLength(n)
        guard let slots, let probePSO, let foldPSO else { return nil }

        guard let pk = work[probeKey] else {
            throw ArrowMetalError.invalidArrowArray("no column named \(probeKey) on the probe side")
        }
        let probeValues: MetalArrowBuffer, probeValidity: MetalArrowBuffer?
        switch (pk, table.keyType) {
        case (.int32(let a), "int"): probeValues = a.values; probeValidity = a.validity
        case (.int64(let a), "long"): probeValues = a.values; probeValidity = a.validity
        default:
            throw ArrowMetalError.unsupportedType(
                "join keys \(pk.arrowFormat) / \(table.keys.arrowFormat): both must be int32, or both int64")
        }

        // Resolve this batch's probe-side value columns; the build side's were resolved once.
        var columns: [FusedJoinColumn?] = []
        for (i, side) in sides.enumerated() {
            if let bc = buildColumns[i] { columns.append(bc); continue }
            guard slots[i].readsValue, case .probe(let name) = side, let c = work[name] else {
                columns.append(nil)
                continue
            }
            let fc = try fusedJoinColumn(c, name)
            guard fc.elementType == slots[i].elementType else {
                throw ArrowMetalError.unsupportedType(
                    "column \(name) changed type mid-stream (\(slots[i].elementType) then \(fc.elementType))")
            }
            guard fc.length == n else {
                throw ArrowMetalError.invalidArrowArray("column \(name) has \(fc.length) rows, the batch has \(n)")
            }
            columns.append(fc)
        }

        let blocks = (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize
        let partials = try MetalArrowBuffer.allocate(byteCount: blocks * slots.count * 16, zeroed: false,
                                                     context: context)
        try context.run { enc in
            enc.setComputePipelineState(probePSO)
            table.bindProbe(enc, probeValues: probeValues, probeValidity: probeValidity, rows: n)
            enc.setBuffer(partials.mtl, offset: 0, index: 8)
            for (a, c) in columns.enumerated() {
                // count(*) reads nothing; bind the probe key buffer so every argument is bound.
                let values = c?.values ?? probeValues
                enc.setBuffer(values.mtl, offset: values.offset, index: 9 + 3 * a)
                let v = c?.validity ?? values
                enc.setBuffer(v.mtl, offset: v.offset, index: 10 + 3 * a)
                Dispatch.setUInt(enc, c?.validity == nil ? 0 : 1, index: 11 + 3 * a)
            }
            Dispatch.dispatch1D(enc, probePSO, count: n)
        }
        try context.run { enc in
            enc.setComputePipelineState(foldPSO)
            enc.setBuffer(partials.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(state.mtl, offset: 0, index: 2)
            let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        context.retainUntilFlush(partials)
        context.retainUntilFlush(state)
        context.retainUntilFlush(table)
        context.retainUntilFlush(probeValues)
        if let v = probeValidity { context.retainUntilFlush(v) }
        for c in columns { if let c { context.retainUntilFlush(c.owner) } }
        probeRows += n
        matchedBatches += 1
        return nil                      // the answer is on the GPU; there is nothing to merge
    }

    /// Nothing crosses to the merge stage, so it never records a kernel either.
    public var mergeUsesGPU: Bool { false }

    public func merge(_ partial: Any) throws {}

    public func finish() throws -> StreamResult {
        try context.flush()
        var r = StreamResult()
        let words = withExtendedLifetime(state) { p -> [UInt64] in
            let t = state.typed(UInt64.self)
            return (0..<(specs.count * 2)).map { t[$0] }
        }
        for (i, s) in specs.enumerated() {
            let value = words[2 * i], count = words[2 * i + 1]
            r.scalarNames.append(s.name)
            let kind = slots?[i].kind ?? .int64
            func typed(_ bits: UInt64) -> ExprScalar {
                switch kind {
                case .int64: return .int(Int64(bitPattern: bits))
                case .uint64: return .uint(bits)
                case .double: return .double(Double(bitPattern: bits))
                }
            }
            switch s.op {
            case .count:
                r.scalars.append(.int(Int64(bitPattern: count)))
            case .sum, .min, .max:
                r.scalars.append(count == 0 ? .null : typed(value))
            case .mean:
                if count == 0 { r.scalars.append(.null) }
                else {
                    let total: Double
                    switch kind {
                    case .int64: total = Double(Int64(bitPattern: value))
                    case .uint64: total = Double(value)
                    case .double: total = Double(bitPattern: value)
                    }
                    r.scalars.append(.double(total / Double(count)))
                }
            case .variance, .stddev, .countDistinctApprox:
                r.scalars.append(.null)
            }
        }
        return r
    }
}

// MARK: - Fused join + group-by

/// Broadcast join followed by a streaming group-by, with the joined batch never materialised.
///
/// The scalar form above accumulates in the probe kernel itself. A grouped one cannot: a per-group
/// accumulator needs a hash table probed once per *matched pair*, and the resident group table's
/// race-free insert relies on every key in a dispatch being distinct, which rows are not (§4.1 of
/// `docs/STREAMING.md`). So this keeps the join's index pairs and gathers **only the group keys and
/// the aggregated values** for the matched pairs — two or three columns, never the joined batch's
/// eight — and folds them into the ordinary streaming group-by, whose global table is already
/// resident on the GPU across batches.
///
/// A key may come from either side: a probe-side key is gathered with the left index array, a
/// build-side key with the right one.
public final class BroadcastJoinGroupByOperator: StreamOperator {
    public let build: MetalRecordBatch
    public let probeKey: String
    public let buildKey: String
    public let keys: [String]
    public let aggregates: [StreamAggregate]
    public let filter: Expr?
    private let context: MetalContext
    private let inner: StreamGroupByOperator
    /// The probe-side columns the plan reads; nil until the first batch fixes them.
    private var probeColumns: [String]?

    public init(build: MetalRecordBatch, probeKey: String, buildKey: String, keys: [String],
                aggregates: [StreamAggregate], kind: JoinKind = .inner, filter: Expr? = nil,
                denseKeyCount: Int? = nil, ddof: Int = 0, context: MetalContext = .shared) throws {
        guard kind == .inner else {
            throw ArrowMetalError.unsupportedType(
                "a fused join + group-by is implemented for inner joins only; a left join followed by "
                + "a group-by is not implemented. Use join(...).sink(...) and group the result.")
        }
        guard !keys.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("a fused join + group-by needs at least one key column")
        }
        guard !aggregates.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("a fused join + group-by needs at least one aggregate")
        }
        self.build = build
        self.probeKey = probeKey
        self.buildKey = buildKey
        self.keys = keys
        self.aggregates = aggregates
        self.filter = filter
        self.context = build.firstContext ?? context
        // The inner group-by sees private column names, so a key called "amount" and a summed column
        // called "amount" from the other side cannot collide.
        let renamed = aggregates.enumerated().map { i, a in
            StreamAggregate(a.op, a.column == nil ? nil : "v\(i)", name: a.name)
        }
        inner = StreamGroupByOperator(keys: keys.indices.map { "k\($0)" }, aggregates: renamed,
                                      filter: nil, denseKeyCount: denseKeyCount)
        inner.ddof = ddof
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        if probeColumns == nil {
            guard batch.length > 0 else { return nil }
            var s: [JoinSide] = []
            for n in keys + aggregates.compactMap(\.column) {
                s.append(try resolveJoinSide(n, probeNames: batch.names, buildNames: build.names))
            }
            probeColumns = probeSideColumns(probeKey, s)
        }
        var work = batch
        if let f = filter, let cols = probeColumns {
            // Only the probe-side keys and values reach the join; nothing else is gathered.
            work = try streamFilterProject(batch, filter: f, projections: cols.map { ($0, Expr.column($0)) },
                                           context: batch.firstContext ?? context)
        }
        guard work.length > 0 else { return nil }
        guard let lc = work[probeKey] else {
            throw ArrowMetalError.invalidArrowArray("no column named \(probeKey) on the probe side")
        }
        guard let rc = build[buildKey] else {
            throw ArrowMetalError.invalidArrowArray("no column named \(buildKey) on the build side")
        }
        let li: MetalArray<Int32>, ri: MetalArray<Int32>
        switch (lc, rc) {
        case (.int32(let a), .int32(let b)): (li, ri) = try hashJoin(left: a, right: b, kind: .inner)
        case (.int64(let a), .int64(let b)): (li, ri) = try hashJoin(left: a, right: b, kind: .inner)
        default:
            throw ArrowMetalError.unsupportedType(
                "join keys \(lc.arrowFormat) / \(rc.arrowFormat): both must be int32, or both int64")
        }
        guard li.length > 0 else { return nil }

        /// Gathers one column from whichever side owns it, with that side's index array.
        func gather(_ name: String) throws -> AnyMetalArray {
            switch try resolveJoinSide(name, probeNames: work.names, buildNames: build.names) {
            case .probe(let n): return try work[n]!.take(li)
            case .build(let n): return try build[n]!.take(ri)
            }
        }
        var names: [String] = [], columns: [AnyMetalArray] = []
        for (i, k) in keys.enumerated() {
            names.append("k\(i)")
            columns.append(try gather(k))
        }
        for (i, a) in aggregates.enumerated() {
            guard let c = a.column else { continue }
            names.append("v\(i)")
            columns.append(try gather(c))
        }
        let pairs = try MetalRecordBatch(names: names, columns: columns)
        return try inner.process(pairs)
    }

    public var mergeUsesGPU: Bool { inner.mergeUsesGPU }

    public func merge(_ partial: Any) throws { try inner.merge(partial) }

    public func finish() throws -> StreamResult {
        var r = try inner.finish()
        // The inner table grouped on "k0", "k1", ...; give the caller back its own key names.
        if let b = r.batch {
            var names = b.names
            for (i, k) in keys.enumerated() {
                if let at = names.firstIndex(of: "k\(i)") { names[at] = k }
            }
            r.batch = try MetalRecordBatch(names: names, columns: b.columns)
        }
        return r
    }
}

// MARK: - The builder

/// `StreamQuery(...).join(build, on:)` — a broadcast join waiting for its terminal.
///
/// The terminal decides the plan. `.sink` / `.collect` stream the joined rows out, as they always
/// did; `.sum` / `.aggregate` / `.groupBy` fuse the aggregate into the probe so the joined rows never
/// exist.
public final class StreamJoin {
    public let query: StreamQuery
    public let build: MetalRecordBatch
    public let probeKey: String
    public let buildKey: String
    public let kind: JoinKind

    init(query: StreamQuery, build: MetalRecordBatch, probeKey: String, buildKey: String, kind: JoinKind) {
        self.query = query
        self.build = build
        self.probeKey = probeKey
        self.buildKey = buildKey
        self.kind = kind
    }

    private func executor() -> StreamingExecutor {
        let e = StreamingExecutor(source: query.source, context: query.context)
        e.progress = query.progress
        return e
    }

    // MARK: unfused terminals — the joined rows are written out

    /// Streams the joined rows to `sink`. This is the unfused plan: every matched pair is gathered
    /// into a record batch.
    @discardableResult
    public func sink(_ s: StreamSink) throws -> StreamResult {
        let op = BroadcastJoinOperator(build: build, probeKey: probeKey, buildKey: buildKey,
                                       kind: kind, sink: s, filter: query.filterExpression)
        return try executor().run(op)
    }

    /// Streams the joined rows into an Arrow IPC stream file.
    @discardableResult
    public func sinkIPC(_ url: URL) throws -> StreamResult {
        try sink(try IPCStreamSink(url: url))
    }

    /// Collects the joined rows into one batch. Only for results known to be small.
    public func collect() throws -> MetalRecordBatch? {
        let s = CollectingSink()
        _ = try sink(s)
        return try s.table()
    }

    // MARK: fused terminals — the joined rows never exist

    /// Whole-dataset aggregates over the joined rows, accumulated inside the probe kernel.
    ///
    /// A column name is resolved against the probe side first and then the build side, so
    /// `sum("weight")` reaches a build-side column and `sum("amount")` a probe-side one. A build
    /// column shadowed by a probe column of the same name is reachable as `name_right`, matching the
    /// materialising join's output names.
    public func aggregate(_ specs: [StreamAggregate]) throws -> StreamResult {
        let op = try BroadcastJoinAggregateOperator(build: build, probeKey: probeKey, buildKey: buildKey,
                                                    specs: specs, kind: kind, filter: query.filterExpression,
                                                    context: query.context)
        return try executor().run(op)
    }

    public func sum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.sum, column, name: "sum")]).onlyScalar
    }
    public func count() throws -> Int {
        Int(try aggregate([StreamAggregate(.count, nil, name: "count")]).onlyScalar?.asInt64 ?? 0)
    }
    public func mean(_ column: String) throws -> Double? {
        try aggregate([StreamAggregate(.mean, column, name: "mean")]).onlyScalar?.asDouble
    }
    public func minimum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.min, column, name: "min")]).onlyScalar
    }
    public func maximum(_ column: String) throws -> ExprScalar? {
        try aggregate([StreamAggregate(.max, column, name: "max")]).onlyScalar
    }

    /// Group-by over the joined rows. Keys may come from either side.
    public func groupBy(_ keys: [String], _ aggs: [StreamAggregate], denseKeyCount: Int? = nil,
                        ddof: Int = 0) throws -> StreamResult {
        let op = try BroadcastJoinGroupByOperator(build: build, probeKey: probeKey, buildKey: buildKey,
                                                  keys: keys, aggregates: aggs, kind: kind,
                                                  filter: query.filterExpression,
                                                  denseKeyCount: denseKeyCount, ddof: ddof,
                                                  context: query.context)
        return try executor().run(op)
    }
}

public extension StreamQuery {
    /// Broadcast join: `build` is held in memory and every probe batch is joined against it on the
    /// GPU. The returned builder's terminal decides whether the joined rows are written out or an
    /// aggregate is fused into the probe.
    ///
    ///     try StreamQuery(ipc: path).filter(col("region") < 50)
    ///         .join(dim, on: "region", buildKey: "region")
    ///         .sum("amount")
    func join(_ build: MetalRecordBatch, on probeKey: String, buildKey: String? = nil,
              kind: JoinKind = .inner) -> StreamJoin {
        StreamJoin(query: self, build: build, probeKey: probeKey, buildKey: buildKey ?? probeKey, kind: kind)
    }
}
