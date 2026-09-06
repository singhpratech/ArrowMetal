import Foundation

// Running a physical plan.
//
// The whole tree runs inside one `MetalContext.batch { }`, so every kernel it dispatches goes into one
// command buffer and the ~150 µs round trip is paid once for the query rather than once per operator.
// Where an operator's *length* is decided by the GPU — a filter — the result is a pending array whose
// length lives in a device buffer, and the next kernel binds that buffer instead of a CPU-known count
// (`docs/DESIGN.md`, "Lengths flow on the GPU"), so a filter feeding a projection feeding another
// filter never returns to the CPU.
//
// Four operators are unavoidable sync points, because the CPU has to know a count before it can size
// the next dispatch: `GroupByKeys` (the number of groups), the hash join (the number of pairs), the
// top-k selection and `explode` (the total child length). `MetalContext.flush(reopen: true)` commits,
// waits and reopens the batch at each of those, so batching resumes immediately afterwards.
//
// Buffers come from and go back to `MetalContext.pool`, which parks rather than recycles while a batch
// is open, so an intermediate's memory is reused by the next operator of the same shape without ever
// being observed by pending GPU work.

/// Runs physical plans.
public enum Executor {

    /// Runs `plan` and returns its output as a record batch.
    public static func run(_ plan: PhysicalPlan) throws -> MetalRecordBatch {
        let ctx = try context(of: plan)
        return try ctx.batch { try node(plan) }
    }

    static func context(of plan: PhysicalPlan) throws -> MetalContext {
        switch plan {
        case .source(let src, _): return src.batch.columns.first?.metalContext ?? .shared
        case .fusedProject(let c, _), .maskFilter(let c, _), .fusedAggregate(let c, _, _),
             .hashAggregate(let c, _, _, _), .sort(let c, _), .topK(let c, _, _), .slice(let c, _, _),
             .distinct(let c, _), .window(let c, _), .explode(let c, _), .rename(let c, _):
            return try context(of: c)
        case .hashJoin(let a, _, _), .asofJoin(let a, _, _): return try context(of: a)
        case .concat(let xs):
            guard let head = xs.first else { throw ArrowMetalError.invalidArrowArray("concat with no inputs") }
            return try context(of: head)
        }
    }

    // MARK: - Operators

    static func node(_ plan: PhysicalPlan) throws -> MetalRecordBatch {
        switch plan {
        case .source(let src, let cols):
            return try src.batch.selecting(cols)

        case .rename(let child, let ps):
            let input = try node(child)
            var cols: [AnyMetalArray] = []
            for p in ps { cols.append(try column(p.expr, of: input)) }
            return try MetalRecordBatch(names: ps.map(\.name), columns: cols)

        case .fusedProject(let child, let f):
            let input = try node(child)
            var byName: [String: AnyMetalArray] = [:]
            if !f.computed.isEmpty {
                let q = ExprQuery(filter: f.filter,
                                  terminal: .project(f.computed.map { ExprQuery.Projection(name: $0.name, expr: $0.expr) }))
                let r = try input.query(q)
                for (i, n) in r.names.enumerated() { byName[n] = r.columns[i] }
            }
            for p in f.passthrough { byName[p.name] = try column(p.expr, of: input) }
            var cols: [AnyMetalArray] = []
            for n in f.order {
                guard let c = byName[n] else { throw ArrowMetalError.invalidArrowArray("projection lost column \(n)") }
                cols.append(c)
            }
            return try MetalRecordBatch(names: f.order, columns: cols)

        case .maskFilter(let child, let pred):
            let input = try node(child)
            let r = try input.query(query().project([("__am_mask", pred)]))
            guard case .boolean(let mask) = r.columns[0] else {
                throw ArrowMetalError.invalidArrowArray("filter predicate did not produce a boolean")
            }
            return try input.filter(mask)

        case .fusedAggregate(let child, let filter, let aggs):
            let input = try node(child)
            let r = try input.query(ExprQuery(filter: filter, terminal: .aggregate(aggs)))
            let schema = PlanSchema.of(input)
            var names: [String] = [], cols: [AnyMetalArray] = []
            for (i, n) in r.scalarNames.enumerated() {
                names.append(n)
                let want = (try? schema.aggregateField(aggs[i]))?.exprType
                cols.append(try scalarColumn(r.scalars[i], as: want, context: try context(of: child)))
            }
            return try MetalRecordBatch(names: names, columns: cols)

        case .hashAggregate(let child, let keys, let aggs, let fused):
            return try groupAggregate(try node(child), keys: keys, aggregates: aggs, fused: fused)

        case .sort(let child, let keys):
            let input = try node(child)
            guard !keys.isEmpty else { return input }
            return try input.sorted(by: keys.map { ($0.column, $0.descending) })

        case .topK(let child, let key, let k):
            let input = try node(child)
            guard let c = input[key.column] else {
                throw ArrowMetalError.invalidArrowArray("top-k: no column named \(key.column)")
            }
            let idx = try topKIndices(c, k: k, largest: key.descending)
            return try input.take(idx)

        case .slice(let child, let offset, let count):
            let input = try node(child)
            let n = input.length
            let lo = Swift.min(Swift.max(offset, 0), n)
            let len = Swift.max(Swift.min(count, n - lo), 0)
            if lo == 0 && len == n { return input }
            return try input.slice(offset: lo, length: len)

        case .distinct(let child, let subset):
            let input = try node(child)
            let names = subset ?? input.names
            var cols: [AnyMetalArray] = []
            for n in names {
                guard let c = input[n] else { throw ArrowMetalError.invalidArrowArray("unique: no column named \(n)") }
                cols.append(c)
            }
            guard input.length > 0 else { return input }
            let gk = try GroupByKeys(columns: cols)
            // `representativeRows` is the lowest row index of each group; sorting them restores the
            // input order, which is what "keep the first occurrence" means.
            let keep = try gk.representativeRows().sorted()
            return try input.take(keep)

        case .hashJoin(let l, let r, let spec):
            let left = try node(l), right = try node(r)
            return try left.joined(right, leftOn: spec.leftOn, rightOn: spec.rightOn,
                                   how: spec.how, suffix: spec.suffix)

        case .asofJoin(let l, let r, let spec):
            let left = try node(l), right = try node(r)
            return try left.joinedAsof(right, leftOn: spec.leftOn, rightOn: spec.rightOn,
                                       by: spec.by, byRight: spec.byRight, strategy: spec.strategy,
                                       tolerance: spec.tolerance, suffix: spec.suffix)

        case .concat(let xs):
            return try concatRecordBatches(try xs.map { try node($0) })

        case .window(let child, let specs):
            return try WindowOps.apply(try node(child), specs)

        case .explode(let child, let cols):
            return try explode(try node(child), cols)
        }
    }

    // MARK: - Helpers

    /// One column of a batch for an expression: a bare reference costs nothing, anything else is one
    /// fused kernel.
    static func column(_ e: Expr, of batch: MetalRecordBatch) throws -> AnyMetalArray {
        if case .column(let n) = e {
            guard let c = batch[n] else { throw ArrowMetalError.invalidArrowArray("no column named \(n)") }
            return c
        }
        let r = try batch.query(query().project([("__am_value", e)]))
        return r.columns[0]
    }

    static func topKIndices(_ c: AnyMetalArray, k: Int, largest: Bool) throws -> MetalArray<Int32> {
        switch c {
        case .int8(let a): return try a.topK(k, largest: largest)
        case .int16(let a): return try a.topK(k, largest: largest)
        case .int32(let a): return try a.topK(k, largest: largest)
        case .int64(let a): return try a.topK(k, largest: largest)
        case .uint8(let a): return try a.topK(k, largest: largest)
        case .uint16(let a): return try a.topK(k, largest: largest)
        case .uint32(let a): return try a.topK(k, largest: largest)
        case .uint64(let a): return try a.topK(k, largest: largest)
        case .float32(let a): return try a.topK(k, largest: largest)
        case .float64(let a): return try a.topK(k, largest: largest)
        default:
            let idx = try c.argsortIndices(descending: largest)
            return try idx.slice(offset: 0, length: Swift.min(k, idx.length))
        }
    }

    /// A one-row column holding an aggregate's scalar result.
    static func scalarColumn(_ s: ExprScalar, as want: ExprType?, context ctx: MetalContext) throws -> AnyMetalArray {
        func make<T: ArrowPrimitive>(_ v: T?, _ wrap: (MetalArray<T>) -> AnyMetalArray) throws -> AnyMetalArray {
            if let v { return wrap(try MetalArray<T>([v], context: ctx)) }
            return wrap(try MetalArray<T>([nil] as [T?], context: ctx))
        }
        let t: ExprType
        switch (want, s) {
        case (let w?, _) where w.isNumeric: t = w
        case (_, .double): t = .float64
        case (_, .uint): t = .uint64
        default: t = .int64
        }
        switch t {
        case .int8: return try make(s.asInt64.map { Int8(truncatingIfNeeded: $0) }) { .int8($0) }
        case .int16: return try make(s.asInt64.map { Int16(truncatingIfNeeded: $0) }) { .int16($0) }
        case .int32: return try make(s.asInt64.map { Int32(truncatingIfNeeded: $0) }) { .int32($0) }
        case .int64: return try make(s.asInt64) { .int64($0) }
        case .uint8: return try make(s.asInt64.map { UInt8(truncatingIfNeeded: $0) }) { .uint8($0) }
        case .uint16: return try make(s.asInt64.map { UInt16(truncatingIfNeeded: $0) }) { .uint16($0) }
        case .uint32: return try make(s.asInt64.map { UInt32(truncatingIfNeeded: $0) }) { .uint32($0) }
        case .uint64: return try make(s.asInt64.map { UInt64(bitPattern: $0) }) { .uint64($0) }
        case .float32: return try make(s.asDouble.map { Float($0) }) { .float32($0) }
        default: return try make(s.asDouble) { .float64($0) }
        }
    }

    // MARK: - Group-by

    static func groupAggregate(_ input: MetalRecordBatch, keys: [NamedExpr],
                               aggregates aggs: [ExprAggregate], fused: Bool) throws -> MetalRecordBatch {
        let ctx = input.columns.first?.metalContext ?? .shared
        var keyCols: [AnyMetalArray] = []
        for k in keys { keyCols.append(try column(k.expr, of: input)) }
        guard input.length > 0 else {
            // An empty input has no groups; the output schema still has to be right.
            var names = keys.map(\.name), cols = try keyCols.map { try $0.slice(offset: 0, length: 0) }
            let schema = PlanSchema.of(input)
            for a in aggs {
                names.append(a.name)
                let t = (try? schema.aggregateField(a))?.exprType ?? .float64
                cols.append(try emptyColumn(t, ctx))
            }
            return try MetalRecordBatch(names: names, columns: cols)
        }
        let gk = try GroupByKeys(columns: keyCols)
        try ctx.flush(reopen: true)         // the group count sizes every dispatch below
        var names = keys.map(\.name)
        var cols = try gk.groupKeys()

        if fused, gk.groupCount > 0 {
            // One fused kernel for every aggregate at once, over the dense ids as the group key.
            var qNames = input.names + ["__am_gid"]
            var qCols = input.columns + [AnyMetalArray.int32(gk.ids)]
            // Drop columns the aggregates never read so the kernel binds fewer buffers.
            var used = Set<String>(["__am_gid"])
            for a in aggs { used.formUnion(a.expr?.referencedColumns ?? []) }
            var keepN: [String] = [], keepC: [AnyMetalArray] = []
            for (i, n) in qNames.enumerated() where used.contains(n) { keepN.append(n); keepC.append(qCols[i]) }
            qNames = keepN; qCols = keepC
            let q = ExprQuery(groupKey: .column("__am_gid"), keyCount: gk.groupCount, keyName: "__am_gid",
                              terminal: .aggregate(aggs))
            let r = try runExprQuery(q, names: qNames, columns: qCols, context: ctx)
            for (i, n) in r.names.enumerated() where n != "__am_gid" {
                names.append(n)
                cols.append(try gk.trimExported(r.columns[i]))
            }
            return try MetalRecordBatch(names: names, columns: cols)
        }

        for a in aggs {
            names.append(a.name)
            cols.append(try gk.trimExported(try aggregateOne(gk, a, over: input, ctx)))
        }
        return try MetalRecordBatch(names: names, columns: cols)
    }

    /// One aggregate through `GroupBy`'s own kernels, for the types the fused group-by cannot take
    /// (float64 sums, 64-bit min/max, anything the 32-bit atomics of the fused path reject).
    static func aggregateOne(_ gk: GroupByKeys, _ a: ExprAggregate, over input: MetalRecordBatch,
                             _ ctx: MetalContext) throws -> AnyMetalArray {
        let gb = gk.groupBy
        guard let e = a.expr else {
            guard a.op == .count else { throw ArrowMetalError.invalidArrowArray("\(a.op.rawValue) needs an expression") }
            return .int64(try gb.count())
        }
        let values = try column(e, of: input)
        switch a.op {
        case .count:
            switch values {
            case .int8(let v): return .int64(try gb.count(v))
            case .int16(let v): return .int64(try gb.count(v))
            case .int32(let v): return .int64(try gb.count(v))
            case .int64(let v): return .int64(try gb.count(v))
            case .uint8(let v): return .int64(try gb.count(v))
            case .uint16(let v): return .int64(try gb.count(v))
            case .uint32(let v): return .int64(try gb.count(v))
            case .uint64(let v): return .int64(try gb.count(v))
            case .float32(let v): return .int64(try gb.count(v))
            case .float64(let v): return .int64(try gb.count(v))
            case .boolean(let v): return .int64(try gb.count(try v.toUInt8Array()))
            default: return .int64(try gb.count())
            }
        case .sum:
            switch values {
            case .int8(let v): return .int64(try gb.sum(v))
            case .int16(let v): return .int64(try gb.sum(v))
            case .int32(let v): return .int64(try gb.sum(v))
            case .int64(let v): return .int64(try gb.sum(v))
            case .uint8(let v): return .uint64(try gb.sumUnsigned(try v.cast(to: UInt64.self)))
            case .uint16(let v): return .uint64(try gb.sumUnsigned(try v.cast(to: UInt64.self)))
            case .uint32(let v): return .uint64(try gb.sumUnsigned(try v.cast(to: UInt64.self)))
            case .uint64(let v): return .uint64(try gb.sumUnsigned(v))
            case .float32(let v): return .float64(try gb.sumFloatAsDouble(v))
            case .float64(let v): return .float64(try gb.sumDouble(v))
            default: throw ArrowMetalError.unsupportedType("group-by sum over \(values.arrowFormat)")
            }
        case .mean:
            switch values {
            case .int8(let v): return .float64(try gb.mean(v))
            case .int16(let v): return .float64(try gb.mean(v))
            case .int32(let v): return .float64(try gb.mean(v))
            case .int64(let v): return .float64(try gb.mean(v))
            case .uint8(let v): return .float64(try gb.mean(v))
            case .uint16(let v): return .float64(try gb.mean(v))
            case .uint32(let v): return .float64(try gb.mean(v))
            case .uint64(let v): return .float64(try gb.mean(v))
            case .float32(let v): return .float64(try gb.meanFloat(v))
            case .float64(let v): return .float64(try gb.meanDouble(v))
            default: throw ArrowMetalError.unsupportedType("group-by mean over \(values.arrowFormat)")
            }
        case .min, .max:
            let isMin = a.op == .min
            func mm<T: ArrowPrimitive>(_ v: MetalArray<T>) throws -> MetalArray<T> {
                isMin ? try gb.min64(v) : try gb.max64(v)
            }
            switch values {
            case .int8(let v): return .int8(try mm(v))
            case .int16(let v): return .int16(try mm(v))
            case .int32(let v): return .int32(try mm(v))
            case .int64(let v): return .int64(try mm(v))
            case .uint8(let v): return .uint8(try mm(v))
            case .uint16(let v): return .uint16(try mm(v))
            case .uint32(let v): return .uint32(try mm(v))
            case .uint64(let v): return .uint64(try mm(v))
            case .float32(let v): return .float32(try mm(v))
            case .float64(let v): return .float64(try mm(v))
            default: throw ArrowMetalError.unsupportedType("group-by \(a.op.rawValue) over \(values.arrowFormat)")
            }
        }
    }

    static func emptyColumn(_ t: ExprType, _ ctx: MetalContext) throws -> AnyMetalArray {
        switch t {
        case .int8: return .int8(try MetalArray<Int8>([], context: ctx))
        case .int16: return .int16(try MetalArray<Int16>([], context: ctx))
        case .int32: return .int32(try MetalArray<Int32>([], context: ctx))
        case .int64: return .int64(try MetalArray<Int64>([], context: ctx))
        case .uint8: return .uint8(try MetalArray<UInt8>([], context: ctx))
        case .uint16: return .uint16(try MetalArray<UInt16>([], context: ctx))
        case .uint32: return .uint32(try MetalArray<UInt32>([], context: ctx))
        case .uint64: return .uint64(try MetalArray<UInt64>([], context: ctx))
        case .float32: return .float32(try MetalArray<Float>([], context: ctx))
        case .float64: return .float64(try MetalArray<Double>([], context: ctx))
        case .boolean: return .boolean(try MetalBooleanArray([], context: ctx))
        case .utf8: return .string(try MetalStringArray([], context: ctx))
        }
    }

    // MARK: - Explode

    /// Arrow / Polars `explode`: one output row per element of the list column, the other columns
    /// repeated. The repeat pattern comes from the list offsets, which the CPU has to read (they are
    /// the output length), so this operator is a sync point; the gather itself is a GPU `take`.
    static func explode(_ input: MetalRecordBatch, _ columns: [String]) throws -> MetalRecordBatch {
        guard let first = columns.first else { return input }
        guard columns.count == 1 else {
            throw ArrowMetalError.unsupportedType("explode of several columns at once is not implemented")
        }
        guard let col = input[first], case .list(let list) = col else {
            throw ArrowMetalError.unsupportedType("explode: column \(first) is \(input[first]?.arrowFormat ?? "missing"), not a list")
        }
        let ctx = col.metalContext
        try ctx.flush()
        let n = list.length
        let offs = list.offsets.typed(Int32.self)
        var parent: [Int32] = [], child: [Int32] = []
        parent.reserveCapacity(n)
        for i in 0..<n {
            let lo = Int(offs[i]), hi = Int(offs[i + 1])
            if !list.isValid(i) || hi <= lo {
                // A null or empty list still produces one row, with a null value (Polars' behaviour).
                parent.append(Int32(i)); child.append(-1)
            } else {
                for j in lo..<hi { parent.append(Int32(i)); child.append(Int32(j)) }
            }
        }
        let parentIdx = try MetalArray<Int32>(parent, context: ctx)
        let childIdx = try MetalArray<Int32>(child.map { $0 < 0 ? nil : $0 }, context: ctx)
        var cols: [AnyMetalArray] = []
        for (i, name) in input.names.enumerated() {
            if name == first { cols.append(try list.values.take(childIdx)) }
            else { cols.append(try input.columns[i].take(parentIdx)) }
        }
        return try MetalRecordBatch(names: input.names, columns: cols)
    }
}

// MARK: - The user-facing entry point

/// A lazy query over Metal-resident data: build it, `explain()` it, `collect()` it.
///
/// ```swift
/// let sales = PlanSource(name: "sales", batch: batch)
/// let df = LazyFrame(sales)
///     .filter(col("amount") > 100)
///     .groupBy([NamedExpr("region", col("region"))], [ExprAggregate(.sum, col("amount"), name: "total")])
///     .sort([SortKey("total", descending: true)])
///     .limit(10)
/// print(try df.explain())
/// let out = try df.collect()
/// ```
public struct LazyFrame {
    public var plan: LogicalPlan
    public var optimizer = Optimizer()

    public init(_ plan: LogicalPlan) { self.plan = plan }
    public init(_ source: PlanSource) { self.plan = .scan(source, columns: nil) }
    public init(_ batch: MetalRecordBatch, name: String = "batch") {
        self.plan = .scan(PlanSource(name: name, batch: batch), columns: nil)
    }

    private func with(_ p: LogicalPlan) -> LazyFrame { var c = self; c.plan = p; return c }

    public func filter(_ e: Expr) -> LazyFrame { with(.filter(plan, e)) }
    public func select(_ ps: [NamedExpr]) -> LazyFrame { with(.project(plan, ps)) }
    public func select(_ names: [String]) -> LazyFrame { select(names.map { NamedExpr($0, .column($0)) }) }
    public func withColumns(_ ps: [NamedExpr]) -> LazyFrame { with(.withColumns(plan, ps)) }
    public func aggregate(_ aggs: [ExprAggregate]) -> LazyFrame { with(.aggregate(plan, aggs)) }
    public func groupBy(_ keys: [NamedExpr], _ aggs: [ExprAggregate]) -> LazyFrame {
        with(.groupAggregate(plan, keys: keys, aggregates: aggs))
    }
    public func groupBy(_ keys: [String], _ aggs: [ExprAggregate]) -> LazyFrame {
        groupBy(keys.map { NamedExpr($0, .column($0)) }, aggs)
    }
    public func sort(_ keys: [SortKey]) -> LazyFrame { with(.sort(plan, keys)) }
    public func sort(_ column: String, descending: Bool = false) -> LazyFrame {
        sort([SortKey(column, descending: descending)])
    }
    public func limit(_ n: Int, offset: Int = 0) -> LazyFrame { with(.limit(plan, count: n, offset: offset)) }
    public func unique(subset: [String]? = nil) -> LazyFrame { with(.distinct(plan, subset: subset)) }
    public func join(_ other: LazyFrame, leftOn: [String], rightOn: [String], how: JoinHow,
                     suffix: String = "_right") -> LazyFrame {
        with(.join(plan, other.plan, JoinSpec(leftOn: leftOn, rightOn: rightOn, how: how, suffix: suffix)))
    }
    public func join(_ other: LazyFrame, on: [String], how: JoinHow = .inner) -> LazyFrame {
        join(other, leftOn: on, rightOn: on, how: how)
    }
    public func joinAsof(_ other: LazyFrame, _ spec: AsofSpec) -> LazyFrame {
        with(.joinAsof(plan, other.plan, spec))
    }
    public func concat(_ others: [LazyFrame]) -> LazyFrame { with(.union([plan] + others.map(\.plan))) }
    public func window(_ specs: [WindowSpec]) -> LazyFrame { with(.window(plan, specs)) }
    public func explode(_ columns: [String]) -> LazyFrame { with(.explode(plan, columns)) }

    /// The output schema without running anything.
    public func schema() throws -> PlanSchema { try optimized().schema() }

    public func optimized() throws -> LogicalPlan {
        var o = optimizer
        return try o.run(plan)
    }

    /// The optimized logical plan and the physical plan it lowers to, the way Polars' `explain()`
    /// prints one. `optimized: false` shows the plan as written.
    public func explain(optimized doOptimize: Bool = true) throws -> String {
        guard doOptimize else { return plan.describe() }
        var o = optimizer
        let logical = try o.run(plan)
        let physical = try PhysicalPlanner.plan(logical)
        var s = "LOGICAL PLAN\n" + logical.describe(indent: 1)
        s += "\n\nPHYSICAL PLAN\n" + physical.describe(indent: 1)
        if !o.applied.isEmpty { s += "\n\nRULES APPLIED: " + o.applied.joined(separator: ", ") }
        return s
    }

    /// Runs the plan and returns the result.
    public func collect(optimize doOptimize: Bool = true) throws -> MetalRecordBatch {
        let logical = doOptimize ? try optimized() : plan
        _ = try logical.schema()
        return try Executor.run(try PhysicalPlanner.plan(logical))
    }
}
