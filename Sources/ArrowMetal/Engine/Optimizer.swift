import Foundation

// Plan rewrites. Every one is a pure `LogicalPlan -> LogicalPlan` function; the driver runs them to a
// fixed point and records which ones fired, so `explain()` can say what happened.
//
// The rules, and why each one is worth having on a GPU specifically:
//
// | rule | what it does | why it pays here |
// |---|---|---|
// | constant folding | evaluates literal-only subtrees, collapses `and(true, x)` | shrinks the MSL the kernel compiler sees |
// | filter fusion | `filter(filter(x, a), b)` -> `filter(x, and(a, b))` | one compaction pipeline instead of two |
// | predicate pushdown | moves conjuncts below projections, sorts, group-bys, joins, unions | a GPU sort or hash join costs far more per row than a predicate |
// | projection pruning | narrows scans and drops unused projection outputs | fewer columns is fewer bytes, and this engine is memory bound |
// | expression CSE | drops duplicate outputs with the same canonical text | the kernel's own CSE only sees one query at a time |
// | join reordering | puts the smaller estimated side on the build side of an inner join | the build side is the one that goes in the hash table |
// | fusion planning | marks maximal element-wise + filter + aggregate regions | one kernel per region instead of one per operator |
//
// Fusion planning is not in this file — it is the physical planner's job (`PhysicalPlan.swift`),
// because it is a choice of kernel, not a change of meaning. Everything here preserves the result
// exactly, row for row and null for null.

/// Row-count estimates, and optionally exact distinct counts computed on the GPU.
public struct PlanStats {
    /// Fraction of rows a predicate is assumed to keep when nothing better is known.
    public var filterSelectivity = 0.25
    /// When set, `distinctCount` runs a real GPU `GroupByKeys` pass over a scan column instead of
    /// assuming every row is distinct. It costs a pass over the column, so it is off by default and
    /// only worth turning on for a plan that will run for far longer than the estimate costs.
    public var useGPUDistinctCounts = false
    public init() {}

    private final class Cache: @unchecked Sendable { var v: [String: Int] = [:] }
    private let cache = Cache()

    /// Distinct values of a scan column: exact when `useGPUDistinctCounts`, otherwise the row count.
    public func distinctCount(_ src: PlanSource, _ column: String) -> Int {
        let rows = src.batch.length
        guard useGPUDistinctCounts, let c = src.batch[column], rows > 0 else { return Swift.max(rows, 1) }
        let key = "\(ObjectIdentifier(src).hashValue)/\(column)"
        if let v = cache.v[key] { return v }
        let v = (try? GroupByKeys(columns: [c]).groupCount) ?? rows
        cache.v[key] = Swift.max(v, 1)
        return Swift.max(v, 1)
    }

    /// Estimated output rows of a plan.
    public func rows(_ plan: LogicalPlan) -> Int {
        switch plan {
        case .scan(let src, _): return src.batch.length
        case .filter(let c, _): return Swift.max(Int(Double(rows(c)) * filterSelectivity), 1)
        case .project(let c, _), .withColumns(let c, _), .window(let c, _): return rows(c)
        case .aggregate: return 1
        case .groupAggregate(let c, let keys, _):
            // Without column statistics a group-by is assumed to fold by a constant factor per key.
            var n = rows(c)
            for _ in keys { n = Swift.max(n / 16, 1) }
            return n
        case .sort(let c, _), .distinct(let c, _): return rows(c)
        case .limit(let c, let n, let o): return Swift.max(Swift.min(rows(c) - o, n), 0)
        case .join(let l, let r, let spec):
            let a = rows(l), b = rows(r)
            switch spec.how {
            case .semi: return Swift.max(Swift.min(a, b), 1)
            case .anti: return Swift.max(a / 2, 1)
            case .inner: return Swift.max(Swift.min(a, b), 1)
            case .left: return a
            case .right: return b
            case .full: return a + b
            }
        case .joinAsof(let l, _, _): return rows(l)
        case .union(let xs): return xs.reduce(0) { $0 + rows($1) }
        case .explode(let c, _): return rows(c) * 2
        }
    }
}

/// The rewrite driver.
public struct Optimizer {
    public var stats = PlanStats()
    /// Rules to skip, by name, for tests that want to see one rule in isolation.
    public var disabled: Set<String> = []
    public init() {}

    /// The rules that fired, in the order they first fired. `explain()` prints this.
    public private(set) var applied: [String] = []

    public static func optimize(_ plan: LogicalPlan) throws -> LogicalPlan {
        var o = Optimizer()
        return try o.run(plan)
    }

    public mutating func run(_ plan: LogicalPlan) throws -> LogicalPlan {
        var p = plan
        for _ in 0..<8 {
            let before = p.describe()
            p = fold(p)
            p = fuseFilters(p)
            p = pushFilters(p)
            p = reorderJoins(p)
            p = eliminateCommonSubexpressions(p)
            p = prune(p, required: nil)
            if p.describe() == before { break }
        }
        _ = try p.schema()          // the rewrites must not have broken the plan
        return p
    }

    private mutating func note(_ rule: String) {
        if !applied.contains(rule) { applied.append(rule) }
    }
    private func enabled(_ rule: String) -> Bool { !disabled.contains(rule) }

    // MARK: - Constant folding

    private mutating func fold(_ plan: LogicalPlan) -> LogicalPlan {
        guard enabled("constant_folding") else { return plan }
        let out = plan.mappingExpressions { Optimizer.foldExpr($0) }
        if out.describe() != plan.describe() { note("constant_folding") }
        return out.mappingChildren { fold($0) }
    }

    /// Evaluates literal-only subtrees and collapses the identities of `and` / `or` / `not`.
    public static func foldExpr(_ e: Expr) -> Expr {
        switch e {
        case .binary(let op, let a0, let b0):
            let a = foldExpr(a0), b = foldExpr(b0)
            // `and` / `or` propagate nulls; only their Kleene forms let a literal absorb the other
            // side. `false and null` is *null* under `and`, so folding it to `false` would change a
            // projection's result (a filter cannot tell the two apart, but `select` can).
            if op == .and || op == .andKleene {
                if op.isKleene {
                    if case .bool(false) = a { return .bool(false) }
                    if case .bool(false) = b { return .bool(false) }
                }
                if case .bool(true) = a { return b }
                if case .bool(true) = b { return a }
                if a == b { return a }
            }
            if op == .or || op == .orKleene {
                if op.isKleene {
                    if case .bool(true) = a { return .bool(true) }
                    if case .bool(true) = b { return .bool(true) }
                }
                if case .bool(false) = a { return b }
                if case .bool(false) = b { return a }
                if a == b { return a }
            }
            if let x = intLiteral(a), let y = intLiteral(b) {
                switch op {
                case .add: return .int(x &+ y)
                case .sub: return .int(x &- y)
                case .mul: return .int(x &* y)
                // `Int64.min / -1` overflows; the GPU wraps rather than trapping, so fold the same way.
                case .div: return y == 0 ? .int(0) : .int(x.dividedReportingOverflow(by: y).partialValue)
                case .eq: return .bool(x == y)
                case .ne: return .bool(x != y)
                case .lt: return .bool(x < y)
                case .le: return .bool(x <= y)
                case .gt: return .bool(x > y)
                case .ge: return .bool(x >= y)
                default: break
                }
            }
            if let x = doubleLiteral(a), let y = doubleLiteral(b), op.isArithmetic || op.isComparison {
                switch op {
                case .add: return .double(x + y)
                case .sub: return .double(x - y)
                case .mul: return .double(x * y)
                case .div: return .double(x / y)
                case .eq: return .bool(x == y)
                case .ne: return .bool(x != y)
                case .lt: return .bool(x < y)
                case .le: return .bool(x <= y)
                case .gt: return .bool(x > y)
                case .ge: return .bool(x >= y)
                default: break
                }
            }
            // `x * 1`, `x + 0`, `x / 1` disappear.
            if op == .mul, intLiteral(b) == 1 { return a }
            if op == .mul, intLiteral(a) == 1 { return b }
            if op == .add, intLiteral(b) == 0 { return a }
            if op == .sub, intLiteral(b) == 0 { return a }
            if op == .div, intLiteral(b) == 1 { return a }
            return .binary(op, a, b)

        case .unary(let op, let a0):
            let a = foldExpr(a0)
            if op == .not, case .bool(let v) = a { return .bool(!v) }
            if op == .negate, let x = intLiteral(a) { return .int(0 &- x) }
            // `-Int64.min` overflows; the GPU's `abs` wraps to Int64.min, so fold the same way.
            if op == .abs, let x = intLiteral(a) { return .int(x < 0 ? (0 &- x) : x) }
            if op == .not, case .unary(.not, let inner) = a { return inner }
            return .unary(op, a)

        case .cast(let a0, let t):
            let a = foldExpr(a0)
            if let x = intLiteral(a), t.isInteger { return .typedInt(x, t) }
            if let x = doubleLiteral(a), t.isFloat { return .typedDouble(x, t) }
            return .cast(a, t)

        case .ifElse(let c0, let a0, let b0):
            let c = foldExpr(c0), a = foldExpr(a0), b = foldExpr(b0)
            if case .bool(true) = c { return a }
            if case .bool(false) = c { return b }
            // `if_else(c, a, a)` is *not* `a`: Arrow's if_else is null wherever the condition is null,
            // and nothing here knows whether `c` can be. Only a literal condition folds away.
            return .ifElse(c, a, b)

        case .coalesce(let xs):
            var out: [Expr] = []
            for x in xs.map(foldExpr) {
                if case .nullLiteral = x { continue }
                out.append(x)
                if !mayBeNullLiteral(x) { break }        // nothing after a non-null literal can be reached
            }
            if out.isEmpty { return xs.map(foldExpr).last ?? .int(0) }
            return out.count == 1 ? out[0] : .coalesce(out)

        case .fillNull(let a0, let b0):
            let a = foldExpr(a0), b = foldExpr(b0)
            if case .nullLiteral = a { return b }
            return .fillNull(a, b)

        case .isNull(let a0): return .isNull(foldExpr(a0))
        case .isValid(let a0): return .isValid(foldExpr(a0))
        case .isIn(let a0, let xs): return .isIn(foldExpr(a0), xs.map(foldExpr))
        case .stringMatch(let p, let a0, let s): return .stringMatch(p, foldExpr(a0), s)
        default: return e
        }
    }

    private static func intLiteral(_ e: Expr) -> Int64? {
        switch e { case .int(let v): return v; case .typedInt(let v, _): return v; default: return nil }
    }
    private static func doubleLiteral(_ e: Expr) -> Double? {
        switch e {
        case .double(let v): return v
        case .typedDouble(let v, _): return v
        case .int(let v): return Double(v)
        case .typedInt(let v, _): return Double(v)
        default: return nil
        }
    }
    private static func mayBeNullLiteral(_ e: Expr) -> Bool {
        switch e { case .int, .typedInt, .double, .typedDouble, .bool, .string: return false; default: return true }
    }

    // MARK: - Filter fusion

    private mutating func fuseFilters(_ plan: LogicalPlan) -> LogicalPlan {
        guard enabled("filter_fusion") else { return plan }
        let p = plan.mappingChildren { fuseFilters($0) }
        if case .filter(let inner, let outerPred) = p, case .filter(let base, let innerPred) = inner {
            note("filter_fusion")
            return fuseFilters(.filter(base, Optimizer.foldExpr(.binary(.and, innerPred, outerPred))))
        }
        return p
    }

    // MARK: - Predicate pushdown

    private mutating func pushFilters(_ plan: LogicalPlan) -> LogicalPlan {
        guard enabled("predicate_pushdown") else { return plan }
        let p = plan.mappingChildren { pushFilters($0) }
        guard case .filter(let child, let pred) = p else { return p }
        let parts = pred.conjuncts
        guard !parts.isEmpty else { return p }

        switch child {
        case .project(let base, let ps):
            var map: [String: Expr] = [:]
            for x in ps { map[x.name] = x.expr }
            let names = Set(ps.map(\.name))
            var pushed: [Expr] = [], kept: [Expr] = []
            for c in parts {
                if Set(c.referencedColumns).isSubset(of: names) { pushed.append(c.substituting(map)) } else { kept.append(c) }
            }
            guard !pushed.isEmpty else { return p }
            note("predicate_pushdown")
            let below = LogicalPlan.project(.filter(base, Expr.allOf(pushed)!), ps)
            return kept.isEmpty ? pushFilters(below) : .filter(pushFilters(below), Expr.allOf(kept)!)

        case .withColumns(let base, let ps):
            // Only conjuncts that touch none of the added columns can move below.
            let added = Set(ps.map(\.name))
            var pushed: [Expr] = [], kept: [Expr] = []
            for c in parts {
                if Set(c.referencedColumns).isDisjoint(with: added) { pushed.append(c) } else { kept.append(c) }
            }
            guard !pushed.isEmpty else { return p }
            note("predicate_pushdown")
            let below = LogicalPlan.withColumns(.filter(base, Expr.allOf(pushed)!), ps)
            return kept.isEmpty ? pushFilters(below) : .filter(pushFilters(below), Expr.allOf(kept)!)

        case .sort(let base, let keys):
            note("predicate_pushdown")
            return pushFilters(.sort(.filter(base, pred), keys))

        case .distinct(let base, let subset):
            // A predicate over the distinct subset commutes with the deduplication.
            if let subset, !Set(pred.referencedColumns).isSubset(of: Set(subset)) { return p }
            note("predicate_pushdown")
            return pushFilters(.distinct(.filter(base, pred), subset: subset))

        case .union(let xs):
            note("predicate_pushdown")
            return .union(xs.map { pushFilters(.filter($0, pred)) })

        case .groupAggregate(let base, let keys, let aggs):
            // A conjunct over the group keys alone selects whole groups, so it can run first.
            var map: [String: Expr] = [:]
            for k in keys { map[k.name] = k.expr }
            let keyNames = Set(keys.map(\.name))
            var pushed: [Expr] = [], kept: [Expr] = []
            for c in parts {
                if Set(c.referencedColumns).isSubset(of: keyNames) { pushed.append(c.substituting(map)) } else { kept.append(c) }
            }
            guard !pushed.isEmpty else { return p }
            note("predicate_pushdown")
            let below = LogicalPlan.groupAggregate(.filter(base, Expr.allOf(pushed)!), keys: keys, aggregates: aggs)
            return kept.isEmpty ? pushFilters(below) : .filter(pushFilters(below), Expr.allOf(kept)!)

        case .join(let l, let r, let spec):
            guard let ls = try? l.schema(), let rs = try? r.schema() else { return p }
            let lNames = Set(ls.names), rNames = Set(rs.names)
            // Only unambiguous names may move: a name on both sides was renamed by the join.
            let leftOnly = lNames.subtracting(rNames), rightOnly = rNames.subtracting(lNames)
            var toLeft: [Expr] = [], toRight: [Expr] = [], kept: [Expr] = []
            for c in parts {
                let cols = Set(c.referencedColumns)
                if cols.isSubset(of: leftOnly), spec.how == .inner || spec.how == .left || spec.how == .semi || spec.how == .anti {
                    toLeft.append(c)
                } else if cols.isSubset(of: rightOnly), spec.how == .inner || spec.how == .right {
                    toRight.append(c)
                } else {
                    kept.append(c)
                }
            }
            guard !toLeft.isEmpty || !toRight.isEmpty else { return p }
            note("predicate_pushdown")
            let nl = toLeft.isEmpty ? l : LogicalPlan.filter(l, Expr.allOf(toLeft)!)
            let nr = toRight.isEmpty ? r : LogicalPlan.filter(r, Expr.allOf(toRight)!)
            let below = LogicalPlan.join(pushFilters(nl), pushFilters(nr), spec)
            return kept.isEmpty ? below : .filter(below, Expr.allOf(kept)!)

        case .explode(let base, let cols):
            let exploded = Set(cols)
            var pushed: [Expr] = [], kept: [Expr] = []
            for c in parts {
                if Set(c.referencedColumns).isDisjoint(with: exploded) { pushed.append(c) } else { kept.append(c) }
            }
            guard !pushed.isEmpty else { return p }
            note("predicate_pushdown")
            let below = LogicalPlan.explode(.filter(base, Expr.allOf(pushed)!), cols)
            return kept.isEmpty ? pushFilters(below) : .filter(pushFilters(below), Expr.allOf(kept)!)

        default:
            return p
        }
    }

    // MARK: - Projection pruning

    /// Narrows scans and drops projection outputs nothing above asks for. `required == nil` means
    /// "every column of this node's output is needed" (the root).
    private mutating func prune(_ plan: LogicalPlan, required: Set<String>?) -> LogicalPlan {
        guard enabled("projection_pruning") else { return plan }
        switch plan {
        case .scan(let src, let cols):
            let have = cols ?? src.schema.names
            guard let required else { return plan }
            let keep = have.filter { required.contains($0) }
            guard !keep.isEmpty, keep.count < have.count else { return plan }
            note("projection_pruning")
            return .scan(src, columns: keep)

        case .filter(let c, let pred):
            let need = required.map { $0.union(pred.referencedColumns) }
            return .filter(prune(c, required: need), pred)

        case .project(let c, let ps):
            var outs = ps
            if let required {
                let keep = ps.filter { required.contains($0.name) }
                if !keep.isEmpty && keep.count < ps.count { note("projection_pruning"); outs = keep }
            }
            var need = Set<String>()
            for p in outs { need.formUnion(p.expr.referencedColumns) }
            return .project(prune(c, required: need), outs)

        case .withColumns(let c, let ps):
            var outs = ps
            if let required {
                let keep = ps.filter { required.contains($0.name) }
                if keep.count < ps.count { note("projection_pruning"); outs = keep }
            }
            if outs.isEmpty { return prune(c, required: required) }
            var need = required.map { $0.subtracting(outs.map(\.name)) } ?? Set<String>()
            for p in outs { need.formUnion(p.expr.referencedColumns) }
            // A `with_columns` keeps every input column, so without a `required` set nothing is known.
            return .withColumns(prune(c, required: required == nil ? nil : need), outs)

        case .aggregate(let c, let aggs):
            var need = Set<String>()
            for a in aggs { need.formUnion(a.expr?.referencedColumns ?? []) }
            return .aggregate(prune(c, required: need), aggs)

        case .groupAggregate(let c, let keys, let aggs):
            var need = Set<String>()
            for k in keys { need.formUnion(k.expr.referencedColumns) }
            for a in aggs { need.formUnion(a.expr?.referencedColumns ?? []) }
            return .groupAggregate(prune(c, required: need), keys: keys, aggregates: aggs)

        case .sort(let c, let keys):
            let need = required.map { $0.union(keys.map(\.column)) }
            return .sort(prune(c, required: need), keys)

        case .limit(let c, let n, let o): return .limit(prune(c, required: required), count: n, offset: o)

        case .distinct(let c, let subset):
            let need = required.map { r in subset.map { r.union($0) } ?? r }
            return .distinct(prune(c, required: subset == nil ? nil : need), subset: subset)

        case .join(let l, let r, let spec):
            guard let ls = try? l.schema(), let rs = try? r.schema() else { return plan }
            var lNeed = Set(spec.leftOn), rNeed = Set(spec.rightOn)
            if let required {
                lNeed.formUnion(ls.names.filter { required.contains($0) })
                // Right columns may have been renamed by the suffix; map back.
                var taken = ls.names
                for f in rs.fields {
                    if let j = spec.rightOn.firstIndex(of: f.name), spec.leftOn[j] == f.name { continue }
                    let out = MetalRecordBatch.uniqueName(f.name, taken: taken, suffix: spec.suffix)
                    taken.append(out)
                    if required.contains(out) { rNeed.insert(f.name) }
                }
            } else {
                lNeed.formUnion(ls.names); rNeed.formUnion(rs.names)
            }
            return .join(prune(l, required: lNeed), prune(r, required: rNeed), spec)

        case .joinAsof(let l, let r, let spec):
            return .joinAsof(prune(l, required: nil), prune(r, required: nil), spec)

        case .union(let xs): return .union(xs.map { prune($0, required: required) })

        case .window(let c, let specs):
            guard let required else { return .window(prune(c, required: nil), specs) }
            let keep = specs.filter { required.contains($0.name) }
            var need = required.subtracting(specs.map(\.name))
            for s in (keep.isEmpty ? specs : keep) {
                need.formUnion(s.partitionBy)
                need.formUnion(s.orderBy.map(\.column))
                if let c = s.function.inputColumn { need.insert(c) }
            }
            if keep.count < specs.count && !keep.isEmpty { note("projection_pruning") }
            return .window(prune(c, required: need), keep.isEmpty ? specs : keep)

        case .explode(let c, let cols):
            let need = required.map { $0.union(cols) }
            return .explode(prune(c, required: need), cols)
        }
    }

    // MARK: - Common subexpression elimination

    /// Two projection outputs with the same canonical text are one output, referenced twice.
    private mutating func eliminateCommonSubexpressions(_ plan: LogicalPlan) -> LogicalPlan {
        guard enabled("expression_cse") else { return plan }
        let p = plan.mappingChildren { eliminateCommonSubexpressions($0) }
        func dedupe(_ ps: [NamedExpr]) -> [NamedExpr] {
            var seen = Set<String>(), out: [NamedExpr] = []
            for x in ps {
                let key = "\(x.name)=\(x.expr)"
                if seen.insert(key).inserted { out.append(x) }
            }
            return out
        }
        switch p {
        // `project` is *not* deduplicated: it names its output columns positionally, so
        // `select(col("a").alias("x"), col("a").alias("x"))` really is two columns and dropping one
        // would give the optimized plan a narrower schema than the unoptimized one. `with_columns`
        // does merge by name (its schema replaces a repeated name in place), so it can be.
        case .withColumns(let c, let ps):
            let d = dedupe(ps)
            if d.count < ps.count { note("expression_cse"); return .withColumns(c, d) }
            return p
        default: return p
        }
    }

    // MARK: - Join reordering

    /// An inner join is commutative, and `hashJoin` builds its table from the **right** side, so the
    /// smaller estimated input belongs there. When the estimate says the sides are the wrong way round
    /// they are swapped and a projection puts the columns back in the order the caller asked for.
    ///
    /// Only applied when the two schemas share no column names outside the keys, because a shared name
    /// is renamed by the join's suffix and the rename would follow the swap.
    private mutating func reorderJoins(_ plan: LogicalPlan) -> LogicalPlan {
        guard enabled("join_reorder") else { return plan }
        let p = plan.mappingChildren { reorderJoins($0) }
        guard case .join(let l, let r, let spec) = p, spec.how == .inner,
              let before = try? p.schema(), let ls = try? l.schema(), let rs = try? r.schema()
        else { return p }
        let shared = Set(ls.names).intersection(rs.names)
        guard shared.isSubset(of: Set(spec.leftOn).intersection(spec.rightOn)) else { return p }
        let lRows = stats.rows(l), rRows = stats.rows(r)
        guard rRows > lRows else { return p }
        note("join_reorder")
        let swapped = LogicalPlan.join(r, l, JoinSpec(leftOn: spec.rightOn, rightOn: spec.leftOn,
                                                     how: .inner, suffix: spec.suffix))
        return .project(swapped, before.names.map { NamedExpr($0, .column($0)) })
    }
}

// MARK: - Structural helpers

extension LogicalPlan {
    /// A copy with `f` applied to every child, leaving this node's own shape alone.
    func mappingChildren(_ f: (LogicalPlan) -> LogicalPlan) -> LogicalPlan {
        switch self {
        case .scan: return self
        case .filter(let c, let e): return .filter(f(c), e)
        case .project(let c, let ps): return .project(f(c), ps)
        case .withColumns(let c, let ps): return .withColumns(f(c), ps)
        case .aggregate(let c, let a): return .aggregate(f(c), a)
        case .groupAggregate(let c, let k, let a): return .groupAggregate(f(c), keys: k, aggregates: a)
        case .sort(let c, let k): return .sort(f(c), k)
        case .limit(let c, let n, let o): return .limit(f(c), count: n, offset: o)
        case .distinct(let c, let s): return .distinct(f(c), subset: s)
        case .join(let a, let b, let s): return .join(f(a), f(b), s)
        case .joinAsof(let a, let b, let s): return .joinAsof(f(a), f(b), s)
        case .union(let xs): return .union(xs.map(f))
        case .window(let c, let s): return .window(f(c), s)
        case .explode(let c, let s): return .explode(f(c), s)
        }
    }

    /// A copy with `f` applied to every expression this node owns (not its children's).
    func mappingExpressions(_ f: (Expr) -> Expr) -> LogicalPlan {
        switch self {
        case .filter(let c, let e): return .filter(c, f(e))
        case .project(let c, let ps): return .project(c, ps.map { NamedExpr($0.name, f($0.expr)) })
        case .withColumns(let c, let ps): return .withColumns(c, ps.map { NamedExpr($0.name, f($0.expr)) })
        case .aggregate(let c, let a):
            return .aggregate(c, a.map { ExprAggregate($0.op, $0.expr.map(f), name: $0.name) })
        case .groupAggregate(let c, let k, let a):
            return .groupAggregate(c, keys: k.map { NamedExpr($0.name, f($0.expr)) },
                                   aggregates: a.map { ExprAggregate($0.op, $0.expr.map(f), name: $0.name) })
        default: return self
        }
    }
}
