import Foundation

// Choosing kernels for a logical plan.
//
// The one decision that matters here is **fusion**: which parts of the plan become a single
// runtime-generated Metal kernel and which become their own dispatch. The rule is short, because the
// expression compiler decides what it can do:
//
// * An element-wise region — any tree of arithmetic, comparisons, null logic, casts and string
//   predicates — is one kernel.
// * A `filter` immediately under a `select` or an `aggregate` joins that kernel, so the predicate is
//   evaluated inside the same pass that computes the outputs and no boolean column is ever written.
// * A `select` output that is a bare column reference is not a kernel at all: the column is carried
//   through by reference.
// * Everything that moves rows — sort, join, group-by over keys that are not already dense, window,
//   explode, distinct — is its own physical node, because no kernel of the compiler produces it.
//
// A region stops being fusable the moment an output is `utf8` (the compiler will not materialise a
// string column), or a column it must read is temporal, decimal, list, struct or dictionary encoded.
// Those cases fall back to `maskFilter` — one fused kernel for the predicate, then the ordinary
// compaction kernels per column — which is exactly what the engine did before fusion existed.

/// A fused element-wise region: an optional predicate, the outputs one kernel computes, and the
/// outputs it only has to carry through.
public struct FusedProject {
    /// Evaluated inside the same kernel as `computed` when it is set.
    public var filter: Expr?
    /// Outputs the kernel evaluates.
    public var computed: [NamedExpr]
    /// Outputs that are a bare reference to an input column: no kernel touches them.
    public var passthrough: [NamedExpr]
    /// The output column order.
    public var order: [String]

    public init(filter: Expr?, computed: [NamedExpr], passthrough: [NamedExpr], order: [String]) {
        self.filter = filter; self.computed = computed; self.passthrough = passthrough; self.order = order
    }

    var label: String {
        var parts: [String] = []
        if !computed.isEmpty { parts.append(computed.map(\.description).joined(separator: ", ")) }
        if !passthrough.isEmpty { parts.append(passthrough.map(\.description).joined(separator: ", ")) }
        var s = "[\(parts.joined(separator: ", "))]"
        if let f = filter { s += " FILTER \(f)" }
        return s
    }
}

public indirect enum PhysicalPlan {
    /// The columns of a source, by reference. No kernel.
    case source(PlanSource, columns: [String])
    /// One fused kernel (or none, when every output is a pass-through).
    case fusedProject(PhysicalPlan, FusedProject)
    /// One fused kernel for the predicate, then the compaction kernels: the fallback when an output
    /// cannot be materialised by the expression compiler.
    case maskFilter(PhysicalPlan, Expr)
    /// Whole-input reductions in one kernel, with the predicate compiled in.
    case fusedAggregate(PhysicalPlan, filter: Expr?, aggregates: [ExprAggregate])
    /// Dense group ids from `GroupByKeys`, then the aggregates. `fusedAggregates` says whether the
    /// aggregates go through one fused group-by kernel or through `GroupBy`'s per-aggregate kernels.
    case hashAggregate(PhysicalPlan, keys: [NamedExpr], aggregates: [ExprAggregate], fusedAggregates: Bool)
    case sort(PhysicalPlan, [SortKey])
    /// A single-key sort feeding a limit: the top-k selection kernel, no full sort.
    case topK(PhysicalPlan, SortKey, k: Int)
    case slice(PhysicalPlan, offset: Int, count: Int)
    case distinct(PhysicalPlan, subset: [String]?)
    case hashJoin(PhysicalPlan, PhysicalPlan, JoinSpec)
    case asofJoin(PhysicalPlan, PhysicalPlan, AsofSpec)
    case concat([PhysicalPlan])
    case window(PhysicalPlan, [WindowSpec])
    case explode(PhysicalPlan, [String])
    /// Reorder and rename only.
    case rename(PhysicalPlan, [NamedExpr])
}

public enum PhysicalPlanner {

    /// Lowers a logical plan to physical operators, fusing what the expression compiler can fuse.
    public static func plan(_ logical: LogicalPlan) throws -> PhysicalPlan {
        switch logical {
        case .scan(let src, let cols):
            return .source(src, columns: cols ?? src.schema.names)

        case .project(let child, let ps):
            return try projection(child, ps, schemaOf: try child.schema())

        case .withColumns(let child, let ps):
            let s = try child.schema()
            var outs = s.names.map { NamedExpr($0, .column($0)) }
            for p in ps {
                if let i = outs.firstIndex(where: { $0.name == p.name }) { outs[i] = p } else { outs.append(p) }
            }
            return try projection(child, outs, schemaOf: s)

        case .filter(let child, let pred):
            let s = try child.schema()
            // A filter with nothing above it still fuses when every column it must carry is one the
            // compiler can materialise; otherwise the predicate becomes a mask.
            let outs = s.names.map { NamedExpr($0, .column($0)) }
            if canFuse(pred, outs, s) {
                return .fusedProject(try plan(child),
                                     FusedProject(filter: pred, computed: outs, passthrough: [], order: s.names))
            }
            return .maskFilter(try plan(child), pred)

        case .aggregate(let child, let aggs):
            if case .filter(let base, let pred) = child, try canFuseAggregate(aggs, pred, base.schema()) {
                return .fusedAggregate(try plan(base), filter: pred, aggregates: aggs)
            }
            return .fusedAggregate(try plan(child), filter: nil, aggregates: aggs)

        case .groupAggregate(let child, let keys, let aggs):
            let s = try child.schema()
            let fused = try aggs.allSatisfy { try fusableGroupAggregate($0, s) }
            return .hashAggregate(try plan(child), keys: keys, aggregates: aggs, fusedAggregates: fused)

        case .sort(let child, let keys):
            return .sort(try plan(child), keys)

        case .limit(let child, let n, let o):
            // `sort` then `head` is a top-k selection when there is one key and no offset.
            if o == 0, case .sort(let base, let keys) = child, keys.count == 1, n > 0, n <= 1024 {
                let s = try base.schema()
                if let f = s[keys[0].column], f.exprType?.isNumeric == true {
                    return .slice(.topK(try plan(base), keys[0], k: n), offset: 0, count: n)
                }
            }
            return .slice(try plan(child), offset: o, count: n)

        case .distinct(let child, let subset):
            return .distinct(try plan(child), subset: subset)

        case .join(let l, let r, let spec):
            return .hashJoin(try plan(l), try plan(r), spec)

        case .joinAsof(let l, let r, let spec):
            return .asofJoin(try plan(l), try plan(r), spec)

        case .union(let xs):
            return .concat(try xs.map { try plan($0) })

        case .window(let child, let specs):
            return .window(try plan(child), specs)

        case .explode(let child, let cols):
            return .explode(try plan(child), cols)
        }
    }

    /// A projection, fusing the filter beneath it when the expression compiler can carry every output.
    private static func projection(_ child: LogicalPlan, _ ps: [NamedExpr], schemaOf s: PlanSchema) throws -> PhysicalPlan {
        if case .filter(let base, let pred) = child {
            let bs = try base.schema()
            if canFuse(pred, ps, bs) {
                return .fusedProject(try plan(base),
                                     FusedProject(filter: pred, computed: ps, passthrough: [], order: ps.map(\.name)))
            }
            return split(.maskFilter(try plan(base), pred), ps, bs)
        }
        return split(try plan(child), ps, s)
    }

    /// Splits outputs into "a kernel computes it" and "carry the column through".
    private static func split(_ input: PhysicalPlan, _ ps: [NamedExpr], _ s: PlanSchema) -> PhysicalPlan {
        var computed: [NamedExpr] = [], passthrough: [NamedExpr] = []
        for p in ps {
            if case .column = p.expr { passthrough.append(p) } else { computed.append(p) }
        }
        if computed.isEmpty { return .rename(input, ps) }
        return .fusedProject(input, FusedProject(filter: nil, computed: computed,
                                                 passthrough: passthrough, order: ps.map(\.name)))
    }

    /// True when one fused filter+project kernel can produce every one of these outputs.
    static func canFuse(_ pred: Expr, _ ps: [NamedExpr], _ s: PlanSchema) -> Bool {
        let cols = s.exprColumns
        var referenced = Set(pred.referencedColumns)
        for p in ps { referenced.formUnion(p.expr.referencedColumns) }
        for c in referenced where cols[c] == nil { return false }
        let em = ExprEmitter(schema: cols)
        guard let pt = try? em.typeOf(pred), pt == .boolean else { return false }
        for p in ps {
            guard let t = try? em.typeOf(p.expr) else { return false }
            // A `utf8` output cannot be materialised, so a string column can only be read, not carried.
            if t == .utf8 { return false }
        }
        return true
    }

    static func canFuseAggregate(_ aggs: [ExprAggregate], _ pred: Expr, _ s: PlanSchema) -> Bool {
        let cols = s.exprColumns
        var referenced = Set(pred.referencedColumns)
        for a in aggs { referenced.formUnion(a.expr?.referencedColumns ?? []) }
        for c in referenced where cols[c] == nil { return false }
        let em = ExprEmitter(schema: cols)
        return (try? em.typeOf(pred)) == .boolean
    }

    /// The fused group-by kernel finishes its sums with 32-bit atomics, which bounds the types it
    /// takes (see docs/EXPR.md). Everything else goes through `GroupBy`'s own kernels instead.
    static func fusableGroupAggregate(_ a: ExprAggregate, _ s: PlanSchema) throws -> Bool {
        guard let e = a.expr else { return a.op == .count }
        let cols = s.exprColumns
        for c in e.referencedColumns where cols[c] == nil { return false }
        guard let t = try? ExprEmitter(schema: cols).typeOf(e) else { return false }
        switch a.op {
        case .count: return true
        case .sum, .mean: return t.isInteger || t == .float32
        case .min, .max: return (t.isInteger && t.bitWidth <= 32) || t == .float32
        }
    }
}

// MARK: - Printing

extension PhysicalPlan {
    public func describe(indent: Int = 0) -> String {
        let pad = String(repeating: "  ", count: indent)
        func node(_ label: String, _ kids: [PhysicalPlan]) -> String {
            ([pad + label] + kids.map { $0.describe(indent: indent + 1) }).joined(separator: "\n")
        }
        switch self {
        case .source(let src, let cols):
            return pad + "SOURCE \(src.name) [\(cols.joined(separator: ", "))] "
                 + "\(cols.count)/\(src.schema.names.count) columns, \(src.batch.length) rows"
        case .fusedProject(let c, let f):
            let kind = f.computed.isEmpty ? "CARRY" : (f.filter == nil ? "FUSED-PROJECT" : "FUSED-FILTER-PROJECT")
            return node("\(kind) \(f.label)", [c])
        case .maskFilter(let c, let p): return node("MASK-FILTER \(p)", [c])
        case .fusedAggregate(let c, let f, let aggs):
            return node("FUSED-AGGREGATE [\(aggs.map(\.canonical).joined(separator: ", "))]"
                        + (f.map { " FILTER \($0)" } ?? ""), [c])
        case .hashAggregate(let c, let keys, let aggs, let fused):
            return node("HASH-AGGREGATE [\(keys.map(\.description).joined(separator: ", "))] "
                        + "AGG [\(aggs.map(\.canonical).joined(separator: ", "))] "
                        + (fused ? "(one fused kernel)" : "(per-aggregate kernels)"), [c])
        case .sort(let c, let keys): return node("LEXSORT [\(keys.map(\.description).joined(separator: ", "))]", [c])
        case .topK(let c, let key, let k): return node("TOP-K \(k) BY \(key.description)", [c])
        case .slice(let c, let o, let n): return node(o == 0 ? "HEAD \(n)" : "SLICE \(o), \(n)", [c])
        case .distinct(let c, let s): return node("DISTINCT\(s.map { " [\($0.joined(separator: ", "))]" } ?? "")", [c])
        case .hashJoin(let a, let b, let s): return node("HASH-JOIN \(s.description)", [a, b])
        case .asofJoin(let a, let b, let s): return node("ASOF-JOIN \(s.description)", [a, b])
        case .concat(let xs): return node("CONCAT (\(xs.count) inputs)", xs)
        case .window(let c, let specs): return node("WINDOW [\(specs.map(\.description).joined(separator: ", "))]", [c])
        case .explode(let c, let cols): return node("EXPLODE [\(cols.joined(separator: ", "))]", [c])
        case .rename(let c, let ps): return node("PROJECT [\(ps.map(\.description).joined(separator: ", "))]", [c])
        }
    }
}

extension PhysicalPlan: CustomStringConvertible {
    public var description: String { describe() }
}
