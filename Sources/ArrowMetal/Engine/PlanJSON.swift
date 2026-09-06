import Foundation

// The serialised form of a logical plan.
//
// Expressions already have a text form — the s-expression grammar of `docs/EXPR.md` — and it is the
// cache key the fused compiler uses, so a plan reuses it verbatim for every expression it contains and
// only has to spell the *operators*. JSON is the wrapper, because a plan is a tree of records with
// optional fields, which is exactly the shape JSON is good at and exactly the shape an s-expression
// with positional arguments is bad at.
//
// This is what the C ABI (`am_plan_run`, `am_plan_explain`) takes and what `python/arrowmetal/lazy.py`
// produces. One text in, one record batch out, with the optimizer running in between.
//
//     {"op": "sort", "by": [["total", true]], "input":
//       {"op": "group_by", "keys": [["region", "(col \"region\")"]],
//        "aggs": [["sum", "total", "(col \"amount\")"]], "input":
//         {"op": "filter", "predicate": "(gt (col \"amount\") (int 100))",
//          "input": {"op": "scan", "source": "sales"}}}}
//
// Every node:
//
// | op | fields |
// |---|---|
// | `scan` | `source`, `columns`? |
// | `filter` | `input`, `predicate` |
// | `select` / `with_columns` | `input`, `exprs`: `[[name, sexpr], ...]` |
// | `aggregate` | `input`, `aggs`: `[[op, name, sexpr?], ...]` |
// | `group_by` | `input`, `keys`: `[[name, sexpr], ...]`, `aggs` |
// | `sort` | `input`, `by`: `[[column, descending], ...]` |
// | `limit` | `input`, `count`, `offset`? |
// | `unique` | `input`, `subset`? |
// | `join` | `left`, `right`, `left_on`, `right_on`, `how`, `suffix`? |
// | `join_asof` | `left`, `right`, `left_on`, `right_on`, `by`?, `by_right`?, `strategy`?, `tolerance`?, `suffix`? |
// | `concat` | `inputs` |
// | `window` | `input`, `specs`: `[{name, fn, column?, n?, partition_by?, order_by?}, ...]` |
// | `explode` | `input`, `columns` |

public enum PlanJSON {

    /// Parses a plan, resolving `scan` nodes against the sources given by name.
    public static func parse(_ text: String, sources: [String: PlanSource]) throws -> LogicalPlan {
        guard let data = text.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ArrowMetalError.invalidArrowArray("plan: not a JSON object")
        }
        return try node(root, sources)
    }

    static func node(_ o: [String: Any], _ sources: [String: PlanSource]) throws -> LogicalPlan {
        guard let op = o["op"] as? String else { throw ArrowMetalError.invalidArrowArray("plan: node has no \"op\"") }
        func input(_ key: String = "input") throws -> LogicalPlan {
            guard let d = o[key] as? [String: Any] else {
                throw ArrowMetalError.invalidArrowArray("plan: \(op) has no \"\(key)\"")
            }
            return try node(d, sources)
        }
        func strings(_ key: String) -> [String] { (o[key] as? [Any])?.compactMap { $0 as? String } ?? [] }
        func namedExprs(_ key: String) throws -> [NamedExpr] {
            guard let xs = o[key] as? [Any] else { return [] }
            return try xs.map { x in
                guard let pair = x as? [Any], pair.count == 2, let n = pair[0] as? String, let t = pair[1] as? String else {
                    throw ArrowMetalError.invalidArrowArray("plan: \(op).\(key) entries are [name, sexpr] pairs")
                }
                return NamedExpr(n, try Expr(text: t))
            }
        }
        func aggregates() throws -> [ExprAggregate] {
            guard let xs = o["aggs"] as? [Any] else { return [] }
            return try xs.map { x in
                guard let t = x as? [Any], t.count >= 2, let opName = t[0] as? String, let n = t[1] as? String,
                      let aggOp = ExprAggregate.Op(rawValue: opName) else {
                    throw ArrowMetalError.invalidArrowArray("plan: aggs entries are [op, name, sexpr?]")
                }
                var e: Expr? = nil
                if t.count > 2, let s = t[2] as? String, !s.isEmpty { e = try Expr(text: s) }
                return ExprAggregate(aggOp, e, name: n)
            }
        }
        func sortKeys(_ key: String) -> [SortKey] {
            guard let xs = o[key] as? [Any] else { return [] }
            return xs.compactMap { x in
                guard let pair = x as? [Any], let c = pair.first as? String else { return nil }
                let d = pair.count > 1 ? ((pair[1] as? Bool) ?? ((pair[1] as? NSNumber)?.boolValue ?? false)) : false
                return SortKey(c, descending: d)
            }
        }

        switch op {
        case "scan":
            guard let name = o["source"] as? String, let src = sources[name] else {
                throw ArrowMetalError.invalidArrowArray("plan: unknown source \"\(o["source"] as? String ?? "?")\"")
            }
            let cols = o["columns"] as? [Any]
            return .scan(src, columns: cols.map { $0.compactMap { $0 as? String } })

        case "filter":
            guard let p = o["predicate"] as? String else { throw ArrowMetalError.invalidArrowArray("plan: filter needs a predicate") }
            return .filter(try input(), try Expr(text: p))

        case "select": return .project(try input(), try namedExprs("exprs"))
        case "with_columns": return .withColumns(try input(), try namedExprs("exprs"))
        case "aggregate": return .aggregate(try input(), try aggregates())
        case "group_by": return .groupAggregate(try input(), keys: try namedExprs("keys"), aggregates: try aggregates())
        case "sort": return .sort(try input(), sortKeys("by"))
        case "limit":
            let n = (o["count"] as? NSNumber)?.intValue ?? Int.max
            let off = (o["offset"] as? NSNumber)?.intValue ?? 0
            return .limit(try input(), count: n, offset: off)
        case "unique":
            let subset = (o["subset"] as? [Any])?.compactMap { $0 as? String }
            return .distinct(try input(), subset: subset)

        case "join":
            guard let howName = o["how"] as? String, let how = JoinHow(rawValue: howName) else {
                throw ArrowMetalError.invalidArrowArray("plan: join \"how\" must be one of \(JoinHow.allCases.map(\.rawValue))")
            }
            return .join(try input("left"), try input("right"),
                         JoinSpec(leftOn: strings("left_on"), rightOn: strings("right_on"), how: how,
                                  suffix: (o["suffix"] as? String) ?? "_right"))

        case "join_asof":
            guard let l = o["left_on"] as? String, let r = o["right_on"] as? String else {
                throw ArrowMetalError.invalidArrowArray("plan: join_asof needs left_on and right_on")
            }
            let by = strings("by")
            let byRight = o["by_right"] == nil ? by : strings("by_right")
            let strat = AsofStrategy(rawValue: (o["strategy"] as? String) ?? "backward") ?? .backward
            let tol = (o["tolerance"] as? NSNumber)?.int64Value
            return .joinAsof(try input("left"), try input("right"),
                             AsofSpec(leftOn: l, rightOn: r, by: by, byRight: byRight, strategy: strat,
                                      tolerance: tol, suffix: (o["suffix"] as? String) ?? "_right"))

        case "concat":
            guard let xs = o["inputs"] as? [Any] else { throw ArrowMetalError.invalidArrowArray("plan: concat needs inputs") }
            return .union(try xs.map { x in
                guard let d = x as? [String: Any] else { throw ArrowMetalError.invalidArrowArray("plan: concat input") }
                return try node(d, sources)
            })

        case "window":
            guard let xs = o["specs"] as? [Any] else { throw ArrowMetalError.invalidArrowArray("plan: window needs specs") }
            var specs: [WindowSpec] = []
            for x in xs {
                guard let d = x as? [String: Any], let name = d["name"] as? String, let fn = d["fn"] as? String else {
                    throw ArrowMetalError.invalidArrowArray("plan: window spec needs name and fn")
                }
                let column = d["column"] as? String
                let k = (d["n"] as? NSNumber)?.intValue ?? 1
                let f: WindowFunction
                switch fn {
                case "row_number": f = .rowNumber
                case "rank": f = .rank
                case "dense_rank": f = .denseRank
                case "lag": f = .lag(try need(column, fn), k)
                case "lead": f = .lead(try need(column, fn), k)
                case "cum_sum": f = .cumSum(try need(column, fn))
                case "rolling_sum": f = .rollingSum(try need(column, fn), k)
                case "rolling_mean": f = .rollingMean(try need(column, fn), k)
                case "rolling_min": f = .rollingMin(try need(column, fn), k)
                case "rolling_max": f = .rollingMax(try need(column, fn), k)
                case "sum", "min", "max", "mean", "count":
                    guard let aggOp = ExprAggregate.Op(rawValue: fn) else {
                        throw ArrowMetalError.invalidArrowArray("plan: window fn \(fn)")
                    }
                    f = .partitionAggregate(aggOp, try need(column, fn))
                default:
                    throw ArrowMetalError.invalidArrowArray("plan: unknown window function \"\(fn)\"")
                }
                let partitionBy = (d["partition_by"] as? [Any])?.compactMap { $0 as? String } ?? []
                var order: [SortKey] = []
                if let ob = d["order_by"] as? [Any] {
                    order = ob.compactMap { x in
                        guard let pair = x as? [Any], let c = pair.first as? String else { return nil }
                        let desc = pair.count > 1 ? ((pair[1] as? Bool) ?? ((pair[1] as? NSNumber)?.boolValue ?? false)) : false
                        return SortKey(c, descending: desc)
                    }
                }
                specs.append(WindowSpec(name: name, function: f, partitionBy: partitionBy, orderBy: order))
            }
            return .window(try input(), specs)

        case "explode": return .explode(try input(), strings("columns"))

        default:
            throw ArrowMetalError.invalidArrowArray("plan: unknown operator \"\(op)\"")
        }
    }

    private static func need(_ c: String?, _ fn: String) throws -> String {
        guard let c else { throw ArrowMetalError.invalidArrowArray("plan: window function \(fn) needs a \"column\"") }
        return c
    }

    /// Parses, optimizes and runs a plan.
    public static func run(_ text: String, sources: [String: PlanSource], optimize: Bool = true) throws -> MetalRecordBatch {
        try LazyFrame(try parse(text, sources: sources)).collect(optimize: optimize)
    }

    /// Parses and explains a plan without running it.
    public static func explain(_ text: String, sources: [String: PlanSource], optimize: Bool = true) throws -> String {
        try LazyFrame(try parse(text, sources: sources)).explain(optimized: optimize)
    }
}
