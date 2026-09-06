import Foundation

// The logical plan: what a query means, with nothing said about how it runs.
//
// `Expr` (Sources/ArrowMetal/Expr) is the element-wise half of a query — arithmetic, comparisons,
// null logic, string predicates — and `ExprCompiler` turns a whole tree of it into one Metal kernel.
// What it cannot express is everything that moves rows around: sorting, joining, grouping by keys it
// does not already have dense, windows, limits. This file is that layer.
//
// A `LogicalPlan` is a value. Building one runs nothing; it type-checks (`schema()` raises on an
// unknown column or an impossible operator) and it prints (`explain()`). `Optimizer.optimize` rewrites
// it, `PhysicalPlan.plan` chooses kernels for it, and `Executor` runs it.

// MARK: - Schema

/// One column of a plan's output.
public struct PlanField: Equatable, Sendable {
    public var name: String
    /// Arrow C Data Interface format string ("i", "u", "tsu:UTC", ...).
    public var format: String
    /// The expression compiler's type, when this column is one it can read.
    public var exprType: ExprType?
    public var nullable: Bool

    public init(name: String, format: String, exprType: ExprType?, nullable: Bool) {
        self.name = name; self.format = format; self.exprType = exprType; self.nullable = nullable
    }
    public init(name: String, _ t: ExprType, nullable: Bool = true) {
        self.init(name: name, format: t.arrowFormat, exprType: t, nullable: nullable)
    }
    /// True when a value of this column can be read by a fused kernel.
    public var isFusable: Bool { exprType != nil }
}

/// An ordered set of named, typed columns.
public struct PlanSchema: Equatable, Sendable {
    public var fields: [PlanField]
    public init(_ fields: [PlanField] = []) { self.fields = fields }

    public var names: [String] { fields.map(\.name) }
    public subscript(name: String) -> PlanField? { fields.first { $0.name == name } }
    public func contains(_ name: String) -> Bool { self[name] != nil }

    /// The subset of this schema the expression compiler can read, as it wants it.
    var exprColumns: [String: ExprColumnInfo] {
        var out: [String: ExprColumnInfo] = [:]
        for f in fields { if let t = f.exprType { out[f.name] = ExprColumnInfo(type: t, nullable: f.nullable) } }
        return out
    }

    static func of(_ batch: MetalRecordBatch) -> PlanSchema {
        PlanSchema(batch.names.enumerated().map { i, n in
            let c = batch.columns[i]
            return PlanField(name: n, format: c.arrowFormat, exprType: c.fusableExprType,
                             nullable: c.validityBuffer != nil)
        })
    }
}

extension AnyMetalArray {
    /// The `ExprType` a fused kernel would read this column as, or nil when it cannot read it.
    var fusableExprType: ExprType? {
        switch self {
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .uint8: return .uint8
        case .uint16: return .uint16
        case .uint32: return .uint32
        case .uint64: return .uint64
        case .float32: return .float32
        case .float64: return .float64
        case .boolean: return .boolean
        case .string: return .utf8
        case .extended(let e): return e.storage.fusableExprType
        default: return nil
        }
    }
}

// MARK: - Plan pieces

/// A named output expression: `col("a") * 2` as `"double_a"`.
public struct NamedExpr: Hashable, Sendable {
    public var name: String
    public var expr: Expr
    public init(_ name: String, _ expr: Expr) { self.name = name; self.expr = expr }
    /// True when this is a bare pass-through of a column of the same name.
    public var isIdentity: Bool { expr == .column(name) }
    public var description: String { isIdentity ? name : "\(expr) AS \(quoteExprString(name))" }
}

/// One key of a multi-column sort.
public struct SortKey: Hashable, Sendable {
    public var column: String
    public var descending: Bool
    public init(_ column: String, descending: Bool = false) { self.column = column; self.descending = descending }
    public var description: String { descending ? "\(column) DESC" : column }
}

/// The equi-join half of a join node.
public struct JoinSpec: Hashable, Sendable {
    public var leftOn: [String]
    public var rightOn: [String]
    public var how: JoinHow
    public var suffix: String
    public init(leftOn: [String], rightOn: [String], how: JoinHow, suffix: String = "_right") {
        self.leftOn = leftOn; self.rightOn = rightOn; self.how = how; self.suffix = suffix
    }
    public var description: String {
        let on = zip(leftOn, rightOn).map { $0 == $1 ? $0 : "\($0)=\($1)" }.joined(separator: ", ")
        return "\(how.rawValue.uppercased()) ON [\(on)]"
    }
}

/// The as-of join node's parameters.
public struct AsofSpec: Hashable, Sendable {
    public var leftOn: String
    public var rightOn: String
    public var by: [String]
    public var byRight: [String]
    public var strategy: AsofStrategy
    public var tolerance: Int64?
    public var suffix: String
    public init(leftOn: String, rightOn: String, by: [String] = [], byRight: [String]? = nil,
                strategy: AsofStrategy = .backward, tolerance: Int64? = nil, suffix: String = "_right") {
        self.leftOn = leftOn; self.rightOn = rightOn; self.by = by; self.byRight = byRight ?? by
        self.strategy = strategy; self.tolerance = tolerance; self.suffix = suffix
    }
    public var description: String {
        var s = "ASOF \(strategy.rawValue) ON [\(leftOn == rightOn ? leftOn : "\(leftOn)=\(rightOn)")]"
        if !by.isEmpty { s += " BY [\(by.joined(separator: ", "))]" }
        if let t = tolerance { s += " TOLERANCE \(t)" }
        return s
    }
}

/// The window functions the engine evaluates per partition.
public enum WindowFunction: Hashable, Sendable {
    case rowNumber
    case rank
    case denseRank
    /// `lag(column, n)` — the value `n` rows earlier in the partition's order.
    case lag(String, Int)
    case lead(String, Int)
    case cumSum(String)
    case rollingSum(String, Int)
    case rollingMean(String, Int)
    case rollingMin(String, Int)
    case rollingMax(String, Int)
    /// A whole-partition aggregate broadcast back to every row of the partition.
    case partitionAggregate(ExprAggregate.Op, String)

    var inputColumn: String? {
        switch self {
        case .rowNumber, .rank, .denseRank: return nil
        case .lag(let c, _), .lead(let c, _), .cumSum(let c),
             .rollingSum(let c, _), .rollingMean(let c, _), .rollingMin(let c, _), .rollingMax(let c, _):
            return c
        case .partitionAggregate(_, let c): return c
        }
    }
    var label: String {
        switch self {
        case .rowNumber: return "row_number()"
        case .rank: return "rank()"
        case .denseRank: return "dense_rank()"
        case .lag(let c, let n): return "lag(\(c), \(n))"
        case .lead(let c, let n): return "lead(\(c), \(n))"
        case .cumSum(let c): return "cum_sum(\(c))"
        case .rollingSum(let c, let w): return "rolling_sum(\(c), \(w))"
        case .rollingMean(let c, let w): return "rolling_mean(\(c), \(w))"
        case .rollingMin(let c, let w): return "rolling_min(\(c), \(w))"
        case .rollingMax(let c, let w): return "rolling_max(\(c), \(w))"
        case .partitionAggregate(let op, let c): return "\(op.rawValue)(\(c)) OVER PARTITION"
        }
    }
}

/// One window column: a function, a partitioning and an order.
public struct WindowSpec: Hashable, Sendable {
    public var name: String
    public var function: WindowFunction
    public var partitionBy: [String]
    public var orderBy: [SortKey]
    public init(name: String, function: WindowFunction, partitionBy: [String] = [], orderBy: [SortKey] = []) {
        self.name = name; self.function = function; self.partitionBy = partitionBy; self.orderBy = orderBy
    }
    public var description: String {
        var s = "\(function.label) AS \(name)"
        if !partitionBy.isEmpty { s += " OVER [\(partitionBy.joined(separator: ", "))]" }
        if !orderBy.isEmpty { s += " ORDER BY [\(orderBy.map(\.description).joined(separator: ", "))]" }
        return s
    }
}

/// A named table the plan reads. Holds the Metal-resident batch, so a plan is self-contained.
public final class PlanSource: @unchecked Sendable {
    public let name: String
    public let batch: MetalRecordBatch
    public let schema: PlanSchema
    public init(name: String, batch: MetalRecordBatch) {
        self.name = name; self.batch = batch; self.schema = PlanSchema.of(batch)
    }
}

// MARK: - The plan

public indirect enum LogicalPlan {
    /// Read `columns` of a source (nil means every column).
    case scan(PlanSource, columns: [String]?)
    /// Keep the rows where `predicate` is true and not null.
    case filter(LogicalPlan, Expr)
    /// Output exactly these columns.
    case project(LogicalPlan, [NamedExpr])
    /// Keep every input column and add or replace these.
    case withColumns(LogicalPlan, [NamedExpr])
    /// Whole-input reductions: one output row.
    case aggregate(LogicalPlan, [ExprAggregate])
    /// One output row per distinct key combination.
    case groupAggregate(LogicalPlan, keys: [NamedExpr], aggregates: [ExprAggregate])
    case sort(LogicalPlan, [SortKey])
    /// `head`: `count` rows starting at `offset`.
    case limit(LogicalPlan, count: Int, offset: Int)
    /// Distinct rows over `subset` (nil = every column), keeping the first occurrence.
    case distinct(LogicalPlan, subset: [String]?)
    case join(LogicalPlan, LogicalPlan, JoinSpec)
    case joinAsof(LogicalPlan, LogicalPlan, AsofSpec)
    /// Vertical concatenation of plans with identical schemas.
    case union([LogicalPlan])
    /// Adds one column per spec, aligned to the input rows.
    case window(LogicalPlan, [WindowSpec])
    /// One output row per element of the list column; the other columns repeat.
    case explode(LogicalPlan, [String])

    public var children: [LogicalPlan] {
        switch self {
        case .scan: return []
        case .filter(let c, _), .project(let c, _), .withColumns(let c, _), .aggregate(let c, _),
             .groupAggregate(let c, _, _), .sort(let c, _), .limit(let c, _, _), .distinct(let c, _),
             .window(let c, _), .explode(let c, _):
            return [c]
        case .join(let a, let b, _), .joinAsof(let a, let b, _): return [a, b]
        case .union(let xs): return xs
        }
    }
}

// MARK: - Schema inference and type checking

extension LogicalPlan {
    /// The output schema, raising on an unknown column, an untypeable expression or a mismatched union.
    public func schema() throws -> PlanSchema {
        switch self {
        case .scan(let src, let cols):
            guard let cols else { return src.schema }
            var out = PlanSchema()
            for c in cols {
                guard let f = src.schema[c] else {
                    throw ArrowMetalError.invalidArrowArray("scan \(src.name): no column named \(c)")
                }
                out.fields.append(f)
            }
            return out

        case .filter(let child, let pred):
            let s = try child.schema()
            let t = try s.type(of: pred, what: "filter predicate")
            guard t == .boolean else {
                throw ArrowMetalError.invalidArrowArray("filter predicate is \(t.rawValue), not boolean")
            }
            return s

        case .project(let child, let ps):
            let s = try child.schema()
            guard !ps.isEmpty else { throw ArrowMetalError.invalidArrowArray("select needs at least one column") }
            return PlanSchema(try ps.map { try s.field(for: $0) })

        case .withColumns(let child, let ps):
            var s = try child.schema()
            for p in ps {
                let f = try s.field(for: p)
                if let i = s.fields.firstIndex(where: { $0.name == p.name }) { s.fields[i] = f } else { s.fields.append(f) }
            }
            return s

        case .aggregate(let child, let aggs):
            let s = try child.schema()
            return PlanSchema(try aggs.map { try s.aggregateField($0) })

        case .groupAggregate(let child, let keys, let aggs):
            let s = try child.schema()
            guard !keys.isEmpty else { throw ArrowMetalError.invalidArrowArray("group_by needs at least one key") }
            var out = PlanSchema(try keys.map { try s.field(for: $0) })
            out.fields += try aggs.map { try s.aggregateField($0) }
            return out

        case .sort(let child, let keys):
            let s = try child.schema()
            for k in keys where !s.contains(k.column) {
                throw ArrowMetalError.invalidArrowArray("sort: no column named \(k.column)")
            }
            return s

        case .limit(let child, _, _), .distinct(let child, _):
            let s = try child.schema()
            if case .distinct(_, let subset) = self, let subset {
                for c in subset where !s.contains(c) {
                    throw ArrowMetalError.invalidArrowArray("unique: no column named \(c)")
                }
            }
            return s

        case .join(let l, let r, let spec):
            let ls = try l.schema(), rs = try r.schema()
            guard spec.leftOn.count == spec.rightOn.count, !spec.leftOn.isEmpty else {
                throw ArrowMetalError.invalidArrowArray("join: \(spec.leftOn.count) left keys against \(spec.rightOn.count) right keys")
            }
            for c in spec.leftOn where !ls.contains(c) {
                throw ArrowMetalError.invalidArrowArray("join: no left column named \(c)")
            }
            for c in spec.rightOn where !rs.contains(c) {
                throw ArrowMetalError.invalidArrowArray("join: no right column named \(c)")
            }
            if spec.how == .semi || spec.how == .anti { return ls }
            var out = ls
            // Every left column can be null after a right or full join; every right column after a
            // left or full one.
            if spec.how == .right || spec.how == .full {
                for i in out.fields.indices where !spec.leftOn.contains(out.fields[i].name) { out.fields[i].nullable = true }
            }
            for f in rs.fields {
                if let j = spec.rightOn.firstIndex(of: f.name), spec.leftOn[j] == f.name { continue }
                var g = f
                g.name = MetalRecordBatch.uniqueName(f.name, taken: out.names, suffix: spec.suffix)
                if spec.how == .left || spec.how == .full { g.nullable = true }
                out.fields.append(g)
            }
            return out

        case .joinAsof(let l, let r, let spec):
            let ls = try l.schema(), rs = try r.schema()
            guard ls.contains(spec.leftOn) else {
                throw ArrowMetalError.invalidArrowArray("join_asof: no left column named \(spec.leftOn)")
            }
            guard rs.contains(spec.rightOn) else {
                throw ArrowMetalError.invalidArrowArray("join_asof: no right column named \(spec.rightOn)")
            }
            var dropped = Set<String>()
            if spec.leftOn == spec.rightOn { dropped.insert(spec.rightOn) }
            for (i, n) in spec.byRight.enumerated() where i < spec.by.count && n == spec.by[i] { dropped.insert(n) }
            var out = ls
            for f in rs.fields where !dropped.contains(f.name) {
                var g = f
                g.name = MetalRecordBatch.uniqueName(f.name, taken: out.names, suffix: spec.suffix)
                g.nullable = true
                out.fields.append(g)
            }
            return out

        case .union(let plans):
            guard let head = plans.first else { throw ArrowMetalError.invalidArrowArray("concat needs at least one input") }
            let s = try head.schema()
            for p in plans.dropFirst() {
                let o = try p.schema()
                guard o.names == s.names else {
                    throw ArrowMetalError.invalidArrowArray("concat: column names differ (\(s.names) vs \(o.names))")
                }
            }
            return s

        case .window(let child, let specs):
            var s = try child.schema()
            for spec in specs {
                for c in spec.partitionBy where !s.contains(c) {
                    throw ArrowMetalError.invalidArrowArray("window: no partition column named \(c)")
                }
                for k in spec.orderBy where !s.contains(k.column) {
                    throw ArrowMetalError.invalidArrowArray("window: no order column named \(k.column)")
                }
                let f = try s.windowField(spec)
                if let i = s.fields.firstIndex(where: { $0.name == spec.name }) { s.fields[i] = f } else { s.fields.append(f) }
            }
            return s

        case .explode(let child, let cols):
            var s = try child.schema()
            for c in cols {
                guard let i = s.fields.firstIndex(where: { $0.name == c }) else {
                    throw ArrowMetalError.invalidArrowArray("explode: no column named \(c)")
                }
                // The element type is only known from the array itself; mark it opaque and nullable.
                s.fields[i] = PlanField(name: c, format: "?", exprType: nil, nullable: true)
            }
            return s
        }
    }
}

extension PlanSchema {
    /// The `ExprType` of an expression over this schema.
    func type(of e: Expr, what: String) throws -> ExprType {
        let em = ExprEmitter(schema: exprColumns)
        for n in e.referencedColumns {
            guard let f = self[n] else { throw ArrowMetalError.invalidArrowArray("\(what): no column named \(n)") }
            guard f.exprType != nil else {
                throw ArrowMetalError.unsupportedType("\(what): column \(n) is \(f.format), which the expression compiler does not read")
            }
        }
        guard let t = try em.typeOf(e) else { return .int64 }     // a bare untyped literal
        return t
    }

    func field(for p: NamedExpr) throws -> PlanField {
        // A bare column reference keeps its Arrow type even when the compiler cannot read it, so a
        // plan may carry a temporal, list or decimal column straight through a projection.
        if case .column(let n) = p.expr {
            guard var f = self[n] else { throw ArrowMetalError.invalidArrowArray("select: no column named \(n)") }
            f.name = p.name
            return f
        }
        let t = try type(of: p.expr, what: "select \(p.name)")
        return PlanField(name: p.name, t, nullable: exprMayBeNull(p.expr))
    }

    func aggregateField(_ a: ExprAggregate) throws -> PlanField {
        switch a.op {
        case .count: return PlanField(name: a.name, .int64, nullable: false)
        case .mean: return PlanField(name: a.name, .float64)
        case .sum:
            guard let e = a.expr else { throw ArrowMetalError.invalidArrowArray("sum needs an expression") }
            let t = try type(of: e, what: "sum \(a.name)")
            if t.isFloat { return PlanField(name: a.name, t == .float32 ? .float64 : .float64) }
            return PlanField(name: a.name, t.isSigned ? .int64 : .uint64)
        case .min, .max:
            guard let e = a.expr else { throw ArrowMetalError.invalidArrowArray("\(a.op.rawValue) needs an expression") }
            return PlanField(name: a.name, try type(of: e, what: "\(a.op.rawValue) \(a.name)"))
        }
    }

    func windowField(_ spec: WindowSpec) throws -> PlanField {
        switch spec.function {
        case .rowNumber, .rank, .denseRank: return PlanField(name: spec.name, .int32, nullable: false)
        case .lag(let c, _), .lead(let c, _), .rollingMin(let c, _), .rollingMax(let c, _), .cumSum(let c):
            guard var f = self[c] else { throw ArrowMetalError.invalidArrowArray("window: no column named \(c)") }
            f.name = spec.name; f.nullable = true
            return f
        case .rollingSum(let c, _):
            guard var f = self[c] else { throw ArrowMetalError.invalidArrowArray("window: no column named \(c)") }
            f.name = spec.name; f.nullable = true
            return f
        case .rollingMean: return PlanField(name: spec.name, .float64)
        case .partitionAggregate(let op, let c):
            guard self[c] != nil else { throw ArrowMetalError.invalidArrowArray("window: no column named \(c)") }
            return try aggregateField(ExprAggregate(op, .column(c), name: spec.name))
        }
    }

    /// Conservative nullability of an expression: only literals and `is_null`-shaped nodes are known
    /// non-null; everything reading a nullable column may be null.
    func exprMayBeNull(_ e: Expr) -> Bool {
        switch e {
        case .column(let n): return self[n]?.nullable ?? true
        case .int, .typedInt, .double, .typedDouble, .bool, .string: return false
        case .nullLiteral: return true
        case .isNull, .isValid, .isIn: return false
        case .binary(_, let a, let b): return exprMayBeNull(a) || exprMayBeNull(b)
        case .unary(_, let a), .cast(let a, _), .stringMatch(_, let a, _): return exprMayBeNull(a)
        case .ifElse(let c, let a, let b): return exprMayBeNull(c) || exprMayBeNull(a) || exprMayBeNull(b)
        case .coalesce(let xs): return xs.allSatisfy { exprMayBeNull($0) }
        case .fillNull(let a, let b): return exprMayBeNull(a) && exprMayBeNull(b)
        }
    }
}

// MARK: - Expression helpers the optimizer needs

extension Expr {
    /// The conjuncts of a chain of `and` / `and_kleene`, so a predicate can be pushed one piece at a time.
    public var conjuncts: [Expr] {
        if case .binary(let op, let a, let b) = self, op == .and || op == .andKleene {
            return a.conjuncts + b.conjuncts
        }
        return [self]
    }

    /// `and` of a list, or nil when the list is empty.
    public static func allOf(_ xs: [Expr]) -> Expr? {
        guard var acc = xs.first else { return nil }
        for x in xs.dropFirst() { acc = .binary(.and, acc, x) }
        return acc
    }

    /// Replaces every column reference by the expression bound to it, leaving unbound ones alone.
    public func substituting(_ map: [String: Expr]) -> Expr {
        switch self {
        case .column(let n): return map[n] ?? self
        case .int, .typedInt, .double, .typedDouble, .bool, .string, .nullLiteral: return self
        case .binary(let op, let a, let b): return .binary(op, a.substituting(map), b.substituting(map))
        case .unary(let op, let a): return .unary(op, a.substituting(map))
        case .cast(let a, let t): return .cast(a.substituting(map), t)
        case .ifElse(let c, let a, let b): return .ifElse(c.substituting(map), a.substituting(map), b.substituting(map))
        case .coalesce(let xs): return .coalesce(xs.map { $0.substituting(map) })
        case .fillNull(let a, let b): return .fillNull(a.substituting(map), b.substituting(map))
        case .isNull(let a): return .isNull(a.substituting(map))
        case .isValid(let a): return .isValid(a.substituting(map))
        case .isIn(let a, let xs): return .isIn(a.substituting(map), xs.map { $0.substituting(map) })
        case .stringMatch(let p, let a, let s): return .stringMatch(p, a.substituting(map), s)
        }
    }
}

// MARK: - Printing

extension LogicalPlan {
    /// The plan as an indented tree, root first, the way Polars prints one.
    public func describe(indent: Int = 0) -> String {
        let pad = String(repeating: "  ", count: indent)
        func node(_ label: String, _ kids: [LogicalPlan]) -> String {
            ([pad + label] + kids.map { $0.describe(indent: indent + 1) }).joined(separator: "\n")
        }
        switch self {
        case .scan(let src, let cols):
            let shown = cols ?? src.schema.names
            let total = src.schema.names.count
            return pad + "SCAN \(src.name) [\(shown.joined(separator: ", "))] "
                 + "\(shown.count)/\(total) columns, \(src.batch.length) rows"
        case .filter(let c, let p): return node("FILTER \(p)", [c])
        case .project(let c, let ps): return node("SELECT [\(ps.map(\.description).joined(separator: ", "))]", [c])
        case .withColumns(let c, let ps): return node("WITH_COLUMNS [\(ps.map(\.description).joined(separator: ", "))]", [c])
        case .aggregate(let c, let aggs): return node("AGGREGATE [\(aggs.map(\.canonical).joined(separator: ", "))]", [c])
        case .groupAggregate(let c, let keys, let aggs):
            return node("GROUP_BY [\(keys.map(\.description).joined(separator: ", "))] "
                        + "AGG [\(aggs.map(\.canonical).joined(separator: ", "))]", [c])
        case .sort(let c, let keys): return node("SORT BY [\(keys.map(\.description).joined(separator: ", "))]", [c])
        case .limit(let c, let n, let o): return node(o == 0 ? "LIMIT \(n)" : "SLICE \(o), \(n)", [c])
        case .distinct(let c, let s): return node("UNIQUE\(s.map { " [\($0.joined(separator: ", "))]" } ?? "")", [c])
        case .join(let a, let b, let spec): return node("JOIN \(spec.description)", [a, b])
        case .joinAsof(let a, let b, let spec): return node("JOIN_\(spec.description)", [a, b])
        case .union(let xs): return node("CONCAT (\(xs.count) inputs)", xs)
        case .window(let c, let specs): return node("WINDOW [\(specs.map(\.description).joined(separator: ", "))]", [c])
        case .explode(let c, let cols): return node("EXPLODE [\(cols.joined(separator: ", "))]", [c])
        }
    }
}

extension LogicalPlan: CustomStringConvertible {
    public var description: String { describe() }
}
