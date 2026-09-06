import Foundation

// The query plan: an optional filter, an optional group-by key, and one terminal (project or aggregate).
// `MetalRecordBatch.query(_:)` compiles the whole thing into one Metal kernel and runs it.

/// One aggregate in an `aggregate` or `group_by` terminal.
public struct ExprAggregate: Hashable, Sendable {
    public enum Op: String, Hashable, Sendable, CaseIterable { case sum, count, min, max, mean }
    public var op: Op
    /// The expression being aggregated. `nil` is only valid for `count`, which then counts rows.
    public var expr: Expr?
    /// Output name.
    public var name: String

    public init(_ op: Op, _ expr: Expr?, name: String) { self.op = op; self.expr = expr; self.name = name }

    var canonical: String { "(\(op.rawValue) \(quoteExprString(name))\(expr.map { " \($0)" } ?? ""))" }
}

/// A compiled-to-one-kernel query over a set of equal-length columns.
public struct ExprQuery: Hashable, Sendable {
    public enum Terminal: Hashable, Sendable {
        /// Materialise one output column per expression (compacted when there is a filter).
        case project([Projection])
        /// Whole-input reductions.
        case aggregate([ExprAggregate])
    }
    public struct Projection: Hashable, Sendable {
        public var name: String
        public var expr: Expr
        public init(name: String, expr: Expr) { self.name = name; self.expr = expr }
    }
    /// Rows are kept where this is true and not null (Arrow `filter` with the "drop" null behaviour).
    public var filter: Expr?
    /// A dense integer key expression in `0 ..< keyCount`, turning the terminal into a group-by.
    /// Rows whose key is null or out of range are skipped, as in `GroupBy`.
    public var groupKey: Expr?
    public var keyCount: Int = 0
    /// Name of the emitted key column.
    public var keyName: String = "key"
    public var terminal: Terminal

    public init(filter: Expr? = nil, groupKey: Expr? = nil, keyCount: Int = 0, keyName: String = "key",
                terminal: Terminal) {
        self.filter = filter; self.groupKey = groupKey; self.keyCount = keyCount
        self.keyName = keyName; self.terminal = terminal
    }

    /// The s-expression form; also the cache key prefix for the compiled kernel.
    public var canonical: String {
        var parts: [String] = []
        if let f = filter { parts.append("(filter \(f))") }
        if let k = groupKey { parts.append("(group_by \(keyCount) \(quoteExprString(keyName)) \(k))") }
        switch terminal {
        case .project(let ps):
            parts.append("(project " + ps.map { "(as \(quoteExprString($0.name)) \($0.expr))" }.joined(separator: " ") + ")")
        case .aggregate(let aggs):
            parts.append("(aggregate " + aggs.map(\.canonical).joined(separator: " ") + ")")
        }
        return "(query " + parts.joined(separator: " ") + ")"
    }
}

extension ExprQuery: CustomStringConvertible {
    public var description: String { canonical }
}

// MARK: - Builder DSL

/// Fluent builder: `query().filter(col("a") > 1).sum(col("b"))`.
public struct ExprQueryBuilder {
    var filterExpr: Expr?
    var groupKey: Expr?
    var keyCount = 0
    var keyName = "key"

    public init() {}

    public func filter(_ e: Expr) -> ExprQueryBuilder {
        var c = self; c.filterExpr = c.filterExpr.map { .binary(.and, $0, e) } ?? e; return c
    }
    public func groupBy(_ key: Expr, keyCount: Int, name: String = "key") -> ExprQueryBuilder {
        var c = self; c.groupKey = key; c.keyCount = keyCount; c.keyName = name; return c
    }
    public func project(_ projections: [ExprQuery.Projection]) -> ExprQuery {
        ExprQuery(filter: filterExpr, groupKey: groupKey, keyCount: keyCount, keyName: keyName,
                  terminal: .project(projections))
    }
    /// `project(["a": col("a") * 2])` with names taken in order.
    public func project(_ pairs: [(String, Expr)]) -> ExprQuery {
        project(pairs.map { ExprQuery.Projection(name: $0.0, expr: $0.1) })
    }
    /// Project bare column references, keeping their names.
    public func project(_ names: [String]) -> ExprQuery {
        project(names.map { ExprQuery.Projection(name: $0, expr: .column($0)) })
    }
    public func aggregate(_ aggs: [ExprAggregate]) -> ExprQuery {
        ExprQuery(filter: filterExpr, groupKey: groupKey, keyCount: keyCount, keyName: keyName,
                  terminal: .aggregate(aggs))
    }
    public func sum(_ e: Expr, name: String = "sum") -> ExprQuery { aggregate([ExprAggregate(.sum, e, name: name)]) }
    public func min(_ e: Expr, name: String = "min") -> ExprQuery { aggregate([ExprAggregate(.min, e, name: name)]) }
    public func max(_ e: Expr, name: String = "max") -> ExprQuery { aggregate([ExprAggregate(.max, e, name: name)]) }
    public func mean(_ e: Expr, name: String = "mean") -> ExprQuery { aggregate([ExprAggregate(.mean, e, name: name)]) }
    public func count(_ e: Expr? = nil, name: String = "count") -> ExprQuery {
        aggregate([ExprAggregate(.count, e, name: name)])
    }
}

/// Entry point for the builder DSL: `query().filter(...).sum(...)`.
public func query() -> ExprQueryBuilder { ExprQueryBuilder() }

extension Expr {
    public func sum(name: String = "sum") -> ExprQuery { query().sum(self, name: name) }
    public func min(name: String = "min") -> ExprQuery { query().min(self, name: name) }
    public func max(name: String = "max") -> ExprQuery { query().max(self, name: name) }
    public func mean(name: String = "mean") -> ExprQuery { query().mean(self, name: name) }
    public func count(name: String = "count") -> ExprQuery { query().count(self, name: name) }
}

// MARK: - Results

/// One scalar produced by an `aggregate` terminal.
public enum ExprScalar: Equatable, Sendable {
    case null
    case int(Int64)
    case uint(UInt64)
    case double(Double)

    public var asDouble: Double? {
        switch self {
        case .null: return nil
        case .int(let v): return Double(v)
        case .uint(let v): return Double(v)
        case .double(let v): return v
        }
    }
    public var asInt64: Int64? {
        switch self {
        case .null: return nil
        case .int(let v): return v
        case .uint(let v): return Int64(bitPattern: v)
        case .double(let v): return Int64(v)
        }
    }
}

/// What a query produced: named columns (project and group-by) or named scalars (aggregate).
public struct ExprQueryResult {
    public var names: [String] = []
    public var columns: [AnyMetalArray] = []
    public var scalarNames: [String] = []
    public var scalars: [ExprScalar] = []

    public subscript(name: String) -> AnyMetalArray? {
        names.firstIndex(of: name).map { columns[$0] }
    }
    public func scalar(_ name: String) -> ExprScalar? {
        scalarNames.firstIndex(of: name).map { scalars[$0] }
    }
    /// The single scalar of a one-aggregate query.
    public var onlyScalar: ExprScalar? { scalars.count == 1 ? scalars[0] : nil }
    /// The columns as a record batch (project and group-by terminals).
    public func recordBatch() throws -> MetalRecordBatch { try MetalRecordBatch(names: names, columns: columns) }
}

// MARK: - Entry points

extension MetalRecordBatch {
    /// Compiles `q` into one Metal kernel over this batch's columns and runs it.
    ///
    /// The kernel is cached on the canonical text of the query plus the column types, so the second call
    /// with the same shape pays no compilation. Joins an open `batch { }` like every other kernel.
    public func query(_ q: ExprQuery) throws -> ExprQueryResult {
        try ExprCompiler.run(q, names: names, columns: columns, context: firstContext ?? .shared)
    }

    var firstContext: MetalContext? { columns.first.map { $0.anyContext } }
}

extension AnyMetalArray {
    var anyContext: MetalContext {
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
        case .binary(let a): return a.context
        default: return .shared
        }
    }
}

/// Runs a query over named columns without building a `MetalRecordBatch` first.
public func runExprQuery(_ q: ExprQuery, names: [String], columns: [AnyMetalArray],
                         context: MetalContext = .shared) throws -> ExprQueryResult {
    try ExprCompiler.run(q, names: names, columns: columns, context: context)
}
