import Foundation

// `Router.explain`: the decision a routed call would get, without running it, with the table row it
// came from. It goes through `Router.route`, the rule every routed call uses, with the crossover
// table and mode in force on the calling thread. `python -m arrowmetal.router explain` prints it.

/// What `Router.explain` found for one (operation, value type, row count).
public struct RouteExplanation: Sendable {
    /// The operation as named to `explain` ("add", "multiply", "filter_where", ...).
    public let operation: String
    public let dtype: String
    /// The mode that decided (the per-thread override when one is set, else the process mode).
    public let mode: RouterMode
    public let decision: RouteDecision
    /// The table row consulted: multiply's own row for "multiply", else the routed operation's row.
    public let row: RouterRow
    public let table: RouterCrossovers
}

extension Router {
    /// Operation names `explain` accepts, and the routed operation each one is decided as.
    public static let explainOperations: [(name: String, op: RoutedOp)] = [
        ("sum", .sum), ("min", .min), ("max", .max), ("compare", .compare),
        ("add", .arithmetic), ("subtract", .arithmetic), ("multiply", .arithmetic), ("divide", .arithmetic),
        ("filter", .filter), ("filter_where", .filter), ("group_by_sum", .groupBySum),
    ]

    /// Value types `explain` accepts (Arrow names).
    public static let explainTypes = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                                      "float32", "float64"]

    /// The decision `operation` over `rows` values of `dtype` gets under the table and mode in force
    /// on this thread, outside a batch, on settled input. `keyCount` is the group-by's key count. Nil
    /// for an unknown operation or type. Does not record a last decision.
    public static func explain(operation: String, dtype: String, rows: Int, keyCount: Int = 1000) -> RouteExplanation? {
        guard let op = explainOperations.first(where: { $0.name == operation })?.op else { return nil }
        let t = dtype.lowercased()
        let facts: (String?, Bool, String)?
        switch t {
        case "int8": facts = explainFacts(operation, Int8.self, keyCount)
        case "uint8": facts = explainFacts(operation, UInt8.self, keyCount)
        case "int16": facts = explainFacts(operation, Int16.self, keyCount)
        case "uint16": facts = explainFacts(operation, UInt16.self, keyCount)
        case "int32": facts = explainFacts(operation, Int32.self, keyCount)
        case "uint32": facts = explainFacts(operation, UInt32.self, keyCount)
        case "int64": facts = explainFacts(operation, Int64.self, keyCount)
        case "uint64": facts = explainFacts(operation, UInt64.self, keyCount)
        case "float32": facts = explainFacts(operation, Float.self, keyCount)
        case "float64": facts = explainFacts(operation, Double.self, keyCount)
        default: facts = nil
        }
        guard let f = facts else { return nil }
        let (unavailable, measured, typeName) = f
        let table = self.table
        let thread = threadMode
        let m = thread ?? mode
        let isMultiply = operation == "multiply"
        let row = isMultiply ? table.multiply : table.rows[op]!
        let d = route(op, rows: rows, mode: m, threadOverride: thread != nil, crossover: row.crossover,
                      cpuPath: unavailable, measured: measured, pending: false, batching: false, typeName: typeName)
        return RouteExplanation(operation: operation, dtype: t, mode: m, decision: d, row: row, table: table)
    }

    /// (why there is no CPU path, whether the table covers the type, type name) as the routed entry
    /// points compute them for `T`.
    static func explainFacts<T: ArrowPrimitive>(_ operation: String, _: T.Type, _ keyCount: Int) -> (String?, Bool, String) {
        let unavailable: String?
        switch operation {
        case "min", "max": unavailable = RouterCPU.minMaxUnavailable(T.self)
        case "add": unavailable = RouterCPU.arithmeticUnavailable(T.self, .add)
        case "subtract": unavailable = RouterCPU.arithmeticUnavailable(T.self, .sub)
        case "multiply": unavailable = RouterCPU.arithmeticUnavailable(T.self, .mul)
        case "divide": unavailable = RouterCPU.arithmeticUnavailable(T.self, .div)
        case "filter_where": unavailable = RouterCPU.filterWhereUnavailable(T.self)
        case "group_by_sum": unavailable = RouterCPU.groupBySumUnavailable(keyCount: keyCount)
        default: unavailable = nil
        }
        return (unavailable, measured(T.self), typeName(T.self))
    }
}
