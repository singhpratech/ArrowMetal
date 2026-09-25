import Foundation

// CPU/GPU router. Design and rules: docs/DESIGN.md, section "CPU/GPU router".
//
// The routed operations call `Router.decide` once, at the top of their public entry point, before
// anything is encoded. The answer is `Router.route`, a pure function of the operation, the value
// type, the row count, the mode, the input's batch state and the crossover table in force (the
// shipped `RouterTable`, generated from the measured sweep by Benchmarks/router_table.py, or a
// per-machine table from `python -m arrowmetal.router calibrate`, loaded once at first use or by
// `Router.loadTable`). Nothing is timed or estimated at call time, and no state is kept across calls
// beyond the table, the process mode, a per-thread override and the per-thread last decision (which
// exists only so callers can see what happened).

/// The operations the router can send to the CPU.
public enum RoutedOp: String, CaseIterable, Sendable {
    case sum, min, max, compare, arithmetic, filter
    /// `GroupBy.sum` over integer values with at most `Router.groupBySumMaxKeys` keys.
    case groupBySum = "group_by_sum"

    /// Number used by the C ABI (`am_router_crossover`, `am_router_last_op`).
    public var code: Int32 { Int32(RoutedOp.allCases.firstIndex(of: self)!) }
}

/// Where an operation ran.
public enum RoutePath: String, Sendable { case gpu, cpu }

/// Process-wide or per-thread routing policy.
public enum RouterMode: String, CaseIterable, Sendable {
    /// Use the crossover table (the default).
    case auto
    /// Always the GPU kernel (the behaviour before the router existed).
    case gpu
    /// The CPU loop wherever one exists for the operation and type.
    case cpu
}

/// Why the router chose a path.
public enum RouteReason: Equatable, Sendable {
    /// `auto`: the input is below the table's crossover for this operation.
    case belowCrossover(crossover: Int)
    /// `auto`: the input is at or above the crossover.
    case atOrAboveCrossover(crossover: Int)
    /// `auto`: the table has no measured crossover for this value type, so the GPU stays.
    case notMeasuredForType(String)
    /// A mode other than `auto` was in force (`thread` says whether it was the per-thread override).
    case forced(RouterMode, thread: Bool)
    /// A batch is open: queued GPU work may be producing this input and the floor is shared.
    case batchOpen
    /// The input's length is still being decided by GPU work in the open batch.
    case pendingInput
    /// The operation has no CPU loop for this input; the GPU runs it, as before the router.
    case noCPUPath(String)

    public var description: String {
        switch self {
        case .belowCrossover(let c): return "below the \(c)-row crossover"
        case .atOrAboveCrossover(let c): return "at or above the \(c)-row crossover"
        case .notMeasuredForType(let t): return "no measured crossover for \(t)"
        case .forced(let m, let thread): return "forced \(m.rawValue) by the \(thread ? "per-thread override" : "process mode")"
        case .batchOpen: return "batch open"
        case .pendingInput: return "pending input"
        case .noCPUPath(let why): return "no alternative: \(why)"
        }
    }
}

/// One routing decision, as recorded for `Router.lastDecision`.
public struct RouteDecision: Equatable, Sendable, CustomStringConvertible {
    public let op: RoutedOp
    public let path: RoutePath
    public let reason: RouteReason
    public let rows: Int
    public var isCPU: Bool { path == .cpu }
    public var description: String { "\(op.rawValue): \(path.rawValue), \(reason.description) (\(rows) rows)" }
}

public enum Router {
    /// Largest key count for which `GroupBy.sum` has a CPU path: the low-cardinality shape the table
    /// was measured on (1,000 keys), and the GPU's own threadgroup-private table limit.
    public static let groupBySumMaxKeys = 1024

    // MARK: process mode

    private static let modeLock = NSLock()
    private static var _mode: RouterMode = initialMode()

    /// `ARROWMETAL_ROUTER=gpu|cpu|auto`, read once; anything else (or unset) is `auto`.
    static func initialMode() -> RouterMode {
        guard let v = ProcessInfo.processInfo.environment["ARROWMETAL_ROUTER"]?.lowercased() else { return .auto }
        return RouterMode(rawValue: v) ?? .auto
    }

    /// The process-wide policy. Starts from `ARROWMETAL_ROUTER`, `auto` when unset.
    public static var mode: RouterMode {
        get { modeLock.lock(); defer { modeLock.unlock() }; return _mode }
        set { modeLock.lock(); _mode = newValue; modeLock.unlock() }
    }

    /// Whether `ARROWMETAL_ROUTER` was set in the environment (test harnesses respect it).
    public static var environmentMode: RouterMode? {
        ProcessInfo.processInfo.environment["ARROWMETAL_ROUTER"].flatMap { RouterMode(rawValue: $0.lowercased()) }
    }

    // MARK: per-thread state

    final class ThreadState {
        var override: RouterMode?
        var last: RouteDecision?
    }

    private static let stateKey: pthread_key_t = {
        var k = pthread_key_t()
        pthread_key_create(&k) { raw in Unmanaged<ThreadState>.fromOpaque(raw).release() }
        return k
    }()

    static var threadState: ThreadState {
        if let raw = pthread_getspecific(stateKey) { return Unmanaged<ThreadState>.fromOpaque(raw).takeUnretainedValue() }
        let s = ThreadState()
        pthread_setspecific(stateKey, Unmanaged.passRetained(s).toOpaque())
        return s
    }

    /// The per-thread override, nil when the process mode applies. Prefer `withMode`, which restores it.
    public static var threadMode: RouterMode? {
        get { threadState.override }
        set { threadState.override = newValue }
    }

    /// Runs `body` with `mode` in force on this thread: the per-call override.
    ///
    ///     let s = try Router.withMode(.cpu) { try column.sum() }
    public static func withMode<R>(_ mode: RouterMode, _ body: () throws -> R) rethrows -> R {
        let st = threadState
        let saved = st.override
        st.override = mode
        defer { st.override = saved }
        return try body()
    }

    /// The last decision a routed operation made on this thread (nil before the first one).
    public static var lastDecision: RouteDecision? { threadState.last }

    /// Forgets this thread's last decision.
    public static func clearLastDecision() { threadState.last = nil }

    // MARK: the table

    private static let tableLock = NSLock()
    private static var _tableLoadError: String?
    private static var _table: RouterCrossovers = {
        let env = ProcessInfo.processInfo.environment
        let (t, err) = RouterCrossovers.initial(environment: env, home: env["HOME"] ?? NSHomeDirectory(),
                                                chip: RouterCrossovers.machineChip)
        _tableLoadError = err
        return t
    }()

    /// The crossover table in force. At first use it is `ARROWMETAL_ROUTER_TABLE` (a path, or `shipped`),
    /// else this machine's `~/.arrowmetal/router/<chip id>.json` when one exists, else the shipped table.
    public static var table: RouterCrossovers {
        tableLock.lock(); defer { tableLock.unlock() }
        return _table
    }

    /// Why the table named at first use (environment or per-machine file) could not be loaded, if it
    /// could not; the shipped table is in force then.
    public static var tableLoadError: String? {
        tableLock.lock(); defer { tableLock.unlock() }
        _ = _table
        return _tableLoadError
    }

    /// Replaces the table in force with the JSON table at `path`; on error the table is unchanged.
    public static func loadTable(path: String) throws {
        let t = try RouterCrossovers.load(path: path)
        tableLock.lock(); _ = _table; _table = t; _tableLoadError = nil; tableLock.unlock()
    }

    /// Puts the shipped table back in force.
    public static func useShippedTable() {
        tableLock.lock(); _ = _table; _table = .shipped; _tableLoadError = nil; tableLock.unlock()
    }

    /// The crossover the table in force holds for `op`, in rows. For `.arithmetic` this is the
    /// add/subtract row; `crossoverRows(arithmetic:)` gives the row a particular arithmetic operation uses.
    public static func crossoverRows(_ op: RoutedOp) -> Int { table.crossover(op) }

    /// The crossover `auto` uses for one arithmetic operation: multiply has its own row, add and
    /// subtract share the arithmetic row. Divide is not routed and returns nil.
    public static func crossoverRows(arithmetic op: ArithmeticOp) -> Int? { table.crossover(arithmetic: op) }

    // MARK: the decision

    /// Value-type class the table was measured on. Integer columns of every width use the int64 rows;
    /// floating-point columns have no measured crossover, so `auto` keeps them on the GPU.
    @inline(__always) static func measured<T: ArrowPrimitive>(_: T.Type) -> Bool { !T.isFloatingPoint }

    /// The decision for one call. `cpuPath` is nil when a CPU loop exists for this input, or the reason
    /// it does not. `pending` and `batching` describe the input and the calling thread. `crossover`
    /// replaces the table's row for `op` (multiply's own row).
    static func decide(_ op: RoutedOp, rows: Int, cpuPath unavailable: String?, measured: Bool,
                       pending: Bool, batching: @autoclosure () -> Bool, typeName: @autoclosure () -> String,
                       crossover: Int? = nil) -> RouteDecision {
        let st = threadState
        let thread = st.override
        let d = route(op, rows: rows, mode: thread ?? mode, threadOverride: thread != nil,
                      crossover: crossover ?? crossoverRows(op), cpuPath: unavailable, measured: measured,
                      pending: pending, batching: batching(), typeName: typeName())
        st.last = d
        return d
    }

    /// The routing rule itself, a pure function: the same arguments give the same decision on every
    /// call, and nothing is read or written besides the arguments. `crossover` is the table row for
    /// this operation (`crossoverRows(_:)`, or multiply's own row); `batching` and `typeName` are
    /// evaluated only when the rule reaches them.
    public static func route(_ op: RoutedOp, rows: Int, mode m: RouterMode, threadOverride: Bool = false,
                             crossover c: Int, cpuPath unavailable: String? = nil, measured: Bool = true,
                             pending: Bool = false, batching: @autoclosure () -> Bool = false,
                             typeName: @autoclosure () -> String = "int64") -> RouteDecision {
        if let why = unavailable { return RouteDecision(op: op, path: .gpu, reason: .noCPUPath(why), rows: rows) }
        if pending { return RouteDecision(op: op, path: .gpu, reason: .pendingInput, rows: rows) }
        if m == .gpu { return RouteDecision(op: op, path: .gpu, reason: .forced(.gpu, thread: threadOverride), rows: rows) }
        if batching() { return RouteDecision(op: op, path: .gpu, reason: .batchOpen, rows: rows) }
        if m == .cpu { return RouteDecision(op: op, path: .cpu, reason: .forced(.cpu, thread: threadOverride), rows: rows) }
        if !measured { return RouteDecision(op: op, path: .gpu, reason: .notMeasuredForType(typeName()), rows: rows) }
        return rows < c ? RouteDecision(op: op, path: .cpu, reason: .belowCrossover(crossover: c), rows: rows)
                        : RouteDecision(op: op, path: .gpu, reason: .atOrAboveCrossover(crossover: c), rows: rows)
    }

    /// Decision for a single-input operation over `a`.
    static func decide<T: ArrowPrimitive>(_ op: RoutedOp, _ a: MetalArray<T>, cpuPath unavailable: String? = nil) -> RouteDecision {
        decide(op, rows: a.pending ? a.capacityLength : a.knownLength, cpuPath: unavailable, measured: measured(T.self),
               pending: a.pending, batching: a.context.isBatching, typeName: typeName(T.self))
    }

    /// Decision for a two-input operation (both inputs must be settled for the CPU to read them).
    static func decide<T: ArrowPrimitive>(_ op: RoutedOp, _ a: MetalArray<T>, _ b: MetalArray<T>,
                                          cpuPath unavailable: String? = nil) -> RouteDecision {
        decide(op, rows: a.pending ? a.capacityLength : a.knownLength, cpuPath: unavailable, measured: measured(T.self),
               pending: a.pending || b.pending, batching: a.context.isBatching, typeName: typeName(T.self))
    }

    /// Decisions for arithmetic. The table's arithmetic row was measured on `add`, and `subtract` costs
    /// the same per element. A 64-bit integer multiply costs the CPU more than an add, so `multiply`
    /// uses its own row (`crossoverRows(arithmetic: .mul)`), fitted from the shipped multiply loop.
    static func decideArithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ b: MetalArray<T>?, _ op: ArithmeticOp) -> RouteDecision {
        decide(.arithmetic, rows: a.pending ? a.capacityLength : a.knownLength,
               cpuPath: RouterCPU.arithmeticUnavailable(T.self, op), measured: measured(T.self),
               pending: a.pending || (b?.pending ?? false), batching: a.context.isBatching,
               typeName: typeName(T.self), crossover: crossoverRows(arithmetic: op))
    }

    /// Decision for `filter(mask)`.
    static func decide<T: ArrowPrimitive>(_ op: RoutedOp, _ a: MetalArray<T>, mask: MetalBooleanArray) -> RouteDecision {
        decide(op, rows: a.pending ? a.capacityLength : a.knownLength, cpuPath: nil, measured: measured(T.self),
               pending: a.pending || mask.pending, batching: a.context.isBatching, typeName: typeName(T.self))
    }

    /// Decision for `GroupBy.sum` (rows are the key column's length).
    static func decideGroupBySum<K: ArrowIndex, T: ArrowPrimitive>(keys: MetalArray<K>, values: MetalArray<T>,
                                                                   keyCount: Int) -> RouteDecision {
        decide(.groupBySum, rows: keys.pending ? keys.capacityLength : keys.knownLength,
               cpuPath: RouterCPU.groupBySumUnavailable(keyCount: keyCount), measured: measured(T.self),
               pending: keys.pending || values.pending, batching: values.context.isBatching, typeName: typeName(T.self))
    }

    static func typeName<T: ArrowPrimitive>(_: T.Type) -> String {
        switch T.arrowFormat {
        case "f": return "float32"
        case "g": return "float64"
        default: return T.mslType
        }
    }
}
