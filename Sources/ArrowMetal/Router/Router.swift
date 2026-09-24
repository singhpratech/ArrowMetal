import Foundation

// CPU/GPU router. Design and rules: docs/DESIGN.md, section "CPU/GPU router".
//
// The routed operations call `Router.decide` once, at the top of their public entry point, before
// anything is encoded. The answer is a lookup in `RouterTable` (generated from the measured sweep by
// Benchmarks/router_table.py) plus a few structural rules; nothing is estimated at run time and no
// state is kept across calls beyond the process mode, a per-thread override and the per-thread last
// decision (which exists only so callers can see what happened).

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

    /// The crossover the table holds for `op`, in rows.
    public static func crossoverRows(_ op: RoutedOp) -> Int { RouterTable.crossoverRows(op) }

    // MARK: the decision

    /// Value-type class the table was measured on. Integer columns of every width use the int64 rows;
    /// floating-point columns have no measured crossover, so `auto` keeps them on the GPU.
    @inline(__always) static func measured<T: ArrowPrimitive>(_: T.Type) -> Bool { !T.isFloatingPoint }

    /// The decision for one call. `cpuPath` is nil when a CPU loop exists for this input, or the reason
    /// it does not. `pending` and `batching` describe the input and the calling thread.
    static func decide(_ op: RoutedOp, rows: Int, cpuPath unavailable: String?, measured: Bool,
                       pending: Bool, batching: @autoclosure () -> Bool, typeName: @autoclosure () -> String) -> RouteDecision {
        let st = threadState
        let d: RouteDecision
        if let why = unavailable {
            d = RouteDecision(op: op, path: .gpu, reason: .noCPUPath(why), rows: rows)
        } else if pending {
            d = RouteDecision(op: op, path: .gpu, reason: .pendingInput, rows: rows)
        } else {
            let thread = st.override
            let m = thread ?? mode
            if m == .gpu {
                d = RouteDecision(op: op, path: .gpu, reason: .forced(.gpu, thread: thread != nil), rows: rows)
            } else if batching() {
                d = RouteDecision(op: op, path: .gpu, reason: .batchOpen, rows: rows)
            } else if m == .cpu {
                d = RouteDecision(op: op, path: .cpu, reason: .forced(.cpu, thread: thread != nil), rows: rows)
            } else if !measured {
                d = RouteDecision(op: op, path: .gpu, reason: .notMeasuredForType(typeName()), rows: rows)
            } else {
                let c = RouterTable.crossoverRows(op)
                d = rows < c ? RouteDecision(op: op, path: .cpu, reason: .belowCrossover(crossover: c), rows: rows)
                             : RouteDecision(op: op, path: .gpu, reason: .atOrAboveCrossover(crossover: c), rows: rows)
            }
        }
        st.last = d
        return d
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

    /// Decisions for arithmetic. The table's row was measured on `add`; `subtract` costs the same per
    /// element, while a 64-bit integer multiply is slower on the CPU than an add, so `multiply` has no
    /// measured crossover and `auto` keeps it on the GPU.
    static func decideArithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ b: MetalArray<T>?, _ op: ArithmeticOp) -> RouteDecision {
        let isMeasured = measured(T.self) && op != .mul
        return decide(.arithmetic, rows: a.pending ? a.capacityLength : a.knownLength,
                      cpuPath: RouterCPU.arithmeticUnavailable(T.self, op), measured: isMeasured,
                      pending: a.pending || (b?.pending ?? false), batching: a.context.isBatching,
                      typeName: op == .mul && measured(T.self) ? "multiply" : typeName(T.self))
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
