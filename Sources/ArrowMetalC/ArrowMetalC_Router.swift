import Foundation
import ArrowMetal

// The CPU/GPU router over the C ABI (include/arrowmetal.h, "CPU/GPU router"; docs/DESIGN.md).
// Modes: 0 auto, 1 gpu, 2 cpu. Paths: 0 gpu, 1 cpu. Ops: 0 sum, 1 min, 2 max, 3 compare,
// 4 arithmetic, 5 filter, 6 group-by sum.

private func mode(_ code: Int32) -> RouterMode? {
    switch code { case 0: return .auto; case 1: return .gpu; case 2: return .cpu; default: return nil }
}
private func code(_ m: RouterMode) -> Int32 {
    switch m { case .auto: return 0; case .gpu: return 1; case .cpu: return 2 }
}

/// Sets the process-wide mode. Returns 0, or 2 for an unknown mode code.
@_cdecl("am_router_set_mode")
public func am_router_set_mode(_ m: Int32) -> Int32 {
    guard let v = mode(m) else { return 2 }
    Router.mode = v
    return 0
}

/// The process-wide mode (starts from ARROWMETAL_ROUTER, auto when unset).
@_cdecl("am_router_get_mode")
public func am_router_get_mode() -> Int32 { code(Router.mode) }

/// Sets the calling thread's override; -1 clears it. Returns 0, or 2 for an unknown mode code.
@_cdecl("am_router_set_thread_mode")
public func am_router_set_thread_mode(_ m: Int32) -> Int32 {
    if m == -1 { Router.threadMode = nil; return 0 }
    guard let v = mode(m) else { return 2 }
    Router.threadMode = v
    return 0
}

/// The calling thread's override, -1 when none is set.
@_cdecl("am_router_get_thread_mode")
public func am_router_get_thread_mode() -> Int32 { Router.threadMode.map(code) ?? -1 }

/// The last routing decision on the calling thread. Returns 1 and fills the out-parameters (any may
/// be NULL), or 0 when no routed operation has run on this thread since the last clear.
@_cdecl("am_router_last")
public func am_router_last(_ op: UnsafeMutablePointer<Int32>?, _ path: UnsafeMutablePointer<Int32>?,
                           _ rows: UnsafeMutablePointer<Int64>?) -> Int32 {
    guard let d = Router.lastDecision else { return 0 }
    op?.pointee = d.op.code
    path?.pointee = d.path == .gpu ? 0 : 1
    rows?.pointee = Int64(d.rows)
    return 1
}

private let reasonKey = "ArrowMetalC.routerReasonC"

/// The last decision's reason as text ("below the 176850-row crossover", "batch open", ...), or NULL
/// when there is none. Thread-local; valid until the next call of this function on the same thread.
@_cdecl("am_router_last_reason")
public func am_router_last_reason() -> UnsafePointer<CChar>? {
    let td = Thread.current.threadDictionary
    if let old = td[reasonKey] as? UnsafeMutablePointer<CChar> { free(old); td[reasonKey] = nil }
    guard let d = Router.lastDecision else { return nil }
    let c = strdup(d.reason.description)!
    td[reasonKey] = c
    return UnsafePointer(c)
}

/// Forgets the calling thread's last decision.
@_cdecl("am_router_clear_last")
public func am_router_clear_last() { Router.clearLastDecision() }

/// The crossover the table in force holds for op, in rows; -1 for an unknown op code.
@_cdecl("am_router_crossover")
public func am_router_crossover(_ op: Int32) -> Int64 {
    guard op >= 0, Int(op) < RoutedOp.allCases.count else { return -1 }
    return Int64(Router.crossoverRows(RoutedOp.allCases[Int(op)]))
}

/// The crossover `auto` uses for multiply, in rows (op 4, arithmetic, is the add/subtract row).
@_cdecl("am_router_multiply_crossover")
public func am_router_multiply_crossover() -> Int64 {
    Int64(Router.crossoverRows(arithmetic: .mul)!)
}

// MARK: the table in force, and explaining a decision

private func routerSetError(_ message: String) { Thread.current.threadDictionary["ArrowMetalC.lastError"] = message }

/// A per-thread C copy of `text`, valid until the next call that uses the same key on this thread.
private func routerThreadString(_ key: String, _ text: String) -> UnsafePointer<CChar> {
    let td = Thread.current.threadDictionary
    if let old = td[key] as? UnsafeMutablePointer<CChar> { free(old) }
    let c = strdup(text)!
    td[key] = c
    return UnsafePointer(c)
}

private func routerJSON(_ obj: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) else { return "{}" }
    return String(decoding: data, as: UTF8.self)
}

private func routerRowJSON(_ r: RouterRow) -> [String: Any] {
    ["label": r.label, "crossover_rows": r.crossover, "step_rows": r.stepRows, "bracket_low_rows": r.bracketLowRows,
     "points": r.points.map { ["rows": $0.rows, "gpu_us": $0.gpuMicros, "cpu_us": $0.cpuMicros] as [String: Any] }]
}

private func routerTableJSON(_ t: RouterCrossovers) -> [String: Any] {
    var rows: [String: Any] = [:]
    for op in RoutedOp.allCases { rows[op.rawValue] = routerRowJSON(t.rows[op]!) }
    rows["multiply"] = routerRowJSON(t.multiply)
    var shipped: [String: Any] = [:]
    for op in RoutedOp.allCases { shipped[op.rawValue] = RouterCrossovers.shipped.crossover(op) }
    shipped["multiply"] = RouterCrossovers.shipped.multiply.crossover
    var out: [String: Any] = ["shipped": t.isShipped, "source": t.source, "header": t.header,
                              "crossovers": rows, "shipped_ops": t.shippedOps, "shipped_crossovers": shipped,
                              "chip_id": RouterCrossovers.chipID(RouterCrossovers.machineChip)]
    if case .file(let p) = t.origin { out["path"] = p }
    if let c = t.chip { out["chip"] = c }
    if let d = t.date { out["date"] = d }
    if let g = t.grid { out["grid"] = g }
    if let e = Router.tableLoadError { out["load_error"] = e }
    return out
}

/// Replaces the crossover table in force with the JSON table at `path` (the format
/// `python -m arrowmetal.router calibrate` writes); NULL or "shipped" puts the shipped table back.
/// Returns 0, or 1 with am_last_error() saying why (the table in force is then unchanged).
@_cdecl("am_router_load_table")
public func am_router_load_table(_ path: UnsafePointer<CChar>?) -> Int32 {
    guard let path, String(cString: path).lowercased() != "shipped" else { Router.useShippedTable(); return 0 }
    do { try Router.loadTable(path: String(cString: path)); return 0 } catch { routerSetError("\(error)"); return 1 }
}

/// The crossover table in force as a JSON object: "shipped" (bool), "source", "header", "path" (when
/// loaded from a file), "chip", "date", "grid", "crossovers" {op: {"crossover_rows", "step_rows",
/// "bracket_low_rows", "label", "points"}}, "shipped_ops" (rows the file lacked, taken from the
/// shipped table) and "load_error" (why a table named at first use could not be read).
/// Thread-local; valid until the next call.
@_cdecl("am_router_table_info")
public func am_router_table_info() -> UnsafePointer<CChar>? {
    routerThreadString("ArrowMetalC.routerTableInfo", routerJSON(routerTableJSON(Router.table)))
}

/// The path `op` over `rows` values of `dtype` would take under the table and mode in force on this
/// thread (outside a batch, settled input): 0 gpu, 1 cpu, or -1 for an unknown operation or type.
/// `crossover` (may be NULL) receives the table row consulted. Runs nothing and records no last
/// decision. Operations: sum, min, max, compare, add, subtract, multiply, divide, filter,
/// filter_where, group_by_sum (over `key_count` keys). Types: int8 ... uint64, float32, float64.
@_cdecl("am_router_decide")
public func am_router_decide(_ op: UnsafePointer<CChar>?, _ dtype: UnsafePointer<CChar>?, _ rows: Int64,
                             _ keyCount: Int64, _ crossover: UnsafeMutablePointer<Int64>?) -> Int32 {
    guard let op, let dtype, rows >= 0,
          let e = Router.explain(operation: String(cString: op), dtype: String(cString: dtype), rows: Int(rows),
                                 keyCount: Int(keyCount)) else { return -1 }
    crossover?.pointee = Int64(e.row.crossover)
    return e.decision.path == .gpu ? 0 : 1
}

/// `am_router_decide` with its reasons, as a JSON object: "operation", "dtype", "rows", "mode",
/// "path", "reason", "routed_op", "thread_override", "row" (the table row consulted) and "table" (as
/// am_router_table_info). NULL with am_last_error() set for an unknown operation or type.
/// Thread-local; valid until the next call.
@_cdecl("am_router_explain")
public func am_router_explain(_ op: UnsafePointer<CChar>?, _ dtype: UnsafePointer<CChar>?, _ rows: Int64,
                              _ keyCount: Int64) -> UnsafePointer<CChar>? {
    guard let op, let dtype else { routerSetError("am_router_explain: operation and dtype are required"); return nil }
    let name = String(cString: op), type = String(cString: dtype)
    guard rows >= 0 else { routerSetError("am_router_explain: rows must be at least 0"); return nil }
    guard let e = Router.explain(operation: name, dtype: type, rows: Int(rows), keyCount: Int(keyCount)) else {
        if !Router.explainOperations.contains(where: { $0.name == name }) {
            routerSetError("unknown operation \(name); one of " + Router.explainOperations.map(\.name).joined(separator: ", "))
        } else {
            routerSetError("unknown dtype \(type); one of " + Router.explainTypes.joined(separator: ", "))
        }
        return nil
    }
    let obj: [String: Any] = ["operation": e.operation, "dtype": e.dtype, "rows": Int(rows), "key_count": Int(keyCount),
                              "mode": e.mode.rawValue, "path": e.decision.path.rawValue,
                              "reason": e.decision.reason.description, "routed_op": e.decision.op.rawValue,
                              "thread_override": Router.threadMode != nil,
                              "row": routerRowJSON(e.row), "table": routerTableJSON(e.table)]
    return routerThreadString("ArrowMetalC.routerExplain", routerJSON(obj))
}
