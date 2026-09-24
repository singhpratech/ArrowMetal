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

/// The crossover the shipped table holds for op, in rows; -1 for an unknown op code.
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
