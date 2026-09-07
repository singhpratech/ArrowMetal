import Foundation
import CArrowABI
import ArrowMetal

// The lazy query engine over the C ABI.
//
// A caller registers its tables once (`am_plan_source_create`, which takes the column handles it already
// holds), then sends a plan as JSON (`docs/ENGINE.md` and `Sources/ArrowMetal/Engine/PlanJSON.swift`
// carry the grammar). The plan is type-checked, optimized and run, and the result comes back as a
// handle whose columns are ordinary `am_*` array handles.
//
// Registering the sources separately is what keeps the plan text free of data: the same text can be
// re-run against new sources, and `am_plan_explain` can print a plan without touching the GPU.

private let planErrorKey = "ArrowMetalC.lastError"
private func setPlanError(_ e: Error) { Thread.current.threadDictionary[planErrorKey] = "\(e)" }

@inline(__always) private func planHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}

/// A registered table: a name plus the Metal-resident batch it scans.
final class PlanSourceBox {
    let source: PlanSource
    init(_ s: PlanSource) { source = s }
}

/// A finished plan: named columns, with C strings kept alive for the caller.
final class PlanResultBox {
    let batch: MetalRecordBatch
    private var strings: [UnsafeMutablePointer<CChar>] = []
    init(_ b: MetalRecordBatch) { batch = b }
    deinit { for s in strings { free(s) } }
    func cString(_ s: String) -> UnsafePointer<CChar> {
        let c = strdup(s)!
        strings.append(c)
        return UnsafePointer(c)
    }
}

@inline(__always) private func sourceBox(_ p: OpaquePointer?) -> PlanSourceBox? {
    guard let p else { return nil }
    return Unmanaged<PlanSourceBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}
@inline(__always) private func resultBox(_ p: OpaquePointer?) -> PlanResultBox? {
    guard let p else { return nil }
    return Unmanaged<PlanResultBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

/// Registers a table the plan can `scan` by name. Column handles are retained by the source.
@_cdecl("am_plan_source_create")
public func am_plan_source_create(_ name: UnsafePointer<CChar>?,
                           _ columns: UnsafeMutablePointer<OpaquePointer?>?,
                           _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                           _ nColumns: Int64,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let name, let columns, let names, let out, nColumns >= 0 else { return 2 }
    do {
        var cols: [AnyMetalArray] = [], ns: [String] = []
        for i in 0..<Int(nColumns) {
            guard let a = planHandle(columns[i]), let n = names[i] else { return 2 }
            cols.append(a)
            ns.append(String(cString: n))
        }
        let batch = try MetalRecordBatch(names: ns, columns: cols)
        let src = PlanSource(name: String(cString: name), batch: batch)
        out.pointee = OpaquePointer(Unmanaged.passRetained(PlanSourceBox(src)).toOpaque())
        return 0
    } catch {
        setPlanError(error)
        return 1
    }
}

@_cdecl("am_plan_source_release")
public func am_plan_source_release(_ p: OpaquePointer?) {
    guard let p else { return }
    Unmanaged<PlanSourceBox>.fromOpaque(UnsafeRawPointer(p)).release()
}

private func collectSources(_ sources: UnsafeMutablePointer<OpaquePointer?>?, _ n: Int64) -> [String: PlanSource]? {
    guard let sources else { return n == 0 ? [:] : nil }
    var map: [String: PlanSource] = [:]
    for i in 0..<Int(n) {
        guard let b = sourceBox(sources[i]) else { return nil }
        map[b.source.name] = b.source
    }
    return map
}

/// Runs a JSON plan against the registered sources. `optimize` is 0 or 1.
@_cdecl("am_plan_run")
public func am_plan_run(_ planText: UnsafePointer<CChar>?,
                        _ sources: UnsafeMutablePointer<OpaquePointer?>?,
                        _ nSources: Int64,
                        _ optimize: Int32,
                        _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let planText, let out, nSources >= 0, let map = collectSources(sources, nSources) else { return 2 }
    // A caller that never returns to a run loop — a Python `for` loop, say — never drains the
    // autorelease pool, so every Metal object a query autoreleases would live until the process
    // exits. Draining here keeps a long run of small queries flat instead of growing ~2.5 KB a query.
    return autoreleasepool {
        do {
            let batch = try PlanJSON.run(String(cString: planText), sources: map, optimize: optimize != 0)
            out.pointee = OpaquePointer(Unmanaged.passRetained(PlanResultBox(batch)).toOpaque())
            return 0
        } catch {
            setPlanError(error)
            return 1
        }
    }
}

/// The optimized logical plan and the physical plan it lowers to, as text. Valid until the next call
/// on this thread; NULL with `am_last_error()` set when the plan does not type-check.
@_cdecl("am_plan_explain")
public func am_plan_explain(_ planText: UnsafePointer<CChar>?,
                            _ sources: UnsafeMutablePointer<OpaquePointer?>?,
                            _ nSources: Int64,
                            _ optimize: Int32) -> UnsafePointer<CChar>? {
    guard let planText, nSources >= 0, let map = collectSources(sources, nSources) else { return nil }
    return autoreleasepool {
        do {
            let text = try PlanJSON.explain(String(cString: planText), sources: map, optimize: optimize != 0)
            let key = "ArrowMetalC.planExplain"
            if let old = Thread.current.threadDictionary[key] as? UnsafeMutablePointer<CChar> { free(old) }
            let c = strdup(text)!
            Thread.current.threadDictionary[key] = c
            return UnsafePointer(c)
        } catch {
            setPlanError(error)
            return nil
        }
    }
}

@_cdecl("am_plan_column_count")
public func am_plan_column_count(_ r: OpaquePointer?) -> Int64 { Int64(resultBox(r)?.batch.names.count ?? -1) }

@_cdecl("am_plan_row_count")
public func am_plan_row_count(_ r: OpaquePointer?) -> Int64 { Int64(resultBox(r)?.batch.length ?? -1) }

@_cdecl("am_plan_column_name")
public func am_plan_column_name(_ r: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let b = resultBox(r), i >= 0, i < Int64(b.batch.names.count) else { return nil }
    return b.cString(b.batch.names[Int(i)])
}

/// Hands out column `i` as an ordinary array handle, which the caller releases with `am_release`.
@_cdecl("am_plan_column")
public func am_plan_column(_ r: OpaquePointer?, _ i: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = resultBox(r), let out, i >= 0, i < Int64(b.batch.columns.count) else { return 2 }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(b.batch.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_plan_result_release")
public func am_plan_result_release(_ r: OpaquePointer?) {
    guard let r else { return }
    Unmanaged<PlanResultBox>.fromOpaque(UnsafeRawPointer(r)).release()
}
