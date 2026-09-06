import Foundation
import CArrowABI
import ArrowMetal

// The fused expression compiler over the C ABI. One call takes a set of named columns and a serialised
// query (the s-expression grammar documented in include/arrowmetal.h) and returns a result handle that
// carries either output columns (project / group_by) or scalars (aggregate).

private let exprErrorKey = "ArrowMetalC.lastError"
private func setExprError(_ e: Error) { Thread.current.threadDictionary[exprErrorKey] = "\(e)" }

@inline(__always) private func exprHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}

/// A finished query: named columns and named scalars, with C strings kept alive for the caller.
final class QueryResultBox {
    let result: ExprQueryResult
    private var strings: [UnsafeMutablePointer<CChar>] = []
    init(_ r: ExprQueryResult) { result = r }
    deinit { for s in strings { free(s) } }
    func cString(_ s: String) -> UnsafePointer<CChar> {
        let c = strdup(s)!
        strings.append(c)
        return UnsafePointer(c)
    }
}

@inline(__always) private func resultBox(_ p: OpaquePointer?) -> QueryResultBox? {
    guard let p else { return nil }
    return Unmanaged<QueryResultBox>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue()
}

@_cdecl("am_query")
public func am_query(_ columns: UnsafeMutablePointer<OpaquePointer?>?,
                     _ names: UnsafeMutablePointer<UnsafePointer<CChar>?>?,
                     _ nColumns: Int64,
                     _ exprText: UnsafePointer<CChar>?,
                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let columns, let names, let exprText, let out, nColumns >= 0 else { return 2 }
    do {
        var cols: [AnyMetalArray] = [], ns: [String] = []
        for i in 0..<Int(nColumns) {
            guard let a = exprHandle(columns[i]) else { return 2 }
            guard let n = names[i] else { return 2 }
            cols.append(a)
            ns.append(String(cString: n))
        }
        let q = try ExprQuery(text: String(cString: exprText))
        let r = try runExprQuery(q, names: ns, columns: cols)
        out.pointee = OpaquePointer(Unmanaged.passRetained(QueryResultBox(r)).toOpaque())
        return 0
    } catch {
        setExprError(error)
        return 1
    }
}

@_cdecl("am_query_column_count")
public func am_query_column_count(_ r: OpaquePointer?) -> Int64 { Int64(resultBox(r)?.result.names.count ?? -1) }

@_cdecl("am_query_column_name")
public func am_query_column_name(_ r: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let b = resultBox(r), i >= 0, i < Int64(b.result.names.count) else { return nil }
    return b.cString(b.result.names[Int(i)])
}

@_cdecl("am_query_column")
public func am_query_column(_ r: OpaquePointer?, _ i: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let b = resultBox(r), let out, i >= 0, i < Int64(b.result.columns.count) else { return 2 }
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(b.result.columns[Int(i)])).toOpaque())
    return 0
}

@_cdecl("am_query_scalar_count")
public func am_query_scalar_count(_ r: OpaquePointer?) -> Int64 { Int64(resultBox(r)?.result.scalars.count ?? -1) }

@_cdecl("am_query_scalar_name")
public func am_query_scalar_name(_ r: OpaquePointer?, _ i: Int64) -> UnsafePointer<CChar>? {
    guard let b = resultBox(r), i >= 0, i < Int64(b.result.scalarNames.count) else { return nil }
    return b.cString(b.result.scalarNames[Int(i)])
}

/// out_kind: 0 int64 in out_i64, 1 uint64 in the same slot, 2 float64 in out_f64.
@_cdecl("am_query_scalar")
public func am_query_scalar(_ r: OpaquePointer?, _ i: Int64,
                            _ outI64: UnsafeMutablePointer<Int64>?, _ outF64: UnsafeMutablePointer<Double>?,
                            _ outKind: UnsafeMutablePointer<Int32>?, _ isNull: UnsafeMutablePointer<Int32>?) -> Int32 {
    guard let b = resultBox(r), i >= 0, i < Int64(b.result.scalars.count) else { return 2 }
    isNull?.pointee = 0
    switch b.result.scalars[Int(i)] {
    case .null: isNull?.pointee = 1; outKind?.pointee = 0; outI64?.pointee = 0; outF64?.pointee = 0
    case .int(let v): outKind?.pointee = 0; outI64?.pointee = v
    case .uint(let v): outKind?.pointee = 1; outI64?.pointee = Int64(bitPattern: v)
    case .double(let v): outKind?.pointee = 2; outF64?.pointee = v
    }
    return 0
}

@_cdecl("am_query_result_release")
public func am_query_result_release(_ r: OpaquePointer?) {
    guard let r else { return }
    Unmanaged<QueryResultBox>.fromOpaque(UnsafeRawPointer(r)).release()
}

/// Type-checks and canonicalises a query without running it. Returns the canonical text (valid until
/// the next call on this thread) or NULL with am_last_error() set.
@_cdecl("am_query_canonical")
public func am_query_canonical(_ exprText: UnsafePointer<CChar>?) -> UnsafePointer<CChar>? {
    guard let exprText else { return nil }
    do {
        let q = try ExprQuery(text: String(cString: exprText))
        let key = "ArrowMetalC.queryCanonical"
        if let old = Thread.current.threadDictionary[key] as? UnsafeMutablePointer<CChar> { free(old) }
        let c = strdup(q.canonical)!
        Thread.current.threadDictionary[key] = c
        return UnsafePointer(c)
    } catch {
        setExprError(error)
        return nil
    }
}
