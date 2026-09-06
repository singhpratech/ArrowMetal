import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the nested types: list ("+l", "+L", "+w:N"), struct ("+s"), map ("+m") and dense/sparse
// union ("+ud:", "+us:"). `am_import` / `am_export` already carry them through the C Data Interface,
// and `am_filter` / `am_take` / `am_slice` already work on them, because they go through AnyMetalArray;
// what this file adds is the nested compute surface and child navigation.
//
// Handles and error reporting follow ArrowMetalC.swift exactly: the same retained `Box` and the same
// thread-local error slot, so am_last_error() reports failures from here too.

private let nestedErrorKey = "ArrowMetalC.lastError"
private func nsFail(_ e: Error) -> Int32 { Thread.current.threadDictionary[nestedErrorKey] = "\(e)"; return 1 }

@inline(__always) private func nsHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func nsRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { return nsFail(error) }
}

/// Arrow `list_value_length`: an int32 array of per-row child counts, null where the row is null.
/// Accepts a list, large_list, fixed_size_list or map array.
@_cdecl("am_list_value_length")
public func am_list_value_length(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = nsHandle(a), out != nil else { return 2 }
    return nsRun(out) { .int32(try x.listValueLength()) }
}

/// Arrow `list_flatten`: the child array restricted to the range the list references.
@_cdecl("am_list_flatten")
public func am_list_flatten(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = nsHandle(a), out != nil else { return 2 }
    return nsRun(out) { try x.listFlatten() }
}

/// Arrow `list_element`: element `index` of every row, null where the row is null or too short.
@_cdecl("am_list_element")
public func am_list_element(_ a: OpaquePointer?, _ index: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = nsHandle(a), out != nil else { return 2 }
    return nsRun(out) { try x.listElement(Int(clamping: index)) }
}

/// Arrow `struct_field`: one field of a struct array by name.
@_cdecl("am_struct_field")
public func am_struct_field(_ a: OpaquePointer?, _ name: UnsafePointer<CChar>?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = nsHandle(a), let name, out != nil else { return 2 }
    let n = String(cString: name)
    return nsRun(out) { try x.structField(n) }
}

/// Number of child arrays: 1 for a list or a map (its `entries` struct) or a dictionary (its values),
/// one per field for a struct, one per variant for a union, 0 for a flat array. -1 for a null handle.
@_cdecl("am_child_count")
public func am_child_count(_ a: OpaquePointer?) -> Int64 {
    guard let x = nsHandle(a) else { return -1 }
    return Int64(x.children.count)
}

/// Child `i` of a nested (or dictionary) array; see am_child_count for the ordering.
@_cdecl("am_child")
public func am_child(_ a: OpaquePointer?, _ i: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = nsHandle(a), out != nil else { return 2 }
    return nsRun(out) {
        let kids = x.children
        guard i >= 0, Int(i) < kids.count else {
            throw ArrowMetalError.invalidArrowArray("child \(i) is out of range (\(x.arrowFormat) has \(kids.count))")
        }
        return kids[Int(i)]
    }
}
