import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the GPU hash join (`Kernels/Join.swift`) and for `dictionary_encode` over every column type
// (`Kernels/DictionaryCompute.swift`). Both existed on the Swift side only; these are the entry points the
// Python, Rust, Go and C bindings call.

private let errorKey = "ArrowMetalC.lastError"
private func fail(_ e: Error) -> Int32 { Thread.current.threadDictionary[errorKey] = "\(e)"; return 1 }

@inline(__always) private func array(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func produce(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>) {
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
}

/// The int64 view of a join key column: int32 and int64 keys join natively, temporal columns join on their
/// storage integers, and everything else is rejected rather than silently cast.
private func joinKey(_ a: AnyMetalArray, _ side: String) throws -> AnyMetalArray {
    switch a {
    case .int32, .int64: return a
    case .temporal(let t): return t.asInt64.map { AnyMetalArray.int64($0) } ?? .int32(t.asInt32!)
    case .dictionary(let codes, _): return .int32(codes)
    default:
        throw ArrowMetalError.unsupportedType("am_join \(side) keys must be int32, int64, a temporal or a "
                                              + "dictionary column, got \(a.arrowFormat)")
    }
}

/// Arrow's `hash_join` index form: the matching (left row, right row) pairs of an equi-join.
///
/// `join_type` is 0 for an inner join and 1 for a left outer join. Both outputs are int32 index arrays of
/// the same length: `out_left_idx[i]` is a row of `left_keys` and `out_right_idx[i]` the row of
/// `right_keys` it matches. Duplicate keys on either side produce every combination; null keys never match;
/// with a left join an unmatched left row appears once with a null right index. The pair order is
/// unspecified — apply the indices with `take` on whichever columns the caller wants.
///
/// `MetalRecordBatch.join` on the Swift side is these indices plus a `take` of every column of both sides,
/// with the right key column dropped.
@_cdecl("am_join")
public func am_join(_ left: OpaquePointer?, _ right: OpaquePointer?, _ joinType: Int32,
                    _ outLeft: UnsafeMutablePointer<OpaquePointer?>?,
                    _ outRight: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let l = array(left), let r = array(right), let outLeft, let outRight else { return 2 }
    do {
        guard let kind = joinType == 0 ? JoinKind.inner : (joinType == 1 ? JoinKind.left : nil) else {
            throw ArrowMetalError.invalidArrowArray("am_join: join_type must be 0 (inner) or 1 (left), got \(joinType)")
        }
        let lk = try joinKey(l, "left"), rk = try joinKey(r, "right")
        let pair: (leftIndices: MetalArray<Int32>, rightIndices: MetalArray<Int32>)
        switch (lk, rk) {
        case (.int32(let a), .int32(let b)): pair = try hashJoin(left: a, right: b, kind: kind)
        case (.int64(let a), .int64(let b)): pair = try hashJoin(left: a, right: b, kind: kind)
        case (.int32(let a), .int64(let b)): pair = try hashJoin(left: try a.cast(to: Int64.self), right: b, kind: kind)
        case (.int64(let a), .int32(let b)): pair = try hashJoin(left: a, right: try b.cast(to: Int64.self), kind: kind)
        default:
            throw ArrowMetalError.unsupportedType("am_join: key types \(l.arrowFormat) and \(r.arrowFormat) do not match")
        }
        produce(.int32(pair.leftIndices), outLeft)
        produce(.int32(pair.rightIndices), outRight)
        return 0
    } catch { return fail(error) }
}

/// Arrow `dictionary_encode` for every column type: `out_codes` are dense int32 codes and `out_values` the
/// distinct values they index, so `codes[i]` names `values[codes[i]]` and a null row gives a null code.
///
/// utf8 and binary columns go through the host hash map (`Kernels/StringDictionary.swift`); primitive,
/// temporal and boolean columns go through the GPU `unique()` pipeline (`dictionaryEncoded()`), which is
/// what `am_str_dictionary_encode` could not reach. A column that is already dictionary encoded is
/// returned as its own codes and values.
@_cdecl("am_dictionary_encode")
public func am_dictionary_encode(_ a: OpaquePointer?, _ outCodes: UnsafeMutablePointer<OpaquePointer?>?,
                                 _ outValues: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let outCodes, let outValues else { return 2 }
    do {
        if case .string(let s) = x {
            let (c, u) = try s.dictionaryEncode()
            produce(.int32(c), outCodes)
            produce(.string(u), outValues)
            return 0
        }
        guard case .dictionary(let codes, let values) = try x.dictionaryEncoded() else {
            throw ArrowMetalError.unsupportedType("dictionary_encode does not support \(x.arrowFormat)")
        }
        produce(.int32(codes), outCodes)
        produce(values, outValues)
        return 0
    } catch { return fail(error) }
}
