import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the remaining Arrow selection / sort / random / aggregate functions:
// am_inverse_permutation, am_scatter, am_winsorize, am_rank, am_random, am_true_unless_null,
// am_count_all, am_first_last, am_str_extra and am_pivot_wider. Handles and error reporting follow
// ArrowMetalC.swift exactly: the same retained `Box` and the same thread-local error slot, so
// am_last_error() reports failures from here too.
//
//   am_rank       0 rank_quantile (float64)   1 rank_normal (float64)   2 rank_normal (float32)
//   am_str_extra  0 utf8_swapcase             1 utf8_zero_fill (p1 = width, arg = pad character)

private let selErrorKey = "ArrowMetalC.lastError"
private func selSetError(_ e: Error) { Thread.current.threadDictionary[selErrorKey] = "\(e)" }

@inline(__always) private func selHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func selRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch {
        selSetError(error)
        return 1
    }
}

// MARK: - Type-erased selection operations

/// The new selection functions on `MetalArray<T>`, erased so the C layer does not repeat a ten-way
/// switch per entry point.
protocol SelectionCOps {
    func selInversePermutation(_ maxIndex: Int64) throws -> MetalArray<Int32>
    func selScatter(_ indices: MetalArray<Int32>, _ maxIndex: Int64) throws -> AnyMetalArray
    func selWinsorize(_ lower: Double, _ upper: Double) throws -> AnyMetalArray
    func selRank(_ op: Int32) throws -> AnyMetalArray
    func selFirstLast(_ skipNulls: Bool) throws -> MetalStructArray
}

extension MetalArray: SelectionCOps {
    func selInversePermutation(_ maxIndex: Int64) throws -> MetalArray<Int32> {
        try inversePermutation(maxIndex: maxIndex)
    }
    func selScatter(_ indices: MetalArray<Int32>, _ maxIndex: Int64) throws -> AnyMetalArray {
        wrap(try scattered(to: indices, maxIndex: maxIndex))
    }
    func selWinsorize(_ lower: Double, _ upper: Double) throws -> AnyMetalArray {
        wrap(try winsorize(lowerLimit: lower, upperLimit: upper))
    }
    func selRank(_ op: Int32) throws -> AnyMetalArray {
        switch op {
        case 0: return .float64(try rankQuantile())
        case 1: return .float64(try rankNormal())
        case 2: return .float32(try rankNormalFloat32())
        default: throw ArrowMetalError.invalidArrowArray("unknown rank op \(op)")
        }
    }
    func selFirstLast(_ skipNulls: Bool) throws -> MetalStructArray { try firstLast(skipNulls: skipNulls) }
}

private func withSelection<R>(_ a: AnyMetalArray, _ body: (any SelectionCOps) throws -> R) throws -> R {
    switch a {
    case .int8(let x): return try body(x)
    case .uint8(let x): return try body(x)
    case .int16(let x): return try body(x)
    case .uint16(let x): return try body(x)
    case .int32(let x): return try body(x)
    case .uint32(let x): return try body(x)
    case .int64(let x): return try body(x)
    case .uint64(let x): return try body(x)
    case .float32(let x): return try body(x)
    case .float64(let x): return try body(x)
    case .temporal(let t):
        // A temporal column ranks and winsorizes through its storage integers, as the aggregates do.
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .boolean, .string, .binary, .decimal, .list, .structure, .map, .union, .runEndEncoded:
        throw ArrowMetalError.unsupportedType("this operation needs a primitive array, got \(a.arrowFormat)")
    case .dictionary:
        throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    }
}

/// An int32 index column, narrowing int64 / uint32 index columns on the way in.
private func indexColumn(_ a: AnyMetalArray) throws -> MetalArray<Int32> {
    switch a {
    case .int32(let x): return x
    case .int8(let x): return try x.cast(to: Int32.self)
    case .uint8(let x): return try x.cast(to: Int32.self)
    case .int16(let x): return try x.cast(to: Int32.self)
    case .uint16(let x): return try x.cast(to: Int32.self)
    case .uint32(let x): return try x.cast(to: Int32.self)
    case .int64(let x): return try x.cast(to: Int32.self)
    case .uint64(let x): return try x.cast(to: Int32.self)
    default: throw ArrowMetalError.unsupportedType("index columns must be integers, got \(a.arrowFormat)")
    }
}

// MARK: - Exported operations

/// Arrow `inverse_permutation`: for the i-th index, the index-th output element is i. The output has
/// `max_index + 1` elements, or the input's length when `max_index` is negative; unassigned slots are
/// null and duplicate indices resolve to the last (largest) source position. Always int32.
@_cdecl("am_inverse_permutation")
public func am_inverse_permutation(_ a: OpaquePointer?, _ maxIndex: Int64,
                                   _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { .int32(try withSelection(x) { try $0.selInversePermutation(maxIndex) }) }
}

/// Arrow `scatter`: the i-th value goes to the position named by the i-th index. Works for every
/// column type, nested ones included, because it is the inverse permutation used as a `take`.
@_cdecl("am_scatter")
public func am_scatter(_ values: OpaquePointer?, _ indices: OpaquePointer?, _ maxIndex: Int64,
                       _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let v = selHandle(values), let i = selHandle(indices) else { return 2 }
    return selRun(out) { try v.scattered(to: try indexColumn(i), maxIndex: maxIndex) }
}

/// Arrow `winsorize`: clamps to the nearest quantiles at `lower_limit` and `upper_limit`.
@_cdecl("am_winsorize")
public func am_winsorize(_ a: OpaquePointer?, _ lower: Double, _ upper: Double,
                         _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { try withSelection(x) { try $0.selWinsorize(lower, upper) } }
}

/// Arrow `rank_quantile` (op 0) and `rank_normal` in float64 (op 1) or float32 (op 2).
@_cdecl("am_rank")
public func am_rank(_ a: OpaquePointer?, _ op: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { try withSelection(x) { try $0.selRank(op) } }
}

/// Arrow `random`: `count` uniform float64 values in [0, 1) from Philox4x32-10 keyed by `seed`.
@_cdecl("am_random")
public func am_random(_ count: Int64, _ seed: UInt64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    selRun(out) { .float64(try ArrowRandom.uniform(count: Int(count), seed: seed)) }
}

/// Arrow `true_unless_null`: true for every valid row, null for every null one.
@_cdecl("am_true_unless_null")
public func am_true_unless_null(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { .boolean(try x.trueUnlessNull()) }
}

/// Arrow `count_all`: the number of rows, valid or not. -1 for a null handle.
@_cdecl("am_count_all")
public func am_count_all(_ a: OpaquePointer?) -> Int64 { Int64(selHandle(a)?.countAll ?? -1) }

/// Arrow `first_last`: a one-row struct with fields `first` and `last` of the column's own type.
@_cdecl("am_first_last")
public func am_first_last(_ a: OpaquePointer?, _ skipNulls: Int32,
                          _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { .structure(try withSelection(x) { try $0.selFirstLast(skipNulls != 0) }) }
}

/// Arrow `utf8_swapcase` (op 0) and `utf8_zero_fill` (op 1, `p1` = width, `arg` = the pad character).
@_cdecl("am_str_extra")
public func am_str_extra(_ a: OpaquePointer?, _ op: Int32, _ arg: UnsafePointer<CChar>?, _ argLen: Int64,
                         _ p1: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) {
        guard case .string(let s) = x else {
            throw ArrowMetalError.unsupportedType("string transforms need a utf8 array, got \(x.arrowFormat)")
        }
        switch op {
        case 0: return .string(try s.utf8Swapcase())
        case 1:
            let pad = arg.map { String(decoding: UnsafeRawBufferPointer(start: $0, count: Int(argLen)), as: UTF8.self) } ?? "0"
            return .string(try s.utf8ZeroFill(width: Int(p1), padding: pad.isEmpty ? "0" : pad))
        default: throw ArrowMetalError.invalidArrowArray("unknown string transform op \(op)")
        }
    }
}

/// Arrow `pivot_wider`: one struct row with a field per entry of `key_names`.
/// `raise_unexpected != 0` is Arrow's `unexpected_key_behavior = "raise"`.
@_cdecl("am_pivot_wider")
public func am_pivot_wider(_ keys: OpaquePointer?, _ values: OpaquePointer?,
                           _ keyNames: UnsafePointer<UnsafePointer<CChar>?>?, _ nKeys: Int64,
                           _ raiseUnexpected: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let k = selHandle(keys), let v = selHandle(values), let keyNames, nKeys >= 0 else { return 2 }
    var names: [String] = []
    for i in 0..<Int(nKeys) {
        guard let p = keyNames[i] else { return 2 }
        names.append(String(cString: p))
    }
    return selRun(out) {
        .structure(try PivotWider.pivot(keys: k, values: v, keyNames: names,
                                        unexpectedKey: raiseUnexpected != 0 ? .raise : .ignore))
    }
}

/// Arrow `make_struct`: composes equal-length columns into a struct-typed column. Metadata only —
/// the children are shared, nothing is copied and no kernel runs.
@_cdecl("am_make_struct")
public func am_make_struct(_ arrays: UnsafeMutablePointer<OpaquePointer?>?,
                           _ names: UnsafePointer<UnsafePointer<CChar>?>?, _ count: Int64,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let arrays, let names, count > 0 else { return 2 }
    var children: [AnyMetalArray] = []
    var fieldNames: [String] = []
    for i in 0..<Int(count) {
        guard let c = selHandle(arrays[i]), let n = names[i] else { return 2 }
        children.append(c)
        fieldNames.append(String(cString: n))
    }
    return selRun(out) {
        let n = children[0].length
        for c in children where c.length != n { throw ArrowMetalError.lengthMismatch(n, c.length) }
        return .structure(try MetalStructArray(length: n, nullCount: 0, validity: nil,
                                               names: fieldNames, children: children,
                                               context: children[0].context))
    }
}

// MARK: - Associative transforms and partial sort, wired through to C

/// The associative transforms and the partial sort, erased over the element type.
protocol AssociativeCOps {
    func selUnique() throws -> AnyMetalArray
    func selValueCounts() throws -> MetalStructArray
    func selPartitionNth(_ n: Int64) throws -> MetalArray<Int32>
}

extension MetalArray: AssociativeCOps {
    func selUnique() throws -> AnyMetalArray { wrap(try unique()) }
    func selValueCounts() throws -> MetalStructArray {
        let (values, counts) = try valueCounts()
        return try MetalStructArray(length: values.length, nullCount: 0, validity: nil,
                                    names: ["values", "counts"],
                                    children: [wrap(values), .int64(counts)], context: context)
    }
    func selPartitionNth(_ n: Int64) throws -> MetalArray<Int32> { try partitionNthIndices(Int(n)) }
}

private func withAssociative<R>(_ a: AnyMetalArray, _ body: (any AssociativeCOps) throws -> R) throws -> R {
    switch a {
    case .int8(let x): return try body(x)
    case .uint8(let x): return try body(x)
    case .int16(let x): return try body(x)
    case .uint16(let x): return try body(x)
    case .int32(let x): return try body(x)
    case .uint32(let x): return try body(x)
    case .int64(let x): return try body(x)
    case .uint64(let x): return try body(x)
    case .float32(let x): return try body(x)
    case .float64(let x): return try body(x)
    case .temporal(let t):
        switch t.storage {
        case .int32(let x): return try body(x)
        case .int64(let x): return try body(x)
        }
    case .boolean, .string, .binary, .decimal, .list, .structure, .map, .union, .runEndEncoded:
        throw ArrowMetalError.unsupportedType("this operation needs a primitive array, got \(a.arrowFormat)")
    case .dictionary:
        throw ArrowMetalError.unsupportedType("decode the dictionary array first")
    }
}

/// Arrow `unique`: the distinct non-null values. ArrowMetal returns them **ascending** (one GPU sort
/// plus a run scan), where Arrow returns them in order of first appearance.
@_cdecl("am_unique")
public func am_unique(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { try withAssociative(x) { try $0.selUnique() } }
}

/// Arrow `value_counts`: a struct column with fields `values` and `counts` (int64), the values
/// ascending rather than in order of first appearance.
@_cdecl("am_value_counts")
public func am_value_counts(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { .structure(try withAssociative(x) { try $0.selValueCounts() }) }
}

/// Arrow `partition_nth_indices`: indices that put the n smallest values first. ArrowMetal answers
/// with the full stable argsort, which satisfies the contract and costs one radix sort.
@_cdecl("am_partition_nth_indices")
public func am_partition_nth_indices(_ a: OpaquePointer?, _ n: Int64,
                                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = selHandle(a) else { return 2 }
    return selRun(out) { .int32(try withAssociative(x) { try $0.selPartitionNth(n) }) }
}
