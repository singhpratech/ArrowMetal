import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the option-carrying forms of the functions whose plain entry points take no options:
//
//   am_cast_ex                a cast with Arrow's CastOptions and, for a list or struct target, the
//                             child formats (comma separated)
//   am_argsort_ex             array_sort_indices / sort_indices with null_placement
//   am_partition_nth_ex       partition_nth_indices with null_placement
//   am_rank_ex                rank with tiebreaker, sort direction and null_placement
//   am_rank_quantile_ex       rank_quantile / rank_normal with direction and null_placement
//   am_is_in_ex, am_index_in_ex        set lookup with null_matching_behavior
//   am_unique_ex, am_value_counts_ex, am_dictionary_encode_ex    with the distinct-value order
//   am_round_temporal_ex      floor / ceil / round with the whole RoundTemporalOptions surface
//   am_list_parent_indices64  list_parent_indices in int64, as pyarrow returns it
//
// Enumerations, all matching the Python layer:
//   null_placement    0 at_end (Arrow's default)  1 at_start
//   tiebreaker        0 min   1 max   2 first   3 dense
//   null_matching     0 match  1 skip  2 emit_null  3 inconclusive
//   value order       0 first_appearance (Arrow's own)  1 sorted (the cheaper GPU pass)
//   round mode        0 floor  1 ceil  2 round
//   cast flags        bit 0 allow_int_overflow, 1 allow_time_truncate, 2 allow_time_overflow,
//                     3 allow_decimal_truncate, 4 allow_float_truncate, 5 allow_invalid_utf8
//   round flags       bit 0 week_starts_monday, 1 ceil_is_strictly_greater, 2 calendar_based_origin

private let optErrorKey = "ArrowMetalC.lastError"
private func optSetError(_ e: Error) { Thread.current.threadDictionary[optErrorKey] = "\(e)" }

@inline(__always) private func optHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func optRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch {
        optSetError(error)
        return 1
    }
}

private func placement(_ v: Int32) -> NullPlacement { v == 1 ? .atStart : .atEnd }

private func tiebreaker(_ v: Int32) throws -> RankTiebreaker {
    switch v {
    case 0: return .min
    case 1: return .max
    case 2: return .first
    case 3: return .dense
    default: throw ArrowMetalError.invalidArrowArray("unknown rank tiebreaker \(v)")
    }
}

private func nullMatching(_ v: Int32) throws -> SetLookupNullMatching {
    switch v {
    case 0: return .match
    case 1: return .skip
    case 2: return .emitNull
    case 3: return .inconclusive
    default: throw ArrowMetalError.invalidArrowArray("unknown null_matching_behavior \(v)")
    }
}

private func valueOrder(_ v: Int32) -> ValueOrder { v == 1 ? .sorted : .firstAppearance }

/// utf8 and binary share one layout and one set of kernels, so both reach the string hash table.
private func bytesColumn(_ a: AnyMetalArray) -> MetalStringArray? {
    switch a {
    case .string(let s), .binary(let s): return s
    default: return nil
    }
}

// MARK: - cast

/// Arrow `cast` with `CastOptions`. `child_formats` is a comma-separated list of the target formats of
/// a list child or of a struct's fields, or null for a flat target.
@_cdecl("am_cast_ex")
public func am_cast_ex(_ a: OpaquePointer?, _ format: UnsafePointer<CChar>?,
                       _ childFormats: UnsafePointer<CChar>?, _ flags: UInt32,
                       _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a), let format else { return 2 }
    let target = String(cString: format)
    let kids: [String] = childFormats.map { String(cString: $0) }
        .map { $0.isEmpty ? [] : $0.split(separator: ",").map(String.init) } ?? []
    return optRun(out) { try x.cast(to: target, options: CastOptions(bits: flags), childFormats: kids) }
}

// MARK: - sorts

/// Arrow `array_sort_indices` / single-key `sort_indices` with `null_placement`.
@_cdecl("am_argsort_ex")
public func am_argsort_ex(_ a: OpaquePointer?, _ descending: Int32, _ nullPlacement: Int32,
                          _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        .int32(try x.argsortIndices(descending: descending != 0, nullPlacement: placement(nullPlacement)))
    }
}

/// Arrow `partition_nth_indices` with `null_placement`. A GPU radix select, not a sort.
@_cdecl("am_partition_nth_ex")
public func am_partition_nth_ex(_ a: OpaquePointer?, _ pivot: Int64, _ nullPlacement: Int32,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        .int32(try withOptionPrimitive(x) { try $0.optPartitionNth(Int(pivot), placement(nullPlacement)) })
    }
}

/// Multi-key `sort_indices` with `null_placement`, which applies to every key as Arrow's does.
@_cdecl("am_lexsort_ex")
public func am_lexsort_ex(_ columns: UnsafeMutablePointer<OpaquePointer?>?,
                          _ descending: UnsafePointer<Int32>?, _ count: Int64,
                          _ nullPlacement: Int32,
                          _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let columns, count > 0 else { return 2 }
    var cols: [AnyMetalArray] = []
    var desc: [Bool] = []
    for i in 0..<Int(count) {
        guard let c = optHandle(columns[i]) else { return 2 }
        cols.append(c)
        desc.append(descending.map { $0[i] != 0 } ?? false)
    }
    return optRun(out) {
        .int32(try lexsortIndices(cols, descending: desc, nullPlacement: placement(nullPlacement)))
    }
}

// MARK: - ranks

/// Arrow `rank`: `tiebreaker` 0 min, 1 max, 2 first, 3 dense.
@_cdecl("am_rank_ex")
public func am_rank_ex(_ a: OpaquePointer?, _ tb: Int32, _ descending: Int32, _ nullPlacement: Int32,
                       _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        .int32(try withOptionPrimitive(x) {
            try $0.optRank(try tiebreaker(tb), descending != 0, placement(nullPlacement))
        })
    }
}

/// Arrow `rank_quantile` (op 0) and `rank_normal` in float64 (1) or float32 (2), with the direction
/// and null placement.
@_cdecl("am_rank_quantile_ex")
public func am_rank_quantile_ex(_ a: OpaquePointer?, _ op: Int32, _ descending: Int32,
                                _ nullPlacement: Int32,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        try withOptionPrimitive(x) { try $0.optRankQuantile(op, descending != 0, placement(nullPlacement)) }
    }
}

// MARK: - set lookup

/// Arrow `is_in` with `null_matching_behavior`.
@_cdecl("am_is_in_ex")
public func am_is_in_ex(_ a: OpaquePointer?, _ setArray: OpaquePointer?, _ behavior: Int32,
                        _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a), let s = optHandle(setArray) else { return 2 }
    return optRun(out) {
        let b = try nullMatching(behavior)
        if let probe = bytesColumn(x), let set = bytesColumn(s) {
            return .boolean(try probe.isIn(set, nullMatching: b))
        }
        return .boolean(try withOptionPrimitive(x) { try $0.optIsIn(s, b) })
    }
}

/// Arrow `index_in` with `null_matching_behavior`.
@_cdecl("am_index_in_ex")
public func am_index_in_ex(_ a: OpaquePointer?, _ setArray: OpaquePointer?, _ behavior: Int32,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a), let s = optHandle(setArray) else { return 2 }
    return optRun(out) {
        let b = try nullMatching(behavior)
        if let probe = bytesColumn(x), let set = bytesColumn(s) {
            return .int32(try probe.indexIn(set, nullMatching: b))
        }
        return .int32(try withOptionPrimitive(x) { try $0.optIndexIn(s, b) })
    }
}

// MARK: - distinct values

/// Arrow `unique` in first-appearance order (0) or ascending (1).
@_cdecl("am_unique_ex")
public func am_unique_ex(_ a: OpaquePointer?, _ order: Int32,
                         _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        if let s = bytesColumn(x) {
            let u = try s.unique(order: valueOrder(order))
            if case .binary = x { return .binary(u) }
            return .string(u)
        }
        return try withOptionPrimitive(x) { try $0.optUnique(valueOrder(order)) }
    }
}

/// Arrow `value_counts` in first-appearance order (0) or ascending (1), as a struct of the values and
/// their int64 counts.
@_cdecl("am_value_counts_ex")
public func am_value_counts_ex(_ a: OpaquePointer?, _ order: Int32,
                               _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) {
        if let s = bytesColumn(x) {
            let (values, counts) = try s.valueCounts(order: valueOrder(order))
            var wrapped = AnyMetalArray.string(values)
            if case .binary = x { wrapped = .binary(values) }
            return .structure(try MetalStructArray(length: values.length, nullCount: 0, validity: nil,
                                                   names: ["values", "counts"],
                                                   children: [wrapped, .int64(counts)],
                                                   context: values.context))
        }
        return .structure(try withOptionPrimitive(x) { try $0.optValueCounts(valueOrder(order)) })
    }
}

/// Arrow `dictionary_encode` in first-appearance order (0) or ascending (1). Writes the int32 codes
/// into `codes` and the dictionary into `values`.
@_cdecl("am_dictionary_encode_ex")
public func am_dictionary_encode_ex(_ a: OpaquePointer?, _ order: Int32,
                                    _ codes: UnsafeMutablePointer<OpaquePointer?>?,
                                    _ values: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    do {
        let pair: (AnyMetalArray, AnyMetalArray)
        if let s = bytesColumn(x) {
            let (c, u) = try s.dictionaryEncode()
            var codes = c
            var dictionary = u
            if order == 1 {
                // The sorted dictionary needs the codes recoded through it, which the primitive path
                // does on the GPU; for utf8 the distinct set is small, so it is done here.
                dictionary = try s.unique(order: .sorted)
                let names = u.toArray(), sortedNames = dictionary.toArray()
                var position: [String: Int32] = [:]
                for (i, name) in sortedNames.enumerated() { if let name { position[name] = Int32(i) } }
                let recode = try MetalArray<Int32>(names.map { $0.flatMap { position[$0] } ?? 0 },
                                                   context: s.context)
                codes = try recode.take(c)
            }
            var wrapped = AnyMetalArray.string(dictionary)
            if case .binary = x { wrapped = .binary(dictionary) }
            pair = (.int32(codes), wrapped)
        } else {
            pair = try withOptionPrimitive(x) { try $0.optDictionaryEncode(valueOrder(order)) }
        }
        codes?.pointee = OpaquePointer(Unmanaged.passRetained(Box(pair.0)).toOpaque())
        values?.pointee = OpaquePointer(Unmanaged.passRetained(Box(pair.1)).toOpaque())
        return 0
    } catch {
        optSetError(error)
        return 1
    }
}

// MARK: - temporal rounding

/// Arrow `floor_temporal` (0), `ceil_temporal` (1) and `round_temporal` (2) with the whole
/// `RoundTemporalOptions` surface. `unit` indexes `TemporalRoundUnit.allCases` order as the Python
/// layer spells it.
@_cdecl("am_round_temporal_ex")
public func am_round_temporal_ex(_ a: OpaquePointer?, _ mode: Int32, _ unit: UnsafePointer<CChar>?,
                                 _ multiple: Int64, _ flags: UInt32,
                                 _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a), let unit else { return 2 }
    return optRun(out) {
        guard case .temporal(let t) = x else {
            throw ArrowMetalError.unsupportedType("temporal rounding needs a temporal array, got \(x.arrowFormat)")
        }
        let name = String(cString: unit)
        guard let u = TemporalRoundUnit(rawValue: name) else {
            throw ArrowMetalError.invalidArrowArray("unknown rounding unit \(name)")
        }
        let options = RoundTemporalOptions(multiple: Int(multiple), unit: u,
                                           weekStartsMonday: flags & 1 != 0,
                                           ceilIsStrictlyGreater: flags & 2 != 0,
                                           calendarBasedOrigin: flags & 4 != 0)
        switch mode {
        case 0: return .temporal(try t.floorTemporal(options))
        case 1: return .temporal(try t.ceilTemporal(options))
        case 2: return .temporal(try t.roundTemporal(options))
        default: throw ArrowMetalError.invalidArrowArray("unknown temporal rounding mode \(mode)")
        }
    }
}

// MARK: - list_parent_indices in int64

/// Arrow `list_parent_indices` in int64, which is the width pyarrow returns.
@_cdecl("am_list_parent_indices64")
public func am_list_parent_indices64(_ a: OpaquePointer?,
                                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = optHandle(a) else { return 2 }
    return optRun(out) { .int64(try x.listParentIndices64()) }
}

// MARK: - Type erasure

/// The option-carrying operations on `MetalArray<T>`, erased so each entry point above stays one line.
protocol OptionCOps {
    func optPartitionNth(_ pivot: Int, _ placement: NullPlacement) throws -> MetalArray<Int32>
    func optRank(_ tb: RankTiebreaker, _ descending: Bool, _ placement: NullPlacement) throws -> MetalArray<Int32>
    func optRankQuantile(_ op: Int32, _ descending: Bool, _ placement: NullPlacement) throws -> AnyMetalArray
    func optIsIn(_ set: AnyMetalArray, _ behavior: SetLookupNullMatching) throws -> MetalBooleanArray
    func optIndexIn(_ set: AnyMetalArray, _ behavior: SetLookupNullMatching) throws -> MetalArray<Int32>
    func optUnique(_ order: ValueOrder) throws -> AnyMetalArray
    func optValueCounts(_ order: ValueOrder) throws -> MetalStructArray
    func optDictionaryEncode(_ order: ValueOrder) throws -> (AnyMetalArray, AnyMetalArray)
}

extension MetalArray: OptionCOps {
    func optPartitionNth(_ pivot: Int, _ placement: NullPlacement) throws -> MetalArray<Int32> {
        try partitionNthIndices(pivot, nullPlacement: placement)
    }
    func optRank(_ tb: RankTiebreaker, _ descending: Bool, _ placement: NullPlacement) throws -> MetalArray<Int32> {
        try rank(tiebreaker: tb, descending: descending, nullPlacement: placement)
    }
    func optRankQuantile(_ op: Int32, _ descending: Bool, _ placement: NullPlacement) throws -> AnyMetalArray {
        switch op {
        case 0: return .float64(try rankQuantile(descending: descending, nullPlacement: placement))
        case 1: return .float64(try rankNormal(descending: descending, nullPlacement: placement))
        case 2: return .float32(try rankNormalFloat32(descending: descending, nullPlacement: placement))
        default: throw ArrowMetalError.invalidArrowArray("unknown rank op \(op)")
        }
    }
    private func sameSet(_ set: AnyMetalArray) throws -> MetalArray<T> {
        guard let s = unwrap(set, T.self) else {
            throw ArrowMetalError.unsupportedType("value set is \(set.arrowFormat), not \(T.arrowFormat)")
        }
        return s
    }
    func optIsIn(_ set: AnyMetalArray, _ behavior: SetLookupNullMatching) throws -> MetalBooleanArray {
        try isIn(try sameSet(set), nullMatching: behavior)
    }
    func optIndexIn(_ set: AnyMetalArray, _ behavior: SetLookupNullMatching) throws -> MetalArray<Int32> {
        try indexIn(try sameSet(set), nullMatching: behavior)
    }
    func optUnique(_ order: ValueOrder) throws -> AnyMetalArray { wrap(try unique(order: order)) }
    func optValueCounts(_ order: ValueOrder) throws -> MetalStructArray {
        let (values, counts) = try valueCounts(order: order)
        return try MetalStructArray(length: values.length, nullCount: 0, validity: nil,
                                    names: ["values", "counts"],
                                    children: [wrap(values), .int64(counts)], context: context)
    }
    func optDictionaryEncode(_ order: ValueOrder) throws -> (AnyMetalArray, AnyMetalArray) {
        let (codes, unique) = try dictionaryEncode(order: order)
        return (.int32(codes), wrap(unique))
    }
}

private func withOptionPrimitive<R>(_ a: AnyMetalArray, _ body: (any OptionCOps) throws -> R) throws -> R {
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
    default:
        throw ArrowMetalError.unsupportedType("this operation needs a primitive array, got \(a.arrowFormat)")
    }
}
