import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the remaining Arrow type-matrix rows and the type-adjacent functions:
// `null`, `float16`, `decimal32` / `decimal64`, the three `interval` layouts, `fixed_size_binary`,
// `list_view` / `large_list_view`, extension types, and the two timezone functions.
//
// `am_import` / `am_export` already carry every one of these through the C Data Interface, and
// `am_filter` / `am_take` / `am_slice` already work on them, because they go through `AnyMetalArray`;
// what this file adds is the compute each type has. Handles and error reporting follow ArrowMetalC.swift
// exactly, so `am_last_error()` reports failures from here too. The op tables are in include/arrowmetal.h.

private let txErrorKey = "ArrowMetalC.lastError"
private func txFail(_ e: Error) -> Int32 { Thread.current.threadDictionary[txErrorKey] = "\(e)"; return 1 }

@inline(__always) private func txHandle(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
private func txRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { return txFail(error) }
}

private var txStrings: [String: UnsafeMutablePointer<CChar>] = [:]
private let txStringLock = NSLock()
/// A stable C string per distinct value, so a returned pointer stays valid for the process's lifetime.
private func txStable(_ s: String) -> UnsafePointer<CChar> {
    txStringLock.lock(); defer { txStringLock.unlock() }
    if let c = txStrings[s] { return UnsafePointer(c) }
    let c = strdup(s)!
    txStrings[s] = c
    return UnsafePointer(c)
}

// MARK: - float16

/// Casts between `float16` and `float32` on the GPU. `to_half` non-zero narrows a float32 column to
/// float16 (round to nearest-even, overflow to +/-infinity); zero widens a float16 column to float32
/// (exact). Every other input type is an error.
@_cdecl("am_cast_float16")
public func am_cast_float16(_ a: OpaquePointer?, _ to_half: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        if to_half != 0 {
            guard case .float32(let f) = x.storageArray else {
                throw ArrowMetalError.unsupportedType("casting to float16 needs a float32 column, got \(x.arrowFormat)")
            }
            return .float16(try f.toFloat16())
        }
        guard case .float16(let h) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("casting from float16 needs a float16 column, got \(x.arrowFormat)")
        }
        return .float32(try h.toFloat32())
    }
}

// MARK: - decimal32 / decimal64

/// Widens a `decimal32` / `decimal64` column to `decimal128` on the GPU, so the existing `am_decimal_op`
/// kernels can run on it. A decimal128 column is returned unchanged.
@_cdecl("am_decimal_widen")
public func am_decimal_widen(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        switch x.storageArray {
        case .smallDecimal(let d): return .decimal(try d.toDecimal128())
        case .decimal(let d): return .decimal(d)
        default: throw ArrowMetalError.unsupportedType("am_decimal_widen needs a decimal column, got \(x.arrowFormat)")
        }
    }
}

/// Narrows a `decimal128` column back to `decimal32` (`bit_width` 32) or `decimal64` (64) on the GPU,
/// keeping the scale. A value that does not fit wraps, which is Arrow's unchecked cast.
@_cdecl("am_decimal_narrow")
public func am_decimal_narrow(_ a: OpaquePointer?, _ bit_width: Int32, _ precision: Int64,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        guard case .decimal(let d) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_decimal_narrow needs a decimal128 column, got \(x.arrowFormat)")
        }
        let maxP = bit_width == 32 ? 9 : 18
        let p = precision > 0 ? Int(precision) : Swift.min(d.type.precision, maxP)
        return .smallDecimal(try d.narrowed(to: try ArrowSmallDecimalType(precision: p, scale: d.type.scale,
                                                                          bitWidth: Int(bit_width))))
    }
}

// MARK: - fixed_size_binary

/// Arrow `equal` (`op` 0) / `not_equal` (`op` 1) over a `fixed_size_binary` column, on the GPU.
/// Pass `b_or_null` for the array form, or `scalar_bytes` / `len` (exactly the element width) for the
/// scalar form. Null in, null out; the array form ANDs the two validity bitmaps.
@_cdecl("am_fixed_binary_compare")
public func am_fixed_binary_compare(_ a: OpaquePointer?, _ op: Int32, _ b_or_null: OpaquePointer?,
                                    _ scalar_bytes: UnsafePointer<UInt8>?, _ len: Int64,
                                    _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        guard case .fixedBinary(let f) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_fixed_binary_compare needs a fixed_size_binary column, got \(x.arrowFormat)")
        }
        guard op == 0 || op == 1 else {
            throw ArrowMetalError.invalidArrowArray("am_fixed_binary_compare op must be 0 (equal) or 1 (not_equal)")
        }
        let cmp: CompareOp = op == 0 ? .eq : .ne
        if let bh = b_or_null, let y = txHandle(bh) {
            guard case .fixedBinary(let g) = y.storageArray else {
                throw ArrowMetalError.unsupportedType("cannot compare \(x.arrowFormat) with \(y.arrowFormat)")
            }
            return .boolean(try f.compare(cmp, g))
        }
        guard let sp = scalar_bytes else {
            throw ArrowMetalError.invalidArrowArray("am_fixed_binary_compare needs either b or scalar_bytes")
        }
        return .boolean(try f.compare(cmp, Array(UnsafeBufferPointer(start: sp, count: Int(len)))))
    }
}

/// FNV-1a 64 over each element's bytes (an ArrowMetal extension, not Arrow's `hash64`), on the GPU.
/// Null in, null out. The output is a uint64 column.
@_cdecl("am_fixed_binary_hash64")
public func am_fixed_binary_hash64(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        guard case .fixedBinary(let f) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_fixed_binary_hash64 needs a fixed_size_binary column, got \(x.arrowFormat)")
        }
        return .uint64(try f.hash64())
    }
}

// MARK: - interval

/// Arrow `add(timestamp | date, interval)` on the GPU: `interval` is a "tiM", "tiD" or "tin" column of
/// the same length as `a`, or of length 1 to broadcast. Month arithmetic clamps the day to the target
/// month's length. The result has `a`'s type and is null wherever either side is.
@_cdecl("am_add_interval")
public func am_add_interval(_ a: OpaquePointer?, _ interval: OpaquePointer?,
                            _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), let ivh = interval, let iv = txHandle(ivh), out != nil else { return 2 }
    return txRun(out) {
        guard case .temporal(let t) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_add_interval needs a date or timestamp column, got \(x.arrowFormat)")
        }
        guard case .interval(let i) = iv.storageArray else {
            throw ArrowMetalError.unsupportedType("am_add_interval needs an interval column, got \(iv.arrowFormat)")
        }
        return .temporal(try t.addInterval(i))
    }
}

// MARK: - nested extras

/// Arrow `list_parent_indices`: for every child element the list references, the int32 index of the row
/// that covers it. Accepts a list, large_list, fixed_size_list or map column. Never has nulls.
@_cdecl("am_list_parent_indices")
public func am_list_parent_indices(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) { .int32(try x.storageArray.listParentIndices()) }
}

/// Arrow `list_slice`: `row[start:stop:step]` for every row, as a variable-length list ("+l").
/// A negative `stop` means "to the end of each row"; `start` must be >= 0 and `step` >= 1.
@_cdecl("am_list_slice")
public func am_list_slice(_ a: OpaquePointer?, _ start: Int64, _ stop: Int64, _ step: Int64,
                          _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        try x.storageArray.listSlice(start: Int(start), stop: stop < 0 ? nil : Int(stop), step: Int(step))
    }
}

/// Arrow `map_lookup`: the value(s) whose key matches, per row.
///
/// `occurrence` is 0 first, 1 last, 2 all. For a map with utf8 or binary keys the key is `key_bytes` /
/// `len`; for a map with integer keys it is the first 8 bytes of `key_bytes` read as a little-endian
/// int64 (`len` must be 8). `first` / `last` return the item type, `all` returns a list of it, and both
/// are null where the row is null or the key is absent.
@_cdecl("am_map_lookup")
public func am_map_lookup(_ a: OpaquePointer?, _ key_bytes: UnsafePointer<UInt8>?, _ len: Int64,
                          _ occurrence: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), let kb = key_bytes, out != nil else { return 2 }
    return txRun(out) {
        guard case .map(let m) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_map_lookup needs a map column, got \(x.arrowFormat)")
        }
        guard let occ = MapLookupOccurrence(rawValue: Int(occurrence)) else {
            throw ArrowMetalError.invalidArrowArray("am_map_lookup occurrence must be 0 (first), 1 (last) or 2 (all)")
        }
        let bytes = Array(UnsafeBufferPointer(start: kb, count: Int(len)))
        let key: MapLookupKey
        switch m.keys {
        case .string, .binary: key = .bytes(bytes)
        default:
            guard bytes.count == 8 else {
                throw ArrowMetalError.invalidArrowArray("an integer map key must be 8 little-endian bytes, got \(bytes.count)")
            }
            var v: Int64 = 0
            for i in 0..<8 { v |= Int64(bytes[i]) << (8 * Int64(i)) }
            key = .integer(v)
        }
        return try m.mapLookup(key, occurrence: occ)
    }
}

// MARK: - timezones (CPU)

private func tzHandling(_ v: Int32) -> String { ["raise", "earliest", "latest"][Swift.max(0, Swift.min(2, Int(v)))] }

/// Arrow `assume_timezone`: reads a naive timestamp column as wall-clock times in `tz` and returns the
/// instants they name, tagged with that timezone. **CPU** — the tz database is host data.
///
/// `ambiguous` and `nonexistent` are 0 raise (the default), 1 earliest, 2 latest.
@_cdecl("am_assume_timezone")
public func am_assume_timezone(_ a: OpaquePointer?, _ tz: UnsafePointer<CChar>?, _ ambiguous: Int32,
                               _ nonexistent: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), let tz, out != nil else { return 2 }
    let name = String(cString: tz)
    return txRun(out) {
        guard case .temporal(let t) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_assume_timezone needs a timestamp column, got \(x.arrowFormat)")
        }
        let amb = ArrowAmbiguousHandling(rawValue: tzHandling(ambiguous)) ?? .raise
        let non = ArrowNonexistentHandling(rawValue: tzHandling(nonexistent)) ?? .raise
        return .temporal(try t.assumeTimezone(name, ambiguous: amb, nonexistent: non))
    }
}

/// Arrow `local_timestamp`: the wall-clock time each instant names in the column's own timezone, as a
/// naive timestamp of the same unit. A column with no timezone comes back unchanged. **CPU**.
@_cdecl("am_local_timestamp")
public func am_local_timestamp(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        guard case .temporal(let t) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_local_timestamp needs a timestamp column, got \(x.arrowFormat)")
        }
        return .temporal(try t.localTimestamp())
    }
}

// MARK: - extension types

/// `ARROW:extension:name` of an extension column, or NULL when the column is not an extension type.
/// The pointer is owned by the library and stays valid for the process's lifetime.
@_cdecl("am_extension_name")
public func am_extension_name(_ a: OpaquePointer?) -> UnsafePointer<CChar>? {
    guard let x = txHandle(a), let name = x.extensionName else { return nil }
    return txStable(name)
}

/// `ARROW:extension:metadata` of an extension column as raw bytes, or NULL when there is none.
/// `out_len` receives the byte count. The pointer is owned by the library and stays valid for the
/// process's lifetime (the metadata of an extension type is a short, fixed string in practice).
@_cdecl("am_extension_metadata")
public func am_extension_metadata(_ a: OpaquePointer?, _ out_len: UnsafeMutablePointer<Int64>?) -> UnsafePointer<CChar>? {
    guard let x = txHandle(a), let md = x.extensionMetadata else { out_len?.pointee = 0; return nil }
    out_len?.pointee = Int64(md.count)
    return txStable(String(decoding: md, as: UTF8.self))
}

/// The storage column of an extension array (the array itself for every other type), so a caller can run
/// the ordinary kernels on it without going through an export/import round trip.
@_cdecl("am_extension_storage")
public func am_extension_storage(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) { x.storageArray }
}

/// Tags a column as the storage of an extension type, so `am_export` writes the two metadata keys and a
/// consumer that knows the type reconstructs it. `metadata` may be NULL (with `metadata_len` 0).
@_cdecl("am_extension_wrap")
public func am_extension_wrap(_ a: OpaquePointer?, _ name: UnsafePointer<CChar>?,
                              _ metadata: UnsafePointer<UInt8>?, _ metadata_len: Int64,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), let name, out != nil else { return 2 }
    let n = String(cString: name)
    let md = metadata.map { Array(UnsafeBufferPointer(start: $0, count: Int(metadata_len))) }
    return txRun(out) { x.asExtensionType(name: n, metadata: md) }
}

// MARK: - null

/// Builds a `null` column of `length` elements (there is nothing to import, so this is how a caller makes
/// one without going through the C Data Interface).
@_cdecl("am_null_array")
public func am_null_array(_ length: Int64, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard out != nil, length >= 0 else { return 2 }
    return txRun(out) { .null(MetalNullArray(length: Int(length))) }
}

// MARK: - interval differences

/// The three Arrow difference functions that return an interval, on the GPU: `kind` 0
/// `month_interval_between` ("tiM"), 1 `day_time_interval_between` ("tiD"), 2
/// `month_day_nano_interval_between` ("tin"). `a` is Arrow's `start` and `b` its `end`; both must be
/// date or timestamp columns of the same length, and the result is null wherever either side is.
@_cdecl("am_interval_between")
public func am_interval_between(_ a: OpaquePointer?, _ b: OpaquePointer?, _ kind: Int32,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), let bh = b, let y = txHandle(bh), out != nil else { return 2 }
    return txRun(out) {
        guard let k = ArrowIntervalBetween(rawValue: Int(kind)) else {
            throw ArrowMetalError.invalidArrowArray("am_interval_between kind must be 0 (month), 1 (day_time) or 2 (month_day_nano)")
        }
        return try x.intervalBetween(y, kind: k)
    }
}

/// The fields of an interval column, as three plain int32 / int32 / int64 columns, so a caller can read
/// the values even where its Arrow binding cannot represent the interval type itself (pyarrow 25 cannot
/// wrap `interval[month]` or `interval[day_time]` arrays in Python). `field` is 0 months, 1 days,
/// 2 nanoseconds; a field the layout does not carry comes back as zeros.
@_cdecl("am_interval_field")
public func am_interval_field(_ a: OpaquePointer?, _ field: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = txHandle(a), out != nil else { return 2 }
    return txRun(out) {
        guard case .interval(let iv) = x.storageArray else {
            throw ArrowMetalError.unsupportedType("am_interval_field needs an interval column, got \(x.arrowFormat)")
        }
        guard field >= 0, field <= 2 else {
            throw ArrowMetalError.invalidArrowArray("am_interval_field field must be 0 (months), 1 (days) or 2 (nanoseconds)")
        }
        let n = iv.length
        let values = iv.toArray()
        if field == 2 {
            var out64: [Int64?] = []
            out64.reserveCapacity(n)
            for v in values { out64.append(v.map { $0.nanoseconds }) }
            return .int64(try MetalArray<Int64>(out64, context: iv.context))
        }
        var out32: [Int32?] = []
        out32.reserveCapacity(n)
        for v in values { out32.append(v.map { field == 0 ? $0.months : $0.days }) }
        return .int32(try MetalArray<Int32>(out32, context: iv.context))
    }
}
