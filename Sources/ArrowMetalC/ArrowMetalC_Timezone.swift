import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the two timezone functions the GPU transition table made cheap enough to expose on their
// own: the per-row UTC offset and the metadata-only retag between zones.
//
// `am_assume_timezone`, `am_local_timestamp` (ArrowMetalC_TypesExtra.swift), `is_dst` (op 5 of
// `am_temporal_extra`) and `strftime` / `strptime` (`am_parse`) all sit on the same table and keep
// their existing entry points. Handles and error reporting follow ArrowMetalC.swift exactly, so
// `am_last_error()` reports failures from here too.

private let tzErrorKey = "ArrowMetalC.lastError"
private func tzFail(_ e: Error) -> Int32 { Thread.current.threadDictionary[tzErrorKey] = "\(e)"; return 1 }

@inline(__always) private func tzTemporal(_ p: OpaquePointer?) throws -> MetalTemporalArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let any = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    guard case .temporal(let t) = any.storageArray else {
        throw ArrowMetalError.unsupportedType("expected a timestamp column, got \(any.arrowFormat)")
    }
    return t
}

private func tzRun(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { return tzFail(error) }
}

/// The UTC offset in seconds that applies to each value in the column's own timezone, as int32.
///
/// **GPU**: one pass over the zone's transition table. A naive timestamp answers 0 everywhere and a
/// fixed-offset zone answers its own offset.
@_cdecl("am_utc_offset")
public func am_utc_offset(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    return tzRun(out) { .int32(try tzTemporal(a).utcOffset()) }
}

/// Arrow's `cast` between timezones: retags a timestamp with `tz`, or strips the timezone when `tz` is
/// NULL or empty. Metadata only — Arrow stores a timestamp as UTC ticks whatever timezone the type
/// carries, so no value changes. An unknown zone is an error.
@_cdecl("am_to_timezone")
public func am_to_timezone(_ a: OpaquePointer?, _ tz: UnsafePointer<CChar>?,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    let name = tz.map { String(cString: $0) }
    return tzRun(out) { .temporal(try tzTemporal(a).toTimezone(name)) }
}
