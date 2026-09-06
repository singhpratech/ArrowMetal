import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the temporal, binary and dictionary types. `am_format` already reports their Arrow format
// strings ("tdD", "tss:UTC", "z", and the index format "i" for dictionary arrays) through AnyMetalArray.

private let errorKey = "ArrowMetalC.lastError"
private func fail(_ e: Error) -> Int32 { Thread.current.threadDictionary[errorKey] = "\(e)"; return 1 }

@inline(__always) private func array(_ p: OpaquePointer?) -> AnyMetalArray? {
    guard let p else { return nil }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func produce(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>) {
    out.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
}

/// Extracts a calendar field from a temporal array (UTC): 0 year, 1 month, 2 day, 3 day-of-week
/// (Monday = 0), 4 hour, 5 minute, 6 second. The result is an int32 array with the same validity.
@_cdecl("am_temporal_extract")
public func am_temporal_extract(_ a: OpaquePointer?, _ field: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do {
        guard case .temporal(let t) = x else {
            throw ArrowMetalError.unsupportedType("expected a temporal array, got \(x.arrowFormat)")
        }
        guard let f = TemporalField(rawValue: Int(field)) else {
            throw ArrowMetalError.invalidArrowArray("unknown temporal field \(field)")
        }
        produce(.int32(try t.extract(f)), out)
        return 0
    } catch { return fail(error) }
}

/// Rescales a timestamp, duration or time array to another unit: 0 seconds, 1 milli, 2 micro, 3 nano.
@_cdecl("am_temporal_cast_unit")
public func am_temporal_cast_unit(_ a: OpaquePointer?, _ unit: Int32, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do {
        guard case .temporal(let t) = x else {
            throw ArrowMetalError.unsupportedType("expected a temporal array, got \(x.arrowFormat)")
        }
        let units: [ArrowTemporalUnit] = [.second, .milli, .micro, .nano]
        guard unit >= 0, Int(unit) < units.count else {
            throw ArrowMetalError.invalidArrowArray("unknown temporal unit \(unit)")
        }
        produce(.temporal(try t.castUnit(to: units[Int(unit)])), out)
        return 0
    } catch { return fail(error) }
}

/// Materialises a dictionary-encoded array (`take` of the values by the codes). Other arrays pass through.
@_cdecl("am_dictionary_decode")
public func am_dictionary_decode(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let x = array(a), let out else { return 2 }
    do { produce(try x.decode(), out); return 0 } catch { return fail(error) }
}
