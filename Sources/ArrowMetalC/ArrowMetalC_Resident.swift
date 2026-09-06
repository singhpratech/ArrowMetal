import Foundation
import ArrowMetal

// Resident-worker mode and the low-latency dispatch path, over the C ABI. See docs/RESIDENT.md.

/// Requests the persistent GPU worker. Returns 1 if it took effect, 0 if it did not.
/// On Apple silicon it always returns 0: a running kernel and the CPU are not cache coherent
/// through shared storage, so a work-queue worker cannot beat the command-buffer round trip.
/// `am_resident_mode_reason()` has the detail.
@_cdecl("am_resident_mode")
public func am_resident_mode(_ on: Int32) -> Int32 {
    MetalContext.shared.setResidentMode(on != 0) ? 1 : 0
}

/// 1 when a persistent GPU worker is available on this device, 0 otherwise (always 0 today).
@_cdecl("am_resident_mode_available")
public func am_resident_mode_available() -> Int32 { ResidentMode.available ? 1 : 0 }

private let residentReasonC = strdup(ResidentMode.unavailableReason)!

/// Why resident mode is unavailable. Static string, valid for the lifetime of the library.
@_cdecl("am_resident_mode_reason")
public func am_resident_mode_reason() -> UnsafePointer<CChar>? { UnsafePointer(residentReasonC) }

/// Turns the low-latency completion wait on (non-zero) or off (0). On means the command buffer
/// signals an `MTLSharedEvent` whose value the CPU polls out of memory instead of calling
/// `waitUntilCompleted`; measured ~65 µs against ~78 µs per round trip on M4 Max. Default: on.
/// Returns the setting now in force.
@_cdecl("am_low_latency_wait")
public func am_low_latency_wait(_ on: Int32) -> Int32 {
    MetalContext.shared.lowLatencyWait = on != 0
    return MetalContext.shared.lowLatencyWait ? 1 : 0
}

/// Microseconds the CPU spins before blocking on a command buffer. 0 blocks immediately, which
/// costs latency but frees the core. Returns the value now in force.
@_cdecl("am_spin_microseconds")
public func am_spin_microseconds(_ us: Int64) -> Int64 {
    if us >= 0 { MetalContext.shared.spinMicroseconds = UInt64(us) }
    return Int64(MetalContext.shared.spinMicroseconds)
}
