import Foundation
import CArrowABI

// Arrow `assume_timezone` and `local_timestamp`.
//
// Both run on the **GPU**. A timezone is a step function over a few hundred instants — 559 transitions
// for `America/New_York` between 1800 and 2200 — so the transition table is enumerated once per zone
// from Foundation's `TimeZone`, uploaded once and cached (see `Kernels/TimezoneGPU.swift`), and each
// function is then a single pass whose per-row work is a ten-step binary search and an add.
//
// What remains here is the **host fallback**, kept as the reference implementation and still used for a
// zone Foundation will not enumerate, for values outside the tabulated 1800-2200 window, and on a
// virtual GPU. It converts on the host, sharded over `DispatchQueue.concurrentPerform`, with a
// per-shard cache of the current offset's validity interval so a run of nearby timestamps costs one tz
// lookup rather than one per row. The GPU table is built from exactly the two Foundation primitives
// this path calls, `secondsFromGMT(for:)` and `nextDaylightSavingTimeTransition(after:)`, and verified
// against them at both ends of every interval before use, so the two paths agree by construction.
//
// Past 2038 both paths report the *projected* rules Foundation extends the last known rule with, which
// is what Foundation's own enumeration returns; a zone whose real rules change after that date will
// differ from a tz database released later, exactly as the host path has always differed.
//
// Neither function changes the resolution: a `timestamp[ns]` stays `timestamp[ns]`, and the sub-second
// part of every value is carried across untouched (offsets are whole seconds in the tz database, and
// have been for every zone since 1972).

/// What `assume_timezone` does with a local time that occurs twice (the hour repeated at a DST fall-back).
public enum ArrowAmbiguousHandling: String, Sendable, CaseIterable {
    /// Throw, which is Arrow's default.
    case raise
    /// Take the earlier of the two instants (the larger UTC offset).
    case earliest
    /// Take the later of the two instants (the smaller UTC offset).
    case latest
}

/// What `assume_timezone` does with a local time that never occurs (the hour skipped at a DST spring-forward).
public enum ArrowNonexistentHandling: String, Sendable, CaseIterable {
    /// Throw, which is Arrow's default.
    case raise
    /// The last instant before the gap.
    case earliest
    /// The first instant after the gap.
    case latest
}

/// Resolves an Arrow timezone string: an IANA name ("America/New_York", "UTC") or a fixed offset
/// ("+02:00", "-0530"), which is what a C Data Interface timestamp format may carry.
func resolveTimeZone(_ s: String) throws -> TimeZone {
    if let tz = TimeZone(identifier: s) { return tz }
    if let tz = TimeZone(abbreviation: s) { return tz }
    // Fixed offset forms.
    var body = Substring(s)
    var sign = 1
    if body.hasPrefix("+") { body = body.dropFirst() }
    else if body.hasPrefix("-") { sign = -1; body = body.dropFirst() }
    else { throw ArrowMetalError.unsupportedType("unknown timezone \(s)") }
    let digits = body.filter { $0 != ":" }
    guard digits.allSatisfy({ $0.isNumber }), digits.count == 4 || digits.count == 2,
          let h = Int(digits.prefix(2)) else { throw ArrowMetalError.unsupportedType("unknown timezone \(s)") }
    let m = digits.count == 4 ? (Int(digits.suffix(2)) ?? 0) : 0
    guard let tz = TimeZone(secondsFromGMT: sign * (h * 3600 + m * 60)) else {
        throw ArrowMetalError.unsupportedType("unknown timezone \(s)")
    }
    return tz
}

/// One shard's view of a timezone: `secondsFromGMT` with the current offset's validity interval cached,
/// so a run of adjacent instants costs one lookup.
private struct TimeZoneCache {
    let tz: TimeZone
    private var start: Int64 = 1
    private var end: Int64 = 0
    private var offset: Int = 0

    init(_ tz: TimeZone) { self.tz = tz }

    /// UTC offset in seconds at the instant `t` (seconds since the epoch).
    mutating func offset(at t: Int64) -> Int {
        if t >= start && t < end { return offset }
        let d = Date(timeIntervalSince1970: Double(t))
        offset = tz.secondsFromGMT(for: d)
        start = t
        if let next = tz.nextDaylightSavingTimeTransition(after: d) {
            end = Int64(next.timeIntervalSince1970.rounded(.down))
            if end <= start { end = start + 1 }
        } else {
            end = Int64.max
        }
        return offset
    }

    /// The instant of the transition that creates the gap containing local second `local`, given the
    /// offsets on either side of it.
    func transition(near local: Int64, before: Int, after: Int) -> Int64? {
        let probe = Date(timeIntervalSince1970: Double(local - Int64(after) - 1))
        guard let next = tz.nextDaylightSavingTimeTransition(after: probe) else { return nil }
        let t = Int64(next.timeIntervalSince1970.rounded(.down))
        guard t <= local - Int64(before) else { return nil }
        return t
    }
}

/// Errors these two functions raise, spelled the way Arrow spells them.
enum TimeZoneResolution {
    case ok(Int)                 // the UTC offset to subtract
    case ambiguous(early: Int, late: Int)
    case nonexistent(before: Int, after: Int)
}

/// Every UTC offset that makes `local` (seconds, read as a wall clock) a real instant in `tz`.
private func resolveLocal(_ local: Int64, _ cache: inout TimeZoneCache) -> TimeZoneResolution {
    let tz = cache.tz
    func offsetAt(_ instant: Int64) -> Int { tz.secondsFromGMT(for: Date(timeIntervalSince1970: Double(instant))) }
    let guess = offsetAt(local)
    var candidates: [Int] = [guess]
    let c1 = offsetAt(local - Int64(guess))
    if !candidates.contains(c1) { candidates.append(c1) }
    let c2 = offsetAt(local - Int64(c1))
    if !candidates.contains(c2) { candidates.append(c2) }
    // Probe a day either side so the two offsets around a transition are both seen.
    for probe in [offsetAt(local - 86_400), offsetAt(local + 86_400)] where !candidates.contains(probe) {
        candidates.append(probe)
    }
    var valid: [Int] = []
    for o in candidates where offsetAt(local - Int64(o)) == o && !valid.contains(o) { valid.append(o) }
    if valid.count == 1 { return .ok(valid[0]) }
    if valid.count >= 2 {
        // The earlier instant is the one with the larger offset.
        return .ambiguous(early: valid.max()!, late: valid.min()!)
    }
    return .nonexistent(before: candidates.min()!, after: candidates.max()!)
}

extension MetalTemporalArray {
    /// Arrow `assume_timezone`: reads the values as wall-clock times in `tz` and returns the instants they
    /// name, tagged with that timezone. The unit is unchanged and the sub-second part is carried through.
    ///
    /// **CPU**, because the tz database is host data; sharded over `DispatchQueue.concurrentPerform`.
    /// The input must be a naive `timestamp` (no timezone), which is what Arrow requires.
    ///
    /// A local time that occurs twice (a DST fall-back) or never (a spring-forward) is an error by
    /// default, matching Arrow's `ambiguous = "raise"` / `nonexistent = "raise"`; `earliest` and `latest`
    /// select the earlier / later instant of an ambiguous time, and the last instant before / first
    /// instant after the gap for a nonexistent one.
    public func assumeTimezone(_ tz: String, ambiguous: ArrowAmbiguousHandling = .raise,
                               nonexistent: ArrowNonexistentHandling = .raise) throws -> MetalTemporalArray {
        guard case .timestamp(let unit, let existing) = type, existing == nil else {
            throw ArrowMetalError.unsupportedType("assume_timezone needs a timestamp column with no timezone, got \(type.arrowFormat)")
        }
        let zone = try resolveTimeZone(tz)
        if let gpu = try assumeTimezoneGPU(unit: unit, zone: tz, name: tz,
                                           ambiguous: ambiguous, nonexistent: nonexistent) {
            return gpu
        }
        let per = unit.perSecond
        let src = try int64Values()
        let n = src.length
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 1), zeroed: false, context: ctx)
        var failure: String? = nil
        let lock = NSLock()
        try withExtendedLifetime(src) {
            let sp = src.valuePointer
            let dp = out.mutableTyped(Int64.self)
            let bm = src.validity?.typed(UInt8.self)
            shard(n) { lo, hi in
                var cache = TimeZoneCache(zone)
                for i in lo..<hi {
                    if let bm, !Bitmap.isSet(bm, i) { dp[i] = 0; continue }
                    let v = sp[i]
                    let seconds = tzFloorDiv(v, per)
                    let sub = v - seconds * per
                    switch resolveLocal(seconds, &cache) {
                    case .ok(let o):
                        dp[i] = (seconds - Int64(o)) * per + sub
                    case .ambiguous(let early, let late):
                        switch ambiguous {
                        case .raise:
                            lock.lock()
                            if failure == nil { failure = "timestamp \(v) is ambiguous in \(tz) (it occurs twice)" }
                            lock.unlock()
                            dp[i] = 0
                        case .earliest: dp[i] = (seconds - Int64(early)) * per + sub
                        case .latest: dp[i] = (seconds - Int64(late)) * per + sub
                        }
                    case .nonexistent(let before, let after):
                        let t = cache.transition(near: seconds, before: before, after: after)
                        switch nonexistent {
                        case .raise:
                            lock.lock()
                            if failure == nil { failure = "timestamp \(v) does not exist in \(tz) (it falls in a DST gap)" }
                            lock.unlock()
                            dp[i] = 0
                        case .earliest: dp[i] = t.map { $0 * per - 1 } ?? ((seconds - Int64(before)) * per + sub)
                        case .latest: dp[i] = t.map { $0 * per } ?? ((seconds - Int64(after)) * per + sub)
                        }
                    }
                }
            }
        }
        if let failure { throw ArrowMetalError.invalidArrowArray(failure) }
        let wide = MetalArray<Int64>(length: n, nullCount: src.nullCount, validity: src.validity,
                                     values: out, context: ctx)
        return try MetalTemporalArray(type: .timestamp(unit, timezone: tz), wide)
    }

    /// Arrow `local_timestamp`: the wall-clock time each instant names in the column's own timezone,
    /// returned as a naive `timestamp` of the same unit. **CPU**, for the same reason as `assume_timezone`.
    public func localTimestamp() throws -> MetalTemporalArray {
        guard case .timestamp(let unit, let tzName) = type else {
            throw ArrowMetalError.unsupportedType("local_timestamp needs a timestamp column, got \(type.arrowFormat)")
        }
        guard let tzName, !tzName.isEmpty else {
            // Arrow returns a naive timestamp unchanged.
            return self
        }
        let zone = try resolveTimeZone(tzName)
        if let gpu = try localTimestampGPU(unit: unit, zone: tzName) { return gpu }
        let per = unit.perSecond
        let src = try int64Values()
        let n = src.length
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 1), zeroed: false, context: ctx)
        withExtendedLifetime(src) {
            let sp = src.valuePointer
            let dp = out.mutableTyped(Int64.self)
            let bm = src.validity?.typed(UInt8.self)
            shard(n) { lo, hi in
                var cache = TimeZoneCache(zone)
                for i in lo..<hi {
                    if let bm, !Bitmap.isSet(bm, i) { dp[i] = 0; continue }
                    let v = sp[i]
                    let seconds = tzFloorDiv(v, per)
                    let sub = v - seconds * per
                    let o = cache.offset(at: seconds)
                    dp[i] = (seconds + Int64(o)) * per + sub
                }
            }
        }
        let wide = MetalArray<Int64>(length: n, nullCount: src.nullCount, validity: src.validity,
                                     values: out, context: ctx)
        return try MetalTemporalArray(type: .timestamp(unit, timezone: nil), wide)
    }
}

/// Floor division (the value may be negative, i.e. before 1970).
@inline(__always) private func tzFloorDiv(_ a: Int64, _ b: Int64) -> Int64 {
    var q = a / b
    if a % b != 0 && ((a < 0) != (b < 0)) { q -= 1 }
    return q
}

/// Runs `body` over contiguous chunks of `0..<n` in parallel (one chunk per shard, at least 4096 rows).
private func shard(_ n: Int, _ body: @escaping (Int, Int) -> Void) {
    guard n > 0 else { return }
    let minChunk = 4096
    let cores = ProcessInfo.processInfo.activeProcessorCount
    let shards = Swift.max(1, Swift.min(cores, (n + minChunk - 1) / minChunk))
    if shards == 1 { body(0, n); return }
    let chunk = (n + shards - 1) / shards
    DispatchQueue.concurrentPerform(iterations: shards) { s in
        let lo = s * chunk
        let hi = Swift.min(lo + chunk, n)
        if lo < hi { body(lo, hi) }
    }
}
