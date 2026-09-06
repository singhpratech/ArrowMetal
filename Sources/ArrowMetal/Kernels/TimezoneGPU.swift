import Foundation
import Darwin
import Metal

// The IANA timezone database on the GPU.
//
// A timezone is a step function: a sorted list of instants at which the UTC offset changes, and one
// offset (plus a DST flag and an abbreviation) between consecutive instants. That is host data, but it
// is *small* — `America/New_York` has 559 transitions between 1800 and 2200, `Europe/Berlin` 467,
// `Asia/Kolkata` 7 — so it does not have to stay on the host. This file enumerates the table once per
// zone through Foundation's `TimeZone`, uploads it once, caches it for the process's lifetime, and then
// `local_timestamp`, `assume_timezone`, `is_dst` and the raw offset lookup are each a single GPU pass
// whose per-row work is a ten-step binary search and an add.
//
// The table is built from exactly the two Foundation primitives the host path in `Timezone.swift` uses
// — `secondsFromGMT(for:)` and `nextDaylightSavingTimeTransition(after:)` — so the two paths agree by
// construction, and the build is verified against `secondsFromGMT` at both ends of every interval
// before the table is used. Anything the enumeration cannot describe (a zone Foundation will not
// enumerate, a value outside the covered range, more transitions than the cap) falls back to the host
// path, which remains the reference implementation.

/// One timezone's transition table, resident on the GPU.
///
/// Interval `i` covers the UTC seconds `[transUTC[i-1], transUTC[i])` — unbounded below for `i == 0`
/// and above for `i == count` — and carries `offsets[i]`, `dstFlags[i]` and `abbrev[i]`.
/// `transLocal[i]` is `transUTC[i] + offsets[i+1]`, the local wall-clock instant at which interval
/// `i + 1` begins; it is the key `assume_timezone` searches.
final class TimeZoneTable: @unchecked Sendable {

    /// The UTC range the enumeration covers: 1800-01-01 to 2200-01-01. Everything Arrow can express in
    /// `timestamp[s]` outside that window falls back to the host path.
    static let loSecond: Int64 = -5_364_662_400
    static let hiSecond: Int64 = 7_258_118_400
    /// Refuse to build a table larger than this (no real zone comes close; 1800-2200 tops out at ~560).
    static let maxTransitions = 4_096
    /// Bytes reserved per interval for the `%Z` abbreviation, NUL padded.
    static let abbrevStride = 8

    let name: String
    let count: Int                     // transitions; there are count + 1 intervals
    let offsets: [Int32]
    let dstFlags: [UInt8]
    let transUTC: [Int64]
    let transLocal: [Int64]
    let abbrev: [UInt8]                // (count + 1) * abbrevStride

    let bufTransUTC: MetalArrowBuffer
    let bufTransLocal: MetalArrowBuffer
    let bufOffsets: MetalArrowBuffer
    let bufDST: MetalArrowBuffer
    let bufAbbrev: MetalArrowBuffer

    /// True when the zone never changes offset (UTC, a fixed "+05:30", `Etc/GMT+7`): every kernel here
    /// then degenerates to an add, so the callers short-circuit rather than dispatch.
    var isFixed: Bool { count == 0 }
    /// The single offset of a fixed zone.
    var fixedOffset: Int32 { offsets[0] }

    private init(name: String, transUTC: [Int64], offsets: [Int32], dstFlags: [UInt8],
                 abbrev: [UInt8], context: MetalContext) throws {
        self.name = name
        self.count = transUTC.count
        self.transUTC = transUTC
        self.offsets = offsets
        self.dstFlags = dstFlags
        self.abbrev = abbrev
        self.transLocal = transUTC.enumerated().map { $0.element + Int64(offsets[$0.offset + 1]) }
        func upload<T>(_ a: [T]) throws -> MetalArrowBuffer {
            try a.withUnsafeBytes {
                try MetalArrowBuffer.copy(from: $0.baseAddress ?? UnsafeRawPointer(bitPattern: 8)!,
                                          byteCount: Swift.max($0.count, 8), context: context)
            }
        }
        self.bufTransUTC = try upload(self.transUTC)
        self.bufTransLocal = try upload(self.transLocal)
        self.bufOffsets = try upload(offsets)
        self.bufDST = try upload(dstFlags)
        self.bufAbbrev = try upload(abbrev)
    }

    // MARK: - Building

    private static let lock = NSLock()
    private static var cache: [String: TimeZoneTable?] = [:]

    /// `ARROWMETAL_TZ_HOST=1` disables every GPU timezone path, leaving the host implementation. Set by
    /// the tests so the two can be compared row for row.
    static let hostOnly = ProcessInfo.processInfo.environment["ARROWMETAL_TZ_HOST"] == "1"

    /// The table for `name`, built and uploaded on first use and cached for the process's lifetime.
    /// Returns nil when the zone cannot be tabulated, which is the caller's signal to use the host path.
    ///
    /// Keyed by zone name and device: a table is a handful of kilobytes, so caching the *failure* too
    /// keeps a pathological zone from being re-enumerated on every call.
    static func table(for name: String, context: MetalContext) -> TimeZoneTable? {
        let key = "\(name)#\(UInt(bitPattern: ObjectIdentifier(context)))"
        lock.lock()
        if let hit = cache[key] { lock.unlock(); return hit }
        lock.unlock()
        let built = try? build(name: name, context: context)
        lock.lock()
        cache[key] = built
        lock.unlock()
        return built
    }

    private static func build(name: String, context: MetalContext) throws -> TimeZoneTable? {
        guard let zone = try? resolveTimeZone(name) else { return nil }
        let lo = Date(timeIntervalSince1970: Double(loSecond))
        let hi = Date(timeIntervalSince1970: Double(hiSecond))
        var instants = Set<Int64>()
        var cursor = lo
        while let next = zone.nextDaylightSavingTimeTransition(after: cursor), next < hi {
            guard next > cursor else { return nil }                    // no progress: give up
            guard instants.count < maxTransitions else { return nil }
            instants.insert(Int64(next.timeIntervalSince1970.rounded(.down)))
            cursor = next
        }
        // The zoneinfo file also changes the *designation* without changing the offset — US "war time"
        // becomes "peace time" in September 1945 at the same -0400 — and `%Z` has to see those. Adding
        // them as extra transitions is free for every other kernel: the two intervals either side then
        // simply carry the same offset.
        let zoneinfo = ZoneInfoFile(name: name)
        if let zi = zoneinfo {
            for (k, t) in zi.transitions.enumerated() where t > loSecond && t < hiSecond {
                let prev = k == 0 ? 0 : Int(zi.typeIndex[k - 1]), cur = Int(zi.typeIndex[k])
                guard prev < zi.types.count, cur < zi.types.count else { continue }
                if zi.types[prev].name != zi.types[cur].name { instants.insert(t) }
            }
        }
        guard instants.count <= maxTransitions else { return nil }
        let transUTC = instants.sorted()
        var offsets: [Int32] = [Int32(zone.secondsFromGMT(for: lo))]
        var dstFlags: [UInt8] = [zone.isDaylightSavingTime(for: lo) ? 1 : 0]
        for t in transUTC {
            let d = Date(timeIntervalSince1970: Double(t))
            offsets.append(Int32(zone.secondsFromGMT(for: d)))
            dstFlags.append(zone.isDaylightSavingTime(for: d) ? 1 : 0)
        }
        // Verify: the offset Foundation reports just before and just after every transition must be the
        // one the table claims. A zone that fails this is left to the host path rather than silently
        // answered wrongly.
        for i in 0..<transUTC.count {
            let t = transUTC[i]
            let before = Int32(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(t - 1))))
            if before != offsets[i] { return nil }
        }
        let abbrev = abbreviations(zoneinfo: zoneinfo, zone: zone, offsets: offsets, transUTC: transUTC)
        return try TimeZoneTable(name: name, transUTC: transUTC, offsets: offsets, dstFlags: dstFlags,
                                 abbrev: abbrev, context: context)
    }

    /// One NUL-padded `%Z` abbreviation per interval.
    ///
    /// The names Arrow prints ("CET", "CEST", "EWT", "IST", "-03") are the tz database's own designation
    /// strings, which Foundation does not expose — `TimeZone.abbreviation(for:)` answers "GMT+1" for
    /// Berlin. They are read instead straight out of the compiled zoneinfo file, looked up at the
    /// instant each interval begins so that two types sharing an offset are still told apart (US
    /// "war time" in 1942 is EWT, not EDT, at the same -0400). A zone with no readable zoneinfo file,
    /// or one whose file disagrees with Foundation about the offset, falls back to Foundation's
    /// abbreviation — still a correct, if differently spelled, name for that offset.
    private static func abbreviations(zoneinfo: ZoneInfoFile?, zone: TimeZone, offsets: [Int32],
                                      transUTC: [Int64]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: offsets.count * abbrevStride)
        for i in 0..<offsets.count {
            // The instant the interval begins (one second before the first transition for interval 0).
            let at: Int64 = transUTC.isEmpty ? 0 : (i == 0 ? transUTC[0] - 1 : transUTC[i - 1])
            var text = zoneinfo?.designation(at: at, offset: offsets[i])
            if text == nil {
                text = zone.abbreviation(for: Date(timeIntervalSince1970: Double(at)))
            }
            let bytes = Array((text ?? "UTC").utf8.prefix(abbrevStride - 1))
            for (k, b) in bytes.enumerated() { out[i * abbrevStride + k] = b }
        }
        return out
    }
}

/// The parts of a compiled zoneinfo (TZif) file `%Z` needs: the transition instants, the local-time
/// type each one selects, and each type's UTC offset and designation. The 64-bit (version 2) block is
/// preferred so that transitions past 2038 are described; a version-1 file falls back to its own block.
private struct ZoneInfoFile {
    var transitions: [Int64] = []
    var typeIndex: [UInt8] = []
    var types: [(offset: Int32, name: String)] = []

    init?(name: String) {
        guard !name.contains(".."), !name.hasPrefix("/"),
              name.allSatisfy({ $0.isLetter || $0.isNumber || "/_+-".contains($0) }) else { return nil }
        var data: Data? = nil
        for root in ["/usr/share/zoneinfo/", "/var/db/timezone/zoneinfo/"] {
            if let d = try? Data(contentsOf: URL(fileURLWithPath: root + name)) { data = d; break }
        }
        guard let d = data, d.count > 44,
              d[0] == 0x54, d[1] == 0x5A, d[2] == 0x69, d[3] == 0x66 else { return nil }   // "TZif"
        let version = d[4]

        func be32(_ at: Int) -> Int {
            Int(d[at]) << 24 | Int(d[at + 1]) << 16 | Int(d[at + 2]) << 8 | Int(d[at + 3])
        }
        /// Reads the six counts of a header at `at`, or nil when they are not sane.
        func counts(_ at: Int) -> (isutc: Int, isstd: Int, leap: Int, time: Int, type: Int, char: Int)? {
            guard at + 44 <= d.count else { return nil }
            let c = (be32(at + 20), be32(at + 24), be32(at + 28), be32(at + 32), be32(at + 36), be32(at + 40))
            guard c.0 >= 0, c.1 >= 0, c.2 >= 0, c.3 >= 0, c.4 > 0, c.4 < 4096, c.5 >= 0, c.5 < 8192
            else { return nil }
            return c
        }
        guard let c1 = counts(0) else { return nil }
        var base = 44, timeWidth = 4, leapWidth = 8, c = c1
        if version >= 0x32 {                                    // '2' or later: use the 64-bit block
            let v1size = c1.time * 5 + c1.type * 6 + c1.char + c1.leap * 8 + c1.isstd + c1.isutc
            guard let c2 = counts(44 + v1size) else { return nil }
            base = 44 + v1size + 44
            timeWidth = 8
            leapWidth = 12
            c = c2
        }
        _ = leapWidth
        let typesAt = base + c.time * (timeWidth + 1)
        let charsAt = typesAt + c.type * 6
        guard charsAt + c.char <= d.count else { return nil }
        for i in 0..<c.time {
            let p = base + i * timeWidth
            var v: Int64 = 0
            for k in 0..<timeWidth { v = (v << 8) | Int64(d[p + k]) }
            if timeWidth == 4 { v = Int64(Int32(truncatingIfNeeded: v)) }
            transitions.append(v)
            typeIndex.append(d[base + c.time * timeWidth + i])
        }
        for i in 0..<c.type {
            let p = typesAt + i * 6
            let raw = UInt32(d[p]) << 24 | UInt32(d[p + 1]) << 16 | UInt32(d[p + 2]) << 8 | UInt32(d[p + 3])
            var idx = charsAt + Int(d[p + 5])
            var bytes: [UInt8] = []
            while idx < d.count, d[idx] != 0, bytes.count < TimeZoneTable.abbrevStride - 1 {
                bytes.append(d[idx]); idx += 1
            }
            types.append((Int32(bitPattern: raw), String(decoding: bytes, as: UTF8.self)))
        }
        guard !types.isEmpty else { return nil }
    }

    /// The designation in force at UTC second `at`, provided its offset is the `offset` the caller
    /// expects; nil when the file disagrees, so the caller can fall back.
    func designation(at: Int64, offset: Int32) -> String? {
        var lo = 0, hi = transitions.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if transitions[mid] <= at { lo = mid + 1 } else { hi = mid }
        }
        // Before the first transition tzcode uses the first non-DST type, which is type 0 in practice.
        let t = lo == 0 ? 0 : Int(typeIndex[lo - 1])
        guard t < types.count, types[t].offset == offset, !types[t].name.isEmpty else { return nil }
        return types[t].name
    }
}

/// Mirrors the `tz_params` struct in `TimezoneGPUSource`.
struct TZParams {
    var per: Int64
    var loSecond: Int64
    var hiSecond: Int64
    var count: UInt32
    var hasValidity: UInt32
    var ambiguous: UInt32
    var nonexistent: UInt32
}

/// Which value the flags buffer reports; the slot numbering is the kernel's.
enum TZFlagSlot: Int { case outOfRange = 0, ambiguous = 1, nonexistent = 2 }

/// The shared dispatch plumbing: three atomic flag slots read back after every pass.
enum TZDispatch {
    static let flagNone: UInt32 = 0xFFFF_FFFF

    static func flagBuffer(_ ctx: MetalContext) throws -> MetalArrowBuffer {
        let b = try MetalArrowBuffer.allocate(byteCount: 16, zeroed: false, context: ctx)
        let p = b.mutableTyped(UInt32.self)
        for i in 0..<4 { p[i] = flagNone }
        return b
    }

    /// The lowest row index that tripped `slot`, or nil.
    static func flag(_ b: MetalArrowBuffer, _ slot: TZFlagSlot) -> Int? {
        let v = b.typed(UInt32.self)[slot.rawValue]
        return v == flagNone ? nil : Int(v)
    }

    static func pipeline(_ ctx: MetalContext, _ fn: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: TimezoneGPUSource.source, function: fn, cacheKey: "timezonegpu/\(fn)")
    }
}

extension MetalTemporalArray {

    /// The GPU table for `name`, or nil when this column cannot use it (not int64 storage, a virtual
    /// GPU, or a zone Foundation will not enumerate).
    ///
    /// Set `ARROWMETAL_TZ_HOST=1` to force every timezone function onto the host path; the tests use it
    /// to check the two implementations against each other row for row.
    func timezoneTable(_ name: String) -> TimeZoneTable? {
        guard type.usesInt64, !context.isVirtualDevice, !TimeZoneTable.hostOnly else { return nil }
        return TimeZoneTable.table(for: name, context: context)
    }

    /// Rebuilds a timestamp array of `newType` around a freshly written int64 values buffer, sharing
    /// this array's validity.
    private func timestampFrom(_ out: MetalArrowBuffer, _ newType: ArrowTemporalType) throws -> MetalTemporalArray {
        try MetalTemporalArray(type: newType,
                               MetalArray<Int64>(length: length, nullCount: nullCount, validity: validity,
                                                 values: out, context: context))
    }

    private func tzParams(_ table: TimeZoneTable, per: Int64,
                          ambiguous: UInt32 = 0, nonexistent: UInt32 = 0) -> TZParams {
        TZParams(per: per, loSecond: TimeZoneTable.loSecond, hiSecond: TimeZoneTable.hiSecond,
                 count: UInt32(table.count), hasValidity: validity == nil ? 0 : 1,
                 ambiguous: ambiguous, nonexistent: nonexistent)
    }

    // MARK: - local_timestamp

    /// `local_timestamp` on the GPU, or nil when the table cannot answer (out-of-range values included,
    /// which is discovered only after the pass and costs one discarded dispatch).
    func localTimestampGPU(unit: ArrowTemporalUnit, zone: String) throws -> MetalTemporalArray? {
        guard let table = timezoneTable(zone) else { return nil }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let per = unit.perSecond
        if table.isFixed {                                       // no transitions: a plain add
            let shifted = try int64Values().arithmetic(.add, Int64(table.fixedOffset) * per)
            return try MetalTemporalArray(type: .timestamp(unit, timezone: nil), shifted)
        }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        if n > 0 {
            let flags = try TZDispatch.flagBuffer(ctx)
            let pso = try TZDispatch.pipeline(ctx, "tz_local")
            var P = tzParams(table, per: per)
            let vals = values, vb = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBytes(&P, length: MemoryLayout<TZParams>.stride, index: 3)
                enc.setBuffer(table.bufTransUTC.mtl, offset: table.bufTransUTC.offset, index: 4)
                enc.setBuffer(table.bufOffsets.mtl, offset: table.bufOffsets.offset, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            if TZDispatch.flag(flags, .outOfRange) != nil { return nil }
        }
        return try timestampFrom(out, .timestamp(unit, timezone: nil))
    }

    // MARK: - assume_timezone

    /// `assume_timezone` on the GPU, or nil when the table cannot answer. Throws for an ambiguous or
    /// nonexistent local time under the `raise` policy, naming the same row the host path would.
    func assumeTimezoneGPU(unit: ArrowTemporalUnit, zone: String, name: String,
                           ambiguous: ArrowAmbiguousHandling,
                           nonexistent: ArrowNonexistentHandling) throws -> MetalTemporalArray? {
        guard let table = timezoneTable(zone) else { return nil }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let per = unit.perSecond
        if table.isFixed {
            let shifted = try int64Values().arithmetic(.sub, Int64(table.fixedOffset) * per)
            return try MetalTemporalArray(type: .timestamp(unit, timezone: name), shifted)
        }
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        if n > 0 {
            let flags = try TZDispatch.flagBuffer(ctx)
            let pso = try TZDispatch.pipeline(ctx, "tz_assume")
            var P = tzParams(table, per: per,
                             ambiguous: UInt32(policyCode(ambiguous.rawValue)),
                             nonexistent: UInt32(policyCode(nonexistent.rawValue)))
            let vals = values, vb = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBytes(&P, length: MemoryLayout<TZParams>.stride, index: 3)
                enc.setBuffer(table.bufTransUTC.mtl, offset: table.bufTransUTC.offset, index: 4)
                enc.setBuffer(table.bufTransLocal.mtl, offset: table.bufTransLocal.offset, index: 5)
                enc.setBuffer(table.bufOffsets.mtl, offset: table.bufOffsets.offset, index: 6)
                enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 8)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            if TZDispatch.flag(flags, .outOfRange) != nil { return nil }
            if let i = TZDispatch.flag(flags, .ambiguous) {
                throw ArrowMetalError.invalidArrowArray(
                    "timestamp \(self[i] ?? 0) is ambiguous in \(name) (it occurs twice)")
            }
            if let i = TZDispatch.flag(flags, .nonexistent) {
                throw ArrowMetalError.invalidArrowArray(
                    "timestamp \(self[i] ?? 0) does not exist in \(name) (it falls in a DST gap)")
            }
        }
        return try timestampFrom(out, .timestamp(unit, timezone: name))
    }

    private func policyCode(_ s: String) -> Int {
        switch s { case "earliest": return 1; case "latest": return 2; default: return 0 }
    }

    // MARK: - is_dst

    /// `is_dst` on the GPU, or nil when the table cannot answer.
    func isDSTGPU(unit: ArrowTemporalUnit, zone: String) throws -> MetalBooleanArray? {
        guard let table = timezoneTable(zone) else { return nil }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 4),
                                                zeroed: true, context: ctx)
        if table.isFixed {
            // A zone with no transitions never observes DST unless it is permanently on it.
            if table.dstFlags[0] != 0 {
                let p = out.mutableTyped(UInt8.self)
                for i in 0..<n { Bitmap.set(p, i) }
            }
            return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out,
                                     context: ctx)
        }
        if n > 0 {
            let flags = try TZDispatch.flagBuffer(ctx)
            let pso = try TZDispatch.pipeline(ctx, "tz_is_dst")
            var P = tzParams(table, per: unit.perSecond)
            let vals = values, vb = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBytes(&P, length: MemoryLayout<TZParams>.stride, index: 3)
                enc.setBuffer(table.bufTransUTC.mtl, offset: table.bufTransUTC.offset, index: 4)
                enc.setBuffer(table.bufDST.mtl, offset: table.bufDST.offset, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: BitmapOps.words(bits: n))
            }
            if TZDispatch.flag(flags, .outOfRange) != nil { return nil }
        }
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - to_timezone

    /// Arrow's `cast` between timezones: retags a `timestamp` with `tz` (or with none when `tz` is nil).
    ///
    /// Metadata only, and deliberately so — Arrow stores a timestamp as UTC ticks whatever timezone the
    /// type carries, so moving a column from `America/New_York` to `Asia/Tokyo` changes no value. The
    /// function that *does* change values is `local_timestamp`, and the one that reads naive values as
    /// wall clocks is `assume_timezone`. The zone is resolved first, so an unknown one is an error here
    /// rather than at the first use.
    public func toTimezone(_ tz: String?) throws -> MetalTemporalArray {
        guard case .timestamp(let unit, _) = type else {
            throw ArrowMetalError.unsupportedType("to_timezone is only defined for timestamps, not \(type.arrowFormat)")
        }
        if let tz, !tz.isEmpty {
            _ = try resolveTimeZone(tz)
            return try MetalTemporalArray(type: .timestamp(unit, timezone: tz), try int64Values())
        }
        return try MetalTemporalArray(type: .timestamp(unit, timezone: nil), try int64Values())
    }

    // MARK: - utc_offset

    /// The UTC offset in seconds that applies to each value in the column's own timezone, as int32.
    ///
    /// This is the lookup `local_timestamp` and `strftime`'s `%z` / `%Z` are built on, exposed on its
    /// own because it is the cheapest way to ask what a zone was doing at a set of instants: one GPU
    /// pass, no calendar arithmetic. A naive timestamp answers 0 everywhere, as a fixed-offset zone
    /// answers its own offset.
    public func utcOffset() throws -> MetalArray<Int32> {
        guard case .timestamp(let unit, let tzName) = type else {
            throw ArrowMetalError.unsupportedType("utc_offset is only defined for timestamps, not \(type.arrowFormat)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        guard let tzName, !tzName.isEmpty else {
            return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
        }
        if let table = timezoneTable(tzName), !table.isFixed, n > 0 {
            let flags = try TZDispatch.flagBuffer(ctx)
            let pso = try TZDispatch.pipeline(ctx, "tz_offset")
            var P = tzParams(table, per: unit.perSecond)
            let vals = values, vb = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBytes(&P, length: MemoryLayout<TZParams>.stride, index: 3)
                enc.setBuffer(table.bufTransUTC.mtl, offset: table.bufTransUTC.offset, index: 4)
                enc.setBuffer(table.bufOffsets.mtl, offset: table.bufOffsets.offset, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            if TZDispatch.flag(flags, .outOfRange) == nil {
                return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out,
                                         context: ctx)
            }
        }
        // Host fallback: a fixed zone, an unenumerable one, or values outside the tabulated range.
        let zone = try resolveTimeZone(tzName)
        let per = unit.perSecond
        let src = try int64Values()
        withExtendedLifetime(src) {
            let sp = src.valuePointer
            let dp = out.mutableTyped(Int32.self)
            let bm = src.validity?.typed(UInt8.self)
            for i in 0..<n {
                if let bm, !Bitmap.isSet(bm, i) { dp[i] = 0; continue }
                var s = sp[i] / per
                if sp[i] % per != 0 && (sp[i] < 0) != (per < 0) { s -= 1 }
                dp[i] = Int32(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(s))))
            }
        }
        return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }
}
