import Foundation
import Darwin
import Metal

/// The unit a temporal value is rounded to by ``MetalTemporalArray/floorTemporal(to:multiple:)`` and
/// friends. `nanosecond` … `day` are fixed-length and round by integer arithmetic in the value's own
/// resolution; `month`, `quarter` and `year` go through the civil calendar.
public enum TemporalRoundUnit: String, Sendable, CaseIterable {
    case nanosecond, microsecond, millisecond, second, minute, hour, day, month, quarter, year

    /// Nanoseconds in one unit, or nil for the calendar units.
    var nanoseconds: Int64? {
        switch self {
        case .nanosecond: return 1
        case .microsecond: return 1_000
        case .millisecond: return 1_000_000
        case .second: return 1_000_000_000
        case .minute: return 60_000_000_000
        case .hour: return 3_600_000_000_000
        case .day: return 86_400_000_000_000
        case .month, .quarter, .year: return nil
        }
    }
    /// Calendar months in one unit, or nil for the fixed-length units.
    var months: Int64? {
        switch self {
        case .month: return 1
        case .quarter: return 3
        case .year: return 12
        default: return nil
        }
    }
}

/// Temporal rounding, temporal arithmetic and the calendar fields `Temporal.swift` does not extract.
///
/// Everything here is **GPU** except `strftime` and `strptime`, which format and parse against the C
/// library on the host. UTC throughout: a timestamp's timezone is carried through as metadata and
/// never applied, exactly as `Temporal.swift`'s extraction kernels treat it.
extension MetalTemporalArray {

    // MARK: - Plumbing

    /// Nanoseconds in one tick of this array's storage (a whole day for `date32`).
    var nanosecondsPerTick: Int64 {
        if case .date32 = type { return 86_400_000_000_000 }
        return 1_000_000_000 / type.unit.perSecond
    }
    /// Ticks of this array's storage in one day.
    var ticksPerDay: Int64 { 86_400_000_000_000 / nanosecondsPerTick }

    private var mslType: String { type.usesInt64 ? "long" : "int" }

    private func mathPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: TemporalMathSource.source(T: mslType),
                             function: fn, cacheKey: "temporalmath/\(mslType)/\(fn)")
    }

    /// Rebuilds an array of this array's own type from a values buffer of the same width.
    private func sameTyped(_ out: MetalArrowBuffer, length n: Int) throws -> MetalTemporalArray {
        if type.usesInt64 {
            return try MetalTemporalArray(type: type, MetalArray<Int64>(length: n, nullCount: nullCount,
                                                                        validity: validity, values: out,
                                                                        context: context))
        }
        return try MetalTemporalArray(type: type, MetalArray<Int32>(length: n, nullCount: nullCount,
                                                                    validity: validity, values: out,
                                                                    context: context))
    }

    // MARK: - Rounding

    /// Arrow `floor_temporal`: the largest multiple of `multiple` × `unit` at or below each value.
    public func floorTemporal(to unit: TemporalRoundUnit, multiple: Int = 1) throws -> MetalTemporalArray {
        try roundTemporal(mode: 0, to: unit, multiple: multiple)
    }
    /// Arrow `ceil_temporal`: the smallest multiple of `multiple` × `unit` at or above each value.
    /// A value already on a boundary is unchanged (Arrow's `ceil_is_strictly_greater = false`).
    public func ceilTemporal(to unit: TemporalRoundUnit, multiple: Int = 1) throws -> MetalTemporalArray {
        try roundTemporal(mode: 1, to: unit, multiple: multiple)
    }
    /// Arrow `round_temporal`: the nearest multiple of `multiple` × `unit`. A value exactly halfway
    /// rounds **up** (toward +infinity), which is Arrow's behaviour and not "half to even".
    public func roundTemporal(to unit: TemporalRoundUnit, multiple: Int = 1) throws -> MetalTemporalArray {
        try roundTemporal(mode: 2, to: unit, multiple: multiple)
    }

    /// The shared driver. Rounding to a unit finer than the array's own resolution is the identity.
    private func roundTemporal(mode: Int, to unit: TemporalRoundUnit, multiple: Int) throws -> MetalTemporalArray {
        guard multiple >= 1 else {
            throw ArrowMetalError.invalidArrowArray("temporal rounding needs multiple >= 1, got \(multiple)")
        }
        let kind: Int, p: Int64
        if let months = unit.months {
            switch type {
            case .date32, .date64, .timestamp:
                kind = 1
                p = months * Int64(multiple)
            case .time32, .time64, .duration:
                throw ArrowMetalError.unsupportedType(
                    "rounding to \(unit.rawValue) needs a date, which \(type.arrowFormat) does not carry")
            }
        } else {
            let ns = unit.nanoseconds!
            let tick = nanosecondsPerTick
            if ns < tick || ns % tick != 0 { return self }        // finer than the storage: nothing to do
            kind = 0
            p = (ns / tick) * Int64(multiple)
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let width = type.usesInt64 ? 8 : 4
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * width, width), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try mathPipeline("temporal_round")
            let vals = values
            var pv = p, tpd = ticksPerDay
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, mode, index: 2)
                Dispatch.setUInt(enc, kind, index: 3)
                enc.setBytes(&pv, length: 8, index: 4)
                enc.setBytes(&tpd, length: 8, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return try sameTyped(out, length: n)
    }

    // MARK: - Arithmetic

    /// Adds a `duration` column, element-wise. The duration is rescaled to this array's resolution
    /// first, so `timestamp[us] + duration[s]` works. The result keeps this array's type (and
    /// timezone); validity is the AND of both sides, Arrow's `EMIT_NULL`.
    ///
    /// `date32` is rejected: its tick is a whole day, so adding a sub-day duration has no meaning
    /// here. Round or cast to a timestamp first.
    public func addDuration(_ d: MetalTemporalArray) throws -> MetalTemporalArray {
        guard case .duration = d.type else {
            throw ArrowMetalError.unsupportedType("addDuration needs a duration column, got \(d.type.arrowFormat)")
        }
        guard length == d.length else { throw ArrowMetalError.lengthMismatch(length, d.length) }
        if case .date32 = type {
            throw ArrowMetalError.unsupportedType("addDuration is not defined for date32; cast to a timestamp first")
        }
        let scaled = try d.castUnit(to: type.unit)
        let sum = try int64Values().arithmetic(.add, try scaled.int64Values())
        return try type.usesInt64 ? MetalTemporalArray(type: type, sum)
                                  : MetalTemporalArray(type: type, try sum.cast(to: Int32.self))
    }

    /// Adds a scalar duration expressed in **this array's own ticks** (microseconds for a
    /// `timestamp[us]`, days for a `date32`). Wraps like every other integer kernel here.
    public func addDuration(_ ticks: Int64) throws -> MetalTemporalArray {
        switch storage {
        case .int64(let a): return try MetalTemporalArray(type: type, try a.arithmetic(.add, ticks))
        case .int32(let a):
            guard let t = Int32(exactly: ticks) else {
                throw ArrowMetalError.invalidArrowArray("\(ticks) does not fit \(type.arrowFormat)")
            }
            return try MetalTemporalArray(type: type, try a.arithmetic(.add, t))
        }
    }

    /// `self - other` as a `duration`. Both sides must be the same family: two timestamps, two
    /// durations, two `date32`s (the result is a duration in seconds) or two `date64`s (milliseconds).
    /// Timestamps and durations of different resolutions are both rescaled to the finer of the two.
    public func subtractTemporal(_ other: MetalTemporalArray) throws -> MetalTemporalArray {
        guard length == other.length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        func difference(_ unit: ArrowTemporalUnit, scale: Int64) throws -> MetalTemporalArray {
            var a = try int64Values(), b = try other.int64Values()
            if scale != 1 {
                a = try a.arithmetic(.mul, scale)
                b = try b.arithmetic(.mul, scale)
            }
            return try MetalTemporalArray(type: .duration(unit), try a.arithmetic(.sub, b))
        }
        switch (type, other.type) {
        case (.timestamp(let ua, _), .timestamp(let ub, _)), (.duration(let ua), .duration(let ub)):
            let unit = ua.perSecond >= ub.perSecond ? ua : ub
            let a = try castUnit(to: unit), b = try other.castUnit(to: unit)
            return try MetalTemporalArray(type: .duration(unit),
                                          try a.int64Values().arithmetic(.sub, try b.int64Values()))
        case (.date32, .date32): return try difference(.second, scale: 86_400)
        case (.date64, .date64): return try difference(.milli, scale: 1)
        default:
            throw ArrowMetalError.unsupportedType(
                "cannot subtract \(other.type.arrowFormat) from \(type.arrowFormat)")
        }
    }

    /// Arrow `days_between(self, other)`: the number of whole UTC days from `self` to `other`, that
    /// is the difference of the two calendar dates. Positive when `other` is later. Both sides are
    /// floored to their day first, so `days_between(23:59, 00:01 next day)` is 1.
    public func daysBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        guard length == other.length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        let a = try toDate32(), b = try other.toDate32()
        guard let ai = a.asInt32, let bi = b.asInt32 else {
            throw ArrowMetalError.unsupportedType("daysBetween needs date-carrying columns")
        }
        return try bi.cast(to: Int64.self).arithmetic(.sub, try ai.cast(to: Int64.self))
    }

    // MARK: - Calendar fields (UTC, GPU)

    /// Day of the year, 1 for 1 January.
    public func dayOfYear() throws -> MetalArray<Int32> { try calendarField(0, needsDate: true) }
    /// Calendar quarter, 1 through 4.
    public func quarter() throws -> MetalArray<Int32> { try calendarField(1, needsDate: true) }
    /// ISO 8601 week number, 1 through 53 (the week owning this date's Thursday).
    public func isoWeek() throws -> MetalArray<Int32> { try calendarField(2, needsDate: true) }
    /// ISO 8601 week-numbering year, which can differ from the calendar year in late December and
    /// early January.
    public func isoYear() throws -> MetalArray<Int32> { try calendarField(3, needsDate: true) }
    /// Milliseconds since the last full second (Arrow's `millisecond`).
    public func millisecond() throws -> MetalArray<Int32> { try calendarField(4, needsDate: false) }
    /// Microseconds since the last full millisecond (Arrow's `microsecond`).
    public func microsecond() throws -> MetalArray<Int32> { try calendarField(5, needsDate: false) }
    /// Nanoseconds since the last full microsecond (Arrow's `nanosecond`).
    public func nanosecond() throws -> MetalArray<Int32> { try calendarField(6, needsDate: false) }

    private func calendarField(_ field: Int, needsDate: Bool) throws -> MetalArray<Int32> {
        guard let (mode, divisor) = type.extraction else {
            throw ArrowMetalError.unsupportedType("calendar fields are not defined for \(type.arrowFormat)")
        }
        if needsDate && mode == 2 {
            throw ArrowMetalError.unsupportedType("date fields are not defined for \(type.arrowFormat)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try mathPipeline("temporal_fields")
            let vals = values
            var div = divisor, nsPerTick = mode == 0 ? Int64(0) : 1_000_000_000 / divisor
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, field, index: 2)
                Dispatch.setUInt(enc, mode, index: 3)
                enc.setBytes(&div, length: 8, index: 4)
                enc.setBytes(&nsPerTick, length: 8, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Arrow `is_leap_year`: true when the value's UTC year has 366 days, in the proleptic Gregorian
    /// calendar with astronomical year numbering (year 0 is a leap year).
    public func isLeapYear() throws -> MetalBooleanArray {
        guard let (mode, divisor) = type.extraction, mode != 2 else {
            throw ArrowMetalError.unsupportedType("is_leap_year is not defined for \(type.arrowFormat)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: ctx)
        if n > 0 {
            let pso = try mathPipeline("temporal_leap")
            let vals = values
            var div = divisor
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, mode, index: 2)
                enc.setBytes(&div, length: 8, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: BitmapOps.words(bits: n))
            }
        }
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - Formatting and parsing (CPU)

    /// Arrow `strftime`, UTC. `format` is a **C `strftime` format string** (`%Y-%m-%d %H:%M:%S`),
    /// evaluated by the C library against a `struct tm` built with `gmtime_r` — not a `DateFormatter`
    /// Unicode pattern. One extension beyond C: `%f` expands to the six-digit fractional second, so
    /// `"%Y-%m-%dT%H:%M:%S.%f"` prints microseconds. Null in, null out; a value whose seconds do not
    /// fit a `time_t` calendar comes back null.
    public func strftime(_ format: String) throws -> MetalStringArray {
        guard let (mode, divisor) = type.extraction else {
            throw ArrowMetalError.unsupportedType("strftime is not defined for \(type.arrowFormat)")
        }
        let n = length
        var rows = [String?](repeating: nil, count: n)
        for i in 0..<n {
            guard let v = self[i] else { continue }
            var seconds: Int64
            var subTicks: Int64 = 0
            if mode == 0 {
                seconds = v * 86_400
            } else {
                let s = Self.floorDiv(v, divisor)
                subTicks = v - s * divisor
                seconds = s
            }
            let micros = divisor <= 1_000_000 ? subTicks * (1_000_000 / divisor) : subTicks / (divisor / 1_000_000)
            rows[i] = Self.formatUTC(seconds: seconds, microseconds: micros, format: format)
        }
        return try MetalStringArray(rows, context: context)
    }

    static func floorDiv(_ a: Int64, _ b: Int64) -> Int64 {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }

    /// `strftime` against a UTC `tm`, with `%f` pre-expanded to six fractional digits.
    static func formatUTC(seconds: Int64, microseconds: Int64, format: String) -> String? {
        var t = time_t(clamping: seconds)
        guard Int64(t) == seconds else { return nil }
        var tmv = tm()
        guard gmtime_r(&t, &tmv) != nil else { return nil }
        var expanded = ""
        var it = format.makeIterator()
        var pending = it.next()
        while let ch = pending {
            pending = it.next()
            if ch == "%", let next = pending {
                pending = it.next()
                if next == "f" {
                    expanded += String(format: "%06d", Int(microseconds))
                } else {
                    expanded.append(ch); expanded.append(next)
                }
            } else {
                expanded.append(ch)
            }
        }
        var size = 256
        while size <= 65536 {
            var buf = [CChar](repeating: 0, count: size)
            let written = expanded.withCString { Darwin.strftime(&buf, size, $0, &tmv) }
            if written > 0 || expanded.isEmpty { return String(cString: buf) }
            size *= 4
        }
        return nil
    }
}

extension MetalStringArray {
    /// Arrow `strptime`, UTC. `format` is a **C `strptime` format string**; the whole value must be
    /// consumed, so trailing text is a failure. Fields the format does not mention default to
    /// 1970-01-01 00:00:00. A row that does not parse comes back null, or throws when `strict` is set.
    /// Always CPU (`strptime` + `timegm`), sharded over `DispatchQueue.concurrentPerform`.
    public func strptime(_ format: String, unit: ArrowTemporalUnit = .second,
                         timezone: String? = nil, strict: Bool = false) throws -> MetalTemporalArray {
        let n = length, ctx = context
        let outVals = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                     zeroed: true, context: ctx)
        let vals = outVals.mutableTyped(Int64.self), bits = outValid.mutableTyped(UInt8.self)
        let scale = unit.perSecond
        forEachRowConcurrently { i, s in
            guard let s else { return }
            var tmv = tm()
            tmv.tm_mday = 1
            tmv.tm_year = 70
            let parsed: Bool = s.withCString { cs in
                format.withCString { cf in
                    guard let rest = Darwin.strptime(cs, cf, &tmv) else { return false }
                    return rest.pointee == 0
                }
            }
            guard parsed else { return }
            let seconds = Int64(timegm(&tmv))
            let (ticks, overflow) = seconds.multipliedReportingOverflow(by: scale)
            guard !overflow else { return }
            vals[i] = ticks
            Bitmap.set(bits, i)
        }
        let out = MetalArray<Int64>(length: n, nullCount: 0, validity: outValid, values: outVals, context: ctx)
        out.recomputeNullCount()
        if strict, out.nullCount > nullCount {
            throw ArrowMetalError.invalidArrowArray(
                "\(out.nullCount - nullCount) of \(n) values do not match the format \"\(format)\"")
        }
        return try MetalTemporalArray(type: .timestamp(unit, timezone: timezone), out)
    }
}
