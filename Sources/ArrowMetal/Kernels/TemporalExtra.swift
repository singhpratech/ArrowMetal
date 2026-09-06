import Foundation
import Metal

/// The Arrow temporal functions `Temporal.swift` (year … second) and `Kernels/TemporalMath.swift`
/// (rounding, `day_of_year`, `quarter`, the ISO fields, the subsecond components, `days_between`)
/// leave over: the option-carrying week numbers, the two struct-valued extractors, `subsecond`,
/// `is_dst` and every `*_between` difference.
///
/// Everything here is **GPU** except `is_dst`, which needs the timezone database and so runs on the
/// host, sharded over `DispatchQueue.concurrentPerform`. UTC throughout, exactly as the rest of the
/// package: a timestamp's timezone rides along as metadata and is never applied to the value — the
/// one exception being `is_dst`, whose whole job is to ask what that timezone was doing.
///
/// Results are int64 wherever pyarrow's are, so `week`, `us_week`, `us_year`, `day_of_week` with
/// options, the `iso_calendar` / `year_month_day` children and every `*_between` come back as
/// `MetalArray<Int64>`; `subsecond` is float64 and `is_dst` boolean.
extension MetalTemporalArray {

    // MARK: - Plumbing

    /// MSL element type of the storage.
    private var extraMSLType: String { type.usesInt64 ? "long" : "int" }

    private func extraPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: TemporalExtraSource.source(T: extraMSLType), function: fn,
                             cacheKey: "temporalextra/\(extraMSLType)/\(fn)")
    }

    /// Ticks of this array's storage in one UTC day, or nil for a column that carries no date
    /// (`time32`, `time64`, `duration`). This is the divisor that turns a stored value into days
    /// since the epoch, and it is 1 for `date32`, whose tick already *is* a day.
    private var dateTicksPerDay: Int64? {
        switch type {
        case .date32: return 1
        case .date64: return 86_400_000
        case .timestamp(let u, _): return 86_400 * u.perSecond
        case .time32, .time64, .duration: return nil
        }
    }

    private func requireDate(_ what: String) throws -> Int64 {
        guard let t = dateTicksPerDay else {
            throw ArrowMetalError.unsupportedType("\(what) needs a column that carries a date, not \(type.arrowFormat)")
        }
        return t
    }

    /// One int64 field per row. Validity (and so the null count) is shared with the input.
    private func extraField(_ field: Int, ticksPerDay: Int64, p1: Int = 0, p2: Int64 = 0) throws -> MetalArray<Int64> {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try extraPipeline("temporal_extra_field")
            let vals = values
            var tpd = ticksPerDay, q2 = p2
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, field, index: 2)
                enc.setBytes(&tpd, length: 8, index: 3)
                Dispatch.setUInt(enc, p1, index: 4)
                enc.setBytes(&q2, length: 8, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<Int64>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// Three int64 children in one pass, wrapped in a struct that carries this array's own validity.
    /// A null row makes the *struct* null, and the children stay valid holding whatever the calendar
    /// made of the stored value — exactly the shape pyarrow's `iso_calendar` produces.
    private func extraStruct(_ field: Int, what: String, names: [String]) throws -> MetalStructArray {
        let ticksPerDay = try requireDate(what)
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let bufs = try (0..<3).map { _ in
            try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        }
        if n > 0 {
            let pso = try extraPipeline("temporal_extra_struct")
            let vals = values
            var tpd = ticksPerDay
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, field, index: 2)
                enc.setBytes(&tpd, length: 8, index: 3)
                for (k, b) in bufs.enumerated() { enc.setBuffer(b.mtl, offset: b.offset, index: 4 + k) }
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let children: [AnyMetalArray] = bufs.map {
            .int64(MetalArray<Int64>(length: n, nullCount: 0, validity: nil, values: $0, context: ctx))
        }
        return try MetalStructArray(length: n, nullCount: nullCount, validity: validity, names: names,
                                    children: children, context: ctx)
    }

    // MARK: - Component extraction

    /// Arrow `week` with the full `WeekOptions`.
    ///
    /// * `weekStartsMonday` — weeks start on Monday, otherwise on Sunday.
    /// * `countFromZero` — number the weeks against the value's own calendar year, so a date at the
    ///   start of a year that belongs to the previous year's last week comes out as 0 rather than
    ///   52 or 53.
    /// * `firstWeekIsFullyInYear` — week 1 is the first week lying wholly inside January. When it is
    ///   false the ISO majority rule applies instead: the week belongs to the year owning its fourth
    ///   day, so a week beginning on 29, 30 or 31 December is week 1 of the following year.
    ///
    /// The defaults are Arrow's and reproduce `iso_week`.
    public func week(weekStartsMonday: Bool = true, countFromZero: Bool = false,
                     firstWeekIsFullyInYear: Bool = false) throws -> MetalArray<Int64> {
        let bits = (weekStartsMonday ? 1 : 0) | (countFromZero ? 2 : 0) | (firstWeekIsFullyInYear ? 4 : 0)
        return try extraField(0, ticksPerDay: try requireDate("week"), p1: bits)
    }

    /// Arrow `us_week`: `week` with Sunday-start weeks and the majority rule, 1 through 53.
    public func usWeek() throws -> MetalArray<Int64> {
        try week(weekStartsMonday: false, countFromZero: false, firstWeekIsFullyInYear: false)
    }

    /// Arrow `us_year`: the US epidemiological week-numbering year, that is the year owning the
    /// Wednesday of this date's Sunday-start week.
    public func usYear() throws -> MetalArray<Int64> {
        try extraField(1, ticksPerDay: try requireDate("us_year"))
    }

    /// Arrow `day_of_week` with `DayOfWeekOptions`. `weekStart` uses the ISO numbering (1 = Monday …
    /// 7 = Sunday) and is unaffected by `countFromZero`, which only decides whether the answer starts
    /// at 0 or at 1. `dayOfWeek()` (in `Temporal.swift`) is the int32 default, Monday = 0.
    public func dayOfWeek(countFromZero: Bool, weekStart: Int) throws -> MetalArray<Int64> {
        guard weekStart >= 1, weekStart <= 7 else {
            throw ArrowMetalError.invalidArrowArray("week_start must be between 1 (Monday) and 7 (Sunday), got \(weekStart)")
        }
        return try extraField(2, ticksPerDay: try requireDate("day_of_week"),
                              p1: countFromZero ? 1 : 0, p2: Int64(weekStart))
    }

    /// Arrow `iso_calendar`: a struct of `iso_year`, `iso_week` and `iso_day_of_week` (1 = Monday).
    public func isoCalendar() throws -> MetalStructArray {
        try extraStruct(1, what: "iso_calendar", names: ["iso_year", "iso_week", "iso_day_of_week"])
    }

    /// Arrow `year_month_day`: a struct of `year`, `month` and `day`.
    public func yearMonthDay() throws -> MetalStructArray {
        try extraStruct(0, what: "year_month_day", names: ["year", "month", "day"])
    }

    /// Arrow `subsecond`: the fraction of a second, in [0, 1), as float64.
    ///
    /// Floor semantics like every other component here, so a value half a second before the epoch
    /// has subsecond 0.5. `date32` and `date64` are accepted and answer 0 (pyarrow has no kernel for
    /// them); `duration`, which carries no clock, is rejected.
    public func subsecond() throws -> MetalArray<Double> {
        guard let (_, divisor) = type.extraction else {
            throw ArrowMetalError.unsupportedType("subsecond is not defined for \(type.arrowFormat)")
        }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TemporalExtraSource.subsecondSource(T: extraMSLType),
                                       function: "temporal_subsecond",
                                       cacheKey: "temporalextra/\(extraMSLType)/temporal_subsecond")
            let vals = values
            var div = divisor
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                enc.setBytes(&div, length: 8, index: 2)
                enc.setBuffer(out.mtl, offset: out.offset, index: 3)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<Double>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - is_dst (GPU)

    /// Arrow `is_dst`: whether each value falls in daylight saving time in the column's own timezone.
    ///
    /// Only a `timestamp` carrying a timezone can answer; a naive timestamp throws, as pyarrow does.
    /// **GPU**: the zone's DST flags ride along with the transition table `Kernels/TimezoneGPU.swift`
    /// uploads once per zone, so this is one pass and a binary search per row. The host loop below is
    /// the fallback for a zone Foundation will not enumerate and for values outside the tabulated
    /// window. A fixed offset ("+02:00") never observes DST and answers false everywhere.
    public func isDST() throws -> MetalBooleanArray {
        guard case .timestamp(let unit, let tz) = type else {
            throw ArrowMetalError.unsupportedType("is_dst is only defined for timestamps, not \(type.arrowFormat)")
        }
        guard let name = tz, !name.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("Timestamps have no timezone. Cannot determine DST.")
        }
        guard let zone = Self.lookupTimeZone(name) else {
            throw ArrowMetalError.invalidArrowArray("unknown timezone \"\(name)\"")
        }
        if let gpu = try isDSTGPU(unit: unit, zone: name) { return gpu }
        let n = length, ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1),
                                                zeroed: true, context: ctx)
        if n > 0 {
            let bits = out.mutableTyped(UInt8.self)
            let vals = int64Storage()
            let valid = validity?.typed(UInt8.self)
            let scale = unit.perSecond
            // A multiple of 8 so that two workers never touch the same bitmap byte.
            let chunk = 4_096
            let chunks = (n + chunk - 1) / chunk
            func work(_ c: Int) {
                let lo = c * chunk, hi = Swift.min(lo + chunk, n)
                for i in lo..<hi {
                    if let valid, !Bitmap.isSet(valid, i) { continue }
                    let seconds = Self.floorDiv(vals(i), scale)
                    if zone.isDaylightSavingTime(for: Date(timeIntervalSince1970: Double(seconds))) {
                        Bitmap.set(bits, i)
                    }
                }
            }
            if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
        }
        return MetalBooleanArray(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    /// A closure reading row `i` as an Int64, whatever the storage width is. `valuePointer`
    /// materialises a pending batched result first, so the host sees finished values.
    private func int64Storage() -> (Int) -> Int64 {
        switch storage {
        case .int64(let a): let p = a.valuePointer; return { p[$0] }
        case .int32(let a): let p = a.valuePointer; return { Int64(p[$0]) }
        }
    }

    /// An IANA name, a POSIX abbreviation, or a fixed "+HH:MM" / "-HHMM" offset.
    static func lookupTimeZone(_ s: String) -> TimeZone? {
        if let z = TimeZone(identifier: s) { return z }
        if let z = TimeZone(abbreviation: s) { return z }
        var text = Substring(s)
        guard let sign = text.first, sign == "+" || sign == "-" else { return nil }
        text = text.dropFirst()
        let digits = text.filter { $0.isNumber }
        guard digits.count == 4, text.allSatisfy({ $0.isNumber || $0 == ":" }) else { return nil }
        let hours = Int(digits.prefix(2))!, minutes = Int(digits.suffix(2))!
        guard minutes < 60 else { return nil }
        let seconds = (hours * 3_600 + minutes * 60) * (sign == "-" ? -1 : 1)
        return TimeZone(secondsFromGMT: seconds)
    }

    // MARK: - Differences

    /// The nanoseconds one tick of `unit` covers, for the fixed-length `*_between` units.
    private enum BetweenUnit: Int64 {
        case hour = 3_600_000_000_000, minute = 60_000_000_000, second = 1_000_000_000
        case millisecond = 1_000_000, microsecond = 1_000, nanosecond = 1
    }

    /// `floor(v * num / den)` maps a stored value onto the ruler a difference is counted on.
    private static func ruler(tickNanoseconds: Int64, targetNanoseconds: Int64) -> (num: Int64, den: Int64) {
        tickNanoseconds >= targetNanoseconds ? (tickNanoseconds / targetNanoseconds, 1)
                                             : (1, targetNanoseconds / tickNanoseconds)
    }

    private func runBetween(_ other: MetalTemporalArray, kind: Int,
                            a: (num: Int64, den: Int64), b: (num: Int64, den: Int64),
                            param: Int64) throws -> MetalArray<Int64> {
        guard length == other.length else { throw ArrowMetalError.lengthMismatch(length, other.length) }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let av = try int64Values(), bv = try other.int64Values()
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TemporalExtraSource.source(T: "long"),
                                       function: "temporal_between",
                                       cacheKey: "temporalextra/long/temporal_between")
            var aNum = a.num, aDen = a.den, bNum = b.num, bDen = b.den, p = param
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(av.values.mtl, offset: av.values.offset, index: 0)
                enc.setBuffer(bv.values.mtl, offset: bv.values.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setUInt(enc, kind, index: 3)
                enc.setBytes(&aNum, length: 8, index: 4)
                enc.setBytes(&aDen, length: 8, index: 5)
                enc.setBytes(&bNum, length: 8, index: 6)
                enc.setBytes(&bDen, length: 8, index: 7)
                enc.setBytes(&p, length: 8, index: 8)
                enc.setBuffer(out.mtl, offset: out.offset, index: 9)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let v = try BitmapOps.combineValidity(ctx, av.validity, bv.validity, bits: n)
        let res = MetalArray<Int64>(length: n, nullCount: 0, validity: v, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// A calendar-based difference: both sides are reduced to days since the epoch first.
    private func calendarBetween(_ other: MetalTemporalArray, kind: Int, what: String,
                                 param: Int64 = 0) throws -> MetalArray<Int64> {
        let a = try requireDate(what), b = try other.requireDate(what)
        return try runBetween(other, kind: kind, a: (1, a), b: (1, b), param: param)
    }

    /// A fixed-length difference: both sides are floored to `unit` first.
    private func unitBetween(_ other: MetalTemporalArray, _ unit: BetweenUnit,
                             what: String) throws -> MetalArray<Int64> {
        for side in [self, other] where side.isDurationColumn {
            throw ArrowMetalError.unsupportedType("\(what) is not defined for \(side.type.arrowFormat)")
        }
        let a = Self.ruler(tickNanoseconds: nanosecondsPerTick, targetNanoseconds: unit.rawValue)
        let b = Self.ruler(tickNanoseconds: other.nanosecondsPerTick, targetNanoseconds: unit.rawValue)
        return try runBetween(other, kind: 4, a: a, b: b, param: 0)
    }

    private var isDurationColumn: Bool { if case .duration = type { return true } else { return false } }

    /// Arrow `years_between(self, other)`: calendar years crossed, that is the difference of the two
    /// calendar years. Positive when `other` is later.
    public func yearsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try calendarBetween(other, kind: 0, what: "years_between")
    }

    /// Arrow `quarters_between`: the difference of `year * 4 + quarter`.
    public func quartersBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try calendarBetween(other, kind: 1, what: "quarters_between")
    }

    /// The int64 month difference, that is the difference of `year * 12 + month`. Arrow spells the
    /// same quantity `month_interval_between` and returns it as a `month` interval; this is the plain
    /// count.
    public func monthsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try calendarBetween(other, kind: 2, what: "months_between")
    }

    /// Arrow `weeks_between`: week boundaries crossed, both sides floored to the start of their week
    /// first. `weekStart` uses the ISO numbering (1 = Monday … 7 = Sunday). `countFromZero` is part
    /// of Arrow's `DayOfWeekOptions` but does not affect this function, and is accepted (and ignored)
    /// so the signature matches.
    public func weeksBetween(_ other: MetalTemporalArray, countFromZero: Bool = true,
                             weekStart: Int = 1) throws -> MetalArray<Int64> {
        guard weekStart >= 1, weekStart <= 7 else {
            throw ArrowMetalError.invalidArrowArray("week_start must be between 1 (Monday) and 7 (Sunday), got \(weekStart)")
        }
        return try calendarBetween(other, kind: 3, what: "weeks_between", param: Int64(weekStart))
    }

    /// Arrow `hours_between`: hour boundaries crossed.
    public func hoursBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .hour, what: "hours_between")
    }
    /// Arrow `minutes_between`: minute boundaries crossed.
    public func minutesBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .minute, what: "minutes_between")
    }
    /// Arrow `seconds_between`: second boundaries crossed.
    public func secondsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .second, what: "seconds_between")
    }
    /// Arrow `milliseconds_between`: millisecond boundaries crossed.
    public func millisecondsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .millisecond, what: "milliseconds_between")
    }
    /// Arrow `microseconds_between`: microsecond boundaries crossed.
    public func microsecondsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .microsecond, what: "microseconds_between")
    }
    /// Arrow `nanoseconds_between`: nanosecond boundaries crossed. The count wraps in int64 for spans
    /// beyond about 292 years, which is what Arrow does too.
    public func nanosecondsBetween(_ other: MetalTemporalArray) throws -> MetalArray<Int64> {
        try unitBetween(other, .nanosecond, what: "nanoseconds_between")
    }
}
