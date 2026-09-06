import Foundation
import Metal
import CArrowABI

// The three Arrow difference functions that *return* an interval:
//
//   month_interval_between          -> interval[month]           ("tiM")
//   day_time_interval_between       -> interval[day_time]        ("tiD")
//   month_day_nano_interval_between -> interval[month_day_nano]  ("tin")
//
// Arrow defines all three field by field, on the civil calendar in UTC, and so does this file:
//
//   * months  = month boundaries crossed, `(y2 - y1) * 12 + (m2 - m1)` — the difference of the two
//               timestamps truncated to the month, *not* a "whole months elapsed" count. So
//               2020-01-31 -> 2020-02-01 is one month, and 2020-01-01 -> 2020-01-31 is zero.
//   * days    = for month_day_nano, the difference of the day-of-month fields (`d2 - d1`), which may be
//               negative while the month count is positive; for day_time, the difference of the two
//               timestamps truncated to the day.
//   * sub-day = the difference of the two times of day, in nanoseconds (month_day_nano) or truncated to
//               milliseconds (day_time). It too may have the opposite sign to the day count.
//
// Everything runs on the GPU in one kernel (`Kernels/TypesExtraSource.swift`), with the two columns first
// brought to a common resolution by the existing rescale. The result is null wherever either side is.
// The int64 `*_between` family (`days_between`, `hours_between`, ...) is not here.

/// Which interval type a `*_interval_between` call produces.
public enum ArrowIntervalBetween: Int, Sendable, CaseIterable {
    case month = 0, dayTime = 1, monthDayNano = 2

    public var unit: ArrowIntervalUnit {
        switch self {
        case .month: return .months
        case .dayTime: return .dayTime
        case .monthDayNano: return .monthDayNano
        }
    }
}

extension MetalTemporalArray {
    /// Arrow `month_interval_between`: the number of month boundaries crossed from `self` to `other`,
    /// as an `interval[month]` column. GPU.
    public func monthIntervalBetween(_ other: MetalTemporalArray) throws -> MetalIntervalArray {
        try intervalBetween(other, kind: .month)
    }
    /// Arrow `day_time_interval_between`: whole days plus the millisecond difference of the two times of
    /// day, as an `interval[day_time]` column. GPU.
    public func dayTimeIntervalBetween(_ other: MetalTemporalArray) throws -> MetalIntervalArray {
        try intervalBetween(other, kind: .dayTime)
    }
    /// Arrow `month_day_nano_interval_between`: month boundaries crossed, the day-of-month difference and
    /// the nanosecond difference of the two times of day, as an `interval[month_day_nano]` column. GPU.
    public func monthDayNanoIntervalBetween(_ other: MetalTemporalArray) throws -> MetalIntervalArray {
        try intervalBetween(other, kind: .monthDayNano)
    }

    /// One kernel behind all three. `self` is Arrow's `start`, `other` its `end`.
    public func intervalBetween(_ other: MetalTemporalArray, kind: ArrowIntervalBetween) throws -> MetalIntervalArray {
        let n = length
        guard other.length == n else { throw ArrowMetalError.lengthMismatch(n, other.length) }
        let ctx = context
        var (a, ua) = try Self.canonicalTicks(self)
        var (b, ub) = try Self.canonicalTicks(other)
        // Bring both to the finer of the two resolutions; widening is an exact multiply.
        if ua.perSecond < ub.perSecond {
            a = try a.arithmetic(.mul, ub.perSecond / ua.perSecond); ua = ub
        } else if ub.perSecond < ua.perSecond {
            b = try b.arithmetic(.mul, ua.perSecond / ub.perSecond); ub = ua
        }
        let ticksPerDay = 86_400 * ua.perSecond
        let nanoPerTick = 1_000_000_000 / ua.perSecond

        let unit = kind.unit
        try Dispatch.checkLength(n)
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * unit.byteWidth, 1), zeroed: true, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TypesExtraSource.interval, function: "temporal_interval_between",
                                       cacheKey: "interval/between")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
                enc.setBuffer(b.values.mtl, offset: b.values.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                var tpd = ticksPerDay; enc.setBytes(&tpd, length: 8, index: 3)
                var npt = nanoPerTick; enc.setBytes(&npt, length: 8, index: 4)
                Dispatch.setUInt(enc, kind.rawValue, index: 5)
                enc.setBuffer(out.mtl, offset: out.offset, index: 6)
                enc.setBuffer(out.mtl, offset: out.offset, index: 7)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let v = try BitmapOps.combineValidity(ctx, validity, other.validity, bits: Swift.max(n, 1))
        let res = MetalIntervalArray(unit: unit, length: n, nullCount: 0, validity: v, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }

    /// A temporal column as int64 ticks plus the unit those ticks are in: `date32`'s whole days become
    /// seconds, `date64` is already milliseconds, and a timestamp keeps its own unit. Time-of-day and
    /// duration columns carry no date, so they are rejected.
    static func canonicalTicks(_ t: MetalTemporalArray) throws -> (MetalArray<Int64>, ArrowTemporalUnit) {
        switch t.type {
        case .date32: return (try t.int64Values().arithmetic(.mul, 86_400), .second)
        case .date64: return (try t.int64Values(), .milli)
        case .timestamp(let u, _): return (try t.int64Values(), u)
        case .time32, .time64, .duration:
            throw ArrowMetalError.unsupportedType("*_interval_between needs a date or timestamp column, got \(t.type.arrowFormat)")
        }
    }
}

extension AnyMetalArray {
    /// Arrow `month_interval_between` / `day_time_interval_between` / `month_day_nano_interval_between`
    /// over two date or timestamp columns.
    public func intervalBetween(_ other: AnyMetalArray, kind: ArrowIntervalBetween) throws -> AnyMetalArray {
        guard case .temporal(let a) = storageArray else {
            throw ArrowMetalError.unsupportedType("*_interval_between needs a date or timestamp column, got \(arrowFormat)")
        }
        guard case .temporal(let b) = other.storageArray else {
            throw ArrowMetalError.unsupportedType("*_interval_between needs a date or timestamp column, got \(other.arrowFormat)")
        }
        return .interval(try a.intervalBetween(b, kind: kind))
    }
}
