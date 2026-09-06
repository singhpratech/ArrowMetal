import XCTest
import Foundation
import CArrowABI
@testable import ArrowMetal

/// The Arrow temporal functions added by `Kernels/TemporalExtra.swift`: the option-carrying week
/// numbers, `us_week` / `us_year`, `iso_calendar`, `year_month_day`, `day_of_week` with options,
/// `subsecond`, `is_dst` and every `*_between` difference.
///
/// Every expectation is an oracle built on Foundation's `Calendar` (Gregorian and ISO 8601) in UTC,
/// or on `TimeZone` for `is_dst` — never on a second copy of the kernel's own arithmetic. The
/// pyarrow.compute cross-check for the same functions lives in `python/tests/test_arrowmetal.py`.
final class TemporalExtraTests: XCTestCase {

    /// Deterministic so a failure can be reproduced.
    struct ExtraRNG: RandomNumberGenerator {
        var state: UInt64
        init(_ seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 1 }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    /// Empty, one row, a partial bitmap word, several threadgroups, and a size that is no multiple
    /// of anything.
    static let sizes = [0, 1, 33, 4097, 300_003]

    /// 1900-01-01 and 2200-01-01 as UTC seconds; every random test spans them.
    static let lowSecond: Int64 = -2_208_988_800
    static let highSecond: Int64 = 7_258_118_400

    // MARK: - Foundation oracles

    private func utcCalendar() -> Calendar {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }
    private func isoCalendarUTC() -> Calendar {
        var c = Calendar(identifier: .iso8601)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }

    private func date(days: Int64) -> Date { Date(timeIntervalSince1970: Double(days) * 86_400) }

    /// Gregorian year / month / day of a day number, straight out of Foundation.
    private func ymd(_ days: Int64, _ cal: Calendar) -> (y: Int, m: Int, d: Int) {
        let c = cal.dateComponents([.year, .month, .day], from: date(days: days))
        return (c.year!, c.month!, c.day!)
    }
    /// ISO weekday of a day number (1 = Monday … 7 = Sunday), from Foundation's 1 = Sunday numbering.
    private func isoWeekday(_ days: Int64, _ cal: Calendar) -> Int {
        (cal.component(.weekday, from: date(days: days)) + 5) % 7 + 1
    }
    /// Day number of 1 January of `year`, memoised because the tests ask for it once per row.
    private func january1(_ year: Int, _ cal: Calendar, _ cache: inout [Int: Int64]) -> Int64 {
        if let v = cache[year] { return v }
        var dc = DateComponents()
        dc.year = year; dc.month = 1; dc.day = 1
        let v = Int64((cal.date(from: dc)!.timeIntervalSince1970 / 86_400).rounded(.down))
        cache[year] = v
        return v
    }

    private func floorDiv(_ a: Int64, _ b: Int64) -> Int64 {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }

    /// Arrow `week` for one day number, with the calendar work done by Foundation.
    private func oracleWeek(_ days: Int64, weekStartsMonday: Bool, countFromZero: Bool,
                            firstWeekIsFullyInYear: Bool, _ cal: Calendar,
                            _ cache: inout [Int: Int64]) -> Int64 {
        let weekStart = weekStartsMonday ? 1 : 7
        let offset = (isoWeekday(days, cal) - weekStart + 7) % 7
        let weekStartDay = days - Int64(offset)
        let pivot = weekStartDay + (firstWeekIsFullyInYear ? 0 : 3)
        let year = ymd(countFromZero ? days : pivot, cal).y
        return floorDiv(pivot - january1(year, cal, &cache), 7) + 1
    }

    /// Arrow `us_year`: the year owning the Wednesday of this Sunday-start week.
    private func oracleUSYear(_ days: Int64, _ cal: Calendar) -> Int64 {
        let offset = (isoWeekday(days, cal) - 7 + 7) % 7
        return Int64(ymd(days - Int64(offset) + 3, cal).y)
    }

    /// The start of the week containing `days`, for a week beginning on `weekStart` (1 = Monday).
    private func weekFloor(_ days: Int64, weekStart: Int, _ cal: Calendar) -> Int64 {
        days - Int64((isoWeekday(days, cal) - weekStart + 7) % 7)
    }

    // MARK: - Sample data

    /// Random UTC seconds spanning 1900-2200 with the awkward instants pinned in front.
    private func randomSeconds(_ n: Int, seed: UInt64) -> [Int64] {
        guard n > 0 else { return [] }
        var g = ExtraRNG(seed)
        let span = UInt64(Self.highSecond - Self.lowSecond)
        var out = (0..<n).map { _ in Self.lowSecond + Int64(g.next() % span) }
        // The epoch, the day and the year around it, and the week-numbering boundaries.
        let pinned: [Int64] = [
            0, -1, 1, -86_400, 86_400, -86_401,
            Self.lowSecond, Self.highSecond - 1,
            -2_208_988_801,            // 1899-12-31T23:59:59Z
            951_782_400,               // 2000-02-29
            1_388_448_000,             // 2013-12-31
            1_388_534_400,             // 2014-01-01
            1_419_984_000,             // 2014-12-31
            1_420_070_400,             // 2015-01-01
            1_451_606_400,             // 2016-01-01
            1_451_779_200,             // 2016-01-03, a Sunday
            4_102_444_800,             // 2100-01-01
            7_258_118_399,             // 2199-12-31T23:59:59Z
        ]
        for (i, v) in pinned.enumerated() where i < n { out[i] = v }
        return out
    }

    /// Every seventh row null, so validity is exercised at every size.
    private func nullable(_ values: [Int64], nullEvery: Int = 7) -> [Int64?] {
        values.enumerated().map { $0.offset % nullEvery == 3 ? nil : $0.element }
    }

    private func timestamp(_ values: [Int64?], _ unit: ArrowTemporalUnit,
                           timezone: String? = nil) throws -> MetalTemporalArray {
        let scale = unit.perSecond
        return try MetalTemporalArray(type: .timestamp(unit, timezone: timezone),
                                      values.map { $0.map { $0 &* scale } })
    }

    // MARK: - week / us_week / us_year

    func testWeekMatchesFoundationForEveryOptionCombination() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        var cache: [Int: Int64] = [:]
        let n = 6_000
        let seconds = randomSeconds(n, seed: 1_001)
        // Plus every day of four full years around the awkward boundaries, so no combination can
        // hide behind a sparse sample.
        var days = seconds.map { floorDiv($0, 86_400) }
        for d in stride(from: Int64(16_000), to: Int64(17_500), by: 1) { days.append(d) }
        for d in stride(from: Int64(-25_600), to: Int64(-24_100), by: 1) { days.append(d) }

        let d32 = try MetalTemporalArray(type: .date32, days.map { Optional($0) })
        for weekStartsMonday in [true, false] {
            for countFromZero in [true, false] {
                for firstWeekIsFullyInYear in [true, false] {
                    let got = try d32.week(weekStartsMonday: weekStartsMonday,
                                           countFromZero: countFromZero,
                                           firstWeekIsFullyInYear: firstWeekIsFullyInYear).toRawArray()
                    for i in 0..<days.count {
                        let want = oracleWeek(days[i], weekStartsMonday: weekStartsMonday,
                                              countFromZero: countFromZero,
                                              firstWeekIsFullyInYear: firstWeekIsFullyInYear,
                                              cal, &cache)
                        if got[i] != want {
                            return XCTFail("week(\(weekStartsMonday), \(countFromZero), \(firstWeekIsFullyInYear)) "
                                           + "on day \(days[i]): \(got[i]) != \(want)")
                        }
                    }
                }
            }
        }
        // The default combination is Arrow's iso_week, which TemporalMath already computes.
        XCTAssertEqual(try d32.week().toRawArray(), try d32.isoWeek().toRawArray().map(Int64.init))
    }

    func testUSWeekAndUSYearMatchFoundation() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        var cache: [Int: Int64] = [:]
        let seconds = randomSeconds(20_000, seed: 2_002)
        let ts = try timestamp(seconds.map { Optional($0) }, .second)
        let usWeek = try ts.usWeek().toRawArray()
        let usYear = try ts.usYear().toRawArray()
        for i in 0..<seconds.count {
            let days = floorDiv(seconds[i], 86_400)
            let wantWeek = oracleWeek(days, weekStartsMonday: false, countFromZero: false,
                                      firstWeekIsFullyInYear: false, cal, &cache)
            if usWeek[i] != wantWeek { return XCTFail("us_week at \(seconds[i]): \(usWeek[i]) != \(wantWeek)") }
            let wantYear = oracleUSYear(days, cal)
            if usYear[i] != wantYear { return XCTFail("us_year at \(seconds[i]): \(usYear[i]) != \(wantYear)") }
        }
    }

    func testWeekAcrossUnitsAndTypes() throws {
        try requireRealGPU()
        // 2016-01-03 is a Sunday: iso_week 53 of 2015, us_week 1 of 2016.
        let seconds: [Int64] = [1_451_779_200, 1_451_606_400, 0, -86_400]
        for unit in ArrowTemporalUnit.allCases {
            let ts = try timestamp(seconds.map { Optional($0) }, unit)
            XCTAssertEqual(try ts.week().toRawArray(), [53, 53, 1, 1], unit.arrowName)
            // 1970-01-01 is a Thursday, so its Sunday-start week began on 1969-12-28 and belongs to
            // 1969 — week 53 of the US year 1969, not week 1 of 1970.
            XCTAssertEqual(try ts.usWeek().toRawArray(), [1, 52, 53, 53], unit.arrowName)
            XCTAssertEqual(try ts.usYear().toRawArray(), [2016, 2015, 1969, 1969], unit.arrowName)
        }
        let d32 = try MetalTemporalArray(type: .date32, seconds.map { Optional(floorDiv($0, 86_400)) })
        XCTAssertEqual(try d32.week().toRawArray(), [53, 53, 1, 1])
        let d64 = try MetalTemporalArray(type: .date64, seconds.map { Optional($0 * 1_000) })
        XCTAssertEqual(try d64.week().toRawArray(), [53, 53, 1, 1])
        XCTAssertEqual(try d64.usWeek().toRawArray(), [1, 52, 53, 53])
        // Columns with no date cannot answer.
        XCTAssertThrowsError(try MetalTemporalArray(type: .time32(.second), [0]).week())
        XCTAssertThrowsError(try MetalTemporalArray(type: .time64(.nano), [0]).usWeek())
        XCTAssertThrowsError(try MetalTemporalArray(type: .duration(.second), [0]).usYear())
    }

    // MARK: - iso_calendar / year_month_day

    func testIsoCalendarMatchesFoundation() throws {
        try requireRealGPU()
        let iso = isoCalendarUTC()
        let cal = utcCalendar()
        let seconds = randomSeconds(20_000, seed: 3_003)
        let ts = try timestamp(seconds.map { Optional($0) }, .milli, timezone: "UTC")
        let s = try ts.isoCalendar()
        XCTAssertEqual(s.names, ["iso_year", "iso_week", "iso_day_of_week"])
        XCTAssertEqual(s.arrowFormat, "+s")
        XCTAssertEqual(s.length, seconds.count)
        let year = s.children[0].asInt64!.toRawArray()
        let week = s.children[1].asInt64!.toRawArray()
        let dow = s.children[2].asInt64!.toRawArray()
        for i in 0..<seconds.count {
            let d = Date(timeIntervalSince1970: Double(seconds[i]))
            let c = iso.dateComponents([.yearForWeekOfYear, .weekOfYear], from: d)
            let days = floorDiv(seconds[i], 86_400)
            if year[i] != Int64(c.yearForWeekOfYear!) || week[i] != Int64(c.weekOfYear!)
                || dow[i] != Int64(isoWeekday(days, cal)) {
                return XCTFail("iso_calendar at \(seconds[i]): \(year[i])-W\(week[i])-\(dow[i]) vs \(c)")
            }
        }
        // The children agree with the standalone iso_week / iso_year kernels.
        XCTAssertEqual(week, try ts.isoWeek().toRawArray().map(Int64.init))
        XCTAssertEqual(year, try ts.isoYear().toRawArray().map(Int64.init))
    }

    func testYearMonthDayMatchesFoundation() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        let seconds = randomSeconds(20_000, seed: 4_004)
        let ts = try timestamp(seconds.map { Optional($0) }, .micro)
        let s = try ts.yearMonthDay()
        XCTAssertEqual(s.names, ["year", "month", "day"])
        let year = s.children[0].asInt64!.toRawArray()
        let month = s.children[1].asInt64!.toRawArray()
        let day = s.children[2].asInt64!.toRawArray()
        for i in 0..<seconds.count {
            let want = ymd(floorDiv(seconds[i], 86_400), cal)
            if year[i] != Int64(want.y) || month[i] != Int64(want.m) || day[i] != Int64(want.d) {
                return XCTFail("year_month_day at \(seconds[i]): \(year[i])-\(month[i])-\(day[i]) vs \(want)")
            }
        }
    }

    func testStructResultsCarryNullsAndExport() throws {
        try requireRealGPU()
        let ts = try timestamp([0, nil, 86_400], .second)
        for s in [try ts.isoCalendar(), try ts.yearMonthDay()] {
            XCTAssertEqual(s.length, 3)
            XCTAssertEqual(s.nullCount, 1)
            XCTAssertTrue(s.isValid(0))
            XCTAssertFalse(s.isValid(1))
            // struct_field propagates the struct's own nulls into the child, as it does everywhere.
            let f = try s.structField(s.names[0])
            XCTAssertEqual(f.nullCount, 1)
            XCTAssertEqual(f.length, 3)
            // The struct exports through the C Data Interface with three int64 children.
            var schema = ArrowSchema(), array = ArrowArray()
            s.exportArrowSchema(name: "cal", into: &schema)
            s.exportArrowArray(into: &array)
            XCTAssertEqual(String(cString: schema.format), "+s")
            XCTAssertEqual(schema.n_children, 3)
            XCTAssertEqual(String(cString: schema.children[0]!.pointee.format), "l")
            XCTAssertEqual(String(cString: schema.children[0]!.pointee.name), s.names[0])
            XCTAssertEqual(array.length, 3)
            schema.release?(&schema)
        }
        XCTAssertThrowsError(try MetalTemporalArray(type: .time32(.second), [0]).isoCalendar())
        XCTAssertThrowsError(try MetalTemporalArray(type: .duration(.milli), [0]).yearMonthDay())
    }

    // MARK: - day_of_week options

    func testDayOfWeekOptionsMatchFoundation() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        let seconds = randomSeconds(5_000, seed: 5_005)
        let ts = try timestamp(seconds.map { Optional($0) }, .second)
        for countFromZero in [true, false] {
            for weekStart in 1...7 {
                let got = try ts.dayOfWeek(countFromZero: countFromZero, weekStart: weekStart).toRawArray()
                for i in 0..<seconds.count {
                    let iso = isoWeekday(floorDiv(seconds[i], 86_400), cal)
                    let want = Int64((iso - weekStart + 7) % 7 + (countFromZero ? 0 : 1))
                    if got[i] != want {
                        return XCTFail("day_of_week(\(countFromZero), \(weekStart)) at \(seconds[i]): \(got[i]) != \(want)")
                    }
                }
            }
        }
        // The default options reproduce the int32 dayOfWeek() Temporal.swift already has.
        XCTAssertEqual(try ts.dayOfWeek(countFromZero: true, weekStart: 1).toRawArray(),
                       try ts.dayOfWeek().toRawArray().map(Int64.init))
        XCTAssertThrowsError(try ts.dayOfWeek(countFromZero: true, weekStart: 0))
        XCTAssertThrowsError(try ts.dayOfWeek(countFromZero: true, weekStart: 8))
        XCTAssertThrowsError(try MetalTemporalArray(type: .time32(.milli), [0]).dayOfWeek(countFromZero: true, weekStart: 1))
    }

    // MARK: - subsecond

    func testSubsecondMatchesExactFraction() throws {
        try requireRealGPU()
        var g = ExtraRNG(6_006)
        let n = 50_000
        var ticks: [Int64] = []
        for _ in 0..<n {
            let seconds = Self.lowSecond + Int64(g.next() % UInt64(Self.highSecond - Self.lowSecond))
            ticks.append(seconds &* 1_000_000_000 &+ Int64(g.next() % 1_000_000_000))
        }
        ticks[0] = 0; ticks[1] = -1; ticks[2] = 999_999_999; ticks[3] = -1_500_000_123
        let ns = try MetalTemporalArray(type: .timestamp(.nano, timezone: nil), ticks.map { Optional($0) })
        let got = try ns.subsecond().toRawArray()
        for i in 0..<n {
            let sub = ticks[i] - floorDiv(ticks[i], 1_000_000_000) * 1_000_000_000
            let want = Double(sub) / Double(1_000_000_000)
            if got[i] != want { return XCTFail("subsecond at \(ticks[i]): \(got[i]) != \(want)") }
        }
        // Every unit, and the time-of-day types.
        XCTAssertEqual(try timestamp([1], .second).subsecond().toRawArray(), [0.0])
        XCTAssertEqual(try MetalTemporalArray(type: .timestamp(.milli, timezone: nil), [1_500, -1_500])
                        .subsecond().toRawArray(), [0.5, 0.5])
        XCTAssertEqual(try MetalTemporalArray(type: .timestamp(.micro, timezone: nil), [1_250_000])
                        .subsecond().toRawArray(), [0.25])
        XCTAssertEqual(try MetalTemporalArray(type: .time32(.milli), [86_399_999, 0]).subsecond().toRawArray(),
                       [0.999, 0.0])
        XCTAssertEqual(try MetalTemporalArray(type: .time64(.nano), [1_500_000_123]).subsecond().toRawArray(),
                       [Double(500_000_123) / 1e9])
        // date32 and date64 carry no fraction; duration carries no clock at all.
        XCTAssertEqual(try MetalTemporalArray(type: .date32, [19_000, -1]).subsecond().toRawArray(), [0.0, 0.0])
        XCTAssertEqual(try MetalTemporalArray(type: .date64, [0]).subsecond().toRawArray(), [0.0])
        XCTAssertThrowsError(try MetalTemporalArray(type: .duration(.nano), [1]).subsecond())
    }

    func testHourMinuteSecondOnTimeColumns() throws {
        try requireRealGPU()
        // Temporal.swift already accepts time32 / time64 for the clock fields; this pins it down.
        let t32s = try MetalTemporalArray(type: .time32(.second), [0, 3_661, 86_399])
        XCTAssertEqual(try t32s.hour().toRawArray(), [0, 1, 23])
        XCTAssertEqual(try t32s.minute().toRawArray(), [0, 1, 59])
        XCTAssertEqual(try t32s.second().toRawArray(), [0, 1, 59])
        let t32m = try MetalTemporalArray(type: .time32(.milli), [0, 3_661_500, 86_399_999])
        XCTAssertEqual(try t32m.hour().toRawArray(), [0, 1, 23])
        XCTAssertEqual(try t32m.second().toRawArray(), [0, 1, 59])
        let t64u = try MetalTemporalArray(type: .time64(.micro), [0, 3_661_000_001, 86_399_999_999])
        XCTAssertEqual(try t64u.hour().toRawArray(), [0, 1, 23])
        XCTAssertEqual(try t64u.minute().toRawArray(), [0, 1, 59])
        let t64n = try MetalTemporalArray(type: .time64(.nano), [3_661_000_000_001])
        XCTAssertEqual(try t64n.second().toRawArray(), [1])
    }

    // MARK: - is_dst

    func testIsDSTMatchesFoundationTimeZone() throws {
        try requireRealGPU()
        let zoneName = "America/New_York"
        let zone = TimeZone(identifier: zoneName)!
        let seconds = randomSeconds(50_000, seed: 7_007)
        let ts = try timestamp(seconds.map { Optional($0) }, .second, timezone: zoneName)
        let got = try ts.isDST().toArray()
        for i in 0..<seconds.count {
            let want = zone.isDaylightSavingTime(for: Date(timeIntervalSince1970: Double(seconds[i])))
            if got[i] != want { return XCTFail("is_dst at \(seconds[i]): \(String(describing: got[i])) != \(want)") }
        }
        // Known instants: 2023-01-01 is EST, 2023-06-29 is EDT.
        let pinned = try timestamp([1_672_531_200, 1_688_000_000, nil], .micro, timezone: zoneName)
        XCTAssertEqual(try pinned.isDST().toArray(), [false, true, nil])
        XCTAssertEqual(try pinned.isDST().nullCount, 1)
        // A fixed offset never observes DST; a naive timestamp cannot answer at all.
        XCTAssertEqual(try timestamp([0, 1_688_000_000], .second, timezone: "+02:00").isDST().toArray(),
                       [false, false])
        XCTAssertEqual(try timestamp([1_688_000_000], .second, timezone: "UTC").isDST().toArray(), [false])
        XCTAssertThrowsError(try timestamp([0], .second).isDST())
        XCTAssertThrowsError(try timestamp([0], .second, timezone: "Mars/Olympus").isDST())
        XCTAssertThrowsError(try MetalTemporalArray(type: .date32, [0]).isDST())
    }

    func testIsDSTAcrossUnitsAndSouthernHemisphere() throws {
        try requireRealGPU()
        let zoneName = "Australia/Sydney"
        let zone = TimeZone(identifier: zoneName)!
        let seconds: [Int64] = [1_672_531_200, 1_688_000_000, 0, -86_400, 4_102_444_800]
        for unit in ArrowTemporalUnit.allCases {
            let ts = try timestamp(seconds.map { Optional($0) }, unit, timezone: zoneName)
            let got = try ts.isDST().toArray()
            for i in 0..<seconds.count {
                XCTAssertEqual(got[i], zone.isDaylightSavingTime(for: Date(timeIntervalSince1970: Double(seconds[i]))),
                               "\(unit.arrowName) row \(i)")
            }
        }
    }

    // MARK: - Differences

    func testCalendarBetweenMatchesFoundation() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        let n = 30_000
        let a = randomSeconds(n, seed: 8_008)
        let b = randomSeconds(n, seed: 9_009)
        let ta = try timestamp(a.map { Optional($0) }, .second)
        let tb = try timestamp(b.map { Optional($0) }, .second)
        let years = try ta.yearsBetween(tb).toRawArray()
        let quarters = try ta.quartersBetween(tb).toRawArray()
        let months = try ta.monthsBetween(tb).toRawArray()
        for i in 0..<n {
            let x = ymd(floorDiv(a[i], 86_400), cal), y = ymd(floorDiv(b[i], 86_400), cal)
            let wantYears = Int64(y.y - x.y)
            let wantQuarters = Int64((y.y * 4 + (y.m - 1) / 3) - (x.y * 4 + (x.m - 1) / 3))
            let wantMonths = Int64((y.y * 12 + (y.m - 1)) - (x.y * 12 + (x.m - 1)))
            if years[i] != wantYears || quarters[i] != wantQuarters || months[i] != wantMonths {
                return XCTFail("row \(i) (\(a[i]) -> \(b[i])): years \(years[i])/\(wantYears), "
                               + "quarters \(quarters[i])/\(wantQuarters), months \(months[i])/\(wantMonths)")
            }
        }
    }

    func testWeeksBetweenForEveryOptionCombination() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        let n = 8_000
        let a = randomSeconds(n, seed: 10_010)
        let b = randomSeconds(n, seed: 11_011)
        let ta = try timestamp(a.map { Optional($0) }, .second)
        let tb = try timestamp(b.map { Optional($0) }, .second)
        for countFromZero in [true, false] {
            for weekStart in 1...7 {
                let got = try ta.weeksBetween(tb, countFromZero: countFromZero, weekStart: weekStart).toRawArray()
                for i in 0..<n {
                    let wa = weekFloor(floorDiv(a[i], 86_400), weekStart: weekStart, cal)
                    let wb = weekFloor(floorDiv(b[i], 86_400), weekStart: weekStart, cal)
                    let want = (wb - wa) / 7
                    if got[i] != want {
                        return XCTFail("weeks_between(\(countFromZero), \(weekStart)) row \(i): \(got[i]) != \(want)")
                    }
                }
            }
        }
        XCTAssertThrowsError(try ta.weeksBetween(tb, weekStart: 0))
        XCTAssertThrowsError(try ta.weeksBetween(tb, weekStart: 8))
    }

    func testFixedUnitBetweenTruncatesEachSide() throws {
        try requireRealGPU()
        let n = 30_000
        var g = ExtraRNG(12_012)
        // Nanosecond ticks so every unit has something to truncate.
        var a: [Int64] = [], b: [Int64] = []
        for _ in 0..<n {
            let sa = Self.lowSecond + Int64(g.next() % UInt64(Self.highSecond - Self.lowSecond))
            let sb = Self.lowSecond + Int64(g.next() % UInt64(Self.highSecond - Self.lowSecond))
            a.append(sa &* 1_000_000_000 &+ Int64(g.next() % 1_000_000_000))
            b.append(sb &* 1_000_000_000 &+ Int64(g.next() % 1_000_000_000))
        }
        a[0] = -1; b[0] = 0
        a[1] = 0; b[1] = -1
        a[2] = -3_600_000_000_001; b[2] = -1
        let ta = try MetalTemporalArray(type: .timestamp(.nano, timezone: nil), a.map { Optional($0) })
        let tb = try MetalTemporalArray(type: .timestamp(.nano, timezone: nil), b.map { Optional($0) })
        let units: [(String, Int64, [Int64])] = [
            ("hours", 3_600_000_000_000, try ta.hoursBetween(tb).toRawArray()),
            ("minutes", 60_000_000_000, try ta.minutesBetween(tb).toRawArray()),
            ("seconds", 1_000_000_000, try ta.secondsBetween(tb).toRawArray()),
            ("milliseconds", 1_000_000, try ta.millisecondsBetween(tb).toRawArray()),
            ("microseconds", 1_000, try ta.microsecondsBetween(tb).toRawArray()),
        ]
        for (name, ticks, got) in units {
            for i in 0..<n {
                let want = floorDiv(b[i], ticks) - floorDiv(a[i], ticks)
                if got[i] != want { return XCTFail("\(name)_between row \(i): \(got[i]) != \(want)") }
            }
        }
        // Nanoseconds are the raw difference, and wrap in int64 exactly as Arrow's does.
        let nanos = try ta.nanosecondsBetween(tb).toRawArray()
        for i in 0..<n where nanos[i] != b[i] &- a[i] {
            return XCTFail("nanoseconds_between row \(i): \(nanos[i]) != \(b[i] &- a[i])")
        }
    }

    func testBetweenAcrossMixedUnitsAndTypes() throws {
        try requireRealGPU()
        // 2021-03-04T23:59:00Z -> 2021-03-05T00:01:00Z is one day, one hour and two minutes apart.
        let secs = try timestamp([1_614_902_340], .second)
        let millis = try MetalTemporalArray(type: .timestamp(.milli, timezone: "UTC"), [1_614_902_460_000])
        XCTAssertEqual(try secs.daysBetween(millis).toRawArray(), [1])
        XCTAssertEqual(try secs.hoursBetween(millis).toRawArray(), [1])
        XCTAssertEqual(try secs.minutesBetween(millis).toRawArray(), [2])
        XCTAssertEqual(try secs.secondsBetween(millis).toRawArray(), [120])
        XCTAssertEqual(try secs.millisecondsBetween(millis).toRawArray(), [120_000])
        XCTAssertEqual(try millis.secondsBetween(secs).toRawArray(), [-120])
        // date32 against a nanosecond timestamp: the day is expanded, the timestamp truncated.
        let d32 = try MetalTemporalArray(type: .date32, [Int64(0)])
        let ns = try MetalTemporalArray(type: .timestamp(.nano, timezone: nil), [86_400_000_000_001])
        XCTAssertEqual(try d32.hoursBetween(ns).toRawArray(), [24])
        XCTAssertEqual(try d32.secondsBetween(ns).toRawArray(), [86_400])
        XCTAssertEqual(try d32.nanosecondsBetween(ns).toRawArray(), [86_400_000_000_001])
        XCTAssertEqual(try d32.yearsBetween(ns).toRawArray(), [0])
        // date32 against date64, and both directions of a negative span.
        let d64 = try MetalTemporalArray(type: .date64, [Int64(400) * 86_400_000])
        XCTAssertEqual(try d32.yearsBetween(d64).toRawArray(), [1])
        XCTAssertEqual(try d64.yearsBetween(d32).toRawArray(), [-1])
        XCTAssertEqual(try d32.quartersBetween(d64).toRawArray(), [4])
        XCTAssertEqual(try d32.monthsBetween(d64).toRawArray(), [13])
        XCTAssertEqual(try d32.weeksBetween(d64).toRawArray(), [57])
        // time32 / time64 have a clock but no date: the fixed units work, the calendar ones do not.
        let t32 = try MetalTemporalArray(type: .time32(.second), [Int64(0)])
        let t64 = try MetalTemporalArray(type: .time64(.micro), [Int64(3_600_000_001)])
        XCTAssertEqual(try t32.secondsBetween(t64).toRawArray(), [3_600])
        XCTAssertEqual(try t32.hoursBetween(t64).toRawArray(), [1])
        XCTAssertThrowsError(try t32.yearsBetween(t64))
        XCTAssertThrowsError(try t32.weeksBetween(t64))
        // duration carries no clock and no date.
        let dur = try MetalTemporalArray(type: .duration(.second), [Int64(0)])
        XCTAssertThrowsError(try dur.secondsBetween(dur))
        XCTAssertThrowsError(try dur.yearsBetween(dur))
        XCTAssertThrowsError(try secs.hoursBetween(dur))
        // Lengths must match.
        XCTAssertThrowsError(try secs.yearsBetween(try timestamp([0, 1], .second)))
    }

    func testBetweenPropagatesNullsFromBothSides() throws {
        try requireRealGPU()
        let a = try timestamp([0, nil, 86_400, nil], .second)
        let b = try timestamp([86_400, 0, nil, nil], .second)
        for got in [try a.yearsBetween(b), try a.quartersBetween(b), try a.monthsBetween(b),
                    try a.weeksBetween(b), try a.hoursBetween(b), try a.minutesBetween(b),
                    try a.secondsBetween(b), try a.millisecondsBetween(b),
                    try a.microsecondsBetween(b), try a.nanosecondsBetween(b)] {
            XCTAssertEqual(got.length, 4)
            XCTAssertEqual(got.nullCount, 3)
            XCTAssertNotNil(got.toArray()[0])
            XCTAssertNil(got.toArray()[1])
            XCTAssertNil(got.toArray()[2])
            XCTAssertNil(got.toArray()[3])
        }
        XCTAssertEqual(try a.hoursBetween(b).toArray(), [24, nil, nil, nil])
        // One side without nulls keeps the other's validity, zero-copy.
        let dense = try timestamp([1, 2, 3, 4], .second)
        XCTAssertEqual(try dense.secondsBetween(b).toArray(), [86_399, -2, nil, nil])
    }

    // MARK: - Shapes

    func testEveryFunctionAtEverySize() throws {
        try requireRealGPU()
        let cal = utcCalendar()
        let iso = isoCalendarUTC()
        var cache: [Int: Int64] = [:]
        for n in Self.sizes {
            let seconds = randomSeconds(n, seed: UInt64(13_013 + n))
            let values = nullable(seconds)
            let nulls = values.filter { $0 == nil }.count
            for unit in ArrowTemporalUnit.allCases {
                let ts = try timestamp(values, unit, timezone: "UTC")
                // Sample rather than walk 300k rows through Foundation for every unit.
                let step = Swift.max(1, n / 512)

                let week = try ts.week()
                XCTAssertEqual(week.length, n); XCTAssertEqual(week.nullCount, nulls)
                let usWeek = try ts.usWeek(), usYear = try ts.usYear()
                let dow = try ts.dayOfWeek(countFromZero: false, weekStart: 7)
                let sub = try ts.subsecond()
                let dst = try ts.isDST()
                let ic = try ts.isoCalendar(), ymdS = try ts.yearMonthDay()
                XCTAssertEqual(usWeek.length, n); XCTAssertEqual(usYear.nullCount, nulls)
                XCTAssertEqual(dow.length, n); XCTAssertEqual(dow.nullCount, nulls)
                XCTAssertEqual(sub.length, n); XCTAssertEqual(sub.nullCount, nulls)
                XCTAssertEqual(dst.length, n); XCTAssertEqual(dst.nullCount, nulls)
                XCTAssertEqual(ic.length, n); XCTAssertEqual(ic.nullCount, nulls)
                XCTAssertEqual(ymdS.length, n); XCTAssertEqual(ymdS.nullCount, nulls)

                let weekV = week.toArray(), usWeekV = usWeek.toArray(), usYearV = usYear.toArray()
                let dowV = dow.toArray(), subV = sub.toArray(), dstV = dst.toArray()
                let icYear = ic.children[0].asInt64!.toRawArray()
                let ymdDay = ymdS.children[2].asInt64!.toRawArray()
                for i in stride(from: 0, to: n, by: step) {
                    guard let s = values[i] else {
                        XCTAssertNil(weekV[i]); XCTAssertNil(subV[i]); XCTAssertNil(dstV[i])
                        continue
                    }
                    let days = floorDiv(s, 86_400)
                    XCTAssertEqual(weekV[i], oracleWeek(days, weekStartsMonday: true, countFromZero: false,
                                                        firstWeekIsFullyInYear: false, cal, &cache))
                    XCTAssertEqual(usWeekV[i], oracleWeek(days, weekStartsMonday: false, countFromZero: false,
                                                          firstWeekIsFullyInYear: false, cal, &cache))
                    XCTAssertEqual(usYearV[i], oracleUSYear(days, cal))
                    XCTAssertEqual(dowV[i], Int64((isoWeekday(days, cal) - 7 + 7) % 7 + 1))
                    XCTAssertEqual(subV[i], 0.0, "whole seconds have no fraction")
                    XCTAssertEqual(dstV[i], false, "UTC never observes DST")
                    XCTAssertEqual(icYear[i], Int64(iso.component(.yearForWeekOfYear, from: date(days: days))))
                    XCTAssertEqual(ymdDay[i], Int64(ymd(days, cal).d))
                }

                // Differences against a copy shifted by exactly forty days.
                let shifted = try timestamp(values.map { $0.map { $0 &+ 40 * 86_400 } }, unit)
                for got in [try ts.yearsBetween(shifted), try ts.quartersBetween(shifted),
                            try ts.monthsBetween(shifted), try ts.weeksBetween(shifted),
                            try ts.hoursBetween(shifted), try ts.minutesBetween(shifted),
                            try ts.secondsBetween(shifted), try ts.millisecondsBetween(shifted),
                            try ts.microsecondsBetween(shifted), try ts.nanosecondsBetween(shifted)] {
                    XCTAssertEqual(got.length, n)
                    XCTAssertEqual(got.nullCount, nulls)
                }
                let days = try ts.daysBetween(shifted).toArray()
                let weeks = try ts.weeksBetween(shifted).toArray()
                for i in stride(from: 0, to: n, by: step) where values[i] != nil {
                    XCTAssertEqual(days[i], 40)
                    XCTAssertTrue(weeks[i] == 5 || weeks[i] == 6, "40 days is five or six week boundaries")
                }
            }
            // date32 and date64 shapes too.
            let d32 = try MetalTemporalArray(type: .date32, values.map { $0.map { floorDiv($0, 86_400) } })
            XCTAssertEqual(try d32.week().length, n)
            XCTAssertEqual(try d32.week().nullCount, nulls)
            XCTAssertEqual(try d32.isoCalendar().nullCount, nulls)
            XCTAssertEqual(try d32.yearsBetween(d32).toRawArray().allSatisfy { $0 == 0 }, true)
            let d64 = try MetalTemporalArray(type: .date64, values.map { $0.map { $0 &* 1_000 } })
            XCTAssertEqual(try d64.usWeek().length, n)
            XCTAssertEqual(try d64.subsecond().nullCount, nulls)
        }
    }
}
