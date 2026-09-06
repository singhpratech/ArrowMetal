import XCTest
import CArrowABI
@testable import ArrowMetal

/// Builds C Data Interface structs the way pyarrow / arrow-rs / nanoarrow would: page-aligned buffers,
/// a release callback, and children hanging off `dictionary`. Everything is freed by `destroy()`.
final class CProducer {
    private var allocs: [UnsafeMutableRawPointer] = []
    private var lists: [UnsafeMutablePointer<UnsafeRawPointer?>] = []
    private var arrays: [UnsafeMutablePointer<ArrowArray>] = []
    private var schemas: [UnsafeMutablePointer<ArrowSchema>] = []
    private var strings: [UnsafeMutablePointer<CChar>] = []

    func raw(_ byteCount: Int) -> UnsafeMutableRawPointer {
        let n = Swift.max(byteCount, 1)
        let p = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: metalPageSize())
        memset(p, 0, n)
        allocs.append(p)
        return p
    }
    func copied<T>(_ values: [T]) -> UnsafeRawPointer {
        let p = raw(MemoryLayout<T>.stride * Swift.max(values.count, 1))
        values.withUnsafeBytes { b in if b.count > 0 { memcpy(p, b.baseAddress!, b.count) } }
        return UnsafeRawPointer(p)
    }
    func bitmap(_ valid: [Bool]) -> UnsafeRawPointer {
        let p = raw(Bitmap.byteCount(bits: valid.count))
        let bp = p.assumingMemoryBound(to: UInt8.self)
        for (i, v) in valid.enumerated() where v { Bitmap.set(bp, i) }
        return UnsafeRawPointer(p)
    }

    func array(length: Int, nullCount: Int, buffers: [UnsafeRawPointer?]) -> UnsafeMutablePointer<ArrowArray> {
        let list = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: Swift.max(buffers.count, 1))
        for (i, b) in buffers.enumerated() { list[i] = b }
        lists.append(list)
        let a = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
        a.initialize(to: ArrowArray())
        a.pointee.length = Int64(length)
        a.pointee.null_count = Int64(nullCount)
        a.pointee.offset = 0
        a.pointee.n_buffers = Int64(buffers.count)
        a.pointee.n_children = 0
        a.pointee.buffers = list
        a.pointee.release = { p in p?.pointee.release = nil }
        arrays.append(a)
        return a
    }

    /// A `utf8`, `binary` or `large_binary` array (validity, offsets, data).
    func varBinary(_ values: [[UInt8]?], large: Bool) -> UnsafeMutablePointer<ArrowArray> {
        var data: [UInt8] = []
        var valid: [Bool] = []
        var off32: [Int32] = [0], off64: [Int64] = [0]
        for v in values {
            if let v { data.append(contentsOf: v); valid.append(true) } else { valid.append(false) }
            off32.append(Int32(data.count)); off64.append(Int64(data.count))
        }
        let nulls = valid.filter { !$0 }.count
        return array(length: values.count, nullCount: nulls,
                     buffers: [nulls == 0 ? nil : bitmap(valid), large ? copied(off64) : copied(off32), copied(data)])
    }

    func schema(_ format: String, dictionary: UnsafeMutablePointer<ArrowSchema>? = nil) -> UnsafeMutablePointer<ArrowSchema> {
        let s = UnsafeMutablePointer<ArrowSchema>.allocate(capacity: 1)
        s.initialize(to: ArrowSchema())
        let f = strdup(format)!
        strings.append(f)
        s.pointee.format = UnsafePointer(f)
        s.pointee.name = nil
        s.pointee.metadata = nil
        s.pointee.flags = Int64(ARROW_FLAG_NULLABLE)
        s.pointee.n_children = 0
        s.pointee.children = nil
        s.pointee.dictionary = dictionary
        s.pointee.release = { p in p?.pointee.release = nil }
        schemas.append(s)
        return s
    }

    func destroy() {
        for a in arrays { a.deallocate() }
        for s in schemas { s.deallocate() }
        for l in lists { l.deallocate() }
        for p in allocs { p.deallocate() }
        for s in strings { free(s) }
        arrays = []; schemas = []; lists = []; allocs = []; strings = []
    }
}

final class TemporalTests: XCTestCase {
    // MARK: - Types and format strings

    func testTemporalFormatStrings() throws {
        let cases: [(String, ArrowTemporalType)] = [
            ("tdD", .date32), ("tdm", .date64),
            ("tts", .time32(.second)), ("ttm", .time32(.milli)),
            ("ttu", .time64(.micro)), ("ttn", .time64(.nano)),
            ("tss:", .timestamp(.second, timezone: nil)), ("tsm:UTC", .timestamp(.milli, timezone: "UTC")),
            ("tsu:Europe/Berlin", .timestamp(.micro, timezone: "Europe/Berlin")),
            ("tsn:+02:00", .timestamp(.nano, timezone: "+02:00")),
            ("tDs", .duration(.second)), ("tDm", .duration(.milli)),
            ("tDu", .duration(.micro)), ("tDn", .duration(.nano)),
        ]
        for (f, t) in cases {
            XCTAssertEqual(try ArrowTemporalType.parse(f), t, f)
            XCTAssertEqual(t.arrowFormat, f, f)
        }
        XCTAssertEqual(ArrowTemporalType.date32.usesInt64, false)
        XCTAssertEqual(ArrowTemporalType.time32(.milli).usesInt64, false)
        XCTAssertEqual(ArrowTemporalType.time64(.nano).usesInt64, true)
        XCTAssertEqual(ArrowTemporalType.timestamp(.micro, timezone: "UTC").timezone, "UTC")
        // Not temporal, or not supported.
        XCTAssertNil(ArrowTemporalType(format: "tiM"))       // interval
        XCTAssertNil(ArrowTemporalType(format: "ts"))
        XCTAssertNil(ArrowTemporalType(format: "tsx:UTC"))
        XCTAssertNil(ArrowTemporalType(format: "i"))
        XCTAssertThrowsError(try ArrowTemporalType.parse("tsx:UTC"))
        // Nonsensical widths are rejected rather than silently reinterpreted.
        XCTAssertThrowsError(try MetalTemporalArray(type: .time32(.nano), try MetalArray<Int32>([1])))
        XCTAssertThrowsError(try MetalTemporalArray(type: .date32, try MetalArray<Int64>([1])))
        XCTAssertThrowsError(try MetalTemporalArray(type: .timestamp(.second, timezone: nil), try MetalArray<Int32>([1])))
    }

    // MARK: - C Data Interface round trips

    func testRoundTripEveryTemporalTypeThroughCStructs() throws {
        try requireRealGPU()
        let p = CProducer()
        let types: [ArrowTemporalType] = [
            .date32, .date64, .time32(.second), .time32(.milli), .time64(.micro), .time64(.nano),
            .timestamp(.second, timezone: nil), .timestamp(.milli, timezone: "UTC"),
            .timestamp(.micro, timezone: "Europe/Berlin"), .timestamp(.nano, timezone: nil),
            .duration(.second), .duration(.milli), .duration(.micro), .duration(.nano),
        ]
        do {
            for t in types {
                let values: [Int64] = [0, 1, -1, 123_456, -987_654]
                let valid = [true, true, false, true, true]
                let expected: [Int64?] = [0, 1, nil, 123_456, -987_654]
                let buffers: [UnsafeRawPointer?] = t.usesInt64
                    ? [p.bitmap(valid), p.copied(values)]
                    : [p.bitmap(valid), p.copied(values.map { Int32($0) })]
                let arr = p.array(length: 5, nullCount: 1, buffers: buffers)
                let schema = p.schema(t.arrowFormat)
                let r = try importArrowArray(schema: schema, array: arr)
                guard case .temporal(let ta) = r.array else { return XCTFail("\(t.arrowFormat) did not import as temporal") }
                XCTAssertEqual(ta.type, t, t.arrowFormat)
                XCTAssertEqual(ta.length, 5)
                XCTAssertEqual(ta.nullCount, 1)
                XCTAssertEqual(ta.toArray(), expected, t.arrowFormat)
                XCTAssertEqual(r.array.arrowFormat, t.arrowFormat)
                XCTAssertEqual(r.array.length, 5)
                XCTAssertEqual(r.array.nullCount, 1)
                // Export and re-import: the format string (timezone included) survives.
                var s2 = ArrowSchema(), a2 = ArrowArray()
                r.array.exportArrowSchema(name: "when", into: &s2)
                r.array.exportArrowArray(into: &a2)
                XCTAssertEqual(String(cString: s2.format), t.arrowFormat)
                XCTAssertEqual(String(cString: s2.name), "when")
                XCTAssertEqual(a2.length, 5)
                XCTAssertEqual(a2.n_buffers, 2)
                XCTAssertNil(a2.dictionary)
                let back = try importArrowArray(schema: &s2, array: &a2)
                XCTAssertEqual(back.array.asTemporal?.type, t)
                XCTAssertEqual(back.array.asTemporal?.toArray(), expected, t.arrowFormat)
                s2.release?(&s2)
            }
        }
        p.destroy()
    }

    func testUnsupportedTemporalFormatThrows() throws {
        let p = CProducer()
        let arr = p.array(length: 1, nullCount: 0, buffers: [nil, p.copied([Int32(0)])])
        // "tdX" is not a temporal format at all. ("tiM", the month interval, used to land here; it is now
        // imported as an interval column by TypesExtra.swift, which the assertion below records.)
        XCTAssertThrowsError(try importArrowArray(schema: p.schema("tdX"), array: arr))
        let ivArr = p.array(length: 1, nullCount: 0, buffers: [nil, p.copied([Int32(0)])])
        XCTAssertEqual(try importArrowArray(schema: p.schema("tiM"), array: ivArr).array.arrowFormat, "tiM")
        p.destroy()
    }

    // MARK: - Calendar field extraction against Foundation

    func testExtractionMatchesFoundationCalendarUTC() throws {
        try requireRealGPU()
        let n = 200_000
        var g = SystemRandomNumberGenerator()
        let lo: Int64 = -2_208_988_800     // 1900-01-01T00:00:00Z
        let hi: Int64 = 4_102_444_800      // 2100-01-01T00:00:00Z
        var secs: [Int64] = []
        secs.reserveCapacity(n)
        for _ in 0..<n { secs.append(Int64.random(in: lo...hi, using: &g)) }
        // Pin the interesting cases: the epoch itself and values just before it.
        secs[0] = 0; secs[1] = -1; secs[2] = lo; secs[3] = hi; secs[4] = -86_400; secs[5] = -86_401
        secs[6] = -2_208_988_801; secs[7] = 951_782_400   // 2000-02-29, a leap day

        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: "UTC"), secs.map { Optional($0) })
        let year = try ts.year().toRawArray()
        let month = try ts.month().toRawArray()
        let day = try ts.day().toRawArray()
        let dow = try ts.dayOfWeek().toRawArray()
        let hour = try ts.hour().toRawArray()
        let minute = try ts.minute().toRawArray()
        let second = try ts.second().toRawArray()
        let epochDays = try ts.toDate32().asInt32!.toRawArray()

        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let fields: Set<Calendar.Component> = [.year, .month, .day, .weekday, .hour, .minute, .second]
        var problem: String? = nil
        for i in 0..<n {
            let c = cal.dateComponents(fields, from: Date(timeIntervalSince1970: Double(secs[i])))
            // Foundation: 1 = Sunday ... 7 = Saturday. Arrow's day_of_week: 0 = Monday ... 6 = Sunday.
            let expectedDow = Int32((c.weekday! + 5) % 7)
            let expectedDays = Int32(Int((Double(secs[i]) / 86400.0).rounded(.down)))
            if year[i] != Int32(c.year!) { problem = "year \(year[i]) != \(c.year!)" }
            else if month[i] != Int32(c.month!) { problem = "month \(month[i]) != \(c.month!)" }
            else if day[i] != Int32(c.day!) { problem = "day \(day[i]) != \(c.day!)" }
            else if dow[i] != expectedDow { problem = "day of week \(dow[i]) != \(expectedDow)" }
            else if hour[i] != Int32(c.hour!) { problem = "hour \(hour[i]) != \(c.hour!)" }
            else if minute[i] != Int32(c.minute!) { problem = "minute \(minute[i]) != \(c.minute!)" }
            else if second[i] != Int32(c.second!) { problem = "second \(second[i]) != \(c.second!)" }
            else if epochDays[i] != expectedDays { problem = "epoch days \(epochDays[i]) != \(expectedDays)" }
            if let problem { return XCTFail("\(problem) at row \(i), timestamp \(secs[i])") }
        }
        XCTAssertNil(problem)
    }

    func testExtractionAcrossUnitsAndTypes() throws {
        try requireRealGPU()
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let fields: Set<Calendar.Component> = [.year, .month, .day, .hour, .minute, .second]
        var g = SystemRandomNumberGenerator()
        let n = 20_000
        var secs: [Int64] = []
        for _ in 0..<n { secs.append(Int64.random(in: -2_208_988_800...4_102_444_800, using: &g)) }
        secs[0] = -1; secs[1] = 0; secs[2] = -86_400

        for unit in [ArrowTemporalUnit.milli, .micro, .nano] {
            // A sub-second offset must never change the extracted date or time, even for negative epochs.
            let scale = unit.perSecond
            let ticks = secs.map { $0 * scale + ($0 < 0 ? -(scale / 2) : scale / 2) }
            let ts = try MetalTemporalArray(type: .timestamp(unit, timezone: nil), ticks.map { Optional($0) })
            let year = try ts.year().toRawArray(), hour = try ts.hour().toRawArray(), second = try ts.second().toRawArray()
            for i in 0..<n {
                // Floor semantics: a negative sub-second offset belongs to the previous second.
                let s = secs[i] < 0 ? secs[i] - 1 : secs[i]
                let c = cal.dateComponents(fields, from: Date(timeIntervalSince1970: Double(s)))
                if year[i] != Int32(c.year!) || hour[i] != Int32(c.hour!) || second[i] != Int32(c.second!) {
                    return XCTFail("\(unit.arrowName) row \(i): \(year[i])/\(hour[i])/\(second[i]) vs \(c)")
                }
            }
        }
        // date32: days since the epoch, no time component.
        let d32 = try MetalTemporalArray(type: .date32, [0, -1, 19_000, -25_567])
        XCTAssertEqual(try d32.year().toRawArray(), [1970, 1969, 2022, 1900])
        XCTAssertEqual(try d32.month().toRawArray(), [1, 12, 1, 1])
        XCTAssertEqual(try d32.day().toRawArray(), [1, 31, 8, 1])
        XCTAssertEqual(try d32.hour().toRawArray(), [0, 0, 0, 0])
        XCTAssertEqual(try d32.dayOfWeek().toRawArray(), [3, 2, 5, 0])   // Thu, Wed, Sat, Mon
        // date64: milliseconds since the epoch.
        let d64 = try MetalTemporalArray(type: .date64, [0, -86_400_000, 1_640_995_200_000])
        XCTAssertEqual(try d64.year().toRawArray(), [1970, 1969, 2022])
        XCTAssertEqual(try d64.day().toRawArray(), [1, 31, 1])
        // time32 / time64: time of day only; date fields are rejected.
        let t32 = try MetalTemporalArray(type: .time32(.second), [0, 3_661, 86_399])
        XCTAssertEqual(try t32.hour().toRawArray(), [0, 1, 23])
        XCTAssertEqual(try t32.minute().toRawArray(), [0, 1, 59])
        XCTAssertEqual(try t32.second().toRawArray(), [0, 1, 59])
        XCTAssertThrowsError(try t32.year())
        let t64 = try MetalTemporalArray(type: .time64(.nano), [0, 3_661_000_000_001, 86_399_999_999_999])
        XCTAssertEqual(try t64.hour().toRawArray(), [0, 1, 23])
        XCTAssertEqual(try t64.second().toRawArray(), [0, 1, 59])
        // duration has no calendar meaning.
        XCTAssertThrowsError(try MetalTemporalArray(type: .duration(.second), [1]).hour())
        // Nulls stay null.
        let withNulls = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [0, nil, 86_400])
        XCTAssertEqual(try withNulls.day().toArray(), [1, nil, 2])
        XCTAssertEqual(try withNulls.year().nullCount, 1)
        // Empty arrays are fine.
        XCTAssertEqual(try MetalTemporalArray(type: .date32, [Int64?]()).year().length, 0)
    }

    // MARK: - Forwarded integer kernels

    func testFilterSortAndReduceOnTimestamps() throws {
        try requireRealGPU()
        let n = 100_000
        var g = SystemRandomNumberGenerator()
        var vals: [Int64?] = []
        for i in 0..<n {
            vals.append(i % 11 == 0 ? nil : Int64.random(in: -2_000_000_000_000...2_000_000_000_000, using: &g))
        }
        let ts = try MetalTemporalArray(type: .timestamp(.milli, timezone: "UTC"), vals)
        XCTAssertEqual(ts.length, n)
        XCTAssertEqual(ts.nullCount, vals.filter { $0 == nil }.count)

        let cutoff: Int64 = 0
        let mask = try ts.compare(.gt, cutoff)
        let kept = try ts.filter(mask)
        XCTAssertEqual(kept.type, ts.type)
        XCTAssertEqual(kept.toArray(), vals.filter { ($0 ?? Int64.min) > cutoff })
        XCTAssertEqual(try ts.filter(where: .gt, cutoff).toArray(), kept.toArray())

        XCTAssertEqual(try ts.min(), vals.compactMap { $0 }.min())
        XCTAssertEqual(try ts.max(), vals.compactMap { $0 }.max())

        let sorted = try ts.sorted()
        var expected: [Int64?] = vals.compactMap { $0 }.sorted().map { Optional($0) }
        expected.append(contentsOf: [Int64?](repeating: nil, count: ts.nullCount))
        XCTAssertEqual(sorted.toArray(), expected)
        XCTAssertEqual(sorted.type, ts.type)
        let desc = try ts.sorted(descending: true)
        XCTAssertEqual(desc.toArray().prefix(3).compactMap { $0 }, Array(vals.compactMap { $0 }.sorted().reversed().prefix(3)))

        // take, slice and element-wise comparison against another temporal array.
        let idx = try MetalArray<Int32>([3, 1, 0, Int32(n - 1)])
        XCTAssertEqual(try ts.take(idx).toArray(), [vals[3], vals[1], vals[0], vals[n - 1]])
        let window = try ts.slice(offset: 64, length: 100)
        XCTAssertEqual(window.toArray(), Array(vals[64..<164]))
        let other = try MetalTemporalArray(type: ts.type, vals.reversed().map { $0 })
        XCTAssertEqual(try ts.compare(.lt, other).length, n)
        XCTAssertThrowsError(try ts.compare(.eq, try MetalTemporalArray(type: .date32, vals.map { _ in Int64(0) })))
        // An int32-backed type rejects scalars it could not represent.
        XCTAssertThrowsError(try MetalTemporalArray(type: .date32, [1, 2]).compare(.gt, 1 << 40))
    }

    func testCastUnitAndToDate32() throws {
        try requireRealGPU()
        let secs = try MetalTemporalArray(type: .timestamp(.second, timezone: "UTC"), [0, 1, -1, 1_700_000_000])
        let millis = try secs.castUnit(to: .milli)
        XCTAssertEqual(millis.type, .timestamp(.milli, timezone: "UTC"))
        XCTAssertEqual(millis.toArray(), [0, 1_000, -1_000, 1_700_000_000_000])
        let back = try millis.castUnit(to: .second)
        XCTAssertEqual(back.toArray(), [0, 1, -1, 1_700_000_000])
        XCTAssertEqual(try millis.castUnit(to: .nano).toArray(), [0, 1_000_000_000, -1_000_000_000, 1_700_000_000_000_000_000])
        // Durations keep their kind, times pick the storage width their unit requires.
        XCTAssertEqual(try MetalTemporalArray(type: .duration(.second), [5]).castUnit(to: .milli).type, .duration(.milli))
        let t32 = try MetalTemporalArray(type: .time32(.second), [3_661])
        let t64 = try t32.castUnit(to: .micro)
        XCTAssertEqual(t64.type, .time64(.micro))
        XCTAssertEqual(t64.toArray(), [3_661_000_000])
        XCTAssertEqual(try t64.castUnit(to: .milli).type, .time32(.milli))
        XCTAssertEqual(try t64.castUnit(to: .milli).toArray(), [3_661_000])
        XCTAssertThrowsError(try MetalTemporalArray(type: .date32, [1]).castUnit(to: .milli))
        // toDate32 from every epoch-based type.
        XCTAssertEqual(try secs.toDate32().type, .date32)
        XCTAssertEqual(try secs.toDate32().toArray(), [0, 0, -1, 19_675])
        XCTAssertEqual(try MetalTemporalArray(type: .date64, [-1]).toDate32().toArray(), [-1])
        XCTAssertTrue(try MetalTemporalArray(type: .date32, [7]).toDate32().toArray() == [7])
    }

    func testTemporalColumnInARecordBatch() throws {
        try requireRealGPU()
        let n = 1_000
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: "UTC"), (0..<n).map { Int64($0) * 3_600 })
        let batch = try MetalRecordBatch(names: ["id", "when"], columns: [
            .int32(try MetalArray<Int32>((0..<n).map { Int32($0) })),
            .temporal(ts),
        ])
        let mask = try ts.compare(.ge, 500 * 3_600)
        let filtered = try batch.filter(mask)
        XCTAssertEqual(filtered.length, 500)
        XCTAssertEqual(filtered["when"]!.asTemporal!.toArray().first!, 500 * 3_600)
        XCTAssertEqual(try batch.take(try MetalArray<Int32>([2, 5]))["when"]!.asTemporal?.toArray(), [7_200, 18_000])
        XCTAssertEqual(try batch.slice(offset: 32, length: 2)["when"]!.asTemporal?.toArray(), [115_200, 118_800])
        XCTAssertEqual(batch["when"]!.nullCount, 0)
        // Struct export keeps the temporal format string on the child schema.
        var schema = ArrowSchema(), arr = ArrowArray()
        batch.exportArrowSchema(name: "b", into: &schema)
        batch.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.children[1]!.pointee.format), "tss:UTC")
        let r = try importArrowRecordBatch(schema: &schema, array: &arr)
        XCTAssertEqual(r.batch["when"]!.asTemporal?.type, .timestamp(.second, timezone: "UTC"))
        XCTAssertEqual(r.batch["when"]!.asTemporal?.toArray().last, Int64(n - 1) * 3_600)
        schema.release?(&schema)
    }

    // MARK: - Binary

    func testBinaryRoundTrip() throws {
        try requireRealGPU()
        let p = CProducer()
        let values: [[UInt8]?] = [[0x00, 0x01, 0xFF], [], nil, Array("héllo".utf8), [0xDE, 0xAD, 0xBE, 0xEF]]
        do {
            for large in [false, true] {
                let arr = p.varBinary(values, large: large)
                let r = try importArrowArray(schema: p.schema(large ? "Z" : "z"), array: arr)
                guard case .binary(let b) = r.array else { return XCTFail("binary did not import as .binary") }
                XCTAssertTrue(b.isBinary)
                XCTAssertEqual(b.length, 5)
                XCTAssertEqual(b.nullCount, 1)
                XCTAssertEqual(b.toByteArrays(), values)
                XCTAssertEqual(r.array.arrowFormat, "z", "large_binary is narrowed and exported as binary")
                // Export: always "z", with the three utf8-shaped buffers.
                var s2 = ArrowSchema(), a2 = ArrowArray()
                r.array.exportArrowSchema(name: "blob", into: &s2)
                r.array.exportArrowArray(into: &a2)
                XCTAssertEqual(String(cString: s2.format), "z")
                XCTAssertEqual(a2.n_buffers, 3)
                let back = try importArrowArray(schema: &s2, array: &a2)
                XCTAssertEqual(back.array.asBinary?.toByteArrays(), values)
                s2.release?(&s2)
                // Selection keeps the binary flag.
                let mask = try MetalBooleanArray([true, false, true, true, false])
                XCTAssertEqual(try r.array.filter(mask).arrowFormat, "z")
                XCTAssertEqual(try r.array.filter(mask).asBinary?.toByteArrays(), [values[0], values[2], values[3]])
                XCTAssertEqual(try r.array.take(try MetalArray<Int32>([4, 0])).asBinary?.toByteArrays(), [values[4], values[0]])
                XCTAssertEqual(try r.array.slice(offset: 3, length: 2).asBinary?.toByteArrays(), [values[3], values[4]])
                XCTAssertEqual(try r.array.filter(mask).nullCount, 1)
            }
            // utf8 still imports as .string and exports as "u".
            let u = p.varBinary(values.map { (v: [UInt8]?) -> [UInt8]? in v == nil ? nil : Array("x".utf8) }, large: false)
            let ur = try importArrowArray(schema: p.schema("u"), array: u)
            XCTAssertEqual(ur.array.arrowFormat, "u")
            XCTAssertNotNil(ur.array.asString)
            // Round trip through the Swift constructor.
            let built = try MetalStringArray(bytes: values)
            XCTAssertTrue(built.isBinary)
            XCTAssertEqual(built.toByteArrays(), values)
            XCTAssertEqual(built.nullCount, 1)
        }
        p.destroy()
    }

    // MARK: - Dictionary

    func testDictionaryImportDecodeAndExport() throws {
        try requireRealGPU()
        let p = CProducer()
        let unique = ["alpha", "beta", "gamma"]
        let codeValues: [Int32] = [0, 2, 1, 0, 2, 2, 1, 0]
        let valid = [true, true, true, false, true, true, true, true]
        let expected: [String?] = [ "alpha", "gamma", "beta", nil, "gamma", "gamma", "beta", "alpha"]
        do {
            for idxFormat in ["i", "l"] {
                let valuesArr = p.varBinary(unique.map { Array($0.utf8) }, large: false)
                let codes: [UnsafeRawPointer?] = idxFormat == "i"
                    ? [p.bitmap(valid), p.copied(codeValues)]
                    : [p.bitmap(valid), p.copied(codeValues.map { Int64($0) })]
                let arr = p.array(length: codeValues.count, nullCount: 1, buffers: codes)
                arr.pointee.dictionary = valuesArr
                let schema = p.schema(idxFormat, dictionary: p.schema("u"))
                let r = try importArrowArray(schema: schema, array: arr)
                guard case .dictionary(let c, let v) = r.array else { return XCTFail("did not import as .dictionary") }
                XCTAssertEqual(c.length, 8)
                XCTAssertEqual(c.nullCount, 1)
                XCTAssertEqual(v.asString?.toArray(), unique)
                XCTAssertEqual(r.array.length, 8)
                XCTAssertEqual(r.array.nullCount, 1)
                XCTAssertEqual(r.array.arrowFormat, "i")
                // decode() materialises with take.
                XCTAssertEqual(try r.array.decode().asString?.toArray(), expected)
                // Selection acts on the codes and keeps the values.
                let mask = try MetalBooleanArray([true, true, false, true, false, false, true, true])
                let filtered = try r.array.filter(mask)
                XCTAssertEqual(try filtered.decode().asString?.toArray(), ["alpha", "gamma", nil, "beta", "alpha"])
                XCTAssertEqual(try r.array.take(try MetalArray<Int32>([1, 0])).decode().asString?.toArray(), ["gamma", "alpha"])
                XCTAssertEqual(try r.array.slice(offset: 4, length: 2).decode().asString?.toArray(), ["gamma", "gamma"])
                // Export writes a dictionary schema and array, and re-imports identically.
                var s2 = ArrowSchema(), a2 = ArrowArray()
                r.array.exportArrowSchema(name: "k", into: &s2)
                r.array.exportArrowArray(into: &a2)
                XCTAssertEqual(String(cString: s2.format), "i")
                XCTAssertNotNil(s2.dictionary)
                XCTAssertEqual(String(cString: s2.dictionary!.pointee.format), "u")
                XCTAssertEqual(a2.length, 8)
                XCTAssertEqual(a2.n_buffers, 2)
                XCTAssertNotNil(a2.dictionary)
                XCTAssertEqual(a2.dictionary!.pointee.length, 3)
                let back = try importArrowArray(schema: &s2, array: &a2)
                XCTAssertNotNil(back.array.asDictionary)
                XCTAssertEqual(try back.array.decode().asString?.toArray(), expected)
                s2.release?(&s2)
            }
            // A primitive dictionary (int32 values) works the same way.
            let valuesArr = p.array(length: 3, nullCount: 0, buffers: [nil, p.copied([Int32(10), 20, 30])])
            let arr = p.array(length: 4, nullCount: 0, buffers: [nil, p.copied([Int32(2), 0, 1, 2])])
            arr.pointee.dictionary = valuesArr
            let r = try importArrowArray(schema: p.schema("i", dictionary: p.schema("i")), array: arr)
            XCTAssertEqual(try r.array.decode().asInt32?.toRawArray(), [30, 10, 20, 30])
            // decode() on a non-dictionary array is the identity.
            XCTAssertEqual(try r.array.decode().decode().asInt32?.toRawArray(), [30, 10, 20, 30])
            // Codes feed group-by directly.
            XCTAssertEqual(try r.array.asDictionary!.codes.groupBy(keyCount: 3).count().toRawArray(), [1, 1, 2])
        }
        p.destroy()
    }

    func testDictionaryOfTimestamps() throws {
        try requireRealGPU()
        let p = CProducer()
        do {
            let valuesArr = p.array(length: 3, nullCount: 0, buffers: [nil, p.copied([Int64(0), 86_400, -86_400])])
            let arr = p.array(length: 4, nullCount: 0, buffers: [nil, p.copied([Int32(2), 1, 0, 2])])
            arr.pointee.dictionary = valuesArr
            let r = try importArrowArray(schema: p.schema("i", dictionary: p.schema("tss:UTC")), array: arr)
            guard case .dictionary(_, let v) = r.array else { return XCTFail("not a dictionary") }
            XCTAssertEqual(v.asTemporal?.type, .timestamp(.second, timezone: "UTC"))
            let decoded = try r.array.decode()
            XCTAssertEqual(decoded.asTemporal?.toArray(), [-86_400, 86_400, 0, -86_400])
            XCTAssertEqual(try decoded.asTemporal?.day().toRawArray(), [31, 2, 1, 31])
        }
        p.destroy()
    }
}
