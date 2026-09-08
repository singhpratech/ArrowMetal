import XCTest
import CArrowABI
@testable import ArrowMetal

// The remaining Arrow type-matrix rows and the type-adjacent functions:
// `null`, `float16`, `decimal32` / `decimal64`, the three `interval` layouts, `fixed_size_binary`,
// `list_view` / `large_list_view`, extension types, `list_parent_indices`, `list_slice`, `map_lookup`,
// `assume_timezone`, `local_timestamp` and the three `*_interval_between` functions.
//
// The interop tests build the same C Data Interface structs pyarrow exports (the `cSchema` / `cArray`
// helpers from NestedTests.swift), so the importer meets a foreign-shaped producer rather than
// ArrowMetal's own exports; the kernel tests check every GPU result against a plain Swift oracle at
// 0, 1, 33, 4097 and 200_003 rows, with nulls throughout.

private let sizes = [0, 1, 33, 4097, 200_003]

// MARK: - Host oracles

/// Howard Hinnant's civil-calendar algorithms, on the host, as the oracle for the GPU versions.
enum CivilOracle {
    static func floorDiv(_ a: Int64, _ b: Int64) -> Int64 {
        var q = a / b
        if a % b != 0 && ((a < 0) != (b < 0)) { q -= 1 }
        return q
    }
    static func civilFromDays(_ z0: Int64) -> (y: Int64, m: Int64, d: Int64) {
        var z = z0 + 719_468
        let era = (z >= 0 ? z : z - 146_096) / 146_097
        let doe = z - era * 146_097
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365
        let yy = yoe + era * 400
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp + (mp < 10 ? 3 : -9)
        z = 0
        return (yy + (m <= 2 ? 1 : 0), m, d)
    }
    static func daysFromCivil(_ y0: Int64, _ m: Int64, _ d: Int64) -> Int64 {
        let y = y0 - (m <= 2 ? 1 : 0)
        let era = (y >= 0 ? y : y - 399) / 400
        let yoe = y - era * 400
        let doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return era * 146_097 + doe - 719_468
    }
    static func isLeap(_ y: Int64) -> Bool { (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 }
    static func daysInMonth(_ y: Int64, _ m: Int64) -> Int64 {
        if m == 2 { return isLeap(y) ? 29 : 28 }
        return (m == 4 || m == 6 || m == 9 || m == 11) ? 30 : 31
    }
    /// Adds an interval to `ticks` in a column of `ticksPerDay` ticks per day, with `nanoPerTick` for the
    /// sub-day part. Mirrors `MetalTemporalArray.addInterval`.
    static func addInterval(_ ticks: Int64, _ iv: ArrowInterval, ticksPerDay: Int64, subUnitPerSecond: Int64,
                            ticksPerSecond: Int64) -> Int64 {
        var days = floorDiv(ticks, ticksPerDay)
        let rem = ticks - days * ticksPerDay
        if iv.months != 0 {
            let c = civilFromDays(days)
            let total = c.y * 12 + (c.m - 1) + Int64(iv.months)
            let ny = floorDiv(total, 12)
            let nm = total - ny * 12 + 1
            let nd = Swift.min(c.d, daysInMonth(ny, nm))
            days = daysFromCivil(ny, nm, nd)
        }
        days += Int64(iv.days)
        // The interval's sub-day field, converted to the column's resolution (truncating toward zero).
        let sub = iv.nanoseconds
        let ticksSub: Int64
        if ticksPerSecond >= subUnitPerSecond { ticksSub = sub * (ticksPerSecond / subUnitPerSecond) }
        else { ticksSub = sub / (subUnitPerSecond / ticksPerSecond) }
        return days * ticksPerDay + rem + ticksSub
    }
}

/// FNV-1a 64 over bytes, the oracle for the `fixed_size_binary` hash.
private func fnv1a64(_ bytes: [UInt8]) -> UInt64 {
    var h: UInt64 = 14_695_981_039_346_656_037
    for b in bytes { h ^= UInt64(b); h = h &* 1_099_511_628_211 }
    return h
}

/// A `[UInt8]` C Data Interface metadata blob, as `ExtensionType.swift` encodes it.
private func metaBlob(_ pairs: [(String, String)]) -> [UInt8] {
    var out: [UInt8] = []
    func put(_ v: Int32) { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
    put(Int32(pairs.count))
    for (k, v) in pairs {
        let kb = Array(k.utf8), vb = Array(v.utf8)
        put(Int32(kb.count)); out.append(contentsOf: kb)
        put(Int32(vb.count)); out.append(contentsOf: vb)
    }
    return out
}

final class TypesExtraTests: XCTestCase {

    // MARK: - null

    func testNullImportExportAndSelection() throws {
        for buffers in [[], [nil]] as [[[UInt8]?]] {
            var schema = cSchema("n", name: "z")
            var arr = cArray(length: 7, nullCount: 7, buffers: buffers)
            let r = try importArrowArray(schema: &schema, array: &arr)
            XCTAssertNil(arr.release, "import must move the array")
            let n = try XCTUnwrap(r.array.asNull)
            XCTAssertEqual(n.length, 7)
            XCTAssertEqual(r.array.nullCount, 7)
            XCTAssertEqual(r.array.arrowFormat, "n")

            var outSchema = ArrowSchema(); var outArray = ArrowArray()
            r.array.exportArrowSchema(name: "z", into: &outSchema)
            r.array.exportArrowArray(into: &outArray)
            XCTAssertEqual(String(cString: outSchema.format), "n")
            XCTAssertEqual(outArray.n_buffers, 0)
            XCTAssertEqual(outArray.length, 7)
            let back = try importArrowArray(schema: &outSchema, array: &outArray)
            XCTAssertEqual(back.array.length, 7)
            outSchema.release?(&outSchema)
        }
    }

    func testNullSelection() throws {
        try requireRealGPU()
        for n in sizes {
            let a = AnyMetalArray.null(MetalNullArray(length: n))
            let mask = try MetalBooleanArray((0..<n).map { $0 % 3 == 0 })
            XCTAssertEqual(try a.filter(mask).length, (n + 2) / 3)
            let idx = try MetalArray<Int32>((0..<Swift.min(n, 5)).map { Int32($0) })
            XCTAssertEqual(try a.take(idx).length, Swift.min(n, 5))
            XCTAssertEqual(try a.slice(offset: 0, length: n).length, n)
        }
    }

    // MARK: - float16

    /// Half bit patterns for the values `f`, produced by the host encoder.
    private func halfArray(_ values: [Float?]) throws -> MetalFloat16Array {
        try MetalFloat16Array(values)
    }

    func testFloat16RoundTripAndCasts() throws {
        try requireRealGPU()
        let values: [Float?] = [1.5, nil, -3.25, 0, 65504, -65504, 6.1e-5, 1e-8, .infinity, -.infinity]
        let bits = values.map { $0.map { Float16Bits.encode($0) } }
        let valid = values.map { $0 != nil }
        var schema = cSchema("e", name: "h")
        var arr = cArray(length: values.count, nullCount: 1,
                         buffers: [bitmapBytes(valid), rawBytes(bits.map { $0 ?? 0 })])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let h = try XCTUnwrap(r.array.asFloat16)
        XCTAssertEqual(h.length, values.count)
        XCTAssertEqual(h.nullCount, 1)
        XCTAssertEqual(r.array.arrowFormat, "e")

        // GPU widening matches the host decoder exactly.
        let wide = try h.toFloat32().toArray()
        for i in 0..<values.count {
            guard let b = bits[i] else { XCTAssertNil(wide[i]); continue }
            XCTAssertEqual(wide[i], Float16Bits.decode(b), "element \(i)")
        }
        // Narrowing back is the identity on values that are already representable.
        let back = try h.toFloat32().toFloat16()
        XCTAssertEqual(back.bits.toArray().map { $0 }, bits)

        var outSchema = ArrowSchema(); var outArray = ArrowArray()
        r.array.exportArrowSchema(name: "h", into: &outSchema)
        r.array.exportArrowArray(into: &outArray)
        XCTAssertEqual(String(cString: outSchema.format), "e")
        XCTAssertEqual(outArray.n_buffers, 2)
        let again = try importArrowArray(schema: &outSchema, array: &outArray)
        // The values come back as the *half* each input rounded to: 6.1e-5 is not representable and
        // 1e-8 underflows to zero, exactly as Arrow's cast to float16 would.
        let rounded = values.map { $0.map { Float16Bits.decode(Float16Bits.encode($0)) } }
        XCTAssertEqual(try XCTUnwrap(again.array.asFloat16).toArray(), rounded)
        outSchema.release?(&outSchema)
    }

    func testFloat16ComputeThroughFloat32() throws {
        try requireRealGPU()
        for n in sizes {
            let values: [Float?] = (0..<n).map { $0 % 7 == 3 ? nil : Float($0 % 1000) - 500 }
            let h = try halfArray(values)
            let live = values.compactMap { $0 }
            XCTAssertEqual(try h.min(), live.min(), "n = \(n)")
            XCTAssertEqual(try h.max(), live.max(), "n = \(n)")
            if n > 0 {
                let gt = try h.compare(.gt, 0).toArray()
                for i in 0..<n { XCTAssertEqual(gt[i], values[i].map { $0 > 0 }, "n = \(n) i = \(i)") }
                let sum = try h.sum()
                XCTAssertEqual(sum ?? 0, Double(live.reduce(0, +)), accuracy: Double(live.count) * 1e-3)
            }
            // Selection moves the raw bit patterns.
            let take = try h.take(try MetalArray<Int32>((0..<Swift.min(n, 40)).map { Int32(n - 1 - $0) }))
            XCTAssertEqual(take.toArray(), (0..<Swift.min(n, 40)).map { values[n - 1 - $0] })
            let mask = try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })
            XCTAssertEqual(try h.filter(mask).toArray(), (0..<n).filter { $0 % 2 == 0 }.map { values[$0] })
            if n > 2 { XCTAssertEqual(try h.slice(offset: 1, length: n - 2).toArray(), Array(values[1..<(n - 1)])) }
        }
    }

    // MARK: - decimal32 / decimal64

    func testSmallDecimalRoundTripAndWidening() throws {
        try requireRealGPU()
        for (width, fmt) in [(32, "d:7,2,32"), (64, "d:15,3,64")] {
            let values: [Int64?] = [12345, nil, -678, 0, width == 32 ? 9_999_999 : 999_999_999_999_999]
            let valid = values.map { $0 != nil }
            var schema = cSchema(fmt, name: "d")
            let bytes = width == 32
                ? rawBytes(values.map { Int32($0 ?? 0) })
                : rawBytes(values.map { $0 ?? 0 })
            var arr = cArray(length: values.count, nullCount: 1, buffers: [bitmapBytes(valid), bytes])
            let r = try importArrowArray(schema: &schema, array: &arr)
            let d = try XCTUnwrap(r.array.asSmallDecimal)
            XCTAssertEqual(d.type.bitWidth, width)
            XCTAssertEqual(d.toArray(), values)
            XCTAssertEqual(r.array.arrowFormat, fmt)

            // GPU widening to decimal128 and back.
            let wide = try d.toDecimal128()
            XCTAssertEqual(wide.type.scale, d.type.scale)
            XCTAssertEqual(wide.toArray().map { $0.map { Int64(bitPattern: $0.lo) } }, values)
            let narrow = try wide.narrowed(to: d.type)
            XCTAssertEqual(narrow.toArray(), values)

            // Compute goes through decimal128.
            let sum = try wide.sum()
            XCTAssertEqual(sum.map { Int64(bitPattern: $0.lo) }, values.compactMap { $0 }.reduce(0, +))

            var outSchema = ArrowSchema(); var outArray = ArrowArray()
            r.array.exportArrowSchema(name: "d", into: &outSchema)
            r.array.exportArrowArray(into: &outArray)
            XCTAssertEqual(String(cString: outSchema.format), fmt)
            let again = try importArrowArray(schema: &outSchema, array: &outArray)
            XCTAssertEqual(try XCTUnwrap(again.array.asSmallDecimal).toArray(), values)
            outSchema.release?(&outSchema)
        }
    }

    func testSmallDecimalSelection() throws {
        try requireRealGPU()
        for n in sizes {
            let values: [Int64?] = (0..<n).map { $0 % 5 == 1 ? nil : Int64($0) - 100 }
            let t = try ArrowSmallDecimalType(precision: 15, scale: 3, bitWidth: 64)
            let d = try MetalSmallDecimalArray(type: t, values)
            let mask = try MetalBooleanArray((0..<n).map { $0 % 3 != 0 })
            XCTAssertEqual(try d.filter(mask).toArray(), (0..<n).filter { $0 % 3 != 0 }.map { values[$0] })
            let idx = (0..<Swift.min(n, 50)).map { Int32(n - 1 - $0) }
            XCTAssertEqual(try d.take(try MetalArray<Int32>(idx)).toArray(), idx.map { values[Int($0)] })
            if n > 2 { XCTAssertEqual(try d.slice(offset: 1, length: n - 2).toArray(), Array(values[1..<(n - 1)])) }
        }
    }

    // MARK: - interval

    private func intervalBuffers(_ unit: ArrowIntervalUnit, _ vals: [ArrowInterval?]) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: vals.count * unit.byteWidth)
        for (i, v) in vals.enumerated() {
            guard let v else { continue }
            let base = i * unit.byteWidth
            func put32(_ x: Int32, _ off: Int) { withUnsafeBytes(of: x) { for (k, b) in $0.enumerated() { out[base + off + k] = b } } }
            func put64(_ x: Int64, _ off: Int) { withUnsafeBytes(of: x) { for (k, b) in $0.enumerated() { out[base + off + k] = b } } }
            switch unit {
            case .months: put32(v.months, 0)
            case .dayTime: put32(v.days, 0); put32(Int32(v.nanoseconds / 1_000_000), 4)
            case .monthDayNano: put32(v.months, 0); put32(v.days, 4); put64(v.nanoseconds, 8)
            }
        }
        return out
    }

    func testIntervalRoundTrip() throws {
        try requireRealGPU()
        let cases: [(ArrowIntervalUnit, [ArrowInterval?])] = [
            (.months, [ArrowInterval(months: 1), nil, ArrowInterval(months: -13), ArrowInterval()]),
            (.dayTime, [ArrowInterval(days: 3, nanoseconds: 2_000_000), nil, ArrowInterval(days: -4, nanoseconds: -1_000_000)]),
            (.monthDayNano, [ArrowInterval(months: 2, days: 3, nanoseconds: 123_456_789), nil,
                             ArrowInterval(months: -1, days: -2, nanoseconds: -7)]),
        ]
        for (unit, vals) in cases {
            let valid = vals.map { $0 != nil }
            var schema = cSchema(unit.arrowFormat, name: "iv")
            var arr = cArray(length: vals.count, nullCount: 1,
                             buffers: [bitmapBytes(valid), intervalBuffers(unit, vals)])
            let r = try importArrowArray(schema: &schema, array: &arr)
            let iv = try XCTUnwrap(r.array.asInterval)
            XCTAssertEqual(iv.unit, unit)
            XCTAssertEqual(iv.toArray(), vals)
            XCTAssertEqual(r.array.arrowFormat, unit.arrowFormat)

            var outSchema = ArrowSchema(); var outArray = ArrowArray()
            r.array.exportArrowSchema(name: "iv", into: &outSchema)
            r.array.exportArrowArray(into: &outArray)
            XCTAssertEqual(String(cString: outSchema.format), unit.arrowFormat)
            XCTAssertEqual(outArray.n_buffers, 2)
            let again = try importArrowArray(schema: &outSchema, array: &outArray)
            XCTAssertEqual(try XCTUnwrap(again.array.asInterval).toArray(), vals)
            outSchema.release?(&outSchema)
        }
    }

    func testIntervalSelection() throws {
        try requireRealGPU()
        for n in sizes {
            var vals: [ArrowInterval?] = []
            for i in 0..<n {
                if i % 4 == 2 { vals.append(nil); continue }
                vals.append(ArrowInterval(months: Int32(i % 30), days: Int32(i % 7), nanoseconds: Int64(i) * 13))
            }
            let iv = try MetalIntervalArray(unit: .monthDayNano, vals)
            let mask = try MetalBooleanArray((0..<n).map { $0 % 3 == 1 })
            XCTAssertEqual(try iv.filter(mask).toArray(), (0..<n).filter { $0 % 3 == 1 }.map { vals[$0] })
            let idx = (0..<Swift.min(n, 60)).map { Int32(n - 1 - $0) }
            XCTAssertEqual(try iv.take(try MetalArray<Int32>(idx)).toArray(), idx.map { vals[Int($0)] })
            if n > 2 { XCTAssertEqual(try iv.slice(offset: 1, length: n - 2).toArray(), Array(vals[1..<(n - 1)])) }
        }
    }

    func testAddIntervalMatchesCivilOracle() throws {
        try requireRealGPU()
        // timestamp[s] and timestamp[ns], plus date32, against the host civil-calendar oracle.
        for n in sizes where n > 0 {
            var ticks: [Int64?] = []
            var ivs: [ArrowInterval?] = []
            for i in 0..<n {
                ticks.append(i % 11 == 5 ? nil : Int64(i) * 86_401 - 1_000_000)
                if i % 13 == 7 { ivs.append(nil); continue }
                let months = Int32(i % 25) - 12
                let days = Int32(i % 9) - 4
                let nanos = Int64(i % 5) * 1_000_000_000
                ivs.append(ArrowInterval(months: months, days: days, nanoseconds: nanos))
            }
            let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), ticks)
            let iv = try MetalIntervalArray(unit: .monthDayNano, ivs)
            let got = try ts.addInterval(iv).toArray()
            for i in 0..<n {
                guard let t = ticks[i], let v = ivs[i] else { XCTAssertNil(got[i], "i = \(i)"); continue }
                let want = CivilOracle.addInterval(t, v, ticksPerDay: 86_400, subUnitPerSecond: 1_000_000_000,
                                                   ticksPerSecond: 1)
                XCTAssertEqual(got[i], want, "n = \(n) i = \(i)")
            }
        }
        // The documented clamping rule: 2024-01-31 + 1 month = 2024-02-29.
        let jan31 = CivilOracle.daysFromCivil(2024, 1, 31) * 86_400
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [jan31])
        let one = try MetalIntervalArray(unit: .months, [ArrowInterval(months: 1)])
        XCTAssertEqual(try ts.addInterval(one).toArray(), [CivilOracle.daysFromCivil(2024, 2, 29) * 86_400])
        // A month interval on date32 works; a sub-day interval on date32 is rejected.
        let d32 = try MetalTemporalArray(type: .date32, [CivilOracle.daysFromCivil(2024, 1, 31)])
        XCTAssertEqual(try d32.addInterval(one).toArray(), [CivilOracle.daysFromCivil(2024, 2, 29)])
        let sub = try MetalIntervalArray(unit: .monthDayNano, [ArrowInterval(nanoseconds: 1)])
        XCTAssertThrowsError(try d32.addInterval(sub))
    }

    func testIntervalBetweenMatchesOracle() throws {
        try requireRealGPU()
        // Arrow defines every field as the difference of the corresponding truncated field.
        for n in [1, 33, 4097] {
            var rng = SystemRandomNumberGenerator()
            let lo: Int64 = -2_208_988_800, hi: Int64 = 7_258_118_400        // 1900-01-01 .. 2200-01-01
            let a: [Int64?] = (0..<n).map { i in i % 9 == 4 ? nil : Int64.random(in: lo...hi, using: &rng) }
            let b: [Int64?] = (0..<n).map { i in i % 7 == 3 ? nil : Int64.random(in: lo...hi, using: &rng) }
            let ta = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), a)
            let tb = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), b)
            let mdn = try ta.monthDayNanoIntervalBetween(tb).toArray()
            let mo = try ta.monthIntervalBetween(tb).toArray()
            let dt = try ta.dayTimeIntervalBetween(tb).toArray()
            for i in 0..<n {
                guard let x = a[i], let y = b[i] else {
                    XCTAssertNil(mdn[i]); XCTAssertNil(mo[i]); XCTAssertNil(dt[i]); continue
                }
                let da = CivilOracle.floorDiv(x, 86_400), db = CivilOracle.floorDiv(y, 86_400)
                let ca = CivilOracle.civilFromDays(da), cb = CivilOracle.civilFromDays(db)
                let months = Int32((cb.y - ca.y) * 12 + (cb.m - ca.m))
                let ns = ((y - db * 86_400) - (x - da * 86_400)) * 1_000_000_000
                XCTAssertEqual(mdn[i], ArrowInterval(months: months, days: Int32(cb.d - ca.d), nanoseconds: ns), "i = \(i)")
                XCTAssertEqual(mo[i], ArrowInterval(months: months), "i = \(i)")
                XCTAssertEqual(dt[i], ArrowInterval(days: Int32(db - da), nanoseconds: (ns / 1_000_000) * 1_000_000), "i = \(i)")
            }
        }
    }

    // MARK: - fixed_size_binary

    func testFixedBinaryRoundTripAndCompare() throws {
        try requireRealGPU()
        let w = 4
        let values: [[UInt8]?] = [Array("abcd".utf8), nil, Array("efgh".utf8), Array("abcd".utf8)]
        let valid = values.map { $0 != nil }
        var flat = [UInt8](repeating: 0, count: values.count * w)
        for (i, v) in values.enumerated() { if let v { for (k, b) in v.enumerated() { flat[i * w + k] = b } } }
        var schema = cSchema("w:\(w)", name: "fb")
        var arr = cArray(length: values.count, nullCount: 1, buffers: [bitmapBytes(valid), flat])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let fb = try XCTUnwrap(r.array.asFixedBinary)
        XCTAssertEqual(fb.byteWidth, w)
        XCTAssertEqual(fb.toByteArrays(), values)
        XCTAssertEqual(r.array.arrowFormat, "w:4")

        let eq = try fb.compare(.eq, Array("abcd".utf8)).toArray()
        XCTAssertEqual(eq, [true, nil, false, true])
        let ne = try fb.compare(.ne, Array("abcd".utf8)).toArray()
        XCTAssertEqual(ne, [false, nil, true, false])
        XCTAssertEqual(try fb.compare(.eq, fb).toArray(), [true, nil, true, true])
        XCTAssertThrowsError(try fb.compare(.lt, Array("abcd".utf8)))
        let hashes = try fb.hash64().toArray()
        for (i, v) in values.enumerated() { XCTAssertEqual(hashes[i], v.map(fnv1a64), "i = \(i)") }

        var outSchema = ArrowSchema(); var outArray = ArrowArray()
        r.array.exportArrowSchema(name: "fb", into: &outSchema)
        r.array.exportArrowArray(into: &outArray)
        XCTAssertEqual(String(cString: outSchema.format), "w:4")
        let again = try importArrowArray(schema: &outSchema, array: &outArray)
        XCTAssertEqual(try XCTUnwrap(again.array.asFixedBinary).toByteArrays(), values)
        outSchema.release?(&outSchema)
    }

    func testFixedBinarySelection() throws {
        try requireRealGPU()
        let w = 5
        for n in sizes {
            let values: [[UInt8]?] = (0..<n).map { i in
                i % 6 == 4 ? nil : (0..<w).map { UInt8((i + $0) % 251) }
            }
            let fb = try MetalFixedBinaryArray(byteWidth: w, values)
            let mask = try MetalBooleanArray((0..<n).map { $0 % 4 == 3 })
            XCTAssertEqual(try fb.filter(mask).toByteArrays(), (0..<n).filter { $0 % 4 == 3 }.map { values[$0] })
            let idx = (0..<Swift.min(n, 70)).map { Int32(n - 1 - $0) }
            XCTAssertEqual(try fb.take(try MetalArray<Int32>(idx)).toByteArrays(), idx.map { values[Int($0)] })
            if n > 2 {
                XCTAssertEqual(try fb.slice(offset: 1, length: n - 2).toByteArrays(), Array(values[1..<(n - 1)]))
            }
            if n > 0 {
                let target = values[0]!
                let eq = try fb.compare(.eq, target).toArray()
                for i in 0..<n { XCTAssertEqual(eq[i], values[i].map { $0 == target }, "n = \(n) i = \(i)") }
            }
        }
    }

    // MARK: - list_view / large_list_view

    func testListViewImport() throws {
        try requireRealGPU()
        let child = primitiveCArray([1, 2, 3, 4].map { Int64?($0) }, zero: 0)
        // Contiguous: rows lie back to back, so the child is shared untouched.
        var schema = cSchema("+vl", name: "lv", children: [cSchema("l", name: "item")])
        var arr = cArray(length: 4, nullCount: 0,
                         buffers: [nil, rawBytes([Int32(0), 3, 3, 3]), rawBytes([Int32(3), 0, 0, 1])],
                         children: [child])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let l = try XCTUnwrap(r.array.asList)
        XCTAssertEqual(l.arrowFormat, "+l", "a list view exports as a plain list")
        let vals = try XCTUnwrap(l.values.asInt64).toArray()
        XCTAssertEqual((0..<l.length).map { i in l.valueRange(i).map { rg in rg.map { vals[$0] } } },
                       [[1, 2, 3], [], [], [4]])

        // Non-contiguous and out of order: the child is materialised with a GPU gather.
        let child2 = primitiveCArray([1, 2, 3, 4].map { Int64?($0) }, zero: 0)
        var schema2 = cSchema("+vl", name: "lv", children: [cSchema("l", name: "item")])
        var arr2 = cArray(length: 3, nullCount: 0,
                          buffers: [nil, rawBytes([Int32(3), 0, 1]), rawBytes([Int32(1), 2, 2])],
                          children: [child2])
        let r2 = try importArrowArray(schema: &schema2, array: &arr2)
        let l2 = try XCTUnwrap(r2.array.asList)
        let vals2 = try XCTUnwrap(l2.values.asInt64).toArray()
        XCTAssertEqual((0..<l2.length).map { i in l2.valueRange(i).map { rg in rg.map { vals2[$0] } } },
                       [[4], [1, 2], [2, 3]])

        // large_list_view narrows its int64 offsets and sizes.
        let child3 = primitiveCArray([1, 2, 3, 4].map { Int64?($0) }, zero: 0)
        var schema3 = cSchema("+vL", name: "lv", children: [cSchema("l", name: "item")])
        var arr3 = cArray(length: 2, nullCount: 1,
                          buffers: [bitmapBytes([true, false]), rawBytes([Int64(1), 0]), rawBytes([Int64(2), 9])],
                          children: [child3])
        let r3 = try importArrowArray(schema: &schema3, array: &arr3)
        let l3 = try XCTUnwrap(r3.array.asList)
        XCTAssertEqual(l3.nullCount, 1)
        let vals3 = try XCTUnwrap(l3.values.asInt64).toArray()
        XCTAssertEqual((0..<l3.length).map { i in l3.valueRange(i).map { rg in rg.map { vals3[$0] } } },
                       [[2, 3], nil])
    }

    // MARK: - list_parent_indices / list_slice

    /// A list<int64> built in Swift from Swift rows.
    private func buildList(_ rows: [[Int64]?]) throws -> MetalListArray {
        var flat: [Int64?] = []
        var counts: [Int?] = []
        for r in rows {
            if let r { counts.append(r.count); flat.append(contentsOf: r.map { Optional($0) }) }
            else { counts.append(nil) }
        }
        return try MetalListArray(counts: counts, values: .int64(try MetalArray<Int64>(flat)))
    }

    private func readRows(_ l: MetalListArray) throws -> [[Int64?]?] {
        let vals = try XCTUnwrap(l.values.asInt64).toArray()
        return (0..<l.length).map { i in l.valueRange(i).map { r in r.map { vals[$0] } } }
    }

    func testListParentIndicesAndSlice() throws {
        try requireRealGPU()
        for n in sizes {
            let rows: [[Int64]?] = (0..<n).map { i in
                i % 5 == 2 ? nil : (0..<(i % 4)).map { Int64(i * 10 + $0) }
            }
            let l = try buildList(rows)

            // list_parent_indices: the row covering each child element, in order.
            var expected: [Int32] = []
            for (i, r) in rows.enumerated() { for _ in 0..<(r?.count ?? 0) { expected.append(Int32(i)) } }
            XCTAssertEqual(try l.listParentIndices().toRawArray(), expected, "n = \(n)")

            // list_slice(1, 3): rows[1:3], nulls preserved.
            let sliced = try l.listSlice(start: 1, stop: 3)
            let wantSlice: [[Int64?]?] = rows.map { r in r.map { Array($0.dropFirst(1).prefix(2)).map { Optional($0) } } }
            XCTAssertEqual(try readRows(sliced), wantSlice, "n = \(n)")

            // list_slice(0, nil, 2): every other element.
            let strided = try l.listSlice(start: 0, stop: nil, step: 2)
            let wantStride: [[Int64?]?] = rows.map { r in
                r.map { row in stride(from: 0, to: row.count, by: 2).map { Optional(row[$0]) } }
            }
            XCTAssertEqual(try readRows(strided), wantStride, "n = \(n)")
        }
        let l = try buildList([[1, 2, 3]])
        XCTAssertThrowsError(try l.listSlice(start: -1))
        XCTAssertThrowsError(try l.listSlice(start: 0, step: 0))
    }

    // MARK: - map_lookup

    /// A map<utf8, int64> from Swift rows.
    private func buildMap(_ rows: [[(String, Int64)]?]) throws -> MetalMapArray {
        var keys: [String?] = [], items: [Int64?] = [], counts: [Int?] = []
        for r in rows {
            if let r { counts.append(r.count); for (k, v) in r { keys.append(k); items.append(v) } }
            else { counts.append(nil) }
        }
        let entries = try MetalStructArray(names: ["key", "value"],
                                           children: [.string(try MetalStringArray(keys)),
                                                      .int64(try MetalArray<Int64>(items))])
        let list = try MetalListArray(counts: counts, values: .structure(entries))
        return try MetalMapArray(entries: list)
    }

    func testMapLookupStringKeys() throws {
        try requireRealGPU()
        for n in sizes {
            let rows: [[(String, Int64)]?] = (0..<n).map { i -> [(String, Int64)]? in
                if i % 6 == 4 { return nil }
                var r: [(String, Int64)] = []
                if i % 3 != 1 { r.append(("a", Int64(i))) }
                r.append(("b", Int64(i) * 2))
                if i % 4 == 0 { r.append(("a", Int64(i) * 3)) }
                return r
            }
            let m = try buildMap(rows)
            let first = try m.mapLookup(.string("a"), occurrence: .first)
            let last = try m.mapLookup(.string("a"), occurrence: .last)
            let all = try m.mapLookup(.string("a"), occurrence: .all)
            let firstVals = try XCTUnwrap(first.asInt64).toArray()
            let lastVals = try XCTUnwrap(last.asInt64).toArray()
            let allRows = try readRows(try XCTUnwrap(all.asList))
            for (i, r) in rows.enumerated() {
                let hits = r?.filter { $0.0 == "a" }.map { $0.1 } ?? []
                XCTAssertEqual(firstVals[i], hits.first, "n = \(n) i = \(i)")
                XCTAssertEqual(lastVals[i], hits.last, "n = \(n) i = \(i)")
                XCTAssertEqual(allRows[i], hits.isEmpty ? nil : hits.map { Optional($0) }, "n = \(n) i = \(i)")
            }
            // A key nobody has is null everywhere.
            let miss = try XCTUnwrap(try m.mapLookup(.string("zz")).asInt64).toArray()
            XCTAssertEqual(miss.compactMap { $0 }.count, 0)
        }
    }

    func testMapLookupIntegerKeys() throws {
        try requireRealGPU()
        let counts: [Int?] = [3, nil, 0, 1]
        let keys: [Int32?] = [1, 2, 1, 7]
        let items: [Int64?] = [10, 20, 30, 70]
        let entries = try MetalStructArray(names: ["key", "value"],
                                           children: [.int32(try MetalArray<Int32>(keys)),
                                                      .int64(try MetalArray<Int64>(items))])
        let m = try MetalMapArray(entries: try MetalListArray(counts: counts, values: .structure(entries)))
        XCTAssertEqual(try XCTUnwrap(try m.mapLookup(.integer(1), occurrence: .first).asInt64).toArray(),
                       [10, nil, nil, nil])
        XCTAssertEqual(try XCTUnwrap(try m.mapLookup(.integer(1), occurrence: .last).asInt64).toArray(),
                       [30, nil, nil, nil])
        let all = try readRows(try XCTUnwrap(try m.mapLookup(.integer(1), occurrence: .all).asList))
        XCTAssertEqual(all, [[10, 30], nil, nil, nil])
        XCTAssertEqual(try XCTUnwrap(try m.mapLookup(.integer(7)).asInt64).toArray(), [nil, nil, nil, 70])
        // The wrong key kind for the map's key type is an error, not a silent miss.
        XCTAssertThrowsError(try m.mapLookup(.string("1")))
    }

    // MARK: - extension types

    func testExtensionTypeImportExport() throws {
        try requireRealGPU()
        let w = 16
        let values: [[UInt8]?] = [(0..<w).map { UInt8($0) }, nil, (0..<w).map { UInt8(255 - $0) }]
        var flat = [UInt8](repeating: 0, count: values.count * w)
        for (i, v) in values.enumerated() { if let v { for (k, b) in v.enumerated() { flat[i * w + k] = b } } }
        var blob = metaBlob([("ARROW:extension:name", "arrow.uuid"),
                             ("ARROW:extension:metadata", ""),
                             ("PARQUET:field_id", "17")])
        try blob.withUnsafeMutableBufferPointer { p in
            var schema = cSchema("w:\(w)", name: "u")
            schema.metadata = UnsafeRawPointer(p.baseAddress!).assumingMemoryBound(to: CChar.self)
            var arr = cArray(length: values.count, nullCount: 1, buffers: [bitmapBytes(values.map { $0 != nil }), flat])
            let r = try importArrowArray(schema: &schema, array: &arr)
            XCTAssertEqual(r.array.extensionName, "arrow.uuid")
            XCTAssertEqual(r.array.extensionMetadata, [])
            XCTAssertEqual(r.array.arrowFormat, "w:16", "the format is the storage type's")
            XCTAssertEqual(try XCTUnwrap(r.array.storageArray.asFixedBinary).toByteArrays(), values)
            // Every unrelated metadata key survives.
            let ext = try XCTUnwrap(r.array.asExtension)
            XCTAssertEqual(ext.otherMetadata.string("PARQUET:field_id"), "17")

            // Selection keeps the extension tag.
            let kept = try r.array.filter(try MetalBooleanArray([true, false, true]))
            XCTAssertEqual(kept.extensionName, "arrow.uuid")
            XCTAssertEqual(try XCTUnwrap(kept.storageArray.asFixedBinary).toByteArrays(), [values[0], values[2]])

            // Export writes the metadata back, and a re-import sees the same extension type.
            var outSchema = ArrowSchema(); var outArray = ArrowArray()
            r.array.exportArrowSchema(name: "u", into: &outSchema)
            r.array.exportArrowArray(into: &outArray)
            XCTAssertEqual(String(cString: outSchema.format), "w:16")
            let decoded = ArrowSchemaMetadata.decode(outSchema.metadata)
            XCTAssertEqual(decoded.string("ARROW:extension:name"), "arrow.uuid")
            XCTAssertEqual(decoded.string("PARQUET:field_id"), "17")
            let again = try importArrowArray(schema: &outSchema, array: &outArray)
            XCTAssertEqual(again.array.extensionName, "arrow.uuid")
            XCTAssertEqual(try XCTUnwrap(again.array.storageArray.asFixedBinary).toByteArrays(), values)
            outSchema.release?(&outSchema)
            schema.metadata = nil
        }
    }

    func testMetadataBlobEncoding() throws {
        // The C Data Interface blob: int32 count, then length-prefixed key/value pairs, native endian.
        let pairs = [("k1", "v1"), ("", ""), ("ARROW:extension:name", "x.y")]
        var m = ArrowSchemaMetadata(pairs.map { ArrowSchemaMetadata.Pair(key: $0.0, string: $0.1) })
        let encoded = m.encoded()
        XCTAssertEqual(encoded, metaBlob(pairs))
        encoded.withUnsafeBufferPointer { p in
            let decoded = ArrowSchemaMetadata.decode(UnsafeRawPointer(p.baseAddress!).assumingMemoryBound(to: CChar.self))
            XCTAssertEqual(decoded.pairs.map { $0.key }, pairs.map { $0.0 })
            XCTAssertEqual(decoded.pairs.map { $0.stringValue }, pairs.map { $0.1 })
        }
        m["k1"] = nil
        XCTAssertEqual(m.pairs.count, 2)
        XCTAssertNil(ArrowSchemaMetadata.decode(nil).pairs.first)
    }

    // MARK: - timezones

    func testAssumeTimezoneAndLocalTimestampRoundTrip() throws {
        let tz = "America/New_York"
        let zone = try resolveTimeZone(tz)
        for unit in [ArrowTemporalUnit.second, .milli, .micro, .nano] {
            let per = unit.perSecond
            // Noon UTC on 400 consecutive days: never inside a DST transition in New York.
            let base: Int64 = 1_672_574_400                                   // 2023-01-01T12:00:00Z
            let instants: [Int64?] = (0..<400).map { (i: Int) -> Int64? in
                if i % 17 == 3 { return nil }
                let day: Int64 = base + Int64(i) * 86_400
                return day * per
            }
            let ts = try MetalTemporalArray(type: .timestamp(unit, timezone: tz), instants)
            let local = try ts.localTimestamp()
            XCTAssertEqual(local.type, .timestamp(unit, timezone: nil))
            let localVals = local.toArray()
            for (i, v) in instants.enumerated() {
                guard let v else { XCTAssertNil(localVals[i]); continue }
                let seconds = v / per
                let offset = Int64(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(seconds))))
                XCTAssertEqual(localVals[i], (seconds + offset) * per, "unit \(unit) i = \(i)")
            }
            // Round trip: the same instants come back.
            let again = try local.assumeTimezone(tz)
            XCTAssertEqual(again.toArray(), instants)
            XCTAssertEqual(again.type.timezone, tz)
        }
    }

    func testAssumeTimezoneAmbiguousAndNonexistent() throws {
        let tz = "America/New_York"
        // 2023-11-05 01:30 local happens twice (EDT then EST).
        let ambiguous = CivilOracle.daysFromCivil(2023, 11, 5) * 86_400 + 5_400
        let a = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [ambiguous])
        XCTAssertThrowsError(try a.assumeTimezone(tz)) { XCTAssertTrue("\($0)".contains("ambiguous")) }
        let early = try XCTUnwrap(try a.assumeTimezone(tz, ambiguous: .earliest).toArray()[0])
        let late = try XCTUnwrap(try a.assumeTimezone(tz, ambiguous: .latest).toArray()[0])
        XCTAssertEqual(late - early, 3_600, "the fall-back repeats one hour")
        // Both name the same wall clock.
        for v in [early, late] {
            let back = try MetalTemporalArray(type: .timestamp(.second, timezone: tz), [v]).localTimestamp()
            XCTAssertEqual(back.toArray(), [ambiguous])
        }

        // 2023-03-12 02:30 local never happens (the spring-forward gap).
        let gap = CivilOracle.daysFromCivil(2023, 3, 12) * 86_400 + 9_000
        let g = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [gap])
        XCTAssertThrowsError(try g.assumeTimezone(tz)) { XCTAssertTrue("\($0)".contains("does not exist")) }
        let before = try XCTUnwrap(try g.assumeTimezone(tz, nonexistent: .earliest).toArray()[0])
        let after = try XCTUnwrap(try g.assumeTimezone(tz, nonexistent: .latest).toArray()[0])
        XCTAssertEqual(after - before, 1, "earliest is the last instant before the gap, latest the first after")

        // Nulls stay null and a fixed-offset zone works.
        let n = try MetalTemporalArray(type: .timestamp(.second, timezone: nil), [nil, 0])
        XCTAssertEqual(try n.assumeTimezone("+02:00").toArray(), [nil, -7_200])
        XCTAssertEqual(try n.assumeTimezone("UTC").toArray(), [nil, 0])
        // The wrong input type is rejected.
        let tagged = try MetalTemporalArray(type: .timestamp(.second, timezone: "UTC"), [0])
        XCTAssertThrowsError(try tagged.assumeTimezone(tz))
    }

    func testLocalTimestampLargeAndSharded() throws {
        let tz = "Europe/Berlin"
        let zone = try resolveTimeZone(tz)
        let n = 200_003
        let instants: [Int64?] = (0..<n).map { $0 % 1_009 == 7 ? nil : Int64($0) * 907 }
        let ts = try MetalTemporalArray(type: .timestamp(.second, timezone: tz), instants)
        let got = try ts.localTimestamp().toArray()
        for i in stride(from: 0, to: n, by: 97) {
            guard let v = instants[i] else { XCTAssertNil(got[i]); continue }
            let o = Int64(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(v))))
            XCTAssertEqual(got[i], v + o, "i = \(i)")
        }
    }

    /// The GPU transition table against Foundation itself, over 50k random instants from 1900 to 2100
    /// in every resolution, for the six zones that between them cover northern and southern DST, a
    /// half-hour offset, a zone that abandoned DST and one with no transitions at all.
    func testTimezoneTableMatchesFoundationForEveryZone() throws {
        try requireRealGPU()
        let zones = ["America/New_York", "Europe/Berlin", "Australia/Sydney", "Asia/Kolkata",
                     "America/Sao_Paulo", "UTC"]
        let n = 50_000
        var g = SystemRandomNumberGenerator()
        _ = g
        var state: UInt64 = 0x2026_0906
        func nextSecond() -> Int64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let lo: Int64 = -2_208_988_800, hi: Int64 = 4_102_444_800
            return lo + Int64((state >> 11) % UInt64(hi - lo))
        }
        let seconds = (0..<n).map { _ in nextSecond() }
        for name in zones {
            let zone = try resolveTimeZone(name)
            for unit in [ArrowTemporalUnit.second, .milli, .micro, .nano] {
                let per = unit.perSecond
                let values: [Int64?] = (0..<n).map { $0 % 613 == 11 ? nil : seconds[$0] * per }
                let aware = try MetalTemporalArray(type: .timestamp(unit, timezone: name), values)
                let local = try aware.localTimestamp().toArray()
                let offsets = try aware.utcOffset().toArray()
                let dst = try aware.isDST().toArray()
                for i in stride(from: 0, to: n, by: 11) {
                    guard values[i] != nil else {
                        XCTAssertNil(local[i]); XCTAssertNil(dst[i]); continue
                    }
                    let d = Date(timeIntervalSince1970: Double(seconds[i]))
                    let o = Int64(zone.secondsFromGMT(for: d))
                    XCTAssertEqual(local[i], (seconds[i] + o) * per, "\(name) \(unit) row \(i)")
                    XCTAssertEqual(offsets[i], Int32(o), "\(name) \(unit) offset row \(i)")
                    XCTAssertEqual(dst[i], zone.isDaylightSavingTime(for: d), "\(name) \(unit) dst row \(i)")
                }
                // assume_timezone against a Foundation oracle, with both policies pinned so that
                // ambiguous and nonexistent local times still produce a value.
                let naive = try MetalTemporalArray(type: .timestamp(unit, timezone: nil), values)
                let early = try naive.assumeTimezone(name, ambiguous: .earliest, nonexistent: .earliest)
                    .toArray()
                let late = try naive.assumeTimezone(name, ambiguous: .latest, nonexistent: .latest).toArray()
                for i in stride(from: 0, to: n, by: 101) {
                    guard values[i] != nil else { XCTAssertNil(early[i]); continue }
                    XCTAssertEqual(early[i], Self.expectedAssume(seconds[i], zone, per: per, earliest: true),
                                   "\(name) \(unit) earliest row \(i) local \(seconds[i])")
                    XCTAssertEqual(late[i], Self.expectedAssume(seconds[i], zone, per: per, earliest: false),
                                   "\(name) \(unit) latest row \(i) local \(seconds[i])")
                }
            }
        }
    }

    /// Every ambiguous and nonexistent minute around every DST change from 2000 to 2037, against the
    /// same Foundation oracle and under all four policy combinations.
    func testEveryDSTEdgeMinuteFrom2000To2037() throws {
        try requireRealGPU()
        for name in ["America/New_York", "Europe/Berlin", "Australia/Sydney", "America/Sao_Paulo"] {
            let zone = try resolveTimeZone(name)
            var edges: [Int64] = []
            var cursor = Date(timeIntervalSince1970: 946_684_800)                   // 2000-01-01
            let end = Date(timeIntervalSince1970: 2_145_916_800)                    // 2038-01-01
            while let next = zone.nextDaylightSavingTimeTransition(after: cursor), next < end {
                edges.append(Int64(next.timeIntervalSince1970.rounded(.down)))
                cursor = next
            }
            XCTAssertFalse(edges.isEmpty, name)
            // The wall clocks around each change, minute by minute. The window has to be centred on the
            // *local* time of the transition — in New York that is five hours from the UTC instant —
            // and to span both offsets, since the repeated hour and the skipped hour live between them.
            func offset(at t: Int64) -> Int64 {
                Int64(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(t))))
            }
            var locals: [Int64] = []
            for e in edges {
                let a = offset(at: e - 1), b = offset(at: e)
                locals += stride(from: e + Swift.min(a, b) - 3_600, to: e + Swift.max(a, b) + 3_600,
                                 by: 60).map { $0 }
            }
            let naive = try MetalTemporalArray(type: .timestamp(.second, timezone: nil),
                                               locals.map { Optional($0) })
            for earliest in [true, false] {
                let amb: ArrowAmbiguousHandling = earliest ? .earliest : .latest
                let non: ArrowNonexistentHandling = earliest ? .earliest : .latest
                let got = try naive.assumeTimezone(name, ambiguous: amb, nonexistent: non).toArray()
                for (i, l) in locals.enumerated() {
                    XCTAssertEqual(got[i], Self.expectedAssume(l, zone, per: 1, earliest: earliest),
                                   "\(name) earliest=\(earliest) local \(l)")
                    if got[i] != Self.expectedAssume(l, zone, per: 1, earliest: earliest) { return }
                }
            }
            // And `raise` really does raise somewhere in that set.
            XCTAssertThrowsError(try naive.assumeTimezone(name), name)
        }
    }

    /// Foundation's answer to `assume_timezone` for one wall clock: every UTC offset `o` for which
    /// `local - o` really is at offset `o`. Two of them is an ambiguous time and none is a gap, whose
    /// `earliest` / `latest` are the last instant before and the first instant after the transition.
    private static func expectedAssume(_ local: Int64, _ zone: TimeZone, per: Int64,
                                       earliest: Bool) -> Int64? {
        func offset(at t: Int64) -> Int64 {
            Int64(zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(t))))
        }
        var candidates: [Int64] = []
        for probe in [local, local - offset(at: local), local - 86_400, local + 86_400] {
            let o = offset(at: probe)
            if !candidates.contains(o) { candidates.append(o) }
        }
        let valid = candidates.filter { offset(at: local - $0) == $0 }
        if valid.count == 1 { return (local - valid[0]) * per }
        if valid.count >= 2 {
            let o = earliest ? valid.max()! : valid.min()!
            return (local - o) * per
        }
        // A gap: find the transition that opened it.
        let after = candidates.max()!
        guard let next = zone.nextDaylightSavingTimeTransition(
            after: Date(timeIntervalSince1970: Double(local - after - 1))) else { return nil }
        let t = Int64(next.timeIntervalSince1970.rounded(.down))
        return earliest ? t * per - 1 : t * per
    }

    // MARK: - record batch integration

    func testRecordBatchCarriesEveryNewType() throws {
        try requireRealGPU()
        let n = 33
        let cols: [AnyMetalArray] = [
            .null(MetalNullArray(length: n)),
            .float16(try MetalFloat16Array((0..<n).map { Float($0) })),
            .smallDecimal(try MetalSmallDecimalArray(type: try ArrowSmallDecimalType(precision: 9, scale: 2, bitWidth: 32),
                                                     (0..<n).map { Int64($0) })),
            .interval(try MetalIntervalArray(unit: .monthDayNano, (0..<n).map { ArrowInterval(months: Int32($0)) })),
            .fixedBinary(try MetalFixedBinaryArray(byteWidth: 3, (0..<n).map { i in [UInt8(i % 251), 0, 1] })),
        ]
        let batch = try MetalRecordBatch(names: ["n", "h", "d", "iv", "fb"], columns: cols)
        let mask = try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })
        let kept = try batch.filter(mask)
        XCTAssertEqual(kept.length, (n + 1) / 2)
        for c in kept.columns { XCTAssertEqual(c.length, (n + 1) / 2) }
        let sliced = try batch.slice(offset: 3, length: 10)
        for c in sliced.columns { XCTAssertEqual(c.length, 10) }
        let taken = try batch.take(try MetalArray<Int32>([0, 5, 32]))
        for c in taken.columns { XCTAssertEqual(c.length, 3) }
        // The whole batch survives a C Data Interface round trip.
        var outSchema = ArrowSchema(); var outArray = ArrowArray()
        batch.exportArrowSchema(name: "b", into: &outSchema)
        batch.exportArrowArray(into: &outArray)
        let back = try importArrowRecordBatch(schema: &outSchema, array: &outArray)
        XCTAssertEqual(back.batch.names, batch.names)
        XCTAssertEqual(back.batch.columns.map { $0.arrowFormat }, batch.columns.map { $0.arrowFormat })
        outSchema.release?(&outSchema)
    }
}
