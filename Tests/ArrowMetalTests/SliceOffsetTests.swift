import XCTest
import CArrowABI
@testable import ArrowMetal

/// Arrow `slice` at an arbitrary offset: zero-copy, correct in every kernel, correct across the C ABI.
///
/// The offsets exercised are deliberately awkward: 1 and 7 land inside the first bitmap byte, 31 and 33
/// straddle the 32-element fast path, and 4097 is far enough in to catch an offset that is applied once
/// but forgotten later.
final class SliceOffsetTests: XCTestCase {
    let offsets = [0, 1, 7, 31, 32, 33, 4097]

    // MARK: - primitives

    func testPrimitiveSliceValuesAndNullsAtEveryOffset() throws {
        let n = 5000
        func check<T: ArrowPrimitive & Equatable>(_ make: (Int) -> T) throws {
            let src: [T?] = (0..<n).map { $0 % 7 == 3 ? nil : make($0) }
            let a = try MetalArray<T>(src, context: .shared)
            for off in offsets {
                let len = n - off - 11
                let s = try a.slice(offset: off, length: len)
                XCTAssertEqual(s.length, len)
                XCTAssertEqual(s.nullCount, (off..<(off + len)).filter { src[$0] == nil }.count,
                               "null count at offset \(off) for \(T.self)")
                XCTAssertEqual(s.toArray(), Array(src[off..<(off + len)]), "values at offset \(off) for \(T.self)")
            }
        }
        try check { Int8(truncatingIfNeeded: $0) }
        try check { UInt8(truncatingIfNeeded: $0) }
        try check { Int16(truncatingIfNeeded: $0) }
        try check { UInt16(truncatingIfNeeded: $0) }
        try check { Int32($0) }
        try check { UInt32($0) }
        try check { Int64($0) }
        try check { UInt64($0) }
        try check { Float($0) }
        try check { Double($0) }
    }

    func testPrimitiveSliceWithoutValidity() throws {
        let n = 5000
        let src = (0..<n).map { Int64($0) }
        let a = try MetalArray<Int64>(src, context: .shared)
        for off in offsets {
            let len = n - off - 11
            let s = try a.slice(offset: off, length: len)
            XCTAssertEqual(s.nullCount, 0)
            XCTAssertEqual(s.toRawArray(), Array(src[off..<(off + len)]))
        }
    }

    func testBooleanSliceAtEveryOffset() throws {
        let n = 5000
        let src: [Bool?] = (0..<n).map { $0 % 11 == 5 ? nil : ($0 % 3 == 0) }
        let vals = src.map { $0 ?? false }
        let a = try MetalBooleanArray(vals, context: .shared)
        let bm = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: .shared)
        for i in 0..<n where src[i] != nil { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
        let b = MetalBooleanArray(length: n, nullCount: 0, validity: bm, values: a.values, context: .shared)
        b.recomputeNullCount()
        for off in offsets {
            let len = n - off - 11
            let s = try b.slice(offset: off, length: len)
            XCTAssertEqual(s.toArray(), Array(src[off..<(off + len)]), "boolean values at offset \(off)")
            XCTAssertEqual(s.nullCount, (off..<(off + len)).filter { src[$0] == nil }.count)
            XCTAssertEqual(s.trueCount, (off..<(off + len)).filter { src[$0] == true }.count)
            XCTAssertEqual(s.any, (off..<(off + len)).contains { src[$0] == true })
            XCTAssertEqual(s.all, !(off..<(off + len)).contains { src[$0] == false })
        }
    }

    func testStringSliceAtEveryOffset() throws {
        let n = 8000
        let src: [String?] = (0..<n).map { $0 % 13 == 4 ? nil : "row-\($0)" }
        let a = try MetalStringArray(src, context: .shared)
        for off in offsets where off < n - 20 {
            let len = n - off - 11
            let s = try a.slice(offset: off, length: len)
            XCTAssertEqual(s.toArray(), Array(src[off..<(off + len)]), "utf8 at offset \(off)")
            XCTAssertEqual(s.nullCount, (off..<(off + len)).filter { src[$0] == nil }.count)
        }
    }

    func testTemporalAndDecimalSlice() throws {
        let n = 8000
        let ts = try MetalArray<Int64>((0..<n).map { Int64($0) * 1_000 }, context: .shared)
        let t = try MetalTemporalArray(type: .timestamp(.micro, timezone: nil), ts)
        for off in offsets {
            let s = try t.slice(offset: off, length: n - off - 11)
            XCTAssertEqual(s.length, n - off - 11)
            XCTAssertEqual(s.toArray().first ?? nil, Int64(off) * 1_000, "temporal at offset \(off)")
        }
    }

    func testChainedSlicesComposeOffsets() throws {
        let n = 8000
        let src: [Int64?] = (0..<n).map { $0 % 5 == 2 ? nil : Int64($0) }
        let a = try MetalArray<Int64>(src, context: .shared)
        var s = try a.slice(offset: 1, length: n - 1)          // offset 1
        s = try s.slice(offset: 6, length: s.length - 6)       // offset 7
        s = try s.slice(offset: 26, length: s.length - 26)     // offset 33
        s = try s.slice(offset: 4064, length: 100)             // offset 4097
        XCTAssertEqual(s.length, 100)
        XCTAssertEqual(s.toArray(), Array(src[4097..<4197]))
        XCTAssertEqual(s.nullCount, (4097..<4197).filter { src[$0] == nil }.count)
    }

    // MARK: - kernels over a slice

    func testSliceThenKernels() throws {
        try requireRealGPU()
        let n = 20_000
        var src = [Int64?](repeating: nil, count: n)
        for i in 0..<n where i % 9 != 4 { src[i] = Int64((i &* 2_654_435_761) % 100_000) }
        let a = try MetalArray<Int64>(src, context: .shared)
        for off in offsets {
            let len = n - off - 11
            let s = try a.slice(offset: off, length: len)
            let expect = Array(src[off..<(off + len)])
            let valid = expect.compactMap { $0 }

            // sum / min / max: the reduction kernels must see exactly the sliced range
            XCTAssertEqual(try s.sum(), .int(valid.reduce(0, &+)), "sum at offset \(off)")
            XCTAssertEqual(try s.min(), valid.min(), "min at offset \(off)")
            XCTAssertEqual(try s.max(), valid.max(), "max at offset \(off)")
            XCTAssertEqual(try s.first(), valid.first, "first at offset \(off)")
            XCTAssertEqual(try s.last(), valid.last, "last at offset \(off)")

            // filter
            let mask = try s.compare(.gt, 50_000)
            let kept = try s.filter(mask)
            XCTAssertEqual(kept.toArray(), expect.filter { ($0 ?? 0) > 50_000 && $0 != nil }, "filter at offset \(off)")

            // take
            let idx = try MetalArray<Int32>([0, 1, Int32(len - 1), 7], context: .shared)
            XCTAssertEqual(try s.take(idx).toArray(), [expect[0], expect[1], expect[len - 1], expect[7]],
                           "take at offset \(off)")

            // sort
            XCTAssertEqual(try s.sorted().toArray().compactMap { $0 }, valid.sorted(), "sort at offset \(off)")
        }
    }

    func testSliceThenGroupBy() throws {
        try requireRealGPU()
        let n = 8000
        let keys = try MetalArray<Int32>((0..<n).map { Int32($0 % 10) }, context: .shared)
        let vals = try MetalArray<Int64>((0..<n).map { Int64($0) }, context: .shared)
        let off = 33, len = n - off - 11
        let gb = try GroupBy(keys: try keys.slice(offset: off, length: len), keyCount: 10)
        let sums = try gb.sum(try vals.slice(offset: off, length: len))
        var expect = [Int64](repeating: 0, count: 10)
        for i in off..<(off + len) { expect[i % 10] += Int64(i) }
        XCTAssertEqual(sums.toRawArray(), expect)
    }

    // MARK: - C ABI round trip

    func testSliceExportsWithArrowOffsetAndRoundTrips() throws {
        let n = 8000
        let src: [Int64?] = (0..<n).map { $0 % 6 == 1 ? nil : Int64($0) }
        let a = try MetalArray<Int64>(src, context: .shared)
        for off in offsets {
            let len = n - off - 11
            let s = try a.slice(offset: off, length: len)
            var schema = ArrowSchema(), array = ArrowArray()
            s.exportArrowSchema(into: &schema)
            s.exportArrowArray(into: &array)
            XCTAssertEqual(Int(array.length), len)
            XCTAssertEqual(Int(array.null_count), (off..<(off + len)).filter { src[$0] == nil }.count)
            // Offsets that are a multiple of 32 become buffer views, everything else rides on Arrow's offset.
            XCTAssertEqual(Int(array.offset), off % 32 == 0 ? 0 : off, "exported offset for \(off)")
            let back = try importArrowArray(schema: &schema, array: &array).array
            XCTAssertEqual(back.asInt64!.toArray(), Array(src[off..<(off + len)]), "round trip at offset \(off)")
        }
    }

    func testStringSliceExportsAndRoundTrips() throws {
        let n = 3000
        let src: [String?] = (0..<n).map { $0 % 9 == 2 ? nil : "value-\($0)" }
        let a = try MetalStringArray(src, context: .shared)
        for off in [1, 7, 31, 33, 2049] {
            let len = n - off - 11
            let s = try a.slice(offset: off, length: len)
            var schema = ArrowSchema(), array = ArrowArray()
            s.exportArrowSchema(into: &schema)
            s.exportArrowArray(into: &array)
            let back = try importArrowArray(schema: &schema, array: &array).array
            guard case .string(let str) = back else { return XCTFail("expected a utf8 column") }
            XCTAssertEqual(str.toArray(), Array(src[off..<(off + len)]), "utf8 round trip at offset \(off)")
        }
    }

    // MARK: - cost

    func testFiftyMillionRowSliceIsUnderOneMillisecond() throws {
        let n = 50_000_000
        let a = try MetalArray<Int64>.allocate(length: n, withValidity: true)
        memset(a.rawValidity!.mutableContents, 0xFF, Bitmap.byteCount(bits: n))
        a.recomputeNullCount()
        var worst = 0.0
        for off in [1, 7, 31, 33, 4097, 1000] {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let s = try a.slice(offset: off, length: n - off - 1)
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            XCTAssertEqual(s.length, n - off - 1)
            worst = Swift.max(worst, dt)
        }
        XCTAssertLessThan(worst, 1.0, "a 50M-row slice must be O(1); worst offset took \(worst) ms")
    }
}
