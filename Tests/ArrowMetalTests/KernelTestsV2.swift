import XCTest
import CArrowABI
@testable import ArrowMetal

final class TakeCastSliceTests: XCTestCase {
    func testTakeWithNullsAndNullIndices() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 1000, 100_003] {
            var srcVals: [Int64?] = []
            for i in 0..<n { srcVals.append(i % 7 == 0 ? nil : Int64(i)) }
            let src = try MetalArray<Int64>(srcVals)
            var idxVals: [Int32?] = []
            let modulus = max(n, 1)
            for i in 0..<(n / 2 + 3) {
                if i % 5 == 0 { idxVals.append(nil) } else { idxVals.append(Int32((i * 31) % modulus)) }
            }
            guard n > 0 else {
                let e = try MetalArray<Int32>([Int32]())
                XCTAssertEqual(try src.take(e).length, 0); continue
            }
            let idx = try MetalArray<Int32>(idxVals)
            let g = try src.take(idx)
            var expected: [Int64?] = []
            for iv in idxVals { if let iv { expected.append(src[Int(iv)]) } else { expected.append(nil) } }
            XCTAssertEqual(g.toArray(), expected, "n=\(n)")
            XCTAssertEqual(g.nullCount, expected.filter { $0 == nil }.count)
            // Int64 indices, no nulls anywhere
            let ascending: [Int64] = (0..<n).map { Int64($0) }
            let descending: [Int64] = ascending.reversed()
            let plain = try MetalArray<Int64>(ascending)
            let idx64 = try MetalArray<Int64>(descending)
            XCTAssertEqual(try plain.take(idx64).toRawArray(), descending)
            XCTAssertNil(try plain.take(idx64).validity)
        }
    }

    func testTakeOutOfRangeThrows() throws {
        try requireRealGPU()
        let src = try MetalArray<Int32>([1, 2, 3])
        XCTAssertThrowsError(try src.take(try MetalArray<Int32>([0, 3])))
        XCTAssertThrowsError(try src.take(try MetalArray<Int64>([-1])))
    }

    func testTakeFloat64AndBoolean() throws {
        try requireRealGPU()
        let d = try MetalArray<Double>([1.5, nil, -0.0, .nan, 4])
        let r = try d.take(try MetalArray<Int32>([4, 3, 1, 0]))
        XCTAssertEqual(r[0], 4); XCTAssertTrue(r[1]!.isNaN); XCTAssertNil(r[2]); XCTAssertEqual(r[3], 1.5)
        let b = try MetalBooleanArray([true, false, true, true, false])
        XCTAssertEqual(try b.take(try MetalArray<Int32>([4, 0, 2])).toArray(), [false, true, true])
    }

    func testCast() throws {
        try requireRealGPU()
        let i = try MetalArray<Int32>([-5, nil, 300, 7])
        XCTAssertEqual(try i.cast(to: Int8.self).toArray(), [-5, nil, 44, 7])       // 300 wraps
        XCTAssertEqual(try i.cast(to: Float.self).toArray(), [-5, nil, 300, 7])
        XCTAssertEqual(try i.cast(to: Double.self).toArray(), [-5, nil, 300, 7])    // CPU path
        XCTAssertEqual(try i.cast(to: UInt64.self).toArray(), [UInt64(bitPattern: -5), nil, 300, 7])
        let f = try MetalArray<Float>([1.9, -1.9, 2.5, nil])
        XCTAssertEqual(try f.cast(to: Int32.self).toArray(), [1, -1, 2, nil])
        let d = try MetalArray<Double>([1.9, -1.9, 1e10])
        XCTAssertEqual(try d.cast(to: Int64.self).toArray(), [1, -1, 10_000_000_000])
        XCTAssertEqual(try d.cast(to: Float.self).toArray(), [1.9, -1.9, 1e10])
        XCTAssertTrue(try i.cast(to: Int32.self) === i)
        // large
        let big = try MetalArray<Int64>((0..<200_000).map { Int64($0) * 1000 })
        XCTAssertEqual(try big.cast(to: Int32.self).toRawArray(), (0..<200_000).map { Int32(truncatingIfNeeded: $0 * 1000) })
    }

    func testSliceAlignedIsZeroCopyAndUnalignedCopies() throws {
        try requireRealGPU()
        let n = 1000
        var aVals: [Int32?] = []
        for i in 0..<n { aVals.append(i % 3 == 0 ? nil : Int32(i)) }
        let a = try MetalArray<Int32>(aVals)
        let s1 = try a.slice(offset: 64, length: 100)
        XCTAssertTrue(s1.values.mtl === a.values.mtl, "aligned slice shares the buffer")
        XCTAssertEqual(s1.toArray(), Array(a.toArray()[64..<164]))
        XCTAssertEqual(try s1.sum(), CPUReference.sum(s1))
        XCTAssertEqual(try s1.filter(try s1.compare(.gt, 100)).toArray(), Array(a.toArray()[64..<164]).filter { ($0 ?? 0) > 100 })
        let s2 = try a.slice(offset: 5, length: 100)
        XCTAssertFalse(s2.values.mtl === a.values.mtl)
        XCTAssertEqual(s2.toArray(), Array(a.toArray()[5..<105]))
        XCTAssertEqual(s2.nullCount, Array(a.toArray()[5..<105]).filter { $0 == nil }.count)
        let b = try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })
        XCTAssertEqual(try b.slice(offset: 32, length: 10).toArray(), (32..<42).map { $0 % 2 == 0 })
        XCTAssertEqual(try b.slice(offset: 3, length: 10).toArray(), (3..<13).map { $0 % 2 == 0 })
        // export a slice through the C interface and re-import
        var schema = ArrowSchema(); var arr = ArrowArray()
        s1.exportArrowSchema(into: &schema); s1.exportArrowArray(into: &arr)
        let r = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertEqual(r.array.asInt32?.toArray(), s1.toArray())
        schema.release?(&schema)
    }
}

final class Float64AndBooleanTests: XCTestCase {
    func testFloat64CompareMinMaxFilterOnGPU() throws {
        try requireRealGPU()
        var g = SystemRandomNumberGenerator()
        for n in [0, 1, 31, 32, 33, 4097, 300_000] {
            var vals: [Double?] = []
            for _ in 0..<n { vals.append(Double.random(in: -1000...1000, using: &g)) }
            for i in stride(from: 0, to: n, by: 9) { vals[i] = nil }
            if n > 20 { vals[3] = .nan; vals[4] = -0.0; vals[5] = 0.0; vals[6] = .infinity; vals[7] = -.infinity }
            let a = try MetalArray<Double>(vals)
            let b = try MetalArray<Double>(vals.reversed())
            for op in CompareOp.allCases {
                XCTAssertEqual(try a.compare(op, 0.0).toArray(), try CPUReference.compare(a, op, scalar: 0.0).toArray(), "\(op) n=\(n)")
                XCTAssertEqual(try a.compare(op, .nan).toArray(), try CPUReference.compare(a, op, scalar: .nan).toArray(), "\(op) nan")
                XCTAssertEqual(try a.compare(op, b).toArray(), try CPUReference.compare(a, op, array: b).toArray(), "\(op) array n=\(n)")
            }
            // min/max skip NaN
            let nonNaN = vals.compactMap { $0 }.filter { !$0.isNaN }
            XCTAssertEqual(try a.min(), nonNaN.min())
            XCTAssertEqual(try a.max(), nonNaN.max())
            let m = try a.compare(.gt, 0)
            XCTAssertEqual(try a.filter(m).toArray(), try CPUReference.filter(a, m).toArray())
        }
        let allNaN = try MetalArray<Double>([.nan, nil, .nan])
        XCTAssertNil(try allNaN.min())
        let f = try MetalArray<Float>([.nan, 2, nil, 1])
        XCTAssertEqual(try f.min(), 1); XCTAssertEqual(try f.max(), 2)
        XCTAssertTrue(try f.sum()!.asDouble.isNaN, "sum propagates NaN")
        XCTAssertNil(try MetalArray<Float>([.nan]).max())
    }

    func testBooleanFilterCountAnyAll() throws {
        try requireRealGPU()
        for n in [0, 1, 33, 1000, 70_000] {
            let b = try MetalBooleanArray((0..<n).map { $0 % 3 == 0 })
            let m = try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })
            let expected = (0..<n).filter { $0 % 2 == 0 }.map { $0 % 3 == 0 }
            XCTAssertEqual(try b.filter(m).toArray(), expected, "n=\(n)")
            XCTAssertEqual(b.count, (0..<n).filter { $0 % 3 == 0 }.count)
            XCTAssertEqual(b.any, n > 0)
            XCTAssertEqual(b.all, n <= 1)
        }
        let withNulls = try MetalBooleanArray.allocate(length: 3, withValidity: true)
        Bitmap.set(withNulls.validity!.mutableTyped(UInt8.self), 0); Bitmap.set(withNulls.values.mutableTyped(UInt8.self), 0)
        withNulls.recomputeNullCount()
        XCTAssertEqual(withNulls.nullCount, 2); XCTAssertTrue(withNulls.all); XCTAssertEqual(withNulls.count, 1)
    }
}

func priceValues(_ n: Int) -> [Float?] {
    var v: [Float?] = []
    for i in 0..<n { v.append(i % 4 == 0 ? nil : Float(i) * 0.5) }
    return v
}

final class RecordBatchTests: XCTestCase {
    func makeBatch(_ n: Int) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: ["id", "price", "qty", "flag", "w"], columns: [
            .int64(try MetalArray<Int64>((0..<n).map { Int64($0) })),
            .float32(try MetalArray<Float>(priceValues(n))),
            .int32(try MetalArray<Int32>((0..<n).map { Int32($0 % 10) })),
            .boolean(try MetalBooleanArray((0..<n).map { $0 % 2 == 0 })),
            .float64(try MetalArray<Double>((0..<n).map { Double($0) })),
        ])
    }

    func testFilterTakeSliceSelect() throws {
        try requireRealGPU()
        let n = 10_000
        let b = try makeBatch(n)
        let mask = try b["qty"]!.asInt32!.compare(.ge, 8)
        let f = try b.filter(mask)
        let keptIds = (0..<n).filter { $0 % 10 >= 8 }
        XCTAssertEqual(f.length, keptIds.count)
        XCTAssertEqual(f["id"]!.asInt64!.toRawArray(), keptIds.map { Int64($0) })
        let expectedPrices: [Float?] = keptIds.map { i in i % 4 == 0 ? nil : Float(i) * 0.5 }
        XCTAssertEqual(f["price"]!.asFloat32!.toArray(), expectedPrices)
        XCTAssertEqual(f["flag"]!.asBoolean!.toArray(), keptIds.map { $0 % 2 == 0 })
        XCTAssertEqual(f["w"]!.asFloat64!.toRawArray(), keptIds.map { Double($0) })
        let t = try b.take(try MetalArray<Int32>([5, 0, 9999]))
        XCTAssertEqual(t["id"]!.asInt64!.toRawArray(), [5, 0, 9999])
        XCTAssertEqual(t["flag"]!.asBoolean!.toArray(), [false, true, false])
        let s = try b.slice(offset: 32, length: 4).selecting(["qty", "id"])
        XCTAssertEqual(s.names, ["qty", "id"])
        XCTAssertEqual(s[0].asInt32!.toRawArray(), [2, 3, 4, 5])
        XCTAssertThrowsError(try b.selecting(["nope"]))
        XCTAssertThrowsError(try MetalRecordBatch(names: ["a"], columns: []))
    }

    func testStructExportImportRoundTrip() throws {
        try requireRealGPU()
        let b = try makeBatch(1000)
        var schema = ArrowSchema(); var arr = ArrowArray()
        b.exportArrowSchema(name: "batch", into: &schema)
        b.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.format), "+s")
        XCTAssertEqual(schema.n_children, 5); XCTAssertEqual(arr.n_children, 5); XCTAssertEqual(arr.n_buffers, 1)
        XCTAssertEqual(String(cString: schema.children[1]!.pointee.name), "price")
        XCTAssertEqual(String(cString: schema.children[1]!.pointee.format), "f")
        let r = try importArrowRecordBatch(schema: &schema, array: &arr)
        XCTAssertTrue(r.zeroCopy)
        XCTAssertNil(arr.release)
        XCTAssertEqual(r.batch.names, b.names)
        XCTAssertEqual(r.batch["price"]!.asFloat32!.toArray(), b["price"]!.asFloat32!.toArray())
        XCTAssertEqual(r.batch["flag"]!.asBoolean!.toArray(), b["flag"]!.asBoolean!.toArray())
        XCTAssertTrue(r.batch["id"]!.asInt64!.values.mtl === b["id"]!.asInt64!.values.mtl)
        var dev = ArrowDeviceArray(); b.exportArrowDeviceArray(into: &dev)
        XCTAssertEqual(dev.device_type, ARROW_DEVICE_METAL); dev.array.release?(&dev.array)
        schema.release?(&schema)
    }

    /// A minimal ArrowArrayStream producer yielding `count` struct batches, as a foreign library would.
    func testArrayStreamImport() throws {
        try requireRealGPU()
        final class Producer { var remaining = 3; var batches: [MetalRecordBatch] = [] }
        let p = Producer()
        p.batches = try (0..<3).map { try makeBatch(100 * ($0 + 1)) }
        var stream = ArrowArrayStream()
        stream.private_data = Unmanaged.passRetained(p).toOpaque()
        stream.get_schema = { s, out in
            let p = Unmanaged<Producer>.fromOpaque(s!.pointee.private_data).takeUnretainedValue()
            p.batches[0].exportArrowSchema(into: out!); return 0
        }
        stream.get_next = { s, out in
            let p = Unmanaged<Producer>.fromOpaque(s!.pointee.private_data).takeUnretainedValue()
            if p.remaining == 0 { out!.pointee.release = nil; return 0 }
            p.batches[3 - p.remaining].exportArrowArray(into: out!); p.remaining -= 1; return 0
        }
        stream.get_last_error = { _ in nil }
        stream.release = { s in
            Unmanaged<Producer>.fromOpaque(s!.pointee.private_data).release(); s!.pointee.release = nil
        }
        let batches = try importArrowArrayStream(&stream)
        XCTAssertNil(stream.release)
        XCTAssertEqual(batches.map(\.length), [100, 200, 300])
        XCTAssertEqual(batches[2]["id"]!.asInt64!.toRawArray().last, 299)
        XCTAssertEqual(try batches[1]["w"]!.asFloat64!.sum(), .float(Double(199 * 200 / 2)))
    }
}
