import XCTest
import CArrowABI
@testable import ArrowMetal

final class KernelTests: XCTestCase {
    func randomArray<T: ArrowPrimitive>(_: T.Type, n: Int, nullFraction: Double, gen: (inout SystemRandomNumberGenerator) -> T) throws -> MetalArray<T> {
        var g = SystemRandomNumberGenerator()
        var vals: [T?] = []
        vals.reserveCapacity(n)
        for _ in 0..<n { vals.append(Double.random(in: 0..<1, using: &g) < nullFraction ? nil : gen(&g)) }
        return try MetalArray<T>(vals)
    }

    func checkReductions<T: ArrowPrimitive>(_ a: MetalArray<T>, file: StaticString = #filePath, line: UInt = #line) throws {
        let s = try a.sum(), cs = CPUReference.sum(a)
        if T.isFloatingPoint {
            XCTAssertEqual(s == nil, cs == nil, "sum nil-ness \(T.self)", file: file, line: line)
            if let s, let cs {
                XCTAssertEqual(s.asDouble, cs.asDouble, accuracy: Swift.max(1e-3 * abs(cs.asDouble), 1e-2), "sum \(T.self)", file: file, line: line)
            }
        } else {
            XCTAssertEqual(s, cs, "sum \(T.self)", file: file, line: line)
        }
        XCTAssertEqual(try a.min(), CPUReference.min(a), "min \(T.self)", file: file, line: line)
        XCTAssertEqual(try a.max(), CPUReference.max(a), "max \(T.self)", file: file, line: line)
    }

    func testReductionsAllTypes() throws {
        for n in [0, 1, 7, 33, 255, 256, 257, 100_003, 1_000_000] {
            try checkReductions(try randomArray(Int8.self, n: n, nullFraction: 0.1) { T in Int8.random(in: .min ... .max, using: &T) })
            try checkReductions(try randomArray(UInt8.self, n: n, nullFraction: 0.1) { T in UInt8.random(in: .min ... .max, using: &T) })
            try checkReductions(try randomArray(Int16.self, n: n, nullFraction: 0.1) { T in Int16.random(in: .min ... .max, using: &T) })
            try checkReductions(try randomArray(UInt16.self, n: n, nullFraction: 0.0) { T in UInt16.random(in: .min ... .max, using: &T) })
            try checkReductions(try randomArray(Int32.self, n: n, nullFraction: 0.3) { T in Int32.random(in: -1_000_000...1_000_000, using: &T) })
            try checkReductions(try randomArray(UInt32.self, n: n, nullFraction: 0.1) { T in UInt32.random(in: 0...1_000_000, using: &T) })
            try checkReductions(try randomArray(Int64.self, n: n, nullFraction: 0.1) { T in Int64.random(in: -1_000_000_000...1_000_000_000, using: &T) })
            try checkReductions(try randomArray(UInt64.self, n: n, nullFraction: 0.5) { T in UInt64.random(in: 0...1_000_000_000, using: &T) })
            try checkReductions(try randomArray(Float.self, n: n, nullFraction: 0.1) { T in Float.random(in: -100...100, using: &T) })
            try checkReductions(try randomArray(Double.self, n: n, nullFraction: 0.1) { T in Double.random(in: -100...100, using: &T) })
        }
    }

    func testAllNullSumIsNil() throws {
        let a = try MetalArray<Int32>([nil, nil, nil])
        XCTAssertNil(try a.sum()); XCTAssertNil(try a.min()); XCTAssertNil(try a.max()); XCTAssertNil(try a.mean())
        let e = try MetalArray<Int64>([Int64]())
        XCTAssertNil(try e.sum())
    }

    func testMeanAndOverflowWrap() throws {
        let a = try MetalArray<Int64>([Int64.max, 1])
        XCTAssertEqual(try a.sum(), .int(Int64.min))
        let b = try MetalArray<Int32>([1, 2, nil, 3])
        XCTAssertEqual(try b.mean(), 2.0)
    }

    func testCompareScalarAndArray() throws {
        for n in [0, 1, 31, 32, 33, 1000, 65_537] {
            let a = try randomArray(Int32.self, n: n, nullFraction: 0.2) { T in Int32.random(in: -50...50, using: &T) }
            let b = try randomArray(Int32.self, n: n, nullFraction: 0.2) { T in Int32.random(in: -50...50, using: &T) }
            for op in CompareOp.allCases {
                let g = try a.compare(op, 7), c = try CPUReference.compare(a, op, scalar: 7)
                XCTAssertEqual(g.toArray(), c.toArray(), "\(op) scalar n=\(n)")
                XCTAssertEqual(g.nullCount, c.nullCount)
                let g2 = try a.compare(op, b), c2 = try CPUReference.compare(a, op, array: b)
                XCTAssertEqual(g2.toArray(), c2.toArray(), "\(op) array n=\(n)")
                XCTAssertEqual(g2.nullCount, c2.nullCount)
            }
            let f = try randomArray(Float.self, n: n, nullFraction: 0.0) { T in Float.random(in: -1...1, using: &T) }
            XCTAssertEqual(try f.compare(.gt, 0).toArray(), try CPUReference.compare(f, .gt, scalar: 0).toArray())
        }
    }

    func testBooleanLogic() throws {
        let a = try MetalBooleanArray([true, false, true, false, true])
        let b = try MetalBooleanArray([true, true, false, false, true])
        XCTAssertEqual(try a.and(b).toArray(), [true, false, false, false, true])
        XCTAssertEqual(try a.or(b).toArray(), [true, true, true, false, true])
        XCTAssertEqual(try a.not().toArray(), [false, true, false, true, false])
        XCTAssertEqual(a.trueCount, 3)
    }

    func testArithmetic() throws {
        for n in [0, 1, 255, 10_001] {
            let a = try randomArray(Int64.self, n: n, nullFraction: 0.2) { T in Int64.random(in: -1000...1000, using: &T) }
            let b = try randomArray(Int64.self, n: n, nullFraction: 0.2) { T in Int64.random(in: 1...1000, using: &T) }
            for op in ArithmeticOp.allCases {
                XCTAssertEqual(try a.arithmetic(op, 3).toArray(), try CPUReference.arithmetic(a, op, scalar: 3).toArray(), "\(op)")
                let g = try a.arithmetic(op, b), c = try CPUReference.arithmetic(a, op, array: b)
                XCTAssertEqual(g.toArray(), c.toArray(), "\(op) array")
                XCTAssertEqual(g.nullCount, c.nullCount)
            }
            let f = try randomArray(Float.self, n: n, nullFraction: 0.1) { T in Float.random(in: -1...1, using: &T) }
            let fr = try f.multiply(2.5).toArray(), fc = try CPUReference.arithmetic(f, .mul, scalar: 2.5).toArray()
            XCTAssertEqual(fr, fc)
            // Double runs on the CPU path but through the same API.
            let d = try MetalArray<Double>([1.5, nil, 3.0])
            XCTAssertEqual(try d.add(1).toArray(), [2.5, nil, 4.0])
            XCTAssertEqual(try d.sum(), .float(4.5))
        }
    }

    func testFilter() throws {
        for n in [0, 1, 31, 32, 33, 8191, 8192, 8193, 100_000, 1_000_001] {
            let a = try randomArray(Int32.self, n: n, nullFraction: 0.15) { T in Int32.random(in: -100...100, using: &T) }
            let mask = try a.compare(.gt, 0)
            let g = try a.filter(mask), c = try CPUReference.filter(a, mask)
            XCTAssertEqual(g.length, c.length, "n=\(n)")
            XCTAssertEqual(g.toArray(), c.toArray(), "n=\(n)")
            XCTAssertEqual(g.nullCount, c.nullCount)
            // Non-null array path (no validity buffer).
            let b = try randomArray(Int64.self, n: n, nullFraction: 0) { T in Int64.random(in: 0...9, using: &T) }
            let m2 = try b.compare(.eq, 4)
            XCTAssertEqual(try b.filter(m2).toArray(), try CPUReference.filter(b, m2).toArray())
            // Mask with nulls drops those rows.
            var mv: [Bool?] = (0..<n).map { $0 % 3 == 0 }
            if n > 2 { mv[1] = nil }
            let m3 = try MetalBooleanArray.allocate(length: n, withValidity: true)
            for (i, v) in mv.enumerated() { if let v { Bitmap.set(m3.validity!.mutableTyped(UInt8.self), i); if v { Bitmap.set(m3.values.mutableTyped(UInt8.self), i) } } }
            m3.recomputeNullCount()
            XCTAssertEqual(try b.filter(m3).toArray(), try CPUReference.filter(b, m3).toArray())
        }
    }
}

final class CInteropTests: XCTestCase {
    func testExportImportRoundTripIsZeroCopy() throws {
        let a = try MetalArray<Int64>([1, nil, 3, 4, nil])
        var schema = ArrowSchema(); var arr = ArrowArray()
        a.exportArrowSchema(name: "x", into: &schema)
        a.exportArrowArray(into: &arr)
        XCTAssertEqual(String(cString: schema.format), "l")
        XCTAssertEqual(String(cString: schema.name), "x")
        XCTAssertEqual(arr.length, 5); XCTAssertEqual(arr.null_count, 2); XCTAssertEqual(arr.n_buffers, 2)
        XCTAssertEqual(arr.buffers[1], UnsafeRawPointer(a.values.contents))
        let r = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertTrue(r.zeroCopy, "Metal-allocated buffers are page aligned, so import must not copy")
        XCTAssertNil(arr.release, "import must move the struct")
        guard case .int64(let b) = r.array else { return XCTFail() }
        XCTAssertEqual(b.toArray(), [1, nil, 3, 4, nil])
        XCTAssertEqual(b.values.contents, a.values.contents)
        XCTAssertEqual(try b.sum(), .int(8))
        schema.release?(&schema)
    }

    func testDeviceExport() throws {
        let a = try MetalArray<Float>([1, 2, 3])
        var d = ArrowDeviceArray()
        a.exportArrowDeviceArray(into: &d)
        XCTAssertEqual(d.device_type, ARROW_DEVICE_METAL)
        XCTAssertEqual(d.device_id, -1)
        XCTAssertNil(d.sync_event)
        let bufs = try XCTUnwrap(metalBuffers(of: &d))
        XCTAssertNil(bufs[0]); XCTAssertTrue(bufs[1] === a.values.mtl)
        var schema = ArrowSchema(); a.exportArrowSchema(into: &schema)
        let r = try importArrowDeviceArray(schema: &schema, array: &d)
        guard case .float32(let b) = r.array else { return XCTFail() }
        XCTAssertTrue(r.zeroCopy)
        XCTAssertEqual(b.toRawArray(), [1, 2, 3])
        schema.release?(&schema)
    }

    /// Simulates a foreign producer (e.g. arrow-rs, pyarrow, nanoarrow) handing us a CPU array.
    func foreignArray(values: [Int32], validity: [UInt8]?, offset: Int, pageAligned: Bool,
                      released: UnsafeMutablePointer<Bool>) -> (ArrowArray, [UnsafeMutableRawPointer]) {
        var allocs: [UnsafeMutableRawPointer] = []
        func alloc(_ bytes: Int) -> UnsafeMutableRawPointer {
            let p = UnsafeMutableRawPointer.allocate(byteCount: max(bytes, 1) + (pageAligned ? 0 : 8), alignment: pageAligned ? metalPageSize() : 8)
            allocs.append(p)
            return pageAligned ? p : p + 8  // deliberately misalign
        }
        let vp = alloc(values.count * 4)
        values.withUnsafeBytes { _ = memcpy(vp, $0.baseAddress!, $0.count) }
        var bp: UnsafeMutableRawPointer? = nil
        if let v = validity { bp = alloc(v.count); v.withUnsafeBytes { _ = memcpy(bp!, $0.baseAddress!, $0.count) } }
        let buffers = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 2)
        buffers[0] = bp.map { UnsafeRawPointer($0) }; buffers[1] = UnsafeRawPointer(vp)
        var arr = ArrowArray()
        arr.length = Int64(values.count - offset); arr.null_count = -1; arr.offset = Int64(offset)
        arr.n_buffers = 2; arr.buffers = buffers
        arr.private_data = UnsafeMutableRawPointer(released)
        arr.release = { p in
            guard let p else { return }
            p.pointee.private_data!.assumingMemoryBound(to: Bool.self).pointee = true
            p.pointee.buffers.deallocate()
            p.pointee.release = nil
        }
        return (arr, allocs)
    }

    func testImportForeignPageAlignedIsZeroCopyAndReleasesOnDeinit() throws {
        let released = UnsafeMutablePointer<Bool>.allocate(capacity: 1); released.pointee = false
        var schema = ArrowSchema(); exportArrowSchema(format: "i", into: &schema)
        let vals = (0..<5000).map { Int32($0) }
        var (arr, allocs) = foreignArray(values: vals, validity: nil, offset: 0, pageAligned: true, released: released)
        do {
            let r = try importArrowArray(schema: &schema, array: &arr)
            XCTAssertTrue(r.zeroCopy)
            guard case .int32(let a) = r.array else { return XCTFail() }
            XCTAssertEqual(a.nullCount, 0)
            XCTAssertEqual(try a.sum(), .int(Int64(vals.map(Int.init).reduce(0, +))))
            XCTAssertEqual(try a.filter(try a.compare(.lt, 10)).toRawArray(), (0..<10).map { Int32($0) })
            XCTAssertFalse(released.pointee, "buffers must stay alive while the Metal array lives")
        }
        XCTAssertTrue(released.pointee, "release callback must run when the imported array is freed")
        allocs.forEach { $0.deallocate() }; released.deallocate(); schema.release?(&schema)
    }

    func testImportForeignUnalignedCopiesAndHandlesOffset() throws {
        let released = UnsafeMutablePointer<Bool>.allocate(capacity: 1); released.pointee = false
        var schema = ArrowSchema(); exportArrowSchema(format: "i", into: &schema)
        let vals = (0..<100).map { Int32($0) }
        var validity = [UInt8](repeating: 0xFF, count: 13)
        validity[0] = 0b1111_1110  // element 0 null
        validity[1] = 0b1111_1101  // element 9 null
        var (arr, allocs) = foreignArray(values: vals, validity: validity, offset: 3, pageAligned: false, released: released)
        let r = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertFalse(r.zeroCopy)
        guard case .int32(let a) = r.array else { return XCTFail() }
        XCTAssertEqual(a.length, 97)
        XCTAssertEqual(a.nullCount, 1)
        XCTAssertEqual(a[0], 3); XCTAssertNil(a[6]); XCTAssertEqual(a[7], 10); XCTAssertEqual(a[96], 99)
        allocs.forEach { $0.deallocate() }; released.deallocate(); schema.release?(&schema)
    }

    func testUnsupportedTypeThrows() throws {
        var schema = ArrowSchema(); exportArrowSchema(format: "u", into: &schema)
        var arr = ArrowArray(); arr.n_buffers = 3; arr.release = { _ in }
        XCTAssertThrowsError(try importArrowArray(schema: &schema, array: &arr))
        schema.release?(&schema)
    }
}
