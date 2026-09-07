import XCTest
@testable import ArrowMetal

/// The (small) Parquet writer: what it emits must come back through the reader unchanged, and through
/// pyarrow too — `python/tests/test_parquet.py` checks the pyarrow half.
final class ParquetWriterTests: XCTestCase {

    private func roundTrip(_ batch: MetalRecordBatch, _ options: ParquetWriteOptions,
                           file: StaticString = #filePath, line: UInt = #line) throws -> MetalRecordBatch {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("arrowmetal-parquet-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("out.parquet").path
        try ParquetWriter.write(batch, to: path, options: options)
        return try ParquetFile(path: path).read()
    }

    private func sample(rows n: Int) throws -> MetalRecordBatch {
        func opt<T>(_ i: Int, _ v: T) -> T? { i % 7 == 3 ? nil : v }
        let strings: [String?] = (0..<n).map { i in
            switch i % 6 {
            case 0: return "alpha"
            case 1: return nil
            case 2: return ""
            case 3: return "a rather longer value \(i)"
            case 4: return "héllo wörld \u{1F600}"
            default: return "beta"
            }
        }
        return try MetalRecordBatch(
            names: ["i32", "i64", "i16", "u32", "f32", "f64", "flag", "text", "blob", "ts", "day"],
            columns: [
                .int32(try MetalArray<Int32>((0..<n).map { opt($0, Int32($0 * 3 - 100)) })),
                .int64(try MetalArray<Int64>((0..<n).map { Int64($0) * 1_000_000_007 })),
                .int16(try MetalArray<Int16>((0..<n).map { opt($0, Int16(truncatingIfNeeded: $0 &* 7)) })),
                .uint32(try MetalArray<UInt32>((0..<n).map { UInt32($0 &* 2_000_003) })),
                .float32(try MetalArray<Float>((0..<n).map { opt($0, Float($0) * 0.25) })),
                .float64(try MetalArray<Double>((0..<n).map { Double($0) * -1.5 })),
                .boolean(try MetalBooleanArray((0..<n).map { $0 % 3 == 0 })),
                .string(try MetalStringArray(strings)),
                .binary({ let a = try! MetalStringArray((0..<n).map { i in String(repeating: "z", count: i % 5) }); a.isBinary = true; return a }()),
                .temporal(try MetalTemporalArray(type: .timestamp(.micro, timezone: nil),
                                                 (0..<n).map { opt($0, Int64($0) * 1_000_003) })),
                .temporal(try MetalTemporalArray(type: .date32, (0..<n).map { Int64($0 % 20000) })),
            ])
    }

    func testRoundTripUncompressed() throws {
        try requireRealGPU()
        let batch = try sample(rows: 1000)
        let back = try roundTrip(batch, ParquetWriteOptions(compression: .uncompressed, useDictionary: false))
        compare(batch, back)
    }

    func testRoundTripSnappyDictionary() throws {
        try requireRealGPU()
        let batch = try sample(rows: 1000)
        let back = try roundTrip(batch, ParquetWriteOptions(compression: .snappy, useDictionary: true))
        compare(batch, back)
    }

    func testRoundTripSeveralRowGroups() throws {
        try requireRealGPU()
        let batch = try sample(rows: 1000)
        let back = try roundTrip(batch, ParquetWriteOptions(compression: .snappy, useDictionary: true,
                                                           rowGroupSize: 137))
        compare(batch, back)
    }

    /// The Snappy encoder must produce something the GPU decoder reads back byte for byte, including on
    /// input with long repeats (which is where the back-reference tokens appear).
    func testSnappyRepetitiveData() throws {
        try requireRealGPU()
        let n = 4000
        let strings = (0..<n).map { i in String(repeating: "abcdefgh", count: (i % 40) + 1) }
        let batch = try MetalRecordBatch(names: ["s"], columns: [.string(try MetalStringArray(strings.map { Optional($0) }))])
        let back = try roundTrip(batch, ParquetWriteOptions(compression: .snappy, useDictionary: false))
        compare(batch, back)
    }

    private func compare(_ a: MetalRecordBatch, _ b: MetalRecordBatch,
                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.names, b.names, file: file, line: line)
        for (i, name) in a.names.enumerated() where i < b.columns.count {
            let fa = ParquetTests.fingerprint(a.columns[i])
            let fb = ParquetTests.fingerprint(b.columns[i])
            XCTAssertEqual(fa.count, fb.count, "\(name): length", file: file, line: line)
            for j in 0..<Swift.min(fa.count, fb.count) where fa[j] != fb[j] {
                XCTFail("\(name)[\(j)]: \(fa[j]) != \(fb[j])", file: file, line: line)
                break
            }
        }
    }
}
