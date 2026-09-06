import XCTest
@testable import ArrowMetal

/// Arrow IPC streaming and file format: round trips through our own reader and writer, and
/// cross-checks against pyarrow in both directions.
final class IPCTests: XCTestCase {

    // MARK: - fixtures

    /// A nullable boolean array (`MetalBooleanArray` only builds non-null ones from Swift values).
    private func booleans(_ values: [Bool?]) throws -> MetalBooleanArray {
        let n = values.count
        let bits = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n))
        let validity = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n))
        let v = bits.mutableTyped(UInt8.self), b = validity.mutableTyped(UInt8.self)
        var nulls = 0
        for (i, value) in values.enumerated() {
            guard let value else { nulls += 1; continue }
            Bitmap.set(b, i)
            if value { Bitmap.set(v, i) }
        }
        return MetalBooleanArray(length: n, nullCount: nulls, validity: nulls == 0 ? nil : validity, values: bits)
    }

    /// One batch covering every supported physical type, with nulls in half of the columns.
    private func sampleBatch(rows n: Int, seed: Int = 0) throws -> MetalRecordBatch {
        func opt<T>(_ i: Int, _ v: T) -> T? { i % 3 == 1 ? nil : v }
        let strings: [String?] = (0..<n).map { i in
            switch i % 5 {
            case 0: return "row-\(seed + i)"
            case 1: return nil
            case 2: return ""
            case 3: return "héllo wörld \u{1F600}"
            default: return "日本語のテキスト \(i)"
            }
        }
        return try MetalRecordBatch(names: ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64", "flag", "text"],
                                    columns: [
            .int8(try MetalArray<Int8>((0..<n).map { opt($0, Int8(truncatingIfNeeded: $0 &+ seed)) })),
            .uint8(try MetalArray<UInt8>((0..<n).map { UInt8(truncatingIfNeeded: $0 &* 3) })),
            .int16(try MetalArray<Int16>((0..<n).map { opt($0, Int16(truncatingIfNeeded: -$0 &+ seed)) })),
            .uint16(try MetalArray<UInt16>((0..<n).map { UInt16(truncatingIfNeeded: $0 &* 257) })),
            .int32(try MetalArray<Int32>((0..<n).map { opt($0, Int32($0 + seed)) })),
            .uint32(try MetalArray<UInt32>((0..<n).map { UInt32($0 &* 7) })),
            .int64(try MetalArray<Int64>((0..<n).map { opt($0, Int64($0) * 1_000_000_007) })),
            .uint64(try MetalArray<UInt64>((0..<n).map { UInt64($0) &* 11_400_714_819_323_198_485 })),
            .float32(try MetalArray<Float>((0..<n).map { opt($0, Float($0) * 0.5) })),
            .float64(try MetalArray<Double>((0..<n).map { Double($0) * 0.125 - 3 })),
            .boolean(try booleans((0..<n).map { opt($0, $0 % 2 == 0) })),
            .string(try MetalStringArray(strings)),
        ])
    }

    private func assertEqual(_ a: MetalRecordBatch, _ b: MetalRecordBatch,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.names, b.names, file: file, line: line)
        XCTAssertEqual(a.length, b.length, file: file, line: line)
        guard a.columnCount == b.columnCount else { return XCTFail("column count", file: file, line: line) }
        for i in 0..<a.columnCount {
            let name = a.names[i]
            switch (a.columns[i], b.columns[i]) {
            case (.int8(let x), .int8(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.uint8(let x), .uint8(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.int16(let x), .int16(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.uint16(let x), .uint16(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.int32(let x), .int32(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.uint32(let x), .uint32(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.int64(let x), .int64(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.uint64(let x), .uint64(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.float32(let x), .float32(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.float64(let x), .float64(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.boolean(let x), .boolean(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.string(let x), .string(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            default: XCTFail("column '\(name)' changed type", file: file, line: line)
            }
            XCTAssertEqual(a.columns[i].nullCount, b.columns[i].nullCount, name, file: file, line: line)
        }
    }

    private func temporaryFile(_ ext: String = "arrow") -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("arrowmetal-ipc-\(UUID().uuidString).\(ext)")
    }

    // MARK: - round trips

    func testRoundTripEverySupportedTypeInBothFormats() throws {
        let batch = try sampleBatch(rows: 97)
        for format in [ArrowIPCFormat.stream, .file] {
            let data = try ArrowIPCWriter.encode([batch], format: format)
            let reader = try ArrowIPCReader(data: data)
            XCTAssertEqual(reader.format, format)
            XCTAssertEqual(reader.batchCount, 1)
            XCTAssertEqual(reader.schema.names, batch.names)
            XCTAssertEqual(reader.schema.fields.map(\.type),
                           [.int(bits: 8, signed: true), .int(bits: 8, signed: false),
                            .int(bits: 16, signed: true), .int(bits: 16, signed: false),
                            .int(bits: 32, signed: true), .int(bits: 32, signed: false),
                            .int(bits: 64, signed: true), .int(bits: 64, signed: false),
                            .float(bits: 32), .float(bits: 64), .bool, .utf8])
            assertEqual(batch, try XCTUnwrap(reader.readAll().first))
        }
    }

    func testRoundTripThroughAFileOnDisk() throws {
        let batch = try sampleBatch(rows: 1000, seed: 5)
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        try ArrowIPCWriter.write([batch], to: url)
        let reader = try ArrowIPCReader(url: url)
        XCTAssertEqual(reader.format, .file)
        assertEqual(batch, try XCTUnwrap(reader.readAll().first))
    }

    func testMultipleBatchesAndRandomAccess() throws {
        let batches = try [0, 1, 2, 3].map { try sampleBatch(rows: 10 * $0 + 1, seed: $0 * 100) }
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        try ArrowIPCWriter.write(batches, to: url)
        let reader = try ArrowIPCReader(url: url)
        XCTAssertEqual(reader.batchCount, 4)
        // Out of order, to prove the footer's block offsets are right.
        for i in [3, 0, 2, 1] { assertEqual(batches[i], try reader.batch(at: i)) }
        XCTAssertThrowsError(try reader.batch(at: 4))

        let stream = try ArrowIPCWriter.encode(batches, format: .stream)
        let streamed = try ArrowIPCReader(data: stream).readAll()
        XCTAssertEqual(streamed.map(\.length), batches.map(\.length))
        for (a, b) in zip(batches, streamed) { assertEqual(a, b) }
    }

    func testEmptyBatchesAndAllNullColumns() throws {
        let empty = try sampleBatch(rows: 0)
        XCTAssertEqual(empty.length, 0)
        let allNull = try MetalRecordBatch(names: ["a", "b", "s"], columns: [
            .int64(try MetalArray<Int64>([nil, nil, nil, nil])),
            .boolean(try booleans([nil, nil, nil, nil])),
            .string(try MetalStringArray([nil, nil, nil, nil])),
        ])
        for format in [ArrowIPCFormat.stream, .file] {
            let readBack = try ArrowIPCReader(data: try ArrowIPCWriter.encode([empty], format: format)).readAll()
            XCTAssertEqual(readBack.count, 1)
            assertEqual(empty, readBack[0])

            let nulls = try ArrowIPCReader(data: try ArrowIPCWriter.encode([allNull], format: format)).readAll()
            assertEqual(allNull, nulls[0])
            XCTAssertEqual(nulls[0]["a"]!.nullCount, 4)
            XCTAssertEqual(nulls[0]["s"]!.asString!.toArray(), [nil, nil, nil, nil])
        }
        // Zero batches: the schema still has to survive.
        let schema = ArrowIPCSchema(fields: [ArrowIPCField(name: "x", type: .float(bits: 64))])
        let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([], schema: schema, format: .file))
        XCTAssertEqual(reader.batchCount, 0)
        XCTAssertEqual(reader.schema, schema)
        XCTAssertTrue(try reader.readAll().isEmpty)
    }

    func testUnicodeAndLongStrings() throws {
        let strings: [String?] = ["", "a", nil, "héllo", "日本語", "\u{1F600}\u{1F1EF}\u{1F1F5}",
                                  String(repeating: "λ", count: 5000), "tab\tnewline\nquote\"", "ünïcödé"]
        let batch = try MetalRecordBatch(names: ["s"], columns: [.string(try MetalStringArray(strings))])
        let readBack = try ArrowIPCReader(data: try ArrowIPCWriter.encode([batch])).readAll()[0]
        XCTAssertEqual(readBack["s"]!.asString!.toArray(), strings)
    }

    /// Logical types with no dedicated array class travel as their storage integers.
    func testTemporalAndBinaryLogicalTypes() throws {
        let days = try MetalArray<Int32>([19723, 19724, nil, 0, -1])
        let micros = try MetalArray<Int64>([1_700_000_000_000_000, nil, 0, -86_400_000_000, 42])
        let numeric = try MetalRecordBatch(names: ["day", "when", "elapsed", "clock"], columns: [
            .int32(days), .int64(micros), .int64(micros), .int64(micros),
        ])
        XCTAssertEqual(numeric.length, 5)
        let schema = ArrowIPCSchema(fields: [
            ArrowIPCField(name: "day", type: .date32),
            ArrowIPCField(name: "when", type: .timestamp(.microsecond, timezone: "UTC")),
            ArrowIPCField(name: "elapsed", type: .duration(.nanosecond)),
            ArrowIPCField(name: "clock", type: .time64(.microsecond)),
        ])
        let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([numeric], schema: schema))
        XCTAssertEqual(reader.schema, schema)
        assertEqual(numeric, try reader.batch(at: 0))

        let bytes = try MetalRecordBatch(names: ["blob", "big"], columns: [
            .string(try MetalStringArray(["\u{01}\u{02}", nil, "bytes"])),
            .string(try MetalStringArray(["one", "twö", nil])),
        ])
        let binarySchema = ArrowIPCSchema(fields: [
            ArrowIPCField(name: "blob", type: .binary),
            ArrowIPCField(name: "big", type: .largeUtf8),
        ])
        let binaryReader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([bytes], schema: binarySchema))
        XCTAssertEqual(binaryReader.schema, binarySchema)
        assertEqual(bytes, try binaryReader.batch(at: 0))
    }

    func testSchemaMismatchIsRejected() throws {
        let batch = try MetalRecordBatch(names: ["a"], columns: [.int32(try MetalArray<Int32>([1, 2]))])
        let wrong = ArrowIPCSchema(fields: [ArrowIPCField(name: "a", type: .float(bits: 64))])
        XCTAssertThrowsError(try ArrowIPCWriter.encode([batch], schema: wrong))
        XCTAssertThrowsError(try ArrowIPCWriter.encode([]))
    }

    func testGarbageInputThrows() throws {
        XCTAssertThrowsError(try ArrowIPCReader(data: Data()))
        XCTAssertThrowsError(try ArrowIPCReader(data: Data(repeating: 0x41, count: 512)))
        var truncated = try ArrowIPCWriter.encode([try sampleBatch(rows: 8)], format: .file)
        truncated.removeLast(64)
        XCTAssertThrowsError(try ArrowIPCReader(data: truncated))
        var corrupt = try ArrowIPCWriter.encode([try sampleBatch(rows: 8)], format: .stream)
        for i in 16..<32 { corrupt[i] = 0xEE }
        XCTAssertThrowsError(try ArrowIPCReader(data: corrupt))
    }

    // MARK: - pyarrow cross-checks

    /// A python interpreter with pyarrow: `ARROWMETAL_PYTHON` if it is set and usable, else one on PATH.
    private static let python: String? = {
        func works(_ path: String) -> Bool {
            guard FileManager.default.isExecutableFile(atPath: path) else { return false }
            let p = Process()
            p.executableURL = URL(fileURLWithPath: path)
            p.arguments = ["-c", "import pyarrow"]
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            do { try p.run() } catch { return false }
            p.waitUntilExit()
            return p.terminationStatus == 0
        }
        var candidates: [String] = []
        if let env = ProcessInfo.processInfo.environment["ARROWMETAL_PYTHON"] { candidates.append(env) }
        candidates += ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
        return candidates.first(where: works)
    }()

    private func requirePython() throws -> String {
        guard let python = IPCTests.python else {
            throw XCTSkip("no python3 with pyarrow found (set ARROWMETAL_PYTHON to one)")
        }
        return python
    }

    @discardableResult
    private func runPython(_ script: String, _ arguments: [String]) throws -> String {
        let python = try requirePython()
        let file = temporaryFile("py")
        defer { try? FileManager.default.removeItem(at: file) }
        try script.write(to: file, atomically: true, encoding: .utf8)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = [file.path] + arguments
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        try process.run()
        let stdout = out.fileHandleForReading.readDataToEndOfFile()
        let stderr = err.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "pyarrow", code: Int(process.terminationStatus), userInfo: [
                NSLocalizedDescriptionKey: String(decoding: stderr, as: UTF8.self) + String(decoding: stdout, as: UTF8.self)
            ])
        }
        return String(decoding: stdout, as: UTF8.self)
    }

    /// Everything ArrowMetal writes has to survive a pyarrow read, in both encapsulations.
    func testPyarrowReadsWhatWeWrite() throws {
        _ = try requirePython()
        let batches = try [11, 0, 64].map { try sampleBatch(rows: $0, seed: $0) }
        let fileURL = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: streamURL)
        }
        try ArrowIPCWriter.write(batches, to: fileURL, format: .file)
        try ArrowIPCWriter.write(batches, to: streamURL, format: .stream)

        let script = """
        import sys, pyarrow as pa

        expected_names = ["i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64", "flag", "text"]
        expected_types = [pa.int8(), pa.uint8(), pa.int16(), pa.uint16(), pa.int32(), pa.uint32(),
                          pa.int64(), pa.uint64(), pa.float32(), pa.float64(), pa.bool_(), pa.string()]

        def check(table, label):
            assert table.schema.names == expected_names, (label, table.schema.names)
            for name, want in zip(expected_names, expected_types):
                got = table.schema.field(name).type
                assert got == want, (label, name, got, want)
            assert table.num_rows == 11 + 0 + 64, (label, table.num_rows)
            table.validate(full=True)
            i32 = table.column("i32").to_pylist()
            text = table.column("text").to_pylist()
            flag = table.column("flag").to_pylist()
            i64 = table.column("i64").to_pylist()
            row = 0
            for rows, seed in ((11, 11), (0, 0), (64, 64)):
                for i in range(rows):
                    want = None if i % 3 == 1 else i + seed
                    assert i32[row] == want, (label, row, i32[row], want)
                    assert i64[row] == (None if i % 3 == 1 else i * 1000000007)
                    assert flag[row] == (None if i % 3 == 1 else i % 2 == 0)
                    if i % 5 == 3:
                        assert text[row] == "héllo wörld \\U0001F600", (label, row, text[row])
                    elif i % 5 == 1:
                        assert text[row] is None
                    elif i % 5 == 2:
                        assert text[row] == ""
                    row += 1
            assert row == table.num_rows

        with pa.ipc.open_file(sys.argv[1]) as reader:
            assert reader.num_record_batches == 3, reader.num_record_batches
            assert [reader.get_batch(i).num_rows for i in range(3)] == [11, 0, 64]
            check(reader.read_all(), "file")
        with pa.ipc.open_stream(sys.argv[2]) as reader:
            check(reader.read_all(), "stream")
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// A file produced by pyarrow, including logical types ArrowMetal carries as integers.
    func testWeReadWhatPyarrowWrites() throws {
        _ = try requirePython()
        let fileURL = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: streamURL)
        }
        let script = """
        import sys, pyarrow as pa

        def batch(b):
            n = 5 + b
            ints = pa.array([None if i % 2 else b * 100 + i for i in range(n)], type=pa.int64())
            floats = pa.array([i * 0.5 + b for i in range(n)], type=pa.float64())
            text = pa.array([None if i == 1 else "héllo-%d-%d-日本" % (b, i) for i in range(n)], type=pa.string())
            flags = pa.array([None if i == 2 else (i % 3 == 0) for i in range(n)], type=pa.bool_())
            days = pa.array([19723 + b * 10 + i for i in range(n)], type=pa.int32()).cast(pa.date32())
            when = pa.array([1700000000000000 + b * 1000 + i for i in range(n)], type=pa.int64()).cast(pa.timestamp("us"))
            blob = pa.array([None if i == 0 else bytes([i, b]) for i in range(n)], type=pa.binary())
            big = pa.array(["large-%d-%d" % (b, i) for i in range(n)], type=pa.large_string())
            return pa.record_batch([ints, floats, text, flags, days, when, blob, big],
                                   names=["ints", "floats", "text", "flags", "days", "when", "blob", "big"])

        batches = [batch(b) for b in range(3)]
        with pa.ipc.new_file(sys.argv[1], batches[0].schema) as w:
            for b in batches:
                w.write_batch(b)
        with pa.ipc.new_stream(sys.argv[2], batches[0].schema) as w:
            for b in batches:
                w.write_batch(b)
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")

        for (url, format) in [(fileURL, ArrowIPCFormat.file), (streamURL, .stream)] {
            let reader = try ArrowIPCReader(url: url)
            XCTAssertEqual(reader.format, format)
            XCTAssertEqual(reader.batchCount, 3)
            XCTAssertEqual(reader.schema.names, ["ints", "floats", "text", "flags", "days", "when", "blob", "big"])
            XCTAssertEqual(reader.schema.fields.map(\.type), [
                .int(bits: 64, signed: true), .float(bits: 64), .utf8, .bool,
                .date32, .timestamp(.microsecond, timezone: nil), .binary, .largeUtf8,
            ])
            let batches = try reader.readAll()
            XCTAssertEqual(batches.map(\.length), [5, 6, 7])
            for (b, batch) in batches.enumerated() {
                let n = 5 + b
                XCTAssertEqual(batch["ints"]!.asInt64!.toArray(),
                               (0..<n).map { $0 % 2 == 1 ? nil : Int64(b * 100 + $0) })
                XCTAssertEqual(batch["ints"]!.nullCount, n / 2)
                XCTAssertEqual(batch["floats"]!.asFloat64!.toArray(), (0..<n).map { Double($0) * 0.5 + Double(b) })
                XCTAssertEqual(batch["text"]!.asString!.toArray(),
                               (0..<n).map { $0 == 1 ? nil : "héllo-\(b)-\($0)-日本" })
                XCTAssertEqual(batch["flags"]!.asBoolean!.toArray(),
                               (0..<n).map { $0 == 2 ? nil : $0 % 3 == 0 })
                XCTAssertEqual(batch["days"]!.asInt32!.toArray(), (0..<n).map { Int32(19723 + b * 10 + $0) })
                XCTAssertEqual(batch["when"]!.asInt64!.toArray(), (0..<n).map { Int64(1_700_000_000_000_000 + b * 1000 + $0) })
                XCTAssertEqual(batch["blob"]!.asString!.toArray().first!, nil)
                XCTAssertEqual(batch["big"]!.asString!.toArray(), (0..<n).map { "large-\(b)-\($0)" })
            }
        }
    }

    /// A pyarrow file that uses features we do not implement has to fail with a clear error.
    func testDictionaryEncodedInputIsRejected() throws {
        _ = try requirePython()
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let script = """
        import sys, pyarrow as pa
        col = pa.array(["a", "b", "a", "c"]).dictionary_encode()
        batch = pa.record_batch([col], names=["k"])
        with pa.ipc.new_file(sys.argv[1], batch.schema) as w:
            w.write_batch(batch)
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [url.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
        XCTAssertThrowsError(try ArrowIPCReader(url: url)) { error in
            guard case ArrowIPCError.unsupported(let what) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(what.contains("dictionary"), what)
        }
    }

    /// Reads a 1 GB file and reports throughput. Off by default; set ARROWMETAL_IPC_THROUGHPUT=1.
    func testReadThroughput() throws {
        guard ProcessInfo.processInfo.environment["ARROWMETAL_IPC_THROUGHPUT"] != nil else {
            throw XCTSkip("set ARROWMETAL_IPC_THROUGHPUT=1 to measure IPC read throughput")
        }
        let rowsPerBatch = 8_000_000                       // 64 MB per Int64 column
        let batches = try (0..<8).map { b -> MetalRecordBatch in
            try MetalRecordBatch(names: ["a", "b"], columns: [
                .int64(try MetalArray<Int64>((0..<rowsPerBatch).map { Int64($0 &+ b) })),
                .float64(try MetalArray<Double>((0..<rowsPerBatch).map { Double($0) })),
            ])
        }
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        try ArrowIPCWriter.write(batches, to: url)
        let bytes = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as! Int
        var best = Double.infinity
        for _ in 0..<3 {
            let start = DispatchTime.now().uptimeNanoseconds
            let read = try ArrowIPCReader(url: url).readAll()
            XCTAssertEqual(read.reduce(0) { $0 + $1.length }, rowsPerBatch * 8)
            best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        }
        print(String(format: "IPC read: %.0f MB in %.3f s = %.2f GB/s",
                     Double(bytes) / 1e6, best, Double(bytes) / best / 1e9))
    }
}
