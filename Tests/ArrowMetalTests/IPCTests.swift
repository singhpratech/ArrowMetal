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
            case (.binary(let x), .binary(let y)): XCTAssertEqual(x.toByteArrays(), y.toByteArrays(), name, file: file, line: line)
            case (.temporal(let x), .temporal(let y)):
                XCTAssertEqual(x.type, y.type, name, file: file, line: line)
                XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
            case (.dictionary, .dictionary):
                // Compare decoded: a round trip may renumber the codes.
                assertColumnsEqual(try! a.columns[i].decode(), try! b.columns[i].decode(), name, file: file, line: line)
            default: XCTFail("column '\(name)' changed type", file: file, line: line)
            }
            XCTAssertEqual(a.columns[i].nullCount, b.columns[i].nullCount, name, file: file, line: line)
        }
    }

    /// Value comparison for two columns of the same concrete type (used for decoded dictionaries).
    private func assertColumnsEqual(_ a: AnyMetalArray, _ b: AnyMetalArray, _ name: String,
                                    file: StaticString = #filePath, line: UInt = #line) {
        switch (a, b) {
        case (.string(let x), .string(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
        case (.binary(let x), .binary(let y)): XCTAssertEqual(x.toByteArrays(), y.toByteArrays(), name, file: file, line: line)
        case (.int32(let x), .int32(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
        case (.int64(let x), .int64(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
        case (.float64(let x), .float64(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
        case (.temporal(let x), .temporal(let y)): XCTAssertEqual(x.toArray(), y.toArray(), name, file: file, line: line)
        default: XCTFail("column '\(name)' decoded to \(a.arrowFormat) vs \(b.arrowFormat)", file: file, line: line)
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

    /// Temporal and binary columns keep their logical type through a round trip: the reader hands back
    /// `.temporal` / `.binary`, not the storage integers or the utf8 they are made of.
    func testTemporalAndBinaryLogicalTypes() throws {
        let days = try MetalArray<Int32>([19723, 19724, nil, 0, -1])
        let micros = try MetalArray<Int64>([1_700_000_000_000_000, nil, 0, -86_400_000_000, 42])
        let numeric = try MetalRecordBatch(names: ["day", "when", "elapsed", "clock"], columns: [
            .temporal(try MetalTemporalArray(type: .date32, days)),
            .temporal(try MetalTemporalArray(type: .timestamp(.micro, timezone: "UTC"), micros)),
            .temporal(try MetalTemporalArray(type: .duration(.nano), micros)),
            .temporal(try MetalTemporalArray(type: .time64(.micro), micros)),
        ])
        XCTAssertEqual(numeric.length, 5)
        let schema = ArrowIPCSchema(fields: [
            ArrowIPCField(name: "day", type: .date32),
            ArrowIPCField(name: "when", type: .timestamp(.microsecond, timezone: "UTC")),
            ArrowIPCField(name: "elapsed", type: .duration(.nanosecond)),
            ArrowIPCField(name: "clock", type: .time64(.microsecond)),
        ])
        // The schema derived from the columns themselves matches the explicit one.
        let derived = try ArrowIPCReader(data: try ArrowIPCWriter.encode([numeric]))
        XCTAssertEqual(derived.schema, schema)
        assertEqual(numeric, try derived.batch(at: 0))

        let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([numeric], schema: schema))
        XCTAssertEqual(reader.schema, schema)
        assertEqual(numeric, try reader.batch(at: 0))
        XCTAssertEqual(try reader.batch(at: 0)["day"]!.asTemporal!.toArray(), [19723, 19724, nil, 0, -1])

        // Storage integers still write against an explicit temporal schema; they read back as temporal.
        let asIntegers = try MetalRecordBatch(names: ["day", "when", "elapsed", "clock"], columns: [
            .int32(days), .int64(micros), .int64(micros), .int64(micros),
        ])
        let fromIntegers = try ArrowIPCReader(data: try ArrowIPCWriter.encode([asIntegers], schema: schema))
        assertEqual(numeric, try fromIntegers.batch(at: 0))

        let bytes = try MetalRecordBatch(names: ["blob", "big"], columns: [
            .binary(try MetalStringArray(bytes: [[0x01, 0x02], nil, Array("bytes".utf8)])),
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
                let expectedInts: [Int64?] = (0..<n).map { (i: Int) -> Int64? in
                    if i % 2 == 1 { return nil }
                    return Int64(b * 100 + i)
                }
                XCTAssertEqual(batch["ints"]!.asInt64!.toArray(), expectedInts)
                XCTAssertEqual(batch["ints"]!.nullCount, n / 2)
                XCTAssertEqual(batch["floats"]!.asFloat64!.toArray(), (0..<n).map { Double($0) * 0.5 + Double(b) })
                XCTAssertEqual(batch["text"]!.asString!.toArray(),
                               (0..<n).map { $0 == 1 ? nil : "héllo-\(b)-\($0)-日本" })
                XCTAssertEqual(batch["flags"]!.asBoolean!.toArray(),
                               (0..<n).map { $0 == 2 ? nil : $0 % 3 == 0 })
                XCTAssertEqual(batch["days"]!.asTemporal!.type, .date32)
                XCTAssertEqual(batch["days"]!.asTemporal!.toArray(), (0..<n).map { Int64(19723 + b * 10 + $0) })
                XCTAssertEqual(batch["when"]!.asTemporal!.type, .timestamp(.micro, timezone: nil))
                XCTAssertEqual(batch["when"]!.asTemporal!.toArray(), (0..<n).map { Int64(1_700_000_000_000_000 + b * 1000 + $0) })
                XCTAssertEqual(batch["blob"]!.asBinary!.toByteArrays().first!, nil)
                XCTAssertEqual(batch["blob"]!.asBinary!.bytes(at: 1), [1, UInt8(b)])
                XCTAssertEqual(batch["big"]!.asString!.toArray(), (0..<n).map { "large-\(b)-\($0)" })
            }
        }
    }

    /// A pyarrow file with dictionary-encoded columns reads back as `.dictionary` (codes + values).
    func testDictionaryEncodedInputIsRead() throws {
        _ = try requirePython()
        let url = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: streamURL)
        }
        let script = """
        import sys, pyarrow as pa
        col = pa.array(["a", "b", None, "a", "c"]).dictionary_encode()
        nums = pa.array([10, 20, 10, None, 30], type=pa.int64()).dictionary_encode()
        batch = pa.record_batch([col, nums], names=["k", "n"])
        with pa.ipc.new_file(sys.argv[1], batch.schema) as w:
            w.write_batch(batch)
        with pa.ipc.new_stream(sys.argv[2], batch.schema) as w:
            w.write_batch(batch)
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [url.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
        for source in [url, streamURL] {
            let reader = try ArrowIPCReader(url: source)
            let batch = try reader.batch(at: 0)
            let k = try XCTUnwrap(batch["k"]?.asDictionary)
            XCTAssertEqual(k.values.asString?.toArray(), ["a", "b", "c"])
            XCTAssertEqual(k.codes.toArray(), [0, 1, nil, 0, 2])
            XCTAssertEqual(try batch["k"]!.decode().asString?.toArray(), ["a", "b", nil, "a", "c"])
            let n = try XCTUnwrap(batch["n"]?.asDictionary)
            XCTAssertEqual(n.values.asInt64?.toArray(), [10, 20, 30])
            XCTAssertEqual(try batch["n"]!.decode().asInt64?.toArray(), [10, 20, 10, nil, 30])
        }
    }

    /// Dictionary columns survive our own round trip in both encapsulations.
    func testDictionaryColumnsRoundTrip() throws {
        let codes = try MetalArray<Int32>([0, 2, nil, 1, 1, 0])
        let values = AnyMetalArray.string(try MetalStringArray(["red", "green", "blue"]))
        let column = AnyMetalArray.dictionary(codes: codes, values: values)
        let batch = try MetalRecordBatch(names: ["colour"], columns: [column])
        for format in [ArrowIPCFormat.stream, .file] {
            let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([batch], format: format))
            XCTAssertEqual(reader.schema.fields[0].type, .dictionary(index: .int(bits: 32, signed: true), value: .utf8))
            let back = try reader.batch(at: 0)
            let d = try XCTUnwrap(back["colour"]?.asDictionary)
            XCTAssertEqual(d.codes.toArray(), [0, 2, nil, 1, 1, 0])
            XCTAssertEqual(d.values.asString?.toArray(), ["red", "green", "blue"])
            assertEqual(batch, back)
        }
        // Two batches sharing one dictionary.
        let second = try MetalRecordBatch(names: ["colour"], columns: [
            .dictionary(codes: try MetalArray<Int32>([2, 2]), values: values),
        ])
        let both = try ArrowIPCReader(data: try ArrowIPCWriter.encode([batch, second])).readAll()
        XCTAssertEqual(both.map(\.length), [6, 2])
        XCTAssertEqual(try both[1]["colour"]!.decode().asString?.toArray(), ["blue", "blue"])
        // A second, different dictionary for the same column is refused rather than silently wrong.
        let other = try MetalRecordBatch(names: ["colour"], columns: [
            .dictionary(codes: try MetalArray<Int32>([0]), values: .string(try MetalStringArray(["red"]))),
        ])
        XCTAssertThrowsError(try ArrowIPCWriter.encode([batch, other]))
    }

    /// pyarrow reads the dictionary and temporal columns we write, in both encapsulations.
    func testPyarrowReadsOurDictionaryAndTemporalColumns() throws {
        _ = try requirePython()
        let fileURL = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: streamURL)
        }
        let column = AnyMetalArray.dictionary(codes: try MetalArray<Int32>([0, 2, nil, 1, 1, 0]),
                                              values: .string(try MetalStringArray(["red", "green", "blue"])))
        let stamps = AnyMetalArray.temporal(try MetalTemporalArray(type: .timestamp(.micro, timezone: "UTC"),
                                                                   try MetalArray<Int64>([1, 2, nil, 4, 5, 6])))
        let days = AnyMetalArray.temporal(try MetalTemporalArray(type: .date32,
                                                                 try MetalArray<Int32>([19723, 19724, 0, 1, 2, nil])))
        let blob = AnyMetalArray.binary(try MetalStringArray(bytes: [[1, 2], nil, [], [3], [4, 5, 6], [7]]))
        let batch = try MetalRecordBatch(names: ["colour", "when", "day", "blob"], columns: [column, stamps, days, blob])
        try ArrowIPCWriter.write([batch], to: fileURL, format: .file)
        try ArrowIPCWriter.write([batch], to: streamURL, format: .stream)
        let script = """
        import sys, pyarrow as pa

        def check(table, label):
            table.validate(full=True)
            colour = table.column("colour")
            assert pa.types.is_dictionary(colour.type), (label, colour.type)
            assert colour.type.value_type == pa.string(), (label, colour.type)
            assert colour.to_pylist() == ["red", "blue", None, "green", "green", "red"], (label, colour.to_pylist())
            when = table.column("when")
            assert when.type == pa.timestamp("us", "UTC"), (label, when.type)
            assert when.to_pylist()[2] is None
            assert table.column("day").type == pa.date32(), (label, table.column("day").type)
            assert table.column("day").to_pylist()[5] is None
            blob = table.column("blob")
            assert blob.type == pa.binary(), (label, blob.type)
            assert blob.to_pylist() == [b"\\x01\\x02", None, b"", b"\\x03", b"\\x04\\x05\\x06", b"\\x07"], (label, blob.to_pylist())

        with pa.ipc.open_file(sys.argv[1]) as r:
            check(r.read_all(), "file")
        with pa.ipc.open_stream(sys.argv[2]) as r:
            check(r.read_all(), "stream")
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    // MARK: - the rest of the type matrix

    /// Four rows of every type the writer can flatten: decimals, float16, fixed-size binary, interval,
    /// null, the three list layouts, struct, map, both unions, run-end encoding and an extension type.
    private func typeMatrixBatch() throws -> MetalRecordBatch {
        let entries = try MetalListArray(counts: [2, nil, 1, 1], values: .structure(
            try MetalStructArray(names: ["key", "value"], children: [
                .string(try MetalStringArray(["a", "b", "c", "d"])),
                .int32(try MetalArray<Int32>([1, nil, 3, 4])),
            ])))
        return try MetalRecordBatch(
            names: ["dec128", "dec256", "half", "fixed", "span", "nothing",
                    "list", "biglist", "grid", "person", "lookup", "dense", "sparse", "runs", "tagged"],
            columns: [
                .decimal(try MetalDecimalArray(type: try ArrowDecimalType(precision: 18, scale: 3),
                                               unscaled: [12345, nil, -67890, 0])),
                .decimal(try MetalDecimalArray(type: try ArrowDecimalType(precision: 40, scale: 2, bitWidth: 256),
                                               unscaled: [1, -2, nil, 999_999])),
                .float16(try MetalFloat16Array([1.5, nil, -2.25, 0])),
                .fixedBinary(try MetalFixedBinaryArray(byteWidth: 3, [[1, 2, 3], nil, [4, 5, 6], [0, 0, 0]])),
                .interval(try MetalIntervalArray(unit: .monthDayNano, [
                    ArrowInterval(months: 1, days: 2, nanoseconds: 3), nil,
                    ArrowInterval(months: -4, days: -5, nanoseconds: -6), ArrowInterval(),
                ])),
                .null(MetalNullArray(length: 4)),
                // A null row, an empty row and a null element inside a row.
                .list(try MetalListArray(counts: [2, nil, 0, 3],
                                         values: .int32(try MetalArray<Int32>([1, nil, 7, 8, 9])))),
                // The same shape again, written as large_list against an explicit schema.
                .list(try MetalListArray(counts: [2, nil, 0, 3],
                                         values: .int32(try MetalArray<Int32>([1, nil, 7, 8, 9])))),
                .list(try MetalListArray(counts: [3, nil, 3, 3],
                                         values: .float64(try MetalArray<Double>((0..<12).map { Double($0) / 4 })),
                                         kind: .fixedSize(3))),
                // A null struct whose children are not null at that row.
                .structure(try MetalStructArray(names: ["n", "s"], children: [
                    .int64(try MetalArray<Int64>([10, 20, nil, 40])),
                    .string(try MetalStringArray(["one", "two", "three", nil])),
                ], valid: [true, false, true, true])),
                .map(try MetalMapArray(entries: entries, keysSorted: false)),
                .union(try MetalUnionArray(mode: .dense, length: 4,
                                           typeIds: try MetalArray<Int8>([0, 1, 0, 1]),
                                           offsets: try MetalArray<Int32>([0, 0, 1, 1]),
                                           typeCodes: [0, 1], names: ["ints", "text"],
                                           children: [.int32(try MetalArray<Int32>([10, nil])),
                                                      .string(try MetalStringArray(["x", nil]))])),
                .union(try MetalUnionArray(mode: .sparse, length: 4,
                                           typeIds: try MetalArray<Int8>([3, 7, 7, 3]),
                                           offsets: nil,
                                           typeCodes: [3, 7], names: ["ints", "text"],
                                           children: [.int32(try MetalArray<Int32>([1, 2, 3, nil])),
                                                      .string(try MetalStringArray(["p", "q", nil, "s"]))])),
                .runEndEncoded(runEnds: try MetalArray<Int32>([2, 3, 4]),
                               values: .string(try MetalStringArray(["aa", nil, "cc"]))),
                AnyMetalArray.int64(try MetalArray<Int64>([7, nil, 9, 11]))
                    .asExtensionType(name: "arrowmetal.test", metadata: Array("{\"k\":1}".utf8)),
            ])
    }

    /// The schema `typeMatrixBatch` writes itself as, with "biglist" widened to `large_list`.
    private func typeMatrixSchema(_ batch: MetalRecordBatch) -> ArrowIPCSchema {
        ArrowIPCSchema(fields: zip(batch.names, batch.columns).map { name, column in
            let field = ArrowIPCField(column: column, name: name)
            guard name == "biglist", case .list(let item) = field.type else { return field }
            return ArrowIPCField(name: name, type: .largeList(item))
        })
    }

    /// pyarrow reads every one of them back, in both encapsulations, with the right types and values.
    func testPyarrowReadsTheWholeTypeMatrix() throws {
        _ = try requirePython()
        let batch = try typeMatrixBatch()
        let schema = typeMatrixSchema(batch)
        let fileURL = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: streamURL)
        }
        try ArrowIPCWriter.write([batch], to: fileURL, schema: schema, format: .file)
        try ArrowIPCWriter.write([batch], to: streamURL, schema: schema, format: .stream)

        let script = """
        import sys, pyarrow as pa
        from decimal import Decimal

        def check(table, label):
            table.validate(full=True)
            assert table.num_rows == 4, (label, table.num_rows)
            f = table.schema.field

            assert f("dec128").type == pa.decimal128(18, 3), (label, f("dec128").type)
            assert table.column("dec128").to_pylist() == [Decimal("12.345"), None, Decimal("-67.890"), Decimal("0.000")]
            assert f("dec256").type == pa.decimal256(40, 2), (label, f("dec256").type)
            assert table.column("dec256").to_pylist() == [Decimal("0.01"), Decimal("-0.02"), None, Decimal("9999.99")]

            assert f("half").type == pa.float16(), (label, f("half").type)
            assert [None if v is None else float(v) for v in table.column("half").to_pylist()] == [1.5, None, -2.25, 0.0]

            assert f("fixed").type == pa.binary(3), (label, f("fixed").type)
            assert table.column("fixed").to_pylist() == [b"\\x01\\x02\\x03", None, b"\\x04\\x05\\x06", b"\\x00\\x00\\x00"]

            assert f("span").type == pa.month_day_nano_interval(), (label, f("span").type)
            assert table.column("span").to_pylist() == [pa.MonthDayNano([1, 2, 3]), None,
                                                        pa.MonthDayNano([-4, -5, -6]), pa.MonthDayNano([0, 0, 0])]

            assert f("nothing").type == pa.null(), (label, f("nothing").type)
            assert table.column("nothing").to_pylist() == [None, None, None, None]

            assert f("list").type == pa.list_(pa.field("item", pa.int32())), (label, f("list").type)
            assert table.column("list").to_pylist() == [[1, None], None, [], [7, 8, 9]]
            assert f("biglist").type == pa.large_list(pa.field("item", pa.int32())), (label, f("biglist").type)
            assert table.column("biglist").to_pylist() == [[1, None], None, [], [7, 8, 9]]
            assert f("grid").type == pa.list_(pa.field("item", pa.float64()), 3), (label, f("grid").type)
            assert table.column("grid").to_pylist() == [[0.0, 0.25, 0.5], None, [1.5, 1.75, 2.0], [2.25, 2.5, 2.75]]

            assert f("person").type == pa.struct([pa.field("n", pa.int64()), pa.field("s", pa.string())]), (label, f("person").type)
            assert table.column("person").to_pylist() == [{"n": 10, "s": "one"}, None, {"n": None, "s": "three"}, {"n": 40, "s": None}]

            assert f("lookup").type == pa.map_(pa.string(), pa.int32()), (label, f("lookup").type)
            assert table.column("lookup").to_pylist() == [[("a", 1), ("b", None)], None, [("c", 3)], [("d", 4)]]

            dense = f("dense").type
            assert pa.types.is_union(dense) and dense.mode == "dense", (label, dense)
            assert [dense.field(i).name for i in range(dense.num_fields)] == ["ints", "text"], (label, dense)
            assert list(dense.type_codes) == [0, 1], (label, dense)
            assert table.column("dense").to_pylist() == [10, "x", None, None], (label, table.column("dense").to_pylist())
            sparse = f("sparse").type
            assert pa.types.is_union(sparse) and sparse.mode == "sparse", (label, sparse)
            assert list(sparse.type_codes) == [3, 7], (label, sparse)
            assert table.column("sparse").to_pylist() == [1, "q", None, None], (label, table.column("sparse").to_pylist())

            runs = f("runs").type
            assert pa.types.is_run_end_encoded(runs), (label, runs)
            assert runs.run_end_type == pa.int32() and runs.value_type == pa.string(), (label, runs)
            assert table.column("runs").to_pylist() == ["aa", "aa", None, "cc"], (label, table.column("runs").to_pylist())

            tagged = f("tagged")
            assert tagged.type == pa.int64(), (label, tagged.type)
            assert tagged.metadata == {b"ARROW:extension:name": b"arrowmetal.test",
                                       b"ARROW:extension:metadata": b'{"k":1}'}, (label, tagged.metadata)
            assert table.column("tagged").to_pylist() == [7, None, 9, 11]

        with pa.ipc.open_file(sys.argv[1]) as r:
            check(r.read_all(), "file")
        with pa.ipc.open_stream(sys.argv[2]) as r:
            check(r.read_all(), "stream")
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// A slice at a non-zero offset writes the rows it names and nothing the parent holds before them:
    /// Arrow offsets are absolute in this package, so the naive write would carry the whole prefix.
    func testSlicedColumnsWriteOnlyTheRowsTheyName() throws {
        let text = try MetalStringArray((0..<10).map { i in i == 4 ? nil : String(repeating: "x", count: i + 1) })
        let list = try MetalListArray(counts: [1, 2, nil, 3, 0, 1, 2, 1, 1, 1],
                                      values: .int32(try MetalArray<Int32>((0..<12).map { Int32($0) })))
        let people = try MetalStructArray(names: ["n"],
                                          children: [.int64(try MetalArray<Int64>((0..<10).map { Int64($0) }))],
                                          valid: (0..<10).map { $0 != 3 })
        let batch = try MetalRecordBatch(names: ["text", "big", "list", "person"], columns: [
            .string(try text.slice(offset: 6, length: 3)),
            .string(try text.slice(offset: 6, length: 3)),
            .list(try list.slice(offset: 6, length: 3)),
            .structure(try people.slice(offset: 6, length: 3)),
        ])
        var fields = zip(batch.names, batch.columns).map { ArrowIPCField(column: $0.1, name: $0.0) }
        fields[1] = ArrowIPCField(name: "big", type: .largeUtf8)
        let schema = ArrowIPCSchema(fields: fields)

        // Our own reader takes the two utf8 columns back; the rest it still refuses (see the test below).
        let flat = try MetalRecordBatch(names: ["text", "big"], columns: [batch.columns[0], batch.columns[1]])
        let flatSchema = ArrowIPCSchema(fields: [fields[0], fields[1]])
        for format in [ArrowIPCFormat.stream, .file] {
            let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([flat], schema: flatSchema, format: format))
            let back = try reader.batch(at: 0)
            for name in ["text", "big"] {
                let column = try XCTUnwrap(back[name]?.asString)
                XCTAssertEqual(column.toArray(), ["xxxxxxx", "xxxxxxxx", "xxxxxxxxx"], name)
                // 7 + 8 + 9 bytes, not one of the 16 the first six rows occupy.
                XCTAssertEqual(column.totalBytes, 24, name)
            }
        }

        _ = try requirePython()
        let fileURL = temporaryFile()
        let streamURL = temporaryFile("arrows")
        defer {
            try? FileManager.default.removeItem(at: fileURL)
            try? FileManager.default.removeItem(at: streamURL)
        }
        try ArrowIPCWriter.write([batch], to: fileURL, schema: schema, format: .file)
        try ArrowIPCWriter.write([batch], to: streamURL, schema: schema, format: .stream)
        let script = """
        import sys, pyarrow as pa

        def check(table, label):
            table.validate(full=True)
            assert table.num_rows == 3, (label, table.num_rows)
            assert table.column("text").to_pylist() == ["xxxxxxx", "xxxxxxxx", "xxxxxxxxx"], (label, table.column("text").to_pylist())
            assert table.schema.field("big").type == pa.large_string(), (label, table.schema.field("big").type)
            assert table.column("big").to_pylist() == ["xxxxxxx", "xxxxxxxx", "xxxxxxxxx"], (label, table.column("big").to_pylist())
            assert table.column("list").to_pylist() == [[7, 8], [9], [10]], (label, table.column("list").to_pylist())
            assert table.column("person").to_pylist() == [{"n": 6}, {"n": 7}, {"n": 8}], (label, table.column("person").to_pylist())
            # The offsets are rebased, so the child holds only the elements these rows use.
            assert len(table.column("list").chunk(0).values) == 4, (label, len(table.column("list").chunk(0).values))

        with pa.ipc.open_file(sys.argv[1]) as r:
            check(r.read_all(), "file")
        with pa.ipc.open_stream(sys.argv[2]) as r:
            check(r.read_all(), "stream")
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// Zero rows, and rows that are all null, for every nested layout.
    func testEmptyAndAllNullNestedColumns() throws {
        _ = try requirePython()
        let emptyEntries = try MetalListArray(counts: [], values: .structure(
            try MetalStructArray(names: ["key", "value"], children: [
                .string(try MetalStringArray([])), .int32(try MetalArray<Int32>([])),
            ])))
        let empty = try MetalRecordBatch(names: ["list", "grid", "person", "lookup", "dense", "runs", "nothing"], columns: [
            .list(try MetalListArray(counts: [], values: .int32(try MetalArray<Int32>([])))),
            .list(try MetalListArray(counts: [], values: .int32(try MetalArray<Int32>([])), kind: .fixedSize(2))),
            .structure(try MetalStructArray(names: ["n"], children: [.int64(try MetalArray<Int64>([]))])),
            .map(try MetalMapArray(entries: emptyEntries)),
            .union(try MetalUnionArray(mode: .dense, length: 0, typeIds: try MetalArray<Int8>([]),
                                       offsets: try MetalArray<Int32>([]), typeCodes: [0], names: ["ints"],
                                       children: [.int32(try MetalArray<Int32>([]))])),
            .runEndEncoded(runEnds: try MetalArray<Int32>([]), values: .int64(try MetalArray<Int64>([]))),
            .null(MetalNullArray(length: 0)),
        ])
        XCTAssertEqual(empty.length, 0)
        // Every row null, at every level: a null list, a null struct and a null map.
        let allNull = try MetalRecordBatch(names: ["list", "person", "lookup"], columns: [
            .list(try MetalListArray(counts: [nil, nil], values: .int32(try MetalArray<Int32>([])))),
            .structure(try MetalStructArray(names: ["n"], children: [.int64(try MetalArray<Int64>([1, 2]))],
                                            valid: [false, false])),
            .map(try MetalMapArray(entries: try MetalListArray(counts: [nil, nil], values: .structure(
                try MetalStructArray(names: ["key", "value"], children: [
                    .string(try MetalStringArray([])), .int32(try MetalArray<Int32>([])),
                ]))))),
        ])

        let files = (0..<4).map { _ in temporaryFile() }
        defer { for url in files { try? FileManager.default.removeItem(at: url) } }
        try ArrowIPCWriter.write([empty], to: files[0], format: .file)
        try ArrowIPCWriter.write([empty], to: files[1], format: .stream)
        try ArrowIPCWriter.write([allNull], to: files[2], format: .file)
        try ArrowIPCWriter.write([allNull], to: files[3], format: .stream)
        let script = """
        import sys, pyarrow as pa

        def read(path, opener):
            with opener(path) as r:
                t = r.read_all()
            t.validate(full=True)
            return t

        for path, opener in ((sys.argv[1], pa.ipc.open_file), (sys.argv[2], pa.ipc.open_stream)):
            t = read(path, opener)
            assert t.num_rows == 0, (path, t.num_rows)
            for name in t.schema.names:
                assert t.column(name).to_pylist() == [], (path, name)

        for path, opener in ((sys.argv[3], pa.ipc.open_file), (sys.argv[4], pa.ipc.open_stream)):
            t = read(path, opener)
            assert t.num_rows == 2, (path, t.num_rows)
            assert t.column("list").to_pylist() == [None, None], (path, t.column("list").to_pylist())
            assert t.column("person").to_pylist() == [None, None], (path, t.column("person").to_pylist())
            assert t.column("lookup").to_pylist() == [None, None], (path, t.column("lookup").to_pylist())
        print("ok")
        """
        XCTAssertEqual(try runPython(script, files.map(\.path)).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// Several batches of nested columns in one file: each body is found through the footer's block
    /// offsets, and a zero-row batch between two full ones changes nothing.
    func testSeveralNestedBatchesInOneFile() throws {
        _ = try requirePython()
        func batch(_ counts: [Int?], _ values: [Int32?], _ names: [String?]) throws -> MetalRecordBatch {
            try MetalRecordBatch(names: ["list", "person"], columns: [
                .list(try MetalListArray(counts: counts, values: .int32(try MetalArray<Int32>(values)))),
                .structure(try MetalStructArray(names: ["s"], children: [.string(try MetalStringArray(names))])),
            ])
        }
        let batches = [try batch([1, nil], [5], ["a", nil]),
                       try batch([], [], []),
                       try batch([0, 2, 1], [6, 7, 8], [nil, "b", "c"])]
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

        def check(table, label):
            table.validate(full=True)
            assert table.column("list").to_pylist() == [[5], None, [], [6, 7], [8]], (label, table.column("list").to_pylist())
            assert table.column("person").to_pylist() == [{"s": "a"}, {"s": None}, {"s": None}, {"s": "b"}, {"s": "c"}], \\
                (label, table.column("person").to_pylist())

        with pa.ipc.open_file(sys.argv[1]) as r:
            assert r.num_record_batches == 3, r.num_record_batches
            assert [r.get_batch(i).num_rows for i in range(3)] == [2, 0, 3]
            check(r.read_all(), "file")
        with pa.ipc.open_stream(sys.argv[2]) as r:
            check(r.read_all(), "stream")
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [fileURL.path, streamURL.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// The other two interval units and the two narrow decimals.
    ///
    /// pyarrow 25 reads `decimal32` / `decimal64` values. It reads the *type* of `interval[year_month]`
    /// and `interval[day_time]` and validates their buffers in full, but its Python layer has no array
    /// class for either, so those two are checked as far as pyarrow can go and no further;
    /// `interval[month_day_nano]` is compared value by value in `testPyarrowReadsTheWholeTypeMatrix`.
    func testIntervalUnitsAndNarrowDecimals() throws {
        _ = try requirePython()
        let batch = try MetalRecordBatch(names: ["months", "daytime", "dec32", "dec64"], columns: [
            .interval(try MetalIntervalArray(unit: .months, [ArrowInterval(months: 13), nil, ArrowInterval(months: -2)])),
            .interval(try MetalIntervalArray(unit: .dayTime, [
                ArrowInterval(days: 2, nanoseconds: 3_000_000), nil, ArrowInterval(days: -1, nanoseconds: -4_000_000),
            ])),
            .smallDecimal(try MetalSmallDecimalArray(type: try ArrowSmallDecimalType(precision: 9, scale: 2, bitWidth: 32),
                                                     [123, nil, -456])),
            .smallDecimal(try MetalSmallDecimalArray(type: try ArrowSmallDecimalType(precision: 18, scale: 4, bitWidth: 64),
                                                     [1_234_567, nil, -89])),
        ])
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        try ArrowIPCWriter.write([batch], to: url, format: .file)
        let script = """
        import sys, pyarrow as pa
        from decimal import Decimal
        with pa.ipc.open_file(sys.argv[1]) as r:
            t = r.read_all()
        t.validate(full=True)
        assert t.schema.field("dec32").type == pa.decimal32(9, 2), t.schema.field("dec32").type
        assert t.column("dec32").to_pylist() == [Decimal("1.23"), None, Decimal("-4.56")], t.column("dec32").to_pylist()
        assert t.schema.field("dec64").type == pa.decimal64(18, 4), t.schema.field("dec64").type
        assert t.column("dec64").to_pylist() == [Decimal("123.4567"), None, Decimal("-0.0089")], t.column("dec64").to_pylist()
        assert str(t.schema.field("months").type) == "month_interval", t.schema.field("months").type
        assert str(t.schema.field("daytime").type) == "day_time_interval", t.schema.field("daytime").type
        # pyarrow has no Python array class for these two, so this is as far as it reads them.
        assert t.column("months").null_count == 1 and t.column("daytime").null_count == 1
        assert t.num_rows == 3, t.num_rows
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [url.path]).trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    /// The reader has not grown with the writer: a file carrying a type it cannot build is refused with
    /// a message naming the column, never read as something else.
    func testOurReaderStillRefusesTheTypesItCannotBuild() throws {
        let batch = try typeMatrixBatch()
        for (i, name) in batch.names.enumerated() where name != "tagged" {
            let one = try MetalRecordBatch(names: [name], columns: [batch.columns[i]])
            let data = try ArrowIPCWriter.encode([one], format: .file)
            XCTAssertThrowsError(try ArrowIPCReader(data: data), name) { error in
                guard let e = error as? ArrowIPCError, case .unsupported(let message) = e else {
                    return XCTFail("column '\(name)' failed with \(error)")
                }
                XCTAssertTrue(message.contains(name), "column '\(name)': \(message)")
            }
        }
        // An extension column is its storage type plus metadata, so that one does read back.
        let tagged = try MetalRecordBatch(names: ["tagged"], columns: [batch.columns[batch.names.count - 1]])
        let back = try ArrowIPCReader(data: try ArrowIPCWriter.encode([tagged], format: .file)).batch(at: 0)
        XCTAssertEqual(back["tagged"]?.asInt64?.toArray(), [7, nil, 9, 11])
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
