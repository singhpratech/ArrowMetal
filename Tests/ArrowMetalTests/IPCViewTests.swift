import XCTest
@testable import ArrowMetal

/// Arrow IPC beyond the classic layouts: the view types (`utf8_view`, `binary_view`, `list_view`,
/// `large_list_view`), big-endian sources, the `arrow.fixed_shape_tensor` extension type, and the errors
/// for what stays out of reach (IPC tensor messages, offsets past 2 GB).
///
/// pyarrow 25 is the oracle: it writes the view-typed and tensor inputs, and reads back what ArrowMetal
/// writes. The big-endian inputs are Arrow's own integration files (`Tests/Fixtures/ipc/README.md`).
final class IPCViewTests: XCTestCase {

    // MARK: - helpers

    private static let fixtures = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures").appendingPathComponent("ipc")

    /// A python interpreter with pyarrow: `ARROWMETAL_PYTHON` if it is set and usable, else one on PATH.
    private static let python: String? = {
        func works(_ path: String) -> Bool {
            guard FileManager.default.isExecutableFile(atPath: path) else { return false }
            let p = Process()
            p.executableURL = URL(fileURLWithPath: path)
            p.arguments = ["-c", "import pyarrow, numpy"]
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

    private func temporaryFile(_ ext: String = "arrow") -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("arrowmetal-ipcview-\(UUID().uuidString).\(ext)")
    }

    @discardableResult
    private func runPython(_ script: String, _ arguments: [String]) throws -> String {
        guard let python = IPCViewTests.python else {
            throw XCTSkip("no python3 with pyarrow and numpy found (set ARROWMETAL_PYTHON to one)")
        }
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
        return String(decoding: stdout, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A canonical text form of every value in a column, nested children included.
    private func render(_ column: AnyMetalArray) -> String {
        (0..<column.length).map { renderValue(column, $0) }.joined(separator: ", ")
    }

    private func renderValue(_ c: AnyMetalArray, _ i: Int) -> String {
        func hex(_ bytes: [UInt8]?) -> String {
            bytes.map { $0.map { String(format: "%02x", $0) }.joined() } ?? "null"
        }
        switch c {
        case .int8(let a): return a[i].map { "\($0)" } ?? "null"
        case .uint8(let a): return a[i].map { "\($0)" } ?? "null"
        case .int16(let a): return a[i].map { "\($0)" } ?? "null"
        case .uint16(let a): return a[i].map { "\($0)" } ?? "null"
        case .int32(let a): return a[i].map { "\($0)" } ?? "null"
        case .uint32(let a): return a[i].map { "\($0)" } ?? "null"
        case .int64(let a): return a[i].map { "\($0)" } ?? "null"
        case .uint64(let a): return a[i].map { "\($0)" } ?? "null"
        case .float32(let a): return a[i].map { $0.isNaN ? "nan" : "\($0)" } ?? "null"
        case .float64(let a): return a[i].map { $0.isNaN ? "nan" : "\($0)" } ?? "null"
        case .boolean(let a): return a[i].map { "\($0)" } ?? "null"
        case .string(let a): return a[i].map { "\"\($0)\"" } ?? "null"
        case .binary(let a): return hex(a.bytes(at: i))
        case .temporal(let a): return a[i].map { "\($0)" } ?? "null"
        // The raw little-endian two's-complement bytes: decimal256 values need not fit 128 bits.
        case .decimal(let a): return a[i] == nil ? "null" : hex(a.rawBytes(at: i))
        case .smallDecimal(let a): return a[i].map { "\($0)" } ?? "null"
        case .float16(let a): return a[i].map { "\($0)" } ?? "null"
        case .fixedBinary(let a): return hex(a.bytes(at: i))
        case .interval(let a): return a[i].map { "\($0)" } ?? "null"
        case .null: return "null"
        case .dictionary(let codes, let values): return codes[i].map { renderValue(values, Int($0)) } ?? "null"
        case .list(let a):
            guard let r = a.valueRange(i) else { return "null" }
            return "[" + r.map { renderValue(a.values, $0) }.joined(separator: ", ") + "]"
        case .structure(let a):
            guard a.isValid(i) else { return "null" }
            return "{" + zip(a.names, a.children).map { "\($0.0): \(renderValue($0.1, i))" }.joined(separator: ", ") + "}"
        case .map(let a):
            guard let r = a.valueRange(i) else { return "null" }
            return "{" + r.map { "\(renderValue(a.keys, $0)): \(renderValue(a.items, $0))" }.joined(separator: ", ") + "}"
        case .union(let a):
            guard let where_ = a.location(i) else { return "?" }
            return "\(a.names[where_.child])=\(renderValue(a.children[where_.child], where_.index))"
        case .runEndEncoded(let runEnds, let values):
            var run = 0
            while run < runEnds.length, Int(runEnds[run] ?? 0) <= i { run += 1 }
            return run < runEnds.length ? renderValue(values, run) : "?"
        case .extended(let a): return "<\(a.name)>" + renderValue(a.storage, i)
        }
    }

    // MARK: - view types

    /// The pyarrow script that writes every view type: flat, nested in a struct and in a list, with nulls,
    /// with a list view whose rows are out of order and overlap, and a second batch that is a slice.
    private static let viewScript = """
    import sys, pyarrow as pa

    N = 40
    long = "a string longer than twelve bytes, row %d"
    sv = pa.array([None if i % 5 == 1 else ("s%d" % i if i % 3 else long % i) for i in range(N)], pa.string_view())
    bv = pa.array([None if i % 4 == 2 else bytes([i % 256]) * (i % 20) for i in range(N)], pa.binary_view())
    child = pa.array([None if k % 9 == 4 else k * 10 for k in range(70)], pa.int32())
    # Out of order and overlapping: the reader has to gather these rows from the child.
    lv = pa.ListViewArray.from_arrays(pa.array([(i * 7) % 60 for i in range(N)], pa.int32()),
                                      pa.array([i % 5 for i in range(N)], pa.int32()), child,
                                      mask=pa.array([i % 6 == 3 for i in range(N)]))
    # In order and consecutive (a list cast to a list view): the reader keeps the child as it is.
    llv = pa.array([None if i % 7 == 2 else ["w%d" % k for k in range(i % 4)] for i in range(N)],
                   pa.large_list_view(pa.string()))
    lv64 = pa.ListViewArray.from_arrays(pa.array([(N - 1 - i) * 2 for i in range(N)], pa.int32()),
                                        pa.array([2 if i % 3 else 0 for i in range(N)], pa.int32()),
                                        pa.array(range(2 * N), pa.int64()))
    st = pa.StructArray.from_arrays([sv, lv64], names=["a", "b"], mask=pa.array([i % 8 == 5 for i in range(N)]))
    ls = pa.array([None if i % 6 == 0 else [None if (i + k) % 4 == 0 else long % (i + k) for k in range(i % 3)]
                   for i in range(N)], pa.list_(pa.string_view()))
    lvs = pa.ListViewArray.from_arrays(pa.array([(i * 3) % 30 for i in range(N)], pa.int32()),
                                       pa.array([(i % 3) for i in range(N)], pa.int32()),
                                       pa.array([None if k % 7 == 0 else ("x%d" % k) * (k % 5) for k in range(40)],
                                                pa.string_view()))
    names = ["sv", "bv", "lv", "llv", "st", "ls", "lvs"]
    full = pa.record_batch([sv, bv, lv, llv, st, ls, lvs], names=names)
    batches = [full, full.slice(3, 25)]
    with pa.ipc.new_file(sys.argv[1], full.schema) as w:
        for b in batches: w.write_batch(b)
    with pa.ipc.new_stream(sys.argv[2], full.schema) as w:
        for b in batches: w.write_batch(b)
    opts = pa.ipc.IpcWriteOptions(compression="lz4")
    with pa.ipc.new_stream(sys.argv[3], full.schema, options=opts) as w:
        for b in batches: w.write_batch(b)
    print("ok")
    """

    /// pyarrow writes each view type in the file, stream and LZ4-compressed stream formats; ArrowMetal
    /// reads them into the classic layouts, writes them back (as classic types), and pyarrow finds every
    /// value it wrote.
    func testViewTypesPyarrowWritesRoundTrip() throws {
        let file = temporaryFile(), stream = temporaryFile("arrows"), lz4 = temporaryFile("arrows")
        defer { for u in [file, stream, lz4] { try? FileManager.default.removeItem(at: u) } }
        XCTAssertEqual(try runPython(Self.viewScript, [file.path, stream.path, lz4.path]), "ok")

        var outputs: [URL] = []
        defer { for u in outputs { try? FileManager.default.removeItem(at: u) } }
        for (url, format) in [(file, ArrowIPCFormat.file), (stream, .stream), (lz4, .stream)] {
            let reader = try ArrowIPCReader(url: url)
            XCTAssertEqual(reader.format, format)
            XCTAssertFalse(reader.isBigEndian)
            XCTAssertEqual(reader.schema.names, ["sv", "bv", "lv", "llv", "st", "ls", "lvs"])
            XCTAssertEqual(reader.schema["sv"]?.type, .utf8View)
            XCTAssertEqual(reader.schema["bv"]?.type, .binaryView)
            XCTAssertEqual(reader.schema["lv"]?.type, .listView(ArrowIPCField(name: "item", type: .int(bits: 32, signed: true))))
            XCTAssertEqual(reader.schema["llv"]?.type, .largeListView(ArrowIPCField(name: "item", type: .utf8)))
            XCTAssertEqual(reader.schema["ls"]?.type, .list(ArrowIPCField(name: "item", type: .utf8View)))
            XCTAssertEqual(reader.schema["st"]?.type.classic, .structure([
                ArrowIPCField(name: "a", type: .utf8),
                ArrowIPCField(name: "b", type: .list(ArrowIPCField(name: "item", type: .int(bits: 64, signed: true)))),
            ]))
            let batches = try reader.readAll()
            XCTAssertEqual(batches.map(\.length), [40, 25])
            let full = batches[0], sliced = batches[1]
            // Materialised to the engine's own layouts.
            XCTAssertNotNil(full["sv"]?.asString)
            XCTAssertNotNil(full["bv"]?.asBinary)
            let lv = try XCTUnwrap(full["lv"]?.asList)
            XCTAssertEqual(lv.kind, .variable)
            // The out-of-order rows were gathered: the child holds exactly the elements the rows cover.
            let covered = (0..<40).filter { $0 % 6 != 3 }.map { $0 % 5 }.reduce(0, +)
            XCTAssertEqual(lv.values.length, covered)
            XCTAssertEqual(renderValue(.list(lv), 3), "null")
            XCTAssertEqual(renderValue(.list(lv), 8), "[560, 570, null]")          // offset 56, size 3
            XCTAssertEqual(renderValue(full["sv"]!, 0), "\"a string longer than twelve bytes, row 0\"")
            XCTAssertEqual(renderValue(full["sv"]!, 1), "null")
            XCTAssertEqual(renderValue(full["sv"]!, 2), "\"s2\"")
            XCTAssertEqual(renderValue(full["bv"]!, 5), "0505050505")
            XCTAssertEqual(full["sv"]?.nullCount, 8)
            // A sliced batch reads as rows 3 ..< 28 of the full one.
            for name in full.names {
                let whole = try XCTUnwrap(full[name]), part = try XCTUnwrap(sliced[name])
                XCTAssertEqual(render(part), (3..<28).map { renderValue(whole, $0) }.joined(separator: ", "),
                               "column '\(name)' (\(url.lastPathComponent))")
            }
            // Written back: the reader's own schema (view types) writes the classic types.
            let out = temporaryFile()
            outputs.append(out)
            try ArrowIPCWriter.write(batches, to: out, schema: reader.schema)
        }

        let check = """
        import sys, pyarrow as pa
        exec(open(sys.argv[1]).read().split("names = [")[0])
        want = pa.Table.from_batches([b for b in [pa.record_batch([sv, bv, lv, llv, st, ls, lvs],
                  names=["sv", "bv", "lv", "llv", "st", "ls", "lvs"])]])
        want = pa.concat_tables([want, want.slice(3, 25)])
        classic = {"sv": pa.string(), "bv": pa.binary(), "lv": pa.list_(pa.int32()),
                   "llv": pa.large_list(pa.string()),
                   "st": pa.struct([("a", pa.string()), ("b", pa.list_(pa.int64()))]),
                   "ls": pa.list_(pa.string()), "lvs": pa.list_(pa.string())}
        for path in sys.argv[2:]:
            got = pa.ipc.open_file(path).read_all()
            got.validate(full=True)
            for name, t in classic.items():
                assert got.schema.field(name).type == t, (name, got.schema.field(name).type)
                assert got.column(name).to_pylist() == want.column(name).to_pylist(), name
        print("ok")
        """
        let scriptFile = temporaryFile("py")
        defer { try? FileManager.default.removeItem(at: scriptFile) }
        try Self.viewScript.write(to: scriptFile, atomically: true, encoding: .utf8)
        XCTAssertEqual(try runPython(check, [scriptFile.path] + outputs.map(\.path)), "ok")
    }

    /// Batches past one 64K-row block, which the reader materialises on several cores at once: strings
    /// of every length class (empty, inline, short and long out-of-line, null), blocks that end in
    /// empty and null rows, and an out-of-order list view whose gather indices are written block by block.
    func testLargeViewBatchesSpanSeveralBlocks() throws {
        let input = temporaryFile("arrows"), output = temporaryFile()
        defer { for u in [input, output] { try? FileManager.default.removeItem(at: u) } }
        let script = """
        import sys, numpy as np, pyarrow as pa
        n = 200_003
        lengths = [0, 3, 12, 13, 31, 32, 33, 200]
        def text(i):
            # The last rows of each 64K block are empty or null, so a block's last bytes belong to a short row.
            if i % 65536 >= 65533:
                return [None, "", "ab"][i % 3]
            return None if i % 11 == 5 else (("%07d" % i) * 40)[:lengths[(i * 7) % 8]]
        sv = pa.array([text(i) for i in range(n)], pa.string_view())
        rng = np.random.default_rng(3)
        sizes = rng.integers(0, 6, n).astype(np.int32)
        offsets = rng.integers(0, 500_000 - 6, n).astype(np.int32)
        lv = pa.ListViewArray.from_arrays(pa.array(offsets), pa.array(sizes), pa.array(np.arange(500_000, dtype=np.int64)),
                                          mask=pa.array([i % 13 == 0 for i in range(n)]))
        batch = pa.record_batch([sv, lv], names=["sv", "lv"])
        with pa.ipc.new_stream(sys.argv[1], batch.schema) as w:
            w.write_batch(batch)
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [input.path]), "ok")
        let batches = try ArrowIPCReader(url: input).readAll()
        XCTAssertEqual(batches.map(\.length), [200_003])
        try ArrowIPCWriter.write(batches, to: output)
        let check = """
        import sys, pyarrow as pa
        src = pa.ipc.open_stream(sys.argv[1]).read_all()
        got = pa.ipc.open_file(sys.argv[2]).read_all()
        assert got.column("sv").type == pa.string()
        assert got.column("sv").equals(src.column("sv").cast(pa.string()))
        assert got.column("lv").type == pa.list_(pa.int64())
        assert got.column("lv").to_pylist() == src.column("lv").to_pylist()
        print("ok")
        """
        XCTAssertEqual(try runPython(check, [input.path, output.path]), "ok")
    }

    /// The writer never writes a view type, even when the schema it is handed names one.
    func testViewTypesInAnExplicitSchemaWriteClassicTypes() throws {
        let batch = try MetalRecordBatch(names: ["s"], columns: [.string(try MetalStringArray(["a", nil, "bcd"]))])
        let schema = ArrowIPCSchema(fields: [ArrowIPCField(name: "s", type: .utf8View)])
        let reader = try ArrowIPCReader(data: try ArrowIPCWriter.encode([batch], schema: schema, format: .stream))
        XCTAssertEqual(reader.schema["s"]?.type, .utf8)
        XCTAssertEqual(render(try reader.batch(at: 0)["s"]!), "\"a\", null, \"bcd\"")
        XCTAssertEqual(ArrowIPCType.largeListView(ArrowIPCField(name: "item", type: .binaryView)).classic,
                       .largeList(ArrowIPCField(name: "item", type: .binary)))
    }

    // MARK: - big-endian

    private static let integrationNames = [
        "custom_metadata", "datetime", "dictionary", "dictionary_unsigned", "extension", "interval", "map",
        "nested", "nested_large_offsets", "null", "primitive", "primitive_large_offsets", "primitive_zerolength",
        "recursive_nested", "union",
    ]

    /// Arrow's big-endian integration files read to exactly the values of their little-endian twins, in
    /// both the file and the stream encapsulation: integers, floats, offsets, dates, times, timestamps,
    /// intervals, dictionaries, lists, large lists, structs, maps, unions and nulls.
    func testBigEndianIntegrationFilesMatchTheirLittleEndianTwins() throws {
        let dir = Self.fixtures
        for name in Self.integrationNames {
            let little = try ArrowIPCReader(url: dir.appendingPathComponent("littleendian/generated_\(name).stream"))
            XCTAssertFalse(little.isBigEndian, name)
            let want = try little.readAll()
            for ext in ["arrow_file", "stream"] {
                let big = try ArrowIPCReader(url: dir.appendingPathComponent("bigendian/generated_\(name).\(ext)"))
                XCTAssertTrue(big.isBigEndian, "\(name).\(ext)")
                XCTAssertEqual(big.schema, little.schema, "\(name).\(ext)")
                let got = try big.readAll()
                XCTAssertEqual(got.count, want.count, "\(name).\(ext)")
                for (g, w) in zip(got, want) {
                    XCTAssertEqual(g.length, w.length, "\(name).\(ext)")
                    for (i, column) in w.names.enumerated() {
                        XCTAssertEqual(render(g.columns[i]), render(w.columns[i]), "\(name).\(ext) column '\(column)'")
                        XCTAssertEqual(g.columns[i].nullCount, w.columns[i].nullCount, "\(name).\(ext) column '\(column)'")
                    }
                }
            }
        }
    }

    /// decimal128 and decimal256 from big-endian files: each value is its limbs byte swapped and put back
    /// in little-endian order. pyarrow, reading the same file, gives the same unscaled integers (compared
    /// as their little-endian two's-complement bytes).
    func testBigEndianDecimalsMatchPyarrow() throws {
        let script = """
        import sys, pyarrow as pa
        import decimal
        wide = decimal.Context(prec=100)
        for batch in pa.ipc.open_stream(sys.argv[1]):
            for col in batch.columns:
                s = col.type.scale
                width = col.type.bit_width // 8
                print(", ".join("null" if v is None else
                                (int(v.scaleb(s, context=wide)) % (1 << (8 * width))).to_bytes(width, "little").hex()
                                for v in col.to_pylist()))
        """
        for name in ["decimal", "decimal256"] {
            let url = Self.fixtures.appendingPathComponent("bigendian/generated_\(name).stream")
            let reader = try ArrowIPCReader(url: url)
            XCTAssertTrue(reader.isBigEndian)
            let rendered = try reader.readAll().flatMap { b in b.columns.map(render) }
            let want = try runPython(script, [url.path]).components(separatedBy: "\n")
            XCTAssertEqual(rendered.count, want.count, name)
            for (i, (g, w)) in zip(rendered, want).enumerated() { XCTAssertEqual(g, w, "\(name) column \(i)") }
        }
    }

    /// The byte swaps on their own: every element width, decimals, month_day_nano intervals, and inline
    /// and out-of-line views.
    func testByteSwapKinds() {
        func swapped(_ bytes: [UInt8], _ kind: ArrowIPCByteSwap.Kind) -> [UInt8] {
            var b = bytes
            b.withUnsafeMutableBufferPointer { ArrowIPCByteSwap.swap($0.baseAddress!, byteCount: $0.count, kind) }
            return b
        }
        XCTAssertEqual(swapped([1, 2, 3, 4], .width(2)), [2, 1, 4, 3])
        XCTAssertEqual(swapped([1, 2, 3, 4, 5, 6, 7, 8], .width(4)), [4, 3, 2, 1, 8, 7, 6, 5])
        XCTAssertEqual(swapped(Array(1...8), .width(8)), Array((1...8).reversed()))
        XCTAssertEqual(swapped(Array(1...16), .reverse(16)), Array((1...16).reversed()))
        XCTAssertEqual(swapped(Array(1...32), .reverse(32)), Array((1...32).reversed()))
        XCTAssertEqual(swapped(Array(1...16), .monthDayNano), [4, 3, 2, 1, 8, 7, 6, 5, 16, 15, 14, 13, 12, 11, 10, 9])
        // Inline view (length 5, big-endian): only the length is swapped.
        let inline: [UInt8] = [0, 0, 0, 5] + Array("hello".utf8) + [0, 0, 0, 0, 0, 0, 0]
        XCTAssertEqual(swapped(inline, .views), [5, 0, 0, 0] + Array("hello".utf8) + [0, 0, 0, 0, 0, 0, 0])
        // Out-of-line view (length 20): length, buffer index and offset swapped; the prefix kept.
        let outOfLine: [UInt8] = [0, 0, 0, 20] + Array("pref".utf8) + [0, 0, 0, 1] + [0, 0, 1, 2]
        XCTAssertEqual(swapped(outOfLine, .views), [20, 0, 0, 0] + Array("pref".utf8) + [1, 0, 0, 0] + [2, 1, 0, 0])
        XCTAssertEqual(swapped([9, 8, 7], .none), [9, 8, 7])
    }

    // MARK: - fixed_shape_tensor

    /// pyarrow's FixedShapeTensorArray reads as an `arrow.fixed_shape_tensor` extension column with its
    /// metadata byte for byte, compute runs on the storage, and pyarrow reads what ArrowMetal writes
    /// back as the same tensor type with the same values.
    func testFixedShapeTensorRoundTripsWithPyarrow() throws {
        let file = temporaryFile(), stream = temporaryFile("arrows")
        defer { for u in [file, stream] { try? FileManager.default.removeItem(at: u) } }
        let write = """
        import sys, numpy as np, pyarrow as pa
        plain = pa.FixedShapeTensorArray.from_numpy_ndarray(np.arange(24, dtype=np.float32).reshape(4, 2, 3))
        t = pa.fixed_shape_tensor(pa.int64(), [2, 2], dim_names=["r", "c"], permutation=[1, 0])
        storage = pa.array([[1, 2, 3, 4], None, [5, 6, 7, 8], [9, 10, 11, 12]], pa.list_(pa.int64(), 4))
        named = pa.ExtensionArray.from_storage(t, storage)
        nested = pa.StructArray.from_arrays([plain], names=["t"])
        batch = pa.record_batch([plain, named, nested], names=["plain", "named", "nested"])
        with pa.ipc.new_file(sys.argv[1], batch.schema) as w: w.write_batch(batch)
        with pa.ipc.new_stream(sys.argv[2], batch.schema) as w: w.write_batch(batch)
        print("ok")
        """
        XCTAssertEqual(try runPython(write, [file.path, stream.path]), "ok")

        var outputs: [URL] = []
        defer { for u in outputs { try? FileManager.default.removeItem(at: u) } }
        for url in [file, stream] {
            let reader = try ArrowIPCReader(url: url)
            XCTAssertEqual(reader.schema["plain"]?.extensionName, "arrow.fixed_shape_tensor")
            let batch = try reader.batch(at: 0)
            let plain = try XCTUnwrap(batch["plain"]), named = try XCTUnwrap(batch["named"])
            XCTAssertEqual(plain.extensionName, ArrowFixedShapeTensorType.extensionName)
            // pyarrow's from_numpy_ndarray spells out the identity permutation.
            XCTAssertEqual(plain.extensionMetadata.map { String(decoding: $0, as: UTF8.self) },
                           "{\"shape\":[2,3],\"permutation\":[0,1]}")
            XCTAssertEqual(plain.fixedShapeTensorType, try ArrowFixedShapeTensorType(shape: [2, 3], permutation: [0, 1]))
            XCTAssertEqual(named.extensionMetadata.map { String(decoding: $0, as: UTF8.self) },
                           "{\"shape\":[2,2],\"permutation\":[1,0],\"dim_names\":[\"r\",\"c\"]}")
            let type = try XCTUnwrap(named.fixedShapeTensorType)
            XCTAssertEqual(type.shape, [2, 2])
            XCTAssertEqual(type.dimNames, ["r", "c"])
            XCTAssertEqual(type.permutation, [1, 0])
            XCTAssertEqual(type.metadata, named.extensionMetadata)
            XCTAssertEqual(render(named), "<arrow.fixed_shape_tensor>[1, 2, 3, 4], <arrow.fixed_shape_tensor>null, "
                           + "<arrow.fixed_shape_tensor>[5, 6, 7, 8], <arrow.fixed_shape_tensor>[9, 10, 11, 12]")
            XCTAssertEqual(named.nullCount, 1)
            // Compute runs on the storage: a fixed_size_list whose child holds every element.
            let elements = try plain.storageArray.listFlatten()
            XCTAssertEqual(try XCTUnwrap(elements.asFloat32).sum()?.asDouble, Double((0..<24).reduce(0, +)))
            XCTAssertEqual(try named.storageArray.listValueLength().toArray(), [4, nil, 4, 4])
            guard case .structure(let nested) = try XCTUnwrap(batch["nested"]) else { return XCTFail("nested") }
            XCTAssertEqual(nested.children[0].extensionName, ArrowFixedShapeTensorType.extensionName)
            let out = temporaryFile()
            outputs.append(out)
            try ArrowIPCWriter.write([batch], to: out)
        }
        // A tensor column built in Swift.
        let values = MetalArray<Double>(length: 12, nullCount: 0, validity: nil,
                                        values: try MetalArrowBuffer.allocate(byteCount: 96))
        for i in 0..<12 { values.values.mutableTyped(Double.self)[i] = Double(i) / 4 }
        let storage = try MetalListArray(counts: [6, 6], values: .float64(values), kind: .fixedSize(6))
        let built = try AnyMetalArray.fixedShapeTensor(storage, type: try ArrowFixedShapeTensorType(shape: [3, 2], dimNames: ["x", "y"]))
        XCTAssertThrowsError(try AnyMetalArray.fixedShapeTensor(storage, type: try ArrowFixedShapeTensorType(shape: [4])))
        let own = temporaryFile()
        outputs.append(own)
        try ArrowIPCWriter.write([try MetalRecordBatch(names: ["built"], columns: [built])], to: own, format: .stream)

        let check = """
        import sys, numpy as np, pyarrow as pa
        plain = pa.FixedShapeTensorArray.from_numpy_ndarray(np.arange(24, dtype=np.float32).reshape(4, 2, 3))
        t = pa.fixed_shape_tensor(pa.int64(), [2, 2], dim_names=["r", "c"], permutation=[1, 0])
        named = pa.ExtensionArray.from_storage(t, pa.array([[1, 2, 3, 4], None, [5, 6, 7, 8], [9, 10, 11, 12]],
                                                           pa.list_(pa.int64(), 4)))
        for path in sys.argv[1:3]:
            got = pa.ipc.open_file(path).read_all()
            assert isinstance(got.column("plain").type, pa.FixedShapeTensorType), got.schema
            assert got.column("plain").type == plain.type
            assert got.column("named").type == t, got.column("named").type
            assert got.column("named").type.dim_names == ["r", "c"] and got.column("named").type.permutation == [1, 0]
            assert got.column("plain").combine_chunks().equals(plain)
            assert got.column("named").combine_chunks().equals(named)
            np.testing.assert_array_equal(got.column("plain").combine_chunks().to_numpy_ndarray(),
                                          np.arange(24, dtype=np.float32).reshape(4, 2, 3))
            assert got.column("nested").type.field("t").type == plain.type
        built = pa.ipc.open_stream(sys.argv[3]).read_all().column("built").combine_chunks()
        assert built.type == pa.fixed_shape_tensor(pa.float64(), [3, 2], dim_names=["x", "y"]), built.type
        np.testing.assert_array_equal(built.to_numpy_ndarray(), (np.arange(12) / 4).reshape(2, 3, 2))
        print("ok")
        """
        XCTAssertEqual(try runPython(check, outputs.map(\.path)), "ok")
    }

    /// A tensor column whose metadata disagrees with its storage is malformed, not silently accepted.
    func testFixedShapeTensorMetadataIsCheckedAgainstItsStorage() throws {
        let values = MetalArray<Int32>(length: 4, nullCount: 0, validity: nil,
                                       values: try MetalArrowBuffer.allocate(byteCount: 16))
        let storage = try MetalListArray(counts: [4], values: .int32(values), kind: .fixedSize(4))
        let column = AnyMetalArray.extended(MetalExtensionArray(storage: .list(storage), name: "arrow.fixed_shape_tensor",
                                                                metadata: Array("{\"shape\":[3]}".utf8)))
        let data = try ArrowIPCWriter.encode([try MetalRecordBatch(names: ["t"], columns: [column])], format: .stream)
        XCTAssertThrowsError(try ArrowIPCReader(data: data).batch(at: 0)) { error in
            XCTAssertTrue("\(error)".contains("shape [3] (3 elements) over a fixed_size_list of 4"), "\(error)")
        }
        XCTAssertThrowsError(try ArrowFixedShapeTensorType(shape: [2, 2], permutation: [0, 0]))
        XCTAssertThrowsError(try ArrowFixedShapeTensorType(shape: [2, 2], dimNames: ["a"]))
        XCTAssertThrowsError(try ArrowFixedShapeTensorType(metadata: Array("{\"dim_names\":[]}".utf8)))
    }

    // MARK: - refused

    /// IPC Tensor and SparseTensor messages are refused with an error that names them, in a stream on
    /// their own (what `pyarrow.ipc.write_tensor` writes) and after a schema.
    func testTensorMessagesAreRefused() throws {
        let expected = "IPC Tensor and SparseTensor messages are not read"
        let url = temporaryFile("tensor")
        defer { try? FileManager.default.removeItem(at: url) }
        let script = """
        import sys, numpy as np, pyarrow as pa
        with pa.OSFile(sys.argv[1], "wb") as f:
            pa.ipc.write_tensor(pa.Tensor.from_numpy(np.arange(6, dtype=np.int32).reshape(2, 3)), f)
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [url.path]), "ok")
        XCTAssertThrowsError(try ArrowIPCReader(url: url)) { error in
            XCTAssertTrue("\(error)".contains(expected), "\(error)")
        }

        let batch = try MetalRecordBatch(names: ["x"], columns: [.int32(try MetalArray<Int32>([1, 2, 3]))])
        for header in [FBMessageHeader.tensor, .sparseTensor] {
            var stream = try ArrowIPCWriter.encode([batch], format: .stream)
            stream.removeLast(8)                                     // the end-of-stream marker
            let b = FBBuilder()
            b.startObject(0)
            let table = b.endObject()
            b.startObject(5)
            b.addScalar(id: 0, fbMetadataVersionV5, default: 0)
            b.addScalar(id: 1, header.rawValue, default: 0)
            b.addOffset(id: 2, table)
            let metadata = b.finish(b.endObject())
            let padded = (metadata.count + 7) / 8 * 8
            withUnsafeBytes(of: arrowContinuation.littleEndian) { stream.append(contentsOf: $0) }
            withUnsafeBytes(of: Int32(padded).littleEndian) { stream.append(contentsOf: $0) }
            stream.append(contentsOf: metadata + [UInt8](repeating: 0, count: padded - metadata.count))
            withUnsafeBytes(of: arrowContinuation.littleEndian) { stream.append(contentsOf: $0) }
            withUnsafeBytes(of: Int32(0).littleEndian) { stream.append(contentsOf: $0) }
            XCTAssertThrowsError(try ArrowIPCReader(data: stream)) { error in
                XCTAssertTrue("\(error)".contains(expected), "\(header): \(error)")
            }
        }
    }

    /// Replaces the last occurrence of a little-endian int64 in `data`.
    private func patchLastInt64(_ data: inout Data, _ from: Int64, _ to: Int64) throws {
        let needle = withUnsafeBytes(of: from.littleEndian) { Data($0) }
        let range = try XCTUnwrap(data.range(of: needle, options: .backwards))
        data.replaceSubrange(range, with: withUnsafeBytes(of: to.littleEndian) { Data($0) })
    }

    /// Offsets past 2 GB are the engine's limit (its utf8, binary and list arrays have 32-bit offsets):
    /// every layout that can carry one says so with a clear error instead of reading something else.
    func testOffsetsPast2GBAreAClearError() throws {
        let marker: Int64 = 4660, huge: Int64 = 3_000_000_000
        // large_utf8: one 4660-byte string, its final offset patched to 3e9.
        let text = try MetalStringArray([String(repeating: "x", count: Int(marker))])
        var large = try ArrowIPCWriter.encode([try MetalRecordBatch(names: ["big"], columns: [.string(text)])],
                                              schema: ArrowIPCSchema(fields: [ArrowIPCField(name: "big", type: .largeUtf8)]),
                                              format: .stream)
        try patchLastInt64(&large, marker, huge)
        XCTAssertThrowsError(try ArrowIPCReader(data: large).batch(at: 0)) { error in
            XCTAssertEqual("\(error)", "Unsupported Arrow IPC feature: large binary column 'big' over 2 GB")
        }
        // large_list: one row of 4660 int8 zeros.
        let zeros = MetalArray<Int8>(length: Int(marker), nullCount: 0, validity: nil,
                                     values: try MetalArrowBuffer.allocate(byteCount: Int(marker)))
        let list = try MetalListArray(counts: [Int(marker)], values: .int8(zeros))
        let listSchema = ArrowIPCSchema(fields: [
            ArrowIPCField(name: "biglist", type: .largeList(ArrowIPCField(name: "item", type: .int(bits: 8, signed: true)))),
        ])
        var largeList = try ArrowIPCWriter.encode([try MetalRecordBatch(names: ["biglist"], columns: [.list(list)])],
                                                  schema: listSchema, format: .stream)
        try patchLastInt64(&largeList, marker, huge)
        XCTAssertThrowsError(try ArrowIPCReader(data: largeList).batch(at: 0)) { error in
            XCTAssertEqual("\(error)", "Unsupported Arrow IPC feature: 64-bit offsets over 2 GB (column 'biglist')")
        }

        // large_list_view (sizes patched) and a string_view whose views add up past 2 GB, from pyarrow.
        let listView = temporaryFile("arrows"), views = temporaryFile("arrows")
        defer { for u in [listView, views] { try? FileManager.default.removeItem(at: u) } }
        let script = """
        import sys, struct, pyarrow as pa
        n = 4660
        lv = pa.array([[0] * n], pa.large_list_view(pa.int8()))
        with pa.ipc.new_stream(sys.argv[1], pa.schema([("lv", lv.type)])) as w:
            w.write_batch(pa.record_batch([lv], names=["lv"]))
        # 2100 views of the same 1 MiB: 2100 MiB of logical bytes over one small data buffer.
        size, rows = 1 << 20, 2100
        data = pa.py_buffer(b"z" * size)
        view = struct.pack("<i4sii", size, b"zzzz", 0, 0)
        v = pa.Array.from_buffers(pa.string_view(), rows, [None, pa.py_buffer(view * rows), data])
        with pa.ipc.new_stream(sys.argv[2], pa.schema([("v", v.type)])) as w:
            w.write_batch(pa.record_batch([v], names=["v"]))
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [listView.path, views.path]), "ok")
        var lvData = try Data(contentsOf: listView)
        try patchLastInt64(&lvData, marker, huge)
        XCTAssertThrowsError(try ArrowIPCReader(data: lvData).batch(at: 0)) { error in
            XCTAssertEqual("\(error)", "Unsupported Arrow IPC feature: 64-bit offsets over 2 GB (column 'lv')")
        }
        XCTAssertThrowsError(try ArrowIPCReader(url: views).batch(at: 0)) { error in
            XCTAssertEqual("\(error)", "Unsupported Arrow IPC feature: view column 'v' over 2 GB: "
                           + "the engine's utf8 and binary arrays have 32-bit offsets")
        }
    }

    /// A view that points outside the data buffers the batch holds is malformed.
    func testAViewOutsideItsDataBuffersIsRejected() throws {
        let url = temporaryFile("arrows")
        defer { try? FileManager.default.removeItem(at: url) }
        let script = """
        import sys, struct, pyarrow as pa
        data = pa.py_buffer(b"q" * 32)
        views = struct.pack("<i4sii", 20, b"qqqq", 0, 16)       # bytes 16 ..< 36 of a 32-byte buffer
        v = pa.Array.from_buffers(pa.string_view(), 1, [None, pa.py_buffer(views), data])
        with pa.ipc.new_stream(sys.argv[1], pa.schema([("v", v.type)])) as w:
            w.write_batch(pa.record_batch([v], names=["v"]))
        print("ok")
        """
        XCTAssertEqual(try runPython(script, [url.path]), "ok")
        XCTAssertThrowsError(try ArrowIPCReader(url: url).batch(at: 0)) { error in
            XCTAssertTrue("\(error)".contains("view 0 of column 'v' points at bytes 16..<36 of data buffer 0"), "\(error)")
        }
    }
}
