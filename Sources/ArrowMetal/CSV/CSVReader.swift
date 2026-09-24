import Foundation
import Metal

// Reading a CSV file on the GPU.
//
//   bring the file into one Metal buffer: parallel pread into a page-aligned shared buffer (the
//   default), or mmap + wrap with no copy (`CSVReadOptions.fileAccess = .map`)
//     -> structure pass (CSVScan.swift): every field and record boundary, quote-aware
//     -> csv_check_rows: every record has the header's width, or the error names the row
//     -> per projected column: csv_spans (content span per row; doubled quotes unescaped into a side
//        buffer), csv_classify (Arrow's type inference), one csv_conv_* kernel for the chosen type
//
// The host touches the bytes only for the BOM, `skipRows` (line counting, as pyarrow does it), the
// header names, error messages and the floats the GPU parser defers. Columns that are not projected
// are never converted.

/// A CSV file read on the GPU. `read()` returns a Metal-resident record batch.
///
/// ```swift
/// var o = CSVReadOptions()
/// o.includeColumns = ["price", "qty"]
/// let batch = try CSVReader(path: "trades.csv", options: o).read()
/// ```
public final class CSVReader: @unchecked Sendable {
    public let path: String
    public let options: CSVReadOptions
    public let context: MetalContext
    /// Rows in the last `read()` (the batch has no columns to carry it when the projection is empty).
    public private(set) var numRows = 0

    private let fileSize: Int

    /// `ARROWMETAL_CSV_TRACE=1` prints the wall time of each phase of a read to stderr; `=2` also
    /// runs every kernel in its own command buffer and prints its GPU time.
    static let trace = ProcessInfo.processInfo.environment["ARROWMETAL_CSV_TRACE"] != nil
    static let traceKernels = ProcessInfo.processInfo.environment["ARROWMETAL_CSV_TRACE"] == "2"

    /// Encodes one kernel (or a short chain); under `ARROWMETAL_CSV_TRACE=2` reports its GPU time.
    func gpu(_ name: String, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        let cb = try context.run(body)
        if Self.traceKernels, let cb {
            FileHandle.standardError.write(String(format: "gpu %@ %.3f ms\n", name as NSString,
                                                  (cb.gpuEndTime - cb.gpuStartTime) * 1e3).data(using: .utf8)!)
        }
    }
    private var traceClock: UInt64 = 0
    func mark(_ label: String) {
        guard Self.trace else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        if traceClock != 0 {
            FileHandle.standardError.write(String(format: "csv %-22@ %8.2f ms\n", label as NSString,
                                                  Double(now - traceClock) / 1e6).data(using: .utf8)!)
        }
        traceClock = now
    }

    public init(path: String, options: CSVReadOptions = CSVReadOptions(), context: MetalContext = .shared) throws {
        self.path = path
        // A quote character equal to the delimiter never opens a quote: the delimiter wins, which is
        // how pyarrow reads it (the same table as quote_char=False).
        var options = options
        if options.quoteChar == options.delimiter { options.quoteChar = nil }
        self.options = options
        self.context = context
        var st = stat()
        guard stat(path, &st) == 0 else { throw CSVError.io("cannot open \(path): \(String(cString: strerror(errno)))") }
        fileSize = Int(st.st_size)
        guard fileSize < 0x8000_0000 else {
            throw CSVError.io("\(path) is \(fileSize) bytes; files of 2 GiB and more are not supported yet")
        }
    }

    /// The whole file as one Metal buffer.
    ///
    /// `.read` (the default) preads it, in up to eight ranges of at least 4 MiB at once, into a
    /// page-aligned shared buffer from the context's pool. `.map` maps it and wraps the mapping with no
    /// copy, which measured slower overall than the copy on a warm page cache (Benchmarks/csv_bench.py
    /// records both).
    private func loadFile() throws -> MetalArrowBuffer {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw CSVError.io("cannot open \(path): \(String(cString: strerror(errno)))") }
        defer { close(fd) }
        switch options.fileAccess {
        case .map:
            let region: MappedRegion
            do { region = try MappedRegion(fd: fd, fileOffset: 0, length: roundUp(fileSize, to: metalPageSize())) }
            catch { throw CSVError.io("cannot map \(path): \(error)") }
            return try MetalArrowBuffer.wrapOrCopy(UnsafeRawPointer(region.base), byteCount: fileSize,
                                                   keepAlive: region, context: context).0
        case .read:
            let buf = try MetalArrowBuffer.allocate(byteCount: fileSize, zeroed: false, context: context)
            let base = buf.mutableContents
            let parts = Swift.max(1, Swift.min(8, fileSize / (4 << 20)))
            let step = roundUp((fileSize + parts - 1) / parts, to: metalPageSize())
            var failed = Int32(0)
            let failLock = NSLock()
            DispatchQueue.concurrentPerform(iterations: parts) { i in
                var off = i * step
                let end = Swift.min(fileSize, off + step)
                while off < end {
                    let n = pread(fd, base + off, end - off, off_t(off))
                    if n <= 0 {
                        let e = n == 0 ? EIO : errno
                        failLock.lock(); if failed == 0 { failed = e }; failLock.unlock()
                        return
                    }
                    off += n
                }
            }
            guard failed == 0 else { throw CSVError.io("cannot read \(path): \(String(cString: strerror(failed)))") }
            return buf
        }
    }

    /// Convenience: open and read in one call.
    public static func read(path: String, options: CSVReadOptions = CSVReadOptions(),
                            context: MetalContext = .shared) throws -> MetalRecordBatch {
        try CSVReader(path: path, options: options, context: context).read()
    }

    /// Parses, infers and converts every projected column.
    public func read() throws -> MetalRecordBatch {
        try validateOptions()
        guard fileSize > 0 else { throw CSVError.parse("Empty CSV file") }
        var keep: [AnyObject] = []
        traceClock = 0
        mark("start")
        let file = try loadFile()
        keep.append(file)
        mark("load file")
        let bytes = UnsafeRawBufferPointer(start: file.contents, count: fileSize)
        func phases<R>(_ body: () throws -> R) throws -> R {
            Self.traceKernels ? try body() : try context.batch(body)
        }
        return try phases {
            // A UTF-8 byte order mark is not part of the data.
            var pos = 0
            if fileSize >= 3, bytes[0] == 0xEF, bytes[1] == 0xBB, bytes[2] == 0xBF { pos = 3 }
            if pos == fileSize { throw CSVError.parse("Empty CSV file") }
            let skipped = try skipLines(bytes, from: &pos)
            let s = try scanStructure(file: file, dataStart: pos, dataEnd: fileSize, keep: &keep)
            keep.append(s.events)
            return try assemble(file: file, bytes: bytes, structure: s, dataStart: pos, skipped: skipped, keep: &keep)
        }
    }

    private func validateOptions() throws {
        let o = options
        if o.autogenerateColumnNames && o.columnNames != nil {
            throw CSVError.invalidOptions("ReadOptions: autogenerate_column_names cannot be true when column_names are provided")
        }
        let reserved: Set<UInt8> = [0x0A, 0x0D]
        if reserved.contains(o.delimiter) { throw CSVError.invalidOptions("ParseOptions: delimiter cannot be \\r or \\n") }
        if let q = o.quoteChar {
            if reserved.contains(q) { throw CSVError.invalidOptions("ParseOptions: quote_char cannot be \\r or \\n") }
        }
        if o.skipRows < 0 || o.skipRowsAfterNames < 0 { throw CSVError.invalidOptions("ReadOptions: skip counts must be >= 0") }
    }

    /// pyarrow's `skip_rows`: whole lines, found by their terminators (`\n`, `\r\n` or `\r`), quotes
    /// ignored, empty lines counted. A line with no terminator cannot be skipped.
    private func skipLines(_ bytes: UnsafeRawBufferPointer, from pos: inout Int) throws -> Int {
        let n = options.skipRows
        for _ in 0..<n {
            var i = pos
            while i < fileSize, bytes[i] != 0x0A, bytes[i] != 0x0D { i += 1 }
            guard i < fileSize else {
                throw CSVError.parse("Could not skip initial \(n) rows from CSV file, either file is too short or header is larger than block size")
            }
            pos = (bytes[i] == 0x0D && i + 1 < fileSize && bytes[i + 1] == 0x0A) ? i + 2 : i + 1
        }
        return n
    }

    // MARK: - fields on the host (header names, error messages, deferred floats)

    /// Start and end of field `k`'s raw bytes, the host twin of `csv_raw`.
    func rawSpan(_ ev: UnsafePointer<UInt32>, _ k: Int, _ bytes: UnsafeRawBufferPointer, dataStart: Int) -> (Int, Int) {
        func skipNL(_ p: Int) -> Int {
            var p = p
            while p < fileSize, bytes[p] == 0x0A || bytes[p] == 0x0D { p += 1 }
            return p
        }
        let e = Int(ev[k] & 0x7FFF_FFFF)
        let s: Int
        if k == 0 { s = skipNL(dataStart) } else {
            let pv = ev[k - 1], pp = Int(pv & 0x7FFF_FFFF)
            s = (pv >> 31) != 0 ? skipNL(pp + 1) : pp + 1
        }
        return (s, Swift.max(s, e))
    }

    /// The value of field `k`, unescaped, as bytes.
    func fieldValue(_ ev: UnsafePointer<UInt32>, _ k: Int, _ bytes: UnsafeRawBufferPointer, dataStart: Int) -> [UInt8] {
        let (s, e) = rawSpan(ev, k, bytes, dataStart: dataStart)
        guard let q = options.quoteChar, s < e, bytes[s] == q else { return Array(bytes[s..<e]) }
        var out: [UInt8] = []
        var inQ = true
        var i = s + 1
        while i < e {
            let c = bytes[i]
            if inQ && c == q {
                if options.doubleQuote && i + 1 < e && bytes[i + 1] == q { out.append(q); i += 2; continue }
                inQ = false
            } else { out.append(c) }
            i += 1
        }
        return out
    }

    // MARK: - assembly

    private func assemble(file: MetalArrowBuffer, bytes: UnsafeRawBufferPointer, structure s: CSVStructure,
                          dataStart: Int, skipped: Int, keep: inout [AnyObject]) throws -> MetalRecordBatch {
        let o = options
        let namesFromFile = o.columnNames == nil
        var ev = s.events.typed(UInt32.self)
        // Wait for the boundaries before the host reads any of them.
        try context.syncPoint()
        mark("emit boundaries")

        // pyarrow infers the column count from the first complete line; a file whose first record has
        // no newline after it (or that has no record at all) has none.
        if namesFromFile && (s.count == 0 || s.unterminated && firstRecordEnd(ev, s.count) == s.count - 1) {
            throw CSVError.parse("CSV parse error: Empty CSV file or block: cannot infer number of columns")
        }
        let nCols: Int
        if let names = o.columnNames { nCols = names.count } else { nCols = firstRecordEnd(ev, s.count) + 1 }
        guard nCols > 0 else { throw CSVError.parse("CSV parse error: Empty CSV file or block: cannot infer number of columns") }

        var names: [String]
        let headerRecords: Int
        if let given = o.columnNames {
            names = given; headerRecords = 0
        } else if o.autogenerateColumnNames {
            names = (0..<nCols).map { "f\($0)" }; headerRecords = 0
        } else {
            names = (0..<nCols).map { String(decoding: fieldValue(ev, $0, bytes, dataStart: dataStart), as: UTF8.self) }
            headerRecords = 1
        }

        // pyarrow skips `skip_rows_after_names` rows without checking their width, and counts an empty
        // line among them as a row. Those rows are dropped on the host by moving the start of the
        // boundary list (and of the data) past them; the width check then covers the rows that are
        // read. Without skipped rows the header stays in the list as record 0.
        var s = s
        var dataStart = dataStart
        var rowsBefore = 0                  // rows (records and skipped empty lines) before the list
        if o.skipRowsAfterNames > 0 {
            let (k, cursor, rows) = skipRecords(ev, s.count, bytes, from: headerRecords == 1 ? nCols : 0,
                                                at: headerRecords == 1 ? Int(ev[nCols - 1] & 0x7FFF_FFFF) : dataStart,
                                                headerEnd: headerRecords == 1, count: o.skipRowsAfterNames)
            let view = MetalArrowBuffer(mtl: s.events.mtl, byteCount: (s.count - k) * 4,
                                        offset: s.events.offset + k * 4, keepAlive: s.events)
            s = CSVStructure(events: view, count: s.count - k, unterminated: s.unterminated)
            ev = view.typed(UInt32.self)
            dataStart = cursor
            rowsBefore = headerRecords + rows
        }

        let (firstBad, complexCounts) = try checkFields(s, file: file, nCols: nCols, dataStart: dataStart,
                                                        dataEnd: fileSize, keep: &keep)
        if let k = firstBad {
            let k0 = (k / nCols) * nCols
            var kEnd = k0
            while kEnd < s.count - 1, (ev[kEnd] >> 31) == 0 { kEnd += 1 }
            let start = rawSpan(ev, k0, bytes, dataStart: dataStart).0
            let end = Swift.max(start, Int(ev[kEnd] & 0x7FFF_FFFF))
            // pyarrow quotes at most 100 bytes of the row: longer rows are cut to 96 bytes and " ...".
            let text = end - start > 100
                ? String(decoding: bytes[start..<(start + 96)], as: UTF8.self) + " ..."
                : String(decoding: bytes[start..<end], as: UTF8.self)
            throw CSVError.parse("CSV parse error: Row #\(skipped + rowsBefore + k0 / nCols + 1): Expected \(nCols) columns, got \(kEnd - k0 + 1): \(text)")
        }
        let nRecords = s.count / nCols
        mark("check rows")

        let firstRecord = o.skipRowsAfterNames > 0 ? 0 : Swift.min(nRecords, headerRecords)
        let nRows = nRecords - firstRecord
        try Dispatch.checkLength(nRows)

        // Projection: pyarrow's include_columns, in the order given; empty means every column.
        var plan: [(name: String, source: Int?)] = []
        if let inc = o.includeColumns, !inc.isEmpty {
            for n in inc {
                if let i = names.firstIndex(of: n) { plan.append((n, i)) }
                else if o.includeMissingColumns { plan.append((n, nil)) }
                else { throw CSVError.missingColumn("Column '\(n)' in include_columns does not exist in CSV file") }
            }
        } else {
            plan = names.enumerated().map { ($0.element, $0.offset) }
        }

        let conv = CSVColumnConverter(reader: self, file: file, events: s.events, nCols: nCols,
                                      firstRecord: firstRecord, nRows: nRows, dataStart: dataStart,
                                      dataEnd: fileSize, rowBase: skipped + rowsBefore + firstRecord + 1)
        let columns = try conv.convert(plan: plan, complexCounts: complexCounts, keep: &keep)
        numRows = nRows
        mark("columns")
        return try MetalRecordBatch(names: plan.map { $0.name }, columns: columns)
    }

    /// pyarrow's `skip_rows_after_names`: skips `count` rows starting at field `k` (whose record starts
    /// at or after byte `cursor`), where a row is a record or an empty line. When `headerEnd` is set,
    /// `cursor` is the header's line terminator, which is consumed first. Returns the first field after
    /// the skipped rows, the byte position after them, and how many rows were skipped (fewer than
    /// `count` when the file ends first).
    private func skipRecords(_ ev: UnsafePointer<UInt32>, _ nEvents: Int, _ bytes: UnsafeRawBufferPointer,
                             from k: Int, at cursor: Int, headerEnd: Bool, count: Int) -> (Int, Int, Int) {
        var k = k, pos = cursor, rows = 0
        /// Consumes one line terminator at `pos` (`\r\n`, `\n` or `\r`), if there is one.
        func terminator() {
            guard pos < fileSize else { return }
            if bytes[pos] == 0x0D, pos + 1 < fileSize, bytes[pos + 1] == 0x0A { pos += 2 }
            else if bytes[pos] == 0x0A || bytes[pos] == 0x0D { pos += 1 }
        }
        if headerEnd { terminator() }
        while rows < count, pos < fileSize {
            if bytes[pos] == 0x0A || bytes[pos] == 0x0D {
                terminator()                                     // an empty line counts as a row
            } else {
                guard k < nEvents else { break }
                while k < nEvents - 1, (ev[k] >> 31) == 0 { k += 1 }
                pos = Int(ev[k] & 0x7FFF_FFFF)
                k += 1
                terminator()
            }
            rows += 1
        }
        return (k, pos, rows)
    }

    /// Index of the first record end, or `count - 1` when there is none (cannot happen: the last
    /// boundary always ends a record).
    private func firstRecordEnd(_ ev: UnsafePointer<UInt32>, _ count: Int) -> Int {
        var k = 0
        while k < count - 1, (ev[k] >> 31) == 0 { k += 1 }
        return k
    }
}
