import Foundation
import Metal

// Reading a CSV file on the GPU.
//
//   mmap the file, wrap it as one MTLBuffer (no copy)
//     -> structure pass (CSVScan.swift): every field and record boundary, quote-aware
//     -> csv_check_rows: every record has the header's width, or the error names the row
//     -> per projected column: csv_spans (content span per row; doubled quotes unescaped into a side
//        buffer), csv_classify (Arrow's type inference), one csv_conv_* kernel for the chosen type
//
// The host touches the bytes only for the BOM, `skipRows` (line counting, as pyarrow does it), the
// header names and error messages. Columns that are not projected are never converted.

/// A CSV file mapped for reading on the GPU. `read()` returns a Metal-resident record batch.
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
    private let region: MappedRegion?

    public init(path: String, options: CSVReadOptions = CSVReadOptions(), context: MetalContext = .shared) throws {
        self.path = path
        self.options = options
        self.context = context
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw CSVError.io("cannot open \(path): \(String(cString: strerror(errno)))") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw CSVError.io("cannot stat \(path)") }
        fileSize = Int(st.st_size)
        guard fileSize < 0x8000_0000 else {
            throw CSVError.io("\(path) is \(fileSize) bytes; files of 2 GiB and more are not supported yet")
        }
        if fileSize == 0 {
            region = nil
        } else {
            do {
                region = try MappedRegion(fd: fd, fileOffset: 0, length: roundUp(fileSize, to: metalPageSize()))
            } catch {
                throw CSVError.io("cannot map \(path): \(error)")
            }
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
        guard let region else { throw CSVError.parse("Empty CSV file") }
        let bytes = region.raw
        var keep: [AnyObject] = []
        return try context.batch {
            // A UTF-8 byte order mark is not part of the data.
            var pos = 0
            if fileSize >= 3, bytes[0] == 0xEF, bytes[1] == 0xBB, bytes[2] == 0xBF { pos = 3 }
            if pos == fileSize { throw CSVError.parse("Empty CSV file") }
            let skipped = try skipLines(bytes, from: &pos)
            let (file, _) = try MetalArrowBuffer.wrapOrCopy(UnsafeRawPointer(region.base), byteCount: fileSize,
                                                            keepAlive: region, context: context)
            keep.append(file)
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
            if q == o.delimiter { throw CSVError.invalidOptions("ParseOptions: delimiter cannot be the same as quote_char") }
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

    // MARK: - fields on the host (header names and error messages only)

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
        let ev = s.events.typed(UInt32.self)
        // Wait for the boundaries before the host reads any of them.
        try context.syncPoint()

        // pyarrow infers the column count from the first complete line; a file whose first record has
        // no newline after it (or that has no record at all) has none.
        if namesFromFile && (s.count == 0 || s.unterminated && firstRecordEnd(ev, s.count) == s.count - 1) {
            throw CSVError.parse("CSV parse error: Empty CSV file or block: cannot infer number of columns")
        }
        let nCols: Int
        if let names = o.columnNames { nCols = names.count } else { nCols = firstRecordEnd(ev, s.count) + 1 }
        guard nCols > 0 else { throw CSVError.parse("CSV parse error: Empty CSV file or block: cannot infer number of columns") }

        if let k = try firstRaggedBoundary(s, nCols: nCols, keep: &keep) {
            let k0 = (k / nCols) * nCols
            var kEnd = k0
            while kEnd < s.count - 1, (ev[kEnd] >> 31) == 0 { kEnd += 1 }
            let start = rawSpan(ev, k0, bytes, dataStart: dataStart).0
            let end = Swift.max(start, Int(ev[kEnd] & 0x7FFF_FFFF))
            let text = String(decoding: bytes[start..<end], as: UTF8.self)
            throw CSVError.parse("CSV parse error: Row #\(skipped + k0 / nCols + 1): Expected \(nCols) columns, got \(kEnd - k0 + 1): \(text)")
        }
        let nRecords = s.count / nCols

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
        let firstRecord = Swift.min(nRecords, headerRecords + o.skipRowsAfterNames)
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
                                      dataEnd: fileSize, rowBase: skipped + firstRecord + 1)
        let columns = try conv.convert(plan: plan, keep: &keep)
        numRows = nRows
        return try MetalRecordBatch(names: plan.map { $0.name }, columns: columns)
    }

    /// Index of the first record end, or `count - 1` when there is none (cannot happen: the last
    /// boundary always ends a record).
    private func firstRecordEnd(_ ev: UnsafePointer<UInt32>, _ count: Int) -> Int {
        var k = 0
        while k < count - 1, (ev[k] >> 31) == 0 { k += 1 }
        return k
    }
}
