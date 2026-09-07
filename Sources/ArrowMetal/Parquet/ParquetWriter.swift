import Foundation

// A deliberately small Parquet writer: enough to round-trip everything the reader understands, and no
// more. One data page per column chunk, PLAIN for fixed-width columns, PLAIN or RLE_DICTIONARY for byte
// arrays, RLE definition levels, uncompressed or Snappy. It runs on the host: writing is not the part
// of Parquet that needed a GPU, and keeping it on the CPU keeps it to a few hundred readable lines.

/// Options for `ParquetWriter`.
public struct ParquetWriteOptions: Sendable {
    public var compression: ParquetCodec = .snappy
    public var useDictionary = true
    public var rowGroupSize = 1 << 20
    public var createdBy = "ArrowMetal"

    public init(compression: ParquetCodec = .snappy, useDictionary: Bool = true,
                rowGroupSize: Int = 1 << 20, createdBy: String = "ArrowMetal") {
        self.compression = compression
        self.useDictionary = useDictionary
        self.rowGroupSize = rowGroupSize
        self.createdBy = createdBy
    }
}

public enum ParquetWriter {

    /// Writes `batch` to `path`.
    public static func write(_ batch: MetalRecordBatch, to path: String,
                             options: ParquetWriteOptions = ParquetWriteOptions()) throws {
        guard options.compression == .uncompressed || options.compression == .snappy else {
            throw ParquetError.unsupported("the writer only produces UNCOMPRESSED or SNAPPY files")
        }
        var out = [UInt8]()
        out.append(contentsOf: Array("PAR1".utf8))

        // Columns are materialised once, host side, in the shape the encoders want.
        let columns = try batch.columns.map { try Column($0) }
        let rows = batch.length
        var rowGroups: [WrittenRowGroup] = []
        var start = 0
        while start < rows {
            let n = Swift.min(options.rowGroupSize, rows - start)
            var chunks: [WrittenChunk] = []
            for (i, c) in columns.enumerated() {
                chunks.append(try writeChunk(&out, c, name: batch.names[i], from: start, count: n,
                                             options: options))
            }
            rowGroups.append(WrittenRowGroup(chunks: chunks, numRows: Int64(n)))
            start += n
        }

        let footer = try metadata(names: batch.names, columns: columns, rowGroups: rowGroups,
                                 numRows: Int64(rows), createdBy: options.createdBy)
        out.append(contentsOf: footer)
        var len = UInt32(footer.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(contentsOf: Array("PAR1".utf8))
        try Data(out).write(to: URL(fileURLWithPath: path))
    }

    // MARK: - Column materialisation

    /// A column reduced to what the encoders need: a physical type and either fixed-width bytes or a
    /// list of byte strings, plus a validity view.
    struct Column {
        var physical: ParquetPhysicalType = .int32
        var logical: ParquetLogicalType = .none
        var converted: Int32? = nil
        var width = 0
        var typeLength = 0
        /// Fixed-width values, `width` bytes each (row positioned).
        var fixed: [UInt8] = []
        /// Byte-array values (row positioned; a null row holds an empty entry).
        var bytes: [[UInt8]] = []
        var isNull: [Bool] = []
        var length = 0

        init(_ a: AnyMetalArray) throws {
            let a = try a.decode()                       // dictionary columns are materialised first
            self.length = a.length
            self.isNull = [Bool](repeating: false, count: a.length)
            func fillNulls(_ validity: MetalArrowBuffer?) {
                guard let v = validity else { return }
                let p = v.typed(UInt8.self)
                for i in 0..<length { isNull[i] = !Bitmap.isSet(p, i) }
            }
            func raw<T>(_ arr: MetalArray<T>, _ w: Int, _ type: ParquetPhysicalType) {
                physical = type
                width = w
                fixed = [UInt8](repeating: 0, count: length * w)
                fixed.withUnsafeMutableBytes { memcpy($0.baseAddress!, arr.values.contents, length * w) }
                fillNulls(arr.validity)
            }
            /// Widens a narrow integer column into the INT32 slot Parquet uses for it.
            func widen<T: FixedWidthInteger & ArrowPrimitive>(_ arr: MetalArray<T>, signed: Bool, bits: Int) {
                physical = .int32
                width = 4
                fixed = [UInt8](repeating: 0, count: length * 4)
                let src = arr.values.typed(T.self)
                fixed.withUnsafeMutableBytes { buf in
                    let d = buf.baseAddress!.assumingMemoryBound(to: Int32.self)
                    for i in 0..<length { d[i] = Int32(truncatingIfNeeded: src[i]) }
                }
                logical = .integer(bitWidth: bits, signed: signed)
                fillNulls(arr.validity)
            }
            switch a {
            case .int32(let x): raw(x, 4, .int32)
            case .int64(let x): raw(x, 8, .int64)
            case .float32(let x): raw(x, 4, .float)
            case .float64(let x): raw(x, 8, .double)
            case .uint32(let x): raw(x, 4, .int32); logical = .integer(bitWidth: 32, signed: false)
            case .uint64(let x): raw(x, 8, .int64); logical = .integer(bitWidth: 64, signed: false)
            case .int8(let x): widen(x, signed: true, bits: 8)
            case .int16(let x): widen(x, signed: true, bits: 16)
            case .uint8(let x): widen(x, signed: false, bits: 8)
            case .uint16(let x): widen(x, signed: false, bits: 16)
            case .boolean(let x):
                physical = .boolean
                width = 1
                let bits = x.values.typed(UInt8.self)
                fixed = (0..<length).map { Bitmap.isSet(bits, $0) ? 1 : 0 }
                fillNulls(x.validity)
            case .string(let s), .binary(let s):
                physical = .byteArray
                let off = s.offsets.typed(Int32.self)
                let data = s.data.typed(UInt8.self)
                bytes = (0..<length).map { i in
                    let lo = Int(off[i]), hi = Int(off[i + 1])
                    return Array(UnsafeBufferPointer(start: data + lo, count: Swift.max(hi - lo, 0)))
                }
                if case .string = a { logical = .string; converted = 0 }
                fillNulls(s.validity)
            case .temporal(let t):
                switch t.storage {
                case .int32(let x): raw(x, 4, .int32)
                case .int64(let x): raw(x, 8, .int64)
                }
                switch t.type {
                case .date32: logical = .date
                case .date64: logical = .timestamp(isUTC: false, unit: .millis)
                case .time32(let u): logical = .time(isUTC: false, unit: u == .milli ? .millis : .millis)
                case .time64(let u): logical = .time(isUTC: false, unit: u == .nano ? .nanos : .micros)
                case .timestamp(let u, let tz):
                    logical = .timestamp(isUTC: tz != nil, unit: u == .milli ? .millis : (u == .nano ? .nanos : .micros))
                case .duration: logical = .none
                }
            case .fixedBinary(let f):
                physical = .fixedLenByteArray
                typeLength = f.byteWidth
                width = f.byteWidth
                fixed = [UInt8](repeating: 0, count: length * f.byteWidth)
                fixed.withUnsafeMutableBytes { memcpy($0.baseAddress!, f.values.contents, length * f.byteWidth) }
                fillNulls(f.validity)
            default:
                throw ParquetError.unsupported("writing \(a.arrowFormat) columns")
            }
        }
    }

    struct WrittenChunk {
        var column: Column
        var name: String
        var dictionaryOffset: Int64?
        var dataOffset: Int64
        var numValues: Int64
        var uncompressed: Int64
        var compressed: Int64
        var encodings: [ParquetEncoding]
        var codec: ParquetCodec
    }
    struct WrittenRowGroup {
        var chunks: [WrittenChunk]
        var numRows: Int64
    }

    // MARK: - Chunk writing

    private static func writeChunk(_ out: inout [UInt8], _ c: Column, name: String, from: Int, count: Int,
                                   options: ParquetWriteOptions) throws -> WrittenChunk {
        var dictOffset: Int64? = nil
        var encodings: [ParquetEncoding] = [.rle]
        let start = out.count

        // The definition levels: 1 where the value is present, 0 where it is null. Always written, so
        // every column round-trips as optional.
        var levels = [UInt8](repeating: 1, count: count)
        for i in 0..<count where c.isNull[from + i] { levels[i] = 0 }
        let levelBytes = rleEncode(levels, bitWidth: 1)

        var body = [UInt8]()
        var pageEncoding = ParquetEncoding.plain
        if c.physical == .byteArray && options.useDictionary {
            // Build the dictionary out of the distinct present values, in first-appearance order.
            var index: [Data: Int32] = [:]
            var order: [[UInt8]] = []
            var codes: [Int32] = []
            for i in 0..<count where !c.isNull[from + i] {
                let v = c.bytes[from + i]
                let key = Data(v)
                if let k = index[key] { codes.append(k) }
                else {
                    let k = Int32(order.count)
                    index[key] = k
                    order.append(v)
                    codes.append(k)
                }
            }
            var dictBody = [UInt8]()
            for v in order {
                var l = UInt32(v.count).littleEndian
                withUnsafeBytes(of: &l) { dictBody.append(contentsOf: $0) }
                dictBody.append(contentsOf: v)
            }
            dictOffset = Int64(out.count)
            try writePage(&out, kind: .dictionaryPage, body: dictBody, numValues: order.count,
                          encoding: .plain, codec: options.compression)
            let bw = bitWidth(order.count)
            body = levelBytes4(levelBytes)
            body.append(UInt8(bw))
            body.append(contentsOf: bitPack(codes, bitWidth: bw))
            pageEncoding = .rleDictionary
            encodings.append(.plain)
            encodings.append(.rleDictionary)
        } else {
            var values = [UInt8]()
            switch c.physical {
            case .boolean:
                var bits = [UInt8](repeating: 0, count: (count + 7) / 8)
                var k = 0
                for i in 0..<count where !c.isNull[from + i] {
                    if c.fixed[from + i] != 0 { bits[k >> 3] |= UInt8(1 << (k & 7)) }
                    k += 1
                }
                values = Array(bits.prefix((k + 7) / 8))
            case .byteArray:
                for i in 0..<count where !c.isNull[from + i] {
                    let v = c.bytes[from + i]
                    var l = UInt32(v.count).littleEndian
                    withUnsafeBytes(of: &l) { values.append(contentsOf: $0) }
                    values.append(contentsOf: v)
                }
            default:
                let w = c.width
                values.reserveCapacity(count * w)
                for i in 0..<count where !c.isNull[from + i] {
                    values.append(contentsOf: c.fixed[(from + i) * w..<(from + i + 1) * w])
                }
            }
            body = levelBytes4(levelBytes)
            body.append(contentsOf: values)
            encodings.append(.plain)
        }
        let dataOffset = Int64(out.count)
        try writePage(&out, kind: .dataPage, body: body, numValues: count, encoding: pageEncoding,
                      codec: options.compression)
        return WrittenChunk(column: c, name: name, dictionaryOffset: dictOffset, dataOffset: dataOffset,
                            numValues: Int64(count), uncompressed: Int64(out.count - start),
                            compressed: Int64(out.count - start), encodings: encodings,
                            codec: options.compression)
    }

    /// A v1 page section is preceded by its 4-byte little-endian length.
    private static func levelBytes4(_ levels: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        var l = UInt32(levels.count).littleEndian
        withUnsafeBytes(of: &l) { out.append(contentsOf: $0) }
        out.append(contentsOf: levels)
        return out
    }

    private static func writePage(_ out: inout [UInt8], kind: ParquetPageType, body: [UInt8],
                                  numValues: Int, encoding: ParquetEncoding, codec: ParquetCodec) throws {
        let payload = codec == .snappy ? Snappy.compress(body) : body
        var w = ThriftWriter()
        w.beginRoot()
        w.int(1, Int(kind.rawValue))
        w.int(2, body.count)
        w.int(3, payload.count)
        if kind == .dictionaryPage {
            w.beginStruct(7)
            w.int(1, numValues)
            w.int(2, Int(ParquetEncoding.plain.rawValue))
            w.bool(3, false)
            w.endStruct()
        } else {
            w.beginStruct(5)
            w.int(1, numValues)
            w.int(2, Int(encoding.rawValue))
            w.int(3, Int(ParquetEncoding.rle.rawValue))
            w.int(4, Int(ParquetEncoding.rle.rawValue))
            w.endStruct()
        }
        w.endRoot()
        out.append(contentsOf: w.bytes)
        out.append(contentsOf: payload)
    }

    // MARK: - RLE / bit packing

    static func bitWidth(_ maxValue: Int) -> Int {
        var v = Swift.max(maxValue - 1, 0), w = 0
        while v > 0 { w += 1; v >>= 1 }
        return w
    }

    /// RLE runs only: every run is a varint header (count << 1) plus the value in ceil(bw/8) bytes.
    /// Valid hybrid output, and for definition levels (long runs of 1s) it is also the compact one.
    static func rleEncode(_ values: [UInt8], bitWidth bw: Int) -> [UInt8] {
        var out = [UInt8]()
        var i = 0
        let valueBytes = (bw + 7) / 8
        while i < values.count {
            var j = i
            while j < values.count && values[j] == values[i] { j += 1 }
            var run = j - i
            while run > 0 {
                let take = Swift.min(run, 1 << 30)
                appendVarint(&out, UInt64(take) << 1)
                var v = UInt32(values[i]).littleEndian
                withUnsafeBytes(of: &v) { out.append(contentsOf: $0.prefix(valueBytes)) }
                run -= take
            }
            i = j
        }
        return out
    }

    /// Bit-packed runs of 8 values, LSB first, as the hybrid encoding defines them.
    static func bitPack(_ codes: [Int32], bitWidth bw: Int) -> [UInt8] {
        var out = [UInt8]()
        if bw == 0 {
            // Every value is 0: one RLE run says so without any packed data.
            appendVarint(&out, UInt64(codes.count) << 1)
            out.append(0)
            return out
        }
        var i = 0
        while i < codes.count {
            let groups = Swift.min((codes.count - i + 7) / 8, 63)
            appendVarint(&out, UInt64(groups) << 1 | 1)
            var bitPos = 0
            var packed = [UInt8](repeating: 0, count: (groups * 8 * bw + 7) / 8)
            for k in 0..<(groups * 8) {
                let v = i + k < codes.count ? UInt64(UInt32(bitPattern: codes[i + k])) : 0
                for b in 0..<bw where (v >> UInt64(b)) & 1 == 1 {
                    packed[(bitPos + b) >> 3] |= UInt8(1 << ((bitPos + b) & 7))
                }
                bitPos += bw
            }
            out.append(contentsOf: packed)
            i += groups * 8
        }
        return out
    }

    static func appendVarint(_ out: inout [UInt8], _ v: UInt64) {
        var x = v
        while true {
            if x < 0x80 { out.append(UInt8(x)); return }
            out.append(UInt8((x & 0x7F) | 0x80))
            x >>= 7
        }
    }

    // MARK: - Footer

    private static func metadata(names: [String], columns: [Column], rowGroups: [WrittenRowGroup],
                                 numRows: Int64, createdBy: String) throws -> [UInt8] {
        var w = ThriftWriter()
        w.beginRoot()
        w.int(1, 2)                                       // version
        // schema: the root message followed by one element per column
        w.beginList(2, .structure, count: columns.count + 1)
        w.beginListElement()
        w.string(4, "schema")
        w.int(5, columns.count)
        w.endListElement()
        for (i, c) in columns.enumerated() {
            w.beginListElement()
            w.int(1, Int(c.physical.rawValue))
            if c.physical == .fixedLenByteArray { w.int(2, c.typeLength) }
            w.int(3, Int(ParquetRepetition.optional.rawValue))
            w.string(4, names[i])
            if let cv = c.converted { w.int(6, Int(cv)) }
            writeLogicalType(&w, c.logical, id: 10)
            w.endListElement()
        }
        w.int64(3, numRows)
        w.beginList(4, .structure, count: rowGroups.count)
        for g in rowGroups {
            w.beginListElement()
            w.beginList(1, .structure, count: g.chunks.count)
            for (i, ch) in g.chunks.enumerated() {
                w.beginListElement()
                w.int64(2, ch.dictionaryOffset ?? ch.dataOffset)
                w.beginStruct(3)
                w.int(1, Int(ch.column.physical.rawValue))
                w.int32List(2, ch.encodings.map { $0.rawValue })
                w.stringList(3, [names[i]])
                w.int(4, Int(ch.codec.rawValue))
                w.int64(5, ch.numValues)
                w.int64(6, ch.uncompressed)
                w.int64(7, ch.compressed)
                w.int64(9, ch.dataOffset)
                if let d = ch.dictionaryOffset { w.int64(11, d) }
                w.endStruct()
                w.endListElement()
            }
            w.int64(2, g.chunks.reduce(0) { $0 + $1.uncompressed })
            w.int64(3, g.numRows)
            w.endListElement()
        }
        w.string(6, createdBy)
        w.endRoot()
        return w.bytes
    }

    private static func writeLogicalType(_ w: inout ThriftWriter, _ t: ParquetLogicalType, id: Int16) {
        func unit(_ u: ParquetTimeUnit) -> Int16 {
            switch u {
            case .millis: return 1
            case .micros: return 2
            case .nanos: return 3
            }
        }
        switch t {
        case .none: return
        case .string:
            w.beginStruct(id); w.beginStruct(1); w.endStruct(); w.endStruct()
        case .date:
            w.beginStruct(id); w.beginStruct(6); w.endStruct(); w.endStruct()
        case .time(let utc, let u):
            w.beginStruct(id); w.beginStruct(7)
            w.bool(1, utc)
            w.beginStruct(2); w.beginStruct(unit(u)); w.endStruct(); w.endStruct()
            w.endStruct(); w.endStruct()
        case .timestamp(let utc, let u):
            w.beginStruct(id); w.beginStruct(8)
            w.bool(1, utc)
            w.beginStruct(2); w.beginStruct(unit(u)); w.endStruct(); w.endStruct()
            w.endStruct(); w.endStruct()
        case .integer(let bits, let signed):
            w.beginStruct(id); w.beginStruct(10)
            w.fieldHeader(1, .byte); w.bytes.append(UInt8(bits))
            w.bool(2, signed)
            w.endStruct(); w.endStruct()
        case .decimal(let p, let s):
            w.beginStruct(id); w.beginStruct(5)
            w.int(1, s); w.int(2, p)
            w.endStruct(); w.endStruct()
        default: return
        }
    }
}

// MARK: - Snappy (host encoder)

/// A minimal Snappy block compressor: a 4-byte rolling hash finds back-references, everything else is
/// emitted as literals. The reader's GPU decoder is the only consumer that matters here, but the output
/// is ordinary Snappy and pyarrow reads it too.
enum Snappy {
    static func compress(_ input: [UInt8]) -> [UInt8] {
        var out = [UInt8]()
        ParquetWriter.appendVarint(&out, UInt64(input.count))
        guard input.count >= 15 else {
            emitLiteral(&out, input, 0, input.count)
            return out
        }
        let tableBits = 12
        var table = [Int32](repeating: -1, count: 1 << tableBits)
        var i = 0
        var literalStart = 0
        let limit = input.count - 4
        while i < limit {
            let h = hash(input, i, tableBits)
            let candidate = Int(table[Int(h)])
            table[Int(h)] = Int32(i)
            if candidate >= 0, i - candidate < 65536, matches4(input, candidate, i) {
                emitLiteral(&out, input, literalStart, i - literalStart)
                var len = 4
                while i + len < input.count && input[candidate + len] == input[i + len] && len < 64 { len += 1 }
                emitCopy(&out, offset: i - candidate, length: len)
                i += len
                literalStart = i
            } else {
                i += 1
            }
        }
        emitLiteral(&out, input, literalStart, input.count - literalStart)
        return out
    }

    private static func hash(_ b: [UInt8], _ i: Int, _ bits: Int) -> UInt32 {
        let v = UInt32(b[i]) | (UInt32(b[i + 1]) << 8) | (UInt32(b[i + 2]) << 16) | (UInt32(b[i + 3]) << 24)
        return (v &* 0x1e35a7bd) >> UInt32(32 - bits)
    }
    private static func matches4(_ b: [UInt8], _ a: Int, _ c: Int) -> Bool {
        b[a] == b[c] && b[a + 1] == b[c + 1] && b[a + 2] == b[c + 2] && b[a + 3] == b[c + 3]
    }

    private static func emitLiteral(_ out: inout [UInt8], _ b: [UInt8], _ start: Int, _ n: Int) {
        guard n > 0 else { return }
        let v = n - 1
        if v < 60 {
            out.append(UInt8(v << 2))
        } else {
            var bytes = 0
            var x = v
            while x > 0 { bytes += 1; x >>= 8 }
            out.append(UInt8(((59 + bytes) << 2)))
            x = v
            for _ in 0..<bytes { out.append(UInt8(x & 0xFF)); x >>= 8 }
        }
        out.append(contentsOf: b[start..<(start + n)])
    }

    /// Matches are capped at 64 bytes above, so one tag is always enough.
    private static func emitCopy(_ out: inout [UInt8], offset off: Int, length len: Int) {
        precondition(len >= 4 && len <= 64 && off >= 1 && off < 65536)
        if len <= 11 && off < 2048 {
            out.append(UInt8(((off >> 8) << 5) | ((len - 4) << 2) | 1))
            out.append(UInt8(off & 0xFF))
        } else {
            out.append(UInt8(((len - 1) << 2) | 2))
            out.append(UInt8(off & 0xFF)); out.append(UInt8((off >> 8) & 0xFF))
        }
    }
}
