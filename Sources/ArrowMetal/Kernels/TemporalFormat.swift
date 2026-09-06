import Foundation
import Metal

// `strftime` and `strptime` on the GPU.
//
// The format string is host data, but it is *tiny* and the same for every row, so it does not belong in
// the shader: it is compiled once into a flat list of `uint4` operations (kind plus three arguments)
// and uploaded as a small constant buffer. One generic kernel then serves every format, and no format
// ever costs a shader recompile.
//
// `strftime` is the two-pass shape `StringCast`'s integer→string cast uses — measure every row, scan
// the lengths into the Arrow offsets buffer on the GPU, write the bytes — because the output width is
// data dependent (`%B` is 3 to 9 bytes, `%Z` 3 to 7, `%Y` widens past four digits). Both passes run the
// same `fmt_row`, so a length and its bytes cannot disagree.
//
// Anything the compiler does not recognise leaves `compile` returning nil and the call falls back to
// the C library on the host, which is still the reference implementation for the specifiers that are
// not listed below.

/// A format string compiled into GPU operations.
struct TemporalFormat {
    /// `(kind, a, b, c)` per operation; the kinds are documented on the kernels.
    var ops: [SIMD4<UInt32>] = []
    /// Bytes the literal operations index into.
    var literals: [UInt8] = []
    /// True when the format asks for `%z` or `%Z`, which need the timezone transition table.
    var needsZone = false

    /// `ARROWMETAL_FORMAT_HOST=1` forces `strftime` / `strptime` back onto the C library, so the tests
    /// can compare the two implementations row for row.
    static let hostOnly = ProcessInfo.processInfo.environment["ARROWMETAL_FORMAT_HOST"] == "1"

    // Operation kinds, shared by both kernels.
    private static let kLiteral: UInt32 = 0, kNumber: UInt32 = 1, kName: UInt32 = 2
    private static let kFraction: UInt32 = 3, kOffset: UInt32 = 4, kZone: UInt32 = 5
    private static let kSpace: UInt32 = 6

    /// Numeric fields, in the order the kernels switch on.
    private enum Field: UInt32 {
        case year = 0, month, day, hour24, minute, second, dayOfYear, year2, hour12
        case century, isoYear, isoWeek, weekdayMonday1, weekdaySunday0
    }

    /// `%X` → (field, printed width, pad: 0 zero / 1 space) for formatting.
    private static let printSpecs: [UInt8: (Field, UInt32, UInt32)] = [
        UInt8(ascii: "Y"): (.year, 4, 0), UInt8(ascii: "m"): (.month, 2, 0),
        UInt8(ascii: "d"): (.day, 2, 0), UInt8(ascii: "e"): (.day, 2, 1),
        UInt8(ascii: "H"): (.hour24, 2, 0), UInt8(ascii: "M"): (.minute, 2, 0),
        UInt8(ascii: "S"): (.second, 2, 0), UInt8(ascii: "j"): (.dayOfYear, 3, 0),
        UInt8(ascii: "y"): (.year2, 2, 0), UInt8(ascii: "I"): (.hour12, 2, 0),
        UInt8(ascii: "C"): (.century, 2, 0), UInt8(ascii: "G"): (.isoYear, 4, 0),
        UInt8(ascii: "V"): (.isoWeek, 2, 0), UInt8(ascii: "u"): (.weekdayMonday1, 1, 0),
        UInt8(ascii: "w"): (.weekdaySunday0, 1, 0),
    ]

    /// `%X` → (field, max digits, lowest, highest) for parsing, following BSD `conv_num`'s ranges.
    private static let parseSpecs: [UInt8: (Field, UInt32, UInt32, UInt32)] = [
        UInt8(ascii: "Y"): (.year, 4, 0, 9999), UInt8(ascii: "m"): (.month, 2, 1, 12),
        UInt8(ascii: "d"): (.day, 2, 1, 31), UInt8(ascii: "e"): (.day, 2, 1, 31),
        UInt8(ascii: "H"): (.hour24, 2, 0, 23), UInt8(ascii: "M"): (.minute, 2, 0, 59),
        UInt8(ascii: "S"): (.second, 2, 0, 61), UInt8(ascii: "y"): (.year2, 2, 0, 99),
        UInt8(ascii: "I"): (.hour12, 2, 1, 12),
    ]

    /// Name tables: month abbreviated, month full, weekday abbreviated, weekday full, AM/PM.
    private static let nameSpecs: [UInt8: UInt32] = [
        UInt8(ascii: "b"): 0, UInt8(ascii: "h"): 0, UInt8(ascii: "B"): 1,
        UInt8(ascii: "a"): 2, UInt8(ascii: "A"): 3, UInt8(ascii: "p"): 4,
    ]

    /// The compound specifiers, expanded before anything else sees them.
    private static let compounds: [UInt8: String] = [
        UInt8(ascii: "F"): "%Y-%m-%d", UInt8(ascii: "T"): "%H:%M:%S",
        UInt8(ascii: "D"): "%m/%d/%y", UInt8(ascii: "R"): "%H:%M",
    ]

    /// Compiles `format` for `strftime`, or nil when it uses a specifier this kernel does not know.
    static func compileFormat(_ format: String) -> TemporalFormat? { compile(format, parsing: false) }

    /// Compiles `format` for `strptime`, or nil when it uses a specifier this kernel does not know.
    /// The parser recognises a strict subset of the formatter: the fields whose width is bounded and
    /// unambiguous. `%j`, `%C`, `%G`, `%V`, `%u`, `%w` and `%Z` have no useful inverse here and send
    /// the call to the host.
    static func compileParse(_ format: String) -> TemporalFormat? { compile(format, parsing: true) }

    private static func compile(_ format: String, parsing: Bool) -> TemporalFormat? {
        guard !hostOnly else { return nil }
        var out = TemporalFormat()
        var pending: [UInt8] = []

        func flushLiteral() {
            guard !pending.isEmpty else { return }
            out.ops.append(SIMD4(kLiteral, UInt32(out.literals.count), UInt32(pending.count), 0))
            out.literals += pending
            pending = []
        }

        func walk(_ bytes: [UInt8], depth: Int) -> Bool {
            guard depth < 3 else { return false }
            var i = 0
            while i < bytes.count {
                let c = bytes[i]
                guard c == UInt8(ascii: "%"), i + 1 < bytes.count else {
                    // In a parse format a run of whitespace matches any run of whitespace, as in C.
                    if parsing, c == 0x20 || (c >= 0x09 && c <= 0x0D) {
                        flushLiteral()
                        out.ops.append(SIMD4(kSpace, 0, 0, 0))
                        while i < bytes.count, bytes[i] == 0x20 || (bytes[i] >= 0x09 && bytes[i] <= 0x0D) { i += 1 }
                        continue
                    }
                    pending.append(c)
                    i += 1
                    continue
                }
                let spec = bytes[i + 1]
                i += 2
                if let expansion = compounds[spec] {
                    if !walk(Array(expansion.utf8), depth: depth + 1) { return false }
                    continue
                }
                switch spec {
                case UInt8(ascii: "%"): pending.append(UInt8(ascii: "%"))
                case UInt8(ascii: "n"): pending.append(0x0A)
                case UInt8(ascii: "t"): pending.append(0x09)
                case UInt8(ascii: "f"):
                    flushLiteral()
                    out.ops.append(SIMD4(kFraction, 0, 0, 0))
                case UInt8(ascii: "z"):
                    flushLiteral()
                    out.needsZone = true
                    out.ops.append(SIMD4(kOffset, 0, 0, 0))
                case UInt8(ascii: "Z"):
                    guard !parsing else { return false }
                    flushLiteral()
                    out.needsZone = true
                    out.ops.append(SIMD4(kZone, 0, 0, 0))
                default:
                    if parsing {
                        if let (field, width, lo, hi) = parseSpecs[spec] {
                            flushLiteral()
                            out.ops.append(SIMD4(kNumber, field.rawValue, width, lo | (hi << 16)))
                        } else if let table = nameSpecs[spec] {
                            flushLiteral()
                            out.ops.append(SIMD4(kName, table, 0, 0))
                        } else { return false }
                    } else {
                        if let (field, width, pad) = printSpecs[spec] {
                            flushLiteral()
                            out.ops.append(SIMD4(kNumber, field.rawValue, width, pad))
                        } else if let table = nameSpecs[spec] {
                            flushLiteral()
                            out.ops.append(SIMD4(kName, table, 0, 0))
                        } else { return false }
                    }
                }
            }
            return true
        }

        guard walk(Array(format.utf8), depth: 0) else { return nil }
        flushLiteral()
        guard !out.ops.isEmpty else { return nil }         // an empty format: let the host answer
        return out
    }

    /// The two small buffers the kernels read.
    func upload(_ ctx: MetalContext) throws -> (ops: MetalArrowBuffer, literals: MetalArrowBuffer) {
        let o = try ops.withUnsafeBytes {
            try MetalArrowBuffer.copy(from: $0.baseAddress!, byteCount: $0.count, context: ctx)
        }
        let l: MetalArrowBuffer
        if literals.isEmpty {
            l = try MetalArrowBuffer.allocate(byteCount: 16, zeroed: true, context: ctx)
        } else {
            l = try literals.withUnsafeBytes {
                try MetalArrowBuffer.copy(from: $0.baseAddress!, byteCount: $0.count, context: ctx)
            }
        }
        return (o, l)
    }
}

/// Mirrors `fmt_params` in `TemporalFormatSource`.
private struct FormatParams {
    var divisor: Int64
    var mode: UInt32
    var nops: UInt32
    var count: UInt32
    var hasValidity: UInt32
    var hasTZ: UInt32
    var pad: UInt32 = 0
}

/// Mirrors `prs_params` in `TemporalFormatSource`.
private struct ParseParams {
    var scale: Int64
    var subScale: Int64
    var subDivide: Int64
    var nops: UInt32
    var hasValidity: UInt32
}

extension MetalTemporalArray {

    /// `strftime` on the GPU, or nil when the format uses a specifier the kernel does not know (the
    /// caller then falls back to the C library).
    ///
    /// A `timestamp` carrying a timezone is formatted **in that zone**, which is what pyarrow does and
    /// what makes `%z` and `%Z` meaningful; a naive column, a `date` and a `time` are UTC. `%S` stays
    /// two digits and `%f` is the six-digit fraction, which is this package's documented C-`strftime`
    /// reading and differs from pyarrow's (pyarrow folds the fraction into `%S` and leaves `%f` alone).
    func strftimeGPU(_ format: String) throws -> MetalStringArray? {
        guard let (mode, divisor) = type.extraction, !context.isVirtualDevice,
              let compiled = TemporalFormat.compileFormat(format) else { return nil }
        let ctx = context, n = length
        try Dispatch.checkLength(n)

        // The zone to format in: the column's own when it has one, otherwise UTC.
        var zoneTable = TimeZoneTable.table(for: "UTC", context: ctx)
        var hasTZ: UInt32 = 0
        if case .timestamp(_, let tz) = type, let tz, !tz.isEmpty {
            guard let t = TimeZoneTable.table(for: tz, context: ctx) else { return nil }
            zoneTable = t
            hasTZ = 1
        }
        guard let zone = zoneTable else { return nil }

        let wide = try int64Values()
        let (opsBuf, litBuf) = try compiled.upload(ctx)
        var P = FormatParams(divisor: divisor, mode: mode == 0 ? 0 : 1, nops: UInt32(compiled.ops.count),
                             count: UInt32(zone.count), hasValidity: validity == nil ? 0 : 1, hasTZ: hasTZ)
        let vb = validity ?? wide.values
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)

        func bindCommon(_ enc: MTLComputeCommandEncoder) {
            enc.setBuffer(wide.values.mtl, offset: wide.values.offset, index: 0)
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBytes(&P, length: MemoryLayout<FormatParams>.size, index: 3)
            enc.setBuffer(opsBuf.mtl, offset: opsBuf.offset, index: 4)
            enc.setBuffer(litBuf.mtl, offset: litBuf.offset, index: 5)
            enc.setBuffer(zone.bufTransUTC.mtl, offset: zone.bufTransUTC.offset, index: 6)
            enc.setBuffer(zone.bufOffsets.mtl, offset: zone.bufOffsets.offset, index: 7)
            enc.setBuffer(zone.bufAbbrev.mtl, offset: zone.bufAbbrev.offset, index: 8)
        }

        if n > 0 {
            let pso = try ctx.pipeline(source: TemporalFormatSource.source, function: "fmt_lengths",
                                       cacheKey: "temporalformat/fmt_lengths")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                bindCommon(enc)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 9)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens,
                                               context: ctx).exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pso = try ctx.pipeline(source: TemporalFormatSource.source, function: "fmt_write",
                                       cacheKey: "temporalformat/fmt_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                bindCommon(enc)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 9)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 10)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        ctx.retainUntilFlush(lens)
        ctx.retainUntilFlush(opsBuf)
        ctx.retainUntilFlush(litBuf)
        return MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                offsets: outOffsets, data: outData, context: ctx)
    }
}

extension MetalStringArray {

    /// `strptime` on the GPU, or nil when the format uses a specifier the parser does not know.
    ///
    /// The grammar mirrors the C library's: a numeric field takes at least one and at most its own
    /// width in digits and must land in its range, a run of whitespace in the format matches any run of
    /// whitespace, every other literal must match exactly, and the whole value must be consumed. Month
    /// and weekday names match either spelling, case-insensitively. `%f` is the inverse of this
    /// package's `%f` extension — up to six fractional digits — which the C library has no notion of.
    func strptimeGPU(_ format: String, unit: ArrowTemporalUnit, timezone: String?) throws -> MetalTemporalArray? {
        guard !context.isVirtualDevice, let compiled = TemporalFormat.compileParse(format) else { return nil }
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let scale = unit.perSecond
        let (opsBuf, litBuf) = try compiled.upload(ctx)
        let words = BitmapOps.words(bits: n)
        let outVals = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 8, 8), zeroed: true, context: ctx)
        let outValid = try MetalArrowBuffer.allocate(byteCount: Swift.max(words * 4, 4), zeroed: true, context: ctx)
        var P = ParseParams(scale: scale,
                            subScale: scale >= 1_000_000 ? scale / 1_000_000 : 0,
                            subDivide: scale < 1_000_000 ? 1_000_000 / scale : 0,
                            nops: UInt32(compiled.ops.count), hasValidity: validity == nil ? 0 : 1)
        if n > 0 {
            let vb = validity ?? offsets
            let pso = try ctx.pipeline(source: TemporalFormatSource.source, function: "prs_parse",
                                       cacheKey: "temporalformat/prs_parse")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                enc.setBytes(&P, length: MemoryLayout<ParseParams>.size, index: 4)
                enc.setBuffer(opsBuf.mtl, offset: opsBuf.offset, index: 5)
                enc.setBuffer(litBuf.mtl, offset: litBuf.offset, index: 6)
                enc.setBuffer(outVals.mtl, offset: outVals.offset, index: 7)
                enc.setBuffer(outValid.mtl, offset: outValid.offset, index: 8)
                Dispatch.dispatch1D(enc, pso, count: words)
            }
        }
        ctx.retainUntilFlush(opsBuf)
        ctx.retainUntilFlush(litBuf)
        let out = MetalArray<Int64>(length: n, nullCount: 0, validity: outValid, values: outVals, context: ctx)
        out.recomputeNullCount()
        return try MetalTemporalArray(type: .timestamp(unit, timezone: timezone), out)
    }
}
