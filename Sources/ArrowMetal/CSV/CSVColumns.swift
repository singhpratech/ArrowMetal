import Foundation
import Metal

/// Mirrors `CsvCol` in `CSVSource`.
struct CSVColParams {
    var nRows: UInt32 = 0
    var nCols: UInt32 = 0
    var col: UInt32 = 0
    var firstRecord: UInt32 = 0
    var dataStart: UInt32 = 0
    var dataEnd: UInt32 = 0
    var quote: UInt32 = 256
    var doubleQuote: UInt32 = 1
    var flags: UInt32 = 0
    var decimalPoint: UInt32 = 0x2E
    var nNull: UInt32 = 0
    var nTrue: UInt32 = 0
    var nFalse: UInt32 = 0
    var unitDigits: UInt32 = 0
    var expectZone: UInt32 = 0
    var isSigned: UInt32 = 1
    var maxHex: UInt32 = 16
    var pad0: UInt32 = 0
    var limPos: UInt64 = 0
    var limNeg: UInt64 = 0
}

/// Inference outcomes, in Arrow's order: a column takes the first kind every one of its values
/// converts as (`arrow/csv/inference_internal.h`).
enum CSVKind: Int, CaseIterable {
    case null = 0, int64, bool, date32, time32, timestamp, timestampNS, timestampTZ, timestampTZNS, float64, utf8, binary

    var type: CSVColumnType {
        switch self {
        case .null: return .null
        case .int64: return .int64
        case .bool: return .bool
        case .date32: return .date32
        case .time32: return .time32(.second)
        case .timestamp: return .timestamp(.second, timezone: nil)
        case .timestampNS: return .timestamp(.nano, timezone: nil)
        case .timestampTZ: return .timestamp(.second, timezone: "UTC")
        case .timestampTZNS: return .timestamp(.nano, timezone: "UTC")
        case .float64: return .float64
        case .utf8: return .utf8
        case .binary: return .binary
        }
    }

    static func first(in mask: UInt32) -> CSVKind {
        for k in allCases where mask & (1 << UInt32(k.rawValue)) != 0 { return k }
        return .binary
    }
}

/// Converts the projected columns of one parsed file. Every phase is dispatched for all columns before
/// the host waits on any of them, so a read costs a handful of GPU round trips however wide it is.
final class CSVColumnConverter {
    let reader: CSVReader
    let ctx: MetalContext
    let file: MetalArrowBuffer
    let events: MetalArrowBuffer
    let nCols: Int
    let firstRecord: Int
    let nRows: Int
    let dataStart: Int
    let dataEnd: Int
    private var lists: MetalArrowBuffer! = nil
    private var listBytes: MetalArrowBuffer! = nil
    private var dummy: MetalArrowBuffer! = nil

    init(reader: CSVReader, file: MetalArrowBuffer, events: MetalArrowBuffer, nCols: Int, firstRecord: Int,
         nRows: Int, dataStart: Int, dataEnd: Int) {
        self.reader = reader
        self.ctx = reader.context
        self.file = file
        self.events = events
        self.nCols = nCols
        self.firstRecord = firstRecord
        self.nRows = nRows
        self.dataStart = dataStart
        self.dataEnd = dataEnd
    }

    /// One projected column on its way through the phases.
    final class Work {
        let source: Int
        let forced: CSVColumnType?
        var spans: MetalArrowBuffer! = nil
        var complexCount: MetalArrowBuffer! = nil
        var side: MetalArrowBuffer! = nil
        var mask: MetalArrowBuffer! = nil
        var type: CSVColumnType = .null
        var inferred = false
        var values: MetalArrowBuffer! = nil
        var wide: MetalArrowBuffer! = nil
        var validity: MetalArrowBuffer! = nil
        var err: MetalArrowBuffer! = nil
        var host: MetalArrowBuffer! = nil
        var hostCount: MetalArrowBuffer! = nil
        init(source: Int, forced: CSVColumnType?) { self.source = source; self.forced = forced }
    }

    private var options: CSVReadOptions { reader.options }
    private var words: Int { BitmapOps.words(bits: nRows) }

    func params(_ w: Work) -> CSVColParams {
        let o = options
        var P = CSVColParams()
        P.nRows = UInt32(nRows)
        P.nCols = UInt32(nCols)
        P.col = UInt32(w.source)
        P.firstRecord = UInt32(firstRecord)
        P.dataStart = UInt32(dataStart)
        P.dataEnd = UInt32(dataEnd)
        P.quote = o.quoteChar.map { UInt32($0) } ?? 256
        P.doubleQuote = o.doubleQuote ? 1 : 0
        P.flags = (o.quotedStringsCanBeNull ? 1 : 0) | (o.stringsCanBeNull ? 2 : 0) | (o.checkUTF8 ? 4 : 0)
        P.decimalPoint = UInt32(o.decimalPoint)
        P.nNull = UInt32(o.nullValues.count)
        P.nTrue = UInt32(o.trueValues.count)
        P.nFalse = UInt32(o.falseValues.count)
        return P
    }

    private func buildLists(keep: inout [AnyObject]) throws {
        let o = options
        var entries: [UInt32] = []
        var bytes: [UInt8] = []
        for s in o.nullValues + o.trueValues + o.falseValues {
            let u = Array(s.utf8)
            entries += [UInt32(bytes.count), UInt32(u.count)]
            bytes += u
        }
        if entries.isEmpty { entries = [0, 0] }
        if bytes.isEmpty { bytes = [0] }
        lists = try entries.withUnsafeBytes { try MetalArrowBuffer.copy(from: $0.baseAddress!, byteCount: $0.count, context: ctx) }
        listBytes = try bytes.withUnsafeBytes { try MetalArrowBuffer.copy(from: $0.baseAddress!, byteCount: $0.count, context: ctx) }
        dummy = try MetalArrowBuffer.allocate(byteCount: 16, zeroed: true, context: ctx)
        keep += [lists, listBytes, dummy]
    }

    private func alloc(_ bytes: Int, zeroed: Bool = false, _ keep: inout [AnyObject]) throws -> MetalArrowBuffer {
        let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 16), zeroed: zeroed, context: ctx)
        keep.append(b)
        return b
    }

    func convert(plan: [(name: String, source: Int?)], keep: inout [AnyObject]) throws -> [AnyMetalArray] {
        try buildLists(keep: &keep)
        var work: [Work?] = plan.map { p in
            p.source.map { Work(source: $0, forced: options.columnTypes[p.name]) }
        }
        let live = work.compactMap { $0 }

        // Phase 1: content spans; complex fields (doubled quotes, text after a closing quote) unescaped.
        for w in live {
            w.spans = try alloc(nRows * 8, &keep)
            w.complexCount = try alloc(4, zeroed: true, &keep)
            w.side = dummy
            if nRows > 0 { try spans(w) }
        }
        try ctx.syncPoint()
        for w in live where w.complexCount.typed(UInt32.self)[0] > 0 { try unescape(w, keep: &keep) }

        // Phase 2: inference for every column without an override.
        for w in live where w.forced == nil {
            w.mask = try alloc(4, &keep)
            w.mask.mutableTyped(UInt32.self)[0] = 0xFFF
            w.inferred = true
            if nRows > 0 { try classify(w) }
        }
        try ctx.syncPoint()
        for w in live {
            w.type = w.forced ?? CSVKind.first(in: w.mask.typed(UInt32.self)[0]).type
        }

        // Phase 3: conversion.
        for w in live { try dispatchConversion(w, keep: &keep) }
        try ctx.syncPoint()
        for w in live { try checkErrors(w) }

        var out: [AnyMetalArray] = []
        for (j, p) in plan.enumerated() {
            if let w = work[j] {
                out.append(try finish(w, keep: &keep))
            } else {
                out.append(try allNull(options.columnTypes[p.name] ?? .null))
            }
            work[j] = nil
        }
        return out
    }

    // MARK: phases

    private func spans(_ w: Work) throws {
        var P = params(w)
        let pso = try reader.pipeline("csv_spans")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(file.mtl, offset: file.offset, index: 0)
            enc.setBuffer(events.mtl, offset: events.offset, index: 1)
            enc.setBytes(&P, length: MemoryLayout<CSVColParams>.size, index: 2)
            enc.setBuffer(w.spans.mtl, offset: w.spans.offset, index: 3)
            enc.setBuffer(w.complexCount.mtl, offset: w.complexCount.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: nRows)
        }
    }

    private func unescape(_ w: Work, keep: inout [AnyObject]) throws {
        var P = params(w)
        let lens = try alloc(nRows * 4, &keep)
        let pLen = try reader.pipeline("csv_side_len")
        try ctx.run { enc in
            enc.setComputePipelineState(pLen)
            enc.setBuffer(w.spans.mtl, offset: w.spans.offset, index: 0)
            enc.setBytes(&P, length: MemoryLayout<CSVColParams>.size, index: 1)
            enc.setBuffer(lens.mtl, offset: lens.offset, index: 2)
            Dispatch.dispatch1D(enc, pLen, count: nRows)
        }
        let offsets = try MetalArray<Int32>(length: nRows, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        keep.append(offsets)
        let total = Int(offsets.typed(Int32.self)[nRows])
        guard total >= 0 else { throw CSVError.io("unescaped quoted fields of one column exceed 2 GiB") }
        w.side = try alloc(total, &keep)
        let pUn = try reader.pipeline("csv_unescape")
        try ctx.run { enc in
            enc.setComputePipelineState(pUn)
            enc.setBuffer(file.mtl, offset: file.offset, index: 0)
            enc.setBuffer(events.mtl, offset: events.offset, index: 1)
            enc.setBytes(&P, length: MemoryLayout<CSVColParams>.size, index: 2)
            enc.setBuffer(w.spans.mtl, offset: w.spans.offset, index: 3)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 4)
            enc.setBuffer(w.side.mtl, offset: w.side.offset, index: 5)
            Dispatch.dispatch1D(enc, pUn, count: nRows)
        }
    }

    private func bindRow(_ enc: MTLComputeCommandEncoder, _ w: Work, _ P: inout CSVColParams) {
        enc.setBuffer(file.mtl, offset: file.offset, index: 0)
        enc.setBuffer(w.side.mtl, offset: w.side.offset, index: 1)
        enc.setBuffer(w.spans.mtl, offset: w.spans.offset, index: 2)
        enc.setBytes(&P, length: MemoryLayout<CSVColParams>.size, index: 3)
        enc.setBuffer(lists.mtl, offset: lists.offset, index: 4)
        enc.setBuffer(listBytes.mtl, offset: listBytes.offset, index: 5)
    }

    private func classify(_ w: Work) throws {
        var P = params(w)
        let pso = try reader.pipeline("csv_classify")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            bindRow(enc, w, &P)
            enc.setBuffer(w.mask.mtl, offset: w.mask.offset, index: 6)
            Dispatch.dispatch1D(enc, pso, count: nRows)
        }
    }

    private func intLimits(_ t: CSVColumnType) -> (signed: Bool, width: Int, limPos: UInt64, limNeg: UInt64)? {
        switch t {
        case .int8: return (true, 1, 127, 128)
        case .int16: return (true, 2, 32767, 32768)
        case .int32: return (true, 4, 2_147_483_647, 2_147_483_648)
        case .int64: return (true, 8, UInt64(Int64.max), UInt64(Int64.max) + 1)
        case .uint8: return (false, 1, 255, 0)
        case .uint16: return (false, 2, 65535, 0)
        case .uint32: return (false, 4, 4_294_967_295, 0)
        case .uint64: return (false, 8, UInt64.max, 0)
        default: return nil
        }
    }

    private func unitDigits(_ u: ArrowTemporalUnit) -> UInt32 {
        switch u { case .second: return 0; case .milli: return 3; case .micro: return 6; case .nano: return 9 }
    }

    private func dispatchConversion(_ w: Work, keep: inout [AnyObject]) throws {
        var P = params(w)
        w.validity = try alloc(words * 4, &keep)
        w.err = try alloc(8, &keep)
        w.err.mutableTyped(UInt32.self)[0] = UInt32.max
        w.err.mutableTyped(UInt32.self)[1] = UInt32.max
        var fn: String
        var extra: ((MTLComputeCommandEncoder) -> Void)? = nil
        switch w.type {
        case .null:
            if w.inferred { return }                              // every row is already known to be null
            fn = "csv_conv_null"
        case .bool:
            fn = "csv_conv_bool"
            w.values = try alloc(words * 4, &keep)
        case .float64, .float32:
            let f32 = w.type == .float32
            fn = f32 ? "csv_conv_f32" : "csv_conv_f64"
            w.values = try alloc(nRows * (f32 ? 4 : 8), &keep)
            w.host = try alloc(words * 4, &keep)
            w.hostCount = try alloc(4, zeroed: true, &keep)
            let host = w.host!, hostCount = w.hostCount!
            extra = { enc in
                enc.setBuffer(host.mtl, offset: host.offset, index: 9)
                enc.setBuffer(hostCount.mtl, offset: hostCount.offset, index: 10)
            }
        case .date32:
            fn = "csv_conv_date"
            w.values = try alloc(nRows * 4, &keep)
        case .time32(let u):
            fn = "csv_conv_time32"
            P.unitDigits = unitDigits(u)
            w.values = try alloc(nRows * 4, &keep)
        case .time64(let u):
            fn = "csv_conv_time64"
            P.unitDigits = unitDigits(u)
            w.values = try alloc(nRows * 8, &keep)
        case .timestamp(let u, let tz):
            fn = "csv_conv_ts"
            P.unitDigits = unitDigits(u)
            P.expectZone = (tz?.isEmpty == false) ? 1 : 0
            w.values = try alloc(nRows * 8, &keep)
        case .utf8, .binary:
            fn = "csv_str_len"
            w.values = try alloc(nRows * 4, &keep)
            var check: UInt32 = (w.type == .utf8 && !w.inferred && options.checkUTF8) ? 1 : 0
            extra = { enc in enc.setBytes(&check, length: 4, index: 9) }
        default:
            guard let lim = intLimits(w.type) else { throw CSVError.invalidOptions("unsupported column type \(w.type.arrowName)") }
            fn = "csv_conv_int"
            P.isSigned = lim.signed ? 1 : 0
            P.maxHex = UInt32(2 * lim.width)
            P.limPos = lim.limPos
            P.limNeg = lim.limNeg
            w.wide = try alloc(nRows * 8, &keep)
            if lim.width < 8 { w.values = try alloc(nRows * lim.width, &keep) } else { w.values = w.wide }
        }
        guard nRows > 0 else { return }
        let pso = try reader.pipeline(fn)
        let out = w.wide ?? w.values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            bindRow(enc, w, &P)
            if let out { enc.setBuffer(out.mtl, offset: out.offset, index: 6) }
            enc.setBuffer(w.validity.mtl, offset: w.validity.offset, index: 7)
            enc.setBuffer(w.err.mtl, offset: w.err.offset, index: 8)
            extra?(enc)
            Dispatch.dispatch1D(enc, pso, count: nRows)
        }
        if let lim = intLimits(w.type), lim.width < 8 {
            let pN = try reader.pipeline("csv_narrow")
            var n = UInt32(nRows), width = UInt32(lim.width)
            try ctx.run { enc in
                enc.setComputePipelineState(pN)
                enc.setBuffer(w.wide.mtl, offset: w.wide.offset, index: 0)
                enc.setBuffer(w.values.mtl, offset: w.values.offset, index: 1)
                enc.setBytes(&n, length: 4, index: 2)
                enc.setBytes(&width, length: 4, index: 3)
                Dispatch.dispatch1D(enc, pN, count: nRows)
            }
        }
    }

    /// The (unescaped) text of row `r`, for error messages and the CPU float fallback.
    private func text(_ w: Work, _ r: Int) -> [UInt8] {
        let sp = w.spans.typed(UInt32.self)
        let start = Int(sp[2 * r]), word = sp[2 * r + 1]
        let len = Int(word & 0x3FFF_FFFF)
        let base = (word & 0x4000_0000) != 0 ? w.side.contents : file.contents
        return Array(UnsafeRawBufferPointer(start: base.advanced(by: start), count: len))
    }

    private func checkErrors(_ w: Work) throws {
        guard nRows > 0, let err = w.err else { return }
        let e = err.typed(UInt32.self)
        let invalid = e[0], zone = e[1]
        guard invalid != UInt32.max || zone != UInt32.max else {
            try hostFloats(w)
            return
        }
        let prefix = "In CSV column #\(w.source): CSV conversion error to \(w.type.arrowName): "
        if w.type == .utf8 {
            throw CSVError.conversion(prefix + "invalid UTF8 data")
        }
        let row = Int(Swift.min(invalid, zone))
        let value = String(decoding: text(w, row), as: UTF8.self)
        if case .timestamp(_, let tz) = w.type, zone < invalid {
            if tz?.isEmpty == false {
                throw CSVError.conversion(prefix + "expected a zone offset in '\(value)'. If these timestamps are in local time, parse them as timestamps without timezone, then call assume_timezone.")
            }
            throw CSVError.conversion(prefix + "expected no zone offset in '\(value)'")
        }
        throw CSVError.conversion(prefix + "invalid value '\(value)'")
    }

    /// Values the GPU float parser left undecided: parsed on the CPU (correctly rounded, like the GPU).
    private func hostFloats(_ w: Work) throws {
        guard let hc = w.hostCount, hc.typed(UInt32.self)[0] > 0 else { return }
        let flags = w.host.typed(UInt32.self)
        let bits = w.validity.mutableTyped(UInt32.self)
        let dp = options.decimalPoint
        for wd in 0..<words where flags[wd] != 0 {
            var m = flags[wd]
            while m != 0 {
                let r = wd * 32 + m.trailingZeroBitCount
                m &= m - 1
                var t = text(w, r)
                while let f = t.first, f == 0x20 || f == 0x09 { t.removeFirst() }
                while let l = t.last, l == 0x20 || l == 0x09 { t.removeLast() }
                t = t.map { $0 == dp ? 0x2E : $0 }
                let s = String(decoding: t, as: UTF8.self)
                if w.type == .float32 {
                    guard let v = Float(s) else { throw CSVError.conversion("In CSV column #\(w.source): CSV conversion error to float: invalid value '\(s)'") }
                    w.values.mutableTyped(Float.self)[r] = v
                } else {
                    guard let v = Double(s) else { throw CSVError.conversion("In CSV column #\(w.source): CSV conversion error to double: invalid value '\(s)'") }
                    w.values.mutableTyped(Double.self)[r] = v
                }
                bits[r >> 5] |= 1 << UInt32(r & 31)
            }
        }
    }

    // MARK: results

    /// The validity bitmap, or nil when every row is valid; and the null count.
    private func validity(_ w: Work) -> (MetalArrowBuffer?, Int) {
        let valid = Bitmap.popcount(w.validity.typed(UInt8.self), bits: nRows)
        return valid == nRows ? (nil, 0) : (w.validity, nRows - valid)
    }

    private func finish(_ w: Work, keep: inout [AnyObject]) throws -> AnyMetalArray {
        let n = nRows
        switch w.type {
        case .null: return .null(MetalNullArray(length: n, context: ctx))
        case .bool:
            let (v, nulls) = validity(w)
            return .boolean(MetalBooleanArray(length: n, nullCount: nulls, validity: v, values: w.values, context: ctx))
        case .utf8, .binary:
            return try strings(w, keep: &keep)
        case .date32, .time32:
            let (v, nulls) = validity(w)
            let a = MetalArray<Int32>(length: n, nullCount: nulls, validity: v, values: w.values, context: ctx)
            return .temporal(try MetalTemporalArray(type: temporalType(w.type), a))
        case .time64, .timestamp:
            let (v, nulls) = validity(w)
            let a = MetalArray<Int64>(length: n, nullCount: nulls, validity: v, values: w.values, context: ctx)
            return .temporal(try MetalTemporalArray(type: temporalType(w.type), a))
        case .float32: return .float32(prim(w))
        case .float64: return .float64(prim(w))
        case .int8: return .int8(prim(w))
        case .int16: return .int16(prim(w))
        case .int32: return .int32(prim(w))
        case .int64: return .int64(prim(w))
        case .uint8: return .uint8(prim(w))
        case .uint16: return .uint16(prim(w))
        case .uint32: return .uint32(prim(w))
        case .uint64: return .uint64(prim(w))
        }
    }

    private func prim<T: ArrowPrimitive>(_ w: Work) -> MetalArray<T> {
        let (v, nulls) = validity(w)
        return MetalArray<T>(length: nRows, nullCount: nulls, validity: v, values: w.values, context: ctx)
    }

    private func temporalType(_ t: CSVColumnType) -> ArrowTemporalType {
        switch t {
        case .date32: return .date32
        case .time32(let u): return .time32(u)
        case .time64(let u): return .time64(u)
        case .timestamp(let u, let tz): return .timestamp(u, timezone: tz)
        default: return .date32
        }
    }

    private func strings(_ w: Work, keep: inout [AnyObject]) throws -> AnyMetalArray {
        let n = nRows
        let offsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: w.values, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(offsets.typed(Int32.self)[n])
        guard total >= 0 else {
            throw CSVError.io("column \(w.source) holds more than the 2 GiB an Arrow utf8 array can address")
        }
        let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 && total > 0 {
            var P = params(w)
            let pso = try reader.pipeline("csv_str_copy")
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(file.mtl, offset: file.offset, index: 0)
                enc.setBuffer(w.side.mtl, offset: w.side.offset, index: 1)
                enc.setBuffer(w.spans.mtl, offset: w.spans.offset, index: 2)
                enc.setBytes(&P, length: MemoryLayout<CSVColParams>.size, index: 3)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 4)
                enc.setBuffer(data.mtl, offset: data.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        let (v, nulls) = validity(w)
        let s = MetalStringArray(length: n, nullCount: nulls, validity: v, offsets: offsets, data: data, context: ctx)
        if w.type == .binary { s.isBinary = true; return .binary(s) }
        return .string(s)
    }

    /// An all-null column of `t` (include_missing_columns).
    private func allNull(_ t: CSVColumnType) throws -> AnyMetalArray {
        let n = nRows
        func zeros(_ bytes: Int) throws -> MetalArrowBuffer {
            try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: true, context: ctx)
        }
        func p<T: ArrowPrimitive>(_: T.Type) throws -> MetalArray<T> {
            MetalArray<T>(length: n, nullCount: n, validity: try zeros(Bitmap.byteCount(bits: n)),
                          values: try zeros(n * T.byteWidth), context: ctx)
        }
        switch t {
        case .null: return .null(MetalNullArray(length: n, context: ctx))
        case .bool:
            return .boolean(MetalBooleanArray(length: n, nullCount: n, validity: try zeros(Bitmap.byteCount(bits: n)),
                                              values: try zeros(Bitmap.byteCount(bits: n)), context: ctx))
        case .utf8, .binary:
            let s = MetalStringArray(length: n, nullCount: n, validity: try zeros(Bitmap.byteCount(bits: n)),
                                     offsets: try zeros((n + 1) * 4), data: try zeros(1), context: ctx)
            if t == .binary { s.isBinary = true; return .binary(s) }
            return .string(s)
        case .date32, .time32: return .temporal(try MetalTemporalArray(type: temporalType(t), try p(Int32.self)))
        case .time64, .timestamp: return .temporal(try MetalTemporalArray(type: temporalType(t), try p(Int64.self)))
        case .float32: return .float32(try p(Float.self))
        case .float64: return .float64(try p(Double.self))
        case .int8: return .int8(try p(Int8.self))
        case .int16: return .int16(try p(Int16.self))
        case .int32: return .int32(try p(Int32.self))
        case .int64: return .int64(try p(Int64.self))
        case .uint8: return .uint8(try p(UInt8.self))
        case .uint16: return .uint16(try p(UInt16.self))
        case .uint32: return .uint32(try p(UInt32.self))
        case .uint64: return .uint64(try p(UInt64.self))
        }
    }
}
