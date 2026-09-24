import Foundation
import Metal

// The structure pass: where every field and record of a CSV file ends, found on the GPU.
//
//   csv_summarize    each thread runs its block of bytes from all five parser states at once
//   csv_scan_local   the blocks' transfer functions are prefix-composed inside each threadgroup
//   csv_scan_groups  ... and across threadgroups, giving every block its true start state and the
//                    index of its first boundary
//   csv_emit         each thread re-runs its block from that state, writing boundary positions
//
// See `CSVSource` for the state machine. The host reads back exactly two numbers (the boundary count
// and the final state) before it sizes the boundary buffer.

/// Mirrors `CsvScan` in `CSVSource`.
struct CSVScanParams {
    var dataStart: UInt32 = 0
    var dataEnd: UInt32 = 0
    var blockBytes: UInt32 = 0
    var nBlocks: UInt32 = 0
    var delim: UInt32 = 0
    var quote: UInt32 = 256
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
    var trans: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0)
    var emit: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0)
}

/// The parser as a table: states 0 line start, 1 field start, 2 unquoted field, 3 inside quotes,
/// 4 a quote seen inside quotes; byte classes 0 other, 1 delimiter, 2 quote, 3 newline (`\n` or `\r`).
enum CSVMachine {
    static func next(_ s: Int, _ c: Int, doubleQuote: Bool) -> Int {
        switch s {
        case 0, 1: return [2, 1, 3, 0][c]
        case 2: return [2, 1, 2, 0][c]
        case 3: return c == 2 ? 4 : 3
        default: return [2, 1, doubleQuote ? 3 : 2, 0][c]
        }
    }

    /// Whether leaving state `s` on class `c` ends a field (delimiter) or a record (newline).
    static func emits(_ s: Int, _ c: Int) -> Bool {
        switch c {
        case 1: return s != 3
        case 3: return s == 1 || s == 2 || s == 4
        default: return false
        }
    }

    static func tables(doubleQuote: Bool) -> (trans: [UInt32], emit: [UInt32]) {
        var trans = [UInt32](repeating: 0, count: 4), emit = [UInt32](repeating: 0, count: 4)
        for c in 0..<4 {
            for s in 0..<5 {
                trans[c] |= UInt32(next(s, c, doubleQuote: doubleQuote)) << UInt32(3 * s)
                if emits(s, c) { emit[c] |= 1 << UInt32(s) }
            }
        }
        return (trans, emit)
    }
}

/// The boundaries of one file: `events[k]` is the byte position where field `k` ends, with bit 31 set
/// when that field is the last of its record.
struct CSVStructure {
    let events: MetalArrowBuffer
    let count: Int
    /// Whether the last record ran to the end of the file with no newline after it.
    let unterminated: Bool
}

extension CSVReader {

    /// Runs the structure pass over [dataStart, dataEnd) of `file`.
    func scanStructure(file: MetalArrowBuffer, dataStart: Int, dataEnd: Int, keep: inout [AnyObject]) throws -> CSVStructure {
        let ctx = context
        let blockBytes = Swift.max(1, options.scanBlockBytes)
        let total = dataEnd - dataStart
        let nBlocks = (total + blockBytes - 1) / blockBytes
        if nBlocks == 0 {
            let ev = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: ctx)
            return CSVStructure(events: ev, count: 0, unterminated: false)
        }
        try Dispatch.checkLength(nBlocks)
        let nGroups = (nBlocks + 255) / 256
        let (trans, emit) = CSVMachine.tables(doubleQuote: options.doubleQuote)
        var P = CSVScanParams()
        P.dataStart = UInt32(dataStart)
        P.dataEnd = UInt32(dataEnd)
        P.blockBytes = UInt32(blockBytes)
        P.nBlocks = UInt32(nBlocks)
        P.delim = UInt32(options.delimiter)
        P.quote = options.quoteChar.map { UInt32($0) } ?? 256
        P.trans = (trans[0], trans[1], trans[2], trans[3])
        P.emit = (emit[0], emit[1], emit[2], emit[3])

        let sums = try MetalArrowBuffer.allocate(byteCount: nBlocks * 24, zeroed: false, context: ctx)
        let incl = try MetalArrowBuffer.allocate(byteCount: nBlocks * 24, zeroed: false, context: ctx)
        let totals = try MetalArrowBuffer.allocate(byteCount: nGroups * 24, zeroed: false, context: ctx)
        let starts = try MetalArrowBuffer.allocate(byteCount: (nGroups + 1) * 8, zeroed: true, context: ctx)
        keep += [sums, incl, totals, starts]

        let pSum = try pipeline("csv_summarize")
        let pLocal = try pipeline("csv_scan_local")
        let pGroups = try pipeline("csv_scan_groups")
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        var nb = UInt32(nBlocks), ng = UInt32(nGroups)
        try ctx.run { enc in
            enc.setComputePipelineState(pSum)
            enc.setBuffer(file.mtl, offset: file.offset, index: 0)
            enc.setBytes(&P, length: MemoryLayout<CSVScanParams>.size, index: 1)
            enc.setBuffer(sums.mtl, offset: sums.offset, index: 2)
            Dispatch.dispatch1D(enc, pSum, count: nBlocks)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pLocal)
            enc.setBuffer(sums.mtl, offset: sums.offset, index: 0)
            enc.setBytes(&nb, length: 4, index: 1)
            enc.setBuffer(incl.mtl, offset: incl.offset, index: 2)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: nGroups, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pGroups)
            enc.setBuffer(totals.mtl, offset: totals.offset, index: 0)
            enc.setBytes(&ng, length: 4, index: 1)
            enc.setBuffer(starts.mtl, offset: starts.offset, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        try ctx.syncPoint()
        let end = starts.typed(UInt32.self)
        let finalState = Int(end[2 * nGroups]), emitted = Int(end[2 * nGroups + 1])
        let unterminated = finalState != 0
        let count = emitted + (unterminated ? 1 : 0)
        let events = try MetalArrowBuffer.allocate(byteCount: Swift.max(count, 1) * 4, zeroed: false, context: ctx)
        // A last record with no newline after it ends at the end of the data.
        if unterminated { events.mutableTyped(UInt32.self)[emitted] = UInt32(dataEnd) | 0x8000_0000 }
        if emitted > 0 {
            let pEmit = try pipeline("csv_emit")
            try ctx.run { enc in
                enc.setComputePipelineState(pEmit)
                enc.setBuffer(file.mtl, offset: file.offset, index: 0)
                enc.setBytes(&P, length: MemoryLayout<CSVScanParams>.size, index: 1)
                enc.setBuffer(incl.mtl, offset: incl.offset, index: 2)
                enc.setBuffer(starts.mtl, offset: starts.offset, index: 3)
                enc.setBuffer(events.mtl, offset: events.offset, index: 4)
                Dispatch.dispatch1D(enc, pEmit, count: nBlocks)
            }
        }
        return CSVStructure(events: events, count: count, unterminated: unterminated)
    }

    /// Index of the first boundary where a record does not have exactly `nCols` fields, or nil.
    func firstRaggedBoundary(_ s: CSVStructure, nCols: Int, keep: inout [AnyObject]) throws -> Int? {
        guard s.count > 0 else { return nil }
        let ctx = context
        let bad = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        bad.mutableTyped(UInt32.self)[0] = UInt32.max
        keep.append(bad)
        let pso = try pipeline("csv_check_rows")
        var n = UInt32(s.count), c = UInt32(nCols)
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(s.events.mtl, offset: s.events.offset, index: 0)
            enc.setBytes(&n, length: 4, index: 1)
            enc.setBytes(&c, length: 4, index: 2)
            enc.setBuffer(bad.mtl, offset: bad.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: s.count)
        }
        try ctx.syncPoint()
        let k = bad.typed(UInt32.self)[0]
        return k == UInt32.max ? nil : Int(k)
    }

    func pipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: CSVSource.source, function: fn, cacheKey: "csv/\(fn)")
    }
}
