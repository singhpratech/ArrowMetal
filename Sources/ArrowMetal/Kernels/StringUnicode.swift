import Foundation
import Metal

/// Full-Unicode `utf8_upper` / `utf8_lower` / `utf8_swapcase` / `utf8_capitalize` / `utf8_title` and
/// the `utf8_trim*` family, split **per row** between the GPU and the host.
///
/// ## How the split works
///
/// The two-pass transform pattern already computes one output byte length per row before it writes
/// anything, which is exactly the hook a hybrid needs. `su_tf_len` returns -1 for a row it cannot
/// answer exactly and records that in a flag byte; the host then decides those rows with Swift's
/// `Unicode.Scalar` tables (sharded over `DispatchQueue.concurrentPerform` chunks of 4096 rows),
/// writes their lengths into the same buffer, and the usual GPU scan turns the whole column into the
/// Arrow offsets buffer. `su_tf_write` fills the rows the GPU claimed and the host memcpys its own
/// into the gaps — the buffers are Metal shared memory, so both write into the same allocation.
///
/// A row goes to the host when
///
/// * a case transform meets a code point above U+017F — so Greek, Cyrillic, Armenian, CJK, emoji and
///   combining marks are host-side, and Latin text of every accent is not; or
/// * a trim whose character set has a non-ASCII member meets a row with a byte ≥ 0x80. A trim with an
///   all-ASCII set never leaves the device at all: a continuation byte is always ≥ 0x80, so trimming
///   ASCII bytes off the ends can never split a UTF-8 sequence.
///
/// An all-Latin column therefore never touches the CPU, a Greek column never touches the GPU past the
/// length pass, and a mixed column pays for each row exactly once.
///
/// ## Agreement with pyarrow
///
/// The host mapping is ``UnicodeClass/simpleUpper(_:)`` / ``UnicodeClass/simpleLower(_:)`` — Unicode's
/// **simple** 1:1 mappings, reconstructed from Swift's full ones, which is what utf8proc and therefore
/// pyarrow use. `"ß"` upper-cases to `"ẞ"` rather than `"SS"`, `"ﬁ"` and `"ŉ"` stay put because their
/// full mappings are two characters long, `"İ"` lower-cases to `"i"` alone, and `"Σ"` lower-cases to
/// `"σ"` in every position, final or not — Swift's `String.lowercased()` would apply the contextual
/// final-sigma rule and disagree, which is why the mapping is done one scalar at a time.
///
/// `utf8_swapcase` follows Arrow's rule that a **titlecase** letter is both upper and lower and so
/// stays where it is: `"ǅ"` swaps to `"ǅ"`, while `"Ǆ"` swaps to `"ǆ"` and `"ǆ"` to `"Ǆ"`.
public enum UnicodeTransform: Int, Sendable, CaseIterable {
    case upper = 0
    case lower = 1
    case swapcase = 2
    case capitalize = 3
    case title = 4
    case trim = 5
    case ltrim = 6
    case rtrim = 7
}

extension UnicodeClass {
    /// Arrow's `utf8_swapcase` on one code point: a titlecase letter — both upper and lower — stays
    /// put; otherwise upper-case becomes lower and lower-case becomes upper.
    public static func simpleSwap(_ u: Unicode.Scalar) -> Unicode.Scalar {
        let up = isUpper(u), lo = isLower(u)
        if up && lo { return u }
        if up { return simpleLower(u) }
        if lo { return simpleUpper(u) }
        return u
    }

    /// The host form of one ``UnicodeTransform`` case transform, applied scalar by scalar.
    static func map(_ op: UnicodeTransform, _ s: String) -> String {
        var out = String.UnicodeScalarView()
        var first = true, boundary = true
        for u in s.unicodeScalars {
            switch op {
            case .upper: out.append(simpleUpper(u))
            case .lower: out.append(simpleLower(u))
            case .swapcase: out.append(simpleSwap(u))
            case .capitalize:
                out.append(first ? simpleUpper(u) : simpleLower(u))
            case .title:
                if isCased(u) { out.append(boundary ? simpleUpper(u) : simpleLower(u)); boundary = false }
                else { out.append(u); boundary = true }
            default: out.append(u)
            }
            first = false
        }
        return String(out)
    }
}

extension MetalStringArray {

    private func suPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringUnicodeSource.source, function: fn, cacheKey: "stru/\(fn)")
    }

    /// The hybrid two-pass driver: GPU length pass, host fixup of the rows the GPU declined, GPU scan
    /// into offsets, GPU byte pass, host fixup of the same rows' bytes.
    ///
    /// `hostOnlyIfNonASCII` makes the kernel decline any row with a byte ≥ 0x80 (the trims with a
    /// non-ASCII character set); the case transforms decline per code point instead.
    func unicodeTransform(_ op: UnicodeTransform, arg1: [UInt8] = [],
                          hostOnlyIfNonASCII: Bool = false,
                          host: @escaping (String) -> String) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let hostRows = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), context: ctx)
        let a1 = try sxArgBuffer(arg1)
        let vb = validity ?? a1                                  // never read when flags bit 0 is clear
        let scratch = try MetalArrowBuffer.allocate(byteCount: 1, zeroed: false, context: ctx)
        // One word the length kernel sets when ANY row is declined: the host then knows whether it has
        // anything to do without walking n flag bytes (or allocating n optionals) for an all-GPU column.
        let declined = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: ctx)
        var prm = Self.sbParams(op.rawValue, arg1.count, hostOnlyIfNonASCII ? 1 : 0, 0, 0,
                                validity == nil ? 0 : 1)

        if n > 0 {
            let pLen = try suPipeline("su_tf_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                enc.setBuffer(hostRows.mtl, offset: hostRows.offset, index: 7)
                enc.setBuffer(scratch.mtl, offset: scratch.offset, index: 8)
                enc.setBuffer(declined.mtl, offset: declined.offset, index: 9)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }

        // The rows the GPU declined, decided on the host and their lengths written into the same
        // buffer the scan is about to read.
        // Only the declined rows exist on the host: their indices (found by scanning the flag bytes
        // eight at a time and skipping zero words) and their mapped strings, never an n-sized array.
        var hostIndex: [Int32] = []
        var hostValues: [String] = []
        var anyHost = false
        if n > 0 {
            try ctx.syncPoint()
            anyHost = declined.typed(UInt32.self)[0] != 0
            withExtendedLifetime(self) {
                guard anyHost else { return }
                let flags = hostRows.typed(UInt8.self)
                let words = n / 8
                flags.withMemoryRebound(to: UInt64.self, capacity: Swift.max(words, 1)) { w in
                    for k in 0..<words where w[k] != 0 {
                        for i in (k * 8)..<(k * 8 + 8) where flags[i] != 0 { hostIndex.append(Int32(i)) }
                    }
                }
                for i in (words * 8)..<n where flags[i] != 0 { hostIndex.append(Int32(i)) }
                let m = hostIndex.count
                hostValues = [String](repeating: "", count: m)
                let lenPtr = lens.mutableTyped(Int32.self)
                let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
                hostIndex.withUnsafeBufferPointer { idx in
                    hostValues.withUnsafeMutableBufferPointer { buf in
                        let chunk = 4096
                        let chunks = (m + chunk - 1) / chunk
                        func work(_ c: Int) {
                            let lo = c * chunk, hi = Swift.min(lo + chunk, m)
                            for j in lo..<hi {
                                let i = Int(idx[j])
                                let s = String(decoding: UnsafeBufferPointer(start: d + Int(o[i]),
                                                                             count: Int(o[i + 1] - o[i])),
                                               as: UTF8.self)
                                let v = host(s)
                                buf[j] = v
                                lenPtr[i] = Int32(v.utf8.count)
                            }
                        }
                        if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
                    }
                }
            }
        }

        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try suPipeline("su_tf_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(hostRows.mtl, offset: hostRows.offset, index: 6)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 7)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 8)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        if anyHost {
            // The GPU owns disjoint byte ranges; sync so the two writers never overlap in time.
            try ctx.syncPoint()
            let outOff = outOffsets.typed(Int32.self)
            let dst = outData.mutableTyped(UInt8.self)
            let m = hostIndex.count
            hostIndex.withUnsafeBufferPointer { idx in
                hostValues.withUnsafeBufferPointer { buf in
                    let chunk = 4096
                    let chunks = (m + chunk - 1) / chunk
                    func work(_ c: Int) {
                        let lo = c * chunk, hi = Swift.min(lo + chunk, m)
                        for j in lo..<hi {
                            var p = Int(outOff[Int(idx[j])])
                            for b in buf[j].utf8 { dst[p] = b; p += 1 }
                        }
                    }
                    if chunks == 1 { work(0) } else { DispatchQueue.concurrentPerform(iterations: chunks, execute: work) }
                }
            }
        }
        for b in [a1, scratch, lens, hostRows, declined] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self)
        let out = MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                   offsets: outOffsets, data: outData, context: ctx)
        out.isBinary = isBinary
        return out
    }

    /// Arrow `utf8_upper`: Unicode's simple upper-case mapping, code point by code point.
    public func unicodeUpper() throws -> MetalStringArray {
        try unicodeTransform(.upper) { UnicodeClass.map(.upper, $0) }
    }
    /// Arrow `utf8_lower`.
    public func unicodeLower() throws -> MetalStringArray {
        try unicodeTransform(.lower) { UnicodeClass.map(.lower, $0) }
    }
    /// Arrow `utf8_swapcase`.
    public func unicodeSwapcase() throws -> MetalStringArray {
        try unicodeTransform(.swapcase) { UnicodeClass.map(.swapcase, $0) }
    }
    /// Arrow `utf8_capitalize`: the first code point upper-cased, every later one lower-cased.
    public func unicodeCapitalize() throws -> MetalStringArray {
        try unicodeTransform(.capitalize) { UnicodeClass.map(.capitalize, $0) }
    }
    /// Arrow `utf8_title`: the first cased code point of every word upper-cased, the rest lower-cased,
    /// a word being a maximal run of cased code points.
    public func unicodeTitle() throws -> MetalStringArray {
        try unicodeTransform(.title) { UnicodeClass.map(.title, $0) }
    }

    /// The trim half of the split: `set` is the trim set, `asciiOnlySet` says whether the GPU's byte
    /// set is the whole story (an ASCII set) or only its ASCII part (so non-ASCII rows go to the host).
    func unicodeTrimRows(_ op: UnicodeTransform, set: Set<Unicode.Scalar>) throws -> MetalStringArray {
        let asciiBytes = set.filter { $0.value < 0x80 }.map { UInt8($0.value) }.sorted()
        let hasNonASCII = set.contains { $0.value >= 0x80 }
        let left = (op == .trim || op == .ltrim), right = (op == .trim || op == .rtrim)
        return try unicodeTransform(op, arg1: asciiBytes, hostOnlyIfNonASCII: hasNonASCII) { s in
            var scalars = Array(s.unicodeScalars)[...]
            if left { while let f = scalars.first, set.contains(f) { scalars = scalars.dropFirst() } }
            if right { while let l = scalars.last, set.contains(l) { scalars = scalars.dropLast() } }
            var v = String.UnicodeScalarView()
            for u in scalars { v.append(u) }
            return String(v)
        }
    }

    /// The trim half for Arrow's Unicode whitespace class, which is not a finite literal set: the GPU
    /// takes the rows that are all ASCII (where the class is the ten bytes in
    /// ``unicodeWhitespaceASCII``) and the host takes the rest.
    func unicodeTrimWhitespaceRows(_ op: UnicodeTransform) throws -> MetalStringArray {
        let left = (op == .trim || op == .ltrim), right = (op == .trim || op == .rtrim)
        return try unicodeTransform(op, arg1: Self.unicodeWhitespaceASCII, hostOnlyIfNonASCII: true) { s in
            var scalars = Array(s.unicodeScalars)[...]
            if left { while let f = scalars.first, UnicodeClass.isSpace(f) { scalars = scalars.dropFirst() } }
            if right { while let l = scalars.last, UnicodeClass.isSpace(l) { scalars = scalars.dropLast() } }
            var v = String.UnicodeScalarView()
            for u in scalars { v.append(u) }
            return String(v)
        }
    }
}
