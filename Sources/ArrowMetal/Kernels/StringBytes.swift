import Foundation
import Metal

/// The **byte-indexed** half of Arrow's string surface, and the N-column `binary_join_element_wise`.
///
/// Arrow spells the same shape twice: `utf8_*` counts code points, `ascii_*` and `binary_*` count
/// bytes. `Kernels/StringTransforms.swift` supplies the code point forms; this file supplies the byte
/// forms — `binary_slice`, `binary_reverse` / `ascii_reverse`, `ascii_lpad` / `ascii_rpad` /
/// `ascii_center` — plus the slicing routine that understands Arrow's `step`.
///
/// Everything here is GPU, two-pass (`Kernels/StringBytesSource.swift`): a length kernel, a GPU scan
/// into the Arrow offsets buffer, a byte kernel.
public enum ByteTransform: Int, Sendable, CaseIterable {
    case sliceBytes = 0
    case sliceCodepoints = 1
    case reverseBytes = 2
    case padLeftBytes = 3
    case padRightBytes = 4
    case centerBytes = 5
}

/// Arrow's `null_handling` for `binary_join_element_wise`.
public enum JoinNullHandling: Int, Sendable, CaseIterable {
    /// A null in any column makes the whole output row null (Arrow's default).
    case emitNull = 0
    /// A null column contributes nothing — not even its separator.
    case skip = 1
    /// A null column contributes `nullReplacement` instead.
    case replace = 2

    public init?(name: String) {
        switch name.lowercased() {
        case "emit_null", "emitnull": self = .emitNull
        case "skip": self = .skip
        case "replace": self = .replace
        default: return nil
        }
    }
}

extension MetalStringArray {

    // MARK: - Plumbing

    private func sbPipeline(_ fn: String) throws -> MTLComputePipelineState {
        try context.pipeline(source: StringBytesSource.source, function: fn, cacheKey: "strb/\(fn)")
    }

    /// 24 bytes matching MSL `struct SbParams { uint op; uint n1; int p1; int p2; int p3; uint flags; }`.
    static func sbParams(_ op: Int, _ n1: Int, _ p1: Int, _ p2: Int, _ p3: Int, _ flags: Int) -> [UInt32] {
        [UInt32(op), UInt32(n1),
         UInt32(bitPattern: Int32(clamping: p1)), UInt32(bitPattern: Int32(clamping: p2)),
         UInt32(bitPattern: Int32(clamping: p3)), UInt32(flags)]
    }

    /// The two-pass driver for `StringBytesSource`: length kernel, GPU scan to offsets, byte kernel.
    /// The result carries this array's `isBinary` flag, so a `binary` column stays `binary`.
    func byteTransform(_ op: ByteTransform, arg1: [UInt8] = [],
                       p1: Int = 0, p2: Int = 0, p3: Int = 0) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let a1 = try sxArgBuffer(arg1)
        let vb = validity ?? a1                                  // never read when flags bit 0 is clear
        let scratch = try MetalArrowBuffer.allocate(byteCount: 1, zeroed: false, context: ctx)
        var prm = Self.sbParams(op.rawValue, arg1.count, p1, p2, p3, validity == nil ? 0 : 1)

        if n > 0 {
            let pLen = try sbPipeline("sb_tf_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                enc.setBuffer(scratch.mtl, offset: scratch.offset, index: 7)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try sbPipeline("sb_tf_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                enc.setBytes(&prm, length: 24, index: 4)
                enc.setBuffer(a1.mtl, offset: a1.offset, index: 5)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 6)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 7)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [a1, scratch, lens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self)
        let out = MetalStringArray(length: n, nullCount: nullCount, validity: validity,
                                   offsets: outOffsets, data: outData, context: ctx)
        out.isBinary = isBinary
        return out
    }

    // MARK: - Slicing

    /// Arrow `binary_slice`: `value[start:stop:step]` indexed in **bytes**, with Python's slice rules.
    ///
    /// A negative index counts from the end, both ends clamp, and a negative `step` walks backwards.
    /// `stop == nil` means "to the end" for a positive step and "to the beginning" for a negative one.
    /// The cut is byte-exact, so slicing a `utf8` value can split a UTF-8 sequence — which is why
    /// Arrow gives this one a `binary` result and refuses a `string` input. Always GPU.
    public func binarySlice(start: Int, stop: Int? = nil, step: Int = 1) throws -> MetalStringArray {
        guard step != 0 else { throw ArrowMetalError.invalidArrowArray("binary_slice step cannot be zero") }
        let out = try byteTransform(.sliceBytes, p1: start, p2: Self.sliceStop(stop, step: step), p3: step)
        out.isBinary = true
        return out
    }

    /// The stop bound a missing `stop` stands for: past the end going forwards, before the start
    /// going backwards. Written as the extremes of Int32 because that is what the kernel clamps.
    static func sliceStop(_ stop: Int?, step: Int) -> Int {
        if let stop { return stop }
        return step > 0 ? Int(Int32.max) : Int(Int32.min)
    }

    /// Arrow `binary_reverse`: reverses the **bytes** of every value, which is what Arrow does for a
    /// `binary` column. On non-ASCII `utf8` content this produces invalid UTF-8, exactly as Arrow's
    /// does, so the result is marked `binary`. Always GPU.
    public func binaryReverse() throws -> MetalStringArray {
        let out = try byteTransform(.reverseBytes)
        out.isBinary = true
        return out
    }

    /// Arrow `ascii_reverse`: the same byte-wise reversal, refused on non-ASCII input.
    ///
    /// pyarrow raises `Non-ASCII sequence in input` rather than mangling a UTF-8 sequence, and so does
    /// this. On ASCII input a byte reversal and a code point reversal agree, so the answer equals
    /// ``reverse()``; the result stays `utf8`.
    public func asciiReverse() throws -> MetalStringArray {
        guard try isAllASCII() else {
            throw ArrowMetalError.unsupportedType("ascii_reverse: non-ASCII sequence in input")
        }
        let out = try byteTransform(.reverseBytes)
        out.isBinary = isBinary
        return out
    }

    // MARK: - Byte-counted padding

    /// Arrow `ascii_lpad`: left-pads with `pad` until the value is `width` **bytes** wide.
    ///
    /// This is the distinction Arrow draws between `ascii_lpad` and `utf8_lpad`: the ASCII form counts
    /// bytes, so `"héllo"` (6 bytes) padded to 8 gains two characters where ``padLeft(width:pad:)``
    /// would give it three. `pad` must be a single **byte**.
    public func asciiLpad(width: Int, pad: String = " ") throws -> MetalStringArray {
        try byteTransform(.padLeftBytes, arg1: try Self.asciiPadByte(pad), p1: Swift.max(width, 0))
    }
    /// Arrow `ascii_rpad`: the mirror of ``asciiLpad(width:pad:)``.
    public func asciiRpad(width: Int, pad: String = " ") throws -> MetalStringArray {
        try byteTransform(.padRightBytes, arg1: try Self.asciiPadByte(pad), p1: Swift.max(width, 0))
    }
    /// Arrow `ascii_center`: pads on both sides to `width` **bytes**, the odd pad byte on the right.
    public func asciiCenter(width: Int, pad: String = " ") throws -> MetalStringArray {
        try byteTransform(.centerBytes, arg1: try Self.asciiPadByte(pad), p1: Swift.max(width, 0))
    }

    static func asciiPadByte(_ s: String) throws -> [UInt8] {
        let b = Array(s.utf8)
        guard b.count == 1 else {
            throw ArrowMetalError.invalidArrowArray("ascii padding must be exactly one byte, got \(s.debugDescription)")
        }
        return b
    }

    // MARK: - binary_join_element_wise

    /// Arrow `binary_join_element_wise`: `columns[0] + sep + columns[1] + sep + …`, N columns wide.
    ///
    /// `nullHandling` is Arrow's option:
    ///
    /// * `.emitNull` — a null anywhere makes the whole row null (Arrow's default).
    /// * `.skip` — a null column contributes nothing, not even its separator.
    /// * `.replace` — a null column contributes `nullReplacement`.
    ///
    /// The work is a left fold of one GPU two-pass join step (`sb_cat_len` / `sb_cat_write`) over the
    /// columns, so N columns cost N-1 passes and never leave the device. Under `.skip` the accumulator
    /// stays **null** until something has actually been joined, which is what keeps a row whose first
    /// columns are all null from picking up leading separators.
    ///
    /// **Difference from pyarrow (25.0.1):** with `.skip`, a row whose columns are *all* null joins to
    /// the empty string here. pyarrow drops that row from the output entirely — its result is shorter
    /// than its input — which is a bug in Arrow's offset bookkeeping, not a semantic this reproduces.
    public static func joinElementWise(_ columns: [MetalStringArray], separator: String = "",
                                       nullHandling: JoinNullHandling = .emitNull,
                                       nullReplacement: String = "") throws -> MetalStringArray {
        guard let first = columns.first else {
            throw ArrowMetalError.invalidArrowArray("binary_join_element_wise needs at least one column")
        }
        for c in columns.dropFirst() where c.length != first.length {
            throw ArrowMetalError.lengthMismatch(first.length, c.length)
        }
        if columns.count == 1 {
            switch nullHandling {
            case .emitNull: return first
            case .skip: return try first.fillNullStrings(with: "")
            case .replace: return try first.fillNullStrings(with: nullReplacement)
            }
        }
        var acc = first
        for next in columns.dropFirst() {
            acc = try acc.joinStep(next, separator: separator, nullHandling: nullHandling,
                                   nullReplacement: nullReplacement)
        }
        if nullHandling == .skip, acc.nullCount > 0 {
            // Every column of the row was null: nothing was joined, so the row is the empty string.
            acc = try acc.fillNullStrings(with: "")
        }
        return acc
    }

    /// Replaces null rows with `value`, dropping the validity bitmap. Host-side; used only to close
    /// out a fold, so it touches at most one column.
    func fillNullStrings(with value: String) throws -> MetalStringArray {
        guard nullCount > 0 else { return self }
        try context.syncPoint()
        var rows = [String?](repeating: nil, count: length)
        rows.withUnsafeMutableBufferPointer { buf in
            forEachRowConcurrently { i, s in buf[i] = s ?? value }
        }
        let out = try MetalStringArray(rows, context: context)
        out.isBinary = isBinary
        return out
    }

    /// One step of the fold: `self + separator + other`, under one of Arrow's null-handling rules.
    func joinStep(_ other: MetalStringArray, separator: String, nullHandling: JoinNullHandling,
                  nullReplacement: String) throws -> MetalStringArray {
        let ctx = context, n = length
        try Dispatch.checkLength(n)
        let sepBytes = Array(separator.utf8), replBytes = Array(nullReplacement.utf8)
        let sep = try sxArgBuffer(sepBytes), repl = try sxArgBuffer(replBytes)
        let lens = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 4, 4), zeroed: true, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), context: ctx)
        let va = validity ?? sep, vb = other.validity ?? sep
        let flags = (validity == nil ? 0 : 1) | (other.validity == nil ? 0 : 2)
        var prm = Self.sbParams(0, sepBytes.count, nullHandling.rawValue, replBytes.count, 0, flags)

        if n > 0 {
            let pLen = try sbPipeline("sb_cat_len")
            try ctx.run { enc in
                enc.setComputePipelineState(pLen)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(other.offsets.mtl, offset: other.offsets.offset, index: 1)
                enc.setBuffer(va.mtl, offset: va.offset, index: 2)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
                Dispatch.setLength(enc, n, nil, index: 4)
                enc.setBytes(&prm, length: 24, index: 5)
                enc.setBuffer(lens.mtl, offset: lens.offset, index: 6)
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 7)
                Dispatch.dispatch1D(enc, pLen, count: n)
            }
        }
        let outOffsets = try MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: lens, context: ctx)
            .exclusiveScanToOffsets()
        let total = Int(withExtendedLifetime(outOffsets) { outOffsets.typed(Int32.self)[n] })
        let outData = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if n > 0 {
            let pWrite = try sbPipeline("sb_cat_write")
            try ctx.run { enc in
                enc.setComputePipelineState(pWrite)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
                enc.setBuffer(data.mtl, offset: data.offset, index: 1)
                enc.setBuffer(other.offsets.mtl, offset: other.offsets.offset, index: 2)
                enc.setBuffer(other.data.mtl, offset: other.data.offset, index: 3)
                enc.setBuffer(va.mtl, offset: va.offset, index: 4)
                enc.setBuffer(vb.mtl, offset: vb.offset, index: 5)
                Dispatch.setLength(enc, n, nil, index: 6)
                enc.setBytes(&prm, length: 24, index: 7)
                enc.setBuffer(sep.mtl, offset: sep.offset, index: 8)
                enc.setBuffer(repl.mtl, offset: repl.offset, index: 9)
                enc.setBuffer(outOffsets.mtl, offset: outOffsets.offset, index: 10)
                enc.setBuffer(outData.mtl, offset: outData.offset, index: 11)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        for b in [sep, repl, lens] { ctx.retainUntilFlush(b) }
        ctx.retainUntilFlush(self); ctx.retainUntilFlush(other)
        let outValidity = n > 0 ? try BitmapOps.packBits(ctx, bytes: validBytes, bits: n) : nil
        ctx.retainUntilFlush(validBytes)
        var nulls = 0
        if let v = outValidity {
            try ctx.syncPoint()
            nulls = n - Bitmap.popcount(v.typed(UInt8.self), bits: n)
        }
        let out = MetalStringArray(length: n, nullCount: nulls,
                                   validity: nulls == 0 ? nil : outValidity,
                                   offsets: outOffsets, data: outData, context: ctx)
        out.isBinary = isBinary || other.isBinary
        return out
    }
}
