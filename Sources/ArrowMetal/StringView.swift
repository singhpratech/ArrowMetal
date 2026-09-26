import Foundation
import Metal
import CArrowABI

// Arrow `utf8_view` / `binary_view` ("vu" / "vz"), the string layout Polars keeps its String columns
// in, read by the GPU string kernels without converting it to offsets + bytes first.
//
// ## The layout
//
// One 16-byte view per row. The first four bytes are the length. A string of 12 bytes or fewer is
// stored inline in the remaining twelve; a longer one keeps a 4-byte prefix there, then the index of
// a data buffer and the byte offset of the string inside it:
//
//     length <= 12:  | int32 length | 12 bytes, zero padded       |
//     length  > 12:  | int32 length | 4-byte prefix | int32 buffer | int32 offset |
//
// The data buffers are "variadic": there can be any number of them. Polars grows them up to 16 MB
// each, so a 50M-row column of 13-byte strings carries a few dozen.
//
// ## How a kernel reads it
//
// The kernels are written once, against an accessor that yields `(pointer, length)` for row `i`
// (`StringLayoutSource.accessors`): `StrOff` over offsets + bytes, `StrView` over views. A view
// kernel binds the views where an offsets kernel binds its offsets, and a small table where it binds
// its data: one `{GPU address, byte size}` entry per data buffer (`MTLBuffer.gpuAddress`), so a view
// pointing into any of the buffers is one indexed load away. The data buffers themselves are made
// resident with `useResources`. The accessor checks every out-of-line view against its buffer's size
// and reads a view that points outside it as the empty string, so a malformed view (or the
// unspecified view under a null slot) can never make a kernel read out of bounds.
//
// `StringLayoutSource.variants` turns one templated body into a kernel per layout combination, named
// `name` for offsets and `name_v` (`name_ov`, `name_vo`, `name_vv` for two string arguments) for
// the others; `MetalStringArray.kernelName(_:_:)` picks the one that matches the arrays at hand.
//
// ## What converts
//
// A kernel with no view form reads `MetalStringArray.offsets`, which converts the column once on the
// GPU (lengths, scan, byte copy) and keeps the result, so a column is converted at most once whatever
// runs on it afterwards. `StringViewStorage.conversions` counts those conversions process-wide, and
// `ARROWMETAL_TRACE_VIEW_CONVERSION=1` prints the call stack of each one to stderr.

/// The views and data buffers of a `utf8_view` / `binary_view` column.
public final class StringViewStorage: @unchecked Sendable {
    /// 16 bytes per row, row 0 first (the producer's Arrow offset is already applied).
    public let views: MetalArrowBuffer
    /// The variadic data buffers, in the producer's order (a view's buffer index points into this).
    public let dataBuffers: [MetalArrowBuffer?]
    /// Byte size of each data buffer, as the producer declared it.
    public let dataSizes: [Int]
    /// `{0, count}` followed by one `{GPU address, byte size}` entry per data buffer.
    let table: MetalArrowBuffer
    /// Whether every buffer was mapped without a copy.
    public let zeroCopy: Bool
    /// Bytes copied on import (0 when `zeroCopy`).
    public let copiedBytes: Int
    /// Total bytes of the non-null strings: the size of the data buffer a conversion would write.
    public let logicalBytes: Int

    private let resources: [MTLResource]

    init(views: MetalArrowBuffer, dataBuffers: [MetalArrowBuffer?], dataSizes: [Int], zeroCopy: Bool,
         copiedBytes: Int, logicalBytes: Int, context: MetalContext) throws {
        self.views = views
        self.dataBuffers = dataBuffers
        self.dataSizes = dataSizes
        self.zeroCopy = zeroCopy
        self.copiedBytes = copiedBytes
        self.logicalBytes = logicalBytes
        let t = try MetalArrowBuffer.allocate(byteCount: (dataBuffers.count + 1) * 16, zeroed: true, context: context)
        let p = t.mutableTyped(UInt64.self)
        p[0] = 0; p[1] = UInt64(dataBuffers.count)
        for (k, b) in dataBuffers.enumerated() {
            if let b {
                p[2 + 2 * k] = b.mtl.gpuAddress &+ UInt64(b.offset)
                p[3 + 2 * k] = UInt64(dataSizes[k])
            }
        }
        table = t
        resources = dataBuffers.compactMap { $0?.mtl }
    }

    /// The same data buffers under a different window of views (a slice).
    private init(slicing s: StringViewStorage, views: MetalArrowBuffer) {
        self.views = views
        dataBuffers = s.dataBuffers; dataSizes = s.dataSizes; table = s.table
        zeroCopy = s.zeroCopy; copiedBytes = s.copiedBytes; logicalBytes = s.logicalBytes
        resources = s.resources
    }

    /// Rows `[offset, offset + length)`: a view of the views buffer, the data buffers shared.
    func sliced(offset: Int, length: Int) -> StringViewStorage {
        StringViewStorage(slicing: self, views: views.view(byteOffset: offset * 16, byteCount: length * 16))
    }

    /// Binds views + table at `index`, `index + 1` and makes the data buffers resident.
    func bind(_ enc: MTLComputeCommandEncoder, at index: Int) {
        enc.setBuffer(views.mtl, offset: views.offset, index: index)
        enc.setBuffer(table.mtl, offset: table.offset, index: index + 1)
        if !resources.isEmpty { enc.useResources(resources, usage: .read) }
    }

    /// Row `i`'s bytes on the host, with the same bounds rule as the GPU accessor.
    func bytes(row i: Int) -> UnsafeBufferPointer<UInt8> {
        let v = views.contents.advanced(by: i * 16)
        let len = Int(v.loadUnaligned(as: Int32.self))
        if len <= 12 {
            return UnsafeBufferPointer(start: v.advanced(by: 4).assumingMemoryBound(to: UInt8.self), count: max(len, 0))
        }
        let k = Int(v.loadUnaligned(fromByteOffset: 8, as: Int32.self))
        let off = Int(v.loadUnaligned(fromByteOffset: 12, as: Int32.self))
        guard k >= 0, k < dataBuffers.count, let b = dataBuffers[k], off >= 0, off + len <= dataSizes[k] else {
            return UnsafeBufferPointer(start: nil, count: 0)
        }
        return UnsafeBufferPointer(start: b.contents.advanced(by: off).assumingMemoryBound(to: UInt8.self), count: len)
    }

    // MARK: Conversion counter

    private static let counterLock = NSLock()
    nonisolated(unsafe) private static var conversionCount = 0
    nonisolated(unsafe) private static var convertedRows = 0
    private static let trace = ProcessInfo.processInfo.environment["ARROWMETAL_TRACE_VIEW_CONVERSION"] == "1"

    /// `(columns, rows)` converted from views to offsets + bytes since the process started.
    public static var conversions: (columns: Int, rows: Int) {
        counterLock.lock(); defer { counterLock.unlock() }
        return (conversionCount, convertedRows)
    }

    static func noteConversion(rows: Int) {
        counterLock.lock()
        conversionCount += 1; convertedRows += rows
        counterLock.unlock()
        if trace {
            let stack = Thread.callStackSymbols.dropFirst(2).prefix(12).joined(separator: "\n")
            FileHandle.standardError.write("ArrowMetal: utf8_view column of \(rows) rows converted to offsets + bytes\n\(stack)\n".data(using: .utf8)!)
        }
    }

    // MARK: Conversion to offsets + bytes

    /// Runs `body` in a command buffer of its own, outside any open batch: the conversion can be
    /// triggered from inside another kernel's encoding, where the batch encoder is half-bound.
    static func runStandalone(_ ctx: MetalContext, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw ArrowMetalError.noMetalDevice
        }
        try body(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw ArrowMetalError.pipelineCreationFailed("command buffer failed: \(err)") }
    }

    /// The column as int32 offsets + bytes: lengths, scan and byte copy on the GPU. A null row gets
    /// length 0.
    func convertToOffsets(length n: Int, validity: MetalArrowBuffer?,
                          context ctx: MetalContext) throws -> (MetalArrowBuffer, MetalArrowBuffer) {
        guard logicalBytes < Int(Int32.max) else {
            throw ArrowMetalError.unsupportedType("utf8_view over 2 GB cannot be converted to utf8")
        }
        let offsets = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: n == 0, context: ctx)
        guard n > 0 else {
            return (offsets, try MetalArrowBuffer.allocate(byteCount: 1, context: ctx))
        }
        let src = StringLayoutSource.viewSource
        let pLen = try ctx.pipeline(source: src, function: "sv_lengths", cacheKey: "strview/sv_lengths")
        let p1 = try ctx.pipeline(source: StringSource.source, function: "scan_block", cacheKey: "str/scan_block")
        let p2 = try ctx.pipeline(source: StringSource.source, function: "scan_totals", cacheKey: "str/scan_totals")
        let p3 = try ctx.pipeline(source: StringSource.source, function: "scan_add", cacheKey: "str/scan_add")
        let lens = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let blocks = Swift.max(1, (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let totals = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let grand = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        let vb = validity ?? views
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        try Self.runStandalone(ctx) { enc in
            enc.setComputePipelineState(pLen)
            bind(enc, at: 0)
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            enc.setBuffer(lens.mtl, offset: lens.offset, index: 5)
            Dispatch.dispatch1D(enc, pLen, count: n)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(p1)
            enc.setBuffer(lens.mtl, offset: lens.offset, index: 0)
            Dispatch.setLength(enc, n, nil, index: 1)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 2)
            enc.setBuffer(totals.mtl, offset: 0, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(p2)
            enc.setBuffer(totals.mtl, offset: 0, index: 0)
            Dispatch.setUInt(enc, blocks, index: 1)
            enc.setBuffer(grand.mtl, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(p3)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 0)
            enc.setBuffer(totals.mtl, offset: 0, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(grand.mtl, offset: 0, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: blocks + 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
        }
        let total = Int(offsets.typed(Int32.self)[n])
        let data = try MetalArrowBuffer.allocate(byteCount: Swift.max(total, 1), zeroed: false, context: ctx)
        if total > 0 {
            let pWrite = try ctx.pipeline(source: src, function: "sv_write", cacheKey: "strview/sv_write")
            try Self.runStandalone(ctx) { enc in
                enc.setComputePipelineState(pWrite)
                bind(enc, at: 0)
                enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 2)
                Dispatch.setUInt(enc, n, index: 3)
                enc.setBuffer(data.mtl, offset: data.offset, index: 4)
                Dispatch.dispatch1D(enc, pWrite, count: n)
            }
        }
        return (offsets, data)
    }

    /// Total bytes of the non-null rows (64-bit), one GPU pass over the views.
    static func logicalBytes(views: MetalArrowBuffer, table: MetalArrowBuffer, resources: [MTLResource],
                             length n: Int, validity: MetalArrowBuffer?, context ctx: MetalContext) throws -> Int {
        guard n > 0 else { return 0 }
        let p = try ctx.pipeline(source: StringLayoutSource.viewSource, function: "sv_total",
                                 cacheKey: "strview/sv_total")
        let groups = Swift.min(1024, (n + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let partials = try MetalArrowBuffer.allocate(byteCount: groups * 8, zeroed: true, context: ctx)
        let vb = validity ?? views
        try Self.runStandalone(ctx) { enc in
            enc.setComputePipelineState(p)
            enc.setBuffer(views.mtl, offset: views.offset, index: 0)
            enc.setBuffer(table.mtl, offset: table.offset, index: 1)
            if !resources.isEmpty { enc.useResources(resources, usage: .read) }
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            enc.setBuffer(partials.mtl, offset: partials.offset, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
        }
        let pp = partials.typed(UInt64.self)
        var total: UInt64 = 0
        for g in 0..<groups { total &+= pp[g] }
        return Int(total)
    }
}

// MARK: - The MSL accessors and the per-layout kernel variants

enum StringLayoutSource {
    /// The two row accessors. Include once per library, after `KernelSource.prelude`.
    ///
    /// `row(i, len)` returns a pointer to row `i`'s first byte and sets `len`; kernels written against
    /// it take the accessor as a template parameter and are instantiated per layout by `variants`.
    static let accessors = """

    // ---- string layouts: offsets + bytes, or 16-byte views (StringView.swift) ----
    struct SVBuf { device const uchar* p; ulong n; };
    struct StrOff {
        device const int* o; device const uchar* d;
        inline device const uchar* row(uint i, thread int& len) const { int s = o[i]; len = o[i + 1] - s; return d + s; }
        inline int len(uint i) const { return o[i + 1] - o[i]; }
    };
    struct StrView {
        device const uint4* v; device const SVBuf* b;
        inline device const uchar* row(uint i, thread int& len) const {
            uint4 x = v[i];
            if (x.x <= 12u) { len = (int)x.x; return ((device const uchar*)(v + i)) + 4; }
            if (x.z < (uint)b[0].n) {
                SVBuf e = b[1u + x.z];
                if ((ulong)x.w + (ulong)x.x <= e.n) { len = (int)x.x; return e.p + x.w; }
            }
            len = 0; return (device const uchar*)(v + i);
        }
        inline int len(uint i) const { int l; row(i, l); return l; }
    };

    """

    /// Wrapper kernels for one templated body, one per layout combination of its string arguments.
    ///
    /// `slots` are the buffer indices of the string arguments (each takes that index and the next);
    /// `params` is the rest of the kernel's parameter list, attributes included; `call` is the body's
    /// invocation with `S0`, `S1` standing for the accessors.
    static func variants(_ name: String, slots: [Int], params: String, call: String) -> String {
        var out = ""
        let combos = 1 << slots.count
        for mask in 0..<combos {
            var code = "", args: [String] = [], invocation = call
            for (k, slot) in slots.enumerated() {
                let isView = (mask >> k) & 1 == 1
                code += isView ? "v" : "o"
                if isView {
                    args.append("device const uint4* s\(k)a [[buffer(\(slot))]], device const SVBuf* s\(k)b [[buffer(\(slot + 1))]]")
                    invocation = invocation.replacingOccurrences(of: "S\(k)", with: "StrView{s\(k)a, s\(k)b}")
                } else {
                    args.append("device const int* s\(k)a [[buffer(\(slot))]], device const uchar* s\(k)b [[buffer(\(slot + 1))]]")
                    invocation = invocation.replacingOccurrences(of: "S\(k)", with: "StrOff{s\(k)a, s\(k)b}")
                }
            }
            let fn = code.allSatisfy({ $0 == "o" }) ? name : "\(name)_\(code)"
            out += "kernel void \(fn)(\(args.joined(separator: ", ")), \(params)) { \(invocation); }\n"
        }
        return out
    }

    /// The conversion and import kernels.
    static let viewSource: String = KernelSource.prelude + accessors + """

    // Byte length of every row, 0 for a null row (the conversion's first pass).
    kernel void sv_lengths(device const uint4* v [[buffer(0)]], device const SVBuf* b [[buffer(1)]],
                           device const uchar* validity [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                           constant uint& n [[buffer(4)]], device int* out [[buffer(5)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        if (hasValidity != 0u && !bit_get(validity, i)) { out[i] = 0; return; }
        StrView s{v, b};
        out[i] = s.len(i);
    }
    // Copies every row's bytes to its place in the offsets layout (the conversion's last pass).
    kernel void sv_write(device const uint4* v [[buffer(0)]], device const SVBuf* b [[buffer(1)]],
                         device const int* offsets [[buffer(2)]], constant uint& n [[buffer(3)]],
                         device uchar* out [[buffer(4)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        StrView s{v, b};
        int len; device const uchar* p = s.row(i, len);
        int to = offsets[i], w = offsets[i + 1] - to;      // 0 for a null row
        for (int k = 0; k < w; k++) out[to + k] = p[k];
    }
    // 64-bit total of the non-null rows' lengths: one partial per threadgroup, summed on the host.
    kernel void sv_total(device const uint4* v [[buffer(0)]], device const SVBuf* b [[buffer(1)]],
                         device const uchar* validity [[buffer(2)]], constant uint& hasValidity [[buffer(3)]],
                         constant uint& n [[buffer(4)]], device ulong* partials [[buffer(5)]],
                         uint gid [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                         uint tgid [[threadgroup_position_in_grid]], uint grid [[threads_per_grid]]) {
        // simd_sum has no 64-bit form, so each thread's total goes through threadgroup memory.
        threadgroup ulong sums[TG];
        StrView s{v, b};
        ulong acc = 0ul;
        for (uint i = gid; i < n; i += grid) {
            if (hasValidity != 0u && !bit_get(validity, i)) continue;
            acc += (ulong)s.len(i);
        }
        sums[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            ulong t = 0ul;
            for (uint k = 0; k < TG; k++) t += sums[k];
            partials[tgid] = t;
        }
    }
    """
}

// MARK: - Binding from Swift

extension MetalStringArray {
    /// The kernel variant for these arrays' layouts: `base` when every one is offsets + bytes, else
    /// `base_<code>` with one `o`/`v` per array (`StringLayoutSource.variants`).
    static func kernelName(_ base: String, _ arrays: MetalStringArray...) -> String {
        guard arrays.contains(where: { $0.view != nil }) else { return base }
        return base + "_" + String(arrays.map { $0.view == nil ? "o" : "v" })
    }

    /// Binds this column's layout at `index` and `index + 1`: offsets and data, or views and the data
    /// buffer table (plus the data buffers' residency). Never converts.
    func bindLayout(_ enc: MTLComputeCommandEncoder, at index: Int) {
        if let v = view { v.bind(enc, at: index); return }
        let o = offsets, d = data
        enc.setBuffer(o.mtl, offset: o.offset, index: index)
        enc.setBuffer(d.mtl, offset: d.offset, index: index + 1)
    }

    /// `"view"` for a column held as views, `"view (converted)"` once a kernel without a view form has
    /// converted it, `"offsets"` otherwise.
    public var layoutDescription: String {
        guard view != nil else { return "offsets" }
        return convertedFromView ? "view (converted)" : "view"
    }
}

// MARK: - Import and export

/// `utf8_view` ("vu") / `binary_view` ("vz") import. Buffers: validity, views, the data buffers, then
/// the int64 sizes of the data buffers. Each buffer is mapped without a copy when it is page aligned
/// and copied once otherwise.
func importStringViewArray(binary: Bool, array: UnsafeMutablePointer<ArrowArray>,
                           context: MetalContext) throws -> ImportResult {
    let nb = Int(array.pointee.n_buffers)
    guard nb >= 3, array.pointee.buffers != nil else {
        throw ArrowMetalError.invalidArrowArray("expected at least 3 buffers for a \(binary ? "binary" : "utf8")_view array")
    }
    let owner = ImportedCArray(moving: array)
    let a = owner.array
    let length = Int(a.length), offset = Int(a.offset)
    let k = nb - 3
    let sizesPtr = a.buffers[nb - 1].map { UnsafeRawPointer($0).assumingMemoryBound(to: Int64.self) }
    guard k == 0 || sizesPtr != nil else { throw ArrowMetalError.invalidArrowArray("utf8_view: buffer sizes are null") }

    var zc = true, copied = 0
    func wrap(_ p: UnsafeRawPointer, _ bytes: Int) throws -> MetalArrowBuffer {
        let page = metalPageSize()
        if bytes > 0, UInt(bitPattern: p) % UInt(page) == 0 {
            let (b, z) = try MetalArrowBuffer.wrapOrCopy(p, byteCount: bytes, keepAlive: owner, context: context)
            if !z { zc = false; copied += bytes }
            return b
        }
        zc = false; copied += bytes
        let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: context)
        if bytes > 0 { memcpy(b.mutableContents, p, bytes) }
        return b
    }

    // Validity: the bit offset is applied (a fresh bitmap when it is not 0), as the utf8 import does.
    var validity: MetalArrowBuffer? = nil
    if let vp = a.buffers[0].map({ UnsafeRawPointer($0) }), length > 0 {
        if offset == 0 {
            validity = try wrap(vp, Bitmap.byteCount(bits: length))
        } else {
            zc = false
            let bytes = Bitmap.byteCount(bits: length)
            let b = try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), context: context)
            Bitmap.copyBits(vp.assumingMemoryBound(to: UInt8.self), from: offset, into: b.mutableTyped(UInt8.self), bits: length)
            copied += bytes
            validity = b
        }
    }
    // Views: wrapped from the producer's first view, the Arrow offset applied as a buffer window.
    let views: MetalArrowBuffer
    if length > 0 {
        guard let vp = a.buffers[1].map({ UnsafeRawPointer($0) }) else { throw ArrowMetalError.invalidArrowArray("utf8_view: views buffer is null") }
        let whole = try wrap(vp, (offset + length) * 16)
        views = offset == 0 ? whole : whole.view(byteOffset: offset * 16, byteCount: length * 16)
    } else {
        views = try MetalArrowBuffer.allocate(byteCount: 16, context: context)
    }
    var data: [MetalArrowBuffer?] = []
    var sizes: [Int] = []
    for j in 0..<k {
        let size = Int(sizesPtr![j])
        guard size >= 0 else { throw ArrowMetalError.invalidArrowArray("utf8_view: negative data buffer size") }
        sizes.append(size)
        if size > 0, let dp = a.buffers[2 + j].map({ UnsafeRawPointer($0) }) {
            data.append(try wrap(dp, size))
        } else {
            data.append(nil)
        }
    }
    let resources: [MTLResource] = data.compactMap { $0?.mtl }
    // A provisional storage to size the column: the exact byte total decides whether a conversion to
    // int32 offsets is possible, and is the size that conversion would allocate.
    let probe = try StringViewStorage(views: views, dataBuffers: data, dataSizes: sizes, zeroCopy: zc,
                                      copiedBytes: copied, logicalBytes: 0, context: context)
    let total = try StringViewStorage.logicalBytes(views: views, table: probe.table, resources: resources,
                                                   length: length, validity: validity, context: context)
    // The same limit the large_utf8 import has: every kernel without a view form reads int32 offsets.
    guard total < Int(Int32.max) else {
        throw ArrowMetalError.unsupportedType("\(binary ? "binary" : "utf8")_view over 2 GB")
    }
    let storage = try StringViewStorage(views: views, dataBuffers: data, dataSizes: sizes, zeroCopy: zc,
                                        copiedBytes: copied, logicalBytes: total, context: context)
    let arr = MetalStringArray(length: length, nullCount: 0, validity: validity, view: storage, context: context)
    arr.isBinary = binary
    let nc = Int(a.null_count)
    if nc < 0 || (validity == nil && nc != 0) { arr.recomputeNullCount() } else { arr.setNullCount(validity == nil ? 0 : nc) }
    return ImportResult(array: binary ? .binary(arr) : .string(arr), zeroCopy: zc)
}

extension MetalStringArray {
    /// Exports a view column as `utf8_view` / `binary_view` with its own buffers: validity, views, the
    /// data buffers and their int64 sizes. Nothing is copied.
    func exportViewArray(_ v: StringViewStorage, into out: UnsafeMutablePointer<ArrowArray>) {
        let sizes = Int64Block(v.dataSizes.map { Int64($0) })
        var keeps: [AnyObject] = [self, v, v.views, sizes]
        if let vb = validity { keeps.append(vb) }
        var ptrs: [UnsafeRawPointer?] = [validity?.contents, v.views.contents]
        for b in v.dataBuffers {
            if let b { keeps.append(b); ptrs.append(b.contents) } else { ptrs.append(nil) }
        }
        ptrs.append(UnsafeRawPointer(sizes.pointer))
        fillExportedStringArray(length: length, nullCount: nullCount, keeps: keeps, bufferPtrs: ptrs, into: out)
    }
}

/// A malloc'd int64 array that lives as long as the export holding it (the view sizes buffer).
final class Int64Block {
    let pointer: UnsafeMutablePointer<Int64>
    init(_ values: [Int64]) {
        pointer = .allocate(capacity: Swift.max(values.count, 1))
        for (i, v) in values.enumerated() { pointer[i] = v }
    }
    deinit { pointer.deallocate() }
}
