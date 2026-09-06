import Foundation
import Metal
import CArrowABI

// Arrow run-end encoded arrays ("+r"). The layout is two children and no buffers of its own:
// `run_ends` (int16 / int32 / int64, strictly increasing, the *exclusive* end of each run) and `values`
// (one element per run, nullable). The logical length is the last run end.
//
// ArrowMetal narrows run ends to int32 on import, exactly as it narrows dictionary indices, so every
// kernel downstream stays 32-bit; arrays above 2^31 - 1 elements are out of range anyway
// (`Dispatch.checkLength`).
//
// `decode()` expands a run-end array on the GPU with one binary search per output element, and
// `runEndEncode()` builds one with the boundary-mark + scan pipeline `unique()` uses.

/// Logical length of a run-end encoded array: its last run end, or 0 when there are no runs.
func runEndLogicalLength(_ runEnds: MetalArray<Int32>) -> Int {
    let n = runEnds.length
    guard n > 0 else { return 0 }
    return withExtendedLifetime(runEnds) { Int(runEnds.valuePointer[n - 1]) }
}

/// Rows covered by null runs: the null count a decoded run-end array would have.
func runEndNullCount(_ runEnds: MetalArray<Int32>, _ values: AnyMetalArray) -> Int {
    guard values.nullCount > 0, let bitmap = values.validityBitmap else { return 0 }
    return withExtendedLifetime((runEnds, bitmap)) {
        let ends = runEnds.valuePointer
        let bits = bitmap.typed(UInt8.self)
        var nulls = 0, previous = 0
        for j in 0..<Swift.min(runEnds.length, values.length) {
            let end = Int(ends[j])
            if !Bitmap.isSet(bits, j) { nulls += end - previous }
            previous = end
        }
        return nulls
    }
}

extension AnyMetalArray {
    /// The validity bitmap behind whichever array this is, or nil when it has none.
    var validityBitmap: MetalArrowBuffer? {
        switch self {
        case .int8(let a): return a.validity
        case .uint8(let a): return a.validity
        case .int16(let a): return a.validity
        case .uint16(let a): return a.validity
        case .int32(let a): return a.validity
        case .uint32(let a): return a.validity
        case .int64(let a): return a.validity
        case .uint64(let a): return a.validity
        case .float32(let a): return a.validity
        case .float64(let a): return a.validity
        case .boolean(let a): return a.validity
        case .string(let a): return a.validity
        case .temporal(let a): return a.validity
        case .binary(let a): return a.validity
        case .dictionary(let codes, _): return codes.validity
        case .runEndEncoded(_, let values): return values.validityBitmap
        }
    }

    /// The (runEnds, values) pair of a run-end encoded array, or nil.
    public var asRunEndEncoded: (runEnds: MetalArray<Int32>, values: AnyMetalArray)? {
        if case .runEndEncoded(let r, let v) = self { return (r, v) }
        return nil
    }

    /// Number of runs in a run-end encoded array (its own length is the decoded, logical one).
    public var runCount: Int? {
        if case .runEndEncoded(let r, _) = self { return r.length }
        return nil
    }

    /// Expands a run-end encoded array on the GPU. Other arrays are returned as-is.
    ///
    /// One thread per output element binary-searches the run ends for the run that covers it, and the
    /// resulting run indices gather the values with the existing `take` kernel, so nulls in the values
    /// become nulls in the output.
    public func runEndDecode() throws -> AnyMetalArray {
        guard case .runEndEncoded(let runEnds, let values) = self else { return self }
        let ctx = runEnds.context
        let runs = runEnds.length
        let n = runEndLogicalLength(runEnds)
        try Dispatch.checkLength(n)
        guard n > 0 else { return try values.take(try MetalArray<Int32>([Int32](), context: ctx)) }
        let idx = try MetalArrowBuffer.allocate(byteCount: n * 4, zeroed: false, context: ctx)
        let pso = try Dispatch.pipeline(ctx, family: "ree", source: RunEndSource.source, function: "ree_expand", type: "int")
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(runEnds.values.mtl, offset: runEnds.values.offset, index: 0)
            Dispatch.setUInt(enc, runs, index: 1)
            Dispatch.setLength(enc, n, nil, index: 2)
            enc.setBuffer(idx.mtl, offset: idx.offset, index: 3)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        ctx.retainUntilFlush(runEnds)
        let indices = MetalArray<Int32>(length: n, nullCount: 0, validity: nil, values: idx, context: ctx)
        return try values.take(indices)
    }

    /// Run-end encodes this array (primitive, temporal or boolean). Already-encoded arrays are returned as-is.
    ///
    /// Runs are maximal stretches of adjacent elements that are equal *as raw bit patterns* and equally
    /// null. That is bit equality, not Arrow value equality: `-0.0` and `0.0` start different runs and a
    /// repeated NaN pattern stays one run. Nulls form runs of their own.
    public func runEndEncode() throws -> AnyMetalArray {
        switch self {
        case .runEndEncoded: return self
        case .int8(let a): return try a.runEndEncodedArray { .int8($0) }
        case .uint8(let a): return try a.runEndEncodedArray { .uint8($0) }
        case .int16(let a): return try a.runEndEncodedArray { .int16($0) }
        case .uint16(let a): return try a.runEndEncodedArray { .uint16($0) }
        case .int32(let a): return try a.runEndEncodedArray { .int32($0) }
        case .uint32(let a): return try a.runEndEncodedArray { .uint32($0) }
        case .int64(let a): return try a.runEndEncodedArray { .int64($0) }
        case .uint64(let a): return try a.runEndEncodedArray { .uint64($0) }
        case .float32(let a): return try a.runEndEncodedArray { .float32($0) }
        case .float64(let a): return try a.runEndEncodedArray { .float64($0) }
        case .boolean(let a):
            let (ends, vals) = try a.toUInt8Array().runEndParts()
            return .runEndEncoded(runEnds: ends, values: .boolean(try MetalBooleanArray.fromUInt8Array(vals)))
        case .temporal(let t):
            switch t.storage {
            case .int32(let a):
                let (ends, vals) = try a.runEndParts()
                return .runEndEncoded(runEnds: ends, values: .temporal(try MetalTemporalArray(type: t.type, vals)))
            case .int64(let a):
                let (ends, vals) = try a.runEndParts()
                return .runEndEncoded(runEnds: ends, values: .temporal(try MetalTemporalArray(type: t.type, vals)))
            }
        case .string, .binary, .dictionary:
            throw ArrowMetalError.unsupportedType("run-end encoding is defined for primitive, boolean and temporal arrays, not \(arrowFormat)")
        }
    }
}

extension MetalArray {
    /// Run-end encodes this array: `(runEnds, runValues)`.
    ///
    /// GPU: mark the first element of each run (bit equality plus null-ness of the neighbour), pack the
    /// marks into a bitmap, compact an iota through the existing `filter` to get the run starts, turn
    /// starts into ends with one kernel, and gather the run values with `take`.
    public func runEndParts() throws -> (runEnds: MetalArray<Int32>, values: MetalArray<T>) {
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        guard n > 0 else {
            return (try MetalArray<Int32>([Int32](), context: ctx), try MetalArray<T>([T](), context: ctx))
        }
        let uType = UniqueSource.unsignedType(width: T.byteWidth)
        let src = RunEndSource.marks(U: uType)
        let markBytes = try MetalArrowBuffer.allocate(byteCount: n, zeroed: false, context: ctx)
        let markPSO = try Dispatch.pipeline(ctx, family: "ree", source: src, function: "ree_mark", type: uType)
        let vld = validity ?? values
        try ctx.run { enc in
            enc.setComputePipelineState(markPSO)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 2)
            Dispatch.setLength(enc, n, nil, index: 3)
            enc.setBuffer(markBytes.mtl, offset: markBytes.offset, index: 4)
            Dispatch.dispatch1D(enc, markPSO, count: n)
        }
        ctx.retainUntilFlush(self)
        let selection = try BitmapOps.packBits(ctx, bytes: markBytes, bits: n)
        let mask = MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: selection, context: ctx)
        let starts = try MetalArray<Int32>.iota(n, context: ctx).filter(mask)
        let runs = starts.length
        let ends = try MetalArrowBuffer.allocate(byteCount: Swift.max(runs, 1) * 4, zeroed: false, context: ctx)
        let endPSO = try Dispatch.pipeline(ctx, family: "ree", source: RunEndSource.source, function: "ree_ends", type: "int")
        try ctx.run { enc in
            enc.setComputePipelineState(endPSO)
            enc.setBuffer(starts.values.mtl, offset: starts.values.offset, index: 0)
            Dispatch.setLength(enc, runs, nil, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            enc.setBuffer(ends.mtl, offset: ends.offset, index: 3)
            Dispatch.dispatch1D(enc, endPSO, count: runs)
        }
        let runEnds = MetalArray<Int32>(length: runs, nullCount: 0, validity: nil, values: ends, context: ctx)
        return (runEnds, try take(starts))
    }

    fileprivate func runEndEncodedArray(_ wrapValues: (MetalArray<T>) -> AnyMetalArray) throws -> AnyMetalArray {
        let (ends, vals) = try runEndParts()
        return .runEndEncoded(runEnds: ends, values: wrapValues(vals))
    }
}

// MARK: - Metal shading language

enum RunEndSource {
    /// Expansion (binary search per output element) and run-start to run-end conversion.
    static let source = KernelSource.prelude + """
    // The run covering output index i is the first run whose (exclusive) end is greater than i.
    kernel void ree_expand(device const int* runEnds [[buffer(0)]],
                           constant uint& runs [[buffer(1)]],
                           device const uint* nPtr [[buffer(2)]],
                           device int* out [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        uint lo = 0u, hi = runs;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if ((uint)runEnds[mid] > i) hi = mid; else lo = mid + 1u;
        }
        out[i] = (int)lo;
    }
    // Run ends from run starts: run j ends where run j+1 begins, and the last one ends at the length.
    kernel void ree_ends(device const int* starts [[buffer(0)]],
                         device const uint* runsPtr [[buffer(1)]],
                         constant uint& total [[buffer(2)]],
                         device int* out [[buffer(3)]],
                         uint j [[thread_position_in_grid]]) {
        uint runs = *runsPtr;
        if (j >= runs) return;
        out[j] = (j + 1u < runs) ? starts[j + 1u] : (int)total;
    }
    """

    /// Run boundaries in the original order. `U` is the unsigned type of the element width, so the
    /// comparison is a raw bit comparison and one kernel covers every fixed-width element type.
    static func marks(U: String) -> String { KernelSource.prelude + """
    kernel void ree_mark(device const \(U)* vals [[buffer(0)]],
                         device const uchar* validity [[buffer(1)]],
                         constant uint& hasValidity [[buffer(2)]],
                         device const uint* nPtr [[buffer(3)]],
                         device uchar* markBytes [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
        uint n = *nPtr;
        if (i >= n) return;
        if (i == 0u) { markBytes[0] = 1; return; }
        bool here = hasValidity ? bit_get(validity, i) : true;
        bool prev = hasValidity ? bit_get(validity, i - 1u) : true;
        uchar m;
        if (here != prev) m = 1;                       // a null next to a value always starts a run
        else if (!here) m = 0;                         // two nulls continue one run
        else m = (vals[i] != vals[i - 1u]) ? 1 : 0;
        markBytes[i] = m;
    }
    """ }
}

// MARK: - C Data Interface import

/// Imports a run-end encoded array. Run ends may be int16, int32 or int64 and are narrowed to int32.
func importRunEndArray(schema: UnsafePointer<ArrowSchema>, array: UnsafeMutablePointer<ArrowArray>,
                       context: MetalContext) throws -> ImportResult {
    guard array.pointee.release != nil else { throw ArrowMetalError.releasedArray }
    guard schema.pointee.n_children == 2, array.pointee.n_children == 2,
          let schemaChildren = schema.pointee.children, let arrayChildren = array.pointee.children,
          let runEndSchema = schemaChildren[0], let valueSchema = schemaChildren[1],
          let runEndArray = arrayChildren[0], let valueArray = arrayChildren[1] else {
        throw ArrowMetalError.invalidArrowArray("a run-end encoded array needs run_ends and values children")
    }
    guard array.pointee.offset == 0 else {
        throw ArrowMetalError.unsupportedType("run-end encoded arrays with a non-zero offset (decode on the producer side first)")
    }
    let logical = Int(array.pointee.length)
    // Move the children out so their lifetime is independent of the parent, as the struct importer does.
    func move(_ p: UnsafeMutablePointer<ArrowArray>) -> UnsafeMutablePointer<ArrowArray> {
        let moved = UnsafeMutablePointer<ArrowArray>.allocate(capacity: 1)
        moved.initialize(to: p.pointee)
        p.pointee.release = nil
        return moved
    }
    let movedEnds = move(runEndArray), movedValues = move(valueArray)
    defer { movedEnds.deallocate(); movedValues.deallocate() }
    let endsResult = try importArrowArray(schema: runEndSchema, array: movedEnds, context: context)
    let valuesResult = try importArrowArray(schema: valueSchema, array: movedValues, context: context)
    if let rel = array.pointee.release { rel(array) }
    var runEnds: MetalArray<Int32>
    switch endsResult.array {
    case .int32(let a): runEnds = a
    case .int16(let a): runEnds = try a.cast(to: Int32.self)
    case .int64(let a): runEnds = try a.cast(to: Int32.self)
    default: throw ArrowMetalError.invalidArrowArray("run ends must be int16, int32 or int64")
    }
    var values = valuesResult.array
    // A producer may hand over more runs than the logical length needs (a sliced array that kept its
    // runs). Trim to the runs the logical length covers and clamp the last end.
    let encoded = runEndLogicalLength(runEnds)
    if logical < encoded {
        var keep = 0
        withExtendedLifetime(runEnds) {
            let p = runEnds.valuePointer
            while keep < runEnds.length && Int(p[keep]) < logical { keep += 1 }
            if keep < runEnds.length { keep += 1 }
        }
        // Copy rather than slice: a zero-copy import borrows the producer's memory, and the last run end
        // has to be clamped to the logical length.
        let trimmed = try MetalArray<Int32>.allocate(length: keep, withValidity: false, context: context)
        withExtendedLifetime((runEnds, trimmed)) {
            let src = runEnds.valuePointer, dst = trimmed.mutableValuePointer
            for j in 0..<keep { dst[j] = src[j] }
            if keep > 0 { dst[keep - 1] = Int32(logical) }
        }
        values = try values.slice(offset: 0, length: keep)
        runEnds = trimmed
    } else if logical > encoded {
        throw ArrowMetalError.invalidArrowArray("run-end encoded array declares \(logical) values but its runs cover \(encoded)")
    }
    return ImportResult(array: .runEndEncoded(runEnds: runEnds, values: values),
                        zeroCopy: endsResult.zeroCopy && valuesResult.zeroCopy)
}

// MARK: - C Data Interface export

/// Owns the two exported children of a run-end encoded schema.
private final class RunEndSchemaHolder {
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowSchema>?>
    let childStructs: UnsafeMutablePointer<ArrowSchema>
    let formatC = strdup("+r")!
    let nameC: UnsafeMutablePointer<CChar>
    init(name: String) {
        nameC = strdup(name)!
        children = .allocate(capacity: 2)
        childStructs = .allocate(capacity: 2)
        childStructs.initialize(repeating: ArrowSchema(), count: 2)
        for i in 0..<2 { children[i] = childStructs + i }
    }
    deinit {
        for i in 0..<2 { if let r = childStructs[i].release { r(childStructs + i) } }
        children.deallocate(); childStructs.deallocate(); free(formatC); free(nameC)
    }
}

private final class RunEndArrayHolder {
    let children: UnsafeMutablePointer<UnsafeMutablePointer<ArrowArray>?>
    let childStructs: UnsafeMutablePointer<ArrowArray>
    init() {
        children = .allocate(capacity: 2)
        childStructs = .allocate(capacity: 2)
        childStructs.initialize(repeating: ArrowArray(), count: 2)
        for i in 0..<2 { children[i] = childStructs + i }
    }
    deinit {
        for i in 0..<2 { if let r = childStructs[i].release { r(childStructs + i) } }
        children.deallocate(); childStructs.deallocate()
    }
}

private func releaseRunEndSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<RunEndSchemaHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

private func releaseRunEndArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p = p, let pd = p.pointee.private_data else { return }
    Unmanaged<RunEndArrayHolder>.fromOpaque(pd).release()
    p.pointee.release = nil; p.pointee.private_data = nil
}

/// Writes a run-end encoded schema: "+r" with the `run_ends` (int32) and `values` children.
func exportRunEndSchema(values: AnyMetalArray, name: String, into out: UnsafeMutablePointer<ArrowSchema>) {
    let holder = RunEndSchemaHolder(name: name)
    ArrowMetal.exportArrowSchema(format: "i", name: "run_ends", into: holder.childStructs + 0)
    values.exportArrowSchema(name: "values", into: holder.childStructs + 1)
    // Arrow requires run_ends to be non-nullable.
    holder.childStructs[0].flags = 0
    out.pointee.format = UnsafePointer(holder.formatC)
    out.pointee.name = UnsafePointer(holder.nameC)
    out.pointee.metadata = nil
    out.pointee.flags = 0
    out.pointee.n_children = 2
    out.pointee.children = holder.children
    out.pointee.dictionary = nil
    out.pointee.release = releaseRunEndSchema
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}

/// Writes a run-end encoded array: no buffers of its own, the two children carry everything.
func exportRunEndArray(runEnds: MetalArray<Int32>, values: AnyMetalArray, into out: UnsafeMutablePointer<ArrowArray>) {
    let holder = RunEndArrayHolder()
    runEnds.exportArrowArray(into: holder.childStructs + 0)
    values.exportArrowArray(into: holder.childStructs + 1)
    out.pointee.length = Int64(runEndLogicalLength(runEnds))
    out.pointee.null_count = 0
    out.pointee.offset = 0
    out.pointee.n_buffers = 0
    out.pointee.n_children = 2
    out.pointee.buffers = nil
    out.pointee.children = holder.children
    out.pointee.dictionary = nil
    out.pointee.release = releaseRunEndArray
    out.pointee.private_data = Unmanaged.passRetained(holder).toOpaque()
}
