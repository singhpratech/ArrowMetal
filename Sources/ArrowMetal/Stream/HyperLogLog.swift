import Foundation
import Metal

// A HyperLogLog sketch built on the GPU, one pass per batch, merged across batches on the host.
//
// Exact `count_distinct` over a dataset larger than memory would need the whole distinct set in
// memory. HLL needs 2^p bytes total, no matter how many rows or how many distinct values: every row
// is hashed to 64 bits, the top p bits pick a register and the position of the first 1 in the rest
// updates that register with a max. Registers of two sketches merge by taking the element-wise max,
// which is what makes it a *streaming* aggregate — batch order does not matter and nothing grows.
//
// On the GPU one thread handles one row and updates its register with `atomic_fetch_max_explicit`
// on a 2^p-entry uint array in device memory (64 KB at the default p = 14). Contention is low: the
// hash spreads rows across 16,384 registers and a max only ever moves a register upward, so most
// atomics are no-ops that the hardware resolves without a retry.
//
// Accuracy: the standard error is 1.04 / sqrt(2^p) — 0.81% at p = 14, 0.41% at p = 16. Small
// cardinalities use linear counting (the estimator switches when more than a few registers are
// still zero), which is exact-ish below roughly 2.5 * 2^p rows.

/// A HyperLogLog sketch: 2^precision one-byte registers plus the estimator.
public struct HLLSketch: Sendable, Equatable {
    public let precision: Int
    public private(set) var registers: [UInt8]

    public init(precision: Int = 14) {
        let p = Swift.min(18, Swift.max(4, precision))
        self.precision = p
        self.registers = [UInt8](repeating: 0, count: 1 << p)
    }

    init(precision: Int, registers: [UInt8]) {
        self.precision = precision
        self.registers = registers
    }

    /// Element-wise max: the union of the two sketches' inputs.
    public mutating func merge(_ other: HLLSketch) {
        precondition(other.precision == precision, "HLL sketches must share a precision to merge")
        for i in 0..<registers.count where other.registers[i] > registers[i] {
            registers[i] = other.registers[i]
        }
    }

    public func merged(_ other: HLLSketch) -> HLLSketch {
        var c = self
        c.merge(other)
        return c
    }

    /// Estimated number of distinct non-null values.
    public var estimate: Double {
        let m = Double(registers.count)
        var sum = 0.0
        var zeros = 0
        for r in registers {
            sum += pow(2.0, -Double(r))
            if r == 0 { zeros += 1 }
        }
        let alpha: Double
        switch registers.count {
        case 16: alpha = 0.673
        case 32: alpha = 0.697
        case 64: alpha = 0.709
        default: alpha = 0.7213 / (1.0 + 1.079 / m)
        }
        let raw = alpha * m * m / sum
        // Small-range correction: with registers still empty, linear counting is far more accurate.
        if raw <= 2.5 * m, zeros > 0 { return m * log(m / Double(zeros)) }
        return raw
    }

    /// The estimate rounded to a count.
    public var count: Int { Int(estimate.rounded()) }
    /// The relative standard error this precision guarantees.
    public var standardError: Double { 1.04 / (Double(registers.count)).squareRoot() }
}

enum HLLSource {
    /// Fixed-width values. `LOAD` turns `vals[i]` into the 64 bits that are hashed.
    static func source(T: String, load: String) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        inline ulong am_mix64(ulong z) {
            z += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
        inline uint am_clz64(ulong x) {
            if (x == 0) return 64;
            uint n = 0;
            if ((x >> 32) == 0) { n += 32; x <<= 32; }
            if ((x >> 48) == 0) { n += 16; x <<= 16; }
            if ((x >> 56) == 0) { n += 8;  x <<= 8;  }
            if ((x >> 60) == 0) { n += 4;  x <<= 4;  }
            if ((x >> 62) == 0) { n += 2;  x <<= 2;  }
            if ((x >> 63) == 0) { n += 1; }
            return n;
        }
        inline void am_hll_update(device atomic_uint* regs, ulong h, uint p) {
            uint idx = (uint)(h >> (64 - p));
            ulong rest = h << p;
            uint rank = am_clz64(rest) + 1;
            if (rank > 64 - p) rank = 64 - p;         // the rest only carries 64 - p bits
            atomic_fetch_max_explicit(&regs[idx], rank, memory_order_relaxed);
        }

        kernel void hll_add(device const \(T)* vals [[buffer(0)]],
                            device const uchar* validity [[buffer(1)]],
                            device const uint* nPtr [[buffer(2)]],
                            constant uint& hasV [[buffer(3)]],
                            constant uint& p [[buffer(4)]],
                            device atomic_uint* regs [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
            uint n = nPtr[0];
            if (gid >= n) return;
            if (hasV != 0 && ((validity[gid >> 3] >> (gid & 7)) & 1) == 0) return;
            ulong raw = \(load);
            am_hll_update(regs, am_mix64(raw), p);
        }
        """
    }

    /// utf8 / binary values: hash the bytes with FNV-1a, then mix.
    static let stringSource = """
    #include <metal_stdlib>
    using namespace metal;

    inline ulong am_mix64(ulong z) {
        z += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
    inline uint am_clz64(ulong x) {
        if (x == 0) return 64;
        uint n = 0;
        if ((x >> 32) == 0) { n += 32; x <<= 32; }
        if ((x >> 48) == 0) { n += 16; x <<= 16; }
        if ((x >> 56) == 0) { n += 8;  x <<= 8;  }
        if ((x >> 60) == 0) { n += 4;  x <<= 4;  }
        if ((x >> 62) == 0) { n += 2;  x <<= 2;  }
        if ((x >> 63) == 0) { n += 1; }
        return n;
    }

    kernel void hll_add_str(device const int* offsets [[buffer(0)]],
                            device const uchar* data [[buffer(1)]],
                            device const uchar* validity [[buffer(2)]],
                            device const uint* nPtr [[buffer(3)]],
                            constant uint& hasV [[buffer(4)]],
                            constant uint& p [[buffer(5)]],
                            device atomic_uint* regs [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
        uint n = nPtr[0];
        if (gid >= n) return;
        if (hasV != 0 && ((validity[gid >> 3] >> (gid & 7)) & 1) == 0) return;
        int lo = offsets[gid], hi = offsets[gid + 1];
        ulong h = 0xCBF29CE484222325UL;
        for (int i = lo; i < hi; ++i) {
            h ^= (ulong)data[i];
            h *= 0x100000001B3UL;
        }
        h = am_mix64(h ^ (ulong)(hi - lo));
        uint idx = (uint)(h >> (64 - p));
        ulong rest = h << p;
        uint rank = am_clz64(rest) + 1;
        if (rank > 64 - p) rank = 64 - p;
        atomic_fetch_max_explicit(&regs[idx], rank, memory_order_relaxed);
    }
    """
}

/// Builds a HyperLogLog sketch of one column's non-null values, on the GPU.
///
/// One pass, one atomic max per row into a 2^precision register table in device memory. Null rows are
/// skipped (Arrow's `count_distinct` does not count nulls).
public func gpuHyperLogLog(_ column: AnyMetalArray, precision: Int = 14,
                           into existing: HLLSketch? = nil) throws -> HLLSketch {
    let p = Swift.min(18, Swift.max(4, precision))
    let m = 1 << p
    let ctx = column.anyContext
    let regs = try MetalArrowBuffer.allocate(byteCount: m * 4, context: ctx)
    if let e = existing {
        precondition(e.precision == p)
        let dst = regs.mutableTyped(UInt32.self)
        for i in 0..<m { dst[i] = UInt32(e.registers[i]) }
    }
    try addToHLL(column, precision: p, registers: regs, context: ctx)
    try ctx.syncPoint()
    let src = regs.typed(UInt32.self)
    var out = [UInt8](repeating: 0, count: m)
    for i in 0..<m { out[i] = UInt8(Swift.min(src[i], 255)) }
    return HLLSketch(precision: p, registers: out)
}

private func addToHLL(_ column: AnyMetalArray, precision p: Int,
                      registers: MetalArrowBuffer, context ctx: MetalContext) throws {
    switch column {
    case .int8(let a): try hllPrimitive(a, "char", "(ulong)(long)vals[gid]", p, registers, ctx)
    case .int16(let a): try hllPrimitive(a, "short", "(ulong)(long)vals[gid]", p, registers, ctx)
    case .int32(let a): try hllPrimitive(a, "int", "(ulong)(long)vals[gid]", p, registers, ctx)
    case .int64(let a): try hllPrimitive(a, "long", "(ulong)vals[gid]", p, registers, ctx)
    case .uint8(let a): try hllPrimitive(a, "uchar", "(ulong)vals[gid]", p, registers, ctx)
    case .uint16(let a): try hllPrimitive(a, "ushort", "(ulong)vals[gid]", p, registers, ctx)
    case .uint32(let a): try hllPrimitive(a, "uint", "(ulong)vals[gid]", p, registers, ctx)
    case .uint64(let a): try hllPrimitive(a, "ulong", "vals[gid]", p, registers, ctx)
    // -0.0 and 0.0 are the same value: normalise the bit pattern before hashing.
    case .float32(let a): try hllPrimitive(a, "float", "(ulong)as_type<uint>(vals[gid] + 0.0f)", p, registers, ctx)
    case .float64(let a):
        try hllPrimitive(a, "ulong", "(vals[gid] == 0x8000000000000000UL ? 0UL : vals[gid])", p, registers, ctx)
    case .boolean(let a):
        // A boolean column has at most three distinct values; unpack to bytes and hash those.
        try hllPrimitive(try a.toUInt8Array(), "uchar", "(ulong)(vals[gid] != 0 ? 1 : 0)", p, registers, ctx)
    case .temporal(let t):
        switch t.storage {
        case .int32(let a): try hllPrimitive(a, "int", "(ulong)(long)vals[gid]", p, registers, ctx)
        case .int64(let a): try hllPrimitive(a, "long", "(ulong)vals[gid]", p, registers, ctx)
        }
    case .string(let a), .binary(let a): try hllString(a, p, registers, ctx)
    case .dictionary: try addToHLL(try column.decode(), precision: p, registers: registers, context: ctx)
    default:
        throw ArrowMetalError.unsupportedType("count_distinct_approx does not support \(column.arrowFormat)")
    }
}

private func hllPrimitive<T: ArrowPrimitive>(_ a: MetalArray<T>, _ mslType: String, _ load: String,
                                            _ p: Int, _ regs: MetalArrowBuffer, _ ctx: MetalContext) throws {
    let n = a.dispatchLength
    guard n > 0 else { return }
    try Dispatch.checkLength(n)
    let pso = try Dispatch.pipeline(ctx, family: "hll", source: HLLSource.source(T: mslType, load: load),
                                    function: "hll_add", type: mslType + "/" + load)
    let vv = a.validity ?? a.values
    try ctx.run { enc in
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.values.mtl, offset: a.values.offset, index: 0)
        enc.setBuffer(vv.mtl, offset: vv.offset, index: 1)
        Dispatch.setLength(enc, n, a.lengthBuffer, index: 2)
        Dispatch.setUInt(enc, a.validity == nil ? 0 : 1, index: 3)
        Dispatch.setUInt(enc, p, index: 4)
        enc.setBuffer(regs.mtl, offset: regs.offset, index: 5)
        Dispatch.dispatch1D(enc, pso, count: n)
    }
    ctx.retainUntilFlush(a)
    ctx.retainUntilFlush(regs)
}

private func hllString(_ a: MetalStringArray, _ p: Int, _ regs: MetalArrowBuffer, _ ctx: MetalContext) throws {
    let n = a.length
    guard n > 0 else { return }
    try Dispatch.checkLength(n)
    let pso = try Dispatch.pipeline(ctx, family: "hll", source: HLLSource.stringSource,
                                    function: "hll_add_str", type: "str")
    let vv = a.validity ?? a.offsets
    try ctx.run { enc in
        enc.setComputePipelineState(pso)
        enc.setBuffer(a.offsets.mtl, offset: a.offsets.offset, index: 0)
        enc.setBuffer(a.data.mtl, offset: a.data.offset, index: 1)
        enc.setBuffer(vv.mtl, offset: vv.offset, index: 2)
        Dispatch.setLength(enc, n, nil, index: 3)
        Dispatch.setUInt(enc, a.validity == nil ? 0 : 1, index: 4)
        Dispatch.setUInt(enc, p, index: 5)
        enc.setBuffer(regs.mtl, offset: regs.offset, index: 6)
        Dispatch.dispatch1D(enc, pso, count: n)
    }
    ctx.retainUntilFlush(a)
    ctx.retainUntilFlush(regs)
}
