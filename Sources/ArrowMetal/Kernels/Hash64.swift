import Foundation
import Metal

/// A 64-bit hash of primitive values, on the GPU.
///
/// Arrow C++ exposes no public element-wise hash compute function (its `hash_*` names are grouped
/// aggregates), so this is an ArrowMetal extension, defined here so that it is reproducible from the
/// specification alone:
///
/// > `hash64(v) = fmix64(normalise(v) ^ 0x9E3779B97F4A7C15)`
///
/// where `fmix64` is the MurmurHash3 128-bit finaliser
///
///     k ^= k >> 33;  k *= 0xFF51AFD7ED558CCD;
///     k ^= k >> 33;  k *= 0xC4CEB9FE1A85EC53;
///     k ^= k >> 33;
///
/// and `normalise` maps the value to a 64-bit word:
///
///   * **integers** — the value's own bytes, read as the unsigned type of the same width and
///     zero-extended. `int8(-1)` and `uint8(255)` therefore hash alike, and `int32(1)` hashes like
///     `int64(1)`; the hash identifies a *value*, not a value-and-type pair.
///   * **boolean** — 0 or 1.
///   * **float32 / float64** — the IEEE bit pattern with the two ArrowMetal float normalisations
///     applied first: `-0.0` becomes `+0.0` and every NaN becomes the canonical quiet NaN. That is
///     exactly the equality `unique()`, `is_in` and the group-by keys use, so **equal values always
///     hash equal**, which is the property a hash join needs. float32 patterns are zero-extended, so
///     `float32(1.0)` and `float64(1.0)` hash differently.
///
/// The golden-ratio seed is there so that the value 0 does not hash to 0 (`fmix64(0)` is 0), leaving
/// 0 free as the hash of a null. Nulls hash to 0 **and stay null**: the result shares the input's
/// validity bitmap zero-copy, matching the existing `hash32` over utf8.
///
/// The hash is deterministic, endian-independent (Arrow buffers are little-endian and the kernel
/// reads typed values, not bytes) and identical for a column and any slice of it.
enum Hash64Source {
    /// `T` is the MSL type the values are read as, `normalise` an expression over `v` giving the
    /// 64-bit word to hash.
    static func source(T: String, normalise: String) -> String { KernelSource.prelude + """

    inline ulong h64_fmix(ulong k) {
        k ^= k >> 33;
        k *= 0xFF51AFD7ED558CCDul;
        k ^= k >> 33;
        k *= 0xC4CEB9FE1A85EC53ul;
        k ^= k >> 33;
        return k;
    }
    kernel void h64_hash(device const \(T)* a [[buffer(0)]],
                         device const uchar* validity [[buffer(1)]],
                         device const uint* nPtr [[buffer(2)]],
                         constant uint& hasValidity [[buffer(3)]],
                         device ulong* out [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity && !bit_get(validity, i)) { out[i] = 0ul; return; }
        \(T) v = a[i];
        ulong w = \(normalise);
        out[i] = h64_fmix(w ^ 0x9E3779B97F4A7C15ul);
    }
    // Boolean columns: the value is a bit in a packed bitmap, so it needs its own kernel.
    kernel void h64_hash_bits(device const uchar* bits [[buffer(0)]],
                              device const uchar* validity [[buffer(1)]],
                              device const uint* nPtr [[buffer(2)]],
                              constant uint& hasValidity [[buffer(3)]],
                              device ulong* out [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity && !bit_get(validity, i)) { out[i] = 0ul; return; }
        ulong w = bit_get(bits, i) ? 1ul : 0ul;
        out[i] = h64_fmix(w ^ 0x9E3779B97F4A7C15ul);
    }
    """ }

    /// The normalisation expression and read type for each element type.
    static func spec<T: ArrowPrimitive>(_: T.Type) -> (T: String, normalise: String, key: String) {
        if T.self == Double.self {
            return ("ulong",
                    "((v & 0x7FFFFFFFFFFFFFFFul) == 0ul) ? 0ul : (((v & 0x7FFFFFFFFFFFFFFFul) > 0x7FF0000000000000ul) ? 0x7FF8000000000000ul : v)",
                    "f64")
        }
        if T.self == Float.self {
            return ("uint",
                    "(ulong)(((v & 0x7FFFFFFFu) == 0u) ? 0u : (((v & 0x7FFFFFFFu) > 0x7F800000u) ? 0x7FC00000u : v))",
                    "f32")
        }
        return (MathTypes.unsigned(T.mslType), "(ulong)v", T.mslType)
    }
}

extension MetalArray {
    /// A 64-bit hash of every element (see `Hash64Source` for the exact algorithm). Nulls hash to 0
    /// and stay null; the result shares the input's validity bitmap zero-copy.
    public func hash64() throws -> MetalArray<UInt64> {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        if n > 0 {
            let s = Hash64Source.spec(T.self)
            let pso = try Dispatch.pipeline(ctx, family: "hash64", source: Hash64Source.source(T: s.T, normalise: s.normalise),
                                            function: "h64_hash", type: s.key)
            let vld = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalArray<UInt64>(length: knownLength, nullCount: _nullCount,
                                                 validity: validity, values: out, context: ctx))
    }
}

extension MetalBooleanArray {
    /// A 64-bit hash of a boolean column: `false` hashes as 0, `true` as 1, null as 0 (and stays null).
    public func hash64() throws -> MetalArray<UInt64> {
        let n = dispatchLength
        try Dispatch.checkLength(n)
        let ctx = context
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 8, zeroed: false, context: ctx)
        if n > 0 {
            let pso = try Dispatch.pipeline(ctx, family: "hash64",
                                            source: Hash64Source.source(T: "uchar", normalise: "(ulong)v"),
                                            function: "h64_hash_bits", type: "uchar")
            let vld = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
                Dispatch.setLength(enc, n, lengthBuffer, index: 2)
                Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
                enc.setBuffer(out.mtl, offset: out.offset, index: 4)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
            ctx.retainUntilFlush(self)
        }
        return inheritPending(MetalArray<UInt64>(length: knownLength, nullCount: _nullCount,
                                                 validity: validity, values: out, context: ctx))
    }
}
