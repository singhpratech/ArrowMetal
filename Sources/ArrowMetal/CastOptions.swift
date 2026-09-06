import Foundation
import Metal

/// Arrow's `CastOptions`, field for field, plus the checked numeric cast the `safe` form needs.
///
/// ## What "safe" means
///
/// Arrow's default cast is **safe**: a conversion that would lose information raises rather than
/// quietly producing a different number. Each of the five `allow_*` flags turns one class of loss back
/// into the unchecked C-style behaviour, which is what this package did for every cast until now.
///
/// The classes, and which flag governs each — established against `pyarrow.compute.cast` rather than
/// from the documentation, because the mapping is not always the obvious one:
///
/// | conversion            | what is lost                              | flag                     |
/// |-----------------------|-------------------------------------------|--------------------------|
/// | int -> int            | the value does not fit the target         | `allowIntOverflow`       |
/// | float -> int          | a fractional part, or out of range        | `allowFloatTruncate`     |
/// | int -> float          | mantissa bits (`2^60 + 1` -> float32)     | `allowFloatTruncate`     |
/// | float -> float        | nothing Arrow objects to (overflow gives infinity) | — always allowed |
/// | temporal unit change  | sub-unit digits, or a range overflow      | `allowTimeTruncate` / `allowTimeOverflow` |
/// | decimal rescale       | digits below the target scale             | `allowDecimalTruncate`   |
///
/// ## How the check runs
///
/// The value kernel is the unchecked cast, unchanged, so a checked cast that does not raise is
/// bit-identical to the unchecked one. One extra read-only pass then converts each value *back* to the
/// source type and compares: a cast is lossless exactly when it round-trips. That single predicate
/// covers integer overflow, a float's fractional part, a float out of the integer range (the GPU
/// conversion saturates, so the round trip differs), infinity and NaN — every case in the table above
/// that involves an integer. Nothing is evaluated on a null row.
///
/// The pass writes only into a one-word buffer, through an atomic minimum, so the row it reports is the
/// *first* offending one — the row a sequential Arrow kernel would stop at.
public struct CastOptions: Sendable, Equatable {
    /// Let an integer conversion wrap instead of raising.
    public var allowIntOverflow: Bool
    /// Let a temporal unit conversion drop digits below the target resolution.
    public var allowTimeTruncate: Bool
    /// Let a temporal unit conversion overflow the target's range.
    public var allowTimeOverflow: Bool
    /// Let a decimal rescale drop digits below the target scale.
    public var allowDecimalTruncate: Bool
    /// Let a float lose its fractional part, leave the integer range, or lose mantissa bits.
    public var allowFloatTruncate: Bool
    /// Let invalid UTF-8 through when reinterpreting binary as a string.
    public var allowInvalidUTF8: Bool

    public init(allowIntOverflow: Bool = false, allowTimeTruncate: Bool = false,
                allowTimeOverflow: Bool = false, allowDecimalTruncate: Bool = false,
                allowFloatTruncate: Bool = false, allowInvalidUTF8: Bool = false) {
        self.allowIntOverflow = allowIntOverflow
        self.allowTimeTruncate = allowTimeTruncate
        self.allowTimeOverflow = allowTimeOverflow
        self.allowDecimalTruncate = allowDecimalTruncate
        self.allowFloatTruncate = allowFloatTruncate
        self.allowInvalidUTF8 = allowInvalidUTF8
    }

    /// Arrow's `safe=True`: every flag off, so any loss raises.
    public static let safe = CastOptions()

    /// Arrow's `safe=False`: every flag on, which is what this package's plain `cast` has always done.
    public static let unsafe = CastOptions(allowIntOverflow: true, allowTimeTruncate: true,
                                           allowTimeOverflow: true, allowDecimalTruncate: true,
                                           allowFloatTruncate: true, allowInvalidUTF8: true)

    /// The bit layout the C ABI passes these in, in Arrow's own field order.
    public init(bits: UInt32) {
        self.init(allowIntOverflow: bits & 1 != 0, allowTimeTruncate: bits & 2 != 0,
                  allowTimeOverflow: bits & 4 != 0, allowDecimalTruncate: bits & 8 != 0,
                  allowFloatTruncate: bits & 16 != 0, allowInvalidUTF8: bits & 32 != 0)
    }

    /// That same layout, going the other way.
    public var bits: UInt32 {
        (allowIntOverflow ? 1 : 0) | (allowTimeTruncate ? 2 : 0) | (allowTimeOverflow ? 4 : 0)
            | (allowDecimalTruncate ? 8 : 0) | (allowFloatTruncate ? 16 : 0) | (allowInvalidUTF8 ? 32 : 0)
    }
}

/// The check kernel, generated per source/target pair. `predicate` is the "this row loses something"
/// expression `lossPredicate` builds for that pair.
enum CastCheckSource {
    static func source(From: String, To: String, predicate: String) -> String { KernelSource.prelude + """

    // Flags the first row the conversion loses something on. `flag` starts at UINT_MAX and only ever
    // moves down through an atomic minimum, so the answer does not depend on thread order.
    kernel void cast_check(device const \(From)* a [[buffer(0)]],
                           device const uchar* validity [[buffer(1)]],
                           device const uint* nPtr [[buffer(2)]],
                           constant uint& hasValidity [[buffer(3)]],
                           device atomic_uint* flag [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        if (hasValidity != 0u && !bit_get(validity, i)) return;
        \(From) v = a[i];
        \(To) t = (\(To))v;
        \(From) back = (\(From))t;
        (void)t; (void)back;
        if (\(predicate)) atomic_fetch_min_explicit(flag, i, memory_order_relaxed);
    }
    """ }
}

extension MetalArray {

    /// Arrow `cast` between primitive types, with Arrow's `CastOptions`.
    ///
    /// With `.unsafe` this is exactly `cast(to:)` — the wrapping, C-style conversion — and costs the
    /// same one kernel. With any flag off, the conversions that flag governs are checked and a losing
    /// row raises `ArrowMetalError.overflow` naming it.
    public func cast<U: ArrowPrimitive>(to target: U.Type, options: CastOptions) throws -> MetalArray<U> {
        let out = try cast(to: target)
        if let failure = Self.lossClass(U.self), !options.allows(failure),
           let predicate = Self.lossPredicate(U.self) {
            try check(to: U.self, predicate: predicate,
                      detail: failure.message(from: T.arrowFormat, to: U.arrowFormat))
        }
        return out
    }

    /// The MSL expression that is true exactly on a row Arrow would refuse, in terms of `v` (the source
    /// value), `t` (the converted one) and `back` (the round trip). Nil when nothing can go wrong.
    ///
    /// * **int -> int** — the round trip must return the value *and* keep its sign. The sign clause is
    ///   what catches `int64(-1) -> uint64`, which round-trips bit for bit and is still out of range.
    /// * **float -> int** — the round trip alone: it catches a fractional part, a value past the
    ///   integer range (the conversion saturates, so the trip differs), an infinity and a NaN, which
    ///   never equals itself.
    /// * **int -> float** — Arrow's rule is not "round-trips" but "inside the contiguous integer range
    ///   of the float", 2^24 for float32 and 2^53 for float64, so `2^60 + 1` and `2^31` are both
    ///   refused into float32 even though the second is exactly representable.
    static func lossPredicate<U: ArrowPrimitive>(_: U.Type) -> String? {
        if U.self == T.self { return nil }
        if T.isFloatingPoint && U.isFloatingPoint { return nil }
        if T.isFloatingPoint { return "!(back == v)" }
        if U.isFloatingPoint {
            let limit: Int64 = U.byteWidth == 4 ? 1 << 24 : 1 << 53
            // A source that cannot reach the limit needs no check at all.
            if T.byteWidth * 8 - (T.minValue < 0 as T ? 1 : 0) <= (U.byteWidth == 4 ? 24 : 53) { return nil }
            let hi = "(\(T.mslType))\(limit)"
            return T.minValue < 0 as T ? "v > \(hi) || v < -\(hi)" : "v > \(hi)"
        }
        let sourceSigned = T.minValue < 0 as T, targetSigned = U.minValue < 0 as U
        if sourceSigned && !targetSigned { return "!(back == v) || v < 0" }
        if !sourceSigned && targetSigned { return "!(back == v) || t < 0" }
        return "!(back == v)"
    }

    /// Which of Arrow's loss classes this conversion can fall into, or nil when it cannot lose anything
    /// Arrow objects to (the identity, and every float-to-float pair, where overflow gives infinity).
    static func lossClass<U: ArrowPrimitive>(_: U.Type) -> CastLoss? {
        if U.self == T.self { return nil }
        if T.isFloatingPoint && U.isFloatingPoint { return nil }
        if T.isFloatingPoint { return .floatToInt }
        if U.isFloatingPoint { return .intToFloat }
        return .intOverflow
    }

    /// One read-only pass evaluating `predicate`. Raises naming the first row it fires on.
    private func check<U: ArrowPrimitive>(to _: U.Type, predicate: String, detail: String) throws {
        let n = dispatchLength
        guard n > 0 else { return }
        let ctx = context
        if !Dispatch.runsOnGPU(T.self) || !Dispatch.runsOnGPU(U.self) {
            // Float64 has no native Metal type, so its casts run on the host and so does their check.
            let src = valuePointer
            for i in 0..<length where isValid(i) {
                if Self.losesOnHost(src[i], U.self) {
                    throw ArrowMetalError.overflow(op: "cast", index: i, detail: detail)
                }
            }
            return
        }
        let flag = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: false, context: ctx)
        flag.mutableTyped(UInt32.self)[0] = .max
        let pso = try Dispatch.pipeline(ctx, family: "cast-check",
                                        source: CastCheckSource.source(From: T.mslType, To: U.mslType,
                                                                       predicate: predicate),
                                        function: "cast_check", type: "\(T.mslType)->\(U.mslType)")
        let vld = validity ?? values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(vld.mtl, offset: vld.offset, index: 1)
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            Dispatch.setUInt(enc, validity == nil ? 0 : 1, index: 3)
            enc.setBuffer(flag.mtl, offset: flag.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: n)
        }
        try ctx.syncPoint()
        let first = flag.typed(UInt32.self)[0]
        if first != .max { throw ArrowMetalError.overflow(op: "cast", index: Int(first), detail: detail) }
    }
}

extension MetalArray {
    /// The same three rules as `lossPredicate`, on the host, for the float64 conversions Metal cannot
    /// run (it has no `double`).
    static func losesOnHost<U: ArrowPrimitive>(_ v: T, _: U.Type) -> Bool {
        if U.self == T.self { return false }
        if T.isFloatingPoint && U.isFloatingPoint { return false }
        if T.isFloatingPoint { return T.convert(U.convert(v)) != v }
        if U.isFloatingPoint {
            let limit = Double(U.byteWidth == 4 ? Int64(1) << 24 : Int64(1) << 53)
            let d = v.asDouble
            return d > limit || d < -limit
        }
        let sourceSigned = T.minValue < 0 as T, targetSigned = U.minValue < 0 as U
        let back = T.convert(U.convert(v))
        if back != v { return true }
        if sourceSigned && !targetSigned { return v < 0 as T }
        if !sourceSigned && targetSigned { return U.convert(v) < 0 as U }
        return false
    }
}

/// The classes of loss Arrow's cast flags govern.
public enum CastLoss: Sendable {
    case intOverflow, floatToInt, intToFloat, timeTruncate, timeOverflow, decimalTruncate

    func message(from: String, to: String) -> String {
        switch self {
        case .intOverflow: return "value does not fit \(to) (cast \(from) -> \(to))"
        case .floatToInt: return "value is not an exact \(to) (cast \(from) -> \(to))"
        case .intToFloat: return "value is not exactly representable in \(to) (cast \(from) -> \(to))"
        case .timeTruncate: return "value would lose digits below the \(to) resolution (cast \(from) -> \(to))"
        case .timeOverflow: return "value is out of range for \(to) (cast \(from) -> \(to))"
        case .decimalTruncate: return "rescaling would lose digits (cast \(from) -> \(to))"
        }
    }
}

extension CastOptions {
    /// Whether this loss class is permitted.
    func allows(_ loss: CastLoss) -> Bool {
        switch loss {
        case .intOverflow: return allowIntOverflow
        case .floatToInt, .intToFloat: return allowFloatTruncate
        case .timeTruncate: return allowTimeTruncate
        case .timeOverflow: return allowTimeOverflow
        case .decimalTruncate: return allowDecimalTruncate
        }
    }
}
