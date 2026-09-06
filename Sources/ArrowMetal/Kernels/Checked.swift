import Foundation
import Metal

/// Arrow's *checked* arithmetic: the same answers as the unchecked kernels, except that an element that
/// would have wrapped, divided by zero or left a function's domain raises `ArrowMetalError.overflow`
/// naming the op, the Arrow message and the first offending row.
///
/// **How it runs.** Every checked op is the ordinary unchecked kernel *plus* one read-only check pass,
/// both encoded into a single command buffer (`MetalContext.batch`), so a checked op costs one GPU round
/// trip like an unchecked one. The check pass writes only into an eight-word flag buffer, and only from
/// an element that actually fails — see `CheckedSource` for the mechanism. Because the value kernel is
/// literally the unchecked one, a checked result that does not raise is *bit-identical* to the unchecked
/// result, on every type including the software binary64 path.
///
/// **Batched mode.** Inside `MetalContext.batch { }` the check joins the caller's open command buffer and
/// the flag is read at the sync point, so the error surfaces from `flush()` / `syncPoint()` (or from the
/// first CPU-side read that forces one), not from the call that queued it. Outside a batch the check is
/// synchronous and the call itself throws.
///
/// **Nulls never raise.** The predicate is not evaluated on a null row, on either side of a binary op.
///
/// **Floats.** Arrow's checked float kernels do not treat infinity or NaN as errors: `add_checked` on
/// two huge float64s returns `inf`, and `abs_checked(NaN)` returns NaN. The only float failures are
/// `divide_checked` by zero and the domain errors of `sqrt`, `ln`, `log2`, `log10`, `log1p` and `logb`.
/// ArrowMetal follows that exactly, so on float columns `add`/`subtract`/`multiply`/`power`/`negate`/
/// `abs` skip the check pass altogether and cost nothing extra.
public enum CheckedOp: String, CaseIterable, Sendable {
    case add = "add_checked"
    case subtract = "subtract_checked"
    case multiply = "multiply_checked"
    case divide = "divide_checked"
    case negate = "negate_checked"
    case abs = "abs_checked"
    case power = "power_checked"
    case sqrt = "sqrt_checked"
    case shiftLeft = "shift_left_checked"
    case shiftRight = "shift_right_checked"
    case ln = "ln_checked"
    case log10 = "log10_checked"
    case log2 = "log2_checked"
    case log1p = "log1p_checked"
    case logb = "logb_checked"
    case cumulativeSum = "cumulative_sum_checked"
    case cumulativeProd = "cumulative_prod_checked"
    case pairwiseDiff = "pairwise_diff_checked"
}

/// The device-side flag buffer a check pass reports through, and the error it turns into.
enum CheckedFlags {
    /// Word 0 is the bitmask of failure kinds; words 1... are the first offending row of each kind.
    static func makeBuffer(_ ctx: MetalContext) throws -> MetalArrowBuffer {
        let b = try MetalArrowBuffer.allocate(byteCount: CheckedSource.flagWords * 4, zeroed: true, context: ctx)
        let p = b.mutableTyped(UInt32.self)
        for k in CheckedSource.Failure.allCases { p[1 + k.rawValue] = UInt32.max }
        return b
    }

    /// The error to raise, or nil when every element was in range. When more than one kind fired, the one
    /// with the smallest row wins, so the message names the failure a sequential Arrow kernel would hit first.
    static func error(_ b: MetalArrowBuffer, op: String) -> ArrowMetalError? {
        let p = b.typed(UInt32.self)
        let bits = p[0]
        guard bits != 0 else { return nil }
        var best: (kind: CheckedSource.Failure, index: UInt32)? = nil
        for f in CheckedSource.Failure.allCases where bits & (1 << UInt32(f.rawValue)) != 0 {
            let idx = p[1 + f.rawValue]
            if best == nil || idx < best!.index { best = (f, idx) }
        }
        guard let b = best else { return nil }
        return .overflow(op: op, index: b.index == .max ? nil : Int(b.index), detail: b.kind.message)
    }
}

extension MetalArray {
    /// Generated check kernels for this element type, plus the pipeline cache key.
    static var checkedSource: (source: String, type: String) {
        if T.self == Double.self {
            return (CheckedSource.source(T: "ulong", U: "ulong", width: 64, kind: .float64), "double")
        }
        if T.self == Float.self {
            return (CheckedSource.source(T: "float", U: "float", width: 32, kind: .float32), "float")
        }
        let t = T.mslType
        return (CheckedSource.source(T: t, U: MathTypes.unsigned(t), width: MathTypes.bitWidth(T.self),
                                     kind: MathTypes.isSigned(T.self) ? .signedInt : .unsignedInt), t)
    }

    /// Arrow's checked float kernels raise only on a domain error, never on overflow to infinity, so the
    /// range checks are simply absent on `float32` / `float64`.
    static func needsCheck(_ op: CheckedOp) -> Bool {
        guard T.isFloatingPoint else { return true }
        switch op {
        case .divide, .sqrt, .ln, .log10, .log2, .log1p, .logb: return true
        default: return false
        }
    }

    /// Queues "read the flag and throw" for the sync point. Outside a batch `afterFlush` runs it now.
    private func registerCheck(_ flags: MetalArrowBuffer, _ op: CheckedOp, retaining: [AnyObject]) throws {
        let ctx = context
        try ctx.afterFlush { if let e = CheckedFlags.error(flags, op: op.rawValue) { throw e } }
        ctx.retainUntilFlush(flags)
        for o in retaining { ctx.retainUntilFlush(o) }
    }

    private func checkPipeline(_ function: String) throws -> MTLComputePipelineState {
        let (src, ty) = Self.checkedSource
        return try Dispatch.pipeline(context, family: "checked", source: src, function: function, type: ty)
    }

    /// One-input check pass (`negate`, `abs`, `sqrt`, the logarithms).
    private func checkUnary(_ function: String, _ op: CheckedOp) throws {
        let n = dispatchLength
        guard n > 0 else { return }
        let ctx = context
        let pso = try checkPipeline(function)
        let flags = try CheckedFlags.makeBuffer(ctx)
        let v = validity ?? values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(v.mtl, offset: v.offset, index: 1)
            Dispatch.setLength(enc, n, lengthBuffer, index: 2)
            Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 3)
            enc.setBuffer(flags.mtl, offset: flags.offset, index: 4)
            Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)   // four elements per thread
        }
        try registerCheck(flags, op, retaining: [self])
    }

    /// Check pass against a scalar right-hand side.
    private func checkScalar(_ function: String, _ op: CheckedOp, _ scalar: T) throws {
        let n = dispatchLength
        guard n > 0 else { return }
        let ctx = context
        let pso = try checkPipeline(function)
        let flags = try CheckedFlags.makeBuffer(ctx)
        let v = validity ?? values
        let isDouble = T.self == Double.self
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            if isDouble { Dispatch.setScalar(enc, (scalar as! Double).bitPattern, index: 1) }
            else { Dispatch.setScalar(enc, scalar, index: 1) }
            enc.setBuffer(v.mtl, offset: v.offset, index: 2)
            Dispatch.setLength(enc, n, lengthBuffer, index: 3)
            Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 4)
            enc.setBuffer(flags.mtl, offset: flags.offset, index: 5)
            Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)   // four elements per thread
        }
        try registerCheck(flags, op, retaining: [self])
    }

    /// Check pass against another column. A row is skipped when either side is null, which is exactly the
    /// rule the unchecked binary kernels use to build the output validity.
    private func checkArray(_ function: String, _ op: CheckedOp, _ other: MetalArray<T>) throws {
        let n = dispatchLength
        guard n > 0 else { return }
        let ctx = context
        let pso = try checkPipeline(function)
        let flags = try CheckedFlags.makeBuffer(ctx)
        let va = validity ?? values, vb = other.validity ?? other.values
        try ctx.run { enc in
            enc.setComputePipelineState(pso)
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(other.values.mtl, offset: other.values.offset, index: 1)
            enc.setBuffer(va.mtl, offset: va.offset, index: 2)
            enc.setBuffer(vb.mtl, offset: vb.offset, index: 3)
            Dispatch.setLength(enc, n, lengthBuffer, index: 4)
            Dispatch.setUInt(enc, (validity != nil ? 1 : 0) | (other.validity != nil ? 2 : 0), index: 5)
            enc.setBuffer(flags.mtl, offset: flags.offset, index: 6)
            Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)   // four elements per thread
        }
        try registerCheck(flags, op, retaining: [self, other])
    }

    // MARK: - Checked arithmetic

    /// `add_checked` / `subtract_checked` / `multiply_checked` / `divide_checked` against a scalar.
    public func arithmeticChecked(_ op: ArithmeticOp, _ scalar: T) throws -> MetalArray<T> {
        let checked = Self.checkedArithmeticOp(op)
        return try context.batch {
            let out = try arithmetic(op, scalar)
            if Self.needsCheck(checked) { try checkScalar("chk_scalar_\(Self.kernelSuffix(op))", checked, scalar) }
            return out
        }
    }

    /// `add_checked` / `subtract_checked` / `multiply_checked` / `divide_checked` against another column.
    public func arithmeticChecked(_ op: ArithmeticOp, _ other: MetalArray<T>) throws -> MetalArray<T> {
        let checked = Self.checkedArithmeticOp(op)
        return try context.batch {
            let out = try arithmetic(op, other)
            if Self.needsCheck(checked) { try checkArray("chk_array_\(Self.kernelSuffix(op))", checked, other) }
            return out
        }
    }

    private static func checkedArithmeticOp(_ op: ArithmeticOp) -> CheckedOp {
        switch op {
        case .add: return .add
        case .sub: return .subtract
        case .mul: return .multiply
        case .div: return .divide
        }
    }
    private static func kernelSuffix(_ op: ArithmeticOp) -> String { op.rawValue }

    public func addChecked(_ s: T) throws -> MetalArray<T> { try arithmeticChecked(.add, s) }
    public func addChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmeticChecked(.add, o) }
    public func subtractChecked(_ s: T) throws -> MetalArray<T> { try arithmeticChecked(.sub, s) }
    public func subtractChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmeticChecked(.sub, o) }
    public func multiplyChecked(_ s: T) throws -> MetalArray<T> { try arithmeticChecked(.mul, s) }
    public func multiplyChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmeticChecked(.mul, o) }
    public func divideChecked(_ s: T) throws -> MetalArray<T> { try arithmeticChecked(.div, s) }
    public func divideChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try arithmeticChecked(.div, o) }

    // MARK: - Checked unary

    /// `negate_checked`: `int8 -128` raises, and on an unsigned column every non-zero value raises.
    ///
    /// Arrow ships no unsigned `negate_checked` kernel at all (pyarrow answers
    /// `ArrowNotImplementedError`); ArrowMetal defines it as the only thing a range-checked modular
    /// negation could say.
    public func negateChecked() throws -> MetalArray<T> {
        try context.batch {
            let out = try unaryMath(.negate)
            if Self.needsCheck(.negate) { try checkUnary("chk_unary_negate", .negate) }
            return out
        }
    }

    /// `abs_checked`: only `T.min` on a signed integer column raises.
    public func absChecked() throws -> MetalArray<T> {
        try context.batch {
            let out = try unaryMath(.abs)
            if Self.needsCheck(.abs) { try checkUnary("chk_unary_abs", .abs) }
            return out
        }
    }

    /// `sqrt_checked`: a negative value raises. NaN, `-0.0` and `+inf` do not.
    public func sqrtChecked() throws -> MetalArray<T> {
        try context.batch {
            let out = try unaryMath(.sqrt)
            try checkUnary("chk_unary_sqrt", .sqrt)
            return out
        }
    }

    /// `ln_checked` / `log2_checked` / `log10_checked`: zero raises "logarithm of zero" and a negative
    /// value "logarithm of negative number". NaN and `+inf` pass through.
    public func logChecked(_ op: UnaryMathOp) throws -> MetalArray<T> {
        let checked: CheckedOp
        switch op {
        case .ln: checked = .ln
        case .log2: checked = .log2
        case .log10: checked = .log10
        default: throw ArrowMetalError.unsupportedType("\(op.rawValue) has no checked logarithm form")
        }
        return try context.batch {
            let out = try unaryMath(op)
            try checkUnary("chk_unary_log", checked)
            return out
        }
    }

    public func lnChecked() throws -> MetalArray<T> { try logChecked(.ln) }
    public func log2Checked() throws -> MetalArray<T> { try logChecked(.log2) }
    public func log10Checked() throws -> MetalArray<T> { try logChecked(.log10) }

    /// `log1p_checked`: `-1` raises "logarithm of zero" and anything below it "logarithm of negative number".
    public func log1pChecked() throws -> MetalArray<T> {
        try context.batch {
            let out = try log1p()
            try checkUnary("chk_unary_log1p", .log1p)
            return out
        }
    }

    /// `logb_checked(base)`: both the value and the base must be strictly positive.
    public func logbChecked(_ base: T) throws -> MetalArray<T> {
        try context.batch {
            let out = try logb(base)
            try checkScalar("chk_scalar_logb", .logb, base)
            return out
        }
    }

    /// `logb_checked` against a column of bases.
    public func logbChecked(_ base: MetalArray<T>) throws -> MetalArray<T> {
        try context.batch {
            let out = try logb(base)
            try checkArray("chk_array_logb", .logb, base)
            return out
        }
    }

    // MARK: - Checked power and shifts

    /// `power_checked`. On an integer column a negative exponent raises ("integers to negative integer
    /// powers are not allowed") and so does any repeated-squaring step that would wrap. Float columns
    /// never raise (Arrow lets `power_checked` overflow to infinity).
    public func powerChecked(_ s: T) throws -> MetalArray<T> {
        try context.batch {
            let out = try power(s)
            if Self.needsCheck(.power) { try checkScalar("chk_scalar_power", .power, s) }
            return out
        }
    }

    public func powerChecked(_ o: MetalArray<T>) throws -> MetalArray<T> {
        try context.batch {
            let out = try power(o)
            if Self.needsCheck(.power) { try checkArray("chk_array_power", .power, o) }
            return out
        }
    }

    /// `shift_left_checked` / `shift_right_checked`. Arrow raises when the amount is negative or at least
    /// the *precision* of the type — the bit width for an unsigned column, one less for a signed one, so
    /// `shift_left_checked(int64 1, 63)` raises even though `1 << 63` fits an `uint64`. Bits shifted off
    /// the top are not an error, in Arrow or here.
    public func shiftChecked(_ op: BitwiseOp, _ s: T) throws -> MetalArray<T> {
        let checked = try Self.checkedShiftOp(op)
        return try context.batch {
            let out = try bitwise(op, s)
            try checkScalar("chk_scalar_\(op.rawValue)", checked, s)
            return out
        }
    }

    public func shiftChecked(_ op: BitwiseOp, _ o: MetalArray<T>) throws -> MetalArray<T> {
        let checked = try Self.checkedShiftOp(op)
        return try context.batch {
            let out = try bitwise(op, o)
            try checkArray("chk_array_\(op.rawValue)", checked, o)
            return out
        }
    }

    private static func checkedShiftOp(_ op: BitwiseOp) throws -> CheckedOp {
        switch op {
        case .shl: return .shiftLeft
        case .shr: return .shiftRight
        case .and, .or, .xor:
            throw ArrowMetalError.unsupportedType("bit_wise_\(op.rawValue) has no checked form")
        }
    }

    public func shiftLeftChecked(_ s: T) throws -> MetalArray<T> { try shiftChecked(.shl, s) }
    public func shiftLeftChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try shiftChecked(.shl, o) }
    public func shiftRightChecked(_ s: T) throws -> MetalArray<T> { try shiftChecked(.shr, s) }
    public func shiftRightChecked(_ o: MetalArray<T>) throws -> MetalArray<T> { try shiftChecked(.shr, o) }

    // MARK: - Checked cumulative and pairwise

    /// `cumulative_sum_checked`. The scan itself reassociates, so the check is a second pass over the
    /// finished running values: `out[i]` must be `out[i - 1]` plus `vals[i]` without wrapping, which is
    /// exactly the sequential recurrence Arrow evaluates. Reporting the smallest failing row therefore
    /// names the same element Arrow would stop at.
    public func cumulativeSumChecked() throws -> MetalArray<T> {
        try checkedScan(.cumulativeSum, "chk_cum_add") { try self.cumulative(.sum) }
    }

    /// `cumulative_prod_checked`, verified the same way with the multiplication check.
    public func cumulativeProdChecked() throws -> MetalArray<T> {
        try checkedScan(.cumulativeProd, "chk_cum_mul") { try self.cumulativeProd() }
    }

    private func checkedScan(_ op: CheckedOp, _ function: String,
                             _ compute: () throws -> MetalArray<T>) throws -> MetalArray<T> {
        try context.batch {
            let out = try compute()
            guard Self.needsCheck(op), out.knownLength > 1 else { return out }
            let ctx = context
            let n = out.knownLength
            let pso = try checkPipeline(function)
            let flags = try CheckedFlags.makeBuffer(ctx)
            let v = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(out.values.mtl, offset: out.values.offset, index: 1)
                enc.setBuffer(v.mtl, offset: v.offset, index: 2)
                Dispatch.setLength(enc, n, nil, index: 3)
                Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 4)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)   // four elements per thread
            }
            try registerCheck(flags, op, retaining: [self, out])
            return out
        }
    }

    /// `pairwise_diff_checked`: `out[i] = self[i] - self[i - period]`, raising where that subtraction
    /// would wrap. A row whose partner falls outside the array, or where either side is null, is null in
    /// the output and is never checked.
    public func pairwiseDiffChecked(period: Int = 1) throws -> MetalArray<T> {
        try context.batch {
            let out = try pairwiseDiff(period: period)
            guard Self.needsCheck(.pairwiseDiff), out.knownLength > 0 else { return out }
            let ctx = context
            let n = out.knownLength
            let pso = try checkPipeline("chk_pairwise")
            let flags = try CheckedFlags.makeBuffer(ctx)
            let v = validity ?? values
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(v.mtl, offset: v.offset, index: 1)
                Dispatch.setLength(enc, n, nil, index: 2)
                Dispatch.setScalar(enc, Int32(clamping: period), index: 3)
                Dispatch.setUInt(enc, validity != nil ? 1 : 0, index: 4)
                enc.setBuffer(flags.mtl, offset: flags.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: (n + 3) / 4)   // four elements per thread
            }
            try registerCheck(flags, .pairwiseDiff, retaining: [self, out])
            return out
        }
    }
}
