import Foundation

// Lowering an expression DAG to Metal Shading Language.
//
// The emitter walks the tree once, bottom up, and appends straight-line MSL statements to one
// `inline void am_row(...)` function. Common subexpressions are keyed by their canonical text, so a
// subtree that appears twice is evaluated once. Validity is compiled in as ordinary boolean registers
// next to the values, and a subexpression that cannot be null carries the literal `true` instead of a
// register, which folds away at compile time.

/// Type and nullability of one input column.
struct ExprColumnInfo {
    var type: ExprType
    var nullable: Bool
}

/// A value plus its validity, as MSL expressions.
struct ExprSlot {
    var type: ExprType
    var v: String
    var ok: String
    /// utf8 slots carry buffer accessors instead of a value.
    var strData: String? = nil
    var strBeg: String? = nil
    var strEnd: String? = nil
    var alwaysValid: Bool { ok == "true" }
}

/// Errors from type checking and lowering. Every one names the offending node.
enum ExprError {
    static func unsupported(_ what: String) -> ArrowMetalError { .unsupportedType("expression: \(what)") }
    static func invalid(_ what: String) -> ArrowMetalError { .invalidArrowArray("expression: \(what)") }
}

final class ExprEmitter {
    struct Leaf {
        var name: String
        var type: ExprType
        var nullable: Bool
        var index: Int
    }

    let schema: [String: ExprColumnInfo]
    /// Prefix for file-scope names this emitter generates, so two emitters in one kernel source
    /// (the predicate pass and the projection pass of a filtered project) cannot collide.
    let prefix: String
    private(set) var leaves: [Leaf] = []
    private var leafByName: [String: Int] = [:]
    private(set) var patterns: [[UInt8]] = []
    var body: [String] = []
    var usesDoubleMath = false
    var usesTranscendental = false
    private var cache: [String: ExprSlot] = [:]
    private var convCache: [String: ExprSlot] = [:]
    private var tmp = 0
    private var typeMemo: [String: ExprType?] = [:]

    init(schema: [String: ExprColumnInfo], prefix: String = "") { self.schema = schema; self.prefix = prefix }

    private func next() -> String { tmp += 1; return "t\(tmp)" }

    /// Declares a temporary for `value` (and one for `ok` unless it is statically true).
    private func define(_ type: ExprType, _ value: String, _ ok: String) -> ExprSlot {
        let n = next()
        body.append("\(type.msl) \(n) = \(value);")
        if ok == "true" { return ExprSlot(type: type, v: n, ok: "true") }
        body.append("bool \(n)k = \(ok);")
        return ExprSlot(type: type, v: n, ok: "\(n)k")
    }

    // MARK: type checking

    /// The type of `e`, or nil when it is an untyped literal that adapts to its context.
    func typeOf(_ e: Expr) throws -> ExprType? {
        let key = e.description
        if let t = typeMemo[key] { return t }
        let t = try computeType(e)
        typeMemo[key] = t
        return t
    }

    private func computeType(_ e: Expr) throws -> ExprType? {
        switch e {
        case .column(let n):
            guard let c = schema[n] else { throw ExprError.invalid("no column named \"\(n)\"") }
            return c.type
        case .int, .double: return nil
        case .typedInt(_, let t), .typedDouble(_, let t), .nullLiteral(let t): return t
        case .bool: return .boolean
        case .string: return .utf8
        case .binary(let op, let a, let b):
            if op.isComparison || op.isLogical { return .boolean }
            let ta = try typeOf(a), tb = try typeOf(b)
            if ta == nil && tb == nil { return nil }
            return try promote(ta ?? tb!, tb ?? ta!, op: op.rawValue)
        case .unary(let op, let a):
            switch op {
            case .not: return .boolean
            case .sqrt, .exp, .ln:
                guard let t = try typeOf(a) else { return .float64 }
                return t == .float32 ? .float32 : .float64
            default: return try typeOf(a)
            }
        case .cast(_, let t): return t
        case .ifElse(_, let a, let b):
            let ta = try typeOf(a), tb = try typeOf(b)
            if ta == nil && tb == nil { return nil }
            if ta == nil || tb == nil { return ta ?? tb }
            return try promote(ta!, tb!, op: "if_else")
        case .coalesce(let xs):
            var t: ExprType? = nil
            for x in xs { if let xt = try typeOf(x) { t = t == nil ? xt : try promote(t!, xt, op: "coalesce") } }
            return t
        case .fillNull(let a, let b):
            let ta = try typeOf(a), tb = try typeOf(b)
            if ta == nil && tb == nil { return nil }
            if ta == nil || tb == nil { return ta ?? tb }
            return try promote(ta!, tb!, op: "fill_null")
        case .isNull, .isValid, .isIn, .stringMatch: return .boolean
        }
    }

    /// Arrow's implicit numeric promotion, documented in docs/EXPR.md and checked against
    /// pyarrow.compute: float64 wins over everything, float32 over any integer, integers widen to the
    /// wider of the two, and a mixed-signedness pair goes to a signed type twice the unsigned width
    /// (capped at int64, which is also where uint64 with a signed operand lands).
    func promote(_ a: ExprType, _ b: ExprType, op: String) throws -> ExprType {
        if a == b { return a }
        if a == .boolean || b == .boolean || a == .utf8 || b == .utf8 {
            throw ExprError.unsupported("\(op) has no common type for \(a.rawValue) and \(b.rawValue)")
        }
        if a == .float64 || b == .float64 { return .float64 }
        if a == .float32 || b == .float32 { return .float32 }
        if a.isSigned == b.isSigned { return a.bitWidth >= b.bitWidth ? a : b }
        let s = a.isSigned ? a : b
        let u = a.isSigned ? b : a
        let need = Swift.min(64, Swift.max(s.bitWidth, u.bitWidth * 2))
        switch need {
        case 8: return .int8
        case 16: return .int16
        case 32: return .int32
        default: return .int64
        }
    }

    // MARK: lowering

    func emit(_ e: Expr, hint: ExprType? = nil) throws -> ExprSlot {
        let t = try typeOf(e)
        let key = "\(e)|\(t?.rawValue ?? hint?.rawValue ?? "-")"
        if let s = cache[key] { return s }
        let s = try build(e, hint: hint)
        cache[key] = s
        return s
    }

    /// Emits `e` and converts it to `target`.
    func emit(_ e: Expr, as target: ExprType) throws -> ExprSlot {
        let s = try emit(e, hint: target)
        return try convert(s, to: target)
    }

    private func leaf(_ name: String) throws -> Leaf {
        if let i = leafByName[name] { return leaves[i] }
        guard let c = schema[name] else { throw ExprError.invalid("no column named \"\(name)\"") }
        let l = Leaf(name: name, type: c.type, nullable: c.nullable, index: leaves.count)
        leafByName[name] = leaves.count
        leaves.append(l)
        return l
    }

    private func build(_ e: Expr, hint: ExprType?) throws -> ExprSlot {
        switch e {
        case .column(let n):
            let l = try leaf(n)
            let ok = l.nullable ? "L\(l.index)k" : "true"
            if l.type == .utf8 {
                return ExprSlot(type: .utf8, v: "", ok: ok,
                                strData: "L\(l.index)d", strBeg: "L\(l.index)b", strEnd: "L\(l.index)e")
            }
            return ExprSlot(type: l.type, v: "L\(l.index)v", ok: ok)

        case .int(let v):
            let t = (hint?.isNumeric == true) ? hint! : .int64
            return ExprSlot(type: t, v: try numericLiteral(Double(v), int: v, type: t), ok: "true")
        case .typedInt(let v, let t):
            return ExprSlot(type: t, v: try numericLiteral(Double(v), int: v, type: t), ok: "true")
        case .double(let v):
            let t: ExprType = (hint == .float32) ? .float32 : .float64
            return ExprSlot(type: t, v: try numericLiteral(v, int: Int64(v.isFinite ? v : 0), type: t), ok: "true")
        case .typedDouble(let v, let t):
            return ExprSlot(type: t, v: try numericLiteral(v, int: Int64(v.isFinite ? v : 0), type: t), ok: "true")
        case .bool(let v):
            return ExprSlot(type: .boolean, v: v ? "true" : "false", ok: "true")
        case .string:
            throw ExprError.unsupported("a string literal is only allowed as the pattern of str_eq / starts_with / contains")
        case .nullLiteral(let t):
            if t == .utf8 { throw ExprError.unsupported("null literal of type utf8") }
            return ExprSlot(type: t, v: zeroLiteral(t), ok: "false")

        case .binary(let op, let a, let b): return try buildBinary(op, a, b, hint: hint)
        case .unary(let op, let a): return try buildUnary(op, a, hint: hint)

        case .cast(let a, let t):
            let s = try emit(a, hint: t)
            return try convert(s, to: t, explicit: true)

        case .ifElse(let c, let a, let b):
            let cs = try emit(c, hint: .boolean)
            guard cs.type == .boolean else { throw ExprError.unsupported("if_else condition must be boolean, got \(cs.type.rawValue)") }
            let t = try typeOf(e) ?? hint ?? .int64
            let av = try emit(a, as: t), bv = try emit(b, as: t)
            let ok = cs.alwaysValid && av.alwaysValid && bv.alwaysValid
                ? "true"
                : "(\(cs.ok) && (\(cs.v) ? \(av.ok) : \(bv.ok)))"
            return define(t, "(\(cs.v) ? \(av.v) : \(bv.v))", ok)

        case .coalesce(let xs):
            guard !xs.isEmpty else { throw ExprError.invalid("coalesce needs at least one argument") }
            let t = try typeOf(e) ?? hint ?? .int64
            let slots = try xs.map { try emit($0, as: t) }
            var value = slots.last!.v
            for s in slots.dropLast().reversed() { value = "(\(s.ok) ? \(s.v) : \(value))" }
            let ok = slots.contains { $0.alwaysValid } ? "true"
                                                       : "(" + slots.map(\.ok).joined(separator: " || ") + ")"
            return define(t, value, ok)

        case .fillNull(let a, let b):
            let t = try typeOf(e) ?? hint ?? .int64
            let av = try emit(a, as: t), bv = try emit(b, as: t)
            if av.alwaysValid { return av }
            return define(t, "(\(av.ok) ? \(av.v) : \(bv.v))", bv.alwaysValid ? "true" : "(\(av.ok) || \(bv.ok))")

        case .isNull(let a):
            let s = try emit(a)
            return define(.boolean, s.alwaysValid ? "false" : "(!\(s.ok))", "true")
        case .isValid(let a):
            let s = try emit(a)
            return define(.boolean, s.alwaysValid ? "true" : s.ok, "true")

        case .isIn(let a, let set):
            guard !set.isEmpty else { return define(.boolean, "false", "true") }
            let s = try emit(a)
            guard s.type.isNumeric || s.type == .boolean else {
                throw ExprError.unsupported("is_in needs a numeric or boolean column, got \(s.type.rawValue)")
            }
            var tests: [String] = []
            for lit in set {
                let l = try emit(lit, as: s.type)
                guard l.alwaysValid else { throw ExprError.unsupported("is_in value set must not contain nulls") }
                tests.append(try equality(s.type, s.v, l.v))
            }
            let any = "(" + tests.joined(separator: " || ") + ")"
            return define(.boolean, s.alwaysValid ? any : "(\(s.ok) && \(any))", "true")

        case .stringMatch(let pred, let a, let pattern):
            let s = try emit(a)
            guard s.type == .utf8, let d = s.strData, let b0 = s.strBeg, let e0 = s.strEnd else {
                throw ExprError.unsupported("\(pred.rawValue) needs a utf8 column reference, got \(s.type.rawValue)")
            }
            let bytes = Array(pattern.utf8)
            let pid = patterns.count
            patterns.append(bytes)
            let op = pred == .equals ? 0 : (pred == .startsWith ? 1 : 2)
            return define(.boolean, "am_str_match(\(d), \(b0), \(e0), am_pat\(prefix)\(pid), \(bytes.count)u, \(op)u)", s.ok)
        }
    }

    private func buildBinary(_ op: ExprBinaryOp, _ a: Expr, _ b: Expr, hint: ExprType?) throws -> ExprSlot {
        if op.isLogical {
            let av = try emit(a, hint: .boolean), bv = try emit(b, hint: .boolean)
            guard av.type == .boolean, bv.type == .boolean else {
                throw ExprError.unsupported("\(op.rawValue) needs boolean operands, got \(av.type.rawValue) and \(bv.type.rawValue)")
            }
            let isAnd = op == .and || op == .andKleene
            let value = isAnd ? "(\(av.v) && \(bv.v))" : "(\(av.v) || \(bv.v))"
            if av.alwaysValid && bv.alwaysValid { return define(.boolean, value, "true") }
            if op.isKleene {
                // Kleene: and is false as soon as one side is a valid false, or is true as soon as one
                // side is a valid true; otherwise a null on either side makes the result null.
                let short = isAnd ? "((\(av.ok) && !\(av.v)) || (\(bv.ok) && !\(bv.v)))"
                                  : "((\(av.ok) && \(av.v)) || (\(bv.ok) && \(bv.v)))"
                let ok = "((\(av.ok) && \(bv.ok)) || \(short))"
                let val = isAnd ? "(\(av.ok) && \(av.v) && \(bv.ok) && \(bv.v))"
                                : "((\(av.ok) && \(av.v)) || (\(bv.ok) && \(bv.v)))"
                return define(.boolean, val, ok)
            }
            return define(.boolean, value, "(\(av.ok) && \(bv.ok))")
        }

        let ta = try typeOf(a), tb = try typeOf(b)
        var t: ExprType
        if ta == nil && tb == nil { t = hint?.isNumeric == true ? hint! : .int64 }
        else if ta == nil { t = tb! } else if tb == nil { t = ta! }
        else { t = try promote(ta!, tb!, op: op.rawValue) }
        if op.isBitwise && !t.isInteger {
            throw ExprError.unsupported("\(op.rawValue) needs integer operands, got \(t.rawValue)")
        }
        if (op.isArithmetic || op.isComparison) && !t.isNumeric {
            throw ExprError.unsupported("\(op.rawValue) needs numeric operands, got \(t.rawValue)")
        }
        let av = try emit(a, as: t), bv = try emit(b, as: t)
        let ok = (av.alwaysValid && bv.alwaysValid) ? "true" : "(\(av.ok) && \(bv.ok))"
        if op.isComparison { return define(.boolean, try comparison(op, t, av.v, bv.v), ok) }
        return define(t, try arithmetic(op, t, av.v, bv.v), ok)
    }

    private func buildUnary(_ op: ExprUnaryOp, _ a: Expr, hint: ExprType?) throws -> ExprSlot {
        if op == .not {
            let s = try emit(a, hint: .boolean)
            guard s.type == .boolean else { throw ExprError.unsupported("not needs a boolean operand, got \(s.type.rawValue)") }
            return define(.boolean, "(!\(s.v))", s.ok)
        }
        if op == .sqrt || op == .exp || op == .ln {
            var s = try emit(a, hint: hint ?? .float64)
            guard s.type.isNumeric else { throw ExprError.unsupported("\(op.rawValue) needs a numeric operand, got \(s.type.rawValue)") }
            if s.type.isInteger { s = try convert(s, to: .float64) }   // Arrow: integer in, float64 out
            if s.type == .float32 {
                let f = op == .sqrt ? "sqrt" : (op == .exp ? "exp" : "log")
                return define(.float32, "\(f)(\(s.v))", s.ok)
            }
            usesDoubleMath = true; usesTranscendental = true
            let call: String
            switch op {
            case .sqrt: call = "dt_sqrt(\(s.v))"
            case .exp: call = "d_add(dt_expm1(\(s.v)), 0x3FF0000000000000ul)"
            default: call = "dt_ln(\(s.v))"
            }
            return define(.float64, call, s.ok)
        }
        let s = try emit(a, hint: hint)
        guard s.type.isNumeric else { throw ExprError.unsupported("\(op.rawValue) needs a numeric operand, got \(s.type.rawValue)") }
        switch op {
        case .negate:
            switch s.type {
            case .float32: return define(.float32, "(-\(s.v))", s.ok)
            case .float64: return define(.float64, "(\(s.v) ^ 0x8000000000000000ul)", s.ok)
            default: return define(s.type, "(\(s.type.msl))(0 - (\(s.type.msl))\(s.v))", s.ok)
            }
        case .abs:
            switch s.type {
            case .float32: return define(.float32, "fabs(\(s.v))", s.ok)
            case .float64: return define(.float64, "(\(s.v) & 0x7FFFFFFFFFFFFFFFul)", s.ok)
            default:
                if !s.type.isSigned { return s }
                return define(s.type, "(\(s.v) < 0 ? (\(s.type.msl))(0 - \(s.v)) : \(s.v))", s.ok)
            }
        case .bitNot:
            guard s.type.isInteger else { throw ExprError.unsupported("bit_not needs an integer operand, got \(s.type.rawValue)") }
            return define(s.type, "(\(s.type.msl))(~\(s.v))", s.ok)
        case .round:
            switch s.type {
            case .float32: return define(.float32, "round(\(s.v))", s.ok)
            case .float64: usesDoubleMath = true; return define(.float64, "d_round(\(s.v))", s.ok)
            default: return s
            }
        default:
            throw ExprError.unsupported("unary \(op.rawValue)")
        }
    }

    // MARK: operator code

    private func arithmetic(_ op: ExprBinaryOp, _ t: ExprType, _ a: String, _ b: String) throws -> String {
        if t == .float64 {
            usesDoubleMath = true
            switch op {
            case .add: return "d_add(\(a), \(b))"
            case .sub: return "d_sub(\(a), \(b))"
            case .mul: return "d_mul(\(a), \(b))"
            case .div: return "d_div(\(a), \(b))"
            default: break
            }
        }
        switch op {
        case .add: return "(\(t.msl))(\(a) + \(b))"
        case .sub: return "(\(t.msl))(\(a) - \(b))"
        case .mul: return "(\(t.msl))(\(a) * \(b))"
        case .div:
            if t.isFloat { return "(\(a) / \(b))" }
            // Integer division by zero is defined as 0 here, matching ArrowMetal's existing kernels.
            return "((\(b) == (\(t.msl))0) ? (\(t.msl))0 : (\(t.msl))(\(a) / \(b)))"
        case .bitAnd: return "(\(t.msl))(\(a) & \(b))"
        case .bitOr: return "(\(t.msl))(\(a) | \(b))"
        case .bitXor: return "(\(t.msl))(\(a) ^ \(b))"
        case .shl, .shr:
            // An amount outside [0, width) gives 0 (Arrow leaves the unchecked shifts implementation defined).
            let w = t.bitWidth
            let sh = op == .shl ? "<<" : ">>"
            return "(((long)(\(b)) < 0L || (long)(\(b)) >= \(w)L) ? (\(t.msl))0 : (\(t.msl))(\(a) \(sh) \(b)))"
        default: throw ExprError.unsupported("arithmetic \(op.rawValue)")
        }
    }

    private func equality(_ t: ExprType, _ a: String, _ b: String) throws -> String {
        try comparison(.eq, t, a, b)
    }

    private func comparison(_ op: ExprBinaryOp, _ t: ExprType, _ a: String, _ b: String) throws -> String {
        let sym: String
        switch op {
        case .eq: sym = "=="; case .ne: sym = "!="; case .lt: sym = "<"
        case .le: sym = "<="; case .gt: sym = ">"; default: sym = ">="
        }
        switch t {
        case .float64:
            usesDoubleMath = true
            return "d_cmp_\(op.rawValue)(\(a), \(b))"
        case .float32:
            // Through order-preserving integer keys: exact on subnormals, which Apple GPUs flush to zero.
            return "f_cmp_\(op.rawValue)(\(a), \(b))"
        default:
            return "(\(a) \(sym) \(b))"
        }
    }

    /// Implicit (and explicit `cast`) numeric conversion.
    func convert(_ s: ExprSlot, to t: ExprType, explicit: Bool = false) throws -> ExprSlot {
        if s.type == t { return s }
        let ckey = "\(s.v)|\(s.ok)|\(s.type.rawValue)->\(t.rawValue)"
        if let c = convCache[ckey] { return c }
        let r = try convertUncached(s, to: t, explicit: explicit)
        convCache[ckey] = r
        return r
    }

    private func convertUncached(_ s: ExprSlot, to t: ExprType, explicit: Bool) throws -> ExprSlot {
        if s.type == .utf8 || t == .utf8 { throw ExprError.unsupported("cast between utf8 and \(t.rawValue)") }
        if s.type == .boolean {
            guard t.isNumeric else { throw ExprError.unsupported("cast boolean to \(t.rawValue)") }
            if t == .float64 { usesDoubleMath = true; return define(t, "(\(s.v) ? 0x3FF0000000000000ul : 0ul)", s.ok) }
            return define(t, "(\(t.msl))(\(s.v) ? 1 : 0)", s.ok)
        }
        if t == .boolean {
            guard explicit else { throw ExprError.unsupported("cast \(s.type.rawValue) to boolean must be explicit") }
            if s.type == .float64 { usesDoubleMath = true; return define(t, "((\(s.v) & 0x7FFFFFFFFFFFFFFFul) != 0ul)", s.ok) }
            return define(t, "(\(s.v) != (\(s.type.msl))0)", s.ok)
        }
        usesDoubleMath = usesDoubleMath || s.type == .float64 || t == .float64
        switch (s.type, t) {
        case (.float64, .float32): return define(t, "d_to_float(\(s.v))", s.ok)
        case (.float64, _) where t.isInteger:
            return define(t, "(\(t.msl))d_to_long(\(s.v))", s.ok)
        case (.float32, .float64): return define(t, "d_from_float(\(s.v))", s.ok)
        case (_, .float64) where s.type.isInteger:
            return define(t, s.type.isSigned ? "d_from_long((long)\(s.v))" : "d_from_ulong((ulong)\(s.v))", s.ok)
        default:
            return define(t, "(\(t.msl))\(s.v)", s.ok)
        }
    }

    // MARK: literals

    private func zeroLiteral(_ t: ExprType) -> String {
        switch t {
        case .boolean: return "false"
        case .float32: return "0.0f"
        case .float64: return "0ul"
        default: return "(\(t.msl))0"
        }
    }

    private func numericLiteral(_ d: Double, int: Int64, type: ExprType) throws -> String {
        switch type {
        case .float32:
            return "as_type<float>(\(String(format: "0x%08Xu", Float(d).bitPattern)))"
        case .float64:
            usesDoubleMath = true
            return String(format: "0x%016llXul", d.bitPattern)
        case .boolean:
            throw ExprError.unsupported("numeric literal where a boolean is expected")
        case .utf8:
            throw ExprError.unsupported("numeric literal where a string is expected")
        default:
            // Integer literal: truncated to the target width the way an Arrow cast would.
            _ = d
            if type.isSigned {
                if int == Int64.min { return "(\(type.msl))(-9223372036854775807L - 1L)" }
                return "(\(type.msl))(\(int)L)"
            }
            return "(\(type.msl))(\(UInt64(bitPattern: int))ul)"
        }
    }

    // MARK: assembling the row function

    /// Builds `inline void <name>(...)` from the statements emitted so far plus the given outputs.
    func rowFunction(name: String, outputs: [ExprSlot]) -> String {
        var params: [String] = []
        for l in leaves {
            if l.type == .utf8 {
                params.append("device const uchar* L\(l.index)d")
                params.append("int L\(l.index)b")
                params.append("int L\(l.index)e")
            } else {
                params.append("\(bufferElementType(l)) L\(l.index)v")
            }
            if l.nullable { params.append("bool L\(l.index)k") }
        }
        for (k, o) in outputs.enumerated() {
            params.append("thread \(o.type.msl)& O\(k)")
            params.append("thread bool& O\(k)k")
        }
        var s = "inline void \(name)(" + params.joined(separator: ", ") + ") {\n"
        for line in body { s += "    " + line + "\n" }
        for (k, o) in outputs.enumerated() {
            s += "    O\(k) = \(o.v);\n"
            s += "    O\(k)k = \(o.ok);\n"
        }
        return s + "}\n"
    }

    /// Argument list for a call to the row function: the leaf accessors plus the output variables.
    func callArguments(value: (Leaf) -> String, valid: (Leaf) -> String,
                       strBegin: (Leaf) -> String, strEnd: (Leaf) -> String,
                       outputCount: Int) -> String {
        var args: [String] = []
        for l in leaves {
            if l.type == .utf8 {
                args.append("LD\(l.index)")
                args.append(strBegin(l))
                args.append(strEnd(l))
            } else {
                args.append(value(l))
            }
            if l.nullable { args.append(valid(l)) }
        }
        for k in 0..<outputCount { args.append("O\(k)"); args.append("O\(k)k") }
        return args.joined(separator: ", ")
    }

    /// MSL element type of a leaf's values buffer (booleans are read from a bitmap, so they load as uint words).
    func bufferElementType(_ l: Leaf) -> String { l.type == .boolean ? "bool" : l.type.msl }

    /// `constant` arrays for the string patterns used by the tree.
    var patternConstants: String {
        var s = ""
        for (i, p) in patterns.enumerated() {
            let bytes = p.isEmpty ? "0" : p.map { "\($0)" }.joined(separator: ", ")
            s += "constant uchar am_pat\(prefix)\(i)[] = { \(bytes) };\n"
        }
        return s
    }
}

// MARK: - Shared MSL helpers

enum ExprSource {
    /// Helpers every generated kernel gets: bitmap word loads, exact float32 comparisons, float64
    /// comparisons and conversions, and the literal string matcher.
    static let helpers = """

    // One validity/boolean word (32 rows) from a byte-addressed bitmap. Buffers are page padded, so the
    // whole trailing word is always readable.
    inline uint am_vword(device const uchar* p, uint w) {
        uint b = w * 4u;
        return (uint)p[b] | ((uint)p[b + 1] << 8) | ((uint)p[b + 2] << 16) | ((uint)p[b + 3] << 24);
    }
    inline bool f_isnan32(uint b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
    inline int f_key32(uint b) { if ((b & 0x7FFFFFFFu) == 0u) return 0; int k = (int)b; return k ^ (int)(((uint)(k >> 31)) >> 1); }
    inline bool f_cmp_eq(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return false; return f_key32(x) == f_key32(y); }
    inline bool f_cmp_ne(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return true;  return f_key32(x) != f_key32(y); }
    inline bool f_cmp_lt(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return false; return f_key32(x) <  f_key32(y); }
    inline bool f_cmp_le(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return false; return f_key32(x) <= f_key32(y); }
    inline bool f_cmp_gt(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return false; return f_key32(x) >  f_key32(y); }
    inline bool f_cmp_ge(float a, float b) { uint x = as_type<uint>(a), y = as_type<uint>(b); if (f_isnan32(x) || f_isnan32(y)) return false; return f_key32(x) >= f_key32(y); }
    // Literal pattern match. op: 0 equals, 1 starts_with, 2 contains.
    inline bool am_str_match(device const uchar* data, int b, int e, constant uchar* pat, uint plen, uint op) {
        uint len = (uint)(e - b);
        if (op == 0u) { if (len != plen) return false; for (uint j = 0; j < plen; j++) if (data[b + j] != pat[j]) return false; return true; }
        if (plen > len) return false;
        if (op == 1u) { for (uint j = 0; j < plen; j++) if (data[b + j] != pat[j]) return false; return true; }
        if (plen == 0u) return true;
        for (uint s = 0; s + plen <= len; s++) {
            bool ok = true;
            for (uint j = 0; j < plen && ok; j++) if (data[b + s + j] != pat[j]) ok = false;
            if (ok) return true;
        }
        return false;
    }
    """

    /// Float64 comparisons, conversions and rounding on raw bit patterns. Needs DoubleMath.
    static let doubleHelpers = """

    inline bool d_cmp_eq(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return false; return d_key((long)a) == d_key((long)b); }
    inline bool d_cmp_ne(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return true;  return d_key((long)a) != d_key((long)b); }
    inline bool d_cmp_lt(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return false; return d_key((long)a) <  d_key((long)b); }
    inline bool d_cmp_le(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return false; return d_key((long)a) <= d_key((long)b); }
    inline bool d_cmp_gt(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return false; return d_key((long)a) >  d_key((long)b); }
    inline bool d_cmp_ge(ulong a, ulong b) { if (d_is_nan(a) || d_is_nan(b)) return false; return d_key((long)a) >= d_key((long)b); }
    // Exact widening of an unsigned/signed 64-bit integer to binary64 (round to nearest, ties to even).
    inline ulong d_from_ulong(ulong u) {
        if (u == 0ul) return 0ul;
        int e = 63; while (((u >> e) & 1ul) == 0ul) e--;
        ulong m;
        if (e <= 55) m = u << (55 - e);
        else { int sh = e - 55; ulong st = (u & ((1ul << sh) - 1ul)) ? 1ul : 0ul; m = (u >> sh) | st; }
        return d_finish(0ul, (long)e + 1023L, m);
    }
    inline ulong d_from_long(long v) {
        if (v == 0L) return 0ul;
        bool neg = v < 0L;
        ulong u = neg ? (ulong)(0L - v) : (ulong)v;
        ulong r = d_from_ulong(u);
        return neg ? (r | 0x8000000000000000ul) : r;
    }
    // Truncation toward zero. Infinities and NaN are unspecified, as in Arrow's unchecked cast.
    inline long d_to_long(ulong b) {
        ulong s = b >> 63;
        long e = (long)((b >> 52) & 0x7FFul);
        ulong m = b & 0xFFFFFFFFFFFFFul;
        if (e == 0x7FFL) return s ? LONG_MIN : LONG_MAX;
        if (e < 1023L) return 0L;
        long sh = e - 1023L;
        if (sh > 62L) return s ? LONG_MIN : LONG_MAX;
        ulong v = m | (1ul << 52);
        if (sh >= 52L) v <<= (ulong)(sh - 52L); else v >>= (ulong)(52L - sh);
        return s ? -(long)v : (long)v;
    }
    // Narrowing to float32, round to nearest with ties to even.
    inline float d_to_float(ulong b) {
        ulong s = b >> 63;
        long e = (long)((b >> 52) & 0x7FFul);
        ulong m = b & 0xFFFFFFFFFFFFFul;
        uint sgn = (uint)(s << 31);
        if (e == 0x7FFL) return as_type<float>(sgn | 0x7F800000u | (m ? 0x400000u : 0u));
        if (e == 0L) return as_type<float>(sgn);              // zero or a double subnormal: underflows
        long ue = e - 1023L;
        if (ue > 127L) return as_type<float>(sgn | 0x7F800000u);
        ulong sig = m | (1ul << 52);
        long fe = ue + 127L;
        int shift = (fe >= 1L) ? 29 : (int)(30L - fe);
        if (shift > 60) return as_type<float>(sgn);
        ulong keep = sig >> shift;
        ulong rest = sig & ((1ul << shift) - 1ul);
        ulong hbit = 1ul << (shift - 1);
        if (rest > hbit || (rest == hbit && (keep & 1ul))) keep++;
        if (fe >= 1L) {
            if (keep >> 24) { keep >>= 1; fe++; }
            if (fe >= 255L) return as_type<float>(sgn | 0x7F800000u);
            return as_type<float>(sgn | ((uint)fe << 23) | (uint)(keep & 0x7FFFFFul));
        }
        if (keep >> 23) return as_type<float>(sgn | (1u << 23));
        return as_type<float>(sgn | (uint)keep);
    }
    inline ulong d_trunc(ulong b) {
        long e = (long)((b >> 52) & 0x7FFul);
        if (e >= 1075L) return b;
        if (e < 1023L) return b & 0x8000000000000000ul;
        ulong mask = (1ul << (ulong)(1075L - e)) - 1ul;
        return b & ~mask;
    }
    // Halves away from zero, matching ArrowMetal's `round` unary op (Arrow's half_towards_infinity).
    inline ulong d_round(ulong b) {
        long e = (long)((b >> 52) & 0x7FFul);
        if (e >= 1075L) return b;
        ulong hf = 0x3FE0000000000000ul | (b & 0x8000000000000000ul);
        return d_trunc(d_add(b, hf));
    }
    """

    /// The exclusive scan over per-block selected counts (a copy of the filter pipeline's scan, which
    /// is type independent).
    static let scanKernel = """

    kernel void am_scan(device uint* blockCounts [[buffer(0)]],
                        device const uint* nPtr [[buffer(1)]],
                        device uint* total [[buffer(2)]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
        threadgroup uint simdTotals[32];
        uint blocks = max(1u, ((*nPtr + 31u) / 32u + TG - 1u) / TG);
        uint per = (blocks + TG - 1) / TG;
        uint lo = lid * per, hi = min(blocks, lo + per);
        uint local = 0;
        for (uint b = lo; b < hi; b++) local += blockCounts[b];
        uint pre = simd_prefix_exclusive_sum(local);
        uint t = simd_sum(local);
        if (lane == 0) simdTotals[sgid] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint prefix = 0;
        for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
        uint run = prefix + pre;
        for (uint b = lo; b < hi; b++) { uint c = blockCounts[b]; blockCounts[b] = run; run += c; }
        if (lid == TG - 1) *total = run;
    }
    """
}
