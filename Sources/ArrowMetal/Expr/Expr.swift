import Foundation

// ArrowMetal expression trees. See docs/EXPR.md for the grammar, the promotion rules and the limits.
//
// An `Expr` is a pure value: it names columns, literals and element-wise operators, and says nothing
// about how it runs. `ExprCompiler` lowers a whole tree (plus one terminal: project, filter, a set of
// reductions, or a group-by) into ONE runtime-generated Metal kernel, so a query touches each input
// byte once instead of once per operator.

/// The value types an expression node can have. `utf8` only appears on column references and string
/// literals, and only under the string predicates.
public enum ExprType: String, Hashable, Sendable, CaseIterable {
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
    case float32, float64
    case boolean
    case utf8

    public var isInteger: Bool {
        switch self {
        case .int8, .int16, .int32, .int64, .uint8, .uint16, .uint32, .uint64: return true
        default: return false
        }
    }
    public var isSigned: Bool {
        switch self { case .int8, .int16, .int32, .int64: return true; default: return false }
    }
    public var isFloat: Bool { self == .float32 || self == .float64 }
    public var isNumeric: Bool { isInteger || isFloat }
    /// Width in bits (0 for utf8, 1 for boolean).
    public var bitWidth: Int {
        switch self {
        case .int8, .uint8: return 8
        case .int16, .uint16: return 16
        case .int32, .uint32, .float32: return 32
        case .int64, .uint64, .float64: return 64
        case .boolean: return 1
        case .utf8: return 0
        }
    }
    /// The MSL type a value of this type is carried in. Float64 has no hardware support on Apple GPUs,
    /// so it travels as its raw 64-bit pattern in a `ulong` and goes through the software binary64 helpers.
    public var msl: String {
        switch self {
        case .int8: return "char"
        case .int16: return "short"
        case .int32: return "int"
        case .int64: return "long"
        case .uint8: return "uchar"
        case .uint16: return "ushort"
        case .uint32: return "uint"
        case .uint64: return "ulong"
        case .float32: return "float"
        case .float64: return "ulong"
        case .boolean: return "bool"
        case .utf8: return "void"
        }
    }
    /// Arrow C Data Interface format string.
    public var arrowFormat: String {
        switch self {
        case .int8: return "c"; case .int16: return "s"; case .int32: return "i"; case .int64: return "l"
        case .uint8: return "C"; case .uint16: return "S"; case .uint32: return "I"; case .uint64: return "L"
        case .float32: return "f"; case .float64: return "g"; case .boolean: return "b"; case .utf8: return "u"
        }
    }
    /// Short token used by the serialised (s-expression) form.
    public var token: String {
        switch self {
        case .int8: return "i8"; case .int16: return "i16"; case .int32: return "i32"; case .int64: return "i64"
        case .uint8: return "u8"; case .uint16: return "u16"; case .uint32: return "u32"; case .uint64: return "u64"
        case .float32: return "f32"; case .float64: return "f64"; case .boolean: return "bool"; case .utf8: return "str"
        }
    }
    public static func fromToken(_ s: String) -> ExprType? { ExprType.allCases.first { $0.token == s } }
}

/// Binary operators. Every one is element-wise.
public enum ExprBinaryOp: String, Hashable, Sendable {
    // arithmetic
    case add, sub, mul, div
    // comparison
    case eq, ne, lt, le, gt, ge
    // logical (null propagating) and Kleene
    case and, or, andKleene = "and_kleene", orKleene = "or_kleene"
    // bit
    case bitAnd = "bit_and", bitOr = "bit_or", bitXor = "bit_xor", shl, shr

    var isArithmetic: Bool { self == .add || self == .sub || self == .mul || self == .div }
    var isComparison: Bool {
        switch self { case .eq, .ne, .lt, .le, .gt, .ge: return true; default: return false }
    }
    var isLogical: Bool {
        switch self { case .and, .or, .andKleene, .orKleene: return true; default: return false }
    }
    var isKleene: Bool { self == .andKleene || self == .orKleene }
    var isBitwise: Bool {
        switch self { case .bitAnd, .bitOr, .bitXor, .shl, .shr: return true; default: return false }
    }
}

/// Unary operators.
public enum ExprUnaryOp: String, Hashable, Sendable {
    case negate, abs, sqrt, exp, ln, round
    case not
    case bitNot = "bit_not"
}

/// Literal string predicates: the pattern is a compile-time constant baked into the kernel.
public enum ExprStringPredicate: String, Hashable, Sendable {
    case equals = "str_eq", startsWith = "starts_with", contains
}

/// An element-wise expression over the columns of a record batch.
public indirect enum Expr: Hashable, Sendable {
    /// A column reference; its type comes from the batch.
    case column(String)
    /// An integer literal whose type adapts to the other operand (int64 when nothing constrains it).
    case int(Int64)
    /// An integer literal pinned to one Arrow type.
    case typedInt(Int64, ExprType)
    /// A floating literal whose type adapts to the other operand (float64 when nothing constrains it).
    case double(Double)
    /// A floating literal pinned to one Arrow type.
    case typedDouble(Double, ExprType)
    case bool(Bool)
    case string(String)
    /// A typed null literal.
    case nullLiteral(ExprType)

    case binary(ExprBinaryOp, Expr, Expr)
    case unary(ExprUnaryOp, Expr)
    case cast(Expr, ExprType)
    /// Arrow `if_else(cond, a, b)`: null cond gives null.
    case ifElse(Expr, Expr, Expr)
    /// Arrow `coalesce`: the first non-null of its arguments.
    case coalesce([Expr])
    /// Arrow `fill_null(a, b)`: `a` where valid, otherwise `b`.
    case fillNull(Expr, Expr)
    case isNull(Expr)
    case isValid(Expr)
    /// Arrow `is_in` against a small literal set. Always valid; a null input is `false`.
    case isIn(Expr, [Expr])
    /// A string predicate against a literal pattern.
    case stringMatch(ExprStringPredicate, Expr, String)

    // MARK: convenience

    public static func col(_ name: String) -> Expr { .column(name) }
    public static func lit(_ v: Int) -> Expr { .int(Int64(v)) }
    public static func lit(_ v: Int64) -> Expr { .int(v) }
    public static func lit(_ v: Double) -> Expr { .double(v) }
    public static func lit(_ v: Bool) -> Expr { .bool(v) }
    public static func lit(_ v: String) -> Expr { .string(v) }
}

/// Free function so `col("amount") > 100` reads the way it does in Polars and pyarrow.
public func col(_ name: String) -> Expr { .column(name) }
/// A typed null literal, for `fill_null` and `if_else` branches.
public func nullLit(_ t: ExprType) -> Expr { .nullLiteral(t) }

// Deliberately *not* ExpressibleBy*Literal: making `Expr` absorb bare literals would drag every
// numeric array literal in the package into overload resolution. Scalars get their own operator
// overloads below instead, so `col("amount") > 100` still reads naturally.

// MARK: - Operators

public func + (a: Expr, b: Expr) -> Expr { .binary(.add, a, b) }
public func - (a: Expr, b: Expr) -> Expr { .binary(.sub, a, b) }
public func * (a: Expr, b: Expr) -> Expr { .binary(.mul, a, b) }
public func / (a: Expr, b: Expr) -> Expr { .binary(.div, a, b) }
public func == (a: Expr, b: Expr) -> Expr { .binary(.eq, a, b) }
public func != (a: Expr, b: Expr) -> Expr { .binary(.ne, a, b) }
public func < (a: Expr, b: Expr) -> Expr { .binary(.lt, a, b) }
public func <= (a: Expr, b: Expr) -> Expr { .binary(.le, a, b) }
public func > (a: Expr, b: Expr) -> Expr { .binary(.gt, a, b) }
public func >= (a: Expr, b: Expr) -> Expr { .binary(.ge, a, b) }
public func && (a: Expr, b: @autoclosure () -> Expr) -> Expr { .binary(.and, a, b()) }
public func || (a: Expr, b: @autoclosure () -> Expr) -> Expr { .binary(.or, a, b()) }
public prefix func ! (a: Expr) -> Expr { .unary(.not, a) }
public func & (a: Expr, b: Expr) -> Expr { .binary(.bitAnd, a, b) }
public func | (a: Expr, b: Expr) -> Expr { .binary(.bitOr, a, b) }
public func ^ (a: Expr, b: Expr) -> Expr { .binary(.bitXor, a, b) }

// Scalar right-hand sides, so `col("x") > 100` and `col("x") * 2.5` need no wrapping.
public func + (a: Expr, b: Int) -> Expr { .binary(.add, a, .int(Int64(b))) }
public func - (a: Expr, b: Int) -> Expr { .binary(.sub, a, .int(Int64(b))) }
public func * (a: Expr, b: Int) -> Expr { .binary(.mul, a, .int(Int64(b))) }
public func / (a: Expr, b: Int) -> Expr { .binary(.div, a, .int(Int64(b))) }
public func == (a: Expr, b: Int) -> Expr { .binary(.eq, a, .int(Int64(b))) }
public func != (a: Expr, b: Int) -> Expr { .binary(.ne, a, .int(Int64(b))) }
public func < (a: Expr, b: Int) -> Expr { .binary(.lt, a, .int(Int64(b))) }
public func <= (a: Expr, b: Int) -> Expr { .binary(.le, a, .int(Int64(b))) }
public func > (a: Expr, b: Int) -> Expr { .binary(.gt, a, .int(Int64(b))) }
public func >= (a: Expr, b: Int) -> Expr { .binary(.ge, a, .int(Int64(b))) }
public func + (a: Expr, b: Double) -> Expr { .binary(.add, a, .double(b)) }
public func - (a: Expr, b: Double) -> Expr { .binary(.sub, a, .double(b)) }
public func * (a: Expr, b: Double) -> Expr { .binary(.mul, a, .double(b)) }
public func / (a: Expr, b: Double) -> Expr { .binary(.div, a, .double(b)) }
public func == (a: Expr, b: Double) -> Expr { .binary(.eq, a, .double(b)) }
public func != (a: Expr, b: Double) -> Expr { .binary(.ne, a, .double(b)) }
public func < (a: Expr, b: Double) -> Expr { .binary(.lt, a, .double(b)) }
public func <= (a: Expr, b: Double) -> Expr { .binary(.le, a, .double(b)) }
public func > (a: Expr, b: Double) -> Expr { .binary(.gt, a, .double(b)) }
public func >= (a: Expr, b: Double) -> Expr { .binary(.ge, a, .double(b)) }
public func + (a: Int, b: Expr) -> Expr { .binary(.add, .int(Int64(a)), b) }
public func - (a: Int, b: Expr) -> Expr { .binary(.sub, .int(Int64(a)), b) }
public func * (a: Int, b: Expr) -> Expr { .binary(.mul, .int(Int64(a)), b) }
public func / (a: Int, b: Expr) -> Expr { .binary(.div, .int(Int64(a)), b) }
public func + (a: Double, b: Expr) -> Expr { .binary(.add, .double(a), b) }
public func - (a: Double, b: Expr) -> Expr { .binary(.sub, .double(a), b) }
public func * (a: Double, b: Expr) -> Expr { .binary(.mul, .double(a), b) }
public func / (a: Double, b: Expr) -> Expr { .binary(.div, .double(a), b) }

extension Expr {
    public func and(_ o: Expr) -> Expr { .binary(.and, self, o) }
    public func or(_ o: Expr) -> Expr { .binary(.or, self, o) }
    public func andKleene(_ o: Expr) -> Expr { .binary(.andKleene, self, o) }
    public func orKleene(_ o: Expr) -> Expr { .binary(.orKleene, self, o) }
    public func cast(to t: ExprType) -> Expr { .cast(self, t) }
    public var isNullExpr: Expr { .isNull(self) }
    public var isValidExpr: Expr { .isValid(self) }
    public func fillNull(_ o: Expr) -> Expr { .fillNull(self, o) }
    public func isIn(_ set: [Expr]) -> Expr { .isIn(self, set) }
    public func isIn(_ set: [Int64]) -> Expr { .isIn(self, set.map { .int($0) }) }
    public func isIn(_ set: [Double]) -> Expr { .isIn(self, set.map { .double($0) }) }
    public func startsWith(_ p: String) -> Expr { .stringMatch(.startsWith, self, p) }
    public func contains(_ p: String) -> Expr { .stringMatch(.contains, self, p) }
    public func stringEquals(_ p: String) -> Expr { .stringMatch(.equals, self, p) }
    public var absolute: Expr { .unary(.abs, self) }
    public var negated: Expr { .unary(.negate, self) }
    public var squareRoot: Expr { .unary(.sqrt, self) }
    public var exponential: Expr { .unary(.exp, self) }
    public var naturalLog: Expr { .unary(.ln, self) }
    public var rounded: Expr { .unary(.round, self) }

    /// Every column this expression reads, in first-use order.
    public var referencedColumns: [String] {
        var seen = Set<String>(), out: [String] = []
        func walk(_ e: Expr) {
            switch e {
            case .column(let n): if seen.insert(n).inserted { out.append(n) }
            case .int, .typedInt, .double, .typedDouble, .bool, .string, .nullLiteral: break
            case .binary(_, let a, let b), .fillNull(let a, let b): walk(a); walk(b)
            case .unary(_, let a), .cast(let a, _), .isNull(let a), .isValid(let a), .stringMatch(_, let a, _): walk(a)
            case .ifElse(let c, let a, let b): walk(c); walk(a); walk(b)
            case .coalesce(let xs): xs.forEach(walk)
            case .isIn(let a, let xs): walk(a); xs.forEach(walk)
            }
        }
        walk(self)
        return out
    }

    /// Number of nodes, counting shared subtrees once per occurrence.
    public var nodeCount: Int {
        switch self {
        case .column, .int, .typedInt, .double, .typedDouble, .bool, .string, .nullLiteral: return 1
        case .binary(_, let a, let b), .fillNull(let a, let b): return 1 + a.nodeCount + b.nodeCount
        case .unary(_, let a), .cast(let a, _), .isNull(let a), .isValid(let a), .stringMatch(_, let a, _):
            return 1 + a.nodeCount
        case .ifElse(let c, let a, let b): return 1 + c.nodeCount + a.nodeCount + b.nodeCount
        case .coalesce(let xs): return 1 + xs.reduce(0) { $0 + $1.nodeCount }
        case .isIn(let a, let xs): return 1 + a.nodeCount + xs.reduce(0) { $0 + $1.nodeCount }
        }
    }
}

// MARK: - Serialised form (also the canonical text used as a cache key)

func quoteExprString(_ s: String) -> String {
    var out = "\""
    for c in s.unicodeScalars {
        switch c {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\t": out += "\\t"
        default: out.unicodeScalars.append(c)
        }
    }
    return out + "\""
}

extension Expr: CustomStringConvertible {
    /// The s-expression form. It round-trips through `Expr(text:)` and is the canonical key the
    /// compiled-kernel cache uses, so two structurally identical queries share one pipeline.
    public var description: String {
        switch self {
        case .column(let n): return "(col \(quoteExprString(n)))"
        case .int(let v): return "(int \(v))"
        case .typedInt(let v, let t): return "(\(t.token) \(v))"
        case .double(let v): return "(float \(fullPrecision(v)))"
        case .typedDouble(let v, let t): return "(\(t.token) \(fullPrecision(v)))"
        case .bool(let v): return "(bool \(v))"
        case .string(let s): return "(str \(quoteExprString(s)))"
        case .nullLiteral(let t): return "(null \(t.token))"
        case .binary(let op, let a, let b): return "(\(op.rawValue) \(a) \(b))"
        case .unary(let op, let a): return "(\(op.rawValue) \(a))"
        case .cast(let a, let t): return "(cast \(a) \(t.token))"
        case .ifElse(let c, let a, let b): return "(if_else \(c) \(a) \(b))"
        case .coalesce(let xs): return "(coalesce \(xs.map(\.description).joined(separator: " ")))"
        case .fillNull(let a, let b): return "(fill_null \(a) \(b))"
        case .isNull(let a): return "(is_null \(a))"
        case .isValid(let a): return "(is_valid \(a))"
        case .isIn(let a, let xs): return "(is_in \(a) \(xs.map(\.description).joined(separator: " ")))"
        case .stringMatch(let p, let a, let s): return "(\(p.rawValue) \(a) \(quoteExprString(s)))"
        }
    }
}

/// Shortest decimal string that round-trips to the same Double.
func fullPrecision(_ v: Double) -> String {
    if v.isNaN { return "nan" }
    if v.isInfinite { return v < 0 ? "-inf" : "inf" }
    return "\(v)"
}
