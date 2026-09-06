import Foundation

/// A fixed-width Arrow primitive type that has a matching Metal Shading Language scalar type.
public protocol ArrowPrimitive: Numeric, Comparable {
    /// Arrow C Data Interface format string (e.g. "l" for int64).
    static var arrowFormat: String { get }
    /// Metal Shading Language type name (e.g. "long").
    static var mslType: String { get }
    /// Whether the type is floating point (affects reduction accumulation).
    static var isFloatingPoint: Bool { get }
    static var byteWidth: Int { get }
    static var minValue: Self { get }
    static var maxValue: Self { get }
    var asDouble: Double { get }
    var asInt64: Int64 { get }
    var asUInt64: UInt64 { get }
    /// Conversion from the float accumulator used by GPU reductions (only used for floating point types).
    init(_ f: Float)
    /// Wrapping integer arithmetic / IEEE float arithmetic, matching the GPU kernels. Integer division by zero yields 0.
    static func wrappingApply(_ op: ArithmeticOp, _ a: Self, _ b: Self) -> Self
}

extension ArrowPrimitive where Self: FixedWidthInteger {
    public var asInt64: Int64 { Int64(truncatingIfNeeded: self) }
    public var asUInt64: UInt64 { UInt64(truncatingIfNeeded: self) }
    public static func wrappingApply(_ op: ArithmeticOp, _ a: Self, _ b: Self) -> Self {
        switch op {
        case .add: return a &+ b
        case .sub: return a &- b
        case .mul: return a &* b
        case .div: return b == 0 ? 0 : a.dividedReportingOverflow(by: b).partialValue
        }
    }
}

extension ArrowPrimitive where Self: BinaryFloatingPoint {
    public var asInt64: Int64 { Int64(self) }
    public var asUInt64: UInt64 { UInt64(self) }
    public static func wrappingApply(_ op: ArithmeticOp, _ a: Self, _ b: Self) -> Self {
        switch op {
        case .add: return a + b
        case .sub: return a - b
        case .mul: return a * b
        case .div: return a / b
        }
    }
}

extension Int8: ArrowPrimitive {
    public static var arrowFormat: String { "c" }; public static var mslType: String { "char" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 1 }
    public static var minValue: Int8 { .min }; public static var maxValue: Int8 { .max }
    public var asDouble: Double { Double(self) }
}
extension UInt8: ArrowPrimitive {
    public static var arrowFormat: String { "C" }; public static var mslType: String { "uchar" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 1 }
    public static var minValue: UInt8 { .min }; public static var maxValue: UInt8 { .max }
    public var asDouble: Double { Double(self) }
}
extension Int16: ArrowPrimitive {
    public static var arrowFormat: String { "s" }; public static var mslType: String { "short" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 2 }
    public static var minValue: Int16 { .min }; public static var maxValue: Int16 { .max }
    public var asDouble: Double { Double(self) }
}
extension UInt16: ArrowPrimitive {
    public static var arrowFormat: String { "S" }; public static var mslType: String { "ushort" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 2 }
    public static var minValue: UInt16 { .min }; public static var maxValue: UInt16 { .max }
    public var asDouble: Double { Double(self) }
}
extension Int32: ArrowPrimitive {
    public static var arrowFormat: String { "i" }; public static var mslType: String { "int" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 4 }
    public static var minValue: Int32 { .min }; public static var maxValue: Int32 { .max }
    public var asDouble: Double { Double(self) }
}
extension UInt32: ArrowPrimitive {
    public static var arrowFormat: String { "I" }; public static var mslType: String { "uint" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 4 }
    public static var minValue: UInt32 { .min }; public static var maxValue: UInt32 { .max }
    public var asDouble: Double { Double(self) }
}
extension Int64: ArrowPrimitive {
    public static var arrowFormat: String { "l" }; public static var mslType: String { "long" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 8 }
    public static var minValue: Int64 { .min }; public static var maxValue: Int64 { .max }
    public var asDouble: Double { Double(self) }
}
extension UInt64: ArrowPrimitive {
    public static var arrowFormat: String { "L" }; public static var mslType: String { "ulong" }
    public static var isFloatingPoint: Bool { false }; public static var byteWidth: Int { 8 }
    public static var minValue: UInt64 { .min }; public static var maxValue: UInt64 { .max }
    public var asDouble: Double { Double(self) }
}
extension Float: ArrowPrimitive {
    public static var arrowFormat: String { "f" }; public static var mslType: String { "float" }
    public static var isFloatingPoint: Bool { true }; public static var byteWidth: Int { 4 }
    public static var minValue: Float { -.infinity }; public static var maxValue: Float { .infinity }
    public var asDouble: Double { Double(self) }
}
// Metal has no native 64-bit float. Double columns are stored as-is in unified memory and
// reductions/compares are computed on the GPU using a two-float split representation where possible,
// but for correctness we currently run double kernels on CPU. See `DoubleSupport.md`.
extension Double: ArrowPrimitive {
    public static var arrowFormat: String { "g" }; public static var mslType: String { "double" }
    public static var isFloatingPoint: Bool { true }; public static var byteWidth: Int { 8 }
    public static var minValue: Double { -.infinity }; public static var maxValue: Double { .infinity }
    public var asDouble: Double { self }
}

/// Maps an Arrow format string to a primitive Swift type, or nil for unsupported types.
public func arrowPrimitiveType(forFormat f: String) -> (any ArrowPrimitive.Type)? {
    switch f {
    case "c": return Int8.self
    case "C": return UInt8.self
    case "s": return Int16.self
    case "S": return UInt16.self
    case "i": return Int32.self
    case "I": return UInt32.self
    case "l": return Int64.self
    case "L": return UInt64.self
    case "f": return Float.self
    case "g": return Double.self
    default: return nil
    }
}
