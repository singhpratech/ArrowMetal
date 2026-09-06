import Foundation

/// One `cast` entry point for every column type, dispatching on the Arrow format string of the target.
///
/// Until now the conversions lived under their own names — `cast(to:)` for numbers, `castUnit(to:)` for
/// temporal resolutions, `toDate32()`, `toFloat16()`, `toDecimal128()`, `toStrings()`, `parse(_:)` —
/// and `cast()` reached only the numeric ones. This is the single door Arrow's `cast` is, taking the
/// target as the format string the C data interface already speaks and `CastOptions` alongside it:
///
///     try column.cast(to: "l")                       // int64, unchecked (this package's old default)
///     try column.cast(to: "c", options: .safe)       // int8, raising on a value that does not fit
///     try column.cast(to: "+l", options: .safe)      // list<...>, casting the child (see below)
///
/// ## What it covers
///
/// * numeric to numeric, including float16, with `CastOptions`' overflow and truncation checks
/// * bool to and from numeric (`true` is 1, and any non-zero is `true`, as in Arrow)
/// * numeric and temporal to utf8, and utf8 to numeric or bool, through the existing string kernels
/// * temporal to temporal: a resolution change, `date32`/`date64`/`timestamp` in every direction,
///   with `allowTimeTruncate` and `allowTimeOverflow` honoured
/// * integer to and from `decimal128`, and a decimal rescale, with `allowDecimalTruncate` honoured
/// * decimal to float64
/// * `list<T>` to `list<U>` by casting the child, and `struct` field-wise — the offsets, the validity
///   bitmaps and the field names are shared, so only the leaves cost anything
///
/// The nested targets take the *child* format after the list or struct marker, because an Arrow format
/// string carries only one level: `cast(to: "+l", childFormats: ["l"])` is `list<int64>`.
///
/// ## What it does not cover
///
/// Dictionary, union, run-end-encoded, interval and fixed-size-binary targets are refused, as are
/// string-to-temporal casts (`strptime` does that, with a format), and binary to utf8 does not
/// validate — `allowInvalidUTF8` is accepted and ignored, which is the permissive direction.
extension AnyMetalArray {

    /// Arrow `cast` to the type named by an Arrow format string.
    ///
    /// `childFormats` supplies the target types of a `list` child or of a `struct`'s fields; an empty
    /// list means "leave the children as they are", which turns a nested cast into a no-op.
    public func cast(to format: String, options: CastOptions = .unsafe,
                     childFormats: [String] = []) throws -> AnyMetalArray {
        if format == arrowFormat && childFormats.isEmpty { return self }

        // ---- nested -------------------------------------------------------
        if format.hasPrefix("+l") || format == "+L" || format.hasPrefix("+w:") {
            guard case .list(let l) = self else {
                throw ArrowMetalError.unsupportedType("cast to \(format) needs a list column, got \(arrowFormat)")
            }
            guard let child = childFormats.first else { return self }
            let newChild = try l.values.cast(to: child, options: options,
                                             childFormats: Array(childFormats.dropFirst()))
            return .list(MetalListArray(length: l.length, nullCount: l.nullCount, validity: l.validity,
                                        offsets: l.offsets, values: newChild, kind: l.kind,
                                        fieldName: l.fieldName, context: l.context))
        }
        if format == "+s" {
            guard case .structure(let s) = self else {
                throw ArrowMetalError.unsupportedType("cast to +s needs a struct column, got \(arrowFormat)")
            }
            guard !childFormats.isEmpty else { return self }
            guard childFormats.count == s.children.count else {
                throw ArrowMetalError.invalidArrowArray(
                    "struct cast needs \(s.children.count) field formats, got \(childFormats.count)")
            }
            let kids = try zip(s.children, childFormats).map { try $0.cast(to: $1, options: options) }
            return .structure(try MetalStructArray(length: s.length, nullCount: s.nullCount,
                                                   validity: s.validity, names: s.names, children: kids,
                                                   context: s.context))
        }

        // ---- decimal ------------------------------------------------------
        if let target = ArrowDecimalType(format: format) { return try castToDecimal(target, options) }

        // ---- temporal -----------------------------------------------------
        if let target = ArrowTemporalType(format: format), target.isValid {
            return .temporal(try castToTemporal(target, options))
        }

        // ---- strings and binary -------------------------------------------
        if format == "u" || format == "U" || format == "z" || format == "Z" {
            switch self {
            case .string(let s), .binary(let s): return format == "z" || format == "Z" ? .binary(s) : .string(s)
            case .temporal(let t):
                // A temporal column formats through its integer storage; `strftime` is the calendar form.
                let ints: AnyMetalArray
                switch t.storage {
                case .int32(let a): ints = .int32(a)
                case .int64(let a): ints = .int64(a)
                }
                return .string(try ints.withCastablePrimitive { .string(try $0.toStrings()) }.unwrapString())
            default: return .string(try withCastablePrimitive { .string(try $0.toStrings()) }.unwrapString())
            }
        }
        if format == "b" {
            if case .string(let s) = self { return .boolean(try s.parseBool()) }
            if case .boolean = self { return self }
            return .boolean(try toBoolean())
        }

        // ---- numeric ------------------------------------------------------
        if case .string(let s) = self { return try s.parseTo(format) }
        return try toNumeric(format, options)
    }
}

// MARK: - The pieces

extension AnyMetalArray {

    /// This column as `MetalArray<T>` of whichever primitive it holds, going through the widenings the
    /// non-primitive-but-numeric types offer (bool as bytes, float16 as float32, temporal as its
    /// integer storage, decimal as float64).
    private func numericSource() throws -> AnyMetalArray {
        switch self {
        case .int8, .uint8, .int16, .uint16, .int32, .uint32, .int64, .uint64, .float32, .float64:
            return self
        case .boolean(let b): return .uint8(try b.toUInt8Array())
        case .float16(let h): return .float32(try h.toFloat32())
        case .temporal(let t):
            switch t.storage {
            case .int32(let a): return .int32(a)
            case .int64(let a): return .int64(a)
            }
        case .decimal(let d): return .float64(try d.toFloat64())
        case .smallDecimal(let d): return .float64(try d.toDecimal128().toFloat64())
        case .extended(let e): return try e.storage.numericSource()
        default:
            throw ArrowMetalError.unsupportedType("cast from \(arrowFormat) is not implemented")
        }
    }

    /// Numeric target, named by its format character.
    private func toNumeric(_ format: String, _ options: CastOptions) throws -> AnyMetalArray {
        let source = try numericSource()
        if format == "e" {                                  // float16 goes through float32
            let f = try source.toNumeric("f", options)
            guard case .float32(let a) = f else { throw ArrowMetalError.unsupportedType("cast to float16") }
            return .float16(try a.toFloat16())
        }
        return try source.withCastablePrimitive { a in
            switch format {
            case "c": return .int8(try a.cast(to: Int8.self, options: options))
            case "C": return .uint8(try a.cast(to: UInt8.self, options: options))
            case "s": return .int16(try a.cast(to: Int16.self, options: options))
            case "S": return .uint16(try a.cast(to: UInt16.self, options: options))
            case "i": return .int32(try a.cast(to: Int32.self, options: options))
            case "I": return .uint32(try a.cast(to: UInt32.self, options: options))
            case "l": return .int64(try a.cast(to: Int64.self, options: options))
            case "L": return .uint64(try a.cast(to: UInt64.self, options: options))
            case "f": return .float32(try a.cast(to: Float.self, options: options))
            case "g": return .float64(try a.cast(to: Double.self, options: options))
            default: throw ArrowMetalError.unsupportedType("cast target \(format) is not implemented")
            }
        }
    }

    /// Arrow's numeric-to-bool: zero is false, everything else true, nulls preserved.
    private func toBoolean() throws -> MetalBooleanArray {
        if case .boolean(let b) = self { return b }
        return try numericSource().withCastablePrimitive { .boolean(try $0.nonZeroMask()) }.unwrapBoolean()
    }

    /// Temporal target: a resolution change, plus the `date32` / `date64` / `timestamp` conversions.
    private func castToTemporal(_ target: ArrowTemporalType, _ options: CastOptions) throws -> MetalTemporalArray {
        guard case .temporal(let t) = self else {
            // An integer column reinterpreted as ticks of the target type, which is Arrow's
            // "cast int64 -> timestamp" and costs nothing but a type change.
            let ints = try numericSource()
            if target.usesInt64 {
                return try MetalTemporalArray(type: target,
                                              try ints.withCastablePrimitive { .int64(try $0.cast(to: Int64.self, options: options)) }.unwrapInt64())
            }
            return try MetalTemporalArray(type: target,
                                          try ints.withCastablePrimitive { .int32(try $0.cast(to: Int32.self, options: options)) }.unwrapInt32())
        }
        return try t.cast(to: target, options: options)
    }

    /// Decimal target: from an integer, from a float, or a rescale of another decimal.
    private func castToDecimal(_ target: ArrowDecimalType, _ options: CastOptions) throws -> AnyMetalArray {
        if case .decimal(let d) = self {
            if d.type.scale != target.scale && !options.allowDecimalTruncate && target.scale < d.type.scale {
                try Self.refuseDecimalTruncation(d, toScale: target.scale)
            }
            let rescaled = try d.rescaled(to: target.scale, mode: .truncate)
            return .decimal(try MetalDecimalArray(type: target, length: rescaled.length,
                                                  nullCount: rescaled.nullCount, validity: rescaled.validity,
                                                  values: rescaled.values, context: rescaled.context))
        }
        if case .smallDecimal(let d) = self { return try AnyMetalArray.decimal(try d.toDecimal128()).castToDecimal(target, options) }
        // From a number: multiply by 10^scale in int64 and reinterpret.
        let ints = try numericSource().withCastablePrimitive { .int64(try $0.cast(to: Int64.self, options: options)) }.unwrapInt64()
        var factor: Int64 = 1
        for _ in 0..<target.scale { factor *= 10 }
        let scaled = factor == 1 ? ints : try ints.arithmetic(.mul, factor)
        return .decimal(try MetalDecimalArray.fromInt64(scaled, type: target))
    }

    /// Raises naming the first row a rescale would round away.
    private static func refuseDecimalTruncation(_ d: MetalDecimalArray, toScale: Int) throws {
        let kept = try d.rescaled(to: toScale, mode: .truncate)
        let back = try kept.rescaled(to: d.type.scale, mode: .truncate)
        let same = try d.compare(.eq, back)
        for i in 0..<same.length where same[i] == false {
            throw ArrowMetalError.overflow(op: "cast", index: i,
                                           detail: CastLoss.decimalTruncate.message(from: d.type.arrowFormat,
                                                                                    to: "decimal(scale \(toScale))"))
        }
    }

    // Small unwrappers so the closures above stay one-liners.
    func unwrapString() throws -> MetalStringArray {
        guard case .string(let s) = self else { throw ArrowMetalError.unsupportedType("expected utf8") }
        return s
    }
    func unwrapBoolean() throws -> MetalBooleanArray {
        guard case .boolean(let b) = self else { throw ArrowMetalError.unsupportedType("expected boolean") }
        return b
    }
    func unwrapInt32() throws -> MetalArray<Int32> {
        guard case .int32(let a) = self else { throw ArrowMetalError.unsupportedType("expected int32") }
        return a
    }
    func unwrapInt64() throws -> MetalArray<Int64> {
        guard case .int64(let a) = self else { throw ArrowMetalError.unsupportedType("expected int64") }
        return a
    }

    /// The ten primitive cases, erased so the branches above do not each repeat the switch.
    func withCastablePrimitive(_ body: (any CastablePrimitive) throws -> AnyMetalArray) throws -> AnyMetalArray {
        switch self {
        case .int8(let a): return try body(a)
        case .uint8(let a): return try body(a)
        case .int16(let a): return try body(a)
        case .uint16(let a): return try body(a)
        case .int32(let a): return try body(a)
        case .uint32(let a): return try body(a)
        case .int64(let a): return try body(a)
        case .uint64(let a): return try body(a)
        case .float32(let a): return try body(a)
        case .float64(let a): return try body(a)
        default: throw ArrowMetalError.unsupportedType("cast from \(arrowFormat) is not implemented")
        }
    }
}

/// The cast operations one primitive array offers, erased.
public protocol CastablePrimitive {
    func cast<U: ArrowPrimitive>(to _: U.Type, options: CastOptions) throws -> MetalArray<U>
    /// Arrow's numeric-to-bool: true wherever the value is not zero.
    func nonZeroMask() throws -> MetalBooleanArray
    func toStrings() throws -> MetalStringArray
}

extension MetalArray: CastablePrimitive {
    public func nonZeroMask() throws -> MetalBooleanArray { try compare(.ne, T.zero) }
}

extension MetalStringArray {
    /// utf8 to a numeric type, by the format character.
    func parseTo(_ format: String) throws -> AnyMetalArray {
        switch format {
        case "c": return .int8(try parse(Int8.self, strict: true))
        case "C": return .uint8(try parse(UInt8.self, strict: true))
        case "s": return .int16(try parse(Int16.self, strict: true))
        case "S": return .uint16(try parse(UInt16.self, strict: true))
        case "i": return .int32(try parse(Int32.self, strict: true))
        case "I": return .uint32(try parse(UInt32.self, strict: true))
        case "l": return .int64(try parse(Int64.self, strict: true))
        case "L": return .uint64(try parse(UInt64.self, strict: true))
        case "f": return .float32(try parse(Float.self, strict: true))
        case "g": return .float64(try parse(Double.self, strict: true))
        default: throw ArrowMetalError.unsupportedType("cast utf8 -> \(format) is not implemented")
        }
    }
}

extension MetalTemporalArray {
    /// Arrow `cast` between two temporal types: a resolution change and the date/timestamp conversions,
    /// with `allowTimeTruncate` and `allowTimeOverflow` honoured.
    ///
    /// A `date32` counts whole days and a `date64` whole milliseconds, so both are handled as timestamps
    /// of the matching resolution and converted back at the end. Casting *to* a date floors to the day,
    /// which is what Arrow does and is not considered a loss by either flag.
    public func cast(to target: ArrowTemporalType, options: CastOptions = .unsafe) throws -> MetalTemporalArray {
        if target == type { return self }
        // Everything travels as int64 ticks plus the nanoseconds one tick is worth.
        let fromNS = nanosecondsPerTick
        let toNS = MetalTemporalArray.nanosecondsPerTick(of: target)
        var wide = try int64Values()
        if fromNS != toNS {
            if toNS > fromNS {                                   // coarser target: digits can be lost
                let factor = toNS / fromNS
                if !options.allowTimeTruncate {
                    let kept = try wide.arithmetic(.div, factor)
                    let back = try kept.arithmetic(.mul, factor)
                    let same = try wide.compare(.eq, back)
                    for i in 0..<same.length where same[i] == false {
                        throw ArrowMetalError.overflow(op: "cast", index: i,
                                                       detail: CastLoss.timeTruncate.message(from: type.arrowFormat,
                                                                                             to: target.arrowFormat))
                    }
                }
                wide = try wide.arithmetic(.div, factor)
            } else {                                             // finer target: the range can overflow
                let factor = fromNS / toNS
                if !options.allowTimeOverflow {
                    let scaled = try wide.arithmetic(.mul, factor)
                    let back = try scaled.arithmetic(.div, factor)
                    let same = try wide.compare(.eq, back)
                    for i in 0..<same.length where same[i] == false {
                        throw ArrowMetalError.overflow(op: "cast", index: i,
                                                       detail: CastLoss.timeOverflow.message(from: type.arrowFormat,
                                                                                             to: target.arrowFormat))
                    }
                }
                wide = try wide.arithmetic(.mul, factor)
            }
        }
        if target.usesInt64 { return try MetalTemporalArray(type: target, wide) }
        return try MetalTemporalArray(type: target, try wide.cast(to: Int32.self))
    }

    /// Nanoseconds in one tick of `t` (a whole day for `date32`, a millisecond for `date64`).
    static func nanosecondsPerTick(of t: ArrowTemporalType) -> Int64 {
        switch t {
        case .date32: return 86_400_000_000_000
        case .date64: return 1_000_000
        case .time32(let u), .time64(let u), .timestamp(let u, _), .duration(let u):
            return 1_000_000_000 / u.perSecond
        }
    }
}
