import Foundation
import CArrowABI
import ArrowMetal

// C ABI for regular expressions, the string ↔ number casts and temporal rounding / arithmetic.
// Four entry points, each with an op table documented in include/arrowmetal.h; the numbers below are
// those tables.

/// Op codes accepted by `am_regex`.
enum RegexOp: Int32 {
    case matchSubstring = 0, countSubstring = 1, findSubstring = 2, replaceSubstring = 3, matchLike = 4
    case splitPatternValues = 5, splitPatternOffsets = 6
    case splitWhitespaceValues = 7, splitWhitespaceOffsets = 8
    case extractGroup = 9
    case splitRegexValues = 10, splitRegexOffsets = 11
}

/// Op codes accepted by `am_temporal_math`.
enum TemporalMathOp: Int32 {
    case floor = 0, ceil = 1, round = 2
    case addDuration = 3, subtract = 4, daysBetween = 5
    case quarter = 6, dayOfYear = 7, isoWeek = 8, isoYear = 9, isLeapYear = 10
    case millisecond = 11, microsecond = 12, nanosecond = 13
}

private let errorKeyText = "ArrowMetalC.lastError"
private func setTextError(_ e: Error) { Thread.current.threadDictionary[errorKeyText] = "\(e)" }

@inline(__always) private func anyHandle(_ p: OpaquePointer?) throws -> AnyMetalArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    return Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
}
@inline(__always) private func textHandle(_ p: OpaquePointer?) throws -> MetalStringArray {
    let a = try anyHandle(p)
    switch a {
    case .string(let s), .binary(let s): return s
    default: throw ArrowMetalError.unsupportedType("expected a utf8 array, got \(a.arrowFormat)")
    }
}
@inline(__always) private func temporalHandle(_ p: OpaquePointer?) throws -> MetalTemporalArray {
    let a = try anyHandle(p)
    guard case .temporal(let t) = a else {
        throw ArrowMetalError.unsupportedType("expected a temporal array, got \(a.arrowFormat)")
    }
    return t
}
private func runText(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { setTextError(error); return 1 }
}
/// A UTF-8 argument; a null pointer or a zero length is the empty string.
private func utf8Arg(_ p: UnsafePointer<UInt8>?, _ len: Int64) -> String {
    guard let p, len > 0 else { return "" }
    return String(decoding: UnsafeBufferPointer(start: p, count: Int(len)), as: UTF8.self)
}

// MARK: - Regular expressions, SQL LIKE and splitting

/// Every regex-shaped string function behind one entry point.
///
/// `pattern` is the regular expression (or the literal separator, or the SQL `LIKE` pattern);
/// `repl` is the replacement template for op 3 and the capture-group name for op 9. `flags` bit 0
/// requests case-insensitive matching. See the op table in `arrowmetal.h`.
@_cdecl("am_regex")
public func am_regex(_ a: OpaquePointer?, _ op: Int32,
                     _ pattern: UnsafePointer<UInt8>?, _ len: Int64,
                     _ repl: UnsafePointer<UInt8>?, _ rlen: Int64,
                     _ flags: Int32,
                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil else { return 2 }
    let pat = utf8Arg(pattern, len), rep = utf8Arg(repl, rlen)
    let ignoreCase = (flags & 1) != 0
    return runText(out) {
        let s = try textHandle(a)
        guard let kind = RegexOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown regex op \(op)")
        }
        switch kind {
        case .matchSubstring: return .boolean(try s.matchSubstringRegex(pat, ignoreCase: ignoreCase))
        case .countSubstring: return .int32(try s.countSubstringRegex(pat, ignoreCase: ignoreCase))
        case .findSubstring: return .int32(try s.findSubstringRegex(pat, ignoreCase: ignoreCase))
        case .replaceSubstring: return .string(try s.replaceSubstringRegex(pat, with: rep, ignoreCase: ignoreCase))
        case .matchLike: return .boolean(try s.matchLike(pat, ignoreCase: ignoreCase))
        case .splitPatternValues: return .string(try s.splitPatternPair(pat).values)
        case .splitPatternOffsets: return .int32(try s.splitPatternPair(pat).offsets)
        case .splitWhitespaceValues: return .string(try s.splitWhitespacePair().values)
        case .splitWhitespaceOffsets: return .int32(try s.splitWhitespacePair().offsets)
        case .splitRegexValues: return .string(try s.splitPatternRegexPair(pat, ignoreCase: ignoreCase).values)
        case .splitRegexOffsets: return .int32(try s.splitPatternRegexPair(pat, ignoreCase: ignoreCase).offsets)
        case .extractGroup:
            let groups = try s.extractRegex(pat, ignoreCase: ignoreCase)
            guard let column = groups[rep] else {
                throw ArrowMetalError.invalidArrowArray(
                    "no capture group named \"\(rep)\" in \"\(pat)\"; found \(groups.keys.sorted())")
            }
            return .string(column)
        }
    }
}

// MARK: - Casts between strings and numbers

/// Arrow `cast(utf8)` for a primitive or boolean array: the decimal text of every value.
@_cdecl("am_to_strings")
public func am_to_strings(_ a: OpaquePointer?, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil else { return 2 }
    return runText(out) {
        switch try anyHandle(a) {
        case .int8(let x): return .string(try x.toStrings())
        case .uint8(let x): return .string(try x.toStrings())
        case .int16(let x): return .string(try x.toStrings())
        case .uint16(let x): return .string(try x.toStrings())
        case .int32(let x): return .string(try x.toStrings())
        case .uint32(let x): return .string(try x.toStrings())
        case .int64(let x): return .string(try x.toStrings())
        case .uint64(let x): return .string(try x.toStrings())
        case .float32(let x): return .string(try x.toStrings())
        case .float64(let x): return .string(try x.toStrings())
        case .boolean(let x): return .string(try x.toStrings())
        case let other: throw ArrowMetalError.unsupportedType("am_to_strings does not accept \(other.arrowFormat)")
        }
    }
}

/// Parses a `utf8` array, or formats a temporal one.
///
/// * `a` is `utf8` and `format` is one of `c C s S i I l L f g b` — Arrow `cast` to that type.
///   `strict` makes an unparseable value an error instead of a null.
/// * `a` is `utf8` and `format` is anything else — `strptime` with that C format, UTC, producing
///   `timestamp[us]`. Rescale afterwards with `am_temporal_cast_unit`.
/// * `a` is temporal — `strftime` with that C format, UTC, producing `utf8`. `strict` is ignored.
@_cdecl("am_parse")
public func am_parse(_ a: OpaquePointer?, _ format: UnsafePointer<CChar>?, _ strict: Int32,
                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, let format else { return 2 }
    let f = String(cString: format)
    let isStrict = strict != 0
    return runText(out) {
        let any = try anyHandle(a)
        if case .temporal(let t) = any { return .string(try t.strftime(f)) }
        let s = try textHandle(a)
        switch f {
        case "c": return .int8(try s.parse(Int8.self, strict: isStrict))
        case "C": return .uint8(try s.parse(UInt8.self, strict: isStrict))
        case "s": return .int16(try s.parse(Int16.self, strict: isStrict))
        case "S": return .uint16(try s.parse(UInt16.self, strict: isStrict))
        case "i": return .int32(try s.parse(Int32.self, strict: isStrict))
        case "I": return .uint32(try s.parse(UInt32.self, strict: isStrict))
        case "l": return .int64(try s.parse(Int64.self, strict: isStrict))
        case "L": return .uint64(try s.parse(UInt64.self, strict: isStrict))
        case "f": return .float32(try s.parse(Float.self, strict: isStrict))
        case "g": return .float64(try s.parse(Double.self, strict: isStrict))
        case "b": return .boolean(try s.parseBool(strict: isStrict))
        default: return .temporal(try s.strptime(f, unit: .micro, strict: isStrict))
        }
    }
}

// MARK: - Temporal rounding, arithmetic and calendar fields

/// Temporal rounding, arithmetic and the calendar fields beyond `am_temporal_extract`.
///
/// For ops 0–2 `p1` packs the rounding unit in its low 8 bits and the multiple above them
/// (`unit | (multiple << 8)`; a multiple of 0 means 1). For op 3 pass either `b` (a duration column)
/// or, with `b` NULL, a scalar in `p1` counted in the array's own ticks. Ops 4 and 5 need `b`.
@_cdecl("am_temporal_math")
public func am_temporal_math(_ a: OpaquePointer?, _ op: Int32, _ p1: Int64, _ b: OpaquePointer?,
                             _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil else { return 2 }
    return runText(out) {
        let t = try temporalHandle(a)
        guard let kind = TemporalMathOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown temporal math op \(op)")
        }
        func unitAndMultiple() throws -> (TemporalRoundUnit, Int) {
            let index = Int(p1 & 0xFF)
            guard index >= 0, index < TemporalRoundUnit.allCases.count else {
                throw ArrowMetalError.invalidArrowArray("unknown rounding unit \(index)")
            }
            let multiple = Int(p1 >> 8)
            return (TemporalRoundUnit.allCases[index], multiple <= 0 ? 1 : multiple)
        }
        switch kind {
        case .floor:
            let (u, m) = try unitAndMultiple(); return .temporal(try t.floorTemporal(to: u, multiple: m))
        case .ceil:
            let (u, m) = try unitAndMultiple(); return .temporal(try t.ceilTemporal(to: u, multiple: m))
        case .round:
            let (u, m) = try unitAndMultiple(); return .temporal(try t.roundTemporal(to: u, multiple: m))
        case .addDuration:
            guard b != nil else { return .temporal(try t.addDuration(p1)) }
            return .temporal(try t.addDuration(try temporalHandle(b)))
        case .subtract: return .temporal(try t.subtractTemporal(try temporalHandle(b)))
        case .daysBetween: return .int64(try t.daysBetween(try temporalHandle(b)))
        case .quarter: return .int32(try t.quarter())
        case .dayOfYear: return .int32(try t.dayOfYear())
        case .isoWeek: return .int32(try t.isoWeek())
        case .isoYear: return .int32(try t.isoYear())
        case .isLeapYear: return .boolean(try t.isLeapYear())
        case .millisecond: return .int32(try t.millisecond())
        case .microsecond: return .int32(try t.microsecond())
        case .nanosecond: return .int32(try t.nanosecond())
        }
    }
}
