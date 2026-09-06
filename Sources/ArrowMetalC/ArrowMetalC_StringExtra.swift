import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the remaining Arrow string surface (Kernels/StringExtra.swift and
// Kernels/StringContainment.swift): the character-class predicates, the capitalize / title / center /
// replace-slice / trim / normalize transforms, extract_regex_span, binary_join over a list<utf8>, and
// is_in / index_in over strings.
//
// Five entry points, two of them with an op table documented in include/arrowmetal.h; the numbers
// below are those tables. Handles and error reporting follow ArrowMetalC.swift exactly.

/// Op codes accepted by `am_string_predicate`; the numbering is `StringPredicate` in ArrowMetal.
enum StringPredicateOp: Int32 {
    case asciiIsPrintable = 0, asciiIsTitle = 1, stringIsAscii = 2
    case utf8IsAlnum = 3, utf8IsAlpha = 4, utf8IsDecimal = 5, utf8IsDigit = 6, utf8IsLower = 7
    case utf8IsNumeric = 8, utf8IsPrintable = 9, utf8IsSpace = 10, utf8IsTitle = 11, utf8IsUpper = 12
}

/// Op codes accepted by `am_string_transform`. 0–12 return `utf8` (5 returns `binary`), 13–14 `int32`.
enum StringExtraOp: Int32 {
    case asciiTitle = 0, utf8Capitalize = 1, utf8Title = 2
    case center = 3
    case replaceSlice = 4, binaryReplaceSlice = 5
    case trim = 6, ltrim = 7, rtrim = 8
    case trimWhitespace = 9, ltrimWhitespace = 10, rtrimWhitespace = 11
    case normalize = 12
    case regexSpanStart = 13, regexSpanLength = 14
}

private let errorKeyStringExtra = "ArrowMetalC.lastError"
private func setStringExtraError(_ e: Error) { Thread.current.threadDictionary[errorKeyStringExtra] = "\(e)" }

@inline(__always) private func sxStringHandle(_ p: OpaquePointer?) throws -> MetalStringArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let a = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    switch a {
    case .string(let s), .binary(let s): return s
    default: throw ArrowMetalError.unsupportedType("expected a utf8 or binary array, got \(a.arrowFormat)")
    }
}
@inline(__always) private func sxListHandle(_ p: OpaquePointer?) throws -> MetalListArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let a = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    guard case .list(let l) = a else {
        throw ArrowMetalError.unsupportedType("expected a list<utf8> array, got \(a.arrowFormat)")
    }
    return l
}
private func runStringExtra(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { setStringExtraError(error); return 1 }
}
/// A UTF-8 argument; a null pointer or a zero length is the empty string.
private func sxText(_ p: UnsafePointer<UInt8>?, _ len: Int64) -> String {
    guard let p, len > 0 else { return "" }
    return String(decoding: UnsafeBufferPointer(start: p, count: Int(len)), as: UTF8.self)
}
private func sxBytes(_ p: UnsafePointer<UInt8>?, _ len: Int64) -> [UInt8] {
    guard let p, len > 0 else { return [] }
    return Array(UnsafeBufferPointer(start: p, count: Int(len)))
}
/// `utf8` unless the transform produced Arrow `binary`.
private func sxEmit(_ s: MetalStringArray) -> AnyMetalArray { s.isBinary ? .binary(s) : .string(s) }

/// Every character-class predicate behind one entry point; the op table is in `arrowmetal.h`.
///
/// Ops 0–2 are byte-wise and run entirely on the GPU. Ops 3–12 are Arrow's Unicode `utf8_is_*` family:
/// the same GPU pass answers them and reports which rows carry a byte ≥ 0x80, and only those rows are
/// re-decided on the CPU, so an ASCII-only column never leaves the device.
@_cdecl("am_string_predicate")
public func am_string_predicate(_ a: OpaquePointer?, _ op: Int32,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    return runStringExtra(out) {
        guard let kind = StringPredicateOp(rawValue: op),
              let p = StringPredicate(rawValue: Int(kind.rawValue)) else {
            throw ArrowMetalError.invalidArrowArray("unknown string predicate op \(op)")
        }
        return .boolean(try sxStringHandle(a).predicate(p))
    }
}

/// The remaining string transforms behind one entry point.
///
/// `arg1` / `arg2` are UTF-8 byte arguments (pad character, replacement, trim set, regex pattern,
/// capture-group name) and `p1` / `p2` the integer ones (width, slice bounds, normalisation form,
/// flags); anything an op does not use may be NULL / 0. See the op table in `arrowmetal.h`.
@_cdecl("am_string_transform")
public func am_string_transform(_ a: OpaquePointer?, _ op: Int32, _ p1: Int64, _ p2: Int64,
                                _ arg1: UnsafePointer<UInt8>?, _ len1: Int64,
                                _ arg2: UnsafePointer<UInt8>?, _ len2: Int64,
                                _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    let a1 = sxText(arg1, len1), a2 = sxText(arg2, len2)
    let raw1 = sxBytes(arg1, len1)
    return runStringExtra(out) {
        let s = try sxStringHandle(a)
        guard let kind = StringExtraOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown string transform op \(op)")
        }
        switch kind {
        case .asciiTitle: return .string(try s.asciiTitle())
        case .utf8Capitalize: return .string(try s.utf8Capitalize())
        case .utf8Title: return .string(try s.utf8Title())
        case .center: return .string(try s.center(width: Int(p1), pad: a1.isEmpty ? " " : a1))
        case .replaceSlice: return .string(try s.replaceSlice(start: Int(p1), stop: Int(p2), with: a1))
        case .binaryReplaceSlice:
            return sxEmit(try s.replaceSliceBytes(start: Int(p1), stop: Int(p2), with: raw1))
        case .trim: return .string(try s.utf8Trim(characters: a1))
        case .ltrim: return .string(try s.utf8Ltrim(characters: a1))
        case .rtrim: return .string(try s.utf8Rtrim(characters: a1))
        case .trimWhitespace: return .string(try s.utf8TrimWhitespace())
        case .ltrimWhitespace: return .string(try s.utf8LtrimWhitespace())
        case .rtrimWhitespace: return .string(try s.utf8RtrimWhitespace())
        case .normalize:
            guard let form = UnicodeNormalizationForm(rawValue: Int(p1)) else {
                throw ArrowMetalError.invalidArrowArray("unknown normalisation form \(p1); 0 NFC, 1 NFKC, 2 NFD, 3 NFKD")
            }
            return .string(try s.utf8Normalize(form))
        case .regexSpanStart, .regexSpanLength:
            let spans = try s.extractRegexSpan(a1, ignoreCase: (p2 & 1) != 0)
            guard let span = spans[a2] else {
                throw ArrowMetalError.invalidArrowArray(
                    "no capture group named \"\(a2)\" in \"\(a1)\"; found \(spans.keys.sorted())")
            }
            return .int32(kind == .regexSpanStart ? span.start : span.length)
        }
    }
}

/// Arrow `is_in` over `utf8`: true where the value appears among the non-null values of `set`.
/// A null value is never in the set, so the result never has nulls (`null_matching_behavior = "skip"`).
@_cdecl("am_string_is_in")
public func am_string_is_in(_ a: OpaquePointer?, _ set: OpaquePointer?,
                            _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, set != nil, out != nil else { return 2 }
    return runStringExtra(out) { .boolean(try sxStringHandle(a).isIn(try sxStringHandle(set))) }
}

/// Arrow `index_in` over `utf8`: the int32 position in `set` of each value's first occurrence there,
/// null where the value is null or absent.
@_cdecl("am_string_index_in")
public func am_string_index_in(_ a: OpaquePointer?, _ set: OpaquePointer?,
                               _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, set != nil, out != nil else { return 2 }
    return runStringExtra(out) { .int32(try sxStringHandle(a).indexIn(try sxStringHandle(set))) }
}

/// Arrow `binary_join`: joins the child strings of every row of a `list<utf8>`.
///
/// Pass either a scalar separator in `sep` / `sep_len`, or a per-row `utf8` column in `sep_array`
/// (which then wins). A null row, any null element inside a row, and a null separator all give a null
/// output row.
@_cdecl("am_binary_join")
public func am_binary_join(_ list: OpaquePointer?, _ sep: UnsafePointer<UInt8>?, _ sepLen: Int64,
                           _ sepArray: OpaquePointer?,
                           _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard list != nil, out != nil else { return 2 }
    let separator = sxText(sep, sepLen)
    return runStringExtra(out) {
        let l = try sxListHandle(list)
        if sepArray != nil { return .string(try l.binaryJoin(separator: try sxStringHandle(sepArray))) }
        return .string(try l.binaryJoin(separator: separator))
    }
}
