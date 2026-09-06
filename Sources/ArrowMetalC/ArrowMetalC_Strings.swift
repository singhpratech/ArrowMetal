import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the GPU string transforms (Kernels/StringTransforms.swift). One entry point with an op
// table, plus `am_str_concat` for the one binary operation the single-array signature cannot express.
// The op table is documented in include/arrowmetal.h; the numbers below are that table.

/// Op codes accepted by `am_str_transform`. 0–17 return a `utf8` array, 18–19 an `int32` array and
/// 20–25 a `bool` array.
enum StrTransformOp: Int32 {
    // utf8 -> utf8
    case asciiUpper = 0, asciiLower = 1, utf8Upper = 2, utf8Lower = 3, asciiSwapcase = 4, asciiCapitalize = 5
    case trimWhitespace = 6, ltrimWhitespace = 7, rtrimWhitespace = 8
    case trim = 9, ltrim = 10, rtrim = 11
    case replaceSubstring = 12, repeatCopies = 13, sliceCodeunits = 14, padLeft = 15, padRight = 16, reverse = 17
    // utf8 -> int32
    case countSubstring = 18, findSubstring = 19
    // utf8 -> bool
    case isAlnum = 20, isAlpha = 21, isDigit = 22, isSpace = 23, isUpper = 24, isLower = 25
}

private let errorKeyStrings = "ArrowMetalC.lastError"
private func setStringsError(_ e: Error) { Thread.current.threadDictionary[errorKeyStrings] = "\(e)" }

/// A `utf8` **or** `binary` array. Arrow's `binary_length`, `binary_repeat`, `binary_reverse` and the
/// byte-wise trims are defined on both, `MetalStringArray` stores both, and every transform below
/// carries the `isBinary` flag through — so a `binary` column comes back `binary`.
@inline(__always) private func stringHandle(_ p: OpaquePointer?) throws -> MetalStringArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let a = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    switch a {
    case .string(let s), .binary(let s): return s
    default: throw ArrowMetalError.unsupportedType("expected a utf8 or binary array, got \(a.arrowFormat)")
    }
}
/// `utf8` unless the transform produced (or preserved) Arrow `binary`.
@inline(__always) private func emit(_ s: MetalStringArray) -> AnyMetalArray { s.isBinary ? .binary(s) : .string(s) }
@inline(__always) private func emitString(_ a: AnyMetalArray, _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(a)).toOpaque())
    return 0
}
private func runStrings(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do { return emitString(try body(), out) } catch { setStringsError(error); return 1 }
}
/// A UTF-8 argument; a null pointer or a zero length is the empty string.
private func text(_ p: UnsafePointer<UInt8>?, _ len: Int64) -> String {
    guard let p, len > 0 else { return "" }
    return String(decoding: UnsafeBufferPointer(start: p, count: Int(len)), as: UTF8.self)
}

/// Every string transform behind one entry point.
///
/// `arg1`/`arg2` are UTF-8 byte arguments (pattern, replacement, trim set, pad character) and
/// `p1`/`p2` the integer arguments (max replacements, repeat count, slice bounds, pad width);
/// unused arguments may be NULL / 0. See the op table in `arrowmetal.h`.
@_cdecl("am_str_transform")
public func am_str_transform(_ a: OpaquePointer?, _ op: Int32,
                             _ arg1: UnsafePointer<UInt8>?, _ len1: Int64,
                             _ arg2: UnsafePointer<UInt8>?, _ len2: Int64,
                             _ p1: Int64, _ p2: Int64,
                             _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil else { return 2 }
    let a1 = text(arg1, len1), a2 = text(arg2, len2)
    return runStrings(out) {
        let s = try stringHandle(a)
        guard let kind = StrTransformOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown string transform op \(op)")
        }
        switch kind {
        case .asciiUpper: return emit(try s.asciiUpper())
        case .asciiLower: return emit(try s.asciiLower())
        case .utf8Upper: return emit(try s.utf8Upper())
        case .utf8Lower: return emit(try s.utf8Lower())
        case .asciiSwapcase: return emit(try s.asciiSwapcase())
        case .asciiCapitalize: return emit(try s.asciiCapitalize())
        case .trimWhitespace: return emit(try s.trim())
        case .ltrimWhitespace: return emit(try s.ltrim())
        case .rtrimWhitespace: return emit(try s.rtrim())
        case .trim: return emit(try s.trim(characters: a1))
        case .ltrim: return emit(try s.ltrim(characters: a1))
        case .rtrim: return emit(try s.rtrim(characters: a1))
        case .replaceSubstring: return emit(try s.replaceSubstring(a1, with: a2, maxReplacements: Int(p1)))
        case .repeatCopies: return emit(try s.repeat(Int(p1)))
        case .sliceCodeunits: return emit(try s.sliceCodeunits(start: Int(p1), stop: Int(p2)))
        case .padLeft: return emit(try s.padLeft(width: Int(p1), pad: a1.isEmpty ? " " : a1))
        case .padRight: return emit(try s.padRight(width: Int(p1), pad: a1.isEmpty ? " " : a1))
        case .reverse: return emit(try s.reverse())
        case .countSubstring: return .int32(try s.countSubstring(a1))
        case .findSubstring: return .int32(try s.findSubstring(a1))
        case .isAlnum: return .boolean(try s.classify(.alnum))
        case .isAlpha: return .boolean(try s.classify(.alpha))
        case .isDigit: return .boolean(try s.classify(.digit))
        case .isSpace: return .boolean(try s.classify(.space))
        case .isUpper: return .boolean(try s.classify(.upper))
        case .isLower: return .boolean(try s.classify(.lower))
        }
    }
}

/// Arrow `binary_join_element_wise`: `a[i] + separator + b[i]`, null when either side is null.
@_cdecl("am_str_concat")
public func am_str_concat(_ a: OpaquePointer?, _ b: OpaquePointer?,
                          _ separator: UnsafePointer<UInt8>?, _ sepLen: Int64,
                          _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, b != nil else { return 2 }
    let sep = text(separator, sepLen)
    return runStrings(out) { .string(try stringHandle(a).concat(try stringHandle(b), separator: sep)) }
}
