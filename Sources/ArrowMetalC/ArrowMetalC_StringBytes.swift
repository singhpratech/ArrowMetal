import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the byte-indexed string functions (Kernels/StringBytes.swift), the list-returning
// splitters (Kernels/StringSplit.swift), the N-column binary_join_element_wise and the
// struct-returning extract_regex / extract_regex_span (Kernels/StringStructs.swift).
//
// Four entry points, each with an op table documented in include/arrowmetal.h; the numbers below are
// those tables. Handles and error reporting follow ArrowMetalC.swift exactly.

/// Op codes accepted by `am_byte_transform`. 0 and 2–3 return `binary` or `utf8` as the input was;
/// 1 and 4–6 return the input's own type.
enum ByteTransformOp: Int32 {
    case binarySlice = 0, utf8SliceStep = 1, binaryReverse = 2, asciiReverse = 3
    case asciiLpad = 4, asciiRpad = 5, asciiCenter = 6
}

/// Op codes accepted by `am_split`.
enum SplitOp: Int32 {
    case pattern = 0, patternRegex = 1, asciiWhitespace = 2, utf8Whitespace = 3
}

private let errorKeyStringBytes = "ArrowMetalC.lastError"
private func setStringBytesError(_ e: Error) { Thread.current.threadDictionary[errorKeyStringBytes] = "\(e)" }

@inline(__always) private func sbHandle(_ p: OpaquePointer?) throws -> MetalStringArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let a = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    switch a {
    case .string(let s), .binary(let s): return s
    default: throw ArrowMetalError.unsupportedType("expected a utf8 or binary array, got \(a.arrowFormat)")
    }
}
private func runStringBytes(_ out: UnsafeMutablePointer<OpaquePointer?>?,
                            _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { setStringBytesError(error); return 1 }
}
private func sbEmit(_ s: MetalStringArray) -> AnyMetalArray { s.isBinary ? .binary(s) : .string(s) }
private func sbText(_ p: UnsafePointer<UInt8>?, _ len: Int64) -> String {
    guard let p, len > 0 else { return "" }
    return String(decoding: UnsafeBufferPointer(start: p, count: Int(len)), as: UTF8.self)
}

/// The byte-indexed transforms behind one entry point.
///
/// `p1` / `p2` / `p3` are `start` / `stop` / `step` for the two slicing ops and `width` (in `p1`) for
/// the three padding ones; `arg1` is the pad character. `flags` bit 0 says a `stop` was given at all —
/// without it the slice runs to the end for a positive step and to the beginning for a negative one,
/// which is what Arrow's `SliceOptions` means by its sentinel bounds.
@_cdecl("am_byte_transform")
public func am_byte_transform(_ a: OpaquePointer?, _ op: Int32,
                              _ p1: Int64, _ p2: Int64, _ p3: Int64, _ flags: Int32,
                              _ arg1: UnsafePointer<UInt8>?, _ len1: Int64,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    let pad = sbText(arg1, len1)
    let stop: Int? = (flags & 1) != 0 ? Int(clamping: p2) : nil
    return runStringBytes(out) {
        let s = try sbHandle(a)
        guard let kind = ByteTransformOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown byte transform op \(op)")
        }
        switch kind {
        case .binarySlice:
            return sbEmit(try s.binarySlice(start: Int(clamping: p1), stop: stop, step: Int(clamping: p3)))
        case .utf8SliceStep:
            return sbEmit(try s.sliceCodeunits(start: Int(clamping: p1), stop: stop, step: Int(clamping: p3)))
        case .binaryReverse: return sbEmit(try s.binaryReverse())
        case .asciiReverse: return sbEmit(try s.asciiReverse())
        case .asciiLpad: return sbEmit(try s.asciiLpad(width: Int(p1), pad: pad.isEmpty ? " " : pad))
        case .asciiRpad: return sbEmit(try s.asciiRpad(width: Int(p1), pad: pad.isEmpty ? " " : pad))
        case .asciiCenter: return sbEmit(try s.asciiCenter(width: Int(p1), pad: pad.isEmpty ? " " : pad))
        }
    }
}

/// Arrow's four splitting functions, returning a `list<utf8>` column.
///
/// `flags` bit 0 requests `reverse`, bit 1 case-insensitive matching (regex only). `part` selects
/// what comes back: 0 the list column, 1 the int32 row offsets, 2 the flat `utf8` pieces — the last
/// two being the convenience pair, which is the same buffers with no copy.
@_cdecl("am_split")
public func am_split(_ a: OpaquePointer?, _ op: Int32,
                     _ pattern: UnsafePointer<UInt8>?, _ plen: Int64,
                     _ maxSplits: Int64, _ flags: Int32, _ part: Int32,
                     _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    let pat = sbText(pattern, plen)
    let reverse = (flags & 1) != 0, ignoreCase = (flags & 2) != 0
    return runStringBytes(out) {
        let s = try sbHandle(a)
        guard let kind = SplitOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown split op \(op)")
        }
        let list: MetalListArray
        switch kind {
        case .pattern:
            list = try s.splitPattern(pat, maxSplits: Int(maxSplits), reverse: reverse)
        case .patternRegex:
            list = try s.splitPatternRegex(pat, maxSplits: Int(maxSplits), reverse: reverse,
                                           ignoreCase: ignoreCase)
        case .asciiWhitespace:
            list = try s.splitWhitespace(unicode: false, maxSplits: Int(maxSplits), reverse: reverse)
        case .utf8Whitespace:
            list = try s.splitWhitespace(unicode: true, maxSplits: Int(maxSplits), reverse: reverse)
        }
        switch part {
        case 1: return .int32(try list.stringPair().offsets)
        case 2: return .string(try list.stringPair().values)
        default: return .list(list)
        }
    }
}

/// Arrow `binary_join_element_wise` over N columns and a scalar separator.
///
/// `nullHandling` is 0 emit_null, 1 skip, 2 replace; `repl` is the `null_replacement` for mode 2.
@_cdecl("am_join_element_wise")
public func am_join_element_wise(_ handles: UnsafePointer<OpaquePointer?>?, _ count: Int64,
                                 _ sep: UnsafePointer<UInt8>?, _ sepLen: Int64,
                                 _ nullHandling: Int32,
                                 _ repl: UnsafePointer<UInt8>?, _ replLen: Int64,
                                 _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard let handles, count > 0, out != nil else { return 2 }
    let separator = sbText(sep, sepLen), replacement = sbText(repl, replLen)
    return runStringBytes(out) {
        guard let mode = JoinNullHandling(rawValue: Int(nullHandling)) else {
            throw ArrowMetalError.invalidArrowArray(
                "unknown null_handling \(nullHandling); 0 emit_null, 1 skip, 2 replace")
        }
        var columns: [MetalStringArray] = []
        for i in 0..<Int(count) { columns.append(try sbHandle(handles[i])) }
        return sbEmit(try MetalStringArray.joinElementWise(columns, separator: separator,
                                                           nullHandling: mode,
                                                           nullReplacement: replacement))
    }
}

/// Arrow `extract_regex` (`span == 0`) and `extract_regex_span` (`span == 1`) as a struct column.
///
/// `flags` bit 0 requests case-insensitive matching. The pattern spells its named groups ICU's way,
/// `(?<name>…)`; the Python wrapper rewrites RE2's `(?P<name>…)`.
@_cdecl("am_extract_struct")
public func am_extract_struct(_ a: OpaquePointer?, _ pattern: UnsafePointer<UInt8>?, _ plen: Int64,
                              _ flags: Int32, _ span: Int32,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    let pat = sbText(pattern, plen)
    let ignoreCase = (flags & 1) != 0
    return runStringBytes(out) {
        let s = try sbHandle(a)
        return .structure(span != 0 ? try s.extractRegexSpanStruct(pat, ignoreCase: ignoreCase)
                                    : try s.extractRegexStruct(pat, ignoreCase: ignoreCase))
    }
}
