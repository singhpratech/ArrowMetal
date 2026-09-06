import Foundation
import CArrowABI

// Arrow `binary` ("z") and `large_binary` ("Z"). The layout is exactly utf8's — validity, int32 offsets,
// data bytes — so `MetalStringArray` stores both; an `isBinary` flag decides what the exporter writes.
// Large offsets are narrowed on import, as for large_utf8, so everything downstream stays 32-bit.

/// Marks a string array as holding binary values (exported as "z") and returns it.
@discardableResult
func markBinary(_ s: MetalStringArray) -> MetalStringArray { s.isBinary = true; return s }

extension MetalStringArray {
    /// Bytes of element `i`, or nil when it is null. Works for utf8 and binary alike.
    public func bytes(at i: Int) -> [UInt8]? {
        guard isValid(i) else { return nil }
        let o = offsets.typed(Int32.self), d = data.typed(UInt8.self)
        return Array(UnsafeBufferPointer(start: d + Int(o[i]), count: Int(o[i + 1] - o[i])))
    }
    /// Copies every element out as bytes.
    public func toByteArrays() -> [[UInt8]?] { (0..<length).map { bytes(at: $0) } }

    /// Builds a binary array from byte strings.
    public convenience init(bytes values: [[UInt8]?], context: MetalContext = .shared) throws {
        let n = values.count
        var total = 0
        for v in values { total += v?.count ?? 0 }
        let off = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: context)
        let dat = try MetalArrowBuffer.allocate(byteCount: total, zeroed: false, context: context)
        let hasNulls = values.contains { $0 == nil }
        let bm = hasNulls ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), context: context) : nil
        let op = off.mutableTyped(Int32.self), dp = dat.mutableTyped(UInt8.self)
        var pos = 0, nulls = 0
        for (i, v) in values.enumerated() {
            op[i] = Int32(pos)
            if let v {
                for b in v { dp[pos] = b; pos += 1 }
                if let bm { Bitmap.set(bm.mutableTyped(UInt8.self), i) }
            } else { nulls += 1 }
        }
        op[n] = Int32(pos)
        self.init(length: n, nullCount: nulls, validity: bm, offsets: off, data: dat, context: context)
        self.isBinary = true
    }
}

/// binary / large_binary import. Identical to utf8 apart from the flag on the result.
func importBinaryArray(large: Bool, array: UnsafeMutablePointer<ArrowArray>,
                       context: MetalContext) throws -> ImportResult {
    let r = try importStringArray(large: large, array: array, context: context)
    guard case .string(let s) = r.array else { throw ArrowMetalError.invalidArrowArray("binary import produced \(r.array.arrowFormat)") }
    return ImportResult(array: .binary(markBinary(s)), zeroCopy: r.zeroCopy)
}
