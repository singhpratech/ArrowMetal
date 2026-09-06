import Foundation

/// Straightforward CPU implementations of every kernel. Used for Float64 columns (Metal has no
/// double type) and as the oracle in tests. Deliberately simple and obviously correct.
public enum CPUReference {
    /// Iterates valid indices efficiently: whole bytes of the validity bitmap are tested at once.
    @inline(__always)
    static func forEachValid<T: ArrowPrimitive>(_ a: MetalArray<T>, _ body: (Int) -> Void) {
        guard let v = a.validity else { for i in 0..<a.length { body(i) }; return }
        let bm = v.typed(UInt8.self)
        var i = 0
        let n = a.length
        while i < n {
            let byte = bm[i >> 3]
            if byte == 0xFF && i + 8 <= n { for j in i..<(i + 8) { body(j) }; i += 8; continue }
            if byte == 0 && i + 8 <= n { i += 8; continue }
            if (byte >> (i & 7)) & 1 == 1 { body(i) }
            i += 1
        }
    }

    public static func sum<T: ArrowPrimitive>(_ a: MetalArray<T>) -> SumResult? {
        if a.validCount == 0 { return nil }
        let p = a.valuePointer
        if T.isFloatingPoint {
            var acc = 0.0
            forEachValid(a) { acc += p[$0].asDouble }
            return .float(acc)
        } else if T.minValue < 0 as T {
            var acc: Int64 = 0
            forEachValid(a) { acc &+= p[$0].asInt64 }
            return .int(acc)
        } else {
            var acc: UInt64 = 0
            forEachValid(a) { acc &+= p[$0].asUInt64 }
            return .uint(acc)
        }
    }

    public static func min<T: ArrowPrimitive>(_ a: MetalArray<T>) -> T? {
        if a.validCount == 0 { return nil }
        var acc = T.maxValue
        let p = a.valuePointer
        forEachValid(a) { acc = Swift.min(acc, p[$0]) }
        return acc
    }

    public static func max<T: ArrowPrimitive>(_ a: MetalArray<T>) -> T? {
        if a.validCount == 0 { return nil }
        var acc = T.minValue
        let p = a.valuePointer
        forEachValid(a) { acc = Swift.max(acc, p[$0]) }
        return acc
    }

    public static func compare<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: CompareOp, scalar: T) throws -> MetalBooleanArray {
        let out = try MetalBooleanArray.allocate(length: a.length, withValidity: a.validity != nil, context: a.context)
        let p = a.valuePointer, o = out.values.mutableTyped(UInt8.self)
        for i in 0..<a.length {
            if a.isValid(i) {
                if let v = out.validity { Bitmap.set(v.mutableTyped(UInt8.self), i) }
                if op.eval(p[i], scalar) { Bitmap.set(o, i) }
            }
        }
        out.recomputeNullCount()
        return out
    }

    public static func compare<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: CompareOp, array b: MetalArray<T>) throws -> MetalBooleanArray {
        let hasV = a.validity != nil || b.validity != nil
        let out = try MetalBooleanArray.allocate(length: a.length, withValidity: hasV, context: a.context)
        let pa = a.valuePointer, pb = b.valuePointer, o = out.values.mutableTyped(UInt8.self)
        for i in 0..<a.length where a.isValid(i) && b.isValid(i) {
            if let v = out.validity { Bitmap.set(v.mutableTyped(UInt8.self), i) }
            if op.eval(pa[i], pb[i]) { Bitmap.set(o, i) }
        }
        out.recomputeNullCount()
        return out
    }

    static func applyAny<T: ArrowPrimitive>(_ op: ArithmeticOp, _ x: T, _ y: T) -> T { T.wrappingApply(op, x, y) }

    public static func arithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: ArithmeticOp, scalar: T) throws -> MetalArray<T> {
        let out = MetalArray<T>(length: a.length, nullCount: a.nullCount, validity: a.validity,
                                values: try MetalArrowBuffer.allocate(byteCount: a.length * T.byteWidth, context: a.context), context: a.context)
        let p = a.valuePointer, o = out.mutableValuePointer
        for i in 0..<a.length { o[i] = applyAny(op, p[i], scalar) }
        return out
    }

    public static func arithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: ArithmeticOp, array b: MetalArray<T>) throws -> MetalArray<T> {
        let hasV = a.validity != nil || b.validity != nil
        let out = try MetalArray<T>.allocate(length: a.length, withValidity: hasV, context: a.context)
        let pa = a.valuePointer, pb = b.valuePointer, o = out.mutableValuePointer
        for i in 0..<a.length {
            o[i] = applyAny(op, pa[i], pb[i])
            if let v = out.validity, a.isValid(i) && b.isValid(i) { Bitmap.set(v.mutableTyped(UInt8.self), i) }
        }
        out.recomputeNullCount()
        return out
    }

    public static func filter<T: ArrowPrimitive>(_ a: MetalArray<T>, _ mask: MetalBooleanArray) throws -> MetalArray<T> {
        var idx: [Int] = []
        for i in 0..<a.length where mask[i] == true { idx.append(i) }
        let out = try MetalArray<T>.allocate(length: idx.count, withValidity: a.validity != nil, context: a.context)
        let p = a.valuePointer, o = out.mutableValuePointer
        for (j, i) in idx.enumerated() {
            o[j] = p[i]
            if let v = out.validity, a.isValid(i) { Bitmap.set(v.mutableTyped(UInt8.self), j) }
        }
        out.recomputeNullCount()
        return out
    }
}
