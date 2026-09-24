import Foundation

/// The router's CPU paths: tight, typed, single-threaded loops over the Arrow layout, one generic
/// implementation per shape (reduce, map, map-to-bitmap, compact, low-cardinality dictionary group-by
/// sum). Each produces the same Arrow output as the GPU kernel it stands in for, byte for byte: the
/// same values (including the slots under nulls, which the GPU computes too), the same validity
/// buffer (shared where the GPU shares it), the same null count and the same buffer sizes.
///
/// Floating-point sums reproduce the GPU's summation order exactly (per-thread strided accumulation,
/// the threadgroup tree, then the host combine of the partials), with the software `d_add`'s NaN rule,
/// so they are bit-identical rather than merely close. `CPUReference` stays the tests' oracle.
enum RouterCPU {
    // The entry points carry `@_specialize` for every primitive: callers reach them from generic
    // `MetalArray<T>` methods (and the C ABI through an existential), where the optimiser has no
    // concrete type, and an unspecialised loop runs an order of magnitude slower than the typed one.

    // MARK: bitmap words

    /// Bits `[i, min(i + 64, n))` of `bm` as one little-endian word, `i` a multiple of 64. Full words
    /// are one unaligned load; the tail is assembled byte by byte so nothing past the bitmap is read.
    @inline(__always)
    static func word64(_ bm: UnsafePointer<UInt8>, _ i: Int, _ n: Int) -> UInt64 {
        if i + 64 <= n { return UnsafeRawPointer(bm).loadUnaligned(fromByteOffset: i >> 3, as: UInt64.self) }
        let rem = n - i
        var w: UInt64 = 0
        for b in 0..<((rem + 7) >> 3) { w |= UInt64(bm[(i >> 3) + b]) << UInt64(8 * b) }
        return w & ((1 << UInt64(rem)) - 1)
    }

    @inline(__always)
    static func bit(_ bm: UnsafePointer<UInt8>, _ i: Int) -> Bool { (bm[i >> 3] >> UInt8(i & 7)) & 1 == 1 }

    // MARK: reduce

    /// Folds `f` over the valid values. Invalid slots contribute `identity`, which must be neutral for
    /// `f`, so the inner loop has no branch and vectorises.
    @inline(__always)
    static func reduce<T, A>(_ p: UnsafePointer<T>, _ bm: UnsafePointer<UInt8>?, _ n: Int,
                             identity: T, _ initial: A, _ f: (A, T) -> A) -> A {
        var acc = initial
        guard let bm else {
            for i in 0..<n { acc = f(acc, p[i]) }
            return acc
        }
        var i = 0
        while i + 64 <= n {
            let w = UnsafeRawPointer(bm).loadUnaligned(fromByteOffset: i >> 3, as: UInt64.self)
            if w == ~0 {
                for j in 0..<64 { acc = f(acc, p[i + j]) }
            } else if w != 0 {
                for j in 0..<64 { acc = f(acc, (w >> UInt64(j)) & 1 != 0 ? p[i + j] : identity) }
            }
            i += 64
        }
        if i < n {
            let w = word64(bm, i, n)
            for j in 0..<(n - i) { acc = f(acc, (w >> UInt64(j)) & 1 != 0 ? p[i + j] : identity) }
        }
        return acc
    }

    /// Software `d_add`'s answer where the hardware add produced a NaN: the first NaN operand, quieted;
    /// otherwise (inf + -inf) the positive quiet NaN. Everywhere else the two are the same correctly
    /// rounded IEEE add (DoubleMath.swift, DoubleMathTests).
    @inline(never)
    static func nanOf(_ a: Double, _ b: Double) -> Double {
        if a.isNaN { return Double(bitPattern: a.bitPattern | (1 << 51)) }
        if b.isNaN { return Double(bitPattern: b.bitPattern | (1 << 51)) }
        return Double(bitPattern: 0x7FF8_0000_0000_0000)
    }

    @inline(__always) static func dAdd(_ a: Double, _ b: Double) -> Double {
        let r = a + b
        return r.isNaN ? nanOf(a, b) : r
    }
    /// `d_sub(a, b)` is `d_add(a, -b)` with the sign flipped on the bit pattern, NaN included.
    @inline(__always) static func dSub(_ a: Double, _ b: Double) -> Double {
        let r = a - b
        return r.isNaN ? nanOf(a, Double(bitPattern: b.bitPattern ^ (1 << 63))) : r
    }
    /// `d_mul`: the same NaN rule as `d_add` (0 * inf is the positive quiet NaN).
    @inline(__always) static func dMul(_ a: Double, _ b: Double) -> Double {
        let r = a * b
        return r.isNaN ? nanOf(a, b) : r
    }

    /// `d_from_float`: exact widening; a NaN keeps its payload and is quieted.
    @inline(__always) static func dFromFloat(_ f: Float) -> Double {
        if !f.isNaN { return Double(f) }
        let b = UInt64(f.bitPattern)
        return Double(bitPattern: ((b >> 31) << 63) | 0x7FF0_0000_0000_0000 | (1 << 51) | ((b & 0x7F_FFFF) << 29))
    }

    /// Float sum in the GPU's exact order (KernelSource.reductions `reduce_sum` with the software
    /// double add, then `finaliseSum`). Thread `t` of `groups * 256` adds the 4-element blocks
    /// `t, t + grid, ...` then tail element `t`; each threadgroup folds its 256 accumulators as a tree;
    /// the host adds the group partials in order.
    static func floatSumGPUOrder(_ n: Int, _ bm: UnsafePointer<UInt8>?, _ load: (Int) -> Double) -> Double {
        let tg = Dispatch.threadgroupSize
        let groups = Swift.max(1, Swift.min(2048, (n + tg - 1) / tg))
        let grid = groups * tg
        let n4 = n & ~3
        let acc = UnsafeMutablePointer<Double>.allocate(capacity: grid)
        defer { acc.deallocate() }
        acc.initialize(repeating: 0, count: grid)
        var t = 0
        var i = 0
        while i < n4 {
            var s = acc[t]
            if let bm {
                let vb = (bm[i >> 3] >> UInt8(i & 7)) & 0xF
                if vb & 1 != 0 { s = dAdd(s, load(i)) }
                if vb & 2 != 0 { s = dAdd(s, load(i + 1)) }
                if vb & 4 != 0 { s = dAdd(s, load(i + 2)) }
                if vb & 8 != 0 { s = dAdd(s, load(i + 3)) }
            } else {
                s = dAdd(dAdd(dAdd(dAdd(s, load(i)), load(i + 1)), load(i + 2)), load(i + 3))
            }
            acc[t] = s
            t += 1; if t == grid { t = 0 }
            i += 4
        }
        for k in n4..<n where bm == nil || bit(bm!, k) { acc[k - n4] = dAdd(acc[k - n4], load(k)) }
        var total = 0.0
        for g in 0..<groups {
            let base = acc + g * tg
            var s = tg / 2
            while s > 0 {
                for l in 0..<s { base[l] = dAdd(base[l], base[l + s]) }
                s >>= 1
            }
            total += base[0]           // finaliseSum: a plain host add, in group order
        }
        return total
    }

    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func sum<T: ArrowPrimitive>(_ a: MetalArray<T>) -> SumResult? {
        if a.validCount == 0 { return nil }
        let n = a.knownLength
        let vals = a.values, vld = a.validity
        return withExtendedLifetime((vals, vld)) { () -> SumResult in
            let bm = vld?.typed(UInt8.self)
            let raw = vals.contents
            if T.self == Double.self {
                let p = raw.assumingMemoryBound(to: Double.self)
                return .float(floatSumGPUOrder(n, bm) { p[$0] })
            }
            if T.self == Float.self {
                let p = raw.assumingMemoryBound(to: Float.self)
                return .float(floatSumGPUOrder(n, bm) { dFromFloat(p[$0]) })
            }
            let p = raw.assumingMemoryBound(to: T.self)
            if T.minValue < 0 as T {
                return .int(reduce(p, bm, n, identity: 0 as T, Int64(0)) { $0 &+ $1.asInt64 })
            }
            return .uint(reduce(p, bm, n, identity: 0 as T, UInt64(0)) { $0 &+ $1.asUInt64 })
        }
    }

    /// Min (`isMin`) or max of the valid values. Integers compare directly; Float64 compares the GPU's
    /// order-preserving keys (NaN skipped, -0 and +0 one key), so a column of NaNs gives nil as it does
    /// there. Float32 has no CPU path (see `minMaxUnavailable`).
    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func minMax<T: ArrowPrimitive>(_ a: MetalArray<T>, isMin: Bool) -> T? {
        if a.validCount == 0 { return nil }
        let n = a.knownLength
        let vals = a.values, vld = a.validity
        return withExtendedLifetime((vals, vld)) { () -> T? in
            let bm = vld?.typed(UInt8.self)
            if T.self == Double.self {
                let q = vals.contents.assumingMemoryBound(to: Int64.self)
                let ident = isMin ? Int64.max : Int64.min
                @inline(__always) func key(_ b: Int64) -> Int64 {
                    let mag = b & 0x7FFF_FFFF_FFFF_FFFF
                    if mag > 0x7FF0_0000_0000_0000 { return ident }             // NaN: skipped
                    if mag == 0 { return 0 }
                    return b ^ Int64(bitPattern: UInt64(bitPattern: b >> 63) >> 1)
                }
                // Null slots feed a quiet NaN pattern, whose key is `ident`: neutral for both.
                let nanBits = Int64(bitPattern: 0x7FF8_0000_0000_0000)
                let k = isMin ? reduce(q, bm, n, identity: nanBits, ident) { Swift.min($0, key($1)) }
                              : reduce(q, bm, n, identity: nanBits, ident) { Swift.max($0, key($1)) }
                // `ident` is the key of a NaN pattern, never of a value, so it means "nothing counted".
                return k == ident ? nil : (Dispatch.doubleFromKey(k) as! T)
            }
            let p = vals.typed(T.self)
            return isMin ? reduce(p, bm, n, identity: T.maxValue, T.maxValue) { Swift.min($0, $1) }
                         : reduce(p, bm, n, identity: T.minValue, T.minValue) { Swift.max($0, $1) }
        }
    }

    static func minMaxUnavailable<T: ArrowPrimitive>(_: T.Type) -> String? {
        T.self == Float.self ? "no bit-exact CPU loop for float32 min/max (the GPU compares in hardware float, which flushes subnormals)" : nil
    }

    // MARK: map-to-bitmap

    /// Packs `pred(i)` for `i < n` into 32-bit words, bits past `n` zero: the GPU compare kernels' layout.
    @inline(__always)
    static func mapToBitmap(_ n: Int, _ out: UnsafeMutablePointer<UInt32>, _ pred: (Int) -> Bool) {
        var base = 0
        var w = 0
        while base + 32 <= n {
            var bits: UInt32 = 0
            for j in 0..<32 { bits |= (pred(base + j) ? 1 : 0) << UInt32(j) }
            out[w] = bits
            base += 32; w += 1
        }
        if base < n {
            var bits: UInt32 = 0
            for j in 0..<(n - base) { bits |= (pred(base + j) ? 1 : 0) << UInt32(j) }
            out[w] = bits
        }
    }

    @inline(__always)
    static func comparePacked<T: ArrowPrimitive>(_ n: Int, _ out: UnsafeMutablePointer<UInt32>, _ op: CompareOp,
                                                 _ x: (Int) -> T, _ y: (Int) -> T) {
        switch op {
        case .eq: mapToBitmap(n, out) { x($0) == y($0) }
        case .ne: mapToBitmap(n, out) { x($0) != y($0) }
        case .lt: mapToBitmap(n, out) { x($0) < y($0) }
        case .le: mapToBitmap(n, out) { x($0) <= y($0) }
        case .gt: mapToBitmap(n, out) { x($0) > y($0) }
        case .ge: mapToBitmap(n, out) { x($0) >= y($0) }
        }
    }

    /// `compare(op, scalar)`: every slot is compared (the GPU does not look at validity either) and the
    /// input's validity is shared, as on the GPU. IEEE semantics for floats: NaN compares false except
    /// `ne`, -0 == +0, subnormals exact — what the GPU's key-based float kernels compute.
    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func compare<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: CompareOp, _ scalar: T) throws -> MetalBooleanArray {
        let n = a.knownLength
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: a.context)
        let vals = a.values
        withExtendedLifetime(vals) {
            let p = vals.typed(T.self)
            comparePacked(n, out.mutableTyped(UInt32.self), op, { p[$0] }, { _ in scalar })
        }
        return MetalBooleanArray(length: n, nullCount: a._nullCount, validity: a.validity, values: out, context: a.context)
    }

    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func compare<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: CompareOp, _ b: MetalArray<T>) throws -> MetalBooleanArray {
        let n = a.knownLength
        let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: a.context)
        let va = a.values, vb = b.values
        withExtendedLifetime((va, vb)) {
            let p = va.typed(T.self), q = vb.typed(T.self)
            comparePacked(n, out.mutableTyped(UInt32.self), op, { p[$0] }, { q[$0] })
        }
        let v = try andValidity(a.validity, b.validity, n, a.context)
        let res = MetalBooleanArray(length: n, nullCount: 0, validity: v, values: out, context: a.context)
        res.recomputeNullCount()
        return res
    }

    /// `BitmapOps.combineValidity` on the host: nil and nil is nil, one bitmap is shared, two are ANDed
    /// word by word into a fresh bitmap of the same size the GPU allocates.
    static func andValidity(_ a: MetalArrowBuffer?, _ b: MetalArrowBuffer?, _ n: Int, _ ctx: MetalContext) throws -> MetalArrowBuffer? {
        switch (a, b) {
        case (nil, nil): return nil
        case (let x?, nil): return x
        case (nil, let y?): return y
        case (let x?, let y?):
            let out = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: n), zeroed: false, context: ctx)
            withExtendedLifetime((x, y)) {
                let p = x.typed(UInt8.self), q = y.typed(UInt8.self)
                let o = out.mutableContents
                var i = 0
                while i < n {
                    let w = word64(p, i, n) & word64(q, i, n)
                    let bytes = Swift.min(8, (n - i + 7) >> 3)
                    withUnsafeBytes(of: w) { o.advanced(by: i >> 3).copyMemory(from: $0.baseAddress!, byteCount: bytes) }
                    i += 64
                }
            }
            return out
        }
    }

    // MARK: map

    static func arithmeticUnavailable<T: ArrowPrimitive>(_: T.Type, _ op: ArithmeticOp) -> String? {
        if op == .div { return "divide is not routed" }
        if T.self == Float.self { return "no bit-exact CPU loop for float32 arithmetic (the GPU flushes subnormals)" }
        return nil
    }

    /// Float64 add/subtract/multiply with the software kernels' NaN rule (`d_add`, `d_sub`, `d_mul`).
    @inline(__always)
    static func mapDouble(_ n: Int, _ out: UnsafeMutablePointer<Double>, _ op: ArithmeticOp,
                          _ x: (Int) -> Double, _ y: (Int) -> Double) {
        switch op {
        case .add: for i in 0..<n { out[i] = dAdd(x(i), y(i)) }
        case .sub: for i in 0..<n { out[i] = dSub(x(i), y(i)) }
        case .mul: for i in 0..<n { out[i] = dMul(x(i), y(i)) }
        case .div: preconditionFailure("divide is not routed")
        }
    }

    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func arithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: ArithmeticOp, _ scalar: T) throws -> MetalArray<T> {
        let n = a.knownLength
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: a.context)
        let vals = a.values
        withExtendedLifetime(vals) {
            if T.self == Double.self {
                let p = vals.typed(Double.self), s = scalar as! Double
                mapDouble(n, out.mutableTyped(Double.self), op, { p[$0] }, { _ in s })
            } else {
                let p = vals.typed(T.self)
                mapIntegers(n, out.mutableTyped(T.self), op, { p[$0] }, { _ in scalar })
            }
        }
        return MetalArray<T>(length: n, nullCount: a._nullCount, validity: a.validity, values: out, context: a.context)
    }

    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func arithmetic<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: ArithmeticOp, _ b: MetalArray<T>) throws -> MetalArray<T> {
        let n = a.knownLength
        let out = try MetalArrowBuffer.allocate(byteCount: n * T.byteWidth, zeroed: false, context: a.context)
        let va = a.values, vb = b.values
        withExtendedLifetime((va, vb)) {
            if T.self == Double.self {
                let p = va.typed(Double.self), q = vb.typed(Double.self)
                mapDouble(n, out.mutableTyped(Double.self), op, { p[$0] }, { q[$0] })
            } else {
                let p = va.typed(T.self), q = vb.typed(T.self)
                mapIntegers(n, out.mutableTyped(T.self), op, { p[$0] }, { q[$0] })
            }
        }
        let v = try andValidity(a.validity, b.validity, n, a.context)
        let res = MetalArray<T>(length: n, nullCount: 0, validity: v, values: out, context: a.context)
        res.recomputeNullCount()
        return res
    }

    /// Wrapping integer add/subtract/multiply (`T.wrappingApply`, the GPU's modular arithmetic).
    @inline(__always)
    static func mapIntegers<T: ArrowPrimitive>(_ n: Int, _ out: UnsafeMutablePointer<T>, _ op: ArithmeticOp,
                                               _ x: (Int) -> T, _ y: (Int) -> T) {
        switch op {
        case .add: for i in 0..<n { out[i] = T.wrappingApply(.add, x(i), y(i)) }
        case .sub: for i in 0..<n { out[i] = T.wrappingApply(.sub, x(i), y(i)) }
        case .mul: for i in 0..<n { out[i] = T.wrappingApply(.mul, x(i), y(i)) }
        case .div: preconditionFailure("divide is not routed")
        }
    }

    // MARK: compact

    /// Stream compaction by 64-bit selection words (`sel(i)` for word start `i`, bits past `n` clear):
    /// the selected values in order, and when the input has validity (and anything was selected) an
    /// output bitmap carrying the selected rows' validity — the GPU `compact`'s exact result.
    @inline(__always)
    static func compact<T: ArrowPrimitive>(_ a: MetalArray<T>, _ sel: (Int) -> UInt64) throws -> MetalArray<T> {
        let n = a.knownLength
        let wordCount = (n + 63) >> 6
        let words = UnsafeMutablePointer<UInt64>.allocate(capacity: Swift.max(wordCount, 1))
        defer { words.deallocate() }
        var count = 0
        for w in 0..<wordCount { let x = sel(w << 6); words[w] = x; count += x.nonzeroBitCount }
        let vals = a.values, vld = a.validity
        let outValues = try MetalArrowBuffer.allocate(byteCount: count * T.byteWidth, zeroed: false, context: a.context)
        let outValidity = (vld != nil && count > 0)
            ? try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: count), context: a.context) : nil
        withExtendedLifetime((vals, vld)) {
            let p = vals.typed(T.self), o = outValues.mutableTyped(T.self)
            var k = 0
            if let vld, let ov = outValidity {
                let bm = vld.typed(UInt8.self), ob = ov.mutableTyped(UInt8.self)
                for w in 0..<wordCount {
                    var x = words[w]
                    let base = w << 6
                    while x != 0 {
                        let i = base + x.trailingZeroBitCount
                        o[k] = p[i]
                        if bit(bm, i) { Bitmap.set(ob, k) }
                        k += 1
                        x &= x &- 1
                    }
                }
            } else {
                for w in 0..<wordCount {
                    var x = words[w]
                    let base = w << 6
                    while x != 0 {
                        o[k] = p[base + x.trailingZeroBitCount]
                        k += 1
                        x &= x &- 1
                    }
                }
            }
        }
        let res = MetalArray<T>(length: count, nullCount: 0, validity: outValidity, values: outValues, context: a.context)
        res.recomputeNullCount()
        return res
    }

    /// `filter(mask)`: selected where the mask is true and valid (null selection behaviour "drop").
    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    @_specialize(where T == Float)
    @_specialize(where T == Double)
    static func filter<T: ArrowPrimitive>(_ a: MetalArray<T>, _ mask: MetalBooleanArray) throws -> MetalArray<T> {
        let n = a.knownLength
        let mv = mask.values, mvld = mask.validity
        return try withExtendedLifetime((mv, mvld)) {
            let m = mv.typed(UInt8.self)
            if let mvld {
                let v = mvld.typed(UInt8.self)
                return try compact(a) { word64(m, $0, n) & word64(v, $0, n) }
            }
            return try compact(a) { word64(m, $0, n) }
        }
    }

    static func filterWhereUnavailable<T: ArrowPrimitive>(_: T.Type) -> String? {
        T.self == Float.self ? "no bit-exact CPU loop for the fused float32 predicate (the GPU compares in hardware float)" : nil
    }

    /// Fused `filter(where: op, scalar)` for integer columns: the predicate and the validity build each
    /// selection word, then the same compaction.
    @_specialize(where T == Int8)
    @_specialize(where T == UInt8)
    @_specialize(where T == Int16)
    @_specialize(where T == UInt16)
    @_specialize(where T == Int32)
    @_specialize(where T == UInt32)
    @_specialize(where T == Int64)
    @_specialize(where T == UInt64)
    static func filterWhere<T: ArrowPrimitive>(_ a: MetalArray<T>, _ op: CompareOp, _ scalar: T) throws -> MetalArray<T> {
        let n = a.knownLength
        let vals = a.values, vld = a.validity
        // The predicate is packed first (the compare loop), then ANDed with the validity per word.
        let words = (n + 63) >> 6
        let pred = UnsafeMutablePointer<UInt64>.allocate(capacity: Swift.max(words, 1))
        defer { pred.deallocate() }
        if words > 0 { pred[words - 1] = 0 }
        return try withExtendedLifetime((vals, vld)) {
            let p = vals.typed(T.self)
            comparePacked(n, UnsafeMutableRawPointer(pred).assumingMemoryBound(to: UInt32.self), op, { p[$0] }, { _ in scalar })
            let packed = UnsafeRawPointer(pred).assumingMemoryBound(to: UInt8.self)
            if let vld {
                let bm = vld.typed(UInt8.self)
                return try compact(a) { word64(packed, $0, n) & word64(bm, $0, n) }
            }
            return try compact(a) { word64(packed, $0, n) }
        }
    }

    // MARK: low-cardinality dictionary group-by sum

    static func groupBySumUnavailable(keyCount: Int) -> String? {
        keyCount > Router.groupBySumMaxKeys ? "more than \(Router.groupBySumMaxKeys) keys" : nil
    }

    /// `GroupBy.sumUnsigned`: the same loop, kept unsigned. A wrapping 64-bit sum has the same bits
    /// in Int64 and UInt64, and the buffers are the ones `sumUnsigned` allocates (`keyCount` words and a
    /// `keyCount`-bit validity bitmap), so the result is the GPU's byte for byte.
    static func groupBySumUnsigned<K: ArrowIndex>(keys: MetalArray<K>, keyCount: Int,
                                                  values: MetalArray<UInt64>) throws -> MetalArray<UInt64> {
        let r = try groupBySum(keys: keys, keyCount: keyCount, values: values)
        return MetalArray<UInt64>(length: keyCount, nullCount: r.nullCount, validity: r.validity, values: r.values, context: r.context)
    }

    /// `GroupBy.sum` over integer values: one pass, a `keyCount`-slot table of wrapping 64-bit sums and
    /// counts. Null keys, keys outside `0 ..< keyCount` and null values are skipped; a key that counted
    /// nothing is null with a 0 in its slot — the GPU accumulate + finalize result.
    @_specialize(where K == Int32, T == Int8)
    @_specialize(where K == Int32, T == UInt8)
    @_specialize(where K == Int32, T == Int16)
    @_specialize(where K == Int32, T == UInt16)
    @_specialize(where K == Int32, T == Int32)
    @_specialize(where K == Int32, T == UInt32)
    @_specialize(where K == Int32, T == Int64)
    @_specialize(where K == Int32, T == UInt64)
    @_specialize(where K == Int64, T == Int8)
    @_specialize(where K == Int64, T == UInt8)
    @_specialize(where K == Int64, T == Int16)
    @_specialize(where K == Int64, T == UInt16)
    @_specialize(where K == Int64, T == Int32)
    @_specialize(where K == Int64, T == UInt32)
    @_specialize(where K == Int64, T == Int64)
    @_specialize(where K == Int64, T == UInt64)
    @_specialize(where K == UInt32, T == Int8)
    @_specialize(where K == UInt32, T == UInt8)
    @_specialize(where K == UInt32, T == Int16)
    @_specialize(where K == UInt32, T == UInt16)
    @_specialize(where K == UInt32, T == Int32)
    @_specialize(where K == UInt32, T == UInt32)
    @_specialize(where K == UInt32, T == Int64)
    @_specialize(where K == UInt32, T == UInt64)
    static func groupBySum<K: ArrowIndex, T: ArrowPrimitive>(keys: MetalArray<K>, keyCount: Int,
                                                             values: MetalArray<T>) throws -> MetalArray<Int64> where T: FixedWidthInteger {
        let n = keys.knownLength
        let ctx = values.context
        let out = try MetalArrowBuffer.allocate(byteCount: keyCount * 8, context: ctx)
        let valid = try MetalArrowBuffer.allocate(byteCount: Bitmap.byteCount(bits: keyCount), context: ctx)
        let counts = UnsafeMutablePointer<UInt32>.allocate(capacity: keyCount)
        defer { counts.deallocate() }
        counts.initialize(repeating: 0, count: keyCount)
        let kv = keys.values, kvld = keys.validity, vv = values.values, vvld = values.validity
        withExtendedLifetime((kv, kvld, vv, vvld)) {
            let kp = kv.typed(K.self), vp = vv.typed(T.self)
            let sums = out.mutableTyped(Int64.self)
            let kb = kvld?.typed(UInt8.self), vb = vvld?.typed(UInt8.self)
            let kc = Int64(keyCount)
            @inline(__always) func add(_ i: Int) {
                let k = kp[i].asInt64
                if k < 0 || k >= kc { return }
                sums[Int(k)] &+= Int64(truncatingIfNeeded: vp[i])
                counts[Int(k)] &+= 1
            }
            if kb == nil && vb == nil {
                for i in 0..<n { add(i) }
            } else {
                var i = 0
                while i < n {
                    var w: UInt64 = i + 64 <= n ? ~0 : (1 << UInt64(n - i)) - 1
                    if let kb { w &= word64(kb, i, n) }
                    if let vb { w &= word64(vb, i, n) }
                    while w != 0 { add(i + w.trailingZeroBitCount); w &= w &- 1 }
                    i += 64
                }
            }
            let vbits = valid.mutableTyped(UInt8.self)
            for k in 0..<keyCount where counts[k] != 0 { Bitmap.set(vbits, k) }
        }
        let res = MetalArray<Int64>(length: keyCount, nullCount: 0, validity: valid, values: out, context: ctx)
        res.recomputeNullCount()
        return res
    }
}
