import Foundation
import Metal

// The group-by terminal: one privatised table of 32-bit atomics per threadgroup (or device-wide tables
// when the key space is large), fed by the same fused expression body as every other shape.

/// How one group-by aggregate accumulates in a table of 32-bit atomics.
enum GBKind {
    case count, sumI, sumU, sumF32, minI, maxI, minU, maxU, minF, maxF

    /// Initial table word.
    var initWord: UInt32 {
        switch self {
        case .count, .sumI, .sumU, .sumF32: return 0
        case .minI: return UInt32(bitPattern: Int32.max)
        case .maxI: return UInt32(bitPattern: Int32.min)
        case .minU: return UInt32.max
        case .maxU: return 0
        case .minF: return UInt32(bitPattern: fkey32(0x7F80_0000))       // +inf
        case .maxF: return UInt32(bitPattern: fkey32(0xFF80_0000))       // -inf
        }
    }
    /// The MSL identity used when merging the per-threadgroup tables.
    var mergeIdentity: String {
        switch self {
        case .minI, .minF: return "INT_MAX"
        case .maxI, .maxF: return "INT_MIN"
        case .minU: return "UINT_MAX"
        case .maxU: return "0u"
        default: return "0"
        }
    }
}

/// The order-preserving float32 key, in Swift (mirrors `f_key32` in MSL).
func fkey32(_ b: UInt32) -> Int32 {
    if b & 0x7FFF_FFFF == 0 { return 0 }
    let k = Int32(bitPattern: b)
    return k ^ Int32(bitPattern: UInt32(bitPattern: k >> 31) >> 1)
}
func fromFkey32(_ k: Int32) -> Float {
    let bits = k < 0 ? UInt32(bitPattern: k ^ 0x7FFF_FFFF) : UInt32(bitPattern: k)
    return Float(bitPattern: bits)
}

extension ExprCompiler {
    static var gbMaxPrivateSlots: Int { 2048 }

    static func gbKind(_ op: ExprAggregate.Op, _ t: ExprType?, _ name: String) throws -> GBKind {
        guard let t else { return .count }
        switch op {
        case .count: return .count
        case .sum, .mean:
            if t == .float64 { throw ExprError.unsupported("group_by \(op.rawValue) \"\(name)\" over float64; cast the expression to float32") }
            if t == .float32 { return .sumF32 }
            guard t.isInteger else { throw ExprError.unsupported("group_by \(op.rawValue) \"\(name)\" over \(t.rawValue)") }
            return t.isSigned ? .sumI : .sumU
        case .min, .max:
            let isMin = op == .min
            if t == .float32 { return isMin ? .minF : .maxF }
            guard t.isInteger, t.bitWidth <= 32 else {
                throw ExprError.unsupported("group_by \(op.rawValue) \"\(name)\" needs a 32-bit or narrower integer or a float32 expression (Metal has no 64-bit atomics); cast first")
            }
            return t.isSigned ? (isMin ? .minI : .maxI) : (isMin ? .minU : .maxU)
        }
    }

    static func runGroupBy(_ q: ExprQuery, _ aggs: [ExprAggregate], schema: [String: ExprColumnInfo],
                           key: String, inputs: [String: Input], n: Int,
                           lengthBuffer: MetalArrowBuffer?, ctx: MetalContext) throws -> ExprQueryResult {
        let K = q.keyCount
        guard K > 0, K <= Int(UInt32.max) / 8 else { throw ExprError.invalid("group_by keyCount out of range: \(K)") }

        let c = try cached("groupby|\(q.canonical)|\(key)") {
            let comp = Compiled()
            let em = ExprEmitter(schema: schema)
            var slots: [ExprSlot] = []
            var predIndex: Int? = nil
            if let f = q.filter {
                let s = try em.emit(f, hint: .boolean)
                guard s.type == .boolean else { throw ExprError.unsupported("filter predicate must be boolean, got \(s.type.rawValue)") }
                predIndex = slots.count
                slots.append(s)
            }
            let keySlot = try em.emit(q.groupKey!)
            guard keySlot.type.isInteger else {
                throw ExprError.unsupported("group_by key must be an integer expression, got \(keySlot.type.rawValue)")
            }
            let keyIndex = slots.count
            slots.append(keySlot)

            var plans: [AggPlan] = []
            for a in aggs {
                if a.op == .count && a.expr == nil {
                    plans.append(AggPlan(op: .count, name: a.name, kind: .countRows, valueType: nil, slot: nil))
                    continue
                }
                guard let e = a.expr else { throw ExprError.invalid("\(a.op.rawValue) \"\(a.name)\" needs an expression") }
                let s = try em.emit(e)
                guard s.type.isNumeric else { throw ExprError.unsupported("group_by \(a.op.rawValue) of a \(s.type.rawValue) expression") }
                plans.append(AggPlan(op: a.op, name: a.name, kind: .countValues, valueType: s.type, slot: slots.count))
                slots.append(s)
            }
            comp.aggs = plans
            comp.leaves = em.leaves
            var gkinds: [GBKind] = []
            for p in plans { gkinds.append(try gbKind(p.op, p.valueType, p.name)) }
            comp.gbKinds = gkinds

            // The accumulate body, generated once per address space.
            func accumulate(space: String, lo: (String) -> String, hi: (String) -> String, cnt: (String) -> String) -> String {
                var s = ""
                for (k, p) in plans.enumerated() {
                    let slot = "\(k)u * K + kk"
                    let kind = gkinds[k]
                    var inner = ""
                    switch kind {
                    case .count: inner = ""
                    case .sumI, .sumU:
                        inner += "            long sv = (long)O\(p.slot!);\n"
                        inner += "            uint vlo = (uint)sv; uint vhi = (uint)((ulong)sv >> 32);\n"
                        inner += "            uint old = atomic_fetch_add_explicit(\(lo(slot)), vlo, memory_order_relaxed);\n"
                        inner += "            atomic_fetch_add_explicit(\(hi(slot)), vhi + ((old + vlo < old) ? 1u : 0u), memory_order_relaxed);\n"
                    case .sumF32:
                        inner += "            float fv = O\(p.slot!);\n"
                        inner += "            uint old = atomic_load_explicit(\(lo(slot)), memory_order_relaxed);\n"
                        inner += "            while (!atomic_compare_exchange_weak_explicit(\(lo(slot)), &old, as_type<uint>(as_type<float>(old) + fv), memory_order_relaxed, memory_order_relaxed)) {}\n"
                    case .minI, .maxI:
                        let f = kind == .minI ? "min" : "max"
                        inner += "            atomic_fetch_\(f)_explicit((\(space) atomic_int*)\(lo(slot)), (int)O\(p.slot!), memory_order_relaxed);\n"
                    case .minU, .maxU:
                        let f = kind == .minU ? "min" : "max"
                        inner += "            atomic_fetch_\(f)_explicit(\(lo(slot)), (uint)O\(p.slot!), memory_order_relaxed);\n"
                    case .minF, .maxF:
                        let f = kind == .minF ? "min" : "max"
                        inner += "            atomic_fetch_\(f)_explicit((\(space) atomic_int*)\(lo(slot)), f_key32(as_type<uint>(O\(p.slot!))), memory_order_relaxed);\n"
                    }
                    if let sl = p.slot {
                        var g = "O\(sl)k"
                        if kind == .minF || kind == .maxF { g += " && !f_isnan32(as_type<uint>(O\(sl)))" }
                        s += "        if (\(g)) {\n"
                        s += "            atomic_fetch_add_explicit(\(cnt(slot)), 1u, memory_order_relaxed);\n"
                        s += inner
                        s += "        }\n"
                    } else {
                        s += "        atomic_fetch_add_explicit(\(cnt(slot)), 1u, memory_order_relaxed);\n"
                    }
                }
                return s
            }

            var rowGuard = ""
            if let pi = predIndex { rowGuard += "        if (!(O\(pi) && O\(pi)k)) continue;\n" }
            rowGuard += "        if (!O\(keyIndex)k) continue;\n"
            rowGuard += "        long kraw = (long)O\(keyIndex);\n"
            rowGuard += "        if (kraw < 0 || kraw >= (long)K) continue;\n"
            rowGuard += "        uint kk = (uint)kraw;\n"

            var (params, bindings, idx) = leafParams(em.leaves, from: 0)
            comp.bindings = bindings
            comp.nPtrIndex = idx
            params.append("device const uint* nPtr [[buffer(\(idx))]]"); idx += 1
            comp.extra["K"] = idx
            params.append("constant uint& K [[buffer(\(idx))]]"); idx += 1
            comp.extra["chunk"] = idx
            params.append("constant uint& chunk [[buffer(\(idx))]]"); idx += 1
            comp.extra["partials"] = idx
            params.append("device ulong* partials [[buffer(\(idx))]]"); idx += 1
            comp.extra["pcounts"] = idx
            params.append("device uint* pcounts [[buffer(\(idx))]]"); idx += 1
            let devStart = idx
            comp.extra["dlo"] = devStart
            comp.extra["dhi"] = devStart + 1
            comp.extra["dcnt"] = devStart + 2
            var devParams = params
            devParams.append("device atomic_uint* dlo [[buffer(\(devStart))]]")
            devParams.append("device atomic_uint* dhi [[buffer(\(devStart + 1))]]")
            devParams.append("device atomic_uint* dcnt [[buffer(\(devStart + 2))]]")

            let call = scalarCall(em, outputs: slots.count, wordValidity: false)
            let nAgg = plans.count

            var src = prelude(em)
            src += em.rowFunction(name: "am_row", outputs: slots)
            src += "\n#define AM_SLOTS \(gbMaxPrivateSlots)u\n"

            src += "\nkernel void am_gb_priv(" + params.joined(separator: ", ")
            src += ",\n                       uint lid [[thread_index_in_threadgroup]],\n"
            src += "                       uint tgid [[threadgroup_position_in_grid]]) {\n"
            src += "    threadgroup atomic_uint tlo[AM_SLOTS];\n"
            src += "    threadgroup atomic_uint thi[AM_SLOTS];\n"
            src += "    threadgroup atomic_uint tcnt[AM_SLOTS];\n"
            src += "    uint n = *nPtr;\n"
            src += "    uint slots = K * \(nAgg)u;\n"
            src += "    for (uint s = lid; s < slots; s += TG) {\n"
            src += "        uint a = s / K;\n        uint iv = 0u;\n"
            for (k, kind) in gkinds.enumerated() where kind.initWord != 0 {
                src += "        if (a == \(k)u) iv = \(kind.initWord)u;\n"
            }
            src += "        atomic_store_explicit(&tlo[s], iv, memory_order_relaxed);\n"
            src += "        atomic_store_explicit(&thi[s], 0u, memory_order_relaxed);\n"
            src += "        atomic_store_explicit(&tcnt[s], 0u, memory_order_relaxed);\n    }\n"
            src += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            src += "    uint start = tgid * chunk;\n    uint end = min(n, start + chunk);\n"
            src += declareOutputs(slots, indent: "    ")
            src += "    for (uint i = start + lid; i < end; i += TG) {\n"
            src += "        \(call)\n"
            src += rowGuard
            src += accumulate(space: "threadgroup", lo: { "&tlo[\($0)]" }, hi: { "&thi[\($0)]" }, cnt: { "&tcnt[\($0)]" })
            src += "    }\n"
            src += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            src += "    for (uint s = lid; s < slots; s += TG) {\n"
            src += "        uint l = atomic_load_explicit(&tlo[s], memory_order_relaxed);\n"
            src += "        uint h = atomic_load_explicit(&thi[s], memory_order_relaxed);\n"
            src += "        partials[tgid * slots + s] = ((ulong)h << 32) | (ulong)l;\n"
            src += "        pcounts[tgid * slots + s] = atomic_load_explicit(&tcnt[s], memory_order_relaxed);\n    }\n}\n"

            src += "\nkernel void am_gb_dev(" + devParams.joined(separator: ", ")
            src += ",\n                      uint lid [[thread_index_in_threadgroup]],\n"
            src += "                      uint tgid [[threadgroup_position_in_grid]]) {\n"
            src += "    uint n = *nPtr;\n"
            src += "    uint start = tgid * chunk;\n    uint end = min(n, start + chunk);\n"
            src += declareOutputs(slots, indent: "    ")
            src += "    for (uint i = start + lid; i < end; i += TG) {\n"
            src += "        \(call)\n"
            src += rowGuard
            src += accumulate(space: "device", lo: { "&dlo[\($0)]" }, hi: { "&dhi[\($0)]" }, cnt: { "&dcnt[\($0)]" })
            src += "    }\n}\n"

            src += "\nkernel void am_gb_finalize(device const ulong* partials [[buffer(0)]],\n"
            src += "                           device const uint* pcounts [[buffer(1)]],\n"
            src += "                           constant uint& K [[buffer(2)]],\n"
            src += "                           constant uint& numTG [[buffer(3)]],\n"
            src += "                           device ulong* out [[buffer(4)]],\n"
            src += "                           device ulong* counts [[buffer(5)]],\n"
            src += "                           uint s [[thread_position_in_grid]]) {\n"
            src += "    uint slots = K * \(nAgg)u;\n    if (s >= slots) return;\n    uint a = s / K;\n"
            src += "    ulong c = 0ul;\n    for (uint t = 0; t < numTG; t++) c += (ulong)pcounts[t * slots + s];\n"
            src += "    counts[s] = c;\n    out[s] = 0ul;\n"
            for (k, kind) in gkinds.enumerated() {
                src += "    if (a == \(k)u) {\n"
                switch kind {
                case .count: src += "        out[s] = 0ul;\n"
                case .sumI, .sumU:
                    src += "        ulong acc = 0ul;\n        for (uint t = 0; t < numTG; t++) acc += partials[t * slots + s];\n        out[s] = acc;\n"
                case .sumF32:
                    src += "        float acc = 0.0f;\n        for (uint t = 0; t < numTG; t++) if (pcounts[t * slots + s]) acc += as_type<float>((uint)partials[t * slots + s]);\n        out[s] = (ulong)as_type<uint>(acc);\n"
                case .minI, .maxI, .minF, .maxF:
                    let f = (kind == .minI || kind == .minF) ? "min" : "max"
                    src += "        int m = \(kind.mergeIdentity);\n        for (uint t = 0; t < numTG; t++) if (pcounts[t * slots + s]) m = \(f)(m, (int)(uint)partials[t * slots + s]);\n        out[s] = (ulong)(uint)m;\n"
                case .minU, .maxU:
                    let f = kind == .minU ? "min" : "max"
                    src += "        uint m = \(kind.mergeIdentity);\n        for (uint t = 0; t < numTG; t++) if (pcounts[t * slots + s]) m = \(f)(m, (uint)partials[t * slots + s]);\n        out[s] = (ulong)m;\n"
                }
                src += "    }\n"
            }
            src += "}\n"

            src += "\nkernel void am_gb_pack(device const uint* lo [[buffer(0)]], device const uint* hi [[buffer(1)]],\n"
            src += "                       device const uint* cnt [[buffer(2)]], constant uint& K [[buffer(3)]],\n"
            src += "                       device ulong* out [[buffer(4)]], device ulong* counts [[buffer(5)]],\n"
            src += "                       uint s [[thread_position_in_grid]]) {\n"
            src += "    uint slots = K * \(nAgg)u;\n    if (s >= slots) return;\n"
            src += "    out[s] = ((ulong)hi[s] << 32) | (ulong)lo[s];\n    counts[s] = (ulong)cnt[s];\n}\n"
            comp.source = src
            return comp
        }

        let kinds = c.gbKinds
        let valueTypes = c.aggs.map { $0.valueType }
        let nAgg = c.aggs.count
        let slots = K * nAgg
        let usePrivate = slots <= gbMaxPrivateSlots
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let cap = usePrivate ? (slots > 256 ? 256 : 1024) : 4096
        let numTG = Swift.max(1, Swift.min(cap, (n + 4095) / 4096))
        let chunk = Swift.max(1, (n + numTG - 1) / numTG)
        let out = try MetalArrowBuffer.allocate(byteCount: slots * 8, zeroed: true, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: slots * 8, zeroed: true, context: ctx)
        let skey = sourceKey(c.source)

        if usePrivate {
            let partials = try MetalArrowBuffer.allocate(byteCount: numTG * slots * 8, zeroed: true, context: ctx)
            let pcounts = try MetalArrowBuffer.allocate(byteCount: numTG * slots * 4, zeroed: true, context: ctx)
            let accPSO = try pipeline(c, "am_gb_priv", ctx, key: skey)
            let finPSO = try pipeline(c, "am_gb_finalize", ctx, key: skey)
            try ctx.run { enc in
                enc.setComputePipelineState(accPSO)
                bind(enc, leaves: c.leaves, bindings: c.bindings, inputs: inputs)
                Dispatch.setLength(enc, n, lengthBuffer, index: c.nPtrIndex)
                Dispatch.setUInt(enc, K, index: c.extra["K"]!)
                Dispatch.setUInt(enc, chunk, index: c.extra["chunk"]!)
                enc.setBuffer(partials.mtl, offset: 0, index: c.extra["partials"]!)
                enc.setBuffer(pcounts.mtl, offset: 0, index: c.extra["pcounts"]!)
                enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(finPSO)
                enc.setBuffer(partials.mtl, offset: 0, index: 0)
                enc.setBuffer(pcounts.mtl, offset: 0, index: 1)
                Dispatch.setUInt(enc, K, index: 2)
                Dispatch.setUInt(enc, numTG, index: 3)
                enc.setBuffer(out.mtl, offset: 0, index: 4)
                enc.setBuffer(counts.mtl, offset: 0, index: 5)
                Dispatch.dispatch1D(enc, finPSO, count: slots)
            }
            ctx.retainUntilFlush(partials); ctx.retainUntilFlush(pcounts)
        } else {
            let lo = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: true, context: ctx)
            let hi = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: true, context: ctx)
            let cnt = try MetalArrowBuffer.allocate(byteCount: slots * 4, zeroed: true, context: ctx)
            withExtendedLifetime(lo) {
                let p = lo.mutableTyped(UInt32.self)
                for a in 0..<nAgg where kinds[a].initWord != 0 {
                    let iv = kinds[a].initWord
                    for k in 0..<K { p[a * K + k] = iv }
                }
            }
            let accPSO = try pipeline(c, "am_gb_dev", ctx, key: skey)
            let packPSO = try pipeline(c, "am_gb_pack", ctx, key: skey)
            try ctx.run { enc in
                enc.setComputePipelineState(accPSO)
                bind(enc, leaves: c.leaves, bindings: c.bindings, inputs: inputs)
                Dispatch.setLength(enc, n, lengthBuffer, index: c.nPtrIndex)
                Dispatch.setUInt(enc, K, index: c.extra["K"]!)
                Dispatch.setUInt(enc, chunk, index: c.extra["chunk"]!)
                enc.setBuffer(out.mtl, offset: 0, index: c.extra["partials"]!)     // unused on this path
                enc.setBuffer(cnt.mtl, offset: 0, index: c.extra["pcounts"]!)      // unused on this path
                enc.setBuffer(lo.mtl, offset: 0, index: c.extra["dlo"]!)
                enc.setBuffer(hi.mtl, offset: 0, index: c.extra["dhi"]!)
                enc.setBuffer(cnt.mtl, offset: 0, index: c.extra["dcnt"]!)
                enc.dispatchThreadgroups(MTLSize(width: numTG, height: 1, depth: 1), threadsPerThreadgroup: tg)
                enc.memoryBarrier(scope: .buffers)
                enc.setComputePipelineState(packPSO)
                enc.setBuffer(lo.mtl, offset: 0, index: 0)
                enc.setBuffer(hi.mtl, offset: 0, index: 1)
                enc.setBuffer(cnt.mtl, offset: 0, index: 2)
                Dispatch.setUInt(enc, K, index: 3)
                enc.setBuffer(out.mtl, offset: 0, index: 4)
                enc.setBuffer(counts.mtl, offset: 0, index: 5)
                Dispatch.dispatch1D(enc, packPSO, count: slots)
            }
            ctx.retainUntilFlush(lo); ctx.retainUntilFlush(hi); ctx.retainUntilFlush(cnt)
        }
        for (_, i) in inputs { ctx.retainUntilFlush(i.owner) }
        ctx.retainUntilFlush(out); ctx.retainUntilFlush(counts)
        try ctx.syncPoint()

        var r = ExprQueryResult()
        let keys = try MetalArray<Int32>.allocate(length: K, withValidity: false, context: ctx)
        let kp = keys.mutableValuePointer
        for k in 0..<K { kp[k] = Int32(k) }
        r.names.append(q.keyName)
        r.columns.append(.int32(keys))
        try withExtendedLifetime((out, counts)) {
            let o = out.typed(UInt64.self), cn = counts.typed(UInt64.self)
            for (a, plan) in c.aggs.enumerated() {
                r.names.append(plan.name)
                r.columns.append(try gbColumn(plan, kinds[a], valueTypes[a], o: o, cn: cn, base: a * K, K: K, ctx: ctx))
            }
        }
        return r
    }

    /// Turns one aggregate's finished table into an Arrow column; groups with no contributing value are null.
    static func gbColumn(_ plan: AggPlan, _ kind: GBKind, _ vt: ExprType?,
                         o: UnsafePointer<UInt64>, cn: UnsafePointer<UInt64>, base: Int, K: Int,
                         ctx: MetalContext) throws -> AnyMetalArray {
        func build<T: ArrowPrimitive>(_: T.Type, nullable: Bool, _ value: (Int) -> T) throws -> AnyMetalArray {
            let a = try MetalArray<T>.allocate(length: K, withValidity: nullable, context: ctx)
            let d = a.mutableValuePointer
            let v = nullable ? a.validity!.mutableTyped(UInt8.self) : nil
            for k in 0..<K {
                if nullable && cn[base + k] == 0 { continue }
                d[k] = value(k)
                if let v { Bitmap.set(v, k) }
            }
            a.recomputeNullCount()
            return arrowMetalWrap(a)
        }
        if plan.op == .count { return try build(Int64.self, nullable: false) { Int64(bitPattern: cn[base + $0]) } }
        if plan.op == .mean {
            return try build(Double.self, nullable: true) { k in
                let c = Double(cn[base + k])
                switch kind {
                case .sumI: return Double(Int64(bitPattern: o[base + k])) / c
                case .sumU: return Double(o[base + k]) / c
                default: return Double(Float(bitPattern: UInt32(truncatingIfNeeded: o[base + k]))) / c
                }
            }
        }
        if plan.op == .sum {
            switch kind {
            case .sumI: return try build(Int64.self, nullable: true) { Int64(bitPattern: o[base + $0]) }
            case .sumU: return try build(UInt64.self, nullable: true) { o[base + $0] }
            default: return try build(Double.self, nullable: true) {
                Double(Float(bitPattern: UInt32(truncatingIfNeeded: o[base + $0])))
            }
            }
        }
        let t = vt ?? .int32
        func narrow<T: ArrowPrimitive & FixedWidthInteger>(_: T.Type, signed: Bool) throws -> AnyMetalArray {
            try build(T.self, nullable: true) { k in
                let w = UInt32(truncatingIfNeeded: o[base + k])
                return signed ? T(truncatingIfNeeded: Int32(bitPattern: w)) : T(truncatingIfNeeded: w)
            }
        }
        switch t {
        case .int8: return try narrow(Int8.self, signed: true)
        case .int16: return try narrow(Int16.self, signed: true)
        case .int32: return try narrow(Int32.self, signed: true)
        case .uint8: return try narrow(UInt8.self, signed: false)
        case .uint16: return try narrow(UInt16.self, signed: false)
        case .uint32: return try narrow(UInt32.self, signed: false)
        case .float32:
            return try build(Float.self, nullable: true) {
                fromFkey32(Int32(bitPattern: UInt32(truncatingIfNeeded: o[base + $0])))
            }
        default:
            throw ExprError.unsupported("group_by min/max output type \(t.rawValue)")
        }
    }
}
