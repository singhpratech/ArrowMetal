import Foundation
import Metal

// The four terminal shapes: project, filter + project, aggregate, group-by aggregate.
// Each one generates its kernels from the same lowered expression body (`am_row`).

extension ExprCompiler {

    // MARK: - project (no filter): one pass, one dispatch

    static func runProject(_ q: ExprQuery, _ ps: [ExprQuery.Projection], schema: [String: ExprColumnInfo],
                           key: String, inputs: [String: Input], n: Int, knownLength: Int,
                           lengthBuffer: MetalArrowBuffer?, ctx: MetalContext) throws -> ExprQueryResult {
        let c = try cached("project|\(q.canonical)|\(key)") {
            let em = ExprEmitter(schema: schema)
            var slots: [ExprSlot] = []
            for p in ps { slots.append(try em.emit(p.expr)) }
            for (i, s) in slots.enumerated() where s.type == .utf8 {
                throw ExprError.unsupported("project output \"\(ps[i].name)\" is utf8; string outputs are not materialised")
            }
            let comp = Compiled()
            var (params, bindings, idx) = leafParams(em.leaves, from: 0)
            comp.bindings = bindings
            comp.leaves = em.leaves
            comp.nPtrIndex = idx
            params.append("device const uint* nPtr [[buffer(\(idx))]]"); idx += 1
            var decls = "", stmts = "", tail = ""
            for (k, s) in slots.enumerated() {
                comp.outputTypes.append(s.type)
                comp.outputNullable.append(!s.alwaysValid)
                comp.outValueIndex.append(idx)
                params.append("device \(s.type == .boolean ? "uint" : s.type.msl)* out\(k) [[buffer(\(idx))]]"); idx += 1
                if s.alwaysValid { comp.outValidityIndex.append(nil) } else {
                    comp.outValidityIndex.append(idx)
                    params.append("device uint* outv\(k) [[buffer(\(idx))]]"); idx += 1
                }
                if s.type == .boolean {
                    decls += "    uint OB\(k) = 0u;\n"
                    stmts += "        if (O\(k)) OB\(k) |= (1u << j);\n"
                    tail += "    out\(k)[w] = OB\(k);\n"
                } else {
                    stmts += "        out\(k)[i] = O\(k);\n"
                }
                if !s.alwaysValid {
                    decls += "    uint OV\(k) = 0u;\n"
                    stmts += "        if (O\(k)k) OV\(k) |= (1u << j);\n"
                    tail += "    outv\(k)[w] = OV\(k);\n"
                }
            }
            var src = prelude(em)
            src += em.rowFunction(name: "am_row", outputs: slots)
            src += "\nkernel void am_project(" + params.joined(separator: ", ")
            src += ", uint w [[thread_position_in_grid]]) {\n"
            src += "    uint n = *nPtr;\n    uint base = w * 32u;\n    if (base >= n) return;\n"
            src += "    uint limit = min(32u, n - base);\n"
            src += decls
            src += declareOutputs(slots, indent: "    ")
            src += rowLoop(em, outputs: slots.count, statements: stmts, indent: "    ")
            src += tail
            src += "}\n"
            comp.source = src
            return comp
        }

        let words = BitmapOps.words(bits: n)
        var values: [MetalArrowBuffer] = [], validities: [MetalArrowBuffer?] = []
        for (k, t) in c.outputTypes.enumerated() {
            let bytes = t == .boolean ? Bitmap.byteCount(bits: n) : n * byteWidth(t)
            values.append(try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: ctx))
            validities.append(c.outputNullable[k]
                ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), zeroed: false, context: ctx)
                : nil)
        }
        if n > 0 {
            let pso = try pipeline(c, "am_project", ctx, key: sourceKey(c.source))
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                bind(enc, leaves: c.leaves, bindings: c.bindings, inputs: inputs)
                Dispatch.setLength(enc, n, lengthBuffer, index: c.nPtrIndex)
                for (k, v) in values.enumerated() {
                    enc.setBuffer(v.mtl, offset: v.offset, index: c.outValueIndex[k])
                    if let vi = c.outValidityIndex[k], let vb = validities[k] {
                        enc.setBuffer(vb.mtl, offset: vb.offset, index: vi)
                    }
                }
                Dispatch.dispatch1D(enc, pso, count: words)
            }
        }
        for (_, i) in inputs { ctx.retainUntilFlush(i.owner) }
        var r = ExprQueryResult()
        for (k, t) in c.outputTypes.enumerated() {
            r.names.append(ps[k].name)
            r.columns.append(try makeArray(t, length: knownLength, values: values[k], validity: validities[k],
                                           ctx: ctx, pendingLength: lengthBuffer, capacity: n))
        }
        return r
    }

    // MARK: - filter + project: count, scan, scatter

    static func runFilteredProject(_ q: ExprQuery, _ ps: [ExprQuery.Projection], schema: [String: ExprColumnInfo],
                                   key: String, inputs: [String: Input], n: Int,
                                   lengthBuffer: MetalArrowBuffer?, ctx: MetalContext) throws -> ExprQueryResult {
        let predicate = q.filter!
        let c = try cached("filterproject|\(q.canonical)|\(key)") {
            let comp = Compiled()
            // Pass 1: the predicate only.
            let pem = ExprEmitter(schema: schema, prefix: "p")
            let pslot = try pem.emit(predicate, hint: .boolean)
            guard pslot.type == .boolean else {
                throw ExprError.unsupported("filter predicate must be boolean, got \(pslot.type.rawValue)")
            }
            comp.countLeaves = pem.leaves
            var (cparams, cbind, cidx) = leafParams(pem.leaves, from: 0)
            comp.countBindings = cbind
            comp.countExtra["nPtr"] = cidx
            cparams.append("device const uint* nPtr [[buffer(\(cidx))]]"); cidx += 1
            comp.countExtra["sel"] = cidx
            cparams.append("device uint* sel [[buffer(\(cidx))]]"); cidx += 1
            comp.countExtra["blockCounts"] = cidx
            cparams.append("device uint* blockCounts [[buffer(\(cidx))]]"); cidx += 1

            var src = ""
            src += pem.rowFunction(name: "am_pred", outputs: [pslot])
            src += "\nkernel void am_count(" + cparams.joined(separator: ", ") + """
            ,
                                  uint w [[thread_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
                threadgroup uint simdTotals[32];
                uint n = *nPtr;
                uint base = w * 32u;
                uint word = 0u;
                bool O0 = false; bool O0k = false;
                if (base < n) {
                    uint limit = min(32u, n - base);

            """
            src += rowLoop(pem, outputs: 1, statements: "        if (O0 && O0k) word |= (1u << j);\n",
                           indent: "        ", callName: "am_pred")
            src += """
                    sel[w] = word;
                }
                uint cnt = popcount(word);
                uint t = simd_sum(cnt);
                if (lane == 0) simdTotals[sgid] = t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid == 0) { uint total = 0; for (uint k = 0; k < TG / 32u; k++) total += simdTotals[k]; blockCounts[tgid] = total; }
            }

            """
            src += ExprSource.scanKernel

            // Pass 2: the projections, evaluated only for selected rows.
            let em = ExprEmitter(schema: schema, prefix: "q")
            var slots: [ExprSlot] = []
            for p in ps { slots.append(try em.emit(p.expr)) }
            for (i, s) in slots.enumerated() where s.type == .utf8 {
                throw ExprError.unsupported("project output \"\(ps[i].name)\" is utf8; string outputs are not materialised")
            }
            comp.leaves = em.leaves
            var (params, bindings, idx) = leafParams(em.leaves, from: 0)
            comp.bindings = bindings
            comp.nPtrIndex = idx
            params.append("device const uint* nPtr [[buffer(\(idx))]]"); idx += 1
            comp.extra["sel"] = idx
            params.append("device const uint* sel [[buffer(\(idx))]]"); idx += 1
            comp.extra["blockOffsets"] = idx
            params.append("device const uint* blockOffsets [[buffer(\(idx))]]"); idx += 1
            var stores = ""
            for (k, s) in slots.enumerated() {
                comp.outputTypes.append(s.type)
                comp.outputNullable.append(!s.alwaysValid)
                comp.outValueIndex.append(idx)
                params.append("device \(s.type == .boolean ? "uchar" : s.type.msl)* out\(k) [[buffer(\(idx))]]"); idx += 1
                if s.alwaysValid { comp.outValidityIndex.append(nil) } else {
                    comp.outValidityIndex.append(idx)
                    params.append("device uchar* outv\(k) [[buffer(\(idx))]]"); idx += 1
                }
                stores += s.type == .boolean ? "            out\(k)[pos] = O\(k) ? 1 : 0;\n"
                                             : "            out\(k)[pos] = O\(k);\n"
                if !s.alwaysValid { stores += "            outv\(k)[pos] = O\(k)k ? 1 : 0;\n" }
            }
            src += "\n" + em.rowFunction(name: "am_row", outputs: slots)
            src += "\nkernel void am_scatter(" + params.joined(separator: ", ") + """
            ,
                                    uint w [[thread_position_in_grid]],
                                    uint lid [[thread_index_in_threadgroup]],
                                    uint tgid [[threadgroup_position_in_grid]],
                                    uint sgid [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
                threadgroup uint simdTotals[32];
                uint n = *nPtr;
                uint base = w * 32u;
                uint word = 0u;
                if (base < n) {
                    word = sel[w];
                    uint limit = n - base;
                    if (limit < 32u) word &= (1u << limit) - 1u;
                }
                uint cnt = popcount(word);
                uint localOff = simd_prefix_exclusive_sum(cnt);
                uint t = simd_sum(cnt);
                if (lane == 0) simdTotals[sgid] = t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint prefix = 0;
                for (uint k = 0; k < sgid; k++) prefix += simdTotals[k];
                uint pos = blockOffsets[tgid] + prefix + localOff;
                if (base >= n) return;

            """
            for l in em.leaves {
                if l.type == .boolean { src += "    uint BW\(l.index) = am_vword(LB\(l.index), w);\n" }
                if l.nullable { src += "    uint VW\(l.index) = am_vword(LV\(l.index), w);\n" }
            }
            src += declareOutputs(slots, indent: "    ")
            src += """
                uint wd = word;
                while (wd) {
                    uint j = ctz(wd);
                    wd &= wd - 1u;
                    uint i = base + j;
                    \(scalarCall(em, outputs: slots.count, wordValidity: true))

            """
            src += stores
            src += "        pos++;\n    }\n}\n"
            comp.source = prelude([pem, em]) + src
            return comp
        }

        let words = BitmapOps.words(bits: n)
        let blocks = Swift.max(1, (words + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize)
        let sel = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 4), zeroed: true, context: ctx)
        let blockCounts = try MetalArrowBuffer.allocate(byteCount: blocks * 4, zeroed: false, context: ctx)
        let total = try MetalArrowBuffer.allocate(byteCount: 4, zeroed: true, context: ctx)
        var values: [MetalArrowBuffer] = [], validBytes: [MetalArrowBuffer?] = []
        for (k, t) in c.outputTypes.enumerated() {
            let bytes = t == .boolean ? n : n * byteWidth(t)
            values.append(try MetalArrowBuffer.allocate(byteCount: Swift.max(bytes, 1), zeroed: false, context: ctx))
            validBytes.append(c.outputNullable[k]
                ? try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: false, context: ctx) : nil)
        }
        let skey = sourceKey(c.source)
        let countPSO = try pipeline(c, "am_count", ctx, key: skey)
        let scanPSO = try pipeline(c, "am_scan", ctx, key: skey)
        let scatterPSO = try pipeline(c, "am_scatter", ctx, key: skey)
        let tg = MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1)
        let grid = MTLSize(width: blocks, height: 1, depth: 1)
        try ctx.run { enc in
            enc.setComputePipelineState(countPSO)
            bind(enc, leaves: c.countLeaves, bindings: c.countBindings, inputs: inputs)
            Dispatch.setLength(enc, n, lengthBuffer, index: c.countExtra["nPtr"]!)
            enc.setBuffer(sel.mtl, offset: sel.offset, index: c.countExtra["sel"]!)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: c.countExtra["blockCounts"]!)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: 0)
            Dispatch.setLength(enc, n, lengthBuffer, index: 1)
            enc.setBuffer(total.mtl, offset: 0, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: tg)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(scatterPSO)
            bind(enc, leaves: c.leaves, bindings: c.bindings, inputs: inputs)
            Dispatch.setLength(enc, n, lengthBuffer, index: c.nPtrIndex)
            enc.setBuffer(sel.mtl, offset: sel.offset, index: c.extra["sel"]!)
            enc.setBuffer(blockCounts.mtl, offset: 0, index: c.extra["blockOffsets"]!)
            for (k, v) in values.enumerated() {
                enc.setBuffer(v.mtl, offset: v.offset, index: c.outValueIndex[k])
                if let vi = c.outValidityIndex[k], let vb = validBytes[k] {
                    enc.setBuffer(vb.mtl, offset: vb.offset, index: vi)
                }
            }
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
        for (_, i) in inputs { ctx.retainUntilFlush(i.owner) }
        ctx.retainUntilFlush(sel); ctx.retainUntilFlush(blockCounts)
        for v in validBytes { if let v { ctx.retainUntilFlush(v) } }

        var r = ExprQueryResult()
        if ctx.isBatching {
            for (k, t) in c.outputTypes.enumerated() {
                var validity: MetalArrowBuffer? = nil
                if let vb = validBytes[k] { validity = try BitmapOps.packBits(ctx, bytes: vb, bits: n, lengthBuffer: total) }
                var vals = values[k]
                if t == .boolean { vals = try BitmapOps.packBits(ctx, bytes: vals, bits: n, lengthBuffer: total) }
                r.names.append(ps[k].name)
                r.columns.append(try makeArray(t, length: 0, values: vals, validity: validity, ctx: ctx,
                                               pendingLength: total, capacity: n))
            }
            return r
        }
        let outLen = withExtendedLifetime(total) { Int(total.typed(UInt32.self)[0]) }
        for (k, t) in c.outputTypes.enumerated() {
            var validity: MetalArrowBuffer? = nil
            if let vb = validBytes[k], outLen > 0 { validity = try BitmapOps.packBits(ctx, bytes: vb, bits: outLen) }
            var vals = values[k]
            if t == .boolean {
                vals = outLen > 0 ? try BitmapOps.packBits(ctx, bytes: vals, bits: outLen)
                                  : try MetalArrowBuffer.allocate(byteCount: 1, context: ctx)
            } else {
                vals = vals.view(byteOffset: 0, byteCount: outLen * byteWidth(t))
            }
            r.names.append(ps[k].name)
            r.columns.append(try makeArray(t, length: outLen, values: vals, validity: validity, ctx: ctx,
                                           pendingLength: nil, capacity: outLen))
        }
        return r
    }

    // MARK: - aggregate: threadgroup partials, finished on the CPU

    static func runReduce(_ q: ExprQuery, _ aggs: [ExprAggregate], schema: [String: ExprColumnInfo],
                          key: String, inputs: [String: Input], n: Int,
                          lengthBuffer: MetalArrowBuffer?, ctx: MetalContext) throws -> ExprQueryResult {
        let c = try cached("reduce|\(q.canonical)|\(key)") {
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
            var plans: [AggPlan] = []
            for a in aggs {
                if a.op == .count && a.expr == nil {
                    plans.append(AggPlan(op: .count, name: a.name, kind: .countRows, valueType: nil, slot: nil))
                    continue
                }
                guard let e = a.expr else { throw ExprError.invalid("\(a.op.rawValue) \"\(a.name)\" needs an expression") }
                let s = try em.emit(e)
                guard s.type.isNumeric || (a.op == .count && s.type == .boolean) else {
                    throw ExprError.unsupported("\(a.op.rawValue) of a \(s.type.rawValue) expression")
                }
                let kind = try reduceKind(a.op, s.type)
                if kind.usesDoubleMath { em.usesDoubleMath = true }
                plans.append(AggPlan(op: a.op, name: a.name, kind: kind, valueType: s.type, slot: slots.count))
                slots.append(s)
            }
            comp.aggs = plans
            comp.leaves = em.leaves
            var (params, bindings, idx) = leafParams(em.leaves, from: 0)
            comp.bindings = bindings
            comp.nPtrIndex = idx
            params.append("device const uint* nPtr [[buffer(\(idx))]]"); idx += 1
            comp.extra["groups"] = idx
            params.append("constant uint& groups [[buffer(\(idx))]]"); idx += 1
            comp.extra["partials"] = idx
            params.append("device ulong* partials [[buffer(\(idx))]]"); idx += 1
            comp.extra["counts"] = idx
            params.append("device uint* pcounts [[buffer(\(idx))]]"); idx += 1

            var decls = "", shared = "", stmts = "", tree = "", tail = ""
            for (k, p) in plans.enumerated() {
                decls += "    \(p.kind.accType) a\(k) = \(p.kind.initExpr); uint n\(k) = 0u;\n"
                shared += "    threadgroup \(p.kind.accType) sh\(k)[TG]; threadgroup uint sc\(k)[TG];\n"
                var acc = ""
                if let s = p.slot {
                    let v = "O\(s)"
                    var inner = "                a\(k) = \(p.kind.combine("a\(k)", v)); n\(k)++;\n"
                    if let g = p.kind.guardExpr(v) { inner = "                if (\(g)) {\n    \(inner)                }\n" }
                    acc = "            if (O\(s)k) {\n\(inner)            }\n"
                } else {
                    acc = "            n\(k)++;\n"
                }
                stmts += acc
                tree += """
                            { \(p.kind.accType) va = sh\(k)[lid], vb = sh\(k)[lid + s]; sh\(k)[lid] = \(p.kind.merge("va", "vb")); sc\(k)[lid] += sc\(k)[lid + s]; }

                """
                tail += "        partials[\(k)u * groups + tgid] = \(p.kind.store("sh\(k)[0]"));\n"
                tail += "        pcounts[\(k)u * groups + tgid] = sc\(k)[0];\n"
            }
            if let pi = predIndex { stmts = "        if (O\(pi) && O\(pi)k) {\n" + stmts + "        }\n" }

            var src = prelude(em)
            src += em.rowFunction(name: "am_row", outputs: slots)
            src += "\nkernel void am_reduce(" + params.joined(separator: ", ") + """
            ,
                                  uint gid [[thread_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint gridSize [[threads_per_grid]]) {

            """
            src += shared
            src += "    uint n = *nPtr;\n    uint words = (n + 31u) / 32u;\n"
            src += decls
            src += declareOutputs(slots, indent: "    ")
            src += "    for (uint w = gid; w < words; w += gridSize) {\n"
            src += "        uint base = w * 32u;\n        uint limit = min(32u, n - base);\n"
            src += rowLoop(em, outputs: slots.count, statements: stmts, indent: "        ")
            src += "    }\n"
            for (k, _) in plans.enumerated() { src += "    sh\(k)[lid] = a\(k); sc\(k)[lid] = n\(k);\n" }
            src += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            src += "    for (uint s = TG / 2; s > 0; s >>= 1) {\n        if (lid < s) {\n"
            src += tree
            src += "        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n"
            src += "    if (lid == 0) {\n" + tail + "    }\n}\n"
            comp.source = src
            return comp
        }

        let groups = Swift.max(1, Swift.min(2048, (BitmapOps.words(bits: n) + Dispatch.threadgroupSize - 1) / Dispatch.threadgroupSize))
        let nAgg = c.aggs.count
        let partials = try MetalArrowBuffer.allocate(byteCount: nAgg * groups * 8, zeroed: true, context: ctx)
        let counts = try MetalArrowBuffer.allocate(byteCount: nAgg * groups * 4, zeroed: true, context: ctx)
        if n > 0 {
            let pso = try pipeline(c, "am_reduce", ctx, key: sourceKey(c.source))
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                bind(enc, leaves: c.leaves, bindings: c.bindings, inputs: inputs)
                Dispatch.setLength(enc, n, lengthBuffer, index: c.nPtrIndex)
                Dispatch.setUInt(enc, groups, index: c.extra["groups"]!)
                enc.setBuffer(partials.mtl, offset: 0, index: c.extra["partials"]!)
                enc.setBuffer(counts.mtl, offset: 0, index: c.extra["counts"]!)
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: Dispatch.threadgroupSize, height: 1, depth: 1))
            }
        }
        for (_, i) in inputs { ctx.retainUntilFlush(i.owner) }
        ctx.retainUntilFlush(partials); ctx.retainUntilFlush(counts)
        try ctx.syncPoint()

        var r = ExprQueryResult()
        withExtendedLifetime((partials, counts)) {
            let p = partials.typed(UInt64.self), cn = counts.typed(UInt32.self)
            for (k, plan) in c.aggs.enumerated() {
                var total: UInt64 = 0
                for g in 0..<groups { total &+= UInt64(cn[k * groups + g]) }
                r.scalarNames.append(plan.name)
                if total == 0 && plan.op != .count { r.scalars.append(.null); continue }
                r.scalars.append(finishReduce(plan, p: p, base: k * groups, counts: cn, cbase: k * groups,
                                              groups: groups, total: total))
            }
        }
        return r
    }

    static func reduceKind(_ op: ExprAggregate.Op, _ t: ExprType) throws -> AggKind {
        switch op {
        case .count: return .countValues
        case .sum, .mean:
            if t == .float64 { return .sumF64 }
            if t == .float32 { return .sumF32 }
            return t.isSigned ? .sumInt : .sumUInt
        case .min, .max:
            let isMin = op == .min
            if t == .float64 { return isMin ? .minF64 : .maxF64 }
            if t == .float32 { return isMin ? .minF32 : .maxF32 }
            if t.isSigned { return isMin ? .minInt : .maxInt }
            return isMin ? .minUInt : .maxUInt
        }
    }

    static func finishReduce(_ plan: AggPlan, p: UnsafePointer<UInt64>, base: Int,
                             counts: UnsafePointer<UInt32>, cbase: Int, groups: Int, total: UInt64) -> ExprScalar {
        switch plan.kind {
        case .countRows, .countValues: return .int(Int64(bitPattern: total))
        case .sumInt:
            var acc: Int64 = 0
            for g in 0..<groups { acc &+= Int64(bitPattern: p[base + g]) }
            return plan.op == .mean ? .double(Double(acc) / Double(total)) : .int(acc)
        case .sumUInt:
            var acc: UInt64 = 0
            for g in 0..<groups { acc &+= p[base + g] }
            return plan.op == .mean ? .double(Double(acc) / Double(total)) : .uint(acc)
        case .sumF32, .sumF64:
            var acc = 0.0
            for g in 0..<groups { acc += Double(bitPattern: p[base + g]) }
            return plan.op == .mean ? .double(acc / Double(total)) : .double(acc)
        case .minInt, .maxInt:
            var acc: Int64 = plan.kind == .minInt ? .max : .min
            for g in 0..<groups where counts[cbase + g] > 0 {
                let v = Int64(bitPattern: p[base + g])
                acc = plan.kind == .minInt ? Swift.min(acc, v) : Swift.max(acc, v)
            }
            return .int(acc)
        case .minUInt, .maxUInt:
            var acc: UInt64 = plan.kind == .minUInt ? .max : .min
            for g in 0..<groups where counts[cbase + g] > 0 {
                let v = p[base + g]
                acc = plan.kind == .minUInt ? Swift.min(acc, v) : Swift.max(acc, v)
            }
            return .uint(acc)
        case .minF32, .maxF32:
            var acc = plan.kind == .minF32 ? Double.infinity : -Double.infinity
            for g in 0..<groups where counts[cbase + g] > 0 {
                let v = Double(Float(bitPattern: UInt32(truncatingIfNeeded: p[base + g])))
                acc = plan.kind == .minF32 ? Swift.min(acc, v) : Swift.max(acc, v)
            }
            return .double(acc)
        case .minF64, .maxF64:
            var acc = plan.kind == .minF64 ? Double.infinity : -Double.infinity
            for g in 0..<groups where counts[cbase + g] > 0 {
                let v = Dispatch.doubleFromKey(Int64(bitPattern: p[base + g]))
                acc = plan.kind == .minF64 ? Swift.min(acc, v) : Swift.max(acc, v)
            }
            return .double(acc)
        }
    }
}
