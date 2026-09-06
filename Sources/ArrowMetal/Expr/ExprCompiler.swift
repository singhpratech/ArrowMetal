import Foundation
import Metal

// Compiling a whole query into one Metal kernel.
//
// Shapes (each is one command buffer, and joins an open `batch { }`):
//   project, no filter        1 dispatch    read every input once, write every output once
//   project + filter          3 dispatches  count -> scan -> scatter, the compaction pipeline
//   aggregate (+ filter)      1 dispatch    threadgroup partials, finished on the CPU
//   group_by (+ filter)       2 dispatches  privatised tables, then a merge over the partials
//
// Everything the expression itself does — arithmetic, comparisons, null logic, casts, string
// predicates — happens in registers inside those kernels, so the inputs are read once.

/// How one aggregate accumulates on the GPU.
enum AggKind {
    case countRows, countValues
    case sumInt, sumUInt, sumF32, sumF64
    case minInt, maxInt, minUInt, maxUInt, minF32, maxF32, minF64, maxF64

    var needsValue: Bool { if case .countRows = self { return false }; return true }
    var accType: String {
        switch self {
        case .countRows, .countValues: return "uint"
        case .sumInt, .minInt, .maxInt, .minF64, .maxF64: return "long"
        case .sumUInt, .minUInt, .maxUInt, .sumF32, .sumF64: return "ulong"
        case .minF32, .maxF32: return "float"
        }
    }
    var initExpr: String {
        switch self {
        case .countRows, .countValues: return "0u"
        case .sumInt: return "0L"
        case .sumUInt, .sumF32, .sumF64: return "0ul"
        case .minInt, .minF64: return "LONG_MAX"
        case .maxInt, .maxF64: return "LONG_MIN"
        case .minUInt: return "ULONG_MAX"
        case .maxUInt: return "0ul"
        case .minF32: return "INFINITY"
        case .maxF32: return "-INFINITY"
        }
    }
    /// A guard on the incoming value: min/max skip NaN, as Arrow does.
    func guardExpr(_ v: String) -> String? {
        switch self {
        case .minF32, .maxF32: return "!f_isnan32(as_type<uint>(\(v)))"
        case .minF64, .maxF64: return "!d_is_nan(\(v))"
        default: return nil
        }
    }
    /// Folding one value into the accumulator.
    func combine(_ acc: String, _ v: String) -> String {
        switch self {
        case .countRows, .countValues: return acc
        case .sumInt: return "\(acc) + (long)\(v)"
        case .sumUInt: return "\(acc) + (ulong)\(v)"
        case .sumF32: return "d_add(\(acc), d_from_float(\(v)))"
        case .sumF64: return "d_add(\(acc), \(v))"
        case .minInt: return "min(\(acc), (long)\(v))"
        case .maxInt: return "max(\(acc), (long)\(v))"
        case .minUInt: return "min(\(acc), (ulong)\(v))"
        case .maxUInt: return "max(\(acc), (ulong)\(v))"
        case .minF32: return "min(\(acc), \(v))"
        case .maxF32: return "max(\(acc), \(v))"
        case .minF64: return "min(\(acc), d_key((long)\(v)))"
        case .maxF64: return "max(\(acc), d_key((long)\(v)))"
        }
    }
    /// Folding two accumulators (the threadgroup tree).
    func merge(_ a: String, _ b: String) -> String {
        switch self {
        case .countRows, .countValues: return "\(a) + \(b)"
        case .sumInt, .sumUInt: return "\(a) + \(b)"
        case .sumF32, .sumF64: return "d_add(\(a), \(b))"
        case .minInt, .minUInt, .minF32, .minF64: return "min(\(a), \(b))"
        case .maxInt, .maxUInt, .maxF32, .maxF64: return "max(\(a), \(b))"
        }
    }
    /// Accumulator to the `ulong` partial slot.
    func store(_ a: String) -> String {
        switch self {
        case .minF32, .maxF32: return "(ulong)as_type<uint>(\(a))"
        case .sumInt, .minInt, .maxInt, .minF64, .maxF64: return "(ulong)\(a)"
        case .countRows, .countValues: return "(ulong)\(a)"
        default: return "\(a)"
        }
    }
    var usesDoubleMath: Bool {
        switch self { case .sumF32, .sumF64, .minF64, .maxF64: return true; default: return false }
    }
}

/// One planned aggregate.
struct AggPlan {
    var op: ExprAggregate.Op
    var name: String
    var kind: AggKind
    var valueType: ExprType?
    /// Index into the kernel's output slots (nil for `count` over rows).
    var slot: Int?
}

enum ExprCompiler {

    // MARK: - Column inputs

    struct Input {
        var type: ExprType
        var values: MetalArrowBuffer
        var data: MetalArrowBuffer?
        var validity: MetalArrowBuffer?
        var dispatchLength: Int
        var knownLength: Int
        var lengthBuffer: MetalArrowBuffer?
        var owner: AnyObject
    }

    static func input(_ a: AnyMetalArray, name: String) throws -> Input {
        func prim<T: ArrowPrimitive>(_ x: MetalArray<T>, _ t: ExprType) -> Input {
            Input(type: t, values: x.values, data: nil, validity: x.validity,
                  dispatchLength: x.dispatchLength, knownLength: x.knownLength,
                  lengthBuffer: x.lengthBuffer, owner: x)
        }
        switch a {
        case .int8(let x): return prim(x, .int8)
        case .int16(let x): return prim(x, .int16)
        case .int32(let x): return prim(x, .int32)
        case .int64(let x): return prim(x, .int64)
        case .uint8(let x): return prim(x, .uint8)
        case .uint16(let x): return prim(x, .uint16)
        case .uint32(let x): return prim(x, .uint32)
        case .uint64(let x): return prim(x, .uint64)
        case .float32(let x): return prim(x, .float32)
        case .float64(let x): return prim(x, .float64)
        case .boolean(let x):
            return Input(type: .boolean, values: x.values, data: nil, validity: x.validity,
                         dispatchLength: x.dispatchLength, knownLength: x.knownLength,
                         lengthBuffer: x.lengthBuffer, owner: x)
        case .string(let x), .binary(let x):
            return Input(type: .utf8, values: x.offsets, data: x.data, validity: x.validity,
                         dispatchLength: x.length, knownLength: x.length, lengthBuffer: nil, owner: x)
        case .extended(let e): return try input(e.storage, name: name)
        default:
            throw ExprError.unsupported("column \"\(name)\" is \(a.arrowFormat), which the expression compiler does not read")
        }
    }

    // MARK: - Compiled artefacts

    /// One kernel argument that comes from a column, in buffer-index order.
    struct Binding {
        enum Role { case values, data, validity }
        var leaf: Int
        var role: Role
        var index: Int
    }

    final class Compiled {
        var source = ""
        var leaves: [ExprEmitter.Leaf] = []
        var bindings: [Binding] = []
        var outputTypes: [ExprType] = []
        var outputNullable: [Bool] = []
        var nPtrIndex = 0
        var outValueIndex: [Int] = []
        var outValidityIndex: [Int?] = []
        var extra: [String: Int] = [:]
        // The predicate-only counting pass of a filtered project.
        var countLeaves: [ExprEmitter.Leaf] = []
        var countBindings: [Binding] = []
        var countExtra: [String: Int] = [:]
        var aggs: [AggPlan] = []
        var gbKinds: [GBKind] = []
        var pipelines: [String: MTLComputePipelineState] = [:]
    }

    private static let lock = NSLock()
    private static var cache: [String: Compiled] = [:]
    /// Number of distinct query shapes lowered to MSL in this process. Running the same shape again
    /// must not increment it; `ExprTests.testPipelineCacheHit` asserts that.
    public private(set) static var compileCount = 0

    static func cached(_ key: String, _ make: () throws -> Compiled) throws -> Compiled {
        lock.lock()
        if let c = cache[key] { lock.unlock(); return c }
        lock.unlock()
        let c = try make()
        lock.lock()
        if let existing = cache[key] { lock.unlock(); return existing }
        cache[key] = c
        compileCount += 1
        lock.unlock()
        return c
    }

    // MARK: - Entry point

    static func run(_ q: ExprQuery, names: [String], columns: [AnyMetalArray],
                    context: MetalContext) throws -> ExprQueryResult {
        guard names.count == columns.count else { throw ExprError.invalid("names/columns count mismatch") }
        var inputs: [String: Input] = [:]
        for (i, n) in names.enumerated() where inputs[n] == nil { inputs[n] = try input(columns[i], name: n) }

        let used = referencedColumns(q)
        var n = 0, knownLength = -1
        var lengthBuffer: MetalArrowBuffer? = nil
        for name in used {
            guard let c = inputs[name] else { throw ExprError.invalid("no column named \"\(name)\"") }
            n = Swift.max(n, c.dispatchLength)
            if let lb = c.lengthBuffer { lengthBuffer = lb }
            else if knownLength < 0 { knownLength = c.knownLength }
            else if knownLength != c.knownLength { throw ArrowMetalError.lengthMismatch(knownLength, c.knownLength) }
        }
        if used.isEmpty { n = columns.first?.length ?? 0; knownLength = n }
        if knownLength < 0 { knownLength = n }
        try Dispatch.checkLength(n)

        var schema: [String: ExprColumnInfo] = [:]
        for (name, c) in inputs { schema[name] = ExprColumnInfo(type: c.type, nullable: c.validity != nil) }
        let schemaKey = used.sorted().map { "\($0):\(schema[$0]!.type.rawValue):\(schema[$0]!.nullable)" }
                                     .joined(separator: ",")

        switch q.terminal {
        case .project(let ps):
            guard q.groupKey == nil else { throw ExprError.invalid("group_by needs an aggregate terminal, not project") }
            guard !ps.isEmpty else { throw ExprError.invalid("project needs at least one output") }
            if q.filter == nil {
                return try runProject(q, ps, schema: schema, key: schemaKey, inputs: inputs,
                                      n: n, knownLength: knownLength, lengthBuffer: lengthBuffer, ctx: context)
            }
            return try runFilteredProject(q, ps, schema: schema, key: schemaKey, inputs: inputs,
                                          n: n, lengthBuffer: lengthBuffer, ctx: context)
        case .aggregate(let aggs):
            guard !aggs.isEmpty else { throw ExprError.invalid("aggregate needs at least one aggregation") }
            if q.groupKey != nil {
                return try runGroupBy(q, aggs, schema: schema, key: schemaKey, inputs: inputs,
                                      n: n, lengthBuffer: lengthBuffer, ctx: context)
            }
            return try runReduce(q, aggs, schema: schema, key: schemaKey, inputs: inputs,
                                 n: n, lengthBuffer: lengthBuffer, ctx: context)
        }
    }

    static func referencedColumns(_ q: ExprQuery) -> [String] {
        var seen = Set<String>(), out: [String] = []
        func add(_ e: Expr) { for c in e.referencedColumns where seen.insert(c).inserted { out.append(c) } }
        if let f = q.filter { add(f) }
        if let k = q.groupKey { add(k) }
        switch q.terminal {
        case .project(let ps): for p in ps { add(p.expr) }
        case .aggregate(let aggs): for a in aggs { if let e = a.expr { add(e) } }
        }
        return out
    }

    // MARK: - Source assembly helpers

    static func leafParams(_ leaves: [ExprEmitter.Leaf], from start: Int)
        -> (params: [String], bindings: [Binding], next: Int) {
        var params: [String] = [], bindings: [Binding] = [], idx = start
        for l in leaves {
            switch l.type {
            case .utf8:
                params.append("device const int* LO\(l.index) [[buffer(\(idx))]]")
                bindings.append(Binding(leaf: l.index, role: .values, index: idx)); idx += 1
                params.append("device const uchar* LD\(l.index) [[buffer(\(idx))]]")
                bindings.append(Binding(leaf: l.index, role: .data, index: idx)); idx += 1
            case .boolean:
                params.append("device const uchar* LB\(l.index) [[buffer(\(idx))]]")
                bindings.append(Binding(leaf: l.index, role: .values, index: idx)); idx += 1
            default:
                params.append("device const \(l.type.msl)* LB\(l.index) [[buffer(\(idx))]]")
                bindings.append(Binding(leaf: l.index, role: .values, index: idx)); idx += 1
            }
            if l.nullable {
                params.append("device const uchar* LV\(l.index) [[buffer(\(idx))]]")
                bindings.append(Binding(leaf: l.index, role: .validity, index: idx)); idx += 1
            }
        }
        return (params, bindings, idx)
    }

    static func prelude(_ emitters: ExprEmitter...) -> String { prelude(emitters) }

    static func prelude(_ emitters: [ExprEmitter]) -> String {
        var s = KernelSource.prelude
        let dbl = emitters.contains { $0.usesDoubleMath }
        if dbl { s += DoubleMath.msl }
        if emitters.contains(where: { $0.usesTranscendental }) { s += DoubleTranscendental.msl }
        s += ExprSource.helpers
        if dbl { s += ExprSource.doubleHelpers }
        s += "\n"
        for em in emitters { s += em.patternConstants }
        return s
    }

    /// The 32-row loop, with a four-wide vector path when every leaf is a plain numeric column.
    static func rowLoop(_ em: ExprEmitter, outputs: Int, statements: String, indent: String = "        ",
                        callName: String = "am_row") -> String {
        let leaves = em.leaves
        var s = ""
        for l in leaves {
            if l.type == .boolean { s += "\(indent)uint BW\(l.index) = am_vword(LB\(l.index), w);\n" }
            if l.nullable { s += "\(indent)uint VW\(l.index) = am_vword(LV\(l.index), w);\n" }
        }
        func valid(_ l: ExprEmitter.Leaf) -> String { "(((VW\(l.index) >> j) & 1u) != 0u)" }
        func scalarValue(_ l: ExprEmitter.Leaf) -> String {
            l.type == .boolean ? "(((BW\(l.index) >> j) & 1u) != 0u)" : "LB\(l.index)[i]"
        }
        let strBeg: (ExprEmitter.Leaf) -> String = { "LO\($0.index)[i]" }
        let strEnd: (ExprEmitter.Leaf) -> String = { "LO\($0.index)[i + 1]" }
        let scalarArgs = em.callArguments(value: scalarValue, valid: valid, strBegin: strBeg, strEnd: strEnd,
                                          outputCount: outputs)
        var scalar = "\(indent)for (uint j = 0; j < limit; j++) {\n\(indent)    uint i = base + j;\n"
        scalar += "\(indent)    \(callName)(\(scalarArgs));\n" + statements + "\(indent)}\n"

        let vectorisable = !leaves.isEmpty && leaves.allSatisfy { $0.type.isNumeric }
        guard vectorisable else { return s + scalar }
        var vec = "\(indent)if (limit == 32u) {\n\(indent)    for (uint c = 0; c < 8u; c++) {\n"
        for l in leaves {
            vec += "\(indent)        \(l.type.msl)4 g\(l.index) = *(device const \(l.type.msl)4*)(LB\(l.index) + base + c * 4u);\n"
        }
        for lane in 0..<4 {
            let comp = ["x", "y", "z", "w"][lane]
            let args = em.callArguments(value: { "g\($0.index).\(comp)" }, valid: valid,
                                        strBegin: strBeg, strEnd: strEnd, outputCount: outputs)
            vec += "\(indent)        {\n\(indent)            uint j = c * 4u + \(lane)u;\n"
            vec += "\(indent)            uint i = base + j;\n"
            vec += "\(indent)            \(callName)(\(args));\n"
            vec += statements + "\(indent)        }\n"
        }
        vec += "\(indent)    }\n\(indent)} else {\n" + scalar + "\(indent)}\n"
        return s + vec
    }

    /// One scalar call at row `i` (the scatter and group-by kernels are not word shaped).
    static func scalarCall(_ em: ExprEmitter, outputs: Int, wordValidity: Bool, callName: String = "am_row") -> String {
        func valid(_ l: ExprEmitter.Leaf) -> String {
            wordValidity ? "(((VW\(l.index) >> j) & 1u) != 0u)" : "bit_get(LV\(l.index), i)"
        }
        func value(_ l: ExprEmitter.Leaf) -> String {
            if l.type == .boolean { return wordValidity ? "(((BW\(l.index) >> j) & 1u) != 0u)" : "bit_get(LB\(l.index), i)" }
            return "LB\(l.index)[i]"
        }
        return "\(callName)(" + em.callArguments(value: value, valid: valid,
                                            strBegin: { "LO\($0.index)[i]" }, strEnd: { "LO\($0.index)[i + 1]" },
                                            outputCount: outputs) + ");"
    }

    static func declareOutputs(_ slots: [ExprSlot], indent: String) -> String {
        var s = ""
        for (k, o) in slots.enumerated() {
            s += "\(indent)\(o.type.msl) O\(k) = \(zero(o.type)); bool O\(k)k = false;\n"
        }
        return s
    }
    static func zero(_ t: ExprType) -> String {
        switch t {
        case .boolean: return "false"
        case .float32: return "0.0f"
        case .float64: return "0ul"
        default: return "(\(t.msl))0"
        }
    }

    // MARK: - Binding

    static func bind(_ enc: MTLComputeCommandEncoder, leaves: [ExprEmitter.Leaf], bindings: [Binding],
                     inputs: [String: Input]) {
        for b in bindings {
            let c = inputs[leaves[b.leaf].name]!
            let buf: MetalArrowBuffer
            switch b.role {
            case .values: buf = c.values
            case .data: buf = c.data ?? c.values
            case .validity: buf = c.validity ?? c.values
            }
            enc.setBuffer(buf.mtl, offset: buf.offset, index: b.index)
        }
    }

    static func pipeline(_ c: Compiled, _ fn: String, _ ctx: MetalContext, key: String) throws -> MTLComputePipelineState {
        lock.lock()
        if let p = c.pipelines[fn] { lock.unlock(); return p }
        lock.unlock()
        let p = try ctx.pipeline(source: c.source, function: fn, cacheKey: "expr/\(fn)/\(key)")
        lock.lock(); c.pipelines[fn] = p; lock.unlock()
        return p
    }

    /// A hash of the generated source, used to key the process-wide pipeline cache.
    /// The generated source is the key: `MetalContext` keys its pipeline cache by string, and a hash
    /// collision there would silently hand back the wrong kernel. This is looked up once per compiled
    /// query per function (the `Compiled.pipelines` map absorbs the rest), so the length costs nothing.
    static func sourceKey(_ s: String) -> String { s }

    static func makeArray(_ t: ExprType, length: Int, values: MetalArrowBuffer, validity: MetalArrowBuffer?,
                          ctx: MetalContext, pendingLength: MetalArrowBuffer? = nil, capacity: Int = 0) throws -> AnyMetalArray {
        func finishPrim<T: ArrowPrimitive>(_ a: MetalArray<T>) {
            if let lb = pendingLength {
                a.capacityLength = capacity
                a.deferLength(from: lb) { [weak a] in
                    guard let a else { return }
                    if let v = a.validity { a._nullCount = a._length - Bitmap.popcount(v.typed(UInt8.self), bits: a._length) }
                }
            } else {
                a.recomputeNullCount()
            }
        }
        func wrapPrim<T: ArrowPrimitive>(_: T.Type) -> AnyMetalArray {
            let a = MetalArray<T>(length: pendingLength == nil ? length : 0, nullCount: 0,
                                  validity: validity, values: values, context: ctx)
            finishPrim(a)
            return arrowMetalWrap(a)
        }
        switch t {
        case .int8: return wrapPrim(Int8.self)
        case .int16: return wrapPrim(Int16.self)
        case .int32: return wrapPrim(Int32.self)
        case .int64: return wrapPrim(Int64.self)
        case .uint8: return wrapPrim(UInt8.self)
        case .uint16: return wrapPrim(UInt16.self)
        case .uint32: return wrapPrim(UInt32.self)
        case .uint64: return wrapPrim(UInt64.self)
        case .float32: return wrapPrim(Float.self)
        case .float64: return wrapPrim(Double.self)
        case .boolean:
            let a = MetalBooleanArray(length: pendingLength == nil ? length : 0, nullCount: 0,
                                      validity: validity, values: values, context: ctx)
            if let lb = pendingLength { a.markPending(capacity: capacity, lengthBuffer: lb) }
            else { a.recomputeNullCount() }
            return .boolean(a)
        case .utf8: throw ExprError.unsupported("a utf8 output column")
        }
    }

    static func byteWidth(_ t: ExprType) -> Int { t == .boolean ? 0 : t.bitWidth / 8 }
}

/// `wrap` lives in the C ABI target; this is the same switch for the Swift target.
func arrowMetalWrap<T: ArrowPrimitive>(_ a: MetalArray<T>) -> AnyMetalArray {
    switch a {
    case let x as MetalArray<Int8>: return .int8(x)
    case let x as MetalArray<UInt8>: return .uint8(x)
    case let x as MetalArray<Int16>: return .int16(x)
    case let x as MetalArray<UInt16>: return .uint16(x)
    case let x as MetalArray<Int32>: return .int32(x)
    case let x as MetalArray<UInt32>: return .uint32(x)
    case let x as MetalArray<Int64>: return .int64(x)
    case let x as MetalArray<UInt64>: return .uint64(x)
    case let x as MetalArray<Float>: return .float32(x)
    case let x as MetalArray<Double>: return .float64(x)
    default: fatalError("unreachable")
    }
}
