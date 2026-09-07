import Foundation

// Streaming joins.
//
// **Broadcast join** — the build side fits in memory, the probe side is streamed. The build side is
// read once into one Metal-resident batch; every probe batch runs the existing GPU hash join against
// it and goes straight to the sink. Memory is the build side plus one batch. This is the right plan
// whenever one input is a dimension table, however large the other one is.
//
// **Grace hash join** — both sides are larger than memory. Both are streamed once and partitioned by
// a hash of the join key into `partitions` Arrow IPC files per side. Equal keys always land in the
// same partition, so the join of the whole is the union of the per-partition joins, and each
// partition is small enough to run as an in-memory GPU hash join. Two passes over each input plus
// one write and one read of each, and memory is one partition pair at a time.

/// Broadcast join: a build side held in memory, probe batches streamed through the GPU hash join.
public final class BroadcastJoinOperator: StreamOperator {
    public let build: MetalRecordBatch
    public let probeKey: String
    public let buildKey: String
    public let kind: JoinKind
    public let sink: StreamSink
    public let filter: Expr?
    private var rows = 0

    public init(build: MetalRecordBatch, probeKey: String, buildKey: String, kind: JoinKind = .inner,
                sink: StreamSink, filter: Expr? = nil) {
        self.build = build
        self.probeKey = probeKey
        self.buildKey = buildKey
        self.kind = kind
        self.sink = sink
        self.filter = filter
    }

    public func process(_ batch: MetalRecordBatch) throws -> Any? {
        var work = batch
        if let f = filter {
            work = try streamFilterProject(batch, filter: f, projections: nil,
                                           context: batch.firstContext ?? .shared)
        }
        guard work.length > 0 else { return nil }
        let out = try work.join(build, on: probeKey, rightKey: buildKey, kind: kind)
        _ = out.length
        return out
    }

    /// The sink writes bytes the GPU stage has already flushed; no command buffer.
    public var mergeUsesGPU: Bool { false }

    public func merge(_ partial: Any) throws {
        guard let b = partial as? MetalRecordBatch else { return }
        rows += b.length
        try sink.write(b)
    }

    public func finish() throws -> StreamResult {
        try sink.finish()
        var r = StreamResult()
        r.rowsOut = rows
        if let c = sink as? CollectingSink { r.batch = try c.table() }
        return r
    }
}

/// Reads a whole source into one Metal-resident batch — the build side of a broadcast join.
public func loadBuildSide(_ source: BatchSource) throws -> MetalRecordBatch {
    var batches: [MetalRecordBatch] = []
    while let b = try source.nextBatch() { batches.append(b) }
    guard !batches.isEmpty else { throw ArrowMetalError.invalidArrowArray("build side is empty") }
    return try concatBatches(batches)
}

// MARK: - Grace hash join

/// How a grace hash join was run and what it cost.
public struct GraceJoinStats: Sendable {
    public var partitions = 0
    public var leftRows = 0
    public var rightRows = 0
    public var outputRows = 0
    public var spilledBytes: Int64 = 0
    public var partitionNanos: UInt64 = 0
    public var joinNanos: UInt64 = 0
    public var wallNanos: UInt64 = 0
}

/// Grace hash join over two streamed inputs, neither of which fits in memory.
///
/// Both sides are partitioned by `hash(key) % partitions` into Arrow IPC files under `scratch`, then
/// each partition pair is loaded and joined with the in-memory GPU hash join. The result equals the
/// in-memory join of the whole inputs (the partition function depends only on the key, so a matching
/// pair can never be split across partitions); the row order differs, as it does for any hash join.
///
/// `partitions` must be a power of two. Keys must be int32 or int64 on both sides, as the GPU hash
/// join requires; null keys never match and are dropped during partitioning.
@discardableResult
public func graceHashJoin(left: BatchSource, right: BatchSource,
                          leftKey: String, rightKey: String,
                          kind: JoinKind = .inner,
                          partitions: Int = 16,
                          scratch: URL,
                          sink: StreamSink,
                          deleteSpill: Bool = true,
                          context: MetalContext = .shared) throws -> GraceJoinStats {
    var p = 1
    while p < partitions { p <<= 1 }
    var stats = GraceJoinStats()
    stats.partitions = p
    let t0 = machNow()
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)

    let leftSink = try PartitionedIPCSink(directory: scratch, prefix: "L", parts: p)
    let rightSink = try PartitionedIPCSink(directory: scratch, prefix: "R", parts: p)
    defer {
        if deleteSpill { leftSink.removeFiles(); rightSink.removeFiles() }
    }

    let pt = machNow()
    stats.leftRows = try partition(left, key: leftKey, into: leftSink, parts: p, context: context)
    stats.rightRows = try partition(right, key: rightKey, into: rightSink, parts: p, context: context)
    try leftSink.finish()
    try rightSink.finish()
    stats.partitionNanos = nanos(since: pt)
    for u in leftSink.urls + rightSink.urls {
        let attrs = try? FileManager.default.attributesOfItem(atPath: u.path)
        stats.spilledBytes += (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    }

    let jt = machNow()
    for i in 0..<p {
        guard let lu = leftSink.url(part: i) else { continue }
        guard let ru = rightSink.url(part: i) else {
            // No build rows in this partition: an inner join emits nothing, a left join emits the left
            // rows with null right columns, which `join` produces against an empty build side.
            if kind == .left {
                let ls = try IPCFileSource(url: lu, context: context)
                defer { ls.close() }
                while let lb = try ls.nextBatch() {
                    let empty = try emptyLike(lb, keyColumn: leftKey, rightKey: rightKey, context: context)
                    let out = try lb.join(empty, on: leftKey, rightKey: rightKey, kind: .left)
                    try sink.write(out)
                    stats.outputRows += out.length
                }
            }
            continue
        }
        let rs = try IPCFileSource(url: ru, context: context)
        var buildBatches: [MetalRecordBatch] = []
        while let b = try rs.nextBatch() { buildBatches.append(b) }
        rs.close()
        guard !buildBatches.isEmpty else { continue }
        let build = try concatBatches(buildBatches)

        let ls = try IPCFileSource(url: lu, context: context)
        while let lb = try ls.nextBatch() {
            guard lb.length > 0 else { continue }
            let out = try context.batch { try lb.join(build, on: leftKey, rightKey: rightKey, kind: kind) }
            try sink.write(out)
            stats.outputRows += out.length
        }
        ls.close()
    }
    stats.joinNanos = nanos(since: jt)
    try sink.finish()
    stats.wallNanos = nanos(since: t0)
    return stats
}

/// A zero-row batch with the build side's schema, so a `.left` join against an empty partition still
/// produces the right columns.
private func emptyLike(_ probe: MetalRecordBatch, keyColumn: String, rightKey: String,
                       context: MetalContext) throws -> MetalRecordBatch {
    guard let k = probe[keyColumn] else { throw ArrowMetalError.invalidArrowArray("no column named \(keyColumn)") }
    let empty = try k.rebuild([], context: context)
    return try MetalRecordBatch(names: [rightKey], columns: [empty])
}

/// Splits every batch of `source` by `hash(key) & (parts - 1)` and writes the pieces to `into`.
private func partition(_ source: BatchSource, key: String, into sink: PartitionedIPCSink,
                       parts: Int, context: MetalContext) throws -> Int {
    var rows = 0
    while let b = try source.nextBatch() {
        guard b.length > 0 else { continue }
        guard let k = b[key] else { throw ArrowMetalError.invalidArrowArray("no join key column \(key)") }
        rows += b.length
        let pid = try partitionIds(k, parts: parts)
        for i in 0..<parts {
            let mask = try pid.compare(.eq, Int32(i))
            let part = try b.filter(mask)
            if part.length > 0 { try sink.write(part: i, part) }
        }
    }
    source.close()
    return rows
}

/// Partition id per row: a multiplicative hash of the key, folded to `parts` buckets.
///
/// The hash has to depend on the whole key, not its low bits, or an id column that is a multiple of
/// the partition count lands entirely in one partition. Multiply by a large odd constant (wrapping),
/// then take the *high* bits of the product by shifting right.
func partitionIds(_ key: AnyMetalArray, parts: Int) throws -> MetalArray<Int32> {
    let bits = Int(log2(Double(parts)).rounded())
    let wide: MetalArray<Int64>
    switch key {
    case .int32(let a): wide = try a.cast(to: Int64.self)
    case .int64(let a): wide = a
    case .int16(let a): wide = try a.cast(to: Int64.self)
    case .int8(let a): wide = try a.cast(to: Int64.self)
    case .uint32(let a): wide = try a.cast(to: Int64.self)
    case .uint16(let a): wide = try a.cast(to: Int64.self)
    case .uint8(let a): wide = try a.cast(to: Int64.self)
    case .temporal(let t): wide = try t.int64Values()
    default: throw ArrowMetalError.unsupportedType("grace join keys must be integers, got \(key.arrowFormat)")
    }
    // (key * 0x9E3779B97F4A7C15) >>> (64 - bits): the top `bits` of a Fibonacci-hashed key.
    let mixed = try wide.multiply(Int64(bitPattern: 0x9E37_79B9_7F4A_7C15))
    // Drop the sign bit first, then take the top `bits` of the product.
    let unsigned = try mixed.bitwiseAnd(Int64.max)
    let shifted = try unsigned.shiftRight(Int64(63 - bits))
    let masked = try shifted.bitwiseAnd(Int64(parts - 1))
    return try masked.cast(to: Int32.self)
}
