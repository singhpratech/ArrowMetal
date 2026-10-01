import Foundation
import Metal

/// Helpers shared by kernel wrappers.
enum Dispatch {
    static let threadgroupSize = 256

    /// Pipeline for a type-specialised kernel, keyed by source family + type + function.
    ///
    /// `source` is an autoclosure so a cache hit never pays for generating the MSL (see
    /// `MetalContext.pipeline`). Pass the generator call directly, not a `let` bound above.
    static func pipeline(_ ctx: MetalContext, family: String, source: @autoclosure () -> String,
                         function: String, type: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: source(), function: function, cacheKey: "\(family)/\(type)/\(function)")
    }

    /// Grid of `count` threads in threadgroups of 256 (bounds checks inside kernels handle the tail).
    ///
    /// From 2^32 - 255 elements the grid is 2^24 threadgroups, 2^32 threads, which a plain `(groups, 1, 1)`
    /// grid wraps to zero (see `perGroup`). There it is folded into `rowGrid`'s rows, and every kernel reads
    /// its index through the fold (`foldGridPositions`). Past 2^32 threads a `uint` index cannot address
    /// the elements at all, and the call stops with a message instead of running a wrapped grid.
    static func dispatch1D(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, count: Int) {
        let groups = (count + threadgroupSize - 1) / threadgroupSize
        guard let grid = rowGrid(threadgroups: groups) else {
            fatalError("ArrowMetal: a kernel over \(count) elements needs more than 2^32 thread positions; " +
                       "arrays above 2^32 - 1 elements are not supported")
        }
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
    }

    /// `threadgroups` threadgroups of 256 threads in one logical row, folded like `dispatch1D`, for the
    /// kernels that size their grid themselves (the two-level scans). Throws instead of wrapping when the
    /// grid needs more than 2^32 threads.
    static func dispatchRows(_ enc: MTLComputeCommandEncoder, threadgroups: Int) throws {
        guard let grid = rowGrid(threadgroups: threadgroups) else {
            throw ArrowMetalError.invalidArrowArray(
                "a grid of \(threadgroups) threadgroups of \(threadgroupSize) threads is past the 2^32 thread " +
                "positions a kernel can address; arrays this close to 2^32 elements are not supported here")
        }
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
    }

    /// The grid `dispatch1D` and `dispatchRows` launch for `threadgroups` threadgroups of 256 threads: the
    /// plain `(threadgroups, 1, 1)` below `rowFoldThreads`, otherwise rows of `foldWidth` threadgroups.
    /// Nil when the threads would not fit the 2^32 positions a kernel's `uint` index holds.
    ///
    /// A folded grid launches whole rows. At the hardware limit the count is 2^24 threadgroups, which is
    /// exactly 256 rows, so the folded grid launches the same threadgroups as the plain one would; only
    /// a lowered `rowFoldThreads` (the tests) launches a partial last row's spare threadgroups, whose
    /// threads fail the kernels' `i < n` checks. A kernel that writes one output per threadgroup sizes
    /// that output with `launchedThreadgroups`.
    static func rowGrid(threadgroups g: Int) -> MTLSize? {
        let threads = g * threadgroupSize
        if threads < rowFoldThreads { return MTLSize(width: g, height: 1, depth: 1) }
        guard threads <= 1 << 32 else { return nil }
        return MTLSize(width: foldWidth, height: (g + foldWidth - 1) / foldWidth, depth: 1)
    }

    /// Threadgroups `rowGrid` launches for `threadgroups` (more than asked only on a folded grid's last row).
    static func launchedThreadgroups(_ threadgroups: Int) -> Int {
        guard let g = rowGrid(threadgroups: threadgroups) else { return threadgroups }
        return g.width * g.height
    }

    /// The grid width in threads at which `dispatch1D` and `dispatchRows` fold. The hardware limit is 2^32;
    /// the tests lower it to run row-wise kernels through the folded grid at small sizes.
    nonisolated(unsafe) static var rowFoldThreads = 1 << 32

    /// log2 of the threads in one folded row: `foldWidth` threadgroups of `threadgroupSize` threads.
    static let rowFoldShift = foldShift + threadgroupSize.trailingZeroBitCount

    /// Rewrites every kernel in `source` to read its scalar grid positions through the fold.
    ///
    /// `uint i [[thread_position_in_grid]]` becomes `uint2 am_fold_i [[thread_position_in_grid]]` and the
    /// body starts with `uint i = (am_fold_i.y << 24u) + am_fold_i.x;`; `uint t [[threadgroup_position_in_grid]]`
    /// becomes the same with a shift of 16. On a `(width, 1, 1)` grid `.y` is 0 and the index is the old
    /// one, so a kernel dispatched any other way runs unchanged; on a folded `rowGrid` it is the row-major
    /// position. The kernels keep their scalar signatures in their own sources; `MetalContext.pipeline`
    /// applies this before compiling. A parameter written inside a `#define` (a shared argument list) gets
    /// its declaration in the body of every kernel that names the macro.
    ///
    /// Metal wants a kernel's position inputs all scalar or all of one vector width, so the scalar
    /// `thread_position_in_threadgroup`, `threads_per_threadgroup`, `threads_per_grid` and
    /// `threadgroups_per_grid` become `uint2` too, read as `.x`. The last two are a row's width on a folded
    /// grid; the kernels that read them size their own grids and are never folded.
    static func foldGridPositions(_ source: String) -> String {
        guard source.contains("_position_in_grid") else { return source }
        let ns = source as NSString
        let all = NSRange(location: 0, length: ns.length)
        let matches = gridPositionPattern.matches(in: source, range: all)
        if matches.isEmpty { return source }
        let defines = definePattern.matches(in: source, range: all).map { ($0.range, ns.substring(with: $0.range(at: 1))) }
        func define(at loc: Int) -> String? { defines.first { NSLocationInRange(loc, $0.0) }?.1 }
        func bodyStart(after loc: Int) -> Int? {
            let b = ns.range(of: "{", options: [], range: NSRange(location: loc, length: ns.length - loc))
            return b.location == NSNotFound ? nil : b.location + 1
        }
        // Edits as (location, replaced length, text), applied from the end so locations stay valid.
        var edits: [(Int, Int, String)] = []
        var macroDecls: [String: [String]] = [:]
        for m in matches {
            let name = ns.substring(with: m.range(at: 1))
            let attribute = ns.substring(with: m.range(at: 2))
            edits.append((m.range.location, m.range.length, "uint2 am_fold_\(name) [[\(attribute)]]"))
            let decl: String
            switch attribute {
            case "thread_position_in_grid":
                decl = " uint \(name) = (am_fold_\(name).y << \(rowFoldShift)u) + am_fold_\(name).x;"
            case "threadgroup_position_in_grid":
                decl = " uint \(name) = (am_fold_\(name).y << \(foldShift)u) + am_fold_\(name).x;"
            default:
                decl = " uint \(name) = am_fold_\(name).x;"
            }
            if let macro = define(at: m.range.location) { macroDecls[macro, default: []].append(decl) }
            else if let b = bodyStart(after: m.range.location + m.range.length) { edits.append((b, 0, decl)) }
        }
        for (macro, decls) in macroDecls {
            let use = try! NSRegularExpression(pattern: "\\b\(NSRegularExpression.escapedPattern(for: macro))\\b")
            for u in use.matches(in: source, range: all) where define(at: u.range.location) == nil {
                if let b = bodyStart(after: u.range.location + u.range.length) { edits.append((b, 0, decls.joined())) }
            }
        }
        let out = NSMutableString(string: source)
        // Insertions at one brace keep their order: sort by location, then by edit order, and apply backwards.
        for (loc, len, text) in edits.enumerated().sorted(by: { ($0.element.0, $0.offset) < ($1.element.0, $1.offset) })
            .map(\.element).reversed() {
            out.replaceCharacters(in: NSRange(location: loc, length: len), with: text)
        }
        return out as String
    }

    private static let gridPositionPattern = try! NSRegularExpression(
        pattern: #"\buint\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\[\s*(thread_position_in_grid|threadgroup_position_in_grid|thread_position_in_threadgroup|threads_per_threadgroup|threads_per_grid|threadgroups_per_grid)\s*\]\]"#)
    /// A `#define NAME` with its backslash-continued lines.
    private static let definePattern = try! NSRegularExpression(
        pattern: #"#define[ \t]+([A-Za-z_][A-Za-z0-9_]*)(?:[^\n]*\\\n)*[^\n]*"#)

    /// One threadgroup of `threadsPerGroup` threads per group, for kernels that reduce one group each.
    ///
    /// The GPU holds a grid dimension's thread count (threadgroups × threads per threadgroup) in 32 bits:
    /// at 2^24 threadgroups of 256 threads the width is 2^32, which wraps, and the dispatch then runs
    /// `count mod 2^24` threadgroups with no error. When the width would reach 2^32 the groups are folded
    /// into rows of `foldWidth` threadgroups. The kernel takes `uint2 tgid2 [[threadgroup_position_in_grid]]`,
    /// derives its group as `foldedGroupMSL` and returns when that is `>= count`. Below the limit the grid
    /// is the plain `(count, 1, 1)`, where `tgid2.y` is 0.
    static func perGroup(_ enc: MTLComputeCommandEncoder, count: Int, threadsPerGroup: Int = threadgroupSize) {
        enc.dispatchThreadgroups(perGroupGrid(count: count, threadsPerGroup: threadsPerGroup),
                                 threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    }

    /// Threadgroups per row once `perGroup` folds the groups: a power of two, so the kernel's group is a
    /// shift and an add. Reading `threadgroups_per_grid` instead was measured at almost twice the time of
    /// a light per-group kernel (a gather over 10M groups of 5 rows: 122.6 ms against 64.5 ms).
    static let foldShift = 16
    static let foldWidth = 1 << foldShift
    /// The group of a threadgroup of a `perGroup` grid, in MSL.
    static let foldedGroupMSL = "((tgid2.y << \(foldShift)u) + tgid2.x)"
    /// The grid width in threads at which `perGroup` folds. The hardware limit is 2^32; the tests lower it
    /// to run every per-group kernel through the folded grid at small group counts.
    nonisolated(unsafe) static var foldThreads = 1 << 32

    static func perGroupGrid(count: Int, threadsPerGroup: Int = threadgroupSize) -> MTLSize {
        let c = Swift.max(count, 1)
        if c * threadsPerGroup < foldThreads { return MTLSize(width: c, height: 1, depth: 1) }
        return MTLSize(width: foldWidth, height: (c + foldWidth - 1) / foldWidth, depth: 1)
    }

    /// Binds an element count for a kernel's `device const uint* nPtr` argument. A pending array (its length
    /// still being decided by GPU work in the open batch) binds its length buffer, so the count flows on the
    /// GPU without a sync; otherwise the known length is passed inline.
    static func setLength(_ enc: MTLComputeCommandEncoder, _ n: Int, _ lengthBuffer: MetalArrowBuffer?, index: Int) {
        if let lb = lengthBuffer { enc.setBuffer(lb.mtl, offset: lb.offset, index: index) }
        else { var u = UInt32(n); enc.setBytes(&u, length: 4, index: index) }
    }

    static func setUInt(_ enc: MTLComputeCommandEncoder, _ v: Int, index: Int) {
        var u = UInt32(v)
        enc.setBytes(&u, length: 4, index: index)
    }

    static func setScalar<T: ArrowPrimitive>(_ enc: MTLComputeCommandEncoder, _ v: T, index: Int) {
        withUnsafeBytes(of: v) { enc.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    static func checkLength(_ n: Int) throws {
        guard n <= Int(UInt32.max) else { throw ArrowMetalError.invalidArrowArray("arrays above 2^32 elements are not supported yet") }
    }

    /// The largest row number an index array holds. Index arrays (argsort, top-k, partition_nth, the
    /// lexsort, join indices, ranks, representative rows) are UInt32, so this is 2^32 - 1; the tests lower
    /// it to reach the refusal below at small sizes.
    nonisolated(unsafe) static var maxRowIndex = Int(UInt32.max)

    /// Refuses an index-returning call whose row numbers would pass `maxRowIndex`: `rows` rows are
    /// numbered `0 ..< rows` (`base ..< base + rows` for a call that numbers from an offset), and a row
    /// number that does not fit the UInt32 index type is an error, never a wrapped value.
    static func checkIndexRows(_ rows: Int, base: Int = 0, _ op: String) throws {
        let last = base + rows - 1
        guard last > maxRowIndex else { return }
        throw ArrowMetalError.invalidArrowArray(
            "\(op): row \(last) does not fit the UInt32 index type (row numbers go up to \(maxRowIndex))")
    }

    /// Metal has no `double`. False for Float64 only; the numeric cast and the Float64 overflow check use it to
    /// take their host loop. Arithmetic and sum on Float64 run on the GPU through software binary64
    /// (`DoubleMath`); compare, min, max, filter, take and slice treat the values as raw 64-bit patterns.
    static func runsOnGPU<T: ArrowPrimitive>(_: T.Type) -> Bool { T.self != Double.self }

    /// MSL type used for kernels that only move or order values (filter, take): Float64 becomes `long`.
    static func moveType<T: ArrowPrimitive>(_: T.Type) -> String { T.self == Double.self ? "long" : T.mslType }

    /// Inverse of the MSL `d_key` order-preserving map.
    static func doubleFromKey(_ k: Int64) -> Double {
        let bits = k < 0 ? k ^ 0x7FFF_FFFF_FFFF_FFFF : k
        return Double(bitPattern: UInt64(bitPattern: bits))
    }
}
