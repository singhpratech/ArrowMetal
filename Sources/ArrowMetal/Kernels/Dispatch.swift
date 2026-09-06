import Foundation
import Metal

/// Helpers shared by kernel wrappers.
enum Dispatch {
    static let threadgroupSize = 256

    /// Pipeline for a type-specialised kernel, keyed by source family + type + function.
    static func pipeline(_ ctx: MetalContext, family: String, source: String, function: String, type: String) throws -> MTLComputePipelineState {
        try ctx.pipeline(source: source, function: function, cacheKey: "\(family)/\(type)/\(function)")
    }

    /// Grid of `count` threads in threadgroups of 256 (bounds checks inside kernels handle the tail).
    static func dispatch1D(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, count: Int) {
        let tg = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        let groups = MTLSize(width: (count + threadgroupSize - 1) / threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
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

    /// Metal has no `double`. Arithmetic and sum on Float64 run on the CPU reference path; compare, min, max,
    /// filter, take and slice run on the GPU by treating the values as raw 64-bit patterns.
    static func runsOnGPU<T: ArrowPrimitive>(_: T.Type) -> Bool { T.self != Double.self }

    /// MSL type used for kernels that only move or order values (filter, take): Float64 becomes `long`.
    static func moveType<T: ArrowPrimitive>(_: T.Type) -> String { T.self == Double.self ? "long" : T.mslType }

    /// Inverse of the MSL `d_key` order-preserving map.
    static func doubleFromKey(_ k: Int64) -> Double {
        let bits = k < 0 ? k ^ 0x7FFF_FFFF_FFFF_FFFF : k
        return Double(bitPattern: UInt64(bitPattern: bits))
    }
}
