import Foundation
import Metal

// Parquet physical type + logical annotation -> Arrow type, and the buffer fix-ups a few of the pairings
// need (INT96 to a nanosecond timestamp, a big-endian FIXED_LEN_BYTE_ARRAY to decimal128, an int32/int64
// decimal to decimal128). Everything here operates on buffers the GPU already filled.

enum ParquetTypeMap {

    /// Builds an Arrow array from row-positioned fixed-width values.
    static func array(leaf: ParquetLeaf, file: ParquetFile, values: MetalArrowBuffer, width: Int,
                      length: Int, validity: MetalArrowBuffer?, nullCount: Int,
                      context ctx: MetalContext) throws -> AnyMetalArray {
        switch leaf.physical {
        case .boolean:
            return .boolean(MetalBooleanArray(length: length, nullCount: nullCount, validity: validity,
                                              values: values, context: ctx))
        case .int32:
            return try int32Array(leaf: leaf, values: values, length: length, validity: validity,
                                  nullCount: nullCount, ctx: ctx, file: file)
        case .int64:
            return try int64Array(leaf: leaf, values: values, length: length, validity: validity,
                                  nullCount: nullCount, ctx: ctx, file: file)
        case .float:
            return .float32(MetalArray<Float>(length: length, nullCount: nullCount, validity: validity,
                                              values: values, context: ctx))
        case .double:
            return .float64(MetalArray<Double>(length: length, nullCount: nullCount, validity: validity,
                                               values: values, context: ctx))
        case .int96:
            let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 8, 8), zeroed: true, context: ctx)
            try run(file, ctx, "pq_int96_to_ns", count: length) { enc in
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(out.mtl, offset: out.offset, index: 1)
                Dispatch.setUInt(enc, length, index: 2)
            }
            let a = MetalArray<Int64>(length: length, nullCount: nullCount, validity: validity, values: out, context: ctx)
            return .temporal(try MetalTemporalArray(type: .timestamp(.nano, timezone: nil), a))
        case .fixedLenByteArray:
            return try flbaArray(leaf: leaf, file: file, values: values, width: width, length: length,
                                 validity: validity, nullCount: nullCount, ctx: ctx)
        case .byteArray:
            throw ParquetError.malformed("BYTE_ARRAY has no fixed width")
        }
    }

    private static func int32Array(leaf: ParquetLeaf, values: MetalArrowBuffer, length: Int,
                                   validity: MetalArrowBuffer?, nullCount: Int, ctx: MetalContext,
                                   file: ParquetFile) throws -> AnyMetalArray {
        switch leaf.logicalType {
        case .date:
            let a = MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx)
            return .temporal(try MetalTemporalArray(type: .date32, a))
        case .time(_, let unit):
            let a = MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx)
            return .temporal(try MetalTemporalArray(type: .time32(unit == .millis ? .milli : .second), a))
        case .decimal(let p, let s):
            let out = try widenToDecimal128(file, ctx, values, length: length, width: 4)
            return .decimal(MetalDecimalArray(type: try ArrowDecimalType(precision: p, scale: s),
                                              length: length, nullCount: nullCount, validity: validity,
                                              values: out, context: ctx))
        case .integer(let bits, let signed):
            // Parquet always stores these in an int32 slot; Arrow wants the narrow type.
            switch (bits, signed) {
            case (8, true): return .int8(try narrow(file, ctx, values, length: length, validity: validity, nullCount: nullCount, as: Int8.self))
            case (16, true): return .int16(try narrow(file, ctx, values, length: length, validity: validity, nullCount: nullCount, as: Int16.self))
            case (8, false): return .uint8(try narrow(file, ctx, values, length: length, validity: validity, nullCount: nullCount, as: UInt8.self))
            case (16, false): return .uint16(try narrow(file, ctx, values, length: length, validity: validity, nullCount: nullCount, as: UInt16.self))
            case (32, false): return .uint32(MetalArray<UInt32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx))
            default: return .int32(MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx))
            }
        default:
            return .int32(MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx))
        }
    }

    private static func int64Array(leaf: ParquetLeaf, values: MetalArrowBuffer, length: Int,
                                   validity: MetalArrowBuffer?, nullCount: Int, ctx: MetalContext,
                                   file: ParquetFile) throws -> AnyMetalArray {
        switch leaf.logicalType {
        case .timestamp(let utc, let unit):
            let a = MetalArray<Int64>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx)
            return .temporal(try MetalTemporalArray(type: .timestamp(arrowUnit(unit), timezone: utc ? "UTC" : nil), a))
        case .time(_, let unit):
            let a = MetalArray<Int64>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx)
            return .temporal(try MetalTemporalArray(type: .time64(unit == .nanos ? .nano : .micro), a))
        case .decimal(let p, let s):
            let out = try widenToDecimal128(file, ctx, values, length: length, width: 8)
            return .decimal(MetalDecimalArray(type: try ArrowDecimalType(precision: p, scale: s),
                                              length: length, nullCount: nullCount, validity: validity,
                                              values: out, context: ctx))
        case .integer(_, let signed) where !signed:
            return .uint64(MetalArray<UInt64>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx))
        default:
            return .int64(MetalArray<Int64>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx))
        }
    }

    private static func flbaArray(leaf: ParquetLeaf, file: ParquetFile, values: MetalArrowBuffer, width: Int,
                                  length: Int, validity: MetalArrowBuffer?, nullCount: Int,
                                  ctx: MetalContext) throws -> AnyMetalArray {
        switch leaf.logicalType {
        case .decimal(let p, let s):
            let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 16, 16), zeroed: true, context: ctx)
            try run(file, ctx, "pq_flba_to_decimal128", count: length) { enc in
                enc.setBuffer(values.mtl, offset: values.offset, index: 0)
                enc.setBuffer(out.mtl, offset: out.offset, index: 1)
                Dispatch.setUInt(enc, length, index: 2)
                Dispatch.setUInt(enc, width, index: 3)
            }
            return .decimal(MetalDecimalArray(type: try ArrowDecimalType(precision: p, scale: s),
                                              length: length, nullCount: nullCount, validity: validity,
                                              values: out, context: ctx))
        case .float16:
            let bits = MetalArray<UInt16>(length: length, nullCount: nullCount, validity: validity,
                                          values: values, context: ctx)
            return .float16(MetalFloat16Array(bits: bits))
        default:
            return .fixedBinary(MetalFixedBinaryArray(byteWidth: width, length: length, nullCount: nullCount,
                                                      validity: validity, values: values, context: ctx))
        }
    }

    static func arrowUnit(_ u: ParquetTimeUnit) -> ArrowTemporalUnit {
        switch u {
        case .millis: return .milli
        case .micros: return .micro
        case .nanos: return .nano
        }
    }

    private static func widenToDecimal128(_ file: ParquetFile, _ ctx: MetalContext, _ values: MetalArrowBuffer,
                                          length: Int, width: Int) throws -> MetalArrowBuffer {
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(length * 16, 16), zeroed: true, context: ctx)
        try run(file, ctx, "pq_int_to_decimal128", count: length) { enc in
            enc.setBuffer(values.mtl, offset: values.offset, index: 0)
            enc.setBuffer(out.mtl, offset: out.offset, index: 1)
            Dispatch.setUInt(enc, length, index: 2)
            Dispatch.setUInt(enc, width, index: 3)
        }
        return out
    }

    /// Narrows an int32 slot to a smaller Arrow integer type through the existing cast kernels.
    private static func narrow<T: ArrowPrimitive>(_ file: ParquetFile, _ ctx: MetalContext, _ values: MetalArrowBuffer,
                                                  length: Int, validity: MetalArrowBuffer?, nullCount: Int,
                                                  as: T.Type) throws -> MetalArray<T> {
        let src = MetalArray<Int32>(length: length, nullCount: nullCount, validity: validity, values: values, context: ctx)
        return try src.cast(to: T.self)
    }

    static func run(_ file: ParquetFile, _ ctx: MetalContext, _ function: String, count: Int,
                    _ bind: (MTLComputeCommandEncoder) -> Void) throws {
        guard count > 0 else { return }
        let p = try file.pso(ctx, function)
        try ctx.run { enc in
            enc.setComputePipelineState(p)
            bind(enc)
            Dispatch.dispatch1D(enc, p, count: count)
        }
    }
}

// MARK: - Assembling the final array

extension ParquetLeafData {
    /// The Arrow array for a flat column.
    func arrowArray() throws -> AnyMetalArray {
        let ctx = context
        switch values {
        case .boolean(let bits):
            return .boolean(MetalBooleanArray(length: levels, nullCount: nullCount, validity: validity,
                                              values: bits, context: ctx))
        case .fixed(let buf, let w):
            return try ParquetTypeMap.array(leaf: leaf, file: file, values: buf, width: w, length: levels,
                                            validity: validity, nullCount: nullCount, context: ctx)
        case .bytes(let offsets, let data):
            let arr = MetalStringArray(length: levels, nullCount: nullCount, validity: validity,
                                       offsets: offsets, data: data, context: ctx)
            let utf8: Bool
            switch leaf.logicalType {
            case .string, .json, .enum: utf8 = true
            default: utf8 = false
            }
            arr.isBinary = !utf8
            return utf8 ? .string(arr) : .binary(arr)
        case .dictionary(let codes, let dictValues):
            let c = MetalArray<Int32>(length: levels, nullCount: nullCount, validity: validity,
                                      values: codes, context: ctx)
            return .dictionary(codes: c, values: dictValues)
        }
    }
}
