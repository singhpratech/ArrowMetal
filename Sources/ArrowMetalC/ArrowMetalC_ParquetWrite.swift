import Foundation
import ArrowMetal

// The writer half of the Parquet C ABI. See docs/PARQUET.md for the subset it covers.

@_cdecl("am_parquet_write")
public func am_parquet_write(_ path: UnsafePointer<CChar>?,
                             _ columns: UnsafePointer<OpaquePointer?>?,
                             _ names: UnsafePointer<UnsafePointer<CChar>?>?,
                             _ nColumns: Int64,
                             _ compression: UnsafePointer<CChar>?,
                             _ dictionary: Int32,
                             _ rowGroupSize: Int64) -> Int32 {
    guard let path, let columns, let names, nColumns > 0 else { return 2 }
    do {
        var arrays: [AnyMetalArray] = []
        var columnNames: [String] = []
        for i in 0..<Int(nColumns) {
            guard let h = columns[i], let n = names[i] else { return 2 }
            arrays.append(Unmanaged<Box>.fromOpaque(UnsafeRawPointer(h)).takeUnretainedValue().a)
            columnNames.append(String(cString: n))
        }
        let codecName = compression.map { String(cString: $0).lowercased() } ?? "snappy"
        let codec: ParquetCodec
        switch codecName {
        case "none", "uncompressed", "": codec = .uncompressed
        case "snappy": codec = .snappy
        default: throw ParquetError.unsupported("the writer only produces UNCOMPRESSED or SNAPPY files, not \(codecName)")
        }
        let opts = ParquetWriteOptions(compression: codec, useDictionary: dictionary != 0,
                                       rowGroupSize: rowGroupSize > 0 ? Int(rowGroupSize) : (1 << 20))
        let batch = try MetalRecordBatch(names: columnNames, columns: arrays)
        try ParquetWriter.write(batch, to: String(cString: path), options: opts)
        return 0
    } catch {
        Thread.current.threadDictionary["ArrowMetalC.lastError"] = "\(error)"
        return 1
    }
}
