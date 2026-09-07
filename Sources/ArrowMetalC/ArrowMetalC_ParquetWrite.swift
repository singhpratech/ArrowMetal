import Foundation
import ArrowMetal

// The writer half of the Parquet C ABI. See docs/PARQUET.md for the subset it covers.

/// Argument-validation failure: leave a message naming the function and the argument behind, so
/// `am_last_error()` never hands the caller an unrelated earlier failure.
private func pqWriteBadArgument(_ detail: String) -> Int32 {
    Thread.current.threadDictionary["ArrowMetalC.lastError"] = "am_parquet_write: \(detail)"
    return 2
}

@_cdecl("am_parquet_write")
public func am_parquet_write(_ path: UnsafePointer<CChar>?,
                             _ columns: UnsafePointer<OpaquePointer?>?,
                             _ names: UnsafePointer<UnsafePointer<CChar>?>?,
                             _ nColumns: Int64,
                             _ compression: UnsafePointer<CChar>?,
                             _ dictionary: Int32,
                             _ rowGroupSize: Int64) -> Int32 {
    guard let path else { return pqWriteBadArgument("`path` is NULL") }
    guard let columns else { return pqWriteBadArgument("`columns` is NULL") }
    guard let names else { return pqWriteBadArgument("`names` is NULL") }
    guard nColumns > 0 else {
        return pqWriteBadArgument("`n_columns` is \(nColumns); a file needs at least one column")
    }
    do {
        var arrays: [AnyMetalArray] = []
        var columnNames: [String] = []
        for i in 0..<Int(nColumns) {
            guard let h = columns[i] else { return pqWriteBadArgument("`columns`[\(i)] is NULL") }
            guard let n = names[i] else { return pqWriteBadArgument("`names`[\(i)] is NULL") }
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
