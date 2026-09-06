import Foundation
import Metal

// List assembly: Dremel, for one level of repetition.

extension ParquetLeafData {
    /// Assembles a `list<primitive>` column from the leaf's definition and repetition levels.
    ///
    /// A new row starts wherever the repetition level is 0; an element exists wherever the definition
    /// level reaches the repeated node's level (`dRep`); and the row itself is null when its first
    /// entry's definition level does not even reach the enclosing group. Two prefix sums number the rows
    /// and the elements, the row starts write the offsets, and the child array is the leaf array
    /// compacted to the element positions with the package's existing `filter`.
    func listArray(repeatedDefinition dRep: Int, outerNullable: Bool) throws -> AnyMetalArray {
        let ctx = context
        guard let def = defLevels else {
            throw ParquetError.malformed("a list column must have definition levels")
        }
        let n = levels
        guard n > 0 else {
            let off = try MetalArrowBuffer.allocate(byteCount: 8, context: ctx)
            return .list(MetalListArray(length: 0, nullCount: 0, validity: nil, offsets: off,
                                        values: try arrowArray(), context: ctx))
        }
        // A LIST wrapper whose repeated node produced no repetition levels (a required, single-element
        // list) leaves every entry starting its own row.
        let rep: MetalArrowBuffer
        if let r = repLevels { rep = r }
        else { rep = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: true, context: ctx) }

        let elemFlag = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
        let rowFlag = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
        let elemByte = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1), zeroed: true, context: ctx)
        try ParquetTypeMap.run(file, ctx, "pq_list_flags", count: n) { enc in
            enc.setBuffer(def.mtl, offset: def.offset, index: 0)
            enc.setBuffer(rep.mtl, offset: rep.offset, index: 1)
            Dispatch.setUInt(enc, n, index: 2)
            Dispatch.setUInt(enc, dRep, index: 3)
            enc.setBuffer(elemFlag.mtl, offset: elemFlag.offset, index: 4)
            enc.setBuffer(rowFlag.mtl, offset: rowFlag.offset, index: 5)
            enc.setBuffer(elemByte.mtl, offset: elemByte.offset, index: 6)
        }
        let elemScan = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
        let rowScan = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
        let totalElements = try file.scanU32(ctx, input: elemFlag, output: elemScan, n: n + 1)
        let numRows = try file.scanU32(ctx, input: rowFlag, output: rowScan, n: n + 1)

        let offsets = try MetalArrowBuffer.allocate(byteCount: Swift.max((numRows + 1) * 4, 8), zeroed: true, context: ctx)
        let validBytes = try MetalArrowBuffer.allocate(byteCount: Swift.max(numRows, 1), zeroed: true, context: ctx)
        let rowTotal = try MetalArrowBuffer.allocate(byteCount: 4, context: ctx)
        rowTotal.mutableTyped(UInt32.self)[0] = UInt32(numRows)
        try ParquetTypeMap.run(file, ctx, "pq_list_offsets", count: n + 1) { enc in
            enc.setBuffer(def.mtl, offset: def.offset, index: 0)
            enc.setBuffer(rep.mtl, offset: rep.offset, index: 1)
            enc.setBuffer(rowScan.mtl, offset: rowScan.offset, index: 2)
            enc.setBuffer(elemScan.mtl, offset: elemScan.offset, index: 3)
            Dispatch.setUInt(enc, n, index: 4)
            Dispatch.setUInt(enc, dRep, index: 5)
            enc.setBuffer(rowTotal.mtl, offset: rowTotal.offset, index: 6)
            enc.setBuffer(offsets.mtl, offset: offsets.offset, index: 7)
            enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 8)
        }

        // The child: the leaf array restricted to the entries that carry an element.
        let leafArray = try arrowArray()
        let child: AnyMetalArray
        if totalElements == n {
            child = leafArray
        } else {
            let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 4),
                                                     zeroed: true, context: ctx)
            try ParquetTypeMap.run(file, ctx, "pq_bytes_to_bitmap", count: (n + 31) / 32) { enc in
                enc.setBuffer(elemByte.mtl, offset: elemByte.offset, index: 0)
                Dispatch.setUInt(enc, n, index: 1)
                enc.setBuffer(bits.mtl, offset: bits.offset, index: 2)
            }
            let mask = MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: bits, context: ctx)
            child = try leafArray.filter(mask)
        }

        var validity: MetalArrowBuffer? = nil
        var nulls = 0
        if outerNullable {
            let bits = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: numRows), 4),
                                                     zeroed: true, context: ctx)
            try ParquetTypeMap.run(file, ctx, "pq_bytes_to_bitmap", count: (numRows + 31) / 32) { enc in
                enc.setBuffer(validBytes.mtl, offset: validBytes.offset, index: 0)
                Dispatch.setUInt(enc, numRows, index: 1)
                enc.setBuffer(bits.mtl, offset: bits.offset, index: 2)
            }
            try ctx.syncPoint()
            let set = Bitmap.popcount(bits.typed(UInt8.self), bits: numRows)
            if set < numRows { validity = bits; nulls = numRows - set }
        }
        return .list(MetalListArray(length: numRows, nullCount: nulls, validity: validity,
                                    offsets: offsets, values: child, context: ctx))
    }
}
