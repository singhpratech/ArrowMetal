import Foundation

// Split-block bloom filters, for equality filters.
//
// A writer may store one bloom filter per column chunk (`bloom_filter_offset` in the column metadata): a
// Thrift `BloomFilterHeader` followed by a bitset of 32-byte blocks. Every non-null value of the chunk
// was hashed with xxHash64 (seed 0) over its PLAIN encoding — the little-endian bytes of a number, the
// raw bytes of a string without its length prefix — and the hash set 8 bits in one block. A value whose
// 8 bits are not all set is certainly absent, so a `column == literal` filter drops the row group without
// reading a page. A set of bits proves nothing (a false positive); the row group is read as usual.
//
// Only what can be hashed exactly is looked up: integers stored as INT32 / INT64, strings and binaries,
// and floating-point literals that are exactly representable in the column's type and are not zero or
// NaN (both of which have two encodings). Everything else keeps the row group.

/// One column chunk's split-block bloom filter.
public struct ParquetBloomFilter: Sendable {
    /// The bitset: `blocks * 8` little-endian 32-bit words.
    public let words: [UInt32]
    public var blockCount: Int { words.count / 8 }

    static let salt: [UInt32] = [0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
                                 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31]

    /// False when `hash` was certainly never inserted.
    public func mightContain(hash: UInt64) -> Bool {
        let blocks = UInt64(blockCount)
        guard blocks > 0 else { return true }
        let block = Int(((hash >> 32) &* blocks) >> 32)
        let key = UInt32(truncatingIfNeeded: hash)
        for i in 0..<8 {
            let bit = (key &* Self.salt[i]) >> 27
            if words[block * 8 + i] & (UInt32(1) << bit) == 0 { return false }
        }
        return true
    }
}

/// xxHash64, the hash the Parquet bloom filter specifies (seed 0).
public enum XXHash64 {
    static let p1: UInt64 = 0x9E3779B185EBCA87
    static let p2: UInt64 = 0xC2B2AE3D27D4EB4F
    static let p3: UInt64 = 0x165667B19E3779F9
    static let p4: UInt64 = 0x85EBCA77C2B2AE63
    static let p5: UInt64 = 0x27D4EB2F165667C5

    @inline(__always) static func rotl(_ x: UInt64, _ r: UInt64) -> UInt64 { (x << r) | (x >> (64 - r)) }
    @inline(__always) static func round(_ acc: UInt64, _ input: UInt64) -> UInt64 {
        rotl(acc &+ input &* p2, 31) &* p1
    }
    @inline(__always) static func merge(_ acc: UInt64, _ val: UInt64) -> UInt64 {
        (acc ^ round(0, val)) &* p1 &+ p4
    }

    public static func hash(_ bytes: [UInt8], seed: UInt64 = 0) -> UInt64 {
        bytes.withUnsafeBytes { raw -> UInt64 in
            let n = raw.count
            func u64(_ i: Int) -> UInt64 { UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: i, as: UInt64.self)) }
            func u32(_ i: Int) -> UInt64 { UInt64(UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: i, as: UInt32.self))) }
            var i = 0
            var h: UInt64
            if n >= 32 {
                var v1 = seed &+ p1 &+ p2, v2 = seed &+ p2, v3 = seed, v4 = seed &- p1
                while i + 32 <= n {
                    v1 = round(v1, u64(i)); v2 = round(v2, u64(i + 8))
                    v3 = round(v3, u64(i + 16)); v4 = round(v4, u64(i + 24))
                    i += 32
                }
                h = rotl(v1, 1) &+ rotl(v2, 7) &+ rotl(v3, 12) &+ rotl(v4, 18)
                h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4)
            } else {
                h = seed &+ p5
            }
            h = h &+ UInt64(n)
            while i + 8 <= n {
                h ^= round(0, u64(i))
                h = rotl(h, 27) &* p1 &+ p4
                i += 8
            }
            if i + 4 <= n {
                h ^= u32(i) &* p1
                h = rotl(h, 23) &* p2 &+ p3
                i += 4
            }
            while i < n {
                h ^= UInt64(raw[i]) &* p5
                h = rotl(h, 11) &* p1
                i += 1
            }
            h ^= h >> 33; h = h &* p2
            h ^= h >> 29; h = h &* p3
            h ^= h >> 32
            return h
        }
    }
}

extension ParquetFile {

    /// The bloom filter of one column chunk, or nil when there is none or it is not one this reader
    /// understands (split-block, xxHash64, uncompressed is the only kind the format defines today).
    public func bloomFilter(rowGroup g: Int, column c: Int) throws -> ParquetBloomFilter? {
        guard g >= 0, g < metadata.rowGroups.count, c >= 0, c < metadata.rowGroups[g].columns.count else { return nil }
        let meta = metadata.rowGroups[g].columns[c].meta
        guard let off = meta.bloomFilterOffset, off >= 0, Int(off) < fileSize else { return nil }
        var r = ThriftReader(bytes, at: Int(off))
        var numBytes = -1
        var block = false, xxhash = false, uncompressed = false
        // BloomFilterHeader { 1: numBytes; 2: algorithm { 1: BLOCK }; 3: hash { 1: XXHASH };
        //                     4: compression { 1: UNCOMPRESSED } }
        func union(_ r: inout ThriftReader) throws -> Bool {
            var first = false
            try r.readStruct { r, id, t in
                if id == 1 { first = true }
                try r.skip(t)
                return true
            }
            return first
        }
        try r.readStruct { r, id, _ in
            switch id {
            case 1: numBytes = try r.int(); return true
            case 2: block = try union(&r); return true
            case 3: xxhash = try union(&r); return true
            case 4: uncompressed = try union(&r); return true
            default: return false
            }
        }
        let start = r.pos
        guard block, xxhash, uncompressed, numBytes > 0, numBytes % 32 == 0, numBytes <= 128 << 20,
              start + numBytes <= fileSize else { return nil }
        if let len = meta.bloomFilterLength, len > 0, Int(len) < start - Int(off) + numBytes { return nil }
        var words = [UInt32](repeating: 0, count: numBytes / 4)
        for i in words.indices { words[i] = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: start + 4 * i, as: UInt32.self)) }
        return ParquetBloomFilter(words: words)
    }

    /// The xxHash64 of `value` as the column stores it, or nil when the value cannot be hashed exactly.
    static func bloomHash(_ value: ParquetFilter.Value, _ leaf: ParquetLeaf) -> UInt64? {
        func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }
        switch (leaf.physical, value) {
        case (.int32, .int(let v)):
            if case .integer(_, false) = leaf.logicalType {
                guard let u = UInt32(exactly: v) else { return nil }
                return XXHash64.hash(le(Int32(bitPattern: u)))
            }
            guard let i = Int32(exactly: v) else { return nil }
            return XXHash64.hash(le(i))
        case (.int64, .int(let v)):
            // A negative literal is no value of an unsigned column.
            if case .integer(_, false) = leaf.logicalType, v < 0 { return nil }
            return XXHash64.hash(le(v))
        case (.int64, .uint(let u)):
            guard case .integer(_, false) = leaf.logicalType else { return nil }
            return XXHash64.hash(le(u))
        case (.float, .double(let d)):
            guard let f = Float(exactly: d), f != 0, !f.isNaN else { return nil }
            return XXHash64.hash(le(f.bitPattern))
        case (.float, .int(let v)):
            guard let f = Float(exactly: v), f != 0 else { return nil }
            return XXHash64.hash(le(f.bitPattern))
        case (.double, .double(let d)):
            guard d != 0, !d.isNaN else { return nil }
            return XXHash64.hash(le(d.bitPattern))
        case (.double, .int(let v)):
            guard let d = Double(exactly: v), d != 0 else { return nil }
            return XXHash64.hash(le(d.bitPattern))
        case (.byteArray, .string(let s)):
            return XXHash64.hash(Array(s.utf8))
        case (.fixedLenByteArray, .string(let s)) where s.utf8.count == leaf.typeLength:
            if case .decimal = leaf.logicalType { return nil }
            return XXHash64.hash(Array(s.utf8))
        default:
            return nil
        }
    }

    /// True when the bloom filters of `rowGroup` cannot rule out any of the equality filters.
    func bloomFiltersMayMatch(_ filters: [ParquetFilter], rowGroup g: Int) -> Bool {
        for f in filters where f.op == .eq {
            guard let leaf = leaves.first(where: { $0.dottedPath == f.column || $0.name == f.column }),
                  leaf.maxRepetition == 0, !isDecimalOrUnhashable(leaf),
                  let hash = Self.bloomHash(f.value, leaf),
                  let bloom = try? bloomFilter(rowGroup: g, column: leaf.index) else { continue }
            if !bloom.mightContain(hash: hash) { return false }
        }
        return true
    }

    /// Decimal, INT96 and boolean columns: an integer literal says nothing exact about their bytes.
    private func isDecimalOrUnhashable(_ leaf: ParquetLeaf) -> Bool {
        if case .decimal = leaf.logicalType { return true }
        return leaf.physical == .int96 || leaf.physical == .boolean
    }
}
