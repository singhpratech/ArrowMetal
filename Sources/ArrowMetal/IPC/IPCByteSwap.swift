import Foundation

// MARK: - Big-endian bodies

/// Byte swapping for Arrow IPC data whose schema declares `Endianness.Big`.
///
/// Only the body buffers are big-endian; the FlatBuffers metadata is little-endian in every Arrow file.
/// Each buffer is swapped by the width of the values it holds, following the same pre-order walk the
/// reader uses to assign buffers to fields:
///
/// - validity bitmaps, boolean values, `fixed_size_binary` bytes, union type ids and utf8 / binary data
///   are bytes or bits, and are left alone;
/// - offsets (32 or 64 bit), list-view sizes, dense-union offsets, dictionary indices and every
///   fixed-width value are swapped by their element width;
/// - a decimal is one two's-complement integer of 128 or 256 bits stored as 64-bit limbs, most
///   significant limb first: reversing all 16 or 32 bytes swaps each limb and puts the limbs in
///   little-endian order in one step;
/// - `interval[day_time]` is two int32s, `interval[month_day_nano]` an int32, an int32 and an int64;
///   each part is swapped on its own;
/// - a `utf8_view` / `binary_view` view swaps its int32 length, and for an out-of-line view its int32
///   buffer index and offset; the inline bytes and the 4-byte prefix are data and stay as they are;
/// - children are walked recursively, and a dictionary batch is walked with its value field.
enum ArrowIPCByteSwap {
    /// What to do with one body buffer.
    enum Kind: Equatable {
        /// Bytes or bits: nothing to swap.
        case none
        /// Swap every `width`-byte element (2, 4 or 8).
        case width(Int)
        /// Reverse every `width`-byte element end to end (decimal128 / decimal256).
        case reverse(Int)
        /// `interval[month_day_nano]`: int32 months, int32 days, int64 nanoseconds per 16 bytes.
        case monthDayNano
        /// `utf8_view` / `binary_view` 16-byte views.
        case views
    }

    /// The swap for every buffer of a message body whose columns are `fields`, in body order.
    static func plan(_ fields: [ArrowIPCField], variadicCounts: [Int]) -> [Kind] {
        var out: [Kind] = []
        var variadic = 0
        func fixed(_ bytes: Int) -> Kind { bytes <= 1 ? .none : .width(bytes) }
        func walk(_ field: ArrowIPCField) {
            switch field.type {
            case .dictionary(let index, _):
                // The batch carries only the codes; the values are walked with the dictionary batch.
                out += [.none, fixed(index.storage.fixedByteWidth)]
            case .null:
                break
            case .bool, .fixedSizeBinary:
                out += [.none, .none]
            case .int, .float, .float16, .date32, .date64, .time32, .time64, .timestamp, .duration:
                out += [.none, fixed(field.type.storage.fixedByteWidth)]
            case .decimal(_, _, let bits):
                out += [.none, bits <= 64 ? fixed(bits / 8) : .reverse(bits / 8)]
            case .interval(let unit):
                switch unit {
                case .yearMonth, .dayTime: out += [.none, .width(4)]
                case .monthDayNano: out += [.none, .monthDayNano]
                }
            case .utf8, .binary:
                out += [.none, .width(4), .none]
            case .largeUtf8, .largeBinary:
                out += [.none, .width(8), .none]
            case .utf8View, .binaryView:
                out += [.none, .views]
                let n = variadic < variadicCounts.count ? variadicCounts[variadic] : 0
                variadic += 1
                out += Array(repeating: .none, count: n)
            case .list(let item), .map(let item, _):
                out += [.none, .width(4)]
                walk(item)
            case .largeList(let item):
                out += [.none, .width(8)]
                walk(item)
            case .listView(let item):
                out += [.none, .width(4), .width(4)]
                walk(item)
            case .largeListView(let item):
                out += [.none, .width(8), .width(8)]
                walk(item)
            case .fixedSizeList(let item, _):
                out += [.none]
                walk(item)
            case .structure(let children):
                out += [.none]
                children.forEach(walk)
            case .union(let mode, _, let children):
                out += mode == .dense ? [.none, .width(4)] : [.none]
                children.forEach(walk)
            case .runEndEncoded(let ends, let values):
                walk(ends)
                walk(values)
            }
        }
        fields.forEach(walk)
        return out
    }

    /// Swaps `buffers` in place according to `plan`. Extra buffers on either side are left for the
    /// reader's own buffer accounting to reject.
    static func apply(_ plan: [Kind], to buffers: [MetalArrowBuffer]) {
        for (kind, buffer) in zip(plan, buffers) where kind != .none && buffer.byteCount > 0 {
            swap(buffer.mutableTyped(UInt8.self), byteCount: buffer.byteCount, kind)
        }
    }

    static func swap(_ p: UnsafeMutablePointer<UInt8>, byteCount n: Int, _ kind: Kind) {
        let raw = UnsafeMutableRawPointer(p)
        @inline(__always) func swap32(_ at: Int) {
            raw.storeBytes(of: raw.loadUnaligned(fromByteOffset: at, as: UInt32.self).byteSwapped,
                           toByteOffset: at, as: UInt32.self)
        }
        @inline(__always) func swap64(_ at: Int) {
            raw.storeBytes(of: raw.loadUnaligned(fromByteOffset: at, as: UInt64.self).byteSwapped,
                           toByteOffset: at, as: UInt64.self)
        }
        switch kind {
        case .none:
            return
        case .width(2):
            for i in 0..<(n / 2) {
                raw.storeBytes(of: raw.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self).byteSwapped,
                               toByteOffset: 2 * i, as: UInt16.self)
            }
        case .width(4):
            for i in 0..<(n / 4) { swap32(4 * i) }
        case .width(8):
            for i in 0..<(n / 8) { swap64(8 * i) }
        case .width(let w), .reverse(let w):
            guard w > 1 else { return }
            for e in 0..<(n / w) {
                var lo = e * w, hi = lo + w - 1
                while lo < hi { let t = p[lo]; p[lo] = p[hi]; p[hi] = t; lo += 1; hi -= 1 }
            }
        case .monthDayNano:
            for e in 0..<(n / 16) {
                swap32(16 * e)
                swap32(16 * e + 4)
                swap64(16 * e + 8)
            }
        case .views:
            for e in 0..<(n / 16) {
                let at = 16 * e
                swap32(at)
                // Views of 12 bytes or fewer hold their bytes inline; longer ones point into a data buffer.
                if Int32(bitPattern: raw.loadUnaligned(fromByteOffset: at, as: UInt32.self)) > 12 {
                    swap32(at + 8)
                    swap32(at + 12)
                }
            }
        }
    }
}

extension ArrowIPCStorage {
    /// The value width of a fixed-width storage, and 0 for every other layout.
    var fixedByteWidth: Int {
        if case .fixedWidth(let bytes) = self { return bytes }
        return 0
    }
}
