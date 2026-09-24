import Foundation

// Shared pieces of the Delta Lake and Apache Iceberg readers: the column types a table schema names,
// scalar values (partition values, statistics bounds, filter literals), the per-file read that turns one
// Parquet data file into a batch with the table's schema, and the row filter applied after it.
//
// The table formats are metadata over Parquet. The metadata (a JSON log, Avro manifests) is small and is
// parsed on the CPU; every data file goes through the GPU Parquet reader with a projection, and the
// columns the files do not carry (partition columns, columns added after a file was written) are built
// directly in Metal shared memory.

/// Errors from the Delta Lake and Iceberg readers. The message names the table, the feature or the
/// version at fault.
public enum LakehouseError: Error, CustomStringConvertible {
    /// The path is not a table of the expected format.
    case notATable(String)
    /// The table uses a feature this reader does not implement (named in the message).
    case unsupportedFeature(String)
    /// A requested version or snapshot does not exist.
    case notFound(String)
    /// The metadata is inconsistent or cannot be parsed.
    case malformed(String)
    /// A projected or filtered column does not exist, or a filter literal does not fit its column.
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .notATable(let s): return "Not a table: \(s)"
        case .unsupportedFeature(let s): return "Unsupported table feature: \(s)"
        case .notFound(let s): return "Not found: \(s)"
        case .malformed(let s): return "Malformed table metadata: \(s)"
        case .invalidArgument(let s): return "Invalid argument: \(s)"
        }
    }
}

/// A column type as a Delta or Iceberg schema names it, with the Arrow type the reader produces for it.
public indirect enum LakehouseType: Sendable, Equatable, CustomStringConvertible {
    case boolean
    case int8, int16, int32, int64
    case float32, float64
    case string
    case binary
    /// Days since 1970-01-01.
    case date
    /// Microseconds (or nanoseconds) since the epoch; `utc` marks a time-zone-aware timestamp.
    case timestamp(nanos: Bool, utc: Bool)
    /// Microseconds since midnight (Iceberg `time`).
    case time
    case decimal(precision: Int, scale: Int)
    case fixed(Int)
    case uuid
    /// A nested type (struct, list, map); carried so the error can name it.
    case nested(String)

    public var description: String {
        switch self {
        case .boolean: return "boolean"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float32: return "float32"
        case .float64: return "float64"
        case .string: return "string"
        case .binary: return "binary"
        case .date: return "date"
        case .timestamp(let ns, let utc): return "timestamp[\(ns ? "ns" : "us")\(utc ? ", UTC" : "")]"
        case .time: return "time[us]"
        case .decimal(let p, let s): return "decimal(\(p),\(s))"
        case .fixed(let n): return "fixed[\(n)]"
        case .uuid: return "uuid"
        case .nested(let s): return s
        }
    }

    /// The Arrow C Data Interface format of the column this reader produces.
    public var arrowFormat: String? {
        switch self {
        case .boolean: return "b"
        case .int8: return "c"
        case .int16: return "s"
        case .int32: return "i"
        case .int64: return "l"
        case .float32: return "f"
        case .float64: return "g"
        case .string: return "u"
        case .binary: return "z"
        case .date: return "tdD"
        case .timestamp(let ns, let utc): return (ns ? "tsn:" : "tsu:") + (utc ? "UTC" : "")
        case .time: return "ttu"
        case .decimal(let p, let s): return "d:\(p),\(s)"
        case .fixed(let n): return "w:\(n)"
        case .uuid: return "w:16"
        case .nested: return nil
        }
    }

    var isInteger: Bool {
        switch self {
        case .int8, .int16, .int32, .int64: return true
        default: return false
        }
    }
}

/// One column of a table schema.
public struct LakehouseField: Sendable, Equatable {
    public let name: String
    public let type: LakehouseType
    public let nullable: Bool
    /// Iceberg field id, or Delta's `delta.columnMapping.id` when present.
    public let id: Int?
    /// The column's name inside the data files (Delta column mapping); equals `name` otherwise.
    public let physicalName: String

    public init(name: String, type: LakehouseType, nullable: Bool, id: Int? = nil, physicalName: String? = nil) {
        self.name = name
        self.type = type
        self.nullable = nullable
        self.id = id
        self.physicalName = physicalName ?? name
    }
}

/// A scalar: a partition value, a statistics bound or a filter literal, already in the column's
/// storage domain (a date is its day number, a timestamp its micro- or nanosecond count, a decimal its
/// unscaled integer).
enum LakeScalar: Sendable, Equatable {
    case int(Int64)
    case double(Double)
    case string(String)
    case bool(Bool)
    case bytes([UInt8])
    case decimal(ArrowDecimal128)

    /// -1, 0, 1, or nil when the two cannot be ordered (different kinds, or a NaN).
    static func compare(_ a: LakeScalar, _ b: LakeScalar) -> Int? {
        func cmp<T: Comparable>(_ x: T, _ y: T) -> Int { x < y ? -1 : (x == y ? 0 : 1) }
        switch (a, b) {
        case (.int(let x), .int(let y)): return cmp(x, y)
        case (.double(let x), .double(let y)): return (x.isNaN || y.isNaN) ? nil : cmp(x, y)
        case (.int(let x), .double(let y)):
            if y.isNaN { return nil }
            return exactCompare(x, y)
        case (.double(let x), .int(let y)):
            if x.isNaN { return nil }
            return exactCompare(y, x).map { -$0 }
        case (.string(let x), .string(let y)):
            // Byte-wise UTF-8 order, which is what Parquet, Iceberg and Arrow use for strings.
            let a = Array(x.utf8), b = Array(y.utf8)
            if a == b { return 0 }
            return a.lexicographicallyPrecedes(b) ? -1 : 1
        case (.bool(let x), .bool(let y)): return cmp(x ? 1 : 0, y ? 1 : 0)
        case (.bytes(let x), .bytes(let y)):
            if x == y { return 0 }
            return x.lexicographicallyPrecedes(y) ? -1 : 1
        case (.decimal(let x), .decimal(let y)): return x < y ? -1 : (x == y ? 0 : 1)
        default: return nil
        }
    }

    /// Orders an integer against a double without rounding the integer.
    private static func exactCompare(_ i: Int64, _ d: Double) -> Int? {
        // Every Int64 is below 2^63 and at or above -2^63; inside that range `floor(d)` converts exactly.
        if d >= 0x1p63 { return -1 }
        if d < -0x1p63 { return 1 }
        let f = d.rounded(.down)
        let fi = Int64(f)
        if i < fi { return -1 }
        if i > fi { return 1 }
        return f == d ? 0 : -1
    }
}

extension Character {
    /// "0" through "9" and nothing else (`isNumber` also accepts "½", "१" and the like).
    var isASCIIDigit: Bool { asciiValue.map { $0 >= 48 && $0 <= 57 } ?? false }
}

extension CompareOp {
    init(_ op: ParquetFilter.Op) {
        switch op {
        case .eq: self = .eq
        case .ne: self = .ne
        case .lt: self = .lt
        case .le: self = .le
        case .gt: self = .gt
        case .ge: self = .ge
        }
    }

    /// Whether `ordering` (-1, 0, 1 for value vs literal) satisfies the comparison.
    func holds(_ ordering: Int) -> Bool {
        switch self {
        case .eq: return ordering == 0
        case .ne: return ordering != 0
        case .lt: return ordering < 0
        case .le: return ordering <= 0
        case .gt: return ordering > 0
        case .ge: return ordering >= 0
        }
    }
}

/// A filter resolved against a table column: the column, the comparison, and the literal converted to
/// the column's storage domain.
struct LakeFilter {
    let field: LakehouseField
    let op: CompareOp
    let literal: LakeScalar
}

/// Whether a range of values `[lo, hi]` (either end unknown when nil) may contain a value satisfying
/// `op literal`. Conservative: true whenever it cannot tell.
func lakeRangeMayMatch(_ op: CompareOp, lo: LakeScalar?, hi: LakeScalar?, literal: LakeScalar) -> Bool {
    let cLo = lo.flatMap { LakeScalar.compare($0, literal) }
    let cHi = hi.flatMap { LakeScalar.compare($0, literal) }
    switch op {
    case .eq:
        if let c = cLo, c > 0 { return false }
        if let c = cHi, c < 0 { return false }
        return true
    case .ne:
        if let a = cLo, let b = cHi, a == 0, b == 0 { return false }
        return true
    case .lt:
        if let c = cLo, c >= 0 { return false }
        return true
    case .le:
        if let c = cLo, c > 0 { return false }
        return true
    case .gt:
        if let c = cHi, c <= 0 { return false }
        return true
    case .ge:
        if let c = cHi, c < 0 { return false }
        return true
    }
}

// MARK: - Literals

enum LakeTime {
    /// The years a date or timestamp literal may name: wide enough for every date32 value, narrow
    /// enough that no day or microsecond arithmetic below can overflow before it is checked.
    static let yearRange = -5_000_000...5_000_000

    /// Days since 1970-01-01 for a proleptic Gregorian date (`year` within `yearRange`).
    static func days(year: Int, month: Int, day: Int) -> Int64 {
        // Howard Hinnant's days_from_civil.
        let y = month <= 2 ? year - 1 : year
        let era = (y >= 0 ? y : y - 399) / 400
        let yoe = y - era * 400
        let mp = (month + 9) % 12
        let doy = (153 * mp + 2) / 5 + day - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return Int64(era * 146097 + doe - 719468)
    }

    /// (year, month, day) for a day number.
    static func civil(_ days: Int64) -> (year: Int, month: Int, day: Int) {
        let z = Int(days) + 719468
        let era = (z >= 0 ? z : z - 146096) / 146097
        let doe = z - era * 146097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
        let y = yoe + era * 400
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp < 10 ? mp + 3 : mp - 9
        return (m <= 2 ? y + 1 : y, m, d)
    }

    /// Parses `YYYY-MM-DD` into a day number.
    static func parseDate(_ s: String) -> Int64? {
        let t = s.trimmingCharacters(in: .whitespaces)
        let parts = t.split(separator: "-", omittingEmptySubsequences: false)
        // A leading minus would split into an empty first part; years before 0 are not supported here.
        guard parts.count == 3, parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isASCIIDigit) }),
              let y = Int(parts[0]), let m = Int(parts[1]), let d = Int(parts[2]),
              yearRange.contains(y), (1...12).contains(m), (1...31).contains(d) else { return nil }
        return days(year: y, month: m, day: d)
    }

    /// Parses `YYYY-MM-DD[ T]HH:MM:SS[.fffffffff][Z|+HH:MM|-HH:MM]` into a count of `unitsPerSecond`
    /// since the epoch. A timestamp without an offset is taken as UTC (which is also what a
    /// time-zone-free timestamp's wall clock means for storage).
    static func parseTimestamp(_ s: String, unitsPerSecond: Int64) -> Int64? {
        var t = s.trimmingCharacters(in: .whitespaces)
        if t.count == 10, let d = parseDate(t) { return d * 86400 * unitsPerSecond }
        guard t.count >= 19 else { return nil }
        let datePart = String(t.prefix(10))
        guard let day = parseDate(datePart) else { return nil }
        let sep = t[t.index(t.startIndex, offsetBy: 10)]
        guard sep == " " || sep == "T" else { return nil }
        t = String(t.dropFirst(11))
        var offsetSeconds: Int64 = 0
        if t.hasSuffix("Z") {
            t.removeLast()
        } else if t.count >= 6 {
            let tail = t.suffix(6)
            let sign = tail.first
            if sign == "+" || sign == "-" {
                let hm = tail.dropFirst().split(separator: ":")
                if hm.count == 2, hm.allSatisfy({ $0.count == 2 && $0.allSatisfy(\.isASCIIDigit) }),
                   let h = Int64(hm[0]), let m = Int64(hm[1]) {
                    offsetSeconds = (h * 3600 + m * 60) * (sign == "-" ? -1 : 1)
                    t.removeLast(6)
                }
            }
        }
        let hms = t.split(separator: ":", omittingEmptySubsequences: false)
        guard hms.count == 3 else { return nil }
        let secParts = hms[2].split(separator: ".", omittingEmptySubsequences: false)
        guard secParts.count <= 2 else { return nil }
        // Two-digit hour, minute and second fields, so the arithmetic below stays small.
        for part in [hms[0], hms[1], secParts[0]] {
            guard part.count == 2, part.allSatisfy(\.isASCIIDigit) else { return nil }
        }
        guard let h = Int64(hms[0]), let mi = Int64(hms[1]), let sec = Int64(secParts[0]) else { return nil }
        var frac: Int64 = 0
        if secParts.count == 2 {
            var digits = String(secParts[1])
            guard digits.allSatisfy(\.isASCIIDigit), !digits.isEmpty else { return nil }
            let want = unitsPerSecond == 1_000_000_000 ? 9 : 6
            if digits.count > want { digits = String(digits.prefix(want)) }
            while digits.count < want { digits += "0" }
            frac = Int64(digits) ?? 0
            if unitsPerSecond != 1_000_000_000 && unitsPerSecond != 1_000_000 { return nil }
        }
        // `day` is bounded by `yearRange`, so the seconds cannot overflow; the scaling to micro- or
        // nanoseconds can, and a literal outside the column's range is refused.
        let seconds = day * 86400 + h * 3600 + mi * 60 + sec - offsetSeconds
        let (scaled, o1) = seconds.multipliedReportingOverflow(by: unitsPerSecond)
        let (total, o2) = scaled.addingReportingOverflow(frac)
        return (o1 || o2) ? nil : total
    }

    /// Parses a decimal literal ("12.340", "-5", "1e3" is not accepted) into an unscaled value at `scale`.
    static func parseDecimal(_ s: String, scale: Int) -> ArrowDecimal128? {
        var t = s.trimmingCharacters(in: .whitespaces)
        var negative = false
        if t.hasPrefix("-") { negative = true; t.removeFirst() } else if t.hasPrefix("+") { t.removeFirst() }
        let parts = t.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count <= 2, !t.isEmpty else { return nil }
        let whole = String(parts[0])
        var frac = parts.count == 2 ? String(parts[1]) : ""
        // ASCII digits only: `Character.isNumber` also accepts "½" and other scripts' digits.
        guard (whole + frac).allSatisfy(\.isASCIIDigit), !(whole + frac).isEmpty, scale >= 0, scale <= 76 else { return nil }
        // Digits beyond the scale must be zero for the literal to be representable exactly.
        if frac.count > scale {
            guard frac.dropFirst(scale).allSatisfy({ $0 == "0" }) else { return nil }
            frac = String(frac.prefix(scale))
        }
        while frac.count < scale { frac += "0" }
        // A decimal128 holds at most 38 significant digits; a longer literal does not fit any column.
        let digits = (whole + frac).drop { $0 == "0" }
        guard digits.count <= 38 else { return nil }
        var acc = ArrowDecimal128(0)
        let ten = ArrowDecimal128(10)
        for ch in digits {
            acc = acc * ten + ArrowDecimal128(Int64(ch.asciiValue! - 48))
        }
        return negative ? ArrowDecimal128(0) - acc : acc
    }
}

/// Converts a user filter literal to `type`'s storage domain. Throws when the literal cannot be read as
/// a value of that type (a string compared against an integer column, say).
func lakeLiteral(_ v: ParquetFilter.Value, for field: LakehouseField) throws -> LakeScalar {
    func bad() -> LakehouseError {
        let shown: String
        switch v {
        case .int(let i): shown = String(i)
        case .uint(let u): shown = String(u)
        case .double(let d): shown = String(d)
        case .string(let s): shown = "\"\(s)\""
        }
        return .invalidArgument("filter literal \(shown) does not fit column \(field.name) of type \(field.type)")
    }
    switch field.type {
    case .int8, .int16, .int32, .int64:
        switch v {
        case .int(let i): return .int(i)
        // Above Int64.max: no Iceberg or Delta integer column can hold it, and comparing it through a
        // Double would round Int64.max up to the literal and prune rows that match. Refuse it.
        case .uint: throw bad()
        case .double(let d): return .double(d)
        case .string(let s):
            if let i = Int64(s) { return .int(i) }
            if let d = Double(s) { return .double(d) }
            throw bad()
        }
    case .float32, .float64:
        switch v {
        case .int(let i): return .double(Double(i))
        case .uint(let u): return .double(Double(u))
        case .double(let d): return .double(d)
        case .string(let s):
            guard let d = Double(s) else { throw bad() }
            return .double(d)
        }
    case .string:
        guard case .string(let s) = v else { throw bad() }
        return .string(s)
    case .binary, .fixed, .uuid:
        guard case .string(let s) = v else { throw bad() }
        return .bytes(Array(s.utf8))
    case .boolean:
        switch v {
        case .int(let i) where i == 0 || i == 1: return .bool(i == 1)
        case .string(let s) where s.lowercased() == "true" || s.lowercased() == "false": return .bool(s.lowercased() == "true")
        default: throw bad()
        }
    case .date:
        switch v {
        case .int(let i):
            guard Int32(exactly: i) != nil else { throw bad() }
            return .int(i)
        case .string(let s):
            guard let d = LakeTime.parseDate(s), Int32(exactly: d) != nil else { throw bad() }
            return .int(d)
        case .double, .uint: throw bad()
        }
    case .timestamp(let ns, _):
        switch v {
        case .int(let i): return .int(i)
        case .string(let s):
            guard let t = LakeTime.parseTimestamp(s, unitsPerSecond: ns ? 1_000_000_000 : 1_000_000) else { throw bad() }
            return .int(t)
        case .double, .uint: throw bad()
        }
    case .time:
        guard case .int(let i) = v else { throw bad() }
        return .int(i)
    case .decimal(_, let scale):
        switch v {
        case .int(let i):
            guard let d = LakeTime.parseDecimal(String(i), scale: scale) else { throw bad() }
            return .decimal(d)
        case .uint(let u):
            // Exact: a decimal of this scale either holds the integer or parseDecimal refuses it.
            guard let d = LakeTime.parseDecimal(String(u), scale: scale) else { throw bad() }
            return .decimal(d)
        case .string(let s):
            guard let d = LakeTime.parseDecimal(s, scale: scale) else { throw bad() }
            return .decimal(d)
        case .double(let x):
            // Accept a double only when its shortest decimal form is exact at this scale.
            guard x.isFinite, let d = LakeTime.parseDecimal(String(x), scale: scale) else { throw bad() }
            return .decimal(d)
        }
    case .nested(let s):
        throw LakehouseError.invalidArgument("cannot filter on nested column \(field.name) (\(s))")
    }
}

/// Resolves `(column, op, value)` filters against a schema.
func lakeResolveFilters(_ filters: [ParquetFilter], schema: [LakehouseField], table: String) throws -> [LakeFilter] {
    try filters.map { f in
        guard let field = schema.first(where: { $0.name == f.column }) else {
            throw LakehouseError.invalidArgument("filter column \(f.column) is not in the schema of \(table) "
                                                 + "(columns: \(schema.map { $0.name }.joined(separator: ", ")))")
        }
        return LakeFilter(field: field, op: CompareOp(f.op), literal: try lakeLiteral(f.value, for: field))
    }
}

// MARK: - Paths

enum LakePath {
    /// A local filesystem path for a `file:` URI or a plain path. Other schemes are rejected.
    static func local(_ uri: String) throws -> String {
        if uri.hasPrefix("file://") {
            let rest = String(uri.dropFirst("file://".count))
            // file:///abs -> /abs; file://localhost/abs -> /abs
            if rest.hasPrefix("/") { return rest.removingPercentEncoding ?? rest }
            if let slash = rest.firstIndex(of: "/") {
                let p = String(rest[slash...])
                return p.removingPercentEncoding ?? p
            }
            return rest
        }
        if uri.hasPrefix("file:") {
            let p = String(uri.dropFirst("file:".count))
            return p.removingPercentEncoding ?? p
        }
        if let colon = uri.firstIndex(of: ":"), let slash = uri.firstIndex(of: "/"), colon < slash,
           uri[uri.index(after: colon)...].hasPrefix("//") {
            let scheme = uri[..<colon]
            throw LakehouseError.unsupportedFeature("\(scheme):// storage (\(uri)); only local files are read")
        }
        return uri
    }

    static func join(_ base: String, _ rel: String) -> String {
        if rel.hasPrefix("/") { return rel }
        return base.hasSuffix("/") ? base + rel : base + "/" + rel
    }

    static func exists(_ p: String) -> Bool { FileManager.default.fileExists(atPath: p) }

    static func isDirectory(_ p: String) -> Bool {
        var d: ObjCBool = false
        return FileManager.default.fileExists(atPath: p, isDirectory: &d) && d.boolValue
    }
}

// MARK: - Column construction

enum LakeColumns {
    /// A column of `length` copies of `value` (all null when `value` is nil), typed as `type`.
    static func constant(_ type: LakehouseType, _ value: LakeScalar?, length n: Int,
                         context ctx: MetalContext = .shared, column: String) throws -> AnyMetalArray {
        func validity() throws -> MetalArrowBuffer? {
            value == nil ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), zeroed: true, context: ctx) : nil
        }
        func fill<T: ArrowPrimitive>(_: T.Type, _ v: T) throws -> MetalArray<T> {
            let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * T.byteWidth, 1), zeroed: true, context: ctx)
            if value != nil {
                let p = vb.mutableTyped(T.self)
                for i in 0..<n { p[i] = v }
            }
            return MetalArray<T>(length: n, nullCount: value == nil ? n : 0, validity: try validity(), values: vb, context: ctx)
        }
        func intValue() throws -> Int64 {
            switch value {
            case .none: return 0
            case .int(let i)?: return i
            case .bool(let b)?: return b ? 1 : 0
            case .double(let d)?:
                guard let i = Int64(exactly: d) else {
                    throw LakehouseError.malformed("value \(d) for \(type) column \(column) is not an integer in range")
                }
                return i
            default: throw LakehouseError.malformed("value \(String(describing: value)) for \(type) column \(column)")
            }
        }
        func checked<T: FixedWidthInteger>(_: T.Type) throws -> T {
            let i = try intValue()
            guard let v = T(exactly: i) else {
                throw LakehouseError.malformed("value \(i) does not fit \(type) column \(column)")
            }
            return v
        }
        switch type {
        case .int8: return .int8(try fill(Int8.self, try checked(Int8.self)))
        case .int16: return .int16(try fill(Int16.self, try checked(Int16.self)))
        case .int32: return .int32(try fill(Int32.self, try checked(Int32.self)))
        case .int64: return .int64(try fill(Int64.self, try intValue()))
        case .float32, .float64:
            var d = 0.0
            switch value {
            case .double(let x)?: d = x
            case .int(let i)?: d = Double(i)
            case .none: break
            default: throw LakehouseError.malformed("value \(String(describing: value)) for \(type) column \(column)")
            }
            return type == .float32 ? .float32(try fill(Float.self, Float(d))) : .float64(try fill(Double.self, d))
        case .boolean:
            let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), zeroed: true, context: ctx)
            switch value {
            case .none, .bool(false)?: break
            case .bool(true)?:
                let p = vb.mutableTyped(UInt8.self)
                for i in 0..<n { Bitmap.set(p, i) }
            default: throw LakehouseError.malformed("value \(String(describing: value)) for boolean column \(column)")
            }
            return .boolean(MetalBooleanArray(length: n, nullCount: value == nil ? n : 0, validity: try validity(),
                                              values: vb, context: ctx))
        case .string, .binary:
            var bytes: [UInt8] = []
            switch value {
            case .string(let s)?: bytes = Array(s.utf8)
            case .bytes(let b)?: bytes = b
            case .none: break
            default: throw LakehouseError.malformed("value \(String(describing: value)) for \(type) column \(column)")
            }
            let w = bytes.count
            let off = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: true, context: ctx)
            let dat = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * w, 1), zeroed: false, context: ctx)
            let op = off.mutableTyped(Int32.self), dp = dat.mutableTyped(UInt8.self)
            guard n * w < Int(Int32.max) else { throw LakehouseError.malformed("constant column \(column) exceeds 2 GB") }
            for i in 0...n { op[i] = Int32(i * w) }
            if w > 0 { for i in 0..<n { for j in 0..<w { dp[i * w + j] = bytes[j] } } }
            let s = MetalStringArray(length: n, nullCount: value == nil ? n : 0, validity: try validity(),
                                     offsets: off, data: dat, context: ctx)
            if type == .binary { s.isBinary = true; return .binary(s) }
            return .string(s)
        case .date:
            let a = try fill(Int32.self, try checked(Int32.self))
            return .temporal(try MetalTemporalArray(type: .date32, a))
        case .timestamp(let ns, let utc):
            let a = try fill(Int64.self, try intValue())
            return .temporal(try MetalTemporalArray(type: .timestamp(ns ? .nano : .micro, timezone: utc ? "UTC" : nil), a))
        case .time:
            let a = try fill(Int64.self, try intValue())
            return .temporal(try MetalTemporalArray(type: .time64(.micro), a))
        case .decimal(let p, let s):
            var d = ArrowDecimal128(0)
            switch value {
            case .decimal(let x)?: d = x
            case .int(let i)?: d = ArrowDecimal128(i)
            case .none: break
            default: throw LakehouseError.malformed("value \(String(describing: value)) for \(type) column \(column)")
            }
            let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 16, 16), zeroed: true, context: ctx)
            if value != nil {
                let q = vb.mutableTyped(UInt64.self)
                for i in 0..<n { q[2 * i] = d.lo; q[2 * i + 1] = d.hi }
            }
            return .decimal(MetalDecimalArray(type: try ArrowDecimalType(precision: p, scale: s), length: n,
                                              nullCount: value == nil ? n : 0, validity: try validity(),
                                              values: vb, context: ctx))
        case .fixed, .uuid:
            let width: Int
            if case .fixed(let x) = type { width = x } else { width = 16 }
            var bytes = [UInt8](repeating: 0, count: width)
            if case .bytes(let b)? = value {
                guard b.count == width else { throw LakehouseError.malformed("value of \(b.count) bytes for \(type) column \(column)") }
                bytes = b
            } else if value != nil {
                throw LakehouseError.malformed("value \(String(describing: value)) for \(type) column \(column)")
            }
            let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * width, 1), zeroed: true, context: ctx)
            if value != nil {
                let p = vb.mutableTyped(UInt8.self)
                for i in 0..<n { for j in 0..<width { p[i * width + j] = bytes[j] } }
            }
            return .fixedBinary(MetalFixedBinaryArray(byteWidth: width, length: n, nullCount: value == nil ? n : 0,
                                                      validity: try validity(), values: vb, context: ctx))
        case .nested(let s):
            throw LakehouseError.unsupportedFeature("column \(column) has nested type \(s); nested columns are not read")
        }
    }

    /// Brings a column read from a data file to the table's type: dictionary columns are decoded, and a
    /// column stored in a narrower or differently-labelled type (an int32 file column in an int64 table
    /// column after a type promotion, a timestamp without its UTC label) is cast.
    static func conform(_ a: AnyMetalArray, to type: LakehouseType, column: String, file: String) throws -> AnyMetalArray {
        guard let want = type.arrowFormat else {
            throw LakehouseError.unsupportedFeature("column \(column) has nested type \(type); nested columns are not read")
        }
        var col = try a.decodedIfDictionary()
        if case .uuid = type, case .fixedBinary = col { return col }
        if col.arrowFormat == want { return col }
        // An Iceberg uuid is FIXED_LEN_BYTE_ARRAY(16); binary with a fixed width stays as it is.
        do {
            col = try col.cast(to: want)
        } catch {
            throw LakehouseError.malformed("column \(column) in \(file) is \(col.arrowFormat), which does not convert to \(type): \(error)")
        }
        return col
    }
}

// MARK: - Row filter

enum LakeRowFilter {
    /// The rows of `batch` where `column op literal` holds (nulls never match, as in Arrow and SQL).
    static func mask(_ col: AnyMetalArray, _ op: CompareOp, _ literal: LakeScalar, column: String) throws -> MetalBooleanArray {
        let c = try col.decodedIfDictionary()
        switch c {
        case .int8(let a): return try integerMask(a, op, literal)
        case .int16(let a): return try integerMask(a, op, literal)
        case .int32(let a): return try integerMask(a, op, literal)
        case .int64(let a): return try integerMask(a, op, literal)
        case .uint8(let a): return try integerMask(a, op, literal)
        case .uint16(let a): return try integerMask(a, op, literal)
        case .uint32(let a): return try integerMask(a, op, literal)
        case .uint64(let a): return try integerMask(a, op, literal)
        case .float32(let a):
            guard let d = double(literal) else { throw typeError(column, c, literal) }
            return try float32Mask(a, op, d)
        case .float64(let a):
            guard let d = double(literal) else { throw typeError(column, c, literal) }
            return try a.compare(op, d)
        case .temporal(let t):
            guard case .int(let v) = literal else { throw typeError(column, c, literal) }
            return try t.compare(op, v)
        case .decimal(let d):
            guard case .decimal(let v) = literal else { throw typeError(column, c, literal) }
            return try d.compare(op, v)
        case .string(let s), .binary(let s):
            let lit: [UInt8]
            switch literal {
            case .string(let x): lit = Array(x.utf8)
            case .bytes(let b): lit = b
            default: throw typeError(column, c, literal)
            }
            return try bytesMask(s, op, lit)
        case .boolean(let b):
            guard case .bool(let v) = literal else { throw typeError(column, c, literal) }
            try b.context.flush()
            let vals = b.toArray()
            return try hostMask(count: b.length, validity: b.validity, context: b.context) { i in
                guard let x = vals[i] else { return false }
                return op.holds(x == v ? 0 : (x ? 1 : -1))
            }
        default:
            throw LakehouseError.invalidArgument("cannot filter column \(column) of Arrow type \(c.arrowFormat)")
        }
    }

    private static func double(_ l: LakeScalar) -> Double? {
        switch l {
        case .int(let i): return Double(i)
        case .double(let d): return d
        default: return nil
        }
    }

    private static func typeError(_ column: String, _ c: AnyMetalArray, _ l: LakeScalar) -> LakehouseError {
        .invalidArgument("filter literal \(l) does not fit column \(column) (Arrow type \(c.arrowFormat))")
    }

    /// An integer column against an integer or double literal, on the GPU. A literal outside the column's
    /// range, or a non-integral double, is rewritten to the equivalent integer comparison or to a
    /// constant answer, so the result matches comparing the exact values.
    private static func integerMask<T: ArrowPrimitive & FixedWidthInteger>(_ a: MetalArray<T>, _ op: CompareOp,
                                                                          _ literal: LakeScalar) throws -> MetalBooleanArray {
        var op = op
        var k: Int64
        switch literal {
        case .int(let i): k = i
        case .double(let d):
            if d.isNaN { return try constantMask(a, op == .ne) }
            // Beyond the Int64 range every value is below (or above) the literal; inside it the
            // rounded literal converts exactly.
            if d >= 0x1p63 { return try constantMask(a, op == .lt || op == .le || op == .ne) }
            if d < -0x1p63 { return try constantMask(a, op == .gt || op == .ge || op == .ne) }
            if d == d.rounded() { k = Int64(d) }
            else {
                switch op {
                case .eq: return try constantMask(a, false)
                case .ne: return try constantMask(a, true)
                case .gt, .ge: op = .ge; k = Int64(d.rounded(.up))
                case .lt, .le: op = .le; k = Int64(d.rounded(.down))
                }
            }
        default:
            throw LakehouseError.invalidArgument("filter literal \(literal) does not fit an integer column")
        }
        // Clamp the literal into the column's range.
        let tooBig: Bool, tooSmall: Bool
        if T.isSigned {
            tooBig = k > Int64(T.max)
            tooSmall = k < Int64(T.min)
        } else {
            tooBig = k >= 0 && UInt64(k) > UInt64(T.max)
            tooSmall = k < 0
        }
        if tooBig { return try constantMask(a, op == .lt || op == .le || op == .ne) }
        if tooSmall { return try constantMask(a, op == .gt || op == .ge || op == .ne) }
        return try a.compare(op, T(k))
    }

    /// A float32 column against a double literal, compared as the exact values (as pyarrow does, by
    /// widening the column): the literal is rounded to its nearest float and, when that changes it, the
    /// comparison is rewritten so no float lies between the two.
    private static func float32Mask(_ a: MetalArray<Float>, _ op: CompareOp, _ d: Double) throws -> MetalBooleanArray {
        let f = Float(d)
        if d.isNaN || Double(f) == d { return try a.compare(op, f) }
        let above = Double(f) > d        // f is the nearest float above d (or +inf), else the one below
        switch op {
        case .eq: return try constantMask(a, false)
        case .ne: return try constantMask(a, true)
        case .lt, .le: return try a.compare(above ? .lt : .le, f)
        case .gt, .ge: return try a.compare(above ? .ge : .gt, f)
        }
    }

    /// Every valid row `answer`, every null row null.
    private static func constantMask<T: ArrowPrimitive>(_ a: MetalArray<T>, _ answer: Bool) throws -> MetalBooleanArray {
        try a.context.flush()
        return try hostMask(count: a.length, validity: a.validity, context: a.context) { _ in answer }
    }

    /// Byte-wise comparison of a string or binary column with a literal (CPU; the bytes are in unified
    /// memory, so nothing is copied).
    private static func bytesMask(_ s: MetalStringArray, _ op: CompareOp, _ lit: [UInt8]) throws -> MetalBooleanArray {
        try s.context.flush()
        let off = s.offsets.typed(Int32.self)
        let data = s.data.typed(UInt8.self)
        return try lit.withUnsafeBufferPointer { lp in
            try hostMask(count: s.length, validity: s.validity, context: s.context) { i in
                let lo = Int(off[i]), hi = Int(off[i + 1])
                let n = hi - lo, m = lp.count
                let common = Swift.min(n, m)
                let c = common == 0 ? 0 : memcmp(data + lo, lp.baseAddress!, common)
                let ordering = c != 0 ? (c < 0 ? -1 : 1) : (n == m ? 0 : (n < m ? -1 : 1))
                return op.holds(ordering)
            }
        }
    }

    /// A boolean array computed on the host; the validity bitmap is shared with the input column.
    static func hostMask(count n: Int, validity: MetalArrowBuffer?, context ctx: MetalContext,
                         _ predicate: (Int) -> Bool) throws -> MetalBooleanArray {
        let vb = try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: n), 1), zeroed: true, context: ctx)
        let p = vb.mutableTyped(UInt8.self)
        for i in 0..<n where predicate(i) { Bitmap.set(p, i) }
        var nulls = 0
        if let v = validity { nulls = n - Bitmap.popcount(v.typed(UInt8.self), bits: n) }
        return MetalBooleanArray(length: n, nullCount: nulls, validity: validity, values: vb, context: ctx)
    }

    /// Applies every filter (AND) to `batch`, whose column `i` is `specs[i]`.
    static func apply(_ batch: MetalRecordBatch, _ filters: [(column: Int, op: CompareOp, literal: LakeScalar)],
                      names: [String]) throws -> MetalRecordBatch {
        var b = batch
        for f in filters {
            if b.length == 0 { break }
            let m = try mask(b.columns[f.column], f.op, f.literal, column: names[f.column])
            b = try b.filter(m)
        }
        return b
    }
}

// MARK: - Reading one data file

/// Where one output column of a data-file read comes from.
enum LakeColumnSource {
    /// A top-level column of the Parquet file, by its name there.
    case parquet(String)
    /// The same value on every row (a partition value), or null (a column the file predates).
    case constant(LakeScalar?)
}

enum LakeDataFile {
    /// Reads one Parquet data file into a batch with exactly `fields` (names and types), then applies
    /// the row filters. `resolve` sees the opened file and says where each field comes from, which is
    /// how a column is found by field id (Iceberg) or physical name (Delta), and how a column missing
    /// from an older file becomes nulls. `rowGroupFilters` (named by the file's own column names) prune
    /// whole row groups by the footer statistics first; `rowFilters` index into `fields`.
    static func read(path: String, fields: [LakehouseField],
                     resolve: (ParquetFile) throws -> [LakeColumnSource],
                     rowGroupFilters: (ParquetFile) -> [(column: String, filter: LakeFilter)],
                     rowFilters: [(column: Int, op: CompareOp, literal: LakeScalar)],
                     context: MetalContext = .shared) throws -> MetalRecordBatch {
        let file: ParquetFile
        do { file = try ParquetFile(path: path, context: context) }
        catch { throw LakehouseError.malformed("data file \(path): \(error)") }
        let sources = try resolve(file)
        var wanted: [String] = []
        for s in sources { if case .parquet(let n) = s, !wanted.contains(n) { wanted.append(n) } }
        let (rgFilters, rowGroups) = rowGroupPlan(file, rowGroupFilters(file))
        let opts = ParquetReadOptions(columns: wanted, rowGroups: rowGroups, dictionaryEncoded: false, filters: rgFilters)
        let n: Int
        var read: MetalRecordBatch? = nil
        if wanted.isEmpty {
            n = try file.rowCount(opts)
        } else {
            let b = try file.read(opts)
            n = b.length
            read = b
        }
        var cols: [AnyMetalArray] = []
        cols.reserveCapacity(fields.count)
        for (f, s) in zip(fields, sources) {
            switch s {
            case .parquet(let name):
                guard let b = read, let i = b.names.firstIndex(of: name) else {
                    throw LakehouseError.malformed("data file \(path) has no column \(name)")
                }
                cols.append(try LakeColumns.conform(b.columns[i], to: f.type, column: f.name, file: path))
            case .constant(let v):
                cols.append(try LakeColumns.constant(f.type, v, length: n, context: context, column: f.name))
            }
        }
        let batch = try MetalRecordBatch(names: fields.map { $0.name }, columns: cols)
        return try LakeRowFilter.apply(batch, rowFilters, names: fields.map { $0.name })
    }

    /// How the filters prune `file`'s row groups by the footer statistics. Only comparisons the
    /// statistics decide exactly as the row filter does are used, so pruning never drops a matching row:
    ///
    /// - string and binary filters are decided here, byte-wise on `min_value` / `max_value` (the row
    ///   filter's order, and Parquet's), since a binary literal may be bytes that are not UTF-8, which a
    ///   `ParquetFilter` string cannot carry;
    /// - an integer column's double literal of any magnitude: the reader compares an integer statistic
    ///   with a double exactly, never through a rounded double;
    /// - a timestamp only when the file stores the table's unit;
    /// - floats never under `!=` (a NaN matches it and statistics do not count NaNs).
    ///
    /// Filters on columns the file does not have at the top level are skipped.
    static func rowGroupPlan(_ file: ParquetFile, _ filters: [(column: String, filter: LakeFilter)])
        -> (parquet: [ParquetFilter], rowGroups: [Int]?) {
        var parquet: [ParquetFilter] = []
        var byteFilters: [(leaf: ParquetLeaf, op: CompareOp, literal: [UInt8])] = []
        for (name, f) in filters {
            // The Parquet filter takes the first leaf whose dotted path *or* last name matches; use it only
            // when that leaf is the top-level column itself.
            guard file.fields.contains(where: { $0.name == name }),
                  let leaf = file.leaves.first(where: { $0.dottedPath == name || $0.name == name }),
                  leaf.dottedPath == name else { continue }
            let value: ParquetFilter.Value?
            switch (f.field.type, f.literal) {
            case (.string, .string(let x)), (.binary, .string(let x)):
                if leaf.physical == .byteArray { byteFilters.append((leaf, f.op, Array(x.utf8))) }
                continue
            case (.string, .bytes(let b)), (.binary, .bytes(let b)):
                if leaf.physical == .byteArray { byteFilters.append((leaf, f.op, b)) }
                continue
            case (.int8, .int(let i)), (.int16, .int(let i)), (.int32, .int(let i)), (.int64, .int(let i)):
                if case .integer(_, false) = leaf.logicalType { value = nil } else { value = .int(i) }
            case (.int8, .double(let d)), (.int16, .double(let d)), (.int32, .double(let d)), (.int64, .double(let d)):
                if case .integer(_, false) = leaf.logicalType { value = nil }
                else { value = .double(d) }
            case (.date, .int(let i)):
                value = leaf.logicalType == .date ? .int(i) : nil
            case (.timestamp(let ns, _), .int(let i)):
                if case .timestamp(_, let unit) = leaf.logicalType, leaf.physical == .int64, unit == (ns ? .nanos : .micros) {
                    value = .int(i)
                } else { value = nil }
            case (.float32, .double(let d)), (.float64, .double(let d)):
                value = f.op == .ne ? nil : .double(d)
            case (.boolean, .bool(let b)):
                value = leaf.physical == .boolean ? .int(b ? 1 : 0) : nil
            default:
                value = nil
            }
            if let value { parquet.append(ParquetFilter(column: name, op: ParquetFilter.Op(f.op), value: value)) }
        }
        guard !byteFilters.isEmpty else { return (parquet, nil) }
        let groups = file.metadata.rowGroups.indices.filter { g in
            let rg = file.metadata.rowGroups[g]
            return byteFilters.allSatisfy { bf in
                guard bf.leaf.index < rg.columns.count, let st = rg.columns[bf.leaf.index].meta.statistics,
                      let lo = st.minValue, let hi = st.maxValue else { return true }
                return lakeRangeMayMatch(bf.op, lo: .bytes(lo), hi: .bytes(hi), literal: .bytes(bf.literal))
            }
        }
        return (parquet, Array(groups))
    }

    /// Concatenates one column's per-file parts. Decimal and fixed-width binary columns (which the
    /// engine's `concatMetalArrays` does not take) are appended here the same way: a copy of the
    /// fixed-width values and of the validity bits, in unified memory.
    static func concat(_ parts: [AnyMetalArray]) throws -> AnyMetalArray {
        guard let head = parts.first else { throw LakehouseError.malformed("nothing to concatenate") }
        if parts.count == 1 { return head }
        func fixedWidth(width: Int, lengths: [Int], nulls: [Int], values: [MetalArrowBuffer],
                        validity: [MetalArrowBuffer?], ctx: MetalContext) throws -> (MetalArrowBuffer, MetalArrowBuffer?, Int, Int) {
            try ctx.flush()
            let total = lengths.reduce(0, +)
            let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(total * width, 1), zeroed: false, context: ctx)
            let anyNulls = validity.contains { $0 != nil }
            let bm = anyNulls ? try MetalArrowBuffer.allocate(byteCount: Swift.max(Bitmap.byteCount(bits: total), 1), context: ctx) : nil
            var row = 0
            for i in parts.indices {
                if lengths[i] > 0 { memcpy(out.mutableContents.advanced(by: row * width), values[i].contents, lengths[i] * width) }
                if let bm {
                    copyValidityBits(dst: bm.mutableTyped(UInt8.self), dstOffset: row,
                                     src: validity[i].map { $0.typed(UInt8.self) }, count: lengths[i])
                }
                row += lengths[i]
            }
            return (out, bm, total, nulls.reduce(0, +))
        }
        switch head {
        case .decimal(let h):
            let ds: [MetalDecimalArray] = try parts.map {
                guard case .decimal(let d) = $0, d.type == h.type else {
                    throw LakehouseError.malformed("cannot concatenate \(head.arrowFormat) with \($0.arrowFormat)")
                }
                return d
            }
            let (v, bm, n, nulls) = try fixedWidth(width: h.type.byteWidth, lengths: ds.map { $0.length },
                                                   nulls: ds.map { $0.nullCount }, values: ds.map { $0.values },
                                                   validity: ds.map { $0.validity }, ctx: h.context)
            return .decimal(MetalDecimalArray(type: h.type, length: n, nullCount: nulls, validity: bm, values: v, context: h.context))
        case .fixedBinary(let h):
            let fs: [MetalFixedBinaryArray] = try parts.map {
                guard case .fixedBinary(let f) = $0, f.byteWidth == h.byteWidth else {
                    throw LakehouseError.malformed("cannot concatenate \(head.arrowFormat) with \($0.arrowFormat)")
                }
                return f
            }
            let (v, bm, n, nulls) = try fixedWidth(width: h.byteWidth, lengths: fs.map { $0.length },
                                                   nulls: fs.map { $0.nullCount }, values: fs.map { $0.values },
                                                   validity: fs.map { $0.validity }, ctx: h.context)
            return .fixedBinary(MetalFixedBinaryArray(byteWidth: h.byteWidth, length: n, nullCount: nulls,
                                                      validity: bm, values: v, context: h.context))
        default:
            return try concatMetalArrays(parts)
        }
    }

    /// Runs `body` for every data file, several files at a time, and returns the batches in file order.
    ///
    /// Each file's decode is a chain of small kernels with a few CPU round trips in between (page
    /// totals, offsets), so a table of many files spends most of a serial read waiting. Command buffers
    /// are per thread in `MetalContext`, so files read on different threads overlap those waits with
    /// each other's GPU work. The first error, in file order, is rethrown.
    static func readAll(count: Int, body: @escaping (Int) throws -> MetalRecordBatch) throws -> [MetalRecordBatch] {
        if count == 0 { return [] }
        let width = Swift.min(count, Swift.max(1, Swift.min(ProcessInfo.processInfo.activeProcessorCount, 8)))
        if width == 1 { return try (0..<count).map(body) }
        var results = [Result<MetalRecordBatch, Error>?](repeating: nil, count: count)
        let lock = NSLock()
        var next = 0
        DispatchQueue.concurrentPerform(iterations: width) { _ in
            while true {
                lock.lock()
                let i = next
                next += 1
                lock.unlock()
                if i >= count { return }
                let r = Result { try body(i) }
                lock.lock()
                results[i] = r
                lock.unlock()
            }
        }
        return try results.map { try $0!.get() }
    }

    /// Empty columns with the table's types, for a scan that selects no files.
    static func empty(_ fields: [LakehouseField], context: MetalContext = .shared) throws -> MetalRecordBatch {
        let cols = try fields.map { try LakeColumns.constant($0.type, nil, length: 0, context: context, column: $0.name) }
        return try MetalRecordBatch(names: fields.map { $0.name }, columns: cols)
    }

    /// Concatenates per-file batches (or builds the empty batch) and keeps the first `keep` columns.
    static func finish(_ parts: [MetalRecordBatch], fields: [LakehouseField], keep: Int,
                       context: MetalContext = .shared) throws -> MetalRecordBatch {
        let nonEmpty = parts.filter { $0.length > 0 }
        let all: MetalRecordBatch
        if nonEmpty.isEmpty { all = try empty(fields, context: context) }
        else if nonEmpty.count == 1 { all = nonEmpty[0] }
        else {
            var cols: [AnyMetalArray] = []
            for i in 0..<fields.count { cols.append(try concat(nonEmpty.map { $0.columns[i] })) }
            all = try MetalRecordBatch(names: fields.map { $0.name }, columns: cols)
        }
        if keep == all.columnCount { return all }
        return try MetalRecordBatch(names: Array(all.names.prefix(keep)), columns: Array(all.columns.prefix(keep)))
    }
}
