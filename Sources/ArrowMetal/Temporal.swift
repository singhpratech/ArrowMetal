import Foundation
import Metal
import CArrowABI

// Arrow temporal types (date, time, timestamp, duration). Values are plain integers, so every kernel that
// already exists for Int32 / Int64 works unchanged; this file adds the type metadata, the C Data Interface
// format strings and the calendar-field extraction kernels (UTC only).

/// Resolution of a temporal value.
public enum ArrowTemporalUnit: String, Sendable, CaseIterable {
    case second = "s", milli = "m", micro = "u", nano = "n"

    /// Ticks of this unit in one second.
    public var perSecond: Int64 {
        switch self {
        case .second: return 1
        case .milli: return 1_000
        case .micro: return 1_000_000
        case .nano: return 1_000_000_000
        }
    }
    /// Arrow's spelling in `pa.timestamp(...)` and friends.
    public var arrowName: String {
        switch self {
        case .second: return "s"
        case .milli: return "ms"
        case .micro: return "us"
        case .nano: return "ns"
        }
    }
}

/// An Arrow temporal type together with its C Data Interface format string.
///
/// | type | format | storage |
/// |---|---|---|
/// | date32 (days) | `tdD` | int32 |
/// | date64 (ms) | `tdm` | int64 |
/// | time32 (s, ms) | `tts`, `ttm` | int32 |
/// | time64 (us, ns) | `ttu`, `ttn` | int64 |
/// | timestamp (s, ms, us, ns) | `tss:`, `tsm:`, `tsu:`, `tsn:` (+ timezone) | int64 |
/// | duration (s, ms, us, ns) | `tDs`, `tDm`, `tDu`, `tDn` | int64 |
public enum ArrowTemporalType: Hashable, Sendable {
    case date32
    case date64
    case time32(ArrowTemporalUnit)
    case time64(ArrowTemporalUnit)
    /// Timezone is the string after the colon in the format ("UTC", "+02:00", ...). Values are always UTC
    /// ticks since the epoch, exactly as Arrow specifies; the timezone is display metadata.
    case timestamp(ArrowTemporalUnit, timezone: String?)
    case duration(ArrowTemporalUnit)

    /// Parses an Arrow format string, returning nil when it is not a (supported) temporal type.
    public init?(format f: String) {
        switch f {
        case "tdD": self = .date32; return
        case "tdm": self = .date64; return
        case "tts": self = .time32(.second); return
        case "ttm": self = .time32(.milli); return
        case "ttu": self = .time64(.micro); return
        case "ttn": self = .time64(.nano); return
        case "tDs": self = .duration(.second); return
        case "tDm": self = .duration(.milli); return
        case "tDu": self = .duration(.micro); return
        case "tDn": self = .duration(.nano); return
        default: break
        }
        let c = Array(f)
        guard c.count >= 4, c[0] == "t", c[1] == "s", c[3] == ":",
              let u = ArrowTemporalUnit(rawValue: String(c[2])) else { return nil }
        let tz = String(c[4...])
        self = .timestamp(u, timezone: tz.isEmpty ? nil : tz)
    }

    /// Parses a format string or throws `unsupportedType`.
    public static func parse(_ f: String) throws -> ArrowTemporalType {
        guard let t = ArrowTemporalType(format: f), t.isValid else { throw ArrowMetalError.unsupportedType(f) }
        return t
    }

    public var arrowFormat: String {
        switch self {
        case .date32: return "tdD"
        case .date64: return "tdm"
        case .time32(let u): return "tt" + u.rawValue
        case .time64(let u): return "tt" + u.rawValue
        case .timestamp(let u, let tz): return "ts" + u.rawValue + ":" + (tz ?? "")
        case .duration(let u): return "tD" + u.rawValue
        }
    }

    /// time32 is only defined for seconds and milliseconds, time64 only for microseconds and nanoseconds.
    public var isValid: Bool {
        switch self {
        case .time32(let u): return u == .second || u == .milli
        case .time64(let u): return u == .micro || u == .nano
        default: return true
        }
    }

    /// Whether values are stored as int64 (int32 otherwise).
    public var usesInt64: Bool {
        switch self {
        case .date32, .time32: return false
        case .date64, .time64, .timestamp, .duration: return true
        }
    }

    public var unit: ArrowTemporalUnit {
        switch self {
        case .date32: return .second            // days; unit is not meaningful
        case .date64: return .milli
        case .time32(let u), .time64(let u), .duration(let u): return u
        case .timestamp(let u, _): return u
        }
    }

    public var timezone: String? {
        if case .timestamp(_, let tz) = self { return tz }
        return nil
    }

    /// How the extraction kernel must read the value: mode 0 = whole days, 1 = ticks since the epoch,
    /// 2 = ticks since midnight. Nil for types with no calendar meaning (duration).
    var extraction: (mode: Int, divisor: Int64)? {
        switch self {
        case .date32: return (0, 1)
        case .date64: return (1, 1_000)
        case .timestamp(let u, _): return (1, u.perSecond)
        case .time32(let u), .time64(let u): return (2, u.perSecond)
        case .duration: return nil
        }
    }
}

/// Calendar field extracted by `MetalTemporalArray.extract`.
public enum TemporalField: Int, Sendable, CaseIterable {
    case year = 0, month, day, dayOfWeek, hour, minute, second
    /// True for fields that need a date (not available on time-of-day types).
    var needsDate: Bool { rawValue <= 3 }
}

/// Internal field id for "days since the epoch", used by `toDate32`.
private let epochDaysField = 7

/// An Arrow temporal array: an integer array plus its temporal type.
///
/// Every kernel is the integer kernel underneath, so compare / filter / take / slice / min / max /
/// sort / argsort are forwarded to `MetalArray<Int32>` or `MetalArray<Int64>` without a copy.
public final class MetalTemporalArray: @unchecked Sendable {
    /// The integer array holding the values, either 32-bit (date32, time32) or 64-bit (everything else).
    public enum Storage {
        case int32(MetalArray<Int32>)
        case int64(MetalArray<Int64>)
    }

    public let type: ArrowTemporalType
    public let storage: Storage

    public init(type: ArrowTemporalType, _ values: MetalArray<Int32>) throws {
        guard type.isValid else { throw ArrowMetalError.unsupportedType(type.arrowFormat) }
        guard !type.usesInt64 else { throw ArrowMetalError.unsupportedType("\(type.arrowFormat) is stored as int64, not int32") }
        self.type = type
        self.storage = .int32(values)
    }

    public init(type: ArrowTemporalType, _ values: MetalArray<Int64>) throws {
        guard type.isValid else { throw ArrowMetalError.unsupportedType(type.arrowFormat) }
        guard type.usesInt64 else { throw ArrowMetalError.unsupportedType("\(type.arrowFormat) is stored as int32, not int64") }
        self.type = type
        self.storage = .int64(values)
    }

    /// Builds from Swift values (int32 storage narrows, throwing on overflow).
    public convenience init(type: ArrowTemporalType, _ values: [Int64?], context: MetalContext = .shared) throws {
        if type.usesInt64 {
            try self.init(type: type, try MetalArray<Int64>(values, context: context))
        } else {
            var narrow: [Int32?] = []
            narrow.reserveCapacity(values.count)
            for v in values {
                guard let v else { narrow.append(nil); continue }
                guard let n = Int32(exactly: v) else { throw ArrowMetalError.invalidArrowArray("\(v) does not fit \(type.arrowFormat)") }
                narrow.append(n)
            }
            try self.init(type: type, try MetalArray<Int32>(narrow, context: context))
        }
    }

    public var arrowFormat: String { type.arrowFormat }
    public var length: Int { switch storage { case .int32(let a): return a.length; case .int64(let a): return a.length } }
    public var nullCount: Int { switch storage { case .int32(let a): return a.nullCount; case .int64(let a): return a.nullCount } }
    public var context: MetalContext { switch storage { case .int32(let a): return a.context; case .int64(let a): return a.context } }
    public var validity: MetalArrowBuffer? { switch storage { case .int32(let a): return a.validity; case .int64(let a): return a.validity } }
    public var values: MetalArrowBuffer { switch storage { case .int32(let a): return a.values; case .int64(let a): return a.values } }

    public var asInt32: MetalArray<Int32>? { if case .int32(let a) = storage { return a } else { return nil } }
    public var asInt64: MetalArray<Int64>? { if case .int64(let a) = storage { return a } else { return nil } }

    /// The values widened to int64 (no copy when they are already int64).
    public func int64Values() throws -> MetalArray<Int64> {
        switch storage {
        case .int64(let a): return a
        case .int32(let a): return try a.cast(to: Int64.self)
        }
    }

    public func isValid(_ i: Int) -> Bool {
        switch storage { case .int32(let a): return a.isValid(i); case .int64(let a): return a.isValid(i) }
    }
    public subscript(i: Int) -> Int64? {
        switch storage { case .int32(let a): return a[i].map(Int64.init); case .int64(let a): return a[i] }
    }
    public func toArray() -> [Int64?] { (0..<length).map { self[$0] } }

    // MARK: - Forwarded integer kernels

    public func compare(_ op: CompareOp, _ scalar: Int64) throws -> MetalBooleanArray {
        switch storage {
        case .int64(let a): return try a.compare(op, scalar)
        case .int32(let a):
            guard let s = Int32(exactly: scalar) else {
                throw ArrowMetalError.invalidArrowArray("scalar \(scalar) is out of range for \(type.arrowFormat)")
            }
            return try a.compare(op, s)
        }
    }

    public func compare(_ op: CompareOp, _ other: MetalTemporalArray) throws -> MetalBooleanArray {
        switch (storage, other.storage) {
        case (.int32(let a), .int32(let b)): return try a.compare(op, b)
        case (.int64(let a), .int64(let b)): return try a.compare(op, b)
        default: throw ArrowMetalError.unsupportedType("cannot compare \(type.arrowFormat) with \(other.type.arrowFormat)")
        }
    }

    public func filter(_ mask: MetalBooleanArray) throws -> MetalTemporalArray {
        switch storage {
        case .int32(let a): return try MetalTemporalArray(type: type, try a.filter(mask))
        case .int64(let a): return try MetalTemporalArray(type: type, try a.filter(mask))
        }
    }

    public func filter(where op: CompareOp, _ scalar: Int64) throws -> MetalTemporalArray {
        try filter(try compare(op, scalar))
    }

    public func take<I: ArrowIndex>(_ indices: MetalArray<I>) throws -> MetalTemporalArray {
        switch storage {
        case .int32(let a): return try MetalTemporalArray(type: type, try a.take(indices))
        case .int64(let a): return try MetalTemporalArray(type: type, try a.take(indices))
        }
    }

    public func slice(offset: Int, length: Int) throws -> MetalTemporalArray {
        switch storage {
        case .int32(let a): return try MetalTemporalArray(type: type, try a.slice(offset: offset, length: length))
        case .int64(let a): return try MetalTemporalArray(type: type, try a.slice(offset: offset, length: length))
        }
    }

    /// Smallest value, or nil when every element is null.
    public func min() throws -> Int64? {
        switch storage { case .int32(let a): return try a.min().map(Int64.init); case .int64(let a): return try a.min() }
    }
    /// Largest value, or nil when every element is null.
    public func max() throws -> Int64? {
        switch storage { case .int32(let a): return try a.max().map(Int64.init); case .int64(let a): return try a.max() }
    }

    /// Indices that sort the values ascending (stable, nulls last).
    public func argsort(descending: Bool = false) throws -> MetalArray<Int32> {
        switch storage {
        case .int32(let a): return try a.argsort(descending: descending)
        case .int64(let a): return try a.argsort(descending: descending)
        }
    }
    /// Sorted copy (nulls last).
    public func sorted(descending: Bool = false) throws -> MetalTemporalArray {
        try take(try argsort(descending: descending))
    }

    // MARK: - Calendar fields (UTC)

    public func year() throws -> MetalArray<Int32> { try extract(.year) }
    public func month() throws -> MetalArray<Int32> { try extract(.month) }
    public func day() throws -> MetalArray<Int32> { try extract(.day) }
    /// Day of the week, Monday = 0 ... Sunday = 6 (Arrow's default for `day_of_week`).
    public func dayOfWeek() throws -> MetalArray<Int32> { try extract(.dayOfWeek) }
    public func hour() throws -> MetalArray<Int32> { try extract(.hour) }
    public func minute() throws -> MetalArray<Int32> { try extract(.minute) }
    public func second() throws -> MetalArray<Int32> { try extract(.second) }

    /// Extracts one calendar field on the GPU. Nulls stay null (the validity bitmap is shared).
    public func extract(_ field: TemporalField) throws -> MetalArray<Int32> {
        try extract(fieldID: field.rawValue, needsDate: field.needsDate)
    }

    /// Converts to `date32` (days since the epoch, UTC).
    public func toDate32() throws -> MetalTemporalArray {
        if case .date32 = type { return self }
        return try MetalTemporalArray(type: .date32, try extract(fieldID: epochDaysField, needsDate: true))
    }

    private func extract(fieldID: Int, needsDate: Bool) throws -> MetalArray<Int32> {
        guard let (mode, divisor) = type.extraction else {
            throw ArrowMetalError.unsupportedType("calendar fields are not defined for \(type.arrowFormat)")
        }
        if needsDate && mode == 2 {
            throw ArrowMetalError.unsupportedType("date fields are not defined for \(type.arrowFormat)")
        }
        let ctx = context
        let n = length
        try Dispatch.checkLength(n)
        let mslT = type.usesInt64 ? "long" : "int"
        let pso = try ctx.pipeline(source: TemporalSource.source(T: mslT), function: "temporal_extract",
                                   cacheKey: "temporal/\(mslT)/extract")
        let out = try MetalArrowBuffer.allocate(byteCount: Swift.max(n, 1) * 4, zeroed: false, context: ctx)
        let vals = values
        if n > 0 {
            try ctx.run { enc in
                enc.setComputePipelineState(pso)
                enc.setBuffer(vals.mtl, offset: vals.offset, index: 0)
                Dispatch.setLength(enc, n, nil, index: 1)
                Dispatch.setUInt(enc, fieldID, index: 2)
                Dispatch.setUInt(enc, mode, index: 3)
                var d = divisor
                enc.setBytes(&d, length: 8, index: 4)
                enc.setBuffer(out.mtl, offset: out.offset, index: 5)
                Dispatch.dispatch1D(enc, pso, count: n)
            }
        }
        return MetalArray<Int32>(length: n, nullCount: nullCount, validity: validity, values: out, context: ctx)
    }

    // MARK: - Unit conversion

    /// Rescales to another resolution. Timestamps stay timestamps (timezone preserved), durations stay
    /// durations, and time32/time64 pick the storage width the target unit requires.
    /// Narrowing truncates toward zero, matching Arrow's unchecked cast.
    public func castUnit(to unit: ArrowTemporalUnit) throws -> MetalTemporalArray {
        let from: ArrowTemporalUnit
        let target: ArrowTemporalType
        switch type {
        case .timestamp(let u, let tz): from = u; target = .timestamp(unit, timezone: tz)
        case .duration(let u): from = u; target = .duration(unit)
        case .time32(let u), .time64(let u):
            from = u
            target = (unit == .second || unit == .milli) ? .time32(unit) : .time64(unit)
        case .date32, .date64:
            throw ArrowMetalError.unsupportedType("castUnit is not defined for \(type.arrowFormat)")
        }
        if from == unit && target.usesInt64 == type.usesInt64 { return self }
        let wide = try int64Values()
        let a = from.perSecond, b = unit.perSecond
        let scaled: MetalArray<Int64> = a == b ? wide
            : (b > a ? try wide.arithmetic(.mul, b / a) : try wide.arithmetic(.div, a / b))
        if target.usesInt64 { return try MetalTemporalArray(type: target, scaled) }
        return try MetalTemporalArray(type: target, try scaled.cast(to: Int32.self))
    }
}

// MARK: - Metal shading language

/// Calendar arithmetic on the GPU: `civil_from_days` is Howard Hinnant's algorithm from
/// "chrono-Compatible Low-Level Date Algorithms" (public domain), valid for the whole int64 day range
/// in the proleptic Gregorian calendar. UTC only: no timezone or leap-second handling.
enum TemporalSource {
    static func source(T: String) -> String { KernelSource.prelude + """
    inline long t_floordiv(long a, long b) {
        long q = a / b;
        if ((a % b != 0L) && ((a < 0L) != (b < 0L))) q -= 1L;
        return q;
    }
    // days is the count of days since 1970-01-01 (negative before it).
    inline void civil_from_days(long z, thread long& y, thread long& m, thread long& d) {
        z += 719468L;
        long era = (z >= 0L ? z : z - 146096L) / 146097L;
        long doe = z - era * 146097L;                                          // [0, 146096]
        long yoe = (doe - doe / 1460L + doe / 36524L - doe / 146096L) / 365L;   // [0, 399]
        long yy = yoe + era * 400L;
        long doy = doe - (365L * yoe + yoe / 4L - yoe / 100L);                  // [0, 365]
        long mp = (5L * doy + 2L) / 153L;                                       // [0, 11]
        d = doy - (153L * mp + 2L) / 5L + 1L;                                   // [1, 31]
        m = mp + (mp < 10L ? 3L : -9L);                                         // [1, 12]
        y = yy + (m <= 2L ? 1L : 0L);
    }
    // field: 0 year, 1 month, 2 day, 3 day-of-week (Monday = 0), 4 hour, 5 minute, 6 second, 7 epoch days.
    // mode:  0 value is whole days, 1 value is ticks since the epoch, 2 value is ticks since midnight.
    kernel void temporal_extract(device const \(T)* vals [[buffer(0)]],
                                 device const uint* nPtr [[buffer(1)]],
                                 constant uint& field [[buffer(2)]],
                                 constant uint& mode [[buffer(3)]],
                                 constant long& divisor [[buffer(4)]],
                                 device int* out [[buffer(5)]],
                                 uint i [[thread_position_in_grid]]) {
        if (i >= *nPtr) return;
        long v = (long)vals[i];
        long days = 0L, sod = 0L;
        if (mode == 0u) {
            days = v;
        } else {
            long s = t_floordiv(v, divisor);
            if (mode == 1u) { days = t_floordiv(s, 86400L); sod = s - days * 86400L; }
            else { sod = s % 86400L; if (sod < 0L) sod += 86400L; }
        }
        int r = 0;
        switch (field) {
            case 0u: { long y, m, d; civil_from_days(days, y, m, d); r = (int)y; break; }
            case 1u: { long y, m, d; civil_from_days(days, y, m, d); r = (int)m; break; }
            case 2u: { long y, m, d; civil_from_days(days, y, m, d); r = (int)d; break; }
            case 3u: { r = (int)(((days + 3L) % 7L + 7L) % 7L); break; }
            case 4u: { r = (int)(sod / 3600L); break; }
            case 5u: { r = (int)((sod / 60L) % 60L); break; }
            case 6u: { r = (int)(sod % 60L); break; }
            default: { r = (int)days; break; }
        }
        out[i] = r;
    }
    """ }
}

// MARK: - C Data Interface

extension MetalTemporalArray {
    /// Exports through the CPU C Data Interface (zero-copy: the values are the integer buffers).
    public func exportArrowArray(into out: UnsafeMutablePointer<ArrowArray>) {
        switch storage {
        case .int32(let a): a.exportArrowArray(into: out)
        case .int64(let a): a.exportArrowArray(into: out)
        }
    }
    public func exportArrowSchema(name: String = "", into out: UnsafeMutablePointer<ArrowSchema>) {
        ArrowMetal.exportArrowSchema(format: type.arrowFormat, name: name, into: out)
    }
    public func exportArrowDeviceArray(into out: UnsafeMutablePointer<ArrowDeviceArray>) {
        switch storage {
        case .int32(let a): a.exportArrowDeviceArray(into: out)
        case .int64(let a): a.exportArrowDeviceArray(into: out)
        }
    }
}

/// Imports a temporal array: the values go through the primitive path, the type comes from the format string.
func importTemporalArray(type: ArrowTemporalType, array: UnsafeMutablePointer<ArrowArray>,
                         context: MetalContext) throws -> ImportResult {
    let fmt = strdup(type.usesInt64 ? "l" : "i")!
    defer { free(fmt) }
    let schema = UnsafeMutablePointer<ArrowSchema>.allocate(capacity: 1)
    schema.initialize(to: ArrowSchema())
    defer { schema.deallocate() }
    schema.pointee.format = UnsafePointer(fmt)
    let r = try importArrowArray(schema: schema, array: array, context: context)
    switch r.array {
    case .int32(let a): return ImportResult(array: .temporal(try MetalTemporalArray(type: type, a)), zeroCopy: r.zeroCopy)
    case .int64(let a): return ImportResult(array: .temporal(try MetalTemporalArray(type: type, a)), zeroCopy: r.zeroCopy)
    default: throw ArrowMetalError.invalidArrowArray("temporal values must be int32 or int64")
    }
}
