import Foundation

/// A CSV reader failure. The messages follow `pyarrow.csv.read_csv`'s own wording, so an error names
/// the same row, column and value pyarrow's would.
public enum CSVError: Error, CustomStringConvertible {
    /// The file could not be opened, mapped or is too large.
    case io(String)
    /// The bytes do not form a table: an empty file, a row of the wrong width, rows that cannot be
    /// skipped.
    case parse(String)
    /// A value does not convert to its column's type (a `columnTypes` override, or a null column).
    case conversion(String)
    /// `includeColumns` names a column the file does not have.
    case missingColumn(String)
    /// The options contradict each other.
    case invalidOptions(String)

    public var description: String {
        switch self {
        case .io(let s): return "CSV I/O error: \(s)"
        case .parse(let s), .conversion(let s), .missingColumn(let s), .invalidOptions(let s): return s
        }
    }
}

/// The type a CSV column is converted to, for `CSVReadOptions.columnTypes`.
public enum CSVColumnType: Hashable, Sendable {
    case null
    case bool
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
    case float32, float64
    case utf8
    case binary
    case date32
    case time32(ArrowTemporalUnit)
    case time64(ArrowTemporalUnit)
    case timestamp(ArrowTemporalUnit, timezone: String?)

    /// Parses an Arrow C Data Interface format string ("l", "g", "u", "tdD", "tss:UTC", ...).
    public init?(format f: String) {
        switch f {
        case "n": self = .null
        case "b": self = .bool
        case "c": self = .int8
        case "s": self = .int16
        case "i": self = .int32
        case "l": self = .int64
        case "C": self = .uint8
        case "S": self = .uint16
        case "I": self = .uint32
        case "L": self = .uint64
        case "f": self = .float32
        case "g": self = .float64
        case "u": self = .utf8
        case "z": self = .binary
        default:
            guard let t = ArrowTemporalType(format: f), t.isValid else { return nil }
            switch t {
            case .date32: self = .date32
            case .time32(let u): self = .time32(u)
            case .time64(let u): self = .time64(u)
            case .timestamp(let u, let tz): self = .timestamp(u, timezone: tz)
            default: return nil
            }
        }
    }

    /// The Arrow C Data Interface format string.
    public var arrowFormat: String {
        switch self {
        case .null: return "n"
        case .bool: return "b"
        case .int8: return "c"
        case .int16: return "s"
        case .int32: return "i"
        case .int64: return "l"
        case .uint8: return "C"
        case .uint16: return "S"
        case .uint32: return "I"
        case .uint64: return "L"
        case .float32: return "f"
        case .float64: return "g"
        case .utf8: return "u"
        case .binary: return "z"
        case .date32: return "tdD"
        case .time32(let u), .time64(let u): return "tt" + u.rawValue
        case .timestamp(let u, let tz): return "ts" + u.rawValue + ":" + (tz ?? "")
        }
    }

    /// pyarrow's `str(type)`, which its conversion errors quote.
    public var arrowName: String {
        switch self {
        case .null: return "null"
        case .bool: return "bool"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .float32: return "float"
        case .float64: return "double"
        case .utf8: return "string"
        case .binary: return "binary"
        case .date32: return "date32[day]"
        case .time32(let u): return "time32[\(u.arrowName)]"
        case .time64(let u): return "time64[\(u.arrowName)]"
        case .timestamp(let u, let tz):
            if let tz, !tz.isEmpty { return "timestamp[\(u.arrowName), tz=\(tz)]" }
            return "timestamp[\(u.arrowName)]"
        }
    }
}

/// Options for `CSVReader`, named after `pyarrow.csv`'s `ReadOptions`, `ParseOptions` and
/// `ConvertOptions` and defaulting to the same values.
///
/// Parsing always follows RFC 4180 with quoted newlines allowed, which is pyarrow's behaviour with
/// `newlines_in_values=True`; empty lines are always skipped (`ignore_empty_lines=True`). `escape_char`,
/// `timestamp_parsers`, `auto_dict_encode` and `invalid_row_handler` have no equivalent yet (see
/// docs/CSV.md).
public struct CSVReadOptions: Sendable {
    /// pyarrow's default `null_values`.
    public static let defaultNullValues = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
                                           "1.#IND", "1.#QNAN", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
    public static let defaultTrueValues = ["1", "True", "TRUE", "true"]
    public static let defaultFalseValues = ["0", "False", "FALSE", "false"]
    /// Bytes each GPU thread scans in the structure pass.
    public static let defaultScanBlockBytes = 1024

    // ReadOptions
    /// Lines skipped before the header, counted as pyarrow counts them: by line terminator, quotes
    /// ignored, empty lines included.
    public var skipRows = 0
    /// Records skipped after the header (these are parsed records, quotes respected).
    public var skipRowsAfterNames = 0
    /// Column names; when set, the first record is data.
    public var columnNames: [String]? = nil
    /// Name the columns `f0`, `f1`, ... and treat the first record as data.
    public var autogenerateColumnNames = false

    // ParseOptions
    public var delimiter = UInt8(ascii: ",")
    /// The quote character, or nil for no quoting at all.
    public var quoteChar: UInt8? = UInt8(ascii: "\"")
    /// Inside a quoted field, `""` is one quote.
    public var doubleQuote = true

    // ConvertOptions
    /// Columns to return, in this order. Nil or empty returns every column. A column that is not
    /// listed is never converted.
    public var includeColumns: [String]? = nil
    /// A listed column the file lacks comes back as all nulls instead of raising.
    public var includeMissingColumns = false
    /// Type overrides by column name; every other column is inferred.
    public var columnTypes: [String: CSVColumnType] = [:]
    public var nullValues = CSVReadOptions.defaultNullValues
    public var trueValues = CSVReadOptions.defaultTrueValues
    public var falseValues = CSVReadOptions.defaultFalseValues
    /// Whether a string column's null spellings are null (pyarrow's default: they are strings).
    public var stringsCanBeNull = false
    /// Whether a quoted value can be null.
    public var quotedStringsCanBeNull = true
    /// Whether string columns are validated as UTF-8; an inferred column that fails becomes `binary`.
    public var checkUTF8 = true
    public var decimalPoint = UInt8(ascii: ".")

    // ArrowMetal
    /// Bytes per GPU thread in the structure scan. Only the speed depends on it.
    public var scanBlockBytes = CSVReadOptions.defaultScanBlockBytes

    public init() {}
}
