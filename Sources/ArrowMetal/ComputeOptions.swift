import Foundation

/// The option surfaces Arrow's compute functions carry, spelled the way Swift spells things.
///
/// Each of these mirrors one field of a `pyarrow.compute` options class, with the same default, so a
/// call that passes nothing behaves like the pyarrow call that passes nothing — except where a note on
/// the function says otherwise.

// MARK: - Ordering

/// Where the null rows of a key column go in a sorted order (Arrow's `null_placement`).
///
/// The placement is independent of the sort direction: `atEnd` puts nulls after every value in an
/// ascending *and* in a descending order, which is what Arrow does.
public enum NullPlacement: String, Sendable, CaseIterable {
    /// Nulls after every value (Arrow's default, `"at_end"`).
    case atEnd
    /// Nulls before every value (`"at_start"`).
    case atStart

    /// The Arrow spelling (`"at_end"` / `"at_start"`).
    public var arrowName: String { self == .atEnd ? "at_end" : "at_start" }

    /// Parses the Arrow spelling.
    public init?(arrowName: String) {
        switch arrowName {
        case "at_end", "atEnd": self = .atEnd
        case "at_start", "atStart": self = .atStart
        default: return nil
        }
    }
}

/// How `rank` numbers the rows of a tie group (Arrow's `tiebreaker`).
public enum RankTiebreaker: String, Sendable, CaseIterable {
    /// Every row of the group takes the group's *first* 1-based position; the next group skips the gap.
    case min
    /// Every row takes the group's *last* 1-based position.
    case max
    /// Rows are numbered in the sorted order, ties broken by the original row order (SQL `ROW_NUMBER`).
    case first
    /// The 1-based index of the distinct value, with no gaps (SQL `DENSE_RANK`).
    case dense

    /// The Arrow spelling, which is the same word.
    public var arrowName: String { rawValue }

    /// Parses the Arrow spelling.
    public init?(arrowName: String) { self.init(rawValue: arrowName) }
}

// MARK: - Set lookup

/// What `is_in` and `index_in` do with a null (Arrow's `null_matching_behavior`).
///
/// The four behaviours differ only in how a **null input row** — and, for `inconclusive`, a non-matching
/// row when the value set itself holds a null — is reported. Non-null rows that match are unaffected.
public enum SetLookupNullMatching: String, Sendable, CaseIterable {
    /// A null input matches a null in the value set (`is_in` gives true, `index_in` the position of the
    /// set's first null). This is pyarrow's default (`skip_nulls=False`).
    case match
    /// A null input never matches: `is_in` gives false and `index_in` gives null
    /// (pyarrow's `skip_nulls=True`).
    case skip
    /// A null input gives a null output.
    case emitNull
    /// A null input gives null, and so does a non-matching input when the value set contains a null —
    /// SQL's three-valued `IN`.
    case inconclusive

    /// The Arrow spelling (`"match"`, `"skip"`, `"emit_null"`, `"inconclusive"`).
    public var arrowName: String {
        switch self {
        case .match: return "match"
        case .skip: return "skip"
        case .emitNull: return "emit_null"
        case .inconclusive: return "inconclusive"
        }
    }

    /// Parses the Arrow spelling.
    public init?(arrowName: String) {
        switch arrowName {
        case "match": self = .match
        case "skip": self = .skip
        case "emit_null", "emitNull": self = .emitNull
        case "inconclusive": self = .inconclusive
        default: return nil
        }
    }
}

// MARK: - Distinct values

/// The order the distinct values of `unique`, `value_counts` and `dictionary_encode` come back in.
public enum ValueOrder: String, Sendable, CaseIterable {
    /// Ascending by value, which is the order the GPU sort already leaves them in and the cheaper of the
    /// two (no extra pass). Nulls are dropped.
    case sorted
    /// Order of first appearance in the input, which is what Arrow returns. Costs one group-min over the
    /// row indices, one stable argsort of those minima and one gather on top of the sorted pass.
    case firstAppearance

    /// The name the Python layer uses (`"sorted"` / `"first_appearance"`).
    public var arrowName: String { self == .sorted ? "sorted" : "first_appearance" }

    /// Parses that name.
    public init?(arrowName: String) {
        switch arrowName {
        case "sorted": self = .sorted
        case "first_appearance", "firstAppearance", "first": self = .firstAppearance
        default: return nil
        }
    }
}
