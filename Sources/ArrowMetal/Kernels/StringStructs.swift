import Foundation

/// `extract_regex` and `extract_regex_span` in the shape Arrow gives them: a **struct column**.
///
/// `Kernels/Regex.swift` and `Kernels/StringExtra.swift` return a dictionary of columns, which is
/// what ArrowMetal could express before `Sources/ArrowMetal/Nested.swift` had a struct type. Both
/// dictionary forms stay — they are the convenient thing to hold in Swift — and these two build
/// Arrow's own shape on top of them:
///
/// * `extract_regex` → `struct<group: utf8, …>`, one field per **named** capture group, in the order
///   the groups appear in the pattern.
/// * `extract_regex_span` → `struct<group: fixed_size_list<int32>[2], …>`, each field a `(start,
///   length)` pair counted in **bytes**, exactly as pyarrow's is.
///
/// In both, a row that does not match and a row that is null are a **null struct**, so the field
/// columns underneath them are never read. Inside a matching row, a named group that took part in no
/// alternative is the empty string for `extract_regex` (Arrow's answer) and a null list for
/// `extract_regex_span`.
///
/// Host-side throughout — ICU decides the match — but the GPU literal pre-filter in
/// `Kernels/StringLike.swift` still keeps the rows that cannot match away from the engine.
extension MetalStringArray {

    /// Arrow `extract_regex` as a struct column, one `utf8` field per named capture group.
    public func extractRegexStruct(_ pattern: String, ignoreCase: Bool = false) throws -> MetalStructArray {
        let names = Self.namedGroups(in: pattern)
        guard !names.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("extractRegex needs at least one named group, e.g. (?<year>\\d+)")
        }
        let columns = try extractRegex(pattern, ignoreCase: ignoreCase)
        // A row matched exactly when its groups came back non-null; the columns all agree on that,
        // so the first one decides and the struct's own validity carries it.
        let first = columns[names[0]]!
        let valid = (0..<length).map { first.isValid($0) }
        let children: [AnyMetalArray] = names.map { .string(columns[$0]!) }
        return try MetalStructArray(names: names, children: children,
                                    valid: valid.contains(false) ? valid : nil, context: context)
    }

    /// Arrow `extract_regex_span` as a struct column, one `fixed_size_list<int32>[2]` field per named
    /// capture group holding that group's `(start, length)` in bytes.
    public func extractRegexSpanStruct(_ pattern: String, ignoreCase: Bool = false) throws -> MetalStructArray {
        let names = Self.namedGroups(in: pattern)
        guard !names.isEmpty else {
            throw ArrowMetalError.invalidArrowArray("extractRegexSpan needs at least one named group, e.g. (?<year>\\d+)")
        }
        let spans = try extractRegexSpan(pattern, ignoreCase: ignoreCase)
        let n = length
        var children: [AnyMetalArray] = []
        var rowMatched = [Bool](repeating: false, count: n)
        for name in names {
            guard let (start, len) = spans[name] else {
                throw ArrowMetalError.invalidArrowArray("no capture group named \"\(name)\" in \"\(pattern)\"")
            }
            // Two int32 slots per row, flattened; the list row is null where the group is.
            let values = try MetalArrowBuffer.allocate(byteCount: Swift.max(n * 2 * 4, 4),
                                                       zeroed: true, context: context)
            let p = values.mutableTyped(Int32.self)
            var valid = [Bool](repeating: false, count: n)
            try context.syncPoint()
            withExtendedLifetime((start, len)) {
                let s = start.valuePointer, l = len.valuePointer
                for i in 0..<n where start.isValid(i) {
                    p[2 * i] = s[i]
                    p[2 * i + 1] = l[i]
                    valid[i] = true
                    rowMatched[i] = true
                }
            }
            let child = MetalArray<Int32>(length: n * 2, nullCount: 0, validity: nil,
                                          values: values, context: context)
            let offsets = try MetalArrowBuffer.allocate(byteCount: (n + 1) * 4, zeroed: false, context: context)
            let op = offsets.mutableTyped(Int32.self)
            for i in 0...n { op[i] = Int32(i * 2) }
            let bm = try NestedSupport.bitmap(valid, context)
            children.append(.list(MetalListArray(length: n, nullCount: valid.filter { !$0 }.count,
                                                 validity: bm, offsets: offsets, values: .int32(child),
                                                 kind: .fixedSize(2), context: context)))
        }
        // A row is a null struct when no group of it matched — which, for a matching row, cannot
        // happen unless every group sat in an unused alternative.
        return try MetalStructArray(names: names, children: children,
                                    valid: rowMatched.contains(false) ? rowMatched : nil, context: context)
    }
}
