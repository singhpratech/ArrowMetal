import Foundation

/// Arrow's `null_matching_behavior` for `is_in` and `index_in`, over both the primitive lookup in
/// `Kernels/Structural.swift` and the string hash table in `Kernels/StringContainment.swift`.
///
/// ## Why this is a layer and not four kernels
///
/// The four behaviours differ only in what happens to a **null input row**, and — for `inconclusive` —
/// to a non-matching row when the value set itself holds a null. The value of a *matching* row is the
/// same in all four. So the existing kernels, which implement `skip` (a null never matches), already
/// compute everything; the rest is a rewrite of the validity bitmap plus, for `match`, one boolean
/// `or` or one `if_else`. All of that is either metadata (no kernel at all) or an existing GPU kernel,
/// so no behaviour costs an extra pass over the data beyond the one it genuinely needs.
///
/// Written out, with `base` the `skip` answer and `S` = "the value set contains a null":
///
/// | behaviour      | `is_in`                              | `index_in`                          |
/// |----------------|--------------------------------------|-------------------------------------|
/// | `skip`         | `base`                               | `base`                              |
/// | `match`        | `base or isNull` when `S`            | first null of the set where isNull  |
/// | `emitNull`     | `base`, null where the input is null | `base` (already null there)         |
/// | `inconclusive` | true, else null, when `S`            | `base` (already null there)         |
///
/// `inconclusive` is SQL's three-valued `IN`: `x IN (2, NULL)` is true for 2, null for anything else.
/// Its `is_in` result is therefore *only ever true or null* when the set holds a null, which is why the
/// validity bitmap it needs is the value bitmap itself.
enum SetLookup {

    /// Rewrites a `skip`-semantics `is_in` result into whichever behaviour the caller asked for.
    ///
    /// `inputValidity` is the probe column's validity bitmap (nil when it has no nulls) and
    /// `setHasNull` says whether the value set contains one.
    static func isIn(base: MetalBooleanArray, length n: Int, inputValidity: MetalArrowBuffer?,
                     setHasNull: Bool, behavior: SetLookupNullMatching,
                     context ctx: MetalContext) throws -> MetalBooleanArray {
        switch behavior {
        case .skip:
            return base
        case .match:
            guard setHasNull, let inputValidity, n > 0 else { return base }
            return try base.or(try nullMask(length: n, validity: inputValidity, context: ctx))
        case .emitNull:
            guard let inputValidity else { return base }
            return reValidated(base, length: n, validity: inputValidity, context: ctx)
        case .inconclusive:
            if setHasNull {
                // Valid exactly where the answer is true; everything else is "cannot tell".
                return reValidated(base, length: n, validity: base.values, context: ctx)
            }
            guard let inputValidity else { return base }
            return reValidated(base, length: n, validity: inputValidity, context: ctx)
        }
    }

    /// The same for `index_in`. Only `match` differs from the kernels' own answer, because `skip`,
    /// `emitNull` and `inconclusive` all report null for a null input and for a miss.
    static func indexIn(base: MetalArray<Int32>, length n: Int, inputValidity: MetalArrowBuffer?,
                        setFirstNull: Int?, behavior: SetLookupNullMatching,
                        context ctx: MetalContext) throws -> MetalArray<Int32> {
        guard behavior == .match, let firstNull = setFirstNull, let inputValidity, n > 0 else { return base }
        let isNull = try nullMask(length: n, validity: inputValidity, context: ctx)
        return try MetalArray<Int32>.ifElse(isNull, Int32(firstNull), base)
    }

    /// A boolean column that is true exactly where the validity bitmap says "null".
    private static func nullMask(length n: Int, validity: MetalArrowBuffer,
                                 context ctx: MetalContext) throws -> MetalBooleanArray {
        try MetalBooleanArray(length: n, nullCount: 0, validity: nil, values: validity, context: ctx).not()
    }

    /// The same values under a different validity bitmap. Nothing is copied and no kernel runs.
    private static func reValidated(_ a: MetalBooleanArray, length n: Int, validity: MetalArrowBuffer,
                                    context ctx: MetalContext) -> MetalBooleanArray {
        let out = MetalBooleanArray(length: n, nullCount: 0, validity: validity, values: a.values, context: ctx)
        out.recomputeNullCount()
        return out
    }

    /// Whether a value set carries a null, and the position of its first one.
    static func setNulls(length: Int, nullCount: Int, validity: MetalArrowBuffer?) -> (hasNull: Bool, first: Int?) {
        guard nullCount > 0, let validity else { return (false, nil) }
        let bm = validity.typed(UInt8.self)
        for i in 0..<length where !Bitmap.isSet(bm, i) { return (true, i) }
        return (false, nil)
    }
}

extension MetalArray {
    /// Arrow `is_in` with the full `null_matching_behavior` surface.
    ///
    /// `match` is pyarrow's default (`skip_nulls=False`); this package's own default stays `skip`, which
    /// is what the kernel computes and what every existing caller already got.
    public func isIn(_ set: MetalArray<T>,
                     nullMatching: SetLookupNullMatching) throws -> MetalBooleanArray {
        let base = try isIn(set)
        guard nullMatching != .skip else { return base }
        let s = SetLookup.setNulls(length: set.length, nullCount: set.nullCount, validity: set.validity)
        return try SetLookup.isIn(base: base, length: length, inputValidity: validity,
                                  setHasNull: s.hasNull, behavior: nullMatching, context: context)
    }

    /// Arrow `is_in` against a host-side set, with `null_matching_behavior`.
    public func isIn(_ set: [T?], nullMatching: SetLookupNullMatching) throws -> MetalBooleanArray {
        try isIn(try MetalArray<T>(set, context: context), nullMatching: nullMatching)
    }

    /// Arrow `index_in` with the full `null_matching_behavior` surface.
    public func indexIn(_ set: MetalArray<T>,
                        nullMatching: SetLookupNullMatching) throws -> MetalArray<Int32> {
        let base = try indexIn(set)
        guard nullMatching == .match else { return base }
        let s = SetLookup.setNulls(length: set.length, nullCount: set.nullCount, validity: set.validity)
        return try SetLookup.indexIn(base: base, length: length, inputValidity: validity,
                                     setFirstNull: s.first, behavior: nullMatching, context: context)
    }

    /// Arrow `index_in` against a host-side set, with `null_matching_behavior`.
    public func indexIn(_ set: [T?], nullMatching: SetLookupNullMatching) throws -> MetalArray<Int32> {
        try indexIn(try MetalArray<T>(set, context: context), nullMatching: nullMatching)
    }
}

extension MetalStringArray {
    /// Arrow `is_in` over utf8, with the full `null_matching_behavior` surface. The matching itself is
    /// the GPU string hash table; only the null rows are reinterpreted here.
    public func isIn(_ set: MetalStringArray,
                     nullMatching: SetLookupNullMatching) throws -> MetalBooleanArray {
        let base = try isIn(set)
        guard nullMatching != .skip else { return base }
        let s = SetLookup.setNulls(length: set.length, nullCount: set.nullCount, validity: set.validity)
        return try SetLookup.isIn(base: base, length: length, inputValidity: validity,
                                  setHasNull: s.hasNull, behavior: nullMatching, context: context)
    }

    /// Arrow `index_in` over utf8, with the full `null_matching_behavior` surface.
    public func indexIn(_ set: MetalStringArray,
                        nullMatching: SetLookupNullMatching) throws -> MetalArray<Int32> {
        let base = try indexIn(set)
        guard nullMatching == .match else { return base }
        let s = SetLookup.setNulls(length: set.length, nullCount: set.nullCount, validity: set.validity)
        return try SetLookup.indexIn(base: base, length: length, inputValidity: validity,
                                     setFirstNull: s.first, behavior: nullMatching, context: context)
    }

    /// Arrow `is_in` against a host-side value set, with `null_matching_behavior`.
    public func isIn(_ set: [String?], nullMatching: SetLookupNullMatching) throws -> MetalBooleanArray {
        try isIn(try MetalStringArray(set, context: context), nullMatching: nullMatching)
    }

    /// Arrow `index_in` against a host-side value set, with `null_matching_behavior`.
    public func indexIn(_ set: [String?], nullMatching: SetLookupNullMatching) throws -> MetalArray<Int32> {
        try indexIn(try MetalStringArray(set, context: context), nullMatching: nullMatching)
    }
}
