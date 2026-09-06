"""Every string and binary function ArrowMetal answers to, against `pyarrow.compute`, option by option.

`test_functions.py` calls each registry row once, on one small example. This file is the differential
run: the same functions over columns of 0, 1, 33, 4097 and 500 003 rows, built out of accented Latin,
Greek, Cyrillic, CJK, emoji, combining marks, titlecase letters, Unicode whitespace, empty strings and
nulls — and across every option Arrow defines, not just the default.

pyarrow.compute is the oracle throughout. Where ArrowMetal deliberately differs from it the test says
so and checks the documented answer instead; there are three such places, each a bug in Arrow rather
than a gap here:

* `binary_slice` with the default `stop` and a negative `step` — pyarrow overflows `INT64_MIN + len`
  and reads out of bounds.
* `binary_join_element_wise(null_handling="skip")` on a row whose columns are *all* null — pyarrow
  emits no offset for that row and returns an array shorter than its input.
* `utf8_split_whitespace` on a whitespace run that reaches the far end of the scan — pyarrow splits
  it into two separators instead of one, so `"a  "` comes back as three pieces rather than two.
  `ascii_split_whitespace` has no such trouble, and neither does this.
"""
import pytest

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

import arrowmetal as am

SIZES = [0, 1, 33, 4097, 500_003]

# Latin with accents, Greek, Cyrillic, CJK, emoji, a combining mark, the titlecase digraphs, Unicode
# whitespace of several widths, the empty string, and the awkward case-mapping code points.
PIECES = [
    "Hello", "WORLD", "hello", "", "a", "ab", "abc123", "007",
    " ", "  ", "\t", "\n", "\x1c", "\x85", "\xa0", " ", "　", "​",
    "Ünïcödé", "ünïcödé", "ÜNÏCÖDÉ", "Straße", "ß", "İstanbul", "ﬁx", "ŉ", "µ", "ĸ", "ſ", "ı",
    "αβγ", "ΑΒΓ", "ΣΊΣΥΦΟΣ", "σίσυφος", "привет", "ПРИВЕТ", "日本語", "😀", "🎉x",
    "ǅungla", "Ǆ", "ǅ", "ǆ", "½", "²", "Ⅷ", "ⅷ", "٣٤", "é", "x-ray", "O'Neil", "a,b", "a,,b",
]


def _rng(seed):
    """A tiny deterministic generator, so a failure is reproducible without depending on `random`."""
    state = (seed * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def sample(n, seed, null_every=7, pieces=PIECES):
    nxt = _rng(seed)
    out = []
    for i in range(n):
        if null_every and i % null_every == 3:
            out.append(None)
            continue
        out.append("".join(pieces[nxt() % len(pieces)] for _ in range(1 + nxt() % 3)))
    return out


def strings(n, seed, **kw):
    return pa.array(sample(n, seed, **kw), type=pa.string())


def ascii_strings(n, seed, **kw):
    only = [p for p in PIECES if all(ord(c) < 0x80 for c in p)]
    return pa.array(sample(n, seed, pieces=only, **kw), type=pa.string())


def binaries(n, seed, **kw):
    rows = sample(n, seed, **kw)
    return pa.array([None if r is None else r.encode("utf-8") for r in rows], type=pa.binary())


def A(x):
    return am.MetalArray.from_arrow(x)


def same(got, want, what=""):
    """Compare an ArrowMetal answer with pyarrow's, as Python values."""
    g = got.to_arrow().to_pylist() if isinstance(got, am.MetalArray) else got
    w = want.to_pylist() if hasattr(want, "to_pylist") else want
    assert g == w, what


# ---------------------------------------------------------------------------
# binary_slice and utf8_slice_codeunits: start / stop / step, negatives included

SLICE_CASES = [
    (0, None, 1), (1, 4, 1), (-3, None, 1), (1, -1, 1), (2, 2, 1), (0, 100, 3), (-100, 100, 2),
    (5, 0, -1), (-1, -5, -2), (100, -100, -3), (9, 9, 1), (0, 6, 2), (3, 1, -1),
]


@pytest.mark.parametrize("n", SIZES)
def test_binary_slice(n):
    b = binaries(n, 101)
    for start, stop, step in SLICE_CASES:
        got = A(b).binary_slice(start, stop, step)
        if stop is None and step < 0:
            # pyarrow overflows INT64_MIN + len here; the documented answer is Python's slice.
            want = [None if v is None else v[start::step] for v in b.to_pylist()]
            assert got.to_arrow().to_pylist() == want
            continue
        same(got, pc.binary_slice(b, start, stop, step), f"binary_slice{(start, stop, step)} n={n}")
    assert pc.binary_slice(b, 1, 4).type == pa.binary()
    assert A(b).binary_slice(1, 4).type == "binary"


@pytest.mark.parametrize("n", SIZES)
def test_utf8_slice_codeunits_with_step(n):
    s = strings(n, 103)
    for start, stop, step in SLICE_CASES:
        got = A(s).slice_codeunits(start, stop, step)
        if stop is None and step < 0:
            want = [None if v is None else v[start::step] for v in s.to_pylist()]
            assert got.to_arrow().to_pylist() == want
            continue
        same(got, pc.utf8_slice_codeunits(s, start, stop, step), f"utf8_slice{(start, stop, step)} n={n}")


def test_slice_step_zero_is_refused():
    s = strings(4, 105)
    with pytest.raises(am.ArrowMetalError):
        A(s).binary_slice(0, 1, 0)
    with pytest.raises(am.ArrowMetalError):
        A(s).slice_codeunits(0, 1, 0)


# ---------------------------------------------------------------------------
# The byte operations Arrow defines on `binary` as well as `utf8`


@pytest.mark.parametrize("n", SIZES)
def test_binary_columns_are_accepted(n):
    b = binaries(n, 111)
    same(A(b).byte_length(), pc.binary_length(b), f"binary_length n={n}")
    same(A(b).repeat(3), pc.binary_repeat(b, 3), f"binary_repeat n={n}")
    same(A(b).binary_reverse(), pc.binary_reverse(b), f"binary_reverse n={n}")
    same(A(b).binary_replace_slice(1, 3, b"XY"),
         pc.binary_replace_slice(b, 1, 3, b"XY"), f"binary_replace_slice n={n}")
    same(A(b).binary_slice(1, 5), pc.binary_slice(b, 1, 5), f"binary_slice n={n}")


def test_binary_results_keep_the_binary_type():
    b = pa.array([b"ab", None], type=pa.binary())
    for call in [lambda x: x.repeat(2), lambda x: x.binary_reverse(), lambda x: x.binary_slice(0, 1)]:
        assert call(A(b)).to_arrow().type == pa.binary()


@pytest.mark.parametrize("n", SIZES)
def test_ascii_reverse_and_binary_reverse(n):
    a = ascii_strings(n, 113)
    same(A(a).ascii_reverse(), pc.ascii_reverse(a), f"ascii_reverse n={n}")
    # On ASCII the byte reversal and the code point reversal are the same answer.
    same(A(a).ascii_reverse(), pc.utf8_reverse(a))
    s = strings(n, 115)
    same(A(s).binary_reverse(), pc.binary_reverse(s.cast(pa.binary())), f"binary_reverse n={n}")
    same(A(s).str_reverse(), pc.utf8_reverse(s), f"utf8_reverse n={n}")


def test_ascii_reverse_refuses_non_ascii():
    s = pa.array(["é"])
    with pytest.raises(am.ArrowMetalError):
        A(s).ascii_reverse()
    with pytest.raises(pa.ArrowInvalid):
        pc.ascii_reverse(s)


# ---------------------------------------------------------------------------
# ascii_* counts bytes where utf8_* counts code points


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("width", [0, 1, 8, 12])
def test_padding_byte_versus_code_point(n, width):
    s = strings(n, 121)
    same(A(s).ascii_lpad(width, "*"), pc.ascii_lpad(s, width, "*"), f"ascii_lpad {width} n={n}")
    same(A(s).ascii_rpad(width, "*"), pc.ascii_rpad(s, width, "*"), f"ascii_rpad {width} n={n}")
    same(A(s).ascii_center(width, "*"), pc.ascii_center(s, width, "*"), f"ascii_center {width} n={n}")
    same(A(s).pad_left(width, "*"), pc.utf8_lpad(s, width, "*"), f"utf8_lpad {width} n={n}")
    same(A(s).pad_right(width, "*"), pc.utf8_rpad(s, width, "*"), f"utf8_rpad {width} n={n}")
    same(A(s).utf8_center(width, "*"), pc.utf8_center(s, width, "*"), f"utf8_center {width} n={n}")


def test_the_ascii_utf8_pair_really_differs():
    s = pa.array(["héllo"])                       # six bytes, five code points
    assert A(s).ascii_lpad(8, "*").to_arrow().to_pylist() == ["**héllo"]
    assert A(s).pad_left(8, "*").to_arrow().to_pylist() == ["***héllo"]
    assert A(s).ascii_center(8, "*").to_arrow().to_pylist() == ["*héllo*"]
    assert A(s).utf8_center(8, "*").to_arrow().to_pylist() == ["*héllo**"]


# ---------------------------------------------------------------------------
# Full Unicode case mapping

CASE_CALLS = [
    ("utf8_upper", lambda x: x.upper()),
    ("utf8_lower", lambda x: x.lower()),
    ("utf8_swapcase", lambda x: x.utf8_swapcase()),
    ("utf8_capitalize", lambda x: x.utf8_capitalize()),
    ("utf8_title", lambda x: x.utf8_title()),
    ("ascii_upper", lambda x: x.ascii_upper()),
    ("ascii_lower", lambda x: x.ascii_lower()),
    ("ascii_swapcase", lambda x: x.swapcase()),
    ("ascii_capitalize", lambda x: x.capitalize()),
    ("ascii_title", lambda x: x.ascii_title()),
]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name,call", CASE_CALLS, ids=[c[0] for c in CASE_CALLS])
def test_case_mapping(n, name, call):
    s = strings(n, 131)
    same(call(A(s)), getattr(pc, name)(s), f"{name} n={n}")


@pytest.mark.parametrize("name,call", CASE_CALLS, ids=[c[0] for c in CASE_CALLS])
def test_case_mapping_pure_ascii_column(name, call):
    """An all-ASCII column never leaves the GPU; it still has to give pyarrow's answer."""
    s = ascii_strings(4097, 133)
    same(call(A(s)), getattr(pc, name)(s), name)


def test_case_mapping_every_code_point_to_u017f():
    """The GPU table claims U+0000-U+017F exactly, so check every code point in it."""
    s = pa.array([chr(c) for c in range(0x180)])
    for name, call in CASE_CALLS[:5]:
        same(call(A(s)), getattr(pc, name)(s), name)


def test_case_mapping_beyond_the_latin_blocks():
    """The rows the host takes: Greek final sigma, titlecase digraphs, the iota-subscript blocks,
    ligatures whose full mapping is longer than one character, and Osage."""
    s = pa.array(["ΣΊΣΥΦΟΣ", "σίσυφος", "ǅ", "Ǆ", "ǆ", "ᾈ", "ᾀ", "ﬀ", "ﬁ", "ΐ", "և", "𐐨", "𐒰",
                  "ͅ", "Ⅷ", "ⅷ", "ẞ", "ß", "İ", "ı"])
    for name, call in CASE_CALLS[:5]:
        same(call(A(s)), getattr(pc, name)(s), name)


# ---------------------------------------------------------------------------
# The Unicode predicates and trims, per row

PREDICATES = ["ascii_is_alnum", "ascii_is_alpha", "ascii_is_decimal", "ascii_is_lower",
              "ascii_is_printable", "ascii_is_space", "ascii_is_title", "ascii_is_upper",
              "string_is_ascii", "utf8_is_alnum", "utf8_is_alpha", "utf8_is_decimal",
              "utf8_is_digit", "utf8_is_lower", "utf8_is_numeric", "utf8_is_printable",
              "utf8_is_space", "utf8_is_title", "utf8_is_upper"]
_PREDICATE_METHOD = {"ascii_is_alnum": "is_alnum", "ascii_is_alpha": "is_alpha",
                     "ascii_is_decimal": "is_digit", "ascii_is_lower": "is_lower",
                     "ascii_is_space": "is_space", "ascii_is_upper": "is_upper"}


@pytest.mark.parametrize("n", SIZES)
def test_character_class_predicates(n):
    s = strings(n, 141)
    for name in PREDICATES:
        method = _PREDICATE_METHOD.get(name, name)
        same(getattr(A(s), method)(), getattr(pc, name)(s), f"{name} n={n}")


TRIM_SETS = ["", "Hlo", "éH  ", "　x", "½", "aeiou"]


@pytest.mark.parametrize("n", SIZES)
def test_trims(n):
    s = strings(n, 151)
    same(A(s).utf8_trim(), pc.utf8_trim_whitespace(s), f"utf8_trim_whitespace n={n}")
    same(A(s).utf8_ltrim(), pc.utf8_ltrim_whitespace(s), f"utf8_ltrim_whitespace n={n}")
    same(A(s).utf8_rtrim(), pc.utf8_rtrim_whitespace(s), f"utf8_rtrim_whitespace n={n}")
    same(A(s).trim(), pc.ascii_trim_whitespace(s), f"ascii_trim_whitespace n={n}")
    for chars in TRIM_SETS:
        same(A(s).utf8_trim(chars), pc.utf8_trim(s, chars), f"utf8_trim {chars!r} n={n}")
        same(A(s).utf8_ltrim(chars), pc.utf8_ltrim(s, chars), f"utf8_ltrim {chars!r} n={n}")
        same(A(s).utf8_rtrim(chars), pc.utf8_rtrim(s, chars), f"utf8_rtrim {chars!r} n={n}")


# ---------------------------------------------------------------------------
# Splitting: max_splits, reverse, the empty end pieces and the list column

WS_CORPUS = ["  x  ", "a b  c", "", " ", None, "one", "a\tb\nc", "a\x1cb", "　y　",
             "\xa0z", "a b", "a​b", "   ", "\n\n"]
PAT_CORPUS = ["allbll", "l", "", "lal", None, "xx", ",a,b,", ",,", "é,é", "a,b,c,d"]


@pytest.mark.parametrize("max_splits", [-1, 0, 1, 2, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_ascii_split_whitespace_options(max_splits, reverse):
    s = pa.array(WS_CORPUS, type=pa.string())
    kw = {} if max_splits < 0 else {"max_splits": max_splits, "reverse": reverse}
    same(A(s).split_whitespace(max_splits=max_splits, reverse=reverse),
         pc.ascii_split_whitespace(s, **kw), f"ascii_split_whitespace {max_splits} {reverse}")


# Arrow's Unicode whitespace class, spelled out: Zs, Zl and Zp plus U+0009-U+000D, U+001C-U+001F and
# U+0085. U+200B (zero-width space, category Cf) deliberately is not in it.
_UNICODE_WS = set("\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0     　")
_UNICODE_WS |= {chr(c) for c in range(0x2000, 0x200B)}
_ASCII_WS = set("\t\n\v\f\r ")


def split_ws_oracle(s, unicode, max_splits=-1, reverse=False):
    """Arrow's whitespace split, done properly: cut at every maximal run of whitespace, keeping the
    empty pieces a leading or trailing run leaves behind."""
    ws = _UNICODE_WS if unicode else _ASCII_WS
    runs, i = [], 0
    while i < len(s):
        if s[i] in ws:
            start = i
            while i < len(s) and s[i] in ws:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    if 0 <= max_splits < len(runs):
        runs = runs[len(runs) - max_splits:] if reverse else runs[:max_splits]
    if max_splits == 0:
        runs = []
    out, last = [], 0
    for a, b in runs:
        out.append(s[last:a])
        last = b
    out.append(s[last:])
    return out


def check_ws(s, unicode, max_splits=-1, reverse=False):
    """Our answer against the oracle, and against pyarrow on the rows where pyarrow is right."""
    rows = s.to_pylist()
    got = A(s).split_whitespace(unicode=unicode, max_splits=max_splits,
                                reverse=reverse).to_arrow().to_pylist()
    want = [None if r is None else split_ws_oracle(r, unicode, max_splits, reverse) for r in rows]
    assert got == want, f"split_whitespace(unicode={unicode}, {max_splits}, {reverse})"
    kw = {} if max_splits < 0 else {"max_splits": max_splits, "reverse": reverse}
    fn = pc.utf8_split_whitespace if unicode else pc.ascii_split_whitespace
    theirs = fn(s, **kw).to_pylist()
    if not unicode:
        assert got == theirs, "ascii_split_whitespace must match pyarrow exactly"
        return
    # pyarrow's utf8 form splits the run that reaches the far end of the scan into two separators;
    # compare only the rows where that cannot happen.
    edge_ws = "".join(_UNICODE_WS)
    for i, r in enumerate(rows):
        if not r:
            continue
        run = len(r) - len(r.lstrip(edge_ws)) if reverse else len(r) - len(r.rstrip(edge_ws))
        if run >= 2:
            continue
        assert got[i] == theirs[i], f"row {r!r} disagreed with pyarrow"


@pytest.mark.parametrize("max_splits", [-1, 0, 1, 2, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_utf8_split_whitespace_options(max_splits, reverse):
    check_ws(pa.array(WS_CORPUS, type=pa.string()), True, max_splits, reverse)


def test_utf8_split_whitespace_keeps_runs_maximal():
    """The documented difference, pinned: a maximal run is exactly one separator."""
    s = pa.array(["a  ", "a   ", "    ", "a b  "])
    assert A(s).split_whitespace(unicode=True).to_arrow().to_pylist() == \
        [["a", ""], ["a", ""], ["", ""], ["a", "b", ""]]
    # pyarrow splits the trailing run into two separators, so every row gains one empty piece.
    assert pc.utf8_split_whitespace(s).to_pylist() == \
        [["a", "", ""], ["a", "", ""], ["", "", ""], ["a", "b", "", ""]]
    # ascii_split_whitespace is correct, and this agrees with it.
    same(A(s).split_whitespace(), pc.ascii_split_whitespace(s))


@pytest.mark.parametrize("n", SIZES)
def test_split_whitespace_at_every_size(n):
    s = strings(n, 161)
    same(A(s).split_whitespace(), pc.ascii_split_whitespace(s), f"ascii ws n={n}")
    check_ws(s, True)


@pytest.mark.parametrize("pattern", [",", "l", ",,", "é", "a,b"])
@pytest.mark.parametrize("max_splits", [-1, 0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_split_pattern_options(pattern, max_splits, reverse):
    s = pa.array(PAT_CORPUS, type=pa.string())
    kw = {} if max_splits < 0 else {"max_splits": max_splits, "reverse": reverse}
    same(A(s).split_pattern(pattern, max_splits=max_splits, reverse=reverse),
         pc.split_pattern(s, pattern, **kw), f"split_pattern {pattern!r} {max_splits} {reverse}")


@pytest.mark.parametrize("n", SIZES)
def test_split_pattern_at_every_size(n):
    s = strings(n, 163)
    same(A(s).split_pattern(","), pc.split_pattern(s, ","), f"split_pattern n={n}")
    same(A(s).split_pattern("l", max_splits=2), pc.split_pattern(s, "l", max_splits=2), f"n={n}")
    same(A(s).split_pattern("l", max_splits=2, reverse=True),
         pc.split_pattern(s, "l", max_splits=2, reverse=True), f"n={n}")


@pytest.mark.parametrize("max_splits", [-1, 0, 1, 2])
def test_split_pattern_regex_options(max_splits):
    s = pa.array(PAT_CORPUS, type=pa.string())
    kw = {} if max_splits < 0 else {"max_splits": max_splits}
    same(A(s).split_pattern("l+", regex=True, max_splits=max_splits),
         pc.split_pattern_regex(s, "l+", **kw), f"split_pattern_regex {max_splits}")


def test_split_pattern_regex_refuses_reverse():
    s = pa.array(PAT_CORPUS, type=pa.string())
    with pytest.raises(am.ArrowMetalError):
        A(s).split_pattern("l+", regex=True, max_splits=1, reverse=True)
    with pytest.raises(pa.ArrowNotImplementedError):
        pc.split_pattern_regex(s, "l+", max_splits=1, reverse=True)


def test_split_returns_a_list_column_and_a_pair():
    s = pa.array(PAT_CORPUS, type=pa.string())
    listed = A(s).split_pattern(",")
    assert listed.to_arrow().type == pa.list_(pa.string())
    offsets, values = A(s).split_pattern_pair(",")
    flat = values.to_arrow().to_pylist()
    offs = offsets.to_arrow().to_pylist()
    rebuilt = [None if v is None else flat[offs[i]:offs[i + 1]] for i, v in enumerate(PAT_CORPUS)]
    assert rebuilt == listed.to_arrow().to_pylist()
    ws_offsets, ws_values = A(s).split_whitespace_pair()
    assert ws_offsets.to_arrow().to_pylist()[-1] == len(ws_values.to_arrow())


def test_split_pattern_refuses_an_empty_pattern():
    with pytest.raises(am.ArrowMetalError):
        A(pa.array(["a"])).split_pattern("")


# ---------------------------------------------------------------------------
# binary_join_element_wise: N columns and every null_handling


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("null_handling", ["emit_null", "skip", "replace"])
def test_join_element_wise(n, width, null_handling):
    cols = [strings(n, 171 + k, null_every=3 + k) for k in range(width)]
    got = am.binary_join_element_wise(cols, "-", null_handling, "?").to_arrow().to_pylist()
    rows = [c.to_pylist() for c in cols]
    want = []
    for i in range(n):
        cells = [r[i] for r in rows]
        if null_handling == "emit_null":
            want.append(None if any(c is None for c in cells) else "-".join(cells))
        elif null_handling == "skip":
            want.append("-".join(c for c in cells if c is not None))
        else:
            want.append("-".join("?" if c is None else c for c in cells))
    assert got == want
    # pyarrow agrees wherever its `skip` bug cannot bite (no row is entirely null).
    if null_handling != "skip" or not any(all(r[i] is None for r in rows) for i in range(n)):
        same(got, pc.binary_join_element_wise(*cols, pa.scalar("-"), null_handling=null_handling,
                                              null_replacement="?"))


def test_join_element_wise_all_null_row_is_the_documented_difference():
    x = pa.array([None, "a"])
    y = pa.array([None, "b"])
    assert am.binary_join_element_wise([x, y], "-", "skip").to_arrow().to_pylist() == ["", "a-b"]
    # pyarrow drops the all-null row entirely, so its result is one element short.
    assert len(pc.binary_join_element_wise(x, y, pa.scalar("-"), null_handling="skip")) == 1


def test_join_element_wise_at_500k():
    n = 500_003
    cols = [strings(n, 181 + k, null_every=0) for k in range(3)]
    got = am.binary_join_element_wise(cols, "|")
    same(got, pc.binary_join_element_wise(*cols, pa.scalar("|")))


def test_join_element_wise_rejects_bad_input():
    with pytest.raises(am.ArrowMetalError):
        am.binary_join_element_wise([], "-")
    with pytest.raises(am.ArrowMetalError):
        am.binary_join_element_wise([pa.array(["a"]), pa.array(["a", "b"])], "-")
    with pytest.raises(am.ArrowMetalError):
        am.binary_join_element_wise([pa.array(["a"])], "-", null_handling="nope")


# ---------------------------------------------------------------------------
# extract_regex / extract_regex_span as struct columns

PATTERN = r"(?P<letter>[a-z])(?P<digits>\d+)"


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_extract_regex_struct(n):
    rows = sample(n, 191, pieces=["a1", "b22", "zz", "", "q7x", "é9"])
    s = pa.array(rows, type=pa.string())
    got = A(s).extract_regex_struct(PATTERN).to_arrow()
    want = pc.extract_regex(s, PATTERN)
    assert got.type == want.type
    assert got.to_pylist() == want.to_pylist()


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_extract_regex_span_struct(n):
    rows = sample(n, 193, pieces=["a1", "b22", "zz", "", "q7x", "é9"])
    s = pa.array(rows, type=pa.string())
    got = A(s).extract_regex_span_struct(PATTERN).to_arrow()
    want = pc.extract_regex_span(s, PATTERN)
    assert got.type == want.type
    assert got.to_pylist() == want.to_pylist()


def test_extract_regex_dict_form_still_works():
    s = pa.array(["a1", "b22", None, "zz"])
    groups = A(s).extract_regex(r"(?<letter>[a-z])(?<digits>\d+)")
    assert groups["letter"].to_arrow().to_pylist() == ["a", "b", None, None]
    spans = A(s).extract_regex_span(r"(?<letter>[a-z])(?<digits>\d+)")
    assert spans["digits"][0].to_arrow().to_pylist() == [1, 1, None, None]


def test_extract_regex_needs_a_named_group():
    s = pa.array(["a1"])
    with pytest.raises(am.ArrowMetalError):
        A(s).extract_regex_struct(r"[a-z]\d+")


# ---------------------------------------------------------------------------
# match_like on the GPU, and the regex pre-filter

LIKE_PATTERNS = ["abc", "abc%", "%abc", "%abc%", "a_c", "_bc", "ab_", "%a_c%", "a%b%c", "%", "%%",
                 "", "_", "___", r"a\%b", r"a\_b", "%é%", "_é_", "cust_1%", "c_st%1", "日%本",
                 "%_%", "a%", "%c"]


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_match_like(n):
    s = strings(n, 201)
    for p in LIKE_PATTERNS:
        same(A(s).match_like(p), pc.match_like(s, p), f"match_like {p!r} n={n}")


def test_match_like_at_500k():
    s = strings(500_003, 203)
    for p in ["cust_1%", "%a_c%", "%abc%"]:
        same(A(s).match_like(p), pc.match_like(s, p), f"match_like {p!r}")


def test_match_like_underscore_counts_code_points():
    s = pa.array(["é", "ab", "日", "🎉", "aé", ""])
    for p in ["_", "__", "", "%_%"]:
        same(A(s).match_like(p), pc.match_like(s, p), f"match_like {p!r}")


REGEX_PATTERNS = ["ab", "abc[0-9]+", r"\d{2}-ab-\d+", "a.c", "(ab|zz)", "ab$", "^abc", "ab*c",
                  "ab+c", "x?abc", "[a-z]+ab", "ab(c|d)ef", r"\.ab\.", "a{2,3}bcd"]


@pytest.mark.parametrize("n", [0, 1, 33, 4097])
def test_regex_functions_with_the_prefilter(n):
    """RE2 and ICU disagree at the edges (word boundaries, backreferences, lookaround), so the
    patterns above are restricted to constructs the two engines read identically."""
    s = strings(n, 211)
    for p in REGEX_PATTERNS:
        same(A(s).match_substring_regex(p), pc.match_substring_regex(s, p), f"match {p!r} n={n}")
        same(A(s).count_substring_regex(p), pc.count_substring_regex(s, p), f"count {p!r} n={n}")
        same(A(s).find_substring_regex(p), pc.find_substring_regex(s, p), f"find {p!r} n={n}")
        same(A(s).replace_substring_regex(p, "Z"),
             pc.replace_substring_regex(s, p, "Z"), f"replace {p!r} n={n}")


def test_regex_prefilter_at_500k():
    s = strings(500_003, 213)
    for p in [r"\d{2}-ab-\d+", "abc[0-9]+", "[a-z]+ab"]:
        same(A(s).match_substring_regex(p), pc.match_substring_regex(s, p), f"match {p!r}")


# ---------------------------------------------------------------------------
# Shapes: the empty column and the single row have to survive every kernel


def test_empty_and_single_row_shapes():
    for n in [0, 1]:
        s = strings(n, 221, null_every=1)
        b = binaries(n, 221, null_every=1)
        assert len(A(s).binary_slice(0, 2).to_arrow()) == n
        assert len(A(b).binary_reverse().to_arrow()) == n
        assert len(A(s).ascii_center(4, "*").to_arrow()) == n
        assert len(A(s).upper().to_arrow()) == n
        assert len(A(s).utf8_title().to_arrow()) == n
        assert len(A(s).split_pattern(",").to_arrow()) == n
        assert len(A(s).split_whitespace(unicode=True).to_arrow()) == n
        assert len(A(s).match_like("%a%").to_arrow()) == n
        assert len(am.binary_join_element_wise([s, s], "-").to_arrow()) == n
        assert len(A(s).extract_regex_struct(PATTERN).to_arrow()) == n
