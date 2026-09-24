# CSV on the GPU

ArrowMetal reads CSV with Metal compute kernels: the file's bytes go into one shared-memory buffer, a
quote-aware scan finds every field and record boundary on the GPU, and each column is typed with
`pyarrow.csv.read_csv`'s inference rules and converted by a kernel straight into Arrow arrays. The host
parses bytes only for a byte order mark, `skip_rows` and `skip_rows_after_names`, the header names, error
messages and the rare float the GPU parser hands back (below).

The oracle is `pyarrow.csv.read_csv`. For every file and option set in `python/tests/test_csv.py` the two
readers return the same table — names, types, validity and values, floating point bit for bit — or fail
with the same message.

- `Sources/ArrowMetal/CSV/CSVReader.swift` — `CSVReader`, getting the file to the GPU, the header and the
  projection.
- `Sources/ArrowMetal/CSV/CSVScan.swift` — the structure pass and the row check.
- `Sources/ArrowMetal/CSV/CSVColumns.swift` — inference and conversion.
- `Sources/ArrowMetal/CSV/CSVOptions.swift` — `CSVReadOptions`, `CSVColumnType`, `CSVError`.
- `Sources/ArrowMetal/Kernels/CSVSource.swift` — every CSV kernel.
- `Sources/ArrowMetal/Kernels/CSVFloatSource.swift`, `CSVFloatTable.swift` — decimal text to float64 /
  float32 on the GPU, shared with `MetalStringArray.parse(Double.self)`.
- `Sources/ArrowMetalC/ArrowMetalC_CSV.swift` — the C ABI (`am_csv_*` in `include/arrowmetal.h`).

## Using it

```swift
var o = CSVReadOptions()
o.includeColumns = ["price", "qty"]
o.columnTypes = ["qty": .int32]
let batch = try CSVReader(path: "trades.csv", options: o).read()     // MetalRecordBatch
```

```python
import arrowmetal as am
import pyarrow.csv as pc

cols = am.read_csv("trades.csv", include_columns=["price", "qty"])   # ColumnSet of MetalArrays
total = cols["price"].sum()                                          # already on the GPU

# pyarrow's own option objects work too, and read_csv_table returns a pyarrow.Table (zero copy)
t = am.read_csv_table("trades.csv", convert_options=pc.ConvertOptions(column_types={"qty": "int32"}))
```

```c
am_csv_options o;
am_csv_options_init(&o);
const char* keep[] = {"price", "qty"};
o.include_columns = keep; o.n_include_columns = 2;
am_csv_reader* r; am_csv_batch* b;
am_csv_open("trades.csv", &o, &r);
am_csv_read(r, &b);          /* am_csv_batch_rows / _columns / _column_name / _column, as for Parquet */
```

## The pipeline

```
  the file -> one Metal buffer          parallel pread into a pooled, page-aligned shared buffer
  csv_summarize                         each thread runs its block of bytes from every parser state
  csv_scan_local, csv_scan_groups       the blocks' transfer functions, prefix-composed
  csv_emit                              every field and record boundary, as a byte position
  csv_check_fields                      every record has the header's width; complex fields counted
  csv_classify on a sample              Arrow's inference over the first 8192 rows
  csv_convert_rows                      every column converted in one row-major pass, each row checked
                                        against the sampled type
  (strings) offsets scan, csv_str_copy  the utf8 / binary columns' bytes
```

### Getting the bytes to the GPU

By default the file is read with `pread` — up to eight ranges in parallel — into a page-aligned buffer from the
context's pool, which is already a Metal shared buffer. `fileAccess = .map` (`file_access="map"`) maps
the file instead and wraps the mapping with `makeBuffer(bytesNoCopy:)`, as the Parquet reader does.
No copy is made, but on a warm page cache the mapped read measured slower than the copy overall
(105.39 ms against 51.8 ms at 10 M rows in `Benchmarks/results/csv_bench_2026-09-24.csv`);
`Benchmarks/csv_bench.py` records both (`arrowmetal` and `arrowmetal_map`), and `ARROWMETAL_CSV_TRACE=1`
shows where the time goes phase by phase.

### Finding every field: the parser as a table, scanned

RFC 4180 as Arrow reads it is a five-state machine over four byte classes:

| state | other | delimiter | quote | `\n` or `\r` |
|---|---|---|---|---|
| 0 line start | 2 | 1, *field end* | 3 | 0 (an empty line, skipped) |
| 1 field start | 2 | 1, *field end* | 3 | 0, *record end* |
| 2 unquoted field | 2 | 1, *field end* | 2 (a literal quote) | 0, *record end* |
| 3 inside quotes | 3 | 3 | 4 | 3 |
| 4 quote inside quotes | 2 | 1, *field end* | 3 (a doubled quote) | 0, *record end* |

A quote opens a quoted section only at the start of a field. Anywhere else it is an ordinary byte, and
after a closing quote the rest of the field is literal: `ab"c` reads as `ab"c` and `"ab"cd` as `abcd`, which
is what pyarrow does. A quote-parity scan cannot express that, so the scan runs the exact
generalisation of it. Each thread takes a block of bytes (1024 by default, `scanBlockBytes`) and runs it
from all five start states at once, recording for each the end state and the number of boundaries it
passed — the block's *transfer function*. In practice the five runs collapse onto two states (inside
quotes or not) within the first line, and the thread continues with just those two. Transfer functions
compose associatively, so a prefix scan over them (256-wide in threadgroup memory, then once across the
threadgroup totals) gives every block its true start state and the index of its first boundary, and a
second pass over the bytes writes each boundary's position. A boundary is one `uint32`: the byte
position, with bit 31 set when it ends a record.

The table is data passed to the kernels, so the delimiter, the quote character, quoting off and
`double_quote` are all the same code. `\r\n`, a lone `\r` and `\n` all end a record: the second byte of
a `\r\n` pair lands in state 0 as an empty line, and empty lines are skipped, as pyarrow's
`ignore_empty_lines=True` does. A last record with no newline after it is closed by one boundary the host
appends at the end of the data.

`csv_check_fields` then checks the shape in one pass over the boundaries: every record has the header's
width exactly when the record ends fall at `k = w - 1, 2w - 1, ...`, so the first boundary that breaks the
pattern names the offending row. The same pass counts each column's *complex* fields — a doubled quote, or
bytes after a closing quote — whose value is not one contiguous span of the file.

### Spans without materialising them

A field's value is found from two neighbouring boundaries: the bytes between them, or between the quotes
of a quoted field. Column kernels derive it on the fly, so a column costs no per-row span storage. A
column that has complex fields (usually none) materialises its spans, and its complex values are
unescaped into a side buffer by `csv_side_len`, a prefix sum and `csv_unescape`.

### Type inference: a sample, then a checked guess

Arrow infers a column's type as the first kind in a fixed order that every value converts as (null values
convert as anything, except strings when `strings_can_be_null` is off). `csv_classify` computes, for each
row, the set of kinds it converts as and ANDs them over a column (`simd_and`, one atomic per SIMD group).
It runs over the first 8192 rows only. Every kind ahead of the sampled answer already fails on some
sampled row, so the sampled type is Arrow's answer for the whole column exactly when every remaining row
also converts as it — and that is checked by the conversion itself, which flags any row that does not
convert. A column whose guess fails (a float after 8192 integers, say) is inferred over every row
(`csv_kind_range`, then `csv_classify` when rows disagree) and converted again. The file's bytes are
therefore read once per column set in the common case, not twice. `test_type_decided_after_the_sample`
places the deciding value past the sample for thirteen type transitions.

### Conversion: one pass over the rows

`csv_convert_rows` converts every column that has no complex fields in a single kernel: one thread per
row walks the row's fields left to right and writes each column's value, with validity (and booleans)
packed 32 rows at a time by `simd_ballot`. One kernel per column would read every cache line of the file
and of the boundaries once per column; this reads them once. Each column's values are a sub-range of one
shared buffer that stays alive as long as any of them does. Columns with complex fields use one kernel per
column over their materialised spans.

**Integers** follow Arrow's CSV integer grammar, which is not the grammar of Arrow's `cast`: spaces and
tabs around the value are trimmed, `0x` / `0X` with one to twice-the-byte-width hex digits is accepted
(the digits are the value's bit pattern, so `0xFF` is `-1` as `int8`), and `+` is rejected. The package's
string → integer cast (`MetalStringArray.parse(Int64.self)`, `str_parse_int`) implements the cast grammar —
`+` accepted, no trimming, no hex — so the reader has its own kernel rather than reusing that one;
`CSVReaderTests.testIntegerGrammarIsTheCSVOneNotTheCasts` pins the difference. The kernel accumulates up
to 19 significant digits without a range check and checks only the final value, so no 64-bit division
runs on the GPU.

**Floats** parse on the GPU with no floating-point arithmetic (Metal has no `double`): see below.

**Dates, times and timestamps** follow Arrow's `ParseYYYY_MM_DD`, its time-of-day parser and
`ParseTimestampISO8601`, transcribed into the kernel: `YYYY-MM-DD` validated as a calendar date,
`hh:mm` / `hh:mm:ss` / `hh:mm:ss.fraction`, the `T` or space separator, and a zone of `Z`, `±HH`,
`±HHMM` or `±HH:MM` that is applied to the value. The package's GPU `strptime` takes one format per
call and ISO-8601 as Arrow accepts it is several (a date alone, `hh`, `hh:mm`, `hh:mm:ss`, a fraction,
four zone spellings), so it is parsed by its own grammar rather than by `strptime`.

**Strings** are a length pass, the GPU offsets scan, and a copy of each row's bytes (from the file, or
from the side buffer for an unescaped value).

## Float parsing on the GPU

`MetalStringArray.parse(Double.self)` and `parse(Float.self)` were Swift's `Double(_:)` / `Float(_:)` on
the CPU. They now run on the GPU and return the same bits for every row, and the CSV reader uses the same
core (`CSVFloatSource`).

The digits go into a 64-bit integer `w` (the first 19 significant digits) and a power of ten `q`; the
Eisel-Lemire algorithm (the one in fast_float, which Arrow's CSV reader uses) multiplies `w` by a 128-bit
approximation of 5^q from a table of 651 entries, using `mulhi` on `ulong` for the high half, and rounds
to nearest-even from the bits below the mantissa. For up to 19 significant digits that is always
correctly rounded, the same value a correct `strtod` returns. Anything it cannot settle exactly goes back
to the CPU for that row: more than 19 significant digits where the 19-digit prefix and the prefix plus one
round differently; the one product shape the original algorithm leaves to a fallback; and, for the
`parse` API, every spelling outside the plain decimal form and `inf` / `infinity` / `nan` (hex floats,
`nan(payload)`, `snan`, embedded NUL, a malformed exponent). The CPU parses those with the initialiser the
API always used. `ARROWMETAL_FLOAT_PARSE_HOST=1` keeps `parse` on the CPU path.

`CSVFloatParseTests` compares the GPU path with the CPU path bit for bit, validity included, for float64
and float32 over: a hand-written adversarial list (exact halfway points such as `9007199254740993`, the
subnormal boundary `2.4703282292062327e-324` / `...328e-324`, the overflow boundary around
`1.7976931348623158e308`, an 808-digit mantissa, `1e-400`, `1e99999999999999999999`, the float32
boundaries, every inf / nan spelling, hex floats); 60,000 random strings over the float alphabet; 120,000
random decimals of 1 to 30 digits with exponents from -360 to 359; and 100,000 random doubles printed
shortest, as `%.17g` and as `%.25e`, plus odd integers above 2^53 (exact halfway cases) and random
float32 values. It also checks that the GPU decided most rows itself, and that ordinary numbers of up to
19 significant digits never need the CPU. In the CSV reader the same kernel is compared with pyarrow's
parse, bit for bit, by every float case in `test_csv.py`.

## Matching pyarrow

pyarrow's rules were established by probing pyarrow 25.0.1 (the probe below), and every row of this table
is a case in `python/tests/test_csv.py`.

| Rule | pyarrow 25.0.1, matched |
|---|---|
| Inference order | null → int64 → bool → date32 → time32[s] → timestamp[s] → timestamp[ns] → timestamp[s, tz=UTC] → timestamp[ns, tz=UTC] → float64 → string → binary (the first kind every value converts as) |
| `null_values` default | `""`, `#N/A`, `#N/A N/A`, `#NA`, `-1.#IND`, `-1.#QNAN`, `-NaN`, `-nan`, `1.#IND`, `1.#QNAN`, `N/A`, `NA`, `NULL`, `NaN`, `n/a`, `nan`, `null` — matched exactly, no trimming (`" NA"` is not null) |
| `true_values` / `false_values` default | `1 True TRUE true` / `0 False FALSE false`, exact (`tRue` and `" true"` are strings); a column of `0` and `1` is int64 because int64 comes first, but `1` with `true` is bool |
| Strings and nulls | `strings_can_be_null=False`: a string column keeps `NA` and `""` as strings; an all-null column is `null` |
| Quoted values | converted like unquoted ones (`"1"` is int64, `"2020-01-01"` is date32); with `quoted_strings_can_be_null=True` a quoted `""` or `"NA"` is null |
| Integers | `-?digits` in range, or `0x` + 1..16 hex digits; spaces and tabs trimmed; `+3` makes the column float64; out of int64 range makes it float64 |
| Floats | one optional `+` or `-`; `digits[.digits]` / `.digits` with an optional exponent; `inf`, `infinity`, `nan`, `nan(chars)` in any case; spaces and tabs trimmed; no hex; `nan` is null by default because it is in `null_values`, while `NAN` is a NaN; NaN is the quiet NaN with the sign given |
| Dates and times | `YYYY-MM-DD` validated (`2021-02-29` is a string); `hh:mm` and `hh:mm:ss` are time32[s], a fraction makes the column a string; spaces and tabs trimmed |
| Timestamps | not trimmed; `YYYY-MM-DD` alone counts; seconds 60 and hour 24 are rejected; a fraction (up to 9 digits) makes it timestamp[ns], provided the value fits in int64 nanoseconds as Arrow computes it (whole seconds scaled first, then the fraction added: 1677-09-21 00:12:44 through 2262-04-11 23:47:16.854775807, so 1677-09-21 00:12:43.145224192 is outside), and a fractional value outside that range makes the column a string; without a fraction the column is timestamp[s] for any year; a forced timestamp[ns] column raises `invalid value` for a value outside the range; an offset or `Z` makes it UTC with the offset applied; naive and zoned values together are a string column |
| Structure | ragged rows raise `CSV parse error: Row #N: Expected w columns, got m: <row>`, N counting skipped lines and records, not empty lines, and `<row>` the row's text when it is at most 100 bytes, otherwise its first 96 bytes and ` ...` (a row that runs to the end of the file inside an open quote is quoted without its last line terminator, `\n`, `\r` or `\r\n`); a header with no newline after it raises `Empty CSV file or block: cannot infer number of columns`; an empty file raises `Empty CSV file`; a BOM is skipped |
| `skip_rows` | whole lines by their terminators, quotes ignored, empty lines counted; a line with no terminator cannot be skipped |
| `skip_rows_after_names` | rows after the header, skipped without a width check (`a,b` then `1,2,3` is skipped, not an error); a quoted newline stays inside its row; an empty line counts as one of the skipped rows; row numbers in later errors count the skipped rows |
| `delimiter` equal to `quote_char` | accepted; the delimiter wins and no field is quoted, the same table as `quote_char=False` |
| Errors | `In CSV column #c: Row #N: CSV conversion error to <type>: invalid value '<v>'`, and the two zone-offset messages for timestamp columns |

The probe that established the table, abridged (`probe(text, **options)` prints pyarrow's types and
values or its error):

```python
import io, pyarrow as pa, pyarrow.csv as pc
def probe(text, **kw):
    try:
        t = pc.read_csv(io.BytesIO(text.encode()), **kw)
        print([(f.name, str(f.type), t.column(i).to_pylist()) for i, f in enumerate(t.schema)])
    except Exception as e:
        print(type(e).__name__, e)
print(pc.ConvertOptions().null_values, pc.ConvertOptions().true_values, pc.ConvertOptions().false_values)
probe("a\n1\n+3\n")                            # double: '+' is not an integer
probe("a\n0x1F\n")                             # int64 31
probe("a\n 1\n2 \n")                           # int64: trimmed
probe("a\n true\n")                            # string: booleans are not trimmed
probe("a\n1\ntrue\n")                          # bool
probe("a\n12:34:56.123\n")                     # string
probe("a\n2020-01-01 12:34:56 \n")             # string: timestamps are not trimmed
probe("a\n 2020-01-01\n")                      # date32: dates are
probe("a\n2020-01-01 12:34:56+01:00\n")        # timestamp[s, tz=UTC], 11:34:56
probe("a\n2020-01-01 12:34:56Z\n2020-01-01 12:34:56\n")   # string
probe("a\nNAN\n-NAN\nnan(123)\n")              # double NaN (nan alone is a null value)
probe("a,b,c\n\n1,2,3\n4,5\n", read_options=pc.ReadOptions(use_threads=False))   # Row #3
probe('"x\ny"\na,b\n1,2\n', read_options=pc.ReadOptions(skip_rows=1))           # skip_rows ignores quotes
probe("a,b\n1,2,3\n4,5\n", read_options=pc.ReadOptions(skip_rows_after_names=1))  # a=[4] b=[5]: not width-checked
probe("a\n1000-01-01 00:00:00.5\n")             # string: outside int64 nanoseconds
probe("a\n1000-01-01 00:00:00\n")               # timestamp[s]
probe('a"b\n1"2\n', parse_options=pc.ParseOptions(delimiter='"'))                  # a=[1] b=[2]
probe("a\n1\n" + "x" * 300 + ",1\n", read_options=pc.ReadOptions(use_threads=False))  # row text cut: 96 bytes + " ..."
```

## Options

| pyarrow | Swift `CSVReadOptions` | C `am_csv_options` |
|---|---|---|
| `ReadOptions.skip_rows` | `skipRows` | `skip_rows` |
| `ReadOptions.skip_rows_after_names` | `skipRowsAfterNames` | `skip_rows_after_names` |
| `ReadOptions.column_names` | `columnNames` | `column_names`, `n_column_names` |
| `ReadOptions.autogenerate_column_names` | `autogenerateColumnNames` | `autogenerate_column_names` |
| `ParseOptions.delimiter` | `delimiter` | `delimiter` |
| `ParseOptions.quote_char` (or `False`) | `quoteChar` (or `nil`) | `quote_char` (or -1) |
| `ParseOptions.double_quote` | `doubleQuote` | `double_quote` |
| `ConvertOptions.include_columns` | `includeColumns` | `include_columns`, `n_include_columns` |
| `ConvertOptions.include_missing_columns` | `includeMissingColumns` | `include_missing_columns` |
| `ConvertOptions.column_types` | `columnTypes` (`CSVColumnType`) | `column_type_names` + `column_type_formats` (Arrow format strings) |
| `ConvertOptions.null_values` | `nullValues` | `null_values`, `n_null_values` |
| `ConvertOptions.true_values` / `false_values` | `trueValues` / `falseValues` | `true_values` / `false_values` |
| `ConvertOptions.strings_can_be_null` | `stringsCanBeNull` | `strings_can_be_null` |
| `ConvertOptions.quoted_strings_can_be_null` | `quotedStringsCanBeNull` | `quoted_strings_can_be_null` |
| `ConvertOptions.check_utf8` | `checkUTF8` | `check_utf8` |
| `ConvertOptions.decimal_point` | `decimalPoint` | `decimal_point` |
| — | `scanBlockBytes` | `scan_block_bytes` |
| — | `fileAccess` (`.read`, `.map`) | `file_access` (0, 1) |

`column_types` takes null, bool, the eight integer types, float32, float64, utf8, binary, date32,
time32[s|ms], time64[us|ns] and timestamp[s|ms|us|ns] with or without a timezone. In Python the keywords
have pyarrow's names and win over the option objects.

Column names and error messages can hold NUL bytes (a header field `a\0`, an error quoting the value
`1\0`). The C ABI hands both out as NUL-terminated strings and also gives their byte length,
`am_csv_batch_column_name_length` and `am_csv_last_error(&length)`; Python reads them by length, so they
arrive whole (`test_nul_bytes_in_names_and_messages`). Option strings go in the same way: each string
array in `am_csv_options` has a `*_lengths` array beside it (NULL means NUL-terminated strings), and Python
fills them, so `include_columns=["a\0b"]`, a `column_types` key, `column_names`, `null_values`,
`true_values` and `false_values` holding a NUL byte match as they do in pyarrow
(`test_nul_bytes_in_option_strings`, `test_c_option_lengths`). A NUL `delimiter`, `quote_char` or
`decimal_point` is refused, as pyarrow refuses it (`test_nul_option_characters_are_refused`).

## Differences from pyarrow

- **Quoted newlines are always parsed**, which is pyarrow's `newlines_in_values=True`; the tests run
  pyarrow that way. pyarrow's default (`False`) reads a quoted newline correctly while it sits inside one
  of its blocks and raises when one straddles a block boundary;
  `test_quoted_newline_across_pyarrow_blocks` shows both.
- **The ragged-row and conversion errors always carry `Row #N`**, which is the message pyarrow's serial
  reader (`use_threads=False`) gives; its threaded reader leaves the row number out.
- **`skip_rows_after_names` with no complete row after the header**: when nothing after the header
  ends with a line terminator outside quotes (`a,b\n`, or `a,b\n1,2` with no final newline), pyarrow
  raises `straddling object straddles two block boundaries (try to increase block size?)`; this reader
  skips what is there and returns an empty table
  (`test_skip_rows_after_names_with_no_complete_row`).
- **A header name that is not valid UTF-8** has each invalid sequence replaced by U+FFFD (`\xef\xbba`
  becomes `\ufffda`); pyarrow keeps the name's bytes as they are, and its `schema.names` then fails to
  decode in Python (`test_header_name_not_utf8`).
- **The table comes back with one chunk per column**, where pyarrow's has one chunk per block it read; the
  tests compare values, not chunking.
- **Not supported yet**, each raising `NotImplementedError` from Python rather than reading differently:
  `escape_char`, `ignore_empty_lines=False`, `timestamp_parsers`, `auto_dict_encode`, an `encoding` other
  than UTF-8, `invalid_row_handler`, and every `column_types` type outside the list above, among them
  decimal, dictionary, date64, duration, float16, large_string and large_binary (pyarrow reads all of
  these except float16, for which it raises its own `ArrowNotImplementedError`). The input is a file
  path given as `str` or `os.PathLike` (a bytes path is refused, as pyarrow refuses it:
  `test_bytes_path_is_refused`); pyarrow also takes file objects, and decompresses a path ending in `.gz`, `.bz2`, `.lz4`, `.zst` or `.br`, which `am.read_csv` refuses (`test_compressed_extensions_are_refused`).

## Tests

- `Tests/ArrowMetalTests/CSVReaderTests.swift` — the structure scan against an independent byte-at-a-time
  CPU parser over 24 random files full of structural hazards (quoted delimiters and newlines, `\r\n`,
  `\r`, doubled quotes, text after a closing quote, mid-field quotes, empty lines, BOM, missing final
  newline) at scan block sizes 1, 2, 3, 5, 16, 64, 257 and 1024 bytes, in both file-access modes; inference,
  forced types, the error messages, projection, skip options, and a float column against Swift's parse.
- `Tests/ArrowMetalTests/CSVFloatParseTests.swift` — the GPU float parse against the CPU path, above.
- `python/tests/test_csv.py` — the differential against `pyarrow.csv.read_csv`: one case per rule in the
  table above, every option, 60 random files with a fixed seed (every inferred type, nulls, quoting,
  embedded delimiters, quotes and newlines, mixed line endings, BOM, block sizes 1 to 100), 3000-row
  files dense with quotes, doubled quotes and `\r\n` read at scan block sizes 1, 2, 7, 64 and 1024 (so
  they land across block boundaries throughout), a 300-column file, the default null and boolean spellings
  read from pyarrow itself, a quoted newline across one of pyarrow's blocks,
  a 20,000-row file, the type decided after the inference sample, and files written by Polars and DuckDB.

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter "CSVReaderTests|CSVFloatParseTests"
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_csv.py -q
PYTHONPATH=python python Benchmarks/csv_bench.py --rows 1000000,10000000
```

## Benchmarks

`Benchmarks/csv_bench.py` reads a mixed-type file (int64, float64, string, bool, date, timestamp and a
quoted text column) at 1M and 10M rows with ArrowMetal, `pyarrow.csv.read_csv`, `polars.read_csv`,
`pandas.read_csv` (pyarrow engine and default engine) and DuckDB's `read_csv`, and writes the median,
min and max of each. The numbers here are from a quiet run, `Benchmarks/results/csv_bench_2026-09-24.csv`;
the load average and the file-sync process's CPU at its start are recorded in
`Benchmarks/results/bench_conditions_2026-09-24.txt`. In that file, median wall time:

| reader | 1 M rows | 10 M rows |
|---|---:|---:|
| `arrowmetal` (`am.read_csv`, pread) | 8.46 ms | 51.8 ms |
| `arrowmetal_table` (`am.read_csv_table`) | 7.79 ms | 50.84 ms |
| `arrowmetal_map` (`file_access="map"`) | 14.56 ms | 105.39 ms |
| `polars` | 10.37 ms | 98.75 ms |
| `pyarrow` | 16.74 ms | 158.81 ms |
| `pandas_pyarrow` | 21.91 ms | 196.04 ms |
| `duckdb` | 56.06 ms | 261.12 ms |
| `pandas_c` | 403.25 ms | 4174.89 ms |

The default read and `read_csv_table` are ahead of every CPU reader at both sizes; at 10 M rows
`arrowmetal` reads 16580.9 MB/s against Polars' 8697.9. The mapped read is to improve: at 10 M rows it
takes 105.39 ms against Polars' 98.75 ms, and at 1 M rows 14.56 ms against 10.37 ms.

An earlier run, taken while other work shared the GPU, is kept as history in
`Benchmarks/results/csv_bench_2026-09-23_provisional.csv` (named for the lane's day; the run itself
went past midnight, so its `date` column reads 2026-09-24).

`ARROWMETAL_CSV_TRACE=1` prints the wall time of each phase of a read to stderr, and `=2` runs every
kernel in its own command buffer and prints its GPU time.

## Limits

- **Files of 2 GiB and more** are rejected (boundaries are 31-bit positions), and so is a single field
  of 1 GiB or more.
- **Per-field work is one thread per field**: a quoted field of many megabytes is walked and copied by a
  single GPU thread.
- **Arrays are capped at 2^32 elements**, and a utf8 column at the 2 GB its 32-bit offsets address, as
  everywhere else in ArrowMetal.
- The options listed under *Differences* are not implemented.
