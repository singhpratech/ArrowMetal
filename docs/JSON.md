# JSON on the GPU

ArrowMetal reads newline-delimited JSON with Metal compute kernels and returns Arrow columns with the
semantics of `pyarrow.json.read_json`: the same inferred types, the same field order, the same handling
of missing keys and nulls, the same explicit-schema options and the same error texts. Finding the
records, validating the JSON grammar, matching keys to fields, unescaping strings and parsing ISO-8601
timestamps all run on the GPU; number text is gathered on the GPU and handed to
`MetalStringArray.parse`.

```python
import arrowmetal as am
cols = am.read_json("events.jsonl")               # ColumnSet of MetalArrays, already on the GPU
cols["latency_ms"].mean()
table = am.read_json_table("events.jsonl")        # the same columns as a pyarrow.Table (zero copy)

import pyarrow as pa, pyarrow.json as pj
am.read_json_table("events.jsonl", parse_options=pj.ParseOptions(
    explicit_schema=pa.schema([("id", pa.int32())]), unexpected_field_behavior="ignore"))
am.read_json_table("events.jsonl", explicit_schema=pa.schema([("id", pa.int32())]))  # same, as keywords
```

```swift
let table = try JSONReader.read(path: "events.jsonl")            // JSONTable: names, columns, rowCount
let typed = try JSONReader(path: "events.jsonl").read(JSONReadOptions(
    explicitSchema: [JSONField("id", .int32)], unexpectedFieldBehavior: .ignore))
```

The C ABI is `am_json_open` / `am_json_open_buffer` / `am_json_read` and the `am_json_batch_*`
accessors in `include/arrowmetal.h`, shaped like `am_parquet_batch`; the explicit schema crosses as a
struct-typed `ArrowSchema`.

- `Sources/ArrowMetal/JSON/JSONReader.swift` — the public API, the options and the read pipeline.
- `Sources/ArrowMetal/JSON/JSONColumns.swift` — keys to fields, type inference, the column builders and
  the recursion into nested values.
- `Sources/ArrowMetal/Kernels/JSONSource.swift` — every kernel; `JSONKernels.swift` dispatches them.
- `Sources/ArrowMetalC/ArrowMetalC_JSON.swift` — the C ABI.

## The pipeline

```
  file bytes (parallel pread into a pooled Metal buffer)
    -> structure    one thread per 256-byte block: escapes, strings, depth, record boundaries
    -> walk         one thread per record: full grammar check, one entry per field (key span,
                    value span, kind), counted then written at scanned offsets
    -> keys         positional byte compare against the first record's keys; the rest through the
                    GPU dictionary encoder
    -> slot matrix  entry index per (field, row), -1 for a missing key; kinds OR-reduced per field
    -> columns      one validity pass, one text gather and one parse per scalar type; nested values
                    walk again one level down
```

### Records

A backslash escapes the byte after it, and whether a block of the file starts inside an escape depends
only on the parity of the backslash run that ends the last block before it that is not all backslashes.
`jb_escape` writes that as a key per block and a max-scan gives every block its escape carry. With the
carry known, a block's unescaped quotes are counted, and a sum-scan of their parity gives the in-string
state at every block start; brackets outside strings are counted the same way, and a sum-scan gives the
nesting depth at every block start.

Inside a block the work is bit masks over 64-byte chunks: SWAR compares classify eight bytes at a time
into backslashes, quotes and brackets (`'['` and `'{'` differ in one bit, as do `']'` and `'}'`), the
escaped bytes come from the carry (with a byte loop only for chunks that hold a backslash), and the bytes
inside strings are the prefix XOR of the unescaped-quote mask. The record pass then visits only the
brackets, in order, and reads one at a time only the bytes that sit between them at depth 0 — normally a
newline. A `{` at depth 0 opens a record and the bracket that brings the depth back to 0 closes it;
records need not be one per line (`{"a":1}{"a":2}` is two records, and an object may span lines), which
is what pyarrow reads too. A `null` at depth 0 is a record whose fields are all null, as pyarrow reads it
after the first object. Anything else at depth 0 that is not whitespace is an error, reported with the
text a sequential parser gives (`Column() changed from object to array`, `The document is empty.`, a
string's own error when the string is malformed).

### The walk

Each record is walked by one thread with an explicit container stack (1024 levels). The walk checks the
whole grammar as RapidJSON — the parser inside pyarrow — checks it: literals, the number grammar
including `NaN`, `Inf` and `Infinity`, RapidJSON's "Number too big to be stored in double" rule for
positive exponents, string escapes, `\u` surrogate pairs, control characters in strings, and every
structural position. Errors carry RapidJSON's texts. For every immediate child of the record it emits an
entry: the key span, the value span and a kind (null, false, true, integer that fits int64, other
number, string, object, array) with flags for escaped keys and strings and for `NaN`/`Inf`. The walk runs
once to count and once to write at the scanned offsets.

When a record holds a syntax error, the walk still reports the children it reached — including a key
whose value never came and the kind of a container it had opened — so the columns see what a sequential
parser had seen before the error. That is how a type conflict or a repeated key earlier in the same
record is reported ahead of the syntax error, as pyarrow reports it.

### Keys and fields

Field order is first appearance. Entry *k* of every record is compared, byte for byte, with the key of
entry *k* of the first record that has fields; when every entry matches, the field ids are the positions
and nothing else is needed. Keys that do not match are unescaped, prefixed by the reference keys, and
dictionary-encoded on the GPU (`dictionaryEncode`, first-seen order), which gives new fields their ids in
order of first appearance. The entries are then scattered into a slot matrix — one column of row slots
per field, -1 where a record lacks the key — with an atomic minimum, so a key named twice in one object
leaves the later entry without its slot: that is the "was specified twice" check. For groups of up to 64
fields the same scatter OR-reduces each field's kinds through threadgroup atomics; wider groups take a
separate pass. Matrices above 512 MB are built a group of fields at a time.

### Columns

Every field's kinds decide its type (the table below) or, against an explicit type, whether the column
conflicts. Columns that come out as the same scalar type are built together — one validity pass, one
gather and one parse for all of them — so a file with hundreds of fields costs a few dispatches per type
rather than per column.

- **Numbers**: the value text is gathered into one utf8 array and parsed by `MetalStringArray.parse` —
  integers on the GPU; floats on the CPU in this build, the path the string-to-float work speeds up.
  `NaN`, `Inf` and `Infinity` are set by the reader rather than left to the text parser.
- **Strings**: a length pass and a write pass; strings without escapes are copied, escaped ones decoded
  (`\uXXXX` and surrogate pairs to UTF-8) on the GPU.
- **Timestamps**: a string column whose every value is an ISO-8601 timestamp becomes `timestamp[s]`, as
  in pyarrow. The kernel (`jt_parse`) follows the rules of Arrow's `ParseTimestampISO8601`: the date
  alone or with an hour, hour and minute, or hour, minute and second after a space or a `T`; then `Z` or
  one of three offset forms, converted to UTC; fractions only where the unit holds them. It is its own
  kernel rather than a call to the `strptime` kernel in `TemporalFormat.swift` because one `strptime`
  format describes one layout, and inference has to accept all of them, with their zones, in one pass.
- **Booleans**, **validity**: bitmaps written 32 rows per thread.
- **Nested values**: a struct or list column gathers the spans of its objects or arrays and walks them
  one level down with the same kernel; a struct's fields go through the same key matching and column
  building, and a list's elements become its child column with the walk's offsets as the list offsets.

## Type inference

| JSON values in a column | Arrow type | Probe in `python/tests/test_json.py` |
|---|---|---|
| only `null` (or the key never present) | `null` | `null_only` |
| `true` / `false` | `bool` | `bools` |
| integers that fit int64 | `int64` | `basic`, `int64_max`, `int64_min` |
| any number with `.`, `e`, `NaN`/`Inf`, or outside int64 | `double` (integers promoted) | `int_then_double`, `int_overflow_is_double`, `nan_inf` |
| strings, every one an ISO-8601 timestamp | `timestamp[s]` | `ts_*` |
| other strings | `string` | `ts_then_string`, `strings_and_nulls` |
| objects | `struct`, fields in first-appearance order, missing fields null | `struct`, `struct_missing_in_row` |
| arrays | `list<item>`, the item type inferred over all elements | `list`, `list_of_structs` |
| only empty arrays | `list<item: null>` | `list_only_empty` |
| two classes (number, string, boolean, object, array) | error | `conflict_*` |

The ISO-8601 forms that infer as timestamps, all probed against pyarrow: `YYYY-MM-DD`,
`YYYY-MM-DD[ T]hh`, `...hh:mm`, `...hh:mm:ss`, each followed optionally by `Z`, `+hh`, `+hhmm` or
`+hh:mm` (not after a bare date). Fractions (`.123`), a lower-case `t`, hour 24, second 60, an invalid
calendar day, a five-digit or negative year and `YYYY-MM-DDZ` keep the column a string.

## pyarrow's behaviour, as probed

Every behaviour below is an input in `PROBES` or `EXPLICIT` in `python/tests/test_json.py`, read by both
readers and compared: the schema, the values and the nulls, or the error text.

| Input | pyarrow 25.0.1 and ArrowMetal |
|---|---|
| `{"a":1}` then `{"a":"x"}` | `JSON parse error: Column(/a) changed from number to string in row 1` |
| `{"a":1,"a":2}` | `JSON parse error: Column(/a) was specified twice in row 0` |
| `{"s":{"x":1}}` then `{"s":{"x":"a"}}` | `... Column(/s/x) changed from number to string in row 1` |
| `{"l":[1,"a"]}` | `... Column(/l/[]) changed from number to string in row 0` |
| a conflict, then a syntax error in the same record | the conflict |
| a syntax error, then a conflict in a later record | the syntax error |
| `{"a":1` (truncated) | `... Missing a comma or '}' after an object member. in row 0` |
| `{"a":1} x` | `... Invalid value. in row 1` |
| `{"a":1}}` | `JSON parse error: The document is empty.` |
| `[1,2]` at the top level | `... Column() changed from object to array in row 0` |
| an empty file | `Empty JSON file` |
| only whitespace or newlines | zero rows, zero columns |
| `{}` rows | rows with no columns (`read_json_table` keeps the row count) |
| a UTF-8 BOM at the start | skipped |
| `\r\n`, `\r`, tabs, spaces between records | whitespace |
| `{"a":1e309}` | `... Number too big to be stored in double. in row 0` |
| `{"a":1.7976931348623157e309}`, `{"a":10e308}` | `inf` |
| invalid UTF-8 inside a string | passed through unchanged (neither reader validates) |

With an explicit schema, the schema's fields come first in schema order and the others follow in order
of first appearance (`infer`), are dropped (`ignore`) or fail with `JSON parse error: unexpected field`
(`error`); `ignore` and `error` have no effect without a schema, and a name given twice keeps its first
type. A value of the wrong class is a conflict against the schema's class
(`Column(/b) changed from string to number in row 0`); a value of the right class that does not convert
fails with `Failed to convert JSON to int8, couldn't parse:300`. Nested struct and list types apply the
same rules one level down.

## Explicit-schema types

`bool`, `int8` to `int64`, `uint8` to `uint64`, `float`, `double`, `string`, `timestamp` in any unit and
timezone (fractions up to the unit's precision, zone offsets converted to UTC), `list` and `struct` of
those. Any other type is rejected with an error naming the field (`date32`, `decimal128`, `binary`,
`large_string`, `float16`, `dictionary`, `null`; `test_explicit_types_outside_the_supported_set_are_rejected`).

## Differences from pyarrow 25.0.1

Each has a test of its own in `python/tests/test_json.py` that checks the difference is exactly this.

- **Row numbers in error messages count from the start of the file.** pyarrow parses in 1 MiB blocks and
  counts rows from the start of the block; the two agree for files within one block and whenever pyarrow
  is given a block as large as the file (`test_error_row_counts_from_the_start_of_the_file`).
- **Field order is always first appearance.** pyarrow's threaded read of a file larger than one block can
  order fields by a later block; with `use_threads=False` its order is first appearance, which is
  ArrowMetal's (`test_field_order_is_first_appearance`).
- **An object may span a line anywhere in the file.** With `newlines_in_values=False` pyarrow splits
  blocks at newlines and fails on an object that crosses a block boundary; ArrowMetal has no blocks and
  reads it, as pyarrow does when the file fits in one block
  (`test_multiline_object_across_a_block_boundary`). `newlines_in_values`, `block_size` and
  `use_threads` are accepted and do not change the result.
- **Leading nulls in a list whose item type is not yet known are kept.** pyarrow 25.0.1 drops them and
  shifts the values that follow (`[null,1]` then `[2]` reads as `[1,2]` then `[0]`); with a struct item
  its process can end. ArrowMetal returns the nulls, which is what pyarrow returns once the type is given
  explicitly (`test_pyarrow_list_leading_null_defect`).
- **A file may start with a `null` record.** It reads as a row of nulls, as pyarrow reads a `null` after
  the first object; pyarrow's process ends on a leading one (`test_leading_null_record`).
- **Nesting deeper than 1024 levels is an error.** pyarrow's process does not survive 2000 levels
  (`test_nesting_limit`). Columns nested deeper than 64 levels read on the GPU but cannot cross into
  pyarrow, whose C Data importer stops at 64 (`test_nesting_deeper_than_pyarrows_import_limit`).
- **Explicit `decimal128`, `binary` and `large_string` are rejected**, where pyarrow converts to them
  (`test_pyarrow_converts_to_types_this_reader_rejects`).
- **When several explicit-schema values fail to convert, the one earliest in the file is named.** pyarrow
  names one of them, depending on its conversion order
  (`test_conversion_error_names_the_first_failing_value`).
- **Errors are `ArrowMetalError`**, with pyarrow's text; pyarrow raises `ArrowInvalid`.

## Benchmarks

`Benchmarks/json_bench.py` reads the same generated event-log file with `am.read_json`,
`am.read_json_table`, `pyarrow.json.read_json`, `polars.read_ndjson`, `pandas.read_json(lines=True)` and
DuckDB's `read_json` (each producing an in-memory table), at 1 M and 10 M rows, in a flat shape and a
nested one that adds a struct and a list per record, after checking ArrowMetal's table against
pyarrow's. The first results, `Benchmarks/results/json_bench_2026-09-23_provisional.csv`, are a smoke
run taken while other work shared the GPU and the cores; the published numbers will come from a quiet
rerun.

```
PYTHONPATH=python python Benchmarks/json_bench.py --rows 1000000,10000000 --shapes flat,nested
PYTHONPATH=python python Benchmarks/json_bench.py --rows 1000000 --readers arrowmetal,pyarrow,polars
```

## Tests

- `python/tests/test_json.py` — every probe and explicit-schema case against pyarrow; 60 seeded random
  files mixing every type, nulls, missing keys, shuffled key order, unicode and escapes; every prefix of
  eight small files (truncation at each byte); 400 files with one or two bytes replaced; escapes and
  backslash runs straddling structure blocks at 43 alignments; wide files, a one-record 5000-field file,
  a file where every record has a new key, long escaped strings and a file past pyarrow's block size;
  the API; and one test per documented difference.
- `Tests/ArrowMetalTests/JSONReaderTests.swift` — the max-scan against a CPU loop; record boundaries of
  random documents (strings holding brackets, escaped quotes, backslash runs across blocks, multi-line
  objects, `null` records) against a sequential CPU scanner; top-level error positions and codes; the
  walk's key spans, value spans, kinds and flags; RapidJSON's error texts; conflicts and repeated keys;
  timestamps against a day-counting reference; explicit schemas; nested structs and lists; random flat
  files against Foundation's `JSONSerialization`; a 400-field file; files on disk.

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter JSONReaderTests
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_json.py -q
```

## Limits and what is to improve

- **Inputs of 4 GiB or more** are rejected: byte positions are 32-bit on the GPU.
- **The whole input is read into memory** and parsed in one pass; there is no streaming or chunked read
  yet.
- **One thread walks each record**, so a single very large record — one line of many megabytes, or one
  long string — is walked serially. Splitting long records and long strings across a SIMD group is to
  improve.
- **Each nested level costs its own set of dispatches**, so deeply nested documents take time
  proportional to their depth.
- **Floats are parsed on the CPU** by `MetalStringArray.parse` in this build.
- **Explicit-schema types** are the set above.
- Arrays are capped at 2^32 elements, as everywhere else in ArrowMetal, and a column set's text at 2 GiB
  (32-bit utf8 offsets).
