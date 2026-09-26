# Parquet on the GPU

ArrowMetal reads Apache Parquet with Metal compute kernels: from file bytes to Arrow arrays in shared
memory, with the CPU reading column data only for the ZSTD, GZIP and BROTLI codecs and for the
SNAPPY pages the GPU would decode one at a time (dictionary pages, and chunks of a few pages), which
are decompressed on the host straight into the shared buffer the GPU decoders read. Snappy and LZ4
decompression, definition levels, dictionary indices, the delta encodings and `BYTE_STREAM_SPLIT` are
all kernels. The host parses the Thrift footer and the per-page Thrift headers — metadata, not data —
and everything after that runs on the GPU.

cuDF does this for NVIDIA.

- `Sources/ArrowMetal/Parquet/Thrift.swift` — a hand-rolled Thrift compact-protocol reader and writer, in
  the spirit of the FlatBuffers reader in `IPC/FlatBuffers.swift`: no code generator, no runtime, just the
  handful of structs the format defines.
- `Sources/ArrowMetal/Parquet/ParquetMetadata.swift` — the subset of `parquet.thrift` this needs.
- `Sources/ArrowMetal/Parquet/ParquetFile.swift` — the mapped file and the schema tree.
- `Sources/ArrowMetal/Parquet/ParquetReader.swift`, `ParquetColumnDecode.swift`, `ParquetValueDecode.swift`,
  `ParquetTypeMap.swift`, `ParquetList.swift` — the read pipeline.
- `Sources/ArrowMetal/Parquet/ParquetNested.swift` — structs, maps and lists at any depth, from their leaves.
- `Sources/ArrowMetal/Parquet/ParquetArrowSchema.swift` — the stored `ARROW:schema` and what it restores.
- `Sources/ArrowMetal/Parquet/ParquetPageIndex.swift`, `ParquetBloomFilter.swift` — page-level skipping
  with the column and offset indexes, and row-group skipping with bloom filters.
- `Sources/ArrowMetal/Kernels/DecompressSource.swift` / `Decompress.swift` — Snappy and LZ4 on the GPU.
- `Sources/ArrowMetal/Kernels/ParquetDecodeSource.swift` — every decoding kernel.
- `Sources/ArrowMetal/Parquet/ParquetWriter.swift` — a small host-side writer, for round trips.

## The pipeline

For one leaf column, across every selected row group at once:

```
  page headers (host, Thrift only)
    -> decompression       GPU for SNAPPY / LZ4 / LZ4_RAW; host for ZSTD / GZIP / BROTLI and for
                           SNAPPY dictionary pages and dispatches of at most 16 pages;
                           nothing at all for UNCOMPRESSED
    -> pq_page_layout      finds each page's level and value sections *inside* the page
    -> pq_decode_levels    definition levels -> one byte per row, plus each row's rank
    -> pq_page_scan        per-page offsets into the chunk's dense value section
    -> one kernel per encoding, each writing dense values
    -> pq_scatter          dense values -> row positions (skipped when the column has no nulls)
    -> pq_levels_to_bitmap definition levels -> an Arrow validity bitmap
```

### The file's bytes are the GPU's bytes

`ParquetFile` `mmap`s the whole file. `mmap` always returns a page-aligned address, so any range of the
mapping can be wrapped with `makeBuffer(bytesNoCopy:)` — the same trick `MetalArrowBuffer.wrapOrCopy` uses
for zero-copy Arrow import. A column chunk's pages are then addressable by a kernel as byte offsets into
that buffer, and pages are faulted in by whichever kernel first touches them. An **uncompressed** file
needs no staging buffer at all: the mapped file *is* the page buffer the decoders read, so the values go
from page cache to Arrow array without the host ever loading them.

Wrapping is not free — it puts the bytes in the GPU's page tables, at roughly 17 ms per gigabyte on an
M4 Max — so a read wraps exactly the byte span its columns occupy, once, and caches it. That matters
because column chunks are interleaved by row group: in a file with fifty row groups, *one* column's chunks
already span nearly the whole file, so wrapping per column would map the same pages once per column. It
also means **holding the `ParquetFile` across queries is worth real time**; see the benchmarks.

Chunks are addressed with 32-bit offsets relative to the wrap's base, so file size is not a limit; a
single column chunk larger than 4 GiB is (it raises `ParquetError.unsupported`).

### Page layout is computed on the GPU, not the host

A v1 data page stores `<4-byte length><repetition levels><4-byte length><definition levels><values>`. Those
two lengths live *inside* the page, which for a compressed column means they only exist after
decompression. `pq_page_layout` therefore runs as a kernel: one thread per page, filling in the byte
offsets of the three sections and, for a dictionary page, the index bit width stored in its first byte.
A v2 page carries the two lengths in its Thrift header, so the host fills those in and the kernel only
does the arithmetic.

### The parallel RLE strategy

Definition levels, repetition levels and dictionary indices all use Parquet's RLE / bit-packing hybrid: a
sequence of runs, each introduced by a varint header whose low bit picks the run kind (0 = a repeated
value, 1 = a group of 8·n bit-packed values). Run boundaries are only known after the previous header has
been read, so the headers cannot be parsed in parallel. The values inside them can, and that is where all
the work is.

One threadgroup (256 threads) owns one page and walks it in batches of up to 2048 values:

1. **Thread 0 scans forward over run headers only.** It never touches the packed value bytes — for a
   bit-packed run it computes the run's byte length from the width and skips it — until it has covered
   2048 values or filled its table of 256 run records. A record is `(kind, value or byte offset of the
   packed data, values of this run already emitted, where this run starts inside the batch)`. A run that
   spills past the batch is left half-consumed and resumed by the next batch, so runs of any length work
   and a page of any size works.
2. **A threadgroup barrier publishes the table.**
3. **Every thread takes a contiguous slice of 8 batch positions** and binary-searches the run table (256
   entries, threadgroup memory, a few cycles) for the run each position falls in, then decodes its level:
   a constant for an RLE run, an LSB-first bit-field read for a bit-packed one. All of the bit unpacking
   happens here, 256-wide.
4. **The same pass scans "is this level the max definition level?" across the batch** — 8 values serially
   per thread, then `simd_prefix_exclusive_sum` across the 32 lanes of each SIMD group, then a 256-wide
   combine. That gives every row its **rank**: the index of its value in the page's dense, null-free value
   section.

Ranks are the load-bearing idea. Because every row knows where its value sits in the dense section, every
value decoder can be a pure gather, no decoder has to think about nulls, and a column with no nulls skips
the scatter entirely because dense positions and row positions coincide.

A page therefore costs one serial header walk of a few instructions per run, and everything else runs
256-wide while thousands of pages run at once across the GPU.

### Dense values, then one scatter

Every value decoder writes its page's values densely: value *j* of a page lands at `nonNullOffset + j`.
One `pq_scatter` at the end moves the dense values into their row positions using the ranks. Keeping the
two apart means each encoding needs exactly one kernel, and a chunk whose pages **mix encodings** — which
is what a writer produces when a dictionary outgrows its budget and the remaining pages fall back to
`PLAIN` — just runs two kernels over two slices of the page list into the same output buffers.

Variable-length values never get copied twice. Each encoding only fills, for every dense value slot,
*where its bytes are* and *how many*, always as offsets into the page buffer (or, for `DELTA_BYTE_ARRAY`,
into a dense staging buffer built by its own kernel). One prefix sum over per-row lengths produces the
Arrow offsets buffer, and one gather moves every byte of the column in parallel.

### Several row groups, one dictionary

A dictionary column is decoded across every selected row group in a single pass even though each row group
has its own dictionary page: the dictionaries are concatenated, and `pq_decode_rle_values` adds each
page's dictionary base to its codes as it writes them. The result is one Arrow dictionary array over a merged (not deduplicated)
dictionary, which is exactly what Arrow allows.

### Decompression

Snappy and LZ4 are byte-oriented LZ77: a stream of tokens, each either "copy N literal bytes from the
input" or "copy N bytes from N' bytes back in the output". A single block cannot be parsed in parallel —
but a Parquet file has thousands of pages, each an independent block, and that is the parallelism.

One SIMD group (32 lanes) owns one page. Lane 0 walks the token stream and broadcasts each parsed token
with `simd_broadcast`; all 32 lanes then move that token's bytes. A 256-thread threadgroup per page was
tried and is *worse* — a threadgroup barrier per token costs more than the extra lanes are worth, because
real pages have tens of thousands of small tokens rather than a few large ones. Two details do help:

- **The token stream is staged in threadgroup memory.** Every tag byte lane 0 reads is a dependent
  device-memory load, and one of those costs hundreds of cycles. All 32 lanes cooperatively stage the next
  8 KB of the compressed stream into threadgroup memory and the parse reads from there; payload bytes are
  still read straight from device memory, where the reads are wide and coalesced and off the dependency
  chain.
- **Overlapping back-references are still copied in parallel.** When `offset < length` (which is how both
  formats encode runs) the copy repeats a pattern of `offset` bytes, so output byte *i* is
  `out[dst - offset + (i % offset)]` and every one of those source bytes was written before this token
  began. `simdgroup_barrier(mem_flags::mem_device)` before a back-reference orders the stores; a literal
  token reads only the input and skips the barrier.

Per-block status codes are written back so corrupt input becomes an error instead of silently wrong data.
A page that decompresses to nonsense is a separate question: the value decoders clamp what they read to
the page they were given (a `BYTE_ARRAY` length field, for instance), so damaged bytes produce wrong
*values* but never an out-of-bounds access and never an unbounded loop.

A token stream is serial, so a page is one SIMD group's work however large it is, and a dispatch
takes as long as its slowest page. That is fine across the hundreds of pages of a large column chunk
and not for a page the GPU decodes on its own. A column chunk's dictionary page is one such page:
pyarrow writes up to 1 MB of dictionary before falling back to `PLAIN`, and a 790 KB SNAPPY dictionary
page of random `int64` values took about 94 ms on the GPU, where one CPU core decodes a Snappy block
of that size and content in about 0.5 ms. So SNAPPY dictionary pages, and SNAPPY dispatches of at most 16 pages, are decompressed on the
host by a bounds-checked Snappy decoder (`SnappyHost` in `Decompress.swift`), like the codecs below.
Reading that file (1,000,000 rows, an `int64` and a `float64` column, pyarrow's defaults) takes 20-36 ms
per read, against 102-106 ms measured before the change on the same file shape; five columns of a
10,000,000-row, 7-column SNAPPY file take 57-66 ms with the file open and 105-133 ms through a fresh
open, against 221 ms and 305-358 ms measured before the change on the same file shape.
Data pages in larger dispatches stay on the GPU: 50 token-dense pages of 160 KB (one `int64` column of
1,000,000 random values below 10^9) take 15.4 ms there, 123 pages of 64 KB 6.7 ms.

`ZSTD`, `GZIP` and `BROTLI` are decompressed on the host, straight into the shared-memory buffer the GPU
decoders read, with pages spread across cores by `DispatchQueue.concurrentPerform`. GZIP and BROTLI go
through Foundation's Compression framework (`COMPRESSION_ZLIB` is raw DEFLATE, so the gzip container is
stripped first). **The macOS SDK has no `COMPRESSION_ZSTD`**, so libzstd is looked up with `dlopen` at
first use — `$ARROWMETAL_ZSTD`, then `/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`. When it is not
installed a ZSTD column raises `ParquetError.unsupported` naming the missing library rather than returning
wrong data; that is the documented host fallback.

## Supported matrix

### Encodings

| Encoding | Types | Where | Notes |
|---|---|---|---|
| `PLAIN` | all physical types | **GPU** | fixed width is a page-sized copy; `BYTE_ARRAY` is a per-page length walk plus a parallel gather |
| `RLE` | definition and repetition levels, `BOOLEAN` values | **GPU** | the batched run-table decoder above |
| `PLAIN_DICTIONARY`, `RLE_DICTIONARY` | all | **GPU** | returns an Arrow dictionary array, or materialised values with `dictionaryEncoded: false` |
| `DELTA_BINARY_PACKED` | `INT32`, `INT64` | **GPU** | thread 0 parses block headers, all threads unpack their miniblock deltas, a log-step threadgroup scan turns deltas into values |
| `DELTA_LENGTH_BYTE_ARRAY` | `BYTE_ARRAY` | **GPU** | delta-packed lengths, then one prefix sum and one gather |
| `DELTA_BYTE_ARRAY` | `BYTE_ARRAY` | **GPU** | prefix and suffix lengths delta-packed; the prefix chain is serial, so one threadgroup walks its page's values in order while all 256 threads move each value's bytes |
| `BYTE_STREAM_SPLIT` | `FLOAT`, `DOUBLE`, `FIXED_LEN_BYTE_ARRAY` | **GPU** | byte *k* of value *j* is at plane *k*, slot *j* |
| `BIT_PACKED` (deprecated level encoding) | levels | — | rejected with `ParquetError.unsupported`; no writer has emitted it since 2015 |

### Codecs

| Codec | Where | Notes |
|---|---|---|
| `UNCOMPRESSED` | **GPU** (no work) | the mapped file is the page buffer |
| `SNAPPY` | **GPU**, host for dictionary pages and for dispatches of at most 16 pages | one SIMD group per page on the GPU |
| `LZ4` | **GPU** | the Hadoop framing (big-endian sizes) is detected in the kernel |
| `LZ4_RAW` | **GPU** | |
| `ZSTD` | host | `dlopen` of libzstd; a clear error when it is missing |
| `GZIP` | host | Compression framework, gzip container stripped |
| `BROTLI` | host | `COMPRESSION_BROTLI` |
| `LZO` | — | `ParquetError.unsupported` |

### Types

| Parquet | Arrow | Notes |
|---|---|---|
| `BOOLEAN` | `bool` | bit-packed `PLAIN` or `RLE` |
| `INT32` | `int32` | |
| `INT32` + `UNKNOWN` (Arrow's `null` type) | `null` | |
| `INT32` + `INT(8\|16\|32, signed)` | `int8` / `int16` / `int32` | narrowed with the existing cast kernels |
| `INT32` + `INT(8\|16\|32, unsigned)` | `uint8` / `uint16` / `uint32` | |
| `INT32` + `DATE` | `date32` | |
| `INT32` + `TIME(MILLIS)` | `time32[ms]` | |
| `INT32` + `DECIMAL(p,s)` | `decimal128(p,s)` | widened on the GPU |
| `INT64` | `int64` | |
| `INT64` + `INT(64, unsigned)` | `uint64` | |
| `INT64` + `TIMESTAMP(unit)` | `timestamp[unit]`, `UTC` when `isAdjustedToUTC` | the stored zone instead of `UTC` when `ARROW:schema` has one, below |
| `INT64` + `TIME(MICROS\|NANOS)` | `time64[us\|ns]` | |
| `INT64` + `DECIMAL(p,s)` | `decimal128(p,s)` | |
| `INT96` | `timestamp[ns]` | Julian day + nanoseconds, converted on the GPU |
| `FLOAT` / `DOUBLE` | `float32` / `float64` | |
| `BYTE_ARRAY` | `binary` | |
| `BYTE_ARRAY` + `STRING` / `JSON` / `ENUM` | `utf8` | |
| `FIXED_LEN_BYTE_ARRAY` | `fixed_size_binary(n)` | |
| `FIXED_LEN_BYTE_ARRAY` + `DECIMAL(p,s)` | `decimal128(p,s)` | big-endian, sign-extended on the GPU |
| `FIXED_LEN_BYTE_ARRAY` + `FLOAT16` | `float16` | |
| `list<T>` (3-level and 2-level) | `list<T>` | Dremel assembly on the GPU, below; `T` may itself be nested |
| `struct` (a group without a `LIST` / `MAP` annotation) | `struct<...>` | reassembled from its leaves; its leaves also read on their own by dotted path |
| `map` (`MAP` / `MAP_KEY_VALUE`) | `map<K, V>` | a list of key/value entries; `V` may be nested |

### Nesting

Structs, maps and lists nest to any depth — `list<list<T>>`, `list<struct<...>>`, `struct<list<...>>`,
`map<K, list<V>>`, `list<map<K, V>>` — with nulls and empty lists at every level, in the three-level
(`group (LIST) { repeated group list { element } }`) and two-level (`group (LIST) { repeated element }`)
list shapes. Every Arrow-level field carries three numbers from the schema: the definition level at
which it is present, the largest repetition level that starts a new element of it, and the definition
level of its nearest repeated ancestor. Over the level entries of any one leaf beneath the field, an
entry is one **slot** of the field when its repetition level is at most the first and its definition
level reaches the third, and the slot is non-null when the definition level reaches the second. So:

- a **leaf** is its decoded array compacted to its own slots with the package's existing `filter`;
- a **struct** is its children plus a validity bitmap, read off one leaf's entries at the struct's slots;
- a **list** is its child plus offsets: at each of the list's slots, the number of child slots before it,
  which is a prefix sum of the child's slot flags read at the list's own slots;
- a **map** is a list whose child is the key/value entries struct.

Each of those is one flag kernel over the entries (`pq_nest_flags`), one prefix sum and one scatter
(`pq_nest_scatter`), and every leaf is decoded once however many fields read its levels
(`ParquetNested.swift`). A one-level `list<primitive>` keeps the dedicated kernel pair it had before, which
reads the same levels the same way. A struct's leaves still read on their own by dotted path:
`am.read_parquet(path, columns=["addr.city"])` returns the `city` leaf as a flat column, null wherever
`addr` or `city` is null.

### The stored Arrow schema

Parquet has no time zone (only `isAdjustedToUTC`), no duration, no decimal32 / decimal64, no
fixed-size list, no dictionary type, no extension types and no per-field metadata. Arrow writers —
pyarrow, Arrow C++, Polars, arrow-rs — therefore store the original Arrow schema in the file's key/value
metadata under `ARROW:schema` (base64 of an IPC `Schema` message), and the reader decodes it with the IPC
FlatBuffers views in `IPC/FlatBuffers.swift` and puts back what the Parquet schema lost
(`ParquetArrowSchema.swift`):

| Stored Arrow type | Parquet stores | Comes back as |
|---|---|---|
| `timestamp[unit, tz=Z]` | `TIMESTAMP(isAdjustedToUTC)` | `timestamp[unit, tz=Z]`, on the stored unit (a seconds timestamp is stored, and read, as milliseconds) |
| `duration[unit]` | plain `INT64` | `duration[unit]` |
| `decimal32(p,s)` / `decimal64(p,s)` | `INT32` / `INT64` `DECIMAL` | `decimal32(p,s)` / `decimal64(p,s)` |
| `fixed_size_list<T>[n]` | a list | `fixed_size_list<T>[n]`; a null row gets `n` null child slots |
| `dictionary<K, T>` over strings or binaries (a pandas categorical, a Polars `Categorical` or `Enum`) | the values | `dictionary<int32, T>`, unordered, dictionary encoded whatever the `dictionary` switch says (pyarrow keeps the stored index type `K` and the ordered flag; see Limits); a dictionary over any other value type (integers, timestamps, dates) reads as that value type, as pyarrow reads it |
| an extension type | its storage | the storage wrapped in `MetalExtensionArray`, so a consumer that knows the type rebuilds it |
| field `custom_metadata` | — | `ParquetFile.arrowFieldMetadata(column:)`, `am_parquet_field_metadata`, `f.field_metadata(column)`, and on every field of `read_parquet_table`'s Table |

Time zones and durations are restored inside structs, lists and maps too. The file's other
key/value metadata is the schema metadata (`arrowSchemaMetadata`, `am_parquet_schema_metadata`,
`f.schema_metadata`), and a Parquet `field_id` shows up as `PARQUET:field_id`, as it does in pyarrow.
The `null` type needs no stored schema: Parquet annotates a null column `UNKNOWN`, which reads as `null`,
at the top level and inside lists, maps and structs (`list<null>`, `map<string, null>`,
`list<struct<a: null, ...>>`).
The stored fields are matched to the Parquet columns by position, as Arrow's reader matches them; a
stored schema with a different number of top-level fields is ignored, and its `ARROW:schema` key stays
in the schema metadata, as pyarrow keeps it. A stored type the column cannot take (a dictionary claim over
a struct, say) is ignored for that column. A file without `ARROW:schema` reads exactly as its Parquet
schema describes it. A file whose `ARROW:schema` is not base64, or does not decode as a Schema message,
reads the same way, with the undecodable value left in the schema metadata; pyarrow refuses to open such
a file (`Invalid base64 input`, `Corrupted metadata length`), and `test_parquet_nested.py` checks both
sides. A stored field nested more than 64 levels deep is read down to 64 levels and taken as a type no
column matches, so that column reads as its Parquet schema says and the other stored fields still apply,
as in pyarrow (`test_a_stored_field_nested_past_64_levels_is_ignored_like_pyarrow`). A leaf read on its
own by dotted path always reads as the Parquet schema describes it.

## Projection and predicate pushdown

```swift
let f = try ParquetFile(path: "trades.parquet")
let batch = try f.read(ParquetReadOptions(
    columns: ["price", "qty"],
    rowGroups: [0, 1],
    filters: [ParquetFilter(column: "price", op: .gt, value: .int(100))]))
```

```python
cols = am.read_parquet("trades.parquet", columns=["price", "qty"],
                       filters=[("price", ">", 100)])
# already on the GPU, dictionary-encoded or not
total = cols["price"].sum()

# Across several queries, keep the handle: mapping the file is a per-open cost.
f = am.ParquetFile("trades.parquet")
for day in days:
    px = f.read(columns=["price"], filters=[("day", "==", day)])["price"]

# Or let the open-file cache hold it (below).
px = am.read_parquet("trades.parquet", columns=["price"], cache=True)["price"]
```

Only the requested column chunks are ever touched — the other columns' pages are never even faulted
in. `filters` is evaluated against the footer's `min_value` / `max_value` statistics per row group; a
row group whose range cannot contain a match is skipped without reading a page. When the file also has a
column index and an offset index, the same filters then skip pages inside the row groups that remain
(next section). Either way the result is a superset of the matching rows: follow it with `filter` (or a
fused `am.query`) to get exactly the rows. A filter value is a string, a boolean, an integer or a float;
a date or timestamp column is filtered by its stored integer (days since the epoch, or ticks in the
column's unit), and Python raises on any other value (a `datetime.date`, a `Decimal`). A literal of
another kind than the column's (a string against a number, in the C and Swift filter text) never rules a
row group or page out. A string value may hold any character: Python quotes it and escapes `"` and `\`,
so a `;`, a quote or an operator inside it is part of the value; a column name cannot hold `=`, `!`, `<`,
`>` or `;`, and Python raises on one that does. An integer above the int64 range stays exact and is
compared as unsigned against a `uint64` column's statistics, which are read as unsigned
(`test_uint64_literals_past_the_signed_range`,
`test_string_literals_with_quotes_semicolons_and_backslashes`). String and binary statistics are compared
the way Parquet orders them, by unsigned bytes: the min / max stay raw bytes and a string literal is
compared as its UTF-8 bytes, so a composed and a decomposed accent, `Z` before `a` before `é`, and an
emoji above every BMP character are all ordered as pyarrow orders them, row groups and page index alike
(`test_string_statistics_are_ordered_by_bytes`). An integer statistic and a float literal are compared
exactly, never through a double: 2^53 + 1 is above the literal 2^53
(`test_int64_statistics_against_a_double_literal_near_2_pow_53`). A decimal column's statistics hold
its unscaled integer, so they never rule a row group or page out
(`test_decimal_statistics_stored_as_integers_are_not_compared_unscaled`). On a `float` or `double` column `!=`
never rules a row group or page out: writers leave NaN out of min / max, so a range of one value equal to
the literal can still hold a NaN, which `!=` keeps. pyarrow's `read_table(filters=...)` rules out a row
group whose min and max both equal the literal, and so leaves out that group's NaN rows; ArrowMetal
returns them (`test_not_equal_keeps_a_nan_row_group_where_pyarrow_drops_it`).
`ParquetFile.selectedRowGroups(_:)` / `ParquetFile.selected_row_groups(...)` report what the row-group
statistics keep without reading anything, and `ParquetFile.column_null_count(column)`
(`am_parquet_column_null_count`) a top-level column's null count: 0 for a `required` column, else
the sum of the row groups' `null_count` statistics, or None when a row group does not record it.

### Page-level skipping

A writer may store, after the row groups, a **column index** (each data page's min, max and whether it
holds only nulls) and an **offset index** (where each data page starts and its first row). pyarrow writes
them with `write_page_index=True`; Polars writes them by default. With them a filter narrows each row
group to *candidate row ranges* (`ParquetPageIndex.swift`):

1. For each filter on a flat column, a page whose [min, max] cannot satisfy it, or that holds only nulls,
   rules out its rows. The ranges the filters leave are intersected; a row group left with none is not
   read at all, even when its row-group statistics let it through. A page counts as all-null only when
   the index's null count for it covers every row: Polars (1.44) flags each page that holds a NaN as a
   null page with a null count of 0, and such a page is kept whole. A NaN min or max rules nothing out,
   as the format asks. Polars also leaves the flagged pages out of the row group's min / max, so when a
   chunk's column index shows such a page, that column's row-group statistics do not drop the row
   group; `pyarrow.parquet.read_table(filters=...)` trusts them and returns no rows for a filter outside
   the unflagged pages' range, where ArrowMetal returns every match
   (`test_polars_row_group_statistics_leave_nan_pages_out`).
2. Every flat column with an offset index decodes only the data pages that overlap a candidate range.
   The page list comes from the offset index, so a skipped page is never decompressed, never decoded,
   and its header is never read.
3. Pages have different boundaries in different columns, so every column — nested ones and those
   without an offset index included, which decode their row groups whole — is trimmed to exactly the
   candidate rows with one `filter`. All columns of the result cover the same rows.

The rows that match a filter are the same with and without the index; the index only makes the superset
smaller. `ParquetFile.usePageIndex` (`f.use_page_index` in Python, `am_parquet_set_page_index` in C)
turns it off, and `lastReadStatistics` (`f.last_read_stats`, `am_parquet_last_read_stats`) reports what
the last read did: row groups read and skipped by statistics, by the page index and by bloom filters
(below), data pages decoded and skipped, rows returned. `test_parquet_nested.py` compares the exact
matches with and without the index against pyarrow over fifteen filter sets on five files (pyarrow in
three page layouts and once without an index, and Polars), and six filter sets on a float column with a
NaN every 97th row, written by Polars and by pyarrow, and `!=` on pages of 64 rows where a NaN hides in
a page whose min and max both equal the literal; `ParquetPageIndexTests` also checks that the
skipped and decoded pages of flat columns add up to the pages of the row groups read. A filter on a
column inside a list does not narrow pages; the row-group statistics still apply to it.

### Bloom filters

A writer may also store a split-block bloom filter per column chunk (pyarrow with `bloom_filter_options`,
DuckDB for its dictionary-encoded columns). An `==` filter hashes its literal the way the writer hashed
the column's values — xxHash64 of the PLAIN encoding — and a row group whose filter does not have all 8
bits of that hash set is dropped before any page is read, even when its min/max statistics let the value
through (`ParquetBloomFilter.swift`). A set of bits is only a maybe, so the row group is then read as
usual; a bloom filter never drops a row group that holds the value. Only literals with one exact
encoding are looked up — integers stored as INT32 / INT64, strings and binaries, and floating-point
values exactly representable in the column's type other than zero and NaN (which have two encodings
each) — and anything else keeps the row group. `ParquetFile.useBloomFilters` (`f.use_bloom_filters`,
`am_parquet_set_bloom_filters`) turns it off, and the read statistics count the row groups it dropped.
`ParquetBloomFilterTests` checks xxHash64 against its published vectors and every value of the fixture
against its row group's filter; `test_parquet_nested.py` looks up values in seven columns of a pyarrow
file and one of a DuckDB file and requires pyarrow's exact matches with the filters on and off.

## The open-file cache

Opening a file maps it, and the first read of a column hands the mapped pages it covers to Metal; a
`ParquetFile` keeps both for as long as it lives (above, "The file's bytes are the GPU's bytes"). A
caller that cannot hold the handle itself — one `read_parquet` call per query, or the Polars engine,
which sees a file path in every plan — reads through a process-wide cache of open files instead:

```python
cols = am.read_parquet("trades.parquet", columns=["price"], cache=True)   # opens and maps
cols = am.read_parquet("trades.parquet", columns=["price"], cache=True)   # the same open file

am.parquet_cache_info()     # {"entries", "bytes", "max_entries", "max_bytes", "hits", "misses",
                            #  "invalidations", "files"}
am.parquet_cache_limit(max_entries=4, max_bytes=8 << 30)
am.clear_parquet_cache()
```

- **Keyed by the file's identity and state**: its real path, device, inode, modification time in
  nanoseconds and size. A file rewritten in place (new size or new modification time) or replaced by
  another under the same name (new inode) is opened afresh on its next read, and its stale entry is
  dropped then (`invalidations` counts those).
- **Bounded**: least recently used entries are evicted above `max_entries` (16 by default) or above
  `max_bytes`, the sum of the cached files' sizes (a quarter of physical memory by default). A file
  larger than `max_bytes` is read without being cached; `max_entries=0` turns the cache off.
- **Shared**: reads of one cached file are serialised by a lock per file, so `last_read_stats`
  belongs to the read that took it. Evicting or clearing an entry closes the file once the last read
  holding it returns.
- `read_parquet` without `cache=True` opens and closes the file on every call, as before.

`python/tests/test_polars_engine.py` checks reuse, invalidation on a new size, a new modification time
with the same size, a touched file and a replaced file, LRU order, both bounds and `clear`.

Cold is the first read in the process (the cache cleared before each run; the file itself stays in
the OS page cache), warm is the next read of the same file through the cache. 50,000,000 rows x 8
columns, the files of `Benchmarks/parquet_bench.py`, best of 3,
`Benchmarks/results/parquet_cache_2026-09-25-quiet.csv` (run conditions in
`Benchmarks/results/bench_conditions_2026-09-25-quiet.txt`):

| codec | file | measure | cold ms | warm ms |
|---|---:|---|---:|---:|
| snappy | 1.65 GB | open only | 0.27 | 0.02 |
| snappy | | read `price` (400 MB of values) | 149.36 | 9.36 |
| snappy | | read `price` + sum | 153.95 | 11.01 |
| snappy | | read all 8 columns | 354.52 | 211.85 |
| lz4 | 1.67 GB | open only | 0.24 | 0.02 |
| lz4 | | read `price` | 152.65 | 9.12 |
| lz4 | | read `price` + sum | 155.00 | 11.03 |
| lz4 | | read all 8 columns | 334.12 | 182.36 |
| none | 2.23 GB | open only | 0.23 | 0.01 |
| none | | read `price` | 191.30 | 7.14 |
| none | | read `price` + sum | 193.62 | 6.79 |
| none | | read all 8 columns | 241.45 | 33.33 |

Opening the file (the footer and the mapping) is 0.23-0.27 ms. The rest of the cold cost is the first
read's hand-off of the mapped pages to Metal and the faults of a fresh mapping: 149-191 ms on a
one-column read, which the warm read does not pay (7.14-9.36 ms).

```
PYTHONPATH=python python Benchmarks/parquet_bench.py --rows 50000000 --codecs snappy,lz4,none \
    --cache --skip-main --keep --cache-out parquet_cache.csv
```

## Benchmarks

Measured on an Apple M4 Max (Mac16,6, 64 GB), macOS 26.x, release build. 50,000,000 rows x 8 columns
(`int64`, `int64`, `int32`, `float64`, `float64`, dictionary-encoded `string`, `timestamp[us]`, `bool`),
1 MB data pages. Best of 3 in-process runs, caches warm. The two tables below are from
`Benchmarks/results/parquet_bench_2026-09-25-quiet.txt` (ArrowMetal 0.2.0; run conditions in
`Benchmarks/results/bench_conditions_2026-09-25-quiet.txt`).

`wall ms` is elapsed time for the whole read; `CPU ms` is process CPU time over the same interval, so GPU
work does not appear in it; `ttfc` is *time to first compute* — read one `float64` column and sum it,
which is the smallest query anyone actually runs.

### Whole-table read, 50 M rows x 8 columns

| codec | reader | wall ms | CPU ms | MB/s | ttfc ms |
|---|---|---:|---:|---:|---:|
| snappy | **arrowmetal (GPU)** | 369 | **175** | 4476 | 154 |
| snappy | pyarrow.parquet | 175 | 1246 | 9464 | 84 |
| snappy | polars | 99 | 1302 | 16751 | 20 |
| snappy | pandas | 239 | 1414 | 6919 | 94 |
| lz4 | **arrowmetal (GPU)** | 340 | **177** | 4909 | 154 |
| lz4 | pyarrow.parquet | 169 | 1045 | 9895 | 87 |
| lz4 | polars | 82 | 1052 | 20411 | 17 |
| lz4 | pandas | 249 | 1218 | 6694 | 109 |
| none | **arrowmetal (GPU)** | 250 | **227** | 8897 | 214 |
| none | pyarrow.parquet | 167 | 766 | 13359 | 85 |
| none | polars | 75 | 936 | 29860 | 19 |
| none | pandas | 248 | 949 | 8968 | 102 |

ArrowMetal is 1.5-4.1x behind Polars and pyarrow on wall time (1.5-2.1x behind pyarrow, 3.3-4.1x
behind Polars) and **3.4-7.4x ahead on CPU time** (3.4-7.1x ahead of pyarrow, 4.1-7.4x ahead of
Polars): the decode is work the host never does. The `ttfc` column above re-opens the file for every
query, which is the wrong way to hold a 2 GB file and costs ArrowMetal the most, because mapping it and
handing its pages to Metal is a per-open cost the CPU readers do not have. Keep the handle, which is what a query engine does:

### One `float64` column (400 MB of values), file handle kept open

| codec | reader | read ms | sum ms |
|---|---|---:|---:|
| snappy | **arrowmetal (GPU)** | **10** | **2** |
| snappy | pyarrow.ParquetFile | 55 | 6 |
| lz4 | **arrowmetal (GPU)** | **12** | **2** |
| lz4 | pyarrow.ParquetFile | 51 | 6 |
| none | **arrowmetal (GPU)** | **6** | **3** |
| none | pyarrow.ParquetFile | 46 | 6 |

That is the shape a query actually has — open once, project a column, compute — and ArrowMetal is
4.3-7.7x ahead on the read (5.5x Snappy, 4.3x LZ4, 7.7x uncompressed) and 2-3x ahead on the reduction,
because the values are already in GPU memory when the reduction starts. 400 MB decoded in 6 ms is
67 GB/s.

### Decompression on its own

One `int64` column of 20 M rows, deliberately *structured* data (an LCG sequence, so Snappy finds many
short matches and emits many tokens — the hard case for a GPU), 1 MB pages:

| codec | file MB | am ms | am MB/s | decode alone |
|---|---:|---:|---:|---:|
| none | 160 | 4.0 | 40154 | — |
| snappy | 152 | 33.6 | 4762 | 5.4 GB/s |
| lz4 | 161 | 4.4 | 36711 | (pyarrow wrote it barely compressed) |
| zstd (host, libzstd) | 120 | 14.1 | 11325 | 15.8 GB/s |

Page size, same column at 5 M rows (40 MB of values), MB/s of decoded output:

| page size | none | snappy | lz4 | zstd (host) | gzip (host) |
|---|---:|---:|---:|---:|---:|
| 1 MB | 19791 | 2226 | 5468 | 3766 | 1658 |
| 256 KB | 14947 | 2245 | 6572 | 5437 | 2411 |
| 64 KB | 13710 | 2511 | 3548 | 4648 | — |

Reproduce with:

```
PYTHONPATH=python python Benchmarks/parquet_bench.py --rows 50000000 --codecs snappy,lz4,none --codec-scan --repeat 3
```

Nested reads — a struct, a list, a list of lists, a map and a list of structs, at 1 M and 10 M rows,
against pyarrow, Polars and DuckDB — have their own script. Its numbers here are from
`Benchmarks/results/parquet_nested_2026-09-24.csv`. The run conditions
at its start are recorded in `Benchmarks/results/bench_conditions_2026-09-24.txt`; this run
started at a load average of 6.10, busier than the other
benchmarks of that day. Wall time in that file, with the fastest CPU reader of each row:

| shape | ArrowMetal, 1 M | fastest CPU, 1 M | ArrowMetal, 10 M | fastest CPU, 10 M |
|---|---:|---:|---:|---:|
| struct | 10.35 ms | 11.44 ms (pyarrow) | 19.99 ms | 27.14 ms (Polars) |
| list | 6.1 ms | 4.96 ms (pyarrow) | 17.54 ms | 14.46 ms (Polars) |
| list of list | 7.39 ms | 9.34 ms (pyarrow) | 25.56 ms | 26.16 ms (Polars) |
| map | 9.69 ms | 17.65 ms (pyarrow) | 36.0 ms | 39.21 ms (Polars) |
| list of struct | 10.76 ms | 6.09 ms (pyarrow) | 24.14 ms | 25.17 ms (pyarrow) |
| all five | 42.81 ms | 19.8 ms (pyarrow) | 115.24 ms | 91.38 ms (pyarrow) |

ArrowMetal is ahead on the struct, list of list and map reads at both sizes and on the list of struct
read at 10 M rows. To improve: the list read (6.1 ms against pyarrow's 4.96 at 1 M rows, 17.54 ms
against Polars' 14.46 at 10 M), the list of struct read at 1 M rows (10.76 ms against 6.09), and the
read of all five columns together (42.81 ms against pyarrow's 19.8 at 1 M rows, 115.24 ms against
91.38 at 10 M). The host CPU time is lower on every row: all five columns at 10 M rows cost 34.99 ms
of CPU against pyarrow's 593.14 and Polars' 1269.84 (`cpu_ms`). Every ArrowMetal row matches
pyarrow's table (`match`). An earlier smoke run, recorded while other work shared the GPU, is kept as
history in `Benchmarks/results/parquet_nested_2026-09-23_provisional.csv`.

```
PYTHONPATH=python python Benchmarks/parquet_nested_bench.py --rows 1000000,10000000 --repeat 3 --out results.csv
```

### What the numbers say

- **CPU time is the headline.** Reading the whole 50 M-row table costs the host 175-227 ms of CPU against
  766-1414 ms for the CPU readers. The decode is compute the process never does, so the cores stay free
  for whatever else is running.
- **The arrays land in GPU memory already.** Summing the column ArrowMetal just decoded takes 2-3 ms; the
  CPU reader pays 6 ms *and* had to materialise the array first. There is no import step, because the
  decode wrote into Metal shared memory in the first place.
- **Opening the file is a real per-open cost, so do not re-open it.** Mapping a 2 GB file and handing its
  pages to Metal costs ~17 ms per gigabyte, and the minor faults of a fresh mapping cost more again;
  together they are most of the `ttfc` column. Holding the `ParquetFile` across queries, which is what
  a query engine does, removes all of it and turns the same read into 6-12 ms.
- **Snappy on *compressible* data is the weak spot, and the reason is structural.** An LZ77 token stream
  is serial, so a page is decoded by one SIMD group, and the cost scales with the number of *tokens*, not
  with bytes. Incompressible pages are one huge literal and decode at memory speed — the `price` column,
  400 MB of random doubles, comes back in 10 ms. A page full of short matches costs a token each, and an
  LCG-generated `int64` column decodes at 2-5 GB/s where the uncompressed path does 14-40 GB/s. Smaller
  pages help a little (the table above) but do not change the shape of it: this is the one part of
  Parquet that a GPU is structurally bad at; the GPU decode is behind the CPU path there.
- **Where the GPU is unambiguously ahead is the uncompressed and dictionary paths**, which is also where
  a GPU-resident analytics stack wants to be: 40 GB/s for a plain `int64` column, and a dictionary column
  that comes back as an Arrow dictionary array without materialising a single string.

## The writer

`ParquetWriter` (and `am.write_parquet`, `am_parquet_write`) is deliberately small: it exists so
ArrowMetal can round-trip its own output, not to compete with pyarrow's writer. It runs on the host —
writing is not the part of Parquet that needed a GPU.

- `PLAIN` for fixed-width columns, `PLAIN` or `RLE_DICTIONARY` for byte arrays.
- `RLE` definition levels; every column is written as `optional`, so nulls round-trip.
- Uncompressed or Snappy (a compact hash-table Snappy encoder in `ParquetWriter.swift`).
- Data page v1, one data page per column chunk, `rowGroupSize` rows per row group.
- Types: all integers, `float32`, `float64`, `bool`, `utf8`, `binary`, `fixed_size_binary`, `date32`,
  `time32`, `time64`, `timestamp`. Dictionary columns are materialised first. Decimals, lists, structs and
  the other nested types are not written.

Files it writes read back with identical values, nulls and types under ArrowMetal *and* pyarrow — over
int64, float64, string, bool and timestamp columns, uncompressed and Snappy; `ParquetWriterTests` and
`python/tests/test_parquet.py::test_writer_round_trip` check both directions.

## Correctness

- `Tests/Fixtures/generate_parquet.py` writes 44 small fixtures (1.8 MB total, committed): every logical
  dataset once per encoding/codec/page-version variant, plus long strings (up to 64 KB), lists of `int64`
  and of `string`, several row groups, decimals stored both as `FIXED_LEN_BYTE_ARRAY` and as `INT32` /
  `INT64`, a struct column, all-null columns, an empty file, a single-row file and INT96 timestamps.
- `Tests/ArrowMetalTests/ParquetTests.swift` reads **every variant of a dataset and compares them element
  for element**, which checks the encodings against each other, plus known-value, type, list, statistics
  and projection tests.
- `python/tests/test_parquet.py` reads **every fixture twice** — once on the GPU, once with
  `pyarrow.parquet` — and asserts the values, nulls and types are identical: 88 checks over the fixture
  set, plus projection, row-group selection, statistics pushdown, dictionary output, struct leaves and a
  50 M-row round trip behind `ARROWMETAL_PARQUET_BIG=1`.
- `python/tests/test_parquet_robustness.py` damages a file two hundred ways — truncation, a broken magic,
  a footer length larger than the file, single-byte damage in the footer and in the pages, a hand-built
  footer nesting Thrift structs 60,000 deep, a 2^64 length, a `num_children` past the schema — and
  requires every one of them to raise rather than crash, hang or read outside the mapping.
- `ParquetWriterTests` and the Python writer test close the round trip.
- `Tests/Fixtures/generate_parquet_nested.py` writes the nested fixtures under `Tests/Fixtures/nested`
  (committed): structs three levels deep, structs of strings and binaries, `map<string, int64>`,
  `map<int32, string>`, maps of structs and of lists, `list<list<T>>`, `list<list<list<int32>>>`,
  `list<struct<...>>`, `struct<list<...>>` and `list<map<...>>`, with nulls and empty lists at every level —
  written by pyarrow in four encoding / codec / page-version variants with 512-byte pages (so levels cross
  hundreds of page boundaries), by DuckDB (`COPY ... TO ... (FORMAT parquet)`) and by Polars wherever it can
  express the shape (it has no map type).
- `Tests/ArrowMetalTests/ParquetNestedTests.swift` checks known values against the generator's formulas
  and every writer's and variant's file against every other's, row by row.
- `python/tests/test_parquet_nested.py` reads every nested fixture with ArrowMetal and with
  `pyarrow.parquet.read_table` and requires the same values and the same types, apart from the three
  differences listed under Limits (32-bit offsets and no view layouts, nullable struct members, the index
  type and ordered flag of a restored dictionary), each of which has its own test showing the difference is exactly that; it also damages nested files 200 ways and
  requires every read to raise or return rather than crash.
- The same generator writes the `ARROW:schema` fixtures: time zones on every unit (named zones, a fixed
  offset, UTC, naive), the `null` type (also below lists, maps and structs), durations, decimal32 /
  decimal64, a fixed-size list, categoricals over strings, binaries, integers, timestamps and dates (and
  a pandas categorical, a Polars `Categorical` and a Polars `Enum`), field metadata and a `field_id`, a registered
  (`arrow.uuid`) and two unregistered extension types, zones and durations inside a struct, a list and a
  map, the view and 64-bit-offset layouts, and the same columns with no `ARROW:schema` (from pyarrow and
  DuckDB). DuckDB's `KV_METADATA` writes the crafted ones: an
  `ARROW:schema` that is not base64, one that is a truncated message, stored schemas with fewer and more
  fields than the file and one whose first field is renamed, and a dictionary claim over a struct.
  `ParquetArrowSchemaTests` and the `ARROW:schema` half of `test_parquet_nested.py` check types, values,
  field metadata and schema metadata against pyarrow.
- It also writes the page-index fixtures (pyarrow with `write_page_index=True` in three page layouts and
  once without, and Polars; plus a float column with NaN from Polars and pyarrow, a NaN inside a page
  whose other values all equal the filter literal, and `uint64` values past the int64 range) and the
  bloom-filter fixtures (pyarrow with `bloom_filter_options`, and DuckDB), which `ParquetPageIndexTests`,
  `ParquetFilterEdgeTests`, `ParquetBloomFilterTests` and the second half of `test_parquet_nested.py` read; that file also damages
  the page indexes 180 ways and reads them with filters, requiring every read to raise or return, and
  reads 40,000 rows of repeated columns three times over.

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter "Parquet"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter "Parquet"
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_parquet.py python/tests/test_parquet_nested.py python/tests/test_parquet_robustness.py -q
python Tests/Fixtures/generate_parquet_nested.py      # regenerate the nested, schema, page-index and bloom fixtures
ARROWMETAL_PARQUET_BIG=1 PYTHONPATH=python python -m pytest python/tests/test_parquet.py -q -k fifty
```

## Limits

- **Duplicate column names are kept, not merged.** `read` hands back a `ColumnSet` — a dict that is
  positional underneath — so `read_parquet(path, columns=["a", "a"])` returns both columns and a file
  with two columns of the same name reads both of them. Indexing by name gives the first; `names`,
  `columns`, `items()` and iteration walk all of them in order.
- **`BIT_PACKED`** (the deprecated level encoding) and **LZO** are rejected.
- **Encrypted files** are not supported.
- **A single column chunk above 4 GiB** is rejected (the file itself has no size limit).
- **Statistics pushdown returns a superset of the matching rows**: row-group granular, or page granular
  when the file has a page index. The deprecated `min`/`max` fields are only used when
  `min_value`/`max_value` are absent (they use a signed byte order that is wrong for strings, which is why
  Parquet deprecated them).
- **ZSTD needs libzstd** installed; see above.
- **The members of a struct and the element of a list are exported as nullable.** A writer that declares
  a struct member `required` gets `not null` on that member from `pyarrow.parquet.read_table`; the values
  are the same, and the member reads as nullable here.
- **A restored dictionary type has `int32` indices and no ordered flag.** The engine's dictionary arrays
  carry `int32` codes only, so a stored `dictionary<int8 | uint8 | uint32, T>` (pandas stores a categorical
  with `int8` indices, Polars a `Categorical` with `uint32` and an `Enum` with `uint8`, ordered) reads here
  as `dictionary<int32, T>`, unordered, where `pyarrow.parquet.read_table` keeps the stored index type
  and flag. The values are the same
  (`test_restored_dictionaries_have_int32_indices_and_no_ordered_flag`).
- **Custom metadata on a nested field is not carried.** A top-level column's field metadata comes back
  (above); metadata on a struct member, a list element or a map value does not, because the engine's
  nested arrays have no per-field metadata. The types still compare equal.
- **`large_string`, `large_binary` and `large_list` come back 32-bit, and the view types as their
  non-view twins.** ArrowMetal narrows 64-bit offsets everywhere and has no view layouts, so a column
  whose stored Arrow type is `large_string` or `string_view` reads here as `string`, `large_list` or
  `list_view` as `list`, with the same values. Polars records `large_string` / `large_list` for every
  string and list column it writes.
- **`decimal256` (precision above 38) is rejected**, as is any Arrow type ArrowMetal does not carry.
- **A dictionary-encoded column decodes itself where the kernels cannot use the codes.** `read_parquet`
  returns dictionary-encoded columns encoded (`dictionary=False`, which `read_parquet_table` passes,
  materialises them instead), and the entry points that need the values rather than the codes decode
  once on the way in: the reductions and statistical aggregates, arithmetic, comparison, `cast`, the
  sorts and `top_k`, and the element-wise maths kernels. The decode is a GPU `take` of the values by
  the codes, so it costs one pass; `dictionary=False` at read time is still the cheaper way to run
  many operations over the same column. The one thing that does not decode for you is the fused
  expression compiler behind `am.query` / `am.scan(...)`, which reads flat columns only: hand it a column
  read with `dictionary=False`.
- **`PLAIN` `BYTE_ARRAY` pages are walked by one thread each** to find the value boundaries — the format
  gives no other option — so a byte-array column with a handful of very large pages has less parallelism
  than a wide one.
- Arrays are capped at 2^32 elements, as everywhere else in ArrowMetal.
