# Parquet on the GPU

ArrowMetal reads Apache Parquet with Metal compute kernels: from file bytes to Arrow arrays in shared
memory, with the CPU never reading a byte of column data. Decompression, definition levels, dictionary
indices, the delta encodings and `BYTE_STREAM_SPLIT` are all kernels. The host parses the Thrift footer
and the per-page Thrift headers — metadata, not data — and everything after that runs on the GPU.

cuDF does this for NVIDIA. As far as we know nothing did it for Metal.

- `Sources/ArrowMetal/Parquet/Thrift.swift` — a hand-rolled Thrift compact-protocol reader and writer, in
  the spirit of the FlatBuffers reader in `IPC/FlatBuffers.swift`: no code generator, no runtime, just the
  handful of structs the format defines.
- `Sources/ArrowMetal/Parquet/ParquetMetadata.swift` — the subset of `parquet.thrift` this needs.
- `Sources/ArrowMetal/Parquet/ParquetFile.swift` — the mapped file and the schema tree.
- `Sources/ArrowMetal/Parquet/ParquetReader.swift`, `ParquetColumnDecode.swift`, `ParquetValueDecode.swift`,
  `ParquetTypeMap.swift`, `ParquetList.swift` — the read pipeline.
- `Sources/ArrowMetal/Kernels/DecompressSource.swift` / `Decompress.swift` — Snappy and LZ4 on the GPU.
- `Sources/ArrowMetal/Kernels/ParquetDecodeSource.swift` — every decoding kernel.
- `Sources/ArrowMetal/Parquet/ParquetWriter.swift` — a small host-side writer, for round trips.

## The pipeline

For one leaf column, across every selected row group at once:

```
  page headers (host, Thrift only)
    -> decompression       GPU for SNAPPY / LZ4 / LZ4_RAW; host for ZSTD / GZIP / BROTLI;
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
has its own dictionary page: the dictionaries are concatenated, and `pq_dict_rebase` adds each page's
dictionary base to its codes. The result is one Arrow dictionary array over a merged (not deduplicated)
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
| `DELTA_BYTE_ARRAY` | `BYTE_ARRAY` | **GPU** | prefix and suffix lengths delta-packed; the prefix chain is genuinely serial, so one threadgroup walks its page's values in order while all 256 threads move each value's bytes |
| `BYTE_STREAM_SPLIT` | `FLOAT`, `DOUBLE`, `FIXED_LEN_BYTE_ARRAY` | **GPU** | byte *k* of value *j* is at plane *k*, slot *j* |
| `BIT_PACKED` (deprecated level encoding) | levels | — | rejected with `ParquetError.unsupported`; no writer has emitted it since 2015 |

### Codecs

| Codec | Where | Notes |
|---|---|---|
| `UNCOMPRESSED` | **GPU** (no work) | the mapped file is the page buffer |
| `SNAPPY` | **GPU** | one SIMD group per page |
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
| `INT32` + `INT(8\|16\|32, signed)` | `int8` / `int16` / `int32` | narrowed with the existing cast kernels |
| `INT32` + `INT(8\|16\|32, unsigned)` | `uint8` / `uint16` / `uint32` | |
| `INT32` + `DATE` | `date32` | |
| `INT32` + `TIME(MILLIS)` | `time32[ms]` | |
| `INT32` + `DECIMAL(p,s)` | `decimal128(p,s)` | widened on the GPU |
| `INT64` | `int64` | |
| `INT64` + `INT(64, unsigned)` | `uint64` | |
| `INT64` + `TIMESTAMP(unit)` | `timestamp[unit]`, `UTC` when `isAdjustedToUTC` | |
| `INT64` + `TIME(MICROS\|NANOS)` | `time64[us\|ns]` | |
| `INT64` + `DECIMAL(p,s)` | `decimal128(p,s)` | |
| `INT96` | `timestamp[ns]` | Julian day + nanoseconds, converted on the GPU |
| `FLOAT` / `DOUBLE` | `float32` / `float64` | |
| `BYTE_ARRAY` | `binary` | |
| `BYTE_ARRAY` + `STRING` / `JSON` / `ENUM` | `utf8` | |
| `FIXED_LEN_BYTE_ARRAY` | `fixed_size_binary(n)` | |
| `FIXED_LEN_BYTE_ARRAY` + `DECIMAL(p,s)` | `decimal128(p,s)` | big-endian, sign-extended on the GPU |
| `FIXED_LEN_BYTE_ARRAY` + `FLOAT16` | `float16` | |
| `list<T>` (3-level and 2-level) | `list<T>` | Dremel assembly on the GPU, below |
| `struct` | — | read its leaves by dotted path; a struct column raises `ParquetError.unsupported` |
| `map` | — | not yet |

### Nesting

Flat columns and `list<primitive>` are supported, including null lists, empty lists and null elements, in
both the three-level (`group (LIST) { repeated group list { element } }`) and two-level
(`group (LIST) { repeated element }`) shapes. The assembly is one kernel plus two prefix sums: a new row
starts wherever the repetition level is 0, an element exists wherever the definition level reaches the
repeated node's level, and the row is null when its first entry's definition level does not reach the
enclosing group. Numbering the rows and the elements with two scans lets the row starts write the offsets
buffer directly, and the child array is the leaf array compacted to the element positions with the
package's existing `filter`.

Struct columns are read leaf by leaf: `am.read_parquet(path, columns=["addr.city"])` works, but a
`struct` is not reassembled into `MetalStructArray` yet.

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
total = cols["price"].sum()          # already on the GPU

# Across several queries, keep the handle: mapping the file is a per-open cost.
f = am.ParquetFile("trades.parquet")
for day in days:
    px = f.read(columns=["price"], filters=[("day", "==", day)])["price"]
```

Only the requested column chunks are ever touched — their pages are never even faulted in. `filters` is
evaluated against the footer's `min_value` / `max_value` statistics per row group; a row group whose range
cannot contain a match is skipped without reading a page. Pushdown is row-group granular, so the result is
a superset of the matching rows: follow it with `filter` (or a fused `am.query`) to get exactly the rows.
`ParquetFile.selectedRowGroups(_:)` / `ParquetFile.selected_row_groups(...)` report what a filter keeps
without reading anything.

## Benchmarks

Measured on an Apple M4 Max (Mac16,6, 64 GB), macOS 26.x, release build. 50,000,000 rows x 8 columns
(`int64`, `int64`, `int32`, `float64`, `float64`, dictionary-encoded `string`, `timestamp[us]`, `bool`),
1 MB data pages. Best of 3 in-process runs, caches warm.

`wall ms` is elapsed time for the whole read; `CPU ms` is process CPU time over the same interval, so GPU
work does not appear in it; `ttfc` is *time to first compute* — read one `float64` column and sum it,
which is the smallest query anyone actually runs.

### Whole-table read, 50 M rows x 8 columns

| codec | reader | wall ms | CPU ms | MB/s | ttfc ms |
|---|---|---:|---:|---:|---:|
| snappy | **arrowmetal (GPU)** | 440 | **225** | 3759 | 189 |
| snappy | pyarrow.parquet | 282 | 1955 | 5855 | 127 |
| snappy | polars | 157 | 1469 | 10502 | 36 |
| snappy | pandas | 441 | 2253 | 3750 | 132 |
| lz4 | **arrowmetal (GPU)** | 394 | **217** | 4240 | 203 |
| lz4 | pyarrow.parquet | 293 | 1738 | 5701 | 122 |
| lz4 | polars | 93 | 1109 | 17957 | 21 |
| lz4 | pandas | 306 | 1598 | 5450 | 118 |
| none | **arrowmetal (GPU)** | 305 | **262** | 7301 | 250 |
| none | pyarrow.parquet | 206 | 951 | 10814 | 101 |
| none | polars | 84 | 861 | 26592 | 28 |
| none | pandas | 317 | 1245 | 7030 | 134 |

ArrowMetal is 1.4-3.6x behind Polars and pyarrow on wall time and **4-9x ahead on CPU time**: the decode is
work the host never does. The `ttfc` column above re-opens the file for every query, which is the wrong
way to hold a 2 GB file and costs ArrowMetal the most, because mapping it and handing its pages to Metal
is a per-open cost the CPU readers do not have. Keep the handle, which is what a query engine does:

### One `float64` column (400 MB of values), file handle kept open

| codec | reader | read ms | sum ms |
|---|---|---:|---:|
| snappy | **arrowmetal (GPU)** | **11** | **2** |
| snappy | pyarrow.ParquetFile | 98 | 8 |
| lz4 | **arrowmetal (GPU)** | **16** | **3** |
| lz4 | pyarrow.ParquetFile | 105 | 7 |
| none | **arrowmetal (GPU)** | **7** | **3** |
| none | pyarrow.ParquetFile | 57 | 7 |

That is the shape a query actually has — open once, project a column, compute — and it is a 6-9x win on
the read plus a 2-3x win on the reduction, because the values are already in GPU memory when the
reduction starts. 400 MB decoded in 7 ms is 57 GB/s.

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
PYTHONPATH=python python Benchmarks/parquet_bench.py --rows 50000000 --codecs snappy,lz4,none --codec-scan
```

### What the numbers say

- **CPU time is the headline.** Reading the whole 50 M-row table costs the host 217-262 ms of CPU against
  951-2253 ms for the CPU readers. The decode is compute the process never does, so the cores stay free
  for whatever else is running.
- **The arrays land in GPU memory already.** Summing the column ArrowMetal just decoded takes 2-3 ms; the
  CPU readers pay 7-8 ms *and* had to materialise the array first. There is no import step, because the
  decode wrote into Metal shared memory in the first place.
- **Opening the file is a real per-open cost, so do not re-open it.** Mapping a 2 GB file and handing its
  pages to Metal costs ~17 ms per gigabyte plus the minor faults of a fresh mapping — around 230 ms for
  this file, which is most of the `ttfc` column. Holding the `ParquetFile` across queries, which is what
  a query engine does, removes all of it and turns the same read into 7-16 ms.
- **Snappy on *compressible* data is the weak spot, and the reason is structural.** An LZ77 token stream
  is serial, so a page is decoded by one SIMD group, and the cost scales with the number of *tokens*, not
  with bytes. Incompressible pages are one huge literal and decode at memory speed — the `price` column,
  400 MB of random doubles, comes back in 11 ms. A page full of short matches costs a token each, and an
  LCG-generated `int64` column decodes at 2-5 GB/s where the uncompressed path does 14-40 GB/s. Smaller
  pages help a little (the table above) but do not change the shape of it: this is the one part of
  Parquet that a GPU is structurally bad at, and it is honest to say so.
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

Files it writes are read back byte-identically by ArrowMetal *and* by pyarrow — `ParquetWriterTests` and
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
  `pyarrow.parquet` — and asserts the values, nulls and types are identical: 91 checks over the fixture
  set, plus projection, row-group selection, statistics pushdown, dictionary output, struct leaves and a
  50 M-row round trip behind `ARROWMETAL_PARQUET_BIG=1`.
- `ParquetWriterTests` and the Python writer test close the round trip.

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter "ParquetTests|ParquetWriterTests"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release --filter "ParquetTests|ParquetWriterTests"
swift build -c release --product ArrowMetalC
PYTHONPATH=python python -m pytest python/tests/test_parquet.py -q
ARROWMETAL_PARQUET_BIG=1 PYTHONPATH=python python -m pytest python/tests/test_parquet.py -q -k fifty
```

## Limits

- **Struct and map columns** are not reassembled; read struct leaves by dotted path.
- **Nested lists** (`list<list<T>>`) are not assembled — one level of repetition only.
- **`BIT_PACKED`** (the deprecated level encoding) and **LZO** are rejected.
- **Encrypted files** are not supported.
- **A single column chunk above 4 GiB** is rejected (the file itself has no size limit).
- **Column and offset indexes** in the footer are parsed past but not used: page-level skipping by index
  is not implemented, only row-group skipping by statistics.
- **Bloom filters** are ignored.
- **Statistics pushdown is row-group granular**, and the deprecated `min`/`max` fields are only used when
  `min_value`/`max_value` are absent (they use a signed byte order that is wrong for strings, which is why
  Parquet deprecated them).
- **ZSTD needs libzstd** installed; see above.
- **`PLAIN` `BYTE_ARRAY` pages are walked by one thread each** to find the value boundaries — the format
  gives no other option — so a byte-array column with a handful of very large pages has less parallelism
  than a wide one.
- Arrays are capped at 2^32 elements, as everywhere else in ArrowMetal.
