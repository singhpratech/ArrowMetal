package arrowmetal_test

import (
	"fmt"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/memory"
	am "github.com/singhpratech/ArrowMetal/go/arrowmetal"
)

// chunkTypes are the types the chunked import takes, as arrow-go spells them.
var chunkTypes = []arrow.DataType{
	arrow.PrimitiveTypes.Int8,
	arrow.PrimitiveTypes.Uint16,
	arrow.PrimitiveTypes.Int32,
	arrow.PrimitiveTypes.Int64,
	arrow.PrimitiveTypes.Float32,
	arrow.PrimitiveTypes.Float64,
	arrow.FixedWidthTypes.Boolean,
	arrow.FixedWidthTypes.Date32,
	&arrow.TimestampType{Unit: arrow.Microsecond},
	&arrow.Decimal128Type{Precision: 18, Scale: 2},
	&arrow.FixedSizeBinaryType{ByteWidth: 3},
	arrow.BinaryTypes.String,
	arrow.BinaryTypes.LargeString,
	arrow.BinaryTypes.Binary,
	arrow.BinaryTypes.StringView,
}

// genArray builds n rows of dt with a null every nullEvery rows (0: none; 1: all null).
func genArray(t *testing.T, alloc memory.Allocator, dt arrow.DataType, n, nullEvery int, seed int64) arrow.Array {
	t.Helper()
	b := array.NewBuilder(alloc, dt)
	defer b.Release()
	raw := genInt64Seed(n, seed)
	for i, x := range raw {
		if nullEvery > 0 && i%nullEvery == 0 {
			b.AppendNull()
			continue
		}
		switch bb := b.(type) {
		case *array.Int8Builder:
			bb.Append(int8(x))
		case *array.Uint16Builder:
			bb.Append(uint16(x))
		case *array.Int32Builder:
			bb.Append(int32(x))
		case *array.Int64Builder:
			bb.Append(x)
		case *array.Float32Builder:
			bb.Append(float32(x) / 8)
		case *array.Float64Builder:
			bb.Append(float64(x) / 16)
		case *array.BooleanBuilder:
			bb.Append(x&1 == 1)
		case *array.Date32Builder:
			bb.Append(arrow.Date32(x % 40000))
		case *array.TimestampBuilder:
			bb.Append(arrow.Timestamp(x * 1000))
		case *array.Decimal128Builder:
			bb.Append(decimal128.FromI64(x))
		case *array.FixedSizeBinaryBuilder:
			bb.Append([]byte{byte(x), byte(x >> 8), byte(x >> 16)})
		case *array.StringBuilder:
			bb.Append(fmt.Sprintf("s%d", x%1000))
		case *array.LargeStringBuilder:
			bb.Append(fmt.Sprintf("L%d", x%777))
		case *array.BinaryBuilder:
			bb.Append([]byte(fmt.Sprintf("b%d", x%99)))
		case *array.StringViewBuilder:
			// Both view forms: short strings inline, long ones in a data buffer.
			if x%2 == 0 {
				bb.Append(fmt.Sprintf("v%d", x%50))
			} else {
				bb.Append(fmt.Sprintf("a-longer-string-than-twelve-bytes-%d", x%321))
			}
		default:
			t.Fatalf("genArray: no generator for %s", dt)
		}
	}
	return b.NewArray()
}

type chunkLayout struct {
	name  string
	build func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array
}

// chunkLayouts are the shapes the chunked import has to get right: empty chunks, one-row chunks,
// sliced chunks (offset != 0, bit offsets for validity and boolean), all-null chunks, chunks without
// a validity bitmap, one chunk, and many small chunks.
var chunkLayouts = []chunkLayout{
	{"mixed", func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array {
		big := genArray(t, alloc, dt, 5000, 7, 1)
		defer big.Release()
		return []arrow.Array{
			genArray(t, alloc, dt, 0, 0, 2),    // empty
			genArray(t, alloc, dt, 1, 0, 3),    // one row
			array.NewSlice(big, 3, 1203),       // sliced, odd offset
			genArray(t, alloc, dt, 64, 1, 4),   // all null
			genArray(t, alloc, dt, 1000, 0, 5), // no nulls, no bitmap
			array.NewSlice(big, 4001, 4002),    // one-row slice
			genArray(t, alloc, dt, 0, 0, 6),    // empty again
			array.NewSlice(big, 1999, 5000),    // sliced to the end
			genArray(t, alloc, dt, 1, 1, 7),    // one null row
			genArray(t, alloc, dt, 333, 3, 8),  // nulls
		}
	}},
	{"one-chunk", func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array {
		return []arrow.Array{genArray(t, alloc, dt, 777, 5, 9)}
	}},
	{"all-empty", func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array {
		return []arrow.Array{genArray(t, alloc, dt, 0, 0, 1), genArray(t, alloc, dt, 0, 0, 2)}
	}},
	{"all-null", func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array {
		return []arrow.Array{genArray(t, alloc, dt, 10, 1, 1), genArray(t, alloc, dt, 1, 1, 2),
			genArray(t, alloc, dt, 100, 1, 3)}
	}},
	{"many-small", func(t *testing.T, alloc memory.Allocator, dt arrow.DataType) []arrow.Array {
		out := make([]arrow.Array, 1000)
		for i := range out {
			out[i] = genArray(t, alloc, dt, i%5, 3, int64(i))
		}
		return out
	}},
}

func releaseAll(as []arrow.Array) {
	for _, a := range as {
		a.Release()
	}
}

// TestImportChunksEqualsConcatenation: for every type and layout, ImportChunks exported back equals
// Import of array.Concatenate exported back, and equals the concatenation itself. Built with a
// checked allocator, so every chunk has to be released exactly once by the end.
func TestImportChunksEqualsConcatenation(t *testing.T) {
	requireLib(t)
	for _, dt := range chunkTypes {
		for _, lay := range chunkLayouts {
			t.Run(dt.String()+"/"+lay.name, func(t *testing.T) {
				alloc := memory.NewCheckedAllocator(memory.NewGoAllocator())
				chunks := lay.build(t, alloc, dt)

				got, err := am.ImportChunks(chunks)
				if err != nil {
					t.Fatalf("ImportChunks: %v", err)
				}
				gotA, err := got.Export()
				if err != nil {
					t.Fatal(err)
				}

				merged, err := array.Concatenate(chunks, alloc)
				if err != nil {
					t.Fatal(err)
				}
				ref, err := am.Import(merged)
				if err != nil {
					t.Fatal(err)
				}
				refA, err := ref.Export()
				if err != nil {
					t.Fatal(err)
				}

				if gotA.Len() != merged.Len() || gotA.NullN() != merged.NullN() {
					t.Fatalf("len/nulls %d/%d, want %d/%d", gotA.Len(), gotA.NullN(), merged.Len(), merged.NullN())
				}
				if !array.Equal(gotA, refA) {
					t.Fatalf("ImportChunks differs from Import(Concatenate):\n got %v\nwant %v", gotA, refA)
				}
				if arrow.TypeEqual(gotA.DataType(), merged.DataType()) && !array.Equal(gotA, merged) {
					t.Fatalf("ImportChunks differs from the concatenation:\n got %v\nwant %v", gotA, merged)
				}

				gotA.Release()
				refA.Release()
				got.Release()
				ref.Release()
				merged.Release()
				releaseAll(chunks)
				alloc.AssertSize(t, 0)
			})
		}
	}
}

// TestImportChunksThroughKernels sums, sorts and group-bys a chunked Int64 column and checks the
// answers against the same calls on the imported concatenation.
func TestImportChunksThroughKernels(t *testing.T) {
	requireLib(t)
	dt := arrow.PrimitiveTypes.Int64
	var chunks []arrow.Array
	for i := 0; i < 40; i++ {
		c := genArray(t, mem, dt, 25000+i, 9, int64(i+1))
		if i%4 == 1 {
			s := array.NewSlice(c, 17, int64(c.Len()-3))
			c.Release()
			c = s
		}
		chunks = append(chunks, c)
	}
	defer releaseAll(chunks)
	got, err := am.ImportChunks(chunks)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Release()
	merged, err := array.Concatenate(chunks, mem)
	if err != nil {
		t.Fatal(err)
	}
	defer merged.Release()
	ref := importArr(t, merged)

	gs, _ := got.Sum()
	rs, _ := ref.Sum()
	if gs.Int64() != rs.Int64() || gs.Valid != rs.Valid {
		t.Fatalf("Sum %v, want %v", gs, rs)
	}
	for _, o := range allSortOptions() {
		gi, err := got.ArgsortWith(o)
		if err != nil {
			t.Fatal(err)
		}
		ri, err := ref.ArgsortWith(o)
		if err != nil {
			t.Fatal(err)
		}
		equalIdx(t, "ArgsortWith on chunks", indicesOf(t, gi), indicesOf(t, ri))
	}
}

// TestImportChunkedAndColumn covers the arrow.Chunked and record-batch forms and a source built from
// batches, against the concatenation.
func TestImportChunkedAndColumn(t *testing.T) {
	requireLib(t)
	alloc := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer alloc.AssertSize(t, 0)

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "k", Type: arrow.PrimitiveTypes.Int64, Nullable: true},
		{Name: "s", Type: arrow.BinaryTypes.String, Nullable: true},
	}, nil)
	var batches []arrow.RecordBatch
	for i := 0; i < 12; i++ {
		n := 1000 * i // the first batch is empty
		k := genArray(t, alloc, schema.Field(0).Type, n, 5, int64(i))
		s := genArray(t, alloc, schema.Field(1).Type, n, 3, int64(i+100))
		batches = append(batches, array.NewRecordBatch(schema, []arrow.Array{k, s}, int64(n)))
		k.Release()
		s.Release()
	}
	defer func() {
		for _, b := range batches {
			b.Release()
		}
	}()

	for col := 0; col < 2; col++ {
		h, err := am.ImportColumn(batches, col)
		if err != nil {
			t.Fatal(err)
		}
		chunks := make([]arrow.Array, len(batches))
		for i, b := range batches {
			chunks[i] = b.Column(col)
		}
		ch := arrow.NewChunked(schema.Field(col).Type, chunks)
		h2, err := am.ImportChunked(ch)
		if err != nil {
			t.Fatal(err)
		}
		merged, err := array.Concatenate(chunks, alloc)
		if err != nil {
			t.Fatal(err)
		}
		a1, _ := h.Export()
		a2, _ := h2.Export()
		if !array.Equal(a1, merged) || !array.Equal(a2, merged) {
			t.Fatalf("column %d: chunked import differs from the concatenation", col)
		}
		a1.Release()
		a2.Release()
		h.Release()
		h2.Release()
		merged.Release()
		ch.Release()
	}

	// A Chunked with no chunks is an empty array of its type.
	empty := arrow.NewChunked(arrow.PrimitiveTypes.Float64, nil)
	he, err := am.ImportChunked(empty)
	if err != nil {
		t.Fatal(err)
	}
	if he.Len() != 0 || he.Format() != "g" {
		t.Fatalf("empty Chunked: len %d format %q", he.Len(), he.Format())
	}
	he.Release()
	empty.Release()

	// A source from the batches answers a plan like the concatenation does.
	src, err := am.NewSourceFromBatches("b", batches)
	if err != nil {
		t.Fatal(err)
	}
	res, err := am.RunPlan(`{"op":"aggregate","aggs":[["sum","t","(col \"k\")"],["count","c","(col \"s\")"]],`+
		`"input":{"op":"scan","source":"b"}}`, true, src)
	if err != nil {
		t.Fatal(err)
	}
	var wantSum, wantCount int64
	for _, b := range batches {
		k := b.Column(0).(*array.Int64)
		for i := 0; i < k.Len(); i++ {
			if k.IsValid(i) {
				wantSum += k.Value(i)
			}
		}
		wantCount += int64(b.Column(1).Len() - b.Column(1).NullN())
	}
	sc, _ := res.Column(0)
	cc, _ := res.Column(1)
	s, _ := sc.Sum()
	c, _ := cc.Sum()
	if s.Int64() != wantSum || c.Int64() != wantCount {
		t.Fatalf("plan over batches: sum %v count %v, want %d %d", s, c, wantSum, wantCount)
	}
	sc.Release()
	cc.Release()
	res.Release()
	src.Release()
}

// TestImportChunksFallbackAndErrors: a dictionary type goes through the concatenation and still
// equals it; ChunksSupported says which path a type takes; mixed types and empty input are errors
// that leave every chunk the caller's.
func TestImportChunksFallbackAndErrors(t *testing.T) {
	requireLib(t)
	alloc := memory.NewCheckedAllocator(memory.NewGoAllocator())
	defer alloc.AssertSize(t, 0)

	if ok, err := am.ChunksSupported(arrow.PrimitiveTypes.Int64); err != nil || !ok {
		t.Fatalf("ChunksSupported(int64) = %v, %v; want true", ok, err)
	}
	dictType := &arrow.DictionaryType{IndexType: arrow.PrimitiveTypes.Int32, ValueType: arrow.BinaryTypes.String}
	if ok, err := am.ChunksSupported(dictType); err != nil || ok {
		t.Fatalf("ChunksSupported(dictionary) = %v, %v; want false", ok, err)
	}

	var dchunks []arrow.Array
	for i := 0; i < 3; i++ {
		b := array.NewDictionaryBuilder(alloc, dictType).(*array.BinaryDictionaryBuilder)
		for j := 0; j < 50+i; j++ {
			if j%7 == 0 {
				b.AppendNull()
			} else {
				_ = b.AppendString(fmt.Sprintf("d%d", (j+i)%5))
			}
		}
		dchunks = append(dchunks, b.NewArray())
		b.Release()
	}
	h, err := am.ImportChunks(dchunks)
	if err != nil {
		t.Fatalf("ImportChunks(dictionary): %v", err)
	}
	merged, err := array.Concatenate(dchunks, alloc)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := am.Import(merged)
	if err != nil {
		t.Fatal(err)
	}
	ga, _ := h.Export()
	ra, _ := ref.Export()
	if !array.Equal(ga, ra) {
		t.Fatalf("dictionary chunks: got %v, want %v", ga, ra)
	}
	ga.Release()
	ra.Release()
	h.Release()
	ref.Release()
	merged.Release()
	releaseAll(dchunks)

	i64 := genArray(t, alloc, arrow.PrimitiveTypes.Int64, 10, 0, 1)
	f64 := genArray(t, alloc, arrow.PrimitiveTypes.Float64, 10, 0, 1)
	if _, err := am.ImportChunks([]arrow.Array{i64, f64}); err == nil {
		t.Fatal("ImportChunks with two types did not fail")
	}
	if _, err := am.ImportChunks(nil); err == nil {
		t.Fatal("ImportChunks(nil) did not fail")
	}
	if _, err := am.ImportColumn(nil, 0); err == nil {
		t.Fatal("ImportColumn(nil) did not fail")
	}
	// The chunks are still the caller's, whole, after the refusals.
	if i64.(*array.Int64).Value(3) != genInt64Seed(10, 1)[3] {
		t.Fatal("chunk changed by a refused import")
	}
	i64.Release()
	f64.Release()
}

// TestImportChunksOutlivesChunks: the imported array keeps its values after every chunk is released
// and collected.
func TestImportChunksOutlivesChunks(t *testing.T) {
	requireLib(t)
	dt := arrow.PrimitiveTypes.Int64
	chunks := []arrow.Array{genArray(t, mem, dt, 3000, 4, 1), genArray(t, mem, dt, 5000, 0, 2)}
	merged, _ := array.Concatenate(chunks, mem)
	defer merged.Release()
	h, err := am.ImportChunks(chunks)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Release()
	releaseAll(chunks)
	chunks = nil
	for i := 0; i < 3; i++ {
		churn := make([][]byte, 64)
		for j := range churn {
			churn[j] = make([]byte, 1<<16)
			for k := range churn[j] {
				churn[j][k] = 0xAB
			}
		}
		_ = churn
	}
	out := exportArr(t, h)
	if !array.Equal(out, merged) {
		t.Fatal("imported chunks changed after the chunks were released")
	}
}
