// ArrowMetal underneath ordinary DuckDB SQL: an optimizer extension that moves eligible aggregates
// onto the GPU without the query changing.
//
//   LOAD 'duckdb-extension/build/arrowmetal_rewrite.duckdb_extension';
//   SELECT k, sum(v), count(*), min(v), max(v), avg(v) FROM t WHERE v > 0 GROUP BY k;
//   -- EXPLAIN shows ARROWMETAL_AGGREGATE where DuckDB would have had HASH_GROUP_BY
//
// WHY A C++ EXTENSION. DuckDB's C extension API (duckdb_extension.h, which the table-function
// extension in arrowmetal_extension.cpp uses) has no optimizer hook: its only planner-adjacent entry
// point is the replacement scan. The optimizer hook, `OptimizerExtension`, is C++ only, so this file is
// built against the DuckDB v1.5.5 headers and stamped with the CPP ABI, which DuckDB loads only into
// the exact engine version it was built for. See docs/DUCKDB.md §4b.
//
// WHAT IT DOES. After DuckDB's own optimizers have run, it looks for a LogicalAggregate whose shape it
// can answer exactly - SUM / COUNT / COUNT(*) / MIN / MAX / AVG over integer columns (MIN and MAX also
// over DATE and TIMESTAMP), no DISTINCT, no FILTER, no ORDER BY inside the aggregate, at most one GROUP
// BY column that is an integer, DATE, TIMESTAMP or VARCHAR column - sitting on a chain of projections
// and filters over a table function whose row count DuckDB knows. In 'auto' mode, when DuckDB's
// estimate of the rows reaching the aggregate is at or above both the router's crossover
// (Benchmarks/results/router_2026-09-17.json) and the measured floor of the query's shape class
// (Benchmarks/results/duckdb_rewrite_2026-09-23_provisional.csv), the LogicalAggregate is replaced by
// ARROWMETAL_AGGREGATE. Everything below the aggregate - the scan, the pushed-down filters, the
// projections - is still DuckDB's, planned and run exactly as before.
//
// ARROWMETAL_AGGREGATE is a parallel sink. Each DuckDB worker thread reserves row positions for its
// chunk and copies the aggregate's input columns into page-aligned buffers outside any lock. A
// fixed-width column's buffer is a slab from a pool kept across queries, imported into ArrowMetal once
// and handed over as a zero-copy slice. An ungrouped aggregate, or a group-by over a narrow integer
// key, is cut into blocks that a GPU worker thread aggregates while DuckDB is still scanning, and the
// per-block results are merged on the host; everything else runs once over the whole input when the
// pipeline finishes. The result goes back to DuckDB with the aggregate's own column bindings and types.
//
// EXACTNESS. Integer sums are exact to DuckDB's HUGEINT: a BIGINT column is summed as its high and low
// 32-bit halves (each of which fits in 64 bits over fewer than 2^31 rows) and recombined in 128 bits,
// unless DuckDB's own statistics already proved the sum fits in 64 bits (sum_no_overflow). AVG is
// finished with the same arithmetic DuckDB's avg uses. Floating-point SUM and AVG are not rewritten:
// DuckDB's own float sums depend on thread scheduling, so there is no single answer to match.
//
// CONTROLS.
//   SET arrowmetal_rewrite = 'auto';   -- default: supported shapes, at the sizes measured faster
//   SET arrowmetal_rewrite = 'off';    -- never rewrite
//   SET arrowmetal_rewrite = 'force';  -- rewrite every supported shape regardless of size (for tests)
//   SET arrowmetal_rewrite_block_rows = 16777216;  -- rows per block of a streamed plan
//   SELECT * FROM arrowmetal_rewrites();  -- every decision this process made, oldest first, with reason
//   EXPLAIN ...                           -- ARROWMETAL_AGGREGATE in the plan means the query was rewritten

#include "duckdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include "arrowmetal.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>

namespace duckdb {
namespace arrowmetal_rewrite {

//===--------------------------------------------------------------------===//
// Crossovers, from Benchmarks/results/router_2026-09-17.json ("vs_fastest_library"). The row count
// at which ArrowMetal's kernel first beats the fastest CPU library on the same operation. A query is
// rewritten in 'auto' mode only when DuckDB's estimate of the rows reaching the aggregate is at or
// above the largest crossover among its aggregates. python/tests/test_duckdb_rewrite.py checks these
// constants against the JSON, so they cannot drift from it silently.
//===--------------------------------------------------------------------===//
struct Crossovers {
	// "reductions: sum(int64, 10% nulls)", "min(...)", "max(...)", "mean(...)"
	static constexpr int64_t SUM = 10000000;
	static constexpr int64_t MIN = 50000000;
	static constexpr int64_t MAX = 10000000;
	static constexpr int64_t MEAN = 1000000;
	// "group-by: {sum,count,min,max,mean} by int32 key (1000 groups)" - all five are 10M
	static constexpr int64_t GROUP_1K = 10000000;
	// "group-by: {sum,count,min,max,mean} by int32 key (100000 groups)" - all five are 100k
	static constexpr int64_t GROUP_100K = 100000;
	// "group-by: sum by utf8 key (1000 distinct)" and "(100000 distinct)" - both 10M
	static constexpr int64_t GROUP_UTF8 = 10000000;
};

//===--------------------------------------------------------------------===//
// Measured floors. The router's crossovers compare kernels on data already in GPU memory; here the
// rows first have to be gathered out of DuckDB's scan, and DuckDB's own aggregate is fast. So 'auto'
// also requires the query's shape class to have been measured faster than DuckDB's operators, from
// the row count below, in Benchmarks/duckdb_rewrite_bench.py
// (Benchmarks/results/duckdb_rewrite_2026-09-23_provisional.csv). A class with no floor here was not
// measured faster at 1M, 10M or 50M rows and is never rewritten in 'auto'. The effective threshold is
// the larger of the two. These are from a provisional run on a shared machine; they are revisited
// with the quiet rerun.
//===--------------------------------------------------------------------===//
struct Measured {
	// Ungrouped, with two or more of SUM/MIN/MAX/AVG, or one DuckDB keeps in a 128-bit state.
	static constexpr int64_t UNGROUPED = 10000000;
	// Integer key within the fused group-by's range, DuckDB estimating MANY_GROUPS groups or more.
	static constexpr int64_t DENSE_MANY_GROUPS = 10000000;
	// Integer key within the fused group-by's range, fewer groups, three or more of SUM/MIN/MAX/AVG.
	static constexpr int64_t DENSE_FEW_GROUPS = 50000000;
	// Integer key over a wider range (ArrowMetal's hash group-by), MANY_GROUPS groups or more.
	static constexpr int64_t HASH_MANY_GROUPS = 50000000;
};

// Where the router's 1,000-group and 100,000-group classes meet (see Analyse).
static constexpr int64_t MANY_GROUPS = 10000;

// The fused group-by keeps one slot per key value; above this span the hash group-by is used instead.
// At 10M rows a 1M-key fused group-by ran in about half the hash group-by's time.
static constexpr int64_t DENSE_KEY_SPAN = int64_t(1) << 20;
// A group-by is streamed only when its key range is at most this, which keeps the host-side merge of
// the per-block partials to a direct-indexed array.
static constexpr int64_t STREAM_KEY_SPAN = 65536;
// Rows per block of a streamed plan, unless SET arrowmetal_rewrite_block_rows says otherwise.
static constexpr int64_t DEFAULT_BLOCK_ROWS = int64_t(1) << 24;
// Slots the fused group-by keeps in threadgroup memory (ExprCompiler.gbMaxPrivateSlots).
static constexpr int64_t FUSED_PRIVATE_SLOTS = 2048;
// The GPU kernels index rows with 32-bit integers, and a 32-bit value summed over fewer than 2^31
// rows cannot leave int64. A plan that is not streamed and would see more rows is left to DuckDB.
static constexpr int64_t MAX_ROWS = (int64_t(1) << 31) - 1;

enum class Mode { AUTO, OFF, FORCE };

static Mode GetMode(ClientContext &context) {
	Value value;
	if (context.TryGetCurrentSetting("arrowmetal_rewrite", value) && !value.IsNull()) {
		auto text = StringUtil::Lower(value.ToString());
		if (text == "off" || text == "false" || text == "0") {
			return Mode::OFF;
		}
		if (text == "force") {
			return Mode::FORCE;
		}
	}
	return Mode::AUTO;
}

//===--------------------------------------------------------------------===//
// The decision log behind arrowmetal_rewrites()
//===--------------------------------------------------------------------===//
struct Decision {
	int64_t id = 0;
	string decision; // "rewritten" or "kept"
	string reason;
	string shape;
	int64_t input_rows = -1;
	int64_t threshold_rows = -1;
	// Filled in when a rewritten plan runs.
	string path;
	int64_t rows_seen = -1;
	int64_t groups = -1;
	double gpu_ms = -1;
};

static std::mutex g_log_lock;
static std::deque<Decision> g_log;
static int64_t g_next_id = 1;
static constexpr size_t LOG_CAPACITY = 1024;

static int64_t Record(Decision decision) {
	std::lock_guard<std::mutex> guard(g_log_lock);
	decision.id = g_next_id++;
	g_log.push_back(decision);
	while (g_log.size() > LOG_CAPACITY) {
		g_log.pop_front();
	}
	return decision.id;
}

static void RecordRun(int64_t id, const string &path, int64_t rows, int64_t groups, double gpu_ms) {
	std::lock_guard<std::mutex> guard(g_log_lock);
	for (auto it = g_log.rbegin(); it != g_log.rend(); ++it) {
		if (it->id == id) {
			it->path = path;
			it->rows_seen = rows;
			it->groups = groups;
			it->gpu_ms = gpu_ms;
			return;
		}
	}
}

// One GPU at a time: ArrowMetal serialises on the device anyway, and holding this keeps two queries'
// command buffers from interleaving.
static std::recursive_mutex g_gpu_lock;

// ARROWMETAL_REWRITE_TRACE=1 prints the time since `start` at each Finalize step to stderr.
static void Trace(const char *step, std::chrono::steady_clock::time_point start) {
	static const bool enabled = getenv("ARROWMETAL_REWRITE_TRACE") != nullptr;
	if (enabled) {
		fprintf(stderr, "arrowmetal_rewrite: %-8s %8.3f ms\n", step,
		        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	}
}


//===--------------------------------------------------------------------===//
// Column kinds the sink gathers
//===--------------------------------------------------------------------===//
enum class Kind : uint8_t { I8, I16, I32, I64, U8, U16, U32, U64, STR };

static idx_t KindWidth(Kind kind) {
	switch (kind) {
	case Kind::I8:
	case Kind::U8:
		return 1;
	case Kind::I16:
	case Kind::U16:
		return 2;
	case Kind::I32:
	case Kind::U32:
		return 4;
	case Kind::I64:
	case Kind::U64:
		return 8;
	default:
		return 0;
	}
}

static bool KindSigned(Kind kind) {
	return kind == Kind::I8 || kind == Kind::I16 || kind == Kind::I32 || kind == Kind::I64;
}

static const char *KindFormat(Kind kind) {
	switch (kind) {
	case Kind::I8: return "c";
	case Kind::I16: return "s";
	case Kind::I32: return "i";
	case Kind::I64: return "l";
	case Kind::U8: return "C";
	case Kind::U16: return "S";
	case Kind::U32: return "I";
	case Kind::U64: return "L";
	case Kind::STR: return "U";
	}
	return "";
}

// The numeric types whose values the GPU aggregates exactly. DATE and TIMESTAMP travel as their
// storage integers: only MIN and MAX accept them (SQL has no SUM of a date).
static bool NumericKind(const LogicalType &type, Kind &kind) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT: kind = Kind::I8; return true;
	case LogicalTypeId::SMALLINT: kind = Kind::I16; return true;
	case LogicalTypeId::INTEGER: kind = Kind::I32; return true;
	case LogicalTypeId::BIGINT: kind = Kind::I64; return true;
	case LogicalTypeId::UTINYINT: kind = Kind::U8; return true;
	case LogicalTypeId::USMALLINT: kind = Kind::U16; return true;
	case LogicalTypeId::UINTEGER: kind = Kind::U32; return true;
	case LogicalTypeId::UBIGINT: kind = Kind::U64; return true;
	case LogicalTypeId::DATE: kind = Kind::I32; return true;
	case LogicalTypeId::TIMESTAMP: kind = Kind::I64; return true;
	default: return false;
	}
}

static bool IsTemporal(const LogicalType &type) {
	return type.id() == LogicalTypeId::DATE || type.id() == LogicalTypeId::TIMESTAMP;
}

// A cast that keeps every value of its source: integer to a wider integer that holds its whole range.
static bool ValuePreservingCast(const LogicalType &source, const LogicalType &target) {
	Kind s, t;
	if (IsTemporal(source) || IsTemporal(target) || !NumericKind(source, s) || !NumericKind(target, t)) {
		return false;
	}
	const idx_t sw = KindWidth(s), tw = KindWidth(t);
	if (KindSigned(s) == KindSigned(t)) {
		return tw >= sw;
	}
	if (!KindSigned(s) && KindSigned(t)) {
		return tw > sw;
	}
	return false; // signed into unsigned loses the negatives
}

//===--------------------------------------------------------------------===//
// What the rewritten operator computes
//===--------------------------------------------------------------------===//
enum class AggOp : uint8_t { COUNT_STAR, COUNT, SUM, MIN, MAX, AVG };

struct AggSpec {
	AggOp op = AggOp::COUNT_STAR;
	idx_t input = DConstants::INVALID_INDEX; // index into Spec::inputs
	// The integer width DuckDB's own aggregate works at (the function's argument type): 16, 32 or 64.
	idx_t semantic_bits = 64;
	// sum_no_overflow: DuckDB's statistics proved the total fits in 64 bits.
	bool proven_no_overflow = false;
	LogicalType return_type;
	string name;
};

struct Spec {
	vector<LogicalType> input_types; // the gathered columns, as the child produces them
	vector<Kind> input_kinds;
	bool has_key = false; // input 0 is the group key when set
	vector<AggSpec> aggs;
	vector<LogicalType> result_types; // key (if any), then one per aggregate
	int64_t decision_id = 0;
	// Streamed plans aggregate block by block while DuckDB scans (see "Blocks").
	bool stream = false;
	idx_t block_rows = 0;
};

//===--------------------------------------------------------------------===//
// Page-aligned, lazily committed memory. mmap hands back pages Metal can wrap without copying, and
// untouched pages cost nothing, so a region can be sized for the largest possible input.
//===--------------------------------------------------------------------===//
static constexpr size_t PAGE = 16384; // Apple silicon's VM page, what makeBuffer(bytesNoCopy:) wants

static size_t PageRound(size_t n) {
	return (n + PAGE - 1) / PAGE * PAGE;
}

static uint8_t *MapBytes(size_t bytes) {
	void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (p == MAP_FAILED) {
		throw OutOfMemoryException("arrowmetal_rewrite: could not map %llu bytes", (unsigned long long)bytes);
	}
	return static_cast<uint8_t *>(p);
}

// A plain mapping, for the string key's buffers.
struct Region {
	uint8_t *ptr = nullptr;
	size_t bytes = 0;

	Region() = default;
	Region(const Region &) = delete;
	Region &operator=(const Region &) = delete;
	~Region() {
		Free();
	}
	void Allocate(size_t size) {
		Free();
		bytes = PageRound(size < PAGE ? PAGE : size);
		ptr = MapBytes(bytes);
	}
	void Free() {
		if (ptr) {
			munmap(ptr, bytes);
			ptr = nullptr;
			bytes = 0;
		}
	}
};

//===--------------------------------------------------------------------===//
// Slabs: a fixed-width column's memory, kept from one query to the next together with its Metal
// wrapping.
//
// Wrapping fresh memory for the GPU is not free: makeBuffer(bytesNoCopy:) and the GPU's first touch of
// a new mapping take time in proportion to its size, and a fresh mapping's first CPU writes are page
// faults. So each slab is imported into ArrowMetal once, over its whole capacity, and every query
// afterwards writes its rows into the same memory and hands ArrowMetal a zero-copy slice of them.
// Freed slabs wait in a pool (least recently used first out) holding at most SLAB_POOL_BYTES.
//===--------------------------------------------------------------------===//
static constexpr size_t SLAB_POOL_BYTES = size_t(4) << 30;

struct Slab {
	Kind kind = Kind::I64;
	idx_t capacity = 0;
	uint8_t *base = nullptr;
	size_t bytes = 0;
	uint64_t *validity = nullptr; // Arrow bitmap, all-valid between uses
	uint8_t *values = nullptr;
	bool bitmap_dirty = false;       // the last user cleared some bits
	am_array *with_validity = nullptr;    // imported once, on first use
	am_array *without_validity = nullptr;

	Slab(Kind kind_p, idx_t capacity_p) : kind(kind_p), capacity(capacity_p) {
		const size_t bitmap_bytes = PageRound(((capacity + 64) / 64) * 8);
		bytes = bitmap_bytes + PageRound(capacity * KindWidth(kind) + 64);
		base = MapBytes(bytes);
		validity = reinterpret_cast<uint64_t *>(base);
		values = base + bitmap_bytes;
		std::memset(validity, 0xFF, bitmap_bytes);
	}
	~Slab() {
		if (with_validity) {
			am_release(with_validity);
		}
		if (without_validity) {
			am_release(without_validity);
		}
		munmap(base, bytes);
	}
	void ResetBitmap() {
		if (bitmap_dirty) {
			std::memset(validity, 0xFF, ((capacity + 64) / 64) * 8);
			bitmap_dirty = false;
		}
	}
};

static std::mutex g_slab_lock;
// Deliberately never destroyed: releasing Metal-backed arrays from a static destructor at process exit
// would race ArrowMetal's own teardown.
static auto *g_slabs = new std::deque<unique_ptr<Slab>>();
static size_t g_slab_bytes = 0;

static unique_ptr<Slab> TakeSlab(Kind kind, idx_t rows) {
	{
		std::lock_guard<std::mutex> guard(g_slab_lock);
		idx_t best = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < g_slabs->size(); i++) {
			auto &slab = *(*g_slabs)[i];
			if (slab.kind == kind && slab.capacity >= rows && slab.capacity / 2 <= rows &&
			    (best == DConstants::INVALID_INDEX || slab.capacity < (*g_slabs)[best]->capacity)) {
				best = i;
			}
		}
		if (best != DConstants::INVALID_INDEX) {
			auto slab = std::move((*g_slabs)[best]);
			g_slabs->erase(g_slabs->begin() + int64_t(best));
			g_slab_bytes -= slab->bytes;
			slab->ResetBitmap();
			return slab;
		}
	}
	return make_uniq<Slab>(kind, rows);
}

static void ReturnSlab(unique_ptr<Slab> slab) {
	vector<unique_ptr<Slab>> evicted;
	{
		std::lock_guard<std::mutex> guard(g_slab_lock);
		if (slab->bytes > SLAB_POOL_BYTES) {
			evicted.push_back(std::move(slab));
		} else {
			g_slab_bytes += slab->bytes;
			g_slabs->push_back(std::move(slab));
			while (g_slab_bytes > SLAB_POOL_BYTES) {
				g_slab_bytes -= g_slabs->front()->bytes;
				evicted.push_back(std::move(g_slabs->front()));
				g_slabs->pop_front();
			}
		}
	}
	// Releasing an evicted slab's arrays is an ArrowMetal call, so it happens under the GPU lock (and not
	// under the pool's).
	if (!evicted.empty()) {
		std::lock_guard<std::recursive_mutex> gpu(g_gpu_lock);
		evicted.clear();
	}
}

static inline void ClearBit(uint64_t *bitmap, idx_t row) {
	__atomic_fetch_and(&bitmap[row >> 6], ~(uint64_t(1) << (row & 63)), __ATOMIC_RELAXED);
}

// Virtual bytes reserved per row for a string column. Untouched pages are never committed, so this is
// address space, not memory; a table whose strings average more than this goes through the overflow.
static constexpr size_t STRING_BYTES_PER_ROW = 64;

//===--------------------------------------------------------------------===//
// Blocks: where the gathered rows live
//
// Every gathered input is stored in blocks of `block_rows` rows, one buffer per input: a Slab for a
// fixed-width column, plain mappings for the string key. Each chunk reserves a run of global row
// positions, and row r lives in block r / block_rows at offset r % block_rows.
//
// A streamed plan (an ungrouped aggregate, or a group-by whose integer key DuckDB's statistics put in
// a small range) uses blocks of arrowmetal_rewrite_block_rows rows and hands each block to a GPU
// worker thread the moment its last row lands, so the GPU aggregates while DuckDB is still scanning;
// Finalize then only has the last, partly filled block left, and merges the per-block partial
// results on the host. Every other plan (a string key, a wide key range, 64-bit MIN/MAX under a
// GROUP BY) needs all rows in one array for ArrowMetal's hash group-by, so it uses a single block sized
// for the whole input.
//===--------------------------------------------------------------------===//
struct BlockColumn {
	Kind kind = Kind::I64;
	unique_ptr<Slab> slab; // fixed width
	Region data;           // STR: the UTF-8 bytes
	Region offsets;        // STR: int64 offsets, rows + 1 of them
	Region bits;           // STR: the validity bitmap
	// Where the gather writes. The bitmap starts all-valid; a NULL clears its bit.
	uint64_t *validity = nullptr;
	uint8_t *values = nullptr;
	int64_t *offsets_ptr = nullptr;
	std::atomic<int64_t> nulls {0};
	size_t byte_capacity = 0; // STR only
	size_t bytes_used = 0;    // STR only, guarded by the reservation lock
	idx_t capacity = 0;       // rows

	BlockColumn(Kind kind_p, idx_t rows, size_t bytes) : kind(kind_p) {
		if (kind == Kind::STR) {
			const size_t bitmap_bytes = ((rows + 64) / 64) * 8;
			bits.Allocate(bitmap_bytes);
			std::memset(bits.ptr, 0xFF, bitmap_bytes);
			offsets.Allocate((rows + 1) * sizeof(int64_t));
			data.Allocate(bytes);
			byte_capacity = data.bytes;
			validity = reinterpret_cast<uint64_t *>(bits.ptr);
			values = data.ptr;
			offsets_ptr = reinterpret_cast<int64_t *>(offsets.ptr);
			capacity = rows;
		} else {
			slab = TakeSlab(kind, rows);
			validity = slab->validity;
			values = slab->values;
			capacity = slab->capacity;
		}
	}
	~BlockColumn() {
		if (slab) {
			slab->bitmap_dirty = nulls.load() > 0;
			ReturnSlab(std::move(slab));
		}
	}
};

struct Block {
	idx_t index = 0;
	idx_t capacity = 0; // rows
	vector<unique_ptr<BlockColumn>> columns;
	std::atomic<idx_t> filled {0};
	// The integer key's range over the block's valid keys (min > max while none has been seen).
	std::atomic<int64_t> key_min {NumericLimits<int64_t>::Maximum()};
	std::atomic<int64_t> key_max {NumericLimits<int64_t>::Minimum()};
	bool processed = false; // touched only by the worker, then by Finalize after the worker has stopped

	// `exact`: the block holds exactly `rows` rows (a streamed plan's blocks). Otherwise it may hold as
	// many as the smallest of its buffers does, since a pooled slab can be larger than asked for.
	Block(const vector<Kind> &kinds, idx_t index_p, idx_t rows, bool exact) : index(index_p), capacity(rows) {
		for (auto kind : kinds) {
			columns.push_back(make_uniq<BlockColumn>(kind, rows, rows * STRING_BYTES_PER_ROW + (1 << 20)));
		}
		if (!exact && !columns.empty()) {
			capacity = columns[0]->capacity;
			for (auto &column : columns) {
				capacity = MinValue<idx_t>(capacity, column->capacity);
			}
		}
	}

	void NoteKeys(int64_t lo, int64_t hi) {
		int64_t current = key_min.load(std::memory_order_relaxed);
		while (lo < current && !key_min.compare_exchange_weak(current, lo, std::memory_order_relaxed)) {
		}
		current = key_max.load(std::memory_order_relaxed);
		while (hi > current && !key_max.compare_exchange_weak(current, hi, std::memory_order_relaxed)) {
		}
	}
	bool HasKeys() const {
		return key_min.load() <= key_max.load();
	}
};

struct StreamAccumulator;

class RewriteGlobalState : public GlobalSinkState {
public:
	RewriteGlobalState(ClientContext &context, const Spec &spec_p, idx_t capacity_rows);
	~RewriteGlobalState() override;

	// The block holding global row positions [b * block_rows, (b + 1) * block_rows), created on first use.
	Block &GetBlock(idx_t b) {
		Block *block = directory[b].load(std::memory_order_acquire);
		if (block) {
			return *block;
		}
		std::lock_guard<std::mutex> guard(alloc_lock);
		block = directory[b].load(std::memory_order_acquire);
		if (!block) {
			blocks.push_back(make_uniq<Block>(spec.input_kinds, b, block_rows, true));
			block = blocks.back().get();
			directory[b].store(block, std::memory_order_release);
		}
		return *block;
	}

	// Rows [0, RegionRows()) of the blocks hold data; anything else is in `overflow`.
	idx_t RegionRows() const {
		if (has_strings) {
			return locked_rows;
		}
		const idx_t limit = max_blocks * block_rows;
		const idx_t total = reserved.load();
		if (total <= limit) {
			return total;
		}
		const idx_t hole = hole_start.load();
		return hole < limit ? hole : limit;
	}

	void Enqueue(Block &block);
	void StopWorker();
	void WorkerLoop();

	const Spec &spec;
	bool stream = false;
	bool has_strings = false;
	idx_t block_rows = 0;
	idx_t max_blocks = 0;
	unique_ptr<std::atomic<Block *>[]> directory;
	std::mutex alloc_lock;
	vector<unique_ptr<Block>> blocks;

	std::atomic<idx_t> reserved {0};                           // fixed-width plans
	std::atomic<idx_t> hole_start {DConstants::INVALID_INDEX}; // a reservation that ran past the blocks
	std::mutex reserve_lock;                                   // string plans
	idx_t locked_rows = 0;                                     // string plans, guarded by reserve_lock

	std::mutex overflow_lock;
	vector<LogicalType> overflow_types;
	unique_ptr<ColumnDataCollection> overflow;

	// The GPU worker of a streamed plan.
	std::mutex queue_lock;
	std::condition_variable queue_cv;
	std::deque<Block *> queue;
	bool closing = false;
	bool worker_started = false;
	std::thread worker;
	string worker_error;
	idx_t streamed_blocks = 0; // blocks the worker finished
	unique_ptr<StreamAccumulator> accumulator;

	ColumnDataCollection result;
	int64_t groups = 0;
};

class RewriteSourceState : public GlobalSourceState {
public:
	ColumnDataScanState scan;
	bool initialized = false;
};

//===--------------------------------------------------------------------===//
// Gathering one chunk
//===--------------------------------------------------------------------===//
template <class T>
static void GatherFixed(const UnifiedVectorFormat &format, idx_t first, idx_t count, T *target) {
	auto source = reinterpret_cast<const T *>(format.data); // raw bytes: the width is all that matters here
	if (!format.sel->IsSet()) {
		std::memcpy(target, source + first, count * sizeof(T));
		return;
	}
	for (idx_t i = 0; i < count; i++) {
		target[i] = source[format.sel->get_index(first + i)];
	}
}

// The valid keys' range among rows [first, first + count) of the chunk; false when all are NULL.
template <class T>
static bool KeyRangeOf(const UnifiedVectorFormat &format, idx_t first, idx_t count, int64_t &lo, int64_t &hi) {
	auto source = reinterpret_cast<const T *>(format.data);
	bool seen = false;
	const bool all_valid = format.validity.AllValid();
	for (idx_t i = first; i < first + count; i++) {
		const idx_t index = format.sel->get_index(i);
		if (!all_valid && !format.validity.RowIsValid(index)) {
			continue;
		}
		const int64_t v = static_cast<int64_t>(source[index]);
		if (!seen) {
			lo = hi = v;
			seen = true;
		} else {
			lo = v < lo ? v : lo;
			hi = v > hi ? v : hi;
		}
	}
	return seen;
}

static void NoteKeyRange(Block &block, Kind kind, const UnifiedVectorFormat &format, idx_t first, idx_t count) {
	int64_t lo = 0, hi = 0;
	bool seen = false;
	switch (kind) {
	case Kind::I8: seen = KeyRangeOf<int8_t>(format, first, count, lo, hi); break;
	case Kind::I16: seen = KeyRangeOf<int16_t>(format, first, count, lo, hi); break;
	case Kind::I32: seen = KeyRangeOf<int32_t>(format, first, count, lo, hi); break;
	case Kind::I64: seen = KeyRangeOf<int64_t>(format, first, count, lo, hi); break;
	case Kind::U8: seen = KeyRangeOf<uint8_t>(format, first, count, lo, hi); break;
	case Kind::U16: seen = KeyRangeOf<uint16_t>(format, first, count, lo, hi); break;
	case Kind::U32: seen = KeyRangeOf<uint32_t>(format, first, count, lo, hi); break;
	default: return; // U64 and strings never take the fused group-by
	}
	if (seen) {
		block.NoteKeys(lo, hi);
	}
}

// Copies rows [first, first + count) of the chunk into `column` at rows [row0, row0 + count); a string
// column's bytes go at byte0.
static void CopyRows(BlockColumn &column, const UnifiedVectorFormat &format, idx_t first, idx_t count, idx_t row0,
                     size_t byte0) {
	if (!format.validity.AllValid()) {
		int64_t nulls = 0;
		for (idx_t i = 0; i < count; i++) {
			if (!format.validity.RowIsValid(format.sel->get_index(first + i))) {
				ClearBit(column.validity, row0 + i);
				nulls++;
			}
		}
		if (nulls) {
			column.nulls.fetch_add(nulls, std::memory_order_relaxed);
		}
	}
	switch (column.kind) {
	case Kind::I8:
	case Kind::U8:
		GatherFixed<uint8_t>(format, first, count, column.values + row0);
		break;
	case Kind::I16:
	case Kind::U16:
		GatherFixed<uint16_t>(format, first, count, reinterpret_cast<uint16_t *>(column.values) + row0);
		break;
	case Kind::I32:
	case Kind::U32:
		GatherFixed<uint32_t>(format, first, count, reinterpret_cast<uint32_t *>(column.values) + row0);
		break;
	case Kind::I64:
	case Kind::U64:
		GatherFixed<uint64_t>(format, first, count, reinterpret_cast<uint64_t *>(column.values) + row0);
		break;
	case Kind::STR: {
		auto strings = UnifiedVectorFormat::GetData<string_t>(format);
		size_t at = byte0;
		for (idx_t i = 0; i < count; i++) {
			column.offsets_ptr[row0 + i] = static_cast<int64_t>(at);
			const idx_t index = format.sel->get_index(first + i);
			if (format.validity.RowIsValid(index)) {
				const auto &s = strings[index];
				const auto size = s.GetSize();
				std::memcpy(column.values + at, s.GetData(), size);
				at += size;
			}
		}
		break;
	}
	}
}

static size_t StringBytes(const UnifiedVectorFormat &format, idx_t count) {
	size_t bytes = 0;
	auto strings = UnifiedVectorFormat::GetData<string_t>(format);
	for (idx_t i = 0; i < count; i++) {
		const idx_t index = format.sel->get_index(i);
		if (format.validity.RowIsValid(index)) {
			bytes += strings[index].GetSize();
		}
	}
	return bytes;
}

//===--------------------------------------------------------------------===//
// The physical operator
//===--------------------------------------------------------------------===//
class PhysicalArrowMetalAggregate : public PhysicalOperator {
public:
	PhysicalArrowMetalAggregate(PhysicalPlan &physical_plan, Spec spec_p, vector<idx_t> child_columns_p,
	                            idx_t capacity_rows_p, idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, spec_p.result_types, estimated_cardinality),
	      spec(std::move(spec_p)), child_columns(std::move(child_columns_p)), capacity_rows(capacity_rows_p) {
	}

	Spec spec;
	vector<idx_t> child_columns; // which column of the child's chunk feeds each gathered input
	idx_t capacity_rows;

	string GetName() const override {
		return "ARROWMETAL_AGGREGATE";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		if (spec.has_key) {
			result["Groups"] = "#" + to_string(child_columns[0]);
		}
		string aggs;
		for (idx_t i = 0; i < spec.aggs.size(); i++) {
			aggs += (i ? "\n" : "") + spec.aggs[i].name;
		}
		result["Aggregates"] = aggs;
		result["Engine"] = spec.stream ? "ArrowMetal (Metal GPU), streamed in blocks" : "ArrowMetal (Metal GPU)";
		SetEstimatedCardinality(result, estimated_cardinality);
		return result;
	}

	// Sink interface
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return false;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<RewriteGlobalState>(context, spec, capacity_rows);
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &gstate = input.global_state.Cast<RewriteGlobalState>();
		const idx_t count = chunk.size();
		if (count == 0) {
			return SinkResultType::NEED_MORE_INPUT;
		}
		const idx_t ncols = child_columns.size();
		vector<UnifiedVectorFormat> formats(ncols);
		for (idx_t c = 0; c < ncols; c++) {
			chunk.data[child_columns[c]].ToUnifiedFormat(count, formats[c]);
		}

		if (gstate.has_strings) {
			// Rows and bytes are reserved together, so the offsets stay in row order.
			const size_t string_bytes = StringBytes(formats[0], count);
			Block &block = *gstate.blocks[0];
			auto &strings = *block.columns[0];
			idx_t row0 = 0;
			size_t byte0 = 0;
			bool fits;
			{
				std::lock_guard<std::mutex> guard(gstate.reserve_lock);
				fits = gstate.locked_rows + count <= block.capacity &&
				       strings.bytes_used + string_bytes <= strings.byte_capacity;
				if (fits) {
					row0 = gstate.locked_rows;
					byte0 = strings.bytes_used;
					gstate.locked_rows += count;
					strings.bytes_used += string_bytes;
				}
			}
			if (fits) {
				for (idx_t c = 0; c < ncols; c++) {
					CopyRows(*block.columns[c], formats[c], 0, count, row0, byte0);
				}
				return SinkResultType::NEED_MORE_INPUT;
			}
			return KeepAside(context, gstate, chunk);
		}

		const idx_t row0 = gstate.reserved.fetch_add(count, std::memory_order_relaxed);
		const idx_t limit = gstate.max_blocks * gstate.block_rows;
		if (row0 + count > limit) {
			if (row0 < limit) {
				gstate.hole_start.store(row0);
			}
			return KeepAside(context, gstate, chunk);
		}
		const bool keyed = spec.has_key;
		for (idx_t done = 0; done < count;) {
			const idx_t row = row0 + done;
			Block &block = gstate.GetBlock(row / gstate.block_rows);
			const idx_t at = row % gstate.block_rows;
			const idx_t n = MinValue<idx_t>(count - done, gstate.block_rows - at);
			for (idx_t c = 0; c < ncols; c++) {
				CopyRows(*block.columns[c], formats[c], done, n, at, 0);
			}
			if (keyed) {
				NoteKeyRange(block, spec.input_kinds[0], formats[0], done, n);
			}
			// The thread that writes a block's last row hands the block to the GPU.
			if (block.filled.fetch_add(n, std::memory_order_acq_rel) + n == gstate.block_rows && gstate.stream) {
				gstate.Enqueue(block);
			}
			done += n;
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	// A chunk past the blocks (a prepared statement run after the table grew, say): its gathered columns
	// wait in a ColumnDataCollection for Finalize.
	SinkResultType KeepAside(ExecutionContext &context, RewriteGlobalState &gstate, DataChunk &chunk) const {
		DataChunk kept;
		kept.InitializeEmpty(gstate.overflow_types);
		for (idx_t c = 0; c < child_columns.size(); c++) {
			kept.data[c].Reference(chunk.data[child_columns[c]]);
		}
		kept.SetCardinality(chunk.size());
		std::lock_guard<std::mutex> guard(gstate.overflow_lock);
		if (!gstate.overflow) {
			gstate.overflow = make_uniq<ColumnDataCollection>(context.client, gstate.overflow_types);
		}
		gstate.overflow->Append(kept);
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override {
		return SinkCombineResultType::FINISHED;
	}

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	// Source interface
	bool IsSource() const override {
		return true;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<RewriteSourceState>();
	}

protected:
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		auto &gstate = sink_state->Cast<RewriteGlobalState>();
		auto &state = input.global_state.Cast<RewriteSourceState>();
		if (!state.initialized) {
			gstate.result.InitializeScan(state.scan);
			state.initialized = true;
		}
		gstate.result.Scan(state.scan, chunk);
		return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
	}
};

//===--------------------------------------------------------------------===//
// The GPU half: Arrow arrays over the gathered buffers, and the ArrowMetal calls
//===--------------------------------------------------------------------===//
struct ArrowHolder {
	const void *buffers[3] = {nullptr, nullptr, nullptr};
};

static void ReleaseArray(ArrowArray *array) {
	delete static_cast<ArrowHolder *>(array->private_data);
	array->release = nullptr;
}

static void ReleaseSchema(ArrowSchema *schema) {
	schema->release = nullptr;
}

static string AmError() {
	const char *msg = am_last_error();
	return msg && *msg ? string(msg) : string("unknown ArrowMetal error");
}

[[noreturn]] static void Fail(const string &what) {
	throw InvalidInputException("arrowmetal_rewrite: %s: %s (SET arrowmetal_rewrite = 'off' runs the query on "
	                            "DuckDB's own operators)",
	                            what, AmError());
}

// Owns an am_array handle.
struct Handle {
	am_array *ptr = nullptr;
	Handle() = default;
	Handle(const Handle &) = delete;
	Handle &operator=(const Handle &) = delete;
	Handle(Handle &&other) noexcept : ptr(other.ptr) {
		other.ptr = nullptr;
	}
	~Handle() {
		if (ptr) {
			am_release(ptr);
		}
	}
};

// Wraps memory this file owns as an Arrow array, zero-copy (every buffer is page aligned). The Arrow
// release only drops the holder: the memory belongs to a Slab or a Region, which outlives the handle.
static am_array *ImportBuffers(Kind kind, int64_t length, int64_t null_count, const void *validity,
                               const void *second, const void *third) {
	auto holder = new ArrowHolder();
	holder->buffers[0] = validity;
	holder->buffers[1] = second;
	holder->buffers[2] = third;
	ArrowSchema schema;
	std::memset(&schema, 0, sizeof(schema));
	schema.format = KindFormat(kind);
	schema.name = "";
	schema.flags = ARROW_FLAG_NULLABLE;
	schema.release = ReleaseSchema;
	ArrowArray array;
	std::memset(&array, 0, sizeof(array));
	array.length = length;
	array.null_count = null_count;
	array.n_buffers = kind == Kind::STR ? 3 : 2;
	array.buffers = holder->buffers;
	array.private_data = holder;
	array.release = ReleaseArray;
	am_array *out = nullptr;
	const int rc = am_import(&schema, &array, &out);
	if (schema.release) {
		schema.release(&schema);
	}
	if (rc != 0) {
		if (array.release) {
			array.release(&array);
		}
		Fail("importing a gathered column");
	}
	return out;
}

// The first `rows` of a gathered column as an ArrowMetal array. A fixed-width column is a slice of
// its slab's long-lived import (made on the slab's first use); a string column is imported fresh.
static am_array *ImportColumn(BlockColumn &column, idx_t rows) {
	const int64_t nulls = column.nulls.load();
	if (column.kind == Kind::STR) {
		column.offsets_ptr[rows] = static_cast<int64_t>(column.bytes_used);
		return ImportBuffers(Kind::STR, int64_t(rows), nulls, nulls ? column.validity : nullptr, column.offsets_ptr,
		                     column.values);
	}
	auto &slab = *column.slab;
	am_array *&base = nulls ? slab.with_validity : slab.without_validity;
	if (!base) {
		// null_count -1: ArrowMetal counts the bitmap itself. Each slice counts its own rows again.
		base = ImportBuffers(slab.kind, int64_t(slab.capacity), nulls ? -1 : 0, nulls ? slab.validity : nullptr,
		                     slab.values, nullptr);
	}
	am_array *out = nullptr;
	if (am_slice(base, 0, int64_t(rows), &out) != 0) {
		Fail("slicing a gathered column");
	}
	return out;
}

// A column read back from the GPU: 64-bit slots (signed or unsigned bit patterns) or strings.
struct HostColumn {
	string format;
	vector<int64_t> values;
	vector<string> strings;
	vector<uint8_t> valid;
};

template <class T>
static void ReadValues(const void *data, int64_t n, int64_t off, HostColumn &out) {
	auto values = static_cast<const T *>(data) + off;
	for (int64_t i = 0; i < n; i++) {
		out.values[i] = static_cast<int64_t>(values[i]);
	}
}

static void ReadColumn(am_array *handle, HostColumn &out) {
	ArrowSchema schema;
	ArrowArray array;
	std::memset(&schema, 0, sizeof(schema));
	std::memset(&array, 0, sizeof(array));
	if (am_export(handle, &schema, &array) != 0) {
		Fail("reading a result column");
	}
	struct Releaser {
		ArrowSchema &s;
		ArrowArray &a;
		~Releaser() {
			if (a.release) {
				a.release(&a);
			}
			if (s.release) {
				s.release(&s);
			}
		}
	} releaser {schema, array};
	out.format = schema.format ? schema.format : "";
	const int64_t n = array.length;
	const int64_t off = array.offset;
	auto validity = array.n_buffers > 0 ? static_cast<const uint64_t *>(array.buffers[0]) : nullptr;
	const void *data = array.n_buffers > 1 ? array.buffers[1] : nullptr;
	out.values.assign(n, 0);
	out.valid.assign(n, 1);
	if (validity) {
		for (int64_t i = 0; i < n; i++) {
			const int64_t r = i + off;
			if (!((validity[r >> 6] >> (r & 63)) & 1)) {
				out.valid[i] = 0;
			}
		}
	}
	const string &f = out.format;
	if (f == "c") {
		ReadValues<int8_t>(data, n, off, out);
	} else if (f == "C") {
		ReadValues<uint8_t>(data, n, off, out);
	} else if (f == "s") {
		ReadValues<int16_t>(data, n, off, out);
	} else if (f == "S") {
		ReadValues<uint16_t>(data, n, off, out);
	} else if (f == "i" || f == "tdD") {
		ReadValues<int32_t>(data, n, off, out);
	} else if (f == "I") {
		ReadValues<uint32_t>(data, n, off, out);
	} else if (f == "l" || f == "L" || f.rfind("ts", 0) == 0) {
		ReadValues<int64_t>(data, n, off, out);
	} else if (f == "g") {
		std::memcpy(out.values.data(), static_cast<const double *>(data) + off, size_t(n) * 8);
	} else if (f == "u" || f == "U") {
		out.strings.assign(n, string());
		auto bytes = static_cast<const char *>(array.buffers[2]);
		for (int64_t i = 0; i < n; i++) {
			if (!out.valid[i]) {
				continue;
			}
			const int64_t r = i + off;
			int64_t begin, end;
			if (f == "u") {
				begin = static_cast<const int32_t *>(data)[r];
				end = static_cast<const int32_t *>(data)[r + 1];
			} else {
				begin = static_cast<const int64_t *>(data)[r];
				end = static_cast<const int64_t *>(data)[r + 1];
			}
			out.strings[i].assign(bytes + begin, size_t(end - begin));
		}
	} else {
		throw InternalException("arrowmetal_rewrite: unexpected result format '%s'", f);
	}
}

// One GPU-side aggregate the plan needs, deduplicated across the SQL aggregates that share it.
enum class Need : uint8_t { ROWS, COUNT, SUM, SUM_HI, SUM_LO, MIN, MAX };

struct NeedKey {
	Need need;
	idx_t input;
	bool operator==(const NeedKey &o) const {
		return need == o.need && input == o.input;
	}
};

struct Plan {
	vector<NeedKey> needs;
	idx_t Add(Need need, idx_t input) {
		for (idx_t i = 0; i < needs.size(); i++) {
			if (needs[i] == NeedKey {need, input}) {
				return i;
			}
		}
		needs.push_back(NeedKey {need, input});
		return needs.size() - 1;
	}
};

// Whether a SUM/AVG over `input` has to be split into 32-bit halves to stay exact.
static bool NeedsSplit(const Spec &spec, const AggSpec &agg) {
	if (agg.proven_no_overflow) {
		return false;
	}
	const Kind kind = spec.input_kinds[agg.input];
	return KindWidth(kind) == 8; // a 64-bit source can overflow int64; anything narrower cannot
}

struct AggSlots {
	idx_t count = DConstants::INVALID_INDEX;
	idx_t sum = DConstants::INVALID_INDEX;
	idx_t hi = DConstants::INVALID_INDEX;
	idx_t lo = DConstants::INVALID_INDEX;
	idx_t extreme = DConstants::INVALID_INDEX;
};

// `nullable[c]` says whether gathered column c holds any NULL; a count over a column without one is the
// row count, which the plan already has.
static vector<AggSlots> PlanNeeds(const Spec &spec, const vector<bool> &nullable, Plan &plan) {
	vector<AggSlots> slots(spec.aggs.size());
	plan.Add(Need::ROWS, DConstants::INVALID_INDEX); // always slot 0: rows per group
	auto count = [&](idx_t input) -> idx_t { return nullable[input] ? plan.Add(Need::COUNT, input) : 0; };
	for (idx_t a = 0; a < spec.aggs.size(); a++) {
		auto &agg = spec.aggs[a];
		auto &s = slots[a];
		switch (agg.op) {
		case AggOp::COUNT_STAR:
			break;
		case AggOp::COUNT:
			s.count = count(agg.input);
			break;
		case AggOp::SUM:
		case AggOp::AVG:
			s.count = count(agg.input);
			if (NeedsSplit(spec, agg)) {
				s.hi = plan.Add(Need::SUM_HI, agg.input);
				s.lo = plan.Add(Need::SUM_LO, agg.input);
			} else {
				s.sum = plan.Add(Need::SUM, agg.input);
			}
			break;
		case AggOp::MIN:
			s.count = count(agg.input);
			s.extreme = plan.Add(Need::MIN, agg.input);
			break;
		case AggOp::MAX:
			s.count = count(agg.input);
			s.extreme = plan.Add(Need::MAX, agg.input);
			break;
		}
	}
	return slots;
}

static string ColName(idx_t input) {
	return "c" + to_string(input);
}

// The fused-expression text for one need; `name` labels its output.
static string NeedText(const NeedKey &need, const string &name) {
	const string col = need.input == DConstants::INVALID_INDEX ? "" : "(col \"" + ColName(need.input) + "\")";
	switch (need.need) {
	case Need::ROWS: return "(count \"" + name + "\")";
	case Need::COUNT: return "(count \"" + name + "\" " + col + ")";
	case Need::SUM: return "(sum \"" + name + "\" " + col + ")";
	case Need::SUM_HI: return "(sum \"" + name + "\" (shr " + col + " (i64 32)))";
	case Need::SUM_LO: return "(sum \"" + name + "\" (bit_and " + col + " (i64 4294967295)))";
	case Need::MIN: return "(min \"" + name + "\" " + col + ")";
	case Need::MAX: return "(max \"" + name + "\" " + col + ")";
	}
	return "";
}

// Per-group results: one 64-bit slot per need, as a signed or unsigned bit pattern, plus validity.
struct Groups {
	idx_t count = 0;
	HostColumn key;                      // the group key per group (grouped queries)
	vector<vector<int64_t>> slot;        // [need][group]
	vector<bool> slot_unsigned;          // [need]
	vector<vector<__int128>> wide;       // [need][group]: merged sums, when the plan was streamed
};

struct QueryResultHolder {
	am_query_result *ptr = nullptr;
	~QueryResultHolder() {
		if (ptr) {
			am_query_result_release(ptr);
		}
	}
};

// Runs `text` over the gathered columns and appends the scalars (one row) to `groups`.
static void RunScalarQuery(vector<am_array *> &columns, const string &text, const Plan &plan, Groups &groups) {
	vector<string> names;
	vector<const char *> name_ptrs;
	for (idx_t i = 0; i < columns.size(); i++) {
		names.push_back(ColName(i));
	}
	for (auto &n : names) {
		name_ptrs.push_back(n.c_str());
	}
	QueryResultHolder result;
	if (am_query(columns.data(), name_ptrs.data(), int64_t(columns.size()), text.c_str(), &result.ptr) != 0) {
		Fail("running the fused aggregate");
	}
	const idx_t g = groups.count++;
	for (idx_t n = 0; n < plan.needs.size(); n++) {
		int64_t iv = 0;
		double dv = 0;
		int kind = 0;
		int is_null = 0;
		if (am_query_scalar(result.ptr, int64_t(n), &iv, &dv, &kind, &is_null) != 0) {
			Fail("reading an aggregate");
		}
		groups.slot[n].resize(g + 1);
		groups.slot[n][g] = iv;
		(void)is_null; // nullness comes from the counts
		groups.slot_unsigned[n] = kind == 1;
	}
}

static string AggregateList(const Plan &plan) {
	string text;
	for (idx_t n = 0; n < plan.needs.size(); n++) {
		text += " " + NeedText(plan.needs[n], "n" + to_string(n));
	}
	return text;
}

static bool SignedResultFormat(const string &f) {
	return f == "c" || f == "s" || f == "i" || f == "l" || f == "tdD" || f.rfind("ts", 0) == 0;
}

// The fused dense group-by: one pass, one slot per key value in [key_min, key_max].
static void RunDense(vector<am_array *> &columns, const Plan &plan, int64_t key_min, int64_t span,
                     int64_t key_nulls, Kind key_kind, Groups &groups) {
	vector<string> names;
	vector<const char *> name_ptrs;
	for (idx_t i = 0; i < columns.size(); i++) {
		names.push_back(ColName(i));
	}
	for (auto &n : names) {
		name_ptrs.push_back(n.c_str());
	}
	const string key = "(sub (cast (col \"c0\") i64) (i64 " + to_string(key_min) + "))";
	// The fused group-by keeps its table in threadgroup memory while span x aggregates fits in
	// FUSED_PRIVATE_SLOTS, and falls back to contended device-wide atomics above that. For small key
	// spans it is cheaper to read the columns again than to leave threadgroup memory, so the needs are
	// split across as many queries as keep each one private.
	idx_t per_query = plan.needs.size();
	if (span * int64_t(plan.needs.size()) > FUSED_PRIVATE_SLOTS && span <= FUSED_PRIVATE_SLOTS / 2) {
		per_query = idx_t(FUSED_PRIVATE_SLOTS / span);
	}
	vector<HostColumn> host(plan.needs.size());
	const auto traced = std::chrono::steady_clock::now();
	for (idx_t first = 0; first < plan.needs.size(); first += per_query) {
		const idx_t last = MinValue<idx_t>(plan.needs.size(), first + per_query);
		string aggs;
		for (idx_t n = first; n < last; n++) {
			aggs += " " + NeedText(plan.needs[n], "n" + to_string(n));
		}
		const string text = "(query (group_by " + to_string(span) + " \"key\" " + key + ") (aggregate" + aggs + "))";
		QueryResultHolder result;
		if (am_query(columns.data(), name_ptrs.data(), int64_t(columns.size()), text.c_str(), &result.ptr) != 0) {
			Fail("running the fused group-by");
		}
		Trace("  query", traced);
		// Column 0 is the key slot index, then one column per need in order.
		for (idx_t n = first; n < last; n++) {
			Handle column;
			if (am_query_column(result.ptr, int64_t(n - first + 1), &column.ptr) != 0) {
				Fail("reading a grouped aggregate");
			}
			ReadColumn(column.ptr, host[n]);
		}
		Trace("  read", traced);
	}
	auto &rows = host[0]; // Need::ROWS is always slot 0
	vector<uint32_t> present;
	present.reserve(size_t(span));
	for (int64_t k = 0; k < span; k++) {
		if (rows.values[k] > 0) {
			present.push_back(uint32_t(k));
		}
	}
	const idx_t count = present.size();
	groups.key.format = KindFormat(key_kind);
	groups.key.values.resize(count);
	groups.key.valid.assign(count, 1);
	for (idx_t g = 0; g < count; g++) {
		groups.key.values[g] = key_min + int64_t(present[g]);
	}
	for (idx_t n = 0; n < plan.needs.size(); n++) {
		auto &slot = groups.slot[n];
		const auto &values = host[n].values;
		slot.resize(count);
		for (idx_t g = 0; g < count; g++) {
			slot[g] = values[present[g]];
		}
		groups.slot_unsigned[n] = !SignedResultFormat(host[n].format);
	}
	groups.count = count;
	if (key_nulls > 0) {
		// Rows whose key is NULL form one more group, as in SQL.
		const string null_text = "(query (filter (is_null (col \"c0\"))) (aggregate" + AggregateList(plan) + "))";
		const idx_t before = groups.count;
		auto unsigned_flags = groups.slot_unsigned;
		RunScalarQuery(columns, null_text, plan, groups);
		groups.slot_unsigned = unsigned_flags; // keep the grouped columns' signedness
		groups.key.values.push_back(0);
		groups.key.valid.push_back(0);
		(void)before;
	}
}

// The hash group-by: am_group_by_keys, then one am_group_agg_ex per need.
static void RunHash(vector<am_array *> &columns, const Plan &plan, Groups &groups) {
	am_groupby *gb = nullptr;
	am_array *key_columns[1] = {columns[0]};
	if (am_group_by_keys(key_columns, 1, &gb) != 0) {
		Fail("grouping the key column");
	}
	struct GroupByHolder {
		am_groupby *ptr;
		~GroupByHolder() {
			am_group_by_release(ptr);
		}
	} holder {gb};
	{
		Handle keys;
		if (am_group_by_keys_result(gb, 0, &keys.ptr) != 0) {
			Fail("reading the group keys");
		}
		ReadColumn(keys.ptr, groups.key);
	}
	groups.count = groups.key.valid.size();

	// The 32-bit halves of any column that must be summed in two parts, computed once per column.
	vector<std::pair<idx_t, std::pair<unique_ptr<Handle>, unique_ptr<Handle>>>> halves;
	auto half = [&](idx_t input, bool high) -> am_array * {
		for (auto &h : halves) {
			if (h.first == input) {
				return high ? h.second.first->ptr : h.second.second->ptr;
			}
		}
		vector<string> names;
		vector<const char *> name_ptrs;
		for (idx_t i = 0; i < columns.size(); i++) {
			names.push_back(ColName(i));
		}
		for (auto &n : names) {
			name_ptrs.push_back(n.c_str());
		}
		const string col = "(col \"" + ColName(input) + "\")";
		const string text = "(query (project (as \"h\" (shr " + col + " (i64 32))) (as \"l\" (bit_and " + col +
		                    " (i64 4294967295)))))";
		QueryResultHolder result;
		if (am_query(columns.data(), name_ptrs.data(), int64_t(columns.size()), text.c_str(), &result.ptr) != 0) {
			Fail("splitting a column into 32-bit halves");
		}
		auto hi = make_uniq<Handle>();
		auto lo = make_uniq<Handle>();
		if (am_query_column(result.ptr, 0, &hi->ptr) != 0 || am_query_column(result.ptr, 1, &lo->ptr) != 0) {
			Fail("reading the 32-bit halves");
		}
		halves.emplace_back(input, std::make_pair(std::move(hi), std::move(lo)));
		auto &h = halves.back();
		return high ? h.second.first->ptr : h.second.second->ptr;
	};

	for (idx_t n = 0; n < plan.needs.size(); n++) {
		const auto &need = plan.needs[n];
		HostColumn host;
		if (need.need == Need::COUNT && need.input == 0 && columns[0] && groups.key.format.size() &&
		    (groups.key.format == "u" || groups.key.format == "U")) {
			// COUNT of the string key itself: every row of a group shares its key, so the count is the
			// group's row count, or 0 for the NULL-key group. (ArrowMetal has no grouped count over utf8.)
			Handle rows;
			if (am_group_agg_ex(gb, nullptr, 1, 0.0, &rows.ptr) != 0) {
				Fail("counting rows per group");
			}
			ReadColumn(rows.ptr, host);
			for (idx_t g = 0; g < groups.count; g++) {
				if (!groups.key.valid[g]) {
					host.values[g] = 0;
				}
			}
		} else {
			Handle out;
			int rc = 0;
			switch (need.need) {
			case Need::ROWS: rc = am_group_agg_ex(gb, nullptr, 1, 0.0, &out.ptr); break;
			case Need::COUNT: rc = am_group_agg_ex(gb, columns[need.input], 2, 0.0, &out.ptr); break;
			case Need::SUM: rc = am_group_agg_ex(gb, columns[need.input], 0, 0.0, &out.ptr); break;
			case Need::SUM_HI: rc = am_group_agg_ex(gb, half(need.input, true), 0, 0.0, &out.ptr); break;
			case Need::SUM_LO: rc = am_group_agg_ex(gb, half(need.input, false), 0, 0.0, &out.ptr); break;
			case Need::MIN: rc = am_group_agg_ex(gb, columns[need.input], 4, 0.0, &out.ptr); break;
			case Need::MAX: rc = am_group_agg_ex(gb, columns[need.input], 5, 0.0, &out.ptr); break;
			}
			if (rc != 0) {
				Fail("running a grouped aggregate");
			}
			ReadColumn(out.ptr, host);
		}
		groups.slot[n] = std::move(host.values);
		groups.slot_unsigned[n] = !SignedResultFormat(host.format);
	}
}

//===--------------------------------------------------------------------===//
// Finishing each SQL aggregate from its slots, with DuckDB's own arithmetic
//===--------------------------------------------------------------------===//
static hugeint_t ToHuge(__int128 v) {
	return hugeint_t(static_cast<int64_t>(v >> 64), static_cast<uint64_t>(v));
}

static __int128 SlotSum(const Groups &groups, idx_t slot, idx_t g) {
	if (slot < groups.wide.size() && !groups.wide[slot].empty()) {
		return groups.wide[slot][g];
	}
	return groups.slot_unsigned[slot] ? __int128(static_cast<uint64_t>(groups.slot[slot][g]))
	                                  : __int128(groups.slot[slot][g]);
}

template <class T>
static void WriteInts(Vector &vec, const vector<int64_t> &source, idx_t start, idx_t n) {
	auto out = FlatVector::GetData<T>(vec);
	for (idx_t i = 0; i < n; i++) {
		out[i] = static_cast<T>(source[start + i]);
	}
}

// Integer slots into a vector of `type` (the value is already in that type's range).
static void WriteIntegral(Vector &vec, PhysicalType type, const vector<int64_t> &source, idx_t start, idx_t n) {
	switch (type) {
	case PhysicalType::INT8: WriteInts<int8_t>(vec, source, start, n); break;
	case PhysicalType::INT16: WriteInts<int16_t>(vec, source, start, n); break;
	case PhysicalType::INT32: WriteInts<int32_t>(vec, source, start, n); break;
	case PhysicalType::INT64: WriteInts<int64_t>(vec, source, start, n); break;
	case PhysicalType::UINT8: WriteInts<uint8_t>(vec, source, start, n); break;
	case PhysicalType::UINT16: WriteInts<uint16_t>(vec, source, start, n); break;
	case PhysicalType::UINT32: WriteInts<uint32_t>(vec, source, start, n); break;
	case PhysicalType::UINT64: WriteInts<uint64_t>(vec, source, start, n); break;
	default:
		throw InternalException("arrowmetal_rewrite: unexpected integral result type");
	}
}

static void WriteResult(const Spec &spec, const vector<AggSlots> &slots, const Groups &groups,
                        ColumnDataCollection &collection) {
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), spec.result_types);
	const idx_t total = groups.count;
	const auto &rows = groups.slot[0];
	for (idx_t start = 0; start < total; start += STANDARD_VECTOR_SIZE) {
		const idx_t n = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - start);
		chunk.Reset();
		idx_t col = 0;
		if (spec.has_key) {
			auto &vec = chunk.data[col++];
			auto &validity = FlatVector::Validity(vec);
			const auto type = spec.result_types[0].InternalType();
			if (type == PhysicalType::VARCHAR) {
				auto out = FlatVector::GetData<string_t>(vec);
				for (idx_t i = 0; i < n; i++) {
					if (groups.key.valid[start + i]) {
						out[i] = StringVector::AddString(vec, groups.key.strings[start + i]);
					}
				}
			} else {
				WriteIntegral(vec, type, groups.key.values, start, n);
			}
			for (idx_t i = 0; i < n; i++) {
				if (!groups.key.valid[start + i]) {
					validity.SetInvalid(i);
				}
			}
		}
		for (idx_t a = 0; a < spec.aggs.size(); a++) {
			auto &agg = spec.aggs[a];
			auto &s = slots[a];
			auto &vec = chunk.data[col++];
			auto &validity = FlatVector::Validity(vec);
			if (agg.op == AggOp::COUNT_STAR) {
				WriteInts<int64_t>(vec, rows, start, n);
				continue;
			}
			const auto &valid = groups.slot[s.count];
			if (agg.op == AggOp::COUNT) {
				WriteInts<int64_t>(vec, valid, start, n);
				continue;
			}
			if (agg.op == AggOp::MIN || agg.op == AggOp::MAX) {
				WriteIntegral(vec, agg.return_type.InternalType(), groups.slot[s.extreme], start, n);
			} else {
				const bool huge = agg.return_type.InternalType() == PhysicalType::INT128;
				for (idx_t i = 0; i < n; i++) {
					const idx_t g = start + i;
					if (valid[g] == 0) {
						continue;
					}
					__int128 total_sum;
					if (s.sum != DConstants::INVALID_INDEX) {
						total_sum = SlotSum(groups, s.sum, g);
					} else {
						total_sum = SlotSum(groups, s.hi, g) * (__int128(1) << 32) + SlotSum(groups, s.lo, g);
					}
					if (agg.op == AggOp::SUM) {
						if (huge) {
							FlatVector::GetData<hugeint_t>(vec)[i] = ToHuge(total_sum);
						} else {
							FlatVector::GetData<int64_t>(vec)[i] = int64_t(total_sum);
						}
					} else if (agg.semantic_bits == 16) {
						// IntegerAverageOperation: an int64 state, double(sum) / double(count)
						FlatVector::GetData<double>(vec)[i] = double(int64_t(total_sum)) / double(uint64_t(valid[g]));
					} else {
						// IntegerAverageOperationHugeint: Hugeint::Cast<long double>(sum) / (long double)count
						long double numerator = 0;
						Hugeint::TryCast<long double>(ToHuge(total_sum), numerator);
						const long double divident = static_cast<long double>(uint64_t(valid[g]));
						FlatVector::GetData<double>(vec)[i] = static_cast<double>(numerator / divident);
					}
				}
			}
			// SUM, AVG, MIN and MAX of a group with no value are NULL.
			for (idx_t i = 0; i < n; i++) {
				if (valid[start + i] == 0) {
					validity.SetInvalid(i);
				}
			}
		}
		chunk.SetCardinality(n);
		collection.Append(chunk);
	}
}

//===--------------------------------------------------------------------===//
// Running the plan over one block
//===--------------------------------------------------------------------===//
struct Arrays {
	vector<am_array *> ptrs;
	~Arrays() {
		for (auto h : ptrs) {
			if (h) {
				am_release(h);
			}
		}
	}
};

// The GPU half over rows [0, rows) of `block`: every need of `plan` into `groups`. Returns the path taken.
static string RunBlock(const Spec &spec, Block &block, idx_t rows, const Plan &plan, Groups &groups) {
	Arrays columns;
	for (auto &column : block.columns) {
		columns.ptrs.push_back(ImportColumn(*column, rows));
	}
	if (!spec.has_key) {
		RunScalarQuery(columns.ptrs, "(query (aggregate" + AggregateList(plan) + "))", plan, groups);
		return "fused aggregate";
	}
	const Kind key_kind = spec.input_kinds[0];
	const bool integral = key_kind != Kind::STR && key_kind != Kind::U64;
	if (integral && !block.HasKeys()) {
		// Every key is NULL: one group.
		RunScalarQuery(columns.ptrs, "(query (aggregate" + AggregateList(plan) + "))", plan, groups);
		groups.key.format = KindFormat(key_kind);
		groups.key.values.assign(1, 0);
		groups.key.valid.assign(1, 0);
		return "fused aggregate (every key NULL)";
	}
	// The span in unsigned arithmetic: key_max - key_min can exceed INT64_MAX.
	const int64_t key_min = block.key_min.load(), key_max = block.key_max.load();
	bool dense = integral && uint64_t(key_max) - uint64_t(key_min) < uint64_t(DENSE_KEY_SPAN);
	// The fused group-by keeps MIN and MAX in 32-bit atomics.
	for (auto &need : plan.needs) {
		if ((need.need == Need::MIN || need.need == Need::MAX) && KindWidth(spec.input_kinds[need.input]) > 4) {
			dense = false;
		}
	}
	if (dense) {
		RunDense(columns.ptrs, plan, key_min, key_max - key_min + 1, block.columns[0]->nulls.load(), key_kind, groups);
		return "fused dense group-by";
	}
	RunHash(columns.ptrs, plan, groups);
	return "hash group-by";
}

static vector<bool> Nullable(const Block &block) {
	vector<bool> nullable;
	for (auto &column : block.columns) {
		nullable.push_back(column->nulls.load() > 0);
	}
	return nullable;
}

static bool IsExtreme(Need need) {
	return need == Need::MIN || need == Need::MAX;
}

//===--------------------------------------------------------------------===//
// Streaming: per-block partial results, merged on the host
//
// Every partial merges exactly: counts and sums add (sums in 128 bits, so a streamed plan has no row
// limit), MIN and MAX take the extreme of the blocks that saw a value, and a group is keyed by its key
// value, with the NULL key as one more group. A block's own plan may drop a COUNT over a column that
// held no NULL in that block; the merge reads the block's row count in its place.
//===--------------------------------------------------------------------===//
struct StreamAccumulator {
	explicit StreamAccumulator(const Spec &spec_p) : spec(spec_p) {
		const vector<bool> all(spec.input_kinds.size(), true);
		slots = PlanNeeds(spec, all, plan);
		const idx_t n = plan.needs.size();
		sums.resize(n);
		extremes.resize(n);
		seen.resize(n);
		is_unsigned.assign(n, false);
		count_of.assign(n, 0);
		for (idx_t l = 0; l < n; l++) {
			const auto &need = plan.needs[l];
			if (need.need != Need::ROWS && need.need != Need::COUNT) {
				for (idx_t c = 0; c < n; c++) {
					if (plan.needs[c] == NeedKey {Need::COUNT, need.input}) {
						count_of[l] = c;
					}
				}
			}
		}
	}

	const Spec &spec;
	Plan plan; // every value input treated as nullable, so every COUNT is there
	vector<AggSlots> slots;
	vector<idx_t> count_of; // a value need's COUNT over the same input
	vector<int64_t> keys;
	vector<uint8_t> key_valid;
	int64_t dense_base = 0;
	vector<uint32_t> dense; // key - dense_base -> group, while the keys stay in a small range
	bool sparse_mode = false;
	std::unordered_map<int64_t, uint32_t> sparse;
	uint32_t null_group = NumericLimits<uint32_t>::Maximum();
	vector<vector<__int128>> sums;    // ROWS, COUNT, SUM, SUM_HI, SUM_LO
	vector<vector<int64_t>> extremes; // MIN, MAX
	vector<vector<uint8_t>> seen;     // MIN, MAX
	vector<bool> is_unsigned;
	string path;
	idx_t merged_blocks = 0;

	uint32_t AddGroup(bool valid, int64_t key) {
		const uint32_t g = uint32_t(keys.size());
		keys.push_back(key);
		key_valid.push_back(valid ? 1 : 0);
		for (idx_t l = 0; l < plan.needs.size(); l++) {
			if (IsExtreme(plan.needs[l].need)) {
				extremes[l].push_back(0);
				seen[l].push_back(0);
			} else {
				sums[l].push_back(0);
			}
		}
		return g;
	}

	uint32_t GroupFor(bool valid, int64_t key) {
		constexpr uint32_t NONE = NumericLimits<uint32_t>::Maximum();
		if (!spec.has_key) {
			return keys.empty() ? AddGroup(true, 0) : 0;
		}
		if (!valid) {
			if (null_group == NONE) {
				null_group = AddGroup(false, 0);
			}
			return null_group;
		}
		if (!sparse_mode) {
			if (dense.empty()) {
				dense_base = key;
				dense.assign(1, NONE);
			}
			const __int128 top = __int128(dense_base) + __int128(dense.size()) - 1;
			if (key < dense_base || __int128(key) > top) {
				const __int128 lo = key < dense_base ? __int128(key) : __int128(dense_base);
				const __int128 hi = __int128(key) > top ? __int128(key) : top;
				if (hi - lo < __int128(4 * DENSE_KEY_SPAN)) {
					vector<uint32_t> grown(size_t(hi - lo + 1), NONE);
					std::memcpy(grown.data() + size_t(__int128(dense_base) - lo), dense.data(),
					            dense.size() * sizeof(uint32_t));
					dense.swap(grown);
					dense_base = int64_t(lo);
				} else {
					for (idx_t i = 0; i < dense.size(); i++) {
						if (dense[i] != NONE) {
							sparse[int64_t(__int128(dense_base) + __int128(i))] = dense[i];
						}
					}
					dense.clear();
					sparse_mode = true;
				}
			}
			if (!sparse_mode) {
				auto &slot = dense[size_t(__int128(key) - __int128(dense_base))];
				if (slot == NONE) {
					slot = AddGroup(true, key);
				}
				return slot;
			}
		}
		auto entry = sparse.find(key);
		if (entry != sparse.end()) {
			return entry->second;
		}
		const uint32_t g = AddGroup(true, key);
		sparse.emplace(key, g);
		return g;
	}

	void Merge(const Groups &groups, const Plan &block_plan, const vector<bool> &nullable) {
		const idx_t n = plan.needs.size();
		vector<idx_t> src(n, 0);
		for (idx_t l = 0; l < n; l++) {
			const auto &need = plan.needs[l];
			if (need.need == Need::COUNT && !nullable[need.input]) {
				continue; // the block's row count stands in
			}
			bool found = false;
			for (idx_t b = 0; b < block_plan.needs.size(); b++) {
				if (block_plan.needs[b] == need) {
					src[l] = b;
					found = true;
				}
			}
			if (!found) {
				throw InternalException("arrowmetal_rewrite: a block's plan lacks a need");
			}
		}
		for (idx_t i = 0; i < groups.count; i++) {
			const bool valid = !spec.has_key || groups.key.valid[i];
			const uint32_t g = GroupFor(valid, valid && spec.has_key ? groups.key.values[i] : 0);
			for (idx_t l = 0; l < n; l++) {
				const auto need = plan.needs[l].need;
				const idx_t s = src[l];
				const int64_t v = groups.slot[s][i];
				if (need == Need::ROWS || need == Need::COUNT) {
					sums[l][g] += v;
					continue;
				}
				// A value need counts only where this block had a value of that input in this group.
				if (groups.slot[src[count_of[l]]][i] <= 0) {
					continue;
				}
				const bool uns = groups.slot_unsigned[s];
				if (!IsExtreme(need)) {
					sums[l][g] += uns ? __int128(uint64_t(v)) : __int128(v);
					continue;
				}
				is_unsigned[l] = uns;
				if (!seen[l][g]) {
					extremes[l][g] = v;
					seen[l][g] = 1;
				} else {
					const bool better = need == Need::MIN ? (uns ? uint64_t(v) < uint64_t(extremes[l][g]) : v < extremes[l][g])
					                                     : (uns ? uint64_t(v) > uint64_t(extremes[l][g]) : v > extremes[l][g]);
					if (better) {
						extremes[l][g] = v;
					}
				}
			}
		}
		merged_blocks++;
	}

	void Finish(Groups &out) {
		if (!spec.has_key && keys.empty()) {
			AddGroup(true, 0); // no rows at all: one row, counts 0, everything else NULL
		}
		const idx_t n = plan.needs.size();
		out.count = keys.size();
		if (spec.has_key) {
			out.key.format = KindFormat(spec.input_kinds[0]);
			out.key.values = keys;
			out.key.valid = key_valid;
		}
		out.slot.assign(n, vector<int64_t>());
		out.wide.assign(n, vector<__int128>());
		out.slot_unsigned = is_unsigned;
		for (idx_t l = 0; l < n; l++) {
			const auto need = plan.needs[l].need;
			if (IsExtreme(need)) {
				out.slot[l] = extremes[l];
			} else if (need == Need::ROWS || need == Need::COUNT) {
				out.slot[l].resize(out.count);
				for (idx_t g = 0; g < out.count; g++) {
					out.slot[l][g] = int64_t(sums[l][g]);
				}
			} else {
				out.wide[l] = sums[l];
			}
		}
	}
};

// Runs one block through the GPU and merges its partial result. The block's buffers go back to the slab
// pool at once, so a streamed plan holds only the blocks being filled or waiting.
static void ProcessBlock(RewriteGlobalState &gstate, Block &block, idx_t rows) {
	std::lock_guard<std::recursive_mutex> gpu(g_gpu_lock);
	const auto nullable = Nullable(block);
	Plan plan;
	PlanNeeds(gstate.spec, nullable, plan);
	Groups groups;
	groups.slot.resize(plan.needs.size());
	groups.slot_unsigned.assign(plan.needs.size(), false);
	gstate.accumulator->path = RunBlock(gstate.spec, block, rows, plan, groups);
	gstate.accumulator->Merge(groups, plan, nullable);
	block.processed = true;
	block.columns.clear();
}

RewriteGlobalState::RewriteGlobalState(ClientContext &context, const Spec &spec_p, idx_t capacity_rows)
    : spec(spec_p), stream(spec_p.stream), overflow_types(spec_p.input_types), result(context, spec_p.result_types) {
	for (auto kind : spec.input_kinds) {
		has_strings = has_strings || kind == Kind::STR;
	}
	capacity_rows = MaxValue<idx_t>(capacity_rows, STANDARD_VECTOR_SIZE);
	if (stream) {
		// Blocks of the configured size, or one block when the whole input is smaller than that.
		block_rows = MinValue<idx_t>(spec.block_rows, (capacity_rows + 63) / 64 * 64);
		max_blocks = capacity_rows / block_rows + 1024;
		directory = unique_ptr<std::atomic<Block *>[]>(new std::atomic<Block *>[max_blocks]);
		for (idx_t b = 0; b < max_blocks; b++) {
			directory[b].store(nullptr);
		}
		accumulator = make_uniq<StreamAccumulator>(spec);
	} else {
		max_blocks = 1;
		directory = unique_ptr<std::atomic<Block *>[]>(new std::atomic<Block *>[1]);
		blocks.push_back(make_uniq<Block>(spec.input_kinds, 0, capacity_rows, false));
		block_rows = blocks[0]->capacity;
		directory[0].store(blocks[0].get());
	}
}

RewriteGlobalState::~RewriteGlobalState() {
	StopWorker();
}

void RewriteGlobalState::Enqueue(Block &block) {
	std::lock_guard<std::mutex> guard(queue_lock);
	if (!worker_started) {
		worker = std::thread([this]() { WorkerLoop(); });
		worker_started = true;
	}
	queue.push_back(&block);
	queue_cv.notify_one();
}

void RewriteGlobalState::StopWorker() {
	{
		std::lock_guard<std::mutex> guard(queue_lock);
		closing = true;
	}
	queue_cv.notify_all();
	if (worker_started && worker.joinable()) {
		worker.join();
	}
}

void RewriteGlobalState::WorkerLoop() {
	while (true) {
		Block *block = nullptr;
		{
			std::unique_lock<std::mutex> guard(queue_lock);
			queue_cv.wait(guard, [this]() { return closing || !queue.empty(); });
			if (queue.empty()) {
				return;
			}
			block = queue.front();
			queue.pop_front();
		}
		if (!worker_error.empty()) {
			continue; // Finalize reports the first error
		}
		try {
			ProcessBlock(*this, *block, block_rows);
			streamed_blocks++;
		} catch (std::exception &e) {
			ErrorData error(e);
			worker_error = error.RawMessage();
		}
	}
}

// The rare path of a single-block plan: some chunks did not fit. Moves everything into a block large
// enough for all of it - the rows already gathered, then the kept-aside chunks - so the GPU still sees
// one array per column.
static void AppendOverflow(RewriteGlobalState &gstate) {
	auto &overflow = *gstate.overflow;
	auto &old = *gstate.blocks[0];
	const idx_t kept = gstate.RegionRows();
	const idx_t total = kept + overflow.Count();
	size_t extra_bytes = 0;
	if (gstate.has_strings) {
		for (auto &chunk : overflow.Chunks()) {
			UnifiedVectorFormat format;
			chunk.data[0].ToUnifiedFormat(chunk.size(), format);
			extra_bytes += StringBytes(format, chunk.size());
		}
	}
	auto grown = make_uniq<Block>(vector<Kind>(), 0, total, false);
	for (auto &column : old.columns) {
		auto fresh = make_uniq<BlockColumn>(column->kind, total, column->bytes_used + extra_bytes + 1);
		std::memcpy(fresh->validity, column->validity, ((kept + 63) / 64) * 8);
		// Bits past `kept` in the last copied word belong to rows never written; mark them valid.
		if (kept % 64) {
			fresh->validity[kept / 64] |= ~((uint64_t(1) << (kept % 64)) - 1);
		}
		if (column->kind == Kind::STR) {
			std::memcpy(fresh->offsets_ptr, column->offsets_ptr, kept * sizeof(int64_t));
			std::memcpy(fresh->values, column->values, column->bytes_used);
			fresh->bytes_used = column->bytes_used;
		} else {
			std::memcpy(fresh->values, column->values, kept * KindWidth(column->kind));
		}
		fresh->nulls.store(column->nulls.load());
		grown->columns.push_back(std::move(fresh));
	}
	grown->capacity = total;
	grown->key_min.store(old.key_min.load());
	grown->key_max.store(old.key_max.load());
	idx_t row = kept;
	for (auto &chunk : overflow.Chunks()) {
		const idx_t count = chunk.size();
		for (idx_t c = 0; c < grown->columns.size(); c++) {
			UnifiedVectorFormat format;
			chunk.data[c].ToUnifiedFormat(count, format);
			auto &column = *grown->columns[c];
			const size_t byte0 = column.bytes_used;
			if (column.kind == Kind::STR) {
				column.bytes_used += StringBytes(format, count);
			}
			CopyRows(column, format, 0, count, row, byte0);
			if (c == 0 && gstate.spec.has_key) {
				NoteKeyRange(*grown, column.kind, format, 0, count);
			}
		}
		row += count;
	}
	gstate.blocks[0] = std::move(grown);
	gstate.directory[0].store(gstate.blocks[0].get());
	gstate.block_rows = total;
	gstate.reserved.store(total);
	gstate.hole_start.store(DConstants::INVALID_INDEX);
	gstate.locked_rows = total;
	gstate.overflow.reset();
}

// The rare path of a streamed plan: rows past the last block run through fresh blocks here.
static void StreamOverflow(RewriteGlobalState &gstate) {
	unique_ptr<Block> block;
	idx_t filled = 0;
	const idx_t size = gstate.block_rows;
	for (auto &chunk : gstate.overflow->Chunks()) {
		const idx_t count = chunk.size();
		vector<UnifiedVectorFormat> formats(chunk.ColumnCount());
		for (idx_t c = 0; c < chunk.ColumnCount(); c++) {
			chunk.data[c].ToUnifiedFormat(count, formats[c]);
		}
		for (idx_t done = 0; done < count;) {
			if (!block) {
				block = make_uniq<Block>(gstate.spec.input_kinds, 0, size, true);
				filled = 0;
			}
			const idx_t n = MinValue<idx_t>(count - done, size - filled);
			for (idx_t c = 0; c < formats.size(); c++) {
				CopyRows(*block->columns[c], formats[c], done, n, filled, 0);
			}
			if (gstate.spec.has_key) {
				NoteKeyRange(*block, gstate.spec.input_kinds[0], formats[0], done, n);
			}
			filled += n;
			done += n;
			if (filled == size) {
				ProcessBlock(gstate, *block, filled);
				block.reset();
			}
		}
	}
	if (block && filled) {
		ProcessBlock(gstate, *block, filled);
	}
	gstate.overflow.reset();
}

SinkFinalizeType PhysicalArrowMetalAggregate::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                       OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<RewriteGlobalState>();
	const auto started = std::chrono::steady_clock::now();
	const idx_t overflowed = gstate.overflow ? gstate.overflow->Count() : 0;
	Groups groups;
	vector<AggSlots> slots;
	string path;
	idx_t rows = 0;

	if (gstate.stream) {
		gstate.StopWorker();
		if (!gstate.worker_error.empty()) {
			throw InvalidInputException(gstate.worker_error);
		}
		Trace("worker", started);
		rows = gstate.RegionRows();
		const idx_t nblocks = (rows + gstate.block_rows - 1) / gstate.block_rows;
		for (idx_t b = 0; b < nblocks; b++) {
			Block *block = gstate.directory[b].load();
			if (block && !block->processed) {
				ProcessBlock(gstate, *block, MinValue<idx_t>(gstate.block_rows, rows - b * gstate.block_rows));
			}
		}
		if (overflowed > 0) {
			StreamOverflow(gstate);
			rows += overflowed;
		}
		auto &acc = *gstate.accumulator;
		acc.Finish(groups);
		slots = acc.slots;
		path = (acc.path.empty() ? string("empty") : acc.path) + ", streamed in " + to_string(acc.merged_blocks) +
		       (acc.merged_blocks == 1 ? " block" : " blocks") + ", " + to_string(gstate.streamed_blocks) +
		       " handed to the GPU as they filled";
	} else {
		if (overflowed > 0) {
			AppendOverflow(gstate);
		}
		rows = gstate.RegionRows();
		if (int64_t(rows) > MAX_ROWS) {
			throw InvalidInputException("arrowmetal_rewrite: %llu rows reached the aggregate, more than the GPU "
			                            "path takes (2^31 - 1); SET arrowmetal_rewrite = 'off' for this query",
			                            (unsigned long long)rows);
		}
		Block &block = *gstate.blocks[0];
		Plan plan;
		slots = PlanNeeds(spec, Nullable(block), plan);
		groups.slot.resize(plan.needs.size());
		groups.slot_unsigned.assign(plan.needs.size(), false);
		if (rows == 0) {
			// No input: an ungrouped aggregate still answers one row (counts 0, everything else NULL), a
			// grouped one answers none.
			path = "empty";
			if (!spec.has_key) {
				groups.count = 1;
				for (idx_t n = 0; n < plan.needs.size(); n++) {
					groups.slot[n].assign(1, 0);
				}
			}
		} else {
			std::lock_guard<std::recursive_mutex> gpu(g_gpu_lock);
			path = RunBlock(spec, block, rows, plan, groups);
		}
	}
	Trace("gpu", started);
	if (overflowed > 0) {
		path += " (" + to_string(overflowed) + " rows past the reserved buffers)";
	}
	WriteResult(spec, slots, groups, gstate.result);
	Trace("write", started);
	gstate.groups = int64_t(groups.count);
	const double ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
	RecordRun(spec.decision_id, path, int64_t(rows), int64_t(groups.count), ms);
	// The gathered columns are no longer needed; hand them back before the result is scanned.
	{
		std::lock_guard<std::recursive_mutex> gpu(g_gpu_lock);
		gstate.blocks.clear();
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// The logical operator that stands in for the LogicalAggregate
//===--------------------------------------------------------------------===//
class LogicalArrowMetalAggregate : public LogicalExtensionOperator {
public:
	LogicalArrowMetalAggregate(Spec spec_p, idx_t group_index_p, idx_t aggregate_index_p, idx_t capacity_rows_p)
	    : spec(std::move(spec_p)), group_index(group_index_p), aggregate_index(aggregate_index_p),
	      capacity_rows(capacity_rows_p) {
	}

	Spec spec;
	idx_t group_index;
	idx_t aggregate_index;
	idx_t capacity_rows;

	vector<ColumnBinding> GetColumnBindings() override {
		vector<ColumnBinding> result;
		if (spec.has_key) {
			result.emplace_back(group_index, 0);
		}
		for (idx_t i = 0; i < spec.aggs.size(); i++) {
			result.emplace_back(aggregate_index, i);
		}
		return result;
	}

	vector<idx_t> GetTableIndex() const override {
		return {group_index, aggregate_index};
	}

	string GetName() const override {
		return "ARROWMETAL_AGGREGATE";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		string aggs;
		for (idx_t i = 0; i < spec.aggs.size(); i++) {
			aggs += (i ? "\n" : "") + spec.aggs[i].name;
		}
		result["Aggregates"] = aggs;
		SetParamsEstimatedCardinality(result);
		return result;
	}

	string GetExtensionName() const override {
		return "arrowmetal_rewrite";
	}

	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		auto &child = planner.CreatePlan(*children[0]);
		// After column binding resolution each expression is a reference into the child's chunk.
		vector<idx_t> child_columns;
		for (auto &expr : expressions) {
			if (expr->GetExpressionClass() != ExpressionClass::BOUND_REF) {
				throw InternalException("arrowmetal_rewrite: expected a column reference");
			}
			child_columns.push_back(expr->Cast<BoundReferenceExpression>().index);
		}
		auto &op = planner.Make<PhysicalArrowMetalAggregate>(spec, std::move(child_columns), capacity_rows,
		                                                     estimated_cardinality);
		op.children.push_back(child);
		return op;
	}

protected:
	void ResolveTypes() override {
		types = spec.result_types;
	}
};

//===--------------------------------------------------------------------===//
// The optimizer: find eligible aggregates and swap them
//===--------------------------------------------------------------------===//
struct Candidate {
	string reason; // empty when eligible
	Spec spec;
	vector<unique_ptr<Expression>> inputs; // column references, in gathered order
	int64_t threshold = 0;
	int64_t input_rows = -1;
	string shape;
	string shape_class;
	int64_t measured_floor = -1; // -1: the class was not measured faster at any size
};

// max - min of a key column's statistics, or -1 when DuckDB has none.
static int64_t StatsSpan(const BaseStatistics &stats, Kind kind) {
	if (!NumericStats::HasMinMax(stats)) {
		return -1;
	}
	__int128 lo = 0, hi = 0;
	auto min = NumericStats::Min(stats), max = NumericStats::Max(stats);
	switch (kind) {
	case Kind::I8: lo = min.GetValueUnsafe<int8_t>(); hi = max.GetValueUnsafe<int8_t>(); break;
	case Kind::I16: lo = min.GetValueUnsafe<int16_t>(); hi = max.GetValueUnsafe<int16_t>(); break;
	case Kind::I32: lo = min.GetValueUnsafe<int32_t>(); hi = max.GetValueUnsafe<int32_t>(); break;
	case Kind::I64: lo = min.GetValueUnsafe<int64_t>(); hi = max.GetValueUnsafe<int64_t>(); break;
	case Kind::U8: lo = min.GetValueUnsafe<uint8_t>(); hi = max.GetValueUnsafe<uint8_t>(); break;
	case Kind::U16: lo = min.GetValueUnsafe<uint16_t>(); hi = max.GetValueUnsafe<uint16_t>(); break;
	case Kind::U32: lo = min.GetValueUnsafe<uint32_t>(); hi = max.GetValueUnsafe<uint32_t>(); break;
	default: return -1;
	}
	const __int128 span = hi - lo;
	return span < 0 || span > __int128(INT64_MAX) ? -1 : int64_t(span);
}

// Finds (or adds) the gathered input for a column binding; returns its index.
static idx_t InputFor(Candidate &c, BoundColumnRefExpression &ref, Kind kind, const LogicalType &type) {
	for (idx_t i = 0; i < c.inputs.size(); i++) {
		auto &existing = c.inputs[i]->Cast<BoundColumnRefExpression>();
		if (existing.binding == ref.binding) {
			return i;
		}
	}
	c.inputs.push_back(ref.Copy());
	c.spec.input_types.push_back(type);
	c.spec.input_kinds.push_back(kind);
	return c.inputs.size() - 1;
}

static int64_t CrossoverFor(AggOp op, bool grouped, bool string_key, bool many_groups) {
	if (grouped) {
		if (string_key) {
			return Crossovers::GROUP_UTF8;
		}
		return many_groups ? Crossovers::GROUP_100K : Crossovers::GROUP_1K;
	}
	switch (op) {
	case AggOp::SUM: return Crossovers::SUM;
	case AggOp::MIN: return Crossovers::MIN;
	case AggOp::MAX: return Crossovers::MAX;
	case AggOp::AVG: return Crossovers::MEAN;
	default: return 0; // a count needs no kernel
	}
}

static void Analyse(ClientContext &context, LogicalAggregate &aggr, Candidate &c) {
	auto &spec = c.spec;
	if (!aggr.grouping_functions.empty() || aggr.grouping_sets.size() > 1) {
		c.reason = "GROUPING SETS / ROLLUP / CUBE";
		return;
	}
	if (aggr.groups.size() > 1) {
		c.reason = "more than one GROUP BY column";
		return;
	}
	if (aggr.children.size() != 1) {
		c.reason = "unexpected plan shape";
		return;
	}

	// The key.
	bool string_key = false;
	if (aggr.groups.size() == 1) {
		auto &group = *aggr.groups[0];
		if (group.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			c.reason = "GROUP BY an expression rather than a column";
			return;
		}
		Kind kind;
		if (group.return_type.id() == LogicalTypeId::VARCHAR) {
			kind = Kind::STR;
			string_key = true;
		} else if (!NumericKind(group.return_type, kind)) {
			c.reason = "GROUP BY column of type " + group.return_type.ToString();
			return;
		}
		spec.has_key = true;
		InputFor(c, group.Cast<BoundColumnRefExpression>(), kind, group.return_type);
		spec.result_types.push_back(group.return_type);
		c.shape = "GROUP BY " + group.return_type.ToString() + ": ";
	}

	// The aggregates.
	bool any_kernel = false;
	for (auto &expr : aggr.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			c.reason = "non-aggregate expression";
			return;
		}
		auto &bound = expr->Cast<BoundAggregateExpression>();
		const string &fname = bound.function.name;
		if (bound.IsDistinct() || bound.filter || bound.order_bys) {
			c.reason = fname + " with DISTINCT, FILTER or ORDER BY";
			return;
		}
		AggSpec agg;
		agg.return_type = bound.return_type;
		agg.name = bound.GetName();
		if (fname == "count_star") {
			agg.op = AggOp::COUNT_STAR;
		} else if (fname == "count") {
			agg.op = AggOp::COUNT;
		} else if (fname == "sum" || fname == "sum_no_overflow") {
			agg.op = AggOp::SUM;
			agg.proven_no_overflow = fname == "sum_no_overflow";
		} else if (fname == "min") {
			agg.op = AggOp::MIN;
		} else if (fname == "max") {
			agg.op = AggOp::MAX;
		} else if (fname == "avg") {
			agg.op = AggOp::AVG;
		} else {
			c.reason = "aggregate " + fname;
			return;
		}
		if (agg.op != AggOp::COUNT_STAR) {
			if (bound.children.size() != 1) {
				c.reason = "aggregate " + fname + " with " + to_string(bound.children.size()) + " arguments";
				return;
			}
			if (bound.bind_info) {
				c.reason = "aggregate " + fname + " with bind data (DECIMAL)";
				return;
			}
			// The argument: a column, possibly under a cast that keeps every value.
			Expression *arg = bound.children[0].get();
			const LogicalType semantic = arg->return_type;
			if (arg->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
				auto &cast = arg->Cast<BoundCastExpression>();
				if (cast.try_cast || !ValuePreservingCast(cast.child->return_type, cast.return_type)) {
					c.reason = fname + " over a cast " + cast.child->return_type.ToString() + " -> " +
					           cast.return_type.ToString();
					return;
				}
				arg = cast.child.get();
			}
			if (arg->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				c.reason = fname + " over an expression rather than a column";
				return;
			}
			Kind kind;
			const LogicalType &source = arg->return_type;
			if (!NumericKind(source, kind)) {
				c.reason = fname + " over " + source.ToString();
				return;
			}
			if ((agg.op == AggOp::SUM || agg.op == AggOp::AVG) && IsTemporal(source)) {
				c.reason = fname + " over " + source.ToString();
				return;
			}
			if (agg.op == AggOp::SUM || agg.op == AggOp::AVG) {
				switch (semantic.InternalType()) {
				case PhysicalType::INT16: agg.semantic_bits = 16; break;
				case PhysicalType::INT32: agg.semantic_bits = 32; break;
				case PhysicalType::INT64: agg.semantic_bits = 64; break;
				default:
					c.reason = fname + " at " + semantic.ToString();
					return;
				}
				const auto rt = agg.return_type.InternalType();
				if ((agg.op == AggOp::SUM && rt != PhysicalType::INT128 && rt != PhysicalType::INT64) ||
				    (agg.op == AggOp::AVG && rt != PhysicalType::DOUBLE)) {
					c.reason = fname + " returning " + agg.return_type.ToString();
					return;
				}
			}
			if ((agg.op == AggOp::MIN || agg.op == AggOp::MAX) && agg.return_type != source &&
			    agg.return_type != semantic) {
				c.reason = fname + " returning " + agg.return_type.ToString();
				return;
			}
			if (agg.op == AggOp::COUNT && agg.return_type.id() != LogicalTypeId::BIGINT) {
				c.reason = "count returning " + agg.return_type.ToString();
				return;
			}
			agg.input = InputFor(c, arg->Cast<BoundColumnRefExpression>(), kind, source);
		}
		if (agg.op != AggOp::COUNT && agg.op != AggOp::COUNT_STAR) {
			any_kernel = true;
		}
		c.shape += (c.shape.empty() || c.shape.back() == ' ' ? "" : ", ") + agg.name;
		spec.aggs.push_back(agg);
		spec.result_types.push_back(agg.return_type);
	}
	if (!spec.has_key && !any_kernel) {
		c.reason = "only counts, which need no kernel";
		return;
	}
	if (spec.has_key && spec.input_kinds[0] == Kind::STR) {
		for (auto &agg : spec.aggs) {
			if (agg.input == 0 && agg.op != AggOp::COUNT) {
				c.reason = agg.name + " over the VARCHAR key";
				return;
			}
		}
	}

	// The input: projections and filters over one table function whose size DuckDB knows.
	LogicalOperator *node = aggr.children[0].get();
	while (node->type == LogicalOperatorType::LOGICAL_PROJECTION ||
	       node->type == LogicalOperatorType::LOGICAL_FILTER) {
		if (node->children.size() != 1) {
			c.reason = "unexpected plan shape";
			return;
		}
		node = node->children[0].get();
	}
	if (node->type != LogicalOperatorType::LOGICAL_GET) {
		c.reason = "input is a " + LogicalOperatorToString(node->type) + ", not a table scan";
		return;
	}
	auto &get = node->Cast<LogicalGet>();
	if (!get.function.cardinality) {
		c.reason = "the source " + get.function.name + " does not report its size";
		return;
	}
	auto stats = get.function.cardinality(context, get.bind_data.get());
	if (!stats || !stats->has_estimated_cardinality) {
		c.reason = "the source " + get.function.name + " does not report its size";
		return;
	}
	auto &child = *aggr.children[0];
	int64_t rows = child.has_estimated_cardinality ? int64_t(child.estimated_cardinality)
	                                               : int64_t(child.EstimateCardinality(context));
	c.input_rows = rows;
	c.shape = get.function.name + " -> " + c.shape;

	// The key's range from DuckDB's statistics, when it has them.
	int64_t span = -1;
	if (spec.has_key && !string_key && !aggr.group_stats.empty() && aggr.group_stats[0]) {
		span = StatsSpan(*aggr.group_stats[0], spec.input_kinds[0]);
	}
	// Streamed (see "Blocks"): an ungrouped aggregate, or a group-by whose integer key the statistics put
	// in a small range and whose MIN/MAX fit the fused group-by's 32-bit slots.
	bool wide_extreme = false;
	for (auto &agg : spec.aggs) {
		if ((agg.op == AggOp::MIN || agg.op == AggOp::MAX) && KindWidth(spec.input_kinds[agg.input]) > 4) {
			wide_extreme = true;
		}
	}
	spec.stream = !spec.has_key || (!string_key && spec.input_kinds[0] != Kind::U64 && span >= 0 &&
	                                span < STREAM_KEY_SPAN && !wide_extreme);
	// A streamed plan works on one block at a time and sums across blocks in 128 bits; the others hand
	// ArrowMetal the whole input as one array.
	if (!spec.stream && (rows > MAX_ROWS || int64_t(stats->estimated_cardinality) > MAX_ROWS)) {
		c.reason = "more rows than the GPU path takes (2^31 - 1)";
		return;
	}

	// DuckDB's estimate of the group count (from its distinct-count sketches, within about 25% on the
	// benchmark's tables) picks the router's 1,000-group or 100,000-group class, split at 10,000, the
	// geometric middle of the two.
	const idx_t groups_estimate =
	    aggr.has_estimated_cardinality ? aggr.estimated_cardinality : aggr.EstimateCardinality(context);
	const bool many_groups = spec.has_key && groups_estimate >= MANY_GROUPS;
	int64_t threshold = 0;
	for (auto &agg : spec.aggs) {
		threshold = MaxValue<int64_t>(threshold, CrossoverFor(agg.op, spec.has_key, string_key, many_groups));
	}
	c.threshold = threshold;

	// The measured half of the gate: the shape classes the provisional benchmark found faster than
	// DuckDB's own operators, and from how many rows (see `Measured`).
	idx_t kernels = 0;
	bool hugeint_state = false;
	for (auto &agg : spec.aggs) {
		if (agg.op == AggOp::SUM || agg.op == AggOp::AVG || agg.op == AggOp::MIN || agg.op == AggOp::MAX) {
			kernels++;
		}
		// DuckDB accumulates these in 128 bits: sum over INTEGER/BIGINT it could not prove fits in 64
		// bits, and avg over INTEGER/BIGINT always.
		if ((agg.op == AggOp::SUM && !agg.proven_no_overflow && agg.semantic_bits >= 32) ||
		    (agg.op == AggOp::AVG && agg.semantic_bits >= 32)) {
			hugeint_state = true;
		}
	}
	if (!spec.has_key) {
		if (hugeint_state || kernels >= 2) {
			c.shape_class = "ungrouped";
			c.measured_floor = Measured::UNGROUPED;
		} else {
			c.shape_class = "ungrouped, one aggregate with a 64-bit state";
		}
	} else if (string_key) {
		c.shape_class = "VARCHAR key";
	} else {
		const bool dense = spec.input_kinds[0] != Kind::U64 && span >= 0 && span < DENSE_KEY_SPAN;
		if (!dense && many_groups) {
			c.shape_class = "hash group-by, an estimated 10k or more groups";
			c.measured_floor = Measured::HASH_MANY_GROUPS;
		} else if (!dense) {
			c.shape_class = "hash group-by, an estimated fewer than 10k groups";
		} else if (many_groups) {
			c.shape_class = "fused group-by, an estimated 10k or more groups";
			c.measured_floor = Measured::DENSE_MANY_GROUPS;
		} else if (kernels >= 3) {
			c.shape_class = "fused group-by, fewer groups, three or more aggregates";
			c.measured_floor = Measured::DENSE_FEW_GROUPS;
		} else {
			c.shape_class = "fused group-by, fewer groups, one or two aggregates";
		}
	}
}

// SET arrowmetal_rewrite_block_rows: rows per block of a streamed plan, rounded to a multiple of 64.
static idx_t BlockRows(ClientContext &context) {
	Value value;
	int64_t rows = DEFAULT_BLOCK_ROWS;
	if (context.TryGetCurrentSetting("arrowmetal_rewrite_block_rows", value) && !value.IsNull()) {
		rows = value.GetValue<int64_t>();
	}
	rows = MaxValue<int64_t>(rows, 2048);
	rows = MinValue<int64_t>(rows, int64_t(1) << 30);
	return idx_t(rows + 63) / 64 * 64;
}

static void VisitPlan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &op, Mode mode) {
	for (auto &child : op->children) {
		VisitPlan(input, child, mode);
	}
	if (op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return;
	}
	auto &aggr = op->Cast<LogicalAggregate>();
	Candidate c;
	Analyse(input.context, aggr, c);
	Decision d;
	d.shape = c.shape;
	d.input_rows = c.input_rows;
	// The auto threshold: the router's crossover and the class's measured floor, whichever is larger;
	// NULL in the log when the class has no measured floor.
	d.threshold_rows = c.measured_floor < 0 ? -1 : MaxValue<int64_t>(c.threshold, c.measured_floor);
	if (c.reason.empty() && mode == Mode::AUTO) {
		if (c.measured_floor < 0) {
			c.reason = "not measured faster than DuckDB: " + c.shape_class;
		} else if (c.input_rows < c.threshold) {
			c.reason = "below the crossover: " + c.shape_class;
		} else if (c.input_rows < c.measured_floor) {
			c.reason = "below the measured floor: " + c.shape_class;
		}
	}
	if (!c.reason.empty()) {
		d.decision = "kept";
		d.reason = c.reason;
		Record(d);
		return;
	}
	d.decision = "rewritten";
	d.reason = mode == Mode::FORCE ? "forced" : "at or above the threshold: " + c.shape_class;
	c.spec.block_rows = BlockRows(input.context);
	c.spec.decision_id = Record(d);

	// Capacity: the most rows the source can hand over, so the sink rarely has to grow.
	idx_t capacity = 0;
	{
		LogicalOperator *node = aggr.children[0].get();
		while (!node->children.empty()) {
			node = node->children[0].get();
		}
		auto &get = node->Cast<LogicalGet>();
		auto stats = get.function.cardinality(input.context, get.bind_data.get());
		capacity = stats->has_max_cardinality ? stats->max_cardinality : stats->estimated_cardinality;
	}
	auto replacement =
	    make_uniq<LogicalArrowMetalAggregate>(std::move(c.spec), aggr.group_index, aggr.aggregate_index, capacity);
	replacement->expressions = std::move(c.inputs);
	replacement->children = std::move(aggr.children);
	replacement->has_estimated_cardinality = aggr.has_estimated_cardinality;
	replacement->estimated_cardinality = aggr.estimated_cardinality;
	replacement->ResolveOperatorTypes();
	op = std::move(replacement);
}

static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	const Mode mode = GetMode(input.context);
	if (mode == Mode::OFF) {
		return;
	}
	VisitPlan(input, plan, mode);
}

//===--------------------------------------------------------------------===//
// arrowmetal_rewrites(): the decision log
//===--------------------------------------------------------------------===//
struct LogState : public GlobalTableFunctionState {
	vector<Decision> rows;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> LogBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &types,
                                        vector<string> &names) {
	names = {"id", "decision", "reason", "shape", "input_rows", "threshold_rows",
	         "path", "rows_seen", "groups", "gpu_ms"};
	types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	         LogicalType::BIGINT, LogicalType::BIGINT,  LogicalType::VARCHAR, LogicalType::BIGINT,
	         LogicalType::BIGINT, LogicalType::DOUBLE};
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> LogInit(ClientContext &, TableFunctionInitInput &) {
	auto state = make_uniq<LogState>();
	std::lock_guard<std::mutex> guard(g_log_lock);
	state->rows.assign(g_log.begin(), g_log.end());
	return std::move(state);
}

static void LogScan(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LogState>();
	idx_t n = 0;
	while (state.offset < state.rows.size() && n < STANDARD_VECTOR_SIZE) {
		auto &d = state.rows[state.offset++];
		output.SetValue(0, n, Value::BIGINT(d.id));
		output.SetValue(1, n, Value(d.decision));
		output.SetValue(2, n, Value(d.reason));
		output.SetValue(3, n, Value(d.shape));
		output.SetValue(4, n, d.input_rows < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.input_rows));
		output.SetValue(5, n, d.threshold_rows < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.threshold_rows));
		output.SetValue(6, n, d.path.empty() ? Value(LogicalType::VARCHAR) : Value(d.path));
		output.SetValue(7, n, d.rows_seen < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.rows_seen));
		output.SetValue(8, n, d.groups < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(d.groups));
		output.SetValue(9, n, d.gpu_ms < 0 ? Value(LogicalType::DOUBLE) : Value::DOUBLE(d.gpu_ms));
		n++;
	}
	output.SetCardinality(n);
}

static void Load(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto &config = DBConfig::GetConfig(db);
	config.AddExtensionOption("arrowmetal_rewrite",
	                          "ArrowMetal aggregate rewrite: 'auto' (at or above the crossover), 'off', or 'force'",
	                          LogicalType::VARCHAR, Value("auto"));
	config.AddExtensionOption("arrowmetal_rewrite_block_rows",
	                          "ArrowMetal aggregate rewrite: rows per block when a plan is streamed to the GPU",
	                          LogicalType::BIGINT, Value::BIGINT(DEFAULT_BLOCK_ROWS));
	OptimizerExtension extension;
	extension.optimize_function = Optimize;
	OptimizerExtension::Register(config, extension);

	TableFunction log("arrowmetal_rewrites", {}, LogScan, LogBind, LogInit);
	loader.RegisterFunction(log);
	loader.SetDescription("ArrowMetal: runs eligible aggregates on the Metal GPU");
}

} // namespace arrowmetal_rewrite
} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(arrowmetal_rewrite, loader) {
	duckdb::arrowmetal_rewrite::Load(loader);
}
}
