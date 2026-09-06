// ArrowMetal as a loadable DuckDB extension.
//
// Built against DuckDB's *C* extension API (duckdb_extension.h), so it needs no DuckDB source tree
// and no C++ ABI match with the host: the loader hands the extension a struct of function pointers
// and everything after that is plain C. The only other dependency is libArrowMetalC.dylib, this
// repository's own C ABI (include/arrowmetal.h).
//
// What it registers (see docs/DUCKDB.md):
//
//   arrowmetal_version()                          -> VARCHAR   library version
//   arrowmetal_device()                           -> VARCHAR   the GPU it will run on
//   arrowmetal_agg(table, column)                 -> one row: sum, count, min, max, mean
//   arrowmetal_group_by(table, key, value)        -> one row per key: key, count, sum, min, max
//   arrowmetal_top_k(table, column, k)            -> the k largest values, descending
//   arrowmetal_sort(table, column)                -> every value, ascending, nulls last
//   arrowmetal_query(table, expr)                 -> one row per aggregate of a fused am_query
//
// The table functions take a table (or view) NAME rather than a subquery: DuckDB's C table-function
// API has no way to accept a relation, so the extension issues `SELECT <column> FROM <table>` on its
// own connection, assembles the result into one contiguous Arrow buffer, and imports that.
//
// WHY THE ASSEMBLY STEP: DuckDB hands results out one DataChunk (2048 rows) at a time. A 2048-row
// GPU dispatch is all latency and no work, so the extension concatenates the chunks into a single
// column first - one memcpy pass at memory bandwidth - and dispatches once over the whole column.
// The Python bridge (python/arrowmetal/duckdb_bridge.py) avoids even that copy, because pyarrow
// hands over a whole table's buffers at once and ArrowMetal wraps them in place. If you care about
// the last copy, use the bridge; if you care about staying inside SQL, use this.

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#define DUCKDB_EXTENSION_NAME arrowmetal
#include "duckdb_extension.h"

#include "arrowmetal.h"

// Every duckdb_* name in this file is a macro that reads a function pointer out of `duckdb_ext_api`.
// DUCKDB_EXTENSION_ENTRYPOINT at the bottom defines that struct; this declares it for the code above.
DUCKDB_EXTENSION_EXTERN

//===--------------------------------------------------------------------===//
// The extension's own connection, opened once when the extension loads.
//
// A table function's bind and init callbacks get no connection and no way to run SQL, so the
// extension needs one of its own to read the table it was pointed at. It cannot keep the
// `duckdb_database` and connect on demand instead: the handle that
// `duckdb_extension_access::get_database` returns belongs to the loader and does not outlive the
// load call, and connecting through it afterwards fails. So the connection is made while that
// handle is still good and kept for the life of the process, serialised by a mutex because one
// DuckDB connection may not be used from two threads at once. Serialising costs nothing here:
// there is one GPU, and the scans would queue on it anyway.
//===--------------------------------------------------------------------===//
static duckdb_connection g_connection = nullptr;
static std::mutex g_connection_lock;

namespace {

std::string am_error() {
	const char *msg = am_last_error();
	return msg && *msg ? std::string(msg) : std::string("unknown ArrowMetal error");
}

//===--------------------------------------------------------------------===//
// One contiguous Arrow column, assembled from DuckDB DataChunks.
//
// For every fixed-width type DuckDB and Arrow agree on the layout exactly - a packed array of
// values plus a little-endian validity bitmap - so assembly is a memcpy per chunk and a bit copy
// per chunk, with no conversion at all.
//===--------------------------------------------------------------------===//
struct ArrowBuffers {
	std::vector<uint8_t> values;
	std::vector<uint64_t> validity;
	const void *pointers[2] = {nullptr, nullptr};
};

void release_array(ArrowArray *array) {
	delete static_cast<ArrowBuffers *>(array->private_data);
	array->release = nullptr;
}

void release_schema(ArrowSchema *schema) {
	schema->release = nullptr;
}

// Arrow format string and element width for the DuckDB types this extension computes on.
// Anything else is refused by name at bind time rather than silently mis-read here.
bool arrow_format_for(duckdb_type type, const char **format, idx_t *width, bool *is_float) {
	*is_float = false;
	switch (type) {
	case DUCKDB_TYPE_TINYINT:   *format = "c"; *width = 1; return true;
	case DUCKDB_TYPE_SMALLINT:  *format = "s"; *width = 2; return true;
	case DUCKDB_TYPE_INTEGER:   *format = "i"; *width = 4; return true;
	case DUCKDB_TYPE_BIGINT:    *format = "l"; *width = 8; return true;
	case DUCKDB_TYPE_UTINYINT:  *format = "C"; *width = 1; return true;
	case DUCKDB_TYPE_USMALLINT: *format = "S"; *width = 2; return true;
	case DUCKDB_TYPE_UINTEGER:  *format = "I"; *width = 4; return true;
	case DUCKDB_TYPE_UBIGINT:   *format = "L"; *width = 8; return true;
	case DUCKDB_TYPE_FLOAT:     *format = "f"; *width = 4; *is_float = true; return true;
	case DUCKDB_TYPE_DOUBLE:    *format = "g"; *width = 8; *is_float = true; return true;
	case DUCKDB_TYPE_DATE:      *format = "tdD"; *width = 4; return true;
	case DUCKDB_TYPE_TIMESTAMP: *format = "tsu:"; *width = 8; return true;
	default: return false;
	}
}

void set_bit(std::vector<uint64_t> &bitmap, int64_t index, bool valid) {
	uint64_t &word = bitmap[static_cast<size_t>(index >> 6)];
	const uint64_t mask = uint64_t(1) << (index & 63);
	if (valid) {
		word |= mask;
	} else {
		word &= ~mask;
	}
}

// Hands one assembled Arrow buffer set to ArrowMetal, which takes ownership of it.
am_array *import_buffers(std::unique_ptr<ArrowBuffers> buffers, const char *format, int64_t rows,
                         int64_t nulls, std::string &error) {
	if (buffers->validity.empty()) {
		buffers->validity.resize(1, ~uint64_t(0));
	}
	buffers->pointers[0] = nulls ? static_cast<const void *>(buffers->validity.data()) : nullptr;
	buffers->pointers[1] = buffers->values.data();

	ArrowSchema schema;
	std::memset(&schema, 0, sizeof(schema));
	schema.format = format;
	schema.name = "";
	schema.release = release_schema;

	ArrowArray array;
	std::memset(&array, 0, sizeof(array));
	array.length = rows;
	array.null_count = nulls;
	array.n_buffers = 2;
	array.buffers = const_cast<const void **>(buffers->pointers);
	array.private_data = buffers.get();
	array.release = release_array;

	am_array *out = nullptr;
	// am_import takes the ArrowArray: on success ArrowMetal owns `buffers` and releases it, on
	// failure the array was not moved and the unique_ptr still owns it.
	const int rc = am_import(&schema, &array, &out);
	if (schema.release) {
		schema.release(&schema);
	}
	if (rc != 0) {
		error = "arrowmetal: " + am_error();
		return nullptr;
	}
	buffers.release();
	return out;
}

// Runs `sql` and lifts ALL of its columns onto the GPU in one scan.
//
// Pulling several columns together matters: the scan is the expensive half of everything this
// extension does, and a group-by that fetched its key and its values separately would pay for the
// table twice. On failure the vector comes back empty and `error` says why.
std::vector<am_array *> pull_columns(duckdb_connection conn, const std::string &sql,
                                     std::string &error) {
	std::vector<am_array *> out;
	duckdb_result result;
	if (duckdb_query(conn, sql.c_str(), &result) == DuckDBError) {
		const char *msg = duckdb_result_error(&result);
		error = msg ? msg : "query failed";
		duckdb_destroy_result(&result);
		return out;
	}

	const idx_t column_count = duckdb_column_count(&result);
	std::vector<const char *> formats(column_count, nullptr);
	std::vector<idx_t> widths(column_count, 0);
	for (idx_t c = 0; c < column_count; c++) {
		bool is_float = false;
		if (!arrow_format_for(duckdb_column_type(&result, c), &formats[c], &widths[c], &is_float)) {
			error = std::string("arrowmetal: column ") + duckdb_column_name(&result, c) +
			        " has a type the GPU path does not handle; the Python bridge (docs/DUCKDB.md) "
			        "covers strings, decimals and nested types";
			duckdb_destroy_result(&result);
			return out;
		}
	}

	std::vector<std::unique_ptr<ArrowBuffers>> buffers;
	std::vector<int64_t> nulls(column_count, 0);
	for (idx_t c = 0; c < column_count; c++) {
		buffers.emplace_back(new ArrowBuffers());
	}

	int64_t rows = 0;
	while (true) {
		duckdb_data_chunk chunk = duckdb_fetch_chunk(result);
		if (!chunk) {
			break;
		}
		const idx_t count = duckdb_data_chunk_get_size(chunk);
		if (count == 0) {
			duckdb_destroy_data_chunk(&chunk);
			continue;
		}
		for (idx_t c = 0; c < column_count; c++) {
			duckdb_vector vector = duckdb_data_chunk_get_vector(chunk, c);
			const uint8_t *data = static_cast<const uint8_t *>(duckdb_vector_get_data(vector));
			const uint64_t *valid = duckdb_vector_get_validity(vector);
			ArrowBuffers &target = *buffers[c];
			const idx_t width = widths[c];

			target.values.resize(static_cast<size_t>((rows + int64_t(count)) * int64_t(width)));
			std::memcpy(target.values.data() + static_cast<size_t>(rows) * width, data,
			            static_cast<size_t>(count) * width);

			// Grow the validity bitmap to cover the new rows, defaulting to valid.
			target.validity.resize(static_cast<size_t>((rows + int64_t(count) + 63) / 64), ~uint64_t(0));
			if (valid) {
				for (idx_t i = 0; i < count; i++) {
					if (!((valid[i / 64] >> (i % 64)) & 1u)) {
						set_bit(target.validity, rows + int64_t(i), false);
						nulls[c]++;
					}
				}
			}
		}
		rows += int64_t(count);
		duckdb_destroy_data_chunk(&chunk);
	}
	duckdb_destroy_result(&result);

	for (idx_t c = 0; c < column_count; c++) {
		am_array *column = import_buffers(std::move(buffers[c]), formats[c], rows, nulls[c], error);
		if (!column) {
			for (am_array *done : out) {
				am_release(done);
			}
			out.clear();
			return out;
		}
		out.push_back(column);
	}
	return out;
}

// The single-column case, which is most of them.
am_array *pull_column(duckdb_connection conn, const std::string &sql, std::string &error) {
	std::vector<am_array *> columns = pull_columns(conn, sql, error);
	return columns.empty() ? nullptr : columns[0];
}

// A scalar coming back from am_reduce / am_query, kept in the widest form of each kind so that an
// exact int64 total is never squeezed through a double on the way out.
struct Scalar {
	bool is_null = true;
	bool is_int = true;
	int64_t i = 0;
	double d = 0.0;

	double as_double() const { return is_int ? static_cast<double>(i) : d; }
};

Scalar reduce(am_array *column, int op, std::string &error) {
	Scalar out;
	int64_t iv = 0;
	double dv = 0.0;
	int kind = 0;
	int is_null = 0;
	if (am_reduce(column, op, &iv, &dv, &kind, &is_null) != 0) {
		error = "arrowmetal: " + am_error();
		return out;
	}
	out.is_null = is_null != 0;
	out.is_int = kind != 2;
	out.i = iv;
	out.d = dv;
	return out;
}

// Exports a MetalArray and reads it back as host-side scalars. Only the fixed-width formats this
// extension can produce ever reach here.
bool read_column(am_array *column, std::vector<Scalar> &out, std::string &error) {
	ArrowSchema schema;
	ArrowArray array;
	std::memset(&schema, 0, sizeof(schema));
	std::memset(&array, 0, sizeof(array));
	if (am_export(column, &schema, &array) != 0) {
		error = "arrowmetal: " + am_error();
		return false;
	}
	const std::string format = schema.format ? schema.format : "";
	const uint64_t *valid = array.n_buffers > 0 ? static_cast<const uint64_t *>(array.buffers[0]) : nullptr;
	const void *data = array.n_buffers > 1 ? array.buffers[1] : nullptr;
	const int64_t offset = array.offset;
	out.resize(static_cast<size_t>(array.length));
	bool ok = true;
	for (int64_t i = 0; i < array.length && ok; i++) {
		Scalar &s = out[static_cast<size_t>(i)];
		const int64_t row = i + offset;
		if (valid && !((valid[row / 64] >> (row % 64)) & 1u)) {
			s.is_null = true;
			continue;
		}
		s.is_null = false;
		s.is_int = true;
		if (format == "c") {
			s.i = static_cast<const int8_t *>(data)[row];
		} else if (format == "C") {
			s.i = static_cast<const uint8_t *>(data)[row];
		} else if (format == "s") {
			s.i = static_cast<const int16_t *>(data)[row];
		} else if (format == "S") {
			s.i = static_cast<const uint16_t *>(data)[row];
		} else if (format == "i" || format == "tdD") {
			s.i = static_cast<const int32_t *>(data)[row];
		} else if (format == "I") {
			s.i = static_cast<const uint32_t *>(data)[row];
		} else if (format == "l" || format == "L" || format.rfind("ts", 0) == 0) {
			s.i = static_cast<const int64_t *>(data)[row];
		} else if (format == "f") {
			s.is_int = false;
			s.d = static_cast<const float *>(data)[row];
		} else if (format == "g") {
			s.is_int = false;
			s.d = static_cast<const double *>(data)[row];
		} else if (format == "b") {
			s.i = (static_cast<const uint8_t *>(data)[row / 8] >> (row % 8)) & 1u;
		} else {
			error = "arrowmetal: cannot return a column of Arrow type '" + format + "' through SQL";
			ok = false;
		}
	}
	if (array.release) {
		array.release(&array);
	}
	if (schema.release) {
		schema.release(&schema);
	}
	return ok;
}

//===--------------------------------------------------------------------===//
// Table function plumbing
//
// Every table function here shares the same shape: bind decides the output columns, init does the
// whole GPU computation into host vectors, and main pages those vectors out 2048 rows at a time.
// Doing the work in init keeps the GPU dispatch count at one per query rather than one per chunk.
//===--------------------------------------------------------------------===//
enum class Kind { Agg, GroupBy, TopK, Sort, Query };

struct BindData {
	Kind kind = Kind::Agg;
	std::string table;
	std::string column;
	std::string key;
	std::string expr;
	int64_t k = 0;
	bool value_is_float = false;
};

struct InitData {
	// Column-major result. `names` is used only by arrowmetal_query.
	std::vector<std::string> names;
	std::vector<std::vector<Scalar>> columns;
	int64_t emitted = 0;
	int64_t rows = 0;
	std::string error;
};

void destroy_bind(void *data) { delete static_cast<BindData *>(data); }
void destroy_init(void *data) { delete static_cast<InitData *>(data); }

std::string parameter_string(duckdb_bind_info info, idx_t index) {
	duckdb_value value = duckdb_bind_get_parameter(info, index);
	char *text = duckdb_get_varchar(value);
	std::string out = text ? text : "";
	duckdb_free(text);
	duckdb_destroy_value(&value);
	return out;
}

int64_t parameter_int(duckdb_bind_info info, idx_t index) {
	duckdb_value value = duckdb_bind_get_parameter(info, index);
	const int64_t out = duckdb_get_int64(value);
	duckdb_destroy_value(&value);
	return out;
}

void add_column(duckdb_bind_info info, const char *name, duckdb_type type) {
	duckdb_logical_type logical = duckdb_create_logical_type(type);
	duckdb_bind_add_result_column(info, name, logical);
	duckdb_destroy_logical_type(&logical);
}

// A quoted SQL identifier: `"my table"`, with any embedded quote doubled.
std::string quote(const std::string &identifier) {
	std::string out = "\"";
	for (char c : identifier) {
		if (c == '"') {
			out += '"';
		}
		out += c;
	}
	out += '"';
	return out;
}

// Is the named column of the named table a floating-point column? Decides whether sums and extremes
// come back as BIGINT or DOUBLE, so it has to be settled at bind time.
bool column_is_float(duckdb_connection conn, const std::string &table, const std::string &column,
                     bool &found, std::string &error) {
	found = false;
	const std::string sql = "SELECT " + quote(column) + " FROM " + quote(table) + " LIMIT 0";
	duckdb_result result;
	if (duckdb_query(conn, sql.c_str(), &result) == DuckDBError) {
		const char *msg = duckdb_result_error(&result);
		error = msg ? msg : "query failed";
		duckdb_destroy_result(&result);
		return false;
	}
	const duckdb_type type = duckdb_column_type(&result, 0);
	duckdb_destroy_result(&result);
	const char *format = nullptr;
	idx_t width = 0;
	bool is_float = false;
	if (!arrow_format_for(type, &format, &width, &is_float)) {
		error = "arrowmetal: column " + column + " has a type the GPU path does not handle here; "
		        "the Python bridge in docs/DUCKDB.md covers strings, decimals and nested types";
		return false;
	}
	found = true;
	return is_float;
}

// Borrows the extension's connection for as long as it is in scope.
struct Connection {
	duckdb_connection handle;
	std::unique_lock<std::mutex> guard;

	Connection() : handle(g_connection), guard(g_connection_lock) {}
	Connection(const Connection &) = delete;
	Connection &operator=(const Connection &) = delete;
};

struct Column {
	am_array *handle = nullptr;
	~Column() {
		if (handle) {
			am_release(handle);
		}
	}
	Column() = default;
	Column(const Column &) = delete;
	Column &operator=(const Column &) = delete;
	explicit operator bool() const { return handle != nullptr; }
};

//===--------------------------------------------------------------------===//
// bind
//===--------------------------------------------------------------------===//
template <Kind KIND>
void bind(duckdb_bind_info info) {
	auto data = std::unique_ptr<BindData>(new BindData());
	data->kind = KIND;
	data->table = parameter_string(info, 0);

	Connection conn;
	if (!conn.handle) {
		duckdb_bind_set_error(info, "arrowmetal: the extension has no connection to the database");
		return;
	}

	std::string error;
	bool found = false;
	if (KIND == Kind::GroupBy) {
		data->key = parameter_string(info, 1);
		data->column = parameter_string(info, 2);
	} else if (KIND == Kind::Query) {
		data->expr = parameter_string(info, 1);
	} else {
		data->column = parameter_string(info, 1);
		if (KIND == Kind::TopK) {
			data->k = parameter_int(info, 2);
			if (data->k < 0) {
				duckdb_bind_set_error(info, "arrowmetal_top_k: k must not be negative");
				return;
			}
		}
	}

	if (KIND != Kind::Query) {
		data->value_is_float = column_is_float(conn.handle, data->table, data->column, found, error);
		if (!found) {
			duckdb_bind_set_error(info, error.c_str());
			return;
		}
	}
	const duckdb_type value_type = data->value_is_float ? DUCKDB_TYPE_DOUBLE : DUCKDB_TYPE_BIGINT;

	switch (KIND) {
	case Kind::Agg:
		add_column(info, "sum", value_type);
		add_column(info, "count", DUCKDB_TYPE_BIGINT);
		add_column(info, "min", value_type);
		add_column(info, "max", value_type);
		add_column(info, "mean", DUCKDB_TYPE_DOUBLE);
		duckdb_bind_set_cardinality(info, 1, true);
		break;
	case Kind::GroupBy: {
		bool key_found = false;
		std::string key_error;
		const bool key_is_float = column_is_float(conn.handle, data->table, data->key, key_found, key_error);
		if (!key_found) {
			duckdb_bind_set_error(info, key_error.c_str());
			return;
		}
		add_column(info, "key", key_is_float ? DUCKDB_TYPE_DOUBLE : DUCKDB_TYPE_BIGINT);
		add_column(info, "count", DUCKDB_TYPE_BIGINT);
		add_column(info, "sum", value_type);
		add_column(info, "min", value_type);
		add_column(info, "max", value_type);
		break;
	}
	case Kind::TopK:
	case Kind::Sort:
		add_column(info, "value", value_type);
		break;
	case Kind::Query:
		add_column(info, "name", DUCKDB_TYPE_VARCHAR);
		add_column(info, "value", DUCKDB_TYPE_DOUBLE);
		add_column(info, "exact", DUCKDB_TYPE_BIGINT);
		break;
	}
	duckdb_bind_set_bind_data(info, data.release(), destroy_bind);
}

//===--------------------------------------------------------------------===//
// init: the whole computation, one GPU dispatch chain per query
//===--------------------------------------------------------------------===//
void run_agg(duckdb_connection conn, const BindData &bind_data, InitData &init) {
	Column column;
	column.handle = pull_column(conn, "SELECT " + quote(bind_data.column) + " FROM " +
	                                      quote(bind_data.table), init.error);
	if (!column) {
		return;
	}
	const Scalar total = reduce(column.handle, 0, init.error);
	const Scalar low = reduce(column.handle, 1, init.error);
	const Scalar high = reduce(column.handle, 2, init.error);
	const Scalar mean = reduce(column.handle, 3, init.error);
	if (!init.error.empty()) {
		return;
	}
	Scalar count;
	count.is_null = false;
	count.is_int = true;
	count.i = am_length(column.handle) - am_null_count(column.handle);

	init.columns = {{total}, {count}, {low}, {high}, {mean}};
	init.rows = 1;
}

void run_group_by(duckdb_connection conn, const BindData &bind_data, InitData &init) {
	// One scan for both columns: reading the table twice would cost more than the group-by does.
	std::vector<am_array *> pulled = pull_columns(
	    conn, "SELECT " + quote(bind_data.key) + ", " + quote(bind_data.column) + " FROM " +
	              quote(bind_data.table), init.error);
	if (pulled.size() < 2) {
		for (am_array *handle : pulled) {
			am_release(handle);
		}
		return;
	}
	Column keys, values;
	keys.handle = pulled[0];
	values.handle = pulled[1];

	am_array *key_columns[1] = {keys.handle};
	am_groupby *gb = nullptr;
	if (am_group_by_keys(key_columns, 1, &gb) != 0) {
		init.error = "arrowmetal: " + am_error();
		return;
	}

	// op numbering is the C ABI contract (include/arrowmetal.h): 0 sum, 2 count, 4 min, 5 max.
	struct Output {
		am_array *handle = nullptr;
	} key_out, count_out, sum_out, min_out, max_out;
	bool ok = am_group_by_keys_result(gb, 0, &key_out.handle) == 0 &&
	          am_group_agg_ex(gb, values.handle, 2, 0.0, &count_out.handle) == 0 &&
	          am_group_agg_ex(gb, values.handle, 0, 0.0, &sum_out.handle) == 0 &&
	          am_group_agg_ex(gb, values.handle, 4, 0.0, &min_out.handle) == 0 &&
	          am_group_agg_ex(gb, values.handle, 5, 0.0, &max_out.handle) == 0;
	if (!ok) {
		init.error = "arrowmetal: " + am_error();
	} else {
		init.columns.resize(5);
		am_array *handles[5] = {key_out.handle, count_out.handle, sum_out.handle, min_out.handle,
		                        max_out.handle};
		for (int i = 0; i < 5 && init.error.empty(); i++) {
			read_column(handles[i], init.columns[static_cast<size_t>(i)], init.error);
		}
		init.rows = init.columns[0].empty() ? 0 : static_cast<int64_t>(init.columns[0].size());
	}
	for (am_array *handle : {key_out.handle, count_out.handle, sum_out.handle, min_out.handle,
	                         max_out.handle}) {
		if (handle) {
			am_release(handle);
		}
	}
	am_group_by_release(gb);
}

void run_top_k(duckdb_connection conn, const BindData &bind_data, InitData &init) {
	Column column;
	column.handle = pull_column(conn, "SELECT " + quote(bind_data.column) + " FROM " +
	                                      quote(bind_data.table), init.error);
	if (!column) {
		return;
	}
	Column indices, values;
	if (am_top_k(column.handle, bind_data.k, 1, &indices.handle) != 0 ||
	    am_take(column.handle, indices.handle, &values.handle) != 0) {
		init.error = "arrowmetal: " + am_error();
		return;
	}
	init.columns.resize(1);
	if (read_column(values.handle, init.columns[0], init.error)) {
		init.rows = static_cast<int64_t>(init.columns[0].size());
	}
}

void run_sort(duckdb_connection conn, const BindData &bind_data, InitData &init) {
	Column column;
	column.handle = pull_column(conn, "SELECT " + quote(bind_data.column) + " FROM " +
	                                      quote(bind_data.table), init.error);
	if (!column) {
		return;
	}
	Column sorted;
	if (am_sort(column.handle, 0, &sorted.handle) != 0) {
		init.error = "arrowmetal: " + am_error();
		return;
	}
	init.columns.resize(1);
	if (read_column(sorted.handle, init.columns[0], init.error)) {
		init.rows = static_cast<int64_t>(init.columns[0].size());
	}
}

void run_query(duckdb_connection conn, const BindData &bind_data, InitData &init) {
	// Every column of the table that the GPU path can read becomes a named input to am_query; the
	// expression text decides which of them are actually touched.
	duckdb_result probe;
	const std::string probe_sql = "SELECT * FROM " + quote(bind_data.table) + " LIMIT 0";
	if (duckdb_query(conn, probe_sql.c_str(), &probe) == DuckDBError) {
		const char *msg = duckdb_result_error(&probe);
		init.error = msg ? msg : "query failed";
		duckdb_destroy_result(&probe);
		return;
	}
	std::vector<std::string> names;
	for (idx_t i = 0; i < duckdb_column_count(&probe); i++) {
		const char *format = nullptr;
		idx_t width = 0;
		bool is_float = false;
		if (arrow_format_for(duckdb_column_type(&probe, i), &format, &width, &is_float)) {
			names.push_back(duckdb_column_name(&probe, i));
		}
	}
	duckdb_destroy_result(&probe);
	if (names.empty()) {
		init.error = "arrowmetal_query: the table has no column the GPU path can read";
		return;
	}

	// All of them in one scan, so a five-column expression does not read the table five times.
	std::string projection;
	for (size_t i = 0; i < names.size(); i++) {
		projection += (i ? ", " : "") + quote(names[i]);
	}
	std::vector<am_array *> pulled = pull_columns(
	    conn, "SELECT " + projection + " FROM " + quote(bind_data.table), init.error);
	if (pulled.size() != names.size()) {
		for (am_array *handle : pulled) {
			am_release(handle);
		}
		return;
	}
	std::vector<Column> columns(names.size());
	std::vector<am_array *> handles;
	std::vector<const char *> name_pointers;
	for (size_t i = 0; i < names.size(); i++) {
		columns[i].handle = pulled[i];
		handles.push_back(pulled[i]);
		name_pointers.push_back(names[i].c_str());
	}

	am_query_result *result = nullptr;
	if (am_query(handles.data(), name_pointers.data(), static_cast<int64_t>(handles.size()),
	             bind_data.expr.c_str(), &result) != 0) {
		init.error = "arrowmetal: " + am_error();
		return;
	}
	if (am_query_column_count(result) > 0) {
		init.error = "arrowmetal_query: this table function returns aggregates; a query with a "
		             "(project ...) or (group_by ...) terminal produces columns, which SQL cannot "
		             "receive here - use arrowmetal_group_by, or the Python bridge";
		am_query_result_release(result);
		return;
	}
	const int64_t count = am_query_scalar_count(result);
	init.columns.resize(3);
	init.names.reserve(static_cast<size_t>(count));
	for (int64_t i = 0; i < count; i++) {
		const char *name = am_query_scalar_name(result, i);
		init.names.push_back(name ? name : "");
		int64_t iv = 0;
		double dv = 0.0;
		int kind = 0;
		int is_null = 0;
		if (am_query_scalar(result, i, &iv, &dv, &kind, &is_null) != 0) {
			init.error = "arrowmetal: " + am_error();
			break;
		}
		Scalar name_slot;   // the VARCHAR column is served from init.names, this is just a placeholder
		name_slot.is_null = false;
		Scalar value;
		value.is_null = is_null != 0;
		value.is_int = false;
		value.d = kind == 2 ? dv : static_cast<double>(iv);
		Scalar exact;
		// `exact` carries the integer answer undamaged: a sum past 2^53 is exact here and rounded
		// in `value`, which is the whole reason both columns exist.
		exact.is_null = is_null != 0 || kind == 2;
		exact.is_int = true;
		exact.i = iv;
		init.columns[0].push_back(name_slot);
		init.columns[1].push_back(value);
		init.columns[2].push_back(exact);
	}
	init.rows = static_cast<int64_t>(init.names.size());
	am_query_result_release(result);
}

void init_function(duckdb_init_info info) {
	auto *bind_data = static_cast<BindData *>(duckdb_init_get_bind_data(info));
	auto init = std::unique_ptr<InitData>(new InitData());

	Connection conn;
	if (!conn.handle) {
		init->error = "arrowmetal: could not open a connection to the database";
	} else {
		switch (bind_data->kind) {
		case Kind::Agg:     run_agg(conn.handle, *bind_data, *init); break;
		case Kind::GroupBy: run_group_by(conn.handle, *bind_data, *init); break;
		case Kind::TopK:    run_top_k(conn.handle, *bind_data, *init); break;
		case Kind::Sort:    run_sort(conn.handle, *bind_data, *init); break;
		case Kind::Query:   run_query(conn.handle, *bind_data, *init); break;
		}
	}
	if (!init->error.empty()) {
		duckdb_init_set_error(info, init->error.c_str());
	}
	duckdb_init_set_init_data(info, init.release(), destroy_init);
}

//===--------------------------------------------------------------------===//
// main: page the computed result out
//===--------------------------------------------------------------------===//
void main_function(duckdb_function_info info, duckdb_data_chunk output) {
	auto *bind_data = static_cast<BindData *>(duckdb_function_get_bind_data(info));
	auto *init = static_cast<InitData *>(duckdb_function_get_init_data(info));
	if (!init->error.empty()) {
		duckdb_function_set_error(info, init->error.c_str());
		return;
	}

	const idx_t capacity = duckdb_vector_size();
	const int64_t remaining = init->rows - init->emitted;
	const idx_t count = remaining <= 0 ? 0 : static_cast<idx_t>(
	                        remaining < static_cast<int64_t>(capacity) ? remaining : int64_t(capacity));
	if (count == 0) {
		duckdb_data_chunk_set_size(output, 0);
		return;
	}

	const size_t start = static_cast<size_t>(init->emitted);
	for (size_t c = 0; c < init->columns.size(); c++) {
		duckdb_vector vector = duckdb_data_chunk_get_vector(output, static_cast<idx_t>(c));
		duckdb_vector_ensure_validity_writable(vector);
		uint64_t *validity = duckdb_vector_get_validity(vector);
		const std::vector<Scalar> &source = init->columns[c];

		// arrowmetal_query's first column is the aggregate's name, which lives in init->names.
		const bool is_name_column = bind_data->kind == Kind::Query && c == 0;
		int64_t *integers = nullptr;
		double *doubles = nullptr;
		if (!is_name_column) {
			void *data = duckdb_vector_get_data(vector);
			// Column types settled in bind(): count/exact and the integral value columns are BIGINT,
			// the rest DOUBLE.
			const bool as_double =
			    (bind_data->kind == Kind::Agg && (c == 4 || (bind_data->value_is_float && c != 1))) ||
			    (bind_data->kind == Kind::GroupBy && bind_data->value_is_float && c >= 2) ||
			    ((bind_data->kind == Kind::TopK || bind_data->kind == Kind::Sort) &&
			     bind_data->value_is_float) ||
			    (bind_data->kind == Kind::Query && c == 1);
			if (as_double) {
				doubles = static_cast<double *>(data);
			} else {
				integers = static_cast<int64_t *>(data);
			}
		}

		for (idx_t i = 0; i < count; i++) {
			const Scalar &s = source[start + i];
			if (is_name_column) {
				duckdb_vector_assign_string_element(vector, i, init->names[start + i].c_str());
				continue;
			}
			if (s.is_null) {
				duckdb_validity_set_row_validity(validity, i, false);
				continue;
			}
			duckdb_validity_set_row_validity(validity, i, true);
			if (doubles) {
				doubles[i] = s.as_double();
			} else {
				integers[i] = s.is_int ? s.i : static_cast<int64_t>(s.d);
			}
		}
	}
	init->emitted += static_cast<int64_t>(count);
	duckdb_data_chunk_set_size(output, count);
}

//===--------------------------------------------------------------------===//
// Scalar functions: the two that answer "did it load, and what will it run on"
//===--------------------------------------------------------------------===//
void version_function(duckdb_function_info, duckdb_data_chunk input, duckdb_vector output) {
	const idx_t count = duckdb_data_chunk_get_size(input);
	const char *text = am_version();
	for (idx_t i = 0; i < count; i++) {
		duckdb_vector_assign_string_element(output, i, text ? text : "");
	}
}

void device_function(duckdb_function_info, duckdb_data_chunk input, duckdb_vector output) {
	const idx_t count = duckdb_data_chunk_get_size(input);
	const char *text = am_device_name();
	for (idx_t i = 0; i < count; i++) {
		duckdb_vector_assign_string_element(output, i, text ? text : "");
	}
}

bool register_scalar(duckdb_connection conn, const char *name, duckdb_scalar_function_t fn) {
	duckdb_scalar_function function = duckdb_create_scalar_function();
	duckdb_scalar_function_set_name(function, name);
	duckdb_logical_type varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
	duckdb_scalar_function_set_return_type(function, varchar);
	duckdb_scalar_function_set_function(function, fn);
	const bool ok = duckdb_register_scalar_function(conn, function) == DuckDBSuccess;
	duckdb_destroy_logical_type(&varchar);
	duckdb_destroy_scalar_function(&function);
	return ok;
}

bool register_table(duckdb_connection conn, const char *name, duckdb_table_function_bind_t bind_fn,
                    const std::vector<duckdb_type> &parameters) {
	duckdb_table_function function = duckdb_create_table_function();
	duckdb_table_function_set_name(function, name);
	for (duckdb_type type : parameters) {
		duckdb_logical_type logical = duckdb_create_logical_type(type);
		duckdb_table_function_add_parameter(function, logical);
		duckdb_destroy_logical_type(&logical);
	}
	duckdb_table_function_set_bind(function, bind_fn);
	duckdb_table_function_set_init(function, init_function);
	duckdb_table_function_set_function(function, main_function);
	const bool ok = duckdb_register_table_function(conn, function) == DuckDBSuccess;
	duckdb_destroy_table_function(&function);
	return ok;
}

}  // namespace

//===--------------------------------------------------------------------===//
// Entrypoint
//===--------------------------------------------------------------------===//
DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection, duckdb_extension_info info,
                            struct duckdb_extension_access *access) {
	duckdb_database *database = access->get_database(info);
	if (!database || duckdb_connect(*database, &g_connection) == DuckDBError) {
		access->set_error(info, "arrowmetal: could not open the extension's own connection");
		return false;
	}

	const auto V = DUCKDB_TYPE_VARCHAR;
	const auto B = DUCKDB_TYPE_BIGINT;
	const bool ok =
	    register_scalar(connection, "arrowmetal_version", version_function) &&
	    register_scalar(connection, "arrowmetal_device", device_function) &&
	    register_table(connection, "arrowmetal_agg", bind<Kind::Agg>, {V, V}) &&
	    register_table(connection, "arrowmetal_group_by", bind<Kind::GroupBy>, {V, V, V}) &&
	    register_table(connection, "arrowmetal_top_k", bind<Kind::TopK>, {V, V, B}) &&
	    register_table(connection, "arrowmetal_sort", bind<Kind::Sort>, {V, V}) &&
	    register_table(connection, "arrowmetal_query", bind<Kind::Query>, {V, V});
	if (!ok) {
		access->set_error(info, "arrowmetal: failed to register the extension's functions");
		return false;
	}
	return true;
}
