package org.arrowmetal.spark;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * The plugin's only native boundary: one Arrow column in, through the Arrow C Data Interface, and the
 * GPU's sort order out, by calling libArrowMetalC.dylib with Java's foreign function API (no JNI).
 *
 * The column is built in off-heap memory as an ArrowArray / ArrowSchema pair (layouts from the Arrow
 * C Data Interface spec), imported with am_import, sorted with am_argsort_ex2 and the uint32 indices
 * are read back through am_export.
 */
public final class ArrowMetalNative {
    /** Arrow format strings for the key types the plugin hands over. */
    public static final String INT64 = "l", INT32 = "i", FLOAT64 = "g", FLOAT32 = "f";

    private static final Linker LINKER = Linker.nativeLinker();
    private static final MethodHandle AM_IMPORT, AM_ARGSORT_EX2, AM_EXPORT, AM_RELEASE, AM_LAST_ERROR, AM_VERSION;
    private static final MethodHandle AM_GROUP_BY_KEYS, AM_GROUP_COUNT, AM_GROUP_KEYS_RESULT, AM_GROUP_AGG_EX, AM_GROUP_RELEASE;
    private static final MemorySegment RELEASE_SCHEMA_STUB, RELEASE_ARRAY_STUB;
    private static final FunctionDescriptor RELEASE_DESC = FunctionDescriptor.ofVoid(ADDRESS);

    // ArrowSchema: format, name, metadata, flags, n_children, children, dictionary, release, private_data
    private static final long SCHEMA_SIZE = 72, SCHEMA_RELEASE = 56;
    // ArrowArray: length, null_count, offset, n_buffers, n_children, buffers, children, dictionary, release, private_data
    private static final long ARRAY_SIZE = 80, ARRAY_RELEASE = 64;

    static {
        String lib = System.getenv().getOrDefault("ARROWMETAL_LIB",
                System.getProperty("arrowmetal.lib", "libArrowMetalC.dylib"));
        SymbolLookup sym = SymbolLookup.libraryLookup(lib, Arena.global());
        AM_IMPORT = LINKER.downcallHandle(sym.find("am_import").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        AM_ARGSORT_EX2 = LINKER.downcallHandle(sym.find("am_argsort_ex2").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS));
        AM_EXPORT = LINKER.downcallHandle(sym.find("am_export").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
        AM_RELEASE = LINKER.downcallHandle(sym.find("am_release").orElseThrow(), FunctionDescriptor.ofVoid(ADDRESS));
        AM_LAST_ERROR = LINKER.downcallHandle(sym.find("am_last_error").orElseThrow(), FunctionDescriptor.of(ADDRESS));
        AM_VERSION = LINKER.downcallHandle(sym.find("am_version").orElseThrow(), FunctionDescriptor.of(ADDRESS));
        AM_GROUP_BY_KEYS = LINKER.downcallHandle(sym.find("am_group_by_keys").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
        AM_GROUP_COUNT = LINKER.downcallHandle(sym.find("am_group_by_group_count").orElseThrow(),
                FunctionDescriptor.of(JAVA_LONG, ADDRESS));
        AM_GROUP_KEYS_RESULT = LINKER.downcallHandle(sym.find("am_group_by_keys_result").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
        AM_GROUP_AGG_EX = LINKER.downcallHandle(sym.find("am_group_agg_ex").orElseThrow(),
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ValueLayout.JAVA_DOUBLE, ADDRESS));
        AM_GROUP_RELEASE = LINKER.downcallHandle(sym.find("am_group_by_release").orElseThrow(),
                FunctionDescriptor.ofVoid(ADDRESS));
        try {
            MethodHandles.Lookup l = MethodHandles.lookup();
            MethodType mt = MethodType.methodType(void.class, MemorySegment.class);
            RELEASE_SCHEMA_STUB = LINKER.upcallStub(l.findStatic(ArrowMetalNative.class, "releaseSchema", mt), RELEASE_DESC, Arena.global());
            RELEASE_ARRAY_STUB = LINKER.upcallStub(l.findStatic(ArrowMetalNative.class, "releaseArray", mt), RELEASE_DESC, Arena.global());
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private ArrowMetalNative() {}

    // The producer's release callbacks. The memory behind the structs belongs to the caller's arena, so
    // releasing only marks the struct released, as the C Data Interface requires.
    static void releaseSchema(MemorySegment s) {
        s.reinterpret(SCHEMA_SIZE).set(ADDRESS, SCHEMA_RELEASE, MemorySegment.NULL);
    }

    static void releaseArray(MemorySegment a) {
        a.reinterpret(ARRAY_SIZE).set(ADDRESS, ARRAY_RELEASE, MemorySegment.NULL);
    }

    public static String version() {
        try {
            return ((MemorySegment) AM_VERSION.invokeExact()).reinterpret(256).getString(0);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    private static String lastError() throws Throwable {
        return ((MemorySegment) AM_LAST_ERROR.invokeExact()).reinterpret(4096).getString(0);
    }

    private static void check(int rc, String what) throws Throwable {
        if (rc != 0) throw new IllegalStateException("ArrowMetal " + what + ": " + lastError());
    }

    /** Bytes per value for a format string. */
    public static int width(String format) {
        switch (format) {
            case INT64: case FLOAT64: return 8;
            case INT32: case FLOAT32: return 4;
            default: throw new IllegalArgumentException(format);
        }
    }

    /**
     * Sort order of n keys on the GPU. `values` holds n fixed-width values of `format`; `validity` is an
     * Arrow validity bitmap (bit set = valid) or NULL when there are no nulls. float order 2 is
     * "NaN largest, -0.0 ties +0.0", Spark's order for floats. Returns the row numbers in sorted order;
     * the sort is stable.
     */
    public static int[] argsort(MemorySegment values, MemorySegment validity, long n, long nullCount,
                                String format, boolean descending, boolean nullsFirst) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment schema = arena.allocate(SCHEMA_SIZE, 8);
            MemorySegment array = arena.allocate(ARRAY_SIZE, 8);
            MemorySegment buffers = arena.allocate(16, 8);
            buffers.set(ADDRESS, 0, nullCount > 0 ? validity : MemorySegment.NULL);
            buffers.set(ADDRESS, 8, values);
            schema.set(ADDRESS, 0, arena.allocateFrom(format));
            schema.set(ADDRESS, 8, MemorySegment.NULL);
            schema.set(ADDRESS, 16, MemorySegment.NULL);
            schema.set(JAVA_LONG, 24, 2L);          // ARROW_FLAG_NULLABLE
            schema.set(JAVA_LONG, 32, 0L);
            schema.set(ADDRESS, 40, MemorySegment.NULL);
            schema.set(ADDRESS, 48, MemorySegment.NULL);
            schema.set(ADDRESS, SCHEMA_RELEASE, RELEASE_SCHEMA_STUB);
            schema.set(ADDRESS, 64, MemorySegment.NULL);
            array.set(JAVA_LONG, 0, n);
            array.set(JAVA_LONG, 8, nullCount);
            array.set(JAVA_LONG, 16, 0L);
            array.set(JAVA_LONG, 24, 2L);
            array.set(JAVA_LONG, 32, 0L);
            array.set(ADDRESS, 40, buffers);
            array.set(ADDRESS, 48, MemorySegment.NULL);
            array.set(ADDRESS, 56, MemorySegment.NULL);
            array.set(ADDRESS, ARRAY_RELEASE, RELEASE_ARRAY_STUB);
            array.set(ADDRESS, 72, MemorySegment.NULL);

            MemorySegment handle = arena.allocate(ADDRESS);
            check((int) AM_IMPORT.invokeExact(schema, array, handle), "import");
            MemorySegment col = handle.get(ADDRESS, 0);
            MemorySegment out = arena.allocate(ADDRESS);
            try {
                check((int) AM_ARGSORT_EX2.invokeExact(col, descending ? 1 : 0, nullsFirst ? 1 : 0, 2, out), "argsort");
            } finally {
                AM_RELEASE.invokeExact(col);
            }
            MemorySegment idx = out.get(ADDRESS, 0);
            MemorySegment xs = arena.allocate(SCHEMA_SIZE, 8);
            MemorySegment xa = arena.allocate(ARRAY_SIZE, 8);
            try {
                check((int) AM_EXPORT.invokeExact(idx, xs, xa), "export");
            } finally {
                AM_RELEASE.invokeExact(idx);
            }
            try {
                long len = xa.get(JAVA_LONG, 0);
                long off = xa.get(JAVA_LONG, 16);
                MemorySegment bufs = xa.get(ADDRESS, 40).reinterpret(16);
                MemorySegment data = bufs.get(ADDRESS, 8).reinterpret((off + len) * 4);
                int[] order = new int[(int) len];
                MemorySegment.copy(data, ValueLayout.JAVA_INT, off * 4, order, 0, (int) len);
                return order;      // uint32 row numbers; a partition is far below 2^31 rows
            } finally {
                callRelease(xa, ARRAY_RELEASE);
                callRelease(xs, SCHEMA_RELEASE);
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** A column read back from the GPU: its Arrow format, length, values and validity (null = no nulls). */
    public static final class Column {
        public final String format;
        public final int length;
        public final long[] longs;      // format "l"
        public final int[] ints;        // format "i" (and date32 storage)
        public final double[] doubles;  // format "g"
        public final byte[] validity;   // Arrow bitmap, bit set = valid; null when the column has no nulls
        Column(String format, int length, long[] longs, int[] ints, double[] doubles, byte[] validity) {
            this.format = format; this.length = length; this.longs = longs; this.ints = ints;
            this.doubles = doubles; this.validity = validity;
        }
        public boolean isNull(int i) { return validity != null && (validity[i >>> 3] & (1 << (i & 7))) == 0; }
    }

    /** Fills an ArrowSchema / ArrowArray pair for one fixed-width column and imports it (am_import).
     *  The memory stays the caller's: `arena` must outlive the returned handle (release with release()). */
    public static MemorySegment importColumn(Arena arena, String format, MemorySegment values, MemorySegment validity,
                                             long n, long nullCount) {
        try {
            MemorySegment schema = arena.allocate(SCHEMA_SIZE, 8);
            MemorySegment array = arena.allocate(ARRAY_SIZE, 8);
            MemorySegment buffers = arena.allocate(16, 8);
            buffers.set(ADDRESS, 0, nullCount > 0 ? validity : MemorySegment.NULL);
            buffers.set(ADDRESS, 8, values);
            schema.set(ADDRESS, 0, arena.allocateFrom(format));
            schema.set(ADDRESS, 8, MemorySegment.NULL);
            schema.set(ADDRESS, 16, MemorySegment.NULL);
            schema.set(JAVA_LONG, 24, 2L);
            schema.set(JAVA_LONG, 32, 0L);
            schema.set(ADDRESS, 40, MemorySegment.NULL);
            schema.set(ADDRESS, 48, MemorySegment.NULL);
            schema.set(ADDRESS, SCHEMA_RELEASE, RELEASE_SCHEMA_STUB);
            schema.set(ADDRESS, 64, MemorySegment.NULL);
            array.set(JAVA_LONG, 0, n);
            array.set(JAVA_LONG, 8, nullCount);
            array.set(JAVA_LONG, 16, 0L);
            array.set(JAVA_LONG, 24, 2L);
            array.set(JAVA_LONG, 32, 0L);
            array.set(ADDRESS, 40, buffers);
            array.set(ADDRESS, 48, MemorySegment.NULL);
            array.set(ADDRESS, 56, MemorySegment.NULL);
            array.set(ADDRESS, ARRAY_RELEASE, RELEASE_ARRAY_STUB);
            array.set(ADDRESS, 72, MemorySegment.NULL);
            MemorySegment handle = arena.allocate(ADDRESS);
            check((int) AM_IMPORT.invokeExact(schema, array, handle), "import");
            return handle.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static void release(MemorySegment amArray) {
        try {
            AM_RELEASE.invokeExact(amArray);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** Copies a GPU column back into Java arrays (am_export) and releases the handle. */
    public static Column readAndRelease(MemorySegment amArray) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment xs = arena.allocate(SCHEMA_SIZE, 8);
            MemorySegment xa = arena.allocate(ARRAY_SIZE, 8);
            try {
                check((int) AM_EXPORT.invokeExact(amArray, xs, xa), "export");
            } finally {
                AM_RELEASE.invokeExact(amArray);
            }
            try {
                String format = xs.get(ADDRESS, 0).reinterpret(64).getString(0);
                int len = (int) xa.get(JAVA_LONG, 0);
                long nulls = xa.get(JAVA_LONG, 8);
                long off = xa.get(JAVA_LONG, 16);
                MemorySegment bufs = xa.get(ADDRESS, 40).reinterpret(16);
                byte[] validity = null;
                MemorySegment vb = bufs.get(ADDRESS, 0);
                if (nulls != 0 && !vb.equals(MemorySegment.NULL)) {
                    vb = vb.reinterpret((off + len + 7) / 8 + 1);
                    validity = new byte[(len + 7) / 8];
                    for (int i = 0; i < len; i++) {
                        long bit = off + i;
                        if ((vb.get(ValueLayout.JAVA_BYTE, bit >>> 3) & (1 << (bit & 7))) != 0)
                            validity[i >>> 3] |= (byte) (1 << (i & 7));
                    }
                }
                MemorySegment data = bufs.get(ADDRESS, 8);
                switch (format) {
                    case "l": {
                        long[] v = new long[len];
                        MemorySegment.copy(data.reinterpret((off + len) * 8), ValueLayout.JAVA_LONG, off * 8, v, 0, len);
                        return new Column(format, len, v, null, null, validity);
                    }
                    case "i": case "tdD": {
                        int[] v = new int[len];
                        MemorySegment.copy(data.reinterpret((off + len) * 4), ValueLayout.JAVA_INT, off * 4, v, 0, len);
                        return new Column(format, len, null, v, null, validity);
                    }
                    case "g": {
                        double[] v = new double[len];
                        MemorySegment.copy(data.reinterpret((off + len) * 8), ValueLayout.JAVA_DOUBLE, off * 8, v, 0, len);
                        return new Column(format, len, null, null, v, validity);
                    }
                    default:
                        throw new IllegalStateException("ArrowMetal returned a column of format " + format);
                }
            } finally {
                callRelease(xa, ARRAY_RELEASE);
                callRelease(xs, SCHEMA_RELEASE);
            }
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** am_group_by_keys over one key column; returns the am_groupby handle. */
    public static MemorySegment groupBy(MemorySegment keyArray) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cols = arena.allocate(ADDRESS);
            cols.set(ADDRESS, 0, keyArray);
            MemorySegment out = arena.allocate(ADDRESS);
            check((int) AM_GROUP_BY_KEYS.invokeExact(cols, 1L, out), "group_by_keys");
            return out.get(ADDRESS, 0);
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static Column groupKeys(MemorySegment gb) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            check((int) AM_GROUP_KEYS_RESULT.invokeExact(gb, 0L, out), "group_by_keys_result");
            return readAndRelease(out.get(ADDRESS, 0));
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    /** One grouped aggregate (am_group_agg_ex op table: 0 sum, 1 count_all, 2 count, 4 min, 5 max). */
    public static Column groupAgg(MemorySegment gb, MemorySegment values, int op) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ADDRESS);
            MemorySegment v = values == null ? MemorySegment.NULL : values;
            check((int) AM_GROUP_AGG_EX.invokeExact(gb, v, op, 0.0, out), "group_agg");
            return readAndRelease(out.get(ADDRESS, 0));
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    public static void groupRelease(MemorySegment gb) {
        try {
            AM_GROUP_RELEASE.invokeExact(gb);
        } catch (Throwable t) {
            throw new RuntimeException(t);
        }
    }

    private static void callRelease(MemorySegment struct, long releaseOffset) throws Throwable {
        MemorySegment fn = struct.get(ADDRESS, releaseOffset);
        if (fn.equals(MemorySegment.NULL)) return;
        LINKER.downcallHandle(fn, RELEASE_DESC).invokeExact(struct);
    }
}
