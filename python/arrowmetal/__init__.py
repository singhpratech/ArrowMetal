"""ArrowMetal for Python: Apache Arrow arrays on the Apple silicon GPU.

Thin ctypes wrapper over libArrowMetalC. Accepts anything that speaks the Arrow PyCapsule protocol
(pyarrow arrays, Polars series via .to_arrow(), pandas arrow-backed columns) and returns pyarrow arrays,
so results drop straight back into Polars, pandas, DuckDB or pyarrow.compute.

    import pyarrow as pa, arrowmetal as am
    col = am.MetalArray.from_arrow(pa.array([1, None, 3]))
    kept = col.filter_where(">", 1)           # runs on the GPU
    kept.to_arrow()                           # -> pyarrow.Array([3])
"""
import ctypes, ctypes.util, decimal, os, struct, sys
import pyarrow as pa

__version__ = "0.1.0"

_OPS = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5, "eq": 0, "ne": 1, "lt": 2, "le": 3, "gt": 4, "ge": 5}
_ARITH = {"+": 0, "-": 1, "*": 2, "/": 3, "add": 0, "sub": 1, "mul": 2, "div": 3}
_AGG = {"sum": 0, "count": 1, "min": 2, "max": 3, "mean": 4, "count_values": 5}
_STRUCT = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "l": "q", "L": "Q", "f": "f", "g": "d"}


def _find_library():
    env = os.environ.get("ARROWMETAL_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "libArrowMetalC.dylib"),
        os.path.join(here, "..", "..", ".build", "release", "libArrowMetalC.dylib"),
        os.path.join(here, "..", "..", ".build", "debug", "libArrowMetalC.dylib"),
        "/usr/local/lib/libArrowMetalC.dylib",
        "/opt/homebrew/lib/libArrowMetalC.dylib",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise OSError("libArrowMetalC.dylib not found; build with `swift build -c release --product ArrowMetalC` "
                  "or set ARROWMETAL_LIB")


_lib = ctypes.CDLL(_find_library())
_P = ctypes.c_void_p
_lib.am_version.restype = ctypes.c_char_p
_lib.am_device_name.restype = ctypes.c_char_p
_lib.am_last_error.restype = ctypes.c_char_p
_lib.am_format.restype = ctypes.c_char_p
_lib.am_format.argtypes = [_P]
_lib.am_length.restype = ctypes.c_int64
_lib.am_length.argtypes = [_P]
_lib.am_null_count.restype = ctypes.c_int64
_lib.am_null_count.argtypes = [_P]
_lib.am_release.argtypes = [_P]
_lib.am_import.argtypes = [_P, _P, ctypes.POINTER(_P)]
_lib.am_export.argtypes = [_P, _P, _P]
_lib.am_reduce.argtypes = [_P, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
                           ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
for name, extra in [("am_compare_scalar", [ctypes.c_int, _P]), ("am_compare_array", [ctypes.c_int, _P]),
                    ("am_arith_scalar", [ctypes.c_int, _P]), ("am_arith_array", [ctypes.c_int, _P]),
                    ("am_cast", [ctypes.c_char_p]), ("am_bool_and", [_P]), ("am_bool_or", [_P]), ("am_bool_not", []),
                    ("am_filter", [_P]), ("am_filter_where", [ctypes.c_int, _P]), ("am_take", [_P]),
                    ("am_slice", [ctypes.c_int64, ctypes.c_int64]),
                    ("am_argsort", [ctypes.c_int]), ("am_sort", [ctypes.c_int]),
                    ("am_top_k", [ctypes.c_int64, ctypes.c_int])]:
    getattr(_lib, name).argtypes = [_P] + extra + [ctypes.POINTER(_P)]
    getattr(_lib, name).restype = ctypes.c_int
_lib.am_str_unary.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]; _lib.am_str_unary.restype = ctypes.c_int
_lib.am_str_match.argtypes = [_P, ctypes.c_int, ctypes.c_char_p, ctypes.c_int64, ctypes.POINTER(_P)]; _lib.am_str_match.restype = ctypes.c_int
_lib.am_str_equals_array.argtypes = [_P, _P, ctypes.POINTER(_P)]; _lib.am_str_equals_array.restype = ctypes.c_int
_lib.am_str_dictionary_encode.argtypes = [_P, ctypes.POINTER(_P), ctypes.POINTER(_P)]; _lib.am_str_dictionary_encode.restype = ctypes.c_int
_lib.am_temporal_extract.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]; _lib.am_temporal_extract.restype = ctypes.c_int
_lib.am_temporal_cast_unit.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]; _lib.am_temporal_cast_unit.restype = ctypes.c_int
_lib.am_dictionary_decode.argtypes = [_P, ctypes.POINTER(_P)]; _lib.am_dictionary_decode.restype = ctypes.c_int

_lib.am_str_transform.argtypes = [_P, ctypes.c_int, ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
                                  ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_str_transform.restype = ctypes.c_int
_lib.am_str_concat.argtypes = [_P, _P, ctypes.c_char_p, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_str_concat.restype = ctypes.c_int
_lib.am_group_by.argtypes = [_P, ctypes.c_int64, ctypes.c_int, _P, ctypes.POINTER(_P)]
_lib.am_group_by.restype = ctypes.c_int
_lib.am_batch_begin.restype = ctypes.c_int
_lib.am_batch_end.restype = ctypes.c_int
_lib.am_unary.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]; _lib.am_unary.restype = ctypes.c_int
_lib.am_binary.argtypes = [_P, ctypes.c_int, _P, _P, ctypes.POINTER(_P)]; _lib.am_binary.restype = ctypes.c_int
_lib.am_cumulative.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]; _lib.am_cumulative.restype = ctypes.c_int
_lib.am_decimal_op.argtypes = [_P, ctypes.c_int, _P, _P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_decimal_op.restype = ctypes.c_int

# Op numbering is the C ABI contract; see include/arrowmetal.h.
_UNARY = {"negate": 0, "abs": 1, "sign": 2, "sqrt": 3, "exp": 4, "ln": 5, "log10": 6, "log2": 7,
          "floor": 8, "ceil": 9, "round": 10, "trunc": 11, "bitwise_not": 12}
_BINARY = {"bitwise_and": 0, "bitwise_or": 1, "bitwise_xor": 2, "shift_left": 3, "shift_right": 4,
           "modulo": 5, "power": 6, "min_element_wise": 7, "max_element_wise": 8}
_CUMULATIVE = {"sum": 0, "min": 1, "max": 2}


_UNITS = {"s": "s", "m": "ms", "u": "us", "n": "ns"}
_TEMPORAL_FIELDS = {"year": 0, "month": 1, "day": 2, "day_of_week": 3, "hour": 4, "minute": 5, "second": 6}


# Decimal op numbering is the C ABI contract; see include/arrowmetal.h.
_DECIMAL_ROUND = {"round": 12, "ceil": 13, "floor": 14, "truncate": 15, "trunc": 15}


def _decimal_type(fmt):
    """pyarrow type for an Arrow decimal format string ("d:p,s" / "d:p,s,256"), or None."""
    if not fmt.startswith("d:"):
        return None
    parts = fmt[2:].split(",")
    if len(parts) < 2:
        return None
    p, s = int(parts[0]), int(parts[1])
    return pa.decimal256(p, s) if len(parts) > 2 and parts[2] == "256" else pa.decimal128(p, s)


def _decimal_scale(fmt):
    return int(fmt[2:].split(",")[1])


def _decimal_scalar(v, scale):
    """16 little-endian bytes for a decimal scalar. An int is the unscaled value; a Decimal, float or
    string is the real value and is multiplied by 10^scale (halves away from zero)."""
    if isinstance(v, int):
        unscaled = v
    else:
        d = v if isinstance(v, decimal.Decimal) else decimal.Decimal(str(v))
        unscaled = int((d * (10 ** scale)).to_integral_value(rounding=decimal.ROUND_HALF_UP))
    return ctypes.create_string_buffer(int(unscaled).to_bytes(16, "little", signed=True), 16)


def _temporal_type(fmt):
    """pyarrow type for an Arrow temporal format string, or None when it is not temporal."""
    if fmt == "tdD":
        return pa.date32()
    if fmt == "tdm":
        return pa.date64()
    if len(fmt) >= 3 and fmt[0] == "t" and fmt[2] in _UNITS:
        unit, kind = _UNITS[fmt[2]], fmt[1]
        if kind == "t":
            return pa.time32(unit) if unit in ("s", "ms") else pa.time64(unit)
        if kind == "D":
            return pa.duration(unit)
        if kind == "s" and len(fmt) >= 4 and fmt[3] == ":":
            tz = fmt[4:]
            return pa.timestamp(unit, tz or None)
    return None


def version():
    return _lib.am_version().decode()


def device_name():
    return _lib.am_device_name().decode()


class ArrowMetalError(RuntimeError):
    pass


def _check(rc):
    if rc != 0:
        raise ArrowMetalError((_lib.am_last_error() or b"unknown error").decode())


def _call(fn, *args):
    out = _P()
    _check(fn(*args, ctypes.byref(out)))
    return MetalArray(out)


class MetalArray:
    """A Metal-resident Arrow array. Immutable; every operation returns a new array."""

    def __init__(self, handle):
        self._h = handle

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.am_release(h)
            self._h = None

    # ---- interop
    @classmethod
    def from_arrow(cls, obj):
        """Import from anything with __arrow_c_array__ (pyarrow.Array, ChunkedArray of one chunk, Polars .to_arrow(), ...)."""
        if not isinstance(obj, pa.Array):
            if isinstance(obj, pa.ChunkedArray):
                obj = obj.combine_chunks()
            elif hasattr(obj, "__arrow_c_array__"):
                obj = pa.array(obj)
            else:
                obj = pa.array(obj)
        schema = _ArrowSchema()
        array = _ArrowArray()
        obj._export_to_c(ctypes.addressof(array), ctypes.addressof(schema))
        try:
            return _call(_lib.am_import, ctypes.addressof(schema), ctypes.addressof(array))
        finally:
            if schema.release:
                schema.release(ctypes.byref(schema))

    def to_arrow(self):
        """Export as a pyarrow.Array. Zero-copy: the pyarrow array keeps the Metal buffers alive."""
        schema = _ArrowSchema()
        array = _ArrowArray()
        _check(_lib.am_export(self._h, ctypes.addressof(schema), ctypes.addressof(array)))
        return pa.Array._import_from_c(ctypes.addressof(array), ctypes.addressof(schema))

    def __arrow_c_array__(self, requested_schema=None):
        return self.to_arrow().__arrow_c_array__(requested_schema)

    # ---- properties
    def __len__(self):
        return _lib.am_length(self._h)

    @property
    def null_count(self):
        return _lib.am_null_count(self._h)

    @property
    def format(self):
        return _lib.am_format(self._h).decode()

    @property
    def type(self):
        fmt = self.format
        alias = {"c": "int8", "C": "uint8", "s": "int16", "S": "uint16", "i": "int32", "I": "uint32",
                 "l": "int64", "L": "uint64", "f": "float32", "g": "float64", "b": "bool",
                 "u": "string", "U": "large_string", "z": "binary", "Z": "large_binary"}.get(fmt)
        if alias is not None and fmt != "i":
            return pa.type_for_alias(alias)
        temporal = _temporal_type(fmt)
        if temporal is not None:
            return temporal
        dec = _decimal_type(fmt)
        if dec is not None:
            return dec
        # "i" is also the index format of a dictionary array; ask the exported schema which one it is.
        return self.to_arrow().type

    def __repr__(self):
        return f"MetalArray({self.type}, len={len(self)}, nulls={self.null_count}, device={device_name()!r})"

    def _scalar(self, v):
        fmt = self.format
        # A decimal scalar is 16 little-endian bytes, which is what am_compare_scalar expects for "d:p,s".
        if fmt.startswith("d:"):
            return _decimal_scalar(v, _decimal_scale(fmt))
        code = _STRUCT.get(fmt)
        if code is None:
            raise ArrowMetalError("scalar operations need a primitive array")
        return ctypes.create_string_buffer(struct.pack(code, v), 8)

    # ---- reductions
    def _reduce(self, op):
        i = ctypes.c_int64(); f = ctypes.c_double(); kind = ctypes.c_int(); null = ctypes.c_int()
        _check(_lib.am_reduce(self._h, op, ctypes.byref(i), ctypes.byref(f), ctypes.byref(kind), ctypes.byref(null)))
        if null.value:
            return None
        if kind.value == 0:
            return i.value
        if kind.value == 1:
            return i.value & 0xFFFFFFFFFFFFFFFF
        return f.value

    def sum(self): return self._reduce(0)
    def min(self): return self._reduce(1)
    def max(self): return self._reduce(2)
    def mean(self): return self._reduce(3)

    # ---- element-wise
    def compare(self, op, other):
        if isinstance(other, MetalArray):
            return _call(_lib.am_compare_array, self._h, _OPS[op], other._h)
        return _call(_lib.am_compare_scalar, self._h, _OPS[op], self._scalar(other))

    def __eq__(self, o): return self.str_equals(o) if self.format == "u" else self.compare("==", o)
    def __ne__(self, o): return self.compare("!=", o)
    def __lt__(self, o): return self.compare("<", o)
    def __le__(self, o): return self.compare("<=", o)
    def __gt__(self, o): return self.compare(">", o)
    def __ge__(self, o): return self.compare(">=", o)
    __hash__ = object.__hash__

    def arith(self, op, other):
        if isinstance(other, MetalArray):
            return _call(_lib.am_arith_array, self._h, _ARITH[op], other._h)
        return _call(_lib.am_arith_scalar, self._h, _ARITH[op], self._scalar(other))

    def __add__(self, o): return self.arith("+", o)
    def __sub__(self, o): return self.arith("-", o)
    def __mul__(self, o): return self.arith("*", o)
    def __truediv__(self, o): return self.arith("/", o)

    def cast(self, target):
        fmt = target if len(target) == 1 else pa.type_for_alias(target)
        if not isinstance(fmt, str):
            fmt = {"int8": "c", "uint8": "C", "int16": "s", "uint16": "S", "int32": "i", "uint32": "I", "int64": "l",
                   "uint64": "L", "float": "f", "float32": "f", "double": "g", "float64": "g"}[str(fmt)]
        return _call(_lib.am_cast, self._h, fmt.encode())

    def __and__(self, o): return _call(_lib.am_bool_and, self._h, o._h)
    def __or__(self, o): return _call(_lib.am_bool_or, self._h, o._h)
    def __invert__(self): return _call(_lib.am_bool_not, self._h)

    # ---- selection
    def filter(self, mask): return _call(_lib.am_filter, self._h, mask._h)
    def filter_where(self, op, scalar): return _call(_lib.am_filter_where, self._h, _OPS[op], self._scalar(scalar))
    def take(self, indices):
        if not isinstance(indices, MetalArray):
            indices = MetalArray.from_arrow(pa.array(indices, type=pa.int32()))
        return _call(_lib.am_take, self._h, indices._h)
    def slice(self, offset, length): return _call(_lib.am_slice, self._h, offset, length)

    # ---- sorting (GPU radix sort; stable, nulls last)
    def argsort(self, descending=False):
        """Int32 indices that sort the array (Arrow `array_sort_indices`)."""
        return _call(_lib.am_argsort, self._h, 1 if descending else 0)

    def sort(self, descending=False):
        """Sorted copy of the array."""
        return _call(_lib.am_sort, self._h, 1 if descending else 0)

    def top_k(self, k, largest=True):
        """Int32 indices of the k largest (or smallest) values, in sorted order."""
        return _call(_lib.am_top_k, self._h, k, 1 if largest else 0)

    # ---- strings (utf8)
    def byte_length(self): return _call(_lib.am_str_unary, self._h, 0)
    def char_length(self): return _call(_lib.am_str_unary, self._h, 1)
    def hash32(self): return _call(_lib.am_str_unary, self._h, 2)
    def _match(self, pred, pattern):
        b = pattern.encode("utf-8")
        return _call(_lib.am_str_match, self._h, pred, b, len(b))
    def str_equals(self, other):
        if isinstance(other, MetalArray):
            return _call(_lib.am_str_equals_array, self._h, other._h)
        return self._match(0, other)
    def starts_with(self, p): return self._match(1, p)
    def ends_with(self, p): return self._match(2, p)
    def str_contains(self, p): return self._match(3, p)
    def dictionary_encode(self):
        """Returns (codes: int32 MetalArray, unique: string MetalArray). Use codes.group_by(len(unique))."""
        c = _P(); u = _P()
        _check(_lib.am_str_dictionary_encode(self._h, ctypes.byref(c), ctypes.byref(u)))
        return MetalArray(c), MetalArray(u)

    # ---- temporal (date, time, timestamp), extracted in UTC
    def _temporal(self, field):
        return _call(_lib.am_temporal_extract, self._h, _TEMPORAL_FIELDS[field])

    def year(self): return self._temporal("year")
    def month(self): return self._temporal("month")
    def day(self): return self._temporal("day")
    def day_of_week(self): return self._temporal("day_of_week")   # Monday = 0 ... Sunday = 6
    def hour(self): return self._temporal("hour")
    def minute(self): return self._temporal("minute")
    def second(self): return self._temporal("second")

    def cast_unit(self, unit):
        """Rescales a timestamp, duration or time array to 's', 'ms', 'us' or 'ns'."""
        return _call(_lib.am_temporal_cast_unit, self._h, ["s", "ms", "us", "ns"].index(unit))

    # ---- dictionary-encoded arrays
    def decode(self):
        """Materialises a dictionary-encoded array (take of the values by the codes)."""
        return _call(_lib.am_dictionary_decode, self._h)

    # ---- string transforms (utf8 -> utf8, int32 or bool); op table in include/arrowmetal.h
    def _transform(self, op, arg1="", arg2="", p1=0, p2=0):
        a1 = arg1.encode("utf-8") if isinstance(arg1, str) else bytes(arg1)
        a2 = arg2.encode("utf-8") if isinstance(arg2, str) else bytes(arg2)
        return _call(_lib.am_str_transform, self._h, op, a1, len(a1), a2, len(a2), p1, p2)

    def ascii_upper(self): return self._transform(0)
    def ascii_lower(self): return self._transform(1)

    def upper(self):
        """Uppercase (ASCII, Latin-1 Supplement and Latin Extended-A; other code points pass through)."""
        return self._transform(2)

    def lower(self):
        """Lowercase over the same blocks as upper()."""
        return self._transform(3)

    def swapcase(self): return self._transform(4)
    def capitalize(self): return self._transform(5)

    def trim(self, characters=None):
        """Strip both ends: ASCII whitespace by default, or any byte in `characters`."""
        return self._transform(6) if characters is None else self._transform(9, characters)

    def ltrim(self, characters=None):
        return self._transform(7) if characters is None else self._transform(10, characters)

    def rtrim(self, characters=None):
        return self._transform(8) if characters is None else self._transform(11, characters)

    def replace(self, pattern, replacement, max_replacements=-1):
        """Replace non-overlapping occurrences, left to right. An empty pattern is the identity."""
        return self._transform(12, pattern, replacement, max_replacements)

    def repeat(self, n):
        """n copies of each string concatenated (Arrow binary_repeat)."""
        return self._transform(13, p1=n)

    def slice_codeunits(self, start, stop=None):
        """Substring by code point index; negative indices count from the end (Arrow utf8_slice_codeunits)."""
        return self._transform(14, p1=start, p2=(2**63 - 1) if stop is None else stop)

    def pad_left(self, width, pad=" "): return self._transform(15, pad, p1=width)
    def pad_right(self, width, pad=" "): return self._transform(16, pad, p1=width)

    def str_reverse(self):
        """Reverse the code points of each string (Arrow utf8_reverse)."""
        return self._transform(17)

    def str_concat(self, other, separator=""):
        """a + separator + b element-wise (Arrow binary_join_element_wise); null in either side gives null."""
        sep = separator.encode("utf-8")
        return _call(_lib.am_str_concat, self._h, other._h, sep, len(sep))

    def count_substring(self, pattern):
        """int32 count of non-overlapping occurrences; an empty pattern counts code points + 1."""
        return self._transform(18, pattern)

    def find_substring(self, pattern):
        """int32 byte offset of the first occurrence, -1 when absent."""
        return self._transform(19, pattern)

    def is_alnum(self): return self._transform(20)
    def is_alpha(self): return self._transform(21)
    def is_digit(self): return self._transform(22)
    def is_space(self): return self._transform(23)
    def is_upper(self): return self._transform(24)
    def is_lower(self): return self._transform(25)

    # ---- group-by over dense keys in [0, key_count)
    def group_by(self, key_count):
        return GroupBy(self, key_count)

    # ---- structural, conditional and set lookup (GPU)
    def _same_type_array(self, values):
        """Anything array-like becomes a MetalArray of this array's element type."""
        if isinstance(values, MetalArray):
            return values
        return MetalArray.from_arrow(pa.array(list(values), type=self.type))

    def _branch(self, v, hint=None):
        """An if_else branch: an array as given, or a scalar broadcast to this array's length.
        `hint` is the Arrow type the other branch already fixed, so the two agree."""
        if isinstance(v, MetalArray):
            return v
        if isinstance(v, (pa.Array, pa.ChunkedArray)) or hasattr(v, "__arrow_c_array__"):
            return MetalArray.from_arrow(v)
        if isinstance(v, (list, tuple)):
            return MetalArray.from_arrow(pa.array(list(v), type=hint))
        return MetalArray.from_arrow(pa.array([v] * len(self), type=hint))

    def is_null(self):
        """Arrow `is_null`: boolean array, true where this element is null. Never null itself."""
        return _call(_lib.am_is_null, self._h)

    def is_valid(self):
        """Arrow `is_valid`: the complement of `is_null`."""
        return _call(_lib.am_is_valid, self._h)

    def fill_null(self, v):
        """Arrow `fill_null`: nulls become `v`; the result has no nulls."""
        if self.format == "b":
            return _call(_lib.am_fill_null, self._h, ctypes.create_string_buffer(bytes([1 if v else 0]), 8))
        return _call(_lib.am_fill_null, self._h, self._scalar(v))

    def drop_null(self):
        """Arrow `drop_null`: the non-null elements, in order."""
        return _call(_lib.am_drop_null, self._h)

    def if_else(self, left, right):
        """Arrow `if_else` with this boolean array as the condition: `self ? left : right`.
        Either branch may be an array or a scalar; a null condition yields a null element."""
        hint = next((v.type for v in (left, right) if isinstance(v, MetalArray)), None)
        l = self._branch(left, hint)                     # keep both alive across the call
        r = self._branch(right, l.type)
        return _call(_lib.am_if_else, self._h, l._h, r._h)

    def is_in(self, values):
        """Arrow `is_in`: boolean array, true where the element is among the non-null `values`.
        Nulls in `values` are ignored and a null element never matches, so the result has no nulls."""
        s = self._same_type_array(values)                # keep it alive across the call
        return _call(_lib.am_is_in, self._h, s._h)

    def index_in(self, values):
        """Arrow `index_in`: int32 index into `values` of each element's first occurrence there,
        null where the element is null or absent."""
        s = self._same_type_array(values)                # keep it alive across the call
        return _call(_lib.am_index_in, self._h, s._h)

    def and_kleene(self, o):
        """Arrow `and_kleene`: three-valued AND (`false AND null` is `false`)."""
        return _call(_lib.am_and_kleene, self._h, o._h)

    def or_kleene(self, o):
        """Arrow `or_kleene`: three-valued OR (`true OR null` is `true`)."""
        return _call(_lib.am_or_kleene, self._h, o._h)

    # ---- element-wise math (GPU; nulls in, nulls out)
    def unary(self, op):
        """One unary math op by name: negate, abs, sign, sqrt, exp, ln, log10, log2, floor, ceil,
        round, trunc, bitwise_not. See include/arrowmetal.h for the exact semantics."""
        return _call(_lib.am_unary, self._h, _UNARY[op])

    def negate(self): return self.unary("negate")
    def abs(self): return self.unary("abs")
    def sign(self): return self.unary("sign")
    def sqrt(self): return self.unary("sqrt")
    def exp(self): return self.unary("exp")
    def ln(self): return self.unary("ln")
    def log10(self): return self.unary("log10")
    def log2(self): return self.unary("log2")
    def floor(self): return self.unary("floor")
    def ceil(self): return self.unary("ceil")
    def round(self):
        """Rounds halves away from zero. The identity on an integer column."""
        return self.unary("round")
    def trunc(self): return self.unary("trunc")

    def binary(self, op, other):
        """One binary math op by name against a MetalArray or a scalar: bitwise_and, bitwise_or,
        bitwise_xor, shift_left, shift_right, modulo, power, min_element_wise, max_element_wise."""
        code = _BINARY[op]
        if isinstance(other, MetalArray):
            return _call(_lib.am_binary, self._h, code, other._h, None)
        return _call(_lib.am_binary, self._h, code, None, self._scalar(other))

    # Integer bit-wise ops. `__and__`/`__or__`/`__invert__` are the boolean-bitmap kernels, so these
    # value-level ops are spelled out by name instead.
    def bitwise_and(self, other): return self.binary("bitwise_and", other)
    def bitwise_or(self, other): return self.binary("bitwise_or", other)
    def bitwise_xor(self, other): return self.binary("bitwise_xor", other)
    def bitwise_not(self): return self.unary("bitwise_not")
    def shift_left(self, other):
        """Shift counts outside [0, bit width) give 0, rather than raising as Arrow does."""
        return self.binary("shift_left", other)
    def shift_right(self, other):
        """Arithmetic on a signed column, logical on an unsigned one; an out-of-range count gives the
        sign fill (signed) or 0 (unsigned)."""
        return self.binary("shift_right", other)

    def modulo(self, other):
        """C remainder: the sign follows the dividend. Integer x % 0 is defined as 0."""
        return self.binary("modulo", other)

    def power(self, other):
        """Repeated squaring on integers (wrapping; a negative exponent is defined as 0), `pow` on
        floats. Not implemented for float64."""
        return self.binary("power", other)

    def min_element_wise(self, other): return self.binary("min_element_wise", other)
    def max_element_wise(self, other): return self.binary("max_element_wise", other)

    def __mod__(self, o): return self.modulo(o)
    def __pow__(self, o): return self.power(o)

    # ---- cumulative (two-level GPU scan; output null where input null, the run unbroken)
    def cumulative(self, op): return _call(_lib.am_cumulative, self._h, _CUMULATIVE[op])
    def cumulative_sum(self): return self.cumulative("sum")
    def cumulative_min(self): return self.cumulative("min")
    def cumulative_max(self): return self.cumulative("max")

    # ---- decimals (decimal128 "d:p,s", decimal256 "d:p,s,256"); op table in include/arrowmetal.h.
    # The comparison operators (==, <, ...) already work on decimal columns through the ordinary compare
    # path; a scalar may be an int (the unscaled value) or a Decimal / float / string (the real value).
    def _decimal_op(self, op, other=None, scalar=None, p1=0):
        b = other._h if isinstance(other, MetalArray) else None
        return _call(_lib.am_decimal_op, self._h, op, b, scalar, p1)

    def _decimal_operand(self, other):
        """(array_handle, scalar_buffer) for a decimal binary op."""
        if isinstance(other, MetalArray):
            return other, None
        return None, _decimal_scalar(other, _decimal_scale(self.format))

    def decimal_add(self, other):
        """Element-wise sum with another decimal column of the same scale, or with a scalar. Wraps."""
        a, s = self._decimal_operand(other)
        return self._decimal_op(6, a, s)

    def decimal_sub(self, other):
        """Element-wise difference with another decimal column of the same scale, or with a scalar."""
        a, s = self._decimal_operand(other)
        return self._decimal_op(7, a, s)

    def decimal_mul(self, other):
        """Multiply by another decimal column (result scale = s1 + s2, precision p1 + p2 + 1) or by an
        integer scalar (the type is unchanged)."""
        if isinstance(other, MetalArray):
            return self._decimal_op(8, other, None)
        return self._decimal_op(8, None, ctypes.create_string_buffer(struct.pack("q", int(other)), 8))

    def decimal_round(self, scale, mode="round"):
        """Rescale to `scale` decimal places: 'round' (halves away from zero), 'ceil', 'floor' or
        'truncate'. Scaling up is exact."""
        return self._decimal_op(_DECIMAL_ROUND[mode], None, None, scale)

    def to_float64(self):
        """The decimal values as float64 (unscaled / 10^scale). Runs on the host, 53 bits of precision."""
        return self._decimal_op(16)


class GroupBy:
    def __init__(self, keys, key_count):
        self.keys, self.key_count = keys, key_count

    def _agg(self, name, values):
        return _call(_lib.am_group_by, self.keys._h, self.key_count, _AGG[name], values._h if values is not None else None)

    def count(self): return self._agg("count", None)
    def sum(self, values): return self._agg("sum", values)
    def min(self, values): return self._agg("min", values)
    def max(self, values): return self._agg("max", values)
    def mean(self, values): return self._agg("mean", values)
    def count_values(self, values): return self._agg("count_values", values)


class batch:
    """Context manager: kernels issued inside append to one GPU command buffer and run once at exit
    (or at the first result read). Cuts per-call latency from ~150 µs to ~10 µs for chains of operations."""

    def __enter__(self):
        _check(_lib.am_batch_begin())
        return self

    def __exit__(self, exc_type, exc, tb):
        _check(_lib.am_batch_end())
        return False


class _ArrowSchema(ctypes.Structure):
    pass


_ArrowSchema._fields_ = [("format", ctypes.c_char_p), ("name", ctypes.c_char_p), ("metadata", ctypes.c_char_p),
                         ("flags", ctypes.c_int64), ("n_children", ctypes.c_int64),
                         ("children", ctypes.POINTER(ctypes.POINTER(_ArrowSchema))),
                         ("dictionary", ctypes.POINTER(_ArrowSchema)),
                         ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowSchema))), ("private_data", ctypes.c_void_p)]


class _ArrowArray(ctypes.Structure):
    pass


_ArrowArray._fields_ = [("length", ctypes.c_int64), ("null_count", ctypes.c_int64), ("offset", ctypes.c_int64),
                        ("n_buffers", ctypes.c_int64), ("n_children", ctypes.c_int64),
                        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
                        ("children", ctypes.POINTER(ctypes.POINTER(_ArrowArray))),
                        ("dictionary", ctypes.POINTER(_ArrowArray)),
                        ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArray))), ("private_data", ctypes.c_void_p)]


def array(obj):
    """Shorthand for MetalArray.from_arrow."""
    return MetalArray.from_arrow(obj)


# ---- structural, conditional and set-lookup entry points
for _name, _extra in [("am_is_null", []), ("am_is_valid", []), ("am_fill_null", [_P]), ("am_drop_null", []),
                      ("am_if_else", [_P, _P]), ("am_is_in", [_P]), ("am_index_in", [_P]),
                      ("am_and_kleene", [_P]), ("am_or_kleene", [_P])]:
    getattr(_lib, _name).argtypes = [_P] + _extra + [ctypes.POINTER(_P)]
    getattr(_lib, _name).restype = ctypes.c_int
_lib.am_coalesce.argtypes = [ctypes.POINTER(_P), ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_coalesce.restype = ctypes.c_int


def coalesce(*arrays):
    """Arrow `coalesce`: the first non-null value across the arrays, element-wise. All of them must
    have the same type and length; an element is null only when it is null in every input."""
    if not arrays:
        raise ArrowMetalError("coalesce needs at least one array")
    handles = (_P * len(arrays))(*[a._h for a in arrays])
    out = _P()
    _check(_lib.am_coalesce(handles, len(arrays), ctypes.byref(out)))
    return MetalArray(out)


# ---- nested types: list ("+l"), large_list ("+L"), fixed_size_list ("+w:N"), struct ("+s"),
# map ("+m") and dense/sparse union ("+ud:", "+us:").
#
# filter / take / slice and the Arrow C Data round trip already work on them through the generic
# entry points; these add the nested compute surface and child navigation. Methods are attached to
# MetalArray here rather than in the class body so this section stays self-contained.
for _name, _extra in [("am_list_value_length", []), ("am_list_flatten", []),
                      ("am_list_element", [ctypes.c_int64]), ("am_struct_field", [ctypes.c_char_p]),
                      ("am_child", [ctypes.c_int64])]:
    getattr(_lib, _name).argtypes = [_P] + _extra + [ctypes.POINTER(_P)]
    getattr(_lib, _name).restype = ctypes.c_int
_lib.am_child_count.argtypes = [_P]
_lib.am_child_count.restype = ctypes.c_int64


def _list_value_length(self):
    """Arrow `list_value_length`: int32 child count per row, null where the row is null."""
    return _call(_lib.am_list_value_length, self._h)


def _list_flatten(self):
    """Arrow `list_flatten`: the child array, restricted to the range this list references."""
    return _call(_lib.am_list_flatten, self._h)


def _list_element(self, index):
    """Arrow `list_element`: element `index` of every row; null where the row is null or too short."""
    return _call(_lib.am_list_element, self._h, index)


def _struct_field(self, name):
    """Arrow `struct_field`: one field of a struct array by name."""
    return _call(_lib.am_struct_field, self._h, name.encode("utf-8"))


def _child_count(self):
    """Number of child arrays: 1 for a list, map or dictionary, one per field/variant for a struct
    or union, 0 for a flat array."""
    return _lib.am_child_count(self._h)


def _child(self, i):
    """Child `i`: a list's values, a map's `entries` struct, a struct's field, a union's variant."""
    return _call(_lib.am_child, self._h, i)


MetalArray.list_value_length = _list_value_length
MetalArray.list_flatten = _list_flatten
MetalArray.list_element = _list_element
MetalArray.struct_field = _struct_field
MetalArray.child_count = _child_count
MetalArray.child = _child

_NESTED_PREFIXES = ("+l", "+L", "+w:", "+s", "+m", "+u")
_prev_type_property = MetalArray.type


@property
def _type_with_nested(self):
    """pyarrow type of this array. Nested formats carry their children, so the type comes from the
    exported schema; everything else keeps the flat mapping."""
    if self.format.startswith(_NESTED_PREFIXES):
        return self.to_arrow().type
    return _prev_type_property.fget(self)


MetalArray.type = _type_with_nested
# ---- regex, string <-> number casts, temporal rounding and arithmetic
# Appended rather than written into the class body so the file stays additive.
# Op numbering is the C ABI contract; see include/arrowmetal.h.
import re as _re

_lib.am_regex.argtypes = [_P, ctypes.c_int, ctypes.c_char_p, ctypes.c_int64,
                          ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_regex.restype = ctypes.c_int
_lib.am_to_strings.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_to_strings.restype = ctypes.c_int
_lib.am_parse.argtypes = [_P, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_parse.restype = ctypes.c_int
_lib.am_temporal_math.argtypes = [_P, ctypes.c_int, ctypes.c_int64, _P, ctypes.POINTER(_P)]
_lib.am_temporal_math.restype = ctypes.c_int

_REGEX = {"match_substring_regex": 0, "count_substring_regex": 1, "find_substring_regex": 2,
          "replace_substring_regex": 3, "match_like": 4,
          "split_pattern_values": 5, "split_pattern_offsets": 6,
          "split_whitespace_values": 7, "split_whitespace_offsets": 8,
          "extract_regex": 9, "split_pattern_regex_values": 10, "split_pattern_regex_offsets": 11}
_TEMPORAL_MATH = {"floor": 0, "ceil": 1, "round": 2, "add_duration": 3, "subtract": 4,
                  "days_between": 5, "quarter": 6, "day_of_year": 7, "iso_week": 8, "iso_year": 9,
                  "is_leap_year": 10, "millisecond": 11, "microsecond": 12, "nanosecond": 13}
_ROUND_UNITS = ["nanosecond", "microsecond", "millisecond", "second", "minute", "hour",
                "day", "month", "quarter", "year"]
_PARSE_FORMATS = {"int8": "c", "uint8": "C", "int16": "s", "uint16": "S", "int32": "i", "uint32": "I",
                  "int64": "l", "uint64": "L", "float": "f", "float32": "f", "double": "g",
                  "float64": "g", "bool": "b", "boolean": "b"}
# The (?<name>...) groups of a pattern, in order. ICU spells named groups this way; RE2 (and so
# pyarrow) writes (?P<name>...), which is not accepted here.
_NAMED_GROUP = _re.compile(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>")


def _am_regex_call(self, op, pattern="", repl="", ignore_case=False):
    p = pattern.encode("utf-8") if isinstance(pattern, str) else bytes(pattern)
    r = repl.encode("utf-8") if isinstance(repl, str) else bytes(repl)
    return _call(_lib.am_regex, self._h, _REGEX[op], p, len(p), r, len(r), 1 if ignore_case else 0)


def _am_match_substring_regex(self, pattern, ignore_case=False):
    """Arrow `match_substring_regex`: boolean array, true where the pattern matches anywhere.
    A pattern with no metacharacter (or `^literal`) runs on the GPU; anything else on the CPU."""
    return _am_regex_call(self, "match_substring_regex", pattern, ignore_case=ignore_case)


def _am_count_substring_regex(self, pattern, ignore_case=False):
    """int32 count of non-overlapping matches per value."""
    return _am_regex_call(self, "count_substring_regex", pattern, ignore_case=ignore_case)


def _am_find_substring_regex(self, pattern, ignore_case=False):
    """int32 byte offset of the first match, -1 when there is none."""
    return _am_regex_call(self, "find_substring_regex", pattern, ignore_case=ignore_case)


def _am_replace_substring_regex(self, pattern, replacement, ignore_case=False):
    """Arrow `replace_substring_regex`, every match. The replacement is an ICU template: capture
    groups are $1, $2 ... (RE2, and so pyarrow, writes \\1, \\2)."""
    return _am_regex_call(self, "replace_substring_regex", pattern, replacement, ignore_case=ignore_case)


def _am_extract_regex(self, pattern, ignore_case=False):
    """Arrow `extract_regex` as a dict of {group name: MetalArray}. The pattern needs at least one
    (?<name>...) group; a row that does not match is null in every column."""
    names = []
    for name in _NAMED_GROUP.findall(pattern):
        if name not in names:
            names.append(name)
    if not names:
        raise ArrowMetalError("extract_regex needs at least one named group, e.g. (?<year>\\d+)")
    return {n: _am_regex_call(self, "extract_regex", pattern, n, ignore_case=ignore_case) for n in names}


def _am_match_like(self, pattern, ignore_case=False):
    """Arrow `match_like`: SQL LIKE, `%` for any run of characters and `_` for exactly one; a
    backslash escapes them. A pure prefix / suffix / contains / equality pattern runs on the GPU."""
    return _am_regex_call(self, "match_like", pattern, ignore_case=ignore_case)


def _am_split_pattern(self, pattern, regex=False, ignore_case=False):
    """Splits each value, returning the (offsets, values) pair of an Arrow list<utf8>: row i owns
    values[offsets[i]:offsets[i + 1]]. ArrowMetal has no list type, hence the pair."""
    kind = "split_pattern_regex" if regex else "split_pattern"
    offsets = _am_regex_call(self, kind + "_offsets", pattern, ignore_case=ignore_case)
    values = _am_regex_call(self, kind + "_values", pattern, ignore_case=ignore_case)
    return offsets, values


def _am_split_whitespace(self):
    """Splits on runs of ASCII whitespace, as Python's str.split() does. Returns (offsets, values)."""
    return (_am_regex_call(self, "split_whitespace_offsets"),
            _am_regex_call(self, "split_whitespace_values"))


def _am_to_strings(self):
    """Arrow `cast(utf8)`: the decimal text of every value. Integers format on the GPU; floats and
    booleans on the CPU. A float keeps its `.0` (1.0, not 1) and uses Swift's exponent form."""
    return _call(_lib.am_to_strings, self._h)


def _am_parse(self, target, strict=False):
    """Arrow `cast` from utf8 to a numeric or boolean type. Integers parse on the GPU: the whole
    value must match [+-]?[0-9]+. A value that does not parse is null, or an error when strict."""
    name = target if isinstance(target, str) else str(target)
    fmt = name if len(name) == 1 and name in "cCsSiIlLfgb" else _PARSE_FORMATS.get(name)
    if fmt is None:
        raise ArrowMetalError(f"cannot parse into {target}")
    return _call(_lib.am_parse, self._h, fmt.encode(), 1 if strict else 0)


def _am_strftime(self, fmt):
    """Formats a temporal column in UTC with a C strftime format. `%f` is an ArrowMetal extension for
    the six-digit fractional second. CPU."""
    return _call(_lib.am_parse, self._h, fmt.encode(), 0)


def _am_strptime(self, fmt):
    """Parses a utf8 column with a C strptime format, UTC, into timestamp[us]. Values that do not
    parse come back null. CPU."""
    return _call(_lib.am_parse, self._h, fmt.encode(), 0)


def _am_temporal_math(self, op, p1=0, other=None):
    return _call(_lib.am_temporal_math, self._h, _TEMPORAL_MATH[op], p1,
                 other._h if other is not None else None)


def _am_round_arg(unit, multiple):
    if unit not in _ROUND_UNITS:
        raise ArrowMetalError(f"unknown rounding unit {unit!r}; expected one of {_ROUND_UNITS}")
    if multiple < 1:
        raise ArrowMetalError("temporal rounding needs multiple >= 1")
    return _ROUND_UNITS.index(unit) | (multiple << 8)


def _am_floor_temporal(self, unit, multiple=1):
    """Arrow `floor_temporal`: the largest multiple of `multiple` x `unit` at or below each value."""
    return _am_temporal_math(self, "floor", _am_round_arg(unit, multiple))


def _am_ceil_temporal(self, unit, multiple=1):
    """Arrow `ceil_temporal`. A value already on a boundary is left alone."""
    return _am_temporal_math(self, "ceil", _am_round_arg(unit, multiple))


def _am_round_temporal(self, unit, multiple=1):
    """Arrow `round_temporal`. A value exactly halfway rounds up (toward +infinity)."""
    return _am_temporal_math(self, "round", _am_round_arg(unit, multiple))


def _am_add_duration(self, other):
    """Adds a duration column (rescaled to this array's unit) or a scalar count of this array's own
    ticks. The result keeps this array's type."""
    if isinstance(other, MetalArray):
        return _am_temporal_math(self, "add_duration", 0, other)
    return _am_temporal_math(self, "add_duration", int(other))


def _am_subtract_temporal(self, other):
    """`self - other` as a duration, in the finer of the two resolutions."""
    return _am_temporal_math(self, "subtract", 0, other)


def _am_days_between(self, other):
    """Arrow `days_between(self, other)`: whole UTC days from self to other, int64."""
    return _am_temporal_math(self, "days_between", 0, other)


MetalArray.match_substring_regex = _am_match_substring_regex
MetalArray.count_substring_regex = _am_count_substring_regex
MetalArray.find_substring_regex = _am_find_substring_regex
MetalArray.replace_substring_regex = _am_replace_substring_regex
MetalArray.extract_regex = _am_extract_regex
MetalArray.match_like = _am_match_like
MetalArray.split_pattern = _am_split_pattern
MetalArray.split_whitespace = _am_split_whitespace
MetalArray.to_strings = _am_to_strings
MetalArray.parse = _am_parse
MetalArray.strftime = _am_strftime
MetalArray.strptime = _am_strptime
MetalArray.floor_temporal = _am_floor_temporal
MetalArray.ceil_temporal = _am_ceil_temporal
MetalArray.round_temporal = _am_round_temporal
MetalArray.add_duration = _am_add_duration
MetalArray.subtract_temporal = _am_subtract_temporal
MetalArray.days_between = _am_days_between
MetalArray.quarter = lambda self: _am_temporal_math(self, "quarter")
MetalArray.day_of_year = lambda self: _am_temporal_math(self, "day_of_year")
MetalArray.iso_week = lambda self: _am_temporal_math(self, "iso_week")
MetalArray.iso_year = lambda self: _am_temporal_math(self, "iso_year")
MetalArray.is_leap_year = lambda self: _am_temporal_math(self, "is_leap_year")
MetalArray.millisecond = lambda self: _am_temporal_math(self, "millisecond")
MetalArray.microsecond = lambda self: _am_temporal_math(self, "microsecond")
MetalArray.nanosecond = lambda self: _am_temporal_math(self, "nanosecond")

# cast() gains a string target: cast("string") / cast(pa.string()) is to_strings().
_am_numeric_cast = MetalArray.cast


def _am_cast(self, target):
    """Arrow `cast`. Numeric targets run the GPU cast kernel; "string" / "utf8" formats the values
    (see to_strings)."""
    name = target if isinstance(target, str) else str(target)
    if name in ("string", "utf8", "large_string", "u", "U"):
        return self.to_strings()
    return _am_numeric_cast(self, target)


MetalArray.cast = _am_cast
# ---- window, pairwise, rolling-window and multi-column sort entry points
_lib.am_window.argtypes = [_P, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, _P, ctypes.POINTER(_P)]
_lib.am_window.restype = ctypes.c_int
_lib.am_lexsort.argtypes = [ctypes.POINTER(_P), ctypes.POINTER(ctypes.c_int), ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_lexsort.restype = ctypes.c_int

# Op numbering is the C ABI contract; see include/arrowmetal.h.
_WINDOW = {"row_number": 0, "rank": 1, "dense_rank": 2, "percent_rank": 3, "cume_dist": 4,
           "shift": 5, "pairwise_diff": 6, "cumulative_prod": 7, "cumulative_mean": 8,
           "rolling_sum": 9, "rolling_min": 10, "rolling_max": 11, "rolling_mean": 12}


def _window(self, op, p1=0, p2=0, scalar=None):
    """One window / shift / pairwise / rolling op by name; the table is in include/arrowmetal.h."""
    return _call(_lib.am_window, self._h, _WINDOW[op], p1, p2, scalar)


MetalArray.window = _window
MetalArray._window = _window


def _no_arg_window(op, doc):
    def f(self):
        return self._window(op)
    f.__name__ = op
    f.__doc__ = doc
    return f


# The ranking functions take no arguments and rank ascending with nulls last, SQL-style: nulls sort
# after every value and form one tie group, so none of them ever returns a null.
for _op, _doc in [
    ("row_number", "SQL ROW_NUMBER(): 1-based position in sorted order, aligned to the original rows."),
    ("rank", "SQL RANK(): the position of the first row of each tie group, so ranks skip after a tie."),
    ("dense_rank", "SQL DENSE_RANK(): 1-based index of each distinct value, with no gaps."),
    ("percent_rank", "SQL PERCENT_RANK(): (rank - 1) / (n - 1) as float64, 0 for a single row."),
    ("cume_dist", "SQL CUME_DIST(): the fraction of rows at or before this row's value, as float64."),
    ("cumulative_prod", "Arrow cumulative_prod: running product, null exactly where the input is null."),
    ("cumulative_mean", "Arrow cumulative_mean: running mean of the non-null values so far, as float64."),
]:
    setattr(MetalArray, _op, _no_arg_window(_op, _doc))
del _op, _doc


def _shift(self, by, fill=None):
    """Lag (positive `by`) or lead (negative one): out[i] = self[i - by]. A row that would read outside
    the array takes `fill`, or becomes null when `fill` is None."""
    return self._window("shift", p1=by, scalar=None if fill is None else self._scalar(fill))


def _pairwise_diff(self, period=1):
    """Arrow pairwise_diff: out[i] = self[i] - self[i - period], null where either side is missing."""
    return self._window("pairwise_diff", p1=period)


def _rolling(op):
    def f(self, window, min_periods=None):
        return self._window(op, p1=window, p2=0 if min_periods is None else min_periods)
    f.__name__ = op
    f.__doc__ = ("Trailing rolling %s over `window` rows ending at each output, null until `min_periods` "
                 "non-null rows are in the window (which defaults to the whole window). min and max scan "
                 "the window; sum and mean are O(n) prefix-sum differences, so one NaN or infinity in a "
                 "float column affects every later window." % op.split("_")[1])
    return f


MetalArray.shift = _shift
MetalArray.pairwise_diff = _pairwise_diff
for _op in ("rolling_sum", "rolling_min", "rolling_max", "rolling_mean"):
    setattr(MetalArray, _op, _rolling(_op))
del _op


def lexsort_indices(columns, descending=None):
    """Multi-column (lexicographic) sort: int32 indices ordering the rows by each column in turn, the
    first column being the most significant.

    `descending` is one flag per column, or None for all ascending. Successive stable GPU radix argsorts
    from the least significant key upwards; nulls come last in every key, in both directions.

        idx = am.lexsort_indices([region, revenue], [False, True])
        region.take(idx), revenue.take(idx)
    """
    cols = [c if isinstance(c, MetalArray) else MetalArray.from_arrow(c) for c in columns]
    if not cols:
        raise ArrowMetalError("lexsort needs at least one column")
    flags = descending if descending is not None else [False] * len(cols)
    if len(flags) != len(cols):
        raise ArrowMetalError(f"descending has {len(flags)} entries for {len(cols)} columns")
    handles = (_P * len(cols))(*[c._h for c in cols])
    desc = (ctypes.c_int * len(cols))(*[1 if d else 0 for d in flags])
    out = _P()
    _check(_lib.am_lexsort(handles, desc, len(cols), ctypes.byref(out)))
    return MetalArray(out)


# ---- trigonometry, the remaining boolean operators, float classification, the conditional
# ---- transforms and a 64-bit value hash. Op numbering is the C ABI contract; see include/arrowmetal.h.
_lib.am_trig.argtypes = [_P, ctypes.c_int, _P, ctypes.POINTER(_P)]
_lib.am_trig.restype = ctypes.c_int
_lib.am_logical.argtypes = [_P, ctypes.c_int, _P, ctypes.POINTER(_P)]
_lib.am_logical.restype = ctypes.c_int
_lib.am_float_class.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_float_class.restype = ctypes.c_int
_lib.am_fill_null_direction.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_fill_null_direction.restype = ctypes.c_int
_lib.am_case_when.argtypes = [ctypes.POINTER(_P), ctypes.POINTER(_P), ctypes.c_int64, _P, ctypes.POINTER(_P)]
_lib.am_case_when.restype = ctypes.c_int
_lib.am_choose.argtypes = [_P, ctypes.POINTER(_P), ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_choose.restype = ctypes.c_int
_lib.am_replace_with_mask.argtypes = [_P, _P, _P, ctypes.POINTER(_P)]
_lib.am_replace_with_mask.restype = ctypes.c_int
_lib.am_indices_nonzero.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_indices_nonzero.restype = ctypes.c_int
_lib.am_hash64.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_hash64.restype = ctypes.c_int

_TRIG = {"sin": 0, "cos": 1, "tan": 2, "asin": 3, "acos": 4, "atan": 5,
         "sinh": 6, "cosh": 7, "tanh": 8, "asinh": 9, "acosh": 10, "atanh": 11,
         "atan2": 12,
         "sin_checked": 13, "cos_checked": 14, "tan_checked": 15, "asin_checked": 16,
         "acos_checked": 17, "acosh_checked": 18, "atanh_checked": 19}
_LOGICAL = {"xor": 0, "and_not": 1, "and_not_kleene": 2}
_FLOAT_CLASS = {"is_nan": 0, "is_finite": 1, "is_inf": 2}


def _as_metal(x):
    return x if isinstance(x, MetalArray) else MetalArray.from_arrow(x)


def _trig_method(name, doc):
    op = _TRIG[name]

    def f(self):
        return _call(_lib.am_trig, self._h, op, None)

    f.__name__ = name
    f.__doc__ = doc
    return f


# float32 runs Metal's library functions (with the six hyperbolics written out from well-conditioned
# identities, because Metal's own lose accuracy and get +/-infinity wrong); float64 runs a software
# binary64 implementation on the GPU, since Metal has no double. Measured against the host libm over a
# million random arguments per function the worst case is 4 ulp (float32) and 5 ulp (float64).
for _name in ("sin", "cos", "tan", "asin", "acos", "atan",
              "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"):
    setattr(MetalArray, _name, _trig_method(
        _name, "Arrow `%s`, element-wise on a float32 or float64 column. Null in, null out." % _name))

# Arrow's `_checked` twins: same values, but a domain violation on a non-null row raises instead of
# returning NaN. asin/acos need |x| <= 1, acosh needs x >= 1, atanh needs |x| < 1, and sin/cos/tan
# reject +/-infinity. NaN never raises and null rows are never inspected, as in pyarrow.
for _name in ("sin_checked", "cos_checked", "tan_checked", "asin_checked",
              "acos_checked", "acosh_checked", "atanh_checked"):
    setattr(MetalArray, _name, _trig_method(
        _name, "Arrow `%s`: as %s, but raises on a domain violation." % (_name, _name[:-8])))
del _name


def _am_atan2(self, other):
    """Arrow `atan2(y, x)` with this column as y: the angle of (x, y) in [-pi, pi].

    `other` may be another column or a scalar (which is broadcast into a column first). Follows the
    C99 special-value table, including the four +/-0 and four +/-infinity cases."""
    if not isinstance(other, MetalArray):
        if hasattr(other, "__arrow_c_array__") or isinstance(other, (pa.Array, pa.ChunkedArray, list)):
            other = MetalArray.from_arrow(other)
        else:
            other = MetalArray.from_arrow(pa.array([other] * len(self), type=self.type))
    return _call(_lib.am_trig, self._h, _TRIG["atan2"], other._h)


MetalArray.atan2 = _am_atan2


def _logical_method(name, doc):
    op = _LOGICAL[name]

    def f(self, other):
        # Bind the import to a local: a temporary MetalArray would be released before the call.
        o = _as_metal(other)
        return _call(_lib.am_logical, self._h, op, o._h)

    f.__name__ = name
    f.__doc__ = doc
    return f


MetalArray.xor = _logical_method(
    "xor", "Arrow `xor` over two boolean columns. Nulls propagate (output validity is the AND).")
MetalArray.and_not = _logical_method(
    "and_not", "Arrow `and_not`: `a AND NOT b` over two boolean columns. Nulls propagate.")
MetalArray.and_not_kleene = _logical_method(
    "and_not_kleene",
    "Arrow `and_not_kleene`: three-valued `a AND NOT b`. A valid false on the left or a valid true "
    "on the right gives false even when the other side is null.")
MetalArray.__xor__ = MetalArray.xor


def _float_class_method(name, doc):
    op = _FLOAT_CLASS[name]

    def f(self):
        return _call(_lib.am_float_class, self._h, op)

    f.__name__ = name
    f.__doc__ = doc
    return f


# Arrow defines all three on every numeric type, not only the float ones, and propagates nulls.
MetalArray.is_nan = _float_class_method(
    "is_nan", "Arrow `is_nan`. False everywhere on an integer column; null where the input is null.")
MetalArray.is_finite = _float_class_method(
    "is_finite", "Arrow `is_finite`. True everywhere on an integer column; null where the input is null.")
MetalArray.is_inf = _float_class_method(
    "is_inf", "Arrow `is_inf`. False everywhere on an integer column; null where the input is null.")


def _am_fill_null_forward(self):
    """Arrow `fill_null_forward`: every null takes the value of the nearest non-null element before
    it. Leading nulls stay null. One GPU max-scan plus a gather."""
    return _call(_lib.am_fill_null_direction, self._h, 1)


def _am_fill_null_backward(self):
    """Arrow `fill_null_backward`: every null takes the value of the nearest non-null element after
    it. Trailing nulls stay null."""
    return _call(_lib.am_fill_null_direction, self._h, 0)


MetalArray.fill_null_forward = _am_fill_null_forward
MetalArray.fill_null_backward = _am_fill_null_backward


def case_when(conds, values, default=None):
    """Arrow `case_when`: each row takes the value of the first condition that is true.

    `conds` are boolean columns and `values` value columns of one type and length, one per condition;
    `default` (or None, giving nulls) supplies the rows no condition matches. A **null condition
    counts as false** and the row falls through, which is what Arrow does; a null in the chosen
    branch's values does make the output null.

        am.case_when([x > 10, x > 5], [big, medium], small)
    """
    cs = [_as_metal(c) for c in conds]
    vs = [_as_metal(v) for v in values]
    if len(cs) != len(vs) or not cs:
        raise ArrowMetalError(f"case_when needs one value column per condition, got {len(cs)} and {len(vs)}")
    ch = (_P * len(cs))(*[c._h for c in cs])
    vh = (_P * len(vs))(*[v._h for v in vs])
    out = _P()
    d = None if default is None else _as_metal(default)
    _check(_lib.am_case_when(ch, vh, len(cs), None if d is None else d._h, ctypes.byref(out)))
    return MetalArray(out)


def choose(indices, values):
    """Arrow `choose`: `values[indices[i]][i]`, element-wise.

    `indices` is an int32, int64 or uint32 column; a null index gives a null output, and an index
    outside `[0, len(values))` raises, as in Arrow."""
    idx = _as_metal(indices)
    vs = [_as_metal(v) for v in values]
    if not vs:
        raise ArrowMetalError("choose needs at least one value column")
    vh = (_P * len(vs))(*[v._h for v in vs])
    out = _P()
    _check(_lib.am_choose(idx._h, vh, len(vs), ctypes.byref(out)))
    return MetalArray(out)


def _am_replace_with_mask(self, mask, replacements):
    """Arrow `replace_with_mask`: rows where `mask` is true take the next value from `replacements`,
    in order; rows where the mask is null become null; every other row keeps its own value.

    `replacements` must hold at least as many elements as the mask has valid trues (fewer raises, a
    surplus is ignored, both as in pyarrow)."""
    # Bind the imports to locals: a temporary MetalArray would be released before the call.
    m, r = _as_metal(mask), _as_metal(replacements)
    return _call(_lib.am_replace_with_mask, self._h, m._h, r._h)


def _am_indices_nonzero(self):
    """Arrow `indices_nonzero`: the uint64 row numbers where the value is valid and not zero.

    `-0.0` counts as zero and every NaN counts as non-zero, as in Arrow."""
    return _call(_lib.am_indices_nonzero, self._h)


def _am_hash64(self):
    """A 64-bit hash of every element, as uint64 (an ArrowMetal extension: Arrow has no element-wise
    hash function).

    MurmurHash3's finaliser over the value's own bytes, seeded with the golden ratio; `-0.0` hashes
    as `+0.0` and every NaN as one canonical NaN, so Arrow-equal values always hash equal. Nulls hash
    to 0 and stay null. See include/arrowmetal.h for the exact definition."""
    return _call(_lib.am_hash64, self._h)


MetalArray.replace_with_mask = _am_replace_with_mask
MetalArray.indices_nonzero = _am_indices_nonzero
MetalArray.hash64 = _am_hash64
