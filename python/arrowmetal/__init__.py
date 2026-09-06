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
# ---- statistical and positional aggregates, run-end encoding (see include/arrowmetal.h)
_lib.am_reduce_ex.argtypes = [_P, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_int64),
                              ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                              ctypes.POINTER(ctypes.c_int)]
_lib.am_reduce_ex.restype = ctypes.c_int
for _name in ("am_run_end_encode", "am_run_end_decode"):
    getattr(_lib, _name).argtypes = [_P, ctypes.POINTER(_P)]
    getattr(_lib, _name).restype = ctypes.c_int

_REDUCE_EX = {"product": 0, "variance": 1, "variance_sample": 2, "stddev": 3, "stddev_sample": 4,
              "quantile": 5, "median": 6, "mode": 7, "count_distinct": 8, "first": 9, "last": 10,
              "index": 11, "any": 12, "all": 13, "min_of_min_max": 14, "max_of_min_max": 15,
              "mode_count": 16}


def _reduce_ex(self, op, p1=0.0):
    """One scalar aggregate through am_reduce_ex; None when the column has no answer."""
    i, f = ctypes.c_int64(), ctypes.c_double()
    kind, null = ctypes.c_int(), ctypes.c_int()
    _check(_lib.am_reduce_ex(self._h, _REDUCE_EX[op], float(p1), ctypes.byref(i), ctypes.byref(f),
                             ctypes.byref(kind), ctypes.byref(null)))
    if null.value:
        return None
    if kind.value == 2:
        return f.value
    if kind.value == 1:
        return i.value & 0xFFFFFFFFFFFFFFFF
    return i.value


def _product(self):
    """Arrow `product` of the non-null values (integers wrap in 64 bits)."""
    return _reduce_ex(self, "product")


def _variance(self, ddof=0):
    """Population variance (ddof=0) or sample variance (ddof=1) of the non-null values."""
    return _reduce_ex(self, "variance" if ddof == 0 else "variance_sample")


def _stddev(self, ddof=0):
    """Population or sample standard deviation."""
    return _reduce_ex(self, "stddev" if ddof == 0 else "stddev_sample")


def _quantile(self, q):
    """Exact quantile with linear interpolation; q is clamped to [0, 1]."""
    return _reduce_ex(self, "quantile", q)


def _median(self):
    """Arrow `approximate_median`, computed exactly (the values are sorted on the GPU)."""
    return _reduce_ex(self, "median")


def _mode(self):
    """(value, count) of the most common non-null value; ties go to the smallest value."""
    value = _reduce_ex(self, "mode")
    if value is None:
        return None
    return (value, _reduce_ex(self, "mode_count"))


def _count_distinct(self):
    """Number of distinct non-null values."""
    return _reduce_ex(self, "count_distinct")


def _first(self):
    """First non-null value, or None."""
    return _reduce_ex(self, "first")


def _last(self):
    """Last non-null value, or None."""
    return _reduce_ex(self, "last")


def _index(self, value):
    """Row of the first occurrence of `value`, or -1 when it is absent."""
    return _reduce_ex(self, "index", value)


def _min_max(self):
    """(min, max) in one pass over the values."""
    return (_reduce_ex(self, "min_of_min_max"), _reduce_ex(self, "max_of_min_max"))


def _any(self):
    """True when any valid value of a boolean column is true."""
    v = _reduce_ex(self, "any")
    return None if v is None else bool(v)


def _all(self):
    """True when every valid value of a boolean column is true."""
    v = _reduce_ex(self, "all")
    return None if v is None else bool(v)


def _run_end_encode(self):
    """Run-end encode a primitive, boolean or temporal column ("+r": run_ends plus values)."""
    return _call(_lib.am_run_end_encode, self._h)


def _run_end_decode(self):
    """Expand a run-end encoded column back into a flat one."""
    return _call(_lib.am_run_end_decode, self._h)


MetalArray.product = _product
MetalArray.variance = _variance
MetalArray.stddev = _stddev
MetalArray.quantile = _quantile
MetalArray.median = _median
MetalArray.mode = _mode
MetalArray.count_distinct = _count_distinct
MetalArray.first = _first
MetalArray.last = _last
MetalArray.index = _index
MetalArray.min_max = _min_max
MetalArray.any = _any
MetalArray.all = _all
MetalArray.run_end_encode = _run_end_encode
MetalArray.run_end_decode = _run_end_decode
# ---- the remaining Arrow string surface: character-class predicates, capitalize / title / center /
# replace-slice / trim / normalize, extract_regex_span, binary_join and string is_in / index_in.
# Op numbering is the C ABI contract; see include/arrowmetal.h.
_lib.am_string_predicate.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_string_predicate.restype = ctypes.c_int
_lib.am_string_transform.argtypes = [_P, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
                                     ctypes.POINTER(_P)]
_lib.am_string_transform.restype = ctypes.c_int
_lib.am_string_is_in.argtypes = [_P, _P, ctypes.POINTER(_P)]
_lib.am_string_is_in.restype = ctypes.c_int
_lib.am_string_index_in.argtypes = [_P, _P, ctypes.POINTER(_P)]
_lib.am_string_index_in.restype = ctypes.c_int
_lib.am_binary_join.argtypes = [_P, ctypes.c_char_p, ctypes.c_int64, _P, ctypes.POINTER(_P)]
_lib.am_binary_join.restype = ctypes.c_int

_STRING_PREDICATE = {"ascii_is_printable": 0, "ascii_is_title": 1, "string_is_ascii": 2,
                     "utf8_is_alnum": 3, "utf8_is_alpha": 4, "utf8_is_decimal": 5, "utf8_is_digit": 6,
                     "utf8_is_lower": 7, "utf8_is_numeric": 8, "utf8_is_printable": 9,
                     "utf8_is_space": 10, "utf8_is_title": 11, "utf8_is_upper": 12}
_STRING_EXTRA = {"ascii_title": 0, "utf8_capitalize": 1, "utf8_title": 2, "utf8_center": 3,
                 "utf8_replace_slice": 4, "binary_replace_slice": 5,
                 "utf8_trim": 6, "utf8_ltrim": 7, "utf8_rtrim": 8,
                 "utf8_trim_whitespace": 9, "utf8_ltrim_whitespace": 10, "utf8_rtrim_whitespace": 11,
                 "utf8_normalize": 12, "extract_regex_span_start": 13, "extract_regex_span_length": 14}
_NORMALIZATION_FORMS = {"NFC": 0, "NFKC": 1, "NFD": 2, "NFKD": 3}
_STRING_FORMATS = ("u", "U", "z", "Z")


def _am_string_predicate(self, op):
    """One character-class predicate by name; the table is in include/arrowmetal.h."""
    return _call(_lib.am_string_predicate, self._h, _STRING_PREDICATE[op])


def _am_string_transform(self, op, p1=0, p2=0, arg1="", arg2=""):
    """One of the extra string transforms by name; the table is in include/arrowmetal.h."""
    a1 = arg1.encode("utf-8") if isinstance(arg1, str) else bytes(arg1)
    a2 = arg2.encode("utf-8") if isinstance(arg2, str) else bytes(arg2)
    return _call(_lib.am_string_transform, self._h, _STRING_EXTRA[op], p1, p2,
                 a1, len(a1), a2, len(a2))


MetalArray.string_predicate = _am_string_predicate
MetalArray.string_transform = _am_string_transform


def _predicate_method(op, doc):
    def f(self):
        return _am_string_predicate(self, op)
    f.__name__ = op
    f.__doc__ = doc
    return f


# The Unicode predicates run on the GPU for every row whose bytes are all < 0x80 and on the CPU
# (Swift's Unicode.Scalar.Properties, sharded over 4096-row chunks) for the rest, so an ASCII-only
# column never leaves the device. Nulls propagate.
for _op, _doc in [
    ("ascii_is_printable", "Every byte is in 0x20-0x7E. The empty string is true."),
    ("ascii_is_title", "Byte-wise title case over runs of ASCII letters; at least one letter."),
    ("string_is_ascii", "Every byte is < 0x80. The empty string is true."),
    ("utf8_is_alnum", "Non-empty and every code point is a letter or a number."),
    ("utf8_is_alpha", "Non-empty and every code point is in an L* category."),
    ("utf8_is_decimal", "Non-empty and every code point is category Nd."),
    ("utf8_is_digit", "Non-empty and every code point is category Nd or No."),
    ("utf8_is_lower", "At least one cased code point and no upper-case one."),
    ("utf8_is_numeric", "Non-empty and every code point is category Nd, Nl or No."),
    ("utf8_is_printable", "No Cc/Cf/Cs/Co/Cn/Zs/Zl/Zp code point, except U+0020. Empty is true."),
    ("utf8_is_space", "Non-empty and every code point is Unicode whitespace (U+200B is not)."),
    ("utf8_is_title", "At least one cased code point, in title case."),
    ("utf8_is_upper", "At least one cased code point and no lower-case one."),
]:
    setattr(MetalArray, _op, _predicate_method(_op, _doc))
del _op, _doc


def _am_ascii_title(self):
    """Arrow `ascii_title`: byte-wise, the first ASCII letter of every run of ASCII letters is upper-
    cased and the rest lower-cased, so "unicode" with accents becomes uNicoDe. Always GPU."""
    return _am_string_transform(self, "ascii_title")


def _am_utf8_capitalize(self):
    """Arrow `utf8_capitalize`: first code point upper-cased, every later one lower-cased."""
    return _am_string_transform(self, "utf8_capitalize")


def _am_utf8_title(self):
    """Arrow `utf8_title`: the first cased code point of every word upper-cased, the rest lower-cased,
    where a word is a maximal run of cased code points."""
    return _am_string_transform(self, "utf8_title")


def _am_utf8_center(self, width, pad=" "):
    """Arrow `utf8_center`: pads to `width` code points on both sides, the odd pad character going on
    the right ("a" centred in 4 is "*a**"). `pad` must be exactly one code point."""
    return _am_string_transform(self, "utf8_center", p1=width, arg1=pad)


def _am_utf8_replace_slice(self, start, stop, replacement):
    """Arrow `utf8_replace_slice`: replaces code points [start, stop) with `replacement`. Negative
    indices count from the end, both ends clamp, and stop < start inserts without deleting."""
    return _am_string_transform(self, "utf8_replace_slice", p1=start, p2=stop, arg1=replacement)


def _am_binary_replace_slice(self, start, stop, replacement):
    """Arrow `binary_replace_slice`: the same substitution indexed in bytes, returning `binary`."""
    return _am_string_transform(self, "binary_replace_slice", p1=start, p2=stop, arg1=replacement)


def _trim_method(with_set, without_set, doc):
    def f(self, characters=None):
        if characters is None:
            return _am_string_transform(self, without_set)
        return _am_string_transform(self, with_set, arg1=characters)
    f.__name__ = with_set
    f.__doc__ = doc
    return f


def _am_utf8_normalize(self, form):
    """Arrow `utf8_normalize`: "NFC", "NFKC", "NFD" or "NFKD". Always CPU (Foundation)."""
    key = form.upper() if isinstance(form, str) else form
    if key not in _NORMALIZATION_FORMS:
        raise ArrowMetalError("normalisation form must be one of NFC, NFKC, NFD, NFKD, got %r" % (form,))
    return _am_string_transform(self, "utf8_normalize", p1=_NORMALIZATION_FORMS[key])


def _am_extract_regex_span(self, pattern, ignore_case=False):
    """Arrow `extract_regex_span`, as {group name: (start, length)} pairs of int32 arrays.

    Offsets and lengths count bytes, as Arrow's do. The pattern needs at least one (?<name>...) group;
    a row that does not match, a null row, and a group that took part in no alternative are null in
    both arrays. Always CPU (ICU, sharded over 4096-row chunks)."""
    import re as _re
    names = _re.findall(r"\(\?P?<([A-Za-z_][A-Za-z0-9_]*)>", pattern)
    if not names:
        raise ArrowMetalError("extract_regex_span needs at least one named group, e.g. (?<year>\\d+)")
    flags = 1 if ignore_case else 0
    return {n: (_am_string_transform(self, "extract_regex_span_start", p2=flags, arg1=pattern, arg2=n),
                _am_string_transform(self, "extract_regex_span_length", p2=flags, arg1=pattern, arg2=n))
            for n in dict.fromkeys(names)}


def _am_binary_join(self, separator):
    """Arrow `binary_join`: joins the child strings of every row of a list<utf8>.

    `separator` is a scalar string or a per-row string column. An empty row joins to the empty string;
    a null row, any null element inside a row, and a null separator give a null output row."""
    if isinstance(separator, str):
        sep = separator.encode("utf-8")
        return _call(_lib.am_binary_join, self._h, sep, len(sep), None)
    s = separator if isinstance(separator, MetalArray) else MetalArray.from_arrow(separator)
    return _call(_lib.am_binary_join, self._h, b"", 0, s._h)


MetalArray.ascii_title = _am_ascii_title
MetalArray.utf8_capitalize = _am_utf8_capitalize
MetalArray.utf8_title = _am_utf8_title
MetalArray.utf8_center = _am_utf8_center
MetalArray.utf8_replace_slice = _am_utf8_replace_slice
MetalArray.binary_replace_slice = _am_binary_replace_slice
MetalArray.utf8_trim = _trim_method(
    "utf8_trim", "utf8_trim_whitespace",
    "Arrow `utf8_trim` / `utf8_trim_whitespace`: strips both ends, of Unicode whitespace by default or "
    "of any code point in `characters`. GPU for an ASCII character set or an all-ASCII column.")
MetalArray.utf8_ltrim = _trim_method(
    "utf8_ltrim", "utf8_ltrim_whitespace", "Leading-only form of utf8_trim.")
MetalArray.utf8_rtrim = _trim_method(
    "utf8_rtrim", "utf8_rtrim_whitespace", "Trailing-only form of utf8_trim.")
MetalArray.utf8_normalize = _am_utf8_normalize
MetalArray.extract_regex_span = _am_extract_regex_span
MetalArray.binary_join = _am_binary_join

# is_in() and index_in() gain string support: a utf8 or binary column goes to the GPU string hash
# table, everything else keeps the primitive binary-search path.
_am_primitive_is_in = MetalArray.is_in
_am_primitive_index_in = MetalArray.index_in


def _am_is_in_any(self, values):
    """Arrow `is_in`: boolean array, true where the element is among the non-null `values`.

    Nulls in `values` are ignored and a null element is never in the set, so the result never has
    nulls (pyarrow's `skip_nulls=True`; `skip_nulls=False` is not implemented)."""
    if self.format in _STRING_FORMATS:
        s = values if isinstance(values, MetalArray) else MetalArray.from_arrow(values)
        return _call(_lib.am_string_is_in, self._h, s._h)
    return _am_primitive_is_in(self, values)


def _am_index_in_any(self, values):
    """Arrow `index_in`: int32 index into `values` of each element's first occurrence there, null
    where the element is null or absent."""
    if self.format in _STRING_FORMATS:
        s = values if isinstance(values, MetalArray) else MetalArray.from_arrow(values)
        return _call(_lib.am_string_index_in, self._h, s._h)
    return _am_primitive_index_in(self, values)


MetalArray.is_in = _am_is_in_any
MetalArray.index_in = _am_index_in_any
# ---- the remaining Arrow type-matrix rows and the type-adjacent functions
#
# null ("n"), float16 ("e"), decimal32 / decimal64 ("d:p,s,32" / "d:p,s,64"), the three interval layouts
# ("tiM", "tiD", "tin"), fixed_size_binary ("w:N"), list_view / large_list_view ("+vl" / "+vL", imported
# as "+l"), extension types, and the timezone functions. Appended rather than written into the class body
# so this section stays self-contained; op numbering is the C ABI contract (see include/arrowmetal.h).
for _name, _extra in [("am_cast_float16", [ctypes.c_int]), ("am_decimal_widen", []),
                      ("am_decimal_narrow", [ctypes.c_int, ctypes.c_int64]),
                      ("am_fixed_binary_compare", [ctypes.c_int, _P, ctypes.c_char_p, ctypes.c_int64]),
                      ("am_fixed_binary_hash64", []), ("am_add_interval", [_P]),
                      ("am_list_parent_indices", []),
                      ("am_list_slice", [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]),
                      ("am_map_lookup", [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int]),
                      ("am_assume_timezone", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]),
                      ("am_local_timestamp", []), ("am_extension_storage", []),
                      ("am_extension_wrap", [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]),
                      ("am_interval_between", [_P, ctypes.c_int]),
                      ("am_interval_field", [ctypes.c_int])]:
    getattr(_lib, _name).argtypes = [_P] + _extra + [ctypes.POINTER(_P)]
    getattr(_lib, _name).restype = ctypes.c_int
_lib.am_extension_name.argtypes = [_P]
_lib.am_extension_name.restype = ctypes.c_char_p
_lib.am_extension_metadata.argtypes = [_P, ctypes.POINTER(ctypes.c_int64)]
_lib.am_extension_metadata.restype = ctypes.c_char_p
_lib.am_null_array.argtypes = [ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_null_array.restype = ctypes.c_int

_OCCURRENCE = {"first": 0, "last": 1, "all": 2}
_TZ_HANDLING = {"raise": 0, "earliest": 1, "latest": 2}
_INTERVAL_BETWEEN = {"month": 0, "day_time": 1, "month_day_nano": 2}
_INTERVAL_FIELDS = {"months": 0, "days": 1, "nanoseconds": 2}
# pyarrow can build a type object for month_day_nano_interval only; interval[month] and
# interval[day_time] have no Python type or Array class (pyarrow 25), so `type` and `to_arrow()` raise
# for those two and `interval_field()` is the way to read their values.
_INTERVAL_TYPES = {"tiM": "interval[month]", "tiD": "interval[day_time]", "tin": "interval[month_day_nano]"}


def _extra_type(fmt):
    """pyarrow type for one of the formats this section adds, or None when it is not one of them."""
    if fmt == "n":
        return pa.null()
    if fmt == "e":
        return pa.float16()
    if fmt.startswith("w:"):
        return pa.binary(int(fmt[2:]))
    if fmt == "tin":
        return pa.month_day_nano_interval()
    if fmt in ("tiM", "tiD"):
        raise ArrowMetalError(
            f"pyarrow has no Python type for {_INTERVAL_TYPES[fmt]}; read the values with "
            "interval_field('months' | 'days' | 'nanoseconds')")
    if fmt.startswith("d:"):
        parts = fmt[2:].split(",")
        if len(parts) == 3 and parts[2] in ("32", "64"):
            p, s = int(parts[0]), int(parts[1])
            return pa.decimal32(p, s) if parts[2] == "32" else pa.decimal64(p, s)
    return None


_type_before_extra = MetalArray.type


@property
def _type_with_extra(self):
    """pyarrow type of this array, including the types added by this section. An extension column
    reports its extension type (the schema carries `ARROW:extension:name`), not its storage type."""
    if self.extension_name is not None:
        return self.to_arrow().type
    t = _extra_type(self.format)
    if t is not None:
        return t
    return _type_before_extra.fget(self)


def _to_float32(self):
    """Arrow `cast(float32)` of a float16 column: exact, one GPU pass through Metal's native `half`."""
    return _call(_lib.am_cast_float16, self._h, 0)


def _to_float16(self):
    """Arrow `cast(float16)` of a float32 column: round to nearest-even on the GPU, overflowing to
    +/-infinity. Arithmetic is never done in half precision - compute in float32 and cast back here."""
    return _call(_lib.am_cast_float16, self._h, 1)


def _to_decimal128(self):
    """GPU widening cast of a decimal32 / decimal64 column to decimal128, which is where every decimal
    kernel lives. A decimal128 column comes back unchanged."""
    return _call(_lib.am_decimal_widen, self._h)


def _to_small_decimal(self, bit_width, precision=0):
    """GPU narrowing cast of a decimal128 column back to decimal32 (`bit_width` 32) or decimal64 (64),
    keeping the scale. A value that does not fit wraps, which is Arrow's unchecked cast."""
    if bit_width not in (32, 64):
        raise ArrowMetalError("bit_width must be 32 or 64")
    return _call(_lib.am_decimal_narrow, self._h, bit_width, precision)


def _fixed_binary_compare(self, op, other):
    """Arrow `equal` / `not_equal` over a fixed_size_binary column, on the GPU (byte compare).
    `other` is another MetalArray of the same width, or a bytes value exactly one element wide."""
    code = {"==": 0, "eq": 0, "equal": 0, "!=": 1, "ne": 1, "not_equal": 1}.get(op)
    if code is None:
        raise ArrowMetalError("fixed_size_binary supports == and != only")
    if isinstance(other, MetalArray):
        return _call(_lib.am_fixed_binary_compare, self._h, code, other._h, None, 0)
    b = other.encode("utf-8") if isinstance(other, str) else bytes(other)
    return _call(_lib.am_fixed_binary_compare, self._h, code, None, b, len(b))


def _fixed_binary_hash64(self):
    """FNV-1a 64 over each element's bytes (an ArrowMetal extension, not Arrow's `hash64`), on the GPU.
    Null in, null out; the result is a uint64 column."""
    return _call(_lib.am_fixed_binary_hash64, self._h)


def _add_interval(self, interval):
    """Arrow `add(timestamp | date, interval)` on the GPU. `interval` is an interval column of the same
    length, or of length 1 to broadcast. Month arithmetic clamps the day to the target month's length
    (2024-01-31 + 1 month = 2024-02-29), as Arrow does; days are whole UTC days and the sub-day part is
    truncated toward zero when the column is coarser than the interval."""
    iv = interval if isinstance(interval, MetalArray) else MetalArray.from_arrow(interval)
    return _call(_lib.am_add_interval, self._h, iv._h)


def _list_parent_indices(self):
    """Arrow `list_parent_indices`: for every child element the list references, the index of the row
    that covers it. GPU (one binary search per element). ArrowMetal returns **int32** where pyarrow
    returns int64, because list offsets are int32 throughout this package; the values are the same."""
    return _call(_lib.am_list_parent_indices, self._h)


def _list_slice(self, start, stop=None, step=1):
    """Arrow `list_slice`: `row[start:stop:step]` for every row, as a variable-length list. `stop=None`
    slices to the end of each row. `start` must be >= 0 and `step` >= 1, as Arrow requires; a null row
    stays null and a row shorter than `start` becomes empty. GPU."""
    return _call(_lib.am_list_slice, self._h, start, -1 if stop is None else stop, step)


def _map_lookup(self, key, occurrence="first"):
    """Arrow `map_lookup`: the value(s) whose key matches, per row. `occurrence` is "first", "last" or
    "all"; "all" returns a list of the item type. Both are null where the row is null or the key is
    absent, matching pyarrow. Keys may be utf8 / binary (pass a str or bytes) or an integer type (pass
    an int). GPU: one key compare per entry inside each row's range."""
    occ = _OCCURRENCE.get(occurrence)
    if occ is None:
        raise ArrowMetalError('occurrence must be "first", "last" or "all"')
    if isinstance(key, bool) or not isinstance(key, (str, bytes, bytearray, int)):
        raise ArrowMetalError("map_lookup key must be a str, bytes or int")
    if isinstance(key, int):
        b = int(key).to_bytes(8, "little", signed=True)
    else:
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    return _call(_lib.am_map_lookup, self._h, b, len(b), occ)


def _assume_timezone(self, tz, ambiguous="raise", nonexistent="raise"):
    """Arrow `assume_timezone`: reads a naive timestamp column as wall-clock times in `tz` and returns
    the instants they name, tagged with that timezone. The unit and the sub-second part are unchanged.

    **CPU**: the tz database is host data, so the per-value offsets are computed on the host (sharded
    across cores) rather than on the GPU. A local time that occurs twice or never is an error by
    default; pass "earliest" / "latest" to pick one, as Arrow does."""
    a = _TZ_HANDLING.get(ambiguous)
    n = _TZ_HANDLING.get(nonexistent)
    if a is None or n is None:
        raise ArrowMetalError('ambiguous / nonexistent must be "raise", "earliest" or "latest"')
    return _call(_lib.am_assume_timezone, self._h, tz.encode("utf-8"), a, n)


def _local_timestamp(self):
    """Arrow `local_timestamp`: the wall-clock time each instant names in the column's own timezone, as
    a naive timestamp of the same unit. A column with no timezone comes back unchanged. **CPU**, for
    the same reason as `assume_timezone`."""
    return _call(_lib.am_local_timestamp, self._h)


def _interval_between(self, other, kind):
    """One of Arrow's three interval-producing differences (`self` is `start`, `other` is `end`), on the
    GPU: "month", "day_time" or "month_day_nano". Every field is the difference of the corresponding
    truncated field, as Arrow defines it - months are month boundaries crossed, and the day and sub-day
    fields may have the opposite sign."""
    code = _INTERVAL_BETWEEN.get(kind)
    if code is None:
        raise ArrowMetalError('kind must be "month", "day_time" or "month_day_nano"')
    b = other if isinstance(other, MetalArray) else MetalArray.from_arrow(other)
    return _call(_lib.am_interval_between, self._h, b._h, code)


def _interval_field(self, field):
    """One field of an interval column as a plain integer column: "months" (int32), "days" (int32) or
    "nanoseconds" (int64). A field the layout does not carry comes back as zeros. This is how to read
    an interval[month] or interval[day_time] column, which pyarrow cannot wrap in Python."""
    code = _INTERVAL_FIELDS.get(field)
    if code is None:
        raise ArrowMetalError('field must be "months", "days" or "nanoseconds"')
    return _call(_lib.am_interval_field, self._h, code)


@property
def _extension_name(self):
    """`ARROW:extension:name` of an extension column, or None when this is not an extension type."""
    v = _lib.am_extension_name(self._h)
    return None if v is None else v.decode()


@property
def _extension_metadata(self):
    """`ARROW:extension:metadata` of an extension column as bytes, or None when there is none."""
    n = ctypes.c_int64()
    v = _lib.am_extension_metadata(self._h, ctypes.byref(n))
    return None if v is None else v[:n.value]


def _extension_storage(self):
    """The storage column of an extension array (the array itself for every other type)."""
    return _call(_lib.am_extension_storage, self._h)


def _as_extension_type(self, name, metadata=None):
    """Tags this column as the storage of an extension type, so `to_arrow()` writes
    `ARROW:extension:name` / `:metadata` and pyarrow rebuilds the extension type when it is registered."""
    md = None if metadata is None else (metadata.encode("utf-8") if isinstance(metadata, str) else bytes(metadata))
    return _call(_lib.am_extension_wrap, self._h, name.encode("utf-8"), md, 0 if md is None else len(md))


MetalArray.to_float32 = _to_float32
MetalArray.to_float16 = _to_float16
MetalArray.to_decimal128 = _to_decimal128
MetalArray.to_small_decimal = _to_small_decimal
MetalArray.fixed_binary_compare = _fixed_binary_compare
MetalArray.hash64 = _fixed_binary_hash64
MetalArray.add_interval = _add_interval
MetalArray.list_parent_indices = _list_parent_indices
MetalArray.list_slice = _list_slice
MetalArray.map_lookup = _map_lookup
MetalArray.assume_timezone = _assume_timezone
MetalArray.local_timestamp = _local_timestamp
MetalArray.interval_between = _interval_between
MetalArray.interval_field = _interval_field
MetalArray.extension_name = _extension_name
MetalArray.extension_metadata = _extension_metadata
MetalArray.extension_storage = _extension_storage
MetalArray.as_extension_type = _as_extension_type
MetalArray.month_interval_between = lambda self, other: _interval_between(self, other, "month")
MetalArray.day_time_interval_between = lambda self, other: _interval_between(self, other, "day_time")
MetalArray.month_day_nano_interval_between = lambda self, other: _interval_between(self, other, "month_day_nano")
# `type` is replaced last, so `extension_name` is already attached when the property runs.
MetalArray.type = _type_with_extra


def nulls(length):
    """A `null` column of `length` elements: every value null, no buffers."""
    out = _P()
    _check(_lib.am_null_array(length, ctypes.byref(out)))
    return MetalArray(out)
# ---- the temporal functions beyond am_temporal_extract / am_temporal_math: the option-carrying week
# numbers, the struct-valued extractors, subsecond, is_dst and every *_between difference.
# Appended rather than written into the class body so the file stays additive.
# Op numbering is the C ABI contract; see include/arrowmetal.h.
_lib.am_temporal_extra.argtypes = [_P, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, _P, ctypes.POINTER(_P)]
_lib.am_temporal_extra.restype = ctypes.c_int

_TEMPORAL_EXTRA = {"week": 0, "us_week": 1, "us_year": 2, "iso_calendar": 3, "year_month_day": 4,
                   "is_dst": 5, "day_of_week": 6, "subsecond": 7,
                   "years_between": 8, "quarters_between": 9, "months_between": 10,
                   "weeks_between": 11, "hours_between": 12, "minutes_between": 13,
                   "seconds_between": 14, "milliseconds_between": 15,
                   "microseconds_between": 16, "nanoseconds_between": 17}


def _am_temporal_extra(self, op, p1=0, p2=0, other=None):
    """One temporal op by name; the table is in include/arrowmetal.h."""
    return _call(_lib.am_temporal_extra, self._h, _TEMPORAL_EXTRA[op], int(p1), int(p2),
                 other._h if other is not None else None)


def _am_week(self, week_starts_monday=True, count_from_zero=False, first_week_is_fully_in_year=False):
    """Arrow `week` with the full WeekOptions, int64, UTC.

    `count_from_zero` numbers the weeks against the value's own calendar year, so a date at the start
    of a year that belongs to the previous year's last week comes out as 0 rather than 52 or 53.
    `first_week_is_fully_in_year` makes week 1 the first week lying wholly inside January; without it
    the ISO majority rule applies and a week beginning on 29, 30 or 31 December is week 1 of the next
    year. The defaults reproduce iso_week."""
    bits = ((1 if week_starts_monday else 0) | (2 if count_from_zero else 0)
            | (4 if first_week_is_fully_in_year else 0))
    return _am_temporal_extra(self, "week", bits)


def _am_weeks_between(self, other, count_from_zero=True, week_start=1):
    """Arrow `weeks_between(self, other)`: week boundaries crossed, both sides floored to the start of
    their week first. `week_start` is 1 = Monday ... 7 = Sunday. `count_from_zero` is part of Arrow's
    DayOfWeekOptions and is accepted for signature parity, but does not change the answer."""
    return _am_temporal_extra(self, "weeks_between", 1 if count_from_zero else 0, week_start,
                              other=other)


def _am_day_of_week_options(self, count_from_zero=True, week_start=1):
    """Arrow `day_of_week` with DayOfWeekOptions; `week_start` uses the ISO numbering (1 = Monday ...
    7 = Sunday) and is unaffected by `count_from_zero`. The default options keep the existing int32
    result (Monday = 0); any other combination returns int64, as pyarrow does."""
    if count_from_zero and week_start == 1:
        return _am_day_of_week_int32(self)
    return _am_temporal_extra(self, "day_of_week", 1 if count_from_zero else 0, week_start)


def _no_arg_temporal_extra(op, doc):
    def f(self):
        return _am_temporal_extra(self, op)
    f.__name__ = op
    f.__doc__ = doc
    return f


def _between_temporal_extra(op, doc):
    def f(self, other):
        return _am_temporal_extra(self, op, other=other)
    f.__name__ = op
    f.__doc__ = doc
    return f


_am_day_of_week_int32 = MetalArray.day_of_week
MetalArray.week = _am_week
MetalArray.weeks_between = _am_weeks_between
MetalArray.day_of_week = _am_day_of_week_options

for _op, _doc in [
    ("us_week", "Arrow us_week: the week number with Sunday-start weeks and the majority rule, 1-53."),
    ("us_year", "Arrow us_year: the US epidemiological week-numbering year, that is the year owning "
                "the Wednesday of this date's Sunday-start week."),
    ("iso_calendar", "Arrow iso_calendar: a struct of int64 iso_year, iso_week and iso_day_of_week "
                     "(1 = Monday). Read a field with struct_field(name)."),
    ("year_month_day", "Arrow year_month_day: a struct of int64 year, month and day."),
    ("is_dst", "Arrow is_dst: whether each value falls in daylight saving time in the column's own "
               "timezone. Needs a timestamp carrying a timezone; a naive one is an error. CPU."),
    ("subsecond", "Arrow subsecond: the fraction of a second, in [0, 1), as float64. date32 and "
                  "date64 answer 0 (pyarrow has no kernel for them); duration is rejected."),
]:
    setattr(MetalArray, _op, _no_arg_temporal_extra(_op, _doc))
del _op, _doc

# Every *_between counts boundaries crossed from self to other: each side is truncated to the unit
# first and the difference taken afterwards, so it is not the truncated difference. Positive when
# `other` is later. The two sides may differ in unit and in type, which is more permissive than
# pyarrow, where both arguments must have the same type.
for _op, _doc in [
    ("years_between", "Arrow years_between: the difference of the two calendar years."),
    ("quarters_between", "Arrow quarters_between: the difference of year * 4 + quarter."),
    ("months_between", "The int64 month difference, that is the difference of year * 12 + month. "
                       "Arrow spells the same quantity month_interval_between and returns an interval."),
    ("hours_between", "Arrow hours_between: hour boundaries crossed."),
    ("minutes_between", "Arrow minutes_between: minute boundaries crossed."),
    ("seconds_between", "Arrow seconds_between: second boundaries crossed."),
    ("milliseconds_between", "Arrow milliseconds_between: millisecond boundaries crossed."),
    ("microseconds_between", "Arrow microseconds_between: microsecond boundaries crossed."),
    ("nanoseconds_between", "Arrow nanoseconds_between: nanosecond boundaries crossed; wraps in "
                            "int64 past about 292 years, as Arrow's does."),
]:
    setattr(MetalArray, _op, _between_temporal_extra(_op, _doc))
del _op, _doc


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

# ---- hash64 is defined for both fixed_size_binary (FNV-1a over the element bytes) and the
# primitive types (the 64-bit value hash); dispatch on the column's format.
_hash64_primitive = _am_hash64


def _hash64_any(self):
    """64-bit hash per element on the GPU: FNV-1a over the bytes of a fixed_size_binary column,
    the value hash for primitive and boolean columns. Null in, null out."""
    if self.format.startswith("w:"):
        return _fixed_binary_hash64(self)
    return _hash64_primitive(self)


MetalArray.hash64 = _hash64_any


# ---- group-by over arbitrary key columns, the rest of the grouped aggregates, and the scalar
# skew / kurtosis / tdigest. Mirrors the op tables in include/arrowmetal.h.
_lib.am_group_by_keys.argtypes = [ctypes.POINTER(_P), ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_group_by_keys.restype = ctypes.c_int
_lib.am_group_by_group_count.argtypes = [_P]
_lib.am_group_by_group_count.restype = ctypes.c_int64
_lib.am_group_by_keys_result.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_group_by_keys_result.restype = ctypes.c_int
_lib.am_group_by_ids.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_group_by_ids.restype = ctypes.c_int
_lib.am_group_by_release.argtypes = [_P]
_lib.am_group_agg_ex.argtypes = [_P, _P, ctypes.c_int, ctypes.c_double, ctypes.POINTER(_P)]
_lib.am_group_agg_ex.restype = ctypes.c_int
_lib.am_group_pivot_wider.argtypes = [_P, _P, _P, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
                                      ctypes.POINTER(_P)]
_lib.am_group_pivot_wider.restype = ctypes.c_int
_lib.am_reduce_ex2.argtypes = [_P, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_int)]
_lib.am_reduce_ex2.restype = ctypes.c_int

_GROUP_AGG = {"sum": 0, "count_all": 1, "count": 2, "mean": 3, "min": 4, "max": 5, "min_max": 6,
              "first": 7, "last": 8, "first_last": 9, "one": 10, "list": 11, "distinct": 12,
              "count_distinct": 13, "any": 14, "all": 15, "product": 16,
              "variance": 17, "variance_sample": 18, "stddev": 19, "stddev_sample": 20,
              "approximate_median": 21, "quantile": 22, "skew": 23, "kurtosis": 24, "tdigest": 25}
_REDUCE_EX2 = {"skew": 0, "kurtosis": 1, "tdigest": 2, "skew_sample": 3, "kurtosis_sample": 4}


class GroupByKeys:
    """Arrow `hash_*` aggregation over arbitrary key columns.

    Built by `am.group_by([...])`. The key columns are mapped to dense group ids on the GPU — any key
    type works (integers sparse or negative, float32/float64 with -0.0 == 0.0 and one NaN group, bool,
    temporal, utf8, binary, dictionary, decimal) and several columns fold together — and every
    aggregate then runs against those ids.

        gb = am.group_by([region, year])
        keys = gb.keys()                       # one pyarrow array per key column, one row per group
        totals = gb.sum(revenue).to_arrow()

    A null key forms its own group, as in Arrow. The group ORDER is deterministic but is not pyarrow's
    first-seen order: label the rows with `keys()` and sort both sides before comparing.
    """

    def __init__(self, columns):
        cols = [c if isinstance(c, MetalArray) else MetalArray.from_arrow(c) for c in columns]
        if not cols:
            raise ArrowMetalError("group_by needs at least one key column")
        self._columns = cols
        handles = (_P * len(cols))(*[c._h for c in cols])
        out = _P()
        _check(_lib.am_group_by_keys(handles, len(cols), ctypes.byref(out)))
        self._h = out
        self.group_count = _lib.am_group_by_group_count(self._h)

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                _lib.am_group_by_release(self._h)
                self._h = None
        except Exception:
            pass

    def __len__(self):
        return self.group_count

    def keys(self):
        """The distinct key values, one row per group, in group order: one pyarrow array per key column."""
        out = []
        for i in range(len(self._columns)):
            h = _P()
            _check(_lib.am_group_by_keys_result(self._h, i, ctypes.byref(h)))
            out.append(MetalArray(h).to_arrow())
        return out

    def ids(self):
        """The dense group id of every row (int32, never null)."""
        return _call(_lib.am_group_by_ids, self._h)

    def _agg(self, name, values, p1=0.0):
        # Bind the lifted column to a local: a temporary MetalArray would be released — and its
        # device handle freed — before `am_group_agg_ex` ever read it.
        column = None if values is None else (values if isinstance(values, MetalArray)
                                              else MetalArray.from_arrow(values))
        out = _P()
        _check(_lib.am_group_agg_ex(self._h, None if column is None else column._h,
                                    _GROUP_AGG[name], float(p1), ctypes.byref(out)))
        return MetalArray(out)

    def sum(self, values): return self._agg("sum", values)
    def count(self, values): return self._agg("count", values)
    def count_all(self): return self._agg("count_all", None)
    def mean(self, values): return self._agg("mean", values)
    def min(self, values): return self._agg("min", values)
    def max(self, values): return self._agg("max", values)

    def min_max(self, values):
        """Arrow `hash_min_max`: one struct<min, max> column, both extremes from one read of the values."""
        return self._agg("min_max", values)

    def first(self, values): return self._agg("first", values)
    def last(self, values): return self._agg("last", values)

    def first_last(self, values):
        """Arrow `hash_first_last`: one struct<first, last> column."""
        return self._agg("first_last", values)

    def one(self, values):
        """Arrow `hash_one`: one value per group — here always the group's lowest row, null included."""
        return self._agg("one", values)

    def list(self, values):
        """Arrow `hash_list`: every value of the group, in row order, as a list column."""
        return self._agg("list", values)

    def distinct(self, values):
        """Arrow `hash_distinct`: the distinct non-null values of the group, ascending, as a list column."""
        return self._agg("distinct", values)

    def count_distinct(self, values): return self._agg("count_distinct", values)
    def any(self, values): return self._agg("any", values)
    def all(self, values): return self._agg("all", values)

    def product(self, values):
        """Arrow `hash_product`, on the GPU: a segmented multiply reduction. Integers wrap in 64 bits."""
        return self._agg("product", values)

    def variance(self, values, ddof=0):
        return self._agg("variance" if ddof == 0 else "variance_sample", values)

    def stddev(self, values, ddof=0):
        return self._agg("stddev" if ddof == 0 else "stddev_sample", values)

    def approximate_median(self, values):
        """Arrow `hash_approximate_median`, computed exactly (a GPU sort by (group, value), not a sketch)."""
        return self._agg("approximate_median", values)

    def quantile(self, values, q):
        """Exact per-group quantile with linear interpolation; q is clamped to [0, 1]."""
        return self._agg("quantile", values, q)

    def skew(self, values):
        """Arrow `hash_skew`, biased (population) as Arrow's default is."""
        return self._agg("skew", values)

    def kurtosis(self, values):
        """Arrow `hash_kurtosis`: excess kurtosis, biased."""
        return self._agg("kurtosis", values)

    def tdigest(self, values, q=0.5):
        """Arrow `hash_tdigest`: a t-digest estimate of q per group (GPU sort, host centroid merge)."""
        return self._agg("tdigest", values, q)

    def pivot_wider(self, pivot_keys, values, names):
        """Arrow `hash_pivot_wider` over a utf8 pivot-key column: a struct with one field per name."""
        p = pivot_keys if isinstance(pivot_keys, MetalArray) else MetalArray.from_arrow(pivot_keys)
        v = values if isinstance(values, MetalArray) else MetalArray.from_arrow(values)
        encoded = [n.encode() for n in names]
        arr = (ctypes.c_char_p * len(encoded))(*encoded)
        out = _P()
        _check(_lib.am_group_pivot_wider(self._h, p._h, v._h, arr, len(encoded), ctypes.byref(out)))
        return MetalArray(out)


def group_by(keys):
    """Group by one or more key columns of any supported type: `am.group_by([region, year])`.

    A single array is accepted for a single key column. Returns a `GroupByKeys`."""
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    return GroupByKeys(keys)


def _reduce_ex2(self, op, p1=0.0):
    f, null = ctypes.c_double(), ctypes.c_int()
    _check(_lib.am_reduce_ex2(self._h, _REDUCE_EX2[op], float(p1), ctypes.byref(f), ctypes.byref(null)))
    return None if null.value else f.value


def _skew(self, biased=True):
    """Arrow `skew`: the third standardised central moment, biased (population) by default."""
    return _reduce_ex2(self, "skew" if biased else "skew_sample")


def _kurtosis(self, biased=True):
    """Arrow `kurtosis`: excess kurtosis, biased by default."""
    return _reduce_ex2(self, "kurtosis" if biased else "kurtosis_sample")


def _tdigest(self, q=0.5):
    """Arrow `tdigest`: the t-digest estimate of quantile q (GPU sort, one host centroid merge, delta 100).

    A sketch, so it agrees with pyarrow.compute.tdigest to within the sketch's own error rather than
    exactly; q = 0 and q = 1 are the exact minimum and maximum. `quantile()` is the exact answer."""
    return _reduce_ex2(self, "tdigest", q)


MetalArray.skew = _skew
MetalArray.kurtosis = _kurtosis
MetalArray.tdigest = _tdigest
# ---- remaining selection / sort / random / aggregate functions (see include/arrowmetal.h)
#
# inverse_permutation, scatter, winsorize, rank_quantile, rank_normal, random, true_unless_null,
# count_all, first_last, utf8_swapcase, utf8_zero_fill, make_struct and pivot_wider, plus the
# Arrow-named aliases for calls that already exist under a different name. Appended rather than
# written into the class body so this file stays additive.
_lib.am_inverse_permutation.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_inverse_permutation.restype = ctypes.c_int
_lib.am_scatter.argtypes = [_P, _P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_scatter.restype = ctypes.c_int
_lib.am_winsorize.argtypes = [_P, ctypes.c_double, ctypes.c_double, ctypes.POINTER(_P)]
_lib.am_winsorize.restype = ctypes.c_int
_lib.am_rank.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_rank.restype = ctypes.c_int
_lib.am_random.argtypes = [ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(_P)]
_lib.am_random.restype = ctypes.c_int
_lib.am_true_unless_null.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_true_unless_null.restype = ctypes.c_int
_lib.am_count_all.argtypes = [_P]
_lib.am_count_all.restype = ctypes.c_int64
_lib.am_first_last.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_first_last.restype = ctypes.c_int
_lib.am_str_extra.argtypes = [_P, ctypes.c_int, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64,
                              ctypes.POINTER(_P)]
_lib.am_str_extra.restype = ctypes.c_int
_lib.am_pivot_wider.argtypes = [_P, _P, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64, ctypes.c_int,
                                ctypes.POINTER(_P)]
_lib.am_pivot_wider.restype = ctypes.c_int
_lib.am_make_struct.argtypes = [ctypes.POINTER(_P), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
                                ctypes.POINTER(_P)]
_lib.am_make_struct.restype = ctypes.c_int

# Op numbering is the C ABI contract; see include/arrowmetal.h.
_RANK = {"quantile": 0, "normal": 1, "normal_f32": 2}
_STR_EXTRA = {"utf8_swapcase": 0, "utf8_zero_fill": 1}


def _as_array(obj):
    """Anything that can be a MetalArray: one already, or something with the Arrow C array protocol."""
    return obj if isinstance(obj, MetalArray) else MetalArray.from_arrow(obj)


def _inverse_permutation(self, max_index=-1):
    """Arrow `inverse_permutation`: for the i-th index, the index-th output element is i.

    The output has `max_index + 1` elements, or this column's length when `max_index` is negative.
    A position no index names comes back null; when several positions name the same one the last
    wins, as in Arrow (and deterministically here - the GPU scatter takes an atomic maximum over the
    source positions). Null indices are skipped and an index outside [0, max_index] raises. The
    result is always int32; Arrow's `output_type` option is not implemented.
    """
    return _call(_lib.am_inverse_permutation, self._h, max_index)


def _scatter(self, indices, max_index=-1):
    """Arrow `scatter`: place the i-th value at the position named by the i-th index.

    Same shape rules as `inverse_permutation` - unassigned positions null, duplicates last-wins -
    and it works for every column type, because it is that inverse permutation used as a `take`.
    """
    idx = _as_array(indices)          # keep the handle alive across the call
    return _call(_lib.am_scatter, self._h, idx._h, max_index)


def _winsorize(self, lower_limit, upper_limit):
    """Arrow `winsorize`: clamp to the quantiles at `lower_limit` and `upper_limit`.

    The limits are Arrow's *nearest* quantiles rather than interpolated ones: with m non-null,
    non-NaN values sorted ascending, a limit q picks sorted[round(q * (m - 1))], halfway rounding to
    the even index. Nulls stay null and NaNs pass through unchanged. One GPU sort, one clamp kernel.
    """
    return _call(_lib.am_winsorize, self._h, float(lower_limit), float(upper_limit))


def _rank_quantile(self):
    """Arrow `rank_quantile`: (average 1-based rank of the row's tie group - 0.5) / n, as float64.

    Nulls sort last as one tie group (pyarrow's default null_placement="at_end") and NaN is one
    value after +inf; no result is ever null. Arrow's sort_keys / null_placement options are not
    implemented - this is always the single ascending key with nulls at the end.
    """
    return _call(_lib.am_rank, self._h, _RANK["quantile"])


def _rank_normal(self, float32=False):
    """Arrow `rank_normal`: the normal percent-point function of `rank_quantile()`.

    float64 by default, with the inverse CDF evaluated on the host through Wichura's AS 241 (about
    1e-16 relative), because Metal has neither `double` nor `log`/`exp`/`erfc` for the software
    binary64. `float32=True` runs it entirely on the GPU (Acklam plus one Halley refinement) and
    lands within about 1e-6 of the float64 answer.
    """
    return _call(_lib.am_rank, self._h, _RANK["normal_f32" if float32 else "normal"])


def _true_unless_null(self):
    """Arrow `true_unless_null`: true for every valid row, null for every null one."""
    return _call(_lib.am_true_unless_null, self._h)


def _count_all(self):
    """Arrow `count_all`: the number of rows, valid or not."""
    return int(_lib.am_count_all(self._h))


def _first_last(self, skip_nulls=True):
    """Arrow `first_last`: a one-row struct with fields `first` and `last` of this column's type.

    With `skip_nulls` the first and last non-null values are used (both null when no row is valid);
    with `skip_nulls=False` the first and last rows are taken as they are.
    """
    return _call(_lib.am_first_last, self._h, 1 if skip_nulls else 0)


def _utf8_swapcase(self):
    """Arrow `utf8_swapcase`, on the GPU over Basic Latin, Latin-1 Supplement and Latin Extended-A.

    Code points outside those blocks - and U+00DF, which Arrow swaps to U+1E9E - pass through
    unchanged rather than being mangled, so the output is always valid UTF-8.
    """
    return _call(_lib.am_str_extra, self._h, _STR_EXTRA["utf8_swapcase"], b"", 0, 0)


def _utf8_zero_fill(self, width, padding="0"):
    """Arrow `utf8_zero_fill`: left-pad to `width` code points, inserting the padding after a leading
    + or -. Strings already at or over `width` are unchanged; the content need not be numeric."""
    pad = padding.encode()
    return _call(_lib.am_str_extra, self._h, _STR_EXTRA["utf8_zero_fill"], pad, len(pad), int(width))


def random(n, initializer=0):
    """Arrow `random`: `n` uniform float64 values in [0, 1), generated on the GPU.

    Philox4x32-10 (Salmon, Moraes, Dror & Shaw, SC'11) keyed by `initializer`: element i comes from
    the counter (i, 0, 0, 0), so the stream depends only on the seed - not on the device or the
    launch geometry - and a prefix of a long draw equals a short draw with the same seed. Pass
    "system" for a seed from the operating system's random source.

    The stream is ArrowMetal's own; it does not reproduce the numbers Arrow C++ generates for the
    same seed (Arrow uses pcg32_fast on the host).
    """
    if initializer == "system":
        seed = int.from_bytes(os.urandom(8), "little")
    elif isinstance(initializer, int):
        seed = initializer & 0xFFFFFFFFFFFFFFFF
    else:
        raise ArrowMetalError('random initializer must be an int or "system"')
    return _call(_lib.am_random, int(n), seed)


def make_struct(arrays, names):
    """Arrow `make_struct`: compose equal-length columns into one struct column. Metadata only."""
    cols = [_as_array(a) for a in arrays]
    if not cols:
        raise ArrowMetalError("make_struct needs at least one column")
    if len(names) != len(cols):
        raise ArrowMetalError(f"make_struct got {len(names)} names for {len(cols)} columns")
    handles = (_P * len(cols))(*[c._h for c in cols])
    cnames = (ctypes.c_char_p * len(names))(*[n.encode() for n in names])
    out = _P()
    _check(_lib.am_make_struct(handles, cnames, len(cols), ctypes.byref(out)))
    return MetalArray(out)


def pivot_wider(pivot_keys, pivot_values, key_names, unexpected_key_behavior="ignore"):
    """Arrow `pivot_wider`: a one-row struct with a field per entry of `key_names`.

    A key that never appears, or appears only with a null value, gives a null field; a key carrying
    more than one non-null value raises, as in Arrow. The key column may be utf8, binary, dictionary
    or any integer type. This one runs on the host: the output is one row wide however long the
    input is.
    """
    if unexpected_key_behavior not in ("ignore", "raise"):
        raise ArrowMetalError('unexpected_key_behavior must be "ignore" or "raise"')
    keys, values = _as_array(pivot_keys), _as_array(pivot_values)
    names = list(key_names)
    cnames = (ctypes.c_char_p * max(len(names), 1))(*[n.encode() for n in names])
    out = _P()
    _check(_lib.am_pivot_wider(keys._h, values._h, cnames, len(names),
                               1 if unexpected_key_behavior == "raise" else 0, ctypes.byref(out)))
    return MetalArray(out)


MetalArray.inverse_permutation = _inverse_permutation
MetalArray.scatter = _scatter
MetalArray.winsorize = _winsorize
MetalArray.rank_quantile = _rank_quantile
MetalArray.rank_normal = _rank_normal
MetalArray.true_unless_null = _true_unless_null
MetalArray.count_all = _count_all
MetalArray.first_last = _first_last
MetalArray.utf8_swapcase = _utf8_swapcase
MetalArray.utf8_zero_fill = _utf8_zero_fill

# Arrow-named aliases for calls that already exist under an ArrowMetal name. Each is the same code
# path bound to a second name, not a second implementation.
MetalArray.array_filter = MetalArray.filter                 # Arrow array_filter
MetalArray.array_take = MetalArray.take                     # Arrow array_take
MetalArray.array_sort_indices = MetalArray.argsort          # Arrow array_sort_indices
MetalArray.sort_indices = MetalArray.argsort                # Arrow sort_indices, single key
MetalArray.dictionary_decode = MetalArray.decode            # Arrow dictionary_decode
MetalArray.invert = MetalArray.__invert__                   # Arrow invert
MetalArray.ascii_swapcase = MetalArray.swapcase             # Arrow ascii_swapcase
MetalArray.ascii_lpad = MetalArray.pad_left                 # Arrow ascii_lpad / utf8_lpad
MetalArray.ascii_rpad = MetalArray.pad_right                # Arrow ascii_rpad / utf8_rpad
MetalArray.utf8_lpad = MetalArray.pad_left
MetalArray.utf8_rpad = MetalArray.pad_right


def _top_k_unstable(self, k):
    """Arrow `top_k_unstable`: indices of the k largest values (GPU partial selection for k <= 1024)."""
    return self.top_k(k, largest=True)


def _bottom_k_unstable(self, k):
    """Arrow `bottom_k_unstable`: indices of the k smallest values."""
    return self.top_k(k, largest=False)


def _select_k_unstable(self, k, largest=False):
    """Arrow `select_k_unstable`: indices of the k best values, smallest first by default."""
    return self.top_k(k, largest=largest)


MetalArray.top_k_unstable = _top_k_unstable
MetalArray.bottom_k_unstable = _bottom_k_unstable
MetalArray.select_k_unstable = _select_k_unstable


# ---- associative transforms and the partial sort, wired through to Python
_lib.am_unique.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_unique.restype = ctypes.c_int
_lib.am_value_counts.argtypes = [_P, ctypes.POINTER(_P)]
_lib.am_value_counts.restype = ctypes.c_int
_lib.am_partition_nth_indices.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_partition_nth_indices.restype = ctypes.c_int


def _unique(self):
    """Arrow `unique`: the distinct non-null values.

    ArrowMetal returns them **ascending** (one GPU sort plus a run scan); Arrow returns them in order
    of first appearance. Sort the pyarrow answer, or ours, when the two have to line up.
    """
    return _call(_lib.am_unique, self._h)


def _value_counts(self):
    """Arrow `value_counts`: a struct column with fields `values` and `counts` (int64).

    Same ascending order as `unique()`, where Arrow uses order of first appearance.
    """
    return _call(_lib.am_value_counts, self._h)


def _partition_nth_indices(self, n):
    """Arrow `partition_nth_indices`: indices that put the n smallest values first.

    Answered with the full stable GPU argsort, which satisfies the contract; there is no cheaper
    partial-partition kernel yet (`top_k` is the one that does less work than a full sort).
    """
    return _call(_lib.am_partition_nth_indices, self._h, int(n))


def _count(self, mode="only_valid"):
    """Arrow `count`: the number of valid rows ("only_valid"), null rows ("only_null") or all rows
    ("all"). O(1) metadata — the null count is already carried on the column."""
    if mode == "only_valid":
        return len(self) - self.null_count
    if mode == "only_null":
        return self.null_count
    if mode == "all":
        return len(self)
    raise ArrowMetalError('count mode must be "only_valid", "only_null" or "all"')


MetalArray.unique = _unique
MetalArray.value_counts = _value_counts
MetalArray.partition_nth_indices = _partition_nth_indices
MetalArray.count = _count
# ---- checked (overflow-raising) arithmetic and the remaining element-wise math
# Op numbering is the C ABI contract; the tables are in include/arrowmetal.h.
_lib.am_unary_checked.argtypes = [_P, ctypes.c_int, ctypes.POINTER(_P)]
_lib.am_unary_checked.restype = ctypes.c_int
_lib.am_binary_checked.argtypes = [_P, ctypes.c_int, _P, _P, ctypes.POINTER(_P)]
_lib.am_binary_checked.restype = ctypes.c_int
_lib.am_cumulative_checked.argtypes = [_P, ctypes.c_int, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_cumulative_checked.restype = ctypes.c_int
_lib.am_math_extra.argtypes = [_P, ctypes.c_int, _P, _P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_math_extra.restype = ctypes.c_int

_UNARY_CHECKED = {"negate": 0, "abs": 1, "sqrt": 2, "ln": 3, "log10": 4, "log2": 5, "log1p": 6}
_BINARY_CHECKED = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3, "power": 4,
                   "shift_left": 5, "shift_right": 6, "logb": 7}
_CUMULATIVE_CHECKED = {"cumulative_sum": 0, "cumulative_prod": 1, "pairwise_diff": 2}
_MATH_EXTRA = {"expm1": 0, "log1p": 1, "logb": 2, "hypot": 3,
               "round": 4, "round_to_multiple": 5, "round_binary": 6}
# Arrow's RoundMode, in Arrow's own numbering.
_ROUND_MODES = ("down", "up", "towards_zero", "towards_infinity", "half_down", "half_up",
                "half_towards_zero", "half_towards_infinity", "half_to_even", "half_to_odd")


def _round_mode_code(mode):
    """Arrow round-mode name (or index) to its number."""
    if isinstance(mode, int):
        if not 0 <= mode < len(_ROUND_MODES):
            raise ArrowMetalError(f"unknown round mode {mode}")
        return mode
    try:
        return _ROUND_MODES.index(mode)
    except ValueError:
        raise ArrowMetalError(f"unknown round mode {mode!r}; expected one of {', '.join(_ROUND_MODES)}")


def _unary_checked(self, op):
    """One checked unary op by name: negate, abs, sqrt, ln, log10, log2, log1p.

    The values are those of the unchecked op; an element outside the valid range raises
    ArrowMetalError naming the Arrow message and the first offending row, for example
    "negate_checked: overflow at index 3"."""
    return _call(_lib.am_unary_checked, self._h, _UNARY_CHECKED[op])


def _binary_checked(self, op, other):
    """One checked binary op by name against a MetalArray or a scalar: add, subtract, multiply,
    divide, power, shift_left, shift_right, logb."""
    code = _BINARY_CHECKED[op]
    if isinstance(other, MetalArray):
        return _call(_lib.am_binary_checked, self._h, code, other._h, None)
    return _call(_lib.am_binary_checked, self._h, code, None, self._scalar(other))


def _math_extra(self, op, other=None, scalar=None, p1=0):
    """One expm1 / log1p / logb / hypot / rounding op; the table is in include/arrowmetal.h."""
    b = other._h if isinstance(other, MetalArray) else None
    return _call(_lib.am_math_extra, self._h, _MATH_EXTRA[op], b, scalar, p1)


MetalArray.unary_checked = _unary_checked
MetalArray.binary_checked = _binary_checked
MetalArray.math_extra = _math_extra
MetalArray._math_extra = _math_extra


def _checked_binary_method(name, doc):
    def f(self, other):
        return _binary_checked(self, name, other)
    f.__name__ = name + "_checked"
    f.__doc__ = doc
    return f


def _checked_unary_method(name, doc):
    def f(self):
        return _unary_checked(self, name)
    f.__name__ = name + "_checked"
    f.__doc__ = doc
    return f


# On a float column the four arithmetic ops never raise: Arrow treats an overflow to infinity and a NaN
# as ordinary results, and so does this package. Only divide-by-zero and the log/root domain errors do.
for _name, _doc in [
    ("add", "Arrow add_checked: add() with an ArrowMetalError where an integer element would wrap."),
    ("subtract", "Arrow subtract_checked: subtract() with an ArrowMetalError where an integer element would wrap."),
    ("multiply", "Arrow multiply_checked: multiply() with an ArrowMetalError where an integer element would wrap."),
    ("divide", "Arrow divide_checked: raises 'divide by zero' for a zero divisor on any type, and "
               "'overflow' for INT_MIN / -1."),
    ("power", "Arrow power_checked: raises for a negative integer exponent and for any repeated-squaring "
              "step that would wrap. Float columns never raise."),
    ("shift_left", "Arrow shift_left_checked: raises when the shift amount is negative or at least the "
                   "precision of the type (the bit width, less one on a signed column), so "
                   "shift_left_checked(int64, 63) raises. Bits shifted off the top are not an error."),
    ("shift_right", "Arrow shift_right_checked: same amount check as shift_left_checked."),
    ("logb", "Arrow logb_checked(base): raises when the value or the base is zero or negative."),
]:
    setattr(MetalArray, _name + "_checked", _checked_binary_method(_name, _doc))
del _name, _doc

for _name, _doc in [
    ("negate", "Arrow negate_checked: raises for INT_MIN on a signed column and, unlike pyarrow (which "
               "has no unsigned kernel at all), for every non-zero value on an unsigned one."),
    ("abs", "Arrow abs_checked: raises only for INT_MIN on a signed integer column."),
    ("sqrt", "Arrow sqrt_checked: raises 'square root of negative number'. NaN, -0.0 and +inf do not raise."),
    ("ln", "Arrow ln_checked: raises 'logarithm of zero' or 'logarithm of negative number'."),
    ("log10", "Arrow log10_checked: same domain check as ln_checked."),
    ("log2", "Arrow log2_checked: same domain check as ln_checked."),
    ("log1p", "Arrow log1p_checked: the domain boundary is -1, so -1 raises 'logarithm of zero' and "
              "anything below it 'logarithm of negative number'."),
]:
    setattr(MetalArray, _name + "_checked", _checked_unary_method(_name, _doc))
del _name, _doc


def _cumulative_sum_checked(self):
    """Arrow cumulative_sum_checked: the running sum, raising where a step would wrap.

    Null rows are skipped and the running value carries across them, which is this package's
    cumulative_sum() behaviour; pyarrow's default instead makes every row after a null null."""
    return _call(_lib.am_cumulative_checked, self._h, _CUMULATIVE_CHECKED["cumulative_sum"], 0)


def _cumulative_prod_checked(self):
    """Arrow cumulative_prod_checked: the running product, raising where a step would wrap."""
    return _call(_lib.am_cumulative_checked, self._h, _CUMULATIVE_CHECKED["cumulative_prod"], 0)


def _pairwise_diff_checked(self, period=1):
    """Arrow pairwise_diff_checked: self[i] - self[i - period], raising where that would wrap."""
    return _call(_lib.am_cumulative_checked, self._h, _CUMULATIVE_CHECKED["pairwise_diff"], period)


MetalArray.cumulative_sum_checked = _cumulative_sum_checked
MetalArray.cumulative_prod_checked = _cumulative_prod_checked
MetalArray.pairwise_diff_checked = _pairwise_diff_checked


def _expm1(self):
    """Arrow expm1: exp(x) - 1, accurate for small x. Float columns only (cast an integer one first)."""
    return _math_extra(self, "expm1")


def _log1p(self):
    """Arrow log1p: ln(1 + x), accurate for small x. x == -1 gives -inf and x < -1 gives NaN;
    log1p_checked() raises on both."""
    return _math_extra(self, "log1p")


def _logb(self, base):
    """Arrow logb(x, base) = ln(x) / ln(base), with a scalar base or a column of bases."""
    if isinstance(base, MetalArray):
        return _math_extra(self, "logb", other=base)
    return _math_extra(self, "logb", scalar=self._scalar(base))


def _hypot(self, other):
    """Arrow hypot: sqrt(x^2 + y^2), scaled so that a large or tiny pair neither overflows nor
    underflows on the way. An infinite operand gives inf even opposite a NaN, as IEEE-754 prescribes."""
    if isinstance(other, MetalArray):
        return _math_extra(self, "hypot", other=other)
    return _math_extra(self, "hypot", scalar=self._scalar(other))


def _round_to_multiple(self, multiple, mode="half_to_even"):
    """Arrow round_to_multiple: round_int(x / multiple) * multiple. `multiple` must be positive.

    `mode` is any Arrow RoundMode name: down, up, towards_zero, towards_infinity, half_down, half_up,
    half_towards_zero, half_towards_infinity, half_to_even (the default), half_to_odd."""
    return _math_extra(self, "round_to_multiple", scalar=self._scalar(multiple), p1=_round_mode_code(mode))


def _round_binary(self, ndigits, mode="half_to_even"):
    """Arrow round_binary: round() with one ndigits per row (an int32 column, or anything pyarrow can
    turn into one). The result is null wherever either column is."""
    if not isinstance(ndigits, MetalArray):
        ndigits = MetalArray.from_arrow(pa.array(ndigits, pa.int32()))
    return _math_extra(self, "round_binary", other=ndigits, p1=_round_mode_code(mode))


# round() keeps its no-argument meaning (halves away from zero); passing ndigits or mode selects the
# general Arrow round kernel, whose default mode is Arrow's own half_to_even.
_am_round_halves_away = MetalArray.round


def _am_round(self, ndigits=None, mode=None):
    """Arrow round. With no arguments, halves go away from zero (this method's historical behaviour).
    With `ndigits` and/or `mode` this is Arrow's round(x, ndigits, round_mode), evaluated as
    round_int(x * 10^ndigits) / 10^ndigits and defaulting to ndigits=0, mode="half_to_even"."""
    if ndigits is None and mode is None:
        return _am_round_halves_away(self)
    p1 = _round_mode_code(mode if mode is not None else "half_to_even") | (int(ndigits or 0) << 8)
    return _math_extra(self, "round", p1=p1)


MetalArray.expm1 = _expm1
MetalArray.log1p = _log1p
MetalArray.logb = _logb
MetalArray.hypot = _hypot
MetalArray.round_to_multiple = _round_to_multiple
MetalArray.round_binary = _round_binary
MetalArray.round = _am_round

# ---- decimal columns reach the rest of the numeric surface.
#
# `am_decimal_op` defines negate (9), abs (10), sign (11) and the sum / min / max reductions (18-20),
# but every one of them was unreachable from Python: `unary()` and `_reduce()` route a decimal column
# to the primitive entry points, which reject it. Dispatching on the format here is what makes
# `x.negate()`, `x.abs()`, `x.sign()`, `x.sum()`, `x.min()` and `x.max()` work on a decimal column, as
# the matching pyarrow.compute functions do.
_DECIMAL_UNARY_OPS = {"negate": 9, "abs": 10, "sign": 11}
_DECIMAL_REDUCE_OPS = {"sum": 18, "min": 19, "max": 20}
_primitive_unary = MetalArray.unary
_primitive_reduce = MetalArray._reduce


def _unary_any(self, op):
    """One unary math op by name. A decimal column takes the decimal kernels for negate, abs and sign;
    every other op, and every other type, keeps the primitive path."""
    if self.format.startswith("d:") and op in _DECIMAL_UNARY_OPS:
        return self._decimal_op(_DECIMAL_UNARY_OPS[op])
    return _primitive_unary(self, op)


def _reduce_any(self, op):
    """sum / min / max / mean. A decimal column reduces through `am_decimal_op`, which answers with a
    length-1 decimal array (a 128-bit total does not fit an int64 out-parameter); the value comes back
    as a Python `decimal.Decimal`, which is what `pc.sum(...).as_py()` returns too."""
    name = ("sum", "min", "max", "mean")[op]
    if self.format.startswith("d:"):
        if name not in _DECIMAL_REDUCE_OPS:
            raise ArrowMetalError(f"{name} is not defined for a decimal column")
        result = self._decimal_op(_DECIMAL_REDUCE_OPS[name]).to_arrow()
        return result[0].as_py() if len(result) else None
    return _primitive_reduce(self, op)


MetalArray.unary = _unary_any
MetalArray._reduce = _reduce_any


# --- Dispatch latency (see docs/RESIDENT.md) -------------------------------------------------

_lib.am_resident_mode.argtypes = [ctypes.c_int]; _lib.am_resident_mode.restype = ctypes.c_int
_lib.am_resident_mode_available.argtypes = []; _lib.am_resident_mode_available.restype = ctypes.c_int
_lib.am_resident_mode_reason.argtypes = []; _lib.am_resident_mode_reason.restype = ctypes.c_char_p
_lib.am_low_latency_wait.argtypes = [ctypes.c_int]; _lib.am_low_latency_wait.restype = ctypes.c_int
_lib.am_spin_microseconds.argtypes = [ctypes.c_int64]; _lib.am_spin_microseconds.restype = ctypes.c_int64


def resident_mode(on=True):
    """Ask for a persistent GPU worker: one long-running kernel spinning on a work queue in unified
    memory, so a small op costs a memory round trip instead of a command buffer.

    Returns whether it took effect. On Apple silicon that is always False -- a running Metal kernel
    and the CPU are not cache coherent through shared storage, so a CPU store reaches a spinning
    kernel only when the line is evicted, measured at 0.4 to 1.4 seconds against a 65 microsecond
    command-buffer round trip. `resident_mode_reason()` has the detail."""
    return bool(_lib.am_resident_mode(1 if on else 0))


def resident_mode_available():
    """Whether a persistent GPU worker is available on this device (always False today)."""
    return bool(_lib.am_resident_mode_available())


def resident_mode_reason():
    """Why resident mode is unavailable, with the measured numbers."""
    return _lib.am_resident_mode_reason().decode()


def low_latency_wait(on=True):
    """Wait for command buffers on an MTLSharedEvent the CPU polls out of memory rather than calling
    waitUntilCompleted: about 65 microseconds against 78 per round trip on an M4 Max. On by default.
    Returns the setting now in force."""
    return bool(_lib.am_low_latency_wait(1 if on else 0))


def spin_microseconds(microseconds=None):
    """How long the CPU spins before it blocks on a command buffer. 0 blocks immediately, which costs
    latency but frees the core. Call with no argument to read the current value."""
    return int(_lib.am_spin_microseconds(-1 if microseconds is None else int(microseconds)))


# ---------------------------------------------------------------------------------------------------
# Fused expression queries: one Metal kernel for a whole expression DAG.
#
# Build an expression with am.col(...) and Python operators, finish it with a terminal (project,
# a reduction, or a group-by), then run it over a record batch, a dict of arrays, or a pyarrow Table:
#
#     import arrowmetal as am
#     am.query(tbl, am.filter((am.col("region") == 2) & (am.col("amount") > 100)).sum(am.col("amount")))
#     am.query(tbl, ((am.col("a") * 2 + am.col("b")) / (am.col("c") + 1) - am.col("d")).alias("r").project())
#     am.query(tbl, am.group_by(am.col("k"), 1000).sum(am.col("v")))
#
# The whole tree is lowered to one runtime-generated kernel: the inputs are read once no matter how
# many operators there are. The wire format is the s-expression grammar in include/arrowmetal.h;
# `.sexpr()` shows it and `repr()` shows a Polars-like rendering. See docs/EXPR.md.
_lib.am_query.argtypes = [ctypes.POINTER(_P), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int64,
                          ctypes.c_char_p, ctypes.POINTER(_P)]
_lib.am_query.restype = ctypes.c_int
_lib.am_query_column_count.argtypes = [_P]
_lib.am_query_column_count.restype = ctypes.c_int64
_lib.am_query_column_name.argtypes = [_P, ctypes.c_int64]
_lib.am_query_column_name.restype = ctypes.c_char_p
_lib.am_query_column.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(_P)]
_lib.am_query_column.restype = ctypes.c_int
_lib.am_query_scalar_count.argtypes = [_P]
_lib.am_query_scalar_count.restype = ctypes.c_int64
_lib.am_query_scalar_name.argtypes = [_P, ctypes.c_int64]
_lib.am_query_scalar_name.restype = ctypes.c_char_p
_lib.am_query_scalar.argtypes = [_P, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                                 ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int)]
_lib.am_query_scalar.restype = ctypes.c_int
_lib.am_query_result_release.argtypes = [_P]
_lib.am_query_canonical.argtypes = [ctypes.c_char_p]
_lib.am_query_canonical.restype = ctypes.c_char_p

# Arrow type name -> the grammar's type token.
_EXPR_TYPES = {"int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
               "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
               "float": "f32", "float32": "f32", "double": "f64", "float64": "f64",
               "bool": "bool", "boolean": "bool", "string": "str", "utf8": "str"}


def _expr_type_token(t):
    if isinstance(t, str):
        tok = _EXPR_TYPES.get(t)
    else:
        tok = _EXPR_TYPES.get(str(t))
    if tok is None:
        raise ArrowMetalError(f"unsupported expression type {t!r}; expected one of {sorted(set(_EXPR_TYPES))}")
    return tok


def _sq(s):
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


class Expr:
    """One node of an ArrowMetal expression tree.

    Immutable. Operators (+ - * / == != < <= > >= & | ~) and the methods below build new nodes;
    nothing runs until the expression reaches am.query(). `repr()` renders it the way Polars does,
    `.sexpr()` gives the wire form."""

    __slots__ = ("_s", "_r", "_name")

    def __init__(self, sexpr, text, name=None):
        self._s, self._r, self._name = sexpr, text, name

    def __repr__(self):
        return self._r

    def sexpr(self):
        """The serialised s-expression, the exact text the C ABI takes."""
        return self._s

    def alias(self, name):
        """Names this expression for `project`."""
        return Expr(self._s, self._r, name)

    # ---- operators
    def _bin(self, op, sym, other, reverse=False):
        a, b = (_as_expr(other), self) if reverse else (self, _as_expr(other))
        return Expr(f"({op} {a._s} {b._s})", f"({a._r} {sym} {b._r})")

    def __add__(self, o): return self._bin("add", "+", o)
    def __radd__(self, o): return self._bin("add", "+", o, True)
    def __sub__(self, o): return self._bin("sub", "-", o)
    def __rsub__(self, o): return self._bin("sub", "-", o, True)
    def __mul__(self, o): return self._bin("mul", "*", o)
    def __rmul__(self, o): return self._bin("mul", "*", o, True)
    def __truediv__(self, o): return self._bin("div", "/", o)
    def __rtruediv__(self, o): return self._bin("div", "/", o, True)
    def __eq__(self, o): return self._bin("eq", "==", o)
    def __ne__(self, o): return self._bin("ne", "!=", o)
    def __lt__(self, o): return self._bin("lt", "<", o)
    def __le__(self, o): return self._bin("le", "<=", o)
    def __gt__(self, o): return self._bin("gt", ">", o)
    def __ge__(self, o): return self._bin("ge", ">=", o)
    def __and__(self, o): return self._bin("and", "&", o)
    def __or__(self, o): return self._bin("or", "|", o)
    def __xor__(self, o): return self._bin("bit_xor", "^", o)
    def __invert__(self): return Expr(f"(not {self._s})", f"(~{self._r})")
    def __neg__(self): return Expr(f"(negate {self._s})", f"(-{self._r})")
    def __hash__(self): return hash(self._s)

    def and_kleene(self, o):
        """Three-valued AND: false wins over null (Arrow's and_kleene)."""
        return self._bin("and_kleene", "&k", o)

    def or_kleene(self, o):
        """Three-valued OR: true wins over null (Arrow's or_kleene)."""
        return self._bin("or_kleene", "|k", o)

    def bitwise_and(self, o): return self._bin("bit_and", "&&", o)
    def bitwise_or(self, o): return self._bin("bit_or", "||", o)
    def shift_left(self, o): return self._bin("shl", "<<", o)
    def shift_right(self, o): return self._bin("shr", ">>", o)

    def _un(self, op, text=None):
        return Expr(f"({op} {self._s})", f"{text or op}({self._r})")

    def abs(self): return self._un("abs")
    def sqrt(self): return self._un("sqrt")
    def exp(self): return self._un("exp")
    def ln(self): return self._un("ln")
    def round(self):
        """Halves away from zero (Arrow's round with mode='half_towards_infinity')."""
        return self._un("round")
    def bitwise_not(self): return self._un("bit_not")

    def cast(self, t):
        tok = _expr_type_token(t)
        return Expr(f"(cast {self._s} {tok})", f"{self._r}.cast({tok})")

    def is_null(self): return Expr(f"(is_null {self._s})", f"{self._r}.is_null()")
    def is_valid(self): return Expr(f"(is_valid {self._s})", f"{self._r}.is_valid()")

    def fill_null(self, other):
        o = _as_expr(other)
        return Expr(f"(fill_null {self._s} {o._s})", f"{self._r}.fill_null({o._r})")

    def is_in(self, values):
        items = [_as_expr(v) for v in values]
        return Expr("(is_in " + self._s + " " + " ".join(i._s for i in items) + ")",
                    f"{self._r}.is_in({[v for v in values]!r})")

    def starts_with(self, pattern):
        return Expr(f"(starts_with {self._s} {_sq(pattern)})", f"{self._r}.starts_with({pattern!r})")

    def contains(self, pattern):
        return Expr(f"(contains {self._s} {_sq(pattern)})", f"{self._r}.contains({pattern!r})")

    def str_equals(self, pattern):
        return Expr(f"(str_eq {self._s} {_sq(pattern)})", f"{self._r}.str_equals({pattern!r})")

    # ---- terminals
    def sum(self, name=None): return Query().sum(self, name)
    def min(self, name=None): return Query().min(self, name)
    def max(self, name=None): return Query().max(self, name)
    def mean(self, name=None): return Query().mean(self, name)
    def count(self, name=None): return Query().count(self, name)

    def project(self):
        """A one-column project of this expression."""
        return Query().project([self])


def _as_expr(v):
    if isinstance(v, Expr):
        return v
    if isinstance(v, bool):
        return Expr(f"(bool {'true' if v else 'false'})", repr(v))
    if isinstance(v, int):
        return Expr(f"(int {v})", repr(v))
    if isinstance(v, float):
        return Expr(f"(float {v!r})", repr(v))
    if isinstance(v, str):
        return Expr(f"(str {_sq(v)})", repr(v))
    if v is None:
        raise ArrowMetalError("a bare None has no type; use am.null('int64') for a typed null literal")
    raise ArrowMetalError(f"cannot use {type(v).__name__} as an expression literal")


def col(name):
    """A column reference: `am.col("amount") > 100`."""
    return Expr(f"(col {_sq(name)})", f'col("{name}")', name)


def lit(v, type=None):
    """A literal. Without `type` an integer or float adapts to whatever it is compared with;
    with `type` it is pinned (`am.lit(2, "int32")`)."""
    if type is None:
        return _as_expr(v)
    tok = _expr_type_token(type)
    if tok in ("f32", "f64"):
        return Expr(f"({tok} {float(v)!r})", repr(v))
    if tok == "bool":
        return Expr(f"(bool {'true' if v else 'false'})", repr(v))
    if tok == "str":
        return Expr(f"(str {_sq(v)})", repr(v))
    return Expr(f"({tok} {int(v)})", repr(v))


def null(type):
    """A typed null literal, for fill_null / if_else branches."""
    return Expr(f"(null {_expr_type_token(type)})", f"null({type})")


def if_else(cond, a, b):
    """Arrow if_else: null where `cond` is null, otherwise the chosen branch (and its validity)."""
    c, x, y = _as_expr(cond), _as_expr(a), _as_expr(b)
    return Expr(f"(if_else {c._s} {x._s} {y._s})", f"if_else({c._r}, {x._r}, {y._r})")


def coalesce_expr(*exprs):
    """Arrow coalesce: the first non-null of its arguments."""
    xs = [_as_expr(e) for e in exprs]
    return Expr("(coalesce " + " ".join(x._s for x in xs) + ")",
                "coalesce(" + ", ".join(x._r for x in xs) + ")")


class Query:
    """A whole query: an optional filter, an optional group-by key, and one terminal.

    Chain it: `am.filter(pred).project([...])`, `am.filter(pred).sum(expr)`,
    `am.group_by(key, 1000).sum(value)`. `.sexpr()` is the wire form."""

    def __init__(self):
        self._filter = None
        self._key = None
        self._key_count = 0
        self._key_name = "key"
        self._project = None
        self._aggs = []

    def _copy(self):
        q = Query()
        q._filter, q._key, q._key_count = self._filter, self._key, self._key_count
        q._key_name, q._project, q._aggs = self._key_name, self._project, list(self._aggs)
        return q

    def filter(self, pred):
        """Keep the rows where `pred` is true and not null (Arrow's 'drop' null behaviour)."""
        q = self._copy()
        p = _as_expr(pred)
        q._filter = p if q._filter is None else (q._filter & p)
        return q

    def group_by(self, key, key_count, name="key"):
        """Group by a dense integer key expression in [0, key_count). Rows whose key is null or out
        of range are skipped. The result carries one row per key, in key order."""
        q = self._copy()
        q._key, q._key_count, q._key_name = _as_expr(key), int(key_count), name
        return q

    def project(self, exprs):
        """Materialise one output column per expression. `exprs` may be a list of Expr (named by
        `.alias()`, or by the column they read), a list of (name, Expr) pairs, or a dict."""
        q = self._copy()
        out = []
        if isinstance(exprs, dict):
            items = list(exprs.items())
        else:
            items = list(exprs)
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                name, e = item[0], _as_expr(item[1])
            else:
                e = _as_expr(item)
                name = e._name or f"col{i}"
            out.append((name, e))
        q._project = out
        return q

    def _agg(self, op, e, name):
        q = self._copy()
        q._aggs = list(q._aggs) + [(op, name or (op if e is None or e._name is None else e._name), e)]
        return q

    def sum(self, e, name=None): return self._agg("sum", _as_expr(e), name)
    def min(self, e, name=None): return self._agg("min", _as_expr(e), name)
    def max(self, e, name=None): return self._agg("max", _as_expr(e), name)
    def mean(self, e, name=None): return self._agg("mean", _as_expr(e), name)

    def count(self, e=None, name=None):
        """count() counts rows that pass the filter; count(expr) counts non-null values of expr."""
        return self._agg("count", None if e is None else _as_expr(e), name or "count")

    def aggregate(self, aggs):
        """Several aggregates in one kernel: [("sum", "total", expr), ("count", "n", None), ...]."""
        q = self._copy()
        q._aggs = list(q._aggs) + [(op, name, None if e is None else _as_expr(e)) for op, name, e in aggs]
        return q

    def sexpr(self):
        parts = []
        if self._filter is not None:
            parts.append(f"(filter {self._filter._s})")
        if self._key is not None:
            parts.append(f"(group_by {self._key_count} {_sq(self._key_name)} {self._key._s})")
        if self._project is not None:
            parts.append("(project " + " ".join(f"(as {_sq(n)} {e._s})" for n, e in self._project) + ")")
        elif self._aggs:
            body = []
            for op, name, e in self._aggs:
                body.append(f"({op} {_sq(name)}" + (f" {e._s})" if e is not None else ")"))
            parts.append("(aggregate " + " ".join(body) + ")")
        else:
            raise ArrowMetalError("a query needs a terminal: .project([...]) or .sum(...)/.count()/...")
        return "(query " + " ".join(parts) + ")"

    def __repr__(self):
        bits = []
        if self._filter is not None:
            bits.append(f"filter({self._filter._r})")
        if self._key is not None:
            bits.append(f"group_by({self._key._r}, {self._key_count})")
        if self._project is not None:
            bits.append("project([" + ", ".join(f"{n}={e._r}" for n, e in self._project) + "])")
        else:
            bits.append(", ".join(f"{op}({'' if e is None else e._r}).alias({n!r})"
                                  for op, name, e in self._aggs for n in [name]))
        return "Query." + ".".join(b for b in bits if b)


def filter(pred):
    """Start a query with a row filter: `am.filter(am.col("x") > 3).sum(am.col("y"))`."""
    return Query().filter(pred)


_group_by_keys = group_by      # the hash group-by over arbitrary key columns, defined above


def group_by(keys, key_count=None, name="key"):
    """Two forms.

    `am.group_by([region, year])` groups arbitrary key *columns* and returns a `GroupByKeys`
    (unchanged from before).

    `am.group_by(am.col("k"), 1000)` starts a fused expression query grouped by a dense integer key
    expression in [0, 1000): `am.group_by(am.col("k"), 1000).sum(am.col("v"))`."""
    if isinstance(keys, Expr):
        if key_count is None:
            raise ArrowMetalError("group_by(expr, key_count) needs the size of the dense key space")
        return Query().group_by(keys, key_count, name)
    return _group_by_keys(keys)


def project(exprs):
    """Start a projection-only query."""
    return Query().project(exprs)


def _as_pa_array(c):
    """A ChunkedArray flattened to one Array (pyarrow versions differ on what combine_chunks returns)."""
    if isinstance(c, pa.ChunkedArray):
        if c.num_chunks == 1:
            return c.chunk(0)
        if c.num_chunks == 0:
            return pa.array([], c.type)
        return pa.concat_arrays([ch for ch in c.chunks])
    return c


def _query_columns(data):
    """(names, arrays) from a dict, a pyarrow RecordBatch/Table, or a Polars DataFrame."""
    if isinstance(data, dict):
        return list(data.keys()), list(data.values())
    if isinstance(data, pa.Table):
        return list(data.column_names), [_as_pa_array(c) for c in data.columns]
    if isinstance(data, pa.RecordBatch):
        return list(data.schema.names), [data.column(i) for i in range(data.num_columns)]
    if hasattr(data, "to_arrow") and hasattr(data, "columns"):      # polars.DataFrame
        t = data.to_arrow()
        return list(t.column_names), [_as_pa_array(c) for c in t.columns]
    raise ArrowMetalError("query() needs a dict of arrays, a pyarrow RecordBatch/Table, or a Polars DataFrame")


def query(data, q):
    """Runs a fused query over `data` and returns pyarrow arrays (project / group_by) or Python
    scalars (aggregate; a single aggregate comes back bare, several as a dict by name).

    `data` is a dict of arrays (pyarrow, numpy, MetalArray, lists), a pyarrow RecordBatch or Table,
    or a Polars DataFrame. Columns are imported into Metal memory zero-copy where the producer's
    buffers allow it, so passing MetalArrays you already hold avoids any import at all."""
    if isinstance(q, Expr):
        q = Query().project([q])
    names, arrays = _query_columns(data)
    handles = [a if isinstance(a, MetalArray) else MetalArray.from_arrow(a) for a in arrays]
    n = len(handles)
    harr = (_P * max(n, 1))(*[h._h for h in handles])
    narr = (ctypes.c_char_p * max(n, 1))(*[nm.encode() for nm in names])
    out = _P()
    _check(_lib.am_query(harr, narr, n, q.sexpr().encode(), ctypes.byref(out)))
    try:
        ncol = _lib.am_query_column_count(out)
        if ncol > 0:
            res = {}
            for i in range(ncol):
                nm = _lib.am_query_column_name(out, i).decode()
                h = _P()
                _check(_lib.am_query_column(out, i, ctypes.byref(h)))
                res[nm] = MetalArray(h).to_arrow()
            return res
        vals = {}
        for i in range(_lib.am_query_scalar_count(out)):
            nm = _lib.am_query_scalar_name(out, i).decode()
            iv, fv = ctypes.c_int64(), ctypes.c_double()
            kind, isnull = ctypes.c_int(), ctypes.c_int()
            _check(_lib.am_query_scalar(out, i, ctypes.byref(iv), ctypes.byref(fv),
                                        ctypes.byref(kind), ctypes.byref(isnull)))
            if isnull.value:
                vals[nm] = None
            elif kind.value == 2:
                vals[nm] = fv.value
            elif kind.value == 1:
                vals[nm] = iv.value & 0xFFFFFFFFFFFFFFFF
            else:
                vals[nm] = iv.value
        return next(iter(vals.values())) if len(vals) == 1 else vals
    finally:
        _lib.am_query_result_release(out)


def query_canonical(q):
    """Parses and type-checks the query text without running it, returning its canonical form.
    Raises ArrowMetalError naming the offending node when the text is not a valid query."""
    text = q.sexpr() if isinstance(q, (Query, Expr)) else q
    r = _lib.am_query_canonical(text.encode())
    if r is None:
        _check(1)
    return r.decode()


class _ExprNamespace:
    """`am.expr.col(...)`, for callers who prefer a namespace to bare module functions."""
    col = staticmethod(col)
    lit = staticmethod(lit)
    null = staticmethod(null)
    if_else = staticmethod(if_else)
    coalesce = staticmethod(coalesce_expr)
    filter = staticmethod(filter)
    group_by = staticmethod(group_by)
    project = staticmethod(project)
    query = staticmethod(query)
    Expr = Expr
    Query = Query


expr = _ExprNamespace()


# ---------------------------------------------------------------------------------------------------
# The lazy query engine (python/arrowmetal/lazy.py).
#
# `am.scan(...)` starts a Polars-shaped lazy query. Nothing runs until `.collect()`: the plan goes to
# the Swift engine as one JSON document, is type-checked and optimized (predicate pushdown, projection
# pruning, filter fusion, constant folding, CSE, join reordering), lowered to physical operators with
# maximal fused Metal kernels, and executed inside one command buffer. See docs/ENGINE.md.
#
#     q = (am.scan(table)
#            .filter(am.col("amount") > 100)
#            .group_by("region").agg(am.agg.sum("amount", "total"))
#            .sort("total", descending=True)
#            .limit(10))
#     q.explain()          # the optimized plan, the way Polars prints one
#     q.collect()          # -> pyarrow.Table
# ---------------------------------------------------------------------------------------------------

from . import lazy as _lazy_module

_lazy_module._bind({
    "lib": _lib,
    "P": _P,
    "check": _check,
    "MetalArray": MetalArray,
    "Expr": Expr,
    "as_expr": _as_expr,
    "col": col,
    "pa": pa,
    "ArrowMetalError": ArrowMetalError,
    "query_columns": _query_columns,
})

LazyFrame = _lazy_module.LazyFrame
Agg = _lazy_module.Agg
agg = _lazy_module.agg
scan = _lazy_module.scan
lazy = _lazy_module


def concat(frames):
    """Vertical concatenation of lazy frames with identical schemas (SQL `UNION ALL`)."""
    frames = list(frames)
    if not frames:
        raise ArrowMetalError("concat needs at least one frame")
    return frames[0].concat(frames[1:])
