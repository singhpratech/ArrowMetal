"""ArrowMetal for Python: Apache Arrow arrays on the Apple silicon GPU.

Thin ctypes wrapper over libArrowMetalC. Accepts anything that speaks the Arrow PyCapsule protocol
(pyarrow arrays, Polars series via .to_arrow(), pandas arrow-backed columns) and returns pyarrow arrays,
so results drop straight back into Polars, pandas, DuckDB or pyarrow.compute.

    import pyarrow as pa, arrowmetal as am
    col = am.MetalArray.from_arrow(pa.array([1, None, 3]))
    kept = col.filter_where(">", 1)           # runs on the GPU
    kept.to_arrow()                           # -> pyarrow.Array([3])
"""
import ctypes, ctypes.util, os, struct, sys
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
                    ("am_slice", [ctypes.c_int64, ctypes.c_int64])]:
    getattr(_lib, name).argtypes = [_P] + extra + [ctypes.POINTER(_P)]
    getattr(_lib, name).restype = ctypes.c_int
_lib.am_group_by.argtypes = [_P, ctypes.c_int64, ctypes.c_int, _P, ctypes.POINTER(_P)]
_lib.am_group_by.restype = ctypes.c_int
_lib.am_batch_begin.restype = ctypes.c_int
_lib.am_batch_end.restype = ctypes.c_int


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
        return pa.type_for_alias({"c": "int8", "C": "uint8", "s": "int16", "S": "uint16", "i": "int32", "I": "uint32",
                                  "l": "int64", "L": "uint64", "f": "float32", "g": "float64", "b": "bool"}[self.format])

    def __repr__(self):
        return f"MetalArray({self.type}, len={len(self)}, nulls={self.null_count}, device={device_name()!r})"

    def _scalar(self, v):
        code = _STRUCT.get(self.format)
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

    def __eq__(self, o): return self.compare("==", o)
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

    # ---- group-by over dense keys in [0, key_count)
    def group_by(self, key_count):
        return GroupBy(self, key_count)


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
