"""The `pl.col(...).arrowmetal` expression namespace: ArrowMetal kernels inside a Polars lazy plan.

Tier 1 (`polars_bridge`) runs the GPU *around* Polars -- you collect a frame and then hand it over.
This module runs it *inside* Polars: every method below is a Polars expression plugin, so it
composes with `select`, `with_columns`, `filter`, `group_by`, `over`, and takes part in the
optimiser's projection and predicate pushdown.

    import polars as pl, arrowmetal.polars_plugin  # registers the namespace

    lf.select(pl.col("amount").arrowmetal.sum())
    lf.with_columns(pl.col("name").arrowmetal.upper())
    lf.select(pl.col("amount").arrowmetal.filter_sum(pl.col("region") == 2))

Building the plugin
-------------------
The Rust crate lives in `polars-plugin/`. From the repository root:

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\
        swift build -c release --product ArrowMetalC      # libArrowMetalC.dylib first
    cd polars-plugin && cargo build --release             # -> target/release/libarrowmetal_polars.dylib

`cargo build` is enough -- the plugin is a plain `cdylib` that Polars `dlopen`s, not a Python
extension module, so `maturin` is not needed and the maturin layout below is untested (the crate
has no pyproject.toml). `plugin_path()` finds `polars-plugin/target/release/` on its own, and
`ARROWMETAL_POLARS_PLUGIN` overrides the search.

The crate pins `polars` 0.55.1 / `pyo3-polars` 0.28, the Rust crates py-polars 1.44.x is built
from. Polars checks the plugin ABI when it loads the library and refuses a mismatched pair with
"this Polars engine doesn't support plugin version", so a different Polars needs a matching
re-pin; see docs/POLARS.md.
"""
import os
import sys

import polars as pl
from polars.plugins import register_plugin_function

from . import ArrowMetalError

__all__ = ["plugin_path", "available", "ArrowMetalExpr"]

_LIB = "libarrowmetal_polars.dylib"


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("ARROWMETAL_POLARS_PLUGIN")
    if env:
        yield env
    # A SwiftPM/cargo checkout: python/arrowmetal -> <repo>/polars-plugin/target/release
    yield os.path.join(here, "..", "..", "polars-plugin", "target", "release", _LIB)
    yield os.path.join(here, "..", "..", "polars-plugin", "target", "debug", _LIB)
    # A wheel that shipped the plugin next to the package.
    yield os.path.join(here, _LIB)
    # `maturin develop` installs it as its own top-level package in the virtualenv.
    for p in sys.path:
        if p:
            yield os.path.join(p, "arrowmetal_polars", _LIB)


def plugin_path():
    """The path to `libarrowmetal_polars.dylib`, or None when it has not been built.

    Set `ARROWMETAL_POLARS_PLUGIN` to override the search.
    """
    for c in _candidates():
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def available():
    """True when the Rust plugin is built and loadable."""
    return plugin_path() is not None


def _path_or_raise():
    p = plugin_path()
    if p is None:
        raise ArrowMetalError(
            f"{_LIB} not found. Build it with `cd polars-plugin && cargo build --release` "
            "(after `swift build -c release --product ArrowMetalC`), or set "
            "ARROWMETAL_POLARS_PLUGIN to the built library."
        )
    return p


def _call(name, args, *, kwargs=None, **flags):
    return register_plugin_function(
        plugin_path=_path_or_raise(),
        function_name=name,
        args=args,
        kwargs=kwargs,
        use_abs_path=True,
        **flags,
    )


@pl.api.register_expr_namespace("arrowmetal")
class ArrowMetalExpr:
    """`pl.col("x").arrowmetal.<op>()` -- one ArrowMetal kernel, evaluated inside the lazy plan.

    Each call moves the Series into Metal memory over the Arrow C Data Interface (zero-copy when
    the buffers are page aligned, which they are at any real size), runs one GPU kernel, and hands
    the result back as a Series. Nothing is re-implemented in Rust: the kernels are the ones in
    the Swift package, reached through `polars-plugin/arrowmetal-sys`.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # -- reductions ------------------------------------------------------------------------
    def sum(self) -> pl.Expr:
        """Arrow `sum` on the GPU. Integers widen to Int64 (UInt64 when unsigned), floats to
        Float64 -- Arrow's rule, and Polars' own for the narrow integer types."""
        return _call("arrowmetal_sum", self._expr, returns_scalar=True)

    def min(self) -> pl.Expr:
        """Keeps the column's dtype. NaN is skipped, and an all-NaN column is null (pyarrow
        returns NaN there)."""
        return _call("arrowmetal_min", self._expr, returns_scalar=True)

    def max(self) -> pl.Expr:
        return _call("arrowmetal_max", self._expr, returns_scalar=True)

    def mean(self) -> pl.Expr:
        return _call("arrowmetal_mean", self._expr, returns_scalar=True)

    def filter_sum(self, predicate: pl.Expr) -> pl.Expr:
        """`sum` over the rows where `predicate` is true, as one GPU compaction plus one
        reduction -- the filtered column is never materialised back into Polars.

            pl.col("amount").arrowmetal.filter_sum(pl.col("region") == 2)
        """
        return _call("arrowmetal_filter_sum", [self._expr, predicate], returns_scalar=True)

    # -- selection -------------------------------------------------------------------------
    def top_k(self, k: int, *, largest: bool = True) -> pl.Expr:
        """The `k` largest values. Ties and nulls follow ArrowMetal's sort order (stable, nulls
        last, NaN after +inf), which need not match Polars' `top_k` tie order."""
        return _call("arrowmetal_top_k", self._expr,
                     kwargs={"k": int(k), "largest": bool(largest)}, changes_length=True)

    def bottom_k(self, k: int) -> pl.Expr:
        return self.top_k(k, largest=False)

    # -- element-wise ----------------------------------------------------------------------
    def hash64(self) -> pl.Expr:
        """A 64-bit hash per value (UInt64). Arrow-equal values hash equal; a null hashes to 0
        and stays null."""
        return _call("arrowmetal_hash64", self._expr, is_elementwise=True)

    def contains(self, pattern: str) -> pl.Expr:
        """Literal substring search on the GPU (not a regex -- use `pl.col(...).str.contains` for
        that)."""
        return _call("arrowmetal_contains", self._expr,
                     kwargs={"pattern": str(pattern)}, is_elementwise=True)

    def starts_with(self, pattern: str) -> pl.Expr:
        return _call("arrowmetal_starts_with", self._expr,
                     kwargs={"pattern": str(pattern)}, is_elementwise=True)

    def ends_with(self, pattern: str) -> pl.Expr:
        return _call("arrowmetal_ends_with", self._expr,
                     kwargs={"pattern": str(pattern)}, is_elementwise=True)

    def upper(self) -> pl.Expr:
        """Simple 1:1 case mapping over Basic Latin, Latin-1 Supplement and Latin Extended-A.
        The multi-character expansions (U+00DF -> SS) are not applied; Polars' own
        `str.to_uppercase()` does apply them."""
        return _call("arrowmetal_upper", self._expr, is_elementwise=True)

    def lower(self) -> pl.Expr:
        return _call("arrowmetal_lower", self._expr, is_elementwise=True)

    def _arith(self, op, value):
        """Sends the scalar across without a float round trip.

        `float(value)` loses the low bits of any integer past 2^53, so `add(2**60 + 1)` used to
        add `2**60` -- neither what Polars' own `pl.col("x") + (2**60 + 1)` does nor what the
        tier-1 bridge does. An integer therefore travels as its exact decimal digits and only a
        float travels as `value`; the Rust side narrows either one to the column's element type
        and refuses what will not fit, the same call `struct.pack` refuses in tier 1.
        """
        kwargs = {"op": op, "int_value": None, "value": None}
        try:
            exact = value.__index__()       # int, bool, numpy integer, ...
        except (AttributeError, TypeError):
            exact = None
        if exact is None:
            kwargs["value"] = float(value)
        else:
            kwargs["int_value"] = str(exact)
            try:
                kwargs["value"] = float(exact)
            except OverflowError:
                pass                        # past f64 entirely; no column type can take it
        return _call("arrowmetal_arith_scalar", self._expr, kwargs=kwargs, is_elementwise=True)

    def add(self, value) -> pl.Expr:
        """Arithmetic against a scalar, keeping the column's own type: integers wrap and integer
        division by zero yields 0, which is Arrow's unchecked behaviour (Polars raises).

        Wrapping is about the *arithmetic*, not the operand. A scalar the column's type cannot
        hold -- `add(1000)` on an Int8 column, `add(-1)` on a UInt8 one, `add(1.5)` on any integer
        one -- raises, which is what the tier-1 bridge does; it is never silently clamped to 127,
        to 0, or truncated to 1.
        """
        return self._arith("add", value)

    def sub(self, value) -> pl.Expr:
        return self._arith("sub", value)

    def mul(self, value) -> pl.Expr:
        return self._arith("mul", value)

    def truediv(self, value) -> pl.Expr:
        return self._arith("div", value)

    div = truediv

    # -- aggregation -----------------------------------------------------------------------
    def group_by_sum(self, values: pl.Expr) -> pl.Expr:
        """Group by this column and sum `values`: one GPU group-by plus one segmented sum,
        returned as a struct column of `n_groups` rows.

            df.select(pl.col("k").arrowmetal.group_by_sum(pl.col("v"))).unnest("...")

        A plugin expression answers with a single Series, so a grouped result has to be a struct;
        `unnest` it to get two columns. Polars' plugin API has no hook for contributing a hash
        aggregate to the group-by engine itself, so this runs as a projection over the whole
        frame rather than inside `df.group_by(...).agg(...)`; `df.arrowmetal.group_by(...)`
        (tier 1) is the ergonomic spelling.

        Group order is ArrowMetal's -- ascending by key for numeric, boolean, temporal and
        decimal keys, first-seen for utf8 -- so sort both sides before comparing.
        """
        return _call("arrowmetal_group_by_sum", [self._expr, values], changes_length=True)

    # -- introspection ---------------------------------------------------------------------
    def device(self) -> pl.Expr:
        """A one-row String column naming the library version and the Metal device, so a plan can
        prove it really reached the GPU."""
        return _call("arrowmetal_device", self._expr, returns_scalar=True)
