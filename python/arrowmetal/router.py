"""`python -m arrowmetal.router`: the CPU/GPU router's crossover table on this Mac.

    python -m arrowmetal.router calibrate [--quick] [--out PATH] [--csv PATH]
        Runs the router check sweep here (every routed operation timed with the router pinned to the
        GPU and to the CPU loop, on int64 columns with 10% nulls) and writes the fitted table as JSON to
        ~/.arrowmetal/router/<chip>.json, which new processes on this machine then load at startup.
        --quick runs the shorter grid; the output says which grid ran.

    python -m arrowmetal.router explain <op> <rows> [--dtype T] [--nulls FRAC] [--keys K] [--json]
        Prints the decision the router makes for that call, the table row and crossover it came from,
        and whether the table is the shipped one or this machine's.

docs/CROSSOVER.md describes the pipeline.
"""
import sys
import types

from ._router_calibrate import calibrate, explain_text, main  # noqa: F401

# `arrowmetal.router` is also the per-thread override (`with am.router("cpu"):`). Importing this module
# makes Python rebind that attribute to the module, so the module is callable with the same meaning.
if __name__ != "__main__":
    _scope = getattr(sys.modules[__package__], "_RouterScope", None)
    if _scope is not None:
        class _CallableRouterModule(types.ModuleType):
            def __call__(self, mode):
                return _scope(mode)
        sys.modules[__name__].__class__ = _CallableRouterModule

if __name__ == "__main__":
    sys.exit(main())
