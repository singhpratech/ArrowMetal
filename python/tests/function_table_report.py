#!/usr/bin/env python3
"""Print the Arrow function-name registry as Markdown.

    PYTHONPATH=python python python/tests/function_table_report.py            # summary + full table
    PYTHONPATH=python python python/tests/function_table_report.py --summary  # counts only
    PYTHONPATH=python python python/tests/function_table_report.py Sorts Selections
    PYTHONPATH=python python python/tests/function_table_report.py --page > docs/ARROW_FUNCTIONS.md

With section names it prints only those sections, in the order given. `--page` writes the whole of
`docs/ARROW_FUNCTIONS.md`, intro included, so that file is never edited by hand.

The numbers come from `arrowmetal.functions`, the same table `python/tests/test_functions.py` calls
row by row, so a status here has been executed against pyarrow.compute rather than asserted.
"""
import sys

from arrowmetal import functions as F

_INTRO = """# Apache Arrow compute functions, one row per name

<!-- Generated. Do not edit by hand: regenerate with
     PYTHONPATH=python python python/tests/function_table_report.py --page > docs/ARROW_FUNCTIONS.md -->

Every Apache Arrow v25 compute function name — the {total} of them, being the {docs} in the
[C++ compute function list](https://arrow.apache.org/docs/cpp/compute.html) plus the {hashes}
`hash_*` grouped aggregates `pyarrow` registers — with what ArrowMetal 0.1.0 does about it.

This is the by-name page. [COVERAGE.md](COVERAGE.md) is the by-family page: it groups these functions
and explains how each family works, with the Arrow **type** matrix and the interop status alongside.
Come here to answer "is `<name>` covered, and how?"; go there for "how does this family work?".

## How the table is produced

`python/arrowmetal/functions.py` holds a registry with exactly one entry per Arrow name. Each entry
carries the status, the Swift file that implements it, the ArrowMetal call that reaches it, a note,
and two executable pieces: a **call adapter** that runs the function through ArrowMetal under Arrow's
own argument and option names, and an **oracle** that produces the answer `pyarrow.compute` gives for
the same input.

`python/tests/test_functions.py` then does four things, and the third is the point:

1. asserts the registry covers every name `pyarrow.compute.list_functions()` reports, so the list
   cannot drift as Arrow grows, and invents none of its own;
2. asserts every row is well formed — a status from the vocabulary below, a note, and, for a row that
   claims to work, a call, an example input and a named source file;
3. **runs** every `gpu` / `cpu` / `partial` row through `arrowmetal.functions.call_function` and
   compares the result to `pyarrow.compute`, value for value, with a second input in a different
   Arrow type family for the rows whose claim spans several. Float comparisons use the tolerance
   recorded for that row in `arrowmetal.functions.TOLERANCE`; where an answer legitimately differs
   from Arrow's, the row carries an oracle that checks the property Arrow actually specifies, and the
   note says what the difference is;
4. asserts that a `missing` row raises rather than quietly doing something.

So a status here is a measurement, not an intention. Nothing in this table is reachable-in-principle:
if it says `gpu`, `cpu` or `partial`, a test called it this run.

## What the statuses mean

| Status | Meaning |
|---|---|
| **GPU** | A Metal kernel does the work. Host code sets up buffers and reads the answer back, nothing more. |
| **CPU** | Implemented and reachable through the ArrowMetal API, but the work happens on the host. Every row here says *why* the host is the right place — a timezone database, a Unicode table, an ICU regex, or an output that is one row wide however long the input is. |
| **Partial** | Reachable, with a stated limitation. Three different things wear this label and each note says which: (a) an option or an input type Arrow supports and this does not; (b) an answer that deliberately differs from Arrow's — group order, ascending `unique`, an int32 where Arrow returns int64; (c) an evaluation genuinely split between the GPU and the host, of which the ten `utf8_is_*` predicates are the clearest case: they answer every row on the GPU and re-decide on the CPU only the rows carrying a byte >= 0x80. |
| **Missing** | Not implemented. The note says why. |
| **Pending** | Reserved for a name landing on an unmerged branch. No row carries it today. |

Precision is recorded, not glossed. Three families of float difference exist and the note on each row
names the one that applies: a float32 evaluation of a float64 column (about 1e-7 relative — Metal has
no `double` transcendentals, so `exp`, the plain logarithms, `sqrt` and `power` widen a `float`
result); the software binary64 routines (within 5 ulp of the host libm — the trigonometric family,
`expm1`, `log1p`, `logb`, `hypot`); and the grouped moments, whose deviations are formed in float32
about a float64 mean (about 1e-5 relative).

## Regenerating this file

No Makefile, one command:

```sh
PYTHONPATH=python python python/tests/function_table_report.py --page > docs/ARROW_FUNCTIONS.md
```

Run it from the repository root, with a Python that has `pyarrow` installed and the ArrowMetal dylib
built (`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release --product \
ArrowMetalC`). Check the numbers first with:

```sh
PYTHONPATH=python python -m pytest python/tests/test_functions.py -q
```

## Summary by section
"""


def page():
    counts = F.status_counts()["Total"]
    hashes = sum(1 for n in F.list_functions() if n.startswith("hash_"))
    total = len(F.list_functions())
    out = [_INTRO.format(total=total, docs=total - hashes, hashes=hashes)]
    out.append(F.summary_markdown())
    out.append("")
    out.append(f"Totals: **{counts.get(F.GPU, 0)} gpu**, **{counts.get(F.CPU, 0)} cpu**, "
               f"**{counts.get(F.PARTIAL, 0)} partial**, **{counts.get(F.MISSING, 0)} missing**, "
               f"**{counts.get(F.PENDING, 0)} pending** over {total} Arrow function names.")
    out.append("")
    out.append("## Every Arrow function name")
    out.append("")
    out.append(F.markdown_table())
    out.append("")
    out.append("---")
    out.append("")
    out.append("Version 0.1.0. Read alongside [COVERAGE.md](COVERAGE.md), [ROADMAP.md](../ROADMAP.md) "
               "and [DESIGN.md](DESIGN.md).")
    return "\n".join(out)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    unknown = [s for s in args if s not in F.SECTIONS]
    if unknown:
        sys.stderr.write(f"unknown section(s): {', '.join(unknown)}\n"
                         f"known sections: {', '.join(F.SECTIONS)}\n")
        return 2
    if "--page" in flags:
        print(page())
        return 0
    print(f"<!-- generated by python/tests/function_table_report.py; "
          f"{len(F.list_functions())} Arrow function names -->")
    print()
    print(F.summary_markdown())
    if "--summary" not in flags:
        print()
        print(F.markdown_table(args or None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
