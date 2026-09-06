# ArrowMetal documentation

Start here. Everything in this folder is written so that someone who has never seen the code can verify
every claim the project makes.

| Read this | To learn |
|---|---|
| [../README.md](../README.md) | What ArrowMetal is, the pitch, headline numbers, quick start in Swift and Python |
| [COVERAGE.md](COVERAGE.md) | Exactly which Apache Arrow compute functions and types are supported, and how (GPU, CPU, partial), function by function |
| [EVALUATION.md](EVALUATION.md) | How correctness is checked: CPU oracles, the scenario matrix, bit-exact IEEE-754 tests, and the differential harness against pyarrow.compute, with open findings |
| [TESTING.md](TESTING.md) | Every test suite, what it covers, and the commands to run it |
| [BENCHMARKS.md](BENCHMARKS.md) | Every benchmark round with hardware, methodology, wall time and CPU time; the tables the README quotes |
| [DESIGN.md](DESIGN.md) | How it works: buffers, kernels, batching, GPU-side lengths, software Float64, where the time goes |
| [EXPR.md](EXPR.md) | Fused expression queries: one runtime-generated kernel for a whole expression DAG, the grammar, how nulls are compiled, the numbers and the limits |
| [DECISIONS.md](DECISIONS.md) | Why it is built this way, one dated entry per decision |
| [FINDINGS.md](FINDINGS.md) | Things learned the hard way: toolchain quirks, Metal limits, bugs and their lessons |
| [../ROADMAP.md](../ROADMAP.md) | What is next and what is open for contributors |
| [../CHANGELOG.md](../CHANGELOG.md) | What is in the (unreleased) 0.1.0 |
| [../Benchmarks/README.md](../Benchmarks/README.md) | The three benchmark programs and the fairness rules |
| [../python/README.md](../python/README.md) | The Python package: install, wheel build, usage |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | How to add a kernel, test it, and benchmark it |

## Reproducing everything

All of it runs on any Apple silicon Mac with Xcode installed:

```
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test               # Swift suite, debug
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test -c release    # same, optimised
swift build -c release --product ArrowMetalC && PYTHONPATH=python python -m pytest python/tests -q
swift run -c release arrowmetal-bench            # Metal vs all CPU cores vs Accelerate, with CPU-ms
python Benchmarks/python_bench.py                # Polars / pyarrow / pandas on the same data shapes
PYTHONPATH=python python Benchmarks/python_gpu_bench.py   # ArrowMetal from Python vs Polars, in-process
PYTHONPATH=python python Benchmarks/expr_bench.py         # fused expression queries vs one kernel per operator
swift run -c release arrowmetal-examples         # six end-to-end scenarios
```

Every benchmark table in the repository names the machine it was measured on. Numbers from GitHub-hosted
runners are never published: their GPU is virtual.
