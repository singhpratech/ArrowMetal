# Roadmap

Where ArrowMetal is heading. There are no dates, and the order can change. What each release delivered
is in [CHANGELOG.md](../CHANGELOG.md).

## Performance

- Faster reads for the newer file and table formats.
- Lower cost per call on small inputs.
- More operations that choose between the CPU and the GPU on their own.

## Types and interop

- More of the Arrow type system and compute options.
- Wider Arrow C Data Interface and IPC support.

## Integrations

- Deeper Polars and DuckDB integration.
- More of the Python data ecosystem.

## Languages

Python, Swift, C, Rust, Go, TypeScript and R today, all over one C ABI; more languages can use the same
header.

## Non-goals

- Anything but Apple silicon.
- A dataframe API of its own: ArrowMetal is the engine under Polars, DuckDB and pandas.
- Approximate answers where Arrow specifies exact ones.

Ideas and requests are welcome as GitHub issues.
