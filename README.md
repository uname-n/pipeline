# pipeline

**composable, serializable data pipelines for polars and pandas**

`pipeline` is a lightweight utility for building lazy, reusable sequences of
`LazyFrame` transformations. Pipelines can be composed, serialized, persisted,
and later replayed against new datasets—whether backed by polars or pandas.

It’s designed for situations where you want **portable, repeatable data-shaping
logic** without tightly coupling transformations to a single dataframe instance.

## Installation

```bash
poetry add git+https://github.com/uname-n/pipeline.git
```