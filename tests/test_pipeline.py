from datetime import datetime

import pandas as pd
import polars as pl

from pipeline import Pipeline, _decode, _encode

def test_pipeline_exec_with_polars_and_pandas():
    pipeline = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    ).select(
        pl.col("idx"),
        pl.col("x"),
        pipeline_name="select_cols",
    )

    pl_data = pl.DataFrame({"idx": [1, 2], "f1": [3, 4], "f2": [5, 6]})
    pl_result = pipeline.exec(pl_data)
    assert isinstance(pl_result, pl.DataFrame)
    assert pl_result.to_dict(as_series=False) == {"idx": [1, 2], "x": [8, 10]}

    pd_data = pd.DataFrame({"idx": [1, 2], "f1": [3, 4], "f2": [5, 6]})
    pd_result = pipeline.exec(pd_data)
    assert isinstance(pd_result, pl.DataFrame)
    assert pd_result.to_dict(as_series=False) == {"idx": [1, 2], "x": [8, 10]}

def test_pipeline_save_load_roundtrip(tmp_path):
    pipeline = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    )
    path = pipeline.save(tmp_path / "pipeline.polars")
    loaded = Pipeline.load(path)

    assert repr(loaded) == repr(pipeline)

def test_pipeline_save_creates_dir_and_loads(tmp_path):
    pipeline = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    )
    target_dir = tmp_path / "pipeline.polars"
    path = pipeline.save(target_dir)
    assert path.exists()
    loaded = Pipeline.load(target_dir)
    assert repr(loaded) == repr(pipeline)

def test_pipeline_exec_lazyframe_lazy_true():
    pipeline = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    )
    lf = pl.DataFrame({"f1": [1, 2], "f2": [3, 4]}).lazy()
    result = pipeline.exec(lf, lazy=True)
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().to_dict(as_series=False) == {"f1": [1, 2], "f2": [3, 4], "x": [4, 6]}

def test_pipeline_exec_nested_pipeline_join():
    base = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    )
    nested = base.select(pl.col("idx"), pl.col("x"), pipeline_name="select_x")
    joined = base.join(nested, on="idx", pipeline_name="join_nested")

    df = pl.DataFrame({"idx": [1, 2], "f1": [3, 4], "f2": [5, 6]})
    result = joined.exec(df)
    assert result.to_dict(as_series=False) == {
        "idx": [1, 2],
        "f1": [3, 4],
        "f2": [5, 6],
        "x": [8, 10],
        "x_right": [8, 10],
    }

def test_pipeline_save_load_roundtrip_exec(tmp_path):
    pipeline = Pipeline().with_columns(
        (pl.col("f1") + pl.col("f2")).alias("x"),
        pipeline_name="make_x",
    ).select(
        pl.col("idx"),
        pl.col("x"),
        pipeline_name="select_cols",
    )
    path = pipeline.save(f"{tmp_path}/pipeline.polars")
    loaded = Pipeline.load(path)

    df = pl.DataFrame({"idx": [1, 2], "f1": [3, 4], "f2": [5, 6]})
    result = loaded.exec(df)
    assert result.to_dict(as_series=False) == {"idx": [1, 2], "x": [8, 10]}

def test__encode__decode_expr_and_bytes():
    expr = pl.col("a") + 1
    _encoded = _encode(expr)
    _decoded = _decode(_encoded)
    df = pl.DataFrame({"a": [1, 2]})
    result = df.select(_decoded)
    assert result.to_dict(as_series=False) == {"a": [2, 3]}

    payload = b"pipeline-bytes"
    assert _decode(_encode(payload)) == payload

def test_big_pipeline():
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "group": ["a", "b", "a", "b", "a"],
            "f1": [10, 20, 30, 40, 50],
            "f2": [1, 2, 3, 4, 5],
            "txt": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "ts": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
            ],
            "vals": [[1, 2], [3], [4, 5, 6], [], [7, 8]],
        }
    )

    pipeline = (
        Pipeline()
        .with_columns(
            (pl.col("f1") + pl.col("f2")).alias("sum"),
            (pl.col("f1") - pl.col("f2")).alias("diff"),
            (pl.col("f1") / pl.col("f2")).alias("ratio"),
            pl.when(pl.col("f1") > 25).then(pl.lit("big")).otherwise(pl.lit("small")).alias("size"),
            pl.concat_str([pl.col("txt"), pl.lit("_"), pl.col("group")]).alias("label"),
            pl.col("txt").str.to_uppercase().alias("txt_upper"),
            pl.col("vals").list.sum().alias("vals_sum"),
            pl.struct(["f1", "f2"]).alias("pair"),
            pl.col("ts").dt.year().alias("year"),
            pipeline_name="feature_engineering",
        )
        .with_columns(
            (pl.col("sum") * 2).alias("sum2"),
            pl.col("ratio").round(2).alias("ratio2"),
            pl.col("label").str.slice(0, 4).alias("label_slice"),
            pl.col("pair").struct.field("f1").alias("pair_f1"),
            (pl.col("vals_sum") + pl.col("f1")).alias("mix"),
            pl.col("txt").str.contains("a").alias("has_a"),
            pl.col("sum").rolling_mean(window_size=2).alias("sum_rm"),
            pipeline_name="more_features",
        )
        .filter(pl.col("f1") > 0, pipeline_name="filter_positive")
        .sort("idx", pipeline_name="sort_idx")
        .with_row_index("row_nr", pipeline_name="row_index")
        .with_columns(
            pl.col("row_nr").cast(pl.Int64).alias("row_nr64"),
            pipeline_name="cast_index",
        )
        .select(
            "idx",
            "group",
            "f1",
            "f2",
            "sum",
            "sum2",
            "diff",
            "ratio2",
            "size",
            "label",
            "label_slice",
            "txt_upper",
            "vals_sum",
            "pair_f1",
            "year",
            "mix",
            "has_a",
            "sum_rm",
            "row_nr64",
            pipeline_name="select_cols",
        )
        .join(
            Pipeline()
            .group_by("group", pipeline_name="group_by")
            .agg(
                pl.col("sum").sum().alias("sum_total"),
                pl.col("sum").mean().alias("sum_mean"),
                pl.col("idx").n_unique().alias("idx_nunique"),
                pl.col("label").unique(maintain_order=True).alias("labels"),
                pl.col("has_a").any().alias("has_any_a"),
                pipeline_name="agg_group",
            ),
            on="group",
            how="left",
            pipeline_name="join_group_stats",
        )
        .join(
            Pipeline()
            .select("idx", "sum", pipeline_name="select_for_rank")
            .with_columns(
                pl.col("sum").rank(method="dense").alias("sum_rank"),
                pipeline_name="rank_sum",
            ),
            on="idx",
            how="left",
            pipeline_name="join_ranks",
        )
        .with_columns(
            (pl.col("sum_total") - pl.col("sum")).alias("sum_other"),
            pl.col("labels").list.len().alias("label_count"),
            pl.col("sum_rank").cast(pl.Int64).alias("sum_rank_int"),
            pipeline_name="post_join_features",
        )
        .sort(["group", "idx"], pipeline_name="final_sort")
    )

    expected = (
        df.lazy()
        .with_columns(
            (pl.col("f1") + pl.col("f2")).alias("sum"),
            (pl.col("f1") - pl.col("f2")).alias("diff"),
            (pl.col("f1") / pl.col("f2")).alias("ratio"),
            pl.when(pl.col("f1") > 25).then(pl.lit("big")).otherwise(pl.lit("small")).alias("size"),
            pl.concat_str([pl.col("txt"), pl.lit("_"), pl.col("group")]).alias("label"),
            pl.col("txt").str.to_uppercase().alias("txt_upper"),
            pl.col("vals").list.sum().alias("vals_sum"),
            pl.struct(["f1", "f2"]).alias("pair"),
            pl.col("ts").dt.year().alias("year"),
        )
        .with_columns(
            (pl.col("sum") * 2).alias("sum2"),
            pl.col("ratio").round(2).alias("ratio2"),
            pl.col("label").str.slice(0, 4).alias("label_slice"),
            pl.col("pair").struct.field("f1").alias("pair_f1"),
            (pl.col("vals_sum") + pl.col("f1")).alias("mix"),
            pl.col("txt").str.contains("a").alias("has_a"),
            pl.col("sum").rolling_mean(window_size=2).alias("sum_rm"),
        )
        .filter(pl.col("f1") > 0)
        .sort("idx")
        .with_row_index("row_nr")
        .with_columns(pl.col("row_nr").cast(pl.Int64).alias("row_nr64"))
        .select(
            "idx",
            "group",
            "f1",
            "f2",
            "sum",
            "sum2",
            "diff",
            "ratio2",
            "size",
            "label",
            "label_slice",
            "txt_upper",
            "vals_sum",
            "pair_f1",
            "year",
            "mix",
            "has_a",
            "sum_rm",
            "row_nr64",
        )
        .join(
            df.lazy()
            .with_columns(
                (pl.col("f1") + pl.col("f2")).alias("sum"),
                pl.concat_str([pl.col("txt"), pl.lit("_"), pl.col("group")]).alias("label"),
                pl.col("txt").str.contains("a").alias("has_a"),
            )
            .group_by("group")
            .agg(
                pl.col("sum").sum().alias("sum_total"),
                pl.col("sum").mean().alias("sum_mean"),
                pl.col("idx").n_unique().alias("idx_nunique"),
                pl.col("label").unique(maintain_order=True).alias("labels"),
                pl.col("has_a").any().alias("has_any_a"),
            ),
            on="group",
            how="left",
        )
        .join(
            df.lazy()
            .with_columns((pl.col("f1") + pl.col("f2")).alias("sum"))
            .select("idx", "sum")
            .with_columns(pl.col("sum").rank(method="dense").alias("sum_rank")),
            on="idx",
            how="left",
        )
        .with_columns(
            (pl.col("sum_total") - pl.col("sum")).alias("sum_other"),
            pl.col("labels").list.len().alias("label_count"),
            pl.col("sum_rank").cast(pl.Int64).alias("sum_rank_int"),
        )
        .sort(["group", "idx"])
        .collect()
    )

    result = pipeline.exec(df)
    assert result.to_dict(as_series=False) == expected.to_dict(as_series=False)
