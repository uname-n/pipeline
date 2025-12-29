from sys import path; from os.path import dirname; path.insert(0, dirname(dirname(__file__)))

from pipeline import Pipeline
import polars as pl

pipeline = Pipeline()
pipeline = pipeline.with_columns(
    (pl.col("f1") + pl.col("f2") + pl.col("f3")).alias("x"), pipeline_name="create_feature_x"
)

out = (
    pipeline.select(
        pl.col("idx"),
        pl.col("a").struct.field("fa1").alias("fa1"),
        pl.col("a").struct.field("fa2").alias("fa2"),
        pipeline_name="select_a_fields"
    )
    .with_columns(
        (pl.col("fa1") + pl.col("fa2")).alias("y"),
        pipeline_name="create_feature_y"
    )
    .select(
        pl.col("idx"), pl.col("y"),
        pipeline_name="select_idx_y"
    )
)
pipeline = pipeline.join(out, on="idx", pipeline_name="join_a_dataset").drop("a")

data = pl.DataFrame([dict(idx=i, f1=i,f2=i,f3=i,a=dict(fa1=i,fa2=i)) for i in range(10)])
print(data)

pipeline.save(path="pipeline.polars", schema=data.schema)
print(pipeline)

loaded = Pipeline.load(path="pipeline.polars")
output = loaded.exec(data)
print(output)

print(loaded.build_pydantic_models()[0](**data.row(0, named=True)))