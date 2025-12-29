from __future__ import annotations
from warnings import warn

from typing import Any, Dict, Iterable, Tuple, Type, overload
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO

from pydantic import BaseModel, create_model
from cbor2 import dump, load

import polars as pl
import pandas as pd

# ============================
# Errors
# ============================

class PipelineError(Exception): pass
class SerializationError(PipelineError): pass
class SchemaError(PipelineError): pass

# ============================
# Expr
# ============================

@dataclass(frozen=True, slots=True)
class Expr:
    op: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Tuple[Tuple[str, Any], ...]

    @staticmethod
    def build(op: str, name: str | None, args: Iterable[Any], kwargs: Dict[str, Any]) -> "Expr":
        return Expr(
            op=op,
            name=name or op,
            args=tuple(args),
            kwargs=tuple(sorted(kwargs.items())),
        )

    def apply(self, lf: pl.LazyFrame, resolve: callable) -> pl.LazyFrame:
        fn = getattr(lf, self.op, None)
        if fn is None:
            raise AttributeError(f"LazyFrame has no method '{self.op}'")

        return fn(
            *[resolve(a, lf) for a in self.args],
            **{k: resolve(v, lf) for k, v in self.kwargs},
        )

# ============================
# Serialization helpers
# ============================

_TYPE = "__type__"

def _encode(obj: Any) -> Any:
    if isinstance(obj, Pipeline):
        return {_TYPE: "pipeline", "actions": [_action_to_dict(a) for a in obj.actions]}
    if isinstance(obj, pl.Expr):
        return {_TYPE: "expr", "value": obj.meta.serialize()}
    return obj

def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict) or _TYPE not in obj:
        return obj
    if obj[_TYPE] == "expr":
        return pl.Expr.deserialize(obj["value"])
    if obj[_TYPE] == "pipeline":
        return Pipeline(tuple(_action_from_dict(a) for a in obj["actions"]))
    raise SerializationError(f"Unknown encoded type: {obj[_TYPE]}")

def _action_to_dict(e: Expr) -> dict:
    return {
        "op": e.op,
        "name": e.name,
        "args": [_encode(x) for x in e.args],
        "kwargs": [(k, _encode(v)) for k, v in e.kwargs],
    }

def _action_from_dict(d: dict) -> Expr:
    return Expr(
        op=d["op"],
        name=d["name"],
        args=tuple(_decode(x) for x in d["args"]),
        kwargs=tuple((k, _decode(v)) for k, v in d["kwargs"]),
    )

# ============================
# Pydantic
# ============================

def _model_from_schema(schema: pl.Schema, name: str) -> Type[BaseModel]:
    fields = {}
    for col, dtype in schema.items():
        if isinstance(dtype, pl.Struct):
            nested = pl.Schema({f.name: f.dtype for f in dtype.fields})
            fields[col] = (_model_from_schema(nested, f"{name}_{col}"), ...)
        else:
            fields[col] = (dtype.to_python(), ...)
    return create_model(name, **fields)

class Pipeline:
    __actions: Tuple[Expr, ...] = ()
    __schemas: Tuple[pl.Schema, pl.Schema] | None = None

    def __init__(self, actions: Tuple[Expr, ...] = (), schemas: Tuple[pl.Schema, pl.Schema] | None = None):
        self.__actions = actions
        self.__schemas = schemas

    @property
    def actions(self) -> Tuple[Expr, ...]:
        return self.__actions

    @property
    def schemas(self) -> Tuple[pl.Schema, pl.Schema] | None:
        return self.__schemas

    def __getattr__(self, name: str):
        def call(*args, **kwargs):
            label = kwargs.pop("pipeline_name", None)
            action = Expr.build(name, label, args, kwargs)
            return Pipeline(self.__actions + (action,), self.__schemas)
        return call

    def _resolve(self, value: Any, lf: pl.LazyFrame) -> Any:
        if isinstance(value, Pipeline):
            return value.exec(lf, lazy=True)
        return value

    def _lazy_frame(self, data) -> pl.LazyFrame:
        if isinstance(data, pl.LazyFrame): return data
        if isinstance(data, pl.DataFrame): return data.lazy()
        if isinstance(data, pd.DataFrame): return pl.from_pandas(data).lazy()
        raise TypeError("Invalid input")
    
    @overload
    def exec(self, data, *, lazy: bool = False, pandas: bool = False) -> pl.DataFrame: ...
    @overload
    def exec(self, data, *, lazy: bool = True,  pandas: bool = False) -> pl.LazyFrame: ...
    @overload
    def exec(self, data, *, lazy: bool = False, pandas: bool = True) -> pd.DataFrame: ...

    def exec(self, data, *, lazy=False, pandas=False):
        if lazy and pandas:
            raise ValueError("lazy and pandas are mutually exclusive")

        lf = self._lazy_frame(data)
        for a in self.__actions:
            lf = a.apply(lf, self._resolve)

        if lazy: return lf
        out = lf.collect()
        return out.to_pandas() if pandas else out
    
    def pydantic_models(self):
        if not self.schemas:
            raise SchemaError("No schema defined")
        i, o = self.schemas
        return (
            _model_from_schema(i, "PipelineInput"),
            _model_from_schema(o, "PipelineOutput"),
        )

    def build_pydantic_models(self):
        return self.pydantic_models()
    
    # ---- persistence ----

    def save(self, path: str | Path = "pipeline.polars", schema: pl.Schema = None) -> None:
        payload = {
            "polars_version": pl.__version__,
            "actions": [_action_to_dict(a) for a in self.actions],
            "schemas": None,
        }

        if schema:
            input_frame = schema.to_frame()
            output_frame = self.exec(input_frame)
            payload["schemas"] = (input_frame.serialize(), output_frame.serialize())

        with Path(path).open("wb") as f:
            dump(payload, f)
            
        return path

    @staticmethod
    def load(path: str | Path) -> "Pipeline":
        with Path(path).open("rb") as f:
            raw = load(f)

        if raw.get("polars_version") != pl.__version__:
            warn("Polars version mismatch", RuntimeWarning)

        actions = tuple(_action_from_dict(a) for a in raw["actions"])
        schemas = raw.get("schemas")

        if schemas and schemas[0] and schemas[1]:
            i = pl.DataFrame.deserialize(BytesIO(schemas[0])).schema
            o = pl.DataFrame.deserialize(BytesIO(schemas[1])).schema
            return Pipeline(actions, (i, o))

        return Pipeline(actions)

    # ---- repr ----

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        if not self.actions:
            return "Pipeline()"
        def _format_value(value: Any) -> str:
            if isinstance(value, pl.Expr):
                return f"<Expr {value!s}>"
            return repr(value)
        lines = ["Pipeline("]
        for a in self.actions:
            lines.append(f"  {a.name}(")
            for x in a.args:
                lines.append(f"    {_format_value(x)}")
            for k, v in a.kwargs:
                lines.append(f"    {k}={_format_value(v)}")
            lines.append("  )")
        lines.append(")")
        return "\n".join(lines)
