from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Any
import pandas as pd
from .paths import P
from .manifest import log_artifact

def _now(fmt="%Y%m%d-%H%M%S"):
    return time.strftime(fmt, time.localtime())

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def _make_name(module: str, dataset: str, desc: str, ext: str, ts_fmt: str = "%Y%m%d-%H%M%S") -> str:
    ts = _now(ts_fmt)
    return f"{_safe_name(module)}__{_safe_name(dataset)}__{_safe_name(desc)}__{ts}.{ext.lstrip('.')}"

def save_json(obj: Any, module="lab", dataset="generic", desc="metrics", tags=None, meta=None) -> Path:
    P.metrics.mkdir(parents=True, exist_ok=True)
    name = _make_name(module, dataset, desc, "json")
    path = P.metrics / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log_artifact(path, kind="metrics", tags=tags or [module, dataset, "json"], meta=meta or {})
    return path

def save_df(df: pd.DataFrame, module="lab", dataset="generic", desc="table", tags=None, meta=None, fmt="parquet") -> Path:
    P.tables.mkdir(parents=True, exist_ok=True)
    name = _make_name(module, dataset, desc, "parquet" if fmt=="parquet" else "csv")
    path = P.tables / name
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    log_artifact(path, kind="table", tags=tags or [module, dataset, fmt], meta=meta or {"rows": int(getattr(df, "shape", [0,0])[0])})
    return path

def save_figure(fig, module="lab", dataset="generic", desc="figure", tags=None, meta=None, ext="png", dpi=200):
    P.figures.mkdir(parents=True, exist_ok=True)
    name = _make_name(module, dataset, desc, ext)
    path = P.figures / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log_artifact(path, kind="figure", tags=tags or [module, dataset, "figure"], meta=meta or {})
    return path

def save_bytes(b: bytes, module="lab", dataset="generic", desc="blob", tags=None, meta=None, ext="bin") -> Path:
    P.artifacts.mkdir(parents=True, exist_ok=True)
    name = _make_name(module, dataset, desc, ext)
    path = P.artifacts / name
    with open(path, "wb") as f:
        f.write(b)
    log_artifact(path, kind="blob", tags=tags or [module, dataset, "blob"], meta=meta or {"size": len(b)})
    return path
