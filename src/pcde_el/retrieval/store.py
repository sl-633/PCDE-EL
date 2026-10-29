"""Saving and loading retrieval results under results/<dataset>/<condition>/<source>/.

hits.parquet holds both dictionaries (D0, D1) and, for dense retrieval,
every model tag; the loaders filter by them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pcde_el import paths
from pcde_el.retrieval.aggregation import candidates

DICT_TAGS = ("D0", "D1")
BM25_MODEL = "bm25"


def save(
    dataset: str,
    condition: str,
    source: str,
    hits: pd.DataFrame,
    queries: pd.DataFrame,
    outcome: pd.DataFrame,
    meta: dict,
) -> None:
    out = paths.retrieval_dir(dataset, condition, source)
    out.mkdir(parents=True, exist_ok=True)
    hits.to_parquet(out / "hits.parquet", index=False)
    queries.to_parquet(out / "queries.parquet", index=False)
    outcome.to_parquet(out / "outcome.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}/{{hits, queries, outcome}}.parquet, meta.json")


def _read(dataset: str, condition: str, source: str, name: str) -> pd.DataFrame:
    path = paths.retrieval_dir(dataset, condition, source) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run the {source} retrieval script first.")
    return pd.read_parquet(path)


def _model_tag(source: str, model: str | None) -> str:
    return BM25_MODEL if source == "bm25" else (model or "ft_gen")


def load_queries(dataset: str, condition: str, source: str) -> pd.DataFrame:
    q = _read(dataset, condition, source, "queries").reset_index(drop=True)
    q["gold_pid"] = q["gold_pid"].astype(str)
    return q


def load_entries(
    dataset: str,
    condition: str,
    source: str,
    dict_tag: str = "D1",
    model: str | None = None,
) -> pd.DataFrame:
    h = _read(dataset, condition, source, "hits")
    h = h[(h["model"] == _model_tag(source, model)) & (h["dict"] == dict_tag)]
    if h.empty:
        raise ValueError(f"No {source} hits for model={model} dict={dict_tag}")
    return h.reset_index(drop=True)


def load_candidates(
    dataset: str, condition: str, source: str, model: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """D1 candidates for calibration, plus the queries and gold ids."""
    queries = load_queries(dataset, condition, source)
    cand = candidates(load_entries(dataset, condition, source, "D1", model))
    return cand, queries, queries["gold_pid"].to_numpy()


def load_outcome(
    dataset: str, condition: str, source: str, dict_tag: str, model: str | None = None
) -> pd.DataFrame | None:
    """Per-query outcome sorted by qid, or None if that run does not exist."""
    try:
        o = _read(dataset, condition, source, "outcome")
    except FileNotFoundError:
        return None
    o = o[(o["model"] == _model_tag(source, model)) & (o["dict"] == dict_tag)]
    return o.sort_values("qid").reset_index(drop=True) if not o.empty else None
