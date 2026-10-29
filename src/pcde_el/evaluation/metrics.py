"""Per-query outcomes and the paper's metrics.

Pass@k = Coverage@50 x Resolution@k: coverage asks whether the gold entry
made the top 50 at all, resolution whether it was ranked within k once it did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_query_outcome(
    entries: pd.DataFrame, gold: np.ndarray, group_cols: tuple[str, ...] = ()
) -> pd.DataFrame:
    """One row per (group..., qid) with gold_rank, covered, pass_at_{1,5,10}, rr.

    Every query appears in every group, including queries with no
    candidates at all (gold_rank NaN, counted as misses).
    """
    e = entries
    is_gold = e["pid"].to_numpy() == gold[e["qid"].to_numpy()]
    key = list(group_cols) + ["qid"]
    gold_rank = e[is_gold].groupby(key)["pid_rank"].min().rename("gold_rank")

    if group_cols:
        groups = [sorted(e[c].unique()) for c in group_cols]
        index = pd.MultiIndex.from_product(groups + [range(len(gold))], names=key)
    else:
        index = pd.RangeIndex(len(gold), name="qid")
    out = gold_rank.reindex(index).to_frame().reset_index()

    r = out["gold_rank"]
    out["covered"] = r.notna()
    for k in (1, 5, 10):
        out[f"pass_at_{k}"] = r.le(k).fillna(False).astype(bool)
    out["rr"] = (1.0 / r).fillna(0.0)
    return out


def summarize(
    outcome: pd.DataFrame, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Metrics for one group's outcome; `mask` (indexed by qid) restricts the queries."""
    o = outcome if mask is None else outcome[mask[outcome["qid"].to_numpy()]]
    num_cov = o["covered"].sum()
    return {
        "n": len(o),
        "Coverage@50": float(o["covered"].mean()),
        "Resolution@1": float(o["pass_at_1"].sum() / num_cov)
        if num_cov
        else float("nan"),
        "Resolution@5": float(o["pass_at_5"].sum() / num_cov)
        if num_cov
        else float("nan"),
        "Pass@1": float(o["pass_at_1"].mean()),
        "Pass@5": float(o["pass_at_5"].mean()),
        "MRR": float(o["rr"].mean()),
    }
