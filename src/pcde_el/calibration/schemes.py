"""Calibration schemes and re-ranking.

    subtract:  s(e) = max(s_o, s_g - lam)      lam >= 0
    multiply:  s(e) = max(s_o, lam * s_g)      0 <= lam <= 1

Only the generated side is changed. Since the entry score is a max, an
entry with a good original name is never hurt by calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pcde_el.retrieval.aggregation import NEG_INF

SCHEMES = ("subtract", "multiply")
NO_CALIBRATION = {"subtract": 0.0, "multiply": 1.0}  # lam that leaves scores unchanged


def calibrated_generated(s_g: np.ndarray, lam: float, scheme: str) -> np.ndarray:
    if scheme == "subtract":
        return s_g - lam
    elif scheme == "multiply":
        # Scaling only lowers a score if it is non-negative (cosine is, after
        # training; BM25 always is). NEG_INF must stay NEG_INF, not become 0.
        finite = s_g > NEG_INF / 2
        if (s_g[finite] < 0).any():
            raise ValueError("Negative scores for multiply scheme is not supported")
        return np.where(finite, s_g * lam, NEG_INF)
    else:
        raise ValueError(f"Unknown scheme {scheme!r}")


@dataclass
class Ranking:
    """Gold rank per query (1-based, inf if not in the top k_cov), indexed by qid."""

    gold_rank: np.ndarray

    @property
    def correct1(self) -> np.ndarray:
        return self.gold_rank == 1

    @property
    def correct5(self) -> np.ndarray:
        return self.gold_rank <= 5

    @property
    def covered(self) -> np.ndarray:
        return np.isfinite(self.gold_rank)

    @property
    def reciprocal_rank(self) -> np.ndarray:
        return np.where(self.covered, 1.0 / self.gold_rank, 0.0)

    def subset(self, qids: np.ndarray) -> Ranking:
        return Ranking(self.gold_rank[qids])

    def metrics(self) -> dict[str, float]:
        n_cov = self.covered.sum()
        res = lambda c: float(c.sum() / n_cov) if n_cov > 0 else float("nan")
        return {
            "P@1": float(self.correct1.mean()),
            "P@5": float(self.correct5.mean()),
            "MRR": float(self.reciprocal_rank.mean()),
            "Coverage": float(self.covered.mean()),
            "Resolution@1": res(self.correct1),
            "Resolution@5": res(self.correct5),
        }


def rank(
    cand: pd.DataFrame,
    gold: np.ndarray,
    lam: float,
    scheme: str = "subtract",
    k_cov: int = 50,
    drop_generated: bool = False,
) -> Ranking:
    """Re-rank the candidates with calibrated scores and locate the gold entry.

    drop_generated ignores generated names entirely (the "original only"
    baseline). Ties are broken by pid so results are deterministic.
    """
    s_o = cand["s_o"].to_numpy()
    if drop_generated:
        s_g = np.full(len(cand), NEG_INF)
    else:
        s_g = calibrated_generated(cand["s_g"].to_numpy(), lam, scheme)

    s = np.maximum(s_o, s_g)
    alive = s > NEG_INF / 2

    d = pd.DataFrame(
        {
            "qid": cand["qid"].to_numpy()[alive],
            "pid": cand["pid"].to_numpy()[alive],
            "s": s[alive],
        }
    )

    d = d.sort_values(["qid", "s", "pid"], ascending=[True, False, True])
    d["rank"] = d.groupby("qid", sort=False).cumcount() + 1
    d = d[d["rank"] <= k_cov]

    qid = d["qid"].to_numpy()
    is_gold = d["pid"].to_numpy() == gold[qid]
    # Queries whose gold entry is not in the top k_cov keep rank inf.
    gold_rank = np.full(len(gold), np.inf)
    np.minimum.at(gold_rank, qid[is_gold], d["rank"].to_numpy()[is_gold].astype(float))

    return Ranking(gold_rank)
