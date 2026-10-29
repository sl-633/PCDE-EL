"""Out-of-fold lambda selection.

lambda is chosen by Pass@1 on labelled queries and evaluated on queries it
was not chosen on: the official train+dev -> test split, or for datasets
without one, K folds over gold entries (all mentions of an entry land in
the same fold, so nothing about the entry leaks into its own selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pcde_el.calibration.schemes import NO_CALIBRATION, Ranking, rank
from pcde_el.evaluation.stats import clustered_bootstrap, mcnemar_exact_p, significant
from pcde_el.retrieval.aggregation import NEG_INF


def lambda_grid(cand: pd.DataFrame, scheme: str, n: int = 21) -> np.ndarray:
    """Candidate lambdas.

    multiply: evenly spaced in [0, 1].
    subtract: score scales differ between retrievers, so the grid comes from
    the data: 0, quantiles of the margins by which a generated name beats the
    entry's best original name, and a value just above the largest margin.
    """
    if scheme == "multiply":
        return np.unique(np.linspace(0.0, 1.0, n))
    s_o, s_g = cand["s_o"].to_numpy(), cand["s_g"].to_numpy()
    both = (s_o > NEG_INF / 2) & (s_g > NEG_INF / 2)
    gap = (s_g - s_o)[both]
    gap = gap[gap > 0]
    if len(gap) == 0:
        raise ValueError("No entry has generated evidence outranking original evidence")
    return np.unique(
        np.concatenate(
            [[0.0], np.quantile(gap, np.linspace(0, 1, n)), [float(gap.max()) * 1.001]]
        )
    )


class LambdaSweep:
    """Rank every query once per grid lambda, up front.

    Selecting lambda on any subset of queries is then just a column slice,
    which is what makes the data-efficiency study cheap.
    """

    def __init__(
        self,
        cand: pd.DataFrame,
        gold: np.ndarray,
        scheme: str,
        k_cov: int = 50,
        n_lambda: int = 21,
        grid: np.ndarray | None = None,
    ):
        self.scheme = scheme
        self.grid = lambda_grid(cand, scheme, n_lambda) if grid is None else grid
        self.rankings = [
            rank(cand, gold, float(lam), scheme, k_cov) for lam in self.grid
        ]
        self._correct1 = np.stack([r.correct1 for r in self.rankings])

    def select(self, qids: np.ndarray) -> tuple[int, float]:
        """Grid index with the best Pass@1 on qids, and that Pass@1.
        Ties go to the smallest index, i.e. the mildest calibration."""
        if len(qids) == 0:
            # An empty mean is NaN and argmax over all-NaN silently returns 0,
            # i.e. the first grid lambda would be "selected" with no evidence.
            raise ValueError("cannot select lambda on an empty set of queries")
        acc = self._correct1[:, qids].mean(axis=1)
        i = int(np.argmax(acc))
        return i, float(acc[i])

    def lam(self, i: int) -> float:
        return float(self.grid[i])

    def position(self, i: int) -> float:
        """Index scaled to [0, 1], comparable across grids of different lambdas."""
        return i / (len(self.grid) - 1) if len(self.grid) > 1 else 0.0


def entry_folds(entries: np.ndarray, folds: int, seed: int = 0) -> np.ndarray:
    """Fold id per query, assigned per gold entry."""
    uniq = np.array(sorted(set(entries)))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    return np.array([fold_of[e] for e in entries])


def selection_splits(
    test: pd.DataFrame,
    protocol: str,
    folds: int = 5,
    seed: int = 0,
    list_col: str = "list",
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(label, selection qids, report qids)], one tuple per fold."""
    all_q = np.arange(len(test))
    if protocol == "split":
        if list_col not in test.columns:
            raise ValueError(f"protocol 'split' needs a '{list_col} column")
        sel = all_q[test[list_col].isin(["train", "dev"]).to_numpy()]
        rep = all_q[(test[list_col] == "test").to_numpy()]
        return [("train+dev", sel, rep)]

    if protocol == "cv":
        fold = entry_folds(test["gold_pid"].astype(str).to_numpy(), folds, seed)
        return [
            (f"fold_{f}", all_q[fold != f], all_q[fold == f])
            for f in range(folds)
            if (fold == f).any()
        ]

    raise ValueError(f"unknown protocol {protocol!r}")


def report_qids(splits) -> np.ndarray:
    return np.sort(np.concatenate([rep for _, _, rep in splits]))


@dataclass
class OutOfFold:
    report_q: np.ndarray
    ranking: Ranking
    chosen: list[dict] = field(default_factory=list)


def out_of_fold(sweep: LambdaSweep, splits) -> OutOfFold:
    """Each report query is ranked with the lambda selected on the other side
    of its split; the folds are then stitched into one ranking."""
    n_q = sweep.rankings[0].gold_rank.shape[0]
    gold_rank = np.full(n_q, np.inf)
    chosen = []
    for label, sel, rep in splits:
        i, acc = sweep.select(sel)
        gold_rank[rep] = sweep.rankings[i].gold_rank[rep]
        chosen.append(
            {
                "split": label,
                "lambda": sweep.lam(i),
                "selection_p1": acc,
                "n_select": len(sel),
                "n_report": len(rep),
            }
        )
    rq = report_qids(splits)
    return OutOfFold(rq, Ranking(gold_rank[rq]), chosen)


def reference_rankings(
    cand: pd.DataFrame, gold: np.ndarray, k_cov: int = 50
) -> dict[str, Ranking]:
    """Baselines: D1 as is, and D1 with generated names ignored."""
    return {
        "uncalibrated": rank(cand, gold, NO_CALIBRATION["subtract"], "subtract", k_cov),
        "generated_dropped": rank(
            cand, gold, 0.0, "subtract", k_cov, drop_generated=True
        ),
    }


def calibrate(
    cand: pd.DataFrame,
    test: pd.DataFrame,
    scheme: str,
    protocol: str,
    k_cov: int = 50,
    n_lambda: int = 21,
    folds: int = 5,
    seed: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Select lambda out-of-fold and compare against the baselines.

    Returns a JSON-ready summary (chosen lambdas, metrics, paired tests vs
    uncalibrated) and per-query gold ranks on the report queries.
    """
    gold = test["gold_pid"].astype(str).to_numpy()
    sweep = LambdaSweep(cand, gold, scheme, k_cov, n_lambda)
    splits = selection_splits(test, protocol, folds, seed)
    oof = out_of_fold(sweep, splits)
    refs = {
        k: r.subset(oof.report_q)
        for k, r in reference_rankings(cand, gold, k_cov).items()
    }
    cal, unc = oof.ranking, refs["uncalibrated"]
    clusters = gold[oof.report_q]

    def test_vs(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict:
        d, lo, hi = clustered_bootstrap(a, b, c, seed=seed)
        return {
            "delta": d,
            "ci": [lo, hi],
            "significant": significant(lo, hi),
            "mcnemar_p": mcnemar_exact_p(int((~b & a).sum()), int((b & ~a).sum())),
        }

    # Calibration only re-ranks the top k_cov, so coverage is unchanged and
    # Resolution@5 can be compared on the same covered queries.
    cov = unc.covered
    summary = {
        "scheme": scheme,
        "protocol": protocol,
        "n_reported": len(oof.report_q),
        "lambda_grid": [float(x) for x in sweep.grid],
        "chosen": oof.chosen,
        "metrics": {
            "generated_dropped": refs["generated_dropped"].metrics(),
            "uncalibrated": unc.metrics(),
            "calibrated": cal.metrics(),
        },
        "vs_uncalibrated": {
            "P@1": test_vs(cal.correct1, unc.correct1, clusters),
            "P@5": test_vs(cal.correct5, unc.correct5, clusters),
            "Resolution@5": test_vs(
                cal.correct5[cov], unc.correct5[cov], clusters[cov]
            ),
        },
    }
    per_query = pd.DataFrame(
        {
            "qid": oof.report_q,
            "gold_pid": clusters,
            "rank_generated_dropped": refs["generated_dropped"].gold_rank,
            "rank_uncalibrated": unc.gold_rank,
            "rank_calibrated": cal.gold_rank,
        }
    )
    return summary, per_query
