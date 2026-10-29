"""How many labelled queries does lambda selection need?

The evaluation queries stay fixed. For each budget B, draw B labelled
queries from the pool, select lambda on them, and score it on the
evaluation queries; repeat. Both schemes use the same draws, so they can
be compared draw by draw.

    floor    uncalibrated D1
    ceiling  lambda tuned on the evaluation queries themselves (oracle)
    fraction of ceiling = (score - floor) / (ceiling - floor)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcde_el.calibration.schemes import SCHEMES, Ranking
from pcde_el.calibration.selection import LambdaSweep, reference_rankings
from pcde_el.evaluation.stats import clustered_bootstrap, mcnemar_exact_p

DEFAULT_BUDGETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
THRESHOLDS = (0.5, 0.75, 0.9, 0.95, 0.99)
METRICS = {
    "p1": "P@1",
    "p5": "P@5",
    "coverage": "Coverage",
    "resolution1": "Resolution@1",
    "resolution5": "Resolution@5",
    "mrr": "MRR",
}
PER_QUERY_OUTCOME = {"p1": "correct1", "p5": "correct5"}

Split = tuple[str, np.ndarray, np.ndarray]  # (name, labelled pool, evaluation queries)


@dataclass
class DataEfficiencyResult:
    samples: pd.DataFrame  # one row per (split, budget, repeat, scheme)
    refs: pd.DataFrame  # one row per split: original, uncalibrated and oracle scores
    curve: pd.DataFrame  # one row per (scheme, budget), plus budget "full"
    win_rate: pd.DataFrame  # one row per budget
    summary: dict


def run(
    cand: pd.DataFrame,
    gold: np.ndarray,
    splits: Sequence[Split],
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    repeats: int = 30,
    k_cov: int = 50,
    n_lambda: int = 21,
    seed: int = 0,
) -> DataEfficiencyResult:
    """Budgets at or above the pool size are skipped; the whole pool is always
    evaluated once as budget "full". With several splits (CV), results are
    pooled, weighted by the number of evaluation queries."""
    sweeps = {
        scheme: LambdaSweep(cand, gold, scheme, k_cov, n_lambda) for scheme in SCHEMES
    }
    baselines = reference_rankings(cand, gold, k_cov)
    rng = np.random.default_rng(seed)

    samples, refs, oracles = [], [], []
    for name, pool, report in splits:
        oracle = {scheme: _oracle(sweep, report) for scheme, sweep in sweeps.items()}
        refs.append(_reference_row(name, pool, report, baselines, oracle))
        oracles.append({scheme: ranking for scheme, (_, ranking) in oracle.items()})

        for budget in budgets:
            if budget >= len(pool):
                continue
            for repeat in range(repeats):
                # Uniform over queries, not stratified by entry: a real
                # annotation budget labels whatever queries it gets.
                labelled = rng.choice(pool, size=budget, replace=False)
                samples += _evaluate_selection(
                    sweeps,
                    labelled,
                    report,
                    fold=name,
                    kind="sampled",
                    budget=budget,
                    rep=repeat,
                )
        samples += _evaluate_selection(
            sweeps, pool, report, fold=name, kind="full", budget=len(pool), rep=0
        )

    samples, refs = pd.DataFrame(samples), pd.DataFrame(refs)
    floor, ceiling = _floor_and_ceiling(refs)
    curve = _learning_curve(samples, floor, ceiling)
    summary = {
        "floor": floor,
        "ceiling": ceiling,
        "budget_needed": _budget_needed(curve),
        "ceiling_gap": _ceiling_gap(gold, splits, oracles, seed),
    }
    return DataEfficiencyResult(
        samples, refs, curve, _paired_win_rate(samples), summary
    )


# ---------- per split


def _scores(prefix: str, ranking: Ranking) -> dict[str, float]:
    metrics = ranking.metrics()
    return {f"{prefix}_{short}": metrics[name] for short, name in METRICS.items()}


def _oracle(sweep: LambdaSweep, report: np.ndarray) -> tuple[float, Ranking]:
    best, _ = sweep.select(report)
    return sweep.lam(best), sweep.rankings[best].subset(report)


def _reference_row(
    name: str,
    pool: np.ndarray,
    report: np.ndarray,
    baselines: dict[str, Ranking],
    oracle: dict[str, tuple[float, Ranking]],
) -> dict:
    row = {"fold": name, "n_pool": len(pool), "n_report": len(report)}
    row |= _scores("original", baselines["generated_dropped"].subset(report))
    row |= _scores("expanded", baselines["uncalibrated"].subset(report))
    for scheme, (lam, ranking) in oracle.items():
        row[f"oracle_lambda_{scheme}"] = lam
        row |= _scores(f"oracle_{scheme}", ranking)
    return row


def _evaluate_selection(
    sweeps: dict[str, LambdaSweep],
    labelled: np.ndarray,
    report: np.ndarray,
    **draw: object,
) -> list[dict]:
    rows = []
    for scheme, sweep in sweeps.items():
        chosen, _ = sweep.select(labelled)
        rows.append(
            {
                **draw,
                "scheme": scheme,
                "lambda": sweep.lam(chosen),
                "lambda_pos": sweep.position(chosen),
                "n_report": len(report),
                **_scores("held", sweep.rankings[chosen].subset(report)),
            }
        )
    return rows


# ---------- pooling


def _mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    return float(np.average(values, weights=weights))


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def _fraction_recovered(value: float, floor: float, ceiling: float) -> float:
    span = ceiling - floor
    return (value - floor) / span if span > 0 else float("nan")


def _floor_and_ceiling(refs: pd.DataFrame) -> tuple[dict, dict]:
    weights = refs["n_report"]
    floor = {m: _mean(refs[f"expanded_{m}"], weights) for m in METRICS}
    ceiling = {
        scheme: {m: _mean(refs[f"oracle_{scheme}_{m}"], weights) for m in METRICS}
        for scheme in SCHEMES
    }
    return floor, ceiling


def _curve_point(
    scheme: str,
    budget: int | str,
    draws: pd.DataFrame,
    floor: dict,
    ceiling: dict,
    weights: pd.Series | None,
) -> dict:
    point = {"scheme": scheme, "budget": budget, "n_samples": len(draws)}
    for column in ("lambda", "lambda_pos"):
        point[f"mean_{column}"] = _mean(draws[column], weights)
        point[f"std_{column}"] = _std(draws[column])
    for m in METRICS:
        held = draws[f"held_{m}"]
        point[f"mean_{m}"] = _mean(held, weights)
        point[f"std_{m}"] = _std(held)
        point[f"frac_of_ceiling_{m}"] = _fraction_recovered(
            point[f"mean_{m}"], floor[m], ceiling[m]
        )
    return point


def _learning_curve(samples: pd.DataFrame, floor: dict, ceiling: dict) -> pd.DataFrame:
    points = []
    for scheme in SCHEMES:
        mine = samples[samples["scheme"] == scheme]
        sampled, full = mine[mine["kind"] == "sampled"], mine[mine["kind"] == "full"]
        for budget, draws in sampled.groupby("budget"):
            points.append(
                _curve_point(
                    scheme, int(budget), draws, floor, ceiling[scheme], weights=None
                )
            )
        points.append(
            _curve_point(
                scheme, "full", full, floor, ceiling[scheme], weights=full["n_report"]
            )
        )
    return pd.DataFrame(points)


def _first_budget_reaching(
    points: pd.DataFrame, metric: str, threshold: float
) -> int | None:
    reached = points.loc[points[f"frac_of_ceiling_{metric}"] >= threshold, "budget"]
    return int(reached.iloc[0]) if len(reached) else None


def _budget_needed(curve: pd.DataFrame) -> dict:
    sampled = curve[curve["budget"] != "full"]
    needed = {}
    for scheme in SCHEMES:
        points = (
            sampled[sampled["scheme"] == scheme]
            .astype({"budget": int})
            .sort_values("budget")
        )
        needed[scheme] = {
            m: {f"{t:.0%}": _first_budget_reaching(points, m, t) for t in THRESHOLDS}
            for m in METRICS
        }
    return needed


def _paired_win_rate(samples: pd.DataFrame) -> pd.DataFrame:
    """Share of identical draws on which multiply >= subtract."""
    draws = samples[samples["kind"] == "sampled"]
    if draws.empty:
        return pd.DataFrame()
    per_metric = []
    for m in METRICS:
        paired = draws.pivot_table(
            index=["fold", "budget", "rep"], columns="scheme", values=f"held_{m}"
        )
        per_metric.append(
            pd.DataFrame(
                {
                    f"multiply_win_rate_{m}": (
                        paired["multiply"] >= paired["subtract"]
                    ).astype(float),
                    f"mean_diff_{m}": paired["multiply"] - paired["subtract"],
                }
            )
            .groupby(level="budget")
            .mean()
        )
    table = pd.concat(per_metric, axis=1)
    table["n"] = draws.groupby("budget").size() / len(SCHEMES)
    return table.reset_index()


def _ceiling_gap(
    gold: np.ndarray,
    splits: Sequence[Split],
    oracles: list[dict[str, Ranking]],
    seed: int = 0,
) -> dict:
    """Do the two schemes' oracle ceilings actually differ? Paired
    entry-cluster bootstrap + exact McNemar on P@1 and P@5."""
    clusters = gold[np.concatenate([report for _, _, report in splits])]
    gap = {}
    for m, outcome in PER_QUERY_OUTCOME.items():
        subtract = np.concatenate([getattr(o["subtract"], outcome) for o in oracles])
        multiply = np.concatenate([getattr(o["multiply"], outcome) for o in oracles])
        diff, lo, hi = clustered_bootstrap(multiply, subtract, clusters, seed=seed)
        multiply_only, subtract_only = (
            int((multiply & ~subtract).sum()),
            int((subtract & ~multiply).sum()),
        )
        gap[m] = {
            "multiply_minus_subtract": diff,
            "ci": [lo, hi],
            "multiply_only_correct": multiply_only,
            "subtract_only_correct": subtract_only,
            "mcnemar_p": mcnemar_exact_p(multiply_only, subtract_only),
            "n": len(clusters)
        }
    return gap
