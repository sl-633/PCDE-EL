"""From name hits to entry rankings.

Retrievers return names; evaluation and calibration work on entries. An
entry's score is the max over its retrieved names, kept separately for
original (s_o) and generated (s_g) names so calibration can reweight the
generated side later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcde_el.data.schema import is_generated
from pcde_el.normalization import normalize

# Stands in for "no name of this provenance was retrieved". A finite value
# keeps the arithmetic in calibration well defined.
NEG_INF = -1e30

def name_table(dictionary: pd.DataFrame, level: int) -> pd.DataFrame:
    """One row per indexable name. Names that normalize to "" are dropped, and
    name_row is the row's position, i.e. the id the retrievers return."""
    named_t = dictionary[["name", "pid", "category"]].reset_index(drop=True).copy()
    named_t["pid"] = named_t["pid"].astype(str)
    named_t["text"] = named_t["name"].map(lambda x: normalize(x, level=level))
    named_t = named_t[named_t["text"].str.len() > 0].reset_index(drop=True)
    named_t["is_generated"] = named_t["category"].map(is_generated)
    named_t["name_row"] = np.arange(len(named_t))
    
    return named_t


def aggregate_entries(flat: pd.DataFrame, names: pd.DataFrame, pid_depth: int=50, tie_break: str="pid") -> pd.DataFrame:
    """Turn (qid, name_row, score, name_rank) hits into the top `pid_depth`
    entries per query.

    Besides score, s_o and s_g, each row records the best original and best
    generated name, which side won, and pid_rank (1-based).

    tie_break: "pid" sorts tied entries by id. "name_rank" first uses the
    retriever's rank of each entry's best name, which reproduces Lucene's
    order for BM25.
    """
    pid_list = names["pid"].to_numpy()
    gen_list = names["is_generated"].to_numpy()
    flat_t = flat[["qid", "name_row", "score", "name_rank"]].copy()
    rows = flat_t["name_row"].to_numpy()
    flat_t["pid"] = pid_list[rows]
    flat_t["is_generated"] = gen_list[rows]
    flat_t["_ori"] = np.where(~flat_t["is_generated"], flat_t["score"], NEG_INF)
    flat_t["_gen"] = np.where(flat_t["is_generated"], flat_t["score"], NEG_INF)

    grouped = flat_t.groupby(["qid", "pid"], sort=False)

    output = grouped.agg(
        score=("score", "max"),
        s_o=("_ori", "max"),
        s_g=("_gen", "max"),
        best_name_rank=("name_rank", "min"),
        n_names_hit=("score", "size"),
    ).reset_index()

    # Best name per entry and side: highest score, then the retriever's rank.
    sorted_t = flat_t.sort_values(["score", "name_rank"], ascending=[False, True])

    name_list = names["name"].to_numpy()
    category_list = names["category"].to_numpy()

    for mask, side in ((~sorted_t["is_generated"], "orig"), (sorted_t["is_generated"], "gen")):
        best_name = sorted_t[mask].drop_duplicates(subset=["qid", "pid"], keep="first")[["qid", "pid", "name_row"]]
        best_name[f"best_{side}_name"] = name_list[best_name["name_row"].to_numpy()]
        best_name[f"best_{side}_category"] = category_list[best_name["name_row"].to_numpy()]

        output = output.merge(best_name.drop(columns=["name_row"]), on=["qid", "pid"], how="left")
    output["winning_source"] = np.where(output["s_g"] > output["s_o"], "generated", "original")

    if tie_break == "pid":
        key, ascending = ["qid", "score", "pid"], [True, False, True]
    elif tie_break == "name_rank":
        key, ascending = ["qid", "score", "best_name_rank", "pid"], [True, False, True, True]
    else:
        raise ValueError(f"Invalid tie_break value: {tie_break}. Must be 'pid' or 'name_rank'.")

    output = output.sort_values(key, ascending=ascending)
    output["pid_rank"] = output.groupby("qid", sort=False).cumcount() + 1
    output = output[output["pid_rank"] <= pid_depth].reset_index(drop=True)

    return output


def merge_subqueries(flat: pd.DataFrame, parent_qid: np.ndarray) -> pd.DataFrame:
    """Merge hits of several sub-queries (e.g. a mention and its LLM-expanded
    forms) into their parent query, keeping each name's best score and rank."""
    flat_t = flat.assign(qid=parent_qid[flat["qid"].to_numpy()])
    grouped = flat_t.groupby(["qid", "name_row"], sort=False)
    output = grouped.agg(
        score=("score", "max"),
        name_rank=("name_rank", "min"),
    )
    return output.reset_index()


def candidates(entries: pd.DataFrame) -> pd.DataFrame:
    """The columns calibration needs, with missing scores set to NEG_INF."""
    cand = entries[["qid", "pid", "s_o", "s_g"]].copy()
    cand["pid"] = cand["pid"].astype(str)
    for c in ["s_o", "s_g"]:
        cand[c] = cand[c].replace([-np.inf, np.inf], NEG_INF)
        cand[c] = cand[c].fillna(NEG_INF)
    return cand.reset_index(drop=True)
