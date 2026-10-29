"""BM25 retrieval over D0 and D1.

Names are indexed one document per name, the top names per query are
aggregated into entries, and the result is written to
results/<dataset>/<condition>/bm25/. Requires Pyserini and Java 21.

    python scripts/05_retrieve_bm25.py --dataset ncbi_disease --condition gpt-d1
"""

from __future__ import annotations

import argparse
import shutil

import pandas as pd
from pcde_el import paths
from pcde_el.data.schema import load_dict, load_test
from pcde_el.evaluation.metrics import per_query_outcome
from pcde_el.normalization import normalize
from pcde_el.retrieval import bm25, store
from pcde_el.retrieval.aggregation import aggregate_entries, name_table

LEVEL = 1  # lexical retrieval uses the punctuation-insensitive normalization


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--name-depth", type=int, default=1000)
    ap.add_argument("--pid-depth", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--keep-index", action="store_true", help="keep the Lucene build cache"
    )
    args = ap.parse_args()

    test = load_test(paths.test_csv(args.dataset))
    gold = test["gold_pid"].astype(str).to_numpy()
    queries = [normalize(m, LEVEL) for m in test["mention"]]
    work = paths.retrieval_dir(args.dataset, args.condition, "bm25") / "_index"
    dicts = {
        "D0": paths.d0_csv(args.dataset),
        "D1": paths.d1_csv(args.dataset, args.condition),
    }

    all_hits = []
    for tag, csv_path in dicts.items():
        names = name_table(load_dict(csv_path), level=LEVEL)
        print(f"[{tag}] indexing {len(names):,} names")
        index_dir = bm25.build_index(names["text"].tolist(), work / tag, args.threads)
        flat = bm25.search(index_dir, queries, args.name_depth, threads=args.threads)
        # BM25 scores tie often; fall back to Lucene's own name order.
        hits = aggregate_entries(flat, names, args.pid_depth, tie_break="name_rank")
        hits.insert(0, "dict", tag)
        hits.insert(0, "model", store.BM25_MODEL)
        all_hits.append(hits)
        print(f"[{tag}] {len(flat):,} name hits -> {len(hits):,} entry rows")

    hits = pd.concat(all_hits, ignore_index=True)
    outcome = per_query_outcome(hits, gold, group_cols=("model", "dict"))
    store.save(
        args.dataset,
        args.condition,
        "bm25",
        hits,
        test,
        outcome,
        {
            "normalization_level": LEVEL,
            "k1": bm25.K1,
            "b": bm25.B,
            "name_depth": args.name_depth,
            "pid_depth": args.pid_depth,
            "dicts": {k: str(v) for k, v in dicts.items()},
            "tie_break": "name_rank",
        },
    )
    if not args.keep_index:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
