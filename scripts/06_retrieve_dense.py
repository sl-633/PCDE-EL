"""Dense retrieval over D0 and D1 with each bi-encoder.

Model tags:
    vanilla   the dataset's base model, not fine-tuned
    ft_d0     fine-tuned on D0
    ft_gen    fine-tuned on the D1 condition

Results go to results/<dataset>/<condition>/dense/.

    python scripts/06_retrieve_dense.py --dataset ncbi_disease --condition gpt-d1
    python scripts/06_retrieve_dense.py --dataset ncbi_disease --condition gpt-d1 \\
        --models ft_gen --model-path ft_gen=/path/to/checkpoint
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from pcde_el import paths
from pcde_el.data.schema import load_dict, load_test
from pcde_el.evaluation.metrics import per_query_outcome
from pcde_el.normalization import normalize
from pcde_el.retrieval import dense, store
from pcde_el.retrieval.aggregation import NEG_INF, aggregate_entries, name_table
from pcde_el.retrieval.encoder import best_device, encode, load_encoder

LEVEL = 0  # must match training (training/pairs.py)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--models", nargs="+", default=["vanilla", "ft_d0", "ft_gen"])
    ap.add_argument("--model-path", nargs="*", default=[], help="tag=path overrides")
    ap.add_argument("--name-depth", type=int, default=1000)
    ap.add_argument("--pid-depth", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=32)
    args = ap.parse_args()

    overrides = dict(spec.split("=", 1) for spec in args.model_path)
    models = {
        t: overrides.get(t) or paths.resolve_model(args.dataset, args.condition, t)
        for t in args.models
    }
    device = best_device()

    test = load_test(paths.test_csv(args.dataset))
    gold = test["gold_pid"].astype(str).to_numpy()
    queries = [normalize(m, LEVEL) for m in test["mention"]]
    dicts = {
        "D0": paths.d0_csv(args.dataset),
        "D1": paths.d1_csv(args.dataset, args.condition),
    }
    tables = {tag: name_table(load_dict(p), level=LEVEL) for tag, p in dicts.items()}

    all_hits = []
    for mtag, mpath in models.items():
        print(f"[{mtag}] {mpath}")
        tokenizer, model = load_encoder(mpath, device)
        # Queries are encoded once per model and searched against both dictionaries.
        q_emb = encode(
            queries,
            model,
            tokenizer,
            device,
            args.batch_size,
            args.max_length,
            f"{mtag}:queries",
        )
        for dtag, names in tables.items():
            d_emb = encode(
                names["text"].tolist(),
                model,
                tokenizer,
                device,
                args.batch_size,
                args.max_length,
                f"{mtag}:{dtag}",
            )
            idx, score = dense.search(q_emb, d_emb, args.name_depth, device)
            hits = aggregate_entries(
                dense.to_flat(idx, score), names, args.pid_depth, tie_break="pid"
            )
            hits.insert(0, "dict", dtag)
            hits.insert(0, "model", mtag)
            all_hits.append(hits)
            # s_o is NEG_INF when none of the entry's original names was retrieved.
            only_gen = (hits["s_o"] <= NEG_INF / 2).mean()
            print(
                f"  [{mtag}/{dtag}] {len(hits):,} entry rows, "
                f"{only_gen:.1%} reachable only via generated names"
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    hits = pd.concat(all_hits, ignore_index=True)
    outcome = per_query_outcome(hits, gold, group_cols=("model", "dict"))
    store.save(
        args.dataset,
        args.condition,
        "dense",
        hits,
        test,
        outcome,
        {
            "models": models,
            "normalization_level": LEVEL,
            "name_depth": args.name_depth,
            "pid_depth": args.pid_depth,
            "max_length": args.max_length,
            "pooling": "mean",
            "dicts": {k: str(v) for k, v in dicts.items()},
            "tie_break": "pid",
        },
    )


if __name__ == "__main__":
    main()
