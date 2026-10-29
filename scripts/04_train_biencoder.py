"""Fine-tune a bi-encoder on the names of D0 or of a D1 condition.

The base model and number of steps default to the dataset's config. The
checkpoint goes to models/<dataset>/<dict>/; an existing one is kept unless
--force is given.

    python scripts/04_train_biencoder.py --dataset ncbi_disease --dict d0
    python scripts/04_train_biencoder.py --dataset ncbi_disease --dict gpt-d1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcde_el import paths
from pcde_el.config import get_dataset
from pcde_el.data.schema import load_dict
from pcde_el.training.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dict", required=True, help="'d0' or a condition such as gpt-d1")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None, help="override models/<dataset>/<dict>")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    dict_csv = (
        paths.d0_csv(args.dataset)
        if args.dict == "d0"
        else paths.d1_csv(args.dataset, args.dict)
    )
    out = (
        paths.model_dir(args.dataset, args.dict)
        if args.output is None
        else Path(args.output)
    )
    if (out / "training_config.json").exists() and not args.force:
        print(f"{out} is already trained; pass --force to retrain")
        return

    cfg = TrainConfig(
        base_model=args.base_model or ds.base_model,
        steps=args.steps or ds.train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train(load_dict(dict_csv), cfg, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
