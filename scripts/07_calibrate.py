"""Provenance calibration with out-of-fold lambda selection.

Reads the stored retrieval results and writes
results/<dataset>/<condition>/calibration/<retriever>_<scheme>.{json,csv}.
The protocol defaults to the dataset's config: the official split for the
biomedical datasets, 5-fold CV over entries for PoLyMer.

    python scripts/07_calibrate.py --dataset ncbi_disease --condition gpt-d1 --source bm25
    python scripts/07_calibrate.py --dataset ncbi_disease --condition gpt-d1 --source dense --model ft_gen
"""

from __future__ import annotations

import argparse
import json

from pcde_el import paths
from pcde_el.calibration.schemes import SCHEMES
from pcde_el.calibration.selection import calibrate
from pcde_el.config import get_dataset
from pcde_el.retrieval import store


def retriever_label(source: str, model: str | None) -> str:
    return "bm25" if source == "bm25" else f"dense-{model}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--source", choices=["bm25", "dense"], required=True)
    ap.add_argument(
        "--model", default="ft_gen", help="dense model tag (vanilla/ft_d0/ft_gen)"
    )
    ap.add_argument("--scheme", choices=[*SCHEMES, "both"], default="both")
    ap.add_argument("--protocol", choices=["split", "cv"], default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-lambda", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    protocol = args.protocol or get_dataset(args.dataset).protocol
    model = args.model if args.source == "dense" else None
    cand, test, _ = store.load_candidates(
        args.dataset, args.condition, args.source, model
    )
    out_dir = paths.calibration_dir(args.dataset, args.condition)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scheme in SCHEMES if args.scheme == "both" else [args.scheme]:
        summary, per_query = calibrate(
            cand,
            test,
            scheme,
            protocol,
            folds=args.folds,
            n_lambda=args.n_lambda,
            seed=args.seed,
        )
        stem = out_dir / f"{retriever_label(args.source, model)}_{scheme}"
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        per_query.to_csv(stem.with_suffix(".csv"), index=False)

        m = summary["metrics"]
        lams = ", ".join(f"{c['lambda']:.3f}" for c in summary["chosen"])
        print(
            f"[{scheme}] protocol={protocol} n={summary['n_reported']} lambda=[{lams}]"
        )
        for name in ("uncalibrated", "calibrated"):
            print(
                f"  {name:13s} P@1 {m[name]['P@1']:.4f}  P@5 {m[name]['P@5']:.4f}  "
                f"Res@5 {m[name]['Resolution@5']:.4f}  Cov {m[name]['Coverage']:.4f}"
            )
        t = summary["vs_uncalibrated"]["P@5"]
        print(
            f"  P@5 delta {t['delta']:+.4f}  95% CI [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}]"
            f"{'  *' if t['significant'] else ''}"
        )
        print(f"  wrote {stem}.json/.csv")


if __name__ == "__main__":
    main()
