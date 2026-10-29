"""Run retrieval, training and calibration for one (dataset, condition).

LLM generation (02/03) is not included: it needs an Ollama server and is
cached, so run it once by hand. Training is skipped for checkpoints that
already exist, so the D0 model is shared across conditions.

    python scripts/run_pipeline.py --dataset ncbi_disease --condition gpt-d1
    python scripts/run_pipeline.py --dataset ncbi_disease --condition gpt-d1 --stages bm25,calibrate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGES = ["bm25", "train_d0", "train_gen", "dense", "calibrate"]


def commands(ds: str, cond: str, py: str, smoke: bool) -> dict[str, list[list[str]]]:
    run = lambda script, *a: [py, str(SCRIPTS / script), "--dataset", ds, *a]  # noqa: E731
    steps = ["--steps", "100"] if smoke else []
    return {
        "bm25": [run("05_retrieve_bm25.py", "--condition", cond)],
        "train_d0": [run("04_train_biencoder.py", "--dict", "d0", *steps)],
        "train_gen": [run("04_train_biencoder.py", "--dict", cond, *steps)],
        "dense": [run("06_retrieve_dense.py", "--condition", cond)],
        "calibrate": [
            run("07_calibrate.py", "--condition", cond, "--source", "bm25"),
            run(
                "07_calibrate.py",
                "--condition",
                cond,
                "--source",
                "dense",
                "--model",
                "vanilla",
            ),
            run(
                "07_calibrate.py",
                "--condition",
                cond,
                "--source",
                "dense",
                "--model",
                "ft_gen",
            ),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--smoke", action="store_true", help="train for 100 steps only")
    ap.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; known: {STAGES}")

    cmds = commands(args.dataset, args.condition, sys.executable, args.smoke)
    for stage in stages:
        for cmd in cmds[stage]:
            print(f"\n=== [{stage}] {' '.join(cmd[1:])}", flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
