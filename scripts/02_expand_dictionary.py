"""Generate aliases for every D0 entry with an LLM and build D1.

Generations are cached in data/<dataset>/generation/ and the run can be
interrupted and resumed. D1 is written to data/<dataset>/processed/<condition>.csv.
Requires a running Ollama server.

    python scripts/02_expand_dictionary.py --dataset ncbi_disease --condition gpt-d1
    python scripts/02_expand_dictionary.py --dataset ncbi_disease --build-only   # rebuild D1 from the cache
"""

from __future__ import annotations

import argparse

from pcde_el import paths
from pcde_el.config import get_dataset
from pcde_el.expansion import dictionary
from pcde_el.expansion.llm_client import OllamaGenerator, load_json_or, run_resumable


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--condition", default="gpt-d1", help="name of the resulting D1")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument(
        "--think",
        default=None,
        help="override reasoning setting (default: 'low' for gpt-oss, off otherwise)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="stop after N new entries (dry run)"
    )
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    cfg = get_dataset(args.dataset)
    d0_csv = paths.d0_csv(args.dataset)
    json_path = paths.dictionary_generation_json(args.dataset, args.model)
    entries = dictionary.load_entries(d0_csv)
    output = load_json_or(
        json_path,
        {
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "categories": list(cfg.categories),
                "prompt": str(paths.prompt_file("dictionary", args.dataset).name),
                "setting": "zero_shot",
            },
            "results": {},
        },
    )
    print(f"{len(entries):,} entries, {len(output['results']):,} already generated")

    if not args.build_only:
        # --think accepts true/false or a level such as "low".
        think = {"false": False, "true": True}.get(str(args.think).lower(), args.think)
        generator = OllamaGenerator(args.model, args.host, think)
        new = run_resumable(
            entries,
            lambda pid: dictionary.generate_for_entry(
                generator, args.dataset, cfg.categories, pid, entries[pid]
            ),
            output,
            json_path,
            limit=args.limit,
        )
        print(f"generated {new:,} new entries -> {json_path}")

    out_csv = paths.d1_csv(args.dataset, args.condition)
    stats = dictionary.build_d1(d0_csv, output["results"], cfg.categories, out_csv)
    print(f"wrote {out_csv}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
