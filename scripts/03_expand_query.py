"""Query expansion baseline: ask an LLM for the formal name of each test mention.

Only used for the comparison in Appendix B. Generations are cached and
resumable, and the formal names are written to data/<dataset>/generation/.

    python scripts/03_expand_query.py --dataset ncbi_disease
"""

from __future__ import annotations

import argparse

from pcde_el import paths
from pcde_el.data.schema import load_test
from pcde_el.expansion import query
from pcde_el.expansion.llm_client import OllamaGenerator, load_json_or, run_resumable


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    mentions = query.unique_mentions(load_test(paths.test_csv(args.dataset)))
    json_path = paths.query_generation_json(args.dataset, args.model)
    output = load_json_or(
        json_path,
        {"metadata": {"dataset": args.dataset, "model": args.model}, "results": {}},
    )
    print(
        f"{len(mentions):,} unique mentions, {len(output['results']):,} already generated"
    )

    if not args.build_only:
        generator = OllamaGenerator(args.model, args.host)
        run_resumable(
            mentions,
            lambda m: query.generate_for_mention(generator, args.dataset, m),
            output,
            json_path,
            limit=args.limit,
        )

    out_csv = paths.formal_names_csv(args.dataset, args.model)
    print(f"wrote {out_csv}: {query.write_formal_names(output['results'], out_csv)}")


if __name__ == "__main__":
    main()
