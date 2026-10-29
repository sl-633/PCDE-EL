"""Query expansion baseline (Appendix B): ask the LLM for the formal names
of a test mention, to compare against expanding the dictionary instead."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pcde_el.expansion.llm_client import answer_part
from pcde_el.paths import prompt_file


def build_prompt(dataset: str, mention: str) -> str:
    template = prompt_file("query", dataset).read_text(encoding="utf-8").strip()
    return template.replace("{{QUERY}}", mention).strip()


def parse_response(raw: str) -> tuple[list[str], bool]:
    """Expects a JSON list. Otherwise falls back to one name per line and
    reports the parse as failed."""
    text = answer_part(raw)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            out, seen = [], set()
            for v in obj:
                s = " ".join(str(v).split())
                if s and s.casefold() not in seen:
                    seen.add(s.casefold())
                    out.append(s)
            return out, True
    except json.JSONDecodeError:
        pass
    return [ln.strip("-•\t ") for ln in text.splitlines() if ln.strip()], False


def generate_for_mention(generator, dataset: str, mention: str) -> dict:
    raw, latency = generator(build_prompt(dataset, mention))
    names, is_ok = parse_response(raw)
    return {
        "query": mention,
        "formal_names": names,
        "parse_success": is_ok,
        "raw_response": raw,
        "time": f"{latency:.2f}",
    }


def unique_mentions(test: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(str(m) for m in test["mention"]))


def write_formal_names(results: dict, out_csv: Path) -> dict:
    rows = [
        {
            "mention": m,
            "formal_names": json.dumps(r.get("formal_names", []), ensure_ascii=False),
        }
        for m, r in results.items()
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["mention", "formal_names"]).to_csv(out_csv, index=False)
    return {
        "mentions": len(rows),
        "empty": sum(not r.get("formal_names") for r in results.values()),
        "parse_failures": sum(not r.get("parse_success") for r in results.values()),
    }


def load_formal_names(csv_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    return dict(zip(df["mention"], df["formal_names"].map(json.loads)))
