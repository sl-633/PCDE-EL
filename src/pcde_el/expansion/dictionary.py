"""Dictionary expansion: prompt the LLM with each D0 entry, parse the
generated names, and merge them into D1 with a generated_ category."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

from pcde_el.data.schema import GENERATED_PREFIX
from pcde_el.expansion.llm_client import answer_part, clean_names
from pcde_el.paths import prompt_file

# Order in which an entry's existing names are shown in the prompt.
CATEGORY_DISPLAY_ORDER = [
    "record_preferred_term",
    "preferred_label",
    "synonym",
    "other_name",
    "abbreviation",
    "representative_name",
    "source_based_name",
    "structure_based_name",
]

Entries = OrderedDict[str, OrderedDict[str, list[str]]]


def load_entries(d0_csv: Path) -> Entries:
    """pid -> category -> names, in file order."""
    entries: OrderedDict = OrderedDict()
    seen: dict[str, set[str]] = defaultdict(set)
    with open(d0_csv, encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            name, pid = (row.get("name") or "").strip(), (row.get("pid") or "").strip()
            category = (row.get("category") or "").strip() or "unknown"
            if not name or not pid:
                raise ValueError(f"Empty name or pid at {d0_csv}:{line_no}")
            if name.casefold() in seen[pid]:
                continue
            seen[pid].add(name.casefold())
            entries.setdefault(pid, OrderedDict()).setdefault(category, []).append(name)
    return entries


def format_entry(pid: str, by_category) -> str:
    """The entry as shown to the LLM: its id, then one JSON list per category."""
    ranked = [c for c in CATEGORY_DISPLAY_ORDER if c in by_category]
    ordered = ranked + [c for c in by_category if c not in ranked]
    lines = [f"concept_id: {pid}"]

    lines += [f"{c}: {json.dumps(by_category[c], ensure_ascii=False)}" for c in ordered]
    return "\n".join(lines)


def build_prompt(dataset: str, pid: str, by_category) -> str:
    template = prompt_file("dictionary", dataset).read_text(encoding="utf-8").strip()
    # Zero-shot: the few-shot slot is left empty.
    return (
        template.replace("{{FEW_SHOT_TEXT}}", "")
        .replace("{{TARGET_ENTITY}}", format_entry(pid, by_category))
        .strip()
    )


def parse_response(
    raw: str, categories: tuple[str, ...]
) -> tuple[dict[str, list[str]], bool]:
    """Returns ({category: names}, parsed_ok).

    Tries the whole answer as JSON first, then the first {...} span, since
    models sometimes wrap the object in prose. Categories the model did not
    return come back empty.
    """
    text = answer_part(raw)
    start, end = text.find("{"), text.find("}")
    braced = text[start: end + 1] if start != -1 and end > start else ""
    for candidate in (text, braced):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            return {c: clean_names(obj.get(c, [])) for c in categories}, True
    return {c: [] for c in categories}, False


def generate_for_entry(
    generator, dataset: str, categories: tuple[str, ...], pid: str, by_category
) -> list:
    """One cache record; the raw response is kept so D1 can be rebuilt offline."""
    raw, latency = generator(build_prompt(dataset, pid, by_category))
    names, is_ok = parse_response(raw, categories)
    return {
        "concept_id": pid,
        "input_names": {c: list(v) for c, v in by_category.items()},
        "generated_names": names,
        "parse_success": is_ok,
        "raw_response": raw,
        "time": f"{latency:.2f}",
    }


def build_d1(
    d0_csv: Path, results: dict, categories: tuple[str, ...], out_csv: Path
) -> dict:
    """Write D1 = D0 + generated names, and return summary counts.

    A generated name that the entry already has (case-insensitively) is
    dropped. The same string may still be generated for different entries;
    those collisions are counted, not removed.
    """
    rows: list[tuple[str, str, str]] = []
    existing: dict[str, set[str]] = defaultdict(set)
    with open(d0_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name, pid = (row.get("name") or "").strip(), (row.get("pid") or "").strip()
            if name and pid:
                rows.append(
                    (name, pid, (row.get("category") or "").strip() or "unknown")
                )
                existing[pid].add(name.casefold())
    num_d0 = len(rows)

    stats: dict = {
        "d0_rows": num_d0,
        "dropped_duplicate_in_entry": 0,
        "parse_failures": 0,
        "entries_with_generation": 0,
        "per_type": defaultdict(int),
    }

    for pid, record in results.items():
        stats["parse_failures"] += not record.get("parse_success")
        added = 0
        for name_type in categories:
            for name in record.get("generated_names", {}).get(name_type, []):
                if name.casefold() in existing[pid]:
                    stats["dropped_duplicate_in_entry"] += 1
                    continue
                existing[pid].add(name.casefold())
                rows.append((name, pid, f"{GENERATED_PREFIX}{name_type}"))
                stats["per_type"][name_type] += 1
                added += 1
        stats["entries_with_generation"] += added > 0
    stats["generated_rows"] = len(rows) - num_d0

    owners: dict[str, set[str]] = defaultdict(set)

    for name, pid, _ in rows:
        owners[name.casefold()].add(pid)
    stats["strings_shared_across_entries"] = sum(len(v) > 1 for v in owners.values())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "pid", "category"])
        writer.writerows(rows)
    stats["per_type"] = dict(stats["per_type"])
    return stats
