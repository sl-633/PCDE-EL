"""Turn mention-level annotations (NCBI-disease, BC5CDR) into a context-free
test set of (mention, gold id) pairs.

Per split, an annotation is dropped if it has no id, an id known to be
unmappable, or still more than one id after legacy ids are updated, or if
its id is not in the dictionary. The rest are deduplicated on
(casefolded mention, gold id). Deduplication is per split, so the same pair
can appear in train and test.
"""

import csv
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

BARE_MESH_ID = re.compile(r"^[CD]\d+$", flags=re.IGNORECASE)
PREFIXED_MESH_ID = re.compile(r"^MESH:[CD]\d+$", flags=re.IGNORECASE)
OMIM_ID = re.compile(r"^OMIM:\d+$", flags=re.IGNORECASE)
MULTI_ID_SEPARATOR = re.compile(r"[|+]")

SPLITS = ("train", "dev", "test")


def normalize_space(value: str | None) -> str:
    return " ".join(str(value).split()) if value else ""


def dedup_key(mention: str) -> str:
    return normalize_space(mention).casefold()


def normalize_identifier(value: str | None) -> str:
    """D008107 -> MESH:D008107; ids that already have a prefix are only upper-cased."""
    value = normalize_space(value).upper()
    if BARE_MESH_ID.fullmatch(value):
        return f"MESH:{value}"
    return value


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def split_concept_ids(raw: str, missing: frozenset[str] = frozenset({""})) -> list[str]:
    """'D003110|D008228' -> ['MESH:D003110', 'MESH:D008228'].

    `missing` holds placeholder values (BC5CDR uses "-1") that mean "no id".
    """
    raw = normalize_space(raw)
    if raw in missing:
        return []
    ids = []
    for piece in MULTI_ID_SEPARATOR.split(raw):
        piece = normalize_space(piece)
        if piece not in missing:
            ids.append(normalize_identifier(piece))
    return unique_preserving_order(ids)


def load_dictionary_ids(
    dictionary_csv: Path, equivalent_ids_csv: Path | None = None
) -> set[str]:
    """Every id that counts as covered by the dictionary, including alternate ids."""
    ids: set[str] = set()
    with open(dictionary_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ids.add(normalize_identifier(row["pid"]))
    if equivalent_ids_csv is not None:
        with open(equivalent_ids_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ids.add(normalize_identifier(row["equivalent_id"]))
    return ids


def iter_pubtator_entities(path: Path) -> Iterator[tuple[str, str, str, int]]:
    """Yield (mention, entity type, raw id field, line number) per annotation line."""
    with open(path, encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            # Skip blank lines and title/abstract lines.
            if not line.strip() or "|t|" in line or "|a|" in line:
                continue
            fields = line.split("\t")
            # BC5CDR chemical-induced-disease relation lines.
            if len(fields) == 4 and fields[1].strip().upper() == "CID":
                continue
            if len(fields) not in (6, 7):
                raise ValueError(
                    f"{path}:{line_no}: unexpected {len(fields)} fields: {line!r}"
                )
            mention, entity_type, raw_ids = fields[3], fields[4], fields[5]
            if not normalize_space(mention):
                raise ValueError(f"{path}:{line_no}: empty mention")
            yield (
                normalize_space(mention),
                normalize_space(entity_type),
                raw_ids,
                line_no,
            )


def filter_and_dedup(
    annotations: Iterable[tuple[str, list[str]]],
    *,
    valid_ids: set[str],
    legacy_to_current: dict[str, str],
    excluded_ids: dict[str, str],
) -> tuple[list[tuple[str, str]], Counter]:
    """Apply the filters in the module docstring to one split.

    Returns (mention, gold id) rows sorted for a stable output, and counts
    of what was dropped and why.
    """
    stats: Counter = Counter()
    kept: dict[tuple[str, str], str] = {}
    for mention, ids in annotations:
        stats["input"] += 1
        if not ids:
            stats["no_id"] += 1
            continue
        if any(i in excluded_ids for i in ids):
            stats["unmappable"] += 1
            continue
        mapped = unique_preserving_order(
            normalize_identifier(legacy_to_current.get(i, i)) for i in ids
        )
        if len(mapped) != 1:
            stats["multi_concept"] += 1
            continue
        gold = mapped[0]
        if gold not in valid_ids:
            stats["not_in_dictionary"] += 1
            continue
        key = (dedup_key(mention), gold)
        # Keep the first surface form seen.
        if key in kept:
            stats["duplicate"] += 1
            continue
        kept[key] = mention
    rows = sorted(
        ((m, gold) for (_, gold), m in kept.items()),
        key=lambda r: (dedup_key(r[0]), r[1]),
    )
    stats["output"] = len(rows)
    return rows, stats
