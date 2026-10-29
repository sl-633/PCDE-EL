"""NCBI-disease: MEDIC dictionary (CTD_diseases.csv) and the NCBI disease corpus.

MEDIC gives each disease one preferred name and a flat synonym list, so D0
has two categories: record_preferred_term and synonym. AltDiseaseIDs are
written to a separate table and only used to decide coverage.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

from pcde_el.data.annotations import (
    SPLITS,
    filter_and_dedup,
    iter_pubtator_entities,
    load_dictionary_ids,
    normalize_space,
    split_concept_ids,
)

CTD_COLUMNS = [
    "DiseaseName",
    "DiseaseID",
    "AltDiseaseIDs",
    "Definition",
    "ParentIDs",
    "TreeNumbers",
    "ParentTreeNumbers",
    "Synonyms",
    "SlimMappings",
]

VALID_CONCEPT_ID = re.compile(r"^(?:MESH:[CD]\d+|OMIM:\d+)$", flags=re.IGNORECASE)

# Corpus ids whose MEDIC successor was verified by hand.
LEGACY_TO_CURRENT_ID = {
    "MESH:C537666": "MESH:D020388",
    "MESH:C538219": "MESH:D018223",
    "OMIM:102600": "MESH:C538228",
    "MESH:C531603": "MESH:D000347",
    "MESH:C535804": "MESH:D000080362",
    "MESH:C538554": "MESH:D007965",
    "MESH:C531610": "MESH:D019896",
    "OMIM:164160": "OMIM:614962",
    "MESH:C538392": "MESH:D015417",
}
# Corpus ids with no defensible successor; their annotations are dropped.
EXCLUDED_UNMAPPABLE_IDS = {
    "MESH:C531652": "No sufficiently reliable current MEDIC successor was identified.",
    "MESH:C538138": "Overlaps with multiple current glycerol-related concepts.",
    "OMIM:600698": "HMGA2/HMGIC gene entry; the mention describes gene deficiency, not a disease.",
    "MESH:D001405": "Refers to a clinical manifestation rather than a disease concept.",
    "OMIM:103600": "Only a broader current concept could be identified.",
}


def _canonical_id(value: str | None) -> str:
    value = normalize_space(value)
    if ":" not in value:
        return value
    prefix, ident = value.split(":", 1)
    return f"{prefix.upper()}:{ident}"


def _pipe_values(value: str | None) -> list[str]:
    return [v for item in (value or "").split("|") if (v := normalize_space(item))]


def _iter_ctd_rows(path: Path) -> Iterator[dict[str, str]]:
    """CTD releases ship with '#' comment lines and sometimes no header."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
    first = next(csv.reader([lines[0]]))
    has_header = (
        len(
            {v.strip().lstrip("#").casefold() for v in first}
            & {"diseasename", "diseaseid", "altdiseaseids", "synonyms"}
        )
        >= 2
    )
    if has_header:
        reader = csv.DictReader(
            lines[1:], fieldnames=[v.strip().lstrip("#") for v in first]
        )
    else:
        reader = csv.DictReader(lines, fieldnames=CTD_COLUMNS)
    for row in reader:
        yield {
            normalize_space(k): normalize_space(v)
            for k, v in row.items()
            if k is not None
        }


def build_d0(ctd_csv: Path) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Returns (name, pid, category) rows and (pid, alternate id) rows.

    Names are deduplicated case-insensitively within an entry; the first
    occurrence of a repeated DiseaseID wins.
    """
    entries: dict[str, tuple[list[str], list[str], bool]] = {}
    for row in _iter_ctd_rows(ctd_csv):
        cid = _canonical_id(row.get("DiseaseID"))
        if not cid or not VALID_CONCEPT_ID.fullmatch(cid) or cid in entries:
            continue
        preferred = normalize_space(row.get("DiseaseName"))
        labels, seen = [], set()
        for label in [preferred, *_pipe_values(row.get("Synonyms"))]:
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                labels.append(label)
        alt = [
            a
            for a in (_canonical_id(x) for x in _pipe_values(row.get("AltDiseaseIDs")))
            if VALID_CONCEPT_ID.fullmatch(a)
        ]
        entries[cid] = (labels, alt, bool(preferred))

    dict_rows, equiv_rows = [], []
    for cid in sorted(entries):
        labels, alt, has_preferred = entries[cid]
        for i, label in enumerate(labels):
            category = (
                "record_preferred_term" if (i == 0 and has_preferred) else "synonym"
            )
            dict_rows.append((label, cid, category))
        seen = {cid.upper()}
        for a in alt:
            if a.upper() not in seen:
                seen.add(a.upper())
                equiv_rows.append((cid, a))
    return dict_rows, equiv_rows


def build_test(
    d0_csv: Path, equivalent_ids_csv: Path | None, split_files: dict[str, Path]
) -> list[tuple[str, str, str]]:
    valid = load_dictionary_ids(d0_csv, equivalent_ids_csv)
    out = []
    for split in SPLITS:
        annos = [
            (m, split_concept_ids(raw))
            for m, etype, raw, _ in iter_pubtator_entities(split_files[split])
            # A composite mention names several diseases at once.
            if etype.casefold() != "compositemention"
        ]
        rows, stats = filter_and_dedup(
            annos,
            valid_ids=valid,
            legacy_to_current=LEGACY_TO_CURRENT_ID,
            excluded_ids=EXCLUDED_UNMAPPABLE_IDS,
        )
        print(f"  [{split}] {dict(stats)}")
        out.extend((m, gold, split) for m, gold in rows)
    return out
