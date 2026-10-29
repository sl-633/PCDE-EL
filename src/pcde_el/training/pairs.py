"""Training pairs: two names of the same entry form a positive pair.

Entries with many names would otherwise dominate training, and D1 entries
have more names than D0 ones. Capping the pairs per entry keeps the D0 and
D1 models comparable.
"""

from __future__ import annotations

import random
from itertools import permutations

import pandas as pd
from pcde_el.normalization import normalize

NORM_LEVEL = 0  # must match the level used at retrieval time
MAX_PAIRS_PER_ENTRY = 20


def names_by_entry(dictionary: pd.DataFrame) -> dict[str, list[str]]:
    """pid -> normalized names, without duplicates."""
    by_pid: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for name, pid in zip(dictionary["name"], dictionary["pid"]):
        s = normalize(name, NORM_LEVEL)
        if not s:
            continue
        pid = str(pid)
        keys = seen.setdefault(pid, set())
        if s.casefold() not in keys:
            keys.add(s.casefold())
            by_pid.setdefault(pid, []).append(s)
    return by_pid


def build_pairs(
    dictionary: pd.DataFrame,
    max_pairs_per_entry: int = MAX_PAIRS_PER_ENTRY,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """(pid, name_a, name_b) for every ordered pair of names, sampled down to
    `max_pairs_per_entry` per entry. Single-name entries contribute nothing."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str, str]] = []
    for pid, names in names_by_entry(dictionary).items():
        if len(names) < 2:
            continue
        candidates = list(permutations(names, 2))
        if len(candidates) > max_pairs_per_entry:
            candidates = rng.sample(candidates, max_pairs_per_entry)
        pairs.extend((pid, a, b) for a, b in candidates)

    rng.shuffle(pairs)
    return pairs
