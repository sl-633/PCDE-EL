"""String normalization shared by indexing, querying and training.

Level 0 (dense): NFKC, unified dashes, Greek letters spelled out, casefold,
    Roman numerals after type/class/stage/grade/group, collapsed whitespace.
Level 1 (BM25): level 0, then punctuation replaced by spaces.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Hashable, Iterable

_GREEK_LOWER = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}

_GREEK = {**_GREEK_LOWER, **{k.upper(): v for k, v in _GREEK_LOWER.items()}}
_GREEK["µ"] = "mu"

_DASHES = "‐‑‒–—―−﹘﹣－"
_DASH_TABLE = {ord(c): "-" for c in _DASHES}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\W_+]", re.UNICODE)

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
          "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}

# Only convert after these words: a bare "i" or "v" is usually not a numeral.
_ROMAN_RE = re.compile(r"\b(type|class|stage|grade|group)\s+([ivx]+)\b")

def _roman_to_arabic(s: str) -> str:
    return _ROMAN_RE.sub(lambda m: f"{m.group(1)} {_ROMAN.get(m.group(2), m.group(2))}", s)

def normalize(text: object, level: int=1) -> str:
    if level not in (0, 1):
        raise ValueError(f"level must be 0 or 1, but here is {level!r}")
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_DASH_TABLE)
    s = "".join(_GREEK.get(ch, ch) for ch in s)
    s = s.casefold()
    s = _roman_to_arabic(s)
    s = _WS_RE.sub(" ", s).strip()
    if level == 1:
        s = _PUNCT_RE.sub(" ", s)
        s = _WS_RE.sub(" ", s).strip()
    return s

def build_lexical_index(names: Iterable[str], pids: Iterable[Hashable], level: int=1) -> dict[str, list[Hashable]]:
    """Normalized name -> pids carrying it, in first-seen order."""
    index: dict[str, list[Hashable]] = {}
    for name, pid in zip(names, pids):
        key = normalize(name, level)
        if not key:
            continue
        bucket = index.setdefault(key, [])
        if pid not in bucket:
            bucket.append(pid)
    return index