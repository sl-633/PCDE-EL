"""File formats shared by every stage.

Dictionary CSV: name, pid, category. Rows added by the LLM have categories
starting with "generated_", which is how provenance is tracked downstream.
Test CSV: mention, gold_pid, plus optional columns such as list (split).
"""

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

GENERATED_PREFIX = "generated_"
DICT_COLUMNS = ["name", "pid", "category"]
TEST_COLUMNS = ["mention", "gold_pid"]


def is_generated(category: object) -> bool:
    return str(category).startswith(GENERATED_PREFIX)


def _require(df: pd.DataFrame, columns: Sequence[str], path: Path | str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def load_dict(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"pid": str})
    _require(df, DICT_COLUMNS, path)
    df["is_generated"] = df["category"].map(is_generated)
    return df


def load_test(path: Path | str) -> pd.DataFrame:
    # The row position is the query id (qid) used everywhere downstream.
    df = pd.read_csv(path, dtype={"gold_pid": str})
    _require(df, TEST_COLUMNS, path)
    return df.reset_index(drop=True)


def write_rows(
    path: Path | str, header: Sequence[str], rows: Iterable[Sequence]
) -> int:
    """Write a CSV with a header and return the number of data rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            n += 1
    return n
