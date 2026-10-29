"""BM25 over individual names with Pyserini (Lucene).

Each name is one document. Stemming and stopword removal are off: names are
short and often technical, so both do more harm than good.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

K1, B = 0.9, 0.4  # Pyserini defaults


def build_index(texts: list[str], work_dir: Path, threads: int = 8) -> Path:
    """Index texts[i] as document "n{i}", so a hit's docid maps back to row i."""
    corpus_dir, index_dir = work_dir / "corpus", work_dir / "index"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    with (corpus_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, text in enumerate(texts):
            fh.write(
                json.dumps({"id": f"n{i}", "contents": text}, ensure_ascii=False) + "\n"
            )

    if index_dir.exists():
        shutil.rmtree(index_dir)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(corpus_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(threads),
            "--storeRaw",
            "--stemmer",
            "none",
            "--keepStopwords",
        ],
        check=True,
    )

    return index_dir


def search(
    index_dir: Path,
    queries: list[str],
    name_depth: int = 1000,
    k1: float = K1,
    b: float = B,
    threads: int = 8,
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Returns one row per (qid, name_row) hit with score and 0-based rank.
    Duplicate query strings are searched once; empty queries get no hits."""

    from pyserini.analysis import get_lucene_analyzer
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
    searcher.set_bm25(k1, b)

    rows_by_text: dict[str, list[int]] = {}

    for qid, q in enumerate(queries):
        if q.strip():
            rows_by_text.setdefault(q, []).append(qid)
    uniq = list(rows_by_text)

    out = []
    for start in tqdm(
        range(0, len(uniq), batch_size),
        desc="bm25",
        leave=False,
        disable=len(uniq) <= batch_size,
    ):
        chunk = uniq[start : start + batch_size]
        hits = searcher.batch_search(
            chunk, [str(i) for i in range(len(chunk))], k=name_depth, threads=threads
        )
        for local_id, hs in hits.items():
            scored = [
                (int(h.docid[1:]), float(h.score), rank) for rank, h in enumerate(hs)
            ]
            for qid in rows_by_text[chunk[int(local_id)]]:
                out.extend((qid, row, score, rank) for row, score, rank in scored)

    return pd.DataFrame(out, columns=["qid", "name_row", "score", "name_rank"])
