"""Exact nearest-name search over normalized embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def search(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    name_depth: int,
    device: str,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-`name_depth` names per query, as (name indices, scores), each of
    shape (n_queries, k). Queries are processed in chunks to bound memory."""
    k = min(name_depth, d_emb.shape[0])
    d_dev = d_emb.to(device)
    idx_all, score_all = [], []
    for i in range(0, q_emb.shape[0], chunk):
        sims = q_emb[i : i + chunk].to(device) @ d_dev.T
        s, j = torch.topk(sims, k=k, dim=1)
        idx_all.append(j.cpu().numpy())
        score_all.append(s.cpu().numpy())
    del d_dev
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(idx_all), np.concatenate(score_all)


def to_flat(name_idx: np.ndarray, name_score: np.ndarray) -> pd.DataFrame:
    """One row per (query, retrieved name), the input format of aggregate_entries."""
    n_q, k = name_idx.shape
    return pd.DataFrame(
        {
            "qid": np.repeat(np.arange(n_q), k),
            "name_row": name_idx.ravel(),
            "score": name_score.ravel().astype(np.float64),
            "name_rank": np.tile(np.arange(k), n_q),
        }
    )
