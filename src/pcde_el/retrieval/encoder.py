"""Bi-encoder loading and encoding. Embeddings are mean-pooled and
L2-normalized, so a dot product is cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Average over real tokens only; padding is masked out.
    m =mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def load_encoder(path: str, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 512,
    max_length: int = 32,
    desc: str = "encode",
) -> torch.Tensor:
    """(len(texts), hidden) float32 tensor on the CPU."""
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            hidden = model(**enc).last_hidden_state
        out.append(
            F.normalize(mean_pooling(hidden, enc["attention_mask"]), dim=1)
            .float()
            .cpu()
        )
    return torch.cat(out) if out else torch.empty(0)
