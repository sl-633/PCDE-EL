from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcde_el.retrieval.encoder import best_device, mean_pooling
from pcde_el.training.loss import multi_positive_infonce
from pcde_el.training.pairs import MAX_PAIRS_PER_ENTRY, NORM_LEVEL, build_pairs
from tqdm import tqdm


@dataclass
class TrainConfig:
    base_model: str
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    tau: float = 0.07
    max_length: int = 32
    max_pairs_per_entry: int = MAX_PAIRS_PER_ENTRY
    fp16: bool = True
    seed: int = 42
    log_every: int = 100


class SharedBiEncoder(nn.Module):
    """One encoder for both sides; logits are cosine similarities / tau."""

    def __init__(self, model_name: str, tau: float = 0.07):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tau = tau

    def encode(self, inputs: dict) -> torch.Tensor:
        out = self.encoder(**inputs)
        return F.normalize(
            mean_pooling(out.last_hidden_state, inputs["attention_mask"]), dim=1
        )

    def forward(self, a_inputs: dict, b_inputs: dict) -> torch.Tensor:
        return (self.encode(a_inputs) @ self.encode(b_inputs).T) / self.tau


def train(dictionary: pd.DataFrame, cfg: TrainConfig, output_dir: Path) -> Path:
    """Fine-tune on name pairs from `dictionary` and save a Hugging Face
    checkpoint plus training_config.json to output_dir."""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = best_device()
    use_amp = device == "cuda" and cfg.fp16
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = SharedBiEncoder(cfg.base_model, tau=cfg.tau).to(device)

    pairs = build_pairs(dictionary, cfg.max_pairs_per_entry, cfg.seed)
    if not pairs:
        raise RuntimeError("No training pairs, every entry has a single name.")
    print(
        f"{len(pairs):,} pairs; {cfg.steps:,} steps * batch {cfg.batch_size} = {cfg.steps * cfg.batch_size / len(pairs):.2f} passes over the pool."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.steps * cfg.warmup_ratio), cfg.steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def tok(texts: list[str]) -> dict:
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    model.train()
    rng = random.Random(cfg.seed)
    running = 0.0
    pbar = tqdm(range(1, cfg.steps + 1), desc="train")
    for step in pbar:
        # Sample with replacement for a fixed number of steps, rather than
        # running epochs: D0 and D1 then get the same number of updates
        # regardless of how many pairs each has.
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(cfg.batch_size)]
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(tok([b[1] for b in batch]), tok([b[2] for b in batch]))
            loss = multi_positive_infonce(logits, [b[0] for b in batch])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        if step % cfg.log_every == 0:
            pbar.set_postfix(loss=f"{running / cfg.log_every:.4f}")
            running = 0.0

    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "n_pairs": len(pairs),
                "norm_level": NORM_LEVEL,
                "pooling": "mean",
                "similarity": "cosine",
                "objective": "multi-positive in-batch InfoNCE, averaged over positive pairs",
            },
            indent=2,
        )
    )
    return output_dir
