from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def positive_mask(entry_ids: Sequence[str], device=None) -> torch.Tensor:
    """mask[i, j] is True when batch items i and j belong to the same entry."""
    lookup: dict[str, int] = {}
    idx = torch.as_tensor(
        [lookup.setdefault(e, len(lookup)) for e in entry_ids], device=device
    )
    return idx[:, None] == idx[None, :]


def multi_positive_infonce(
    logits: torch.Tensor, entry_ids: Sequence[str]
) -> torch.Tensor:
    """In-batch InfoNCE where every same-entry item counts as a positive.

    Two pairs from the same entry can land in one batch; treating the other
    one as a negative would push synonyms apart. The loss is averaged over
    positive pairs, not over rows.
    """
    positive = positive_mask(entry_ids, logits.device)
    log_probs = F.log_softmax(logits, dim=1)
    return -(log_probs * positive).sum() / positive.sum().clamp(min=1)
