"""Training objectives."""

import torch
import torch.nn.functional as F


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07):
    """Symmetric InfoNCE with in-batch negatives.

    `anchor[i]` and `positive[i]` are the two views of segment i; every other
    segment in the batch is a negative.
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    logits = anchor @ positive.t() / temperature
    targets = torch.arange(anchor.shape[0], device=anchor.device)

    loss_forward = F.cross_entropy(logits, targets)
    loss_backward = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_forward + loss_backward)
