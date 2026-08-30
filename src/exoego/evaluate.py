"""Evaluation metrics, all computed on held-out egocentric clips only."""

import numpy as np
import torch
import torch.nn.functional as F


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean())


def mean_per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged recall over classes present in the evaluation set.

    Verb classes are heavily imbalanced ("pick up" alone is ~19% of segments),
    so top-1 alone would reward predicting the majority class.
    """
    predictions = logits.argmax(dim=-1).cpu().numpy()
    targets = labels.cpu().numpy()

    recalls = []
    for label in np.unique(targets):
        mask = targets == label
        recalls.append(float((predictions[mask] == label).mean()))
    return float(np.mean(recalls))


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, vocab) -> dict:
    """Per-verb recall, for the per-task breakdown.

    Only classes present in the evaluation set get an entry; a class with no
    support has no meaningful recall and must not be reported as zero.
    """
    predictions = logits.argmax(dim=-1).cpu().numpy()
    targets = labels.cpu().numpy()

    out = {}
    for label in np.unique(targets):
        mask = targets == label
        out[vocab[int(label)]] = {
            "accuracy": float((predictions[mask] == label).mean()),
            "support": int(mask.sum()),
        }
    return out


def retrieval_map(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Ego->ego same-verb retrieval mAP, excluding the query itself.

    A classifier-free probe of representation quality: it reads the embedding
    geometry directly rather than whatever the linear head learned.
    """
    normalized = F.normalize(embeddings, dim=-1)
    similarity = normalized @ normalized.t()
    count = similarity.shape[0]
    similarity.fill_diagonal_(float("-inf"))

    order = similarity.argsort(dim=-1, descending=True)
    ranked_labels = labels[order]
    relevant = (ranked_labels == labels.unsqueeze(1)).float()

    positions = torch.arange(1, count, device=embeddings.device).float()
    relevant = relevant[:, : count - 1]
    cumulative = relevant.cumsum(dim=1)
    precision = cumulative / positions.unsqueeze(0)

    total_relevant = relevant.sum(dim=1)
    keep = total_relevant > 0
    if keep.sum() == 0:
        return 0.0
    average_precision = (precision * relevant).sum(dim=1)[keep] / total_relevant[keep]
    return float(average_precision.mean())
