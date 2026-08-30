"""Training loop for one (config, label budget, seed) run.

Fairness constraint: every config takes the *same* number of supervised
gradient steps on the *same* labelled subset. Configs differ only in whether an
auxiliary contrastive term is added, and where its second view comes from. That
keeps "does exo help" from collapsing into "did this config train longer".
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import stratified_subset, temporal_crop
from .evaluate import (
    mean_per_class_accuracy,
    per_class_accuracy,
    retrieval_map,
    top1_accuracy,
)
from .heads import TemporalHead
from .objectives import info_nce

CONTRASTIVE_MODES = ("none", "exo", "ego_aug")


def build_heads(config, feature_dim, num_frames, num_classes, device):
    ego_head = TemporalHead(
        in_dim=feature_dim,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        proj_dim=config["proj_dim"],
        num_frames=num_frames,
        dropout=config["dropout"],
    ).to(device)

    exo_head = None
    if config["contrastive"] == "exo":
        # Separate head: grayscale fisheye ego and fixed RGB exo have very
        # different statistics. Only the ego head survives to evaluation.
        exo_head = TemporalHead(
            in_dim=feature_dim,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            num_classes=num_classes,
            proj_dim=config["proj_dim"],
            num_frames=num_frames,
            dropout=config["dropout"],
        ).to(device)
    return ego_head, exo_head


def run_experiment(config, train_store, eval_store, budget, seed, device):
    if config["contrastive"] not in CONTRASTIVE_MODES:
        raise ValueError(f"unknown contrastive mode {config['contrastive']!r}")

    started = time.time()
    torch.manual_seed(seed)
    numpy_rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    train_ego, train_exo, train_labels = train_store.tensors(device)
    eval_ego, _, eval_labels = eval_store.tensors(device)

    labelled = stratified_subset(train_store.labels, budget, numpy_rng)
    labelled_idx = torch.from_numpy(labelled).to(device)
    effective_budget = len(labelled)

    # Unlabelled ego/exo pairs stay available at every budget -- only the labels
    # are restricted. That is the "fewer demonstrations" claim being tested.
    if config.get("pairs_follow_labels", False):
        pair_pool = labelled_idx
    else:
        pair_pool = torch.arange(len(train_store), device=device)

    ego_head, exo_head = build_heads(
        config, train_store.dim, train_store.num_frames, len(train_store.vocab), device
    )

    parameters = list(ego_head.parameters())
    if exo_head is not None:
        parameters += list(exo_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["steps"])

    batch_labeled = min(config["batch_labeled"], max(1, effective_budget))
    ego_head.train()
    if exo_head is not None:
        exo_head.train()

    for step in range(config["steps"]):
        pick = torch.randint(0, effective_budget, (batch_labeled,),
                             device=device, generator=generator)
        rows = labelled_idx[pick]
        _, logits, _ = ego_head(train_ego[rows])
        loss = F.cross_entropy(logits, train_labels[rows])

        if config["contrastive"] != "none":
            pair_pick = torch.randint(0, len(pair_pool), (config["batch_pairs"],),
                                      device=device, generator=generator)
            pair_rows = pair_pool[pair_pick]
            anchor_features = train_ego[pair_rows]

            if config["contrastive"] == "exo":
                _, _, anchor_projection = ego_head(anchor_features)
                _, _, other_projection = exo_head(train_exo[pair_rows])
            else:
                view_one = temporal_crop(anchor_features, generator, config["min_scale"])
                view_two = temporal_crop(anchor_features, generator, config["min_scale"])
                _, _, anchor_projection = ego_head(view_one)
                _, _, other_projection = ego_head(view_two)

            loss = loss + config["lambda"] * info_nce(
                anchor_projection, other_projection, config["temperature"]
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    ego_head.eval()
    with torch.no_grad():
        embeddings, logits, _ = ego_head(eval_ego)

    per_task = per_class_accuracy(logits, eval_labels, train_store.vocab)

    summary = {
        "config": config["name"],
        "budget": budget,
        "effective_budget": effective_budget,
        "seed": seed,
        "top1": round(top1_accuracy(logits, eval_labels), 5),
        "mean_per_class": round(mean_per_class_accuracy(logits, eval_labels), 5),
        "retrieval_map": round(retrieval_map(embeddings, eval_labels), 5),
        "n_pairs": int(len(pair_pool)) if config["contrastive"] != "none" else 0,
        "wall_s": round(time.time() - started, 1),
    }
    return summary, per_task
