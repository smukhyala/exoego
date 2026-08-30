"""Can synchronized exo/ego pairs transfer an ego task head to exo video?

This is a deliberately small proxy for the world-model claim.  It does not
claim robot one-shot imitation.  It tests one prerequisite for that claim:

    Can an unlabelled, synchronized exo/ego corpus teach a bridge that lets a
    task head trained on ego representations understand exo observations from
    entirely held-out recordings?

The bridge is a ridge-regression map from exo features into ego feature space.
It is trained without action labels.  A negative control receives the same
features, model, and optimization, but the exo segments are shifted within each
recording so the views no longer show the same moment.  The action head is fit
only on labelled ego segments and is frozen for every exo condition.

Train, validation, and test are the official Assembly101 recording-disjoint
splits.  Validation selects the bridge regularization using only paired feature
reconstruction and selects the action-head regularization using only ego action
accuracy.  Test is read once after those choices are locked.

The cached TSM features were themselves trained for Assembly101 action
recognition.  This experiment therefore isolates whether PAIRING transfers
task signal across views; it is not evidence that the representation was
learned from raw video without labels.

Run:
    python -m src.paired_bridge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier


CACHE = Path("data/cache/aligned.npz")
RESULTS = Path("results/paired_bridge.json")
FIGURE = Path("results/paired_bridge.png")
BRIDGE_ALPHAS = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
HEAD_ALPHAS = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]


def segment_rows(labels: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Map consecutive (recording, action) frame runs to segment ids."""
    changed = np.empty(len(labels), dtype=bool)
    changed[0] = True
    changed[1:] = ((labels[1:] != labels[:-1])
                   | (sequences[1:] != sequences[:-1]))
    return np.cumsum(changed) - 1


def pool_segments(features: np.ndarray, segment_ids: np.ndarray,
                  n_segments: int, chunk: int = 8_192) -> np.ndarray:
    """Mean-pool a large float16 frame array without a 1 GB cast temporary."""
    pooled = np.zeros((n_segments, features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), chunk):
        stop = min(start + chunk, len(features))
        np.add.at(
            pooled,
            segment_ids[start:stop],
            features[start:stop].astype(np.float32),
        )
    counts = np.bincount(segment_ids, minlength=n_segments).astype(np.float32)
    return pooled / np.maximum(counts[:, None], 1.0)


def shifted_within_recording(features: np.ndarray,
                             sequences: np.ndarray) -> np.ndarray:
    """Break simultaneity while preserving recording/view distributions."""
    shifted = np.empty_like(features)
    for sequence in np.unique(sequences):
        rows = np.flatnonzero(sequences == sequence)
        offset = max(1, len(rows) // 4)
        shifted[rows] = np.roll(features[rows], offset, axis=0)
    return shifted


def mean_per_class(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in np.unique(target):
        mask = target == label
        recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls))


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-12)


def retrieval_metrics(query: np.ndarray, target: np.ndarray,
                      sequences: np.ndarray) -> dict[str, float]:
    """Retrieve the exact synchronized ego segment within each recording."""
    ranks = []
    for sequence in np.unique(sequences):
        rows = np.flatnonzero(sequences == sequence)
        q = query[rows]
        t = target[rows]
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        order = np.argsort(-(q @ t.T), axis=1)
        rank = np.argmax(order == np.arange(len(rows))[:, None], axis=1) + 1
        ranks.extend(rank.tolist())
    ranks_array = np.asarray(ranks)
    return {
        "recall_at_1": float(np.mean(ranks_array <= 1)),
        "recall_at_5": float(np.mean(ranks_array <= 5)),
        "median_rank": float(np.median(ranks_array)),
    }


def recording_bootstrap(correct: np.ndarray, sequences: np.ndarray,
                        samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(sequences)
    draws = np.empty(samples, dtype=np.float64)
    rows_by_sequence = {s: np.flatnonzero(sequences == s) for s in unique}
    for index in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_sequence[s] for s in chosen])
        draws[index] = np.mean(correct[rows])
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def delta_bootstrap(paired_correct: np.ndarray, raw_correct: np.ndarray,
                    sequences: np.ndarray, samples: int,
                    seed: int) -> tuple[float, float, float]:
    """Paired bootstrap over recordings, preserving within-video dependence."""
    rng = np.random.default_rng(seed)
    unique = np.unique(sequences)
    rows_by_sequence = {s: np.flatnonzero(sequences == s) for s in unique}
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_sequence[s] for s in chosen])
        draws[index] = (np.mean(paired_correct[rows])
                        - np.mean(raw_correct[rows]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), float(np.mean(draws <= 0.0))


def plot_results(metrics: dict[str, dict[str, float]], majority: float,
                 output: Path) -> None:
    order = ["raw_exo", "shifted_control", "paired_bridge", "ego_same_view"]
    labels = ["raw exo", "mispaired\ncontrol", "paired\nexo→ego", "ego\nsame-view"]
    values = [metrics[name]["top1"] for name in order]
    lows = [metrics[name]["top1_ci95"][0] for name in order]
    highs = [metrics[name]["top1_ci95"][1] for name in order]
    errors = np.asarray([
        [value - low for value, low in zip(values, lows)],
        [high - value for value, high in zip(values, highs)],
    ])

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    colors = ["#9a9a9a", "#d98e54", "#3274a1", "#707070"]
    bars = axis.bar(labels, values, color=colors, yerr=errors, capsize=4)
    axis.axhline(majority, color="#444444", linestyle="--", linewidth=1)
    axis.text(-0.45, majority + 0.003, "majority", color="#444444", fontsize=9)
    axis.set_ylabel("held-out coarse-action top-1")
    axis.set_title("Synchronized pairs teach an exo→ego task bridge")
    axis.set_ylim(0, max(highs) * 1.25)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                  f"{value:.1%}", ha="center", va="bottom", fontsize=10)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"missing {args.cache}; run: python -m src.build_cache")

    archive = np.load(args.cache)
    frame_labels = archive["y"]
    frame_sequences = archive["seq"]
    frame_splits = archive["split"]
    segment_ids = segment_rows(frame_labels, frame_sequences)
    n_segments = int(segment_ids[-1]) + 1

    labels = np.zeros(n_segments, dtype=np.int64)
    sequences = np.zeros(n_segments, dtype=np.int64)
    splits = np.zeros(n_segments, dtype=np.int8)
    labels[segment_ids] = frame_labels
    sequences[segment_ids] = frame_sequences
    splits[segment_ids] = frame_splits

    ego = pool_segments(archive["ego"], segment_ids, n_segments)
    exo = pool_segments(archive["exo"], segment_ids, n_segments)
    archive.close()

    train, validation, test = splits == 0, splits == 1, splits == 2
    print(
        f"segments train={train.sum()} validation={validation.sum()} "
        f"test={test.sum()} | test recordings={len(np.unique(sequences[test]))}"
    )

    # One shared, unlabelled normalization.  Neither view is privileged here.
    train_views = np.concatenate([ego[train], exo[train]])
    mean = train_views.mean(axis=0)
    scale = train_views.std(axis=0) + 1e-4
    del train_views
    ego = (ego - mean) / scale
    exo = (exo - mean) / scale

    train_exo_shifted = shifted_within_recording(exo[train], sequences[train])

    # Select the bridge without action labels: minimize ego reconstruction MSE
    # on the held-out validation recordings.
    bridge_candidates = []
    for alpha in BRIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(exo[train], ego[train])
        prediction = model.predict(exo[validation])
        mse = float(np.mean((prediction - ego[validation]) ** 2))
        cosine = float(np.mean(row_cosine(prediction, ego[validation])))
        bridge_candidates.append((mse, -cosine, alpha, model))
        print(f"bridge alpha={alpha:>8g} val_mse={mse:.4f} cosine={cosine:.4f}")
    _, _, bridge_alpha, paired_bridge = min(
        bridge_candidates, key=lambda item: (item[0], item[1])
    )
    shifted_control = Ridge(alpha=bridge_alpha).fit(
        train_exo_shifted, ego[train]
    )

    # Select the task head from ego validation only.  Exo labels cannot affect
    # either the head or its hyperparameter.
    head_candidates = []
    for alpha in HEAD_ALPHAS:
        head = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(
            ego[train], labels[train]
        )
        prediction = head.predict(ego[validation])
        macro = mean_per_class(labels[validation], prediction)
        top1 = float(np.mean(prediction == labels[validation]))
        head_candidates.append((-macro, -top1, alpha, head))
        print(f"head   alpha={alpha:>8g} ego_val_top1={top1:.4f} macro={macro:.4f}")
    _, _, head_alpha, task_head = min(
        head_candidates, key=lambda item: (item[0], item[1])
    )

    test_views = {
        "raw_exo": exo[test],
        "shifted_control": shifted_control.predict(exo[test]),
        "paired_bridge": paired_bridge.predict(exo[test]),
        "ego_same_view": ego[test],
    }
    test_labels = labels[test]
    test_sequences = sequences[test]
    metrics = {}
    correct = {}
    for name, features in test_views.items():
        prediction = task_head.predict(features)
        is_correct = prediction == test_labels
        correct[name] = is_correct
        ci = recording_bootstrap(
            is_correct, test_sequences, args.bootstrap, args.seed
        )
        view_metrics = {
            "top1": float(np.mean(is_correct)),
            "mean_per_class": mean_per_class(test_labels, prediction),
            "top1_ci95": list(ci),
            "predicted_classes": int(len(np.unique(prediction))),
        }
        if name != "ego_same_view":
            view_metrics.update({
                "ego_alignment_mse": float(
                    np.mean((features - ego[test]) ** 2)
                ),
                "ego_alignment_cosine": float(
                    np.mean(row_cosine(features, ego[test]))
                ),
                **retrieval_metrics(features, ego[test], test_sequences),
            })
        metrics[name] = view_metrics

    delta = metrics["paired_bridge"]["top1"] - metrics["raw_exo"]["top1"]
    delta_ci = delta_bootstrap(
        correct["paired_bridge"], correct["raw_exo"], test_sequences,
        args.bootstrap, args.seed,
    )
    majority_label = int(np.bincount(test_labels).argmax())
    majority_prediction = np.full_like(test_labels, majority_label)
    majority = float(np.mean(majority_prediction == test_labels))
    majority_macro = mean_per_class(test_labels, majority_prediction)

    result = {
        "question": "can synchronized pairs transfer an ego task head to exo video?",
        "unit": "coarse action segment",
        "splits": {
            "train_segments": int(train.sum()),
            "validation_segments": int(validation.sum()),
            "test_segments": int(test.sum()),
            "test_recordings": int(len(np.unique(test_sequences))),
        },
        "selection": {
            "bridge_alpha": bridge_alpha,
            "bridge_criterion": "minimum paired validation MSE; no action labels",
            "head_alpha": head_alpha,
            "head_criterion": "maximum ego-only validation mean-per-class accuracy",
        },
        "majority_baseline": majority,
        "majority_mean_per_class": majority_macro,
        "test": metrics,
        "paired_minus_raw_top1": delta,
        "paired_minus_raw_top1_ci95": list(delta_ci[:2]),
        "bootstrap_probability_delta_le_zero": delta_ci[2],
        "bootstrap_samples": args.bootstrap,
        "control": (
            "exo segments cyclically shifted within each training recording by "
            "one quarter of its segments"
        ),
        "limitation": (
            "TSM source features were supervised for Assembly101 action recognition; "
            "this isolates the value of synchronization, not label-free raw-video learning"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    plot_results(metrics, majority, args.figure)

    print("\nHELD-OUT TEST — one ego-trained head for every condition")
    print(f"{'condition':<20}{'top-1':>10}{'macro':>10}{'95% CI':>22}")
    print("-" * 62)
    for name in ["raw_exo", "shifted_control", "paired_bridge", "ego_same_view"]:
        item = metrics[name]
        low, high = item["top1_ci95"]
        print(
            f"{name:<20}{item['top1']:>10.3f}{item['mean_per_class']:>10.3f}"
            f"  [{low:.3f}, {high:.3f}]"
        )
    print(f"{'majority':<20}{majority:>10.3f}{majority_macro:>10.3f}")
    print(
        f"\npaired - raw: {delta * 100:+.1f} points "
        f"(recording-bootstrap 95% CI "
        f"[{delta_ci[0] * 100:+.1f}, {delta_ci[1] * 100:+.1f}])"
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
