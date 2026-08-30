"""Sweep configs x label budgets x seeds; append results to a CSV."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exoego.data import FeatureStore
from exoego.encoders import pick_device
from exoego.paths import features_dir, repo_root, results_dir
from exoego.train import run_experiment


def load_config(name: str) -> dict:
    config_dir = repo_root() / "configs"
    with open(config_dir / "base.yaml") as handle:
        config = yaml.safe_load(handle)
    with open(config_dir / f"{name}.yaml") as handle:
        config.update(yaml.safe_load(handle))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=["ego_only", "ego_exo", "ego_ego"])
    parser.add_argument("--budgets", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--feature-dir", default=None,
                        help="override the cached-feature directory")
    parser.add_argument("--backbone", default=None,
                        help="override the backbone in base.yaml (selects the feature cache)")
    args = parser.parse_args()

    device = pick_device(args.device)
    base = load_config(args.configs[0])
    backbone = args.backbone if args.backbone else base["backbone"]
    if args.feature_dir:
        feature_dir = Path(args.feature_dir)
    else:
        feature_dir = features_dir() / f"{backbone}_T{base['frames']}"
    if not feature_dir.exists():
        raise SystemExit(f"no cached features at {feature_dir}; run 04_extract_features.py")

    # One store per distinct exo array, shared across configs that use it.
    stores = {}

    def get_stores(exo_name):
        if exo_name not in stores:
            stores[exo_name] = (FeatureStore(feature_dir, "train", exo_name=exo_name),
                                FeatureStore(feature_dir, "eval", exo_name=exo_name))
        return stores[exo_name]

    train_store, eval_store = get_stores("exo")
    print(f"train {len(train_store)} clips | eval {len(eval_store)} clips | "
          f"dim {train_store.dim} | T {train_store.num_frames} | device {device}")

    out_path = Path(args.out) if args.out else results_dir() / "label_efficiency.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_rows = []
    for config_name in args.configs:
        config = load_config(config_name)
        train_store, eval_store = get_stores(config.get("exo_name", "exo"))
        budgets = args.budgets if args.budgets else config["budgets"]
        seeds = args.seeds if args.seeds else config["seeds"]

        for budget in budgets:
            resolved = len(train_store) if budget < 0 else budget
            for seed in seeds:
                result, per_task = run_experiment(config, train_store, eval_store,
                                                  resolved, seed, device)
                result["requested_budget"] = budget
                rows.append(result)
                for verb, stats in per_task.items():
                    task_rows.append({
                        "config": config_name,
                        "budget": resolved,
                        "seed": seed,
                        "verb_cls": verb,
                        "accuracy": round(stats["accuracy"], 5),
                        "support": stats["support"],
                    })
                print(f"{config_name:9s} budget={resolved:5d} seed={seed} "
                      f"top1={result['top1']:.3f} mpc={result['mean_per_class']:.3f} "
                      f"mAP={result['retrieval_map']:.3f} ({result['wall_s']}s)",
                      flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(frame)} rows)")

    task_frame = pd.DataFrame(task_rows)
    task_path = out_path.parent / "per_task.csv"
    task_frame.to_csv(task_path, index=False)
    print(f"wrote {task_path} ({len(task_frame)} rows)")


if __name__ == "__main__":
    main()
