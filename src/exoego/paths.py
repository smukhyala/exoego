"""Filesystem layout.

Dataset files live outside git. Override the root with EXOEGO_DATA_ROOT.
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / "datasets" / "egoexo"


def data_root() -> Path:
    override = os.environ.get("EXOEGO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return DEFAULT_DATA_ROOT


def assembly_root() -> Path:
    return data_root() / "assembly101"


def recordings_dir() -> Path:
    return assembly_root() / "recordings"


def clips_dir() -> Path:
    return assembly_root() / "clips"


def features_dir() -> Path:
    return assembly_root() / "features"


def manifests_dir() -> Path:
    return assembly_root() / "manifests"


def annotations_dir() -> Path:
    """Clone of https://github.com/assembly-101/assembly101-annotations."""
    return data_root() / "external" / "assembly101-annotations"


def fine_grained_dir() -> Path:
    return annotations_dir() / "fine-grained-annotations"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    return repo_root() / "results"


def ensure_dirs() -> None:
    targets = [recordings_dir(), clips_dir(), features_dir(), manifests_dir(), results_dir()]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
