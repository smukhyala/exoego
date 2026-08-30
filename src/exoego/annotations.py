"""Assembly101 fine-grained annotation loading and ego/exo pairing.

Two facts drive this module, both verified against the data:

1. Every action segment is annotated once per camera view, with *identical*
   start/end frames across all 12 views. Pairing ego to exo is therefore an
   exact join on (seq, start_frame, end_frame, action_id) -- no temporal
   alignment is needed.
2. Annotation frame numbers are at 30fps while the source videos are 60fps.
   So `seconds = frame / 30` and `source_frame_index = 2 * annotation_frame`.
"""

import pandas as pd

from . import views as views_mod
from .paths import fine_grained_dir

SPLITS = ("train", "validation", "test")

ANNOTATION_FPS = 30.0
SOURCE_FPS = 60.0
FRAME_SCALE = int(SOURCE_FPS / ANNOTATION_FPS)  # annotation frame -> source frame

_USECOLS = [
    "video",
    "start_frame",
    "end_frame",
    "action_id",
    "verb_id",
    "noun_id",
    "action_cls",
    "verb_cls",
    "noun_cls",
    "toy_id",
    "is_shared",
]


def load_split(split: str) -> pd.DataFrame:
    """Raw per-view annotation rows for one official split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    frame = pd.read_csv(fine_grained_dir() / f"{split}.csv", usecols=_USECOLS)
    frame["seq"] = frame["video"].str.split("/").str[0]
    frame["view"] = frame["video"].str.split("/").str[1].str.removesuffix(".mp4")
    return frame


def recording_views(frame: pd.DataFrame) -> dict:
    """Map recording name -> set of view names present in the annotations.

    Derived from the data rather than hardcoded, so both ego camera naming
    families are handled without special-casing.
    """
    available = {}
    for seq, group in frame.groupby("seq", sort=False):
        available[seq] = set(group["view"].unique())
    return available


def verb_vocab() -> list:
    """The 24 verb classes, in a stable alphabetical order."""
    actions = pd.read_csv(fine_grained_dir() / "actions.csv", usecols=["verb_cls"])
    return sorted(actions["verb_cls"].unique())


def make_segment_id(seq: str, start_frame: int, end_frame: int, action_id: int) -> str:
    return f"{seq}__{start_frame:09d}_{end_frame:09d}_a{action_id:04d}"


def segments(split: str, min_duration_s: float = 0.0) -> pd.DataFrame:
    """One row per action segment, with the chosen ego and exo view attached.

    Recordings lacking either an ego or an exo view are dropped.
    """
    frame = load_split(split)
    available = recording_views(frame)

    ego_by_seq = {}
    exo_by_seq = {}
    for seq, view_set in available.items():
        ego_by_seq[seq] = views_mod.pick_ego(view_set)
        exo_by_seq[seq] = views_mod.pick_exo(view_set)

    key = ["seq", "start_frame", "end_frame", "action_id"]
    seg = frame.drop_duplicates(subset=key).copy()
    seg = seg.drop(columns=["video", "view"])

    seg["split"] = split
    seg["ego_view"] = seg["seq"].map(ego_by_seq)
    seg["exo_view"] = seg["seq"].map(exo_by_seq)
    seg = seg[seg["ego_view"].notna() & seg["exo_view"].notna()]

    seg["dur_s"] = (seg["end_frame"] - seg["start_frame"]) / ANNOTATION_FPS
    if min_duration_s > 0.0:
        seg = seg[seg["dur_s"] >= min_duration_s]

    segment_ids = []
    for row in seg.itertuples(index=False):
        segment_ids.append(
            make_segment_id(row.seq, row.start_frame, row.end_frame, row.action_id)
        )
    seg["segment_id"] = segment_ids

    ordered = [
        "segment_id", "seq", "split", "start_frame", "end_frame", "dur_s",
        "action_id", "action_cls", "verb_id", "verb_cls", "noun_id", "noun_cls",
        "toy_id", "is_shared", "ego_view", "exo_view",
    ]
    return seg[ordered].sort_values(["seq", "start_frame"]).reset_index(drop=True)
