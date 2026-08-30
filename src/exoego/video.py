"""Frame sampling and single-pass video decoding.

Feature extraction deliberately avoids seeking once per segment: a 237s 1080p
recording holds ~225 segments, and per-segment seeking costs hours across the
dataset. Instead we compute every frame index the whole video needs, then walk
the file once with a sequential read and hand back only those frames.
"""

import cv2
import numpy as np

from .annotations import FRAME_SCALE

SHORT_SIDE = 256
CROP = 224

# Assembly101's egocentric cameras are 636x480 monochrome; the exocentric ones
# are 1920x1080 RGB. EGO_ROWS is ego's vertical resolution, used to knock the exo
# stream down to matching image quality for the ego_exo_degraded ablation.
EGO_ROWS = 480


def timeline_scale(view_frames: int, reference_frames: int) -> float:
    """Factor mapping the canonical annotation timeline onto one view's frames.

    Ego and exo streams of the same recording do not always have equal frame
    counts: some views drop frames uniformly through the recording, so a view
    can end up to ~70 frames (~1.2s) short over a 28,000-frame session. That
    drift is cumulative, not a trailing truncation -- measured by motion-energy
    cross-correlation, the ego/exo lag on the worst recording grows from +13
    frames in the first half to +61 in the second.

    Annotations live on the longer (canonical) timeline: in recordings where the
    two views differ, 2 * max(end_frame) can exceed the shorter view's frame
    count while still fitting the longer one. So we scale each view by its share
    of the reference length. Rescaling collapses the measured lag from +61 to 0
    frames and raises cross-view correlation, confirming the drift is linear.
    """
    if reference_frames <= 0:
        return 1.0
    return view_frames / reference_frames


def sample_source_indices(start_frame: int, end_frame: int, num_frames: int,
                          scale: float = 1.0, max_index=None) -> np.ndarray:
    """Source (60fps) frame indices for one segment, given 30fps annotation frames.

    Uniformly spans the segment. Segments shorter than `num_frames` repeat
    frames rather than being dropped -- a quarter of Assembly101 segments are
    under 16 frames at 30fps. `scale` corrects for per-view frame drop; see
    `timeline_scale`.
    """
    last = max(start_frame, end_frame - 1)
    annotation_idx = np.linspace(start_frame, last, num_frames)
    source_idx = np.rint(annotation_idx * FRAME_SCALE * scale).astype(np.int64)
    if max_index is not None:
        source_idx = np.clip(source_idx, 0, max_index)
    return source_idx


def degrade_to_ego(frame_bgr: np.ndarray) -> np.ndarray:
    """Reduce an exo frame to ego-grade image quality, preserving its viewpoint.

    Drops resolution to ego's 480 rows and removes colour, leaving a 3-channel
    BGR frame (ego videos already decode as three equal channels, so downstream
    code is unchanged).

    Aspect ratio is deliberately preserved rather than forced to ego's exact
    636x480. `preprocess` normalises the SHORT side to 256 and centre-crops 224,
    so the retained field of view depends on aspect ratio: 1920x1080 keeps 49.2%
    of frame width, while a 636x480 squash would keep 66.1% -- a different slice
    of the scene, plus anisotropic stretch. Preserving aspect keeps framing
    byte-identical to the undegraded exo path, so resolution and colour are the
    only variables that change.
    """
    height, width = frame_bgr.shape[:2]
    if height <= EGO_ROWS:
        small = frame_bgr
    else:
        new_width = int(round(width * EGO_ROWS / height))
        small = cv2.resize(frame_bgr, (new_width, EGO_ROWS), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def frame_count(video_path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"could not open {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return total


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> uint8 RGB, short side 256, centre-cropped to 224x224."""
    height, width = frame_bgr.shape[:2]
    if height < width:
        new_height = SHORT_SIDE
        new_width = int(round(width * SHORT_SIDE / height))
    else:
        new_width = SHORT_SIDE
        new_height = int(round(height * SHORT_SIDE / width))
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top = (new_height - CROP) // 2
    left = (new_width - CROP) // 2
    cropped = resized[top:top + CROP, left:left + CROP]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def iter_needed_frames(video_path, needed_sorted, degrade: bool = False):
    """Yield (source_index, preprocessed_frame) for each index in `needed_sorted`.

    `needed_sorted` must be sorted ascending. Indices past the end of the file
    are silently skipped; callers pad short segments.

    `degrade` applies `degrade_to_ego` before `preprocess`, i.e. at the only
    point where the native-resolution frame exists. Degrading here rather than
    downstream is what makes it a true input-quality ablation: the backbone
    genuinely never sees the high-resolution colour pixels.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"could not open {video_path}")

    try:
        position = 0
        cursor = 0
        limit = len(needed_sorted)
        while cursor < limit:
            target = needed_sorted[cursor]
            # Duplicate targets (very short segments) share one decoded frame.
            if target < position:
                cursor += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            if position == target:
                if degrade:
                    frame = degrade_to_ego(frame)
                processed = preprocess(frame)
                while cursor < limit and needed_sorted[cursor] == target:
                    yield target, processed
                    cursor += 1
            position += 1
    finally:
        capture.release()
