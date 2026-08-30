"""Camera models for AssemblyHands ego/exo rigs.

Two distinct sources of camera parameters, and mixing them up is the classic
way to get a reprojection that is subtly, confusingly wrong:

  nimble_json_calib/*.json   ORIGINAL camera models (with distortion) for all
                             12 cameras of a sequence. Ships with the toolkit.
  annotations/*_ego_calib_*  Parameters for the RECTIFIED ego images that are
                             actually distributed. Use these to reproject onto
                             a downloaded ego frame.

World coordinates are millimetres. ModelViewMatrix is 4x4 world->camera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """A pinhole camera with OpenCV radial-tangential distortion."""

    serial: str
    mounting: str  # "Egocentric" | "Exocentric"
    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics
    dist: np.ndarray  # (k1 k2 p1 p2 k3) OpenCV order
    T_world_cam: np.ndarray  # 4x4 world -> camera

    @property
    def is_ego(self) -> bool:
        return self.mounting == "Egocentric"

    @property
    def R(self) -> np.ndarray:
        return self.T_world_cam[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T_world_cam[:3, 3]

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix, ignoring distortion. For triangulation."""
        return self.K @ self.T_world_cam[:3, :]

    @property
    def center_world(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    def project(self, pts_world: np.ndarray, distort: bool = True) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixels.

        Points behind the camera come back as NaN rather than silently folding
        onto the image plane, which is what makes a bad extrinsic look plausible.
        """
        pts_world = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam = pts_world @ self.R.T + self.t
        z = cam[:, 2]
        out = np.full((len(cam), 2), np.nan)
        ok = z > 1e-6
        if not ok.any():
            return out

        xy = cam[ok, :2] / z[ok, None]
        if distort:
            k1, k2, p1, p2, k3 = self.dist
            x, y = xy[:, 0], xy[:, 1]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            xy = np.stack([x_d, y_d], axis=1)

        out[ok] = xy @ self.K[:2, :2].T + self.K[:2, 2]
        return out


def load_nimble_calib(path: str | Path) -> dict[str, Camera]:
    """Load a nimble_json_calib sequence file -> {serial: Camera}."""
    records = json.loads(Path(path).read_text())
    cams: dict[str, Camera] = {}
    for rec in records:
        c = rec["Camera"]
        K = np.array(
            [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]]
        )
        dist = np.array(
            [c.get("k1", 0.0), c.get("k2", 0.0), c.get("p1", 0.0), c.get("p2", 0.0), c.get("k3", 0.0)]
        )
        cams[c["SerialNo"]] = Camera(
            serial=c["SerialNo"],
            mounting=rec["MountingLocation"],
            width=c["ImageSizeX"],
            height=c["ImageSizeY"],
            K=K,
            dist=dist,
            T_world_cam=np.array(c["ModelViewMatrix"], dtype=np.float64),
        )
    return cams


def triangulate(cams: list[Camera], uvs: np.ndarray) -> np.ndarray:
    """Direct linear transform across N views.

    cams: N cameras. uvs: Nx2 pixel observations (NaN rows are dropped).
    Returns a world-coordinate 3D point, or NaN if fewer than two views.
    """
    rows = []
    for cam, uv in zip(cams, uvs):
        if not np.all(np.isfinite(uv)):
            continue
        P = cam.P
        rows.append(uv[0] * P[2] - P[0])
        rows.append(uv[1] * P[2] - P[1])
    if len(rows) < 4:  # need >= 2 views
        return np.full(3, np.nan)
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_many(cams: list[Camera], uvs: np.ndarray) -> np.ndarray:
    """uvs: (n_views, n_points, 2) -> (n_points, 3) in world coordinates."""
    uvs = np.asarray(uvs)
    return np.array([triangulate(cams, uvs[:, j]) for j in range(uvs.shape[1])])
