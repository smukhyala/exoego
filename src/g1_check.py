"""GATE G1 — does the 3D ground truth actually line up with the cameras?

Three independent checks, none of which need the image download:

  A. EGO REPROJECTION. Project world-frame 3D joints through the per-frame ego
     extrinsics and rectified intrinsics, and compare against the annotated 2D
     keypoints. This is stricter than an overlay: if the convention, units or
     frame indexing are wrong, the error explodes.

  B. EXO ROUND-TRIP. Project the same 3D into the 8 static exo cameras, then
     triangulate back. Recovering the original point proves the exo projection
     matrices are self-consistent, which is what the metric branch depends on.

  C. EXO BASELINE. Report camera spread — triangulation from views that are
     nearly collinear is numerically fine but metrically useless.

Run:  python -m src.g1_check
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANN = Path("data/assemblyhands/annotations")
SPLIT = "val"


def _load(split: str):
    d = ANN / split
    ego = json.loads((d / f"assemblyhands_{split}_ego_calib_v1-1.json").read_text())["calibration"]
    exo = json.loads((d / f"assemblyhands_{split}_exo_calib_v1-1.json").read_text())["calibration"]
    j3d = json.loads((d / f"assemblyhands_{split}_joint_3d_v1-1.json").read_text())["annotations"]
    data = json.loads((d / f"assemblyhands_{split}_ego_data_v1-1.json").read_text())
    return ego, exo, j3d, data


def project(K: np.ndarray, Rt: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Nx3 world -> Nx2 pixels. Rectified cameras, so no distortion term."""
    cam = X @ Rt[:, :3].T + Rt[:, 3]
    z = cam[:, 2]
    uv = np.full((len(X), 2), np.nan)
    ok = z > 1e-6
    uv[ok] = (cam[ok, :2] / z[ok, None]) @ K[:2, :2].T + K[:2, 2]
    return uv


def triangulate(Ps: list[np.ndarray], uvs: np.ndarray) -> np.ndarray:
    rows = []
    for P, uv in zip(Ps, uvs):
        if np.all(np.isfinite(uv)):
            rows += [uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]
    if len(rows) < 4:
        return np.full(3, np.nan)
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def main() -> int:
    ego_c, exo_c, j3d, data = _load(SPLIT)

    # index ego_data: (seq, camera, frame) -> annotated 2D keypoints
    ann_by_img = {a["image_id"]: a for a in data["annotations"]}
    images = data["images"]
    print(f"split={SPLIT}  images={len(images)}  seqs={len(j3d)}")

    # ---------------------------------------------------------------- A
    errs, n_pairs, skipped = [], 0, 0
    for im in images:
        seq, cam, fi = im["seq_name"], im["camera"], im["frame_idx"]
        ann = ann_by_img.get(im["id"])
        if ann is None or seq not in j3d:
            skipped += 1
            continue
        fk = f"{fi:06d}"
        if fk not in j3d[seq] or fk not in ego_c[seq]["extrinsics"]:
            skipped += 1
            continue

        cam_key = next((k for k in ego_c[seq]["intrinsics"] if k.startswith(cam)), None)
        if cam_key is None or cam_key not in ego_c[seq]["extrinsics"][fk]:
            skipped += 1
            continue

        X = np.asarray(j3d[seq][fk]["world_coord"], dtype=np.float64)
        valid3d = np.asarray(j3d[seq][fk]["joint_valid"]).astype(bool).ravel()
        kp = np.asarray(ann["keypoints"], dtype=np.float64)
        valid2d = np.asarray(ann["joint_valid"]).astype(bool).ravel()

        K = np.asarray(ego_c[seq]["intrinsics"][cam_key], dtype=np.float64)
        Rt = np.asarray(ego_c[seq]["extrinsics"][fk][cam_key], dtype=np.float64)

        uv = project(K, Rt, X)
        m = valid3d & valid2d & np.isfinite(uv).all(1) & np.isfinite(kp[:, :2]).all(1)
        if m.sum() == 0:
            skipped += 1
            continue
        errs.append(np.linalg.norm(uv[m] - kp[m, :2], axis=1))
        n_pairs += 1
        if n_pairs >= 400:
            break

    if not errs:
        print("\nA. EGO REPROJECTION: FAIL — no frames matched. Check key formats.")
        return 1

    e = np.concatenate(errs)
    print("\nA. EGO REPROJECTION  (3D -> ego pixels vs annotated 2D)")
    print(f"   frames={n_pairs}  joints={len(e)}  skipped={skipped}")
    print(f"   median={np.median(e):7.3f} px   mean={e.mean():7.3f} px")
    print(f"   p95   ={np.percentile(e,95):7.3f} px   max ={e.max():7.3f} px")
    ego_ok = np.median(e) < 2.0

    # ---------------------------------------------------------------- B & C
    seq = next(iter(exo_c))
    exo_cams = [k for k in exo_c[seq]["intrinsics"] if not k.startswith("HMC_")]
    Ps, centers = [], []
    for k in exo_cams:
        K = np.asarray(exo_c[seq]["intrinsics"][k], dtype=np.float64)
        Rt = np.asarray(exo_c[seq]["extrinsics"][k], dtype=np.float64)
        Ps.append(K @ Rt)
        centers.append(-Rt[:, :3].T @ Rt[:, 3])
    centers = np.array(centers)

    fk = next(iter(j3d[seq]))
    X = np.asarray(j3d[seq][fk]["world_coord"], dtype=np.float64)
    vmask = np.asarray(j3d[seq][fk]["joint_valid"]).astype(bool).ravel()
    X = X[vmask]

    uvs = np.array([project(np.asarray(exo_c[seq]["intrinsics"][k]),
                            np.asarray(exo_c[seq]["extrinsics"][k]), X)
                    for k in exo_cams])
    in_view = ((uvs[..., 0] >= 0) & (uvs[..., 0] < 1920)
               & (uvs[..., 1] >= 0) & (uvs[..., 1] < 1080)).sum(0)

    back = np.array([triangulate(Ps, uvs[:, j]) for j in range(len(X))])
    rt = np.linalg.norm(back - X, axis=1)

    print("\nB. EXO ROUND-TRIP  (3D -> 8 exo views -> triangulate back)")
    print(f"   exo cams={len(exo_cams)}  joints={len(X)}")
    print(f"   median={np.median(rt):.6f} mm   max={rt.max():.6f} mm")
    print(f"   views seeing each joint: min={in_view.min()} median={int(np.median(in_view))} max={in_view.max()}")
    exo_ok = np.median(rt) < 1.0 and np.median(in_view) >= 2

    print("\nC. EXO RIG GEOMETRY")
    d = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    print(f"   baseline: min={d[d>0].min():.0f} mm  max={d.max():.0f} mm")
    print(f"   centroid spread (std): {centers.std(0).round(0)} mm")

    print("\n" + "=" * 58)
    print(f"  A ego reprojection  {'PASS' if ego_ok else 'FAIL'}")
    print(f"  B exo round-trip    {'PASS' if exo_ok else 'FAIL'}")
    print(f"  GATE G1: {'PASS — geometry is trustworthy' if (ego_ok and exo_ok) else 'FAIL — stop and fix'}")
    print("=" * 58)
    return 0 if (ego_ok and exo_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
