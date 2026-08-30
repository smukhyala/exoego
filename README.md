# ExoEgo

**Exo video teaches a model the metric scale that ego video cannot see.
That lets ego-only factory footage drive a real arm.**

Hackathon build: OpenArm v1 + the World Context industrial egocentric dataset,
with AssemblyHands supplying the paired ego/exo supervision that World Context lacks.

---

## The problem, stated precisely

We were given 424 clips of industrial egocentric video with IMU sidecars. Its own
data card says the thing that decides this project:

> camera-to-IMU translation, rolling shutter, accelerometer bias and **metric scale
> were not calibrated**

Ego video gives you rotation. It does not give you metric depth. A hand trajectory
recovered from a single head-mounted camera is correct in shape and wrong in size,
and an arm driven by it reaches to the wrong place.

Third-person (exo) video supplies exactly the missing quantity, because two or more
views of the same hand triangulate to a metric position.

So the claim is not the vague "more cameras help." It is specific and falsifiable:

> **A model that learns metric scale from exo supervision recovers 3D hand
> trajectories from ego-only video more accurately than a monocular ego baseline —
> accurately enough to drive a physical arm.**

## Why AssemblyHands

World Context is ego-only, so it cannot supply the supervision. AssemblyHands can:

| | |
|---|---|
| Rig | 4 egocentric mono cameras (636×480) + 8 static exocentric RGB (1920×1080) |
| Ground truth | 3D hand keypoints in world coordinates, millimetres, 42 joints |
| Calibration | per-frame ego extrinsics; static exo extrinsics; rectified intrinsics |
| Domain | bench-top assembly and disassembly — the same posture as our target tasks |
| Access | public Google Drive, no approval gate, CC BY-NC 4.0 |

The domains line up better than expected. AssemblyHands ego cameras are wide-angle
mono; normalized focal length is 192/636 ≈ 0.30 against World Context's 635/1920 ≈ 0.33.
Comparable field of view, which is what matters for transfer.

**The ego cameras are monochrome.** World Context is colour. Convert World Context
frames to greyscale before inference or the model quietly degrades.

---

## The key shortcut: the ground truth *is* the exo signal

AssemblyHands' 3D annotations were produced automatically from the 8 static exo
cameras. So we do not need to download 50 GB of exo video to have an exo signal —
**the provided 3D ground truth already is the exo-derived measurement.**

This collapses the critical path enormously:

- exo branch = the shipped 3D ground truth (already downloaded, 2.2 GB)
- ego branch = what we estimate from a single ego image
- the comparison between them = the entire T1 result

Exo videos become optional, wanted only for a visual demo of the triangulation.
Do not put them on the critical path.

---

## Pipeline

```
AssemblyHands                          World Context                OpenArm v1
─────────────                          ─────────────                ──────────
ego image (mono, 636×480)              ego clip (fisheye 1080p)
   │                                      │
   │  MediaPipe → 2D hand                 │  undistort (Kannala-Brandt)
   ▼                                      │  → greyscale → MediaPipe
[ ego-only 3D ]  scale-ambiguous          ▼
   │                                   [ 2D hand + IMU orientation ]
   │  ◀── supervised by ───┐              │
   ▼                       │              ▼
[ distillation head ]      │           [ metric 3D trajectory ]  ── retarget ──▶ arm
   ego features → metric   │              │                                      │
   scale correction        │              │                                      ▼
   │                       │              │                                  A/B trials
   ▼                       │              │                              raw-ego vs distilled
[ metric 3D ] ─── error ───┘              │
                    ▲                     │
        3D GT = exo-triangulated ─────────┘
```

Left half proves the claim. Right half puts it on hardware.

---

## Stages

### Stage 0 — Validate the geometry ✅ DONE

`python -m src.g1_check`

Three checks that need no images:

- **A. Ego reprojection.** Project world 3D through per-frame ego extrinsics and
  rectified intrinsics, compare against annotated 2D. Result: **median 0.000 px**
  over 400 frames / 8,734 joints.
- **B. Exo round-trip.** Project 3D into the 8 exo views, triangulate back.
  Result: **median 0.000 mm**, every joint visible in all 8 views.
- **C. Rig geometry.** Exo baselines 589–1956 mm — well conditioned for triangulation.

**Read A honestly.** Zero error means the annotated 2D *is* a reprojection of the 3D
through these parameters. It proves our camera conventions, units and frame indexing
are right. It does **not** independently prove the 3D sits on real hands in the pixels.
That check is Stage 1.

### Stage 1 — Ego images and visual confirmation

- Check the size of the `val` split before pulling anything. Val only. Not train (490K images).
- Overlay projected 3D onto real ego frames. **This is the real G1.** If the skeleton
  does not land on the hands, stop — everything downstream is built on sand.
- Build the paired sample index: `(ego image, 2D keypoints, 3D world GT, camera params)`.

### Stage 2 — Ego-only baseline (the thing to beat)

MediaPipe on the ego image gives 2D landmarks plus a root-relative 2.5D estimate with
**no metric scale**. Two honest baselines:

- **B0 — arbitrary scale.** Take the monocular estimate as-is. Expect large error.
- **B1 — oracle scale.** Fit the single best global scale per sequence, in hindsight.
  This is deliberately generous: it is the best a monocular method could do if someone
  handed it the right scale. Beating B1 is the real result.

Metric: **MPJPE in mm**, root-aligned, plus wrist-trajectory error, which is what the
arm actually consumes.

### Stage 3 — Exo-supervised distillation

Small head: ego features → metric scale/depth correction, supervised by the
exo-derived 3D GT. Frozen visual features, cached once. Trains in minutes on the M4 Max.

Report **error vs number of training sequences** — a sample-efficiency curve, not a
single number. Three lines: B0, B1, distilled.

### Stage 4 — Transfer to World Context

- Filter to `calibration_status == "intrinsics_and_gyro"` (200 of 424 clips).
- Undistort with the Kannala-Brandt fisheye intrinsics → convert to **greyscale**.
- IMU: subtract bias (**bias is dps, samples are rad/s — convert**), apply the
  gyro→camera matrix `M`, apply `time_offset_ms`, integrate to orientation.
  *Never apply the GPMF ORIN remap first.*
- Compensate head rotation out of the hand trajectory. Run the distilled model.
- **Gate G3:** hand detected in ≥30% of frames, or switch task.

### Stage 5 — Retarget to OpenArm

- Normalize the metric 3D wrist trajectory into the arm's workspace box.
- IK against the OpenArm URDF (ikpy/pinocchio); fallback is reduced-DoF direct mapping.
- Rate-limit, clamp to soft limits, plot before anything touches hardware.

### Stage 6 — The hardware A/B

Same task, same framing, N trials each:

- arm driven by the **raw ego** trajectory
- arm driven by the **exo-distilled** trajectory

Count successes. This is the thesis, demonstrated physically rather than tabulated.

---

## Task choice

`bottle-surface-buffing` — planar, repetitive, one grasp, no regrasping. It minimises
gripper actuation, which minimises the chance of damaging the third-party grippers.

Backup: `component-alignment-sticker-application`.
Avoid `garment-folding-*` — deformables are the hardest thing in manipulation.

## Robot safety

Current limit **before** any position command, every time.

- Set torque/current to ~15% of rated, then verify it clamps by stalling a joint by hand.
  If it does not stall, the limit is not applied.
- Discover gripper range by slow current-limited closing until current rises — never by
  commanding endpoints. Soft limits at 90% of the discovered range.
- Watchdog on sustained current. E-stop reachable at all times.

---

## Result tiers

| Tier | Result | Robot? |
|---|---|---|
| **T1** | mm-error table + sample-efficiency curve, ego-only vs exo-distilled | No |
| **T2** | OpenArm executes a World Context-derived trajectory | Yes |
| **T3** | Hardware A/B, raw-ego vs exo-distilled, N trials | Yes |

T1 is the floor and stands alone. Protect it first.

## Layout

```
src/calib.py        camera models, projection, triangulation
src/g1_check.py     Stage 0 geometry validation
data/assemblyhands/annotations/    2.2 GB, downloaded
third_party/assemblyhands-toolkit/ upstream loaders + bundled nimble calib
CHECKLIST.md        hour-by-hour run sheet with hard gates
```

Run sheet: https://claude.ai/code/artifact/16040e8b-79f9-4f4f-b6db-50dbbc6d117a

## Sources

- [AssemblyHands](https://assemblyhands.github.io/) · [toolkit](https://github.com/facebookresearch/assemblyhands-toolkit) · [Assembly101](https://arxiv.org/pdf/2203.14712)
- [Ego-Exo4D](https://docs.ego-exo4d-data.org/getting-started/) — applied, ~48 h approval, not on the critical path
- AssemblyHands is CC BY-NC 4.0. Non-commercial.
