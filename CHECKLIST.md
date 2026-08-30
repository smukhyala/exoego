# ExoEgo — Hackathon Execution Checklist

**Thesis:** Exo video teaches a model the metric scale that ego video cannot see.
That lets ego-only factory footage drive a real arm.

Assumes ~7 hours remain. All times are T+ from now.

---

## Result tiers — know what you ship if things break

| Tier | Result | Needs robot? |
|---|---|---|
| **T1 (must have)** | mm-error table: ego-only vs ego+exo 3D hand pose on AssemblyHands | No |
| **T2** | OpenArm executes a trajectory derived from World Context footage | Yes |
| **T3** | Arm A/B: raw-ego-driven vs exo-distilled-driven, N trials each | Yes |

T1 alone is a defensible submission. Protect it first.

## Hard gates — decide and move on, do not negotiate

- **G1 @ T+1:00** — 3D ground truth reprojects onto the hands in both ego and exo views.
  Fail → we've misread the dataset. Stop everything on Track B and fix.
- **G2 @ T+1:30** — one OpenArm motor responds to a commanded position.
  Fail → drop to perception-gating fallback, robot becomes camera-only. No retry.
- **G3 @ T+2:00** — a hand is detected in ≥30% of World Context frames for the chosen task.
  Fail → switch task. Do not attempt to fix the detector.
- **FREEZE @ T+5:30** — no new code. Only plots, slides, video.

---

## T+0:00 · Setup — everyone, 20 min

- [ ] `brew install ffmpeg`
- [ ] `pip install opencv-python mediapipe scipy scikit-learn matplotlib tqdm gdown`
- [ ] Apply for Ego-Exo4D access — 2 min, free option, may unlock later
- [ ] Start AssemblyHands **annotations** download in background (small, gates everything)
- [ ] Agree repo layout: `data/ src/ results/` — one branch, small commits
- [ ] **Copy World Context `meta/` + chosen task clips from USB to local SSD.**
      Reading video repeatedly off the USB will bottleneck every later step. 590 GB free.

---

## Track A · OpenArm — P1, starts immediately

Safety first, always. Current limit before position command.

- [ ] CAN bring-up: `sudo ip link set can0 up type can bitrate 1000000`, confirm `candump can0` shows traffic
- [ ] Enumerate motor IDs
- [ ] **Set torque/current limit ~15% of rated BEFORE any motion command**
- [ ] Verify the limit actually clamps — stall a joint by hand. If it doesn't stall, the limit is not applied
- [ ] One joint responds to a slow position command  ← **GATE G2 @ T+1:30**
- [ ] Gripper calibration script: slow current-limited close, log current continuously,
      detect range from current rise (never by commanding endpoints), soft limits at 90% of range
- [ ] Watchdog: cut on sustained high current. E-stop reachable at all times
- [ ] Expose `move_to(joint_targets)` + `home()` as the only public API
- [ ] IK: OpenArm URDF via ikpy/pinocchio — fallback is reduced-DoF direct mapping

**Fallback if G2 fails:** train a grasp-moment / hand-contact classifier on World Context clips,
run it on the Jetson against the arm's camera, let it trigger a scripted motion.
Dataset-trained perception gating real control still qualifies for the bonus track.

---

## Track B · AssemblyHands ego/exo — P2

- [ ] Download order: **annotations → one split of ego images → exo only for takes actually used**.
      Do not pull all 490K ego images.
- [ ] Loader: given a frame → ego image, exo images, 3D keypoints, intrinsics, extrinsics
- [ ] **Reproject 3D GT into ego and exo, overlay, eyeball it**  ← **GATE G1 @ T+1:00**
- [ ] Ego-only 3D estimate (monocular, scale-ambiguous) — MediaPipe or HaMeR
- [ ] Exo triangulation across static cams using provided extrinsics → metric 3D
- [ ] Error vs GT for both → **T1 RESULT: mm error table**
- [ ] Distillation head: ego features → metric scale correction, supervised by exo-derived 3D.
      Small model, minutes to train. This is the thing that transfers.

---

## Track C · World Context → arm-ready trajectories — P3

- [ ] Filter clips to `calibration_status == "intrinsics_and_gyro"` (200 of 424)
- [ ] Pick primary task: `bottle-surface-buffing` (planar, repetitive, one grasp — lowest gripper risk).
      Backup: `component-alignment-sticker-application`. Avoid `garment-folding-*` (deformables).
- [ ] Decode at 5 fps → undistort with Kannala-Brandt intrinsics → **convert to grayscale**
      (AssemblyHands ego cams are monochrome; skip this and the model silently degrades)
- [ ] IMU: load sidecar → subtract bias (**bias is in dps, samples are rad/s — convert**)
      → apply gyro→camera matrix `M` → apply `time_offset_ms` → integrate to orientation
      → *never apply the GPMF ORIN remap first*
- [ ] Hand detection rate on chosen task  ← **GATE G3 @ T+2:00**
- [ ] Head-rotation-compensated 3D hand trajectory, smoothed

---

## Track D · Integration — P4, joins T+2:30

- [ ] Retarget: 3D hand trajectory → normalize into arm workspace box → IK → joint targets
- [ ] Rate-limit and clamp every command. Plot the trajectory before it touches hardware
- [ ] Dry run: plot/sim only, confirm nothing exceeds soft limits
- [ ] **T2 RESULT:** arm executes a World Context-derived trajectory
- [ ] **T3 RESULT:** arm A/B — raw-ego vs exo-distilled trajectory, N trials each, count successes
- [ ] Record video of both runs, same framing

---

## T+5:30 · Freeze and present

- [ ] **CODE FREEZE**
- [ ] Plot 1: mm error, ego vs ego+exo (T1)
- [ ] Plot 2: arm success counts, ego vs exo-distilled (T3)
- [ ] Slide: quote the World Context data card — metric scale was never calibrated.
      Their own documentation is the motivation for exo.
- [ ] Video: arm moving, driven by factory footage
- [ ] Bonus track framing: the dataset is in the causal chain that moves the arm — say it explicitly

---

## Sources

- AssemblyHands — https://assemblyhands.github.io/
- Toolkit — https://github.com/facebookresearch/assemblyhands-toolkit
- Assembly101 paper — https://arxiv.org/pdf/2203.14712
- Ego-Exo4D — https://docs.ego-exo4d-data.org/getting-started/
- Charades-Ego (fallback) — https://prior.allenai.org/projects/charades
