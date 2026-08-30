# ExoEgo

**Analysis of paired egocentric / exocentric industrial video from World Context.**

World Context supplied a synchronized ego/exo pair from a scooter assembly floor.
This repo recovers the synchronization, then measures what the exocentric view
provides that the egocentric view cannot.

---

## The pair

| | ego | exo |
|---|---|---|
| file | `GX014991-ego-C2920.MP4` | `GX010104-exo-C7459.MP4` |
| mount | head-worn | ceiling, looking down |
| duration | 1129.7 s | 910.0 s |
| video | 1920x1080 HEVC 29.97 fps | 1920x1080 HEVC 29.97 fps |
| audio | present | present |
| telemetry | GPMF `bin_data`, 1129 pkts | GPMF `bin_data`, 910 pkts |

Two things here are **not** in the packaged 424-clip release: **audio**, and
**GPMF telemetry on both cameras**. The release stripped audio and shipped IMU
for ego only. Both matter — audio is what made synchronization possible.

## Stage 1 — Synchronization ✅

`python -m src.sync_audio`

The cameras were started ~11 s apart, so nothing else is possible until the
offset is known.

**First attempt failed.** Correlating log-RMS energy envelopes gave a peak of
0.195 that sat *below* the 99.9th percentile of the correlogram — no peak at all.
The reason is the room: a workshop is dominated by continuous broadband noise
(compressors, fans) that swamps an energy envelope and carries no timing
information.

**What worked** was a spectral-flux onset envelope, which discards steady-state
energy and keeps transients — tool strikes, dropped parts — which is what the two
microphones genuinely share.

```
offset          +11.16 s   (ego t=0 -> exo t=11.16)
peak            0.1355     3.55x the 99.9th percentile
top candidates  11.15 / 11.16 / 11.17 s   (adjacent lags, +-10 ms)
overlap         898.8 s = 15.0 min = 26,937 frame pairs
```

Confirmed visually: at the aligned timestamps both views show the same grey
battery panel being fitted, the same worker in a red plaid shirt, and the ego
wearer's own "apollo" shirt matches the person seen from overhead.

## Stage 2 — What exo provides ✅

`python -m src.exo_analysis`

### Camera instability — the mechanism

Mean frame-to-frame pixel change (0-255), 2 fps:

| | median | p90 |
|---|---|---|
| ego | 44.02 | 58.45 |
| exo | 6.43 | 12.45 |

**The ego view changes 6.85x as much between frames.** The exo camera is bolted
to the ceiling; the ego camera rides a head that turns to talk, fetch parts and
check other work. That instability is *why* the ego view keeps losing the task,
and it is measured without any semantics, so no detector bias can touch it.

### Motion blur

| | median | p10 |
|---|---|---|
| ego | 1977.3 | 1347.1 |
| exo | 2161.3 | 2037.9 |

**57.3%** of ego frames are blurrier than the exo camera's 10th percentile.

### What we could NOT show, and why

A first pass asked YOLO "can you see the scooter" per view and reported exo at
2.0% against ego's 19.9%. **That was an artifact and it is retracted.** A
COCO-trained detector has never seen a partly-assembled scooter from directly
overhead: it scores the work object at 0.08-0.22 confidence in the exo view and
0.53-0.75 in ego. The same bias suppressed exo person counts.

Sweeping the threshold rather than picking one shows person visibility is
threshold-dependent and inconclusive:

| conf | ego mean | exo mean | exo/ego |
|---|---|---|---|
| 0.05 | 5.84 | 4.52 | 0.77 |
| 0.20 | 2.25 | 1.98 | 0.88 |
| 0.50 | 1.12 | 0.88 | 0.79 |

The ratio stays below 1 at every threshold, but overhead people are out of
distribution for this detector, so this does **not** establish that ego sees more
people. It establishes that **off-the-shelf semantic detectors are not comparable
across viewpoints** — which is itself a real finding for anyone building ego/exo
benchmarks.

**Methodological rule this yields:** never compare two viewpoints at a single
detector confidence threshold. Sweep it, or use view-agnostic measures.

## Layout

```
src/sync_audio.py     onset-based ego/exo synchronization
src/exo_analysis.py   view-agnostic comparison + threshold sweep
src/exo_coverage.py   first-pass semantic analysis (superseded; kept for the record)
src/calib.py          camera models, projection, triangulation
src/rig_analysis.py   AssemblyHands rig noise-propagation study
results/              sync.json, exo_analysis.json, rig_analysis.json
```

## Earlier work, still valid

`python -m src.rig_analysis` — on AssemblyHands calibration, an upper bound on
what each rig can achieve given realistic 2D detection error. Monocular ego error
is **flat** across detection noise (205.7 mm at 0 px, 206.1 mm at 4 px) because
the error is scale ambiguity, not detection quality: *you cannot fix a monocular
rig with a better hand detector.* An 8-camera exo rig reaches 1.1 mm.
