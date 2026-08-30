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

---

## Stage 3 — Does exo video reduce the labels ego needs? (Assembly101)

The World Context pair above is a single session: enough to characterise *what the
two viewpoints differ in*, not enough to train on. This stage moves to Assembly101,
which ships 8 fixed and 4 head-mounted cameras recording the same assembly sessions,
and asks the quantitative version of the question: **does exocentric video reduce the
number of labelled egocentric clips needed to learn a task?**

Evaluation is ego-only throughout; exo is unlabelled auxiliary training signal that is
never available at test time.

`make find && make download && make features && make train && make results`

### The experiment

Three configs. All are evaluated **ego-only**; exo is never available at eval time.

| Config | Loss | Role |
|---|---|---|
| `ego_only` | `CE(clf(z_ego), verb)` | baseline |
| `ego_exo` | `CE(...) + λ·InfoNCE(p(z_ego), p(z_exo))` | treatment |
| `ego_ego` | `CE(...) + λ·InfoNCE(p(z_ego), p(z_ego_crop))` | **control** |

`ego_ego` is the load-bearing part of the design. It matches `ego_exo`'s loss shape and
parameter count, but its second view is another temporal crop of the *same ego clip*, so
it carries no exocentric information. Without it, a win for `ego_exo` is indistinguishable
from "adding any contrastive regulariser helps".

Two further fairness constraints:

- **Equal supervised updates.** `steps` is fixed across configs, so no config wins by
  training longer.
- **Labels are restricted; pairs are not.** At every budget, `ego_exo` sees all unlabelled
  ego/exo pairs — that *is* the "fewer demonstrations" claim. `pairs_follow_labels: true`
  switches to the stricter variant.

Evaluation reports top-1, **mean-per-class** accuracy (verb classes are heavily imbalanced
— "pick up" alone is ~19% of segments), and classifier-free ego→ego retrieval mAP.

#### Go/no-go gate

Before reading any `ego_exo` vs `ego_only` delta, `ego_only` at the full budget must clearly
beat the majority-class baseline. Otherwise the comparison is measuring noise and the
backbone is the thing to change. `06_report.py` evaluates this automatically and refuses to
endorse the curves when it fails.

**This gate has already earned its keep.** The first run used frozen per-frame DINOv2
ViT-S/14 and failed it:

| backbone | eval top-1 | mean-per-class | gate |
|---|---|---|---|
| DINOv2 ViT-S/14 (per-frame) | 0.179 | 0.094 | **FAIL** (majority = 0.197) |
| VideoMAE, Kinetics (video-native) | **0.290** | **0.154** | **PASS** (1.5x majority, 3.6x chance) |

Under DINOv2 all three configs landed within 0.0015 top-1 of each other. That was not a
null result about exocentric video — it was the encoder pinning every config to the floor.
A linear probe on the same cached features scored **0.95 train / 0.155 eval**, the
signature of memorising recordings: per-frame appearance features encode *which recording*
far better than *which verb*, and "pick up" vs "put down" differ only in motion direction,
which per-frame pooling discards. Swapping to a video-native encoder fixed it.

Default backbone is therefore `videomae` (`configs/base.yaml`); pass `--backbone dinov2s`
to reproduce the negative result.

### What the data actually looks like

Established by inspection, not assumed — each of these shaped the code:

| Fact | Consequence |
|---|---|
| Every segment is annotated once **per view** with identical frame numbers | Ego↔exo pairing is an exact join, not an alignment problem |
| Annotation frames are @30fps; source video is 60fps | `seconds = frame/30`, `source_index = 2 × annotation_frame` |
| Official splits are recording- **and** toy-disjoint (211/62/88) | Leak-free train/eval split for free |
| ~226 segments per recording, 24 verbs, median segment 0.93s | 20 recordings ≈ 4,300 paired segments |
| Exo views run 1.6–4.0 GB; ego views ~30 MB | View choice *is* the download budget |
| Segment density varies 15–45 per minute | Ranking by density buys the same data for ~⅓ fewer bytes |

Together the last two cut the download from **40.4 GB to 10.1 GB** with no loss of segments.

#### Ego/exo drift — the one real trap

Ego and exo frame counts are *usually* equal but not always: some views drop frames
uniformly, losing up to ~70 frames over a 28,000-frame session. This is **cumulative
drift, not trailing truncation**. Measured by motion-energy cross-correlation on the worst
recording, the ego/exo lag grows from +13 frames in the first half to **+61 frames
(≈1.0s)** in the second — longer than the median action segment. Left uncorrected, late
clips in an "aligned" pair show *different actions*, silently destroying the cross-view
objective.

Annotations follow the longer, canonical timeline (in some recordings `2 × max(end_frame)`
exceeds the shorter view's frame count while fitting the longer one). So each view is
rescaled by its share of the reference length (`video.timeline_scale`). Verification:

| Recording | Raw lag (1st half → 2nd half) | After rescale |
|---|---|---|
| zero-delta control | −1 → 0 frames | −1 → 0 frames |
| worst case (Δ=69) | +13 → +61 frames | **−2 → 0 frames** |

Cross-view correlation *rises* after the correction (0.43 → 0.47), confirming the drift is
linear. `tests/test_sync.py` guards this.

#### VideoMAE loads without its attention biases

`transformers` expects `query.bias` / `key.bias` / `value.bias`, but VideoMAE checkpoints
store the attention bias as separate `q_bias` / `v_bias` tensors (key bias is structurally
zero). A plain `from_pretrained` therefore drops them and silently re-initialises to zero:
the load *succeeds*, the model runs, and it is quietly missing every learned attention bias
(query-bias magnitude 260.7 across 12 layers). `encoders.restore_videomae_attention_bias`
remaps them, and `tests/test_encoders.py` guards it — nothing else would reveal it.

### Setup

```bash
make setup      # arm64 venv + deps
make external   # clone the annotation repo
```

**Apple Silicon:** the venv must be **arm64**. An x86_64 Python (Intel Homebrew under
Rosetta) cannot install PyTorch at all, and would get no MPS acceleration if it could.
`make setup` prints the architecture; confirm it says `arm64` and `mps True`.

**Dataset access:** videos come from the gated HF dataset
[`cvml-nus/assembly101`](https://huggingface.co/datasets/cvml-nus/assembly101). Accept the
terms once, then `hf auth login`. Annotation CSVs come from
[`assembly101-annotations`](https://github.com/assembly-101/assembly101-annotations).

### Pipeline

```bash
make find       # select recordings -> manifests/{recordings,segments}.csv
make download   # one ego + one exo view each (~10 GB)
make features   # cache frozen VideoMAE features (~30 min)
make test       # integrity + sync checks
make train      # 3 configs x 7 budgets x 5 seeds
make results    # curve + summary CSV
make ui         # interactive per-task page -> ui/label_efficiency.html
```

`make ui` renders `ui/template.html` with the real sweep numbers injected as JSON:
the pooled label-efficiency curve with the "N× fewer labels" read-off, a per-verb
grid of small multiples sorted by improvement, and the pass/fail gate stated up
front. It is monochrome by design — series identity is carried by line style and
direct labels rather than hue, so it survives print, colour-blindness and
forced-colors mode.

`make clips` crops a sample of synchronised ego/exo clips for eyeballing. The ML path does
**not** read them — `04_extract_features.py` decodes each source video once instead, since
seeking once per segment across ~4,300 segments costs hours.

---

### Scope of Stage 3

Human ego/exo representation learning only — no robot control. The point is to make the
representation question clean and measurable first.

**A null result is a real outcome here.** With a frozen backbone at this scale `ego_exo`
may not beat `ego_only`; the control, 5 seeds, and error bars exist so that outcome is
interpretable rather than ambiguous. Read the gate before the curves: a delta measured
below the gate is seed noise, not evidence.

---

## Layout

```
World Context / AssemblyHands analysis
  src/sync_audio.py       onset-based ego/exo synchronization
  src/exo_analysis.py     view-agnostic comparison + threshold sweep
  src/exo_coverage.py     first-pass semantic analysis (superseded; kept for the record)
  src/calib.py            camera models, projection, triangulation
  src/rig_analysis.py     AssemblyHands rig noise-propagation study
  src/view_gap.py         viewpoint label-efficiency gap
  src/pretrain_ablation.py  ego+exo pretraining ablation
  src/auto_label.py       per-view auto-labelling + self-agreement control
  annotate/               ego/exo annotation tool (serve.py provides Range support)
  results/                *.json

Assembly101 label-efficiency pipeline (Stage 3)
  configs/                base.yaml + one file per config
  src/exoego/             annotations, views, video, encoders, heads,
                          objectives, data, train, evaluate
  scripts/                01_find … 07_build_ui, run in order
  tests/                  sync, manifest, feature, and encoder integrity
  ui/template.html        label-efficiency page (rendered by 07_build_ui.py)
```

## Earlier work, still valid

`python -m src.rig_analysis` — on AssemblyHands calibration, an upper bound on
what each rig can achieve given realistic 2D detection error. Monocular ego error
is **flat** across detection noise (205.7 mm at 0 px, 206.1 mm at 4 px) because
the error is scale ambiguity, not detection quality: *you cannot fix a monocular
rig with a better hand detector.* An 8-camera exo rig reaches 1.1 mm.
