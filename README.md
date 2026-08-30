# exoego

**Does exocentric human video reduce the amount of demonstration data needed to learn
from egocentric video?**

Assembly101 records the same assembly sessions from 8 fixed third-person cameras and 4
head-mounted cameras at once. That gives synchronised ego/exo pairs of the *same action*
for free, so exo footage can be used as unlabelled auxiliary signal while the model is
evaluated strictly on ego.

The headline experiment is a **label-efficiency curve**: verb-classification accuracy on
held-out ego clips as a function of how many labelled ego clips the model was given. If
exo helps, the gain should be largest where labels are scarcest.

## The experiment

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

### Go/no-go gate

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

## What the data actually looks like

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

### Ego/exo drift — the one real trap

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

### VideoMAE loads without its attention biases

`transformers` expects `query.bias` / `key.bias` / `value.bias`, but VideoMAE checkpoints
store the attention bias as separate `q_bias` / `v_bias` tensors (key bias is structurally
zero). A plain `from_pretrained` therefore drops them and silently re-initialises to zero:
the load *succeeds*, the model runs, and it is quietly missing every learned attention bias
(query-bias magnitude 260.7 across 12 layers). `encoders.restore_videomae_attention_bias`
remaps them, and `tests/test_encoders.py` guards it — nothing else would reveal it.

## Setup

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

## Pipeline

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

## Layout

```
configs/          base.yaml + one file per config
src/exoego/       annotations, views, video, encoders, heads, objectives, data, train, evaluate
scripts/          01_find … 07_build_ui, run in order
ui/               template.html (design) + label_efficiency.html (generated)
tests/            sync, manifest, and feature integrity
results/          label_efficiency.csv, summary_*.csv, curve png
```

Dataset files live outside git under `~/datasets/egoexo` (override with
`EXOEGO_DATA_ROOT`).

## Scope

Human ego/exo representation learning only — no robot control. The point is to make the
representation question clean and measurable first.

**A null result is a real outcome here.** With a frozen backbone at this scale `ego_exo`
may not beat `ego_only`; the control, 5 seeds, and error bars exist so that outcome is
interpretable rather than ambiguous. Read the gate before the curves: a delta measured
below the gate is seed noise, not evidence.
