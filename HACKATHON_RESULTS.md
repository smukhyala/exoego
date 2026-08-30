# ExoEgo — defensible hackathon result

## One-line result

Synchronized exo/ego pairs teach a representation bridge that lets an
ego-trained task head understand third-person observations from unseen
recordings, while exo geometry supplies metric 3D context unavailable to a
single wearable camera.

## Evidence produced from the local data

### 1. Pairing transfers task signal across viewpoints

`python -m src.paired_bridge`

- Train: 1,710 paired segments from 143 recordings.
- Validation: 511 segments from 39 held-out recordings.
- Test: 875 segments from 68 held-out recordings.
- The bridge uses synchronized features but no action labels.
- The task head uses ego labels only and is frozen for every exo condition.

| Input to the ego-trained head | Top-1 | Mean-per-class |
|---|---:|---:|
| Raw exo | 8.0% | 2.3% |
| Mispaired control | 3.1% | 1.3% |
| **Synchronized-pair bridge** | **14.2%** | **4.2%** |
| Majority baseline | 5.5% | 0.7% |

The paired gain over raw exo is **+6.2 points**. A paired bootstrap over whole
test recordings gives a 95% confidence interval of **+4.2 to +8.3 points**.
Shifting the exo view in time within the same recording removes the benefit,
showing that correspondence—not merely the same worker, toy, or room—is the
useful supervision.

### 2. Exo supplies metric 3D context

`python -m src.g1_check && python -m src.rig_analysis`

- Ego 3D-to-2D reprojection: pass, 0 px median error on 8,734 annotated joints.
- Exo projection/triangulation round-trip: pass, 0 mm numerical error.
- At 2 px simulated hand-detection noise, the 8-camera exo rig has 1.1 mm
  median triangulation error.
- A monocular ego estimate with one oracle global scale per frame still has
  205.9 mm median error on this geometry.
- Even a 2% absolute-depth error in an otherwise oracle monocular estimate
  produces 9.1 mm median error, versus 1.1 mm for exo triangulation.

This establishes the teaching signal exo can provide: task correspondence plus
metric position and scale. The exo cameras are needed during data collection,
not when the learned ego/robot policy is deployed.

## What to say on stage

> We tested the bridge a robot needs before it can learn by watching. On 68
> unseen recordings, synchronized third-person/first-person pairs nearly
> doubled task recognition through an ego-trained head, from 8.0% to 14.2%.
> Deliberately breaking synchronization destroyed the gain. Separately, the
> real camera calibrations show why exo is a useful teacher: it recovers metric
> hand position at roughly millimetre scale, while monocular ego remains depth
> ambiguous. Pair the views during pretraining; discard the wearable camera at
> deployment.

## What this does not prove

Do not claim that one-shot robot imitation is complete. The paired-bridge source
features were already supervised for Assembly101 action recognition, and the
rig result propagates synthetic pixel noise through real calibration rather
than running a learned hand detector. The result proves two prerequisites:
paired cross-view transfer and a metric exo teaching signal. A raw-video
self-supervised world model plus robot task-success evaluation remains the next
experiment.

## Demo artifacts

- `results/paired_bridge.png` — primary chart.
- `results/paired_bridge.json` — metrics and confidence intervals.
- `results/rig_analysis.json` — camera-geometry results.
- `src/paired_bridge.py` — fully reproducible paired/mispaired experiment.
