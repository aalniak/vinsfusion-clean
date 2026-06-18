# XFeat + LighterGlue front-end for VINS-Fusion — current status

Mono-inertial VIO front-end on **Jetson AGX Orin 64GB** (JetPack 6 / L4T r36.4, CUDA 12.6,
**TensorRT 10.3** FP16). Goal: use **XFeat** (learned detector/descriptor) and **LighterGlue**
(learned matcher) to make feature tracking more robust than classical KLT, in real time, without
sacrificing the long, continuous tracks that triangulation/depth need.

This file is the single source of truth for what exists, what was measured, the parameters, and the
candidate next steps. Loop closure is **out of scope** (tracking only).

---

## 1. What was built

The tracker layer (`featureTracker/feature_tracker.h`) dispatches to `FeatureTrackerKLT`. The XFeat +
LighterGlue work is integrated **inside** `FeatureTrackerKLT` (not a separate tracker), so it composes
with the existing KLT geometry/IMU/back-end plumbing. Everything is gated behind `xfeat_*` params and
is a no-op when disabled.

New TensorRT wrappers:
- `featureTracker/xfeat_trt.{h,cpp}` — XFeat sparse extractor (image → keypoints, scores, 64-D
  L2-normalized descriptors). Resizes to engine size, rescales keypoints back to image coords.
- `featureTracker/lighterglue_trt.{h,cpp}` — LighterGlue matcher (kpts0/desc0/kpts1/desc1 →
  matches0/mscores0). MNN-cosine fallback path also available.
- `featureTracker/feature_tracker_xfeat.{h,cpp}` — standalone match-based tracker. **Parked** (compiles,
  not routed); kept for reference / future relocalization.

Mechanisms added to `FeatureTrackerKLT` (each behind a param, both CPU `trackImage` and CUDA
`trackImageCUDA` paths):
1. **Hybrid detect+track** — new features are seeded from XFeat keypoints (score-sorted, respecting the
   `min_dist` mask) instead of Shi-Tomasi; KLT optical flow then tracks them. `detectNewFeatures()`.
2. **Descriptor-consistency cleaning** — drop tracks whose appearance jumped (KLT silently snapped to a
   wrong feature). `cleanDriftedTracks()` + rolling reference `ref_desc_`.
3. **Track recovery** — revive KLT-lost tracks from a nearby confident LighterGlue match.
   `recoverLostTracks()`. (Parked, off.)
4. **Guided KLT init** — LighterGlue prev↔cur matches → motion model → KLT initial-flow guess.
   `extractAndGuide()` + `computeGuidedPrediction()`. (Parked, off.)

LighterGlue is only run when recovery or guided-init is enabled; the default path (hybrid ± cleaning) is
**XFeat-only** and runs no matcher.

---

## 2. What worked (and what didn't)

Benchmarks: m3ed Spot sequences + EuRoC MH_04, mono+IMU, no loop closure, ATE RMSE (m) via `evo_ape`
SE3-aligned. GPU pinned to 1.3 GHz (`jetson_clocks`). VINS is near-deterministic (std ≈ 0 on most
sequences; variance only near estimator failure), so single-run numbers are reliable.

### ✅ Hybrid (XFeat-detect + KLT-track) — THE win, shipped ON
Replacing Shi-Tomasi with XFeat seeds fixed the original short-track problem (mean track length
2.0 → 9.2; ≥10-frame tracks 2% → 23%, matching KLT) and beats KLT on hard/degraded scenes.

Equal-density benchmark (XFeat & KLT both `max_cnt=300/min_dist=20`), 8 sequences:

| sequence | XFeat | KLT | winner |
|---|---|---|---|
| forest_hard   | 0.451 | 0.683 | XFeat −34% |
| forest_road   | 0.517 | 0.650 | XFeat −20% |
| building_loop | 0.324 | 0.409 | XFeat −21% |
| stairwell     | 0.347 | 0.425 | XFeat −18% |
| stairs        | 0.176 | 0.120 | KLT −32% |
| penno_day     | 0.888 | 0.767 | KLT −14% |
| skatepark     | 0.164 | 0.163 | tie |
| EuRoC MH_04   | 0.174 | 0.149 | KLT −14% |

**Pattern:** XFeat-hybrid wins on repetitive/natural texture + motion blur + degraded conditions
(forest, building loop, stairwell); KLT wins where there is strong corner structure (stairs) or easy
lighting (penno day, EuRoC). XFeat-hybrid is the better **default for hard conditions** and is notably
more stable (KLT diverges on stairs/forest_road when starved to `max_cnt=150`).

### ✅ Descriptor-consistency cleaning — the one refinement worth shipping (opt-in)
Drop a track when its current XFeat descriptor's cosine similarity to a **rolling** reference collapses
(= sudden wrong-feature jump). Rolling (not birth-anchor) is essential: it tolerates slow legitimate
appearance change so it never kills the long tracks VIO depends on.

A/B (clean off vs on; hybrid on, recovery off; `max_cnt=300`), at the validated threshold **0.7**:

| sequence | off | on | Δ |
|---|---|---|---|
| forest_hard   | 0.503 | 0.430 | **−14.4%** |
| forest_road   | 0.624 | 0.549 | **−12.1%** |
| building_loop | 0.356 | 0.324 | **−8.9%** |
| stairs        | 0.080 | 0.081 | +1.5% (neutral, +1 mm) |

3 wins / 1 neutral / **0 regressions / 0 divergence**, ~−0.18 m net. Threshold is the quantity↔quality
dial: **0.7** drops only unambiguous garbage (cos 0.28–0.6, ~0.1–0.2% of tracks) and helps every regime;
0.8 also strips borderline tracks and regressed quantity-limited forest_road (+32%). Sub-millisecond
(reuses the per-frame XFeat extraction, no extra TRT call).

### ⚖️ Track recovery — scene-dependent, parked OFF
Best variant (`v2`, flow-validated borrowed displacement) nets positive in absolute ATE (forest_road
−34% / −21 cm) but loses the stairs win and is scene-dependent. Kept behind `xfeat_recover` as the best
implementation if revisited. (Lesson: never override KLT geometry with an absolute match position — a
prior "snap to match destination" variant injected px-level error and diverged. Borrow the *displacement*
`q + (dst − src)`, gated by local flow consistency.)

### ❌ Guided KLT init — wash, parked OFF
Pyramidal KLT-from-prev already handles inter-frame displacement in these sequences, so the guess mostly
adds noise except in strong-parallax scenes. Global-homography mode hurts badly; per-point mode is ~wash.
Possible salvage: gate to large-displacement frames only.

### ❌ Things that hurt (reverted / removed)
- **cornerSubPix on XFeat seeds** (`xfeat_subpix`) — relocates learned-repeatable points onto classical
  corners; default OFF.
- **rejectWithF** (RANSAC-F outlier rejection) — 1px F-threshold culls good tracks under near-planar /
  forward motion; call removed.
- **Birth-anchor cleaning** — punishes legitimate long-track appearance evolution → kills the highest-
  value long tracks → diverged stairs to >11 km. Replaced by the rolling reference.

### Meta-lesson
The learned front-end's value is as a **detector** (and as a descriptor source for spotting unambiguous
track failures), **not** as a matching-based override of KLT. KLT is already a strong frame-to-frame
tracker; second-guessing it with matches (guided-init, recovery) is wash-to-harmful, while conservative,
descriptor-based *removal* of certain-garbage is a clean, free win.

---

## 3. Performance (FPS)

`trtexec`, TRT 10.3 FP16, batch 1, GPU pinned 1.3 GHz, includes H2D/D2H:

| engine | resolution | latency (mean) | throughput |
|---|---|---|---|
| XFeat extractor        | 1280×800 (m3ed)  | 7.82 ms | **132 FPS** |
| XFeat extractor        | 736×480 (EuRoC)  | 2.86 ms | **361 FPS** |
| LighterGlue (1024 kpt) | 1280×800         | 3.03 ms | **333 FPS** |
| LighterGlue (1024 kpt) | 752×480          | 3.06 ms | **331 FPS** |

LighterGlue is ~3 ms regardless of resolution (fixed 1024×1024 attention). XFeat scales with pixels.

- **Default path (XFeat only):** ~7.8 ms @1280×800 (~128 FPS) / ~2.9 ms @736×480 (~350 FPS).
- **+ LighterGlue** (only if recover/guided enabled): ~10.9 ms / ~5.9 ms.
- Camera rate is 20–25 Hz → ≥3.7× margin even on the heaviest path. **Firmly real-time.** Cleaning adds
  sub-ms. Matches the ~8 ms/frame seen in live `vins_node` logs.

**Critical:** GPU DVFS dominates (≈3.3× idle→pinned). Run `sudo jetson_clocks` before benchmarking.

---

## 4. Parameters (`parameters.{h,cpp}`, set in YAML)

| param | default | meaning |
|---|---|---|
| `xfeat_enable` | 0 | **Hybrid: seed new features from XFeat, track with KLT. The win — set 1.** |
| `xfeat_engine_path` | "" | XFeat extractor FP16 engine (per camera resolution) |
| `xfeat_lighterglue_engine_path` | "" | LighterGlue matcher engine (only needed for recover/guided) |
| `xfeat_matcher` | 0 | 0: LighterGlue, 1: MNN cosine fallback |
| `xfeat_min_conf` | 0.10 | min match confidence to keep/propagate a track |
| `xfeat_score_thr` | 0.05 | min XFeat keypoint score to seed a new feature |
| `xfeat_subpix` | 0 | cornerSubPix-refine seeds — **hurts, leave 0** |
| `xfeat_clean` | 0 | **Descriptor cleaning: drop appearance-jumped tracks. Clean win — set 1.** |
| `xfeat_clean_thr` | 0.7 | min per-frame cosine vs rolling reference (validated operating point) |
| `xfeat_clean_radius` | 5.0 | px: nearest-keypoint proxy radius |
| `xfeat_recover` | 0 | revive KLT-lost tracks via LighterGlue match — scene-dependent, parked |
| `xfeat_recover_radius` | 30.0 | px: search radius for the nearest confident match |
| `xfeat_recover_flow_tol` | 3.0 | px: max neighbour-flow disagreement before a borrow is rejected |
| `xfeat_recover_max_ratio` | 0.5 | cap recovered tracks at this fraction of live tracks per frame |
| `xfeat_guided_init` | 0 | 1: homography / 2: per-point → KLT init guess — wash, parked |

**Recommended shipping config** (XFeat hybrid + cleaning):
```yaml
xfeat_enable: 1
xfeat_engine_path: "/datasets/xfeat_<W>x<H>_fp16.engine"
xfeat_clean: 1            # opt-in quality win
xfeat_clean_thr: 0.7
xfeat_recover: 0
xfeat_guided_init: 0
```

**Operating-point note:** XFeat's benchmark wins above were measured at `max_cnt=300 / min_dist=20`. The
checked-in base configs ship at `max_cnt=150 / min_dist=30` (the permanent CPU-tracking change:
`use_depth:0`, `use_cuda_in_optimization:0`, `use_cuda_in_tracking:0`, `max_cnt:150`, `min_dist:30`,
`max_solver_time:0.04`, `max_num_iterations:6`). To reproduce the XFeat wins set `max_cnt=300`,
`min_dist=20`. Density is the main accuracy lever for both trackers.

---

## 5. Build & run

- Engines live in `engines/` (host) → mounted at `/datasets` in the container. **Engines are TRT
  patch-version locked** (built with the container's `/usr/bin/trtexec` = 10.3.0.26; host apt 10.3.0.30
  is incompatible — "expecting 10.3.0.26"). Rebuild per camera resolution (kpt normalization is baked at
  export).
- Build: `catkin build vins` in `ros:vins-depth-trt-cu-pose`, workspace `/home/nvidia/.ws/vinsfusion`
  mounted at `/root/catkin_ws`, this repo at `src/VINS-Fusion`.
- Export pipeline: `../export/` (`export_xfeat.py`, `export_lighterglue.py`) in the `xfeat-export:latest`
  image. XFeat recall@1.5px 0.958; LighterGlue 86% RANSAC inliers vs MNN's 67%.

---

## 6. What to implement next (candidate ideas)

Organized by the strength each plays to. Highest-leverage bets first within each group.

### Leveraging LighterGlue (wide-baseline / hard matching)
1. **Keyframe-to-keyframe long-range matching** *(top bet)* — KLT does frame-to-frame; separately match
   the current frame against the **last keyframe** (wide baseline) with LighterGlue to re-establish
   large-parallax correspondences KLT dropped. Plays to LighterGlue's strength instead of duelling KLT;
   restores exactly the observations triangulation wants.
2. **Guided KLT init** — LighterGlue prev↔cur → motion model → KLT initial-flow guess, **gated to
   large-displacement frames only** (un-gated it's a wash; see §2).
3. **Track recovery** — already built (`xfeat_recover` v2); revisit with keyframe matching instead of
   frame-to-frame.
4. **Mutual-consistency outlier rejection** — use LighterGlue mutual matches as a geometry-free outlier
   filter in place of RANSAC-F (which hurt).
5. **Multi-hypothesis matching** for repetitive texture — keep top-2 candidates, let IMU/geometry
   disambiguate.

### Leveraging XFeat (detector / descriptor — ~free, already runs each frame)
6. **Semi-dense (XFeat-star) descriptor tracking** *(biggest effort, biggest payoff)* — export XFeat's
   **dense** descriptor map and track each feature by sampling at its predicted sub-pixel location +
   local match. Does recovery *and* cleaning at the exact tracked point (no sparse nearest-keypoint
   proxy — removes the proxy noise that limited current cleaning/recovery). Fully learned long tracks.
7. **Dynamic / non-rigid masking** — flag features whose descriptor *and* epipolar residual are jointly
   inconsistent with the rigid-scene hypothesis (moving foliage/water/crowds) and downweight. Relevant
   to forest sequences.
8. **Adaptive detection density** — use XFeat's score-map as a live texture signal to modulate
   `max_cnt`/`min_dist` per frame.
9. **Learned sub-pixel refinement** — use XFeat's match-refinement head (not cornerSubPix, which hurt).

### Feeding the back-end softly (turn learned signals into estimator inputs)
10. **Match-confidence-weighted features** *(clean, low-risk)* — pass XFeat score / LighterGlue
    confidence into the optimizer as a per-feature information weight, so it trusts confident, well-
    localized points more — a soft alternative to hard keep/drop.
11. **Split-track merging** — detect when a re-seeded new ID is the same physical landmark as a lost
    track (descriptor match) and merge → longer effective tracks.
12. **IMU-predicted match gating** — restrict matching to the IMU-predicted region for robustness under
    fast motion.

**Deferred / blocked-on-export:** #6 (semi-dense) is the principled enabler for several others — it
removes the sparse-proxy limitation that capped both cleaning and recovery.

---

## 7. Repository map (this change)

Modified: `CMakeLists.txt`, `estimator/parameters.{h,cpp}`,
`featureTracker/{feature_tracker.h, feature_tracker_klt.{h,cpp}}`, + 12 base configs
(EuRoC + m3ed, permanent CPU-tracking operating point).
New: `featureTracker/{xfeat_trt, lighterglue_trt, feature_tracker_xfeat}.{h,cpp}`.

A/B experiment configs (`config/m3ed/{rec,gi,clean}_*.yaml`) are throwaway scaffolding and are **not**
committed; regenerate from the base configs if re-validating.
