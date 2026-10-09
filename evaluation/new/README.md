# Tracker evaluation (`evaluation/new/`)

- **What the rows, columns and metrics mean** (no results): [METRICS.md](METRICS.md)
- **Results of a run, with a short analysis**: [reports/](reports/), e.g. [2026-10-09](reports/2026-10-09_results.md)
- **Print a results CSV as readable tables** (display only): `scripts/show_metrics.py`, see below

Scores multi-object bird trackers against hand-labelled ground truth (GT) with
[TrackEval](https://github.com/JonathonLuiten/TrackEval) as the engine: HOTA, Identity (IDF1),
CLEAR (MOTA) and counts, for every tracker output variant, with a centre-distance similarity
(DotD) as the primary measure and IoU alongside. The specification and the decisions behind it
are in [TASK.md](TASK.md).

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r evaluation/new/requirements.txt
```

TrackEval is installed from GitHub at a pinned commit. It still uses numpy aliases removed in
numpy 1.24 (`np.float`, `np.int`, `np.bool`); `birdeval/_numpy_compat.py` restores them as the
Python builtins they always meant before TrackEval is imported, so no result changes. The tests
do the same for py-motmetrics (`np.asfarray`).

## Run

1. Copy `configs/default.yaml` to `configs/local.yaml` (git-ignored) and set the paths: the
   videos, the GT directory, the results roots, the alias file and the external references.
   Relative paths are resolved against the repository root.
2. Evaluate:

```bash
.venv/bin/python evaluation/new/scripts/evaluate.py --config evaluation/new/configs/local.yaml --out evaluation/new/results/run1
```

`--gt`, `--videos` and `--results PATH=v2:gt,yolo PATH=legacy:yolo` override the config.
`--parallel N` turns on TrackEval's parallel mode; TrackEval parallelises over the videos of one
tracker and starts a process pool per tracker, so it only pays off with many videos.
`evaluation/new/results/` and the metadata cache `evaluation/new/cache/` are git-ignored.

To look at a results table without opening the CSV (one table per input; default variant and
primary similarity unless told otherwise, see `--help`):

```bash
.venv/bin/python evaluation/new/scripts/show_metrics.py evaluation/new/results/run1/metrics_pooled.csv
```

Tests (the ones on the real clips are skipped if the data or the re-runs are not available):

```bash
.venv/bin/python -m pytest -q evaluation/new/tests
```

## Inputs

- **GT**: `<gt_dir>/<video>_reference_tracks_corrected.csv` (`frame_number, obj_id, x1, y1, x2,
  y2, bbox_size`, corner boxes). A video is evaluated only if its GT exists; anything else is
  listed as skipped in `evaluate.log` and `run_info.json`.
- **Video**: `<videos_dir>/<video>.mp4`. The frame count (and fps, size) always comes from the
  video, cached in `evaluation/new/cache/`. Every frame is evaluated, including frames with no GT
  and no prediction (6C42 has GT only in frames 53 to 293: a predicted box in frame 10 is a false
  positive).
- **Tracker results**, discovered in every configured root:
  - `v2` layout (labelled runner output, `research/runner/rerun_experiments.py`):
    `<root>/<input>/<experiment>/<CLIP>_tracks.csv` with columns `frame_number, track_id, x, y, w,
    h, row_type, track_confirmed`;
  - `legacy` layout (old runner output): `<root>/<experiment>/<CLIP>[_suffix]_tracks.csv`, columns
    `frame_number, track_id, x, y, w, h`. Empty (header-only) files are failed runs, not scored.
  - A CSV belongs to a video if its name starts with the video stem or its first `_` token is the
    clip code (the last `_` token of the stem, e.g. `1DAC`).
- **External references** (the cloud tracker): corner-format CSVs, evaluated as they are, as their
  own input (`cloud`), never pooled with the kit's experiments.
- **Aliases**: configs whose full parameter set is identical (after the runner's defaults) are run
  once; `experiments/results_v2/aliases.yaml` lists the other names, shown in the `aliases`
  column instead of as duplicate rows.

## Variants: which rows of a tracker run are scored

The runner records each row with a `row_type`: `matched` (the track was matched to a detection;
the box is that detection), `pre_confirmation` (a matched frame before the track had
`tentative_threshold` matches, written afterwards) and `predicted` (a confirmed track alive but
unmatched; the box is the Kalman prediction for that frame).

| variant | rows | what it measures |
|---|---|---|
| `matched_online` | `matched` rows of confirmed tracks | what the tracker reports live, without the frozen boxes of the old output |
| **`matched_final`** (default) | `matched` + `pre_confirmation` rows of confirmed tracks | the final tracks, each including the frames before it was confirmed |
| `matched_final_all` | `matched_final` + never-confirmed tracks | the same, plus the short tracks that never confirmed (mostly noise) |
| `predicted` | `matched_online` + `predicted` rows | also scores where the tracker places a bird while it has no detection |
| `frozen` | `matched_online` + the last matched box repeated while a track coasts | reproduces the old runner's output exactly (tested) |
| `legacy` | an old-format CSV as it is | comparable only to `frozen` |
| `as-is` | an external reference as it is | the cloud tracker |

The default was fixed before any results were seen and is not chosen by score. All variants are
always computed and reported; `is_default_variant` marks the default.

What the default does **not** capture: `matched_*` variants never score where a tracker puts a
bird while it has no detection; that is what `predicted` measures. With GT detections as the
tracker's input, every `predicted` row is a false positive by construction (GT has no box in
those frames, otherwise the tracker would have had a detection there). With real detections a
`predicted` row can be a true positive: a visible bird the detector missed. Identity continuity
across gaps is still scored in every variant, through the track ids: a bird that keeps its id
across a gap scores better than one that gets a new id.

## Similarity

- **DotD** (primary): `exp(-d / s)`, `d` the distance between box centres in pixels. It ignores
  box size and overlap, which suits birds of 6 to 15 px where a 2 px offset already drops IoU to
  0.38. The scale `s`:
  - `per_track` (primary): for each GT track, the median of `sqrt(w * h)` over its boxes, at least
    `s_min` = 4 px. Computed from GT only.
  - `fixed`: one constant, 9 px by default.
  - `per_box`: `sqrt(w * h)` of each GT box, at least `s_min`.
- **IoU**: standard intersection over union. Predicted boxes with non-positive size (a coasting
  Kalman state can shrink below zero) have zero area, so their IoU is 0.

DotD inside HOTA follows the SMOT4SB challenge's SO-HOTA. Only `fixed` with one scale for the
whole dataset is comparable to published SO-HOTA scores; `per_track` and `per_box` are our own
adaptive scales and are not numerically comparable to them. DotD only sees centres: two boxes
with the same centre and different sizes score 1 against each other.

With DotD, **LocA** (HOTA) and **MOTP** (CLEAR) are the mean DotD of matched pairs, not the mean
IoU.

Thresholds, verified in the TrackEval source: CLEAR and Identity count a pair as a match when its
similarity is at least 0.5; HOTA sweeps alpha from 0.05 to 0.95 in steps of 0.05 and reports the
mean over alphas. With DotD, 0.5 means a centre within `s * ln 2` (about 0.69 s) of the GT centre.

## Outputs

- `metrics_per_video.csv`: one row per (experiment, input, variant, similarity, s_scheme, video).
- `metrics_pooled.csv`: per (experiment, input, variant, similarity, s_scheme), two rows:
  `aggregation = pooled` (TrackEval's combined sequence: counts summed over the videos, scores
  computed on the sums) and `aggregation = per_video_mean` (the plain mean of the per-video
  rows, every video weighted equally). Pooled rows are dominated by videos with many boxes.
- `config_used.yaml`, `run_info.json` (git commit, TrackEval version and commit, timestamp,
  command, discovered runs, skipped files), `evaluate.log`.

Columns:

| group | columns |
|---|---|
| identity of the row | `experiment`, `aliases`, `input` (`gt`, `yolo`, `cloud`), `variant`, `is_default_variant`, `similarity`, `s_scheme`, `is_primary_similarity`, `video` or `aggregation`, `n_videos`, `videos` |
| HOTA | `HOTA`, `DetA`, `AssA`, `LocA`, `DetRe`, `DetPr`, `AssRe`, `AssPr` (mean over alpha) |
| Identity | `IDF1`, `IDP`, `IDR`, `IDTP`, `IDFP`, `IDFN` |
| CLEAR | `MOTA`, `MOTP`, `IDSW`, `Frag`, `MT`, `PT`, `ML`, `CLR_TP`, `CLR_FP`, `CLR_FN`, `MODA`, `CLR_Re`, `CLR_Pr` |
| counts | `n_pred_ids`, `n_gt_ids`, `id_ratio` (= pred / GT ids), `n_pred_boxes`, `n_gt_boxes`, `n_frames` (from the video), `trackeval_frames` (CLEAR's own count; 0 if a tracker has no boxes) |
| our diagnostics | `diag_id_changes`, `diag_predicted_nonpositive_size`, `diag_predicted_centre_outside_frame` |
| GT size (per video) | `gt_sqrt_area_median`, `gt_sqrt_area_p5`, `gt_sqrt_area_p95` |

Inputs (`gt` = the tracker was fed the GT boxes as detections, `yolo` = real detections,
`cloud` = the cloud tracker's own output) are never pooled or averaged together.

## Our own diagnostics (not standard metrics)

Columns prefixed `diag_` are ours, not part of any MOT benchmark.

- **`diag_id_changes`**: in every frame, GT and predicted boxes are paired by a plain Hungarian
  matching on the similarity (no preference for last frame's pairs) and pairs below 0.5 are
  dropped; each time a predicted track is paired with a different GT bird than the last time it
  was paired counts as one change. Why it exists: CLEAR's `IDSW` keeps last frame's pairing
  whenever it is still above 0.5, so a track that keeps jumping between two nearly overlapping
  birds is not counted. On 1DAC with GT detections (baseline, `matched_online`) the tracker makes
  79 such jumps, checked exactly by box identity; `diag_id_changes` reports 79 with IoU, while
  `IDSW` reports 17 (DotD) or 32 (IoU). The birds are a median 1.4 px apart at the jump.
  **Caveats**: it can over-count with noisy detections, because a box between two birds, or a
  merged box over several, can flip its best pairing from frame to frame without a real identity
  error; and with DotD, boxes sharing a centre tie (on that same run DotD gives 87, 8 more than
  the exact count: 6 from birds with identical centres, 2 from a frame where two GT birds share
  one box).
- **`diag_predicted_nonpositive_size`**, **`diag_predicted_centre_outside_frame`**: per run and
  video, the number of `predicted` rows whose Kalman box has width or height <= 0, or whose
  centre is outside the frame (a track coasting after its bird left the frame). They describe the
  raw run and are repeated on every variant's row; empty for legacy and external results.

## Known metric behaviour to keep in mind

- **Frag counts GT annotation gaps.** CLEAR forgets that a bird was tracked if the bird has no GT
  box in a frame (occluded or not visible, so not labelled). When it reappears and is matched
  again, CLEAR counts a fragment, even if the same track followed it perfectly. Measured with an
  exact re-implementation of CLEAR's Frag (`birdeval/diagnostics.py`, `FragAttribution`), on the
  baseline experiment with DotD (per_track):

  | clip | GT gaps inside tracks | input | variant | Frag | from GT gaps |
  |---|---|---|---|---|---|
  | 1DAC | 111 | gt | matched_final | 110 | 110 |
  | 1DAC | 111 | gt | matched_online | 116 | 109 |
  | 1DAC | 111 | yolo | matched_final | 345 | 1 |
  | 6C42 | 0 | gt | matched_final | 0 | 0 |
  | 6C42 | 0 | yolo | matched_final | 3 | 0 |

  With GT detections almost all of Frag is this artefact; with real detections almost none is.
  Compare Frag only between runs on the same GT.
- **IDSW undercounts swaps between close birds** (see `diag_id_changes` above).
- **IoU under-matches small or merged boxes.** On 1DAC with YOLO detections, IoU gives IDSW 3 vs
  130 for DotD: the detector's flock boxes are about 2.4 times larger than the birds and often
  cover several, so IoU rarely reaches 0.5 and there are hardly any matches left to switch.
- **MOTA can be negative** when false positives exceed the GT boxes (YOLO input, `predicted`).

## Cross-check with py-motmetrics

`tests/test_motmetrics_crosscheck.py` recomputes IDF1, MOTA, IDSW, FP and FN with py-motmetrics,
using IoU >= 0.5 and, to validate the DotD path, a centre distance <= `s * ln 2` (equivalent to
DotD >= 0.5 for a fixed s). On the toy cases everything agrees exactly. On 1DAC, IDF1, FP and FN
agree exactly and IDSW differs by at most 2: TrackEval continues only pairs from the immediately
previous frame inside one Hungarian that maximises similarity, while motmetrics first keeps each
object's last pairing from any earlier frame and then minimises distance. This is a definitional
difference, documented in the test, not forced to agree.

## Layout

```
birdeval/
  formats.py            readers: GT, labelled and legacy tracker CSVs, corner-format references
  variants.py           derive the variants from a labelled tracker CSV
  similarity.py         IoU and DotD (fixed, per_track, per_box)
  metadata.py           frame count, fps, size from the video, cached
  trackeval_dataset.py  TrackEval dataset class (one timestep per video frame)
  diagnostics.py        our diagnostics as TrackEval metrics
  run.py                evaluate trackers on videos, flatten results
  batch.py              discovery, grouping, result tables
  _numpy_compat.py      numpy aliases for TrackEval
scripts/evaluate.py     CLI
scripts/verify_frozen.py  check a labelled CSV reproduces a legacy CSV exactly
scripts/show_metrics.py   print a results CSV as readable tables (display only)
METRICS.md              what every row, column and metric means
reports/                fixed write-ups of evaluation runs
configs/default.yaml    documented config with placeholder paths
tests/                  toy cases, readers, variants, similarity, motmetrics cross-check
```

Not built yet, on purpose: plots and dashboards, post-processed variants
(`filter_deleted_tracks`, `concatenate_tracks`), GT-free metrics for unlabelled videos.
