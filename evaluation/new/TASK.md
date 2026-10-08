# Task: build the tracker evaluation module

You are working in the `bird-tracking` repo. Your job is to build a metrics module that
evaluates multi-object bird trackers against hand-labelled ground truth (GT), and to make a
small change to the tracker runner so its output can be evaluated correctly.

Work step by step. At each CHECKPOINT below, stop, summarise what you did and what you found,
and wait for my confirmation before continuing. Do not make product decisions on your own: if
something is unclear or you find that an assumption in this file is wrong, stop and ask. The
DECISIONS block below is final unless I change it.

Do not modify any file under `spoor_tracker/tracker-kit/tracker/`: those are verbatim copies of
production code (see `spoor_tracker/tracker-kit/PROVENANCE.md`). The only tracker-kit file you
may change is `spoor_tracker/tracker-kit/run_tracker.py`.

---

## DECISIONS (set by me, do not change)

| topic | decision |
|---|---|
| Similarity measures | DotD (centre distance) as primary, IoU alongside. Both always computed. |
| DotD scale s | `per_track` (primary): median of sqrt(box area) over each GT track, with floor `s_min = 4` px. Also support `fixed` (a constant, default 9 px) and `per_box` (with the same floor) as config options, reported alongside for comparison. |
| Metrics | HOTA family (HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr), Identity (IDF1, IDP, IDR), CLEAR (MOTA, MOTP, IDSW, Frag, MT, PT, ML, FP, FN, and the raw counts), Count (number of predicted IDs, GT IDs, and their ratio). |
| Aggregation | Both: pooled over all videos (TrackEval's combined sequence) and the mean of per-video scores. |
| Runner output | Record everything with a `row_type` column (see section 4). The evaluator derives output variants. |
| Default evaluation variant | `matched_final`: matched rows only, including pre-confirmation frames, confirmed tracks only. All other variants are also computed and reported, never hidden. The default was fixed before seeing any results and must not be changed based on which variant scores best. |
| Existing results | Every `experiments/results/<name>/` folder contains a tracker CSV in the legacy format, which includes frozen rows. They are evaluated only as the `legacy` variant. Fair comparison between experiments requires re-running them with the new runner (section 4). |
| Post-processing | Not now. Evaluate the raw tracker output. Leave room for a post-processed variant later. |
| Library | TrackEval (JonathonLuiten/TrackEval) as the engine, with a custom dataset class. py-motmetrics only as an independent cross-check in the tests. |
| Spoor's own `evaluate_track_quality` | Not involved. Do not import or reproduce it. |
| Frame indexing | 0-based everywhere internally. The frame count comes from the video file, never from a CSV. |

---

## 1. Context

### The project
Spoor's bird tracker, isolated as a kit in `spoor_tracker/tracker-kit/`. Read these first:

- `spoor_tracker/tracker-kit/README.md` (what the tracker does, frame by frame)
- `spoor_tracker/tracker-kit/PROVENANCE.md`
- `spoor_tracker/tracker-kit/run_tracker.py` (the runner you will change)
- `spoor_tracker/tracker-kit/tracker/simple_sort_tracker.py` and `tracked_object.py`
- `data/README.md`

Two clips have hand-labelled GT (annotated in Supervisely, exported to CSV):

- `20250920_063942_6C42`: 671 frames, one bird, labelled in frames 53 to 293 only.
- `20251012_164031_1DAC`: 1009 frames, 25 birds, up to 25 per frame, labelled in every frame.

Both are 2160x3840 (portrait 4K), 25 fps. Birds are tiny here (median box about 12x10 px and
9x7 px) but the module must work for thousands of videos with birds from 6 px to 300 px anywhere
in the frame. Nothing may be hard-coded to these two clips.

### GT annotation conventions (fixed, do not question)
- A GT box exists only in frames where the bird is visible. Occluded or unseen frames are
  skipped, so a GT track can have short gaps.
- No ignore regions. Every visible bird anywhere in the frame is labelled.

### Findings you must take into account
1. **The runner writes frozen boxes.** `run_tracker.py` writes every CONFIRMED track that is
   still alive, using `tracked_object.detections[-1].bounding_box`. After a bird's last
   detection the track coasts for up to `max_age - 1` (29) frames and the runner repeats the same
   box each frame. In the 1DAC run on GT detections, 815 of 10906 rows are such rows. Against
   this GT they are false positives. This is a runner choice, not tracker behaviour:
   `TrackedObject.detections` only ever contains matched detections.
2. **Pre-confirmation frames are never written.** A track is written only once it is CONFIRMED
   (after `tentative_threshold` = 3 matches), so its first 2 matched frames are missing.
3. **Never-confirmed tracks are never written.** They end up in `deleted_tracked_objects` and are
   never read.
4. **The alignment check in `run_tracker.py` refuses GT detections that do not start at frame
   0** (6C42 starts at 53). The check must instead read the frame count from the video and only
   refuse when a CSV frame number is outside `0 .. frame_count - 1`.
5. **The 1DAC run on GT detections has about 79 identity jumps** between GT birds even though it
   looks perfect in the annotated video (for example track 3 at frames 89 to 101, track 22 at
   frames 467 to 518). Your metrics must make this visible.
6. **IoU is unsuitable as the only similarity for these birds.** A 9x7 box shifted 2 px in x and
   y has IoU 0.38. This is why DotD is the primary similarity.

---

## 2. File formats

### GT tracks (input), one file per video
`data/<video>_reference_tracks_corrected.csv`, columns:
`frame_number, obj_id, x1, y1, x2, y2, bbox_size`
- 0-based frame numbers, matching the cut `.mp4` in `data/`.
- Corner format. `w = x2 - x1`, `h = y2 - y1`, `bbox_size = w * h` (verified).
- Windows line endings. One frame has two GT objects with an identical box; treat it as data.
- There may be no GT at all in some frames (6C42 frames 0 to 52 and 294 to 670). Those frames
  still exist and must be evaluated as empty frames.

### GT detections (tracker input), one file per video
`data/<video>_detections_corrected.csv`, columns:
`frame_number, frame_timestamp, x, y, w, h, area, classifier_name`
Same boxes as the GT tracks without IDs. Only needed to run the tracker.

### Tracker output (current)
`frame_number, track_id, x, y, w, h` (floats, top-left corner, 0-based frames). You will extend
this, see section 4. Keep the old format readable by the evaluator too, but **do not** treat its
rows as matched: a legacy file contains frozen rows that cannot be told apart reliably from real
matches (a bird can hold still), so legacy files are evaluated only as the `legacy` variant.

### Experiment results
`experiments/results/<experiment_name>/` folders exist already (baseline, motion, appearance,
kalman_4d_..., etc.), and each contains a tracker CSV in the legacy format. Experiment configs
are in `experiments/configs/`. Before writing any discovery code, inspect them and tell me at
CHECKPOINT 1:
- how a result CSV is named and how to tell which video it belongs to;
- which config produced each result folder, and whether every folder can be re-run from its
  config with the new runner;
- what is in `cloud_reference/` (this is the cloud tracker, not the kit's runner, and may have a
  different format such as `obj_id, x1, y1, x2, y2`), and in `test_original_GT1/` and
  `test_association_original/`;
- what `experiments/results/notes.md` says that matters here.
Then propose how the evaluator maps a result folder to (experiment, video, tracker CSV, config).

---

## 3. Where the code goes

Put the evaluation module in `evaluation/new/`. Proposed layout (adjust only if you find a reason
and tell me):

```
evaluation/new/
  README.md                  what this is, how to run it, how to read the output
  pyproject.toml or requirements.txt
  configs/
    default.yaml             similarity + s scheme + metrics + variants + paths
  birdeval/
    __init__.py
    formats.py               readers for GT tracks, tracker CSV (old and new), video metadata
    variants.py              derive evaluation variants from a labelled tracker CSV
    similarity.py            DotD (fixed, per_track, per_box) and IoU
    trackeval_dataset.py     custom TrackEval dataset class
    run.py                   evaluate one (video, tracker CSV) pair
    batch.py                 discover and evaluate many; write result tables
    metadata.py              frame count, fps, width, height from the video (cached to json)
  tests/
    test_formats.py
    test_variants.py
    test_similarity.py
    test_metrics_toy.py      hand-made cases with known answers
    test_motmetrics_crosscheck.py
    fixtures/                tiny synthetic GT and tracker CSVs
  scripts/
    evaluate.py              CLI entry point
```

Keep `evaluation/spoor/` untouched.

---

## 4. Runner change (`run_tracker.py`)

Add a new output format while keeping the old one available behind `--legacy-output`.

New columns: `frame_number, track_id, x, y, w, h, row_type, track_confirmed`

`row_type` is one of:
- `matched`: the track was matched to a detection in this frame; the box is that detection.
  This is exactly what `TrackedObject.detections` contains.
- `pre_confirmation`: a matched row from before the track reached `tentative_threshold`
  matches. Written retroactively, from the track's own `detections` list.
- `predicted`: the track was alive and confirmed but unmatched in this frame; the box is the
  Kalman-predicted box **for this frame**. Careful: after `match_and_track` returns, the Kalman
  state has already been predicted for the *next* frame, so the prediction for frame f must be
  read from `kalman_tracker_state.to_tlwh()` *before* calling `match_and_track` for frame f.

`track_confirmed` is True if the track ever reached CONFIRMED. Never-confirmed tracks are
written too (all their rows are `pre_confirmation`, `track_confirmed = False`).

Do not write frozen rows. The evaluator reconstructs them from the last `matched` row when it
needs the `frozen` variant.

Simplest implementation: collect `predicted` rows during the loop, and build `matched` and
`pre_confirmation` rows after `finalize_tracking()` from `deleted_tracked_objects` (every track
ends up there). Keep the annotated-video behaviour unchanged.

Also fix the alignment check as described in finding 4.

**Regression check (required):** run the new runner on the 1DAC GT detections with the same
parameters as an existing legacy 1DAC result (I will tell you which one at CHECKPOINT 1). Show
that the `frozen` variant rebuilt from the new output is identical to that legacy CSV (same
frame numbers, track ids and boxes), and that the `matched_online` variant equals the legacy
CSV with its frozen rows removed. Report the row counts of every variant.

**Re-running experiments:** after the regression check passes, propose (do not run yet) how to
re-run every experiment whose config is available with the new runner, writing the new CSVs next
to the legacy ones without overwriting them.

CHECKPOINT 1: after reading the code and data, inspecting `experiments/results/`, and proposing
the result-folder mapping, but before writing code.

CHECKPOINT 2: after the runner change and the regression check.

---

## 5. Evaluation variants (`variants.py`)

From one new-format tracker CSV derive these, each a plain (frame, id, box) table:

| variant | rows included |
|---|---|
| `matched_online` | `matched` rows of confirmed tracks (today's output without frozen rows) |
| `matched_final` | `matched` + `pre_confirmation` rows of confirmed tracks (**default**) |
| `matched_final_all` | as `matched_final` plus never-confirmed tracks |
| `predicted` | `matched_online` + `predicted` rows |
| `frozen` | `matched_online` + reconstructed frozen rows: in every frame where a confirmed track has a `predicted` row, repeat its last matched box instead. Must reproduce today's legacy output exactly; test this. |
| `legacy` | an old-format CSV as it is, no derivation. Comparable only to `frozen`. |

The README must explain, in plain language, what each variant measures and what the default
does **not** capture. In particular: `matched_*` variants do not score where a tracker places a
bird while it has no detection (that is what `predicted` measures); with GT detections as input
every `predicted` row is a false positive by construction, because GT has no box in those frames,
but with real detections a `predicted` row can be a true positive (a visible bird the detector
missed). Identity continuity across gaps is still scored in every variant through the track ids.

All variants are evaluated and reported; `matched_final` is marked as the default in outputs.

---

## 6. Similarity (`similarity.py`)

For a frame with GT boxes G and predicted boxes P, return a |G| x |P| matrix in [0, 1].

- `iou`: standard IoU on `x, y, w, h` boxes.
- `dotd`: `exp(-d / s)` where `d` is the Euclidean distance between box centres and `s` is the
  scale:
  - `fixed`: a constant from the config (default 9.0 px).
  - `per_track`: for each GT track, `s = max(s_min, median over its frames of sqrt(w * h))`.
    Computed from GT only, once per video. Row i of the matrix uses the s of GT box i's track.
  - `per_box`: `s = max(s_min, sqrt(w * h))` of GT box i.
  - `s_min` default 4.0 px.
  This follows the SMOT4SB challenge's "SO-HOTA" idea (DotD inside HOTA). Note that per_track and
  per_box are our own adaptive variants and are not numerically comparable to published SO-HOTA
  scores; `fixed` with a dataset-wide s is the comparable one. Say this in the README.

---

## 7. TrackEval integration (`trackeval_dataset.py`)

- Install TrackEval from GitHub (`pip install git+https://github.com/JonathonLuiten/TrackEval`).
  It is old. If it fails on recent numpy because of removed aliases (`np.float`, `np.int`,
  `np.bool`), apply the minimal fix (a pinned fork, a patch applied at install, or a small
  monkeypatch in our package); tell me which you chose and why at CHECKPOINT 3.
- Write one dataset class subclassing TrackEval's `_BaseDataset` that:
  - loads our internal (frame, id, box) tables for GT and tracker, with the video frame count
    from `metadata.py`, so empty frames are counted;
  - implements `_calculate_similarities` by calling our `similarity.py` with the configured
    measure and s scheme;
  - works for a list of sequences (videos) and a list of trackers (experiment variants) in one
    `Evaluator.evaluate` call, so TrackEval produces the pooled `COMBINED_SEQ` result itself.
- Metrics: TrackEval's `HOTA`, `CLEAR`, `Identity`, `Count`. Before relying on it, verify in the
  TrackEval source that CLEAR and Identity threshold the similarity matrix at 0.5 and that HOTA
  sweeps alpha from 0.05 to 0.95; report what you find. With DotD, `LocA` and `MOTP` become
  mean DotD of matched pairs rather than mean IoU; state this in the README.
- Confidence scores are not used by these metrics. Do not invent a confidence column.
- Per-video mean aggregation is computed by our code from the per-sequence results, not by
  TrackEval.

CHECKPOINT 3: after the dataset class runs end to end on 1DAC with the default config, before
the batch tooling and the tests. Show me the full metric table for 1DAC for all variants and
both similarities, and point at where the 79 identity jumps show up (IDSW, AssA, IDF1).

---

## 8. Batch run and outputs (`batch.py`, `scripts/evaluate.py`)

- `python evaluation/new/scripts/evaluate.py --config evaluation/new/configs/default.yaml
  --results experiments/results --gt data --out evaluation/new/results/<timestamp>/`
- Discovery follows the mapping agreed at CHECKPOINT 1. A video is evaluated only if its GT file
  exists; skipped videos are listed in the log, not silently ignored.
- Outputs:
  - `metrics_per_video.csv`: one row per (experiment, variant, similarity, s_scheme, video), all
    metrics as columns, plus `n_frames`, `n_gt_ids`, `n_gt_boxes`, `n_pred_ids`, `n_pred_boxes`,
    and the GT size statistics of the video (median sqrt area, 5th and 95th percentile).
  - `metrics_pooled.csv`: same columns, one row per (experiment, variant, similarity, s_scheme)
    with `aggregation = pooled` and another with `aggregation = per_video_mean`.
  - `config_used.yaml` and `run_info.json` (git commit, TrackEval version, timestamp, command).
- This has to scale to thousands of videos and many experiments: evaluate videos independently,
  cache video metadata, allow TrackEval's parallel option, and never load more than one video's
  tables at a time outside TrackEval. Do not build any plotting or dashboard yet.

---

## 9. Tests

- Toy cases in `tests/fixtures/` with hand-computed expected values (write the expected values
  in the test and explain in a comment how they were computed): one perfect track; one GT bird
  covered by two predicted tracks (fragmentation, 1 IDSW); two GT birds whose predicted ids swap
  mid-way (2 IDSW); a predicted track with no GT (FP); a GT track with no prediction (FN); empty
  frames at the start and end of a sequence.
- `test_variants.py`: the `frozen` variant rebuilt from the new runner output equals the legacy
  output on the 1DAC data.
- `test_motmetrics_crosscheck.py`: with py-motmetrics, compute IDF1, MOTA, IDSW on the toy
  cases and on 1DAC using IoU at 0.5, and compare with TrackEval's numbers. Document any
  definitional differences you find rather than forcing them to agree. Also run the cross-check
  with a centre-distance metric to validate the DotD path.
- A test that the per_track s is computed from GT only and respects `s_min`.

CHECKPOINT 4: all tests passing, batch run completed on the two clips over every result folder
that has GT, README written. Summarise the results and anything that surprised you.

---

## 10. Things explicitly out of scope now (do not build)

- Plots, dashboards, or any comparison UI.
- Post-processing (`filter_deleted_tracks`, `concatenate_tracks`) variants.
- GT-free proxy metrics for unlabelled videos.
- Tracker parameter tuning. Do not touch tracker parameters or defaults.
- Supervisely export conversion (the GT CSVs are already produced).