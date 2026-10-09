# How to read the evaluation tables

A short guide to what is in `metrics_per_video.csv` and `metrics_pooled.csv`: where the data
comes from, what one row is, and what every column means. It contains no results. For how to run
the evaluation and the technical details, see [README.md](README.md).

Arrows: **↑ higher is better**, **↓ lower is better**, **=** neither (a count or a setting that
describes the row).

## 1. The data

We have two short clips from a portrait 4K camera (2160 x 3840 px, 25 fps), where every visible
bird was labelled by hand, frame by frame, with a box and an identity. That hand labelling is the
**ground truth (GT)**: the right answer the trackers are compared against.

| clip | frames | birds | what makes it hard |
|---|---|---|---|
| `6C42` | 671 | 1 | a single bird near a wind turbine, labelled only in frames 53 to 293; the rest of the clip has no bird, so anything a tracker reports there is wrong |
| `1DAC` | 1009 | 25 | a dense flock, up to 25 birds at once, often almost on top of each other |

The birds are tiny: a typical box is about 8 to 11 px across.

A tracker gets **detections** (boxes per frame, with no identity) and links them over time into
**tracks** (one id per bird). It can only be as good as the detections it is given, so the same
tracker is evaluated on different inputs. The `input` column says which:

| `input` | what the tracker was given | what it tells you |
|---|---|---|
| `gt` | the GT boxes themselves, with their identities removed | **tracking alone**: detection is perfect, so every error is the tracker linking boxes to the wrong bird |
| `yolo` | the real detector's boxes (the production YOLO pipeline) | **the real system**: detector errors (missed birds, extra boxes, one box over several birds) plus tracking errors |
| `cloud` | nothing: these are tracks from Spoor's separate cloud tracker, a different, heavier pipeline | an **external reference** to compare against, scored as it is |

Rows with different inputs are never added together or averaged. Compare experiments within one
input.

## 2. What one row is

Each row describes one tracker output, scored one way:

| column | meaning |
|---|---|
| `experiment` | the tracker configuration (a file in `experiments/configs/`), e.g. `baseline`, `motion`, `k4t20_fused` |
| `aliases` | other configuration names with exactly the same parameters, so the same run (e.g. `baseline` also stands for `original`, `kalman_normal`, `test_association_original`) |
| `input` | `gt`, `yolo` or `cloud`, see above |
| `variant` | which of the tracker's boxes are scored, see below |
| `is_default_variant` | `True` for the variant to look at first (`matched_final`) |
| `similarity`, `s_scheme` | how "is this box on that bird" is decided, see section 3 |
| `is_primary_similarity` | `True` for the main measure (`dotd` with `per_track`) |
| `video` | (per-video table) which clip |
| `aggregation` | (pooled table) `pooled`: all clips' boxes counted together, so the clip with many boxes (1DAC) dominates; `per_video_mean`: the plain average of the clips' scores, each clip weighted equally |
| `n_videos`, `videos` | (pooled table) which clips were combined |

**Variants.** While a tracker runs, each box it writes is one of three kinds: *matched* (the
track found a detection in this frame), *pre-confirmation* (a matched frame from before the track
was accepted as real; tracks need a few matches before they count) or *predicted* (the track had
no detection this frame and the tracker guessed where the bird went).

| `variant` | boxes scored |
|---|---|
| `matched_final` (default) | matched + pre-confirmation boxes of accepted tracks |
| `matched_online` | matched boxes of accepted tracks only (what is known live) |
| `matched_final_all` | like `matched_final`, plus tracks that were never accepted (mostly noise) |
| `predicted` | `matched_online` plus the guessed positions while a track has no detection |
| `frozen` | `matched_online` plus the last seen box repeated while a track has no detection (what the old runner wrote) |
| `legacy` | an old result file, scored as it is (only comparable to `frozen`) |
| `as-is` | the cloud tracker's file, scored as it is |

## 3. Matching a box to a bird

Every metric first decides, frame by frame, which tracker box belongs to which GT bird, based on
a **similarity** between 0 (unrelated) and 1 (identical):

- `dotd` (primary): based on the distance between the box centres, `exp(-distance / s)`. It is
  1 when the centres coincide and falls as they move apart. `s` sets how forgiving it is:
  `per_track` (the size of that bird), `fixed` (9 px for every bird), `per_box` (the size of that
  box). Suits tiny birds, where a 2 px offset is a small error.
- `iou`: how much the two boxes overlap. Standard, but harsh on tiny boxes: 2 px off already
  drops a 9 x 7 px box to 0.38.

A pair counts as a match when the similarity is at least 0.5 (HOTA instead averages over many
thresholds, see below).

## 4. The metrics

Terms used below: a **true positive (TP)** is a tracker box matched to a GT bird; a **false
positive (FP)** is a tracker box with no bird; a **false negative (FN)** is a GT bird with no
tracker box. An **identity switch** is when a bird's matched track id changes.

### HOTA family: the main summary scores

| column | | meaning |
|---|---|---|
| `HOTA` | ↑ | overall tracking quality, 0 to 1: balances finding the birds (`DetA`) and keeping their identities (`AssA`). The single number to look at first |
| `DetA` | ↑ | detection accuracy: how well boxes are found, ignoring identity |
| `DetRe` | ↑ | detection recall: share of GT birds that were found |
| `DetPr` | ↑ | detection precision: share of tracker boxes that are real birds |
| `AssA` | ↑ | association accuracy: how consistently each bird keeps one track, and each track one bird |
| `AssRe` | ↑ | association recall: how much of each bird's life its track covers (low = the bird is split over several tracks) |
| `AssPr` | ↑ | association precision: how much of each track is the same bird (low = a track jumps between birds) |
| `LocA` | ↑ | localisation accuracy: how well matched boxes line up (with `dotd`, how close the centres are) |

All HOTA values are averaged over match thresholds from 0.05 to 0.95.

### Identity: does each bird keep one id over the whole clip?

| column | | meaning |
|---|---|---|
| `IDF1` | ↑ | 0 to 1: each GT bird is paired with one track for the whole clip; this is the share of boxes on the right pair. Very sensitive to id swaps and splits |
| `IDP` | ↑ | identity precision: share of tracker boxes on their paired bird |
| `IDR` | ↑ | identity recall: share of GT boxes covered by their paired track |
| `IDTP` | ↑ | number of boxes on the right pairing |
| `IDFP` | ↓ | number of tracker boxes not on their paired bird |
| `IDFN` | ↓ | number of GT boxes not covered by their paired track |

### CLEAR: classic counts

| column | | meaning |
|---|---|---|
| `MOTA` | ↑ | 1 minus (misses + false boxes + identity switches) / number of GT boxes. 1 is perfect; it goes **negative** when a tracker reports more wrong boxes than there are birds |
| `MOTP` | ↑ | average similarity of matched pairs (how well they line up) |
| `MODA` | ↑ | like MOTA without identity switches |
| `CLR_Re` | ↑ | share of GT boxes matched |
| `CLR_Pr` | ↑ | share of tracker boxes matched |
| `CLR_TP` | ↑ | matched boxes |
| `CLR_FP` | ↓ | tracker boxes with no bird |
| `CLR_FN` | ↓ | birds with no tracker box |
| `IDSW` | ↓ | identity switches. Misses switches between two birds that are almost on top of each other (see `diag_id_changes`) |
| `Frag` | ↓ | times a bird's tracking was interrupted and resumed. Also counts every gap in the hand labelling (a frame where the bird was hidden), so compare it only on the same clip |
| `MT` | ↑ | mostly tracked: birds matched in more than 80% of their frames |
| `PT` | = | partly tracked: birds matched in 20 to 80% of their frames |
| `ML` | ↓ | mostly lost: birds matched in less than 20% of their frames |

### Counts

| column | | meaning |
|---|---|---|
| `n_gt_ids` | = | number of birds in the GT |
| `n_pred_ids` | = | number of track ids the tracker produced. Ideally equal to `n_gt_ids` |
| `id_ratio` | = | `n_pred_ids / n_gt_ids`. 1 is ideal; above 1, birds are split over several ids (or there are false tracks); below 1, birds are missed or merged |
| `n_gt_boxes` | = | number of GT boxes |
| `n_pred_boxes` | = | number of tracker boxes scored |
| `n_frames` | = | number of frames evaluated (every frame of the clip, including frames with no bird) |
| `trackeval_frames` | = | the same count as seen by the CLEAR metric; 0 only if the tracker reported nothing at all |

### Our own diagnostics (not standard metrics)

| column | | meaning |
|---|---|---|
| `diag_id_changes` | ↓ | how many times a track jumps to a different bird. Unlike `IDSW` it also catches jumps between birds that nearly overlap. Can over-count when detections are noisy |
| `diag_predicted_nonpositive_size` | ↓ | guessed boxes (variant `predicted`) whose size went to zero or below: a sign the guess drifted badly. Empty for `legacy` and `cloud` |
| `diag_predicted_centre_outside_frame` | = | guessed boxes whose centre is outside the image, usually a track coasting after its bird flew out of view |

### About the clip (same for every row of a video)

| column | | meaning |
|---|---|---|
| `gt_sqrt_area_median` | = | typical bird size in that clip, in pixels (square root of box area) |
| `gt_sqrt_area_p5`, `gt_sqrt_area_p95` | = | the small and large end of the bird sizes (5th and 95th percentile) |

In `pooled` rows every count is summed over the clips, and ids from different clips are counted
separately: with both clips, `n_gt_ids` is 26 (25 birds in 1DAC + 1 in 6C42). In `per_video_mean`
rows, counts are averages over the clips.

`tracker_csv` (per-video table) is the file that was scored.

## 5. Reading a table in practice

1. Keep one `input` at a time, and start from `is_default_variant` and `is_primary_similarity`.
2. Rank by `HOTA`; then look at `DetA` and `AssA` to see whether the difference comes from
   finding birds or from keeping identities.
3. Check `IDF1`, `IDSW`, `diag_id_changes` and `id_ratio` for identity problems.
4. Look at both `pooled` and `per_video_mean`: if they disagree, one clip is driving the result.
