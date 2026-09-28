# data/

The two real recordings this kit was designed for have landed, and the detection CSVs are now **genuine
per-frame detector output** (previously a cloud-tracks fallback -- see the run notes below):

```
data/
  20250920_063942_6C42.mp4                       bird-and-turbine clip, cut to frames 2000-2670 of the original (671 frames)
  20250920_063942_6C42_detections.csv            genuine birdwatcher rgb_yolo detections for this clip (classifier_name = crop-classifier)
  20250920_063942_6C42_cloud_reference_tracks.csv  stored cloud tracker output for the same window, NOT fed to the tracker
  20251012_164031_1DAC.mp4                       dense-flock clip, cut to frames 4001-5009 of the original (1009 frames)
  20251012_164031_1DAC_detections.csv            genuine birdwatcher rgb_yolo detections for this clip (classifier_name = crop-classifier)
  20251012_164031_1DAC_cloud_reference_tracks.csv  stored cloud tracker output for the same window, NOT fed to the tracker
```

Both originals are **2160x3840 -- portrait 4K, taller than it is wide**, 25fps, ~300s (7500-7502 frames), `.mkv`
containers. The portrait orientation is not a typo and it matters when you read the coordinates: `x` runs 0..2159
and `y` runs 0..3839, so a bird crossing the frame horizontally covers far less distance than one crossing
vertically (`ffprobe` reports `width=2160 height=3840` with no rotation side data, and the detections bear it out:
max `x+w` is 2157 on `6C42` / 2147 on `1DAC`, while max `y+h` reaches 3397 / 3450). **`cv2.VideoCapture` opens and
decodes both directly, no fix needed** -- checked, not assumed (`cv2.getBuildInformation()`'s FFMPEG backend handles
the container; `run_tracker.py` and the cut/verification steps below all ran against the `.mkv` sources and the
`.mp4` cuts without any reader change).

## The detection run -- genuine per-frame YOLO detections

`*_detections.csv` is now the real output of `birdwatcher.main_detections` (`--pipeline-type rgb_yolo`), the
detection-only entrypoint the kit is meant to consume. `classifier_name` is `crop-classifier` in every row -- the
actual model name, not the old `CLOUD_TRACKS_FALLBACK_NOT_A_DETECTOR` sentinel. What was run, exactly:

- **Weights.** The pipeline's one learned model is the `crop-classifier` YOLO classifier. Its `best.pt` was exported
  from MLflow **on a machine that has that access** and handed to this environment as a plain file. **No
  credentials were placed in this sandbox** -- it has no AWS keys and no model-registry DB, so `get_yolo_weights`
  (which normally does a registry-DB lookup + S3 download) was monkeypatched at runtime to load the local
  `best.pt`. No `spoor-main` source file was modified; the shim lives in a standalone runner.
- **Configs.** Both recordings are `camera_id` 77 = **baltic-eagle-camera-4** (from the mirror `videos` table), so
  both used `--camera-config config_baltic_eagle_camera_4 --resolution-config config_4k`. Both names are stock
  entries in `libs/configs-manager/.../config_files/`; nothing was added to the shared config set for this run.
  (Watch out for the similarly-named `test_resolution_config_4k.yaml` fixture next to them -- it is missing the
  required `rendering_parameters.color` field and will not load. Use `config_4k`.)
- **Windows + the cut trap.** Detection ran the original `.mkv`s over the same windows as the clips (`6C42`:
  original frames 2000-2670; `1DAC`: 4001-5009), then the CSV `frame_number` was rebased to 0 to match the cut
  `.mp4`s. One subtlety worth knowing: the frame-difference detector produces nothing on the very first frame it
  sees (no predecessor to difference against), so each run starts **one frame early** (from 1999 / 4000) purely to
  give clip-frame 0 a real predecessor; that lead-in frame is then dropped. The result covers exactly
  `frame_number` 0..`frame_count-1`, which is what `run_tracker.py`'s alignment check requires (verified: it does
  not refuse either pair).

**These detections are genuinely harder for the tracker than the old fallback was, and that is the point.** The
fallback fed `run_tracker.py` the cloud tracker's own already-associated, filtered, one-box-per-bird trajectories.
A real per-frame YOLO pass instead produces missed frames (a bird blending into background or sky), the occasional
double box on one bird, and background/edge noise boxes -- so `run_tracker.py` has real association work to do, and
it shows (see Findings). `*_cloud_reference_tracks.csv` (with `obj_id` intact) is kept as a *separate* comparison
trajectory from a different, heavier pipeline; it is **not** ground truth fed into the tracker.

## Why these two windows

`main_detections.py`/`run_tracker.py`'s trap: a CSV cut with `--start-frame-number` renumbers `frame_number` from 0
relative to the cut, but an uncut video's frame count is not renumbered. Feeding one against the other silently
misaligns every detection. **Both clips here are the video and the CSV cut to the exact same bounds** (verified:
`run_tracker.py`'s alignment refusal did not fire on either pair -- see `out/`).

Windows were chosen from the stored cloud tracks (`tracks` in the mirror), not the first N frames:

- **`20250920_063942_6C42` (bird and turbine), frames 2000-2670 of the original 7502.** Cloud `obj_id` 372 is
  tracked continuously for 2053-2292, then nothing is seen again in that position until `obj_id` 440 appears at
  frame 2482, 11px from where 372 was last seen -- a ~190-frame (7.6s) gap consistent with the turbine blade
  occluding the bird, and the cloud tracker's own segmentation treats it as two identities across that gap. A
  second, similar break follows shortly after (`obj_id` 440 -> 458, ending frame 2590). Cloud `obj_id`s present in
  this window: 353, 363, 372, 381, 390, 440, 458, 468 (8 distinct).
- **`20251012_164031_1DAC` (dense flock), frames 4001-5009 of the original 7502.** This is the clip's densest
  passage: coexisting cloud `obj_id` count (cut numbering) builds from 1 bird around frame 40, to 4+ by frame 200,
  peaks at 12 simultaneous identities near frame 380, and tapers back to 1-2 by frame 560. Cloud `obj_id`s present
  in this window: 491, 496, 503, 508, 511, 514, 521, 522, 527, 528, 530, 532, 542, 549, 559, 561, 647 (17 distinct).

## Findings -- what the kit's tracker actually did on genuine detections

Run with the kit's default parameters (`--euclidean-matching-threshold 250 --max-age 30 --tentative-threshold 3`).
Frame numbers are in the cut clip's own numbering (frame 0 = the first frame of the `.mp4` here), matching
`out/*_tracks.csv` and the annotated videos.

**Detection density (real detector, sparse and noisy as expected):**

| clip | detections | frames with >=1 detection | genuine vs fallback input |
|---|---|---|---|
| `20250920_063942_6C42` | 734 | 486 / 671 (72%) | fallback covered every frame; real detector drops out on ~28% |
| `20251012_164031_1DAC` | 4424 | 989 / 1009 (98%) | many boxes/frame in the flock, including doubles and edge noise |

**Track counts -- verified before/after (same clips, same tracker, same settings; only the detections differ):**

| clip | cloud identities in window (reference) | fallback run (clean boxes) | **genuine detections** |
|---|---|---|---|
| `20250920_063942_6C42` | 8 | 8 track ids | **35 track ids (4.4x)** |
| `20251012_164031_1DAC` | 17 | 20 track ids | **78 track ids (4.6x)** |

The fallback produced almost exactly one track per cloud identity because it was fed the cloud tracker's own clean
trajectories. On genuine detections the kit's tracker emits **4.4x-4.6x more identities than there are birds.**
Two mechanisms drive it, both visible in the annotated videos in `out/`:

- **Fragmentation from missed frames.** The real detector loses a bird for stretches (72% frame coverage on `6C42`
  means gaps of several frames are common). When a gap exceeds `--max-age 30`, the coasting track ages out and the
  bird re-enters as a brand-new id -- one bird becomes several sequential tracks. Every emitted `6C42` track still
  spans >=22 frames (min span 22, median 47), so these are substantial tracks, not one-frame noise.
- **Spurious tracks from detector noise.** Background/edge boxes that the fallback never contained (e.g. small boxes
  at the frame border) can repeat for a few frames, pass `--tentative-threshold 3`, and confirm an extra track that
  corresponds to no real bird -- which then coasts under `--max-age`, so these are not necessarily short-span
  (every emitted track spans at least 9-22 frames; none is one-frame noise). Fragmentation is the larger driver of
  the 4.4x-4.6x; noise adds to it.

This is the honest result the kit is meant to show: **the tracker's own association work, hidden by the fallback,
is exposed by genuine detections, and at default settings it over-segments heavily -- worse on the fast, crowded
flock clip than on the single-bird occlusion clip.** The teaching question is not "tune the thresholds until the
count matches" but which parameter could plausibly help and why (top-level README section 5), and why
proximity-only matching with no appearance signal fragments under missed frames and dense crossings (section 3, and
`PROVENANCE.md` on the `(x, y, area)` feature). To reproduce any number here, open `out/*_tracks.csv` (columns
`frame_number, track_id, x, y, w, h`) and the annotated `out/*_tracked.mp4`.

## Regenerating

The detections here were produced with `birdwatcher.main_detections` against the original `.mkv`s. To regenerate
in an environment that has the `crop-classifier` weights available (as a local `best.pt`, or via a real
registry-DB + S3 + AWS setup), per clip:

```
# original frame windows (rebase the resulting frame_number to 0 to match the cut .mp4):
#   6C42: --start-frame-number 1999 --stop-frame-number 2671   (lead-in 1999 dropped -> clip frames 0..670)
#   1DAC: --start-frame-number 4000 --stop-frame-number 5010   (lead-in 4000 dropped -> clip frames 0..1008)
python -m birdwatcher.main_detections \
  --pipeline-type rgb_yolo \
  --video-path <original .mkv> \
  --camera-config config_baltic_eagle_camera_4 \
  --resolution-config config_4k \
  --start-frame-number <N> --stop-frame-number <M> \
  --output-csv <out>.csv
```

Then rebase `frame_number` (and `frame_timestamp`) by the window start so the CSV runs 0..`frame_count-1` of the
cut clip, and re-run `run_tracker.py`. If the weights or the frame windows change, re-run the kit and update the
Findings above rather than assuming they still hold.
