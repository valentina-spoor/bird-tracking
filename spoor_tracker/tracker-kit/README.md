# Tracker kit

Spoor's bird tracker, isolated from the fleet: no pipeline, no database, no model. Feed it a
video and a CSV of per-frame bounding boxes, and it writes out an annotated video and a track
CSV. This is the same tracker that runs in the bird-watcher pipeline today, copied and shimmed
to run standalone -- see `PROVENANCE.md` for exactly what changed and why.

## 1. What you are looking at

A detector answers "is there a bird here, in this one frame" -- a bounding box with no memory of
any other frame. A **track** is the tracker's running hypothesis that boxes in different frames
are the same real bird. **Association** is the decision, made every frame, of which box belongs
to which track. That decision is the entire job: one bird, one track, one color in the annotated
output. If a bird's color changes mid-flight, or two birds trade colors, that is the tracker
getting association wrong, not a rendering bug.

## 2. Install and run

```
pip install -r requirements.txt
python generate_synthetic_data.py                 # writes data/synthetic/synthetic.mp4 + .csv
python run_tracker.py \
  --video data/synthetic/synthetic.mp4 \
  --detections data/synthetic/synthetic_detections.csv \
  --out out/synthetic_tracked.mp4 \
  --tracks out/synthetic_tracks.csv
```

The real recordings have landed in `data/` (see `data/README.md` for exactly what is there, the
detection-run caveat, and the findings from running the kit on them):

```
python run_tracker.py \
  --video data/20250920_063942_6C42.mp4 \
  --detections data/20250920_063942_6C42_detections.csv \
  --out out/20250920_063942_6C42_tracked.mp4 \
  --tracks out/20250920_063942_6C42_tracks.csv

python run_tracker.py \
  --video data/20251012_164031_1DAC.mp4 \
  --detections data/20251012_164031_1DAC_detections.csv \
  --out out/20251012_164031_1DAC_tracked.mp4 \
  --tracks out/20251012_164031_1DAC_tracks.csv
```

## 3. What the tracker does, frame by frame

`SimpleSORTTracker.match_and_track(detections)` (`tracker/simple_sort_tracker.py`), called once
per video frame:

1. **Associate, confirmed tracks first.** Every confirmed track competes for a match through a
   cascade ordered by how recently it was last seen (`tracker/matching.py`): the cost is a
   nearest-neighbor distance in `feature` space, gated by whether the candidate is a statistically
   plausible sighting given the Kalman filter's own uncertainty. Anything left over -- plus every
   still-unconfirmed track -- is matched by a second, simpler pass: nearest predicted box position
   by straight-line distance.
2. **Update or spawn.** A matched track is corrected toward its detection, and confirmed once it
   has accumulated `tentative_threshold` matches. An unmatched detection starts a new, unconfirmed
   track.
3. **Cull.** A confirmed track survives up to `max_age` consecutive misses; an unconfirmed one is
   dropped the moment it isn't matched.
4. **Predict.** Every surviving track is advanced one Kalman step, unconditionally -- this
   happens on every single call, matched or not, so `run_tracker.py` must call
   `match_and_track` exactly once per real video frame, in order (see its own docstring).

Only `CONFIRMED` tracks are rendered and written to the track CSV, matching the fleet's own
`DisplayTracksSink` convention.

## 4. What to watch for

Each confirmed track is drawn as its box, its id, and one dot per frame it has been matched in,
in the track's own color. The dots are **every** detection the track has accumulated, not a
recent tail, so a track that has been alive a while shows its whole flight path; a track that
dies stops being drawn and takes its path with it.

- **A color changing mid-flight** on one continuous-looking path: an identity switch.
- **A bird picking up a new color** after a gap: fragmentation -- the tracker lost it and started
  over.
- **Two colors flickering on one bird:** duplicate tracks.

Bird-watcher has no retired baseline tracker to run alongside this one for contrast -- see
`PROVENANCE.md` for why a fabricated second tracker would not be honest. What the kit offers
instead, run and confirmed on `data/synthetic/`:

- **Default run** (`--max-age 30`, `--tentative-threshold 3`, `--euclidean-matching-threshold
  250`, all the pipeline's real production values): **3** track ids total, one per bird.
  `bird-c` (40 px/frame, occluded for a 3-frame/120px gap) holds its id across the whole gap
  (frames 2-73 under one id, uninterrupted). `bird-a`/`bird-b` cross head-on around frame 125 and
  both hold their ids straight through it -- checked directly, not assumed: each track's x
  position moves monotonically at its own bird's exact velocity across the crossing, with no
  swap.
- **`--max-age 2`** (below the 3-frame gap length): **4** track ids. `bird-c`'s track is deleted
  after 2 missed frames (its id ends at frame 5) and a new id is born once it reappears and
  re-confirms (starting at frame 10) -- the same fragmentation the edge kit's retired baseline
  used to demonstrate, produced here with the *same* tracker by giving it less patience than the
  occlusion is long, not a different algorithm.
- **What did *not* work, and why that is itself the finding:** lowering
  `--euclidean-matching-threshold` from 250 to 100 changes nothing (still 3 ids, gap still
  bridged). The second matching stage compares a candidate against the Kalman filter's own
  *predicted* position for this frame, not the last position the bird was actually seen at --
  for a bird moving at constant velocity, that prediction lands close to the true position
  regardless of how long the gap was, so a 120px gap never has to cross a 100px threshold at
  all. This is the real behavioral difference from the edge kit's retired baseline, which matched
  against the *last-seen* position with a fixed radius and so failed exactly this kind of gap.

## 5. The knobs

All three constructor parameters of `SimpleSORTTracker`, exposed as CLI flags:

| parameter | flag | default | what it does |
|---|---|---|---|
| `euclidean_matching_threshold` | `--euclidean-matching-threshold` | 250 (config_8k.yaml's fleet-tuned `distance_threshold`) | max distance accepted as the same track, in *both* matching stages |
| `max_age` | `--max-age` | 30 (the pipeline's own value, `bird_watcher_tracking_pipeline.py:143`) | consecutive missed frames a confirmed track survives before deletion |
| `tentative_threshold` | `--tentative-threshold` | 3 (same pipeline) | matched detections needed before a new track is confirmed (and rendered) |

There is no separate Kalman-noise tuning here -- unlike the edge tracker's
`PredictConfirmMatchTrackerParameters`, `SimpleSORTTracker`'s constructor is just these three
values; the Kalman filter's own process/measurement noise (`tracker/kalman_filter.py`) is fixed.

Predict what changing one will do before you change it, then run the kit and check -- section 4
above is exactly that exercise, already done once for you.

## 6. Honest limits of this kit

- One camera, image-plane only -- there is no world-coordinate output here, matching what this
  tracker itself produces (unlike, e.g., a stereo pipeline).
- This runner calls only `match_and_track` and `finalize_tracking`. It deliberately does **not**
  call `filter_deleted_tracks` or `concatenate_tracks`: those exist on `SimpleSORTTracker` and
  the real pipeline uses both (short/circular-track filtering, and stitching tracks that end and
  restart near each other within a time window) as *post*-processing, downstream of the tracker
  itself. What you see here is the tracker's raw frame-by-frame output, the same scoping choice
  the edge kit made for its own post-processing steps.
- `data/synthetic/` is generated, not real detector output -- see `data/README.md` for the real
  recordings this kit was designed for and their status.

## 7. Metrics

Deliberately not included -- pick what you want to measure once you understand what the tracker
is doing. Two starting points: `py-motmetrics` (a general MOT metrics library, install it
yourself if you want IDF1/MOTA/etc.), or Spoor's own dependency-free implementation at
`libs/prediction-eval/src/spoorpredictioneval/track_quality.py` (`evaluate_track_quality`), which
is what Spoor actually uses.
