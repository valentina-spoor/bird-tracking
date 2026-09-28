# Provenance

This kit was rebuilt from `fm/tracker-kit-build` at `d8d5796de`, which copied the **edge**
tracker (`edge/compute/src/spooredge/compute/flows/rgb_yolo/nodes/trackers/`). The captain
corrected the talk to the **bird-watcher** tracker instead, so this revision copies from a
different source entirely: `libs/tracker/src/spoortracker/`, at spoor-main commit
`145fc6696241ec3ea1e1d3b930164bebf226538b` (`origin/main`). Each copied file also carries this
same note at its own top. To check a file against the original:

```
git show 145fc6696241ec3ea1e1d3b930164bebf226538b:<source path> | diff - tracker-kit/tracker/<file>
```

| kit file | source path | changed |
|---|---|---|
| `tracker/kalman_filter.py` | `libs/tracker/src/spoortracker/kalman_filter.py` | nothing but the header comment |
| `tracker/tracked_object_state.py` | `libs/tracker/src/spoortracker/tracked_object_state.py` | nothing but the header comment |
| `tracker/nn_matching.py` | `libs/tracker/src/spoortracker/nn_matching.py` | nothing but the header comment |
| `tracker/kalman_tracker_state.py` | `libs/tracker/src/spoortracker/kalman_tracker_state.py` | imports only: box/detection types now come from `tracker.shim` |
| `tracker/tracked_object.py` | `libs/tracker/src/spoortracker/tracked_object.py` | imports only: `DetectionNoCrop` now comes from `tracker.shim` |
| `tracker/linear_assignment.py` | `libs/tracker/src/spoortracker/linear_assignment.py` | imports only |
| `tracker/distance_matching.py` | `libs/tracker/src/spoortracker/distance_matching.py` | imports only |
| `tracker/matching.py` | `libs/tracker/src/spoortracker/matching.py` | imports only |
| `tracker/simple_sort_tracker.py` | `libs/tracker/src/spoortracker/simple_sort_tracker.py` | imports only |
| `tracker/shim.py` | none -- new file | replaces `spoorcontainers`'s bounding-box and detection dataclasses/transforms |

## Why a copy, not an import

`spoorcontainers` (the package these nine files actually import) is much lighter than the edge
tracker's dependencies -- just `numpy`, no `spoorflow`, no message bus, no Flow node framework
at all: **the bird-watcher tracker is a plain Python class**, not a Flow node, so this revision
needed no equivalent of the old kit's `Node`/`EndOfStreamPacket`/`NodeDeclaration` shimming.
But `spoorcontainers` is still a monorepo-internal package that would need an editable `uv`
workspace install to import, which defeats "two bachelor students can `pip install -r
requirements.txt` and run it anywhere." The tradeoff is the same as before: **a copy drifts
from the original silently.** Nothing rebuilds this kit when `simple_sort_tracker.py` changes
upstream. Where this kit eventually lives is the captain's decision, not this worker's; this
manifest exists so whoever inherits that decision can diff against the original first.

## What changed switching from the edge tracker to bird-watcher's

Recorded here because the captain asked to know early if this would invalidate more of the old
kit than it preserved. It did not -- but almost everything at the tracker/adapter layer still
needed a full rewrite, not a patch:

- **The interface is smaller and simpler**, exactly as the captain's brief predicted:
  `match_and_track(detections)` / `.tracked_objects` / `finalize_tracking()`, vs. the edge
  tracker's `Node.receive("trackable_candidates", packet)` returning a `PerceptionUpdatePacket`.
  Track ids are plain ints (`itertools.count()`), not uuids.
- **The shim shrank, not grew.** The edge kit's `shim.py` replaced an entire Flow node/packet
  framework (~180 lines). This one only replaces bounding-box/detection dataclasses (~140
  lines) -- there is no Flow framework in this tracker's import graph to stand in for at all.
- **`sequence_number` had no equivalent, and that mattered more than expected.** The edge
  tracker took an explicit frame-gap input (`source_frame.sequence_number`) so a track's
  predicted uncertainty grew with however many frames were actually missed. This tracker's
  Kalman filter has a hardcoded `dt = 1.0` (`tracker/kalman_filter.py`): every call to
  `match_and_track` advances every track by exactly one step, matched or not, with **no
  elapsed-time input at all**. `run_tracker.py`'s frame-alignment refusal check is kept (a CSV
  cut against the wrong video is still a real bug -- detections would attach to the wrong
  frames), but the reason it matters had to be rewritten: it is no longer about protecting a
  frame-gap-derived quantity, since this tracker has none.
- **`feature` replaced `sequence_number` as the field that had to be derived correctly, not
  defaulted.** `frame_timestamp`, `is_predicted_by_tracker`, `video_name_id`,
  `frame_timestamp_unit` and `frame_rate` on `DetectionNoCrop` are carried but never read by any
  of the nine copied files (verified by grep, same discipline as the edge kit's PROVENANCE) --
  those are defaulted freely. `feature`, by contrast, feeds `matching.py`'s confirmed-track
  cascade directly and is not optional. Production's own caller, `tracker_filter.py`, derives it
  from the detection's own box (`(x, y, area)`, the box's own top-left corner and area) rather
  than from any separate signal -- `run_tracker.py` follows that exactly. See `tracker/shim.py`'s
  module docstring.
- **There is no retired baseline tracker to compare against.** The edge kit's B0 control
  (`--tracker greedy-nn`) was a real, separately-implemented, once-shipped tracker still in the
  repo. Bird-watcher has no analog: `SimpleSORTTracker` is the only tracker in this pipeline, and
  there is nothing else to construct honestly as a second "tracker." The kit's README explains
  this plainly instead of inventing one. What the kit does offer instead -- verified by actually
  running it, not assumed -- is a real fragmentation demonstration using this same tracker's own
  `--max-age` parameter: see README section 4.
- **The appearance-matching cascade is real code, but not real appearance matching in this
  pipeline.** `matching.py`'s `gated_metric` looks like DeepSORT's re-identification stage, and
  it runs unconditionally for every confirmed track. But bird-watcher's only caller,
  `tracker_filter.py`, feeds it `feature = (x, y, area)` -- position and size, not a visual
  embedding. Tested directly: varying `--euclidean-matching-threshold` from 250 down to 100 on
  the synthetic occlusion-gap clip changed nothing (still 3 track ids, bird-c's identity held) --
  because the *second* matching stage (`distance_matching.closest_track_cost`) compares against
  the Kalman filter's own **predicted** position, not the last-observed one, so a threshold this
  loose never had a chance to bind for a constant-velocity gap. This is documented as an honest
  finding, not built into the runner as a fake second configuration.
