# Tracker evaluation — reference code

A self-contained extract of Spoor's prediction-evaluation library, for Diego and Valentina
to read before building evaluation for the tracker work.

This is **reference material, not a dependency**. Do not import it into what you build and
do not try to keep it in sync with Spoor's copy. Read it, understand why it is shaped the
way it is, then write your own.

Everything here is pure Python standard library. No database, no S3, no credentials, no
Spoor services. Both of these work on a clean machine with nothing installed but pytest:

```sh
python3 -m pytest tests -q
PYTHONPATH=. python3 example/score_two_runs.py
```

About 1,900 lines of library, plus an example and some tests. Read it in the order below
rather than front to back; the first four files are 460 lines and carry the whole idea.

## Start here

Read in this order. It is roughly the order of dependency, and each file makes the next
one legible.

| Order | File | What it establishes |
| --- | --- | --- |
| 1 | `spoorpredictioneval/contracts.py` | The vocabulary. `Detection`, `TrackUpdate`, `Track`, `RunResult`, `Provenance`. Everything else is defined in these terms. |
| 2 | `spoorpredictioneval/result_store.py` | The persistence port, and the content-addressed run id. 56 lines, and the most important file here. |
| 3 | `spoorpredictioneval/metrics.py` | Box matching by IoU. The primitive the rest builds on. |
| 4 | `spoorpredictioneval/track_view.py` | Turning a stream of per-frame updates into `Track` objects. |
| 5 | `spoorpredictioneval/banded.py` | Confidence-banded detection comparison. |
| 6 | `spoorpredictioneval/track_quality.py` | Identity switches, fragmentation, IDF1, and the association choice. |
| 7 | `spoorpredictioneval/recall_decomposition.py` | Why each reference box was or was not found. |
| 8 | `spoorpredictioneval/compare.py` | The orchestration: resolve two runs, evaluate, render. |
| 9 | `example/score_two_runs.py` | All of it end to end, on synthetic data you can edit. |

`spoorpredictioneval/overlay.py` is not on the path. It draws boxes on frames for visual
inspection, which you will want eventually but not while you are learning the shape.

Then read `tests/test_examples.py`. It is nine tests written as five patterns, one per
kind of thing worth testing: a pure function at its boundaries, a transformation checked
by its structure, a rule that is not obvious from the code, a round trip through a port,
and a property rather than a value. Each is commented with why that shape exists.

This is deliberately not Spoor's test suite. The real library has far more, and you are
meant to design your own evaluation rather than inherit someone else's assertions. Nine
examples are enough to show what good looks like.

## Seven ideas worth stealing

**The evaluator never knows where results are stored.** `ResultStore` is a `Protocol` with
two methods, `persist` and `resolve`. `JsonlResultStore` writes local files. Spoor's
production store writes to a dataset manager backed by S3 and a database. The comparison
code cannot tell the difference and has never been changed to accommodate either. That is
precisely why this bundle runs on your laptop with nothing installed — the infrastructure
was removable because it was never entangled in the first place. If you take one thing
from this code, take this.

**Define the data model before the metrics.** `contracts.py` is 148 lines and comes first.
Once `Detection`, `Track` and `RunResult` exist, every metric is a function over them, and
two different producers can be compared because they both speak the same types. Metrics
written directly against a tracker's internal state cannot be reused and cannot be tested.

**Provenance travels with the result.** A `RunResult` carries the config hash, git commit
and model hash that produced it. A comparison without that is a number you cannot defend
three weeks later, because you no longer know what produced either side.

**Run ids are content-addressed.** `content_run_id` hashes provenance plus output, with no
clock and no randomness. A byte-identical run yields the same id. This makes "did anything
actually change?" answerable by comparing two strings.

**Never a single threshold.** `banded.py` carries the design principle as a comment:
precision and recall are reported per confidence band, not collapsed to one number at one
cutoff. A model that got worse only on low-confidence detections looks unchanged under a
single threshold. Run the example and look at how the spurious detection lands in exactly
one band.

**Association is a decision, not a default.** `track_quality.py` supports both
`center_distance` and `iou`. This matters enormously for your work: a bird eight pixels
across, offset by three pixels, has an IoU near zero while being obviously the same bird.
IoU is the wrong criterion for small targets and will silently tell you your tracker is
broken. The example uses `center_distance` with an 8-pixel threshold for that reason.

**Diagnose, don't just score.** `recall_decomposition.py` does not report "we missed 12
boxes". It reports whether each one was `under_confident`, `under_localized` or
`no_overlap`. Those point at three different bugs in three different parts of a pipeline.
A metric tells you something is wrong; a decomposition tells you where to look.

## What the example shows

`example/score_two_runs.py` builds two runs of two objects over 40 frames. The reference is
clean. The candidate has three defects injected, one per failure mode:

| Injected | Caught by |
| --- | --- |
| One dropped detection | recall 0.988 in the top band, `no_overlap` in the decomposition |
| One spurious detection | precision 0.000 in band `[0.50,0.70)`, raised as a finding |
| Identity swap at frame 25 | `idf1=0.613`, `identity_switches=2`, `fragmentation=1`, track count 3 vs 2 |

Change the defects and watch which numbers move. That exercise will teach you more about
MOT metrics than reading the papers first.

## What was removed, and why

Three things are missing relative to Spoor's copy, all deliberately:

- **`stores/dataset_store.py`** — the production `ResultStore` implementation. It binds to
  Spoor's dataset manager, S3 and the production database. Removing it cost nothing
  elsewhere, which is the point made above.
- **The edge adapters** — `corpus.py`, `gt_adapter.py`, `cloud_adapter.py`, `adapter.py`.
  These translate Spoor's specific output formats and storage into the canonical contracts.
  Their *shape* is worth knowing about — an adapter per producer, all targeting one set of
  types — but the contents are specific to systems you have not seen.
- **HOTA, MOTA and MOTP** — not removed; they genuinely do not exist yet. `track_quality.py`
  implements IDF1, identity switches and fragmentation. Implementing the others is live work
  on the team's board, so understand what is here before assuming what should be added.

## How this relates to your milestones

Milestone 3 asks for an evaluation framework and a baseline performance report. This code is
the closest existing example of what that framework looks like when it is done well. You are
not asked to reproduce it. You are asked to understand why the boundaries fall where they do,
and then to make your own decisions for the tracker evaluation you build.

Two questions worth being able to answer before you write code:

1. What is your reference? Comparing a run against a *previous run* tells you something
   changed. Comparing against *independent labels* tells you whether it is correct. These are
   different claims and they need different data. Be explicit about which you are making.
2. What is the unit you evaluate — a single tracker invocation, or a whole pipeline? The
   answer changes the contracts.

Ask questions on either of those. They are the interesting part.
