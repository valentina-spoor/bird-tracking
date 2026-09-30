#!/usr/bin/env bash
# Run one or more experiment configs on both real clips, without writing annotated videos.
#   research/runner/run_experiments.sh original fused k4t20_fused
# Results land in experiments/results/<config name>/<clip>_tracks.csv. Clips run in parallel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KIT="$ROOT/spoor_tracker/tracker-kit"
declare -A CLIPS=( [1DAC]=20251012_164031_1DAC [6C42]=20250920_063942_6C42 )
cd "$KIT"
for name in "$@"; do
  out="$ROOT/experiments/results/$name"; mkdir -p "$out"
  for clip in "${!CLIPS[@]}"; do
    id="${CLIPS[$clip]}"
    python3 run_tracker.py --video "data/$id.mp4" --detections "data/${id}_detections.csv" \
      --tracks "$out/${clip}_tracks.csv" --no-video --config "$ROOT/experiments/configs/$name.json" \
      > "$out/${clip}_run.log" 2>&1 &
  done
  wait
  echo "done $name"
done
