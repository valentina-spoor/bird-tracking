"""Frame count, fps, width and height of a video, read from the video file and cached to JSON.

The frame count always comes from the video, never from a CSV: a frame with no GT and no
prediction still exists and is evaluated as an empty frame. The cache key is the video's
resolved path, size and modification time, so a replaced video is re-read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    width: int
    height: int


def _read(video: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video {video}")
    try:
        meta = VideoMetadata(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if meta.frame_count <= 0:
        raise RuntimeError(f"{video}: the container reports no frame count")
    return meta


def video_metadata(video: Path, cache_dir: Path | None = None) -> VideoMetadata:
    video = Path(video).resolve()
    if cache_dir is None:
        return _read(video)
    stat = video.stat()
    key = {"path": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_file = Path(cache_dir) / f"{video.stem}.metadata.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached.get("key") == key:
            return VideoMetadata(**cached["metadata"])
    meta = _read(video)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"key": key, "metadata": asdict(meta)}, indent=2) + "\n")
    return meta
