"""Handcrafted visual descriptors for small, distant birds.

Spoor's detections are ~15-40 px boxes on a 4K frame, so a learned re-ID embedding has very
little signal to work with. Instead each detection gets three cheap cues cropped from the frame:

* ``color``    -- HSV histogram inside an elliptical mask (what colour is the object).
* ``patch``    -- zero-mean, unit-norm grayscale thumbnail (what shape/texture, compared by NCC).
* ``contrast`` -- signed gray-level difference between the object and a ring around it. This is
                  the polarity cue: a dark bird on bright sky and a bright blade edge against a
                  dark background have opposite signs, which box geometry can never tell apart.

Every distance returned here is in [0, 1], 0 meaning identical, so they can be blended with a
pixel distance in ``association_costs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PATCH_SIZE = 12
HIST_BINS = (8, 4, 4)  # H, S, V
RING_SCALE = 2.0  # surrounding ring is the box scaled by this factor
CONTRAST_SCALE = 64.0  # gray levels at which the contrast distance saturates to 1
MIN_SIDE = 6  # boxes smaller than this are grown so the crop has some pixels to work with
NEUTRAL_DISTANCE = 0.5  # used when a cue is undefined (empty crop, flat patch)

CUE_NAMES = ("color", "patch", "contrast")


@dataclass
class Appearance:
    color: np.ndarray  # (prod(HIST_BINS),) L1-normalised
    patch: np.ndarray  # (PATCH_SIZE**2,) zero-mean unit-norm, or all zeros if the crop was flat
    contrast: float

    def blended(self, new: "Appearance", alpha: float) -> "Appearance":
        """Exponential moving average, ``alpha`` is the weight kept on the old template."""
        color = alpha * self.color + (1.0 - alpha) * new.color
        patch = alpha * self.patch + (1.0 - alpha) * new.patch
        norm = np.linalg.norm(patch)
        patch = patch / norm if norm > 1e-6 else np.zeros_like(patch)
        contrast = alpha * self.contrast + (1.0 - alpha) * new.contrast
        return Appearance(color, patch, contrast)


def extract_appearance(frame_bgr: np.ndarray, bounding_box) -> Appearance | None:
    """Describe the object inside ``bounding_box`` (a BoundingBoxXYXY). None if the box is off-frame."""
    frame_h, frame_w = frame_bgr.shape[:2]
    cx = (bounding_box.x1 + bounding_box.x2) / 2.0
    cy = (bounding_box.y1 + bounding_box.y2) / 2.0
    half_w = max(bounding_box.x2 - bounding_box.x1, MIN_SIDE) / 2.0
    half_h = max(bounding_box.y2 - bounding_box.y1, MIN_SIDE) / 2.0

    ix1, ix2 = int(np.floor(cx - half_w)), int(np.ceil(cx + half_w))
    iy1, iy2 = int(np.floor(cy - half_h)), int(np.ceil(cy + half_h))
    rx1, rx2 = int(np.floor(cx - half_w * RING_SCALE)), int(np.ceil(cx + half_w * RING_SCALE))
    ry1, ry2 = int(np.floor(cy - half_h * RING_SCALE)), int(np.ceil(cy + half_h * RING_SCALE))

    # Clip to the frame; the ring is the outer region and the inner box is cut out of it.
    rx1, ry1, rx2, ry2 = max(rx1, 0), max(ry1, 0), min(rx2, frame_w), min(ry2, frame_h)
    ix1, iy1, ix2, iy2 = max(ix1, rx1), max(iy1, ry1), min(ix2, rx2), min(iy2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    region = frame_bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sl = (slice(iy1 - ry1, iy2 - ry1), slice(ix1 - rx1, ix2 - rx1))
    inner_gray, inner_hsv = gray[sl], hsv[sl]

    inner_h, inner_w = inner_gray.shape
    mask = np.zeros((inner_h, inner_w), np.uint8)
    cv2.ellipse(mask, (inner_w // 2, inner_h // 2), (max(inner_w // 2, 1), max(inner_h // 2, 1)), 0, 0, 360, 255, -1)

    hist = cv2.calcHist([inner_hsv], [0, 1, 2], mask, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    total = hist.sum()
    hist = hist / total if total > 0 else np.full(hist.size, 1.0 / hist.size)

    thumb = cv2.resize(inner_gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    thumb = thumb / norm if norm > 1e-6 else np.zeros_like(thumb)

    object_mean = float(inner_gray[mask > 0].mean()) if mask.any() else float(inner_gray.mean())
    ring_mask = np.ones(gray.shape, bool)
    ring_mask[sl] = False
    contrast = object_mean - float(gray[ring_mask].mean()) if ring_mask.any() else 0.0

    return Appearance(hist.astype(np.float32), thumb, contrast)


def appearance_distance_matrix(
    templates: list[Appearance | None],
    candidates: list[Appearance | None],
    cue_weights: tuple[float, float, float],
) -> np.ndarray:
    """(len(templates), len(candidates)) blended appearance distance in [0, 1]."""
    n, m = len(templates), len(candidates)
    if n == 0 or m == 0:
        return np.zeros((n, m))

    def stack(items, attr, size):
        out = np.zeros((len(items), size))
        for i, item in enumerate(items):
            if item is not None:
                out[i] = getattr(item, attr)
        return out

    valid_t = np.array([t is not None for t in templates])
    valid_c = np.array([c is not None for c in candidates])
    both = valid_t[:, None] & valid_c[None, :]

    color_t, color_c = stack(templates, "color", int(np.prod(HIST_BINS))), stack(candidates, "color", int(np.prod(HIST_BINS)))
    bhattacharyya = np.clip(np.sqrt(np.clip(color_t, 0, None)) @ np.sqrt(np.clip(color_c, 0, None)).T, 0.0, 1.0)
    d_color = np.sqrt(np.clip(1.0 - bhattacharyya, 0.0, 1.0))

    patch_t, patch_c = stack(templates, "patch", PATCH_SIZE**2), stack(candidates, "patch", PATCH_SIZE**2)
    d_patch = np.clip((1.0 - patch_t @ patch_c.T) / 2.0, 0.0, 1.0)
    flat = (np.linalg.norm(patch_t, axis=1) < 1e-6)[:, None] | (np.linalg.norm(patch_c, axis=1) < 1e-6)[None, :]
    d_patch = np.where(flat, NEUTRAL_DISTANCE, d_patch)

    con_t = stack(templates, "contrast", 1)[:, 0]
    con_c = stack(candidates, "contrast", 1)[:, 0]
    d_contrast = np.clip(np.abs(con_t[:, None] - con_c[None, :]) / CONTRAST_SCALE, 0.0, 1.0)

    weights = np.asarray(cue_weights, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("cue_weights must have a positive sum")
    blended = (weights[0] * d_color + weights[1] * d_patch + weights[2] * d_contrast) / weights.sum()
    return np.where(both, blended, NEUTRAL_DISTANCE)
