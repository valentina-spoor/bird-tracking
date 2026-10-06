"""One comparison of a candidate run against a reference on both axes at once.

Detection drift (confidence-banded precision/recall plus recall attribution) and identity
continuity (IDF1, switches, fragmentation, track signals) are two lenses on the same pair of runs.
This module composes the existing per-axis primitives into a single ``EvaluationReport`` and renders
it as one human report, so a caller scores once and sees both. Comparisons report evidence; they do
not decide whether a run or an optimization is acceptable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spoorpredictioneval.contracts import (
    Detection,
    FrameDetections,
    LocalizationSummary,
    Provenance,
    RunResult,
)
from spoorpredictioneval.metrics import center_distance, match_boxes_by_iou
from spoorpredictioneval.recall_decomposition import (
    RecallDecomposition,
    decompose_recall,
)
from spoorpredictioneval.regression import RegressionReport, Tolerance, score_regression
from spoorpredictioneval.track_quality import (
    DEFAULT_ASSOCIATION,
    AssociationMode,
    TrackQualityReport,
    TrackSignals,
    evaluate_track_quality,
    track_signals,
)

_PROVENANCE_EXTRA_KEYS: tuple[str, ...] = ("predictor", "tracker", "detector", "decoder", "label")


@dataclass(frozen=True)
class EvaluationReport:
    """A candidate-vs-reference comparison on both the detection and the track axis.

    ``regression`` is the confidence-banded detection drift; ``recall_decomposition`` attributes the
    reference boxes the candidate did not find; ``localization`` is the center-error distribution over
    matched candidate-reference pairs; ``track_quality`` is the candidate's identity continuity against
    the reference; ``reference_track_signals`` describes the reference's own tracks (the candidate's are
    on ``track_quality.candidate_signals``). Provenance of both runs travels with the report so the
    comparison can be trusted.
    """

    regression: RegressionReport
    recall_decomposition: RecallDecomposition
    localization: LocalizationSummary
    track_quality: TrackQualityReport
    reference_track_signals: TrackSignals
    candidate_provenance: Provenance
    reference_provenance: Provenance


def evaluate_run(
    candidate: RunResult,
    reference: RunResult,
    tolerance: Tolerance | None = None,
    association: AssociationMode = DEFAULT_ASSOCIATION,
    association_threshold: float | None = None,
    confidence_floor: float = 0.7,
) -> EvaluationReport:
    """Compare ``candidate`` against ``reference`` on detection and track axes in one pass.

    Fans out to ``score_regression`` (banded precision/recall and contextual findings),
    ``decompose_recall`` (recall attribution at the tolerance's IoU threshold and ``confidence_floor``),
    ``summarize_localization`` over the matched pairs (center error, informational),
    ``evaluate_track_quality`` (identity continuity under ``association`` / ``association_threshold``),
    and ``track_signals`` on the reference. ``association_threshold`` of ``None`` uses the association
    measure's default. Never raises on an empty candidate.
    """
    tolerance = tolerance or Tolerance()
    return EvaluationReport(
        regression=score_regression(candidate, reference, tolerance),
        recall_decomposition=decompose_recall(
            candidate.correctness,
            reference.correctness,
            iou_threshold=tolerance.iou_threshold,
            confidence_floor=confidence_floor,
        ),
        localization=summarize_localization(
            matched_detection_pairs(candidate.correctness, reference.correctness, tolerance.iou_threshold)
        ),
        track_quality=evaluate_track_quality(
            candidate, reference, association=association, threshold=association_threshold
        ),
        reference_track_signals=track_signals(reference),
        candidate_provenance=candidate.provenance,
        reference_provenance=reference.provenance,
    )


def evaluate_refs(
    store,
    candidate_ref: str,
    reference_ref: str,
    tolerance: Tolerance | None = None,
    association: AssociationMode = DEFAULT_ASSOCIATION,
    association_threshold: float | None = None,
    confidence_floor: float = 0.7,
) -> EvaluationReport:
    """Resolve two runs from a store by ref and evaluate the candidate against the reference.

    ``store`` must expose ``resolve(ref) -> RunResult``. Delegates to ``evaluate_run``; the
    reference's role (ground truth vs prior run) is decided by what it holds.
    """
    return evaluate_run(
        store.resolve(candidate_ref),
        store.resolve(reference_ref),
        tolerance=tolerance,
        association=association,
        association_threshold=association_threshold,
        confidence_floor=confidence_floor,
    )


def matched_detection_pairs(
    candidate: FrameDetections, reference: FrameDetections, iou_threshold: float
) -> list[tuple[Detection, Detection]]:
    """Matched ``(candidate, reference)`` detection pairs, frame by frame, under the one-to-one IoU
    matching at ``iou_threshold``.

    The pair set is exactly the matches the banded detection scorer counts as ``matched`` at the same
    threshold, so any summary over these pairs describes the associations in the banded
    precision/recall comparison.
    """
    pairs: list[tuple[Detection, Detection]] = []
    for frame_number in sorted(set(candidate) | set(reference)):
        candidate_detections = candidate.get(frame_number, [])
        reference_detections = reference.get(frame_number, [])
        matching = match_boxes_by_iou(
            [detection.box_xywh for detection in candidate_detections],
            [detection.box_xywh for detection in reference_detections],
            iou_threshold,
        )
        for candidate_index, reference_index in matching.matched:
            pairs.append((candidate_detections[candidate_index], reference_detections[reference_index]))
    return pairs


def summarize_localization(pairs: list[tuple[Detection, Detection]]) -> LocalizationSummary:
    """Center-error distribution over matched candidate-reference detection ``pairs``.

    Each pair contributes its center offset in source pixels (euclidean distance between the two box
    centers) and that offset divided by the reference box diagonal. Returns count plus median, p95,
    and max of both distributions. p95 is the linear interpolation between the two closest ranks. An
    empty ``pairs`` yields count 0 and 0.0 for every statistic; a degenerate reference box (zero
    diagonal) contributes 0.0 to the normalized distribution.
    """
    pixel_offsets: list[float] = []
    normalized_offsets: list[float] = []
    for candidate_detection, reference_detection in pairs:
        offset = center_distance(candidate_detection.box_xywh, reference_detection.box_xywh)
        pixel_offsets.append(offset)
        _, _, reference_width, reference_height = reference_detection.box_xywh
        diagonal = math.hypot(reference_width, reference_height)
        normalized_offsets.append(offset / diagonal if diagonal > 0 else 0.0)

    pixel_offsets.sort()
    normalized_offsets.sort()
    return LocalizationSummary(
        count=len(pairs),
        median_pixels=_percentile(pixel_offsets, 0.5),
        p95_pixels=_percentile(pixel_offsets, 0.95),
        max_pixels=_percentile(pixel_offsets, 1.0),
        median_normalized=_percentile(normalized_offsets, 0.5),
        p95_normalized=_percentile(normalized_offsets, 0.95),
        max_normalized=_percentile(normalized_offsets, 1.0),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Value at ``fraction`` of ``sorted_values`` by linear interpolation between the two closest
    ranks. Empty -> 0.0. ``fraction`` of 1.0 returns the maximum."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = rank - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def render_report(report: EvaluationReport) -> str:
    """Render one evidence report without turning comparison thresholds into control flow."""
    lines: list[str] = []
    lines += _provenance_lines("candidate", report.candidate_provenance)
    lines += _provenance_lines("reference", report.reference_provenance)
    lines += _detection_lines(report)
    lines += _localization_lines(report.localization)
    lines += _decomposition_lines(report.recall_decomposition)
    lines += _track_lines(report)
    lines.append("")
    lines.append(
        "findings: "
        f"{len(report.regression.correctness_findings)} correctness, "
        f"{len(report.regression.perf_findings)} performance"
    )
    return "\n".join(lines)


def _provenance_lines(role: str, provenance: Provenance) -> list[str]:
    lines = [
        f"{(role + ':').ljust(11)} git={provenance.git_commit} "
        f"model={provenance.model_hash} config={provenance.config_hash}"
    ]
    extra = provenance.extra or {}
    compact = [f"{key}={extra[key]}" for key in _PROVENANCE_EXTRA_KEYS if key in extra]
    if compact:
        lines.append("  " + " ".join(compact))
    manifest = extra.get("manifest")
    if isinstance(manifest, dict):
        lines.append("  manifest:")
        lines += [f"    {key}={value}" for key, value in manifest.items()]
    return lines


def _detection_lines(report: EvaluationReport) -> list[str]:
    regression = report.regression
    lines = ["", "=== detection (confidence bands, candidate vs reference) ==="]
    lines.append(f"{'band':>14}{'precision':>12}{'recall':>10}{'matched':>10}{'spurious':>10}{'missed':>10}")
    for band in regression.bands:
        lines.append(
            f"{f'[{band.low:.2f},{band.high:.2f})':>14}"
            f"{band.precision:>12.3f}{band.recall:>10.3f}"
            f"{band.matched:>10}{band.spurious:>10}{band.missed:>10}"
        )

    if regression.correctness_findings:
        lines.append("")
        lines.append("correctness findings (drift beyond tolerance):")
        for finding in regression.correctness_findings:
            lines.append(
                f"  band [{finding.band[0]:.2f},{finding.band[1]:.2f}) {finding.metric} "
                f"= {finding.value:.3f} (drop {finding.drop:.3f})"
            )
    else:
        lines.append("")
        lines.append("correctness: no drift beyond tolerance")

    lines.append("")
    lines.append("performance (per-stage, informational on dev CPU):")
    if regression.perf_findings:
        for finding in regression.perf_findings:
            lines.append(
                f"  {finding.stage}: {finding.reference_ms_per_call:.1f} -> "
                f"{finding.candidate_ms_per_call:.1f} ms/call ({finding.fraction * 100:+.0f}%)"
            )
    else:
        lines.append("  no stage regressed beyond tolerance")

    return lines


def _localization_lines(summary: LocalizationSummary) -> list[str]:
    lines = ["", "localization (center error over matched pairs):"]
    if summary.count == 0:
        lines.append("  no matched pairs")
        return lines
    lines.append(f"  matched pairs: {summary.count}")
    lines.append(f"  median (px): {summary.median_pixels:.2f}")
    lines.append(f"  p95 (px): {summary.p95_pixels:.2f}")
    lines.append(f"  max (px): {summary.max_pixels:.2f}")
    lines.append(f"  median (/ref-diagonal): {summary.median_normalized:.3f}")
    lines.append(f"  p95 (/ref-diagonal): {summary.p95_normalized:.3f}")
    lines.append(f"  max (/ref-diagonal): {summary.max_normalized:.3f}")
    return lines


def _decomposition_lines(decomposition: RecallDecomposition) -> list[str]:
    lines = ["", "recall decomposition (why each reference box was or was not found):"]
    lines.append(decomposition.as_table())
    if decomposition.under_localized_best_ious:
        ious = sorted(decomposition.under_localized_best_ious)
        lines.append(
            f"under_localized best-IoU: min={ious[0]:.2f} median={ious[len(ious) // 2]:.2f} max={ious[-1]:.2f}"
        )
        scales = sorted(decomposition.under_localized_scale_ratios)
        offsets = sorted(decomposition.under_localized_center_offsets)
        lines.append(
            f"under_localized candidate/reference area: median={scales[len(scales) // 2]:.2f} "
            f"(<1 = candidate box smaller); center offset/diagonal: median={offsets[len(offsets) // 2]:.2f}"
        )
    if decomposition.under_confident_confidences:
        confs = sorted(decomposition.under_confident_confidences)
        lines.append(
            f"under_confident conf: min={confs[0]:.2f} median={confs[len(confs) // 2]:.2f} max={confs[-1]:.2f}"
        )
    return lines


def _track_lines(report: EvaluationReport) -> list[str]:
    quality = report.track_quality
    lines = ["", "=== tracks (identity continuity, candidate vs reference) ==="]
    lines.append(
        f"idf1={quality.idf1:.3f}  identity_switches={quality.identity_switches}  fragmentation={quality.fragmentation}"
    )
    lines.append(
        f"id_precision={quality.id_precision:.3f}  id_recall={quality.id_recall:.3f}  "
        f"association={quality.association} threshold={quality.association_threshold:g}"
    )
    lines.append("")
    lines += _track_signals_lines(quality.candidate_signals, report.reference_track_signals)
    return lines


def _track_signals_lines(candidate: TrackSignals, reference: TrackSignals) -> list[str]:
    rows = [
        ("track_count", candidate.track_count, reference.track_count),
        ("min_length", candidate.min_length, reference.min_length),
        ("median_length", candidate.median_length, reference.median_length),
        ("max_length", candidate.max_length, reference.max_length),
        ("gap_count", candidate.gap_count, reference.gap_count),
        ("tracks_with_gaps", candidate.tracks_with_gaps, reference.tracks_with_gaps),
        ("max_gap", candidate.max_gap, reference.max_gap),
        ("total_gap_frames", candidate.total_gap_frames, reference.total_gap_frames),
    ]
    lines = [f"{'signal':<18}{'candidate':>12}{'reference':>12}"]
    lines += [f"{name:<18}{str(cand_value):>12}{str(ref_value):>12}" for name, cand_value, ref_value in rows]
    return lines
