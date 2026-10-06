"""Compare a candidate run with a reference run on two axes.

Correctness: directional confidence-banded precision/recall of candidate versus reference. A
band whose precision or recall falls below ``1 - tolerance`` is reported as a finding.

Performance: per-stage ms/call delta. A stage slower than the reference by more than the tolerated
fraction is reported as a finding. These numbers are dev-CPU interim ``--stage-timing`` and do not
answer the Jetson throughput claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spoorpredictioneval.banded import BAND_EDGES, BandResult, banded_detection
from spoorpredictioneval.contracts import RunResult


@dataclass(frozen=True)
class Tolerance:
    iou_threshold: float = 0.5
    max_precision_drop: float = 0.02
    max_recall_drop: float = 0.02
    max_stage_regression_fraction: float = 0.25
    band_edges: tuple[float, ...] = BAND_EDGES


@dataclass(frozen=True)
class CorrectnessFinding:
    band: tuple[float, float]
    metric: str  # "precision" or "recall"
    value: float
    drop: float  # 1.0 - value


@dataclass(frozen=True)
class PerfFinding:
    stage: str
    reference_ms_per_call: float
    candidate_ms_per_call: float
    fraction: float  # (candidate - reference) / reference


@dataclass(frozen=True)
class RegressionReport:
    bands: list[BandResult]
    correctness_findings: list[CorrectnessFinding] = field(default_factory=list)
    perf_findings: list[PerfFinding] = field(default_factory=list)


def score_regression(
    candidate: RunResult, reference: RunResult, tolerance: Tolerance | None = None
) -> RegressionReport:
    tolerance = tolerance or Tolerance()
    bands = banded_detection(
        candidate.correctness, reference.correctness, tolerance.iou_threshold, tolerance.band_edges
    )

    correctness_findings: list[CorrectnessFinding] = []
    for band in bands:
        precision_drop = 1.0 - band.precision
        if precision_drop > tolerance.max_precision_drop:
            correctness_findings.append(
                CorrectnessFinding((band.low, band.high), "precision", band.precision, precision_drop)
            )
        recall_drop = 1.0 - band.recall
        if recall_drop > tolerance.max_recall_drop:
            correctness_findings.append(CorrectnessFinding((band.low, band.high), "recall", band.recall, recall_drop))

    perf_findings: list[PerfFinding] = []
    for stage, candidate_stat in candidate.performance.items():
        reference_stat = reference.performance.get(stage)
        if reference_stat is None or reference_stat.ms_per_call <= 0:
            continue
        fraction = (candidate_stat.ms_per_call - reference_stat.ms_per_call) / reference_stat.ms_per_call
        if fraction > tolerance.max_stage_regression_fraction:
            perf_findings.append(PerfFinding(stage, reference_stat.ms_per_call, candidate_stat.ms_per_call, fraction))

    return RegressionReport(bands=bands, correctness_findings=correctness_findings, perf_findings=perf_findings)


def score_against(
    store, candidate_ref: str, reference_ref: str, tolerance: Tolerance | None = None
) -> RegressionReport:
    """Resolve two runs from a store by ref and score the candidate against the reference.

    Accuracy (reference is ground truth) and regression (reference is a prior run) are the same call;
    the reference's role is decided by what it holds.
    """
    return score_regression(store.resolve(candidate_ref), store.resolve(reference_ref), tolerance)
