"""Self-contained predict/eval seam for the edge-compute experiment.

Public surface: the canonical contracts, the ResultStore port + JSONL impl, and the regression
scorer. Metric primitives live in ``metrics``/``banded``.
"""

from __future__ import annotations

from spoorpredictioneval.banded import BAND_EDGES, BandResult, banded_detection
from spoorpredictioneval.compare import (
    EvaluationReport,
    evaluate_refs,
    evaluate_run,
    matched_detection_pairs,
    render_report,
    summarize_localization,
)
from spoorpredictioneval.contracts import (
    Detection,
    FrameDetections,
    LocalizationSummary,
    Provenance,
    RunResult,
    StageStat,
    StageTiming,
    Track,
    TrackUpdate,
)
from spoorpredictioneval.jsonl_store import JsonlResultStore
from spoorpredictioneval.metrics import Counts, Matching, iou, match_boxes_by_iou
from spoorpredictioneval.recall_decomposition import (
    RecallDecomposition,
    decompose_recall,
)
from spoorpredictioneval.regression import (
    CorrectnessFinding,
    PerfFinding,
    RegressionReport,
    Tolerance,
    score_against,
    score_regression,
)
from spoorpredictioneval.result_store import ResultStore, content_run_id
from spoorpredictioneval.track_quality import (
    ASSOCIATION_CENTER_DISTANCE,
    ASSOCIATION_IOU,
    DEFAULT_ASSOCIATION,
    AssociationMode,
    TrackQualityReport,
    TrackSignals,
    evaluate_track_quality,
    track_signals,
)
from spoorpredictioneval.track_view import (
    BandPopulation,
    RunView,
    SalienceFilter,
    reconstruct_tracks,
    summarize_run,
)

__version__ = "0.0.1"

__all__ = [
    "BAND_EDGES",
    "BandResult",
    "banded_detection",
    "Detection",
    "FrameDetections",
    "Provenance",
    "RunResult",
    "StageStat",
    "StageTiming",
    "Track",
    "TrackUpdate",
    "BandPopulation",
    "RunView",
    "SalienceFilter",
    "reconstruct_tracks",
    "summarize_run",
    "JsonlResultStore",
    "Counts",
    "Matching",
    "iou",
    "match_boxes_by_iou",
    "CorrectnessFinding",
    "PerfFinding",
    "RegressionReport",
    "Tolerance",
    "score_against",
    "score_regression",
    "ResultStore",
    "content_run_id",
    "EvaluationReport",
    "evaluate_run",
    "evaluate_refs",
    "render_report",
    "matched_detection_pairs",
    "summarize_localization",
    "LocalizationSummary",
    "decompose_recall",
    "RecallDecomposition",
    "evaluate_track_quality",
    "track_signals",
    "TrackQualityReport",
    "TrackSignals",
    "AssociationMode",
    "ASSOCIATION_CENTER_DISTANCE",
    "ASSOCIATION_IOU",
    "DEFAULT_ASSOCIATION",
]
