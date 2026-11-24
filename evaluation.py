"""Evaluation interfaces for future held-out link-prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class EvaluationResult:
    """Metrics from one actual evaluation run."""

    auc: float
    precision_at_k: float
    execution_time_seconds: float


def evaluate_scores(
    labels: Sequence[int], scores: Sequence[float],
    ranked_labels: Sequence[int], k: int, execution_time_seconds: float,
) -> EvaluationResult:
    """Compute metrics from experiment labels and model scores.

    ``labels`` and ``scores`` must correspond to the same held-out candidate
    links. ``ranked_labels`` must be ordered by predicted score, best first.
    """
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if execution_time_seconds < 0:
        raise ValueError("execution_time_seconds cannot be negative.")
    # Keep sklearn isolated to evaluation so recommendation code can be used
    # independently while experiments and their dependencies are prepared.
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    top_k = ranked_labels[:k]
    precision = float(sum(top_k) / len(top_k)) if top_k else 0.0
    return EvaluationResult(auc, precision, float(execution_time_seconds))
