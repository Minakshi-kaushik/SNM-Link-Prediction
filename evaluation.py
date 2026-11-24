"""Evaluation protocol for link prediction: split, metrics, experiment runner.

Protocol (identical for every algorithm and dataset)
----------------------------------------------------
1. Start from the preprocessed undirected simple graph G = (V, E).
2. HIDE a random fraction (default 10 %) of the edges as positive test links.
   A random spanning forest is always kept in the training graph, so hiding
   edges never disconnects the graph (only non-forest edges can be hidden).
3. Sample the same number of NEGATIVE test pairs uniformly from the non-edges
   of the ORIGINAL graph G (so no negative is a true or hidden link).
4. Every algorithm scores the SAME candidate pairs (positives + negatives)
   using ONLY the training graph G_train.
5. Metrics
   * AUC-ROC            : ROC-AUC of the scores against positive/negative labels
                          (ties count 1/2, as in scikit-learn).
   * Precision@K        : fraction of the K highest-scored candidates that are
                          hidden edges. Default K = number of hidden edges.
   * Recall@K           : hidden edges found in the top K / all hidden edges
                          (equals Precision@K when K = #positives and the test
                          set is balanced; also reported for K=100 and K=1000).
   * Execution time     : wall-clock seconds to build the index and score all
                          candidate pairs on the training graph.
6. Candidates are shuffled with the seed before ranking so ties (very common
   for CN/JC/PA) are broken randomly but reproducibly, not by candidate order.
7. Repeat for several seeds; report mean and standard deviation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from algorithms import (ALL_ALGORITHMS, RWR, SHORT_NAMES, _resolve_params,
                        normalize_algorithm, score_pairs)

DEFAULT_SEED = 42
DEFAULT_TEST_FRACTION = 0.10
FIXED_KS = (100, 1000)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvaluationResult:
    """Metrics from one actual evaluation run."""

    auc: float
    precision_at_k: float
    execution_time_seconds: float
    recall_at_k: float = float("nan")
    k: int = 0
    extra: dict = field(default_factory=dict)


def evaluate_scores(
    labels: Sequence[int], scores: Sequence[float],
    ranked_labels: Sequence[int] | None = None, k: int | None = None,
    execution_time_seconds: float = 0.0,
) -> EvaluationResult:
    """Compute metrics from labels (1 = hidden true link) and scores.

    ``labels`` and ``scores`` correspond to the same candidate pairs.
    ``ranked_labels`` (labels ordered by score, best first) is optional; when
    omitted it is derived from ``scores`` with a stable sort. ``k`` defaults to
    the number of positive labels.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("labels and scores must have the same non-zero length.")
    if labels.min() == labels.max():
        raise ValueError("labels must contain both positive and negative examples.")
    n_pos = int(labels.sum())
    k = n_pos if k is None else int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if execution_time_seconds < 0:
        raise ValueError("execution_time_seconds cannot be negative.")
    from sklearn.metrics import roc_auc_score  # isolated so algorithms.py stays light

    auc = float(roc_auc_score(labels, scores))
    ranked = (np.asarray(ranked_labels, dtype=int) if ranked_labels is not None
              else labels[np.argsort(-scores, kind="stable")])
    top = ranked[:k]
    hits = float(top.sum())
    extra = {}
    for fixed in FIXED_KS:
        if fixed <= len(ranked):
            extra[f"precision_at_{fixed}"] = float(ranked[:fixed].sum() / fixed)
            extra[f"recall_at_{fixed}"] = float(ranked[:fixed].sum() / n_pos)
    return EvaluationResult(auc, hits / len(top) if len(top) else 0.0,
                            float(execution_time_seconds), hits / n_pos, k, extra)


# --------------------------------------------------------------------------- #
# Train / test split by hiding edges
# --------------------------------------------------------------------------- #
@dataclass
class LinkPredictionSplit:
    train_graph: nx.Graph
    test_pos: list            # hidden true edges
    test_neg: list            # sampled non-edges
    seed: int
    test_fraction: float
    n_original_edges: int
    candidates: list = field(default_factory=list)   # shuffled (u, v) pairs
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def n_test(self) -> int:
        return len(self.test_pos)


def _random_spanning_forest(n: int, edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask over ``edges`` marking a random spanning forest (Kruskal, random order)."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    in_forest = np.zeros(len(edges), dtype=bool)
    for e in rng.permutation(len(edges)):
        ra, rb = find(int(edges[e, 0])), find(int(edges[e, 1]))
        if ra != rb:
            parent[ra] = rb
            in_forest[e] = True
    return in_forest


def make_split(graph: nx.Graph, test_fraction: float = DEFAULT_TEST_FRACTION,
               seed: int = DEFAULT_SEED, keep_connected: bool = True) -> LinkPredictionSplit:
    """Hide ``test_fraction`` of the edges and sample an equal number of non-edges."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if graph.is_directed():
        graph = nx.Graph(graph)
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    edges = np.array([(pos[u], pos[v]) for u, v in graph.edges() if u != v], dtype=np.int64)
    m = len(edges)
    if m < 10:
        raise ValueError("The graph needs at least 10 edges for an evaluation split.")

    eligible = (np.flatnonzero(~_random_spanning_forest(n, edges, rng))
                if keep_connected else np.arange(m))
    n_test = max(1, int(round(test_fraction * m)))
    if n_test > len(eligible):
        n_test = len(eligible)
    if n_test < 1:
        raise ValueError("The graph is a forest; no edge can be hidden without disconnecting it.")
    hidden = rng.choice(eligible, size=n_test, replace=False)
    hidden_mask = np.zeros(m, dtype=bool)
    hidden_mask[hidden] = True

    train = nx.Graph()
    train.add_nodes_from(nodes)
    train.add_edges_from((nodes[a], nodes[b]) for a, b in edges[~hidden_mask])
    test_pos = [(nodes[a], nodes[b]) for a, b in edges[hidden_mask]]

    # negatives: uniform random non-edges of the ORIGINAL graph
    lo, hi = edges.min(axis=1), edges.max(axis=1)
    existing = np.sort(lo * n + hi)
    max_pairs = n * (n - 1) // 2 - m
    n_neg = min(n_test, max_pairs)
    chosen: set[int] = set()
    while len(chosen) < n_neg:
        a = rng.integers(0, n, size=2 * (n_neg - len(chosen)) + 16)
        b = rng.integers(0, n, size=len(a))
        code = np.minimum(a, b) * n + np.maximum(a, b)
        ok = (a != b) & ~np.isin(code, existing)
        for c in code[ok].tolist():
            chosen.add(c)
            if len(chosen) == n_neg:
                break
    test_neg = [(nodes[c // n], nodes[c % n]) for c in sorted(chosen)]

    candidates = test_pos + test_neg
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg), dtype=int)
    order = rng.permutation(len(candidates))     # shuffled -> random tie-breaking
    candidates = [candidates[i] for i in order]
    return LinkPredictionSplit(train, test_pos, test_neg, seed, test_fraction, m,
                               candidates, labels[order])


# --------------------------------------------------------------------------- #
# Running algorithms
# --------------------------------------------------------------------------- #
def evaluate_algorithm(split: LinkPredictionSplit, algorithm: str, k: int | None = None,
                       **params) -> EvaluationResult:
    """Score the split's candidates with one algorithm (on the TRAIN graph) and evaluate."""
    algorithm = normalize_algorithm(algorithm)
    started = time.perf_counter()
    scores = score_pairs(split.train_graph, split.candidates, algorithm, **params)
    elapsed = time.perf_counter() - started
    return evaluate_scores(split.labels, scores, None, k, elapsed)


def run_experiment(
    graph: nx.Graph, dataset: str = "dataset", algorithms: Sequence[str] = ALL_ALGORITHMS,
    seeds: Sequence[int] = (DEFAULT_SEED,), test_fraction: float = DEFAULT_TEST_FRACTION,
    k: int | None = None, rwr_params: dict | None = None, verbose: bool = False,
) -> pd.DataFrame:
    """Run every algorithm on the same splits. One row per (algorithm, seed).

    This is also the function the UI can call for an uploaded dataset:
    ``run_experiment(graph, "my upload", seeds=[42])``.
    """
    rwr_params = dict(rwr_params or {})
    rows = []
    for run_id, seed in enumerate(seeds):
        split = make_split(graph, test_fraction, seed)
        for algorithm in map(normalize_algorithm, algorithms):
            params = rwr_params if algorithm == RWR else {}
            res = evaluate_algorithm(split, algorithm, k, **params)
            used = _resolve_params(algorithm, params)
            row = {
                "dataset": dataset, "algorithm": algorithm, "run_id": run_id,
                "split_seed": seed, "test_fraction": test_fraction,
                "n_nodes": graph.number_of_nodes(), "n_edges": split.n_original_edges,
                "n_train_edges": split.train_graph.number_of_edges(),
                "n_test_pos": len(split.test_pos), "n_test_neg": len(split.test_neg),
                "k": res.k, "auc": res.auc, "precision_at_k": res.precision_at_k,
                "recall_at_k": res.recall_at_k,
                "execution_time_seconds": res.execution_time_seconds,
                "rwr_restart_probability": used.get("restart_probability", np.nan),
                "rwr_max_iterations": used.get("max_iterations", np.nan),
                **res.extra,
            }
            rows.append(row)
            if verbose:
                print(f"[{dataset}] seed={seed} {SHORT_NAMES[algorithm]:>3}  AUC={res.auc:.4f}  "
                      f"P@K={res.precision_at_k:.4f}  time={res.execution_time_seconds:.3f}s", flush=True)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows into one row per (dataset, algorithm): mean and std."""
    metrics = [c for c in ("auc", "precision_at_k", "recall_at_k", "execution_time_seconds",
                           "precision_at_100", "precision_at_1000") if c in runs.columns]
    keep_first = ["k", "test_fraction", "n_nodes", "n_edges", "n_train_edges", "n_test_pos",
                  "n_test_neg", "rwr_restart_probability", "rwr_max_iterations"]
    out = []
    for (dataset, algorithm), g in runs.groupby(["dataset", "algorithm"], sort=False):
        row = {"dataset": dataset, "algorithm": algorithm, "n_runs": len(g),
               "split_seeds": " ".join(map(str, g["split_seed"]))}
        for c in keep_first:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        for c in metrics:
            row[c] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        out.append(row)
    front = ["dataset", "algorithm", "auc", "precision_at_k", "execution_time_seconds"]
    frame = pd.DataFrame(out)
    return frame[front + [c for c in frame.columns if c not in front]]
