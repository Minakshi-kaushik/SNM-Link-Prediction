r"""Link-prediction algorithms with ONE common interface.

Implemented algorithms (all computed from scratch on a sparse adjacency matrix)
-------------------------------------------------------------------------------
Let Γ(x) be the neighbour set of x and k_x = |Γ(x)| its degree.

* Common Neighbors (CN)           s(x,y) = |Γ(x) ∩ Γ(y)|
* Jaccard Coefficient (JC)        s(x,y) = |Γ(x) ∩ Γ(y)| / |Γ(x) ∪ Γ(y)|
* Adamic-Adar (AA)                s(x,y) = Σ_{z ∈ Γ(x)∩Γ(y)} 1 / ln(k_z)
* Preferential Attachment (PA)    s(x,y) = k_x · k_y
* Random Walk with Restart (RWR)  r_x = α e_x + (1-α) Pᵀ r_x,   P = D⁻¹A
                                  s(x,y) = r_x[y] + r_y[x]      (symmetrised)
  α is the restart probability (``restart_probability``). r_x is computed by
  power iteration, stopped after ``max_iterations`` or when the L1 change is
  below ``tolerance``. On an undirected graph r_y[x] = r_x[y]·k_x/k_y
  (reversibility), so the symmetric score needs only the walk started at x.

Common interface (what the UI should call)
------------------------------------------
    run_link_prediction(graph, algorithm, source=None, top_k=10, **params) -> dict
    predict_links(graph, algorithm, source=None, top_k=10, candidate_pairs=None, **params)
                                                                      -> pandas.DataFrame
    score_pairs(graph, pairs, algorithm, **params)                    -> numpy.ndarray
    score_candidates(graph, source, algorithm, **params)              -> [(target, score)]

``algorithm`` is one of ``ALL_ALGORITHMS`` (or its short name CN/JC/AA/PA/RWR).
RWR parameters (all optional): ``restart_probability`` (default 0.15),
``max_iterations`` (default 100), ``tolerance`` (default 1e-8), ``symmetric``
(default True). Other algorithms ignore these parameters.

Directed graphs are converted to undirected on the fly (an edge exists if it
exists in either direction). Scores are never cached or hardcoded.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

COMMON_NEIGHBORS = "Common Neighbors"
JACCARD = "Jaccard Similarity"
ADAMIC_ADAR = "Adamic-Adar"
PREFERENTIAL_ATTACHMENT = "Preferential Attachment"
RWR = "Random Walk with Restart"

# Kept with its original 4 entries because app.py appends RWR to this tuple itself.
SUPPORTED_ALGORITHMS = (COMMON_NEIGHBORS, JACCARD, ADAMIC_ADAR, PREFERENTIAL_ATTACHMENT)
ALL_ALGORITHMS = (*SUPPORTED_ALGORITHMS, RWR)

ALGORITHM_ALIASES = {
    "CN": COMMON_NEIGHBORS, "JC": JACCARD, "AA": ADAMIC_ADAR,
    "PA": PREFERENTIAL_ATTACHMENT, "RWR": RWR,
    "Jaccard Coefficient": JACCARD, "Jaccard": JACCARD,
}
SHORT_NAMES = {COMMON_NEIGHBORS: "CN", JACCARD: "JC", ADAMIC_ADAR: "AA",
               PREFERENTIAL_ATTACHMENT: "PA", RWR: "RWR"}

DEFAULT_RWR_PARAMS = {
    "restart_probability": 0.15,
    "max_iterations": 100,
    "tolerance": 1e-8,
    "symmetric": True,
}

_ROW_CHUNK = 256  # sources processed at once (memory: chunk x n floats)


# --------------------------------------------------------------------------- #
# Graph index: node <-> integer mapping + sparse adjacency
# --------------------------------------------------------------------------- #
class GraphIndex:
    """Sparse, immutable view of a graph. Build once, score many times."""

    def __init__(self, graph: nx.Graph):
        if isinstance(graph, GraphIndex):  # pragma: no cover - defensive
            raise TypeError("Pass a networkx graph, not a GraphIndex.")
        self.nodes: list = list(graph.nodes())
        self.index: dict = {node: i for i, node in enumerate(self.nodes)}
        n = len(self.nodes)
        edges = [(self.index[u], self.index[v]) for u, v in graph.edges() if u != v]
        if edges:
            e = np.asarray(edges, dtype=np.int64)
            rows = np.concatenate([e[:, 0], e[:, 1]])
            cols = np.concatenate([e[:, 1], e[:, 0]])
        else:
            rows = cols = np.empty(0, dtype=np.int64)
        adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0  # unweighted, symmetric, no self loops
        self.n = n
        self.A: sp.csr_matrix = adjacency
        self.deg: np.ndarray = np.asarray(adjacency.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            log_deg = np.log(self.deg)
        # Adamic-Adar weight; a common neighbour always has degree >= 2.
        self.aa_weight = np.where(self.deg > 1, 1.0 / np.where(log_deg > 0, log_deg, 1.0), 0.0)
        self.inv_deg = np.divide(1.0, self.deg, out=np.zeros(n), where=self.deg > 0)

    def ids(self, nodes: Iterable) -> np.ndarray:
        try:
            return np.fromiter((self.index[x] for x in nodes), dtype=np.int64)
        except KeyError as exc:
            raise nx.NodeNotFound(f"Node {exc.args[0]!r} is not in the graph.") from None


def _as_index(graph) -> GraphIndex:
    return graph if isinstance(graph, GraphIndex) else GraphIndex(graph)


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #
def normalize_algorithm(algorithm: str) -> str:
    name = ALGORITHM_ALIASES.get(algorithm, algorithm)
    if name not in ALL_ALGORITHMS:
        raise ValueError(f"Algorithm {algorithm!r} is not implemented. "
                         f"Choose one of {list(ALL_ALGORITHMS)}.")
    return name


def _resolve_params(algorithm: str, params: dict) -> dict:
    unknown = set(params) - set(DEFAULT_RWR_PARAMS)
    if unknown:
        raise TypeError(f"Unknown parameter(s): {sorted(unknown)}. "
                        f"Valid parameters: {sorted(DEFAULT_RWR_PARAMS)}.")
    if algorithm != RWR:
        return {}
    merged = {**DEFAULT_RWR_PARAMS, **{k: v for k, v in params.items() if v is not None}}
    if not 0.0 < merged["restart_probability"] < 1.0:
        raise ValueError("restart_probability must be strictly between 0 and 1.")
    if int(merged["max_iterations"]) < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if merged["tolerance"] <= 0:
        raise ValueError("tolerance must be positive.")
    merged["max_iterations"] = int(merged["max_iterations"])
    return merged


# --------------------------------------------------------------------------- #
# Random Walk with Restart
# --------------------------------------------------------------------------- #
def rwr_vectors(index: GraphIndex, sources: np.ndarray, restart_probability: float = 0.15,
                max_iterations: int = 100, tolerance: float = 1e-8):
    """Stationary RWR distributions for several source nodes at once.

    Returns ``(X, iterations)`` where column j of X (shape n x len(sources)) is the
    distribution r_s for ``sources[j]`` and ``iterations`` is the number of power
    iterations actually used. Iteration: X <- α·E + (1-α)·A·D⁻¹·X.
    """
    alpha = float(restart_probability)
    sources = np.asarray(sources, dtype=np.int64)
    n, m = index.n, len(sources)
    restart = np.zeros((n, m))
    restart[sources, np.arange(m)] = 1.0
    x = restart.copy()
    used = 0
    for used in range(1, max_iterations + 1):
        x_new = alpha * restart + (1.0 - alpha) * (index.A @ (x * index.inv_deg[:, None]))
        delta = np.abs(x_new - x).sum(axis=0).max()
        x = x_new
        if delta < tolerance:
            break
    return x, used


# --------------------------------------------------------------------------- #
# Scoring all targets for a block of source rows  (dense: len(rows) x n)
# --------------------------------------------------------------------------- #
def _score_rows(index: GraphIndex, rows: np.ndarray, algorithm: str, params: dict) -> np.ndarray:
    A, deg = index.A, index.deg
    block = A[rows]
    if algorithm == COMMON_NEIGHBORS:
        return (block @ A).toarray()
    if algorithm == JACCARD:
        common = (block @ A).toarray()
        union = deg[rows][:, None] + deg[None, :] - common
        return np.divide(common, union, out=np.zeros_like(common), where=union > 0)
    if algorithm == ADAMIC_ADAR:
        return (block @ sp.diags(index.aa_weight) @ A).toarray()
    if algorithm == PREFERENTIAL_ATTACHMENT:
        return deg[rows][:, None] * deg[None, :]
    # RWR
    x, _ = rwr_vectors(index, rows, params["restart_probability"],
                       params["max_iterations"], params["tolerance"])
    scores = x.T  # scores[i, t] = r_{rows[i]}[t]
    if params["symmetric"]:
        scores = scores * (1.0 + deg[rows][:, None] * index.inv_deg[None, :])
    return scores


# --------------------------------------------------------------------------- #
# Scoring explicit node pairs (used by the evaluation protocol)
# --------------------------------------------------------------------------- #
def _greedy_sources(ii: np.ndarray, jj: np.ndarray):
    """Pick few endpoints so that every pair has one chosen endpoint (greedy cover)."""
    incident: dict[int, list[int]] = {}
    for p, (a, b) in enumerate(zip(ii.tolist(), jj.tolist())):
        incident.setdefault(a, []).append(p)
        incident.setdefault(b, []).append(p)
    covered = np.zeros(len(ii), dtype=bool)
    source_of = np.full(len(ii), -1, dtype=np.int64)
    for node in sorted(incident, key=lambda v: -len(incident[v])):
        todo = [p for p in incident[node] if not covered[p]]
        if todo:
            covered[todo] = True
            source_of[todo] = node
    return source_of


def _score_pairs_index(index: GraphIndex, ii: np.ndarray, jj: np.ndarray,
                       algorithm: str, params: dict) -> np.ndarray:
    if len(ii) == 0:
        return np.empty(0)
    A, deg = index.A, index.deg
    if algorithm == PREFERENTIAL_ATTACHMENT:
        return deg[ii] * deg[jj]
    if algorithm in (COMMON_NEIGHBORS, JACCARD, ADAMIC_ADAR):
        out = np.empty(len(ii))
        for s in range(0, len(ii), 4000):
            a, b = ii[s:s + 4000], jj[s:s + 4000]
            both = A[a].multiply(A[b]).tocsr()
            if algorithm == ADAMIC_ADAR:
                out[s:s + 4000] = both @ index.aa_weight
                continue
            common = np.asarray(both.sum(axis=1)).ravel()
            if algorithm == COMMON_NEIGHBORS:
                out[s:s + 4000] = common
            else:
                union = deg[a] + deg[b] - common
                out[s:s + 4000] = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
        return out
    # RWR
    if params["symmetric"]:
        source_of = _greedy_sources(ii, jj)
        target_of = np.where(source_of == ii, jj, ii)
    else:
        source_of, target_of = ii, jj
    uniq = np.unique(source_of)
    out = np.empty(len(ii))
    for s in range(0, len(uniq), _ROW_CHUNK):
        chunk = uniq[s:s + _ROW_CHUNK]
        x, _ = rwr_vectors(index, chunk, params["restart_probability"],
                           params["max_iterations"], params["tolerance"])
        col_of = {int(node): c for c, node in enumerate(chunk)}
        sel = np.flatnonzero(np.isin(source_of, chunk))
        cols = np.fromiter((col_of[int(source_of[p])] for p in sel), dtype=np.int64, count=len(sel))
        value = x[target_of[sel], cols]
        if params["symmetric"]:
            t_deg = deg[target_of[sel]]
            value = value * (1.0 + np.divide(deg[source_of[sel]], t_deg,
                                             out=np.zeros(len(sel)), where=t_deg > 0))
        out[sel] = value
    return out


def score_pairs(graph, pairs: Sequence[tuple], algorithm: str, **params) -> np.ndarray:
    """Score explicit node pairs. Returns an array aligned with ``pairs``.

    ``graph`` may be a networkx graph or a prebuilt ``GraphIndex``.
    """
    algorithm = normalize_algorithm(algorithm)
    params = _resolve_params(algorithm, params)
    index = _as_index(graph)
    pairs = list(pairs)
    ii = index.ids(u for u, _ in pairs)
    jj = index.ids(v for _, v in pairs)
    return _score_pairs_index(index, ii, jj, algorithm, params)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _frame(index: GraphIndex, algorithm: str, u_idx, v_idx, scores) -> pd.DataFrame:
    frame = pd.DataFrame({
        "rank": np.arange(1, len(scores) + 1),
        "node_u": [index.nodes[i] for i in u_idx],
        "node_v": [index.nodes[i] for i in v_idx],
        "score": np.asarray(scores, dtype=float),
    })
    frame["algorithm"] = algorithm
    return frame


def predict_links(graph, algorithm: str, source=None, top_k: int | None = 10,
                  candidate_pairs: Sequence[tuple] | None = None, **params) -> pd.DataFrame:
    """Rank missing links. Output columns: rank, node_u, node_v, score, algorithm.

    Modes
    -----
    * ``candidate_pairs`` given : score exactly those pairs.
    * ``source`` given          : rank every non-neighbour of ``source``.
    * neither                   : global top-K over all non-adjacent pairs
                                  (processed in blocks; RWR is slow on big graphs).
    Ties are broken deterministically by node order. ``top_k=None`` returns all.
    """
    algorithm = normalize_algorithm(algorithm)
    params = _resolve_params(algorithm, params)
    index = _as_index(graph)
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer (or None for all).")

    if candidate_pairs is not None:
        ii = index.ids(u for u, _ in candidate_pairs)
        jj = index.ids(v for _, v in candidate_pairs)
        scores = _score_pairs_index(index, ii, jj, algorithm, params)
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
        return _frame(index, algorithm, ii[order], jj[order], scores[order])

    if source is not None:
        if source not in index.index:
            raise nx.NodeNotFound(f"Source node {source!r} is not in the graph.")
        s = index.index[source]
        row = _score_rows(index, np.array([s]), algorithm, params)[0]
        excluded = index.A[s].toarray().ravel() > 0
        excluded[s] = True
        candidates = np.flatnonzero(~excluded)
        order = candidates[np.argsort(-row[candidates], kind="stable")][:top_k]
        return _frame(index, algorithm, np.full(len(order), s), order, row[order])

    # global top-K
    k = top_k if top_k is not None else 100
    cols = np.arange(index.n)
    best_s, best_u, best_v = [], [], []
    for start in range(0, index.n, _ROW_CHUNK):
        rows = np.arange(start, min(index.n, start + _ROW_CHUNK))
        block = _score_rows(index, rows, algorithm, params).astype(float)
        mask = (index.A[rows].toarray() > 0) | (cols[None, :] <= rows[:, None])
        block[mask] = -np.inf
        flat = block.ravel()
        order = np.argsort(-flat, kind="stable")[:k]
        order = order[np.isfinite(flat[order])]
        best_s.append(flat[order])
        best_u.append(rows[order // index.n])
        best_v.append(order % index.n)
    s_all, u_all, v_all = map(np.concatenate, (best_s, best_u, best_v))
    order = np.lexsort((v_all, u_all, -s_all))[:k]
    return _frame(index, algorithm, u_all[order], v_all[order], s_all[order])


def score_candidates(graph: nx.Graph, source, algorithm: str, **params) -> list[tuple]:
    """Backward-compatible helper used by the current dashboard.

    Returns ALL non-neighbours of ``source`` as ``[(target, score), ...]``, best first.
    """
    frame = predict_links(graph, algorithm, source=source, top_k=None, **params)
    return list(zip(frame["node_v"].tolist(), frame["score"].astype(float).tolist()))


def run_link_prediction(graph: nx.Graph, algorithm: str, source=None, top_k: int = 10,
                        **params) -> dict:
    """One-call entry point for the UI.

    Returns a dict with: ``predictions`` (DataFrame), ``ranked`` ([(target, score)]
    when a source is given), ``algorithm``, ``params`` (the parameters actually
    used), ``source``, ``elapsed_seconds`` (measured) and ``n_nodes``/``n_edges``.
    """
    algorithm = normalize_algorithm(algorithm)
    used = _resolve_params(algorithm, params)
    started = time.perf_counter()
    frame = predict_links(graph, algorithm, source=source, top_k=top_k, **params)
    elapsed = time.perf_counter() - started
    return {
        "algorithm": algorithm,
        "params": used,
        "source": source,
        "predictions": frame,
        "ranked": list(zip(frame["node_v"].tolist(), frame["score"].tolist())) if source is not None else None,
        "elapsed_seconds": elapsed,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
    }
