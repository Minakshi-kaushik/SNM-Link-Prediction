"""Correctness checks: every algorithm against an independent reference.

CN / JC / AA / PA  -> compared with networkx's own implementations.
RWR                -> compared with the exact closed-form solution
                      r_x = α (I - (1-α) A D^-1)^-1 e_x  (dense linear solve).
Also checks the shared interface and the evaluation split.

Run:  python validate_algorithms.py
"""

from __future__ import annotations

import io
import sys

import networkx as nx
import numpy as np

import algorithms as alg
import evaluation as ev
from data_loader import preprocess_graph, load_edge_list

TOL = 1e-9
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def test_graphs() -> dict[str, nx.Graph]:
    return {
        "karate": preprocess_graph(nx.karate_club_graph()),
        "powerlaw_cluster(300)": preprocess_graph(nx.powerlaw_cluster_graph(300, 4, 0.3, seed=1)),
        "les_miserables": preprocess_graph(nx.convert_node_labels_to_integers(nx.les_miserables_graph())),
    }


def sample_pairs(g: nx.Graph, n: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    out = set()
    while len(out) < n:
        u, v = rng.choice(len(nodes), 2, replace=False)
        out.add((nodes[u], nodes[v]))
    return sorted(out)


def exact_rwr(g: nx.Graph, alpha: float):
    nodes = list(g.nodes())
    A = nx.to_numpy_array(g, nodelist=nodes)
    deg = A.sum(axis=1)
    W = A / np.where(deg > 0, deg, 1.0)[None, :]
    R = alpha * np.linalg.inv(np.eye(len(nodes)) - (1 - alpha) * W)
    return R, nodes


def main() -> int:
    for gname, g in test_graphs().items():
        pairs = sample_pairs(g, 400, seed=7)
        nxs = {
            alg.COMMON_NEIGHBORS: [len(list(nx.common_neighbors(g, u, v))) for u, v in pairs],
            alg.JACCARD: [p for _, _, p in nx.jaccard_coefficient(g, pairs)],
            alg.ADAMIC_ADAR: [p for _, _, p in nx.adamic_adar_index(g, pairs)],
            alg.PREFERENTIAL_ATTACHMENT: [p for _, _, p in nx.preferential_attachment(g, pairs)],
        }
        for a, ref in nxs.items():
            mine = alg.score_pairs(g, pairs, a)
            err = float(np.max(np.abs(mine - np.asarray(ref, dtype=float))))
            check(f"{gname}: {alg.SHORT_NAMES[a]} == networkx", err < TOL, f"(max abs err {err:.2e})")

        for alpha in (0.15, 0.5):
            R, nodes = exact_rwr(g, alpha)
            pos = {v: i for i, v in enumerate(nodes)}
            exact = np.array([R[pos[v], pos[u]] + R[pos[u], pos[v]] for u, v in pairs])
            mine = alg.score_pairs(g, pairs, alg.RWR, restart_probability=alpha,
                                   max_iterations=2000, tolerance=1e-14)
            err = float(np.max(np.abs(mine - exact)))
            check(f"{gname}: RWR (alpha={alpha}) == exact solve", err < 1e-9, f"(max abs err {err:.2e})")

    g = preprocess_graph(nx.karate_club_graph())
    for a in alg.ALL_ALGORITHMS:
        res = alg.run_link_prediction(g, a, source=0, top_k=5)
        df = res["predictions"]
        check(f"interface: run_link_prediction[{alg.SHORT_NAMES[a]}] columns",
              list(df.columns) == ["rank", "node_u", "node_v", "score", "algorithm"])
        check(f"interface: {alg.SHORT_NAMES[a]} sorted best-first",
              bool((np.diff(df["score"].to_numpy()) <= 1e-15).all()))
        check(f"interface: {alg.SHORT_NAMES[a]} excludes existing neighbours of source",
              not any(g.has_edge(0, v) or v == 0 for v in df["node_v"]))
        glob = alg.predict_links(g, a, top_k=5)
        check(f"interface: {alg.SHORT_NAMES[a]} global top-K has 5 rows, no existing edges",
              len(glob) == 5 and not any(g.has_edge(u, v) for u, v in zip(glob.node_u, glob.node_v)))
    full = alg.score_candidates(g, 0, alg.ADAMIC_ADAR)
    direct = alg.score_pairs(g, [(0, t) for t, _ in full[:10]], alg.ADAMIC_ADAR)
    check("interface: source-mode scores == pair-mode scores",
          np.allclose([s for _, s in full[:10]], direct))
    s1 = alg.score_pairs(g, [(0, 9)], alg.RWR, restart_probability=0.15)[0]
    s2 = alg.score_pairs(g, [(0, 9)], alg.RWR, restart_probability=0.6)[0]
    s3 = alg.score_pairs(g, [(0, 9)], alg.RWR, max_iterations=1)[0]
    check("RWR: restart_probability changes scores", abs(s1 - s2) > 1e-6)
    check("RWR: max_iterations changes scores", abs(s1 - s3) > 1e-6)
    for bad, exc in ((lambda: alg.predict_links(g, "Nope"), ValueError),
                     (lambda: alg.predict_links(g, "CN", source=999), nx.NodeNotFound),
                     (lambda: alg.predict_links(g, "RWR", restart_probability=1.5), ValueError),
                     (lambda: alg.predict_links(g, "CN", top_k=0), ValueError)):
        try:
            bad()
            check("error handling", False, "(no exception)")
        except exc:
            check("error handling raises " + exc.__name__, True)
    dg = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    out = alg.run_link_prediction(dg, "CN", top_k=3)["predictions"]
    check("directed graph with string ids accepted", len(out) > 0)

    g = preprocess_graph(nx.powerlaw_cluster_graph(400, 4, 0.3, seed=2))
    sp = ev.make_split(g, 0.1, seed=42)
    check("split: positives are real edges", all(g.has_edge(u, v) for u, v in sp.test_pos))
    check("split: positives removed from train", not any(sp.train_graph.has_edge(u, v) for u, v in sp.test_pos))
    check("split: negatives are non-edges of ORIGINAL graph", not any(g.has_edge(u, v) for u, v in sp.test_neg))
    check("split: balanced classes", len(sp.test_pos) == len(sp.test_neg))
    check("split: train stays connected", nx.is_connected(sp.train_graph) == nx.is_connected(g))
    check("split: train+test edge count", sp.train_graph.number_of_edges() + len(sp.test_pos) == g.number_of_edges())
    sp2 = ev.make_split(g, 0.1, seed=42)
    check("split: reproducible for same seed", sp.candidates == sp2.candidates)
    check("split: different seed differs", ev.make_split(g, 0.1, seed=43).candidates != sp.candidates)

    labels = [1, 0, 1, 0, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    r = ev.evaluate_scores(labels, scores, k=3)
    check("metrics: AUC hand example", abs(r.auc - 5 / 9) < 1e-12, f"(got {r.auc:.4f}, expected 0.5556)")
    check("metrics: Precision@3 hand example", abs(r.precision_at_k - 2 / 3) < 1e-12)
    check("metrics: Recall@3 hand example", abs(r.recall_at_k - 2 / 3) < 1e-12)

    runs = ev.run_experiment(g, "synthetic", seeds=[1, 2], verbose=False)
    check("run_experiment: 5 algorithms x 2 seeds rows", len(runs) == 10)
    check("run_experiment: AUC in [0,1] for all", bool(runs["auc"].between(0, 1).all()))
    check("run_experiment: link prediction beats chance (AA, CN, RWR)",
          bool((runs[runs.algorithm.isin([alg.ADAMIC_ADAR, alg.COMMON_NEIGHBORS, alg.RWR])]["auc"] > 0.55).all()))
    check("summarize_runs: 5 rows", len(ev.summarize_runs(runs)) == 5)

    text = "# comment\nsource,target\n1,2\n2,3\n3,1\n3,3\n1,2\n"
    gl = load_edge_list(io.StringIO(text))
    check("loader: header/comment/self-loop/duplicate handled",
          gl.number_of_edges() == 3 and gl.number_of_nodes() == 3)
    gl = load_edge_list(io.StringIO("1 2\n2 3\n3 1\n"))
    check("loader: space-separated", gl.number_of_edges() == 3)
    gl = load_edge_list(io.StringIO("a\tb\nb\tc\n"))
    check("loader: tab-separated string ids", set(gl.nodes()) == {"a", "b", "c"})

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
