"""Link-prediction score interfaces and baseline implementations.

Scores are demonstrations on the selected graph. They are not experimental
results and must not be interpreted as evidence about real-world performance.
"""

from __future__ import annotations

import networkx as nx

SUPPORTED_ALGORITHMS = (
    "Common Neighbors",
    "Jaccard Similarity",
    "Adamic-Adar",
    "Preferential Attachment",
)


def score_candidates(
    graph: nx.Graph, source: int | str, algorithm: str
) -> list[tuple[int | str, float]]:
    """Return non-neighbor candidates ranked by an implemented score.

    The source node is excluded. Results are sorted by descending score and
    then by stringified node ID for deterministic tie-breaking.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Source node {source!r} is not in the graph.")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Algorithm {algorithm!r} is not implemented.")

    scorers = {
        "Common Neighbors": nx.common_neighbor_centrality,
        "Jaccard Similarity": nx.jaccard_coefficient,
        "Adamic-Adar": nx.adamic_adar_index,
        "Preferential Attachment": nx.preferential_attachment,
    }
    candidates = [
        (source, node)
        for node in graph
        if node != source and not graph.has_edge(source, node)
    ]
    scores = scorers[algorithm](graph, candidates)
    ranked = [(target, float(score)) for _, target, score in scores]
    return sorted(ranked, key=lambda item: (-item[1], str(item[0])))


def random_walk_with_restart(*_args: object, **_kwargs: object) -> None:
    """Placeholder until the research RWR implementation is integrated."""
    raise NotImplementedError(
        "Random Walk with Restart is planned but is not implemented yet."
    )
