"""Dataset loading and preprocessing for the link-prediction project.

Public API (used by the UI and by the experiments)
--------------------------------------------------
load_dataset(name)                      -> (graph, is_demonstration)
load_edge_list(path_or_buffer, ...)     -> nx.Graph   (any uploaded TXT/CSV/TSV edge list)
graph_from_edge_table(frame, src, dst)  -> nx.Graph   (UI upload helper, unchanged signature)
preprocess_graph(graph, ...)            -> nx.Graph   (undirected, simple, optional LCC)
dataset_summary(graph)                  -> dict       (nodes, edges, density, clustering ...)

Datasets
--------
* ``Facebook (ego-Facebook)``  MAIN experiment. SNAP "Social circles: Facebook",
  file ``data/facebook_combined.txt``. 4,039 nodes, 88,234 undirected edges.
  Source: J. McAuley and J. Leskovec, "Learning to Discover Social Circles in
  Ego Networks", NIPS 2012. https://snap.stanford.edu/data/ego-Facebook.html
* ``Karate Club (demo)``       tiny graph used only for UI demonstrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import networkx as nx
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

FACEBOOK_NAME = "Facebook (ego-Facebook)"
KARATE_NAME = "Karate Club (demo)"

DATASET_FILES = {
    FACEBOOK_NAME: ("facebook_combined.txt", False),  # (filename, natively directed?)
}

DATASET_SOURCES = {
    FACEBOOK_NAME: "https://snap.stanford.edu/data/ego-Facebook.html",
    KARATE_NAME: "Zachary (1977), as shipped with NetworkX",
}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_graph(
    graph: nx.Graph,
    *,
    to_undirected: bool = True,
    remove_self_loops: bool = True,
    largest_component: bool = False,
) -> nx.Graph:
    """Return a clean, simple, unweighted, undirected graph.

    Steps (recorded in ``graph.graph['preprocessing']``):
      1. directed -> undirected (an edge exists if it exists in either direction)
      2. duplicate / reciprocal edges collapse (simple graph)
      3. self-loops removed
      4. optionally keep only the largest connected component
      5. edge attributes are dropped (the algorithms are unweighted)
    """
    steps: list[str] = []
    if graph.is_directed():
        if not to_undirected:
            raise ValueError("The algorithms in this project operate on undirected graphs.")
        steps.append("directed->undirected")
    clean = nx.Graph()
    clean.add_nodes_from(graph.nodes())
    clean.add_edges_from(graph.edges())
    steps.append("duplicates collapsed")
    if remove_self_loops:
        loops = list(nx.selfloop_edges(clean))
        clean.remove_edges_from(loops)
        steps.append(f"removed {len(loops)} self-loops")
    if largest_component and clean.number_of_nodes():
        keep = max(nx.connected_components(clean), key=len)
        dropped = clean.number_of_nodes() - len(keep)
        clean = clean.subgraph(keep).copy()
        steps.append(f"kept largest connected component (dropped {dropped} nodes)")
    clean.graph["preprocessing"] = steps
    return clean


def dataset_summary(graph: nx.Graph) -> dict:
    """Basic statistics for the paper / dashboard (computed, never hardcoded)."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    return {
        "nodes": n,
        "edges": m,
        "directed": graph.is_directed(),
        "density": nx.density(graph) if n > 1 else 0.0,
        "avg_degree": (2 * m / n) if n else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_clustering": nx.average_clustering(graph) if n else 0.0,
        "connected_components": nx.number_connected_components(graph) if n else 0,
        "largest_component_nodes": len(max(nx.connected_components(graph), key=len)) if n else 0,
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_edge_list(
    source: str | Path | IO,
    *,
    directed: bool = False,
    largest_component: bool = False,
) -> nx.Graph:
    """Load ANY edge list (TXT / CSV / TSV; space, comma, tab or ; separated).

    Lines starting with ``#`` or ``%`` are comments. Only the first two columns
    are used as (source, target); extra columns (weights, timestamps) are ignored.
    Node IDs become integers when every ID is numeric, otherwise they stay strings.
    A textual header row (e.g. ``source,target``) is detected and dropped.
    """
    frame = pd.read_csv(
        source,
        sep=r"[,\s;]+",
        engine="python",
        header=None,
        usecols=[0, 1],
        names=["source", "target"],
        dtype=str,
        skip_blank_lines=True,
    )
    frame = frame.dropna()
    frame = frame[~frame["source"].str.startswith(("#", "%"))]
    if frame.empty:
        raise ValueError("No edges were found in the file.")
    numeric = frame["source"].str.fullmatch(r"-?\d+") & frame["target"].str.fullmatch(r"-?\d+")
    if numeric.all():
        frame = frame.astype(int)
    elif numeric.iloc[1:].all() and not numeric.iloc[0]:  # header row
        frame = frame.iloc[1:].astype(int)
    graph = graph_from_edge_table(frame, "source", "target", directed=directed)
    return preprocess_graph(graph, largest_component=largest_component)


def load_karate_club() -> nx.Graph:
    """NetworkX's small deterministic graph, for UI demonstrations only."""
    return preprocess_graph(nx.karate_club_graph())


def _load_named_file(name: str, *, largest_component: bool) -> nx.Graph:
    filename, directed = DATASET_FILES[name]
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} was not found at {path}. Download it from {DATASET_SOURCES[name]} "
            f"and place the edge list at that path."
        )
    graph = load_edge_list(path, directed=directed, largest_component=largest_component)
    graph.graph["name"] = name
    return graph


def load_facebook(largest_component: bool = True) -> nx.Graph:
    """MAIN dataset: SNAP ego-Facebook (already connected and undirected)."""
    return _load_named_file(FACEBOOK_NAME, largest_component=largest_component)


def available_datasets() -> list[str]:
    """Names that ``load_dataset`` can currently load (files present on disk)."""
    names = [KARATE_NAME]
    names += [n for n, (f, _) in DATASET_FILES.items() if (DATA_DIR / f).is_file()]
    return names


def load_dataset(name: str) -> tuple[nx.Graph, bool]:
    """Return ``(graph, is_demonstration)`` for a named dataset."""
    if name == KARATE_NAME:
        return load_karate_club(), True
    if name == FACEBOOK_NAME:
        return load_facebook(), False
    raise ValueError(f"Unknown dataset: {name}")


def graph_from_edge_table(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
    *,
    directed: bool = False,
    remove_duplicates: bool = True,
    remove_self_loops: bool = True,
) -> nx.Graph:
    """Build a graph from selected table columns while preserving node IDs.

    Values are kept as read by pandas (numeric columns stay numeric, mixed or
    string identifiers stay strings). Empty endpoints are rejected.
    """
    if source_column not in frame.columns or target_column not in frame.columns:
        raise ValueError("Choose source and target columns present in the uploaded table.")
    if source_column == target_column:
        raise ValueError("Source and target columns must be different.")
    edges = frame[[source_column, target_column]].dropna()
    if edges.empty:
        raise ValueError("No rows with both source and target values were found.")
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for source, target in edges.itertuples(index=False, name=None):
        if remove_self_loops and source == target:
            continue
        if remove_duplicates and graph.has_edge(source, target):
            continue
        graph.add_edge(source, target)
    if graph.number_of_edges() == 0:
        raise ValueError("No edges remain after applying the selected cleanup options.")
    return graph
