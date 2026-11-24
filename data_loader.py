"""Load built-in demonstration data and the planned Email-Eu-core dataset."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_karate_club() -> nx.Graph:
    """Return NetworkX's small, deterministic graph for UI demonstrations."""
    return nx.karate_club_graph()


def load_email_eu_core() -> nx.Graph:
    """Load an edge-list file if the user has placed it under data/.

    Expected local filename: email-Eu-core.txt (whitespace-delimited edges).
    The dataset is not downloaded automatically.
    """
    path = DATA_DIR / "email-Eu-core.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Email-Eu-core data was not found at {path}. Add the dataset edge "
            "list there to enable it. The Karate Club graph remains available "
            "for UI testing."
        )
    graph = nx.read_edgelist(path, nodetype=int, comments="#")
    if graph.number_of_nodes() == 0:
        raise ValueError(f"The dataset file at {path} contains no graph nodes.")
    return graph


def load_dataset(name: str) -> tuple[nx.Graph, bool]:
    """Return (graph, is_demonstration) for a named dataset."""
    if name == "Karate Club (demo)":
        return load_karate_club(), True
    if name == "Email-Eu-core":
        return load_email_eu_core(), False
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

    Values are kept as read by pandas (so numeric columns remain numeric and
    mixed/string identifiers remain strings). Empty endpoints are rejected.
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
