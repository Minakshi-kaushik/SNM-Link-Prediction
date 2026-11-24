"""Run the full link-prediction experiment and save the results.

Usage
-----
    python run_experiments.py                       # all datasets found in data/, 10 seeds
    python run_experiments.py --datasets facebook   # one dataset
    python run_experiments.py --edge-list my.csv    # any extra uploaded edge list
    python run_experiments.py --seeds 42 43 44 --test-fraction 0.1 --out-dir results

Outputs (in --out-dir, default ./)
    results.csv            one row per (dataset, algorithm): mean and std over seeds
    results_per_seed.csv   one row per (dataset, algorithm, seed): the raw runs
    experiment_meta.json   dataset statistics, preprocessing steps, seeds, library versions

Every number is produced by actually running the algorithms; nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn

import data_loader as dl
import evaluation as ev
from algorithms import ALL_ALGORITHMS

DEFAULT_SEEDS = tuple(range(42, 52))   # 10 repetitions: 42 ... 51


def pick_datasets(names, edge_lists):
    """Yield (display_name, graph, source_note)."""
    wanted = names or ["facebook"]
    registry = {"facebook": dl.FACEBOOK_NAME}
    missing = []
    for key in wanted:
        if key not in registry:
            raise SystemExit(f"Unknown dataset {key!r}. Choose from {sorted(registry)}.")
        name = registry[key]
        filename = dl.DATASET_FILES[name][0]
        if not (dl.DATA_DIR / filename).is_file():
            missing.append((name, dl.DATA_DIR / filename, dl.DATASET_SOURCES[name]))
            continue
        graph, _ = dl.load_dataset(name)
        yield name, graph, dl.DATASET_SOURCES[name]
    for path in edge_lists:
        graph = dl.load_edge_list(path)
        yield Path(path).stem, graph, f"user file {path}"
    for name, path, url in missing:
        print(f"WARNING: skipped {name}: {path} not found (download: {url})", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", choices=["facebook"], default=None)
    ap.add_argument("--edge-list", nargs="*", default=[], help="extra edge-list files (txt/csv/tsv)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--test-fraction", type=float, default=ev.DEFAULT_TEST_FRACTION)
    ap.add_argument("--rwr-restart", type=float, default=0.15, help="RWR restart probability alpha")
    ap.add_argument("--rwr-max-iter", type=int, default=100, help="RWR maximum power iterations")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rwr_params = {"restart_probability": args.rwr_restart, "max_iterations": args.rwr_max_iter}

    all_runs, meta_datasets = [], {}
    for name, graph, source in pick_datasets(args.datasets, args.edge_list):
        stats = dl.dataset_summary(graph)
        print(f"\n=== {name}: {stats['nodes']} nodes, {stats['edges']} edges "
              f"(avg degree {stats['avg_degree']:.2f}, clustering {stats['avg_clustering']:.3f}) ===", flush=True)
        runs = ev.run_experiment(graph, name, ALL_ALGORITHMS, seeds=args.seeds,
                                 test_fraction=args.test_fraction, rwr_params=rwr_params, verbose=True)
        all_runs.append(runs)
        meta_datasets[name] = {"source": source, "statistics": stats,
                               "preprocessing": graph.graph.get("preprocessing", [])}

    if not all_runs:
        print("ERROR: no dataset could be loaded, so no results were produced.\n"
              f"Place the edge-list files in {dl.DATA_DIR} (see README.md) or pass --edge-list.",
              file=sys.stderr)
        return 1

    runs = pd.concat(all_runs, ignore_index=True)
    summary = ev.summarize_runs(runs)
    runs.to_csv(out_dir / "results_per_seed.csv", index=False)
    summary.to_csv(out_dir / "results.csv", index=False)
    meta = {
        "seeds": args.seeds,
        "test_fraction": args.test_fraction,
        "negative_sampling": "uniform non-edges of the original graph, equal to #hidden edges",
        "k_definition": "K = number of hidden test edges (also reported: K=100, K=1000)",
        "rwr_parameters": rwr_params,
        "datasets": meta_datasets,
        "versions": {"python": platform.python_version(), "networkx": nx.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    show = summary[["dataset", "algorithm", "auc", "auc_std", "precision_at_k", "precision_at_k_std",
                    "recall_at_k", "execution_time_seconds"]]
    print("\n" + show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {out_dir/'results.csv'}, {out_dir/'results_per_seed.csv'}, {out_dir/'experiment_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
