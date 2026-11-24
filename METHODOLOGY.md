# Algorithms, Dataset, and Experimental Methodology

Sparse-matrix implementations, unified dashboard interface, validation suite, and reproducible evaluation pipeline for five link-prediction algorithms:

1. **Common Neighbors (CN)**
2. **Jaccard Coefficient / Jaccard Similarity (JC)**
3. **Adamic-Adar (AA)**
4. **Preferential Attachment (PA)**
5. **Random Walk with Restart (RWR)**

All predictions, rankings, and evaluation metrics are computed dynamically from actual algorithm execution (nothing is hardcoded or cached).

---

## Added / Updated Files

| File | Purpose |
|---|---|
| `algorithms.py` | Sparse-matrix implementations of CN, JC, AA, PA, and RWR with a unified API (`run_link_prediction`, `predict_links`, `score_pairs`, `score_candidates`). |
| `evaluation.py` | Train-test splitting (with spanning-forest connectivity preservation), negative sampling, AUC-ROC, Precision@K, Recall@K, execution timing, and multi-seed experiment runner (`run_experiment`, `summarize_runs`). |
| `data_loader.py` | Dataset loaders and preprocessing (`load_dataset`, `load_edge_list`, `graph_from_edge_table`, `preprocess_graph`, `dataset_summary`) supporting arbitrary uploaded TXT/CSV/TSV edge lists and DataFrames. |
| `run_experiments.py` | CLI script to run multi-seed experiments and write CSV/JSON results. |
| `validate_algorithms.py` | Independent correctness checks comparing CN/JC/AA/PA against NetworkX and RWR against exact dense linear solve, plus split, metric, interface, and loader tests. |
| `results/results.csv` | Mean and standard deviation of experimental metrics across 10 seeds for each algorithm. |
| `results/results_per_seed.csv` | Raw per-seed experimental results (50 rows: 5 algorithms × 10 seeds). |
| `results/experiment_meta.json` | Recorded dataset statistics, preprocessing steps, seeds, RWR parameters, and package versions. |

---

## Algorithm Explanations & Mathematical Formulations

Let $G = (V, E)$ be a simple, unweighted, undirected graph. Let $\Gamma(x)$ denote the neighbor set of node $x$, and $k_x = |\Gamma(x)|$ denote its degree. Directed inputs are converted to undirected graphs on the fly (an edge exists if it exists in either direction).

### 1. Common Neighbors (CN)
Measures the count of shared neighbors between nodes $x$ and $y$. Nodes with more mutual connections are more likely to form a direct link.

$$s_{\text{CN}}(x, y) = |\Gamma(x) \cap \Gamma(y)|$$

### 2. Jaccard Coefficient (JC)
Normalizes Common Neighbors by the size of the union of both neighborhoods, penalizing high-degree nodes that share neighbors purely due to high connectivity.

$$s_{\text{JC}}(x, y) = \frac{|\Gamma(x) \cap \Gamma(y)|}{|\Gamma(x) \cup \Gamma(y)|} = \frac{|\Gamma(x) \cap \Gamma(y)|}{k_x + k_y - |\Gamma(x) \cap \Gamma(y)|}$$

(Defined as $0$ when $|\Gamma(x) \cup \Gamma(y)| = 0$.)

### 3. Adamic-Adar (AA)
Weights each shared neighbor $z$ inversely by the logarithm of its degree, giving higher importance to shared neighbors that are selective (low-degree) rather than hubs.

$$s_{\text{AA}}(x, y) = \sum_{z \in \Gamma(x) \cap \Gamma(y)} \frac{1}{\ln(k_z)}$$

### 4. Preferential Attachment (PA)
Models rich-get-richer network growth: high-degree nodes are more likely to attract new edges. Depends only on node degrees, not shared neighbors.

$$s_{\text{PA}}(x, y) = k_x \cdot k_y$$

### 5. Random Walk with Restart (RWR)
Captures global multi-hop proximity via a personalized PageRank / random walk starting from source node $x$. At each step, the walker either restarts at $x$ with probability $\alpha$ (`restart_probability`), or transitions to a uniformly chosen neighbor with probability $1 - \alpha$:

$$\mathbf{r}_x = \alpha \mathbf{e}_x + (1 - \alpha) P^\top \mathbf{r}_x, \qquad P = D^{-1} A$$

where $A$ is the adjacency matrix, $D$ is the diagonal degree matrix, and $\mathbf{e}_x$ is the one-hot indicator vector for node $x$. Implemented via sparse power iteration (`rwr_vectors` in `algorithms.py`), stopping when `max_iterations` is reached or the max column $L_1$ change drops below `tolerance`:

$$\mathbf{X}^{(t+1)} = \alpha \mathbf{E} + (1 - \alpha) A D^{-1} \mathbf{X}^{(t)}$$

The default score between $x$ and $y$ is symmetrized:

$$s_{\text{RWR}}(x, y) = r_x[y] + r_y[x]$$

Using random-walk reversibility on undirected graphs ($r_y[x] = r_x[y] \cdot \frac{k_x}{k_y}$), the symmetrized score is computed from a single walk started at $x$ (configurable via `symmetric=True`).

**Configurable RWR parameters:**
- `restart_probability` ($\alpha \in (0, 1)$, default `0.15`)
- `max_iterations` (positive int, default `100`)
- `tolerance` (positive float, default `1e-8`)
- `symmetric` (bool, default `True`)

---

## Dataset & Preprocessing

### Main Experiment Dataset: `Facebook (ego-Facebook)`
- **Source:** SNAP "Social circles: Facebook" — J. McAuley and J. Leskovec, *"Learning to Discover Social Circles in Ego Networks"*, NIPS 2012.
  URL: https://snap.stanford.edu/data/ego-Facebook.html
- **Configuration:** Undirected, unweighted, simple graph (`directed: false`).
- **Size:** `4,039` nodes, `88,234` edges (`avg_degree = 43.69`, `max_degree = 1,045`, `density = 0.01082`, `avg_clustering = 0.6055`, `1` connected component).

### Preprocessing Steps (`data_loader.preprocess_graph`)
Every loaded or uploaded dataset undergoes the same clean preprocessing pipeline (recorded in `graph.graph["preprocessing"]`):
1. Convert directed graphs to undirected (`directed->undirected`, if applicable).
2. Collapse duplicate/reciprocal edges into a simple unweighted graph (`duplicates collapsed`).
3. Remove self-loops (`removed 0 self-loops` on ego-Facebook).
4. Optionally retain only the largest connected component (`kept largest connected component (dropped 0 nodes)` on ego-Facebook).

### Compatibility with Uploaded Datasets
The pipeline is dataset-agnostic:
- `data_loader.load_edge_list(source, directed=False, largest_component=False)` parses any TXT, CSV, or TSV edge list (comma, whitespace, or semicolon separated; `#`/`%` comments; automatic header detection; integer or string node IDs).
- `data_loader.graph_from_edge_table(frame, source_column, target_column, ...)` builds graphs directly from uploaded pandas DataFrames in the dashboard.
- `Karate Club (demo)` (Zachary 1977, 34 nodes, 78 edges) is included via NetworkX for lightweight UI demonstrations.

---

## Experimental Methodology (`evaluation.py`)

All five algorithms are evaluated on the exact same dataset splits and candidate pairs:

1. **Train-Test Split (`make_split`):**
   - `test_fraction = 0.10` (10% of edges hidden as positive test links: `8,823` edges hidden, `79,411` edges kept in `G_train`).
   - A random spanning forest is computed first (`_random_spanning_forest`) and excluded from removal so hiding test edges never disconnects the graph.
2. **Negative Sampling:**
   - `8,823` negative node pairs (equal to `#test_pos`) are sampled uniformly at random from the non-edges of the **original** graph $G$, guaranteeing no negative pair is a true or hidden edge.
3. **Candidate Shuffling:**
   - Positive and negative candidate pairs (`17,646` total) are combined and shuffled with the split seed before scoring so score ties are broken randomly and reproducibly.
4. **Repetitions / Random Seeds:**
   - Repeated over **10 seeds**: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51`.
5. **Metrics (`evaluate_scores`):**
   - **AUC-ROC:** Area under the ROC curve via `sklearn.metrics.roc_auc_score`.
   - **Precision@K:** Fraction of top-$K$ ranked candidate pairs that are hidden true edges. Default $K = \text{number of hidden edges}$ (`8,823`), with `precision_at_100` and `precision_at_1000` also recorded.
   - **Recall@K:** Fraction of all hidden edges retrieved in the top $K$ (equals Precision@K at $K = 8,823$ since positive and negative test sets are equal in size; `recall_at_100` and `recall_at_1000` are also in `results/results_per_seed.csv`).
   - **Execution Time:** Wall-clock seconds (`time.perf_counter()`) to build the sparse `GraphIndex` and score all `17,646` candidate pairs on `G_train`.

---

## Experimental Results (`Facebook (ego-Facebook)`, 10 Seeds)

Mean ± standard deviation from [results/results.csv](results/results.csv) (`n_nodes = 4039`, `n_edges = 88234`, `n_train_edges = 79411`, `n_test_pos = 8823`, `n_test_neg = 8823`, `K = 8823`; RWR with `restart_probability = 0.15`, `max_iterations = 100`):

| Algorithm | AUC-ROC | Precision@K | Recall@K | Precision@100 | Precision@1000 | Execution Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Adamic-Adar** | **0.9950 ± 0.0006** | **0.9760 ± 0.0011** | **0.9760 ± 0.0011** | **1.0000 ± 0.0000** | **0.9996 ± 0.0005** | 0.0757 ± 0.0114 |
| **Common Neighbors** | 0.9941 ± 0.0006 | 0.9733 ± 0.0012 | 0.9733 ± 0.0012 | 1.0000 ± 0.0000 | 0.9995 ± 0.0005 | 0.0761 ± 0.0050 |
| **Random Walk with Restart** | 0.9935 ± 0.0007 | 0.9727 ± 0.0014 | 0.9727 ± 0.0014 | 0.9990 ± 0.0032 | 0.9964 ± 0.0018 | 28.7309 ± 1.0090 |
| **Jaccard Similarity** | 0.9919 ± 0.0005 | 0.9575 ± 0.0015 | 0.9575 ± 0.0015 | 0.9890 ± 0.0110 | 0.9983 ± 0.0012 | 0.0776 ± 0.0058 |
| **Preferential Attachment** | 0.8420 ± 0.0021 | 0.7601 ± 0.0024 | 0.7601 ± 0.0024 | 0.9770 ± 0.0149 | 0.9839 ± 0.0032 | **0.0541 ± 0.0068** |

### Performance Comparison
- **Local neighborhood methods (AA, CN, JC) and RWR** achieve $> 0.99$ AUC-ROC and $> 0.95$ Precision@K on Facebook because social networks exhibit strong triadic closure (average clustering coefficient $0.606$).
- **Adamic-Adar** achieves the highest accuracy across every metric (AUC `0.9950`, P@K `0.9760`), slightly outperforming Common Neighbors (`0.9941`) by down-weighting high-degree hub intermediaries.
- **RWR** achieves nearly identical accuracy to CN (`0.9935` AUC) while capturing multi-hop structure, at higher computational cost (`28.73s` vs. `~0.08s`) because power iterations run across source nodes.
- **Preferential Attachment** is the fastest (`0.0541s`) but least accurate (`0.8420` AUC, `0.7601` P@K) since it relies solely on node degrees and ignores shared neighborhood structure.

---

## Validation & Reproducing Experiments

1. **Run the validation suite:**
   ```bash
   python validate_algorithms.py
   ```
2. **Run experiments on Facebook (ego-Facebook)** (place unzipped `facebook_combined.txt` in `data/`):
   ```bash
   python run_experiments.py --out-dir results
   ```
3. **Run experiments on any custom uploaded edge list:**
   ```bash
   python run_experiments.py --datasets --edge-list path/to/my_edges.csv --out-dir results
   ```

---

## Dashboard / UI Entry Points

- **Prediction (`algorithms.py`):** `run_link_prediction`, `predict_links`, `score_pairs`, `score_candidates`
- **Evaluation (`evaluation.py`):** `run_experiment(graph, name, seeds=[42])`, `summarize_runs(runs)`
- **Data loading (`data_loader.py`):** `load_edge_list`, `graph_from_edge_table`, `preprocess_graph`, `dataset_summary`
