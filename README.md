# GraphLink

GraphLink is a local Streamlit research workspace with a persistent navigation rail and five task-focused pages: Overview, Explore Network, Link Prediction, Algorithm Comparison, and Research & Methodology. A shared interactive PyVis network, dataset statistics, and workspace status appear above the selected page content. The responsive light interface uses the available page width and readable controls. It includes working Common Neighbors, Jaccard Similarity, Adamic-Adar, and Preferential Attachment recommendations. Random Walk with Restart is selectable, with its restart-probability control, and clearly marked as awaiting backend implementation.

Predictions on the demo graph are examples for UI exploration. They are not research findings or performance measurements. The comparison dashboard reads only measured results from `results/results.csv` and remains empty until actual evaluation results are provided.

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux

## Install and run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the app with the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints the local URL (usually `http://localhost:8501`). No API keys or external services are required. The font stylesheet has a system-font fallback when offline.

## Email-Eu-core dataset

The primary dataset is not redistributed or downloaded by this app. Put a whitespace-delimited edge list named `email-Eu-core.txt` in `data/`. Each row should contain two integer node IDs; lines beginning with `#` are ignored. Select **Email-Eu-core** from the app. Until the file is present, the app displays an explanatory message and the demo graph remains available.

## Upload a network

Expand **Upload a network** in the left sidebar and choose a TXT, CSV, or TSV edge table. Choose the delimiter, whether the first row contains column names, source and target columns, directedness, and duplicate/self-loop handling. Review the preview, then validate and load the graph. The loaded graph stays available in the Streamlit session and supports string or numeric node IDs.

## Project structure

```text
app.py           Streamlit interface and graph rendering
algorithms.py    Baseline scoring functions and RWR interface
data_loader.py  Demo and local dataset loading
evaluation.py   Metrics interface for held-out experiments
data/            Local datasets (not committed)
results/         Evaluation outputs (not committed)
figures/         Generated research figures (not committed)
paper/           Research writing and methodology (not committed)
```

The UI calls the scoring interface in `algorithms.py`; evaluation metrics are isolated in `evaluation.py`. The `evaluate_scores` function expects labels and scores for the same held-out candidate links and ranked labels ordered best-first.

## Experimental results format

Save one measured run per row in `results/results.csv` with these required columns:

```csv
algorithm,auc,precision_at_k,execution_time_seconds
```

Optional metadata columns such as `dataset`, `k`, `run_id`, and `split_seed` are retained in the comparison table. The app does not generate or fill metric values.
