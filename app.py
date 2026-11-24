"""GraphLink: shared network workspace and link prediction interface."""

from __future__ import annotations

import html
import io
import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from algorithms import SUPPORTED_ALGORITHMS, score_candidates
from data_loader import graph_from_edge_table, load_dataset

st.set_page_config(page_title="GraphLink | Network Intelligence", page_icon="G", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --page:#F7F8F6; --card:#FFFFFF; --green:#315D46; --sage:#789985;
  --pale:#E8F0EA; --ink:#25352C; --muted:#65736B; --line:#DDE5DF;
  font-size:16px;
}
html, body { font-size:16px !important; }
.stApp {
  background:var(--page); color:var(--ink);
  font-family:Inter,"Segoe UI",Arial,sans-serif;
  font-size:16px !important; line-height:1.55;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] { font-size:16px !important; }
[data-testid="stHeader"] { background:rgba(247,248,246,.97); height:2.8rem; }
.block-container {
  max-width:none !important; width:100% !important;
  padding:1.5rem 2rem 2.5rem !important;
}
[data-testid="stSidebar"] {
  background:#FFFFFF; border-right:1px solid var(--line);
  width:300px !important; min-width:285px !important;
}
[data-testid="stSidebar"] > div:first-child { padding:1.5rem 1.25rem 2rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.8rem; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  font-size:16px !important; line-height:1.55 !important;
}
.sidebar-brand {
  font-size:30px !important; font-weight:800; color:var(--green);
  letter-spacing:-.04em; line-height:1.2;
}
.sidebar-kicker {
  font-size:13px !important; letter-spacing:.12em;
  color:var(--muted) !important; margin-top:5px;
}
h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.025em; line-height:1.25; }
h1 { font-size:34px !important; margin:.35rem 0 .7rem !important; }
h2 { font-size:28px !important; margin:.5rem 0 .65rem !important; }
h3 { font-size:22px !important; margin:.45rem 0 .6rem !important; }
.stMarkdown p, .stMarkdown li, .stWrite, .stText,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size:16px !important; line-height:1.6 !important;
  color:var(--muted);
}
[data-testid="stCaptionContainer"] {
  font-size:14px !important; line-height:1.5 !important; color:var(--muted);
}
.site-title {
  text-align:center; color:var(--green); font-size:34px !important;
  font-weight:800; letter-spacing:-.04em; line-height:1.2;
  margin:.25rem 0 .3rem;
}
.site-subtitle {
  text-align:center; color:var(--muted); font-size:13px !important;
  letter-spacing:.14em; font-weight:700; margin:0 0 1rem;
}
.page-heading {
  text-align:center; color:var(--ink); font-size:32px !important;
  font-weight:750; letter-spacing:-.035em; line-height:1.25;
  margin:.45rem 0 .35rem;
}
.page-description {
  text-align:center; color:var(--muted); font-size:16px !important;
  line-height:1.6; margin:0 auto 1.35rem; max-width:1100px;
}
.stat-card {
  background:#fff; border:1px solid var(--line); border-radius:16px;
  box-shadow:0 3px 12px rgba(37,53,44,.06); padding:22px 20px;
  text-align:center; height:100%; min-height:122px;
  display:flex; flex-direction:column; justify-content:center;
}
.stat-value { font-size:40px !important; font-weight:800; line-height:1.12; color:var(--green); }
.stat-label { font-size:17px !important; color:var(--muted); font-weight:650; margin-top:9px; }
.status-strip {
  background:#fff; border:1px solid var(--line); border-radius:15px;
  padding:19px 22px; margin:1rem 0 1.2rem;
}
.status-grid { display:grid; grid-template-columns:1.3fr 1fr 1.2fr 1fr 1fr; gap:20px; margin-top:12px; }
.status-label {
  font-size:13px !important; text-transform:uppercase; letter-spacing:.08em;
  color:var(--sage); font-weight:750; line-height:1.4;
}
.status-value {
  font-size:16px !important; color:var(--ink); font-weight:650;
  margin-top:6px; line-height:1.5; overflow-wrap:anywhere;
}
.card { background:#fff; border:1px solid var(--line); border-radius:15px; padding:24px 26px; }
.result-row {
  background:#fff; border:1px solid var(--line); border-radius:13px;
  padding:18px 20px; margin:11px 0; display:flex;
  align-items:center; justify-content:space-between; min-height:82px;
  font-size:16px !important;
}
.pill { display:inline-block; border-radius:22px; padding:7px 12px; background:var(--pale); color:var(--green); font-size:13px !important; font-weight:750; }
div[data-testid="stMetric"] {
  background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:18px 20px; min-height:112px;
}
div[data-testid="stMetricLabel"] { font-size:16px !important; }
div[data-testid="stMetricValue"] { color:var(--green); font-size:30px !important; }
div.stButton > button {
  border-color:var(--line); border-radius:10px; color:var(--green);
  min-height:46px; font-size:16px !important; font-weight:650;
  padding:10px 18px;
}
div.stButton > button[kind="primary"] {
  background:var(--green); color:#fff; border:1px solid var(--green);
  font-weight:700; min-height:48px; font-size:16px !important;
}
[data-testid="stRadio"] label {
  font-size:16px !important; line-height:1.5 !important;
  padding:9px 10px; margin:3px 0; border-radius:9px;
}
[data-testid="stRadio"] label:has(input:checked) { background:var(--pale); color:var(--green) !important; font-weight:700; }
.stSelectbox label, .stSlider label, .stCheckbox label,
.stTextInput label, .stFileUploader label {
  font-size:15px !important; font-weight:650; line-height:1.45 !important;
}
[data-baseweb="select"], [data-baseweb="select"] *,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  font-size:16px !important;
}
[role="listbox"] [role="option"] { font-size:16px !important; min-height:42px; }
[data-testid="stFileUploader"] { font-size:15px !important; }
[data-testid="stExpander"] summary { font-size:16px !important; min-height:48px; }
[data-testid="stDataFrame"] { font-size:15px !important; }
[data-testid="stIFrame"] iframe {
  border:1px solid var(--line); border-radius:15px;
  background:#fff; width:100%;
}
[data-testid="stPlotlyChart"] {
  background:#fff; border:1px solid var(--line);
  border-radius:14px; padding:10px;
}
hr { margin:1.5rem 0 !important; border-color:var(--line) !important; }
@media (min-width:1600px) {
  .block-container { padding-left:2.5rem !important; padding-right:2.5rem !important; }
}
@media (max-width:1100px) {
  [data-testid="stSidebar"] { width:270px !important; min-width:255px !important; }
  .block-container { padding-left:1.35rem !important; padding-right:1.35rem !important; }
  .status-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
}
@media (max-width:700px) {
  .block-container { padding:.9rem .9rem 1.5rem !important; }
  .site-title { font-size:28px !important; }
  .site-subtitle { font-size:11px !important; }
  .page-heading { font-size:26px !important; }
  h1 { font-size:28px !important; } h2 { font-size:24px !important; } h3 { font-size:20px !important; }
  .stat-card { min-height:100px; padding:15px 8px; }
  .stat-value { font-size:30px !important; }
  .stat-label { font-size:14px !important; }
  .status-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
  .status-strip { padding:15px; }
  .stMarkdown p, .stMarkdown li { font-size:15px !important; }
}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("uploaded_graph", None)
st.session_state.setdefault("uploaded_name", None)
st.session_state.setdefault("predictions", None)
st.session_state.setdefault("selected_node", None)
PAGES = ["Overview", "Explore Network", "Link Prediction", "Algorithm Comparison", "Research & Methodology"]


def read_upload(raw: bytes, filename: str, delimiter_choice: str, custom_delimiter: str, has_header: bool) -> pd.DataFrame:
    """Parse a supported edge table using the user's delimiter choice."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".csv", ".tsv"}:
        raise ValueError("Choose a TXT, CSV, or TSV file.")
    if delimiter_choice == "Auto-detect":
        sep: str | None = "\\s+" if suffix == ".txt" else ("\\t" if suffix == ".tsv" else ",")
    elif delimiter_choice == "Whitespace":
        sep = "\\s+"
    elif delimiter_choice == "Custom":
        sep = custom_delimiter or None
        if not sep:
            raise ValueError("Enter a custom delimiter first.")
    else:
        sep = {"Comma (, )": ",", "Tab": "\\t", "Semicolon (;)": ";"}[delimiter_choice]
    return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", comment="#", header=0 if has_header else None)


def render_graph(graph: nx.Graph, selected: object | None, candidates: list[object]) -> None:
    """Render the same wide network visualization above every page section."""
    net = Network(height="620px", width="100%", bgcolor="#FFFFFF", font_color="#25352C", directed=graph.is_directed(), cdn_resources="in_line")
    net.barnes_hut(gravity=-3000, central_gravity=.15, spring_length=145, spring_strength=.04, damping=.16)
    for node in graph.nodes:
        if node == selected:
            fill, border = "#315D46", "#244633"
        elif node in candidates:
            fill, border = "#D6A868", "#B88548"
        else:
            fill, border = "#9CB7A3", "#789985"
        net.add_node(node, label=str(node), title=f"Node {html.escape(str(node))}<br>Degree: {graph.degree(node)}", color={"background": fill, "border": border, "highlight": {"background": "#E8F0EA", "border": "#315D46"}}, size=25 + min(graph.degree(node), 18), font={"size": 20, "color": "#25352C"})
    for left, right in graph.edges:
        net.add_edge(left, right, color={"color": "#D5DDD7", "highlight": "#789985"}, width=1.6)
    net.set_options(json.dumps({"interaction": {"hover": True, "navigationButtons": True, "keyboard": {"enabled": True}, "zoomView": True}, "nodes": {"borderWidth": 1.8}, "edges": {"smooth": {"type": "continuous"}}, "physics": {"stabilization": {"iterations": 180}}}))
    components.html(net.generate_html(), height=635, scrolling=False)


def describe_candidate(graph: nx.Graph, source: object, target: object, algorithm: str) -> str:
    shared = set(graph.neighbors(source)) & set(graph.neighbors(target))
    if algorithm == "Common Neighbors":
        return f"{len(shared)} shared neighbor(s): {', '.join(map(str, sorted(shared, key=str)[:5])) or 'none'}"
    if algorithm == "Jaccard Similarity":
        union = set(graph.neighbors(source)) | set(graph.neighbors(target))
        return f"Neighborhood overlap: {len(shared)} of {len(union)} nodes"
    if algorithm == "Adamic-Adar":
        return f"Shared neighbors weighted by inverse log degree ({len(shared)} shared)"
    return f"Degree product: {graph.degree(source)} x {graph.degree(target)}"


with st.sidebar:
    st.markdown('<div class="sidebar-brand">GraphLink</div><p class="sidebar-kicker">NETWORK INTELLIGENCE</p>', unsafe_allow_html=True)
    st.markdown("#### Dataset")
    uploaded_exists = st.session_state.uploaded_graph is not None
    dataset_choices = ["Karate Club (demo)", "Email-Eu-core"] + ([st.session_state.uploaded_name or "Uploaded dataset"] if uploaded_exists else [])
    if st.session_state.get("dataset_choice") not in dataset_choices:
        st.session_state.dataset_choice = "Karate Club (demo)"
    dataset_name = st.selectbox("Current network", dataset_choices, key="dataset_choice")
    st.divider()
    st.markdown("#### Navigation")
    page = st.radio("Select section", PAGES, label_visibility="collapsed", key="main_navigation")
    st.divider()
    with st.expander("Upload a network", expanded=False):
        upload = st.file_uploader("Choose TXT, CSV, or TSV", type=["txt", "csv", "tsv"], key="dataset_upload")
        delimiter_choice = st.selectbox("Delimiter", ["Auto-detect", "Comma (, )", "Tab", "Semicolon (;)", "Whitespace", "Custom"], key="upload_delimiter")
        custom_delimiter = st.text_input("Custom delimiter", key="custom_delimiter") if delimiter_choice == "Custom" else ""
        if upload is not None:
            try:
                header_default = Path(upload.name).suffix.lower() != ".txt"
                has_header = st.checkbox("First row contains column names", value=header_default, key="upload_has_header")
                frame = read_upload(upload.getvalue(), upload.name, delimiter_choice, custom_delimiter, has_header)
                if len(frame.columns) < 2:
                    st.error("The file must contain at least two columns.")
                else:
                    src_col = st.selectbox("Source column", frame.columns, key="upload_source_column")
                    dst_options = [column for column in frame.columns if column != src_col]
                    dst_col = st.selectbox("Target column", dst_options, key="upload_target_column")
                    directed_upload = st.checkbox("Directed network", key="upload_directed")
                    deduplicate = st.checkbox("Remove duplicate edges", value=True, key="upload_deduplicate")
                    drop_loops = st.checkbox("Remove self-loops", value=True, key="upload_selfloops")
                    st.caption(f"Preview: {len(frame):,} rows · {len(frame.columns)} columns")
                    st.dataframe(frame.head(6), use_container_width=True, hide_index=True, height=170)
                    if st.button("Validate and load network", type="primary", use_container_width=True):
                        try:
                            new_graph = graph_from_edge_table(frame, src_col, dst_col, directed=directed_upload, remove_duplicates=deduplicate, remove_self_loops=drop_loops)
                            st.session_state.uploaded_graph = new_graph
                            st.session_state.uploaded_name = f"Uploaded: {Path(upload.name).name}"
                            st.session_state.dataset_choice = st.session_state.uploaded_name
                            st.session_state.predictions = None
                            st.session_state.selected_node = None
                            st.success("Dataset validated and loaded.")
                            st.rerun()
                        except (ValueError, TypeError) as exc:
                            st.error(str(exc))
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
                st.error(f"Could not parse dataset: {exc}")

try:
    with st.spinner("Loading network..."):
        if dataset_name == (st.session_state.uploaded_name or "Uploaded dataset") and uploaded_exists:
            graph, is_demo, dataset_type = st.session_state.uploaded_graph, False, "User uploaded"
        else:
            graph, is_demo = load_dataset(dataset_name)
            dataset_type = "Demonstration" if is_demo else "Local research dataset"
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(str(exc))
    st.info("Place Email-Eu-core at `data/email-Eu-core.txt`, or choose Karate Club / upload a network.")
    st.stop()

nodes = list(graph.nodes())
if not nodes:
    st.error("The selected dataset contains no nodes.")
    st.stop()
if st.session_state.selected_node not in graph:
    st.session_state.selected_node = nodes[0]
with st.sidebar:
    focus_node = st.selectbox("Highlighted node", nodes, index=nodes.index(st.session_state.selected_node), format_func=str, key="highlight_node", help="Choose a node to highlight it in the shared graph.")
st.session_state.selected_node = focus_node

prediction = st.session_state.predictions
candidates: list[object] = []
if prediction and not prediction.get("error") and prediction.get("dataset") == dataset_name:
    candidates = [candidate for candidate, _ in prediction["ranked"]]
selected_for_graph = st.session_state.get("prediction_source", focus_node) if page == "Link Prediction" else st.session_state.get("explore_node", focus_node) if page == "Explore Network" else focus_node
if selected_for_graph not in graph:
    selected_for_graph = focus_node

st.markdown('<div class="site-title">◎ &nbsp;GraphLink</div><div class="site-subtitle">NETWORK INTELLIGENCE &amp; LINK PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="page-heading">Network Analysis</div><div class="page-description">Explore network structures, analyze connections and discover potential links using graph-based algorithms.</div>', unsafe_allow_html=True)
stat_cols = st.columns(3, gap="medium")
stats = [(f"{graph.number_of_nodes():,}", "Total Nodes"), (f"{graph.number_of_edges():,}", "Total Edges"), (f"{nx.density(graph):.4f}", "Network Density")]
for col, (value, label) in zip(stat_cols, stats):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown(f'''<div class="status-strip"><div class="status-label">Workspace status</div><div class="status-grid"><div><div class="status-label">Dataset</div><div class="status-value">{html.escape(dataset_name)}</div></div><div><div class="status-label">Type</div><div class="status-value">{dataset_type}</div></div><div><div class="status-label">Size</div><div class="status-value">{graph.number_of_nodes():,} nodes · {graph.number_of_edges():,} edges</div></div><div><div class="status-label">Direction</div><div class="status-value">{'Directed' if graph.is_directed() else 'Undirected'}</div></div><div><div class="status-label">Loading</div><div class="status-value">Ready</div></div></div></div>''', unsafe_allow_html=True)
st.markdown("### Network visualization")
st.caption("Drag nodes to arrange · scroll to zoom · pan with graph controls · hover for node ID and degree")
render_graph(graph, selected_for_graph, candidates)
st.markdown('<div><span style="color:#9CB7A3">●</span> Network node &nbsp;&nbsp; <span style="color:#315D46">●</span> Selected node &nbsp;&nbsp; <span style="color:#D6A868">●</span> Predicted candidate</div>', unsafe_allow_html=True)
st.divider()

if page == "Overview":
    st.markdown("## Overview")
    st.markdown("Explore the current dataset, available scoring methods, and the research goal.")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("### Dataset information")
        st.markdown(f"<div class='card'><span class='pill'>{html.escape(dataset_type.upper())}</span><h3>{html.escape(dataset_name)}</h3><p>{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, {'directed' if graph.is_directed() else 'undirected'}.</p><p>Karate Club is included for UI testing; Email-Eu-core is the planned primary dataset.</p></div>", unsafe_allow_html=True)
    with right:
        st.markdown("### Available algorithms")
        st.write("Common Neighbors · Jaccard Similarity · Adamic-Adar · Preferential Attachment")
        st.warning("Random Walk with Restart is selectable, but its backend implementation is pending.")
    st.markdown("### Research objective")
    st.write("Compare traditional similarity-based methods with Random Walk with Restart on real-world social networks using AUC, Precision@K, and measured execution time. No experimental results are reported until a held-out evaluation has been run.")
    a, b = st.columns(2)
    with a:
        if st.button("Explore this network", type="primary", use_container_width=True):
            st.session_state.main_navigation = "Explore Network"
            st.rerun()
    with b:
        if st.button("Open link prediction", use_container_width=True):
            st.session_state.main_navigation = "Link Prediction"
            st.rerun()

elif page == "Explore Network":
    st.markdown("## Explore network")
    st.markdown("Select a node to inspect its local neighborhood. The selection is highlighted in the shared graph above.")
    selected_node = st.selectbox("Selected node", nodes, index=nodes.index(selected_for_graph), format_func=str, key="explore_node")
    st.session_state.selected_node = selected_node
    neighbors = list(graph.neighbors(selected_node))
    detail_cols = st.columns(3)
    detail_cols[0].metric("Node ID", str(selected_node))
    detail_cols[1].metric("Degree", graph.degree(selected_node))
    detail_cols[2].metric("Neighbor count", len(neighbors))
    st.markdown("### Neighbors")
    if neighbors:
        st.dataframe(pd.DataFrame({"Neighbor node": [str(node) for node in neighbors]}), use_container_width=True, hide_index=True, height=min(340, 48 + 36 * len(neighbors)))
    else:
        st.info("This node has no neighbors in the selected graph.")
    if st.button("Clear predicted candidate highlights"):
        st.session_state.predictions = None
        st.rerun()

elif page == "Link Prediction":
    st.markdown("## Link prediction")
    st.markdown("Rank missing connections with a selected method. Demo graph output is for interface exploration, not research evidence.")
    setup, ranked_col = st.columns([.82, 1.8], gap="large")
    with setup:
        st.markdown("### Prediction setup")
        source_node = st.selectbox("Source node", nodes, index=nodes.index(selected_for_graph), format_func=str, key="prediction_source")
        algorithm = st.selectbox("Algorithm", [*SUPPORTED_ALGORITHMS, "Random Walk with Restart"], key="prediction_algorithm")
        top_k = st.slider("Top-K recommendations", 1, 20, 5, key="prediction_top_k")
        if algorithm == "Random Walk with Restart":
            st.slider("Restart probability", .05, .95, .15, .05, key="rwr_restart")
            st.info("RWR is available in the interface; the research implementation is not connected yet, so no prediction will be fabricated.")
        if st.button("Run prediction", type="primary", use_container_width=True):
            st.session_state.selected_node = source_node
            if algorithm == "Random Walk with Restart":
                st.session_state.predictions = {"error": "Random Walk with Restart is not implemented yet. Its restart probability control is ready for backend integration."}
            else:
                try:
                    with st.spinner(f"Scoring candidates with {algorithm}..."):
                        started = time.perf_counter()
                        ranked = score_candidates(graph, source_node, algorithm)
                        elapsed = time.perf_counter() - started
                    st.session_state.predictions = {"ranked": ranked[:top_k], "algorithm": algorithm, "source": source_node, "elapsed": elapsed, "demo": is_demo, "dataset": dataset_name}
                except (ValueError, nx.NetworkXException) as exc:
                    st.session_state.predictions = {"error": str(exc)}
                st.rerun()
    with ranked_col:
        st.markdown("### Ranked candidate links")
        state = st.session_state.predictions
        if not state:
            st.markdown("<div class='card'><h3>Ready to predict</h3><p>Choose a source and scoring method to list missing-link candidates and highlight them in the graph above.</p></div>", unsafe_allow_html=True)
        elif state.get("error"):
            st.warning(state["error"])
        elif state.get("dataset") != dataset_name:
            st.info("The dataset changed. Run a new prediction to score candidates on this graph.")
        else:
            st.markdown(f"<div class='card'><span class='pill'>{'DEMONSTRATION SCORES' if state['demo'] else 'GRAPH SCORES'}</span><h3>{html.escape(state['algorithm'])}</h3><p>Missing links from node {html.escape(str(state['source']))}</p></div>", unsafe_allow_html=True)
            for rank, (candidate, score) in enumerate(state["ranked"], 1):
                explanation = describe_candidate(graph, state["source"], candidate, state["algorithm"])
                st.markdown(f"<div class='result-row'><div><b style='color:#315D46'>#{rank:02d}</b> &nbsp; Node <b>{html.escape(str(candidate))}</b><br><span style='font-size:.9rem;color:#718078'>{html.escape(explanation)}</span></div><div style='text-align:right'><span class='status-label'>Score</span><br><b>{score:.4f}</b></div></div>", unsafe_allow_html=True)
            st.caption(f"Scoring time: {state['elapsed']*1000:.2f} ms. This is not an evaluation benchmark.")

elif page == "Algorithm Comparison":
    st.markdown("## Algorithm comparison")
    st.markdown("Measured experimental metrics are read from `results/results.csv`. No values are generated by the interface.")
    results_path = Path(__file__).resolve().parent / "results" / "results.csv"
    if not results_path.is_file():
        st.markdown("<div class='card'><h3>No experimental results yet</h3><p>Run a documented held-out evaluation and save rows to <code>results/results.csv</code>. Charts and the table will fill from that file.</p><span class='pill'>WAITING FOR EVALUATION</span></div>", unsafe_allow_html=True)
    else:
        try:
            results = pd.read_csv(results_path)
            required = {"algorithm", "auc", "precision_at_k", "execution_time_seconds"}
            if not required.issubset(results.columns):
                st.error("Required columns: algorithm, auc, precision_at_k, execution_time_seconds.")
            elif results.empty:
                st.info("The results file is present but contains no experiment rows.")
            else:
                for column in ("auc", "precision_at_k", "execution_time_seconds"):
                    results[column] = pd.to_numeric(results[column], errors="coerce")
                results = results.dropna(subset=["algorithm", "auc", "precision_at_k", "execution_time_seconds"])
                if results.empty:
                    st.warning("No rows contain valid metric values.")
                else:
                    results["execution_time_ms"] = results["execution_time_seconds"] * 1000
                    chart_cols = st.columns(3)
                    charts = [("AUC", "auc", "#789985"), ("Precision@K", "precision_at_k", "#315D46"), ("Execution time (ms)", "execution_time_ms", "#C19A62")]
                    for col, (title, field, color) in zip(chart_cols, charts):
                        with col:
                            st.markdown(f"### {title}")
                            st.plotly_chart(px.bar(results, x="algorithm", y=field, color_discrete_sequence=[color]), use_container_width=True)
                    st.markdown("### Recorded experiments")
                    st.dataframe(results.drop(columns=["execution_time_ms"]), use_container_width=True, hide_index=True)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read results/results.csv: {exc}")
    st.markdown("### Method summaries")
    st.write("Common Neighbors counts shared neighbors. Jaccard Similarity normalizes overlap. Adamic-Adar downweights high-degree shared neighbors. Preferential Attachment uses endpoint degree products. Random Walk with Restart remains unimplemented.")

else:
    st.markdown("## Research & methodology")
    st.markdown("### Research question")
    st.markdown("> How do traditional similarity-based link prediction algorithms compare with Random Walk with Restart in predicting missing links in real-world social networks?")
    st.markdown("### Research problem and objectives")
    st.write("Link prediction estimates which currently unconnected node pairs may form links. This work compares local similarity methods with Random Walk with Restart under a reproducible evaluation protocol, measuring prediction quality and runtime.")
    st.markdown("### Dataset")
    st.write("Email-Eu-core is the planned primary dataset. Karate Club is included for UI testing. Uploaded edge lists are user supplied and can use arbitrary string or numeric node identifiers.")
    st.markdown("### Methods")
    left, right = st.columns(2)
    with left:
        st.write("**Common Neighbors** counts shared neighbors. **Jaccard Similarity** normalizes neighborhood overlap. **Adamic-Adar** gives more weight to uncommon shared neighbors. **Preferential Attachment** favors pairs with higher degree endpoints.")
    with right:
        st.write("**Random Walk with Restart** estimates proximity by traversing the graph and periodically returning to the source node. The interface includes a restart-probability control; the backend algorithm is still pending.")
    st.markdown("### Evaluation and workflow")
    st.write("Hold out a defined set of positive edges, remove them from the training graph, sample non-edges as negatives, score the same candidate pairs with each method, then calculate AUC, Precision@K, and runtime. Record the split strategy, seed, dataset, and parameters with each run. Results should only be compared when the evaluation setup is documented and compatible.")

st.divider()
st.caption("GraphLink · Social Network Mining research workspace · Predictions on demonstration data are not experimental results.")
