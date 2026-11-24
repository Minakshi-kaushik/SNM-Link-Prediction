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
  --page:#F5F7FA; --card:#FFFFFF; --navy:#24364B; --blue:#416A8A;
  --blue-soft:#EAF1F6; --amber:#D6A34A; --ink:#1E293B;
  --muted:#5B6778; --line:#DCE3EA; font-size:16px;
}
html, body { font-size:16px !important; }
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:var(--page) !important; color:var(--ink) !important;
  font-family:Inter,"Segoe UI",Arial,sans-serif !important;
  font-size:18px !important; line-height:1.6 !important;
}
[data-testid="stHeader"] { background:rgba(245,247,250,.98) !important; height:3rem; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width:none !important; width:100% !important;
  padding:2rem 2.25rem 3rem !important; min-height:100vh;
}
/* Resize the actual sidebar section, not only its inner content. */
section[data-testid="stSidebar"], [data-testid="stSidebar"] {
  width:300px !important; min-width:300px !important; max-width:300px !important;
  flex:0 0 300px !important; background:#FFFFFF !important;
  border-right:1px solid var(--line) !important;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div:first-child {
  width:100% !important; padding:1.7rem 1.35rem 2rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:1rem !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  font-size:17px !important; line-height:1.6 !important; color:var(--ink) !important;
}
[data-testid="stSidebar"] h4 { font-size:18px !important; font-weight:750 !important; }
.sidebar-brand {
  display:block; overflow:visible; padding-top:8px; margin:0 0 4px;
  font-size:32px !important; font-weight:800; color:var(--navy);
  letter-spacing:-.04em; line-height:1.3;
}
.sidebar-kicker {
  font-size:14px !important; letter-spacing:.12em;
  color:var(--muted) !important; margin-top:4px;
}
h1,h2,h3 { color:var(--ink) !important; letter-spacing:-.025em; line-height:1.25 !important; }
h1 { font-size:36px !important; margin:.5rem 0 .8rem !important; }
h2 { font-size:30px !important; margin:.65rem 0 .8rem !important; }
h3 { font-size:23px !important; margin:.55rem 0 .7rem !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { text-align:center !important; }
.stMarkdown p, .stMarkdown li, .stWrite, .stText,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stText"] {
  font-size:18px !important; line-height:1.65 !important;
  color:var(--muted) !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  font-size:16px !important; line-height:1.55 !important; color:var(--muted) !important;
}
.site-title {
  display:block; overflow:visible; padding-top:12px; margin:.3rem 0 .45rem;
  text-align:center; color:var(--navy); font-size:42px !important;
  font-weight:850; letter-spacing:-.045em; line-height:1.25;
}
.site-subtitle {
  text-align:center; color:var(--muted); font-size:15px !important;
  letter-spacing:.15em; font-weight:750; margin:0 0 1.4rem;
}
.page-heading {
  text-align:center; color:var(--ink); font-size:38px !important;
  font-weight:800; letter-spacing:-.035em; line-height:1.25;
  margin:.55rem 0 .45rem;
}
.page-description {
  text-align:center; color:var(--muted); font-size:18px !important;
  line-height:1.65; margin:0 auto 1.7rem; max-width:1200px;
}
.stat-card {
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  box-shadow:0 4px 16px rgba(36,54,75,.07); padding:26px 22px;
  text-align:center; height:100%; min-height:148px;
  display:flex; flex-direction:column; justify-content:center;
}
.stat-value { font-size:46px !important; font-weight:850; line-height:1.12; color:var(--navy); }
.stat-label { font-size:19px !important; color:var(--muted); font-weight:700; margin-top:11px; }
.status-strip {
  background:#fff; border:1px solid var(--line); border-radius:17px;
  padding:28px 30px; margin:1.25rem 0 1.6rem;
  box-shadow:0 3px 14px rgba(36,54,75,.045);
}
.status-strip > .status-label { font-size:18px !important; margin-bottom:12px; }
.status-grid { display:grid; grid-template-columns:1.35fr 1fr 1.2fr 1fr 1fr; gap:24px; margin-top:14px; }
.status-label {
  font-size:15px !important; text-transform:uppercase; letter-spacing:.09em;
  color:var(--blue); font-weight:800; line-height:1.45;
}
.status-value {
  font-size:19px !important; color:var(--ink); font-weight:700;
  margin-top:8px; line-height:1.55; overflow-wrap:anywhere;
}
.card { background:#fff; border:1px solid var(--line); border-radius:17px; padding:28px 30px; }
.card h3 { text-align:left !important; font-size:22px !important; }
.card p { font-size:17px !important; line-height:1.65 !important; }
.result-row {
  background:#fff; border:1px solid var(--line); border-radius:15px;
  padding:22px 24px; margin:13px 0; display:flex;
  align-items:center; justify-content:space-between; min-height:94px;
  font-size:18px !important;
}
.pill { display:inline-block; border-radius:24px; padding:8px 14px; background:var(--blue-soft); color:var(--navy); font-size:14px !important; font-weight:800; }
div[data-testid="stMetric"] {
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:22px 24px; min-height:130px;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { font-size:17px !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color:var(--navy) !important; font-size:34px !important; }
div.stButton > button {
  border:1px solid var(--line); border-radius:11px; color:var(--navy);
  min-height:52px; font-size:17px !important; font-weight:700;
  padding:12px 20px; background:#fff;
}
div.stButton > button[kind="primary"] {
  background:var(--navy) !important; color:#fff !important;
  border:1px solid var(--navy) !important;
  font-weight:750; min-height:54px; font-size:17px !important;
}
[data-testid="stRadio"] label {
  font-size:17px !important; line-height:1.55 !important;
  padding:11px 12px; margin:5px 0; border-radius:10px;
}
[data-testid="stRadio"] label:has(input:checked) {
  background:var(--blue-soft); color:var(--navy) !important; font-weight:750;
}
.stSelectbox label, .stSlider label, .stCheckbox label,
.stTextInput label, .stFileUploader label,
[data-testid="stWidgetLabel"] p {
  font-size:16px !important; font-weight:700 !important; line-height:1.5 !important;
}
[data-baseweb="select"], [data-baseweb="select"] *,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  font-size:17px !important;
}
[role="listbox"] [role="option"] { font-size:17px !important; min-height:46px; }
[data-testid="stFileUploader"] { font-size:16px !important; }
[data-testid="stExpander"] summary { font-size:17px !important; min-height:54px; }
[data-testid="stDataFrame"] { font-size:16px !important; }
[data-testid="stIFrame"] iframe {
  border:1px solid var(--line); border-radius:17px;
  background:#fff; width:100%;
}
[data-testid="stPlotlyChart"] {
  background:#fff; border:1px solid var(--line);
  border-radius:16px; padding:14px;
}
hr { margin:1.8rem 0 !important; border-color:var(--line) !important; }
@media (min-width:1600px) {
  .block-container { padding-left:2.75rem !important; padding-right:2.75rem !important; }
}
@media (max-width:1100px) {
  section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    width:275px !important; min-width:275px !important; max-width:275px !important; flex-basis:275px !important;
  }
  .block-container { padding-left:1.5rem !important; padding-right:1.5rem !important; }
  .status-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
}
@media (max-width:700px) {
  section[data-testid="stSidebar"], [data-testid="stSidebar"] { width:100% !important; min-width:0 !important; max-width:100% !important; }
  .block-container { padding:1rem 1rem 1.6rem !important; }
  .site-title { font-size:32px !important; }
  .site-subtitle { font-size:12px !important; }
  .page-heading { font-size:29px !important; }
  h1 { font-size:30px !important; } h2 { font-size:26px !important; } h3 { font-size:21px !important; }
  .stat-card { min-height:112px; padding:17px 10px; }
  .stat-value { font-size:34px !important; }
  .stat-label { font-size:15px !important; }
  .status-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:17px; }
  .status-strip { padding:19px; }
  .stMarkdown p, .stMarkdown li { font-size:16px !important; }
}

/* Final component pass: make Streamlit's nested widgets inherit readable sizing. */
.stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stText"],
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stExpander"] *,
.stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stRadio"] *,
.stApp [data-testid="stSelectbox"] *,
.stApp [data-testid="stSlider"] *,
.stApp [data-testid="stCheckbox"] *,
.stApp [data-testid="stTextInput"] * {
  line-height:1.55 !important;
}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stExpander"] p { font-size:17px !important; }
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] > div {
  gap:9px !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] label {
  display:flex !important; align-items:center !important;
  min-height:48px !important; padding:10px 13px !important;
  border:1px solid transparent; border-radius:11px !important;
  font-size:17px !important; font-weight:650 !important;
  color:var(--ink) !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background:var(--blue-soft) !important; border-color:#D2DFE9 !important;
  color:var(--navy) !important; font-weight:800 !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] {
  border:1px solid var(--line) !important; border-radius:13px !important;
  background:#FBFCFD !important; overflow:hidden;
}
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary {
  padding:12px 13px !important; min-height:52px !important;
  font-size:16px !important; font-weight:750 !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  padding:14px !important; border:1px dashed #AABAC8 !important;
  border-radius:11px !important; background:#F8FAFC !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  font-size:15px !important; min-height:40px !important;
}
.stApp [data-testid="stSidebar"] [data-baseweb="select"] {
  min-height:46px !important; border-radius:9px !important;
}
.stApp [data-testid="stSidebar"] hr { margin:1.1rem 0 !important; }
.stApp [data-testid="stAlert"] {
  border-radius:12px !important; padding:14px 17px !important;
}
.stApp [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stApp [data-testid="stDataFrame"] * { font-size:15px !important; }
.stApp [data-testid="stPlotlyChart"] > div { width:100% !important; }
.stApp [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
  align-items:stretch;
}
.site-title {
  font-size:46px !important; padding-top:18px !important;
  padding-bottom:3px !important; min-height:72px;
}
.site-subtitle { font-size:15px !important; margin-bottom:1.6rem !important; }
.page-heading { font-size:40px !important; }
.page-description { font-size:18px !important; margin-bottom:1.8rem !important; }
.status-strip { min-height:150px; }
.status-grid > div { min-width:0; }
.status-value { overflow-wrap:anywhere; }
.stat-card { min-height:156px; }
.result-row b { font-size:19px !important; }
.result-row span { font-size:16px !important; line-height:1.55 !important; }
@media (max-width:700px) {
  .site-title { font-size:34px !important; min-height:58px; }
  .page-heading { font-size:30px !important; }
  .page-description { font-size:16px !important; }
  .stApp [data-testid="stSidebar"] [data-testid="stRadio"] label { min-height:44px !important; }
}

/* High-visibility desktop typography and spacious component scale. */
html { font-size:20px !important; }
body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  font-size:20px !important; line-height:1.65 !important;
}
[data-testid="stMainBlockContainer"], .block-container {
  padding:2.25rem 2.6rem 3.5rem !important;
}
section[data-testid="stSidebar"], [data-testid="stSidebar"] {
  width:340px !important; min-width:340px !important; max-width:340px !important;
  flex-basis:340px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  font-size:20px !important; line-height:1.65 !important;
}
[data-testid="stSidebar"] h4 { font-size:22px !important; }
.sidebar-brand { font-size:38px !important; }
.sidebar-kicker { font-size:16px !important; }
.site-title { font-size:54px !important; min-height:84px; padding-top:20px !important; }
.site-subtitle { font-size:17px !important; }
.page-heading { font-size:48px !important; margin:1rem 0 .6rem !important; }
.page-description { font-size:21px !important; max-width:1400px; margin-bottom:2rem !important; }
h1 { font-size:44px !important; }
h2 { font-size:38px !important; }
h3 { font-size:29px !important; }
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stExpander"] p,
.stApp [data-testid="stCaptionContainer"] p {
  font-size:19px !important; line-height:1.7 !important;
}
.stApp [data-testid="stCaptionContainer"] p { font-size:17px !important; }
.stat-card {
  min-height:190px !important; padding:32px 26px !important;
  justify-content:center !important; align-items:center !important;
}
.stat-value { font-size:58px !important; }
.stat-label { font-size:22px !important; margin-top:14px !important; }
.status-strip { min-height:190px !important; padding:34px 36px !important; }
.status-strip > .status-label { font-size:22px !important; text-align:center !important; }
.status-grid { gap:30px !important; margin-top:20px !important; text-align:center !important; }
.status-label { font-size:17px !important; }
.status-value { font-size:22px !important; line-height:1.6 !important; }
.card { padding:34px 36px !important; text-align:center !important; min-height:220px; }
.card h3 { text-align:center !important; font-size:27px !important; }
.card p { font-size:19px !important; }
.pill { font-size:16px !important; padding:9px 16px !important; }
.result-row { min-height:115px !important; padding:26px 30px !important; }
.result-row b { font-size:22px !important; }
.result-row span { font-size:18px !important; }
div[data-testid="stMetric"] { min-height:160px !important; padding:28px !important; text-align:center !important; }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { font-size:20px !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { font-size:42px !important; }
div.stButton > button { min-height:62px !important; font-size:19px !important; }
div.stButton > button[kind="primary"] { min-height:64px !important; font-size:19px !important; }
.stSelectbox label, .stSlider label, .stCheckbox label,
.stTextInput label, .stFileUploader label,
[data-testid="stWidgetLabel"] p { font-size:19px !important; }
[data-baseweb="select"], [data-baseweb="select"] *,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { font-size:19px !important; }
[data-testid="stRadio"] label { font-size:19px !important; min-height:54px !important; }
[data-testid="stExpander"] summary { font-size:19px !important; min-height:62px !important; }
[data-testid="stDataFrame"] * { font-size:17px !important; }
.network-section-title {
  text-align:center; font-size:32px !important; font-weight:800;
  color:var(--navy); margin:1.5rem 0 .5rem;
}
.network-section-help {
  text-align:center; font-size:18px !important; color:var(--muted);
  margin:0 0 1rem;
}
@media (max-width:1100px) {
  section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    width:300px !important; min-width:300px !important; max-width:300px !important; flex-basis:300px !important;
  }
  .status-grid { grid-template-columns:repeat(3,minmax(0,1fr)); }
}
@media (max-width:700px) {
  html { font-size:16px !important; }
  section[data-testid="stSidebar"], [data-testid="stSidebar"] {
    width:100% !important; min-width:0 !important; max-width:100% !important;
  }
  .site-title { font-size:36px !important; }
  .page-heading { font-size:32px !important; }
  .page-description { font-size:17px !important; }
  h1 { font-size:32px !important; } h2 { font-size:28px !important; } h3 { font-size:23px !important; }
  .stat-card { min-height:130px !important; }
  .stat-value { font-size:40px !important; }
  .stat-label { font-size:17px !important; }
  .status-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px !important; }
  .status-value { font-size:17px !important; }
  .card { min-height:0; padding:22px !important; }
}

/* Accessibility override: prioritize legibility over dense dashboard styling. */
:root {
  font-size:22px !important;
  --page:#F5F7FA; --card:#FFFFFF; --navy:#24364B; --blue:#416A8A;
  --blue-soft:#EAF1F6; --amber:#D6A34A; --ink:#1E293B;
  --muted:#465568; --line:#DCE3EA;
}
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"] {
  font-size:22px !important;
}
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stExpander"] p {
  font-size:22px !important;
  line-height:1.65 !important;
  color:var(--muted) !important;
}
.stApp h1, .stApp [data-testid="stMarkdownContainer"] h1 { font-size:52px !important; line-height:1.2 !important; }
.stApp h2, .stApp [data-testid="stMarkdownContainer"] h2 { font-size:44px !important; line-height:1.25 !important; }
.stApp h3, .stApp [data-testid="stMarkdownContainer"] h3 { font-size:34px !important; line-height:1.3 !important; }
.stApp [data-testid="stCaptionContainer"] p { font-size:19px !important; }
.stApp [data-testid="stSidebar"],
.stApp section[data-testid="stSidebar"] {
  width:370px !important; min-width:370px !important;
  max-width:370px !important; flex:0 0 370px !important;
}
.stApp section[data-testid="stSidebar"] > div,
.stApp [data-testid="stSidebar"] > div:first-child,
.stApp [data-testid="stSidebarContent"] {
  width:100% !important; min-width:0 !important;
  padding:28px 22px 32px !important;
}
.stApp [data-testid="stSidebar"] *,
.stApp section[data-testid="stSidebar"] * {
  font-size:20px !important;
  line-height:1.6 !important;
}
.stApp [data-testid="stSidebar"] .sidebar-brand,
.stApp section[data-testid="stSidebar"] .sidebar-brand {
  font-size:42px !important; line-height:1.25 !important;
}
.stApp [data-testid="stSidebar"] .sidebar-kicker,
.stApp section[data-testid="stSidebar"] .sidebar-kicker {
  font-size:16px !important; letter-spacing:.12em !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] label {
  min-height:58px !important; padding:13px 14px !important;
  font-size:20px !important; margin:6px 0 !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stSidebar"] label { font-size:19px !important; }
.stApp [data-testid="stSidebar"] [data-baseweb="select"],
.stApp [data-testid="stSidebar"] [data-baseweb="select"] * {
  font-size:19px !important; min-height:50px !important;
}
.stApp .site-title { font-size:60px !important; line-height:1.2 !important; min-height:92px !important; }
.stApp .site-subtitle { font-size:19px !important; }
.stApp .page-heading { font-size:54px !important; line-height:1.2 !important; }
.stApp .page-description { font-size:22px !important; line-height:1.65 !important; max-width:1500px !important; }
.stApp .stat-card { min-height:210px !important; padding:36px 28px !important; }
.stApp .stat-value { font-size:68px !important; line-height:1.1 !important; }
.stApp .stat-label { font-size:24px !important; line-height:1.4 !important; }
.stApp .status-strip { min-height:210px !important; padding:38px 40px !important; }
.stApp .status-strip > .status-label { font-size:24px !important; }
.stApp .status-label { font-size:19px !important; }
.stApp .status-value { font-size:24px !important; line-height:1.5 !important; }
.stApp .card { min-height:250px !important; padding:38px 40px !important; }
.stApp .card h3 { font-size:32px !important; }
.stApp .card p { font-size:21px !important; line-height:1.7 !important; }
.stApp .pill { font-size:18px !important; padding:10px 17px !important; }
.stApp .result-row { min-height:130px !important; padding:28px 32px !important; font-size:22px !important; }
.stApp .result-row b { font-size:25px !important; }
.stApp .result-row span { font-size:20px !important; }
.stApp div[data-testid="stMetricLabel"],
.stApp div[data-testid="stMetricLabel"] * { font-size:22px !important; }
.stApp div[data-testid="stMetricValue"],
.stApp div[data-testid="stMetricValue"] * { font-size:48px !important; }
.stApp div.stButton > button {
  min-height:68px !important; font-size:21px !important; padding:16px 24px !important;
}
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stFileUploader"] label,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stExpander"] summary {
  font-size:20px !important; line-height:1.55 !important;
}
.stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea { font-size:20px !important; }
.stApp [data-testid="stDataFrame"] * { font-size:18px !important; }
.stApp .network-section-title { font-size:38px !important; }
.stApp .network-section-help { font-size:20px !important; }
.stApp [data-testid="stMainBlockContainer"], .stApp .block-container {
  padding:34px 42px 54px !important;
}
@media (max-width:1100px) {
  .stApp section[data-testid="stSidebar"],
  .stApp [data-testid="stSidebar"] {
    width:320px !important; min-width:320px !important;
    max-width:320px !important; flex-basis:320px !important;
  }
}
@media (max-width:700px) {
  :root { font-size:17px !important; }
  .stApp section[data-testid="stSidebar"],
  .stApp [data-testid="stSidebar"] {
    width:100% !important; min-width:0 !important; max-width:100% !important;
  }
  .stApp .site-title { font-size:40px !important; }
  .stApp .page-heading { font-size:36px !important; }
  .stApp .stat-value { font-size:44px !important; }
  .stApp .stat-label { font-size:18px !important; }
  .stApp p, .stApp li { font-size:17px !important; }
  .stApp h1 { font-size:36px !important; }
  .stApp h2 { font-size:31px !important; }
  .stApp h3 { font-size:25px !important; }
}


/* High-readability typography pass: prioritize legibility across every UI component. */
:root { font-size:26px !important; }
html, body, .stApp, .stApp *:not(svg):not(svg *) {
  text-rendering: optimizeLegibility;
}
.stApp {
  font-size:26px !important;
}
.stApp :is(p, li, label, span, small, strong, em, a, button,
           input, textarea, select, summary, td, th, figcaption) {
  font-size:26px !important;
  line-height:1.65 !important;
}
.stApp :is([data-testid="stMarkdownContainer"] p,
           [data-testid="stMarkdownContainer"] li,
           [data-testid="stCaptionContainer"] p,
           [data-testid="stAlert"] p,
           [data-testid="stExpander"] p,
           .stMarkdown p, .stMarkdown li) {
  font-size:26px !important;
  line-height:1.7 !important;
  color:#34445a !important;
}
.stApp :is(small, [data-testid="stCaptionContainer"] p, .caption) {
  font-size:22px !important;
  line-height:1.6 !important;
  color:#465568 !important;
}
/* Sidebar: larger navigation, labels, upload controls and more breathing room. */
.stApp section[data-testid="stSidebar"],
.stApp [data-testid="stSidebar"] {
  width:410px !important; min-width:410px !important;
  max-width:410px !important; flex:0 0 410px !important;
}
.stApp section[data-testid="stSidebar"] > div,
.stApp [data-testid="stSidebar"] > div:first-child,
.stApp [data-testid="stSidebarContent"] {
  width:100% !important; min-width:0 !important;
  padding:34px 26px 42px !important;
}
.stApp [data-testid="stSidebar"] :is(p, label, span, small, a, button) {
  font-size:25px !important; line-height:1.65 !important;
  color:#24364B !important;
}
.stApp [data-testid="stSidebar"] .sidebar-brand {
  font-size:48px !important; line-height:1.25 !important;
}
.stApp [data-testid="stSidebar"] .sidebar-kicker {
  font-size:21px !important; line-height:1.5 !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:15px !important; }
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] label {
  min-height:74px !important; padding:18px 18px !important;
  margin:8px 0 !important; border-radius:12px !important;
  font-size:25px !important; font-weight:700 !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stRadio"] label p {
  font-size:25px !important;
}
.stApp [data-testid="stSidebar"] [data-baseweb="select"],
.stApp [data-testid="stSidebar"] [data-baseweb="select"] * {
  min-height:60px !important; font-size:24px !important;
}
.stApp [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-size:23px !important; line-height:1.6 !important;
}
/* Main page hierarchy. */
.stApp .site-title { font-size:68px !important; line-height:1.2 !important; }
.stApp .site-subtitle { font-size:23px !important; }
.stApp .page-heading { font-size:58px !important; line-height:1.2 !important; }
.stApp .page-description { font-size:27px !important; line-height:1.65 !important; }
.stApp h1, .stApp [data-testid="stMarkdownContainer"] h1 {
  font-size:54px !important; line-height:1.25 !important;
}
.stApp h2, .stApp [data-testid="stMarkdownContainer"] h2 {
  font-size:46px !important; line-height:1.3 !important;
}
.stApp h3, .stApp [data-testid="stMarkdownContainer"] h3 {
  font-size:38px !important; line-height:1.35 !important;
}
.stApp .stat-card { min-height:240px !important; padding:42px 32px !important; }
.stApp .stat-value { font-size:76px !important; }
.stApp .stat-label { font-size:28px !important; line-height:1.5 !important; }
.stApp .status-strip { min-height:230px !important; padding:42px 44px !important; }
.stApp .status-strip > .status-label { font-size:28px !important; }
.stApp .status-label { font-size:24px !important; }
.stApp .status-value { font-size:28px !important; line-height:1.6 !important; }
.stApp .card { padding:42px 44px !important; }
.stApp .card h3 { font-size:36px !important; }
.stApp .card p { font-size:26px !important; line-height:1.7 !important; }
.stApp .pill { font-size:22px !important; }
.stApp .result-row { min-height:150px !important; font-size:27px !important; }
.stApp .result-row b { font-size:30px !important; }
.stApp .result-row span { font-size:25px !important; }
.stApp div[data-testid="stMetricLabel"],
.stApp div[data-testid="stMetricLabel"] * { font-size:26px !important; }
.stApp div[data-testid="stMetricValue"],
.stApp div[data-testid="stMetricValue"] * { font-size:54px !important; }
.stApp div.stButton > button,
.stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stFormSubmitButton"] button {
  min-height:78px !important; padding:19px 28px !important;
  font-size:26px !important; line-height:1.4 !important;
  font-weight:750 !important;
}
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stTextInput"] label,
.stApp [data-testid="stFileUploader"] label,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stWidgetLabel"] p {
  font-size:25px !important; line-height:1.6 !important;
}
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea,
.stApp [role="option"] {
  font-size:25px !important; line-height:1.55 !important;
}
.stApp [data-testid="stDataFrame"] * { font-size:23px !important; }
.stApp .network-section-title { font-size:42px !important; }
.stApp .network-section-help { font-size:25px !important; }
.stApp [data-testid="stMainBlockContainer"], .stApp .block-container {
  padding:42px 48px 62px !important;
}
@media (max-width:1100px) {
  .stApp section[data-testid="stSidebar"],
  .stApp [data-testid="stSidebar"] {
    width:340px !important; min-width:340px !important;
    max-width:340px !important; flex-basis:340px !important;
  }
}
@media (max-width:700px) {
  :root { font-size:18px !important; }
  .stApp section[data-testid="stSidebar"],
  .stApp [data-testid="stSidebar"] {
    width:100% !important; min-width:0 !important; max-width:100% !important;
  }
  .stApp :is(p, li, label, span, small, a, button) {
    font-size:18px !important; line-height:1.6 !important;
  }
  .stApp .site-title { font-size:42px !important; }
  .stApp .page-heading { font-size:38px !important; }
  .stApp .stat-value { font-size:46px !important; }
  .stApp .stat-label { font-size:19px !important; }
  .stApp h1 { font-size:38px !important; }
  .stApp h2 { font-size:32px !important; }
  .stApp h3 { font-size:26px !important; }
  .stApp div.stButton > button { min-height:58px !important; font-size:19px !important; }
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
    net = Network(height="900px", width="100%", bgcolor="#FFFFFF", font_color="#1E293B", directed=graph.is_directed(), cdn_resources="in_line")
    net.barnes_hut(gravity=-3000, central_gravity=.15, spring_length=175, spring_strength=.04, damping=.16)
    for node in graph.nodes:
        if node == selected:
            fill, border = "#24364B", "#18283A"
        elif node in candidates:
            fill, border = "#D6A34A", "#B88732"
        else:
            fill, border = "#6F91AE", "#416A8A"
        net.add_node(node, label=str(node), title=f"Node {html.escape(str(node))}<br>Degree: {graph.degree(node)}", color={"background": fill, "border": border, "highlight": {"background": "#EAF1F6", "border": "#24364B"}}, size=11 + min(graph.degree(node), 9), font={"size": 17, "color": "#1E293B"})
    for left, right in graph.edges:
        net.add_edge(left, right, color={"color": "#D5DDD7", "highlight": "#416A8A"}, width=1.6)
    net.set_options(json.dumps({
        "layout": {"improvedLayout": True, "randomSeed": 42},
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": {"enabled": True}, "zoomView": True, "dragView": True, "dragNodes": True},
        "nodes": {"borderWidth": 1.8, "scaling": {"min": 10, "max": 22}, "font": {"size": 17, "face": "Arial"}},
        "edges": {"smooth": {"type": "continuous"}, "color": {"opacity": 0.48}},
        "physics": {"enabled": True, "solver": "barnesHut", "stabilization": {"enabled": True, "iterations": 1000, "updateInterval": 50, "fit": True}, "barnesHut": {"gravitationalConstant": -4200, "centralGravity": 0.18, "springLength": 190, "springConstant": 0.035, "damping": 0.18}}
    }))
    components.html(net.generate_html(), height=915, scrolling=False)


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
st.markdown('<div><span style="color:#6F91AE">●</span> Network node &nbsp;&nbsp; <span style="color:#24364B">●</span> Selected node &nbsp;&nbsp; <span style="color:#D6A34A">●</span> Predicted candidate</div>', unsafe_allow_html=True)
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
                st.markdown(f"<div class='result-row'><div><b style='color:#24364B'>#{rank:02d}</b> &nbsp; Node <b>{html.escape(str(candidate))}</b><br><span style='font-size:.9rem;color:#5B6778'>{html.escape(explanation)}</span></div><div style='text-align:right'><span class='status-label'>Score</span><br><b>{score:.4f}</b></div></div>", unsafe_allow_html=True)
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
                    charts = [("AUC", "auc", "#416A8A"), ("Precision@K", "precision_at_k", "#24364B"), ("Execution time (ms)", "execution_time_ms", "#D6A34A")]
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
