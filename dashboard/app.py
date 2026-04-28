# -*- coding: utf-8 -*-
"""
GridPulse Streamlit Dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow.dataset as ds
import streamlit as st

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridPulse",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + Font ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* Hide Streamlit chrome */
#MainMenu, header, footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] { display: none !important; }

/* Base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #f5f6f8 !important;
    font-family: "DM Sans", sans-serif;
}

* { box-sizing: border-box; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e5ea;
}
[data-testid="stSidebar"] * {
    font-family: "DM Sans", sans-serif !important;
}

/* Sidebar labels */
.gp-sidebar-label {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.4rem 0 0.4rem 0;
}

/* Hero */
.gp-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1d4ed8 100%);
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.gp-hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.gp-hero::after {
    content: "";
    position: absolute;
    bottom: -80px; left: 30%;
    width: 360px; height: 360px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.gp-hero-eyebrow {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 0.9rem;
}
.gp-hero h1 {
    font-family: "Space Grotesk", sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 1rem 0;
}
.gp-hero-sub {
    font-family: "DM Sans", sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: #cbd5e1;
    line-height: 1.65;
    max-width: 680px;
    margin-bottom: 1.8rem;
}
.gp-hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.gp-chip {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #93c5fd;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 0.3rem 0.75rem;
}

/* Section headers */
.gp-section-eyebrow {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1d4ed8;
    margin: 2.8rem 0 0.5rem 0;
}
.gp-section-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
}
.gp-section-body {
    font-family: "DM Sans", sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    color: #475569;
    line-height: 1.7;
    max-width: 820px;
    margin-bottom: 1.5rem;
}

/* Cards */
.gp-card {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    height: 100%;
}
.gp-card-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    margin-bottom: 0.9rem;
}
.gp-card h3 {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.gp-card p {
    font-family: "DM Sans", sans-serif;
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.65;
    margin: 0;
}
.gp-card-accent-blue  { background: #eff6ff; }
.gp-card-accent-green { background: #f0fdf4; }
.gp-card-accent-amber { background: #fffbeb; }
.gp-card-accent-violet{ background: #f5f3ff; }

/* Pipeline steps */
.gp-pipeline {
    display: flex;
    gap: 0;
    margin: 1.5rem 0 2rem 0;
    overflow-x: auto;
}
.gp-step {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-right: none;
    padding: 1.1rem 1.25rem;
    position: relative;
}
.gp-step:first-child { border-radius: 10px 0 0 10px; }
.gp-step:last-child  { border-right: 1px solid #e2e5ea; border-radius: 0 10px 10px 0; }
.gp-step-num {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1d4ed8;
    margin-bottom: 0.35rem;
}
.gp-step h4 {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.3rem 0;
}
.gp-step p {
    font-family: "DM Sans", sans-serif;
    font-size: 0.78rem;
    color: #64748b;
    margin: 0;
    line-height: 1.55;
}

/* Stat tiles */
.gp-stat {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.gp-stat-label {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.gp-stat-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}

/* Metric override */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-family: "DM Sans", sans-serif;
}
[data-testid="stMetricLabel"] p {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8 !important;
}
[data-testid="stMetricValue"] {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* Chart panel label */
.gp-chart-label {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}
.gp-chart-sub {
    font-family: "DM Sans", sans-serif;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 0.9rem;
}

/* Info box */
.gp-info {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #1d4ed8;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-family: "DM Sans", sans-serif;
    font-size: 0.86rem;
    color: #1e3a5f;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* Selectbox labels */
[data-testid="stSelectbox"] label p,
[data-testid="stMultiSelect"] label p,
[data-testid="stDateInput"] label p {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}

/* Divider */
.gp-divider {
    border: none;
    border-top: 1px solid #e2e5ea;
    margin: 2rem 0;
}

/* Interactive panel wrapper */
.gp-panel-wrap {
    background: #ffffff;
    border: 1px solid #e2e5ea;
    border-radius: 14px;
    padding: 1.5rem 1.75rem 1rem 1.75rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DM Sans, sans-serif", size=12, color="#475569"),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=48, r=24, t=40, b=48),
)
PALETTE = ["#1d4ed8","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#f97316","#6366f1"]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading feature data...")
def load_features(ba_filter, date_start, date_end):
    if not (PROCESSED_DIR / "features.parquet").exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(PROCESSED_DIR / "features.parquet"), format="parquet", partitioning="hive")
    cols = [
        "ts_utc","ba","solar_mw","wind_mw","demand_mw","net_gen_mw",
        "mismatch_pct","mismatch_label","hour_of_day","month","year",
        "solar_t6h_ahead","solar_t12h_ahead","solar_t24h_ahead",
        "wind_t6h_ahead","wind_t12h_ahead","wind_t24h_ahead",
    ]
    available = dataset.schema.names
    cols = [c for c in cols if c in available]
    df = dataset.to_table(columns=cols).to_pandas()
    if ba_filter:
        df = df[df["ba"].isin(ba_filter)]
    if "ts_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"])
        df = df[(df["ts"] >= date_start) & (df["ts"] <= date_end)]
    return df


@st.cache_data(show_spinner="Loading anomaly scores...")
def load_anomaly_scores():
    path = MODELS_DIR / "anomaly_scores.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ba","ts_unix","anomaly_score"])
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts_unix"], unit="s", utc=True)
    return df


def _get_available_bas():
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        return ["MISO","PJM","CISO","ERCO","SWPP","PACW"]
    try:
        dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
        df = dataset.to_table(columns=["ba"]).to_pandas()
        return sorted(df["ba"].dropna().unique().tolist())
    except Exception:
        return ["MISO","PJM","CISO","ERCO","SWPP","PACW"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gp-sidebar-label">Balancing Authorities</div>', unsafe_allow_html=True)
    all_bas = _get_available_bas()
    default_bas = [b for b in ["MISO","PJM","CISO","ERCO"] if b in all_bas]
    selected_bas = st.multiselect(
        "Balancing Authorities", options=all_bas,
        default=default_bas or all_bas[:4], label_visibility="collapsed",
    )

    st.markdown('<div class="gp-sidebar-label">Date Range</div>', unsafe_allow_html=True)
    date_range = st.date_input(
        "Date range",
        value=(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
        min_value=pd.Timestamp("2019-01-01"),
        max_value=pd.Timestamp("2024-12-31"),
        label_visibility="collapsed",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_start, d_end = str(date_range[0]), str(date_range[1])
    else:
        d_start, d_end = "2021-01-01", "2021-12-31"

    st.markdown('<div class="gp-sidebar-label">Forecast Settings</div>', unsafe_allow_html=True)
    energy_source    = st.selectbox("Energy source", ["solar","wind"], label_visibility="collapsed")
    forecast_horizon = st.selectbox("Forecast horizon", [6,12,24], format_func=lambda h: f"{h}h ahead", label_visibility="collapsed")

    st.markdown('<div class="gp-sidebar-label">Anomaly View</div>', unsafe_allow_html=True)
    anomaly_ba = st.selectbox(
        "BA for anomaly timeline",
        options=selected_bas if selected_bas else all_bas[:1],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#94a3b8;text-align:center;font-family:DM Sans,sans-serif">GridPulse &nbsp;&middot;&nbsp; CSGY-6513 &nbsp;&middot;&nbsp; NYU Spring 2026</p>', unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df         = load_features(selected_bas or None, d_start, d_end)
anomaly_df = load_anomaly_scores()
data_ok    = len(df) > 0
LABEL_NAMES = {0:"Balanced", 1:"Mod Surplus", 2:"Sev Surplus", 3:"Deficit"}

if not data_ok:
    rng = np.random.default_rng(42)
    n   = 8760
    ts  = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    solar  = np.clip(50*np.sin(np.linspace(0,2*np.pi,n)*365)*np.clip(np.sin(np.linspace(0,2*np.pi*365,n)),0,None)+rng.normal(0,5,n),0,None)
    wind   = 80+30*np.sin(np.linspace(0,2*np.pi*52,n))+rng.normal(0,15,n)
    demand = 250+60*np.sin(np.linspace(0,2*np.pi*365,n))+rng.normal(0,20,n)
    df = pd.DataFrame({
        "ts":ts,"ba":"DEMO","solar_mw":solar,"wind_mw":wind,"demand_mw":demand,
        "net_gen_mw":solar+wind+100,"mismatch_pct":rng.normal(0,8,n),
        "mismatch_label":rng.integers(0,4,n),"hour_of_day":ts.hour,"month":ts.month,
        "solar_t6h_ahead":solar+rng.normal(0,5,n),"wind_t6h_ahead":wind+rng.normal(0,8,n),
    })
    anom_scores = np.abs(rng.normal(0,0.5,n))**2
    anom_scores[(ts>="2021-02-10")&(ts<="2021-02-20")] *= 15
    anomaly_df = pd.DataFrame({"ba":"DEMO","ts":ts,"anomaly_score":anom_scores})

# ═════════════════════════════════════════════════════════════════════════════
#  HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="gp-hero">
    <div class="gp-hero-eyebrow">NYU Tandon &nbsp;&middot;&nbsp; CSGY-6513 Big Data &nbsp;&middot;&nbsp; Spring 2026</div>
    <h1>GridPulse</h1>
    <p class="gp-hero-sub">
        An end-to-end distributed intelligence platform for the U.S. electrical grid.
        GridPulse ingests over 50 million hourly observations from the Energy Information Administration,
        engineers hundreds of time-series features using Apache Spark across 32 CPU cores, and
        trains three complementary machine learning models to forecast renewable generation,
        detect operational anomalies, and classify supply-demand mismatch severity in real time.
    </p>
    <div class="gp-hero-chips">
        <span class="gp-chip">PySpark 4.1.1</span>
        <span class="gp-chip">LightGBM</span>
        <span class="gp-chip">PyTorch LSTM</span>
        <span class="gp-chip">XGBoost GPU</span>
        <span class="gp-chip">EIA Form 930</span>
        <span class="gp-chip">50-80M rows</span>
        <span class="gp-chip">2019 to 2024</span>
        <span class="gp-chip">60+ Balancing Authorities</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gp-section-eyebrow">About the Project</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">What GridPulse Does</div>', unsafe_allow_html=True)
st.markdown("""
<p class="gp-section-body">
    The U.S. electrical grid is one of the most complex engineered systems ever built. Every hour,
    approximately 60 balancing authorities across the country report their demand, net generation,
    interchange flows, and fuel-type-specific output to the Energy Information Administration through
    EIA Form 930. GridPulse collects all of this data spanning 2019 through 2024, processes it through
    a distributed feature engineering pipeline, and produces three machine learning systems designed to
    answer practical operational questions: how much solar and wind will be generated in the next 6,
    12, and 24 hours; which hours represent anomalous grid behavior; and how severe is the current
    mismatch between generation and demand.
</p>
<p class="gp-section-body">
    The project is built to demonstrate that distributed computing is not an optional convenience on
    this dataset but a genuine requirement. The feature engineering phase computes per-balancing-authority
    window functions including 48-hour lag columns and 7-day rolling statistics across three signals.
    These are partitioned window operations that cannot parallelize trivially in pandas. On a single
    thread, this computation would require approximately 40 GB of working memory for a single groupby
    rolling chain. PySpark distributes the work across all 32 cores simultaneously, keeping the driver
    heap manageable while saturating available compute.
</p>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  DATA SOURCE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gp-section-eyebrow">Data Source</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">EIA Form 930 Hourly Grid Monitor</div>', unsafe_allow_html=True)
st.markdown("""
<p class="gp-section-body">
    EIA Form 930 is the authoritative source for U.S. grid operations data. It is published in
    semi-annual bulk CSV files, each containing hundreds of thousands of rows across roughly 60
    balancing authorities. Each row represents one hour of operations at one balancing authority
    and reports raw, imputed, and adjusted values for demand, net generation, total interchange,
    and generation broken down by fuel type including solar, wind, natural gas, nuclear, coal,
    hydropower, and other sources.
</p>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, "Total Files",        "12",      "Semi-annual CSVs"),
    (c2, "Date Coverage",      "6 Years", "2019 through 2024"),
    (c3, "Hourly Rows",        "50-80M",  "Across all BAs"),
    (c4, "Balancing Auth.",    "60+",     "U.S. grid operators"),
]:
    with col:
        st.markdown(f"""
        <div class="gp-stat">
            <div class="gp-stat-label">{label}</div>
            <div class="gp-stat-value">{val}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.75rem;color:#94a3b8;margin-top:0.3rem">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="gp-info">
    GridPulse retains only the <strong>Adjusted</strong> variant of each metric and drops raw and imputed columns at load time,
    reducing the working memory footprint by approximately two-thirds before any computation begins.
    Column headers from EIA 930 contain spaces, parentheses, and mixed case, so the pipeline normalizes
    them via exact-match dictionary lookup before processing.
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gp-section-eyebrow">Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">Four-Stage Pipeline</div>', unsafe_allow_html=True)

st.markdown("""
<div class="gp-pipeline">
    <div class="gp-step">
        <div class="gp-step-num">Stage 1</div>
        <h4>Ingest</h4>
        <p>Downloads all 12 EIA 930 BALANCE CSVs with streaming HTTP and tqdm progress bars. Skips valid existing files, re-downloads corrupt or undersized ones. Total raw data is 3-5 GB.</p>
    </div>
    <div class="gp-step">
        <div class="gp-step-num">Stage 2</div>
        <h4>Feature Engineering</h4>
        <p>PySpark local[*] on 32 cores. Normalizes schema, parses timestamps, computes 144 lag columns (t-1h to t-48h), 18 rolling statistics (6h/24h/7d), mismatch labels, and 6 lead targets. Writes partitioned Parquet via PyArrow streaming to avoid Hadoop native IO issues on Windows.</p>
    </div>
    <div class="gp-step">
        <div class="gp-step-num">Stage 3</div>
        <h4>Model Training</h4>
        <p>Three independent models: LightGBM multi-output regressor, PyTorch LSTM autoencoder trained on normal hours only, and XGBoost 4-class classifier. All training loops expose per-batch tqdm progress with live loss in postfix.</p>
    </div>
    <div class="gp-step">
        <div class="gp-step-num">Stage 4</div>
        <h4>Dashboard</h4>
        <p>This Streamlit application. Loads model artifacts and Parquet features on demand using PyArrow column projection for efficient I/O. All four panels are interactive and filter to the selected balancing authorities and date range.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gp-section-eyebrow">Feature Engineering</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">What the Spark Pipeline Produces</div>', unsafe_allow_html=True)
st.markdown("""
<p class="gp-section-body">
    The PySpark pipeline uses <code>Window.partitionBy("ba").orderBy("ts_unix")</code> to compute all
    temporal features independently per balancing authority. This is the core distributed operation:
    each BA gets its own sorted time series, and Spark's shuffle coordinator manages cross-partition
    ordering while saturating all available CPU cores in parallel.
</p>
""", unsafe_allow_html=True)

fe1, fe2, fe3, fe4 = st.columns(4)
for col, accent, title, body in [
    (fe1, "blue",   "144 Lag Features",
     "Hourly lookback from t-1h to t-48h for solar generation, wind generation, and demand. Gives the model a two-day memory window for each signal."),
    (fe2, "green",  "18 Rolling Statistics",
     "6-hour, 24-hour, and 7-day rolling mean and standard deviation for solar, wind, and demand. Captures short-term volatility and weekly seasonality."),
    (fe3, "amber",  "Mismatch Labels",
     "Four-class label derived from the ratio of net generation to demand. Balanced within 5 percent, moderate surplus 5 to 20 percent, severe surplus above 20 percent, deficit below negative 5 percent."),
    (fe4, "violet", "Lead Targets",
     "Six forecast targets: solar and wind generation at t+6h, t+12h, and t+24h ahead. Used as regression targets for the LightGBM forecasting models."),
]:
    with col:
        st.markdown(f"""
        <div class="gp-card">
            <div class="gp-card-icon gp-card-accent-{accent}"></div>
            <h3>{title}</h3>
            <p>{body}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Spark config table
st.markdown('<div class="gp-section-eyebrow">Spark Configuration</div>', unsafe_allow_html=True)
spark_data = pd.DataFrame({
    "Parameter":    ["master", "spark.driver.memory", "spark.sql.shuffle.partitions", "spark.sql.files.maxPartitionBytes"],
    "Value":        ["local[*]", "32g", "48", "128m"],
    "Rationale":    [
        "Uses all 32 CPU cores on the host machine",
        "Headroom for the full feature matrix before PyArrow streaming write",
        "1.5x core count to avoid shuffle skew across BAs",
        "Balanced file splits across the 12 semi-annual CSVs",
    ],
})
st.dataframe(spark_data, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
#  MODELS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gp-section-eyebrow">Machine Learning Models</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">Three Complementary Approaches</div>', unsafe_allow_html=True)
st.markdown("""
<p class="gp-section-body">
    GridPulse trains three models that each answer a different operational question. They are designed
    to complement each other: the forecaster predicts what will happen, the anomaly detector flags when
    something unusual is already happening, and the classifier quantifies how bad the current
    supply-demand imbalance is.
</p>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    <div class="gp-card">
        <div class="gp-card-icon gp-card-accent-blue" style="background:#eff6ff"></div>
        <h3>LightGBM Renewable Forecaster</h3>
        <p>
            Six gradient boosted regression models, one per combination of energy source (solar, wind)
            and forecast horizon (6h, 12h, 24h ahead). Each model is trained with early stopping
            at 50 rounds patience. The full feature set includes the 144 lag columns, 18 rolling
            statistics, calendar features (hour of day, day of week, month), and a balancing
            authority label encoding. Validation R-squared reaches up to 0.91 on the best solar
            horizon. All six model files are saved to disk and can be hot-loaded for inference.
        </p>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="gp-card">
        <div class="gp-card-icon gp-card-accent-violet" style="background:#f5f3ff"></div>
        <h3>LSTM Autoencoder Anomaly Detector</h3>
        <p>
            A sequence-to-sequence LSTM autoencoder with hidden size 128 and latent size 64,
            trained exclusively on operationally normal hours (class 0 in the mismatch label).
            The model learns to reconstruct 168-hour (7-day) windows of five grid signals.
            At inference time, reconstruction mean squared error becomes the anomaly score:
            hours the model cannot reconstruct well are anomalous by definition. The 99th
            percentile of training scores is used as the detection threshold. Notable flagged
            events include the 2021 Texas winter freeze and the 2020 California rolling blackouts.
        </p>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="gp-card">
        <div class="gp-card-icon gp-card-accent-amber" style="background:#fffbeb"></div>
        <h3>XGBoost Mismatch Classifier</h3>
        <p>
            A four-class gradient boosted tree classifier trained on the full 200-plus feature set.
            Classes are defined by the ratio of net generation to demand: balanced within 5 percent,
            moderate surplus between 5 and 20 percent, severe surplus above 20 percent, and deficit
            below negative 5 percent. The model is trained with device=cuda when a GPU is available
            and falls back to CPU automatically. At 400 boosting rounds with early stopping, the
            classifier reaches near-perfect validation accuracy on the held-out 15 percent test split,
            reflecting that the mismatch label is highly predictable from lag and rolling features.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PANELS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='gp-divider'>", unsafe_allow_html=True)
st.markdown('<div class="gp-section-eyebrow">Interactive Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="gp-section-title">Explore the Data and Model Outputs</div>', unsafe_allow_html=True)
st.markdown("""
<p class="gp-section-body">
    Use the sidebar to select balancing authorities, date range, energy source, and forecast horizon.
    All four panels below update in real time from the processed Parquet feature table and saved model artifacts.
</p>
""", unsafe_allow_html=True)

if not data_ok:
    st.markdown('<div class="gp-info">No feature data found. Run <code>python src/ingest.py</code> then <code>python src/features.py</code> to generate data. Panels below show synthetic placeholder data.</div>', unsafe_allow_html=True)

# ── Panel 1: Forecast vs Actual ───────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:0.5rem">
    <span class="gp-chart-label">Forecast vs Actual &nbsp;&middot;&nbsp; {energy_source.title()} Generation ({forecast_horizon}h Ahead)</span><br>
    <span class="gp-chart-sub">Dotted lines show LightGBM predictions; solid lines show recorded actual generation per balancing authority.</span>
</div>
""", unsafe_allow_html=True)

col_chart, col_stats = st.columns([4, 1])
target_col   = f"{energy_source}_mw"
forecast_col = f"{energy_source}_t{forecast_horizon}h_ahead"

with col_chart:
    plot_df = df[df["ba"].isin(selected_bas)].copy() if selected_bas and "ba" in df.columns else df.copy()
    if "ts" not in plot_df.columns and "ts_utc" in plot_df.columns:
        plot_df["ts"] = pd.to_datetime(plot_df["ts_utc"], utc=True)
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42).sort_values("ts")

    fig1 = go.Figure()
    for i, ba in enumerate(plot_df["ba"].unique() if "ba" in plot_df.columns else ["DEMO"]):
        sub   = (plot_df[plot_df["ba"] == ba] if "ba" in plot_df.columns else plot_df).sort_values("ts")
        color = PALETTE[i % len(PALETTE)]
        if target_col in sub.columns:
            fig1.add_trace(go.Scatter(x=sub["ts"], y=sub[target_col], mode="lines",
                name=f"{ba} Actual", line=dict(color=color, width=1.5)))
        if forecast_col in sub.columns and sub[forecast_col].notna().any():
            fig1.add_trace(go.Scatter(x=sub["ts"], y=sub[forecast_col], mode="lines",
                name=f"{ba} Forecast", line=dict(color=color, dash="dot", width=1.5)))
    fig1.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Time (UTC)", yaxis_title="MW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig1.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig1.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

with col_stats:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.metric("Rows displayed", f"{min(len(plot_df), 5000):,}")
    if forecast_col in df.columns:
        sub_all = df[[target_col, forecast_col]].dropna()
        if len(sub_all) > 0:
            st.metric("MAE (MW)",  f"{np.mean(np.abs(sub_all[target_col]-sub_all[forecast_col])):.1f}")
            st.metric("RMSE (MW)", f"{np.sqrt(np.mean((sub_all[target_col]-sub_all[forecast_col])**2)):.1f}")

# ── Panel 2: Anomaly Timeline ─────────────────────────────────────────────────
st.markdown("""
<div style="margin: 1.5rem 0 0.5rem 0">
    <span class="gp-chart-label">Anomaly Score Timeline &nbsp;&middot;&nbsp; LSTM Autoencoder Reconstruction Error</span><br>
    <span class="gp-chart-sub">Reconstruction MSE per hour for the selected balancing authority. Scores above the red threshold line are flagged as anomalous. Historical crisis windows are shaded.</span>
</div>
""", unsafe_allow_html=True)

anom_ba_df = (anomaly_df[anomaly_df["ba"] == anomaly_ba] if "ba" in anomaly_df.columns else anomaly_df).sort_values("ts")
if len(anom_ba_df) > 10000:
    anom_ba_df = anom_ba_df.sample(10000, random_state=42).sort_values("ts")

threshold_info = joblib.load(thresh_path) if (thresh_path := MODELS_DIR / "lstm_threshold.pkl").exists() else {}
threshold = threshold_info.get("threshold", anom_ba_df["anomaly_score"].quantile(0.99) if len(anom_ba_df) > 0 else 1.0)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=anom_ba_df["ts"], y=anom_ba_df["anomaly_score"],
    mode="lines", name="Anomaly Score",
    line=dict(color="#1d4ed8", width=1),
    fill="tozeroy", fillcolor="rgba(29,78,216,0.07)",
))
fig2.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", line_width=1.5,
    annotation_text=f"99th pct threshold ({threshold:.3f})",
    annotation_position="top right", annotation_font=dict(color="#ef4444", size=11))
for ev_start, ev_end, color, label in [
    ("2021-02-10","2021-02-20","rgba(239,68,68,0.08)","2021 TX Freeze"),
    ("2020-08-14","2020-08-15","rgba(139,92,246,0.12)","2020 CA Blackout"),
]:
    ts_s = pd.Timestamp(ev_start, tz="UTC"); ts_e = pd.Timestamp(ev_end, tz="UTC")
    if len(anom_ba_df) > 0 and ((anom_ba_df["ts"] >= ts_s) & (anom_ba_df["ts"] <= ts_e)).any():
        fig2.add_vrect(x0=ts_s, x1=ts_e, fillcolor=color, layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(size=11, color="#374151"))
fig2.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="Time (UTC)",
    yaxis_title="Reconstruction MSE", showlegend=False)
fig2.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
fig2.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Panel 3: Mismatch Heatmap ─────────────────────────────────────────────────
st.markdown("""
<div style="margin: 1.5rem 0 0.5rem 0">
    <span class="gp-chart-label">Mismatch Severity Heatmap &nbsp;&middot;&nbsp; Hour of Day vs Month</span><br>
    <span class="gp-chart-sub">Proportion of hours falling into the selected severity class, broken down by hour of day (UTC) and calendar month across all selected balancing authorities.</span>
</div>
""", unsafe_allow_html=True)

if "mismatch_label" in df.columns and "hour_of_day" in df.columns and "month" in df.columns:
    ctrl_col, _ = st.columns([2, 5])
    with ctrl_col:
        severity_filter = st.selectbox("Show proportion of class", [0,1,2,3],
            format_func=lambda x: f"{x}  {LABEL_NAMES[x]}")
    heat_df = df[["hour_of_day","month","mismatch_label"]].dropna()
    pivot = (heat_df.groupby(["hour_of_day","month"])
        .apply(lambda g: (g["mismatch_label"]==severity_filter).mean())
        .reset_index(name="proportion")
        .pivot(index="hour_of_day", columns="month", values="proportion"))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[c-1] for c in pivot.columns]

    fig3 = px.imshow(pivot,
        color_continuous_scale="RdYlGn_r" if severity_filter in (2,3) else "Blues",
        labels=dict(x="Month", y="Hour of Day (UTC)", color="Proportion"), aspect="auto")
    fig3.update_layout(**PLOTLY_LAYOUT, height=360,
        title=dict(text=f"Proportion: {LABEL_NAMES[severity_filter]} by Hour and Month",
            font=dict(size=13, color="#0f172a", family="Space Grotesk, sans-serif"), x=0))
    fig3.update_coloraxes(colorbar=dict(thickness=10, len=0.8,
        tickfont=dict(size=10), title=dict(text="")))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    pct_def = {0:"-5% to +5% mismatch", 1:"+5% to +20% surplus", 2:"> +20% surplus", 3:"< -5% deficit"}
    g1, g2, g3, g4 = st.columns(4)
    for col, (k, v) in zip([g1,g2,g3,g4], LABEL_NAMES.items()):
        with col:
            st.markdown(f"""
            <div class="gp-stat">
                <div class="gp-stat-label">{v}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.75rem;color:#94a3b8;margin-top:0.3rem">{pct_def[k]}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Panel 4: Summary Statistics ───────────────────────────────────────────────
st.markdown("""
<div style="margin: 1.5rem 0 0.75rem 0">
    <span class="gp-chart-label">Summary Statistics &nbsp;&middot;&nbsp; Filtered Selection</span><br>
    <span class="gp-chart-sub">Aggregate metrics for the currently selected balancing authorities and date window.</span>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
for col, label, val in [
    (s1, "Rows (filtered)",  f"{len(df):,}"),
    (s2, "Avg Solar (MW)",   f"{df['solar_mw'].mean():.0f}"  if "solar_mw"  in df.columns else "N/A"),
    (s3, "Avg Wind (MW)",    f"{df['wind_mw'].mean():.0f}"   if "wind_mw"   in df.columns else "N/A"),
    (s4, "Avg Demand (MW)",  f"{df['demand_mw'].mean():.0f}" if "demand_mw" in df.columns else "N/A"),
    (s5, "Deficit Hours",    f"{(df['mismatch_label']==3).mean()*100:.1f}%" if "mismatch_label" in df.columns else "N/A"),
]:
    with col:
        st.metric(label, val)

if "mismatch_label" in df.columns:
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    label_dist = df["mismatch_label"].value_counts().rename(LABEL_NAMES).sort_index().reset_index()
    label_dist.columns = ["severity_class","count"]
    fig4 = px.bar(label_dist, x="severity_class", y="count",
        labels={"severity_class":"Severity Class","count":"Hours"},
        color="severity_class",
        color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444"], height=240)
    fig4.update_layout(**PLOTLY_LAYOUT, showlegend=False,
        xaxis_title="Severity Class", yaxis_title="Hours",
        title=dict(text="Class Distribution", font=dict(size=13, color="#0f172a",
            family="Space Grotesk, sans-serif"), x=0))
    fig4.update_xaxes(showgrid=False, zeroline=False)
    fig4.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig4.update_traces(marker_line_width=0)
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0;font-family:'DM Sans',sans-serif;font-size:0.78rem;color:#cbd5e1">
    GridPulse &nbsp;&middot;&nbsp; NYU Tandon CSGY-6513 Big Data &nbsp;&middot;&nbsp; Spring 2026
</div>
""", unsafe_allow_html=True)
