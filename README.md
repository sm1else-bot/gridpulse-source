# GridPulse — Distributed Renewable Energy Forecasting & Grid Mismatch Intelligence

**CSGY-6513 · Big Data · NYU · Spring 2026**

GridPulse is an end-to-end big-data pipeline that ingests 50–80 million hourly grid observations from the U.S. Energy Information Administration (EIA Form 930), engineers rich time-series features with PySpark, and trains three complementary machine-learning models for renewable generation forecasting, anomaly detection, and supply-demand mismatch classification.

---

## Project structure

```
gridpulse-source/
├── src/
│   ├── ingest.py        # EIA 930 bulk CSV downloader
│   ├── features.py      # PySpark distributed feature pipeline
│   └── models.py        # LightGBM / LSTM / XGBoost training
├── dashboard/
│   └── app.py           # Streamlit interactive dashboard
├── notebooks/
│   ├── 01_ingest.ipynb
│   ├── 02_features.ipynb
│   ├── 03_models.ipynb
│   └── 04_evaluation.ipynb
├── data/
│   ├── raw/             # EIA 930 CSVs (downloaded by ingest.py)
│   └── processed/       # features.parquet (output of features.py)
├── models/              # Saved model artefacts
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.13
- Java 17 (`JAVA_HOME` must be set)
- CUDA-capable GPU (optional — LSTM and XGBoost fall back to CPU gracefully)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment check

```bash
java -version          # must be 17+
python -c "import pyspark; print(pyspark.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running the pipeline

### Step 1 — Download raw data

```bash
python src/ingest.py
```

Downloads all 12 EIA 930 balance files (2019–2024) to `data/raw/`.  
Total download: ~3–5 GB. Skips files that already exist.  
Use `--force` to re-download.

### Step 2 — Feature engineering

```bash
python src/features.py
```

Runs the PySpark pipeline on all raw CSVs. Outputs `data/processed/features.parquet`  
(partitioned by `year` and `ba`). On 80M rows this takes 20–60 minutes.

Pass a sampling fraction for a quick test run:

```bash
python src/features.py 0.05   # 5% sample
```

### Step 3 — Model training

```bash
# All three models
python src/models.py

# Individual models
python src/models.py lgbm
python src/models.py lstm
python src/models.py xgb
```

### Step 4 — Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. The dashboard degrades gracefully when model  
artefacts are missing, rendering example panels with synthetic data.

---

## Why PySpark?

EIA Form 930 provides 50–80 million hourly observations across ~60 balancing  
authorities spanning 2019–2024. The feature-engineering step computes **per-BA window  
functions**: 48-hour lag columns and 7-day rolling statistics for three signals  
(solar, wind, demand). These are **partitioned window operations** — each BA requires  
its own sorted time series — and do not parallelize trivially in pandas.

PySpark's `Window.partitionBy("ba").orderBy("ts_unix")` distributes these operations  
across all 32 CPU cores simultaneously, with the shuffle coordinator managing  
cross-partition ordering. A serial pandas approach on this dataset would require  
~40 GB of working memory for a single `.groupby().rolling()` chain; PySpark's  
block-level processing keeps the driver heap below 32 GB while saturating all cores.

### Spark session configuration

| Parameter | Value | Rationale |
|---|---|---|
| `master` | `local[*]` | Uses all 32 cores |
| `spark.driver.memory` | `32g` | Headroom for the feature matrix |
| `spark.sql.shuffle.partitions` | `48` | 1.5× cores — avoids shuffle skew |
| `spark.sql.files.maxPartitionBytes` | `128m` | Balanced file splits |

---

## Models

### LightGBM Renewable Forecaster

Predicts solar and wind generation at t+6h, t+12h, and t+24h horizons.  
Six separate regressors trained with early stopping (50 rounds patience).  
Features: 144 lag columns, 18 rolling-stat columns, calendar features, BA encoding.

### LSTM Autoencoder (PyTorch CUDA)

A sequence-to-sequence LSTM autoencoder (hidden=128, latent=64) trained only on  
"normal" (class-0) operational hours. Reconstruction MSE is the anomaly score;  
the 99th-percentile threshold flags anomalous periods.

### XGBoost Mismatch Severity Classifier

Four-class classification: balanced / moderate surplus / severe surplus / deficit.  
Deficit = net generation < 95% of demand; Severe surplus = net generation > 120%  
of demand. Trained with `device='cuda'` (falls back to CPU). Uses the full 200+  
feature set including lag, rolling, calendar, and raw signal columns.

---

## Data source

**EIA Form 930 — Hourly Electric Grid Monitor**  
URL: https://www.eia.gov/electricity/gridmonitor/about  
Bulk files: https://www.eia.gov/electricity/gridmonitor/bulk-files  

The `EIA930_BALANCE_<YEAR>_<HALF>.csv` files report hourly demand, net generation,  
total interchange, and fuel-type-specific generation for every U.S. balancing  
authority. Data starts 2019-07-01.
