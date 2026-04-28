"""
PySpark feature-engineering pipeline for EIA Form 930 data.

Reads all raw CSV files, normalizes schema, computes:
  - Lag features t-1h through t-48h for solar, wind, and demand
  - Rolling mean+std over 6h / 24h / 7d windows
  - Mismatch labels (4-class severity)
  - Lead targets 6h / 12h / 24h ahead (for LightGBM forecaster)

Writes final feature table to data/processed/features.parquet.
"""

import logging
import os
import urllib.request
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gridpulse.features")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Exact lowercased column name → canonical name.
# Only the (Adjusted) variants are kept; Imputed and raw are dropped before this runs.
_COL_EXACT = {
    "utc time at end of hour":                                    "ts_utc",
    "balancing authority":                                        "ba",
    "demand forecast (mw)":                                       "demand_forecast_mw",
    "demand (mw) (adjusted)":                                     "demand_mw",
    "net generation (mw) (adjusted)":                             "net_gen_mw",
    "total interchange (mw) (adjusted)":                          "interchange_mw",
    "net generation (mw) from coal (adjusted)":                   "coal_mw",
    "net generation (mw) from natural gas (adjusted)":            "gas_mw",
    "net generation (mw) from nuclear (adjusted)":                "nuclear_mw",
    "net generation (mw) from all petroleum products (adjusted)": "oil_mw",
    "net generation (mw) from hydropower and pumped storage (adjusted)": "hydro_mw",
    "net generation (mw) from solar (adjusted)":                  "solar_mw",
    "net generation (mw) from wind (adjusted)":                   "wind_mw",
    "net generation (mw) from other fuel sources (adjusted)":     "other_mw",
    "net generation (mw) from unknown fuel sources (adjusted)":   "unknown_mw",
}

NUMERIC_COLS = [
    "demand_mw", "demand_forecast_mw", "net_gen_mw", "interchange_mw",
    "coal_mw", "gas_mw", "nuclear_mw", "oil_mw",
    "hydro_mw", "solar_mw", "wind_mw", "other_mw", "unknown_mw",
]
LAG_TARGETS = ["solar_mw", "wind_mw", "demand_mw"]
LAG_HOURS = list(range(1, 49))
ROLLING = {"6h": 6, "24h": 24, "7d": 168}  # hours
LEAD_HOURS = [6, 12, 24]
LEAD_TARGETS = ["solar_mw", "wind_mw"]


_WINUTILS_URL = (
    "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe"
)


def _ensure_winutils() -> None:
    """On Windows, download winutils.exe and set HADOOP_HOME.

    We only need winutils.exe — Hadoop calls it via subprocess for chmod/mkdir.
    We intentionally do NOT put .hadoop/bin on PATH or java.library.path:
    if hadoop.dll is absent, NativeCodeLoader.isNativeCodeLoaded() returns false
    and NativeIO silently falls back to pure-Java file access.  Loading a
    version-mismatched hadoop.dll causes UnsatisfiedLinkError, so the safest
    approach on Windows is to leave native IO disabled.
    """
    if os.name != "nt":
        return

    hadoop_home = Path(__file__).parent.parent / ".hadoop"
    bin_dir = hadoop_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    winutils = bin_dir / "winutils.exe"
    if not winutils.exists():
        log.info("Downloading winutils.exe → %s", winutils)
        urllib.request.urlretrieve(_WINUTILS_URL, str(winutils))
        log.info("  winutils.exe  (%.0f KB)", winutils.stat().st_size / 1024)

    os.environ["HADOOP_HOME"] = str(hadoop_home)
    os.environ["hadoop.home.dir"] = str(hadoop_home)
    log.info("HADOOP_HOME=%s", hadoop_home)


def _get_spark() -> SparkSession:
    _ensure_winutils()
    return (
        SparkSession.builder
        .appName("GridPulse-Features")
        .master("local[*]")
        .config("spark.driver.memory", "32g")
        .config("spark.sql.shuffle.partitions", "48")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # v2 committer writes tasks directly to the output dir, skipping
        # getAllCommittedTaskPaths() -> listStatus() -> NativeIO.Windows.access0()
        # which crashes on Windows because hadoop-client-api 3.4.2 has no Java fallback
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )


def _normalize_columns(df):
    """
    1. Drop Imputed and raw (unadjusted) columns immediately to reduce memory.
    2. Rename remaining columns to canonical names via exact-match lookup.
    3. Drop anything not in the keep set.
    """
    # Step 1 — drop Imputed columns and known noise columns up front
    imputed_and_noise = [
        c for c in df.columns
        if "(imputed)" in c.lower()
        or c in ("Sum(Valid DIBAs) (MW)", "Region",
                 "Data Date", "Hour Number", "Local Time at End of Hour")
    ]
    # Also drop the raw (non-adjusted, non-imputed) overlapping columns
    raw_unadjusted = [
        "Demand (MW)", "Net Generation (MW)", "Total Interchange (MW)",
        "Net Generation (MW) from Coal",
        "Net Generation (MW) from Natural Gas",
        "Net Generation (MW) from Nuclear",
        "Net Generation (MW) from All Petroleum Products",
        "Net Generation (MW) from Hydropower and Pumped Storage",
        "Net Generation (MW) from Solar",
        "Net Generation (MW) from Wind",
        "Net Generation (MW) from Other Fuel Sources",
        "Net Generation (MW) from Unknown Fuel Sources",
    ]
    to_drop = [c for c in imputed_and_noise + raw_unadjusted if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    # Step 2 — rename via exact lowercase match
    for raw in df.columns:
        canonical = _COL_EXACT.get(raw.strip().lower())
        if canonical:
            df = df.withColumnRenamed(raw, canonical)

    # Step 3 — drop anything still not needed
    keep = {"ts_utc", "ba"} | set(NUMERIC_COLS)
    drop = [c for c in df.columns if c not in keep]
    if drop:
        df = df.drop(*drop)

    return df


def _cast_numerics(df):
    for col in NUMERIC_COLS:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))
        else:
            df = df.withColumn(col, F.lit(None).cast("double"))
    return df


def _add_time_features(df):
    df = df.withColumn("ts", F.to_timestamp("ts_utc", "MM/dd/yyyy h:mm:ss a"))
    df = df.withColumn("ts_unix", F.unix_timestamp("ts"))
    df = df.withColumn("year",        F.year("ts"))
    df = df.withColumn("month",       F.month("ts"))
    df = df.withColumn("day_of_week", F.dayofweek("ts"))
    df = df.withColumn("hour_of_day", F.hour("ts"))
    df = df.withColumn("week_of_year", F.weekofyear("ts"))
    return df


def _add_lag_features(df, spark):
    """Add lag columns t-1h to t-48h for solar, wind, demand."""
    w_ba = (
        Window
        .partitionBy("ba")
        .orderBy("ts_unix")
    )

    log.info("Adding %d lag features for %d columns ...", len(LAG_HOURS), len(LAG_TARGETS))
    lag_cols = []
    with tqdm(LAG_HOURS, desc="  Lag hours", unit="h", ncols=80) as pbar:
        for h in pbar:
            pbar.set_postfix(lag=f"t-{h}h")
            for col in LAG_TARGETS:
                alias = f"{col[:-3]}_lag_{h}h"   # e.g. solar_lag_1h
                lag_cols.append(F.lag(col, h).over(w_ba).alias(alias))

    df = df.select("*", *lag_cols)
    return df


def _add_rolling_features(df):
    """Add rolling mean and std over 6h / 24h / 7d windows."""
    log.info("Adding rolling statistics (%s windows) ...", list(ROLLING.keys()))
    roll_cols = []
    items = [(label, hours, col) for label, hours in ROLLING.items() for col in LAG_TARGETS]

    with tqdm(items, desc="  Rolling stats", unit="col", ncols=80) as pbar:
        for label, hours, col in pbar:
            pbar.set_postfix(window=label, col=col)
            secs = hours * 3600
            w = (
                Window
                .partitionBy("ba")
                .orderBy("ts_unix")
                .rangeBetween(-(secs - 3600), 0)  # current + preceding (hours-1)
            )
            base = col[:-3]  # strip "_mw"
            roll_cols.append(F.mean(col).over(w).alias(f"{base}_roll{label}_mean"))
            roll_cols.append(F.stddev_pop(col).over(w).alias(f"{base}_roll{label}_std"))

    df = df.select("*", *roll_cols)
    return df


def _add_mismatch_labels(df):
    """Compute mismatch_mw, mismatch_pct, and 4-class mismatch_label."""
    df = df.withColumn("mismatch_mw", F.col("net_gen_mw") - F.col("demand_mw"))
    df = df.withColumn(
        "mismatch_pct",
        F.when(F.col("demand_mw") != 0,
               (F.col("net_gen_mw") - F.col("demand_mw")) / F.col("demand_mw") * 100)
        .otherwise(F.lit(None))
    )
    df = df.withColumn(
        "mismatch_label",
        F.when(F.col("mismatch_pct").isNull(), F.lit(None).cast("int"))
        .when(F.col("mismatch_pct") < -5.0,  F.lit(3))   # deficit
        .when(F.col("mismatch_pct") > 20.0,  F.lit(2))   # severe surplus
        .when(F.col("mismatch_pct") > 5.0,   F.lit(1))   # moderate surplus
        .otherwise(F.lit(0))                               # balanced
    )
    return df


def _add_lead_targets(df):
    """Add lead (future) values for solar+wind generation — training targets."""
    log.info("Adding lead targets at %s hours ahead ...", LEAD_HOURS)
    lead_cols = []
    items = [(h, col) for h in LEAD_HOURS for col in LEAD_TARGETS]
    with tqdm(items, desc="  Lead targets", unit="col", ncols=80) as pbar:
        for h, col in pbar:
            pbar.set_postfix(horizon=f"t+{h}h", col=col)
            w_ba = Window.partitionBy("ba").orderBy("ts_unix")
            base = col[:-3]
            lead_cols.append(F.lead(col, h).over(w_ba).alias(f"{base}_t{h}h_ahead"))
    df = df.select("*", *lead_cols)
    return df


def run_features(sample_frac: float = 1.0) -> None:
    """Full feature engineering pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("EIA930_BALANCE_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No EIA 930 CSV files found in {RAW_DIR}. Run src/ingest.py first."
        )

    spark = _get_spark()
    log.info("Spark session started — executor: local[*]")
    log.info("Found %d CSV files to process", len(csv_files))

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading CSVs ...")
    paths = [str(p) for p in csv_files]
    with tqdm(paths, desc="Reading CSVs", unit="file", ncols=80) as pbar:
        frames = []
        for p in pbar:
            pbar.set_postfix(file=Path(p).name)
            frames.append(
                spark.read.option("header", "true").csv(p)
            )

    df = frames[0]
    for frame in tqdm(frames[1:], desc="Unioning frames", unit="frame", ncols=80):
        df = df.unionByName(frame, allowMissingColumns=True)

    log.info("Raw rows: %s (before dedup)", "~50-80M")

    # ── Schema normalisation ──────────────────────────────────────────────────
    log.info("Normalising schema ...")
    df = _normalize_columns(df)
    df = _cast_numerics(df)
    df = df.dropna(subset=["ts_utc", "ba"])
    df = df.dropDuplicates(["ts_utc", "ba"])

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        log.info("Sampling at %.0f%%", sample_frac * 100)

    # Coalesce nulls in fuel columns to 0 so window math doesn't propagate NaN
    for col in ["solar_mw", "wind_mw", "coal_mw", "gas_mw",
                "nuclear_mw", "oil_mw", "hydro_mw", "other_mw"]:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    # ── Time features ─────────────────────────────────────────────────────────
    log.info("Parsing timestamps and adding time features ...")
    df = _add_time_features(df)
    df = df.filter(F.col("ts").isNotNull())

    # Cache before the multi-pass window operations
    log.info("Caching cleaned base dataframe ...")
    df = df.cache()
    df.count()   # materialise cache
    log.info("Cache materialised.")

    # ── Window features ───────────────────────────────────────────────────────
    df = _add_lag_features(df, spark)
    log.info("Lag features added.")

    df = _add_rolling_features(df)
    log.info("Rolling statistics added.")

    df = _add_mismatch_labels(df)
    log.info("Mismatch labels added.")

    df = _add_lead_targets(df)
    log.info("Lead targets added.")

    # Materialise the full feature DataFrame into Spark's in-memory cache.
    # toLocalIterator() would otherwise compute window functions lazily per
    # partition which can exceed the socket timeout on large datasets.
    log.info("Caching full feature DataFrame (window functions materialised) ...")
    df = df.cache()
    total_rows = df.count()
    log.info("Full feature cache materialised — %d rows.", total_rows)

    # ── Write parquet via pyarrow streaming (bypasses Hadoop filesystem on Windows) ──
    # toLocalIterator() streams rows to the driver — no Hadoop file writes, no Python
    # worker spawn. We batch into 50k-row chunks and write with pyarrow ParquetWriter.
    import shutil
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = PROCESSED_DIR / "features.parquet"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    out_file = str(out_dir / "part-00000.parquet")

    log.info("Streaming feature table → %s  (pyarrow batched writer)", out_file)

    BATCH = 50_000
    writer = None
    pinned_schema = None   # derived from first batch, enforced for all subsequent
    batch: list = []
    written = 0

    def _flush(buf: list) -> None:
        nonlocal writer, pinned_schema, written
        pdf = pd.DataFrame(buf)
        if pinned_schema is None:
            table = pa.Table.from_pandas(pdf, preserve_index=False)
            # Replace pa.null() fields (all-null first batch) with float64
            # so subsequent batches with real values don't cause a type error.
            fixed_fields = [
                pa.field(f.name, pa.float64()) if pa.types.is_null(f.type) else f
                for f in table.schema
            ]
            pinned_schema = pa.schema(fixed_fields)
            table = table.cast(pinned_schema)
            writer = pq.ParquetWriter(out_file, pinned_schema, compression="snappy")
        else:
            # cast to pinned schema so batch-level type differences don't crash
            table = pa.Table.from_pandas(pdf, schema=pinned_schema,
                                         preserve_index=False, safe=False)
        writer.write_table(table)
        written += len(buf)

    row_iter = df.repartition(48).toLocalIterator(prefetchPartitions=True)
    with tqdm(row_iter, desc="Writing parquet", unit="row", unit_scale=True, ncols=80) as pbar:
        for row in pbar:
            batch.append(row.asDict(recursive=True))
            if len(batch) >= BATCH:
                _flush(batch)
                batch.clear()
                pbar.set_postfix(rows=f"{written:,}")

    if batch:
        _flush(batch)

    if writer:
        writer.close()

    log.info("Feature table written successfully — %d rows.", written)

    # Quick summary
    log.info("Schema preview:")
    df.printSchema()
    sample = df.select("ts_utc", "ba", "solar_mw", "wind_mw",
                        "demand_mw", "mismatch_label").limit(3)
    sample.show(truncate=False)

    spark.stop()


def main():
    import sys
    sample = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    run_features(sample_frac=sample)


if __name__ == "__main__":
    main()
