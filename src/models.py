"""
GridPulse model training: three complementary models on the feature table.

  1. LightGBM forecaster  — solar + wind generation at t+6h, t+12h, t+24h
  2. LSTM Autoencoder     — anomaly detection via reconstruction error (PyTorch/CUDA)
  3. XGBoost classifier   — 4-class supply/demand mismatch severity (GPU)

All model artefacts are saved to models/.
"""

import logging
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gridpulse.models")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 168          # 7-day window for LSTM (hours)
LSTM_LATENT = 64
LSTM_HIDDEN = 128
LSTM_EPOCHS = 30
LSTM_BATCH = 512
LGB_ROUNDS = 500
XGB_ROUNDS = 400
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_parquet(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the features parquet, optionally projecting columns."""
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found at {path}. Run src/features.py first."
        )
    log.info("Loading feature table from %s ...", path)
    import pyarrow.dataset as ds
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    table = dataset.to_table(columns=columns)
    df = table.to_pandas()
    log.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def _lag_cols(prefix: str) -> list[str]:
    return [f"{prefix}_lag_{h}h" for h in range(1, 49)]


def _roll_cols(prefix: str) -> list[str]:
    cols = []
    for w in ("6h", "24h", "7d"):
        cols += [f"{prefix}_roll{w}_mean", f"{prefix}_roll{w}_std"]
    return cols


def _time_cols() -> list[str]:
    return ["hour_of_day", "day_of_week", "month", "week_of_year"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. LightGBM Renewable Forecaster
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_TARGETS = [
    ("solar", 6), ("solar", 12), ("solar", 24),
    ("wind",  6), ("wind",  12), ("wind",  24),
]

LGB_FEATURE_COLS = (
    _lag_cols("solar") + _lag_cols("wind") + _lag_cols("demand")
    + _roll_cols("solar") + _roll_cols("wind") + _roll_cols("demand")
    + _time_cols()
    + ["solar_mw", "wind_mw", "demand_mw"]
)


def train_lgbm() -> dict[str, dict]:
    """Train one LightGBM regressor per (source, horizon) pair."""
    log.info("=" * 60)
    log.info("LightGBM Forecaster — %d models", len(FORECAST_TARGETS))

    need = (
        LGB_FEATURE_COLS
        + [f"solar_t{h}h_ahead" for h in [6, 12, 24]]
        + [f"wind_t{h}h_ahead"  for h in [6, 12, 24]]
        + ["solar_mw", "wind_mw"]   # may overlap, fine
    )
    need = list(dict.fromkeys(need))  # deduplicate while preserving order

    df = _load_parquet(columns=[c for c in need if c in _get_parquet_columns()])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=LGB_FEATURE_COLS[:5])

    metrics_all = {}

    with tqdm(FORECAST_TARGETS, desc="LightGBM models", unit="model", ncols=80) as pbar:
        for source, horizon in pbar:
            target_col = f"{source}_t{horizon}h_ahead"
            pbar.set_postfix(model=f"{source}+{horizon}h")

            if target_col not in df.columns:
                log.warning("Target %s not found, skipping.", target_col)
                continue

            sub = df[LGB_FEATURE_COLS + [target_col]].dropna()
            X = sub[LGB_FEATURE_COLS].values.astype(np.float32)
            y = sub[target_col].values.astype(np.float32)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.15, random_state=RANDOM_SEED
            )
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            params = {
                "objective":      "regression_l1",
                "metric":         "mae",
                "num_leaves":     255,
                "learning_rate":  0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq":   5,
                "n_jobs":         -1,
                "seed":           RANDOM_SEED,
                "verbose":        -1,
            }

            # Tqdm callback that shows epoch-level MAE
            class _TqdmCallback:
                def __init__(self, total, pbar_outer):
                    self._outer = pbar_outer
                    self._inner = tqdm(
                        total=total,
                        desc=f"    {source}+{horizon}h",
                        unit="round",
                        leave=False,
                        ncols=80,
                    )
                    self.order = lgb.callback.CallbackEnv.order if hasattr(
                        lgb.callback.CallbackEnv, "order") else 10

                def __call__(self, env):
                    val_mae = env.evaluation_result_list[-1][2]
                    self._inner.set_postfix(val_mae=f"{val_mae:.2f}")
                    self._inner.update(1)
                    if env.iteration + 1 == env.end_iteration:
                        self._inner.close()

            cb = _TqdmCallback(LGB_ROUNDS, pbar)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=LGB_ROUNDS,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                    cb,
                ],
            )

            y_pred = model.predict(X_val)
            mae  = mean_absolute_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred) ** 0.5
            r2   = r2_score(y_val, y_pred)
            print(f"  [{source}+{horizon}h]  MAE={mae:.2f} MW  RMSE={rmse:.2f} MW  R²={r2:.4f}")

            model_path = MODELS_DIR / f"lgbm_{source}_{horizon}h.txt"
            model.save_model(str(model_path))
            log.info("  Saved → %s", model_path.name)
            metrics_all[f"{source}_{horizon}h"] = {"mae": mae, "rmse": rmse, "r2": r2}

    joblib.dump(metrics_all, MODELS_DIR / "lgbm_metrics.pkl")
    log.info("LightGBM training complete.")
    return metrics_all


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM Autoencoder — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output_proj = nn.Linear(hidden, input_dim)

    def forward(self, x):
        # x: (B, T, F)
        _, (h, _) = self.encoder(x)
        z = self.enc_to_latent(h[-1])                # (B, latent)
        h_dec = self.latent_to_dec(z).unsqueeze(0)   # (1, B, hidden)
        # repeat latent as decoder input across all timesteps
        dec_input = h_dec.permute(1, 0, 2).expand(-1, x.size(1), -1)  # (B, T, hidden)
        out, _ = self.decoder(dec_input, (h_dec, torch.zeros_like(h_dec)))
        return self.output_proj(out)                 # (B, T, F)


def _make_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window of seq_len over the time axis."""
    n = len(arr) - seq_len + 1
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    return arr[idx]


LSTM_FEATURE_COLS = ["solar_mw", "wind_mw", "demand_mw",
                     "net_gen_mw", "interchange_mw"]


def train_lstm() -> None:
    """Train LSTM autoencoder on normal operational data."""
    log.info("=" * 60)
    log.info("LSTM Autoencoder — device: %s", DEVICE)

    df = _load_parquet(columns=LSTM_FEATURE_COLS + ["mismatch_label", "ba", "ts_unix"])
    df = df.sort_values(["ba", "ts_unix"])
    df = df.dropna(subset=LSTM_FEATURE_COLS)

    # Train on "normal" (label 0) to learn typical patterns
    normal = df[df["mismatch_label"] == 0]
    log.info("Normal rows for LSTM training: %d", len(normal))

    scaler = StandardScaler()
    feat_arr = scaler.fit_transform(normal[LSTM_FEATURE_COLS].values.astype(np.float32))

    # Build sequences
    sequences = _make_sequences(feat_arr, SEQ_LEN)
    log.info("Sequence tensor: %s", sequences.shape)

    X_tr, X_val = train_test_split(sequences, test_size=0.1, random_state=RANDOM_SEED)
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    tr_dl  = DataLoader(tr_ds, batch_size=LSTM_BATCH, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=LSTM_BATCH, shuffle=False, num_workers=0)

    model = LSTMAutoencoder(
        input_dim=len(LSTM_FEATURE_COLS),
        hidden=LSTM_HIDDEN,
        latent=LSTM_LATENT,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    log.info("Training for %d epochs ...", LSTM_EPOCHS)
    history = []

    with tqdm(range(1, LSTM_EPOCHS + 1), desc="LSTM epochs", unit="epoch", ncols=80) as epoch_pbar:
        for epoch in epoch_pbar:
            # ── train ──
            model.train()
            tr_loss = 0.0
            with tqdm(tr_dl, desc=f"  Epoch {epoch:02d} train", leave=False, unit="batch", ncols=80) as batch_pbar:
                for (xb,) in batch_pbar:
                    xb = xb.to(DEVICE)
                    optimiser.zero_grad()
                    recon = model(xb)
                    loss = criterion(recon, xb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
                    tr_loss += loss.item()
                    batch_pbar.set_postfix(loss=f"{loss.item():.5f}")

            tr_loss /= len(tr_dl)

            # ── validate ──
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (xb,) in val_dl:
                    xb = xb.to(DEVICE)
                    recon = model(xb)
                    val_loss += criterion(recon, xb).item()
            val_loss /= len(val_dl)

            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
            epoch_pbar.set_postfix(
                train=f"{tr_loss:.5f}", val=f"{val_loss:.5f}"
            )
            print(
                f"  Epoch {epoch:02d}/{LSTM_EPOCHS}  "
                f"train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}"
            )

    # ── compute anomaly scores on full dataset ──────────────────────────────
    log.info("Computing anomaly scores on full dataset ...")
    all_feat = scaler.transform(
        df[LSTM_FEATURE_COLS].fillna(0).values.astype(np.float32)
    )
    all_seqs = _make_sequences(all_feat, SEQ_LEN)
    all_ds = TensorDataset(torch.tensor(all_seqs, dtype=torch.float32))
    all_dl = DataLoader(all_ds, batch_size=LSTM_BATCH, shuffle=False, num_workers=0)

    recon_errors = []
    model.eval()
    with torch.no_grad():
        for (xb,) in tqdm(all_dl, desc="  Scoring", unit="batch", ncols=80):
            xb = xb.to(DEVICE)
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()
            recon_errors.append(err)

    scores = np.concatenate(recon_errors)
    # Align scores with dataframe rows (score assigned to last step in each window)
    score_col = np.full(len(df), np.nan)
    score_col[SEQ_LEN - 1:SEQ_LEN - 1 + len(scores)] = scores
    df = df.copy()
    df["anomaly_score"] = score_col

    # Save artefacts
    torch.save(model.state_dict(), MODELS_DIR / "lstm_autoencoder.pt")
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    joblib.dump(pd.DataFrame(history), MODELS_DIR / "lstm_history.pkl")
    df[["ba", "ts_unix", "anomaly_score"]].dropna().to_parquet(
        MODELS_DIR / "anomaly_scores.parquet", index=False
    )
    log.info("LSTM artefacts saved to %s", MODELS_DIR)
    print(f"\n  LSTM final val_loss: {history[-1]['val_loss']:.5f}")
    print(f"  Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  99th percentile threshold: {np.percentile(scores, 99):.4f}")

    # Save threshold
    threshold = float(np.percentile(scores, 99))
    joblib.dump({"threshold": threshold}, MODELS_DIR / "lstm_threshold.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost Mismatch Severity Classifier
# ─────────────────────────────────────────────────────────────────────────────

XGB_FEATURE_COLS = (
    _lag_cols("solar") + _lag_cols("wind") + _lag_cols("demand")
    + _roll_cols("solar") + _roll_cols("wind") + _roll_cols("demand")
    + _time_cols()
    + ["solar_mw", "wind_mw", "demand_mw", "net_gen_mw",
       "interchange_mw", "mismatch_mw", "mismatch_pct"]
)


def train_xgboost() -> dict:
    """Train XGBoost 4-class mismatch severity classifier with GPU."""
    log.info("=" * 60)
    log.info("XGBoost Mismatch Classifier — device: %s", "cuda" if DEVICE == "cuda" else "cpu")

    need = list(dict.fromkeys(XGB_FEATURE_COLS + ["mismatch_label"]))
    df = _load_parquet(columns=[c for c in need if c in _get_parquet_columns()])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mismatch_label"])

    available_feats = [c for c in XGB_FEATURE_COLS if c in df.columns]
    df_clean = df[available_feats + ["mismatch_label"]].dropna()

    X = df_clean[available_feats].values.astype(np.float32)
    y = df_clean["mismatch_label"].values.astype(np.int32)
    label_counts = np.bincount(y)
    log.info("Class distribution: %s", dict(enumerate(label_counts)))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )

    device_param = "cuda" if DEVICE == "cuda" else "cpu"

    class _XGBTqdmCallback(xgb.callback.TrainingCallback):
        def __init__(self, total):
            self._pbar = tqdm(total=total, desc="  XGBoost rounds", unit="round", ncols=80)

        def after_iteration(self, model, epoch, evals_log):
            mlogloss = list(evals_log.get("validation_0", {}).get("mlogloss", [0]))
            loss_val = mlogloss[-1] if mlogloss else 0.0
            self._pbar.set_postfix(val_mlogloss=f"{loss_val:.4f}", round=epoch + 1)
            self._pbar.update(1)
            if epoch % 50 == 0 and epoch > 0:
                print(f"  Round {epoch:4d}  val_mlogloss={loss_val:.4f}")
            return False

        def after_training(self, model):
            self._pbar.close()
            return model

    clf = xgb.XGBClassifier(
        n_estimators=XGB_ROUNDS,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        device=device_param,
        early_stopping_rounds=40,
        random_state=RANDOM_SEED,
        verbosity=0,
        callbacks=[_XGBTqdmCallback(XGB_ROUNDS)],
    )

    print(f"\n  Training XGBoost ({len(X_tr):,} rows, {len(available_feats)} features) ...")
    log.info("Fitting XGBoost ...")

    t0 = time.time()
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred,
                                   target_names=["balanced", "mod_surplus",
                                                 "sev_surplus", "deficit"])
    print("\n  Classification Report:")
    print(report)

    clf.save_model(str(MODELS_DIR / "xgb_mismatch.json"))
    joblib.dump(available_feats, MODELS_DIR / "xgb_feature_names.pkl")
    log.info("XGBoost model saved → xgb_mismatch.json")

    metrics = {
        "report": report,
        "n_features": len(available_feats),
        "best_iteration": clf.best_iteration,
    }
    joblib.dump(metrics, MODELS_DIR / "xgb_metrics.pkl")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _get_parquet_columns() -> list[str]:
    """Return column names from the features parquet without loading data."""
    import pyarrow.dataset as ds
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        return []
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dataset.schema.names


def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "lgbm"):
        train_lgbm()
    if which in ("all", "lstm"):
        train_lstm()
    if which in ("all", "xgb"):
        train_xgboost()


if __name__ == "__main__":
    main()
