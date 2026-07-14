"""Shared data/model loading + business logic for the FastAPI dashboard.

Kept separate from main.py (routes) so the logic can be unit-tested and
reused across route handlers without re-reading files on every request.
"""
import functools
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import (  # noqa: E402
    ALL_FEATURES, BINARY_FEATURES, NOMINAL_FEATURES, NUMERIC_FEATURES,
    ORDINAL_FEATURES, ORDINAL_ORDER, TARGET,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "rumahku_house_prices.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MAX_UPLOAD_ROWS = 5000


@functools.lru_cache(maxsize=1)
def load_data():
    df_raw = pd.read_csv(DATA_PATH)
    df_clean = df_raw.drop_duplicates().drop_duplicates(subset="id_properti").dropna(subset=[TARGET])
    df_clean = df_clean.copy()
    df_clean["harga_per_m2"] = df_clean[TARGET] / df_clean["luas_bangunan"]
    df_clean["rasio_bangunan_tanah"] = df_clean["luas_bangunan"] / df_clean["luas_tanah"]
    return df_raw, df_clean


@functools.lru_cache(maxsize=1)
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_name.json")) as f:
        best_name = json.load(f)["best_model"]
    return model, best_name


@functools.lru_cache(maxsize=1)
def load_metrics():
    return pd.read_csv(os.path.join(MODELS_DIR, "model_comparison_metrics.csv"), index_col=0)


@functools.lru_cache(maxsize=1)
def load_feature_importance():
    return pd.read_csv(os.path.join(MODELS_DIR, "feature_importance_best_model.csv"))


@functools.lru_cache(maxsize=1)
def load_test_predictions():
    return pd.read_csv(os.path.join(MODELS_DIR, "test_predictions.csv"))


def get_valid_categories(df: pd.DataFrame):
    valid_kota = sorted(df["kota"].dropna().unique().tolist())
    valid_tipe = sorted(df["tipe_properti"].dropna().unique().tolist())
    return valid_kota, valid_tipe


def validate_and_clean_batch(df_upload: pd.DataFrame, df_reference: pd.DataFrame):
    """Coerce an uploaded dataframe into a safe input for the model pipeline.

    Returns (cleaned_df, warnings, fatal_errors). If fatal_errors is non-empty,
    cleaned_df is None and prediction must not proceed.
    """
    warning_msgs, fatal_errors = [], []

    if df_upload.empty:
        return None, warning_msgs, ["File kosong — tidak ada baris data."]

    missing_cols = [c for c in ALL_FEATURES if c not in df_upload.columns]
    if missing_cols:
        return None, warning_msgs, [f"Kolom wajib tidak ditemukan di file: {', '.join(missing_cols)}"]

    if len(df_upload) > MAX_UPLOAD_ROWS:
        warning_msgs.append(f"File berisi {len(df_upload):,} baris, hanya {MAX_UPLOAD_ROWS:,} baris "
                             f"pertama yang diproses agar aplikasi tetap responsif.")
        df_upload = df_upload.head(MAX_UPLOAD_ROWS)

    valid_kota, valid_tipe = get_valid_categories(df_reference)
    clean = df_upload[ALL_FEATURES].copy()

    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        before_na = clean[col].isna().sum()
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        new_na = clean[col].isna().sum() - before_na
        if new_na > 0:
            warning_msgs.append(f"{new_na} nilai tidak valid pada kolom '{col}' diisi otomatis "
                                 f"(median/modus data historis).")

    clean["kondisi_bangunan"] = clean["kondisi_bangunan"].where(
        clean["kondisi_bangunan"].isin(ORDINAL_ORDER[0]), np.nan)
    clean["tipe_properti"] = clean["tipe_properti"].where(
        clean["tipe_properti"].isin(valid_tipe), np.nan)
    clean["kota"] = clean["kota"].where(clean["kota"].isin(valid_kota), np.nan)

    n_invalid_cat = clean[ORDINAL_FEATURES + NOMINAL_FEATURES].isna().sum().sum()
    if n_invalid_cat > 0:
        warning_msgs.append(f"{n_invalid_cat} nilai kategori tidak dikenal (kota/tipe/kondisi) "
                             f"diisi otomatis dengan nilai terbanyak dari data historis.")

    return clean, warning_msgs, fatal_errors
