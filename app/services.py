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
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import ALL_FEATURES, ORDINAL_ORDER, TARGET  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "rumahku_house_prices.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


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


@functools.lru_cache(maxsize=1)
def load_best_hyperparameters():
    """Hyperparameters found by RandomizedSearchCV/GridSearchCV for the
    winning model (src/train_model.py), keyed by model name."""
    path = os.path.join(MODELS_DIR, "best_hyperparameters.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_valid_categories(df: pd.DataFrame):
    valid_kota = sorted(df["kota"].dropna().unique().tolist())
    valid_tipe = sorted(df["tipe_properti"].dropna().unique().tolist())
    return valid_kota, valid_tipe
