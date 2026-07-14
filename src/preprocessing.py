"""Shared preprocessing pipeline for the Rumahku house price model.

Used by both the training script (src/train_model.py) and the Streamlit app
so that raw input data always goes through the exact same cleaning and
encoding logic as during training.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = [
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "jumlah_lantai",
    "usia_bangunan",
    "jarak_pusat_kota",
]
BINARY_FEATURES = ["garasi"]
ORDINAL_FEATURES = ["kondisi_bangunan"]
ORDINAL_ORDER = [["Buruk", "Cukup", "Baik"]]
NOMINAL_FEATURES = ["tipe_properti", "kota"]
TARGET = "harga_jual"

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing target (cannot impute ground truth for
    supervised learning) and remove exact duplicate rows / duplicate ids."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="id_properti")
    df = df.dropna(subset=[TARGET])
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        # Scale first, then impute: KNNImputer measures distance between rows
        # to find the "nearest" neighbors, so features must already be on a
        # comparable scale or a large-magnitude column (e.g. luas_tanah in the
        # hundreds) would dominate the distance calculation. StandardScaler is
        # NaN-tolerant (computes mean/std ignoring NaN, leaves NaN in place),
        # so it's safe to run before the missing values are filled in.
        ("scaler", StandardScaler()),
        ("imputer", KNNImputer(n_neighbors=5)),
    ])
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # unknown_value=-1 lets unseen categories (e.g. typos from an uploaded CSV)
        # pass through as -1 instead of raising, then get re-imputed below.
        ("encoder", OrdinalEncoder(categories=ORDINAL_ORDER, handle_unknown="use_encoded_value",
                                    unknown_value=-1)),
        ("post_impute", SimpleImputer(missing_values=-1, strategy="most_frequent")),
    ])
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("bin", binary_pipeline, BINARY_FEATURES),
        ("ord", ordinal_pipeline, ORDINAL_FEATURES),
        ("nom", nominal_pipeline, NOMINAL_FEATURES),
    ])
    return preprocessor


def split_X_y(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y
