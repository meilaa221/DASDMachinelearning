"""Train and compare regression models for Rumahku house price prediction.

Pipeline: load -> clean -> split -> preprocess -> train 4 models -> evaluate
-> save best model + metrics so the Streamlit app never needs to retrain.
"""
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from preprocessing import build_preprocessor, clean_data, load_raw_data, split_X_y

# Cosmetic-only: LightGBM's sklearn wrapper warns about feature-name checks
# between fit/predict inside a Pipeline even though results are unaffected.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

DATA_PATH = "../data/rumahku_house_prices.csv"
MODELS_DIR = "../models"
RANDOM_STATE = 42


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["train_time_sec"] = round(train_time, 3)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"{name:20s} -> RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.4f}")

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nModel terbaik: {best_name} (R2={results[best_name]['R2']:.4f})")

    joblib.dump(best_pipeline, f"{MODELS_DIR}/best_model.pkl")
    with open(f"{MODELS_DIR}/best_model_name.json", "w") as f:
        json.dump({"best_model": best_name}, f, indent=2)

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(f"{MODELS_DIR}/model_comparison_metrics.csv")

    # Save test-set predictions for the best model so the dashboard can plot
    # actual vs predicted without retraining.
    y_pred_best = best_pipeline.predict(X_test)
    test_predictions = X_test.copy()
    test_predictions["harga_aktual"] = y_test.values
    test_predictions["harga_prediksi"] = y_pred_best
    test_predictions.to_csv(f"{MODELS_DIR}/test_predictions.csv", index=False)

    with open(f"{MODELS_DIR}/all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Feature importance from the actual winning model (the one deployed to
    # best_model.pkl), not a fixed algorithm, so the analysis always matches
    # what's really running in production. Falls back to |coef_| for linear
    # models, which don't expose feature_importances_.
    best_model_step = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(best_model_step, "feature_importances_"):
        importances = best_model_step.feature_importances_
    else:
        importances = np.abs(best_model_step.coef_)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(f"{MODELS_DIR}/feature_importance_best_model.csv", index=False)

    print("\nModel, metrics, dan feature importance tersimpan di folder models/")


if __name__ == "__main__":
    main()
