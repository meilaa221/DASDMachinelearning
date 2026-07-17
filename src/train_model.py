"""Train and compare regression models for Rumahku house price prediction.

Pipeline: load -> clean -> split -> preprocess -> feature selection ->
hyperparameter search -> train 4 models -> evaluate -> save best model +
metrics so the FastAPI app never needs to retrain.
"""
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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


TUNING_SPECS = {
    "Linear Regression": dict(
        estimator=LinearRegression(),
        param_grid={"model__fit_intercept": [True, False]},
        search="grid",
    ),
    "Random Forest": dict(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 10, 15, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        search="random",
        n_iter=25,
    ),
    "XGBoost": dict(
        estimator=xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [3, 4, 5, 6, 7],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        search="random",
        n_iter=25,
    ),
    "LightGBM": dict(
        estimator=lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        param_grid={
            "model__n_estimators": [200, 300, 400, 500],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        search="random",
        n_iter=25,
    ),
}


def main():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---- Tahap 1: Seleksi model — 4 model dilatih dengan hyperparameter
    # default/manual (baseline yang adil, sama untuk semua) untuk menentukan
    # algoritma mana yang paling cocok untuk dataset ini. ----
    baseline_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=-1,
        ),
    }

    print("=== Tahap 1: Seleksi Model (hyperparameter default) ===")
    results = {}
    fitted_pipelines = {}
    for name, model in baseline_models.items():
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
    print(f"\nModel terpilih dari seleksi: {best_name} (R2={results[best_name]['R2']:.4f})")

    # ---- Tahap 2: Hyperparameter tuning + feature selection — HANYA untuk
    # model yang menang di Tahap 1, bukan keempatnya, supaya waktu tuning
    # difokuskan ke algoritma yang sudah terbukti paling cocok. ----
    print(f"\n=== Tahap 2: Hyperparameter Tuning & Feature Selection ({best_name}) ===")
    spec = TUNING_SPECS[best_name]
    tuned_pipeline_template = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("model", spec["estimator"]),
    ])

    n_encoded_features = len(build_preprocessor().fit(X_train).get_feature_names_out())
    k_choices = sorted({k for k in (8, 10, 12, 14, n_encoded_features) if k <= n_encoded_features})
    param_grid = {"feature_selection__k": k_choices, **spec["param_grid"]}

    if spec["search"] == "grid":
        search = GridSearchCV(tuned_pipeline_template, param_grid, scoring="r2", cv=5, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            tuned_pipeline_template, param_grid, n_iter=spec["n_iter"], scoring="r2",
            cv=5, n_jobs=-1, random_state=RANDOM_STATE,
        )

    t0 = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - t0

    best_pipeline = search.best_estimator_
    y_pred_tuned = best_pipeline.predict(X_test)
    tuned_metrics = evaluate(y_test, y_pred_tuned)
    tuned_metrics["train_time_sec"] = round(tuning_time, 3)
    tuned_metrics["cv_best_r2"] = round(search.best_score_, 4)

    n_fits = search.n_splits_ * len(search.cv_results_["params"])
    print(f"Selesai: {n_fits} fit ({tuning_time:.1f}s). CV R2 terbaik: {tuned_metrics['cv_best_r2']:.4f}")
    print(f"R2 sebelum tuning : {results[best_name]['R2']:.4f}")
    print(f"R2 sesudah tuning : {tuned_metrics['R2']:.4f}  "
          f"({'+' if tuned_metrics['R2'] >= results[best_name]['R2'] else ''}"
          f"{tuned_metrics['R2'] - results[best_name]['R2']:.4f})")

    best_hyperparams = {
        k: (int(v) if isinstance(v, np.integer) else v)
        for k, v in search.best_params_.items()
    }

    # Metrik model terpilih diganti dengan hasil SESUDAH tuning, supaya tabel
    # perbandingan & halaman /evaluasi di dashboard selalu merefleksikan model
    # yang benar-benar dideploy (best_model.pkl), bukan versi sebelum tuning.
    results[best_name] = tuned_metrics
    fitted_pipelines[best_name] = best_pipeline

    joblib.dump(best_pipeline, f"{MODELS_DIR}/best_model.pkl")
    with open(f"{MODELS_DIR}/best_model_name.json", "w") as f:
        json.dump({"best_model": best_name}, f, indent=2)

    # Hyperparameter terbaik hasil tuning, disimpan untuk transparansi laporan.
    with open(f"{MODELS_DIR}/best_hyperparameters.json", "w") as f:
        json.dump({best_name: best_hyperparams}, f, indent=2)

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
    # models, which don't expose feature_importances_. Filtered through the
    # feature_selection step so names line up 1:1 with the (possibly reduced)
    # set of columns the model was actually trained on.
    best_model_step = best_pipeline.named_steps["model"]
    all_feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_mask = best_pipeline.named_steps["feature_selection"].get_support()
    feature_names = all_feature_names[selected_mask]
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
