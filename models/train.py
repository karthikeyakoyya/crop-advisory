"""
Model training pipeline for crop yield prediction.

Two phases, both logged to MLflow:
    1. Baseline: PyCaret's regression AutoML compares ~15 algorithms quickly
       so we have an honest reference point before hand-tuning anything.
    2. Production: a tuned XGBoost regressor, plus two auxiliary
       GradientBoostingRegressor quantile models (p10 / p90) so every
       prediction ships with an uncertainty band instead of a bare point
       estimate.

Run:
    python models/train.py
Requires data/processed/features.csv (see feature_engineering.py).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET = "target_yield_kg_per_ha"
CATEGORICAL_COLS = ["district_name", "crop_name", "season_name", "irrigation_source"]
NUMERIC_COLS = [
    "area_hectares", "sowing_month", "cycle_days", "growth_stage_fraction",
    "pre_sowing_rain_30d", "pre_sowing_rain_60d", "pre_sowing_rain_90d",
    "season_total_rainfall_mm", "season_rain_days", "season_avg_tmax_c",
    "season_avg_tmin_c", "season_avg_humidity_pct", "temp_trend_c_per_month",
    "soil_ph", "organic_carbon_pct", "nitrogen_kg_ha", "phosphorus_kg_ha",
    "potassium_kg_ha", "soil_moisture_pct",
]


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run models/feature_engineering.py first")
    df = pd.read_csv(path)
    df = df.dropna(subset=NUMERIC_COLS + [TARGET])
    logger.info("Loaded %d feature rows for training", len(df))
    return df


def run_pycaret_baseline(train_df: pd.DataFrame) -> dict:
    """Fast AutoML sweep to establish an honest baseline before hand-tuning.
    Falls back gracefully (with a clear log message) if PyCaret isn't
    installed, since it's a heavier optional dependency."""
    try:
        from pycaret.regression import compare_models, pull, setup
    except ImportError:
        logger.warning(
            "PyCaret not installed — skipping AutoML baseline. "
            "Install with `pip install pycaret` to enable it."
        )
        return {"skipped": True}

    logger.info("Running PyCaret AutoML baseline sweep")
    setup(
        data=train_df[NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]],
        target=TARGET,
        categorical_features=CATEGORICAL_COLS,
        session_id=RANDOM_SEED,
        verbose=False,
        html=False,
    )
    best = compare_models(n_select=1, verbose=False)
    leaderboard = pull()
    top_row = leaderboard.iloc[0].to_dict()
    logger.info("PyCaret best baseline model: %s (MAE=%.1f, R2=%.3f)",
                leaderboard.index[0], top_row.get("MAE", float("nan")), top_row.get("R2", float("nan")))
    return {
        "skipped": False,
        "best_model_name": str(leaderboard.index[0]),
        "mae": float(top_row.get("MAE", np.nan)),
        "r2": float(top_row.get("R2", np.nan)),
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )


def train_xgboost(X_train, y_train) -> Pipeline:
    """Tuned production model. Grid is intentionally small so this stays
    fast to run and to re-run; expand it once you have compute budget."""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    search = GridSearchCV(
        pipeline, param_grid, cv=4, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("Best XGBoost params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_


def train_quantile_models(X_train, y_train) -> tuple[Pipeline, Pipeline]:
    """p10 / p90 quantile regressors used to build the yield confidence
    interval returned by the API. GradientBoostingRegressor's built-in
    quantile loss is a simple, well-understood choice for this."""
    lower = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            loss="quantile", alpha=0.10, n_estimators=300, max_depth=3,
            learning_rate=0.05, random_state=RANDOM_SEED,
        )),
    ])
    upper = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            loss="quantile", alpha=0.90, n_estimators=300, max_depth=3,
            learning_rate=0.05, random_state=RANDOM_SEED,
        )),
    ])
    lower.fit(X_train, y_train)
    upper.fit(X_train, y_train)
    return lower, upper


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def main() -> None:
    df = load_features()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_automl"):
        baseline_metrics = run_pycaret_baseline(pd.concat([X_train, y_train.rename(TARGET)], axis=1))
        mlflow.log_params({"phase": "baseline", "n_train_rows": len(X_train)})
        if not baseline_metrics.get("skipped"):
            mlflow.log_metrics({"baseline_mae": baseline_metrics["mae"], "baseline_r2": baseline_metrics["r2"]})
            mlflow.log_param("baseline_best_model", baseline_metrics["best_model_name"])

    with mlflow.start_run(run_name="xgboost_production"):
        logger.info("Training tuned XGBoost production model")
        model, best_params = train_xgboost(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        logger.info("XGBoost test metrics: %s", metrics)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        # cloudpickle serialization: MLflow's newer default (skops) rejects
        # XGBoost's Booster type as "untrusted" for a pipeline like ours;
        # cloudpickle is the right choice here since we control both the
        # training and serving environment (see models/predict.py).
        mlflow.sklearn.log_model(
            model, "xgboost_yield_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        logger.info("Training p10/p90 quantile models for confidence intervals")
        lower_model, upper_model = train_quantile_models(X_train, y_train)
        lower_metrics = evaluate(lower_model, X_test, y_test)
        upper_metrics = evaluate(upper_model, X_test, y_test)
        mlflow.log_metrics({f"lower_{k}": v for k, v in lower_metrics.items()})
        mlflow.log_metrics({f"upper_{k}": v for k, v in upper_metrics.items()})

        # coverage check: how often does the true value actually fall in
        # the predicted [p10, p90] band? Logged transparently, not hidden.
        lower_preds = lower_model.predict(X_test)
        upper_preds = upper_model.predict(X_test)
        coverage = float(np.mean((y_test.values >= lower_preds) & (y_test.values <= upper_preds)))
        mlflow.log_metric("interval_coverage_80pct_target", coverage)
        logger.info("Empirical 80%% interval coverage: %.1f%%", coverage * 100)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "xgboost_yield_model.joblib")
        joblib.dump(lower_model, MODELS_DIR / "quantile_lower_model.joblib")
        joblib.dump(upper_model, MODELS_DIR / "quantile_upper_model.joblib")

        metadata = {
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "target": TARGET,
            "test_metrics": metrics,
            "interval_coverage_80pct_target": coverage,
        }
        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
