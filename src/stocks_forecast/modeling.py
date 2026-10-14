"""Chronological model comparison and final next-session model training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = logging.getLogger("stocks_forecast.modeling")
TARGET = "target_close_next"
DATE_FIELDS = {"trade_date", "target_date"}


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config, path.parent.parent


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_features(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows or TARGET not in fields:
        raise ValueError(f"Feature table is empty or missing {TARGET}: {path}")
    return rows, fields


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_rub": float(mean_absolute_error(y_true, y_pred)),
        "rmse_rub": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
    }


def model_factory(config: dict[str, Any]) -> dict[str, Any]:
    options = config["modeling"]
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=int(options.get("forest_estimators", 400)),
            min_samples_leaf=int(options.get("forest_min_samples_leaf", 3)),
            max_features=0.8,
            random_state=int(config["project"].get("random_seed", 42)),
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=int(options.get("boosting_iterations", 300)),
            max_depth=2,
            learning_rate=0.03,
            loss="huber",
            random_state=int(config["project"].get("random_seed", 42)),
        ),
    }


def feature_set_candidates(feature_names: list[str]) -> dict[str, list[str]]:
    """Candidate feature families; the winning set is selected on validation only."""
    price_levels = [name for name in feature_names if name.startswith("close_")]
    calendar = [name for name in feature_names if name.startswith(("weekday_", "month_", "day_of_month_"))]
    returns = [name for name in feature_names if name.startswith("return_")]
    volume = [name for name in feature_names if name.startswith("volume_")]
    candidates = {
        "all_features": feature_names,
        "without_calendar": [name for name in feature_names if name not in calendar],
        "without_returns_and_volatility": [name for name in feature_names if name not in returns],
        "without_volume": [name for name in feature_names if name not in volume],
        "price_levels_only": price_levels,
    }
    return {name: columns for name, columns in candidates.items() if columns}


def model_params(model: Any) -> dict[str, Any]:
    selected = {"n_estimators", "min_samples_leaf", "max_features", "max_depth", "learning_rate", "loss", "random_state"}
    return {key: value for key, value in model.get_params(deep=False).items() if key in selected}


def train_and_evaluate(config_path: Path) -> dict[str, Any]:
    config, root = load_config(config_path)
    features_path = resolve_path(root, config["storage"]["processed_dir"]) / "features.csv"
    rows, fields = read_features(features_path)
    # A feed that covers only a handful of recent sessions would make historical
    # validation misleading: nearly all training rows would contain artificial zeros.
    news_sessions = sum(float(row.get("news_count", "0") or 0) > 0 for row in rows)
    news_coverage_pct = 100.0 * news_sessions / len(rows)
    minimum_news_coverage_pct = float(config["modeling"].get("minimum_news_coverage_pct", 20.0))
    news_loaded = news_sessions > 0 and news_coverage_pct >= minimum_news_coverage_pct
    feature_names = [
        field for field in fields
        if field not in DATE_FIELDS | {TARGET}
        and (news_loaded or not field.startswith("news_"))
        and field != "news_source_loaded"
    ]
    if not feature_names:
        raise ValueError("No model features were found")

    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([float(row[TARGET]) for row in rows], dtype=float)
    dates = [row["trade_date"] for row in rows]
    validation_days = int(config["modeling"]["validation_days"])
    test_days = int(config["modeling"]["test_days"])
    validation_start = len(rows) - validation_days - test_days
    test_start = len(rows) - test_days
    if validation_start < 100:
        raise ValueError(
            f"Only {len(rows)} labelled rows; need more than 100 train rows plus "
            f"{validation_days} validation and {test_days} test rows"
        )

    validation_first_date = dates[validation_start]
    test_first_date = dates[test_start]
    train_indices = [i for i in range(validation_start) if rows[i]["target_date"] < validation_first_date]
    train_validation_indices = [i for i in range(test_start) if rows[i]["target_date"] < test_first_date]
    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_start:test_start], y[validation_start:test_start]
    X_train_validation, y_train_validation = X[train_validation_indices], y[train_validation_indices]
    X_test, y_test = X[test_start:], y[test_start:]
    candidates = model_factory(config)
    validation_results: dict[str, dict[str, float]] = {
        "naive_last_close": score(y_validation, X_validation[:, feature_names.index("close_at_forecast")])
    }
    candidate_sets = feature_set_candidates(feature_names)
    selection_details: dict[str, Any] = {
        "naive_last_close": {"selected_set": "close_at_forecast only", "validation_metrics_by_set": {}}
    }
    selected_features_by_model: dict[str, list[str]] = {}
    for name in candidates:
        candidate_metrics: dict[str, dict[str, float]] = {}
        for set_name, selected_features in candidate_sets.items():
            columns = [feature_names.index(feature) for feature in selected_features]
            model = model_factory(config)[name]
            model.fit(X_train[:, columns], y_train)
            candidate_metrics[set_name] = score(y_validation, model.predict(X_validation[:, columns]))
        best_set = min(candidate_metrics, key=lambda key: candidate_metrics[key]["mae_rub"])
        selected_features_by_model[name] = candidate_sets[best_set]
        validation_results[name] = candidate_metrics[best_set]
        selection_details[name] = {
            "selected_set": best_set,
            "selected_features": candidate_sets[best_set],
            "validation_metrics_by_set": candidate_metrics,
        }

    selected_name = min(validation_results, key=lambda name: validation_results[name]["mae_rub"])
    if selected_name == "naive_last_close":
        test_prediction = X_test[:, feature_names.index("close_at_forecast")]
        final_model = None
        selected_feature_names = ["close_at_forecast"]
    else:
        selected_feature_names = selected_features_by_model[selected_name]
        selected_columns = [feature_names.index(feature) for feature in selected_feature_names]
        final_model = model_factory(config)[selected_name]
        final_model.fit(X_train_validation[:, selected_columns], y_train_validation)
        test_prediction = final_model.predict(X_test[:, selected_columns])
    test_results = score(y_test, test_prediction)

    # Save auditable test predictions and lightweight monitoring summaries.
    test_rows = rows[test_start:]
    prediction_rows = []
    for source_row, actual, predicted in zip(test_rows, y_test, test_prediction):
        prediction_rows.append({
            "forecast_date": source_row["trade_date"],
            "target_date": source_row["target_date"],
            "actual_close": float(actual),
            "predicted_close": float(predicted),
            "absolute_error_rub": float(abs(actual - predicted)),
            **{f"feature_{name}": float(source_row[name]) for name in selected_feature_names},
        })
    rolling_window = min(20, len(prediction_rows))
    rolling_mae = [
        float(np.mean([row["absolute_error_rub"] for row in prediction_rows[max(0, i - rolling_window + 1):i + 1]]))
        for i in range(len(prediction_rows))
    ]
    for row, value in zip(prediction_rows, rolling_mae):
        row["rolling_20_session_mae_rub"] = value
    monthly: dict[str, list[float]] = {}
    for row in prediction_rows:
        monthly.setdefault(row["target_date"][:7], []).append(row["absolute_error_rub"])
    monthly_mae = {month: {"rows": len(errors), "mae_rub": float(np.mean(errors))} for month, errors in monthly.items()}
    recent_mae = float(np.mean([row["absolute_error_rub"] for row in prediction_rows[-rolling_window:]]))
    reference_mae = validation_results[selected_name]["mae_rub"]
    drift_ratio = recent_mae / reference_mae if reference_mae else None
    monitoring = {
        "experiment_id": None,  # filled after experiment_id is generated below
        "monitoring_window": {"start": test_rows[0]["target_date"], "end": test_rows[-1]["target_date"], "rows": len(test_rows)},
        "reference_validation_mae_rub": reference_mae,
        "holdout_test_mae_rub": test_results["mae_rub"],
        "recent_window_sessions": rolling_window,
        "recent_window_mae_rub": recent_mae,
        "recent_to_validation_mae_ratio": drift_ratio,
        "warning_threshold_ratio": 1.5,
        "status": "review" if drift_ratio is not None and drift_ratio > 1.5 else "within_reference_threshold",
        "monthly_mae": monthly_mae,
        "interpretation": "Historical holdout monitoring only; alert threshold is a review trigger, not proof of model drift.",
    }

    # Freeze the test result above; train the deployable version using all labelled history.
    deployment_name = selected_name
    deployment_model = None
    if selected_name != "naive_last_close":
        deployment_model = model_factory(config)[selected_name]
        selected_columns = [feature_names.index(feature) for feature in selected_features_by_model[selected_name]]
        deployment_model.fit(X[:, selected_columns], y)

    experiment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_dir = resolve_path(root, config["storage"]["model_dir"])
    reports_dir = resolve_path(root, config["storage"]["reports_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_file = None
    params = {"baseline": "predict the current close", "features": selected_feature_names}
    if deployment_model is not None:
        params = {**model_params(deployment_model), "features": selected_features_by_model[selected_name]}
        model_file = f"sber_{experiment_id[:8]}.joblib"
        joblib.dump(
            {"model": deployment_model, "features": selected_features_by_model[selected_name], "model_name": deployment_name,
             "trained_through": dates[-1], "experiment_id": experiment_id},
            model_dir / model_file,
        )

    result = {
        "experiment_id": experiment_id,
        "created_at_utc": created_at,
        "ticker": config["instrument"]["ticker"],
        "target": "next trading session close (RUB)",
        "feature_source_status": (
            "price/volume/calendar only; news omitted because historical coverage is below threshold"
            if news_sessions and not news_loaded else
            "price/volume/calendar only; no news records found" if not news_sessions else
            "news features included; coverage threshold satisfied"
        ),
        "news_coverage": {
            "sessions_with_news": news_sessions,
            "feature_rows": len(rows),
            "coverage_percent": round(news_coverage_pct, 4),
            "minimum_to_include_percent": minimum_news_coverage_pct,
            "included_in_model": news_loaded,
        },
        "data_rows": len(rows),
        "feature_count": len(selected_feature_names),
        "train_period": {"start": dates[0], "end": rows[train_indices[-1]]["trade_date"], "rows": len(train_indices)},
        "validation_period": {"start": dates[validation_start], "end": dates[test_start - 1], "rows": validation_days},
        "test_period": {"start": dates[test_start], "end": dates[-1], "rows": test_days},
        "validation_metrics": validation_results,
        "selected_model": selected_name,
        "validation_feature_selection": selection_details,
        "untouched_test_metrics": test_results,
        "model_parameters": params,
        "deployment_model_file": model_file,
        "selection_rule": "Lowest validation MAE; test window was not used for selection.",
        "leakage_control": "Training rows whose next-session label falls on or after the validation/test cutoff are purged at split boundaries.",
        "reproducibility": {"random_seed": config["project"].get("random_seed", 42)},
        "interpretability": {
            "selected_model_explanation": "naive_last_close predicts the next close as the latest observed close; only close_at_forecast affects its prediction."
            if selected_name == "naive_last_close" else "See model parameters and feature columns for the selected fitted model.",
            "selected_model_used_feature": "close_at_forecast" if selected_name == "naive_last_close" else None,
        },
    }
    monitoring["experiment_id"] = experiment_id
    result["monitoring"] = monitoring
    report_path = reports_dir / "model_comparison.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    comparison_path = reports_dir / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "validation_mae_rub", "validation_rmse_rub", "validation_mape_percent", "test_mae_rub", "test_rmse_rub", "test_mape_percent", "test_r2"])
        writer.writeheader()
        for name, metrics in validation_results.items():
            test_metrics = test_results if name == selected_name else {}
            writer.writerow({
                "model": name,
                "validation_mae_rub": metrics["mae_rub"],
                "validation_rmse_rub": metrics["rmse_rub"],
                "validation_mape_percent": metrics["mape_percent"],
                "test_mae_rub": test_metrics.get("mae_rub", ""),
                "test_rmse_rub": test_metrics.get("rmse_rub", ""),
                "test_mape_percent": test_metrics.get("mape_percent", ""),
                "test_r2": test_metrics.get("r2", ""),
            })
    prediction_path = reports_dir / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    monitoring_path = reports_dir / "monitoring.json"
    with monitoring_path.open("w", encoding="utf-8") as stream:
        json.dump(monitoring, stream, ensure_ascii=False, indent=2)
    LOGGER.info("Selected %s using validation MAE; untouched test MAE %.2f RUB", selected_name, test_results["mae_rub"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare time-series forecasting models")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        print(json.dumps(train_and_evaluate(args.config), ensure_ascii=False, indent=2))
        return 0
    except Exception:
        LOGGER.exception("Model comparison failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
