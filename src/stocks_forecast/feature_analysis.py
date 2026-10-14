"""Validation-only feature selection and model interpretation diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from stocks_forecast.modeling import (
    DATE_FIELDS,
    TARGET,
    load_config,
    model_factory,
    read_features,
    resolve_path,
    score,
)

LOGGER = logging.getLogger("stocks_forecast.feature_analysis")


def groups_for(feature_names: list[str]) -> dict[str, list[str]]:
    groups = {
        "price_levels": [name for name in feature_names if name.startswith(("close_",))],
        "returns_and_volatility": [name for name in feature_names if name.startswith(("return_",))],
        "trading_volume": [name for name in feature_names if name.startswith("volume_")],
        "calendar": [name for name in feature_names if name.startswith(("weekday_", "month_", "day_of_month_"))],
    }
    return {name: members for name, members in groups.items() if members}


def run_feature_analysis(config_path: Path) -> dict[str, Any]:
    config, root = load_config(config_path)
    reports_dir = resolve_path(root, config["storage"]["reports_dir"])
    comparison_path = reports_dir / "model_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    features_path = resolve_path(root, config["storage"]["processed_dir"]) / "features.csv"
    rows, fields = read_features(features_path)
    feature_names = [
        field for field in fields
        if field not in DATE_FIELDS | {TARGET}
        and (comparison.get("news_coverage", {}).get("included_in_model", False) or not field.startswith("news_"))
        and field != "news_source_loaded"
    ]
    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([float(row[TARGET]) for row in rows], dtype=float)
    validation_days = int(config["modeling"]["validation_days"])
    test_days = int(config["modeling"]["test_days"])
    validation_start = len(rows) - validation_days - test_days
    test_start = len(rows) - test_days
    if validation_start < 100:
        raise ValueError("Not enough rows for train, validation, and test diagnostics")

    # Purge a training label that crosses the validation boundary.
    validation_first = rows[validation_start]["trade_date"]
    train_idx = [i for i in range(validation_start) if rows[i]["target_date"] < validation_first]
    validation_idx = np.arange(validation_start, test_start)
    test_idx = np.arange(test_start, len(rows))
    groups = groups_for(feature_names)
    group_sets = {
        "all_features": feature_names,
        "without_calendar": [n for n in feature_names if n not in groups.get("calendar", [])],
        "without_returns_and_volatility": [n for n in feature_names if n not in groups.get("returns_and_volatility", [])],
        "without_volume": [n for n in feature_names if n not in groups.get("trading_volume", [])],
        "price_levels_only": groups.get("price_levels", []),
    }
    group_sets = {key: value for key, value in group_sets.items() if value}

    def estimator(name: str):
        if name == "naive_last_close":
            return None
        return model_factory(config)[name]

    # Ablations compare feature families on validation only. Test data is not used
    # to select features or models.
    ablation: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in ["naive_last_close", "linear_regression", "random_forest", "gradient_boosting"]:
        ablation[model_name] = {}
        for group_name, selected in group_sets.items():
            if model_name == "naive_last_close":
                prediction = X[validation_idx, feature_names.index("close_at_forecast")]
            else:
                model = estimator(model_name)
                columns = [feature_names.index(name) for name in selected]
                model.fit(X[train_idx][:, columns], y[train_idx])
                prediction = model.predict(X[validation_idx][:, columns])
            ablation[model_name][group_name] = score(y[validation_idx], prediction)

    # Two independent importance methods for tree models: impurity reduction
    # and permutation. The latter is measured on validation and final holdout.
    importance: dict[str, Any] = {}
    partial_dependence: dict[str, Any] = {}
    for model_name in ["random_forest", "gradient_boosting"]:
        model = estimator(model_name)
        model.fit(X[train_idx], y[train_idx])
        tree_imp = getattr(model, "feature_importances_", None)
        importance[model_name] = {
            "training_fit": "train period only",
            "intrinsic_impurity_importance": [
                {"feature": feature, "importance": float(value)}
                for feature, value in sorted(zip(feature_names, tree_imp), key=lambda pair: pair[1], reverse=True)
            ],
            "permutation_importance": {},
        }
        # One-dimensional partial dependence, using only the validation feature
        # distribution as the reference population.
        partial_dependence[model_name] = {}
        for feature in [name for name in ("return_1d", "close_at_forecast") if name in feature_names]:
            column = feature_names.index(feature)
            grid = np.unique(np.quantile(X[validation_idx, column], np.linspace(0.1, 0.9, 7)))
            rows_for_grid = []
            for value in grid:
                perturbed = X[validation_idx].copy()
                perturbed[:, column] = value
                rows_for_grid.append({"feature_value": float(value), "mean_predicted_close_rub": float(np.mean(model.predict(perturbed)))})
            partial_dependence[model_name][feature] = rows_for_grid
        for period, idx in (("validation", validation_idx), ("test_holdout_diagnostic", test_idx)):
            measured = permutation_importance(
                model,
                X[idx],
                y[idx],
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=int(config["project"].get("random_seed", 42)),
                n_jobs=1,
            )
            importance[model_name]["permutation_importance"][period] = [
                {"feature": feature, "mae_increase_rub": float(mean), "std_rub": float(std)}
                for feature, mean, std in sorted(
                    zip(feature_names, measured.importances_mean, measured.importances_std),
                    key=lambda triple: triple[1], reverse=True,
                )
            ]

    explanation_rule = (
        "The one-step forecast equals the latest available close."
        if comparison["selected_model"] == "naive_last_close"
        else "For an individual forecast, compare each feature at its current value with its training median while holding other features fixed."
    )
    baseline_explanation = {
        "selected_model": comparison["selected_model"],
        "rule": explanation_rule,
        "feature": "close_at_forecast" if comparison["selected_model"] == "naive_last_close" else None,
        "validation_start": rows[validation_start]["trade_date"],
        "validation_end": rows[test_start - 1]["trade_date"],
    }
    result = {
        "experiment_id": comparison["experiment_id"],
        "target": comparison["target"],
        "methodology": {
            "selection_period": "validation only",
            "holdout_use": "post-selection diagnostic of importance stability only; not used to select features or models",
            "importance_methods": ["tree impurity decrease", "permutation increase in MAE"],
            "random_seed": int(config["project"].get("random_seed", 42)),
            "permutation_repeats": 5,
        },
        "feature_groups": groups,
        "validation_ablation": ablation,
        "tree_model_importance": importance,
        "partial_dependence": partial_dependence,
        "selected_forecast_explanation": baseline_explanation,
        "selection_note": "Feature groups are compared on chronological validation data. The untouched test remains reserved for final score reporting; its importance calculation is a separate post-selection diagnostic.",
    }
    (reports_dir / "feature_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote feature analysis to %s", reports_dir / "feature_analysis.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare feature groups and explain tree models")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        print(json.dumps(run_feature_analysis(args.config), ensure_ascii=False, indent=2))
        return 0
    except Exception:
        LOGGER.exception("Feature analysis failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
