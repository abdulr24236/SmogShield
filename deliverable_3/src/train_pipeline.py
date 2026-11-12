"""Refit the frozen SmogShield pipelines and persist artifacts for Track A.

This reproduces the Deliverable 2 modelling contract (same chronological split,
train-only preprocessing, same frozen hyperparameters) and persists the fitted
pipelines to ``models/`` so the Streamlit app can serve real-time inference
without retraining on every launch.

No fitted binaries are committed: this script is the single source that
regenerates them from fixed seeds, matching the Deliverable 2 reproducibility
policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---- make the parent SmogShield `src` importable ---------------------------
ROOT = Path(__file__).resolve().parents[1]
PARENT_SRC = ROOT.parent / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from data import load_config, prepare_model_data  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from .hourly import build_hourly_profile

MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = ROOT.parent / "data" / "processed" / "features_daily.csv"

# 80% central prediction interval (10th / 90th percentile).
INTERVAL_LOW = 0.10
INTERVAL_HIGH = 0.90

# Frozen hyperparameters (Deliverable 2, validation-selected).
CLF_PARAMS = {
    "min_samples_leaf": 10,
    "max_leaf_nodes": 31,
    "max_iter": 100,
    "learning_rate": 0.1,
    "l2_regularization": 0.1,
    # These two match the Deliverable 2 estimator exactly (see
    # src/classification.py: build_and_tune_hgb) and are required to reproduce
    # the reported validation/test scores.
    "class_weight": "balanced",
    "early_stopping": False,
    "random_state": 42,
}
REG_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_iter": 200,
    "random_state": 42,
}


def _scorable_mask(frame: pd.DataFrame, col: str = "pm25") -> pd.Series:
    """Rows where today's PM2.5 is finite (same mask every method scores on)."""
    return pd.Series(
        np.isfinite(pd.to_numeric(frame[col], errors="coerce")),
        index=frame.index,
        name="scorable",
    )


def _interval_bounds(
    point_pipe, low_pipe, high_pipe, X: pd.DataFrame, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated interval bounds around the point forecast."""
    point = point_pipe.predict(X)
    low = np.minimum(low_pipe.predict(X), high_pipe.predict(X))
    high = np.maximum(low_pipe.predict(X), high_pipe.predict(X))
    # Keep the point forecast inside its own interval even when the quantile
    # models cross, then widen by the validation-calibrated scale.
    low = np.minimum(low, point)
    high = np.maximum(high, point)
    lower = point + scale * (low - point)
    upper = point + scale * (high - point)
    return np.maximum(lower, 0.0), upper


def _interval_coverage(
    point_pipe, low_pipe, high_pipe, X: pd.DataFrame, y: pd.Series, scale: float
) -> float:
    """Empirical share of outcomes inside the calibrated central interval."""
    lower, upper = _interval_bounds(point_pipe, low_pipe, high_pipe, X, scale)
    return float(np.mean((y.to_numpy() >= lower) & (y.to_numpy() <= upper)))


def _calibrate_interval_scale(
    point_pipe, low_pipe, high_pipe, X: pd.DataFrame, y: pd.Series
) -> float:
    """Smallest widening factor that reaches the nominal 80% coverage.

    Quantile HGB intervals under-cover on this small dataset, so the widths are
    scaled using validation outcomes only. The test split is never used to pick
    the factor; it is only used to report coverage afterwards.
    """
    for scale in np.arange(0.80, 3.01, 0.05):
        coverage = _interval_coverage(
            point_pipe, low_pipe, high_pipe, X, y, float(scale)
        )
        if coverage >= 0.80:
            return float(round(scale, 2))
    return 3.0


def _top_importance(
    pipe, X: pd.DataFrame, y: pd.Series, scoring: str, n_repeats: int = 5, top: int = 15
) -> list[dict[str, object]]:
    """Validation-set permutation importance, sorted descending."""
    result = permutation_importance(
        pipe,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=1,
    )
    pairs = sorted(
        zip(X.columns, result.importances_mean), key=lambda item: item[1], reverse=True
    )
    return [
        {"feature": str(feature), "importance": float(score)}
        for feature, score in pairs[:top]
    ]


def _pattern_insights() -> dict[str, object]:
    """Seasonal and winter-stagnation summary from the feature table.

    A winter (Dec-Feb) day is called stagnant when its mean wind speed sits in
    the winter lower quartile and it recorded at most 0.5 mm of rain. This is a
    descriptive proxy, not a physical dispersion model.
    """
    if not FEATURES_CSV.exists():
        return {}
    frame = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    frame = frame[frame["pm25"].notna()].copy()
    if frame.empty:
        return {}

    seasonal = frame.groupby("season")["pm25"].mean().to_dict()

    # Historical monthly medians power the app's seasonally plausible defaults.
    # Use only physically valid PM2.5 values so a sensor spike cannot set the
    # default for an entire month.
    monthly = (
        frame.loc[frame["pm25"].between(0, 500), ["month", "pm25"]]
        .groupby("month")["pm25"]
        .median()
        .round(1)
    )
    monthly_median_pm25 = {str(int(k)): float(v) for k, v in monthly.items()}

    winter = frame[frame["month"].isin([12, 1, 2])].copy()
    winter = winter[np.isfinite(pd.to_numeric(winter["wind_speed_mean_kmh"], errors="coerce"))]
    if winter.empty:
        return {
            "seasonal_mean_pm25": {k: float(v) for k, v in seasonal.items()},
            "monthly_median_pm25": monthly_median_pm25,
        }

    wind_q25 = float(winter["wind_speed_mean_kmh"].quantile(0.25))
    rain = winter["precip_sum_mm"].fillna(0.0)
    stagnant_mask = (winter["wind_speed_mean_kmh"] <= wind_q25) & (rain <= 0.5)
    stagnant = winter[stagnant_mask]
    ventilated = winter[~stagnant_mask]

    stagnant_mean = float(stagnant["pm25"].mean()) if not stagnant.empty else None
    ventilated_mean = float(ventilated["pm25"].mean()) if not ventilated.empty else None
    difference_pct = None
    if stagnant_mean is not None and ventilated_mean not in (None, 0.0):
        difference_pct = (stagnant_mean - ventilated_mean) / ventilated_mean * 100.0

    return {
        "definition": (
            "Winter (Dec-Feb) days in the lowest winter wind quartile with "
            "at most 0.5 mm of rain"
        ),
        "winter_wind_q25_kmh": round(wind_q25, 2),
        "winter_days": int(len(winter)),
        "stagnant_days": int(len(stagnant)),
        "stagnant_share_pct": round(len(stagnant) / len(winter) * 100.0, 1),
        "stagnant_mean_pm25": round(stagnant_mean, 2) if stagnant_mean is not None else None,
        "ventilated_mean_pm25": round(ventilated_mean, 2) if ventilated_mean is not None else None,
        "difference_pct": round(difference_pct, 1) if difference_pct is not None else None,
        "seasonal_mean_pm25": {str(k): float(v) for k, v in seasonal.items()},
        "monthly_median_pm25": monthly_median_pm25,
    }


def main() -> None:
    config = load_config()

    # --- Classification pipeline (six-class next-day AQI category) ----------
    # The scorable mask defines evaluation rows only (Deliverable 2 contract),
    # so training uses the full chronological training split.
    clf_data = prepare_model_data(config, "classification")
    clf_pipe = clf_data.make_model_pipeline(
        HistGradientBoostingClassifier(**CLF_PARAMS)
    )
    clf_pipe.fit(clf_data.X_train_raw, clf_data.y_train)
    joblib.dump(clf_pipe, MODELS / "classification_pipeline.joblib")

    # --- Regression pipeline (next-day PM2.5) -------------------------------
    reg_data = prepare_model_data(config, "regression")
    reg_pipe = reg_data.make_model_pipeline(
        HistGradientBoostingRegressor(**REG_PARAMS)
    )
    reg_pipe.fit(reg_data.X_train_raw, reg_data.y_train)
    joblib.dump(reg_pipe, MODELS / "regression_pipeline.joblib")

    # --- Quantile models for the central 80% prediction interval ------------
    reg_low = reg_data.make_model_pipeline(
        HistGradientBoostingRegressor(
            loss="quantile", quantile=INTERVAL_LOW, **REG_PARAMS
        )
    )
    reg_high = reg_data.make_model_pipeline(
        HistGradientBoostingRegressor(
            loss="quantile", quantile=INTERVAL_HIGH, **REG_PARAMS
        )
    )
    reg_low.fit(reg_data.X_train_raw, reg_data.y_train)
    reg_high.fit(reg_data.X_train_raw, reg_data.y_train)
    joblib.dump(reg_low, MODELS / "regression_q10_pipeline.joblib")
    joblib.dump(reg_high, MODELS / "regression_q90_pipeline.joblib")

    val_mask = _scorable_mask(reg_data.X_validation_raw)
    test_mask = _scorable_mask(reg_data.X_test_raw)
    interval_scale = _calibrate_interval_scale(
        reg_pipe,
        reg_low,
        reg_high,
        reg_data.X_validation_raw[val_mask],
        reg_data.y_validation[val_mask],
    )
    coverage_validation = _interval_coverage(
        reg_pipe,
        reg_low,
        reg_high,
        reg_data.X_validation_raw[val_mask],
        reg_data.y_validation[val_mask],
        interval_scale,
    )
    coverage_test = _interval_coverage(
        reg_pipe,
        reg_low,
        reg_high,
        reg_data.X_test_raw[test_mask],
        reg_data.y_test[test_mask],
        interval_scale,
    )

    # --- Global feature importance on validation (for the app/report) -------
    clf_val_mask = _scorable_mask(clf_data.X_validation_raw)
    importance_regression = _top_importance(
        reg_pipe,
        reg_data.X_validation_raw[val_mask],
        reg_data.y_validation[val_mask],
        scoring="neg_root_mean_squared_error",
    )
    importance_classification = _top_importance(
        clf_pipe,
        clf_data.X_validation_raw[clf_val_mask],
        clf_data.y_validation[clf_val_mask],
        scoring="f1_macro",
        n_repeats=3,
    )
    patterns = _pattern_insights()
    hourly_profile = build_hourly_profile()
    if hourly_profile:
        patterns["hourly"] = hourly_profile

    # --- Metadata + defaults for the inference/app layer --------------------
    # Self-contained defaults so inference does not need the raw feature table
    # at prediction time (e.g. on a hosted demo).
    numeric_defaults = {
        c: float(clf_data.X_train_raw[c].median()) for c in clf_data.numeric_features
    }
    categorical_defaults = {
        c: clf_data.X_train_raw[c].astype(str).mode().iloc[0]
        for c in clf_data.categorical_features
    }
    meta = {
        "feature_columns": list(clf_data.X_train_raw.columns),
        "class_labels": list(clf_data.labels),
        "numeric_features": clf_data.numeric_features,
        "categorical_features": clf_data.categorical_features,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "n_features": int(len(clf_data.feature_names)),
        "interval": {
            "level": round(INTERVAL_HIGH - INTERVAL_LOW, 2),
            "quantiles": [INTERVAL_LOW, INTERVAL_HIGH],
            "calibration_scale": interval_scale,
            "coverage_validation": round(coverage_validation, 4),
            "coverage_test": round(coverage_test, 4),
            "note": (
                "Quantile HGB bounds widened by a factor chosen on the "
                "validation split to reach 80% empirical coverage; test "
                "coverage is reported only, never used for tuning."
            ),
        },
        "feature_importance": {
            "regression": importance_regression,
            "classification": importance_classification,
            "note": (
                "Permutation importance on the validation split. Regression "
                "scores are the increase in RMSE when a feature is shuffled; "
                "classification scores are the drop in macro F1."
            ),
        },
        "patterns": patterns,
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("Deliverable 3 Track A: model artifacts written to", MODELS)
    print("  classification_pipeline.joblib  (HGB, frozen hyperparameters)")
    print("  regression_pipeline.joblib      (HGB, frozen hyperparameters)")
    print("  regression_q10_pipeline.joblib  (80% interval, lower)")
    print("  regression_q90_pipeline.joblib  (80% interval, upper)")
    print("  metadata.json")
    print(
        "  interval calibration scale: "
        f"{interval_scale:.2f}  |  coverage: "
        f"validation {coverage_validation:.1%}, test {coverage_test:.1%}"
    )


if __name__ == "__main__":
    main()
