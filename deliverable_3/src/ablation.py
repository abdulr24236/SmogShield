"""Weather/calendar ablation for Deliverable 3 (Phase 5/6).

Compares three feature sets with the frozen Deliverable 2 hyperparameters:

    history          PM2.5 history, day-quality flags, today's AQI/category
    history+calendar adds calendar, season and holiday features
                     (the proxies used because no Lahore traffic data exists)
    full             adds the weather block

Both tasks are refit from scratch for every feature set, always using the
shared chronological split and train-only preprocessing. Test metrics are
reported but never used to choose a feature set: this is a descriptive study,
not a model-selection step.

Run from ``deliverable_3``:

    python -m src.ablation

Writes ``models/ablation.json`` and prints a table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from .train_pipeline import CLF_PARAMS, REG_PARAMS, _scorable_mask

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

CATEGORICAL = {"aqi_category"}

HISTORY = {
    "pm25",
    "valid_hours",
    "usable_day",
    "aqi",
    "aqi_category",
    "pm25_log",
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_roll3",
    "pm25_roll7",
    "pm25_diff1",
}

CALENDAR = {
    "month",
    "day_of_year",
    "weekday",
    "is_weekend",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "weekday_sin",
    "weekday_cos",
    "is_ramadan",
    "is_eid",
    "is_public_holiday",
    "season_autumn",
    "season_spring",
    "season_summer",
    "season_winter",
}

WEATHER_PREFIXES = (
    "temp_",
    "humidity_",
    "dew_point_",
    "pressure_",
    "wind_",
    "precip_",
    "rain_",
    "shortwave_",
    "cloud_cover_",
)

FEATURE_SETS = {
    "history": HISTORY,
    "history+calendar": HISTORY | CALENDAR,
}


def weather_columns(all_columns: list[str]) -> set[str]:
    return {c for c in all_columns if c.startswith(WEATHER_PREFIXES)}


def check_partition(all_columns: list[str]) -> None:
    """Fail loudly if the hand-written groups no longer cover the table."""
    weather = weather_columns(all_columns)
    covered = HISTORY | CALENDAR | weather
    missing = set(all_columns) - covered
    extra = covered - set(all_columns)
    if missing or extra:
        raise ValueError(
            f"Feature groups out of date. Missing: {sorted(missing)}; "
            f"unknown: {sorted(extra)}"
        )
    if HISTORY & CALENDAR or HISTORY & weather or CALENDAR & weather:
        raise ValueError("Feature groups overlap.")


def fit_and_score(data, task: str, columns: list[str]) -> dict[str, object]:
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    estimator = (
        HistGradientBoostingClassifier(**CLF_PARAMS)
        if task == "classification"
        else HistGradientBoostingRegressor(**REG_PARAMS)
    )
    pipe = Pipeline(
        [
            ("preprocessor", _make_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )

    # Match the Deliverable 2 contract: the scorable mask is for evaluation
    # only, so the model trains on the full chronological training split.
    pipe.fit(data.X_train_raw[columns], data.y_train)

    scores: dict[str, object] = {}
    for split in ("validation", "test"):
        frame = getattr(data, f"X_{split}_raw")[columns]
        target = getattr(data, f"y_{split}")
        mask = _scorable_mask(getattr(data, f"X_{split}_raw"))
        prediction = pipe.predict(frame[mask])
        if task == "classification":
            scores[split] = {
                "macro_f1": round(
                    float(f1_score(target[mask], prediction, average="macro")), 4
                ),
                "n": int(mask.sum()),
            }
        else:
            scores[split] = {
                "rmse": round(
                    float(mean_squared_error(target[mask], prediction) ** 0.5), 4
                ),
                "mae": round(
                    float(mean_absolute_error(target[mask], prediction)), 4
                ),
                "n": int(mask.sum()),
            }
    return {"n_features": len(columns), "metrics": scores}


def _make_preprocessor(numeric: list[str], categorical: list[str]):
    """Import the shared preprocessor lazily so the parent path is set first."""
    import sys

    parent_src = ROOT.parent / "src"
    if str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
    from data import make_preprocessor

    return make_preprocessor(numeric, categorical)


def main() -> None:
    import sys

    parent_src = ROOT.parent / "src"
    if str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
    from data import load_config, prepare_model_data

    config = load_config()
    results: dict[str, object] = {}

    for task in ("regression", "classification"):
        data = prepare_model_data(config, task)
        all_columns = list(data.X_train_raw.columns)
        check_partition(all_columns)
        weather = weather_columns(all_columns)
        feature_sets = {
            **FEATURE_SETS,
            "full": set(all_columns),
        }
        task_results: dict[str, object] = {}
        for name, columns in feature_sets.items():
            ordered = [c for c in all_columns if c in columns]
            print(f"[{task}] fitting {name} ({len(ordered)} features)...")
            task_results[name] = fit_and_score(data, task, ordered)
        results[task] = task_results

    MODELS.mkdir(parents=True, exist_ok=True)
    out = MODELS / "ablation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("Feature-set ablation (frozen hyperparameters)")
    print(f"{'task':16s} {'feature set':18s} {'n':>3s} {'val':>10s} {'test':>10s}")
    for task, task_results in results.items():
        for name, entry in task_results.items():
            val, test = entry["metrics"]["validation"], entry["metrics"]["test"]
            key = "macro_f1" if task == "classification" else "rmse"
            print(
                f"{task:16s} {name:18s} {entry['n_features']:3d} "
                f"{val[key]:10.4f} {test[key]:10.4f}"
            )
    print()
    print("Wrote", out)


if __name__ == "__main__":
    main()
