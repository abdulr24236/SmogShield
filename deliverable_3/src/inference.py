"""Real-time inference for the Deliverable 3 Track A app.

Loads the fitted pipelines persisted by ``src.train_pipeline.py`` and exposes a
single ``predict`` call: given today's inputs, return tomorrow's predicted PM2.5
(regression) and AQI category (classification), plus the persistence baseline
(tomorrow = today) for comparison.

If model artifacts are missing, ``load`` raises a clear error pointing the user
at ``python -m src.train_pipeline``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

CLF_PATH = MODELS / "classification_pipeline.joblib"
REG_PATH = MODELS / "regression_pipeline.joblib"
REG_LOW_PATH = MODELS / "regression_q10_pipeline.joblib"
REG_HIGH_PATH = MODELS / "regression_q90_pipeline.joblib"
META_PATH = MODELS / "metadata.json"

# One-at-a-time perturbations used for the per-forecast explanation panel.
SENSITIVITY_INPUTS: tuple[tuple[str, str, float], ...] = (
    # A larger PM2.5 step is used because tree predictions are piecewise
    # constant: a +10 µg/m³ change often stays inside the same leaf and
    # reports a misleading 0.0. +25 crosses splits more reliably.
    ("pm25", "PM2.5 +25 µg/m³", 25.0),
    ("temperature_c", "Temperature +5 °C", 5.0),
    ("humidity_pct", "Humidity +10 points", 10.0),
    ("wind_kmh", "Wind +5 km/h", 5.0),
)

# Cautious, non-medical guidance keyed on the regression-implied category.
GUIDANCE: dict[str, tuple[str, str]] = {
    "Good/Moderate": (
        "Air quality is relatively favourable.",
        "Normal outdoor activity is reasonable for most people.",
    ),
    "Unhealthy for Sensitive": (
        "Sensitive groups should take care.",
        "Children, older adults, and people with asthma or heart conditions may "
        "reduce prolonged or intense outdoor exertion.",
    ),
    "Unhealthy": (
        "Consider reducing outdoor exertion.",
        "Shorten or lower the intensity of outdoor exercise, especially for "
        "children and sensitive groups.",
    ),
    "Very Unhealthy": (
        "Avoid prolonged outdoor activity.",
        "Keep windows closed during peak pollution and postpone outdoor "
        "exercise; follow official advisories.",
    ),
    "Hazardous (301-400)": (
        "Avoid outdoor activity.",
        "Remain indoors where possible and follow official health advisories.",
    ),
    "Hazardous (401-500)": (
        "Avoid outdoor activity.",
        "Remain indoors and follow official emergency guidance.",
    ),
}


def guidance_for(category: str) -> dict[str, str]:
    """Return cautious, non-medical guidance for an AQI category."""
    headline, detail = GUIDANCE.get(
        category,
        (
            "Check official guidance.",
            "Air-quality conditions may be uncertain; follow local advisories.",
        ),
    )
    return {
        "category": category,
        "headline": headline,
        "detail": detail,
        "disclaimer": (
            "Cautious, non-medical guidance. Not a substitute for official "
            "health advice."
        ),
    }

# US EPA PM2.5 breakpoints (2012 version) — the exact table used by
# data_scripts/03_pm25_aqi_and_eda.py to build the AQI labels. Keeping the
# derived-category view on the same thresholds makes it directly comparable
# with the classifier and with the Deliverable 2 targets.
AQI_BREAKPOINTS: tuple[tuple[float, float, int, int], ...] = (
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
)
AQI_NAMES: tuple[str, ...] = (
    "Good",
    "Moderate",
    "Unhealthy for Sensitive",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous (301-400)",
    "Hazardous (401-500)",
)
# The modelling target merges Good and Moderate into one class.
AQI_LABEL_MERGE = {"Good": "Good/Moderate", "Moderate": "Good/Moderate"}


def pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 (µg/m³) to AQI using the project's EPA breakpoints."""
    c = math.floor(max(float(pm25), 0.0) * 10.0) / 10.0
    for low, high, aqi_low, aqi_high in AQI_BREAKPOINTS:
        if c <= high:
            return int(round(aqi_low + (aqi_high - aqi_low) * (c - low) / (high - low)))
    return 500


def aqi_to_category(aqi: float) -> str:
    """Map an AQI value to the project's merged category labels."""
    for (_low, _high, _aqi_low, aqi_high), name in zip(AQI_BREAKPOINTS, AQI_NAMES):
        if aqi <= aqi_high:
            return AQI_LABEL_MERGE.get(name, name)
    return AQI_NAMES[-1]


class SmogShieldInference:
    def __init__(self) -> None:
        self._clf = None
        self._reg = None
        self._reg_low = None
        self._reg_high = None
        self._meta = None

    def _ensure_loaded(self) -> None:
        if self._clf is None or self._reg is None or self._meta is None:
            if not (CLF_PATH.exists() and REG_PATH.exists() and META_PATH.exists()):
                raise FileNotFoundError(
                    "Model artifacts not found. Run: "
                    "python -m src.train_pipeline  (from this folder)"
                )
            self._clf = joblib.load(CLF_PATH)
            self._reg = joblib.load(REG_PATH)
            if REG_LOW_PATH.exists() and REG_HIGH_PATH.exists():
                self._reg_low = joblib.load(REG_LOW_PATH)
                self._reg_high = joblib.load(REG_HIGH_PATH)
            self._meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    @property
    def feature_columns(self) -> list[str]:
        self._ensure_loaded()
        return list(self._meta["feature_columns"])

    @property
    def class_labels(self) -> list[str]:
        self._ensure_loaded()
        return list(self._meta["class_labels"])

    @property
    def n_features(self) -> int:
        self._ensure_loaded()
        return int(self._meta["n_features"])

    @property
    def feature_importance(self) -> dict[str, list[dict[str, object]]]:
        self._ensure_loaded()
        return dict(self._meta.get("feature_importance", {}))

    @property
    def patterns(self) -> dict[str, object]:
        self._ensure_loaded()
        return dict(self._meta.get("patterns", {}))

    @property
    def monthly_median_pm25(self) -> dict[str, float]:
        """Historical PM2.5 median for each calendar month (1-12)."""
        self._ensure_loaded()
        raw = self._meta.get("patterns", {}).get("monthly_median_pm25", {})
        return {str(k): float(v) for k, v in raw.items()}

    @property
    def interval_metadata(self) -> dict[str, object]:
        self._ensure_loaded()
        return dict(self._meta.get("interval", {}))

    def explain(
        self, inputs: dict[str, Any], base_prediction: float
    ) -> list[dict[str, object]]:
        """One-at-a-time sensitivity of the PM2.5 forecast to each input.

        This is a local explanation: it shows how the current forecast would
        move if a single input changed by a fixed, interpretable amount. It is
        model-agnostic and needs no extra dependency.
        """
        self._ensure_loaded()
        rows: list[dict[str, object]] = []
        for key, label, delta in SENSITIVITY_INPUTS:
            value = inputs.get(key)
            if value is None:
                continue
            perturbed = dict(inputs)
            perturbed[key] = float(value) + delta
            changed = float(self._reg.predict(self.build_feature_row(perturbed))[0])
            rows.append(
                {
                    "input": label,
                    "delta_input": delta,
                    "change": round(changed - base_prediction, 2),
                }
            )
        rows.sort(key=lambda row: abs(float(row["change"])), reverse=True)
        return rows

    def build_feature_row(self, inputs: dict[str, Any]) -> pd.DataFrame:
        """Assemble a single-row feature frame.

        Starts from self-contained defaults (medians / modes stored at refit
        time), then maps the app's friendly inputs onto the model's real
        feature names and derives date/pm25-based features. Any key that
        already matches a feature column is applied last as an explicit
        override.
        """
        self._ensure_loaded()
        row: dict[str, Any] = {}
        for c in self._meta["numeric_features"]:
            row[c] = self._meta["numeric_defaults"].get(c, 0.0)
        for c in self._meta["categorical_features"]:
            row[c] = self._meta["categorical_defaults"].get(c, "")

        def setv(col: str, val: Any) -> None:
            if col in row and val is not None:
                row[col] = val

        # --- PM2.5 and its derived steady-state features ---------------------
        pm25 = inputs.get("pm25")
        if pm25 is not None:
            pm25 = float(pm25)
            setv("pm25", pm25)
            setv("pm25_log", math.log1p(max(pm25, 0.0)))
            for c in ("pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3", "pm25_roll7"):
                setv(c, pm25)
            setv("pm25_diff1", 0.0)
            # Keep the numeric AQI feature consistent with the entered PM2.5.
            # The app does not ask for the numeric AQI, so without this the
            # model would always see the training median (165), which can be
            # badly inconsistent on clean or hazardous days.
            setv("aqi", float(pm25_to_aqi(pm25)))

        # --- Current-day AQI category (raw 7-label vocabulary) ---------------
        setv("aqi_category", inputs.get("aqi_category"))

        # --- Weather (mapped to the model's column names) --------------------
        temp = inputs.get("temperature_c")
        if temp is not None:
            setv("temp_mean_c", temp)
            setv("temp_max_c", temp)
            setv("temp_min_c", temp)
            setv("temp_mean_c_lag1", temp)
        humidity = inputs.get("humidity_pct")
        if humidity is not None:
            setv("humidity_mean_pct", humidity)
            setv("humidity_mean_pct_lag1", humidity)
        wind = inputs.get("wind_kmh")
        if wind is not None:
            setv("wind_speed_mean_kmh", wind)
            setv("wind_speed_max_kmh", wind)
            setv("wind_speed_mean_kmh_lag1", wind)

        # --- Date-derived calendar features ----------------------------------
        date_str = inputs.get("date")
        if date_str:
            d = pd.Timestamp(date_str)
            month = int(d.month)
            doy = int(d.timetuple().tm_yday)
            weekday = int(d.weekday())
            setv("month", month)
            setv("day_of_year", doy)
            setv("weekday", weekday)
            setv("is_weekend", int(weekday >= 5))
            setv("month_sin", math.sin(2 * math.pi * month / 12))
            setv("month_cos", math.cos(2 * math.pi * month / 12))
            setv("doy_sin", math.sin(2 * math.pi * doy / 365.25))
            setv("doy_cos", math.cos(2 * math.pi * doy / 365.25))
            setv("weekday_sin", math.sin(2 * math.pi * weekday / 7))
            setv("weekday_cos", math.cos(2 * math.pi * weekday / 7))
            for season in ("spring", "summer", "autumn", "winter"):
                setv(f"season_{season}", 0)
            if month in (12, 1, 2):
                setv("season_winter", 1)
            elif month in (3, 4, 5):
                setv("season_spring", 1)
            elif month in (6, 7, 8):
                setv("season_summer", 1)
            else:
                setv("season_autumn", 1)

        # --- Explicit feature-column overrides (win last) --------------------
        for c, v in inputs.items():
            if c in row and v is not None:
                row[c] = v

        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Predict next-day PM2.5 and AQI category from today's inputs."""
        self._ensure_loaded()
        X = self.build_feature_row(inputs)

        pm25_pred = float(self._reg.predict(X)[0])
        clf_idx = int(self._clf.predict(X)[0])
        category_pred = self.class_labels[clf_idx]

        # Class probabilities let the app show movement even when the
        # arg-max label does not change.
        probabilities: dict[str, float] = {}
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)[0]
            for col, code in enumerate(self._clf.classes_):
                probabilities[self.class_labels[int(code)]] = float(proba[col])

        # Regression-implied category: convert predicted PM2.5 through the same
        # EPA breakpoints used to build the labels. This is monotonic in PM2.5,
        # unlike the six-class classifier.
        implied_aqi = pm25_to_aqi(pm25_pred)
        implied_category = aqi_to_category(implied_aqi)

        # Calibrated central prediction interval (quantile bounds widened by a
        # validation-chosen factor; test coverage is reported, never tuned).
        interval = None
        if self._reg_low is not None and self._reg_high is not None:
            meta_interval = self._meta.get("interval", {})
            scale = float(meta_interval.get("calibration_scale", 1.0))
            low = float(self._reg_low.predict(X)[0])
            high = float(self._reg_high.predict(X)[0])
            low, high = min(low, high), max(low, high)
            low, high = min(low, pm25_pred), max(high, pm25_pred)
            interval = {
                "low": round(max(pm25_pred + scale * (low - pm25_pred), 0.0), 1),
                "high": round(pm25_pred + scale * (high - pm25_pred), 1),
                "level": float(meta_interval.get("level", 0.8)),
            }

        explanation = self.explain(inputs, pm25_pred)
        guidance = guidance_for(implied_category)

        # Persistence baseline: tomorrow = today.
        today_pm25 = inputs.get("pm25")
        today_category = inputs.get("aqi_category")
        persistence = {
            "pm25": float(today_pm25) if today_pm25 is not None else None,
            "aqi_category": str(today_category) if today_category is not None else None,
        }

        return {
            "predicted_next_pm25": round(pm25_pred, 2),
            "predicted_next_aqi_category": category_pred,
            "predicted_aqi": implied_aqi,
            "predicted_category_from_pm25": implied_category,
            "interval": interval,
            "explanation": explanation,
            "guidance": guidance,
            "probabilities": probabilities,
            "persistence": persistence,
            "n_features": int(self._meta["n_features"]),
        }


# Convenience module-level instance.
inference = SmogShieldInference()
