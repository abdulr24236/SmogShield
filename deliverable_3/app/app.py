"""SmogShield Lahore — Deliverable 3 Track A: next-day air-quality forecast app.

Streamlit frontend that serves real-time next-day PM2.5 and AQI-category
forecasts from the frozen HGB pipelines persisted by ``src.train_pipeline.py``.

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the package root (this folder) is on the path so `src` imports work
# regardless of how Streamlit is invoked.
HERE = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.inference import aqi_to_category, inference, pm25_to_aqi  # noqa: E402

st.set_page_config(
    page_title="SmogShield Lahore — Next-Day Forecast",
    page_icon="🌫️",
    layout="centered",
)

st.title("🌫️ SmogShield Lahore")
st.caption(
    "Deliverable 3 · Track A (Application) — real-time next-day air-quality forecast. "
    "Models are refit from fixed seeds by ``src/train_pipeline.py``; no fitted "
    "binaries are committed."
)

# ---------------------------------------------------------------------------
# Sidebar: model management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    if st.button("Refit / persist model artifacts"):
        with st.spinner("Refitting pipelines from committed scripts..."):
            proc = subprocess.run(
                [sys.executable, "-m", "src.train_pipeline"],
                capture_output=True,
                text=True,
                cwd=str(HERE),
            )
        if proc.returncode == 0:
            st.success("Artifacts refreshed.")
            st.code(proc.stdout[-1500:])
        else:
            st.error("Refit failed.")
            st.code(proc.stderr[-2000:])
    st.divider()
    st.markdown(
        "Artifacts live in `models/` (not committed). "
        "First run `python -m src.train_pipeline` if predictions are unavailable."
    )

# ---------------------------------------------------------------------------
# Input form: today's values
# ---------------------------------------------------------------------------
st.subheader("Enter today's conditions")

# Seasonally plausible defaults: historical monthly medians from metadata.
# A fixed 60 µg/m³ is unrealistically clean for November in Lahore; using the
# month's historical median avoids opening the demo on a misleading scenario.
today = date.today()
try:
    monthly_medians = inference.monthly_median_pm25
except Exception:
    monthly_medians = {}
default_pm25 = float(monthly_medians.get(str(today.month), 60.0))
default_category = aqi_to_category(pm25_to_aqi(default_pm25))
if default_category == "Good/Moderate":
    # The model's merged target label is not one of the raw category inputs.
    default_category = "Moderate"

CATEGORY_OPTIONS = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous (301-400)",
    "Hazardous (401-500)",
]
category_index = (
    CATEGORY_OPTIONS.index(default_category)
    if default_category in CATEGORY_OPTIONS
    else 3
)

st.caption(
    f"Initial PM2.5 and category defaults use the historical median for "
    f"{today:%B} ({default_pm25:.1f} µg/m³, {default_category}); adjust them "
    "to your local reading."
)

col1, col2 = st.columns(2)
with col1:
    forecast_date = st.date_input(
        "Today's date (feature day)",
        value=today,
        help=(
            "Calendar features are taken from this date. The forecast is always "
            "for the following day, not for this date."
        ),
    )
    pm25 = st.number_input(
        "Today's PM2.5 (µg/m³)", min_value=0.0, value=default_pm25, step=1.0
    )
    temperature = st.number_input("Temperature (°C)", value=22.0, step=0.5)
with col2:
    aqi_category = st.selectbox(
        "Today's AQI category",
        CATEGORY_OPTIONS,
        index=category_index,
    )
    humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, value=7.0, step=0.5)

st.caption(
    "Features not collected here use training-set defaults. PM2.5 lags and "
    "rolling windows are treated as a steady state equal to today's value, so "
    "this is an illustrative single-day forecast rather than a production input."
)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Forecast next day", type="primary"):
    try:
        inputs = {
            "pm25": pm25,
            "aqi_category": aqi_category,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_kmh": wind_speed,
            "date": str(forecast_date),
        }
        result = inference.predict(inputs)

        st.divider()
        target_date = forecast_date + timedelta(days=1)
        st.subheader(f"Next-day forecast for {target_date:%d %b %Y}")

        p = result["persistence"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted PM2.5", f"{result['predicted_next_pm25']:.1f} µg/m³")
        with m2:
            st.metric("AQI implied by PM2.5", result["predicted_aqi"])
        with m3:
            st.metric("Implied AQI category", result["predicted_category_from_pm25"])

        n1, n2, n3 = st.columns(3)
        with n1:
            st.metric("Classifier category", result["predicted_next_aqi_category"])
        with n2:
            probs = result.get("probabilities") or {}
            conf = max(probs.values()) if probs else None
            st.metric("Classifier confidence", f"{conf:.0%}" if conf is not None else "–")
        with n3:
            st.metric(
                "Persistence (tomorrow = today)",
                f"{p['pm25']:.1f} µg/m³" if p["pm25"] is not None else "–",
            )

        st.caption(
            "**Implied category** converts the regression's predicted PM2.5 with "
            "the same US-EPA breakpoints used to build the training labels, so it "
            "moves monotonically. The **classifier** is a separate six-class model "
            "and is biased toward `Unhealthy`."
        )

        interval = result.get("interval")
        if interval:
            coverage = inference.interval_metadata
            coverage_text = ""
            if coverage.get("coverage_test") is not None:
                coverage_text = f" (held-out test coverage {coverage['coverage_test']:.0%})"
            st.caption(
                f"**{int(interval['level'] * 100)}% prediction interval:** "
                f"{interval['low']:.1f} – {interval['high']:.1f} µg/m³"
                f"{coverage_text}. Quantile bounds calibrated on validation only."
            )

        guidance = result.get("guidance")
        if guidance:
            st.success(f"**{guidance['headline']}** {guidance['detail']}")
            st.caption(
                f"{guidance['disclaimer']} Follows the implied AQI category."
            )

        explanation = result.get("explanation") or []
        if explanation:
            st.markdown("#### Why this forecast?")
            st.caption(
                "One-at-a-time sensitivity: the change in the PM2.5 forecast "
                "when a single input is increased by a fixed amount."
            )
            explanation_df = pd.DataFrame(
                {
                    "Change in forecast (µg/m³)": [
                        row["change"] for row in explanation
                    ]
                },
                index=[row["input"] for row in explanation],
            )
            st.bar_chart(explanation_df, horizontal=True, height=230)

        importance = inference.feature_importance.get("regression") or []
        if importance:
            with st.expander("What drives forecasts overall (permutation importance)"):
                st.caption(
                    "Increase in validation RMSE when a feature is shuffled; "
                    "higher means the model relies on it more."
                )
                importance_df = pd.DataFrame(
                    {
                        "Increase in RMSE": [
                            row["importance"] for row in importance[:10]
                        ]
                    },
                    index=[row["feature"] for row in importance[:10]],
                )
                st.bar_chart(importance_df, horizontal=True, height=320)

        patterns = inference.patterns
        if patterns and patterns.get("stagnant_mean_pm25") is not None:
            with st.expander("Winter stagnation pattern (historical)"):
                st.markdown(
                    f"**{patterns['stagnant_share_pct']}%** of winter days "
                    f"({patterns['stagnant_days']} of {patterns['winter_days']}) "
                    f"were stagnant — {patterns['definition'].lower()}. "
                    f"Mean PM2.5 on those days was "
                    f"**{patterns['stagnant_mean_pm25']:.0f} µg/m³** versus "
                    f"**{patterns['ventilated_mean_pm25']:.0f} µg/m³** on "
                    f"better-ventilated days ({patterns['difference_pct']:+.0f}%)."
                )
                seasonal = patterns.get("seasonal_mean_pm25") or {}
                order = ["winter", "spring", "summer", "autumn"]
                names = [season for season in order if season in seasonal]
                if names:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Mean PM2.5 (µg/m³)": [
                                    round(seasonal[name], 1) for name in names
                                ]
                            },
                            index=[name.title() for name in names],
                        )
                    )

        hourly = patterns.get("hourly") if patterns else None
        if hourly:
            with st.expander("Best time outdoors (historical)"):
                window = hourly["cleanest_window"]
                peak = hourly["peak_hour"]
                st.markdown(
                    f"Historically, the cleanest three-hour window is "
                    f"**{window['start_hour']:02d}:00–{window['end_hour']:02d}:59**, "
                    f"averaging **{window['mean_pm25']:.0f} µg/m³** versus "
                    f"**{peak['mean_pm25']:.0f} µg/m³** at the worst hour "
                    f"({peak['hour']:02d}:00)."
                )
                winter_window = (hourly.get("seasonal_cleanest_window") or {}).get(
                    "winter"
                )
                if winter_window:
                    st.markdown(
                        f"Even the best winter window averages "
                        f"**{winter_window['mean_pm25']:.0f} µg/m³**, so timing "
                        "helps but does not make winter air safe."
                    )
                by_hour = hourly["by_hour"]
                hourly_df = pd.DataFrame(
                    {
                        "Mean PM2.5 (µg/m³)": [
                            by_hour[str(hour)]["mean"] for hour in range(24)
                        ]
                    },
                    index=[f"{hour:02d}:00" for hour in range(24)],
                )
                st.bar_chart(hourly_df, height=260)
                st.caption(
                    f"{hourly['caveat']} Source: {hourly['source']} "
                    f"({hourly['valid_hours']:,} valid hours, "
                    f"{hourly['date_min']} to {hourly['date_max']})."
                )

        if result["predicted_category_from_pm25"] != result["predicted_next_aqi_category"]:
            st.warning(
                f"The two views disagree: predicted PM2.5 implies "
                f"**{result['predicted_category_from_pm25']}**, while the classifier "
                f"returns **{result['predicted_next_aqi_category']}**. In the D2 "
                "evaluation the classifier was weaker than persistence, so the "
                "implied category is the more defensible of the two."
            )

        if p["pm25"] is not None:
            delta = result["predicted_next_pm25"] - p["pm25"]
            st.markdown(
                f"**Change vs persistence:** {delta:+.1f} µg/m³  ·  "
                f"**Persistence AQI category:** {p['aqi_category'] or '–'}  ·  "
                f"**Features used:** {len(inference.feature_columns)} raw "
                f"({result['n_features']} encoded)"
            )

        probs = result.get("probabilities") or {}
        if probs:
            st.markdown(
                "**Category probabilities** — the top label can stay the same "
                "while these bars still move, because the classifier is biased "
                "toward the majority class (`Unhealthy`)."
            )
            prob_df = pd.DataFrame(
                {"Probability": [probs.get(label, 0.0) for label in inference.class_labels]},
                index=inference.class_labels,
            )
            st.bar_chart(prob_df, horizontal=True, height=260)

        st.info(
            "**What is the persistence baseline?** It assumes tomorrow repeats "
            "today exactly (same PM2.5, same category). It is the simplest "
            "possible forecast, so any model has to beat it to be useful. "
            "In the Deliverable 2 evaluation the frozen HGB did **not** beat "
            "persistence on the held-out test set (RMSE 64.92 vs 64.82 µg/m³; "
            "macro-F1 0.478 vs 0.516), so treat this app as an illustrative "
            "demonstration rather than a proven improvement."
        )

    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
