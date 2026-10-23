#!/usr/bin/env python3
"""
09_build_features.py -- SmogShield Lahore, Deliverable 1, Phase 2 (Feature Strategy)

Builds the daily modelling table and two EDA figures:

    processed/features_daily.csv
    outputs/eda_charts/correlation_heatmap.png
    outputs/eda_charts/weather_vs_pm25.png

The script demonstrates the Phase 2 feature strategy:
  * calendar features (raw + cyclical encoding),
  * lag and rolling features for PM2.5 and weather,
  * log transform for the skewed PM2.5 target,
  * robust scaling (median / IQR) for meteorological variables. The statistics are computed
    on ALL rows here, so they must be re-fitted on the training rows only when modelling,
  * one-hot encoding of season.

No modelling is done here. Predictors only use information available on the forecast day
(lags, rolling means of past days, same-day weather); the targets are next-day values.
Holidays are a simplified list (fixed dates + Eid); religious dates may be off by one day.

Inputs
------
    processed/pm25_daily_clean.csv   (made by 02_clean_pm25_simple.py and 03_pm25_aqi_and_eda.py)
    raw/weather/open_meteo_era5_lahore_daily_2019_2026.csv

Run
---
    python data_scripts/09_build_features.py

Requires pandas, numpy and matplotlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
CHARTS = ROOT / "outputs" / "eda_charts"   # small PNGs are allowed in the repo
CHARTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Calendar proxies (approximate; moon-sighting can shift a date by +/- 1 day)
# --------------------------------------------------------------------------- #
# Ramadan start/end (first and last day of fasting)
RAMADAN = {
    2019: ("2019-05-06", "2019-06-04"),
    2020: ("2020-04-24", "2020-05-23"),
    2021: ("2021-04-13", "2021-05-12"),
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-21"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-30"),
}
# Eid al-Fitr and Eid al-Adha (first day; celebrations last 3-4 days)
EID_FITR = {
    2019: "2019-06-05", 2020: "2020-05-24", 2021: "2021-05-13",
    2022: "2022-05-02", 2023: "2023-04-22", 2024: "2024-04-10",
    2025: "2025-03-31",
}
EID_ADHA = {
    2019: "2019-08-12", 2020: "2020-07-31", 2021: "2021-07-20",
    2022: "2022-07-10", 2023: "2023-06-29", 2024: "2024-06-17",
    2025: "2025-06-07",
}

# Fixed-date public holidays in Pakistan
FIXED_HOLIDAYS = [(2, 5), (3, 23), (5, 1), (8, 14), (12, 25)]


def _in_range(series: pd.Series, start: str, end: str) -> pd.Series:
    return (series >= pd.Timestamp(start)) & (series <= pd.Timestamp(end))


def build_calendar_flags(dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"date": dates})

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    out["weekday"] = out["date"].dt.weekday  # Monday=0
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)

    # Cyclical encoding: a smooth representation of periodic time.
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)

    # Season (meteorological): DJF, MAM, JJA, SON
    month_to_season = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    out["season"] = out["month"].map(month_to_season)

    # Religious periods
    is_ramadan = pd.Series(False, index=out.index)
    for year, (start, end) in RAMADAN.items():
        is_ramadan |= _in_range(out["date"], start, end)
    out["is_ramadan"] = is_ramadan.astype(int)

    is_eid = pd.Series(False, index=out.index)
    for table in (EID_FITR, EID_ADHA):
        for year, first_day in table.items():
            start = pd.Timestamp(first_day)
            is_eid |= (out["date"] >= start) & (out["date"] <= start + pd.Timedelta(days=3))
    out["is_eid"] = is_eid.astype(int)

    # Public holidays = fixed dates + Eid days
    is_fixed = pd.Series(False, index=out.index)
    for month, day in FIXED_HOLIDAYS:
        is_fixed |= (out["month"] == month) & (out["date"].dt.day == day)
    out["is_public_holiday"] = (is_fixed | is_eid).astype(int)

    return out.drop(columns=["date"])


# --------------------------------------------------------------------------- #
# Load inputs
# --------------------------------------------------------------------------- #
def load_pm25() -> pd.DataFrame:
    # Only the cleaned file is accepted: it has one row per calendar day, so shift() means 'previous day'.
    candidates = [PROCESSED / "pm25_daily_clean.csv"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("processed/pm25_daily_clean.csv not found. Run 02 and 03 first.")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalise to a single target column name.
    if "pm25_clean" in df.columns:
        df["pm25"] = pd.to_numeric(df["pm25_clean"], errors="coerce")
    elif "pm25_mean_ugm3" in df.columns:
        df["pm25"] = pd.to_numeric(df["pm25_mean_ugm3"], errors="coerce")
    elif "pm25_mean" in df.columns:
        df["pm25"] = pd.to_numeric(df["pm25_mean"], errors="coerce")
    else:
        raise KeyError("Could not find a PM2.5 value column.")

    keep = ["date", "pm25"]
    for col in ("valid_hours", "usable_day", "aqi", "aqi_category"):
        if col in df.columns:
            keep.append(col)
    return df[keep].sort_values("date").reset_index(drop=True)


def load_weather() -> pd.DataFrame:
    path = RAW / "weather" / "open_meteo_era5_lahore_daily_2019_2026.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, skiprows=3)  # 3 metadata lines
    df = df.rename(columns={df.columns[0]: "date"})

    # "temperature_2m_mean (°C)" -> "temperature_2m_mean"
    df = df.rename(columns={c: c.split(" (")[0] for c in df.columns})

    friendly = {
        "temperature_2m_mean": "temp_mean_c",
        "temperature_2m_max": "temp_max_c",
        "temperature_2m_min": "temp_min_c",
        "relative_humidity_2m_mean": "humidity_mean_pct",
        "dew_point_2m_mean": "dew_point_mean_c",
        "surface_pressure_mean": "pressure_mean_hpa",
        "wind_speed_10m_mean": "wind_speed_mean_kmh",
        "wind_speed_10m_max": "wind_speed_max_kmh",
        "wind_direction_10m_dominant": "wind_dir_dominant_deg",
        "precipitation_sum": "precip_sum_mm",
        "rain_sum": "rain_sum_mm",
        "shortwave_radiation_sum": "shortwave_sum_mj_m2",
        "cloud_cover_mean": "cloud_cover_mean_pct",
    }
    df = df.rename(columns=friendly)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    weather_cols = [c for c in friendly.values() if c in df.columns]
    df = df.dropna(subset=weather_cols, how="all")  # drop the empty 2026-10-08 row
    return df[["date"] + weather_cols].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def robust_scale(series: pd.Series) -> pd.Series:
    """Median / IQR scaling (robust to smog outliers).
    Uses all rows; re-fit on the training rows only when modelling to avoid leakage."""
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    if not iqr or np.isnan(iqr):
        return series - median
    return (series - median) / iqr


def build_features(pm25: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = pm25.merge(weather, on="date", how="left").sort_values("date").reset_index(drop=True)

    calendar = build_calendar_flags(df["date"])
    df = pd.concat([df, calendar], axis=1)

    # --- target: tomorrow's PM2.5 (never use today's value as a feature without lag)
    df["target_next_pm25"] = df["pm25"].shift(-1)
    df["target_next_log"] = np.log1p(df["target_next_pm25"])
    if "aqi_category" in df.columns:  # classification target: tomorrow's AQI category
        df["target_next_aqi_category"] = df["aqi_category"].shift(-1)

    # --- log transform for the heavily right-skewed PM2.5 distribution
    df["pm25_log"] = np.log1p(df["pm25"])

    # --- PM2.5 lags and rolling means (shifted so no future information leaks)
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag2"] = df["pm25"].shift(2)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_roll3"] = df["pm25"].shift(1).rolling(3, min_periods=2).mean()
    df["pm25_roll7"] = df["pm25"].shift(1).rolling(7, min_periods=3).mean()
    df["pm25_diff1"] = df["pm25"] - df["pm25"].shift(1)

    # --- weather lags (yesterday's weather is known when forecasting tomorrow)
    for col in ("temp_mean_c", "humidity_mean_pct", "wind_speed_mean_kmh", "pressure_mean_hpa"):
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # --- robust scaling of meteorological variables
    for col in (
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "humidity_mean_pct",
        "dew_point_mean_c",
        "pressure_mean_hpa",
        "wind_speed_mean_kmh",
        "wind_speed_max_kmh",
        "precip_sum_mm",
        "shortwave_sum_mj_m2",
        "cloud_cover_mean_pct",
    ):
        if col in df.columns:
            df[f"{col}_scaled"] = robust_scale(df[col])

    # --- one-hot season (drop-first avoided on purpose: all four are informative)
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)

    return df


# --------------------------------------------------------------------------- #
# EDA figures
# --------------------------------------------------------------------------- #
def correlation_heatmap(df: pd.DataFrame) -> Path:
    variables = [
        "pm25",
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "humidity_mean_pct",
        "dew_point_mean_c",
        "pressure_mean_hpa",
        "wind_speed_mean_kmh",
        "wind_speed_max_kmh",
        "precip_sum_mm",
        "shortwave_sum_mj_m2",
        "cloud_cover_mean_pct",
    ]
    variables = [c for c in variables if c in df.columns]
    data = df.loc[df.get("usable_day", True) == True, variables].apply(pd.to_numeric, errors="coerce")  # noqa: E712
    corr = data.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Correlation between daily PM2.5 and meteorological variables\nLahore, 2019–2025", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()

    out = CHARTS / "correlation_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def weather_vs_pm25(df: pd.DataFrame) -> Path:
    pairs = [
        ("wind_speed_mean_kmh", "Mean wind speed (km/h)"),
        ("humidity_mean_pct", "Mean relative humidity (%)"),
        ("temp_mean_c", "Mean temperature (°C)"),
        ("pressure_mean_hpa", "Mean surface pressure (hPa)"),
    ]
    pairs = [(c, label) for c, label in pairs if c in df.columns]

    mask = df.get("usable_day", True) == True  # noqa: E712
    data = df.loc[mask, ["pm25"] + [c for c, _ in pairs]].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label) in zip(axes.ravel(), pairs):
        x = data[col]
        y = data["pm25"]
        ax.scatter(x, y, s=8, alpha=0.35, color="#1f4e79", edgecolors="none")
        if len(data) > 2:
            slope, intercept = np.polyfit(x, y, 1)
            order = np.argsort(x.values)
            ax.plot(x.values[order], (slope * x.values + intercept)[order], color="#c00000", linewidth=1.6)
            r = x.corr(y)
            ax.set_title(f"{label}  (r = {r:.2f})", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("PM2.5 (µg/m³)", fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("Daily PM2.5 vs weather, Lahore 2019–2025 (usable days only)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = CHARTS / "weather_vs_pm25.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    print(f"Data folder: {DATA}")
    pm25 = load_pm25()
    weather = load_weather()
    print(f"PM2.5 rows : {len(pm25):,}  ({pm25['date'].min().date()} -> {pm25['date'].max().date()})")
    print(f"Weather rows: {len(weather):,}  ({weather['date'].min().date()} -> {weather['date'].max().date()})")

    features = build_features(pm25, weather)
    out_csv = PROCESSED / "features_daily.csv"
    features.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(features):,} rows, {len(features.columns)} columns)")

    chart1 = correlation_heatmap(features)
    chart2 = weather_vs_pm25(features)
    print(f"Wrote {chart1}")
    print(f"Wrote {chart2}")

    print("\nFeature groups:")
    groups = {
        "target": ["target_next_pm25", "target_next_log", "target_next_aqi_category"],
        "pm25 history": ["pm25", "pm25_log", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3", "pm25_roll7", "pm25_diff1"],
        "weather": [c for c in features.columns if c.endswith("_scaled")],
        "calendar": ["year", "month", "day_of_year", "weekday", "is_weekend", "is_ramadan", "is_eid", "is_public_holiday"],
        "encoding": [c for c in features.columns if c.endswith("_sin") or c.endswith("_cos") or c.startswith("season_")],
    }
    for name, cols in groups.items():
        present = [c for c in cols if c in features.columns]
        print(f"  {name:14s}: {len(present):2d} -> {', '.join(present[:6])}{' ...' if len(present) > 6 else ''}")

    required = ["target_next_pm25", "pm25_lag1", "month_sin", "season_winter", "wind_speed_mean_kmh_scaled"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        print(f"\n[error] missing expected columns: {missing}")
        return 1
    print("\nOK: feature table and EDA charts generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
