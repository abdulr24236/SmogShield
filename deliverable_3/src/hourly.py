"""Hour-of-day PM2.5 profile for the Deliverable 3 outdoor-planning feature.

Reads the cleaned hourly ground-truth file produced by
``data_scripts/02_clean_pm25_simple.py`` (OpenAQ sensor 25135, Lahore US
Diplomatic Post) and summarises mean PM2.5 by local hour and season.

The output is descriptive and station-specific. It is not a today-specific
forecast and does not model hour-to-hour dynamics; it answers "historically,
which hours are least polluted?".

If the hourly file is absent (for example on a clean clone where data has not
been restored), ``build_hourly_profile`` returns an empty dict and the app
simply omits the card.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOURLY_CSV = ROOT / "data" / "processed" / "pm25_hourly.csv"

SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _outlier_mask(values: pd.Series) -> pd.Series:
    """Handle both real booleans and string flags from older exports."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_hourly_profile(csv_path: Path | None = None) -> dict[str, object]:
    """Return the historical hour-of-day profile, or ``{}`` if unavailable."""
    path = Path(csv_path) if csv_path is not None else HOURLY_CSV
    if not path.exists():
        return {}

    frame = pd.read_csv(path)
    if "time_local" not in frame.columns or "pm25" not in frame.columns:
        return {}

    frame["time_local"] = pd.to_datetime(frame["time_local"], errors="coerce")
    frame["pm25"] = pd.to_numeric(frame["pm25"], errors="coerce")
    frame = frame.dropna(subset=["time_local", "pm25"])
    frame = frame[(frame["pm25"] >= 0) & (frame["pm25"] <= 500)]

    if "outlier" in frame.columns:
        frame = frame[~_outlier_mask(frame["outlier"])]

    if frame.empty:
        return {}

    frame["hour"] = frame["time_local"].dt.hour
    frame["season"] = frame["time_local"].dt.month.map(SEASONS)

    overall = frame.groupby("hour")["pm25"].agg(["mean", "median", "count"])
    by_hour = {
        str(int(hour)): {
            "mean": round(float(row["mean"]), 1),
            "median": round(float(row["median"]), 1),
            "count": int(row["count"]),
        }
        for hour, row in overall.iterrows()
    }

    means = overall["mean"].to_numpy(dtype=float)
    window_means = [
        (float(np.mean(means[start : start + 3])), start) for start in range(22)
    ]
    window_mean, window_start = min(window_means)

    seasonal = (
        frame.groupby(["season", "hour"])["pm25"].mean().round(1).unstack("hour")
    )
    seasonal_by_hour = {
        str(season): [
            float(seasonal.loc[season, hour])
            if season in seasonal.index and hour in seasonal.columns
            else None
            for hour in range(24)
        ]
        for season in ("winter", "spring", "summer", "autumn")
    }

    seasonal_windows: dict[str, dict[str, float | int]] = {}
    for season, values in seasonal_by_hour.items():
        clean = [value for value in values if value is not None]
        if len(clean) != 24:
            continue
        windows = [
            (float(np.mean(clean[start : start + 3])), start) for start in range(22)
        ]
        best, best_start = min(windows)
        seasonal_windows[season] = {
            "start_hour": int(best_start),
            "end_hour": int(best_start + 2),
            "mean_pm25": round(best, 1),
        }

    peak_hour = int(np.argmax(means))

    return {
        "source": (
            "OpenAQ sensor 25135 (Lahore US Diplomatic Post), cleaned by "
            "data_scripts/02_clean_pm25_simple.py"
        ),
        "valid_hours": int(len(frame)),
        "date_min": frame["time_local"].min().strftime("%Y-%m-%d"),
        "date_max": frame["time_local"].max().strftime("%Y-%m-%d"),
        "by_hour": by_hour,
        "cleanest_window": {
            "start_hour": int(window_start),
            "end_hour": int(window_start + 2),
            "mean_pm25": round(window_mean, 1),
        },
        "peak_hour": {
            "hour": peak_hour,
            "mean_pm25": round(float(means[peak_hour]), 1),
        },
        "seasonal_by_hour": seasonal_by_hour,
        "seasonal_cleanest_window": seasonal_windows,
        "caveat": (
            "Descriptive station history in Pakistan local time, not a "
            "today-specific forecast. Winter afternoons remain polluted."
        ),
    }
