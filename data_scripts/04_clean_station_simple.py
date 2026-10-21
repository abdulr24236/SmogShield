# 04_clean_station_simple.py
# Cleans the station AQI + weather file (daily, 2019-2023).
#
# Input : data/raw/validation/station_aqi_weather_lahore_daily_2019_2023.csv   (never modified)
# Output: data/processed/station_daily.csv
# Run from the main project folder:  python data_scripts/04_clean_station_simple.py

import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/validation/station_aqi_weather_lahore_daily_2019_2023.csv")
print("Rows loaded:", len(df))

# ---------- 1. dates (the file uses day-month-year with a 2-digit year, e.g. 01-06-19) ----------
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%y")
df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

# ---------- 2. impossible values -> empty (NaN) ----------
# A value outside what is physically possible in Lahore is a sensor/typing error, not a measurement.
limits = {                        # column prefix: (lowest, highest) allowed
    "temp_f":       (30, 125),    # degrees F
    "dew_point_f":  (15, 90),     # degrees F
    "humidity_percent": (1, 100),
    "wind_speed_mph":   (0, 80),
    "pressure_in":  (27.5, 30.5), # inches of mercury
}
bad_total = 0
for prefix, (low, high) in limits.items():
    for kind in ["min", "avg", "max"]:
        col = kind + "_" + prefix
        wrong = (df[col] < low) | (df[col] > high)
        # a minimum wind speed of 0 is a real value (calm air), so don't touch it
        if col == "min_wind_speed_mph":
            wrong = df[col] > high
        bad_total += wrong.sum()
        df.loc[wrong, col] = np.nan
print("Impossible values turned into empty:", bad_total)

# ---------- 3. min <= avg <= max must hold; if not, the row's triple is unreliable ----------
inconsistent = 0
for prefix in limits:
    lo, av, hi = "min_" + prefix, "avg_" + prefix, "max_" + prefix
    wrong = (df[lo] > df[av]) | (df[av] > df[hi])
    inconsistent += wrong.sum()
    df.loc[wrong, [lo, av, hi]] = np.nan
print("Rows where min/avg/max disagree (set empty):", inconsistent)

# ---------- 4. units: convert to the same units as the Open-Meteo weather file ----------
for kind in ["min", "avg", "max"]:
    df[kind + "_temp_c"] = (df[kind + "_temp_f"] - 32) * 5 / 9
    df[kind + "_dew_point_c"] = (df[kind + "_dew_point_f"] - 32) * 5 / 9
    df[kind + "_wind_speed_kmh"] = df[kind + "_wind_speed_mph"] * 1.60934
    df[kind + "_pressure_hpa"] = df[kind + "_pressure_in"] * 33.8639
df = df.drop(columns=[c for c in df.columns if c.endswith(("_f", "_mph", "_in"))])
df = df.rename(columns={"aqi_pm2.5": "aqi"})

# ---------- 5. missing values ----------
# The target (aqi) is NEVER filled: a missing day stays empty and is left out of training.
print("AQI missing days:", df["aqi"].isna().sum(), "of", len(df))
# Weather: only the daily AVERAGE columns are filled, and only up to 3 days in a row, using the
# previous day (no future values). Longer gaps stay empty.
avg_cols = [c for c in df.columns if c.startswith("avg_")]
print("Empty weather averages before filling:", int(df[avg_cols].isna().sum().sum()))
df[avg_cols] = df[avg_cols].ffill(limit=3)
print("Empty weather averages after filling :", int(df[avg_cols].isna().sum().sum()))

# ---------- 6. outliers in AQI: just report, never delete (smog peaks are real) ----------
print("AQI max:", df["aqi"].max(), "| days with AQI above 400:", int((df["aqi"] > 400).sum()))

df.to_csv("data/processed/station_daily.csv", index=False)
print("Saved data/processed/station_daily.csv with shape", df.shape)
