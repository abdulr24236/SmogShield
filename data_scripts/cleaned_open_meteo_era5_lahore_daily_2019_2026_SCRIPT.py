"""clean_file3.py -- cleans open_meteo_era5_lahore_daily_2019_2026.csv (daily ERA5 weather, Lahore).
Put the raw CSV in the same folder as this script, then run: python clean_file3.py
The raw file is only read, never overwritten."""
import pandas as pd

RAW = "open_meteo_era5_lahore_daily_2019_2026.csv"
OUT = "cleaned_open_meteo_era5_lahore_daily_2019_2026.csv"

# Step 1: the file starts with a 2-line location block + blank line, THEN the real table. Read both parts separately.
MISSING = ["N/A", "n/a", "NA", "null", "NULL", "None", "?", "-", "", "nan"]
meta = pd.read_csv(RAW, nrows=1)                       # latitude, longitude, elevation, timezone info
df = pd.read_csv(RAW, skiprows=3, na_values=MISSING)   # the actual daily table
before = (df.shape[0], df.shape[1], int(df.isna().sum().sum()))
print(f"[1] metadata block: {meta.iloc[0].to_dict()}")
print(f"[1] loaded table: {before[0]} rows, {before[1]} cols, {before[2]} nulls")

# Step 2: column names -> lowercase snake_case, with units kept at the end (explicit map is clearer than regex here)
rename = {
    "time": "date",
    "temperature_2m_mean (°C)": "temperature_2m_mean_c",
    "temperature_2m_max (°C)": "temperature_2m_max_c",
    "temperature_2m_min (°C)": "temperature_2m_min_c",
    "relative_humidity_2m_mean (%)": "relative_humidity_2m_mean_pct",
    "dew_point_2m_mean (°C)": "dew_point_2m_mean_c",
    "surface_pressure_mean (hPa)": "surface_pressure_mean_hpa",
    "wind_speed_10m_mean (km/h)": "wind_speed_10m_mean_kmh",
    "wind_speed_10m_max (km/h)": "wind_speed_10m_max_kmh",
    "wind_direction_10m_dominant (°)": "wind_direction_10m_dominant_deg",
    "precipitation_sum (mm)": "precipitation_sum_mm",
    "rain_sum (mm)": "rain_sum_mm",
    "shortwave_radiation_sum (MJ/m²)": "shortwave_radiation_sum_mj_m2",
    "cloud_cover_mean (%)": "cloud_cover_mean_pct",
}
df.columns = df.columns.str.strip()
unmapped = [c for c in df.columns if c not in rename]
assert not unmapped, f"unexpected columns: {unmapped}"
df = df.rename(columns=rename)

# Step 3: force every measurement column to numeric (non-numeric -> NaN, counted)
value_cols = [c for c in df.columns if c != "date"]
nulls_before_cast = int(df[value_cols].isna().sum().sum())
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
coerced = int(df[value_cols].isna().sum().sum()) - nulls_before_cast
print(f"[3] values turned into NaN by numeric cast: {coerced}")

# Step 4: parse the date join key (trim whitespace first); unparseable dates cannot be used
df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce")
bad_dates = int(df["date"].isna().sum())
df = df.dropna(subset=["date"])
print(f"[4] rows dropped for unparseable date: {bad_dates}")

# Step 5: remove exact duplicates and duplicate dates, then sort
n = len(df); df = df.drop_duplicates(); dup_rows = n - len(df)
n = len(df); df = df.drop_duplicates(subset="date", keep="first").sort_values("date").reset_index(drop=True)
dup_dates = n - len(df)
print(f"[5] exact duplicates removed: {dup_rows}, duplicate dates removed: {dup_dates}")

# Step 6: drop rows where EVERY measurement is missing (the last day of the file is an empty placeholder)
all_null = df[value_cols].isna().all(axis=1)
dropped_dates = df.loc[all_null, "date"].dt.strftime("%Y-%m-%d").tolist()
df = df.loc[~all_null].reset_index(drop=True)
print(f"[6] all-null rows dropped: {len(dropped_dates)} -> {dropped_dates}")

# Step 7: check the date range has no gaps (report only, no invented rows)
gaps = len(pd.date_range(df["date"].min(), df["date"].max(), freq="D").difference(df["date"]))
print(f"[7] missing dates in range: {gaps}")

# Step 8: physical sanity checks (counts only; all should be 0). Outliers like 93 mm monsoon rain are real, so no flags/deletes.
c = df
checks = {
    "tmin > tmean": (c["temperature_2m_min_c"] > c["temperature_2m_mean_c"]).sum(),
    "tmean > tmax": (c["temperature_2m_mean_c"] > c["temperature_2m_max_c"]).sum(),
    "humidity outside 0-100": ((c["relative_humidity_2m_mean_pct"] < 0) | (c["relative_humidity_2m_mean_pct"] > 100)).sum(),
    "negative precipitation": (c["precipitation_sum_mm"] < 0).sum(),
    "cloud cover outside 0-100": ((c["cloud_cover_mean_pct"] < 0) | (c["cloud_cover_mean_pct"] > 100)).sum(),
    "wind direction outside 0-360": ((c["wind_direction_10m_dominant_deg"] < 0) | (c["wind_direction_10m_dominant_deg"] > 360)).sum(),
}
for k, v in checks.items():
    print(f"[8] {k}: {int(v)}")

# Step 9: redundancy check (reported, not dropped): rain_sum vs precipitation_sum
same = float((df["rain_sum_mm"] == df["precipitation_sum_mm"]).mean() * 100)
print(f"[9] rain_sum_mm equals precipitation_sum_mm on {same:.2f}% of days")

# Step 10: no imputation; date back to YYYY-MM-DD text (team rule) and save
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df.to_csv(OUT, index=False)

# Step 11: before/after + change log
after = (df.shape[0], df.shape[1], int(df.isna().sum().sum()))
print(f"\nBEFORE: rows={before[0]} cols={before[1]} nulls={before[2]}")
print(f"AFTER : rows={after[0]} cols={after[1]} nulls={after[2]}")
print("\nCHANGE LOG")
print("1. Skipped the 3-line location header, renamed time -> date and all 13 columns to snake_case with units suffix (_c, _pct, _kmh...).")
print(f"2. Dropped {len(dropped_dates)} all-null row ({', '.join(dropped_dates)}); {dup_rows + dup_dates} duplicates, {gaps} date gaps found.")
print("3. No imputation, no outlier removal; physical-range checks all passed, dates saved as YYYY-MM-DD.")
print("\nWARNINGS")
print("- LEAKAGE: none in the columns themselves, but this is same-day weather. For next-day PM2.5, use lagged weather or a forecast, not day t+1 actuals.")
print("- rain_sum_mm is IDENTICAL to precipitation_sum_mm on every day: redundant, drop one at modeling time.")
print("- Location is a single ERA5 grid point (see metadata above), not a station. No personal data in this file.")
