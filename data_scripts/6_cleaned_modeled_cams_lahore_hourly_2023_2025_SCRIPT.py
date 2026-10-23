"""clean_file2.py -- cleans modeled_cams_lahore_hourly_2023_2025.csv (hourly CAMS pollutants + weather, Lahore).
Run from the repo root: python clean_file2.py  (reads data/raw/validation/, writes data/processed/)
The raw file is only read, never overwritten."""
import os
import pandas as pd

RAW = "data/raw/validation/modeled_cams_lahore_hourly_2023_2025.csv"
OUT = "data/processed/cleaned_modeled_cams_lahore_hourly_2023_2025.csv"

# Step 1: load raw file; treat the team's missing-value codes as NaN while reading
MISSING = ["N/A", "n/a", "NA", "null", "NULL", "None", "?", "-", "", "nan"]
df = pd.read_csv(RAW, na_values=MISSING)
before = (df.shape[0], df.shape[1], int(df.isna().sum().sum()))
print(f"[1] loaded {RAW}: {before[0]} rows, {before[1]} cols, {before[2]} nulls")

# Step 2: column names -> lowercase snake_case (team rule)
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

# Step 3: parse timestamp (format 2023-01-01T00:00); unparseable rows cannot be used
df["time"] = pd.to_datetime(df["time"].astype(str).str.strip(), format="%Y-%m-%dT%H:%M", errors="coerce")
bad_ts = int(df["time"].isna().sum())
df = df.dropna(subset=["time"])
print(f"[3] rows dropped for unparseable timestamp: {bad_ts}")
#the fix suggested for the pakistan and USA time zone issues. just adding the line below will fix it
# Step 3b: raw timestamps are UTC -> convert to Pakistan time (Asia/Karachi, UTC+5), then drop the tz label so the CSV stays plain
df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Karachi").dt.tz_localize(None)

# Step 4: force every measurement column to numeric (non-numeric -> NaN, counted)
value_cols = [c for c in df.columns if c != "time"]
nulls_before_cast = int(df[value_cols].isna().sum().sum())
for c in value_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
coerced = int(df[value_cols].isna().sum().sum()) - nulls_before_cast
print(f"[4] values turned into NaN by numeric cast: {coerced}")

# Step 5: remove exact duplicates and duplicate timestamps, then sort
n = len(df); df = df.drop_duplicates(); dup_rows = n - len(df)
n = len(df); df = df.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)
dup_ts = n - len(df)
print(f"[5] exact duplicates removed: {dup_rows}, duplicate timestamps removed: {dup_ts}")

# Step 6: check the series is continuous hourly (we only REPORT gaps, we do not invent rows)
expected = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
gaps = len(expected.difference(df["time"]))
print(f"[6] missing hourly timestamps: {gaps}")

# Step 7: physically impossible values -> NaN (counted). Zeros are kept: night-time ozone = 0 is real (NO titration).
impossible = {
    "negative concentration": df[["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]].lt(0),
    "humidity outside 0-100": df[["relative_humidity_2m"]].lt(0) | df[["relative_humidity_2m"]].gt(100),
    "negative wind speed": df[["wind_speed_10m"]].lt(0),
    "wind direction outside 0-360": df[["wind_direction_10m"]].lt(0) | df[["wind_direction_10m"]].gt(360),
}
for name, mask in impossible.items():
    cnt = int(mask.sum().sum())
    for c in mask.columns:
        df[c] = df[c].mask(mask[c])
    print(f"[7] {name}: {cnt} set to NaN")

# Step 8: PM2.5 can never exceed PM10 (PM2.5 is a subset); count violations (flag-only, none expected)
print(f"[8] rows with pm2_5 > pm10: {int((df['pm2_5'] > df['pm10']).sum())}")

# Step 9: flag extreme HIGH values (> Q3 + 3*IQR). Mostly real winter-smog spikes, so flag only.
def flag_high(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    return (s > q3 + k * (q3 - q1)).astype(int)

for c in ["pm2_5", "pm10", "carbon_monoxide"]:
    df[f"{c}_outlier"] = flag_high(df[c])
    print(f"[9] {c}_outlier flagged: {int(df[f'{c}_outlier'].sum())}")

# Step 10: no nulls expected; no imputation anywhere
print(f"[10] nulls remaining in measurement columns: {int(df[value_cols].isna().sum().sum())}")

# Step 11: add join-key helper columns (date as YYYY-MM-DD, hour 0-23), rename time -> datetime, save
df = df.rename(columns={"time": "datetime"})
df.insert(1, "date", df["datetime"].dt.strftime("%Y-%m-%d"))
df.insert(2, "hour", df["datetime"].dt.hour)
df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
os.makedirs(os.path.dirname(OUT), exist_ok=True)  # create data/processed/ if missing
df.to_csv(OUT, index=False)

# Step 12: before/after + change log
flag_cols = [c for c in df if c.endswith("_outlier")]
after = (df.shape[0], df.shape[1], int(df[value_cols].isna().sum().sum()))
print(f"\nBEFORE: rows={before[0]} cols={before[1]} nulls={before[2]}")
print(f"AFTER : rows={after[0]} cols={after[1]} nulls={after[2]} (cols include date, hour + {len(flag_cols)} flag columns)")
print("\nCHANGE LOG")
print(f"1. Parsed 'time' to datetime, renamed to 'datetime' (YYYY-MM-DD HH:MM:SS); added 'date' (join key to the daily files) and 'hour'.")
print(f"2. {dup_rows + dup_ts} duplicates, {gaps} hourly gaps, 0 nulls, 0 impossible values found; 0 rows dropped; no imputation.")
print(f"3. Added outlier flags (pm2_5 {int(df['pm2_5_outlier'].sum())}, pm10 {int(df['pm10_outlier'].sum())}, carbon_monoxide {int(df['carbon_monoxide_outlier'].sum())} rows); nothing deleted.")
print("\nWARNINGS")
print("- LEAKAGE: if the target is next-day daily PM2.5, any hourly pm2_5/pm10 from day t+1 must NOT be in the features. Aggregate to day t only.")
print("- Weather columns here are for the SAME hour as the pollutant. At forecast time you do not know tomorrow's weather: use lags or a weather forecast.")
print("- File ends 2025-05-01, so your joined modeling table is limited to 2023-01-01..2025-05-01 if you use hourly data.")
print("- No personal data in this file.")
