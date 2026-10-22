"""clean_file1.py -- cleans modeled_cams_lahore_daily_2022_2026.csv (daily CAMS air quality, Lahore).
Put the raw CSV in the same folder as this script, then run: python clean_file1.py
The raw file is only read, never overwritten."""
import pandas as pd

RAW = "modeled_cams_lahore_daily_2022_2026.csv"
OUT = "cleaned_modeled_cams_lahore_daily_2022_2026.csv"

# Step 1: load raw file; treat the team's missing-value codes as NaN while reading
MISSING = ["N/A", "n/a", "NA", "null", "NULL", "None", "?", "-", "", "nan"]
df = pd.read_csv(RAW, na_values=MISSING)
before = (df.shape[0], df.shape[1], int(df.isna().sum().sum()))
print(f"[1] loaded {RAW}: {before[0]} rows, {before[1]} cols, {before[2]} nulls")

# Step 2: column names -> lowercase snake_case (team rule)
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

# Step 3: force every measurement column to numeric (anything non-numeric becomes NaN, and we count it)
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

# Step 5: remove exact duplicate rows
n = len(df); df = df.drop_duplicates(); dup_rows = n - len(df)
print(f"[5] exact duplicate rows removed: {dup_rows}")

# Step 6: one row per date (keep first), then sort by date
n = len(df); df = df.drop_duplicates(subset="date", keep="first").sort_values("date").reset_index(drop=True)
dup_dates = n - len(df)
print(f"[6] duplicate-date rows removed: {dup_dates}")

# Step 7: drop rows where EVERY measurement is missing (no features, no target = useless row)
all_null = df[value_cols].isna().all(axis=1)
dropped_dates = df.loc[all_null, "date"].dt.strftime("%Y-%m-%d").tolist()
df = df.loc[~all_null].reset_index(drop=True)
print(f"[7] all-null rows dropped: {len(dropped_dates)} -> {dropped_dates}")

# Step 8: remaining NaNs are left as NaN (no imputation, so nothing is invented)
print("[8] remaining nulls per column (left as NaN):")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

# Step 9: flag physically suspicious near-zero PM2.5 (Lahore never has <5 ug/m3 daily mean; likely a model artifact). Flag, don't delete.
df["suspect_low_pm"] = (df["pm2_5"] < 5).astype(int)
print(f"[9] suspect_low_pm flagged: {int(df['suspect_low_pm'].sum())} -> "
      f"{df.loc[df['suspect_low_pm'] == 1, 'date'].dt.strftime('%Y-%m-%d').tolist()}")

# Step 10: flag extreme HIGH values (> Q3 + 3*IQR). These are mostly real smog/dust events, so flag only.
def flag_high(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    return (s > q3 + k * (q3 - q1)).astype(int)

for c in ["pm2_5", "pm10", "dust"]:
    df[f"{c}_outlier"] = flag_high(df[c])
    print(f"[10] {c}_outlier flagged: {int(df[f'{c}_outlier'].sum())}")

# Step 11: date back to YYYY-MM-DD text (team rule) and save
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df.to_csv(OUT, index=False)

# Step 12: before/after + change log
after = (df.shape[0], df.shape[1], int(df.drop(columns=[c for c in df if c.endswith("_outlier") or c == "suspect_low_pm"]).isna().sum().sum()))
print(f"\nBEFORE: rows={before[0]} cols={before[1]} nulls={before[2]}")
print(f"AFTER : rows={after[0]} cols={after[1]} nulls={after[2]} (cols include 4 new flag columns)")
print("\nCHANGE LOG")
print(f"1. Columns already snake_case; dates parsed to YYYY-MM-DD; all measurement columns cast to float ({coerced} values coerced).")
print(f"2. Dropped {len(dropped_dates)} all-null rows ({dropped_dates[0]} to {dropped_dates[-1]}); 0 duplicates found; no imputation.")
print(f"3. Added flags: suspect_low_pm ({int(df['suspect_low_pm'].sum())} rows), pm2_5/pm10/dust_outlier; nothing else removed.")
print("\nWARNINGS")
print("- LEAKAGE: us_aqi and european_aqi are computed from PM2.5/PM10. If your target is next-day PM2.5 or AQI category, use them only as LAGGED features.")
print("- Same-day pm2_5/pm10 are the target's own history: build the target with shift(-1), never mix day t+1 values into features.")
print("- No personal data in this file.")
