# 02_clean_pm25_simple.py
# Cleans the raw OpenAQ hourly PM2.5 file and makes an hourly and a daily version.
#
# Before running: make sure these folders exist (create them by hand if not):
#   data/raw/ground_truth/   (with the raw csv inside)
#   data/processed/
# Run this from the main project folder:  python data_scripts/02_clean_pm25_simple.py

import pandas as pd
import numpy as np

# ---------- 1. load ----------
df = pd.read_csv("data/raw/ground_truth/openaq_us_post_25135_hourly.csv")
print("Rows loaded:", len(df))

# ---------- 2. dates ----------
# date_utc is the START of each hour in UTC. Pakistan time = UTC + 5 hours.
df["time_utc"] = pd.to_datetime(df["date_utc"], utc=True).dt.tz_localize(None)

# ---------- 3. duplicates ----------
before = len(df)
df = df.drop_duplicates(subset="time_utc")
df = df.sort_values("time_utc")
print("Duplicate hours removed:", before - len(df))

# ---------- 4. missing and impossible values ----------
# Empty readings stay empty (NaN). We do NOT fill PM2.5 with a mean or median.
df["pm25"] = df["value"]
negatives = (df["pm25"] < 0).sum()
df.loc[df["pm25"] < 0, "pm25"] = np.nan        # PM2.5 can't be negative
print("Negative values turned into empty:", negatives)
print("Empty readings:", df["pm25"].isna().sum())

# ---------- 5. add the hours that are missing completely ----------
all_hours = pd.date_range(df["time_utc"].min(), df["time_utc"].max(), freq="h")
df = df.set_index("time_utc").reindex(all_hours)
df.index.name = "time_utc"
df = df.reset_index()
print("Hours in the full period:", len(df), "| hours with a value:", df["pm25"].notna().sum())

# ---------- 6. outliers: flag them, do NOT delete them ----------
# Smog extremes are real. We only mark very unusual values (log scale, because the data is skewed).
logs = np.log1p(df["pm25"])
q1 = logs.quantile(0.25)
q3 = logs.quantile(0.75)
iqr = q3 - q1
df["outlier"] = (logs < q1 - 3 * iqr) | (logs > q3 + 3 * iqr)
print("Outliers flagged:", df["outlier"].sum(), "| exact zeros:", (df["pm25"] == 0).sum(), "| max:", df["pm25"].max())

# ---------- 7. save the hourly file ----------
df["time_local"] = df["time_utc"] + pd.Timedelta(hours=5)
hourly = df[["time_utc", "time_local", "pm25", "outlier"]]
hourly.to_csv("data/processed/pm25_hourly.csv", index=False)

# ---------- 8. daily file ----------
# A day counts as usable only if it has at least 18 hours with a value.
df["date"] = df["time_local"].dt.floor("D")
daily = df.groupby("date")["pm25"].agg(valid_hours="count", pm25_mean="mean").reset_index()
daily["usable_day"] = daily["valid_hours"] >= 18
daily.to_csv("data/processed/pm25_daily.csv", index=False)
print("Days:", len(daily), "| usable days:", daily["usable_day"].sum())
