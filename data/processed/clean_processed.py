"""
Cleanup script for: data/processed/
Files covered:
  - lahore_us_post_pm25_daily_2019_2025.csv   (has 276 nulls in pm25_mean_ugm3)
  - lahore_us_post_pm25_hourly_2019_2025.csv
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# 1. lahore_us_post_pm25_daily_2019_2025.csv
# ─────────────────────────────────────────────────────────────
def clean_pm25_daily():
    path = os.path.join(DATA_DIR, "lahore_us_post_pm25_daily_2019_2025.csv")
    df = pd.read_csv(path)

    # Parse date
    df["date_local"] = pd.to_datetime(df["date_local"])

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Clip negatives — PM2.5 cannot be negative
    df["pm25_mean_ugm3"] = df["pm25_mean_ugm3"].clip(lower=0)

    # Flag missing PM2.5 rows (nulls come from days with insufficient hourly coverage)
    # usable_day already marks these; just ensure consistency
    df.loc[df["pm25_mean_ugm3"].isna(), "usable_day"] = False

    # Sort chronologically
    df.sort_values("date_local", inplace=True)
    df.reset_index(drop=True, inplace=True)

    null_count = df["pm25_mean_ugm3"].isna().sum()
    print(f"[pm25_daily] shape={df.shape}, pm25 nulls remaining={null_count}")
    print(f"  Note: {null_count} days have no PM2.5 reading (usable_day=False) — kept as NaN intentionally.")
    return df


# ─────────────────────────────────────────────────────────────
# 2. lahore_us_post_pm25_hourly_2019_2025.csv
# ─────────────────────────────────────────────────────────────
def clean_pm25_hourly():
    path = os.path.join(DATA_DIR, "lahore_us_post_pm25_hourly_2019_2025.csv")
    df = pd.read_csv(path)

    # Parse timestamps
    df["timestamp_utc_period_start"] = pd.to_datetime(df["timestamp_utc_period_start"], utc=True)
    df["timestamp_local"]            = pd.to_datetime(df["timestamp_local"])

    # Strip whitespace from source
    df["source"] = df["source"].str.strip()

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Clip negative PM2.5
    df["pm25_ugm3"] = df["pm25_ugm3"].clip(lower=0)

    # Sort chronologically
    df.sort_values("timestamp_utc_period_start", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[pm25_hourly] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    daily  = clean_pm25_daily()
    hourly = clean_pm25_hourly()

    print("\nAll processed files cleaned successfully.")
    print("  Tip: call each function and save with df.to_csv(..., index=False) to overwrite.")
