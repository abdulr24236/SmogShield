"""
Cleanup script for: data/raw/ground_truth/
Files covered:
  - openaq_us_post_25135_hourly.csv  (30 columns, raw OpenAQ API dump)
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_openaq_hourly():
    path = os.path.join(DATA_DIR, "openaq_us_post_25135_hourly.csv")
    df = pd.read_csv(path)

    # ── Rename dot-notation columns to snake_case ──────────────────────────
    df.columns = (
        df.columns
        .str.replace(".", "_", regex=False)
        .str.lower()
    )

    # ── Parse all datetime columns ────────────────────────────────────────
    dt_cols = [c for c in df.columns if "utc" in c or "local" in c]
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=("utc" in col))

    # ── Drop exact duplicate rows ─────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── Clip negative PM2.5 values ────────────────────────────────────────
    df["value"] = df["value"].clip(lower=0)

    # ── Remove rows with coverage below 50% (unreliable readings) ─────────
    before = len(df)
    df = df[df["coverage_percentcoverage"] >= 50.0].copy()
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed} rows with coverage < 50%")

    # ── Sort by UTC start time ────────────────────────────────────────────
    df.sort_values("date_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[openaq_hourly] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    openaq = clean_openaq_hourly()

    print("\nAll ground_truth files cleaned successfully.")
    print("  Tip: call the function and save with df.to_csv(..., index=False) to overwrite.")
