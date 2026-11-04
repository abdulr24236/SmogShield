"""
Cleanup script for: data/archive_unused/
Files covered:
  - lahore_complete_data_july_to_dec_2024.csv
  - pakistan_air_quality_annual_summary.csv
  - weather.csv
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# 1. lahore_complete_data_july_to_dec_2024.csv
# ─────────────────────────────────────────────────────────────
def clean_lahore_complete():
    path = os.path.join(DATA_DIR, "lahore_complete_data_july_to_dec_2024.csv")
    df = pd.read_csv(path)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Sort by time
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Clip obviously impossible values (AQI must be 1-5 categories; pm2.5/pm10 >= 0)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    print(f"[lahore_complete] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. pakistan_air_quality_annual_summary.csv
# ─────────────────────────────────────────────────────────────
def clean_pakistan_summary():
    path = os.path.join(DATA_DIR, "pakistan_air_quality_annual_summary.csv")
    df = pd.read_csv(path)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values with per-city median (city+year grouped)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        df[col] = df.groupby("city")[col].transform(
            lambda x: x.fillna(x.median())
        )
    # If still NaN (e.g. only one row per city), fill with global median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Sort
    df.sort_values(["city", "year"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[pakistan_summary] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. weather.csv
# ─────────────────────────────────────────────────────────────
def clean_weather():
    path = os.path.join(DATA_DIR, "weather.csv")
    df = pd.read_csv(path)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Strip whitespace from city
    df["city"] = df["city"].str.strip()

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Clip negatives for rain / solar_radiation
    df["rain"] = df["rain"].clip(lower=0)
    df["solar_radiation"] = df["solar_radiation"].clip(lower=0)

    # Sort
    df.sort_values(["city", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[weather] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lahore  = clean_lahore_complete()
    pak_sum = clean_pakistan_summary()
    weather = clean_weather()

    print("\nAll archive_unused files cleaned successfully.")
    print("  Tip: call each function and save with df.to_csv(..., index=False) to overwrite.")
