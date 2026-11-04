"""
Cleanup script for: data/raw/weather/
Files covered:
  - open_meteo_era5_lahore_daily_2019_2026.csv
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_era5_daily():
    path = os.path.join(DATA_DIR, "open_meteo_era5_lahore_daily_2019_2026.csv")
    df = pd.read_csv(path)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Drop exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Clip negatives for precipitation, rain, and radiation (physically impossible)
    non_neg_cols = [
        "precipitation_sum",
        "rain_sum",
        "shortwave_radiation_sum",
        "cloud_cover_mean",
    ]
    df[non_neg_cols] = df[non_neg_cols].clip(lower=0)

    # Clip cloud_cover_mean to [0, 100] %
    df["cloud_cover_mean"] = df["cloud_cover_mean"].clip(upper=100)

    # Clip relative_humidity_2m_mean to [0, 100] %
    df["relative_humidity_2m_mean"] = df["relative_humidity_2m_mean"].clip(lower=0, upper=100)

    # Sort chronologically
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[era5_daily] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    era5 = clean_era5_daily()

    print("\nAll raw/weather files cleaned successfully.")
    print("  Tip: call the function and save with df.to_csv(..., index=False) to overwrite.")
