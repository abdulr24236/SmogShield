"""
Cleanup script for: data/raw/validation/
Files covered:
  - city_info.csv
  - data_dictionary.csv
  - modeled_cams_lahore_daily_2022_2026.csv
  - modeled_cams_lahore_hourly_2023_2025.csv
  - station_aqi_weather_lahore_daily_2019_2023.csv
"""

import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# 1. city_info.csv  (1 row — just validate / tidy)
# ─────────────────────────────────────────────────────────────
def clean_city_info():
    path = os.path.join(DATA_DIR, "city_info.csv")
    df = pd.read_csv(path)

    # Strip strings
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    print(f"[city_info] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. data_dictionary.csv  (metadata — just tidy strings)
# ─────────────────────────────────────────────────────────────
def clean_data_dictionary():
    path = os.path.join(DATA_DIR, "data_dictionary.csv")
    df = pd.read_csv(path)

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[data_dictionary] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. modeled_cams_lahore_daily_2022_2026.csv
# ─────────────────────────────────────────────────────────────
def clean_cams_daily():
    path = os.path.join(DATA_DIR, "modeled_cams_lahore_daily_2022_2026.csv")
    df = pd.read_csv(path)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Clip negatives for pollutant concentrations and AQI
    non_neg_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "aerosol_optical_depth",
                    "dust", "uv_index", "us_aqi", "european_aqi"]
    df[non_neg_cols] = df[non_neg_cols].clip(lower=0)

    # Sort chronologically
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[cams_daily] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. modeled_cams_lahore_hourly_2023_2025.csv
# ─────────────────────────────────────────────────────────────
def clean_cams_hourly():
    path = os.path.join(DATA_DIR, "modeled_cams_lahore_hourly_2023_2025.csv")
    df = pd.read_csv(path)

    # Parse timestamp
    df["time"] = pd.to_datetime(df["time"])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Clip negatives for pollutants
    pollutant_cols = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide",
                      "sulphur_dioxide", "carbon_monoxide"]
    df[pollutant_cols] = df[pollutant_cols].clip(lower=0)

    # Sort chronologically
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[cams_hourly] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 5. station_aqi_weather_lahore_daily_2019_2023.csv
# ─────────────────────────────────────────────────────────────
def clean_station_aqi_weather():
    path = os.path.join(DATA_DIR, "station_aqi_weather_lahore_daily_2019_2023.csv")
    df = pd.read_csv(path)

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Convert Fahrenheit temp columns to Celsius (optional but standardizes units)
    temp_f_cols = ["max_temp_f", "avg_temp_f", "min_temp_f"]
    for col in temp_f_cols:
        new_col = col.replace("_f", "_c")
        df[new_col] = (df[col] - 32) * 5 / 9
    df.drop(columns=temp_f_cols, inplace=True)

    # Convert dew point F → C
    dew_f_cols = ["max_dew_point_f", "avg_dew_point_f", "min_dew_point_f"]
    for col in dew_f_cols:
        new_col = col.replace("_f", "_c")
        df[new_col] = (df[col] - 32) * 5 / 9
    df.drop(columns=dew_f_cols, inplace=True)

    # Convert wind speed mph → km/h
    wind_mph_cols = ["max_wind_speed_mph", "avg_wind_speed_mph", "min_wind_speed_mph"]
    for col in wind_mph_cols:
        new_col = col.replace("_mph", "_kmh")
        df[new_col] = df[col] * 1.60934
    df.drop(columns=wind_mph_cols, inplace=True)

    # Clip negatives for AQI, humidity, pressure
    df["aqi_pm2.5"] = df["aqi_pm2.5"].clip(lower=0)
    humidity_cols = [c for c in df.columns if "humidity" in c]
    df[humidity_cols] = df[humidity_cols].clip(lower=0, upper=100)

    # Sort chronologically
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[station_aqi_weather] shape={df.shape}, nulls={df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    city_info   = clean_city_info()
    data_dict   = clean_data_dictionary()
    cams_daily  = clean_cams_daily()
    cams_hourly = clean_cams_hourly()
    station     = clean_station_aqi_weather()

    print("\nAll validation files cleaned successfully.")
    print("  Tip: call each function and save with df.to_csv(..., index=False) to overwrite.")
