# 03_pm25_aqi_and_eda.py
# Finishes the daily PM2.5 file: model-ready column, AQI, AQI category, and EDA charts.
#
# Input : data/processed/pm25_daily.csv        (made by 02_clean_pm25_simple.py)
# Output: data/processed/pm25_daily_clean.csv
#         outputs/eda_charts/*.png             (four charts for the report)
# Run from the main project folder:  python data_scripts/03_pm25_aqi_and_eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

daily = pd.read_csv("data/processed/pm25_daily.csv", parse_dates=["date"])

# 1. model-ready PM2.5: only usable days (at least 18 valid hours) keep a value
daily["pm25_clean"] = daily["pm25_mean"].where(daily["usable_day"])

# 2. AQI and category from PM2.5 (US EPA PM2.5 breakpoints, 2012 version)
breakpoints = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
               (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
names = ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy",
         "Very Unhealthy", "Hazardous (301-400)", "Hazardous (401-500)"]

def to_aqi(c):
    if pd.isna(c):
        return np.nan
    c = np.floor(c * 10) / 10                      # EPA truncates to 1 decimal
    for low, high, aqi_low, aqi_high in breakpoints:
        if c <= high:
            return round(aqi_low + (aqi_high - aqi_low) * (c - low) / (high - low))
    return 500

def to_category(aqi):
    if pd.isna(aqi):
        return np.nan
    for (low, high, aqi_low, aqi_high), name in zip(breakpoints, names):
        if aqi <= aqi_high:
            return name
    return names[-1]

daily["aqi"] = daily["pm25_clean"].map(to_aqi)
daily["aqi_category"] = daily["aqi"].map(to_category)
daily.to_csv("data/processed/pm25_daily_clean.csv", index=False)
print("Saved pm25_daily_clean.csv:", daily.shape, "| usable days:", int(daily["usable_day"].sum()))
print(daily["aqi_category"].value_counts().reindex(names))

# 3. EDA charts (saved as small PNG files for the report)
os.makedirs("outputs/eda_charts", exist_ok=True)

plt.figure(figsize=(7, 4))
plt.hist(daily["pm25_clean"].dropna(), bins=40)
plt.xlabel("Daily mean PM2.5 (µg/m³)"); plt.ylabel("Days"); plt.title("Distribution of daily PM2.5")
plt.tight_layout(); plt.savefig("outputs/eda_charts/pm25_distribution.png", dpi=150); plt.close()

monthly = daily.set_index("date")["pm25_clean"].resample("MS").mean()
plt.figure(figsize=(9, 4))
plt.plot(monthly.index, monthly.values)
plt.ylabel("Monthly mean PM2.5 (µg/m³)"); plt.title("Monthly mean PM2.5")
plt.tight_layout(); plt.savefig("outputs/eda_charts/pm25_monthly.png", dpi=150); plt.close()

coverage = daily.set_index("date")["usable_day"].resample("MS").mean() * 100
plt.figure(figsize=(9, 4))
plt.bar(coverage.index, coverage.values, width=25)
plt.ylabel("% usable days"); plt.title("Data coverage per month")
plt.tight_layout(); plt.savefig("outputs/eda_charts/coverage_by_month.png", dpi=150); plt.close()

counts = daily["aqi_category"].value_counts().reindex(names)
plt.figure(figsize=(9, 4))
plt.bar(range(len(counts)), counts.values)
plt.xticks(range(len(counts)), counts.index, rotation=30, ha="right")
plt.ylabel("Days"); plt.title("AQI category distribution")
plt.tight_layout(); plt.savefig("outputs/eda_charts/aqi_class_distribution.png", dpi=150); plt.close()
print("Charts saved in outputs/eda_charts")
