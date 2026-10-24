# SmogShield

**Explainable AI for Air Pollution Forecasting in Lahore, Pakistan**

## Overview

SmogShield is an explainable AI-powered web application designed to help residents of Lahore understand air pollution and prepare for changing air-quality conditions.

The project forecasts next-day PM2.5 concentrations and Air Quality Index (AQI) categories using historical air-quality and weather data, with calendar-based proxies standing in for traffic (no open, time-aligned Lahore traffic dataset was found).

In addition to forecasting, SmogShield aims to communicate prediction uncertainty, explain the main factors behind forecasts through feature-importance analysis, identify recurring pollution patterns, and highlight historically lower-pollution hours for outdoor activity planning.

## Problem Statement

Air pollution is a recurring challenge in Lahore, particularly during the winter smog season. Current air-quality readings describe present conditions but do not explain what is driving pollution or what tomorrow is likely to bring. SmogShield addresses this gap by combining forecasting, interpretable model outputs and historical pollution analysis in an accessible web application.

## Objectives

- **Next-day forecasting:** forecast PM2.5 and AQI categories.
- **Model comparison:** compare several models against a persistence baseline.
- **Feature evaluation:** test whether weather and calendar features improve accuracy (ablation study).
- **Explainability:** identify the main factors behind predictions.
- **Uncertainty:** provide prediction intervals.
- **Pattern analysis:** examine recurring pollution patterns, including winter stagnant smog.
- **Outdoor planning:** identify historically lower-pollution hours.

## Target Audience

Residents of Lahore, pedestrians, cyclists and commuters planning outdoor activities, parents and schools, outdoor workers, and anyone interested in how weather relates to air pollution. SmogShield is not a medical tool and does not replace official health advisories.

## Data

**Datasets are not stored in this repository** (see `.gitignore`). Raw and cleaned files are on the shared Google Drive folder:
<https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>

Copy them into a local `data/` folder with this layout:

```text
data/
├── raw/
│   ├── ground_truth/   openaq_us_post_25135_hourly.csv
│   ├── weather/        open_meteo_era5_lahore_daily_2019_2026.csv
│   └── validation/     station_aqi_weather_lahore_daily_2019_2023.csv,
│                       modeled_cams_lahore_hourly_2023_2025.csv,
│                       modeled_cams_lahore_daily_2022_2026.csv, city_info.csv, data_dictionary.csv
└── processed/          (created by the scripts)
```

| Dataset | Role | Source |
|---|---|---|
| Hourly PM2.5, US Diplomatic Post Lahore | Main target | OpenAQ v3 API, sensor 25135 |
| Daily weather (ERA5) | Features | Open-Meteo Historical Weather API |
| Station AQI + weather, 2019-2023 | Cross-check | GitHub `taimoorbarlas15/Lahore_Air_Quality` |
| CAMS hourly and daily (modelled) | Comparison only | Kaggle `aadbutt/dataset`; Zenodo 10.5281/zenodo.18673725 |

Sources, licences, row counts and caveats for every file (including the ones we rejected) are in [`DATA_SOURCES.md`](DATA_SOURCES.md). Download evidence, hashes and exact requests are in `reports/data_collection_log.md`. Modelled (CAMS) data is never treated as ground truth.

## Running the pipeline

Install the requirements once, then run the scripts from the project root, in order:

```bash
pip install -r requirements.txt
python data_scripts/01_download_data.py        # optional: re-download sources (OpenAQ needs OPENAQ_API_KEY)
python data_scripts/02_clean_pm25_simple.py    # hourly + daily PM2.5
python data_scripts/03_pm25_aqi_and_eda.py     # AQI, category, 4 EDA charts
python data_scripts/04_clean_station_simple.py # station AQI + weather
# 05-07: clean ERA5 weather and the two CAMS files (see data_scripts/)
python data_scripts/08_build_calendar.py       # calendar table
python data_scripts/09_build_features.py       # feature table + 2 EDA charts
```

Cleaning rules: raw files are never edited; the target is never filled; extreme values are flagged, not deleted; a day is usable only with at least 18 valid hours.

## Repository Structure

```text
SmogShield/
├── README.md
├── DATA_SOURCES.md
├── requirements.txt
├── .gitignore
├── data_scripts/        Phase 2 code (01-09)
├── outputs/
│   └── eda_charts/      EDA figures used in the report
└── reports/             LaTeX sources and the data collection log
```


## Team

- **Muhammad Haider Mughal (3006)** — calendar construction and feature engineering: public holidays, Ramadan/Eid flags, lagged and rolling features, and the EDA charts.
- **Muhammad Ayub Butt (3076)** — data cleaning: CAMS daily/hourly validation data and ERA5 weather, including timezone and unit standardisation.
- **Abdul Rehman (3059)** — data collection and provenance, OpenAQ PM2.5 station cleaning and the AQI pipeline, report integration and reproducibility checks.

## License

The project licence will be determined before public release. Third-party datasets remain subject to their own licensing and attribution requirements.
