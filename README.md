# SmogShield

**Explainable AI for Air Pollution Forecasting in Lahore, Pakistan**

## Overview

SmogShield is an explainable AI-powered web application designed to help residents of Lahore understand air pollution and prepare for changing air-quality conditions.

The project aims to forecast next-day PM2.5 concentrations and Air Quality Index (AQI) categories using historical air-quality, weather, and traffic-related data. When direct traffic data is unavailable, calendar-based proxies may be used.

In addition to forecasting, SmogShield aims to communicate prediction uncertainty, explain the main factors influencing forecasts through feature-importance analysis, identify recurring pollution patterns, and highlight historically lower-pollution hours for outdoor activity planning.

## Problem Statement

Air pollution is a recurring environmental challenge in Lahore, particularly during the winter smog season. Current air-quality readings provide information about present conditions but may not fully explain the factors contributing to pollution or indicate what conditions could be like the following day.

SmogShield aims to address this gap by combining air-quality forecasting, interpretable model outputs, and historical pollution analysis in an accessible web application.

## Objectives

- **Next-Day Forecasting:** Forecast PM2.5 concentrations and AQI categories.
- **Model Comparison:** Compare multiple machine-learning models to evaluate forecasting performance.
- **Feature Evaluation:** Investigate whether weather and traffic-related features improve prediction accuracy.
- **Explainability:** Identify the main factors influencing model predictions using feature-importance analysis.
- **Uncertainty Estimation:** Provide prediction intervals to communicate forecast uncertainty.
- **Pollution Pattern Analysis:** Examine recurring pollution patterns, including wintertime stagnant-smog conditions.
- **Outdoor Planning:** Identify historically lower-pollution hours to support more informed outdoor activity planning.

## Target Audience

SmogShield is intended for:

- Residents of Lahore seeking accessible air-quality information.
- Pedestrians, cyclists, and people planning outdoor activities.
- Commuters and visitors unfamiliar with local pollution patterns.
- Individuals interested in understanding air pollution and its relationship with weather and other relevant factors.

## Data Sources

The project uses datasets obtained from sources including Kaggle, AQICN, and Zenodo. Individual dataset sources, licenses, and collection details will be documented as they are verified.

### Processed Datasets

- `lahore_us_post_pm25_daily_2019_2025.csv`
- `lahore_us_post_pm25_hourly_2019_2025.csv`

### Raw Datasets

**Ground Truth**
- `openaq_us_post_25135_hourly.csv`

**Validation**
- `city_info.csv`
- `data_dictionary.csv`
- `modeled_cams_lahore_daily_2022_2026.csv`
- `modeled_cams_lahore_hourly_2023_2025.csv`
- Station-level AQI and weather data — filename to be verified.

**Weather**
- `open_meteo_era5_lahore_daily_2019_2026.csv`

### Archived Datasets

The `archive_unused` directory contains:

- `lahore_complete_data_july_to_dec_2024.csv`
- `pakistan_air_quality_annual_summary.csv`
- `weather.csv`

These files are retained for reference and are not necessarily included in the final modeling pipeline.

## Methodology

The planned project workflow consists of the following stages:

1. **Data Acquisition:** Document dataset sources and collection methods.
2. **Data Cleaning:** Inspect missing values, duplicate records, inconsistent data, and potential outliers.
3. **Feature Engineering:** Prepare relevant air-quality, weather, temporal, and traffic-related features.
4. **Exploratory Data Analysis:** Analyze distributions, correlations, and temporal pollution patterns.
5. **Model Development:** Compare suitable machine-learning models for PM2.5 forecasting and AQI-category prediction.
6. **Model Evaluation:** Evaluate predictive performance and assess the contribution of additional features.
7. **Explainability and Uncertainty:** Analyze feature importance and develop prediction intervals.
8. **Application Development:** Present forecasts and supporting explanations through a web interface.

The final algorithms, evaluation metrics, and implementation details will be determined during development.

## Technology Stack

The technology stack is to be finalized. Python is planned for data processing and machine-learning development.

## Repository Structure

The repository will organize project documentation, data-processing scripts, analysis outputs, and reports.

```text
SmogShield/
├── README.md
├── .gitignore
├── DATA_SOURCES.md
├── data_scripts/
├── outputs/
│   └── eda_charts/
└── reports/
```

Large raw datasets will be managed separately where appropriate. Dataset access, attribution, and licensing requirements will be documented as they are verified.

## Project Status

**Current stage:** Problem definition and data foundations.

- [x] Initial project objectives defined.
- [x] Initial dataset inventory identified.
- [ ] Dataset sources and licensing documented.
- [ ] Data-cleaning scripts implemented.
- [ ] Feature engineering completed.
- [ ] Exploratory data analysis completed.
- [ ] Machine-learning models selected and evaluated.
- [ ] Forecast explanations and uncertainty estimates implemented.
- [ ] Web application developed and evaluated.

## Expected Outcome

SmogShield aims to make air-quality forecasting more understandable by combining next-day predictions, explanations of influential factors, uncertainty estimates, and historical pollution analysis.

The project is intended to support informed outdoor planning. Forecasts and associated guidance will require appropriate evaluation before being relied upon for real-world decisions.

## Team

Team information will be added as the project documentation is finalized.

## License

The project license will be determined before public release. Third-party datasets remain subject to their respective licensing and attribution requirements.
