# SmogShield

**Explainable AI for Air Pollution Forecasting in Lahore, Pakistan**

## Overview

SmogShield is an explainable-AI project that aims to help residents of Lahore understand air pollution and prepare for changing air-quality conditions. Deliverable 1 covers the problem definition and data foundations, Deliverable 2 the modelling and evaluation core, and Deliverable 3 the user-facing Streamlit application, prediction intervals, explanations, and the final merged report.

The project forecasts next-day PM2.5 concentrations and Air Quality Index (AQI) categories using historical air-quality and weather data, with calendar-based proxies standing in for traffic (no open, time-aligned Lahore traffic dataset was found).

In addition to forecasting, SmogShield aims to communicate prediction uncertainty, explain the main factors behind forecasts through feature-importance analysis, identify recurring pollution patterns, and highlight historically lower-pollution hours for outdoor activity planning.

## Problem Statement

Air pollution is a recurring challenge in Lahore, particularly during the winter smog season. Current air-quality readings describe present conditions but do not explain what is driving pollution or what tomorrow is likely to bring. SmogShield addresses this gap by combining forecasting, interpretable model outputs and historical pollution analysis in an accessible web application.

## Project objectives (overall vision)

- **Next-day forecasting:** forecast PM2.5 and AQI categories.
- **Model comparison:** compare several models against a persistence baseline.
- **Feature evaluation:** test whether weather and calendar features improve accuracy (ablation study).
- **Explainability:** identify the main factors behind predictions.
- **Uncertainty:** provide prediction intervals.
- **Pattern analysis:** examine recurring pollution patterns, including winter stagnant smog.
- **Outdoor planning:** identify historically lower-pollution hours.

### Scope of Deliverable 2

Deliverable 2 delivers the modelling and evaluation core: six-class next-day AQI
classification, next-day PM2.5 regression, and subgroup bias/error analysis
against persistence baselines. It uses the shared chronological split,
train-only preprocessing, and the scorable-row mask.

Explainability (via permutation importance and local sensitivity), prediction
intervals, the weather/calendar ablation study, seasonal pattern analysis,
lower-pollution-hour planning, and the web application are delivered in
Deliverable 3. They were **not** part of Deliverable 2, so the Deliverable 2
report should not be read as containing them. SHAP in particular was not
implemented; the application uses model-agnostic permutation importance and
one-at-a-time sensitivity instead.

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

## Modelling setup

After restoring `data/processed/features_daily.csv` from the shared Drive, or rebuilding it with `python data_scripts/09_build_features.py`, run the shared contract check from the repository root:

```bash
python -m src.check_setup
```

It prints the chronological split, feature count and per-split class support. The modelling rules are in [`docs/modeling_contract.md`](docs/modeling_contract.md); `src/data.py` and `src/evaluate.py` are the only shared data and metric implementations.

The final Deliverable 2 environment used Python 3.14.4, pandas 3.0.3,
scikit-learn 1.9.1, and matplotlib 3.11.2, captured from `python --version` and
`pip freeze` in the final-run environment. `requirements.txt` requires
`scikit-learn>=1.3`.

Run the metric tests after changing `src/evaluate.py`:

```bash
python -m unittest discover -s tests -t .
```

## Deliverable 2 model runs

After restoring `data/processed/features_daily.csv` from the shared Drive, or
rebuilding it with `python data_scripts/09_build_features.py`, the three
Deliverable 2 entry points are:

```bash
python -m src.classification    # six-class next-day AQI classification
python -m src.regression        # next-day PM2.5 regression
python -m src.bias_analysis     # subgroup bias and error analysis
```

All three use the shared split, scorable-row mask, and metric helpers in
`src/data.py` and `src/evaluate.py`. Generated metrics and charts are written
under `outputs/` and are intentionally not committed.

The run order matters: `classification` and `regression` create
`outputs/model_results/classification_predictions.csv` and
`regression_predictions.csv`; `bias_analysis` reads both files and must run
last. On the final machine, classification took about 7 minutes (427.92 s),
regression took about 38 seconds, and the bias analysis was quick.

## Deliverable 2 results

| Task | Selected model | Validation | Test | Persistence validation | Persistence test | Test skill vs persistence | Outcome |
|---|---|---|---:|---:|---:|---|---|
| Classification | HistGradientBoosting | macro F1 0.4941 | macro F1 0.4780 | macro F1 0.4766 | macro F1 0.5157 | — | did not beat persistence on test macro F1 |
| Regression | HistGradientBoosting | RMSE 45.6243 | RMSE 64.9198 | RMSE 51.5457 | RMSE 64.8175 | −0.16% | essentially tied with persistence on test RMSE |

Models were selected on validation only; test results were not used to change
the selection. For regression, HGB's test RMSE was 64.9198 versus persistence's
64.8175, a 0.10 RMSE gap. Ridge reached test RMSE **62.0956** (skill **+4.20%**),
but switching to Ridge after seeing test would violate the validation-only
selection rule. The report states this distinction directly.

Islamic-event subgroup coverage: the validation join contains 25 scorable
Ramadan rows and no scorable Eid rows. The feature table flags Eid on 10–12
April and 17–19 June 2024, but the AQI label is missing on those dates; all six
dates also lack a next-day target. The test join contains only the `Normal`
group. This is a label-coverage limitation, not a calendar-flag gap, and the
report makes no Eid-effect claim.

## Deliverable 3 application

Deliverable 3 (Track A) turns the frozen D2 pipelines into a working,
explainable application without retraining or reselecting the model. See
[`deliverable_3/README.md`](deliverable_3/README.md) for the full guide.

- **App:** `deliverable_3/app/app.py` — six inputs (date, PM2.5, AQI category,
  temperature, humidity, wind), next-day PM2.5 and AQI, two category views,
  80% interval, local sensitivity, global importance, seasonal defaults,
  winter-stagnation context, and an hourly profile.
- **Inference:** `deliverable_3/src/inference.py` — feature assembly, numeric AQI
  derived from PM2.5, interval construction, explanations, and guidance.
- **Artifacts:** refit with `python -m src.train_pipeline` from
  `deliverable_3`; no fitted binaries or processed data are committed.
- **Tests:** 14 tests pass; the Streamlit AppTest reports no exceptions, four
  charts, and three expanders.
- **Final report:** [`deliverable_3/report/report_d3.pdf`](deliverable_3/report/report_d3.pdf)
  — 47 pages merging D1, D2, and D3.
- **Launchers:** `setup_env.bat` (create `.venv` and install dependencies),
  `setup_models.bat` (refit), `run_app.bat` (start the app),
  `run_ablation.bat` (ablation study).

D3 adds an 80% prediction interval with **80.3% validation** and **77.0% test**
coverage. The honest headline is unchanged: regression essentially ties
persistence on test (RMSE 64.92 versus 64.82) and the classifier loses on
macro-F1 (0.478 versus 0.516). The application shows persistence on every
forecast and warns when the two category views disagree.

## Model weights and external artifacts

**No trained model binaries are stored in this repository.** The selected
classification and regression pipelines can be refit by the committed scripts
using fixed seeds from the shared feature table; the bias analysis trains
nothing and reads the saved prediction files. This keeps the repository well
below GitHub's 100 MB per-file limit. The `.gitignore` excludes `.joblib`,
`.pkl`, `.pt`, `.pth`, and `.onnx` model files.

The external download link for project data and, if exported, fitted model
weights is the shared Google Drive folder:

<https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>

The current submission does not include pretrained weights: the reproducible
code is the source of truth. If reviewers require fitted weight files, export
them to a `models/` subfolder in that Drive folder and update this README; do not
commit the binaries to Git. See [`models/README.md`](models/README.md).

## Repository Structure

```text
SmogShield/
├── README.md
├── DATA_SOURCES.md
├── requirements.txt
├── .gitignore
├── configs/             shared split, target and feature configuration
├── data_scripts/        Phase 2 code (01-09)
├── docs/                modelling contract for Deliverable 2
├── models/              model-weight policy; no binaries committed
├── notebooks/           exploratory notebooks
├── src/                 shared data/metrics, classification, regression and bias analysis
├── deliverable_3/       Phase 5-6 app, inference, tests, report (see its README)
├── tests/               unit tests for the shared metric helpers
├── outputs/
│   ├── eda_charts/      EDA figures used in the report
│   ├── model_charts/    Deliverable 2 figures (generated)
│   └── model_results/   Deliverable 2 metrics (generated)
└── reports/             LaTeX sources and the data collection log
```


## Team

- **Muhammad Haider Mughal (3006)** — D1 calendar construction and feature engineering; D2 six-class next-day AQI classification (`src/classification.py`).
- **Muhammad Ayub Butt (3076)** — D1 CAMS/ERA5 cleaning; D2 PM2.5 regression and subgroup bias analysis (`src/regression.py`, `src/bias_analysis.py`).
- **Abdul Rehman (3059)** — data collection and provenance, OpenAQ PM2.5 station cleaning and the AQI pipeline, shared modelling foundation (`src/data.py`, `src/evaluate.py`), integration, D2 report, D3 application/inference/report, and reproducibility checks.

## License

The project licence will be determined before public release. Third-party datasets remain subject to their own licensing and attribution requirements.
