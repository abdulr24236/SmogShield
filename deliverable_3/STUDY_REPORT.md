# SmogShield Lahore — Technical Study Report

**Deliverable 3, Track A — next-day air-quality forecasting app**

Version 1.2 · 11 November 2026

This document explains, in plain language, how the project works end to end: what the user enters, what the system does with it, what every major term means, how those terms apply in this specific project, and which file does what. It also answers one important question directly: **what changed after Deliverable 2?**

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [The problem and the question](#2-the-problem-and-the-question)
3. [The pipeline in one picture](#3-the-pipeline-in-one-picture)
4. [Stage 1 — Data](#4-stage-1--data)
5. [Stage 2 — Features](#5-stage-2--features)
6. [Stage 3 — Models](#6-stage-3--models)
7. [Stage 4 — Inference and explanations](#7-stage-4--inference-and-explanations)
8. [Stage 5 — The web app](#8-stage-5--the-web-app)
9. [Results](#9-results)
10. [What changed after Deliverable 2?](#10-what-changed-after-deliverable-2)
11. [File-by-file map](#11-file-by-file-map)
12. [How to run and reproduce](#12-how-to-run-and-reproduce)
13. [Limitations and honest framing](#13-limitations-and-honest-framing)
14. [Viva-ready answers](#14-viva-ready-answers)
15. [Glossary](#15-glossary)

---

## 1. Executive summary

SmogShield is an **explainable, next-day air-quality forecasting app for Lahore**. The user provides six pieces of information about today — date, PM2.5, AQI category, temperature, humidity, and wind speed. The app returns:

- tomorrow's predicted **PM2.5 concentration**;
- tomorrow's **AQI and AQI category**, in two independent views;
- an **80% prediction interval**;
- a **persistence baseline** for comparison;
- a **local explanation** of which inputs moved this forecast;
- a **global explanation** of what drives forecasts overall;
- historical context: **winter stagnation**, **seasonal averages**, and the **cleanest hours of the day**;
- cautious, non-medical **outdoor-activity guidance**.

The modelling core is intentionally frozen from Deliverable 2. On the held-out test set, the regression model **essentially ties persistence** (RMSE 64.92 vs 64.82 µg/m³), and the classifier is **weaker than persistence** (macro-F1 0.478 vs 0.516). The project reports this honestly. Its contribution is the end-to-end explainable system and the rigorous evaluation, not a claim of beating a strong baseline.

---

## 2. The problem and the question

Lahore has severe seasonal air pollution, especially in winter. A current PM2.5 reading tells residents what the air is like now, but not what tomorrow is likely to bring, how uncertain that forecast is, or which factors are driving it.

The project asks:

1. Can we forecast **next-day PM2.5** and **next-day AQI category** from today's air-quality, weather, and calendar conditions?
2. Can we quantify the uncertainty of that forecast?
3. Can we explain why the model produced a given forecast?
4. Do weather and calendar features actually improve accuracy over a simple baseline?
5. Can we identify recurring pollution patterns and historically cleaner hours for outdoor planning?

The app is the user-facing answer to questions 1–3 and 5. Question 4 is answered by the ablation study, including a negative result.

---

## 3. The pipeline in one picture

```
Raw data
  ├─ OpenAQ PM2.5 (sensor 25135, Lahore US Diplomatic Post)
  ├─ Open-Meteo / ERA5 daily weather
  ├─ Calendar and Islamic-event dates
  └─ CAMS modelled data (validation context)
        │
        ▼
Cleaning and feature building
  └─ data/processed/features_daily.csv  (2,113 days × 63 columns)
        │
        ▼
Chronological split with a 7-day purge
  ├─ train: 917 scorable days
  ├─ validation: 244 scorable days
  └─ test: 222 scorable days
        │
        ▼
46 raw model features (80 encoded columns)
  ├─ air quality and PM2.5 history
  ├─ weather
  ├─ calendar and events
  └─ data-quality flags
        │
        ├──────────────────────────────┐
        ▼                              ▼
HGB regression                  HGB classifier
(next-day PM2.5)                (6-class AQI category)
        │                              │
        ├─ quantile HGB 10 / 90 ───────┤
        │   (80% interval)             │
        ▼                              ▼
        └──────────► inference.py ◄────┘
                         │
                         ▼
                    app.py (Streamlit)
```

---

## 4. Stage 1 — Data

### 4.1 Sources

| Source | What it provides | Role |
|---|---|---|
| OpenAQ sensor 25135 | Hourly PM2.5 at the Lahore US Diplomatic Post | Ground-truth air quality |
| Open-Meteo / ERA5 | Daily temperature, humidity, dew point, pressure, wind, rain, solar radiation, cloud cover | Weather features |
| Calendar tables | Month, day-of-year, weekday, Ramadan, Eid, public holidays | Seasonality and traffic proxy |
| CAMS modelled data | Modelled pollutant and weather fields | Context and validation, not the main target |

No open, time-aligned Lahore traffic dataset was found. **Calendar features — weekday, weekend, and public holidays — stand in as a traffic proxy.** This is a documented limitation, not a hidden assumption.

### 4.2 The daily table

The shared feature table is:

- **2,113 daily rows**
- **63 columns**
- **2019-05-23 to 2025-03-04**

Only rows with a valid *today* PM2.5 can be scored, because the model needs today's value to build lags and to compute the persistence baseline. That gives:

| Split | Dates | Scorable rows |
|---|---|---:|
| Train | up to 2023-06-30 | 917 |
| Validation | 2023-07-08 to 2024-06-30 | 244 |
| Test | 2024-07-08 onward | 222 |

The **7-day gap** between splits is deliberate. The target is tomorrow's value, and the longest rolling feature uses a 7-day window. Without the gap, information from the end of one split could leak into the start of the next.

### 4.3 Targets

The table contains three target columns:

- `target_next_pm25` — tomorrow's PM2.5, used by regression;
- `target_next_log` — tomorrow's log PM2.5, retained for completeness;
- `target_next_aqi_category` — tomorrow's AQI category, used by classification.

All target columns are removed from the feature matrix.

### 4.4 AQI categories

The raw data has seven US-EPA categories:

1. Good
2. Moderate
3. Unhealthy for Sensitive
4. Unhealthy
5. Very Unhealthy
6. Hazardous (301–400)
7. Hazardous (401–500)

For classification, **Good and Moderate are merged** into `Good/Moderate`. The reason is support: Good has only 9 training rows and none in validation or test, so a seven-class model cannot learn or evaluate that class reliably. The final model therefore has **six classes**.

The raw seven-label vocabulary is still used in the app's input form and in the EPA conversion, so the user sees the familiar categories.

### 4.5 AQI conversion

AQI is computed from PM2.5 using the **US-EPA 2012 breakpoints**:

| PM2.5 (µg/m³) | AQI | Category |
|---:|---:|---|
| 0.0–12.0 | 0–50 | Good |
| 12.1–35.4 | 51–100 | Moderate |
| 35.5–55.4 | 101–150 | Unhealthy for Sensitive |
| 55.5–150.4 | 151–200 | Unhealthy |
| 150.5–250.4 | 201–300 | Very Unhealthy |
| 250.5–350.4 | 301–400 | Hazardous (301–400) |
| 350.5–500.4 | 401–500 | Hazardous (401–500) |

The regression-implied category in the app uses exactly this table, so it is comparable with the training labels and with the classifier.

---

## 5. Stage 2 — Features

The model uses **46 raw features**, which expand to **80 columns after one-hot encoding** (`aqi_category` becomes multiple indicator columns), grouped as follows.

### 5.1 Air quality and history

- `pm25` — today's PM2.5
- `pm25_log` — log(1 + PM2.5)
- `pm25_lag1`, `pm25_lag2`, `pm25_lag3` — the previous three days
- `pm25_roll3`, `pm25_roll7` — 3-day and 7-day rolling means
- `pm25_diff1` — day-to-day change
- `aqi` — today's AQI value
- `aqi_category` — today's category (categorical)
- `valid_hours`, `usable_day` — data-quality flags

### 5.2 Weather

- Temperature: `temp_mean_c`, `temp_max_c`, `temp_min_c`, `temp_mean_c_lag1`
- Humidity: `humidity_mean_pct`, `humidity_mean_pct_lag1`
- Dew point: `dew_point_mean_c`
- Pressure: `pressure_mean_hpa`, `pressure_mean_hpa_lag1`
- Wind: `wind_speed_mean_kmh`, `wind_speed_max_kmh`, `wind_speed_mean_kmh_lag1`, `wind_dir_dominant_deg`
- Rain: `precip_sum_mm`, `rain_sum_mm`
- Solar and cloud: `shortwave_sum_mj_m2`, `cloud_cover_mean_pct`

### 5.3 Calendar and events

- `month`, `day_of_year`, `weekday`, `is_weekend`
- Cyclical encodings: `month_sin`, `month_cos`, `doy_sin`, `doy_cos`, `weekday_sin`, `weekday_cos`
- Season dummies: `season_spring`, `season_summer`, `season_autumn`, `season_winter`
- Events: `is_ramadan`, `is_eid`, `is_public_holiday`

### 5.4 Why these features

- **Lags and rolling means** capture the strong day-to-day persistence of PM2.5.
- **Weather** captures ventilation and dispersion: wind, rain, pressure, and temperature.
- **Calendar features** capture seasonality and the weekly rhythm of traffic and human activity.
- **Events** capture unusual emission or activity patterns during Ramadan, Eid, and public holidays.

The table excludes `season` and `year` as raw columns: `season` is represented by the dummies, and `year` is excluded because training ends in 2023 while validation and test continue to 2025, so a tree model cannot extrapolate it safely. Columns ending in `_scaled` are also excluded because they were scaled on the full dataset and would leak validation/test statistics; scaling is refit on the training split only.

---

## 6. Stage 3 — Models

### 6.1 What is HGB?

**HGB** stands for **Histogram-based Gradient Boosting**.

Gradient boosting builds a sequence of small decision trees. Each new tree is trained to correct the errors of the trees already built. The prediction is the sum of all trees, shrunk by a learning rate so that no single tree dominates.

The “histogram-based” part means continuous features are first binned into a small number of buckets (for example, 256). This makes training much faster and more memory-efficient than exact split search, especially on tabular data. Scikit-learn's `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier` also handle missing values natively and do not require feature scaling.

**Why HGB here?**

- The data is tabular, not images or text.
- Relationships are nonlinear and involve interactions (for example, low wind plus winter plus high PM2.5).
- It handles missing values and mixed feature types gracefully.
- It is fast enough to refit in seconds.
- It was selected on the validation split in Deliverable 2, not on the test set.

**What HGB cannot do:**

- It cannot extrapolate beyond the range of the training data. A tree predicts a leaf value, not a trend line.
- Its predictions are **piecewise constant**. A small change in an input may not change the prediction at all if it stays inside the same leaf. This matters for the local sensitivity panel (see Section 7.4).
- It can overfit small datasets if trees are too deep or too numerous.

### 6.2 Regression model — next-day PM2.5

The regression model predicts a continuous number: tomorrow's PM2.5 in µg/m³.

Frozen settings:

| Setting | Value |
|---|---|
| Model | `HistGradientBoostingRegressor` |
| Learning rate | 0.05 |
| Max depth | 3 |
| Max iterations | 200 |
| Random seed | 42 |

These settings were selected on the validation split in Deliverable 2 and are frozen for Deliverable 3.

### 6.3 Classification model — next-day AQI category

The classifier predicts one of six categories directly. It does not predict PM2.5 first; it is a separate model trained on the category labels.

Frozen settings:

| Setting | Value |
|---|---|
| Model | `HistGradientBoostingClassifier` |
| Learning rate | 0.1 |
| Max leaf nodes | 31 |
| Min samples per leaf | 10 |
| Max iterations | 100 |
| L2 regularization | 0.1 |
| Class weight | `balanced` |
| Early stopping | `False` |
| Random seed | 42 |

`class_weight="balanced"` compensates for the fact that `Unhealthy` is much more common than the hazardous classes. Without it, the model collapses toward the majority class even more strongly.

`predict_proba` returns a probability for each of the six classes. The app displays these as bars. The predicted class is the **arg-max** — the class with the highest probability.

**Known weakness:** on the test set, the classifier's macro-F1 is **0.478**, below persistence's **0.516**. It is biased toward `Unhealthy`. The app therefore treats the regression-implied category as the more defensible view and warns when the two disagree.

### 6.4 The persistence baseline

Persistence is the simplest possible forecast:

> Tomorrow will be the same as today.

In symbols:

```
prediction(t+1) = observed(t)
```

For classification, tomorrow's category is today's category. For PM2.5, tomorrow's value is today's value.

Persistence is a **strong baseline** for one-day-ahead air quality because PM2.5 is highly autocorrelated: polluted days tend to be followed by polluted days, and clean days by clean days. Weather also changes slowly relative to a single day. Any model that claims to add value must beat this baseline.

In this project:

| Task | Persistence validation | Persistence test |
|---|---:|---:|
| Regression RMSE | 51.55 | **64.82** |
| Classification macro-F1 | 0.4766 | **0.5157** |

The regression model's test RMSE is **64.92**, essentially the same as persistence. The classifier's test macro-F1 is **0.478**, below persistence. This is the honest headline.

### 6.5 Model comparison and selection

Deliverable 2 compared several models:

| Task | Models compared | Selected on validation |
|---|---|---|
| Regression | Persistence, Ridge, HGB | HGB (RMSE 45.62 vs Ridge 45.69) |
| Classification | Majority, Persistence, Logistic Regression, HGB, MLP | HGB (macro-F1 0.4941) |

The validation margins were small. On test, Ridge actually beat HGB for regression (62.10 vs 64.92), and persistence beat the classifier. This is **selection variance**, not evidence that the modelling pipeline is broken. The project reports it rather than switching models after seeing test scores.

### 6.6 Quantile models and the 80% interval

A point forecast says “about 129 µg/m³.” An interval says “we expect the true value to fall between 52 and 160 µg/m³, with 80% confidence.”

The interval is produced by two additional HGB models trained with **quantile loss**:

- one for the 10th percentile (lower bound);
- one for the 90th percentile (upper bound).

Together they form an **80% central prediction interval**. The bounds are then widened by a calibration factor chosen on the validation split so that the empirical coverage is close to 80%. On this data the factor is **1.00**, so no widening was needed. Validation coverage is **80.3%**; test coverage is **77.0%**.

The test coverage is reported, never used to tune the factor. That distinction matters: tuning on test would make the reported coverage optimistic.

---

## 7. Stage 4 — Inference and explanations

### 7.1 Feature assembly

The app collects only six inputs. `inference.py` builds the full 46-feature raw row (80 encoded columns) in four steps:

1. Start from **training medians** for all numeric features and **training modes** for categorical features.
2. Map the friendly inputs onto the model's real column names:
   - PM2.5 → `pm25`, `pm25_log`, numeric `aqi` (via the EPA conversion), lags 1–3, rolling 3/7, difference = 0;
   - temperature → mean, max, min, and lag;
   - humidity → mean and lag;
   - wind → mean, max, and lag;
   - date → month, day-of-year, weekday, weekend, sin/cos terms, season dummies;
   - AQI category → `aqi_category`.
3. Set the lags and rolling windows equal to today's PM2.5 — a **steady-state assumption**. The app does not know the actual previous days, so it assumes the current level has been stable.
4. Apply any explicit feature-column overrides last.

This is why the app is described as an **illustrative single-day forecast**, not a production data feed. The 13 features the app does not collect or derive — dew point, pressure, wind direction, rain, solar radiation, cloud cover, holiday flags, and data-quality flags — use training-set medians or modes. They are plausible values, not live observations.

### 7.2 Prediction

`predict()` returns:

- `predicted_next_pm25` — the regression output;
- `predicted_aqi` — the AQI implied by that PM2.5;
- `predicted_category_from_pm25` — the implied category;
- `predicted_next_aqi_category` — the classifier's arg-max;
- `probabilities` — the classifier's six class probabilities;
- `interval` — the calibrated 80% interval;
- `persistence` — tomorrow = today;
- `explanation` — local sensitivity;
- `guidance` — cautious outdoor advice;
- `n_features` — 80 encoded features (46 raw inputs; one-hot encoding expands `aqi_category`).

### 7.3 Global explanation — permutation importance

Permutation importance answers: **which features does the model rely on overall?**

The method shuffles one feature at a time in the validation set, measures how much the RMSE gets worse, and repeats. A large increase means the model relies on that feature.

Top regression features in this project:

| Feature | Increase in validation RMSE |
|---|---:|
| `pm25` | 14.51 |
| `aqi` | 4.20 |
| `day_of_year` | 4.15 |
| `pm25_lag1` | 2.93 |
| `wind_dir_dominant_deg` | 2.50 |
| `doy_cos` | 1.36 |
| `month_cos` | 1.33 |
| `pm25_roll3` | 1.22 |

Caveat: permutation importance can be misleading when features are correlated. Shuffling `pm25` while leaving its lags intact breaks a strong relationship, so the importance can overstate the unique contribution of any one correlated feature.

### 7.4 Local explanation — one-at-a-time sensitivity

The local explanation answers: **for this specific forecast, what happens if I change one input?**

The app perturbs four inputs by a fixed amount:

| Input change | Delta |
|---|---:|
| PM2.5 | +25 µg/m³ |
| Temperature | +5 °C |
| Humidity | +10 percentage points |
| Wind speed | +5 km/h |

It rebuilds the feature row, predicts again, and reports the change in the PM2.5 forecast. The bars are sorted by absolute effect.

This is **not causal inference**. It is a local finite difference: “holding everything else fixed, this input change moves the forecast by this much.”

**Why a larger PM2.5 step?** HGB predictions are piecewise constant, so a small perturbation can stay inside the same leaf and report **0.0** even when the feature matters globally. The app therefore uses a **+25 µg/m³** stress test for PM2.5; the other inputs keep their fixed steps. This removes most zero-effect artifacts, but the result is still a local finite difference, not a smooth slope or a causal effect. A Shapley-value method would give a more principled decomposition and is future work.

### 7.5 Guidance

The app maps the regression-implied category to cautious, non-medical advice:

| Category | Headline |
|---|---|
| Good/Moderate | Air quality is relatively favourable. |
| Unhealthy for Sensitive | Sensitive groups should take care. |
| Unhealthy | Consider reducing outdoor exertion. |
| Very Unhealthy | Avoid prolonged outdoor activity. |
| Hazardous (301–400) | Avoid outdoor activity. |
| Hazardous (401–500) | Avoid outdoor activity. |

Every guidance message carries a disclaimer: it is not a substitute for official health advice.

### 7.6 Two category views

The app deliberately shows **two** category estimates:

1. **Classifier arg-max** — the six-class model's direct prediction.
2. **Regression-implied category** — convert the predicted PM2.5 through the EPA breakpoints.

They can disagree. When they do, the app warns and says the implied category is the more defensible view, because the classifier was weaker than persistence on test. Showing both is more honest than hiding the disagreement.

---

## 8. Stage 5 — The web app

### 8.1 Inputs

| Input | Unit | Default | Notes |
|---|---|---:|---|
| Today's date (feature day) | date | today | Calendar features come from this date; the forecast is for the next day |
| Today's PM2.5 | µg/m³ | Historical monthly median (e.g., 234.2 in November) | Also sets lags, rolling windows, and numeric AQI |
| Today's AQI category | 7 raw labels | Derived from the default PM2.5 (e.g., Very Unhealthy in November) | Categorical feature |
| Temperature | °C | 22 | Mean, max, min, lag |
| Relative humidity | % | 60 | Mean and lag |
| Wind speed | km/h | 7 | Mean, max, lag |

The initial PM2.5 and category defaults come from the historical monthly medians, so the app opens on a seasonally plausible scenario rather than a fixed clean-day value. They are defaults, not live observations; the user can override both.

### 8.2 Outputs

- Predicted next-day PM2.5 (one decimal).
- AQI implied by PM2.5.
- Implied AQI category.
- Classifier category and confidence.
- Persistence PM2.5 and category.
- 80% prediction interval.
- Category probability bars.
- “Why this forecast?” sensitivity chart.
- “What drives forecasts overall” permutation-importance chart.
- “Winter stagnation pattern” expander.
- “Best time outdoors” expander.
- Guidance banner.
- Disagreement warning when the two category views differ.

### 8.3 Worked examples

These are actual outputs from the current model.

| | Example A: autumn | Example B: winter smog |
|---|---|---|
| Feature date | 10 Nov 2026 | 15 Dec 2026 |
| Today's PM2.5 | 150 µg/m³ | 250 µg/m³ |
| Today's category | Unhealthy | Very Unhealthy |
| Temperature | 22 °C | 12 °C |
| Humidity | 60% | 75% |
| Wind | 7 km/h | 3 km/h |
| **Predicted PM2.5** | **167.4** | **280.0** |
| AQI | 218 | 330 |
| Implied category | Very Unhealthy | Hazardous (301–400) |
| Classifier category | Very Unhealthy (98.9%) | Very Unhealthy (93.3%) |
| 80% interval | 103.6–248.5 | 150.5–381.2 |
| Persistence | 150 | 250 |
| Largest sensitivity | Temperature +5 °C → +14.7 | Wind +5 km/h → −36.5 |
| PM2.5 +25 µg/m³ | +2.5 | +32.3 |

Example B is the more interesting one: the classifier says `Very Unhealthy`, but the regression-implied category is `Hazardous (301–400)`. The app shows the disagreement warning. The wind sensitivity is large and negative, which matches the physical intuition that stronger wind disperses pollution.

A third example illustrates a caveat. The November default is the historical median (234.2 µg/m³), but if you override it with a clean value of 60 µg/m³ on a November date, the prediction is about **124–129 µg/m³** (128.9 on 10 Nov; 124.1 on 11 Nov). The model sees November seasonality and defaults for uncollected weather features, and it expects higher pollution than the clean input suggests. The interval is wide (about 51.6–160.0 on 10 Nov) and contains 60. This is a consequence of the six-input, defaults-heavy design, and it is why the app calls itself illustrative.

---

## 9. Results

### 9.1 Regression — next-day PM2.5

| Model | Validation RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Persistence | 51.55 | **64.82** | 39.98 | 0.712 |
| Ridge | 45.69 | **62.10** | 38.48 | 0.736 |
| HGB (selected) | **45.62** | **64.92** | 40.77 | 0.711 |

The HGB wins on validation by a hair and ties persistence on test. Ridge's test advantage is real but was not the validation-selected model; switching to it now would be post-hoc test selection.

### 9.2 Classification — next-day AQI category

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| Majority | 0.0776 | 0.0858 | 0.347 |
| Persistence | 0.4766 | **0.5157** | 0.568 |
| Logistic Regression | 0.4665 | 0.4601 | 0.523 |
| HGB (selected) | **0.4941** | **0.4780** | 0.586 |

The classifier beats persistence on validation but loses on test. The gap is documented, and the app recommends the implied category.

### 9.3 Prediction intervals

| Split | Coverage | Nominal |
|---|---:|---:|
| Validation | 80.3% | 80% |
| Test | 77.0% | 80% |

Calibration scale: 1.00. Test coverage is reported only.

### 9.4 Ablation — do weather and calendar help?

| Feature set | Regression val RMSE | Regression test RMSE | Classification val macro-F1 | Classification test macro-F1 |
|---|---:|---:|---:|---:|
| History only | 52.18 | 63.88 | 0.4663 | 0.4715 |
| History + calendar | 49.97 | 65.16 | 0.4319 | 0.5155 |
| Full (history + calendar + weather) | **45.62** | **64.92** | **0.4941** | **0.4780** |

The full model is best on validation for both tasks, but not consistently on test. Calendar features improve classification test performance but worsen regression test performance. The honest conclusion: **there is no consistent held-out gain from adding weather and calendar features on this dataset.** The ablation is descriptive, not a model-selection step.

### 9.5 Winter stagnation pattern

A winter day (December–February) is labelled **stagnant** when:

- its mean wind speed is in the winter lower quartile, and
- it recorded at most 0.5 mm of rain.

Results:

- **24.6%** of winter days are stagnant (100 of 407).
- Mean PM2.5 on stagnant days: **294.9 µg/m³**.
- Mean on better-ventilated days: **193.7 µg/m³**.
- Difference: **+52%**.

Seasonal means:

| Season | Mean PM2.5 |
|---|---:|
| Winter | 218.6 |
| Autumn | 146.0 |
| Spring | 77.7 |
| Summer | 55.1 |

These are descriptive statistics, not a dispersion model.

### 9.6 Best time outdoors

From 36,820 valid hourly observations (2019-05-23 to 2025-02-18):

| Window | Mean PM2.5 |
|---|---:|
| 14:00–16:59 | **85.2** |
| 22:00 (worst hour) | **138.4** |

The afternoon minimum holds in every season, but winter's best window still averages **155.4 µg/m³**. Timing helps; it does not make winter air safe.

---

## 10. What changed after Deliverable 2?

Short answer:

- **The trained model and its reported scores: no change.** The Deliverable 3 refit reproduces Deliverable 2 exactly.
- **The codebase: yes, substantial additions.** Deliverable 3 is a new application and explainability layer built around the frozen model.

### 10.1 What did not change

| Item | Status |
|---|---|
| `src/data.py`, `src/classification.py`, `src/regression.py`, `src/evaluate.py`, `src/bias_analysis.py` | Unchanged; no modified tracked files |
| `configs/base.yaml` | Unchanged |
| Deliverable 2 report (`final_work_deliverable_2/report_d2.tex` / `.pdf`) | Unchanged |
| Deliverable 2 metrics | Reproduced exactly by the D3 refit |
| Last tracked commit | `56dea73`; no tracked modifications since |

The exact reproduction is a strength: it shows the D3 artifacts are not a different model with different numbers.

### 10.2 What is new in Deliverable 3

| Area | New files / features |
|---|---|
| Web app | `deliverable_3/app/app.py` |
| Inference layer | `deliverable_3/src/inference.py` |
| Refit and artifact pipeline | `deliverable_3/src/train_pipeline.py` |
| Ablation | `deliverable_3/src/ablation.py` |
| Hourly profile | `deliverable_3/src/hourly.py` |
| Tests | `deliverable_3/tests/test_inference.py` |
| Docs and dependencies | `README.md`, `requirements.txt` |
| Launchers | `run_app.bat`, `setup_models.bat`, `run_ablation.bat` |
| New analysis outputs | 80% intervals, permutation importance, stagnation, seasonal means, monthly medians, hourly profile |

### 10.3 Reproducibility and inference fixes

During the D3 refit, the classifier initially did not match Deliverable 2. The cause was two missing settings:

- `class_weight="balanced"`
- `early_stopping=False`

Adding them to the D3 training script made the refit match D2 exactly. **Deliverable 2 itself was not edited.** The fix was to the new D3 script, so the D2 code and report remain the reference.

Two later inference fixes and one app-default change improved internal consistency without retraining the model:

- **Numeric AQI is now derived from the entered PM2.5** using the same EPA breakpoints, instead of staying at the training median of 165.
- **The local PM2.5 sensitivity step is now +25 µg/m³**, not +10, because tree predictions are piecewise constant and a +10 step often reported 0.0.
- **The app's initial PM2.5 and category defaults now use historical monthly medians** (234.2 µg/m³ and Very Unhealthy in November), so the demo opens on a realistic seasonal scenario rather than a fixed 60 µg/m³.

All are inference/app-layer changes. The fitted pipelines and all reported model metrics are unchanged.

### 10.4 Is anything committed?

No. `git status` shows `deliverable_3/`, `final_work_deliverable_2/`, and `instructions_for_teammates.pdf` as untracked. The last tracked commit is `56dea73`. Nothing from D3 has been pushed to GitHub yet. This is a submission task, not a modelling change.

---

## 11. File-by-file map

### 11.1 Root project — shared data and Deliverable 2

| File | What it does | Inputs | Outputs |
|---|---|---|---|
| `configs/base.yaml` | Single source of truth for seed, paths, targets, split dates, label merge, class order, metrics | — | Configuration consumed by `src/data.py` |
| `src/data.py` | Loads the feature table, builds X/y, applies the chronological split and 7-day purge, fits preprocessing on train only, exposes the scorable mask | `features_daily.csv`, config | Train/validation/test matrices and metadata |
| `src/classification.py` | Deliverable 2 classifier: tunes Logistic Regression, HGB, and MLP; selects on validation macro-F1; saves metrics and confusion matrices | Shared data contract | Classification metrics, predictions, hyperparameters |
| `src/regression.py` | Deliverable 2 regression: tunes Ridge and HGB; selects on validation RMSE; saves metrics | Shared data contract | Regression metrics, predictions, hyperparameters |
| `src/evaluate.py` | Shared metric functions: RMSE, MAE, R², accuracy, macro-F1, per-class reports | Predictions and truth | Metric dictionaries |
| `src/bias_analysis.py` | Subgroup error and bias analysis by category, season, year | Predictions | Bias tables and interpretation |
| `src/check_setup.py` | Smoke-tests the data contract and split | Feature table | Console checks |
| `data_scripts/01_download_data.py` | Downloads raw data | Source APIs / files | Raw CSVs |
| `data_scripts/02_clean_pm25_simple.py` | Cleans hourly PM2.5 and flags outliers | Raw hourly PM2.5 | `data/processed/pm25_hourly.csv` |
| `data_scripts/03_pm25_aqi_and_eda.py` | Converts PM2.5 to AQI, builds labels, runs EDA | Cleaned PM2.5 | AQI columns, EDA charts |
| `data_scripts/04_clean_station_simple.py` | Cleans station metadata | Raw station file | Clean station table |
| `data_scripts/08_build_calendar.py` | Builds calendar, Ramadan, Eid, and public-holiday features | Dates | Calendar columns |
| `data_scripts/09_build_features.py` | Builds the daily feature table | Cleaned data + calendar | `data/processed/features_daily.csv` |
| `data_scripts/5_...daily...py` | Cleans daily CAMS modelled data | Raw CAMS daily | Clean CAMS daily |
| `data_scripts/6_...hourly...py` | Cleans hourly CAMS modelled data | Raw CAMS hourly | Clean CAMS hourly |
| `data_scripts/7_...open_meteo...py` | Cleans daily ERA5 weather | Raw ERA5 | Clean weather daily |
| `outputs/model_results/*` | Stored metrics, predictions, hyperparameters, bias tables | Model runs | JSON/CSV evidence |
| `outputs/model_charts/*`, `outputs/eda_charts/*` | Figures | Model/EDA runs | PNG charts |
| `reports/phase1_final.tex`, `phase2_final.tex`, `report_main_final.tex/pdf` | Deliverable 1 report | Phase text | D1 PDF |
| `final_work_deliverable_2/report_d2.tex/pdf` | Deliverable 2 report | Phase text and tables | D2 PDF |

### 11.2 Deliverable 3 — the app layer

| File | What it does | Inputs | Outputs |
|---|---|---|---|
| `deliverable_3/app/app.py` | Streamlit UI: input form, metrics, charts, expanders, guidance | User inputs, `src.inference` | Browser UI |
| `deliverable_3/src/inference.py` | Loads four joblib pipelines and `metadata.json`; maps six inputs to 46 raw features (80 encoded columns), derives numeric AQI from PM2.5, and predicts PM2.5, AQI, category, interval, sensitivity, guidance, persistence | Joblib artifacts, metadata, user inputs | Prediction dictionary |
| `deliverable_3/src/train_pipeline.py` | Refits regression, classification, and q10/q90 pipelines with frozen settings; computes permutation importance, stagnation, seasonal means, monthly medians, and merges the hourly profile; writes artifacts | `features_daily.csv`, hourly CSV (optional) | Four `.joblib` files, `metadata.json` |
| `deliverable_3/src/ablation.py` | Runs history → +calendar → full feature-set comparison for regression and classification | Feature table | `models/ablation.json` |
| `deliverable_3/src/hourly.py` | Builds the hour-of-day and seasonal PM2.5 profile | `data/processed/pm25_hourly.csv` | Profile dictionary merged into metadata |
| `deliverable_3/tests/test_inference.py` | Fourteen unit tests: breakpoints, labels, feature columns, intervals, explanations, guidance, patterns, monthly medians, hourly profile | Artifacts | Pass/fail |
| `deliverable_3/models/*.joblib` | Fitted pipelines: regression, q10, q90, classification | Generated by `train_pipeline.py` | Loaded by `inference.py` |
| `deliverable_3/models/metadata.json` | Feature list, defaults, class labels, interval metadata, importance, patterns, monthly medians, hourly profile | Generated | Consumed by `inference.py` / app |
| `deliverable_3/models/ablation.json` | Ablation results | Generated by `ablation.py` | Evidence for the report |
| `deliverable_3/models/.gitignore` | Keeps generated binaries and JSON artifacts out of Git | — | Git rules |
| `deliverable_3/run_app.bat` | Double-click launcher for the app | — | Starts Streamlit |
| `deliverable_3/setup_models.bat` | Double-click launcher to refit models | — | Runs `train_pipeline` |
| `deliverable_3/run_ablation.bat` | Double-click launcher for the ablation | — | Runs `ablation` |
| `deliverable_3/requirements.txt` | Python dependencies | — | Installable environment |
| `deliverable_3/README.md` | Usage, artifacts, methods, submission checklist | — | Project documentation |

### 11.3 Which files the app actually uses at runtime

```
app/app.py
  └─ src/inference.py
       ├─ models/regression_pipeline.joblib
       ├─ models/regression_q10_pipeline.joblib
       ├─ models/regression_q90_pipeline.joblib
       ├─ models/classification_pipeline.joblib
       └─ models/metadata.json
```

The app does **not** import `src/classification.py` or `src/regression.py` at runtime. Those are Deliverable 2 training scripts. The D3 `train_pipeline.py` is the script that produces the artifacts the app loads.

---

## 12. How to run and reproduce

From the `deliverable_3` folder:

```bash
# 1. Install dependencies (once)
python -m pip install -r requirements.txt

# 2. Refit the frozen pipelines and write artifacts
python -m src.train_pipeline

# 3. Run the app
python -m streamlit run app/app.py

# 4. Run the tests
python -m unittest discover -s tests -t . -v

# 5. Run the ablation (optional)
python -m src.ablation
```

Double-click alternatives:

- `setup_models.bat` — refit models;
- `run_app.bat` — launch the app;
- `run_ablation.bat` — run the ablation.

**Important:** the three `.bat` files currently contain a machine-specific path to a temporary virtual environment. They work on this machine but will fail on another machine. Before submission they should be changed to use a project-local `.venv` or fall back to `py`/`python` on `PATH`.

The data files are not committed. Restore `data/processed/features_daily.csv` and `data/processed/pm25_hourly.csv` before refitting. The model binaries and generated JSON are also not committed; they are regenerated by `train_pipeline.py`.

---

## 13. Limitations and honest framing

1. **One station, not a city.** The target is the Lahore US Diplomatic Post sensor (OpenAQ 25135). Results may not generalize to other neighborhoods.
2. **The regression ties persistence on test.** RMSE 64.92 vs 64.82 is a 0.16% difference — effectively a tie. The classifier is worse than persistence on test.
3. **The classifier is biased toward `Unhealthy`.** Its 99.9% confidence on a clear `Unhealthy` case shows overconfidence.
4. **Only six user inputs.** Thirteen features — including dew point, pressure, wind direction, rain, solar radiation, cloud cover, holiday flags, and data-quality flags — use training medians/modes. The forecast is illustrative, not a live data feed.
5. **Lags and rolling windows are a steady-state assumption.** The app assumes today's PM2.5 has been stable for the past week.
6. **Some inputs still use defaults.** The app's initial PM2.5 and category values are historical monthly medians, not live readings. Numeric AQI is derived from PM2.5, but the app does not derive Ramadan/Eid/public-holiday flags from the selected date; those flags remain at their default 0. Dew point, pressure, wind direction, rain, solar radiation, cloud cover, and data-quality flags also use training medians.
7. **Local sensitivity is a finite difference.** HGB is piecewise constant, so the app uses a +25 µg/m³ stress test for PM2.5 to avoid most zero-effect artifacts. The reported change is still local and associational, not a causal slope.
8. **Interval coverage on test is 77%, below the nominal 80%.**
9. **No direct traffic data.** Weekday, weekend, and holiday features are proxies.
10. **No causal claims.** Permutation importance and sensitivity are associational.
11. **Guidance is non-medical.** It is not a substitute for official health advice.
12. **Rare hazardous classes have small support.** Per-class metrics for the two hazardous bands should be read with caution.

These are not hidden. They are part of the project's honest evaluation.

---

## 14. Viva-ready answers

**Why HGB?**
> “It is a strong tabular model: nonlinear, handles interactions and missing values, needs little preprocessing, and trains in seconds. It was selected on validation in Deliverable 2, not on test.”

**Why doesn't it beat persistence?**
> “At a one-day horizon, PM2.5 is highly autocorrelated, so persistence is a very strong baseline. Our regression essentially ties it — 64.92 vs 64.82 µg/m³, a 0.16% difference — and we report that honestly. The contribution is the explainable end-to-end system and calibrated uncertainty, not a new accuracy record.”

**Why show two AQI categories?**
> “The classifier is a direct six-class model, but it was weaker than persistence on test. The regression-implied category converts predicted PM2.5 through the same EPA breakpoints used to build the labels, so it is monotonic and more defensible. When they disagree, the app says so.”

**Why an 80% interval, and why 77% test coverage?**
> “We train quantile HGB models at the 10th and 90th percentiles, then widen by a factor chosen on validation to reach 80% empirical coverage. The factor came out to 1.00; validation coverage is 80.3%, test coverage is 77.0%. We report test coverage but never tune on it.”

**Why sensitivity instead of SHAP?**
> “For a single-row app, one-at-a-time sensitivity is simple, fast, and model-agnostic, and permutation importance covers the global view. SHAP would give a more theoretically grounded local decomposition; it is future work.”

**Why only six inputs?**
> “Usability. The remaining features use training medians and modes, and lags are held at today's value. That is why the app calls itself illustrative. A production version would ingest live weather and history.”

**What changed after Deliverable 2?**
> “The model and its reported scores are unchanged; the D3 refit reproduces D2 exactly. What is new is the application layer: inference, the Streamlit app, prediction intervals, explanations, the ablation, and the hourly profile. The D2 code and report were not edited.”

**What is the date field?**
> “It is today's feature date. Calendar features are derived from it, and the forecast is always for the following day. It is not an arbitrary-date or multi-day forecast.”

---

## 15. Glossary

| Term | Meaning in this project |
|---|---|
| **PM2.5** | Fine particulate matter with diameter ≤ 2.5 µm, in µg/m³. The regression target. |
| **AQI** | Air Quality Index, a 0–500 scale computed from PM2.5 using EPA breakpoints. |
| **HGB** | Histogram-based Gradient Boosting; an ensemble of small trees trained sequentially to correct errors. |
| **Regression** | Predicting a continuous number (tomorrow's PM2.5). |
| **Classification** | Predicting a discrete class (tomorrow's AQI category). |
| **Persistence** | The baseline “tomorrow = today.” |
| **RMSE** | Root mean squared error; same units as the target, penalizes large errors. |
| **MAE** | Mean absolute error; average absolute miss, same units. |
| **R²** | Fraction of variance explained; 1 is perfect, 0 is no better than the mean. |
| **Macro-F1** | Average F1 across classes, treating each class equally; suitable for imbalance. |
| **Weighted-F1** | F1 weighted by class support; dominated by common classes. |
| **Precision / recall** | Of the predicted positives, how many are correct / of the true positives, how many were found. |
| **Confusion matrix** | Table of predicted versus true classes. |
| **Prediction interval** | A range for a future observation, here the central 80%. |
| **Quantile regression** | Training a model to predict a conditional percentile rather than the mean. |
| **Coverage** | Fraction of true values that fall inside the interval. |
| **Calibration** | Adjusting interval width so empirical coverage matches the nominal level. |
| **Permutation importance** | Global importance: increase in error when a feature is shuffled. |
| **Sensitivity** | Local importance: change in forecast when one input is perturbed. |
| **Ablation** | Removing feature groups to test whether they improve performance. |
| **Data leakage** | Information from validation/test contaminating training; prevented by the 7-day purge and train-only scaling. |
| **Class imbalance** | Some categories are much rarer than others; handled with `class_weight="balanced"` and macro-F1. |
| **Piecewise constant** | A tree model's prediction stays flat within a leaf, so small input changes may have no effect. |

---

*End of study report.*
