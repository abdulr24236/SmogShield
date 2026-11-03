# Deliverable 2 — Modelling Contract (v1)

This document freezes the shared modelling decisions before any model is
trained. Every member must use the same split, feature rules and metric
functions.

## 1. Tasks

**Primary task: next-day AQI-category classification.**

- Target: `target_next_aqi_category`
- Classes: Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy,
  Hazardous (301-400), Hazardous (401-500)
- This is the task used for the Precision, Recall, F1 and AUC-ROC analysis
  required by Phase 4.

**Secondary task: next-day PM2.5 regression.**

- Target: `target_next_pm25`
- Metrics: RMSE, MAE and R²
- This provides a continuous forecast and a comparison with a persistence
  baseline.

The target is tomorrow's value. Today's PM2.5, AQI, weather and calendar
information are known at forecast time and may be used as features. Future
values may not.

## 2. Data

Input file:

```text
data/processed/features_daily.csv
```

It is produced by `data_scripts/09_build_features.py` and is not committed to
Git. Restore it from the shared Drive or regenerate it from the Phase 2 raw
files.

The current generated table has:

- 2,113 daily rows and 63 columns.
- 1,505 rows with a next-day target.
- 1,294 rows with a next-day target, lag 3 and all weather values.

Re-check these numbers with `python -m src.check_setup` after regenerating the
file. The cleaned daily AQI series has 1,506 usable days. Its category counts
are:

| Category | Days |
|---|---:|
| Good | 9 |
| Moderate | 126 |
| Unhealthy for Sensitive | 269 |
| Unhealthy | 647 |
| Very Unhealthy | 256 |
| Hazardous (301-400) | 139 |
| Hazardous (401-500) | 60 |

The “Good” class is very rare. In the current date split it appears only in the
training period (9 rows) and is absent from validation and test. Its per-class
recall and one-vs-rest AUC therefore cannot be estimated on those splits. Do not
hide this: report per-class recall and macro F1, state that Good has no test
support, and discuss the instability of macro metrics for rare classes. A
coarser grouping may be used only as a sensitivity analysis, not as the primary
target.

Current class support by split:

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| Good | 9 | 0 | 0 |
| Moderate | 100 | 16 | 8 |
| Unhealthy for Sensitive | 175 | 53 | 34 |
| Unhealthy | 488 | 80 | 77 |
| Very Unhealthy | 148 | 47 | 61 |
| Hazardous (301-400) | 72 | 43 | 24 |
| Hazardous (401-500) | 30 | 11 | 19 |
| **Total** | **1,022** | **250** | **223** |

`ModelData.summary()["class_counts"]` returns this table, and
`class_support_warnings` names every class with zero support in a split. The
generated table must be included in the report.

## 3. Split

Chronological split with a purge gap:

| Split | Dates | Current usable rows |
|---|---|---:|
| Train | 2019-05-23 through 2023-06-30 | 1,022 |
| Validation | 2023-07-14 through 2024-06-03 | 250 |
| Test | 2024-07-08 through 2025-02-16 | 223 |

The table shows the current data. The validation/test *boundaries* in
`configs/base.yaml` are 2024-06-30 and the last available date; the actual last
usable row can be earlier because rows without a next-day target are dropped.

`gap_days = 7` removes 2023-07-01..2023-07-07 and 2024-07-01..2024-07-07. The
gap prevents next-day targets at the end of one period from overlapping the
start of the next.

This is roughly a 68/17/15 time split by rows. Do not use random K-Fold: it
would train on future days and evaluate on past days.

Rules:

- Fit imputers, encoders and scalers on the training split only.
- Use the validation split for tuning and model selection.
- Touch the test split once, after the model and hyperparameters are frozen.
- Report all three row counts and date ranges.
- For tuning, use `data.X_train_raw` with
  `data.make_model_pipeline(estimator)` and `TimeSeriesSplit`. Do not
  cross-validate on `data.X_train`: its preprocessor has already seen the whole
  training split, including each fold's held-out rows.

## 4. Features and leakage

The shared loader in `src/data.py`:

- drops `date` and every `target_next_*` column;
- drops the precomputed `*_scaled` weather columns, because Phase 2 scaled them
  on all rows;
- drops the redundant `season` string because `season_*` dummies already exist;
- keeps current-day PM2.5, AQI, weather and calendar features;
- one-hot encodes `aqi_category` and median-imputes numeric features;
- applies `RobustScaler` to numeric features after fitting on train only.

The loader has a hard guard that raises if any `target_*` or `*_scaled` column
survives the config lists. Do not remove this guard to work around a config
typo; fix the config instead. The class labels are also required in
`configs/base.yaml`; they are never inferred from the data, because that would
make the order alphabetical and include validation/test labels.

No model may add a feature derived from `shift(-1)` or from the test period.
Any new feature must be added to the shared loader or explicitly documented and
tested for leakage.

## 5. Metrics

Classification:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- One-vs-rest macro AUC-ROC
- Confusion matrix and per-class precision/recall/F1

Macro metrics are primary because the classes are imbalanced. AUC-ROC is
reported when probabilities are available; it is undefined if a class is absent
from the evaluation set, and that limitation must be stated rather than filled
with a fake value.

Regression:

- RMSE
- MAE
- R²

## 6. Baselines

Classification:

1. Majority class.
2. Logistic regression with class weights.
3. Histogram gradient boosting.
4. Optional: a small MLP, if it improves validation performance.

Regression:

1. Persistence: tomorrow = today.
2. Ridge regression.
3. Histogram gradient boosting.

Every model must be compared with its baseline on the same validation split.

## 7. Error analysis and bias check

Subgroups to examine:

- AQI category
- Season (winter, spring, summer, autumn)
- Year
- Weekday versus weekend
- Ramadan and Eid periods
- Public holidays
- PM2.5 range / pollution severity

The dataset contains one monitoring station and no demographic attributes.
State this explicitly: the bias check is environmental and temporal, not
demographic. Report where the model fails and whether errors are systematic for
high-pollution days, winter smog episodes, or particular years.

## 8. Reproducibility rules

- Python dependencies are in `requirements.txt`.
- Random seed: `42` from `configs/base.yaml`.
- Data and large model artifacts stay out of Git; use the shared Drive or a
  release link.
- Each script must be runnable from the repository root.
- Every result in the report must be reproducible from a committed script and
  the shared data.
- Use Git CLI for add/commit/push. Do not use the GitHub “Upload Files” button.

## 9. Code ownership

| Member | Modules |
|---|---|
| Abdul Rehman | `src/data.py`, `src/evaluate.py`, report integration |
| Muhammad Haider Mughal | `src/classification.py`, classification tuning |
| Muhammad Ayub Butt | `src/regression.py`, `src/bias_analysis.py` |

Each member commits their own files. Shared files are changed only after
coordination.
