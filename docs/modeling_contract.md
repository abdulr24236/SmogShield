# Deliverable 2 — Modelling Contract (v1)

This document freezes the shared modelling decisions before any model is
trained. Every member must use the same split, feature rules and metric
functions.

## 1. Tasks

**Primary task: next-day AQI-category classification.**

- Target: `target_next_aqi_category`
- Classes: Good/Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy,
  Hazardous (301-400), Hazardous (401-500)
- The raw table keeps Good and Moderate separate. For modelling, Good is merged
  into Moderate because it has only 9 training rows and zero validation/test
  rows. This yields six classes, all present in every split.
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
file. The cleaned daily AQI series has 1,506 usable days. Its raw category
counts are:

| Category | Days |
|---|---:|
| Good | 9 |
| Moderate | 126 |
| Unhealthy for Sensitive | 269 |
| Unhealthy | 647 |
| Very Unhealthy | 256 |
| Hazardous (301-400) | 139 |
| Hazardous (401-500) | 60 |

The raw Good class is too rare to model separately: it has 9 training rows and
zero validation/test rows, so its per-class recall and one-vs-rest AUC cannot be
estimated on those splits. The classification target therefore merges Good into
Moderate through `data.label_merge` in `configs/base.yaml`. This is a modelling
decision, not a data deletion; the raw table still contains all seven
categories. Report the merge and the raw counts.

The resulting six-class support by split is:

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| Good/Moderate | 109 | 16 | 8 |
| Unhealthy for Sensitive | 175 | 53 | 34 |
| Unhealthy | 488 | 80 | 77 |
| Very Unhealthy | 148 | 47 | 61 |
| Hazardous (301-400) | 72 | 43 | 24 |
| Hazardous (401-500) | 30 | 11 | 19 |
| **Total** | **1,022** | **250** | **223** |

Every class now has support in every split, so multiclass AUC-ROC is defined.
The 8 test rows in Good/Moderate and 11 validation rows in Hazardous
(401-500) are still small; report their per-class precision/recall as noisy
rather than treating them as precise estimates. The two Hazardous bands are
kept separate; merging them is allowed only as a documented sensitivity
analysis.

`ModelData.summary()["class_counts"]` returns this six-class table and
`ModelData.label_merge` records the merge. `class_support_warnings` still names
any future class with zero support. The support table must appear in the report.

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
The actual validation end is 2024-06-03, not 2024-06-30; the remaining June days
do not have a usable next-day target. This is expected, not a config typo.

The feature table has one row per calendar day (2,113 rows over a 2,113-day
span). The lower usable counts in each split come from missing PM2.5/target
values, not from missing calendar rows. `python -m src.check_setup` verifies
that `target_next_pm25` and `pm25_lag1` match the adjacent calendar dates.

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
  `data.make_model_pipeline(estimator)` and
  `TimeSeriesSplit(n_splits=..., gap=7)`. Do not cross-validate on
  `data.X_train`: its preprocessor has already seen the whole training split,
  including each fold's held-out rows.

The test split is touched once per final model. Choose hyperparameters with
training-only time-series CV and/or the validation split, freeze the model, and
then evaluate it once on test. Train the final model on the training split only,
not on train plus validation. Refitting on train plus validation after looking
at validation would make validation an optimistic, post-hoc estimate and would
change the model after the test decision has been made.

## 4. Features and leakage

The shared loader in `src/data.py`:

- drops `date` and every `target_next_*` column;
- drops the precomputed `*_scaled` weather columns, because Phase 2 scaled them
  on all rows;
- drops the redundant `season` string because `season_*` dummies already exist;
- drops `year` from model inputs because training ends in 2023 while
  validation/test continue to 2025; trees cannot extrapolate unseen years and
  linear/neural models may extrapolate an unstable trend. `month`,
  `day_of_year` and the cyclical encodings carry the seasonality;
- keeps current-day PM2.5, AQI, weather and calendar features;
- one-hot encodes `aqi_category` and median-imputes numeric features;
- adds missing-value indicators to both imputers. Lag missingness is 29.3% in
  train, 8.0% in validation and 1.3% in test, so the indicator is informative
  rather than noise. The imputers are fitted on train only;
- applies `RobustScaler` to numeric features after fitting on train only.

The loader has a hard guard that raises if any `target_*` or `*_scaled` column
survives the config lists. Do not remove this guard to work around a config
typo; fix the config instead. The class labels are also required in
`configs/base.yaml`; they are never inferred from the data, because that would
make the order alphabetical and include validation/test labels. The optional
`label_merge` mapping is applied only to the classification target, before
encoding.

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

Macro metrics are primary because the classes are imbalanced. The six-class
target has support for every class in every split, so multiclass AUC-ROC is
defined. If a future split lacks a class, leave that metric undefined rather
than filling it with a fake value; `classification_metrics` records the reason
in `roc_auc_note` and emits a warning. Report the support table and note that
the per-class estimates for Good/Moderate and Hazardous (401-500) are noisy.

`classification_metrics` requires `labels` whenever `y_score` is passed,
validates that the probability matrix has one column per label, and returns
`class_names` for the confusion matrix. Rows must sum to 1 within `1e-4`;
the helper renormalizes only inside that tolerance, which removes float32
softmax noise, and raises for anything further off. For scikit-learn models,
call `align_proba(model.predict_proba(X), model.classes_,
len(data.labels))` before passing the result. This keeps probability columns in
the fixed class-code order even if a model saw fewer classes.

Regression:

- RMSE
- MAE
- R²
- Optional `rmse_skill_vs_persistence = 1 - rmse_model / rmse_persistence`

`regression_metrics` raises when targets are non-finite. Non-finite predictions
also raise unless `allow_missing=True`; only the persistence baseline should use
that option, and it must report `n_dropped`. For very small slices, report the
support `n`; if a confidence interval is needed, use bootstrap resampling in the
report rather than presenting a point estimate as precise.

`ModelData.scorable_rows` defines the single evaluation row set for each split:
rows where today's `pm25` is finite. The masks are aligned with the raw frames
and target Series, and `ModelData.summary()` reports how many rows are kept and
dropped. Apply `data.scorable_rows[split]` to every method, including
persistence, so no model is scored on rows where persistence had to impute the
current-day value. Do not use the mask to change training rows.

```python
mask = data.scorable_rows["validation"]
y_true = data.y_validation[mask].to_numpy()
y_pred = predictions[mask.to_numpy()]
# Also filter y_score and any per-row analysis with the same mask.
```

## 6. Baselines

Classification:

1. Majority class.
2. Persistence: tomorrow's category = today's `aqi_category`.
3. Logistic regression with class weights.
4. Histogram gradient boosting.
5. Optional: a small MLP, if it improves validation performance.

Regression:

1. Persistence: tomorrow's PM2.5 = today's `pm25`.
2. Ridge regression.
3. Histogram gradient boosting.

The shared helpers are in `src/evaluate.py`:

- `majority_class_predictions(y_train, n_samples)`
- `persistence_class_predictions(current_categories, labels, label_merge,
  fallback)`
- `persistence_regression_predictions(current_pm25, fallback)`

For the classification persistence baseline, pass the majority-class code as
`fallback` when today's category is missing, and state which fallback was used.
For regression persistence, do not fill missing current-day PM2.5 values; score
it on `data.scorable_rows[split]` like every other method. If a caller evaluates
the unfiltered persistence output instead, `allow_missing=True` is permitted
only for that baseline and `n_dropped` must be reported. Persistence is a strong
baseline because AQI and PM2.5 are strongly autocorrelated; every model must beat
it on the same validation and test rows, not just beat the majority class.

## 7. Error analysis and bias check

Subgroups to examine:

- AQI category
- Season (winter, spring, summer, autumn)
- Year
- Weekday versus weekend
- Ramadan and Eid periods
- Public holidays
- PM2.5 range / pollution severity

The class mix shifts across periods. `Unhealthy` is about 48% of training rows
but 32% of validation and 35% of test. The two Hazardous bands together are
about 10% of training but 22% of validation and 19% of test. This may reflect
seasonal composition, missing training days/targets, or real worsening; do not
assume one cause. Expect models to under-predict severity on the later data and
report that possibility rather than claiming a causal trend.

The dataset contains one monitoring station and no demographic attributes.
State this explicitly: the bias check is environmental and temporal, not
demographic. Report where the model fails and whether errors are systematic for
high-pollution days, winter smog episodes, or particular years.

## 8. Reproducibility rules

- Python dependencies are in `requirements.txt`.
- Random seed: `42` from `configs/base.yaml`.
- Data and large model artifacts stay out of Git; use the shared Drive or a
  release link. `.gitignore` covers `__pycache__/`, `data/processed/`, model
  binaries and generated model result/chart files; the folder README
  placeholders remain tracked.
- Each script must be runnable from the repository root.
- Every result in the report must be reproducible from a committed script and
  the shared data.
- `python -m src.check_setup` must report 0 bad next-day targets and 0 bad
  lag-1 values. The feature table is complete daily; target/lag missingness
  comes from missing PM2.5 values, not row-shift gaps.
- Run `python -m unittest discover -s tests -t .` after changing
  `src/evaluate.py`. The tests use synthetic arrays and do not need the data
  file.
- `09_build_features.py` has no future or whole-dataset fill for PM2.5. Its only
  fill is `fillna(0)` for calendar flags after merging the calendar table. The
  `*_scaled` columns use all-row statistics and are explicitly excluded from
  modelling. The separate `04_clean_station_simple.py` cross-check uses
  backward-only `ffill(limit=3)` for weather averages; it does not feed the
  model table.
- Use Git CLI for add/commit/push. Do not use the GitHub “Upload Files” button.

## 9. Code ownership

| Member | Modules |
|---|---|
| Abdul Rehman | `src/data.py`, `src/evaluate.py`, report integration |
| Muhammad Haider Mughal | `src/classification.py`, classification tuning |
| Muhammad Ayub Butt | `src/regression.py`, `src/bias_analysis.py` |

Each member commits their own files. Shared files are changed only after
coordination.
