# Verification evidence

## Final code state

- Repository: `C:\Users\Abdul Rehman\Projects\Smog_Shield_Phase_II\SmogShield`
- Branch: `main`
- Final merge commit: `a55a7f5` (`Merge pull request #9 from abdulr24236/ayub/regression-bias`)
- Previous merge: `20fa79e` (`Merge pull request #10 from abdulr24236/haider/classification`)

The final pipeline was run after updating local `main` to `a55a7f5`.

## Commands run

```powershell
python -m src.check_setup
python -m unittest discover -s tests -t .
python -m src.classification
python -m src.regression
python -m src.bias_analysis
```

Run logs are in this folder:

- `final-main-classification.log`
- `final-main-regression.log`
- `final-main-bias.log`

## `check_setup`

```text
CALENDAR ALIGNMENT
  rows: 2113  span: 2113  missing dates: 0
  bad next-day PM2.5: 0
  bad lag-1 PM2.5: 0
  bad next-day category: 0

CLASSIFICATION
  rows:     train=1022  validation=250  test=223
  scorable: train=917  validation=244  test=222
  dropped:  train=105  validation=6  test=1
  train:    2019-05-23 -> 2023-06-30
  val:      2023-07-14 -> 2024-06-03
  test:     2024-07-08 -> 2025-02-16
  classes:  6

REGRESSION
  rows:     train=1022  validation=250  test=223
  scorable: train=917  validation=244  test=222
  dropped:  train=105  validation=6  test=1
```

`Shared setup OK.`

## Unit tests

```text
Ran 15 tests in 0.165s
OK
```

## Classification outputs

- Predictions: `metrics/classification_predictions.csv`
- Shape: **473 rows × 12 columns**
- Scorable rows: validation **244**, test **222**, total **466**
- Selected model: **HistGradientBoosting**
- Validation macro F1: **0.4941**
- Test macro F1: **0.4780**
- Persistence test macro F1: **0.5157**
- Test accuracy: **0.5856**
- Test one-vs-rest macro AUC: **0.8886**
- SHA-256:

```text
47D8E0A89FC3F37EA6B74E807DCF566FBDD6D79F734D5A05CFD56124888EB2EB
```

This matches the verified Drive copy byte-for-byte.

## Regression outputs

- Predictions: `metrics/regression_predictions.csv`
- Shape: **473 rows × 6 columns**
- Scorable rows: validation **244**, test **222**, total **466**
- Selected model: **hist_gb**
- Validation RMSE: **45.6243**
- Test RMSE: **64.9198**
- Test persistence RMSE: **64.8175**
- Test Ridge RMSE: **62.0956**
- Validation dropped: **6**; test dropped: **1**

The final code uses the shared scorable mask and calls the metric helper with
`allow_missing=False` for every method. `allow_missing=True` remains reserved
for an unfiltered persistence baseline only; the persistence `n_dropped`
values are still recorded in the metrics JSON and the report table.

## Bias analysis

- Regression rows loaded: **466**
- Classification rows loaded: **466**
- Rows surviving the join: **466**
- Subgroup rows: **47**
- Dimensions: **7**
- Charts: **14**
- Missing-value counts: all zero for every dimension in validation and test
- `model_reg`: `hist_gb`
- `model_class`: `HistGradientBoosting`
- Test `Islamic_Event`: only `Normal`, so no test Ramadan/Eid comparison

## Final report build

- `report_d2.pdf` was compiled directly from `report_d2.tex` with Tectonic
  0.17.0.
- Page count: **17**.
- The PDF is explicitly labelled **Phase 3 (Model Development)** and
  **Phase 4 (Evaluation and Bias Analysis)**.
- It contains the architecture diagram, classification/regression curves,
  bias charts, and the Git-author contribution table mapping commit authors to
  project modules.

## Git hygiene

The frozen code history is at `a55a7f5` on `main`. At package-finalisation time,
`git status --short` showed only the intentional local documentation edits and
the untracked review package:

```text
 M README.md
 M models/README.md
?? final_work_deliverable_2/
?? instructions_for_teammates.pdf
```

The two README edits document the model-weight/external-artifact policy and the
D2 module ownership. No generated prediction CSV, metrics JSON, or chart was
tracked. The review package copies these artifacts for evidence, not for
committing.

The report and package list exactly the three human contributors:

- Muhammad Haider Mughal (3006)
- Muhammad Ayub Butt (3076)
- Abdul Rehman (3059)

No AI/OpenCode co-author trailer was added to any commit.
