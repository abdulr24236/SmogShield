# Final Deliverable 2 package

SmogShield Lahore — Deliverable 2: Phase 3 (Model Development) and Phase 4 (Evaluation and Bias Analysis).

This folder is a self-contained review/submission package built from the frozen
final outputs on `main` at commit `a55a7f5`.

## Start here

- **`report_d2.pdf`** — compiled LaTeX final report (17 pages).
- **`report_d2.tex`** — LaTeX source used to build the PDF; also suitable for Overleaf.
- **`report_d2.html`** — printable HTML companion/fallback.
- **`CHECKLIST.md`** — contract requirements mapped to report sections/files.
- **`evidence/verification.md`** — final run evidence and numerical checks.

## Key final results

| Task | Selected model | Validation | Test | Persistence validation | Persistence test | Test skill vs persistence | Verdict |
|---|---|---|---:|---:|---:|---|---|
| Classification | HistGradientBoosting | macro F1 0.4941 | macro F1 0.4780 | macro F1 0.4766 | macro F1 0.5157 | — | did not beat persistence on test macro F1 |
| Regression | HGB | RMSE 45.6243 | RMSE 64.9198 | RMSE 51.5457 | RMSE 64.8175 | −0.16% | essentially tied with persistence on test RMSE |

Models were selected on validation only; test results were not used to change
the selection. Regression Ridge test RMSE was **62.0956** (skill **+4.20%**),
which did beat persistence, but switching to Ridge after seeing test would
violate the validation-only selection rule. The report states this distinction
explicitly.

Islamic-event subgroup coverage: the validation join contains 25 scorable
Ramadan rows and no scorable Eid rows. The feature table flags Eid on 10–12
April and 17–19 June 2024, but the AQI label is missing on those dates; all six
dates also lack a next-day target. The test join contains only the `Normal`
group. This is a label-coverage limitation, not a calendar-flag gap, and the
report makes no Eid-effect claim.

## Official submission checklist

| Requirement | Status | Evidence |
|---|---|---|
| GitHub repository with updated code in `/src` or `/models` | Met | `main` at `a55a7f5`; `src/classification.py`, `src/regression.py`, `src/bias_analysis.py`, shared `src/data.py` and `src/evaluate.py` |
| `requirements.txt` present | Met | Repository root `requirements.txt` |
| No model binary over 100 MB pushed to Git | Met | No model binaries are tracked; `.gitignore` excludes them |
| External weights link in `README.md` if weights exist | Met for this submission | No separate weights are required; the external Drive host is documented in the root `README.md` and `models/README.md` |
| Updated PDF covering Phase 3 and Phase 4 | Met | `report_d2.pdf` and `report_d2.tex` |
| Architecture diagrams | Met | Report §1.2 and `figures/architecture.png` |
| Loss/accuracy curves | Met | Report §5 and §6: HGB validation curve, logistic-regression curve, regression learning curve, confusion matrix, residual/error charts |
| Contribution Table mapping Git commit authors to modules | Met | Report §9.1 and `tables/git_authors.tex` |

## Report contents

The report includes the architecture diagram, six-class support table, raw
seven-class counts, split and scorable-row table, classification metrics,
per-class metrics, confusion matrix, regression metrics, hyperparameters,
bias subgroup table, Git-author contribution table mapping commit authors to
modules, limitations, and reproducibility checks. Sections 4–6 are Phase 3;
Sections 7–10 are Phase 4.

## Model weights and external artifacts

No trained model binaries are stored in this package or in Git. The selected
classification and regression pipelines can be refit from the committed code
using fixed seeds; the bias analysis reads saved predictions and trains nothing.
The external host for project data, frozen predictions, and any exported weights
is the shared Google Drive folder:

<https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>

## Folder contents

```text
final_work_deliverable_2/
├── report_d2.pdf              final compiled LaTeX report
├── report_d2.tex              LaTeX source used for the PDF
├── report_d2.html             printable HTML companion/fallback
├── README.md                  this file
├── CHECKLIST.md               contract-to-report checklist
├── MANIFEST.csv               file sizes and SHA-256 hashes
├── tables/                    LaTeX and CSV versions of every report table
├── figures/                   final classification, regression, and bias charts
├── metrics/                   frozen metrics JSON, predictions, and class mix
├── evidence/                  run logs, verification notes, and checks
├── scripts/                   table, HTML, and build scripts
└── data/                      data-dependency note (data itself is not copied)
```

## Reproduce the tables and report

From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
python scripts/verify_package.py
```

`build_report.ps1` regenerates the tables and architecture diagram, builds the
HTML companion, and compiles `report_d2.tex` with Tectonic or `pdflatex`. The
tables are included from `tables/` and figures from `figures/`. If no LaTeX
engine is installed, use `-HtmlFallback`; this writes
`report_d2_html.pdf` and does not overwrite the LaTeX-built `report_d2.pdf`.

## Data dependency

The modelling table `data/processed/features_daily.csv` is intentionally not
copied into this package. It is generated data and is not committed to Git.
Restore it from the shared Drive or regenerate it from the Phase 2 raw files.
See `data/README.md` for the shared Drive link and expected checks.

## Important rules preserved

- One chronological split with a seven-day purge gap.
- Preprocessing fitted on the training split only.
- Test touched once per final model; no test-time retuning.
- All methods scored on the same scorable rows.
- The final code calls the metric helper with `allow_missing=False` for every
  method. `allow_missing=True` remains reserved for an unfiltered persistence
  baseline only; the persistence `n_dropped` values are still reported.
- Generated prediction CSVs, metrics JSON, and charts are review artifacts and
  are not intended to be committed to Git.
- Exactly the three project contributors are listed; no AI co-author trailer was
  added to Git history.
