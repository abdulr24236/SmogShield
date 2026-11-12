# Deliverable 2 checklist

This checklist maps the official submission checklist and the requirements in
`docs/modeling_contract.md` to the final report and package files.

## Official submission checklist

| Official requirement | Status | Evidence |
|---|---|---|
| GitHub repository: updated code in `/src` or `/models` | Complete | `main` at `a55a7f5`; `src/classification.py`, `src/regression.py`, `src/bias_analysis.py`, shared `src/data.py` and `src/evaluate.py` |
| `requirements.txt` present | Complete | Repository root `requirements.txt` |
| Model weights: no binary over 100 MB in Git | Complete | No model binaries tracked; `.gitignore` excludes `.joblib`, `.pkl`, `.pt`, `.pth`, `.onnx` |
| Model weights: external link in `README.md` if weights exist | Complete for this submission | No separate weights required; Drive host documented in root `README.md`, `models/README.md`, and this package's `README.md` |
| LaTeX report PDF covering Phase 3 and Phase 4 | Complete | `report_d2.pdf`; `report_d2.tex`; Sections 4–6 = Phase 3, Sections 7–10 = Phase 4 |
| Architecture diagrams | Complete | §1.2; `figures/architecture.png` |
| Loss/accuracy curves | Complete | §5: HGB validation curve and LR C-curve; §6: regression learning curve and diagnostic charts |
| Contribution Table mapping Git commit authors to modules | Complete | §9.1; `tables/git_authors.tex`; commit hashes included |

## Modelling-contract checklist

| Requirement | Status | Where |
|---|---|---|
| Phase 3 and Phase 4 coverage explicitly labelled | Complete | Report title; §1.1; Phase 3 headings §4–6; Phase 4 headings §7–10 |
| Architecture section and diagram | Complete | §1.2; `figures/architecture.png` |
| Git-author contribution table mapping authors to modules | Complete | §9.1; `tables/git_authors.tex` |
| Primary six-class next-day AQI classification | Complete | `report_d2.pdf` §1, §5; `metrics/classification_metrics.json` |
| Secondary next-day PM2.5 regression | Complete | `report_d2.pdf` §1, §6; `metrics/regression_metrics.json` |
| Chronological split with purge gap | Complete | §2; `tables/split_summary.tex` |
| Report all three row counts and date ranges | Complete | §2; `tables/split_summary.tex` |
| Train-only imputers/scalers | Complete | §3 |
| Test touched once per final model | Complete | §4, §6 |
| Report raw seven-class counts | Complete | §2.1; `tables/raw_category_counts.tex` |
| Report Good-to-Moderate merge | Complete | §2.1 |
| Six-class support table in report | Complete | §2.1; `tables/support_by_split.tex` |
| State rare-class estimates are noisy | Complete | §2.1, §5, §8 |
| Classification accuracy | Complete | §5; `tables/classification_metrics.tex` |
| Balanced accuracy | Complete | §5; `tables/classification_metrics.tex` |
| Macro precision and recall | Complete | §5; `tables/classification_metrics.tex` |
| Macro F1 and weighted F1 | Complete | §5; `tables/classification_metrics.tex` |
| One-vs-rest macro AUC-ROC | Complete | §5; `tables/classification_metrics.tex` |
| Confusion matrix | Complete | §5; `tables/classification_confusion_test.tex`, Figure 2 |
| Per-class precision/recall/F1 | Complete | §5; `tables/classification_per_class_test.tex` |
| Majority, persistence, logistic regression, HGB | Complete | §4.1, §5 |
| State classification persistence fallback | Complete | §4.1: Unhealthy, code 2 |
| Optional MLP decision | Complete | §4.1: excluded at validation |
| Regression RMSE, MAE, R² | Complete | §6; `tables/regression_metrics.tex` |
| Regression persistence comparison | Complete | §6; `tables/regression_metrics.tex` |
| Same scorable-row mask for all methods | Complete | §2, §4; `metrics/` |
| Only persistence may use `allow_missing=True` if unfiltered | Complete — final run used the scorable mask and `allow_missing=False` for all methods | §2, §4; `evidence/verification.md` |
| Report dropped rows | Complete | §2; `tables/regression_metrics.tex` |
| Bias dimensions: AQI, season, year, weekday/weekend, Ramadan/Eid, holidays, severity | Complete | §7; `tables/bias_test.tex`; `figures/bias_*.png` |
| Report class-mix shift | Complete | §7 |
| State one station and no demographics | Complete | §7, §8 |
| Report where model fails / systematic errors | Complete | §7, §8 |
| High-severity under-prediction finding | Complete | §7 |
| Reproducibility: seed, commands, data source | Complete | §9; `evidence/verification.md` |
| `check_setup` zero bad targets/lags | Complete | §9; `evidence/verification.md` |
| Unit tests pass | Complete | §9; `evidence/verification.md` |
| Generated CSVs/JSON/charts not committed | Complete | `README.md`; `evidence/verification.md` |
| Exactly three human contributors | Complete | Report title page; `README.md` |
| No AI co-author trailer | Complete | `evidence/verification.md` |

## Final numerical checks

- Classification predictions: **473 rows / 466 scorable**.
- Regression predictions: **473 rows / 466 scorable**.
- Classification test macro F1: **0.4780**; persistence: **0.5157**.
- Regression HGB test RMSE: **64.9198**; persistence: **64.8175**.
- Regression Ridge test RMSE: **62.0956**.
- Bias join: **466 rows**; **47** subgroup rows; **7** dimensions; **14** charts.
- Classification prediction SHA-256:
  `47D8E0A89FC3F37EA6B74E807DCF566FBDD6D79F734D5A05CFD56124888EB2EB`.
