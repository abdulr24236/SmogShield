# SmogShield — Deliverable 3 (Track A: Application)

Final Deployment & Technical Report. Track A of the deliverable-3 specification: a
user-facing Streamlit application that serves real-time next-day air-quality
forecasts (PM2.5 regression + six-class AQI category classification) from the
frozen SmogShield modelling pipeline.

## What this folder contains

- `app/app.py` — Streamlit frontend (single-day forecast form + results)
- `src/inference.py` — loads the frozen HGB pipelines and serves predictions, intervals, explanations, and guidance
- `src/train_pipeline.py` — thin wrapper that refits models from the committed SmogShield `src/` scripts
- `src/ablation.py` — weather/calendar feature-set ablation for both tasks
- `src/hourly.py` — historical hour-of-day PM2.5 profile for outdoor planning
- `report/report_d3.pdf` — compiled final report (47 pages)
- `report/` — LaTeX source, report figures, and the figure-generation script
- `requirements.txt` — pinned Python dependencies
- `models/` — fitted model artifacts (created at runtime; **not** committed)
- `tests/` — 14 tests for the inference and app contract
- `setup_env.bat`, `setup_models.bat`, `run_app.bat`, `run_ablation.bat` — double-click Windows launchers

## How models are produced

No fitted model binaries are committed. The pipeline is refit from fixed seeds by
the committed scripts in the parent `SmogShield/src/` (classification, regression,
bias_analysis). `src/train_pipeline.py` runs that refit and persists the artifacts
to `models/`. The app loads them on startup; if they are missing, `src/inference.py`
raises a clear error pointing at `setup_models.bat` / `python -m src.train_pipeline`
rather than silently serving a partial model.

The refit reproduces the Deliverable 2 selected models exactly: validation/test
macro F1 0.4941/0.4780 for classification and RMSE 45.6243/64.9198 for
regression. It writes six artifacts:

| Artifact | Purpose |
|---|---|
| `classification_pipeline.joblib` | six-class next-day AQI category |
| `regression_pipeline.joblib` | next-day PM2.5 point forecast |
| `regression_q10_pipeline.joblib` | lower 10% quantile |
| `regression_q90_pipeline.joblib` | upper 90% quantile |
| `metadata.json` | feature contract, seasonally plausible monthly defaults, interval calibration, permutation importance, seasonal/stagnation summary |
| `ablation.json` | feature-set ablation results (written by `src/ablation.py`) |

Training uses the full chronological training split; the scorable-row mask is used
for evaluation only, matching the Deliverable 2 contract.

## Run locally

On Windows, the simplest path is to double-click `setup_env.bat` once (it creates
a project-local `.venv` and installs the pinned dependencies), then
`setup_models.bat` (one-time refit), then `run_app.bat`. The launchers use
`.venv` if it exists and otherwise fall back to `py` or `python` on `PATH`; they
do not depend on a machine-specific virtual-environment path.

The command-line equivalent, from this folder, is:

```bash
python -m src.train_pipeline          # refit + persist model artifacts (one-time)
python -m src.ablation                # optional: weather/calendar ablation table
python -m unittest discover -s tests -t .   # 14 tests
python -m streamlit run app/app.py    # open the app
```

The app reads today's PM2.5, AQI category, weather, and calendar inputs and returns
tomorrow's predicted PM2.5 concentration and AQI category, alongside the
persistence baseline (tomorrow = today) for comparison. The numeric AQI feature is
derived from the entered PM2.5 with the same EPA conversion used for the labels, so
the feature row stays consistent on clean and hazardous days. Initial PM2.5 and
category values are the historical medians for the current month (e.g., 234 µg/m³
and Very Unhealthy in November), so the app opens on a seasonally plausible
scenario; adjust them to your local reading.

### Two category views

The results panel deliberately shows the AQI category two ways:

1. **Classifier category** — the six-class HGB classifier's arg-max label, with its
   class probabilities. In the Deliverable 2 evaluation this model reached a test
   macro-F1 of 0.478 versus 0.516 for persistence, and it is strongly biased toward
   the majority class (`Unhealthy`). Small input changes often leave the top label
   unchanged even though the probability bars move.
2. **Implied AQI category** — the regression's predicted PM2.5 converted through the
   same US-EPA PM2.5 breakpoints (2012 version) that were used to build the training
   labels (`data_scripts/03_pm25_aqi_and_eda.py`). This view is monotonic in PM2.5
   and therefore always responds to input changes.

When the two disagree the app flags it explicitly. This is intentional: it keeps the
demo honest about the classifier's weakness instead of hiding it behind a single
label. The breakpoints are implemented once in `src/inference.py`
(`pm25_to_aqi` / `aqi_to_category`) and covered by boundary tests in
`tests/test_inference.py`.

## Uncertainty, explanation, and guidance

**80% prediction interval.** Two quantile HGB models (10th and 90th percentile)
use the same split and preprocessing as the point model. Their bounds are widened
by a factor chosen on the validation split to reach nominal coverage; the test
split is used only to report coverage. Current empirical coverage is **80.3% on
validation** and **77.0% on test** for an 80% interval. The test figure is the
honest out-of-sample number; the gap reflects chronological drift and the small
sample.

**Per-forecast explanation.** For each forecast the app perturbs one input at a
time by a fixed, interpretable amount (PM2.5 +25 µg/m³, temperature +5 °C,
humidity +10 points, wind +5 km/h) and shows how the PM2.5 forecast moves. The
larger PM2.5 step avoids misleading zero-effect results from piecewise-constant
tree predictions. This is a model-agnostic local sensitivity analysis, not a
causal claim.

**Global importance.** `train_pipeline.py` computes permutation importance on the
validation split (increase in RMSE for regression, drop in macro F1 for
classification) and stores the top features in `metadata.json`. The app shows the
regression ranking in an expander.

**Guidance.** The app maps the implied AQI category to cautious, non-medical
advice and repeats that it is not a substitute for official health advice.

**Winter stagnation pattern.** Using the feature table, a winter day (Dec-Feb) is
labelled stagnant when its mean wind speed is in the winter lower quartile and it
recorded at most 0.5 mm of rain. On the current data, 24.6% of winter days (100 of
407) meet this definition. Their mean PM2.5 is **294.9 µg/m³** versus
**193.7 µg/m³** on better-ventilated days, a **+52%** difference. Seasonal means
are winter 218.6, autumn 146.0, spring 77.7, and summer 55.1 µg/m³. These are
descriptive statistics, not a dispersion model.

**Best time outdoors (historical).** `src/hourly.py` reads the cleaned hourly
ground-truth file (`data/processed/pm25_hourly.csv`, 36,820 valid hours from
2019-05-23 to 2025-02-18), removes flagged outliers, and averages PM2.5 by local
hour and season. The cleanest three-hour window is **14:00–16:59**, averaging
**85.2 µg/m³**, versus **138.4 µg/m³** at the worst hour (22:00). The afternoon
minimum holds in every season, but the winter window still averages
**155.4 µg/m³**, so timing helps without making winter air safe. This is a
station-specific historical description, not a today-specific hourly forecast.

The hourly source file is not committed (`data/processed/` is gitignored). If it
is absent, the app simply omits the card and the rest of the pipeline still runs.

## Ablation: do weather and calendar features help?

`python -m src.ablation` refits both tasks on three nested feature sets with the
frozen hyperparameters. Test metrics are reported but never used to choose a set.

| Task | Feature set | Features | Validation | Test |
|---|---|---:|---:|---:|
| Regression (RMSE) | history | 12 | 52.18 | 63.88 |
| Regression (RMSE) | history + calendar | 29 | 49.97 | 65.16 |
| Regression (RMSE) | full (adds weather) | 46 | **45.62** | 64.92 |
| Classification (macro F1) | history | 12 | 0.4663 | 0.4715 |
| Classification (macro F1) | history + calendar | 29 | 0.4319 | **0.5155** |
| Classification (macro F1) | full (adds weather) | 46 | **0.4941** | 0.4780 |

Interpretation, stated cautiously:

- **Weather improves validation RMSE but not test RMSE.** The full model is best
  on validation (45.62) yet slightly worse than the history-only model on test
  (64.92 versus 63.88), a difference of about 1 µg/m³.
- **Calendar proxies do not help regression** on test (65.16 versus 63.88).
- **Classification is unstable across splits.** Calendar features are best on test
  (0.5155) but worst on validation (0.4319); the full model is best on validation
  (0.4941) but only mid-table on test (0.4780).
- **Conclusion:** there is no consistent evidence that adding weather or calendar
  features improves held-out accuracy on this dataset. Traffic is not observed
  directly; calendar variables are proxies, so this study cannot isolate a traffic
  effect.

This is a negative but legitimate finding: it explains why the persistence
baseline remains competitive and why the app presents forecasts with explicit
uncertainty and caveats.

## Final report

The merged Deliverable 1–3 report is compiled at:

- `report/report_d3.pdf` — 47 pages, including D1 Phase 1–2, the reused D2 Phase
  3–4 body, the new D3 Phase 5–6 chapters, viva Q\&A, glossary, file map, and
  references.
- `report/report_d3.tex` — master LaTeX source.
- `report/phase56_final.tex`, `report/phase34_final.tex`,
  `report/appendix_final.tex` — chapter sources.
- `report/make_figures.py` — regenerates the report figures and
  `report/examples.json` from the frozen artifacts.

Build the report from `report/` with:

```bash
tectonic report_d3.tex
```

## Submission artefacts

- GitHub repo: production-ready (`requirements.txt`, descriptive `README.md`, all source)
- Compiled final report: `report/report_d3.pdf` (Deliverables 1–3, Phases 1–6)
- Working Streamlit application and 14 passing tests
- Reproducible artifact pipeline: no fitted binaries or processed data committed
