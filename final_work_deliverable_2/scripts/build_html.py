#!/usr/bin/env python
"""Build a printable HTML version of the Deliverable 2 report.

Run from the package root:

    python scripts/build_html.py

The script reads the frozen tables in ``tables/`` and figures in ``figures/``
and writes ``report_d2.html``.  The HTML is the source used for the PDF built
by ``build_report.ps1``.  It does not rerun any model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"


def table(name: str, number: int, caption: str, note: str = "") -> str:
    df = pd.read_csv(TABLES / f"{name}.csv")
    body = df.to_html(index=False, escape=True, classes="data", border=0)
    note_html = f'<p class="table-note">{note}</p>' if note else ""
    return (
        f'<div class="table-block" id="table-{number}">'
        f'<div class="table-title">Table {number}. {caption}</div>'
        f'{body}{note_html}</div>'
    )


def figure(filename: str, number: int, caption: str, css: str = "wide") -> str:
    path = FIGURES / filename
    if not path.exists():
        return ""
    return (
        f'<figure class="report-figure {css}" id="figure-{number}">'
        f'<img src="figures/{filename}" alt="{filename}">'
        f'<figcaption>Figure {number}. {caption}</figcaption></figure>'
    )


def main() -> None:
    with (METRICS / "classification_metrics.json").open(encoding="utf-8") as fh:
        clf = json.load(fh)
    with (METRICS / "regression_metrics.json").open(encoding="utf-8") as fh:
        reg = json.load(fh)
    with (METRICS / "bias_subgroup_metrics.json").open(encoding="utf-8") as fh:
        bias = json.load(fh)

    selected = clf["selected_model_metrics"]
    test = selected["test"]
    val = selected["validation"]
    reg_test = reg["hist_gb_test"]
    reg_val = reg["hist_gb_validation"]
    ridge_test = reg["ridge_test"]
    pers_test = reg["persistence_test"]

    f1_test = clf["selected_model_metrics"]["test"]["f1_macro"]
    f1_val = clf["selected_model_metrics"]["validation"]["f1_macro"]
    pers_f1_test = clf["baselines"]["persistence_test"]["f1_macro"]
    acc_test = clf["selected_model_metrics"]["test"]["accuracy"]
    pers_acc_test = clf["baselines"]["persistence_test"]["accuracy"]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SmogShield Lahore — Deliverable 2 (Phase 3 and Phase 4)</title>
<style>
  @page {{ size: A4; margin: 17mm 16mm 18mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    color: #1d2733; line-height: 1.42; font-size: 10.2pt;
    margin: 0; padding: 0;
  }}
  h1 {{ font-size: 22pt; line-height: 1.15; text-align: center; margin: 0 0 2mm 0; }}
  h2 {{ font-size: 15pt; margin: 8mm 0 2mm 0; padding-bottom: 1mm; border-bottom: 2px solid #2c5f7c; color: #173b51; page-break-after: avoid; }}
  h3 {{ font-size: 12pt; margin: 5mm 0 1.5mm 0; color: #1f4d68; page-break-after: avoid; }}
  p {{ margin: 0 0 2.5mm 0; text-align: justify; }}
  ul, ol {{ margin: 0 0 2.5mm 5mm; padding-left: 4mm; }}
  li {{ margin-bottom: 1.2mm; }}
  .subtitle {{ text-align: center; font-size: 12pt; color: #36566a; margin-bottom: 2mm; }}
  .authors {{ text-align: center; font-size: 10.5pt; margin-bottom: 1mm; }}
  .date {{ text-align: center; color: #586b78; margin-bottom: 6mm; }}
  .abstract {{
    background: #f2f7fa; border-left: 4px solid #2c5f7c;
    padding: 4mm 5mm; margin: 3mm 0 5mm 0; font-size: 9.7pt;
  }}
  .key-box {{
    border: 1px solid #b7cbd7; background: #f8fbfc; padding: 3mm 4mm;
    margin: 3mm 0 4mm 0; page-break-inside: avoid;
  }}
  .key-box strong {{ color: #173b51; }}
  .table-block {{ margin: 3mm 0 5mm 0; page-break-inside: avoid; }}
  .table-title {{ font-weight: 600; color: #173b51; margin-bottom: 1mm; font-size: 9.4pt; }}
  table.data {{ width: 100%; border-collapse: collapse; font-size: 8.4pt; }}
  table.data th {{ background: #e7f0f5; color: #173b51; border-top: 1.5px solid #4d7e98; border-bottom: 1px solid #4d7e98; padding: 1.4mm 1.2mm; text-align: left; }}
  table.data td {{ border-bottom: 0.5px solid #cddbe3; padding: 1.1mm 1.2mm; vertical-align: top; }}
  table.data tr:nth-child(even) td {{ background: #fafcfd; }}
  table.data th:not(:first-child), table.data td:not(:first-child) {{ text-align: right; }}
  .table-note {{ font-size: 8.1pt; color: #526875; margin-top: 1mm; text-align: left; }}
  .report-figure {{ margin: 4mm 0 5mm 0; page-break-inside: avoid; text-align: center; }}
  .report-figure img {{ max-width: 100%; height: auto; }}
  .report-figure.wide img {{ width: 100%; }}
  .report-figure.medium img {{ width: 78%; }}
  figcaption {{ font-size: 8.3pt; color: #526875; margin-top: 1.2mm; text-align: left; }}
  .small {{ font-size: 9pt; color: #526875; }}
  .mono {{ font-family: Consolas, "Courier New", monospace; font-size: 8.7pt; }}
  .pagebreak {{ page-break-before: always; }}
  .two-col {{ display: flex; gap: 5mm; }}
  .two-col > div {{ flex: 1; }}
  .footer-note {{ margin-top: 8mm; border-top: 1px solid #b7cbd7; padding-top: 2mm; font-size: 8.2pt; color: #586b78; }}
  code {{ font-family: Consolas, "Courier New", monospace; font-size: 8.7pt; background: #f1f5f7; padding: 0 1mm; }}
  @media print {{
    h2 {{ break-after: avoid; }}
    .table-block, .report-figure, .key-box {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<h1>SmogShield Lahore</h1>
<div class="subtitle">Deliverable 2: Phase 3 (Model Development) and Phase 4 (Evaluation and Bias Analysis)</div>
<div class="authors">Muhammad Haider Mughal (3006) &nbsp;|&nbsp; Muhammad Ayub Butt (3076) &nbsp;|&nbsp; Abdul Rehman (3059)</div>
<div class="date">9 November 2026</div>

<div class="abstract">
<b>Abstract.</b> This report documents the Deliverable 2 work for SmogShield and covers Phase 3 (model development) and Phase 4 (evaluation and bias analysis). The primary task is six-class next-day AQI-category classification; the secondary task is next-day PM2.5 regression. All experiments use one chronological split, a seven-day purge gap, train-only preprocessing, and a test set touched once per final model. The selected classifier, HistGradientBoosting, achieved a validation macro F1 of {f1_val:.4f} and a test macro F1 of {f1_test:.4f}. It did not beat the persistence baseline on test ({pers_f1_test:.4f}), although its accuracy was slightly higher ({acc_test:.4f} versus {pers_acc_test:.4f}). For regression, HistGradientBoosting was selected on validation RMSE ({reg_val['rmse']:.4f} versus Ridge {reg['ridge_validation']['rmse']:.4f}); its test RMSE was {reg_test['rmse']:.4f}, essentially tied with persistence ({pers_test['rmse']:.4f}, a 0.10 gap and skill −0.0016). Ridge did beat persistence on test with RMSE {ridge_test['rmse']:.4f}. The bias analysis shows especially large errors on the most severe pollution days; because severity groups are defined by the observed outcome, the sign is partly constructed, and the error magnitude is the informative result. The report states these results as observed rather than tuning the test set after the fact.
</div>

<div class="key-box">
<strong>Key outcomes.</strong>
Classification HGB did <b>not</b> beat persistence on test macro F1
({f1_test:.4f} versus {pers_f1_test:.4f}). Regression HGB was selected on
validation only; its test RMSE was <b>essentially tied</b> with persistence
({reg_test['rmse']:.4f} versus {pers_test['rmse']:.4f}, a 0.10 gap and skill
−0.16%). Ridge did beat persistence ({ridge_test['rmse']:.4f}, skill +4.20%),
but switching to it after seeing test would violate the selection rule. The
final modelling code state is <code>a55a7f5</code> on <code>main</code>;
the README was updated at <code>56dea73</code>.
</div>

<h2>1. Introduction and objectives</h2>
<p>SmogShield forecasts next-day air quality for Lahore, Pakistan. The modelling work has two tasks: (1) predict the next-day AQI category among six classes, and (2) predict next-day PM2.5 concentration as a continuous value. The primary task supports the classification analysis required for the project: accuracy, balanced accuracy, macro precision, macro recall, macro F1, weighted F1, one-vs-rest macro AUC-ROC, a confusion matrix, and per-class precision/recall/F1. Macro metrics are primary because the classes are imbalanced. The regression task reports RMSE, MAE, and R², and compares every model with a persistence baseline.</p>
<p>The guiding rule is that tomorrow's value is unknown at forecast time. Only today's PM2.5, AQI, weather, and calendar information may be used as model inputs. No feature may be derived from <code>shift(-1)</code> or from the test period.</p>

<h3>1.1 Phase coverage</h3>
<p>This report is the Deliverable 2 submission for Phase 3 and Phase 4. Sections 4–6 cover Phase 3: the model and baseline protocol, classification results, and regression results. Sections 7–10 cover Phase 4: bias and error analysis, limitations and recommendations, reproducibility, and conclusions. Sections 2–3 document the shared data and preprocessing contract that both phases depend on, and Section 1.2 gives the architecture.</p>

<h3>1.2 System architecture</h3>
<p>Figure 1 summarises the Deliverable 2 path from raw sources to the report. Phase 2 cleaning and feature building produce the shared daily table. The shared loader applies the chronological split, the Good/Moderate label merge, train-only preprocessing, leakage guards, and the scorable-row mask. Classification and regression then run as separate model tracks. Their final predictions feed the shared metrics, the bias/error analysis, the frozen output files, and the report. A future application layer would consume the same forecasts and charts; it would not retrain models. The web application itself is future work and is not part of this deliverable.</p>
{figure("architecture.png", 1, "Deliverable 2 modelling and evaluation architecture. The shared loader is the single source of split, label, preprocessing, and scorable-row rules. The application forecasts shown in the last box are future work and are not part of Deliverable 2.", "wide")}

<h2>2. Data and chronological split</h2>
<p>The shared input table is <code>data/processed/features_daily.csv</code>. It contains 2,113 daily rows and 63 columns. It is generated by the Phase 2 feature script and is not committed to Git; it must be restored from the shared Drive or regenerated. The final run used the following split.</p>
{table("split_summary", 1, "Chronological split with a seven-day purge gap. Scorable rows require a finite current-day PM2.5 value.")}
<p>The split is chronological, not random. The seven-day gap removes 2023-07-01 through 2023-07-07 and 2024-07-01 through 2024-07-07 so that a next-day target at the end of one period cannot overlap the first rows of the next period. The validation end is 2024-06-03 rather than the configured 2024-06-30 because the remaining June rows do not have a usable next-day target. This is expected.</p>
<p>The split contains 1,022 training rows, 250 validation rows, and 223 test rows. The scorable-row mask, which requires a finite current-day PM2.5 value, leaves 917 training rows, 244 validation rows, and 222 test rows. All methods, including persistence, are scored on the same scorable rows. Six validation rows and one test row are dropped for this reason. Classification and regression each produce 473 prediction rows (250 validation plus 223 test), of which 466 are scorable.</p>

<h3>2.1 Target merge and class support</h3>
<p>The raw table has seven AQI categories. Good has only nine training rows and zero validation and test rows, so it cannot support a separate class. For modelling, Good is merged into Moderate. This is a label decision, not a deletion: the raw category table still contains all seven categories.</p>
{table("raw_category_counts", 2, "Raw category counts before the Good-to-Moderate merge.")}
<p>After the merge, the six-class support is shown below. Every class has support in every split, so multiclass AUC-ROC is defined. The eight test rows in Good/Moderate and the eleven validation rows in Hazardous 401–500 are small; their per-class precision and recall are noisy, not precise estimates. The two Hazardous bands remain separate in the main analysis.</p>
{table("support_by_split", 3, "Six-class support by chronological split. Good and Moderate are merged for modelling.")}
<p>Table 2 counts the current-day AQI category over all feature-table rows, while Table 3 counts next-day targets inside the chronological splits. Their totals therefore differ (1,506 versus 1,495).</p>

<h2>3. Features, leakage controls, and preprocessing</h2>
<p>The shared loader drops <code>date</code>, every <code>target_next_*</code> column, the precomputed <code>*_scaled</code> weather columns, the redundant <code>season</code> string, and <code>year</code> from model inputs. The last choice avoids asking trees to extrapolate to unseen years and avoids unstable linear year trends. Month, day-of-year, and cyclical encodings carry seasonality.</p>
<p>The loader keeps current-day PM2.5, AQI, weather, and calendar features. It one-hot encodes the current AQI category, median-imputes numeric features, adds missing-value indicators, and applies a robust scaler. Imputers and scalers are fitted on the training split only. In the model frames, <code>pm25_lag1</code> is missing for 13.4% of training rows, 4.0% of validation rows, and 0.4% of test rows, so missing-value indicators are retained rather than treated as pure noise. The loader also raises if a target or precomputed scaled column survives the config lists. The final transformed feature count is 80, from 46 raw features, confirmed by <code>check_setup</code>.</p>
<p>The loader uses the fixed class labels in <code>configs/base.yaml</code>; labels are never inferred alphabetically from the data. This prevents validation or test labels from changing the class order. The optional label merge is applied only to the classification target, before encoding. The final run used seed 42 and train-only fitting throughout.</p>

<h2>4. Phase 3: Models, baselines, and selection protocol</h2>
<h3>4.1 Classification</h3>
<p>The classification candidates were majority class, persistence, logistic regression with class weights, histogram gradient boosting (HGB), and an optional small MLP retained only if it improved validation performance. The persistence baseline uses the majority training class code, 2 (Unhealthy), as its fallback when today's category is missing. The MLP was excluded after validation macro F1 of 0.4124 did not exceed HGB's 0.4941.</p>
<p>Hyperparameters were tuned with five-fold <code>TimeSeriesSplit</code> on the raw training frame, with a seven-day gap. Tuning never cross-validated on the already-preprocessed training matrix. The final model was refitted on the training split only, not on train plus validation. The test set was then touched once for the final model. The classification selection metric was validation macro F1 over all six configured labels. The CV score is computed per fold over the classes present in that fold, so it is not directly comparable with the six-label validation score; that caveat is recorded in the metrics file.</p>
{table("hyperparameters", 4, "Selected hyperparameters. Classification used five-fold time-series CV with a seven-day gap and selected on validation macro F1; regression selected on validation RMSE.", "Status “completed” means tuned but not selected; “excluded” means the optional MLP did not beat HGB on validation.")}
<p>Regression used persistence, Ridge, and histogram gradient boosting, with the same five-fold time-series procedure and seven-day gap. Model selection used validation RMSE. The selected model was refitted on training rows only and evaluated once on test. Persistence is not imputed for missing current-day PM2.5; it is scored on the same scorable rows as the models. The final code calls the metric helper with <code>allow_missing=False</code> for every method. The option <code>allow_missing=True</code> remains reserved for an unfiltered persistence baseline only; the persistence <code>n_dropped</code> values are 6 validation and 1 test, the same rows on which every method is scored.</p>

<h2>5. Phase 3: Classification results</h2>
<p>Table 5 reports validation and test metrics for the majority, persistence, logistic regression, and HGB candidates. HGB was selected on validation macro F1 (0.4941 versus 0.4665 for logistic regression). On test, HGB reached macro F1 0.4780, below persistence's 0.5157. HGB's test accuracy was higher than persistence (0.5856 versus 0.5676), but accuracy is dominated by the larger classes; macro F1 exposes the weaker performance on rare classes.</p>
{table("classification_metrics", 5, "Classification results on the chronological validation and test splits. Macro F1 is the selection metric.")}
<p>The selected HGB model's test AUC-ROC is 0.8886 and logistic regression's is 0.8719. The one-vs-rest macro AUC is computed from the aligned six-column probability matrix. Rows sum to one within the helper's tolerance. The majority baseline has no probability output, so its AUC is reported as undefined rather than filled with a fake value. HGB's test AUC (0.8886) is higher than its validation AUC (0.8765) even though its test macro F1 is below persistence. AUC measures probability ranking rather than the thresholded label decision, and persistence has no probability output, so the two metrics answer different questions. Persistence's test macro precision, recall, and F1 are all 0.5157, and its weighted F1 equals accuracy (0.5676), because predicted and true class counts coincide exactly on the test rows. This is not a copy error; it is a property of the persistence shift on this series.</p>
{table("classification_per_class_test", 6, "Per-class test performance of the selected HistGradientBoosting classifier. Good/Moderate and Hazardous 401–500 have small support and should be treated as noisy estimates.")}
<p>The largest class, Unhealthy, has F1 0.7013, while Very Unhealthy has F1 0.7040. The two Hazardous bands are weaker: 0.3600 and 0.2353. The confusion matrix below shows frequent confusion between Unhealthy for Sensitive and Unhealthy, and between the two Hazardous bands and Very Unhealthy. This is consistent with ordered severity classes and with the class imbalance.</p>
{table("classification_confusion_test", 7, "Selected HGB test confusion matrix. Codes: 0 Good/Moderate, 1 Unhealthy for Sensitive, 2 Unhealthy, 3 Very Unhealthy, 4 Hazardous 301–400, 5 Hazardous 401–500.")}
{figure("classification_confusion_histgradientboosting_test.png", 2, "Selected HGB test confusion matrix. The raw matrix is also given in Table 7.", "medium")}
{figure("classification_hgb_val_curve.png", 3, "Validation macro F1 for the HGB configuration used in model selection.", "medium")}
{figure("classification_lr_C_curve.png", 4, "Logistic-regression validation curve over the regularisation strength C. The selected value was C = 0.1.", "medium")}
<p><b>Training curves.</b> A per-epoch training-loss curve exists only for the MLP, which was excluded because its validation macro F1 (0.4124) did not exceed that of HGB (0.4941). For logistic regression (full-batch solver) and HGB (no per-epoch loss recorded in our configuration) we report validation-score curves instead (Figures 3, 4, and 9). Figure 3 is nearly flat (macro F1 between about 0.490 and 0.502), so the iteration count is not a sensitive parameter.</p>

<h2>6. Phase 3: Regression results</h2>
<p>Table 8 reports the regression comparison. On validation, HGB had the lowest RMSE (45.6243), narrowly ahead of Ridge (45.6916); both beat persistence (51.5457) by about 11.4–11.5%. On test, however, HGB reached 64.9198, essentially tied with persistence (64.8175, a 0.10 RMSE gap); its skill score was −0.0016. Ridge reached 62.0956 on test and beat persistence, with skill 0.0420. R² follows the same pattern: Ridge's test R² is 0.7356, HGB's is 0.7110, and persistence's is 0.7119.</p>
<p>This is a genuine validation-to-test reversal, not a reporting error. The selection rule chose HGB on validation RMSE before test was touched. The report keeps that selection and reports the test outcome honestly. It does not switch the final model to Ridge after seeing test.</p>
{table("regression_metrics", 8, "Regression results on the chronological validation and test splits. Skill is 1 − RMSE_model / RMSE_persistence.")}
<p>All models are scored on the same 244 validation and 222 test scorable rows; the 6 validation and 1 test rows dropped for missing current-day PM2.5 are not scored.</p>
{figure("regression_metric_comparison.png", 5, "Regression metric comparison across candidate models and splits.", "medium")}
{figure("regression_hist_gb_pred_vs_actual.png", 6, "Selected HGB regression: predicted versus actual next-day PM2.5.", "wide")}
{figure("regression_hist_gb_residuals_over_time.png", 7, "Selected HGB regression residuals over time.", "wide")}
{figure("regression_hist_gb_error_by_severity.png", 8, "Selected HGB regression error by pollution severity.", "medium")}
{figure("regression_hgb_learning_curve.png", 9, "HGB regression learning curve used as training diagnostic material.", "medium")}

<h2>7. Phase 4: Bias and error analysis</h2>
<p>The bias analysis joins the final regression and classification predictions on date and split. It uses 466 joined rows: 244 validation and 222 test. The subgroup table has 47 rows across seven dimensions: AQI category, season, year, weekday versus weekend, Islamic events, public holidays, and PM2.5 severity band. Missing subgroup keys were counted per split and dimension; all counts were zero in the final join.</p>
<p>The joined validation rows contain 25 scorable Ramadan rows and no scorable Eid rows. The feature table flags Eid on 10–12 April and 17–19 June 2024, but the AQI label is missing on those dates; all six dates also lack a next-day target, so they never enter the split. This is a label-coverage limitation, not a calendar-flag gap. The test split has only one Islamic-event group, <code>Normal</code>; no test Ramadan or Eid comparison is therefore possible. Public-holiday test rows number only seven, so the holiday result is marked as small-support. The analysis is environmental and temporal, not demographic: the dataset has one monitoring station and no demographic attributes. It cannot establish fairness across people or neighbourhoods.</p>
{table("bias_missing_counts", 9, "Missing subgroup keys after joining regression and classification predictions.")}
<p>Table 10 reports the test subgroups. Positive signed error is y_true − y_pred, so a positive value means under-prediction relative to the observed PM2.5. The most important pattern is the severe-pollution tail. Hazardous 401–500 has RMSE 162.14 and signed error 141.70; the 301–500 severity band has RMSE 99.60 and signed error 75.72; and the 500+ band, with only five rows, has RMSE 242.00 and signed error 230.76. These are large errors, not just noise, although the 500+ estimate is unstable because of its tiny support. Severity bands and AQI-category groups are defined by the observed outcome, so high-severity groups look under-predicted and low-severity groups over-predicted partly by construction. The size of the errors (for example RMSE 99.60 on the 301–500 band), not the sign, is the informative result.</p>
{table("bias_test", 10, "Test-set subgroup error and classification performance. Positive signed error means under-prediction. Small-support groups (n < 30) are identified in the final column.", "Subgroup macro F1 is computed within each subgroup over the true classes present, so it is not comparable across groups or with Table 5. For AQI Category rows, use accuracy (= recall). The full 14-chart bias set is included in figures/; the report shows selected test charts below.")}
<p>Seasonal errors are also uneven. Winter test RMSE is 72.38 with signed error −17.43, indicating average over-prediction, while autumn RMSE is 75.32 with signed error +14.15, indicating average under-prediction. Summer is much easier (RMSE 19.65). Test 2025 has only 45 rows (1 January–16 February, winter), whereas 2024 has 177 mixed-season rows. The 2025 comparison is therefore confounded with season: RMSE is 69.86 versus 63.60, and macro F1 is 0.4306 versus 0.4832, but this is not a clean temporal comparison. Weekday RMSE is higher than weekend RMSE (68.32 versus 55.64), though weekend macro F1 is lower (0.4276 versus 0.4875).</p>
{figure("bias_Severity_Band_test.png", 10, "Test signed error and classification accuracy by PM2.5 severity band. Because groups are defined by the observed outcome, the sign is partly constructed; the error magnitude is the informative result. Small groups are marked with an asterisk.", "wide")}
{figure("bias_Season_test.png", 11, "Test signed error and classification accuracy by season.", "wide")}
{figure("bias_AQI_Category_test.png", 12, "Test signed error and classification accuracy by observed AQI category.", "wide")}
{figure("bias_Year_test.png", 13, "Test signed error and classification accuracy by year.", "wide")}
{figure("bias_Weekday_Weekend_test.png", 14, "Test signed error and classification accuracy for weekday versus weekend rows.", "wide")}
{figure("bias_Public_Holiday_test.png", 15, "Test signed error and classification accuracy for normal days versus public holidays. Holiday support is only seven rows.", "wide")}
<p>The class mix also shifts across periods. Unhealthy is about 48% of training rows but 32% of validation and 35% of test. The two Hazardous bands together are about 10% of training but 22% of validation and 19% of test. This may reflect seasonal composition, missing training days or targets, or real worsening; the data do not identify a single cause. The later-period under-prediction of severe days is consistent with that shift, but it is not proof of a causal trend.</p>

<h2>8. Phase 4: Limitations and recommendations</h2>
<ul>
  <li><b>Rare-class uncertainty.</b> Good/Moderate has 8 test rows and Hazardous 401–500 has 19. Their per-class precision/recall are noisy. The same applies to small bias subgroups such as public holidays and the 500+ severity band.</li>
  <li><b>Single station.</b> The dataset represents one monitoring location and has no demographic attributes. The bias check cannot support claims about population-level fairness.</li>
  <li><b>Distribution shift.</b> The class mix and the validation-to-test error pattern change over time. A model selected on validation can fail to beat persistence on a later period, as happened on both tasks (macro F1 below persistence for classification, and essentially tied on RMSE for regression).</li>
  <li><b>Coverage gaps.</b> The validation split contains 250 usable rows over the 326 days from 2023-07-14 to 2024-06-03 (about 77%), because many days lack an AQI label or a next-day target; the test split has 223 rows over the 224 days from 2024-07-08 to 2025-02-16 and contains no spring rows. Validation and test therefore cover different seasonal mixes, which limits how directly their metrics can be compared.</li>
  <li><b>Macro F1 versus accuracy.</b> HGB's test accuracy is higher than persistence's, but its macro F1 is lower. Reporting only accuracy would obscure the rare-class weakness.</li>
  <li><b>No traffic data.</b> Calendar features are proxies; no open, time-aligned Lahore traffic dataset was available. This limits the explanatory scope of the features.</li>
  <li><b>No test-time retuning.</b> The report preserves the validation-based selection. Switching to Ridge after seeing test would be post-hoc and would invalidate the test-once rule.</li>
</ul>
<p>Recommended next steps are to collect more severe-pollution days, evaluate cost-sensitive or class-balanced variants with a pre-registered validation rule, add calibrated probabilities and prediction intervals, and test on a later period or a second station. Any such change should be selected without using the current test set.</p>

<h2>9. Phase 4: Reproducibility</h2>
<p>The final results were produced from merge commit <code>a55a7f5</code> on <code>main</code>; the repository README was updated afterwards (commit <code>56dea73</code>). The final pipeline was run from the repository root with the commands below.</p>
<pre class="mono">python -m src.check_setup
python -m unittest discover -s tests -t .
python -m src.classification
python -m src.regression
python -m src.bias_analysis</pre>
<p><code>check_setup</code> reported 0 bad next-day PM2.5 values, 0 bad lag-1 values, and 0 bad next-day categories. The unit tests ran 15 tests and passed. Classification completed in 427.92 seconds and regression in 37.61 seconds. The bias analysis then joined the final prediction files.</p>
<p>The classification prediction file has 473 rows and 12 columns; 466 rows are scorable. Its SHA-256 hash is <code>47D8E0A89FC3F37EA6B74E807DCF566FBDD6D79F734D5A05CFD56124888EB2EB</code>, matching the verified Drive copy. Regression also has 473 prediction rows, 466 scorable.</p>
<p>Dependencies are listed in <code>requirements.txt</code>. The final environment used pandas 3.0.3, scikit-learn 1.9.1, and matplotlib 3.11.2. The seed is 42 from <code>configs/base.yaml</code>. Generated prediction CSVs, metrics JSON, and charts are review artifacts and should not be committed to Git; the data and large artifacts belong on the shared Drive or a release link. No fitted model weights are published: the selected pipelines are refit from fixed seeds by the committed scripts (see the repository README).</p>

<h3>9.1 Git-author contributions</h3>
<p>Table 11 is the Git-author contribution table: it maps each commit author to the project modules and representative commits they own. The repository history contains only the three project contributors. The Git identities used in the D2 work are Abdul Rehman &lt;arehman4275@gmail.com&gt;, Muhammad Haider Mughal &lt;haiderirfanmm@gmail.com&gt;, and Muhammad Ayub Butt &lt;purestudy83010@gmail.com&gt;. Some earlier commits by the same author also appear under the display name “Haider Mughal” with the same e-mail address. No AI or OpenCode co-author trailer was added.</p>
{table("git_authors", 11, "Git-author contribution table mapping commit authors to project modules. Exactly the three human project contributors are listed.")}

<h2>10. Phase 4: Conclusion</h2>
<p>The primary classification model, HGB, was selected on validation macro F1 and reached {f1_test:.4f} on test. It did not beat persistence's {pers_f1_test:.4f} macro F1, even though its accuracy was slightly higher. The secondary regression model, HGB, was selected on validation RMSE only; test results were not used to change the selection. Its test RMSE was {reg_test['rmse']:.4f}, essentially tied with persistence's {pers_test['rmse']:.4f} (a 0.10 gap and skill −0.0016). Ridge did beat persistence on test with RMSE {ridge_test['rmse']:.4f}, but switching to it after seeing test would violate the validation-only selection rule. The bias analysis shows that the largest error magnitudes occur on the most severe pollution days; because those groups are defined by the observed outcome, the sign is partly constructed, so the magnitude is the informative result. Seasonal and temporal errors are also uneven. These results are reported as observed, without retuning on the test set.</p>

<h2>References</h2>
<p class="small">U.S. Environmental Protection Agency, “Technical Assistance Document for the Reporting of Daily Air Quality — the Air Quality Index (AQI),” EPA-454/B-18-007, September 2018.</p>
<p class="small">World Health Organization, “Ambient (outdoor) air pollution,” fact sheet, 24 October 2024.</p>
<p class="small">R. Majeed et al., “Solving the mysteries of Lahore smog: the fifth season in the country,” Frontiers in Sustainable Cities, vol. 5, art. 1314426, 2024.</p>
<p class="small">F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.</p>
<p class="small">J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.</p>

<div class="footer-note">
Generated from the frozen final outputs on <code>main</code> at <code>a55a7f5</code>
(the README was updated at <code>56dea73</code>).
The LaTeX source is <code>report_d2.tex</code>; tables are in <code>tables/</code>,
figures in <code>figures/</code>, frozen metrics in <code>metrics/</code>, and
verification evidence in <code>evidence/</code>.
</div>

</body>
</html>
"""

    out = ROOT / "report_d2.html"
    out.write_text(html, encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
