#!/usr/bin/env python
"""Build the Deliverable 2 tables from the frozen final metrics.

Run from the package root:

    python scripts/build_tables.py

The script reads JSON/CSV files in ``metrics/`` and writes both CSV and
LaTeX versions into ``tables/``.  It does not retrain any model and does not
modify the original outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = [
    "Good/Moderate",
    "Unhealthy for Sensitive",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous (301-400)",
    "Hazardous (401-500)",
]

RAW_COUNTS = [
    ("Good", 9),
    ("Moderate", 126),
    ("Unhealthy for Sensitive", 269),
    ("Unhealthy", 647),
    ("Very Unhealthy", 256),
    ("Hazardous (301-400)", 139),
    ("Hazardous (401-500)", 60),
]


def esc(value) -> str:
    """Escape a value for ordinary LaTeX text."""
    s = "" if value is None else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def f(value, digits=4):
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def pct(value, digits=1):
    if value is None:
        return "--"
    return f"{100.0 * float(value):.{digits}f}\\%"


def write_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    path = TABLES / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_text(name: str, text: str) -> None:
    (TABLES / f"{name}.tex").write_text(text, encoding="utf-8")


def table_float(caption: str, label: str, body: str, size: str = r"\small") -> str:
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{size}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )


def main() -> None:
    with (METRICS / "classification_metrics.json").open(encoding="utf-8") as fh:
        clf = json.load(fh)
    with (METRICS / "regression_metrics.json").open(encoding="utf-8") as fh:
        reg = json.load(fh)
    with (METRICS / "classification_hyperparameters.json").open(encoding="utf-8") as fh:
        clf_hp = json.load(fh)
    with (METRICS / "regression_hyperparameters.json").open(encoding="utf-8") as fh:
        reg_hp = json.load(fh)
    with (METRICS / "bias_subgroup_metrics.json").open(encoding="utf-8") as fh:
        bias = json.load(fh)

    summary = clf["data_summary"]
    split = summary["split"]
    scorable = summary["scorable_rows"]

    # ---------------------------------------------------------------- support
    support_rows = []
    for label in CLASS_LABELS:
        vals = [summary["class_counts"][s][label] for s in ("train", "validation", "test")]
        support_rows.append(
            {
                "Category": label,
                "Train": vals[0],
                "Validation": vals[1],
                "Test": vals[2],
                "Total": sum(vals),
            }
        )
    totals = {
        "Category": "Total",
        "Train": sum(r["Train"] for r in support_rows),
        "Validation": sum(r["Validation"] for r in support_rows),
        "Test": sum(r["Test"] for r in support_rows),
        "Total": sum(r["Total"] for r in support_rows),
    }
    support_rows.append(totals)
    write_csv("support_by_split", support_rows, ["Category", "Train", "Validation", "Test", "Total"])
    body = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "AQI category & Train & Validation & Test & Total \\\\\n"
        "\\midrule\n"
        + "\n".join(
            f"{esc(r['Category'])} & {r['Train']:,} & {r['Validation']:,} & {r['Test']:,} & {r['Total']:,} \\\\"
            for r in support_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "support_by_split",
        table_float(
            "Six-class support by chronological split. Good and Moderate are merged for modelling.",
            "tab:support",
            body,
        ),
    )

    # ------------------------------------------------------------ raw counts
    raw_rows = [{"Category": c, "Days": n} for c, n in RAW_COUNTS]
    raw_rows.append({"Category": "Total", "Days": sum(n for _, n in RAW_COUNTS)})
    write_csv("raw_category_counts", raw_rows, ["Category", "Days"])
    body = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Raw AQI category & Days \\\\\n\\midrule\n"
        + "\n".join(f"{esc(r['Category'])} & {r['Days']:,} \\\\" for r in raw_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "raw_category_counts",
        table_float(
            "Raw category counts before the Good-to-Moderate merge.",
            "tab:raw-counts",
            body,
        ),
    )

    # ------------------------------------------------------------ split table
    split_rows = [
        {
            "Split": "Train",
            "Start": split["train_range"][0],
            "End": split["train_range"][1],
            "Rows": split["train_rows"],
            "Scorable": scorable["train"],
            "Gap (days)": split["gap_days"],
        },
        {
            "Split": "Validation",
            "Start": split["validation_range"][0],
            "End": split["validation_range"][1],
            "Rows": split["validation_rows"],
            "Scorable": scorable["validation"],
            "Gap (days)": split["gap_days"],
        },
        {
            "Split": "Test",
            "Start": split["test_range"][0],
            "End": split["test_range"][1],
            "Rows": split["test_rows"],
            "Scorable": scorable["test"],
            "Gap (days)": split["gap_days"],
        },
    ]
    write_csv("split_summary", split_rows, ["Split", "Start", "End", "Rows", "Scorable", "Gap (days)"])
    body = (
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Split & Start & End & Rows & Scorable & Gap \\\\\n\\midrule\n"
        + "\n".join(
            f"{r['Split']} & {r['Start']} & {r['End']} & {r['Rows']:,} & {r['Scorable']:,} & {r['Gap (days)']} \\\\"
            for r in split_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "split_summary",
        table_float(
            "Chronological split with a seven-day purge gap. Scorable rows require a finite current-day PM2.5 value.",
            "tab:split",
            body,
        ),
    )

    # -------------------------------------------------------- classification
    def clf_metrics(model_key: str, split_name: str) -> dict:
        if model_key == "majority":
            return clf["baselines"][f"majority_{split_name}"]
        if model_key == "persistence":
            return clf["baselines"][f"persistence_{split_name}"]
        if model_key == "logistic_regression":
            return clf["models"]["logistic_regression"][split_name]
        if model_key == "hgb":
            return clf["models"]["hgb"][split_name]
        raise KeyError(model_key)

    model_names = {
        "majority": "Majority class",
        "persistence": "Persistence",
        "logistic_regression": "Logistic regression",
        "hgb": "HGB (selected)",
    }
    clf_rows = []
    for model_key in ("majority", "persistence", "logistic_regression", "hgb"):
        for split_name in ("validation", "test"):
            m = clf_metrics(model_key, split_name)
            clf_rows.append(
                {
                    "Model": model_names[model_key],
                    "Split": split_name.title(),
                    "Accuracy": m["accuracy"],
                    "Balanced accuracy": m["balanced_accuracy"],
                    "Macro precision": m["precision_macro"],
                    "Macro recall": m["recall_macro"],
                    "Macro F1": m["f1_macro"],
                    "Weighted F1": m["f1_weighted"],
                    "AUC-ROC": m.get("roc_auc_ovr_macro"),
                }
            )
    write_csv(
        "classification_metrics",
        [
            {
                **{k: v for k, v in r.items() if k in ("Model", "Split")},
                **{k: f(r[k]) for k in ("Accuracy", "Balanced accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1", "AUC-ROC")},
            }
            for r in clf_rows
        ],
        ["Model", "Split", "Accuracy", "Balanced accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1", "AUC-ROC"],
    )
    body = (
        "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "Model & Split & Acc. & Bal. acc. & Macro P & Macro R & Macro F1 & Wtd. F1 & AUC \\\\\n\\midrule\n"
    )
    for r in clf_rows:
        name = esc(r["Model"])
        if r["Model"].startswith("HGB"):
            name = "\\textbf{HGB}"
        body += (
            f"{name} & {r['Split']} & {f(r['Accuracy'])} & {f(r['Balanced accuracy'])} & "
            f"{f(r['Macro precision'])} & {f(r['Macro recall'])} & {f(r['Macro F1'])} & "
            f"{f(r['Weighted F1'])} & {f(r['AUC-ROC'])} \\\\\n"
        )
    body += "\\bottomrule\n\\end{tabular}\n"
    write_text(
        "classification_metrics",
        table_float(
            "Classification results on the chronological validation and test splits. Macro F1 is the selection metric.",
            "tab:clf-metrics",
            body,
            size=r"\scriptsize",
        ),
    )

    # ------------------------------------------------- per-class test table
    test_per = clf["selected_model_metrics"]["test"]["per_class"]
    per_rows = []
    for label in CLASS_LABELS:
        m = test_per[label]
        per_rows.append(
            {
                "Category": label,
                "Support": int(m["support"]),
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1-score"],
            }
        )
    write_csv("classification_per_class_test", per_rows, ["Category", "Support", "Precision", "Recall", "F1"])
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Category & Support & Precision & Recall & F1 \\\\\n\\midrule\n"
        + "\n".join(
            f"{esc(r['Category'])} & {r['Support']} & {f(r['Precision'])} & {f(r['Recall'])} & {f(r['F1'])} \\\\"
            for r in per_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "classification_per_class_test",
        table_float(
            "Per-class test performance of the selected HistGradientBoosting classifier. Good/Moderate and Hazardous (401--500) have small support and should be treated as noisy estimates.",
            "tab:clf-per-class",
            body,
        ),
    )

    # ------------------------------------------------- confusion matrix table
    cm = clf["selected_model_metrics"]["test"]["confusion_matrix"]
    cm_rows = []
    for i, row in enumerate(cm):
        cm_rows.append({"True": i, **{f"Pred {j}": v for j, v in enumerate(row)}})
    write_csv("classification_confusion_test", cm_rows, ["True"] + [f"Pred {j}" for j in range(6)])
    body = (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "True & Pred 0 & Pred 1 & Pred 2 & Pred 3 & Pred 4 & Pred 5 \\\\\n\\midrule\n"
        + "\n".join(
            f"{i} & " + " & ".join(str(v) for v in row) + r" \\"
            for i, row in enumerate(cm)
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "classification_confusion_test",
        table_float(
            "Selected HGB test confusion matrix. Codes: 0 Good/Moderate, 1 Unhealthy for Sensitive, 2 Unhealthy, 3 Very Unhealthy, 4 Hazardous (301--400), 5 Hazardous (401--500).",
            "tab:clf-confusion",
            body,
            size=r"\scriptsize",
        ),
    )

    # ------------------------------------------------------- regression
    reg_names = {
        "persistence": "Persistence",
        "ridge": "Ridge",
        "hist_gb": "HGB (selected)",
    }
    reg_rows = []
    for key in ("persistence", "ridge", "hist_gb"):
        for split_name in ("validation", "test"):
            m = reg[f"{key}_{split_name}"]
            reg_rows.append(
                {
                    "Model": reg_names[key],
                    "Split": split_name.title(),
                    "RMSE": m["rmse"],
                    "MAE": m["mae"],
                    "R2": m["r2"],
                    "Skill vs persistence": m.get("rmse_skill_vs_persistence"),
                    "n": m["n_samples"],
                    "Dropped": m.get("n_dropped"),
                }
            )
    write_csv(
        "regression_metrics",
        [
            {
                **{k: v for k, v in r.items() if k in ("Model", "Split", "n", "Dropped")},
                **{k: f(r[k]) for k in ("RMSE", "MAE", "R2", "Skill vs persistence")},
            }
            for r in reg_rows
        ],
        ["Model", "Split", "RMSE", "MAE", "R2", "Skill vs persistence", "n", "Dropped"],
    )
    body = (
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Model & Split & RMSE & MAE & $R^2$ & Skill & $n$ \\\\\n\\midrule\n"
    )
    for r in reg_rows:
        name = esc(r["Model"])
        if r["Model"].startswith("HGB"):
            name = "\\textbf{HGB}"
        body += (
            f"{name} & {r['Split']} & {f(r['RMSE'])} & {f(r['MAE'])} & {f(r['R2'])} & "
            f"{f(r['Skill vs persistence'])} & {r['n']} \\\\\n"
        )
    body += "\\bottomrule\n\\end{tabular}\n"
    write_text(
        "regression_metrics",
        table_float(
            "Regression results on the chronological validation and test splits. Skill is $1-\\mathrm{RMSE}_{model}/\\mathrm{RMSE}_{persistence}$.",
            "tab:reg-metrics",
            body,
        ),
    )

    # ------------------------------------------------------- hyperparameters
    hp_rows = []
    for item in clf_hp["tuning"]:
        hp_rows.append(
            {
                "Task": "Classification",
                "Model": item["model"],
                "Selected parameters": json.dumps(item["selected_params"], sort_keys=True),
                "Validation metric": f"macro F1 = {item['validation_f1_macro']:.4f}",
                "Status": "selected" if item.get("selected") else item.get("tuning_status", "not selected"),
            }
        )
    for key, label in (("ridge", "Ridge"), ("hist_gb", "HGB")):
        item = reg_hp[key]
        hp_rows.append(
            {
                "Task": "Regression",
                "Model": label,
                "Selected parameters": json.dumps(item["selected_params"], sort_keys=True),
                "Validation metric": f"RMSE = {item['held_out_val_rmse']:.4f}",
                "Status": "selected" if key == "hist_gb" else "not selected",
            }
        )
    write_csv("hyperparameters", hp_rows, ["Task", "Model", "Selected parameters", "Validation metric", "Status"])
    body = (
        "\\begin{tabular}{llL{5.0cm}L{3.2cm}l}\n\\toprule\n"
        "Task & Model & Selected parameters & Validation metric & Status \\\\\n\\midrule\n"
    )
    for r in hp_rows:
        body += (
            f"{esc(r['Task'])} & {esc(r['Model'])} & \\texttt{{{esc(r['Selected parameters'])}}} & "
            f"{esc(r['Validation metric'])} & {esc(r['Status'])} \\\\\n"
        )
    body += "\\bottomrule\n\\end{tabular}\n"
    write_text(
        "hyperparameters",
        table_float(
            "Selected hyperparameters. Classification used five-fold time-series CV with a seven-day gap and selected on validation macro F1; regression selected on validation RMSE.",
            "tab:hyperparameters",
            body,
            size=r"\scriptsize",
        ),
    )

    # ------------------------------------------------------------- bias table
    df = pd.DataFrame(bias["results"])
    test_df = df[df["split"] == "test"].copy()
    order = {
        "AQI Category": 0,
        "Season": 1,
        "Year": 2,
        "Weekday_Weekend": 3,
        "Severity_Band": 4,
        "Islamic_Event": 5,
        "Public_Holiday": 6,
    }
    group_order = {
        "AQI Category": CLASS_LABELS,
        "Season": ["Winter", "Spring", "Summer", "Autumn"],
        "Year": ["2023", "2024", "2025"],
        "Weekday_Weekend": ["Weekday", "Weekend"],
        "Severity_Band": ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500", "500+"],
        "Islamic_Event": ["Normal", "Ramadan", "Eid"],
        "Public_Holiday": ["Normal", "Holiday"],
    }
    test_df["_order"] = test_df["dimension"].map(order)
    test_df["_group_order"] = test_df.apply(
        lambda r: group_order.get(r["dimension"], []).index(r["group"])
        if r["group"] in group_order.get(r["dimension"], [])
        else 999,
        axis=1,
    )
    test_df = test_df.sort_values(["_order", "_group_order", "group"])
    bias_rows = []
    for _, r in test_df.iterrows():
        bias_rows.append(
            {
                "Dimension": r["dimension"],
                "Group": r["group"],
                "n": int(r["n_samples"]),
                "RMSE": r["reg_rmse"],
                "MAE": r["reg_mae"],
                "Signed error": r["mean_signed_error"],
                "Accuracy": r["cls_accuracy"],
                "Macro F1": r["cls_macro_f1"],
                "Small support": "yes" if isinstance(r.get("warning"), str) else "no",
            }
        )
    write_csv(
        "bias_test",
        [
            {
                **{k: v for k, v in r.items() if k in ("Dimension", "Group", "n", "Small support")},
                **{k: f(r[k]) for k in ("RMSE", "MAE", "Signed error", "Accuracy", "Macro F1")},
            }
            for r in bias_rows
        ],
        ["Dimension", "Group", "n", "RMSE", "MAE", "Signed error", "Accuracy", "Macro F1", "Small support"],
    )
    body = (
        "{\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{longtable}{@{}llrrrrrr@{}}\n"
        "\\caption{Test-set subgroup error and classification performance. Positive signed error means the true value is above the prediction (under-prediction). Small-support groups are marked with $^{*}$.}\\label{tab:bias-test}\\\\\n"
        "\\toprule\n"
        "Dimension & Group & $n$ & RMSE & MAE & Signed err. & Acc. & Macro F1 \\\\\n\\midrule\n"
        "\\endfirsthead\n"
        "\\toprule\n"
        "Dimension & Group & $n$ & RMSE & MAE & Signed err. & Acc. & Macro F1 \\\\\n\\midrule\n"
        "\\endhead\n"
    )
    for r in bias_rows:
        star = "$^{*}$" if r["Small support"] == "yes" else ""
        body += (
            f"{esc(r['Dimension'])} & {esc(r['Group'])}{star} & {r['n']} & {f(r['RMSE'])} & "
            f"{f(r['MAE'])} & {f(r['Signed error'])} & {f(r['Accuracy'])} & {f(r['Macro F1'])} \\\\\n"
        )
    body += "\\bottomrule\n\\end{longtable}\n}\n"
    write_text("bias_test", body)

    # --------------------------------------------------- bias missing counts
    missing = bias.get("missing_counts", {})
    miss_rows = []
    for dim in sorted(missing.get("validation", {})):
        miss_rows.append(
            {
                "Dimension": dim,
                "Validation missing": missing.get("validation", {}).get(dim, 0),
                "Test missing": missing.get("test", {}).get(dim, 0),
            }
        )
    write_csv("bias_missing_counts", miss_rows, ["Dimension", "Validation missing", "Test missing"])
    body = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Dimension & Validation missing & Test missing \\\\\n\\midrule\n"
        + "\n".join(
            f"{esc(r['Dimension'])} & {r['Validation missing']} & {r['Test missing']} \\\\"
            for r in miss_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "bias_missing_counts",
        table_float(
            "Missing subgroup keys after joining regression and classification predictions.",
            "tab:bias-missing",
            body,
            size=r"\small",
        ),
    )

    # ------------------------------------------------------ model selection
    sel_rows = [
        {
            "Task": "Classification",
            "Selected model": "HistGradientBoosting",
            "Selection metric": "validation macro F1",
            "Validation": f(clf["selected_model_val_f1_macro"]),
            "Test": f(clf["selected_model_metrics"]["test"]["f1_macro"]),
            "Persistence test": f(clf["baselines"]["persistence_test"]["f1_macro"]),
            "Verdict": "did not beat persistence on test",
        },
        {
            "Task": "Regression",
            "Selected model": "HGB",
            "Selection metric": "validation RMSE",
            "Validation": f(reg["selected_val_rmse"]),
            "Test": f(reg["hist_gb_test"]["rmse"]),
            "Persistence test": f(reg["persistence_test"]["rmse"]),
            "Verdict": "did not beat persistence on test; Ridge did",
        },
    ]
    write_csv(
        "model_selection",
        sel_rows,
        ["Task", "Selected model", "Selection metric", "Validation", "Test", "Persistence test", "Verdict"],
    )
    body = (
        "\\begin{tabular}{llllll}\n\\toprule\n"
        "Task & Selected & Selection metric & Val. & Test & Persistence test \\\\\n\\midrule\n"
        + "\n".join(
            f"{esc(r['Task'])} & {esc(r['Selected model'])} & {esc(r['Selection metric'])} & "
            f"{esc(r['Validation'])} & {esc(r['Test'])} & {esc(r['Persistence test'])} \\\\"
            for r in sel_rows
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_text(
        "model_selection",
        table_float(
            "Model selection outcomes and the test comparison with persistence.",
            "tab:model-selection",
            body,
            size=r"\small",
        ),
    )

    print("Wrote tables to", TABLES)
    for path in sorted(TABLES.glob("*")):
        print(" ", path.name)


if __name__ == "__main__":
    main()
