"""
Bias and Error Analysis for Deliverable 2.

Run instructions:
    1. python -m src.regression
    2. Copy classification_predictions.csv from the shared Drive
       into outputs/model_results/
    3. python -m src.bias_analysis

Outputs:
    - outputs/model_results/bias_subgroup_metrics.json
    - outputs/model_results/bias_subgroup_metrics.csv
    - outputs/model_results/bias_interpretation.txt
    - outputs/model_results/bias_class_mix.csv
    - outputs/model_charts/bias_*.png
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.data import load_config, prepare_model_data, resolve_path


EXPECTED_DIMENSIONS = [
    "AQI Category",
    "Season",
    "Year",
    "Weekday_Weekend",
    "Severity_Band",
    "Islamic_Event",
    "Public_Holiday",
]


def load_predictions(config):
    """Loads and joins regression and classification predictions."""
    results_dir = resolve_path(config["paths"]["results"])
    reg_path = results_dir / "regression_predictions.csv"
    class_path = results_dir / "classification_predictions.csv"

    if not reg_path.exists():
        sys.exit(f"ERROR: {reg_path} not found. Run src.regression first.")
    if not class_path.exists():
        sys.exit(
            f"ERROR: {class_path} not found. Download it from the shared "
            f"Drive and place it in {results_dir}."
        )

    reg_df = pd.read_csv(reg_path)
    class_df = pd.read_csv(class_path)

    reg_df = reg_df[reg_df["scorable"] == True].copy()
    class_df = class_df[class_df["scorable"] == True].copy()

    reg_df["date"] = pd.to_datetime(reg_df["date"]).dt.strftime("%Y-%m-%d")
    class_df["date"] = pd.to_datetime(class_df["date"]).dt.strftime("%Y-%m-%d")

    print(f"Rows from regression: {len(reg_df)}")
    print(f"Rows from classification: {len(class_df)}")

    merged = pd.merge(
        reg_df, class_df, on=["date", "split"],
        suffixes=("_reg", "_class"), how="inner",
    )
    print(f"Rows surviving the join: {len(merged)}")

    return {
        "merged": merged,
        "reg_rows": int(len(reg_df)),
        "class_rows": int(len(class_df)),
        "join_rows": int(len(merged)),
    }


def get_subgroup_metrics(y_true, y_pred, y_true_cls, y_pred_cls, class_labels):
    """Calculates regression and classification metrics for a subgroup."""
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0, "warning": "Empty group"}

    res = {"n_samples": n}
    res["reg_rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    res["reg_mae"] = float(np.mean(np.abs(y_true - y_pred)))
    res["mean_signed_error"] = float(np.mean(y_true - y_pred))
    res["cls_accuracy"] = float(np.mean(y_true_cls == y_pred_cls))

    present_classes = sorted(
        set(y_true_cls.tolist()) | set(y_pred_cls.tolist())
    )
    if len(present_classes) > 0:
        res["cls_macro_f1"] = float(
            f1_score(
                y_true_cls, y_pred_cls, labels=present_classes,
                average="macro", zero_division=0,
            )
        )
    else:
        res["cls_macro_f1"] = None

    per_class_errors = {}
    for i, label in enumerate(class_labels):
        mask = y_true_cls == i
        if mask.sum() > 0:
            per_class_errors[label] = int((y_pred_cls[mask] != i).sum())
    res["per_class_errors"] = per_class_errors

    if n < 30:
        res["warning"] = f"Small support (n={n} < 30)"
    return res


def main():
    config = load_config()
    data = prepare_model_data(config, "regression")

    print("Loading predictions...")
    pred_info = load_predictions(config)
    merged = pred_info["merged"]

    val_df = data.split.validation
    test_df = data.split.test
    all_dates = pd.concat([val_df, test_df], ignore_index=True)
    all_dates["date"] = pd.to_datetime(
        all_dates[config["data"]["date_column"]]
    ).dt.strftime("%Y-%m-%d")

    calendar_cols = ["date", "year", "month"]
    for col in ["is_ramadan", "is_eid", "is_public_holiday", "is_weekend"]:
        if col in all_dates.columns:
            calendar_cols.append(col)

    merged = pd.merge(
        merged, all_dates[calendar_cols].drop_duplicates("date"),
        on="date", how="left",
    )

    bins = [0, 50, 100, 150, 200, 300, 500, np.inf]
    labels = ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500", "500+"]
    merged["severity_band"] = pd.cut(
        merged["y_true"], bins=bins, labels=labels, include_lowest=True
    )

    labels_list = config["data"]["classification_labels"]

    merged["aqi_category"] = merged["y_true_code"].apply(
        lambda x: labels_list[int(x)] if pd.notna(x) else "Missing"
    )

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    merged["season"] = merged["month"].map(season_map)

    # Weekday/Weekend — explicit numeric coercion, do not use fillna(0)
    if "is_weekend" in merged.columns:
        flag = pd.to_numeric(merged["is_weekend"], errors="coerce")
        merged["weekday_weekend"] = flag.map({1: "Weekend", 0: "Weekday"})
    else:
        merged["weekday_weekend"] = pd.to_datetime(
            merged["date"]
        ).dt.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")

    # Islamic events — explicit coercion, keep genuine missing as Unknown
    if "is_ramadan" in merged.columns and "is_eid" in merged.columns:
        ramadan = pd.to_numeric(merged["is_ramadan"], errors="coerce")
        eid = pd.to_numeric(merged["is_eid"], errors="coerce")
        both_missing = ramadan.isna() | eid.isna()
        merged["islamic_event"] = np.select(
            [ramadan == 1, eid == 1, both_missing],
            ["Ramadan", "Eid", "Unknown"],
            default="Normal",
        )
    else:
        merged["islamic_event"] = "Unknown"

    # Public holiday — explicit coercion
    if "is_public_holiday" in merged.columns:
        holiday = pd.to_numeric(
            merged["is_public_holiday"], errors="coerce"
        )
        merged["holiday"] = holiday.map({1: "Holiday", 0: "Normal"})
    else:
        merged["holiday"] = "Unknown"

    subgroups = {
        "AQI Category": "aqi_category",
        "Season": "season",
        "Year": "year",
        "Weekday_Weekend": "weekday_weekend",
        "Severity_Band": "severity_band",
        "Islamic_Event": "islamic_event",
        "Public_Holiday": "holiday",
    }

    all_results = []

    for split in ["validation", "test"]:
        split_df = merged[merged["split"] == split]
        if len(split_df) == 0:
            continue

        print(f"\nAnalyzing {split} split ({len(split_df)} rows)...")

        # Guard: every expected dimension must be present and non-degenerate
        for dim_name in EXPECTED_DIMENSIONS:
            if dim_name not in subgroups:
                raise ValueError(
                    f"Expected dimension {dim_name} not in subgroups map"
                )
            col = subgroups[dim_name]
            if col not in split_df.columns:
                raise ValueError(
                    f"Dimension {dim_name} (column {col}) missing from split {split}"
                )
            missing_count = int(split_df[col].isna().sum())
            print(f"  [{split}] {dim_name}: {missing_count} missing values")
            unique_vals = split_df[col].dropna().unique()
            if len(unique_vals) == 0:
                raise ValueError(
                    f"Dimension {dim_name} in split {split} has no non-null values"
                )
            if all(str(v) == "Unknown" for v in unique_vals):
                raise ValueError(
                    f"Dimension {dim_name} in split {split} has all Unknown values"
                )
            if len(unique_vals) == 1:
                print(
                    f"  WARNING: [{split}] {dim_name} has only one group: "
                    f"{unique_vals[0]}"
                )

        for dim_name, col in subgroups.items():
            for group in split_df[col].dropna().unique():
                group_data = split_df[split_df[col] == group]
                metrics = get_subgroup_metrics(
                    group_data["y_true"].values,
                    group_data["y_pred"].values,
                    group_data["y_true_code"].values.astype(int),
                    group_data["y_pred_code"].values.astype(int),
                    labels_list,
                )
                metrics.update(
                    {"split": split, "dimension": dim_name, "group": str(group)}
                )
                all_results.append(metrics)

    # Charts for all 7 dimensions per split
    charts_dir = resolve_path(config["paths"]["charts"])
    charts_dir.mkdir(parents=True, exist_ok=True)

    for split in ["validation", "test"]:
        for dim_name in EXPECTED_DIMENSIONS:
            dim_rows = [
                r for r in all_results
                if r["split"] == split and r["dimension"] == dim_name
            ]
            if not dim_rows:
                continue

            groups = [r["group"] for r in dim_rows]
            mse_vals = [r.get("mean_signed_error", 0) for r in dim_rows]
            acc_vals = [r.get("cls_accuracy", 0) or 0 for r in dim_rows]
            ns = [r["n_samples"] for r in dim_rows]

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            bars = axes[0].bar(
                groups, mse_vals,
                color=["red" if v > 0 else "blue" for v in mse_vals],
            )
            axes[0].axhline(0, color="black", linestyle="--")
            axes[0].set_ylabel("Mean Signed Error (Actual - Predicted)")
            axes[0].set_title(
                f"{dim_name} Signed Error ({split})\n"
                f"(positive = under-prediction)"
            )
            axes[0].tick_params(axis="x", rotation=45)
            for i, bar in enumerate(bars):
                label = f"n={ns[i]}"
                if ns[i] < 30:
                    label += " *"
                y = bar.get_height()
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, y, label,
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=8,
                )

            bars2 = axes[1].bar(groups, acc_vals, color="seagreen")
            axes[1].set_ylabel("Classification Accuracy")
            axes[1].set_title(
                f"{dim_name} Classification Accuracy ({split})"
            )
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].set_ylim(0, 1.05)
            for i, bar in enumerate(bars2):
                label = f"n={ns[i]}"
                if ns[i] < 30:
                    label += " *"
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8,
                )

            plt.tight_layout()
            plt.savefig(
                charts_dir / f"bias_{dim_name.replace(' ', '_')}_{split}.png"
            )
            plt.close()

    # Model names
    def _first_non_null(df, col):
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                return str(vals.iloc[0])
        return "unknown"

    model_reg = _first_non_null(merged, "model_reg")
    model_class = _first_non_null(merged, "model_class")

    # Class-mix table (true next-day class counts by split and year)
    class_mix = (
        merged.groupby(["split", "year", "y_true_code"])
        .size()
        .reset_index(name="count")
    )
    class_mix["aqi_label"] = class_mix["y_true_code"].apply(
        lambda x: labels_list[int(x)] if pd.notna(x) else "Missing"
    )

    results_dir = resolve_path(config["paths"]["results"])

    df_results = pd.DataFrame(all_results)
    df_results["per_class_errors"] = df_results["per_class_errors"].apply(
        lambda d: json.dumps(d) if isinstance(d, dict) else d
    )
    df_results.to_csv(results_dir / "bias_subgroup_metrics.csv", index=False)

    class_mix.to_csv(results_dir / "bias_class_mix.csv", index=False)

    json_output = {
        "model_reg": model_reg,
        "model_class": model_class,
        "join_counts": {
            "regression_rows": pred_info["reg_rows"],
            "classification_rows": pred_info["class_rows"],
            "join_rows": pred_info["join_rows"],
        },
        "class_mix_note": (
            "Computed from joined rows where scorable=True "
            "(only scorable prediction rows)."
        ),
        "results": all_results,
    }
    with open(results_dir / "bias_subgroup_metrics.json", "w") as f:
        json.dump(json_output, f, indent=4, default=str)

    interpretation = f"""Bias Analysis Interpretation
================================
- Single monitoring station. No demographic attributes are available.
- This is an environmental and temporal bias check, not demographic.
- Class mix shifts across periods. Do not make causal claims.
- Grouped by: TRUE next-day AQI category (y_true_code), NOT today's aqi_category.
  The AQI Category group therefore also conditions on the outcome, like the
  severity bands do.
- Severity Bands: Grouping by actual PM2.5 conditions on the outcome.
  High bands will look under-predicted and low bands over-predicted by construction.
- Severity bands are RAW PM2.5 CONCENTRATION RANGES (0-50, 51-100, ... ug/m3),
  NOT AQI category breakpoints, so they will not line up exactly with the
  AQI Category grouping above.
- Sign convention: Mean Signed Error = Actual - Predicted. Positive = under-prediction.
- Macro F1 averages over the classes PRESENT in that subgroup.
- Small support warning: n < 30 flagged in the 'warning' column and marked '*' on charts.
- Note: Good/Moderate and Hazardous (401-500) are noisy due to small support.

Model analysed:
- Regression model (model_reg): {model_reg}
- Classification model (model_class): {model_class}

Join counts:
- Regression prediction rows: {pred_info["reg_rows"]}
- Classification prediction rows: {pred_info["class_rows"]}
- Rows surviving the join: {pred_info["join_rows"]}

Class mix table:
- Saved to bias_class_mix.csv (true next-day class counts by split and year).
- Computed from the joined prediction rows where scorable=True (only scorable
  rows are used; non-scorable rows are excluded before the join).
"""
    with open(results_dir / "bias_interpretation.txt", "w") as f:
        f.write(interpretation)

    print(f"\nSaved bias analysis to {results_dir}")
    print(f"  model_reg: {model_reg}")
    print(f"  model_class: {model_class}")


if __name__ == "__main__":
    main()