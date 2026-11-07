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
    - outputs/model_charts/bias_*.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_config, prepare_model_data, resolve_path


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

    # Only keep scorable rows
    reg_df = reg_df[reg_df["scorable"] == True].copy()
    class_df = class_df[class_df["scorable"] == True].copy()

    # Normalize date to string for reliable join
    reg_df["date"] = pd.to_datetime(reg_df["date"]).dt.strftime("%Y-%m-%d")
    class_df["date"] = pd.to_datetime(class_df["date"]).dt.strftime("%Y-%m-%d")

    print(f"Rows from regression: {len(reg_df)}")
    print(f"Rows from classification: {len(class_df)}")

    merged = pd.merge(
        reg_df, class_df, on=["date", "split"],
        suffixes=("_reg", "_class"), how="inner",
    )
    print(f"Rows surviving the join: {len(merged)}")
    return merged


def get_subgroup_metrics(y_true, y_pred, y_true_cls, y_pred_cls, class_labels):
    """Calculates regression and classification metrics for a subgroup."""
    n = len(y_true)
    if n == 0:
        return {"n_samples": 0, "warning": "Empty group"}

    res = {"n_samples": n}
    res["reg_rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    res["reg_mae"] = float(np.mean(np.abs(y_true - y_pred)))
    # Sign convention: Actual - Predicted. Positive = under-prediction
    res["mean_signed_error"] = float(np.mean(y_true - y_pred))

    # Classification metrics
    res["cls_accuracy"] = float(np.mean(y_true_cls == y_pred_cls))

    # Macro F1 over classes present in this subgroup
    present_classes = sorted(
        set(y_true_cls.tolist()) | set(y_pred_cls.tolist())
    )
    if len(present_classes) > 0:
        from sklearn.metrics import f1_score
        res["cls_macro_f1"] = float(
            f1_score(
                y_true_cls, y_pred_cls, labels=present_classes,
                average="macro", zero_division=0,
            )
        )
    else:
        res["cls_macro_f1"] = None

    # Per-class error counts
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
    # Fix 8: Only use prepare_model_data to get data.split
    data = prepare_model_data(config, "regression")

    print("Loading predictions...")
    merged = load_predictions(config)

    # Fix 8: Pull calendar features from data.split
    val_df = data.split.validation
    test_df = data.split.test
    all_dates = pd.concat([val_df, test_df], ignore_index=True)
    all_dates["date"] = pd.to_datetime(
        all_dates[config["data"]["date_column"]]
    ).dt.strftime("%Y-%m-%d")

    # Keep only the columns we need to avoid collision
    calendar_cols = ["date", "year", "month"]
    for col in ["is_ramadan", "is_eid", "is_public_holiday", "is_weekend"]:
        if col in all_dates.columns:
            calendar_cols.append(col)

    merged = pd.merge(
        merged, all_dates[calendar_cols].drop_duplicates("date"),
        on="date", how="left",
    )

    # Fix 10: Severity bands using PM2.5 concentration ranges
    bins = [0, 50, 100, 150, 200, 300, 500, np.inf]
    labels = ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500", "500+"]
    merged["severity_band"] = pd.cut(
        merged["y_true"], bins=bins, labels=labels, include_lowest=True
    )

    # Fix 8: Map integer codes to string labels
    labels_list = config["data"]["classification_labels"]

    # Build subgroups
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

    if "is_weekend" in merged.columns:
        merged["weekday_weekend"] = merged["is_weekend"].map(
            {True: "Weekend", False: "Weekday", 1: "Weekend", 0: "Weekday"}
        )
    else:
        merged["weekday_weekend"] = pd.to_datetime(merged["date"]).dt.dayofweek.map(
            lambda x: "Weekend" if x >= 5 else "Weekday"
        )

    if "is_ramadan" in merged.columns and "is_eid" in merged.columns:
        merged["islamic_event"] = np.where(
            merged["is_ramadan"].fillna(0).astype(int) == 1, "Ramadan",
            np.where(
                merged["is_eid"].fillna(0).astype(int) == 1, "Eid", "Normal"
            ),
        )
    else:
        merged["islamic_event"] = "Unknown"

    if "is_public_holiday" in merged.columns:
        merged["holiday"] = merged["is_public_holiday"].map(
            {True: "Holiday", False: "Normal", 1: "Holiday", 0: "Normal"}
        ).fillna("Unknown")
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

        # Fix 13: Chart per dimension per split (season example)
        season_rows = [
            r for r in all_results
            if r["split"] == split and r["dimension"] == "Season"
        ]
        if season_rows:
            seasons = [r["group"] for r in season_rows]
            mse_vals = [r["mean_signed_error"] for r in season_rows]
            plt.figure(figsize=(8, 5))
            plt.bar(
                seasons, mse_vals,
                color=["red" if v > 0 else "blue" for v in mse_vals],
            )
            plt.axhline(0, color="black", linestyle="--")
            plt.ylabel("Mean Signed Error (Actual - Predicted)")
            plt.title(f"Over/Under-prediction by Season ({split})")
            plt.tight_layout()
            plt.savefig(
                resolve_path(config["paths"]["charts"])
                / f"bias_season_error_{split}.png"
            )
            plt.close()

    # Save outputs
    results_dir = resolve_path(config["paths"]["results"])

    df_results = pd.DataFrame(all_results)
    # Flatten per_class_errors for CSV
    df_results["per_class_errors"] = df_results["per_class_errors"].apply(
        lambda d: json.dumps(d) if isinstance(d, dict) else d
    )
    df_results.to_csv(results_dir / "bias_subgroup_metrics.csv", index=False)

    with open(results_dir / "bias_subgroup_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4, default=str)

    # Fix 13: Interpretation text
    interpretation = """Bias Analysis Interpretation
================================
- Single monitoring station. No demographic attributes are available.
- This is an environmental and temporal bias check, not demographic.
- Class mix shifts across periods. Do not make causal claims.
- Grouped by: TRUE next-day AQI category (y_true_code), NOT today's aqi_category.
  This answers "where does the model fail by true severity?"
- Severity Bands: Grouping by actual PM2.5 conditions on the outcome.
  High bands will look under-predicted and low bands over-predicted by construction.
- Sign convention: Mean Signed Error = Actual - Predicted. Positive = under-prediction.
- Macro F1 averages over the classes PRESENT in that subgroup.
- Small support warning: n < 30 flagged in the 'warning' column.
- Note: Good/Moderate and Hazardous (401-500) are noisy due to small support.
"""
    with open(results_dir / "bias_interpretation.txt", "w") as f:
        f.write(interpretation)

    print(f"\nSaved bias analysis to {results_dir}")


if __name__ == "__main__":
    main()