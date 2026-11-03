"""Smoke-test the shared data contract.

Run from the repository root after restoring data/processed/features_daily.csv:

    python -m src.check_setup
"""

from __future__ import annotations

import pandas as pd

from src.data import load_config, load_features, prepare_model_data


def calendar_alignment_counts(config: dict) -> dict[str, int]:
    """Check that targets and lag-1 use the adjacent calendar day."""
    df = load_features(config)
    date_col = config["data"]["date_column"]
    dates = df[date_col]
    by_date = df.set_index(date_col)

    next_day = dates + pd.Timedelta(days=1)
    previous_day = dates - pd.Timedelta(days=1)
    next_pm25 = next_day.map(by_date["pm25"])
    previous_pm25 = previous_day.map(by_date["pm25"])

    target = df["target_next_pm25"]
    lag1 = df["pm25_lag1"]
    bad_target = int(
        (target.notna() & ~((next_pm25 - target).abs() < 1e-6)).sum()
    )
    bad_lag1 = int(
        (lag1.notna() & ~((previous_pm25 - lag1).abs() < 1e-6)).sum()
    )

    next_category = next_day.map(by_date["aqi_category"])
    target_category = df["target_next_aqi_category"]
    bad_category = int(
        (
            target_category.notna()
            & (next_category.astype(object) != target_category.astype(object))
        ).sum()
    )

    span = int((dates.max() - dates.min()).days + 1)
    return {
        "rows": int(len(df)),
        "span": span,
        "missing_dates": span - int(len(df)),
        "bad_target": bad_target,
        "bad_lag1": bad_lag1,
        "bad_category": bad_category,
    }


def main() -> None:
    config = load_config()
    alignment = calendar_alignment_counts(config)
    print("CALENDAR ALIGNMENT")
    print(
        f"  rows: {alignment['rows']}  "
        f"span: {alignment['span']}  "
        f"missing dates: {alignment['missing_dates']}"
    )
    print(f"  bad next-day PM2.5: {alignment['bad_target']}")
    print(f"  bad lag-1 PM2.5: {alignment['bad_lag1']}")
    print(f"  bad next-day category: {alignment['bad_category']}")
    failures = {
        key: value
        for key, value in alignment.items()
        if key.startswith("bad_") and value
    }
    if failures:
        raise RuntimeError(f"Calendar alignment failed: {failures}")

    for task in ("classification", "regression"):
        data = prepare_model_data(config, task)
        summary = data.summary()
        print(f"\n{task.upper()}")
        print(f"  target:   {summary['target']}")
        print(
            f"  features: {summary['n_features']} transformed "
            f"({summary['n_raw_features']} raw)"
        )
        print(
            "  rows:     "
            f"train={summary['train_rows']}  "
            f"validation={summary['validation_rows']}  "
            f"test={summary['test_rows']}"
        )
        split = summary["split"]
        print(f"  train:    {split['train_range'][0]} -> {split['train_range'][1]}")
        print(
            "  val:      "
            f"{split['validation_range'][0]} -> {split['validation_range'][1]}"
        )
        print(f"  test:     {split['test_range'][0]} -> {split['test_range'][1]}")
        if data.labels is not None:
            print(f"  classes:  {len(data.labels)}")
            if summary.get("label_merge"):
                print(f"  merge:    {summary['label_merge']}")
            for split_name, counts in summary["class_counts"].items():
                rendered = ", ".join(
                    f"{label}={count}" for label, count in counts.items()
                )
                print(f"  {split_name:10s} {rendered}")
            support_warnings = summary.get("class_support_warnings", {})
            for split_name, missing in support_warnings.items():
                print(
                    f"  WARNING: {split_name} has no rows for "
                    f"{', '.join(missing)}"
                )
    print("\nShared setup OK.")


if __name__ == "__main__":
    main()
