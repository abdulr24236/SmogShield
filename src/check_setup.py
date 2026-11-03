"""Smoke-test the shared data contract.

Run from the repository root after restoring data/processed/features_daily.csv:

    python -m src.check_setup
"""

from __future__ import annotations

from src.data import load_config, prepare_model_data


def main() -> None:
    config = load_config()
    for task in ("classification", "regression"):
        data = prepare_model_data(config, task)
        summary = data.summary()
        print(f"\n{task.upper()}")
        print(f"  target:   {summary['target']}")
        print(f"  features: {summary['n_features']}")
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
    print("\nShared setup OK.")


if __name__ == "__main__":
    main()
