"""
Regression Modeling Pipeline for Deliverable 2.

Run instructions:
    python -m src.regression

Outputs:
    - outputs/model_results/regression_metrics.json
    - outputs/model_results/regression_hyperparameters.json
    - outputs/model_results/regression_predictions.csv
    - outputs/model_charts/regression_*.png
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error

from src.data import load_config, prepare_model_data, resolve_path
import src.evaluate as evaluate


def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def tune_models(data, config):
    """Tunes Ridge and HistGradientBoosting using TimeSeriesSplit(gap=7)."""
    tscv = TimeSeriesSplit(n_splits=5, gap=7)
    seed = config.get("seed", 42)

    param_grids = {
        "ridge": {
            "model": Ridge(random_state=seed),
            "grid": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        },
        "hist_gb": {
            "model": HistGradientBoostingRegressor(random_state=seed),
            "grid": {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
        },
    }

    best_models = {}
    hyperparameters = {}
    scoring_func = make_scorer(rmse_scorer, greater_is_better=False)

    print("Tuning models...")
    for name, info in param_grids.items():
        print(f"  Tuning {name}...")
        pipeline = data.make_model_pipeline(info["model"])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=info["grid"],
            cv=tscv,
            scoring=scoring_func,
            n_jobs=-1,
            error_score="raise",  # Fix 14
        )

        grid_search.fit(data.X_train_raw, data.y_train)

        best_models[name] = grid_search.best_estimator_
        hyperparameters[name] = {
            "search_space": {k: list(v) for k, v in info["grid"].items()},
            "selected_params": {
                k.replace("model__", ""): v
                for k, v in grid_search.best_params_.items()
            },
            "cv_folds": 5,
            "cv_gap": 7,
            "scoring": "neg_root_mean_squared_error",
            "best_cv_rmse": float(-grid_search.best_score_),
        }
        print(f"    Best params: {hyperparameters[name]['selected_params']}")

    return best_models, hyperparameters


def generate_learning_curve(data, config, charts_dir):
    """Fix 15: Learning curve for HGB max_iter."""
    print("Generating HGB learning curve...")
    seed = config.get("seed", 42)
    max_iters = [10, 50, 100, 200, 300, 500]
    val_rmses = []

    mask = data.scorable_rows["validation"].to_numpy()
    y_true = data.y_validation[mask].to_numpy()

    for it in max_iters:
        model = data.make_model_pipeline(
            HistGradientBoostingRegressor(max_iter=it, random_state=seed)
        )
        model.fit(data.X_train_raw, data.y_train)
        preds = model.predict(data.X_validation_raw)[mask]
        val_rmses.append(float(np.sqrt(mean_squared_error(y_true, preds))))

    plt.figure(figsize=(8, 5))
    plt.plot(max_iters, val_rmses, marker="o")
    plt.xlabel("HGB max_iter")
    plt.ylabel("Validation RMSE")
    plt.title("HGB Learning Curve (Validation RMSE vs max_iter)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "regression_hgb_learning_curve.png")
    plt.close()


def evaluate_and_plot(data, best_models, config):
    results = {}
    charts_dir = resolve_path(config["paths"]["charts"])
    os.makedirs(charts_dir, exist_ok=True)

    # Fix 6: Fit each model once
    print("Fitting final models...")
    fitted_models = {}
    for name, model in best_models.items():
        model.fit(data.X_train_raw, data.y_train)
        fitted_models[name] = model

    # Cache predictions per (model, split)
    preds_cache = {}

    for split in ["validation", "test"]:
        print(f"\n--- Evaluating on {split} ---")
        mask = data.scorable_rows[split]
        mask_np = mask.to_numpy()

        if split == "validation":
            y_true_full = data.y_validation
            X_raw = data.X_validation_raw
            dates = data.split.validation[config["data"]["date_column"]]
        else:
            y_true_full = data.y_test
            X_raw = data.X_test_raw
            dates = data.split.test[config["data"]["date_column"]]

        y_true = y_true_full[mask_np].to_numpy()
        n_dropped = int(len(y_true_full) - len(y_true))

        # Persistence baseline
        pers_preds_full = evaluate.persistence_regression_predictions(
            X_raw["pm25"]
        )
        pers_metrics = evaluate.regression_metrics(
            y_true=y_true,
            y_pred=pers_preds_full[mask_np],
            allow_missing=False,
        )
        pers_metrics["n_dropped"] = n_dropped
        results[f"persistence_{split}"] = pers_metrics
        preds_cache[("persistence", split)] = (
            dates, y_true_full, pers_preds_full, mask_np
        )

        # Trained models
        for name, model in fitted_models.items():
            preds_full = model.predict(X_raw)
            metrics = evaluate.regression_metrics(
                y_true=y_true,
                y_pred=preds_full[mask_np],
                allow_missing=False,
                persistence_rmse=pers_metrics["rmse"],
            )
            metrics["n_dropped"] = n_dropped
            results[f"{name}_{split}"] = metrics
            preds_cache[(name, split)] = (
                dates, y_true_full, preds_full, mask_np
            )

    # Fix 2: Select model by validation RMSE
    val_rmse_dict = {
        m: results[f"{m}_validation"]["rmse"] for m in best_models.keys()
    }
    selected_model = min(val_rmse_dict, key=val_rmse_dict.get)
    results["selected_model"] = selected_model
    results["selected_val_rmse"] = val_rmse_dict[selected_model]
    results["n_dropped"] = {
        "validation": results["persistence_validation"]["n_dropped"],
        "test": results["persistence_test"]["n_dropped"],
    }

    # Fix 1: Save predictions for ONLY the selected model
    pred_rows = []
    for split in ["validation", "test"]:
        dates, y_true_full, preds_full, mask_np = preds_cache[
            (selected_model, split)
        ]
        pred_rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d").values,
                    "split": split,
                    "model": selected_model,
                    "y_true": y_true_full.values,
                    "y_pred": preds_full,
                    "scorable": mask_np,
                }
            )
        )

    df_preds = pd.concat(pred_rows, ignore_index=True)
    preds_path = (
        resolve_path(config["paths"]["results"]) / "regression_predictions.csv"
    )
    df_preds.to_csv(preds_path, index=False)

    expected_rows = (
        data.summary()["validation_rows"] + data.summary()["test_rows"]
    )
    assert len(df_preds) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(df_preds)}"
    )
    print(f"Saved {len(df_preds)} predictions to {preds_path}")

    # Save metrics
    metrics_path = (
        resolve_path(config["paths"]["results"]) / "regression_metrics.json"
    )
    evaluate.save_metrics(results, metrics_path)
    print(f"Saved metrics to {metrics_path}")

    # --- CHARTS ---
    print("Generating charts...")
    test_mask = data.scorable_rows["test"].to_numpy()
    y_test_true = data.y_test[test_mask].to_numpy()
    dates_test = data.split.test.loc[
        test_mask, config["data"]["date_column"]
    ]

    for name in [selected_model, "ridge", "persistence"]:
        if name not in fitted_models and name != "persistence":
            continue
        dates, y_true_full, preds_full, mask_np = preds_cache[(name, "test")]
        preds = preds_full[test_mask]
        residuals = y_test_true - preds

        # Chart 1: Pred vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_true, preds, alpha=0.5)
        lims = [
            min(y_test_true.min(), preds.min()),
            max(y_test_true.max(), preds.max()),
        ]
        plt.plot(lims, lims, "r--", label="Perfect")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        plt.title(f"Pred vs Actual ({name} - Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / f"regression_{name}_pred_vs_actual.png")
        plt.close()

        # Chart 2: Residuals vs Pred
        plt.figure(figsize=(8, 6))
        plt.scatter(preds, residuals, alpha=0.5)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs Pred ({name} - Test)")
        plt.tight_layout()
        plt.savefig(charts_dir / f"regression_{name}_residuals_vs_pred.png")
        plt.close()

        # Chart 3: Residuals over Time
        plt.figure(figsize=(10, 4))
        plt.plot(dates_test, residuals, marker=".", linestyle="-", alpha=0.7)
        plt.axhline(0, color="r", linestyle="--")
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.title(f"Residuals over Time ({name} - Test)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            charts_dir / f"regression_{name}_residuals_over_time.png"
        )
        plt.close()

    # Fix 4: Error by Severity (proper concentration bands)
    _, _, sel_preds_full, _ = preds_cache[(selected_model, "test")]
    df_sev = pd.DataFrame(
        {
            "y_true": y_test_true,
            "y_pred": sel_preds_full[test_mask],
        }
    )
    df_sev["error"] = np.abs(df_sev["y_true"] - df_sev["y_pred"])

    bins = [0, 50, 100, 150, 200, 300, 500, np.inf]
    labels = ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500", "500+"]
    df_sev["severity"] = pd.cut(
        df_sev["y_true"], bins=bins, labels=labels, include_lowest=True
    )
    sev_counts = df_sev["severity"].value_counts().reindex(labels).fillna(0).astype(int)
    sev_errors = df_sev.groupby("severity", observed=False)["error"].mean().dropna()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        sev_errors.index.astype(str), sev_errors.values, color="skyblue"
    )
    plt.xlabel("Actual PM2.5 Concentration Range (ug/m3)")
    plt.ylabel("MAE")
    plt.title(f"Error by PM2.5 Range ({selected_model} - Test)")
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={sev_counts.iloc[i]}",
            ha="center",
            va="bottom",
        )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir / f"regression_{selected_model}_error_by_severity.png")
    plt.close()

    # Chart 5: Val/Test comparison
    plt.figure(figsize=(10, 5))
    x_labels = list(best_models.keys()) + ["persistence"]
    val_rmse = [results[f"{m}_validation"]["rmse"] for m in x_labels]
    test_rmse = [results[f"{m}_test"]["rmse"] for m in x_labels]
    x = np.arange(len(x_labels))
    width = 0.35

    plt.bar(
        x - width / 2, val_rmse, width, label="Validation RMSE",
        color="orange", alpha=0.7
    )
    plt.bar(
        x + width / 2, test_rmse, width, label="Test RMSE",
        color="teal", alpha=0.7
    )
    plt.xticks(x, x_labels)
    plt.ylabel("RMSE")
    plt.title("Validation vs Test RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "regression_metric_comparison.png")
    plt.close()

    return results


def main():
    start_time = time.time()
    config = load_config()
    data = prepare_model_data(config, "regression")

    best_models, hyperparameters = tune_models(data, config)
    generate_learning_curve(
        data, config, resolve_path(config["paths"]["charts"])
    )

    results = evaluate_and_plot(data, best_models, config)

    # Fix 3: Add held-out val RMSE to hyperparameters
    for name in best_models:
        hyperparameters[name]["held_out_val_rmse"] = results[
            f"{name}_validation"
        ]["rmse"]

    hyp_path = (
        resolve_path(config["paths"]["results"])
        / "regression_hyperparameters.json"
    )
    with open(hyp_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Saved hyperparameters to {hyp_path}")

    runtime = time.time() - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds")
    print("Regression pipeline complete!")


if __name__ == "__main__":
    main()