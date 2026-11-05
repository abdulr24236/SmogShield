import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Prevents GUI backend issues
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.data import load_config, prepare_model_data, resolve_path
import src.evaluate as evaluate


def tune_models(data, config):
    """Tunes Ridge and HistGradientBoosting using TimeSeriesSplit(gap=7)."""
    # The contract states gap=7 to prevent target leakage
    tscv = TimeSeriesSplit(n_splits=5, gap=7)
    
    # Define parameter grids
    param_grids = {
        "ridge": {
            "model": Ridge(random_state=config.get("seed", 42)),
            "grid": {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
        },
        "hist_gb": {
            "model": HistGradientBoostingRegressor(random_state=config.get("seed", 42)),
            "grid": {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5]
            }
        }
    }
    
    best_models = {}
    hyperparameters = {}
    
    print("Tuning models...")
    for name, info in param_grids.items():
        print(f"  Tuning {name}...")
        # Use make_model_pipeline to ensure preprocessor is fit only on training folds
        pipeline = data.make_model_pipeline(info["model"])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=info["grid"],
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        
        # Train on raw data so the pipeline handles preprocessing within CV safely
        grid_search.fit(data.X_train_raw, data.y_train)
        
        best_models[name] = grid_search.best_estimator_
        # Extract only the model hyperparameters for the JSON output
        hyperparameters[name] = {
            k.replace("model__", ""): v for k, v in grid_search.best_params_.items()
        }
        print(f"    Best params: {hyperparameters[name]}")
        
    return best_models, hyperparameters


def evaluate_and_plot(data, best_models, config):
    """Evaluates models on validation and test sets, and generates charts."""
    results = {}
    predictions_dict = {}
    
    charts_dir = resolve_path(config["paths"]["charts"])
    os.makedirs(charts_dir, exist_ok=True)
    
    splits = ["validation", "test"]
    
    for split in splits:
        print(f"\n--- Evaluating on {split} ---")
        mask = data.scorable_rows[split]
        
        if split == "validation":
            y_true_full = data.y_validation
            X_raw = data.X_validation_raw
        else:
            y_true_full = data.y_test
            X_raw = data.X_test_raw
            
        y_true = y_true_full[mask].to_numpy()
        
        # 1. Persistence Baseline
        # Contract: Only persistence may use allow_missing=True if necessary.
        # Since we pre-filter using the scorable mask, persistence will not have NaNs here.
        persistence_preds_full = evaluate.persistence_regression_predictions(X_raw["pm25"])
        persistence_preds_masked = persistence_preds_full[mask.to_numpy()]
        
        pers_metrics = evaluate.regression_metrics(
            y_true=y_true,
            y_pred=persistence_preds_masked,
            allow_missing=False 
        )
        results[f"persistence_{split}"] = pers_metrics
        predictions_dict[f"persistence_{split}"] = persistence_preds_masked
        persistence_rmse = pers_metrics["rmse"]
        
        # 2. Trained Models
        for name, model in best_models.items():
            # Train final model on train only
            model.fit(data.X_train_raw, data.y_train)
            
            # Predict and filter by mask
            preds_full = model.predict(X_raw)
            preds_masked = preds_full[mask.to_numpy()]
            
            metrics = evaluate.regression_metrics(
                y_true=y_true,
                y_pred=preds_masked,
                allow_missing=False,
                persistence_rmse=persistence_rmse
            )
            results[f"{name}_{split}"] = metrics
            predictions_dict[f"{name}_{split}"] = preds_masked

    # Save metrics to JSON
    metrics_path = resolve_path(config["paths"]["results"]) / "regression_metrics.json"
    evaluate.save_metrics(results, metrics_path)
    print(f"\nSaved metrics to {metrics_path}")
    
    # --- GENERATE CHARTS (Section 2.4) ---
    print("Generating charts...")
    test_mask = data.scorable_rows["test"].to_numpy()
    y_test_true = data.y_test[test_mask].to_numpy()
    dates_test = data.split.test.loc[test_mask, config["data"]["date_column"]]
    
    main_model = "hist_gb"
    
    for name in [main_model, "ridge", "persistence"]:
        if name not in best_models and name != "persistence":
            continue
            
        preds = predictions_dict[f"{name}_test"]
        residuals = y_test_true - preds
        
        # Chart 1: Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_true, preds, alpha=0.5)
        lims = [min(y_test_true.min(), preds.min()), max(y_test_true.max(), preds.max())]
        plt.plot(lims, lims, 'r--', label="Perfect Prediction")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        plt.title(f"Predicted vs Actual PM2.5 ({name} - Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts_dir / f"regression_{name}_pred_vs_actual.png")
        plt.close()
        
        # Chart 2: Residuals vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(preds, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Predicted PM2.5")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title(f"Residuals vs Predicted ({name} - Test)")
        plt.tight_layout()
        plt.savefig(charts_dir / f"regression_{name}_residuals_vs_pred.png")
        plt.close()
        
        # Chart 3: Residuals over Time
        plt.figure(figsize=(10, 4))
        plt.plot(dates_test, residuals, marker='.', linestyle='-', alpha=0.7)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Residuals")
        plt.title(f"Residuals over Time ({name} - Test)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(charts_dir / f"regression_{name}_residuals_over_time.png")
        plt.close()

    # Chart 4: Error by pollution severity (Using HistGB)
    df_sev = pd.DataFrame({
        "y_true": y_test_true, 
        "y_pred": predictions_dict[f"{main_model}_test"],
        "error": np.abs(y_test_true - predictions_dict[f"{main_model}_test"])
    })
    
    bins = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good/Mod(0-50)", "Moderate(51-100)", "USG(101-150)", "Unhealthy(151-200)", "Very Unhealthy(201-300)", "Hazardous(300+)"]
    df_sev["severity"] = pd.cut(df_sev["y_true"], bins=bins, labels=labels, right=True)
    sev_errors = df_sev.groupby("severity", observed=False)["error"].mean().dropna()
    
    plt.figure(figsize=(10, 5))
    sev_errors.plot(kind="bar", color="skyblue")
    plt.xlabel("Actual PM2.5 Severity Band")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Error by Pollution Severity ({main_model} - Test)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir / f"regression_{main_model}_error_by_severity.png")
    plt.close()
    
    # Chart 5: Validation and test metric comparison
    plt.figure(figsize=(10, 5))
    x_labels = list(best_models.keys()) + ["persistence"]
    val_rmse = [results[f"{m}_validation"]["rmse"] for m in x_labels]
    test_rmse = [results[f"{m}_test"]["rmse"] for m in x_labels]
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    plt.bar(x - width/2, val_rmse, width, label='Validation RMSE', color='orange', alpha=0.7)
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE', color='teal', alpha=0.7)
    plt.xticks(x, x_labels)
    plt.ylabel("RMSE")
    plt.title("Validation vs Test RMSE Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "regression_metric_comparison.png")
    plt.close()

    print(f"Saved charts to {charts_dir}")


def main():
    config = load_config()
    data = prepare_model_data(config, "regression")
    
    best_models, hyperparameters = tune_models(data, config)
    
    # Save hyperparameters (Section 2.2)
    hyp_path = resolve_path(config["paths"]["results"]) / "regression_hyperparameters.json"
    with open(hyp_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Saved hyperparameters to {hyp_path}")
    
    evaluate_and_plot(data, best_models, config)
    
    print("\nRegression pipeline complete!")

if __name__ == "__main__":
    main()