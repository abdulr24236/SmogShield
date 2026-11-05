import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
import holidays
import hijridate

from src.data import load_config, prepare_model_data, resolve_path
import src.evaluate as evaluate

def get_subgroup_metrics(y_true, y_pred, mask, group_series):
    """Helper to calculate metrics for a specific subgroup."""
    combined_mask = mask & group_series.notna()
    
    if combined_mask.sum() < 5:
        return {"n_samples": int(combined_mask.sum()), "warning": "Small support (n < 5)"}
        
    yt = y_true[combined_mask]
    yp = y_pred[combined_mask]
    
    try:
        metrics = evaluate.regression_metrics(yt, yp, allow_missing=False)
        metrics["mean_signed_error"] = float(np.mean(yt - yp))
        return metrics
    except Exception as e:
        return {"error": str(e)}

def identify_special_events(df, date_col="date"):
    """Identifies Ramadan, Eid, and Public Holidays."""
    dates = pd.to_datetime(df[date_col])
    years = range(dates.dt.year.min(), dates.dt.year.max() + 1)
    
    # 1. Public Holidays (Assuming Pakistan based on context)
    pk_holidays = holidays.Pakistan(years=years)
    df["is_public_holiday"] = dates.dt.date.isin(pk_holidays)
    
    # 2. Ramadan and Eid
    events = []
    for d in dates:
        try:
            h_date = hijridate.to_hijri(d)
            month = h_date.month
            day = h_date.day
            
            if month == 9:
                events.append("Ramadan")
            elif month == 10 and day in [1, 2, 3]:
                events.append("Eid al-Fitr")
            elif month == 12 and day in [10, 11, 12]:
                events.append("Eid al-Adha")
            else:
                events.append("Normal")
        except Exception:
            events.append("Normal")
            
    df["islamic_event"] = events
    return df

def main():
    config = load_config()
    data = prepare_model_data(config, "regression")
    
    print("Training main model for bias analysis...")
    base_model = HistGradientBoostingRegressor(
        random_state=42, max_iter=200, learning_rate=0.1, max_depth=5
    )
    # We MUST wrap the model in the pipeline to handle the text categories!
    model = data.make_model_pipeline(base_model)
    model.fit(data.X_train_raw, data.y_train)
    
    # We will analyze the Test split
    split = "test"
    mask = data.scorable_rows[split]
    X_raw = data.X_test_raw
    y_true_full = data.y_test
    
    preds_full = model.predict(X_raw)
    
    # Filter to scorable rows
    y_true = y_true_full[mask]
    y_pred = preds_full[mask.to_numpy()]
    
    df = X_raw[mask].copy()
    df["actual_pm25"] = y_true
    df["predicted_pm25"] = y_pred
    
    # FIX: data.py removes the 'date' column from X_raw. We must pull it from the raw split.
    df["date"] = data.split.test.loc[mask, "date"].values
    
    print("Identifying temporal events (Ramadan, Eid, Holidays)...")
    df = identify_special_events(df)
    
    print("Defining subgroups...")
    df["aqi_category"] = df["aqi_category"].astype(str)
    
    df["month"] = pd.to_datetime(df["date"]).dt.month
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter', 
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    df["season"] = df["month"].map(season_map)
    
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    
    bins = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good/Mod", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
    df["severity_band"] = pd.cut(df["actual_pm25"], bins=bins, labels=labels)
    
    subgroups = {
        "AQI Category": "aqi_category",
        "Season": "season",
        "Year": "year",
        "Weekday_Weekend": "is_weekend",
        "Severity_Band": "severity_band",
        "Islamic_Event": "islamic_event",
        "Public_Holiday": "is_public_holiday"
    }
    
    bias_results = {}
    
    for name, col in subgroups.items():
        print(f"Analyzing {name}...")
        bias_results[name] = {}
        for group in df[col].dropna().unique():
            group_data = df[df[col] == group]
            local_mask = pd.Series(True, index=group_data.index)
            bias_results[name][str(group)] = get_subgroup_metrics(
                group_data["actual_pm25"], 
                group_data["predicted_pm25"],
                local_mask,
                group_data[col]
            )
            
    out_dir = resolve_path(config["paths"]["results"])
    out_path = out_dir / "bias_analysis_regression.json"
    evaluate.save_metrics(bias_results, out_path)
    print(f"Saved bias analysis to {out_path}")
    
    plt.figure(figsize=(8, 5))
    seasons = list(bias_results.get("Season", {}).keys())
    mse_vals = [bias_results["Season"][s].get("mean_signed_error", 0) for s in seasons]
    
    plt.bar(seasons, mse_vals, color=['red' if v > 0 else 'blue' for v in mse_vals])
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Mean Signed Error (Actual - Predicted)")
    plt.title("Over/Under-prediction by Season (Test Set)")
    plt.tight_layout()
    
    chart_path = resolve_path(config["paths"]["charts"]) / "bias_analysis_season_error.png"
    plt.savefig(chart_path)
    plt.close()

if __name__ == "__main__":
    main()