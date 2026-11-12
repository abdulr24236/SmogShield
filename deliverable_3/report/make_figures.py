"""Generate the figures and worked examples used by the final D3 report.

Run from ``deliverable_3``:
    python report/make_figures.py

The script reads the frozen metadata/ablation artifacts and the persisted
pipelines. It never refits or changes a model. All figures are written to
``report/figures/`` and the two worked examples to ``report/examples.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "src"))

from data import load_config, prepare_model_data  # noqa: E402
from src.inference import aqi_to_category, inference, pm25_to_aqi  # noqa: E402

META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
ABLATION = json.loads((ROOT / "models" / "ablation.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Shared plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

BLUE = "#2E5EAA"
ORANGE = "#D97706"
GREEN = "#2A9D8F"
RED = "#B91C1C"
GREY = "#6B7280"
LIGHT = "#DBEAFE"


def save(fig: plt.Figure, name: str) -> None:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path.relative_to(PROJECT))


# ---------------------------------------------------------------------------
# 1. Ablation
# ---------------------------------------------------------------------------
def plot_ablation() -> None:
    labels = ["History", "History +\ncalendar", "Full\n(+ weather)"]
    keys = ["history", "history+calendar", "full"]

    reg_val = [ABLATION["regression"][k]["metrics"]["validation"]["rmse"] for k in keys]
    reg_test = [ABLATION["regression"][k]["metrics"]["test"]["rmse"] for k in keys]
    clf_val = [
        ABLATION["classification"][k]["metrics"]["validation"]["macro_f1"] for k in keys
    ]
    clf_test = [
        ABLATION["classification"][k]["metrics"]["test"]["macro_f1"] for k in keys
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))

    for ax, val, test, title, ylab, ylim, fmt in (
        (
            axes[0],
            reg_val,
            reg_test,
            "Regression: RMSE (lower is better)",
            "RMSE (µg/m³)",
            (0, 78),
            "{:.2f}",
        ),
        (
            axes[1],
            clf_val,
            clf_test,
            "Classification: macro F1 (higher is better)",
            "Macro F1",
            (0, 0.62),
            "{:.3f}",
        ),
    ):
        b1 = ax.bar(x - width / 2, val, width, label="Validation", color=BLUE)
        b2 = ax.bar(x + width / 2, test, width, label="Test", color=ORANGE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper left", ncol=2)
        for bars in (b1, b2):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    fig.suptitle(
        "Nested feature-set ablation with frozen hyperparameters", fontsize=11
    )
    fig.tight_layout()
    save(fig, "fig_ablation.png")


# ---------------------------------------------------------------------------
# 2. Permutation importance
# ---------------------------------------------------------------------------
def plot_importance() -> None:
    reg = META["feature_importance"]["regression"][:12]
    clf = META["feature_importance"]["classification"][:12]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, rows, title, xlab, color in (
        (
            axes[0],
            reg,
            "Regression: increase in validation RMSE",
            "Increase in RMSE (µg/m³)",
            BLUE,
        ),
        (
            axes[1],
            clf,
            "Classification: drop in validation macro F1",
            "Drop in macro F1",
            GREEN,
        ),
    ):
        names = [row["feature"] for row in rows][::-1]
        values = [row["importance"] for row in rows][::-1]
        ax.barh(names, values, color=color, height=0.68)
        ax.set_xlabel(xlab)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for y, value in enumerate(values):
            ax.text(
                value,
                y,
                f" {value:.3f}" if value < 1 else f" {value:.2f}",
                va="center",
                fontsize=7,
            )
        ax.set_xlim(0, max(values) * 1.22)
    fig.suptitle("Permutation importance on the validation split", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_importance.png")


# ---------------------------------------------------------------------------
# 3. Seasonal profile
# ---------------------------------------------------------------------------
def plot_seasonal() -> None:
    monthly = META["patterns"]["monthly_median_pm25"]
    months = list(range(1, 13))
    values = [monthly[str(m)] for m in months]
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    seasonal = META["patterns"]["seasonal_mean_pm25"]
    order = ["winter", "spring", "summer", "autumn"]
    seasonal_values = [seasonal[s] for s in order]
    seasonal_names = ["Winter\n(Dec–Feb)", "Spring\n(Mar–May)", "Summer\n(Jun–Aug)", "Autumn\n(Sep–Nov)"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    colors = [BLUE] * 12
    colors[10] = ORANGE
    colors[11] = RED
    axes[0].bar(names, values, color=colors, width=0.72)
    axes[0].set_ylabel("Median PM2.5 (µg/m³)")
    axes[0].set_title("Historical monthly median PM2.5")
    axes[0].set_ylim(0, 290)
    for i, value in enumerate(values):
        axes[0].text(i, value + 5, f"{value:.0f}", ha="center", fontsize=7)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].text(
        4.3,
        266,
        "November and December are the highest months",
        fontsize=7.2,
        color=GREY,
    )

    axes[1].bar(
        seasonal_names,
        seasonal_values,
        color=[BLUE, GREEN, GREEN, ORANGE],
        width=0.62,
    )
    axes[1].set_ylabel("Mean PM2.5 (µg/m³)")
    axes[1].set_title("Historical seasonal mean PM2.5")
    axes[1].set_ylim(0, 260)
    for i, value in enumerate(seasonal_values):
        axes[1].text(i, value + 4, f"{value:.0f}", ha="center", fontsize=8)
    fig.suptitle("Seasonality in the cleaned Lahore daily series", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_seasonal.png")


# ---------------------------------------------------------------------------
# 4. Hourly profile
# ---------------------------------------------------------------------------
def plot_hourly() -> None:
    hourly = META["patterns"]["hourly"]
    by_hour = hourly["by_hour"]
    hours = list(range(24))
    overall = [by_hour[str(h)]["mean"] for h in hours]
    winter = hourly["seasonal_by_hour"]["winter"]
    summer = hourly["seasonal_by_hour"]["summer"]

    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.plot(hours, overall, marker="o", ms=3.5, color=GREY, lw=1.6, label="All seasons")
    ax.plot(hours, winter, marker="o", ms=3.5, color=BLUE, lw=1.8, label="Winter (Dec–Feb)")
    ax.plot(hours, summer, marker="o", ms=3.0, color=GREEN, lw=1.2, alpha=0.8, label="Summer (Jun–Aug)")
    ax.axvspan(14, 16.99, color=LIGHT, alpha=0.75, zorder=0)
    ax.axvline(22, color=RED, ls="--", lw=1.0)
    ax.text(14.05, 245, "cleanest 3-hour window\n14:00–16:59", fontsize=7.5, color=BLUE)
    ax.text(21.55, 245, "peak 22:00", fontsize=7.5, color=RED, ha="right")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=0, fontsize=7)
    ax.set_xlabel("Hour of day (Pakistan local time)")
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    ax.set_title("Mean PM2.5 by hour: all seasons, winter and summer")
    ax.set_ylim(30, 270)
    ax.legend(loc="upper left", ncol=3, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_hourly.png")


# ---------------------------------------------------------------------------
# 5. Interval coverage and default example
# ---------------------------------------------------------------------------
def plot_interval(default: dict) -> None:
    interval = META["interval"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5))

    names = ["Nominal", "Validation", "Test"]
    values = [interval["level"] * 100, interval["coverage_validation"] * 100, interval["coverage_test"] * 100]
    bars = axes[0].bar(names, values, color=[GREY, BLUE, ORANGE], width=0.55)
    axes[0].axhline(80, color=RED, ls="--", lw=1.0)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title("80% interval: nominal vs empirical coverage")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=8)

    y = 0
    low = default["low"]
    high = default["high"]
    pred = default["prediction"]
    persistence = default["persistence"]
    axes[1].hlines(y, low, high, color=BLUE, lw=6, alpha=0.35)
    axes[1].plot([low, high], [y, y], "|", color=BLUE, ms=14, mew=2)
    axes[1].plot(pred, y, "o", color=BLUE, ms=8, label=f"Forecast {pred:.1f}")
    axes[1].plot(persistence, y + 0.32, "D", color=ORANGE, ms=6, label=f"Persistence {persistence:.1f}")
    axes[1].set_yticks([])
    axes[1].set_ylim(-0.45, 0.75)
    axes[1].set_xlim(100, 320)
    axes[1].set_xlabel("Next-day PM2.5 (µg/m³)")
    axes[1].set_title("Default November example")
    axes[1].text(low, -0.22, f"{low:.1f}", ha="center", fontsize=8, color=BLUE)
    axes[1].text(high, -0.22, f"{high:.1f}", ha="center", fontsize=8, color=BLUE)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Uncertainty is reported, not hidden", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_interval.png")


# ---------------------------------------------------------------------------
# 6. Backtest example
# ---------------------------------------------------------------------------
def plot_backtest(backtest: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    rows = [
        ("Today's PM2.5", backtest["today_pm25"], None, GREY),
        ("Persistence", backtest["persistence"], None, ORANGE),
        ("App six-input forecast", backtest["app_prediction"], (backtest["app_low"], backtest["app_high"]), BLUE),
        ("Full-feature pipeline", backtest["full_prediction"], None, GREEN),
        ("Actual next day", backtest["actual_next_day"], None, RED),
    ]
    for i, (label, value, interval, color) in enumerate(rows):
        ax.plot(value, i, "o", color=color, ms=7)
        if interval:
            ax.hlines(i, interval[0], interval[1], color=color, lw=5, alpha=0.3)
            ax.text(interval[0], i - 0.28, f"{interval[0]:.1f}", ha="center", fontsize=7, color=color)
            ax.text(interval[1], i - 0.28, f"{interval[1]:.1f}", ha="center", fontsize=7, color=color)
        ax.text(value + 5, i + 0.13, f"{value:.1f}", fontsize=8, color=color)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_xlim(90, 330)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_title("Backtest: 1 November 2024 inputs → 2 November 2024 outcome")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_backtest.png")


# ---------------------------------------------------------------------------
# 7. Architecture diagram
# ---------------------------------------------------------------------------
def _box(ax, xy, width, height, title, lines, color, title_size=8.5, body_size=7.0) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height - 0.06, title, ha="center", va="top", fontsize=title_size, weight="bold", color=color)
    ax.text(x + width / 2, y + height - 0.20, "\n".join(lines), ha="center", va="top", fontsize=body_size, color="#1F2937")


def _arrow(ax, start, end, color=GREY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
        )
    )


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.02, 0.62), 0.26, 0.28, "Data layer", ["data_scripts", "processed features"], GREY, title_size=12.5, body_size=10.5)
    _box(ax, (0.36, 0.62), 0.28, 0.28, "Model layer", ["train_pipeline", "HGB + quantile HGB"], BLUE, title_size=12.5, body_size=10.5)
    _box(ax, (0.70, 0.62), 0.28, 0.28, "Application layer", ["inference", "features + intervals"], GREEN, title_size=12.5, body_size=10.5)
    _box(ax, (0.02, 0.14), 0.26, 0.28, "Evaluation layer", ["ablation + tests", "reproducibility"], RED, title_size=12.5, body_size=10.5)
    _box(ax, (0.36, 0.14), 0.28, 0.28, "User outputs", ["forecast + guidance", "explanations"], GREY, title_size=12.5, body_size=10.5)
    _box(ax, (0.70, 0.14), 0.28, 0.28, "Interface layer", ["Streamlit app", "six inputs"], ORANGE, title_size=12.5, body_size=10.5)

    _arrow(ax, (0.28, 0.76), (0.36, 0.76))
    _arrow(ax, (0.64, 0.76), (0.70, 0.76))
    _arrow(ax, (0.84, 0.62), (0.84, 0.42))
    _arrow(ax, (0.70, 0.28), (0.64, 0.28))
    _arrow(ax, (0.16, 0.42), (0.40, 0.62), color=RED)
    ax.text(0.24, 0.54, "evaluate / refit", fontsize=8.5, color=RED, ha="center")
    ax.text(0.5, 0.04, "Everything is regenerated from committed scripts; fitted binaries and processed data are gitignored.", ha="center", fontsize=9.0, color=GREY)
    ax.set_title("SmogShield Deliverable 3: system architecture", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_architecture.png")


# ---------------------------------------------------------------------------
# 8. UI flow
# ---------------------------------------------------------------------------
def plot_ui_flow() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.02, 0.54, "1. Inputs", ["date · PM2.5 · category", "temperature · humidity · wind"], BLUE),
        (0.36, 0.54, "2. Forecast", ["PM2.5 · implied AQI", "classifier label · confidence"], GREEN),
        (0.70, 0.54, "3. Uncertainty", ["80% interval", "coverage note"], ORANGE),
        (0.70, 0.10, "4. Guidance", ["cautious category text", "non-medical disclaimer"], RED),
        (0.36, 0.10, "5. Explanation", ["local sensitivity", "+ global importance"], BLUE),
        (0.02, 0.10, "6. Context", ["stagnation · seasons", "hourly window"], GREY),
    ]
    width, height = 0.28, 0.30
    for x, y, title, lines, color in boxes:
        _box(ax, (x, y), width, height, title, lines, color, title_size=11.5, body_size=9.5)

    _arrow(ax, (0.30, 0.69), (0.36, 0.69))
    _arrow(ax, (0.64, 0.69), (0.70, 0.69))
    _arrow(ax, (0.84, 0.54), (0.84, 0.40))
    _arrow(ax, (0.70, 0.25), (0.64, 0.25))
    _arrow(ax, (0.36, 0.25), (0.30, 0.25))

    ax.set_title("Forecast screen: from six inputs to an explained, bounded forecast", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_ui_flow.png")


# ---------------------------------------------------------------------------
# Worked examples computed from the frozen pipelines
# ---------------------------------------------------------------------------
def compute_examples() -> dict:
    config = load_config()
    reg_data = prepare_model_data(config, "regression")
    target = config["data"]["target_regression"]
    test = reg_data.split.test
    date_col = config["data"]["date_column"]

    row = test[test[date_col] == pd.Timestamp("2024-11-01")]
    if row.empty:
        raise RuntimeError("2024-11-01 is not in the test split")
    idx = row.index[0]
    X_row = reg_data.X_test_raw.loc[[idx]]
    reg_pipe = joblib.load(ROOT / "models" / "regression_pipeline.joblib")
    full_prediction = float(reg_pipe.predict(X_row)[0])
    actual = float(row[target].iloc[0])
    today_pm25 = float(row["pm25"].iloc[0])

    category = str(row["aqi_category"].iloc[0])
    if category == "Good/Moderate":
        category = "Moderate"

    def _num(name: str) -> float | None:
        value = row[name].iloc[0]
        return None if pd.isna(value) else float(value)

    app_inputs = {
        "pm25": today_pm25,
        "aqi_category": category,
        "temperature_c": _num("temp_mean_c"),
        "humidity_pct": _num("humidity_mean_pct"),
        "wind_kmh": _num("wind_speed_mean_kmh"),
        "date": "2024-11-01",
    }
    app_result = inference.predict(app_inputs)

    monthly = META["patterns"]["monthly_median_pm25"]
    default_pm25 = float(monthly["11"])
    default_category = aqi_to_category(pm25_to_aqi(default_pm25))
    if default_category == "Good/Moderate":
        default_category = "Moderate"
    default_inputs = {
        "pm25": default_pm25,
        "aqi_category": default_category,
        "temperature_c": 22.0,
        "humidity_pct": 60.0,
        "wind_kmh": 7.0,
        "date": "2026-11-12",
    }
    default_result = inference.predict(default_inputs)

    examples = {
        "backtest": {
            "feature_date": "2024-11-01",
            "target_date": "2024-11-02",
            "today_pm25": round(today_pm25, 2),
            "app_prediction": round(float(app_result["predicted_next_pm25"]), 2),
            "app_low": round(float(app_result["interval"]["low"]), 1),
            "app_high": round(float(app_result["interval"]["high"]), 1),
            "full_prediction": round(full_prediction, 2),
            "actual_next_day": round(actual, 2),
            "persistence": round(today_pm25, 2),
            "app_category": app_result["predicted_category_from_pm25"],
            "classifier_category": app_result["predicted_next_aqi_category"],
            "explanation": app_result["explanation"],
        },
        "default": {
            "feature_date": "2026-11-12",
            "pm25": round(default_pm25, 1),
            "pm25_category": default_category,
            "prediction": round(float(default_result["predicted_next_pm25"]), 2),
            "aqi": int(default_result["predicted_aqi"]),
            "implied_category": default_result["predicted_category_from_pm25"],
            "classifier_category": default_result["predicted_next_aqi_category"],
            "low": round(float(default_result["interval"]["low"]), 1),
            "high": round(float(default_result["interval"]["high"]), 1),
            "persistence": round(default_pm25, 1),
            "explanation": default_result["explanation"],
        },
    }
    (OUT / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print("wrote", (OUT / "examples.json").relative_to(PROJECT))
    return examples


def main() -> None:
    plot_ablation()
    plot_importance()
    plot_seasonal()
    plot_hourly()
    examples = compute_examples()
    plot_interval(examples["default"])
    plot_backtest(examples["backtest"])
    plot_architecture()
    plot_ui_flow()
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
