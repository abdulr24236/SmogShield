"""Next-day AQI classification — Deliverable 2.

Owner: Muhammad Haider Mughal

Run from the repository root:
    python -m src.classification

Outputs (all git-ignored; regenerate from this script + shared data):
    outputs/model_results/classification_metrics.json
    outputs/model_results/classification_hyperparameters.json
    outputs/model_charts/classification_confusion_matrix.png
    outputs/model_charts/classification_hgb_val_curve.png
    outputs/model_charts/classification_lr_C_curve.png
    outputs/model_charts/classification_mlp_loss_curve.png   (MLP only)

Design rules (from docs/modeling_contract.md):
  - Chronological split, gap_days=7; labels and merge from configs/base.yaml.
  - Tune on X_train_raw with make_model_pipeline + TimeSeriesSplit(gap=7).
  - Final model trained on training split only (not train+validation).
  - Test touched once per final model after hyperparameters are frozen.
  - Scoring via shared scorable_rows masks and classification_metrics.
  - align_proba used for every predict_proba call before classification_metrics.
  - AUC errors are never silently dropped; roc_auc_note is preserved.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Project-local imports — both modules live in the same package (src/).
# ---------------------------------------------------------------------------
from src.data import load_config, prepare_model_data
from src.evaluate import (
    align_proba,
    classification_metrics,
    majority_class_predictions,
    persistence_class_predictions,
    save_metrics,
    print_metrics,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "model_charts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seed (from config; kept as module-level constant for reproducibility notes)
# ---------------------------------------------------------------------------
_SEED = 42


# ===========================================================================
# Helpers
# ===========================================================================


def _scorable_subset(
    mask: pd.Series,
    *arrays: np.ndarray | pd.Series,
) -> list[np.ndarray]:
    """Filter every array/Series to the scorable rows defined by *mask*.

    The mask is aligned to the array index when the input is a Series, and
    applied positionally when it is a plain ndarray.
    """
    idx = mask.to_numpy(dtype=bool)
    result = []
    for arr in arrays:
        if isinstance(arr, pd.Series):
            result.append(arr.to_numpy()[idx])
        else:
            result.append(np.asarray(arr)[idx])
    return result


def _int_labels(labels: list[str]) -> list[int]:
    """Return the integer class codes aligned with *labels*."""
    return list(range(len(labels)))


def _score_model(
    name: str,
    y_true_full: pd.Series,
    y_pred_full: np.ndarray,
    mask: pd.Series,
    labels: list[str],
    y_score_full: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one model/baseline on the shared scorable rows.

    Parameters
    ----------
    name:
        Human-readable label for printing.
    y_true_full, y_pred_full:
        Full-split true labels and predictions (integer codes), aligned with
        the split index before masking.
    mask:
        Boolean Series from ``data.scorable_rows[split]``.
    labels:
        Ordered class-name strings from ``data.labels``.
    y_score_full:
        Optional full-split probability matrix (n_samples, n_classes) already
        aligned with *labels* via ``align_proba``.
    """
    int_lbl = _int_labels(labels)

    if y_score_full is not None:
        y_true_s, y_pred_s, y_score_s = _scorable_subset(
            mask, y_true_full, y_pred_full, y_score_full
        )
    else:
        y_true_s, y_pred_s = _scorable_subset(mask, y_true_full, y_pred_full)
        y_score_s = None

    metrics = classification_metrics(
        y_true=y_true_s,
        y_pred=y_pred_s,
        y_score=y_score_s,
        labels=int_lbl,
        class_names=labels,
    )
    print_metrics(metrics, title=name)
    return metrics


# ===========================================================================
# Baselines
# ===========================================================================


def run_majority_baseline(data: Any, split: str) -> dict[str, Any]:
    """Predict the training majority class for every scorable row in *split*.

    The majority code is derived from y_train; no fitting on validation/test.
    """
    y_train = data.y_train
    mask = data.scorable_rows[split]

    if split == "train":
        y_true_full = data.y_train
    elif split == "validation":
        y_true_full = data.y_validation
    else:
        y_true_full = data.y_test

    n_full = len(y_true_full)
    y_pred_full = majority_class_predictions(y_train.to_numpy(), n_full)

    return _score_model(
        f"Majority baseline [{split}]",
        y_true_full,
        y_pred_full,
        mask,
        data.labels,
    )


def _current_aqi_category(X_raw: pd.DataFrame) -> pd.Series:
    """Extract the raw current-day AQI category string from *X_raw*."""
    if "aqi_category" not in X_raw.columns:
        raise KeyError(
            "Feature frame does not contain 'aqi_category'. "
            "Cannot build a persistence baseline."
        )
    return X_raw["aqi_category"]


def run_persistence_baseline(data: Any, split: str) -> dict[str, Any]:
    """Predict tomorrow = today's AQI category (with label_merge + fallback).

    Fallback: the training majority class (documented).  The persistence
    baseline uses the raw 'aqi_category' column from X_*_raw; it is not
    allowed to look at the encoded target.

    Missing-category rows get the majority-class code as fallback.  The
    fallback class is 'Unhealthy' (code 2 in the six-class ordering) because
    it dominates the training distribution.  The exact majority is computed
    dynamically from y_train so it stays correct if the data changes.
    """
    # Determine majority-class fallback from training labels
    y_train_arr = data.y_train.to_numpy()
    values, counts = np.unique(y_train_arr, return_counts=True)
    fallback_code = int(values[int(np.argmax(counts))])
    fallback_name = data.labels[fallback_code]

    if split == "train":
        X_raw = data.X_train_raw
        y_true_full = data.y_train
    elif split == "validation":
        X_raw = data.X_validation_raw
        y_true_full = data.y_validation
    else:
        X_raw = data.X_test_raw
        y_true_full = data.y_test

    current_cats = _current_aqi_category(X_raw)
    mask = data.scorable_rows[split]

    y_pred_full = persistence_class_predictions(
        current_categories=current_cats,
        labels=data.labels,
        label_merge=data.label_merge,
        fallback=fallback_code,
    )

    print(
        f"\n[Persistence baseline] fallback for missing current category: "
        f"'{fallback_name}' (code {fallback_code})"
    )

    return _score_model(
        f"Persistence baseline [{split}]",
        y_true_full,
        y_pred_full,
        mask,
        data.labels,
    )


# ===========================================================================
# Tuning helpers
# ===========================================================================


def _time_series_cv(n_splits: int = 5, gap: int = 7) -> TimeSeriesSplit:
    """Return the shared TimeSeriesSplit used for all tuning."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def _tune_model(
    data: Any,
    estimator: Any,
    param_distributions: dict[str, Any],
    n_iter: int,
    cv: TimeSeriesSplit,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    """Run RandomizedSearchCV on X_train_raw, return best pipeline + info.

    The search is run on X_train_raw (not X_train) so that each fold's
    preprocessor is fitted only on that fold's training rows — this avoids
    leaking held-out scaling statistics.

    Selection metric: macro F1 (primary).
    Secondary checks: balanced_accuracy, roc_auc_ovr_macro (reported but not
    used for selection).

    Returns
    -------
    best_pipeline:
        The refitted best pipeline (fitted on the full training raw data).
    tuning_info:
        A dict recording the search space, selected values, CV folds,
        selection metric and best validation score.
    """
    pipeline = data.make_model_pipeline(estimator)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        refit=True,          # refit on full X_train_raw after search
        n_jobs=-1,
        random_state=_SEED,
        return_train_score=False,
        error_score="raise",
    )

    print(f"\n[Tuning] {name} — {n_iter} random iterations, {cv.n_splits} folds …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        search.fit(data.X_train_raw, data.y_train)

    best_params = {
        # Strip the "model__" prefix added by Pipeline for readability.
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    best_score = float(search.best_score_)

    tuning_info: dict[str, Any] = {
        "model": name,
        "cv_folds": cv.n_splits,
        "cv_gap_days": cv.gap,
        "selection_metric": "f1_macro",
        "best_cv_f1_macro": round(best_score, 6),
        "search_space": {
            key.replace("model__", ""): (
                str(value) if not isinstance(value, (list, tuple)) else value
            )
            for key, value in param_distributions.items()
        },
        "selected_params": best_params,
        "n_iter": n_iter,
    }

    print(f"  Best CV macro F1: {best_score:.4f}")
    print(f"  Selected params : {best_params}")

    return search.best_estimator_, tuning_info


# ===========================================================================
# Logistic regression
# ===========================================================================


def build_and_tune_logistic(
    data: Any, cv: TimeSeriesSplit
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune a regularised logistic regression on the training split.

    Search space (§1.4):
        C, penalty, class_weight, max_iter
    """
    param_distributions = {
        "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [500, 1000, 2000, 5000],
    }

    estimator = LogisticRegression(
        solver="saga",  # supports both l1 and l2
        random_state=_SEED,
    )

    return _tune_model(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=36,
        cv=cv,
        name="LogisticRegression",
    )


# ===========================================================================
# Histogram gradient boosting
# ===========================================================================


def build_and_tune_hgb(
    data: Any, cv: TimeSeriesSplit
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune a HistGradientBoostingClassifier on the training split.

    Search space (§1.4):
        learning_rate, max_iter, max_leaf_nodes, l2_regularization,
        min_samples_leaf
    """
    param_distributions = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "model__max_iter": [100, 200, 300, 500],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        "model__min_samples_leaf": [10, 20, 30, 50],
    }

    estimator = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=_SEED,
        early_stopping=False,  # tuning controls iterations explicitly
    )

    return _tune_model(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=48,
        cv=cv,
        name="HistGradientBoosting",
    )


# ===========================================================================
# MLP (optional — trained only if it beats HGB on validation macro F1)
# ===========================================================================


def build_and_tune_mlp(
    data: Any, cv: TimeSeriesSplit
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune a small MLP on the training split.

    Search space (§1.4):
        hidden layers, alpha, learning_rate, early_stopping
    """
    param_distributions = {
        "model__hidden_layer_sizes": [
            (64,),
            (128,),
            (64, 32),
            (128, 64),
            (64, 64),
            (128, 64, 32),
        ],
        "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "model__learning_rate": ["constant", "adaptive"],
        "model__early_stopping": [True],   # always use early stopping for MLP
        "model__validation_fraction": [0.1],
        "model__n_iter_no_change": [15],
        "model__max_iter": [300, 500],
    }

    estimator = MLPClassifier(
        activation="relu",
        solver="adam",
        random_state=_SEED,
    )

    return _tune_model(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=24,
        cv=cv,
        name="MLP",
    )


# ===========================================================================
# Validation scoring (used for MLP inclusion decision)
# ===========================================================================


def _val_macro_f1(
    pipeline: Pipeline,
    data: Any,
) -> float:
    """Compute macro F1 on the scorable validation rows."""
    mask = data.scorable_rows["validation"]
    y_pred_full = pipeline.predict(data.X_validation_raw)
    y_true_s, y_pred_s = _scorable_subset(mask, data.y_validation, y_pred_full)
    from sklearn.metrics import f1_score
    return float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0))


# ===========================================================================
# Final model evaluation (train only → val + test once)
# ===========================================================================


def evaluate_final(
    name: str,
    pipeline: Pipeline,
    data: Any,
) -> dict[str, dict[str, Any]]:
    """Refit on training split, then score validation and test once.

    Parameters
    ----------
    name:
        Display name for logging.
    pipeline:
        Already-fitted pipeline (from tuning). Its hyperparameters are frozen;
        we refit fresh on X_train_raw only to produce a clean final model.
    data:
        ModelData container.

    Returns
    -------
    A dict with keys 'validation' and 'test', each containing the full
    classification_metrics output.
    """
    print(f"\n[Final] Refitting {name} on training split …")

    # Clone hyperparameters, refit on X_train_raw only.
    from sklearn.base import clone as sk_clone
    fresh = sk_clone(pipeline)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        fresh.fit(data.X_train_raw, data.y_train)

    results: dict[str, dict[str, Any]] = {}

    for split_name in ("validation", "test"):
        if split_name == "validation":
            X_raw = data.X_validation_raw
            y_true_full = data.y_validation
        else:
            X_raw = data.X_test_raw
            y_true_full = data.y_test

        mask = data.scorable_rows[split_name]

        # Predictions
        y_pred_full = fresh.predict(X_raw)

        # Probabilities — align to fixed class-code order
        estimator_step = fresh.named_steps["model"]
        if hasattr(estimator_step, "predict_proba"):
            proba_raw = fresh.predict_proba(X_raw)
            y_score_full = align_proba(
                proba_raw,
                fresh.classes_,
                len(data.labels),
            )
        else:
            y_score_full = None

        metrics = _score_model(
            f"{name} [{split_name}]",
            y_true_full,
            y_pred_full,
            mask,
            data.labels,
            y_score_full,
        )
        results[split_name] = metrics

    return results


# ===========================================================================
# Chart helpers
# ===========================================================================


def _save_fig(path: Path) -> None:
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path.relative_to(PROJECT_ROOT)}")


def plot_confusion_matrix(
    name: str,
    metrics: dict[str, Any],
    labels: list[str],
    split: str,
) -> None:
    """Plot and save the confusion matrix for one model/split."""
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    short = [lbl[:20] for lbl in labels]   # truncate long names
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"{name} — Confusion matrix ({split})")
    fig.tight_layout()

    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    path = CHARTS_DIR / f"classification_confusion_{safe_name}_{split}.png"
    _save_fig(path)


def plot_hgb_val_curve(
    pipeline: Pipeline,
    data: Any,
    name: str = "HistGradientBoosting",
) -> None:
    """Plot macro F1 vs number of boosting iterations on the validation set.

    The pipeline must have a 'model' step that is a
    HistGradientBoostingClassifier.  The curve is generated by scoring the
    staged predictions produced by warm-starting a sequence of models with
    increasing max_iter.

    Note: HistGradientBoostingClassifier does not expose staged_predict, so
    we re-train with warm-starting at exponentially spaced iteration counts.
    """
    from sklearn.metrics import f1_score as sk_f1
    from sklearn.base import clone as sk_clone

    hgb_step = pipeline.named_steps["model"]
    if not isinstance(hgb_step, HistGradientBoostingClassifier):
        print(f"  [Skip] {name} val curve: model step is not HGB.")
        return

    # Freeze the best HGB hyperparameters (excluding max_iter).
    best_params = hgb_step.get_params()
    max_iters = [10, 25, 50, 75, 100, 150, 200, 300, 500]
    max_iters = [m for m in max_iters if m <= (best_params.get("max_iter") or 300)]
    if not max_iters:
        max_iters = [best_params.get("max_iter", 100)]

    mask_val = data.scorable_rows["validation"]
    scores = []

    for n_iter in max_iters:
        params_i = {**best_params, "max_iter": n_iter, "early_stopping": False}
        hgb_i = HistGradientBoostingClassifier(**params_i)
        pipe_i = sk_clone(pipeline)
        pipe_i.set_params(model=hgb_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_i.fit(data.X_train_raw, data.y_train)
        y_pred = pipe_i.predict(data.X_validation_raw)
        y_t, y_p = _scorable_subset(mask_val, data.y_validation, y_pred)
        f1 = sk_f1(y_t, y_p, average="macro", zero_division=0)
        scores.append(f1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(max_iters, scores, marker="o", linewidth=2)
    ax.set_xlabel("max_iter (boosting rounds)")
    ax.set_ylabel("Validation macro F1")
    ax.set_title(f"{name} — Validation macro F1 vs iterations")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(CHARTS_DIR / "classification_hgb_val_curve.png")


def plot_lr_C_curve(
    pipeline: Pipeline,
    data: Any,
) -> None:
    """Plot validation macro F1 vs regularisation strength C for LogReg.

    Note: Loss curves are not meaningful for logistic regression trained with
    full-batch solvers; we report a hyperparameter sensitivity curve instead.
    """
    from sklearn.metrics import f1_score as sk_f1
    from sklearn.base import clone as sk_clone

    lr_step = pipeline.named_steps["model"]
    if not isinstance(lr_step, LogisticRegression):
        print("  [Skip] LR C-curve: model step is not LogisticRegression.")
        return

    best_params = lr_step.get_params()
    C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    mask_val = data.scorable_rows["validation"]
    scores = []

    for C in C_values:
        params_i = {**best_params, "C": C}
        lr_i = LogisticRegression(**params_i)
        pipe_i = sk_clone(pipeline)
        pipe_i.set_params(model=lr_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe_i.fit(data.X_train_raw, data.y_train)
        y_pred = pipe_i.predict(data.X_validation_raw)
        y_t, y_p = _scorable_subset(mask_val, data.y_validation, y_pred)
        f1 = sk_f1(y_t, y_p, average="macro", zero_division=0)
        scores.append(f1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(C_values, scores, marker="s", linewidth=2)
    ax.axvline(best_params["C"], color="red", linestyle="--", label=f"Selected C={best_params['C']}")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("LogisticRegression — Validation macro F1 vs C")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(CHARTS_DIR / "classification_lr_C_curve.png")


def plot_mlp_loss_curve(pipeline: Pipeline) -> None:
    """Plot training loss curve from MLP's loss_curve_ attribute."""
    mlp_step = pipeline.named_steps["model"]
    if not isinstance(mlp_step, MLPClassifier):
        print("  [Skip] MLP loss curve: model step is not MLP.")
        return
    if not hasattr(mlp_step, "loss_curve_") or mlp_step.loss_curve_ is None:
        print("  [Skip] MLP loss curve: loss_curve_ not available.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mlp_step.loss_curve_, linewidth=2, label="Training loss")
    if hasattr(mlp_step, "validation_scores_") and mlp_step.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(
            mlp_step.validation_scores_,
            linewidth=2, color="orange", linestyle="--",
            label="Validation accuracy",
        )
        ax2.set_ylabel("Validation accuracy")
        ax2.legend(loc="upper right")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title("MLP — Training loss per epoch")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(CHARTS_DIR / "classification_mlp_loss_curve.png")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("=" * 70)
    print("SmogShield — Next-day AQI classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load shared data
    # ------------------------------------------------------------------
    config = load_config()
    seed = int(config.get("seed", _SEED))
    data = prepare_model_data(config, "classification")

    labels: list[str] = data.labels  # type: ignore[assignment]
    n_classes = len(labels)
    int_lbl = _int_labels(labels)

    print("\nData summary:")
    summary = data.summary()
    print(f"  Task:    {summary['task']}")
    print(f"  Target:  {summary['target']}")
    print(
        f"  Rows:    train={summary['train_rows']} "
        f"val={summary['validation_rows']} "
        f"test={summary['test_rows']}"
    )
    print(f"  Classes: {n_classes} — {labels}")
    if data.label_merge:
        print(f"  Merge:   {data.label_merge}")
    for split_name, counts in summary["class_counts"].items():
        rendered = ", ".join(f"{lbl}={cnt}" for lbl, cnt in counts.items())
        print(f"  {split_name:12s} {rendered}")
    scorable = summary["scorable_rows"]
    print(
        f"  Scorable: train={scorable['train']} "
        f"val={scorable['validation']} "
        f"test={scorable['test']}"
    )

    # ------------------------------------------------------------------
    # 2. Baselines
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    baseline_metrics: dict[str, dict[str, Any]] = {}

    for split_name in ("train", "validation"):
        baseline_metrics[f"majority_{split_name}"] = run_majority_baseline(
            data, split_name
        )
        baseline_metrics[f"persistence_{split_name}"] = run_persistence_baseline(
            data, split_name
        )

    # ------------------------------------------------------------------
    # 3. Hyperparameter tuning (train split only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING (training split, TimeSeriesSplit gap=7)")
    print("=" * 70)

    cv = _time_series_cv(n_splits=5, gap=7)
    tuning_records: list[dict[str, Any]] = []

    lr_pipeline, lr_tuning = build_and_tune_logistic(data, cv)
    tuning_records.append(lr_tuning)

    hgb_pipeline, hgb_tuning = build_and_tune_hgb(data, cv)
    tuning_records.append(hgb_tuning)

    # ------------------------------------------------------------------
    # 4. MLP — include only if it beats HGB on validation macro F1
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MLP (optional — included only if val macro F1 > HGB)")
    print("=" * 70)

    hgb_val_f1 = _val_macro_f1(hgb_pipeline, data)
    print(f"  HGB validation macro F1 (reference): {hgb_val_f1:.4f}")

    mlp_pipeline: Pipeline | None = None
    mlp_tuning: dict[str, Any] | None = None
    mlp_val_f1 = 0.0

    try:
        mlp_pipeline_candidate, mlp_tuning_candidate = build_and_tune_mlp(data, cv)
        mlp_val_f1 = _val_macro_f1(mlp_pipeline_candidate, data)
        print(f"  MLP validation macro F1:              {mlp_val_f1:.4f}")

        if mlp_val_f1 > hgb_val_f1:
            print("  → MLP included (beats HGB on validation macro F1).")
            mlp_pipeline = mlp_pipeline_candidate
            mlp_tuning = mlp_tuning_candidate
            tuning_records.append(mlp_tuning_candidate)
        else:
            print("  → MLP excluded (does not improve over HGB).")
    except Exception as exc:
        print(f"  [Warning] MLP tuning failed: {exc}. Skipping MLP.")

    # ------------------------------------------------------------------
    # 5. Final model evaluation (train → val + test, test touched once)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (refit on train only; test touched once)")
    print("=" * 70)

    all_metrics: dict[str, Any] = {
        "baselines": baseline_metrics,
        "models": {},
        "hyperparameters": tuning_records,
        "data_summary": {
            "task": summary["task"],
            "target": summary["target"],
            "train_rows": summary["train_rows"],
            "validation_rows": summary["validation_rows"],
            "test_rows": summary["test_rows"],
            "scorable_rows": scorable,
            "class_labels": labels,
            "label_merge": data.label_merge,
            "class_counts": summary["class_counts"],
            "split": summary["split"],
        },
    }

    # Logistic regression
    lr_results = evaluate_final("LogisticRegression", lr_pipeline, data)
    all_metrics["models"]["logistic_regression"] = lr_results

    # Histogram gradient boosting
    hgb_results = evaluate_final("HistGradientBoosting", hgb_pipeline, data)
    all_metrics["models"]["hgb"] = hgb_results

    # MLP (conditional)
    if mlp_pipeline is not None:
        mlp_results = evaluate_final("MLP", mlp_pipeline, data)
        all_metrics["models"]["mlp"] = mlp_results
    else:
        all_metrics["models"]["mlp_note"] = (
            f"MLP excluded: val macro F1 ({mlp_val_f1:.4f}) "
            f"did not exceed HGB ({hgb_val_f1:.4f})."
        )

    # ------------------------------------------------------------------
    # 6. Summary comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — Validation macro F1")
    print("=" * 70)

    def _get_val_f1(blob: dict[str, Any]) -> float:
        try:
            return float(blob.get("validation", blob).get("f1_macro", float("nan")))
        except Exception:
            return float("nan")

    comparison = {
        "Majority (val)": _get_val_f1({"validation": baseline_metrics.get("majority_validation", {})}),
        "Persistence (val)": _get_val_f1({"validation": baseline_metrics.get("persistence_validation", {})}),
        "LogReg (val)": _get_val_f1(lr_results),
        "HGB (val)": _get_val_f1(hgb_results),
    }
    if mlp_pipeline is not None:
        comparison["MLP (val)"] = _get_val_f1(all_metrics["models"]["mlp"])

    for model_name, f1 in sorted(comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:30s}  macro F1 = {f1:.4f}")

    # ------------------------------------------------------------------
    # 7. Charts
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHARTS")
    print("=" * 70)

    # Confusion matrix for HGB (selected model — typically best)
    hgb_val_metrics = hgb_results.get("validation", {})
    if hgb_val_metrics.get("confusion_matrix"):
        plot_confusion_matrix("HGB", hgb_val_metrics, labels, "validation")
    hgb_test_metrics = hgb_results.get("test", {})
    if hgb_test_metrics.get("confusion_matrix"):
        plot_confusion_matrix("HGB", hgb_test_metrics, labels, "test")

    # LR confusion matrix
    lr_val_metrics = lr_results.get("validation", {})
    if lr_val_metrics.get("confusion_matrix"):
        plot_confusion_matrix("LR", lr_val_metrics, labels, "validation")

    # HGB validation curve (macro F1 vs iterations)
    # NOTE: a training loss curve is not meaningful for HistGradientBoosting
    # because it optimises a custom objective over leaves, not a tracked loss.
    print(
        "  Note: HistGradientBoostingClassifier does not expose a per-epoch "
        "training loss. Validation macro F1 vs max_iter is plotted instead."
    )
    plot_hgb_val_curve(hgb_pipeline, data)

    # LR: loss curves not meaningful (SAGA converges to full log-loss minimum;
    # convergence info is in n_iter_, not a per-step loss).
    print(
        "  Note: LogisticRegression with SAGA solver does not expose a "
        "per-iteration loss curve. A validation macro F1 vs C sensitivity "
        "curve is plotted instead."
    )
    plot_lr_C_curve(lr_pipeline, data)

    # MLP loss curve (if MLP was included)
    if mlp_pipeline is not None:
        from sklearn.base import clone as sk_clone
        fresh_mlp = sk_clone(mlp_pipeline)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh_mlp.fit(data.X_train_raw, data.y_train)
        plot_mlp_loss_curve(fresh_mlp)

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    metrics_path = RESULTS_DIR / "classification_metrics.json"
    save_metrics(all_metrics, metrics_path)
    print(f"  Metrics  → {metrics_path.relative_to(PROJECT_ROOT)}")

    hyperparams_path = RESULTS_DIR / "classification_hyperparameters.json"
    save_metrics({"tuning": tuning_records}, hyperparams_path)
    print(f"  Params   → {hyperparams_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Classification script complete.")
    print(
        "Test touched once per final model. "
        "No further tuning after this point."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
