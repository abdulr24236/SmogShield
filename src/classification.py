"""Next-day AQI classification — Deliverable 2.

Owner: Muhammad Haider Mughal

Run from the repository root:
    python -m src.classification

Outputs (all git-ignored; regenerate from this script + shared data):
    outputs/model_results/classification_metrics.json
    outputs/model_results/classification_hyperparameters.json
    outputs/model_results/classification_predictions.csv
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

import time
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
from sklearn.utils.validation import check_is_fitted

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
_SEED = int(load_config()["seed"])


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


def _compute_val_macro_f1(pipeline: Pipeline, data: Any) -> float:
    """Compute validation macro F1 on the scorable validation rows using classification_metrics."""
    mask = data.scorable_rows["validation"]
    y_pred_full = pipeline.predict(data.X_validation_raw)
    y_true_s, y_pred_s = _scorable_subset(mask, data.y_validation, y_pred_full)
    metrics = classification_metrics(
        y_true=y_true_s,
        y_pred=y_pred_s,
        labels=_int_labels(data.labels),
        class_names=data.labels,
    )
    return float(metrics["f1_macro"])


# ===========================================================================
# Baselines
# ===========================================================================


def get_persistence_fallback(data: Any) -> tuple[int, str]:
    """Determine majority-class fallback code and name dynamically from training labels."""
    y_train_arr = data.y_train.to_numpy()
    values, counts = np.unique(y_train_arr, return_counts=True)
    fallback_code = int(values[int(np.argmax(counts))])
    fallback_name = str(data.labels[fallback_code])
    return fallback_code, fallback_name


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

    metrics = _score_model(
        f"Majority baseline [{split}]",
        y_true_full,
        y_pred_full,
        mask,
        data.labels,
    )
    metrics["roc_auc_ovr_macro"] = None
    metrics["roc_auc_note"] = "no probabilities"
    return metrics


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

    Fallback: the training majority class (computed dynamically from y_train).
    The persistence baseline uses the raw 'aqi_category' column from X_*_raw;
    it is not allowed to look at the encoded target. Missing-category rows
    get the majority-class code as fallback.
    """
    fallback_code, fallback_name = get_persistence_fallback(data)

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

    metrics = _score_model(
        f"Persistence baseline [{split}]",
        y_true_full,
        y_pred_full,
        mask,
        data.labels,
    )
    metrics["roc_auc_ovr_macro"] = None
    metrics["roc_auc_note"] = "no probabilities"
    return metrics


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

    Selection metric: macro F1 only.

    Returns
    -------
    best_pipeline:
        The refitted best pipeline (fitted on the full training raw data).
    tuning_info:
        A dict recording the search space, selected values, CV folds,
        selection metric, best CV score, and validation macro F1.
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
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        search.fit(data.X_train_raw, data.y_train)

    best_params = {
        # Strip the "model__" prefix added by Pipeline for readability.
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    best_score = float(search.best_score_)

    # Compute validation macro F1 consistently using classification_metrics
    val_f1 = _compute_val_macro_f1(search.best_estimator_, data)

    tuning_info: dict[str, Any] = {
        "model": name,
        "hyperparameter_configuration": best_params,
        "selected_params": best_params,
        "validation_f1_macro": round(val_f1, 6),
        "selected": False,
        "tuning_status": "completed",
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
        "n_iter": n_iter,
    }

    print(f"  Best CV macro F1       : {best_score:.4f}")
    print(f"  Validation macro F1    : {val_f1:.4f}")
    print(f"  Selected params        : {best_params}")

    return search.best_estimator_, tuning_info


# ===========================================================================
# Logistic regression
# ===========================================================================


def build_and_tune_logistic(
    data: Any, cv: TimeSeriesSplit
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune a regularised logistic regression on the training split.

    Search space (§1.4):
        C, penalty, class_weight, max_iter (max_iter=5000 removed for runtime budget)
    """
    param_distributions = {
        "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
        "model__max_iter": [500, 1000, 2000],
    }

    estimator = LogisticRegression(
        solver="saga",  # supports both l1 and l2
        random_state=_SEED,
    )

    return _tune_model(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=12,
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
        n_iter=16,
        cv=cv,
        name="HistGradientBoosting",
    )


# ===========================================================================
# MLP
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
        n_iter=8,
        cv=cv,
        name="MLP",
    )


# ===========================================================================
# Model Selection Helper
# ===========================================================================


def select_best_model(
    candidates: list[tuple[str, Pipeline, dict[str, Any]]]
) -> tuple[str, Pipeline, dict[str, Any]]:
    """Select the model with the highest validation macro F1.

    Deterministic tie-breaking rule: if validation macro F1 scores are identical,
    break the tie by alphabetical order of the model name.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-c[2]["validation_f1_macro"], c[0]),
    )
    return sorted_candidates[0]


# ===========================================================================
# Final model evaluation (validation + test)
# ===========================================================================


def evaluate_final(
    name: str,
    pipeline: Pipeline,
    data: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Score validation and test using the fitted pipeline without redundant refitting.

    Parameters
    ----------
    name:
        Display name for logging.
    pipeline:
        Fitted pipeline (from tuning or fresh clone).
    data:
        ModelData container.

    Returns
    -------
    results:
        Dict with keys 'validation' and 'test', each containing classification_metrics.
    predictions:
        Dict with keys 'validation' and 'test', each containing:
        - 'y_pred_full': 1D array of predicted class integer codes for all split rows.
        - 'y_score_full': 2D array of aligned probabilities for all split rows.
    """
    print(f"\n[Final] Evaluating {name} on validation and test splits …")

    try:
        check_is_fitted(pipeline)
        model_pipe = pipeline
    except Exception:
        from sklearn.base import clone as sk_clone
        model_pipe = sk_clone(pipeline)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            model_pipe.fit(data.X_train_raw, data.y_train)

    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, Any]] = {}

    for split_name in ("validation", "test"):
        if split_name == "validation":
            X_raw = data.X_validation_raw
            y_true_full = data.y_validation
        else:
            X_raw = data.X_test_raw
            y_true_full = data.y_test

        mask = data.scorable_rows[split_name]

        # Predictions
        y_pred_full = model_pipe.predict(X_raw)

        # Probabilities — align to fixed class-code order
        estimator_step = model_pipe.named_steps["model"]
        if hasattr(estimator_step, "predict_proba"):
            proba_raw = model_pipe.predict_proba(X_raw)
            y_score_full = align_proba(
                proba_raw,
                model_pipe.classes_,
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
        predictions[split_name] = {
            "y_pred_full": y_pred_full,
            "y_score_full": y_score_full,
        }

    return results, predictions


# ===========================================================================
# Predictions CSV Generator
# ===========================================================================


def generate_predictions_csv(
    selected_name: str,
    selected_preds: dict[str, dict[str, Any]],
    data: Any,
    output_path: Path,
) -> pd.DataFrame:
    """Generate per-row predictions CSV for the selected model on validation and test splits.

    Contains exactly:
    date, split, model, y_true_code, y_pred_code,
    proba_0, proba_1, proba_2, proba_3, proba_4, proba_5, scorable
    """
    assert len(data.labels) == 6, f"Expected 6 classes, got {len(data.labels)}"
    dfs = []
    for split_name in ("validation", "test"):
        if split_name == "validation":
            split_df = data.split.validation
            y_true_series = data.y_validation
        else:
            split_df = data.split.test
            y_true_series = data.y_test

        date_col = data.split.date_column
        dates = pd.to_datetime(split_df[date_col]).dt.strftime("%Y-%m-%d").to_numpy()
        scorable_mask = data.scorable_rows[split_name].to_numpy(dtype=bool)

        y_true_codes = y_true_series.to_numpy(dtype=int)
        y_pred_codes = np.asarray(selected_preds[split_name]["y_pred_full"], dtype=int)
        proba_matrix = selected_preds[split_name]["y_score_full"]

        n_rows = len(dates)
        split_records = {
            "date": dates,
            "split": [split_name] * n_rows,
            "model": [selected_name] * n_rows,
            "y_true_code": y_true_codes,
            "y_pred_code": y_pred_codes,
            "proba_0": proba_matrix[:, 0] if proba_matrix is not None else [np.nan] * n_rows,
            "proba_1": proba_matrix[:, 1] if proba_matrix is not None else [np.nan] * n_rows,
            "proba_2": proba_matrix[:, 2] if proba_matrix is not None else [np.nan] * n_rows,
            "proba_3": proba_matrix[:, 3] if proba_matrix is not None else [np.nan] * n_rows,
            "proba_4": proba_matrix[:, 4] if proba_matrix is not None else [np.nan] * n_rows,
            "proba_5": proba_matrix[:, 5] if proba_matrix is not None else [np.nan] * n_rows,
            "scorable": scorable_mask,
        }
        dfs.append(pd.DataFrame(split_records))

    predictions_df = pd.concat(dfs, ignore_index=True)

    col_order = [
        "date", "split", "model", "y_true_code", "y_pred_code",
        "proba_0", "proba_1", "proba_2", "proba_3", "proba_4", "proba_5",
        "scorable",
    ]
    predictions_df = predictions_df[col_order]

    expected_rows = (
        data.summary()["validation_rows"]
        + data.summary()["test_rows"]
    )
    assert len(predictions_df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(predictions_df)}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"  Saved predictions CSV → {output_path.relative_to(PROJECT_ROOT)}")
    return predictions_df


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
    is_default: bool = False,
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
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path.relative_to(PROJECT_ROOT)}")
    if is_default:
        default_path = CHARTS_DIR / "classification_confusion_matrix.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {default_path.relative_to(PROJECT_ROOT)}")
    plt.close()


def plot_hgb_val_curve(
    pipeline: Pipeline,
    data: Any,
    name: str = "HistGradientBoosting",
) -> None:
    """Plot macro F1 vs number of boosting iterations on the validation set.

    The pipeline must have a 'model' step that is a
    HistGradientBoostingClassifier. The curve is generated by refitting
    models with increasing max_iter (HistGradientBoosting does not support
    warm_start).
    """
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
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            pipe_i.fit(data.X_train_raw, data.y_train)
        y_pred = pipe_i.predict(data.X_validation_raw)
        y_t, y_p = _scorable_subset(mask_val, data.y_validation, y_pred)
        f1 = float(
            classification_metrics(
                y_true=y_t,
                y_pred=y_p,
                labels=_int_labels(data.labels),
                class_names=data.labels,
            )["f1_macro"]
        )
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
) -> dict[str, Any]:
    """Plot validation macro F1 vs regularisation strength C for LogReg.

    Note: Loss curves are not meaningful for logistic regression trained with
    full-batch solvers; we report a hyperparameter sensitivity curve instead.
    Evaluates approximately 6 C-values for runtime efficiency.
    """
    from sklearn.base import clone as sk_clone

    lr_step = pipeline.named_steps["model"]
    if not isinstance(lr_step, LogisticRegression):
        print("  [Skip] LR C-curve: model step is not LogisticRegression.")
        return {}

    best_params = lr_step.get_params()
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mask_val = data.scorable_rows["validation"]
    scores = []

    for C in C_values:
        params_i = {**best_params, "C": C}
        lr_i = LogisticRegression(**params_i)
        pipe_i = sk_clone(pipeline)
        pipe_i.set_params(model=lr_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            pipe_i.fit(data.X_train_raw, data.y_train)
        y_pred = pipe_i.predict(data.X_validation_raw)
        y_t, y_p = _scorable_subset(mask_val, data.y_validation, y_pred)
        f1 = float(
            classification_metrics(
                y_true=y_t,
                y_pred=y_p,
                labels=_int_labels(data.labels),
                class_names=data.labels,
            )["f1_macro"]
        )
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

    return {
        "C_values": C_values,
        "validation_f1_macro": [round(s, 6) for s in scores],
        "selected_C": float(best_params.get("C", 1.0)),
    }


def plot_mlp_loss_curve(pipeline: Pipeline) -> None:
    """Plot training loss curve from MLP's loss_curve_ attribute without refitting."""
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
    start_time = time.perf_counter()

    print("=" * 70)
    print("SmogShield — Next-day AQI classification")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load shared data
    # ------------------------------------------------------------------
    config = load_config()
    data = prepare_model_data(config, "classification")

    labels: list[str] = data.labels  # type: ignore[assignment]
    n_classes = len(labels)

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
    # 2. Baselines (train, validation, and test)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    fallback_code, fallback_name = get_persistence_fallback(data)
    print(
        f"\n[Persistence baseline] fallback for missing current category: "
        f"'{fallback_name}' (code {fallback_code})"
    )

    baseline_metrics: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
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

    # Logistic Regression (budget: 12)
    lr_pipeline, lr_tuning = build_and_tune_logistic(data, cv)
    tuning_records.append(lr_tuning)

    # HistGradientBoosting (budget: 16)
    hgb_pipeline, hgb_tuning = build_and_tune_hgb(data, cv)
    tuning_records.append(hgb_tuning)

    # ------------------------------------------------------------------
    # 4. MLP tuning (budget: 8)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MLP TUNING (budget: 8)")
    print("=" * 70)

    hgb_val_f1 = hgb_tuning["validation_f1_macro"]
    mlp_pipeline: Pipeline | None = None
    mlp_tuning: dict[str, Any] | None = None
    mlp_val_f1 = 0.0
    mlp_failure: str | None = None

    try:
        mlp_pipeline_cand, mlp_tuning_cand = build_and_tune_mlp(data, cv)
        mlp_val_f1 = mlp_tuning_cand["validation_f1_macro"]
        print(f"  HGB validation macro F1: {hgb_val_f1:.4f}")
        print(f"  MLP validation macro F1: {mlp_val_f1:.4f}")

        mlp_tuning = mlp_tuning_cand
        tuning_records.append(mlp_tuning)

        if mlp_val_f1 > hgb_val_f1:
            print("  → MLP included (beats HGB on validation macro F1).")
            mlp_pipeline = mlp_pipeline_cand
            mlp_tuning["tuning_status"] = "completed"
        else:
            print("  → MLP excluded (does not improve over HGB).")
            mlp_tuning["tuning_status"] = "excluded"
            mlp_tuning["selected"] = False
    except Exception as exc:
        mlp_failure = str(exc)
        print(f"  [Warning] MLP tuning failed: {exc}. Skipping MLP.")
        tuning_records.append({
            "model": "MLP",
            "tuning_status": "failed",
            "selected": False,
            "validation_f1_macro": None,
            "error": str(exc),
        })

    # ------------------------------------------------------------------
    # 5. Model Selection (based on validation macro F1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL SELECTION (based on validation macro F1)")
    print("=" * 70)

    eligible_models = [
        ("HistGradientBoosting", hgb_pipeline, hgb_tuning),
        ("LogisticRegression", lr_pipeline, lr_tuning),
    ]
    if mlp_pipeline is not None and mlp_tuning is not None:
        eligible_models.append(("MLP", mlp_pipeline, mlp_tuning))

    selected_name, selected_pipeline, selected_tuning = select_best_model(eligible_models)

    # Mark selection status in tuning records
    for rec in tuning_records:
        rec["selected"] = (rec["model"] == selected_name)
        if rec["selected"]:
            rec["tuning_status"] = "selected"

    selected_val_f1 = selected_tuning["validation_f1_macro"]
    print(f"\n  Eligible model validation macro F1 scores:")
    for model_n, _, t_rec in eligible_models:
        print(f"    - {model_n:25s}: {t_rec['validation_f1_macro']:.4f}")
    print(f"  >>> Selected model: {selected_name} (val macro F1 = {selected_val_f1:.4f})")

    # ------------------------------------------------------------------
    # 6. Final Model Evaluation (validation and test splits; test touched once)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (validation and test splits; test touched once)")
    print("=" * 70)

    all_metrics: dict[str, Any] = {
        "selected_model": selected_name,
        "selected_model_val_f1_macro": selected_val_f1,
        "persistence_fallback_class": fallback_name,
        "persistence_fallback_code": fallback_code,
        "cv_validation_f1_caveat": (
            "best_cv_f1_macro is computed per fold over the classes present in that fold; "
            "validation_f1_macro uses all six configured labels. "
            "The two values are not directly comparable."
        ),
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

    # Evaluate Logistic Regression
    lr_results, lr_preds = evaluate_final("LogisticRegression", lr_pipeline, data)
    all_metrics["models"]["logistic_regression"] = lr_results

    # Evaluate HistGradientBoosting
    hgb_results, hgb_preds = evaluate_final("HistGradientBoosting", hgb_pipeline, data)
    all_metrics["models"]["hgb"] = hgb_results

    # Evaluate MLP if included
    mlp_results = None
    mlp_preds = None
    if mlp_pipeline is not None:
        mlp_results, mlp_preds = evaluate_final("MLP", mlp_pipeline, data)
        all_metrics["models"]["mlp"] = mlp_results
    elif mlp_failure is not None:
        all_metrics["models"]["mlp_note"] = "MLP tuning failed"
    else:
        all_metrics["models"]["mlp_note"] = (
            f"MLP excluded: val macro F1 ({mlp_val_f1:.4f}) "
            f"did not exceed HGB ({hgb_tuning['validation_f1_macro']:.4f})."
        )

    # Reference predictions and results for the selected model
    if selected_name == "HistGradientBoosting":
        selected_preds = hgb_preds
        selected_results = hgb_results
    elif selected_name == "LogisticRegression":
        selected_preds = lr_preds
        selected_results = lr_results
    else:
        selected_preds = mlp_preds
        selected_results = mlp_results

    all_metrics["selected_model_metrics"] = {
        "model": selected_name,
        "validation": selected_results["validation"],
        "test": selected_results["test"],
    }

    # ------------------------------------------------------------------
    # 7. Generate Per-Row Predictions CSV (Deliverable for Ayub's bias analysis)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-ROW PREDICTIONS CSV")
    print("=" * 70)

    predictions_csv_path = RESULTS_DIR / "classification_predictions.csv"
    predictions_df = generate_predictions_csv(
        selected_name=selected_name,
        selected_preds=selected_preds,
        data=data,
        output_path=predictions_csv_path,
    )

    # ------------------------------------------------------------------
    # 8. Summary comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY — Validation and Test Macro F1")
    print("=" * 70)

    def _get_f1(metrics_dict: dict[str, Any], split: str) -> float:
        try:
            return float(metrics_dict[split]["f1_macro"])
        except Exception:
            return float("nan")

    print(f"  {'Model':30s} | {'Val Macro F1':>13s} | {'Test Macro F1':>13s}")
    print("  " + "-" * 62)
    print(f"  {'Logistic Regression':30s} | {_get_f1(lr_results, 'validation'):13.4f} | {_get_f1(lr_results, 'test'):13.4f}")
    print(f"  {'HistGradientBoosting':30s} | {_get_f1(hgb_results, 'validation'):13.4f} | {_get_f1(hgb_results, 'test'):13.4f}")
    if mlp_results is not None:
        print(f"  {'MLP':30s} | {_get_f1(mlp_results, 'validation'):13.4f} | {_get_f1(mlp_results, 'test'):13.4f}")
    elif mlp_failure is not None:
        print(f"  {'MLP (failed)':30s} | {'failed':>13s} | {'N/A (failed)':>13s}")
    else:
        print(f"  {'MLP (excluded)':30s} | {mlp_val_f1:13.4f} | {'N/A (excluded)':>13s}")
    print(f"  {'Majority baseline':30s} | {baseline_metrics['majority_validation']['f1_macro']:13.4f} | {baseline_metrics['majority_test']['f1_macro']:13.4f}")
    print(f"  {'Persistence baseline':30s} | {baseline_metrics['persistence_validation']['f1_macro']:13.4f} | {baseline_metrics['persistence_test']['f1_macro']:13.4f}")
    print("  " + "-" * 62)
    print(f"  Selected Model: {selected_name} (Val Macro F1 = {selected_val_f1:.4f})")

    # ------------------------------------------------------------------
    # 9. Charts
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHARTS")
    print("=" * 70)

    # Confusion matrix for selected model
    sel_val_cm = selected_results.get("validation", {})
    if sel_val_cm.get("confusion_matrix"):
        plot_confusion_matrix(selected_name, sel_val_cm, labels, "validation")
    sel_test_cm = selected_results.get("test", {})
    if sel_test_cm.get("confusion_matrix"):
        plot_confusion_matrix(selected_name, sel_test_cm, labels, "test", is_default=True)

    # Confusion matrix for Logistic Regression if not selected
    if selected_name != "LogisticRegression":
        lr_val_cm = lr_results.get("validation", {})
        if lr_val_cm.get("confusion_matrix"):
            plot_confusion_matrix("LogisticRegression", lr_val_cm, labels, "validation")

    # HGB validation curve (macro F1 vs iterations)
    plot_hgb_val_curve(hgb_pipeline, data)

    # LR C-curve (6 points)
    c_curve_data = plot_lr_C_curve(lr_pipeline, data)
    all_metrics["c_curve_results"] = c_curve_data

    # MLP loss curve (if MLP was included)
    if mlp_pipeline is not None:
        plot_mlp_loss_curve(mlp_pipeline)

    # ------------------------------------------------------------------
    # 10. Save outputs
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
    # 11. Output validation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)
    expected_rows = summary["validation_rows"] + summary["test_rows"]
    actual_rows = len(predictions_df)
    val_rows = int((predictions_df["split"] == "validation").sum())
    test_rows = int((predictions_df["split"] == "test").sum())
    scorable_rows = int(predictions_df["scorable"].sum())
    unscorable_rows = int((~predictions_df["scorable"]).sum())

    print(f"Expected rows: {expected_rows}")
    print(f"Actual rows: {actual_rows}")
    print(f"Validation rows: {val_rows}")
    print(f"Test rows: {test_rows}")
    print(f"Scorable rows: {scorable_rows}")
    print(f"Unscorable rows: {unscorable_rows}")
    print(f"Selected model: {selected_name}")

    total_time = time.perf_counter() - start_time
    print(f"\nTotal classification experiment runtime: {total_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
