"""Shared metric functions for Deliverable 2.

Every model must use these functions so validation and test results are
comparable. Classification metrics are macro-averaged because the AQI classes
are strongly imbalanced; per-class precision/recall is also returned.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_score: np.ndarray | None = None,
    labels: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return classification metrics.

    Parameters
    ----------
    y_true, y_pred:
        Integer class codes.
    y_score:
        Optional predicted probabilities with shape ``(n_samples, n_classes)``
        (or ``(n_samples,)`` for binary problems). Required for AUC-ROC.
    labels:
        All class codes in the fixed order used by the model.
    class_names:
        Human-readable names aligned with ``labels``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(labels)
    if class_names is None:
        class_names = [str(label) for label in labels]
    if len(class_names) != len(labels):
        raise ValueError("class_names and labels must have the same length.")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    auc = None
    if y_score is not None:
        scores = np.asarray(y_score)
        try:
            if scores.ndim == 1:
                auc = float(roc_auc_score(y_true, scores))
            elif scores.shape[1] == 2:
                auc = float(roc_auc_score(y_true, scores[:, 1]))
            else:
                auc = float(
                    roc_auc_score(
                        y_true,
                        scores,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                )
        except ValueError:
            # AUC is undefined if a class is absent from y_true, for example.
            auc = None

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc_ovr_macro": auc,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "per_class": report,
    }


def majority_class_predictions(
    y_train: np.ndarray | list[int],
    n_samples: int,
) -> np.ndarray:
    """Predict the most frequent training class for every evaluation row."""
    y = np.asarray(y_train)
    if y.size == 0:
        raise ValueError("Cannot build a majority baseline from an empty y_train.")
    values, counts = np.unique(y, return_counts=True)
    majority = int(values[int(np.argmax(counts))])
    return np.full(int(n_samples), majority, dtype=int)


def persistence_class_predictions(
    current_categories: Sequence[Any],
    labels: Sequence[str],
    label_merge: Mapping[str, str] | None = None,
    fallback: int | None = None,
) -> np.ndarray:
    """Predict tomorrow's category from today's observed AQI category.

    ``fallback`` is the encoded class used when today's category is missing.
    Pass the majority-class code for a fully defined baseline, or leave it as
    ``None`` to fail loudly on missing current categories.
    """
    mapping = {str(label): index for index, label in enumerate(labels)}
    merge = {
        str(source): str(destination)
        for source, destination in (label_merge or {}).items()
    }
    predictions: list[int] = []
    for value in current_categories:
        if pd.isna(value):
            if fallback is None:
                raise ValueError(
                    "Persistence baseline found a missing current AQI category; "
                    "provide fallback=<majority-class code>."
                )
            predictions.append(int(fallback))
            continue

        key = merge.get(str(value), str(value))
        if key not in mapping:
            raise ValueError(
                f"Current category {value!r} maps to unknown label {key!r}. "
                f"Known labels: {list(mapping)}"
            )
        predictions.append(mapping[key])
    return np.asarray(predictions, dtype=int)


def persistence_regression_predictions(
    current_pm25: Sequence[float],
    fallback: float | None = None,
) -> np.ndarray:
    """Predict tomorrow's PM2.5 as today's observed PM2.5.

    If ``fallback`` is provided, it replaces missing current-day PM2.5 values.
    Otherwise missing values are returned as NaN and the metric function will
    evaluate only the finite pairs.
    """
    values = np.asarray(current_pm25, dtype=float)
    if fallback is not None:
        values = np.where(np.isfinite(values), values, float(fallback))
    return values


def regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, Any]:
    """Return RMSE, MAE and R-squared for a regression model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("No finite prediction/target pairs to evaluate.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "n_samples": int(mask.sum()),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _to_jsonable(value: Any) -> Any:
    """Convert numpy values and NaN to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write a metrics dictionary as readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def print_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Print the scalar metrics without dumping the full per-class report."""
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if key in {"per_class", "confusion_matrix"}:
            continue
        print(f"{key}: {value}")
