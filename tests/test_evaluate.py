"""Unit tests for the shared metric helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from src.evaluate import (
    align_proba,
    classification_metrics,
    majority_class_predictions,
    persistence_class_predictions,
    regression_metrics,
    save_metrics,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_perfect_predictions_have_macro_f1_one(self) -> None:
        y = np.array([0, 1, 2, 2, 1])
        names = ["good", "bad", "worst"]
        metrics = classification_metrics(
            y,
            y,
            labels=[0, 1, 2],
            class_names=names,
        )
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["class_names"], names)

    def test_scores_require_labels(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(
                [0, 1],
                [0, 1],
                y_score=np.array([[1.0, 0.0], [0.0, 1.0]]),
            )

    def test_score_columns_must_match_labels(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(
                [0, 1, 2],
                [0, 1, 2],
                y_score=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                labels=[0, 1, 2],
            )

    def test_valid_probabilities_compute_auc(self) -> None:
        metrics = classification_metrics(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            y_score=np.array(
                [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
            ),
            labels=[0, 1],
        )
        self.assertAlmostEqual(metrics["roc_auc_ovr_macro"], 1.0)
        self.assertIsNone(metrics["roc_auc_note"])

    def test_probability_rows_within_tolerance_are_renormalized(self) -> None:
        scores = np.array(
            [[0.90005, 0.10005], [0.20005, 0.80005],
             [0.80005, 0.20005], [0.30005, 0.70005]]
        )
        metrics = classification_metrics(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            y_score=scores,
            labels=[0, 1],
        )
        self.assertAlmostEqual(metrics["roc_auc_ovr_macro"], 1.0)
        self.assertIsNone(metrics["roc_auc_note"])

    def test_probability_rows_outside_tolerance_raise(self) -> None:
        scores = np.array([[0.9, 0.1002], [0.2, 0.8]])
        with self.assertRaises(ValueError):
            classification_metrics(
                [0, 1],
                [0, 1],
                y_score=scores,
                labels=[0, 1],
            )

    def test_undefined_auc_is_reported_not_hidden(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        scores = np.tile(np.array([0.4, 0.4, 0.2]), (2, 1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = classification_metrics(
                y_true,
                y_pred,
                y_score=scores,
                labels=[0, 1, 2],
                class_names=["a", "b", "c"],
            )
        self.assertIsNone(metrics["roc_auc_ovr_macro"])
        self.assertIsNotNone(metrics["roc_auc_note"])
        self.assertTrue(any("AUC-ROC not computed" in str(w.message) for w in caught))

    def test_majority_baseline_accuracy_equals_majority_share(self) -> None:
        y_train = np.array([0, 0, 0, 1])
        y_eval = np.array([0, 0, 0, 1])
        predictions = majority_class_predictions(y_train, len(y_eval))
        metrics = classification_metrics(
            y_eval,
            predictions,
            labels=[0, 1],
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_persistence_applies_label_merge(self) -> None:
        labels = ["Good/Moderate", "Unhealthy"]
        predictions = persistence_class_predictions(
            ["Good", "Moderate", "Unhealthy", None],
            labels,
            label_merge={"Good": "Good/Moderate", "Moderate": "Good/Moderate"},
            fallback=0,
        )
        np.testing.assert_array_equal(predictions, np.array([0, 0, 1, 0]))

    def test_align_proba_places_columns_in_fixed_order(self) -> None:
        proba = np.array([[0.2, 0.8], [0.6, 0.4]])
        aligned = align_proba(proba, model_classes=[2, 0], n_classes=3)
        expected = np.array([[0.8, 0.0, 0.2], [0.4, 0.0, 0.6]])
        np.testing.assert_allclose(aligned, expected)


class RegressionMetricTests(unittest.TestCase):
    def test_nan_predictions_raise_by_default(self) -> None:
        with self.assertRaises(ValueError):
            regression_metrics([1.0, 2.0], [1.0, np.nan])

    def test_persistence_missing_predictions_can_be_dropped(self) -> None:
        metrics = regression_metrics(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, np.nan],
            allow_missing=True,
        )
        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["n_dropped"], 1)
        self.assertAlmostEqual(metrics["rmse"], 0.0)

    def test_nonfinite_targets_always_raise(self) -> None:
        with self.assertRaises(ValueError):
            regression_metrics([1.0, np.nan], [1.0, 2.0], allow_missing=True)

    def test_persistence_skill_is_added_when_requested(self) -> None:
        metrics = regression_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            persistence_rmse=2.0,
        )
        self.assertAlmostEqual(metrics["rmse_skill_vs_persistence"], 1.0)


class JsonTests(unittest.TestCase):
    def test_save_metrics_converts_nonfinite_values_to_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_metrics(
                {"inf": np.inf, "nan": np.nan, "ok": 1},
                Path(directory) / "metrics.json",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(payload["inf"])
        self.assertIsNone(payload["nan"])
        self.assertEqual(payload["ok"], 1)


if __name__ == "__main__":
    unittest.main()
