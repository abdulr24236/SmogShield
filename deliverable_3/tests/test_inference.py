"""Smoke tests for the Deliverable 3 Track A inference path (unittest style)."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.hourly import build_hourly_profile  # noqa: E402
from src.inference import (  # noqa: E402
    GUIDANCE,
    aqi_to_category,
    guidance_for,
    inference,
    pm25_to_aqi,
)


class InferenceSmokeTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(
            inference.class_labels,
            [
                "Good/Moderate",
                "Unhealthy for Sensitive",
                "Unhealthy",
                "Very Unhealthy",
                "Hazardous (301-400)",
                "Hazardous (401-500)",
            ],
        )

    def test_feature_columns_nonempty(self):
        self.assertGreater(len(inference.feature_columns), 0)
        self.assertEqual(inference.n_features, 80)
        # A built feature row must carry exactly the raw input columns.
        row = inference.build_feature_row({"pm25": 50.0, "aqi_category": "Unhealthy"})
        self.assertEqual(list(row.columns), inference.feature_columns)
        self.assertEqual(len(row.columns), len(inference.feature_columns))

    def test_predict_returns_expected_keys(self):
        result = inference.predict(
            {
                "pm25": 80.0,
                "aqi_category": "Unhealthy",
                "temperature_c": 25.0,
                "humidity_pct": 55.0,
                "wind_kmh": 8.0,
                "date": "2025-01-10",
            }
        )
        self.assertIn("predicted_next_pm25", result)
        self.assertIn("predicted_next_aqi_category", result)
        self.assertIn("probabilities", result)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertIn("interval", result)
        self.assertIn("explanation", result)
        self.assertIn("guidance", result)
        self.assertIn("persistence", result)
        self.assertEqual(result["persistence"]["aqi_category"], "Unhealthy")
        self.assertEqual(result["persistence"]["pm25"], 80.0)
        self.assertIsInstance(result["predicted_next_pm25"], float)

    def test_pm25_breakpoints_match_project_labels(self):
        # Edge values from data_scripts/03_pm25_aqi_and_eda.py.
        cases = {
            12.0: ("Good/Moderate", 50),
            35.4: ("Good/Moderate", 100),
            35.5: ("Unhealthy for Sensitive", 101),
            55.4: ("Unhealthy for Sensitive", 150),
            55.5: ("Unhealthy", 151),
            150.4: ("Unhealthy", 200),
            150.5: ("Very Unhealthy", 201),
            250.4: ("Very Unhealthy", 300),
            250.5: ("Hazardous (301-400)", 301),
            350.4: ("Hazardous (301-400)", 400),
            350.5: ("Hazardous (401-500)", 401),
        }
        for pm25, (category, aqi) in cases.items():
            with self.subTest(pm25=pm25):
                self.assertEqual(pm25_to_aqi(pm25), aqi)
                self.assertEqual(aqi_to_category(aqi), category)

    def test_implied_category_is_monotonic_in_pm25(self):
        order = inference.class_labels  # already ordered low -> high severity
        ranks = []
        for pm25 in (10.0, 40.0, 80.0, 160.0, 300.0):
            result = inference.predict(
                {"pm25": pm25, "aqi_category": "Unhealthy", "date": "2025-01-10"}
            )
            ranks.append(order.index(result["predicted_category_from_pm25"]))
        self.assertEqual(ranks, sorted(ranks))
        self.assertGreater(ranks[-1], ranks[0])

    def test_interval_is_ordered_and_contains_point_forecast(self):
        result = inference.predict(
            {"pm25": 90.0, "aqi_category": "Unhealthy", "date": "2025-01-10"}
        )
        interval = result["interval"]
        self.assertIsNotNone(interval)
        self.assertLessEqual(interval["low"], result["predicted_next_pm25"])
        self.assertGreaterEqual(interval["high"], result["predicted_next_pm25"])
        self.assertGreaterEqual(interval["low"], 0.0)
        self.assertEqual(interval["level"], 0.8)
        # The calibration metadata must exist and be plausible.
        self.assertGreaterEqual(inference.interval_metadata["coverage_test"], 0.6)

    def test_explanation_lists_every_collected_input(self):
        result = inference.predict(
            {
                "pm25": 90.0,
                "aqi_category": "Unhealthy",
                "temperature_c": 22.0,
                "humidity_pct": 60.0,
                "wind_kmh": 7.0,
                "date": "2025-01-10",
            }
        )
        inputs = {row["input"] for row in result["explanation"]}
        self.assertEqual(len(inputs), 4)
        self.assertTrue(any("PM2.5" in label for label in inputs))
        self.assertTrue(any("Wind" in label for label in inputs))

    def test_numeric_aqi_is_derived_from_pm25(self):
        row = inference.build_feature_row(
            {"pm25": 150.0, "aqi_category": "Unhealthy", "date": "2025-01-10"}
        )
        self.assertEqual(float(row["aqi"].iloc[0]), float(pm25_to_aqi(150.0)))

    def test_pm25_sensitivity_uses_larger_step(self):
        result = inference.predict(
            {
                "pm25": 150.0,
                "aqi_category": "Unhealthy",
                "temperature_c": 22.0,
                "humidity_pct": 60.0,
                "wind_kmh": 7.0,
                "date": "2025-01-10",
            }
        )
        pm25_row = next(
            row for row in result["explanation"] if "PM2.5" in row["input"]
        )
        self.assertEqual(pm25_row["delta_input"], 25.0)

    def test_guidance_covers_every_category(self):
        for category in inference.class_labels:
            with self.subTest(category=category):
                guidance = guidance_for(category)
                self.assertIn("headline", guidance)
                self.assertIn("detail", guidance)
                self.assertIn("disclaimer", guidance)
        self.assertEqual(set(GUIDANCE), set(inference.class_labels))

    def test_global_importance_and_patterns_exist(self):
        importance = inference.feature_importance.get("regression") or []
        self.assertGreater(len(importance), 0)
        self.assertIn("feature", importance[0])
        self.assertIn("importance", importance[0])
        patterns = inference.patterns
        self.assertIn("stagnant_mean_pm25", patterns)
        self.assertGreater(patterns["stagnant_days"], 0)

    def test_monthly_medians_cover_the_year(self):
        medians = inference.monthly_median_pm25
        self.assertEqual(set(medians), {str(i) for i in range(1, 13)})
        self.assertTrue(all(value > 0 for value in medians.values()))
        # Winter smog should exceed the monsoon months.
        self.assertGreater(medians["11"], medians["7"])

    def test_hourly_profile_is_complete(self):
        profile = build_hourly_profile()
        self.assertTrue(profile)
        self.assertEqual(len(profile["by_hour"]), 24)
        self.assertGreater(profile["valid_hours"], 10000)
        window = profile["cleanest_window"]
        self.assertGreaterEqual(window["start_hour"], 0)
        self.assertLessEqual(window["end_hour"], 23)
        self.assertLess(window["mean_pm25"], profile["peak_hour"]["mean_pm25"])
        self.assertIn("winter", profile["seasonal_cleanest_window"])
        # The stored metadata must carry the same profile for the app.
        self.assertIn("hourly", inference.patterns)

    def test_predictions_respond_to_pm25(self):
        low = inference.predict(
            {"pm25": 20.0, "aqi_category": "Moderate", "date": "2025-01-10"}
        )
        high = inference.predict(
            {"pm25": 200.0, "aqi_category": "Very Unhealthy", "date": "2025-01-10"}
        )
        self.assertNotEqual(low["predicted_next_pm25"], high["predicted_next_pm25"])
        self.assertGreater(
            high["predicted_next_pm25"], low["predicted_next_pm25"]
        )
        # Probabilities must move even when the arg-max label happens not to.
        self.assertNotEqual(low["probabilities"], high["probabilities"])
