"""Unit tests for feature engineering."""

import unittest
from src.feature_engineering import AquaticFeatureExtractor


class TestFeatureEngineering(unittest.TestCase):

    def test_cyclical_time_features(self):
        f_midday = AquaticFeatureExtractor.extract_temporal_cyclical_features(hour_of_day=12.0)
        # sin(pi) ~ 0, cos(pi) = -1
        self.assertAlmostEqual(f_midday["diurnal_sin"], 0.0, delta=0.01)
        self.assertAlmostEqual(f_midday["diurnal_cos"], -1.0, delta=0.01)

        f_midnight = AquaticFeatureExtractor.extract_temporal_cyclical_features(hour_of_day=0.0)
        self.assertAlmostEqual(f_midnight["diurnal_sin"], 0.0, delta=0.01)
        self.assertAlmostEqual(f_midnight["diurnal_cos"], 1.0, delta=0.01)

    def test_apparent_oxygen_deficit(self):
        # Temp = 20 C, measured = 7.0 mg/L (deficit > 0)
        aod = AquaticFeatureExtractor.compute_apparent_oxygen_deficit(temp_c=20.0, measured_do=7.0)
        self.assertGreater(aod, 1.5)

    def test_rolling_statistics(self):
        vals = [10.0, 12.0, 14.0, 16.0, 18.0]
        stats = AquaticFeatureExtractor.compute_rolling_statistics(vals, window_size=3)
        self.assertEqual(len(stats), 5)
        # Last window is [14.0, 16.0, 18.0], mean = 16.0
        self.assertAlmostEqual(stats[-1]["rolling_mean"], 16.0, delta=0.01)
        self.assertEqual(stats[-1]["rolling_min"], 14.0)
        self.assertEqual(stats[-1]["rolling_max"], 18.0)

    def test_build_ml_feature_matrix(self):
        dummy_series = [
            {"timestamp_hr": float(h), "temp_c": 18.0 + 0.1 * h, "do_mg_l": 8.5, "turbidity_fnu": 2.0, "chla_ug_l": 5.0}
            for h in range(20)
        ]
        X, y, f_names = AquaticFeatureExtractor.build_ml_feature_matrix(dummy_series, "chla_ug_l")
        self.assertEqual(len(X), 19)
        self.assertEqual(len(y), 19)
        self.assertEqual(len(f_names), 8)
        self.assertEqual(len(X[0]), 8)


if __name__ == "__main__":
    unittest.main()
