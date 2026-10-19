"""Unit tests for ML models and Physics-Informed algorithms."""

import unittest
from src.ml_models import ChlorophyllForecaster, AlgaeBloomClassifier, AquaticMetrics
from src.physics_informed import PhysicsInformedLakeModel


class TestMlAndPhysics(unittest.TestCase):

    def test_chlorophyll_forecaster(self):
        # Synthetic linear relationship
        X = [[1.0, float(i)] for i in range(30)]
        y = [2.0 * float(i) + 5.0 for i in range(30)]

        forecaster = ChlorophyllForecaster(l2_reg=0.1)
        forecaster.fit(X, y, epochs=100, lr=0.1)
        preds = forecaster.predict(X)

        self.assertEqual(len(preds), len(y))
        r2 = AquaticMetrics.r2_score(y, preds)
        self.assertGreater(r2, 0.85)

    def test_algae_bloom_classifier(self):
        # Samples with high Chl-a (>= 20) vs low (< 20)
        X = [[float(i), float(i)*1.5] for i in range(40)]
        y_chla = [5.0 if i < 20 else 35.0 for i in range(40)]

        classifier = AlgaeBloomClassifier(bloom_threshold_ug_l=20.0)
        classifier.fit(X, y_chla, epochs=50, lr=0.1)
        preds = classifier.predict(X)

        y_true = [1 if v >= 20.0 else 0 for v in y_chla]
        f1_metrics = AquaticMetrics.classification_f1(y_true, preds)
        self.assertGreater(f1_metrics["accuracy"], 0.80)

    def test_physics_heat_source(self):
        # Solar heating at surface vs at depth 10m
        h_0m = PhysicsInformedLakeModel.heat_source_term(surface_irradiance_w_m2=600.0, depth_m=0.0)
        h_10m = PhysicsInformedLakeModel.heat_source_term(surface_irradiance_w_m2=600.0, depth_m=10.0)
        self.assertGreater(h_0m, 0.0)
        self.assertGreater(h_0m, h_10m)

    def test_physics_pde_loss(self):
        # Simulated grid of 5 time steps, 4 depth levels
        temps = [
            [22.0 - 1.5 * z for z in range(4)]
            for _ in range(5)
        ]
        loss = PhysicsInformedLakeModel.evaluate_physics_loss(
            temps, dt_seconds=3600.0, dz_meters=1.0, surface_irradiance_w_m2=400.0
        )
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
