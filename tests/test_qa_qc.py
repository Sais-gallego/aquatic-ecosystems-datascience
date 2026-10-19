"""Unit tests for QARTOD QA/QC Pipeline."""

import unittest
from src.qa_qc_pipeline import QartodPipeline, QartodFlag


class TestQartodPipeline(unittest.TestCase):

    def test_gross_range(self):
        self.assertEqual(QartodPipeline.test_gross_range("temp_c", 21.5), QartodFlag.PASS)
        # Unrealistic water temperature
        self.assertEqual(QartodPipeline.test_gross_range("temp_c", 75.0), QartodFlag.FAIL)
        self.assertEqual(QartodPipeline.test_gross_range("temp_c", -10.0), QartodFlag.FAIL)
        # Missing
        self.assertEqual(QartodPipeline.test_gross_range("temp_c", None), QartodFlag.MISSING)

    def test_spike_detection(self):
        values = [18.0, 18.2, 18.1, 38.0, 18.3]
        # Index 1 is normal
        flag_normal = QartodPipeline.test_spike(values, 1, spike_threshold=5.0)
        # Index 3 is a spike
        flag_spike = QartodPipeline.test_spike(values, 3, spike_threshold=5.0)
        self.assertEqual(flag_normal, QartodFlag.PASS)
        self.assertEqual(flag_spike, QartodFlag.FAIL)

    def test_rate_of_change(self):
        self.assertEqual(QartodPipeline.test_rate_of_change("temp_c", 18.0, 19.0), QartodFlag.PASS)
        # 12 °C shift in 1 hour is suspect
        self.assertEqual(QartodPipeline.test_rate_of_change("temp_c", 18.0, 30.0), QartodFlag.SUSPECT)

    def test_flatline_detection(self):
        frozen_sensor = [15.2001, 15.2001, 15.2001, 15.2001]
        active_sensor = [15.2, 15.4, 15.8, 16.1]
        self.assertEqual(QartodPipeline.test_flatline(frozen_sensor), QartodFlag.FAIL)
        self.assertEqual(QartodPipeline.test_flatline(active_sensor), QartodFlag.PASS)

    def test_linear_imputation(self):
        raw = [10.0, None, None, 13.0, 14.0]
        imputed = QartodPipeline.impute_linear(raw)
        self.assertEqual(len(imputed), 5)
        self.assertAlmostEqual(imputed[1], 11.0, delta=0.01)
        self.assertAlmostEqual(imputed[2], 12.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
