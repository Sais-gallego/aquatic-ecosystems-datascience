"""Unit tests for Aquatic Buoy sensor telemetry and encoding."""

import unittest
from src.sensor_telemetry import AquaticBuoyNode


class TestSensorTelemetry(unittest.TestCase):

    def setUp(self):
        self.buoy = AquaticBuoyNode(station_id=101, lat=60.1699, lon=24.9384, battery_mah=5000.0)

    def test_buoy_simulation_step(self):
        reading = self.buoy.step_environment(time_hours=12.0, base_temp_c=18.0)
        self.assertEqual(reading["station_id"], 101)
        self.assertGreater(reading["temp_c"], 10.0)
        self.assertGreater(reading["do_mg_l"], 0.0)
        self.assertGreater(reading["chla_ug_l"], 0.0)
        self.assertLessEqual(reading["battery_pct"], 100.0)

    def test_biofouling_accumulation(self):
        self.assertEqual(self.buoy.biofouling_factor, 0.0)
        for hr in range(50):
            reading = self.buoy.step_environment(time_hours=float(hr), biofouling_rate=0.01)
        self.assertGreater(self.buoy.biofouling_factor, 0.4)
        self.assertGreater(reading["biofouling_pct"], 40.0)

    def test_lorawan_binary_codec(self):
        reading = {
            "station_id": 402,
            "temp_c": 17.85,
            "do_mg_l": 9.42,
            "chla_ug_l": 14.30,
            "battery_pct": 85.0,
            "biofouling_pct": 12.0
        }
        payload = self.buoy.encode_lorawan_payload(reading)
        # Exactly 10 bytes: uint16 + int16 + uint16 + uint16 + uint8 + uint8
        self.assertEqual(len(payload), 10)

        decoded = AquaticBuoyNode.decode_lorawan_payload(payload)
        self.assertEqual(decoded["station_id"], 402)
        self.assertAlmostEqual(decoded["temp_c"], 17.85, delta=0.02)
        self.assertAlmostEqual(decoded["do_mg_l"], 9.42, delta=0.02)
        self.assertAlmostEqual(decoded["chla_ug_l"], 14.30, delta=0.02)
        self.assertAlmostEqual(decoded["battery_pct"], 85.0, delta=0.5)
        self.assertFalse(decoded["biofouling_alert"])
        self.assertFalse(decoded["low_battery_alert"])


if __name__ == "__main__":
    unittest.main()
