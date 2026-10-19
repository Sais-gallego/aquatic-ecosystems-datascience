"""Unit tests for limnological physics and domain foundations."""

import unittest
from src.domain_foundations import (
    dissolved_oxygen_saturation,
    water_density,
    carlson_trophic_state_index,
    find_thermocline,
    brunt_vaisala_frequency_squared
)


class TestDomainFoundations(unittest.TestCase):

    def test_do_saturation_freshwater(self):
        # At 20 °C and 0 PSU salinity, DO saturation is ~9.09 mg/L
        do_sat_20c = dissolved_oxygen_saturation(20.0, 0.0)
        self.assertAlmostEqual(do_sat_20c, 9.09, delta=0.2)

        # Colder water holds more oxygen
        do_sat_4c = dissolved_oxygen_saturation(4.0, 0.0)
        self.assertGreater(do_sat_4c, do_sat_20c)
        self.assertAlmostEqual(do_sat_4c, 13.1, delta=0.3)

    def test_water_density(self):
        # Maximum density of freshwater occurs around 3.98 °C (~1000 kg/m^3)
        rho_4c = water_density(4.0, 0.0)
        rho_25c = water_density(25.0, 0.0)
        self.assertAlmostEqual(rho_4c, 1000.0, delta=0.5)
        self.assertLess(rho_25c, rho_4c)

        # Seawater (35 PSU) is denser
        rho_sea = water_density(20.0, 35.0)
        self.assertGreater(rho_sea, 1020.0)

    def test_carlson_trophic_state_index(self):
        # Low chlorophyll-a = Oligotrophic
        res_oligo = carlson_trophic_state_index(chla_ug_l=1.0, secchi_m=6.0)
        self.assertEqual(res_oligo["classification"], "Oligotrophic")
        self.assertLess(res_oligo["composite_tsi"], 40.0)

        # High chlorophyll-a = Eutrophic / Hypereutrophic
        res_eutro = carlson_trophic_state_index(chla_ug_l=40.0, secchi_m=0.8)
        self.assertIn(res_eutro["classification"], ["Eutrophic", "Hypereutrophic"])
        self.assertGreater(res_eutro["composite_tsi"], 60.0)

    def test_thermocline_detection(self):
        depths = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        # Sharp drop between 4m and 6m
        temps = [22.0, 21.5, 20.0, 14.0, 10.0, 8.0, 7.5]
        depth, gradient = find_thermocline(depths, temps)
        self.assertEqual(depth, 5.0)
        self.assertAlmostEqual(gradient, 3.0, delta=0.1)

    def test_brunt_vaisala_frequency(self):
        depths = [0.0, 5.0, 10.0]
        temps = [22.0, 18.0, 10.0]  # Stably stratified
        n2 = brunt_vaisala_frequency_squared(depths, temps)
        self.assertEqual(len(n2), 2)
        # All N^2 should be non-negative for stable stratification
        for val in n2:
            self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
