"""
Aquatic Sensor Telemetry & In-Situ Buoy Simulation.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

import math
import struct
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class AquaticBuoyNode:
    """
    Simulates a solar-powered environmental buoy equipped with optical fluorometer,
    membrane DO sensor, CTD probe, battery monitor, and LoRaWAN uplink packetizer.
    """

    def __init__(self, station_id: int, lat: float, lon: float, battery_mah: float = 5000.0):
        self.station_id = station_id
        self.lat = lat
        self.lon = lon
        self.battery_mah = battery_mah
        self.max_battery = battery_mah
        self.biofouling_factor = 0.0  # 0.0 (clean) to 1.0 (fully fouled)

    def step_environment(
        self,
        time_hours: float,
        base_temp_c: float = 18.0,
        algae_bloom: bool = False,
        biofouling_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Simulates diurnal and physical variations of water parameters over a 1-hour interval.
        """
        # Diurnal water temperature cycle (peaks at 16:00)
        diurnal_temp = 2.0 * math.sin((time_hours - 10.0) * math.pi / 12.0)
        true_temp = base_temp_c + diurnal_temp + random.gauss(0, 0.1)

        # Diurnal photosynthesis Dissolved Oxygen cycle (supersaturated in afternoon, consumed at night)
        diurnal_do_variation = 1.8 * math.sin((time_hours - 8.0) * math.pi / 12.0)
        base_do = 8.5
        if algae_bloom:
            # Daytime super-saturation followed by nocturnal severe hypoxia
            true_do = base_do + 2.5 * diurnal_do_variation + (2.0 if 10 <= (time_hours % 24) <= 18 else -4.0)
            true_chla = 35.0 + 10.0 * math.sin(time_hours / 24.0) + random.gauss(0, 2.0)
            true_turbidity = 18.0 + random.gauss(0, 1.5)
        else:
            true_do = base_do + diurnal_do_variation + random.gauss(0, 0.15)
            true_chla = 4.0 + random.gauss(0, 0.5)
            true_turbidity = 2.5 + random.gauss(0, 0.3)

        # Apply biofouling degradation to optical sensor readings
        self.biofouling_factor = min(1.0, self.biofouling_factor + biofouling_rate)
        # Optical attenuation decreases apparent chlorophyll-a fluorescence over time
        measured_chla = max(0.0, true_chla * (1.0 - 0.7 * self.biofouling_factor))
        # Membrane fouling increases DO measurement lag and reduces apparent DO
        measured_do = max(0.0, true_do * (1.0 - 0.4 * self.biofouling_factor))

        # Solar harvesting vs telemetry discharge
        hour_of_day = time_hours % 24
        solar_generation_ma = 80.0 * max(0.0, math.sin((hour_of_day - 6.0) * math.pi / 12.0)) if 6 <= hour_of_day <= 18 else 0.0
        quiescent_drain_ma = 12.0  # sensors + controller
        telemetry_tx_drain_ma = 8.0  # transmission energy amortized over 1 hr
        net_current = solar_generation_ma - (quiescent_drain_ma + telemetry_tx_drain_ma)
        self.battery_mah = max(0.0, min(self.max_battery, self.battery_mah + net_current))
        battery_pct = (self.battery_mah / self.max_battery) * 100.0

        return {
            "station_id": self.station_id,
            "timestamp_hr": time_hours,
            "temp_c": round(true_temp, 2),
            "do_mg_l": round(max(0.0, measured_do), 2),
            "chla_ug_l": round(max(0.0, measured_chla), 2),
            "turbidity_fnu": round(max(0.0, true_turbidity), 2),
            "battery_pct": round(battery_pct, 1),
            "biofouling_pct": round(self.biofouling_factor * 100.0, 1)
        }

    def encode_lorawan_payload(self, reading: Dict[str, Any]) -> bytes:
        """
        Packs environmental observation into a compact 10-byte binary payload for LoRaWAN / NB-IoT:
        - station_id (uint16: 2 bytes)
        - temp_c * 100 (int16: 2 bytes, -327.68 to +327.67)
        - do_mg_l * 100 (uint16: 2 bytes, 0.00 to 655.35)
        - chla_ug_l * 100 (uint16: 2 bytes, 0.00 to 655.35)
        - battery_pct * 2 (uint8: 1 byte, 0 to 200 => 0.0% to 100.0%)
        - flags (uint8: 1 byte, bit 0: biofouling warning, bit 1: low battery alert)
        """
        station_id = reading["station_id"]
        temp_raw = int(reading["temp_c"] * 100)
        do_raw = int(reading["do_mg_l"] * 100)
        chla_raw = int(reading["chla_ug_l"] * 100)
        batt_raw = int(reading["battery_pct"] * 2)

        flags = 0
        if reading["biofouling_pct"] > 30.0:
            flags |= 0x01
        if reading["battery_pct"] < 20.0:
            flags |= 0x02

        return struct.pack(">HhhHBB", station_id, temp_raw, do_raw, chla_raw, batt_raw, flags)

    @staticmethod
    def decode_lorawan_payload(payload: bytes) -> Dict[str, Any]:
        """Unpacks 10-byte binary telemetry packet into readable dictionary."""
        station_id, temp_raw, do_raw, chla_raw, batt_raw, flags = struct.unpack(">HhhHBB", payload)
        return {
            "station_id": station_id,
            "temp_c": temp_raw / 100.0,
            "do_mg_l": do_raw / 100.0,
            "chla_ug_l": chla_raw / 100.0,
            "battery_pct": batt_raw / 2.0,
            "biofouling_alert": bool(flags & 0x01),
            "low_battery_alert": bool(flags & 0x02)
        }
