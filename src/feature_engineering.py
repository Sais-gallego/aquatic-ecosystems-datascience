"""
Feature Engineering for Aquatic Time Series & Remote Sensing.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

import math
from typing import List, Dict, Any, Tuple
from src.domain_foundations import dissolved_oxygen_saturation


class AquaticFeatureExtractor:
    """Extracts domain-informed physical, temporal, and rolling features for aquatic ML models."""

    @staticmethod
    def extract_temporal_cyclical_features(hour_of_day: float, day_of_year: float = 180.0) -> Dict[str, float]:
        """
        Transforms periodic time variables into continuous 2D sinusoidal embeddings:
        sin/cos diurnal (24 hr) and annual (365.25 day) cycles.
        """
        hour_rad = 2.0 * math.pi * (hour_of_day % 24.0) / 24.0
        day_rad = 2.0 * math.pi * (day_of_year % 365.25) / 365.25
        return {
            "diurnal_sin": round(math.sin(hour_rad), 4),
            "diurnal_cos": round(math.cos(hour_rad), 4),
            "annual_sin": round(math.sin(day_rad), 4),
            "annual_cos": round(math.cos(day_rad), 4)
        }

    @staticmethod
    def compute_apparent_oxygen_deficit(temp_c: float, measured_do: float, salinity_psu: float = 0.0) -> float:
        """
        Computes Apparent Oxygen Deficit (AOD) in mg/L:
        AOD = DO_saturation(T, S) - DO_measured
        Positive AOD indicates net respiration / organic matter decomposition;
        Negative AOD indicates net photosynthetic supersaturation.
        """
        do_sat = dissolved_oxygen_saturation(temp_c, salinity_psu)
        return round(do_sat - measured_do, 3)

    @staticmethod
    def compute_rolling_statistics(values: List[float], window_size: int = 6) -> List[Dict[str, float]]:
        """
        Computes trailing rolling mean, standard deviation, min, and max over a specified window.
        """
        results = []
        n = len(values)
        for i in range(n):
            start_idx = max(0, i - window_size + 1)
            sub = values[start_idx:i+1]
            w_len = len(sub)
            mean_val = sum(sub) / w_len
            variance = sum((x - mean_val)**2 for x in sub) / (w_len - 1 if w_len > 1 else 1)
            std_val = math.sqrt(variance)
            results.append({
                "rolling_mean": round(mean_val, 3),
                "rolling_std": round(std_val, 3),
                "rolling_min": round(min(sub), 3),
                "rolling_max": round(max(sub), 3)
            })
        return results

    @classmethod
    def build_ml_feature_matrix(
        cls,
        time_series: List[Dict[str, Any]],
        target_param: str = "chla_ug_l"
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Constructs supervised feature matrix X and target y for predictive modeling:
        Features included:
        - Current temp_c
        - Diurnal sin & cos
        - Apparent oxygen deficit
        - Rolling temp mean (6-step)
        - Rolling DO mean (6-step)
        - Rolling turbidity mean (6-step)
        - Lag-1 target value
        """
        temps = [r.get("temp_c", 15.0) for r in time_series]
        dos = [r.get("do_mg_l", 8.0) for r in time_series]
        turbs = [r.get("turbidity_fnu", 2.0) for r in time_series]
        targets = [r.get(target_param, 0.0) for r in time_series]

        roll_temp = cls.compute_rolling_statistics(temps, window_size=6)
        roll_do = cls.compute_rolling_statistics(dos, window_size=6)
        roll_turb = cls.compute_rolling_statistics(turbs, window_size=6)

        feature_names = [
            "temp_c",
            "diurnal_sin",
            "diurnal_cos",
            "aod_mg_l",
            "roll_temp_mean",
            "roll_do_mean",
            "roll_turb_mean",
            "lag1_target"
        ]

        X: List[List[float]] = []
        y: List[float] = []

        for i in range(1, len(time_series)):
            hr = time_series[i].get("timestamp_hr", 12.0)
            cyclical = cls.extract_temporal_cyclical_features(hr)
            t_val = temps[i]
            do_val = dos[i]
            aod = cls.compute_apparent_oxygen_deficit(t_val, do_val)

            row = [
                float(t_val),
                cyclical["diurnal_sin"],
                cyclical["diurnal_cos"],
                float(aod),
                roll_temp[i]["rolling_mean"],
                roll_do[i]["rolling_mean"],
                roll_turb[i]["rolling_mean"],
                float(targets[i-1])  # lag-1
            ]
            X.append(row)
            y.append(float(targets[i]))

        return X, y, feature_names
