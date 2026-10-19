"""
QARTOD Quality Assurance & Quality Control (QA/QC) Pipeline.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


class QartodFlag:
    """Standard IOOS QARTOD Quality Control Flags."""
    PASS = 1
    NOT_EVALUATED = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


class QartodPipeline:
    """
    Implements automated real-time quality control checks for aquatic sensor streams
    following NOAA / IOOS QARTOD specifications.
    """

    DEFAULT_GROSS_RANGES = {
        "temp_c": (-2.0, 45.0),
        "do_mg_l": (0.0, 25.0),
        "chla_ug_l": (0.0, 200.0),
        "turbidity_fnu": (0.0, 1000.0),
        "battery_pct": (0.0, 100.0)
    }

    DEFAULT_RATE_THRESHOLDS = {
        "temp_c": 5.0,        # max 5 °C per hour
        "do_mg_l": 6.0,       # max 6 mg/L per hour
        "chla_ug_l": 50.0,    # max 50 ug/L per hour
        "turbidity_fnu": 200.0
    }

    @classmethod
    def test_gross_range(cls, param_name: str, value: Optional[float]) -> int:
        """
        Gross Range Test: checks whether reading is within physically plausible bounds.
        """
        if value is None:
            return QartodFlag.MISSING
        if param_name not in cls.DEFAULT_GROSS_RANGES:
            return QartodFlag.NOT_EVALUATED
        low, high = cls.DEFAULT_GROSS_RANGES[param_name]
        if value < low or value > high:
            return QartodFlag.FAIL
        return QartodFlag.PASS

    @classmethod
    def test_spike(cls, values: List[Optional[float]], current_idx: int, spike_threshold: float) -> int:
        """
        Spike Test: calculates deviation of a single point from its immediate neighbors.
        test_val = |v_t - (v_{t-1} + v_{t+1}) / 2|
        """
        if current_idx <= 0 or current_idx >= len(values) - 1:
            return QartodFlag.NOT_EVALUATED
        v_prev = values[current_idx - 1]
        v_curr = values[current_idx]
        v_next = values[current_idx + 1]
        if v_prev is None or v_curr is None or v_next is None:
            return QartodFlag.MISSING

        reference = (v_prev + v_next) / 2.0
        diff = abs(v_curr - reference)
        if diff > spike_threshold:
            return QartodFlag.FAIL
        return QartodFlag.PASS

    @classmethod
    def test_rate_of_change(cls, param_name: str, v_prev: Optional[float], v_curr: Optional[float], dt_hours: float = 1.0) -> int:
        """
        Rate of Change Test: detects unphysically rapid shifts over consecutive samples.
        """
        if v_prev is None or v_curr is None or dt_hours <= 0:
            return QartodFlag.NOT_EVALUATED
        threshold = cls.DEFAULT_RATE_THRESHOLDS.get(param_name, 10.0)
        rate = abs(v_curr - v_prev) / dt_hours
        if rate > threshold:
            return QartodFlag.SUSPECT
        return QartodFlag.PASS

    @classmethod
    def test_flatline(cls, window_values: List[Optional[float]], tolerance: float = 1e-4) -> int:
        """
        Flatline Test: detects frozen sensors outputting identical readings over multiple time steps.
        """
        valid = [v for v in window_values if v is not None]
        if len(valid) < 3:
            return QartodFlag.NOT_EVALUATED
        val_range = max(valid) - min(valid)
        if val_range <= tolerance:
            return QartodFlag.FAIL
        return QartodFlag.PASS

    @classmethod
    def run_all_checks(cls, time_series: List[Dict[str, Any]], param_name: str) -> List[int]:
        """
        Executes complete battery of QARTOD tests across time series and produces primary flags.
        Flag priority: FAIL (4) > SUSPECT (3) > PASS (1).
        """
        n = len(time_series)
        flags = [QartodFlag.PASS] * n
        vals = [r.get(param_name) for r in time_series]

        for i in range(n):
            v = vals[i]
            # 1. Gross range
            r_flag = cls.test_gross_range(param_name, v)
            if r_flag == QartodFlag.FAIL:
                flags[i] = QartodFlag.FAIL
                continue

            # 2. Rate of change
            if i > 0 and flags[i] != QartodFlag.FAIL:
                roc_flag = cls.test_rate_of_change(param_name, vals[i-1], v)
                if roc_flag == QartodFlag.SUSPECT and flags[i] == QartodFlag.PASS:
                    flags[i] = QartodFlag.SUSPECT

            # 3. Spike test
            if 0 < i < n - 1 and flags[i] != QartodFlag.FAIL:
                spike_flag = cls.test_spike(vals, i, cls.DEFAULT_RATE_THRESHOLDS.get(param_name, 10.0) * 1.5)
                if spike_flag == QartodFlag.FAIL:
                    flags[i] = QartodFlag.FAIL

            # 4. Flatline test over 4 consecutive steps
            if i >= 3:
                flat_flag = cls.test_flatline(vals[i-3:i+1])
                if flat_flag == QartodFlag.FAIL:
                    flags[i] = QartodFlag.FAIL

        return flags

    @staticmethod
    def impute_linear(values: List[Optional[float]]) -> List[float]:
        """Performs forward/backward bounded linear interpolation on missing sensor samples."""
        res = [v for v in values]
        n = len(res)
        
        # Forward fill leading Nones with first valid value
        first_valid_idx = next((i for i, v in enumerate(res) if v is not None), None)
        if first_valid_idx is None:
            return [0.0] * n
        for i in range(first_valid_idx):
            res[i] = res[first_valid_idx]

        # Interpolate interior gaps
        i = 0
        while i < n:
            if res[i] is None:
                start_idx = i - 1
                end_idx = i
                while end_idx < n and res[end_idx] is None:
                    end_idx += 1
                if end_idx < n:
                    v_start = res[start_idx]
                    v_end = res[end_idx]
                    step = (v_end - v_start) / (end_idx - start_idx)
                    for k in range(start_idx + 1, end_idx):
                        res[k] = v_start + step * (k - start_idx)
                else:
                    # Trailing Nones: backward fill
                    for k in range(start_idx + 1, n):
                        res[k] = res[start_idx]
                i = end_idx
            else:
                i += 1

        return [float(v) for v in res]
