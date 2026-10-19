"""
Domain Foundations & Limnological Physics.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

import math
from typing import List, Tuple, Dict, Any, Optional


def dissolved_oxygen_saturation(temp_c: float, salinity_psu: float = 0.0) -> float:
    """
    Computes dissolved oxygen saturation concentration (mg/L) at standard 1 atm pressure
    using the Garcia & Gordon (1992) formulation (recommended by SCOR/IAPSO).
    
    :param temp_c: Water temperature in degrees Celsius (-2 to 40).
    :param salinity_psu: Salinity in practical salinity units (PSU, 0 to 42).
    :return: DO saturation concentration in mg/L.
    """
    t_kelvin = temp_c + 273.15
    if t_kelvin <= 0:
        return 0.0
    
    # Scaled temperature variable Ts
    ts = math.log((298.15 - temp_c) / t_kelvin)

    # Garcia-Gordon coefficients for oxygen concentration in umol/kg
    a0 = 5.80871
    a1 = 3.20291
    a2 = 4.17887
    a3 = 5.10006
    a4 = -0.098864
    a5 = 7.55359

    b0 = -0.0388396
    b1 = -0.0564714
    b2 = -0.0767028
    b3 = -0.0075597
    c0 = -0.000273815

    ln_c = (
        a0 + a1 * ts + a2 * (ts ** 2) + a3 * (ts ** 3) + a4 * (ts ** 4) + a5 * (ts ** 5)
        + salinity_psu * (b0 + b1 * ts + b2 * (ts ** 2) + b3 * (ts ** 3))
        + c0 * (salinity_psu ** 2)
    )
    
    c_umol_per_kg = math.exp(ln_c)
    # Convert umol/kg to mg/L using molar mass of O2 (31.9988 g/mol) and approx density of water (1.025 kg/L for sea, 1.0 for fresh)
    water_density_kg_per_l = water_density(temp_c, salinity_psu) / 1000.0
    do_mg_per_l = (c_umol_per_kg * 31.9988 / 1000.0) * water_density_kg_per_l
    return do_mg_per_l


def water_density(temp_c: float, salinity_psu: float = 0.0) -> float:
    """
    Calculates water density (kg/m^3) at atmospheric pressure using Millero and Poisson (1981).
    """
    # Pure water density
    rho_0 = (
        999.842594 + 6.793952e-2 * temp_c - 9.095290e-3 * (temp_c ** 2)
        + 1.001685e-4 * (temp_c ** 3) - 1.120083e-6 * (temp_c ** 4) + 6.536332e-9 * (temp_c ** 5)
    )
    if salinity_psu <= 0.0:
        return rho_0

    # Salinity adjustments
    a = (
        8.24493e-1 - 4.0899e-3 * temp_c + 7.6438e-5 * (temp_c ** 2)
        - 8.2467e-7 * (temp_c ** 3) + 5.3875e-9 * (temp_c ** 4)
    )
    b = -5.72466e-3 + 1.0227e-4 * temp_c - 1.6546e-6 * (temp_c ** 2)
    c = 4.8314e-4

    return rho_0 + a * salinity_psu + b * (salinity_psu ** 1.5) + c * (salinity_psu ** 2)


def carlson_trophic_state_index(chla_ug_l: Optional[float] = None, secchi_m: Optional[float] = None) -> Dict[str, Any]:
    """
    Computes Carlson's Trophic State Index (TSI) for evaluating lake eutrophication.
    TSI < 40: Oligotrophic (clear, low productivity)
    40 - 50: Mesotrophic (moderate productivity)
    50 - 70: Eutrophic (high productivity, algal bloom risk)
    > 70: Hypereutrophic (severe blooms, surface scum, anoxia)
    """
    tsi_chla = None
    tsi_sd = None
    
    if chla_ug_l is not None and chla_ug_l > 0:
        tsi_chla = 9.81 * math.log(chla_ug_l) + 30.6
    if secchi_m is not None and secchi_m > 0:
        tsi_sd = 60.0 - 14.41 * math.log(secchi_m)
        
    scores = [s for s in [tsi_chla, tsi_sd] if s is not None]
    composite_tsi = sum(scores) / len(scores) if scores else 0.0
    
    if composite_tsi < 40.0:
        classification = "Oligotrophic"
    elif composite_tsi < 50.0:
        classification = "Mesotrophic"
    elif composite_tsi <= 70.0:
        classification = "Eutrophic"
    else:
        classification = "Hypereutrophic"

    return {
        "tsi_chla": tsi_chla,
        "tsi_secchi": tsi_sd,
        "composite_tsi": composite_tsi,
        "classification": classification
    }


def find_thermocline(depths_m: List[float], temps_c: List[float]) -> Tuple[float, float]:
    """
    Identifies thermocline depth (depth of maximum negative temperature gradient -dT/dz)
    and the corresponding maximum gradient (°C/m).
    """
    if len(depths_m) < 2 or len(depths_m) != len(temps_c):
        return (0.0, 0.0)

    max_gradient = 0.0
    thermocline_depth = depths_m[0]

    for i in range(len(depths_m) - 1):
        dz = depths_m[i + 1] - depths_m[i]
        if dz > 0:
            gradient = (temps_c[i] - temps_c[i + 1]) / dz
            if gradient > max_gradient:
                max_gradient = gradient
                thermocline_depth = (depths_m[i] + depths_m[i + 1]) / 2.0

    return (thermocline_depth, max_gradient)


def brunt_vaisala_frequency_squared(depths_m: List[float], temps_c: List[float], salinity_psu: float = 0.0) -> List[float]:
    """
    Computes the square of the Brunt-Väisälä buoyancy frequency N^2 (s^-2):
    N^2 = (g / rho_0) * (d rho / dz)
    Positive N^2 indicates stable stratification; negative N^2 indicates convective overturning.
    """
    g = 9.80665
    rho_0 = 1000.0
    n2_values = []

    densities = [water_density(t, salinity_psu) for t in temps_c]

    for i in range(len(depths_m) - 1):
        dz = depths_m[i + 1] - depths_m[i]
        drho = densities[i + 1] - densities[i]
        if dz > 0:
            n2 = (g / rho_0) * (drho / dz)
            n2_values.append(n2)
        else:
            n2_values.append(0.0)

    return n2_values
