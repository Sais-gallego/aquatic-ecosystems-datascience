"""
Physics-Informed Modeling (PINN) for Vertical Lake Stratification.
DATA20067 - Data Science for Monitoring Aquatic Ecosystems, University of Helsinki.
"""

import math
from typing import List, Tuple, Dict, Any


class PhysicsInformedLakeModel:
    """
    Implements 1D vertical heat diffusion and solar radiation penetration
    for physics-guided aquatic neural networks.
    """

    KAPPA_Z = 1.2e-4  # Vertical thermal eddy diffusivity (m^2/s)
    RHO_W = 1000.0    # Density of freshwater (kg/m^3)
    CP_W = 4184.0     # Specific heat capacity of water (J / kg*K)
    LIGHT_EXTINCTION_ETA = 0.45  # Turbidity-dependent light attenuation coefficient (m^-1)

    @classmethod
    def solar_flux_at_depth(cls, surface_irradiance_w_m2: float, depth_m: float) -> float:
        """
        Beer-Lambert law for solar irradiance penetration into water column:
        I(z) = I_0 * exp(-eta * z)
        """
        return surface_irradiance_w_m2 * math.exp(-cls.LIGHT_EXTINCTION_ETA * depth_m)

    @classmethod
    def heat_source_term(cls, surface_irradiance_w_m2: float, depth_m: float) -> float:
        """
        Volumetric solar heating rate (K/s):
        H(z) / (rho * Cp) = (eta * I(z)) / (rho * Cp)
        """
        i_z = cls.solar_flux_at_depth(surface_irradiance_w_m2, depth_m)
        q_volumetric = cls.LIGHT_EXTINCTION_ETA * i_z  # W/m^3
        return q_volumetric / (cls.RHO_W * cls.CP_W)

    @classmethod
    def compute_pde_residual(
        cls,
        temp_prev_t: float,
        temp_curr_t: float,
        temp_above_z: float,
        temp_curr_z: float,
        temp_below_z: float,
        dt_seconds: float,
        dz_meters: float,
        surface_irradiance_w_m2: float,
        depth_m: float
    ) -> float:
        """
        Computes finite-difference residual of 1D vertical thermal diffusion PDE:
        R = dT/dt - kappa_z * d2T/dz2 - Source
        """
        if dt_seconds <= 0 or dz_meters <= 0:
            return 0.0

        # Time derivative dT/dt
        dt_dt = (temp_curr_t - temp_prev_t) / dt_seconds

        # Spatial second derivative d2T/dz2
        d2t_dz2 = (temp_below_z - 2.0 * temp_curr_z + temp_above_z) / (dz_meters ** 2)

        # Source heating
        source = cls.heat_source_term(surface_irradiance_w_m2, depth_m)

        residual = dt_dt - cls.KAPPA_Z * d2t_dz2 - source
        return residual

    @classmethod
    def evaluate_physics_loss(
        cls,
        predicted_temps: List[List[float]],  # grid of [time_step][depth_step]
        dt_seconds: float,
        dz_meters: float,
        surface_irradiance_w_m2: float
    ) -> float:
        """
        Computes mean squared PDE residual loss across spatio-temporal grid.
        """
        num_t = len(predicted_temps)
        if num_t < 2:
            return 0.0
        num_z = len(predicted_temps[0])
        if num_z < 3:
            return 0.0

        squared_residuals = []
        for t in range(1, num_t):
            for z in range(1, num_z - 1):
                temp_prev_t = predicted_temps[t - 1][z]
                temp_curr_t = predicted_temps[t][z]
                temp_above_z = predicted_temps[t][z - 1]
                temp_curr_z = predicted_temps[t][z]
                temp_below_z = predicted_temps[t][z + 1]
                depth_m = z * dz_meters

                r = cls.compute_pde_residual(
                    temp_prev_t, temp_curr_t,
                    temp_above_z, temp_curr_z, temp_below_z,
                    dt_seconds, dz_meters,
                    surface_irradiance_w_m2, depth_m
                )
                squared_residuals.append(r ** 2)

        return sum(squared_residuals) / len(squared_residuals) if squared_residuals else 0.0
