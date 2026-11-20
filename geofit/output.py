"""Tables written by the inversion and the forward model."""

import numpy as np
import pandas as pd

from . import forward
from .defaults import C2K, Z_MAX

GEOTHERM_COLUMNS = [
    "Depth (km)",
    "Pressure (GPa)",
    "Temperature (Celsius)",
    "Adiabatic temperature (Celsius)",
    "Thermal expansion (per K)",
    "Specific heat capacity (J per kg per K)",
    "Density (kg per cubic metre)",
    "Thermal conductivity (W per m per K)",
]

# Column names of the inverted parameters, in model-vector order
SURFACE_HEAT_FLUX = "Surface heat flux (mW per squared metre)"
CRUSTAL_THICKNESS = "Crustal thickness (km)"
CRUSTAL_HEATING = "Crustal radiogenic heating (microW per cubic metre)"
MANTLE_HEATING = "Mantle radiogenic heating (microW per cubic metre)"
POTENTIAL_TEMPERATURE = "Mantle potential temperature (Celsius)"
DATA_UNCERTAINTY = "P error on datapoints (Pa)"


def geotherm_table(crust_model, mantle_model, g, z_crust, h_crust, h_mantle, surface_heat_flux, adiabat, alphaD):
    """A geotherm with its thermal properties, and the depth of the MBL [m].

    z_crust is in m. The table is every km (plus the smoothed corner), in the units of
    GEOTHERM_COLUMNS; properties are from the crustal model above z_crust and the mantle model below.
    """
    z, T, P, z_mbl = forward.compute_geotherm(
        crust_model, mantle_model, g, z_crust, Z_MAX, h_crust, h_mantle, surface_heat_flux, adiabat, alphaD
    )
    in_crust = z < z_crust
    properties = [
        np.where(in_crust, crust, mantle)
        for crust, mantle in zip(crust_model.compute(z, P, T), mantle_model.compute(z, P, T))
    ]
    columns = [z / 1e3, P / 1e9, T - C2K, adiabat.T(P) - C2K, *properties]
    return pd.DataFrame(np.column_stack(columns), columns=GEOTHERM_COLUMNS), z_mbl
