"""Forward model against exact solutions for constant properties."""

import numpy as np
import pytest
from scipy.optimize import brentq

from geofit import forward
from geofit.defaults import P_SURFACE, T_SURFACE, Z_MAX, Z_MAX_ADIABAT
from geofit.materials import constant_model

ALPHA, CP, K = 3e-5, 1187.0, 2.5
RHO_C, RHO_M = 2700.0, 3300.0
G = 9.81
TP = 1603.15
CRUST = constant_model(ALPHA, 750.0, RHO_C, K)
MANTLE = constant_model(ALPHA, CP, RHO_M, K)


@pytest.fixture(scope="module")
def adiabat():
    return forward.compute_adiabat(MANTLE, forward.gravity_constant, TP, P_SURFACE, Z_MAX_ADIABAT)


def adiabat_T(P):
    """dT/dz = alpha g T / cp with dP/dz = rho g gives T = Tp exp(alpha (P - P0) / (rho cp))."""
    return TP * np.exp(ALPHA * (P - P_SURFACE) / (RHO_M * CP))


def test_adiabat(adiabat):
    P = np.linspace(P_SURFACE, 12e9, 50)
    assert adiabat.T(P) == pytest.approx(adiabat_T(P), abs=0.05)


def test_geotherm(adiabat):
    q0, zc, h = 0.06, 35e3, 1e-6
    z, T, P, z_lab = forward.compute_geotherm(
        CRUST, MANTLE, forward.gravity_constant, zc, Z_MAX, h, 0.0, q0, adiabat, alphaD=0.0
    )
    assert np.all(np.diff(z) > 0) and z[-1] == Z_MAX

    # Crust: T = T0 + q0 z / k - H z^2 / (2k); mantle (no heat production): linear with the Moho heat flux
    T_moho = T_SURFACE + q0 * zc / K - h * zc**2 / (2 * K)
    q_moho = q0 - h * zc
    P_moho = P_SURFACE + RHO_C * G * zc

    def T_exact(z):
        return np.where(z <= zc, T_SURFACE + q0 * z / K - h * z**2 / (2 * K), T_moho + q_moho * (z - zc) / K)

    def P_exact(z):
        return np.where(z <= zc, P_SURFACE + RHO_C * G * z, P_moho + RHO_M * G * (z - zc))

    z_exact = brentq(lambda z: T_exact(z) - adiabat_T(P_exact(z)), zc, Z_MAX)
    assert z_lab == pytest.approx(z_exact, abs=100.0)

    above = z <= z_lab
    assert T[above] == pytest.approx(T_exact(z[above]), abs=0.01)
    assert P == pytest.approx(P_exact(np.minimum(z, z_lab)) + RHO_M * G * np.maximum(z - z_lab, 0), rel=1e-6)
    # Below the thermal LAB the profile follows the adiabat
    assert T[~above] == pytest.approx(adiabat_T(P[~above]), abs=0.1)


def test_smoothed_corner(adiabat):
    args = (CRUST, MANTLE, forward.gravity_constant, 35e3, Z_MAX, 1e-6, 0.0, 0.06, adiabat)
    z0, T0, P0, z_lab = forward.compute_geotherm(*args, alphaD=0.0)
    z, T, P, z_lab_smooth = forward.compute_geotherm(*args, alphaD=0.2)
    assert z_lab_smooth == z_lab
    assert np.all(np.diff(z) > 0)
    # Unchanged outside the window of width 0.2 z_lab centred on z_lab
    outside = np.abs(z - z_lab) > 0.1 * z_lab + 1.0  # (the window ends are new points)
    assert np.isin(z[outside], z0).all()
    assert T[outside] == pytest.approx(np.interp(z[outside], z0, T0))
    # The smoothed temperature lies below the sharp corner
    inside = ~outside
    assert np.all(T[inside] <= np.interp(z[inside], z0, T0) + 1e-6)


@pytest.mark.parametrize(
    "q0, h, error",
    [
        (0.25, 1e-6, forward.AdiabatInCrustError),  # far too hot: reaches the adiabat in the crust
        (0.03, 1e-6, forward.NegativeFluxError),  # crustal heat production exceeds the surface heat flux
        (0.04, 1e-6, forward.NoAdiabatError),  # too cold to reach the adiabat above 400 km
    ],
)
def test_impossible_geotherms(adiabat, q0, h, error):
    with pytest.raises(error):
        forward.compute_geotherm(CRUST, MANTLE, forward.gravity_constant, 35e3, Z_MAX, h, 0.0, q0, adiabat)


def test_gravity():
    assert forward.gravity_constant(100e3) == 9.81
    assert forward.gravity(0.0) == pytest.approx(9.81)
    # The two parts of the fit meet at 80 km
    assert forward.gravity(80e3 - 1) == pytest.approx(forward.gravity(80e3 + 1), rel=1e-6)
    assert forward.gravity(400e3) == pytest.approx(9.81 * 1.0171182)
