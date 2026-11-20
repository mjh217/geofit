"""Thermal property parameterisations against values from their papers."""

import numpy as np
import pytest
from scipy.optimize import brentq

from geofit.defaults import C2K
from geofit.materials import (
    ALPHA_CRUST,
    ALPHA_FORSTERITE,
    Density,
    HasterokConductivity,
    RadiativeConductivity,
    compression_crust,
    compression_mantle,
    cp_crust,
    cp_mantle,
    k_osako,
    kappa_crust,
    kappa_olivine,
    make_model,
)


def test_crust_whittington():
    # Whittington et al. (2009) eqs. 1-4 by hand, for an average crust of 221.78 g/mol
    assert kappa_crust(300.0) == pytest.approx((567.3 / 300.0 - 0.062) * 1e-6)
    assert kappa_crust(1000.0) == pytest.approx((0.732 - 0.000135 * 1000.0) * 1e-6)
    assert cp_crust(300.0) == pytest.approx((199.5 + 0.0857 * 300.0 - 5.0e6 / 300.0**2) / 0.22178)
    assert cp_crust(1000.0) == pytest.approx((229.32 + 0.0323 * 1000.0) / 0.22178, rel=1e-9)
    # Arrays give the same as scalars
    assert cp_crust(np.array([300.0, 1000.0])) == pytest.approx([cp_crust(300.0), cp_crust(1000.0)])


def test_expansivity_and_mantle_heat_capacity():
    # Bouhifd et al. (1996, Table 2), Stephenson et al. (2024, Table 1), Korenaga & Korenaga (2016, eq. 2)
    assert ALPHA_FORSTERITE(1000.0) == pytest.approx(2.832e-5 + 0.758e-8 * 1000.0)
    assert ALPHA_CRUST(1000.0) == pytest.approx(1.0e-5 + 2.9e-8 * 1000.0)
    assert cp_mantle(1600.0) == pytest.approx(1580 - 12230 / 40.0 - 1694e6 / 1600.0**3)
    # The integral of alpha from 273.15 K
    T = np.linspace(C2K, 1500.0, 2001)
    integral = np.sum(0.5 * (ALPHA_FORSTERITE(T[1:]) + ALPHA_FORSTERITE(T[:-1])) * np.diff(T))
    assert ALPHA_FORSTERITE.integral(1500.0) == pytest.approx(integral, rel=1e-6)


def test_olivine_conductivity_matches_grose_afonso_figure_3():
    # Fo90 lattice diffusivity at 0 C is about 2.6 mm^2/s
    assert kappa_olivine(C2K) == pytest.approx(2.635e-6)
    # For 1 cm grains, radiative conductivity peaks at about 1.6 W/m/K near 1050 C
    T = np.linspace(0, 1500, 1501) + C2K
    k_rad = RadiativeConductivity(1.0)(T)
    assert k_rad.max() == pytest.approx(1.62, abs=0.02)
    assert T[np.argmax(k_rad)] - C2K == pytest.approx(1043, abs=10)


def test_osako_matches_averaged_olivine():
    # At 293 K and 0 GPa, Osako et al. (2004, Table 2, B0) give 6.61, 3.98 and 5.91 W/m/K for the three axes
    assert k_osako(0.0, 293.0) == pytest.approx(np.mean([6.61, 3.98, 5.91]), abs=0.05)


def test_hasterok():
    k = HasterokConductivity(2.5)
    # k0 at room temperature and pressure; tends to k0 (n - 1) / n when hot
    assert k(0.0, 25.0 + C2K) == pytest.approx(2.5)
    assert k(0.0, 5000.0 + C2K) == pytest.approx(2.5 * (k.n - 1) / k.n)
    # 10% per GPa on the first term
    assert k(1e9, 5000.0 + C2K) == pytest.approx(1.1 * 2.5 * (k.n - 1) / k.n)


def birch_murnaghan(K0, P, Kp=4.8):
    """V0/V from the third-order Birch-Murnaghan equation of state (Grose & Afonso, 2013, eq. 3)."""

    def pressure(x):
        return 1.5 * K0 * (x ** (7 / 3) - x ** (5 / 3)) * (1 + 0.75 * (Kp - 4) * (x ** (2 / 3) - 1))

    return brentq(lambda x: pressure(x) - P, 0.9, 1.5)


@pytest.mark.parametrize("compression, K0, P_max", [(compression_crust, 90e9, 2e9), (compression_mantle, 130e9, 14e9)])
def test_compression_fits(compression, K0, P_max):
    for P in np.linspace(0, P_max, 15):
        assert compression(P) == pytest.approx(birch_murnaghan(K0, P), abs=1e-3)


@pytest.mark.parametrize(
    "rho", [Density(3300.0, compression_mantle, ALPHA_FORSTERITE), Density(2700.0, compression_crust, ALPHA_CRUST)]
)
def test_density(rho):
    # Surface density at the surface; less dense when hot, denser when compressed
    assert rho(101325.0, C2K) == pytest.approx(rho.rho0, rel=1e-3)
    assert rho(0.0, 1000.0 + C2K) < rho(0.0, C2K) < rho(3e9, C2K)


def test_models():
    for name, layer in [("constant", "crust"), ("hasterok", "crust"), ("hoggard", "crust"), ("constant", "mantle"),
                        ("osako", "mantle"), ("hoggard", "mantle")]:
        m = make_model(name, layer, 3e-5, 1000.0, 3000.0, 2.5)
        # Scalars in, scalars out; arrays in, arrays out, including the constant properties
        assert all(np.ndim(v) == 0 for v in m.compute(1e3, 1e8, 600.0))
        values = m.compute(np.array([1e3, 2e3]), np.array([1e8, 2e8]), np.array([600.0, 700.0]))
        assert all(np.shape(v) == (2,) for v in values)
    for name, layer in [("unknown", "crust"), ("osako", "crust"), ("hasterok", "mantle")]:
        with pytest.raises(ValueError):
            make_model(name, layer, 0, 0, 0, 0)


def test_hoggard_models_combine_the_parameterisations():
    P, T = 3e9, 1200.0
    mantle = make_model("hoggard", "mantle", 0, 0, 3300.0, 0)
    rho = Density(3300.0, compression_mantle, ALPHA_FORSTERITE)(P, T)
    k = kappa_olivine(T) * rho * cp_mantle(T) * np.exp(4.0 / 130e9 * P) + RadiativeConductivity(0.3)(T)
    assert mantle.compute(0.0, P, T) == pytest.approx((ALPHA_FORSTERITE(T), cp_mantle(T), rho, k))

    crust = make_model("hoggard", "crust", 0, 0, 2700.0, 0)
    rho = Density(2700.0, compression_crust, ALPHA_CRUST)(P / 3, T)
    k = kappa_crust(T) * rho * cp_crust(T)
    assert crust.compute(0.0, P / 3, T) == pytest.approx((ALPHA_CRUST(T), cp_crust(T), rho, k))
