"""Material models against values from their papers."""

import numpy as np
import pytest
from scipy.optimize import brentq

from geofit.defaults import C2K
from geofit.materials import (
    HasterokModel,
    HoggardMantleModel,
    OsakoModel,
    HoggardCrustModel,
    compression_crust,
    compression_mantle,
    make_model,
)


def test_hoggard_crust():
    m = HoggardCrustModel()
    # Eqs. 1-4 by hand, for an average crust of 221.78 g/mol
    assert m.kappa(300.0) == pytest.approx((567.3 / 300.0 - 0.062) * 1e-6)
    assert m.kappa(1000.0) == pytest.approx((0.732 - 0.000135 * 1000.0) * 1e-6)
    assert m.cp(300.0) == pytest.approx((199.5 + 0.0857 * 300.0 - 5.0e6 / 300.0**2) / 0.22178)
    assert m.cp(1000.0) == pytest.approx((229.32 + 0.0323 * 1000.0) / 0.22178, rel=1e-9)
    # Arrays give the same as scalars
    assert m.cp(np.array([300.0, 1000.0])) == pytest.approx([m.cp(300.0), m.cp(1000.0)])


def test_hoggard_mantle_heat_capacity_and_expansivity():
    m = HoggardMantleModel()
    # Korenaga & Korenaga (2016, eq. 2) and Bouhifd et al. (1996, Table 2)
    assert m.cp(1600.0) == pytest.approx(1580 - 12230 / 40.0 - 1694e6 / 1600.0**3)
    assert m.alpha(1000.0) == pytest.approx(2.832e-5 + 0.758e-8 * 1000.0)


def test_hoggard_mantle_conductivity_matches_grose_afonso_figure_3():
    # Fo90 lattice diffusivity at 0 C is about 2.6 mm^2/s
    assert HoggardMantleModel().kappa_lattice(C2K) == pytest.approx(2.635e-6)
    # For 1 cm grains, radiative conductivity peaks at about 1.6 W/m/K near 1050 C
    m = HoggardMantleModel(grain_size=1.0)
    T = np.linspace(0, 1500, 1501) + C2K
    k_rad = m.k_radiative(T)
    assert k_rad.max() == pytest.approx(1.62, abs=0.02)
    assert T[np.argmax(k_rad)] - C2K == pytest.approx(1043, abs=10)


def test_osako_matches_averaged_olivine():
    # At 293 K and 0 GPa, Osako et al. (2004, Table 2, B0) give 6.61, 3.98 and 5.91 W/m/K for the three axes
    k = OsakoModel(3e-5, 1187.0, 3300.0).compute(0.0, 0.0, 293.0)[3]
    assert k == pytest.approx(np.mean([6.61, 3.98, 5.91]), abs=0.05)


def test_hasterok():
    m = HasterokModel(3e-5, 750.0, 2700.0, 2.5)
    # k0 at room temperature and pressure; tends to k0 (n - 1) / n when hot
    assert m.compute(0.0, 0.0, 25.0 + C2K)[3] == pytest.approx(2.5)
    assert m.compute(0.0, 0.0, 5000.0 + C2K)[3] == pytest.approx(2.5 * (m.n - 1) / m.n)
    # 10% per GPa on the first term
    assert m.compute(0.0, 1e9, 5000.0 + C2K)[3] == pytest.approx(1.1 * 2.5 * (m.n - 1) / m.n)


def birch_murnaghan(K0, P, Kp=4.8):
    """V0/V from the third-order Birch-Murnaghan equation of state (Grose & Afonso, 2013, eq. 3)."""

    def pressure(x):
        return 1.5 * K0 * (x ** (7 / 3) - x ** (5 / 3)) * (1 + 0.75 * (Kp - 4) * (x ** (2 / 3) - 1))

    return brentq(lambda x: pressure(x) - P, 0.9, 1.5)


@pytest.mark.parametrize("compression, K0, P_max", [(compression_crust, 90e9, 2e9), (compression_mantle, 130e9, 14e9)])
def test_compression_fits(compression, K0, P_max):
    for P in np.linspace(0, P_max, 15):
        assert compression(P) == pytest.approx(birch_murnaghan(K0, P), abs=1e-3)


@pytest.mark.parametrize("model", [HoggardMantleModel(rho=3300.0), HoggardCrustModel(rho=2700.0)])
def test_density(model):
    # Surface density at the surface; less dense when hot, denser when compressed
    assert model.rho(101325.0, C2K) == pytest.approx(model.rho0, rel=1e-3)
    assert model.rho(0.0, 1000.0 + C2K) < model.rho(0.0, C2K) < model.rho(3e9, C2K)


def test_shapes():
    # Scalars in, scalars out; arrays in, arrays out, including the constant properties
    for name, layer in [("constant", "crust"), ("osako", "mantle"), ("hasterok", "crust"), ("hoggard", "crust"),
                        ("hoggard", "mantle")]:
        m = make_model(name, layer, 3e-5, 1000.0, 3000.0, 2.5)
        assert all(np.ndim(v) == 0 for v in m.compute(1e3, 1e8, 600.0))
        values = m.compute(np.array([1e3, 2e3]), np.array([1e8, 2e8]), np.array([600.0, 700.0]))
        assert all(np.shape(v) == (2,) for v in values)
    assert isinstance(make_model("hoggard", "crust", 0, 0, 2700.0, 0), HoggardCrustModel)
    assert isinstance(make_model("hoggard", "mantle", 0, 0, 3300.0, 0), HoggardMantleModel)
    assert make_model("hoggard", "mantle", 0, 0, 3300.0, 0).xA == HoggardMantleModel(grain_size=0.3).xA
    for name, layer in [("unknown", "crust"), ("osako", "crust"), ("hasterok", "mantle")]:
        with pytest.raises(ValueError):
            make_model(name, layer, 0, 0, 0, 0)
