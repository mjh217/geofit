"""Thermal and density properties of the crust and mantle.

Each model returns thermal expansivity alpha [1/K], specific heat capacity cp [J/kg/K],
density rho [kg/m^3] and thermal conductivity k [W/m/K] at a given depth [m], pressure [Pa] and
temperature [K]. Choose one for the crust and one for the mantle with `make_model`.
"""

import abc

import numpy as np

from .defaults import C2K


class Model(metaclass=abc.ABCMeta):
    """Base class: subclasses implement `compute`."""

    @abc.abstractmethod
    def compute(self, depth, pressure, temperature):
        """Return (alpha, cp, rho, k) at depth [m], pressure [Pa] and temperature [K]."""

    @staticmethod
    def _ones(*values):
        """1.0 for scalar inputs, or an array of ones as long as the longest array input.

        Multiplying by this gives every property the shape of the inputs, including those that
        are constant.
        """
        size = 0
        for value in values:
            if not isinstance(value, (int, float)):
                size = max(size, len(value))
        return 1.0 if size == 0 else np.ones(size)


class ConstantModel(Model):
    """All four properties constant."""

    def __init__(self, alpha, cp, rho, k):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.k = k

    def compute(self, depth, pressure, temperature):
        ones = self._ones(depth, pressure, temperature)
        return self.alpha * ones, self.cp * ones, self.rho * ones, self.k * ones


class OsakoModel(Model):
    """Constant alpha, cp and rho, with temperature- and pressure-dependent mantle conductivity.

    Mantle model mmod = 4 of FITPLOT: k = (1.175 + 1264 / T) exp(0.038 P), with T in K and P in GPa.
    """

    def __init__(self, alpha, cp, rho):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho

    def compute(self, depth, pressure, temperature):
        k = (1.175 + 1264.0 / temperature) * np.exp(0.038 * pressure / 1e9)
        ones = self._ones(depth, pressure, temperature)
        return self.alpha * ones, self.cp * ones, self.rho * ones, k * ones


class HasterokModel(Model):
    """Constant alpha, cp and rho, with temperature- and pressure-dependent crustal conductivity.

    Not yet selectable: its units still need checking against Hasterok & Chapman (2011).
    """

    def __init__(self, alpha, cp, rho, crustk=2.5):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.crustk = crustk

    def compute(self, depth, pressure, temperature):
        n = 6.4 - 2.3 * np.log(self.crustk)
        beta = 0.1
        k1 = self.crustk * ((n - 1.0) / n) * (1.0 + beta * pressure)
        k2 = self.crustk * np.exp(-(temperature - 25.0) / 300.0) * (1.0 / n) * (1.0 + beta * pressure)
        k = k1 + k2
        ones = self._ones(depth, pressure, temperature)
        return self.alpha * ones, self.cp * ones, self.rho * ones, k * ones


def compression_crust(P):
    """Isothermal compression V0/V of the crust at pressure P [Pa].

    Polynomial fit to the third-order Birch-Murnaghan equation of state of Grose & Afonso (2013),
    with K0 = 90 GPa and K' = 4.8.
    """
    return 1.00000236 + (1.10952146e-11 - 2.12421383e-22 * P) * P


def compression_mantle(P):
    """Isothermal compression V0/V of the mantle at pressure P [Pa].

    As `compression_crust`, with K0 = 130 GPa and K' = 4.8.
    """
    return 1.00009143 + (7.57454484e-12 - 8.00340143e-23 * P) * P


def _density(rho0, compression, alpha0, alpha1, ag, P, T):
    """Density with pressure and temperature (Grose & Afonso, 2013), T in K.

    rho0 is the surface density, `compression` = V0/V at P, alpha = alpha0 + alpha1 T is the
    thermal expansivity and ag the Anderson-Gruneisen parameter. Thermal expansion is integrated
    from the surface temperature and reduced with compression.
    """
    alpha_ratio = compression * np.exp((ag + 1.0) * (1.0 / compression - 1.0))
    expansion = alpha0 * (T - C2K) + 0.5 * alpha1 * (T**2 - C2K**2)
    return rho0 * compression * (1.0 - alpha_ratio * expansion)


class HoggardModel(Model):
    """Mantle (olivine) properties that depend on temperature and pressure.

    - alpha: forsterite, Bouhifd et al. (1996).
    - cp: Korenaga & Korenaga (2016), with no pressure dependence (very weak over upper-mantle
      pressures; Hofmeister, 2007).
    - rho: Grose & Afonso (2013), using `compression_mantle` and the Anderson-Gruneisen parameter
      for Fo90 (Anderson & Isaak, 1995).
    - k: lattice diffusivity of mantle olivine (Grose & Afonso, 2013, fitted to Pertermann &
      Hofmeister, 2006), times rho cp, with the pressure dependence of Hofmeister (2007) for hard
      mantle phases, plus radiative conductivity for the given grain size (Grose & Afonso, 2013,
      after Hofmeister, 2005, with typos corrected).
    """

    def __init__(self, rho=3.3e3, grain_size=0.1, bulk_modulus=130e9):
        self.rho0 = rho

        # Thermal expansivity, alpha = alpha0 + alpha1 T (Bouhifd et al., 1996)
        self.alpha0 = 2.832e-5
        self.alpha1 = 0.758e-8

        # Heat capacity coefficients (Korenaga & Korenaga, 2016)
        self.c0 = 1580.0
        self.c1 = 12230.0
        self.c2 = 1694e6

        # d(ln k)/dP for lattice conductivity (Hofmeister, 2007)
        self.dlogk_dP = 4.0 / bulk_modulus

        # Lattice thermal diffusivity coefficients (Grose & Afonso, 2013)
        self.kappa0 = 0.565
        self.kappa1 = 0.67
        self.kappa2 = 590.0
        self.kappa3 = 1.4
        self.kappa4 = 135.0

        # Radiative conductivity coefficients for grain size in cm (Grose & Afonso, 2013)
        d = grain_size
        self.Ar = 1.8 * (1.0 - np.exp(-(d**1.3) / 0.15)) - (1.0 - np.exp(-np.sqrt(d) / 5.0))
        self.Br = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-(d**3) / 10.0)
        self.Ta = 490.0 + 1850.0 * np.exp(-(d**0.315) / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.Tb = 2700.0 + 9000.0 * np.exp(-np.sqrt(d) / 0.205)
        self.xA = 167.5 + 505.0 * np.exp(-np.sqrt(d) / 0.85)
        self.xB = 465.0 + 1700.0 * np.exp(-(d**0.94) / 0.175)

        # Anderson-Gruneisen parameter for Fo90 (Anderson & Isaak, 1995)
        self.ag = 6.0

    def alpha(self, T):
        return self.alpha0 + self.alpha1 * T

    def cp(self, T):
        return self.c0 - self.c1 / np.sqrt(T) - self.c2 / T**3

    def rho(self, P, T):
        return _density(self.rho0, compression_mantle(P), self.alpha0, self.alpha1, self.ag, P, T)

    def kappa_lattice(self, T):
        """Lattice thermal diffusivity [m^2/s]; the fit uses T in Celsius."""
        return (
            self.kappa0 + self.kappa1 * np.exp((C2K - T) / self.kappa2) + self.kappa3 * np.exp((C2K - T) / self.kappa4)
        ) * 1e-6

    def k_radiative(self, T):
        """Radiative conductivity [W/m/K]; the fit uses T in Celsius. Pressure dependence unconstrained."""
        return self.Ar * np.exp(-(((T - C2K) - self.Ta) ** 2) / (2.0 * self.xA**2)) + self.Br * np.exp(
            -(((T - C2K) - self.Tb) ** 2) / (2.0 * self.xB**2)
        )

    def k(self, P, T):
        return self.kappa_lattice(T) * self.rho(P, T) * self.cp(T) * np.exp(self.dlogk_dP * P) + self.k_radiative(T)

    def compute(self, depth, pressure, temperature):
        alpha = self.alpha(temperature)
        cp = self.cp(temperature)
        rho = self.rho(pressure, temperature)
        k = self.k(pressure, temperature)
        ones = self._ones(depth, pressure, temperature)
        return alpha * ones, cp * ones, rho * ones, k * ones


class WhittingtonModel(Model):
    """Crustal properties that depend on temperature and pressure.

    - alpha: Stephenson et al. (2024), from the data of Skinner (1966) and Cooper & Simmons (1977).
    - cp and thermal diffusivity: Whittington et al. (2009), eqs. 1-4, with a change of fit at 846 K.
    - rho: Grose & Afonso (2013), using `compression_crust`.
    - k: diffusivity times rho cp.
    """

    def __init__(self, rho=2.7e3):
        self.rho0 = rho

        # Thermal expansivity, alpha = alpha0 + alpha1 T (Stephenson et al., 2024)
        self.alpha0 = 1.0e-5
        self.alpha1 = 2.9e-8

        # Anderson-Gruneisen parameter
        self.ag = 6.0

    def kappa(self, T):
        """Thermal diffusivity [m^2/s] (Whittington et al., 2009, eqs. 1-2)."""
        if np.isscalar(T):
            kappa = 567.3 / T - 0.062 if T < 846.0 else 0.732 - 0.000135 * T
            return kappa / 1e6
        return [self.kappa(t) for t in T]

    def cp(self, T):
        """Specific heat capacity [J/kg/K] (Whittington et al., 2009, eqs. 3-4), per kg of 221.78 g/mol."""
        if np.isscalar(T):
            if T < 846.0:
                return (199.5 + 0.0857 * T - 5e6 / T**2) / 221.78e-3
            return (229.32 + 0.0323 * T - 47.9e-6 / T**2) / 221.78e-3
        return [self.cp(t) for t in T]

    def alpha(self, T):
        return self.alpha0 + self.alpha1 * T

    def rho(self, P, T):
        return _density(self.rho0, compression_crust(P), self.alpha0, self.alpha1, self.ag, P, T)

    def k(self, P, T):
        return self.kappa(T) * self.rho(P, T) * self.cp(T)

    def compute(self, depth, pressure, temperature):
        alpha = self.alpha(temperature)
        cp = self.cp(temperature)
        rho = self.rho(pressure, temperature)
        k = self.k(pressure, temperature)
        ones = self._ones(depth, pressure, temperature)
        return alpha * ones, cp * ones, rho * ones, k * ones


MODELS = ("constant", "osako", "hoggard", "whittington")


def make_model(name, alpha, cp, rho, k, **kwargs):
    """Build a model by name (case-insensitive).

    alpha, cp, rho and k are the constant values, used where the model does not compute them.
    kwargs (grain_size, bulk_modulus) are passed to the Hoggard model.
    """
    name = name.lower()
    if name == "constant":
        return ConstantModel(alpha, cp, rho, k)
    if name == "osako":
        return OsakoModel(alpha, cp, rho)
    if name == "hoggard":
        return HoggardModel(rho, **kwargs)
    if name == "whittington":
        return WhittingtonModel(rho)
    raise ValueError(f"Unknown model '{name}' (choose from {', '.join(MODELS)})")
