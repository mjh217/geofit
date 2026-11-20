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
    """Mantle: constant alpha, cp and rho, with temperature- and pressure-dependent olivine conductivity.

    k = (1.175 + 1264 / T) exp(0.038 P), with T in K and P in GPa. This is the average over the three
    crystal axes of the olivine fits of Osako et al. (2004, Table 2), lambda = C0 + C1/T = 1.61 + 1732/T
    measured at 8.3 GPa, reduced to 0 GPa with their average pressure dependence lambda = B0 exp(B1 P),
    B1 = 0.038 / GPa.
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
    """Crust: constant alpha, cp and rho, with an empirical temperature- and pressure-dependent conductivity.

    Goes et al. (2020, section 3.2, unnumbered equations on p. 6), for crustal rocks:
        k = k1 + k2,  k1 = k0 (n - 1) / n (1 + beta P),  k2 = k0 exp(-(T - 25) / 300) / n,
    with T in Celsius, P in GPa, k0 the room-temperature conductivity, beta = 0.1 / GPa and
    n = 6.4 - 2.3 log k0 (taken as the natural log). k2 is printed without the 1/n; it is included
    here so that k = k0 at 25 C and 0 GPa. At high temperature k tends to k0 (n - 1) / n.
    """

    def __init__(self, alpha, cp, rho, k0):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.k0 = k0
        self.n = 6.4 - 2.3 * np.log(k0)
        self.beta = 0.1  # [1/GPa]

    def compute(self, depth, pressure, temperature):
        k1 = self.k0 * (self.n - 1.0) / self.n * (1.0 + self.beta * pressure / 1e9)
        k2 = self.k0 * np.exp(-(temperature - C2K - 25.0) / 300.0) / self.n
        ones = self._ones(depth, pressure, temperature)
        return self.alpha * ones, self.cp * ones, self.rho * ones, (k1 + k2) * ones


def compression_crust(P):
    """Isothermal compression V0/V of the crust at pressure P [Pa].

    Quadratic fit to the third-order Birch-Murnaghan equation of state (Grose & Afonso, 2013, eq. 3)
    with K0 = 90 GPa (Stephenson et al., 2024, Table 1) and K' = 4.8; within 0.001 of it to 2 GPa.
    """
    return 1.00000236 + (1.10952146e-11 - 2.12421383e-22 * P) * P


def compression_mantle(P):
    """Isothermal compression V0/V of the mantle at pressure P [Pa].

    As `compression_crust`, with K0 = 130 GPa and K' = 4.8 (Grose & Afonso, 2013, para. 11); within
    0.001 of the equation of state to 14 GPa.
    """
    return 1.00009143 + (7.57454484e-12 - 8.00340143e-23 * P) * P


def _density(rho0, compression, alpha0, alpha1, ag, P, T):
    """Density with pressure and temperature (Grose & Afonso, 2013, eqs. 4-6), T in K.

    rho0 is the surface density, `compression` = V0/V at P, alpha = alpha0 + alpha1 T is the
    thermal expansivity and ag the Anderson-Gruneisen parameter delta_T.
    """
    # alpha(P, T) / alpha(T), eq. 5
    alpha_ratio = compression * np.exp((ag + 1.0) * (1.0 / compression - 1.0))
    # Integral of alpha(T) from the surface temperature (273.15 K) to T
    expansion = alpha0 * (T - C2K) + 0.5 * alpha1 * (T**2 - C2K**2)
    # rho(P) = rho0 V0/V (eq. 4), reduced by thermal expansion (eq. 6)
    return rho0 * compression * (1.0 - alpha_ratio * expansion)


class HoggardMantleModel(Model):
    """Mantle (olivine) properties that depend on temperature and pressure (Hoggard et al., 2026).

    - alpha: forsterite, alpha0 + alpha1 T (Bouhifd et al., 1996, Table 2).
    - cp: Korenaga & Korenaga (2016, eq. 2), with no pressure dependence (very weak over upper-mantle
      pressures; Hofmeister, 2007).
    - rho: Grose & Afonso (2013, eqs. 4-6), using `compression_mantle` and delta_T = 6 for Fo90
      (Grose & Afonso, 2013, para. 11, after Anderson & Isaak, 1995).
    - k: lattice diffusivity of olivine (Korenaga & Korenaga, 2016, eq. 3, from Grose & Afonso,
      2013, eq. 7, fitted to Pertermann & Hofmeister, 2006) times rho cp, with the pressure
      dependence d(ln k)/dP = K'/K, K' = 4, for hard mantle phases (Hofmeister, 2007, eq. 11) as in
      Korenaga & Korenaga (2016, eq. 4); plus radiative conductivity for the given grain size
      (Grose & Afonso, 2013, eqs. A1-A2, after Hofmeister, 2005).
    """

    def __init__(self, rho=3.3e3, grain_size=0.3, bulk_modulus=130e9):
        self.rho0 = rho

        # Thermal expansivity, alpha = alpha0 + alpha1 T (Bouhifd et al., 1996, Table 2, "this work", 300-2150 K)
        self.alpha0 = 2.832e-5
        self.alpha1 = 0.758e-8

        # Heat capacity, cp = c0 - c1 T^-0.5 - c2 T^-3 (Korenaga & Korenaga, 2016, eq. 2)
        self.c0 = 1580.0
        self.c1 = 12230.0
        self.c2 = 1694e6

        # d(ln k)/dP = K'/K with K' = 4 for lattice conductivity (Hofmeister, 2007, eq. 11)
        self.dlogk_dP = 4.0 / bulk_modulus

        # Lattice thermal diffusivity a, b, c, d, e (Korenaga & Korenaga, 2016, eq. 3)
        self.kappa0 = 0.565
        self.kappa1 = 0.67
        self.kappa2 = 590.0
        self.kappa3 = 1.4
        self.kappa4 = 135.0

        # Radiative conductivity coefficients for grain size d in cm (Grose & Afonso, 2013, eq. A2). The
        # printed d0.315, d0.5 and d0.94 are read as powers of d (superscripts lost in typesetting).
        d = grain_size
        self.Ar = 1.8 * (1.0 - np.exp(-(d**1.3) / 0.15)) - (1.0 - np.exp(-np.sqrt(d) / 5.0))
        self.Br = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-(d**3) / 10.0)
        self.Ta = 490.0 + 1850.0 * np.exp(-(d**0.315) / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.Tb = 2700.0 + 9000.0 * np.exp(-np.sqrt(d) / 0.205)
        self.xA = 167.5 + 505.0 * np.exp(-np.sqrt(d) / 0.85)
        self.xB = 465.0 + 1700.0 * np.exp(-(d**0.94) / 0.175)

        # Anderson-Gruneisen parameter delta_T for Fo90 (Grose & Afonso, 2013, para. 11)
        self.ag = 6.0

    def alpha(self, T):
        return self.alpha0 + self.alpha1 * T

    def cp(self, T):
        return self.c0 - self.c1 / np.sqrt(T) - self.c2 / T**3

    def rho(self, P, T):
        return _density(self.rho0, compression_mantle(P), self.alpha0, self.alpha1, self.ag, P, T)

    def kappa_lattice(self, T):
        """Lattice thermal diffusivity [m^2/s] (Korenaga & Korenaga, 2016, eq. 3), a function of T - 273."""
        return (
            self.kappa0 + self.kappa1 * np.exp((C2K - T) / self.kappa2) + self.kappa3 * np.exp((C2K - T) / self.kappa4)
        ) * 1e-6

    def k_radiative(self, T):
        """Radiative conductivity [W/m/K] (Grose & Afonso, 2013, eq. A1), with T in Celsius.

        The second term uses (T - TB)^2: the printed (T - TB) is a typo. No pressure dependence.
        """
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


class HoggardCrustModel(Model):
    """Crustal properties that depend on temperature and pressure (Hoggard et al., 2026).

    - alpha: alpha0 + alpha1 T (Stephenson et al., 2024, eq. 3 and Table 1).
    - thermal diffusivity and cp of average crust: Whittington et al. (2009, eqs. 1-4), with a change
      of fit at the quartz alpha-beta transition (846 K).
    - rho: Grose & Afonso (2013, eqs. 4-6), using `compression_crust`.
    - k: diffusivity times rho cp.
    """

    def __init__(self, rho=2.7e3):
        self.rho0 = rho

        # Thermal expansivity, alpha = alpha0 + alpha1 T (Stephenson et al., 2024, Table 1)
        self.alpha0 = 1.0e-5
        self.alpha1 = 2.9e-8

        # Anderson-Gruneisen parameter delta_T, as for the mantle (Grose & Afonso, 2013, para. 11)
        self.ag = 6.0

    def kappa(self, T):
        """Thermal diffusivity [m^2/s] (Whittington et al., 2009, eq. 1 below 846 K, eq. 2 above)."""
        if np.isscalar(T):
            kappa = 567.3 / T - 0.062 if T < 846.0 else 0.732 - 0.000135 * T
            return kappa / 1e6
        return [self.kappa(t) for t in T]

    def cp(self, T):
        """Specific heat capacity [J/kg/K] (Whittington et al., 2009, eq. 3 below 846 K, eq. 4 above).

        The equations give J/mol/K for a mean molar mass of 221.78 g/mol. Eq. 3 uses 5.0e6 T^-2, as in
        their correction (printed as 5.0e-6).
        """
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


# Models available for each layer
CRUST_MODELS = ("constant", "hasterok", "hoggard")
MANTLE_MODELS = ("constant", "osako", "hoggard")


def make_model(name, layer, alpha, cp, rho, k, **kwargs):
    """Build a model by name (case-insensitive) for layer "crust" or "mantle".

    CRUST_MODELS and MANTLE_MODELS list the names allowed for each layer.
    alpha, cp, rho and k are the constant values, used where the model does not compute them (for
    the Hasterok model, k is the room-temperature conductivity k0). "hoggard" gives the crustal or
    mantle model of Hoggard et al. (2026), depending on the layer; kwargs (grain_size,
    bulk_modulus) are passed to the mantle version.
    """
    name = name.lower()
    allowed = CRUST_MODELS if layer == "crust" else MANTLE_MODELS
    if name not in allowed:
        raise ValueError(f"Unknown {layer} model '{name}' (choose from {', '.join(allowed)})")
    if name == "constant":
        return ConstantModel(alpha, cp, rho, k)
    if name == "osako":
        return OsakoModel(alpha, cp, rho)
    if name == "hasterok":
        return HasterokModel(alpha, cp, rho, k)
    if layer == "mantle":
        return HoggardMantleModel(rho, **kwargs)
    return HoggardCrustModel(rho)
