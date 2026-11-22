"""Thermal and density properties of the crust and mantle.

Each property has its own parameterisations below: thermal expansivity alpha [1/K], specific heat
capacity cp [J/kg/K], thermal diffusivity kappa [m^2/s], thermal conductivity k [W/m/K] and density
rho [kg/m^3], as functions of pressure P [Pa] and temperature T [K]. A `Model` combines one of each
for the crust or the mantle; `make_model` builds the models available from the command line.
"""

import numpy as np

from .defaults import C2K

# --------------------------------------------------------------------------
# Thermal expansivity
# --------------------------------------------------------------------------


class LinearExpansivity:
    """alpha = alpha0 + alpha1 T, with T in K."""

    def __init__(self, alpha0, alpha1):
        self.alpha0 = alpha0
        self.alpha1 = alpha1

    def __call__(self, T):
        return self.alpha0 + self.alpha1 * T

    def integral(self, T):
        """Integral of alpha from the surface temperature (273.15 K) to T."""
        return self.alpha0 * (T - C2K) + 0.5 * self.alpha1 * (T**2 - C2K**2)


# Forsterite (Bouhifd et al., 1996, Table 2, "this work", 300-2150 K)
ALPHA_FORSTERITE = LinearExpansivity(2.832e-5, 0.758e-8)

# Continental crust (Stephenson et al., 2024, eq. 3 and Table 1)
ALPHA_CRUST = LinearExpansivity(1.0e-5, 2.9e-8)


# --------------------------------------------------------------------------
# Specific heat capacity
# --------------------------------------------------------------------------


def cp_mantle(T):
    """Mantle heat capacity (Korenaga & Korenaga, 2016, eq. 2): cp = 1580 - 12230 T^-0.5 - 1694e6 T^-3.

    No pressure dependence, which is very weak over upper-mantle pressures (Hofmeister, 2007).
    """
    return 1580.0 - 12230.0 / np.sqrt(T) - 1694e6 / T**3


def cp_crust(T):
    """Heat capacity of average crust (Whittington et al., 2009, eq. 3 below 846 K, eq. 4 above).

    The equations give J/mol/K for a mean molar mass of 221.78 g/mol; Hoggard et al. (2026, eq. 23)
    give them per kg. Eq. 3 uses 5.0e6 T^-2, as in
    their correction (printed as 5.0e-6).
    """
    if np.isscalar(T):
        if T < 846.0:
            return (199.5 + 0.0857 * T - 5e6 / T**2) / 221.78e-3
        return (229.32 + 0.0323 * T - 47.9e-6 / T**2) / 221.78e-3
    return [cp_crust(t) for t in T]


# --------------------------------------------------------------------------
# Thermal diffusivity
# --------------------------------------------------------------------------


def kappa_olivine(T):
    """Lattice diffusivity of olivine (Korenaga & Korenaga, 2016, eq. 3, from Grose & Afonso, 2013, eq. 7).

    kappa = [a + b exp(-(T - 273) / c) + d exp(-(T - 273) / e)] 1e-6, with a = 0.565, b = 0.67,
    c = 590, d = 1.4 and e = 135, fitted to Fo90 data of Pertermann & Hofmeister (2006).
    """
    return (0.565 + 0.67 * np.exp((C2K - T) / 590.0) + 1.4 * np.exp((C2K - T) / 135.0)) * 1e-6


def kappa_crust(T):
    """Diffusivity of average crust (Whittington et al., 2009, eq. 1 below 846 K, eq. 2 above).

    Hoggard et al. (2026, eq. 24) give these in m^2/s.
    """
    if np.isscalar(T):
        kappa = 567.3 / T - 0.062 if T < 846.0 else 0.732 - 0.000135 * T
        return kappa / 1e6
    return [kappa_crust(t) for t in T]


# --------------------------------------------------------------------------
# Thermal conductivity
# --------------------------------------------------------------------------


def k_osako(P, T):
    """Olivine conductivity: k = (1.175 + 1264 / T) exp(0.038 P), with T in K and P in GPa.

    The average over the three crystal axes of the olivine fits of Osako et al. (2004, Table 2),
    lambda = C0 + C1/T = 1.61 + 1732/T measured at 8.3 GPa, reduced to 0 GPa with their average
    pressure dependence lambda = B0 exp(B1 P), B1 = 0.038 / GPa.
    """
    return (1.175 + 1264.0 / T) * np.exp(0.038 * P / 1e9)


class HasterokConductivity:
    """Empirical conductivity of crustal rocks (Goes et al., 2020, section 3.2, unnumbered equations on p. 6).

    k = k1 + k2,  k1 = k0 (n - 1) / n (1 + beta P),  k2 = k0 exp(-(T - 25) / 300) / n,
    with T in Celsius, P in GPa, k0 the room-temperature conductivity, beta = 0.1 / GPa and
    n = 6.4 - 2.3 log k0 (taken as the natural log). k2 is printed without the 1/n; it is included
    here so that k = k0 at 25 C and 0 GPa. At high temperature k tends to k0 (n - 1) / n.
    """

    def __init__(self, k0):
        self.k0 = k0
        self.n = 6.4 - 2.3 * np.log(k0)
        self.beta = 0.1  # [1/GPa]

    def __call__(self, P, T):
        k1 = self.k0 * (self.n - 1.0) / self.n * (1.0 + self.beta * P / 1e9)
        k2 = self.k0 * np.exp(-(T - C2K - 25.0) / 300.0) / self.n
        return k1 + k2


class RadiativeConductivity:
    """Radiative conductivity of olivine for grain size d [cm] (Grose & Afonso, 2013, eqs. A1-A2).

    A Gaussian fit in temperature (Celsius) to the model of Hofmeister (2005). In eq. A1 the second
    term uses (T - TB)^2, and in eq. A2 the printed d0.315, d0.5 and d0.94 are powers of d: both are
    typesetting errors. No pressure dependence.
    """

    def __init__(self, grain_size):
        d = grain_size
        self.A = 1.8 * (1.0 - np.exp(-(d**1.3) / 0.15)) - (1.0 - np.exp(-np.sqrt(d) / 5.0))
        self.B = 11.7 * np.exp(-d / 0.159) + 6.0 * np.exp(-(d**3) / 10.0)
        self.TA = 490.0 + 1850.0 * np.exp(-(d**0.315) / 0.825) + 875.0 * np.exp(-d / 0.18)
        self.TB = 2700.0 + 9000.0 * np.exp(-np.sqrt(d) / 0.205)
        self.xA = 167.5 + 505.0 * np.exp(-np.sqrt(d) / 0.85)
        self.xB = 465.0 + 1700.0 * np.exp(-(d**0.94) / 0.175)

    def __call__(self, T):
        return self.A * np.exp(-(((T - C2K) - self.TA) ** 2) / (2.0 * self.xA**2)) + self.B * np.exp(
            -(((T - C2K) - self.TB) ** 2) / (2.0 * self.xB**2)
        )


# --------------------------------------------------------------------------
# Density
# --------------------------------------------------------------------------


def compression_crust(P):
    """Isothermal compression V0/V of the crust at pressure P [Pa].

    Quadratic fit (Hoggard et al., 2026, eq. 22) to the third-order Birch-Murnaghan equation of state
    (Grose & Afonso, 2013, eq. 3) with K0 = 90 GPa (Stephenson et al., 2024, Table 1) and K' = 4.8;
    within 0.001 of it to 2 GPa.
    """
    return 1.00000236 + (1.10952146e-11 - 2.12421383e-22 * P) * P


def compression_mantle(P):
    """Isothermal compression V0/V of the mantle at pressure P [Pa].

    Quadratic fit (Hoggard et al., 2026, eq. 8), as `compression_crust`, with K0 = 130 GPa and K' = 4.8
    (Grose & Afonso, 2013, para. 11); within 0.001 of the equation of state to 14 GPa.
    """
    return 1.00009143 + (7.57454484e-12 - 8.00340143e-23 * P) * P


class Density:
    """Density with pressure and temperature (Grose & Afonso, 2013, eqs. 4-6).

    rho0 is the surface density, `compression` gives V0/V at P, `expansivity` is a LinearExpansivity
    and delta_T is the Anderson-Gruneisen parameter (6 for Fo90; Grose & Afonso, 2013, para. 11).
    """

    def __init__(self, rho0, compression, expansivity, delta_T=6.0):
        self.rho0 = rho0
        self.compression = compression
        self.expansivity = expansivity
        self.delta_T = delta_T

    def alpha_ratio(self, P):
        """alpha(P, T) / alpha(T) at pressure P (Grose & Afonso, 2013, eq. 5; Hoggard et al., 2026, eq. 6)."""
        compression = self.compression(P)
        return compression * np.exp((self.delta_T + 1.0) * (1.0 / compression - 1.0))

    def alpha(self, P, T):
        """Thermal expansivity alpha(P, T) [1/K]: alpha(T) of `expansivity` reduced by compression."""
        return self.expansivity(T) * self.alpha_ratio(P)

    def __call__(self, P, T):
        # rho(P) = rho0 V0/V (eq. 4), reduced by thermal expansion (eq. 6)
        return self.rho0 * self.compression(P) * (1.0 - self.alpha_ratio(P) * self.expansivity.integral(T))


# --------------------------------------------------------------------------
# Models: one parameterisation of each property
# --------------------------------------------------------------------------


class Model:
    """A crust or mantle model: alpha, cp, rho and k, each a function of (P, T)."""

    def __init__(self, alpha, cp, rho, k):
        self.alpha = alpha
        self.cp = cp
        self.rho = rho
        self.k = k

    def compute(self, depth, pressure, temperature):
        """(alpha, cp, rho, k) at depth [m], pressure [Pa] and temperature [K], shaped like the inputs."""
        ones = _ones(depth, pressure, temperature)
        P, T = pressure, temperature
        return self.alpha(P, T) * ones, self.cp(P, T) * ones, self.rho(P, T) * ones, self.k(P, T) * ones


def _ones(*values):
    """1.0 for scalar inputs, or an array of ones as long as the longest array input (so that
    constant properties take the shape of the inputs)."""
    size = 0
    for value in values:
        if not isinstance(value, (int, float)):
            size = max(size, len(value))
    return 1.0 if size == 0 else np.ones(size)


def _constant(value):
    return lambda P, T: value


def constant_model(alpha, cp, rho, k):
    """All four properties constant."""
    return Model(_constant(alpha), _constant(cp), _constant(rho), _constant(k))


def osako_model(alpha, cp, rho):
    """Mantle: constant alpha, cp and rho, with the olivine conductivity of Osako et al. (2004)."""
    return Model(_constant(alpha), _constant(cp), _constant(rho), k_osako)


def hasterok_model(alpha, cp, rho, k0):
    """Crust: constant alpha, cp and rho, with the conductivity of Goes et al. (2020) for k0 at room temperature."""
    return Model(_constant(alpha), _constant(cp), _constant(rho), HasterokConductivity(k0))


def hoggard_crust_model(rho0=2.7e3):
    """Crustal model of Hoggard et al. (2026).

    alpha(T) of continental crust (Stephenson et al., 2024) with pressure dependence alpha(P, T)
    (Hoggard et al., 2026, eq. 6); diffusivity and cp of average crust (Whittington et al., 2009);
    density (Grose & Afonso, 2013) with K0 = 90 GPa; k = kappa rho cp (Hoggard et al., 2026, eq. 11).
    """
    rho = Density(rho0, compression_crust, ALPHA_CRUST)

    def k(P, T):
        return kappa_crust(T) * rho(P, T) * cp_crust(T)

    return Model(rho.alpha, lambda P, T: cp_crust(T), rho, k)


def hoggard_mantle_model(rho0=3.3e3, grain_size=0.3, bulk_modulus=130e9):
    """Mantle model of Hoggard et al. (2026).

    alpha(T) of forsterite (Bouhifd et al., 1996) with pressure dependence alpha(P, T) (Hoggard et
    al., 2026, eq. 6), which sets the adiabat; cp (Korenaga & Korenaga, 2016); density (Grose &
    Afonso, 2013) with K0 = 130 GPa; k = kappa rho cp exp(P K'/K) + k_rad (Hoggard et al., 2026,
    eqs. 13 and 21), from the olivine lattice
    diffusivity, the pressure dependence d(ln k)/dP = K'/K with K' = 4 for hard mantle phases
    (Hofmeister, 2007, eq. 11; as in Korenaga & Korenaga, 2016, eq. 4) and the radiative
    conductivity for the grain size [cm] (Grose & Afonso, 2013).
    """
    rho = Density(rho0, compression_mantle, ALPHA_FORSTERITE)
    radiative = RadiativeConductivity(grain_size)
    dlogk_dP = 4.0 / bulk_modulus

    def k(P, T):
        return kappa_olivine(T) * rho(P, T) * cp_mantle(T) * np.exp(dlogk_dP * P) + radiative(T)

    return Model(rho.alpha, lambda P, T: cp_mantle(T), rho, k)


# Models available for each layer
CRUST_MODELS = ("constant", "hasterok", "hoggard")
MANTLE_MODELS = ("constant", "osako", "hoggard")


def make_model(name, layer, alpha, cp, rho, k, **kwargs):
    """Build a model by name (case-insensitive) for layer "crust" or "mantle".

    CRUST_MODELS and MANTLE_MODELS list the names allowed for each layer. alpha, cp, rho and k are
    the constant values, used where the model does not compute them (for the Hasterok model, k is
    the room-temperature conductivity k0); rho is also the surface density of the Hoggard models.
    kwargs (grain_size, bulk_modulus) are passed to the Hoggard mantle model.
    """
    name = name.lower()
    allowed = CRUST_MODELS if layer == "crust" else MANTLE_MODELS
    if name not in allowed:
        raise ValueError(f"Unknown {layer} model '{name}' (choose from {', '.join(allowed)})")
    if name == "constant":
        return constant_model(alpha, cp, rho, k)
    if name == "osako":
        return osako_model(alpha, cp, rho)
    if name == "hasterok":
        return hasterok_model(alpha, cp, rho, k)
    if layer == "mantle":
        return hoggard_mantle_model(rho, **kwargs)
    return hoggard_crust_model(rho)
