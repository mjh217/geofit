"""Forward model: the mantle adiabat and a conductive geotherm for given parameters.

The geotherm is integrated downwards from the surface, through the crust and then the mantle,
until it meets the adiabat at the base of the mechanical boundary layer (MBL), i.e. the mechanical
lithosphere-asthenosphere boundary. Below that it follows the adiabat. Depths are in m, pressures
in Pa, temperatures in K and heat flux in W/m^2.
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

from .defaults import P_SURFACE, T_SURFACE


class NegativeFluxError(Exception):
    """Heat flux falls to zero before the geotherm meets the adiabat: the surface heat flux is too low."""


class NoAdiabatError(Exception):
    """The geotherm does not meet the adiabat above the maximum depth."""


class AdiabatInCrustError(Exception):
    """The geotherm reaches the adiabat within the crust, so the lithosphere would be thinner than the crust."""


class IntegrationError(Exception):
    """The mantle integration stopped for an unexpected reason."""


def gravity(z):
    """Gravitational acceleration [m/s^2] at depth z [m]: a quadratic fit to 80 km and linear below, valid to about 400 km."""
    z_km = np.asarray(z / 1.0e3)
    ratio = np.empty_like(z_km)
    shallow = z_km <= 80.0
    ratio[shallow] = -4.14511244e-7 * z_km[shallow] ** 2 + 1.01825756e-4 * z_km[shallow] + 1.0
    ratio[~shallow] = 3.63281500e-5 * z_km[~shallow] + 1.00258694
    return 9.81 * ratio


def gravity_constant(z):
    """Constant gravitational acceleration [m/s^2]."""
    return 9.81


class Adiabat:
    """Interpolated mantle adiabat: depth, pressure and temperature on a common profile."""

    def __init__(self, z, T, P):
        self.Pz = scipy.interpolate.interp1d(P, z, kind="linear")  # depth at a pressure
        self.zP = scipy.interpolate.interp1d(z, P, kind="linear")  # pressure at a depth
        self.PT = scipy.interpolate.interp1d(P, T, kind="cubic")  # temperature at a pressure

    def T(self, P):
        """Adiabatic temperature [K] at pressure P [Pa]."""
        return self.PT(P)

    def tail(self, z0, z_max, P0):
        """The adiabat below a point at depth z0 and pressure P0, sampled every km down to z_max.

        Depths below z0 are measured along the adiabat from the depth at which it reaches P0, so
        the tail starts at the pressure of the point it joins.
        """
        z = np.linspace(int(z0 / 1e3) * 1e3 + 1e3, int(z_max / 1e3) * 1e3, int(z_max / 1e3) - int(z0 / 1e3))
        P = np.empty(z.size)
        T = np.empty(z.size)
        z_adiabat0 = self.Pz(P0)
        for i in range(z.size):
            P[i] = self.zP(z[i] - z0 + z_adiabat0)
            T[i] = self.PT(P[i])
        return z, T, P


def compute_adiabat(model, g, tp, P0, z_max):
    """Integrate the adiabat for a mantle model down to z_max.

    dT/dz = alpha g T / cp and dP/dz = rho g, starting from T = tp (the potential temperature) and
    P = P0 at the surface. Computed once per potential temperature and used to find where each
    geotherm meets the adiabat.
    """

    def derivatives(z, y):
        T, P = y
        alpha, cp, rho, k = model.compute(z, P, T)
        g_z = g(z)
        return [alpha * g_z * T / cp, rho * g_z]

    sol = scipy.integrate.solve_ivp(derivatives, (0.0, z_max), [tp, P0], dense_output=True)
    return Adiabat(sol.t, sol.y[0], sol.y[1])


def smooth_corner(z, T, P, z_mbl, width, npoints=16):
    """Round the corner where the geotherm meets the adiabat.

    Temperature over a window of the given width centred on z_mbl is replaced by a cubic spline
    through the profile either side, so that T is smooth in z. P is interpolated linearly at the
    new points. If the window extends below the profile, the last segment is extrapolated.
    """
    if width <= 0.0:
        return z, T, P

    z_top = z_mbl - width / 2.0
    z_bot = z_mbl + width / 2.0
    beyond_profile = z_bot > z[-1]
    if beyond_profile:
        T_top = np.interp([z_top], z, T)[0]
        P_top = np.interp([z_top], z, P)[0]
        T_bot = T[-2] + (z_bot - z[-2]) * (T[-1] - T[-2]) / (z[-1] - z[-2])
        P_bot = P[-2] + (z_bot - z[-2]) * (P[-1] - P[-2]) / (z[-1] - z[-2])
    else:
        T_top, T_bot = np.interp([z_top, z_bot], z, T)
        P_top, P_bot = np.interp([z_top, z_bot], z, P)

    above = z <= z_top
    z_above = np.concatenate((z[above], [z_top]))
    T_above = np.concatenate((T[above], [T_top]))
    P_above = np.concatenate((P[above], [P_top]))

    below = z >= z_bot
    z_below = np.concatenate(([z_bot], z[below]))
    T_below = np.concatenate(([T_bot], T[below]))
    P_below = np.concatenate(([P_bot], P[below]))

    z_mid = np.linspace(z_top, z_bot, npoints + 2)[1:-1]
    spline = scipy.interpolate.CubicSpline(np.concatenate((z_above, z_below)), np.concatenate((T_above, T_below)))
    T_mid = spline(z_mid)
    if beyond_profile:
        P_mid = scipy.interpolate.interp1d(z, P, kind="linear", bounds_error=False, fill_value="extrapolate")(z_mid)
    else:
        P_mid = np.interp(z_mid, z, P)

    return (
        np.concatenate((z_above, z_mid, z_below)),
        np.concatenate((T_above, T_mid, T_below)),
        np.concatenate((P_above, P_mid, P_below)),
    )


def compute_geotherm(crust_model, mantle_model, g, z_crust, z_max, h_crust, h_mantle, surface_heat_flux, adiabat, alphaD=0.0):
    """Conductive geotherm for the given parameters, joined to the adiabat below the MBL.

    Integrates dT/dz = q / k, dP/dz = rho g and dq/dz = -H downwards from the surface (T = 0 C,
    P = 1 atm, q = surface_heat_flux): first through the crust (thickness z_crust, heat production
    h_crust [W/m^3]), then through the mantle (heat production h_mantle) until T reaches the
    adiabat at depth z_mbl. Below that the profile follows the adiabat down to z_max. If alphaD > 0,
    the corner at z_mbl is smoothed over a width of alphaD * z_mbl (the thermal boundary layer).

    Returns depth, temperature and pressure every km, and z_mbl. Raises AdiabatInCrustError if T
    reaches the adiabat within the crust, NegativeFluxError if the heat flux reaches zero first, and
    NoAdiabatError if the adiabat is not reached above z_max.
    """

    def derivatives(model, heat_production):
        def f(z, y):
            T, P, q = y
            alpha, cp, rho, k = model.compute(z, P, T)
            return [q / k, rho * g(z), -heat_production]

        return f

    def flux_zero(z, y):
        return y[2]

    def reaches_adiabat(z, y):
        return adiabat.T(y[1]) - y[0]

    flux_zero.terminal = True
    reaches_adiabat.terminal = True

    # Crust, stopping if T reaches the adiabat (otherwise T runs away, e.g. for a very high surface
    # heat flux, and the material models are evaluated at impossible temperatures)
    sol_crust = scipy.integrate.solve_ivp(
        derivatives(crust_model, h_crust),
        (0.0, z_crust),
        [T_SURFACE, P_SURFACE, surface_heat_flux],
        dense_output=True,
        first_step=1.0e3,
        max_step=5.0e3,
        events=[reaches_adiabat],
    )
    if sol_crust.status == 1:
        raise AdiabatInCrustError("The geotherm reaches the adiabat within the crust")
    if sol_crust.y[2][-1] <= 0.0:
        raise NegativeFluxError("Heat flux is negative at the base of the crust")

    # Mantle, stopping when the heat flux reaches zero or T reaches the adiabat

    sol_mantle = scipy.integrate.solve_ivp(
        derivatives(mantle_model, h_mantle),
        (z_crust, z_max),
        [sol_crust.y[0][-1], sol_crust.y[1][-1], sol_crust.y[2][-1]],
        dense_output=True,
        first_step=1e3,
        max_step=5e3,
        events=[flux_zero, reaches_adiabat],
    )
    if sol_mantle.status == 0:
        raise NoAdiabatError()
    if sol_mantle.status != 1:
        raise IntegrationError(f"Mantle integration failed: {sol_mantle.message}")
    if sol_mantle.t_events[0].size > 0:
        raise NegativeFluxError("Heat flux goes negative in the mantle")
    if sol_mantle.t_events[1].size != 1:
        raise IntegrationError(f"Unexpected events: {sol_mantle.t_events}")

    z_mbl = sol_mantle.t[-1]

    # Sample the crust and mantle solutions every km (each integration's end point is dropped as
    # it duplicates the next start), then add the adiabat below
    crust_km = int(sol_crust.t[-1] / 1e3)
    mantle_km = int(sol_mantle.t[-1] / 1e3)
    z_c = np.linspace(0.0, crust_km * 1e3, crust_km + 1)
    z_m = np.linspace(crust_km * 1e3 + 1e3, mantle_km * 1e3, mantle_km - crust_km)
    z_a, T_a, P_a = adiabat.tail(sol_mantle.t[-1], z_max, sol_mantle.y[1][-1])

    z = np.concatenate((z_c, z_m, z_a))
    T = np.concatenate((sol_crust.sol(z_c)[0], sol_mantle.sol(z_m)[0], T_a))
    P = np.concatenate((sol_crust.sol(z_c)[1], sol_mantle.sol(z_m)[1], P_a))

    if alphaD > 0.0:
        z, T, P = smooth_corner(z, T, P, z_mbl, alphaD * z_mbl)

    return z, T, P, z_mbl
