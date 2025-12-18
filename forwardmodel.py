import numpy as np
import scipy.integrate
import scipy.interpolate


def little_g(z):
    """
    Returns acceleration due to gravity, g, as a function of depth, z, in metres using a
    polynomial approximation. Only accurate up to around 0 - 400km depth
    """

    def rg(z):
        z = np.asarray(z)
        result = np.empty_like(z)

        mask_shallow = z <= 80e3
        result[mask_shallow] = (
            -4.14511244e-7 * z[mask_shallow] ** 2
            + 1.01825756e-4 * z[mask_shallow]
            + 1.0
        )
        result[~mask_shallow] = 3.63281500e-5 * z[~mask_shallow] + 1.00258694

        return result.item() if isinstance(z, (int, float)) else result

    return 9.81 * rg(z / 1.0e3)


def little_g_const(z):
    """
    Return constant acceleration due to gravity (for comparison)
    """
    return 9.81


def spline_smooth_elbow(z, T, P, zMBL, D, npoints=16):
    """
    Use a spline interpolant to smooth the corner at zMBL where the geotherm intersects with the
    adiabat. Resulting points represent a C1 smooth set of points for T as a function of z. P is
    reinterpolated at the new points linearly.
    """

    if D <= 0.0:
        return z, T, P

    z_top = zMBL - D / 2.0
    z_bot = zMBL + D / 2.0
    if z_bot > z[-1]:
        T_top = np.interp([z_top], z, T)[0]
        P_top = np.interp([z_top], z, P)[0]

        # Extrapolate bottom points
        T_bot = T[-2] + (z_bot - z[-2]) * (T[-1] - T[-2]) / (z[-1] - z[-2])
        P_bot = P[-2] + (z_bot - z[-2]) * (P[-1] - P[-2]) / (z[-1] - z[-2])
    else:
        T_top, T_bot = np.interp([z_top, z_bot], z, T)
        P_top, P_bot = np.interp([z_top, z_bot], z, P)

    pre_indices = np.where(z <= z_top)[0]

    z_pre = np.concatenate((z[pre_indices], [z_top]))
    T_pre = np.concatenate((T[pre_indices], [T_top]))
    P_pre = np.concatenate((P[pre_indices], [P_top]))

    post_indices = np.where(z >= z_bot)[0]

    z_post = np.concatenate(([z_bot], z[post_indices]))
    T_post = np.concatenate(([T_bot], T[post_indices]))
    P_post = np.concatenate(([P_bot], P[post_indices]))

    z_mid = np.linspace(z_top, z_bot, npoints + 2)[1:-1]

    cT = scipy.interpolate.CubicSpline(
        np.concatenate((z_pre, z_post)), np.concatenate((T_pre, T_post))
    )
    T_mid = cT(z_mid)

    if z_bot > z[-1]:
        Pext = scipy.interpolate.interp1d(
            z, P, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        P_mid = Pext(z_mid)
    else:
        P_mid = np.interp(z_mid, z, P)

    z_smooth = np.concatenate((z_pre, z_mid, z_post))
    T_smooth = np.concatenate((T_pre, T_mid, T_post))
    P_smooth = np.concatenate((P_pre, P_mid, P_post))

    P = scipy.interpolate.interp1d(P_smooth, T_smooth)

    return z_smooth, T_smooth, P_smooth


class Adiabat:
    def __init__(self, z, T, P):
        self.Pz = scipy.interpolate.interp1d(P, z, kind="linear")
        self.zP = scipy.interpolate.interp1d(z, P, kind="linear")
        self.PT = scipy.interpolate.interp1d(P, T, kind="cubic")

    def T(self, P):
        return self.PT(P)

    def get_tail(self, z0, z_max, T, P, nsteps=None):
        nsteps = int(z_max / 1e3) - int(z0 / 1e3)
        z = np.linspace(int(z0 / 1e3) * 1e3 + 1e3, int(z_max / 1e3) * 1e3, nsteps)

        zp = self.Pz(P)

        zo = np.empty((nsteps,))
        Po = np.empty((nsteps,))
        To = np.empty((nsteps,))

        for i in range(nsteps):
            az = z[i] - z0 + zp

            zo[i] = z[i]
            Po[i] = self.zP(az)
            To[i] = self.PT(Po[i])

        return zo, To, Po


def compute_adiabat(model, g, tp, P0, depth_max):
    """
    Compute a consistent relationship between pressure and adiabatic temperature for
    determining termination conditions (one of) of the acutal forward model. This needs
    to be computed once for a given Mantle Model to generate a pressure-temperature map
    to quickly evaluate when the MBL is reached.
    """

    #
    # For the adiabat, we integrate the coupled equations as a function of depth
    #
    # dTdz = alpha g T/cp
    # dPdz = rho g
    #
    # up to a maximum depth with boundary conditions at the surface T(0) = tp and P(0) = P0
    #
    def adiabat_closure(model, g):
        def dz(t, y):
            T, P = y
            alpha, cp, rho, cond = model.compute(t, P, T)

            _g = g(t)
            dTdz = alpha * _g * T / cp
            dPdz = rho * _g

            return [dTdz, dPdz]

        return dz

    sol = scipy.integrate.solve_ivp(
        adiabat_closure(model, g), (0.0, depth_max), [tp, P0], dense_output=True
    )

    #
    # We actually want a pressure-temperature cutoff so we create an interpolator for this
    #
    return Adiabat(sol.t, sol.y[0], sol.y[1])


class NegativeFluxException(Exception):
    pass


class NoAdiabatException(Exception):
    pass


def compute_geotherm(
    crust_model,
    mantle_model,
    g,
    z_crust,
    z_max,
    h_crust,
    h_mantle,
    surface_heat_flux,
    adiabat,
    alphaD=0.0,
):
    #
    # For the geotherm, we integrate the coupled equations as a function of depth until
    # we reach the MBL (pressure-temperature intersects with the adiabat). We treat the
    # crust and mantle separately (they're separate models). We terminate if the flux
    # goes negative as this indicates the Earth has become heat absorbing and indicates
    # the surface heat flux is too low to be consistent.
    #
    # dTdz = -flux/cond
    # dPdz = rho g
    # dfluxdz = -heatgen
    #
    def flux_closure(model, g, heat_gen):
        def dz(t, y):
            T, P, F = y

            alpha, cp, rho, cond = model.compute(t, P, T)

            dTdz = F / cond
            dPdz = rho * g(t)
            dFdz = -heat_gen

            return [dTdz, dPdz, dFdz]

        return dz

    #
    # Initial conditions
    #
    T0 = 273.15
    P0 = 101325.0
    F0 = surface_heat_flux

    #
    # Integrate through the crust
    #
    sol_crust = scipy.integrate.solve_ivp(
        flux_closure(crust_model, g, h_crust),
        (0.0, z_crust),
        [T0, P0, F0],
        dense_output=True,
        first_step=1.0e3,
        max_step=5.0e3,
    )

    #
    # If the flux is <= 0 at the base of the crust, terminate early with an error
    #
    if sol_crust.y[2][-1] <= 0.0:
        raise NegativeFluxException("Flux is negative at the base of the crust")

    #
    # Generate the termination conditions for integrating through the mantle.
    #
    flux_condition = lambda t, y: y[2]
    flux_condition.terminal = True

    adiabat_condition = lambda t, y: adiabat.T(y[1]) - y[0]
    adiabat_condition.terminal = True

    #
    # Integrate through the mantle
    #
    sol_mantle = scipy.integrate.solve_ivp(
        flux_closure(mantle_model, g, h_mantle),
        (z_crust, z_max),
        [sol_crust.y[0][-1], sol_crust.y[1][-1], sol_crust.y[2][-1]],
        dense_output=True,
        first_step=1e3,
        max_step=5e3,
        events=[flux_condition, adiabat_condition],
    )

    if sol_mantle.status == 0:
        # This means that neither the flux or adiabatic condition triggered
        raise NoAdiabatException()

    elif sol_mantle.status == 1:
        # Determine which event was triggered
        if len(sol_mantle.t_events) == 2:
            if sol_mantle.t_events[0].size > 0:
                raise NegativeFluxException("Flux goes negative in the mantle")

            elif sol_mantle.t_events[1].size == 1:
                #
                # Found the MBL point
                #
                z_MBL = sol_mantle.t[-1]

                #
                # Assemble the necessary profiles

                adiabat_z, adiabat_T, adiabat_P = adiabat.get_tail(
                    sol_mantle.t[-1], z_max, sol_mantle.y[0][-1], sol_mantle.y[1][-1]
                )

                #
                # The [:-1]'s here represent the fact that our integration duplicates points but
                # we don't need then (and they cause problems for interpolation) as T and P are continuous
                # w.r.t. z
                #
                z_crust = np.linspace(
                    0.0,
                    int(sol_crust.t[-1] / 1e3) * 1e3,
                    int(sol_crust.t[-1] / 1e3) + 1,
                )
                z_mantle = np.linspace(
                    int(sol_crust.t[-1] / 1e3) * 1e3 + 1e3,
                    int(sol_mantle.t[-1] / 1e3) * 1e3,
                    int(sol_mantle.t[-1] / 1e3) - int(sol_crust.t[-1] / 1e3),
                )
                z = np.concatenate((z_crust, z_mantle, adiabat_z))
                T = np.concatenate(
                    (sol_crust.sol(z_crust)[0], sol_mantle.sol(z_mantle)[0], adiabat_T)
                )
                P = np.concatenate(
                    (sol_crust.sol(z_crust)[1], sol_mantle.sol(z_mantle)[1], adiabat_P)
                )

                if alphaD > 0.0:
                    z, T, P = spline_smooth_elbow(z, T, P, z_MBL, alphaD * z_MBL)

                return z, T, P, z_MBL

            else:
                #
                # Unknown:
                #
                print("Unknown:", sol_mantle.t_events)
        else:
            print("Unexpected len:", len(sol_mantle.t_events))
            print(sol_mantle.t_events)

    else:
        print("Unexpected status:", sol_mantle.status)
