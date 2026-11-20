"""Priors, likelihood and posterior for the geotherm inversion.

The model vector x holds, in order: surface heat flux [W/m^2], crustal thickness [km] and crustal
heat production [W/m^3], then any optional parameters in the order mantle heat production
[W/m^3], potential temperature [K] and data uncertainty in pressure [Pa].
"""

import numpy as np
import scipy.interpolate
import scipy.stats

from . import forward

LOG_SQRT_2PI = 0.9189385332046727  # ln(sqrt(2 pi))


def gaussian_prior(mean, std):
    return scipy.stats.norm(loc=mean, scale=std)


def truncated_gaussian_prior(lower, upper, mean, std):
    """Gaussian prior cut off at lower and upper, given in standard deviations from the mean."""
    return scipy.stats.truncnorm(lower, upper, loc=mean, scale=std)


def uniform_prior(vmin, vmax):
    return scipy.stats.uniform(loc=vmin, scale=vmax - vmin)


def make_likelihood(crust_model, mantle_model, g, z_max, alphaD, adiabat, h_mantle, sigma, P_data, T_data):
    """Log-likelihood of the P-T data for a model vector x.

    For each datum, the misfit is the difference between its pressure and the pressure of the
    geotherm at its temperature, with Gaussian uncertainty sigma [Pa]. Models whose geotherm
    cannot be computed (e.g. the heat flux runs out) have zero likelihood. h_mantle, sigma and
    adiabat are fixed here, but can be replaced for parameters that are inverted for (see the
    `add_*` functions below).
    """

    def log_likelihood(x, h_mantle=h_mantle, sigma=sigma, adiabat=adiabat):
        surface_heat_flux = x[0]
        z_crust = x[1] * 1.0e3
        h_crust = x[2]
        if h_crust < 0.0:
            raise ValueError("Crustal heat production is negative")
        if z_crust < 0.0:
            return -np.inf

        try:
            Z, T, P, z_mbl = forward.compute_geotherm(
                crust_model, mantle_model, g, z_crust, z_max, h_crust, h_mantle, surface_heat_flux, adiabat, alphaD
            )
        except (
            forward.AdiabatInCrustError,
            forward.NegativeFluxError,
            forward.NoAdiabatError,
            forward.IntegrationError,
            ValueError,
        ):
            return -np.inf

        residual = P_data - scipy.interpolate.interp1d(T, P, kind="cubic")(T_data)
        return -0.5 * np.sum((residual / sigma) ** 2) - np.log(sigma) * P_data.size - LOG_SQRT_2PI * P_data.size

    return log_likelihood


# Each optional parameter is appended to the model vector. These wrappers take it off the end and
# pass it to the likelihood, so they are applied in the same order as the parameters were added.


def add_mantle_heat_production(log_likelihood):
    def wrapped(x, **kwargs):
        kwargs["h_mantle"] = x[-1]
        return log_likelihood(x[:-1], **kwargs)

    return wrapped


def add_potential_temperature(log_likelihood, mantle_model, g, P0, z_max):
    """The adiabat is recomputed for each potential temperature."""

    def wrapped(x, **kwargs):
        kwargs["adiabat"] = forward.compute_adiabat(mantle_model, g, x[-1], P0, z_max)
        return log_likelihood(x[:-1], **kwargs)

    return wrapped


def add_data_uncertainty(log_likelihood):
    """Hierarchical inversion: the data uncertainty sigma is itself a parameter."""

    def wrapped(x, **kwargs):
        kwargs["sigma"] = x[-1]
        return log_likelihood(x[:-1], **kwargs)

    return wrapped


def make_log_posterior(priors, log_likelihood):
    """Log-posterior: sum of the log-priors plus the log-likelihood (-inf outside the priors)."""

    def log_posterior(x):
        log_prior = 0.0
        for xi, prior in zip(x, priors):
            lp = prior.logpdf(xi)
            if not np.isfinite(lp):
                return -np.inf
            log_prior += lp

        ll = log_likelihood(x)
        if not np.isfinite(ll):
            return -np.inf
        return log_prior + ll

    return log_posterior
