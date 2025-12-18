import numpy as np
import scipy.stats

import forwardmodel


def mk_gaussian_prior(mu, std):
    return scipy.stats.norm(loc=mu, scale=std)

def sample_gaussian_prior(mu, std, nosamples):
    return scipy.stats.norm.rvs(loc=mu, scale=std, size=nosamples)


def mk_truncated_gaussian_prior(trunc_factor_min, trunc_factor_max, mu, std):
    return scipy.stats.truncnorm(trunc_factor_min, trunc_factor_max, loc=mu, scale=std)

def sample_truncated_gaussian_prior(trunc_factor_min, trunc_factor_max, mu, std, nosamples):
    return scipy.stats.truncnorm.rvs(trunc_factor_min, trunc_factor_max, loc=mu, scale=std, size=nosamples)


def mk_uniform_prior(vmin, vmax):
    return scipy.stats.uniform(loc=vmin, scale=vmax - vmin)

def sample_uniform_prior(vmin, vmax, nosamples):
    return scipy.stats.uniform.rvs(loc=vmin, scale=vmax - vmin, size=nosamples)



def likelihood_base(
    crust_model,
    mantle_model,
    g,
    z_max,
    alphaD,
    _adiabat,
    _h_mantle,
    _sigma,
    Pdata,
    Tdata,
):
    def loglikelihood(x, h_mantle=_h_mantle, sigma=_sigma, adiabat=_adiabat):
        surface_heat_flux = x[0]
        z_crust = x[1] * 1.0e3
#        if x[2] > 0.0:
#            raise Exception("Values out of range")
#        h_crust = 10 ** x[2]

        if x[2] < 0.0:
            raise Exception("Radiogenic heat production is negative.")
        h_crust = x[2]

        if z_crust < 0.0:
            return -np.inf

        try:
            Z, T, P, z_mbl = forwardmodel.compute_geotherm(
                crust_model,
                mantle_model,
                g,
                z_crust,
                z_max,
                h_crust,
                h_mantle,
                surface_heat_flux,
                adiabat,
                alphaD,
            )

        except forwardmodel.NegativeFluxException:
            return -np.inf
        except forwardmodel.NoAdiabatException:
            return -np.inf
        except ValueError:
            return -np.inf

        # PT = scipy.interpolate.interp1d(P, T, kind="cubic")
        # TP = scipy.interpolate.interp1d(T, P, kind="cubic")

        # residual_T = (Tdata - PT(Pdata)) / 3.0
        # residual_P = (Pdata - TP(Tdata)) / 3e8
        # residual = np.hstack((residual_T, residual_P))

        # return (
        #     -0.5 * np.sum((residual) ** 2)
        #     # - np.log(sigma) * Pdata.size
        #     # - 0.9189385332046727
        # )

        # PT = scipy.interpolate.interp1d(P, T, kind="cubic")
        TP = scipy.interpolate.interp1d(T, P, kind="cubic")

        # residual = Tdata - PT(Pdata)
        residual = Pdata - TP(Tdata)

        return (
            -0.5 * np.sum((residual / sigma) ** 2)
            - np.log(sigma) * Pdata.size
            - 0.9189385332046727
        )

    return loglikelihood


def likelihood_augment_hmantle(base_likelihood):
    def augmented_likelihood(x, **kwargs):
        hmantle = x[-1]
        x = x[:-1]

        kwargs["h_mantle"] = hmantle
        return base_likelihood(x, **kwargs)

    return augmented_likelihood


def likelihood_augment_tp(base_likelihood, mantle_model, g, P0, z_max):
    def augmented_likelihood(x, **kwargs):
        tp = x[-1]
        x = x[:-1]

        adiabat = forwardmodel.compute_adiabat(mantle_model, g, tp, P0, z_max)

        kwargs["adiabat"] = adiabat
        return base_likelihood(x, **kwargs)

    return augmented_likelihood


def likelihood_augment_hierarchical(base_likelihood):
    def augmented_likelihood(x, **kwargs):
        sigma = x[-1]
        x = x[:-1]

        kwargs["sigma"] = sigma
        return base_likelihood(x, **kwargs)

    return augmented_likelihood


def make_log_posterior(priors, likelihood):
    def log_posterior(x):
        log_prior = 0.0
        for xi, prior in zip(x, priors):
            lp = prior.logpdf(xi)
            if np.isfinite(lp):
                log_prior += lp
            else:
                return -np.inf

        ll = likelihood(x)
        if np.isfinite(ll):
            return log_prior + ll
        else:
            return -np.inf

    return log_posterior


def randomly_interpolate_model(models):
    rows, cols = models.shape
    if rows == 1:
        return models
    else:
        factors = np.random.dirichlet(np.ones((rows,)))
        return factors.dot(models)


def resample(models, loglikelihoods, nwalkers):
    order = np.argsort(loglikelihoods)[::-1]
    olike = loglikelihoods[order]

    i = 2
    z = 0.0

    while i < olike.size:
        mu = np.mean(olike[:i])
        std = np.std(olike[:i])

        z = np.abs(olike[i] - mu) / std
        if z < 20.0:
            i += 1
        else:
            break

    if i >= nwalkers:
        # Nothing to do
        print("Sufficient number of good initial walkers found.")
        return models[order[:nwalkers], :]
    else:
        good = order[:i]
        bad = order[i:]

        print("%d good walkers, %d bad walkers." % (good.size, bad.size))

        for j in bad:
            models[j, :] = randomly_interpolate_model(models[good, :])

        return models[order[:nwalkers], :]
