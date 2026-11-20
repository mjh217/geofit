"""MCMC sampling of the posterior with emcee.

A larger ensemble of walkers is first run for a burn-in period. The best walkers are then kept
(walkers stuck in poor regions are replaced by random mixtures of the good ones) and used to start
the main run.
"""

import numpy as np
from emcee import EnsembleSampler


def initial_walkers(priors, log_likelihood, n, max_attempts=100):
    """n starting models drawn from the priors, redrawing any with zero likelihood."""
    walkers = np.empty((n, len(priors)))
    for i, prior in enumerate(priors):
        walkers[:, i] = prior.rvs(size=n)

    for i in range(n):
        attempts = 0
        while not np.isfinite(log_likelihood(walkers[i])):
            if attempts > max_attempts:
                raise RuntimeError("Failed to generate a valid initial model.")
            for j, prior in enumerate(priors):
                walkers[i, j] = prior.rvs()
            attempts += 1
    return walkers


def keep_best_walkers(models, log_probs, n):
    """The n best of an ensemble of models, after replacing outliers.

    Models are ranked by log-probability. Going down the ranking, a model is an outlier once it is
    more than 20 standard deviations below the mean of those above it; outliers and everything below
    are replaced by random (Dirichlet-weighted) mixtures of the good models.
    """
    order = np.argsort(log_probs)[::-1]
    ranked = log_probs[order]

    n_good = 2
    while n_good < ranked.size:
        mean = np.mean(ranked[:n_good])
        std = np.std(ranked[:n_good])
        if not np.abs(ranked[n_good] - mean) / std < 20.0:  # also stops on NaN (e.g. 0/0)
            break
        n_good += 1

    if n_good >= n:
        print("Sufficient number of good initial walkers found.")
        return models[order[:n], :]

    good = order[:n_good]
    bad = order[n_good:]
    print(f"{good.size} good walkers, {bad.size} bad walkers.")
    for j in bad:
        models[j, :] = _random_mixture(models[good, :])
    return models[order[:n], :]


def _random_mixture(models):
    if models.shape[0] == 1:
        return models
    weights = np.random.dirichlet(np.ones((models.shape[0],)))
    return weights.dot(models)


def sample_posterior(log_posterior, log_likelihood, priors, nwalkers, oversampling, burnin, samples):
    """Run the burn-in and main samplers; returns the main emcee sampler."""
    ndim = len(priors)
    n_burnin_walkers = nwalkers * oversampling
    while True:
        walkers = initial_walkers(priors, log_likelihood, n_burnin_walkers)

        print("Starting initial burn-in sampling...")
        burnin_sampler = EnsembleSampler(n_burnin_walkers, ndim, log_posterior)
        state = burnin_sampler.run_mcmc(walkers, burnin)

        print(f"Select {nwalkers} of the original {n_burnin_walkers} walkers:")
        walkers = keep_best_walkers(state.coords, state.log_prob, nwalkers)

        print("Initiating main statistical sampling...")
        try:
            sampler = EnsembleSampler(nwalkers, ndim, log_posterior)
            sampler.run_mcmc(walkers, samples)
            break
        except ValueError:
            # emcee rejects starting walkers that are too close together; start again
            print("Initial samples are too close. Retrying...")

    print("Individual walker model acceptance fractions = ", sampler.acceptance_fraction)
    return sampler
