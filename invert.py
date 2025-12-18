from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emcee import EnsembleSampler

# pygeotherm1d
import bayesian as bsn
import defaults
import forwardmodel as fwd_mdl
from model import load_from_string

if __name__ == "__main__":
    parser = ArgumentParser()

    # Input data
    parser.add_argument("input", type=str, help="Input TP constraints .csv file (Celsius, GPa)")

    # Output options
    parser.add_argument("--samplepriors", action="store_true", help="Draw random samples from the prior PDFs")
    parser.add_argument("--iwrite", action="store_true", help="Write out all sampled geotherms to an individual csv file")

    # Default options
    parser.add_argument("--approxg", action="store_true", help="Use non-constant g with depth (approximation)")

    parser.add_argument("--crust", type=str, default="Constant", help="Crustal thermal parameters")
    parser.add_argument("--mantle", type=str, default="Constant", help="Mantle thermal parameters")

    # Parameters for constant thermal parameterisation models
    parser.add_argument("--crust-rho", type=float, default=defaults.rho_crust, help="Default crustal density")
    parser.add_argument("--crust-alpha", type=float, default=defaults.alpha_crust, help="Default crustal alpha")
    parser.add_argument("--crust-cp", type=float, default=defaults.cp_crust, help="Default crustal Cp")
    parser.add_argument("--crust-cond", type=float, default=defaults.k_crust, help="Default crust conductivity")

    parser.add_argument("--mantle-rho", type=float, default=defaults.rho_mantle, help="Default mantle density")
    parser.add_argument("--mantle-alpha", type=float, default=defaults.alpha_mantle, help="Default mantle alpha")
    parser.add_argument("--mantle-cp", type=float, default=defaults.cp_mantle, help="Default mantle Cp")
    parser.add_argument("--mantle-cond", type=float, default=defaults.k_mantle, help="Default mantle conductivity")

    parser.add_argument("--mantle-grain-size", type=float, default=defaults.grain_size_mantle, help="Default mantle grain size in cm")

    parser.add_argument("--crust-K", type=float, default=defaults.bulk_modulus_crust, help="Default crustal bulk modulus")
    parser.add_argument("--mantle-K", type=float, default=defaults.bulk_modulus_mantle, help="Default mantle bulk modulus")

    # Default values for other important thermal parameters
    parser.add_argument("--alphaD", type=float, default=0.2, help="TBL thickness as a fraction of thermal LAB depth (default = 0.2, disable = 0)")
    parser.add_argument("--Tp", type=float, default=defaults.Tp, help="Adiabatic temperature at surface (K)")
    parser.add_argument("--Hmantle", type=float, default=0.0, help="Mantle radiogenic heat production")
    parser.add_argument("--sigma", type=float, default=3e8, help="Data P error level in Pa")

    # Optional parameters that can be added to the inversion
    parser.add_argument("--invertHmantle", action="store_true", help="Invert for Hmantle")
    parser.add_argument("--invertTp", action="store_true", help="Invert for Tp")
    parser.add_argument("--hierarchical", action="store_true", help="Invert with hierarchical data error")

    #
    # Parameters for constructing the priors
    #

    # Surface heat flux
#    parser.add_argument("--surfaceheatfluxmean", type=float, default=0.05, help="Surface heat flux prior mean (W/m^2)")
#    parser.add_argument("--surfaceheatfluxstd", type=float, default=0.01, help="Surface heat flux prior std. dev. (W/m^2)")
    parser.add_argument("--surfaceheatfluxmin", type=float, default=0.01, help="Surface heat flux prior mean (W/m^2)")
    parser.add_argument("--surfaceheatfluxmax", type=float, default=0.25, help="Surface heat flux prior std. dev. (W/m^2)")

    # Crustal radiogenic heat production
    parser.add_argument("--hcrustmean", type=float, default=1e-6, help="Crustal heat prod. prior mean (W/m^3)")
    parser.add_argument("--hcruststd", type=float, default=1.0e-6, help="Crustal heat prod. prior std. dev. (W/m^3)")
    parser.add_argument("--hcrustmin", type=float, default=0, help="Crustal heat prod. prior lower cutoff (W/m^3)")
    parser.add_argument("--hcrustmax", type=float, default=10e-6, help="Crustal heat prod. prior upper cutoff (W/m^3)")

    # Crustal thickness
    parser.add_argument("--zcrustmean", type=float, default=30.0, help="Crustal thickness prior mean (km)")
    parser.add_argument("--zcruststd", type=float, default=3.0, help="Crustal thickness prior std. dev. (km)")
    parser.add_argument("--zcrusttrunc", type=float, default=2, help="Truncate crustal thickness prior at +- X*std.")

    # Mantle potential temperature
    parser.add_argument("--tpmean", type=float, default=defaults.Tp, help="Mantle potential temperature prior mean (Kelvin)")
    parser.add_argument("--tpstd", type=float, default=25.0, help="Mantle potential temperature prior std. dev. (Kelvin)")
    parser.add_argument("--tptunc", type=float, default=2, help="Truncate mantle potential temperature prior at +- X*std.")

    # Mantle radiogenic heat production
    parser.add_argument("--Hmantlemean", type=float, default=1e-8, help="Lithospheric mantle heat prod. prior mean (W/m^3)")
    parser.add_argument("--Hmantlestd", type=float, default=0.5, help="Lithospheric mantle heat prod. prior std. dev. in base-10 log")

    # Hierarchical P error on input data
    parser.add_argument("--sigma-min", type=float, default=1e7, help="Hierarchical input P error min")
    parser.add_argument("--sigma-max", type=float, default=1e9, help="Hierarchical input P error max")

    #
    # emcee MCMC parameters
    #
    parser.add_argument("--nwalkers", type=int, default=8, help="No. of final MCMC walkers")
    parser.add_argument("--oversampling", type=int, default=4, help="Oversampling walker factor for initial burn-in phase")
    parser.add_argument("--burnin", type=int, default=200, help="No. of samples during burn-in phase")
    parser.add_argument("--samples", type=int, default=1000, help="Final no. of samples")
    parser.add_argument("--priorsamples", type=int, default=1000000, help="No. of random samples of prior PDFs")

    args = parser.parse_args()



    # Load the P-T constraints
    PTdata = np.loadtxt(args.input, delimiter=",", skiprows=1, usecols=[0, 1])
    pressurePa = np.atleast_2d(PTdata)[:, 1] * 1e9
    temperatureK = np.atleast_2d(PTdata)[:, 0] + 273.15

    # Construct the thermal parameter models for crust and mantle -- N.B. Needs crustal bulk modulus
    crust_model = load_from_string(
        args.crust, args.crust_alpha, args.crust_cp, args.crust_rho, args.crust_cond
    )
    mantle_model = load_from_string(
        args.mantle,
        args.mantle_alpha,
        args.mantle_cp,
        args.mantle_rho,
        args.mantle_cond,
        grain_size=args.mantle_grain_size,
        bulk_modulus=args.mantle_K,
    )

    # Construct gravitational acceleration as a function of depth
    _g = fwd_mdl.little_g if args.approxg else fwd_mdl.little_g_const

    # Construct the adiabat
    adiabat = fwd_mdl.compute_adiabat(mantle_model, _g, args.Tp, defaults.P0, defaults.Z_MAX_ADIABAT)

    # Construct the priors    
    # prior_shf = bsn.mk_gaussian_prior(args.surfaceheatfluxmean, args.surfaceheatfluxstd)
    prior_shf = bsn.mk_uniform_prior(args.surfaceheatfluxmin, args.surfaceheatfluxmax)

    prior_zc = bsn.mk_truncated_gaussian_prior(-args.zcrusttrunc, args.zcrusttrunc, args.zcrustmean, args.zcruststd)

    prior_hc = bsn.mk_truncated_gaussian_prior((args.hcrustmin - args.hcrustmean)/args.hcruststd, (args.hcrustmax - args.hcrustmean)/args.hcruststd, args.hcrustmean, args.hcruststd)

    priors = [prior_shf, prior_zc, prior_hc]

    if args.invertHmantle:
        priors.append(bsn.mk_gaussian_prior(np.log10(args.Hmantlemean), args.Hmantlestd))

    if args.invertTp:
        priors.append(bsn.mk_truncated_gaussian_prior(-args.tptunc, args.tptunc, args.tpmean, args.tpstd))

    if args.hierarchical:
        priors.append(bsn.mk_uniform_prior(args.sigma_min, args.sigma_max))

    # Construct likelihood function
    likelihood = bsn.likelihood_base(
        crust_model,
        mantle_model,
        _g,
        defaults.Z_MAX,
        args.alphaD,
        adiabat,
        args.Hmantle,
        args.sigma,
        pressurePa,
        temperatureK,
    )
    # Append optional parameters to likelihood function (must be in same order as priors above)
    if args.invertHmantle:
        likelihood = bsn.likelihood_augment_Hmantle(likelihood)
    if args.invertTp:
        likelihood = bsn.likelihood_augment_tp(likelihood, mantle_model, _g, defaults.P0, defaults.Z_MAX_ADIABAT)
    if args.hierarchical:
        likelihood = bsn.likelihood_augment_hierarchical(likelihood)

    # Create the log posterior function
    log_posterior = bsn.make_log_posterior(priors, likelihood)

    # Generate initial burn-in oversampled models
    ndim = len(priors)
    sampled_successfully = False

    while not sampled_successfully:
        initial_state = np.empty((args.nwalkers * args.oversampling, ndim))
        for i, prior in enumerate(priors):
            initial_state[:, i] = prior.rvs(size=args.nwalkers * args.oversampling)

        # Resample to obtain valid initial states
        for i in range(args.nwalkers * args.oversampling):
            ll = likelihood(initial_state[i])
            attempts = 0
            while not np.isfinite(ll):
                if attempts > 100:
                    raise RuntimeError("Failed to generate a valid initial model.")

                for j, prior in enumerate(priors):
                    initial_state[i, j] = prior.rvs()
                ll = likelihood(initial_state[i])

                attempts += 1

        # Run initial samplers
        print("Starting initial burn-in sampling...")
        initial_sampler = EnsembleSampler(args.nwalkers * args.oversampling, ndim, log_posterior)
        state = initial_sampler.run_mcmc(initial_state, args.burnin)

        # Resample down to the final number of desired walkers by randomly selecting the best performing models
        print(f"Select {args.nwalkers} of the original {args.nwalkers * args.oversampling} walkers:")
        initial_state = bsn.resample(state.coords, state.log_prob, args.nwalkers)

        print("Initiating main statistical sampling...")
        try:
            sampler = EnsembleSampler(args.nwalkers, ndim, log_posterior)
            sampler.run_mcmc(initial_state, args.samples)
            sampled_successfully = True
        except ValueError:
            # Sampler can error if starting models are too close to oneanother (condition number error)
            # If this happens, just try again
            print("Initial samples are too close. Retrying...")
            pass

    print("Individual walker model acceptance fractions = ",sampler.acceptance_fraction)


    # Write out resuts and plot figures

    #
    # Write out the prior parameter choices -- N.B. Could add truncation info later
    #
    flat_samples = sampler.get_chain(discard=0, thin=5, flat=True)
    log_prob, _ = sampler.compute_log_prob(flat_samples)

    priors = [
        [args.surfaceheatfluxmin * 1e3, args.surfaceheatfluxmax * 1e3],
        [args.zcrustmean, args.zcruststd],
        [1e6 * args.hcrustmean, 1e6* (args.hcrustmean - 10 ** (np.log10(args.hcrustmean) - args.hcruststd)),],
    ]
    optional_priors = []

    columns = [
        "Log probability",
        "Surface heat flux (mW per squared metre)",
        "Crustal thickness (km)",
        "Crustal radiogenic heating (microW per cubic metre)",
    ]
    optional_columns = []

    if args.invertHmantle:
        optional_priors.append([args.Hmantlemean, 10**args.Hmantlestd])
        optional_columns.append("Mantle radiogenic heating (microW per cubic metre)")
    if args.invertTp:
        optional_priors.append([args.tpmean - 273.15, args.tpstd])
        optional_columns.append("Mantle potential temperature (Celsius)")
    if args.hierarchical:
        optional_priors.append([args.sigma_min, args.sigma_max])
        optional_columns.append("P error on datapoints (Pa)")

    prior_output = pd.DataFrame(
        priors + optional_priors,
        index=columns[1:] + optional_columns,
        columns=["Mean or Min", "Standard deviation or Max"],
    )
    Path("output").mkdir(parents=True, exist_ok=True)
    prior_output.to_csv(Path("output") / f"{Path(args.input).stem}_prior.csv")


    # Generate and write out random samples of the prior probabilities -- N.B. Could somehow edit to force use of same distributions set initially
    if args.samplepriors:
        print(f"Generating random draw of {args.priorsamples} samples from prior PDFs...")

        shf_prior_samples = bsn.sample_uniform_prior(args.surfaceheatfluxmin, args.surfaceheatfluxmax, args.priorsamples)
        zc_prior_samples = bsn.sample_truncated_gaussian_prior(-args.zcrusttrunc, args.zcrusttrunc, args.zcrustmean, args.zcruststd, args.priorsamples)
        Hc_prior_samples = bsn.sample_truncated_gaussian_prior((args.hcrustmin - args.hcrustmean)/args.hcruststd, (args.hcrustmax - args.hcrustmean)/args.hcruststd, args.hcrustmean, args.hcruststd, args.priorsamples)

        prior_samples = np.column_stack((1e3*shf_prior_samples, zc_prior_samples, 1e6*Hc_prior_samples))

        columns1 = [
            "Surface heat flux (mW per squared metre)",
            "Crustal thickness (km)",
            "Crustal radiogenic heating (microW per cubic metre)"
        ]
        optional_columns1 = []

        if args.invertHmantle:
            Hm_prior_samples = bsn.sample_gaussian_prior(np.log10(args.Hmantlemean), args.Hmantlestd, args.priorsamples)
            prior_samples = np.column_stack((prior_samples, 1e6*Hm_prior_samples))
            optional_columns1.append("Mantle radiogenic heating (microW per cubic metre)")
        if args.invertTp:
            Tp_prior_samples = bsn.sample_truncated_gaussian_prior(-args.tptunc, args.tptunc, args.tpmean, args.tpstd, args.priorsamples)
            prior_samples = np.column_stack((prior_samples, Tp_prior_samples - 273.15))
            optional_columns1.append("Mantle potential temperature (Celsius)")
        if args.hierarchical:
            sigma_prior_samples = bsn.sample_uniform_prior(args.sigma_min, args.sigma_max, args.priorsamples)
            prior_samples = np.column_stack((prior_samples, sigma_prior_samples))
            optional_columns1.append("P error on datapoints (Pa)")

        sampled_priors = pd.DataFrame(prior_samples, columns=columns1 + optional_columns1)
        sampled_priors.to_csv(Path("output") / f"{Path(args.input).stem}_prior_samples.csv")



    z_mbl_samples = np.empty(flat_samples.shape[0])
    geotherm_output = np.empty_like(z_mbl_samples, dtype=object)

    fig, ax = plt.subplots()

    Pmax = 0.0
    for i, sample in enumerate(flat_samples):
        shf, zc, hc, *optional = sample

        h_mantle = 0.0
        if args.invertHmantle:
            h_mantle = optional[0]
            optional = optional[1:]

        # Default adiabat computed above
        if args.invertTp:
            tp = optional[0]
            optional = optional[1:]
            adiabat = fwd_mdl.compute_adiabat(mantle_model, _g, tp, defaults.P0, defaults.Z_MAX_ADIABAT)

        sigma = args.sigma
        if args.hierarchical:
            sigma = optional[0]

        zc *= 1e3
#        hc = 10**hc
        hc = hc

        sZ, sT, sP, z_mbl = fwd_mdl.compute_geotherm(
            crust_model,
            mantle_model,
            _g,
            zc,
            defaults.Z_MAX,
            hc,
            h_mantle,
            shf,
            adiabat,
            args.alphaD,
        )

        # Prepare the geotherm arrays for output
        geotherm_arrays = [sZ / 1e3, sP / 1e9, sT - 273.15, adiabat.T(sP) - 273.15]
        therm_exp = np.empty_like(sZ)
        heat_cap = np.empty_like(sZ)
        density = np.empty_like(sZ)
        therm_cond = np.empty_like(sZ)
        for arr, crust_val in zip(
            [therm_exp, heat_cap, density, therm_cond], crust_model.compute(sZ, sP, sT)
        ):
            arr[sZ < zc] = crust_val[sZ < zc]
        for arr, mantle_val in zip(
            [therm_exp, heat_cap, density, therm_cond], mantle_model.compute(sZ, sP, sT)
        ):
            arr[sZ >= zc] = mantle_val[sZ >= zc]
        geotherm_arrays.extend([therm_exp, heat_cap, density, therm_cond])

        geotherm_output[i] = pd.DataFrame(
            np.column_stack(geotherm_arrays),
            columns=[
                "Depth (km)",
                "Pressure (GPa)",
                "Temperature (Celsius)",
                "Adiabatic temperature (Celsius)",
                "Thermal expansion (per K)",
                "Specific heat capacity (J per kg per K)",
                "Density (kg per cubic metre)",
                "Thermal conductivity (W per m per K)",
            ],
        )

        # Plot of all the geotherms
        ax.plot(sT - 273.15, sZ / 1e3, linestyle="solid", color="grey", linewidth=0.25, alpha=0.1)

        z_mbl_samples[i] = z_mbl / 1e3

        Pmax = max([Pmax, sP[-1]])

    # Write out the posterior parameter values
    flat_samples[:, 0] *= 1e3
#    flat_samples[:, 2] = 1e6 * 10 ** flat_samples[:, 2]
    flat_samples[:, 2] = 1e6 * flat_samples[:, 2]
    if "Mantle potential temperature (Celsius)" in optional_columns:
        optional_index = optional_columns.index(
            "Mantle potential temperature (Celsius)"
        )
        flat_samples[:, len(columns) - 1 + optional_index] -= 273.15

    posterior_output = pd.DataFrame(
        np.column_stack((log_prob, flat_samples, z_mbl_samples)),
        columns=columns + optional_columns + ["LAB depth (km)"],
    )
    posterior_output.to_csv(Path("output") / f"{Path(args.input).stem}_posterior.csv")

    # Write out the associated geotherms
    np.savez_compressed(Path("output") / f"{Path(args.input).stem}_geotherm", geotherm=geotherm_output)



    flat_samples[:, 0] /= 1e3
#    flat_samples[:, 2] = np.log10(flat_samples[:, 2] / 1e6)
    flat_samples[:, 2] = flat_samples[:, 2] / 1e6

    most_probable_model = np.argmax(log_prob)
    shf, zc, hc = flat_samples[most_probable_model, :][:3]
    optional = flat_samples[most_probable_model, :][3:]

    h_mantle = 0.0
    if args.invertHmantle:
        h_mantle = optional[0]
        optional = optional[1:]

    # Default adiabat computed above
    if args.invertTp:
        tp = optional[0]
        optional = optional[1:]
        adiabat = fwd_mdl.compute_adiabat(
            mantle_model, _g, tp + 273.15, defaults.P0, defaults.Z_MAX_ADIABAT
        )

    sigma = args.sigma
    if args.hierarchical:
        sigma = optional[0]

    zc *= 1e3
#    hc = 10**hc
    hc = hc

    sZ, sT, sP, z_mbl = fwd_mdl.compute_geotherm(
        crust_model,
        mantle_model,
        _g,
        zc,
        defaults.Z_MAX,
        hc,
        h_mantle,
        shf,
        adiabat,
        args.alphaD,
    )


    # write out all geotherms into individual CSV files
    if args.iwrite:
        print("Writing out individual geotherms...")
        Path(f"output/{Path(args.input).stem}_geotherms").mkdir(parents=True, exist_ok=True)
        data = np.load(f"output/{Path(args.input).stem}_geotherm.npz", allow_pickle=True)
        num_entries = len(data['geotherm'])
        for i in range(num_entries):
            data["geotherm"][i].to_csv(f"output/{Path(args.input).stem}_geotherms/geotherm_{i}.csv")


    # Add best fitting geotherm to plot
    ax.plot(sT - 273.15, sZ / 1e3, linestyle="solid", color="red", linewidth=1.0, alpha=1.0)

    ax.set_xlim(left=0.0)
    ax.set_ylim(300, 0.0)
    ax.scatter(temperatureK - 273.15, adiabat.Pz(pressurePa) / 1e3)

    fig, ax = plt.subplots(2, 2)
    fig.set_tight_layout(True)

    ax[0][0].set_title("Surface Heat Flux")
    ax[0][0].hist(flat_samples[:, 0] * 1e3, bins=20)

    ax[0][1].set_title("z crust")
    ax[0][1].hist(flat_samples[:, 1], bins=20)

    ax[1][0].set_title("H crust")
    ax[1][0].hist(10 ** flat_samples[:, 2], bins=20)

    ax[1][1].set_title("z MBL")
    ax[1][1].hist(z_mbl_samples, bins=20)

    optindex = 3
    if args.invertHmantle:
        fig, ax = plt.subplots()

        ax.set_title("H Mantle")
        ax.hist(flat_samples[:, optindex], bins=20)
        optindex += 1

    if args.invertTp:
        fig, ax = plt.subplots()

        ax.set_title("Tp")
        ax.hist(flat_samples[:, optindex], bins=20)
        optindex += 1

    if args.hierarchical:
        fig, ax = plt.subplots()

        ax.set_title("Hierarchical Sigma")
        ax.hist(flat_samples[:, optindex], bins=20)

    fig, ax = plt.subplots()

    ax.plot(sampler.lnprobability.T)

#    plt.show()
