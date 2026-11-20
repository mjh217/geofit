"""Command line: invert P-T data for a family of geotherms, or compute a single forward geotherm.

    geofit DATA.csv [options]      Bayesian inversion of P-T constraints
    geofit forward [options]       one geotherm for given parameters
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bayesian, forward, output
from . import defaults as d
from .inversion import sample_posterior
from .materials import MODELS, make_model


def add_model_options(parser):
    """Options shared by the inversion and the forward model."""
    models = parser.add_argument_group("thermal models")
    models.add_argument("--crust", default="Constant", help=f"crustal model: {', '.join(MODELS)} (default: %(default)s)")
    models.add_argument("--mantle", default="Constant", help=f"mantle model: {', '.join(MODELS)} (default: %(default)s)")
    models.add_argument("--approxg", action="store_true", help="let gravity vary with depth (default: constant 9.81 m/s^2)")
    models.add_argument(
        "--alphaD",
        type=float,
        default=0.2,
        help="width of the smoothed corner where the geotherm meets the adiabat, as a fraction of its depth "
        "(0 = no smoothing; default: %(default)s)",
    )

    constants = parser.add_argument_group("constant values (used where a model does not compute them)")
    constants.add_argument("--crust-rho", type=float, default=d.RHO_CRUST, help="crustal density [kg/m^3]")
    constants.add_argument("--crust-alpha", type=float, default=d.ALPHA_CRUST, help="crustal thermal expansivity [1/K]")
    constants.add_argument("--crust-cp", type=float, default=d.CP_CRUST, help="crustal heat capacity [J/kg/K]")
    constants.add_argument("--crust-cond", type=float, default=d.K_CRUST, help="crustal conductivity [W/m/K]")
    constants.add_argument("--mantle-rho", type=float, default=d.RHO_MANTLE, help="mantle density [kg/m^3]")
    constants.add_argument("--mantle-alpha", type=float, default=d.ALPHA_MANTLE, help="mantle thermal expansivity [1/K]")
    constants.add_argument("--mantle-cp", type=float, default=d.CP_MANTLE, help="mantle heat capacity [J/kg/K]")
    constants.add_argument("--mantle-cond", type=float, default=d.K_MANTLE, help="mantle conductivity [W/m/K]")

    hoggard = parser.add_argument_group("Hoggard mantle model")
    hoggard.add_argument(
        "--mantle-grain-size", type=float, default=d.GRAIN_SIZE_MANTLE, help="grain size [cm] (default: %(default)s)"
    )
    hoggard.add_argument(
        "--mantle-K",
        type=float,
        default=d.BULK_MODULUS_MANTLE,
        help="bulk modulus [Pa] for the pressure dependence of conductivity; the density equations of state "
        "use fixed K0 = 90 GPa (crust) and 130 GPa (mantle) (default: %(default)s)",
    )


def add_mantle_options(parser, note=""):
    """--Tp and --Hmantle (in the inversion, the fixed values unless inverted for)."""
    parser.add_argument(
        "--Tp", type=float, default=d.TP, help=f"mantle potential temperature [K]{note} (default: %(default)s)"
    )
    parser.add_argument(
        "--Hmantle", type=float, default=0.0, help=f"lithospheric mantle heat production [W/m^3]{note} (default: %(default)s)"
    )


def build_models(args):
    crust = make_model(args.crust, args.crust_alpha, args.crust_cp, args.crust_rho, args.crust_cond)
    mantle = make_model(
        args.mantle,
        args.mantle_alpha,
        args.mantle_cp,
        args.mantle_rho,
        args.mantle_cond,
        grain_size=args.mantle_grain_size,
        bulk_modulus=args.mantle_K,
    )
    g = forward.gravity if args.approxg else forward.gravity_constant
    return crust, mantle, g


def inversion_parser():
    parser = argparse.ArgumentParser(
        prog="geofit",
        description="Bayesian inversion of P-T constraints for steady-state geotherms. "
        "Use 'geofit forward -h' for a single forward geotherm.",
    )
    parser.add_argument("input", help="P-T constraints CSV: temperature [C] and pressure [GPa] in the first two columns")
    parser.add_argument("--output-dir", default="output", help="directory for the results (default: %(default)s)")
    parser.add_argument("--seed", type=int, help="random seed, for reproducible runs")
    parser.add_argument("--samplepriors", action="store_true", help="also write random samples of the priors")
    parser.add_argument("--iwrite", action="store_true", help="also write every sampled geotherm to its own CSV file")
    add_model_options(parser)
    fixed = parser.add_argument_group("fixed values of parameters that can be inverted for")
    add_mantle_options(fixed, note=", unless --invertTp/--invertHmantle")
    fixed.add_argument(
        "--sigma", type=float, default=3e8, help="uncertainty of the data pressures [Pa], unless --hierarchical (default: %(default)s)"
    )

    optional = parser.add_argument_group("optional parameters to invert for")
    optional.add_argument("--invertHmantle", action="store_true", help="invert for mantle heat production")
    optional.add_argument("--invertTp", action="store_true", help="invert for mantle potential temperature")
    optional.add_argument("--hierarchical", action="store_true", help="invert for the data uncertainty sigma")

    priors = parser.add_argument_group("priors")
    priors.add_argument("--surfaceheatfluxmin", type=float, default=0.01, help="surface heat flux minimum [W/m^2] (uniform prior)")
    priors.add_argument("--surfaceheatfluxmax", type=float, default=0.25, help="surface heat flux maximum [W/m^2]")
    priors.add_argument("--zcrustmean", type=float, default=30.0, help="crustal thickness mean [km] (truncated Gaussian prior)")
    priors.add_argument("--zcruststd", type=float, default=3.0, help="crustal thickness standard deviation [km]")
    priors.add_argument("--zcrusttrunc", type=float, default=2, help="crustal thickness cut-off, in standard deviations from the mean")
    priors.add_argument("--hcrustmean", type=float, default=1e-6, help="crustal heat production mean [W/m^3] (truncated Gaussian prior)")
    priors.add_argument("--hcruststd", type=float, default=1.0e-6, help="crustal heat production standard deviation [W/m^3]")
    priors.add_argument("--hcrustmin", type=float, default=0, help="crustal heat production minimum [W/m^3]")
    priors.add_argument("--hcrustmax", type=float, default=10e-6, help="crustal heat production maximum [W/m^3]")
    priors.add_argument("--Hmantlemean", type=float, default=1e-8, help="mantle heat production mean [W/m^3] (truncated Gaussian prior)")
    priors.add_argument("--Hmantlestd", type=float, default=1e-8, help="mantle heat production standard deviation [W/m^3]")
    priors.add_argument("--Hmantlemin", type=float, default=0, help="mantle heat production minimum [W/m^3]")
    priors.add_argument("--Hmantlemax", type=float, default=1e-6, help="mantle heat production maximum [W/m^3]")
    priors.add_argument("--tpmean", type=float, default=d.TP, help="potential temperature mean [K] (truncated Gaussian prior)")
    priors.add_argument("--tpstd", type=float, default=25.0, help="potential temperature standard deviation [K]")
    priors.add_argument(
        "--tptrunc", "--tptunc", dest="tptrunc", type=float, default=2,
        help="potential temperature cut-off, in standard deviations from the mean",
    )
    priors.add_argument("--sigma-min", type=float, default=1e7, help="data uncertainty minimum [Pa] (uniform prior)")
    priors.add_argument("--sigma-max", type=float, default=1e9, help="data uncertainty maximum [Pa]")

    mcmc = parser.add_argument_group("MCMC sampling (emcee)")
    mcmc.add_argument("--nwalkers", type=int, default=8, help="number of walkers in the main run (default: %(default)s)")
    mcmc.add_argument("--oversampling", type=int, default=4, help="factor more walkers for the burn-in (default: %(default)s)")
    mcmc.add_argument("--burnin", type=int, default=200, help="burn-in steps (default: %(default)s)")
    mcmc.add_argument("--samples", type=int, default=1000, help="steps in the main run (default: %(default)s)")
    mcmc.add_argument("--priorsamples", type=int, default=1000000, help="random samples for --samplepriors (default: %(default)s)")
    return parser


def build_priors(args):
    """Priors in model-vector order, with their output column names and units."""

    def truncated(mean, std, vmin, vmax):
        return bayesian.truncated_gaussian_prior((vmin - mean) / std, (vmax - mean) / std, mean, std)

    priors = [
        bayesian.uniform_prior(args.surfaceheatfluxmin, args.surfaceheatfluxmax),
        bayesian.truncated_gaussian_prior(-args.zcrusttrunc, args.zcrusttrunc, args.zcrustmean, args.zcruststd),
        truncated(args.hcrustmean, args.hcruststd, args.hcrustmin, args.hcrustmax),
    ]
    if args.invertHmantle:
        priors.append(truncated(args.Hmantlemean, args.Hmantlestd, args.Hmantlemin, args.Hmantlemax))
    if args.invertTp:
        priors.append(bayesian.truncated_gaussian_prior(-args.tptrunc, args.tptrunc, args.tpmean, args.tpstd))
    if args.hierarchical:
        priors.append(bayesian.uniform_prior(args.sigma_min, args.sigma_max))
    return priors


def parameter_columns(args):
    """Output column name, and the conversion from SI, of each parameter in model-vector order."""
    columns = [
        (output.SURFACE_HEAT_FLUX, lambda x: x * 1e3),
        (output.CRUSTAL_THICKNESS, lambda x: x),
        (output.CRUSTAL_HEATING, lambda x: 1e6 * x),
    ]
    if args.invertHmantle:
        columns.append((output.MANTLE_HEATING, lambda x: 1e6 * x))
    if args.invertTp:
        columns.append((output.POTENTIAL_TEMPERATURE, lambda x: x - d.C2K))
    if args.hierarchical:
        columns.append((output.DATA_UNCERTAINTY, lambda x: x))
    return columns


def prior_summary(args):
    """The prior parameters, in output units."""
    rows = {
        output.SURFACE_HEAT_FLUX: [args.surfaceheatfluxmin * 1e3, args.surfaceheatfluxmax * 1e3],
        output.CRUSTAL_THICKNESS: [args.zcrustmean, args.zcruststd],
        output.CRUSTAL_HEATING: [1e6 * args.hcrustmean, 1e6 * args.hcruststd],
    }
    if args.invertHmantle:
        rows[output.MANTLE_HEATING] = [1e6 * args.Hmantlemean, 1e6 * args.Hmantlestd]
    if args.invertTp:
        rows[output.POTENTIAL_TEMPERATURE] = [args.tpmean - d.C2K, args.tpstd]
    if args.hierarchical:
        rows[output.DATA_UNCERTAINTY] = [args.sigma_min, args.sigma_max]
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Mean or Min", "Standard deviation or Max"])


def unpack(sample, args):
    """Surface heat flux, crustal thickness [m], crustal and mantle heat production and Tp of a model vector."""
    shf, z_crust, h_crust, *optional = sample
    h_mantle = optional.pop(0) if args.invertHmantle else args.Hmantle
    tp = optional.pop(0) if args.invertTp else None
    return shf, z_crust * 1e3, h_crust, h_mantle, tp


def invert(argv):
    args = inversion_parser().parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Data: temperature [C] and pressure [GPa]
    data = np.atleast_2d(np.loadtxt(args.input, delimiter=",", skiprows=1, usecols=[0, 1]))
    P_data = data[:, 1] * 1e9
    T_data = data[:, 0] + d.C2K

    crust_model, mantle_model, g = build_models(args)
    adiabat = forward.compute_adiabat(mantle_model, g, args.Tp, d.P_SURFACE, d.Z_MAX_ADIABAT)

    priors = build_priors(args)
    log_likelihood = bayesian.make_likelihood(
        crust_model, mantle_model, g, d.Z_MAX, args.alphaD, adiabat, args.Hmantle, args.sigma, P_data, T_data
    )
    # Same order as the priors
    if args.invertHmantle:
        log_likelihood = bayesian.add_mantle_heat_production(log_likelihood)
    if args.invertTp:
        log_likelihood = bayesian.add_potential_temperature(log_likelihood, mantle_model, g, d.P_SURFACE, d.Z_MAX_ADIABAT)
    if args.hierarchical:
        log_likelihood = bayesian.add_data_uncertainty(log_likelihood)
    log_posterior = bayesian.make_log_posterior(priors, log_likelihood)

    sampler = sample_posterior(
        log_posterior, log_likelihood, priors, args.nwalkers, args.oversampling, args.burnin, args.samples
    )
    samples = sampler.get_chain(discard=0, thin=5, flat=True)
    log_prob, _ = sampler.compute_log_prob(samples)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    columns = parameter_columns(args)
    names = [name for name, _ in columns]

    prior_summary(args).to_csv(out / f"{stem}_prior.csv")

    if args.samplepriors:
        print(f"Generating random draw of {args.priorsamples} samples from prior PDFs...")
        draws = [convert(prior.rvs(size=args.priorsamples)) for prior, (_, convert) in zip(priors, columns)]
        pd.DataFrame(np.column_stack(draws), columns=names).to_csv(out / f"{stem}_prior_samples.csv")

    # Geotherm of every posterior sample
    geotherms = np.empty(samples.shape[0], dtype=object)
    z_mbl = np.empty(samples.shape[0])
    for i, sample in enumerate(samples):
        shf, z_crust, h_crust, h_mantle, tp = unpack(sample, args)
        if tp is not None:
            adiabat = forward.compute_adiabat(mantle_model, g, tp, d.P_SURFACE, d.Z_MAX_ADIABAT)
        geotherms[i], z = output.geotherm_table(
            crust_model, mantle_model, g, z_crust, h_crust, h_mantle, shf, adiabat, args.alphaD
        )
        z_mbl[i] = z / 1e3

    converted = np.column_stack([convert(samples[:, j]) for j, (_, convert) in enumerate(columns)])
    posterior = pd.DataFrame(
        np.column_stack((log_prob, converted, z_mbl)), columns=["Log probability", *names, "LAB depth (km)"]
    )
    posterior.to_csv(out / f"{stem}_posterior.csv")
    np.savez_compressed(out / f"{stem}_geotherm", geotherm=geotherms)

    if args.iwrite:
        print("Writing out individual geotherms...")
        folder = out / f"{stem}_geotherms"
        folder.mkdir(parents=True, exist_ok=True)
        for i, table in enumerate(geotherms):
            table.to_csv(folder / f"geotherm_{i}.csv")
    return 0


def forward_parser():
    parser = argparse.ArgumentParser(
        prog="geofit forward", description="Compute one steady-state geotherm for given parameters."
    )
    parser.add_argument("--surfaceheatflux", type=float, required=True, help="surface heat flux [W/m^2]")
    parser.add_argument("--zcrust", type=float, required=True, help="crustal thickness [km]")
    parser.add_argument("--hcrust", type=float, required=True, help="crustal heat production [W/m^3]")
    add_mantle_options(parser)
    parser.add_argument("-o", "--output", default="geotherm.csv", help="geotherm CSV (default: %(default)s)")
    add_model_options(parser)
    return parser


def forward_geotherm(argv):
    args = forward_parser().parse_args(argv)
    crust_model, mantle_model, g = build_models(args)
    adiabat = forward.compute_adiabat(mantle_model, g, args.Tp, d.P_SURFACE, d.Z_MAX_ADIABAT)
    try:
        table, z_mbl = output.geotherm_table(
            crust_model, mantle_model, g, args.zcrust * 1e3, args.hcrust, args.Hmantle, args.surfaceheatflux,
            adiabat, args.alphaD,
        )
    except forward.AdiabatInCrustError as err:
        sys.exit(f"No geotherm: {err} (surface heat flux too high)")
    except forward.NegativeFluxError as err:
        sys.exit(f"No geotherm: {err} (surface heat flux too low for this heat production)")
    except forward.NoAdiabatError:
        sys.exit(f"No geotherm: it does not reach the adiabat above {d.Z_MAX / 1e3:.0f} km")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(f"LAB depth {z_mbl / 1e3:.1f} km; wrote {args.output}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "forward":
        return forward_geotherm(argv[1:])
    return invert(argv)


if __name__ == "__main__":
    sys.exit(main())
