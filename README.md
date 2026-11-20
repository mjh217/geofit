# GeoFit

GeoFit estimates steady-state continental geotherms from pressure-temperature (P-T) constraints, such as thermobarometry of mantle xenoliths and xenocrysts. Each geotherm is found by integrating the equation for 1-D vertical heat conduction downwards from the surface until it meets the mantle adiabat. It is defined by:

1. Surface heat flux
2. Crustal thickness
3. Bulk radiogenic heat production of the crust
4. Mantle potential temperature (default 1330 °C)
5. Bulk radiogenic heat production of the lithospheric mantle (default 0)

The density and thermal properties (α, C<sub>P</sub>, k, ρ) of the crust and mantle can be chosen from several models, some of which depend on temperature and pressure. GeoFit then uses Bayesian inference (MCMC sampling with [emcee](https://emcee.readthedocs.io)) to find the posterior distribution of these parameters, given their prior distributions and the P-T data. It also gives the depth of the lithosphere-asthenosphere boundary (LAB) and the range of temperatures at each depth.

This code was developed at the Australian National University as part of Geoscience Australia's [*Resourcing Australia's Prosperity*](https://www.ga.gov.au/scientific-topics/resourcing-australias-prosperity) initiative.

## Installation

```bash
pip install -e .
```

This needs numpy, scipy, pandas and emcee, and gives the `geofit` command (or run `python -m geofit`).

## Inversion

```bash
geofit example_data/brigooda_TP.csv --crust Whittington --mantle Hoggard --invertTp
```

The input is a CSV with a header row, temperature (°C) in the first column and pressure (GPa) in the second; other columns are ignored. Results are written to `output/` (or `--output-dir`), named after the input file:

- `<name>_prior.csv`: the prior distributions used.
- `<name>_posterior.csv`: posterior samples (every 5th step of each walker), with their log-probability, parameters and LAB depth (km).
- `<name>_geotherm.npz`: the geotherm of each posterior sample, as a table of depth, pressure, temperature, adiabatic temperature and thermal properties. Read with `np.load(path, allow_pickle=True)["geotherm"]`.
- `<name>_prior_samples.csv` (with `--samplepriors`): random samples of the priors, for comparison with the posterior.
- `<name>_geotherms/geotherm_<i>.csv` (with `--iwrite`): each geotherm as its own CSV.

Surface heat flux, crustal thickness and crustal heat production are always inverted for. `--invertTp`, `--invertHmantle` and `--hierarchical` add the potential temperature, mantle heat production and the uncertainty of the data pressures (otherwise fixed by `--Tp`, `--Hmantle` and `--sigma`). `geofit -h` lists the options, including the priors, the thermal models and the MCMC settings. `--seed` makes a run reproducible.

## Forward model

To compute a single geotherm without an inversion:

```bash
geofit forward --surfaceheatflux 0.06 --zcrust 35 --hcrust 1e-6 --Tp 1603.15 --Hmantle 0 \
    --crust Whittington --mantle Hoggard -o geotherm.csv
```

This writes the same table as the inversion's geotherms and prints the LAB depth. Units are W/m<sup>2</sup>, km, W/m<sup>3</sup> and K.

## Thermal models

Choose with `--crust` and `--mantle`. Where a model does not compute a property, it uses the constant value set by `--crust-rho`, `--crust-alpha`, `--crust-cp`, `--crust-cond` (or the `--mantle-` equivalents).

| Model | Intended for | Temperature and pressure dependence |
|---|---|---|
| `Constant` | crust or mantle | none |
| `Hasterok` | crust | k (Goes et al., 2020), with `--crust-cond` as the room-temperature value |
| `Whittington` | crust | α (Stephenson et al., 2024); k and C<sub>P</sub> (Whittington et al., 2009); ρ (Grose & Afonso, 2013) |
| `Osako` | mantle | k (Osako et al., 2004) |
| `Hoggard` | mantle | α (Bouhifd et al., 1996); C<sub>P</sub> and lattice k (Korenaga & Korenaga, 2016); radiative k for grain size `--mantle-grain-size` and ρ (Grose & Afonso, 2013); pressure dependence of k (Hofmeister, 2007) |

`geofit/materials.py` gives the equations used and their sources.

## Reproducing Hoggard et al. (2026)

For each site, Hoggard et al. (2026) ran

```bash
geofit SITE_TP.csv --crust Whittington --mantle Hoggard --mantle-grain-size 0.3 --invertTp --samplepriors \
    --zcrustmean 30
```

with `--zcrustmean 30` in Phanerozoic regions and `--zcrustmean 40` in cratonic and craton-adjacent regions. The other priors are the defaults.

## Tests

```bash
pip install -e ".[test]" && pytest
```

The tests check that seeded inversions reproduce the outputs of the version used by Hoggard et al. (2026), that the forward model matches exact solutions, and that the thermal models match values from their papers.

## Citation

If you use GeoFit, please cite:

> Hoggard, M. J., Sudholz, Z. J., Hawkins, R., Duvernay, T., Bagchi, S., Richards, F. D., Jaques, A. L., Yaxley, G. M., Czarnota, K., & Haynes, M. (2026). Paleothermal structure of the Australian continental lithosphere. *Geochemistry, Geophysics, Geosystems* (accepted).

## Licence

GeoFit is released under the MIT licence. See `LICENSE`.
