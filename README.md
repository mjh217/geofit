# GeoFit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24389533.svg)](https://doi.org/10.5281/zenodo.24389533)

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

This install requires numpy, scipy, pandas and emcee, and gives the `geofit` command (or run `python -m geofit`).

## Forward model

To compute a single geotherm for specific values of the five governing parameters above, run as:

```bash
geofit forward --surfaceheatflux 0.06 --zcrust 35 --hcrust 1e-6 --Tp 1603.15 --Hmantle 0 \
    --crust Hoggard --mantle Hoggard -o geotherm.csv
```

The options `--crust` and `--mantle` select the parameterisations for [thermal properties](#thermal-property-parameterisations). Output prints the depth of the thermal lithosphere-asthenosphere boundary (LAB), where the conductive geotherm meets the adiabat, and writes a CSV with one row per depth giving depth (km), pressure (GPa), temperature (°C), the adiabatic temperature at that pressure (°C), and the thermal expansivity (K<sup>-1</sup>), specific heat capacity (J kg<sup>-1</sup> K<sup>-1</sup>), density (kg m<sup>-3</sup>), and thermal conductivity (W m<sup>-1</sup> K<sup>-1</sup>).

Rows are spaced every km from the surface to 400 km. Across the thermal boundary layer, of thickness `--alphaD` (default 0.2) times the thermal LAB depth and centred on it, the corner between the geotherm and the adiabat is rounded with a spline, sampled at 16 points. Below it, the temperature follows the adiabat. If no steady-state geotherm is possible for your chosen parameters (e.g., surface heat flux is too high), an explanation will be given and nothing is written to output.

## Inversion

To find the optimal geotherm to fit a suite of P-T datapoints, run:

```bash
geofit example_data/brigooda_TP.csv --crust Hoggard --mantle Hoggard --invertTp
```

The input is a CSV with a header row, temperature (°C) in the first column and pressure (GPa) in the second, with other columns ignored. Outputs are written to `output/` (or `--output-dir`) and consist of:

- `<name>_prior.csv`: the prior distributions used.
- `<name>_prior_samples.csv` (optional, select with `--samplepriors`): random samples of the priors, for comparison with the posterior.
- `<name>_posterior.csv`: posterior samples (every 5th step of each walker), with their log-probability, parameters and LAB depth (km).
- `<name>_geotherm.npz`: the geotherm of each posterior sample, as a table of depth, pressure, temperature, adiabatic temperature and thermal properties. Read with `np.load(path, allow_pickle=True)["geotherm"]`.
- `<name>_geotherms/geotherm_<i>.csv` (optional, select with `--iwrite`): each geotherm is output in its own CSV (N.B., produces many output files).

Surface heat flux, crustal thickness and crustal heat production are always inverted for. `--invertTp`, `--invertHmantle` and `--hierarchical` add the mantle potential temperature, lithospheric mantle heat production, and uncertainty on the data pressures (otherwise fixed by `--Tp`, `--Hmantle` and `--sigma`). `geofit -h` lists the options, including the priors, the thermal property parameterisations available, and the MCMC settings. `--seed` allows for making a run reproducible.

## Thermal property parameterisations

Choose with `--crust` (`Constant`, `Hasterok` or `Hoggard`) and `--mantle` (`Constant`, `Osako` or `Hoggard`). Where a model does not compute a property, it uses the constant value set by `--crust-rho`, `--crust-alpha`, `--crust-cp`, `--crust-cond` (or the `--mantle-` equivalents).

| Model | Layer | Temperature and pressure dependence |
|---|---|---|
| `Constant` | crust or mantle | none |
| `Hasterok` | crust | k (Goes et al., 2020), with `--crust-cond` as the room-temperature value |
| `Osako` | mantle | k (Osako et al., 2004) |
| `Hoggard` (crust) | crust | α (Stephenson et al., 2024); k and C<sub>P</sub> (Whittington et al., 2009); ρ (Grose & Afonso, 2013) |
| `Hoggard` (mantle) | mantle | α (Bouhifd et al., 1996); C<sub>P</sub> and lattice k (Korenaga & Korenaga, 2016); radiative k for grain size `--mantle-grain-size` (default 0.3 cm) and ρ (Grose & Afonso, 2013); pressure dependence of k (Hofmeister, 2007) |

The `Hoggard` crust and mantle models are the combinations used by Hoggard et al. (2026). [`geofit/materials.py`](geofit/materials.py) gives the underlying equations and their sources.

## Reproducing Hoggard et al. (2026)

For each site, Hoggard et al. (2026) ran

```bash
geofit SITE_TP.csv --crust Hoggard --mantle Hoggard --invertTp --samplepriors --zcrustmean 30
```

with `--zcrustmean 30` in Phanerozoic regions and `--zcrustmean 40` in cratonic and craton-adjacent regions. The other priors are the default values.

## Tests

```bash
pip install -e ".[test]" && pytest
```

The tests check that seeded inversions reproduce the outputs of the version used by Hoggard et al. (2026), that the forward model matches exact solutions, and that the thermal property parameterisations match values from their papers.

## Citation

If you use GeoFit, please cite:

> Hoggard, M. J., Sudholz, Z. J., Hawkins, R., Duvernay, T., Bagchi, S., Richards, F. D., Jaques, A. L., Yaxley, G. M., Czarnota, K., & Haynes, M. (2026). Paleothermal structure of the Australian continental lithosphere. *Geochemistry, Geophysics, Geosystems* (accepted).

Each release of GeoFit is archived on Zenodo at [10.5281/zenodo.24389533](https://doi.org/10.5281/zenodo.24389533), which always resolves to the latest version; each release also has its own DOI (v2026.11: [10.5281/zenodo.24389534](https://doi.org/10.5281/zenodo.24389534)).

## Licence

GeoFit is released under the MIT licence. See [`LICENSE`](LICENSE).
