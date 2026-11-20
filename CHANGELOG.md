# Changelog

Releases use dated version numbers (`vYYYY.MM`). Each GitHub release is archived on Zenodo with its own DOI.

## v2026.11

Release accompanying Hoggard et al. (2026, *Geochemistry, Geophysics, Geosystems*). Seeded runs with the
settings of that paper give identical results to the scripts first uploaded here in December 2025 (apart from
the corrected Gaussian normalisation of the likelihood, which shifts the log-probability by a constant and does
not change the posterior).

- The code is now a Python package with a `geofit` command, comments citing the source equations, and tests.
- `geofit forward` computes a single geotherm.
- New options `--seed` and `--output-dir`. The unused plots, and `--crust-K` (which had no effect), are removed.
- `--invertHmantle` works, with a truncated Gaussian prior on mantle heat production (`--Hmantlemean`,
  `--Hmantlestd`, `--Hmantlemin`, `--Hmantlemax`).
- The `Hasterok` crustal conductivity model (Goes et al., 2020) can be selected, with its units corrected.
- The crustal model `Whittington` is renamed `Hoggard`, so `--crust Hoggard --mantle Hoggard` selects the
  crustal and mantle models of Hoggard et al. (2026).
- `--mantle-grain-size` defaults to 0.3 cm (was 0.1 cm), as used by Hoggard et al. (2026).
- Each thermal model can only be used for its layer: `--crust` takes `Constant`, `Hasterok` or `Hoggard`, and
  `--mantle` takes `Constant`, `Osako` or `Hoggard`.
- Geotherms that reach the adiabat within the crust are rejected; these models previously produced warnings and
  were never accepted.
- Fixed: `--approxg` gravity below 80 km; posterior geotherms now use `--Hmantle` when it is not inverted for;
  `_prior.csv` reports the crustal heat production standard deviation correctly.
