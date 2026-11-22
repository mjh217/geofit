"""Seeded inversions reproduce reference outputs exactly.

The references in tests/data were made by seeding numpy's random generator (which emcee and scipy
draw from). The "osako" reference was made with the GeoFit scripts as used for Hoggard et al. (2026),
before the code became a package. The "paper" reference was also made with them, and so reproduced by
v2026.11, but has since been remade for the fix to alpha(P, T) in the Hoggard models (see CHANGELOG),
which moved only the LAB depths and geotherms. Short runs keep the tests quick (about a minute in all).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geofit.cli import main

DATA = Path(__file__).parent / "data"

RUNS = {
    # The settings of Hoggard et al. (2026): Hoggard crust and mantle, grain size 0.3 cm
    "paper": ["--seed", "11", "--crust", "Hoggard", "--mantle", "Hoggard", "--invertTp", "--samplepriors",
              "--priorsamples", "200"],
    # Other options: Osako mantle, data uncertainty as a parameter, no smoothing at the base
    "osako": ["--seed", "12", "--mantle", "Osako", "--hierarchical", "--alphaD", "0"],
}
SHORT = ["--oversampling", "2", "--burnin", "10", "--samples", "10"]


@pytest.mark.parametrize("run", RUNS)
def test_reproduces_reference(run, tmp_path):
    out = tmp_path / "output"
    assert main([str(DATA / "brigooda_TP.csv"), "--output-dir", str(out), *RUNS[run], *SHORT]) == 0

    for suffix in ["_posterior.csv", "_prior_samples.csv"]:
        reference = DATA / f"{run}{suffix}"
        if reference.exists():
            assert (out / f"brigooda_TP{suffix}").read_text() == reference.read_text(), suffix

    geotherms = np.load(out / "brigooda_TP_geotherm.npz", allow_pickle=True)["geotherm"]
    table = pd.concat(dict(enumerate(geotherms)), names=["sample", "row"])
    assert table.to_csv() == (DATA / f"{run}_geotherms.csv").read_text()


def test_prior_summary(tmp_path):
    main([str(DATA / "brigooda_TP.csv"), "--output-dir", str(tmp_path), "--seed", "1", *SHORT, "--samples", "5",
          "--invertHmantle"])
    prior = pd.read_csv(tmp_path / "brigooda_TP_prior.csv", index_col=0)
    assert prior.loc["Surface heat flux (mW per squared metre)"].tolist() == [10.0, 250.0]
    assert prior.loc["Crustal radiogenic heating (microW per cubic metre)"].tolist() == [1.0, 1.0]
    assert prior.loc["Mantle radiogenic heating (microW per cubic metre)"].tolist() == pytest.approx([0.01, 0.01])
