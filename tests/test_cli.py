"""The forward command, and options of the inversion."""

from pathlib import Path

import pandas as pd
import pytest

from geofit import defaults as d
from geofit import forward
from geofit.cli import main
from geofit.materials import make_model
from geofit.output import GEOTHERM_COLUMNS

DATA = Path(__file__).parent / "data"
SHORT = ["--oversampling", "2", "--burnin", "10", "--samples", "10"]


def test_forward(tmp_path, capsys):
    out = tmp_path / "geotherm.csv"
    args = ["forward", "--surfaceheatflux", "0.06", "--zcrust", "35", "--hcrust", "1e-6", "-o", str(out)]
    assert main(args) == 0
    assert "LAB depth" in capsys.readouterr().out
    table = pd.read_csv(out, index_col=0)
    assert list(table.columns) == GEOTHERM_COLUMNS
    assert table["Temperature (Celsius)"].iloc[0] == 0.0
    assert table["Depth (km)"].iloc[-1] == 400.0

    # A hotter mantle gives a shallower base
    assert main([*args, "--Tp", str(1400 + 273.15), "-o", str(tmp_path / "hot.csv")]) == 0
    assert "LAB depth" in capsys.readouterr().out


def test_forward_impossible(tmp_path):
    with pytest.raises(SystemExit, match="surface heat flux too high"):
        main(["forward", "--surfaceheatflux", "0.25", "--zcrust", "35", "--hcrust", "1e-6", "-o", str(tmp_path / "x")])


def test_adiabat_margin(tmp_path, capsys):
    """Data within the margin of the adiabat of the prior mean (or fixed) Tp are discarded before fitting."""
    data = pd.read_csv(DATA / "brigooda_TP.csv")
    T, P = data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy() * 1e9
    mantle = make_model("constant", "mantle", d.ALPHA_MANTLE, d.CP_MANTLE, d.RHO_MANTLE, d.K_MANTLE)
    margin = 400.0

    for options, tp in [([], d.TP), (["--invertTp", "--tpmean", "1600", "--tpstd", "20"], 1600)]:
        adiabat = forward.compute_adiabat(mantle, forward.gravity_constant, tp, d.P_SURFACE, d.Z_MAX_ADIABAT)
        expected = T > adiabat.T(P) - d.C2K - margin
        assert 0 < expected.sum() < expected.size  # the margin splits this dataset

        out = tmp_path / str(tp)
        args = [str(DATA / "brigooda_TP.csv"), "--output-dir", str(out), "--seed", "1", "--adiabat-margin",
                str(margin), *options, *SHORT]
        assert main(args) == 0
        assert f"Discarding {expected.sum()} of {expected.size}" in capsys.readouterr().out
        table = pd.read_csv(out / "brigooda_TP_adiabat_margin.csv", index_col=0)
        assert table["Discarded"].tolist() == expected.tolist()


def test_adiabat_margin_errors(tmp_path):
    args = [str(DATA / "brigooda_TP.csv"), "--output-dir", str(tmp_path), *SHORT]
    with pytest.raises(SystemExit, match="No P-T data left"):
        main([*args, "--adiabat-margin", "2000"])
    with pytest.raises(SystemExit):
        main([*args, "--adiabat-margin", "-10"])
