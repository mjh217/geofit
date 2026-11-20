"""The forward command."""

import pandas as pd
import pytest

from geofit.cli import main
from geofit.output import GEOTHERM_COLUMNS


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
