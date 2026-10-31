"""The p_T-integrated p+Pb pion tables of the centrality samples (thesis Tab.
``tabptiall``), converted 2026-09-29 by ``pa_data/convert_ptint_centrality.py``:
file set, headers, x_F grids, positivity, and the <nu> ordering of the sets."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

PA = Path(__file__).resolve().parents[1] / "pa_data"
SAMPLES = {"alltrig": ("5.1", 9), "set1": ("2.9", 9), "set2": ("4.1", 9), "set3": ("5.2", 9), "set4": ("6.0", 9), "set5": ("6.9", 8)}
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _read(path: Path):
    header = [l for l in path.read_text().splitlines() if l.startswith("#")]
    body = np.array([[float(x) for x in l.split()] for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")])
    return header, body


@pytest.mark.parametrize("charge", ["piplus", "piminus"])
@pytest.mark.parametrize("sample", list(SAMPLES))
def test_centrality_ptint_file(sample, charge):
    nu, n_rows = SAMPLES[sample]
    header, body = _read(PA / f"ptintegrated_{charge}_pPb_{sample}.txt")
    assert any(f"<nu> = {nu}" in l for l in header)
    assert any("Tab. 'tabptiall'" in l for l in header)
    assert body.shape == (n_rows, 7)
    assert np.allclose(body[:, 0], GRID[:n_rows])
    assert np.all(body[:, [1, 3, 5]] > 0.0)
    # the absolute errors are value x percent / 100 of the thesis: positive and below
    # the values, except where the header documents a non-positive printed error
    documented = {round(float(m.group(1)), 3) for m in (re.search(r"at xF = ([0-9.]+) ", l) for l in header
                                                    if "statistical error at" in l) if m}
    for row in body:
        errors = row[[2, 4, 6]]
        if round(float(row[0]), 3) in documented:
            continue
        assert np.all(errors > 0.0) and errors[0] < row[1] and errors[1] < row[3], (sample, charge, row)


def test_the_one_documented_source_anomaly():
    header, body = _read(PA / "ptintegrated_piplus_pPb_set2.txt")
    assert any("int.002.tex L9" in l and "-1.0 %" in l for l in header)
    assert body[np.isclose(body[:, 0], 0.40), 2][0] < 0.0


def test_sets_are_ordered_in_nu_and_the_forward_density_falls():
    nus = []
    forward = []
    for s in ("set1", "set2", "set3", "set4", "set5"):
        header, body = _read(PA / f"ptintegrated_piminus_pPb_{s}.txt")
        nus.append(float(re.search(r"<nu> = ([0-9.]+)", " ".join(header)).group(1)))
        forward.append(body[np.isclose(body[:, 0], 0.30), 1][0])
    assert nus == sorted(nus)
    assert all(a > b for a, b in zip(forward, forward[1:]))  # stopping: fewer forward pions with nu


BOIMSKA = {"proton": 46, "antiproton": 38, "pi_minus": 43, "pi_plus": 46}
LASZLO = {"pi_plus": 20, "pi_minus": 20, "proton": 21, "antiproton": 13, "k_plus": 20, "k_minus": 18}


@pytest.mark.parametrize("species", list(BOIMSKA))
def test_boimska_central_tables(species):
    header, body = _read(PA / f"pPb_central_boimska2004_{species}.txt")
    assert any("CERN-THESIS-2004-035" in l for l in header)
    assert body.shape == (BOIMSKA[species], 5)
    assert np.all(body[:, 3] > 0.0) and np.all((body[:, 4] > 0.0) & (body[:, 4] < body[:, 3]))
    assert np.all(body[:, 1] > body[:, 0])


def test_boimska_spot_values():
    # values as printed x 10^3 in the source tables, divided by 10^3 in the files
    spots = {("proton", 0.0, 0.05, 0.125): (0.203, 0.008), ("antiproton", 0.0, 0.1, 1.790): (0.000060, 0.000014),
             ("pi_minus", 0.0, 0.05, 0.300): (1.291, 0.015), ("pi_plus", 0.3, 0.45, 1.790): (0.00018, 0.00002)}
    for (sp, lo, hi, pt), (v, e) in spots.items():
        _, body = _read(PA / f"pPb_central_boimska2004_{sp}.txt")
        row = body[np.isclose(body[:, 0], lo) & np.isclose(body[:, 1], hi) & np.isclose(body[:, 2], pt)]
        assert row.shape[0] == 1 and np.isclose(row[0, 3], v) and np.isclose(row[0, 4], e), (sp, lo, hi, pt)


@pytest.mark.parametrize("species", list(LASZLO))
def test_laszlo_y0_tables(species):
    header, body = _read(PA / f"pPb_y0_laszlo2008_{species}.txt")
    assert any("Table 12" in l for l in header) and any("NOT minimum bias" in l for l in header)
    assert body.shape == (LASZLO[species], 4)
    assert np.all(np.diff(body[:, 0]) > 0) and np.all(body[:, 2] > 0) and np.all(body[:, 3] > 0)
