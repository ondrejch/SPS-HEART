"""The arXiv abstract's plain text keeps its operators (the second review round of 2026-11-01, item C3)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("package_arxiv", ROOT / "helpers" / "publishing" / "package_arxiv.py")
PA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PA)


def test_the_range_and_the_uncertainty_keep_their_operators():
    text = PA.plain_text(r"on $0 \le x_F \le 0.95$, with $\epsilon = 0.057 \pm 0.035$ and $S \approx 1.6$~units.")
    assert text == "on 0 <= x_F <= 0.95, with epsilon = 0.057 +/- 0.035 and S ~ 1.6 units."


def test_a_longer_command_is_not_cut_at_an_operator_prefix():
    # \left must not become "<=ft", \leq is its own operator
    assert PA.plain_text(r"$a \leq b$ and \textbf{bold}") == "a <= b and bold"
    assert "<=ft" not in PA.plain_text(r"$\left( x \right)$")


def test_percent_and_ties():
    assert PA.plain_text(r"within 6\%~of the data") == "within 6% of the data"
