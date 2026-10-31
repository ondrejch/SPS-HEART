"""Branching-closure and spectral-normalization self-check tests
(Round 2, card 8).

Task ``TASK-20260913-r2-branching-spectral-normalization-tests`` (plan
section 22 task 16; plan sections 8.2, 8.4).  Four deterministic tests
carry the card's pre-registered tolerances on hand-constructed records:

- ``test_branching_overnormalized``: a hand-built parent whose
  branching fractions sum to ``1.0000001`` is a violation naming the
  parent (the over-normalization clause: sums may exceed 1 by at most
  ``1e-9``).
- ``test_branching_gap_flagged``: a hand-built parent with residual
  ``0.25`` is a ``violations`` entry (the residual-budget clause:
  residuals above ``0.20`` are unexplained gaps); a parent with
  residual ``0.15`` is recorded in ``per_parent`` without a violation.
- ``test_branching_radiative_budget_adjusts_bound``: the card-2
  re-open of the card-8 check round (the eta/eta-prime radiative
  budgets): a documented radiative budget adjusts the record's
  residual bound to ``0.20 + radiative_budget`` (residual ``0.25``
  with budget ``0.10`` is explained, with budget ``0.04`` stays a
  violation naming the documented budget), and a record without a
  documented budget keeps the plain pre-registered bound.
- ``test_spectral_integral``: a hand-built Delta-like record has
  integral ``1`` within ``1e-6`` on the pre-registered 10,001-node
  grid, and the support-discipline values at ``threshold - 1e-12`` and
  ``m_up + 1e-12`` are exactly ``0.0`` (through the report fields and
  through direct card-3 ``spectral_function`` calls).
- ``test_open_weight_sweep``: a hand-built record with all channels
  open has no violations on its 100-point grid; a hand-built record
  whose width law goes to zero inside its support (the channel closed
  at the pole: ``q(M0) = 0``, zero width at every mass) is a violation
  naming the mass (the width-normalization guard of plan section 8.4).

Statistical card test:

- ``test_spectral_sampling_1e6`` (1,000,000 draws through the card-3
  ``draw_mass`` on the hand-built Delta-like record over the full
  support, fixed seed): every draw lies in the support (the sampling
  complement of the open-weight guard), and the 10-bin draw histogram
  matches the normalized density to 3-sigma per bin (the ``1/sqrt(N)``
  convention; the per-bin expectation is the 1,001-node trapezoid
  integral of ``rho`` over the bin).

Committed-catalog resolution test (card-2 re-open):

- ``test_committed_catalog_radiative_budgets_resolve_violations``:
  the card-2 committed catalog documents the eta (pdg 221) and
  eta-prime(958) (pdg 377) radiative/leptonic budgets on the additive
  ``radiative_budget`` record field (the measured non-hadronic modes
  of the PDG 2024 light unflavored mesons summary table, limits
  excluded), and the branching-closure check evaluated with those
  budgets reports zero violations while the same check without them
  reports exactly the two documented residuals; the pre-registered
  ``0.20`` constant is unchanged.  The orphan-record revision of
  2026-10-29 removed the eta-prime record at 377 (no class and no
  channel reaches it; the kernel's eta' is the record 331 of
  ``KernelConfig.eta_prime``), so the eta's is the one budget left.

The full-catalog reports run over the card-2 committed catalog in the
card's acceptance sweep (the run-dir report), which reads only
``config/v2_singlet/resonance_catalog.json`` and the vendored PDG
table; the unit tests here are hermetic hand-constructed records.
Provenance is ``None`` by design: hand-constructed fixtures sit before
the catalog import, where the card-1 schema permits it.

Hermetic except the one committed-catalog resolution test: hand-
constructed records only; no event generation, no external tables;
the only file read is ``config/v2_singlet/resonance_catalog.json``
(the same committed file the card-2 committed-catalog tests read).
"""

from __future__ import annotations

import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
import numpy as np  # noqa: E402

from na49_mixing.hadron_catalog import (  # noqa: E402
    DecayChannel,
    HadronCatalog,
    HadronRecord,
)
from na49_mixing.pdg_database import get_mass  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    BRANCHING_SUM_TOL,
    OPEN_WEIGHT_SWEEP_POINTS,
    RESIDUAL_BUDGET_BOUND,
    SPECTRAL_INTEGRAL_TOL,
    SPECTRAL_QUADRATURE_NODES,
    SUPPORT_PROBE_EPS,
    branching_closure_report,
    open_weight_sweep,
    radiative_budgets_from_json,
    spectral_normalization_report,
)
from na49_mixing.resonance_widths import (  # noqa: E402
    draw_mass,
    spectral_function,
)

# Vendored-table masses (the card-2 catalog values), GeV/c^2.
MP = 0.93827231
MPI = 0.1395699
MPI0 = 0.1349766

# The committed card-2 catalog (read only by the committed-catalog
# resolution test below; the card-2 re-open of the card-8 round).
CATALOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config" / "v2_singlet" / "resonance_catalog.json"
)

# The documented radiative/leptonic budgets of the two eta records: the
# measured non-hadronic modes of the PDG 2024 light unflavored mesons
# summary table (S. Navas et al., Phys. Rev. D 110, 030001 (2024),
# pdg.lbl.gov/2024/tables/rpp2024-tab-mesons-light.pdf), the same table
# the eta-prime channels cite; limits are excluded.  eta (pdg 221):
# 2gamma, pi+ pi- gamma, e+ e- gamma, pi0 2gamma, mu+ mu- gamma,
# 2e+ 2e-, pi+ pi- e+ e- (gamma), mu+ mu-.
ETA_RADIATIVE_ITEMS = (0.3936, 0.0428, 0.0069, 0.000255, 0.00031,
                       0.000024, 0.000268, 0.0000058)
ETA_PDG_ID = 221
ETAPRIME_PDG_ID = 377      # card-pinned key; the true PDG MC id is 331

# Statistical test (pre-registered draw count and seed).
N_DRAWS = 1_000_000
SAMPLING_SEED = 20260913
N_BINS = 10


# ----------------------------------------------------------------------
# hand-built fixtures
# ----------------------------------------------------------------------
def _channel(parent_pdg, daughters, branching, L=0):
    return DecayChannel(
        parent_pdg=int(parent_pdg),
        daughters=tuple(int(d) for d in daughters),
        branching_fraction=float(branching),
        orbital_angular_momentum=int(L),
        is_three_body=False,
        provenance=None,
    )


_DAUGHTER_POLE = {2212: MP, 2112: MP + 0.001293, 211: MPI, -211: MPI,
                  111: MPI0}


def _record(pdg_id, name, channels, pole_mass, nominal_width, m_up):
    """A hand-built decayable record; the threshold is the highest
    channel threshold (the card-1 structural rule)."""
    threshold = max(
        math.fsum(_DAUGHTER_POLE[int(d)] for d in ch.daughters)
        for ch in channels
    )
    return HadronRecord(
        pdg_id=int(pdg_id),
        name=name,
        B=1, Q=1, S=0,
        isospin=0.5, isospin3=1,
        spin=0.5, parity="+",
        pole_mass=float(pole_mass),
        nominal_width=float(nominal_width),
        mass_range=(float(threshold), float(m_up)),
        decay_channels=tuple(channels),
        stable_for_decay=False,
        provenance=None,
    )


def _delta_like() -> HadronRecord:
    """The hand-built Delta(1232)-like record: one N-pi channel at
    L = 1, the card-2 Delta dumphigh upper cut ``m_up = 2.0``."""
    return _record(
        2214, "Delta+",
        [_channel(2214, (2212, 111), 1.0, L=1)],
        pole_mass=1.232, nominal_width=0.117, m_up=2.0,
    )


@lru_cache(maxsize=1)
def _delta_masses() -> dict[int, float]:
    record = _delta_like()
    return {
        int(d): float(get_mass(int(d)))
        for channel in record.decay_channels
        for d in channel.daughters
    }


# ----------------------------------------------------------------------
# deterministic: branching closure
# ----------------------------------------------------------------------
def test_branching_overnormalized():
    """A parent with BR sum 1.0000001 is a violation naming the parent
    (the pre-registered ``1e-9`` over-normalization bound)."""
    parent = _record(
        87000311, "Pover",
        [_channel(87000311, (211, -211), 0.5, L=0),
         _channel(87000311, (2212, 111), 0.5000001, L=0)],
        pole_mass=1.3, nominal_width=0.15, m_up=1.9,
    )
    report = branching_closure_report(HadronCatalog([parent]))
    entry = next(e for e in report.per_parent if e.pdg_id == 87000311)
    assert abs(entry.hadronic_br_sum - 1.0000001) <= 1e-15
    assert entry.hadronic_br_sum > 1.0 + BRANCHING_SUM_TOL
    assert abs(entry.residual_budget - (1.0 - 1.0000001)) <= 1e-15
    assert not entry.within_bound
    assert report.violations, "the over-normalized parent must be flagged"
    message = " ".join(report.violations)
    assert "Pover" in message and "87000311" in message
    assert "over-normalized" in message
    assert str(BRANCHING_SUM_TOL) in message
    # the residual (negative here) is not the flagged clause
    assert all("residual budget" not in v for v in report.violations)


def test_branching_gap_flagged():
    """A parent with residual 0.25 is a ``violations`` entry; a parent
    with residual 0.15 is recorded without a violation (the
    pre-registered residual bound 0.20)."""
    gap = _record(
        87000321, "Pgap",
        [_channel(87000321, (211, -211), 0.75, L=0)],
        pole_mass=1.3, nominal_width=0.15, m_up=1.9,
    )
    fine = _record(
        87000322, "Pfine",
        [_channel(87000322, (211, -211), 0.85, L=0)],
        pole_mass=1.3, nominal_width=0.15, m_up=1.9,
    )
    report = branching_closure_report(HadronCatalog([gap, fine]))
    by_pdg = {e.pdg_id: e for e in report.per_parent}
    assert abs(by_pdg[87000321].residual_budget - 0.25) <= 1e-12
    assert abs(by_pdg[87000322].residual_budget - 0.15) <= 1e-12
    assert not by_pdg[87000321].within_bound
    assert by_pdg[87000322].within_bound
    gap_messages = [v for v in report.violations if "Pgap" in v]
    assert len(gap_messages) == 1, report.violations
    assert "residual budget" in gap_messages[0]
    assert "0.25" in gap_messages[0]
    assert str(RESIDUAL_BUDGET_BOUND) in gap_messages[0]
    assert all("Pfine" not in v for v in report.violations)
    assert abs(report.max_residual - 0.25) <= 1e-12


def test_branching_radiative_budget_adjusts_bound():
    """The card-2 re-open clause: a documented radiative budget adjusts
    the record's residual bound to ``0.20 + radiative_budget`` (the
    pre-registered ``0.20`` constant is unchanged).  A residual 0.25
    with documented budget 0.10 is explained (no violation,
    ``within_bound``); the same residual with budget 0.04 (adjusted
    bound 0.24) stays a violation naming the documented budget; a
    record without a documented budget keeps the plain pre-registered
    bound (the unexplained-gap clause, as before)."""
    prad = _record(
        87000341, "Prad",
        [_channel(87000341, (211, -211), 0.75, L=0)],
        pole_mass=1.3, nominal_width=0.15, m_up=1.9,
    )
    assert abs((1.0 - 0.75) - 0.25) <= 1e-15

    # explained: the adjusted bound 0.20 + 0.10 absorbs the residual
    explained = branching_closure_report(
        HadronCatalog([prad]), radiative_budgets={87000341: 0.10})
    entry = next(e for e in explained.per_parent if e.pdg_id == 87000341)
    assert entry.radiative_budget == 0.10
    assert entry.within_bound
    assert explained.violations == ()
    assert abs(explained.max_residual - 0.25) <= 1e-12

    # exceeded: the adjusted bound 0.20 + 0.04 stays below the residual
    exceeded = branching_closure_report(
        HadronCatalog([prad]), radiative_budgets={87000341: 0.04})
    entry = next(e for e in exceeded.per_parent if e.pdg_id == 87000341)
    assert entry.radiative_budget == 0.04
    assert not entry.within_bound
    exceeded_messages = [v for v in exceeded.violations if "Prad" in v]
    assert len(exceeded_messages) == 1, exceeded.violations
    assert "residual budget" in exceeded_messages[0]
    assert "0.25" in exceeded_messages[0]
    assert "0.04" in exceeded_messages[0]
    assert "radiative" in exceeded_messages[0]

    # absent budget: the plain pre-registered bound (as before)
    absent = branching_closure_report(HadronCatalog([prad]))
    entry = next(e for e in absent.per_parent if e.pdg_id == 87000341)
    assert entry.radiative_budget == 0.0
    assert not entry.within_bound
    absent_messages = [v for v in absent.violations if "Prad" in v]
    assert len(absent_messages) == 1, absent.violations
    assert "residual budget" in absent_messages[0]
    assert str(RESIDUAL_BUDGET_BOUND) in absent_messages[0]


def test_committed_catalog_radiative_budgets_resolve_violations():
    """The card-2 re-open resolution on the committed catalog, after the
    radiative revision of 2026-10-02.  The Round 2 records left the
    radiative modes of eta and eta-prime out (residuals 0.4442 and 0.351,
    above the pre-registered 0.20 bound) and documented them on the
    additive ``radiative_budget`` field.  The revision represents the
    dominant radiative modes as channels with photons (eta: gamma gamma,
    pi+ pi- gamma; eta-prime: rho0 gamma, omega gamma, gamma gamma), so the
    remaining budgets are the small Dalitz and leptonic modes, every
    residual lies within the 0.20 bound, and the check reports no
    violation with or without the budgets.  The eta-prime record left the
    committed catalog in the orphan-record revision of 2026-10-29."""
    catalog = HadronCatalog.from_json(CATALOG_PATH)
    budgets = radiative_budgets_from_json(CATALOG_PATH)
    assert set(budgets) == {ETA_PDG_ID}, budgets
    assert ETAPRIME_PDG_ID not in {int(r.pdg_id) for r in catalog.records()}
    for pdg in (ETA_PDG_ID,):
        record = catalog.get(pdg)
        represented = math.fsum(ch.branching_fraction for ch in record.decay_channels)
        assert budgets[pdg] == pytest.approx(1.0 - represented, abs=1e-9)
    # the eta channels carry the two leading measured radiative modes; the
    # rest of the itemized measured modes is the remaining budget
    eta = catalog.get(ETA_PDG_ID)
    radiative = {tuple(sorted(int(d) for d in ch.daughters)): ch.branching_fraction
                 for ch in eta.decay_channels if ch.is_radiative}
    assert radiative == {(22, 22): ETA_RADIATIVE_ITEMS[0], (-211, 22, 211): ETA_RADIATIVE_ITEMS[1]}
    assert budgets[ETA_PDG_ID] == pytest.approx(math.fsum(ETA_RADIATIVE_ITEMS[2:]), abs=1e-4)

    plain = branching_closure_report(catalog)
    assert plain.violations == (), plain.violations
    assert abs(plain.max_residual - budgets[ETA_PDG_ID]) <= 1e-9
    adjusted = branching_closure_report(catalog, radiative_budgets=budgets)
    assert adjusted.violations == (), adjusted.violations
    assert adjusted.max_residual == plain.max_residual
    entries = {e.pdg_id: e for e in adjusted.per_parent}
    for pdg in (ETA_PDG_ID,):
        assert entries[pdg].radiative_budget == budgets[pdg]
        assert entries[pdg].within_bound
    assert len(adjusted.per_parent) == len(plain.per_parent)


# ----------------------------------------------------------------------
# deterministic: spectral normalization
# ----------------------------------------------------------------------
def test_spectral_integral():
    """The hand-built Delta-like record: the 10,001-node trapezoid
    integral of the card-3 rho over the support is 1 within 1e-6, and
    the support-discipline values at ``threshold - 1e-12`` and
    ``m_up + 1e-12`` are exactly 0.0."""
    record = _delta_like()
    masses = _delta_masses()
    catalog = HadronCatalog([record])
    report = spectral_normalization_report(catalog)
    assert len(report.per_record) == 1
    entry = report.per_record[0]
    assert entry.pdg_id == 2214
    assert abs(entry.integral - 1.0) <= SPECTRAL_INTEGRAL_TOL, entry.integral
    assert entry.max_below_threshold_value == 0.0
    assert entry.max_above_uppercut_value == 0.0
    assert entry.within_tolerance
    assert report.violations == ()
    assert abs(report.max_abs_error - abs(entry.integral - 1.0)) <= 1e-18

    # the same probes directly through the card-3 surface
    lo, hi = record.mass_range
    assert spectral_function(lo - SUPPORT_PROBE_EPS, record, masses) == 0.0
    assert spectral_function(hi + SUPPORT_PROBE_EPS, record, masses) == 0.0
    # the report's grid is the pre-registered one
    assert SPECTRAL_QUADRATURE_NODES == 10_001
    # a positive density inside the support (the fixture is alive)
    assert spectral_function(0.5 * (lo + hi), record, masses) > 0.0


# ----------------------------------------------------------------------
# deterministic: open-weight sweep
# ----------------------------------------------------------------------
def test_open_weight_sweep():
    """(a) A hand-built record with all channels open has no violations
    on its 100-point grid; (b) a record whose width law goes to zero
    inside its support (the channel closed at the pole) is a violation
    naming the mass."""
    # (a) the Delta-like fixture: every grid mass above the threshold
    # carries a finite, strictly positive total open weight
    record = _delta_like()
    report = open_weight_sweep(HadronCatalog([record]))
    assert report.violations == ()
    assert report.records_checked == 1
    assert report.points_checked == OPEN_WEIGHT_SWEEP_POINTS

    # (b) the channel closed at the pole: q(M0) = 0, so the width law
    # returns exactly 0.0 at every mass although the realized masses
    # are strictly above the record threshold
    zero = _record(
        87000331, "Pzero",
        [_channel(87000331, (2212, 211), 1.0, L=0)],
        pole_mass=1.0, nominal_width=0.2, m_up=2.5,
    )
    assert zero.mass_range[0] > zero.pole_mass  # closed at the pole
    report_zero = open_weight_sweep(HadronCatalog([zero]))
    assert report_zero.violations, "the zero-width record must be flagged"
    message = " ".join(report_zero.violations)
    assert "Pzero" in message and "87000331" in message
    assert "total open weight" in message
    # the violation names a mass strictly inside the support, and it is
    # the first above-threshold grid point of the pre-registered grid
    lo, hi = zero.mass_range
    match = re.search(r"at mass ([^ ]+) GeV", message)
    assert match, message
    named_mass = float(match.group(1))
    expected_first = lo + (hi - lo) / (OPEN_WEIGHT_SWEEP_POINTS - 1)
    assert lo < named_mass <= hi
    assert abs(named_mass - expected_first) <= 1e-12


# ----------------------------------------------------------------------
# statistical: 1,000,000 draws through the card-3 draw_mass
# ----------------------------------------------------------------------
def _draw_sample():
    """1,000,000 draws from the card-3 ``draw_mass`` on the hand-built
    Delta-like record over the full support (fixed seed)."""
    record = _delta_like()
    masses = _delta_masses()
    lo, hi = record.mass_range
    rng = np.random.default_rng(SAMPLING_SEED)
    draws = np.empty(N_DRAWS, dtype=np.float64)
    for i in range(N_DRAWS):
        draws[i] = draw_mass((lo, hi), record, masses, rng)
    return record, masses, draws


def _bin_probabilities(record, masses):
    """The normalized density's mass in each of the 10 equal bins of
    the support (the 1,001-node trapezoid integral of ``rho`` per
    bin)."""
    lo, hi = record.mass_range
    edges = np.linspace(lo, hi, N_BINS + 1)
    probabilities = []
    for k in range(N_BINS):
        xs = np.linspace(edges[k], edges[k + 1], 1_001)
        ys = np.array([
            spectral_function(float(x), record, masses) for x in xs
        ])
        probabilities.append(float(
            (0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum()
        ))
    return edges, probabilities


def test_spectral_sampling_1e6():
    """1,000,000 draws over the full support: every draw lies in the
    support (the sampling complement of the open-weight guard), and
    the 10-bin draw histogram matches the normalized density to
    3-sigma per bin (the ``1/sqrt(N)`` convention)."""
    record, masses, draws = _draw_sample()
    lo, hi = record.mass_range
    assert draws.shape == (N_DRAWS,)
    assert np.all(np.isfinite(draws))
    assert np.all(draws >= lo) and np.all(draws <= hi), (
        "every draw must lie in the support"
    )

    edges, probabilities = _bin_probabilities(record, masses)
    counts, _ = np.histogram(draws, bins=edges)
    total_probability = math.fsum(probabilities)
    # the bin probabilities tile the support: their sum is 1 to the
    # trapezoid accuracy of the density (far below the 3-sigma scale)
    assert abs(total_probability - 1.0) <= 1e-4, total_probability
    for k in range(N_BINS):
        p = probabilities[k]
        expected = N_DRAWS * p
        sigma = math.sqrt(N_DRAWS * p * (1.0 - p))
        deviation = abs(int(counts[k]) - expected)
        assert deviation <= 3.0 * sigma, (
            f"bin {k}: {int(counts[k])} draws vs expected {expected:.1f} "
            f"(deviation {deviation:.1f}, 3 sigma {3.0 * sigma:.1f})"
        )
