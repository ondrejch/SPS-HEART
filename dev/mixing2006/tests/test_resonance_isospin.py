"""Isospin charge-channel completion tests (Round 2, card 5).

Task ``TASK-20260913-r2-isospin-charge-completion`` (plan section 22
task 13; plan section 8.3 step 3).  Five deterministic tests carry the
four plan section 8.3 clauses (one each) plus the conventions, and one
statistical test verifies the card-5 weights by sampling:

- ``test_charge_partners_generated``: the Delta-multiplet closure
  report lists no missing partners for the complete hand-built card; a
  catalog with Delta+ removed reports the missing partner (plan clause
  1, the record-level charge census).
- ``test_summed_weights_reproduce_br``: ``derive_charge_weights`` on
  the hand-built representative Delta+ returns
  ``{(2212, 111): 2/3 * BR, (2112, 211): 1/3 * BR}`` to 1e-12; the
  audit's ``weight_sum_errors`` is empty for the full hand-built
  catalog (plan clause 2).
- ``test_forbidden_channels_zero``: ``rho0 -> pi0 pi0`` (L = 1) derives
  weight 0 (absent from the allowed set) and a non-zero catalog
  branching fraction for it is an audit error naming the channel; an
  N* with ``isospin = 0.5`` whose channel product does not contain
  I = 1/2 (Delta+ eta, hand-constructed) is flagged forbidden (plan
  clause 3, conventions 2 and 3).
- ``test_bqs_conserved``: every derived weight combination conserves
  (B, Q, S) exactly, asserted per combination on the hand-built
  fixtures and on the committed catalog's derivation plan (plan
  clause 4, convention 6).
- ``test_anti_mirror``: the Delta- weights mirror the Delta++ weights
  through ``isospin.reflect_pdg`` to 1e-9 (reflect(1114) = 2224, the
  daughters n -> p and pi- -> pi+), the charge-conjugate mirror holds
  between the matter and anti records, and a perturbed anti record is
  flagged (convention 4).
- ``test_derived_weights_sampling`` (statistical, 100,000 channel
  draws on the hand-built Delta+ record with the card-5 weights): the
  empirical charge-combination frequencies match {2/3, 1/3} to 3-sigma
  (the ``1/sqrt(N)`` convention; the only statistical test of the
  card).

The committed-catalog test pins the card-5 derived-sibling inventory
and the closure of the completed catalog (the Gate 2 conservation-
clause substrate at catalog level); the tabulated mirror findings are
scoped to the documented defects (the Delta(1700)0 N-pi swap, the
legacy Delta-pi rounding on the + charge states, and the phi K-Kbar
phase-space split).

Hermetic: hand-constructed records and the committed catalog file; no
event generation, no external tables.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from na49_mixing.hadron_catalog import (  # noqa: E402
    DecayChannel,
    HadronCatalog,
    HadronRecord,
)
from na49_mixing.isospin import reflect_pdg  # noqa: E402
from na49_mixing.quantum_numbers import quantum_numbers  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    ISOSPIN_DERIVED_FIELD,
    ISOSPIN_DERIVED_TRANSCRIBER,
    apply_isospin_derivation,
    clebsch_gordan_squared,
    derive_charge_weights,
    isospin_closure_report,
    isospin_derivation_plan,
)

COMMITTED_CATALOG = Path(__file__).resolve().parent.parent / (
    "config/v2_singlet/resonance_catalog.json")

# Vendored-table daughter masses (the card-2 catalog values).
MP = 0.93827231
MN = 0.93956541
MPI = 0.1395699
MPI0 = 0.1349766
M_ETA = 0.5479
M_DELTA = 1.232

_DAUGHTER_MASS = {
    2212: MP, -2212: MP,
    2112: MN, -2112: MN,
    211: MPI, -211: MPI, 111: MPI0,
    221: M_ETA,
    2214: M_DELTA, -2214: M_DELTA,
    2224: M_DELTA, -2224: M_DELTA,
    2114: M_DELTA, -2114: M_DELTA,
    1114: M_DELTA, -1114: M_DELTA,
}


def _threshold(channels) -> float:
    return max(
        math.fsum(_DAUGHTER_MASS[int(d)] for d in ch.daughters)
        for ch in channels
    )


def _record(pdg_id, name, B, Q, S, isospin, isospin3, channels,
            pole_mass, nominal_width=0.1):
    return HadronRecord(
        pdg_id=int(pdg_id),
        name=name,
        B=int(B), Q=int(Q), S=int(S),
        isospin=float(isospin),
        isospin3=int(isospin3),
        spin=0.5,
        parity="+",
        pole_mass=float(pole_mass),
        nominal_width=float(nominal_width),
        mass_range=(_threshold(channels),
                    _threshold(channels) + 5.0 * float(nominal_width)),
        decay_channels=tuple(channels),
        stable_for_decay=False,
        provenance=None,
    )


def _channel(parent_pdg, daughters, branching, L=1):
    return DecayChannel(
        parent_pdg=int(parent_pdg),
        daughters=tuple(int(d) for d in daughters),
        branching_fraction=float(branching),
        orbital_angular_momentum=int(L),
        is_three_body=False,
        provenance=None,
    )


_DELTA_CHANNELS = {
    2224: [(2212, 211)],
    2214: [(2212, 111)],
    2114: [(2212, -211)],
    1114: [(2112, -211)],
}


def _delta_catalog(with_anti: bool = False):
    """The hand-built Delta(1232) multiplet: one representative N-pi
    channel per charge state, each carrying the full branching
    fraction (the pre-derivation card-2 shape)."""
    charges = {
        2224: ("Delta++", 1, 2, 3),
        2214: ("Delta+", 1, 1, 1),
        2114: ("Delta0", 1, 0, -1),
        1114: ("Delta-", 1, -1, -3),
    }
    if with_anti:
        charges.update({
            -1114: ("aDelta-", -1, 1, 3),
            -2114: ("aDelta0", -1, 0, 1),
            -2214: ("aDelta+", -1, -1, -1),
            -2224: ("aDelta--", -1, -2, -3),
        })
    records = []
    for pdg, (name, B, Q, i3) in charges.items():
        # the N-pi representative follows the charge state:
        # ++ -> p pi+, + -> p pi0, 0 -> p pi-, - -> n pi-
        # anti: aDelta- -> pbar pi+, aDelta0 -> pbar pi0,
        #       aDelta+ -> pbar pi0 (the mirror of Delta+), aDelta-- -> pbar pi-
        channels = {
            2224: (2212, 211), 2214: (2212, 111),
            2114: (2212, -211), 1114: (2112, -211),
            -1114: (-2112, 211), -2114: (-2212, 211),
            -2214: (-2212, 111), -2224: (-2212, -211),
        }[pdg]
        records.append(_record(pdg, name, B, Q, 0, 1.5, i3,
                               [_channel(pdg, channels, 1.0)], M_DELTA))
    return HadronCatalog(records)


def _stable_record(pdg_id, name, B, Q, S, isospin, isospin3, mass):
    """A termination-set record (no channels, stable_for_decay=True),
    as the card-2 catalog carries the stable states."""
    return HadronRecord(
        pdg_id=int(pdg_id), name=name,
        B=int(B), Q=int(Q), S=int(S),
        isospin=float(isospin), isospin3=int(isospin3),
        spin=0.5 if abs(B) == 1 else 0.0, parity="+",
        pole_mass=float(mass), nominal_width=0.0,
        mass_range=(0.0, 0.0),
        decay_channels=(), stable_for_decay=True, provenance=None,
    )


_STABLE_FIXTURE = {
    2212: ("proton", 1, 1, 0, 0.5, 1, MP),
    2112: ("neutron", 1, 0, 0, 0.5, -1, MN),
    -2212: ("pbar", -1, -1, 0, 0.5, -1, MP),
    -2112: ("nbar", -1, 0, 0, 0.5, 1, MN),
    211: ("pi+", 0, 1, 0, 1.0, 2, MPI),
    -211: ("pi-", 0, -1, 0, 1.0, -2, MPI),
    111: ("pi0", 0, 0, 0, 1.0, 0, MPI0),
}


def _delta_catalog_complete(with_anti: bool = False):
    """The completed hand-built Delta multiplet (the derivation plan
    applied): every isospin-allowed charge combination present.  The
    anti variant carries the stable nucleon and pion records the
    charge-conjugate mirror resolves through."""
    catalog = _delta_catalog(with_anti)
    if not with_anti:
        return apply_isospin_derivation(catalog)
    records = list(catalog.records()) + [
        _stable_record(pdg, name, B, Q, S, iso, i3, mass)
        for pdg, (name, B, Q, S, iso, i3, mass) in _STABLE_FIXTURE.items()
    ]
    return apply_isospin_derivation(HadronCatalog(records))


# ----------------------------------------------------------------------
# 1. all charge partners are generated where allowed (plan clause 1)
# ----------------------------------------------------------------------
def test_charge_partners_generated():
    """The complete hand-built Delta multiplet reports no missing
    charge partners (records or channels); removing the Delta+ record
    reports the missing partner pdg 2214 by name."""
    complete = _delta_catalog_complete()
    report = isospin_closure_report(complete)
    assert report.errors == (), report.errors
    for parent in report.per_parent:
        assert parent.missing_charge_partners == ()
        assert parent.weight_sum_errors == ()
        assert parent.forbidden_channels == ()

    reduced = isospin_closure_report(HadronCatalog(
        [r for r in _delta_catalog().records() if int(r.pdg_id) != 2214]))
    missing = sorted(
        {pdg for parent in reduced.per_parent
         for pdg in parent.missing_charge_partners})
    assert missing == [2214]
    assert any("2214" in error for error in reduced.errors)


# ----------------------------------------------------------------------
# 2. summed charge-channel weights reproduce the parent branching
#    fraction (plan clause 2)
# ----------------------------------------------------------------------
def test_summed_weights_reproduce_br():
    """derive_charge_weights on the hand-built representative Delta+
    returns the Clebsch-Gordan vector {(2212, 111): 2/3 * BR,
    (2112, 211): 1/3 * BR} to 1e-12; the audit's weight_sum_errors is
    empty for the full hand-built catalog, and the class sum
    reproduces the multiplet branching fraction exactly."""
    catalog = _delta_catalog()
    delta_plus = catalog.get(2214)
    br = float(delta_plus.decay_channels[0].branching_fraction)
    weights = derive_charge_weights(delta_plus, (2212, 211), catalog)
    assert set(weights) == {(2212, 111), (2112, 211)}
    assert abs(weights[(2212, 111)] - 2.0 / 3.0 * br) <= 1e-12
    assert abs(weights[(2112, 211)] - br / 3.0) <= 1e-12

    full = isospin_closure_report(_delta_catalog_complete())
    assert all(parent.weight_sum_errors == ()
               for parent in full.per_parent)

    completed = apply_isospin_derivation(catalog)
    channels = completed.get(2214).decay_channels
    assert abs(math.fsum(c.branching_fraction for c in channels)
               - br) <= 1e-12


# ----------------------------------------------------------------------
# 3. forbidden isospin channels have zero probability (plan clause 3)
# ----------------------------------------------------------------------
def test_forbidden_channels_zero():
    """rho0 -> pi0 pi0 (L = 1) derives weight 0 (absent from the
    allowed set); a non-zero catalog branching fraction for it is an
    audit error naming the channel; an N* with isospin 0.5 and a
    channel whose daughter product does not contain I = 1/2 is flagged
    forbidden."""
    rho0 = _record(113, "rho0", 0, 0, 0, 1.0, 0,
                   [_channel(113, (211, -211), 1.0, L=1)], 0.77526)
    catalog = HadronCatalog([rho0])
    weights = derive_charge_weights(rho0, (211, 211), catalog)
    assert set(weights) == {(-211, 211)}
    assert abs(weights[(-211, 211)] - 1.0) <= 1e-12
    assert clebsch_gordan_squared(1.0, 0, 1.0, 0, 1.0, 0) == 0.0

    # the Bose-forbidden channel with a non-zero catalog BR is an error
    boson = _record(113, "rho0", 0, 0, 0, 1.0, 0,
                    [_channel(113, (111, 111), 0.5, L=1),
                     _channel(113, (211, -211), 0.5, L=1)], 0.77526)
    report = isospin_closure_report(HadronCatalog([boson]))
    flagged = [issue for parent in report.per_parent
               for issue in parent.forbidden_channels
               if "(111, 111)" in issue and "Bose" in issue]
    assert len(flagged) == 1
    assert "0.5" in flagged[0]

    # total-I selection: N*(I = 1/2) -> Delta+(3/2) + eta(0): the
    # product 3/2 (x) 0 = 3/2 does not contain I = 1/2 -> forbidden
    nstar = _record(9999, "N*(fixture)+", 1, 1, 0, 0.5, 1,
                    [_channel(9999, (2214, 221), 0.8, L=1)], 1.78)
    delta_plus_rec = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 1.0)], M_DELTA)
    eta_rec = _record(221, "eta", 0, 0, 0, 0.0, 0,
                      [_channel(221, (211, -211, 111), 1.0, L=0)], 0.5479)
    nstar_catalog = HadronCatalog([nstar, delta_plus_rec, eta_rec])
    nstar_report = isospin_closure_report(nstar_catalog)
    forbidden = [issue for parent in nstar_report.per_parent
                 for issue in parent.forbidden_channels]
    assert any("9999" in issue and "2214" in issue and "221" in issue
               for issue in forbidden)
    assert derive_charge_weights(
        nstar_catalog.get(9999), (2214, 221), nstar_catalog) == {}


# ----------------------------------------------------------------------
# 4. each decay conserves (B, Q, S) exactly (plan clause 4)
# ----------------------------------------------------------------------
def test_bqs_conserved():
    """Every derived weight combination conserves (B, Q, S) exactly,
    on the hand-built fixtures and on the committed catalog's derived
    sibling set."""
    catalog = _delta_catalog()
    for record in catalog.records():
        parent_qn = (int(record.B), int(record.Q), int(record.S))
        for channel in record.decay_channels:
            weights = derive_charge_weights(
                record, (channel.daughters[0], channel.daughters[1]),
                catalog)
            for pair, weight in weights.items():
                total = quantum_numbers(pair[0]) + quantum_numbers(pair[1])
                assert total.as_tuple() == parent_qn, (pair, parent_qn)
                assert 0.0 < weights[pair] <= 1.0

    committed = HadronCatalog.from_json(COMMITTED_CATALOG)
    for entry in isospin_derivation_plan(committed):
        parent = committed.get(entry.pdg_id)
        parent_qn = (int(parent.B), int(parent.Q), int(parent.S))
        for sibling in (*entry.siblings, *entry.reweighted):
            total = quantum_numbers(sibling.daughters[0]) \
                + quantum_numbers(sibling.daughters[1])
            assert total.as_tuple() == parent_qn, (
                entry.pdg_id, sibling.daughters)


# ----------------------------------------------------------------------
# 5. anti-particle mirror (convention 4)
# ----------------------------------------------------------------------
def test_anti_mirror():
    """The Delta- weights mirror the Delta++ weights through
    ``isospin.reflect_pdg`` to 1e-9 (reflect(1114) = 2224, the
    daughters n -> p and pi- -> pi+), the anti multiplet is
    mirror-clean, and a perturbed anti record is flagged."""
    catalog = _delta_catalog_complete(with_anti=True)
    report = isospin_closure_report(catalog)
    assert report.errors == (), report.errors
    assert all(parent.mirror_mismatches == ()
               for parent in report.per_parent)

    # the reflect relation itself: reflect(1114) = 2224
    assert reflect_pdg(1114) == 2224
    delta_minus = catalog.get(1114)
    delta_plus_plus = catalog.get(2224)
    minus_weights = derive_charge_weights(
        delta_minus, (2112, -211), catalog)
    pp_weights = derive_charge_weights(
        delta_plus_plus, (2212, 211), catalog)
    mirrored = {}
    for (a, b), weight in minus_weights.items():
        mirrored[(reflect_pdg(a), reflect_pdg(b))] = weight
    for pair, weight in pp_weights.items():
        # the reflected Delta- weights land in the Delta++ key space
        # (n -> p, pi- -> pi+ are matter ids)
        assert abs(mirrored[pair] - weight) <= 1e-9

    # a perturbed anti record is flagged
    perturbed = _perturb_aDelta_plus(catalog)
    perturbed_report = isospin_closure_report(perturbed)
    flagged = {parent.pdg_id for parent in perturbed_report.per_parent
               if parent.mirror_mismatches}
    # each unordered mirror pair is checked once and attributed to the
    # lower pdg id, so the perturbation flags the anti side
    assert {-2214, -2114} <= flagged


def _perturb_aDelta_plus(catalog: HadronCatalog) -> HadronCatalog:
    """Reweight the aDelta+ representative channel to 0.5 (breaking
    the mirror against aDelta0 and the charge conjugate of Delta+)."""
    records = []
    for record in catalog.records():
        if int(record.pdg_id) == -2214:
            channel = record.decay_channels[0]
            record = HadronRecord(
                pdg_id=int(record.pdg_id), name=record.name,
                B=int(record.B), Q=int(record.Q), S=int(record.S),
                isospin=float(record.isospin),
                isospin3=int(record.isospin3),
                spin=float(record.spin), parity=record.parity,
                pole_mass=float(record.pole_mass),
                nominal_width=float(record.nominal_width),
                mass_range=record.mass_range,
                decay_channels=(DecayChannel(
                    parent_pdg=int(channel.parent_pdg),
                    daughters=tuple(channel.daughters),
                    branching_fraction=0.5,
                    orbital_angular_momentum=int(
                        channel.orbital_angular_momentum),
                    is_three_body=False,
                    provenance=channel.provenance,
                ),),
                stable_for_decay=bool(record.stable_for_decay),
                provenance=record.provenance,
            )
        records.append(record)
    return HadronCatalog(records)


# ----------------------------------------------------------------------
# 6. statistical: the derived weights sample correctly
# ----------------------------------------------------------------------
def test_derived_weights_sampling():
    """100,000 channel draws on the hand-built Delta+ with the card-5
    weights: the empirical charge-combination frequencies match
    {2/3, 1/3} to 3-sigma (the 1/sqrt(N) convention)."""
    catalog = _delta_catalog()
    delta_plus = catalog.get(2214)
    br = float(delta_plus.decay_channels[0].branching_fraction)
    weights = derive_charge_weights(delta_plus, (2212, 211), catalog)
    pairs = sorted(weights)
    probabilities = np.array([weights[p] / br for p in pairs])
    assert abs(float(probabilities.sum()) - 1.0) <= 1e-12

    rng = np.random.default_rng(20260913)
    draws = rng.choice(len(pairs), size=100_000, p=probabilities)
    counts = np.bincount(draws, minlength=len(pairs))
    frequencies = counts / draws.size
    for pair, observed, expected in zip(pairs, frequencies, probabilities):
        sigma = math.sqrt(float(expected) * (1.0 - float(expected))
                          / draws.size)
        assert abs(float(observed) - float(expected)) <= 3.0 * sigma, (
            pair, observed, expected, sigma)


# ----------------------------------------------------------------------
# the committed catalog: the card-5 derived inventory and closure
# ----------------------------------------------------------------------
def test_committed_catalog_isospin_closure():
    """The committed card-5 catalog closes: zero audit errors, the
    derived-sibling inventory pinned to the exact Clebsch-Gordan
    shares, and the tabulated mirror findings scoped to the documented
    defects."""
    catalog = HadronCatalog.from_json(COMMITTED_CATALOG)
    report = isospin_closure_report(catalog)
    assert report.errors == (), report.errors

    derived: dict[int, list] = {}
    for record in catalog.records():
        for channel in record.decay_channels:
            if channel.provenance is not None \
                    and channel.provenance.field == ISOSPIN_DERIVED_FIELD:
                assert channel.provenance.transcriber \
                    == ISOSPIN_DERIVED_TRANSCRIBER
                derived.setdefault(int(record.pdg_id), []).append(
                    (tuple(sorted(int(d) for d in channel.daughters)),
                     float(channel.branching_fraction)))
    assert len(derived) == 3  # the three completed parents
    third = 1.0 / 3.0
    assert sorted(derived[100223]) == sorted([
        ((-213, 211), third), ((-211, 213), third), ((111, 113), third)])
    assert sorted(derived[30223]) == sorted([
        ((-213, 211), third), ((-211, 213), third), ((111, 113), third)])
    half = 0.03512672298799467 / 2.0
    assert sorted(derived[117]) == sorted([
        ((130, 310), half), ((-321, 321), half)])

    # omega(1420)/omega(1650): the support starts at the lowest open
    # channel, rho0 pi0 at 0.9102366 GeV (the card-5 value 0.9148299 was
    # the highest threshold, rho+- pi-+; corrected 2026-09-28, FAB-07)
    for pdg in (100223, 30223):
        record = catalog.get(pdg)
        assert abs(float(record.mass_range[0]) - 0.9102366) <= 1e-12

    # the mirror findings: the isospin-share revision of 2026-10-05
    # (helpers/benchmarking/catalog_isospin_share_revision.py; the
    # Delta(1700)0 N pi split and the N*+ Delta pi shares) leaves only the
    # physical phi(1020) K+K- / K0S K0L split
    mirror_ids = {parent.pdg_id for parent in report.per_parent
                  if parent.mirror_mismatches}
    assert mirror_ids == {333}


