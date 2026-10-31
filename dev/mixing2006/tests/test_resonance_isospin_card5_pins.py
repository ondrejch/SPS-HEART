"""Card-5 isospin layer: the committed-behavior pins (Round 2, card 5).

Task ``TASK-20260913-r2-isospin-charge-completion`` (plan section 22
task 13; plan section 8.3 step 3).  This sibling of
``tests/test_resonance_isospin.py`` pins the committed card-5 surface of
``na49_mixing/resonance_decay.py`` on the scope items the primary file
leaves open:

- ``test_clebsch_gordan_pre_registered_values``: the pre-registered
  Clebsch-Gordan values of the card (Delta+ ``p pi0 : n pi+ = 2/3 : 1/3``,
  Delta++ ``p pi+ = 1``, rho0 ``pi+ pi- = 1`` for the unordered pair) and
  the exact Delta0 values.  The card's pre-registration line lists
  "Delta0 (I3 = 0) -> p pi- : n pi0 = 1/2 : 1/2"; the Delta0 sits at
  I3 = -1/2 of the I = 3/2 multiplet, and the exact Clebsch-Gordan
  coefficients give ``p pi- : n pi0 = 1/3 : 2/3`` (independent
  lowering-operator computation; the module's docstring pre-registered
  check list lists Delta+ / Delta++ / rho0 only).  This test pins the
  exact physics and documents the card-line discrepancy.
- ``test_clebsch_gordan_zero_cases``: the zero cases of
  ``clebsch_gordan_squared`` (I3 non-conserving, I3 out of range, the
  forbidden couplings 1 (x) 1 with no I = 3/2 or I = 1/2 component, the
  triangle violation 1 (x) 0 with no I = 2) and the component closure
  rule (the I = 1/2 and I = 3/2 components of N-pi sum to 1 at each
  fixed (M, m_b, m_c)).
- ``test_derivation_tabulated_wins``: convention 1, tabulated-wins:
  per-charge tabulated branching fractions are used as-is (a CG-
  deviating 0.6/0.4 split and the uniform 0.5/0.5 split both return
  unchanged); the audit never repairs a tabulated class: it carries no
  ``weight_sum_errors`` or ``forbidden_channels`` and flags the
  CG-deviating shares as "the tabulated charge share" warnings.
- ``test_derivation_class_sum_reproduces_br``: convention 1, derived-
  completes with a representative carrying BR = 0.5 (not 1.0): the
  sibling weights are the exact Clebsch-Gordan shares (2/3 and 1/3 of
  the class branching fraction) and the class sum reproduces the
  multiplet branching fraction to ``WEIGHT_SUM_TOL`` (1e-6), in the
  derived catalog and after ``apply_isospin_derivation``.
- ``test_closure_names_missing_channel_partners``: the closure audit's
  channel-level completeness: the pre-derivation hand-built Delta
  catalog names the missing charge-partner channels ((2112, 211) of
  Delta+, (2112, 111) of Delta0) and the completed catalog is clean; a
  single-record Delta+ with only the p pi0 channel names the missing
  (2112, 211) channel and the record-level census names the partner
  pdg 2114.
- ``test_anti_mirror_reflect_sweep``: convention 4 over the whole
  hand-built Delta family: every parent's weights equal its
  ``isospin.reflect_pdg`` partner's weights with the daughters mapped
  through ``reflect_pdg`` to 1e-9 (matter and anti matter), and the
  charge-conjugate mirror of Delta+ is pinned per charge on aDelta+
  (p -> pbar, n -> nbar, pi+ -> pi-, pi0 -> pi0).
- ``test_apply_processes_all_plan_entries_per_parent``: the apply step
  processes every plan entry of a parent (a parent may carry several
  incomplete multiplet classes, one entry each): a Delta+ with the
  N-pi representative (2212, 111) and the Delta-pi representative
  (2214, 111) is completed in both classes to the exact Clebsch-Gordan
  shares, with the threshold following the highest new channel.

Hermetic: hand-constructed records only; no event generation, no
external tables.  The fixtures are the committed file's hand-built
Delta card (imported, not duplicated).
"""

from __future__ import annotations

import math

from na49_mixing.hadron_catalog import (  # noqa: E402
    HadronCatalog,
)
from na49_mixing.isospin import reflect_pdg  # noqa: E402
from na49_mixing.resonance_decay import (  # noqa: E402
    MIRROR_MATCH_TOL,
    WEIGHT_SUM_TOL,
    apply_isospin_derivation,
    clebsch_gordan_squared,
    derive_charge_weights,
    isospin_closure_report,
    isospin_derivation_plan,
)

# The committed card-5 fixtures (the hand-built Delta(1232) card).
from tests.test_resonance_isospin import (  # noqa: E402
    M_DELTA,
    MPI,
    _channel,
    _delta_catalog,
    _delta_catalog_complete,
    _record,
)


# ----------------------------------------------------------------------
# 1. Clebsch-Gordan correctness: the pre-registered values
# ----------------------------------------------------------------------
def test_clebsch_gordan_pre_registered_values():
    """The card's pre-registered values and the exact Delta0 values.

    Two-units I3 convention of the catalog (isospin3 = 2 * I3):
    Delta+ at I3 = +1, Delta0 at I3 = -1 (I3 = -1/2), Delta++ at
    I3 = +3; rho0 at I3 = 0."""
    # Delta+ (I3 = +1/2) -> p pi0 : n pi+ = 2/3 : 1/3 (pre-registered)
    assert abs(clebsch_gordan_squared(1.5, 1, 0.5, 1, 1.0, 0)
               - 2.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(1.5, 1, 0.5, -1, 1.0, 2)
               - 1.0 / 3.0) <= 1e-12

    # Delta0: the card line reads "Delta0 (I3 = 0) -> p pi- : n pi0 =
    # 1/2 : 1/2".  The Delta0 is the I3 = -1/2 member of the I = 3/2
    # multiplet (I3 = 0 does not occur for I = 3/2), and the exact
    # coefficients at I3 = -1/2 are p pi- : n pi0 = 1/3 : 2/3
    # (|3/2, -1/2> = sqrt(2/3) |n pi0> + sqrt(1/3) |p pi->, the
    # lowering-operator computation the card-5 module's closed form
    # implements).  The committed module returns exactly these values;
    # the card-line discrepancy is reported to the orchestrator.
    assert abs(clebsch_gordan_squared(1.5, -1, 0.5, 1, 1.0, -2)
               - 1.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(1.5, -1, 0.5, -1, 1.0, 0)
               - 2.0 / 3.0) <= 1e-12

    # Delta++ (I3 = +3/2) -> p pi+ = 1 (pre-registered): the only
    # I3-conserving combination.
    assert abs(clebsch_gordan_squared(1.5, 3, 0.5, 1, 1.0, 2)
               - 1.0) <= 1e-12

    # rho0 (I = 1, I3 = 0) -> pi+ pi- (pre-registered "= 1" for the
    # unordered pair): the raw single-ordering coefficient is 1/2
    # (<1, +1; 1, -1 | 1, 0> = 1/sqrt(2)); the module enumerates the
    # identical-multiplet class over unordered pairs and sums both
    # orderings, which is the pre-registered weight 1.
    assert abs(clebsch_gordan_squared(1.0, 0, 1.0, 2, 1.0, -2)
               - 0.5) <= 1e-12
    assert abs(clebsch_gordan_squared(1.0, 0, 1.0, 2, 1.0, -2)
               + clebsch_gordan_squared(1.0, 0, 1.0, -2, 1.0, 2)
               - 1.0) <= 1e-12
    # the pi0 pi0 partner (identical daughters, I3 = 0): coefficient 0
    assert clebsch_gordan_squared(1.0, 0, 1.0, 0, 1.0, 0) == 0.0


def test_clebsch_gordan_zero_cases():
    """The zero cases: I3 non-conserving, I3 out of range, the
    forbidden couplings, the triangle violation, and the component
    closure rule of the N-pi product 1/2 (x) 1 = 1/2 (oplus) 3/2."""
    # I3 non-conserving (I3_b + I3_c != I3_parent): zero
    assert clebsch_gordan_squared(1.5, 1, 0.5, 1, 1.0, 2) == 0.0 \
        # p pi+ carries I3 = +3/2, not the Delta+ I3 = +1/2
    assert clebsch_gordan_squared(1.0, 0, 1.0, 0, 1.0, 2) == 0.0 \
        # pi0 pi+ carries I3 = +1, not the rho0 I3 = 0

    # I3 out of range of the multiplet: zero
    assert clebsch_gordan_squared(1.5, 5, 0.5, 1, 1.0, 4) == 0.0 \
        # |I3| = +5/2 exceeds I = 3/2
    assert clebsch_gordan_squared(1.0, 0, 1.0, 3, 1.0, -3) == 0.0 \
        # a pion carries |I3| = at most 1, not 3/2

    # Forbidden couplings (I3-conserving, the triangle holds for the
    # first two only in the integer sense): 1 (x) 1 = 0 (oplus) 1
    # (oplus) 2 contains no I = 3/2 or I = 1/2 component; the
    # triangle 1 (x) 0 = 1 contains no I = 2.
    assert clebsch_gordan_squared(1.5, 0, 1.0, 1, 1.0, -1) == 0.0
    assert clebsch_gordan_squared(0.5, 1, 1.0, 1, 1.0, 0) == 0.0
    assert clebsch_gordan_squared(2.0, 2, 1.0, 2, 0.0, 0) == 0.0

    # Component closure: at fixed M = +1/2, m_b = +1/2 (p), m_c = 0
    # (pi0), the I = 1/2 and I = 3/2 components carry 1/3 and 2/3 and
    # sum to 1; at fixed m_b = -1/2 (n), m_c = +1 (pi+) they carry
    # 2/3 and 1/3.
    assert abs(clebsch_gordan_squared(0.5, 1, 0.5, 1, 1.0, 0)
               - 1.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(1.5, 1, 0.5, 1, 1.0, 0)
               - 2.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(0.5, 1, 0.5, 1, 1.0, 0)
               + clebsch_gordan_squared(1.5, 1, 0.5, 1, 1.0, 0)
               - 1.0) <= 1e-12
    assert abs(clebsch_gordan_squared(0.5, 1, 0.5, -1, 1.0, 2)
               - 2.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(1.5, 1, 0.5, -1, 1.0, 2)
               - 1.0 / 3.0) <= 1e-12
    assert abs(clebsch_gordan_squared(0.5, 1, 0.5, -1, 1.0, 2)
               + clebsch_gordan_squared(1.5, 1, 0.5, -1, 1.0, 2)
               - 1.0) <= 1e-12


# ----------------------------------------------------------------------
# 2. derivation: the two halves of convention 1
# ----------------------------------------------------------------------
def test_derivation_tabulated_wins():
    """Per-charge tabulated branching fractions are used as-is, and a
    CG-deviating tabulated split is flagged (warning), never repaired."""
    # a CG-deviating 0.6/0.4 split: the weights come back unchanged
    dplus = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 0.6),
         _channel(2214, (2112, 211), 0.4)], M_DELTA)
    catalog = HadronCatalog([dplus])
    weights = derive_charge_weights(dplus, (2212, 211), catalog)
    assert set(weights) == {(2212, 111), (2112, 211)}
    assert weights[(2212, 111)] == 0.6
    assert weights[(2112, 211)] == 0.4
    # the class sum is the sum of the tabulated per-charge values
    assert math.fsum(weights.values()) == 1.0

    # the audit never repairs a tabulated class: no weight-sum error,
    # no forbidden channel, and the CG-deviating shares are flagged as
    # "the tabulated charge share" warnings (one per channel)
    report = isospin_closure_report(catalog)
    parent = next(p for p in report.per_parent if p.pdg_id == 2214)
    assert parent.weight_sum_errors == ()
    assert parent.forbidden_channels == ()
    flagged = [message for message in report.warnings
               if "the tabulated charge share" in message]
    assert len(flagged) == 2
    assert any("(2212, 111)" in message for message in flagged)
    assert any("(2112, 211)" in message for message in flagged)

    # the uniform 0.5/0.5 split ("equal weights because the charge
    # states exist") is used as-is as well and flagged, not re-derived
    uniform = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 0.5),
         _channel(2214, (2112, 211), 0.5)], M_DELTA)
    uniform_catalog = HadronCatalog([uniform])
    uniform_weights = derive_charge_weights(uniform, (2212, 211),
                                            uniform_catalog)
    assert uniform_weights[(2212, 111)] == 0.5
    assert uniform_weights[(2112, 211)] == 0.5
    uniform_report = isospin_closure_report(uniform_catalog)
    assert any("the tabulated charge share" in message
               for message in uniform_report.warnings)


def test_derivation_class_sum_reproduces_br():
    """A representative carrying BR = 0.5 is completed to the exact
    Clebsch-Gordan shares; the class sum reproduces the multiplet
    branching fraction to WEIGHT_SUM_TOL.  The representative is the
    I3-conserving p pi0 channel (the card-2 representative of the
    Delta+); the p pi+ combination does not conserve I3 for the
    Delta+ and is not a representative of its class."""
    dplus = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 0.5)], M_DELTA)
    catalog = HadronCatalog([dplus])
    weights = derive_charge_weights(dplus, (2212, 211), catalog)
    assert set(weights) == {(2212, 111), (2112, 211)}
    assert abs(weights[(2212, 111)] - 2.0 / 3.0 * 0.5) <= 1e-12
    assert abs(weights[(2112, 211)] - 1.0 / 3.0 * 0.5) <= 1e-12
    assert abs(math.fsum(weights.values()) - 0.5) <= WEIGHT_SUM_TOL

    # the completed catalog preserves the multiplet branching fraction
    completed = apply_isospin_derivation(catalog)
    channels = completed.get(2214).decay_channels
    assert abs(math.fsum(c.branching_fraction for c in channels)
               - 0.5) <= 1e-12


# ----------------------------------------------------------------------
# 3. closure audit: the missing charge-partner channels are named
# ----------------------------------------------------------------------
def test_closure_names_missing_channel_partners():
    """The pre-derivation representative-only Delta card names the
    missing charge-partner channels (channel-level completeness); the
    completed card is clean; a single-record Delta+ names both the
    missing channel and the missing record partner."""
    raw = _delta_catalog()
    report = isospin_closure_report(raw)
    # the full four-member multiplet passes the record-level census
    assert all(parent.missing_charge_partners == ()
               for parent in report.per_parent)
    # the channel-level gaps are named: Delta+ is missing the n pi+
    # channel, Delta0 the n pi0 channel (the 2/3 Clebsch-Gordan share)
    assert len(report.errors) == 2
    assert any("2214" in error and "(2112, 211)" in error
               for error in report.errors)
    assert any("2114" in error and "(2112, 111)" in error
               for error in report.errors)

    # the derivation plan clears the gaps: the completed card is clean
    completed_report = isospin_closure_report(apply_isospin_derivation(raw))
    assert completed_report.errors == ()

    # a single-record Delta+ with only the p pi0 channel: the
    # channel-level gap names (2112, 211), and the record-level census
    # names the missing partner pdg 2114 (I3 = -1/2)
    single = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 1.0)], M_DELTA)
    single_report = isospin_closure_report(HadronCatalog([single]))
    parent = single_report.per_parent[0]
    assert parent.pdg_id == 2214
    assert parent.missing_charge_partners == (2114,)
    assert any("(2112, 211)" in error for error in single_report.errors)
    assert any("2114" in error and "I3 = -0.5" in error
               for error in single_report.errors)


# ----------------------------------------------------------------------
# 4. anti-particle mirror: the reflect_pdg sweep over the Delta family
# ----------------------------------------------------------------------
def test_anti_mirror_reflect_sweep():
    """Every Delta parent's weights equal its ``reflect_pdg``
    partner's with the daughters mapped through ``reflect_pdg`` to
    1e-9 (matter and anti matter); the charge-conjugate mirror of
    Delta+ is pinned per charge on aDelta+."""
    catalog = _delta_catalog_complete(with_anti=True)
    for pdg in (2224, 2214, 2114, 1114, -2224, -2214, -2114, -1114):
        parent = catalog.get(pdg)
        channel = parent.decay_channels[0]
        pair = (int(channel.daughters[0]), int(channel.daughters[1]))
        mine = derive_charge_weights(parent, pair, catalog)
        partner_id = reflect_pdg(pdg)
        partner = catalog.get(partner_id)
        pchannel = partner.decay_channels[0]
        ppair = (int(pchannel.daughters[0]), int(pchannel.daughters[1]))
        theirs = derive_charge_weights(partner, ppair, catalog)
        mirrored = {(reflect_pdg(a), reflect_pdg(b)): weight
                    for (a, b), weight in mine.items()}
        assert set(mirrored) == set(theirs), (pdg, partner_id)
        for key, weight in theirs.items():
            assert abs(mirrored[key] - weight) <= MIRROR_MATCH_TOL, (
                pdg, partner_id, key)

    # the charge-conjugate mirror of Delta+ (p -> pbar, n -> nbar,
    # pi+ -> pi-, pi0 -> pi0): aDelta+ carries the mirrored weights
    delta_plus = derive_charge_weights(catalog.get(2214), (2212, 211),
                                       catalog)
    a_delta_plus = derive_charge_weights(catalog.get(-2214), (-2212, 211),
                                         catalog)
    assert set(a_delta_plus) == {(-2212, 111), (-2112, -211)}
    assert abs(a_delta_plus[(-2212, 111)] - 2.0 / 3.0) <= MIRROR_MATCH_TOL
    assert abs(a_delta_plus[(-2112, -211)] - 1.0 / 3.0) <= MIRROR_MATCH_TOL
    # and the mirror relation itself, per charge
    assert abs(delta_plus[(2212, 111)] - a_delta_plus[(-2212, 111)]) \
        <= MIRROR_MATCH_TOL
    assert abs(delta_plus[(2112, 211)] - a_delta_plus[(-2112, -211)]) \
        <= MIRROR_MATCH_TOL


# ----------------------------------------------------------------------
# 5. apply_isospin_derivation: every plan entry per parent is applied
# ----------------------------------------------------------------------
def test_apply_processes_all_plan_entries_per_parent():
    """A parent with several incomplete multiplet classes (one plan
    entry per class) is fully processed: the Delta+ carries the N-pi
    representative (2212, 111) and the Delta-pi representative
    (2214, 111), and the completed record carries both classes'
    Clebsch-Gordan completions with the threshold following the highest
    new channel."""
    dplus = _record(
        2214, "Delta+", 1, 1, 0, 1.5, 1,
        [_channel(2214, (2212, 111), 0.6),
         _channel(2214, (2214, 111), 0.4)], M_DELTA)
    catalog = HadronCatalog([
        dplus,
        _record(2224, "Delta++", 1, 2, 0, 1.5, 3,
                [_channel(2224, (2212, 211), 1.0)], M_DELTA),
        _record(2114, "Delta0", 1, 0, 0, 1.5, -1,
                [_channel(2114, (2212, -211), 1.0)], M_DELTA),
        _record(1114, "Delta-", 1, -1, 0, 1.5, -3,
                [_channel(1114, (2112, -211), 1.0)], M_DELTA),
    ])

    # the regression condition: two plan entries for the Delta+, one
    # per incomplete class
    entries = [entry for entry in isospin_derivation_plan(catalog)
               if entry.pdg_id == 2214]
    assert sorted(entry.multiplet for entry in entries) == \
        [(2212, 111), (2214, 111)]

    completed = apply_isospin_derivation(catalog)
    weights = {
        tuple(sorted(int(d) for d in ch.daughters)):
            float(ch.branching_fraction)
        for ch in completed.get(2214).decay_channels
    }
    assert set(weights) == {
        (111, 2212), (211, 2112),          # the N-pi class
        (111, 2214), (211, 2114), (-211, 2224),  # the Delta-pi class
    }
    # the N-pi class: the exact Clebsch-Gordan shares of BR = 0.6
    # (p pi0 : n pi+ = 2/3 : 1/3)
    assert abs(weights[(111, 2212)] - 2.0 / 3.0 * 0.6) <= 1e-12
    assert abs(weights[(211, 2112)] - 1.0 / 3.0 * 0.6) <= 1e-12
    # the Delta-pi class (3/2 (x) 1 contains I = 3/2): the exact
    # Clebsch-Gordan shares of BR = 0.4 — (Delta0, pi+) = 8/15,
    # (Delta+, pi0) = 1/15, (Delta++, pi-) = 6/15 (independent
    # lowering-operator computation)
    assert abs(weights[(211, 2114)] - 0.4 * 8.0 / 15.0) <= 1e-12
    assert abs(weights[(111, 2214)] - 0.4 * 1.0 / 15.0) <= 1e-12
    assert abs(weights[(-211, 2224)] - 0.4 * 6.0 / 15.0) <= 1e-12
    # each class sum reproduces its multiplet branching fraction
    assert abs(weights[(111, 2212)] + weights[(211, 2112)] - 0.6) <= 1e-12
    assert abs(weights[(211, 2114)] + weights[(111, 2214)]
               + weights[(-211, 2224)] - 0.4) <= 1e-12
    # the threshold follows the highest new channel (Delta pi)
    completed_record = completed.get(2214)
    assert abs(float(completed_record.mass_range[0])
               - (M_DELTA + MPI)) <= 1e-12

    # the single-class parents are unaffected (the Delta0 keeps its
    # N-pi completion through the multi-entry path)
    delta0 = completed.get(2114)
    delta0_weights = {
        tuple(sorted(int(d) for d in ch.daughters)):
            float(ch.branching_fraction)
        for ch in delta0.decay_channels
    }
    assert abs(delta0_weights[(111, 2112)] - 2.0 / 3.0) <= 1e-12
    assert abs(delta0_weights[(-211, 2212)] - 1.0 / 3.0) <= 1e-12
