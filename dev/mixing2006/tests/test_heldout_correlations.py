"""Held-out source-local correlation helper tests (M9, TASK-20260905-physics-review).

Task TASK-20260905-physics-review, milestone M9, pinning the committed
behavior of the vendored mirror
``helpers/benchmarking/staged_heldout_correlations.py`` at 0f7fede (byte-
identical to the monorepo helper at 9467d76, sha256
e50c7afb9ddd8d864aca7e1aad811b57be0bc63b382aaba024f101f70539e722):

- The association machinery (pre-registered O1) on synthetic held-out
  events with known ``source_id`` links: every forward K+ sharing a source
  with a strange baryon gives F1 = 1, none gives F1 = 0, half gives
  F1 = 1/2, with the numerator/denominator counts and the per-K+ co-source
  strange-baryon counts pinned exactly. The forward cut (xF = 2 pz /
  sqrt(s) >= 0.2) is pinned on the denominator (a backward K+ that shares
  its source with a strange baryon does not count), and the association is
  scoped per event (identical ``source_id`` values in different events do
  not co-associate).
- The shuffle-null closed forms: the baseline is a within-event uniform
  permutation of the ``source_id`` multiset (without replacement). The
  pre-registration worked example (multiset {0, 0, 1}) and its
  relabeling (multiset {0, 1, 1}) both carry the co-assignment probability
  (2/3)(1/2) = 1/3, and the p-pbar pair event [p, p, pbar] with multiset
  {0, 0, 1} carries the pair-existence probability 2/3; the empirical
  rates from seeded runs are pinned to the closed forms within the
  pre-registered 5-sigma tolerance (see T_NULL / NULL_SIGMA below). The
  structural invariants are pinned exactly: the per-source
  particle-multiplicity multiset, the per-particle species and momenta,
  and the source records (masses, roles) are preserved by every
  permutation, the index selections are invariant, and
  ``with_shuffled_sources`` consumes exactly one ``rng.permutation(n)``
  draw per call (the exact RNG stream is pinned).
- The Wilson score interval: the (n + z^2) form is cross-checked against
  the independent two-line form at the run-record counts (5061, 7053) to
  1e-12, the endpoints at k = n and k = 0 are pinned exactly (1.0 and
  0.0), the pre-commit regression is pinned (the buggy half-width
  z*sqrt(p(1-p) + z^2/(4n))/(n + z^2) is exactly sqrt(n) times smaller
  than the corrected half-width, REPORT.md deviation note), the
  run-record interval and the run-record Wald standard error are pinned
  (hermetic constants; the run record is their source), and the n = 0
  degeneracy returns NaN.
- The verdict machinery against the pre-registered bands: the boundary
  values of ``verdict_o1`` (D1 = 0.20 with Z = 3.0 passes; D1 = 0.20 with
  Z < 3.0 is inconclusive; D1 = 0.19 is inconclusive; D1 = 0.05 fails,
  the fail side being the inclusive ``max_D1: 0.05`` while the
  inconclusive wording "0.05 < D1 < 0.20" is open at 0.05), the
  ``verdict_traceability`` boundaries (N_pbar = 30 passes, 29 is
  inconclusive, one violation fails at any N_pbar), the
  ``verdict_signed_shift`` boundaries (Z = 3.0 inclusive, zero effect is
  inconclusive), and the O2 conjunction rule (all pass passes; one fail
  fails; one inconclusive with the rest pass is inconclusive, per the
  O2_verdict_rule string).
- The pre-registration cross-check: it passes on the run-record values
  and refuses (SystemExit) a mismatched N, a mismatched seed, a
  mismatched band value, a mismatched calibration-set parameter, and the
  analogous mismatches on the pair threshold, the shuffle realizations,
  and the strange-baryon list.
- The estimator edge cases: an event with no forward K+, an event with no
  pbar, sources carrying a single particle (no co-assignment is possible
  under the without-replacement permutation; pinned exactly, not
  statistically), and the pair-threshold boundary (a parent source mass
  exactly at 1.876544 GeV is NOT a violation: the band defines violations
  as mass < threshold, strictly).

Call-path choices (recorded per the card): the pure functions operate on
held-out event records, so all machinery tests build the records directly
through ``heldout_event_from_arrays`` (the hermetic construction path);
no production events are generated, no NA49 tables are read, and no
network is touched. The run-record numbers (5061/7053, the Wilson
interval, the Wald standard error, the seeds, the bands) are pinned as
hermetic constants; their source is
``dev/00runs/staged_heldout_2026-09-07/`` (pre_registration.json and
heldout_correlations_report.json), which the tests do not read.

Pre-registered tolerances (one rationale each):
- closed-form float agreement (Wilson two-line form, Wald standard
  error, p-pbar invariant mass): absolute 1e-12; the expressions are
  algebraically identical and the difference is pure double rounding.
- exact integer / equality pins (counts, fractions 0, 1/2, 1, endpoints
  0.0 and 1.0, threshold boundary, RNG stream): no tolerance; the math
  is exact in double precision.
- empirical null rates: seeded runs of T_NULL = 20000 realizations
  against a 5-sigma band NULL_SIGMA * sqrt(p(1-p)/T_NULL); the
  per-realization indicator is Bernoulli, the 5-sigma band carries a
  spurious-failure chance of about 2.9e-7 under the correct null, and
  T_NULL = 20000 keeps the three statistical pins below 0.5 s of wall
  time.
- Wilson regression ratio (corrected half-width / buggy half-width):
  relative 1e-12 against exactly sqrt(n); the algebra
  p(1-p)n + z^2/4 = n (p(1-p) + z^2/(4n)) makes the ratio sqrt(n) up to
  rounding.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from helpers.benchmarking import staged_heldout_correlations as H

REPO_ROOT = Path(__file__).resolve().parents[1]  # dev/mixing2006

SQ = H.SQRT_S_GEV

# Dedicated hermetic test seed block: clear of the production M9 block
# (20310831, 20310832..20310847, 20312831) and of the next milestone
# offset (20320831), so the test stream never overlaps a registered run.
TEST_SEED_BASE = 20330001

T_NULL = 20000  # seeded null realizations per closed form (see docstring)
NULL_SIGMA = 5.0  # pre-registered 5-sigma band on the empirical null rate


def _ev(source_masses, roles, pdg, src, mom):
    """Build a HeldoutEvent through the hermetic construction path."""
    return H.heldout_event_from_arrays(
        source_masses, roles, pdg, src, mom, sqrt_s=SQ
    )


# Synthetic events with known source links (card acceptance criterion 3).
# Forward K+: pdg 321 with pz = 4.0 -> xF = 8/17.27 = 0.463 >= 0.2.
# Non-forward K+: pz = 1.0 -> xF = 0.116 < 0.2.
EV_ALL = _ev(  # both forward K+ share their source with a strange baryon
    [3.0, 2.2, 2.0],
    ("projectile", "target", "projectile"),
    [321, 3122, 321, 3112, 211],
    [0, 0, 1, 1, 2],
    [[1.2, 0.1, 0.0, 4.0], [1.2, 0.0, 0.0, 0.5], [1.2, 0.0, 0.1, 4.0],
     [1.3, 0.0, 0.0, -0.5], [0.3, 0.0, 0.0, 0.2]],
)
EV_NONE = _ev(  # forward K+ never share a source with a strange baryon
    [3.0, 2.2, 2.0, 2.4],
    ("projectile", "target", "projectile", "target"),
    [321, 321, 3122, 3212],
    [0, 1, 2, 3],
    [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.1, 4.0],
     [1.2, 0.0, 0.0, 0.3], [1.3, 0.0, 0.0, -0.3]],
)
EV_HALF = _ev(  # one of two forward K+ is associated; the third, backward,
    # K+ shares source 0 with the Lambda but must not enter the denominator
    [3.0, 2.2, 2.0],
    ("projectile", "target", "projectile"),
    [321, 3122, 321, 321],
    [0, 0, 1, 0],
    [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.0, 0.5], [1.2, 0.0, 0.1, 4.0],
     [0.6, 0.0, 0.0, 1.0]],
)
EV_ISOLATION_A = _ev([2.9, 2.1], ("projectile", "target"),
                     [3122, 211], [0, 1],
                     [[1.2, 0.0, 0.0, 0.4], [0.3, 0.0, 0.0, -0.4]])
EV_ISOLATION_B = _ev([3.0, 2.2], ("projectile", "target"),
                     [321, 211], [0, 1],
                     [[1.2, 0.0, 0.0, 4.0], [0.3, 0.0, 0.0, -0.3]])
EV_DOUBLE = _ev(  # two strange baryons co-source with the single forward K+
    [3.0, 2.1],
    ("projectile", "target"),
    [321, 3122, 3212, 211],
    [0, 0, 0, 1],
    [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.0, 0.4], [1.3, 0.0, 0.0, -0.4],
     [0.3, 0.0, 0.0, 0.1]],
)

# p-pbar events for the O2 machinery.
EV_PBAR_OK = _ev(  # pbar parent source 2.1 GeV, above the threshold
    [3.0, 2.5, 2.1],
    ("projectile", "target", "projectile"),
    [321, 3122, -2212],
    [0, 1, 2],
    [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.0, -4.0], [1.1, 0.0, 0.0, 0.2]],
)
EV_PBAR_BELOW = _ev(  # pbar parent source 1.5 GeV, below the threshold
    [1.5, 3.0],
    ("projectile", "target"),
    [-2212, 211],
    [0, 1],
    [[1.0, 0.0, 0.0, 0.2], [0.3, 0.0, 0.0, -0.2]],
)
EV_NO_PBAR = _ev([2.5, 2.0], ("projectile", "target"),
                 [2212, 211], [0, 1],
                 [[1.1, 0.0, 0.0, 0.3], [0.3, 0.0, 0.0, -0.3]])
EV_PAIR_2 = _ev(  # p, p, pbar all in one source -> two same-source pairs
    [3.2],
    ("projectile",),
    [2212, 2212, -2212],
    [0, 0, 0],
    [[1.2, 0.3, 0.0, 0.0], [1.2, -0.3, 0.0, 0.0], [1.1, 0.0, 0.5, 0.0]],
)
EV_PAIR_0 = _ev([2.5, 2.4], ("projectile", "target"),
                [2212, -2212], [0, 1],
                [[1.2, 0.0, 0.0, 0.4], [1.1, 0.0, 0.0, -0.4]])
EV_PAIR_1 = _ev(  # only the second proton shares the pbar's source
    [2.5, 2.4],
    ("projectile", "target"),
    [2212, 2212, -2212],
    [0, 1, 1],
    [[1.2, 0.0, 0.0, 0.4], [1.2, 0.0, 0.0, -0.2], [1.1, 0.0, 0.5, 0.1]],
)


# --- Layout: the vendored mirror is the module under test --------------------


def test_vendored_mirror_is_the_module_under_test():
    # The nested suite runs from dev/mixing2006; the helper must resolve to
    # the vendored copy (byte-identical mirror), not the monorepo tree, and
    # the tree detection must land on this tree (RUNS_ROOT = 00runs).
    assert H.__file__.endswith(
        "dev/mixing2006/helpers/benchmarking/staged_heldout_correlations.py"
    )
    assert H.MIXING2006_DIR == REPO_ROOT
    assert H.RUNS_ROOT == REPO_ROOT / "00runs"


# --- Pre-registered protocol constants ---------------------------------------


def test_pre_registered_protocol_constants():
    # Seeds, cut, species, threshold, and the sqrt(s) of
    # Reaction(158.0, 2212, 2212): the committed record (pre-registration
    # and run artifact at dev/00runs/staged_heldout_2026-09-07).
    assert H.CARD == "TASK-20260905-physics-review"
    assert H.MILESTONE == "M9" and H.PHASE == "A"
    assert H.RUN_DESCRIPTION == "staged_heldout_2026-09-07"
    assert H.STAGE_LABEL == "heldout_correlations"
    assert H.PRODUCTION_MODE == "regge"
    assert H.BEAM_MOMENTUM_GEV_C == 158.0
    assert H.CONFIG_SEED == 20260831
    assert H.PRODUCTION_SEED == 20310831
    assert H.SHUFFLE_SEED_BASE == 20310832
    assert H.SHUFFLE_REALIZATIONS == 16
    assert H.BOOTSTRAP_SEED == 20312831
    assert H.BOOTSTRAP_REPLICATES == 200
    assert H.PILOT_SEED == 20309831
    assert H.FORWARD_XF_MIN == 0.2
    assert H.K_PLUS_PDG == 321
    assert H.PROTON_PDG == 2212 and H.ANTIPROTON_PDG == -2212
    assert H.STRANGE_BARYON_IDS == (3122, 3212, 3112)
    assert H.PAIR_THRESHOLD_GEV == 1.876544
    assert H.WILSON_Z == 1.959963984540054
    assert H.SQRT_S_GEV == 17.270205201372406
    assert H.ARTIFACT_SCHEMA_VERSION == 1


def test_pre_registered_bands_literal():
    # The falsifier bands are pre-registered before the events exist; any
    # drift changes the pass/fail record.
    assert H.PRE_REGISTERED_BANDS == {
        "O1_klambda_association": {
            "statistic": "D1 = F1 - F1_shuffle_mean",
            "pass": {"min_D1": 0.2, "min_Z": 3.0},
            "fail": {"max_D1": 0.05},
            "inconclusive": "0.05 < D1 < 0.20, or D1 >= 0.20 with Z1 < 3.0",
        },
        "O2a_traceability": {
            "statistic": "F_tr = 1 - n_below / N_pbar (violations = pbars with parent-source mass < 1.876544 GeV)",
            "pass": {"max_violations": 0, "min_n_pbar": 30},
            "fail": {"min_violations": 1},
            "inconclusive": "zero violations but N_pbar < 30",
        },
        "O2a_parent_mass_shift": {
            "statistic": "dM_parent = mean(parent masses) - mean(unconditional source masses)",
            "pass": {"sign": "positive", "min_Z": 3.0},
            "fail": {"sign": "negative", "min_Z": 3.0},
            "inconclusive": "Z < 3.0 in either direction",
        },
        "O2a_event_conditioned_mass_shift": {
            "statistic": "dM_event = mean(source masses of pbar-carrying events) - mean(unconditional source masses)",
            "pass": {"sign": "positive", "min_Z": 3.0},
            "fail": {"sign": "negative", "min_Z": 3.0},
            "inconclusive": "Z < 3.0 in either direction",
        },
        "O2b_pair_structure": {
            "statistic": "D2 = mean(shuffled same-source p-pbar invariant masses) - mean(real same-source p-pbar invariant masses)",
            "pass": {"sign": "positive", "min_Z": 3.0},
            "fail": {"sign": "negative", "min_Z": 3.0},
            "inconclusive": "Z < 3.0 in either direction",
        },
        "O2_verdict_rule": (
            "O2 passes only if all four components (traceability, parent-source mass "
            "shift, event-conditioned mass shift, pair structure) pass; any failing "
            "component makes O2 fail; otherwise O2 is inconclusive"
        ),
    }


def test_m8_parameter_set_literal():
    # The calibration-set decision of record (M8 combined-error parameter
    # set); the helper applies these ten values in generate_events.
    assert H.M8_PARAMETER_SET == {
        "regge.alpha_exc0": 1.1418864566363858,
        "regge.effective_transfer_slope": 8.504140762083612,
        "regge.edge_power": 2.3769493784466396,
        "continuum.match_mass": 2.680194110727528,
        "continuum.pion_multiplicity_slope": 1.5226475471033067,
        "continuum.strange_lambda_strength": 0.30184606498191385,
        "continuum.strange_sigma_strength": 0.3777282734799746,
        "continuum.strange_pair_strength": 0.19187855210738686,
        "continuum.pair_strength": 0.08411858717003319,
        "continuum.pair_threshold_power": 1.7610429093564994,
    }


# --- O1 association machinery on synthetic sources ---------------------------


def test_association_all_forward_kplus_associated():
    # Every forward K+ shares a source with a strange baryon -> F1 = 1,
    # pinned exactly (2/2).
    n_num, n_den, co_counts = H.association_fraction([EV_ALL], sqrt_s=SQ)
    assert (n_num, n_den) == (2, 2)
    assert co_counts == [1, 1]
    assert n_num / n_den == 1.0


def test_association_no_forward_kplus_associated():
    # No forward K+ shares a source with a strange baryon -> F1 = 0,
    # pinned exactly (0/2); the co-source counts are all zero.
    n_num, n_den, co_counts = H.association_fraction([EV_NONE], sqrt_s=SQ)
    assert (n_num, n_den) == (0, 2)
    assert co_counts == [0, 0]
    assert n_num / n_den == 0.0


def test_association_half_associated_with_forward_cut():
    # One of the two forward K+ is associated -> F1 = 1/2 exactly, with the
    # backward K+ (xF = 0.116 < 0.2) excluded from the denominator despite
    # sharing source 0 with the Lambda: n_den == 2, not 3.
    n_num, n_den, co_counts = H.association_fraction([EV_HALF], sqrt_s=SQ)
    assert (n_num, n_den) == (1, 2)
    assert co_counts == [1, 0]
    assert n_num / n_den == 0.5


def test_association_is_scoped_per_event():
    # EV_ISOLATION_A carries the strange baryon at source_id 0 and
    # EV_ISOLATION_B the forward K+ at source_id 0: the identical source_id
    # values belong to different events and must not co-associate.
    n_num, n_den, co_counts = H.association_fraction(
        [EV_ISOLATION_A, EV_ISOLATION_B], sqrt_s=SQ
    )
    assert (n_num, n_den) == (0, 1)
    assert co_counts == [0]


def test_association_counts_multiple_co_source_baryons_once():
    # Two strange baryons share the forward K+'s source: the per-K+
    # co-source count is 2 but the numerator counts the K+ once.
    n_num, n_den, co_counts = H.association_fraction([EV_DOUBLE], sqrt_s=SQ)
    assert (n_num, n_den) == (1, 1)
    assert co_counts == [2]


def test_reverse_association_and_role_diagnostics():
    # reverse: (3 of 5 strange baryons share a source with a forward K+)
    #   EV_ALL: 2 of 2 (Lambda s0 and Sigma- s1, both carry forward K+)
    #   EV_NONE: 0 of 2   EV_HALF: 1 of 1
    # role: the associated forward K+ of EV_HALF shares source 0, whose
    #   role is "projectile".
    rev_with, rev_total = H.reverse_association_fraction(
        [EV_ALL, EV_NONE, EV_HALF]
    )
    assert (rev_with, rev_total) == (3, 5)
    counts, total = H.association_by_source_role([EV_HALF])
    assert counts == {"projectile": 1} and total == 1


def test_heldout_event_from_arrays_refuses_out_of_range_source_id():
    with pytest.raises(ValueError, match="source_id out of range"):
        H.heldout_event_from_arrays(
            [3.0, 2.0], ("projectile", "target"), [321, 211], [0, 5],
            [[1.2, 0.0, 0.0, 4.0], [0.3, 0.0, 0.0, -0.3]], sqrt_s=SQ,
        )


# --- Shuffle-null closed forms -----------------------------------------------


def test_shuffle_preserves_structure_invariants():
    # 100 seeded realizations: the per-source particle-multiplicity
    # multiset, the species, the momenta, the source records, and the index
    # selections are all invariant under the shuffle (exact pins).
    rng = np.random.default_rng(TEST_SEED_BASE)
    for _ in range(100):
        sh = EV_HALF.with_shuffled_sources(rng)
        assert sorted(sh.src.tolist()) == sorted(EV_HALF.src.tolist())
        assert np.array_equal(sh.pdg, EV_HALF.pdg)
        assert np.array_equal(sh.mom, EV_HALF.mom)
        assert np.array_equal(sh.source_masses, EV_HALF.source_masses)
        assert sh.source_roles == EV_HALF.source_roles
        assert np.array_equal(sh.idx_forward_kplus, EV_HALF.idx_forward_kplus)
        assert np.array_equal(sh.idx_strange_baryon, EV_HALF.idx_strange_baryon)
        assert np.array_equal(sh.idx_pbar, EV_HALF.idx_pbar)
        assert np.array_equal(sh.idx_proton, EV_HALF.idx_proton)


def test_shuffle_consumes_exactly_one_permutation_per_call():
    # The exact RNG stream is pinned: call i draws the i-th
    # rng.permutation(n_particles) of a fresh generator at the test seed.
    # A regression that consumed an extra draw (or a different draw) per
    # call would desynchronize the stream.
    ev = EV_HALF
    rng = np.random.default_rng(TEST_SEED_BASE + 1)
    ref = np.random.default_rng(TEST_SEED_BASE + 1)
    for _ in range(3):
        sh = ev.with_shuffled_sources(rng)
        assert np.array_equal(sh.src, ev.src[ref.permutation(ev.src.size)])


def test_shuffle_null_rate_worked_example_one_third():
    # Pre-registration worked example: [K+ s0, Lambda s0, pion s1],
    # multiset {0, 0, 1}; P(K+ and Lambda co-source) = (2/3)(1/2) = 1/3.
    # T_NULL = 20000 seeded realizations (seed TEST_SEED_BASE + 2) within
    # the pre-registered 5-sigma band.
    ev = _ev([3.0, 2.2], ("projectile", "target"),
             [321, 3122, 211], [0, 0, 1],
             [[1.2, 0.1, 0.0, 4.0], [1.2, 0.0, 0.0, 0.5], [0.2, 0.0, 0.0, -0.5]])
    p = 1.0 / 3.0
    rng = np.random.default_rng(TEST_SEED_BASE + 2)
    n_assoc = 0
    for _ in range(T_NULL):
        sh = ev.with_shuffled_sources(rng)
        a, d, _ = H.association_fraction([sh], sqrt_s=SQ)
        n_assoc += a if d else 0
    p_hat = n_assoc / T_NULL
    se = math.sqrt(p * (1.0 - p) / T_NULL)
    assert p_hat == pytest.approx(p, abs=NULL_SIGMA * se)


def test_shuffle_null_rate_relabelled_multiset_one_third():
    # The relabeling [K+ s0, Lambda s1, pion s1], multiset {0, 1, 1}: the
    # null depends on the multiplicity structure, not the labels; the
    # closed form is again (2/3)(1/2) = 1/3 (T_NULL, seed + 3, 5 sigma).
    ev = _ev([3.0, 2.5], ("projectile", "target"),
             [321, 3122, 211], [0, 1, 1],
             [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.0, -4.0], [0.2, 0.0, 0.0, 0.1]])
    p = 1.0 / 3.0
    rng = np.random.default_rng(TEST_SEED_BASE + 3)
    n_assoc = 0
    for _ in range(T_NULL):
        sh = ev.with_shuffled_sources(rng)
        a, d, _ = H.association_fraction([sh], sqrt_s=SQ)
        n_assoc += a if d else 0
    p_hat = n_assoc / T_NULL
    se = math.sqrt(p * (1.0 - p) / T_NULL)
    assert p_hat == pytest.approx(p, abs=NULL_SIGMA * se)


def test_shuffle_null_pair_rate_two_thirds():
    # [p s0, p s0, pbar s1], multiset {0, 0, 1}: a same-source p-pbar pair
    # exists under the null iff the pbar lands in a source-0 slot,
    # P = 2/3 (T_NULL seeded realizations, seed + 4, 5 sigma).
    ev = _ev([3.2, 2.4], ("projectile", "target"),
             [2212, 2212, -2212], [0, 0, 1],
             [[1.2, 0.0, 0.0, 0.4], [1.2, 0.0, 0.0, -0.4], [1.1, 0.0, 0.0, 0.2]])
    p = 2.0 / 3.0
    rng = np.random.default_rng(TEST_SEED_BASE + 4)
    n_pair_events = 0
    for _ in range(T_NULL):
        sh = ev.with_shuffled_sources(rng)
        arrs = H.same_source_ppbar_invariant_masses([sh])
        n_pair_events += 1 if arrs[0].size else 0
    p_hat = n_pair_events / T_NULL
    se = math.sqrt(p * (1.0 - p) / T_NULL)
    assert p_hat == pytest.approx(p, abs=NULL_SIGMA * se)


# --- Wilson interval pins ------------------------------------------------------


def test_wilson_matches_two_line_form_at_run_record_counts():
    # (i) The (n + z^2) form cross-checked against the independent two-line
    # form at the run-record counts (k, n) = (5061, 7053), to 1e-12.
    k, n = 5061, 7053
    z = H.WILSON_Z
    lo, hi = H.wilson_interval(k, n)
    p = k / n
    center_ref = (p + z * z / (2.0 * n)) / (1.0 + z * z / n)
    half_ref = (
        z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / (1.0 + z * z / n)
    )
    assert lo == pytest.approx(center_ref - half_ref, abs=1e-12)
    assert hi == pytest.approx(center_ref + half_ref, abs=1e-12)


def test_wilson_endpoints_at_k_equals_n_and_k_zero():
    # (ii) k = n -> right endpoint exactly 1.0; (iii) k = 0 -> left endpoint
    # exactly 0.0 (exact in double precision, pinned by equality).
    n = 7053
    lo_n, hi_n = H.wilson_interval(n, n)
    assert hi_n == 1.0 and lo_n < 1.0
    lo_0, hi_0 = H.wilson_interval(0, n)
    assert lo_0 == 0.0 and hi_0 > 0.0


def test_wilson_buggy_half_width_regression_factor_sqrt_n():
    # (iv) Regression pin: the pre-commit buggy half-width
    # z*sqrt(p(1-p) + z^2/(4n))/(n + z^2) is exactly sqrt(n) times smaller
    # than the corrected half-width (the corrected radicand is n times the
    # buggy one: p(1-p)n + z^2/4 = n (p(1-p) + z^2/(4n))). The ratio is
    # pinned at relative 1e-12 (the actual deviation is rounding, ~1e-14).
    k, n = 5061, 7053
    z = H.WILSON_Z
    p = k / n
    half_fixed = z * math.sqrt(p * (1.0 - p) * n + z * z / 4.0) / (n + z * z)
    half_buggy = z * math.sqrt(p * (1.0 - p) + z * z / (4.0 * n)) / (n + z * z)
    assert half_fixed / half_buggy == pytest.approx(math.sqrt(n), rel=1e-12)
    # And the corrected half-width is exactly what wilson_interval uses:
    lo, hi = H.wilson_interval(k, n)
    center = (k + z * z / 2.0) / (n + z * z)
    assert lo == pytest.approx(center - half_fixed, abs=1e-15)
    assert hi == pytest.approx(center + half_fixed, abs=1e-15)


def test_wilson_run_record_interval_and_wald_se():
    # (v) The run record (dev/00runs/staged_heldout_2026-09-07,
    # heldout_correlations_report.json) is the source of these constants;
    # the test itself stays hermetic. The interval is pinned by equality
    # (the arithmetic is deterministic); the Wald standard error to 1e-12.
    k, n = 5061, 7053
    lo, hi = H.wilson_interval(k, n)
    assert lo == 0.7069444430599059
    assert hi == 0.7279526731511867
    f1 = k / n
    assert H.wald_se(f1, n) == pytest.approx(0.005360459386513337, abs=1e-12)
    # The O2 traceability interval at (n_pbar - n_below, n_pbar) =
    # (3249, 3249): the run-record right endpoint is exactly 1.0.
    lo_tr, hi_tr = H.wilson_interval(3249, 3249)
    assert hi_tr == 1.0 and lo_tr == pytest.approx(0.9988190451734813, abs=1e-12)


def test_wilson_degenerate_counts():
    # n = 0: both helpers return NaN (documented degenerate behavior).
    lo, hi = H.wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    assert math.isnan(H.wald_se(0.5, 0))


# --- Verdict machinery against the pre-registered bands ------------------------


def test_verdict_o1_boundaries():
    # Pass: D1 >= 0.20 with Z >= 3.0, both boundaries inclusive.
    assert H.verdict_o1(0.20, 3.0) == "pass"
    assert H.verdict_o1(0.5, 10.0) == "pass"
    # D1 >= 0.20 with Z < 3.0: inconclusive (second clause of the band).
    assert H.verdict_o1(0.25, 2.0) == "inconclusive"
    # 0.05 < D1 < 0.20: inconclusive, however large Z is.
    assert H.verdict_o1(0.19, 100.0) == "inconclusive"
    assert H.verdict_o1(0.10, 50.0) == "inconclusive"
    # Reading recorded: the fail side is the inclusive max_D1 = 0.05
    # (D1 <= 0.05 fails); the inconclusive wording "0.05 < D1 < 0.20" is
    # open at 0.05, so D1 = 0.05 falls on the fail side.
    assert H.verdict_o1(0.05, 100.0) == "fail"
    assert H.verdict_o1(0.04, 100.0) == "fail"
    assert H.verdict_o1(0.0, 0.0) == "fail"
    assert H.verdict_o1(-0.5, 50.0) == "fail"
    # The production run sits in the pass band: D1 = 0.272668, Z1 = 34.719.
    assert H.verdict_o1(0.27266765915213387, 34.7190092821764) == "pass"


def test_verdict_traceability_boundaries():
    # Reading recorded: min_n_pbar = 30 is inclusive (N_pbar = 30 passes);
    # the inconclusive wording "N_pbar < 30" is strict; one violation
    # (min_violations = 1) fails at any N_pbar, the fail side carrying no
    # N condition and taking precedence.
    assert H.verdict_traceability(0, 30) == "pass"
    assert H.verdict_traceability(0, 100) == "pass"
    assert H.verdict_traceability(0, 29) == "inconclusive"
    assert H.verdict_traceability(0, 0) == "inconclusive"
    assert H.verdict_traceability(1, 30) == "fail"
    assert H.verdict_traceability(1, 5) == "fail"
    assert H.verdict_traceability(2, 100) == "fail"


def test_verdict_signed_shift_boundaries():
    # Z = 3.0 is inclusive in both directions; a zero effect is
    # inconclusive regardless of Z; |Z| < 3.0 is inconclusive.
    assert H.verdict_signed_shift(0.5, 3.0) == "pass"
    assert H.verdict_signed_shift(0.5, 2.99) == "inconclusive"
    assert H.verdict_signed_shift(-0.5, 3.0) == "fail"
    assert H.verdict_signed_shift(-0.5, 2.99) == "inconclusive"
    assert H.verdict_signed_shift(0.0, 100.0) == "inconclusive"
    assert H.verdict_signed_shift(0.0, 0.0) == "inconclusive"


def test_verdict_o2_conjunction():
    # O2_verdict_rule: all four components pass -> pass; any failing
    # component -> fail; otherwise (one inconclusive, the rest pass) ->
    # inconclusive.
    comps = lambda trace, parent, event, pair: {  # noqa: E731
        "traceability": trace, "parent_mass_shift": parent,
        "event_conditioned_mass_shift": event, "pair_structure": pair,
    }
    assert H.verdict_o2(comps("pass", "pass", "pass", "pass")) == "pass"
    assert H.verdict_o2(comps("pass", "fail", "pass", "pass")) == "fail"
    assert H.verdict_o2(comps("fail", "pass", "pass", "pass")) == "fail"
    assert H.verdict_o2(comps("pass", "pass", "pass", "fail")) == "fail"
    assert H.verdict_o2(comps("pass", "pass", "fail", "pass")) == "fail"
    assert H.verdict_o2(comps("pass", "pass", "pass", "inconclusive")) == "inconclusive"
    assert H.verdict_o2(comps("inconclusive", "pass", "pass", "pass")) == "inconclusive"
    assert H.verdict_o2(
        comps("inconclusive", "inconclusive", "pass", "pass")
    ) == "inconclusive"
    assert H.verdict_o2(comps("fail", "inconclusive", "pass", "pass")) == "fail"


def test_z_ratio_degenerate_se():
    # The Z machinery: effect / se, with se = 0 (or non-finite) mapping to
    # +/-inf by the sign of the effect and 0.0 at a zero effect.
    assert H._z_ratio(0.5, 0.1) == pytest.approx(5.0, abs=1e-12)
    assert H._z_ratio(0.5, 0.0) == float("inf")
    assert H._z_ratio(-0.5, 0.0) == float("-inf")
    assert H._z_ratio(0.0, 0.0) == 0.0
    assert H._z_ratio(0.5, float("nan")) == float("inf")


# --- Pre-registration cross-check ----------------------------------------------


def _prereg_record() -> dict:
    """The run-record values of the blocks the cross-check consumes."""
    return {
        "calibration_set_decision_of_record": {
            "m8_fitted_parameters": copy.deepcopy(H.M8_PARAMETER_SET),
        },
        "seed_protocol": {
            "m9_production_seed": 20310831,
            "shuffle_seed_base": 20310832,
            "bootstrap_seed": 20312831,
            "stage_label": "heldout_correlations",
        },
        "event_count": {"n_events": 100000},
        "uncertainty_methods": {
            "shuffle_realizations": 16,
            "bootstrap_replicates": 200,
        },
        "cuts": {"forward_xf_min": 0.2},
        "species": {"strange_baryon_pdg_ids": [3122, 3212, 3112]},
        "definitions": {"pair_threshold_gev": 1.876544},
        "pre_registered_bands": copy.deepcopy(H.PRE_REGISTERED_BANDS),
    }


def test_cross_check_preregistration_passes_on_run_record_values():
    # Positive control: the run-record values pass the cross-check
    # (n_events, seed, realizations, replicates as registered).
    H.cross_check_preregistration(
        _prereg_record(), 100000, 20310831, 16, 200
    )


@pytest.mark.parametrize(
    "mutate, kwargs",
    [
        # Mismatched N: the argument disagrees with the registered count.
        (lambda r: None, dict(n_events=99999, seed=20310831,
                              shuffle_realizations=16, bootstrap_replicates=200)),
        # Mismatched seed: the argument disagrees with the registered seed.
        (lambda r: None, dict(n_events=100000, seed=20310832,
                              shuffle_realizations=16, bootstrap_replicates=200)),
        # Mismatched band value: the file's O1 pass band drifts.
        (lambda r: r["pre_registered_bands"]["O1_klambda_association"]["pass"].__setitem__("min_D1", 0.25),
         dict(n_events=100000, seed=20310831,
              shuffle_realizations=16, bootstrap_replicates=200)),
        # Mismatched calibration-set parameter: one M8 value drifts.
        (lambda r: r["calibration_set_decision_of_record"]["m8_fitted_parameters"].__setitem__("regge.alpha_exc0", 1.1418864566363858 + 1e-6),
         dict(n_events=100000, seed=20310831,
              shuffle_realizations=16, bootstrap_replicates=200)),
        # Mismatched pair threshold in the file.
        (lambda r: r["definitions"].__setitem__("pair_threshold_gev", 1.88),
         dict(n_events=100000, seed=20310831,
              shuffle_realizations=16, bootstrap_replicates=200)),
        # Mismatched shuffle realizations (argument).
        (lambda r: None, dict(n_events=100000, seed=20310831,
                              shuffle_realizations=17, bootstrap_replicates=200)),
        # Mismatched strange-baryon list in the file.
        (lambda r: r["species"].__setitem__("strange_baryon_pdg_ids", [3122]),
         dict(n_events=100000, seed=20310831,
              shuffle_realizations=16, bootstrap_replicates=200)),
    ],
    ids=[
        "mismatched_n_events", "mismatched_seed", "mismatched_band_value",
        "mismatched_calibration_parameter", "mismatched_pair_threshold",
        "mismatched_shuffle_realizations", "mismatched_strange_baryon_ids",
    ],
)
def test_cross_check_preregistration_refuses_mismatches(mutate, kwargs):
    # Each negative control must refuse to run with the pre-registration
    # mismatch wording (SystemExit), before any event generation.
    prereg = _prereg_record()
    mutate(prereg)
    with pytest.raises(SystemExit, match="pre-registration mismatch"):
        H.cross_check_preregistration(prereg, **kwargs)


# --- Estimator edge cases -------------------------------------------------------


def test_event_without_forward_kplus_contributes_nothing():
    # An event whose only K+ is backward (xF = 0.116) despite sharing its
    # source with a strange baryon: the denominator stays empty.
    ev = _ev([2.9, 2.1], ("projectile", "target"),
             [321, 3122], [0, 0],
             [[0.6, 0.0, 0.0, 1.0], [1.2, 0.0, 0.0, -0.4]])
    assert H.association_fraction([ev], sqrt_s=SQ) == (0, 0, [])
    # An event with no K+ at all: the same.
    ev2 = _ev([2.9, 2.1], ("projectile", "target"),
              [211, 211], [0, 1],
              [[0.6, 0.0, 0.0, 0.1], [0.6, 0.0, 0.0, -0.1]])
    assert H.association_fraction([ev2], sqrt_s=SQ) == (0, 0, [])


def test_event_without_pbar_contributes_nothing_to_traceability():
    # EV_NO_PBAR has no pbar: zero per-event pbar content, an empty parent
    # mass list, no violations, and exclusion from the conditioned sample.
    recs = H.pbar_parent_source_records([EV_NO_PBAR, EV_PBAR_OK])
    assert np.array_equal(recs["n_pbar_e"], [0.0, 1.0])
    assert np.array_equal(recs["has_pbar_e"], [0.0, 1.0])
    assert np.array_equal(recs["sum_parent_e"], [0.0, 2.1])
    assert np.array_equal(recs["n_src_e"], [2.0, 3.0])
    assert np.array_equal(recs["sum_src_e"], [4.5, 7.6])
    assert np.array_equal(recs["parent_masses"], [2.1])
    assert recs["n_below"] == 0
    all_masses = H.all_source_masses([EV_NO_PBAR, EV_PBAR_OK])
    assert np.array_equal(all_masses, [2.5, 2.0, 3.0, 2.5, 2.1])
    cond_masses = H.pbar_event_source_masses([EV_NO_PBAR, EV_PBAR_OK])
    assert np.array_equal(cond_masses, [3.0, 2.5, 2.1])


def test_single_particle_sources_admit_no_co_assignment():
    # Every source carries exactly one particle (multiset {0, 1, 2}):
    # without replacement, a permutation assigns distinct sources to the
    # three particles, so no co-assignment is possible. Pinned exactly
    # (0 in every one of 500 realizations), not statistically.
    ev = _ev([3.0, 2.5, 2.1], ("projectile", "target", "projectile"),
             [321, 3122, -2212], [0, 1, 2],
             [[1.2, 0.0, 0.0, 4.0], [1.2, 0.0, 0.0, -4.0], [1.1, 0.0, 0.0, 0.2]])
    rng = np.random.default_rng(TEST_SEED_BASE + 5)
    for _ in range(500):
        sh = ev.with_shuffled_sources(rng)
        n_num, n_den, _ = H.association_fraction([sh], sqrt_s=SQ)
        assert (n_num, n_den) == (0, 1)
        assert H.same_source_ppbar_invariant_masses([sh])[0].size == 0


def test_pair_threshold_boundary_exact():
    # Boundary reading: the band defines violations as
    # "pbars with parent-source mass < 1.876544 GeV" (strict), and the
    # helper implements pm < pair_threshold_gev. A parent source mass
    # exactly at the threshold (the same double, H.PAIR_THRESHOLD_GEV) is
    # NOT a violation; one 1e-6 GeV below it is.
    ev_at = _ev([H.PAIR_THRESHOLD_GEV, 2.5], ("projectile", "target"),
                [-2212, 211], [0, 1],
                [[1.0, 0.0, 0.0, 0.2], [0.3, 0.0, 0.0, -0.2]])
    ev_below = _ev([H.PAIR_THRESHOLD_GEV - 1e-6, 2.5],
                   ("projectile", "target"),
                   [-2212, 211], [0, 1],
                   [[1.0, 0.0, 0.0, 0.2], [0.3, 0.0, 0.0, -0.2]])
    recs_at = H.pbar_parent_source_records([ev_at])
    assert recs_at["n_below"] == 0
    assert np.array_equal(recs_at["parent_masses"], [H.PAIR_THRESHOLD_GEV])
    recs_below = H.pbar_parent_source_records([ev_below])
    assert recs_below["n_below"] == 1
    # The traceability verdict of the record (0 violations, N_pbar = 3249)
    # and of the boundary event (0 violations, N_pbar = 1) differ only by
    # the minimum-sample clause: pass vs inconclusive.
    assert H.verdict_traceability(recs_at["n_below"], 1) == "inconclusive"
    assert H.verdict_traceability(recs_at["n_below"], 3249) == "pass"


def test_same_source_ppbar_pair_counts_and_masses():
    # EV_PAIR_2: p, p, pbar in one source -> two same-source pairs; the
    # invariant mass of each pair is the four-vector sum closed form
    # sqrt((1.1+1.2)^2 - 0.3^2 - 0.5^2) = sqrt(4.95) to 1e-12.
    pairs = H.same_source_ppbar_invariant_masses([EV_PAIR_2])
    assert pairs[0].size == 2
    m_ref = math.sqrt(4.95)
    for m in pairs[0]:
        assert float(m) == pytest.approx(m_ref, abs=1e-12)
    # EV_PAIR_0: different sources -> no pair; EV_PAIR_1: one shared
    # source -> exactly one pair.
    assert H.same_source_ppbar_invariant_masses([EV_PAIR_0])[0].size == 0
    assert H.same_source_ppbar_invariant_masses([EV_PAIR_1])[0].size == 1
    # An event with a pbar but no proton: an empty array, not an error.
    assert H.same_source_ppbar_invariant_masses([EV_PBAR_BELOW])[0].size == 0


# --- Fused pass against the public estimators ---------------------------------


def test_one_pass_agrees_with_public_estimators():
    # The production run consumes _one_pass; pin it against the public
    # estimators on a mixed synthetic sample (no event generation).
    sample = [EV_ALL, EV_NONE, EV_HALF, EV_DOUBLE, EV_PBAR_OK, EV_PBAR_BELOW,
              EV_NO_PBAR, EV_PAIR_2, EV_PAIR_0, EV_PAIR_1]
    st = H._one_pass(sample, SQ, H.PAIR_THRESHOLD_GEV)
    n_num, n_den, _ = H.association_fraction(sample, sqrt_s=SQ)
    assert (st["n_assoc"], st["n_kfwd"]) == (n_num, n_den)
    recs = H.pbar_parent_source_records(sample)
    assert st["parent_below"] == recs["n_below"]
    assert st["parent_count"] == recs["parent_masses"].size
    assert st["parent_masses"] == recs["parent_masses"].tolist()
    pair_arrays = H.same_source_ppbar_invariant_masses(sample)
    assert st["pair_count"] == sum(a.size for a in pair_arrays)
    assert st["pair_sum"] == pytest.approx(
        sum(float(a.sum()) for a in pair_arrays if a.size), abs=1e-9
    )
    assert np.array_equal(st["per_event"]["n_pbar"], recs["n_pbar_e"])
    assert np.array_equal(st["per_event"]["has_pbar"], recs["has_pbar_e"])
    # Per-forward-K+ co-source strange-baryon counts, in idx_fwd order:
    # EV_ALL [1, 1], EV_NONE [0, 0], EV_HALF [1, 0], EV_DOUBLE [2],
    # EV_PBAR_OK [0].
    assert st["co_counts"] == [1, 1, 0, 0, 1, 0, 2, 0]
    assert (st["n_assoc"], st["n_kfwd"]) == (4, 8)
