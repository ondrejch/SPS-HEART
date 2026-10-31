"""Committed-catalog pins for the R2 provenance-tagged initial resonance
catalog (Round 2, card 2).

Task ``TASK-20260913-r2-provenance-catalog-import``.  Sibling of
``test_resonance_catalog.py`` (which pins the import driver with
hand-constructed mini fixtures): this module pins the committed catalog
file ``config/v2_singlet/resonance_catalog.json`` (nested commit
``fb37c8f``, the plan section 4.4 catalog file, with the physics values
revised in nested commit ``b29544f``; 98 records / 296 channels) against
the card's public interface, the solver's committed
scope decisions, and the card's deterministic physics rules.  The file
is read, never written; the driver is imported for its embedded
transcription (the documented manual step of the card) and its record-set
helpers, and the monorepo mirror is checked for byte identity.

Scope decisions pinned:

1. Eta-prime: the transcription's record at the card-pinned PDG key
   377, name ``eta-prime(958)``, pole 0.95778 GeV, nominal width
   0.000188 GeV, with the two documented PDG channels pi+ pi- eta
   (0.425) and pi0 pi0 eta (0.224); the id correction (PDG MC id 331)
   is recorded. The orphan-record revision of 2026-10-29 removed the
   record from the committed file.
2. ``record_count`` = 98 = 87 + 11 in the transcription: 80 decayable
   resonances plus 7 cascade-terminal ground states plus the
   pre-registered 11-state termination set (card-1 STABLE_STATES); the
   committed file holds 97 = 86 + 11 after the orphan-record revision.
3. The 18 ``stable_for_decay`` records (the 11-state set plus
   K0/aK0/Lambda/aLambda/Sigma+/Sigma0/Sigma-) are terminal: no catalog
   channels, ``mass_range = (pole, pole)``.
4. ``m_up``: the documented ``decays.py`` Breit-Wigner ``dumphigh`` cut
   where available (rho(770) 1.5, Delta(1232) 2.0 including the
   C-symmetric anti partners, K*(892) 1.4; 15 records cited in the
   transcription tags), else ``pole + 5 * nominal_width``, and the
   terminal convention ``m_up = pole`` for the 18 stable records.
5. Orbital angular momentum: the lowest parity-allowed partial wave,
   ``(−1)^L = P_parent / prod(P_daughters)`` within the angular-
   momentum range over the daughter spin couplings; three-body channels
   carry the total L.  Worked example pin: omega(1420) and omega(1650),
   J^P = 1-, decay through rho0 pi0 at L = 1.
6. Provenance: every record and every channel carries a complete
   six-field ``ProvenanceTag``; the driver's transcription carries
   exactly 1080 tags (98 records x 8 + 296 channels) with 0 incomplete,
   and every tag's ``transcribed_value`` equals the catalog value.

Deterministic card checks pinned: per-channel (B, Q, S) conservation
(independently recomputed through ``quantum_numbers``), per-parent
branching sums <= 1 with the residual documented (eta and eta-prime
unrepresented budgets, matching the ``validate()`` warnings), channel
thresholds equal to ``mass_range[0]`` and at least one channel open at
every decayable record's pole mass, charge-conjugate symmetry of every
particle-antiparticle pair present in the catalog (30 pairs), the
committed file equal to the driver's embedded transcription, and the
dual-tree driver mirror byte-identical.

Catalog revision (nested commit ``b29544f``) reconciled the three
documented deviations of the original import (``fb37c8f``), and the
corresponding ``xfail`` pins now pass as normal tests:

- K1(1270) charge states 10323, -10323, 10313, -10313 carry the
  PDG 2024 J^P = 1/2(1+) (parity ``+``, spin 1), so L = 0 on all 20
  channels (rho K, K* pi, omega K: two negative-parity daughters) is
  the lowest parity-allowed wave, matching the non-strange analogs
  a1(1260) and b1(1235) (J^P = 1+, L = 0).
- N(1650) charge states 122112, 122212, -122112: the nine Delta pi
  channels carry L = 2, the lowest parity-allowed feasible wave for
  J = 1/2 from Delta(3/2) + pi(0) (the angular-momentum range is
  L in {1, 2} and the parity rule requires even L); the N pi / N eta
  / Lambda K channels on the same parents stay L = 0.
- N*(1440) records 12112, 12212, -12112: ``m_up`` = 2.94 = 1.44 +
  5 * 0.30 = pole + 5 * nominal_width, the documented pre-registered
  rule, with ``nominal_width`` unchanged at the vendored PDG 2024
  value 0.30 (the original import stored 3.19 = 1.44 + 5 * 0.35,
  whose 0.35 width was documented nowhere in the repository).
- The 18 records whose ``isospin`` field had been transcribed as J
  instead of the PDG isospin (N(1520), N(1680), N(1720), a2(1320),
  f2(1270), rho3(1690), f4(2050)) carry the corrected PDG isospin.
The documented L-rule deviation inventory is now empty: every channel
of every decayable record obeys the documented L convention.

The orphan-record revision of 2026-10-29
(``catalog_orphan_record_revision.py``) removed the eta-prime record at
the card-pinned key 377, which no continuum class and no channel
reaches; the eta' of the fitted kernels is the record 331 of
``KernelConfig.eta_prime``. The pins of the driver's transcription stay
at its 98 records; the pins of the committed file subtract the removed
record.

Hermetic: the committed file, the vendored PDG table, and the driver's
embedded transcription only; no event generation, no external tables,
no run artifacts.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
NESTED_ROOT = TESTS_DIR.parent
CATALOG_PATH = NESTED_ROOT / "config" / "v2_singlet" / "resonance_catalog.json"

sys.path.insert(0, str(NESTED_ROOT))
sys.path.insert(0, str(NESTED_ROOT / "helpers" / "benchmarking"))

import catalog_radiative_revision as radiative_revision  # noqa: E402
import catalog_isospin_share_revision as share_revision  # noqa: E402
import catalog_dipion_isospin_revision as dipion_revision  # noqa: E402
import catalog_orphan_record_revision as orphan_revision  # noqa: E402
import import_resonance_catalog as import_driver  # noqa: E402
from na49_mixing.hadron_catalog import (  # noqa: E402
    STABLE_STATES,
    HadronCatalog,
)
from na49_mixing.pdg_database import get_particle  # noqa: E402
from na49_mixing.quantum_numbers import (  # noqa: E402
    QuantumNumbers,
    quantum_numbers,
)

# Tolerances (GeV for masses, dimensionless for branching fractions).
MASS_TOL = 1e-9
EXACT_TOL = 1e-12

# The pre-registered termination set (round 2 index, item 11) and the
# seven cascade-terminal ground states (scope decisions 2/3).
PRE_REGISTERED_STABLE = frozenset({
    2212, 2112, -2212, -2112,     # p, n, pbar, nbar
    211, -211, 111,               # pi+, pi-, pi0
    321, -321,                    # K+, K-
    310, 130,                     # K0S, K0L
})
GROUND_TERMINAL = frozenset({
    311, -311,                    # K0, aK0
    3122, -3122,                  # Lambda, aLambda
    3222, 3212, 3112,             # Sigma+, Sigma0, Sigma-
})
EXPECTED_STABLE_IDS = PRE_REGISTERED_STABLE | GROUND_TERMINAL

# The 15 records whose m_up follows the documented decays.py
# Breit-Wigner dumphigh cut (scope decision 4): rho(770) 1.5,
# Delta(1232) 2.0 (including the C-symmetric anti partners), K*(892) 1.4.
BW_DUMPHIGH = {
    113: 1.5, 213: 1.5, -213: 1.5,                          # rho(770)
    2224: 2.0, 2214: 2.0, 2114: 2.0, 1114: 2.0,              # Delta(1232)
    -1114: 2.0, -2114: 2.0, -2214: 2.0, -2224: 2.0,          # aDelta(1232)
    323: 1.4, -323: 1.4, 313: 1.4, -313: 1.4,                # K*(892)
}
N1440_IDS = (12112, 12212, -12112)   # the N*(1440) charge states
DELTA_1232_IDS = frozenset({
    1114, 2114, 2214, 2224,           # Delta(1232) charge states
    -1114, -2114, -2214, -2224,       # their anti partners
})

# The channels that violated the documented L convention in the
# original import and were corrected in the b29544f revision: every
# K1(1270) channel (parity rule) and the nine N(1650) Delta pi
# channels (angular-momentum range); all are rule-consistent now.
K1_1270_IDS = frozenset({10323, -10323, 10313, -10313})
N1650_IDS = frozenset({122112, 122212, -122112})

# The non-strange 1+ analogs of the K1(1270): a1(1260) and b1(1235)
# carry the same J^P = 1+ / L = 0 assignment.
A1_1260_IDS = frozenset({20113, 20213, -20213})
B1_1235_IDS = frozenset({10113, 10213, -10213})

# The 18 records whose ``isospin`` field had been transcribed as J
# instead of the PDG isospin (the J-copy bug, corrected in the
# b29544f revision): pdg id -> corrected PDG isospin.
ISOSPIN_PDG_CORRECTION = {
    2116: 0.5, 2216: 0.5, -2116: 0.5,     # N(1520) D13
    22112: 0.5, 22212: 0.5, -22112: 0.5,  # N(1680) F15
    212114: 0.5, 212214: 0.5,             # N(1720) P13
    -212114: 0.5, -212214: 0.5,
    115: 1.0, 215: 1.0, -215: 1.0,        # a2(1320)
    225: 0.0,                             # f2(1270)
    117: 1.0, 217: 1.0, -217: 1.0,        # rho3(1690)
    229: 0.0,                             # f4(2050)
}

# The documented unrepresented channel budgets (branching-fraction sums
# <= 1 with the residual documented).  The Round 2 transcription left the
# radiative modes of eta and eta-prime out (residuals 0.4442 and 0.351);
# the radiative revision of 2026-10-02 (catalog_radiative_revision.py)
# represents them, so the residuals are the small Dalitz, leptonic and
# sub-0.3% modes of eta, eta-prime and phi.
ETA_PDG_ID = 221
ETAPRIME_PDG_ID = 377      # card-pinned key; the true PDG MC id is 331
PHI_PDG_ID = 333
ETA_RESIDUAL = radiative_revision.residual_budget(ETA_PDG_ID)            # 0.0078
PHI_RESIDUAL = radiative_revision.residual_budget(PHI_PDG_ID)            # 0.00161
#: The records the orphan-record revision of 2026-10-29 removed (the
#: eta-prime at the card-pinned key 377), and their tags in the file: one
#: record tag and seven channel tags (the two Round 2 channels and the
#: five of the radiative revision).
ORPHAN_IDS = frozenset(orphan_revision.ORPHAN_RECORDS)
ORPHAN_FILE_TAGS = 1 + 7
#: The channels the radiative revision adds or rewrites (eta +2, omega
#: 2 -> 3, eta-prime +5, phi 2 -> 7).
RADIATIVE_REVISION_CHANNELS = 13
#: The channel the dipion isospin revision adds (Lambda(1520) -> Lambda pi0 pi0).
DIPION_REVISION_CHANNELS = 1
#: The photon's J^P (a stable non-hadronic final state, never a record).
PHOTON_JP = (1.0, "-")

#: Particle-antiparticle pairs present in the committed catalog (every
#: record p for which -p is also a record and (B, Q, S) != (0, 0, 0)).
EXPECTED_ANTI_PAIR_COUNT = 30


@pytest.fixture(scope="module")
def committed_payload() -> dict:
    """The raw JSON of the committed catalog file (plan section 4.4)."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed_catalog() -> HadronCatalog:
    """The committed catalog parsed into ``HadronCatalog``."""
    return HadronCatalog.from_json(CATALOG_PATH)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _sign(parity: str) -> int:
    return 1 if parity == "+" else -1


def _parity_rule_holds(L: int, p_parent: str, p_daughters) -> bool:
    """The documented parity rule (−1)^L = P_parent / prod(P_daughters)."""
    prod = 1
    for p in p_daughters:
        prod *= _sign(p)
    required = -1 if (1 if p_parent == "+" else -1) * prod < 0 else 1
    return ((-1) ** L) == required


def _two_body_j_lattice(s1: float, s2: float) -> list[float]:
    """The daughter spin-coupling values |s1 − s2| .. s1 + s2, step 1."""
    lattice: list[float] = []
    j = abs(s1 - s2)
    while j <= s1 + s2 + 1e-9:
        lattice.append(j)
        j += 1.0
    return lattice


def _l_feasible(L: int, J: float, j_lattice) -> bool:
    """Whether J can be composed from L and some daughter spin j."""
    return any(abs(j - L) <= J + 1e-9 <= j + L + 1e-9 for j in j_lattice)


def _lowest_parity_allowed_L(J: float, j_lattice, p_parent: str,
                             p_daughters) -> int:
    """The smallest L consistent with the parity rule and the angular-
    momentum range (the documented convention, scope decision 5)."""
    for L in range(0, int(J + max(j_lattice)) + 2):
        if _parity_rule_holds(L, p_parent, p_daughters) and _l_feasible(
                L, J, j_lattice):
            return L
    raise AssertionError(
        f"no parity-allowed feasible L for J = {J} (parent parity "
        f"{p_parent})")


def _expected_L(catalog: HadronCatalog, record, channel) -> int:
    """The documented lowest parity-allowed L for one channel (a photon
    daughter carries J^P = 1-; the radiative revision of 2026-10-02)."""
    by_id = {int(r.pdg_id): (float(r.spin), r.parity) for r in catalog.records()}
    by_id[22] = PHOTON_JP
    if channel.is_three_body:
        return 0 if _parity_rule_holds(
            0, record.parity,
            [by_id[int(d)][1] for d in channel.daughters]) else 1
    (d1, d2) = channel.daughters
    return _lowest_parity_allowed_L(
        float(record.spin),
        _two_body_j_lattice(by_id[int(d1)][0], by_id[int(d2)][0]),
        record.parity,
        [by_id[int(d1)][1], by_id[int(d2)][1]])


def _is_previously_deviant_l_channel(record, channel) -> bool:
    """Whether the channel violated the documented L convention in the
    original import (corrected in the b29544f revision): any K1(1270)
    channel, or an N(1650) channel with a Delta(1232) daughter."""
    if int(record.pdg_id) in K1_1270_IDS:
        return not channel.is_three_body
    return (int(record.pdg_id) in N1650_IDS
            and bool({int(d) for d in channel.daughters} & DELTA_1232_IDS))


# ----------------------------------------------------------------------
# Header and validation
# ----------------------------------------------------------------------
def test_header_and_zero_validation_errors(committed_payload,
                                           committed_catalog):
    """The card's public-interface header and a zero-error validation; the
    only warnings are the documented unrepresented channel budgets of the
    eta and the phi."""
    payload = committed_payload
    assert payload["catalog_version"] == 1
    assert payload["schema_version"] == 1
    assert payload["label"] == "r2_initial_catalog"
    assert payload["recorded_on"] == "2026-09-13"
    assert payload["provenance_discipline"] == (
        "plan section 8.2; every physics value carries a ProvenanceTag")
    assert payload["stable_states"] == [
        2212, 2112, -2212, -2112, 211, -211, 111, 321, -321, 310, 130]
    assert len(payload["records"]) == 98 - len(ORPHAN_IDS)

    report = committed_catalog.validate()
    assert report.errors == (), report.errors
    # Exactly the documented residual budgets of the eta and the phi (the
    # radiative revision of 2026-10-02; the eta-prime's left with its
    # record in the orphan-record revision).
    assert len(report.warnings) == 2, report.warnings
    warning_names = {w.split(":", 1)[0].split("record ", 1)[1]
                     for w in report.warnings}
    assert warning_names == {
        f"eta (pdg {ETA_PDG_ID})",
        f"phi(1020) (pdg {PHI_PDG_ID})"}, warning_names


# ----------------------------------------------------------------------
# 1. Eta-prime: the transcription's record at the card-pinned key 377
# ----------------------------------------------------------------------
def test_eta_prime_record(committed_payload, committed_catalog):
    """Scope 1: the driver's transcription carries the eta-prime at the
    card-pinned PDG key 377 (name eta-prime(958); the true PDG MC id is
    331, flagged for a later revision card) with m = 0.95778, Gamma =
    0.000188 GeV and the two documented PDG channels pi+ pi- eta / pi0 pi0
    eta. The orphan-record revision of 2026-10-29 is that revision: no
    class and no channel reaches the record, the kernel's eta' is the
    record 331 of KernelConfig.eta_prime, and the committed file holds
    neither key."""
    entry = next(e for e in import_driver.compose_catalog_values()["records"]
                 if int(e["pdg_id"]) == ETAPRIME_PDG_ID)
    assert entry["name"] == "eta-prime(958)"
    assert entry["values"]["pole_mass"]["transcribed_value"] == 0.95778
    assert entry["values"]["nominal_width"]["transcribed_value"] == 0.000188
    assert (int(entry["B"]), int(entry["Q"]), int(entry["S"])) == (0, 0, 0)
    assert entry["stable_for_decay"] is False
    assert {tuple(sorted(c["daughters"])): c["branching_fraction"] for c in entry["channels"]} == {
        (-211, 211, 221): 0.425, (111, 111, 221): 0.224}
    assert all("PDG 2024" in c["provenance"]["document"] for c in entry["channels"])
    assert "PDG 2024" in entry["record_provenance"]["document"]
    # The id correction is recorded (card-pinned 377 vs PDG MC 331), and
    # neither key is a vendored entry.
    assert "377" in import_driver.RECORD_SET_NOTE
    assert "331" in import_driver.RECORD_SET_NOTE
    assert get_particle(331) is None
    assert get_particle(377) is None

    # The orphan-record revision removed the record from the file.
    assert orphan_revision.ORPHAN_RECORDS == {ETAPRIME_PDG_ID: "eta-prime(958)"}
    ids = {int(r.pdg_id) for r in committed_catalog.records()}
    assert not ids & {ETAPRIME_PDG_ID, 331}
    assert ETAPRIME_PDG_ID not in {int(r["pdg_id"]) for r in committed_payload["records"]}


# ----------------------------------------------------------------------
# 2. Record-count reconciliation
# ----------------------------------------------------------------------
def test_record_count_reconciliation(committed_payload, committed_catalog):
    """Scope 2: record_count = 98 = 87 + 11 in the transcription, with the
    87 comprising 80 decayable resonances plus 7 cascade-terminal ground
    states; the 11 is the pre-registered termination set (card-1
    STABLE_STATES). The committed file holds one decayable record fewer
    after the orphan-record revision: 97 = 86 + 11."""
    records = committed_catalog.records()
    assert len(records) == 98 - len(ORPHAN_IDS)

    header_stable = set(committed_payload["stable_states"])
    assert header_stable == PRE_REGISTERED_STABLE
    assert PRE_REGISTERED_STABLE == STABLE_STATES  # card-1 termination set

    stable = [r for r in records if r.stable_for_decay]
    decayable = [r for r in records if not r.stable_for_decay]
    outside = [r for r in records if int(r.pdg_id) not in header_stable]
    assert len(stable) == 18
    assert len(decayable) == 80 - len(ORPHAN_IDS)
    assert len(outside) == 87 - len(ORPHAN_IDS)
    assert len(records) == len(outside) + 11
    # The 87 = the 80 decayable resonances plus the 7 cascade-terminal
    # ground states.
    assert {int(r.pdg_id) for r in outside} == (
        {int(r.pdg_id) for r in decayable} | GROUND_TERMINAL)
    assert not ({int(r.pdg_id) for r in decayable} & GROUND_TERMINAL)
    # Color-singlet audit: every record carries an integer (B, Q, S).
    for record in records:
        assert record.B == int(record.B) and record.Q == int(record.Q)
        assert record.S == int(record.S)
    # The record set equals the driver's required set (scope decision 2)
    # without the orphan records.
    required = (import_driver.required_unstable_ids()
                | import_driver.required_stable_ids())
    assert ORPHAN_IDS <= set(required)
    assert {int(r.pdg_id) for r in records} == set(required) - ORPHAN_IDS


# ----------------------------------------------------------------------
# 3. Terminal records
# ----------------------------------------------------------------------
def test_terminal_records_are_channel_free(committed_catalog):
    """Scope 3: the 18 stable_for_decay records (11-state set +
    K0/aK0/Lambda/aLambda/Sigma+/Sigma0/Sigma-) are terminal: no catalog
    channels and mass_range = (pole, pole)."""
    records = committed_catalog.records()
    stable = [r for r in records if r.stable_for_decay]
    assert {int(r.pdg_id) for r in stable} == EXPECTED_STABLE_IDS
    for record in stable:
        assert record.decay_channels == ()
        assert record.mass_range == (record.pole_mass, record.pole_mass)
    # Every decayable record has at least one channel (the other side of
    # the card's termination rule).
    for record in records:
        if not record.stable_for_decay:
            assert record.decay_channels != ()


# ----------------------------------------------------------------------
# 4. m_up convention
# ----------------------------------------------------------------------
def test_m_up_bw_dumphigh_documented(committed_catalog):
    """Scope 4 (documented half): the 15 records with a documented
    decays.py Breit-Wigner dumphigh cut (rho(770) 1.5, Delta(1232) 2.0
    including the C-symmetric anti partners, K*(892) 1.4) carry exactly
    that cut as m_up, and the transcription tags cite the documented
    cut for exactly these records."""
    records = {int(r.pdg_id): r for r in committed_catalog.records()}
    for pid, expected in BW_DUMPHIGH.items():
        assert records[pid].mass_range[1] == pytest.approx(expected,
                                                           abs=EXACT_TOL), pid
    values = import_driver.compose_catalog_values()
    cited: dict[int, dict] = {}
    for entry in values["records"]:
        tag = entry["values"]["mass_range"]
        if "dumphigh" in tag["section_or_table"]:
            cited[int(entry["pdg_id"])] = tag
    assert set(cited) == set(BW_DUMPHIGH), sorted(set(cited)
                                                  ^ set(BW_DUMPHIGH))
    for pid, expected in BW_DUMPHIGH.items():
        assert cited[pid]["transcribed_value"] == pytest.approx(
            expected, abs=EXACT_TOL)
        assert f"dumphigh = {expected}" in cited[pid]["section_or_table"]


def test_m_up_pole_plus_five_width_rule(committed_catalog):
    """Scope 4 (rule half): every non-BW decayable record carries
    m_up = pole + 5 * nominal_width, the pre-registered rule (the
    N*(1440) records conform since the b29544f revision)."""
    records = {int(r.pdg_id): r for r in committed_catalog.records()}
    violations = []
    for pid, record in records.items():
        if record.stable_for_decay or pid in BW_DUMPHIGH:
            continue
        expected = record.pole_mass + 5.0 * record.nominal_width
        if abs(record.mass_range[1] - expected) > MASS_TOL:
            violations.append(
                f"{record.name} (pdg {pid}): m_up = {record.mass_range[1]} "
                f"but pole + 5 * nominal_width = {expected}")
    assert not violations, "\n".join(violations)


def test_m_up_n1440_pole_plus_five_width_rule(committed_catalog):
    """The N*(1440) m_up values equal pole + 5 * nominal_width per the
    documented pre-registered rule: 2.94 = 1.44 + 5 * 0.30, with the
    nominal_width unchanged at the vendored PDG 2024 value 0.30 (the
    b29544f revision replaced the original 3.19 = 1.44 + 5 * 0.35,
    whose 0.35 width was documented nowhere in the repository)."""
    records = {int(r.pdg_id): r for r in committed_catalog.records()}
    for pid in N1440_IDS:
        record = records[pid]
        assert record.pole_mass == pytest.approx(1.44, abs=MASS_TOL)
        assert record.nominal_width == pytest.approx(0.30, abs=MASS_TOL)
        expected = record.pole_mass + 5.0 * record.nominal_width
        assert expected == pytest.approx(2.94, abs=MASS_TOL)
        assert record.mass_range[1] == pytest.approx(expected, abs=MASS_TOL)


def test_m_up_terminal_convention(committed_catalog):
    """Scope 4 (terminal half): the 18 stable records carry the terminal
    convention m_up = pole (mass_range = (pole, pole)), and the
    transcription cites it."""
    records = [r for r in committed_catalog.records() if r.stable_for_decay]
    assert len(records) == 18
    for record in records:
        assert record.mass_range == (record.pole_mass, record.pole_mass)
    values = {int(e["pdg_id"]): e for e in
              import_driver.compose_catalog_values()["records"]}
    for record in records:
        tag = values[record.pdg_id]["values"]["mass_range"]
        assert "terminal-record convention" in tag["document"]
        assert tag["transcribed_value"] == pytest.approx(record.pole_mass,
                                                         abs=EXACT_TOL)


# ----------------------------------------------------------------------
# 5. Orbital angular momentum convention
# ----------------------------------------------------------------------
def test_omega_1420_1650_L1(committed_catalog):
    """Scope 5 (worked example): omega(1420) and omega(1650), J^P = 1-,
    decay through rho0 pi0 (daughter parity product +) at the lowest
    parity-allowed L = 1.  Round 2 card 5
    (TASK-20260913-r2-isospin-charge-completion) completes the rho-pi
    class to the exact I = 0 Clebsch-Gordan thirds (rho0 pi0, rho+ pi-,
    rho- pi+), so the single representative channel of the card-2
    transcription becomes the three-channel charge class."""
    for pid in (100223, 30223):
        omega = committed_catalog.get(pid)
        assert (omega.spin, omega.parity) == (1.0, "-")
        channels = omega.decay_channels
        assert len(channels) == 3  # the card-5 completed rho-pi class
        seen = set()
        for channel in channels:
            assert tuple(sorted(int(d) for d in channel.daughters)) in {
                (111, 113), (-213, 211), (-211, 213)}
            seen.add(tuple(sorted(int(d) for d in channel.daughters)))
            assert channel.orbital_angular_momentum == 1
            if channel.provenance is not None and channel.provenance.field \
                    == "branching_fraction":
                # the card-2 representative tag keeps its transcribed
                # value; the per-charge weight is the card-5 derived
                # third
                assert channel.provenance.transcribed_value == \
                    pytest.approx(1.0, abs=EXACT_TOL)
        assert seen == {(111, 113), (-213, 211), (-211, 213)}
        assert all(
            c.branching_fraction == pytest.approx(1.0 / 3.0, abs=EXACT_TOL)
            for c in channels)


def test_l_convention_all_channels(committed_catalog):
    """Scope 5 (rule, all channels): every channel of every decayable
    record obeys the documented convention (parity rule; lowest
    parity-allowed partial wave within the angular-momentum range;
    three-body total L).  The b29544f revision made the 29 previously
    deviating K1(1270) / N(1650)-Delta pi channels conform, so no
    channel is excluded."""
    records = committed_catalog.records()
    violations = []
    for record in records:
        if record.stable_for_decay:
            continue
        for channel in record.decay_channels:
            if channel.orbital_angular_momentum != _expected_L(
                    committed_catalog, record, channel):
                violations.append(
                    f"{record.name} (pdg {record.pdg_id}) -> "
                    f"{tuple(channel.daughters)}: L = "
                    f"{channel.orbital_angular_momentum}, expected "
                    f"{_expected_L(committed_catalog, record, channel)}")
    assert not violations, "\n".join(violations)


def test_l_documented_deviation_inventory(committed_catalog):
    """Post-revision pin (b29544f): the L-rule deviation inventory is
    empty.  Every channel of every decayable record obeys the
    documented convention, and the 29 channels that violated it in the
    original import (the 20 K1(1270) channels and the nine N(1650)
    Delta pi channels) are now rule-consistent: the K1 channels carry
    L = 0 and the Delta pi channels carry L = 2."""
    violations = []
    k1_checked = 0
    n1650_checked = 0
    for record in committed_catalog.records():
        if record.stable_for_decay:
            continue
        for channel in record.decay_channels:
            expected = _expected_L(committed_catalog, record, channel)
            if channel.orbital_angular_momentum != expected:
                violations.append(
                    f"{record.name} (pdg {record.pdg_id}) -> "
                    f"{tuple(channel.daughters)}: L = "
                    f"{channel.orbital_angular_momentum}, expected "
                    f"{expected}")
            if _is_previously_deviant_l_channel(record, channel):
                if int(record.pdg_id) in K1_1270_IDS:
                    k1_checked += 1
                    assert channel.orbital_angular_momentum == 0
                else:
                    n1650_checked += 1
                    assert channel.orbital_angular_momentum == 2
    assert not violations, "\n".join(violations)
    assert k1_checked == 20  # 4 K1(1270) charge states x 5 channels
    assert n1650_checked == 9  # 3 N(1650) charge states x 3 Delta pi


def test_l_k1_1270_parity_rule(committed_catalog):
    """The K1(1270) charge states 10323, -10323, 10313, -10313 carry
    the PDG 2024 J^P = 1/2(1+) (spin 1, parity ``+``), so the lowest
    parity-allowed wave L = 0 on every channel (rho K, K* pi, omega K:
    two negative-parity daughters), matching the non-strange analogs
    a1(1260) and b1(1235); the b29544f revision corrected the original
    1- transcription that made L = 0 parity-forbidden."""
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}
    for pid in sorted(K1_1270_IDS):
        record = by_id[pid]
        assert (record.spin, record.parity) == (1.0, "+"), (
            f"{record.name}: J^P = {record.spin}{record.parity}, "
            f"expected 1+ (PDG 2024 1/2(1+))")
        assert len(record.decay_channels) == 5
        for channel in record.decay_channels:
            assert _expected_L(committed_catalog, record, channel) == 0
            assert channel.orbital_angular_momentum == 0, (
                f"{record.name} -> {tuple(channel.daughters)}: L = "
                f"{channel.orbital_angular_momentum}, expected 0")
    # The non-strange 1+ analogs carry the same J^P / L = 0 assignment.
    for pid in sorted(A1_1260_IDS | B1_1235_IDS):
        record = by_id[pid]
        assert (record.spin, record.parity) == (1.0, "+"), (
            f"{record.name}: J^P = {record.spin}{record.parity}, "
            f"expected 1+")
        for channel in record.decay_channels:
            assert channel.orbital_angular_momentum == 0, (
                f"{record.name} -> {tuple(channel.daughters)}: L = "
                f"{channel.orbital_angular_momentum}, expected 0")


def test_l_n1650_deltapi_lowest_parity_allowed(committed_catalog):
    """The nine N(1650) Delta pi channels on 122112, 122212, -122112
    carry the lowest parity-allowed feasible wave L = 2 (J = 1/2 from
    Delta(3/2) + pi(0): the angular-momentum range is L in {1, 2} and
    the parity rule requires even L), and the N pi / N eta / Lambda K
    channels on the same parents stay L = 0; the b29544f revision
    raised the Delta pi channels from L = 0, which cannot compose
    J = 1/2."""
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}
    checked_deltapi = 0
    checked_other = 0
    for pid in sorted(N1650_IDS):
        record = by_id[pid]
        for channel in record.decay_channels:
            if {int(d) for d in channel.daughters} & DELTA_1232_IDS:
                checked_deltapi += 1
                expected = _expected_L(committed_catalog, record, channel)
                assert expected == 2
                assert channel.orbital_angular_momentum == expected, (
                    f"{record.name} -> {tuple(channel.daughters)}: L = "
                    f"{channel.orbital_angular_momentum}, expected "
                    f"{expected}")
            else:
                checked_other += 1
                assert _expected_L(committed_catalog, record, channel) == 0
                assert channel.orbital_angular_momentum == 0, (
                    f"{record.name} -> {tuple(channel.daughters)}: L = "
                    f"{channel.orbital_angular_momentum}, expected 0")
    assert checked_deltapi == 9  # 3 charge states x 3 Delta pi channels
    assert checked_other == 12  # 3 charge states x 4 N pi / N eta / Lambda K


# ----------------------------------------------------------------------
# Isospin correction (the J-copy bug, corrected in b29544f)
# ----------------------------------------------------------------------
def test_isospin_pdg_correction_18_records(committed_catalog):
    """The 18 records whose ``isospin`` field had been transcribed as
    J instead of the PDG isospin carry the corrected PDG values
    (b29544f): N(1520) D13 I = 1/2, N(1680) F15 I = 1/2, N(1720) P13
    I = 1/2, a2(1320) I = 1, f2(1270) I = 0, rho3(1690) I = 1,
    f4(2050) I = 0.  This pins the committed isospin behavior so the
    J-copy bug cannot regress; ``isospin3`` (the 2 * I3 integers) is
    unchanged by the correction and is not asserted here."""
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}
    assert len(ISOSPIN_PDG_CORRECTION) == 18
    for pid, isospin in ISOSPIN_PDG_CORRECTION.items():
        record = by_id[pid]
        assert record.isospin == isospin, (
            f"{record.name} (pdg {pid}): isospin = {record.isospin}, "
            f"expected {isospin} (PDG)")


# ----------------------------------------------------------------------
# Deterministic card checks
# ----------------------------------------------------------------------
def test_bqs_conservation_per_channel(committed_catalog):
    """Card physics rule: every channel conserves (B, Q, S) exactly,
    independently recomputed through quantum_numbers() (not via
    validate())."""
    violations = []
    for record in committed_catalog.records():
        parent = QuantumNumbers(record.B, record.Q, record.S)
        for channel in record.decay_channels:
            daughters = sum(
                (quantum_numbers(int(d)) for d in channel.daughters),
                QuantumNumbers())
            if daughters != parent:
                violations.append(
                    f"{record.name} (pdg {record.pdg_id}) -> "
                    f"{tuple(channel.daughters)}: daughters sum to "
                    f"{daughters.as_tuple()} but the parent carries "
                    f"{parent.as_tuple()}")
    assert not violations, "\n".join(violations)


def test_branching_fractions_per_parent(committed_catalog):
    """Card physics rule: branching fractions in [0, 1], per-parent sum
    <= 1, with the residual documented: only the eta and the phi carry an
    unrepresented channel budget (the Dalitz, leptonic and sub-0.3% modes
    the radiative revision leaves out), matching the two validate()
    warnings."""
    residuals: dict[int, float] = {}
    for record in committed_catalog.records():
        total = 0.0
        for channel in record.decay_channels:
            assert 0.0 <= channel.branching_fraction <= 1.0
            total = math.fsum([total, channel.branching_fraction])
        if record.decay_channels:
            assert total <= 1.0 + 1e-9, (
                f"{record.name}: branching sum {total} exceeds 1")
            if total < 1.0 - 1e-9:
                residuals[int(record.pdg_id)] = 1.0 - total
    assert set(residuals) == {ETA_PDG_ID, PHI_PDG_ID}, residuals
    assert residuals[ETA_PDG_ID] == pytest.approx(ETA_RESIDUAL,
                                                  abs=EXACT_TOL)
    assert residuals[PHI_PDG_ID] == pytest.approx(PHI_RESIDUAL, abs=EXACT_TOL)
    # The two documented residuals are exactly the validate() warnings.
    report = committed_catalog.validate()
    assert len(report.warnings) == 2
    for warning in report.warnings:
        assert "unrepresented channel budget" in warning
    for pdg in (ETA_PDG_ID, PHI_PDG_ID):
        assert sum(f"pdg {pdg}" in w for w in report.warnings) == 1


def test_thresholds_and_open_channels_at_pole(committed_catalog):
    """Physics rule: mass_range[0] equals the lowest channel threshold
    (the line shape is open wherever any channel is open; the card's
    highest-threshold rule was corrected on 2026-09-28, FAB-07),
    mass_range[1] >= mass_range[0], and every decayable record keeps at
    least one channel open at its pole mass (termination rule)."""
    for record in committed_catalog.records():
        if record.stable_for_decay:
            continue
        thresholds = [
            committed_catalog.threshold(record.pdg_id, channel)
            for channel in record.decay_channels]
        assert record.mass_range[0] == pytest.approx(min(thresholds),
                                                     abs=MASS_TOL), (
            f"{record.name}: mass_range[0] = {record.mass_range[0]} vs "
            f"lowest threshold {min(thresholds)}")
        assert record.mass_range[1] >= record.mass_range[0]
        assert any(th <= record.pole_mass + MASS_TOL for th in thresholds), (
            f"{record.name}: no channel open at pole mass "
            f"{record.pole_mass} (lowest threshold {min(thresholds)})")


def test_charge_conjugate_symmetry_of_anti_pairs(committed_catalog):
    """Card physics rule: for every particle-antiparticle pair present in
    the catalog (p and -p both records, (B, Q, S) != (0, 0, 0)), the
    anti-partner's channel set is the charge-conjugate mirror: daughters
    negated (self-conjugate daughters fixed), identical branching
    fractions, identical L."""
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}

    def mirror(pdgid: int) -> int:
        pdgid = int(pdgid)
        return -pdgid if -pdgid in by_id else pdgid

    pairs = 0
    violations = []
    for pdg_id, record in by_id.items():
        anti_id = -pdg_id
        if anti_id not in by_id or pdg_id < anti_id:
            continue
        if (record.B, record.Q, record.S) == (0, 0, 0):
            continue  # self-conjugate neutral record
        anti = by_id[anti_id]
        pairs += 1
        if (anti.B, anti.Q, anti.S) != (-record.B, -record.Q, -record.S):
            violations.append(
                f"pair ({pdg_id}, {anti_id}): anti (B, Q, S) = "
                f"{(anti.B, anti.Q, anti.S)} is not the negation of "
                f"{(record.B, record.Q, record.S)}")
            continue
        for channel in record.decay_channels:
            mirrored = (tuple(sorted(mirror(d) for d in channel.daughters)),
                        channel.branching_fraction,
                        channel.orbital_angular_momentum)
            match = [
                c for c in anti.decay_channels
                if tuple(sorted(int(d) for d in c.daughters)) == mirrored[0]
                and c.orbital_angular_momentum == mirrored[2]
                and abs(c.branching_fraction - mirrored[1]) <= EXACT_TOL]
            if not match:
                violations.append(
                    f"pair ({pdg_id}, {anti_id}): channel "
                    f"{tuple(channel.daughters)} (bf "
                    f"{channel.branching_fraction}, L "
                    f"{channel.orbital_angular_momentum}) has no "
                    f"charge-conjugate mirror in {anti.name}")
    assert pairs == EXPECTED_ANTI_PAIR_COUNT, (
        f"{pairs} particle-antiparticle pairs, expected "
        f"{EXPECTED_ANTI_PAIR_COUNT}")
    assert not violations, "\n".join(violations)


# ----------------------------------------------------------------------
# 6. Provenance completeness
# ----------------------------------------------------------------------
def test_provenance_tags_complete_and_1080(committed_payload,
                                           committed_catalog):
    """Scope 6: every record and every channel of the committed file
    carries a complete six-field ProvenanceTag (record-level tags carry a
    null transcribed_value by design; every other tag is fully non-null),
    and the driver's transcription carries exactly 1080 tags
    (98 records x 8 + 296 channels) with 0 incomplete."""
    tag_fields = ("document", "section_or_table", "field", "accessed_on",
                  "transcribed_value", "transcriber")

    def incomplete(tag, allow_null_value: bool = False) -> list[str]:
        if not isinstance(tag, dict):
            return ["not a mapping"]
        problems = []
        for field in tag_fields:
            if field not in tag:
                problems.append(f"{field} missing")
            elif tag[field] is None and not (
                    field == "transcribed_value" and allow_null_value):
                problems.append(f"{field} null")
        return problems

    # Committed file: one record tag + one channel tag per channel.
    # Round 2 card 5 (TASK-20260913-r2-isospin-charge-completion) adds
    # five derived sibling channels (isospin-derived provenance), the
    # radiative revision of 2026-10-02 adds 13 channels net and the dipion
    # revision one, and the orphan-record revision removes the eta-prime's
    # record tag and seven channel tags: 98 + 296 + 5 + 13 + 1 - 8 = 405.
    file_tags = 0
    problems = []
    for raw in committed_payload["records"]:
        file_tags += 1
        bad = incomplete(raw["provenance"], allow_null_value=True)
        if bad or raw["provenance"].get("transcribed_value") is not None \
                or raw["provenance"].get("field") != "record":
            problems.append(f"record {raw['pdg_id']}: {bad or 'tag shape'}")
        for channel in raw["decay_channels"]:
            file_tags += 1
            bad = incomplete(channel["provenance"])
            if bad or channel["provenance"].get("field") not in \
                    ("branching_fraction", "isospin-derived"):
                problems.append(
                    f"record {raw['pdg_id']} channel "
                    f"{channel['daughters']}: {bad or 'tag shape'}")
    assert file_tags == (98 + 296 + 5 + RADIATIVE_REVISION_CHANNELS + DIPION_REVISION_CHANNELS
                         - ORPHAN_FILE_TAGS), file_tags
    assert not problems, problems[:8]

    # The transcription: 98 records x (1 record tag + 7 value tags) +
    # 296 channel tags = 1080, 0 incomplete.
    values = import_driver.compose_catalog_values()
    assert len(values["records"]) == 98
    emb_tags = 0
    problems = []
    for entry in values["records"]:
        emb_tags += 1
        bad = incomplete(entry["record_provenance"], allow_null_value=True)
        if bad or entry["record_provenance"]["transcribed_value"] is not \
                None or entry["record_provenance"]["field"] != "record":
            problems.append(f"record {entry['pdg_id']}: {bad or 'shape'}")
        for slot, tag in entry["values"].items():
            emb_tags += 1
            bad = incomplete(tag)
            if bad or tag["field"] != slot:
                problems.append(
                    f"record {entry['pdg_id']} {slot}: {bad or 'field'}")
        for channel in entry["channels"]:
            emb_tags += 1
            bad = incomplete(channel["provenance"])
            if bad or channel["provenance"]["field"] != \
                    "branching_fraction":
                problems.append(
                    f"record {entry['pdg_id']} channel "
                    f"{list(channel['daughters'])}: {bad or 'field'}")
    channel_count = sum(len(e["channels"]) for e in values["records"])
    assert channel_count == 296
    assert emb_tags == 98 * 8 + 296 == 1080, emb_tags
    assert not problems, problems[:8]

    # Every value tag's transcribed_value equals the catalog value it
    # tags, and every channel tag's transcribed_value equals the channel
    # branching fraction.
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}
    for entry in values["records"]:
        if int(entry["pdg_id"]) in ORPHAN_IDS:
            continue
        record = by_id[int(entry["pdg_id"])]
        assert entry["values"]["pole_mass"]["transcribed_value"] == (
            record.pole_mass)
        assert entry["values"]["nominal_width"]["transcribed_value"] == (
            record.nominal_width)
        assert entry["values"]["mass_range"]["transcribed_value"] == (
            record.mass_range[1])
        assert entry["values"]["spin"]["transcribed_value"] == record.spin
        assert entry["values"]["parity"]["transcribed_value"] == \
            record.parity
        assert entry["values"]["isospin"]["transcribed_value"] == \
            record.isospin
        assert entry["values"]["isospin3"]["transcribed_value"] == \
            record.isospin3
        for channel in entry["channels"]:
            match = [
                c for c in record.decay_channels
                if tuple(int(d) for d in c.daughters) ==
                tuple(int(d) for d in channel["daughters"])]
            assert match, f"record {record.pdg_id} channel " \
                          f"{channel['daughters']} missing in the file"
            if match[0].provenance is not None and \
                    match[0].provenance.field == "isospin-derived":
                # the card-5 representative rewrite supersedes the
                # per-charge interpretation of the card-2 tag (the
                # transcribed multiplet BR became the derived per-charge
                # share); the tag itself is checked above
                continue
            if match[0].provenance is not None and \
                    match[0].provenance.accessed_on == radiative_revision.ACCESSED_ON:
                # the radiative revision (2026-10-02) rewrote the channel
                # with its own tag (omega, phi); checked against the
                # revision table in test_committed_file_matches_driver_transcription
                continue
            if match[0].provenance is not None and \
                    match[0].provenance.transcriber == share_revision.TRANSCRIBER:
                # the isospin-share revision (2026-10-05) restamped the
                # channel; checked against the revision in
                # test_committed_file_matches_driver_transcription
                continue
            if match[0].provenance is not None and \
                    match[0].provenance.transcriber == dipion_revision.TRANSCRIBER:
                # the dipion isospin revision (2026-10-21) restamped the
                # channel; checked against the revision in
                # test_committed_file_matches_driver_transcription
                continue
            assert channel["provenance"]["transcribed_value"] == (
                match[0].branching_fraction)


def test_committed_file_matches_driver_transcription(committed_catalog):
    """Write-after-validate content pin: the committed file is exactly
    the driver's embedded transcription materialized (record set,
    (B, Q, S), quantum numbers, per-channel daughters / branching / L /
    three-body flag), so both trees agree on the import content.  Round
    2 card 5 (TASK-20260913-r2-isospin-charge-completion) completes the
    three representative-channel classes on top of the transcription:
    the committed records carry the derived sibling channels and the
    representative rewrites (isospin-derived provenance), whose
    per-charge weights are the Clebsch-Gordan shares of the transcribed
    multiplet branching fraction."""
    card5_siblings = {100223: 2, 30223: 2, 117: 1}
    values = import_driver.compose_catalog_values()
    # the isospin-share revision (2026-10-05) sits on top as well: its
    # channels carry the Clebsch-Gordan shares of their class totals
    shares = share_revision.revised_shares(json.loads(CATALOG_PATH.read_text()))
    # and the dipion isospin revision (2026-10-21): the Lambda(1520)'s
    # Lambda pi pi as an I = 0 dipion, one channel added
    dipion = dipion_revision.revised_fractions(json.loads(CATALOG_PATH.read_text()))
    dipion_added = {pdg: DIPION_REVISION_CHANNELS for pdg in dipion_revision.DIPION_MODES}
    # and the orphan-record revision (2026-10-29): the eta-prime at the
    # card-pinned key 377 removed
    by_id = {int(r.pdg_id): r for r in committed_catalog.records()}
    assert {int(e["pdg_id"]) for e in values["records"]} - ORPHAN_IDS == set(by_id)

    for entry in values["records"]:
        pid = int(entry["pdg_id"])
        if pid in ORPHAN_IDS:
            continue
        record = by_id[pid]
        if pid in radiative_revision.REVISED_CHANNELS:
            # the radiative revision (2026-10-02) sits on top of the
            # transcription: the committed table is the revision's
            table = radiative_revision.REVISED_CHANNELS[pid]
            assert len(record.decay_channels) == len(table)
            for (daughters, br, L, three, _doc, _sec, _new), channel in zip(table, record.decay_channels):
                assert tuple(int(d) for d in channel.daughters) == tuple(daughters)
                assert channel.branching_fraction == pytest.approx(br, abs=EXACT_TOL)
                assert (channel.orbital_angular_momentum, bool(channel.is_three_body)) == (L, three)
            continue
        assert (record.B, record.Q, record.S) == (
            int(entry["B"]), int(entry["Q"]), int(entry["S"]))
        assert record.isospin == float(entry["values"]["isospin"]
                                       ["transcribed_value"])
        assert record.isospin3 == int(entry["values"]["isospin3"]
                                      ["transcribed_value"])
        assert record.spin == float(entry["values"]["spin"]
                                    ["transcribed_value"])
        assert record.parity == str(entry["values"]["parity"]
                                    ["transcribed_value"])
        assert len(record.decay_channels) == len(entry["channels"]) \
            + card5_siblings.get(pid, 0) + dipion_added.get(pid, 0)
        for key, value in dipion.items():
            if key[0] == pid:
                match = [c for c in record.decay_channels if tuple(sorted(int(d) for d in c.daughters)) == key[1]]
                assert match and match[0].branching_fraction == pytest.approx(value, abs=EXACT_TOL), key
        for channel in entry["channels"]:
            match = [
                c for c in record.decay_channels
                if tuple(int(d) for d in c.daughters) ==
                tuple(int(d) for d in channel["daughters"])]
            assert match, f"{record.name}: channel {channel['daughters']}"
            key = (pid, tuple(int(d) for d in match[0].daughters))
            if (pid, tuple(sorted(key[1]))) in dipion:
                continue
            if key in shares:
                # (a channel already at its share keeps its card-2 tag)
                assert match[0].branching_fraction == pytest.approx(shares[key], abs=EXACT_TOL)
                continue
            if match[0].provenance is not None and \
                    match[0].provenance.field == "isospin-derived":
                # the card-5 representative rewrite: the committed
                # weight is the derived share of the transcribed
                # multiplet branching fraction
                continue
            assert match[0].branching_fraction == pytest.approx(
                float(channel["branching_fraction"]), abs=EXACT_TOL)
            assert match[0].orbital_angular_momentum == (
                int(channel["orbital_angular_momentum"]))
            assert bool(match[0].is_three_body) == (
                bool(channel["is_three_body"]))


def test_dual_tree_driver_mirror():
    """The nested driver and its monorepo mirror (outer commit 74bf403)
    are byte-identical, as the card requires."""
    nested_driver = NESTED_ROOT / "helpers" / "benchmarking" / \
        "import_resonance_catalog.py"
    mono_root = NESTED_ROOT.parents[1]
    outer_driver = mono_root / "helpers" / "benchmarking" / \
        "import_resonance_catalog.py"
    if not outer_driver.is_file():
        pytest.skip("monorepo mirror absent (standalone nested checkout)")
    assert nested_driver.read_bytes() == outer_driver.read_bytes()
