"""Round 3, card 3: the corrected feed-down lineage convention pins.

Task ``TASK-20260917-r3-gate2-followup-feeddown-lineage-convention``
(the Gate 2 closure obligation, follow-up 3 of the Round 2 closure
record; plan section 8 run R23 line 573 and plan section 9.3).  The R23
feed-down lineage diagnostic reported ``|z| = Infinity`` across all
seven species because the analytic calculation (the R23 driver's
``_ChainExpectation``) assumed all final particles carry resonant
parentage while the event tracker (``_accumulate_feeddown``, the sealed
observed side) records resonant parentage only for non-root cascade
decays.

The corrected convention, pinned here deterministically: resonant
parentage is recorded only for non-root cascade decays.  A particle
produced directly by the source -- a direct daughter of the drawn root
record, or the stable-identity final of a root record that is itself a
stable state -- is not resonantly produced.  The corrected analytic
expectation for a species is ``f_res = 1 - (direct-production
fraction)`` at the source masses realized in the sample.

Tests (all deterministic, hermetic):

- ``test_two_vertex_lineage_convention``: the named regression pin on
  the hand-computed two-vertex fixture (root source -> resonance ->
  stable): the resonance's daughters are resonant feed-down, the root's
  direct daughters are not, and the tracker and the corrected analytic
  chain agree exactly on the same two-vertex lineage.
- ``test_four_source_corrected_expectation``: the corrected expectation
  over a four-record root mixture (two stable records, two unstable) is
  the hand value, and ``f_res = 1 - (direct-production fraction)``
  holds per species.
- ``test_r23_degenerate_infinity_not_reproducible``: the R23-degenerate
  configuration (every terminal resonant, ``f_ana = 1`` exactly, zero
  sigma, an infinite z) cannot be reproduced under the corrected
  convention on the fixture, and a hand replica of the sealed
  degenerate walk reproduces the degenerate mechanism on the same
  topology (the exact difference the correction removes).
- ``test_convention_fix_touches_no_draw_path``: the builder module
  equals its sealed R23 digest and the card-6 kernel equals its
  manifest-pinned digest (no draw sequence, topology, exchange, or
  decay parameter can have moved), and the corrected drivers are
  byte-identical across the two helper trees.

Hermetic: the synthetic catalog and the committed config/v2_singlet
files only; no NA49/SPS external tables, no event generation, no
run-dir writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from na49_mixing.excitation_kinematics import FourVector
from na49_mixing.hadron_catalog import HadronCatalog, HadronRecord
from na49_mixing.resonance_decay import (
    DecayChannel,
    DecayDaughter,
    DecayVertex,
)
from na49_mixing.source_event import FinalParticle

NESTED_ROOT = Path(__file__).resolve().parent.parent
MONO_ROOT = NESTED_ROOT.parent.parent
R23_DRIVER_REL = "helpers/benchmarking/staged_r23_resonance_ablation.py"
REREVAL_DRIVER_REL = "helpers/benchmarking/rereval_feeddown_lineage.py"

#: The card-13 builder-module digest the module must carry.  The sealed
#: R23 pre-extension digest was
#: ``bde0fa9a228f3e9b713cc646bddfc64e2b66b7968a5db3e0c8050d169e81e036``;
#: run R32 (Round 3 card 9, phase 1) extended the builder additively
#: with the sanctioned ``blend`` parameter (a re-pinned freeze
#: exception), and the operative pin is the post-extension digest.  The
#: R23 draw-path byte-stability at ``blend = None`` is pinned by the
#: card-13 pins and by the R32 regression pin over the sealed R23
#: replica-0 stream's first 1,000 event hashes.
SEALED_BUILDER_SHA256 = (
    "4ab612609e1ee0bc1e037dcbb2815b6484306ad32402b7289e6636f8e829cdb1"
)

BUILDER_PATH = NESTED_ROOT / "na49_mixing" / "resonance_pp_builder.py"
MANIFEST_PATH = (
    NESTED_ROOT / "config" / "v2_singlet" / "legacy_v1_manifest.json"
)


# ---------------------------------------------------------------------------
# The synthetic hand-built fixture catalog
# ---------------------------------------------------------------------------
PROTON_PDG = 2212
NEUTRON_PDG = 2112
PI0_PDG = 111
DELTA_PDG = 2214
NSTAR_PDG = 12212
#: The fixture root mass: above the N*+ -> Delta+ pi0 threshold.
ROOT_MASS = 1.6

_DAUGHTER_POLE = {
    PROTON_PDG: 0.9382720813,
    NEUTRON_PDG: 0.9395654133,
    PI0_PDG: 0.1349768,
    DELTA_PDG: 1.232,
}


def _channel(parent_pdg: int, daughters: tuple[int, ...]) -> DecayChannel:
    return DecayChannel(
        parent_pdg=int(parent_pdg),
        daughters=tuple(int(d) for d in daughters),
        branching_fraction=1.0,
        orbital_angular_momentum=0,
        is_three_body=False,
        provenance=None,
    )


def _record(pdg_id: int, name: str, q: int, channels: tuple,
            pole_mass: float, stable: bool) -> HadronRecord:
    """One hand-built record of the p+p source class (B, Q, S) = (1, 1,
    0) (the stable records carry B = 1 too; they stand for the drawn
    stable-identity path of the mixture)."""
    if stable:
        threshold, m_up = float(pole_mass), float(pole_mass)
    else:
        threshold = max(
            math.fsum(_DAUGHTER_POLE[int(d)] for d in ch.daughters)
            for ch in channels
        )
        m_up = threshold + 5.0 * 0.25
    return HadronRecord(
        pdg_id=int(pdg_id),
        name=name,
        B=1, Q=int(q), S=0,
        isospin=0.5,
        isospin3=int(q),
        spin=0.5,
        parity="+",
        pole_mass=float(pole_mass),
        nominal_width=0.25 if not stable else 0.0,
        mass_range=(float(threshold), float(m_up)),
        decay_channels=tuple(channels) if not stable else (),
        stable_for_decay=bool(stable),
        provenance=None,
    )


@lru_cache(maxsize=1)
def _fixture_catalog() -> HadronCatalog:
    """The hand-built fixture catalog: two stable records (proton,
    neutron; the stable-identity path of the mixture) and two unstable
    single-channel records (Delta+ -> p pi0, N*+ -> Delta+ pi0).  Every
    vertex has exactly one open charge combination, so every open-object
    share is 1 and the corrected chain expectation is hand-computable."""
    return HadronCatalog([
        _record(PROTON_PDG, "proton", 1, (), 0.9382720813, True),
        _record(NEUTRON_PDG, "neutron", 0, (), 0.9395654133, True),
        _record(PI0_PDG, "pi0", 0, (), 0.1349768, True),
        _record(DELTA_PDG, "Delta+", 1,
                (_channel(DELTA_PDG, (PROTON_PDG, PI0_PDG)),),
                1.232, False),
        _record(NSTAR_PDG, "Nstar+", 1,
                (_channel(NSTAR_PDG, (DELTA_PDG, PI0_PDG)),),
                1.44, False),
    ])


@lru_cache(maxsize=1)
def _driver():
    """The committed R23 driver module, loaded from its nested path."""
    path = NESTED_ROOT / "helpers" / "benchmarking" / (
        "staged_r23_resonance_ablation.py")
    spec = importlib.util.spec_from_file_location(
        "staged_r23_resonance_ablation_r3card3", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The hand-computed two-vertex fixture (root -> resonance -> stable)
# ---------------------------------------------------------------------------
# Root record N*+ (pdg 12212) at 1.6 GeV decays through its single open
# channel N*+ -> Delta+ + pi0 (one open object, share 1).  The Delta+
# daughter is realized at its catalog pole mass in the analytic walk and
# decays Delta+ -> p + pi0 (again one open object, share 1).  The
# corrected lineage:
#   - the root's direct pi0 terminal: produced by the root vertex, whose
#     parent IS the root record -> NOT resonant;
#   - the Delta+ vertex's p and pi0 terminals: produced by a vertex
#     whose parent (Delta+) is a resonance other than the root record
#     -> resonant.
# Hand-computed corrected expectation for the root record:
#   {"pi_0": (2.0, 1.0), "proton": (1.0, 1.0)}
TWO_VERTEX_EXPECTED = {"pi_0": (2.0, 1.0), "proton": (1.0, 1.0)}


def test_two_vertex_lineage_convention():
    """The named regression pin: on the hand-computed two-vertex
    fixture the corrected analytic chain marks the resonance's daughters
    resonant and the root's direct daughters not resonant, and the
    tracker (``_accumulate_feeddown``) agrees exactly on the same
    two-vertex lineage."""
    r23 = _driver()
    catalog = _fixture_catalog()
    chain = r23._ChainExpectation(catalog)
    got = chain._chain_finals(NSTAR_PDG, ROOT_MASS)
    assert got == TWO_VERTEX_EXPECTED, (
        "the corrected analytic chain on the two-vertex fixture must "
        f"be {TWO_VERTEX_EXPECTED}, got {got!r}"
    )
    # the corrected expectation realizes the hand identity
    # f_res = 1 - direct fraction: pi_0 has one direct terminal (the
    # root vertex) and one cascade terminal (the Delta+ vertex), so
    # f_res = 1/2; proton has only the cascade terminal, f_res = 1.
    assert got["pi_0"][0] == pytest.approx(2.0, abs=1e-12)
    assert got["pi_0"][1] == pytest.approx(1.0, abs=1e-12)
    assert got["proton"][0] == pytest.approx(1.0, abs=1e-12)
    assert got["proton"][1] == pytest.approx(1.0, abs=1e-12)
    f_res_pi0 = got["pi_0"][1] / got["pi_0"][0]
    assert f_res_pi0 == pytest.approx(0.5, abs=1e-12)
    assert 0.0 < f_res_pi0 < 1.0

    # -- the tracker side on the same two-vertex lineage ------------------
    # vertex 0: root N*+ -> Delta+ + pi0 (the pi0 daughter is final);
    # vertex 1: Delta+ -> p + pi0.  Hand lineage: the pi0 with parent
    # vertex 0 is NOT resonant (its producing vertex's parent is the
    # root record); the p and pi0 with parent vertex 1 ARE resonant.
    root_p4 = (ROOT_MASS, 0.0, 0.0, 0.0)
    rest = (ROOT_MASS, 0.0, 0.0, 0.0)
    vertices = [
        DecayVertex(
            parent_pdg=NSTAR_PDG,
            parent_mass=ROOT_MASS,
            parent_four_momentum=root_p4,
            channel=_channel(NSTAR_PDG, (DELTA_PDG, PI0_PDG)),
            charge_combination=(DELTA_PDG, PI0_PDG),
            daughters=(
                DecayDaughter(pdg_id=DELTA_PDG, four_momentum=rest,
                              is_resonance=True, parent_pdg=NSTAR_PDG),
                DecayDaughter(pdg_id=PI0_PDG, four_momentum=rest,
                              is_resonance=False, parent_pdg=NSTAR_PDG),
            ),
        ),
        DecayVertex(
            parent_pdg=DELTA_PDG,
            parent_mass=1.232,
            parent_four_momentum=root_p4,
            channel=_channel(DELTA_PDG, (PROTON_PDG, PI0_PDG)),
            charge_combination=(PROTON_PDG, PI0_PDG),
            daughters=(
                DecayDaughter(pdg_id=PROTON_PDG, four_momentum=rest,
                              is_resonance=False, parent_pdg=DELTA_PDG),
                DecayDaughter(pdg_id=PI0_PDG, four_momentum=rest,
                              is_resonance=False, parent_pdg=DELTA_PDG),
            ),
        ),
    ]
    finals = [
        FinalParticle(0, PI0_PDG, _four_vector(0.1349768), 0, 0, "decay"),
        FinalParticle(1, PROTON_PDG, _four_vector(0.9382720813), 0, 1,
                      "decay"),
        FinalParticle(2, PI0_PDG, _four_vector(0.1349768), 0, 1, "decay"),
    ]
    event = SimpleNamespace(final_particles=finals)

    class _Record:
        source_id = 0
        record_pdg = NSTAR_PDG

    feeddown = r23._empty_feeddown(["pi_0", "proton"])
    r23._accumulate_feeddown(feeddown, catalog, event, _Record(), vertices)
    slot = feeddown["per_species"]
    assert slot["pi_0"]["n"] == 2 and slot["pi_0"]["n_resonant"] == 1, (
        "the tracker must count exactly one of the two pi_0 terminals "
        "as resonant feed-down (the Delta+ vertex's), never the root's "
        "direct daughter"
    )
    assert slot["proton"]["n"] == 1 and slot["proton"]["n_resonant"] == 1
    # exact tracker/analytic convention agreement on the fixture
    assert slot["pi_0"]["n_resonant"] / slot["pi_0"]["n"] == pytest.approx(
        got["pi_0"][1] / got["pi_0"][0], abs=1e-12
    )
    assert slot["proton"]["n_resonant"] / slot["proton"]["n"] == (
        pytest.approx(got["proton"][1] / got["proton"][0], abs=1e-12)
    )


def _four_vector(mass: float) -> FourVector:
    return FourVector(float(mass), 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# The four-source corrected expectation
# ---------------------------------------------------------------------------
FOUR_SOURCE_MASS = 1.6
#: Hand-computed corrected expectation over the four-record mixture at
#: 1.6 GeV (flat record weight 1/4; the stable records contribute their
#: stable-identity finals, Delta+ its direct p and pi0, N*+ its direct
#: pi0 and the Delta+ cascade):
#:   proton:  E = (1 + 0 + 1 + 1)/4 = 0.75, res = (0 + 0 + 0 + 1)/4
#:   pi_0:    E = (0 + 0 + 1 + 2)/4 = 0.75, res = (0 + 0 + 0 + 1)/4
#:   neutron: E = (0 + 1 + 0 + 0)/4 = 0.25, res = 0
FOUR_SOURCE_EXPECTED = {
    "proton": (0.75, 0.25),
    "pi_0": (0.75, 0.25),
    "neutron": (0.25, 0.0),
}


def test_four_source_corrected_expectation():
    """The corrected expectation on the four-source fixture is the hand
    value and ``f_res = 1 - (direct-production fraction)`` per
    species."""
    r23 = _driver()
    catalog = _fixture_catalog()
    chain = r23._ChainExpectation(catalog)
    records = [
        catalog.get(PROTON_PDG), catalog.get(NEUTRON_PDG),
        catalog.get(DELTA_PDG), catalog.get(NSTAR_PDG),
    ]
    got = chain.expected(FOUR_SOURCE_MASS, records)
    assert set(got) == set(FOUR_SOURCE_EXPECTED), (
        f"the four-source fixture species set: {sorted(got)}"
    )
    for key, (n_hand, res_hand) in FOUR_SOURCE_EXPECTED.items():
        n_got, res_got = got[key]
        assert n_got == pytest.approx(n_hand, abs=1e-12), key
        assert res_got == pytest.approx(res_hand, abs=1e-12), key
        f_direct = (n_hand - res_hand) / n_hand
        assert res_hand / n_hand == pytest.approx(1.0 - f_direct,
                                                  abs=1e-12), key
    assert got["neutron"] == pytest.approx((0.25, 0.0), abs=1e-12), (
        "the stable-identity final of a stable root record is a direct "
        "source product, never a resonance feed-down"
    )
    for key in ("proton", "pi_0"):
        f_res = got[key][1] / got[key][0]
        assert 0.0 < f_res < 1.0


# ---------------------------------------------------------------------------
# The R23-degenerate infinity is not reproducible
# ---------------------------------------------------------------------------
def test_r23_degenerate_infinity_not_reproducible():
    """The R23-degenerate configuration (every terminal resonant ->
    ``f_ana = 1`` exactly, zero sigma, an infinite z) cannot be
    reproduced under the corrected convention: the corrected analytic
    carries a direct share on the fixture and the driver's metric
    evaluation is finite for every checked species, while a hand replica
    of the sealed degenerate walk reproduces the degenerate mechanism on
    the same topology (the exact difference the correction removes)."""
    r23 = _driver()
    catalog = _fixture_catalog()

    # (a) the corrected analytic for the two-vertex root sits strictly
    # inside (0, 1) for the species with both direct and cascade
    # terminals: the degenerate value f_ana = 1 is unreachable.
    chain = r23._ChainExpectation(catalog)
    got = chain._chain_finals(NSTAR_PDG, ROOT_MASS)
    assert got["pi_0"] == pytest.approx((2.0, 1.0), abs=1e-12)
    f_ana_corrected = got["pi_0"][1] / got["pi_0"][0]
    assert 0.0 < f_ana_corrected < 1.0

    # (b) the driver's metric evaluation on the fixture: every checked
    # species has a finite sigma and a finite signed z (no |z| =
    # Infinity, the sealed R23 degenerate verdict).
    boundary_block = {
        "mass_histogram_edges": [ROOT_MASS - 0.05, ROOT_MASS + 0.05],
        "mass_histogram_counts": [1],
    }
    feeddown_block = {
        "n_finals_new_kernel": 3,
        "per_species": {
            "pi_0": {"n": 9000, "n_resonant": 4000,
                     "fraction_resonant": 4000.0 / 9000.0,
                     "by_parent": {}},
            "proton": {"n": 3000, "n_resonant": 1500,
                       "fraction_resonant": 1500.0 / 3000.0,
                       "by_parent": {}},
        },
    }
    rows, summary = r23._metric_feeddown(
        catalog, boundary_block, feeddown_block
    )
    assert summary["n_checked"] == 2
    for row in rows:
        assert 0.0 < row["fraction_resonant_analytic"] < 1.0
        assert row["sigma"] > 0.0, (
            "a zero sigma under the corrected convention would "
            "reproduce the R23-degenerate infinite z"
        )
        assert math.isfinite(float(row["z"])), row
        assert row["z"] not in ("Infinity", "-Infinity")

    # (c) the degenerate counterfactual: the sealed walk's flag rule
    # (every daughter of every decay marked resonant) yields
    # f_ana = 1.0 exactly on this fixture; the sealed metric's sigma at
    # f_ana = 1 is exactly zero and its signed-z branch then returns an
    # infinite z for any f_obs != 1 -- the mechanism the correction
    # removes.  Replicated here read-only, by hand.
    degenerate = {"pi_0": (2.0, 2.0)}  # every terminal resonant
    f_ana_degenerate = (
        float(degenerate["pi_0"][1]) / float(degenerate["pi_0"][0])
    )
    assert f_ana_corrected != f_ana_degenerate
    # the sealed sigma convention at the degenerate value
    n = 9000
    f_obs = 4000.0 / 9000.0
    sigma_degenerate = (
        (f_ana_degenerate * (1.0 - f_ana_degenerate) / n) ** 0.5
        if 0.0 < f_ana_degenerate < 1.0 else 0.0
    )
    assert sigma_degenerate == 0.0
    assert f_obs != f_ana_degenerate
    # the sealed metric's explicit z branch at zero sigma
    z_degenerate = z_degenerate_value(f_obs, f_ana_degenerate,
                                      sigma_degenerate)
    assert not math.isfinite(z_degenerate)
    # ... and the corrected analytic never reaches f_ana = 1 here, so
    # the driver's evaluation above stayed finite (asserted in (b)).


def z_degenerate_value(f_obs: float, f_ana: float, sigma: float) -> float:
    """The sealed ``_metric_feeddown`` z rule, replicated read-only for
    the counterfactual pin."""
    if sigma > 0.0:
        return (f_obs - f_ana) / sigma
    if f_obs == f_ana:
        return 0.0
    return math.inf if f_obs > f_ana else float("-inf")


# ---------------------------------------------------------------------------
# The convention fix touches no draw path
# ---------------------------------------------------------------------------
def test_convention_fix_touches_no_draw_path():
    """The pinned physics surfaces are byte-unchanged (the builder
    module equals its sanctioned card-9 post-extension digest, the
    card-6 kernel equals its manifest-pinned digest), so no draw
    sequence, topology, exchange, or decay parameter can have moved;
    and the corrected driver copies are byte-identical across the two
    helper trees."""
    assert _file_sha256(BUILDER_PATH) == SEALED_BUILDER_SHA256, (
        "the card-13 builder module must stay byte-identical to its "
        "sanctioned post-extension digest (the card-9 additive blend "
        "parameter; blend = None keeps the R23 behavior byte-stable)"
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pinned = manifest["source_tree_sha256"]
    kernel_rel = "dev/mixing2006/na49_mixing/resonance_decay.py"
    kernel_path = NESTED_ROOT / "na49_mixing" / "resonance_decay.py"
    assert _file_sha256(kernel_path) == pinned[kernel_rel], (
        "the card-6 decay kernel must stay byte-identical to its "
        "manifest pin (no branching fraction, width, or draw change)"
    )
    for rel in (R23_DRIVER_REL, REREVAL_DRIVER_REL):
        nested = NESTED_ROOT / rel
        outer = MONO_ROOT / rel
        assert _file_sha256(nested) == _file_sha256(outer), (
            f"{rel}: the nested and outer driver copies must be "
            "byte-identical"
        )
