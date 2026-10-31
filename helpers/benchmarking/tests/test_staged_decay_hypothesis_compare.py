"""M7 helper: committed behavior of the staged H1/H2/H3 comparison harness.

Pins the pure, data-independent surface of
``helpers/benchmarking/staged_decay_hypothesis_compare.py`` (the M7
three-hypothesis decay-comparison driver): the pre-registered design
constants, the JSON/JSON-serialization helpers, the xF/pT observable
conventions (per the M4 binned invariant-density normalization), the
H2 resonance-threshold audit, and the config selector path through
``_dataclass_from_dict(ContinuumParameters, config["continuum"])``.

Data-independent and unmarked; no NA49 tables are read and no events
are generated. The two tests that need the nested ``na49_mixing``
package (the H2 audit's pairing table and the config selector's
``ContinuumParameters``) insert ``dev/mixing2006`` on ``sys.path``
inside the test, following the established helper-test pattern.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
MIXING2006 = REPO_ROOT / "dev" / "mixing2006"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.benchmarking import staged_decay_hypothesis_compare as h  # noqa: E402


def _import_nested() -> None:
    """Put the nested package on sys.path (helper-test pattern)."""
    if str(MIXING2006) not in sys.path:
        sys.path.insert(0, str(MIXING2006))


# --- Pre-registered design constants ------------------------------------


def test_pre_registered_design_constants():
    # The comparison design is pre-registered
    # (dev/00runs/staged_decay_compare_2026-09-05); the harness constants
    # are the committed record. Any drift changes the comparison arms.
    assert h.HYPOTHESES == ("H1", "H2", "H3")
    assert h.PARAMETER_POINTS == ("post_fit", "initial")
    assert h.POST_FIT == {
        "regge.alpha_exc0": 0.2327532335505365,
        "regge.effective_transfer_slope": 8.262950772648646,
    }
    assert h.OBSERVABLES_EVENTS == 500
    assert h.SEED_OFFSET == 30000
    assert h.FIT_EVENTS_PER_EVAL == {"H1": 100, "H2": 200, "H3": 200}
    assert h.FIT_MAX_ITERATIONS == {"H1": 20, "H2": 50, "H3": 50}
    assert h.N_PARAMS_DOF == 10
    assert h.XF_BINS == 20
    assert h.XF_MIN == 0.0
    assert h.XF_MAX == 0.4
    assert h.PT_MIN == 0.05
    assert h.PT_MAX_EXCL == 1.55
    assert h.PT_BIN_WIDTH == 0.05
    assert h.H2_THRESHOLD_MARGIN_GEV == 1e-9
    assert h.ARTIFACT_SCHEMA_VERSION == 1


# --- Serialization and kinematic helpers ---------------------------------


def test_to_jsonable_converts_numpy_scalars_and_nonfinite():
    out = h._to_jsonable(
        {
            "f": np.float64(1.5),
            "i": np.int64(7),
            "b": np.bool_(True),
            "arr": np.array([1.0, 2.0]),
            "nan": float("nan"),
            "inf": float("inf"),
            "tup": (1, 2),
            "nested": {"x": [np.float32(0.5)]},
            "s": "keep",
            "n": None,
        }
    )
    assert out["f"] == 1.5 and isinstance(out["f"], float)
    assert out["i"] == 7 and isinstance(out["i"], int)
    assert out["b"] is True
    assert out["arr"] == [1.0, 2.0]
    assert out["nan"] is None
    assert out["inf"] is None
    assert out["tup"] == [1, 2]
    assert out["nested"] == {"x": [0.5]}
    assert out["s"] == "keep"
    assert out["n"] is None


class _V:
    """Minimal four-vector stub (e, px, py, pz)."""

    def __init__(self, e: float, px: float, py: float, pz: float) -> None:
        self.e, self.px, self.py, self.pz = e, px, py, pz


def test_invariant_mass_matches_hand_computed():
    # Back-to-back tracks: transverse momenta cancel, M = E1 + E2.
    a = _V(1.0, 0.3, 0.0, 0.0)
    b = _V(1.0, -0.3, 0.0, 0.0)
    assert h._invariant_mass(a, b) == pytest.approx(2.0, abs=1e-12)
    # Two tracks at rest: M = m1 + m2.
    assert h._invariant_mass(_V(0.5, 0.0, 0.0, 0.0), _V(0.1, 0.0, 0.0, 0.0)) == pytest.approx(
        0.6, abs=1e-12
    )


@dataclasses.dataclass(frozen=True)
class _ReggeStub:
    alpha_exc0: float = 0.5
    effective_transfer_slope: float = 7.3489827740435505
    edge_power: float = 2.0


@dataclasses.dataclass(frozen=True)
class _ContStub:
    decay_hypothesis: str = "H3"
    match_mass: float = 2.4


def test_arm_parameters_post_fit_overrides_only_moved_fields():
    regge0, cont0 = _ReggeStub(), _ContStub()
    for hypothesis in h.HYPOTHESES:
        regge, cont = h._arm_parameters(hypothesis, "post_fit", regge0, cont0)
        # The two M4-moved production-shape parameters ...
        assert regge.alpha_exc0 == 0.2327532335505365
        assert regge.effective_transfer_slope == 8.262950772648646
        # ... while the untouched field keeps its initial value.
        assert regge.edge_power == 2.0
        assert cont.decay_hypothesis == hypothesis
        assert cont.match_mass == 2.4


def test_arm_parameters_initial_keeps_regge_intact():
    regge0, cont0 = _ReggeStub(), _ContStub()
    regge, cont = h._arm_parameters("H3", "initial", regge0, cont0)
    assert regge == regge0
    assert cont.decay_hypothesis == "H3"


def test_arm_parameters_rejects_unknown_point():
    with pytest.raises(ValueError, match="unknown parameter point 'bogus'"):
        h._arm_parameters("H1", "bogus", _ReggeStub(), _ContStub())


def test_conservation_reassert_returns_count_and_max_residual():
    class _Res:
        def __init__(self, m: float) -> None:
            self._m = m

        def validate(self) -> None:
            return None

        def conservation_residuals(self) -> dict:
            return {"relative_max": self._m}

    events = [_Res(1e-14), _Res(3e-15)]
    n_events, max_residual = h._conservation_reassert(events)
    assert n_events == 2
    assert max_residual == 1e-14

    class _Bad:
        def validate(self) -> None:
            raise RuntimeError("boom")

        def conservation_residuals(self) -> dict:
            return {"relative_max": 0.0}

    # A failed validate() propagates (the driver re-raises the event).
    with pytest.raises(RuntimeError, match="boom"):
        h._conservation_reassert([_Bad()])


# --- xF/pT observable conventions ----------------------------------------


class _Particle:
    def __init__(self, pdg_id: int, p4: _V, source_id: int = 0) -> None:
        self.pdg_id, self.p4, self.source_id = pdg_id, p4, source_id


class _Event:
    def __init__(self, particles: list) -> None:
        self.event_id = 0
        self.final_particles = particles
        self.sources = []


class _Table:
    def __init__(self, pdg_id: int, species: str) -> None:
        self.pdg_id, self.species = pdg_id, species


def test_xf_pt_observables_documented_conventions():
    # sqrt(s) = 10.0, 2 events. Conventions under test (harness
    # docstring): dn/dxF(bin) = 2/(sqrt(s) * N_ev * dxF) * sum(E) with the
    # invariant per-track weight E/pT; mean_pt_x the unweighted track mean
    # over the xF bin; mean_pt_y the energy-weighted track mean; windows
    # 0 <= xF < 0.4 and 0.05 <= pT < 1.55 (both boundaries half-open).
    m_pi = 0.1395699
    m_k = 0.493677  # arbitrary positive mass; the convention is under test
    events = [
        _Event(
            [
                _Particle(-211, _V(math.hypot(0.4, m_pi), 0.4, 0.0, 1.0)),  # xf 0.2 -> bin 10
                _Particle(-211, _V(math.hypot(0.5, m_pi), 0.3, 0.4, 2.1)),  # xf 0.42 -> out
                _Particle(-211, _V(math.hypot(0.05, m_pi), 0.05, 0.0, 0.0)),  # xf 0.0 -> bin 0
                _Particle(-211, _V(math.hypot(1.0, m_pi), 0.8, 0.6, 1.0)),  # xf 0.2 -> bin 10
            ]
        ),
        _Event(
            [
                _Particle(-211, _V(math.hypot(0.0, m_pi), 0.0, 0.0, 0.5)),  # pT 0 -> out
                _Particle(-211, _V(math.hypot(1.0, m_pi), 0.6, 0.8, 1.5)),  # xf 0.3 -> bin 15
                _Particle(321, _V(math.hypot(0.2, m_k), 0.2, 0.0, 0.8)),  # xf 0.16 -> bin 8
                _Particle(-321, _V(math.hypot(0.6, m_k), 0.6, 0.0, 2.0)),  # xf 0.4 -> out
                _Particle(-321, _V(math.hypot(1.55, m_k), 0.0, 1.55, 2.0)),  # pT 1.55 -> out
            ]
        ),
    ]
    tables = [
        _Table(-211, "pi_minus"),
        _Table(321, "k_plus"),
        _Table(-321, "k_minus"),
        _Table(2212, "proton"),
    ]
    out = h._xf_pt_observables(events, tables, 10.0)

    norm = 2.0 / (10.0 * 2.0 * 0.02)  # 2/(sqrt(s) * N_ev * dxF) = 5
    e1 = math.hypot(0.4, m_pi)
    e8 = math.hypot(0.05, m_pi)
    e9 = math.hypot(1.0, m_pi)
    e4 = math.hypot(1.0, m_pi)
    e5 = math.hypot(0.2, m_k)

    pi = out["pi_minus"]
    assert pi["n_tracks_window"] == 4
    assert pi["dn_dx_f"][0] == pytest.approx(norm * e8, rel=1e-12)
    assert pi["dn_dx_f"][10] == pytest.approx(norm * (e1 + e9), rel=1e-12)
    assert pi["dn_dx_f"][15] == pytest.approx(norm * e4, rel=1e-12)
    assert float(pi["dn_dx_f"][19]) == 0.0  # xf == 0.4 boundary is excluded
    assert float(np.sum(pi["dn_dx_f"][1:10])) == 0.0  # bins 1..9 empty
    assert float(np.sum(pi["dn_dx_f"][11:15])) == 0.0  # bins 11..14 empty
    assert float(np.sum(pi["dn_dx_f"][16:20])) == 0.0  # bins 16..19 empty
    assert pi["total_yield_window"] == pytest.approx(norm * (e8 + e1 + e9 + e4), rel=1e-12)
    # Bin 10 carries two tracks: the unweighted and energy-weighted means
    # differ, pinning the <pT>_x / <pT>_y convention split.
    assert pi["mean_pt_x"][10] == pytest.approx((0.4 + 1.0) / 2.0, rel=1e-12)
    assert pi["mean_pt_y"][10] == pytest.approx(
        (0.4 * e1 + 1.0 * e9) / (e1 + e9), rel=1e-12
    )
    assert pi["mean_pt_x"][0] == pytest.approx(0.05, rel=1e-12)
    assert pi["mean_pt_x"][15] == pytest.approx(1.0, rel=1e-12)
    assert np.isnan(pi["mean_pt_x"][5])
    assert pi["window_mean_pt_x"] == pytest.approx((0.05 + 0.4 + 1.0 + 1.0) / 4.0, rel=1e-12)
    assert pi["window_mean_pt_y"] == pytest.approx(
        (0.05 * e8 + 0.4 * e1 + 1.0 * e9 + 1.0 * e4) / (e8 + e1 + e9 + e4), rel=1e-12
    )

    kp = out["k_plus"]
    assert kp["n_tracks_window"] == 1
    assert kp["dn_dx_f"][8] == pytest.approx(norm * e5, rel=1e-12)
    assert kp["mean_pt_x"][8] == pytest.approx(0.2, rel=1e-12)
    assert kp["window_mean_pt_x"] == pytest.approx(0.2, rel=1e-12)

    km = out["k_minus"]  # both tracks outside the windows
    assert km["n_tracks_window"] == 0
    assert km["total_yield_window"] == 0.0
    assert np.isnan(km["mean_pt_x"]).all()
    assert km["window_mean_pt_x"] is None
    assert km["window_mean_pt_y"] is None

    pr = out["proton"]  # table species with no tracks
    assert pr["n_tracks_window"] == 0
    assert pr["window_mean_pt_x"] is None


def test_xf_pt_observables_empty_sample_raises():
    with pytest.raises(ValueError, match="empty sample"):
        h._xf_pt_observables([], [_Table(-211, "pi_minus")], 10.0)


# --- H2 resonance-threshold audit -----------------------------------------


def test_h2_resonance_audit_rederives_pairs_and_threshold():
    _import_nested()
    from na49_mixing.pdg_database import get_mass

    m_p = get_mass(2212)
    m_pi = get_mass(211)
    m_pi0 = get_mass(111)
    m_delta = get_mass(2224)

    # Delta++ at rest -> p + pi+: the pair's invariant mass is the
    # resonance mass, margin = m_delta - (m_p + m_pi).
    lam = m_delta**4 + m_p**4 + m_pi**4 - 2.0 * (
        m_delta**2 * m_p**2 + m_delta**2 * m_pi**2 + m_p**2 * m_pi**2
    )
    p = math.sqrt(lam) / (2.0 * m_delta)
    good = _Event(
        [
            _Particle(2212, _V(math.hypot(p, m_p), p, 0.0, 0.0)),
            _Particle(211, _V(math.hypot(p, m_pi), -p, 0.0, 0.0)),
            _Particle(111, _V(m_pi0, 0.0, 0.0, 0.0)),
        ]
    )
    good.sources = [type("_S", (), {"source_id": 0})()]
    n_pairs, min_margin = h._h2_resonance_audit([good], get_mass)
    assert n_pairs == 1
    assert min_margin == pytest.approx(m_delta - m_p - m_pi, abs=1e-9)

    # A pair whose invariant mass sits below the daughter threshold is a
    # hard error (the audit re-derives the pairs from the final state).
    bad = _Event(
        [
            _Particle(2212, _V(0.1, 0.05, 0.0, 0.0)),
            _Particle(211, _V(0.1, -0.05, 0.0, 0.0)),
            _Particle(111, _V(m_pi0, 0.0, 0.0, 0.0)),
        ]
    )
    bad.sources = [type("_S", (), {"source_id": 0})()]
    with pytest.raises(AssertionError, match="below daughter threshold"):
        h._h2_resonance_audit([bad], get_mass)

    # No nucleon-pion / pion-pion opener: no pairs, zero margin.
    none = _Event(
        [
            _Particle(111, _V(m_pi0, 0.0, 0.0, 0.0)),
            _Particle(111, _V(m_pi0, 0.0, 0.0, 0.0)),
        ]
    )
    none.sources = [type("_S", (), {"source_id": 0})()]
    assert h._h2_resonance_audit([none], get_mass) == (0, 0.0)


# --- Config selector path -------------------------------------------------


def test_config_selector_loads_default_h3_from_bundled_config():
    _import_nested()
    from na49_mixing.continuum_channels import ContinuumParameters
    from na49_mixing.staged_evaluation import _dataclass_from_dict

    config = json.loads(
        (MIXING2006 / "config" / "staged_excitation.json").read_text()
    )
    # The committed bundled continuum block omits decay_hypothesis, so the
    # selector's default (H3, the pre-M7 kernel) is selected for the M4
    # parameter points; the block otherwise matches the dataclass
    # defaults field for field.
    assert "decay_hypothesis" not in config["continuum"]
    loaded = _dataclass_from_dict(ContinuumParameters, config["continuum"])
    assert loaded == ContinuumParameters()
    loaded.validate()
    assert loaded.decay_hypothesis == "H3"

    # An in-memory override selects the other committed hypotheses.
    for hypothesis in ("H1", "H2"):
        arm = _dataclass_from_dict(
            ContinuumParameters, {**config["continuum"], "decay_hypothesis": hypothesis}
        )
        arm.validate()
        assert arm.decay_hypothesis == hypothesis

    # Uncommitted values are rejected by the selector.
    with pytest.raises(ValueError, match="must be one of H1, H2, or H3"):
        _dataclass_from_dict(
            ContinuumParameters, {**config["continuum"], "decay_hypothesis": "H4"}
        ).validate()

    # Seed protocol: the committed config seed plus the harness offset is
    # the shared seed of all comparison arms.
    assert config["seed"] == 20260831
    assert config["seed"] + h.SEED_OFFSET == 20290831
