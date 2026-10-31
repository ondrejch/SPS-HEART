"""Stage-2 fit allow-list tests for the single effective transfer slope.

Task TASK-20260905-physics-review, milestone M3, pinning the committed
behavior of the nested tree at b5fec81 (after M1 c927e3a/4cd0925 and
M2 cde6f9b/9031c35):

- The stage-2 allow-list fits a single effective transfer slope at one
  beam energy. ``regge.effective_transfer_slope`` is the only fittable
  transfer degree of freedom; ``regge.transfer_slope`` (B) and
  ``regge.alpha_p_prime`` are each refused as a free parameter with the
  non-identifiability reason: at one beam momentum the normalized t
  distribution depends only on B_eff = B + 2 alpha'_P ln(s/s0), so
  separating alpha'_P from B requires an energy lever arm or
  independent transfer data. The retained ``regge.alpha_p0`` refusal
  and the refusal of an unset ``effective_transfer_slope`` field are
  pinned as well.
- ``ReggeExcitationParameters.effective_slope(s)`` returns the field
  when it is set (s is ignored) and the
  ``transfer_slope + 2 alpha'_P ln(s/s0)`` combination when the field
  is None; both branches are pinned.
- At the bundled defaults (B = 4.5, alpha'_P = 0.25, s0 = 1.0) and
  158 GeV/c, B_eff = 7.349 GeV^-2 (3 s.f.), and the bundled
  configuration's ``regge.effective_transfer_slope`` is bit-equal to
  that combination at the code's s.

Discrepancy with the card (recorded per the solver's verification):
card M3 acceptance criterion 2 quotes 10.256 GeV^-2, derived from a
collider-style assumption sqrt(s) = 2 * E_beam = 316.006 GeV
(s = 99859.5 GeV^2, ln(s/s0) = 11.512). The code uses the fixed-target
invariant mass s = 2 m_p (m_p + E_beam) = 298.26 GeV^2 (proton mass
0.93827231 GeV, ``na49_mixing.constants.EMPROT``), so the pinned value
is 7.349 GeV^-2, bit-identical to the pre-change combination.

s recomputation (documented per the card): the tests recompute s
independently from ``EMPROT`` with the legacy p+p closed form the code
uses (``reaction.py``: sqrt(s) = sqrt(2 m_p (m_p + sqrt(m_p^2 + p^2)))
for equal proton masses; ``Reaction.s()`` returns that sqrt(s)^2), and
assert the recomputation is bit-equal to
``Reaction(158.0, 2212, 2212).s()`` — the s the event builder passes
to ``transfer_slope`` (``event_builder_pp.py``: summed incoming
four-vectors' ``mass2()``). The general formula
``m_b^2 + m_t^2 + 2 m_t E_beam`` is NOT bit-equal (a last-ulp
difference) and must not be used for the bit pins.

Call-path choices (recorded per the card): the refusal tests call
``_fit_regge`` directly, the private entry point that consumes the
``fit`` block. The refusals fire during parameter validation before
any event generation or RNG is consumed, so the tests stay immediate
and deterministic without seeding. The acceptance test instead runs
the public ``run_staged_evaluation`` on the bundled configuration at
the smallest statistics the machinery allows: 1 event per objective
evaluation, 1 Powell iteration, 4 events per stage. The
behavior-invariance test reuses the
``tests/test_excitation_stages.py`` stage-stopping pattern; that test
pins the runner at the default dataclass parameters, and nothing in
the suite runs the bundled JSON through all stages, so this file adds
that coverage at reduced statistics (events_per_stage 6 instead of
the bundled 100).

M4 (card TASK-20260905-physics-review) moved the bundled
configuration to the binned six-species objective with the fit
enabled, so the two bundled-configuration tests now read the NA49
tables through ``fit.binned`` and are marked ``external_data`` (the
hermetic archive checkouts deselect them). Their assertions pin the
M4 default: the fit is performed, finite, and its parameters freeze
into the p+A stage. The refusal and allow-list tests stay hermetic.
"""

import json
import math
from pathlib import Path

import pytest

from na49_mixing.constants import EMPROT
from na49_mixing.continuum_channels import ContinuumParameters
from na49_mixing.exchange_model import (
    ReggeExcitationParameters,
    SimpleExcitationParameters,
    transfer_slope,
)
from na49_mixing.reaction import Reaction
from na49_mixing.staged_evaluation import _FITTABLE_STAGE2, _fit_regge, run_staged_evaluation

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CONFIG = REPO_ROOT / "config" / "staged_excitation.json"

BEAM_MOMENTUM = 158.0
REFUSAL_SEED = 20260905


def _load_bundled_config() -> dict:
    return json.loads(BUNDLED_CONFIG.read_text())


def _fixed_target_s(beam_momentum: float = BEAM_MOMENTUM) -> float:
    """Recompute the p+p fixed-target Mandelstam s the code uses.

    Legacy p+p closed form, exactly as ``Reaction._compute`` evaluates
    it for the equal-mass case (``reaction.py``):
    sqrt(s) = sqrt(2 m_p (m_p + sqrt(m_p^2 + p^2))), with m_p = EMPROT
    and p the lab beam momentum. The general formula
    m_b^2 + m_t^2 + 2 m_t E_beam differs in the last ulp and must not
    be substituted for the bit pins.
    """
    p = float(beam_momentum)
    m = EMPROT
    sqrt_s = math.sqrt(2.0 * m * (m + math.sqrt(m * m + p * p)))
    return sqrt_s * sqrt_s


def _minimal_fit_cfg(parameters: list[str]) -> dict:
    """Minimal enabled ``fit`` block: named parameters, one target.

    ``targets`` must be non-empty or ``_fit_regge`` early-returns with
    ``performed: False``; ``bounds`` are not consulted before the
    allow-list or unset-field refusals fire.
    """
    return {
        "enabled": True,
        "parameters": list(parameters),
        "targets": {"mean_source_mass": {"value": 2.4, "sigma": 0.25}},
    }


def _refusal_error(parameters: list[str], regge=None) -> ValueError:
    """Run ``_fit_regge`` on the given parameter list and return the ValueError."""
    fit_cfg = _minimal_fit_cfg(parameters)
    with pytest.raises(ValueError) as excinfo:
        _fit_regge(
            fit_cfg,
            SimpleExcitationParameters(),
            regge if regge is not None else ReggeExcitationParameters(),
            ContinuumParameters(),
            BEAM_MOMENTUM,
            seed=REFUSAL_SEED,
        )
    return excinfo.value


def test_stage2_allowlist_carries_only_the_effective_transfer_slope():
    """The allow-list names the effective slope, not the two components."""
    assert "regge.effective_transfer_slope" in _FITTABLE_STAGE2
    assert "regge.alpha_p_prime" not in _FITTABLE_STAGE2
    assert "regge.transfer_slope" not in _FITTABLE_STAGE2


@pytest.mark.parametrize(
    ("parameters", "refused"),
    [
        (["regge.transfer_slope"], "regge.transfer_slope"),
        (["regge.alpha_p_prime"], "regge.alpha_p_prime"),
        # Both listed: the first name in the list is the one refused.
        (
            ["regge.transfer_slope", "regge.alpha_p_prime"],
            "regge.transfer_slope",
        ),
        (
            ["regge.alpha_p_prime", "regge.transfer_slope"],
            "regge.alpha_p_prime",
        ),
    ],
)
def test_fit_refuses_transfer_slope_components_as_free_parameters(parameters, refused):
    """B and alpha'_P are each refused as a separate free parameter.

    A configuration listing either one alone, or both, raises
    ValueError naming the non-identifiability reason: the B_eff
    combination and the energy-lever-arm requirement.
    """
    message = str(_refusal_error(parameters))
    assert f"{refused} is not fittable at one fixed energy" in message
    assert "B_eff = B + 2 alpha'_P ln(s/s0)" in message
    assert (
        "regge.effective_transfer_slope is the single fittable transfer "
        "degree of freedom" in message
    )
    assert "energy lever arm or independent transfer data" in message


def test_fit_retains_alpha_p0_refusal():
    """The pre-existing alpha_p0 refusal is unchanged by M3."""
    message = str(_refusal_error(["regge.alpha_p0"]))
    assert "regge.alpha_p0 is not fittable at one fixed energy" in message
    assert "only changes the overall normalized event weight" in message


def test_fit_refuses_effective_slope_unset_in_configuration():
    """An allow-listed parameter that is None in the config cannot be fitted."""
    message = str(_refusal_error(["regge.effective_transfer_slope"]))
    assert (
        "regge.effective_transfer_slope is not set in the configuration "
        "and cannot be fitted" in message
    )


def test_effective_slope_at_158_gevc_is_7p349_gev2():
    """The combination at 158 GeV/c is 7.349 GeV^-2 (3 s.f.).

    Card acceptance criterion 2 quoted 10.256 GeV^-2 from the
    collider-style assumption sqrt(s) = 2 * E_beam = 316.006 GeV
    (s = 99859.5 GeV^2, ln(s/s0) = 11.512). The code's fixed-target
    kinematics give s = 298.26 GeV^2 and ln(s/s0) = 5.698 instead, so
    the pinned value is 7.349 GeV^-2 from the bundled defaults
    (B = 4.5, alpha'_P = 0.25, s0 = 1.0).
    """
    regge_cfg = _load_bundled_config()["regge"]
    s = _fixed_target_s(BEAM_MOMENTUM)
    # The code's own s source, asserted bit-equal to the recomputation:
    assert s == Reaction(BEAM_MOMENTUM, 2212, 2212).s()
    # The spec's recorded arithmetic, to the quoted digits:
    assert round(math.sqrt(s), 4) == 17.2702  # sqrt(s) [GeV]
    assert round(s, 2) == 298.26  # s [GeV^2]
    assert round(math.log(s / 1.0), 3) == 5.698  # ln(s/s0), s0 = 1
    combo = (
        float(regge_cfg["transfer_slope"])
        + 2.0 * float(regge_cfg["alpha_p_prime"]) * math.log(s / float(regge_cfg["s0"]))
    )
    assert round(combo, 3) == 7.349  # B_eff [GeV^-2], 3 s.f.
    assert round(combo, 2) == 7.35  # 3 significant figures


def test_effective_slope_field_set_overrides_combination_for_any_s():
    """With the field set, effective_slope(s) ignores s entirely."""
    config = _load_bundled_config()
    params = ReggeExcitationParameters(**config["regge"])
    field = float(config["regge"]["effective_transfer_slope"])
    s = _fixed_target_s(BEAM_MOMENTUM)
    for s_arg in (s, 1.0, 0.5 * s, 1e6):
        assert params.effective_slope(s_arg) == field
        # The module-level caller used by the event builders agrees.
        assert transfer_slope(s_arg, "regge", params) == field


def test_effective_slope_field_unset_uses_energy_dependent_combination():
    """With the field None, B + 2 alpha'_P ln(s/s0) applies and varies with s."""
    regge_cfg = _load_bundled_config()["regge"]
    params = ReggeExcitationParameters(
        **{k: v for k, v in regge_cfg.items() if k != "effective_transfer_slope"}
    )
    assert params.effective_transfer_slope is None
    s = _fixed_target_s(BEAM_MOMENTUM)
    s0 = float(regge_cfg["s0"])
    combo = (
        float(regge_cfg["transfer_slope"])
        + 2.0 * float(regge_cfg["alpha_p_prime"]) * math.log(s / s0)
    )
    assert params.effective_slope(s) == combo
    # At s = s0 the combination degenerates to B itself.
    assert params.effective_slope(s0) == float(regge_cfg["transfer_slope"])
    # The combination carries the energy dependence the field removes.
    assert params.effective_slope(100.0 * s) > combo


def test_bundled_config_effective_slope_inside_its_fit_bounds():
    """The bundled field is bit-equal to its own combination and in-bounds.

    ``fit.parameters`` carries the effective slope and none of the
    refused components; the field value lies inside
    ``fit.bounds["regge.effective_transfer_slope"]``.
    """
    config = _load_bundled_config()
    regge_cfg = config["regge"]
    value = float(regge_cfg["effective_transfer_slope"])
    s = _fixed_target_s(BEAM_MOMENTUM)
    combo = (
        float(regge_cfg["transfer_slope"])
        + 2.0 * float(regge_cfg["alpha_p_prime"]) * math.log(s / float(regge_cfg["s0"]))
    )
    assert value == combo  # bit-equal, as verified against the code's s
    low, high = config["fit"]["bounds"]["regge.effective_transfer_slope"]
    assert (low, high) == (5.07, 10.2)
    assert low < value < high
    assert "regge.effective_transfer_slope" in config["fit"]["parameters"]
    assert "regge.alpha_p_prime" not in config["fit"]["parameters"]
    assert "regge.transfer_slope" not in config["fit"]["parameters"]


@pytest.mark.external_data
def test_bundled_fit_block_is_accepted_by_the_runner():
    """The bundled fit parameter list passes the allow-list end to end.

    The bundled configuration enables the binned six-species objective
    (M4), so this test reads the NA49 tables through ``fit.binned``
    (external_data marker). It runs at the smallest statistics the
    machinery allows (1 event per objective evaluation, 1 Powell
    iteration, 4 events per stage); the run must complete, report a
    performed binned fit over exactly the bundled parameter set, and
    keep the regge stage closed.
    """
    config = _load_bundled_config()
    config["events_per_stage"] = 4
    config["through"] = "regge_fit"
    config["fit"]["enabled"] = True
    config["fit"]["events_per_evaluation"] = 1
    config["fit"]["max_iterations"] = 1
    report, _ = run_staged_evaluation(config)
    fit = report["regge_fit"]["fit"]
    assert fit["performed"] is True
    assert fit["objective"] == "binned"
    assert fit["n_objective_evaluations"] >= 1
    assert math.isfinite(fit["chi2"])
    assert set(fit["parameters"]) == set(config["fit"]["parameters"])
    assert set(fit["species"]) == {t["species"] for t in fit["tables"]}
    low, high = config["fit"]["bounds"]["regge.effective_transfer_slope"]
    fitted = fit["parameters"]["regge.effective_transfer_slope"]
    assert low <= fitted <= high
    assert report["regge_fit"]["summary"]["max_conservation_residual"] < 1e-10


@pytest.mark.external_data
def test_staged_runner_bundled_config_passes_all_stages():
    """Behavior invariance: the bundled configuration still passes all stages.

    Reduced statistics for test runtime: events_per_stage 6 instead of
    the bundled 100, and the binned fit at the smallest statistics the
    machinery allows (2 events per objective evaluation, 1 Powell
    iteration). The bundled configuration enables the binned objective
    (M4), so this test reads the NA49 tables (external_data marker) and
    pins the M4 default: the fit is performed and finite, and its
    parameters freeze into the p+A stage.
    """
    config = _load_bundled_config()
    config["events_per_stage"] = 6
    config["fit"]["events_per_evaluation"] = 2
    config["fit"]["max_iterations"] = 1
    report, samples = run_staged_evaluation(config)
    expected = ["simple_excitation", "regge_fit", "baryon_nucleus"]
    assert report["stage_order"] == expected
    assert list(samples) == expected
    for stage in expected:
        assert report[stage]["status"] == "passed"
        assert report[stage]["summary"]["max_conservation_residual"] < 1e-10
    # The bundled fit is enabled by default (binned objective): the fit
    # is performed and finite, and the fitted p+p parameters are the
    # ones frozen into the p+A stage.
    fit = report["regge_fit"]["fit"]
    assert fit["performed"] is True
    assert fit["objective"] == "binned"
    assert math.isfinite(fit["chi2"])
    assert (
        report["baryon_nucleus"]["frozen_pp_parameters"]["regge"][
            "effective_transfer_slope"
        ]
        == fit["parameters"]["regge.effective_transfer_slope"]
    )
