"""The topology template runner (Round 5, card 9).

``TASK-20260925-r5-topology-template-runner``; plan section 22 task 36
(line 1761); plan section 11.1 item 1 (line 824) with run R50 (lines
898-907).  The card packages the landed class modules (cards 1-7, plus
card 8's registered-but-unretained extension) behind one inelastic
class table, the nominal parameter structure, the closed
``topology_selection`` class draw, and the per-event observable
extraction that run R50 (plan lines 898-907) consumes.  The runner
introduces no new physics: every class record it produces is
bit-identical to the class module's own generation on the same
substreams (the dispatch identity, acceptance (iii) of the card).  The
T1 generation, the R50 statistics, and the R50-R55 analysis belong to
the run cards 10-15; this module's trace is the hermetic T0-equivalent
pinned 1,000-event trace.

The class registry (plan section 11.1, lines 820-831)
-----------------------------------------------------

:data:`TOPOLOGY_CLASSES` carries exactly the six entries ``"elastic"``,
``"sd_projectile"``, ``"sd_target"``, ``"dd"``, ``"nd"``, and
``"nd_multi"``.  The elastic class is "used for normalization and code
checks, not included in an inelastic sample" (plan line 824), so its
``inelastic`` flag is ``False``; the multiple-neutral class ``nd_multi``
is registered but unretained (``retained`` False, the card-8 registry
entry) and is excluded from the inelastic draw.  :data:`INELASTIC_SET`
is exactly ``{"sd_projectile", "sd_target", "dd", "nd"}``; the
pre-registered R50 nominal fractions are 0.25 each over these four
classes (round 5 index item 2).

The topology draw
-----------------

:func:`draw_topology` consumes exactly one uniform on the closed
``topology_selection`` substream (plan section 5.2; no new substream is
introduced).  The single uniform partitions the unit interval with the
elastic fraction on the lower edge ``[0, f_elastic)`` and the inelastic
classes on the complement, walked in the fixed :data:`INELASTIC_ORDER`
(projectile single diffraction, target single diffraction, double
diffraction, non-diffractive; the plan line 824 naming order with the
elastic class first).  The caller's *fractions* are the absolute
probabilities of the drawn set: each inelastic class is returned with
probability exactly its given fraction (the walk advances the
cumulative by each class's fraction above the elastic edge, which is
the caller's fractions scaled by ``1 / (1 - f_elastic)`` against the
rescaled inelastic uniform; the validated unit mass
``f_elastic + sum(fractions) = 1`` makes the two forms the same
allocation).  At the nominal point the elastic fraction is the
complement of the inelastic mass (exactly 0.0 at the pre-registered
0.25-each split), so no uniform returns ``"elastic"``; ``"nd_multi"``
is never returned because it is outside :data:`INELASTIC_SET` (checked,
never repaired: unknown class labels and non-unit masses raise).

The per-event observable extraction
-----------------------------------

:func:`extract_observables` computes the R50 stored quantities (plan
line 903: leading baryon, excitation mass, momentum transfer,
multiplicity, central gap) plus the R53 trigger observables (plan
lines 915-918: total multiplicity, central activity ``|y| < 0.5``) and
the seagull moments (plan section 7.3, the m_T-slope moments of the
final-state p_T distributions):

- the topology label;
- the leading-baryon ``x_F = 2 p_L^CM / sqrt(s)`` (Feynman x, plan
  section 7.1) per baryonic line in the overall CM frame, projectile
  side first (the elastic event uses its two outgoing lines; the
  inelastic classes their outgoing baryonic ``lines``, both carrying
  ``B = 1``); every landed class record stores lab-frame four-momenta
  (the class modules' forward fixed-target boost), and the extraction
  applies the inverse boost ``(beta, gamma)`` of the pinned beam
  configuration before forming the Feynman x;
- the excitation masses (the SD excitation, the DD pair, the ND
  central mass, the nd_multi central masses; the elastic class has
  none);
- the momentum transfers (the drawn record: ``t`` for elastic/SD,
  ``(t_1, t_2)`` for DD, ``(t_1, t_2, t_c)`` for ND, and the nd_multi
  ``(t_1, t_2, t_2c, t_c)``);
- the central gap size, the record's ``gap_size`` on the SD and DD
  classes (the plan line 901 gap-size record; a rapidity difference,
  hence frame-independent under the longitudinal boost; ``None`` on
  the classes that carry no gap field);
- the total multiplicity and the central-activity multiplicity in the
  CM frame ``|y^CM| < 0.5`` (the R53 trigger observable; the elastic
  event's outgoing pair is its final state);
- the central and forward pion and proton yields by CM rapidity region
  (central ``|y^CM| < 0.5``, forward otherwise; pions are the charged
  pions and the pi0, protons the ``pdg_id == 2212`` records; the
  antiproton is a distinct species and is not counted as a proton);
- the seagull moment: the arithmetic mean transverse momentum of the
  final state (the per-event building block of the plan section 7.3
  ``<p_T>`` m_T-slope moments; the transverse momentum is invariant
  under the longitudinal boost and is read from the stored lab
  four-momenta; the elastic pair's mean ``p_T``).

The frozen substrate
--------------------

The runner consumes cards 1-8's modules read-only and passes the
frozen simplified cascade through with isotropic decay angles: the R45
alignment result (kappa_B = 0.0, kappa_M = 4.0, b_dec = 2.0 GeV^-2) is
not applied on the topology path, which carries no b_dec form factor,
and the continuum families carry the nominal width 0.5 GeV (corrected
2026-10-06 after the code and physics review's H-6; the earlier text
claimed the R45 point).  No alignment re-fit and no separation claim
exists anywhere in this module (the R46 ``no_refit`` clause), and the
exchange slopes act on the exchange kinematics only, never on any
daughter-level transverse dynamics (plan section 24, line 1866).  The
class fractions, the elastic fraction, and ``P(n >= 2)`` are arguments,
so the R51/R53 fit steps vary them without code change.

Operative resolutions (the card's literals against the committed tree)
----------------------------------------------------------------------

- The runner signatures: the card's sketch compresses the substream
  arguments of :func:`run_topology` to a single ``rng``; the committed
  cards 5-8 pass the closed per-label substream generators, so the
  runner passes them through unchanged (the card-8 recorded pattern).
- The remnant-weight keys resolve to the committed card-3 remnant-kind
  vocabulary (``"intact_proton"``, ``"low_mass_baryonic_resonance"``,
  ``"high_mass_baryonic_continuum"``), the same keys the class modules
  consume; the card's short literals (``"intact"``, ``"low_mass"``,
  ``"high_mass"``) name the same nominal 0.5 / 0.5 / 0.0 split.
- :func:`pinned_trace` loads the committed catalog and the continuum
  families configuration from the package tree when the caller does
  not supply them (the class generators' configuration arguments; the
  card's sketch omits them).
- The return type of :func:`run_topology` is the
  :data:`BaseTopologyEvent` union of the five landed record types
  (there is no shared base class in the landed modules; the union
  keeps the card's annotation name).
- :func:`draw_topology` returns ``"elastic"`` on the lower edge of the
  single uniform when the elastic fraction is positive (plan section
  11.1 item 1 names the elastic topology first); the inelastic walk
  covers the remainder in the fixed :data:`INELASTIC_ORDER` with each
  class returned at exactly its caller fraction (the fractions are the
  absolute probabilities of the drawn set; the unit-mass validation
  makes the scaled and the absolute walk forms the same allocation).

Hermetic: the committed catalog, the continuum families configuration,
the Round 4 cascade modules, and the committed cards 1-8 modules (all
consumed read-only), numpy, and the standard library only; no event
generation driver, no external tables, no Monte Carlo run (the runs
R50-R55 exercise the runner statistically; the acceptance here is the
hermetic pinned trace).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.topology_double_diffraction import DDEvent, generate_dd
from na49_mixing.topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
    generate_elastic,
)
from na49_mixing.topology_mass_laws import MassLawParams
from na49_mixing.topology_multi_neutral import (
    NDMultiEvent,
    generate_nd_multi,
)
from na49_mixing.one_pion_exchange import ope_component_of
from na49_mixing.topology_non_diffractive import NDEvent, generate_nd
from na49_mixing.topology_single_diffraction import FinalHadron, SDEvent, generate_sd
from na49_mixing.flavor_weights import FlavorParams
from na49_mixing.isovector_exchange import ISOVECTOR_CLASSES, draw_exchange_configuration

__all__ = [
    "BEAM_MOMENTUM",
    "CENTRAL_WINDOW",
    "ELASTIC_TRANSFER_SLOPE",
    "INELASTIC_ORDER",
    "INELASTIC_SET",
    "MASTER_SEED",
    "NOMINAL_PARAMS",
    "SQRT_S",
    "TOPOLOGY_CLASSES",
    "BaseTopologyEvent",
    "TopologyNominalParams",
    "TopologyObservables",
    "draw_topology",
    "extract_observables",
    "pinned_trace",
    "run_topology",
]

#: The M8-frozen effective_transfer_slope, byte-identical to the frozen
#: M8 ten-parameter set (hash 358989d84cb03cffbb0120c64992b1bd95ca73260
#: 1135e0aa222a6dda69ec0b1) and to card 1's :data:`~na49_mixing.
#: topology_elastic.ELASTIC_TRANSFER_SLOPE` [GeV^-2].  The R50 nominal
#: value; the elastic module's own constant is consumed read-only and
#: re-exported here so the nominal-parameter pin has a single surface.
ELASTIC_TRANSFER_SLOPE: float = 8.504140762083612

#: The R50 master seed (round 5 index item 9), the default pinned-trace
#: seed (the run cards 10-15 own the R50 replica seeds).
MASTER_SEED: int = 20260925

#: The fixed class ordering of the inelastic walk (plan section 11.1
#: item 1 naming order after the elastic class).
INELASTIC_ORDER: tuple[str, ...] = (
    "sd_projectile",
    "sd_target",
    "dd",
    "nd",
)

#: The six-entry class registry (plan section 11.1 item 1, line 824,
#: with the card-8 registry entry carrying ``retained: False``).  Each
#: entry carries the landed module, the retained flag (the Gate 5
#: retention surface), and the inelastic flag (the elastic class is
#: excluded from every inelastic sample, plan line 824).
TOPOLOGY_CLASSES: Mapping[str, dict] = {
    "elastic": {
        "module": "na49_mixing.topology_elastic",
        "retained": True,
        "inelastic": False,
    },
    "sd_projectile": {
        "module": "na49_mixing.topology_single_diffraction",
        "retained": True,
        "inelastic": True,
    },
    "sd_target": {
        "module": "na49_mixing.topology_single_diffraction",
        "retained": True,
        "inelastic": True,
    },
    "dd": {
        "module": "na49_mixing.topology_double_diffraction",
        "retained": True,
        "inelastic": True,
    },
    "nd": {
        "module": "na49_mixing.topology_non_diffractive",
        "retained": True,
        "inelastic": True,
    },
    "nd_multi": {
        "module": "na49_mixing.topology_multi_neutral",
        "retained": False,
        "inelastic": True,
    },
}

#: The inelastic draw set: exactly the four classes (elastic excluded
#: per plan line 824, nd_multi excluded as the unretained card-8
#: extension).
INELASTIC_SET = frozenset(
    label for label, entry in TOPOLOGY_CLASSES.items()
    if entry["inelastic"] and label != "nd_multi")

#: The fraction-sum tolerance of the draw validation (the house
#: tolerance of the mass checks).
FRACTION_SUM_TOLERANCE: float = 1e-12

#: The central-activity rapidity window of the R53 trigger observable
#: (plan lines 915-918).
CENTRAL_WINDOW: float = 0.5

#: The pion species of the central and forward yields (plan line 906:
#: "central and forward species yields"; the charged pions and the
#: pi0).
PION_IDS = frozenset({111, 211, -211})

#: The proton ``pdg_id`` of the proton-yield counter (the p-bar record
#: ``-2212`` is a distinct species and is not counted).
PROTON_PDG_ID = 2212

#: The union of the five landed record types (the card's
#: ``BaseTopologyEvent`` return surface; there is no shared base class
#: in the landed modules, so the annotation is this union).
BaseTopologyEvent = Union[ElasticEvent, SDEvent, DDEvent, NDEvent,
                          NDMultiEvent]

#: The unretained card-8 default of P(n = 2) (the module surface,
#: consumed read-only here).
DEFAULT_P_N2: float = 0.0


@dataclass(frozen=True)
class TopologyNominalParams:
    """The fixed R50 nominal values (round 5 index item 2); fit
    parameters land in R51 (card 11), not here."""

    epsilon: float = 0.0
    b_sd: float = 8.504140762083612      # GeV^-2, M8-frozen
                                         # effective_transfer_slope
    class_fractions: Mapping[str, float] = field(
        default_factory=lambda: {
            "sd_projectile": 0.25,
            "sd_target": 0.25,
            "dd": 0.25,
            "nd": 0.25,
        })
    remnant_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "intact_proton": 0.5,
            "low_mass_baryonic_resonance": 0.5,
            "high_mass_baryonic_continuum": 0.0,
        })
    p_n2: float = 0.0                    # unretained default, card 8
    flavor: "FlavorParams | None" = None  # Round 6 task 37; None = identity
    epsilon_nd: "float | None" = None    # 2026-09-29 (Block A): the
                                         # non-diffractive mass exponent;
                                         # None = the shared epsilon
    nd_central: bool = True              # 2026-10-02: False draws the
                                         # non-diffractive class without
                                         # its central source (the
                                         # two-component variant)
    isovector_fraction: float = 0.0      # Round 8, 2026-10-04: the share
                                         # of SD, DD and ND events with the
                                         # isovector exchange; 0.0 = the
                                         # vacuum exchange only, no draw
    b_isovector: "float | None" = None   # Round 8, 2026-10-04: the transfer
                                         # slope of the isovector events
                                         # [GeV^-2]; None = the shared b_sd
    isovector_delta_ratio: float = 1.0   # 2026-10-05: the reduced-coupling
                                         # ratio g^2(I=3/2)/g^2(I=1/2) of the
                                         # isovector lines; 1.0 = equal
    b_nd: "float | None" = None          # 2026-10-05: the transfer slope of
                                         # the non-diffractive lines [GeV^-2];
                                         # None = the shared b_sd
    exact_transfer_range: bool = False   # 2026-10-05: draw each 2 -> 2
                                         # class's transfer on its reachable
                                         # window (no angle clamp)
    line_thresholds: bool = False        # 2026-10-06: start each excited
                                         # line's mass law at its own lowest
                                         # direct threshold (review M-6)
    central_recoil: bool = False         # 2026-10-07: the class with the
                                         # central source takes the recoiling
                                         # three-system solve (Stage B)
    epsilon_c: "float | None" = None     # 2026-10-07: the power central mass
                                         # law (M_c^2)^(-1-epsilon_c), its own
                                         # exponent; None = the committed law
    transfer_slope_mass_exponent: "float | None" = None
                                         # 2026-10-08: the factorized vertex
                                         # slopes B = b [(m_p/M_1)^k +
                                         # (m_p/M_2)^k] / 2 of each 2 -> 2
                                         # transfer (needs the exact window);
                                         # None = the single slope b
    low_mass_spin_isospin: bool = False  # 2026-10-08: the vacuum lines draw
                                         # their low-mass record over the
                                         # I = 1/2 records with weight 2J + 1;
                                         # False = uniform over every (1, 1, 0)
                                         # record below the boundary
    nd_duality_mass: "float | None" = None
                                         # 2026-10-12: the vacuum
                                         # non-diffractive lines below this
                                         # mass [GeV] decay as the low-mass
                                         # N* and Delta records
                                         # (na49_mixing.nd_duality); None =
                                         # the continuum cascade
    nd_duality_shape_mass: "float | None" = None
                                         # 2026-10-12: below this mass
                                         # (<= nd_duality_mass) the lines
                                         # take their mass from the records'
                                         # line shapes; None = the continuum
                                         # mass
    duality_line_shapes: bool = False    # 2026-10-18: the duality picks
                                         # (nd_duality_mass and
                                         # diffractive_duality_mass) weight
                                         # each record by its normalized line
                                         # shape with the running width
                                         # (na49_mixing.nd_duality); False =
                                         # the fixed-width Breit-Wigner
    duality_redraw_shapes: bool = False  # 2026-10-20: the redraw of the
                                         # non-diffractive line masses below
                                         # nd_duality_shape_mass follows the
                                         # records' own line shapes (needs
                                         # duality_line_shapes); False = the
                                         # fixed-width mixture
    diffractive_duality_mass: "float | None" = None
                                         # 2026-10-15: a vacuum single- or
                                         # double-diffraction (1, 1, 0) line
                                         # below this mass [GeV] decays as an
                                         # I = 1/2 low-mass record (the vacuum
                                         # exchange's isospin); None = the
                                         # continuum cascade
    intact_vertex_factor: "float | None" = None
                                         # 2026-10-15: under the factorized
                                         # vertex slopes the intact proton of
                                         # a single-diffraction event carries
                                         # the vertex term f instead of 1,
                                         # B = b [f + (m_p/M_X)^k] / 2 (the
                                         # elastic proton vertex); None = 1
    diffractive_xi_max: "float | None" = None
                                         # 2026-10-23: the coherence region of
                                         # diffraction: the single- and
                                         # double-diffraction excited masses
                                         # with M^2/s below this edge
                                         # (Goulianos and Montanha, PRD 59
                                         # (1999) 114017: xi < 0.1); None =
                                         # up to the kinematic edge
    sd_tail: "tuple[float, ...] | None" = None
                                         # 2026-10-23 (R112): the leading-proton
                                         # tail of the single-diffraction
                                         # classes beyond the coherence region,
                                         # (share, xi_lo, xi_hi, a, b): the
                                         # share s of the vacuum events with
                                         # dsigma/dxi ~ xi^a on [xi_lo, xi_hi]
                                         # and the transfer from exp(b t) (the
                                         # reggeized pion exchange of
                                         # Goulianos and Montanha); None = no
                                         # tail.  2026-10-25: (share, xi_lo,
                                         # xi_hi, a, b, xi_c, w) gives the
                                         # law the smooth upper edge
                                         # 1 / (1 + exp((xi - xi_c) / w))
    sd_ope: "tuple[float, ...] | None" = None
                                         # 2026-10-26: the one-pion exchange of
                                         # the single-diffraction classes, (share,
                                         # xi_lo, xi_hi, alpha', r2): a share of the
                                         # vacuum events beside the tail where the
                                         # non-excited line turns into a neutron and
                                         # the excited system is the pi+ p one
                                         # (Q = 2, I = 3/2), (M^2, t) from the pion
                                         # flux (na49_mixing.one_pion_exchange)
    sd_two_sided: "tuple | None" = None
                                         # 2026-10-26: the two-sided hadronization
                                         # of the heavy single-diffraction systems,
                                         # (m_cut, eps_line, b, scope): above m_cut
                                         # a baryonic line along the X flight
                                         # direction and the exchanged object's
                                         # line opposite, the scope's letters
                                         # o (one-pion exchange), t (tail), c
                                         # (coherence region)
    diffractive_pomeron_slope: "float | None" = None
                                         # 2026-11-01: alpha'_P [GeV^-2], the
                                         # Pomeron's shrinkage in the vacuum
                                         # events of the diffractive classes'
                                         # coherence region: the triple-Pomeron
                                         # law at fixed t, xi^(-1-eps-2 alpha' t)
                                         # (topology_mass_laws, "The Pomeron's
                                         # shrinkage"); None = the zero-transfer
                                         # form with the per-mass vertex law
    diffractive_pomeron_linear_t: "float | None" = None
                                         # 2026-11-01: t_lin [GeV^2], the
                                         # shrinkage down to t = -t_lin only
                                         # (the trajectory's linear range; the
                                         # intact proton's break must equal
                                         # it); None = every t
    intact_t_break: "tuple[float, float] | None" = None
                                         # 2026-10-24: the intact proton's
                                         # two-slope transfer in the single-
                                         # diffraction classes, (t_b, rho):
                                         # exp(B t) above -t_b and the slope
                                         # rho B below it (the x_F 0.95
                                         # protons change slope near
                                         # |t| = 1 GeV^2); None = one
                                         # exponential
    nd_symmetric_remnants: bool = False  # 2026-11-04: the two non-diffractive
                                         # remnant masses drawn on one window,
                                         # the pair kept when it leaves a neutral
                                         # pion (symmetric under projectile <->
                                         # target); False = the projectile first,
                                         # the target on the remaining window


#: The R50 nominal parameter set (the module-level frozen instance; the
#: tests pin it field for field against the pre-registered values).
NOMINAL_PARAMS: TopologyNominalParams = TopologyNominalParams()


@dataclass(frozen=True)
class TopologyObservables:
    """The per-event observable record (the R50 stored quantities of
    plan line 903, the R53 trigger observables of plan lines 915-918,
    and the plan section 7.3 seagull moment)."""

    class_label: str
    x_f_projectile: float
    x_f_target: float
    excitation_masses: tuple[float, ...]
    transfers: tuple[float, ...]
    gap_size: Optional[float]
    multiplicity_total: int
    multiplicity_central: int
    central_pions: int
    central_protons: int
    forward_pions: int
    forward_protons: int
    seagull_moment: float


# ----------------------------------------------------------------------
# The closed-substream class draw
# ----------------------------------------------------------------------
def draw_topology(
    fractions: Mapping[str, float],
    elastic_fraction: float,
    rng,
) -> str:
    """The class draw on the closed ``topology_selection`` substream.

    Exactly one uniform is consumed from *rng*.  The elastic fraction
    occupies the lower edge ``[0, elastic_fraction)`` of the unit
    interval (plan section 11.1 item 1 names the elastic topology
    first); above it the inelastic classes are walked in the fixed
    :data:`INELASTIC_ORDER` over the absolute cumulatives, so each
    inelastic class is returned with probability exactly its caller
    fraction: *fractions* are the absolute probabilities of the drawn
    set, equivalently the caller's fractions scaled by
    ``1 / (1 - elastic_fraction)`` against the rescaled inelastic
    uniform (the validated unit mass makes the two forms the same
    allocation; the walk advances the cumulative by each class's
    fraction above the elastic edge).  ``"nd_multi"`` is registered but
    never returned (it is outside :data:`INELASTIC_SET`, the unretained
    card-8 class; the default ``P(n = 2) = 0.0`` keeps it out of R50
    entirely).

    Raises :class:`ValueError` naming the offending labels on unknown
    class labels (anything outside :data:`INELASTIC_SET`), missing
    inelastic classes, negative or non-finite fractions, or a non-unit
    total mass (the elastic fraction plus the inelastic mass must equal
    one within :data:`FRACTION_SUM_TOLERANCE`).
    """

    e_fraction = float(elastic_fraction)
    if not math.isfinite(e_fraction) or not 0.0 <= e_fraction <= 1.0:
        raise ValueError(
            "draw_topology: the elastic fraction must be a finite value "
            f"in [0, 1], got {elastic_fraction!r}"
        )
    unknown = sorted(set(fractions) - INELASTIC_SET)
    if unknown:
        raise ValueError(
            "draw_topology: unknown inelastic class label(s) "
            f"{unknown!r} outside the closed inelastic set "
            f"{sorted(INELASTIC_SET)}; 'nd_multi' is registered but "
            "unretained and 'elastic' is drawn only through the elastic "
            "fraction"
        )
    missing = sorted(set(INELASTIC_ORDER) - set(fractions))
    if missing:
        raise ValueError(
            "draw_topology: the inelastic class mapping is missing the "
            f"class(es) {missing!r}; every class of "
            f"{sorted(INELASTIC_ORDER)} must carry a fraction"
        )
    for label, value in fractions.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"draw_topology: class fraction {label!r} must be a "
                f"finite non-negative value, got {value!r}"
            )
    mass = e_fraction + sum(float(v) for v in fractions.values())
    if abs(mass - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"draw_topology: the total mass (elastic {e_fraction!r} plus "
            f"the inelastic sum {mass - e_fraction!r}) is {mass!r}, not "
            f"unit mass within {FRACTION_SUM_TOLERANCE!r}"
        )
    u = rng.random()
    if u < e_fraction:
        return "elastic"
    # The inelastic walk over the absolute cumulatives: the elastic
    # fraction opens the interval and each inelastic class carries
    # exactly its caller fraction of the unit interval (the
    # "scaled to the inelastic mass" allocation; the validated unit
    # mass makes the scaled and the absolute forms identical).
    cumulative = e_fraction
    for label in INELASTIC_ORDER:
        cumulative += float(fractions[label])
        if u < cumulative:
            return label
    # The float-dust tail of the walk: the last class closes the mass.
    return INELASTIC_ORDER[-1]


# ----------------------------------------------------------------------
# Dispatch to the landed class modules
# ----------------------------------------------------------------------
_REQUIRED_STREAMS = {
    "elastic": ("exchange_kinematics",),
    "sd_projectile": ("spectral_state", "exchange_kinematics",
                      "decay_channel", "decay_kinematics"),
    "sd_target": ("spectral_state", "exchange_kinematics",
                  "decay_channel", "decay_kinematics"),
    "dd": ("exchange_kinematics", "decay_channel", "decay_kinematics"),
    "nd": ("spectral_state", "exchange_kinematics", "decay_channel",
           "decay_kinematics"),
    "nd_multi": ("topology_selection", "spectral_state",
                 "exchange_kinematics", "decay_channel",
                 "decay_kinematics"),
}


def run_topology(
    class_label: str,
    params,
    catalog,
    fams,
    *,
    rng_topology_selection=None,
    rng_spectral_state=None,
    rng_exchange_kinematics=None,
    rng_decay_channel=None,
    rng_decay_kinematics=None,
    weights: Optional[Mapping[str, float]] = None,
    p_n2: Optional[float] = None,
    flavor=None,
    ledger=None,
):
    """Dispatch one event of class *class_label* to the landed class
    module (cards 1-7) and return the class record.

    *flavor* (Round 6, task 37) is the optional
    :class:`~na49_mixing.flavor_weights.FlavorParams` reaching every
    continuum-parent decay of the event; it defaults to the caller
    params' ``flavor`` field and ``None`` is the identity (the Round 5
    traces are reproduced bit for bit).
    *ledger* (Round 6 runs) is an optional list receiving one flavor
    ledger record per drawn continuum vertex of the event (see
    :func:`~na49_mixing.flavor_weights.vertex_ledger_record`); ``None``
    records nothing and no draw changes.

    The streams are the closed per-label substream generators (the
    committed cards 5-8 pattern; the card's sketch compresses them to a
    single ``rng``).  Each class consumes only its own labels: the
    elastic class exactly one uniform on ``exchange_kinematics``; SD its
    four-label order; DD its three-label order (no spectral draw); ND
    its four-label order; ``nd_multi`` the count draw on
    ``topology_selection`` before card 7's order.  A missing required
    stream raises :class:`ValueError` naming the class and the labels
    (checked, never repaired).

    *params* is a :class:`~na49_mixing.topology_mass_laws.MassLawParams`
    or a :class:`TopologyNominalParams` (whose ``epsilon``/``b_sd``
    build the same :class:`MassLawParams`; the class fractions and the
    remnant weights are the caller's draw arguments, fit in R51/R53, not
    here).  *weights* defaults to the caller params' remnant weights
    (the nominal intact 0.5 / low-mass 0.5 / high-mass 0.0 split) and
    *p_n2* to the caller's ``p_n2`` (the unretained card-8 default
    0.0); both reach the class modules unchanged.

    The isovector exchange (Round 8, 2026-10-04): at a positive
    ``isovector_fraction`` of a :class:`TopologyNominalParams` the runner
    draws the exchange configuration of an SD, DD or ND event with two
    uniforms on ``topology_selection`` (after any class draw of the
    caller; :func:`~na49_mixing.isovector_exchange.draw_exchange_configuration`)
    and passes it to the class module.  The elastic class keeps the vacuum
    exchange; ``nd_multi`` refuses a positive fraction.  At the default
    0.0 nothing is drawn.  ``b_isovector`` (when set) is the transfer slope
    of an isovector-drawn event: the pion and rho/a2 exchanges are more
    peripheral than the vacuum exchange, so their momentum transfer has its
    own slope; the mass exponents stay shared.  Vacuum events keep ``b_sd``.
    ``isovector_delta_ratio`` (2026-10-05) weights the I = 3/2 lines of the
    isovector exchange (``na49_mixing.isovector_exchange.isovector_configurations``).
    ``b_nd`` (2026-10-05) is the transfer slope of the non-diffractive
    lines (both ND variants); ``b_isovector`` takes precedence on an
    isovector-drawn ND event.
    """

    if class_label not in TOPOLOGY_CLASSES:
        raise ValueError(
            "run_topology: unknown class label "
            f"{class_label!r}; the registry carries exactly "
            f"{sorted(TOPOLOGY_CLASSES)}"
        )
    if isinstance(params, TopologyNominalParams):
        mass_params = MassLawParams(epsilon=params.epsilon, b_sd=params.b_sd)
        mass_params_nd = (mass_params if params.epsilon_nd is None else
                          MassLawParams(epsilon=float(params.epsilon_nd),
                                        b_sd=params.b_sd))
        if params.b_nd is not None:
            if not (math.isfinite(float(params.b_nd)) and float(params.b_nd) > 0.0):
                raise ValueError(f"run_topology: b_nd must be a positive slope, got {params.b_nd!r}")
            mass_params_nd = MassLawParams(epsilon=params.epsilon if params.epsilon_nd is None else float(params.epsilon_nd),
                                           b_sd=float(params.b_nd))
        default_weights = dict(params.remnant_weights)
        default_p_n2 = params.p_n2
        nd_central = bool(params.nd_central)
        if flavor is None:
            flavor = params.flavor
    else:
        mass_params = params
        mass_params_nd = params
        default_weights = None
        default_p_n2 = DEFAULT_P_N2
        nd_central = True
    if weights is None:
        weights = default_weights
    p_n2_value = default_p_n2 if p_n2 is None else float(p_n2)

    streams = {
        "topology_selection": rng_topology_selection,
        "spectral_state": rng_spectral_state,
        "exchange_kinematics": rng_exchange_kinematics,
        "decay_channel": rng_decay_channel,
        "decay_kinematics": rng_decay_kinematics,
    }
    missing = [label for label in _REQUIRED_STREAMS[class_label]
               if streams[label] is None]
    if missing:
        raise ValueError(
            f"run_topology: class {class_label!r} requires the closed "
            f"substream generator(s) {missing}, got none of them"
        )

    exchange = None
    exact = bool(params.exact_transfer_range) if isinstance(params, TopologyNominalParams) else False
    options = {}
    if exact:
        options["exact_range"] = True
    if isinstance(params, TopologyNominalParams) and params.line_thresholds:
        options["line_thresholds"] = True
    kappa_t = params.transfer_slope_mass_exponent if isinstance(params, TopologyNominalParams) else None
    if kappa_t is not None:
        # the factorized vertex slopes (2026-10-08): defined on the exact
        # transfer window of the 2 -> 2 classes
        if isinstance(kappa_t, bool) or not (math.isfinite(float(kappa_t)) and float(kappa_t) >= 0.0):
            raise ValueError(f"run_topology: transfer_slope_mass_exponent must be finite and non-negative, got {kappa_t!r}")
        if class_label in ("sd_projectile", "sd_target", "dd", "nd"):
            if not exact:
                raise ValueError("run_topology: transfer_slope_mass_exponent needs exact_transfer_range")
            options["slope_mass_exponent"] = float(kappa_t)
    m_dd = params.diffractive_duality_mass if isinstance(params, TopologyNominalParams) else None
    if m_dd is not None:
        # the diffractive lines through the I = 1/2 records (2026-10-15)
        if isinstance(m_dd, bool) or not (math.isfinite(float(m_dd)) and float(m_dd) > 1.08):
            raise ValueError(f"run_topology: diffractive_duality_mass must be a finite mass above the N pi threshold, got {m_dd!r}")
        if class_label in ("sd_projectile", "sd_target", "dd"):
            options["diffractive_duality"] = float(m_dd)
    shapes = isinstance(params, TopologyNominalParams) and bool(params.duality_line_shapes)
    if shapes:
        # the duality picks with the records' own line shapes (2026-10-18)
        if params.nd_duality_mass is None and m_dd is None:
            raise ValueError("run_topology: duality_line_shapes needs nd_duality_mass or diffractive_duality_mass")
        if (class_label == "nd" and params.nd_duality_mass is not None) or (
                class_label in ("sd_projectile", "sd_target", "dd") and m_dd is not None):
            options["duality_line_shapes"] = True
    if isinstance(params, TopologyNominalParams) and bool(params.duality_redraw_shapes):
        # the redraw below the shape mass on the records' line shapes (2026-10-20)
        if not shapes or params.nd_duality_shape_mass is None:
            raise ValueError("run_topology: duality_redraw_shapes needs duality_line_shapes and nd_duality_shape_mass")
        if class_label == "nd":
            options["duality_redraw_shapes"] = True
    if isinstance(params, TopologyNominalParams) and bool(params.nd_symmetric_remnants) and class_label == "nd":
        # the symmetric remnant pair of the non-diffractive class (2026-11-04)
        options["symmetric_remnants"] = True
    f_intact = params.intact_vertex_factor if isinstance(params, TopologyNominalParams) else None
    if f_intact is not None:
        # the intact proton's own vertex slope (2026-10-15): a term of the
        # factorized vertex slopes, read by the single-diffraction classes
        if isinstance(f_intact, bool) or not (math.isfinite(float(f_intact)) and float(f_intact) > 0.0):
            raise ValueError(f"run_topology: intact_vertex_factor must be a finite positive factor, got {f_intact!r}")
        if kappa_t is None:
            raise ValueError("run_topology: intact_vertex_factor needs transfer_slope_mass_exponent")
        if class_label in ("sd_projectile", "sd_target"):
            options["intact_vertex_factor"] = float(f_intact)
    xi_max = params.diffractive_xi_max if isinstance(params, TopologyNominalParams) else None
    if xi_max is not None:
        # the coherence region of diffraction (2026-10-23): the excited masses
        # of the diffractive classes below M^2/s = xi_max
        if isinstance(xi_max, bool) or not (math.isfinite(float(xi_max)) and 0.0 < float(xi_max) <= 0.25):
            raise ValueError(f"run_topology: diffractive_xi_max must be an M^2/s edge in (0, 0.25], got {xi_max!r}")
        if class_label in ("sd_projectile", "sd_target", "dd"):
            options["xi_max"] = float(xi_max)
    alpha_p = params.diffractive_pomeron_slope if isinstance(params, TopologyNominalParams) else None
    if alpha_p is not None:
        # the Pomeron's shrinkage in the diffractive classes (2026-11-01)
        if isinstance(alpha_p, bool) or not (math.isfinite(float(alpha_p)) and float(alpha_p) >= 0.0) or not exact:
            raise ValueError(f"run_topology: diffractive_pomeron_slope is a non-negative alpha' on the exact window, "
                             f"got {alpha_p!r}")
        if class_label in ("sd_projectile", "sd_target", "dd"):
            options["pomeron_slope"] = float(alpha_p)
    t_lin = params.diffractive_pomeron_linear_t if isinstance(params, TopologyNominalParams) else None
    if t_lin is not None:
        if alpha_p is None or isinstance(t_lin, bool) or not (math.isfinite(float(t_lin)) and float(t_lin) > 0.0):
            raise ValueError(f"run_topology: diffractive_pomeron_linear_t is a positive |t| of the Pomeron's shrinkage, got {t_lin!r}")
        if class_label in ("sd_projectile", "sd_target", "dd"):
            options["pomeron_linear_t"] = float(t_lin)
    sd_tail = params.sd_tail if isinstance(params, TopologyNominalParams) else None
    if sd_tail is not None:
        # the leading-proton tail of the single-diffraction classes (2026-10-23, R112)
        vals = tuple(float(v) for v in sd_tail)
        if (len(vals) not in (5, 7) or not all(math.isfinite(v) for v in vals) or not 0.0 <= vals[0] < 1.0
                or not 0.0 < vals[1] < vals[2] < 1.0 or vals[4] <= 0.0 or not exact
                or (len(vals) == 7 and not (vals[1] < vals[5] < vals[2] and vals[6] > 0.0))):
            raise ValueError(f"run_topology: sd_tail is (share, xi_lo, xi_hi, a, b[, xi_c, w]) on the exact window, "
                             f"got {sd_tail!r}")
        if class_label in ("sd_projectile", "sd_target"):
            options["tail_component"] = vals
    sd_ope = params.sd_ope if isinstance(params, TopologyNominalParams) else None
    if sd_ope is not None:
        # the one-pion exchange of the single-diffraction classes (2026-10-26); the form factor a slope r2 or "proton",
        # a sixth entry the pi+ p cross section's law, a seventh the exchanged pion ("isospin": with its neutral partner),
        # an eighth the absorptive correction ("eikonal", 2026-10-27)
        try:
            ope_component_of(sd_ope)
        except ValueError as exc:
            raise ValueError(f"run_topology: sd_ope is (share, xi_lo, xi_hi, alpha', r2 | 'proton'"
                             f"[, pion_proton[, pion[, absorption]]]), "
                             f"got {sd_ope!r}") from exc
        if not exact:
            raise ValueError(f"run_topology: sd_ope needs the exact window, got {sd_ope!r}")
        vals = tuple(v if isinstance(v, str) else float(v) for v in sd_ope)
        if class_label in ("sd_projectile", "sd_target"):
            options["ope_component"] = vals
    sd_two_sided = params.sd_two_sided if isinstance(params, TopologyNominalParams) else None
    if sd_two_sided is not None:
        # the two-sided hadronization of the heavy single-diffraction systems (2026-10-26)
        vals = tuple(sd_two_sided)
        if (len(vals) != 4 or not isinstance(vals[3], str) or not vals[3] or set(vals[3]) - set("otc")
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in vals[:3])
                or float(vals[0]) <= 0.0 or float(vals[2]) <= 0.0):
            raise ValueError(f"run_topology: sd_two_sided is (m_cut, eps_line, b, scope) with m_cut, b > 0 and the scope's "
                             f"letters from 'otc', got {sd_two_sided!r}")
        if class_label in ("sd_projectile", "sd_target"):
            options["two_sided"] = (float(vals[0]), float(vals[1]), float(vals[2]), str(vals[3]))
    t_break = params.intact_t_break if isinstance(params, TopologyNominalParams) else None
    if t_break is not None:
        # the intact proton's two-slope transfer (2026-10-24)
        vals = tuple(t_break)
        if (len(vals) != 2 or any(isinstance(v, bool) for v in vals) or not all(math.isfinite(float(v)) for v in vals)
                or not float(vals[0]) > 0.0 or not 0.0 < float(vals[1]) <= 1.0 or not exact or kappa_t is None):
            raise ValueError("run_topology: intact_t_break is (t_b, rho) with t_b > 0 and 0 < rho <= 1, on the exact "
                             f"window with the factorized vertex slopes, got {t_break!r}")
        if class_label in ("sd_projectile", "sd_target"):
            options["t_break"] = (float(vals[0]), float(vals[1]))
    if isinstance(params, TopologyNominalParams) and params.low_mass_spin_isospin:
        # the isoscalar exchange keeps the proton line at I = 1/2 (2026-10-08);
        # the double-diffraction class draws no remnant
        if class_label in ("sd_projectile", "sd_target", "nd"):
            options["low_mass_spin_isospin"] = True
    if isinstance(params, TopologyNominalParams) and (params.nd_duality_mass is not None
                                                      or params.nd_duality_shape_mass is not None):
        # the low-mass non-diffractive lines through the N* and Delta records (2026-10-12)
        m_d, m_s = params.nd_duality_mass, params.nd_duality_shape_mass
        for name, value in (("nd_duality_mass", m_d), ("nd_duality_shape_mass", m_s)):
            if value is not None and (isinstance(value, bool) or not (math.isfinite(float(value)) and float(value) > 1.08)):
                raise ValueError(f"run_topology: {name} must be a finite mass above the N pi threshold, got {value!r}")
        if m_d is None or (m_s is not None and float(m_s) > float(m_d)):
            raise ValueError(f"run_topology: nd_duality_shape_mass {m_s!r} needs nd_duality_mass at or above it, got {m_d!r}")
        if class_label == "nd":
            options["duality"] = (float(m_d), None if m_s is None else float(m_s))
    fraction = float(params.isovector_fraction) if isinstance(params, TopologyNominalParams) else 0.0
    if fraction != 0.0:
        if class_label == "nd_multi":
            raise ValueError("run_topology: the isovector exchange is not available for the nd_multi class")
        if class_label in ISOVECTOR_CLASSES:
            if streams["topology_selection"] is None:
                raise ValueError(
                    f"run_topology: class {class_label!r} at the isovector fraction {fraction!r} requires the "
                    "topology_selection substream generator (the exchange draw)")
            exchange = draw_exchange_configuration(fraction, streams["topology_selection"],
                                                   float(params.isovector_delta_ratio))
            b_iv = params.b_isovector
            if not exchange.is_vacuum and b_iv is not None:
                if not (math.isfinite(float(b_iv)) and float(b_iv) > 0.0):
                    raise ValueError(f"run_topology: b_isovector must be a positive slope, got {b_iv!r}")
                mass_params = MassLawParams(epsilon=params.epsilon, b_sd=float(b_iv))
                mass_params_nd = MassLawParams(epsilon=params.epsilon if params.epsilon_nd is None else float(params.epsilon_nd),
                                               b_sd=float(b_iv))

    if class_label == "elastic":
        return generate_elastic(streams["exchange_kinematics"])
    if class_label == "sd_projectile":
        return generate_sd(
            "projectile", mass_params, weights, catalog, fams,
            streams["spectral_state"], streams["exchange_kinematics"],
            streams["decay_channel"], streams["decay_kinematics"], flavor=flavor, ledger=ledger,
            exchange=exchange, **options)
    if class_label == "sd_target":
        return generate_sd(
            "target", mass_params, weights, catalog, fams,
            streams["spectral_state"], streams["exchange_kinematics"],
            streams["decay_channel"], streams["decay_kinematics"], flavor=flavor, ledger=ledger,
            exchange=exchange, **options)
    if class_label == "dd":
        return generate_dd(mass_params, catalog, fams,
                           streams["exchange_kinematics"],
                           streams["decay_channel"],
                           streams["decay_kinematics"], flavor=flavor, ledger=ledger,
                           exchange=exchange, **options)
    if class_label == "nd":
        if not nd_central:
            return generate_nd(mass_params_nd, weights, catalog, fams,
                               streams["spectral_state"],
                               streams["exchange_kinematics"],
                               streams["decay_channel"],
                               streams["decay_kinematics"], flavor=flavor, ledger=ledger,
                               central=False, exchange=exchange, **options)
        central_opts = ({"central_recoil": True}
                        if isinstance(params, TopologyNominalParams) and params.central_recoil else {})
        if isinstance(params, TopologyNominalParams) and params.epsilon_c is not None:
            central_opts["central_epsilon"] = float(params.epsilon_c)
        return generate_nd(mass_params_nd, weights, catalog, fams,
                           streams["spectral_state"],
                           streams["exchange_kinematics"],
                           streams["decay_channel"],
                           streams["decay_kinematics"], flavor=flavor, ledger=ledger,
                           exchange=exchange, **options, **central_opts)
    # nd_multi: registered but excluded from the default inelastic draw
    # (unretained, card 8); the label stays dispatchable for the R53
    # conditional extension.  Every nd_multi event carries one or two
    # central sources, which the exact transfer window refuses (no 2 -> 3
    # construction), and its generator has no per-line thresholds, so the
    # topology options are refused here rather than dropped (2026-10-07;
    # review 2026-10-06, Grok M1).
    if options:
        raise ValueError(f"run_topology: nd_multi does not take the topology options {sorted(options)} "
                         "(its central sources refuse the exact transfer window)")
    return generate_nd_multi(
        mass_params_nd, weights, p_n2_value, catalog, fams,
        streams["topology_selection"], streams["spectral_state"],
        streams["exchange_kinematics"], streams["decay_channel"],
        streams["decay_kinematics"], flavor=flavor, ledger=ledger)


# ----------------------------------------------------------------------
# The per-event observable extraction
# ----------------------------------------------------------------------
def _rapidity(four_momentum) -> float:
    """The rapidity of one four-momentum, computed in the frame the
    record is expressed in (``0.5 * ln((E + p_z) / (E - p_z))``; the
    caller passes the stored lab four-momenta or the CM-boosted
    four-momenta of the Feynman-x and central-window quantities).  The
    committed final states are massive on-shell records, so
    ``E > |p_z|`` holds on every record."""

    e, _px, _py, pz = (float(x) for x in four_momentum)
    if not e > abs(pz):
        raise ValueError(
            "extract_observables: the record carries a non-physical "
            f"four-momentum (E = {e!r}, p_z = {pz!r}); no rapidity exists"
        )
    return 0.5 * math.log((e + pz) / (e - pz))


def _transverse_momentum(four_momentum) -> float:
    """The transverse momentum of one four-momentum [GeV/c]."""

    _e, px, py, _pz = (float(x) for x in four_momentum)
    return math.sqrt(px * px + py * py)


def _cm_boost_parameters(sqrt_s: float) -> tuple[float, float]:
    """The lab to overall-CM boost parameters ``(beta, gamma)`` of the
    pinned M8 p+p beam configuration at the caller's *sqrt_s* [GeV]:
    ``beta = p_beam / (E_beam + m_p)`` along the beam axis (the exact
    fixed-target CM velocity) and ``gamma = (E_beam + m_p) / sqrt_s``
    with the same *sqrt_s* the Feynman-x denominator uses, so the
    extraction treats the recorded collision energy consistently as
    the CM energy of the boost and of ``x_F = 2 p_L^CM / sqrt_s``.

    Raises :class:`ValueError` on a non-finite or non-positive
    *sqrt_s* (checked, never repaired).
    """

    root_s = float(sqrt_s)
    if not math.isfinite(root_s) or root_s <= 0.0:
        raise ValueError(
            "extract_observables: sqrt_s must be a finite positive GeV "
            f"value, got {sqrt_s!r}"
        )
    e_beam = math.sqrt(PROTON_MASS * PROTON_MASS
                       + BEAM_MOMENTUM * BEAM_MOMENTUM)
    e_total = e_beam + PROTON_MASS
    beta = BEAM_MOMENTUM / e_total
    gamma = e_total / root_s
    return beta, gamma


def _cm_four_momentum(four_momentum, beta: float, gamma: float):
    """The inverse (lab to CM) longitudinal boost of one four-momentum,
    the class modules' forward fixed-target boost applied in reverse:
    ``(E*, px, py, pz*) = (gamma (E - beta pz), px, py, gamma (pz -
    beta E))``.  The transverse components are invariant."""

    e, px, py, pz = (float(x) for x in four_momentum)
    return (gamma * (e - beta * pz), px, py, gamma * (pz - beta * e))


def _final_state_of(event):
    """The event's final-state hadrons: the recorded ``final_state``
    on the inelastic classes; the elastic event's two outgoing lines
    (the intact-pair pseudostate of the normalization reference)."""

    if isinstance(event, ElasticEvent):
        return tuple(event.outgoing)
    return tuple(event.final_state)


def _baryonic_lines(event):
    """The two outgoing baryonic line states (projectile side first):
    the elastic event's outgoing pair, otherwise the record's
    ``lines`` (every landed class orders the tuple projectile first)."""

    if isinstance(event, ElasticEvent):
        return tuple(event.outgoing)
    return tuple(event.lines)


def extract_observables(
    event,
    sqrt_s: float = SQRT_S,
) -> TopologyObservables:
    """The per-event observable extraction (the Objective's list) from
    one landed class record.

    The Feynman x and the rapidity regions live in the overall CM frame
    of the pinned beam configuration: every landed class record stores
    lab-frame four-momenta (the class modules' forward fixed-target
    boost), and the extraction applies the inverse boost
    :func:`_cm_four_momentum` with the :func:`_cm_boost_parameters`
    ``(beta, gamma)`` at the caller's *sqrt_s* before forming
    ``x_F = 2 p_L^CM / sqrt_s`` and the central window.  The transverse
    momentum is invariant under the longitudinal boost and is read from
    the stored lab four-momenta; the excitation masses, the momentum
    transfers, and the gap size (a rapidity difference) are
    frame-independent records.

    The rapidity regions: central ``|y^CM| < 0.5`` (:data:`CENTRAL_WINDOW`,
    the R53 trigger window), forward otherwise.  Pions are ``pdg_id``
    in :data:`PION_IDS`; protons the ``pdg_id == 2212`` records.  The
    seagull moment is the arithmetic mean transverse momentum of the
    final state (the plan section 7.3 m_T-slope moment building block);
    the elastic pair is the normalization reference's final state, so
    the multiplicity never vanishes on any landed class.
    """

    beta_cm, gamma_cm = _cm_boost_parameters(sqrt_s)

    def _cm(four_momentum):
        return _cm_four_momentum(four_momentum, beta_cm, gamma_cm)

    lines = _baryonic_lines(event)
    if len(lines) != 2:
        raise ValueError(
            "extract_observables: expected two baryonic lines, got "
            f"{len(lines)}"
        )
    x_f_projectile = 2.0 * _cm(lines[0].four_momentum)[3] / sqrt_s
    x_f_target = 2.0 * _cm(lines[1].four_momentum)[3] / sqrt_s

    if isinstance(event, ElasticEvent):
        excitation_masses: tuple[float, ...] = ()
        transfers = (float(event.t),)
        gap_size = None
    elif isinstance(event, SDEvent):
        excitation_masses = (float(event.excitation_mass),)
        transfers = (float(event.t),)
        gap_size = float(event.gap_size)
    elif isinstance(event, DDEvent):
        excitation_masses = (float(event.m1), float(event.m2))
        transfers = (float(event.t1), float(event.t2))
        gap_size = float(event.gap_size)
    elif isinstance(event, NDEvent):
        excitation_masses = (float(event.central_mass),) if event.central_component else ()
        transfers = (float(event.t1), float(event.t2), float(event.t_c))
        gap_size = None
    elif isinstance(event, NDMultiEvent):
        excitation_masses = tuple(float(m) for m in event.central_masses)
        transfers = (float(event.t1), float(event.t2), float(event.t_2c),
                     float(event.t_c))
        gap_size = None
    else:
        raise ValueError(
            "extract_observables: unsupported event record "
            f"{type(event)!r}; the runner dispatches exactly the five "
            "landed record types"
        )

    final = _final_state_of(event)
    central_pions = 0
    central_protons = 0
    forward_pions = 0
    forward_protons = 0
    multiplicity_central = 0
    p_t_sum = 0.0
    for hadron in final:
        y = _rapidity(_cm(hadron.four_momentum))
        in_central = abs(y) < CENTRAL_WINDOW
        if in_central:
            multiplicity_central += 1
        if isinstance(hadron, FinalHadron) and hadron.pdg_id is not None:
            is_pion = int(hadron.pdg_id) in PION_IDS
            is_proton = int(hadron.pdg_id) == PROTON_PDG_ID
            if in_central:
                if is_pion:
                    central_pions += 1
                if is_proton:
                    central_protons += 1
            else:
                if is_pion:
                    forward_pions += 1
                if is_proton:
                    forward_protons += 1
        # The elastic event's pseudostate lines carry no catalog
        # identity (the LineState record); the species yields count
        # only the recorded final-state hadrons.
        p_t_sum += _transverse_momentum(hadron.four_momentum)
    multiplicity_total = len(final)
    seagull_moment = (p_t_sum / multiplicity_total
                      if multiplicity_total else 0.0)

    return TopologyObservables(
        class_label=event.class_label,
        x_f_projectile=x_f_projectile,
        x_f_target=x_f_target,
        excitation_masses=excitation_masses,
        transfers=transfers,
        gap_size=gap_size,
        multiplicity_total=multiplicity_total,
        multiplicity_central=multiplicity_central,
        central_pions=central_pions,
        central_protons=central_protons,
        forward_pions=forward_pions,
        forward_protons=forward_protons,
        seagull_moment=seagull_moment,
    )


# ----------------------------------------------------------------------
# The pinned hermetic trace (the T0-equivalent)
# ----------------------------------------------------------------------
_CATALOG_CACHE = None
_FAMS_CACHE = None


def _committed_surfaces():
    """The committed catalog and continuum families, loaded once from
    the package tree (the run cards supply their own configuration; the
    hermetic trace uses the committed surfaces)."""

    global _CATALOG_CACHE, _FAMS_CACHE
    if _CATALOG_CACHE is None:
        from na49_mixing.hadron_catalog import HadronCatalog
        from na49_mixing.spectral_density import families_from_json
        nested_root = Path(__file__).resolve().parent.parent
        _CATALOG_CACHE = HadronCatalog.from_json(
            nested_root / "config" / "v2_singlet" / "resonance_catalog.json")
        _FAMS_CACHE = families_from_json(
            str(nested_root / "config" / "v2_singlet"
                / "continuum_resonances.json"))
    return _CATALOG_CACHE, _FAMS_CACHE


def pinned_trace(
    params: Optional[TopologyNominalParams] = None,
    seed: int = MASTER_SEED,
    n_events: int = 1000,
    *,
    catalog=None,
    fams=None,
    elastic_fraction: Optional[float] = None,
) -> tuple:
    """The seeded deterministic event trace at the given parameters
    (the hermetic T0-equivalent pinned trace; the T1 generation and the
    R50 statistics belong to run card 10).

    One topology draw (exactly one uniform on the closed
    ``topology_selection`` substream) plus the class generation per
    event index ``i = 0 .. n_events - 1`` on the Round 5 closed
    five-label substream quintuple at replica 0; the elastic fraction
    defaults to the complement of the inelastic mass (exactly 0.0 at
    the nominal point, so no draw returns ``"elastic"``).  Returns the
    tuple of per-event :class:`TopologyObservables`.
    """

    if params is None:
        params = NOMINAL_PARAMS
    if not isinstance(params, TopologyNominalParams):
        raise ValueError(
            "pinned_trace: params must be a TopologyNominalParams (the "
            f"fixed R50 nominal structure), got {type(params)!r}"
        )
    if catalog is None or fams is None:
        committed_catalog, committed_fams = _committed_surfaces()
        catalog = committed_catalog if catalog is None else catalog
        fams = committed_fams if fams is None else fams
    if elastic_fraction is None:
        elastic_fraction = 1.0 - sum(
            float(v) for v in params.class_fractions.values())
    observables = []
    for event_index in range(int(n_events)):
        subs = derive_substreams(int(seed), 0, event_index)
        rng_topology_selection = np.random.default_rng(
            subs.stream_for("topology_selection"))
        rng_spectral_state = np.random.default_rng(
            subs.stream_for("spectral_state"))
        rng_exchange_kinematics = np.random.default_rng(
            subs.stream_for("exchange_kinematics"))
        rng_decay_channel = np.random.default_rng(
            subs.stream_for("decay_channel"))
        rng_decay_kinematics = np.random.default_rng(
            subs.stream_for("decay_kinematics"))
        label = draw_topology(
            params.class_fractions, elastic_fraction,
            rng_topology_selection)
        event = run_topology(
            label, params, catalog, fams,
            rng_topology_selection=rng_topology_selection,
            rng_spectral_state=rng_spectral_state,
            rng_exchange_kinematics=rng_exchange_kinematics,
            rng_decay_channel=rng_decay_channel,
            rng_decay_kinematics=rng_decay_kinematics,
            p_n2=params.p_n2)
        observables.append(extract_observables(event, SQRT_S))
    return tuple(observables)
