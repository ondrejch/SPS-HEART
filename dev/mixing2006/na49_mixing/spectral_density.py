"""Continuum color-singlet spectral families.

Round 3, card 4 (``TASK-20260917-r3-baryonic-continuum-spectral-family``;
plan section 22 task 18, plan sections 9.1-9.2; module name of plan
section 4.3, "continuum color-singlet excitation spectra"), extended
additively by card 5 (``TASK-20260917-r3-mesonic-continuum-spectral-
family``; plan section 22 task 19) and by card 6 (``TASK-20260917-r3-
smooth-catalog-continuum-matching``; plan section 22 task 20, plan
section 9.3).  The module carries the continuum
families of the plan section 9.1 state list as pre-registered spectral
records: each family is a probability density of states over its mass
support, with a minimum mass, a smoothstep threshold ramp, one shared
continuum scale per sector, and the allowed hadronic decay classes.
Card 4 landed the two baryonic families; card 5 lands the mesonic
families, the activated neutral family ``(0, 0, 0)`` and the four
declared, inactive charged and strange families; card 6 lands the
smooth catalog/continuum matching on top of them: the fixed blend
weight ``w(M)``, the catalog and continuum mixture densities, the
blended density with its double-counting identity, and the per-source
decay routing that realizes the blend at the frozen source mass.

Spectral density (plan section 9.2, initial model; round 3 index items
2-4, minimal choice)::

    rho_a(M) = N_a * M**(-b_a) * exp(M / T_H) * S_a(M),

with ``b_a = 0``, ``T_H = 165 MeV`` (the common baryonic continuum scale,
one value shared by both baryonic families; the mesonic scale of card 5
is a distinct named value), and ``N_a`` fixed by unit normalization of
each family over its support on the reference quadrature (plan section
8.2, "Use normalized spectral functions": the families are probability
densities).  ``S_a(M)`` is the smoothstep ramp of width ``W = 0.5 GeV``::

    S_a(M) = 0                                  M <= m_min
    S_a(M) = s^2 (3 - 2 s),  s = (M - m_min)/W  m_min < M < m_min + W
    S_a(M) = 1                                  M >= m_min + W

which is continuous at both junctions with exact zero one-sided
derivatives there (the polynomial ``s^2 (3 - 2 s)`` has derivative
``6 s (1 - s)``), so the density is C^1 at ``m_min`` and at
``m_min + W``.  There is no continuum strength at or below ``m_min``:
the density returns exactly ``0.0`` there (plan section 9.3; index item
3, ``m_min = B = 2.680194110727528 GeV``, the M8-frozen
``continuum.match_mass``; the catalog owns the region below the
boundary).

The mesonic scale (card 5) is the distinct named value of the mesonic
sector: ``T_H = 165 MeV``, the one common mesonic continuum scale fixed
and shared by every activated mesonic family (plan section 9.2: one
common mesonic continuum scale; no separate fitted T_H per sector).  The
baryonic and mesonic scales are separately named so a later
re-registration of one never silently moves the other; both are
pre-registered to the same conventional light-hadron Hagedorn value
(round 3 index item 2).

Active and declared families (card 5, round 3 index item 9): every
family record carries an ``active`` flag and sampling respects it.
:func:`active_families` returns the families of a given ``(B, Q, S)``
that are active, and :func:`family_draw` raises ``ValueError`` naming
any family drawn while inactive.  Card 5 activates exactly the neutral
mesonic family ``(0, 0, 0)``; the charged ``(0, +-1, 0)`` rho-like and
strange ``(0, 0, +-1)`` K-like mesonic families are declared with the
same parameter form and flagged ``active = False``.  They are never
sampled in Round 3 (plan section 9.1: charged or strange mesonic
continuum states only when created by an explicit non-vacuum exchange or
an allowed parent decay; the Round 3 p+p topology is the neutral
two-source exchange, so these families enter the final state only as
decay daughters of catalog states, which the Round 2 kernel handles).

Support and quadrature: the support of every family is
``[m_min, m_max]`` with ``m_max = SUPPORT_M_MAX = 5.0 GeV`` (pre-
registered, recorded per family in the config).  Normalization and the
inverse-CDF draw both use the fixed 10,001-node trapezoid grid over the
support (:data:`REFERENCE_GRID_POINTS`, the Round 2 reference quadrature,
``SPECTRAL_QUADRATURE_NODES`` of ``na49_mixing.resonance_decay``; the
equality is pinned in the card test).  Normalizing on the same fixed
quadrature that measures the report integral makes the unit
normalization grid-convergent by construction, the resolution pattern of
the card-8 check round for the resonance line shapes.

Draw convention: :func:`family_draw` inverts the CDF of the density on
the reference grid and consumes exactly two uniform draws per call
(segment choice plus within-cell position), the ``draw_mass`` convention
of ``na49_mixing.resonance_widths`` (Round 2 card 3).  The drawn mass is
uniform within the selected grid cell of the piecewise-linear CDF
interpolant; no rejection loop is involved and the per-draw uniform
count is stable.

Width model: the family-level constant nominal width
``nominal_width = 0.5 GeV`` is documented metadata of the family record
(the pre-registered minimal choice; the plan fixes no continuum width
law, and the mass-dependent width law is the established-resonance
machinery of plan section 8.2).  It does not alter the spectral
density: the continuum is a density of states, not a line shape.

Allowed hadronic decay classes (plan section 9.1; round 3 index item
10): the catalog multiplet classes open at the source mass that conserve
the family ``(B, Q, S)`` at the vertex, declared as identifiers (the
catalog's multiplet-class keys, the ``(daughter[0], daughter[1])``
flavor pairs of ``na49_mixing.resonance_decay``).  The decay itself runs
through the Round 2 recursive catalog kernel
(``ResonanceDecayer`` of ``na49_mixing.resonance_decay``), which this
module does not modify.  For the baryonic family ``(1, 1, 0)`` the
card-named classes resolve against the committed catalog to the N-pi,
Delta-pi (the N-pi-pi cascade class), K-Lambda, and K-Sigma charge states
that conserve ``(1, 1, 0)``.  For the strange baryonic family
``(1, 0, -1)`` the same resolution applies: no charge state of the
S = 0 class families (K-Lambda, K-Sigma, N-pi) conserves ``S = -1`` at
the vertex, so the allowed set resolves to the strangeness-conserving
catalog classes of the established S = -1 baryon spectrum -- the
Lambda-pi and Sigma-pi classes (the strangeness on the baryon daughter)
and the N-anti-K class (the strangeness on the anti-kaon daughter, the
associated-production partner); all six are recorded channels of the
Lambda(1520) and Sigma(1385) catalog records.  For the mesonic families
(card 5) the same resolution applies: the allowed set of every mesonic
family is the committed catalog's recorded two-body multiplet classes
whose daughter quantum numbers sum to the family ``(B, Q, S)`` and whose
daughter pole-mass sum lies below ``m_min`` (open across the family
support), with the charge-conjugate partners the catalog records.  For
the activated neutral family ``(0, 0, 0)`` this resolves to the card-
named set: the pi-pi classes ``(211, -211)`` and ``(111, 111)`` (the
two-pion system of total isospin I = 0, 1, 2), the K-Kbar classes
``(321, -321)`` and ``(310, 130)`` (the neutral ``(K0, Kbar0)`` flavor
pair in its weak-eigenstate recorded form, the card-2 ``_concrete_pair``
convention), and the scalar/tensor and other neutral meson classes
rho-pi, omega-pi, eta-pi, and eta-eta.  The declared charged and strange
mesonic families carry the analog conserving classes resolved by the
same rule; they are never sampled in Round 3.  The per-class quantum-
number audit against the committed catalog lives in the card tests
``tests/test_spectral_density.py`` (baryonic) and
``tests/test_spectral_density_mesonic.py`` (mesonic).

Color-singlet routing: every family record in
``config/v2_singlet/continuum_resonances.json`` carries
``state_type = "continuum_resonance"`` (the Round 1 card-4 stamp of
``na49_mixing.state_validation``), and no module or configuration name
implies strings, parton splitting, or constituent breakup (plan section
4.3, line 220).

Hermetic: standard library, numpy, and the committed config/catalog
files only; no event generation, no external tables, no fitted values
(the pre-registered parameters come from the card and the round 3 index
items 2-4, and ``N_a`` is a derived normalization).

Smooth catalog/continuum matching (card 6; plan section 22 task 20,
plan section 9.3; round 3 index item 4)
----------------------------------------------------------------

The plan section 9.3 blend replaces the hard catalog/continuum switch
at the catalog-completeness boundary with a smooth matching of
normalized densities::

    rho_total(M) = [1 - w(M)] * rho_catalog(M) + w(M) * rho_cont(M),

with the fixed pre-registered ramp weight :class:`BlendSpec.weight`::

    w(M) = 0             M <= B
    w(M) = (M - B) / W   B < M < B + W
    w(M) = 1             M >= B + W,

``B = BLEND_BOUNDARY = 2.680194110727528 GeV`` (the M8-frozen
``continuum.match_mass``) and ``W = BLEND_RAMP_WIDTH = 0.5 GeV``.  The
ramp is hard-coded (fixed initially, varied only as a systematic, plan
section 9.3): the R32 variants are run-level parameter choices, and no
M8 parameter is touched.

The two blended densities for a source ``(B, Q, S)``:

- :func:`catalog_density` is the sum over the committed catalog
  records' normalized relativistic-Breit-Wigner spectral functions (the
  Round 2 card-3 surface, each record normalized to unit integral over
  its support), restricted to the records whose quantum numbers match
  the source; a stable record (degenerate support) carries no
  normalized line shape and contributes exactly ``0.0``.
- :func:`continuum_density` is the sum over the activated continuum
  families of cards 4-5 with the matching ``(B, Q, S)`` (the declared,
  inactive families are never summed; the card-5 active flag controls
  the mixture).

Blend normalization (the double-counting identity).  The continuum
side is pinned: every family is unit-normalized over its support
(the cards 4-5 contract, verified at load), so the routed continuum
carries ``int_{M>=B} w * rho_cont = 1 - G`` with
``G = int_{B}^{B+W} (1 - w) * rho_cont`` the ramp weight the blend
takes back out of the continuum inside the transition band.  "No
strength is lost or doubled at the boundary" (plan section 9.3) then
fixes the one free normalization of the blend, the overall strength of
the catalog mixture: the catalog enters the blend scaled by the
boundary-matching factor ``lambda`` chosen so that

    lambda * int (1 - w) * rho_catalog_raw  =  1 - int(rho_cont) + G,

i.e. the strength the blend actually takes from the catalog (all of it
below the boundary plus the ``1 - w`` share inside the transition
band) exactly complements the routed continuum strength, and

    int(rho_total) = 1    and    int_{M<B} rho_catalog + int_{M>=B}
    rho_total = 1

hold to quadrature accuracy (``NORMALIZATION_TOLERANCE``) for every
activated source QN set; both are measured by
:func:`blended_normalization_report` on the fixed 10,001-node grid
over the union support ``[m_min, m_max]`` (the lowest matching record
threshold through ``SUPPORT_M_MAX``).  The factor fixes the strength
only: the card's catalog-density definition (the sum of the unit
record line shapes) fixes the shape, the catalog records and the
family records are untouched, and the routed Monte Carlo below never
consults the factor (the routing weight is ``w(M)`` alone), so the
factor cannot drift any frozen draw.

Decay routing (:func:`route_source_decay`; the plan is silent on the
interaction with the frozen M8 source-mass law; round 3 index item 4):
the legacy source-mass draw stays bit-identical to the R23 builder at
the same seed, and the blend acts as a per-source decay-routing weight
evaluated at the frozen source mass ``M``:

- ``M < B``: the catalog mixture, the R23 builder convention.  The
  routing consumes zero uniforms, so the caller's catalog-path draw
  sequence (the spawned ``decay_channel`` child's record draw, the
  card-3 mass draw of the record support truncated at the boundary on
  the spawned ``decay_kinematics`` child, the Round 2 kernel) is
  byte-identical to the no-blend convention at the same seed.
- ``B <= M < B + W``: route to the continuum family mixture with
  probability ``w(M)`` and to the catalog mixture with probability
  ``1 - w(M)``; the routing draw is a single uniform consumed before
  any record draw, so the lineage is traceable from the returned
  route.
- ``M >= B + W``: the continuum family mixture; zero routing uniforms.

The routing result is returned (never swallowed), so the caller stamps
the source lineage: a ``'catalog'`` route keeps the catalog-mixture
lineage (the drawn record's established-resonance identity) and a
``'continuum'`` route stamps ``continuum_resonance`` (the Round 1
card-4 stamp).  For a fixed-mass grid (R31, card 8) the rule applies
per grid node.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np

from .pdg_database import get_particle
from .resonance_widths import spectral_function

__all__ = [
    "ContinuumFamily",
    "BlendSpec",
    "REFERENCE_GRID_POINTS",
    "SUPPORT_M_MAX",
    "CONTINUUM_STATE_TYPE",
    "NORMALIZATION_TOLERANCE",
    "BLEND_BOUNDARY",
    "BLEND_RAMP_WIDTH",
    "active_families",
    "family_density",
    "family_draw",
    "family_normalization_report",
    "families_from_json",
    "catalog_density",
    "continuum_density",
    "BLEND_RAMP_SHAPES",
    "blended_density",
    "route_source_decay",
    "blended_normalization_report",
]


#: Node count of the fixed reference quadrature over the family support:
#: the Round 2 reference quadrature (``SPECTRAL_QUADRATURE_NODES`` of
#: ``na49_mixing.resonance_decay``; the equality is pinned in the card
#: test).  The same grid defines the unit normalization and the
#: inverse-CDF draw table.
REFERENCE_GRID_POINTS = 10_001

#: Upper end of the family mass support, GeV (pre-registered; recorded
#: per family in the config as ``m_max`` and required to equal this
#: constant at load time).
SUPPORT_M_MAX = 5.0

#: The ``state_type`` stamp of every continuum family record (the Round 1
#: card-4 label of ``na49_mixing.state_validation``).
CONTINUUM_STATE_TYPE = "continuum_resonance"

#: The Round 2 spectral-normalization tolerance of the report integral.
NORMALIZATION_TOLERANCE = 1e-6

#: The realized-mass laws of a continuation daughter (the optional
#: ``continuation_mass_law`` config key; 2026-10-03): ``"edge"`` is the
#: card-4 rule, ``"density"`` the density-weighted draw.
CONTINUATION_MASS_LAWS = ("edge", "density")

#: The weights of the emitted sibling at a continuation step (2026-10-06,
#: the code and physics review's M-2): ``"edge"`` is the card-4 weight q* at
#: the fixed mass ``max(floor, edge - nominal_width)``; ``"density"`` is the
#: phase space integrated over the continuation daughter's own law,
#: ``int rho_fam(m) q*^(2L+1) F_L(q*)^2 dm`` on the on-shell window with
#: ``rho_fam`` normalized on the family support ``[floor, ceiling]``.
CONTINUATION_SIBLING_WEIGHTS = ("edge", "density")

#: The daughter-family rules of a continuation step (2026-10-06): ``"first"``
#: takes the first family of the daughter's (B, Q, S) in config order (the
#: committed cascade); ``"isospin"`` enumerates every family of that (B, Q, S),
#: each with its own Clebsch-Gordan share (the isovector mesonic completion,
#: step 3).
CONTINUATION_DAUGHTER_RULES = ("first", "isospin")


@dataclasses.dataclass(frozen=True)
class ContinuumFamily:
    """One continuum color-singlet spectral family (plan section 9.1).

    The record is a frozen value object; ``n_norm`` is the derived unit
    normalization over the support on the reference quadrature (the
    config pins the recorded value and the loader verifies it against
    the recomputation), and ``provenance`` carries one rendered entry
    per config field (the Round 2 catalog provenance discipline).
    """

    family_id: str            # "baryonic_np" / "mesonic_neutral" / ...
    qn: tuple[int, int, int]  # (B, Q, S)
    m_min: float              # GeV; 2.680194110727528 for every family
    t_h: float                # GeV; 165e-3 for this card
    b_power: float            # 0.0 for this card
    n_norm: float             # fixed by unit normalization
    ramp_width: float         # GeV; 0.5 for this card
    nominal_width: float      # GeV; 0.5 for this card
    allowed_decay_classes: tuple[str, ...]
    active: bool              # True for the activated families only
    provenance: tuple[str, ...]
    # GeV; the lowest realized mass of a continuation daughter of this
    # family (Round 5 follow-up Block B stage 2). ``None`` (the absent
    # config key) is ``m_min``. The source density, its ramp and n_norm
    # keep ``m_min``.
    continuation_floor: float | None = None
    # the family's isospin under the config's ``isospin_rule``; ``None``
    # (no rule) leaves the continuum vertices without isospin factors
    isospin: float | None = None
    # GeV; the highest realized mass of a continuation daughter of this
    # family (2026-10-02). ``None`` (the absent config key) is the support
    # edge ``SUPPORT_M_MAX``, the card-4 rule bit for bit. A heavier
    # ceiling lets a state above the support edge cascade step by step
    # instead of shedding its excess into one hadron; the source density,
    # its support and n_norm keep ``SUPPORT_M_MAX``.
    continuation_ceiling: float | None = None
    # the realized mass of a continuation daughter of this family
    # (2026-10-03): ``None`` (the absent config key) or ``"edge"`` is the
    # card-4 rule bit for bit, max(floor, edge - nominal_width), which
    # releases the fixed kinetic energy nominal_width at every step;
    # ``"density"`` draws the mass on [floor, edge] from the family's
    # declared tower shape M**(-b_power) * exp(M / t_h) times the two-body
    # decay momentum (one uniform on the decay_kinematics substream). The
    # source density, its ramp and n_norm are unchanged.
    continuation_mass_law: str | None = None
    # the orbital partial wave of the density-weighted continuation step
    # (2026-10-03, after R73): ``None`` (the absent config key) or 0 keeps
    # the R73 law bit for bit (the weight q*); L >= 1 weights the daughter
    # mass with q***(2L+1) * F_L(q*)**2, the Blatt-Weisskopf barrier of the
    # catalog vertices. Read only by the density law.
    continuation_barrier_l: int | None = None
    # the weight of the emitted sibling at a continuation step of a
    # daughter of this family (2026-10-06): ``None`` (the absent config key)
    # or ``"edge"`` is the card-4 weight bit for bit; ``"density"`` integrates
    # the daughter's density law (CONTINUATION_SIBLING_WEIGHTS; needs the
    # density mass law).
    continuation_sibling_weight: str | None = None
    # the daughter-family rule of a continuation step of a state of this
    # family (2026-10-06): ``None`` (the absent config key) or ``"first"`` is
    # the committed first-family rule bit for bit; ``"isospin"`` enumerates
    # every family of the daughter's (B, Q, S) with its isospin share
    # (CONTINUATION_DAUGHTER_RULES).
    continuation_daughters: str | None = None
    # the continuation scale of a daughter of this family (2026-10-07; card
    # TASK-20261005-hadron-count-vs-mass): ``None`` (the absent config key) is
    # the committed t_h bit for bit; a value alpha sets the density step's
    # scale to max(t_h, alpha * M_parent), in the daughter-mass draw and in the
    # density-integrated sibling weight alike (calibrated on the MARK I e+e-
    # multiplicities and the Goulianos-Sticker-White diffractive law).
    continuation_t_h_alpha: float | None = None
    # the leading-baryon split of a baryonic continuum root of this family
    # (2026-10-08; a run option of na49_mixing.kernel_config, never set by the
    # committed config): ``None`` is the committed cascade bit for bit; a
    # fraction F sends that share of the root decays (depth 0, B = 1) through
    # the split channels only, the established baryon and a mesonic
    # continuum daughter of mass mu, mu^2 drawn from (mu^2)^(-1-epsilon) on
    # [floor^2, edge^2] (leading_split_epsilon, default 0), the baryon
    # directed along the incoming baryon with exp(kappa cos theta)
    # (leading_split_kappa, or kappa = 2 b q*^2 for the transfer slope
    # b = leading_split_slope).
    leading_split_fraction: float | None = None
    leading_split_kappa: float | None = None
    leading_split_slope: float | None = None
    leading_split_epsilon: float | None = None
    # the strange established baryons in the split (2026-10-28): ``None`` or
    # ``True`` admits every established baryon (the committed split);
    # ``False`` takes the hyperons out of the split channels with their
    # share of the split's budget (the non-strange leading baryons as before)
    leading_split_strange: bool | None = None
    # the Regge-flux remainder law of the split (2026-10-28): ``True`` sets
    # the remainder's epsilon per channel from the exchanged trajectory
    # (cascade_decay.regge_split_epsilon); ``None`` or ``False`` keeps the
    # committed law
    leading_split_regge: bool | None = None
    # the leading-meson split of a root of the leading-baryon split
    # (2026-10-11; a run option of na49_mixing.kernel_config, never set by
    # the committed config): ``None`` is the committed cascade bit for bit; a
    # fraction F sends that share of the root decays (one uniform on the
    # decay_channel substream, before the channel uniform) through the
    # continuation channels of an established meson and a baryonic continuum
    # daughter, the remainder's mass from the split's law and the meson
    # directed along the incoming baryon by the split's direction law
    # (leading_split_kappa or leading_split_slope, or kappa = 2 b q*^2 for
    # b = meson_split_slope).
    meson_split_fraction: float | None = None
    meson_split_slope: float | None = None
    # the string-like continuation step of a heavy vertex of this family
    # (2026-10-27; a run option of na49_mixing.kernel_config, never set by the
    # committed config): ``None`` is the committed cascade bit for bit; a mass
    # M_cut [GeV] makes a vertex above it emit one established hadron at a
    # time (the draw over its continuation channels, the direct pairs where
    # the continuation has closed), draw the daughter's mass from
    # (M'^2)^(-1-eps) on [floor^2, edge^2]
    # (string_step_epsilon, default 0: flat in ln M'^2, the cluster law of the
    # two-sided split) in place of the density step, and direct the line's
    # carrier along the line's hemisphere and the sibling toward the center
    # with exp(kappa cos theta), kappa = 2 b q*^2 (string_step_slope b
    # [GeV^-2], default 4, the leading-meson split's law), on the same
    # uniforms: the sibling's p_T follows exp(-b p_T^2) at every vertex mass
    # and the count grows with the logarithm of the mass.
    string_step_mass: float | None = None
    string_step_slope: float | None = None
    string_step_epsilon: float | None = None
    # the floor pool of this family (2026-10-09; a run option of
    # na49_mixing.kernel_config through na49_mixing.floor_pools, never set
    # by the committed config): the further direct pairs of a vertex with no
    # open continuation channel, weighted as the allowed decay classes'
    # direct pairs; the continuation steps keep the allowed decay classes.
    # Empty is the committed cascade bit for bit.
    floor_decay_classes: tuple[str, ...] = ()
    # the baryon-resonance pairs of this family (2026-10-10; a run option of
    # na49_mixing.kernel_config through na49_mixing.resonance_pairs, never
    # set by the committed config): further baryon-antibaryon classes
    # offered with the allowed decay classes, the baryon-pair share of the
    # continuation channels and of the direct pairs each held at its value
    # without them.  Empty is the committed cascade bit for bit.
    resonance_pair_classes: tuple[str, ...] = ()
    # the minimum decay momentum of a continuation step into this family
    # (2026-10-10; a run option of na49_mixing.kernel_config, never set by
    # the committed config): a continuation channel opens only when its
    # sibling's decay momentum at this family's floor mass reaches the value
    # [GeV/c].  ``None`` opens it whenever the window [floor, M - m_sibling]
    # is non-empty, the committed cascade bit for bit.
    continuation_q_min: float | None = None
    # the line-shape masses of the established daughters of a vertex of this
    # family (2026-10-10; a run option of na49_mixing.kernel_config, never
    # set by the committed config): an established daughter at least this
    # wide [GeV] draws its mass from its line shape on the on-shell window
    # (review grok-4.7 M2).  ``None`` places every established daughter at
    # its pole, the committed cascade bit for bit.
    sibling_line_width: float | None = None
    # the isospin mixture of a vertex of this family (2026-10-10; set by the
    # non-diffractive cascade on the families that carry
    # ``non_diffractive_isospin_mixture``, never by the committed config):
    # the vertex counts every charge combination of its continuation
    # channels and direct pairs equally.  That is the equal mixture of its
    # parent isospins, since the squared Clebsch-Gordan couplings of a charge
    # combination summed over the parent isospins are 1.  ``False`` keeps
    # the family's isospin shares, the committed cascade bit for bit.
    isospin_mixture: bool = False
    # the isospin mixture of this family's vertices in the non-diffractive
    # events (2026-10-10; a run option of na49_mixing.kernel_config on the
    # B != 0 families, never set by the committed config): the
    # non-diffractive cascade sets ``isospin_mixture`` on these families,
    # and the diffractive classes keep the isospin shares.
    non_diffractive_isospin_mixture: bool = False
    # the three-body endings (2026-10-20; na49_mixing.three_body_endings,
    # KernelConfig.three_body_endings): the family's three-body classes of
    # stable hadrons "(d0,d1,d2)", which join its direct pool at a vertex with
    # no open continuation channel, and their strength kappa_3 [GeV^-2];
    # empty keeps the two-body cascade bit for bit.
    three_body_classes: tuple[str, ...] = ()
    three_body_strength: float | None = None
    # the running-width line shapes of the resonance-pair direct channels
    # (2026-10-21; a run option of na49_mixing.kernel_config
    # [KernelConfig.resonance_pair_line_shapes], never set by the committed
    # config): a direct baryon-antibaryon pair of this family opens already
    # from the daughters' threshold-mass sum, the unstable daughter drawing
    # its mass from its line shape on [threshold, min(m_up, M - m_other)]
    # and the channel weight evaluating q* at the effective masses
    # min(m_pole, (m_th + (M - m_other)) / 2).  ``False`` keeps the pole-mass
    # rule bit for bit.
    resonance_pair_line_shapes: bool = False
    # the direct baryon pairs in the continuation gap (2026-10-21; a run
    # option of na49_mixing.kernel_config
    # [KernelConfig.direct_baryon_pairs_in_gap], never set by the committed
    # config): at a neutral mesonic vertex of this family whose open
    # continuation channels include no baryon pair, the open direct
    # baryon-pair channels join the continuing draw pool, scaled so their
    # share of the pool equals their share of the direct pool.  ``False``
    # keeps the continuation-only pool bit for bit.
    direct_baryon_pairs_in_gap: bool = False
    # the S-wave law of a baryon-pair creation step (2026-10-21; a run
    # option of na49_mixing.kernel_config [KernelConfig.baryon_pair_s_wave],
    # never set by the committed config): at a continuation step of a
    # B = 0 state of this configuration into a baryonic family, the
    # daughter-mass density draw uses the plain q* law (L = 0, the lowest
    # wave of a pair from a negative-parity cluster; a convention, the
    # continuum carrying no J^P) in the place of the family's orbital
    # barrier.  ``False`` keeps the family's barrier bit
    # for bit.
    baryon_pair_s_wave: bool = False
    # the forward-hemisphere baryon rule (2026-10-21; a run option of
    # na49_mixing.kernel_config [KernelConfig.baryon_forward_hemisphere],
    # never set by the committed config): at a non-split decay vertex of a
    # baryonic state into one established baryon and one mesonic continuum
    # daughter, the baryon keeps the parent's forward hemisphere and the
    # mesonic cluster recoils backward.  ``False`` keeps the isotropic
    # two-body kinematics bit for bit.
    baryon_forward_hemisphere: bool = False


# ----------------------------------------------------------------------
# Spectral density
# ----------------------------------------------------------------------
def _ramp(family: ContinuumFamily, M: float) -> float:
    """The smoothstep ramp ``S_a(M)`` at mass *M*.

    Exactly ``0.0`` at and below ``m_min``, the ``s^2 (3 - 2 s)``
    smoothstep on ``(m_min, m_min + W)``, exactly ``1.0`` from
    ``m_min + W`` upward.  The one-sided derivatives of the polynomial
    vanish at ``s = 0`` and ``s = 1``, so the ramp is C^1 at both
    junctions.
    """
    s = (M - family.m_min) / family.ramp_width
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    return s * s * (3.0 - 2.0 * s)


def family_density(family: ContinuumFamily, M: float) -> float:
    """The normalized spectral density ``rho_a(M)`` of *family* at *M*.

    ``rho_a(M) = n_norm * M**(-b_power) * exp(M / t_h) * S_a(M)`` with
    the smoothstep ramp :func:`_ramp`.  Exactly ``0.0`` at and below
    ``m_min`` (no continuum strength below the catalog-completeness
    boundary) and positive above it; the integral over the support
    ``[m_min, m_max]`` is 1 on the reference quadrature within
    ``NORMALIZATION_TOLERANCE`` (checked by
    :func:`family_normalization_report`).
    """
    M = float(M)
    if not M > family.m_min:
        return 0.0
    shape = M ** (-family.b_power) * math.exp(M / family.t_h)
    return family.n_norm * shape * _ramp(family, M)


# ----------------------------------------------------------------------
# Active-family selection (card 5)
# ----------------------------------------------------------------------
def active_families(
    fams: list[ContinuumFamily],
    qn: tuple[int, int, int],
) -> list[ContinuumFamily]:
    """Families with the given ``(B, Q, S)`` and ``active == True``.

    The card-5 sampling selector: the continuum source draw consults
    only the activated families of the requested quantum numbers.  The
    declared, inactive families (the card-5 charged and strange mesonic
    records) are never returned, so they are never sampled; plan
    section 9.1 brings charged or strange mesonic states into the final
    state only through an explicit non-vacuum exchange or an allowed
    parent decay, which the Round 2 catalog kernel handles.
    """
    wanted = tuple(int(v) for v in qn)
    return [f for f in fams if tuple(f.qn) == wanted and f.active]


# ----------------------------------------------------------------------
# Reference-quadrature helpers
# ----------------------------------------------------------------------
def _grid(family: ContinuumFamily) -> np.ndarray:
    """The fixed reference-quadrature nodes over the family support."""
    return np.linspace(family.m_min, SUPPORT_M_MAX, REFERENCE_GRID_POINTS)


def _density_samples(family: ContinuumFamily, xs: np.ndarray) -> np.ndarray:
    """The density sampled on the nodes *xs* (density is non-negative)."""
    return np.array([family_density(family, float(x)) for x in xs])


def _trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """Trapezoid integral of *ys* over the abscissae *xs*."""
    if xs.size < 2:
        return 0.0
    return float((0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])).sum())


def _unit_n_norm(family: ContinuumFamily) -> float:
    """The unit normalization ``N_a`` on the reference quadrature.

    The reciprocal of the trapezoid integral of the unnormalized shape
    ``M**(-b_power) * exp(M / t_h) * S_a(M)`` over the support; the same
    fixed quadrature later measures the normalized density, so the unit
    normalization is grid-convergent by construction.
    """
    xs = _grid(family)

    def shape(M: float) -> float:
        if not M > family.m_min:
            return 0.0
        return (
            M ** (-family.b_power)
            * math.exp(M / family.t_h)
            * _ramp(family, M)
        )

    ys = np.array([shape(float(x)) for x in xs])
    total = _trapezoid(xs, ys)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            f"family {family.family_id}: the unnormalized spectral shape "
            "integrates to a non-positive value on the reference "
            f"quadrature ({total!r}); the unit normalization is undefined"
        )
    return 1.0 / total


# ----------------------------------------------------------------------
# Inverse-CDF draw (the card-3 draw_mass convention)
# ----------------------------------------------------------------------
#: Pure compute cache of the per-family draw tables, keyed on the frozen
#: family record.  An entry is the tuple ``(lo, hi, cdf, cell_width)``.
#: A stored value is the bit-identical result of the one grid evaluation;
#: the cache is a pure compute cache and consumes no random numbers.
_DRAW_TABLE_CACHE: dict[ContinuumFamily, tuple] = {}


def _draw_table(family: ContinuumFamily) -> tuple:
    """The inverse-CDF draw table on the reference quadrature.

    Per-cell masses of the piecewise-linear density interpolant, the
    normalized cumulative cell masses, and the uniform cell width --
    the ``_draw_entry`` structure of ``na49_mixing.resonance_widths``
    on the 10,001-node reference grid.
    """
    cached = _DRAW_TABLE_CACHE.get(family)
    if cached is not None:
        return cached
    xs = _grid(family)
    ys = _density_samples(family, xs)
    cell_masses = 0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])
    total = float(cell_masses.sum())
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            f"family {family.family_id}: the spectral density integrates "
            "to a non-positive value on the reference quadrature "
            f"({total!r}); the draw table is undefined"
        )
    cdf = np.cumsum(cell_masses) / total
    cell_width = (SUPPORT_M_MAX - family.m_min) / cdf.size
    entry = (family.m_min, SUPPORT_M_MAX, cdf, cell_width)
    _DRAW_TABLE_CACHE[family] = entry
    return entry


def family_draw(family: ContinuumFamily, rng: np.random.Generator) -> float:
    """Draw one mass from ``rho_a``, GeV.

    The inverse CDF of the density on the fixed 10,001-node reference
    grid over ``[m_min, m_max]``: the first uniform selects the grid
    cell through the normalized cumulative cell masses (``searchsorted``
    with ``side="right"``, the last cell clamping the ``u -> 1`` edge),
    the second uniform places the mass uniformly within the cell.  The
    convention consumes exactly two uniforms per draw and follows
    ``draw_mass`` of ``na49_mixing.resonance_widths`` (Round 2 card 3);
    the returned mass lies in ``[m_min, m_max]``.

    Raises ``ValueError`` naming the family when the family is inactive
    (the card-5 ``active`` flag controls sampling; the declared charged
    and strange mesonic families are never sampled in Round 3).
    """
    if not family.active:
        raise ValueError(
            f"family {family.family_id} (qn {tuple(family.qn)}) is not "
            "active: inactive families are declared, not sampled (the "
            "card-5 active flag controls continuum sampling)"
        )
    lo, hi, cdf, cell_width = _draw_table(family)
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    mass = lo + (idx + u_within) * cell_width
    return float(min(max(mass, lo), hi))


# ----------------------------------------------------------------------
# Normalization report
# ----------------------------------------------------------------------
def family_normalization_report(family: ContinuumFamily) -> dict:
    """The unit-normalization report of *family* on the reference grid.

    The trapezoid integral of ``rho_a`` over the support
    ``[m_min, m_max]`` on the fixed 10,001-node grid; it must equal 1
    within ``NORMALIZATION_TOLERANCE = 1e-6`` (the Round 2 tolerance).
    The report is JSON-serializable and carries the family identity, the
    grid identity, the support, the integral, the tolerance, and the
    boolean outcome.
    """
    xs = _grid(family)
    ys = _density_samples(family, xs)
    integral = _trapezoid(xs, ys)
    return {
        "family_id": family.family_id,
        "qn": list(family.qn),
        "grid_points": REFERENCE_GRID_POINTS,
        "support": [family.m_min, SUPPORT_M_MAX],
        "integral": integral,
        "tolerance": NORMALIZATION_TOLERANCE,
        "within_tolerance": abs(integral - 1.0) <= NORMALIZATION_TOLERANCE,
    }


# ----------------------------------------------------------------------
# Config loader
# ----------------------------------------------------------------------
#: The required keys of one family record in
#: ``config/v2_singlet/continuum_resonances.json``.
_REQUIRED_RECORD_KEYS = frozenset({
    "family_id", "qn", "state_type", "m_min", "t_h", "b_power",
    "ramp_width", "nominal_width", "m_max", "n_norm",
    "allowed_decay_classes", "active", "field_provenance",
})

#: The derived normalization is verified against the recorded value at
#: this relative tolerance (the quadrature is deterministic, so the
#: agreement is at float rounding level).
_NORM_RECORDER_REL_TOL = 1e-12


def _render_provenance(field: str, tag: dict) -> str:
    """One provenance line for the family record (Round 2 discipline)."""
    return (
        f"{field} <- {tag.get('document', '')}; "
        f"{tag.get('section_or_table', '')}; "
        f"accessed_on={tag.get('accessed_on', '')}; "
        f"transcribed_value={tag.get('transcribed_value', '')!r}; "
        f"transcriber={tag.get('transcriber', '')}"
    )


def _family_from_record(record: dict) -> ContinuumFamily:
    """Build one :class:`ContinuumFamily` from a config record."""
    missing = _REQUIRED_RECORD_KEYS - set(record)
    if missing:
        raise ValueError(
            "continuum family record is missing required keys: "
            f"{sorted(missing)}"
        )
    family_id = str(record["family_id"])
    qn_raw = record["qn"]
    if not (isinstance(qn_raw, list) and len(qn_raw) == 3):
        raise ValueError(
            f"family {family_id}: qn must be a 3-entry list, got {qn_raw!r}"
        )
    qn = tuple(int(v) for v in qn_raw)
    state_type = str(record["state_type"])
    if state_type != CONTINUUM_STATE_TYPE:
        raise ValueError(
            f"family {family_id}: state_type must be "
            f"{CONTINUUM_STATE_TYPE!r} (the Round 1 card-4 continuum "
            f"stamp), got {state_type!r}"
        )
    m_max = float(record["m_max"])
    if m_max != SUPPORT_M_MAX:
        raise ValueError(
            f"family {family_id}: recorded m_max {m_max!r} does not equal "
            f"the pre-registered support upper end {SUPPORT_M_MAX!r}"
        )
    m_min = float(record["m_min"])
    t_h = float(record["t_h"])
    b_power = float(record["b_power"])
    ramp_width = float(record["ramp_width"])
    nominal_width = float(record["nominal_width"])
    if not t_h > 0.0:
        raise ValueError(f"family {family_id}: t_h must be positive")
    if not ramp_width > 0.0:
        raise ValueError(f"family {family_id}: ramp_width must be positive")
    if not nominal_width >= 0.0:
        raise ValueError(
            f"family {family_id}: nominal_width must be non-negative"
        )
    if m_min + ramp_width > SUPPORT_M_MAX:
        raise ValueError(
            f"family {family_id}: the ramp end m_min + ramp_width = "
            f"{m_min + ramp_width!r} lies beyond the support upper end "
            f"{SUPPORT_M_MAX!r}"
        )
    classes_raw = record["allowed_decay_classes"]
    if not (
        isinstance(classes_raw, list)
        and classes_raw
        and all(isinstance(c, str) and c for c in classes_raw)
    ):
        raise ValueError(
            f"family {family_id}: allowed_decay_classes must be a "
            f"non-empty list of class identifier strings, got "
            f"{classes_raw!r}"
        )
    allowed = tuple(classes_raw)
    if len(set(allowed)) != len(allowed):
        raise ValueError(
            f"family {family_id}: allowed_decay_classes carries duplicate "
            "identifiers"
        )
    provenance_raw = record["field_provenance"]
    if not isinstance(provenance_raw, dict) or not provenance_raw:
        raise ValueError(
            f"family {family_id}: field_provenance must be a non-empty "
            "object keyed by field name"
        )
    provenance = tuple(
        _render_provenance(field, tag)
        for field, tag in provenance_raw.items()
    )
    continuation_floor = None
    if "continuation_floor" in record:
        try:
            continuation_floor = float(record["continuation_floor"])
        except (TypeError, ValueError):
            continuation_floor = math.nan
        if not (math.isfinite(continuation_floor)
                and 0.0 < continuation_floor <= m_min):
            raise ValueError(
                f"family {family_id}: continuation_floor must be finite "
                f"and in (0, m_min = {m_min!r}], got "
                f"{record['continuation_floor']!r}"
            )
    continuation_ceiling = None
    if "continuation_ceiling" in record:
        try:
            continuation_ceiling = float(record["continuation_ceiling"])
        except (TypeError, ValueError):
            continuation_ceiling = math.nan
        if not (math.isfinite(continuation_ceiling)
                and continuation_ceiling >= SUPPORT_M_MAX):
            raise ValueError(
                f"family {family_id}: continuation_ceiling must be finite "
                f"and at least the support edge {SUPPORT_M_MAX!r}, got "
                f"{record['continuation_ceiling']!r}"
            )
    continuation_mass_law = None
    if "continuation_mass_law" in record:
        continuation_mass_law = record["continuation_mass_law"]
        if continuation_mass_law not in CONTINUATION_MASS_LAWS:
            raise ValueError(
                f"family {family_id}: continuation_mass_law must be one of "
                f"{CONTINUATION_MASS_LAWS!r}, got {continuation_mass_law!r}"
            )
    continuation_barrier_l = None
    if "continuation_barrier_l" in record:
        raw_l = record["continuation_barrier_l"]
        if isinstance(raw_l, bool) or not isinstance(raw_l, int) or raw_l < 0:
            raise ValueError(
                f"family {family_id}: continuation_barrier_l must be a "
                f"non-negative integer, got {raw_l!r}"
            )
        continuation_barrier_l = int(raw_l)
    continuation_sibling_weight = None
    if "continuation_sibling_weight" in record:
        continuation_sibling_weight = record["continuation_sibling_weight"]
        if continuation_sibling_weight not in CONTINUATION_SIBLING_WEIGHTS:
            raise ValueError(
                f"family {family_id}: continuation_sibling_weight must be one of "
                f"{CONTINUATION_SIBLING_WEIGHTS!r}, got {continuation_sibling_weight!r}"
            )
    continuation_daughters = None
    if "continuation_daughters" in record:
        continuation_daughters = record["continuation_daughters"]
        if continuation_daughters not in CONTINUATION_DAUGHTER_RULES:
            raise ValueError(
                f"family {family_id}: continuation_daughters must be one of "
                f"{CONTINUATION_DAUGHTER_RULES!r}, got {continuation_daughters!r}"
            )
    continuation_t_h_alpha = None
    if "continuation_t_h_alpha" in record:
        raw_a = record["continuation_t_h_alpha"]
        if isinstance(raw_a, bool) or not isinstance(raw_a, (int, float)) or not math.isfinite(raw_a) or raw_a < 0.0:
            raise ValueError(
                f"family {family_id}: continuation_t_h_alpha must be a non-negative finite number, got {raw_a!r}"
            )
        continuation_t_h_alpha = float(raw_a)
    skeleton = ContinuumFamily(
        family_id=family_id,
        qn=qn,  # type: ignore[arg-type]
        m_min=m_min,
        t_h=t_h,
        b_power=b_power,
        n_norm=1.0,
        ramp_width=ramp_width,
        nominal_width=nominal_width,
        allowed_decay_classes=allowed,
        active=bool(record["active"]),
        provenance=provenance,
        continuation_floor=continuation_floor,
        continuation_ceiling=continuation_ceiling,
        continuation_mass_law=continuation_mass_law,
        continuation_barrier_l=continuation_barrier_l,
        continuation_sibling_weight=continuation_sibling_weight,
        continuation_daughters=continuation_daughters,
        continuation_t_h_alpha=continuation_t_h_alpha,
    )
    recomputed = _unit_n_norm(skeleton)
    recorded = float(record["n_norm"])
    if not math.isclose(
        recomputed, recorded,
        rel_tol=_NORM_RECORDER_REL_TOL, abs_tol=0.0,
    ):
        raise ValueError(
            f"family {family_id}: the recorded n_norm {recorded!r} does "
            "not reproduce the unit normalization on the reference "
            f"quadrature (recomputed {recomputed!r})"
        )
    return dataclasses.replace(skeleton, n_norm=recorded)


def families_from_json(path: str) -> list[ContinuumFamily]:
    """Load the family section of ``continuum_resonances.json``.

    Reads the ``families`` list of the config file at *path* and returns
    one :class:`ContinuumFamily` per record, in file order: the card-4
    baryonic families followed by the card-5 mesonic families, the
    activated and the declared-inactive records alike.  Every record is
    validated at load time: the required keys, the
    ``continuum_resonance`` state stamp, the pre-registered support
    upper end, the positive scales, the class identifier list, and the
    recorded ``n_norm`` against the recomputed unit normalization.
    """
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    records = config.get("families")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"{path}: the config carries no non-empty 'families' list"
        )
    families = [_family_from_record(record) for record in records]
    rule = config.get("isospin_rule")
    if rule is None:
        return families
    if not isinstance(rule, dict) or rule.get("rule") not in ISOSPIN_RULES:
        raise ValueError(
            f"{path}: isospin_rule must name one of {sorted(ISOSPIN_RULES)}, "
            f"got {rule!r}")
    return [dataclasses.replace(f, isospin=family_isospin(f.qn)) for f in families]


#: The closed set of continuum isospin rules.  ``minimal_abs_i3``: every
#: continuum family carries the smallest isospin compatible with its
#: charge, I = |I3| with 2 I3 = 2Q - B - S (Gell-Mann-Nishijima).  For
#: systems excited by an isospin-zero (colorless, neutral) exchange this
#: is the isospin of the excited hadron: 1/2 for an excited nucleon, 0
#: for a central mesonic system (Round 5 follow-up, owner decision
#: 2026-09-30).
ISOSPIN_RULES = frozenset({"minimal_abs_i3"})


def family_isospin(qn) -> float:
    """The minimal isospin |I3| of a (B, Q, S) family."""
    b, q, s = (int(v) for v in qn)
    return abs(2 * q - b - s) / 2.0


# ----------------------------------------------------------------------
# Smooth catalog/continuum matching (card 6; plan section 22 task 20,
# plan section 9.3; round 3 index item 4)
# ----------------------------------------------------------------------
#: The catalog-completeness boundary ``B`` of the blend: the M8-frozen
#: ``continuum.match_mass`` of ``config/v2_singlet/legacy_v1_manifest.json``
#: (round 3 index item 4; the card-4/5 family ``m_min``).
BLEND_BOUNDARY = 2.680194110727528

#: The pre-registered blend ramp width ``W`` (round 3 index item 4; plan
#: section 9.3: "fixed initially and varied only as a systematic" -- the
#: R32 variants are run-level parameter choices, never a refit).
BLEND_RAMP_WIDTH = 0.5

#: The pre-registered in-band ramp shapes of the Round 4 rework card
#: (``TASK-20260920-r4-rework-transition-routing-formulation``, Phase 1):
#: C0 linear (the sealed R32 formulation and the default), C1 smoothstep,
#: C2 quadratic.  A run-level systematic choice, never a fit parameter.
BLEND_RAMP_SHAPES = ("linear", "smoothstep", "quadratic")


@dataclasses.dataclass(frozen=True)
class BlendSpec:
    """The fixed blend weight of the smooth catalog/continuum matching
    (plan section 9.3; round 3 index item 4).

    ``boundary`` is the catalog-completeness boundary ``B`` (default
    :data:`BLEND_BOUNDARY`, the M8-frozen ``continuum.match_mass``) and
    ``ramp_width`` the transition width ``W`` (default
    :data:`BLEND_RAMP_WIDTH`).  :meth:`weight` is the pre-registered
    piecewise ramp, continuous at both junctions.  The record is a
    frozen value object and the ramp carries no state: the blend weight
    is fixed by construction, and no M8 parameter is touched.
    """

    boundary: float = BLEND_BOUNDARY
    ramp_width: float = BLEND_RAMP_WIDTH
    ramp_shape: str = "linear"

    def __post_init__(self) -> None:
        boundary = float(self.boundary)
        ramp_width = float(self.ramp_width)
        if not (math.isfinite(boundary) and boundary > 0.0):
            raise ValueError(
                f"BlendSpec: the boundary must be finite and positive, "
                f"got {self.boundary!r}"
            )
        if not (math.isfinite(ramp_width) and ramp_width > 0.0):
            raise ValueError(
                f"BlendSpec: the ramp width must be finite and positive, "
                f"got {self.ramp_width!r}"
            )
        if self.ramp_shape not in BLEND_RAMP_SHAPES:
            raise ValueError(
                f"BlendSpec: the ramp shape must be one of "
                f"{BLEND_RAMP_SHAPES}, got {self.ramp_shape!r}"
            )

    def weight(self, M: float) -> float:
        """The blend weight ``w(M)`` at mass *M* (the pre-registered
        piecewise ramp).

        Exactly ``0.0`` at and below the boundary ``B`` and exactly
        ``1.0`` from ``B + W`` upward for every shape.  On the open
        transition interval ``(B, B + W)`` with ``s = (M - B) / W``:
        ``"linear"`` is ``s`` (the sealed R32 formulation, the default,
        bit-identical to the pre-2026-09-28 surface); ``"smoothstep"``
        is ``s^2 (3 - 2 s)`` (C^1 at both junctions, the same polynomial
        as the family threshold ramp); ``"quadratic"`` is ``s^2`` (the
        asymmetric in-band profile, band mean 1/3).  The one-sided values
        agree at both junctions for every shape, so the ramp is
        continuous.  The shapes are the pre-registered candidate set of
        the Round 4 rework card
        ``TASK-20260920-r4-rework-transition-routing-formulation``
        (C0, C1, C2); no shape adds a fitted parameter.
        """
        M = float(M)
        if not M > self.boundary:
            return 0.0
        if M >= self.boundary + self.ramp_width:
            return 1.0
        s = (M - self.boundary) / self.ramp_width
        if self.ramp_shape == "smoothstep":
            return s * s * (3.0 - 2.0 * s)
        if self.ramp_shape == "quadratic":
            return s * s
        return s


def _daughter_masses(catalog, record) -> dict[int, float]:
    """Pole masses of every channel daughter of *record* (the card-3
    width law's daughter-mass map; the same construction as the
    kernel's ``ResonanceDecayer._daughter_mass_map`` and the
    ``staged_source_grid`` helper)."""
    masses: dict[int, float] = {}
    for channel in record.decay_channels:
        for daughter in channel.daughters:
            daughter = int(daughter)
            if daughter in masses:
                continue
            try:
                masses[daughter] = float(catalog.get(daughter).pole_mass)
            except KeyError:
                entry = get_particle(daughter)
                if entry is None:
                    raise KeyError(
                        f"daughter pdg_id {daughter} is neither a catalog "
                        "record nor in the vendored PDG table"
                    ) from None
                masses[daughter] = float(entry.mass)
    return masses


#: Pure compute cache of the per-record daughter-mass maps, keyed on
#: ``(record, catalog)`` (the frozen record hashes by value, the catalog
#: by identity).  A stored value is the bit-identical result of the one
#: construction; the cache consumes no random numbers.
_BLEND_MASS_MAP_CACHE: dict[tuple, dict[int, float]] = {}


def _record_density(record, catalog, M: float) -> float:
    """The card-3 normalized Breit-Wigner spectral function of *record*
    at *M* (exactly ``0.0`` for a stable record: a degenerate support
    carries no normalized line shape)."""
    key = (record, catalog)
    dmap = _BLEND_MASS_MAP_CACHE.get(key)
    if dmap is None:
        dmap = _daughter_masses(catalog, record)
        _BLEND_MASS_MAP_CACHE[key] = dmap
    return spectral_function(float(M), record, dmap)


def catalog_density(qn, M: float, catalog) -> float:
    """The raw catalog mixture density ``rho_catalog(M)`` for the source
    quantum numbers *qn*.

    The sum over the committed catalog records' normalized
    relativistic-Breit-Wigner spectral functions (the Round 2 card-3
    surface, each record normalized to unit integral over its support),
    restricted to the records whose ``(B, Q, S)`` matches *qn*.  A
    stable record (degenerate support) carries no normalized line shape
    and contributes exactly ``0.0``.

    This is the shape definition of the card.  Inside the blend
    (:func:`blended_density`, :func:`blended_normalization_report`) the
    mixture carries the boundary-matching strength factor of the module
    docstring -- the double-counting identity fixes the free overall
    normalization of the sum, never the record line shapes.
    """
    wanted = tuple(int(v) for v in qn)
    M = float(M)
    total = 0.0
    for record in catalog.records():
        if (int(record.B), int(record.Q), int(record.S)) != wanted:
            continue
        total += _record_density(record, catalog, M)
    return total


def continuum_density(qn, M: float, fams) -> float:
    """The continuum mixture density ``rho_cont(M)`` for the source
    quantum numbers *qn*.

    The sum over the activated continuum families of cards 4-5 with the
    matching ``(B, Q, S)`` (the card-5 ``active`` flag controls the
    mixture: the declared, inactive charged and strange mesonic records
    are never summed).  Each family is unit-normalized over its support
    ``[m_min, m_max]`` on the reference quadrature, and the density is
    exactly ``0.0`` at and below ``m_min = B`` (no continuum strength
    below the boundary).
    """
    wanted = tuple(int(v) for v in qn)
    M = float(M)
    total = 0.0
    b, q, s = wanted
    wanted3: tuple[int, int, int] = (b, q, s)
    for family in active_families(fams, wanted3):
        total += family_density(family, M)
    return total


def _matching_records(qn, catalog) -> tuple:
    """The catalog records with the source ``(B, Q, S)``, in catalog
    record order."""
    wanted = tuple(int(v) for v in qn)
    return tuple(
        record
        for record in catalog.records()
        if (int(record.B), int(record.Q), int(record.S)) == wanted
    )


def _matching_active_families(qn, fams) -> tuple:
    """The activated families with the source ``(B, Q, S)``, in config
    order (the card-5 ``active`` flag controls the mixture)."""
    b, q, s = (int(v) for v in qn)
    wanted: tuple[int, int, int] = (b, q, s)
    return tuple(active_families(fams, wanted))


def _blended_support(qn, spec, catalog, fams) -> tuple[float, float]:
    """The union support ``[m_min, m_max]`` of the two blended
    densities: the lowest matching record threshold through the support
    upper end (``SUPPORT_M_MAX``, which also covers the continuum
    families and the ramp end ``B + W``)."""
    records = _matching_records(qn, catalog)
    if not records:
        raise ValueError(
            f"blended matching: the catalog carries no record with the "
            f"source quantum numbers {tuple(int(v) for v in qn)}"
        )
    if not _matching_active_families(qn, fams):
        raise ValueError(
            f"blended matching: no activated continuum family carries "
            f"the source quantum numbers {tuple(int(v) for v in qn)}"
        )
    lo = min([float(spec.boundary)]
             + [float(record.mass_range[0]) for record in records])
    hi = max([float(spec.boundary) + float(spec.ramp_width), SUPPORT_M_MAX]
             + [float(record.mass_range[1]) for record in records])
    return lo, hi


#: Pure compute cache of the blend pieces per activated source QN set,
#: keyed on the quantum numbers, the blend spec values, the matching
#: records and families (frozen, hash by value), and the union support.
#: A stored entry is the bit-identical result of the one grid
#: evaluation; the cache is a pure compute cache and consumes no random
#: numbers.
_BLEND_PIECES_CACHE: dict[tuple, dict] = {}


def _blended_pieces(qn, spec: BlendSpec, catalog, fams) -> dict:
    """The report-grid pieces of the blend for *qn* (cached).

    The fixed 10,001-node reference grid over the union support; the
    raw catalog and continuum density samples; the ramp weights; the
    continuum integral ``D``, the ramp complement ``G``, the raw
    captured catalog mass, and the boundary-matching factor ``lambda``
    that realizes the double-counting identity (the module docstring
    carries the derivation).
    """
    wanted = tuple(int(v) for v in qn)
    records = _matching_records(wanted, catalog)
    actives = _matching_active_families(wanted, fams)
    lo, hi = _blended_support(wanted, spec, catalog, fams)
    key = (
        wanted,
        float(spec.boundary),
        float(spec.ramp_width),
        records,
        actives,
        lo,
        hi,
    )
    cached = _BLEND_PIECES_CACHE.get(key)
    if cached is not None:
        return cached
    xs = np.linspace(lo, hi, REFERENCE_GRID_POINTS)
    cat = np.zeros(xs.size)
    for record in records:
        cat += np.array(
            [_record_density(record, catalog, float(x)) for x in xs]
        )
    cont = np.zeros(xs.size)
    for family in actives:
        cont += np.array([family_density(family, float(x)) for x in xs])
    w = np.array([spec.weight(float(x)) for x in xs])
    one_minus_w = 1.0 - w
    cont_integral = _trapezoid(xs, cont)
    ramp_complement = _trapezoid(xs, one_minus_w * cont)
    captured_raw = _trapezoid(xs, one_minus_w * cat)
    if not (math.isfinite(captured_raw) and captured_raw > 0.0):
        raise ValueError(
            f"blended matching: the raw catalog mixture for the source "
            f"quantum numbers {wanted} carries no captured strength "
            f"below the ramp end ({captured_raw!r}); the "
            "boundary-matching normalization is undefined"
        )
    matching_factor = (1.0 - cont_integral + ramp_complement) / captured_raw
    if not (math.isfinite(matching_factor) and matching_factor > 0.0):
        raise ValueError(
            f"blended matching: the boundary-matching factor for the "
            f"source quantum numbers {wanted} is not a positive finite "
            f"number ({matching_factor!r})"
        )
    pieces = {
        "qn": wanted,
        "xs": xs,
        "cat_raw": cat,
        "cont": cont,
        "w": w,
        "one_minus_w": one_minus_w,
        "support": (lo, hi),
        "cont_integral": cont_integral,
        "ramp_complement": ramp_complement,
        "captured_raw": captured_raw,
        "matching_factor": matching_factor,
    }
    _BLEND_PIECES_CACHE[key] = pieces
    return pieces


def blended_density(qn, M: float, spec: BlendSpec, catalog, fams) -> tuple:
    """The blended density ``(rho_total, rho_catalog, rho_cont)`` at
    mass *M* for the source quantum numbers *qn*.

    ``rho_catalog`` is the catalog mixture
    (:func:`catalog_density`) at the boundary-matching strength, and
    ``rho_cont`` the continuum mixture (:func:`continuum_density`), the
    pinned unit-normalized family sum.  ``rho_total`` is the plan
    section 9.3 blend of the two at the fixed ramp weight
    ``spec.weight(M)``:

        rho_total(M) = [1 - w(M)] * rho_catalog(M) + w(M) * rho_cont(M),

    continuous at both junctions ``B`` and ``B + W`` (the catalog side
    is continuous there and the continuum density is exactly ``0.0`` at
    ``B``, where the weight is still ``0.0``).  The double-counting
    identity of the blend is measured by
    :func:`blended_normalization_report`.
    """
    pieces = _blended_pieces(qn, spec, catalog, fams)
    M = float(M)
    rho_catalog = pieces["matching_factor"] * catalog_density(qn, M, catalog)
    rho_cont = continuum_density(qn, M, fams)
    w = spec.weight(M)
    rho_total = (1.0 - w) * rho_catalog + w * rho_cont
    return (rho_total, rho_catalog, rho_cont)


def route_source_decay(qn, M: float, spec: BlendSpec, catalog, fams,
                       rng) -> str:
    """Route one source decay at the frozen source mass *M* (the plan
    section 9.3 blend evaluated at the source mass; round 3 index item
    4).

    Returns ``'catalog'`` or ``'continuum'`` so the caller can stamp
    the source lineage and realize the corresponding path:

    - ``M < B``: ``'catalog'``, the R23 builder convention, consuming
      **zero** uniforms -- the caller's catalog-path draw sequence (the
      spawned ``decay_channel`` child's record draw, the card-3 mass
      draw on the spawned ``decay_kinematics`` child, the Round 2
      kernel) stays byte-identical to the no-blend convention at the
      same seed;
    - ``B <= M < B + W``: exactly **one** uniform is consumed (the
      routing draw, taken before any record draw so the lineage is
      traceable) and the route is ``'continuum'`` exactly when the
      draw ``u < w(M)``;
    - ``M >= B + W``: ``'continuum'``, consuming **zero** uniforms.

    Raises ``ValueError`` naming the quantum numbers when the routed
    path has no state to draw (no matching record is open at *M* on a
    catalog route, or no activated family carries the quantum numbers
    on a continuum route); the validation consumes no uniforms.

    No M8 parameter is touched and no source-mass draw is consumed:
    the routing acts only on the decay layer at the frozen source mass.
    """
    wanted = tuple(int(v) for v in qn)
    M = float(M)
    if M < spec.boundary:
        _require_open_catalog(wanted, M, catalog)
        return "catalog"
    if M >= spec.boundary + spec.ramp_width:
        _require_active_family(wanted, fams)
        return "continuum"
    u = float(rng.random())
    route = "continuum" if u < spec.weight(M) else "catalog"
    if route == "continuum":
        _require_active_family(wanted, fams)
    else:
        _require_open_catalog(wanted, M, catalog)
    return route


def _require_open_catalog(qn, M: float, catalog) -> None:
    """Raise when no matching record is open at *M* (the card-9/R23
    open-set rule ``mass_range[0] <= M``); consumes no uniforms."""
    for record in _matching_records(qn, catalog):
        if float(record.mass_range[0]) <= float(M):
            return
    raise ValueError(
        f"route_source_decay: the catalog path at mass {M!r} GeV has no "
        f"open record with the source quantum numbers {tuple(qn)} "
        "(the card-9/R23 open set is empty; no fallback route exists)"
    )


def _require_active_family(qn, fams) -> None:
    """Raise when no activated family carries *qn*; consumes no
    uniforms."""
    if not _matching_active_families(qn, fams):
        raise ValueError(
            f"route_source_decay: the continuum path has no activated "
            f"family with the source quantum numbers {tuple(qn)} (the "
            "declared inactive families are never sampled)"
        )


def blended_normalization_report(qn, spec: BlendSpec, catalog, fams) -> dict:
    """The double-counting report of the blend for *qn* (JSON-
    serializable).

    Trapezoid integrals of ``rho_total``, ``rho_catalog``, and
    ``rho_cont`` on the fixed 10,001-node reference grid over the union
    support ``[m_min, m_max]`` (:func:`_blended_support`), with the
    double-counting identity of plan section 9.3:

    - ``int(rho_total) = 1`` within ``NORMALIZATION_TOLERANCE``; and
    - ``int_{M<B}(rho_catalog) + int_{M>=B}(rho_total) = 1`` within the
      same tolerance,

    i.e. no strength is lost or doubled at the boundary.  The report
    also carries the raw pieces of the matching (the continuum integral
    ``D``, the ramp complement ``G``, the raw captured catalog mass,
    and the boundary-matching factor) so the normalization is
    reviewable without recomputation.
    """
    pieces = _blended_pieces(qn, spec, catalog, fams)
    xs = pieces["xs"]
    rho_catalog = pieces["matching_factor"] * pieces["cat_raw"]
    rho_total = pieces["one_minus_w"] * rho_catalog + pieces["w"] * pieces["cont"]
    integral_total = _trapezoid(xs, rho_total)
    integral_catalog = _trapezoid(xs, rho_catalog)
    integral_cont = _trapezoid(xs, pieces["cont"])
    below = xs < float(spec.boundary)
    integral_catalog_below = _trapezoid(
        xs, np.where(below, rho_catalog, 0.0)
    )
    integral_total_above = _trapezoid(
        xs, np.where(below, 0.0, rho_total)
    )
    split = integral_catalog_below + integral_total_above
    dev_total = abs(integral_total - 1.0)
    dev_split = abs(split - 1.0)
    return {
        "qn": list(pieces["qn"]),
        "grid_points": REFERENCE_GRID_POINTS,
        "support": [pieces["support"][0], pieces["support"][1]],
        "boundary": float(spec.boundary),
        "ramp_width": float(spec.ramp_width),
        "integrals": {
            "rho_total": integral_total,
            "rho_catalog": integral_catalog,
            "rho_cont": integral_cont,
        },
        "split_integrals": {
            "catalog_below_boundary": integral_catalog_below,
            "rho_total_at_or_above_boundary": integral_total_above,
            "sum": split,
        },
        "catalog_matching": {
            "cont_integral": pieces["cont_integral"],
            "ramp_complement": pieces["ramp_complement"],
            "captured_raw": pieces["captured_raw"],
            "matching_factor": pieces["matching_factor"],
        },
        "double_counting_identity": {
            "total": {
                "integral": integral_total,
                "deviation": dev_total,
                "tolerance": NORMALIZATION_TOLERANCE,
                "within_tolerance": bool(
                    dev_total <= NORMALIZATION_TOLERANCE
                ),
            },
            "split": {
                "integral": split,
                "deviation": dev_split,
                "tolerance": NORMALIZATION_TOLERANCE,
                "within_tolerance": bool(
                    dev_split <= NORMALIZATION_TOLERANCE
                ),
            },
        },
        "tolerance": NORMALIZATION_TOLERANCE,
    }
