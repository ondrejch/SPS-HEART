"""The kernel configuration: the physics options of the cascade in one place.

``TASK-20261004-refactor-run-engine-kernel-config``, piece P1 (2026-10-04).
Before this module every run driver set the cascade options in memory on
the continuum families (``dataclasses.replace(family, continuation_floor=...,
continuation_ceiling=..., continuation_mass_law=..., continuation_barrier_l=...)``)
inside its own ``_load_physics``.  A :class:`KernelConfig` names those
options once, validates them, and :func:`load_kernel` applies them to the
committed catalog and families.  The committed configuration sets none of
the options; a family that already carries one is refused (the options come
from the run, never from the committed config).

Named profiles:

- :data:`LEGACY`: no option, the committed kernel bit for bit (the card-4
  continuation rule at the family edge and support).
- :data:`R74`: the kernel of the sealed R74 run: the 1.7 GeV continuation
  floor, the 17.3 GeV ceiling, the density-weighted continuation step and
  its P-wave barrier.

The floor grid of a run (``n_floors``) varies the floor per branch:
``dataclasses.replace(config, continuation_floor=floor)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

from .conjugate_towers import conjugate_baryon_families, conjugate_catalog
from .continuum_isospin import (SECTORS as ISOSPIN_SECTORS, delta_line_families, isospin_partner_families,
                                isovector_meson_pair_families, isovector_neutral_mesonic_families)
from .floor_pools import SECTORS as FLOOR_POOL_SECTORS, floor_pool_families
from .hadron_catalog import HadronCatalog
from .resonance_pairs import resonance_pair_families
from .isoscalar_dipion import isoscalar_dipion_catalog, isoscalar_dipion_families, load_isoscalar_dipion
from .light_scalar_mesons import light_scalar_mesons_catalog, light_scalar_mesons_families, load_light_scalar_mesons
from .nucleon_kstar_pairs import nucleon_kstar_families
from .pdg_hyperons import load_pdg_hyperons, pdg_hyperons_catalog, pdg_hyperons_families
from .pdg_nucleon_resonances import (load_pdg_nucleon_resonances, pdg_nucleon_resonances_catalog,
                                     pdg_nucleon_resonances_families)
from .eta_prime import eta_prime_catalog, eta_prime_families, load_eta_prime
from .axial_isoscalars import axial_isoscalars_catalog, axial_isoscalars_families, load_axial_isoscalars
from .excited_kaons import excited_kaons_catalog, excited_kaons_families, load_excited_kaons
from .associated_kstar import associated_kstar_families
from .baryon_table_corrections import (baryon_pdg_widths_catalog, baryon_table_corrections_catalog,
                                       load_baryon_table_corrections)
from .xi_baryons import load_xi_baryons, xi_baryons_catalog, xi_baryons_families
from .xi_tower import xi_tower_families
from .omega_baryon import load_omega_baryon, omega_baryon_catalog, omega_baryon_families
from .baryon_pair_completion import MEMBERS as PAIR_COMPLETION_MEMBERS, SCOPES as PAIR_COMPLETION_SCOPES, \
    SHARES as PAIR_COMPLETION_SHARES, baryon_pair_completion_families
from .meson_modes import load_meson_modes, meson_modes_catalog
from .three_body_endings import THREE_BODY_SETS, three_body_families
from .channel_partners import charged_scalar_pair_families, delta_isoscalar_pair_families, hyperon_meson_partner_families
from .spectral_density import (CONTINUATION_DAUGHTER_RULES, CONTINUATION_MASS_LAWS, CONTINUATION_SIBLING_WEIGHTS,
                               SUPPORT_M_MAX, ContinuumFamily, families_from_json)

__all__ = [
    "KernelConfig",
    "LEGACY",
    "R74",
    "PROFILES",
    "CATALOG_REL",
    "FAMILIES_REL",
    "apply_kernel_config",
    "load_kernel",
]

#: The committed catalog and continuum configuration, relative to the
#: nested repository root.
CATALOG_REL = Path("config") / "v2_singlet" / "resonance_catalog.json"
FAMILIES_REL = Path("config") / "v2_singlet" / "continuum_resonances.json"

#: The family fields a kernel configuration sets.
_OPTION_FIELDS = ("continuation_floor", "continuation_ceiling", "continuation_mass_law", "continuation_barrier_l",
                  "continuation_sibling_weight", "continuation_daughters", "continuation_t_h_alpha",
                  "leading_split_fraction", "leading_split_kappa", "leading_split_slope", "leading_split_epsilon",
                  "continuation_q_min", "sibling_line_width", "meson_split_fraction", "meson_split_slope",
                  "meson_split_slope_strange",
                  "string_step_mass", "string_step_slope", "string_step_epsilon", "leading_split_strange",
                  "leading_split_regge", "leading_split_axis")


@dataclass(frozen=True)
class KernelConfig:
    """The cascade options of a run.  ``None`` leaves an option at the
    committed behavior.

    - ``continuation_floor`` [GeV]: the lowest realized mass of a
      continuation daughter (the family minimum mass when ``None``);
    - ``continuation_ceiling`` [GeV]: the highest realized mass of a
      continuation daughter (the 5 GeV support edge when ``None``);
    - ``continuation_mass_law``: ``"edge"`` (the card-4 rule) or
      ``"density"`` (the density-weighted step);
    - ``continuation_barrier_l``: the orbital partial wave of the density
      step (0 or ``None``: the plain q* law; needs the density law);
    - ``continuation_sibling_weight``: the weight of the emitted sibling
      at a continuation step, ``"edge"`` (the card-4 weight at the fixed
      mass) or ``"density"`` (the phase space integrated over the
      daughter's density law, 2026-10-06, the code and physics review's
      M-2; needs the density law);
    - ``continuation_daughters``: the daughter-family rule of a
      continuation step, ``"first"`` (the committed rule) or ``"isospin"``
      (every family of the daughter's (B, Q, S) with its isospin share,
      2026-10-06; the isovector mesonic completion, step 3);
    - ``continuation_t_h_alpha``: the continuation scale of the density
      step, ``t_h = max(t_h, alpha * M_parent)`` in the daughter-mass draw
      and in the density-integrated sibling weight (2026-10-07; card
      TASK-20261005-hadron-count-vs-mass, calibrated on the MARK I e+e-
      multiplicities and the Goulianos-Sticker-White law; needs the
      density law);
    - ``continuation_q_min``: the minimum decay momentum [GeV/c] of a
      continuation step (2026-10-10): a continuation channel opens only
      when its sibling's decay momentum at the daughter's floor mass reaches
      it, so a state within reach of the floor decays into its direct pairs
      instead of emitting a sibling at q* -> 0 (``None``: the committed
      rule, open whenever the window is non-empty);
    - ``sibling_line_width``: the width [GeV] from which an established
      daughter of a continuum vertex draws its mass from its line shape on
      the on-shell window (2026-10-10, review grok-4.7 M2): the window is
      [threshold, min(m_up, M - m_other)] with m_other the continuum
      daughter's floor, or the other established daughter's threshold
      (broad) or pole, and the second broad daughter of a pair takes the
      first one's realized mass; one uniform per drawn daughter on the
      decay_kinematics substream, before the density uniform; the channel
      weights stay at the poles (``None``: every daughter at its pole);
    - ``leading_split_fraction``: the share F of the baryonic continuum
      roots (depth 0, B = 1) that decay through the leading-baryon split
      B* -> B + M*(mu), the established baryon and a mesonic continuum
      daughter (2026-10-08; the fixed-kernel note, "The baryon's energy
      share in the cascade"; needs the density law, whose uniform draws
      mu); the one channel uniform picks the branch and the channel;
    - ``leading_split_kappa``: the split baryon's direction law
      exp(kappa cos theta) about the incoming baryon (0 or ``None``:
      isotropic); ``leading_split_slope`` = b [GeV^-2] instead sets
      kappa = 2 b q*^2, a fixed transfer slope exp(b t') with
      t' = -2 q*^2 (1 - cos theta); at most one of the two;
    - ``leading_split_epsilon``: the remainder's mass law
      (mu^2)^(-1-epsilon) on [floor^2, edge^2] (``None``: 0, flat in
      ln mu^2);
    - ``meson_split_fraction``: the share of the split roots' decays that
      go through the leading-meson split B* -> M + B*(mu), an established
      meson and a baryonic continuum daughter (2026-10-11; the
      fixed-kernel note, "The leading-meson split and the probes at the
      R90 optimum"; needs the leading-baryon split): one uniform on the
      decay_channel substream, before the channel uniform, picks the
      branch, and the branch draws over those channels with the split's
      mass law and the meson along the incoming baryon;
      ``meson_split_slope`` = b [GeV^-2] sets the branch's direction law
      kappa = 2 b q*^2 (``None``: the leading-baryon split's law);
      ``meson_split_slope_strange`` = b_K [GeV^-2] sets it for the branch's
      strange mesons alone, the kaon of p* -> K Y* (2026-11-06, the kaon's
      momentum share at the strangeness-exchange vertex; ``None``: the
      branch's one law);
    - ``leading_split_regge``: ``True`` draws the leading-baryon split's
      mesonic remainder from the Regge flux of the trajectory the leading
      baryon exchanges with the root, (mu^2)^(1 - 2 alpha(0)) in place of
      the committed (mu^2)^(-1-epsilon) with one epsilon (0, the Pomeron's
      law): alpha(0) = 0.5475 (the f/omega family, Donnachie-Landshoff) for
      a same-charge non-strange baryon, -0.93 m_pi^2 (the pion) for a
      charge exchange, -0.93 m_K^2 (the kaon) for a strangeness exchange
      (2026-10-28, the forward Lambda); ``None`` or ``False`` keeps the
      committed law;
    - ``leading_split_strange``: ``False`` takes the strange established
      baryons out of the leading-baryon split's channels together with
      their share of the split's budget (the split fraction at a vertex is
      F times the non-strange channels' weight share, so the non-strange
      leading baryons are drawn as before), and a root hands its
      strangeness over only through the ordinary isotropic continuation
      (2026-10-28, the forward Lambda and K+); ``None`` or ``True`` is the
      committed split.
    - ``string_step_mass`` [GeV]: the string-like continuation step of the
      heavy vertices (2026-10-27): above this mass a vertex emits one
      established hadron at a time (it draws over its continuation channels;
      the direct pairs return where the continuation has closed), draws
      the daughter's mass from (M'^2)^(-1-eps) on [floor^2, edge^2]
      (``string_step_epsilon``, default 0, flat in ln M'^2) in place of the
      density step and directs the line's carrier (the baryon-number
      carrier of a baryonic line, else the continuum daughter) along the
      line's hemisphere and the sibling toward the center with
      exp(kappa cos theta), kappa = 2 b q*^2 (``string_step_slope`` b
      [GeV^-2], default 4), on the same uniforms; the channel weights keep
      the density law's; needs the density law (``None``: the committed
      step bit for bit);
    - ``isospin_completion``: the sectors whose missing charge states join
      as decay-daughter families (``na49_mixing.continuum_isospin``;
      "baryonic", "strange_baryonic", "strange_mesonic"; empty: none);
    - ``delta_lines``: the I = 3/2 root families of the isovector
      exchange's lines at charges +1 and 0
      (``na49_mixing.continuum_isospin.delta_line_families``; needs the
      ``baryonic`` sector);
    - ``isovector_meson_pairs``: the baryon-pair classes of the I = 1
      charged mesonic continuum (p nbar, n pbar;
      ``na49_mixing.continuum_isospin.isovector_meson_pair_families``);
    - ``isovector_neutral_mesons``: the I = 1 neutral mesonic continuum
      family (step 2 of the isovector mesonic completion; needs the
      isospin daughter rule, which alone reaches it);
    - ``three_body_phase_space``: the three-body decay vertex draws its
      intermediate mass with the flat phase-space density
      (``HadronCatalog.three_body_law = "phase_space"``, 2026-10-06; the
      code and physics review's M-3); ``False`` keeps the flat-in-M_bc^2
      draw of the Round 2 vertex;
    - ``isoscalar_dipion``: the isoscalar S-wave dipion of the nucleon
      resonances (2026-10-08; ``na49_mixing.isoscalar_dipion``): the
      f0(500) record, the six N* multiplets at the PDG 2025 hadronic decay
      modes (N sigma and N omega represented, where the committed tables
      fold them into N pi and N rho), and the (N, f0(500)) class of the
      baryonic continuum's direct pool; ``False`` keeps the committed
      catalog and families;
    - ``floor_pools``: the sectors whose heavier two-body pairs join the
      direct pairs of a mesonic continuum state with no open continuation
      channel (2026-10-09; ``na49_mixing.floor_pools``: "charged", the
      I = 1 partners of the neutral pool's heavier pairs; "strange", the
      heavier strange pairs, which needs the ``strange_mesonic``
      completion; empty: none); the continuation steps keep the committed
      classes;
    - ``conjugate_towers``: the charge-conjugate baryonic towers
      (2026-10-09; ``na49_mixing.conjugate_towers``): the conjugate of every
      B = +1 family, every antibaryon record rebuilt as the conjugate of its
      baryon, the two conjugate baryon-pair continuations of a mesonic
      vertex at half the weight each; ``False`` keeps the committed kernel,
      in which the antibaryon of a baryon-pair step is always the bare
      sibling;
    - ``baryon_resonance_pairs``: the baryon-resonance pairs of the neutral
      mesonic family (2026-10-10; ``na49_mixing.resonance_pairs``): the
      pairs of equal isospin among N, N*(1440), N(1520), N(1535) and
      Delta(1232) Deltabar beside p pbar and n nbar, the baryon-pair share
      of the continuation channels and of the direct pairs each held at its
      value without them; needs ``conjugate_towers``;
    - ``non_diffractive_isospin_mixture``: the B != 0 families of the
      non-diffractive events count every charge combination of a vertex
      equally (2026-10-10;
      ``na49_mixing.topology_non_diffractive.non_diffractive_families``),
      the equal mixture of I = 1/2 and 3/2 that the non-vacuum exchanges
      leave on a baryonic line; the diffractive classes keep the isospin
      shares of their I = 1/2 lines;
    - ``light_scalar_mesons``: the f0(980) and a0(980) (2026-10-11;
      ``na49_mixing.light_scalar_mesons``): their catalog records at the
      PDG 2025 estimates, their classes in the neutral mesonic family's
      pool, and the (N, f0(980)) class of the non-strange baryonic family,
      the first kaon-pair channel of a non-strange baryonic vertex;
      ``False`` keeps the committed catalog and families;
    - ``nucleon_kstar_pairs``: the (N, Kbar*(892)) class beside every
      (N, Kbar) class of the strange baryonic families (2026-10-12;
      ``na49_mixing.nucleon_kstar_pairs``), the completion families and
      the conjugate towers included, so a strange baryonic vertex hands
      its strangeness to a vector antikaon as well; ``False`` keeps the
      committed classes;
    - ``pdg_hyperons``: the Lambda and Sigma resonances of the PDG 2025
      summary tables between Sigma(1385) and Lambda(1520) and the 2.0 GeV
      floor, with their (Y*, K) classes at the non-strange baryonic
      families and (Y*, pi) classes at the strange ones (2026-10-15;
      ``na49_mixing.pdg_hyperons``), before the conjugate towers, which
      conjugate the records and the classes; ``False`` keeps the committed
      catalog and classes;
    - ``eta_prime``: the eta'(958) record at its PDG 2025 values and,
      beside every continuum class with an eta, the class with the eta' in
      its place (2026-10-15; ``na49_mixing.eta_prime``), the ninth member
      of the pseudoscalar nonet; ``False`` keeps the committed catalog and
      classes;
    - ``pdg_nucleon_resonances``: the PDG 2025 N and Delta resonances of
      three and four stars between N(1720) and the 2.0 GeV floor that the
      committed catalog lacks, at their PDG masses, widths and listed modes,
      as records: they join the duality picks of the low-mass lines and decay
      as the committed records do (2026-10-21;
      ``na49_mixing.pdg_nucleon_resonances``), before the conjugate towers;
      needs the isoscalar dipion, the light scalar mesons and the eta'
      (daughters of their modes); ``False`` keeps the committed catalog;
    - ``pdg_nucleon_resonance_siblings``: the (N*, pi) classes of those
      states beside the committed ones at the non-strange baryonic lines
      (needs ``pdg_nucleon_resonances``; rejected at the R104 optimum, +309
      fitted and +2,200 fixed, the register section 3.1);
    - ``associated_kstar_pairs``: beside every (hyperon, kaon) class of a
      non-strange baryonic family the class with the K*(892) in the kaon's
      place (2026-10-21; ``na49_mixing.associated_kstar``), the vector
      partners of the associated pairs as (N, Kbar*) partners (N, Kbar) since
      R93; ``False`` keeps the committed classes;
    - ``axial_isoscalars``: the f1(1285) and h1(1170) records at their PDG
      2024 values and, beside every continuum class of an omega and a pion,
      the classes with each in the omega's place (2026-10-21;
      ``na49_mixing.axial_isoscalars``), the I = 0 members of the axial
      nonets whose I = 1 members the catalog carries; ``False`` keeps the
      committed catalog and classes;
    - ``excited_kaons``: the K*(1410), K0*(1430), K1(1400) and K*(1680)
      records at their PDG 2024 values (2026-10-28;
      ``na49_mixing.excited_kaons``) and, beside every baryon-kaon class of
      the baryonic families, the classes with each excited kaon of the
      kaon's charge in the kaon's place; the strange mesons between the
      K1(1270)/K2*(1430) and the continuum floor the committed catalog
      lacks; ``False`` keeps the committed catalog and classes;
    - ``baryon_table_corrections``: the PDG 2025 decay tables of the
      committed Delta(1600) and Delta(1700) (the N* range rule, the
      committed masses and widths; 2026-10-21;
      ``na49_mixing.baryon_table_corrections``), before the conjugate
      towers; ``False`` keeps the committed records;
    - ``baryon_pdg_widths``: the committed N(1440), N(1520), N(1680),
      N(1720) and Delta(1600) at their PDG 2025 Breit-Wigner width
      estimates, the masses kept (2026-10-21;
      ``na49_mixing.baryon_table_corrections``), before the conjugate
      towers; ``False`` keeps the committed widths;
    - ``hyperon_meson_partners``: beside every (Y, pi) class of a strange
      baryonic family the (Y, rho), (Y, omega) and (Y, eta) partners
      (2026-10-15; ``na49_mixing.channel_partners``); ``False`` keeps the
      committed classes;
    - ``delta_isoscalar_pairs``: the (Delta(1232), omega / eta / f0(500) /
      f0(980)) pairs at the four I = 3/2 families and their conjugates
      (2026-10-15; ``na49_mixing.channel_partners``); ``False`` keeps the
      committed classes;
    - ``charged_scalar_pairs``: the charge partners of the light scalar
      mesons' pairs in the charged mesonic pools, (f0(980), pi+-),
      (a0(980)+-, pi0), (a0(980)0, pi+-) (2026-10-15;
      ``na49_mixing.channel_partners``; needs the light scalar mesons).
    - ``xi_tower``: ``True`` appends the S = -2 excited baryonic continuum,
      an isospin doublet and its conjugates on the strange baryonic family's
      law, reached from the strange baryonic lines by continuation and
      decaying through (Xi, pi), (Xi(1530), pi), (Lambda, Kbar),
      (Sigma, Kbar) and (Sigma(1385), Kbar) (2026-10-30;
      ``na49_mixing.xi_tower``; needs xi_baryons and the conjugate towers);
      ``None`` or ``False`` keeps the kernel without it.
    - ``xi_baryons``: the Xi0, Xi- (stable, the weak decays) and Xi(1530)
      records at their PDG 2025 values, with the (Xi, K) and (Xi(1530), K)
      classes at the strange baryonic families (2026-10-16;
      ``na49_mixing.xi_baryons``), before the conjugate towers; ``False``
      keeps the committed catalog, which holds no baryon of strangeness -2.
    - ``pdg_meson_modes``: the rho0, omega, phi(1020), f2(1270), a2(1320),
      K1(1270) and K2*(1430) records at their PDG 2024 masses, widths and
      decay modes (2026-10-16; ``na49_mixing.meson_modes``), the four-body
      and the missing-state modes through their three-body final states;
      ``False`` keeps the committed records, which carry an older edition
      and only their two-body modes with catalog daughters.
    - ``three_body_endings``: the strength kappa_3 [GeV^-2] of the
      three-body endings (2026-10-20; ``na49_mixing.three_body_endings``):
      every family's three-body classes of stable hadrons (pi pi pi, K pi pi,
      N pi pi, Xi K K and the conjugates) join its direct pool, at the
      floor of the cascade, with the weight kappa_3 prod(2J+1) M R_3 / pi
      C_iso beside the two-body pairs; ``None`` keeps the two-body cascade.
    - ``dalitz_matrix_elements``: the three-body vertex draws the measured
      Dalitz distributions of omega -> pi+ pi- pi0 (the P-wave
      |p+ x p-|^2), eta -> 3 pi and eta' -> eta pi pi (2026-10-20;
      ``na49_mixing.dalitz_matrix_elements``, the law
      ``"phase_space_dalitz"``) on the four uniforms of the flat vertex;
      needs ``three_body_phase_space``; ``False`` keeps flat phase space.
    - ``nstar_range_rule``: the N* fractions of the isoscalar dipion at one
      common position inside their PDG ranges, set so that they sum to one
      (2026-10-16; ``isoscalar_dipion.range_position_fractions``), in
      place of the normalized midpoints, which put N(1650) -> N pi at 0.40
      against its listed 50-70%; needs the isoscalar dipion.
    - ``resonance_pair_line_shapes``: the running-width floor of the
      baryon-resonance pair channels (2026-10-21; the card
      TASK-20261021-r106-pbar-and-forward-piplus-mechanisms, piece P1a): a
      direct baryon-antibaryon pair of the family's
      ``resonance_pair_classes`` opens already from the daughters'
      threshold-mass sum (``mass_range[0]`` of the unstable N* / Delta*
      records, the pole mass of the stable p, n and their antibaryons,
      2.0115 GeV for (N, Nbar(1440)) and 2.1556 GeV for (Delta, Deltabar)
      against the pole sums 2.3783 and 2.464), the channel weight
      evaluating q* at the effective masses
      ``min(m_pole, (m_th + (M - m_other)) / 2)`` below the pole sum, and
      the unstable daughter of a drawn direct pair taking its mass from
      its line shape on ``[m_th, min(m_up, M - m_other)]``
      (``na49_mixing.cascade_decay._sibling_line_mass``); the
      ``_hold_pair_shares`` mechanism keeps the total baryon-pair share of
      the floor pool at the bare (p pbar, n nbar) share.  ``False`` keeps
      the pole-mass rule bit for bit.
    - ``direct_baryon_pairs_in_gap``: the direct baryon pairs of the
      continuation gap (2026-10-21; the same card, piece P1b): at a
      neutral mesonic vertex whose open continuation channels include no
      baryon pair (the (2.302, 2.987) GeV window where the pion
      continuation is open and the baryon-pair continuation is closed by
      ``continuation_q_min``), the open direct baryon-pair channels join
      the continuing draw pool of the topology cascades, scaled so their
      share of the pool equals the bare (p pbar, n nbar) share of the
      direct pool at the vertex mass; a drawn direct pair assembles both
      catalog daughters and terminates the branch.  ``False`` keeps the
      continuation-only pool bit for bit.
    - ``baryon_pair_s_wave``: the S-wave law of a baryon-pair creation
      step (2026-10-21; the same card, piece P3): the continuation step
      that creates a baryon-antibaryon pair from a mesonic state (B = 0
      parent, baryonic daughter family) at the lowest orbital wave a
      negative-parity cluster allows, L = 0 (the pair's parity is
      (-1)^(L+1); the continuum carries no J^P, so this is a convention,
      selected at the R106 optimum by -270 +- 39 and -189 +- 39 with the
      yields fixed); the daughter-mass density draw uses ``barrier_l = 0``
      (the plain q* law) in the place of the family's orbital barrier, on
      the same single uniform.
      ``False`` keeps the family's barrier bit for bit.
    - ``baryon_forward_hemisphere``: the forward-hemisphere baryon rule
      (2026-10-21; the same card, piece P4): at a non-split decay vertex
      of a baryonic state (B != 0) into one established baryon (|B| = 1)
      and one mesonic continuum daughter (B = 0), the baryon keeps the
      parent's forward hemisphere, ``cos(theta) = side * |2 u - 1|`` with
      ``side`` the sign of the parent's p_z (the antibaryon of an
      antibaryonic line likewise), so the mesonic cluster recoils toward
      central rapidity (rejected at the R106 optimum, +2,731 and +2,923 with
      the yields fixed, the protons +2,070 and +2,084); the
      same two kinematics uniforms.  ``False`` keeps the isotropic
      kinematics bit for bit.
    - ``baryon_pair_completion``: the baryon-antibaryon pairs of the mesonic
      continuum completed (2026-11-04; ``na49_mixing.baryon_pair_completion``):
      at every mesonic family every octet or decuplet pair (X, Ybar) whose
      (B, Q, S) sums to the family's, selected by the family's isospin
      factor, at the flavor ledger's gamma_s and r_BBbar, the decuplet pairs
      held at the octet pairs' share; "strange" adds the classes with a
      strange member (Lambda Lambdabar, Xi Xibar, Lambda nbar,
      Xi Lambdabar, ...), "strange_neutral" adds them at the neutral family
      only, "charged_nucleon" them at the neutral family and the non-strange
      pairs at the charged families (no strange family), "all" also the
      non-strange pairs of the charged families.  Needs
      the conjugate towers and the Xi baryons.  ``None`` keeps the committed
      pairs bit for bit.  ``baryon_pair_completion_members``: "octet_decuplet"
      (the default) or "octet" (the decuplet classes left out).
      ``baryon_pair_share``: ``None`` (the weight law), "universal" (at the
      completed families but the neutral one, the pairs that create no s-sbar
      take the neutral family's committed pair share at the same mass) or
      "line" (the universal share, offered at those families only on a
      cluster a baryonic line emits directly).
    - ``omega_baryon``: the Omega- (stable, the weak decays) at its PDG 2025
      mass, with the (Omega-, K) classes at the S = -2 continuum and their
      conjugates (2026-11-04; ``na49_mixing.omega_baryon``), the record before
      the conjugate towers; the pair completion then also offers the Omega
      pairs.  Needs the xi tower.  ``False`` keeps the kernel without it.
    - ``baryon_line_kappa``: the aligned continuing baryonic steps (2026-11-05):
      at a baryonic vertex that continues into a baryonic continuum daughter
      beside an established hadron, the continuing daughter's angle to the
      line's beam axis follows exp(kappa cos theta), so the line keeps its
      baryon leading through the cascade and the emitted hadron recoils
      toward the center; the continuation mass law is unchanged.  ``None``
      keeps the isotropic decay bit for bit.  ``baryon_line_kappa_scope``
      "nonstrange" aligns only the steps whose established hadron has S = 0
      (the strangeness-transfer steps keep the isotropic decay).
      ``baryon_line_kappa_axis`` "helicity" takes the angle about the
      parent's own direction of motion in place of the line's beam axis.
    - ``baryon_line_kappa_final``: the aligned final baryonic step (2026-11-05):
      at a baryonic vertex that ends in an established baryon beside a
      mesonic continuum daughter, the baryon's angle to the line's beam axis
      follows exp(kappa cos theta) (the graded form of the rejected hard
      hemisphere rule).  ``None`` keeps the isotropic decay bit for bit.
    - ``baryon_line_kappa_kaon``: the kaon-led direct pair (2026-11-06): at a
      baryonic continuum vertex that ends in an established hyperon beside an
      established strange meson (K or K*), the meson's angle to the line's
      beam axis follows exp(kappa cos theta) and the hyperon recoils (the
      forward K+ deficit at x_F 0.40-0.55, flat in p_T).  ``None`` keeps the
      isotropic decay bit for bit.
    - ``baryon_line_kappa_strange``: the strangeness-transfer step reversed
      (2026-11-05): at a continuing baryonic step whose established hadron
      carries strangeness (a kaon beside a hyperon cluster), the hadron's
      angle to the line's beam axis follows exp(kappa cos theta), so the kaon
      leads and the hyperon cluster recoils.  ``None`` keeps the step as the
      other options leave it.
    """

    continuation_floor: float | None = None
    continuation_ceiling: float | None = None
    continuation_mass_law: str | None = None
    continuation_barrier_l: int | None = None
    continuation_sibling_weight: str | None = None
    continuation_daughters: str | None = None
    continuation_t_h_alpha: float | None = None
    leading_split_fraction: float | None = None
    leading_split_kappa: float | None = None
    leading_split_slope: float | None = None
    leading_split_epsilon: float | None = None
    continuation_q_min: float | None = None
    sibling_line_width: float | None = None
    meson_split_fraction: float | None = None
    meson_split_slope: float | None = None
    meson_split_slope_strange: float | None = None
    string_step_mass: float | None = None
    string_step_slope: float | None = None
    string_step_epsilon: float | None = None
    # the strange established baryons in the leading-baryon split (2026-10-28, the forward Lambda and K+): ``None``
    # or ``True`` is the committed split (every established baryon); ``False`` takes the hyperons out of the split
    # channels with their share of its budget, so a root hands its strangeness over only through the ordinary
    # isotropic continuation and the non-strange leading baryons are drawn as before
    leading_split_strange: bool | None = None
    # the Regge-flux remainder law of the leading-baryon split (2026-10-28, the forward Lambda): ``None`` or ``False``
    # keeps the committed (mu^2)^(-1-epsilon) with the one epsilon (0: the Pomeron's law); ``True`` sets epsilon per
    # split channel from the trajectory the leading baryon exchanges with the root, epsilon = 2 alpha(0) - 2: the
    # f/omega family (0.5475, Donnachie-Landshoff) for a same-charge non-strange baryon, the pion (-0.93 m_pi^2) for
    # a charge exchange, the kaon (-0.93 m_K^2) for a strangeness exchange
    leading_split_regge: bool | None = None
    # the S = -2 excited baryonic continuum (2026-10-30, the Xi against NA61/SHINE): ``None`` or ``False`` keeps the
    # kernel without it; ``True`` appends the (1, 0, -2) and (1, -1, -2) families and their conjugates, built on the
    # strange baryonic family, which the strange lines reach by continuation (``na49_mixing.xi_tower``)
    xi_tower: bool | None = None
    # the axis of the leading splits (2026-11-04, the code review grok47): ``None`` keeps the committed beam axis of
    # the c.m. frame read in the root's rest frame; "gottfried_jackson" takes the incoming baryon's direction in the
    # root's rest frame (the Gottfried-Jackson axis of the side), which differs once the root carries a transfer
    leading_split_axis: str | None = None
    isospin_completion: tuple = ()
    delta_lines: bool = False
    isovector_meson_pairs: bool = False
    three_body_phase_space: bool = False
    isovector_neutral_mesons: bool = False
    isoscalar_dipion: bool = False
    floor_pools: tuple = ()
    conjugate_towers: bool = False
    baryon_resonance_pairs: bool = False
    non_diffractive_isospin_mixture: bool = False
    light_scalar_mesons: bool = False
    nucleon_kstar_pairs: bool = False
    pdg_hyperons: bool = False
    eta_prime: bool = False
    hyperon_meson_partners: bool = False
    delta_isoscalar_pairs: bool = False
    charged_scalar_pairs: bool = False
    xi_baryons: bool = False
    pdg_meson_modes: bool = False
    three_body_endings: float | None = None
    nstar_range_rule: bool = False
    dalitz_matrix_elements: bool = False
    axial_isoscalars: bool = False
    excited_kaons: bool = False
    pdg_nucleon_resonances: bool = False
    pdg_nucleon_resonance_siblings: bool = False
    associated_kstar_pairs: bool = False
    baryon_table_corrections: bool = False
    baryon_pdg_widths: bool = False
    resonance_pair_line_shapes: bool = False
    direct_baryon_pairs_in_gap: bool = False
    baryon_pair_s_wave: bool = False
    baryon_forward_hemisphere: bool = False
    baryon_pair_completion: str | None = None
    baryon_pair_completion_members: str = "octet_decuplet"
    baryon_pair_share: str | None = None
    omega_baryon: bool = False
    closed_channel_reference: bool = False
    baryon_line_kappa: float | None = None
    baryon_line_kappa_scope: str = "all"
    baryon_line_kappa_final: float | None = None
    baryon_line_kappa_kaon: float | None = None
    baryon_line_kappa_strange: float | None = None
    baryon_line_kappa_axis: str = "beam"

    def __post_init__(self) -> None:
        floor, ceiling = self.continuation_floor, self.continuation_ceiling
        if floor is not None and not (isinstance(floor, (int, float)) and math.isfinite(floor) and floor > 0.0):
            raise ValueError(f"KernelConfig: continuation_floor must be a positive finite mass, got {floor!r}")
        if ceiling is not None and not (isinstance(ceiling, (int, float)) and math.isfinite(ceiling) and ceiling >= SUPPORT_M_MAX):
            raise ValueError(f"KernelConfig: continuation_ceiling must be finite and at least {SUPPORT_M_MAX!r}, got {ceiling!r}")
        law = self.continuation_mass_law
        if law is not None and law not in CONTINUATION_MASS_LAWS:
            raise ValueError(f"KernelConfig: continuation_mass_law must be one of {CONTINUATION_MASS_LAWS!r}, got {law!r}")
        barrier = self.continuation_barrier_l
        if barrier is not None:
            if isinstance(barrier, bool) or not isinstance(barrier, int) or barrier < 0:
                raise ValueError(f"KernelConfig: continuation_barrier_l must be a non-negative integer, got {barrier!r}")
            if barrier > 0 and law != "density":
                raise ValueError("KernelConfig: a continuation barrier needs the density continuation law")
        sibling = self.continuation_sibling_weight
        if sibling is not None:
            if sibling not in CONTINUATION_SIBLING_WEIGHTS:
                raise ValueError(f"KernelConfig: continuation_sibling_weight must be one of "
                                 f"{CONTINUATION_SIBLING_WEIGHTS!r}, got {sibling!r}")
            if sibling == "density" and law != "density":
                raise ValueError("KernelConfig: the density sibling weight needs the density continuation law")
        sectors = self.isospin_completion
        if isinstance(sectors, str) or not isinstance(sectors, (tuple, list)):
            raise ValueError(f"KernelConfig: isospin_completion must be a sequence of sectors, got {sectors!r}")
        if len(set(sectors)) != len(sectors) or any(s not in ISOSPIN_SECTORS for s in sectors):
            raise ValueError(f"KernelConfig: isospin_completion sectors must be distinct members of {sorted(ISOSPIN_SECTORS)}, "
                             f"got {sectors!r}")
        object.__setattr__(self, "isospin_completion", tuple(sectors))
        if not isinstance(self.delta_lines, bool):
            raise ValueError(f"KernelConfig: delta_lines must be a bool, got {self.delta_lines!r}")
        if self.delta_lines and "baryonic" not in self.isospin_completion:
            raise ValueError("KernelConfig: delta_lines needs the 'baryonic' isospin completion")
        if not isinstance(self.isovector_meson_pairs, bool):
            raise ValueError(f"KernelConfig: isovector_meson_pairs must be a bool, got {self.isovector_meson_pairs!r}")
        # the n pbar and p nbar pairs continue into the baryonic families of
        # every charge (2026-10-07; review 2026-10-06, Muse M-3)
        if self.isovector_meson_pairs and "baryonic" not in self.isospin_completion:
            raise ValueError("KernelConfig: isovector_meson_pairs needs the 'baryonic' isospin completion")
        alpha = self.continuation_t_h_alpha
        if alpha is not None:
            if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0.0:
                raise ValueError(f"KernelConfig: continuation_t_h_alpha must be a non-negative finite number, got {alpha!r}")
            if law != "density":
                raise ValueError("KernelConfig: continuation_t_h_alpha needs the density continuation law")
            object.__setattr__(self, "continuation_t_h_alpha", float(alpha))
        q_min = self.continuation_q_min
        if q_min is not None:
            if isinstance(q_min, bool) or not isinstance(q_min, (int, float)) or not math.isfinite(q_min) or q_min < 0.0:
                raise ValueError(f"KernelConfig: continuation_q_min must be a non-negative finite momentum, got {q_min!r}")
            object.__setattr__(self, "continuation_q_min", float(q_min))
        line_width = self.sibling_line_width
        if line_width is not None:
            if (isinstance(line_width, bool) or not isinstance(line_width, (int, float)) or not math.isfinite(line_width)
                    or line_width <= 0.0):
                raise ValueError(f"KernelConfig: sibling_line_width must be a positive finite width, got {line_width!r}")
            object.__setattr__(self, "sibling_line_width", float(line_width))
        fraction = self.leading_split_fraction
        split = {name: getattr(self, name) for name in ("leading_split_kappa", "leading_split_slope", "leading_split_epsilon")}
        for name, value in (("leading_split_fraction", fraction), *split.items()):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"KernelConfig: {name} must be a finite number, got {value!r}")
        if fraction is None:
            if any(value is not None for value in split.values()):
                raise ValueError("KernelConfig: the leading-split options need leading_split_fraction")
        else:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"KernelConfig: leading_split_fraction must lie in (0, 1], got {fraction!r}")
            if law != "density":
                raise ValueError("KernelConfig: the leading-baryon split needs the density continuation law")
            if split["leading_split_kappa"] is not None and split["leading_split_slope"] is not None:
                raise ValueError("KernelConfig: set leading_split_kappa or leading_split_slope, not both")
            for name in ("leading_split_kappa", "leading_split_slope"):
                if split[name] is not None and split[name] < 0.0:
                    raise ValueError(f"KernelConfig: {name} must be non-negative, got {split[name]!r}")
            for name, value in (("leading_split_fraction", fraction), *split.items()):
                if value is not None:
                    object.__setattr__(self, name, float(value))
        strange = self.leading_split_strange
        if strange is not None and not isinstance(strange, bool):
            raise ValueError(f"KernelConfig: leading_split_strange must be a bool or None, got {strange!r}")
        if strange is False and fraction is None:
            raise ValueError("KernelConfig: leading_split_strange needs leading_split_fraction")
        regge = self.leading_split_regge
        if regge is not None and not isinstance(regge, bool):
            raise ValueError(f"KernelConfig: leading_split_regge must be a bool or None, got {regge!r}")
        if regge and fraction is None:
            raise ValueError("KernelConfig: leading_split_regge needs leading_split_fraction")
        if regge and split["leading_split_epsilon"] is not None:
            raise ValueError("KernelConfig: leading_split_regge sets the remainder law; drop leading_split_epsilon")
        step_mass = self.string_step_mass
        step = {name: getattr(self, name) for name in ("string_step_slope", "string_step_epsilon")}
        for name, value in (("string_step_mass", step_mass), *step.items()):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"KernelConfig: {name} must be a finite number, got {value!r}")
        if step_mass is None:
            if any(value is not None for value in step.values()):
                raise ValueError("KernelConfig: the string_step options need string_step_mass")
        else:
            if not step_mass > 0.0:
                raise ValueError(f"KernelConfig: string_step_mass must be positive, got {step_mass!r}")
            if law != "density":
                raise ValueError("KernelConfig: the string_step needs the density continuation law")
            if step["string_step_slope"] is not None and step["string_step_slope"] < 0.0:
                raise ValueError(f"KernelConfig: string_step_slope must be non-negative, got {step['string_step_slope']!r}")
            for name, value in (("string_step_mass", step_mass), *step.items()):
                if value is not None:
                    object.__setattr__(self, name, float(value))
        meson, meson_slope = self.meson_split_fraction, self.meson_split_slope
        for name, value in (("meson_split_fraction", meson), ("meson_split_slope", meson_slope)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f"KernelConfig: {name} must be a finite number, got {value!r}")
        if meson is None:
            if meson_slope is not None:
                raise ValueError("KernelConfig: meson_split_slope needs meson_split_fraction")
        else:
            if not 0.0 < meson < 1.0:
                raise ValueError(f"KernelConfig: meson_split_fraction must lie in (0, 1), got {meson!r}")
            if fraction is None:
                raise ValueError("KernelConfig: the leading-meson split needs the leading-baryon split")
            if meson_slope is not None and meson_slope < 0.0:
                raise ValueError(f"KernelConfig: meson_split_slope must be non-negative, got {meson_slope!r}")
            object.__setattr__(self, "meson_split_fraction", float(meson))
            if meson_slope is not None:
                object.__setattr__(self, "meson_split_slope", float(meson_slope))
        kaon_slope = self.meson_split_slope_strange
        if kaon_slope is not None:
            if isinstance(kaon_slope, bool) or not isinstance(kaon_slope, (int, float)) or not (
                    math.isfinite(float(kaon_slope)) and 0.0 <= float(kaon_slope) <= 100.0):
                raise ValueError(f"KernelConfig: meson_split_slope_strange must be a number in [0, 100], got {kaon_slope!r}")
            if meson is None:
                raise ValueError("KernelConfig: meson_split_slope_strange needs meson_split_fraction")
            object.__setattr__(self, "meson_split_slope_strange", float(kaon_slope))
        rule = self.continuation_daughters
        if rule is not None and rule not in CONTINUATION_DAUGHTER_RULES:
            raise ValueError(f"KernelConfig: continuation_daughters must be one of {CONTINUATION_DAUGHTER_RULES!r}, got {rule!r}")
        if not isinstance(self.isovector_neutral_mesons, bool):
            raise ValueError(f"KernelConfig: isovector_neutral_mesons must be a bool, got {self.isovector_neutral_mesons!r}")
        if self.isovector_neutral_mesons and rule != "isospin":
            raise ValueError("KernelConfig: the I = 1 neutral mesonic family needs continuation_daughters 'isospin'")
        if not isinstance(self.three_body_phase_space, bool):
            raise ValueError(f"KernelConfig: three_body_phase_space must be a bool, got {self.three_body_phase_space!r}")
        if not isinstance(self.isoscalar_dipion, bool):
            raise ValueError(f"KernelConfig: isoscalar_dipion must be a bool, got {self.isoscalar_dipion!r}")
        pools = self.floor_pools
        if isinstance(pools, str) or not isinstance(pools, (tuple, list)):
            raise ValueError(f"KernelConfig: floor_pools must be a sequence of sectors, got {pools!r}")
        if len(set(pools)) != len(pools) or any(s not in FLOOR_POOL_SECTORS for s in pools):
            raise ValueError(f"KernelConfig: floor_pools sectors must be distinct members of {list(FLOOR_POOL_SECTORS)}, "
                             f"got {pools!r}")
        object.__setattr__(self, "floor_pools", tuple(pools))
        if "strange" in self.floor_pools and "strange_mesonic" not in self.isospin_completion:
            raise ValueError("KernelConfig: the strange floor pool needs the 'strange_mesonic' isospin completion")
        if not isinstance(self.conjugate_towers, bool):
            raise ValueError(f"KernelConfig: conjugate_towers must be a bool, got {self.conjugate_towers!r}")
        if not isinstance(self.baryon_resonance_pairs, bool):
            raise ValueError(f"KernelConfig: baryon_resonance_pairs must be a bool, got {self.baryon_resonance_pairs!r}")
        if self.baryon_resonance_pairs and not self.conjugate_towers:
            raise ValueError("KernelConfig: the baryon-resonance pairs need the charge-conjugate towers")
        if not isinstance(self.non_diffractive_isospin_mixture, bool):
            raise ValueError(f"KernelConfig: non_diffractive_isospin_mixture must be a bool, "
                             f"got {self.non_diffractive_isospin_mixture!r}")
        if not isinstance(self.light_scalar_mesons, bool):
            raise ValueError(f"KernelConfig: light_scalar_mesons must be a bool, got {self.light_scalar_mesons!r}")
        if not isinstance(self.nucleon_kstar_pairs, bool):
            raise ValueError(f"KernelConfig: nucleon_kstar_pairs must be a bool, got {self.nucleon_kstar_pairs!r}")
        for name in ("hyperon_meson_partners", "delta_isoscalar_pairs", "charged_scalar_pairs", "xi_baryons", "pdg_meson_modes",
                     "nstar_range_rule", "resonance_pair_line_shapes", "direct_baryon_pairs_in_gap",
                     "baryon_pair_s_wave", "baryon_forward_hemisphere"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"KernelConfig: {name} must be a bool, got {getattr(self, name)!r}")
        if self.three_body_endings is not None and (isinstance(self.three_body_endings, bool) or not (
                isinstance(self.three_body_endings, (int, float)) and math.isfinite(float(self.three_body_endings))
                and float(self.three_body_endings) > 0.0)):
            raise ValueError(f"KernelConfig: three_body_endings must be None or a finite positive kappa_3 [GeV^-2], "
                             f"got {self.three_body_endings!r}")
        if not isinstance(self.eta_prime, bool):
            raise ValueError(f"KernelConfig: eta_prime must be a bool, got {self.eta_prime!r}")
        if not isinstance(self.pdg_nucleon_resonance_siblings, bool):
            raise ValueError("KernelConfig: pdg_nucleon_resonance_siblings must be a bool, got "
                             f"{self.pdg_nucleon_resonance_siblings!r}")
        if self.pdg_nucleon_resonance_siblings and not self.pdg_nucleon_resonances:
            raise ValueError("KernelConfig: pdg_nucleon_resonance_siblings needs pdg_nucleon_resonances")
        if not isinstance(self.associated_kstar_pairs, bool):
            raise ValueError(f"KernelConfig: associated_kstar_pairs must be a bool, got {self.associated_kstar_pairs!r}")
        if not isinstance(self.baryon_table_corrections, bool):
            raise ValueError(f"KernelConfig: baryon_table_corrections must be a bool, got {self.baryon_table_corrections!r}")
        if not isinstance(self.baryon_pdg_widths, bool):
            raise ValueError(f"KernelConfig: baryon_pdg_widths must be a bool, got {self.baryon_pdg_widths!r}")
        if not isinstance(self.pdg_nucleon_resonances, bool):
            raise ValueError(f"KernelConfig: pdg_nucleon_resonances must be a bool, got {self.pdg_nucleon_resonances!r}")
        if self.pdg_nucleon_resonances and not (self.isoscalar_dipion and self.light_scalar_mesons and self.eta_prime):
            raise ValueError("KernelConfig: pdg_nucleon_resonances needs isoscalar_dipion, light_scalar_mesons and eta_prime")
        if not isinstance(self.axial_isoscalars, bool):
            raise ValueError(f"KernelConfig: axial_isoscalars must be a bool, got {self.axial_isoscalars!r}")
        if self.axial_isoscalars and not self.light_scalar_mesons:
            # the f1(1285) modes name the a0(980) charge states (the code review glm53f F10: a bare KeyError before)
            raise ValueError("KernelConfig: axial_isoscalars needs light_scalar_mesons (the f1(1285) modes name the "
                             "a0(980) records)")
        if not isinstance(self.excited_kaons, bool):
            raise ValueError(f"KernelConfig: excited_kaons must be a bool, got {self.excited_kaons!r}")
        if not isinstance(self.dalitz_matrix_elements, bool):
            raise ValueError(f"KernelConfig: dalitz_matrix_elements must be a bool, got {self.dalitz_matrix_elements!r}")
        if self.dalitz_matrix_elements and not self.three_body_phase_space:
            raise ValueError("KernelConfig: dalitz_matrix_elements needs three_body_phase_space")
        if not isinstance(self.pdg_hyperons, bool):
            raise ValueError(f"KernelConfig: pdg_hyperons must be a bool, got {self.pdg_hyperons!r}")
        if self.pdg_hyperons and not ("baryonic" in self.isospin_completion and "strange_baryonic" in self.isospin_completion
                                      and self.delta_lines):
            raise ValueError("KernelConfig: pdg_hyperons needs the baryonic and strange baryonic isospin completion and the "
                             "Delta lines (its classes join their families)")
        if self.nstar_range_rule and not self.isoscalar_dipion:
            raise ValueError("KernelConfig: nstar_range_rule needs the isoscalar dipion (its N* tables)")
        if self.leading_split_axis not in (None, "gottfried_jackson"):
            raise ValueError(f"KernelConfig: leading_split_axis must be None or 'gottfried_jackson', "
                             f"got {self.leading_split_axis!r}")
        tower = self.xi_tower
        if tower is not None and not isinstance(tower, bool):
            raise ValueError(f"KernelConfig: xi_tower must be a bool or None, got {tower!r}")
        if tower and not (self.xi_baryons and self.conjugate_towers):
            raise ValueError("KernelConfig: xi_tower needs xi_baryons and the conjugate towers (its classes name the Xi "
                             "baryons and the antihyperons)")
        completion = self.baryon_pair_completion
        if completion is not None and completion not in PAIR_COMPLETION_SCOPES:
            raise ValueError(f"KernelConfig: baryon_pair_completion must be None or one of {PAIR_COMPLETION_SCOPES}, "
                             f"got {completion!r}")
        if self.baryon_pair_completion_members not in PAIR_COMPLETION_MEMBERS:
            raise ValueError(f"KernelConfig: baryon_pair_completion_members must be one of {PAIR_COMPLETION_MEMBERS}, "
                             f"got {self.baryon_pair_completion_members!r}")
        if self.baryon_pair_share not in PAIR_COMPLETION_SHARES:
            raise ValueError(f"KernelConfig: baryon_pair_share must be one of {PAIR_COMPLETION_SHARES}, "
                             f"got {self.baryon_pair_share!r}")
        if self.baryon_pair_share is not None and completion is None:
            raise ValueError("KernelConfig: baryon_pair_share needs baryon_pair_completion")
        if completion is not None and not (self.xi_baryons and self.conjugate_towers):
            raise ValueError("KernelConfig: baryon_pair_completion needs xi_baryons and the conjugate towers (its classes "
                             "name the Xi baryons and the antibaryons)")
        if not isinstance(self.closed_channel_reference, bool):
            raise ValueError(f"KernelConfig: closed_channel_reference must be a bool, got {self.closed_channel_reference!r}")
        if self.baryon_line_kappa is not None and (isinstance(self.baryon_line_kappa, bool)
                                                   or not isinstance(self.baryon_line_kappa, (int, float))
                                                   or not (math.isfinite(float(self.baryon_line_kappa))
                                                           and 0.0 < float(self.baryon_line_kappa) <= 50.0)):
            raise ValueError(f"KernelConfig: baryon_line_kappa must be None or a number in (0, 50], got {self.baryon_line_kappa!r}")
        if self.baryon_line_kappa_kaon is not None and (isinstance(self.baryon_line_kappa_kaon, bool)
                                                        or not isinstance(self.baryon_line_kappa_kaon, (int, float))
                                                        or not (math.isfinite(float(self.baryon_line_kappa_kaon))
                                                                and 0.0 < float(self.baryon_line_kappa_kaon) <= 50.0)):
            raise ValueError(f"KernelConfig: baryon_line_kappa_kaon must be None or a number in (0, 50], got {self.baryon_line_kappa_kaon!r}")
        if self.baryon_line_kappa_final is not None and (isinstance(self.baryon_line_kappa_final, bool)
                                                         or not isinstance(self.baryon_line_kappa_final, (int, float))
                                                         or not (math.isfinite(float(self.baryon_line_kappa_final))
                                                                 and 0.0 < float(self.baryon_line_kappa_final) <= 50.0)):
            raise ValueError(f"KernelConfig: baryon_line_kappa_final must be None or a number in (0, 50], got {self.baryon_line_kappa_final!r}")
        if self.baryon_line_kappa_strange is not None and (isinstance(self.baryon_line_kappa_strange, bool)
                                                           or not isinstance(self.baryon_line_kappa_strange, (int, float))
                                                           or not (math.isfinite(float(self.baryon_line_kappa_strange))
                                                                   and 0.0 < float(self.baryon_line_kappa_strange) <= 50.0)):
            raise ValueError(f"KernelConfig: baryon_line_kappa_strange must be None or a number in (0, 50], got {self.baryon_line_kappa_strange!r}")
        if self.baryon_line_kappa_axis not in ("beam", "helicity"):
            raise ValueError(f"KernelConfig: baryon_line_kappa_axis must be 'beam' or 'helicity', got {self.baryon_line_kappa_axis!r}")
        if self.baryon_line_kappa_scope not in ("all", "nonstrange"):
            raise ValueError(f"KernelConfig: baryon_line_kappa_scope must be 'all' or 'nonstrange', got {self.baryon_line_kappa_scope!r}")
        if not isinstance(self.omega_baryon, bool):
            raise ValueError(f"KernelConfig: omega_baryon must be a bool, got {self.omega_baryon!r}")
        if self.omega_baryon and not self.xi_tower:
            raise ValueError("KernelConfig: omega_baryon needs the xi tower (its classes join the S = -2 continuum)")
        if self.xi_baryons and "strange_baryonic" not in self.isospin_completion:
            raise ValueError("KernelConfig: xi_baryons needs the strange baryonic isospin completion (its classes join "
                             "the charged strange baryonic families)")

    def as_dict(self) -> dict:
        """The options as a JSON-ready mapping (for pre-registrations)."""
        return {**{name: getattr(self, name) for name in _OPTION_FIELDS}, "isospin_completion": list(self.isospin_completion),
                "delta_lines": self.delta_lines, "isovector_meson_pairs": self.isovector_meson_pairs,
                "three_body_phase_space": self.three_body_phase_space,
                "isovector_neutral_mesons": self.isovector_neutral_mesons, "isoscalar_dipion": self.isoscalar_dipion,
                "floor_pools": list(self.floor_pools), "conjugate_towers": self.conjugate_towers,
                "baryon_resonance_pairs": self.baryon_resonance_pairs,
                "non_diffractive_isospin_mixture": self.non_diffractive_isospin_mixture,
                "light_scalar_mesons": self.light_scalar_mesons, "nucleon_kstar_pairs": self.nucleon_kstar_pairs,
                "pdg_hyperons": self.pdg_hyperons, "eta_prime": self.eta_prime,
                "hyperon_meson_partners": self.hyperon_meson_partners, "delta_isoscalar_pairs": self.delta_isoscalar_pairs,
                "charged_scalar_pairs": self.charged_scalar_pairs, "xi_baryons": self.xi_baryons,
                "pdg_meson_modes": self.pdg_meson_modes, "nstar_range_rule": self.nstar_range_rule,
                "three_body_endings": self.three_body_endings,
                "dalitz_matrix_elements": self.dalitz_matrix_elements, "axial_isoscalars": self.axial_isoscalars,
                "pdg_nucleon_resonances": self.pdg_nucleon_resonances,
                "associated_kstar_pairs": self.associated_kstar_pairs, "excited_kaons": self.excited_kaons,
                "xi_tower": self.xi_tower,
                "baryon_table_corrections": self.baryon_table_corrections,
                "baryon_pdg_widths": self.baryon_pdg_widths,
                "pdg_nucleon_resonance_siblings": self.pdg_nucleon_resonance_siblings,
                "resonance_pair_line_shapes": self.resonance_pair_line_shapes,
                "direct_baryon_pairs_in_gap": self.direct_baryon_pairs_in_gap,
                "baryon_pair_s_wave": self.baryon_pair_s_wave,
                "baryon_forward_hemisphere": self.baryon_forward_hemisphere,
                "baryon_pair_completion": self.baryon_pair_completion,
                "baryon_pair_completion_members": self.baryon_pair_completion_members,
                "baryon_pair_share": self.baryon_pair_share, "omega_baryon": self.omega_baryon,
                "closed_channel_reference": self.closed_channel_reference, "baryon_line_kappa": self.baryon_line_kappa,
                "baryon_line_kappa_scope": self.baryon_line_kappa_scope, "baryon_line_kappa_final": self.baryon_line_kappa_final,
                "baryon_line_kappa_kaon": self.baryon_line_kappa_kaon,
                "baryon_line_kappa_strange": self.baryon_line_kappa_strange,
                "baryon_line_kappa_axis": self.baryon_line_kappa_axis}


LEGACY = KernelConfig()
R74 = KernelConfig(continuation_floor=1.7, continuation_ceiling=17.3, continuation_mass_law="density", continuation_barrier_l=1)
PROFILES = {"legacy": LEGACY, "r74": R74}


def apply_kernel_config(fams: list[ContinuumFamily], config: KernelConfig) -> list[ContinuumFamily]:
    """The families with the configuration's options set on every family.

    A family that already carries an option is refused: the committed
    configuration sets none, and the options come from the run.  Under
    :data:`LEGACY` the families come back equal to the input.
    """
    for family in fams:
        for name in _OPTION_FIELDS:
            if getattr(family, name, None) is not None:
                raise ValueError(
                    f"apply_kernel_config: family {family.family_id!r} already sets {name} = "
                    f"{getattr(family, name)!r}; the committed configuration sets no kernel option")
    options = {name: getattr(config, name) for name in _OPTION_FIELDS}
    if options["continuation_floor"] is not None:
        options["continuation_floor"] = float(options["continuation_floor"])
    if options["continuation_ceiling"] is not None:
        options["continuation_ceiling"] = float(options["continuation_ceiling"])
    return [dataclasses.replace(family, **options) for family in fams]


def load_kernel(config: KernelConfig, nested_root) -> tuple[HadronCatalog, list[ContinuumFamily]]:
    """The committed catalog and the committed continuum families with the
    configuration applied (``nested_root`` is the ``dev/mixing2006`` root);
    the isospin-completion families follow the committed ones, the
    Delta-like line families follow those, and all carry the same
    options.  The isoscalar dipion revises the catalog and the committed
    families' direct pools before the completion, which rotates its floor
    class to the charge partners; the light scalar mesons follow it with
    their records and classes.  The floor pools apply next, to the
    committed and the completion families alike.  The charge-conjugate
    towers apply next: the catalog's antibaryons and the conjugates of
    every baryonic family, the floor pools' classes included.  The
    baryon-resonance pairs apply next, on the conjugate catalog.  The
    (N, Kbar*) pairs join every strange baryonic family next, the
    conjugate towers included.  The non-diffractive isospin mixture marks
    the B != 0 families last.  The PDG hyperons join after the Delta lines
    and before the floor pools, so the conjugate towers carry them."""
    root = Path(nested_root)
    catalog = HadronCatalog.from_json(root / CATALOG_REL)
    committed = families_from_json(str(root / FAMILIES_REL))
    if config.isoscalar_dipion:
        dipion = load_isoscalar_dipion(root)
        catalog = isoscalar_dipion_catalog(catalog, dipion, range_rule=config.nstar_range_rule)
        committed = isoscalar_dipion_families(committed, dipion)
    if config.light_scalar_mesons:
        scalars = load_light_scalar_mesons(root)
        catalog = light_scalar_mesons_catalog(catalog, scalars)
        committed = light_scalar_mesons_families(committed, scalars)
    if config.pdg_meson_modes:
        # the PDG 2024 tables of the light mesons (2026-10-16): the committed
        # records replaced, the families and classes unchanged
        catalog = meson_modes_catalog(catalog, load_meson_modes(root))
    if config.three_body_phase_space:
        catalog = catalog.with_three_body_law("phase_space_dalitz" if config.dalitz_matrix_elements else "phase_space")
    fams = apply_kernel_config(committed, config)
    if config.isospin_completion:
        fams = fams + isospin_partner_families(fams, catalog, config.isospin_completion)
    if config.delta_lines:
        fams = fams + delta_line_families(fams, catalog)
    if config.isovector_meson_pairs:
        fams = isovector_meson_pair_families(fams)
    if config.isovector_neutral_mesons:
        fams = fams + isovector_neutral_mesonic_families(fams)
    if config.pdg_hyperons:
        # the PDG hyperon resonances (2026-10-15): their records and classes before the
        # floor pools and the conjugate towers, which conjugate both
        hyperons = load_pdg_hyperons(root)
        catalog = pdg_hyperons_catalog(catalog, hyperons)
        fams = pdg_hyperons_families(fams, hyperons, catalog)
    if config.xi_baryons:
        # the S = -2 baryons (2026-10-16): their records and classes before the conjugate towers
        catalog = xi_baryons_catalog(catalog, load_xi_baryons(root))
        fams = xi_baryons_families(fams)
    if config.omega_baryon:
        # the S = -3 baryon (2026-11-04): its record before the conjugate towers, which build the anti-Omega+
        catalog = omega_baryon_catalog(catalog, load_omega_baryon(root))
    if config.pdg_nucleon_resonances:
        # the PDG N and Delta resonances up to the floor (2026-10-21): their records and (N*, pi) classes before
        # the floor pools and the conjugate towers, which conjugate both
        nres = load_pdg_nucleon_resonances(root)
        catalog = pdg_nucleon_resonances_catalog(catalog, nres)
        if config.pdg_nucleon_resonance_siblings:
            fams = pdg_nucleon_resonances_families(fams, catalog, nres)
    if config.baryon_table_corrections:
        # the PDG 2025 Delta(1600) and Delta(1700) tables (2026-10-21): the committed records replaced before the
        # conjugate towers, which conjugate them
        catalog = baryon_table_corrections_catalog(catalog, load_baryon_table_corrections(root))
    if config.baryon_pdg_widths:
        # the PDG 2025 widths of the committed N* and Delta records (2026-10-21), before the conjugate towers
        catalog = baryon_pdg_widths_catalog(catalog, load_baryon_table_corrections(root))
    if config.floor_pools:
        fams = floor_pool_families(fams, catalog, config.floor_pools)
    if config.conjugate_towers:
        catalog = conjugate_catalog(catalog)
        fams = fams + conjugate_baryon_families(fams, catalog)
    if config.baryon_resonance_pairs:
        fams = resonance_pair_families(fams, catalog)
    if config.nucleon_kstar_pairs:
        fams = nucleon_kstar_families(fams, catalog)
    if config.hyperon_meson_partners:
        # the (Y, rho/omega/eta) partners of the strange baryonic lines (2026-10-15)
        fams = hyperon_meson_partner_families(fams, catalog)
    if config.delta_isoscalar_pairs:
        # the (Delta, I = 0 meson) pairs of the I = 3/2 lines (2026-10-15)
        fams = delta_isoscalar_pair_families(fams, catalog)
    if config.charged_scalar_pairs:
        # the charge partners of the light scalar mesons' pairs (2026-10-15)
        fams = charged_scalar_pair_families(fams, catalog)
    if config.eta_prime:
        # the eta'(958) (2026-10-15): its record and the eta' analog of every eta class, at the
        # end of the load as the probe adds them
        catalog = eta_prime_catalog(catalog, load_eta_prime(root))
        fams = eta_prime_families(fams)
    if config.axial_isoscalars:
        # the f1(1285) and h1(1170) (2026-10-21): their records and the classes beside every (omega, pi) class,
        # at the end of the load as the eta' enters
        catalog = axial_isoscalars_catalog(catalog, load_axial_isoscalars(root))
        fams = axial_isoscalars_families(fams)
    if config.associated_kstar_pairs:
        # the (Y, K*(892)) partners of the associated pairs (2026-10-21), the conjugate towers included
        fams = associated_kstar_families(fams, catalog)
    if config.excited_kaons:
        # the excited kaons (2026-10-28): their records and the classes beside every baryon-kaon class, after the
        # (Y, K*(892)) partners and the conjugate towers
        catalog = excited_kaons_catalog(catalog, load_excited_kaons(root))
        fams = excited_kaons_families(fams, catalog)
    if config.three_body_endings is not None:
        # the three-body endings (2026-10-20): every family's three-body classes of stable hadrons, the conjugate
        # towers included, at the strength kappa_3
        fams = three_body_families(fams, catalog, tuple(THREE_BODY_SETS), float(config.three_body_endings))
    if config.non_diffractive_isospin_mixture:
        fams = [dataclasses.replace(f, non_diffractive_isospin_mixture=True) if int(f.qn[0]) != 0 else f
                for f in fams]
    if config.xi_tower:
        # the S = -2 excited baryonic continuum (2026-10-30): its families at the end of the load, built on the strange
        # baryonic family as the probe hook PROBE_XI_TOWER built them
        fams = xi_tower_families(catalog, fams)
    if config.omega_baryon:
        # the (Omega-, K) classes of the S = -2 continuum and their conjugates (2026-11-04)
        fams = omega_baryon_families(fams, load_omega_baryon(root))
    if config.baryon_pair_completion is not None:
        # the baryon-pair completion of the mesonic continuum (2026-11-04): every mesonic family's octet and decuplet
        # pairs by (B, Q, S), after the Xi tower so every pair's continuation families are loaded
        fams = baryon_pair_completion_families(fams, catalog, config.baryon_pair_completion,
                                               config.baryon_pair_completion_members, config.baryon_pair_share)
    if config.closed_channel_reference:
        # the channels closed at the pole and open above threshold (2026-11-04): a run attribute of the catalog, set
        # last so no rebuild drops it
        catalog = catalog.with_closed_channel_reference(True)
    if config.baryon_line_kappa is not None:
        # the aligned continuing baryonic steps (2026-11-05): every family carries the strength, the cascade reads it at
        # a baryonic vertex that continues into a baryonic continuum daughter beside an established hadron
        fams = [dataclasses.replace(f, baryon_line_kappa=float(config.baryon_line_kappa),
                                    baryon_line_kappa_scope=config.baryon_line_kappa_scope,
                                    baryon_line_kappa_axis=config.baryon_line_kappa_axis) for f in fams]
    if config.baryon_line_kappa_strange is not None:
        # the strangeness-transfer step reversed (2026-11-05): the emitted strange hadron along the line
        fams = [dataclasses.replace(f, baryon_line_kappa_strange=float(config.baryon_line_kappa_strange)) for f in fams]
    if config.baryon_line_kappa_kaon is not None:
        # the kaon-led direct pair (2026-11-06): the strange meson beside an established hyperon along the line
        fams = [dataclasses.replace(f, baryon_line_kappa_kaon=float(config.baryon_line_kappa_kaon)) for f in fams]
    if config.baryon_line_kappa_final is not None:
        # the aligned final baryonic step (2026-11-05): the established baryon of a line's last step along the line
        fams = [dataclasses.replace(f, baryon_line_kappa_final=float(config.baryon_line_kappa_final)) for f in fams]
    mechanism_flags = {"resonance_pair_line_shapes": config.resonance_pair_line_shapes,
                       "direct_baryon_pairs_in_gap": config.direct_baryon_pairs_in_gap,
                       "baryon_pair_s_wave": config.baryon_pair_s_wave,
                       "baryon_forward_hemisphere": config.baryon_forward_hemisphere}
    if any(mechanism_flags.values()):
        # the R106 mechanism probes (2026-10-21; the card
        # TASK-20261021-r106-pbar-and-forward-piplus-mechanisms, pieces P1a,
        # P1b, P3 and P4): the cascade reads the options from the families,
        # so every family of the loaded kernel carries them; the committed
        # configuration and every flag-off kernel keep False
        fams = [dataclasses.replace(f, **mechanism_flags) for f in fams]
    return catalog, fams
