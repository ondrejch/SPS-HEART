"""Continuum hadronic flavor weights (Round 6, task 37; plan section 12).

Plan section 12.2 weights a hadronic final channel ``c`` of a parent ``H``
by

    w_c(H) proportional to C_iso C_spin Phi_c(M_H) F_c(M_H)
                           (gamma_s^had)^(n_ssbar,c) (r_V)^(n_V,c)
                           (r_BBbar)^(n_BBbar,c),

normalized over the open channels of the parent.  The Round 4 channel
enumeration (:func:`na49_mixing.cascade_decay.enumerate_channels`)
already carries ``C_iso C_spin Phi_c F_c`` in ``channel.weight``; this
module supplies the flavor factor and the re-normalization.  Plan section
12.2 also fixes the scope: "Preserve measured branching fractions for
established resonances; apply fitted effective weights primarily to
continuum-state decays", so the factor applies to continuum parents only
and a catalog parent keeps its Round 2 branching fractions.

Parameters (plan section 12.1)
------------------------------
- ``gamma_s_had``: suppression of channels containing newly produced
  strange-antistrange content.  The exponent ``n_ssbar`` counts the
  strange quanta created at the vertex, ``(sum_i |S_i| - |S_H|) / 2``
  over the daughters with the weak neutral kaons K0S and K0L counted as
  one strange quantum each (they carry S = 0 in the catalog's
  weak-eigenstate bookkeeping but are strange mesons in flavor), plus
  the hidden strange content of daughters that carry an s-sbar pair
  without net strangeness (:data:`DEFAULT_HIDDEN_SSBAR_CONTENT`: the phi
  counts 1.0 under ideal mixing, the eta'(958) 0.6, its quark-model s-sbar
  fraction, and the eta 0.0 at the nominal mixing).  The optional
  singlet/octet mixing nuisance of plan
  section 12.1 is the override of that content mapping and is
  constrained by external information, never fit here.
- ``r_v``: vector-to-pseudoscalar mesonic channel weight; ``n_V`` counts
  the established vector mesons among the daughters (a catalog meson
  with J = 1 and negative parity: rho, omega, phi, K*(892), and their
  radial excitations).
- ``r_bbar``: suppression of baryon-antibaryon channels above threshold;
  ``n_BBbar = (sum_i |B_i| - |B_H|) / 2`` counts the baryon pairs
  created at the vertex.
- ``assoc_norm``: the associated-strangeness normalization of plan
  section 12.1, applied ``once`` to a channel of a non-strange baryonic
  parent that produces a hyperon together with a kaon (N* -> K Lambda,
  N* -> K Sigma); 1.0 unless catalog branching and phase space prove
  insufficient (plan line 959).

These are effective hadronic channel weights, never string-breaking
probabilities (plan line 962).  :data:`NOMINAL_FLAVOR` (every factor 1.0)
is the identity: :func:`apply_flavor_weights` returns the enumerated
channels unchanged, object for object, so every Round 2 to 5 trace is
reproduced bit for bit when no flavor parameters are supplied.  The
weights are pure: no random number is consumed and the per-vertex draw
budget is unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

__all__ = [
    "exponent_key",
    "key_weight",
    "vertex_ledger_record",
    "ledger_log_weight",
    "CATALOG_FAMILY_PREFIX",
    "CHANNEL_FACTOR_DAUGHTERS",
    "DEFAULT_HIDDEN_SSBAR_CONTENT",
    "FlavorExponents",
    "FlavorParams",
    "NOMINAL_FLAVOR",
    "STRANGE_QUANTA_OVERRIDE",
    "apply_flavor_weights",
    "flavor_exponents",
    "flavor_weight",
]

#: The catalog-parent / catalog-daughter identifier prefix of the cascade
#: channel records (kept equal to ``cascade_decay.CATALOG_FAMILY_PREFIX``;
#: the test suite pins the equality so this module needs no import of the
#: cascade module).
CATALOG_FAMILY_PREFIX = "catalog:"

#: Hidden strange-antistrange content per established daughter (ideal
#: mixing: the phi is a pure s-sbar state; the eta carries no hidden
#: content at the nominal mixing nuisance).  The eta'(958) (the
#: ``KernelConfig.eta_prime`` record, 2026-10-15) carries 0.6, its
#: quark-model s-sbar fraction cos^2(phi_P) at the mixing angle
#: phi_P = 39 degrees; inert without the record.
DEFAULT_HIDDEN_SSBAR_CONTENT: Mapping[int, float] = {333: 1.0, 331: 0.6}

#: Strange quanta of daughters whose catalog strangeness is not their
#: flavor strangeness: the weak neutral kaons (K0S, K0L) carry S = 0 in
#: the weak-eigenstate bookkeeping and one strange quantum in flavor.
STRANGE_QUANTA_OVERRIDE: Mapping[int, int] = {310: 1, 130: 1}

#: The channel factors of the diagnostic fit (2026-10-17; owner: "ok
#: approve 1 2 3", the model decision register): one free factor per set
#: of established daughters (absolute pdg ids), on top of the four plan
#: factors.  ``w_omega`` the omega, ``w_rho`` the rho (both charges),
#: ``w_etap`` the eta'(958), ``w_lowy`` the Lambda(1520) and Sigma(1385).
#: They are effective factors, outside the model: a FlavorParams that
#: names one in ``channel_factors`` weights each channel by the factor to
#: the power of its count of those daughters, and its ledger keys carry the
#: counts after the four plan exponents, in this order.  Without them every
#: key and weight is the plan's.
CHANNEL_FACTOR_DAUGHTERS: Mapping[str, frozenset] = {
    "w_omega": frozenset({223}),
    "w_rho": frozenset({113, 213}),
    "w_etap": frozenset({331}),
    "w_lowy": frozenset({102134, 3224, 3214, 3114}),
}


def _positive_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not (math.isfinite(value_f) and value_f > 0.0):
        raise ValueError(
            f"FlavorParams.{name} must be a positive finite factor, got {value!r}"
        )
    return value_f


@dataclass(frozen=True)
class FlavorParams:
    """The plan section 12.1 flavor parameters (effective channel weights)."""

    gamma_s_had: float = 1.0
    r_v: float = 1.0
    r_bbar: float = 1.0
    assoc_norm: float = 1.0
    hidden_ssbar_content: Mapping[int, float] = field(
        default_factory=lambda: dict(DEFAULT_HIDDEN_SSBAR_CONTENT)
    )
    channel_factors: Mapping[str, float] = field(default_factory=dict)
    #: r_BBbar on the pair steps that continue only (2026-10-25): a baryon-pair channel without a continuum daughter
    #: (the direct pairs of a floor state, p pbar and n nbar of a neutral mesonic state at 2.0-2.3 GeV) keeps its bare
    #: two-body weight; the default applies r_BBbar to every channel that creates a pair
    bbar_continuation_only: bool = False

    def __post_init__(self) -> None:
        for name in ("gamma_s_had", "r_v", "r_bbar", "assoc_norm"):
            _positive_finite(name, getattr(self, name))
        if not isinstance(self.bbar_continuation_only, bool):
            raise ValueError(f"FlavorParams.bbar_continuation_only must be a bool, got {self.bbar_continuation_only!r}")
        factors = dict(self.channel_factors)
        unknown = sorted(set(factors) - set(CHANNEL_FACTOR_DAUGHTERS))
        if unknown:
            raise ValueError(
                f"FlavorParams.channel_factors: unknown factors {unknown}; "
                f"the set is {sorted(CHANNEL_FACTOR_DAUGHTERS)}"
            )
        object.__setattr__(self, "channel_factors", {
            name: _positive_finite(name, factors[name])
            for name in CHANNEL_FACTOR_DAUGHTERS if name in factors})
        content = {}
        for pdg, value in dict(self.hidden_ssbar_content).items():
            value_f = float(value)
            if not (math.isfinite(value_f) and 0.0 <= value_f <= 2.0):
                raise ValueError(
                    "FlavorParams.hidden_ssbar_content values must lie in "
                    f"[0, 2] (strange quanta per daughter), got {pdg}: {value!r}"
                )
            content[int(pdg)] = value_f
        object.__setattr__(self, "hidden_ssbar_content", content)

    @property
    def channel_names(self) -> tuple:
        """The named channel factors in :data:`CHANNEL_FACTOR_DAUGHTERS`
        order: the extra ledger key entries."""
        return tuple(self.channel_factors)

    def is_identity(self) -> bool:
        """True when every factor equals 1.0 (the weights are then exactly 1)."""
        return (
            self.gamma_s_had == 1.0 and self.r_v == 1.0
            and self.r_bbar == 1.0 and self.assoc_norm == 1.0
            and all(v == 1.0 for v in self.channel_factors.values())
        )


#: The identity flavor point: every effective weight 1.0.
NOMINAL_FLAVOR: FlavorParams = FlavorParams()


@dataclass(frozen=True)
class FlavorExponents:
    """The per-channel exponents of plan section 12.2."""

    n_ssbar: float
    n_v: int
    n_bbar: int
    n_assoc: int
    n_channel: tuple = ()


def _daughter_pdg(daughter_id: str) -> Optional[int]:
    """The catalog pdg id of an established daughter, None for a
    continuum daughter (identified by its family id)."""
    text = str(daughter_id)
    if text.startswith(CATALOG_FAMILY_PREFIX):
        return int(text[len(CATALOG_FAMILY_PREFIX):])
    return None


def _family_qn(fams, family_id: str) -> tuple[int, int, int]:
    for family in fams:
        if str(family.family_id) == str(family_id):
            return tuple(int(v) for v in family.qn)  # type: ignore[return-value]
    raise ValueError(
        f"flavor_weights: no continuum family record carries family_id "
        f"{family_id!r}"
    )


def _established_qn(catalog, pdg: int) -> tuple[int, int, int]:
    try:
        record = catalog.get(int(pdg))
        return (int(record.B), int(record.Q), int(record.S))
    except KeyError:
        from .quantum_numbers import quantum_numbers

        qn = quantum_numbers(int(pdg))
        return (int(qn.B), int(qn.Q), int(qn.S))


def _is_vector_meson(catalog, pdg: int) -> bool:
    try:
        record = catalog.get(int(pdg))
    except KeyError:
        return False
    return int(record.B) == 0 and float(record.spin) == 1.0 and str(record.parity) == "-"


def flavor_exponents(
    parent_qn: Sequence[int],
    channel,
    catalog,
    fams,
    hidden_ssbar_content: Mapping[int, float] = DEFAULT_HIDDEN_SSBAR_CONTENT,
    channel_names: Sequence[str] = (),
    bbar_continuation_only: bool = False,
) -> FlavorExponents:
    """The plan section 12.2 exponents of one enumerated channel.

    *channel_names* (2026-10-17) adds the counts of the named channel
    factors' daughters (:data:`CHANNEL_FACTOR_DAUGHTERS`) as ``n_channel``.

    *bbar_continuation_only* (2026-10-25) counts a created baryon pair only
    on a channel with a continuum daughter: the direct pairs of a floor
    state then carry ``n_bbar = 0`` (:attr:`FlavorParams.bbar_continuation_only`).

    *parent_qn* is the parent's ``(B, Q, S)``; *channel* carries
    ``daughters`` as catalog identifiers ``"catalog:<pdg>"`` or continuum
    family ids.  Pure: no random number is consumed.
    """
    b_parent, _q_parent, s_parent = (int(parent_qn[0]), int(parent_qn[1]), int(parent_qn[2]))
    # an antibaryonic vertex (the charge-conjugate towers, 2026-10-09) reads
    # the strangeness signs conjugated: Lambdabar K- there counts as
    # Lambda K+ at a baryonic vertex
    sign = -1 if b_parent < 0 else 1
    strange_quanta = 0
    baryon_quanta = 0
    hidden = 0.0
    n_v = 0
    hyperon = False
    kaon = False
    for daughter_id in channel.daughters:
        pdg = _daughter_pdg(daughter_id)
        if pdg is None:
            b_d, _q_d, s_d = _family_qn(fams, daughter_id)
            strange_quanta += abs(s_d)
            baryon_quanta += abs(b_d)
            if b_d != 0 and sign * s_d < 0:
                hyperon = True
            if b_d == 0 and sign * s_d > 0:
                kaon = True
            continue
        b_d, _q_d, s_d = _established_qn(catalog, pdg)
        quanta = STRANGE_QUANTA_OVERRIDE.get(abs(pdg), abs(s_d))
        strange_quanta += quanta
        baryon_quanta += abs(b_d)
        hidden += float(hidden_ssbar_content.get(abs(pdg), 0.0))
        if _is_vector_meson(catalog, pdg):
            n_v += 1
        if b_d != 0 and sign * s_d < 0:
            hyperon = True
        if b_d == 0 and (sign * s_d > 0 or (quanta > 0 and s_d == 0)):
            kaon = True
    created_strange = max(strange_quanta - abs(s_parent), 0) / 2.0
    n_bbar = max(baryon_quanta - abs(b_parent), 0) // 2
    if bbar_continuation_only and n_bbar and all(_daughter_pdg(d) is not None for d in channel.daughters):
        # a direct pair (every daughter established): the bare two-body weight (2026-10-25)
        n_bbar = 0
    n_assoc = int(b_parent != 0 and s_parent == 0 and hyperon and kaon)
    n_channel = ()
    if channel_names:
        pdgs = [abs(p) for p in (_daughter_pdg(d) for d in channel.daughters) if p is not None]
        n_channel = tuple(sum(1 for p in pdgs if p in CHANNEL_FACTOR_DAUGHTERS[name]) for name in channel_names)
    return FlavorExponents(
        n_ssbar=created_strange + hidden,
        n_v=int(n_v),
        n_bbar=int(n_bbar),
        n_assoc=n_assoc,
        n_channel=n_channel,
    )


def _channel_factor(counts: Sequence[int], params: FlavorParams) -> float:
    """The product of the named channel factors to the powers *counts*
    (the extra ledger key entries); the counts must match the named set."""
    names = params.channel_names
    if len(counts) != len(names):
        raise ValueError(
            f"flavor_weights: {len(counts)} channel-factor counts against "
            f"the named factors {names}")
    out = 1.0
    for name, n in zip(names, counts):
        if n:
            out *= params.channel_factors[name] ** int(n)
    return out


def flavor_weight(exponents: FlavorExponents, params: FlavorParams) -> float:
    """``gamma_s^n_ssbar r_V^n_V r_BBbar^n_BBbar assoc_norm^n_assoc``,
    times the named channel factors (2026-10-17)."""
    value = (
        params.gamma_s_had ** float(exponents.n_ssbar)
        * params.r_v ** int(exponents.n_v)
        * params.r_bbar ** int(exponents.n_bbar)
        * params.assoc_norm ** int(exponents.n_assoc)
    )
    if exponents.n_channel or params.channel_factors:
        value *= _channel_factor(exponents.n_channel, params)
    return value


def apply_flavor_weights(
    channels: Sequence,
    parent_qn: Sequence[int],
    catalog,
    fams,
    params: Optional[FlavorParams],
) -> list:
    """Re-weight enumerated continuum-parent channels by the flavor factor
    and re-normalize over the vertex.

    Returns the input list unchanged (same objects) when *params* is None
    or the identity, so the Round 4 and 5 traces are reproduced bit for
    bit; an empty enumeration is returned empty (the below-threshold
    state's no-channel rule is the caller's, at every flavor point).
    Otherwise returns new channel records with
    ``weight_i * f_i / sum_j weight_j f_j``.  A vertex whose weighted total
    vanishes raises ``ValueError`` (a degenerate flavor point, checked,
    never repaired).
    """
    if params is None or params.is_identity() or not channels:
        # the identity, or an empty enumeration (a below-threshold state:
        # the caller's no-channel rule applies exactly as at the identity)
        return list(channels)
    factors = [
        flavor_weight(
            flavor_exponents(parent_qn, channel, catalog, fams, params.hidden_ssbar_content,
                             params.channel_names, params.bbar_continuation_only),
            params,
        )
        for channel in channels
    ]
    raw = [float(channel.weight) * factor for channel, factor in zip(channels, factors)]
    total = math.fsum(raw)
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            "apply_flavor_weights: the flavor-weighted vertex total is not "
            f"finite and positive ({total!r}); the flavor point is degenerate"
        )
    return [
        dataclasses.replace(channel, weight=value / total)
        for channel, value in zip(channels, raw)
    ]

# ----------------------------------------------------------------------
# The flavor ledger (Round 6 runs R61 to R65): compact per-vertex
# sufficient statistics for reweighting a generated sample to another
# flavor point without regeneration (plan section 5.3: reweight when
# the parameter changes a density but not the kinematic support; the
# flavor factors are strictly positive, so every open channel stays
# open).
# ----------------------------------------------------------------------
def exponent_key(exponents: FlavorExponents) -> tuple:
    """The hashable ``(n_ssbar, n_V, n_BBbar, n_assoc)`` key of one
    channel's exponents (``n_ssbar`` rounded to six decimals, since the
    hidden-strangeness content may be fractional), followed by the named
    channel factors' counts when there are any (2026-10-17)."""
    return (
        round(float(exponents.n_ssbar), 6),
        int(exponents.n_v),
        int(exponents.n_bbar),
        int(exponents.n_assoc),
    ) + tuple(int(n) for n in exponents.n_channel)


def key_weight(key: tuple, params: FlavorParams) -> float:
    """The flavor factor of an exponent key at *params* (the entries after
    the fourth are the named channel factors' counts, 2026-10-17)."""
    value = (
        params.gamma_s_had ** float(key[0])
        * params.r_v ** int(key[1])
        * params.r_bbar ** int(key[2])
        * params.assoc_norm ** int(key[3])
    )
    if len(key) > 4 or params.channel_factors:
        value *= _channel_factor(key[4:], params)
    return value


def vertex_ledger_record(
    parent_qn: Sequence[int],
    pool: Sequence,
    chosen,
    catalog,
    fams,
    params: Optional[FlavorParams] = None,
):
    """The ledger record of one drawn continuum-parent vertex.

    *pool* is the channel list the draw ran over (the continuum pool or
    the direct pool, with the weights exactly as drawn, so at the
    generation flavor point), *chosen* the drawn channel.  Returns
    ``(chosen_key, {key: summed pool weight})`` or ``None`` when the
    pool spans one exponent key only (the reweighting factor of such a
    vertex is identically one at every flavor point).  Pure: no random
    number is consumed and the pool is not modified.
    """
    hidden = (DEFAULT_HIDDEN_SSBAR_CONTENT if params is None
              else params.hidden_ssbar_content)
    names = () if params is None else params.channel_names
    bbar_cont = False if params is None else params.bbar_continuation_only
    groups: dict = {}
    chosen_key = None
    for channel in pool:
        key = exponent_key(flavor_exponents(parent_qn, channel, catalog, fams, hidden, names, bbar_cont))
        groups[key] = groups.get(key, 0.0) + float(channel.weight)
        if channel is chosen:
            chosen_key = key
    if chosen_key is None:
        raise ValueError("vertex_ledger_record: the chosen channel is not in the pool")
    if len(groups) == 1:
        return None
    return (chosen_key, groups)


def ledger_log_weight(
    records: Sequence,
    params_from: Optional[FlavorParams],
    params_to: FlavorParams,
) -> float:
    """The log importance weight of one event's vertex ledger from the
    generation point *params_from* (``None`` is the identity) to
    *params_to*: per vertex ``log F_c(to) - log F_c(from) +
    log Z(from) - log Z(to)`` with ``Z`` the pool normalization, the
    pool weights carrying the ``from`` factors as drawn."""
    if params_from is None:
        params_from = NOMINAL_FLAVOR
    # the exponent keys were counted at the generation point's hidden s-sbar content and pair scope, which the key does
    # not encode, so a reweight across either is refused (the code reviews ds41f S3 and glm53f F4)
    if dict(params_from.hidden_ssbar_content) != dict(params_to.hidden_ssbar_content):
        raise ValueError("ledger_log_weight: the hidden s-sbar content differs between the two points; the ledger's "
                         "exponents were counted at the generation content")
    if bool(params_from.bbar_continuation_only) != bool(params_to.bbar_continuation_only):
        raise ValueError("ledger_log_weight: bbar_continuation_only differs between the two points; the ledger's "
                         "pair exponents were counted under the generation scope")
    total = 0.0
    for record in records:
        if record is None:
            continue
        chosen_key, groups = record
        z_from = 0.0
        z_to = 0.0
        for key, weight in groups.items():
            z_from += weight
            z_to += weight * key_weight(key, params_to) / key_weight(key, params_from)
        total += (
            math.log(key_weight(chosen_key, params_to))
            - math.log(key_weight(chosen_key, params_from))
            + math.log(z_from) - math.log(z_to)
        )
    return total
