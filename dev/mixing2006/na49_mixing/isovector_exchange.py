"""The isovector (charge) exchange class (Round 8, plan task 47;
``TASK-20261004-r8-isovector-exchange-class``, 2026-10-04).

Up to Round 7 every inelastic event of the topology generator exchanges the
vacuum (isoscalar) object: both outgoing baryonic lines keep the proton
quantum numbers ``(B, Q, S) = (1, 1, 0)``, and a leading neutron or a
leading Delta++ needs a multi-step cascade.  This module adds the isovector
exchange: an I = 1 color-singlet exchange (the rho, a2 and pi trajectories
in Regge language) that transfers the third isospin component ``t`` = -1, 0
or +1 to the projectile line and ``-t`` to the target line.

The line states
---------------

A proton line (I = 1/2, I3 = +1/2) that absorbs an I = 1 exchange with
I3 = t becomes a line of charge ``Q = 1 + t`` and isospin I = 1/2 or 3/2,
with the Clebsch-Gordan share ``|<1/2 1/2; 1 t | I, 1/2 + t>|^2``:

====  ======  ===============  ===============
t     Q       I = 3/2          I = 1/2
====  ======  ===============  ===============
+1    2       1                (forbidden)
0     1       2/3              1/3
-1    0       1/3              2/3
====  ======  ===============  ===============

The exchange couples the two lines through the isospin scalar of two I = 1
objects, so each transfer ``t`` carries the same weight 1/3.  With equal
reduced couplings to the I = 1/2 and I = 3/2 lines (a model assumption of
this first version), the joint probability of the transfer and the two line
isospins is the product of the two shares and 1/3
(:data:`ISOVECTOR_CONFIGURATIONS`, eight configurations).  Two thirds of the
isovector events exchange charge (one line at Q = 2, the other at Q = 0).
The I = 1/2 neutron line takes twice the weight of the I = 1/2 proton line
of the t = 0 transfer, the pion-exchange ratio of pp -> nX to pp -> pX.

The exchange class is drawn per event (:func:`draw_exchange_configuration`)
from the isovector fraction of the run: two uniforms on the
``topology_selection`` substream (the class draw and the configuration
draw) when the fraction is positive, zero draws when it is zero.  The
single-diffraction, double-diffraction and non-diffractive classes use it;
the runner passes the configuration to the class generator.  A
vacuum-drawn event is the committed event on the same substreams.

The lines of an isovector event
-------------------------------

- An excited line (a single- or double-diffraction excitation, an excited
  non-diffractive remnant) starts its cascade in the continuum family of
  the line's ``(B, Q, S)`` and isospin (:func:`line_root_family`).  The
  I = 1/2 lines at Q = 1 and Q = 0 start in ``baryonic_np`` and
  ``baryonic_np_q0``; the I = 3/2 lines at Q = 2, 1 and 0 start in
  ``baryonic_np_q2`` and the Delta-like families ``baryonic_np_delta_q1``
  and ``baryonic_np_delta_q0``.  The run supplies them through
  ``na49_mixing.kernel_config.KernelConfig`` (the ``baryonic`` isospin
  completion and ``delta_lines``).
- A remnant line (the non-excited side of single diffraction, the
  non-diffractive remnants) takes its kind from the run's kind weights
  with one uniform, as :func:`na49_mixing.topology_remnants.select_remnant`
  does (:func:`select_line_remnant`).  The intact kind is the ground state
  of the line (the proton, the neutron, or the Delta(1232) of the line
  charge); the low-mass kind draws over the catalog records of the line's
  charge and isospin below the kernel boundary; the high-mass kind is the
  continuum excitation of the line.  The neutron is thus the intact
  remnant of a Q = 0, I = 1/2 line.  The requested kind
  ``neutron_charge_exchange`` of the Round 5 kind set stays refused.

Under the vacuum exchange nothing changes: the committed remnant selection
(the ten ``(1, 1, 0)`` low-mass records, including the three Delta+ records)
and the ``baryonic_np`` root.

Hermetic: standard library and numpy only; no event generation driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Mapping

import numpy as np

from .exchange_ledger import ISOVECTOR_EXCHANGES, VACUUM, Exchange, LineState
from .hadron_catalog import HadronCatalog, HadronRecord
from .topology_remnants import (
    CONTINUUM_M_MAX,
    KERNEL_BOUNDARY,
    KIND_HIGH_MASS_BARYONIC_CONTINUUM,
    KIND_INTACT_PROTON,
    KIND_LOW_MASS_BARYONIC_RESONANCE,
    PROTON_MASS,
    RemnantSelection,
    _draw_kind,
    _validate_weights,
)

__all__ = [
    "ISOVECTOR_CLASSES",
    "ISOVECTOR_CONFIGURATIONS",
    "ISOVECTOR_EXCHANGE_NAME",
    "PROTON_LINE",
    "VACUUM_CONFIGURATION",
    "VACUUM_EXCHANGE_NAME",
    "ExchangeConfiguration",
    "LineCharge",
    "draw_exchange_configuration",
    "isovector_configurations",
    "ground_state_pdg",
    "line_low_mass_records",
    "line_mass_threshold",
    "line_root_family",
    "line_share",
    "select_line_remnant",
]

VACUUM_EXCHANGE_NAME: str = VACUUM.name
ISOVECTOR_EXCHANGE_NAME: str = "isovector"

#: The topology classes that draw the exchange class (the elastic class
#: stays the vacuum exchange; ``nd_multi`` refuses a positive fraction).
ISOVECTOR_CLASSES: frozenset[str] = frozenset({"sd_projectile", "sd_target", "dd", "nd"})

_ISOSPINS = (0.5, 1.5)


@dataclass(frozen=True)
class LineCharge:
    """The charge ``Q`` and the isospin I of an outgoing non-strange
    baryonic line after the exchange (``(B, S) = (1, 0)``)."""

    Q: int
    isospin: float

    def __post_init__(self) -> None:
        if isinstance(self.Q, bool) or not isinstance(self.Q, int):
            raise ValueError(f"LineCharge: Q must be an integer, got {self.Q!r}")
        if float(self.isospin) not in _ISOSPINS:
            raise ValueError(f"LineCharge: the isospin must be 1/2 or 3/2, got {self.isospin!r}")
        if abs(2 * self.Q - 1) > int(round(2 * float(self.isospin))):
            raise ValueError(f"LineCharge: |I3| = |Q - 1/2| exceeds I for Q = {self.Q}, I = {self.isospin}")
        object.__setattr__(self, "isospin", float(self.isospin))

    @property
    def qn(self) -> tuple[int, int, int]:
        """The line's ``(B, Q, S)``."""
        return (1, int(self.Q), 0)

    @property
    def i3x2(self) -> int:
        """The doubled third isospin component ``2 I3 = 2Q - 1``."""
        return 2 * int(self.Q) - 1


#: The proton line (the vacuum exchange's outgoing line).
PROTON_LINE = LineCharge(1, 0.5)


def line_share(t: int, isospin: float) -> Fraction:
    """``|<1/2 1/2; 1 t | I, 1/2 + t>|^2``: the share of the line isospin
    I when a proton line absorbs the I3 transfer *t* (the table of the
    module docstring; the tests check it against
    ``na49_mixing.resonance_decay.clebsch_gordan_squared``)."""
    table = {1: {1.5: Fraction(1)}, 0: {1.5: Fraction(2, 3), 0.5: Fraction(1, 3)},
             -1: {1.5: Fraction(1, 3), 0.5: Fraction(2, 3)}}
    if t not in table:
        raise ValueError(f"line_share: the I3 transfer must be -1, 0 or +1, got {t!r}")
    return table[t].get(float(isospin), Fraction(0))


@dataclass(frozen=True)
class ExchangeConfiguration:
    """The exchange of one event: its name (``"vacuum"`` or
    ``"isovector"``), the I3 transfer *i3_transfer* to the projectile line
    (the target line takes the opposite), and the two outgoing line
    states."""

    name: str
    i3_transfer: int
    projectile: LineCharge
    target: LineCharge

    def __post_init__(self) -> None:
        if self.name not in (VACUUM_EXCHANGE_NAME, ISOVECTOR_EXCHANGE_NAME):
            raise ValueError(f"ExchangeConfiguration: unknown exchange {self.name!r}")
        t = self.i3_transfer
        if isinstance(t, bool) or t not in (-1, 0, 1):
            raise ValueError(f"ExchangeConfiguration: the I3 transfer must be -1, 0 or +1, got {t!r}")
        if self.name == VACUUM_EXCHANGE_NAME and (t != 0 or self.projectile != PROTON_LINE or self.target != PROTON_LINE):
            raise ValueError("ExchangeConfiguration: the vacuum exchange keeps both proton lines")
        if (self.projectile.Q, self.target.Q) != (1 + t, 1 - t):
            raise ValueError(f"ExchangeConfiguration: the line charges {(self.projectile.Q, self.target.Q)} "
                             f"disagree with the I3 transfer {t}")

    @property
    def is_vacuum(self) -> bool:
        return self.name == VACUUM_EXCHANGE_NAME

    def exchanges(self) -> tuple[Exchange, Exchange]:
        """The ledger exchange records of the two lines (projectile side
        first): the transfer ``(0, t, 0, 2t)`` and its opposite."""
        if self.is_vacuum:
            return (VACUUM, VACUUM)
        t = self.i3_transfer
        return (Exchange(ISOVECTOR_EXCHANGE_NAME, (0, t, 0, 2 * t)),
                Exchange(ISOVECTOR_EXCHANGE_NAME, (0, -t, 0, -2 * t)))

    def records(self) -> tuple[tuple[str, tuple[int, int, int, int]], tuple[str, tuple[int, int, int, int]]]:
        """The exchange records as ``(name, transfer)`` pairs."""
        return tuple((e.name, tuple(e.transfer)) for e in self.exchanges())

    def line_states(self, p4_projectile=None, p4_target=None) -> tuple[LineState, LineState]:
        """The two outgoing ledger line states (projectile side first)."""
        return (LineState(B=1, Q=self.projectile.Q, S=0, I3x2=self.projectile.i3x2, four_momentum=p4_projectile),
                LineState(B=1, Q=self.target.Q, S=0, I3x2=self.target.i3x2, four_momentum=p4_target))


VACUUM_CONFIGURATION = ExchangeConfiguration(VACUUM_EXCHANGE_NAME, 0, PROTON_LINE, PROTON_LINE)


def _isovector_configurations() -> tuple[tuple[Fraction, ExchangeConfiguration], ...]:
    out = []
    for t in (1, 0, -1):
        for i_p in (1.5, 0.5):
            for i_t in (1.5, 0.5):
                p = Fraction(1, 3) * line_share(t, i_p) * line_share(-t, i_t)
                if p:
                    out.append((p, ExchangeConfiguration(ISOVECTOR_EXCHANGE_NAME, t, LineCharge(1 + t, i_p),
                                                         LineCharge(1 - t, i_t))))
    return tuple(out)


#: The isovector configurations with their exact probabilities, in the
#: draw order (t = +1, 0, -1; the projectile isospin 3/2 before 1/2; the
#: target isospin 3/2 before 1/2).
ISOVECTOR_CONFIGURATIONS: tuple[tuple[Fraction, ExchangeConfiguration], ...] = _isovector_configurations()
assert sum(p for p, _c in ISOVECTOR_CONFIGURATIONS) == 1
assert all(c.exchanges()[0] in ISOVECTOR_EXCHANGES for _p, c in ISOVECTOR_CONFIGURATIONS)

_CUMULATIVE: tuple[float, ...] = tuple(
    float(sum(p for p, _c in ISOVECTOR_CONFIGURATIONS[:k + 1])) for k in range(len(ISOVECTOR_CONFIGURATIONS)))


_CONFIGURATIONS_BY_RATIO: dict = {}


def isovector_configurations(delta_ratio: float = 1.0) -> tuple[tuple[float, ExchangeConfiguration], ...]:
    """The configurations and their probabilities at the reduced-coupling
    ratio *delta_ratio* r = g^2(I = 3/2) / g^2(I = 1/2) of a line (2026-10-05).

    Each line's isospin share is its Clebsch-Gordan square times r for
    I = 3/2; the joint weight is the product over the two lines,
    normalized over the eight configurations.  At r = 1 every transfer keeps
    the weight 1/3 and the result is :data:`ISOVECTOR_CONFIGURATIONS`
    exactly.  A Q = 2 line is I = 3/2 at any r, so charge exchange always
    carries one Delta++-like line; r sets the I = 3/2 share of the Q = 0 and
    Q = 1 lines (and, below about 0.25, the charge-exchange share itself)."""
    r = float(delta_ratio)
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError(f"isovector_configurations: the coupling ratio must be positive, got {delta_ratio!r}")
    if r == 1.0:
        return ISOVECTOR_CONFIGURATIONS
    if r not in _CONFIGURATIONS_BY_RATIO:
        # the r = 1 probability is 1/3 times the two shares; r weights each I = 3/2 line
        raw = [(float(p) * (r if c.projectile.isospin == 1.5 else 1.0) * (r if c.target.isospin == 1.5 else 1.0), c)
               for p, c in ISOVECTOR_CONFIGURATIONS]
        total = math.fsum(w for w, _c in raw)
        _CONFIGURATIONS_BY_RATIO[r] = tuple((w / total, c) for w, c in raw)
    return _CONFIGURATIONS_BY_RATIO[r]


def draw_exchange_configuration(fraction: float, rng: np.random.Generator, delta_ratio: float = 1.0) -> ExchangeConfiguration:
    """The exchange of one event at the isovector *fraction*.

    Zero draws and the vacuum configuration at ``fraction == 0``.  Above
    zero, exactly two uniforms from *rng* (the ``topology_selection``
    substream): the class uniform (isovector below *fraction*) and the
    configuration uniform resolved over :func:`isovector_configurations`
    at *delta_ratio* (drawn for a vacuum event too, so the budget is
    fixed; r = 1 is :data:`ISOVECTOR_CONFIGURATIONS`)."""
    f = float(fraction)
    if not (math.isfinite(f) and 0.0 <= f <= 1.0):
        raise ValueError(f"draw_exchange_configuration: the isovector fraction must lie in [0, 1], got {fraction!r}")
    if f == 0.0:
        return VACUUM_CONFIGURATION
    configurations = isovector_configurations(delta_ratio)
    cumulative = _CUMULATIVE if configurations is ISOVECTOR_CONFIGURATIONS else _cumulative_of(configurations)
    u_class = float(rng.random())
    u_config = float(rng.random())
    if not u_class < f:
        return VACUUM_CONFIGURATION
    for edge, (_p, configuration) in zip(cumulative, configurations):
        if u_config < edge:
            return configuration
    return configurations[-1][1]


def _cumulative_of(configurations) -> tuple[float, ...]:
    out, total = [], 0.0
    for p, _c in configurations:
        total += float(p)
        out.append(total)
    return tuple(out)


#: The ground state of each line: the nucleon or the Delta(1232) of the
#: line charge.
_GROUND_STATES = {(1, 0.5): 2212, (0, 0.5): 2112, (2, 1.5): 2224, (1, 1.5): 2214, (0, 1.5): 2114, (-1, 1.5): 1114}


def ground_state_pdg(line: LineCharge) -> int:
    """The pdg id of the line's ground state (the intact kind)."""
    return _GROUND_STATES[(int(line.Q), float(line.isospin))]


def line_low_mass_records(catalog: HadronCatalog, line: LineCharge) -> list[HadronRecord]:
    """The catalog records of the line's ``(B, Q, S)`` and isospin with
    ``mass_range[0]`` below the kernel boundary, in catalog order (the
    low-mass kind of the line; the nucleon or Delta(1232) ground state
    included, as the committed ``(1, 1, 0)`` set includes the proton)."""
    return [r for r in catalog.records()
            if (int(r.B), int(r.Q), int(r.S)) == line.qn and float(r.isospin) == line.isospin
            and float(r.mass_range[0]) < KERNEL_BOUNDARY]


def select_line_remnant(weights: Mapping[str, float], catalog: HadronCatalog, rng_kind: np.random.Generator,
                        mass_sampler: Callable[[float, float], float], line: LineCharge) -> RemnantSelection:
    """One remnant selection on the line *line*: exactly one uniform from
    *rng_kind* (the ``spectral_state`` substream), resolved over the kind
    weights in the sorted-kind order as
    :func:`~na49_mixing.topology_remnants.select_remnant` does; the
    low-mass record reuses the residual of the same uniform.

    - intact: the line's ground state (:func:`ground_state_pdg`) at its
      catalog pole mass;
    - low-mass: a record of :func:`line_low_mass_records` at its pole mass;
    - high-mass: the continuum excitation, the mass from *mass_sampler*
      (``mass_sampler(m_p, m_max)``), no catalog identity.
    """
    normalized = _validate_weights(weights, None)
    u = rng_kind.random()
    drawn, lower, upper = _draw_kind(normalized, u)
    qn = (1, int(line.Q), 0, line.i3x2)
    if drawn == KIND_INTACT_PROTON:
        record = catalog.get(ground_state_pdg(line))
        return RemnantSelection(kind=drawn, pdg_id=int(record.pdg_id), record_id=None,
                                mass=float(record.pole_mass), qn=qn)
    if drawn == KIND_LOW_MASS_BARYONIC_RESONANCE:
        records = line_low_mass_records(catalog, line)
        if not records:
            raise ValueError(f"select_line_remnant: no low-mass record for the line {line}")
        width = upper - lower
        residual = (u - lower) / width if width > 0.0 else 0.0
        record = records[min(int(residual * len(records)), len(records) - 1)]
        return RemnantSelection(kind=drawn, pdg_id=int(record.pdg_id), record_id=str(record.name),
                                mass=float(record.pole_mass), qn=qn)
    if drawn == KIND_HIGH_MASS_BARYONIC_CONTINUUM:
        mass = float(mass_sampler(PROTON_MASS, CONTINUUM_M_MAX))
        return RemnantSelection(kind=drawn, pdg_id=None, record_id=None, mass=mass, qn=qn)
    raise ValueError(f"select_line_remnant: the drawn kind {drawn!r} is not a remnant of an isovector line "
                     "(the neutron arises as the intact remnant of a Q = 0, I = 1/2 line)")


def line_root_family(fams, line: LineCharge):
    """The continuum family where the cascade of an excited line starts:
    the family of the line's ``(B, Q, S)`` and isospin (config order).
    ``ValueError`` when the run supplies none (the isovector lines need the
    ``baryonic`` isospin completion and ``delta_lines`` of
    ``na49_mixing.kernel_config.KernelConfig``)."""
    for family in fams:
        if (tuple(int(v) for v in family.qn) == line.qn and family.isospin is not None
                and float(family.isospin) == line.isospin):
            return family
    raise ValueError(f"line_root_family: no continuum family carries the line (B, Q, S) = {line.qn} with isospin "
                     f"{line.isospin} (KernelConfig: isospin_completion 'baryonic' and delta_lines)")


def line_mass_threshold(fams, catalog: HadronCatalog, line: LineCharge | None = None) -> float:
    """The lower mass edge of an excited line (2026-10-06; code-physics
    review M-6, ``TASK-20261005-cascade-vertex-rules`` item 1): the lowest
    direct two-body threshold of the family where the line's cascade starts.
    Below it the root has no open channel and would end unidentified.
    *line* ``None`` is the vacuum line, whose root is the first family of
    ``(B, Q, S) = (1, 1, 0)``; an isovector line takes
    :func:`line_root_family`.  The values are p pi0 for Q = 1, n pi0 for
    Q = 0 and p pi+ for Q = 2 at the catalog masses."""
    from .cascade_decay import _family_by_qn, _lowest_direct_threshold

    family = _family_by_qn(fams, (1, 1, 0)) if line is None else line_root_family(fams, line)
    if family is None:
        raise ValueError("line_mass_threshold: no continuum family carries the vacuum line (1, 1, 0)")
    return float(_lowest_direct_threshold(family, catalog))
