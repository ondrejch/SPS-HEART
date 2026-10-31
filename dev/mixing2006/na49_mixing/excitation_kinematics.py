"""Lorentz-vector and exact two-body helpers for source construction."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import math
import re
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FourVector:
    """Four-vector in ``(E, px, py, pz)`` convention, in GeV."""

    e: float
    px: float
    py: float
    pz: float

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.e + other.e, self.px + other.px,
            self.py + other.py, self.pz + other.pz,
        )

    def __sub__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.e - other.e, self.px - other.px,
            self.py - other.py, self.pz - other.pz,
        )

    def spatial(self) -> np.ndarray:
        return np.asarray([self.px, self.py, self.pz], dtype=float)

    def mass2(self) -> float:
        return self.e * self.e - self.px * self.px - self.py * self.py - self.pz * self.pz

    def mass(self) -> float:
        return math.sqrt(max(self.mass2(), 0.0))

    def momentum(self) -> float:
        return math.sqrt(self.px * self.px + self.py * self.py + self.pz * self.pz)

    def as_dict(self) -> dict[str, float]:
        return {"e": self.e, "px": self.px, "py": self.py, "pz": self.pz}


def sum_four_vectors(vectors: Iterable[FourVector]) -> FourVector:
    out = FourVector(0.0, 0.0, 0.0, 0.0)
    for vector in vectors:
        out = out + vector
    return out


def kallen(
    a: float | np.ndarray,
    b: float | np.ndarray,
    c: float | np.ndarray,
) -> float | np.ndarray:
    """Kallen function lam(a, b, c) = a*a + b*b + c*c - 2*(a*b + a*c + b*c).

    Pure arithmetic, so it is elementwise correct for NumPy array arguments
    as well as scalars. The scalar two-body path and the vectorized H1
    phase-space ladders share this one formula, so the two cannot drift.
    """
    return a * a + b * b + c * c - 2.0 * a * b - 2.0 * a * c - 2.0 * b * c


def two_body_momentum(parent_mass: float, mass1: float, mass2: float) -> float:
    if parent_mass <= 0.0 or mass1 + mass2 > parent_mass:
        raise ValueError(
            f"two-body threshold failed: {mass1} + {mass2} > {parent_mass}"
        )
    lam = kallen(parent_mass * parent_mass, mass1 * mass1, mass2 * mass2)
    if lam < -1e-12:
        raise ValueError("negative Kallen function in physical two-body construction")
    return math.sqrt(max(lam, 0.0)) / (2.0 * parent_mass)


def boost(vector: FourVector, beta) -> FourVector:
    """Boost *vector* by a three-velocity ``beta``.

    A rest-frame vector boosted with the parent's lab-frame velocity becomes a
    lab-frame vector.  The inverse operation uses ``-beta``.
    """
    b = np.asarray(beta, dtype=float)
    b2 = float(np.dot(b, b))
    if b2 <= 0.0:
        return vector
    if not math.isfinite(b2) or b2 >= 1.0:
        raise ValueError(f"boost requires |beta| < 1, got beta^2={b2}")
    gamma = 1.0 / math.sqrt(1.0 - b2)
    p = vector.spatial()
    bp = float(np.dot(b, p))
    factor = ((gamma - 1.0) * bp / b2) + gamma * vector.e
    p_out = p + factor * b
    e_out = gamma * (vector.e + bp)
    return FourVector(e_out, float(p_out[0]), float(p_out[1]), float(p_out[2]))


def orthonormal_basis(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(z))
    if norm == 0.0:
        z = np.asarray([0.0, 0.0, 1.0])
    else:
        z = z / norm
    helper = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, helper))) > 0.95:
        helper = np.asarray([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def direction_about(axis, cos_theta: float, phi: float) -> np.ndarray:
    x, y, z = orthonormal_basis(axis)
    c = min(max(float(cos_theta), -1.0), 1.0)
    s = math.sqrt(max(1.0 - c * c, 0.0))
    return c * z + s * (math.cos(phi) * x + math.sin(phi) * y)


def sample_cos_from_exp_t(
    rng: np.random.Generator,
    slope: float,
    t_at_minus_one: float,
    t_at_plus_one: float,
) -> tuple[float, float]:
    """Sample a linear-in-cos(theta) transfer with density ``exp(slope*t)``."""
    t_lo = min(t_at_minus_one, t_at_plus_one)
    t_hi = max(t_at_minus_one, t_at_plus_one)
    if t_hi - t_lo <= 1e-15 or abs(slope) <= 1e-14:
        t = t_lo + rng.random() * (t_hi - t_lo)
    else:
        lo = slope * t_lo
        hi = slope * t_hi
        shift = max(lo, hi)
        a = math.exp(lo - shift)
        b = math.exp(hi - shift)
        t = (math.log(a + rng.random() * (b - a)) + shift) / slope
    frac = 0.5 if t_hi == t_lo else (t - t_at_minus_one) / (t_at_plus_one - t_at_minus_one)
    cos_theta = -1.0 + 2.0 * frac
    return min(max(cos_theta, -1.0), 1.0), t


# ---------------------------------------------------------------------------
# Exact n-body phase-space primitives (card M7, hypothesis H1)
# ---------------------------------------------------------------------------
#
# The n-body phase space factors as
#
#     Phi_n = ∫ dM_c² / (2π)  Phi_2(P -> m_1 + M_c)  Phi_{n-1}(M_c -> m_2..m_n)
#
# with the two-body volume Phi_2 ∝ pstar(M, m_1, M_c) / M (integrated over
# the decay angle). In the dM_c measure the cluster mass M_c (the cluster
# m_2..m_n) is therefore drawn with the density
#
#     pstar(M, m_1, M_c) * 2 M_c * Phi_{n-1}(M_c)
#
# and the split is isotropic. The recursion is exact for every n (the inner
# (n-1)-body decay is sampled from its own uniform phase space), and the
# final distribution is independent of the daughter ordering because the
# uniform measure is permutation invariant: any fixed recursion order is an
# exact parameterization of the same Lorentz-invariant phase space.

# Documented sub-tolerance grid convention, shared with
# ``breit_wigner.draw_truncated_bw`` (order-1e-3 grid bias for smooth
# densities, far below every statistical tolerance in use).
PHSP_GRID_SIZE = 1024


def two_body_momentum_safe(parent_mass: float, mass1: float, mass2: float) -> float:
    """Two-body momentum that returns ``0.0`` at or below threshold.

    The safe twin of :func:`two_body_momentum` for density and volume
    evaluations (the H1 phase-space ladder, the H1/H2 line-shape integrands)
    where a below-threshold argument must contribute zero measure instead of
    raising.
    """
    if parent_mass <= 0.0 or mass1 + mass2 >= parent_mass:
        return 0.0
    lam = kallen(parent_mass * parent_mass, mass1 * mass1, mass2 * mass2)
    if lam < 0.0:
        return 0.0
    return math.sqrt(lam) / (2.0 * parent_mass)


def draw_from_grid_density(
    rng: np.random.Generator,
    grid: np.ndarray,
    density: np.ndarray,
) -> float:
    """Draw one value from a 1-D density given on an equally spaced grid.

    Normalized by construction: trapezoid cell masses, inverse CDF, uniform
    within the selected cell (two uniform random numbers, the same
    sub-tolerance grid convention as ``breit_wigner.draw_truncated_bw``).
    Raises ``ValueError`` when the grid mass vanishes, which the callers
    treat as a degenerate or ineligible configuration (the event builder
    retries with a fresh draw).
    """
    g = np.asarray(grid, dtype=np.float64)
    d = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    if g.size < 2:
        raise ValueError("phase-space density grid needs at least two points")
    if g.size != d.size:
        raise ValueError("density grid and values must have equal length")
    cell = 0.5 * (d[1:] + d[:-1]) * (g[1:] - g[:-1])
    total = float(cell.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("phase-space density carries no mass on the grid")
    cdf = np.cumsum(cell) / total
    u_segment = rng.random()
    idx = int(np.searchsorted(cdf, u_segment, side="right"))
    if idx >= cdf.size:
        idx = cdf.size - 1
    u_within = rng.random()
    value = g[idx] + u_within * (g[idx + 1] - g[idx])
    return float(min(max(value, g[0]), g[-1]))


def two_body_split_rest_frame(
    parent_mass: float,
    mass1: float,
    mass2: float,
    rng: np.random.Generator,
) -> tuple[FourVector, FourVector]:
    """Exact isotropic two-body split in the parent rest frame.

    Reuses :func:`two_body_momentum` and :func:`direction_about` with the
    same draw order as the inline H3 kernel (cos(theta) then phi).
    """
    pstar = two_body_momentum(parent_mass, mass1, mass2)
    c = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    d = direction_about([0.0, 0.0, 1.0], c, phi)
    e1 = math.sqrt(mass1 * mass1 + pstar * pstar)
    e2 = math.sqrt(mass2 * mass2 + pstar * pstar)
    return (
        FourVector(e1, *(pstar * d)),
        FourVector(e2, *(-pstar * d)),
    )


def two_body_threshold_tolerance(parent_mass: float, mass1: float, mass2: float) -> float:
    """Largest excess of ``mass1 + mass2`` over ``parent_mass`` treated as roundoff.

    A grid-clipped draw at the exact upper endpoint can round so that the
    sum of the daughter masses exceeds the parent by a few ulps. Larger
    violations are physical threshold failures, not floating-point noise.
    """
    scale = max(abs(parent_mass), abs(mass1), abs(mass2), 1.0)
    return max(8.0 * math.ulp(scale), 1e-15 * scale)


class TwoBodyBelowThreshold(ValueError):
    """A two-body mass draw places the daughters above the parent mass.

    Raised by :func:`two_body_split_safe` when ``mass1 + mass2`` exceeds
    ``parent_mass`` beyond the documented float tolerance
    (:func:`two_body_threshold_tolerance`).  The staged event builders
    treat it as a counted rejection and retry with a fresh draw, so no
    daughter is ever emitted off its configured mass shell.  The name
    states the kinematic correction that rejects the draw; it does not
    name a production mechanism.
    """


def two_body_split_safe(
    parent_mass: float,
    mass1: float,
    mass2: float,
    rng: np.random.Generator,
) -> tuple[FourVector, FourVector]:
    """Two-body split with every daughter on its configured mass shell.

    A mass draw with ``mass1 + mass2`` above ``parent_mass`` beyond
    :func:`two_body_threshold_tolerance` raises
    :class:`TwoBodyBelowThreshold` so the event builder rejects and
    retries; the daughter masses are never moved.  An overshoot inside
    the tolerance is float roundoff of a threshold draw: the split is
    emitted with zero relative momentum and the daughters carry their
    configured masses, so ``m_i^2 = p_i · p_i`` equals the configured
    ``mass_i`` exactly and the parent four-momentum balance absorbs the
    roundoff excess, which is itself within the documented tolerance.
    Above threshold (``mass1 + mass2 < parent_mass``), the exact
    isotropic split of :func:`two_body_split_rest_frame` is used.
    """
    if mass1 < 0.0 or mass2 < 0.0:
        raise ValueError(
            f"two-body split below threshold: negative daughter mass "
            f"m1={mass1} m2={mass2}"
        )
    excess = (mass1 + mass2) - parent_mass
    if excess > two_body_threshold_tolerance(parent_mass, mass1, mass2):
        raise TwoBodyBelowThreshold(
            f"two-body split below threshold: m1+m2={mass1 + mass2} "
            f"> M={parent_mass}"
        )
    if excess >= 0.0:
        # Threshold point: zero relative momentum, configured masses.
        return (
            FourVector(mass1, 0.0, 0.0, 0.0),
            FourVector(mass2, 0.0, 0.0, 0.0),
        )
    return two_body_split_rest_frame(parent_mass, mass1, mass2, rng)


def scatter_two_body(
    total: FourVector,
    incoming_projectile: FourVector,
    outgoing_mass1: float,
    outgoing_mass2: float,
    slope: float,
    rng: np.random.Generator,
) -> tuple[FourVector, FourVector, float]:
    """Construct exact two-body scattering in the total-pair rest frame."""
    root_s = total.mass()
    pstar = two_body_momentum(root_s, outgoing_mass1, outgoing_mass2)
    beta_cm = total.spatial() / total.e
    pin_cm = boost(incoming_projectile, -beta_cm)
    incoming_mass = incoming_projectile.mass()
    ein = pin_cm.e
    e1 = math.sqrt(outgoing_mass1 * outgoing_mass1 + pstar * pstar)
    # t = m_in^2 + m_out^2 - 2(Ein Eout - |pin||pout| cos theta)
    common = incoming_mass * incoming_mass + outgoing_mass1 * outgoing_mass1 - 2.0 * ein * e1
    span = 2.0 * pin_cm.momentum() * pstar
    t_minus = common - span
    t_plus = common + span
    cos_theta, t = sample_cos_from_exp_t(rng, slope, t_minus, t_plus)
    direction = direction_about(pin_cm.spatial(), cos_theta, 2.0 * math.pi * rng.random())
    p1_cm = FourVector(e1, *(pstar * direction))
    p2_cm = FourVector(
        math.sqrt(outgoing_mass2 * outgoing_mass2 + pstar * pstar),
        *(-pstar * direction),
    )
    return boost(p1_cm, beta_cm), boost(p2_cm, beta_cm), t


# ---------------------------------------------------------------------------
# Two-body / three-body on-shell site audit (Round 1, card
# TASK-20260911-r1-two-body-on-shell-repair)
# ---------------------------------------------------------------------------
#
# Static inventory of every four-momentum construction site in the staged
# and legacy two-body/three-body decay paths, per the card's public
# interface.  Scope: sites that construct a daughter (or intermediate
# parent) four-momentum from masses inside the two-body primitives
# (:func:`two_body_split_rest_frame`, :func:`two_body_split_safe`,
# :func:`scatter_two_body`), the staged source-decay kernels
# (``continuum_channels`` hypotheses H1/H2/H3), and the legacy decay
# engine (``event.Event._decay_2body`` / ``_decay_3body``).  The physics
# rule of the card additionally names two non-construction sites that the
# audit must adjudicate: the H2 truncated-BW empty-window fallback to the
# daughter threshold (a threshold draw, on-shell by definition) and the
# legacy below-threshold mass-draw guards (the 2026-08-31
# ``terminal_unstable`` counted-rejection mechanism).
#
# Excluded, and why: shared algebra primitives that transform existing
# four-vectors without touching masses (``FourVector.__add__``/
# ``__sub__``, :func:`sum_four_vectors`, :func:`boost`, the legacy
# ``_lorentz_boost``) preserve mass shells by construction; initial-state
# and spectator constructions outside the decay paths (event builders'
# beam, target, spectator four-vectors) are not decay vertices.  The
# completeness scan below raises on any construction line inside an
# audited function that no registered site claims, so a new site fails
# the pinned inventory test instead of passing silently.

_AUDIT_BROAD_PATTERNS: dict[str, str] = {
    "na49_mixing.excitation_kinematics": (
        r"FourVector\(|two_body_split_safe\(|raise TwoBodyBelowThreshold|return lo"
    ),
    "na49_mixing.continuum_channels": (
        r"FourVector\(|two_body_split_safe\(|raise TwoBodyBelowThreshold|return lo"
    ),
    "na49_mixing.event": (
        r"= \[|p2_px = |pv\[0\] = |p3\[0\] = |e1 = \(|e2 = \("
        r"|\) > m_parent|>= m0|e1 \* e1 < m1 \* m1"
    ),
}

# (module, function, marker, site_kind, on_shell, action)
_AUDIT_SITE_REGISTRY: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "na49_mixing.excitation_kinematics", "two_body_split_rest_frame",
        r"FourVector\(e1", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.excitation_kinematics", "two_body_split_rest_frame",
        r"FourVector\(e2", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.excitation_kinematics", "two_body_split_safe",
        r"raise TwoBodyBelowThreshold", "staged", "rejection", "repaired",
    ),
    (
        "na49_mixing.excitation_kinematics", "two_body_split_safe",
        r"FourVector\(mass1", "staged", "by_construction", "repaired",
    ),
    (
        "na49_mixing.excitation_kinematics", "two_body_split_safe",
        r"FourVector\(mass2", "staged", "by_construction", "repaired",
    ),
    (
        "na49_mixing.excitation_kinematics", "scatter_two_body",
        r"p1_cm = FourVector\(", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.excitation_kinematics", "scatter_two_body",
        r"p2_cm = FourVector\(", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_rest_frame",
        r"FourVector\(parent_mass, 0\.0, 0\.0, 0\.0\)",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_rest_frame",
        r"first = FourVector\(", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_rest_frame",
        r"remainder = FourVector\(", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_phsp",
        r"FourVector\(parent_mass, 0\.0, 0\.0, 0\.0\)",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_phsp",
        r"two_body_split_safe\(parent_mass", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_phsp",
        r"two_body_split_safe\(cluster_mass, rest\[0\]",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_phsp",
        r"two_body_split_safe\(cluster_mass, m_j",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_decay_resonant",
        r"FourVector\(parent_mass, 0\.0, 0\.0, 0\.0\)",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_h2_resonance_mass",
        r"return lo", "staged", "by_construction", "documented",
    ),
    (
        "na49_mixing.continuum_channels", "_h2_emit",
        r"two_body_split_safe\(mass", "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_h2_cascade",
        r"FourVector\(cluster_mass, 0\.0, 0\.0, 0\.0\)",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_h2_cascade",
        r"two_body_split_safe\(cluster_mass, m1, m2",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.continuum_channels", "_h2_cascade",
        r"two_body_split_safe\(cluster_mass, m1, cluster_mass - m1",
        "staged", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_2body",
        r"\(m1 \+ m2\) > m_parent", "legacy", "rejection", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_2body",
        r"e1 = \(m_parent \* m_parent", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_2body",
        r"e1 \* e1 < m1 \* m1", "legacy", "rejection", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_2body",
        r"p1 = \[", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_2body",
        r"p2_px = parent\.get_px\(\)", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"thresh >= m0", "legacy", "rejection", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"e1 = \(m0 \* m0", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"p1 = \[e1", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"p23 = \[", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"e2 = \(m23 \* m23", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"p2 = \[e2", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"p3 = \[math\.sqrt\(pl2", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"pv\[0\] = math\.sqrt", "legacy", "by_construction", "unchanged",
    ),
    (
        "na49_mixing.event", "_decay_3body",
        r"p3\[0\] = math\.sqrt", "legacy", "by_construction", "unchanged",
    ),
)


def _audit_resolve(module: object, module_name: str, function_name: str):
    """Resolve an audited function or method from its module.

    Module-level functions resolve directly; legacy decay-engine methods
    (``Event._decay_2body`` / ``_decay_3body``) resolve from their owner
    class.
    """
    candidate = getattr(module, function_name, None)
    if candidate is not None:
        return candidate
    for value in vars(module).values():
        if inspect.isclass(value):
            owner = vars(value).get(function_name)
            if owner is not None:
                return owner
    raise RuntimeError(
        f"audit site vanished: {module_name}.{function_name} is not "
        f"defined in the module"
    )


def audit_two_body_sites() -> list[dict]:
    """Static pre-audit inventory of two-body/three-body construction sites.

    Returns one entry per four-momentum construction site,
    ``{module, function, line, site_kind, on_shell, documented, action}``,
    in module/registry order with entries of one function in source-line
    order.  ``site_kind`` is ``staged`` or ``legacy``; ``on_shell`` is
    ``by_construction`` (the four-vector is built from the configured
    masses), ``rejection`` (a below-threshold draw is rejected instead of
    emitted), or ``clip_fallback`` (a mass-moving fallback, absent after
    the repair); ``action`` is ``unchanged``, ``repaired``, or
    ``documented``.  Raises ``RuntimeError`` when the audited source no
    longer matches the registered inventory (a site moved out of a
    registered function, vanished, or a new construction line appeared),
    so the pinned inventory test cannot pass against an incomplete audit.
    """
    module_functions: dict[str, list[str]] = {}
    for module_name, function_name, _, _, _, _ in _AUDIT_SITE_REGISTRY:
        functions = module_functions.setdefault(module_name, [])
        if function_name not in functions:
            functions.append(function_name)

    entries: list[dict] = []
    for module_name, function_names in module_functions.items():
        module = importlib.import_module(module_name)
        broad = re.compile(_AUDIT_BROAD_PATTERNS[module_name])
        for function_name in function_names:
            function = _audit_resolve(module, module_name, function_name)
            source = inspect.getsource(function)
            first_line = function.__code__.co_firstlineno
            lines = source.splitlines()
            matches = [
                (first_line + offset, line)
                for offset, line in enumerate(lines)
                if broad.search(line) and not line.lstrip().startswith("def ")
            ]
            for module_reg, fn_reg, marker, site_kind, on_shell, action in (
                _AUDIT_SITE_REGISTRY
            ):
                if module_reg != module_name or fn_reg != function_name:
                    continue
                marker_re = re.compile(marker)
                hits = [(ln, text) for ln, text in matches if marker_re.search(text)]
                if not hits:
                    raise RuntimeError(
                        f"audit site vanished: {module_name}.{function_name} "
                        f"marker {marker!r} no longer matches the source"
                    )
                for line_no, _text in sorted(hits):
                    entries.append(
                        {
                            "module": module_name,
                            "function": function_name,
                            "line": int(line_no),
                            "site_kind": site_kind,
                            "on_shell": on_shell,
                            "documented": True,
                            "action": action,
                        }
                    )
        # Completeness: every construction line inside an audited function
        # must be claimed by a registered site marker.
        for function_name in function_names:
            function = _audit_resolve(module, module_name, function_name)
            source = inspect.getsource(function)
            first_line = function.__code__.co_firstlineno
            unclaimed = [
                (first_line + offset, line.strip())
                for offset, line in enumerate(source.splitlines())
                if broad.search(line)
                and not line.lstrip().startswith("def ")
                and not any(
                    re.search(marker, line)
                    for module_reg, fn_reg, marker, _, _, _ in _AUDIT_SITE_REGISTRY
                    if module_reg == module_name and fn_reg == function_name
                )
            ]
            if unclaimed:
                raise RuntimeError(
                    "new 4-vector construction site not in the audit "
                    f"inventory: {module_name}.{function_name} lines "
                    + ", ".join(f"{ln}: {text}" for ln, text in unclaimed)
                )
    return entries
