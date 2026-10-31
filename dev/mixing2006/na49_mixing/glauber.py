"""Optical Glauber geometry for p+A wounded-nucleon sampling.

Model (Bialas-Bleszynski-Czyz superposition, p+A special case):

- The projectile is a single nucleon that participates in every event.
- The target nucleus has density ``rho(r)`` normalized to a unit
  integral; its thickness function is ``T_A(b) = integral rho dz``.
- At fixed impact parameter *b* the number *N* of encountered target
  nucleons follows a Poisson law with mean ``mu(b) = sigma_NN * T_A(b)``
  (sigma converted to fm^2). An interaction takes place when ``N >= 1``,
  so the number of wounded target nucleons is the conditional variable
  ``K = N | N >= 1`` with

      P(K=k | b) = exp(-mu) mu^k / k! / (1 - exp(-mu)),  k = 1, 2, ...

- Minimum-bias sampling draws *b* from the interaction-weighted measure
  ``P_inel(b) d^2b / sigma_pA`` with ``P_inel(b) = 1 - exp(-mu(b))``.
- Centrality slices are defined on ``K`` (the number of wounded target
  nucleons). They are generated directly from the interaction-weighted
  distribution restricted to the requested ``K`` window
  (:meth:`WoundedNucleonSampler.sample_event_in_range`), not by
  throwing away minimum-bias events; see the method docstring for the
  conditioned measure. This is monotone in the forward-energy
  centrality used by NA49 p+Pb analyses only approximately;
  cross-calibration against the experimental E_BE definition is left
  to benchmark work.

Truncation at ``A``: ``K`` counts distinct wounded target nucleons, so
physically ``K <= A``, while the conditional Poisson above is formally
unbounded. Both the sampler and the analytic slice fractions enforce
the bound the same way: the inversion in ``sample_conditional_wounds``
stops at ``k = A`` and assigns the remaining tail probability to
``K = A``, and ``fraction_for_wound_range`` returns 0 for ``lo > A``
and accumulates the ``K >= A`` tail onto the ``K = A`` bin.

Assumptions of this stage: optical limit, independent sources, no
secondary NN rescattering, single-wound projectile. The projectile-side
emission is drawn exactly once per event regardless of K; each wounded
target nucleon contributes one independently drawn, mirrored
target-side emission (implemented in ``na49_mixing.parallel``).

A point target (``A == 1``) reproduces p+p bookkeeping exactly: every
event has K = 1 and the sampler consumes no random numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MB_PER_FM2 = 10.0  # 1 fm^2 = 10 mb

# Pre-allocation grid budgets for WoundedNucleonSampler. Building the
# thickness tables produces three one-dimensional grids and then a full
# broadcasted b x z table:
#
#     r_grid = arange(0.0,     r_stop,  dr_fm)              # n_r cells
#     z_grid = arange(-z_half, z_stop,  dr_fm)              # n_z cells
#     b_grid = arange(0.0,     b_stop,  dr_fm)              # n_b cells
#     rr     = sqrt(b_grid[:, None]**2 + z_grid[None,:]**2) # n_b x n_z
#     rho_rz = zeros_like(rr)                               # n_b x n_z
#     inside = rr <= r_max                                  # bool mask
#
# Every axis extent is tied to r_max plus a multiple of the surface
# scale (woods_saxon: surface = diffuseness_fm, r_max = R + 10a;
# harmonic_oscillator: surface = radius_fm, r_max = 8R), so each axis
# count must be capped independently. Counting only the r-axis ratio
# ``r_max / dr_fm`` is what let the reviewed case slip through: a step
# just above that ratio admitted thousands of extra cells along the
# derived b and z axes.
#
# _MAX_GRID_CELLS_PER_AXIS caps n_r, n_b and n_z individually. It sits
# two orders of magnitude above the heaviest configured system (Pb208
# Woods-Saxon at the default dr_fm = 0.02 fm needs 606 radial cells,
# 1756 z cells, 687 b cells and a ~1.21e6-cell table, ~40 MB estimated)
# while refusing accidents such as dr_fm = 1e-12 fm (~1e13 cells per
# axis) before a single ``np.arange`` runs.
#
# _BYTES_PER_TABLE_CELL charges each b x z cell conservatively: three
# live float64 copies during construction (the held tables ``rr`` and
# ``rho_rz``, plus the full-size temporary that the broadcast sum
# feeding ``sqrt`` materializes), one further reserved float64 slot as
# slack, and the one-byte boolean interaction mask: 4*8 + 1 = 33 B per
# cell. _MAX_TABLE_MEMORY_BYTES caps the resulting footprint estimate
# for the whole table stage at 4 GB. Under this accounting the
# reviewed admitted-but-fatal Pb208 plan (dr just above the old limit,
# ~2.27e4 x ~5.81e4 = 1.32e9 cells) estimates to ~44 GB and refuses up
# front instead of risking an OOM kill; the finest admitted Pb208 step
# remains near 0.002 fm, and both configured defaults clear the budget
# with two orders of magnitude of headroom.
_MAX_GRID_CELLS_PER_AXIS = 20_000
_BYTES_PER_TABLE_CELL = 33  # 4 float64 slots + 1 bool byte, see above
_MAX_TABLE_MEMORY_BYTES = 4_000_000_000

# Cell count reported for a grid axis whose planned length cannot be
# expressed as a finite double (see _arange_cell_count). Routing those
# requests through the smallest refuseable count makes them land in the
# ordinary per-axis budget refusal, whose message names dr_fm, the
# target, and the coarsening remedy, instead of leaking an
# OverflowError from math.ceil(float("inf")).
_GRID_SPAN_OVERFLOW_SENTINEL = _MAX_GRID_CELLS_PER_AXIS + 1


def _arange_cell_count(start: float, stop: float, step: float) -> int:
    """Return ``len(np.arange(start, stop, step))`` without allocating.

    NumPy sizes float64 ``arange`` spans as ``ceil((stop - start)/step)``
    elements (zero for a non-positive span), which this helper
    reproduces with plain Python floats so callers can plan exact grid
    lengths and their memory cost before building anything.

    A span that exceeds the finite double range has no integer length.
    This holds for an extremely small but positive step (the smallest
    subnormal divides any finite span infinitely often) and for a
    near-maximal ``stop``; the helper then reports
    ``_GRID_SPAN_OVERFLOW_SENTINEL`` so the caller's budget refusal
    fires with its documented message rather than ``math.ceil``
    raising ``OverflowError`` on infinity.
    """
    span = (stop - start) / step
    if math.isinf(span):
        return _GRID_SPAN_OVERFLOW_SENTINEL
    if math.isnan(span):
        # Defensive: production callers arrive with a finite positive
        # step and finite endpoints, which cannot produce a NaN span.
        return _GRID_SPAN_OVERFLOW_SENTINEL
    if span <= 0.0:
        return 0
    return math.ceil(span)


@dataclass(frozen=True)
class _GridPlan:
    """Pre-computed grid geometry for the extended-profile sampler.

    The ``n_*`` fields replicate the exact ``len()`` of the later
    ``np.arange`` grids, and the endpoint fields carry the very values
    those calls receive, so the allocation guard and the construction
    code share a single source of truth.
    """

    n_r: int
    n_b: int
    n_z: int
    r_stop: float
    z_half: float
    z_stop: float
    b_stop: float


def _plan_grids(r_max: float, surface: float, dr_fm: float) -> _GridPlan:
    """Count every sampler grid axis before any allocation happens.

    ``surface`` is the profile's surface scale (the Woods-Saxon
    diffuseness or the harmonic-oscillator radius); the z and b extents
    replicate the ``r_max + 10*surface`` and ``r_max + 3*surface``
    support rules the grid constructions apply below.

    The derived endpoints ``r_stop``, ``z_half``, ``z_stop`` and
    ``b_stop`` are finite-checked before counting. When any of them
    overflows double precision (huge-but-finite geometry inputs), every
    axis count is reported as ``_GRID_SPAN_OVERFLOW_SENTINEL`` so the
    caller's per-axis budget refusal fires with its documented message
    instead of an arithmetic ``OverflowError``.
    """
    r_stop = r_max + dr_fm
    z_half = r_max + 10.0 * surface
    z_stop = z_half + dr_fm
    b_stop = r_max + 3.0 * surface + dr_fm
    extents_finite = (
        math.isfinite(r_stop)
        and math.isfinite(z_half)
        and math.isfinite(z_stop)
        and math.isfinite(b_stop)
    )
    if extents_finite:
        n_r = _arange_cell_count(0.0, r_stop, dr_fm)
        n_b = _arange_cell_count(0.0, b_stop, dr_fm)
        n_z = _arange_cell_count(-z_half, z_stop, dr_fm)
    else:
        n_r = n_b = n_z = _GRID_SPAN_OVERFLOW_SENTINEL
    return _GridPlan(
        n_r=n_r,
        n_b=n_b,
        n_z=n_z,
        r_stop=r_stop,
        z_half=z_half,
        z_stop=z_stop,
        b_stop=b_stop,
    )


def _require_nucleus_int(value: object, name: str) -> int:
    """Normalize *value* to a Python ``int`` without fractional truncation.

    Python and NumPy integer scalars are accepted. Integer-valued
    finite floats (``208.0``) are accepted because they lose no
    information. Booleans are rejected even though they subclass
    ``int``; so are non-integral numerics (``208.5``), strings, and any
    other type. Every rejection raises ``ValueError`` naming *name*
    and echoing the offending value.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"NucleusSpec {name} must be an integer, got {value!r}"
        )
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        converted = int(value)
        if converted == value:
            return converted
    raise ValueError(
        f"NucleusSpec {name} must be an integer, got {value!r}"
    )


def _require_nucleus_float(value: object, name: str) -> float:
    """Normalize *value* to a Python ``float`` for one profile field.

    Python and NumPy numeric scalars (integers and floats) are accepted,
    and so are numeric strings such as ``"6.62"``, which configuration
    layers may deliver for otherwise plain numbers; parsing them here is
    lossless and keeps a raw string from ever being stored. Booleans are
    rejected even though ``float`` would coerce them: a truth flag is
    not a length or cross section. Any other type, or a string that
    ``float()`` cannot parse, raises ``ValueError`` naming *name* and
    the offending value. Value-level constraints (positivity,
    finiteness) remain the caller's business; this helper guarantees
    only the stored type.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"NucleusSpec {name} must be a float, got {value!r}"
        )
    if not isinstance(value, (int, float, np.integer, np.floating, str)):
        raise ValueError(
            f"NucleusSpec {name} must be a float, got {value!r}"
        )
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"NucleusSpec {name} must be a float, got {value!r}"
        ) from None


def _validate_spec_values(
    A: int,
    Z: int,
    profile: str,
    radius_fm: float,
    diffuseness_fm: float,
    quad_alpha: float,
    sigma_nn_inel_mb: float,
) -> None:
    """Reject inconsistent spec values after scalar normalization.

    Called from ``NucleusSpec.__post_init__`` once every scalar field is
    converted to built-in ``int``/``float``, so each value read here is
    exactly the value later consumers see. The NN cross section is
    validated for every target: an ``A == 1`` point target reads solely
    ``sigma_nn_inel_mb`` in its sampling branch, whereas ``radius_fm``,
    ``diffuseness_fm`` and ``quad_alpha`` go unused there and are
    exempt from its early return.
    """
    if A < 1:
        raise ValueError(f"NucleusSpec requires A >= 1, got {A!r}")
    if not (1 <= Z <= A):
        raise ValueError(
            f"NucleusSpec requires 1 <= Z <= A, got Z={Z!r}, A={A!r}"
        )
    if profile not in ("point", "woods_saxon", "harmonic_oscillator"):
        raise ValueError(
            f"NucleusSpec profile must be 'point', 'woods_saxon' or "
            f"'harmonic_oscillator', got {profile!r}"
        )
    if not math.isfinite(sigma_nn_inel_mb) or sigma_nn_inel_mb <= 0.0:
        raise ValueError(
            f"NucleusSpec sigma_nn_inel_mb must be finite and positive, "
            f"got {sigma_nn_inel_mb!r}"
        )
    if A == 1:
        # Point target: only sigma_nn_inel_mb reaches the sampler.
        return
    if not math.isfinite(quad_alpha) or quad_alpha <= -1.0:
        raise ValueError(
            f"NucleusSpec quad_alpha must be finite and > -1, got "
            f"{quad_alpha!r}"
        )
    if profile == "woods_saxon":
        if not (
            math.isfinite(radius_fm)
            and radius_fm > 0.0
            and math.isfinite(diffuseness_fm)
            and diffuseness_fm > 0.0
        ):
            raise ValueError(
                "NucleusSpec woods_saxon profile requires positive finite "
                f"radius_fm and diffuseness_fm, got R={radius_fm!r}, "
                f"a={diffuseness_fm!r}"
            )
    elif profile == "harmonic_oscillator":
        if not (
            math.isfinite(radius_fm)
            and radius_fm > 0.0
        ):
            raise ValueError(
                "NucleusSpec harmonic_oscillator profile requires a "
                f"positive finite radius_fm (the oscillator length a), "
                f"got {radius_fm!r}"
            )


@dataclass(frozen=True)
class NucleusSpec:
    """Target-nucleus specification for p+A generation.

    Attributes
    ----------
    name : str
        Free-form label used in manifests (e.g. ``"Pb208"``).
    A, Z : int
        Mass and charge number. ``A == 1`` forces the point-target
        branch (single free nucleon, K = 1 always).
    profile : str
        Density shape: ``"point"``, ``"woods_saxon"``, or
        ``"harmonic_oscillator"``.
    radius_fm : float
        Woods-Saxon half-density radius R, or the harmonic-oscillator
        length parameter ``a``.
    diffuseness_fm : float
        Woods-Saxon surface diffuseness.
    quad_alpha : float
        Quadratic deformation of the harmonic-oscillator profile,
        ``rho ~ (1 + alpha (r/a)^2) exp(-(r/a)^2)``; 0 gives the pure
        oscillator shape.
    sigma_nn_inel_mb : float
        Inelastic NN cross section at the collision energy in mb.

    Reference shapes used by the default light/heavy systems: Pb208
    follows the standard Woods-Saxon parameterization; C12 uses a pure
    harmonic oscillator whose length reproduces the measured charge rms
    radius (a ~ 1.07 fm, central point density ~0.6 /fm^3). Density
    parameters are configuration inputs; benchmark work should pin them
    against measured pA cross sections before quoting numbers.

    Notes
    -----
    Scalars are normalized at construction, so readers always observe
    built-in ``int`` and ``float``. ``A`` and ``Z`` accept Python or
    NumPy integer scalars and integer-valued finite floats such as
    ``208.0``; booleans, fractional numerics (``208.5``), and strings
    (``"208"``) are rejected.

    The profile fields ``radius_fm``, ``diffuseness_fm``, ``quad_alpha``
    and ``sigma_nn_inel_mb`` are stored as built-in ``float``. Numeric
    strings (``"6.62"``, as produced by some configuration layers) are
    parsed and converted alongside Python and NumPy numeric scalars;
    booleans and values that ``float()`` cannot parse raise
    ``ValueError`` naming the field. Parseability is checked on every
    profile branch, including the point target, so a malformed value is
    never stored; magnitude and positivity constraints then apply
    exactly where the corresponding branch consumes the field.
    """

    name: str = "generic"
    A: int = 208
    Z: int = 82
    profile: str = "woods_saxon"
    radius_fm: float = 6.62
    diffuseness_fm: float = 0.546
    quad_alpha: float = 0.0
    sigma_nn_inel_mb: float = 31.7

    def __post_init__(self) -> None:
        # Normalize every scalar field to built-in int/float before
        # validating, so the validation layer and all later consumers
        # read exactly what is stored (strings never survive raw).
        object.__setattr__(self, "A", _require_nucleus_int(self.A, "A"))
        object.__setattr__(self, "Z", _require_nucleus_int(self.Z, "Z"))
        object.__setattr__(
            self,
            "radius_fm",
            _require_nucleus_float(self.radius_fm, "radius_fm"),
        )
        object.__setattr__(
            self,
            "diffuseness_fm",
            _require_nucleus_float(self.diffuseness_fm, "diffuseness_fm"),
        )
        object.__setattr__(
            self,
            "quad_alpha",
            _require_nucleus_float(self.quad_alpha, "quad_alpha"),
        )
        object.__setattr__(
            self,
            "sigma_nn_inel_mb",
            _require_nucleus_float(self.sigma_nn_inel_mb, "sigma_nn_inel_mb"),
        )
        _validate_spec_values(
            self.A,
            self.Z,
            self.profile,
            self.radius_fm,
            self.diffuseness_fm,
            self.quad_alpha,
            self.sigma_nn_inel_mb,
        )

    def resolved_profile(self) -> str:
        return "point" if int(self.A) == 1 else self.profile


def _rho_woods_saxon(r: np.ndarray, R: float, a: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((np.asarray(r, dtype=float) - R) / a))


def _rho_harmonic_oscillator(
    r: np.ndarray, a: float, alpha: float
) -> np.ndarray:
    x = np.asarray(r, dtype=float) / a
    return (1.0 + alpha * x * x) * np.exp(-x * x)


class WoundedNucleonSampler:
    """Seeded sampler for impact parameter and target-wound multiplicity.

    Parameters
    ----------
    spec : NucleusSpec
        Target nucleus description.
    rng : numpy.random.Generator
        Source of randomness. The point-target branch (``A == 1``)
        never draws from it.
    dr_fm : float
        Radial grid step for thickness/CDF tables. Must be finite and
        positive on every profile branch, including the point target,
        and is bounded by the pre-computed allocation budget checked
        before any table allocation: each grid axis (``n_r``, ``n_b``,
        ``n_z``) holds at most ``_MAX_GRID_CELLS_PER_AXIS`` cells and
        the b x z thickness table's conservative memory estimate fits
        ``_MAX_TABLE_MEMORY_BYTES`` (rationale above the constants).
        Spans beyond double precision, from a subnormal step or
        huge-but-finite geometry, are refused by the same budget
        errors rather than an arithmetic OverflowError. A validated
        NumPy-scalar step is normalized to a built-in float before any
        planning arithmetic runs, so both carrier kinds share identical
        grid construction.
    """

    def __init__(
        self,
        spec: NucleusSpec,
        rng: np.random.Generator,
        dr_fm: float = 0.02,
    ) -> None:
        self.spec = spec
        self._rng = rng
        self.profile = spec.resolved_profile()
        # Validate before the point-target early return so an invalid
        # step raises on every profile branch.
        if not math.isfinite(float(dr_fm)) or float(dr_fm) <= 0.0:
            raise ValueError(
                "WoundedNucleonSampler dr_fm must be finite and positive, "
                f"got {dr_fm!r}"
            )
        # Normalize the validated step to a built-in float so every
        # downstream planning computation runs in pure-Python double
        # arithmetic. A NumPy-scalar step entering _plan_grids unchanged
        # would make its span divisions emit divide-overflow
        # RuntimeWarnings on the way to the intended budget ValueError,
        # contradicting this package's no-warning-on-rejection policy.
        # float() copies the exact double value, so plans and grids stay
        # bit-identical across both carrier kinds.
        dr_fm = float(dr_fm)
        if self.profile == "point":
            self._sigma_pA_mb = float(spec.sigma_nn_inel_mb)
            self._mean_target_wounds = 1.0
            return

        sigma_fm2 = float(spec.sigma_nn_inel_mb) / MB_PER_FM2

        # Density on a fine radial grid, normalized to unit integral.
        r_max = self._density_r_max()
        # Profile surface scale: diffuseness for woods_saxon, radius
        # for harmonic_oscillator (the same value the branches below
        # previously assigned).
        surface = (
            spec.diffuseness_fm
            if self.profile == "woods_saxon"
            else spec.radius_fm
        )
        # Allocation-budget guard, resolved with plain Python floats
        # and integers ahead of every np.arange or broadcast table:
        # each axis length and the conservative b x z memory estimate
        # must fit (byte accounting documented above the budget
        # constants), or construction refuses here with a descriptive
        # ValueError. Grids stay bit-identical: the plan endpoints are
        # the very stop values the arange calls below receive.
        plan = _plan_grids(r_max, surface, dr_fm)
        table_cells = plan.n_b * plan.n_z
        table_bytes = table_cells * _BYTES_PER_TABLE_CELL
        for axis_name, cells in (
            ("n_r", plan.n_r),
            ("n_b", plan.n_b),
            ("n_z", plan.n_z),
        ):
            if cells > _MAX_GRID_CELLS_PER_AXIS:
                raise ValueError(
                    "WoundedNucleonSampler grid-cell budget exceeded: "
                    f"dr_fm={dr_fm!r} for target {spec.name!r} plans "
                    f"{cells} cells along the {axis_name} grid axis "
                    f"(r_max = {r_max:.4g} fm), above the "
                    f"{_MAX_GRID_CELLS_PER_AXIS} budget; the b x z "
                    f"thickness table alone would hold {table_cells} "
                    f"cells (~{table_bytes / 1e9:.2f} GB estimated); "
                    "coarsen dr_fm."
                )
        if table_bytes > _MAX_TABLE_MEMORY_BYTES:
            raise ValueError(
                "WoundedNucleonSampler table-memory budget exceeded: "
                f"dr_fm={dr_fm!r} for target {spec.name!r} plans a "
                f"{plan.n_b} x {plan.n_z} = {table_cells}-cell b x z "
                "thickness table "
                f"({_BYTES_PER_TABLE_CELL} B/cell, "
                f"~{table_bytes / 1e9:.2f} GB estimated), above the "
                f"{_MAX_TABLE_MEMORY_BYTES}-byte "
                f"({_MAX_TABLE_MEMORY_BYTES / 1e9:g} GB) budget; "
                "coarsen dr_fm."
            )

        r_grid = np.arange(0.0, plan.r_stop, dr_fm)
        if self.profile == "woods_saxon":
            rho = _rho_woods_saxon(r_grid, spec.radius_fm, spec.diffuseness_fm)
        else:
            rho = _rho_harmonic_oscillator(
                r_grid, spec.radius_fm, spec.quad_alpha
            )
        norm = (
            4.0
            * math.pi
            * float(np.sum(rho * r_grid**2))
            * dr_fm
        )
        rho = rho / norm

        # Thickness T_A(b) via z-integration; b grid out to where the
        # profile support ends. The endpoint fields of the validated
        # plan are the stop values these arange calls consume.
        z_grid = np.arange(-plan.z_half, plan.z_stop, dr_fm)
        b_grid = np.arange(0.0, plan.b_stop, dr_fm)
        rr = np.sqrt(b_grid[:, None] ** 2 + z_grid[None, :] ** 2)
        rho_rz = np.zeros_like(rr)
        inside = rr <= r_max
        idx = np.clip((rr[inside] / dr_fm).astype(int), 0, len(r_grid) - 1)
        rho_rz[inside] = rho[idx]
        # Column density in nucleons per fm^2: the unit-shape density
        # carries no A, so the thickness scales with the mass number.
        thickness = rho_rz.sum(axis=1) * dr_fm * float(spec.A)

        mu = sigma_fm2 * thickness
        p_inel = -np.expm1(-mu)

        self._dr = dr_fm
        self._b_grid = b_grid
        self._mu = mu
        self._p_inel = p_inel

        # sigma_pA = integral P_inel d^2b = 2 pi sum P_inel b dr
        self._sigma_pA_fm2 = 2.0 * math.pi * float(np.sum(p_inel * b_grid)) * dr_fm
        self._sigma_pA_mb = self._sigma_pA_fm2 * MB_PER_FM2

        # Min-bias CDF over b from the interaction-weighted measure.
        cell = p_inel * b_grid * dr_fm
        cdf = np.cumsum(cell)
        total = cdf[-1]
        if total <= 0.0:
            raise ValueError(
                f"NucleusSpec {spec.name!r}: degenerate interaction "
                "probability (check sigma_nn_inel_mb and density)"
            )
        self._b_cdf = cdf / total

        # E[K | b] = mu / (1 - exp(-mu)); guard the transparent limit.
        e_k_given_b = np.zeros_like(mu)
        nz = p_inel > 0.0
        e_k_given_b[nz] = mu[nz] / (-np.expm1(-mu[nz]))
        self._mean_target_wounds = float(
            2.0 * math.pi * np.sum(p_inel * e_k_given_b * b_grid) * dr_fm
            / self._sigma_pA_fm2
        )

    def _density_r_max(self) -> float:
        spec = self.spec
        if self.profile == "woods_saxon":
            return spec.radius_fm + 10.0 * spec.diffuseness_fm
        return 8.0 * spec.radius_fm

    # ------------------------------------------------------------------
    # Closed-form quantities
    # ------------------------------------------------------------------
    def sigma_pA_mb(self) -> float:
        """Total inelastic p+A cross section in mb."""
        return self._sigma_pA_mb

    def mean_target_wounds_minbias(self) -> float:
        """Minimum-bias mean number of wounded target nucleons."""
        return self._mean_target_wounds

    def fraction_for_wound_range(self, lo: int, hi: int) -> float:
        """Expected minimum-bias fraction of events with lo <= K <= hi.

        The fraction follows the truncated-at-A support of the sampler:
        ranges with ``lo > A`` return exactly 0, and the probability
        mass of ``K >= A`` is piled onto the ``K = A`` bin, matching the
        tail assignment in :meth:`sample_conditional_wounds`. ``hi``
        beyond the densest bin's underflow point is harmless because the
        iteration stops once probabilities drop below 1e-16.
        """
        lo = max(int(lo), 1)
        hi = max(int(hi), lo)
        if self.profile == "point":
            return 1.0 if lo == 1 else 0.0
        marginal = self._slice_marginal_weights(lo, hi)
        return float(np.nansum(marginal)) / self._sigma_pA_fm2

    def _slice_marginal_weights(self, lo: int, hi: int) -> np.ndarray:
        """Interaction-weighted mass of ``lo <= K <= hi`` per b-bin.

        Returns the unnormalized array

            w_i = P_inel(b_i) * P(lo <= K <= hi | b_i) * 2 pi b_i dr

        on the sampler's b grid; ``sum(w) / sigma_pA`` equals
        :meth:`fraction_for_wound_range`. The conditional Poisson is
        the same truncated-at-A rule used by
        :meth:`sample_conditional_wounds`: the pmf runs up to
        ``k = min(hi, A)`` and, when ``hi >= A``, the tail probability
        beyond ``A`` is accumulated onto the ``K = A`` bin. The loop
        keeps the underflow stop of the analytic fractions (per-bin
        pmf peak below 1e-16 ends the accumulation), so both consumers
        of this helper share one truncation convention.
        """
        lo = max(int(lo), 1)
        hi = max(int(hi), lo)
        cap = max(int(self.spec.A), 1)
        marginal = np.zeros_like(self._mu)
        if lo > cap:
            return marginal
        mu = self._mu
        denom_safe = np.maximum(-np.expm1(-mu), 1e-300)
        emu = np.exp(-mu)

        def _pmf(k: int):
            return emu * mu**k / math.factorial(k) / denom_safe

        # Interaction-weighted measure cell: P_inel(b) * 2 pi b db.
        weight = self._p_inel * self._b_grid * self._dr * 2.0 * math.pi

        def _pmf_float(k: int) -> np.ndarray:
            # math.factorial(k) leaves the int64 range for k >= 21, so
            # the raw pmf can come back as an object-dtype array; cast
            # so the accumulation stays a float64 ufunc.
            return np.asarray(_pmf(k), dtype=float)

        k = lo
        while k <= min(hi, cap):
            try:
                pk = _pmf_float(k)
            except (OverflowError, FloatingPointError):
                break
            if k == cap:
                # Tail pile-up: the sampler returns K = cap for every
                # draw the untruncated pmf would place at k >= cap, so
                # the slice weight accumulates P(K >= cap | b) here.
                tail = pk + 0.0
                j = k
                while True:
                    j += 1
                    try:
                        pj = _pmf_float(j)
                    except (OverflowError, FloatingPointError):
                        break
                    tail = tail + pj
                    peak_j = float(np.nanmax(pj)) if pj.size else 0.0
                    if not math.isfinite(peak_j) or peak_j < 1e-16:
                        break
                marginal += tail * weight
                break
            marginal += pk * weight
            peak = float(np.nanmax(pk)) if pk.size else 0.0
            if not math.isfinite(peak) or peak < 1e-16:
                break
            k += 1
        return marginal

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_conditional_wounds(self, mu: float) -> int:
        """Draw K from the conditional Poisson P(K=k|b) for mean *mu*.

        Uses inversion over k with an underflow-based stop, so the
        routine terminates in bounded time even for mu near zero where
        naive rejection sampling would stall. The inversion is truncated
        at ``k_max = max(A, 1)``: K counts distinct wounded nucleons and
        cannot exceed A, so any tail probability accumulated beyond the
        cap is assigned to ``K = A`` (truncated-at-A Poisson).
        """
        if mu <= 0.0:
            return 1
        emu = math.exp(-mu)
        denom = -math.expm1(-mu)
        u = self._rng.random()
        cdf = 0.0
        pk = 0.0
        k = 0
        k_max = max(int(self.spec.A), 1)
        while True:
            k += 1
            if k == 1:
                pk = emu * mu / denom
            else:
                pk = pk * mu / k
            cdf += pk
            if u < cdf or k >= k_max:
                return min(k, k_max)
            if pk < 1e-16 and cdf >= 1.0 - 1e-9:
                return min(k, k_max)

    def sample_event(self) -> Tuple[float, int]:
        """Draw one interacting event; return ``(b_fm, k_target_wounds)``."""
        if self.profile == "point":
            return 0.0, 1
        u = self._rng.random()
        i = int(np.searchsorted(self._b_cdf, u, side="left"))
        i = min(i, len(self._b_grid) - 1)
        b = float(self._b_grid[i])
        return b, self.sample_conditional_wounds(float(self._mu[i]))

    def sample_event_in_range(self, lo: int, hi: int) -> Tuple[float, int]:
        """Draw one event conditioned on ``lo <= K <= hi``.

        Centrality slices are generated from the interaction-weighted
        distribution restricted to the slice, not by rejecting
        minimum-bias events. The joint measure of ``(b, K)`` under
        minimum bias factorizes at fixed *b* into ``P_inel(b) d^2b``
        times the truncated-at-A conditional Poisson; conditioning on
        ``lo <= K <= hi`` therefore samples

            b ~ P_inel(b) P(lo <= K <= hi | b) d^2b / sigma_slice,
            K ~ P(K = k | b) restricted to [lo, min(hi, A)],

        where ``sigma_slice`` is the numerator of
        :meth:`fraction_for_wound_range` and the restricted ``K`` law
        keeps the tail pile-up onto ``K = A`` when ``hi >= A``. The
        per-event cost is exactly two uniform draws, independent of
        how rare the slice is under minimum bias.

        Raises
        ------
        ValueError
            For structurally empty slices (``lo > A``, or a point
            target with a range excluding ``K = 1``) or slices whose
            total interaction-weighted probability underflows to zero.
        """
        lo = max(int(lo), 1)
        hi = max(int(hi), lo)
        cap = max(int(self.spec.A), 1)
        if self.profile == "point":
            if lo > 1:
                raise ValueError(
                    f"Point target has K = 1 always; wound range "
                    f"({lo}, {hi}) carries no probability."
                )
            return 0.0, 1
        if lo > cap:
            raise ValueError(
                f"Wound range ({lo}, {hi}) exceeds the truncated "
                f"support K <= A = {cap}; nothing to draw."
            )
        marginal = self._slice_marginal_weights(lo, hi)
        total = float(np.nansum(marginal))
        if not total > 0.0:
            raise ValueError(
                f"Wound range ({lo}, {hi}) carries no interaction-"
                f"weighted probability for nucleus {self.spec.name!r}."
            )
        cdf = np.cumsum(marginal)
        u_b = self._rng.random()
        i = int(np.searchsorted(cdf, u_b * total, side="left"))
        i = min(i, len(self._b_grid) - 1)
        b = float(self._b_grid[i])
        return b, self._sample_conditional_wounds_restricted(
            float(self._mu[i]), lo, hi
        )

    def _sample_conditional_wounds_restricted(
        self, mu: float, lo: int, hi: int
    ) -> int:
        """Draw K from ``P(K = k | b, lo <= k <= hi)`` at fixed mean *mu*.

        Inversion over the restricted support in log space.
        Algebraically this is the truncated-at-A draw of
        :meth:`sample_conditional_wounds` conditioned on the slice:
        the support stops at ``min(hi, A)``, and when ``hi >= A`` the
        ``K = A`` bin receives the full tail mass
        ``sum_{j >= A} exp(-mu) mu^j / j! / (1 - exp(-mu))``. Because
        the conditional normalization cancels all mu-only factors, the
        weights are proportional to ``mu^k / k!`` and cannot overflow.
        """
        cap = max(int(self.spec.A), 1)
        mu = float(mu)
        if mu <= 0.0:
            # Transparent bin: zero slice weight upstream, so this bin
            # is never selected; guard only against pathological grids.
            return max(lo, 1)
        k_top = min(hi, cap)

        def _log_term(j: int) -> float:
            return j * math.log(mu) - math.lgamma(j + 1.0)

        entries = [(k, _log_term(k)) for k in range(lo, k_top + 1)]
        if hi >= cap:
            # Tail pile-up onto K = cap, summed in log space until a
            # term falls e^-40 below the running maximum.
            terms = [_log_term(cap)]
            lmax = terms[0]
            j = cap
            while True:
                j += 1
                lt = _log_term(j)
                lmax = max(lmax, lt)
                terms.append(lt)
                if lmax - lt > 40.0:
                    break
            m = max(terms)
            log_tail = m + math.log(sum(math.exp(t - m) for t in terms))
            entries[-1] = (entries[-1][0], log_tail)

        shift = max(lw for _, lw in entries)
        weights = [math.exp(lw - shift) for _, lw in entries]
        total = sum(weights)
        u = self._rng.random() * total
        acc = 0.0
        for (k, _), w in zip(entries, weights):
            acc += w
            if u < acc:
                return k
        return entries[-1][0]
