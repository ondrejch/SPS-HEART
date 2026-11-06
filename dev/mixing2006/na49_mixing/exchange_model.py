# The legacy p+A path (marked 2026-11-04): the HEART p+p fits do not run this module; the p+A builder
# (event_builder_pA, parallel, glauber) and its imports do. Code reviews of the p+p model can skip it; review it
# with the p+A work. The cleanup and refactor after the p+A run decides its future.
"""Simple and Regge-informed source-production parameterizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SimpleExcitationParameters:
    mass_scale: float = 0.65
    edge_power: float = 1.5
    transfer_slope: float = 5.5
    grid_size: int = 96

    def validate(self) -> None:
        if not (math.isfinite(self.mass_scale) and self.mass_scale > 0.0):
            raise ValueError("mass_scale must be finite and positive")
        if not (math.isfinite(self.edge_power) and self.edge_power >= 0.0):
            raise ValueError("edge_power must be finite and non-negative")
        if not (math.isfinite(self.transfer_slope) and self.transfer_slope >= 0.0):
            raise ValueError("transfer_slope must be finite and non-negative")
        if not (16 <= int(self.grid_size) <= 512):
            raise ValueError("grid_size must lie in [16, 512]")


@dataclass(frozen=True)
class ReggeExcitationParameters:
    alpha_p0: float = 1.0808
    alpha_p_prime: float = 0.25
    alpha_exc0: float = 0.50
    transfer_slope: float = 4.5
    edge_power: float = 2.0
    s0: float = 1.0
    grid_size: int = 96
    effective_transfer_slope: float | None = None

    def validate(self) -> None:
        values = asdict(self)
        for key, value in values.items():
            if key in {"grid_size", "effective_transfer_slope"}:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
        if self.effective_transfer_slope is not None:
            effective = float(self.effective_transfer_slope)
            if not (math.isfinite(effective) and effective > 0.0):
                raise ValueError("effective_transfer_slope must be finite and positive")
        if self.alpha_p_prime < 0.0 or self.transfer_slope < 0.0:
            raise ValueError("Regge transfer slopes must be non-negative")
        if self.edge_power < 0.0 or self.s0 <= 0.0:
            raise ValueError("edge_power must be non-negative and s0 positive")
        if not (16 <= int(self.grid_size) <= 512):
            raise ValueError("grid_size must lie in [16, 512]")

    def effective_slope(self, s: float) -> float:
        # The normalized fixed-energy event density cannot identify alpha_p0:
        # it contributes only an overall factor. alpha_p_prime does affect the
        # t shape through exp[2 alpha'_P t log(s/s0)], so at one beam energy
        # only the combination B_eff = B + 2 alpha'_P log(s/s0) is
        # identifiable. When the configured effective_transfer_slope is set
        # (single supported beam energy) it overrides the combination and the
        # t distribution at the beam energy is sampled with it directly; the
        # components transfer_slope and alpha_p_prime stay frozen inputs that
        # define the bundled default through the combination.
        if self.effective_transfer_slope is not None:
            return float(self.effective_transfer_slope)
        return self.transfer_slope + 2.0 * self.alpha_p_prime * math.log(s / self.s0)


class SourceMassSampler:
    """Normalized two-source mass sampler on a finite-grid triangle.

    Cell weights are densities with respect to the measure dM1 dM2 on the
    uniform mass grid. Each rectangular cell is weighted at its center and
    sampled uniformly; draws with ``M1 + M2 >= sqrt(s)`` are rejected. Cells
    that cross the triangular boundary therefore integrate the density only
    approximately. In ``regge`` mode the production law is documented as a
    density in dM1^2 dM2^2, so the cell weight carries the Jacobian
    dM1^2 dM2^2 = 4 M1 M2 dM1 dM2; the sampled mass-space density is the
    documented squared-mass law times 4 M1 M2 times the edge suppression.
    In ``simple`` mode the density is a dM1 dM2 density by construction.
    """

    def __init__(
        self,
        sqrt_s: float,
        minimum_mass1: float,
        minimum_mass2: float,
        mode: str,
        parameters: SimpleExcitationParameters | ReggeExcitationParameters,
        rng: np.random.Generator,
    ) -> None:
        self.sqrt_s = float(sqrt_s)
        self.m1_min = float(minimum_mass1)
        self.m2_min = float(minimum_mass2)
        self.mode = str(mode)
        self.parameters = parameters
        self.rng = rng
        parameters.validate()
        if self.m1_min + self.m2_min >= self.sqrt_s:
            raise ValueError("source minima leave no two-body phase space")
        self._prepare()

    def _prepare(self) -> None:
        n = int(self.parameters.grid_size)
        self._e1 = np.linspace(self.m1_min, self.sqrt_s - self.m2_min, n + 1)
        self._e2 = np.linspace(self.m2_min, self.sqrt_s - self.m1_min, n + 1)
        c1 = 0.5 * (self._e1[:-1] + self._e1[1:])
        c2 = 0.5 * (self._e2[:-1] + self._e2[1:])
        m1, m2 = np.meshgrid(c1, c2, indexing="ij")
        edge = 1.0 - (m1 + m2) / self.sqrt_s
        valid = edge > 0.0
        logw = np.full_like(m1, -np.inf)
        if self.mode == "simple":
            p = self.parameters
            assert isinstance(p, SimpleExcitationParameters)
            logw[valid] = (
                -(m1[valid] - self.m1_min + m2[valid] - self.m2_min) / p.mass_scale
                + p.edge_power * np.log(edge[valid])
            )
        elif self.mode == "regge":
            p = self.parameters
            assert isinstance(p, ReggeExcitationParameters)
            # The documented ansatz is a density in dM1^2 dM2^2. The grid is
            # uniform in M1, M2, so the cell weight carries the Jacobian
            # dM1^2 dM2^2 = 4 M1 M2 dM1 dM2.
            logw[valid] = (
                (p.alpha_exc0 - 1.0)
                * (
                    np.log((m1[valid] * m1[valid]) / p.s0)
                    + np.log((m2[valid] * m2[valid]) / p.s0)
                )
                + p.edge_power * np.log(edge[valid])
                + np.log(4.0 * m1[valid] * m2[valid])
            )
        else:
            raise ValueError(f"unknown source-mass mode {self.mode!r}")
        finite = np.isfinite(logw)
        peak = float(np.max(logw[finite]))
        weights = np.zeros_like(logw)
        weights[finite] = np.exp(logw[finite] - peak)
        flat = weights.ravel()
        total = float(flat.sum())
        if not total > 0.0:
            raise ValueError("source-mass density has zero normalization")
        self._cdf = np.cumsum(flat / total)

    def sample(self) -> tuple[float, float]:
        n2 = len(self._e2) - 1
        for _ in range(1000):
            index = int(np.searchsorted(self._cdf, self.rng.random(), side="right"))
            index = min(index, self._cdf.size - 1)
            i, j = divmod(index, n2)
            m1 = self.rng.uniform(self._e1[i], self._e1[i + 1])
            m2 = self.rng.uniform(self._e2[j], self._e2[j + 1])
            if m1 + m2 < self.sqrt_s:
                return float(m1), float(m2)
        raise RuntimeError("failed to draw a source-mass pair inside the physical boundary")


def transfer_slope(
    s: float,
    mode: str,
    parameters: SimpleExcitationParameters | ReggeExcitationParameters,
) -> float:
    if mode == "simple":
        assert isinstance(parameters, SimpleExcitationParameters)
        return parameters.transfer_slope
    assert isinstance(parameters, ReggeExcitationParameters)
    return parameters.effective_slope(s)
