"""Reaction kinematics — Lorentz boost parameters for LAB↔CMS frame.

Ported from ``Treaction.h`` / ``Treaction.cxx``.

Default beam momentum is 158 GeV/c (NA49 SPS Pb+Pb fixed-target) with a
proton beam on a proton target. ``beam_pdg`` / ``target_pdg`` select the
colliding nucleon species (2212 = proton, 2112 = neutron) so elementary
n+p and n+n reactions use the correct masses in the invariant-mass and
boost computation.
"""

from __future__ import annotations
import math
from .constants import EMPROT
from .pdg_database import get_particle, ParticleData
from typing import Optional


class Reaction:
    """Stores the Lorentz boost parameters for a fixed-target reaction.

    Parameters
    ----------
    plab : float
        Lab-frame beam momentum in GeV/c (default 158 GeV/c).
    beam_pdg : int
        PDG id of the beam nucleon (default 2212).
    target_pdg : int
        PDG id of the target nucleon (default 2212).

    Notes
    -----
    The fixed-target invariant mass is s = m_b² + m_t² + 2 m_t E_b^lab.
    For the default proton-proton configuration the original closed-form
    expression is evaluated so that float results stay bit-identical to
    the legacy implementation; component caches depend on that.

    ``plab`` must be a finite positive momentum. Negative momenta are
    unsupported by declaration: the beam energy enters through p², so
    the constructor would silently equate a negative momentum to its
    positive magnitude while flipping the sign of the CMS velocity β.
    """

    def __init__(
        self,
        plab: float = 158.0,
        beam_pdg: int = 2212,
        target_pdg: int = 2212,
    ) -> None:
        plab_value = float(plab)
        if not math.isfinite(plab_value) or plab_value <= 0.0:
            raise ValueError(
                "Reaction plab must be a finite positive GeV/c beam "
                f"momentum, got {plab!r}"
            )
        self._plab = plab_value
        self._beam_pdg = int(beam_pdg)
        self._target_pdg = int(target_pdg)
        self._beam_mass = self._resolve_mass(self._beam_pdg, "beam")
        self._target_mass = self._resolve_mass(self._target_pdg, "target")
        self._compute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_mass(pdg_id: int, role: str) -> float:
        data = get_particle(int(pdg_id))
        if data is None:
            raise ValueError(f"Unknown {role} PDG id {pdg_id} for Reaction.")
        return float(data.mass)

    def _compute(self) -> None:
        p = self._plab
        m_b = self._beam_mass
        m_t = self._target_mass
        e_beam = math.sqrt(m_b * m_b + p * p)
        # total energy in LAB
        self._energy_tot = e_beam + m_t
        if m_b == EMPROT and m_t == EMPROT:
            # Legacy p+p closed form, kept bit-identical to the original
            # expression (regression gate compares exact track floats).
            self._fS = math.sqrt(
                2.0 * EMPROT * (EMPROT + math.sqrt(EMPROT * EMPROT + p * p))
            )
        else:
            # sqrt(s) of the reaction: s = m_b² + m_t² + 2 m_t E_b^lab
            self._fS = math.sqrt(m_b * m_b + m_t * m_t + 2.0 * m_t * e_beam)
        # beta and gamma of the CMS
        self._betha = p / self._energy_tot
        self._gamma = 1.0 / math.sqrt(1.0 - self._betha ** 2)

    # ------------------------------------------------------------------
    # Public accessors (naming mirrors the C++ API)
    # ------------------------------------------------------------------
    def s(self) -> float:
        """Mandelstam *s* of the reaction (GeV²)."""
        return self._fS * self._fS

    def sqrt_s(self) -> float:
        """√s — maximum available energy (GeV)."""
        return self._fS

    def betha(self) -> float:
        """β of the CMS frame."""
        return self._betha

    def gamma(self) -> float:
        """γ of the CMS frame."""
        return self._gamma

    def bet_gam(self) -> float:
        """β·γ of the CMS frame."""
        return self._betha * self._gamma

    def plmax(self) -> float:
        """Maximum allowed longitudinal momentum = √s / 2 (GeV/c)."""
        return self._fS / 2.0

    def plab(self) -> float:
        """Lab-frame beam momentum (GeV/c)."""
        return self._plab

    # ------------------------------------------------------------------
    # Species accessors
    # ------------------------------------------------------------------
    def beam_pdg(self) -> int:
        """PDG id of the beam nucleon."""
        return self._beam_pdg

    def target_pdg(self) -> int:
        """PDG id of the target nucleon."""
        return self._target_pdg

    def beam_mass(self) -> float:
        """Beam nucleon mass (GeV/c²)."""
        return self._beam_mass

    def target_mass(self) -> float:
        """Target nucleon mass (GeV/c²)."""
        return self._target_mass

    # ------------------------------------------------------------------
    # PDG database access
    # ------------------------------------------------------------------
    def get_base_particle(self, pdg_id: int) -> Optional[ParticleData]:
        """Return particle data for *pdg_id*, or ``None``."""
        return get_particle(pdg_id)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Reaction(plab={self._plab} GeV/c, beam_pdg={self._beam_pdg}, "
            f"target_pdg={self._target_pdg}, sqrt_s={self._fS:.4f} GeV, "
            f"beta={self._betha:.6f}, gamma={self._gamma:.4f})"
        )
