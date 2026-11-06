# The legacy p+A path (marked 2026-11-04): the HEART p+p fits do not run this module; the p+A builder
# (event_builder_pA, parallel, glauber) and its imports do. Code reviews of the p+p model can skip it; review it
# with the p+A work. The cleanup and refactor after the p+A run decides its future.
"""Identified particle — port of ``TIdParticle``.

Stores four-momentum and particle identity in the lab frame.
All kinematic observables (CMS quantities, invariant mass, …) are computed
on the fly from the stored lab-frame momentum components.
"""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

from .constants import EMELEK, EMPI, EMKA, EMPROT
from .pdg_database import get_mass, get_charge, is_analysis_stable
from .state_validation import UnrecognizedStateType, state_type_of

if TYPE_CHECKING:
    from .reaction import Reaction

_PI = math.pi


class Particle:
    """An identified particle.

    Parameters
    ----------
    px, py, pz : float
        Lab-frame momentum components (GeV/c).
    charge : float
        Charge in units of *e*.
    mass : float
        Rest mass (GeV/c²).
    pdg_id : int
        PDG Monte Carlo particle ID.
    reaction : Reaction, optional
        Reaction object providing Lorentz boost parameters.
    """

    __slots__ = (
        "_px", "_py", "_pz",
        "_charge", "_mass", "_pdg_id",
        "_is_stable",
        "_n_point",
        "_pflag",
        "_sigbb",
        "reaction",
        "parent",
        "daughter",
        "next_daughter",
        "_weak_feed",
        "_gen_depth",
        "_is_primary",
        "_source_id",
        "state_type",
    )

    def __init__(
        self,
        px: float = 0.0,
        py: float = 0.0,
        pz: float = 0.0,
        charge: float = 0.0,
        mass: float = 0.0,
        pdg_id: int = 0,
        reaction: Optional["Reaction"] = None,
    ) -> None:
        self._px = float(px)
        self._py = float(py)
        self._pz = float(pz)
        self._charge = float(charge)
        self._mass = float(mass)
        self._pdg_id = int(pdg_id)
        self._is_stable: Optional[bool] = None
        self._n_point: int = 0
        self._pflag: int = 0          # final-state baryon flag
        self._sigbb = [999.0, 999.0, 999.0, 999.0]
        self.reaction: Optional["Reaction"] = reaction
        self.parent: Optional["Particle"] = None
        self.daughter: Optional["Particle"] = None
        self.next_daughter: Optional["Particle"] = None
        # Ancestry bookkeeping for feed-down studies:
        #   _weak_feed  -- 1 if ANY ancestor underwent a declared weak decay
        #   _gen_depth  -- cascade generations since the generated parent
        self._weak_feed: int = 0
        self._gen_depth: int = 0
        # 1 if this particle was registered as a generated primary parent
        # (Event.mark_primary); daughters of cascade decays keep 0 so that
        # primary decay outcomes can be counted separately.
        self._is_primary: int = 0
        # Source-track identity used by event mixing: -1 means "unassigned";
        # values >= 0 are stamped by MixBuffer.process_event so that the A-
        # and B-list copies of one physical track share one per-event
        # identifier. Consumers of the buffered lists can use it to
        # recognize the copies of one track; cross-event mixed pairs never
        # compare identifiers.
        self._source_id: int = -1
        # Closed five-value state-type label (plan section 6, tasks 7-8;
        # see na49_mixing.state_validation).  Stamped at construction time
        # via ``state_type_of`` for the legacy Generator path.  ``None`` is
        # a missing label (e.g. a particle with no PDG identity yet, such
        # as a decay daughter before ``set_id``), not an invalid label.
        self.state_type = self._stamp_state_type()

    def _stamp_state_type(self):
        """Stamp the state-type label from the current PDG identity.

        Returns the label, or ``None`` when the PDG id is not a hadron of
        the vendored PDG mass table with integer ``(B, Q, S)`` (a missing
        label, not an invalid label).
        """
        try:
            return state_type_of(self._pdg_id, "final_particle")
        except UnrecognizedStateType:
            return None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_id(
        cls,
        px: float,
        py: float,
        pz: float,
        pdg_id: int,
        reaction: Optional["Reaction"] = None,
    ) -> "Particle":
        """Construct a particle from its PDG id; mass/charge from database."""
        mass = get_mass(pdg_id, default=-10.0)
        charge = get_charge(pdg_id, default=0.0)
        p = cls(px, py, pz, charge, mass, pdg_id, reaction)
        p._is_stable = is_analysis_stable(pdg_id)
        return p

    @classmethod
    def copy(cls, other: "Particle") -> "Particle":
        """Shallow copy of *other* (does not copy parent/daughter links)."""
        p = cls(
            other._px, other._py, other._pz,
            other._charge, other._mass, other._pdg_id,
            other.reaction,
        )
        p._is_stable = other._is_stable
        p._weak_feed = other._weak_feed
        p._gen_depth = other._gen_depth
        p._is_primary = other._is_primary
        p._n_point = other._n_point
        p._pflag = other._pflag
        p._sigbb = list(other._sigbb)
        p._source_id = other._source_id
        p.state_type = other.state_type
        return p

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    def get_px(self) -> float: return self._px
    def get_py(self) -> float: return self._py
    def get_pz(self) -> float: return self._pz
    def get_mass(self) -> float: return self._mass
    def get_id(self) -> int: return self._pdg_id
    def get_charge(self) -> float: return self._charge
    def get_n_point(self) -> int: return self._n_point
    def is_fsb(self) -> int: return self._pflag
    def get_weak_feed(self) -> int: return self._weak_feed
    def get_gen_depth(self) -> int: return self._gen_depth
    def set_weak_feed(self, v: int) -> None: self._weak_feed = int(v)
    def set_gen_depth(self, v: int) -> None: self._gen_depth = int(v)
    def set_is_primary(self, v) -> None: self._is_primary = int(bool(v))
    def get_is_primary(self) -> int: return self._is_primary
    def get_source_id(self) -> int: return self._source_id
    def set_source_id(self, v: int) -> None: self._source_id = int(v)

    def set_p(self, px: float, py: float, pz: float) -> None:
        self._px = float(px)
        self._py = float(py)
        self._pz = float(pz)

    def set_mass(self, mass: float) -> None:
        self._mass = float(mass)

    def set_id(self, pdg_id: int) -> None:
        """Set PDG id and update mass/charge/stability from database."""
        self._pdg_id = int(pdg_id)
        m = get_mass(pdg_id, default=float("nan"))
        if math.isfinite(m) and m > 0.0:
            self._mass = m
        # Unknown PDG keeps the previous mass rather than inventing a negative
        # sentinel that can poison energy/momentum with NaNs later.
        self._charge = get_charge(pdg_id, default=self._charge)
        self._is_stable = is_analysis_stable(pdg_id)
        self.state_type = self._stamp_state_type()

    def set_n_point(self, n: int) -> None:
        self._n_point = int(n)

    def set_fsb(self, flag: int) -> None:
        self._pflag = int(flag)

    def set_sbb(self, gi: int, val: float) -> None:
        if 0 <= gi < 4:
            self._sigbb[gi] = float(val)

    def get_sbb(self, gi: int) -> float:
        if 0 <= gi < 4:
            return self._sigbb[gi]
        return 9999.0

    # ------------------------------------------------------------------
    # Derived kinematic quantities
    # ------------------------------------------------------------------
    def get_e(self) -> float:
        """Total energy in the lab frame (GeV)."""
        return math.sqrt(
            self._mass ** 2 + self._px ** 2 + self._py ** 2 + self._pz ** 2
        )

    def get_p(self) -> float:
        """Total momentum magnitude (GeV/c)."""
        return math.sqrt(self._px ** 2 + self._py ** 2 + self._pz ** 2)

    def get_pt(self) -> float:
        """Transverse momentum (GeV/c)."""
        return math.sqrt(self._px ** 2 + self._py ** 2)

    def get_phi(self) -> float:
        """Azimuthal angle in degrees (0° on the 'good' side for the charge)."""
        return math.atan2(self._py, self._px * self._charge) * 180.0 / _PI

    # ------------------------------------------------------------------
    # CMS-frame quantities (require a Reaction object)
    # ------------------------------------------------------------------
    def _require_reaction(self) -> "Reaction":
        if self.reaction is None:
            raise RuntimeError("Particle has no associated Reaction object")
        return self.reaction

    def get_pz_cms(self, reac: Optional["Reaction"] = None) -> float:
        """Longitudinal momentum in the CMS frame (GeV/c)."""
        r = reac if reac is not None else self._require_reaction()
        # Directly use cached boost parameters from Reaction to avoid repeated
        # tiny method-call overhead in high-statistics loops.
        return r._gamma * self._pz - (r._betha * r._gamma) * self.get_e()

    def get_e_cms(self, reac: Optional["Reaction"] = None) -> float:
        """Energy in the CMS frame (GeV)."""
        r = reac if reac is not None else self._require_reaction()
        pz_cms = self.get_pz_cms(r)
        return math.sqrt(
            self._mass ** 2 + self._px ** 2 + self._py ** 2 + pz_cms ** 2
        )

    def get_xf(self, reac: Optional["Reaction"] = None) -> float:
        """Feynman *x_F* = p_z^CMS / p_max."""
        r = reac if reac is not None else self._require_reaction()
        return self.get_pz_cms(r) * (2.0 / r._fS)

    def get_p_cms(self, reac: Optional["Reaction"] = None) -> float:
        """Total momentum in the CMS frame (GeV/c)."""
        r = reac if reac is not None else self._require_reaction()
        pz_cms = self.get_pz_cms(r)
        return math.sqrt(self._px ** 2 + self._py ** 2 + pz_cms ** 2)

    def get_y_cms(self, reac: Optional["Reaction"] = None) -> float:
        """Rapidity in the CMS frame."""
        r = reac if reac is not None else self._require_reaction()
        # Compute pz_cms once and derive e_cms inline to avoid calling
        # get_e_cms() (which would call get_pz_cms() a second time).
        pz = self.get_pz_cms(r)
        e = math.sqrt(
            self._mass ** 2 + self._px ** 2 + self._py ** 2 + pz ** 2
        )
        denom = e + pz
        numer = e - pz
        if denom <= 0.0 or numer <= 0.0:
            # Preserve rapidity sign in the guarded limit:
            # forward (pz>0) -> large +y, backward (pz<0) -> large -y.
            return 1e10 if pz > 0.0 else -1e10
        return -0.5 * math.log(numer / denom)

    # ------------------------------------------------------------------
    # Invariant mass
    # ------------------------------------------------------------------
    def get_inv_mass(self, other: "Particle") -> float:
        """Invariant mass of the pair (self, other) (GeV/c²)."""
        m1 = self._mass
        m2 = other._mass
        val = (
            m1 * m1 + m2 * m2
            + 2.0 * self.get_e() * other.get_e()
            - 2.0 * self._px * other._px
            - 2.0 * self._py * other._py
            - 2.0 * self._pz * other._pz
        )
        return math.sqrt(max(val, 0.0))

    def get_inv_mass3(self, p1: "Particle", p2: "Particle") -> float:
        """Invariant mass of the triple (self, p1, p2)."""
        e = self.get_e() + p1.get_e() + p2.get_e()
        px = self._px + p1._px + p2._px
        py = self._py + p1._py + p2._py
        pz = self._pz + p1._pz + p2._pz
        return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------
    def is_stable(self) -> bool:
        if self._is_stable is not None:
            return self._is_stable
        self._is_stable = is_analysis_stable(self._pdg_id)
        return self._is_stable

    def set_stable(self, stable: bool) -> None:
        self._is_stable = bool(stable)

    # ------------------------------------------------------------------
    # Arithmetic / combination
    # ------------------------------------------------------------------
    def add(self, other: "Particle") -> None:
        """Add *other* four-momentum to self (in place)."""
        e = self.get_e() + other.get_e()
        self._px += other._px
        self._py += other._py
        self._pz += other._pz
        self._charge += other._charge
        self._mass = math.sqrt(max(
            e * e - self._px ** 2 - self._py ** 2 - self._pz ** 2,
            0.0,
        ))
        # Card-mandated stamp point (plan section 6, tasks 7-8): after the
        # merge, re-assert the state-type label from the surviving PDG
        # identity.  The identity does not change here, so this is a
        # no-op for particles stamped at construction; it fills in a
        # missing label when the merged object gained a table identity.
        if self.state_type is None:
            self.state_type = self._stamp_state_type()

    # ------------------------------------------------------------------
    # Bethe-Bloch dE/dx
    # ------------------------------------------------------------------
    def bethe_bloch(self, bg: float) -> float:
        """Bethe-Bloch dE/dx function (ported from ``TIdParticle::bethe_bloch``)."""
        alfa = -2.0
        min_val = 3.77
        x0 = 1.44
        x1 = 4.0
        pl = 1.53

        bmin = min_val / math.sqrt(1.0 + min_val * min_val)
        a = -alfa / 2.0 * (1.0 - bmin * bmin) / (bmin ** (alfa + 4.0))
        b = (min_val * min_val * (1.0 - bmin * bmin * (1.0 + 2.0 / alfa))
             + math.log(1.0 - bmin * bmin))
        m = (x1 - x0) / ((pl / a - b + 1.0) / 2.0 / math.log(10.0) - x0)

        x = math.log10(bg)
        d = 0.0
        if x > x0:
            d = x - x0 + (x0 - x1) / m
            if x < x1:
                d += ((x1 - x) ** m) / (m * (x1 - x0) ** (m - 1.0))
        d *= 2.0 * math.log(10.0)

        beta = bg / math.sqrt(1.0 + bg * bg)
        gamma = bg / beta
        f = a * beta ** alfa * (b + 2.0 * math.log(gamma) - beta * beta - d)
        # truncated mean correction
        f *= 1.0 + 0.1800 * math.log(f) - 0.0155 * (math.log(f)) ** 2
        return f

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Particle(id={self._pdg_id}, "
            f"px={self._px:.4f}, py={self._py:.4f}, pz={self._pz:.4f}, "
            f"m={self._mass:.4f}, q={self._charge:.0f})"
        )
