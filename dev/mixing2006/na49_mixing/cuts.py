"""Analysis cuts — port of ``TParticleCut``, ``TEventCut``, ``TPairCut``.

:class:`ParticleCut`
    Configurable cut on a single :class:`~na49_mixing.particle.Particle`.
:class:`EventCut`
    Cut on event-level quantities (multiplicity, grey tracks, …).
:class:`PairCut`
    Cut on a pair of particles (opening angle, cos θ*, …).
"""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .particle import Particle
    from .event import Event


# ---------------------------------------------------------------------------
# ParticleCut
# ---------------------------------------------------------------------------

class ParticleCut:
    """Boolean AND of configurable cuts on a particle.

    Each cut is independently switched on via ``set_*_cut()`` methods.
    All active cuts must pass (logical AND) for :meth:`check_particle`
    to return ``True``.
    """

    def __init__(self) -> None:
        self._xf_cut = False
        self._pt_cut = False
        self._id_cut = False
        self._px_cut = False
        self._py_cut = False
        self._pz_cut = False
        self._e_cut = False
        self._pzcms_cut = False
        self._ecms_cut = False
        self._npoint_cut = False
        self._mass_cut = False

        self._xf_min = self._xf_max = 0.0
        self._pt_min = self._pt_max = 0.0
        self._id = 0
        self._px_min = self._px_max = 0.0
        self._py_min = self._py_max = 0.0
        self._pz_min = self._pz_max = 0.0
        self._e_min = self._e_max = 0.0
        self._pzcms_min = self._pzcms_max = 0.0
        self._ecms_min = self._ecms_max = 0.0
        self._np = 0
        self._mass_min = self._mass_max = 0.0
        self._n_should = 0

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------
    def set_id_cut(self, pdg_id: int) -> None:
        if not self._id_cut:
            self._n_should += 1
        self._id_cut = True
        self._id = int(pdg_id)

    def set_xf_cut(self, xf_min: float, xf_max: float) -> None:
        if not self._xf_cut:
            self._n_should += 1
        self._xf_cut = True
        self._xf_min = float(xf_min)
        self._xf_max = float(xf_max)

    def set_pt_cut(self, pt_min: float, pt_max: float) -> None:
        if not self._pt_cut:
            self._n_should += 1
        self._pt_cut = True
        self._pt_min = float(pt_min)
        self._pt_max = float(pt_max)

    def set_px_cut(self, px_min: float, px_max: float) -> None:
        if not self._px_cut:
            self._n_should += 1
        self._px_cut = True
        self._px_min = float(px_min)
        self._px_max = float(px_max)

    def set_py_cut(self, py_min: float, py_max: float) -> None:
        if not self._py_cut:
            self._n_should += 1
        self._py_cut = True
        self._py_min = float(py_min)
        self._py_max = float(py_max)

    def set_pz_cut(self, pz_min: float, pz_max: float) -> None:
        if not self._pz_cut:
            self._n_should += 1
        self._pz_cut = True
        self._pz_min = float(pz_min)
        self._pz_max = float(pz_max)

    def set_e_cut(self, e_min: float, e_max: float) -> None:
        if not self._e_cut:
            self._n_should += 1
        self._e_cut = True
        self._e_min = float(e_min)
        self._e_max = float(e_max)

    def set_pzcms_cut(self, pz_min: float, pz_max: float) -> None:
        if not self._pzcms_cut:
            self._n_should += 1
        self._pzcms_cut = True
        self._pzcms_min = float(pz_min)
        self._pzcms_max = float(pz_max)

    def set_ecms_cut(self, e_min: float, e_max: float) -> None:
        if not self._ecms_cut:
            self._n_should += 1
        self._ecms_cut = True
        self._ecms_min = float(e_min)
        self._ecms_max = float(e_max)

    def set_npoint_cut(self, npoint: int) -> None:
        if not self._npoint_cut:
            self._n_should += 1
        self._npoint_cut = True
        self._np = int(npoint)

    def off_npoint_cut(self) -> None:
        if self._npoint_cut:
            self._n_should -= 1
        self._npoint_cut = False

    def set_mass_cut(self, mass_min: float, mass_max: float) -> None:
        if not self._mass_cut:
            self._n_should += 1
        self._mass_cut = True
        self._mass_min = float(mass_min)
        self._mass_max = float(mass_max)

    def reset(self) -> None:
        """Switch off all cuts."""
        # Class-level __init__ call: identical runtime effect to
        # ``self.__init__()``, expressed soundly for the type checker.
        ParticleCut.__init__(self)

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------
    def check_particle(self, particle: "Particle") -> bool:
        """Return ``True`` if *particle* passes all active cuts."""
        if self._n_should == 0:
            return True
        if self._id_cut:
            if particle.get_id() != self._id:
                return False
        if self._npoint_cut:
            if particle.get_n_point() < self._np:
                return False
        if self._mass_cut:
            if not (self._mass_min <= particle.get_mass() <= self._mass_max):
                return False
        if self._px_cut:
            if not (self._px_min <= particle.get_px() <= self._px_max):
                return False
        if self._py_cut:
            if not (self._py_min <= particle.get_py() <= self._py_max):
                return False
        if self._pz_cut:
            if not (self._pz_min <= particle.get_pz() <= self._pz_max):
                return False
        if self._pt_cut:
            if not (self._pt_min <= particle.get_pt() <= self._pt_max):
                return False
        if self._e_cut:
            if not (self._e_min <= particle.get_e() <= self._e_max):
                return False
        if self._xf_cut:
            if not (self._xf_min <= particle.get_xf() <= self._xf_max):
                return False
        if self._pzcms_cut:
            if not (self._pzcms_min <= particle.get_pz_cms() <= self._pzcms_max):
                return False
        if self._ecms_cut:
            if not (self._ecms_min <= particle.get_e_cms() <= self._ecms_max):
                return False
        return True

    # ------------------------------------------------------------------
    # Structural comparison
    # ------------------------------------------------------------------
    def is_equivalent(self, other: "ParticleCut") -> bool:
        """Return ``True`` when *other* selects exactly the same particles.

        Two cuts are equivalent when the same set of individual cuts is
        active and every active cut carries identical bounds. Event mixing
        uses this to detect identical A/B selections, where each unordered
        particle pair must be counted once instead of twice.
        """
        checks = (
            ("_xf_cut", ("_xf_min", "_xf_max")),
            ("_pt_cut", ("_pt_min", "_pt_max")),
            ("_id_cut", ("_id",)),
            ("_px_cut", ("_px_min", "_px_max")),
            ("_py_cut", ("_py_min", "_py_max")),
            ("_pz_cut", ("_pz_min", "_pz_max")),
            ("_e_cut", ("_e_min", "_e_max")),
            ("_pzcms_cut", ("_pzcms_min", "_pzcms_max")),
            ("_ecms_cut", ("_ecms_min", "_ecms_max")),
            ("_npoint_cut", ("_np",)),
            ("_mass_cut", ("_mass_min", "_mass_max")),
        )
        for flag, names in checks:
            if getattr(self, flag) != getattr(other, flag):
                return False
            if getattr(self, flag):
                # Inactive cuts may carry stale bounds; compare only the
                # bounds of cuts that are actually applied.
                for name in names:
                    if getattr(self, name) != getattr(other, name):
                        return False
        return True

    # ------------------------------------------------------------------
    # I/O (mirrors C++ ostream/istream operators)
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        lines = []
        if self._xf_cut:
            lines.append(f"xf {self._xf_min} {self._xf_max}")
        if self._pt_cut:
            lines.append(f"pt {self._pt_min} {self._pt_max}")
        if self._id_cut:
            lines.append(f"id {self._id}")
        if self._px_cut:
            lines.append(f"px {self._px_min} {self._px_max}")
        if self._py_cut:
            lines.append(f"py {self._py_min} {self._py_max}")
        if self._pz_cut:
            lines.append(f"pz {self._pz_min} {self._pz_max}")
        if self._e_cut:
            lines.append(f"e {self._e_min} {self._e_max}")
        if self._pzcms_cut:
            lines.append(f"pzcms {self._pzcms_min} {self._pzcms_max}")
        if self._ecms_cut:
            lines.append(f"ecms {self._ecms_min} {self._ecms_max}")
        if self._npoint_cut:
            lines.append(f"np {self._np}")
        if self._mass_cut:
            lines.append(f"mass {self._mass_min} {self._mass_max}")
        lines.append("end")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_string(cls, s: str) -> "ParticleCut":
        """Parse a cut from its string representation."""
        cut = cls()
        for line in s.splitlines():
            parts = line.strip().split()
            if not parts or parts[0] == "end":
                break
            key = parts[0]
            if key == "id":
                cut.set_id_cut(int(parts[1]))
            elif key == "xf":
                cut.set_xf_cut(float(parts[1]), float(parts[2]))
            elif key == "pt":
                cut.set_pt_cut(float(parts[1]), float(parts[2]))
            elif key == "px":
                cut.set_px_cut(float(parts[1]), float(parts[2]))
            elif key == "py":
                cut.set_py_cut(float(parts[1]), float(parts[2]))
            elif key == "pz":
                cut.set_pz_cut(float(parts[1]), float(parts[2]))
            elif key == "e":
                cut.set_e_cut(float(parts[1]), float(parts[2]))
            elif key == "pzcms":
                cut.set_pzcms_cut(float(parts[1]), float(parts[2]))
            elif key == "ecms":
                cut.set_ecms_cut(float(parts[1]), float(parts[2]))
            elif key == "np":
                cut.set_npoint_cut(int(parts[1]))
            elif key == "mass":
                cut.set_mass_cut(float(parts[1]), float(parts[2]))
        return cut

    def __repr__(self) -> str:
        return f"ParticleCut(n_should={self._n_should})"


# ---------------------------------------------------------------------------
# EventCut
# ---------------------------------------------------------------------------

class EventCut:
    """Cut on event-level quantities."""

    def __init__(self) -> None:
        self._mult_cut = False
        self._mult_min = 0
        self._mult_max = 100_000
        self._mult_particle: Optional[ParticleCut] = None
        self._grey_cut = False
        self._grey_min = 0
        self._grey_max = 100_000

    def set_mult_cut(self, min_mult: int, max_mult: int) -> None:
        self._mult_cut = True
        self._mult_min = int(min_mult)
        self._mult_max = int(max_mult)

    def set_mult_particle(self, cut: Optional[ParticleCut]) -> None:
        """Set the particle cut used to count tracks for the multiplicity cut."""
        self._mult_particle = cut

    def set_grey_cut(self, min_grey: int, max_grey: int) -> None:
        self._grey_cut = True
        self._grey_min = int(min_grey)
        self._grey_max = int(max_grey)

    def check_event(self, event: "Event") -> bool:
        """Return ``True`` if *event* passes all active event-level cuts.

        A multiplicity window with no particle filter counts every track,
        matching the C++ ``TEventCut`` default.
        """
        if self._mult_cut:
            particles = event.particles()
            if self._mult_particle is not None:
                count = sum(
                    1 for p in particles
                    if self._mult_particle.check_particle(p)
                )
            else:
                count = len(particles)
            if not (self._mult_min <= count <= self._mult_max):
                return False
        if self._grey_cut:
            if not (self._grey_min <= event.get_n_grey() <= self._grey_max):
                return False
        return True

    def __repr__(self) -> str:
        return f"EventCut(mult={self._mult_cut}, grey={self._grey_cut})"

    def __str__(self) -> str:
        lines = []
        if self._mult_cut:
            lines.append(f"mult {self._mult_min} {self._mult_max}")
        if self._grey_cut:
            lines.append(f"grey {self._grey_min} {self._grey_max}")
        lines.append("end")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# PairCut
# ---------------------------------------------------------------------------

class PairCut:
    """Cut on a pair of particles."""

    def __init__(self) -> None:
        self._opening_angle_cut = False
        self._cos_theta_cut = False
        self._inv_mass_cut = False
        self._opening_angle_min = 0.0
        self._opening_angle_max = math.pi
        self._cos_theta_min = -1.0
        self._cos_theta_max = 1.0
        self._inv_mass_min = 0.0
        self._inv_mass_max = 1e9
        self._n_should = 0

    def set_opening_angle_cut(self, angle_min: float, angle_max: float) -> None:
        if not self._opening_angle_cut:
            self._n_should += 1
        self._opening_angle_cut = True
        self._opening_angle_min = float(angle_min)
        self._opening_angle_max = float(angle_max)

    def set_cos_theta_cut(self, cos_min: float, cos_max: float) -> None:
        if not self._cos_theta_cut:
            self._n_should += 1
        self._cos_theta_cut = True
        self._cos_theta_min = float(cos_min)
        self._cos_theta_max = float(cos_max)

    def set_inv_mass_cut(self, m_min: float, m_max: float) -> None:
        if not self._inv_mass_cut:
            self._n_should += 1
        self._inv_mass_cut = True
        self._inv_mass_min = float(m_min)
        self._inv_mass_max = float(m_max)

    def check_pair(self, a: "Particle", b: "Particle") -> bool:
        """Return ``True`` if the pair (a, b) passes all active cuts."""
        if self._n_should == 0:
            return True
        if self._opening_angle_cut or self._cos_theta_cut:
            pa = a.get_p()
            pb = b.get_p()
            dot = (
                a.get_px() * b.get_px()
                + a.get_py() * b.get_py()
                + a.get_pz() * b.get_pz()
            )
            if pa <= 0.0 or pb <= 0.0:
                return False
            cos_angle = max(-1.0, min(1.0, dot / (pa * pb)))
            if self._opening_angle_cut:
                angle = math.acos(cos_angle)
                if not (self._opening_angle_min <= angle <= self._opening_angle_max):
                    return False
            if self._cos_theta_cut:
                if not (self._cos_theta_min <= cos_angle <= self._cos_theta_max):
                    return False
        if self._inv_mass_cut:
            m = a.get_inv_mass(b)
            if not (self._inv_mass_min <= m <= self._inv_mass_max):
                return False
        return True

    def __repr__(self) -> str:
        return f"PairCut(n_should={self._n_should})"

    def __str__(self) -> str:
        lines = []
        if self._opening_angle_cut:
            lines.append(
                f"opening_angle {self._opening_angle_min} {self._opening_angle_max}"
            )
        if self._cos_theta_cut:
            lines.append(f"cos_theta {self._cos_theta_min} {self._cos_theta_max}")
        if self._inv_mass_cut:
            lines.append(f"inv_mass {self._inv_mass_min} {self._inv_mass_max}")
        lines.append("end")
        return "\n".join(lines) + "\n"
