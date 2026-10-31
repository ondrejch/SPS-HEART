"""Event-mixing ring buffer — port of ``MMixEvent`` + ``MMixBuffer``.

:class:`MixEvent`
    Stores the A- and B-particle lists from one event.
:class:`MixBuffer`
    Fixed-depth ring buffer of :class:`MixEvent` objects used for
    combinatorial background estimation.
"""

from __future__ import annotations
from collections import deque
from typing import Optional, List, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .particle import Particle
    from .event import Event
    from .cuts import ParticleCut


class MixEvent:
    """One entry in the mixing buffer: lists of A- and B-particles.

    Particles are *copies* so the buffer owns them independently of the
    original event.
    """

    def __init__(self) -> None:
        self._a_particles: List["Particle"] = []
        self._b_particles: List["Particle"] = []
        self._a_iter: int = 0
        self._b_iter: int = 0

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def add_a(self, particle: "Particle") -> None:
        self._a_particles.append(particle)

    def add_b(self, particle: "Particle") -> None:
        self._b_particles.append(particle)

    def has_a(self) -> bool:
        return len(self._a_particles) > 0

    def has_b(self) -> bool:
        return len(self._b_particles) > 0

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def set_to_first_a(self) -> None:
        self._a_iter = 0

    def set_to_first_b(self) -> None:
        self._b_iter = 0

    def get_next_a(self) -> Optional["Particle"]:
        if self._a_iter < len(self._a_particles):
            p = self._a_particles[self._a_iter]
            self._a_iter += 1
            return p
        return None

    def get_next_b(self) -> Optional["Particle"]:
        if self._b_iter < len(self._b_particles):
            p = self._b_particles[self._b_iter]
            self._b_iter += 1
            return p
        return None

    def a_particles(self) -> List["Particle"]:
        return self._a_particles

    def b_particles(self) -> List["Particle"]:
        return self._b_particles

    def __repr__(self) -> str:
        return f"MixEvent(nA={len(self._a_particles)}, nB={len(self._b_particles)})"


class MixBuffer:
    """Ring buffer of :class:`MixEvent` objects for event mixing.

    The buffer holds at most *max_depth* events.  The most recently
    added event is the "current" event; the rest are used for mixing.

    Parameters
    ----------
    max_depth : int
        Maximum buffer depth (including the current event).
    a_filter : ParticleCut, optional
        Selects A-particles from each event.
    b_filter : ParticleCut, optional
        Selects B-particles from each event.
    """

    def __init__(
        self,
        max_depth: int = 11,
        a_filter: Optional["ParticleCut"] = None,
        b_filter: Optional["ParticleCut"] = None,
    ) -> None:
        from .cuts import ParticleCut  # local import to avoid circular
        self._max_depth = int(max_depth)
        self._buffer: deque[MixEvent] = deque(maxlen=max_depth)
        self._a_filter: "ParticleCut" = a_filter if a_filter is not None else ParticleCut()
        self._b_filter: "ParticleCut" = b_filter if b_filter is not None else ParticleCut()
        self._weight: float = 1.0
        # Mixing iterator state
        self._mix_index: int = 1      # index into _buffer for current mixing event
        self._mix_b_index: int = 0    # b-particle index within the current mix event

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def get_a_filter(self) -> "ParticleCut":
        return self._a_filter

    def get_b_filter(self) -> "ParticleCut":
        return self._b_filter

    def set_a_filter(self, flt: "ParticleCut") -> None:
        self._a_filter = flt

    def set_b_filter(self, flt: "ParticleCut") -> None:
        self._b_filter = flt

    def set_filters(
        self,
        a_filter: "ParticleCut",
        b_filter: "ParticleCut",
    ) -> None:
        self._a_filter = a_filter
        self._b_filter = b_filter

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        """Return ``True`` when the buffer has reached its maximum depth."""
        return len(self._buffer) >= self._max_depth

    def get_depth(self) -> int:
        return self._max_depth

    def get_weight(self) -> float:
        return self._weight

    # ------------------------------------------------------------------
    # Adding events
    # ------------------------------------------------------------------
    def add_mix_event(self, mev: MixEvent) -> None:
        """Push *mev* to the front of the buffer (oldest event is evicted)."""
        self._buffer.appendleft(mev)
        # reset mixing iterator to start from the second event
        self._mix_index = 1
        self._mix_b_index = 0

    def process_event(self, event: "Event") -> bool:
        """Filter particles from *event*, build a :class:`MixEvent`, and store it.

        Every accepted source track receives a per-event source identifier
        (0, 1, 2, ...) that is stamped on all of its copies, so the A- and
        B-list copies of one physical track share the identity. Consumers
        of the buffered lists can use these identifiers to recognize the
        copies of one track; :class:`~na49_mixing.spectra.Spectra` forms
        its real pairs from the original event instead, where object
        identity serves the same purpose. Cross-event mixed pairs never
        compare identifiers, because identifiers are only unique within
        one event.

        Returns ``True`` if the event contributed at least one A- and one
        B-particle (and was therefore added to the buffer).
        """
        from .particle import Particle as P  # local to avoid circular
        mev = MixEvent()
        source_id = 0
        for p in event.particles():
            passed_a = self._a_filter.check_particle(p)
            passed_b = self._b_filter.check_particle(p)
            # A particle passing BOTH filters enters both lists (independent
            # copies), so identical-particle / overlapping-filter mixing is
            # supported. Both copies of one source track carry the same
            # source identifier so the real-pair stage can tell them apart
            # from genuinely distinct tracks.
            # Normalization note: mixed pairs are scaled by the
            # fixed factor 1/(depth-1), which assumes comparable
            # multiplicities and weights across events in the buffer.
            if passed_a or passed_b:
                if passed_a:
                    a_copy = P.copy(p)
                    a_copy.set_source_id(source_id)
                    mev.add_a(a_copy)
                if passed_b:
                    b_copy = P.copy(p)
                    b_copy.set_source_id(source_id)
                    mev.add_b(b_copy)
                source_id += 1

        if mev.has_a() and mev.has_b():
            self.add_mix_event(mev)
            self._weight = event.weight
            return True
        return False

    # ------------------------------------------------------------------
    # Particle access — current event (index 0 in buffer)
    # ------------------------------------------------------------------
    def current_a_particles(self) -> List["Particle"]:
        """Return the A-particle copies of the current event.

        The returned list is the live buffer contents (position 0); treat
        it as read-only. Empty when no event has been processed yet.
        """
        if not self._buffer:
            return []
        return self._buffer[0].a_particles()

    def current_b_particles(self) -> List["Particle"]:
        """Return the B-particle copies of the current event."""
        if not self._buffer:
            return []
        return self._buffer[0].b_particles()

    def get_next_a(self) -> Optional["Particle"]:
        """Return the next A-particle from the *current* event."""
        if not self._buffer:
            return None
        cur = self._buffer[0]
        p = cur.get_next_a()
        if p is None:
            cur.set_to_first_a()
        return p

    def get_next_b(self) -> Optional["Particle"]:
        """Return the next B-particle from the *current* event."""
        if not self._buffer:
            return None
        cur = self._buffer[0]
        p = cur.get_next_b()
        if p is None:
            cur.set_to_first_b()
        return p

    # ------------------------------------------------------------------
    # Particle access — mixing events (indices 1…depth-1)
    # ------------------------------------------------------------------
    def get_next_b_mix(self) -> Optional["Particle"]:
        """Return the next B-particle from the mixing events.

        Iterates through all B-particles in events at buffer positions
        1 … max_depth-1 in sequence.  Returns ``None`` after exhausting
        all mixing events and resets the iterator.
        """
        buf_len = len(self._buffer)
        if buf_len <= 1:
            return None

        while self._mix_index < buf_len:
            mev = self._buffer[self._mix_index]
            p = mev.get_next_b()
            if p is not None:
                return p
            # exhausted this mixing event — advance to next
            mev.set_to_first_b()
            self._mix_index += 1
            self._mix_b_index = 0

        # exhausted all mixing events — reset and signal end
        self._mix_index = 1
        self._mix_b_index = 0
        for i in range(1, buf_len):
            self._buffer[i].set_to_first_b()
        return None

    def __repr__(self) -> str:
        return (
            f"MixBuffer(max_depth={self._max_depth}, "
            f"current_depth={len(self._buffer)}, "
            f"ready={self.is_ready()})"
        )
