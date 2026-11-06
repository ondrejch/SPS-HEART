# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Invariant-mass spectra with event mixing — port of ``TSpectra``.

:class:`Spectra`
    Maintains three histograms (real pairs, mixed pairs, subtracted
    difference) and fills them event-by-event. Real pairs ignore the
    NPoint cut while mixed pairs apply it; the class docstring states
    the exact counting conventions.
"""

from __future__ import annotations
import copy
from typing import Optional, TYPE_CHECKING
import numpy as np

from .histogram import Histogram1D
from .mixing import MixBuffer
from .cuts import ParticleCut, EventCut, PairCut

if TYPE_CHECKING:
    from .event import Event
    from .particle import Particle


class Spectra:
    """Invariant-mass spectra with combinatorial background subtraction.

    Real-pair counting conventions (audit finding 3, Option A):

    - Real pairs are formed directly from the original event through
      copies of the A/B filters with the NPoint cut disabled. The NPoint
      cut therefore applies to mixed pairs only: an event whose tracks
      pass every cut except NPoint still contributes real entries.
    - A track that passes both selections is never paired with itself.
      Within one event distinct physical tracks are distinct objects,
      so object identity plays the role of a source-track identifier.
    - When both members of an unordered pair pass both selections, the
      (i, j) / (j, i) orientations would double-count it; exactly one
      orientation is filled. Pairs where only one orientation exists
      (one member exclusive to one selection) are counted once through
      that single orientation.
    - Cross-event mixed pairs come from the mixing buffer, whose copies
      are selected by the original NPoint-enabled filters; they are
      never filtered by identity.
    - :meth:`get_entries` counts events accepted into the real-pair
      stage: they passed the event cut and produced at least one A and
      one B candidate under the NPoint-free filters. The count is
      independent of the mixing buffer, which may reject the same event
      when none of its tracks passes the NPoint-enabled buffer filters.

    Parameters
    ----------
    name : str
        Unique name used to label histograms.
    is_data : bool
        ``True`` for real data, ``False`` for Monte-Carlo.
    nbins : int
        Number of histogram bins.
    m_low, m_high : float
        Mass range (GeV/c²).
    depth : int
        Mixing-buffer depth (default 11 = 10 mixing events + current).
    """

    def __init__(
        self,
        name: str = "default",
        is_data: bool = True,
        nbins: int = 100,
        m_low: float = 1.0,
        m_high: float = 2.0,
        depth: int = 11,
    ) -> None:
        self.name = str(name)
        self._is_data = bool(is_data)
        self._filled = False
        self._normalization: int = 0
        self._do_mix = True
        # True when the real-pair A and B selections are structurally
        # equivalent (see ParticleCut.is_equivalent); snapshotted once on
        # the first fill, because set_filters() is refused afterwards.
        self._ab_equivalent: bool = False

        self._buffer = MixBuffer(depth)
        self._ev_filter = EventCut()
        self._pair_cut: Optional[PairCut] = None

        # Histograms (set lazily)
        self._mass_h: Optional[Histogram1D] = None
        self._mix_h: Optional[Histogram1D] = None
        self._dif_h: Optional[Histogram1D] = None

        self._set_histo(nbins, m_low, m_high)

        # Copies of the buffer filters with the NPoint cut disabled; they
        # select the real-pair candidates from the original event in
        # fill(). The mixing buffer keeps the original NPoint-enabled
        # filters, so NPoint applies to mixed pairs only.
        self._a_filter: Optional[ParticleCut] = None
        self._b_filter: Optional[ParticleCut] = None

    # ------------------------------------------------------------------
    # Histogram setup
    # ------------------------------------------------------------------
    def _set_histo(self, nbins: int, m_low: float, m_high: float) -> None:
        suffix = "_data_" if self._is_data else "_mc_"
        self._mass_h = Histogram1D(nbins, m_low, m_high,
                                    name=self.name + suffix + "mass")
        self._mix_h = Histogram1D(nbins, m_low, m_high,
                                   name=self.name + suffix + "mix")
        self._dif_h = Histogram1D(nbins, m_low, m_high,
                                   name=self.name + suffix + "dif")

    def set_histo(self, nbins: int, m_low: float, m_high: float) -> None:
        if self._filled:
            return
        self._set_histo(nbins, m_low, m_high)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_filters(
        self,
        a_filter: ParticleCut,
        b_filter: ParticleCut,
    ) -> None:
        if self._filled:
            return
        self._buffer.set_filters(a_filter, b_filter)

    def set_depth(self, depth: int) -> None:
        if self._filled:
            return
        if depth < 2:
            return
        old_a = self._buffer.get_a_filter()
        old_b = self._buffer.get_b_filter()
        self._buffer = MixBuffer(depth, old_a, old_b)

    def set_pair_cut(self, cut: PairCut) -> None:
        if self._filled:
            return
        self._pair_cut = cut

    def set_ev_cut(self, min_mult: int, max_mult: int, cut: Optional[ParticleCut] = None) -> None:
        self._ev_filter.set_mult_cut(min_mult, max_mult)
        if cut is not None:
            self._ev_filter.set_mult_particle(cut)

    def set_mixing(self, do_mix: bool) -> None:
        self._do_mix = bool(do_mix)

    def get_mixing(self) -> bool:
        return self._do_mix

    def is_data(self) -> bool:
        return self._is_data

    # ------------------------------------------------------------------
    # Histogram accessors
    # ------------------------------------------------------------------
    def get_mass_hist(self) -> Optional[Histogram1D]:
        return self._mass_h

    def get_mix_hist(self) -> Optional[Histogram1D]:
        return self._mix_h

    def get_dif_hist(self) -> Histogram1D:
        self.subtract()
        return self._dif_h  # type: ignore[return-value]

    def get_entries(self) -> int:
        return self._normalization

    def get_n_bins(self) -> int:
        return self._mass_h.nbins if self._mass_h else 0

    def get_low_edge(self) -> float:
        return self._mass_h.xlow if self._mass_h else 0.0

    def get_upper_edge(self) -> float:
        return self._mass_h.xhigh if self._mass_h else 0.0

    # ------------------------------------------------------------------
    # Filling
    # ------------------------------------------------------------------
    def fill(self, event: "Event") -> bool:
        """Process one event.

        Real pairs are formed from the original event through the
        NPoint-free filter copies; mixed pairs come from the mixing
        buffer, whose copies are selected by the original NPoint-enabled
        filters. Returns ``True`` if the event was added to the buffer.
        """
        assert self._mass_h is not None
        assert self._mix_h is not None

        # Initialise copied filters on first call
        if not self._filled:
            self._a_filter = copy.deepcopy(self._buffer.get_a_filter())
            self._b_filter = copy.deepcopy(self._buffer.get_b_filter())
            self._a_filter.off_npoint_cut()
            self._b_filter.off_npoint_cut()
            # Structural A/B-equivalence snapshot for real-pair de-
            # duplication. The NPoint-free copies select the paired
            # particles, so equivalence is judged on those.
            self._ab_equivalent = self._a_filter.is_equivalent(
                self._b_filter
            )
            self._filled = True
        # The first fill always initialises the NPoint-free copies above;
        # the assertions are the static narrowing witnesses.
        assert self._a_filter is not None
        assert self._b_filter is not None

        if not self._ev_filter.check_event(event):
            return False

        # ---- Real-pair stage -------------------------------------------
        # Select real-pair candidates directly from the ORIGINAL event
        # with the NPoint-free copies of the A/B filters (documented
        # Option-A convention: real pairs ignore NPoint).
        a_real: list = []
        b_real: list = []
        for p in event.particles():
            passed_a = self._a_filter.check_particle(p)
            passed_b = self._b_filter.check_particle(p)
            if passed_a:
                a_real.append(p)
            if passed_b:
                b_real.append(p)

        if not a_real or not b_real:
            return False

        # Normalization convention: get_entries() counts events accepted
        # into the real-pair stage, i.e. events that passed the event cut
        # and yield at least one A and one B candidate under the
        # NPoint-free filters. The count is deliberately independent of
        # the mixing-buffer veto below: an event whose tracks all fail
        # the NPoint-enabled buffer filters still contributes to the
        # real spectrum, so it must also count here.
        self._normalization += 1

        put = self._buffer.process_event(event)

        if self._buffer.is_ready():
            # Fill real pairs (same-event pairs of the current event)
            # from the candidate lists above. Within one event distinct
            # physical tracks are distinct objects, so object identity
            # identifies source tracks:
            #   a_pos -- first A-list position of each source track,
            #   in_b  -- source tracks that also appear in the B list.
            a_pos: dict[int, int] = {}
            for idx, p in enumerate(a_real):
                a_pos.setdefault(id(p), idx)
            in_b = {id(p) for p in b_real}
            for apart in a_real:
                key_a = id(apart)
                pos_a = a_pos[key_a]
                for bpart in b_real:
                    key_b = id(bpart)
                    # Never pair a track with itself: one physical track
                    # enters both lists when it passes both selections,
                    # and its pair mass would sit at twice its mass.
                    if key_a == key_b:
                        continue
                    # Mirrored duplicate of an unordered pair with both
                    # members in both selections; skip all but the first
                    # orientation.
                    mirrored = (
                        True if self._ab_equivalent
                        else (key_b in a_pos and key_a in in_b)
                    )
                    if mirrored and pos_a >= a_pos[key_b]:
                        continue
                    if self._pair_cut is None or self._pair_cut.check_pair(apart, bpart):
                        self._mass_h.fill(apart.get_inv_mass(bpart), event.weight)

            # Fill mixed pairs. Only an event freshly added to the buffer
            # may advance the mixing iteration: after a rejection,
            # position 0 still holds the previous event and its mixed
            # pairs would be replayed.
            if put and self._do_mix:
                for apart in self._buffer.current_a_particles():
                    while True:
                        bpart = self._buffer.get_next_b_mix()
                        if bpart is None:
                            break
                        if self._pair_cut is None or self._pair_cut.check_pair(apart, bpart):
                            self._mix_h.fill(apart.get_inv_mass(bpart), event.weight)

        return put

    # ------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------
    def _subtract(self) -> None:
        """dif = mass - mix / (depth - 1)."""
        assert self._mass_h is not None
        assert self._mix_h is not None
        assert self._dif_h is not None
        depth = self._buffer.get_depth()
        scale = 1.0 / max(depth - 1, 1)
        self._dif_h.add(self._mass_h)
        self._dif_h.add(self._mix_h, -scale)

    def subtract(self) -> None:
        """Public wrapper for :meth:`_subtract`."""
        # Reset dif_h first so it can be called multiple times safely
        assert self._dif_h is not None
        self._dif_h._contents[:] = 0.0
        self._subtract()

    # ------------------------------------------------------------------
    # Fit the normalisation of mix_h to mass_h in a given range
    # ------------------------------------------------------------------
    def fit(self, mass_min: float, mass_max: float) -> float:
        """Return the optimal scale factor α such that mass ≈ α·mix."""
        assert self._mass_h is not None
        assert self._mix_h is not None
        i_min = self._mass_h.find_bin(mass_min)
        i_max = self._mass_h.find_bin(mass_max)
        up = 0.0
        down = 0.0
        for i in range(i_min, i_max + 1):
            a = self._mass_h.get_bin_content(i)
            b = self._mix_h.get_bin_content(i)
            # approximate error as sqrt(a)
            err = a ** 0.5 if a > 0.0 else 0.0
            if err != 0.0:
                up += a * b / (err * err)
                down += b * b / (err * err)
        return up / down if down != 0.0 else 0.0

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def add(self, other: "Spectra") -> None:
        if self._is_data != other.is_data():
            return
        self.subtract()
        other.subtract()
        if self._mass_h and other._mass_h:
            self._mass_h.add(other._mass_h)
        if self._mix_h and other._mix_h:
            self._mix_h.add(other._mix_h)
        if self._dif_h and other._dif_h:
            self._dif_h.add(other._dif_h)
        self._normalization += other.get_entries()

    def __iadd__(self, other: "Spectra") -> "Spectra":
        self.add(other)
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Spectra(name={self.name!r}, "
            f"is_data={self._is_data}, "
            f"entries={self._normalization})"
        )
