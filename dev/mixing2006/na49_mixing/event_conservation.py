"""Per-vertex conservation audit for staged and legacy events.

For every decay vertex in a generated event this module verifies the
per-vertex conservation laws that plan section 6 (Gate 0) requires:

    P^mu - sum_i p_i^mu = 0        (relative_tolerance on the largest
                                    component)
    B(P) - sum_i B(p_i) = 0        (exact integer)
    Q(P) - sum_i Q(p_i) = 0        (exact integer)
    S(P) - sum_i S(p_i) = 0        (exact integer)
    m_i^2 = p_i^2                  (relative_tolerance * max(m_i, 1))

plus the ancestry conditions: source/parent chains are acyclic and
complete, and every daughter resolves to the vertex it was produced
at.  The audit is read-only: it never assigns to an inspected event,
never calls ``Event.decay()`` or any other mutating method, and never
changes discrete probabilities or four-vectors.  A failing verdict is
a Gate 0 signal for the run cards that consume this module; this
module performs no repair.

Two event representations are audited:

- staged ``ExcitationEvent`` (``na49_mixing.source_event``): each
  ``SourceRecord`` with at least one ``FinalParticle`` daughter is one
  vertex; the daughters of a source are the final particles whose
  ``source_id`` matches the source's ``source_id``.  Note on field
  names: the source records live on the attribute ``event.sources``
  (the plan text writes ``event.souces``), and ``FinalParticle.mass``
  is a derived property (``p4.mass()``), not a stored configured mass.
  The configured daughter mass for the mass-shell check is therefore
  the particle-database mass of the daughter's PDG id, exactly the
  reference ``ExcitationEvent._validate_pdg_on_shell`` uses; the
  reconstructed mass is recomputed by this module from the stored
  four-vector components.  ``event.spectators`` (the p+A spectator
  nucleons) are not source daughters; the same particle-database
  on-shell check applies to each of them, and the ancestry walk treats
  a spectator as a terminal record of the nuclear remnant: the
  builder's no-source sentinel (negative ``source_id``,
  ``parent_id is None``) is complete, while a spectator that claims a
  source or a parent that does not resolve in the event is an orphan.

- legacy ``Event`` (``na49_mixing.event``): a vertex is one parent
  ``Particle`` together with the particles it decays into.  The decay
  engine stamps ``Particle.parent`` at decay time and removes the
  parent from the live particle list, so the vertices are
  reconstructed from the full parent-chain graph.  Starting from the
  terminal tracks, the audit collects every ancestor through
  ``.parent`` and groups all collected particles (terminals plus
  recovered intermediate daughters) by their immediate parent.  A
  two-body cascade ``N* -> Delta + pi -> N + pi + pi`` is therefore
  audited as the two complete vertices ``N* -> [Delta, pi]`` and
  ``Delta -> [N, pi]``.  ``Particle.daughter`` and ``next_daughter``
  are never assigned by the decay engine, so the parent links are the
  only ancestry data used.

  Mass-shell provenance for the legacy path: the configured daughter
  mass is the ``Particle`` mass recorded at decay time
  (``get_mass()``; for a Breit-Wigner daughter the drawn mass, so the
  audit is valid against the frozen ``legacy_v1`` and the re-pinned
  decay engine alike).  ``Particle.get_e()`` is ``sqrt(m^2 + p^2)``
  computed from that same stored mass, so the legacy check compares
  the recomputed invariant mass against the mass that entered
  ``get_e()``; it is the ``Particle`` identity
  ``get_e() = sqrt(m^2 + p^2)``, not an independent four-vector
  on-shell test.  The staged audit keeps the independent
  particle-database reference, and the legacy audit deliberately stays
  on the decay-time mass instead of the PDG pole mass, because a
  Breit-Wigner daughter is supposed to sit on the drawn mass.  The
  ancestry walk additionally requires every terminal track to carry
  ``_gen_depth >= 0`` and a parent chain that terminates at a
  generated primary (``_is_primary``) or at a weak-feed source
  (``_weak_feed``).

Strangeness at declared weak vertices: the legacy decay engine
represents weak feed-down (``WEAK_DECAY_PARENTS``); weak decays change
strangeness by design, so at a vertex whose parent PDG id is in
``WEAK_DECAY_PARENTS`` the baryon-number and charge closures remain
exact checks while the strangeness closure is exempt.  Staged events
contain no weak vertices, so their strangeness closure is exact at
every vertex.

Tolerance provenance: the default ``relative_tolerance = 1e-10`` is
the ``conservation_tolerance`` default of ``PPExcitationConfig`` and
``PAExcitationConfig``, and the deviation scaling
``max(configured_mass, 1.0)`` mirrors
``ExcitationEvent._validate_pdg_on_shell``.  No new tolerance
convention is introduced.

Hermetic: pure reads on in-memory events; no external tables; no
event generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .decays import WEAK_DECAY_PARENTS
from .event import Event
from .excitation_kinematics import FourVector
from .particle import Particle
from .pdg_database import get_mass, get_particle
from .quantum_numbers import QuantumNumbers, quantum_numbers
from .source_event import ExcitationEvent

__all__ = [
    "VertexAuditReport",
    "audit_staged_event",
    "audit_legacy_event",
    "vertex_audit_summary",
]


@dataclass(frozen=True)
class VertexAuditReport:
    """Read-only outcome of one per-vertex conservation audit.

    ``four_momentum``, ``bqs``, ``mass_shell``, and ``ancestry`` are
    plain dictionaries so the report serializes without conversion:

    - ``four_momentum``: ``{"n_checked", "n_failed",
      "max_relative_residual"}``;
    - ``bqs``: ``{"n_checked", "n_failed", "residuals"}`` with
      ``residuals`` the ``(B, Q, S)`` tuples of failed vertices;
    - ``mass_shell``: ``{"n_checked", "n_failed",
      "max_relative_deviation"}``;
    - ``ancestry``: ``{"acyclic", "complete", "orphan_count",
      "cycle_count"}``.

    ``verdict`` is ``"pass"`` when all four checks hold, ``"fail"``
    otherwise.
    """

    event_index: int
    n_vertices: int
    four_momentum: dict = field(default_factory=dict)
    bqs: dict = field(default_factory=dict)
    mass_shell: dict = field(default_factory=dict)
    ancestry: dict = field(default_factory=dict)
    verdict: str = "pass"


def _mass_from_components(e: float, px: float, py: float, pz: float) -> float:
    """Invariant mass recomputed from the stored four-vector components."""
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def _verdict(
    four_momentum: dict, bqs: dict, mass_shell: dict, ancestry: dict
) -> str:
    if (
        four_momentum["n_failed"] == 0
        and bqs["n_failed"] == 0
        and mass_shell["n_failed"] == 0
        and ancestry["acyclic"]
        and ancestry["complete"]
    ):
        return "pass"
    return "fail"


def _staged_ancestry(event: ExcitationEvent) -> dict:
    """Source-chain acyclicity/completeness and orphaned final particles.

    A final particle is an orphan when its ``source_id`` or ``parent_id``
    does not resolve to a source record present in the event; a source
    whose ``parent_source_id`` leaves the event makes the ancestry
    incomplete as well.  Spectators are terminal records of the nuclear
    remnant rather than source daughters: a spectator is an orphan only
    when it claims a source or a parent that does not resolve, and the
    builder's no-source sentinel (negative ``source_id``,
    ``parent_id is None``) is complete.  ``cycle_count`` counts the
    distinct sources that lie on a parent cycle (a self-parent counts
    once).
    """
    source_by_id = {source.source_id: source for source in event.sources}
    orphan_count = 0
    for particle in event.final_particles:
        if particle.source_id not in source_by_id:
            orphan_count += 1
            continue
        if particle.parent_id is None or particle.parent_id not in source_by_id:
            orphan_count += 1
    for particle in event.spectators:
        if particle.parent_id is not None and particle.parent_id not in source_by_id:
            orphan_count += 1
            continue
        if particle.source_id >= 0 and particle.source_id not in source_by_id:
            orphan_count += 1

    # Parent-source chains: classic cycle detection over the (tiny)
    # source graph.  A chain is complete when it terminates at a root
    # with ``parent_source_id is None`` without leaving the event.
    clean: set[int] = set()
    cyclic: set[int] = set()
    for source in event.sources:
        if source.source_id in clean or source.source_id in cyclic:
            continue
        if source.parent_source_id == source.source_id:
            cyclic.add(source.source_id)
            continue
        walk: list[int] = [source.source_id]
        seen: set[int] = {source.source_id}
        node_id = source.parent_source_id
        while True:
            if node_id is None:
                clean.update(walk)
                break
            if node_id in seen:
                cyclic.update(walk[walk.index(node_id):])
                break
            node = source_by_id.get(node_id)
            if node is None:
                # The chain leaves the event: no root source terminates
                # it inside the event, so the ancestry is incomplete.
                orphan_count += 1
                break
            if node.source_id in clean:
                clean.update(walk)
                break
            if node.source_id in cyclic:
                break
            seen.add(node.source_id)
            walk.append(node.source_id)
            node_id = node.parent_source_id

    ancestry = {
        "acyclic": not cyclic,
        "complete": orphan_count == 0,
        "orphan_count": orphan_count,
        "cycle_count": len(cyclic),
    }
    return ancestry


def audit_staged_event(
    event: ExcitationEvent,
    relative_tolerance: float = 1e-10,
    event_index: int = 0,
) -> VertexAuditReport:
    """Audit one staged ``ExcitationEvent`` without modifying it.

    One vertex per ``SourceRecord`` that carries daughters; the checks
    are the per-vertex four-momentum and ``(B, Q, S)`` closures, the
    on-shell status of every daughter and every spectator against the
    particle-database mass (``_validate_pdg_on_shell`` semantics), and
    the source ancestry conditions.
    """
    source_by_id = {source.source_id: source for source in event.sources}
    daughters_by_source: dict[int, list] = {}
    for particle in event.final_particles:
        daughters_by_source.setdefault(particle.source_id, []).append(particle)

    four_momentum = {"n_checked": 0, "n_failed": 0, "max_relative_residual": 0.0}
    bqs = {"n_checked": 0, "n_failed": 0, "residuals": []}
    mass_shell = {"n_checked": 0, "n_failed": 0, "max_relative_deviation": 0.0}

    n_vertices = 0
    for source in event.sources:
        daughters = daughters_by_source.get(source.source_id)
        if not daughters:
            continue
        n_vertices += 1

        total = FourVector(0.0, 0.0, 0.0, 0.0)
        for daughter in daughters:
            total = total + daughter.p4
        residual = total - source.p4
        scale = max(abs(source.p4.e), 1.0)
        relative_residual = (
            max(
                abs(residual.e),
                abs(residual.px),
                abs(residual.py),
                abs(residual.pz),
            )
            / scale
        )
        four_momentum["n_checked"] += 1
        four_momentum["max_relative_residual"] = max(
            float(four_momentum["max_relative_residual"]), relative_residual
        )
        if relative_residual > relative_tolerance:
            four_momentum["n_failed"] += 1

        parent_qn = source.quantum_numbers
        daughter_qn = QuantumNumbers()
        for daughter in daughters:
            daughter_qn = daughter_qn + quantum_numbers(daughter.pdg_id)
        bqs_residual = (
            parent_qn.B - daughter_qn.B,
            parent_qn.Q - daughter_qn.Q,
            parent_qn.S - daughter_qn.S,
        )
        bqs["n_checked"] += 1
        if bqs_residual != (0, 0, 0):
            bqs["n_failed"] += 1
            bqs["residuals"].append(bqs_residual)

        for daughter in daughters:
            if get_particle(daughter.pdg_id) is None:
                continue
            configured = get_mass(daughter.pdg_id)
            reconstructed = _mass_from_components(
                daughter.p4.e, daughter.p4.px, daughter.p4.py, daughter.p4.pz
            )
            deviation = abs(reconstructed - configured) / max(configured, 1.0)
            mass_shell["n_checked"] += 1
            mass_shell["max_relative_deviation"] = max(
                float(mass_shell["max_relative_deviation"]), deviation
            )
            if deviation > relative_tolerance:
                mass_shell["n_failed"] += 1

    # Spectator nucleons (p+A) are not source daughters; the same
    # particle-database on-shell check applies to each of them.
    for spectator in event.spectators:
        if get_particle(spectator.pdg_id) is None:
            continue
        configured = get_mass(spectator.pdg_id)
        reconstructed = _mass_from_components(
            spectator.p4.e, spectator.p4.px, spectator.p4.py, spectator.p4.pz
        )
        deviation = abs(reconstructed - configured) / max(configured, 1.0)
        mass_shell["n_checked"] += 1
        mass_shell["max_relative_deviation"] = max(
            float(mass_shell["max_relative_deviation"]), deviation
        )
        if deviation > relative_tolerance:
            mass_shell["n_failed"] += 1

    ancestry = _staged_ancestry(event)
    return VertexAuditReport(
        event_index=int(event_index),
        n_vertices=n_vertices,
        four_momentum=four_momentum,
        bqs=bqs,
        mass_shell=mass_shell,
        ancestry=ancestry,
        verdict=_verdict(four_momentum, bqs, mass_shell, ancestry),
    )


def _legacy_vertices(event: Event) -> list[tuple[Particle, list[Particle]]]:
    """Reconstruct the decay vertices of a legacy ``Event`` from the
    full parent-chain graph.

    The decay engine stamps ``Particle.parent`` at decay time and
    removes the parent from the particle list once it has decayed, so
    the live list holds only terminal tracks.  The vertex set is
    recovered by walking the parent links upward from every terminal:
    the collected particles are the terminals plus every intermediate
    daughter that itself decayed, and grouping all of them by their
    immediate parent restores the complete daughter lists.  For a
    two-body cascade ``N* -> Delta + pi -> N + pi + pi`` the recovered
    vertices are ``N* -> [Delta, pi]`` and ``Delta -> [N, pi]``.
    ``Particle.daughter`` / ``next_daughter`` are never assigned by the
    engine, so the parent links are the only ancestry data used.  A
    parent that was removed from the particle list after decaying is
    kept alive by these links, so no vertex is lost.
    """
    all_particles: dict[int, Particle] = {}
    for particle in event.particles():
        node: Particle | None = particle
        while node is not None and id(node) not in all_particles:
            all_particles[id(node)] = node
            node = node.parent

    groups: dict[int, tuple[Particle | None, list[Particle]]] = {}
    for particle in all_particles.values():
        parent = particle.parent
        key = id(parent)
        entry = groups.get(key)
        if entry is None:
            groups[key] = (parent, [particle])
        else:
            entry[1].append(particle)
    return [(parent, daughters) for parent, daughters in groups.values() if parent is not None]


def _legacy_ancestry(event: Event) -> dict:
    """Walk the parent links of every terminal track of a legacy event."""
    orphan_count = 0
    cycle_count = 0
    for particle in event.particles():
        if particle.get_gen_depth() < 0:
            orphan_count += 1
            continue
        node: Particle = particle
        seen: set[int] = set()
        while True:
            if node.get_is_primary() or node.get_weak_feed():
                break
            if id(node) in seen:
                cycle_count += 1
                break
            seen.add(id(node))
            if node.parent is None:
                orphan_count += 1
                break
            node = node.parent
    return {
        "acyclic": cycle_count == 0,
        "complete": orphan_count == 0,
        "orphan_count": orphan_count,
        "cycle_count": cycle_count,
    }


def audit_legacy_event(
    event: Event,
    relative_tolerance: float = 1e-10,
    event_index: int = 0,
) -> VertexAuditReport:
    """Audit one legacy ``Event`` without modifying it.

    One vertex per parent recovered from the full parent-chain graph of
    the terminal tracks; the checks are the per-vertex four-momentum
    and ``(B, Q, S)`` closures (strangeness exempt at declared weak
    vertices), the on-shell status of every daughter against the mass
    recorded at decay time, and the ancestry conditions including
    ``_gen_depth``.
    """
    four_momentum = {"n_checked": 0, "n_failed": 0, "max_relative_residual": 0.0}
    bqs = {"n_checked": 0, "n_failed": 0, "residuals": []}
    mass_shell = {"n_checked": 0, "n_failed": 0, "max_relative_deviation": 0.0}

    n_vertices = 0
    for parent, daughters in _legacy_vertices(event):
        n_vertices += 1

        parent_e = parent.get_e()
        parent_px = parent.get_px()
        parent_py = parent.get_py()
        parent_pz = parent.get_pz()
        scale = max(abs(parent_e), 1.0)
        relative_residual = (
            max(
                abs(sum(d.get_e() for d in daughters) - parent_e),
                abs(sum(d.get_px() for d in daughters) - parent_px),
                abs(sum(d.get_py() for d in daughters) - parent_py),
                abs(sum(d.get_pz() for d in daughters) - parent_pz),
            )
            / scale
        )
        four_momentum["n_checked"] += 1
        four_momentum["max_relative_residual"] = max(
            float(four_momentum["max_relative_residual"]), relative_residual
        )
        if relative_residual > relative_tolerance:
            four_momentum["n_failed"] += 1

        parent_qn = quantum_numbers(parent.get_id())
        daughter_qn = QuantumNumbers()
        for daughter in daughters:
            daughter_qn = daughter_qn + quantum_numbers(daughter.get_id())
        bqs_residual = (
            parent_qn.B - daughter_qn.B,
            parent_qn.Q - daughter_qn.Q,
            parent_qn.S - daughter_qn.S,
        )
        # Weak vertices may change strangeness by design; baryon number
        # and charge stay exact there.
        weak_vertex = parent.get_id() in WEAK_DECAY_PARENTS
        if weak_vertex:
            bqs_failed = bqs_residual[0] != 0 or bqs_residual[1] != 0
        else:
            bqs_failed = bqs_residual != (0, 0, 0)
        bqs["n_checked"] += 1
        if bqs_failed:
            bqs["n_failed"] += 1
            bqs["residuals"].append(bqs_residual)

        for daughter in daughters:
            configured = daughter.get_mass()
            reconstructed = _mass_from_components(
                daughter.get_e(),
                daughter.get_px(),
                daughter.get_py(),
                daughter.get_pz(),
            )
            deviation = abs(reconstructed - configured) / max(configured, 1.0)
            mass_shell["n_checked"] += 1
            mass_shell["max_relative_deviation"] = max(
                float(mass_shell["max_relative_deviation"]), deviation
            )
            if deviation > relative_tolerance:
                mass_shell["n_failed"] += 1

    ancestry = _legacy_ancestry(event)
    return VertexAuditReport(
        event_index=int(event_index),
        n_vertices=n_vertices,
        four_momentum=four_momentum,
        bqs=bqs,
        mass_shell=mass_shell,
        ancestry=ancestry,
        verdict=_verdict(four_momentum, bqs, mass_shell, ancestry),
    )


def vertex_audit_summary(reports: list[VertexAuditReport]) -> dict:
    """Reduce per-event audit reports to the run-level summary.

    ``ancestry_violations`` counts orphaned or cyclic ancestry records
    over all reports; ``all_pass`` is true when every report's verdict
    is ``"pass"`` (vacuously true for an empty list).
    """
    reports = list(reports)
    return {
        "n_events": len(reports),
        "n_vertices": sum(report.n_vertices for report in reports),
        "conservation_failures": sum(
            int(report.four_momentum["n_failed"]) for report in reports
        ),
        "offshell_daughters": sum(int(report.mass_shell["n_failed"]) for report in reports),
        "bqs_residual_nonzero": sum(int(report.bqs["n_failed"]) for report in reports),
        "ancestry_violations": sum(
            int(report.ancestry["orphan_count"]) + int(report.ancestry["cycle_count"])
            for report in reports
        ),
        "max_four_momentum_relative_residual": max(
            (float(report.four_momentum["max_relative_residual"]) for report in reports),
            default=0.0,
        ),
        "max_mass_shell_relative_deviation": max(
            (float(report.mass_shell["max_relative_deviation"]) for report in reports),
            default=0.0,
        ),
        "all_pass": all(report.verdict == "pass" for report in reports),
    }
