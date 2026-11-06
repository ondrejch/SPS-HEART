"""The decays the NA49 tables count (2026-11-04; the owner: "can we just decay the ones that NA49 does not correct
for?").

The cascade leaves every weak decay undone and does not perform Sigma0 -> Lambda gamma.  The NA49 tables subtract the
decay products of K0S, Lambda, Sigma+- and anti-Lambda from the pions (EPJC 45 (2006) 343, Sect. 5.6) and of Lambda,
Sigma and their antiparticles from the baryons (EPJC 65 (2010) 9, Sect. 6.4); the kaon tables carry no such correction
(EPJC 68 (2010) 1, the Omega- negligible); strong and electromagnetic decays stay in every table by definition.  The
hadrons outside that list therefore decay in the observable: the Sigma0 to Lambda gamma, the Xi0 and Xi- to Lambda pi,
and the Omega- to Lambda K-, Xi0 pi- and Xi- pi0, with their antiparticles, from
``config/v2_singlet/counted_decays_pdg2025.json`` (the dominant modes renormalized to one).  The Lambda they give stays
stable, as the tables subtract its products, and so do the K0L (it decays outside the detector) and the K0S and Sigma+-.

The decays act on the final state the observables bin (:func:`counted_final_state`, the run spec's
``counted_decays``), not on the event: the cascade keeps these hadrons at their pole masses, as a long-lived particle
is made, and the event record keeps its exact (B, Q, S), which a weak decay changes.  The channel and the decay angles
come from a hash of the hadron's identity and four-momentum, so the event's random streams are untouched and a final
state always decays the same way.  The NA49 pion tables keep the Xi decay pions at their vertex-track efficiency (up to
50% at low x_F); the full decay here is the upper bound of a change estimated below 0.1% of the pion yields.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from .hadron_catalog import HadronCatalog

__all__ = ["COUNTED_DECAYS_REL", "load_counted_decays", "counted_final_state"]

#: The configuration, relative to the nested repository root.
COUNTED_DECAYS_REL = Path("config") / "v2_singlet" / "counted_decays_pdg2025.json"
_SELF_CONJUGATE = {22, 111}
_TABLES: dict = {}


def _conjugate(pdg: int) -> int:
    return pdg if pdg in _SELF_CONJUGATE else -pdg


def load_counted_decays(nested_root) -> dict[int, tuple[tuple[tuple[int, ...], float], ...]]:
    """The decay table ``{pdg: ((daughters, cumulative fraction), ...)}``
    with the antiparticles (memoized per root)."""
    key = str(Path(nested_root).resolve())
    hit = _TABLES.get(key)
    if hit is not None:
        return hit
    config = json.loads((Path(nested_root) / COUNTED_DECAYS_REL).read_text(encoding="utf-8"))
    table = {}
    for spec in config["decays"]:
        channels = [(tuple(int(d) for d in c["daughters"]), float(c["fraction"])) for c in spec["channels"]]
        total = math.fsum(f for _d, f in channels)
        for sign in (1, -1):
            pdg = int(spec["pdg_id"]) * sign
            acc, rows = 0.0, []
            for daughters, fraction in channels:
                acc += fraction / total
                rows.append((tuple(d if sign == 1 else _conjugate(d) for d in daughters), acc))
            table[pdg] = tuple(rows)
    _TABLES[key] = table
    return table


def _uniforms(pdg: int, p4) -> tuple[float, float, float]:
    """Three uniforms in [0, 1) from the hadron's identity and four-momentum."""
    digest = hashlib.sha256(struct.pack("<q4d", int(pdg), *(float(x) for x in p4))).digest()
    return tuple(int.from_bytes(digest[8 * k:8 * k + 7], "little") / float(1 << 56) for k in range(3))


def _boost(p4_rest, parent_p4, parent_mass: float):
    """The rest-frame four-momentum *p4_rest* boosted to the frame where the parent has *parent_p4*."""
    e_p, px_p, py_p, pz_p = (float(x) for x in parent_p4)
    bx, by, bz = px_p / e_p, py_p / e_p, pz_p / e_p
    b2 = bx * bx + by * by + bz * bz
    gamma = e_p / parent_mass
    e, px, py, pz = p4_rest
    bp = bx * px + by * py + bz * pz
    k = (gamma - 1.0) * bp / b2 + gamma * e if b2 > 0.0 else 0.0
    return (gamma * (e + bp), px + k * bx, py + k * by, pz + k * bz)


def _decay(hadron, catalog: HadronCatalog, table, mass_of, final_of):
    rows = table[int(hadron.pdg_id)]
    u_channel, u_cos, u_phi = _uniforms(int(hadron.pdg_id), hadron.four_momentum)
    daughters = next(d for d, cum in rows if u_channel < cum or cum == rows[-1][1])
    m0 = float(hadron.mass)
    m1, m2 = (mass_of(d) for d in daughters)
    q = math.sqrt(max((m0 * m0 - (m1 + m2) ** 2) * (m0 * m0 - (m1 - m2) ** 2), 0.0)) / (2.0 * m0)
    cos_t = 2.0 * u_cos - 1.0
    sin_t = math.sqrt(max(1.0 - cos_t * cos_t, 0.0))
    phi = 2.0 * math.pi * u_phi
    n = (sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t)
    out = []
    for d, m, sign in ((daughters[0], m1, 1.0), (daughters[1], m2, -1.0)):
        rest = (math.sqrt(m * m + q * q), sign * q * n[0], sign * q * n[1], sign * q * n[2])
        out.append(final_of(d, _boost(rest, hadron.four_momentum, m0), hadron.line))
    return out


def counted_final_state(finals, catalog: HadronCatalog, nested_root) -> list:
    """The final state with the counted decays performed, recursively (an
    Omega- to Xi pi to Lambda pi pi); every other hadron unchanged, in
    order."""
    from .topology_single_diffraction import _final_from_catalog

    table = load_counted_decays(nested_root)

    def mass_of(pdg: int) -> float:
        return 0.0 if pdg == 22 else float(catalog._daughter_mass(pdg))

    def final_of(pdg: int, p4, line: str):
        return _final_from_catalog(catalog, pdg, p4, line)

    out, stack = [], list(reversed(list(finals)))
    while stack:
        h = stack.pop()
        if h.pdg_id is not None and int(h.pdg_id) in table:
            stack.extend(reversed(_decay(h, catalog, table, mass_of, final_of)))
        else:
            out.append(h)
    return out
