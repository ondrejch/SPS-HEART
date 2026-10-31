"""Built-in PDG particle database — replacement for ROOT's ``TDatabasePDG``.

Each entry is keyed by PDG Monte Carlo ID and stores ``mass`` (GeV/c²),
``charge`` (in units of the elementary charge), and ``stable`` flag.
Only the particles needed by the NA49 framework are listed; add more as
required.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class ParticleData:
    pdg_id: int
    name: str
    mass: float        # GeV/c²
    charge: float      # units of e
    stable: bool = False
    width: float = 0.0 # decay width [GeV]


# PDG particle table (mass in GeV/c², charge in units of e)
_PDG_TABLE: list[ParticleData] = [
    # leptons
    ParticleData(11,    "e-",       0.00051099,  -1.0, True),
    ParticleData(-11,   "e+",       0.00051099,   1.0, True),
    ParticleData(13,    "mu-",      0.10566,     -1.0, False),
    ParticleData(-13,   "mu+",      0.10566,      1.0, False),
    # pions
    ParticleData(211,   "pi+",      0.1395699,    1.0, True),
    ParticleData(-211,  "pi-",      0.1395699,   -1.0, True),
    ParticleData(111,   "pi0",      0.1349766,    0.0, False),
    # kaons
    ParticleData(321,   "K+",       0.493677,     1.0, True),
    ParticleData(-321,  "K-",       0.493677,    -1.0, True),
    ParticleData(311,   "K0",       0.497611,     0.0, False),
    ParticleData(-311,  "aK0",      0.497611,     0.0, False),
    ParticleData(310,   "K0S",      0.497611,     0.0, False),
    ParticleData(130,   "K0L",      0.497611,     0.0, False),
    # kaon resonances
    ParticleData(323,   "K*(892)+", 0.89166,      1.0, False, 0.0508),
    ParticleData(-323,  "K*(892)-", 0.89166,     -1.0, False, 0.0508),
    ParticleData(313,   "K*(892)0", 0.89555,      0.0, False, 0.0473),
    ParticleData(-313,  "aK*(892)0",0.89555,      0.0, False, 0.0473),
    ParticleData(10313, "K1(1270)0", 1.272,       0.0, False, 0.090),
    ParticleData(-10313,"aK1(1270)0",1.272,       0.0, False, 0.090),
    ParticleData(10323, "K1(1270)+", 1.272,       1.0, False, 0.090),
    ParticleData(-10323,"K1(1270)-", 1.272,      -1.0, False, 0.090),
    ParticleData(315,   "K2*(1430)0",1.4256,      0.0, False, 0.0985),
    ParticleData(-315,  "aK2*(1430)0",1.4256,     0.0, False, 0.0985),
    ParticleData(325,   "K2*(1430)+",1.4256,      1.0, False, 0.0985),
    ParticleData(-325,  "K2*(1430)-",1.4256,     -1.0, False, 0.0985),
    ParticleData(333,   "phi(1020)",1.019461,     0.0, False, 0.004249),
    # nucleons
    ParticleData(2212,  "proton",   0.93827231,   1.0, True),
    ParticleData(-2212, "pbar",     0.93827231,  -1.0, True),
    ParticleData(2112,  "neutron",  0.93956541,   0.0, False),
    ParticleData(-2112, "nbar",     0.93956541,   0.0, False),
    # strange baryons
    ParticleData(3122,  "Lambda",   1.115683,     0.0, False),
    ParticleData(-3122, "aLambda",  1.115683,     0.0, False),
    ParticleData(3222,  "Sigma+",   1.18937,      1.0, False),
    ParticleData(3212,  "Sigma0",   1.192642,     0.0, False),
    ParticleData(3112,  "Sigma-",   1.197449,    -1.0, False),
    ParticleData(102134, "Lambda(1520)", 1.5195,  0.0, False, 0.0156),
    ParticleData(3224,  "Sigma(1385)+", 1.3828,   1.0, False, 0.036),
    ParticleData(3214,  "Sigma(1385)0", 1.3837,   0.0, False, 0.036),
    ParticleData(3114,  "Sigma(1385)-", 1.3872,  -1.0, False, 0.039),
    # Delta resonances
    ParticleData(2224,  "Delta++",  1.232,        2.0, False, 0.117),
    ParticleData(2214,  "Delta+",   1.232,        1.0, False, 0.117),
    ParticleData(2114,  "Delta0",   1.232,        0.0, False, 0.117),
    ParticleData(1114,  "Delta-",   1.232,       -1.0, False, 0.117),
    # Anti-delta charges are the charge conjugates of the matter states:
    # anti(Delta-) has charge +1, anti(Delta+) -1, anti(Delta++) -2.
    # -1114 is used exclusively as physical anti(Delta-); generator cards
    # reference matter Delta states by their true ids (2214 for Delta+).
    ParticleData(-1114, "aDelta-",  1.232,        1.0, False, 0.117),
    ParticleData(-2114, "aDelta0",  1.232,        0.0, False, 0.117),
    ParticleData(-2214, "aDelta+",  1.232,       -1.0, False, 0.117),
    ParticleData(-2224, "aDelta--", 1.232,       -2.0, False, 0.117),
    # higher Delta resonances
    ParticleData(202224, "Delta(1600)++", 1.600,  2.0, False, 0.350),
    ParticleData(202214, "Delta(1600)+",  1.600,  1.0, False, 0.350),
    ParticleData(202114, "Delta(1600)0",  1.600,  0.0, False, 0.350),
    ParticleData(201114, "Delta(1600)-",  1.600, -1.0, False, 0.350),
    ParticleData(122224, "Delta(1700)++", 1.700,  2.0, False, 0.300),
    ParticleData(122214, "Delta(1700)+",  1.700,  1.0, False, 0.300),
    ParticleData(122114, "Delta(1700)0",  1.700,  0.0, False, 0.300),
    ParticleData(121114, "Delta(1700)-",  1.700, -1.0, False, 0.300),
    # N* resonances
    ParticleData(12112, "N*(1440)0",1.440,        0.0, False, 0.300),
    ParticleData(-12112, "aN*(1440)0",1.440,      0.0, False, 0.300),
    ParticleData(12212, "N*(1440)+",1.440,        1.0, False, 0.300),
    ParticleData(2116,  "N*(1520)0",1.520,        0.0, False, 0.115),
    ParticleData(-2116,  "aN*(1520)0",1.520,      0.0, False, 0.115),
    ParticleData(2216,  "N*(1520)+",1.520,        1.0, False, 0.115),
    ParticleData(22212, "N*(1680)+",1.685,        1.0, False, 0.130),
    ParticleData(22112, "N*(1680)0",1.685,        0.0, False, 0.130),
    ParticleData(-22112, "aN*(1680)0",1.685,      0.0, False, 0.130),
    ParticleData(102112, "N(1535)0", 1.535,       0.0, False, 0.150),
    ParticleData(-102112, "aN(1535)0", 1.535,     0.0, False, 0.150),
    ParticleData(102212, "N(1535)+", 1.535,       1.0, False, 0.150),
    ParticleData(122112, "N(1650)0", 1.655,       0.0, False, 0.125),
    ParticleData(-122112, "aN(1650)0", 1.655,     0.0, False, 0.125),
    ParticleData(122212, "N(1650)+", 1.655,       1.0, False, 0.125),
    ParticleData(212114, "N(1720)0", 1.720,       0.0, False, 0.200),
    ParticleData(-212114, "aN(1720)0", 1.720,     0.0, False, 0.200),
    ParticleData(212214, "N(1720)+", 1.720,       1.0, False, 0.200),
    # effective high-mass N* tower (Regge-inspired placeholders)
    ParticleData(9911201, "N_eff(1900)0", 1.900,  0.0, False, 0.250),
    ParticleData(9912201, "N_eff(1900)+", 1.900,  1.0, False, 0.250),
    ParticleData(-9911201, "aN_eff(1900)0", 1.900, 0.0, False, 0.250),
    # Charged anti-partners of the positive matter tower levels (needed for
    # the 1/3-pi0 : 2/3-pi charged split of Rbar -> Rbar' pi cascades).
    ParticleData(-212214, "aN_eff(1720)-", 1.720, -1.0, False, 0.200),
    ParticleData(-9912201, "aN_eff(1900)-", 1.900, -1.0, False, 0.250),
    ParticleData(9921201, "N_eff(2100)0", 2.100,  0.0, False, 0.300),
    ParticleData(9922201, "N_eff(2100)+", 2.100,  1.0, False, 0.300),
    ParticleData(-9921201, "aN_eff(2100)0", 2.100, 0.0, False, 0.300),
    ParticleData(-9922201, "aN_eff(2100)-", 2.100, -1.0, False, 0.300),
    ParticleData(9931201, "N_eff(2300)0", 2.300,  0.0, False, 0.350),
    ParticleData(9932201, "N_eff(2300)+", 2.300,  1.0, False, 0.350),
    ParticleData(-9931201, "aN_eff(2300)0", 2.300, 0.0, False, 0.350),
    ParticleData(-9932201, "aN_eff(2300)-", 2.300, -1.0, False, 0.350),
    # effective meson tower coupled to high-mass N* cascading
    ParticleData(9900113, "M_eff(450)0", 0.450,   0.0, False, 0.180),
    ParticleData(9900213, "M_eff(850)0", 0.850,   0.0, False, 0.220),
    ParticleData(9900313, "M_eff(1250)0", 1.250,  0.0, False, 0.280),
    # rho mesons (ground state)
    ParticleData(113,   "rho0",     0.77526,      0.0, False, 0.1491),
    ParticleData(213,   "rho+",     0.77526,      1.0, False, 0.1491),
    ParticleData(-213,  "rho-",     0.77526,     -1.0, False, 0.1491),
    # axial/vector/tensor isovector mesons
    ParticleData(10113, "b1(1235)0",1.2295,       0.0, False, 0.142),
    ParticleData(10213, "b1(1235)+",1.2295,       1.0, False, 0.142),
    ParticleData(-10213,"b1(1235)-",1.2295,      -1.0, False, 0.142),
    ParticleData(20113, "a1(1260)0",1.230,        0.0, False, 0.420),
    ParticleData(20213, "a1(1260)+",1.230,        1.0, False, 0.420),
    ParticleData(-20213,"a1(1260)-",1.230,       -1.0, False, 0.420),
    ParticleData(115,   "a2(1320)0",1.3183,       0.0, False, 0.107),
    ParticleData(215,   "a2(1320)+",1.3183,       1.0, False, 0.107),
    ParticleData(-215,  "a2(1320)-",1.3183,      -1.0, False, 0.107),
    # rho(1450) excited state
    ParticleData(100113,  "rho(1450)0", 1.465,   0.0, False, 0.400),
    ParticleData(100213,  "rho(1450)+", 1.465,   1.0, False, 0.400),
    ParticleData(-100213, "rho(1450)-", 1.465,  -1.0, False, 0.400),
    # rho(1700) excited state
    ParticleData(30113,  "rho(1700)0", 1.720,    0.0, False, 0.250),
    ParticleData(30213,  "rho(1700)+", 1.720,    1.0, False, 0.250),
    ParticleData(-30213, "rho(1700)-", 1.720,   -1.0, False, 0.250),
    # f2 meson
    ParticleData(225,   "f2(1270)", 1.2755,       0.0, False, 0.187),
    # rho3(1690) tensor mesons
    ParticleData(117,   "rho3(1690)0", 1.6888,    0.0, False, 0.161),
    ParticleData(217,   "rho3(1690)+", 1.6888,    1.0, False, 0.161),
    ParticleData(-217,  "rho3(1690)-", 1.6888,   -1.0, False, 0.161),
    # f4(2050) tensor meson
    ParticleData(229,   "f4(2050)", 2.018,        0.0, False, 0.237),
    # photon
    ParticleData(22,    "gamma",    0.0,           0.0, True),
    # eta
    ParticleData(221,   "eta",      0.547862,      0.0, False, 1.31e-6),
    # omega mesons (ground state)
    ParticleData(223,   "omega",    0.78265,       0.0, False, 0.00849),
    # omega(1420) excited state
    ParticleData(100223, "omega(1420)", 1.410,    0.0, False, 0.290),
    # omega(1650) excited state
    ParticleData(30223,  "omega(1650)", 1.670,    0.0, False, 0.315),
    # special NA49 "cluster" used in FSB identification
    ParticleData(2219,  "cluster",  0.93827231,    1.0, False),
]

# Build lookup dict
_TABLE: dict[int, ParticleData] = {p.pdg_id: p for p in _PDG_TABLE}

# Process-local, concurrency-safe mass/width overrides (no mutation of _TABLE).
# Keys are PDG IDs; values are (mass_GeV, width_GeV).
_MASS_WIDTH_OVERRIDES: ContextVar[Optional[Mapping[int, tuple[float, float]]]] = ContextVar(
    "na49_mass_width_overrides",
    default=None,
)


@contextmanager
def mass_width_overrides(updates: Mapping[int, tuple[float, float]]):
    """Temporarily override mass/width lookups without mutating the PDG table.

    Nested contexts replace (do not merge) the active override map. Use this for
    effective N*/M* tower mass scans so concurrent configs cannot corrupt the
    shared table.
    """
    clean: dict[int, tuple[float, float]] = {}
    for raw_pid, pair in dict(updates).items():
        pid = int(raw_pid)
        mass = float(pair[0])
        width = float(pair[1])
        clean[pid] = (mass, width)
    token = _MASS_WIDTH_OVERRIDES.set(clean)
    try:
        yield clean
    finally:
        _MASS_WIDTH_OVERRIDES.reset(token)


def _override_pair(pdg_id: int) -> Optional[tuple[float, float]]:
    active = _MASS_WIDTH_OVERRIDES.get()
    if not active:
        return None
    return active.get(int(pdg_id))


def get_particle(pdg_id: int) -> Optional[ParticleData]:
    """Return :class:`ParticleData` for the given PDG ID, or ``None``."""
    return _TABLE.get(pdg_id)


def get_mass(pdg_id: int, default: float = 0.0) -> float:
    """Return the mass (GeV/c²) for *pdg_id*, or *default* if unknown."""
    ovr = _override_pair(pdg_id)
    if ovr is not None and math.isfinite(ovr[0]) and ovr[0] > 0.0:
        return float(ovr[0])
    p = _TABLE.get(pdg_id)
    return p.mass if p is not None else default


def get_charge(pdg_id: int, default: float = 0.0) -> float:
    """Return the charge (units of *e*) for *pdg_id*, or *default* if unknown."""
    p = _TABLE.get(pdg_id)
    return p.charge if p is not None else default


def get_width(pdg_id: int, default: float = 0.0) -> float:
    """Return the decay width (GeV) for *pdg_id*, or *default* if unknown."""
    ovr = _override_pair(pdg_id)
    if ovr is not None and math.isfinite(ovr[1]) and ovr[1] >= 0.0:
        return float(ovr[1])
    p = _TABLE.get(pdg_id)
    return p.width if p is not None else default


def is_stable(pdg_id: int) -> bool:
    """Return ``True`` if the particle with *pdg_id* is considered stable."""
    p = _TABLE.get(pdg_id)
    return p.stable if p is not None else False


# Particles treated as stable in the analysis regardless of the PDG flag
_STABLE_IDS = frozenset({2212, -2212, 211, -211, 11, -11, 321, -321})


def is_analysis_stable(pdg_id: int) -> bool:
    """Return ``True`` if the particle is stable *for analysis purposes*."""
    if abs(pdg_id) in {2212, 211, 11, 321}:
        return True
    return is_stable(pdg_id)
