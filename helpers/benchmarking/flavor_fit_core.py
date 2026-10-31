"""Shared core of the Round 6 flavor runs R61 to R65 (review record code
fix C2): ledger-recording mixture generation at a topology point,
vectorized flavor reweighting, species predictions on the NA49 tables,
the plan section 5.4 objective, fits, replica blocks, Hessians, and the
held-out kaon moments.

The flavor weights change densities but never the kinematic support
(every factor is positive), so a sample generated at one flavor point is
reweighted exactly to any other through the per-vertex ledger records of
``na49_mixing.flavor_weights`` (plan section 5.3).  The ledger of one
event is flattened into arrays: per informative vertex the chosen
exponent key, per pool group the key and the summed pool weight as
drawn.  Keys are stored as ``(2 n_ssbar, n_V, n_BBbar, n_assoc)`` small
integers (the hidden-strangeness content is 1.0 for the phi, so twice
``n_ssbar`` is integral on the committed catalog).
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import run_contract as RC
import run_observables as OB

from na49_mixing.flavor_weights import FlavorParams, NOMINAL_FLAVOR
from na49_mixing.hadron_catalog import HadronCatalog
from na49_mixing.importance import ess_fraction
from na49_mixing.rng_substreams import derive_substreams
from na49_mixing.spectral_density import families_from_json
from na49_mixing.topology_runner import TopologyNominalParams, draw_topology, extract_observables, run_topology

__all__ = ["FLAVOR_NAMES", "flavor_from_dict", "flavor_to_dict", "generate_mixture_slot", "arrays_digest",
           "FlavorSample", "FlavorObjective", "run_fit", "hessian", "kaon_pt_moments", "read_ptintegrated"]

FLAVOR_NAMES = ("gamma_s_had", "r_v", "r_bbar", "assoc_norm")
CLASS_ORDER = ("sd_projectile", "sd_target", "dd", "nd")
TRACK_SPECIES = OB.SPECIES_ORDER
SQRT_S = OB.SQRT_S
EVENT_CLOSURE_TOL = 1.0e-9
SE_FLOOR = 1.0e-9
NORM_PRIOR = 0.02
ESS_FLOOR = 0.1
ESS_INTERIOR = 0.3
TRUST_PENALTY = 1.0e12
LOG_BOUNDS = (math.log(0.05), math.log(20.0))


def flavor_from_dict(d: dict) -> FlavorParams:
    return FlavorParams(**{k: float(d.get(k, 1.0)) for k in FLAVOR_NAMES})


def flavor_to_dict(p: FlavorParams) -> dict:
    return {k: float(getattr(p, k)) for k in FLAVOR_NAMES}


def _load_physics():
    nested = RC.locate_roots(Path(__file__))[0]
    catalog = HadronCatalog.from_json(nested / "config" / "v2_singlet" / "resonance_catalog.json")
    fams = families_from_json(str(nested / "config" / "v2_singlet" / "continuum_resonances.json"))
    return catalog, fams


def _line_qn(h):
    return int(h.qn[0]), int(h.qn[1]), int(h.qn[2])


def event_conservation(event) -> float:
    e_beam = math.sqrt(OB.PROTON_MASS ** 2 + OB.BEAM_MOMENTUM ** 2)
    e_total = e_beam + OB.PROTON_MASS
    finals = list(event.final_state)
    sums = [math.fsum(float(h.four_momentum[j]) for h in finals) for j in range(4)]
    bqs = tuple(sum(_line_qn(h)[j] for h in finals) for j in range(3))
    if bqs != (2, 2, 0):
        raise RuntimeError(f"conservation: (B, Q, S) = {bqs}")
    worst = max(abs(sums[0] - e_total) / e_total, abs(sums[1]) / OB.BEAM_MOMENTUM, abs(sums[2]) / OB.BEAM_MOMENTUM,
                abs(sums[3] - OB.BEAM_MOMENTUM) / OB.BEAM_MOMENTUM)
    if worst > EVENT_CLOSURE_TOL:
        raise RuntimeError(f"conservation: closure {worst!r}")
    return worst


def _key_ints(key, scale: int = 2) -> tuple:
    """The integer ledger key: n_ssbar times *scale* (2, the half-integral
    counting through R97; 10 for a fractional hidden s-sbar content, the
    eta'(958) at 0.6, 2026-10-15) and the other exponents as they are,
    the named channel factors' counts included (2026-10-17)."""
    scaled = round(float(scale) * float(key[0]))
    if abs(scaled - float(scale) * float(key[0])) > 1e-6:
        raise ValueError(f"n_ssbar {key[0]!r} is not a multiple of 1/{scale}")
    return (int(scaled),) + tuple(int(v) for v in key[1:])


#: The frozen diffractive remnant split (Round 5 card 3): the un-excited line
#: of a single-diffraction event is intact or a low-mass resonance; the
#: fitted substrate weights apply to the non-diffractive class only.
DIFFRACTIVE_SPLIT = {"intact_proton": 0.5, "low_mass_baryonic_resonance": 0.5, "high_mass_baryonic_continuum": 0.0}


def generate_mixture_slot(seed: int, n_events: int, partition_base: int, topology: dict,
                          flavor_gen: FlavorParams | None) -> dict:
    """One mixture slot at the topology point ``topology = {"fractions":
    {class: f}, "epsilon": e, "b_sd": b}`` and the generation flavor point
    (``None`` is the identity): the class drawn per event on the
    ``topology_selection`` substream with the elastic fraction 0."""
    catalog, fams = _load_physics()
    fractions = {cls: float(topology["fractions"][cls]) for cls in CLASS_ORDER}
    # The substrate's remnant weights are the non-diffractive class's (the
    # R51c fit releases them for that class only). The un-excited line of a
    # single-diffraction event keeps the frozen diffractive split, whose
    # continuum weight is zero by class definition (generate_sd guards it).
    nd_weights = dict(topology.get("remnant_weights") or DIFFRACTIVE_SPLIT)
    diffractive_split = dict(topology.get("diffractive_split") or DIFFRACTIVE_SPLIT)
    common = dict(epsilon=float(topology["epsilon"]), b_sd=float(topology["b_sd"]), class_fractions=fractions, p_n2=0.0, flavor=flavor_gen)
    params_by_class = {cls: TopologyNominalParams(remnant_weights=(nd_weights if cls == "nd" else diffractive_split), **common)
                       for cls in CLASS_ORDER}
    beta, gamma = OB.cm_boost(SQRT_S)
    species_pdg = {name: OB.SPECIES_PDG[name] for name in TRACK_SPECIES}
    cls_index = np.zeros(n_events, dtype=np.int8)
    n_ch = np.zeros(n_events, dtype=np.int16)
    strange = {name: np.zeros(n_events, dtype=np.int16) for name in OB.STRANGE_GROUPS}
    trk_event, trk_species, trk_xf, trk_pt, trk_w = [], [], [], [], []
    led_event, led_chosen, led_gvertex, led_gkey, led_gw = [], [], [], [], []
    vertex_counter = 0
    rolling = hashlib.sha256()
    census = {cls: 0 for cls in CLASS_ORDER}
    worst = 0.0
    t0 = time.perf_counter()
    for i in range(int(n_events)):
        subs = derive_substreams(int(seed), 0, int(partition_base) + i)
        rngs = {f"rng_{label}": np.random.default_rng(subs.stream_for(label))
                for label in ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")}
        label = draw_topology(fractions, 0.0, rngs["rng_topology_selection"])
        if label not in CLASS_ORDER:
            raise RuntimeError(f"drawn class {label!r} outside the inelastic set")
        ledger: list = []
        event = run_topology(label, params_by_class[label], catalog, fams, **rngs, ledger=ledger)
        worst = max(worst, event_conservation(event))
        census[label] += 1
        cls_index[i] = CLASS_ORDER.index(label)
        n_ch[i] = OB.charged_multiplicity(event.final_state)
        counts = OB.strange_counts(event.final_state)
        for name in strange:
            strange[name][i] = counts[name]
        tracks = OB.track_arrays(event.final_state, species_pdg, beta, gamma, SQRT_S)
        line = [label, str(i), str(int(n_ch[i]))]
        for s_index, name in enumerate(TRACK_SPECIES):
            xs, ps, ws = tracks[name]
            for xf, pt, w in zip(xs, ps, ws):
                trk_event.append(i); trk_species.append(s_index); trk_xf.append(xf); trk_pt.append(pt); trk_w.append(w)
                line.append(f"{s_index}:{xf.hex()}:{pt.hex()}:{w.hex()}")
        for record in ledger:
            if record is None:
                continue
            chosen_key, groups = record
            led_event.append(i); led_chosen.append(_key_ints(chosen_key))
            for key, weight in sorted(groups.items()):
                led_gvertex.append(vertex_counter); led_gkey.append(_key_ints(key)); led_gw.append(float(weight))
                line.append(f"L{vertex_counter}:{'/'.join(str(v) for v in _key_ints(key))}:{float(weight).hex()}")
            vertex_counter += 1
        rolling.update(("|".join(line) + "\n").encode("utf-8"))
    return {
        "seed": int(seed), "n_events": int(n_events), "partition_base": int(partition_base),
        "topology": {"fractions": fractions, "epsilon": float(topology["epsilon"]), "b_sd": float(topology["b_sd"]),
                     "remnant_weights_nd": nd_weights, "diffractive_split": diffractive_split},
        "flavor_generation": flavor_to_dict(flavor_gen or NOMINAL_FLAVOR),
        "arrays": {
            "cls": cls_index, "n_ch": n_ch, **{f"strange_{k}": v for k, v in strange.items()},
            "trk_event": np.asarray(trk_event, dtype=np.int32), "trk_species": np.asarray(trk_species, dtype=np.int8),
            "trk_xf": np.asarray(trk_xf, dtype=float), "trk_pt": np.asarray(trk_pt, dtype=float), "trk_w": np.asarray(trk_w, dtype=float),
            "led_event": np.asarray(led_event, dtype=np.int32), "led_chosen": np.asarray(led_chosen, dtype=np.int16).reshape(-1, 4),
            "led_gvertex": np.asarray(led_gvertex, dtype=np.int32), "led_gkey": np.asarray(led_gkey, dtype=np.int16).reshape(-1, 4),
            "led_gw": np.asarray(led_gw, dtype=np.float32),
        },
        "class_census": census, "informative_vertices": int(vertex_counter),
        "conservation": {"events": int(n_events), "violations": 0, "worst_closure_relative": worst, "tolerance": EVENT_CLOSURE_TOL},
        "event_rolling_sha256": rolling.hexdigest(), "tracks": int(len(trk_event)), "wall_seconds": time.perf_counter() - t0,
    }


def arrays_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode()); digest.update(str(arr.dtype).encode()); digest.update(str(arr.shape).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Reweighting
# ----------------------------------------------------------------------
def _log_factor(keys: np.ndarray, params: FlavorParams, scale: int = 2) -> np.ndarray:
    """The log flavor factor of ledger keys whose first entry is n_ssbar
    times *scale* (2 through R97; the run engine stores 10 with the
    eta'(958), review 2026-10-16, L3)."""
    k = np.asarray(keys, dtype=float)
    return (k[:, 0] / float(scale) * math.log(params.gamma_s_had) + k[:, 1] * math.log(params.r_v)
            + k[:, 2] * math.log(params.r_bbar) + k[:, 3] * math.log(params.assoc_norm))


def flavor_log_weights(arrays: dict, n_events: int, params_from: FlavorParams, params_to: FlavorParams) -> np.ndarray:
    """Per-event log importance weights from the generation flavor point to
    *params_to* (the flattened-ledger form of ``ledger_log_weight``)."""
    if flavor_to_dict(params_from) == flavor_to_dict(params_to):
        return np.zeros(n_events)
    n_vertices = int(arrays["led_event"].size)
    if n_vertices == 0:
        return np.zeros(n_events)
    gw = arrays["led_gw"].astype(float)
    gv = arrays["led_gvertex"].astype(np.int64)
    # the slot's n_ssbar scale (the run engine's led_ssbar_scale; slots without it carry 2)
    scale = int(np.asarray(arrays["led_ssbar_scale"]).ravel()[0]) if "led_ssbar_scale" in arrays else 2
    ratio = np.exp(_log_factor(arrays["led_gkey"], params_to, scale) - _log_factor(arrays["led_gkey"], params_from, scale))
    z_from = np.bincount(gv, weights=gw, minlength=n_vertices)[:n_vertices]
    z_to = np.bincount(gv, weights=gw * ratio, minlength=n_vertices)[:n_vertices]
    chosen = (_log_factor(arrays["led_chosen"], params_to, scale)
              - _log_factor(arrays["led_chosen"], params_from, scale))
    per_vertex = chosen + np.log(z_from) - np.log(z_to)
    return np.bincount(arrays["led_event"].astype(np.int64), weights=per_vertex, minlength=n_events)[:n_events]


# ----------------------------------------------------------------------
# Samples and predictions
# ----------------------------------------------------------------------
class FlavorSample:
    """The loaded slots of one generation with the species tracks binned to
    the tables (``{species: table}``)."""

    def __init__(self, slots: Sequence[dict], tables: dict):
        self.tables = tables
        self.blocks = []
        self.generation_point = None
        for meta, arrays in slots:
            point = flavor_from_dict(meta["flavor_generation"])
            if self.generation_point is None:
                self.generation_point = point
            elif flavor_to_dict(point) != flavor_to_dict(self.generation_point):
                raise ValueError("mixed generation flavor points")
            self.blocks.append(self._prepare(meta, arrays))
        self.n_blocks = len(self.blocks)

    def _prepare(self, meta: dict, arrays: dict) -> dict:
        n = int(meta["n_events"])
        binned = {}
        for s_index, name in enumerate(TRACK_SPECIES):
            if name not in self.tables:
                continue
            table = self.tables[name]
            sel = arrays["trk_species"] == s_index
            ev = arrays["trk_event"][sel]; xf = arrays["trk_xf"][sel]; pt = arrays["trk_pt"][sel]; w = arrays["trk_w"][sel]
            xedges = OB.sps_core.edges_from_centers(table.xf_centers); pedges = OB.sps_core.edges_from_centers(table.pt_centers)
            n_x = xedges.size - 1; n_p = pedges.size - 1
            pick = table.bin_index_pt * n_x + table.bin_index_xf
            lookup = np.full(n_x * n_p, -1, dtype=np.int64); lookup[pick] = np.arange(pick.size)
            if xf.size:
                ix, ip = OB.sps_core.grid_indices(xf, pt, xedges, pedges)
                ok = (ix >= 0) & (ix < n_x) & (ip >= 0) & (ip < n_p)
                pos = lookup[ip[ok] * n_x + ix[ok]]; keep = pos >= 0
                binned[name] = (ev[ok][keep].astype(np.int64), pos[keep], w[ok][keep], int(table.data.size),
                                xf[ok][keep], pt[ok][keep])
            else:
                binned[name] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0), int(table.data.size), np.zeros(0), np.zeros(0))
        return {"n": n, "arrays": arrays, "binned": binned, "meta": meta}

    def log_weights(self, block: dict, params: FlavorParams) -> np.ndarray:
        return flavor_log_weights(block["arrays"], block["n"], self.generation_point, params)

    @staticmethod
    def normalized(log_w: np.ndarray) -> np.ndarray:
        w = np.exp(log_w - np.max(log_w))
        return w / np.sum(w)

    def species_prediction(self, block: dict, wn: np.ndarray, name: str) -> np.ndarray:
        ev, pos, w, nbins, _, _ = block["binned"][name]
        table = self.tables[name]
        if ev.size == 0:
            return np.zeros(nbins)
        sums = np.bincount(pos, weights=w * wn[ev], minlength=nbins)[:nbins]
        return sums / (math.pi * SQRT_S * OB.bin_areas(table))

    def mean_pt_by_xf(self, block: dict, wn: np.ndarray, name: str, xf_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted track-level mean p_T of one species per x_F bin (the
        number-density convention of the p_T-integrated tables)."""
        ev, _, _, _, xf, pt = block["binned"][name]
        if ev.size == 0:
            return np.zeros(xf_edges.size - 1), np.zeros(xf_edges.size - 1)
        ix = np.digitize(xf, xf_edges) - 1
        ok = (ix >= 0) & (ix < xf_edges.size - 1)
        num = np.bincount(ix[ok], weights=pt[ok] * wn[ev[ok]], minlength=xf_edges.size - 1)
        den = np.bincount(ix[ok], weights=wn[ev[ok]], minlength=xf_edges.size - 1)
        return num, den


class FlavorObjective:
    """The plan section 5.4 objective for a set of statistics, each a table
    spectrum (``{"name", "species"}``) or a difference of two species on
    their common bins (``{"name", "species", "minus"}``), with one shared
    multiplicative normalization nuisance profiled, optional log-normal
    priors on the free parameters, and the effective-sample-size trust
    region on the flavor weights."""

    def __init__(self, sample: FlavorSample, statistics: Sequence[dict], free: Sequence[str], fixed: FlavorParams,
                 priors: dict | None = None):
        self.s = sample
        self.free = tuple(free)
        self.fixed = fixed
        self.priors = priors or {}
        self.stats = []
        d_parts, v_parts = [], []
        for stat in statistics:
            table = sample.tables[stat["species"]]
            if stat.get("minus"):
                other = sample.tables[stat["minus"]]
                key = lambda t, i: (round(float(t.xf_centers[t.bin_index_xf[i]]), 6), round(float(t.pt_centers[t.bin_index_pt[i]]), 6))  # noqa: E731
                index_other = {key(other, j): j for j in range(other.data.size)}
                pairs = [(i, index_other[key(table, i)]) for i in range(table.data.size) if key(table, i) in index_other]
                ia = np.asarray([p[0] for p in pairs], dtype=np.int64); ib = np.asarray([p[1] for p in pairs], dtype=np.int64)
                d_parts.append(table.data[ia] - other.data[ib]); v_parts.append(table.sigma[ia] ** 2 + other.sigma[ib] ** 2)
                self.stats.append({**stat, "ia": ia, "ib": ib, "bins": int(ia.size)})
            else:
                d_parts.append(table.data); v_parts.append(table.sigma ** 2)
                self.stats.append({**stat, "bins": int(table.data.size)})
        self.d = np.concatenate(d_parts); self.v_data = np.concatenate(v_parts)
        self.cache: dict = {}

    def params_from_x(self, x) -> FlavorParams:
        values = flavor_to_dict(self.fixed)
        for name, xi in zip(self.free, x):
            values[name] = math.exp(float(xi))
        return FlavorParams(**values)

    def x_from_params(self, params: FlavorParams) -> np.ndarray:
        return np.asarray([math.log(getattr(params, name)) for name in self.free], dtype=float)

    def weights(self, params: FlavorParams):
        key = tuple(round(v, 12) for v in flavor_to_dict(params).values())
        if key in self.cache:
            return self.cache[key]
        log_ws = [self.s.log_weights(block, params) for block in self.s.blocks]
        out = ([FlavorSample.normalized(lw) for lw in log_ws], min(ess_fraction(lw) for lw in log_ws))
        self.cache[key] = out
        if len(self.cache) > 64:
            self.cache.pop(next(iter(self.cache)))
        return out

    def block_vector(self, k: int, wn: np.ndarray) -> np.ndarray:
        parts = []
        for stat in self.stats:
            m = self.s.species_prediction(self.s.blocks[k], wn, stat["species"])
            if stat.get("minus"):
                m2 = self.s.species_prediction(self.s.blocks[k], wn, stat["minus"])
                parts.append(m[stat["ia"]] - m2[stat["ib"]])
            else:
                parts.append(m)
        return np.concatenate(parts)

    def evaluate(self, x, blocks: Sequence[int] | None = None, *, details: bool = False):
        params = self.params_from_x(x)
        for xi in x:
            if not (LOG_BOUNDS[0] <= float(xi) <= LOG_BOUNDS[1]):
                return (float("inf"), None) if details else float("inf")
        weights, ess_min = self.weights(params)
        block_ids = list(range(self.s.n_blocks)) if blocks is None else list(blocks)
        vectors = np.asarray([self.block_vector(k, weights[k]) for k in block_ids])
        m = np.mean(vectors, axis=0)
        se = np.std(vectors, axis=0, ddof=1) / math.sqrt(vectors.shape[0]) if vectors.shape[0] > 1 else np.zeros_like(m)
        se = np.maximum(se, SE_FLOOR)
        total_var = self.v_data + se ** 2
        r = (self.d - m) / np.sqrt(total_var)
        design = m / np.sqrt(total_var)
        normal = float(design @ design) + 1.0 / NORM_PRIOR ** 2
        rhs = float(design @ r); eta = rhs / normal
        profiled = float(r @ r) - rhs * eta
        prior_terms = 0.0
        for name, (center, sigma_ln) in self.priors.items():
            if name in self.free:
                prior_terms += ((math.log(getattr(params, name)) - math.log(center)) / sigma_ln) ** 2
        value = profiled + prior_terms
        if ess_min < ESS_FLOOR:
            value += TRUST_PENALTY * (ESS_FLOOR - ess_min) / ESS_FLOOR
        if not details:
            return value
        return value, {"params": flavor_to_dict(params), "ess_min": ess_min, "eta_hat": eta, "minus2lnl_data": profiled,
                       "prior_terms": prior_terms, "m": m, "se": se, "pull": r, "in_trust_region": bool(ess_min >= ESS_FLOOR)}


def run_fit(objective: FlavorObjective, starts: Sequence[np.ndarray], blocks=None, maxfev: int = 3000) -> dict:
    from scipy.optimize import minimize
    best = None; records = []
    for x0 in starts:
        x0 = np.asarray(x0, dtype=float)
        if x0.size == 1:
            res = minimize(lambda x: objective.evaluate(x, blocks), x0, method="Powell", bounds=[LOG_BOUNDS],
                           options={"xtol": 1e-5, "ftol": 1e-8, "maxfev": maxfev})
        else:
            res = minimize(lambda x: objective.evaluate(x, blocks), x0, method="Powell", bounds=[LOG_BOUNDS] * x0.size,
                           options={"xtol": 1e-4, "ftol": 1e-7, "maxfev": maxfev})
        records.append({"x0": x0.tolist(), "x": [float(v) for v in res.x], "fun": float(res.fun), "nfev": int(res.nfev), "success": bool(res.success)})
        if best is None or res.fun < best.fun:
            best = res
    return {"x": np.asarray(best.x, dtype=float), "fun": float(best.fun), "converged": bool(best.success), "nfev": int(best.nfev),
            "message": str(best.message), "starts": records}


def hessian(objective: FlavorObjective, x_opt: np.ndarray, step: float = 0.02) -> dict:
    """Central-difference Hessian in the log parameters; errors of the
    parameters themselves by the delta method (``sigma_p = p sigma_ln``)."""
    n = x_opt.size
    h = np.zeros((n, n)); f0 = objective.evaluate(x_opt)
    for i in range(n):
        ei = np.zeros(n); ei[i] = step
        h[i, i] = (objective.evaluate(x_opt + ei) - 2.0 * f0 + objective.evaluate(x_opt - ei)) / step ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = step
            h[i, j] = h[j, i] = (objective.evaluate(x_opt + ei + ej) - objective.evaluate(x_opt + ei - ej)
                                 - objective.evaluate(x_opt - ei + ej) + objective.evaluate(x_opt - ei - ej)) / (4 * step * step)
    finite = bool(np.all(np.isfinite(h)))
    eig = np.linalg.eigvalsh(h) if finite else np.full(n, np.nan)
    pd = bool(finite and np.all(eig > 0))
    errors = None
    if pd:
        cov = 2.0 * np.linalg.inv(h)
        errors = {name: {"sigma_ln": float(math.sqrt(cov[i, i])), "sigma": float(math.exp(x_opt[i]) * math.sqrt(cov[i, i]))}
                  for i, name in enumerate(objective.free)}
    return {"names": list(objective.free), "step_ln": step, "matrix": h.tolist() if finite else None,
            "eigenvalues": [float(e) for e in eig] if finite else None, "positive_definite": pd, "quoted": pd, "errors": errors}


# ----------------------------------------------------------------------
# Held-out kaon moments (the p_T-integrated tables)
# ----------------------------------------------------------------------
def read_ptintegrated(path: Path) -> dict:
    """Columns ``xF F err dn/dxF err <pT> err <pT^2> err`` (errors in
    percent), skipping comment lines."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 9 or parts[0].startswith("#"):
            continue
        try:
            rows.append([float(v) for v in parts[:9]])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=float)
    return {"xf": arr[:, 0], "F": arr[:, 1], "F_err_pct": arr[:, 2], "dn_dxf": arr[:, 3], "dn_dxf_err_pct": arr[:, 4],
            "mean_pt": arr[:, 5], "mean_pt_err_pct": arr[:, 6], "mean_pt2": arr[:, 7], "mean_pt2_err_pct": arr[:, 8]}


def kaon_pt_moments(sample: FlavorSample, weights: list, species: str, table_path: Path) -> dict:
    """Model versus table ``<p_T>`` per x_F bin (the table's x_F values
    define the bins through midpoint edges; the paper window p_T cut of
    the tracks applies on the model side as recorded)."""
    data = read_ptintegrated(table_path)
    xf_values = data["xf"]
    edges = OB.sps_core.edges_from_centers(xf_values)
    num = np.zeros(edges.size - 1); den = np.zeros(edges.size - 1)
    for k, block in enumerate(sample.blocks):
        n, d = sample.mean_pt_by_xf(block, weights[k], species, edges)
        num += n; den += d
    model = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    rows = []
    for i, xf in enumerate(xf_values):
        if not np.isfinite(model[i]) or i >= model.size:
            continue
        sigma = data["mean_pt"][i] * data["mean_pt_err_pct"][i] / 100.0
        rows.append({"xf": float(xf), "data_mean_pt": float(data["mean_pt"][i]), "data_sigma": float(sigma),
                     "model_mean_pt": float(model[i]), "pull": float((model[i] - data["mean_pt"][i]) / sigma) if sigma > 0 else None})
    pulls = [r["pull"] for r in rows if r["pull"] is not None]
    return {"species": species, "rows": rows, "pull_rms": float(np.sqrt(np.mean(np.square(pulls)))) if pulls else None,
            "note": "table <pT> at fixed x_F; model: weighted track mean p_T per x_F bin over the recorded tracks (p_T > 0)"}
