#!/usr/bin/env python3
"""Generate HEART events at a fitted point and write them as JSON Lines.

The generator runs through the fit driver's own machinery (``run_engine``):
the run description configures the kernel, and the fit report supplies the
point, namely the mass-law shapes, the flavor weights, the leading-split
shares, the isovector fraction, the non-diffractive remnant mixture, and the
event-class fractions.  Each event is drawn in one class with the fitted
class fractions and generated on its own random substreams, so a given
(seed, event index) always gives the same event.  The format is described in
``docs/event-format.md`` of the published repository.

Example (from the repository root)::

    python helpers/benchmarking/heart_events.py \\
        --spec dev/mixing2006/config/runs/r141.json \\
        --report dev/00runs/r141_aligned_lines_2026-11-05/report.json \\
        --events 1000 --out heart_events.jsonl

The final state is the one the fit bins (``--final observed``, the default):
the decays the NA49 tables count are performed (``na49_mixing.counted_decays``).
``--final generator`` writes the cascade's own final state, in which every weak
decay is left undone.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FORMAT = "heart-events/1"
PARTICLE_COLUMNS = ("pdg", "E", "px", "py", "pz", "mass", "B", "Q", "S", "I3x2", "line")
#: event index offset between classes in the substream derivation, as the readout tools use it
CLASS_STRIDE = 1_000_000_000
HERE = Path(__file__).resolve().parent
_STATE: dict = {}


def _engine():
    if "E" not in _STATE:
        spec = importlib.util.spec_from_file_location("heart_run_engine", HERE / "run_engine.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["heart_run_engine"] = module
        spec.loader.exec_module(module)
        _STATE["E"] = module
    return _STATE["E"]


def fitted_point(E, fitted: dict) -> dict:
    """The generation point of a fit report's ``fitted`` block, in the
    engine's coordinates: the shape point (epsilon, b_sd, epsilon_nd and the
    spec's extra shape coordinates), the flavor point, the class fractions,
    the isovector fraction and the non-diffractive remnant weights."""
    shape = ((float(fitted["epsilon"]), float(fitted["b_sd"]), float(fitted["epsilon_nd"]))
             + tuple(float(fitted[n]) for n in E.EXTRA_SHAPE))
    flavor = tuple(float(fitted[n]) for n in E.FLAVOR_ORDER)
    fractions = {c: float(v) for c, v in fitted["class_fractions"].items() if float(v) > 0.0}
    total = sum(fractions.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"fit report: class fractions sum to {total}, not 1")
    return {"shape": shape, "flavor": flavor, "fractions": fractions,
            "f_isovector": float(fitted.get("f_isovector", 0.0)),
            "remnant_weights": {k: float(v) for k, v in fitted["remnant_weights"].items()}}


def setup(spec_path: str, report_path: str) -> dict:
    """Configure the engine from a run description and read the fit report's
    point; returns the context the generation needs."""
    E = _engine()
    spec = E.configure(spec_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    point = fitted_point(E, report["fitted"])
    catalog, fams = E._load_physics(E.FLOOR_GRID[0], point["shape"])
    params = {}
    for cls in point["fractions"]:
        p = E._params_at(cls, point["shape"], point["flavor"])
        p = dataclasses.replace(p, isovector_fraction=point["f_isovector"])
        if cls in E.ND_CLASSES:
            p = dataclasses.replace(p, remnant_weights=dict(point["remnant_weights"]))
        params[cls] = p
    _STATE.update(spec=spec, point=point, catalog=catalog, fams=fams, params=params,
                  spec_path=str(spec_path), report_path=str(report_path))
    return _STATE


def header(seed: int, n_events: int, final: str) -> dict:
    E, point = _STATE["E"], _STATE["point"]
    return {
        "format": FORMAT,
        "spec": _STATE["spec_path"], "report": _STATE["report_path"], "run_id": _STATE["spec"]["identity"].get("run_id"),
        "reaction": "p+p", "beam_momentum_gev_c": 158.0, "sqrt_s_gev": float(E.SQRT_S_CM),
        "frame": "lab (target at rest, beam along +z)", "units": "GeV",
        "seed": int(seed), "events": int(n_events), "final_state": final,
        "class_fractions": point["fractions"], "isovector_fraction": point["f_isovector"],
        "particle_columns": list(PARTICLE_COLUMNS),
    }


def _class_of(seed: int, index: int, classes: list, cumulative: np.ndarray) -> str:
    u = float(np.random.default_rng([int(seed), int(index), 1]).random())
    k = int(np.searchsorted(cumulative, u, side="right"))
    return classes[min(k, len(classes) - 1)]


def generate(seed: int, start: int, stop: int, final: str) -> list[str]:
    """Events ``start`` to ``stop - 1`` of the seed, each as one JSON line."""
    E, point = _STATE["E"], _STATE["point"]
    catalog, fams = _STATE["catalog"], _STATE["fams"]
    classes = sorted(point["fractions"], key=E.CLASS_ORDER.index)
    cumulative = np.cumsum([point["fractions"][c] for c in classes])
    labels = ("topology_selection", "spectral_state", "exchange_kinematics", "decay_channel", "decay_kinematics")
    lines = []
    for i in range(int(start), int(stop)):
        cls = _class_of(seed, i, classes, cumulative)
        subs = E.derive_substreams(int(seed), 0, CLASS_STRIDE * E.CLASS_ORDER.index(cls) + i)
        rngs = {f"rng_{lab}": np.random.default_rng(subs.stream_for(lab)) for lab in labels}
        event = E.run_topology(E.RUNNER_CLASS.get(cls, cls), _STATE["params"][cls], catalog, fams, **rngs)
        hadrons = E.observed_final_state(event, catalog) if final == "observed" else event.final_state
        rows = []
        for h in hadrons:
            qn = tuple(int(x) for x in h.qn)
            rows.append([None if h.pdg_id is None else int(h.pdg_id), *(float(x) for x in h.four_momentum),
                         float(h.mass), *qn, str(h.line)])
        lines.append(json.dumps({"event": i, "class": E.RUNNER_CLASS.get(cls, cls), "particles": rows},
                                separators=(",", ":")))
    return lines


def _chunk(args) -> list[str]:
    spec_path, report_path, seed, start, stop, final = args
    if "params" not in _STATE:
        setup(spec_path, report_path)
    return generate(seed, start, stop, final)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, help="run description JSON (e.g. dev/mixing2006/config/runs/r141.json)")
    ap.add_argument("--report", required=True, help="the fit's report.json (its 'fitted' block is the point)")
    ap.add_argument("--events", type=int, default=1000, help="number of inelastic events")
    ap.add_argument("--seed", type=int, default=1, help="root seed; (seed, event index) fixes each event")
    ap.add_argument("--first", type=int, default=0, help="index of the first event (to continue a sample)")
    ap.add_argument("--final", choices=("observed", "generator"), default="observed",
                    help="observed: the decays the NA49 tables count performed (the fit's final state); "
                         "generator: the cascade's final state, weak decays undone")
    ap.add_argument("--workers", type=int, default=1, help="worker processes (events are split into equal chunks)")
    ap.add_argument("--out", required=True, help="output JSON Lines file")
    a = ap.parse_args(argv)
    if a.events < 1 or a.workers < 1:
        ap.error("--events and --workers must be positive")
    setup(a.spec, a.report)
    t0 = time.perf_counter()
    bounds = np.linspace(a.first, a.first + a.events, a.workers + 1).astype(int)
    tasks = [(a.spec, a.report, a.seed, int(lo), int(hi), a.final) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
    with open(a.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(header(a.seed, a.events, a.final)) + "\n")
        if a.workers == 1:
            for line in generate(a.seed, a.first, a.first + a.events, a.final):
                out.write(line + "\n")
        else:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                for lines in pool.map(_chunk, tasks):
                    out.write("\n".join(lines) + "\n")
    print(f"wrote {a.events} events to {a.out} in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
