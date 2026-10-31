"""Capture the Stage-0 golden baseline fixtures (card TASK-2026-08-31-P1, step 1).

The fixtures anchor two things (card TASK-2026-08-31-resonance-decay-fit-workflow,
Stage 0 and Acceptance item 9):

- the Stage-0 fit quality reference of the staged resonance-decay fit workflow;
- the inactive-path regression of the constraint machinery
  (``tests/test_resonance_constraint_workflow.py``).

They are captured from the *unmodified resonance-template generator*: this
script must be run on the tree BEFORE the P1 constraint machinery is applied
to ``na49_mixing/event.py`` / ``decays.py``.  The artifacts are exactly the
production component-cache artifacts of ``helpers/benchmarking/benchmark_sps_species.py``
(through ``_build_resonance_component_cache_one``) at smoke statistics, plus
one summary JSON per species.

Regeneration:

    cd dev/mixing2006 && python tests/fixtures/make_golden_fixtures.py

The script is deterministic given the salt and module sources in force; the
captured salt is recorded in each summary JSON.  A decay-affecting change
bumps ``RESONANCE_DECAY_LIB_VERSION`` per
``_knowledge/operations/mixing2006-conventions.md`` and the fixtures are then
regenerated deliberately (never migrated in place).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent           # tests/fixtures
PKG_ROOT = HERE.parents[1]                       # dev/mixing2006
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from helpers.benchmarking.benchmark_sps_species import (  # noqa: E402
    RESONANCE_DECAY_LIB_VERSION,
    _build_resonance_component_cache_one,
    _component_seed,
    _resonance_spec_key,
    _resonance_specs_for_particle,
)
from na49_mixing.parallel import (  # noqa: E402
    GenerationConfig,
    generate_tracks_chunked,
)

# Provenance constants: shared by the capture run and the regression test.
BASE_SEED = 20260831
EVENTS_PER_COMPONENT = 200
MAX_DECAY_PASSES = 8
BAND_SEED_STRIDE = 7919
BAND_SEEDS = 7
BAND_SIGMA_MULTIPLIER = 6.0

SPECIES = {"pi_minus": -211, "pi_plus": 211}

COMPONENT_DIRNAME = {"pi_minus": "components/pi_minus", "pi_plus": "components/pi_plus"}
SUMMARY_FILENAME = {
    "pi_minus": "golden_stage0_summary_pi_minus.json",
    "pi_plus": "golden_stage0_summary_pi_plus.json",
}


def build_component_config(spec, component_seed: int, n_events: int) -> GenerationConfig:
    """GenerationConfig mirroring the production component-cache path.

    Mirrors the config assembled by ``_build_resonance_component_cache_one``
    in ``helpers/benchmarking/benchmark_sps_species.py``: beam 158 GeV/c,
    both fragmentation sides, soft decay factory, benchmark antiproton
    injection, inclusive weak feed, no soft overrides.
    """
    return GenerationConfig(
        gen_file=str(spec.gen_file),
        n_events=int(n_events),
        seed=int(component_seed),
        beam_momentum=158.0,
        max_decay_passes=MAX_DECAY_PASSES,
        fragmentation_mode="both",
        decay_files=tuple(str(p) for p in spec.decay_files),
        use_kaon_decay_factory=bool(spec.use_kaon_decay_factory),
        use_soft_decay_factory=bool(spec.use_soft_decay_factory),
        soft_decay_overrides=(),
        inject_antibaryon_resonances=True,
        weak_feed_policy="inclusive_weak",
    )


def build_component_tracks(spec, component_seed: int, n_events: int):
    """Generate final-state tracks for one component, in-process."""
    cfg = build_component_config(spec, component_seed, n_events)
    return generate_tracks_chunked(cfg, n_workers=1, n_chunks=1)


def _std(values: list) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unavailable"


def main() -> int:
    outer_root = PKG_ROOT.parents[1]

    baseline_note = (
        "Working tree as of 2026-09-01: nested dev/mixing2006 HEAD plus its "
        "uncommitted in-flight source-excitation work (na49_mixing/__init__.py "
        "export additions and the untracked quantum_numbers / "
        "excitation_kinematics / exchange_model / continuum_channels / "
        "source_event / event_builder_pp / event_builder_pA / staged_evaluation "
        "modules). Those modules do not participate in the resonance-template "
        "generator path exercised here; the outer tree carries mode-only "
        "(755->644) helper changes. No fully clean checkout exists in this "
        "workspace."
    )

    summary_common = {
        "schema_version": 1,
        "stage": "stage0_golden_baseline",
        "salt_in_force": RESONANCE_DECAY_LIB_VERSION,
        "base_seed": BASE_SEED,
        "events_per_component": EVENTS_PER_COMPONENT,
        "band_seeds": BAND_SEEDS,
        "band_seed_stride": BAND_SEED_STRIDE,
        "band_sigma_multiplier": BAND_SIGMA_MULTIPLIER,
        "max_decay_passes": MAX_DECAY_PASSES,
        "fragmentation_mode": "both",
        "weak_feed_policy": "inclusive_weak",
        "beam_pdg": 2212,
        "target_pdg": 2212,
        "beam_momentum_gev_c": 158.0,
        "nested_head": _git_head(PKG_ROOT),
        "outer_head": _git_head(outer_root),
        "baseline_note": baseline_note,
    }

    written = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for species_name, species_pdg in SPECIES.items():
            specs = _resonance_specs_for_particle(species_pdg, PKG_ROOT)
            components: dict = {}
            for spec in specs:
                spec_key = _resonance_spec_key(spec, PKG_ROOT)
                component_seed = _component_seed(BASE_SEED, spec_key)

                task = {
                    "label": spec.label,
                    "gen_file": str(spec.gen_file),
                    "decay_files": [str(p) for p in spec.decay_files],
                    "use_kaon_decay_factory": bool(spec.use_kaon_decay_factory),
                    "use_soft_decay_factory": bool(spec.use_soft_decay_factory),
                    "cache_file": str(tmp_dir / f"{spec.label}.npz"),
                    "component_seed": component_seed,
                    "spec_key": spec_key,
                    "soft_decay_overrides": {},
                    "weak_feed_policy": "inclusive_weak",
                    "beam_pdg": 2212,
                    "target_pdg": 2212,
                    "nucleus": None,
                    "target_wound_range": None,
                }
                built = _build_resonance_component_cache_one(
                    mixing2006_dir=str(PKG_ROOT),
                    task=task,
                    n_events_per_res=EVENTS_PER_COMPONENT,
                    max_decay_passes=MAX_DECAY_PASSES,
                    fragmentation_mode="both",
                    chunk_workers=1,
                )
                if not built.get("ok"):
                    raise SystemExit(
                        f"golden fixture build failed for {spec.label}: {built}"
                    )

                cache_path = tmp_dir / f"{spec.label}.npz"
                with np.load(cache_path, allow_pickle=False) as z:
                    pdg = np.asarray(z["pdg"])
                    pt = np.asarray(z["pt"], dtype=np.float64)
                    xf = np.asarray(z["xf"], dtype=np.float64)
                    meta = {
                        "meta_decay_lib_version": str(z["meta_decay_lib_version"]),
                        "meta_decay_lib_fingerprint": str(z["meta_decay_lib_fingerprint"]),
                        "meta_n_decayed_parents": int(z["meta_n_decayed_parents"]),
                        "meta_n_decay_kinematic_failures": int(z["meta_n_decay_kinematic_failures"]),
                        "meta_n_primary_generated": int(z["meta_n_primary_generated"]),
                        "meta_n_primary_decayed": int(z["meta_n_primary_decayed"]),
                        "meta_mass_law_json": str(z["meta_mass_law_json"]),
                    }

                # target-species summary statistics from the golden run
                target_mask = pdg == species_pdg
                n_target = int(np.count_nonzero(target_mask))
                pdg_counts = {
                    str(int(p)): int(c)
                    for p, c in zip(*np.unique(pdg, return_counts=True))
                }
                mean_pt_gold = float(pt[target_mask].mean()) if n_target else 0.0
                mean_xf_gold = float(xf[target_mask].mean()) if n_target else 0.0

                # pre-registered seed-variation band: golden + BAND_SEEDS replicates
                target_counts = [n_target]
                mean_pts = [mean_pt_gold]
                mean_xfs = [mean_xf_gold]
                for k in range(1, BAND_SEEDS + 1):
                    seed_k = component_seed + BAND_SEED_STRIDE * k
                    tracks_k = build_component_tracks(spec, seed_k, EVENTS_PER_COMPONENT)
                    pdg_k = np.asarray(tracks_k["pdg"])
                    mask_k = pdg_k == species_pdg
                    n_k = int(np.count_nonzero(mask_k))
                    target_counts.append(n_k)
                    pt_k = np.asarray(tracks_k["pt"], dtype=np.float64)
                    xf_k = np.asarray(tracks_k["xf"], dtype=np.float64)
                    mean_pts.append(float(pt_k[mask_k].mean()) if n_k else 0.0)
                    mean_xfs.append(float(xf_k[mask_k].mean()) if n_k else 0.0)

                sigma_count = _std(target_counts)
                sigma_mean_pt = _std(mean_pts)
                sigma_mean_xf = _std(mean_xfs)
                floor_count = float(n_target) ** 0.5 if n_target else 0.0
                floor_mean_pt = (
                    float(pt[target_mask].std()) / (float(n_target) ** 0.5)
                    if n_target else 0.0
                )
                floor_mean_xf = (
                    float(xf[target_mask].std()) / (float(n_target) ** 0.5)
                    if n_target else 0.0
                )

                band = {
                    "n_target_sigma_seed": sigma_count,
                    "n_target_sigma_used": max(sigma_count, floor_count),
                    "mean_pt_sigma_seed": sigma_mean_pt,
                    "mean_pt_sigma_used": max(sigma_mean_pt, floor_mean_pt),
                    "mean_xf_sigma_seed": sigma_mean_xf,
                    "mean_xf_sigma_used": max(sigma_mean_xf, floor_mean_xf),
                    "n_sigma_multiplier": BAND_SIGMA_MULTIPLIER,
                }

                # store the production-format cache file as the fixture
                dest = HERE / COMPONENT_DIRNAME[species_name]
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_path, dest / f"{spec.label}.npz")

                components[spec.label] = {
                    "spec_key": spec_key,
                    "gen_file": str(spec.gen_file),
                    "component_seed": component_seed,
                    "n_events": EVENTS_PER_COMPONENT,
                    "n_tracks": int(pdg.shape[0]),
                    "n_target_species": n_target,
                    "pdg_counts": pdg_counts,
                    "n_decayed_parents": meta["meta_n_decayed_parents"],
                    "n_decay_kinematic_failures": meta["meta_n_decay_kinematic_failures"],
                    "n_primary_generated": meta["meta_n_primary_generated"],
                    "n_primary_decayed": meta["meta_n_primary_decayed"],
                    "mass_law": json.loads(meta["meta_mass_law_json"]),
                    "decay_lib_fingerprint": meta["meta_decay_lib_fingerprint"],
                    "mc_band": band,
                }

            summary = dict(summary_common)
            summary["species"] = species_name
            summary["species_pdg"] = species_pdg
            summary["component_labels"] = [s.label for s in specs]
            summary["components"] = components
            summary["totals"] = {
                "n_tracks": sum(c["n_tracks"] for c in components.values()),
                "n_target_species": sum(c["n_target_species"] for c in components.values()),
                "n_events": sum(c["n_events"] for c in components.values()),
            }

            out_path = HERE / SUMMARY_FILENAME[species_name]
            out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            written.append(str(out_path))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
