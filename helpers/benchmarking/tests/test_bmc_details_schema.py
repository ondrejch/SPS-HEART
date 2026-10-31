"""Schema contract tests for ``multispecies_constrained_details.npz``.

The constrained-fit benchmark producer stores string metadata as Unicode-dtype
arrays plus an integer ``schema_version`` marker, so benchmark consumers load
result directories pickle-free through the shared validator in
:mod:`helpers.benchmarking.sps_core`. The payloads here are assembled through
the producer's own payload builder
(``benchmark_multispecies_constrained.build_details_npz_payload``), then
mutated to cover each rejection mode: object-dtype metadata (the pre-P3
on-disk format), missing required keys, length mismatches against the
companion summary JSON, and wrong schema versions. Every rejection must raise
``ValueError`` naming the file path and the offending key. Older cached result
directories written with object-dtype metadata are intentionally unloadable;
the expected remediation is regeneration, never pickle loading.

A source-scan section walks the production sources of BOTH helper trees
(monorepo ``helpers/benchmarking`` and vendored
``dev/mixing2006/helpers/benchmarking``, excluding tests/) and fails whenever
a direct ``np.load`` opens this specifically named artifact; unrelated NPZ
formats (``{out_prefix}_details.npz``, gateway pythia details) keep calling
``np.load`` directly.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

import helpers.benchmarking.bootstrap_multispecies_robustness as bootstrap_mod
import helpers.benchmarking.sps_core as sps_core
import helpers.benchmarking.study_component_identifiability as ident_mod
from helpers.benchmarking.sps_core import (
    DETAILS_NPZ_REQUIRED_STRING_KEYS,
    DETAILS_NPZ_SCHEMA_VERSION,
    load_multispecies_details_npz,
)
from helpers.benchmarking.benchmark_multispecies_constrained import (
    SpeciesData,
    build_details_npz_payload,
)

DETAILS_NAME = "multispecies_constrained_details.npz"
SUMMARY_NAME = "multispecies_constrained_summary.json"

COMPONENT_KEYS = ["rho0", "omega", "phi"]
COMPONENT_WEIGHTS = [0.5, 0.3, 0.2]

PIMINUS_TAG = "pim"
KAPLUS_TAG = "kp"
# Per-species fixture values: y and resonance prediction levels differ by a
# constant offset so assertions can tell which array fed which slot.
SPECIES_SPEC = {
    # key: (tag, pdg, mass, y_values, pred_res_values, n_points)
    "pi-": (PIMINUS_TAG, -211, 0.13957, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    "k+": (KAPLUS_TAG, 321, 0.493677, [10.0, 20.0], [11.0, 12.0]),
}


def _species_data(key: str) -> SpeciesData:
    tag, _pdg, _mass, y_vals, res_vals = SPECIES_SPEC[key]
    n = len(y_vals)
    return SpeciesData(
        key=key,
        name=key,
        tag=tag,
        pdg=int(_pdg),
        mass_gev=float(_mass),
        na49_path=None,  # type: ignore[arg-type]
        xf=np.linspace(0.05, 0.15, n),
        pt=np.full(n, 0.20),
        y=np.asarray(y_vals, dtype=float),
        sigma=np.asarray([0.1 + 0.01 * i for i in range(n)], dtype=float),
        xedges=np.asarray([0.0, 0.2, 0.4], dtype=float),
        pedges=np.asarray([0.1, 0.5], dtype=float),
        ix_data=np.arange(n, dtype=np.int64) % 2,
        ip_data=np.zeros(n, dtype=np.int64),
        train_idx=np.arange(n, dtype=np.int64),
        test_idx=np.arange(n, dtype=np.int64),
        matrix=np.zeros((n, len(COMPONENT_KEYS)), dtype=float),
        pred_res=np.asarray(res_vals, dtype=float),
        pred_res_sigma=np.full(n, 0.05),
    )


def _external_maps() -> tuple:
    ext_models = ("PYTHIA8_external", "EPOS_external", "UrQMD_external")
    ext_pred: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in ext_models}
    ext_sigma: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in ext_models}
    level = 1.0
    for model in ext_models:
        for key in SPECIES_SPEC:
            n = len(SPECIES_SPEC[key][3])
            ext_pred[model][key] = np.full(n, level)
            ext_sigma[model][key] = np.full(n, 0.02 * level)
            level += 1.0
    return ext_pred, ext_sigma


def _payload() -> Dict[str, np.ndarray]:
    species_map = {key: _species_data(key) for key in SPECIES_SPEC}
    ext_pred, ext_sigma = _external_maps()
    return build_details_npz_payload(
        component_weights=COMPONENT_WEIGHTS,
        component_keys=list(COMPONENT_KEYS),
        component_labels=list(COMPONENT_KEYS),
        species_data_by_key=species_map,
        pred_full_stack=np.concatenate([sd.pred_res for sd in species_map.values()]),
        y_full_stack=np.concatenate([sd.y for sd in species_map.values()]),
        sigma_full_stack=np.concatenate([sd.sigma for sd in species_map.values()]),
        n_effective=3,
        prior_report={"nstar_smooth": {"ok": False}, "prior_total_rows": 0},
        fit_solver={"method": "scipy.optimize.nnls", "rnorm": 0.25},
        fit_weight_mode="relative",
        fit_objective="nnls_weighted",
        weak_feed_policy="subtract",
        histogram_convention="invariant",
        external_pred_by_model=ext_pred,
        external_sigma_by_model=ext_sigma,
    )


def _summary() -> Dict[str, object]:
    return {
        "weak_feed_policy": "subtract",
        "histogram_convention": "invariant",
        "species": {
            key: {
                "name": key,
                "tag": spec[0],
                "pdg": int(spec[1]),
                "na49_table": "in-memory-fixture",
                "n_points": len(spec[3]),
                "full_points": len(spec[3]),
            }
            for key, spec in SPECIES_SPEC.items()
        },
        "resonance": {
            "n_components": len(COMPONENT_KEYS),
            "component_keys": list(COMPONENT_KEYS),
            "component_labels": list(COMPONENT_KEYS),
            "weights_by_key": dict(zip(COMPONENT_KEYS, COMPONENT_WEIGHTS)),
            "global_metrics": {"n_params": len(COMPONENT_KEYS), "chi2_dof": 0.9},
            "priors": {},
        },
        "fit_weighting": {"mode": "relative"},
        "fit_solver": {"method": "scipy.optimize.nnls"},
    }


def _write_run_dir(
    tmp_path: Path,
    *,
    mutate_payload: Callable[[Dict[str, np.ndarray]], None] | None = None,
    mutate_summary: Callable[[Dict[str, object]], None] | None = None,
) -> Path:
    run_dir = tmp_path / "constrained_refit_p3_fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _payload()
    if mutate_payload is not None:
        mutate_payload(payload)
    np.savez_compressed(run_dir / DETAILS_NAME, **payload)
    sdict = _summary()
    if mutate_summary is not None:
        mutate_summary(sdict)
    (run_dir / SUMMARY_NAME).write_text(json.dumps(sdict), encoding="utf-8")
    return run_dir


def test_producer_payload_uses_unicode_metadata_and_schema_marker() -> None:
    payload = _payload()
    assert payload["schema_version"].dtype == np.int64
    assert int(payload["schema_version"]) == DETAILS_NPZ_SCHEMA_VERSION
    for key in DETAILS_NPZ_REQUIRED_STRING_KEYS:
        assert key in payload
        assert payload[key].dtype.kind == "U", f"{key} must be Unicode dtype"
    for key in (
        "component_weights",
        "resonance_global_full_pred",
        "resonance_global_full_y",
        "resonance_global_full_sigma",
    ):
        assert payload[key].dtype.kind == "f", f"{key} must be floating point"
    assert payload["n_effective"].dtype == np.int64


def test_fresh_npz_loads_pickle_free_end_to_end_through_bootstrap(tmp_path: Path) -> None:
    """A freshly generated details NPZ loads end-to-end with allow_pickle=False."""
    run_dir = _write_run_dir(tmp_path)
    rp = bootstrap_mod._load_run(run_dir=run_dir, label="p3fix", species_csv="")
    assert rp.label == "p3fix"
    assert rp.run_dir == run_dir
    assert set(rp.species_arrays) == set(SPECIES_SPEC)

    pi = rp.species_arrays["pi-"]
    assert np.allclose(pi.y, SPECIES_SPEC["pi-"][3])
    assert np.allclose(pi.sigma, np.asarray([0.1, 0.11, 0.12]))
    assert set(pi.pred_by_model) == {
        "Resonance_NNLS_shared",
        "PYTHIA8_external",
        "EPOS_external",
        "UrQMD_external",
    }
    assert np.allclose(pi.pred_by_model["Resonance_NNLS_shared"], SPECIES_SPEC["pi-"][4])

    kp = rp.species_arrays["k+"]
    assert np.allclose(kp.y, SPECIES_SPEC["k+"][3])

    # global metrics wiring: resonance fit uses n_params from the summary,
    # external models carry one scale parameter per selected species.
    assert rp.n_params_by_model["Resonance_NNLS_shared"] == len(COMPONENT_KEYS)
    for model in ("PYTHIA8_external", "EPOS_external", "UrQMD_external"):
        assert rp.n_params_by_model[model] == len(SPECIES_SPEC)


def test_species_subset_selection(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)
    rp = bootstrap_mod._load_run(run_dir=run_dir, label="sub", species_csv="k+")
    assert set(rp.species_arrays) == {"k+"}
    with pytest.raises(ValueError, match="No requested species found"):
        bootstrap_mod._load_run(run_dir=run_dir, label="bad", species_csv="xi_minus")


def test_object_dtype_metadata_rejected_with_remediation(tmp_path: Path) -> None:
    """Object-dtype payloads (pre-P3 cached runs) name file+run-dir and remediation."""

    def make_legacy(payload: Dict[str, np.ndarray]) -> None:
        payload["component_keys"] = np.asarray(["rho0", "omega", "phi"], dtype=object)

    run_dir = _write_run_dir(tmp_path, mutate_payload=make_legacy)
    details_path = run_dir / DETAILS_NAME
    with pytest.raises(ValueError) as errinfo:
        with load_multispecies_details_npz(details_path, summary_payload=_summary()):
            pass  # pragma: no cover - entering the context must fail above
    msg = str(errinfo.value)
    assert str(details_path) in msg
    assert str(run_dir) in msg
    assert "regenerate with the current producer" in msg


def test_fully_legacy_artifact_rejected_at_load(tmp_path: Path) -> None:
    """An artifact written exactly like pre-P3 caches names the file and remediation."""

    def make_legacy(payload: Dict[str, np.ndarray]) -> None:
        for key in DETAILS_NPZ_REQUIRED_STRING_KEYS:
            values = [str(x) for x in np.asarray(payload[key]).reshape(-1).tolist()]
            if not values:
                values = [str(np.asarray(payload[key]))]
            payload[key] = np.asarray(values, dtype=object)
        payload.pop("schema_version")

    run_dir = _write_run_dir(tmp_path, mutate_payload=make_legacy)
    details_path = run_dir / DETAILS_NAME
    with pytest.raises(ValueError) as errinfo:
        bootstrap_mod._load_run(run_dir=run_dir, label="legacy", species_csv="")
    msg = str(errinfo.value)
    assert str(details_path) in msg
    assert "schema_version" in msg
    assert "missing required dataset" in msg
    assert "regenerate with the current producer" in msg


def test_missing_required_key_rejected(tmp_path: Path) -> None:
    def drop_key(payload: Dict[str, np.ndarray]) -> None:
        payload.pop("meta_histogram_convention")

    run_dir = _write_run_dir(tmp_path, mutate_payload=drop_key)
    details_path = run_dir / DETAILS_NAME
    with pytest.raises(ValueError) as errinfo:
        bootstrap_mod._load_run(run_dir=run_dir, label="x", species_csv="")
    msg = str(errinfo.value)
    assert str(details_path) in msg
    assert "meta_histogram_convention" in msg
    assert "missing required dataset" in msg


def _mutate_truncate_kp_y(payload: Dict[str, np.ndarray]) -> None:
    payload["kp_y"] = payload["kp_y"][:1]


def _mutate_shorten_component_labels(payload: Dict[str, np.ndarray]) -> None:
    payload["component_labels"] = np.asarray(["rho0", "omega"], dtype=str)


def _mutate_extend_global_y(payload: Dict[str, np.ndarray]) -> None:
    payload["resonance_global_full_y"] = np.concatenate([payload["resonance_global_full_y"], [9.9]])


@pytest.mark.parametrize(
    "mutator,offending_key,detail",
    [
        (_mutate_truncate_kp_y, "kp_y", "n_points=2"),
        (_mutate_shorten_component_labels, "component_weights", "(3/2/3)"),
        (_mutate_extend_global_y, "resonance_global_full_pred", "length mismatch"),
    ],
)
def test_length_mismatch_against_summary_or_internals_rejected(
    tmp_path: Path, mutator: Callable[[Dict[str, np.ndarray]], None], offending_key: str, detail: str
) -> None:
    run_dir = _write_run_dir(tmp_path, mutate_payload=mutator)
    details_path = run_dir / DETAILS_NAME
    with pytest.raises(ValueError) as errinfo:
        bootstrap_mod._load_run(run_dir=run_dir, label="x", species_csv="")
    msg = str(errinfo.value)
    assert str(details_path) in msg
    assert offending_key in msg
    assert detail in msg


def test_wrong_schema_version_rejected(tmp_path: Path) -> None:
    def bump_version(payload: Dict[str, np.ndarray]) -> None:
        payload["schema_version"] = np.asarray(DETAILS_NPZ_SCHEMA_VERSION + 7, dtype=np.int64)

    run_dir = _write_run_dir(tmp_path, mutate_payload=bump_version)
    details_path = run_dir / DETAILS_NAME
    with pytest.raises(ValueError) as errinfo:
        bootstrap_mod._load_run(run_dir=run_dir, label="x", species_csv="")
    msg = str(errinfo.value)
    assert str(details_path) in msg
    assert "schema_version" in msg
    assert f"expected {DETAILS_NPZ_SCHEMA_VERSION}" in msg
    assert f"found {DETAILS_NPZ_SCHEMA_VERSION + 7}" in msg


def test_validator_passes_without_summary_payload(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)
    with np.load(run_dir / DETAILS_NAME, allow_pickle=False) as zarr:
        sps_core.validate_multispecies_details_npz(zarr, details_path=run_dir / DETAILS_NAME)


def _summary_without_kplus() -> Dict[str, object]:
    mutated = _summary()
    raw_species = mutated.get("species")
    species: Dict[str, object] = dict(raw_species) if isinstance(raw_species, dict) else {}
    del species["k+"]
    mutated["species"] = species
    return mutated


def test_validator_detects_species_set_mismatch_against_summary() -> None:
    mutated = _summary_without_kplus()
    with pytest.raises(ValueError) as errinfo:
        sps_core.validate_multispecies_details_npz(
            _payload(), details_path=SUMMARY_NAME, summary_payload=mutated
        )
    msg = str(errinfo.value)
    assert SUMMARY_NAME in msg
    assert "species_keys" in msg
    assert "does not match summary JSON" in msg
    assert "'k+'" in msg and "'pi-'" in msg


def test_validator_detects_global_stack_mismatch_vs_summary_point_counts() -> None:
    """An extra summary species entry shifts the expected stack length to 6."""
    payload = _payload()
    tags = [str(x) for x in payload["species_keys"].tolist()]
    payload["species_keys"] = np.asarray(tags + ["fake_extra"], dtype=str)
    one = np.asarray([1.0])
    for suffix in ("xf", "pt", "y", "sigma", "pred_res", "pred_res_sigma"):
        payload[f"fx_{suffix}"] = one.copy()
    payload["fx_full_idx"] = np.zeros(1, dtype=np.int64)
    mutated = _summary()
    mutated["species"] = {
        **dict(mutated["species"]),  # type: ignore[arg-type]
        "fake_extra": {
            "name": "fake_extra",
            "tag": "fx",
            "pdg": 999,
            "na49_table": "in-memory-fixture",
            "n_points": 1,
            "full_points": 1,
        },
    }
    with pytest.raises(ValueError) as errinfo:
        sps_core.validate_multispecies_details_npz(payload, details_path=SUMMARY_NAME, summary_payload=mutated)
    msg = str(errinfo.value)
    assert "resonance_global_full_pred" in msg
    assert "holds 5 points" in msg


def test_identifiability_consumer_uses_shared_picklefree_loader() -> None:
    src_main = inspect.getsource(ident_mod.main)
    assert "load_multispecies_details_npz(" in src_main
    src_all = inspect.getsource(ident_mod)
    assert "allow_pickle=True" not in src_all


# --- consumer scan: direct np.load of the named details artifact --------------

_MONO_HELPERS_ROOT = Path(__file__).resolve().parents[1]
_MIX_HELPERS_ROOT = Path(__file__).resolve().parents[3] / "dev" / "mixing2006" / "helpers" / "benchmarking"


def _iter_production_sources() -> List[Tuple[str, str]]:
    """Source text of every production helper script in BOTH trees.

    Tests directories are excluded: tests legitimately assemble artifacts
    with np.load. Layout-only duplicates are scanned on both sides so a
    regression introduced in one tree fails even before sync normalization.
    """
    out: List[Tuple[str, str]] = []
    for root in (_MONO_HELPERS_ROOT, _MIX_HELPERS_ROOT):
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel.startswith("tests/") or "__pycache__" in path.parts:
                continue
            out.append((f"{root.parent.name}/{root.name}/{rel}", path.read_text(encoding="utf-8")))
    return out


def _is_np_load_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "load"
        and isinstance(func.value, ast.Name)
        and func.value.id in {"np", "numpy"}
    )


def _mentions_details_name(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Constant) and isinstance(sub.value, str) and DETAILS_NAME in sub.value
        for sub in ast.walk(node)
    )


def _scope_nodes(scope: ast.AST) -> List[ast.AST]:
    """Nodes of this scope only; nested function bodies form their own scopes."""
    out: List[ast.AST] = []
    stack = [scope]
    while stack:
        cur = stack.pop()
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, ast.Lambda):
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # scanned separately; decorators/defaults stay top-level here
            out.append(child)
            stack.append(child)
    return out


def _direct_details_load_lines(source: str) -> List[int]:
    """Lines of ``np.load(...)`` calls that open the named details artifact.

    A call is flagged when its argument textually carries the literal artifact
    name or when it is a plain name bound anywhere in the same enclosing scope
    to an expression containing that literal (the historical offender shape:
    ``details_path = run_dir / "multispecies_constrained_details.npz"`` then
    ``np.load(details_path, ...)``). Bindings inside nested functions stay
    local, function parameters carry no bindings, and unrelated NPZ formats
    never mention the literal, so cache loads and other details artifacts pass.
    """
    tree = ast.parse(source)
    hits: List[int] = []

    def scan_scope(scope: ast.AST) -> None:
        nodes = _scope_nodes(scope)
        env = {
            target.id
            for node in nodes
            if isinstance(node, ast.Assign) and _mentions_details_name(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in nodes:
            if isinstance(node, ast.AnnAssign) and node.value is not None and _mentions_details_name(node.value):
                if isinstance(node.target, ast.Name):
                    env.add(node.target.id)
        for node in nodes:
            if not (isinstance(node, ast.Call) and _is_np_load_call(node)):
                continue
            arg_exprs = list(node.args) + [kwarg.value for kwarg in node.keywords]
            flagged = any(
                (isinstance(arg, ast.Name) and arg.id in env) or _mentions_details_name(arg)
                for arg in arg_exprs
            )
            if flagged:
                hits.append(node.lineno)

    scan_scope(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_scope(node)
    return hits


def test_no_direct_details_artifact_load_in_production_sources() -> None:
    """Every production consumer opens the named artifact via the shared loader."""
    violations = [
        f"{label}:line {line}"
        for label, source in _iter_production_sources()
        for line in _direct_details_load_lines(source)
    ]
    assert violations == [], f"direct np.load of {DETAILS_NAME} found in: {violations}"


def test_consumer_scan_flags_direct_named_load_and_allows_unrelated_formats() -> None:
    """Detector self-check on inline snippets: offender flagged, legit loads untouched."""
    direct_bound_name = "\n".join(
        [
            "from pathlib import Path",
            'details_path = Path("run") / "multispecies_constrained_details.npz"',
            "z = np.load(details_path, allow_pickle=False)",
        ]
    )
    assert _direct_details_load_lines(direct_bound_name) == [3]

    direct_literal_with_block = "\n".join(
        [
            'with np.load(Path("x") / "multispecies_constrained_details.npz", allow_pickle=False) as z:',
            "    keys = np.asarray(z['species_keys'])",
        ]
    )
    assert _direct_details_load_lines(direct_literal_with_block) == [1]

    migrated_pattern = "\n".join(
        [
            'details_path = Path("run") / "multispecies_constrained_details.npz"',
            "with load_multispecies_details_npz(details_path, summary_payload=summary) as z:",
            "    yz = np.asarray(z[f'{tag}_y'], dtype=float)",
        ]
    )
    assert _direct_details_load_lines(migrated_pattern) == []

    own_prefix_format = "\n".join(
        [
            'details_path = run_dir / f"{out_prefix}_details.npz"',
            "with np.load(details_path, allow_pickle=False) as z:",
            "    pt = np.asarray(z['pt'], dtype=float)",
        ]
    )
    assert _direct_details_load_lines(own_prefix_format) == []

    gateway_pythia_format = "\n".join(
        [
            'details_path = Path(args.details_npz) if args.details_npz else (run_dir / "na49_benchmark_piminus_gateway_pythia_details.npz")',
            "z = np.load(details_path)",
        ]
    )
    assert _direct_details_load_lines(gateway_pythia_format) == []


# --- positive consumer scan: the shared loader call itself -------------------

_SHARED_LOADER_CONSUMERS: Tuple[str, ...] = (
    # The four consumers migrated by review finding 5 ...
    "study_seagull_dip.py",
    "study_absyield_closure.py",
    "plot_multispecies_sharedfit_spectra.py",
    "postprocess_multispecies_windows.py",
    # ... plus the two already-migrated scripts pinned here so a future
    # edit cannot quietly drop their loader routing either.
    "bootstrap_multispecies_robustness.py",
    "study_component_identifiability.py",
)


def _shared_loader_call_count(source: str) -> int:
    """Number of ``load_multispecies_details_npz(...)`` call sites.

    Matches both import styles found in production code: the bare name
    after ``from sps_core import load_multispecies_details_npz`` and the
    qualified attribute form ``sps_core.load_multispecies_details_npz``.
    """
    count = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_bare_name = (
            isinstance(func, ast.Name)
            and func.id == "load_multispecies_details_npz"
        )
        is_qualified = (
            isinstance(func, ast.Attribute)
            and func.attr == "load_multispecies_details_npz"
        )
        if is_bare_name or is_qualified:
            count += 1
    return count


def test_each_migrated_consumer_calls_shared_loader_in_both_trees() -> None:
    """Positive counterpart of the direct-load scan.

    The negative scan above proves no production source opens the named
    artifact with ``np.load``; this test proves every migrated consumer
    actually calls ``load_multispecies_details_npz(...)`` — i.e. the
    migration did not lose its validated open path (an ``open()`` or
    renamed helper would satisfy the negative scan but fail here).
    Expected symmetrically for monorepo and vendored tree copies.
    """
    sources_by_rel: Dict[str, List[str]] = {}
    for label, source in _iter_production_sources():
        rel = label.split("/", 2)[2]
        sources_by_rel.setdefault(rel, []).append(source)
    problems: List[str] = []
    for script in _SHARED_LOADER_CONSUMERS:
        copies = sources_by_rel.get(script, [])
        if len(copies) != 2:
            problems.append(
                f"{script}: expected 1 copy per tree, found {len(copies)}"
            )
            continue
        for index, source in enumerate(sorted(copies)):
            if _shared_loader_call_count(source) < 1:
                problems.append(
                    f"{script} [tree copy {index}]: no "
                    "load_multispecies_details_npz(...) call"
                )
    assert problems == [], "; ".join(problems)


def test_shared_loader_call_counter_matches_both_import_styles() -> None:
    """Detector self-check on inline snippets for both import styles.

    Also guards against near-miss function names: a renamed loader
    (``load_multiseries_...``) must count zero, so a rename in sps_core
    cannot silently vacate the positive consumer scan.
    """
    bare_name_style = "\n".join(
        [
            "from helpers.benchmarking.sps_core import "
            "load_multispecies_details_npz",
            'details_path = Path("run") / "artifacts.npz"',
            "with load_multispecies_details_npz("
            "details_path, summary_payload=summary) as z:",
            "    keys = z['species_keys']",
        ]
    )
    assert _shared_loader_call_count(bare_name_style) == 1

    qualified_style = "\n".join(
        [
            "import helpers.benchmarking.sps_core as sps_core",
            'z = sps_core.load_multispecies_details_npz('
            '"run/artifacts.npz", summary_payload=summary)',
        ]
    )
    assert _shared_loader_call_count(qualified_style) == 1

    near_miss = "\n".join(
        [
            "import helpers.benchmarking.sps_core as sps_core",
            'z = sps_core.load_multiseries_details_npz('
            '"run/artifacts.npz", summary_payload=summary)',
        ]
    )
    assert _shared_loader_call_count(near_miss) == 0
