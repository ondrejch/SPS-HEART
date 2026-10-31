"""The completion refit R98 (2026-10-15; owner: "yes go the physics way"):
the R97 contract with the completion kernel, generated at the R97 optimum
with assoc_norm set by judgment for the hyperons' K+ yield."""

from __future__ import annotations

import json
from pathlib import Path

from na49_mixing.flavor_weights import DEFAULT_HIDDEN_SSBAR_CONTENT

NESTED_ROOT = Path(__file__).resolve().parents[1]


def test_the_eta_prime_content():
    """The eta' at its quark-model hidden s-sbar content 0.6; the phi at 1, the eta at 0."""
    assert DEFAULT_HIDDEN_SSBAR_CONTENT == {333: 1.0, 331: 0.6}


def test_the_r98_spec_is_r97_with_the_completion_kernel():
    runs = NESTED_ROOT / "config" / "runs"
    r97, r98 = (json.loads((runs / f"{r}.json").read_text()) for r in ("r97", "r98"))
    added = {"pdg_hyperons": True, "eta_prime": True, "delta_isoscalar_pairs": True, "charged_scalar_pairs": True,
             "sibling_line_width": 0.5}
    assert r98["kernel"] == {**r97["kernel"], **added}
    assert "hyperon_meson_partners" not in r98["kernel"]
    assert r98["topology"] == {**r97["topology"], "diffractive_duality_mass": 1.7}
    for block in ("fit", "diffractive_split", "isovector", "fixed", "statistics"):
        assert r98[block] == r97[block]
    assert [p["name"] for p in r98["parameters"]] == [p["name"] for p in r97["parameters"]]
    g0 = r98["generation_0"]
    assert g0["run"] == "r97_intact_vertex_2026-10-15" and g0["fit"] == "free" and set(g0["judgment"]) == {"assoc_norm"}
    gen0 = {p["name"]: p["gen0"] for p in r98["parameters"]}
    assert gen0["assoc_norm"] == 2.78
    assert r98["seeds"]["raw_first"] > r97["seeds"]["raw_first"] + 511
    branch = NESTED_ROOT.parent / "00runs" / "r97_intact_vertex_2026-10-15" / "branch_report_f0.json"
    if branch.is_file():
        fitted = json.loads(branch.read_text())["fitted"]
        for name in g0["from_fit"]:
            assert float(gen0[name]) == float(fitted[name]), name


def test_the_ledger_scale(tmp_path):
    """The flavor ledger stores n_ssbar times 10 with the eta' (content 0.6)
    and reads a slot without the scale array as twice n_ssbar."""
    import importlib.util
    import numpy as np
    import pytest
    path = NESTED_ROOT / "helpers" / "benchmarking" / "run_engine.py"
    spec = importlib.util.spec_from_file_location("run_engine_ledger_scale", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    FC = engine.FC
    assert FC._key_ints((0.6, 1, 0, 0), 10) == (6, 1, 0, 0) and FC._key_ints((1.5, 0, 1, 0)) == (3, 0, 1, 0)
    with pytest.raises(ValueError, match="multiple of 1/2"):
        FC._key_ints((0.6, 0, 0, 0))
    arrays = {"led_event": np.asarray([0], dtype=np.int32), "led_chosen": np.asarray([[6, 0, 0, 0]], dtype=np.int16),
              "led_gvertex": np.asarray([0, 0], dtype=np.int32), "led_gkey": np.asarray([[6, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int16),
              "led_gw": np.asarray([0.3, 0.7], dtype=np.float32), "led_ssbar_scale": np.asarray([10], dtype=np.int16)}
    led = engine.Templates._ledger(arrays)
    assert sorted(led["keys"][:, 0].tolist()) == [0.0, 0.6]
    old = {k: v for k, v in arrays.items() if k != "led_ssbar_scale"}
    old["led_chosen"] = np.asarray([[3, 0, 0, 0]], dtype=np.int16)
    old["led_gkey"] = np.asarray([[3, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int16)
    assert sorted(engine.Templates._ledger(old)["keys"][:, 0].tolist()) == [0.0, 1.5]
    r98 = json.loads((NESTED_ROOT / "config" / "runs" / "r98.json").read_text())
    (tmp_path / "r98.json").write_text(json.dumps(r98))
    engine.configure(tmp_path / "r98.json")
    assert engine.LEDGER_SSBAR_SCALE == 10
    r97 = json.loads((NESTED_ROOT / "config" / "runs" / "r97.json").read_text())
    (tmp_path / "r97.json").write_text(json.dumps(r97))
    engine.configure(tmp_path / "r97.json")
    assert engine.LEDGER_SSBAR_SCALE == 2


def test_the_shared_helper_reads_the_slot_scale():
    """Review 2026-10-16, L3: ``flavor_fit_core.flavor_log_weights`` reads
    n_ssbar at the slot's own scale, so an engine slot at scale 10 and the
    same ledger at scale 2 give the same weights; a slot without the scale
    array reads as 2."""
    import importlib.util
    import numpy as np
    path = NESTED_ROOT / "helpers" / "benchmarking" / "flavor_fit_core.py"
    spec = importlib.util.spec_from_file_location("flavor_fit_core_scale", path)
    FC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FC)
    p0 = FC.FlavorParams(gamma_s_had=0.25, r_v=1.0, r_bbar=1.0, assoc_norm=1.0)
    p1 = FC.FlavorParams(gamma_s_had=0.20, r_v=1.2, r_bbar=1.0, assoc_norm=1.0)
    base = {"led_event": np.asarray([0, 1], dtype=np.int32), "led_gvertex": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "led_gw": np.asarray([0.3, 0.7, 0.5, 0.5], dtype=float)}
    # n_ssbar 1.0 and 0 at the first vertex, 1.5 and 0 (one vector) at the second
    ten = {**base, "led_chosen": np.asarray([[10, 0, 0, 0], [15, 0, 0, 0]]),
           "led_gkey": np.asarray([[10, 0, 0, 0], [0, 0, 0, 0], [15, 0, 0, 0], [0, 1, 0, 0]]),
           "led_ssbar_scale": np.asarray([10])}
    two = {**base, "led_chosen": np.asarray([[2, 0, 0, 0], [3, 0, 0, 0]]),
           "led_gkey": np.asarray([[2, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0]])}
    w10 = FC.flavor_log_weights(ten, 2, p0, p1)
    w2 = FC.flavor_log_weights(two, 2, p0, p1)
    assert np.allclose(w10, w2, rtol=0, atol=1e-12) and np.all(np.abs(w2) > 0)
