#!/usr/bin/env python3
"""Compare Fischer 2022 Table 9 pi- yields to the current cascade model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "00runs"
DEFAULT_TABLE9_JSON = REPO_ROOT / "helpers" / "benchmarking" / "config" / "fischer2022_table9_pi_minus.json"
DEFAULT_BRANCHING_JSON = REPO_ROOT / "helpers" / "benchmarking" / "config" / "soft_resonance_branching_pdg2025.json"


@dataclass(frozen=True)
class ResonanceRow:
    name: str
    pdg_id: int
    sector: str
    total_yield: float
    table_branching_fraction: float
    table_pi_minus_yield: float


@dataclass(frozen=True)
class ResonanceCheck:
    name: str
    pdg_id: int
    sector: str
    total_yield: float
    table_branching_fraction: float
    table_pi_minus_yield: float
    model_expected_pi_minus_per_parent: float
    model_pi_minus_yield: float
    delta_model_minus_table: float
    ratio_model_over_table: float


def _load_rows(table9_path: Path) -> Tuple[List[ResonanceRow], dict]:
    data = json.loads(table9_path.read_text(encoding="utf-8"))
    rows = [ResonanceRow(**row) for row in data["rows"]]
    return rows, data


def _load_branching_config(branching_path: Path) -> dict:
    return json.loads(branching_path.read_text(encoding="utf-8"))


def _load_decay_chain(branching_overrides: Dict[str, float] | None = None):
    mixing2006_dir = REPO_ROOT
    if str(mixing2006_dir) not in sys.path:
        sys.path.insert(0, str(mixing2006_dir))
    from na49_mixing.decays import build_soft_resonance_decays  # pylint: disable=import-error

    return build_soft_resonance_decays(branching_overrides=branching_overrides)


def _load_overrides(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "branching_overrides" in data and isinstance(data["branching_overrides"], dict):
            return {str(k): float(v) for k, v in data["branching_overrides"].items()}
        if "overrides" in data and isinstance(data["overrides"], dict):
            return {str(k): float(v) for k, v in data["overrides"].items()}
        return {str(k): float(v) for k, v in data.items()}
    raise ValueError(f"Invalid overrides JSON shape in {path}.")


def _expected_pi_minus_per_parent(
    pdg_id: int,
    chain: dict,
    memo: Dict[int, float],
    visiting: Set[int],
) -> float:
    if pdg_id == -211:
        return 1.0
    if pdg_id in memo:
        return memo[pdg_id]
    if pdg_id in visiting:
        raise RuntimeError(f"Cycle detected in decay chain at PDG {pdg_id}.")

    pdecays = chain.get(pdg_id)
    if pdecays is None:
        memo[pdg_id] = 0.0
        return 0.0

    visiting.add(pdg_id)
    total = 0.0
    prev = 0.0
    for cum, mode in zip(pdecays._probabilities, pdecays._modes):
        prob = max(float(cum) - prev, 0.0)
        prev = float(cum)
        daughters_pi = 0.0
        for daughter_id in mode.daughters:
            daughters_pi += _expected_pi_minus_per_parent(daughter_id, chain, memo, visiting)
        total += prob * daughters_pi
    visiting.remove(pdg_id)

    memo[pdg_id] = total
    return total


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-15:
        return math.nan
    return num / den


def _run_check(rows: List[ResonanceRow], chain: dict) -> List[ResonanceCheck]:
    memo: Dict[int, float] = {}
    checks: List[ResonanceCheck] = []

    for row in rows:
        exp_pi = _expected_pi_minus_per_parent(row.pdg_id, chain, memo, set())
        model_yield = row.total_yield * exp_pi
        delta = model_yield - row.table_pi_minus_yield
        checks.append(
            ResonanceCheck(
                name=row.name,
                pdg_id=row.pdg_id,
                sector=row.sector,
                total_yield=row.total_yield,
                table_branching_fraction=row.table_branching_fraction,
                table_pi_minus_yield=row.table_pi_minus_yield,
                model_expected_pi_minus_per_parent=exp_pi,
                model_pi_minus_yield=model_yield,
                delta_model_minus_table=delta,
                ratio_model_over_table=_safe_ratio(model_yield, row.table_pi_minus_yield),
            )
        )
    return checks


def _sum_sector(checks: List[ResonanceCheck], sector: str) -> float:
    return sum(c.model_pi_minus_yield for c in checks if c.sector == sector)


def _write_json(
    out_json: Path,
    table9_path: Path,
    branching_path: Path,
    overrides_path: Path | None,
    branching_overrides: Dict[str, float],
    table9_data: dict,
    branching_data: dict,
    checks: List[ResonanceCheck],
) -> None:
    model_total = sum(c.model_pi_minus_yield for c in checks)
    model_baryon = _sum_sector(checks, "baryon")
    model_meson = _sum_sector(checks, "meson")
    table_total_rows = sum(c.table_pi_minus_yield for c in checks)
    table_baryon_rows = sum(c.table_pi_minus_yield for c in checks if c.sector == "baryon")
    table_meson_rows = sum(c.table_pi_minus_yield for c in checks if c.sector == "meson")
    table_total_quoted = float(table9_data["totals"]["table_pi_minus_total"])
    table_baryon_quoted = float(table9_data["totals"]["table_pi_minus_from_baryons"])
    table_meson_quoted = float(table9_data["totals"]["table_pi_minus_from_mesons"])

    payload = {
        "inputs": {
            "table9_json": str(table9_path),
            "branching_config_json": str(branching_path),
            "overrides_json": str(overrides_path) if overrides_path is not None else "",
        },
        "table9_source": table9_data.get("metadata", {}),
        "branching_config_metadata": branching_data.get("metadata", {}),
        "branching_config_sources": branching_data.get("sources", {}),
        "branching_overrides": branching_overrides,
        "checks": [asdict(c) for c in checks],
        "totals": {
            "model_pi_minus_from_baryons": model_baryon,
            "model_pi_minus_from_mesons": model_meson,
            "model_pi_minus_total": model_total,
            "table_pi_minus_from_baryons_rowsum": table_baryon_rows,
            "table_pi_minus_from_mesons_rowsum": table_meson_rows,
            "table_pi_minus_total_rowsum": table_total_rows,
            "table_pi_minus_from_baryons_quoted": table_baryon_quoted,
            "table_pi_minus_from_mesons_quoted": table_meson_quoted,
            "table_pi_minus_total_quoted": table_total_quoted,
            "model_minus_table_total_rowsum": model_total - table_total_rows,
            "model_over_table_total_rowsum": _safe_ratio(model_total, table_total_rows),
            "model_minus_table_total_quoted": model_total - table_total_quoted,
            "model_over_table_total_quoted": _safe_ratio(model_total, table_total_quoted),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(
    out_md: Path,
    table9_path: Path,
    branching_path: Path,
    overrides_path: Path | None,
    branching_overrides: Dict[str, float],
    checks: List[ResonanceCheck],
    table9_data: dict,
    branching_data: dict,
) -> None:
    model_total = sum(c.model_pi_minus_yield for c in checks)
    model_baryon = _sum_sector(checks, "baryon")
    model_meson = _sum_sector(checks, "meson")
    table_total_rows = sum(c.table_pi_minus_yield for c in checks)
    table_baryon_rows = sum(c.table_pi_minus_yield for c in checks if c.sector == "baryon")
    table_meson_rows = sum(c.table_pi_minus_yield for c in checks if c.sector == "meson")
    table_total_quoted = float(table9_data["totals"]["table_pi_minus_total"])
    table_baryon_quoted = float(table9_data["totals"]["table_pi_minus_from_baryons"])
    table_meson_quoted = float(table9_data["totals"]["table_pi_minus_from_mesons"])

    lines: List[str] = []
    lines.append("# Fischer 2022 Table 9 vs Current Cascade: pi- Yield Check")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"- Table 9 JSON: `{table9_path}`")
    lines.append(f"- Branching config JSON: `{branching_path}`")
    lines.append(f"- Branching config name: `{branching_data.get('metadata', {}).get('name', 'unknown')}`")
    if overrides_path is not None:
        lines.append(f"- Overrides JSON: `{overrides_path}`")
        lines.append(f"- Number of active overrides: `{len(branching_overrides)}`")
    lines.append("")
    lines.append("| Resonance | PDG | Total yield | Table B(pi-) | Model E[n(pi-)] | Table pi- yield | Model pi- yield | Delta(model-table) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in checks:
        lines.append(
            f"| {c.name} | {c.pdg_id} | {c.total_yield:.6f} | {c.table_branching_fraction:.6f} | "
            f"{c.model_expected_pi_minus_per_parent:.6f} | {c.table_pi_minus_yield:.6f} | "
            f"{c.model_pi_minus_yield:.6f} | {c.delta_model_minus_table:.6f} |"
        )
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- Model pi- from baryons: `{model_baryon:.6f}`")
    lines.append(f"- Table 9 pi- from baryons (row-sum): `{table_baryon_rows:.6f}`")
    lines.append(f"- Table 9 pi- from baryons (quoted): `{table_baryon_quoted:.6f}`")
    lines.append(f"- Model pi- from mesons: `{model_meson:.6f}`")
    lines.append(f"- Table 9 pi- from mesons (row-sum): `{table_meson_rows:.6f}`")
    lines.append(f"- Table 9 pi- from mesons (quoted): `{table_meson_quoted:.6f}`")
    lines.append(f"- Model pi- total: `{model_total:.6f}`")
    lines.append(f"- Table 9 pi- total (row-sum): `{table_total_rows:.6f}`")
    lines.append(f"- Table 9 pi- total (quoted): `{table_total_quoted:.6f}`")
    lines.append(f"- Model minus Table 9 total (row-sum): `{(model_total - table_total_rows):.6f}`")
    lines.append(f"- Model/Table total ratio (row-sum): `{_safe_ratio(model_total, table_total_rows):.6f}`")
    lines.append(f"- Model minus Table 9 total (quoted): `{(model_total - table_total_quoted):.6f}`")
    lines.append(f"- Model/Table total ratio (quoted): `{_safe_ratio(model_total, table_total_quoted):.6f}`")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Model values are computed analytically from the current `build_soft_resonance_decays()` chain.")
    lines.append("- Daughters without decay entries are treated as terminal for this pi- counting calculation.")
    lines.append("- This checks yield closure of the current branch/cascade setup against Fischer Table 9 row values.")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Fischer 2022 Table 9 pi- yields against current cascade.")
    parser.add_argument("--table9-json", default=str(DEFAULT_TABLE9_JSON))
    parser.add_argument("--branching-json", default=str(DEFAULT_BRANCHING_JSON))
    parser.add_argument("--overrides-json", default="", help="Optional branching-overrides JSON.")
    parser.add_argument("--run-description", default="local_default")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    table9_path = Path(args.table9_json)
    branching_path = Path(args.branching_json)
    overrides_path = Path(args.overrides_json) if args.overrides_json else None
    run_dir = RUNS_ROOT / args.run_description
    out_json = Path(args.out_json) if args.out_json else (run_dir / "fischer2022_table9_pi_minus_yield_check.json")
    out_md = Path(args.out_md) if args.out_md else (run_dir / "FISCHER2022_TABLE9_PI_MINUS_YIELD_CHECK.md")

    if not table9_path.exists():
        raise FileNotFoundError(f"Missing table JSON: {table9_path}")
    if not branching_path.exists():
        raise FileNotFoundError(f"Missing branching JSON: {branching_path}")
    if overrides_path is not None and not overrides_path.exists():
        raise FileNotFoundError(f"Missing overrides JSON: {overrides_path}")

    rows, table9_data = _load_rows(table9_path)
    branching_data = _load_branching_config(branching_path)
    branching_overrides = _load_overrides(overrides_path)
    chain = _load_decay_chain(branching_overrides=branching_overrides)
    checks = _run_check(rows, chain)
    _write_json(
        out_json,
        table9_path,
        branching_path,
        overrides_path,
        branching_overrides,
        table9_data,
        branching_data,
        checks,
    )
    _write_markdown(
        out_md,
        table9_path,
        branching_path,
        overrides_path,
        branching_overrides,
        checks,
        table9_data,
        branching_data,
    )

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")


if __name__ == "__main__":
    main()
