#!/usr/bin/env python3
"""Compare Fischer Table-9, PDG-anchored cascade expectation, and fitted resonance weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "dev" / "00runs"


def _selector_fn_map() -> Dict[str, Callable[[str], bool]]:
    return {
        "Delta0(1232)": lambda lbl: lbl == "delta0",
        "Delta-(1232)": lambda lbl: lbl == "delta-",
        "N*(1440)": lambda lbl: lbl.startswith("n1440"),
        "N*(1520)": lambda lbl: lbl.startswith("n1520"),
        "N*(1680)": lambda lbl: lbl.startswith("n1680"),
        "eta(548)": lambda lbl: lbl == "eta",
        "rho0(770)": lambda lbl: lbl == "rho0",
        "rho-(770)": lambda lbl: lbl == "rho_minus",
        "omega(782)": lambda lbl: lbl == "omega",
        "f2(1270)": lambda lbl: lbl in ("f2", "f2_1270"),
        "rho3^0(1690)": lambda lbl: lbl in ("rho3_zero", "rho3_zero_1690"),
        "rho3-(1690)": lambda lbl: lbl in ("rho3_minus", "rho3_minus_1690"),
        "f4(2050)": lambda lbl: lbl in ("f4", "f4_2050"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Table-9 branching fractions to PDG cascade expectation and fitted weights."
    )
    parser.add_argument("--summary-json", required=True, help="multispecies_constrained_summary.json")
    parser.add_argument(
        "--table9-check-json",
        required=True,
        help="fischer2022_table9_pi_minus_yield_check.json produced by check_fischer2022_table9_pi_minus.py",
    )
    parser.add_argument("--run-description", default="table9_pdg_fit_compare_local")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    check_path = Path(args.table9_check_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
    if not check_path.exists():
        raise FileNotFoundError(f"Missing table9-check JSON: {check_path}")

    run_dir = RUNS_ROOT / args.run_description
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else (run_dir / "table9_pdg_fit_compare.csv")
    out_md = Path(args.out_md) if str(args.out_md).strip() else (run_dir / "TABLE9_PDG_FIT_COMPARE.md")
    out_json = Path(args.out_json) if str(args.out_json).strip() else (run_dir / "table9_pdg_fit_compare.json")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table9_check = json.loads(check_path.read_text(encoding="utf-8"))
    weights_by_label = {
        str(k): float(v) for k, v in summary.get("resonance", {}).get("weights_by_label_sum", {}).items()
    }
    w_total = float(sum(weights_by_label.values()))

    sel_map = _selector_fn_map()
    rows = []
    selected_labels = set()
    for row in table9_check.get("checks", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", ""))
        selector = sel_map.get(name)
        fit_weight = 0.0
        if selector is not None:
            matched = [str(lbl) for lbl in weights_by_label if selector(str(lbl))]
            selected_labels.update(matched)
            fit_weight = float(sum(weights_by_label[lbl] for lbl in matched))
        fit_share_pct = (100.0 * fit_weight / w_total) if w_total > 0.0 else 0.0
        rows.append(
            {
                "name": name,
                "pdg_id": int(row.get("pdg_id", 0)),
                "table_total_yield": float(row.get("total_yield", 0.0)),
                "table_b_pi_minus": float(row.get("table_branching_fraction", 0.0)),
                "pdg_chain_b_pi_minus": float(row.get("model_expected_pi_minus_per_parent", 0.0)),
                "fit_selector_weight_raw": fit_weight,
                "fit_selector_weight_share_percent": fit_share_pct,
            }
        )
    selected_weight_total = float(sum(weights_by_label.get(lbl, 0.0) for lbl in selected_labels))
    nonselected_weight_total = float(max(w_total - selected_weight_total, 0.0))
    nonselected_share_pct = (100.0 * nonselected_weight_total / w_total) if w_total > 0.0 else 0.0
    selected_share_pct = (100.0 * selected_weight_total / w_total) if w_total > 0.0 else 0.0
    top_nonselected = sorted(
        (
            (str(lbl), float(val))
            for lbl, val in weights_by_label.items()
            if str(lbl) not in selected_labels and float(val) > 0.0
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:10]

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "name",
                "pdg_id",
                "table_total_yield",
                "table_b_pi_minus",
                "pdg_chain_b_pi_minus",
                "fit_selector_weight_raw",
                "fit_selector_weight_share_percent",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["name"],
                    r["pdg_id"],
                    r["table_total_yield"],
                    r["table_b_pi_minus"],
                    r["pdg_chain_b_pi_minus"],
                    r["fit_selector_weight_raw"],
                    r["fit_selector_weight_share_percent"],
                ]
            )

    lines = []
    lines.append("# Table-9 / PDG / Fit-Weight Comparison")
    lines.append("")
    lines.append(f"- summary JSON: `{summary_path}`")
    lines.append(f"- table9-check JSON: `{check_path}`")
    lines.append(f"- total fitted weight sum: `{w_total:.10g}`")
    lines.append(
        f"- Table-9 selector-covered fitted weight: `{selected_weight_total:.10g}` ({selected_share_pct:.4f}%)"
    )
    lines.append(
        f"- non-Table-9 fitted weight: `{nonselected_weight_total:.10g}` ({nonselected_share_pct:.4f}%)"
    )
    lines.append("")
    lines.append("| resonance | PDG | Fischer total yield | Fischer B(pi-) | PDG-chain E[n(pi-) per parent] | fitted selector weight | fitted selector share [%] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['pdg_id']} | {r['table_total_yield']:.6f} | "
            f"{r['table_b_pi_minus']:.6f} | {r['pdg_chain_b_pi_minus']:.6f} | "
            f"{r['fit_selector_weight_raw']:.8g} | {r['fit_selector_weight_share_percent']:.4f} |"
        )
    lines.append("")
    lines.append("Top non-Table-9 fitted contributors:")
    lines.append("")
    lines.append("| label | fitted weight | fitted share [%] |")
    lines.append("|---|---:|---:|")
    for lbl, val in top_nonselected:
        share = (100.0 * float(val) / w_total) if w_total > 0.0 else 0.0
        lines.append(f"| {lbl} | {float(val):.8g} | {share:.4f} |")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "summary_json": str(summary_path),
        "table9_check_json": str(check_path),
        "fit_weight_total": w_total,
        "table9_selector_weight_total": selected_weight_total,
        "table9_selector_weight_share_percent": selected_share_pct,
        "non_table9_weight_total": nonselected_weight_total,
        "non_table9_weight_share_percent": nonselected_share_pct,
        "top_non_table9_labels": [
            {
                "label": str(lbl),
                "weight": float(val),
                "share_percent": (100.0 * float(val) / w_total) if w_total > 0.0 else 0.0,
            }
            for lbl, val in top_nonselected
        ],
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved Markdown: {out_md}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
