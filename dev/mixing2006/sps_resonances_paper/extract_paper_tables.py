#!/usr/bin/env python3
"""Generate LaTeX table-row snippets for the SPS resonance paper from a balanced run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SPECIES_LABEL = {
    "pi-": r"$\pi^-$",
    "pi+": r"$\pi^+$",
    "k+": r"$K^+$",
    "k-": r"$K^-$",
    "p": r"$p$",
    "pbar": r"$\bar p$",
}

RATIO_LABEL = {
    "pi+/pi-": r"$\pi^+/\pi^-$",
    "k+/k-": r"$K^+/K^-$",
    "pbar/p": r"$\bar p/p$",
    "k++k-/pi++pi-": r"$(K^+{+}K^-)/(\pi^+{+}\pi^-)$",
}

MODEL_ORDER = [
    ("resonance_shared_full_wrrmse", "na49_mixing"),
    ("pythia_full_wrrmse", "PYTHIA8"),
    ("epos_full_wrrmse", "EPOS"),
    ("urqmd_full_wrrmse", "UrQMD"),
]

RATIO_ORDER = ["pi+/pi-", "k+/k-", "pbar/p", "k++k-/pi++pi-"]
SPECIES_ORDER = ["pi-", "pi+", "k+", "k-", "p", "pbar"]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt_fixed(x: float, places: int) -> str:
    return f"{x:.{places}f}"


def _fmt_ratio(x: float) -> str:
    if abs(x) < 0.1:
        return f"{x:.4f}"
    return f"{x:.3f}"


def _latex_escape_texttt(label: str) -> str:
    return label.replace("_", r"\_")


def _fmt_sci(x: float) -> str:
    if x == 0.0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    return rf"${mant:.3f}\times10^{{{exp}}}$"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _macro_body(rows_text: str) -> str:
    lines = [line.rstrip() for line in rows_text.strip().splitlines()]
    return "\n".join(line + "%" for line in lines)


def _normalize_species_key(text: str) -> str:
    raw = str(text).strip()
    mapping = {
        "pi-": "pi-",
        "pi+": "pi+",
        "k+": "k+",
        "k-": "k-",
        "K+": "k+",
        "K-": "k-",
        "p": "p",
        "pbar": "pbar",
    }
    if raw not in mapping:
        raise KeyError(f"unsupported species key in WRRMSE CSV: {raw}")
    return mapping[raw]


def _wrrmse_rows(wrrmse_csv: Path) -> Tuple[str, Dict[str, object]]:
    rows = _read_csv_rows(wrrmse_csv)
    by_species = {_normalize_species_key(row["species"]): row for row in rows}
    means: Dict[str, float] = {name: 0.0 for _, name in MODEL_ORDER}
    wins: Dict[str, int] = {name: 0 for _, name in MODEL_ORDER}
    out_lines: List[str] = []

    for species in SPECIES_ORDER:
        row = by_species[species]
        model_vals: List[Tuple[str, float]] = []
        rendered: List[str] = []
        for key, name in MODEL_ORDER:
            val = float(row[key])
            model_vals.append((name, val))
            means[name] += val
            rendered.append(_fmt_fixed(val, 4))
        min_val = min(v for _, v in model_vals)
        for name, val in model_vals:
            if math.isclose(val, min_val, rel_tol=0.0, abs_tol=1e-12):
                wins[name] += 1
        out_lines.append(f"{SPECIES_LABEL[species]} & {' & '.join(rendered)} " + r"\\")
    out_lines.append(r"\midrule")
    out_lines.append(
        "Mean over species & "
        + " & ".join(_fmt_fixed(means[name] / len(SPECIES_ORDER), 4) for _, name in MODEL_ORDER)
        + r" \\" 
    )
    out_lines.append(
        "Wins over species & "
        + " & ".join(f"{wins[name]}/{len(SPECIES_ORDER)}" for _, name in MODEL_ORDER)
        + r" \\" 
    )
    payload = {"means": means, "wins": wins}
    return "\n".join(out_lines) + "\n", payload


def _active_coeff_rows(summary_json: Path) -> Tuple[str, Dict[str, object]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    weights = payload.get("resonance", {}).get("active_weights_by_label_sorted", {})
    if not isinstance(weights, dict) or not weights:
        raise RuntimeError(f"No active weights found in {summary_json}")
    total = float(sum(float(v) for v in weights.values()))
    out_lines: List[str] = []
    rows_out: List[Dict[str, object]] = []
    for label, value in sorted(weights.items(), key=lambda kv: float(kv[1]), reverse=True):
        val = float(value)
        share = 100.0 * val / total if total > 0.0 else 0.0
        rows_out.append({"label": label, "value": val, "share_percent": share})
        out_lines.append(
            rf"\texttt{{{_latex_escape_texttt(str(label))}}} & {_fmt_sci(val)} & {share:.2f} \\")
    return "\n".join(out_lines) + "\n", {"rows": rows_out, "total": total}


def _ratio_rows(ratio_csv: Path) -> Tuple[str, Dict[str, object]]:
    rows = _read_csv_rows(ratio_csv)
    by_ratio = {
        row["ratio"]: row
        for row in rows
        if row.get("window", "") == "xf_0.000_0.400"
    }
    out_lines: List[str] = []
    rows_out: List[Dict[str, object]] = []
    for ratio in RATIO_ORDER:
        row = by_ratio[ratio]

        def _cell(column: str) -> Optional[float]:
            # External-model columns are intentionally empty in ratio
            # windows (externals are fenced out of per-window ratios);
            # render them as an em-dash placeholder.
            raw = str(row.get(column, "")).strip()
            return float(raw) if raw else None

        vals = [
            _cell("data"),
            _cell("resonance_shared"),
            _cell("pythia8"),
            _cell("epos"),
            _cell("urqmd"),
        ]
        rows_out.append({"ratio": ratio, "values": vals})
        out_lines.append(
            f"{RATIO_LABEL[ratio]} & "
            + " & ".join("---" if v is None else _fmt_ratio(v) for v in vals)
            + r" \\"
        )
    return "\n".join(out_lines) + "\n", {"rows": rows_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    summary_json = run_dir / "multispecies_constrained_summary.json"
    wrrmse_csv = run_dir / "multispecies_constrained_wrrmse.csv"
    ratio_csv = run_dir / "multispecies_constrained_particle_ratios.csv"
    for path in (summary_json, wrrmse_csv, ratio_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    wrrmse_text, wrrmse_meta = _wrrmse_rows(wrrmse_csv)
    active_text, active_meta = _active_coeff_rows(summary_json)
    ratio_text, ratio_meta = _ratio_rows(ratio_csv)

    _write(out_dir / "table_wrrmse_rows.tex", wrrmse_text)
    _write(out_dir / "table_active_coeff_rows.tex", active_text)
    _write(out_dir / "table_ratio_rows.tex", ratio_text)
    macro_lines = [
        r"\renewcommand{\PaperTableWRRMSERows}{%",
        _macro_body(wrrmse_text),
        r"}",
        r"\renewcommand{\PaperTableActiveCoeffRows}{%",
        _macro_body(active_text),
        r"}",
        r"\renewcommand{\PaperTableRatioRows}{%",
        _macro_body(ratio_text),
        r"}",
        "",
    ]
    _write(out_dir / "paper_tables.tex", "\n".join(macro_lines))
    manifest = {
        "run_dir": str(run_dir),
        "inputs": {
            "summary_json": str(summary_json),
            "wrrmse_csv": str(wrrmse_csv),
            "ratio_csv": str(ratio_csv),
        },
        "tables": {
            "wrrmse": wrrmse_meta,
            "active_coefficients": active_meta,
            "ratios": ratio_meta,
        },
    }
    _write(out_dir / "paper_tables_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir / 'table_wrrmse_rows.tex'}")
    print(f"wrote {out_dir / 'table_active_coeff_rows.tex'}")
    print(f"wrote {out_dir / 'table_ratio_rows.tex'}")
    print(f"wrote {out_dir / 'paper_tables.tex'}")
    print(f"wrote {out_dir / 'paper_tables_manifest.json'}")


if __name__ == "__main__":
    main()
