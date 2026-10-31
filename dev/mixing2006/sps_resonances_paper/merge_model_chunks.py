#!/usr/bin/env python3
"""Merge local chunk CSVs into the merged external baseline files expected by the paper workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

MODEL_PREFIX = {
    "urqmd": "urqmd_pp158_{species_tag}_gateway.csv",
    "epos": "epos_lhc_pp158_{species_tag}_gateway.csv",
    "pythia": "pythia8_softqcd_pp158_{species_tag}_gateway.csv",
}
SPECIES_TAGS = ["piminus", "piplus", "kaplus", "kaminus", "proton", "antiproton"]


def _collect_parts(run_dir: Path, model: str, species_tag: str) -> List[Path]:
    pattern = f"{model}_chunk_*_{species_tag}.csv"
    return sorted(run_dir.glob(pattern))


def _merge_parts(parts: List[Path], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(["xf", "pt", "value"])
        for part in parts:
            with part.open(newline="", encoding="utf-8") as fh_in:
                reader = csv.DictReader(fh_in)
                expected = ["xf", "pt", "value"]
                if list(reader.fieldnames or []) != expected:
                    raise ValueError(f"Unexpected header in {part}: {reader.fieldnames}")
                for row in reader:
                    writer.writerow([row["xf"], row["pt"], row["value"]])
                    row_count += 1
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest-json", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    manifest_rows: List[Dict[str, object]] = []
    for model, merged_name in MODEL_PREFIX.items():
        for species_tag in SPECIES_TAGS:
            parts = _collect_parts(run_dir, model, species_tag)
            if not parts:
                raise FileNotFoundError(f"No chunk CSVs found for model={model}, species_tag={species_tag} in {run_dir}")
            out_csv = run_dir / merged_name.format(species_tag=species_tag)
            rows = _merge_parts(parts, out_csv)
            manifest_rows.append(
                {
                    "model": model,
                    "species_tag": species_tag,
                    "chunks": [str(p) for p in parts],
                    "merged_csv": str(out_csv),
                    "row_count": rows,
                }
            )
            print(f"merged {model}:{species_tag} -> {out_csv} ({rows} rows)")

    manifest_path = (
        Path(args.manifest_json).expanduser().resolve()
        if str(args.manifest_json).strip()
        else (run_dir / "local_external_merge_manifest.json")
    )
    manifest_path.write_text(json.dumps({"rows": manifest_rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
