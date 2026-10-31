#!/usr/bin/env python3
"""Convert the p_T-integrated pion tables of the p+Pb centrality samples
(thesis Tab. ``tabptiall``: ``datables/int.000.tex`` to ``int.005.tex``)
into flat files, with the parser and error conventions of
``convert_ictables.py`` (which converted the minimum-bias ``int.111.tex``).

Written 2026-09-29. The existing flat files are not touched: this script
writes only ``ptintegrated_{piplus,piminus}_pPb_{alltrig,set1..set5}.txt``.
Sample metadata (n_grey selection, events, <nu>) are copied from the
thesis Tab. ``tabcdbins``.
"""
from __future__ import annotations

import random
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_ictables as CI  # noqa: E402  (parse_int, count_int_rows, strip_latex, fmt_dec)

CONV_DATE = "2026-09-29"
SAMPLES = {
    "000": ("alltrig", "All centrality triggered set of p+Pb events", "n_grey > 0", 1667072, "5.1"),
    "001": ("set1", "centrality selected set 1 of p+Pb events", "n_grey = 1, 2", 249509, "2.9"),
    "002": ("set2", "centrality selected set 2 of p+Pb events", "n_grey = 3, 4, 5", 467234, "4.1"),
    "003": ("set3", "centrality selected set 3 of p+Pb events", "n_grey = 6, 7, 8", 490228, "5.2"),
    "004": ("set4", "centrality selected set 4 of p+Pb events", "n_grey = 9 -- 13", 400683, "6.0"),
    "005": ("set5", "centrality selected set 5 of p+Pb events", "n_grey = 14 -- 22", 62039, "6.9"),
}


def write_file(tag: str, ch: str, recs: list, anomalies: list, meta: tuple) -> tuple[Path, int]:
    label, sample, selection, events, nu = meta
    lines = [
        "# NA49 p+Pb pi production, pT-integrated quantities, flat format",
        f"# charge: {CI.CHARGES[ch]}; sample: {sample}",
        f"# centrality selection: {selection}; events: {events}; <nu> = {nu} (thesis Tab. 'tabcdbins')",
        "# system: p+Pb, 158 GeV/c beam momentum (sqrt(s)_NN = 17.2 GeV), NA49",
        "# feed-down correction: applied (feed-down subtracted data)",
        "# source: O. Chvala, PhD thesis, Charles University Prague (2006),"
        f" Tab. 'tabptiall' via datables/int.{tag}.tex",
        "# thesis column semantics (header above \\input of the table in phdthesis.tex):",
        "#   xF | dn/dxF, Delta[%] | <pT> [GeV/c], Delta[%] | <pT^2> [(GeV/c)^2], Delta[%]",
        "#   per charge (pi+ block then pi- block); Delta = statistical error in %",
        "# errors below converted to absolute: abs = value * pct / 100",
        "# columns: xF  dn/dxF  err_abs  mean_pT[GeV/c]  err_abs  mean_pT2[(GeV/c)^2]  err_abs",
        f"# conversion date: {CONV_DATE} by helpers/benchmarking/pa_data/convert_ptint_centrality.py",
        "# note: dn/dxF is the per-event pT-integrated density of the thesis"
        " (per triggered event of the sample); it is NOT divided by a cross section",
    ]
    if tag == "005":
        lines.append("# NOTE: the thesis table for set 5 has no xF = 0.400 row (set 5 covers xF = 0.00..0.35)")
    for a in anomalies:
        lines.append(f"# NOTE: {a}")
    for r in sorted(recs, key=lambda r: r["xf"]):
        for name in ("dndxF", "meanpT", "meanpT2"):
            if Decimal(r[name + "_pct"]) <= 0:
                lines.append(f"# NOTE: int.{tag}.tex L{r['src_line']}: the thesis prints the {name} statistical error at "
                             f"xF = {r['xf_str']} ({CI.CHARGES[ch]}) as {r[name + '_pct']} %; preserved as printed "
                             "(the absolute error below carries the same sign); treat it as unknown")
    body = []
    for r in sorted(recs, key=lambda r: r["xf"]):
        vals = [r["xf"]]
        for name in ("dndxF", "meanpT", "meanpT2"):
            v = Decimal(r[name])
            vals.append(r[name])
            vals.append(CI.fmt_dec(v * Decimal(r[name + "_pct"]) / Decimal(100)))
        body.append(" ".join(str(v) for v in vals))
    path = Path(__file__).resolve().parent / f"ptintegrated_{ch}_pPb_{label}.txt"
    path.write_text("\n".join(lines + body) + "\n", encoding="ascii")
    return path, len(body)


def main() -> int:
    for tag, meta in SAMPLES.items():
        src = CI.SRC_DIR / f"int.{tag}.tex"
        parsed = CI.parse_int(src)
        raw_lines = src.read_text(encoding="ascii").splitlines()
        n_raw = CI.count_int_rows(src)
        for ch in ("piplus", "piminus"):
            recs, anoms = parsed[ch]
            path, n = write_file(tag, ch, recs, anoms, meta)
            assert n == len(recs) == n_raw, f"{path.name}: {n} rows written, {len(recs)} parsed, {n_raw} raw"
            assert all(Decimal(r[k]) > 0 for r in recs for k in ("dndxF", "meanpT", "meanpT2")), path.name
            rng = random.Random(20260929 + int(tag) * 2 + (0 if ch == "piplus" else 1))
            base = 1 if ch == "piplus" else 7
            for r in rng.sample(recs, 2):
                cells = [c.strip() for c in CI.strip_latex(raw_lines[r["src_line"] - 1]).split("&")]
                got = cells[base:base + 6]
                want = [r["dndxF"], r["dndxF_pct"], r["meanpT"], r["meanpT_pct"], r["meanpT2"], r["meanpT2_pct"]]
                assert got == want, f"{path.name} L{r['src_line']}: {got} != {want}"
            xs = [r["xf"] for r in recs]
            print(f"{path.name}: {n} rows (raw {n_raw}), xF {min(xs)}..{max(xs)}, spot checks OK, anomalies {len(anoms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
