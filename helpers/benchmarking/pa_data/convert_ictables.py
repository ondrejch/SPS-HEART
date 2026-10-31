#!/usr/bin/env python3
"""Convert NA49 p+Pb pion tables from the Chvala PhD thesis (2006) LaTeX
datatables into benchmark-ready flat files.

Source tables: /home/o/NA49/phd/datables/
  ictable.111.{piplus,piminus}.dat  minimum-bias invariant cross section
                                    f(xF,pT) [mb/(GeV/c)^2], Tab. tabicmb
  ictable.000.{piplus,piminus}.dat  "All centrality triggered" per-event
                                    invariant density d(xF,pT) [c^3/GeV^2]
  ictable.001..005.*                centrality selected sets 1..5, d(xF,pT)
  int.111.tex                       pT-integrated dn/dxF, <pT>, <pT^2>,
                                    minimum-bias, Tab. tabptimb

Row layout of the ictable sources:
  first cell = pT [GeV/c]; then pairs (value, err_percent) for
  xF = 0.00, 0.05, ..., 0.40 (9 bins). Empty trailing cells are skipped.
int.NNN.tex layout:
  xF; then per charge pi+ then pi-: (dn/dxF, D%), (<pT>, D%), (<pT^2>, D%).

Outputs are written next to this script (pa_data/).
"""

import random
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

SRC_DIR = Path("/home/o/NA49/phd/datables")
OUT_DIR = Path(__file__).resolve().parent

XF_BINS = ["0.00", "0.05", "0.10", "0.15", "0.20",
           "0.25", "0.30", "0.35", "0.40"]

CONV_DATE = date.today().isoformat()

ICTABLE_SETS = {
    "000": ("pPb_dens_minbias_%s.txt",
            "All centrality triggered set of p+Pb events"),
    "001": ("pPb_dens_set1_%s.txt",
            "centrality selected set 1 of p+Pb events"),
    "002": ("pPb_dens_set2_%s.txt",
            "centrality selected set 2 of p+Pb events"),
    "003": ("pPb_dens_set3_%s.txt",
            "centrality selected set 3 of p+Pb events"),
    "004": ("pPb_dens_set4_%s.txt",
            "centrality selected set 4 of p+Pb events"),
    "005": ("pPb_dens_set5_%s.txt",
            "centrality selected set 5 of p+Pb events"),
    "111": ("xsect_%s_pPb.txt",
            "minimum bias p+Pb interactions"),
}

CHARGES = {"piplus": "pi+", "piminus": "pi-"}


def fmt_dec(d):
    """Plain decimal string without exponent or trailing zeros."""
    return format(Decimal(d).normalize(), "f")


def strip_latex(line):
    """Remove trailing row terminator and hline markers from a raw line."""
    line = line.replace("\\hline", " ")
    line = re.sub(r"\\\\\s*$", " ", line)
    return line


def parse_ictable(path):
    """Parse one ictable file.

    Returns (records, anomalies):
      records = list of dicts with keys: src_line, pt_str, xf, value_str,
                errpct_str
      anomalies = list of human-readable warning strings
    """
    records, anomalies = [], []
    with open(path, encoding="ascii", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            cells = [c.strip() for c in strip_latex(raw).split("&")]
            # full tables carry 9 xF pairs (19 cells); centrality set 5
            # covers only xF <= 0.35 and uses 8 pairs (17 cells)
            n_bins = (len(cells) - 1) // 2
            if len(cells) % 2 == 0 or n_bins not in (8, 9):
                anomalies.append(
                    f"{path.name}:L{lineno}: expected 19 or 17 cells, got "
                    f"{len(cells)} -- line kept but structure unexpected")
                if len(cells) < 2:
                    continue
            try:
                pt = Decimal(cells[0])
            except Exception:
                anomalies.append(
                    f"{path.name}:L{lineno}: non-numeric first cell "
                    f"{cells[0]!r}, skipped")
                continue
            pairs_seen = 0
            for k in range(n_bins):
                v = cells[1 + 2 * k]
                e = cells[2 + 2 * k]
                if v == "" and e == "":
                    continue  # silently skip empty pair
                pairs_seen += 1
                if v == "" or e == "":
                    anomalies.append(
                        f"{path.name}:L{lineno}: half-empty pair at xF="
                        f"{XF_BINS[k]} (value={v!r}, err={e!r}), skipped")
                    continue
                try:
                    Decimal(v)
                    Decimal(e)
                except Exception:
                    anomalies.append(
                        f"{path.name}:L{lineno}: unparseable pair at xF="
                        f"{XF_BINS[k]} ({v!r},{e!r}), skipped")
                    continue
                records.append({
                    "src_line": lineno,
                    "pair_idx": k,
                    "xf": XF_BINS[k],
                    "pt_str": cells[0],
                    "pt": pt,
                    "value_str": v,
                    "errpct_str": e,
                })
    # pT labels repeated on more than one source row
    lines_per_pt = {}
    for r in records:
        lines_per_pt.setdefault(r["pt"], set()).add(r["src_line"])
    for k in sorted(v for v in lines_per_pt if len(lines_per_pt[v]) > 1):
        lns = "/".join(str(n) for n in sorted(lines_per_pt[k]))
        anomalies.append(
            f"{path.name}: duplicated pT label {fmt_dec(k)} GeV/c on source "
            f"rows {lns} -- preserved as-is")
    return records, anomalies


def parse_int(path):
    """Parse an int.NNN.tex table.

    Returns dict charge -> (records, anomalies); record has xF plus the
    three quantities with percent errors.
    """
    out = {"piplus": ([], []), "piminus": ([], [])}
    with open(path, encoding="ascii", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            cells = [c.strip() for c in strip_latex(raw).split("&")]
            if len(cells) != 13:
                for ch in out:
                    out[ch][1].append(
                        f"{path.name}:L{lineno}: expected 13 cells, got "
                        f"{len(cells)} -- line skipped")
                continue
            try:
                xf = Decimal(cells[0])
            except Exception:
                for ch in out:
                    out[ch][1].append(
                        f"{path.name}:L{lineno}: non-numeric xF "
                        f"{cells[0]!r}, skipped")
                continue
            for ch, base in (("piplus", 1), ("piminus", 7)):
                rec = {"src_line": lineno, "xf_str": cells[0], "xf": xf}
                ok = True
                for j, name in enumerate(("dndxF", "meanpT", "meanpT2")):
                    v, e = cells[base + 2 * j], cells[base + 2 * j + 1]
                    if v == "" or e == "":
                        out[ch][1].append(
                            f"{path.name}:L{lineno}: empty {name} at "
                            f"xF={cells[0]}, skipped line")
                        ok = False
                        break
                    rec[name] = v
                    rec[name + "_pct"] = e
                if ok:
                    out[ch][0].append(rec)
    return out


def write_xsect_like(fname, quantity, quantity_long, set_label, table_label,
                     records, anomalies):
    lines = [
        "# NA49 p+Pb pi production data converted to flat format",
        f"# quantity: {quantity_long}",
        "# system: p+Pb, 158 GeV/c beam momentum (sqrt(s)_NN = 17.2 GeV), NA49",
        f"# sample: {set_label}",
        "# feed-down correction: applied (feed-down subtracted data from K0s, K0l, Lambda, Sigma- decays)",
        "# source: O. Chvala, PhD thesis, Charles University Prague (2006),"
        f" Tab. '{table_label}' via datables/ictable files",
        "# errors: statistical only (percent in thesis), converted here to absolute",
        "# columns: xF  pT[GeV/c]  value  abs_error",
        f"# conversion date: {CONV_DATE} by helpers/benchmarking/pa_data/convert_ictables.py",
    ]
    for a in anomalies:
        lines.append(f"# NOTE: {a}")
    rows = []
    for r in records:
        value = Decimal(r["value_str"])
        err = value * Decimal(r["errpct_str"]) / Decimal(100)
        rows.append((Decimal(r["xf"]), r["pt"],
                     f"{r['xf']} {r['pt_str']} {r['value_str']} "
                     f"{fmt_dec(err)}"))
    rows.sort(key=lambda t: (t[0], t[1]))
    body = [txt for _, _, txt in rows]
    path = OUT_DIR / fname
    path.write_text("\n".join(lines + body) + "\n", encoding="ascii")
    return path, len(body)


def write_ptint(fname, charge_sym, records, anomalies):
    lines = [
        "# NA49 p+Pb pi production, pT-integrated quantities, flat format",
        f"# charge: {charge_sym}; sample: minimum bias p+Pb interactions",
        "# system: p+Pb, 158 GeV/c beam momentum (sqrt(s)_NN = 17.2 GeV), NA49",
        "# feed-down correction: applied (feed-down subtracted data)",
        "# source: O. Chvala, PhD thesis, Charles University Prague (2006),"
        " Tab. 'tabptimb' via datables/int.111.tex",
        "# thesis column semantics (header above \\input of int.111.tex in phdthesis.tex):",
        "#   xF | dn/dxF, Delta[%] | <pT> [GeV/c], Delta[%] | <pT^2> [(GeV/c)^2], Delta[%]",
        "#   per charge (pi+ block then pi- block); Delta = statistical error in %",
        "# errors below converted to absolute: abs = value * pct / 100",
        "# columns: xF  dn/dxF  err_abs  mean_pT[GeV/c]  err_abs  mean_pT2[(GeV/c)^2]  err_abs",
        f"# conversion date: {CONV_DATE} by helpers/benchmarking/pa_data/convert_ictables.py",
        "# note: dn/dxF is the per-event pT-integrated density of the thesis"
        " (dimension c); it is NOT divided by sigma_inel",
    ]
    for a in anomalies:
        lines.append(f"# NOTE: {a}")
    recs = sorted(records, key=lambda r: r["xf"])
    body = []
    for r in recs:
        vals = [r["xf"]]
        for name in ("dndxF", "meanpT", "meanpT2"):
            v = Decimal(r[name])
            e = v * Decimal(r[name + "_pct"]) / Decimal(100)
            vals.append(r[name])
            vals.append(fmt_dec(e))
        body.append(" ".join(str(v) for v in vals))
    path = OUT_DIR / fname
    path.write_text("\n".join(lines + body) + "\n", encoding="ascii")
    return path, len(body)


def count_raw_cells(path):
    """Independent recount of non-empty data points in an ictable file.

    A point is a non-empty cell at an odd value position of the 18
    quantity cells (positions 0,2,4,... hold values, 1,3,5,... errors).
    """
    total = 0
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            cells = [c.strip() for c in strip_latex(raw).split("&")][1:]
            total += sum(1 for i, c in enumerate(cells)
                         if c != "" and i % 2 == 0)
    return total


def count_int_rows(path):
    """Independent recount of data rows in an int.NNN.tex table."""
    total = 0
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            cells = [c.strip() for c in strip_latex(raw).split("&")]
            try:
                Decimal(cells[0])
            except (IndexError, Exception):
                continue
            total += 1
    return total


def spot_check(path, records, n=5, seed=20260825):
    """Re-extract n random records straight from raw text and compare.

    Returns list of dicts: coord, raw_value, raw_pct, conv_value, conv_pct,
    match.
    """
    rng = random.Random(seed)
    picks = rng.sample(records, min(n, len(records)))
    raw_lines = open(path, encoding="ascii", errors="replace").readlines()
    results = []
    for r in picks:
        raw = raw_lines[r["src_line"] - 1]
        cells = [c.strip() for c in strip_latex(raw).split("&")]
        k = r["pair_idx"]
        raw_v, raw_e = cells[1 + 2 * k], cells[2 + 2 * k]
        match = (raw_v == r["value_str"] and raw_e == r["errpct_str"])
        results.append({
            "coord": (f"{path.name}:L{r['src_line']} xF={r['xf']} "
                      f"(cell pair {k + 1}/9)"),
            "raw_value": raw_v, "raw_pct": raw_e,
            "conv_value": r["value_str"], "conv_pct": r["errpct_str"],
            "match": match,
        })
    return results


def main():
    written = []
    all_anoms = []
    val_report = []

    for code, (fname_tpl, label) in ICTABLE_SETS.items():
        for ch in ("piplus", "piminus"):
            src = SRC_DIR / f"ictable.{code}.{ch}.dat"
            recs, anoms = parse_ictable(src)
            all_anoms.extend(anoms)
            if code == "111":
                q_short = "f(xF,pT)"
                q_long = ("invariant cross section f(xF,pT) "
                          "[mb/(GeV/c)^2], minimum bias")
                tbl = "tabicmb"
            else:
                q_short = "d(xF,pT)"
                q_long = ("per-event invariant density d(xF,pT) "
                          "[c^3/GeV^2]")
                tbl = {("000", "piplus"): "tabicall"}.get((code, ch))
                if tbl is None:
                    tbl = f"tabiccs{int(code)}" if code != "000" else "tabicall"
            fname = fname_tpl % ch
            path, nrows = write_xsect_like(fname, q_short, q_long, label,
                                           tbl, recs, anoms)
            written.append(path)

            # validation
            n_raw_cells = count_raw_cells(src)
            assert nrows == n_raw_cells, (
                f"{fname}: wrote {nrows} rows but raw file has "
                f"{n_raw_cells} non-empty cells")
            pos = all(Decimal(r["value_str"]) > 0 for r in recs)
            assert pos, f"{fname}: non-positive value found"
            xs = [Decimal(r["xf"]) for r in recs]
            ps = [r["pt"] for r in recs]
            checks = spot_check(src, recs)
            val_report.append({
                "file": fname, "rows": nrows, "raw_cells": n_raw_cells,
                "min_xF": min(xs), "max_xF": max(xs),
                "min_pT": min(ps), "max_pT": max(ps), "spot": checks,
            })

    intp = parse_int(SRC_DIR / "int.111.tex")
    for ch in ("piplus", "piminus"):
        recs, anoms = intp[ch]
        all_anoms.extend(anoms)
        fname = f"ptintegrated_{ch}_pPb.txt"
        path, nrows = write_ptint(fname, CHARGES[ch], recs, anoms)
        written.append(path)
        assert nrows == len(recs)
        assert all(Decimal(r[n]) > 0 for r in recs
                   for n in ("dndxF", "meanpT", "meanpT2"))
        xs = [r["xf"] for r in recs]
        n_raw_rows = count_int_rows(SRC_DIR / "int.111.tex")
        assert nrows == n_raw_rows, (
            f"{fname}: wrote {nrows} rows but raw table has "
            f"{n_raw_rows} data rows")
        val_report.append({
            "file": fname, "rows": nrows, "raw_cells": n_raw_rows,
            "min_xF": min(xs), "max_xF": max(xs),
            "min_pT": None, "max_pT": None, "spot": [],
        })

        spot = []
        rng = random.Random(20260825 + (0 if ch == "piplus" else 1))
        raw_lines = open(SRC_DIR / "int.111.tex",
                         encoding="ascii").read().splitlines()
        for r in rng.sample(recs, 2):
            cells = [c.strip() for c
                     in strip_latex(raw_lines[r["src_line"] - 1]).split("&")]
            base = 1 if ch == "piplus" else 7
            got = [cells[base], cells[base + 1], cells[base + 2],
                   cells[base + 3], cells[base + 4], cells[base + 5]]
            want = [r["dndxF"], r["dndxF_pct"], r["meanpT"],
                    r["meanpT_pct"], r["meanpT2"], r["meanpT2_pct"]]
            assert got == want, f"{fname} L{r['src_line']}: {got} != {want}"
            spot.append({
                "coord": (f"int.111.tex:L{r['src_line']} "
                          f"xF={r['xf_str']} ({CHARGES[ch]})"),
                "raw_value": " ".join(got[0::2]),
                "raw_pct": " ".join(got[1::2]),
                "conv_value": " ".join(want[0::2]),
                "conv_pct": " ".join(want[1::2]),
                "match": True,
            })
        val_report[-1]["spot"] = spot

    # ---- console report -------------------------------------------------
    print(f"Wrote {len(written)} files into {OUT_DIR}")
    for v in val_report:
        print(f"\n== {v['file']} ==")
        print(f"   rows={v['rows']}  raw_nonempty_cells={v['raw_cells']}  "
              f"count_match={'OK' if v['rows'] == v['raw_cells'] else 'FAIL'}"
              f"  all_values_positive=OK")
        print(f"   xF range [{fmt_dec(v['min_xF'])}, {fmt_dec(v['max_xF'])}]"
              + ("" if v["min_pT"] is None else
                  f";  pT range [{fmt_dec(v['min_pT'])}, "
                  f"{fmt_dec(v['max_pT'])}] GeV/c"))
        for s in v["spot"]:
            tag = "OK " if s["match"] else "MISMATCH"
            print(f"   [{tag}] {s['coord']}")
            print(f"          raw source cells : value(s)={s['raw_value']} "
                  f"err%={s['raw_pct']}")
            print(f"          converted record: value(s)={s['conv_value']} "
                  f"err%={s['conv_pct']}")

    if all_anoms:
        print("\nANOMALIES:")
        for a in all_anoms:
            uniq = a
            print(f"  - {uniq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
