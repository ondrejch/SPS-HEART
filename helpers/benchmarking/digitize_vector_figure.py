#!/usr/bin/env python3
"""Vector-figure digitizer for NA49 proceedings PDFs.

Extracts data-point coordinates directly from vector-figure drawing
commands (no rasterization), calibrated by tick-label text positions.
Applies to born-digital PDFs only; scanned sources go through
WebPlotDigitizer / Engauge instead (see notes/benchmarks/
NP_DATA_DIGITIZATION_TASK.md).

Currently implemented figure:

  fig2a  andrzej-QM2004.pdf page 2 (index 1), Figure 2(a):
         pi+/pi- ratio in p+p (filled discs) and pi-/pi+ ratio in n+p
         (open circles) versus x_F in the projectile hemisphere.
         x_F here is p_L/p_L^max in the c.m. system (paper footnote),
         consistent with the x_F' convention used at these energies.

Outputs CSV files `xf,value,err` with provenance comment lines, written
to --out-dir (default notes/benchmarks/digitized/na49 relative to the
monorepo root).

Self-check: the extracted mirror pair must satisfy
R_np(x_F) = pi-/pi+|np versus R_pp(x_F) = pi+/pi-|pp; the driver-level
identity compares model curves, while here we merely tabulate both and
their consistency pull for human review.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pymupdf
except ImportError:  # pragma: no cover
    print("ERROR: PyMuPDF required: python3 -m pip install --user pymupdf",
          file=sys.stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DEFAULT = REPO_ROOT / "more_docs" / "andrzej-QM2004.pdf"


# ----------------------------------------------------------------------
# Generic primitives
# ----------------------------------------------------------------------

def _word_centers(page) -> List[Tuple[float, float, str]]:
    out = []
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        out.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, w))
    return out


def _fit_axis(
    centers: List[Tuple[float, float, str]],
    *,
    values: Dict[str, float],
    window: Tuple[float, float, float, float],
    along: str,
) -> Tuple[float, float, int]:
    """Least-squares pixel->data mapping from tick-label centers.

    ``window`` = (cx_lo, cx_hi, cy_lo, cy_hi) selects candidate label
    positions; ``along`` fixes which coordinate the data value maps to
    ('x' -> center-x of the labels varies, 'y' -> center-y varies).
    Outlier rejection runs in PIXEL space with a 2 pt threshold.
    """
    cx_lo, cx_hi, cy_lo, cy_hi = window
    pts = []
    for cx, cy, w in centers:
        v = values.get(w.rstrip("."))
        if v is None:
            continue
        if not (cx_lo <= cx <= cx_hi and cy_lo <= cy <= cy_hi):
            continue
        # duplicate tick values from neighbouring panels: keep the first
        if any(abs(v - p[1]) < 1e-9 for p in pts):
            continue
        coord = cx if along == "x" else cy
        pts.append((coord, float(v)))
    if len(pts) < 2:
        raise ValueError(f"axis calibration: only {len(pts)} labels matched")
    arr_c = np.array([p[0] for p in pts])
    arr_v = np.array([p[1] for p in pts])
    b, a = np.polyfit(arr_c, arr_v, 1)
    for _ in range(2):
        resid_pt = np.abs(arr_v - (a + b * arr_c)) / max(abs(b), 1e-12)
        keep = resid_pt <= 2.0
        if keep.all():
            break
        arr_c, arr_v = arr_c[keep], arr_v[keep]
        b, a = np.polyfit(arr_c, arr_v, 1)
    return float(a), float(b), len(arr_c)


# ----------------------------------------------------------------------
# Figure 2(a), andrzej-QM2004.pdf page index 1
# ----------------------------------------------------------------------

_PANEL = dict(x=(115.0, 290.0), y=(295.0, 407.0))
_X_LABELS = {"0": 0.0, "0.05": 0.05, "0.1": 0.10, "0.15": 0.15, "0.2": 0.20}
_Y_VALUES = {"0.8": 0.8, "1": 1.0, "1.2": 1.2, "1.4": 1.4,
             "1.6": 1.6, "1.8": 1.8, "2": 2.0, "2.2": 2.2}
_LEGEND_GLYPH_PX = 217.8  # x0 of the two legend glyphs (both series)


def _marker_centers(page) -> Tuple[List[dict], List[dict]]:
    """Return (filled_series_points, open_series_points).

    Legend glyphs (at the known legend x position) are excluded from the
    data series and reported separately.
    """
    filled, opened = [], []
    x_lo, x_hi = _PANEL["x"]
    y_lo, y_hi = _PANEL["y"]
    for d in page.get_drawings():
        r = d["rect"]
        if not (x_lo <= r.x0 and r.x1 <= x_hi and y_lo <= r.y0 and r.y1 <= y_hi):
            continue
        # markers are small many-segment circles ~2.5 pt across
        if len(d["items"]) < 20 or not (1.5 < r.width < 6 and 1.5 < r.height < 6):
            continue
        entry = {
            "px": r.x0 + r.width / 2.0,
            "py": r.y0 + r.height / 2.0,
            "is_legend": abs(r.x0 - _LEGEND_GLYPH_PX) < 1.5,
        }
        if d.get("fill") is not None:
            filled.append(entry)
        elif d.get("color") is not None:
            opened.append(entry)
    return filled, opened


def _error_bars(page, px: float, py: float) -> Optional[float]:
    """Half-height of the vertical error bar through a marker, in data units.

    Returns None when no bar segments are found (upper limit arrows etc.).
    """
    segs = []
    for d in page.get_drawings():
        if d.get("fill") is not None or d.get("color") != (0.0, 0.0, 0.0):
            continue
        r = d["rect"]
        if not (_PANEL["y"][0] <= r.y0 and r.y1 <= _PANEL["y"][1]):
            continue
        if r.width > 1.5 or r.height < 1.0 or r.height > 12.0:
            continue
        if abs(r.x0 + r.width / 2.0 - px) > 1.2:
            continue
        segs.append((r.y0, r.y1))
    if not segs:
        return None
    top = min(s[0] for s in segs)
    bot = max(s[1] for s in segs)
    return abs(bot - top) / 2.0


def extract_fig2a(pdf_path: Path) -> Dict[str, object]:
    doc = pymupdf.open(str(pdf_path))
    page = doc[1]
    centers = _word_centers(page)

    # x axis: labels in a horizontal strip under panel (a), left column only
    ax, bx, nx = _fit_axis(
        centers, values=_X_LABELS,
        window=(100.0, 280.0, 402.0, 413.0), along="x")
    # y axis: labels in a vertical strip left of panel (a) only
    ay, by, ny = _fit_axis(
        centers, values=_Y_VALUES,
        window=(95.0, 125.0, 292.0, 408.0), along="y")

    filled, opened = _marker_centers(page)

    def to_points(entries: List[dict]) -> List[dict]:
        pts = []
        for e in entries:
            if e["is_legend"]:
                continue
            xf = ax + bx * e["px"]
            val = ay + by * e["py"]
            err_px = _error_bars(page, e["px"], e["py"])
            err = abs(by) * err_px if err_px is not None else None
            pts.append({"xf": round(xf, 4), "value": round(val, 4),
                        "err": round(err, 4) if err is not None else None})
        pts.sort(key=lambda p: p["xf"])
        return pts

    result = {
        "source": str(pdf_path),
        "page_index": 1,
        "figure": "Figure 2(a)",
        "xf_convention": "pL/pLmax in c.m. (paper footnote)",
        "calibration": {
            "x_labels_used": nx, "x_a": round(ax, 4), "x_b": round(bx, 5),
            "y_labels_used": ny, "y_a": round(ay, 4), "y_b": round(by, 5),
        },
        "series_pp_piplus_over_piminus": to_points(filled),
        "series_np_piminus_over_piplus": to_points(opened),
        "n_legend_glyphs": int(sum(e["is_legend"] for e in filled + opened)),
    }

    # Self-consistency tabulation for review (not an assertion: data may
    # deviate within errors; the model-level identity lives in the driver).
    pp = {p["xf"]: p for p in result["series_pp_piplus_over_piminus"]}
    rows = []
    for p in result["series_np_piminus_over_piplus"]:
        m = pp.get(p["xf"])
        if m and p["err"] and m["err"]:
            diff = p["value"] - m["value"]
            sigma = math_hypot(p["err"], m["err"])
            rows.append({"xf": p["xf"], "pp": m["value"], "np": p["value"],
                         "diff": round(diff, 4),
                         "pull": round(diff / sigma, 2) if sigma else None})
    result["mirror_consistency"] = rows
    return result


def math_hypot(a: float, b: float) -> float:
    return (a * a + b * b) ** 0.5


# ----------------------------------------------------------------------
# Figure 1(b), andrzej-QM2004.pdf page index 1 (top-right panel)
# ----------------------------------------------------------------------

_FIG1B_X_VALUES = {"0": 0.0, "0.2": 0.2, "0.4": 0.4,
                   "0.6": 0.6, "0.8": 0.8, "1": 1.0}
_FIG1B_Y_LOG_VALUES = {"0.1": -1.0, "1": 0.0}  # log10


def extract_fig1b(pdf_path: Path) -> Dict[str, object]:
    """Charged-kaon pT spectra at fixed x_F = 0.1, four series.

    Panel annotation: 158 GeV, x_F = 0.1. Marker shape encodes charge
    (circle = K+, triangle = K-); marker color encodes the system
    (red = p+p, blue = n+p). The pp/np pairs nearly coincide, which is
    the caption's equality statement made visible.
    """
    doc = pymupdf.open(str(pdf_path))
    page = doc[1]
    centers = _word_centers(page)

    ax, bx, nx = _fit_axis(
        centers, values=_FIG1B_X_VALUES,
        window=(295.0, 468.0, 210.0, 223.0), along="x")
    ay, by, ny = _fit_axis(
        centers, values=_FIG1B_Y_LOG_VALUES,
        window=(283.0, 306.0, 90.0, 220.0), along="y")

    series = {("pp", "kplus"): [], ("pp", "kminus"): [],
              ("np", "kplus"): [], ("np", "kminus"): []}
    for d in page.get_drawings():
        r = d["rect"]
        if not (300 <= r.x0 and r.x1 <= 472 and 90 <= r.y0 and r.y1 <= 214):
            continue
        if not (1.5 < r.width < 6 and 1.5 < r.height < 6):
            continue
        f = d.get("fill")
        if f is None:
            continue
        color = tuple(round(c, 2) for c in f)
        system = {"(1.0, 0.0, 0.0)": "pp",
                  "(0.0, 0.0, 1.0)": "np"}.get(str(color))
        if system is None:
            continue
        charge = ("kplus" if len(d["items"]) >= 20
                  else "kminus" if len(d["items"]) <= 8 else None)
        if charge is None:
            continue
        px, py = r.x0 + r.width / 2.0, r.y0 + r.height / 2.0

        tops, bots = [], []
        for d2 in page.get_drawings():
            if d2.get("fill") is not None:
                continue
            r2 = d2["rect"]
            if not (90 <= r2.y0 and r2.y1 <= 214):
                continue
            if r2.width > 1.5 or not (1.0 <= r2.height <= 14.0):
                continue
            if abs(r2.x0 + r2.width / 2.0 - px) > 1.2:
                continue
            tops.append(r2.y0)
            bots.append(r2.y1)
        err_px = (max(bots) - min(tops)) / 2.0 if tops else None

        pT = ax + bx * px
        val_log = ay + by * py
        entry = {"pt": round(pT, 4), "value": round(10.0 ** val_log, 6)}
        if err_px is not None:
            lo = 10.0 ** (val_log - err_px * abs(by))
            hi = 10.0 ** (val_log + err_px * abs(by))
            entry["err"] = round((hi - lo) / 2.0, 6)
        else:
            entry["err"] = None
        series[(system, charge)].append(entry)

    for k in series:
        series[k].sort(key=lambda p: p["pt"])

    return {
        "source": str(pdf_path),
        "page_index": 1,
        "figure": "Figure 1(b)",
        "observable": "F(pT) kaon spectra [mb] at fixed x_F = 0.1, log y; "
                      "shape=charge (circle K+, triangle K-), "
                      "color=system (red pp, blue np)",
        "fixed_xf_bin": 0.1,
        "beam_momentum_gev_c": 158.0,
        "calibration": {
            "x_labels_used": nx, "x_a": round(ax, 4), "x_b": round(bx, 6),
            "y_labels_used": ny, "y_a_log10": round(ay, 4),
            "y_b_log10_per_pt": round(by, 6),
        },
        "n_points": {f"{s}_{c}": len(v) for (s, c), v in series.items()},
        "series_pp_kplus": series[("pp", "kplus")],
        "series_np_kplus": series[("np", "kplus")],
        "series_pp_kminus": series[("pp", "kminus")],
        "series_np_kminus": series[("np", "kminus")],
    }


# ----------------------------------------------------------------------
# Figure 4(a), andrzej-QM2004.pdf page index 4
# ----------------------------------------------------------------------

_FIG4A_X_VALUES = {"0": 0.0, "0.05": 0.05, "0.1": 0.10, "0.15": 0.15,
                   "0.2": 0.20, "0.25": 0.25, "0.3": 0.30, "0.35": 0.35}
_FIG4A_Y_VALUES = {"0": 0.0, "0.05": 0.05, "0.1": 0.10, "0.15": 0.15,
                   "0.2": 0.20, "0.25": 0.25, "0.3": 0.30}


def _norm_axis_word(w: str) -> str:
    """Normalize dash variants used in axis labels (en-dash, minus sign)."""
    return w.replace("\u2013", "-").replace("\u2212", "-").rstrip(".")


def _error_bar_halfspan(page, px: float,
                        y_range: Tuple[float, float]) -> Optional[float]:
    """Half-height of vertical bar segments at *px*, in points."""
    tops, bots = [], []
    for d2 in page.get_drawings():
        if d2.get("fill") is not None:
            continue
        r2 = d2["rect"]
        if not (y_range[0] <= r2.y0 and r2.y1 <= y_range[1]):
            continue
        if r2.width > 1.5 or not (1.0 <= r2.height <= 14.0):
            continue
        if abs(r2.x0 + r2.width / 2.0 - px) > 1.2:
            continue
        tops.append(r2.y0)
        bots.append(r2.y1)
    if not tops:
        return None
    return (max(bots) - min(tops)) / 2.0


def extract_fig4a(pdf_path: Path) -> Dict[str, object]:
    """Anti-proton dN/dxF spectra for four hadron+proton systems.

    Filled red circles = p+p, filled blue circles = n+p, open red =
    pi+ + p, open blue = pi- + p (legend row order from text spans:
    p+p, n+p, pi+, pi-). The pp and np curves are the driver-overlay
    targets; the two pion-beam curves are emitted as extra context
    files.
    """
    doc = pymupdf.open(str(pdf_path))
    page = doc[4]
    centers = []
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                        _norm_axis_word(w)))

    ax, bx, nx = _fit_axis(
        centers, values=_FIG4A_X_VALUES,
        # cx_lo=127 excludes the y-axis corner '0' at cx=123
        window=(127.0, 292.0, 211.0, 227.0), along="x")
    ay, by, ny = _fit_axis(
        centers, values=_FIG4A_Y_VALUES,
        window=(108.0, 130.0, 100.0, 220.0), along="y")

    series = {("pp", "filled"): [], ("np", "filled"): [],
              ("pip", "open"): [], ("pim", "open"): []}
    # legend glyph exclusion window (left of the legend texts)
    leg_x = (200.0, 262.0)
    leg_y = (110.0, 148.0)

    for d in page.get_drawings():
        r = d["rect"]
        if not (113 <= r.x0 and r.x1 <= 288 and 96 <= r.y0 and r.y1 <= 216):
            continue
        if not (1.5 < r.width < 6 and 1.5 < r.height < 6):
            continue
        if len(d["items"]) < 20:
            continue
        f = d.get("fill")
        col = tuple(round(c, 2) for c in (f if f is not None
                                          else d.get("color") or (0, 0, 0)))
        filled = f is not None
        red = col == (1.0, 0.0, 0.0)
        blue = col == (0.0, 0.0, 1.0)
        if not (red or blue):
            continue
        px, py = r.x0 + r.width / 2.0, r.y0 + r.height / 2.0
        if leg_x[0] <= px <= leg_x[1] and leg_y[0] <= py <= leg_y[1]:
            continue
        if filled:
            key = ("pp" if red else "np"), "filled"
        else:
            # open circles: red = pi+ + p, blue = pi- + p per legend rows
            key = ("pip" if red else "pim"), "open"
        err_px = _error_bar_halfspan(page, px, (96.0, 216.0))
        xf = ax + bx * px
        val = ay + by * py
        entry = {"xf": round(xf, 4), "value": round(val, 5)}
        if err_px is not None:
            entry["err"] = round(abs(by) * err_px, 5)
        else:
            entry["err"] = None
        series[key].append(entry)

    out_series = {}
    for k, v in series.items():
        v.sort(key=lambda p: p["xf"])
        name = {"pp": "pp", "np": "np",
                "pip": "piplusp", "pim": "piminusp"}[k[0]]
        out_series[f"series_{name}_pbar"] = v

    result = {
        "source": str(pdf_path),
        "page_index": 4,
        "figure": "Figure 4(a)",
        "observable": "anti-proton dN/dxF, hadron+p at 158 GeV/c; "
                      "filled red=p+p, filled blue=n+p, open red=pi++p, "
                      "open blue=pi-+p",
        "calibration": {
            "x_labels_used": nx, "x_a": round(ax, 4), "x_b": round(bx, 6),
            "y_labels_used": ny, "y_a": round(ay, 4), "y_b": round(by, 6),
        },
        "n_points": {k: len(v) for k, v in out_series.items()},
    }
    result.update(out_series)
    return result

def _write_csv(path: Path, system: str, observable: str,
               series: List[dict], meta: Dict[str, object],
               x_col: str = "xf") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    conv = meta.get("xf_convention") or meta.get("observable", "")
    lines = [
        f"# digitized from vector PDF: {meta['source']}",
        f"# page_index={meta['page_index']} figure={meta['figure']}",
        f"# system={system} observable={observable}",
        f"# variable convention: {conv}",
        f"# method: PyMuPDF get_drawings marker centroids; "
        f"axes calibrated from tick-label centers "
        f"(x:{meta['calibration']['x_labels_used']} labels, "
        f"y:{meta['calibration']['y_labels_used']} labels)",
        f"# tool: digitize_vector_figure.py (PyMuPDF {pymupdf.__version__}) "
        f"on {stamp}",
        "# errors from drawn error-bar half-heights where present",
        f"{x_col},value,err",
    ]
    for p in series:
        x = p[x_col]
        e = "" if p["err"] is None else f"{p['err']}"
        lines.append(f"{x},{p['value']},{e}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="digitize_vector_figure")
    ap.add_argument("--pdf", default=str(PDF_DEFAULT))
    ap.add_argument("--figure", default="fig2a",
                    choices=("fig2a", "fig1b", "fig4a"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # canonical digitized-data home: the monorepo notes/ tree
        base = REPO_ROOT
        out_dir = base / "notes" / "benchmarks" / "digitized" / "na49"
        for _ in range(5):
            if (base / "notes" / "benchmarks").is_dir():
                out_dir = base / "notes" / "benchmarks" / "digitized" / "na49"
                break
            if base.parent == base:
                break
            base = base.parent

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        base = REPO_ROOT
        for _ in range(5):
            cand = base / "more_docs" / pdf_path.name
            if cand.exists():
                pdf_path = cand
                break
            if base.parent == base:
                break
            base = base.parent
    if args.figure == "fig2a":
        meta = extract_fig2a(pdf_path)
    elif args.figure == "fig1b":
        meta = extract_fig1b(pdf_path)
    elif args.figure == "fig4a":
        meta = extract_fig4a(pdf_path)
    else:  # pragma: no cover
        raise SystemExit(f"unknown figure {args.figure}")

    if args.figure == "fig1b":
        for system in ("pp", "np"):
            for charge in ("kplus", "kminus"):
                _write_csv(
                    out_dir / f"na49_{system}_{charge}_pt.csv", system,
                    f"F(pT) kaon {charge} spectrum [mb] at xF=0.1",
                    meta[f"series_{system}_{charge}"], meta,  # type: ignore[arg-type]
                    x_col="pt")
        print(json.dumps(meta, indent=2)[:2000])
        print(f"\nwrote 4 files na49_{{pp,np}}_{{kplus,kminus}}_pt.csv "
              f"in {out_dir}")
        return 0

    if args.figure == "fig4a":
        for system, label in (("pp", "p+p"), ("np", "n+p"),
                              ("piplusp", "pi+ + p"),
                              ("piminusp", "pi- + p")):
            _write_csv(
                out_dir / f"na49_{system}_pbar_xf.csv",
                "pp" if system == "pp" else system,
                f"anti-proton dN/dxF ({label})",
                meta[f"series_{system}_pbar"], meta,  # type: ignore[arg-type]
                x_col="xf")
        print(json.dumps({k: v for k, v in meta.items()
                          if not k.startswith("series")}, indent=2))
        for name in ("pp", "np", "piplusp", "piminusp"):
            rows = meta[f"series_{name}_pbar"]
            print(f"{name}: {len(rows)} points")
        print(f"\nwrote 4 files na49_*_pbar_xf.csv in {out_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "na49_pp_pioverpi_xf.csv", "pp", "pi+/pi-",
               meta["series_pp_piplus_over_piminus"], meta,  # type: ignore[arg-type]
               x_col="xf")
    _write_csv(out_dir / "na49_np_pioverpi_xf.csv", "np", "pi-/pi+",
               meta["series_np_piminus_over_piplus"], meta,  # type: ignore[arg-type]
               x_col="xf")

    print(json.dumps(meta, indent=2)[:2400])
    print(f"\nwrote {out_dir}/na49_{{pp,np}}_pioverpi_xf.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
