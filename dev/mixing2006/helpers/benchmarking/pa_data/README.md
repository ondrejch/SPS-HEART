# pa_data — NA49 p+Pb pion benchmark data (flat files)

Flat-file conversion of the NA49 p+Pb pion tables of
O. Chvala, PhD thesis, Charles University Prague (2006)
(`/home/o/NA49/phd/`, thesis text `phdthesis.tex`, data files `datables/`).

- System: p+Pb at 158 GeV/c beam momentum (sqrt(s)_NN = 17.2 GeV), NA49.
- Data are feed-down corrected (feed-down subtracted; feed-down pions from
  K0_S, K0_L, Lambda and Sigma-minus "V0" decays removed).
- Errors in the thesis tables are statistical only, given in percent;
  the flat files carry absolute errors computed as `abs = value * pct / 100`
  (exact decimal arithmetic, values copied verbatim from the source cells).
- Conversion performed 2026-08-25 by `convert_ictables.py` in this directory
  (re-run with `python3 convert_ictables.py`; it re-reads the thesis
  datatables and rewrites all files below).

## Normalization anchors (thesis Tab. `tabinelcs2`, phdthesis.tex ~L805)

| reaction | sigma_inel [mb] | <nu> |
|----------|-----------------|------|
| p+C      | 226 +/- 7       | 1.69 |
| p+Pb     | 1806 +/- 25     | 3.65 |

These anchor the relation between invariant cross section f(xF,pT)
[mb/(GeV/c)^2] and per-event density d(xF,pT) [c^3/GeV^2]:
dn/dy = (pi/sigma_inel) * Integral f dpT^2.

## File list

Invariant cross section (minimum bias, thesis Tab. `tabicmb`,
source `datables/ictable.111.*.dat`):

| file | rows | quantity |
|------|------|----------|
| xsect_piplus_pPb.txt  | 155 | f(xF,pT) [mb/(GeV/c)^2], pi+ |
| xsect_piminus_pPb.txt | 154 | f(xF,pT) [mb/(GeV/c)^2], pi- |

Per-event invariant density (thesis Tab. `tabicall`, `tabiccs1`..`tabiccs5`,
source `datables/ictable.NNN.*.dat`):

| file | rows | sample |
|------|------|--------|
| pPb_dens_minbias_piplus.txt  | 155 | All centrality triggered, pi+ |
| pPb_dens_minbias_piminus.txt | 155 | All centrality triggered, pi- |
| pPb_dens_set1_piplus.txt     | 155 | centrality selected set 1, pi+ |
| pPb_dens_set1_piminus.txt    | 155 | centrality selected set 1, pi- |
| pPb_dens_set2_piplus.txt     | 155 | centrality selected set 2, pi+ |
| pPb_dens_set2_piminus.txt    | 155 | centrality selected set 2, pi- |
| pPb_dens_set3_piplus.txt     | 155 | centrality selected set 3, pi+ |
| pPb_dens_set3_piminus.txt    | 155 | centrality selected set 3, pi- |
| pPb_dens_set4_piplus.txt     | 155 | centrality selected set 4, pi+ |
| pPb_dens_set4_piminus.txt    | 155 | centrality selected set 4, pi- |
| pPb_dens_set5_piplus.txt     | 138 | centrality selected set 5, pi+ |
| pPb_dens_set5_piminus.txt    | 136 | centrality selected set 5, pi- |

Centrality naming: NNN=000 is the "All centrality triggered" sample of the
thesis (events with at least one grey proton). NNN=001..005 are the centrality
selected sets 1..5, defined by the number of reconstructed grey protons
(thesis Tab. `tabcdbins`; see the table of the pT-integrated centrality
files below for the selections, event counts, and <nu>).

pT-integrated quantities (minimum bias, thesis Tab. `tabptimb`,
source `datables/int.111.tex`):

| file | rows |
|------|------|
| ptintegrated_piplus_pPb.txt  | 9 |
| ptintegrated_piminus_pPb.txt | 9 |

pT-integrated quantities for the centrality samples (thesis Tab. `tabptiall`,
source `datables/int.000.tex`..`int.005.tex`; converted 2026-09-29 by
`convert_ptint_centrality.py`, which reuses the parser and error
conventions of `convert_ictables.py` and leaves the files above untouched):

| file | rows | sample | n_grey | events | <nu> |
|------|------|--------|--------|--------|------|
| ptintegrated_{piplus,piminus}_pPb_alltrig.txt | 9 | all centrality triggered | > 0 | 1667072 | 5.1 |
| ptintegrated_{piplus,piminus}_pPb_set1.txt | 9 | set 1 | 1, 2 | 249509 | 2.9 |
| ptintegrated_{piplus,piminus}_pPb_set2.txt | 9 | set 2 | 3, 4, 5 | 467234 | 4.1 |
| ptintegrated_{piplus,piminus}_pPb_set3.txt | 9 | set 3 | 6, 7, 8 | 490228 | 5.2 |
| ptintegrated_{piplus,piminus}_pPb_set4.txt | 9 | set 4 | 9 -- 13 | 400683 | 6.0 |
| ptintegrated_{piplus,piminus}_pPb_set5.txt | 8 | set 5 | 14 -- 22 | 62039 | 6.9 |

The centrality sets are bins in the number of reconstructed grey
protons (thesis Tab. `tabcdbins`; the minimum-bias sample has <nu> = 3.7
there and 3.65 in Tab. `tabinelcs2`). `int.005.tex` has no xF = 0.400 row.
dn/dxF is per triggered event of the sample.

## Other NA49 p+Pb tables (added 2026-09-29)

Transcribed from two further theses located by the 2026-09-29 literature
survey (sources kept in `/home/o/NA49/2026/ppb_sources_2026-09-29/`, with
`SOURCES_MANIFEST.tsv` giving URLs and sha256):

| file | rows | content |
|------|------|---------|
| pPb_central_boimska2004_proton.txt | 46 | p, central p+Pb (nu-bar ~ 6.2), E d3N/dp3 per event on xF x pT bins |
| pPb_central_boimska2004_antiproton.txt | 38 | pbar, same sample and quantity |
| pPb_central_boimska2004_pi_minus.txt | 43 | pi-, same sample and quantity |
| pPb_central_boimska2004_pi_plus.txt | 46 | pi+, same sample and quantity |
| pPb_y0_laszlo2008_{pi_plus,pi_minus,proton,antiproton,k_plus,k_minus}.txt | 20, 20, 21, 13, 20, 18 | (1/2 pi pT) d2n/(dy dpT) at y = 0 per event, pT 0.35 to 2.7 GeV/c |

- Boimska: B. Boimska, PhD thesis, Warsaw (2004), CERN-THESIS-2004-035,
  Tables C.5 to C.8 (Appendix C.2). The PDF is a scan without a text
  layer; the values were read by eye and confirmed by a second read of a
  zoomed render in every cell, and they pass monotonicity and wide-bin
  checks. The table prints values x 10^3; the files carry the physical
  values. xF bins are ranges as printed (0.0-0.1 and 0.1-0.3 are wider
  bins used at high pT). Only the most central class is tabulated
  (nu-bar ~ 6.2 of the four grey-proton classes 3.1, 4.6, 5.5, 6.2). The
  error convention (reported as statistical only) and the absence of a
  feed-down correction come from the survey and were not re-read.
- Laszlo: A. Laszlo, PhD thesis, ELTE Budapest (2007, posted 2008),
  Table 12, parsed from the PDF text layer by word position (eight cells
  checked against the rendered table). The sample is the full p+Pb data
  set with varied centrality trigger settings, <N_W> = 5.58 and
  <N_BC> = 4.58 (Table 8); it is not minimum bias.
- No NA49 table gives minimum-bias p+Pb protons, antiprotons, or net
  protons, or protons for the intermediate centrality classes; those exist
  only as figures (see `_knowledge/foundations/ppb-data-sources.md`).

## Format

Spectra files (`xsect_*`, `pPb_dens_*`): one row per data point,

    xF  pT[GeV/c]  value  abs_error

sorted by xF ascending, then pT ascending. xF runs over the thesis grid
0.00..0.40 in steps of 0.05; pT covers roughly 0.0..2.1 GeV/c.

Integrated files (`ptintegrated_*`): one row per xF bin,

    xF  dn/dxF  err_abs  mean_pT[GeV/c]  err_abs  mean_pT2[(GeV/c)^2]  err_abs

where dn/dxF is the per-event pT-integrated density of the thesis (it is
not divided by sigma_inel); mean_pT is <pT>, mean_pT2 is <pT^2>.

All lines starting with `#` are comments documenting quantity, system,
sample, error treatment, source table label, conversion date, and any
source anomalies carried into that file.

## Source anomalies (preserved as-is, also flagged in file headers)

1. Duplicated pT labels. In most tables the lowest pT rows appear twice:
   labels 0.10 (two consecutive source rows) and 0.20 (two consecutive
   source rows), e.g. `ictable.111.piplus.dat` rows 2/3 both start `0.10`
   and rows 4/5 both start `0.20`. Whether these are distinct adjacent bins
   that were rounded to the same printed label cannot be decided from the
   datatables alone; the duplicates are preserved as separate points.
   Affected: all converted files EXCEPT `pPb_dens_minbias_*` and
   `pPb_dens_set1_piminus` (see grid note below).
2. Two different pT grids circulate among the tables:
   - fine grid: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, then 0.1 steps to 2.10
     (used by ictable.000.* and ictable.001.piminus);
   - coarse/doubled-label grid: 0.00, 0.10, 0.10, 0.20, 0.20, 0.30, ...
     (used by ictable.001.piplus, 002..005 and 111).
   Comparisons across charges within a set may therefore pair different pT
   grids (e.g. set 1 pi+ vs pi-).
3. Centrality set 5 covers only xF = 0.00..0.35 (17-column thesis tables),
   hence fewer rows (138/136).
4. A few empty high-pT cells in the source (e.g. `ictable.111.piminus.dat`
   has no entry at pT=2.10, xF=0.15, which is why the pi- cross-section file
   has 154 rather than 155 rows). Empty cells are skipped silently, as in
   the printed thesis tables.
5. Thesis typo: in Tab. `tabicmb` the mid-table header over the pi- block
   reads d(xF,pT)/Delta d while the caption defines f(xF,pT)/Delta f for
   both charges; the caption governs (both blocks are invariant cross
   sections).
6. Cosmetic: `ictable.005.*.dat` print pT labels with one decimal ("0.1")
   where other tables use two ("0.10"); values are identical numerically.
7. Negative printed error. `int.002.tex` (centrality set 2) prints the
   dn/dxF statistical error of pi+ at xF = 0.400 as -1.0 %; the neighbouring
   points carry 3 to 6 %. The value is preserved as printed in
   `ptintegrated_piplus_pPb_set2.txt` (negative absolute error) and flagged in
   its header; treat that error as unknown.

## Validation summary (conversion run of 2026-08-25)

For every file: row count equals an independent recount of non-empty source
cells (or source data rows for the integrated tables); all values > 0; five
randomly picked records per spectra file (and two rows per integrated file)
were re-extracted from the raw LaTeX text and compared cell by cell against
the converted output, all matching exactly. Reported ranges: xF within
[0.00, 0.40] ([0.00, 0.35] for set 5); pT within [0.0, 2.1] GeV/c
([0.05, 2.1] on the fine grid).
