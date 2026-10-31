# SPS Resonances Paper Reproduction Pipeline

This directory turns the current NA49 SPS resonance study into a local, step-by-step pipeline.

Repository root for this package is the current `mixing2006` checkout.
The pipeline is designed to run from that repo alone to regenerate plots, scans, and table snippets. The required benchmarking helpers are vendored under `helpers/benchmarking`, runs are written under `00runs`, and external generators are installed under `external`. The paper text is intentionally not vendored here; the canonical manuscript stays in the separate `latex/paper` tree.

The pipeline is organized as numbered stages:

- `00-download_spshadrons`: refresh the SPS HADRONS tables and source documents used by the analysis.
- `01-create_python_env`: create a dedicated virtual environment and install Python dependencies.
- `02-build_external_generators`: bootstrap the HepMC2 and CRMC sources, unpack a user-supplied UrQMD tree, and build the local external generators under `external`.
- `03-run_external_baselines`: generate the local PYTHIA8, EPOS-LHC, and UrQMD baseline CSVs used by the constrained fit.
- `04-run_balanced_constrained_fit`: run the balanced PDG-constrained fit used for the paper tables and core benchmark figures.
- `05-make_bestmix_plots_and_tables`: regenerate the six-species spectra panels, ratio figure, and seagull figure from the balanced run.
- `06-scan_absyield_prior_stiffness`: run the absolute-yield prior scan and summarize it as the paper heatmap.
- `07-scan_table9_fraction`: run the nonlinear Table-9 fraction scan and summarize it as the paper figure.
- `08-scan_nstar_extended_levels`: run the extended N* ladder scans and make the three-way likelihood map.
- `09-study_component_identifiability`: regenerate the identifiability diagnostics for the balanced run.
- `10-scan_nstar_smoothness`: regenerate the N* smoothness-prior scan figure.
- `11-bootstrap_robustness`: regenerate the bootstrap model-band figure.
- `12-assemble_paper`: export regenerated figures and LaTeX table snippets as a paper-asset bundle, and optionally sync them into an external paper tree.
- `13-build_pdf`: optionally compile an external `latex/paper/main.tex` if `PAPER_ROOT` points to that paper tree.

## Entry points

Four scripts run the pipeline from this directory.

- `./run_smoke.sh [OUTPUT_DIR]`: offline acceptance run of the production fit
  path on the committed mini-pipeline fixture under `mini_pipeline/`. The run
  needs no network, no `spshadrons.web.cern.ch` tables, no external
  generators, and no UrQMD. It completes in about a minute and validates the
  artifacts against toleranced reference numbers.
- `./preflight.sh`: environment audit before expensive work. The audit checks
  the required commands, verifies the sha256 of every downloaded SPS file
  against `reproducibility.lock.json` through
  `CHECK_ONLY=1 bash 00-download_spshadrons/run.sh` (no downloads), reports
  free disk space on the `RUNS_ROOT` and `EXTERNAL_ROOT` filesystems against
  `PREFLIGHT_MIN_FREE_GB` (default 20), compares the Python version and the
  pinned package versions with the manifest when the reference venv exists,
  and reports the state of the user-supplied UrQMD input. Each check prints
  one PASS/WARN/FAIL line with a reason. Missing UrQMD and unrecorded
  manifest hashes are warnings; the exit status is non-zero only when a
  failure breaks even the smoke path.
- `./run_full.sh`: full paper reproduction. The script runs `preflight.sh`
  first and aborts on a FAIL verdict, then executes the numbered stages in
  the order listed above with the same `SKIP_EXISTING` semantics as
  `run_all.sh`. Before stage 02 it stops with instructions when the
  user-supplied UrQMD input is missing or its sha256 is unrecorded.
  `RUN_FULL_STAGES="00 01 02"` runs a subset of stages (two-digit numbers or
  full stage names); `--dry-run` prints the stage plan without executing
  anything.
- `./run_all.sh`: legacy whole-pipeline loop, superseded by `run_full.sh` and
  kept unchanged for compatibility.

### UrQMD input and manifest manual steps

UrQMD is not redistributed by this repository, and the pipeline never
downloads it. Full reproduction through stages 02 and 03 needs the licensed
source supplied by the user and three manual entries in
`reproducibility.lock.json`, each of which currently holds a `TODO` marker:

1. `downloads.urqmd_source_tarball.sha256`: place the tarball at
   `external/dist/urqmd-4.0.tar.gz`, or set `URQMD_TARBALL`, or point
   `URQMD_SOURCE_DIR` at an unpacked tree, then record the tarball sha256
   printed by `sha256sum`.
2. `downloads.hepmc2_source_tarball.sha256`: the first run of stage 02
   downloads the HepMC2 archive once, prints its sha256, and stops without
   installing it; record the printed value.
3. `generators.crmc.pinned_commit`: the first run of stage 02 clones the tag
   `v1.7.0-alice1`, prints the resolved commit, and proceeds without pin
   verification; record the printed commit to pin future builds.

Run a single step:

```bash
./04-run_balanced_constrained_fit/run.sh
```

## Configuration

Shared defaults live in `common.sh`.

The most important environment overrides are:

- `PIPELINE_STAMP`: suffix used in run-directory names. Default: `paper_local_repro`.
- `VENV_DIR`: virtualenv location. Default: `<repo>/.venv-sps-resonances-paper`.
- `RUNS_ROOT`: local run-output root. Default: `<repo>/00runs`.
- `EXTERNAL_ROOT`: local external-generator root. Default: `<repo>/external`.
- `PAPER_ASSET_ROOT`: exported figure/table bundle root. Default: `<repo>/00runs/paper_assets_<stamp>`.
- `PAPER_ROOT`: optional external paper tree. If unset, the pipeline auto-detects `../../latex/paper` when it exists; otherwise paper sync/build steps are skipped.
- `CHUNK_COUNT`: number of local external-simulation chunks. Default: `32`.
- `EVENTS_PER_CHUNK`: events per chunk for the external baselines. Default: `5000`.
- `PARALLEL_CHUNKS`: concurrent external-generator chunk jobs. Default: `1`.
- `MAKE_JOBS`: build parallelism for UrQMD/HepMC2/CRMC. Default: local CPU count.
- `RES_EVENTS_PER_RES`: resonance MC statistics used by constrained fits. Default: `120000`.
- `RES_WORKERS`: worker count for resonance cache generation. Default: local CPU count.
- `SKIP_EXISTING`: if `1`, completed run directories are reused when possible. Default: `1`.
- `REPRO_LOCK`: if `1` (default), `01-create_python_env` installs the exact package versions pinned in `requirements-paper.lock.txt`; set `0` for library development installs of latest-compatible packages.

Example:

```bash
PIPELINE_STAMP=myrerun PARALLEL_CHUNKS=4 RES_WORKERS=16 ./run_full.sh
```

## Reproducibility manifest

`reproducibility.lock.json` records the reference environment: the Python version and exact package versions, the URL and sha256 of every downloaded input (SPS HADRONS tables and documents, the HepMC2 archive, the user-supplied UrQMD tarball), the pinned CRMC commit, the generator build options, the source-tree commit IDs, and the reference random seeds and configuration file hashes. `requirements-paper.lock.txt` pins the matching Python package versions and is what `01-create_python_env` installs by default.

Download stages write to a temporary file, verify the sha256 against the manifest, and rename the file into place only after the check passes. An existing file whose hash differs from the manifest is refused until it is deleted explicitly. `CHECK_ONLY=1` on `00-download_spshadrons/run.sh` verifies the files already present and downloads nothing.

## Main outputs

- External baselines: `00runs/local_multispecies_allmodels_singlepass_pythia_<stamp>`
- Balanced fit: `00runs/constrained_refit_pdg_absyield_scan_l0300_s0050_<stamp>`
- Exported paper assets: `00runs/paper_assets_<stamp>`
- Vendored benchmark helpers: `helpers/benchmarking`
- Optional synced paper figures: `<PAPER_ROOT>/fig`
- Optional synced table snippets: `<PAPER_ROOT>/generated`
- Optional paper PDF: `<PAPER_ROOT>/main.pdf`

## Runtime note

The costliest steps are `03`, `06`, `07`, and especially `08`. The N* extended-level scan is a full local grid and is expected to be the longest stage by a large margin.
