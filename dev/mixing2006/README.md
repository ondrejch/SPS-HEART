# mixing2006: NA49 Resonance Analysis Framework

`mixing2006` is the active pure-Python port of the NA49 resonance-analysis and
event-mixing framework. It replaces the original C++/ROOT runtime with
NumPy/SciPy/matplotlib while keeping the physics model, generator cards, and
legacy reference implementation in the same repository.

The repository is centered on:

- resonance production from `.gen` cards,
- two- and three-body decay chains,
- event mixing and invariant-mass spectra,
- fixed-target SPS benchmarking,
- local reproduction of the current SPS resonance paper workflow.

No ROOT installation is required for the Python package.

## Status

- Active development happens in `na49_mixing/`.
- The historical C++/ROOT code is preserved under `.legacy/` for reference and
  cross-checks. This tree is present in development checkouts and excluded
  from review archives.
- The supported command-line surface in this checkout is the example and helper
  scripts under `examples/`, `helpers/benchmarking/`, and
  `sps_resonances_paper/`.

## Repository Layout

- `na49_mixing/`: core package
- `na49_mixing/generators/`: bundled generator cards used by examples and
  benchmarks
- `examples/`: small runnable entry points for generation and plotting
- `tests/`: pytest suite for kinematics, decays, mixing, and parallel helpers
- `helpers/benchmarking/`: benchmark and scan scripts used in SPS studies
- `sps_resonances_paper/`: staged end-to-end reproduction pipeline for the
  current paper workflow
- `.legacy/`: archived C++/ROOT implementation and original analysis assets.
  Present in development checkouts and excluded from review archives.
- `spshadrons.web.cern.ch/`: NA49 and SPS reference data tables. Present in
  development checkouts and excluded from review archives.
- `external/`: local build area for optional external generators used by some
  stages of `sps_resonances_paper/`. Present in development checkouts and
  excluded from review archives.
- `00runs/`: default location for generated run artifacts; quick-start
  commands create their own subfolders on demand

## Main Capabilities

- fixed-target kinematics via `Reaction`
- built-in PDG particle database and decay tables
- generator-driven resonance production from shipped `.gen` files
- two-body and three-body decays, including soft-resonance decay factories
- event-mixing buffers and invariant-mass spectrum utilities
- deterministic chunked parallel generation in `na49_mixing.parallel`
- fragmentation-side control with `projectile`, `target`, and `both`
- NumPy `.npz` persistence, with optional HDF5 support via `h5py`

## Installation

From the repository root:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[hdf5]"
```

Requirements:

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`
- optional: `h5py` for HDF5 I/O

## Quick Start

What you can run depends on what your checkout carries. A source release
archive ships the Python package, the bundled generator cards, and the example
scripts. Development checkouts additionally carry historical decay tables,
reference data tables, and an `external/` build area. The two sections below
separate the command sets accordingly.

### Archive-Safe Quick Start

Everything in this section runs on a clean extraction that completed the
Installation steps above. All inputs are bundled with the package, and all
outputs stay under `00runs/tmp/quickstart/`. Each command uses one process,
a fixed seed, and event counts of a few hundred, so the sequence finishes in
seconds and produces identical results for identical seeds.

Create the output folder:

```bash
mkdir -p 00runs/tmp/quickstart
```

Generate resonance-cascade tracks from the bundled `delta++.gen` card. The
soft-resonance decay factory supplies all decay tables internally, so the run
reads no files beyond the package:

```bash
na49-generate \
  --genfile na49_mixing/generators/michal/deltas-to-pion/delta++.gen \
  --outfile 00runs/tmp/quickstart/delta_pp_tracks.npz \
  --nevents 300 \
  --seed 20260404 \
  --n-workers 1 \
  --use-soft-decay-factory
```

The output stores per-track `pdg`, `xf`, and `pt` arrays together with run
metadata.

Generate a Delta(1232) histogram sample. When `--use-soft-decay-factory` is
set, the example builds its decay tables from
`build_soft_resonance_decays()` instead of reading an external table:

```bash
python examples/generate_deltas.py delta++ \
  --genfile na49_mixing/generators/michal/deltas-to-pion/delta++.gen \
  --use-soft-decay-factory \
  --nevents 500 \
  --seed 424242 \
  --outfile 00runs/tmp/quickstart/delta_pp.npz
```

Plot the parent and daughter x_F distributions of that sample:

```bash
python examples/plot_spectra.py \
  --files 00runs/tmp/quickstart/delta_pp.npz \
  --labels Delta++ \
  --out 00runs/tmp/quickstart/delta_pp.pdf
```

Generate rho(770) and omega(782) samples with their dominant pion decays:

```bash
python examples/generate_rho_omega.py \
  --nevents 200 \
  --seed 12345 \
  --outfile-rho 00runs/tmp/quickstart/rho770.npz \
  --outfile-omega 00runs/tmp/quickstart/omega782.npz
```

Every command above reads only bundled generator cards and built-in decay
factories, and every artifact lands under `00runs/tmp/quickstart/`.

### Full Development Checkout

Development checkouts carry additional trees that review archives exclude.
These are the additional trees:

- `.legacy/resonance-picture/`: original resonance-picture materials,
  including the historic Delta decay tables read by some Python workflows.
- `spshadrons.web.cern.ch/`: NA49 and SPS reference tables published by the
  HADRONS working group, including `data/xsect_piminus.txt`.
- `external/`: build area for optional external generators used by parts of
  the paper pipeline in `sps_resonances_paper/`.

With these trees present, two further workflows open up.

Fill a Delta sample from the historic decay tables rather than from the
built-in factory:

```bash
python examples/generate_deltas.py delta++ \
  --genfile na49_mixing/generators/michal/deltas-to-pion/delta++.gen \
  --decays .legacy/resonance-picture/michal/deltas-to-pion/decays \
  --nevents 500 \
  --seed 424242 \
  --outfile 00runs/tmp/fullcheckout/delta_pp_legacy.npz
```

Benchmark model hypotheses against the NA49 pi- spectrum stored in the SPS
HADRONS table `xsect_piminus.txt`:

```bash
python examples/benchmark_na49_models.py \
  --na49-data spshadrons.web.cern.ch/data/xsect_piminus.txt \
  --pythia-events 2000 \
  --res-events-per-res 500 \
  --out-dir 00runs/tmp/fullcheckout/na49_models
```

## Python API Example

```python
from na49_mixing.parallel import GenerationConfig, generate_tracks_chunked

cfg = GenerationConfig(
    gen_file="na49_mixing/generators/michal/deltas-to-pion/delta++.gen",
    n_events=20000,
    seed=20260404,
    beam_momentum=158.0,
    max_decay_passes=8,
    use_soft_decay_factory=True,
    fragmentation_mode="both",
)

tracks = generate_tracks_chunked(cfg, n_workers=4, n_chunks=8)
print(tracks["pdg"].shape)
print(tracks["source_side"][:10])
```

`generate_tracks_chunked` returns merged track arrays plus per-chunk metadata,
which is the main programmatic entry point for higher-statistics production.

## Tests

Run the full suite from the repository root:

```bash
pytest
```

Useful focused runs:

```bash
pytest tests/test_parallel_generation.py -v
pytest tests/test_soft_resonance_decays.py -v
```

## Benchmarks And Paper Reproduction

Low-level SPS benchmark helpers live in `helpers/benchmarking/`.

For the maintained end-to-end paper workflow, use the staged pipeline in
`sps_resonances_paper/`:

```bash
cd sps_resonances_paper
./run_all.sh
```

Run a single stage directly when needed:

```bash
cd sps_resonances_paper
./04-run_balanced_constrained_fit/run.sh
```

That pipeline manages:

- local Python environment creation,
- optional external-generator builds under `external/`,
- baseline CSV production,
- constrained fits and parameter scans,
- plot and table export for the paper.

See `sps_resonances_paper/README.md` for stage-by-stage details and environment
overrides.

## Legacy Reference

The `.legacy/` tree preserves the original ROOT-based implementation and the
original resonance-picture assets used for cross-checking the Python port.
Development checkouts contain this tree, and review archives exclude it. A
clean extraction therefore carries everything the Archive-Safe Quick Start
needs and nothing that points into `.legacy/`.

## License

GPL-2.0-or-later

## Staged Color-Neutral Excitation Evaluation

The new event-level path is intentionally separate from the frozen
resonance-template benchmark. It progresses through three gated stages:

1. `simple_excitation`: empirical symmetric p+p source masses, exact two-source
   kinematics, legal source-local decays, and conservation audits.
2. `regge_fit`: Regge-informed mass and transfer shapes, with only declared
   stage-2 parameters unlocked for an optional bounded fit.
3. `baryon_nucleus`: the fitted p+p physics is frozen and transferred to a
   Glauber p+A event with multiple target sources and one sequentially excited
   projectile source.

Run the bundled small configuration:

```bash
na49-staged-excitation \
  --config config/staged_excitation.json \
  --output 00runs/tmp/staged_excitation_report.json \
  --events-output 00runs/tmp/staged_excitation_events.jsonl
```

Set `"through"` to `"simple_excitation"`, `"regge_fit"`, or
`"baryon_nucleus"` to stop at a gate. The Regge fit consumes observable
targets from the same JSON file. At fixed beam energy `alpha_p0` remains fixed:
it changes only the normalization of the normalized event law and is therefore
explicitly rejected as a fit parameter.

The p+A baseline uses independent target nucleons at rest (no binding or Fermi
motion), long formation time, no daughter rescattering, exact sequential
two-body closure, and source-local `(B,Q,S)` conservation.
