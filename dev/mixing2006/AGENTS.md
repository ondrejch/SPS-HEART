# mixing2006 Agent Instructions

## Repo Focus

`mixing2006` is the active Python port of the NA49 resonance-analysis and
event-mixing framework.

- Primary code lives in `na49_mixing/`.
- Benchmarks and scan drivers live in `helpers/benchmarking/`.
- The staged paper workflow lives in `sps_resonances_paper/`.
- The original C++/ROOT implementation is preserved in `.legacy/` for
  reference and cross-checks.

Treat `.legacy/` as read-only unless the user explicitly asks for legacy
maintenance.

## Working Surface

Key paths:

- `na49_mixing/`: physics and analysis package
- `na49_mixing/generators/`: bundled `.gen` cards used by examples and fits
- `examples/`: lightweight runnable scripts
- `tests/`: focused pytest coverage for kinematics, decays, mixing, and
  chunked generation
- `helpers/benchmarking/`: SPS benchmarking, constrained fits, and scan scripts
- `sps_resonances_paper/`: numbered end-to-end paper pipeline stages
- `00runs/`: default output root for generated artifacts
- `external/`: local build/install area for external generators used by the
  paper workflow

Work from the repository root unless a script explicitly expects another
directory.

## Entry Points

Editable install:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m pip install -e ".[hdf5]"
```

Examples:

```bash
python examples/generate_deltas.py --help
python examples/generate_rho_omega.py --help
python examples/plot_spectra.py --help
```

Focused validation:

```bash
pytest tests/test_reaction.py -v
pytest tests/test_soft_resonance_decays.py -v
pytest tests/test_parallel_generation.py -v
pytest
```

Paper workflow:

```bash
cd sps_resonances_paper
./04-run_balanced_constrained_fit/run.sh
./run_all.sh
```

## Important Repo-Specific Constraints

- The `na49-generate` console script is implemented in `na49_mixing/_cli.py`.
  Prefer small `--nevents` smoke runs and write under `00runs/`.
- `examples/generate_deltas.py` needs an explicit `--genfile` path from
  `na49_mixing/generators/` and a decay table, typically
  `.legacy/resonance-picture/michal/deltas-to-pion/decays`.
- Paper and benchmark scripts assume repo-root-relative paths for `00runs/`,
  `external/`, and helper config JSON files.
- Full SPS scans are expensive. Prefer focused tests, `--help`, single-stage
  runs, or dry runs unless the user asks for full production.

## Validation Strategy

- Test tiers (dev loop): the full suite runs only at the land gate.
  The targeted set for a change is
  `python helpers/benchmarking/select_tests_for_changes.py <base-ref>`
  from the monorepo root; nested paths run from this tree, helper paths
  from the monorepo root with `PYTHONPATH=.`. In-flight feedback adds
  `NA49_TEST_TIER=fast` (deselects the slow tier in the monorepo-root
  `slow_tests.txt`).
- Start with the smallest test or script that exercises the changed surface.
- For physics changes, verify invariants and targeted pytest modules before
  broader benchmark runs.
- For example or README changes, use small smoke-test event counts and write
  outputs under `00runs/`.
- For `helpers/benchmarking/` and `sps_resonances_paper/`, prefer single-stage
  checks or wrapper dry runs before launching scans.
- Report what you ran and what you did not run.

## Dual `helpers/benchmarking` trees

There are two helper trees:

1. monorepo top-level `helpers/benchmarking/` (gateway + project-root workflows)
2. vendored `dev/mixing2006/helpers/benchmarking/` (paper standalone pipeline)

They must stay **logically** in sync. Path defaults intentionally differ
(`dev/00runs` vs `00runs`, mixing2006 package location). After changing one
tree, either re-vendor or run:

```bash
python helpers/benchmarking/check_helpers_layout_sync.py
```

from the monorepo root. Layout-only path diffs are allowed; logic drift fails.

## Shared benchmark library

Prefer new benchmark logic in:

- `helpers/benchmarking/sps_core.py` — table I/O, paper window, metrics, NNLS helpers

The long CLI drivers (`benchmark_sps_species.py`, `benchmark_multispecies_constrained.py`)
should import from `sps_core` rather than grow more private copies of the same
math. Golden tests:

```bash
PYTHONPATH=. python -m pytest helpers/benchmarking/tests/ -q
```

## Repo-Local Skills

Use the repo-local skills in `skills/` when the task matches:

- `physics-validation`: changes in `na49_mixing/`, decay logic, kinematics,
  fragmentation mode, or benchmark physics outputs
- `example-smoke-test`: changes in `examples/`, README commands, packaging
  paths, or lightweight artifact generation
- `sps-paper-pipeline`: changes in `sps_resonances_paper/` or paper-facing
  benchmark orchestration

See `SKILLS.md` for the index.
