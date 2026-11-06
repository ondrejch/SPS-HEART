# The fit

The published model is the result of one fit, described by `dev/mixing2006/config/runs/r141.json`. Its run
directory is `dev/00runs/r141_aligned_lines_2026-11-05/`, and its result record is
`dev/00runs/paper_record_r141/RECORD.json`.

## How the fit works

The fit generates events at a point in parameter space and evaluates the objective at nearby points by
reweighting those same events. Every parameter the fit varies enters an event through recorded quantities:

- the flavor weights through a ledger of the flavor content at every continuum vertex;
- the leading-split shares and direction laws through the split records of the roots;
- the isovector fraction through each event's exchange configuration;
- the class fractions and the remnant mixture through the event's class and remnants.

The weight of an event at a trial point is therefore exact. When the optimum lies far enough from the generation
point that the effective sample size falls, a new generation of events is produced at the optimum, up to four
generations. The objective is a χ² over the NA49 double-differential cross sections of the six species and the
neutron dn/dx_F. It carries the correlated systematic uncertainties of the NA49 papers as profiled scale factors, a
5% model floor, and the Monte Carlo error of the prediction.

## The steps

The fit driver is `helpers/benchmarking/run_engine.py`. Each call does one step:

```bash
python helpers/benchmarking/run_engine.py --spec dev/mixing2006/config/runs/r141.json --<step> [--run-dir DIR] [--workers N]
```

| Step | What it does |
|---|---|
| `--smoke --run-dir DIR` | generates a few events per class at the first generation point in a scratch directory, a check of the installation |
| `--pre-registration` | writes the seeds, the first generation point (from the earlier fit named in `generation_0`), and the digests of the code and frozen inputs, before any event exists |
| `--generate` | generates every class and replica block of the latest registered generation |
| `--evaluate` | fits the latest complete generation by reweighting, registers the next generation or closes the branch |
| `--profile` | runs the variant fits and composes `report.json` and `REPORT.md` |
| `--gate` | writes the record of the hypothesis test with the fitted values and their errors |
| `--clean-validate` | repeats the evaluation of the final generation and the variants in a clean environment and checks that every hash agrees |
| `--seal` | writes the manifest and the checksum inventory of the run, and refuses when any check of the previous step failed |

Without `--run-dir` the steps act on `dev/00runs/<run_id>/`. The generation and evaluation steps check the digests
of the frozen inputs against the pre-registration and refuse to run when they changed.

## Compute

One generation of the paper's fit is 32 replica blocks of 100,000 events in each of four classes, 12.8 million
events. A core generates about six events per second, so a generation takes about 600 core-hours. The fit of the
paper took four hours on twelve 32-core nodes. `helpers/benchmarking/orchestrate_gateway_joint_fit.py` runs the
steps on a cluster reached over SSH through the client in `helpers/na49_gateway/`, with the cluster described in a
configuration file of the form of `helpers/na49_gateway/config.example.json`. The client's README describes the
setup. On a single machine the steps run unchanged with `--workers`, at the compute cost above.

The `--smoke` step and `heart_events.py` run on a laptop.

## Outputs

The run directory holds:

| File | Contents |
|---|---|
| `pre_registration.json` | the seeds, the first generation point, and the input digests, written before generation |
| `slot_*.npz`, `slot_*.json` | the generated events of each class, replica block, and generation in compact form (tracks with x_F, p_T, and weight; the flavor ledger; the split records) |
| `fit_report_f0_g*.json`, `branch_report_f0.json` | the fit of each generation and the closed branch |
| `report.json`, `REPORT.md` | the result: the fitted values, the Hessian errors, the objective by segment, the profiled scales, the tests, the variants, and the readouts |
| `seal_manifest.json`, `SHA256SUMS.txt`, `provenance.json` | the record of the run's integrity and inputs |

The event files of the paper's fit take about 8 GB. They are not in this repository, and a rerun regenerates them
from the seeds.

The keys of `report.json` that most users need are `fitted` (the optimum), `hessian` (the errors),
`objective_at_optimum` (χ² by species and the scale penalty), `variants` (the systematic variant fits), and the
`readout_*` entries (total yields, π⁰, neutrons, and the transverse-momentum profiles).

## The result record

`dev/00runs/paper_record_r141/RECORD.json` is written by `helpers/publishing/make_paper_record.py` in the development
checkout. It names the run that produced every number of the papers and carries enough of the report to check them
without the events: the run identity and the hashes of its spec, report, and manifest; the code commits the fit ran
on; the fitted parameters with their errors; the objective by segment, checked to add to the total; the profiled
scales; the tests and the variants; the readouts; and the hashes of every file the papers quote.
