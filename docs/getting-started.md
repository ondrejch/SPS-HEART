# Getting started

## Installation

HEART is pure Python (3.9 or later) on NumPy and SciPy. From the repository root:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -e dev/mixing2006            # the generator package na49_mixing
python -m pip install -e "dev/mixing2006[dev]"     # adds pytest for the tests
```

The fit driver and the event writer under `helpers/benchmarking/` run from the repository root and find the package
through their own path setup, so they work with or without the editable install.

## Tests

```bash
cd dev/mixing2006 && python -m pytest
```

This runs the generator's tests and, through `tests/test_helpers_gate.py`, the tests of the fit driver. Tests that
read the SPS tables carry the `external_data` marker. The tables are part of this repository, so
`python -m pytest -m external_data` from `dev/mixing2006` runs them as well. For quick feedback,
`NA49_TEST_TIER=fast` skips the slow tests listed in `slow_tests.txt`. [snapshot.md](snapshot.md) lists any test
module that this tree cannot run.

## Generating events

`helpers/benchmarking/heart_events.py` generates inelastic events at a fitted point and writes them as JSON Lines.
The point is read from a fit report, so the events are those of the published model:

```bash
python helpers/benchmarking/heart_events.py \
    --spec dev/mixing2006/config/runs/r146.json \
    --report dev/00runs/r146_regenerated_2026-11-06/report.json \
    --events 10000 --workers 8 --seed 1 --out heart_events.jsonl
```

| Option | Meaning |
|---|---|
| `--spec` | the run description, which fixes the kernel (catalog, continuum, decay laws) |
| `--report` | a fit report; its `fitted` block is the generation point |
| `--events` | number of inelastic events |
| `--seed`, `--first` | the root seed and the index of the first event; a (seed, index) pair always gives the same event |
| `--final` | `observed` (default) performs the weak decays the NA49 tables count; `generator` leaves every weak decay undone |
| `--workers` | worker processes; the sample is identical for any number of workers |

Each worker builds the decay kernel once, which takes one to two minutes. A core then generates about six events per
second. A sample is extended without overlap by a second call with `--first` set to the number of events already
written.

## A first analysis

The events are written in the laboratory frame. Feynman x and transverse momentum follow from the boost to the
center-of-mass frame ([event-format.md](event-format.md)):

```python
import json, math
import numpy as np

lines = open("heart_events.jsonl").read().splitlines()
head = json.loads(lines[0])
sqrt_s = head["sqrt_s_gev"]
m_p = 0.938272
e_lab = math.hypot(158.0, m_p) + m_p                     # beam proton on a proton at rest
beta, gamma = 158.0 / e_lab, e_lab / sqrt_s

xf_pim, n_events = [], 0
for line in lines[1:]:
    n_events += 1
    for pdg, E, px, py, pz, *_ in json.loads(line)["particles"]:
        if pdg == -211:
            pz_cm = gamma * (pz - beta * E)
            xf_pim.append(2.0 * pz_cm / sqrt_s)

counts, edges = np.histogram(xf_pim, bins=20, range=(0.0, 1.0))
dn_dxf = counts / n_events / np.diff(edges)              # pi- per inelastic event per unit x_F
print(np.round(dn_dxf, 4))
```

The NA49 invariant cross sections f(x_F, p_T) = E d³σ/dp³ that the fit compares with are normalized to the
inelastic cross section σ_inel = 31.78 mb at this energy. A model density per inelastic event becomes a cross section
by that factor. The fit's own binning, including the bin-center correction, is in `helpers/benchmarking/run_observables.py`.
