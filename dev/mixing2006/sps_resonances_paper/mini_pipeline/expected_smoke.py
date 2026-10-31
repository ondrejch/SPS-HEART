"""Pinned reference numbers and tolerances for the mini-pipeline smoke run.

The numbers below were recorded from one reference run of the exact CLI
invocation performed by ``run_smoke.sh`` (recorded 2026-09-01, reference
environment ``dev/mixing2006/.venv-sps-resonances-paper``, numpy 2.5.2 /
scipy 1.18.1, committed fixture ``mini_pipeline/fixture`` at
``fixture_manifest.json`` state: seed 20260830, 40 events per component,
p+Pb wounded-nucleon namespace, Regge tower disabled, decay-library salt
``2026-08-31.2``). The fixture and this header are regenerated whenever a
change to the fingerprinted ``na49_mixing`` modules supersedes the
committed cache fingerprints, most recently after the REV-CUM-01 review
fix, which exports the per-channel audit tables through the chunked
generation path in ``na49_mixing/parallel.py``; the generation stream at
the default configuration is untouched by that change, so every pinned
value reproduced bit-identically at re-recording and only the provenance
state moved (earlier regenerations: P1, salt ``2026-08-31.1``; P2
mesonic staging, salt ``2026-08-31.2``; P2 review fixes, salt
``2026-08-31.2``). Re-recorded 2026-10-28 after the kernel changes since
2026-09-12 moved the decay-library fingerprint at salt ``2026-09-03.3``:
the rebuilt cache shifts several component weights beyond the 5 percent
tolerance (eta 0.086 to 0.042, the K*(892) weights by 10 to 20 percent),
while the global metrics and species WRRMSE stay inside theirs.

The committed component cache pins the random streams, so a rerun on the
same fixture reproduces these numbers to floating-point noise. The
tolerances absorb cross-platform BLAS differences and the small weight
shifts a cache rebuild would produce after a change to the fingerprinted
``na49_mixing`` decay modules; a rebuild that changes the physics at the
Monte-Carlo statistical level (about 1/sqrt(40) per component) exceeds the
weight tolerance on purpose, because in that case the fixture must be
regenerated with ``make_fixture.py`` and the reference numbers re-recorded
here.

Tolerance policy:

- ``WEIGHT_RTOL`` on component weights (5 percent), with ``WEIGHT_ATOL``
  as the floor for weights near zero.
- ``METRIC_RTOL`` on chi2-type metrics (10 percent; these are the most
  sensitive to small weight shifts), ``RMSE_RTOL`` on weighted-rel-RMSE
  scales, ``PULL_RTOL`` on the absolute-yield prior pulls (25 percent).
- ``N_ACTIVE_BAND`` on the count of active weights (NNLS can move a
  component across the activity threshold on a rebuild).
"""

from __future__ import annotations

WEIGHT_RTOL = 0.05
WEIGHT_ATOL = 0.002
METRIC_RTOL = 0.10
RMSE_RTOL = 0.10
PULL_RTOL = 0.25
N_ACTIVE_BAND = 3

# Number of design columns the driver assembles from the committed cache
# (43 specs with the Regge tower disabled, minus the phi_1020 component the
# driver drops because it produces no charged pions).
EXPECTED_N_COMPONENTS = 42
EXPECTED_DROPPED_ZERO_COLUMN_COMPONENTS = ["phi_1020"]

# Species point counts inside the fit window (x_F in [0, 0.4], pT in
# [0.05, 1.5]) from the committed reduced NA49 p+Pb pion tables.
EXPECTED_SPECIES_N_POINTS = {"pi-": 123, "pi+": 123}

# Reference component weights, keyed by component label (the summary's
# weights_by_label_sum; a label sums the tower-split subcomponents, which
# do not occur in this fixture). Only components with weight above 0.01 are
# pinned; smaller weights sit inside the rebuild noise band.
EXPECTED_COMPONENT_WEIGHTS: dict[str, float] = {
    "kstar_plus_892": 0.11766243398949712,
    "delta+": 0.11251889929963846,
    "kstar_zero_892": 0.08608245559440576,
    "omega": 0.08165260708705084,
    "kstar_zero_bar_892": 0.07902429708809791,
    "anti_delta_zero": 0.0648114830371319,
    "eta": 0.04232154053504908,
    "rho3_plus_1690": 0.040187771689184706,
    "rho3_plus": 0.031060001360730664,
    "anti_n1535_to_pbar": 0.028386028051175183,
    "delta-": 0.027216179780540157,
    "delta++": 0.02634693599222785,
    "rho_minus": 0.02608616621533797,
    "kstar_minus_892": 0.025798365800838666,
    "f2_1270": 0.02281382225278235,
    "n1520_inclusive": 0.021516484502826515,
    "anti_n1680_inclusive": 0.021189309723523043,
    "rho3_zero_1690": 0.01904140881913368,
    "anti_delta_minus": 0.019015722506120707,
    "anti_n1520_inclusive": 0.016629796435358123,
    "rho3_minus": 0.014765924695736775,
    "f4_2050": 0.01450625403441263,
    "rho3_minus_1690": 0.011836228041949478,
    "rho0": 0.011558387547422519,
    "anti_n1720_to_pbar": 0.010577630110954828,
}

EXPECTED_N_ACTIVE = 33

# Global stacked metrics (summary key resonance.global_metrics).
EXPECTED_GLOBAL_METRICS: dict[str, float] = {
    "full_chi2": 12310.540373803091,
    "full_smape_percent": 61.470043973329425,
    "full_weighted_rel_rmse": 0.3586476014510168,
    "full_log_pull_rms": 8.54856061306883,
}

# Per-species weighted relative RMSE of the shared resonance fit.
EXPECTED_SPECIES_WRRMSE: dict[str, float] = {
    "pi-": 0.40666027374402836,
    "pi+": 0.3173424344061718,
}

# Absolute-yield prior: fixture targets and the fitted pulls recorded from
# the reference run.
EXPECTED_PRIOR_TARGETS: dict[str, float] = {"pi-": 60.0, "pi+": 65.0}
EXPECTED_PRIOR_PULLS: dict[str, float] = {
    "pi-": -3.190295725490131,
    "pi+": -1.4338180409651653,
}
