# Staged Excitation Implementation

This release adds a separate event-level path beside the legacy
resonance-template benchmark.

## Gates

- **simple_excitation** uses an empirical symmetric source-mass density and
  validates p+p source construction, decays, ancestry, thresholds, and exact
  conservation.
- **regge_fit** switches to the Regge-informed normalized source-mass and
  transfer law. Only parameters declared in the stage-2 allow-list can be
  fitted. `alpha_p0` is intentionally fixed at one beam energy because it is
  an overall normalization in the normalized event sampler. The same
  identifiability logic applies to the transfer slope: at one beam momentum the
  normalized t distribution depends only on the effective transfer slope
  `B_eff = B + 2 alpha'_P ln(s/s0)`, so `regge.transfer_slope` and
  `regge.alpha_p_prime` are refused as separate free parameters and
  `regge.effective_transfer_slope` is the single fittable transfer degree of
  freedom. Extracting `alpha'_P` requires an energy lever arm or independent
  transfer data at more than one beam momentum. The stage-2 fit carries the
  binned six-species objective (`fit.objective: "binned"`, the bundled
  default): a diagonal per-bin chi-square over the six NA49 p+p 158 GeV/c
  spectra (`pi-`, `pi+`, `K+`, `K-`, `p`, `pbar`) in cross-section-per-
  inelastic-event units (`f / sigma_inel` with `sigma_inel = 31.78` mb)
  over the paper window (xF 0.0-0.4, pT 0.05-1.5 GeV/c), with the tabulated
  per-bin uncertainties and the invariant `E/pT` track weighting. All six
  spectra consume one shared event sample per objective evaluation and one
  shared parameter set. The diagnostic aggregate objective against summary
  targets remains available (`fit.objective: "aggregate"`).
- **baryon_nucleus** freezes the accepted p+p parameters and uses the existing
  Glauber geometry to generate one final projectile source plus one target
  source per wounded target nucleon.

**Status.** Three validation statuses are named and separated for this
model. Conservation and algorithm validation are established: exact
four-momentum closure and exact integer B, Q, and S hold for every generated
event, and every production channel opens only above its physical
threshold. Calibration to experimental observables is not established. The
stage-2 fit carries the binned six-species objective (the `regge_fit` entry
above), and a first fit attempt on the six NA49 identified p+p 158 GeV/c
spectra is recorded at `dev/00runs/staged_binned_fit_mc_2026-09-06/` (M8
chi-square 44088.09 on 1017 bins, about 43.8 per degree of freedom). That
goodness of fit rejects an acceptable joint description. The source-local
tests of spec section 9 are architecture-level internal consistency checks.
Particle-ratio, seagull, and p+A held-out evaluations at the frozen M8
point fail the strange, baryon, seagull-shape, minimum-bias p+A, and
centrality sectors; the sequential p+A baseline has been tested and has
failed.

## Baseline limitations

The resolved region currently uses source-level non-strange proxy channels
rather than the complete PDG resonance selector. The continuum channel basis
is charge-complete for the two baryonic source types: it implements the
non-strange, associated-strangeness, strangeness-zero strange-pair, and legal
baryon-pair families with charge-balanced extra pions, thresholded openings,
and exact sequential kinematics. This basis demonstrates exact source-local
conservation and legal thresholds. The stage-2 binned six-species fit is
implemented; its first executed attempt on the six NA49 identified p+p
158 GeV/c spectra is recorded and rejected by goodness of fit (the
`regge_fit` entry above; M8 run record
`dev/00runs/staged_binned_fit_mc_2026-09-06/`). The fit-attempt status and
the decay-kinematics status are separate: because the per-family strengths
are recorded priors and the decay kinematics use the retained recursive
empirical kernel, the decay kinematics remain a conservation demonstrator
rather than a production model for identified spectra until the decay-model
comparison resolves them (spec section 3.4.3). The sequential-projectile,
independent-target p+A baseline has been tested against minimum-bias and
centrality observables and has failed. The p+A initial state is an
independent nucleon-at-rest construction with no binding, Fermi motion, or
daughter rescattering.

## Entry point

Source-only checkouts (review archives) do not include the SPS tables
under `spshadrons.web.cern.ch/`. The archive-runnable command disables
the binned fit:

```bash
na49-staged-excitation \
  --config config/staged_excitation_smoke.json \
  --output staged_evaluation_report.json \
  --events-output staged_excitation_events.jsonl
```

The production binned command still uses `config/staged_excitation.json`.
It requires the NA49 SPS tables. When those files are absent the runner
raises a data-preflight `FileNotFoundError` naming the missing paths
instead of failing on the first table open.

```bash
na49-staged-excitation \
  --config config/staged_excitation.json \
  --output staged_evaluation_report.json \
  --events-output staged_excitation_events.jsonl
```

The JSON report contains stage summaries, fitted values, the fit trace,
configuration hash, and the frozen p+p parameter set used for p+A. JSONL event
records include all source and final-particle four-vectors, source-local
quantum numbers, channel ancestry, collision history, deterministic random
substream labels, and conservation residuals.
