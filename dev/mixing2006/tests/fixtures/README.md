# P1 Golden Fixtures — Stage 0 baseline (card TASK-2026-08-31)

Seeded golden fixtures captured from the working tree as of 2026-08-31,
BEFORE the phase-P1 decision-path switch (sample-then-retry channel
handling still live in `na49_mixing/event.py::_decay_once`). The working
tree at capture time carried the uncommitted P1 partial work in
`na49_mixing/breit_wigner.py` (truncated-draw helpers),
`na49_mixing/decays.py` (mass-conditioned channel-set builder), and
`na49_mixing/event.py` (resolved_match_mass parameter + audit tables);
the truncated-draw helpers were inert at capture time because the decay
engine had not yet been switched to them.

## Capture configuration

- `RESONANCE_DECAY_LIB_VERSION` (salt) in force at capture:
  `2026-08-26.1`. Phase P1 bumps this salt, so component caches built
  from these fixtures' generation code are superseded by design.
- Base seed: `20260831`; per-component seeds derived with
  `benchmark_sps_species._component_seed(base_seed, spec_key)` (recorded
  per component in the JSON payloads).
- Events per component: 300, one chunk, one worker.
- `fragmentation_mode="both"`, `max_decay_passes=8`,
  `inject_antibaryon_resonances=True`, `weak_feed_policy="inclusive_weak"`,
  `beam_momentum=158.0` — the `benchmark_sps_species.py` resonance
  defaults.
- Component sets: `benchmark_sps_species._resonance_specs_for_particle`
  for PDG -211 (pi-, 18 components) and PDG 211 (pi+, 18 components);
  labels and spec keys are recorded in each payload.
- Captured with `dev/00runs/tmp/p1_fixtures/capture_golden_fixtures.py`
  (run artifact; re-runnable only against the pre-P1 decision path).

## File inventory

- `golden_pi_minus.json` — per-component pi- track counts, mean xF and
  mean pT of the species tracks, decay counters, and the capture
  configuration; sha256 at capture 19d8f7f31af73726... (first 16 hex).
- `golden_pi_plus.json` — same structure for pi+; sha256 at capture
  64916f051d707a12... (first 16 hex).
- `golden_stage0_summary_pi_minus.json` and
  `golden_stage0_summary_pi_plus.json` — Stage-0 summary payloads, one per
  species, from the re-runnable capture script `make_golden_fixtures.py`.
  Each records the capture configuration, the per-component decay counters
  and decay-library fingerprints, and the pre-registered seed-variation
  band (seven replicate seeds per component, 6-sigma multiplier) that the
  inactive-path regression compares against. The same script writes the
  component cache files under `components/`. When the `components/` caches
  are absent or incomplete on a checkout (a source-only archive), the
  stage-0 smoke regenerates the missing species' caches in place through
  this capture path and validates them against these committed summaries;
  the summaries themselves are never rewritten by that regeneration. Rerun
  it with
  `cd dev/mixing2006 && python tests/fixtures/make_golden_fixtures.py`;
  the script is deterministic given the salt and module sources in force,
  and it records that salt in each summary payload.

## Baseline facts at capture

- 0 decay kinematic failures across all 36 component runs (the
  sample-then-retry path absorbed every closed-channel draw by
  re-sampling; this is the behavior the constraint machinery replaces).
- Zero-count species components are expected (charge selection: e.g. a
  delta++ component cannot produce pi- inside the tracked feed-down).

## Inactive-path regression expectations

With mesonic branches zero and no caps, the seeded generation is expected
to reproduce components whose reachable cascade runs only untouched code
paths (single-channel, BW-free decay-table parents) bitwise: the
mass-conditioned fast path is bit-matched to the previous selection and
consumes the same single uniform. Components that reach multi-channel or
BW-capable parents are bounded by a pre-registered 3-sigma Poisson band
on the per-component species track counts, with two documented deviation
causes, both stream-level only because the old retry loop's
accepted-channel distribution already equaled the C(M) renormalization:

1. the truncated, explicitly renormalized Breit-Wigner daughter draw
   replaces the previous 500-bin histogram draw in `_decay_2body`
   (documented deviation, card risk note "Bitwise regression
   feasibility");
2. the removed sample-then-retry loop no longer consumes the extra
   uniforms of re-sampled illegal channels, shifting the random-number
   stream position for parents whose realized masses close a declared
   channel (observed at capture-validation time for the rho3_minus
   component: 744 -> 742 pi- tracks, well inside the band).

The regression test never loosens the tolerance post hoc.
