# Event format `heart-events/1`

`helpers/benchmarking/heart_events.py` writes JSON Lines in UTF-8. The first line is a header object, and every
following line is one inelastic event. A file is read with any JSON parser, one line at a time.

## Header

```json
{"format": "heart-events/1",
 "spec": "dev/mixing2006/config/runs/r146.json",
 "report": "dev/00runs/r146_regenerated_2026-11-06/report.json",
 "run_id": "r146_regenerated_2026-11-06",
 "reaction": "p+p", "beam_momentum_gev_c": 158.0, "sqrt_s_gev": 17.270205201372406,
 "frame": "lab (target at rest, beam along +z)", "units": "GeV",
 "seed": 1, "events": 1000, "final_state": "observed",
 "class_fractions": {"dd": 0.0472, "nd0": 0.5771, "sd_projectile": 0.1878, "sd_target": 0.1878},
 "isovector_fraction": 1.6e-09,
 "particle_columns": ["pdg", "E", "px", "py", "pz", "mass", "B", "Q", "S", "I3x2", "line"]}
```

| Field | Meaning |
|---|---|
| `spec`, `report`, `run_id` | the run description and the fit report the events were generated from |
| `sqrt_s_gev` | the center-of-mass energy of the collision |
| `frame`, `units` | four-momenta in the laboratory frame, in GeV (c = 1) |
| `seed`, `events` | the root seed and the number of events in the file |
| `final_state` | `observed` or `generator` (see below) |
| `class_fractions` | the event-class fractions of the fit; `nd0` is the non-diffractive class |
| `particle_columns` | the column order of every particle row |

## Events

```json
{"event": 0, "class": "nd", "particles": [[211, 5.5505, 0.3804, 0.0161, 5.5356, 0.13957, 0, 1, 0, 2, "projectile"], ...]}
```

`event` is the event index within the seed. `class` is the event class: `nd` (non-diffractive), `sd_projectile`
(single diffraction, the beam proton excited), `sd_target` (single diffraction, the target proton excited), or `dd`
(double diffraction). Events are unweighted. Each event's class is drawn with the header's class fractions, so a
sample is a sample of inelastic events.

Each particle row holds:

| Column | Meaning |
|---|---|
| `pdg` | the PDG Monte Carlo code; `null` for a low-mass baryonic excitation with no open decay channel, kept as one system |
| `E`, `px`, `py`, `pz` | the four-momentum in the laboratory frame [GeV]; the beam proton moves along +z with 158 GeV/c, and the target proton is at rest |
| `mass` | the particle's mass [GeV] |
| `B`, `Q`, `S` | baryon number, charge, and strangeness |
| `I3x2` | twice the third isospin component |
| `line` | the origin of the particle: `projectile` or `target` for the two lines of a non-diffractive or double-diffractive event, `excitation` or `remnant` for the excited system and the intact proton of single diffraction |

Every event conserves energy, momentum, baryon number, and charge: the rows sum to E = 158.941 GeV, p_z = 158 GeV/c,
p_x = p_y = 0, B = 2, and Q = 2.

## The final state

The strong and electromagnetic decays of every resonance are performed, so the rows are long-lived hadrons, photons,
and π⁰. The π⁰ is kept undecayed. The two values of `final_state` differ in the weak decays:

- `generator` leaves every weak decay undone. The rows include Λ, Σ±, Σ⁰, Ξ, Ω, K⁰, and their antiparticles.
- `observed` (the default) performs the decays that the NA49 tables count. These are Σ⁰ → Λγ, the Ξ⁰ and Ξ⁻ to Λπ,
  and the Ω⁻ to ΛK⁻, Ξ⁰π⁻, and Ξ⁻π⁰, with their antiparticles. The Λ, Σ±, K⁰_S, and K⁰_L stay undecayed, because the
  NA49 tables subtract their decay products by the feed-down correction. This is the final state that the fit
  compares with the data. The decays are listed in `dev/mixing2006/config/v2_singlet/counted_decays_pdg2025.json`.

Neutral kaons appear in their strangeness states, K⁰ as 311 and K̄⁰ as −311. A decay table that lists a K⁰_S K⁰_L
pair, as that of the φ does, writes the codes 310 and 130. The generator reads 310 as the K⁰ and 130 as the K̄⁰ of
such a pair (`na49_mixing/resonance_decay.py`). A K⁰_S yield is half the sum of the K⁰ and K̄⁰ yields.

## Kinematic variables

The fit's variables are Feynman x, x_F = 2p_z*/√s with p_z* the longitudinal momentum in the center-of-mass frame, and
the transverse momentum p_T = (p_x² + p_y²)^(1/2). The boost from the laboratory frame takes β = p_beam/E_lab and
γ = E_lab/√s, with E_lab = (p_beam² + m_p²)^(1/2) + m_p:

```python
pz_cm = gamma * (pz - beta * E)
x_F = 2.0 * pz_cm / sqrt_s
```

Densities per inelastic event (dn/dx_F, the invariant density) follow by dividing the counts by the number of events
in the file. The NA49 invariant cross sections f = E d³σ/dp³ are the invariant densities times σ_inel = 31.78 mb
([getting-started.md](getting-started.md)).

## Reproducibility

An event is fixed by the seed, its index, the run description, the fit report, and the code. The class draw uses
`numpy.random.default_rng([seed, index, 1])`. The event itself uses the five named substreams of
`na49_mixing.rng_substreams.derive_substreams(seed, 0, 10⁹ × class index + index)`. A file written with any number of
worker processes is therefore identical, and a sample continued with `--first` adds new events without repeating any.
