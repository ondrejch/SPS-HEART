# Inputs

A HEART calculation has four inputs: the run description, the configuration files it points to, the data tables,
and, for event generation at a fitted point, the fit report.

## The run description

A run description is one JSON file under `dev/mixing2006/config/runs/` (schema `na49_mixing.run_spec/1`).
`r146.json` describes the fit of the paper. The other files describe the project's earlier fits, which the tests
read and which record how the model was built up. The fit driver (`helpers/benchmarking/run_engine.py`, `configure()`)
reads it and refuses a file whose sections are inconsistent.

| Section | Contents |
|---|---|
| `identity` | the run identifier and the records it belongs to |
| `statistics` | the number of replica blocks (32), the events per class and block (100,000), and the maximum number of generations (4) |
| `seeds` | the master seed and the event-index partitions |
| `floor_grid` | the continuation floor of the continuum cascade [GeV] |
| `kernel` | the options of the decay kernel (`KernelConfig`, below) |
| `parameters` | every model parameter with its transform, bounds, and `free` flag; 14 of the 17 are fitted |
| `generation_0` | the earlier fit whose optimum is the first generation point, which parameters it supplies, and the values set by judgment |
| `topology` | the exchange and mass-law options: exact transfer ranges, the duality boundaries (2.2 GeV non-diffractive, 1.7 GeV diffractive), the diffractive ξ_max, the Pomeron slope |
| `diffractive_split`, `single_diffraction_window`, `central_component`, `isovector` | the composition of the event classes |
| `fit` | the optimizer settings and the error model: starts, the correlated systematic uncertainties of the NA49 papers, the neutron segment, the bin-center correction |
| `systematic_variants` | the variant fits (the diffractive cross sections moved within their uncertainties, and others) |
| `frozen_extra` | files whose digests are pinned for the run |
| `texts` | the prose of the run's `REPORT.md` |

A new physics variant is a copy of the file with a new `identity.run_id` and the changed entries. The fit driver
records the digests of the description, the code, and the frozen inputs in the run directory, so a run can always be
traced to its inputs.

## Kernel options

`dev/mixing2006/na49_mixing/kernel_config.py` defines `KernelConfig`. Its docstring documents every option. The
`kernel` section of a run description sets them, and `load_kernel()` builds the catalog and the continuum families
from them. Each option added after a fit defaults to the behavior that fit used, so an older run description
reproduces its events. The options of `r146.json` fall into four groups:

- **The continuum and its cascade:** `continuation_floor` and `continuation_ceiling`, the mass law of a continuation
  step (`continuation_mass_law`, `continuation_barrier_l`), the opening threshold `continuation_q_min`, and the
  isospin completion of the families.
- **The catalog:** the PDG nucleon resonances and hyperons, the Ξ and Ω baryons, the light scalar and axial mesons,
  the excited kaons, the η′, the PDG meson modes, and the Dalitz matrix elements (`pdg_*`, `xi_baryons`,
  `omega_baryon`, `light_scalar_mesons`, `axial_isoscalars`, `excited_kaons`, `eta_prime`, `pdg_meson_modes`,
  `dalitz_matrix_elements`).
- **Channel pools:** the baryon pairs (`baryon_resonance_pairs`, `baryon_pair_completion`, `baryon_pair_s_wave`),
  the associated K* pairs, and the floor pools.
- **The decay-angle laws:** the leading-baryon and leading-meson splits (`leading_split_fraction`,
  `leading_split_kappa`, `leading_split_axis`, `meson_split_fraction`, `meson_split_slope`) and the aligned
  continuing baryonic steps (`baryon_line_kappa`, `baryon_line_kappa_scope`).

## Configuration files

`dev/mixing2006/config/v2_singlet/`:

| File | Contents |
|---|---|
| `resonance_catalog.json` | the established hadrons of the catalog and their decay modes |
| `continuum_resonances.json` | the continuum families and their channels |
| `pdg_nucleon_resonances_pdg2025.json`, `pdg_hyperons_pdg2025.json`, `xi_baryons_pdg2025.json`, `omega_baryon_pdg2025.json` | baryon records of the PDG 2025 tables |
| `meson_modes_pdg2024.json`, `light_scalar_mesons_pdg2025.json`, `axial_isoscalars_pdg2024.json`, `excited_kaons_pdg2024.json`, `eta_prime_pdg2025.json`, `isoscalar_dipion_pdg2025.json` | meson records and modes |
| `baryon_table_corrections_pdg2025.json` | corrections of the baryon decay tables |
| `counted_decays_pdg2025.json` | the weak decays the NA49 tables count, performed in the observed final state |
| `external_cross_sections.json` | σ_inel, σ_SD, and σ_DD at 158 GeV/c with their uncertainties and sources |
| `na49_pp_total_yields.json` | NA49 total multiplicities per inelastic event, for a readout outside the fit |
| `likelihood_nuisances.json` | the nuisance terms of the likelihood |
| `seed_ledger.json` | the seed blocks and event partitions used by earlier runs, which a new run must avoid |
| `legacy_v1_manifest.json` | the digest manifest of the frozen inputs |

## Data tables

The fit compares with the NA49 double-differential invariant cross sections of p+p at 158 GeV/c, in the tables of
the SPS HADRONS site, which are part of this repository under `dev/mixing2006/spshadrons.web.cern.ch/data/`. These
are `xsect_piplus.txt`, `xsect_piminus.txt`, `xsect_kaplus_pp.txt`, `xsect_kaminus_pp.txt`, `xsect_proton_pp.txt`,
`xsect_antiproton_pp.txt`, and the p_T-integrated neutron distribution `xsect_ptintegrated_neutron_pp.txt`. The same
directory holds the global interpolations of the SPS HADRONS project and the energy dependence of the kaon yields
(`sdep_*_kaon_pp.txt`).

## The fit report

A fit writes `report.json` in its run directory. Its `fitted` block is the point at which `heart_events.py`
generates events:

| Key | Meaning |
|---|---|
| `epsilon`, `epsilon_nd`, `b_sd`, `intact_vertex_factor` | the mass and transfer laws |
| `sd_tail_share`, `sd_ope_share` | the components of single diffraction |
| `gamma_s_had`, `r_v`, `r_bbar`, `assoc_norm` | the four flavor weights |
| `leading_split_fraction`, `meson_split_fraction`, `leading_split_kappa`, `meson_split_slope` | the leading splits |
| `f_isovector` | the isovector exchange fraction |
| `class_fractions`, `remnant_weights` | the event-class fractions and the non-diffractive remnant mixture |

The report of the paper's fit is `dev/00runs/r146_regenerated_2026-11-06/report.json`. Its contents are described
in [fit.md](fit.md).
