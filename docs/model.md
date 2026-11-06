# The model

This page maps the layers of HEART onto the code. The equations, the parameter values, and the comparison with the
data are in the HEART paper (arXiv: TBD). Module paths are relative to `dev/mixing2006/na49_mixing/`.

## Event classes and the exchange

An inelastic p+p event is drawn in one of four classes. In non-diffractive production both protons are excited. In
single diffraction on the projectile or on the target side one proton stays intact and the other is excited. In
double diffraction both are excited at low mass. The class fractions are external inputs: the single-diffraction
cross section σ_SD(M²/s < 0.05) = 4.9 mb is measured at this energy, and the double-diffraction cross section
σ_DD = 1.5 mb is an estimate (`config/v2_singlet/external_cross_sections.json`).

| Piece | Module |
|---|---|
| class draw and dispatch, the per-event random substreams | `topology_runner.py`, `rng_substreams.py` |
| non-diffractive class (two excited lines, their remnant selection) | `topology_non_diffractive.py`, `topology_remnants.py` |
| single diffraction (triple-Pomeron coherence region, leading-proton tail, one-pion exchange) | `topology_single_diffraction.py`, `one_pion_exchange.py` |
| double diffraction | `topology_double_diffraction.py` |
| mass laws and momentum-transfer laws | `topology_mass_laws.py` |
| the isovector exchange (consistent with zero in the fit) | `isovector_exchange.py` |

The diffractive masses follow the triple-Pomeron form with the Donnachie–Landshoff intercept, ε = 0.0808. The
momentum transfer of every two-body exchange follows exp(Bt) with factorized vertex slopes.

## The excited systems

Each excited system carries definite baryon number, charge, and strangeness. Below the duality boundary
(2.2 GeV on a non-diffractive line, 1.7 GeV on a diffractive one) it is one of the catalogued N* and Δ resonances of
the PDG tables. Above it, it is a member of a resonant continuum family of its quantum numbers, with the Hagedorn
density ρ(M) ∝ exp(M/T_H), T_H = 165 MeV.

| Piece | Module or file |
|---|---|
| the established hadrons and their decay tables | `hadron_catalog.py`, `config/v2_singlet/resonance_catalog.json` and the `*_pdg20*.json` files |
| the continuum families | `spectral_density.py`, `config/v2_singlet/continuum_resonances.json` |
| the kernel options that shape catalog and continuum | `kernel_config.py` (`KernelConfig`, `load_kernel`) |

## The cascade

A continuum state de-excites in two-body steps. At each step it either continues, emitting one established hadron and
remaining a lighter continuum state, or it ends in a direct pair of established hadrons. The channel weight is the
product of the isospin Clebsch–Gordan factor, the spin counting, the decay momentum, and four flavor factors (see
below). The decay angles are isotropic in the state's rest frame, with three exceptions. At the baryonic root of a
line an established meson may leave first along the incoming baryon (the leading-meson split), or the baryon may leave
first beside a mesonic remainder (the leading-baryon split, about the Gottfried–Jackson axis). At a continuing
baryonic step beside a non-strange hadron, the continuing daughter is aligned along the line's beam axis by
exp(κ_L cos θ), κ_L = 1.5.

| Piece | Module |
|---|---|
| the continuum cascade, the splits, the decay angles | `cascade_decay.py` |
| the end of a cascade in established hadrons | `cascade_termination.py` |
| the decays of the established hadrons (PDG tables, Dalitz distributions) | `resonance_decay.py`, `dalitz_matrix_elements.py` |
| the flavor factors γ_s, r_V, r_BB̄, a_YK | `flavor_weights.py` |
| the decays the NA49 tables count (Σ⁰, Ξ, Ω) | `counted_decays.py`, `config/v2_singlet/counted_decays_pdg2025.json` |

## Conservation and reproducibility

Every vertex conserves four-momentum exactly, and baryon number, charge, and strangeness as integers. Each event draws
its random numbers from five named substreams (topology selection, spectral state, exchange kinematics, decay channel,
decay kinematics) derived from the root seed and the event index. An event is therefore reproducible bit for bit,
and two variants of the model can be compared on the same random numbers.

## The parameters

Fourteen parameters are fitted:

- the non-diffractive mass law ε_nd (`epsilon_nd`) and the vertex slope b (`b_sd`);
- the intact proton's vertex term f (`intact_vertex_factor`);
- the single-diffraction tail and one-pion-exchange shares (`sd_tail_share`, `sd_ope_share`);
- the four flavor weights (`gamma_s_had`, `r_v`, `r_bbar`, `assoc_norm`);
- the leading-split shares and direction laws (`leading_split_fraction`, `meson_split_fraction`,
  `leading_split_kappa`, `meson_split_slope`);
- the isovector exchange fraction (`f_isovector`).

The diffractive intercept ε, the non-diffractive remnant mixture, and the central-source fraction are held fixed. The
names are those of the run description's `parameters` list and of the fit report's `fitted` block
([inputs.md](inputs.md)).
