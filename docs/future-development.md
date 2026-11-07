# Future development

HEART is open source (GPL-2.0-or-later), and the directions below are open to anyone. Each entry states what the model does now,
what the extension needs, and which data would test it. Energy dependence is the largest extension, and the spin of
the continuum is the largest approximation inside the model.

## 1. Energy dependence

The model is fitted at one energy, p+p at 158 GeV/c (√s = 17.3 GeV). Its production layer is written in Regge
language, so it carries an explicit energy dependence. The diffractive masses run over ξ = M²/s with the Pomeron's
shrinkage, the pion flux takes σ_πp at the sub-energy ξs, and the cascade acts on whatever masses the exchange
produces. Running it at other energies tests the model as a whole. The criterion is the one the HEART paper states:
the predictions at another energy are made without retuning the decay rules, with only the declared production
inputs changed. A retuning of the catalog, the cascade, the flavor weights, or the split laws at a new energy is a
failed transfer, not a second model.

**What the code needs.**

- **One energy input.** The beam momentum is pinned in `dev/mixing2006/na49_mixing/topology_elastic.py`
  (`BEAM_MOMENTUM = 158.0`, `SQRT_S = 17.270`), which the event-class modules import. It is pinned again in
  `helpers/benchmarking/run_observables.py` (`SQRT_S`, `SQRT_S_CM`) and in the header of `heart_events.py`. The
  first step is to make √s a run-description input that reaches the class modules, the observables, and the event
  writer. With 158 GeV/c as the default, the published events stay bit for bit the same.
- **The external cross sections per energy.** `dev/mixing2006/config/v2_singlet/external_cross_sections.json` holds
  σ_inel = 31.78 mb, σ_SD(M²/s < 0.05) = 4.9 ± 0.8 mb, and σ_DD = 1.5 ± 1.0 mb at 158 GeV/c. The single-diffraction
  compilation that fixes the 158 GeV/c value (Goulianos and Montanha, Phys. Rev. D 59 (1999) 114017, Table 7)
  lists measurements from √s = 14 GeV upward. The p+p total and elastic cross sections of the PDG compilation fix the absorption of
  the pion exchange at each energy.
- **The kinematic edges.** Below log₁₀ s ≈ 3 energy–momentum conservation limits the longitudinal momentum, and
  Fischer et al. (Eur. Phys. J. C 82 (2022) 875) use x_F′ = p_L/p_L^max there. The mass-law and transfer-window
  functions of `topology_mass_laws.py` take √s as an argument, with the pinned value `PINNED_SQRT_S_EXACT` as the
  default. Their rejection rates near threshold need a check.

**What the data are.**

- NA61/SHINE measured the π±, K±, p, and p̄ spectra of p+p at 20, 31, 40, 80, and 158 GeV/c (√s = 6.3 to 17.3 GeV;
  Eur. Phys. J. C 77 (2017) 671) in one detector with one analysis. This is the natural first test.
- The SPS HADRONS global interpolations in this repository (`dev/mixing2006/spshadrons.web.cern.ch/data/`,
  `interpolation_*_pp.txt`) and the kaon yields versus √s from 2.8 to 630 GeV (`sdep_*_kaon_pp.txt`, measurements
  with their references) extend the range to the ISR.

**What it would answer.**

1. *The central region.* The model has no central source. At higher energy the fragmentation region scales and the
   central rapidity plateau rises, so the rise has to come from the growing mass range of the excited systems. This is
   a direct test of the two-component picture.
2. *The multiplicity per unit mass.* Above the open-charm threshold a continuum state's charged multiplicity grows
   linearly with its mass, where a string's grows with ln M². The energy dependence of the total multiplicity tests
   that law over a wider mass range than one energy provides.
3. *The kaon isospin breaking.* Measured K⁰_S/((K⁺ + K⁻)/2) in p+p is 0.91 at √s = 17 GeV (NA61/SHINE, and Brick et
   al. and Jaeger et al. near that energy). Within single experiments it is 0.72–0.78 at √s = 4–7 GeV (Firebaugh et
   al., Fesefeldt et al.), against 1 for isospin-blind kaons. HEART gives 0.92 at 17 GeV from associated production at
   the leading baryon's strangeness-exchange vertex, p* → K⁺Y, which becomes K⁰Y on a neutron line. An energy scan
   predicts the trend toward threshold with no new parameter. It bears directly on the isospin corrections of K/π
   ratios in nuclear collisions (Chvála for NA49, Eur. Phys. J. C 33 (2004) s615).

A first step that stays small: generate at 40 and 80 GeV/c with the 158 GeV/c parameters and the declared cross
sections at those energies, and compare with the NA61/SHINE spectra in the same binning.

## 2. The spin of the continuum

This is the model's largest physics approximation. Above the 2.0 GeV continuation floor a continuum state carries
baryon number, charge, and strangeness, but no spin or parity, so its two-body decays are isotropic in its rest frame.
Three assumed angular laws replace the angular structure that spin would produce: the leading-meson split, the
leading-baryon split, and the aligned continuing steps (`cascade_decay.py`; the options `leading_split_kappa`,
`meson_split_slope`, `baryon_line_kappa`). The physical states in this mass range are high spin, they overlap, and
they come in both parities at the same mass: N(2190) 7/2⁻, N(2220) 9/2⁺, N(2250) 9/2⁻, and N(2600) 11/2⁻, with widths
of 0.40 to 0.65 GeV. A single state of definite parity decays forward–backward symmetrically, so the asymmetric laws
the spectra ask for come from the overlapping sum of states.

**A route to a derived law.** By duality (finite-energy sum rules, Dolen, Horn, and Schmid 1968), the average over
overlapping s-channel resonances equals the t-channel Regge exchange of the same two-body reaction. A continuum vertex
X → a + b can therefore take its angular law from the exchange amplitude exp(bt) of the reaction X represents, which
in the rest frame of X is exp(κ cos θ) with κ = 2bq*². The model's direction laws already have that form. A derived
law takes b, and the daughter that continues the line, from the quark lines of the duality diagram and from measured
two-body angular distributions (πN → πN, πN → KΛ at √s = 2 to 4 GeV). The fitted directions would become
predictions. A second, complementary step assigns each continuum state a spin bounded by the leading Regge trajectory,
J ≤ α₀ + α′M² with α′ ≈ 1 GeV⁻², with both parities, and decays it through helicity amplitudes. Hyperon polarization
and spin transfer then become predictions.

**What can test it.** Individual states above 2 GeV cannot be resolved, so the constraints are statistical:

- a spin-parity decomposition of diffractively excited protons, p + p → p + (pπ⁺π⁻), as COMPASS performed for
  π⁻ + p → π⁻π⁻π⁺ + p in 100 mass bins up to 2.5 GeV (Phys. Rev. D 95 (2017) 032004);
- the spin observables of the fragmentation region, which a spinless cascade predicts to be zero: Λ polarization
  (Heller et al., Phys. Rev. Lett. 41 (1978) 607; R608 in p+p), the analyzing power of forward pions and the spin
  transfer to forward Λ with a polarized beam (FNAL E704: Phys. Lett. B 264 (1991) 462, Phys. Rev. Lett. 78 (1997)
  4003), and the spin-density matrix element ρ₀₀ of forward vector mesons;
- the two-body reactions at √s = 2 to 4 GeV, whose angular distributions fix the dual laws.

The open question of the forward K⁺ (item 4) belongs here: the K⁺ at x_F 0.40–0.55 lack longitudinal momentum at the
late vertices where a hyperon and a kaon are made, and the sign of the angular law at those vertices is what a derived
law would decide.

## 3. Proton–nucleus collisions

Fischer and co-workers developed the two-component picture on nuclear targets as well (Eur. Phys. J. C 49, 919 (2007);
Chvala et al., Eur. Phys. J. C 73, 2329 (2013)). A p+A event can be built from repeated
color-singlet exchanges of the projectile system along its path through the nucleus, with the p+p kernel frozen and
the nuclear geometry and the nucleon–nucleon cross section as external inputs. At most three quantities may be
fitted, once for all targets: the inelasticity of an additional collision, its width, and one saturation parameter.
NA49 published p+C at 158 GeV/c in the same tables and conventions as p+p (`dev/mixing2006/spshadrons.web.cern.ch/`,
the p+C pages), which makes carbon the first held-out prediction. `dev/mixing2006/na49_mixing/event_builder_pA.py`
holds a Glauber p+A event builder of an earlier stage of the project and is a starting point.

## 4. The open structure of the fit

The HEART paper names the residual structure with its size. Each item is a mechanism question:

- the forward K⁺ below the data, the forward K*⁰ excess, and the Λ x_F distribution, read together as the momentum
  sharing at the leading baryon's strangeness-exchange vertex;
- the forward K⁻ transverse shape, low below 0.6 GeV/c and high above 1 GeV/c;
- the Ξ⁻ of the baryonic lines, below the NA61/SHINE yield;
- the very forward soft π⁻ and a central soft-pion imbalance;
- the very forward protons at x_F = 0.90 to 0.95;
- the antiproton excess of n+p over p+p in the preliminary NA49 n+p data, about 18% after the normalization of the
  sample, against none in the model.

## 5. Engineering

- **Speed.** The generator is pure Python at about six events per second per core. A compiled cascade
  (`cascade_decay.py`, for example with Numba or Cython) would cut the cost of a fit in proportion.
- **Event output.** `heart_events.py` writes JSON Lines. A HepMC3 writer with the decay history would serve detector
  simulation.
- **The fit's single-machine steps.** The replica-block refits and the Hessian run on one machine and could be spread
  over a cluster.

## Working on the code

New physics enters as a kernel or run-description option whose default leaves the published events unchanged bit for
bit. A fixed-seed event digest before and after the change shows that. Every option has a test under
`dev/mixing2006/tests/` or `helpers/benchmarking/tests/`, and the test suite passes before a change is merged.
Comparisons between variants use the same random numbers (the same seed and event indices), which makes small
differences measurable.
