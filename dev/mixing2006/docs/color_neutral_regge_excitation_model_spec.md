# Product Specification: Color-Neutral Regge-Informed Excitation and Hadronic Cascade Model

**Status:** Design specification  
**Target domain:** SPS soft-hadron production, initially p+p at 158 GeV/c beam momentum and then p+A  
**Primary goal:** Generate complete, globally conserved hadronic events from color-neutral soft exchange, with each excited source decaying through resolved resonances and a constrained high-mass continuum.

---

## 1. Purpose and scientific scope

This product replaces an inclusive fitted-primary-resonance basis with an event-level physical model. The earlier resonance basis remains useful as a regression benchmark and as a library of resolved low-mass decays, but it must no longer be interpreted as the mechanism of primary production.

The model implements the following physical hypothesis:

> A color-neutral soft exchange creates excited hadronic systems. Each system has fixed hadronic additive quantum numbers and decays independently through resonance cascades and, at sufficiently high mass, a continuum of physically allowed multiparticle channels.

For p+p, the baseline process is

$$
p+p\rightarrow X_{\rm proj}+X_{\rm targ},
$$

where both sources have

$$
(B,Q,S)_{X_{\rm proj}}=(1,1,0),\qquad
(B,Q,S)_{X_{\rm targ}}=(1,1,0).
$$

The exchange carries no net additive quantum numbers. There are no strings, diquarks, junctions, or partonic fragmentation endpoints in the model. A source is a complete color-neutral hadronic excitation, not an isolated parent resonance label.

For p+A, each wounded target nucleon produces one target-like excitation system, while the projectile is one continuing baryonic excitation that undergoes sequential soft interactions and fragments after its final encounter:

$$
p+A\rightarrow X_{\rm proj}^{(\nu)}+\sum_{j=1}^{\nu}X_{{\rm targ},j}+\text{spectators}.
$$

This is the required multiple-fragmentation structure: multiple target sources and one cumulatively excited projectile source.

### 1.1 In scope

- Complete p+p and p+A events with exact global four-momentum conservation.
- Exact conservation of baryon number, electric charge, and strangeness.
- Regge-informed sampling of source masses, momentum transfer, and exchange class.
- Resolved low-mass resonance cascades using a PDG-informed decay library.
- A high-mass hadronic continuum that can produce associated strangeness and baryon-antibaryon pairs above physical threshold.
- A Glauber-based p+A collision geometry and wounded-nucleon construction.
- Direct comparison with NA49 inclusive spectra, particle ratios, target/projectile-separated observables, and correlation-level observables where available.
- Reproducible scans, likelihood fits, and model-falsification tests.

### 1.2 Explicitly out of scope for version 1

- Partonic showers, color strings, diquark fragmentation, and baryon junction transport.
- A claim that a complete triple-Regge calculation is being performed.
- Free extraction of Regge intercepts and residues from one fixed-energy inclusive dataset.
- Amplitude-level interference among the full overlapping high-mass resonance spectrum.
- Intranuclear rescattering of formed daughter hadrons in the baseline p+A release.

---

## 2. Design principles and non-negotiable constraints

### 2.1 Exact event closure

Every generated event shall satisfy

$$
\sum_{\rm incoming}P^\mu=\sum_{\rm final}P^\mu,
$$

within configurable floating-point tolerance, and

$$
\sum_{\rm final}B=\sum_{\rm incoming}B,\qquad
\sum_{\rm final}Q=\sum_{\rm incoming}Q,\qquad
\sum_{\rm final}S=\sum_{\rm incoming}S.
$$

Conservation must be tested before event acceptance and recorded in diagnostics. Events must never be repaired by silently dropping particles, rescaling final momenta, or changing particle identity after decay.

### 2.2 Source-level conservation

Every source carries an immutable quantum-number vector

$$
\mathcal Q_X=(B_X,Q_X,S_X).
$$

All descendants of that source must sum to exactly $$\mathcal Q_X$$. This enforces local compensation of strangeness and local accounting of baryon-antibaryon pair production.

### 2.3 Mass-before-decay architecture

Primary production produces source four-vectors and source quantum numbers, not independently fitted final-particle templates. Decay begins only after all source four-vectors have been accepted under global event closure.

### 2.4 Threshold safety

A channel is eligible only if the source mass exceeds the sum of its daughter masses. In particular, an excitation with $$B=1$$ may produce a proton-antiproton pair only through an allowed baryon-number-preserving final state such as

$$
X_{110}\rightarrow p+p+\bar p+n\pi,
$$

which requires at minimum

$$
M_X\geq 3m_p\simeq2.814\ \mathrm{GeV}.
$$

Antibaryons must not be generated as unconstrained primary anti-baryonic source templates in a model where each p+p source begins with $$B=1$$.

### 2.5 Controlled phenomenology

The model is called **Regge-informed**. Regge phenomenology constrains source mass, transfer, exchange class, and finite-energy trends. The resolved resonance library and continuum decay model provide the hadronic realization. Parameters must have a documented physical role and a bounded scan range.

---

## 3. Physics model

## 3.1 Exchange classes

The baseline exchange is a vacuum, color-neutral Pomeron-like soft exchange, denoted $$P$$. It transfers momentum but no additive quantum numbers.

Optional extensions are implemented as distinct, auditable exchange classes:

| Class | Additive quantum-number transfer | Use |
|---|---:|---|
| Vacuum/Pomeron-like $$P$$ | $$\Delta B=\Delta Q=\Delta S=0$$ | Baseline soft excitation |
| Isoscalar secondary Reggeon | $$\Delta B=\Delta Q=\Delta S=0$$ | Finite-energy non-vacuum correction |
| Isovector/charge-exchange Reggeon | Opposite charge transfer between sources | Only if vacuum exchange fails charge-sensitive tests |
| Strange exchange | Opposite strangeness transfer between sources | Deferred; requires dedicated evidence |

Version 1 must implement only the vacuum-exchange class in production. The data model and event record must nevertheless support exchange-class labels so that extensions do not require a schema change.

## 3.2 p+p source construction

In the p+p center-of-mass frame, sample two source masses $$M_1,M_2$$ and a momentum-transfer variable or equivalent transverse recoil variable. The two sources are constructed as a two-body final state:

$$
p+p\rightarrow X_1(M_1)+X_2(M_2),
$$

with the kinematic requirement

$$
M_1+M_2\leq\sqrt{s}.
$$

The center-of-mass momentum is

$$
p^*=\frac{\lambda^{1/2}(s,M_1^2,M_2^2)}{2\sqrt{s}},
$$

where

$$
\lambda(a,b,c)=a^2+b^2+c^2-2ab-2ac-2bc.
$$

The sampled recoil must lie in the physically allowed two-body range. The generator must reject impossible combinations before source decays are sampled.

### 3.2.1 Joint source weight

Use the following as the version-1 production ansatz:

$$
\frac{d^3\sigma_e}{dt\,dM_1^2\,dM_2^2}
\propto
F_e(t)\,\rho_{110}(M_1)\,\rho_{110}(M_2)\,W_e(s,M_1^2,M_2^2,t)\,
\Theta(\sqrt{s}-M_1-M_2).
$$

Definitions:

- $$e$$ is the exchange class.
- $$F_e(t)$$ is an exchange-proton coupling and transfer form factor.
- $$\rho_{110}(M)$$ is the spectral density for a source with $$B=1,Q=1,S=0$$.
- $$W_e$$ is a finite-energy Regge-informed source-excitation weight.
- The theta function represents the necessary mass boundary; full two-body closure is enforced numerically.

A practical initial parameterization is

$$
F_P(t)=\exp(B_Pt),
$$

and

$$
W_P=\left(\frac{s}{s_0}\right)^{2\alpha_P(t)-2}
\left(\frac{M_1^2}{s_0}\right)^{\alpha_{\rm exc}(0)-1}
\left(\frac{M_2^2}{s_0}\right)^{\alpha_{\rm exc}(0)-1}
D_{\rm edge}(s,M_1,M_2),
$$

with

$$
\alpha_P(t)=\alpha_P(0)+\alpha'_Pt.
$$

**Effective transfer slope.** At fixed beam energy the combined transfer dependence of $$F_P(t)\,W_P$$ is exponential in $$t$$:

$$
F_P(t)\,W_P \propto
\exp\left[\left(B_P+2\alpha'_P\ln(s/s_0)\right)t\right],
$$

because $$\alpha_P(0)$$ contributes only the overall factor $$(s/s_0)^{2\alpha_P(0)-2}$$, which cancels from the normalized fixed-energy $$t$$ distribution. The $$t$$ distribution sampled at the configured beam momentum therefore carries the effective transfer slope

$$
B_{\rm eff}=B_P+2\alpha'_P\ln(s/s_0),
$$

with $$s$$ the fixed-target invariant mass squared in GeV$$^2$$ and $$s_0=1$$ GeV$$^2$$. At the bundled beam momentum of 158 GeV/c the fixed-target kinematics give $$\sqrt{s}=17.2702$$ GeV, $$s=298.26$$ GeV$$^2$$, and $$\ln(s/s_0)=5.698$$, so the bundled values $$B_P=4.5$$ and $$\alpha'_P=0.25$$ define $$B_{\rm eff}=4.5+2\times0.25\times5.698=7.349$$ GeV$$^{-2}$$.

The configuration carries the effective slope as the field $$\texttt{regge.effective\_transfer\_slope}$$. When the field is set, the generator samples every regge-mode $$t$$ distribution with that value, overriding the $$B_P+2\alpha'_P\ln(s/s_0)$$ combination, which applies to configurations that leave the field unset; in the sequential p+A stage the beam-energy slope then applies to each projectile-target encounter. The components $$B_P$$ and $$\alpha'_P$$ remain recorded frozen inputs that define the bundled default through the combination above.

The boundary suppression $$D_{\rm edge}$$ must be explicit, normalized, and documented. It handles the finite SPS phase space where asymptotic triple-Regge expressions are not sufficient.

**Sampled measure.** The ansatz above is stated with respect to the measure $$dt\,dM_1^2\,dM_2^2$$. The implemented sampler draws the two source masses on a grid uniform in $$M_1$$ and $$M_2$$, and the two mass measures are related by

$$
dM_1^2\,dM_2^2 = 4\,M_1M_2\,dM_1\,dM_2.
$$

The Regge-mode cell weight therefore multiplies the factor
$$\left(M_1^2/s_0\right)^{\alpha_{\rm exc}(0)-1}
\left(M_2^2/s_0\right)^{\alpha_{\rm exc}(0)-1} D_{\rm edge}$$
by the Jacobian $$4\,M_1M_2$$. The sampled joint source-mass density in mass space is

$$
\frac{d^2P}{dM_1\,dM_2}
\propto
4\,M_1M_2
\left(\frac{M_1^2}{s_0}\right)^{\alpha_{\rm exc}(0)-1}
\left(\frac{M_2^2}{s_0}\right)^{\alpha_{\rm exc}(0)-1}
D_{\rm edge}(s,M_1,M_2)\,
\Theta(\sqrt{s}-M_1-M_2),
$$

which is proportional to $$(M_1M_2)^{2\alpha_{\rm exc}(0)-1}$$ times the edge suppression. For the default value $$\alpha_{\rm exc}(0)=0.5$$ the mass-space density is flat before edge suppression, and the squared-mass density carries the $$1/(M_1M_2)$$ factor that follows from $$\alpha_{\rm exc}(0)-1=-1/2$$.

The exponent $$\alpha_{\rm exc}(0)$$ is defined with respect to the squared-mass measure: it is the power of $$M_i^2/s_0$$ in the joint density $$d^2P/dM_1^2\,dM_2^2$$. In the sampled mass space it acts on the product $$M_1M_2$$ with power $$2\alpha_{\rm exc}(0)-1$$, so a change of $$\alpha_{\rm exc}(0)$$ moves the sampled source-mass spectrum.

**Empirical mass law and comparison.** The model also supports an empirical source-mass parameterization (`simple` mode) as an alternative to the Regge-informed production law (`regge` mode). In the simple mode the density is stated directly with respect to the measure $$dM_1\,dM_2$$ and the sampler implements it by construction: the cell weight is the exponential mass fall-off $$\exp[-(M_1-m_{1,\min}+M_2-m_{2,\min})/M_{\rm scale}]$$ times the edge suppression $$D_{\rm edge}^{\gamma_{\rm edge}}$$, on the grid uniform in $$M_1$$ and $$M_2$$, with no Jacobian factor. Both laws sample on the same $$96\times 96$$ grid, the same finite-grid approximation of the triangular boundary $$\Theta(\sqrt{s}-M_1-M_2)$$ (each rectangular cell is weighted at its center and sampled uniformly, with draws outside the triangle rejected), the same source threshold minima, and the same edge suppression, so the comparison isolates the mass-law form (power law in the squared-mass measure versus exponential in the mass measure) and their shared transfer-slope and edge-suppression roles.

The two mass laws were evaluated in a pre-registered predictive validation comparison (milestone M10 of TASK-20260905-physics-review, run record `dev/00runs/mass_law_compare_2026-09-07/`):

1. **Protocol:** A longitudinal train/test split partitioned the 1017 M4 windowed bins ($$0 \le x_F \le 0.4$$, $$0.05 \le p_T \le 1.5$$~GeV/$$c$$) at the fragmentation-region boundary $$x_F = 0.2$$. Each law was fitted on the 676 central training bins ($$x_F < 0.2$$) and evaluated on the 341 forward held-out bins ($$x_F \ge 0.2$$). The fit used the M8 uncertainty-aware binned objective ($$K = 8$$ replicas of 2000 events, 16,000 events per evaluation, bounded Powell $$x_{\rm tol}=f_{\rm tol}=10^{-3}$$, cap 10) with the seven continuum parameters frozen at their M8 calibration values and three free parameters per law: `alpha_exc0`, `effective_transfer_slope`, and `edge_power` for the Regge law; `mass_scale`, `edge_power`, and `transfer_slope` for the simple law. Six arms were evaluated across $$\{\text{regge}, \text{simple}\} \times \{H_1, H_2, H_3\}$$ with $$H_3$$ (the recursive empirical kernel) as the primary record. The $$H_1$$ arms ran at a pre-registered declared reduction ($$K = 8 \times 50 = 400$$ events per evaluation, maxiter 2) due to kernel cost; their two-iteration descents enter only the cross-hypothesis spread and carry a $$6.32\times$$ Monte-Carlo noise scaling.
2. **Metrics:** On the 341 held-out bins, the pooled held-out WRRMSE at $$H_3$$ is 1.0477 for the Regge law and 0.8380 for the simple law. Across the three decay hypotheses, the metric ranges are $$[1.0477, 1.9943]$$ (spread $$\text{band}_{\rm regge} = 0.9465$$) for the Regge law and $$[0.8302, 1.8485]$$ (spread $$\text{band}_{\rm simple} = 1.0184$$) for the simple law.
3. **Verdict:** Under the pre-registered rule, the difference $$\delta = M_{\rm simple} - M_{\rm regge} = -0.2097$$ is compared against the combined cross-hypothesis band $$\sqrt{\text{band}_{\rm regge}^2 + \text{band}_{\rm simple}^2} = 1.3903$$. Because $$|\delta| \le 1.3903$$, the data do not distinguish the two mass laws within the bands (`not_distinguished`). In the identical-statistics $$H_2/H_3$$ diagnostic subset, individual bands of 0.2128 (Regge) and 0.0079 (simple) yield a combined diagnostic band of 0.2129. The primary $$H_3$$ difference ($$\delta = -0.2097$$) remains within this combined diagnostic band with a margin of 0.0033, while the $$H_2$$ difference ($$\delta_{H_2} = -0.4303$$) exceeds it. The numerical preference for the simple law is concentrated at $$H_2$$. In all simple-mode arms, the fitted `mass_scale` ran to 4.29--4.62~GeV against its 5.0~GeV upper bound, flattening the exponential toward heavy-mass power-law behavior.
4. **Framing status:** At the current simulation precision and decay-model uncertainty, the analysis cannot establish a preference between the Regge-inspired and empirical source-mass forms. The recorded `not_distinguished` outcome is that conservative statement. It does not positively validate the Regge form. The primary $$H_3$$ numbers favor the simple law, and the identical-statistics $$H_2/H_3$$ diagnostic is hypothesis-dependent; the combined cross-hypothesis band is not a formal model-comparison statistic because the $$H_1$$ arms used lower event statistics and a smaller optimization cap. The Regge-informed framing of the spec is retained as the working production law, not as a demonstrated preference.

## 3.3 Spectral density of an excited baryonic source

The source spectral density is

$$
\rho_{110}(M)=\rho_{\rm resolved}(M)+\rho_{\rm continuum}(M).
$$

The density is a production prior for a complete baryonic source. It is not a list of freely weighted individual resonance templates.

### 3.3.1 Resolved region

For $$M<M_{\rm match}$$, sample explicit low-mass states and coupled decay networks from a maintained resonance database. The initial database includes proton-compatible baryon resonances and the mesonic states needed for their decay chains:

- nucleon, $$\Delta$$, and selected $$N^*$$ states;
- $$\rho$$, $$\omega$$, $$\eta$$, $$f_2$$, and relevant scalar/tensor mesons;
- $$K^*$$ and strange baryon resonances as required by allowed decay chains;
- charge-conjugate states only when the source quantum numbers permit them as part of an internally conserved final state.

Resolved-state selection may use a mass-dependent spectral function,

$$
A_R(M)\propto\frac{M\,\Gamma_R(M)}{(M^2-M_R^2)^2+M^2\Gamma_R^2(M)},
$$

with truncation at physical channel thresholds. Version 1 may retain constant-width line shapes where adequate, but the design must support energy-dependent widths and barrier factors.

### 3.3.2 Continuum region

For $$M\geq M_{\rm match}$$, create a high-mass color-neutral baryonic cluster with fixed $$\mathcal Q_X=(1,1,0)$$. Its decay is generated through a constrained channel sampler and sequential hadronic cascade.

The continuum density may use a Hagedorn-inspired shape with finite-energy suppression, for example

$$
\rho_{\rm continuum}(M)\propto M^{-a}\exp(M/T_H)\,C_{\rm finite}(M;\sqrt{s}),
$$

or an empirically parameterized smooth alternative. The adopted form must be selected by likelihood comparison and not treated as a literal determination of a Hagedorn temperature.

The transition at $$M_{\rm match}$$ must be continuous in total probability. A default scan range is

$$
2.2\ \mathrm{GeV}\leq M_{\rm match}\leq2.8\ \mathrm{GeV}.
$$

## 3.4 Decay dynamics

### 3.4.1 Common requirements

Every decay operation shall:

1. preserve parent four-momentum exactly;
2. preserve $$B,Q,S$$ exactly;
3. require physical phase space;
4. record parent, daughters, channel identifier, sampled mass, and random-number stream key;
5. terminate after a finite configurable depth, with any failure treated as a rejected event rather than an untracked fallback.

### 3.4.2 Resolved decays

Two-body decays use exact rest-frame kinematics followed by Lorentz boost. Version 1 uses isotropic angular generation unless a channel-specific angular distribution is implemented.

Three-body decays use accepted Lorentz-invariant phase-space sampling. The implementation must support channel weights to allow later insertion of Dalitz or sequential-resonance structure.

Priority upgrades after baseline validation:

- energy-dependent widths;
- orbital-angular-momentum barrier factors;
- explicit sequential channels such as $$N^*\rightarrow\Delta\pi\rightarrow N\pi\pi$$;
- channel-specific angular distributions for the dominant resolved resonances.

### 3.4.3 Continuum decays

The continuum channel sampler must choose multiplicity and hadron content subject to the source quantum numbers and mass. It should favor an interpretable sequential structure rather than unconstrained statistical sampling:

$$
X_{110}\rightarrow R_B+R_M,\qquad R_B\rightarrow\cdots,\qquad R_M\rightarrow\cdots
$$

with direct multiparticle channels allowed where the resolved spectrum is insufficient.

Mandatory continuum channel families are:

| Family | Example | Minimum source mass | Physics role |
|---|---|---:|---|
| Non-strange baryonic cascade | $$X_{110}\rightarrow N+n\pi$$ | channel dependent | Baseline soft production |
| Associated strangeness | $$X_{110}\rightarrow K^+\Lambda+n\pi$$ | $$m_K+m_\Lambda+n m_\pi$$ | $$K^+/K^-$$ asymmetry with local $$S=0$$ |
| Associated strangeness | $$X_{110}\rightarrow K^+\Sigma^0+n\pi$$ | $$m_K+m_\Sigma+n m_\pi$$ | Strange-baryon feed-down |
| Strangeness-zero strange pair | $$X_{110}\rightarrow B+K^+K^-$$ in all locally strangeness-zero charge states ($$B+K^+K^-$$, $$B+K^0\bar K^0$$ with the baryon charge completing the source charge) | $$m_B+2m_K$$ for the charged pair, $$m_B+2m_{K^0}$$ for the neutral pair | $$K^-$$ and $$\bar K^0$$ production with local $$S=0$$ |
| Baryon-pair production | $$X_{110}\rightarrow p+p+\bar p+n\pi$$ | $$3m_p+n m_\pi$$ | Antiproton production |
| Mesonic pair source inside baryonic cascade | $$X_{110}\rightarrow N+M^*$$, $$M^*\rightarrow p\bar p+n\pi$$ | physical threshold | Alternative pair-production topology |

The implementation must not include a direct source channel that violates the source's baryon number, such as $$X_{110}\rightarrow\bar N^*+\cdots$$ without a compensating baryonic system.

**Charge-complete basis and status.** The implemented continuum basis samples, for each source type, every charge state consistent with the source quantum numbers. The non-strange family carries charge-balanced extra pions, sampled as $$\pi^0$$ singles and $$\pi^+\pi^-$$ pairs over all compositions with net $$(B,Q,S)=(0,0,0)$$. The strangeness-zero strange-pair families produce $$K^-$$ and $$\bar K^0$$ while preserving the source-local strangeness. The per-family strengths are recorded priors; the isospin-degenerate strange-pair partners share one strength ($$strange\_pair\_strength$$, see section 3.4.4), and the requested weights are exported for auditing against the realized family fractions. The decay kinematics use the retained recursive empirical kernel, which does not claim uniform Lorentz-invariant phase space. The basis therefore demonstrates exact conservation and legal thresholds. It is a conservation demonstrator, not yet a production model for identified spectra, because the per-family strengths are priors and the intermediate-cluster kinematics are not physically resolved. The three-hypothesis decay-model comparison (re-recorded in `dev/00runs/staged_decay_compare_2026-09-05-r2/` after correcting the H1 phase-space Kallen constant term and the antineutron Delta-bar pairing labels, nested commit c085269) names the decay kinematics under study. The implementation selector `decay_hypothesis` (bundled default `H3`) takes three values. `H3` is the retained recursive empirical kernel: intermediate cluster masses from a beta distribution and random daughter ordering, with no claim of uniform Lorentz-invariant phase space; it is the pre-comparison stream, byte-identical by construction. `H1` is numerically sampled Lorentz-invariant phase space with a fixed 1024-point grid approximation: the cluster mass at each recursion level is drawn from the density of states on that grid, and the daughters fill the physical phase space at fixed cluster mass; it carries no PDG inputs. Cell masses are trapezoidal and the draw is uniform inside the chosen cell. `H2` is explicit sequential resonances: the daughter list is paired deterministically by charge into $$\Delta(1232)$$ (nucleon plus pion) and $$\rho(770)$$ (charged pion plus pion, with the neutral pion never opening a pair), each resonance mass is drawn from a truncated p-wave Breit-Wigner line shape built from the PDG mass and width with the exact daughter threshold as the lower edge, and an empty line-shape window emits the daughters at threshold. The comparison runs all three hypotheses at the calibration parameter set of section 6.4 and at the pre-fit point, at 500 events per arm under one shared seed (the config seed plus 30000, distinct from the fit and audit substream offsets), and re-asserts the section 3.4.1 conservation requirements (four-momentum closure, integer $$B,Q,S$$, channel threshold) on every event of every arm; the H2 arms additionally pass a deterministic resonance-pair threshold audit with every resonance invariant mass at or above its daughter threshold. At the recorded statistics the per-hypothesis refits of the ten declared parameters (bounded Powell against the same windowed binned objective; the H2 and H3 refits at 200 events per evaluation converged at 153 and 611 evaluations, the H1 refit at 100 events per evaluation capped at two iterations) provide the spread that the comparison records as the model uncertainty of the decay kinematics. H1 (corrected kernel) moves all ten parameters within its two-iteration descent (alpha_exc0 0.515630, effective_transfer_slope 6.951843, edge_power 4.124132, match_mass 3.410396, pion_multiplicity_slope 1.094825, strange_lambda_strength 0.191588, strange_sigma_strength 0.118541, strange_pair_strength 0.142961, pair_strength 0.0927051, and pair_threshold_power 5.999358 against its upper bound 6.0). H2 moves only the two production-shape parameters (alpha_exc0 0.455086, effective_transfer_slope 7.029486) while the other eight parameters sit at their initial values. H3 moves all ten parameters (alpha_exc0 0.149897, effective_transfer_slope 6.949269, edge_power 4.208444, match_mass 3.425991, pion_multiplicity_slope 1.003057, strange_lambda_strength 0.179171, strange_sigma_strength 0.107764, strange_pair_strength 0.276804, pair_strength 0.0965722, pair_threshold_power 2.292341). The resulting parameter spread across hypotheses is 0.365733 for alpha_exc0 (0.149897 to 0.515630 versus the section 6.4 calibrated value 0.2328), 0.080217 for effective_transfer_slope (6.949269 to 7.029486 GeV$$^{-2}$$ versus the section 6.4 calibrated value 8.2630), 2.208444 for edge_power (2.0 to 4.208444), 1.025991 for match_mass (2.4 to 3.425991), 0.146943 for pion_multiplicity_slope (1.003057 to 1.15), 0.091588 for strange_lambda_strength (0.1 to 0.191588), 0.068541 for strange_sigma_strength (0.05 to 0.118541), 0.226804 for strange_pair_strength (0.05 to 0.276804), 0.081572 for pair_strength (0.015 to 0.0965722), and 3.999358 for pair_threshold_power (2.0 to 5.999358). The decay kernel changes the inferred production shape, so a fit can absorb artifacts of the decay algorithm into the source-mass and production-shape parameters. In the H1 and H3 refits the continuum directions move within the per-evaluation Monte-Carlo noise floor, while the H2 refit leaves them at their initial values, so the continuum directions remain unresolvable at these statistics, consistent with the section 6.4 calibration. The H1 fitted values represent a two-iteration descent from the initial point rather than a converged fit, and the spread carries that convergence deficit alongside the Monte-Carlo noise floor. The per-hypothesis observables (the section 6.4 binned diagnostics, the mean final multiplicity, the longitudinal (xF) spectra, and the mean transverse momentum profiles in both the xF-density and y-density conventions) are reported in the run record. Correlation claims should not be made until the decay kernel is physically resolved and the held-out predictive validation of section 9 is established.

### 3.4.4 Pair-production and strange-channel parameters

Baryon-pair and associated-strangeness production must use a small number of mass-dependent probabilities, not an independent fit coefficient for every final state. Examples:

$$
P_{p\bar p}(M)=0\quad\text{for}\quad M<3m_p,
$$

and above threshold,

$$
P_{p\bar p}(M)=c_{p\bar p}\left(1-\frac{(3m_p)^2}{M^2}\right)^{n_{p\bar p}}f_{\rm suppress}(M).
$$

An analogous thresholded form is used for $$K\Lambda$$ and $$K\Sigma$$. The strangeness-zero strange-pair families share one thresholded strength, $$strange\_pair\_strength$$, with the same $$strange\_threshold\_power$$ as the associated-strangeness families. The parameters are fitted jointly against identified spectra and particle-ratio observables, with priors or bounds recorded in configuration.

---

## 4. p+A extension

## 4.1 Collision geometry and wounded nucleons

A Glauber module samples impact parameter, projectile path, struck-nucleon count $$\nu$$, and struck nucleon identities. It must support nuclear density profiles, proton/neutron composition, and optional Fermi motion.

At a fixed impact parameter, a binomial form may be used for initialization,

$$
P_A(\nu\mid b)={A\choose\nu}[T_A(b)\sigma_{pN}]^\nu[1-T_A(b)\sigma_{pN}]^{A-\nu},
$$

but a Monte Carlo Glauber geometry is the preferred production implementation.

Each struck target nucleon produces one target-side source:

$$
N_j\rightarrow X_{{\rm targ},j}.
$$

Its quantum numbers are inherited from the struck nucleon:

$$
(B,Q,S)=
\begin{cases}
(1,1,0), & \text{proton},\\
(1,0,0), & \text{neutron}.
\end{cases}
$$

## 4.2 Sequential projectile excitation

The projectile must remain one baryonic system throughout the sequence. It does not fragment after each collision.

$$
X_{\rm proj}^{(0)}\xrightarrow{1}X_{\rm proj}^{(1)}\xrightarrow{2}\cdots\xrightarrow{\nu}X_{\rm proj}^{(\nu)}.
$$

At encounter $$j$$:

1. take the current projectile four-vector $$P_{\rm proj}^{(j-1)}$$;
2. take the selected target-nucleon four-vector $$P_{N_j}$$;
3. compute the available collision invariant energy $$s_j=(P_{\rm proj}^{(j-1)}+P_{N_j})^2$$;
4. sample an exchange and target-source mass conditional on $$s_j$$;
5. solve exact two-body or multi-source kinematics for the updated projectile state and target source;
6. reject and resample if any later-required source mass or conservation constraint is impossible;
7. store the transfer, collision index, and projectile state update.

The sampled projectile mass obeys the monotonic rule

$$
M_{\rm proj}^{(j)}\ge \max\left(M_{\rm proj}^{(j-1)},\,M_{{\rm excited},\min}\right),
\qquad j=1,\ldots,\nu,
$$

where $$M_{{\rm excited},\min}$$ is the lowest represented hadronic threshold of the projectile quantum numbers $$(B,Q,S)=(1,1,0)$$. The rule is the cumulative-excitation ansatz, a named model assumption of the baseline: each encounter adds excitation to the one continuing projectile system, and the lower bound keeps the outgoing projectile source above its lowest hadronic state. The ansatz is not a consequence of conservation or of Regge theory. A single collision could transfer energy from an already excited projectile into the newly created target system and leave a lower-mass outgoing projectile source. The baseline excludes that de-excitation channel, and the exclusion is the known limitation of the ansatz.

After the last encounter, all target sources decay independently and the final projectile source decays once.

The event-level condition is

$$
P_{\rm beam}+P_A=P_{\rm proj}^{(\nu)}+\sum_{j=1}^{\nu}P_{{\rm targ},j}+P_{\rm spectators},
$$

before decays. Nuclear binding and recoil treatment must be specified by the chosen nuclear initial-state model. The baseline initial state is independent on-shell nucleons at rest. It carries no nuclear binding, no Fermi motion, and no daughter rescattering, so the p+A output is a baseline sensitivity study in that state. Projectile stopping, backward production, and centrality trends can be sensitive to the omitted effects, and quantitative nuclear claims require systematic comparisons with Fermi motion, nuclear recoil and binding, formation length, and intranuclear reinteraction. A failed kinematic construction at any encounter is handled by resampling the whole Glauber event: the impact parameter and the wounded-nucleon multiplicity are redrawn and the sequence restarts. In the p+Pb208 baseline at 158 GeV/c, that resampling moves the centrality distribution at the sub-percent level. Over 50000 accepted events, the per-bin relative shift of the accepted impact-parameter distribution against the original Glauber draws has a root-mean-square of 0.16% and a maximum of 1.4% over the 0.05 fm impact-parameter bins with at least 50 draws. The accepted wound-count distribution lies inside the bin-by-bin 95% binomial band of the Glauber distribution in every $$\nu$$ bin with at least 30 accepted events.

## 4.3 Formation-time baseline

Version 1 uses a long formation-time approximation: sources are formed and decay after the sequence of primary projectile-nucleon encounters. No daughter hadron rescattering occurs inside the nucleus.

A later optional module may assign a formation length

$$
\ell_f\sim\gamma/\Gamma,
$$

and allow intranuclear decay/reinteraction. This is not part of baseline tuning, because it introduces a separate transport model and additional unconstrained parameters.

## 4.4 p+A physics predictions

The model must predict, rather than impose:

- approximately additive target-side production with wounded-nucleon number;
- projectile-side longitudinal slowing and excitation changes with $$\nu$$;
- a competition between increasing target-source multiplicity and decreasing projectile available energy;
- centrality evolution of pions, kaons, protons, and antiprotons;
- local compensation patterns for strangeness and pair production within each source.

---

## 5. Software architecture

## 5.1 Required modules

| Module | Responsibility |
|---|---|
| `kinematics` | Lorentz vectors, invariant masses, two-body construction, boosts, closure checks |
| `quantum_numbers` | Particle quantum-number table and exact additive conservation utilities |
| `exchange_model` | Exchange-class sampling, $$t$$ distributions, joint source-mass density |
| `spectral_density` | Resolved/continuum source-mass selection and matching |
| `resonance_db` | Resonance properties, decay channels, widths, branching data, validity metadata |
| `decayer` | Two-body, three-body, sequential, and continuum decays |
| `continuum_channels` | Thresholded channel-family construction and mass-dependent branching probabilities |
| `event_builder_pp` | Full p+p event generation |
| `glauber` | Nuclear geometry, wounded nucleons, target identities |
| `event_builder_pA` | Sequential projectile and multiple target-source construction |
| `analysis` | NA49 binning, invariant-cross-section conversion, ratios, seagull observables, correlations |
| `fit` | Likelihood, priors, scans, bootstrap, held-out validation |
| `validation` | Unit tests, conservation audits, channel-threshold tests, reproducibility tests |
| `io` | Versioned configuration, event record, histograms, run metadata |

## 5.2 Event record requirements

Each event record must include:

- run and configuration hashes;
- random seed and deterministic substream identifiers;
- incoming particles/nucleus and center-of-mass definition;
- exchange class and per-collision exchange metadata;
- all source records: four-vector, mass, $$B,Q,S$$, source role, parent collision index;
- all intermediate resonances and continuum clusters;
- complete parent-daughter graph;
- daughter four-vectors and final-state status;
- per-event conservation residuals;
- rejected-event reason if generation fails.

The event graph must permit tracing any final $$K^+$$ or $$\bar p$$ back to its source and channel family.

## 5.3 Configuration discipline

All tunable parameters are defined in a human-readable, versioned configuration file. A run must embed the complete configuration and software revision in output metadata.

Parameters are grouped as:

1. fixed external inputs: particle masses, PDG decay information, beam energy, nuclear density;
2. production-shape parameters: transfer slope, mass-density shape, finite-energy damping; the Regge mass exponent $$\alpha_{\rm exc}(0)$$ is defined in the squared-mass measure $$dM_1^2\,dM_2^2$$ per section 3.2.1; at one beam momentum the production-shape transfer degree of freedom is the effective transfer slope $$B_{\rm eff}=B_P+2\alpha'_P\ln(s/s_0)$$, carried by the configuration field $$\texttt{regge.effective\_transfer\_slope}$$; $$B_P$$ and $$\alpha'_P$$ are recorded frozen components whose separation requires the identifiability policy of section 6.3;
3. resolved/continuum matching parameters;
4. continuum channel-family parameters;
5. optional exchange-mixture parameters;
6. detector/analysis parameters, which must be separated from physics parameters.

No hidden hard-coded channel reweighting is permitted.

---

## 6. Calibration and inference

## 6.1 Calibration philosophy

Fit physical source and channel parameters jointly across species and observables. Do not fit separate primary source weights independently for $$\pi^+$$, $$\pi^-$$, $$K^+$$, $$K^-$$, $$p$$, and $$\bar p$$.

The principal p+p calibration dataset is the NA49 identified-particle invariant spectra in common $$x_F,p_T$$ bins at 158 GeV/c. The analysis layer must reproduce the published invariant-cross-section convention exactly.

Use a likelihood or covariance-aware objective where available. A baseline diagonal objective is

$$
\chi^2(\theta)=\sum_i\left(\frac{M_i(\theta)-D_i}{\sigma_i}\right)^2,
$$

supplemented by log-space residual diagnostics for spectra with large dynamic range.

The binned objective is implemented for the six NA49 p+p 158 GeV/c
identified spectra ($$\pi^-$$, $$\pi^+$$, $$K^+$$, $$K^-$$, $$p$$, and
$$\bar p$$), which are tabulated as the invariant cross section $$f$$ in
mb/GeV$$^2$$. The fit window is the paper window, $$x_F \in [0.0, 0.4]$$
and $$p_T \in [0.05, 1.5]$$ GeV/c. The Monte Carlo sample predicts the
cross section per inelastic event from one generated event per inelastic
event,

$$
M_i = \frac{1}{\pi\sqrt{s}\,N_{\rm ev}}
\sum_{\rm tracks}\frac{E}{p_T}\,\frac{1}{\Delta x_F\,\Delta p_T},
$$

with $$x_F = 2p_L/\sqrt{s}$$ taken from the track four-vectors and the
invariant $$E/p_T$$ track weighting. The data and their per-bin
uncertainties are converted with the same factor,
$$D_i = f_i/\sigma_{\rm inel}$$ and
$$\sigma_i = \sigma^{f}_i/\sigma_{\rm inel}$$, using
$$\sigma_{\rm inel} = 31.78$$ mb, the literature inelastic p+p cross
section quoted by NA49 alongside its measured trigger cross section of
28.23 mb (extracted inelastic cross section 31.46 mb). The per-table
normalization declaration is recorded in the configuration, so a table
already tabulated per inelastic event can be used without the
conversion. Bins without a tabulated uncertainty take the Poisson
data-content uncertainty $$\sqrt{D_i}$$, with a unit-count floor on
empty bins. That fallback is recorded per table and does not fire on the
six 158 GeV/c tables, which carry per-bin errors throughout the window.

## 6.2 Staged fit sequence

1. **Kinematics-only validation:** no fit; verify complete closure and acceptance-safe observable calculation.
2. **Low-mass resolved p+p fit:** constrain transfer and low-mass excitation behavior using pions and protons.
3. **Strangeness extension:** fit thresholded associated-strangeness channels jointly to $$K^+$$ and $$K^-$$.
4. **Pair-production extension:** fit high-mass continuum and $$p\bar p$$ channel strength jointly to $$p$$ and $$\bar p$$.
5. **All-species p+p fit:** retain a single shared parameter set.
6. **Out-of-sample checks:** particle ratios, seagull profiles, and correlation observables not used in primary tuning.
7. **p+A transfer:** tune only nuclear geometry or explicitly declared nuclear parameters; do not retune the p+p decay model per target or centrality bin.

## 6.3 Identifiability policy

The data will not uniquely identify every high-mass resonance or every continuum channel. The product must report identifiable aggregates, such as:

- resolved versus continuum source fraction;
- integrated source-mass distribution and high-mass tail;
- associated-strangeness continuum fraction;
- baryon-pair-capable continuum fraction;
- exchange-class mixture only if supported by data.

It must not report exclusive individual high-mass resonance cross sections unless correlation or invariant-mass data identify them.

Transfer-slope identifiability follows the same logic. At one beam momentum the normalized $$t$$ distribution depends only on the effective transfer slope $$B_{\rm eff}=B_P+2\alpha'_P\ln(s/s_0)$$, because $$\alpha_P(0)$$ cancels from the normalized fixed-energy event law and the slope enters the $$t$$ distribution only through this combination (section 3.2.1). The form-factor slope $$B_P$$ and the Regge slope $$\alpha'_P$$ are therefore not separately identifiable at one beam energy, and the stage-2 fit carries the effective transfer slope as its only fittable transfer parameter. Extracting $$\alpha'_P$$ requires an energy lever arm or independent transfer data at more than one beam momentum. The bundled fit bound $$[5.07,\,10.2]$$ GeV$$^{-2}$$ on the effective transfer slope is the image of the previously recorded $$\alpha'_P\in[0.05,\,0.5]$$ window under the combination at the bundled beam energy ($$B_{\rm eff}(0.05)=5.0698$$ and $$B_{\rm eff}(0.5)=10.198$$ with $$B_P=4.5$$ and $$\ln(s/s_0)=5.698$$), so the transfer coverage of the fit window is unchanged by the reparameterization.

Use profile likelihoods, bootstrap replicas, posterior correlation matrices, and sensitivity studies. Parameters with strong degeneracy must be grouped, constrained externally, or removed.

**Parameter profiles and degeneracy study of record.** The identifiability policy was evaluated in a pre-registered profile study of the six-parameter profile model (milestone M11 of TASK-20260905-physics-review, run record `dev/00runs/parameter_profiles_2026-09-07/`):

1. **Protocol:** Six parameters were freed in the profile model (`regge.alpha_exc0`, `regge.edge_power`, `continuum.strange_threshold_power`, `continuum.pair_threshold_power`, `continuum.match_mass`, `continuum.pion_multiplicity_slope`). The five out-of-set parameters were frozen at their M8 calibration values (`regge.effective_transfer_slope = 8.504141`, `continuum.strange_lambda_strength = 0.301846`, `continuum.strange_sigma_strength = 0.377728`, `continuum.strange_pair_strength = 0.191879`, `continuum.pair_strength = 0.084119`). Scans were evaluated on the M8 uncertainty-aware binned objective ($$K = 8$$ replicas of 2000 events, 1017 binned points, 1011 degrees of freedom) under bounded Powell ($$x_{\rm tol} = f_{\rm tol} = 10^{-3}$$, per-point cap 150 evaluations). The anchor minimum is $$\chi^2_0 = 47298.61$$ on 1011 degrees of freedom, starting from the M8 ten-parameter point where $$\chi^2 = 54809.05$$ under the M11 replica block (seeds 20330832..20330839, base 20330831).
2. **Decisions of record:** All six profiled parameters are retained as-is (`retained as-is`) under the pre-registered decision rule:
   - `regge.alpha_exc0`: retained as-is ($$\max |\rho| = 0.5896 < 0.9$$ against `continuum.pair_threshold_power`; $$2\sigma$$ profile interval covers 20.0% of the governing window).
   - `regge.edge_power`: retained as-is ($$\max |\rho| = 0.0000 < 0.9$$; $$2\sigma$$ profile interval covers 13.1% of the governing window).
   - `continuum.strange_threshold_power`: retained as-is ($$\max |\rho| = 0.4134 < 0.9$$ against `regge.alpha_exc0`; $$2\sigma$$ profile interval covers 20.0% of the governing window).
   - `continuum.pair_threshold_power`: retained as-is ($$\max |\rho| = 0.5896 < 0.9$$ against `regge.alpha_exc0`; $$2\sigma$$ profile interval covers 20.0% of the governing window).
   - `continuum.match_mass`: retained as-is ($$\max |\rho| = 0.0000 < 0.9$$; $$2\sigma$$ profile interval covers 20.0% of the governing window).
   - `continuum.pion_multiplicity_slope`: retained as-is ($$\max |\rho| = 0.5631 < 0.9$$ against `regge.alpha_exc0`; $$2\sigma$$ profile interval covers 15.2% of the governing window).
3. **Caveats of record:**
   - *Staircase landscape:* Off-anchor profile minima lie 3500 to 8500 $$\chi^2$$ units below $$\chi^2_0$$ (38770.9 to 42424.4 against 47298.6). Fresh nuisance refits find deeper shelves on the fixed-seed discrete objective. The $$\Delta\chi^2 = 1.0$$ and $$4.0$$ thresholds record where refit shelves rise above $$\chi^2_0 + \Delta\chi^2$$, yielding four full-window at-edge intervals and two narrow crossings (`edge_power` $$[1.515, 2.267]$$, `pion_multiplicity_slope` $$[1.276, 1.884]$$), reflecting shelf structure rather than smooth likelihood profiles.
   - *Flagged correlation matrix:* The profile-model Hessian at the anchor is not positive definite (one negative eigenvalue, $$-1.63\times 10^5$$), and diagonal rungs for five of six parameters required fallback step sizing. Pseudo-inverse inversion zeroes the non-positive diagonal rows (`regge.edge_power` and `continuum.match_mass`).
   - *Mixed pairs not locally measurable:* In the targeted 2-D sensitivity studies for (`strange_threshold_power`, `strange_lambda_strength`) and (`pair_threshold_power`, `pair_strength`), no step ladder rung entered the finite-difference acceptance window. Their sensitivity rests on the 1-D profile curves and flagged Hessian matrix.
   - *2-D cross sections:* Pair0 (`alpha_exc0`, `edge_power`) and Pair3 (`pion_multiplicity_slope`, `match_mass`) yield $$7\times 7$$ maps with curvature elongation ratios 2.17 and 1.68, well below the corroboration threshold 9.0.
   - *Cap honesty:* 84 of 156 production refit points reached the 150-evaluation cap; their values represent capped descents (best of trace).
   - *Record environment:* All 156 production points ran on the gateway grid (workers c0801--c0808) under one numerical environment (remote NumPy 2.0.2) to eliminate float-summation drift.
4. **Calibration boundary:** The M11 profile study is diagnostic. The profile model is a separate anchored six-parameter model, and its anchor does not replace the M8 ten-parameter point. Under the recorded staircase landscape, non-positive-definite Hessian, fallback finite-difference rungs, zeroed pseudo-inverse rows, and 84 of 156 nuisance refits hitting the evaluation cap, the $$\Delta\chi^2=1,4$$ profile interpretation does not apply. Identifiability remains unresolved under the current finite-Monte-Carlo objective. Profile-likelihood intervals and Hessian correlations are suspended until a stable minimum and a positive-curvature neighborhood are demonstrated. The mechanical `retained as-is` labels above record what the pre-registered decision rule returned; they are not a statistically valid determination of the six parameters.
5. **Provenance:** Canonical run record `dev/00runs/parameter_profiles_2026-09-07/` (`pre_registration.json` SHA-256 `d2e206c9bc5b7cf66ca64fbb6358dd739005585968faeb1deb149dc16788b57a`, `parameter_profiles_report.json`, `REPORT.md`).

## 6.4 Calibration status

The binned six-species objective of section 6.1 has its first executed fit attempt at the pre-registered settings (run record `dev/00runs/staged_binned_fit_2026-09-05/`): a diagonal per-bin chi-square over 1017 windowed binned points of the six NA49 p+p 158 GeV/c identified spectra, in cross-section-per-inelastic-event units ($$f/\sigma_{\rm inel}$$ with $$\sigma_{\rm inel}=31.78$$ mb), over the paper window $$x_F\in[0.0,0.4]$$ and $$p_T\in[0.05,1.5]$$ GeV/c, with one shared parameter set and one shared event sample of 2000 events per objective evaluation. At the fitted parameters the total chi-square is $$3.2328\times10^{6}$$; the per-species chi-squares per degree of freedom are 5644.3 ($$\pi^-$$), 6196.4 ($$\pi^+$$), 769.0 ($$K^+$$), 757.5 ($$K^-$$), 3077.2 ($$p$$), and 306.8 ($$\bar p$$), and the per-species WRRMSE values in the same convention are 0.70, 0.67, 0.91, 0.95, 0.93, and 1.01. The diagnostics use the tabulated per-bin uncertainties, which do not cover correlated systematics, and they carry the identifiability statement of the calibration: the two production-shape parameters, the excitation intercept (0.232753) and the effective transfer slope (8.262951 GeV$$^{-2}$$), moved from their initial values, while the remaining eight fit parameters (the regge-block edge power and the seven continuum-block parameters) sit at their initial values because the continuum directions carry only percent-level directional information per evaluation at 2000 events. That is the regime a Monte-Carlo-replica uncertainty treatment is designed to resolve, and the outcome is consistent with the identifiability policy of section 6.3, which requires parameters with strong degeneracy to be grouped, constrained externally, or removed.

**Error-convention and replica-block comparability.** The $$\chi^2$$ values of the binned objective are bound to their specific Monte Carlo replica seed block (for the M8 calibration, replica seeds 20300831..20300838; for the M11 anchor, replica seeds 20330832..20330839). Because fixed-seed replica binning generates a specific realization of Monte Carlo noise, $$\chi^2$$ values are directly comparable only within the same replica block and numerical environment. The M8 total $$\chi^2 = 44088.09$$ under the M8 block evaluates to $$54809.05$$ under the M11 block, providing a fresh reference point for the M11 profile scan without implying parameter drift.

The binned six-species objective and a first fit attempt are implemented and recorded. They do not establish calibration to experimental observables. The M8 uncertainty-aware result is $$\chi^2=44088.09$$ for 1017 bins and ten fitted parameters, or about $$43.8$$ per degree of freedom; the same parameter point evaluates to $$54809.05$$ on the M11 replica block. Both values reject an acceptable joint description of the six spectra under the stated error model. The earlier M4 data-only result, $$3.2328\times 10^6$$, is worse still. A minimizer having run is not a successful calibration. The correct status is: fit machinery implemented, a first fit attempt executed and rejected by goodness of fit, statistically stable parameter determination not established. The WRRMSE values near unity use a different normalization from the chi-square and do not reverse that rejection.

The first status, conservation and algorithm validation (exact four-momentum closure, exact integer $$B,Q,S$$, legal thresholded production, recorded ancestry), is established and is separate from this fit attempt. For the third status, the $$K^+\Lambda$$ association and $$p\bar p$$ traceability tests of section 9 are architecture-level internal consistency checks of designed source-local channels against a model-defined source-shuffled comparator. They are not experimental validation. The particle-ratio and seagull validation is executed at the frozen M8 parameter set: the charge-ratio part ($\pi^-/\pi^+$ in-window and high-$p_T$) and the $\bar p$ seagull profile pass within pre-registered bands, whereas the strange and baryon ratios ($K^+/K^-$ with $$z = +18.31$$, $p/\bar p$ with $$z = +5.77$$, and forward $p/\pi$ with $$z = -38.94$$ and $$-20.56$$) and the pion and kaon seagull profiles fail the bands at the frozen set. The p+A validation is executed at the minimum-bias level: 0 of 8 cell-set observables pass the pooled kernel-yield band, 6 of 67 $R(x_F)$ nodes pass, the pion $A$-scaling ordering direction disagrees with data at common nodes, and the projectile-stopping falsifier direction fires for forward $p+\mathrm{Pb}$ $R(x_F)$. The centrality test across impact-parameter classes (sets 1..5) is executed at the recorded level through the pre-registered $E_{\rm BE}$-to-$K$ mapping: 0 of 10 pooled kernel yields pass (1 tension, 9 beyond tension), all 88 $dn/dx_F$ nodes and all 80 $R_i(x_F)$ nodes land beyond tension, and the centrality-trend falsifier direction fires for $\pi^-$. The mapping assigns sets 1..5 to wound bands $[2,4]$, $[4,4]$, $[4,7]$, $[6,6]$, and $[7,7]$ with residuals under 2.5%, and all five sets carry the band-ambiguity flag under the $\pm 10\%$ VENUS systematic. Minimum-bias p+A trends stand as M12 recorded. The p+A acceptance audit of section 4.2 (sub-percent centrality shift, and the accepted wound-count distribution inside the bin-by-bin 95% binomial band of the Glauber distribution in every qualifying wound-count bin) is a conservation-level statement about the generation procedure: it shows that the generation loop carries no acceptance bias at the recorded statistics, and it is not a calibration or a predictive result. The publication claims of section 10 remain conditional on the full separation, and the calibration status recorded here does not authorize them on its own.

---

## 7. Validation, acceptance criteria, and falsifiers

## 7.1 Mandatory software acceptance tests

Every release must pass:

1. **Four-momentum test:** every accepted event closes to a configurable relative tolerance, with a default tighter than $$10^{-10}$$ in natural units.
2. **Quantum-number test:** all sources and events preserve $$B,Q,S$$ exactly as integer values.
3. **Threshold test:** no decay occurs below its physical threshold.
4. **Decay-graph test:** every non-final particle has a valid daughter list and every final particle has a valid ancestry.
5. **Reproducibility test:** identical configuration and seed reproduce identical event records and histograms.
6. **Symmetry test:** baseline p+p generation is projectile-target symmetric before analysis cuts.
7. **No-primary-antibaryon-source test:** p+p baseline sources carry $$B=1$$; any final antibaryon is traceable to a baryon-pair-producing legal channel.
8. **p+A accounting test:** total event baryon number equals $$A+1$$ and charge equals $$Z+1$$ for a proton projectile.

## 7.2 Physics acceptance criteria

A model configuration is viable only if it simultaneously:

- describes the principal p+p six-species spectra within predefined fit-quality targets;
- improves or correctly localizes the $$K^+/K^-$$ residual without violating source-level $$S=0$$;
- generates $$\bar p$$ through thresholded legal channels and improves the central $$\bar p$$ discrepancy without degrading proton spectra unacceptably;
- predicts p+p particle ratios and $$\langle p_T\rangle(x_F)$$ profiles not included in the primary objective;
- transfers to target/projectile-separated p+A observables with a single p+p physics parameter set;
- preserves physically interpretable parameter values under bootstrap replicas and reasonable mass-spectrum variations.

Numerical goodness-of-fit thresholds must be set after a baseline pseudodata closure test. They should not be chosen post hoc from the best experimental fit.

## 7.3 Model falsifiers

The following outcomes count as physical falsification or strong evidence of missing dynamics:

1. No vacuum-exchange source-mass distribution can jointly reproduce charge-sensitive spectra and particle ratios under local conservation.
2. The $$\bar p$$ yield requires a continuum pair-production probability inconsistent with the required energy budget or with proton yields.
3. The $$K^+/K^-$$ structure cannot be reproduced with allowed local associated-strangeness channels.
4. p+A target-side scaling or projectile stopping cannot be described by multiple target sources plus one sequential projectile source.
5. Required parameters vary substantially by species, xF window, or nuclear target, indicating that the assumed source dynamics is incomplete.

Only after these tests should non-vacuum exchange, intranuclear reinteraction, or additional collective mechanisms be introduced.

---

## 8. Deliverables and phased implementation

## Phase 0: Audit and regression freeze

**Deliverables**

- Frozen copy of the existing resonance-template generator and benchmark results.
- Automated audit proving which current decay channels generate each final species.
- A documented statement that primary anti-baryonic template sources are not compatible with the new $$B=1$$ source construction.
- Regression tests for retained resolved decay routines.

**Exit criterion:** baseline outputs are reproducible and the old model remains available strictly as a benchmark.

## Phase 1: Core event and source infrastructure

**Deliverables**

- Event record with source-level quantum numbers and full ancestry.
- Exact p+p two-source kinematics.
- Conservation and threshold validation suite.
- Configurable source-mass sampler with a simple symmetric mass density.

**Exit criterion:** generate stable p+p events with two $$\mathcal Q=(1,1,0)$$ sources and zero conservation failures.

## Phase 2: Resolved spectral-density model

**Deliverables**

- Resolved resonance selection conditional on source mass.
- PDG-informed decay implementation with source tracing.
- Continuous resolved/continuum matching interface.
- Fit to pions and protons in p+p.

**Exit criterion:** p+p spectra are generated without independent species-specific source weights.

## Phase 3: Continuum, strangeness, and baryon pairs

**Deliverables**

- Continuum cluster decay channel sampler.
- Thresholded $$K\Lambda$$ and $$K\Sigma$$ channels.
- Thresholded legal baryon-pair production channels.
- Source-mass and channel-family diagnostics for every final $$K$$ and $$\bar p$$.

**Exit criterion:** all $$\bar p$$ are generated from legal pair-producing ancestry, and strange particles obey source-local strangeness conservation.

## Phase 4: Regge-informed production scan

**Deliverables**

- Transfer distribution and joint source-mass ansatz.
- Finite-energy boundary treatment.
- Parameter scan and identifiability report.
- Comparison against the simpler empirical mass-density baseline.

**Exit criterion:** the Regge-informed form is retained only if it improves prediction or parameter stability relative to the empirical baseline.

## Phase 5: p+A wounded-target implementation

**Deliverables**

- Glauber module and target proton/neutron selection.
- Sequential projectile-excitation solver.
- Multiple target-source construction and spectator accounting.
- p+A target/projectile analysis suite.

**Exit criterion:** exact p+A accounting and qualitatively correct centrality trends without per-centrality retuning of the p+p decay model.

## Phase 6: Correlation-level tests and publication analysis

**Deliverables**

- Invariant-mass, conditional-spectrum, balance-function, and pair-correlation tools.
- Held-out validation protocol.
- Full uncertainty, bootstrap, and falsifier report.
- Rewritten physics paper focused on the constrained source-excitation hypothesis.

**Exit criterion:** a documented model comparison that separates what inclusive spectra establish from what correlation data establish.

---

## 9. Recommended analysis observables

### Primary calibration observables

- $$d^2\sigma/dx_Fdp_T$$ for $$\pi^\pm,K^\pm,p,\bar p$$ in common published bins.
- Integrated yields and ratios in matched phase-space windows.
- Charge ratios $$\pi^+/\pi^-$$ and $$K^+/K^-$$.
- $$p/\bar p$$ in central and forward windows.

### Held-out and discriminating observables

- $$\langle p_T\rangle(x_F)$$ seagull profiles.
- Particle ratios in successive $$x_F$$ intervals.
- Conditional strange-hadron yields given a forward $$K^+$$.
- $$K\Lambda$$, $$K\Sigma$$, $$p\bar p$$, $$p\pi$$, and $$K\pi$$ invariant-mass distributions where reconstructible.
- Multiplicity-conditioned spectra.
- Projectile-/target-separated p+A yields versus wounded-nucleon number (executed at the model-internal level; no data counterpart, stated).
- Energy evolution using external phase-space-matched datasets, after p+p validation.

**Source-local correlation predictions and execution.** The model predicts two source-local correlations:

1. **$$K^+\Lambda$$ source-local association:** a forward $$K^+$$ ($$x_F \ge 0.2$$) produced by associated strangeness is statistically associated with a compensating strange baryon from the same projectile-like source. The observable is the fraction $$F_1$$ of forward $$K^+$$ sharing a source with at least one same-event strange baryon ($$\Lambda$$, $$\Sigma^0$$, $$\Sigma^-$$, the complete continuum strange-baryon set), compared against a within-event source-shuffled baseline.
2. **$$p\bar p$$ source-local traceability:** a final $$\bar p$$ is traceable to a pair-capable source with mass above the pair threshold ($$2m_p = 1.876544$$~GeV). The observable evaluates four components: the fraction $$F_{\rm tr}$$ of $$\bar p$$ with parent source mass above threshold (pass requires zero violations with $$N_{\bar p} \ge 30$$); the parent-source mass shift $$dM_{\rm parent} = \langle M_{\rm parent}\rangle - \langle M_{\rm all}\rangle > 0$$; the event-conditioned mass shift $$dM_{\rm event} = \langle M_{\rm cond}\rangle - \langle M_{\rm all}\rangle > 0$$; and the same-source $$p\bar p$$ invariant-mass softening $$D_2 = \langle M_{\rm inv}\rangle_{\rm shuffled} - \langle M_{\rm inv}\rangle_{\rm real} > 0$$. Passing requires all four components to pass.

Both predictions were executed at the calibration-set decision of record: M9 evaluates at the M8 combined-error parameter set -- the current calibration under the uncertainty-aware convention (the M8 record supersedes the M4 record for that convention; M7 declared no hypothesis selection, so no hypothesis-specific set exists; the H3 decay kernel applies as the dataclass default). The production evaluation ($$N = 100{,}000$$ events, seed 20310831, run record `dev/00runs/staged_heldout_2026-09-07/`, report `heldout_correlations_report.json`) yielded:

- **$$K^+\Lambda$$ association:** $$F_1 = 0.7176$$ (5061/7053 forward $$K^+$$, Wilson 95\% [0.7069, 0.7280], Wald SE 0.0054) against a source-shuffled baseline of $$0.4449 \pm 0.0057$$ over 16 realizations; excess $$D_1 = +0.2727$$, significance $$Z_1 = 34.72$$. Pre-registered pass band $$D_1 \ge 0.20$$ with $$Z \ge 3.0$$: **pass**. The numerator accounts for all forward $$K^+$$ born in associated-strangeness families ($$2317\ K\Lambda + 2744\ K\Sigma = 5061$$); 5040 of 5061 associations run through a projectile-role source.
- **$$p\bar p$$ traceability:** $$F_{\rm tr} = 1.0000$$ (3249/3249 $$\bar p$$, zero violations below 1.876544~GeV; minimum parent mass 2.9393~GeV, maximum 13.8216~GeV; Wilson 95\% [0.9988, 1.0000]): **pass**. Parent-source mass shift $$+1.6550 \pm 0.0368$$~GeV ($$Z = 45.01$$, bootstrap 95\% [1.5796, 1.7232]~GeV): **pass**. Event-conditioned mass shift $$+0.4828 \pm 0.0204$$~GeV ($$Z = 23.68$$, bootstrap 95\% [0.4423, 0.5222]~GeV): **pass**. Same-source $$p\bar p$$ invariant mass $$2.3312$$~GeV (6498 real pairs) versus $$2.6415$$~GeV shuffled ($$D_2 = +0.3103 \pm 0.0141$$~GeV, $$Z_2 = 22.06$$): **pass**. Conjunction: **pass**.

Both pre-registered bands were met. These tests confirm that ancestry bookkeeping preserves the designed source-local channels against a source-shuffled comparator. They are architecture-level internal consistency checks. Experimental validation of those correlations requires reconstructed data with detector acceptance and backgrounds treated.

**Held-out p+p particle ratios, seagull profiles, minimum-bias p+A trends, and p+A centrality execution.** The held-out predictive validation was evaluated at the frozen M8 ten-parameter calibration set across two benchmark campaigns:

1. **p+p particle ratios and seagull profiles ($$N = 80{,}000$$ events, run record `dev/00runs/staged_heldout_ratios_pa_2026-09-10/`):** The $$\pi^-/\pi^+$$ charge ratio is established within the pre-registered bands in the in-window ($$z = +1.35$$) and high-$$p_T$$ ($$z = +0.83$$) windows (pass; forward window at tension, $$z = +2.94$$), and the $$\bar p$$ seagull profile passes at all 12 nodes with slope $$z = +1.67$$. Recorded failures at the frozen set include the forward $$p/\pi$$ deficit ($$z = -38.94$$ for $$p/\pi^-$$ and $$-20.56$$ for $$p/\pi^+$$), the in-window $$K^+/K^-$$ ratio ($$z = +18.31$$, model 6.00 versus data 1.54), the in-window $$p/\bar p$$ ratio ($$z = +5.77$$, model 21.36 versus data 8.37), and the pion and kaon seagull profiles (overly soft at $$x_F \le 0.05$$ by 25--45% and overly hard at forward $$x_F$$). Neither pre-registered falsifier direction fired.
2. **Minimum-bias p+A trends ($$N = 240{,}000$$ events, run record `dev/00runs/staged_heldout_ratios_pa_2026-09-10/`):** Evaluated across $$p+\mathrm{Pb}$$, $$p+\mathrm{C}$$, and the elementary reference. Zero of 8 pre-registered cell-set observables pass the pooled kernel-yield band (7 beyond tension; primary $$p+\mathrm{Pb}$$ pion density $$z = +28.72$$ for $$\pi^-$$ and $$+20.14$$ for $$\pi^+$$), and 6 of 67 registered $$R(x_F)$$ nodes pass (54 beyond tension). The model's pion $$A$$-scaling ordering direction disagrees with the data at common nodes ($$x_F \le 0.10$$). Falsifier 4 fired in the projectile-stopping direction (forward $$p+\mathrm{Pb}$$ $$R(x_F)$$ beyond $$k_{\rm tension}$$ across all eight nodes at $$x_F \ge 0.20$$).
3. **p+A centrality classes (sets 1..5, $$N = 480{,}000$$ events, run record `dev/00runs/staged_centrality_sets_2026-09-11/`, `pre_registration.json` SHA-256 `bdc655e002fdc4ba4b48cd98ea53e31e8499b22c3201726aaa43bb36170e79ab`, `centrality_report.json`, `REPORT.md`):** Evaluated across the five NA49 thesis centrality sets mapped to wounded-nucleon bands $[2,4]$, $[4,4]$, $[4,7]$, $[6,6]$, and $[7,7]$ with residuals under 2.5% (all five carry the band-ambiguity flag under the $\pm 10\%$ VENUS systematic). Zero of 10 pooled kernel yields pass (1 tension for set 1 $\pi^+$ with $$z = +3.46$$; 9 beyond tension with $$z = +23.8$$ to $$+96.8$$; forward pion yields overpredicted by 18--89%, 38--89% for $\pi^-$), all 88 $$dn/dx_F$$ nodes land beyond tension ($$x_F = 0$$ nodes strongly negative, forward nodes strongly positive), and all 80 $$R_i(x_F)$$ nodes land beyond tension ($$R_i(x_F)$$ rises from 0.13--0.25 at $$x_F=0$$ to 2.2--14.1 at forward $$x_F$$, whereas data falls from 0.97 to 0.04). F-M13-1 (centrality-trend direction) FIRED for $\pi^-$ (data trend $$-1.94\times 10^{-3}$$ versus model trend $$+2.31\times 10^{-3}$$, opposite signs with both endpoints beyond tension); F-M13-2 did not fire. Projectile-/target-separated yields versus wounded-nucleon number are recorded at the model-internal level (target pion fraction rises 0.56 to 0.69 from set 1 to set 5; inclusive data carry no decomposition).
4. **Status:** The charge-ratio and antiproton seagull sectors pass their pre-registered bands. The baryon, strangeness, seagull-shape, minimum-bias p+A, and p+A centrality sectors fail at the frozen M8 point. The sequential-projectile, independent-target p+A baseline was tested on those observables and failed: zero of 8 minimum-bias cell-set observables passed, the projectile-stopping falsifier fired, zero of 10 centrality-class pooled yields passed, and the centrality-trend falsifier fired for $$\pi^-$$. The next physics step is a revised nuclear mechanism, not another parameter scan of the same frozen construction.

---

## 10. Publication claims permitted by this product

If validation succeeds, the model may claim:

1. A color-neutral exchange followed by independently decaying hadronic excitation systems is a quantitatively viable mechanism for specified SPS observables.
2. Inclusive identified spectra constrain source-mass distributions and aggregate high-mass decay strength, not exclusive high-mass resonance populations.
3. Associated-strangeness and baryon-pair channels are required or constrained by the measured $$K^+/K^-$$ and $$p/\bar p$$ structures.
4. A multiple-target-source plus sequential-projectile-excitation construction provides a falsifiable p+A extension without diquark or string fragmentation.

The model must not claim:

- direct measurement of an individual Pomeron intercept from fixed-energy NA49 inclusive spectra;
- exclusive identification of high-mass $$N^*$$ states from inclusive yields;
- proof that the exchange is uniquely Pomeron-mediated rather than one viable color-neutral soft-excitation description;
- a successful p+A description before the p+A target/projectile and centrality tests have been performed.

These permitted and excluded claims are conditional on the three-status separation of section 6.4. Conservation and algorithm validation are established. The first binned fit attempt of the normalized production ansatz was executed and rejected by goodness of fit (section 6.4). The source-local $$K^+\Lambda$$ and $$p\bar p$$ tests of section 9 are architecture-level internal consistency checks, not experimental validation. Held-out validation against particle ratios, minimum-bias p+A trends, and p+A centrality classes is executed with passing charge-ratio and antiproton seagull sectors and recorded failures in baryon ratios, strangeness, seagull shapes, forward p+A yields, and centrality classes at the frozen M8 point. The sequential p+A baseline has been tested and has failed. Claims 1-4 remain unauthorized, and the excluded-claim text remains in force.

---

## 11. Definition of done

The full product is complete when it can generate p+p and p+A event samples under one versioned physics configuration; preserve all event- and source-level conservation laws exactly; create antiprotons only through physically thresholded baryon-pair production; create strange asymmetries through source-local legal channels; reproduce the chosen p+p calibration observables; and make held-out, testable p+A and correlation-level predictions without hidden species, centrality, or target-specific retuning.
