"""NA49 resonance analysis framework — Python port.

This package re-implements the NA49 C++/ROOT analysis framework in pure Python
using NumPy, SciPy, and matplotlib, with no ROOT dependency.

Modules
-------
constants
    Physical constants (masses, …).
pdg_database
    Built-in PDG particle data (replaces ``TDatabasePDG``).
reaction
    Lorentz boost parameters (replaces ``Treaction``).
histogram
    1-D histogram with inverse-CDF sampling + B-spline interpolation.
breit_wigner
    p-wave Breit-Wigner distribution (replaces ``TBW``).
particle
    Identified particle kinematics (replaces ``TIdParticle``).
decays
    Decay-table I/O (replaces ``PartDecays``).
event
    Event container + 2- and 3-body decay engine (replaces ``TIdEvent``).
event_conservation
    Read-only per-vertex conservation audit for staged and legacy events.
state_validation
    Closed five-value state-type labels and the per-state color-singlet check.
topology_elastic
    The elastic reference event class (Round 5, card 1): the intact-line,
    no-excitation p+p class with the single vacuum exchange and zero
    (B, Q, S, 2I3) transfer, the Regge-motivated exp(B_SD t) draw on the
    exact two-body t range, the exact elastic kinematics, and the
    zero-transfer ledger check; the normalization reference excluded from
    every inelastic sample by its class label.
exchange_ledger
    The explicit exchange quantum-number ledger (Round 5, card 2): the
    per-line ledger states, the closed allowed-exchange set (Round 5:
    the vacuum exchange with the zero transfer), the declared-transfer
    arithmetic, the local decay closure, and the five-rule event audit
    that is the color-singlet audit mechanism of Gate 5 and the run R54
    assertion; pure bookkeeping, zero random numbers.  The ledger
    LineState keeps its module-qualified name (the elastic record holds
    the package-level ``LineState`` export; additive exports only).
topology_remnants
    The leading remnant options of the outgoing lines (Round 5, card 3):
    the closed four-kind remnant set of plan section 11.2 (an intact
    proton; a neutron only through explicit charge exchange, refused at
    this round naming plan task 47; a low-mass baryonic resonance; a
    high-mass baryonic continuum excitation), the exact
    low-mass/high-mass split of the committed-catalog (1, 1, 0) records
    at the frozen kernel boundary B, the one-uniform kind draw over the
    nominal weights with the same-uniform low-mass record draw, and the
    caller-supplied high-mass mass-law attachment at the frozen
    interface; no minimum N-pi excitation is imposed.
topology_mass_laws
    The topology-specific mass laws (Round 5, card 4): the Regge-
    motivated single-diffraction law (M^2)^(-1-epsilon) *
    exp[B_SD (M^2) t] of plan section 11.5 with the exact phase-space
    limits (m_p^2 to (sqrt_s - m_remnant)^2, the exact two-body t range
    closed at t = 0), the 4096-point grid-normalized inverse-CDF M^2
    draw (the documented sub-tolerance convention), the closed-form t
    draw on the exchange_kinematics substream, the separate joint law
    for double excitation (both M^2 from the marginals with the exact
    M_1 + M_2 <= sqrt_s closure by rejection at the pinned cap of 1000
    attempts) and central production (the central mass from the same
    functional form with the total transfer t_c = t_1 + t_2), and the
    effective-combination-only parameter surface (epsilon, B_SD).
topology_single_diffraction
    The projectile and target single-diffraction classes (Round 5,
    card 5): one line carrying the card-3 remnant (nominal intact 0.5 /
    low-mass 0.5 split), the other line a single baryonic excitation
    with mass from card 4's marginal law and the exp(B_SD t) transfer on
    the exchange_kinematics substream, the exact two-body solve of the
    outgoing four-momenta, the decay through the frozen simplified
    cascade (the Round 4 budget per vertex), the vacuum exchange with
    the zero transfer through the card-2 ledger, the R50 gap-size
    record, and the beam-reverse mirror of plan line 826.
topology_non_diffractive
    The one-neutral-excitation non-diffractive class (Round 5, card 7):
    both baryonic lines carrying card-3 remnants at the nominal R50
    split, the central source at (B, Q, S, 2I3) = (0, 0, 0, 0) exactly
    with its mass from card 4's central-production law on the exact
    limits (m_pi0^2 to (sqrt_s - M_1 - M_2)^2) at the total transfer
    t_c = t_1 + t_2, the exact three-system kinematics solve, the three
    systems' decays through the frozen simplified cascade (the card-9
    handoff routing for the baryonic remnants, the continuum cascade
    over the activated neutral-mesonic family for the central source),
    the plan section 11.3 channel set through the committed
    enumeration, and the central-neutrality check of plan line 849.
topology_multi_neutral
    The multiple-neutral-excitation extension (Round 5, card 8): the
    unretained n in {1, 2} two-point count draw on the
    topology_selection substream with P(n = 2) bounded to [0, 0.5] and
    default 0.0, the n = 1 events bit-identical to card 7's ND records,
    the n = 2 second central source from card 4's central law on the
    residual limits (m_pi0^2 to (sqrt(s) - M_1 - M_2 - M_c1)^2) with
    the exact four-system closure by rejection on the pinned cap, both
    sources decaying through the frozen simplified cascade, the
    registry entry with the unretained flag, and the multi
    central-neutrality check over every source.
topology_runner
    The topology template runner (Round 5, card 9): the six-entry class
    registry (the elastic reference excluded from every inelastic
    sample, the unretained nd_multi excluded from the draw), the fixed
    R50 nominal parameter structure (the equal 0.25 inelastic class
    fractions, the M8-frozen B_SD, the nominal remnant split, the
    default P(n = 2) = 0.0), the closed topology_selection class draw
    (exactly one uniform; the elastic class over the elastic fraction,
    the inelastic classes over the fixed order), the per-class dispatch
    to the landed cards 1-8 records, the per-event observable
    extraction (leading-baryon x_F, excitation masses, momentum
    transfers, central gap, total and |y| < 0.5 central-activity
    multiplicities, central/forward pion and proton yields, the
    m_T-slope seagull moment), and the hermetic pinned trace.  No new
    physics: every dispatched record is bit-identical to the class
    module's own generation on the same substreams.
hadron_catalog
    Typed hadron/resonance catalog schema with deterministic validation.
resonance_widths
    Mass-dependent channel widths and the normalized spectral function.
resonance_decay
    Isospin charge-channel completion (Round 2, card 5), the recursive
    two-body decay kernel (card 6), and the branching/spectral
    self-check suite (card 8).
spectral_density
    Continuum color-singlet spectral families (Round 3, cards 4-6): the
    baryonic and mesonic continuum families with their normalized
    spectral densities, smoothstep threshold ramps, inverse-CDF mass
    draws, active-family selection, and allowed hadronic decay classes,
    plus the smooth catalog/continuum matching (card 6): the fixed
    w(M) blend ramp (BlendSpec), the catalog and continuum mixture
    densities, the blended density with its double-counting identity,
    and the per-source decay routing at the frozen source mass.
threshold_metric
    The re-registered threshold-discontinuity metric (Round 3, card 2):
    the full analytic difference with its open-set-change and
    momentum-drift decomposition, the paired 3.0-sigma evaluation, and
    the sealed-archive re-evaluation assembly.
cascade_decay
    Sequential color-singlet resonance decay (Round 4, card 4): the
    cascade state record of plan section 10.1, the quasi-two-body
    channel enumeration of plan section 10.2, and the transparent
    channel weight of plan section 10.3 with the fixed W_flavor /
    W_align hook interface and the frozen per-vertex draw budget.
cascade_axes
    Inherited production axes (Round 4, card 5): the initial source
    axis (the beam direction into the source rest frame for p+p, the
    lab z-axis for isolated sources), the rest-frame axis transport
    between cascade states, and the baryon-line inheritance rule that
    fills the cascade state's axis field; pure Lorentz algebra, zero
    random numbers.
cascade_alignment
    Baryonic and mesonic alignment kernels (Round 4, cards 6-7): the
    axis-dependent angular law dP/dcos(theta) proportional to
    exp[kappa_B cos(theta)] of plan section 10.4 for a baryonic
    daughter that continues the incoming baryonic line, its analytic
    normalization, the closed-form inverse-CDF draw with the bit-stable
    isotropic branch (the R43 ablation baseline), and the vertex-level
    cos(theta) draw and alignment-label assignment that card 4's
    draw_decay hook consumes inside the frozen 1 + 2 per-vertex budget;
    plus the card-7 mesonic even law dP/dcos(theta) proportional to
    exp[kappa_M cos(theta)^2] of plan section 10.4 line 724 for
    symmetric mesonic daughters, its analytic normalization, the
    pre-registered 10,001-node CDF-table sampler with linear
    interpolation, and the mesonic branch of the vertex dispatch and
    label bookkeeping.
cascade_form_factor
    The hadronic decay form factor (Round 4, card 8): the
    decay-transverse vertex factor |F_dec(q_perp)|^2 =
    exp(-b_dec q_perp^2) of plan section 10.5 with the q_perp
    convention (the daughter momentum in the parent rest frame,
    transverse to the unit inherited production axis), the
    pre-registered default b_dec = 1.0 GeV^-2 with the exact 1.0 off
    limit at b_dec = 0, and the attach_to_channel_weight composition
    on card 4's channel_weight (the separate named W_dec factor); a
    deterministic zero-draw weight, applied in the cascade decay
    weighting only, never in the exchange layer.
cascade_termination
    Cascade termination rules and the hard maximum depth guard
    (Round 4, card 9): the catalog handoff predicate of plan section
    10.6 (the committed catalog record within its support routes the
    state to the byte-unchanged Round 2 ResonanceDecayer), the
    direct-channel termination predicate (no open continuum channel
    and a valid direct hadronic channel), the cascade driver that
    applies card 4's draw_decay steps until termination, and the hard
    enforcement of card 4's MAX_CASCADE_DEPTH placeholder through
    CascadeDepthGuardError at depth 12.
generator
    MC particle generator (replaces ``TGenerPart`` + ``spliner.cxx``).
cuts
    Analysis cuts on particles, events, and pairs.
mixing
    Event-mixing ring buffer (replaces ``MMixBuffer``).
spectra
    Invariant-mass spectra with background subtraction (replaces ``TSpectra``).
fits
    Chi² fitting utilities (replaces ``TFits``).
rng_substreams
    Deterministic per-layer RNG substream derivation and manifest writer.
data_io
    HDF5 / NumPy persistence (replaces ROOT ``TFile``/``TTree``).
"""

from .constants import EMELEK, EMPI, EMKA, EMPROT, EMRHO, EMOMEGA, EMETA
from .reaction import Reaction
from .histogram import Histogram1D, spliner
from .breit_wigner import BreitWigner, bw_new, bw_new_array
from .particle import Particle
from .decays import (
    DecayMode,
    ParticleDecays,
    read_decays,
    write_decays,
    build_rho_decays,
    build_omega_decays,
    build_kaon_decays,
    build_soft_resonance_decays,
)
from .event import Event
from .generator import Generator, generator_from_file
from .cuts import ParticleCut, EventCut, PairCut
from .mixing import MixEvent, MixBuffer
from .spectra import Spectra
from .fits import Fitter, MultiFitter
from .rng_substreams import (
    SUBSTREAM_LABELS,
    RngSubstreams,
    derive_substreams,
    substream_manifest_path,
    write_substream_manifest,
)
from .parallel import GenerationConfig, split_event_counts, merge_track_chunks, generate_tracks_chunked
from .pdg_database import mass_width_overrides
from .quantum_numbers import QuantumNumbers, quantum_numbers, sum_quantum_numbers
from .excitation_kinematics import FourVector
from .exchange_model import SimpleExcitationParameters, ReggeExcitationParameters
from .continuum_channels import (
    ContinuumParameters,
    SourceDecayer,
    family_requested_weights,
)
from .source_event import SourceRecord, FinalParticle, ExcitationEvent
from .state_validation import (
    STATE_TYPES,
    NonSingletState,
    UnrecognizedStateType,
    is_color_singlet,
    state_type_of,
    validate_state_type,
)
from .hadron_catalog import (
    CatalogValidationReport,
    DecayChannel,
    HadronCatalog,
    HadronRecord,
    ProvenanceTag,
)
from .resonance_widths import (
    IDENTITY_BARRIER,
    BarrierFactor,
    channel_width,
    decay_momentum,
    draw_mass,
    kallen,
    spectral_function,
    total_width,
)
from .resonance_decay import (
    BranchingClosureReport,
    IsospinCompletionReport,
    OpenWeightSweepReport,
    ParentBranching,
    PerParentIsospinReport,
    RecordSpectral,
    SpectralNormalizationReport,
)
from .threshold_metric import (
    SIGMA_TOLERANCE,
    THRESHOLD_METRIC_ID,
    RecordSet,
    diagnose_residual,
    evaluate_pair,
    full_analytic_difference,
    rereval_report,
    species_key,
)
from .spectral_density import (
    BLEND_BOUNDARY,
    BLEND_RAMP_WIDTH,
    CONTINUUM_STATE_TYPE,
    NORMALIZATION_TOLERANCE,
    REFERENCE_GRID_POINTS,
    SUPPORT_M_MAX,
    BlendSpec,
    ContinuumFamily,
    active_families,
    blended_density,
    blended_normalization_report,
    catalog_density,
    continuum_density,
    family_density,
    family_draw,
    family_normalization_report,
    families_from_json,
    route_source_decay,
)
from .event_builder_pp import PPExcitationConfig, PPEventBuilder
from .event_builder_pA import PAExcitationConfig, PAEventBuilder
from .cascade_decay import (
    CATALOG_FAMILY_PREFIX,
    ISOTROPIC_ALIGNMENT_LABEL,
    KIND_CONTINUUM,
    KIND_ESTABLISHED_HADRON,
    KIND_ESTABLISHED_RESONANCE,
    MAX_CASCADE_DEPTH,
    CascadeChannel,
    CascadeState,
    channel_weight,
    draw_decay,
    enumerate_channels,
)
from .cascade_axes import (
    AXIS_NORM_TOL,
    LAB_Z_AXIS,
    boost_axis_to_rest_frame,
    inherit_axis,
    initial_source_axis,
)
from .cascade_alignment import (
    BARYONIC_ALIGNMENT_LABEL,
    BARYONIC_KAPPA_DEFAULT,
    ISOTROPIC_BRANCH_TOL,
    MESONIC_ALIGNMENT_LABEL,
    MESONIC_CDF_NODES,
    MESONIC_KAPPA_DEFAULT,
    assign_alignment_label,
    baryonic_kernel_density,
    baryonic_normalization,
    continues_baryon_line,
    draw_baryonic_alignment,
    draw_mesonic_alignment,
    draw_vertex_cos_theta,
    mesonic_cdf_table,
    mesonic_kernel_density,
    mesonic_normalization,
)
from .cascade_form_factor import (
    DECAY_FORM_FACTOR_B_DEFAULT,
    attach_to_channel_weight,
    decay_form_factor_weight,
    transverse_momentum,
)
from .cascade_termination import (
    TERMINATION_CATALOG_HANDOFF,
    TERMINATION_DIRECT_CHANNEL,
    CascadeDepthGuardError,
    has_valid_direct_channel,
    handoff_decay,
    is_catalog_handoff,
    run_cascade,
)
from .event_conservation import (
    VertexAuditReport,
    audit_staged_event,
    audit_legacy_event,
    vertex_audit_summary,
)
from .topology_elastic import (
    BEAM_MOMENTUM,
    ELASTIC_TRANSFER_SLOPE,
    PROTON_MASS,
    SQRT_S,
    ElasticEvent,
    ElasticTransferViolation,
    ExchangeRecord,
    LineState,
    check_zero_transfer,
    draw_elastic_t,
    elastic_t_range,
    generate_elastic,
)
# The ledger's LineState is deliberately not re-exported here: the
# elastic record of na49_mixing.topology_elastic holds the package-level
# LineState export and the __init__ change is additive only (Round 5,
# card 2); the ledger record is na49_mixing.exchange_ledger.LineState.
from .exchange_ledger import (
    ALLOWED_EXCHANGES,
    Exchange,
    LedgerAuditReport,
    NonSingletTransferError,
    VACUUM,
    ZERO_TRANSFER,
    apply_transfer,
    audit_event,
    close_decay,
    register_exchange,
)
# The remnant module keeps PROTON_MASS module-level: the package-level
# PROTON_MASS export stays card 1's elastic constant (the same catalog
# value; additive exports only, Round 5, card 3).
from .topology_remnants import (
    AVAILABLE_REMNANT_KINDS,
    CONTINUUM_M_MAX,
    KERNEL_BOUNDARY,
    REMNANT_KINDS,
    REMNANT_QN,
    ChargeExchangeNotImplemented,
    RemnantSelection,
    high_mass_records,
    low_mass_records,
    select_remnant,
)
# The mass-laws module keeps PROTON_MASS module-level: the package-level
# PROTON_MASS export stays card 1's elastic constant (the same catalog
# value; additive exports only, Round 5, card 4).
from .topology_mass_laws import (
    PI0_MASS,
    REJECTION_CAP,
    MassLawParams,
    PhaseSpaceRejectionExceeded,
    draw_central,
    draw_joint_dd,
    draw_marginal,
    draw_t,
    joint_closure,
    marginal_density,
    marginal_grid_mass,
    sd_limits,
    t_range,
)
# The single-diffraction module keeps PROTON_MASS / SQRT_S / BEAM_MOMENTUM
# module-level via card 1's elastic constants (the same catalog values;
# additive exports only, Round 5, card 5).
from .topology_single_diffraction import (
    NOMINAL_PARAMS,
    NOMINAL_SD_REMNANT_WEIGHTS,
    SD_CLASS_LABEL_PROJECTILE,
    SD_CLASS_LABEL_TARGET,
    SDSIDES,
    FinalHadron,
    SDEvent,
    gap_size,
    generate_sd,
    mirror_sd,
    sd_kinematics,
)
# The double-diffraction module keeps PROTON_MASS / SQRT_S / BEAM_MOMENTUM
# module-level via card 1's elastic constants (the same catalog values;
# additive exports only, Round 5, card 6); the final-state hadron record
# stays card 5's package-level FinalHadron export.
from .topology_double_diffraction import (
    DD_CLASS_LABEL,
    DD_LINE_LABELS,
    DDEvent,
    central_gap,
    dd_kinematics,
    generate_dd,
)
# The non-diffractive module keeps PROTON_MASS / SQRT_S / BEAM_MOMENTUM
# module-level via card 1's elastic constants (the same catalog values;
# additive exports only, Round 5, card 7); the final-state hadron record
# stays card 5's package-level FinalHadron export, the mass-law
# parameters stay card 5's package-level NOMINAL_PARAMS export (the same
# frozen values), and the central law's pion mass stays card 4's
# package-level PI0_MASS export.
from .topology_non_diffractive import (
    CENTRAL_LINE_LABEL,
    CENTRAL_QN,
    ND_CLASS_LABEL,
    ND_LINE_LABELS,
    NOMINAL_ND_REMNANT_WEIGHTS,
    CentralNeutralityViolation,
    NDEvent,
    check_central_neutrality,
    generate_nd,
    nd_kinematics,
    nd_limits,
)
# The multiple-neutral-excitation module keeps the mass-law parameters
# and the elastic constants at card 1/card 4's package-level exports (the
# same frozen values; additive exports only, Round 5, card 8), and the
# final-state hadron record stays card 5's package-level FinalHadron
# export.
from .topology_multi_neutral import (
    COUNT_BND,
    DEFAULT_P2,
    ND_MULTI_CLASS_LABEL,
    NDMultiEvent,
    P_N2_TOLERANCE,
    REGISTRY_ENTRY,
    check_multi_central_neutrality,
    generate_nd_multi,
    nd_multi_kinematics,
    nd_multi_limits,
    validate_p_n2,
)

# The topology template runner (Round 5, card 9): the runner's nominal
# parameter instance stays module-level in na49_mixing.topology_runner
# and is re-exported here under its own name NOMINAL_TOPOLOGY_PARAMS;
# the package-level NOMINAL_PARAMS stays card 5's single-diffraction
# mass-law object (the card-7 comment above).
from .topology_runner import (
    INELASTIC_ORDER,
    INELASTIC_SET,
    MASTER_SEED,
    TOPOLOGY_CLASSES,
    TopologyNominalParams,
    TopologyObservables,
    draw_topology,
    extract_observables,
    pinned_trace,
    run_topology,
)
from .topology_runner import NOMINAL_PARAMS as NOMINAL_TOPOLOGY_PARAMS

__version__ = "0.2.0"
__all__ = [
    "EMELEK", "EMPI", "EMKA", "EMPROT", "EMRHO", "EMOMEGA", "EMETA",
    "Reaction",
    "Histogram1D", "spliner",
    "BreitWigner", "bw_new", "bw_new_array",
    "Particle",
    "DecayMode", "ParticleDecays", "read_decays", "write_decays",
    "build_rho_decays", "build_omega_decays", "build_kaon_decays", "build_soft_resonance_decays",
    "Event",
    "Generator", "generator_from_file",
    "ParticleCut", "EventCut", "PairCut",
    "MixEvent", "MixBuffer",
    "Spectra",
    "Fitter", "MultiFitter",
    "SUBSTREAM_LABELS", "RngSubstreams", "derive_substreams",
    "substream_manifest_path", "write_substream_manifest",
    "GenerationConfig", "split_event_counts", "merge_track_chunks", "generate_tracks_chunked",
    "mass_width_overrides",
    "QuantumNumbers", "quantum_numbers", "sum_quantum_numbers",
    "FourVector",
    "SimpleExcitationParameters", "ReggeExcitationParameters",
    "ContinuumParameters", "SourceDecayer", "family_requested_weights",
    "SourceRecord", "FinalParticle", "ExcitationEvent",
    "STATE_TYPES", "state_type_of", "validate_state_type", "is_color_singlet",
    "UnrecognizedStateType", "NonSingletState",
    "PPExcitationConfig", "PPEventBuilder",
    "PAExcitationConfig", "PAEventBuilder",
    "VertexAuditReport", "audit_staged_event", "audit_legacy_event", "vertex_audit_summary",
    "HadronRecord", "DecayChannel", "ProvenanceTag", "HadronCatalog", "CatalogValidationReport",
    "IsospinCompletionReport", "PerParentIsospinReport",
    "BranchingClosureReport", "ParentBranching",
    "SpectralNormalizationReport", "RecordSpectral", "OpenWeightSweepReport",
    "IDENTITY_BARRIER", "BarrierFactor", "channel_width", "decay_momentum",
    "draw_mass", "kallen", "spectral_function", "total_width",
    "THRESHOLD_METRIC_ID", "SIGMA_TOLERANCE",
    "RecordSet", "full_analytic_difference", "evaluate_pair",
    "rereval_report", "diagnose_residual", "species_key",
    "ContinuumFamily", "active_families", "family_density", "family_draw",
    "family_normalization_report", "families_from_json",
    "BlendSpec", "catalog_density", "continuum_density",
    "blended_density", "route_source_decay", "blended_normalization_report",
    "BLEND_BOUNDARY", "BLEND_RAMP_WIDTH",
    "REFERENCE_GRID_POINTS", "SUPPORT_M_MAX", "CONTINUUM_STATE_TYPE",
    "NORMALIZATION_TOLERANCE",
    "MAX_CASCADE_DEPTH", "CATALOG_FAMILY_PREFIX", "ISOTROPIC_ALIGNMENT_LABEL",
    "KIND_ESTABLISHED_HADRON", "KIND_ESTABLISHED_RESONANCE", "KIND_CONTINUUM",
    "CascadeState", "CascadeChannel",
    "enumerate_channels", "channel_weight", "draw_decay",
    "AXIS_NORM_TOL", "LAB_Z_AXIS",
    "initial_source_axis", "boost_axis_to_rest_frame", "inherit_axis",
    "BARYONIC_KAPPA_DEFAULT", "ISOTROPIC_BRANCH_TOL",
    "BARYONIC_ALIGNMENT_LABEL",
    "baryonic_kernel_density", "baryonic_normalization",
    "draw_baryonic_alignment", "continues_baryon_line",
    "draw_vertex_cos_theta", "assign_alignment_label",
    "MESONIC_KAPPA_DEFAULT", "MESONIC_CDF_NODES",
    "MESONIC_ALIGNMENT_LABEL",
    "mesonic_kernel_density", "mesonic_normalization",
    "mesonic_cdf_table", "draw_mesonic_alignment",
    "DECAY_FORM_FACTOR_B_DEFAULT",
    "transverse_momentum", "decay_form_factor_weight",
    "attach_to_channel_weight",
    "TERMINATION_CATALOG_HANDOFF", "TERMINATION_DIRECT_CHANNEL",
    "CascadeDepthGuardError", "is_catalog_handoff",
    "has_valid_direct_channel", "handoff_decay", "run_cascade",
    "ELASTIC_TRANSFER_SLOPE", "PROTON_MASS", "BEAM_MOMENTUM", "SQRT_S",
    "LineState", "ExchangeRecord", "ElasticEvent", "ElasticTransferViolation",
    "elastic_t_range", "draw_elastic_t", "generate_elastic", "check_zero_transfer",
    "ALLOWED_EXCHANGES", "Exchange", "LedgerAuditReport",
    "NonSingletTransferError", "VACUUM", "ZERO_TRANSFER",
    "apply_transfer", "audit_event", "close_decay", "register_exchange",
    "REMNANT_KINDS", "AVAILABLE_REMNANT_KINDS", "KERNEL_BOUNDARY",
    "CONTINUUM_M_MAX", "REMNANT_QN",
    "ChargeExchangeNotImplemented", "RemnantSelection",
    "low_mass_records", "high_mass_records", "select_remnant",
    "PI0_MASS", "REJECTION_CAP", "MassLawParams",
    "PhaseSpaceRejectionExceeded",
    "sd_limits", "t_range", "marginal_density", "marginal_grid_mass",
    "draw_marginal", "draw_t", "joint_closure", "draw_joint_dd",
    "draw_central",
    "SDSIDES", "SD_CLASS_LABEL_PROJECTILE", "SD_CLASS_LABEL_TARGET",
    "NOMINAL_SD_REMNANT_WEIGHTS", "NOMINAL_PARAMS",
    "FinalHadron", "SDEvent", "gap_size", "generate_sd", "mirror_sd",
    "sd_kinematics",
    "DD_CLASS_LABEL", "DD_LINE_LABELS", "DDEvent",
    "central_gap", "dd_kinematics", "generate_dd",
    "ND_CLASS_LABEL", "ND_LINE_LABELS", "CENTRAL_LINE_LABEL",
    "CENTRAL_QN", "NOMINAL_ND_REMNANT_WEIGHTS",
    "CentralNeutralityViolation", "NDEvent",
    "check_central_neutrality", "generate_nd", "nd_kinematics",
    "nd_limits",
    "ND_MULTI_CLASS_LABEL", "COUNT_BND", "DEFAULT_P2", "P_N2_TOLERANCE",
    "NDMultiEvent", "REGISTRY_ENTRY", "check_multi_central_neutrality",
    "generate_nd_multi", "nd_multi_kinematics", "nd_multi_limits",
    "validate_p_n2",
    "TOPOLOGY_CLASSES", "INELASTIC_SET", "INELASTIC_ORDER", "MASTER_SEED",
    "NOMINAL_TOPOLOGY_PARAMS", "TopologyNominalParams",
    "TopologyObservables",
    "draw_topology", "run_topology", "extract_observables", "pinned_trace",
]
