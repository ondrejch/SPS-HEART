"""R65 driver: the flavor closure validation (plan section 12, run R65;
card TASK-20260928-r6-run-r65-flavor-closure-validation).  T3
statistics (20 x 250,000) at the frozen flavor point inherited from
R61-R64 on the R51b topology substrate; no free parameter.  The
comparison covers the K-, K+, antiproton spectra and the K+ - K-
difference, with the integrated ratios, the pion spectra, and the kaon
<p_T> moments as readouts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r65_flavor_closure_validation_2026-09-28", card="TASK-20260928-r6-run-r65-flavor-closure-validation",
    ledger_run="r65_flavor_closure_validation", gate_record="gate6_r65.json",
    raw_first=20303708, raw_last=20303727, partition_base=2_600_000_000, replicas=20, events_per_replica=250_000,
    statistics=({"name": "k_minus_spectrum", "species": "k_minus"}, {"name": "k_plus_spectrum", "species": "k_plus"},
                {"name": "antiproton_spectrum", "species": "antiproton"}, {"name": "k_plus_minus_k_minus", "species": "k_plus", "minus": "k_minus"}),
    free=(), inherit=(("gamma_s_had", "r61_pair_strangeness_fit_2026-09-28"), ("assoc_norm", "r62_associated_strangeness_fit_2026-09-28"),
                      ("r_v", "r63_vector_hidden_strangeness_fit_2026-09-28"), ("r_bbar", "r64_baryon_pair_fit_2026-09-28")),
    priors={}, starts=(), validation_only=True, max_generations=1,
    smoke_seed=20303910, smoke_partition=2_960_000_000,
    schema="na49_mixing.staged_r65_flavor_closure_validation",
    description="T3 validation at the frozen flavor point of R61-R64 on the R51b substrate; no free parameter.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
