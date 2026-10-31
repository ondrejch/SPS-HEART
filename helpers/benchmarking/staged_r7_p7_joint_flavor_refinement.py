"""Round 7 fit P7 driver: the limited joint refinement (plan section 13.2,
P7; card TASK-20260928-r7-fit-p7-joint-flavor-refinement).  Releases the
two flavor parameters that share training data through the strange
sector, gamma_s^had and r_BBbar, jointly on the K-, K+ - K-, and
antiproton statistics, initialized from the R61 and R64 optima with the
associated and vector parameters inherited (R62, R63); the R51b topology
point stays frozen (the plan's P7 releases no non-identifiable
parameter)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r7_p7_joint_flavor_refinement_2026-09-28", card="TASK-20260928-r7-fit-p7-joint-flavor-refinement",
    ledger_run="r7_p7_joint_flavor_refinement", gate_record="gate7_p7.json",
    raw_first=20303728, raw_last=20303735, partition_base=2_700_000_000, replicas=8, events_per_replica=100_000,
    statistics=({"name": "k_minus_spectrum", "species": "k_minus"}, {"name": "k_plus_minus_k_minus", "species": "k_plus", "minus": "k_minus"},
                {"name": "antiproton_spectrum", "species": "antiproton"}),
    free=("gamma_s_had", "r_bbar"),
    inherit=(("gamma_s_had", "r61_pair_strangeness_fit_2026-09-28"), ("assoc_norm", "r62_associated_strangeness_fit_2026-09-28"),
             ("r_v", "r63_vector_hidden_strangeness_fit_2026-09-28"), ("r_bbar", "r64_baryon_pair_fit_2026-09-28")),
    priors={},
    starts=((1.0, 1.0), (0.7, 0.5), (1.5, 2.0), (2.0, 1.0), (1.0, 4.0), (3.0, 0.5), (0.5, 8.0), (1.2, 1.5)),
    smoke_seed=20303912, smoke_partition=2_970_000_000,
    schema="na49_mixing.staged_r7_p7_joint_flavor_refinement",
    description="P7: gamma_s^had and r_BBbar released jointly on K-, K+ - K-, and antiproton; assoc_norm and r_V inherited; topology frozen at R51b.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
