"""R64 driver: the baryon-pair fit (plan section 12, run R64; card
TASK-20260928-r6-run-r64-baryon-pair-fit).  r_BBbar is fit to the NA49
antiproton spectrum (122 bins); the anti-Lambda branch has no local
table.  The readout records the proton and net-proton sums at the
optimum against the fixed point (the plan's check that p - pbar stays
controlled by baryonic-line transport rather than pair production)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r64_baryon_pair_fit_2026-09-28", card="TASK-20260928-r6-run-r64-baryon-pair-fit",
    ledger_run="r64_baryon_pair_fit", gate_record="gate6_r64.json",
    raw_first=20303700, raw_last=20303707, partition_base=2_500_000_000, replicas=8, events_per_replica=100_000,
    statistics=({"name": "antiproton_spectrum", "species": "antiproton"},), free=("r_bbar",),
    inherit=(("gamma_s_had", "r61_pair_strangeness_fit_2026-09-28"), ("assoc_norm", "r62_associated_strangeness_fit_2026-09-28"),
             ("r_v", "r63_vector_hidden_strangeness_fit_2026-09-28")), priors={},
    starts=((0.3,), (0.5,), (1.0,), (2.0,), (4.0,), (8.0,), (0.15,), (1.5,)),
    smoke_seed=20303908, smoke_partition=2_950_000_000,
    schema="na49_mixing.staged_r64_baryon_pair_fit",
    description="r_BBbar fit to the antiproton spectrum with the other flavor weights inherited; the p - pbar transport-control readout recorded.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
