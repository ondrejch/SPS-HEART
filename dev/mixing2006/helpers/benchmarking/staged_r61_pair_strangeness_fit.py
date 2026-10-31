"""R61 driver: the pair-strangeness fit (plan section 12, run R61; card
TASK-20260928-r6-run-r61-pair-strangeness-fit).  gamma_s^had is fit to
the NA49 K- spectrum (148 paper-window bins) on the R51b topology
substrate; associated strangeness, vector, baryon-pair, and topology
parameters are frozen; K+, pions, antiprotons, and the kaon <p_T>
moments are held out.  K_S^0 and central K Kbar yields have no local
table and are recorded absent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r61_pair_strangeness_fit_2026-09-28", card="TASK-20260928-r6-run-r61-pair-strangeness-fit",
    ledger_run="r61_pair_strangeness_fit", gate_record="gate6_r61.json",
    raw_first=20303676, raw_last=20303683, partition_base=2_200_000_000, replicas=8, events_per_replica=100_000,
    statistics=({"name": "k_minus_spectrum", "species": "k_minus"},), free=("gamma_s_had",), inherit=(), priors={},
    starts=((0.3,), (0.5,), (0.7,), (0.85,), (1.0,), (1.4,), (2.0,), (0.6,)),
    smoke_seed=20303902, smoke_partition=2_920_000_000,
    schema="na49_mixing.staged_r61_pair_strangeness_fit",
    description="gamma_s^had fit to the K- spectrum; K+, pions, antiprotons, and the kaon <p_T> moments held out; K_S^0 and central K Kbar yields absent locally.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
