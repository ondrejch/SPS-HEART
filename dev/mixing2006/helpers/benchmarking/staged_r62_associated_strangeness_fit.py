"""R62 driver: the associated-strangeness fit (plan section 12, run R62;
card TASK-20260928-r6-run-r62-associated-strangeness-fit), conditional
on the R61 held-out K+ pull rms exceeding 3.0.  The associated
normalization is fit to the K+ - K- difference on the common bins with
gamma_s^had inherited from R61; the Lambda branch has no local table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r62_associated_strangeness_fit_2026-09-28", card="TASK-20260928-r6-run-r62-associated-strangeness-fit",
    ledger_run="r62_associated_strangeness_fit", gate_record="gate6_r62.json",
    raw_first=20303684, raw_last=20303691, partition_base=2_300_000_000, replicas=8, events_per_replica=100_000,
    statistics=({"name": "k_plus_minus_k_minus", "species": "k_plus", "minus": "k_minus"},), free=("assoc_norm",),
    inherit=(("gamma_s_had", "r61_pair_strangeness_fit_2026-09-28"),), priors={},
    starts=((0.5,), (0.7,), (1.0,), (1.5,), (2.0,), (3.0,), (0.3,), (1.2,)),
    smoke_seed=20303904, smoke_partition=2_930_000_000,
    conditional_trigger={"source_run": "r61_pair_strangeness_fit_2026-09-28", "readout_species": "k_plus", "pull_rms_threshold": 3.0,
                         "rule": "R62 executes only if the R61 optimum leaves the held-out K+ spectrum with a pull rms above 3.0"},
    schema="na49_mixing.staged_r62_associated_strangeness_fit",
    description="associated-strangeness normalization fit to K+ - K- (common bins) with gamma_s^had inherited from R61; conditional on the R61 K+ trigger.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
