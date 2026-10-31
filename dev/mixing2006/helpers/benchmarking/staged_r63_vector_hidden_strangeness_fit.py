"""R63 driver: the vector and hidden-strangeness fit (plan section 12, run
R63; card TASK-20260928-r6-run-r63-vector-hidden-strangeness-fit).  No
K* or phi table exists locally, so the plan's prior-dominance branch
applies: r_V carries a log-normal external prior (center 1, sigma_ln
0.5) and is fit to the K+ and K- spectra, which the vector channels feed
through K* -> K pi; the report states whether the posterior width is
prior dominated."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavor_run_driver import RunSpec, main  # noqa: E402

SPEC = RunSpec(
    run_id="r63_vector_hidden_strangeness_fit_2026-09-28", card="TASK-20260928-r6-run-r63-vector-hidden-strangeness-fit",
    ledger_run="r63_vector_hidden_strangeness_fit", gate_record="gate6_r63.json",
    raw_first=20303692, raw_last=20303699, partition_base=2_400_000_000, replicas=8, events_per_replica=100_000,
    statistics=({"name": "k_minus_spectrum", "species": "k_minus"}, {"name": "k_plus_spectrum", "species": "k_plus"}), free=("r_v",),
    inherit=(("gamma_s_had", "r61_pair_strangeness_fit_2026-09-28"), ("assoc_norm", "r62_associated_strangeness_fit_2026-09-28")),
    priors={"r_v": (1.0, 0.5)},
    starts=((0.5,), (0.7,), (1.0,), (1.5,), (2.0,), (3.0,), (0.3,), (1.2,)),
    smoke_seed=20303906, smoke_partition=2_940_000_000,
    schema="na49_mixing.staged_r63_vector_hidden_strangeness_fit",
    description="r_V with an external log-normal prior (1.0, sigma_ln 0.5) fit to the K+- spectra in the absence of K*/phi data; prior dominance reported.",
)

if __name__ == "__main__":
    raise SystemExit(main(SPEC))
