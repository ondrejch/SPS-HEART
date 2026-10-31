# 04-run_balanced_constrained_fit

Run the balanced constrained configuration used for the paper figures and tables.

Run:

```bash
./04-run_balanced_constrained_fit/run.sh
```

Configuration reproduced here:

- PDG composition prior
- `prior_abs_yield_lambda = 0.3`
- `prior_abs_yield_min_sigma_rel = 0.05`
- `fit_objective = gaussian_point_mle`
- `fit_weight_mode = sigma`
- `res_events_per_res = 120000`
- `res_max_decay_passes = 16`

Outputs:

- `00runs/constrained_refit_pdg_absyield_scan_l0300_s0050_<stamp>`
- `multispecies_constrained_summary.json`
- `multispecies_constrained_details.npz`
- `table9_pdg_fit_weight_compare.csv/.json`
