# 06-scan_absyield_prior_stiffness

Run the absolute-yield prior stiffness scan and summarize it as the paper heatmap.

Run:

```bash
./06-scan_absyield_prior_stiffness/run.sh
```

Grid reproduced here:

- modes: `table9`, `pdg`
- `prior_abs_yield_lambda`: `0.1`, `0.3`, `1.0`
- `prior_abs_yield_min_sigma_rel`: `0.02`, `0.05`, `0.1`

Outputs:

- per-scenario run directories named `constrained_refit_<mode>_absyield_scan_lXXXX_sXXXX_<stamp>`
- summary directory `00runs/absyield_prior_stiffness_<stamp>`
