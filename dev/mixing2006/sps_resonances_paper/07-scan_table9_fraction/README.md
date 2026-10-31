# 07-scan_table9_fraction

Run the nonlinear Table-9 fraction scan and summarize it as the paper figure.

Run:

```bash
./07-scan_table9_fraction/run.sh
```

Grid reproduced here:

- modes: `table9`, `pdg`
- `prior_table9_fraction_lambda`: `0`, `1`, `10`, `100`, `1000`, `10000`, `1000000`
- fixed `prior_abs_yield_lambda = 0.3`
- fixed `prior_abs_yield_min_sigma_rel = 0.05`
