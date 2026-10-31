# 08-scan_nstar_extended_levels

Run the extended N* ladder scan locally and build the three-way likelihood map used in the paper.

Run:

```bash
./08-scan_nstar_extended_levels/run.sh
```

This step runs three local grids:

- unconstrained
- Table-9-constrained
- PDG-constrained

Grid reproduced here:

- `tower_max_values = 1.9,2.1,2.3,2.5,2.7,2.9,3.1`
- `level_counts = 3,4,5,7,10,12,15,20`
- `shapes = linear,regge_m2,hagedorn:0.12,hagedorn:0.16,hagedorn:0.25`
- `fit_objective = nnls_weighted`
- `fit_weight_mode = sigma`
- `res_events_per_res = 120000`

This is the slowest pipeline stage.
