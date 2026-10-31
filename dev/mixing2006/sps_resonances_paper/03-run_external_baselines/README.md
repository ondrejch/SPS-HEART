# 03-run_external_baselines

Generate the local external baseline samples used by the constrained multispecies fit.

Run:

```bash
./03-run_external_baselines/run.sh
```

What this step does:

1. Run UrQMD chunk jobs locally.
2. Run EPOS-LHC chunk jobs locally through CRMC.
3. Run PYTHIA8 SoftQCD chunk jobs locally.
4. Merge the chunk CSVs into the `*_gateway.csv` filenames expected by the constrained benchmark scripts.
5. Run per-species baseline QC with `helpers/benchmarking/benchmark_sps_species.py`.

Key environment overrides:

- `CHUNK_COUNT` (default `32`)
- `EVENTS_PER_CHUNK` (default `5000`)
- `PARALLEL_CHUNKS` (default `1`)
- `SKIP_EXISTING` (default `1`)

Outputs:

- `00runs/local_multispecies_allmodels_singlepass_pythia_<stamp>`
- merged files such as `pythia8_softqcd_pp158_piminus_gateway.csv`
- QC summaries such as `na49_benchmark_piminus_summary.json`
