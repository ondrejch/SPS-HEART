# 12-assemble_paper

Assemble a paper-asset bundle from the regenerated figures and tables.

Run:

```bash
./12-assemble_paper/run.sh
```

This step always writes:

- `00runs/paper_assets_<stamp>/fig/*.png`
- `00runs/paper_assets_<stamp>/generated/table_wrrmse_rows.tex`
- `00runs/paper_assets_<stamp>/generated/table_active_coeff_rows.tex`
- `00runs/paper_assets_<stamp>/generated/table_ratio_rows.tex`

If `PAPER_ROOT` points to an external paper tree, the same assets are also synced into:

- `<PAPER_ROOT>/fig`
- `<PAPER_ROOT>/generated`
