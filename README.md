# SPS-HEART

![HEART model logo](assets/logo/heart_logo.png)

Private snapshot of the files that reproduce the HEART paper.
Assembled by `helpers/publishing/sync_sps_heart.py` from the
development checkout. Re-run that script to refresh this tree.

Layout matches the development checkout:

- `dev/mixing2006/`: the generator (`na49_mixing`) and its tests
- `helpers/benchmarking/`: the fit driver
- `helpers/na49_gateway/`: the cluster client, example config only
- `dev/mixing2006/spshadrons.web.cern.ch/`: the SPS tables
- `dev/mixing2006/config/runs/r136.json`: the paper fit description
- `dev/00runs/`: the report and probe files the table scripts read
- `dev/00runs/paper_record_r136/RECORD.json`: the result record of the fitted model,
  with the run identity, the code commits, the fitted parameters and errors, the
  objective by segment (checked to add to the total), the profiled scales, the
  tests, the variants, the readouts, and the hashes of every file the papers quote

Manuscripts are not in this tree. They stay in their own repositories.
Earlier run descriptions, later trials, and `staged_r<number>` drivers are not copied.
A rerun uses `r136.json`. It does not need the previous `r<number>.json` files.

Some tests pin assertions to the earlier run descriptions. Those files
are not copied, so those tests fail to collect or fail here. The fit
driver, the paper fit description, and the paper result files are complete.

Paper result files copied: 20.
Event samples and generator chunk files are not in this snapshot.
Regenerating the generator comparison needs those samples from the
development checkout.

Not copied: `.legacy/`, cluster `config.na49_project.json`,
virtualenvs, and git history of the source trees.
