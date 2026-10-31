# SPS-HEART

Private snapshot of the files that reproduce the HEART paper.
Assembled by `helpers/publishing/sync_sps_heart.py` from the
development checkout. Re-run that script to refresh this tree.

Layout matches the development checkout:

- `dev/mixing2006/`: the generator (`na49_mixing`) and its tests
- `helpers/benchmarking/`: the fit driver
- `helpers/na49_gateway/`: the cluster client, example config only
- `dev/mixing2006/spshadrons.web.cern.ch/`: the SPS tables
- `dev/mixing2006/config/runs/r128.json`: the paper fit description
- `dev/00runs/`: the report and probe files the table scripts read

Manuscripts are not in this tree. They stay in their own repositories.
Earlier run descriptions, and later trials, are not copied.
A rerun uses `r128.json`. It does not need the previous `r<number>.json` files.

Paper result files copied: 9.
Event samples and generator chunk files are not in this snapshot.
Regenerating the generator comparison needs those samples from the
development checkout.

Not copied: `.legacy/`, cluster `config.na49_project.json`,
virtualenvs, and git history of the source trees.
