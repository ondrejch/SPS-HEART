# SPS-HEART

Private snapshot of the files that reproduce the HEART paper.
Assembled by `helpers/publishing/sync_sps_heart.py` from the
development checkout. Re-run that script to refresh this tree.

Layout matches the development checkout:

- `dev/mixing2006/`: the generator (`na49_mixing`) and its tests
- `helpers/benchmarking/`: the fit driver
- `helpers/na49_gateway/`: the cluster client, example config only
- `dev/mixing2006/spshadrons.web.cern.ch/`: the SPS tables
- `dev/00runs/`: compact fit records

Manuscripts are not in this tree. They stay in their own repositories.

Compact run files copied: 436.
Event samples, npz slot caches, generator chunk files, and any
run file larger than 2 MiB are not in this snapshot. Regenerating
the generator comparison needs those samples from the development
checkout.

Not copied: `.legacy/`, cluster `config.na49_project.json`,
virtualenvs, and git history of the source trees.

Omitted run paths: 203.
