# The snapshot

This repository is assembled from the development checkout of the project by
`helpers/publishing/sync_sps_heart.py`, which is kept in that checkout. The layout matches the
development checkout, so every path in the code and in these pages is valid here.

## Contents

- `dev/mixing2006/`: the generator package `na49_mixing`, its configuration, and its tests
- `dev/mixing2006/spshadrons.web.cern.ch/`: the NA49 and SPS data tables
- `dev/mixing2006/config/runs/`: the run descriptions of the project's fits; `r141.json` is the paper's
- `helpers/benchmarking/`: the fit driver, the event writer, and their tests
- `helpers/na49_gateway/`: the cluster client, with an example configuration only
- `dev/00runs/`: 20 result files of the paper's fit, namely its report, the report and
  manifest of the earlier fit its first generation starts from, the readouts and statistics the
  papers quote, and the result record `paper_record_r141/RECORD.json`

## Left out

- the event samples of the fits (about 8 GB for the paper's fit); a rerun regenerates them from the seeds
- the manuscripts, which are kept in their own repositories
- the historical C++/ROOT implementation (`.legacy/`), virtual environments, and the git history
  of the development trees
- the real cluster configuration (`config.na49_project.json`)

## Tests

Test modules that need inputs this tree leaves out: 0.
