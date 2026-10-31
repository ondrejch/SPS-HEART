# 02-build_external_generators

Bootstrap and build the external event generators used by the local baseline benchmark.

Run:

```bash
./02-build_external_generators/run.sh
```

Expected local layout after the step finishes:

- `external/urqmd-4.0`
- `external/HepMC-2.06.11`
- `external/crmc`
- `external/hepmc2-install`
- `external/crmc/install`

How sources are obtained:

- HepMC2 is downloaded automatically into `external/dist` if missing; the downloaded archive must match the sha256 recorded in `../reproducibility.lock.json`. While no hash is recorded there, the step downloads the archive once, prints its sha256, and stops; record the value in the manifest and rerun to install it verified.
- CRMC is fetched at the commit recorded under `generators.crmc.pinned_commit` in `../reproducibility.lock.json`, and `git rev-parse HEAD` must match that commit after checkout. While no commit is recorded, the step clones the tag, prints the resolved commit, and asks that it be recorded. `CRMC_COMMIT=<sha>` overrides the pin and explicitly breaks reproducibility.
- UrQMD is not redistributed by this repo. Provide either:
  - `URQMD_TARBALL=external/dist/urqmd-4.0.tar.gz` (verified against `downloads.urqmd_source_tarball.sha256` in the manifest when a hash is recorded there)
  - `URQMD_SOURCE_DIR=/absolute/path/to/urqmd-4.0`

Outputs:

- `external/urqmd-4.0/urqmd.x86_64`
- `external/hepmc2-install`
- `external/crmc/install`

Notes:

- This step needs `curl`, `git`, `tar`, `make`, and `cmake`.
- Set `MAKE_JOBS=<n>` to control build parallelism.
- `FORCE_BUILD=1` rebuilds binaries from the existing source trees.
