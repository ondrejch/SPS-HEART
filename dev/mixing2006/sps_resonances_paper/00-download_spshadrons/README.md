# 00-download_spshadrons

Refresh the SPS HADRONS inputs used by the benchmark scripts.

This step downloads the differential and pT-integrated NA49 tables, the key SPS HADRONS landing pages, and the source PDFs referenced by the study.

Run:

```bash
./00-download_spshadrons/run.sh
```

Notes:

- The files are written into `spshadrons.web.cern.ch` in the repo root.
- Every download is verified against the sha256 recorded in `../reproducibility.lock.json` before it is moved into place; an existing file whose hash differs from the manifest is refused until it is deleted explicitly.
- `CHECK_ONLY=1` verifies the sha256 of every file recorded in the manifest, prints one PASS/FAIL line per file, and downloads nothing.
- Existing verified files are kept; `FORCE_DOWNLOAD=1` re-downloads them, and a replacement is installed only after the fresh copy passes the sha256 check.
- This step needs `curl`, `python3`, `sha256sum`, and network access.
