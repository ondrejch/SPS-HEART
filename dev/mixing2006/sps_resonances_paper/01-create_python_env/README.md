# 01-create_python_env

Create the dedicated Python environment used by all later steps.

Run:

```bash
./01-create_python_env/run.sh
```

Installed into:

- `VENV_DIR` (default: `<repo>/.venv-sps-resonances-paper`)

Packages installed:

- the exact package versions pinned in `../requirements-paper.lock.txt` (pip included); they reproduce the reference environment recorded in `../reproducibility.lock.json`
- editable install of the repo root with the `dev` extra (dependencies already pinned)

Notes:

- `REPRO_LOCK=0` installs latest-compatible packages from `pyproject.toml` and `pythia8mc` from `PYTHIA8MC_SPEC=...` instead; use it for library development, since it does not reproduce the reference environment.
- Building `pythia8mc` may require a local compiler toolchain.
- System LaTeX tools are not installed here; step `13-build_pdf` expects `latexmk`, `pdflatex`, and `bibtex` to already be available.
