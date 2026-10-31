# 13-build_pdf

Optionally compile an external paper tree with the regenerated figures and generated table snippets.

Run:

```bash
./13-build_pdf/run.sh
```

Prerequisites:

- `latexmk`
- `pdflatex`
- `bibtex`
- `PAPER_ROOT` must point to a paper tree containing `main.tex`

Output:

- `<PAPER_ROOT>/main.pdf`

If `PAPER_ROOT` is unset and `../../latex/paper` is not present, this step exits successfully without building anything.
