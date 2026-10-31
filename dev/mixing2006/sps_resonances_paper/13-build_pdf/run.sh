#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/common.sh"

require_cmd latexmk
require_cmd pdflatex
require_cmd bibtex

if [[ -z "$PAPER_ROOT" || ! -f "$PAPER_ROOT/main.tex" ]]; then
  log "skipping paper PDF build; set PAPER_ROOT to an external latex/paper tree with main.tex"
  exit 0
fi

log "building paper PDF"
(
  cd "$PAPER_ROOT"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)

log "paper PDF ready: $PAPER_ROOT/main.pdf"
