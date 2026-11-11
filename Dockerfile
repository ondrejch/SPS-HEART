# HEART event generator: the fitted model of arXiv:2611.11570, ready to generate events.
#
#   docker run --rm -v "$PWD":/work ghcr.io/ondrejch/sps-heart:1.1.1 \
#       --events 1000 --workers 4 --out /work/heart_events.jsonl
#
# The image installs the generator package non-editable, so it reads the configuration bundled into the
# package, and carries the fit report that fixes the generation point.
FROM python:3.12-slim

ARG VERSION=1.1.1
LABEL org.opencontainers.image.source="https://github.com/ondrejch/SPS-HEART" \
      org.opencontainers.image.title="HEART" \
      org.opencontainers.image.description="HEART: an event-level resonance-cascade model of soft hadron production in p+p collisions at 158 GeV/c (arXiv:2611.11570)" \
      org.opencontainers.image.licenses="GPL-2.0-or-later" \
      org.opencontainers.image.version="${VERSION}"

COPY dev/mixing2006 /src/mixing2006
RUN pip install --no-cache-dir /src/mixing2006 \
 && rm -rf /src \
 && mkdir -p /opt/heart \
 && ln -s "$(python -c 'from na49_mixing._paths import nested_root; print(nested_root())')/config/runs/r146.json" \
          /opt/heart/r146.json
COPY dev/00runs/r146_regenerated_2026-11-06/report.json /opt/heart/r146_report.json

WORKDIR /work
ENTRYPOINT ["python", "-m", "helpers.benchmarking.heart_events", \
            "--spec", "/opt/heart/r146.json", "--report", "/opt/heart/r146_report.json"]
CMD ["--help"]
