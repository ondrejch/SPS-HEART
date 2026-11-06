# The legacy p+A path (marked 2026-11-04): the HEART p+p fits do not run this module; the p+A builder
# (event_builder_pA, parallel, glauber) and its imports do. Code reviews of the p+p model can skip it; review it
# with the p+A work. The cleanup and refactor after the p+A run decides its future.
"""Physical constants used throughout the NA49 analysis framework.

Ported from ``inc/definicie.h``.
"""

EMELEK: float = 0.00051099   # electron mass [GeV/c²]
EMPI:   float = 0.1395699    # charged pion mass [GeV/c²]
EMKA:   float = 0.493677     # charged kaon mass [GeV/c²]
EMPROT: float = 0.93827231   # proton mass [GeV/c²]
EMNEUT: float = 0.93956541   # neutron mass [GeV/c²]
EMRHO:  float = 0.77526      # rho(770) mass [GeV/c²]
EMOMEGA: float = 0.78265     # omega(782) mass [GeV/c²]
EMETA:  float = 0.547862     # eta mass [GeV/c²]
