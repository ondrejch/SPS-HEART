"""Shared particle/species mappings for SPS benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Species:
    name: str
    pdg: int
    urqmd_ityp: int
    urqmd_iso3: int
    na49_table: str
    tag: str


REPO_ROOT = Path(__file__).resolve().parents[2]
NA49_DATA_DIR = REPO_ROOT / "spshadrons.web.cern.ch" / "data"
PA_DATA_DIR = REPO_ROOT / "helpers" / "benchmarking" / "pa_data"


_SPECIES_BY_KEY = {
    "pi-": Species(
        name="pi-",
        pdg=-211,
        urqmd_ityp=101,
        urqmd_iso3=-2,
        na49_table=str(NA49_DATA_DIR / "xsect_piminus.txt"),
        tag="piminus",
    ),
    "piminus": Species(
        name="pi-",
        pdg=-211,
        urqmd_ityp=101,
        urqmd_iso3=-2,
        na49_table=str(NA49_DATA_DIR / "xsect_piminus.txt"),
        tag="piminus",
    ),
    "pi+": Species(
        name="pi+",
        pdg=211,
        urqmd_ityp=101,
        urqmd_iso3=2,
        na49_table=str(NA49_DATA_DIR / "xsect_piplus.txt"),
        tag="piplus",
    ),
    "piplus": Species(
        name="pi+",
        pdg=211,
        urqmd_ityp=101,
        urqmd_iso3=2,
        na49_table=str(NA49_DATA_DIR / "xsect_piplus.txt"),
        tag="piplus",
    ),
    "k+": Species(
        name="K+",
        pdg=321,
        urqmd_ityp=106,
        urqmd_iso3=1,
        na49_table=str(NA49_DATA_DIR / "xsect_kaplus_pp.txt"),
        tag="kaplus",
    ),
    "kaplus": Species(
        name="K+",
        pdg=321,
        urqmd_ityp=106,
        urqmd_iso3=1,
        na49_table=str(NA49_DATA_DIR / "xsect_kaplus_pp.txt"),
        tag="kaplus",
    ),
    "k-": Species(
        name="K-",
        pdg=-321,
        urqmd_ityp=-106,
        urqmd_iso3=-1,
        na49_table=str(NA49_DATA_DIR / "xsect_kaminus_pp.txt"),
        tag="kaminus",
    ),
    "kaminus": Species(
        name="K-",
        pdg=-321,
        urqmd_ityp=-106,
        urqmd_iso3=-1,
        na49_table=str(NA49_DATA_DIR / "xsect_kaminus_pp.txt"),
        tag="kaminus",
    ),
    "p": Species(
        name="p",
        pdg=2212,
        urqmd_ityp=1,
        urqmd_iso3=1,
        na49_table=str(NA49_DATA_DIR / "xsect_proton_pp.txt"),
        tag="proton",
    ),
    "proton": Species(
        name="p",
        pdg=2212,
        urqmd_ityp=1,
        urqmd_iso3=1,
        na49_table=str(NA49_DATA_DIR / "xsect_proton_pp.txt"),
        tag="proton",
    ),
    "pbar": Species(
        name="pbar",
        pdg=-2212,
        urqmd_ityp=-1,
        urqmd_iso3=-1,
        na49_table=str(NA49_DATA_DIR / "xsect_antiproton_pp.txt"),
        tag="antiproton",
    ),
    "antiproton": Species(
        name="pbar",
        pdg=-2212,
        urqmd_ityp=-1,
        urqmd_iso3=-1,
        na49_table=str(NA49_DATA_DIR / "xsect_antiproton_pp.txt"),
        tag="antiproton",
    ),
}


def canonical_species_keys() -> list[str]:
    return ["pi-", "pi+", "k+", "k-", "p", "pbar"]


def resolve_species(key: str) -> Species:
    spec = _SPECIES_BY_KEY.get(key.strip().lower())
    if spec is None:
        raise KeyError(
            f"Unknown species {key!r}. Expected one of: {', '.join(canonical_species_keys())} "
            "(plus aliases piminus/piplus/kaplus/kaminus/proton/antiproton)."
        )
    return spec


# NA49 p+C invariant-cross-section tables on the spshadrons mirror.
# Kaons were not published for p+C in this data product.
_PC_TABLE_BY_NAME = {
    "pi-": "xsect_piminus_pC.txt",
    "pi+": "xsect_piplus_pC.txt",
    "p": "xsect_proton_pC.txt",
    "pbar": "xsect_antiproton_pC.txt",
}

# NA49 p+Pb charged-pion tables extracted from O. Chvala's PhD thesis
# (2006), Tab. tabicmb (minimum bias, feed-down corrected); converted
# into the framework row layout under helpers/benchmarking/pa_data/.
_PPB_TABLE_BY_NAME = {
    "pi-": "xsect_piminus_pPb.txt",
    "pi+": "xsect_piplus_pPb.txt",
}


def available_data_systems() -> tuple[str, ...]:
    return ("pp", "pC", "pPb")


def na49_table_for(key: str, system: str = "pp") -> str:
    """Resolve the NA49 invariant-cross-section table for *key*/*system*.

    ``system='pp'`` keeps the historical per-species mapping;
    ``system='pC'`` switches to the p+C tables on the spshadrons mirror;
    ``system='pPb'`` uses the thesis-extracted minimum-bias pion tables
    (charged pions only).
    """
    spec = resolve_species(key)
    if system in ("", "pp"):
        return spec.na49_table
    if system == "pC":
        fname = _PC_TABLE_BY_NAME.get(spec.name)
        source_dir = NA49_DATA_DIR
        available = sorted(_PC_TABLE_BY_NAME)
    elif system == "pPb":
        fname = _PPB_TABLE_BY_NAME.get(spec.name)
        source_dir = PA_DATA_DIR
        available = sorted(_PPB_TABLE_BY_NAME)
    else:
        raise ValueError(
            f"Unknown data system {system!r}; expected one of "
            f"{list(available_data_systems())}"
        )
    if fname is None:
        raise ValueError(
            f"{system} data not available for species {spec.name!r}; "
            f"available {system} species: {available}"
        )
    return str(source_dir / fname)
