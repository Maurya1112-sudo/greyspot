"""Registry of London boroughs Greyspot can run against.

The single place to add a new borough when expanding beyond Westminster.
Adding one is a three-line addition here, not a code change elsewhere -
`ingest/stats19.py`, `ingest/network.py` and `scripts/run_pipeline.py` all
take a `Borough` rather than hardcoding a name/ONS code/place string. This
is what makes the "Westminster now, London later" plan (see
`docs/project_management.md`) a config change, not a rewrite.

ONS codes verified against ons.gov.uk on 2026-08-31 (Westminster:
confirmed live in the STATS19 file itself; Lambeth: confirmed via
ons.gov.uk/explore-local-statistics/areas/E09000022-lambeth). Add new
boroughs the same way - verify the code before trusting it, STATS19
filtering silently returns zero rows for a wrong one rather than erroring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Borough:
    name: str
    ons_code: str  # STATS19 local_authority_ons_district value
    osm_place: str  # OSMnx geocodable place name


BOROUGHS: dict[str, Borough] = {
    "Westminster": Borough(
        name="Westminster",
        ons_code="E09000033",
        osm_place="City of Westminster, London, United Kingdom",
    ),
    # Lambeth is deliberately the first expansion target: it's the case
    # study borough of the academic precedent (Gao et al. 2024), so results
    # here are directly comparable to published literature, not just an
    # arbitrary second data point.
    "Lambeth": Borough(
        name="Lambeth",
        ons_code="E09000022",
        osm_place="London Borough of Lambeth, London, United Kingdom",
    ),
    # Westminster's actual neighbouring boroughs (verified 2026-08-31):
    # City of London (east), Camden, Brent, Kensington & Chelsea, and
    # Wandsworth/Lambeth (south, across the Thames). Registered with
    # verified ONS codes so the "Westminster -> surrounding boroughs -> all
    # of London" plan (docs/scaling_to_london.md) can proceed one config
    # entry at a time - not all have been run through the pipeline yet.
    "Camden": Borough(
        name="Camden",
        ons_code="E09000007",
        osm_place="London Borough of Camden, London, United Kingdom",
    ),
    "Kensington and Chelsea": Borough(
        name="Kensington and Chelsea",
        ons_code="E09000020",
        osm_place="Royal Borough of Kensington and Chelsea, London, United Kingdom",
    ),
    "City of London": Borough(
        name="City of London",
        ons_code="E09000001",
        osm_place="City of London, London, United Kingdom",
    ),
    "Brent": Borough(
        name="Brent",
        ons_code="E09000005",
        osm_place="London Borough of Brent, London, United Kingdom",
    ),
    "Wandsworth": Borough(
        name="Wandsworth",
        ons_code="E09000032",
        osm_place="London Borough of Wandsworth, London, United Kingdom",
    ),
    # Tower Hamlets is Gao et al. (2024)'s OWN third case-study region
    # (thesis Table 7.2, alongside Westminster and Lambeth) - added
    # 2026-09-02 to extend the POI/socio-demographic cross-borough check
    # to a third, independent borough, not part of the "Westminster's
    # neighbouring boroughs" expansion plan above. ONS code verified
    # directly from this project's own cached IMD 2019 data
    # (`data/raw/imd2019_london_lsoa.xlsx`'s "Local Authority District
    # name (2019)"/"...code (2019)" columns - 144 matching LSOA rows),
    # not assumed from memory.
    "Tower Hamlets": Borough(
        name="Tower Hamlets",
        ons_code="E09000030",
        osm_place="London Borough of Tower Hamlets, London, United Kingdom",
    ),
}
# Westminster's full ring of neighbours, for reference: City of London,
# Camden, Brent, Kensington and Chelsea, Wandsworth, Lambeth (Wandsworth
# and Lambeth are separated from Westminster by the River Thames). Source:
# verified via web search 2026-08-31, cross-checked against
# ons.gov.uk-derived GSS codes for each.


def get_borough(name: str) -> Borough:
    try:
        return BOROUGHS[name]
    except KeyError:
        raise ValueError(
            f"Unknown borough {name!r}. Known: {sorted(BOROUGHS)}. "
            "Add it to greyspot.ingest.boroughs.BOROUGHS (verify its ONS code first)."
        ) from None


def slug(name: str) -> str:
    """Filesystem/cache-safe borough identifier, e.g. for graph cache filenames."""
    return name.lower().replace(" ", "_")
