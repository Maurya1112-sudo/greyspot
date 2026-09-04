import pytest

from greyspot.ingest.boroughs import BOROUGHS, get_borough, slug


def test_registered_boroughs_have_distinct_ons_codes():
    expected = {
        "Westminster": "E09000033",
        "Lambeth": "E09000022",
        "Camden": "E09000007",
        "Kensington and Chelsea": "E09000020",
        "City of London": "E09000001",
        "Brent": "E09000005",
        "Wandsworth": "E09000032",
    }
    for name, code in expected.items():
        assert name in BOROUGHS
        assert BOROUGHS[name].ons_code == code
    all_codes = [b.ons_code for b in BOROUGHS.values()]
    assert len(all_codes) == len(set(all_codes))  # no duplicate/copy-paste ONS codes


def test_get_borough_returns_registered_entry():
    borough = get_borough("Westminster")
    assert borough.name == "Westminster"
    assert borough.osm_place


def test_get_borough_raises_helpful_error_for_unknown_name():
    with pytest.raises(ValueError, match="Unknown borough"):
        get_borough("Notaborough")


def test_slug_is_filesystem_safe():
    assert slug("City of Westminster") == "city_of_westminster"
    assert " " not in slug("London Borough of Lambeth")
