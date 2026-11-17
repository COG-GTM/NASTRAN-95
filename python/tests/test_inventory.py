import pytest

from nastran95py.reference.inventory import (
    RF1_DECKS,
    TARGET_ELEMENTS,
    build_deck,
    build_manifest,
    main,
    repo_root,
)

EXPECTED_ELEMENTS = {
    "d01011a": {"CROD": 24, "CONROD": 36, "CSHEAR": 36, "CTRMEM": 3, "CQDMEM": 14},
    "d01012a": {"CROD": 24, "CONROD": 36, "CSHEAR": 36, "CTRMEM": 3, "CQDMEM1": 7, "CQDMEM2": 7},
    "d01013a": {"CROD": 24, "CONROD": 36, "CSHEAR": 36, "CTRMEM": 3, "CQDMEM1": 14},
    "d01014a": {"CROD": 24, "CONROD": 36, "CSHEAR": 36, "CTRMEM": 3, "CQDMEM2": 14},
    "d01032a": {"CQDMEM1": 216},
    "d01033a": {"CQDMEM2": 216},
    "d01034a": {"CQDMEM2": 216},
    "d01161a": {"CROD": 5, "CTRMEM": 38, "CQDMEM": 11},
    "t01241a": {"CROD": 3},
}

EXPECTED_PROPERTIES = {
    "d01011a": {"PROD", "PSHEAR", "PTRMEM", "PQDMEM"},
    "d01012a": {"PROD", "PSHEAR", "PTRMEM", "PQDMEM1", "PQDMEM2"},
    "d01013a": {"PROD", "PSHEAR", "PTRMEM", "PQDMEM1"},
    "d01014a": {"PROD", "PSHEAR", "PTRMEM", "PQDMEM2"},
    "d01032a": {"PQDMEM1"},
    "d01033a": {"PQDMEM2"},
    "d01034a": {"PQDMEM2"},
    "d01161a": {"PROD", "PTRMEM", "PQDMEM"},
    "t01241a": {"PROD"},
}

EXPECTED_LOADS = {
    "d01011a": {"FORCE"},
    "d01012a": {"FORCE"},
    "d01013a": {"FORCE"},
    "d01014a": {"FORCE"},
    "d01032a": {"TEMP"},
    "d01033a": {"TEMP"},
    "d01034a": {"TEMP"},
    "d01161a": {"FORCE", "TEMPD"},
    "t01241a": {"FORCE"},
}

EXPECTED_SUBCASES = {"d01012a": [1, 2], "d01161a": [10, 12]}


def test_manifest_lists_the_nine_decks(manifest):
    assert [d["id"] for d in manifest["decks"]] == list(RF1_DECKS)
    assert manifest["rigid_format"] == 1
    assert manifest["fortran_run"] is False
    assert "SPARC" in manifest["fortran_run_reason"]


def test_manifest_is_current():
    assert main(["--check"]) == 0, "run: python -m nastran95py.reference.inventory"


@pytest.mark.parametrize("deck_id", list(RF1_DECKS))
def test_deck_is_rigid_format_1(decks, deck_id):
    ex = decks[deck_id]["executive"]
    assert ex["app"] == "DISPLACEMENT"
    assert ex["rigid_format"] == 1


@pytest.mark.parametrize("deck_id", list(RF1_DECKS))
def test_element_counts(decks, deck_id):
    elements = decks[deck_id]["elements"]
    assert {k: v["count"] for k, v in elements.items()} == EXPECTED_ELEMENTS[deck_id]
    for name, e in elements.items():
        assert name in TARGET_ELEMENTS
        assert len(e["ids"]) == len(set(e["ids"])) == e["count"]
        assert decks[deck_id]["card_counts"][name] >= 1


@pytest.mark.parametrize("deck_id", list(RF1_DECKS))
def test_properties_and_materials_resolve(decks, deck_id):
    d = decks[deck_id]
    assert set(d["properties"]) == EXPECTED_PROPERTIES[deck_id]
    pids = {p["PID"]: p for rows in d["properties"].values() for p in rows}
    mids = {m["MID"] for name in ("MAT1", "MAT2") for m in d["materials"].get(name, [])}
    assert mids, "every deck defines MAT1 or MAT2"
    for name, e in d["elements"].items():
        if name == "CONROD":
            assert {c["mid"] for c in e["connectivity"]} <= mids
            continue
        for pid in e["property_ids"]:
            assert pid in pids, f"{name} references undefined property {pid}"
            assert pids[pid]["MID"] in mids


@pytest.mark.parametrize("deck_id", list(RF1_DECKS))
def test_loads_selected_by_case_control(decks, deck_id):
    d = decks[deck_id]
    assert set(d["loads"]) - {"LOAD"} == EXPECTED_LOADS[deck_id]
    subcases = d["case_control"]["subcases"]
    assert [s["id"] for s in subcases] == EXPECTED_SUBCASES.get(deck_id, [1])
    load_sids = {f["SID"] for name in ("FORCE", "MOMENT") for f in d["loads"].get(name, [])}
    load_sids |= {c["SID"] for c in d["loads"].get("LOAD", [])}
    temp_sids = {int(k) for k in d["loads"].get("TEMP", {})}
    temp_sids |= {int(k) for k in d["loads"].get("TEMPD", {})}
    for sc in subcases:
        sel = sc["selectors"]
        assert "LOAD" in sel or "TEMPERATURE" in sel
        if "LOAD" in sel:
            assert sel["LOAD"]["set"] in load_sids
        if "TEMPERATURE" in sel:
            assert sel["TEMPERATURE"]["set"] in temp_sids
        assert str(sel["SPC"]["set"]) in d["constraints"]["SPC1"]


@pytest.mark.parametrize("deck_id", [d for d in RF1_DECKS if d != "t01241a"])
def test_output_requests_have_listing_sections(decks, deck_id):
    d = decks[deck_id]
    kinds = {s["kind"] for s in d["listing"]["sections"]}
    requested = {k for sc in d["case_control"]["subcases"] for k in sc["outputs"]}
    assert {"DISPLACEMENT", "STRESS"} <= requested
    mapping = {
        "DISPLACEMENT": "displacement",
        "SPCFORCE": "spcforce",
        "STRESS": "stress",
        "FORCE": "force",
        "OLOAD": "oload",
        "ESE": "ese",
        "GPFORCE": "gpforce",
    }
    for request in requested:
        assert mapping[request] in kinds, f"{request} requested but not in listing"
    assert d["listing"]["fatal_messages"] == 0


@pytest.mark.parametrize("deck_id", [d for d in RF1_DECKS if d != "t01241a"])
def test_listing_prints_exactly_the_requested_ids(decks, deck_id):
    d = decks[deck_id]
    checks = d["checks"]["listing_coverage"]
    expected = {
        (sc["id"], req)
        for sc in d["case_control"]["subcases"]
        for req in ("DISPLACEMENT", "STRESS")
        if req in sc["outputs"]
    }
    assert {(c["subcase"], c["request"]) for c in checks} == expected
    for c in checks:
        assert c["missing_from_listing"] == [], c
        assert c["unrequested_in_listing"] == [], c
        assert c["listed"] == c["requested"] > 0


@pytest.mark.parametrize("deck_id", list(RF1_DECKS))
def test_bulk_echo_matches_input_deck(decks, deck_id):
    checks = decks[deck_id]["checks"]
    assert checks["bulk_echo_matches_deck"]
    assert checks["bulk_echo_cards"] == checks["deck_physical_cards"] > 0


def test_stress_sections_cover_every_listed_target_element(decks):
    for deck_id in ("d01011a", "d01012a", "d01013a", "d01014a"):
        types = {
            s["element_type"]
            for s in decks[deck_id]["listing"]["sections"]
            if s["kind"] == "stress"
        }
        expected = {
            t for t in EXPECTED_ELEMENTS[deck_id] if t.startswith(("CQDMEM", "CTRMEM", "CSHEAR"))
        }
        assert types == expected


def test_d01012a_sort2_has_both_subcases(decks):
    for s in decks["d01012a"]["listing"]["sections"]:
        if s["kind"] in ("displacement", "stress"):
            assert s["sort"] == "SORT2"
            assert sorted(s["subcases_in_rows"]) == [1, 2]


def test_d01161a_design_iterations(decks):
    occ = [
        s["occurrence"]
        for s in decks["d01161a"]["listing"]["sections"]
        if s["kind"] == "displacement" and s["subcase"] == 10
    ]
    assert occ == list(range(1, len(occ) + 1)) and len(occ) > 1
    assert decks["d01161a"]["card_counts"]["POPT"] == 1


def test_thermal_decks(decks):
    for deck_id in ("d01032a", "d01033a", "d01034a"):
        d = decks[deck_id]
        assert d["loads"]["TEMP"]["1"]["count"] == d["grids"]["count"]
        assert d["case_control"]["subcases"][0]["selectors"]["TEMPERATURE"]["type"] == "BOTH"
        assert {m["MID"] for m in d["materials"]["MATT1"]} <= {
            m["MID"] for m in d["materials"]["MAT1"]
        }
    scans = decks["d01034a"]["case_control"]["subcases"][0]["scan"]
    assert [s["value"] for s in scans] == ["8, SET 1", "+1500., -1500., SET 2", "10, SET 3"]
    assert all(c["scan_filtered"] for c in decks["d01034a"]["checks"]["listing_coverage"])
    scanned = [s["scan"] for s in decks["d01034a"]["listing"]["sections"] if s["scan"]]
    assert scanned == [
        {"field": "SHEAR-XY", "set": 1},
        {"field": "MAJOR", "set": 2},
        {"field": "MAX-SHR", "set": 3},
    ]
    assert not decks["d01033a"]["case_control"]["subcases"][0]["scan"]


def test_t01241a_kggx_matches_closed_form_rod_stiffness(decks):
    d = decks["t01241a"]
    (cov,) = d["checks"]["listing_coverage"]
    assert cov["listed"] == 0 and cov["missing_from_listing"] == d["grids"]["ids"]
    (kggx,) = d["listing"]["matrices"]
    assert kggx["rows"] == kggx["cols"] == 6 * d["grids"]["count"]
    parity = d["checks"]["kggx_parity"]
    assert parity["max_abs_error"] <= 1e-6 * 2.184e6
    assert parity["max_rel_error"] <= 1e-6


def test_listing_page_count_is_last_printed_page(decks):
    for d in decks.values():
        assert d["listing"]["pages"] >= max(s["pages"][1] for s in d["listing"]["sections"])


def test_manifest_hashes_match_sources(decks):
    import hashlib

    root = repo_root()
    for d in decks.values():
        assert hashlib.sha256((root / d["inp"]).read_bytes()).hexdigest() == d["sha256"]["inp"]
        assert (
            hashlib.sha256((root / d["listing_path"]).read_bytes()).hexdigest()
            == d["sha256"]["listing"]
        )


def test_build_manifest_is_deterministic():
    assert build_manifest() == build_manifest()
    assert build_deck("t01241a", repo_root())["elements"]["CROD"]["ids"] == [60, 61, 62]
