import numpy as np
import pytest

from nastran95py.reference.golden import (
    ATOL_SCALE,
    MARGIN_RTOL,
    MEMBRANE_TYPES,
    RTOL,
    build_all,
    compare,
    golden_path,
    listing_roundtrip,
    load_golden,
    main,
    membrane_consistency,
    tables,
    tolerances,
)
from nastran95py.reference.inventory import repo_root
from nastran95py.reference.listing import index_listing
from nastran95py.reference.results import ListingParseError, parse_tables

DECK_IDS = [
    "d01011a",
    "d01012a",
    "d01013a",
    "d01014a",
    "d01032a",
    "d01033a",
    "d01034a",
    "d01161a",
    "t01241a",
]


@pytest.fixture(scope="module")
def d01011a():
    return tables(load_golden("d01011a"))


def test_golden_file_for_every_inventory_deck(manifest):
    assert sorted(d["id"] for d in manifest["decks"]) == DECK_IDS
    for deck in manifest["decks"]:
        golden = load_golden(deck["id"])
        assert golden["listing_sha256"] == deck["sha256"]["listing"]
        assert golden_path(deck["id"]).exists()


def test_golden_files_are_current():
    assert main(["--check"]) == 0


# Hand-checked against demoout/d01011a.out (page and line numbers in comments).
def test_d01011a_displacements(d01011a):
    disp = d01011a["displacement/grid/sc1/occ1"]
    assert disp.columns == ("t1", "t2", "t3", "r1", "r2", "r3")
    assert len(disp.ids) == 24 and disp.ids[0] == 11 and disp.ids[-1] == 93
    # page 20, line 651: grid 14
    np.testing.assert_array_equal(
        disp.by_id()[14], [4.135543e-03, -5.312650e-04, -8.153134e-02, 0.0, 0.0, 0.0]
    )
    assert disp.by_id()[16][2] == -4.237940e-01
    assert disp.meta["point_types"] == ["G"] * 24


def test_d01011a_spc_forces(d01011a):
    spc = d01011a["spcforce/grid/sc1/occ1"]
    assert len(spc.ids) == 34
    assert spc.by_id()[1][0] == 3.456080e-10  # round-off noise is kept as printed
    assert spc.by_id()[31][0] == -5.060130e03
    np.testing.assert_array_equal(spc.by_id()[6][:3], [3.345866e03, -2.564445e03, 0.0])
    assert spc.by_id()[93][2] == -1.840265e02


def test_d01011a_quad_membrane_stresses(d01011a):
    q = d01011a["stress/CQDMEM/sc1/occ1"]
    assert list(q.ids) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16]
    # page 22: element 4
    np.testing.assert_array_equal(
        q.by_id()[4],
        [1.532354e03, 3.064197e02, -9.642474e02, -28.7780, 2.061972e03, -2.231987e02, 1.142586e03],
    )


def test_d01011a_shear_panel_stresses_two_up(d01011a):
    s = d01011a["stress/CSHEAR/sc1/occ1"]
    assert s.columns == ("max_shear", "avg_shear", "margin")
    assert list(s.ids) == [18, 19, 20, 21, 22, 28, 29, 30, 31, 35, 36, 41, 42, 43, 44, 50]
    row = s.by_id()[30]  # right half of the 4th row on page 23
    assert row[0] == 2.752647e02 and row[1] == -2.133537e02 and np.isnan(row[2])
    assert s.by_id()[50][1] == 1.831746e03


def test_d01011a_triangular_membrane_stresses(d01011a):
    t = d01011a["stress/CTRMEM/sc1/occ1"]
    assert list(t.ids) == [10, 14, 17]
    assert t.column("angle")[2] == -26.5787
    assert t.by_id()[10][0] == 9.285977e02 and t.by_id()[10][6] == 6.329477e02


def test_sort2_regrouped_by_subcase():
    g = tables(load_golden("d01012a"))
    d1, d2 = g["displacement/grid/sc1/occ1"], g["displacement/grid/sc2/occ1"]
    assert d1.meta["sort"] == "SORT2" and list(d1.ids) == list(d2.ids)
    np.testing.assert_array_equal(d1.by_id()[12][:3], [6.614148e-04, 6.103728e-04, 3.598766e-02])
    np.testing.assert_array_equal(d2.by_id()[12][:3], [7.075042e-04, 5.407654e-04, 3.301679e-02])
    # CSHEAR SORT2 packs subcases 1 and 2 on one row
    assert g["stress/CSHEAR/sc1/occ1"].by_id()[19][0] == 8.534302e00
    assert g["stress/CSHEAR/sc2/occ1"].by_id()[19][1] == -1.882340e02


def test_element_forces():
    g = tables(load_golden("d01013a"))
    conrod = g["force/CONROD/sc1/occ1"]
    assert len(conrod.ids) == 36 and conrod.by_id()[105][0] == 1.260784e02
    assert g["force/CROD/sc1/occ1"].by_id()[65][0] == -2.030000e02
    shear = g["force/CSHEAR/sc1/occ1"]
    assert len(shear.columns) == 16 and len(shear.ids) == 36
    row = shear.by_id()[20]
    assert row[0] == -2.88714e01 and row[3] == 2.79911e01 and row[9] == -7.04179e01


def test_rod_stress_margins_and_fsd_occurrences():
    g = tables(load_golden("d01161a"))
    for occ in (1, 2, 3):
        assert f"stress/CTRMEM/sc10/occ{occ}" in g
        assert f"displacement/grid/sc12/occ{occ}" in g
    rod = g["stress/CROD/sc10/occ1"]
    np.testing.assert_array_equal(rod.by_id()[101][:3], [-6.387499e03, 2.9, 0.0])
    assert np.isnan(rod.by_id()[101][3])
    assert rod.by_id()[105][0] == 2.252632e04


def test_matrix_prints_kept():
    (kggx,) = load_golden("t01241a")["matrices"]
    assert kggx["name"] == "KGGX" and len(kggx["entries"]) == 10
    assert load_golden("t01241a")["tables"] == []


@pytest.mark.parametrize("deck_id", DECK_IDS)
def test_tables_cover_index_ids(deck_id, decks):
    """Every id the listing index saw in a SORT1 section is in the golden table."""
    golden = load_golden(deck_id)
    index = decks[deck_id]["listing"]
    for sec in index["sections"]:
        if sec["kind"] not in ("displacement", "spcforce", "oload", "force", "stress"):
            continue
        same = [
            t
            for t in golden["tables"]
            if t["listing_lines"] == sec["lines"] and t["occurrence"] == sec["occurrence"]
        ]
        assert same, sec["title"]
        if sec["sort"] == "SORT1":
            (t,) = same
            # the index keeps the "0" carriage-control column of CSHEAR force rows
            assert [r[0] for r in t["rows"]] == [i for i in sec["ids"] if i != 0]
        else:
            assert sorted(t["subcase"] for t in same) == sorted(sec["subcases_in_rows"])
            for t in same:
                assert [r[0] for r in t["rows"]] == sec["ids"]


@pytest.mark.parametrize("deck_id", DECK_IDS)
def test_listing_roundtrip_exact(deck_id):
    c = listing_roundtrip(load_golden(deck_id))
    assert c.passed and c.max_abs == 0.0


@pytest.mark.parametrize("deck_id", DECK_IDS)
def test_membrane_mohr_circle_within_tolerance(deck_id):
    for t in tables(load_golden(deck_id)).values():
        if t.meta["kind"] == "stress" and t.meta["element_type"] in MEMBRANE_TYPES:
            assert membrane_consistency(t).passed, t.key


def test_scan_rows_match_full_stress_table():
    g = tables(load_golden("d01034a"))
    full = g["stress/CQDMEM2/sc1/occ1"].by_id()
    scans = [t for k, t in g.items() if "/scan-" in k]
    assert len(scans) == 3
    for t in scans:
        for eid, row in t.by_id().items():
            np.testing.assert_array_equal(row, full[eid])


def test_tolerances_scale_with_column_max(d01011a):
    spc = d01011a["spcforce/grid/sc1/occ1"]
    rtol, atol = tolerances(spc)
    assert (rtol == RTOL).all()
    assert atol[0] == pytest.approx(ATOL_SCALE * 5.060130e03)
    # all-zero rotation columns fall back to the table max
    assert atol[3] == pytest.approx(ATOL_SCALE * 5.060130e03)
    rod = tables(load_golden("d01161a"))["stress/CROD/sc10/occ1"]
    assert tolerances(rod)[0][rod.columns.index("axial_margin")] == MARGIN_RTOL


def test_compare_pass_fail_and_blank_fields(d01011a):
    s = d01011a["stress/CSHEAR/sc1/occ1"]
    exact = np.nan_to_num(s.values, nan=123.0)  # blank margins are ignored
    assert compare(s, exact).passed
    near = s.values * (1 + 0.5 * RTOL)
    c = compare(s, np.nan_to_num(near))
    assert c.passed and c.max_rel == pytest.approx(0.5 * RTOL, rel=1e-6)
    off = np.nan_to_num(s.values.copy())
    off[7, 0] *= 1 + 10 * RTOL
    c = compare(s, off)
    assert not c.passed and c.failures == 1
    with pytest.raises(ValueError):
        compare(s, s.values[:-1])


def test_parse_two_up_row_with_blank_right_half():
    header = (
        "         ID.        STRESS       MARGIN        STRESS      MARGIN         ID.  "
        "      STRESS       MARGIN        STRESS      MARGIN"
    )
    text = "\n".join(
        [
            "1     X" + " " * 60 + "/ NASTRAN / PAGE    1",
            "      SUBTITLE",
            "0     LABEL",
            " ",
            "                                     S T R E S S E S   I N   R O D   E L E M E N T S"
            "      ( C R O D )",
            "       ELEMENT       AXIAL       SAFETY      TORSIONAL     SAFETY       ELEMENT",
            header,
            "           105    2.252632E+04   1.1E-01    0.0",
        ]
    )
    (t,) = parse_tables(text, default_subcase=7)
    assert t.subcase == 7 and t.ids == [105]
    assert t.rows == [[2.252632e04, 0.11, 0.0, None]]


def test_unknown_layout_raises():
    text = "\n".join(
        [
            "1     X" + " " * 60 + "/ NASTRAN / PAGE    1",
            "      SUBTITLE",
            "0     LABEL",
            "                      F O R C E S   I N   B A R   E L E M E N T S   ( C B A R )",
            "  ID.   A",
            "    1   1.0",
        ]
    )
    with pytest.raises(ListingParseError):
        parse_tables(text)


def test_build_all_matches_parser_on_listing():
    root = repo_root()
    text = (root / "demoout" / "d01011a.out").read_text(errors="replace")
    assert len(parse_tables(text, 1, index_listing(text))) == 5
    assert set(build_all()) == set(DECK_IDS)
