import numpy as np
import pytest

from nastran95py.bulk import BulkDataError, BulkModel, build_ecpt, build_ecpts, point_to_basic
from nastran95py.coords import CoordType, transform_to_basic
from nastran95py.elements import LAYOUTS
from nastran95py.reference.cards import read_deck


def bulk(*cards: str) -> str:
    return "ID T,T\nCEND\nBEGIN BULK\n" + "\n".join(cards) + "\nENDDATA\n"


def model(*cards: str) -> BulkModel:
    return BulkModel.from_text(bulk(*cards))


@pytest.mark.parametrize(
    ("name", "length", "theta_word"),
    [("SHEAR", 25, None), ("TRMEM", 21, 5), ("QDMEM", 26, 6), ("QDMEM1", 26, 6), ("QDMEM2", 26, 6)],
)
def test_membrane_layout_lengths(name, length, theta_word):
    layout = LAYOUTS[name]
    assert layout.length == length
    offsets, word = {}, 1
    for f in layout.header:
        offsets[f.name] = word
        word += f.count
    assert offsets.get("theta") == theta_word
    assert offsets["mid"] == (6 if theta_word is None else theta_word + 1)
    assert word + 4 * layout.ngrids == length


QUAD = (
    "GRID    1               0.      0.      0.",
    "GRID    2               2.      0.      0.",
    "GRID    3               2.      1.      0.",
    "GRID    4               0.      1.      0.",
    "MAT1    7       1.+7            .3      .1      6.-6    70.",
    "PQDMEM1 3       7       .25     .01",
    "CQDMEM1 9       3       1       2       3       4       30.",
)

QUAD_LARGE = (
    "GRID*   1                               0.              0.              *G1",
    "*G1     0.",
    "GRID*   2                               2.              0.",
    "*       0.",
    "GRID    3               2.      1.      0.",
    "GRID,4,,0.,1.,0.",
    "MAT1,7,1.+7,,.3,.1,6.-6,70.",
    "PQDMEM1*3               7               .25             .01",
    "CQDMEM1,9,3,1,2,3,4,30.",
)


def test_quad_ecpt_words():
    m = model(*QUAD)
    rec = build_ecpts(m)["QDMEM1"][0]
    w = rec.words()
    assert w[:9] == [9, 1, 7, 13, 19, 30.0, 7, 0.25, 0.01]
    assert w[9:13] == [0, 0.0, 0.0, 0.0]
    assert w[17:21] == [0, 2.0, 1.0, 0.0]
    assert w[25] == 0.0
    assert len(w) == 26
    back = LAYOUTS["QDMEM1"].parse(rec.words(raw=True), raw=True)
    assert back.fields["sil"] == rec.data.fields["sil"]
    assert back.fields["nsm"] == pytest.approx(0.01, rel=1e-7)


def test_large_and_free_field_match_small_field():
    small = build_ecpts(model(*QUAD))["QDMEM1"][0].words()
    mixed = build_ecpts(model(*QUAD_LARGE))["QDMEM1"][0].words()
    assert mixed == small


def test_rod_and_conrod_ecpt():
    m = model(
        "GRID    10              0.      0.      0.",
        "GRID    20              3.      4.      0.",
        "MAT1    1       30.+6   11.5+6",
        "PROD    5       1       2.      .5      .1      .2",
        "CROD    1       5       10      20      2       5       20      10",
        "CONROD  3       10      20      1       1.5     .25",
    )
    recs = build_ecpts(m)["ROD"]
    assert [r.card for r in recs] == ["CROD", "CROD", "CONROD"]
    assert recs[0].words()[:8] == [1, 1, 7, 1, 2.0, 0.5, 0.1, 0.2]
    assert recs[1].words()[:3] == [2, 7, 1]
    assert recs[1].data.fields["mid"] == 1
    assert recs[2].words()[:8] == [3, 1, 7, 1, 1.5, 0.25, 0.0, 0.0]
    assert len(recs[2].words()) == 17
    np.testing.assert_allclose(recs[0].data.grids[1].xyz, [3.0, 4.0, 0.0])
    assert m.materials.mat1(1).nu == pytest.approx(30e6 / (2 * 11.5e6) - 1)


@pytest.mark.parametrize(
    ("e", "g", "nu", "expected"),
    [
        ("30.+6", "", ".3", (30e6, 30e6 / 2.6, 0.3)),
        ("30.+6", "12.+6", "", (30e6, 12e6, 0.25)),
        ("", "12.+6", ".25", (30e6, 12e6, 0.25)),
    ],
)
def test_mat1_completion(e, g, nu, expected):
    m = model(f"MAT1    1       {e:<8}{g:<8}{nu:<8}")
    got = m.materials.mat1(1)
    np.testing.assert_allclose((got.e, got.g, got.nu), expected)


def test_shear_and_trmem():
    m = model(
        *QUAD[:5],
        "PSHEAR  2       7       .1      .5",
        "PTRMEM  4       7       .2",
        "CSHEAR  1       2       1       2       3       4",
        "CTRMEM  2       4       1       2       3       15.",
    )
    e = build_ecpts(m)
    s = e["SHEAR"][0].words()
    assert s[:8] == [1, 1, 7, 13, 19, 7, 0.1, 0.5]
    assert len(s) == 25
    t = e["TRMEM"][0].words()
    assert t[:8] == [2, 1, 7, 13, 15.0, 7, 0.2, 0.0]
    assert len(t) == 21


def test_grdset_defaults_and_overrides():
    m = model(
        "CORD2R  3               0.      0.      0.      0.      0.      1.",
        "+       1.      0.      0.",
        "GRDSET          3                               3       456",
        "GRID    1               1.      2.      3.",
        "GRID    2       0       1.      2.      3.      0       12",
    )
    assert (m.grids[1].cp, m.grids[1].cd, m.grids[1].ps) == (3, 3, "456")
    assert (m.grids[2].cp, m.grids[2].cd, m.grids[2].ps) == (0, 0, "12")


def test_cord2_cylindrical_spherical_and_chain():
    m = model(
        "CORD2R  1               1.      2.      3.      1.      2.      4.",
        "+       2.      2.      3.",
        "CORD2C  2       1       0.      0.      0.      0.      0.      1.",
        "+       0.      1.      0.",
        "CORD2S  3               0.      0.      0.      0.      0.      1.",
        "+       1.      0.      0.",
        "GRID    1       2       2.      90.     5.",
        "GRID    2       3       2.      90.     90.",
        "GRID    10              0.      0.      0.",
        "GRID    11              0.      0.      1.",
        "GRID    12      2       1.      0.      0.",
        "CORD1R  4       10      11      12",
        "GRID    13      4       1.      0.      0.",
    )
    # CORD2C 2 is rotated 90 deg about z inside CORD2R 1 (origin 1,2,3).
    np.testing.assert_allclose(m.grids[1].basic, [1.0 - 2.0, 2.0, 3.0 + 5.0], atol=1e-12)
    np.testing.assert_allclose(m.grids[2].basic, [0.0, 2.0, 0.0], atol=1e-12)
    g12 = m.grids[12].basic * [1.0, 1.0, 0.0]
    x_axis = g12 / np.linalg.norm(g12)
    np.testing.assert_allclose(m.grids[13].basic, x_axis, atol=1e-12)
    assert set(m.cstm) == {1, 2, 3, 4}


def test_point_to_basic_matches_transform_of_unit_vectors():
    m = model(
        "CORD2S  3               1.      1.      1.      1.      2.      2.",
        "+       3.      1.      0.",
    )
    cs = m.cstm[3]
    assert cs.ctype == CoordType.SPHERICAL
    p = point_to_basic(cs, [2.0, 40.0, 70.0])
    # radial direction at the point maps to (p - origin)/r
    radial = transform_to_basic(m.cstm, 3, p) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(radial, (p - cs.origin) / 2.0, atol=1e-12)


def test_loads_spcs_and_temperatures():
    m = model(
        *QUAD,
        "FORCE   1       2               10.     0.      -1.     0.",
        "MOMENT  1       3       0       2.      0.      0.      1.",
        "SPC     5       1       123     0.      2       3       .5",
        "SPC1    5       456     1       THRU    4",
        "TEMP    8       1       100.    2       200.    3       300.",
        "TEMPD   8       50.",
    )
    assert [(p.card, p.g, p.scale, p.n) for p in m.loads] == [
        ("FORCE", 2, 10.0, (0.0, -1.0, 0.0)),
        ("MOMENT", 3, 2.0, (0.0, 0.0, 1.0)),
    ]
    assert [(s.card, s.g, s.c, s.d) for s in m.spcs[:2]] == [
        ("SPC", 1, "123", 0.0),
        ("SPC", 2, "3", 0.5),
    ]
    assert [s.g for s in m.spcs if s.card == "SPC1"] == [1, 2, 3, 4]
    rec = build_ecpt(m, m.connections[0], temp_set=8)
    assert rec.data.temperature == pytest.approx((100 + 200 + 300 + 50) / 4)
    assert rec.words()[-1] == pytest.approx(162.5)


def test_spoint_and_seqgp_sil():
    m = model(
        "GRID    1               0.      0.      0.",
        "GRID    2               1.      0.      0.",
        "SPOINT  5",
        "SEQGP   2       0.5",
    )
    assert m.sil == {2: 1, 1: 7, 5: 13}
    assert m.bgpdt(5)[0] == -1


@pytest.mark.parametrize(
    ("cards", "match"),
    [
        ((*QUAD[:5], "CQDMEM1 9       3       1       2       3       4"), "PQDMEM1 3"),
        ((*QUAD[:6], "CQDMEM1 9       3       1       2       3       99"), "undefined grid"),
        (("GRID    1       9       0.      0.      0.",), "coordinate systems"),
        (("GRID    1               0.", "GRID    1               1."), "duplicate"),
        (
            (*QUAD[:6], "CQDMEM1 9       3       1       2       3       3"),
            "repeated grid",
        ),
    ],
)
def test_errors(cards, match):
    with pytest.raises(BulkDataError, match=match):
        build_ecpts(model(*cards))


def test_missing_temperature_set():
    m = model(*QUAD)
    with pytest.raises(BulkDataError, match="temperature set 4"):
        build_ecpts(m, temp_set=4)


def test_unsupported_cards_are_counted():
    deck = read_deck(bulk("PARAM   GRDPNT  0", "GRID    1               0.      0.      0."))
    assert BulkModel.from_deck(deck).ignored == {"PARAM": 1}
