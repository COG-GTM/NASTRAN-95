"""CORD1/CORD2 to CSTM construction (GP1), location conversion and vectorised transforms."""

import numpy as np
import pytest

from nastran95py.coords import (
    CoordSystem,
    CoordType,
    Cord1,
    Cord2,
    CordError,
    Cstm,
    Grid,
    basglb,
    build_cstm,
    cords_from_cards,
    gbtran,
    glbbas,
    grids_from_cards,
    grids_to_basic,
    location_to_basic,
    transform_to_basic,
    transforms_to_basic,
)
from nastran95py.reference.cards import read_deck
from nastran95py.reference.inventory import repo_root

R, C, S = CoordType.RECTANGULAR, CoordType.CYLINDRICAL, CoordType.SPHERICAL
SQ2 = np.sqrt(0.5)


def test_cord2_basic_identity():
    cstm = build_cstm([Cord2(1, R, (0, 0, 0), (0, 0, 1), (1, 0, 0))])
    np.testing.assert_array_equal(cstm[1].origin, 0.0)
    np.testing.assert_allclose(cstm[1].t, np.eye(3))
    np.testing.assert_allclose(transform_to_basic(cstm, 1, [3, 4, 5]), np.eye(3))


def test_cord2_rotated_90_about_z():
    # C on basic +y puts local x on basic +y: a +90 degree rotation about z.
    cstm = build_cstm([Cord2(2, R, (1, 2, 3), (1, 2, 4), (1, 5, 3))])
    rz90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(cstm[2].t, rz90, atol=1e-15)
    np.testing.assert_allclose(cstm[2].origin, [1, 2, 3])
    np.testing.assert_allclose(location_to_basic(cstm[2], [1, 0, 0]), [1, 3, 3])


def test_cord2_rotated_90_about_x():
    # B on basic -y: local z is basic -y, local x stays basic x.
    cstm = build_cstm([Cord2(3, R, (0, 0, 0), (0, -1, 0), (1, 0, 0))])
    np.testing.assert_allclose(cstm[3].t, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-15)


def test_cord2_c_off_plane_is_projected():
    # Only the x-z plane through C matters, not C's distance or z component.
    a = build_cstm([Cord2(4, R, (0, 0, 0), (0, 0, 2), (1, 1, 7))])[4].t
    b = build_cstm([Cord2(4, R, (0, 0, 0), (0, 0, 1), (5, 5, -3))])[4].t
    np.testing.assert_allclose(a, b, atol=1e-15)
    np.testing.assert_allclose(a[:, 0], [SQ2, SQ2, 0], atol=1e-15)


def test_cylindrical_axes_at_theta_0_and_90():
    cstm = build_cstm([Cord2(5, C, (0, 0, 0), (0, 0, 1), (1, 0, 0))])
    cs = cstm[5]
    p0 = location_to_basic(cs, [2.0, 0.0, 1.0])
    p90 = location_to_basic(cs, [2.0, 90.0, 1.0])
    np.testing.assert_allclose(p0, [2, 0, 1], atol=1e-15)
    np.testing.assert_allclose(p90, [0, 2, 1], atol=1e-15)
    # theta = 0: (r, theta, z) directions are basic (x, y, z).
    np.testing.assert_allclose(transform_to_basic(cstm, 5, p0), np.eye(3), atol=1e-15)
    # theta = 90: r is basic y, theta is basic -x.
    np.testing.assert_allclose(
        transform_to_basic(cstm, 5, p90), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15
    )


def test_spherical_location_and_axes():
    cstm = build_cstm([Cord2(6, S, (0, 0, 0), (0, 0, 1), (1, 0, 0))])
    cs = cstm[6]
    np.testing.assert_allclose(location_to_basic(cs, [2, 0, 0]), [0, 0, 2], atol=1e-15)
    np.testing.assert_allclose(location_to_basic(cs, [2, 90, 0]), [2, 0, 0], atol=1e-15)
    np.testing.assert_allclose(location_to_basic(cs, [2, 90, 90]), [0, 2, 0], atol=1e-15)
    # On the equator at phi = 90: rho = +y, theta = -z, phi = -x.
    ta = transform_to_basic(cstm, 6, [0, 2, 0])
    np.testing.assert_allclose(ta, [[0, 0, -1], [1, 0, 0], [0, -1, 0]], atol=1e-15)


def test_cord2_with_cylindrical_rid():
    # RID 10 is cylindrical about basic z; CORD2 points given as (r, theta, z).
    cords = [
        Cord2(11, R, (1, 90, 0), (1, 90, 1), (2, 90, 0), rid=10),
        Cord2(10, C, (0, 0, 0), (0, 0, 1), (1, 0, 0)),
    ]
    cstm = build_cstm(cords)
    assert list(cstm) == [11, 10]
    np.testing.assert_allclose(cstm[11].origin, [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(cstm[11].t, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)


def test_cord2_chain_through_rotated_rid():
    base = Cord2(20, R, (1, 0, 0), (1, 0, 1), (1, 1, 0))  # rotated 90 about z, shifted
    child = Cord2(21, R, (0, 0, 0), (0, 0, 1), (1, 0, 0), rid=20)
    cstm = build_cstm([child, base])
    np.testing.assert_allclose(cstm[21].t, cstm[20].t, atol=1e-15)
    np.testing.assert_allclose(cstm[21].origin, [1, 0, 0])


def test_cord1_from_basic_grids():
    grids = [Grid(1, 0, (1, 1, 1)), Grid(2, 0, (1, 1, 2)), Grid(3, 0, (1, 2, 1))]
    cstm = build_cstm([Cord1(7, C, 1, 2, 3)], grids)
    assert cstm[7].ctype == C
    np.testing.assert_allclose(cstm[7].origin, [1, 1, 1])
    np.testing.assert_allclose(cstm[7].t, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)


def test_cord1_grids_in_solved_system_and_cord2_on_cord1():
    # Grids 2 and 3 are defined in system 30, solved first; system 32 refers to 31.
    cords = [
        Cord2(32, S, (0, 0, 0), (0, 0, 1), (1, 0, 0), rid=31),
        Cord1(31, R, 1, 2, 3),
        Cord2(30, C, (0, 0, 5), (0, 0, 6), (1, 0, 5)),
    ]
    grids = {
        1: Grid(1, 0, (0, 0, 5)),
        2: Grid(2, 30, (0, 0, 1)),
        3: Grid(3, 30, (1, 90, 0)),
    }
    cstm = build_cstm(cords, grids)
    np.testing.assert_allclose(cstm[31].origin, [0, 0, 5])
    np.testing.assert_allclose(cstm[31].t, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_allclose(cstm[32].t, cstm[31].t, atol=1e-15)
    basic = grids_to_basic(cstm, grids.values())
    np.testing.assert_allclose(basic[3], [0, 1, 5], atol=1e-15)


@pytest.mark.parametrize(
    ("cords", "grids", "ierr"),
    [
        ([Cord1(1, R, 1, 2, 9)], [Grid(1, 0, (0, 0, 0)), Grid(2, 0, (0, 0, 1))], 3),
        ([Cord2(1, R, (0, 0, 0), (0, 0, 1), (1, 0, 0), rid=5)], [], 4),
        (
            [
                Cord2(1, R, (0, 0, 0), (0, 0, 1), (1, 0, 0), rid=2),
                Cord2(2, R, (0, 0, 0), (0, 0, 1), (1, 0, 0), rid=1),
            ],
            [],
            5,
        ),
        (
            [Cord1(1, R, 1, 2, 3)],
            [Grid(1, 0, (0, 0, 0)), Grid(2, 0, (0, 0, 1)), Grid(3, 8, (1, 0, 0))],
            6,
        ),
        (
            [Cord1(1, R, 1, 2, 3)],
            [Grid(1, 0, (0, 0, 0)), Grid(2, 0, (0, 0, 1)), Grid(3, 1, (1, 0, 0))],
            5,
        ),
    ],
)
def test_gp1_errors(cords, grids, ierr):
    with pytest.raises(CordError) as exc:
        build_cstm(cords, grids)
    assert exc.value.ierr == ierr


def test_card_validation():
    with pytest.raises(ValueError):
        Cord1(1, R, 1, 1, 2)
    with pytest.raises(ValueError):
        Cord2(1, R, (0, 0, 0), (0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError):
        Cord2(0, R, (0, 0, 0), (0, 0, 1), (1, 0, 0))
    with pytest.raises(ValueError):
        build_cstm([Cord2(1, R, (0, 0, 0), (0, 0, 1), (0, 0, 2))])  # collinear
    with pytest.raises(ValueError):
        build_cstm([Cord2(1, R, (0, 0, 0), (0, 0, 1), (1, 0, 0))] * 2)


DECK = """ID    CORD,TEST
APP   DISPLACEMENT
SOL   1,0
CEND
BEGIN BULK
CORD1C  1       1       2       3       2       4       5       6
CORD2S  3       1       .0      .0      .0      .0      .0      1.      +C3
+C3     1.      .0      .0
CORD2R,4,,1.,2.,3.,1.,2.,4.,+C4
+C4,2.,2.,3.
GRID    1               0.0     0.0     0.0
GRID    2               0.0     0.0     1.0
GRID    3               1.0     0.0     0.0
GRID    4       4       0.0     0.0     0.0
GRID    5       4       0.0     0.0     1.0
GRID    6       4       0.0     1.0     0.0
ENDDATA
"""


def test_cords_and_grids_from_cards():
    cards = read_deck(DECK).bulk
    cords = cords_from_cards(cards)
    assert [(type(c).__name__, c.cid, c.ctype) for c in cords] == [
        ("Cord1", 1, C),
        ("Cord1", 2, C),
        ("Cord2", 3, S),
        ("Cord2", 4, R),
    ]
    assert cords[2].rid == 1 and cords[3].rid == 0
    grids = grids_from_cards(cards)
    assert [(g.gid, g.cp) for g in grids] == [(1, 0), (2, 0), (3, 0), (4, 4), (5, 4), (6, 4)]
    cstm = build_cstm(cords, grids)
    np.testing.assert_allclose(cstm[2].origin, [1, 2, 3])
    np.testing.assert_allclose(cstm[2].t, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)


def _decks_with_cords():
    root = repo_root()
    out = []
    for path in sorted((root / "inp").glob("*.inp")):
        text = path.read_text(errors="replace")
        if "\nCORD" not in text:
            continue
        marks = ()
        try:
            cords_from_cards(read_deck(text).bulk)
        except ValueError:
            # Blank-separated free field and other forms beyond the RF1-scoped deck reader.
            marks = pytest.mark.skip(reason="deck syntax outside nastran95py.reference.cards")
        out.append(pytest.param(text, id=path.stem, marks=marks))
    return out


@pytest.mark.parametrize("text", _decks_with_cords())
def test_demo_deck_cords(text):
    """Every CORD card in ../inp builds a right-handed orthonormal CSTM satisfying its points."""
    cards = read_deck(text).bulk
    cords = cords_from_cards(cards)
    assert cords
    grids = grids_from_cards(cards)
    cstm = build_cstm(cords, grids)
    for cord in cords:
        cs = cstm[cord.cid]
        np.testing.assert_allclose(cs.t.T @ cs.t, np.eye(3), atol=1e-13)
        assert np.linalg.det(cs.t) == pytest.approx(1.0)
        if isinstance(cord, Cord2):
            ref = cstm[cord.rid] if cord.rid else None
            a, b, c = location_to_basic(ref, [cord.a, cord.b, cord.c])
            local_b = cs.t.T @ (b - cs.origin)
            local_c = cs.t.T @ (c - cs.origin)
            scale = max(1.0, np.abs(local_b).max(), np.abs(local_c).max())
            np.testing.assert_allclose(cs.origin, a, atol=1e-12 * scale)
            np.testing.assert_allclose(local_b[:2], 0.0, atol=1e-12 * scale)
            assert local_b[2] > 0
            assert abs(local_c[1]) < 1e-12 * scale and local_c[0] > 0
    located = grids_to_basic(cstm, [g for g in grids if g.cp == 0 or g.cp in cstm])
    assert all(np.all(np.isfinite(v)) for v in located.values())


@pytest.fixture
def mixed_cstm():
    return build_cstm(
        [
            Cord2(1, R, (1, 2, 3), (1, 2, 4), (2, 3, 3)),
            Cord2(2, C, (0, 0, 0), (1, 0, 0), (0, 1, 0)),
            Cord2(3, S, (1, -1, 0), (1, -1, 2), (2, -1, 0)),
        ]
    )


def test_vectorised_matches_scalar(mixed_cstm):
    rng = np.random.default_rng(6)
    pts = rng.uniform(-4, 4, (40, 3))
    cids = rng.integers(0, 4, 40)
    pts[0] = [7, 0, 0]  # on the polar axis of system 2
    cids[0] = 2
    ta = transforms_to_basic(mixed_cstm, cids, pts)
    assert ta.shape == (40, 3, 3)
    for k in range(40):
        np.testing.assert_allclose(ta[k], transform_to_basic(mixed_cstm, cids[k], pts[k]))
    np.testing.assert_allclose(ta[0], mixed_cstm[2].t)
    np.testing.assert_array_equal(gbtran(mixed_cstm, cids, pts), ta)
    v = rng.normal(size=(40, 3))
    g = basglb(mixed_cstm, cids, pts, v)
    np.testing.assert_allclose(np.einsum("nji,nj->ni", ta, v), g)
    np.testing.assert_allclose(glbbas(mixed_cstm, cids, pts, g), v, atol=1e-13)
    assert basglb(mixed_cstm, 2, pts[1], v[1]).shape == (3,)


def test_transforms_shape_errors(mixed_cstm):
    with pytest.raises(ValueError):
        transforms_to_basic(mixed_cstm, 1, np.zeros((2, 2)))
    with pytest.raises(KeyError):
        transforms_to_basic(mixed_cstm, [1, 9], np.zeros((2, 3)))


def test_cstm_round_trip_of_built_systems(mixed_cstm):
    again = Cstm.from_words(mixed_cstm.to_words())
    for cid in mixed_cstm:
        assert isinstance(again[cid], CoordSystem)
        np.testing.assert_allclose(again[cid].t, mixed_cstm[cid].t)
