import numpy as np
import pytest

from nastran95py.coords import (
    CoordSystem,
    CoordType,
    Cstm,
    CstmError,
    basic_to_global,
    global_to_basic,
    transform_to_basic,
)

ROT_Z45 = np.array([[1, -1, 0], [1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)


@pytest.fixture
def cstm():
    return Cstm(
        [
            CoordSystem(1, CoordType.RECTANGULAR, [1, 2, 3], ROT_Z45),
            CoordSystem(2, CoordType.CYLINDRICAL, [0, 0, 0], np.eye(3)),
            CoordSystem(3, CoordType.SPHERICAL, [0, 0, 0], np.eye(3)),
        ]
    )


def test_basic_is_identity(cstm):
    np.testing.assert_array_equal(transform_to_basic(cstm, 0, [5, 6, 7]), np.eye(3))


def test_rectangular_returns_t(cstm):
    np.testing.assert_allclose(transform_to_basic(cstm, 1, [9, 9, 9]), ROT_Z45)


def test_cylindrical_radial_axis(cstm):
    ta = transform_to_basic(cstm, 2, [0.0, 2.0, 5.0])
    np.testing.assert_allclose(ta @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(ta @ [0, 1, 0], [-1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(ta @ [0, 0, 1], [0, 0, 1], atol=1e-15)


def test_spherical_axes(cstm):
    ta = transform_to_basic(cstm, 3, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ta @ [1, 0, 0], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(ta @ [0, 1, 0], [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(ta @ [0, 0, 1], [0, 1, 0], atol=1e-15)


@pytest.mark.parametrize("cid", [2, 3])
def test_polar_axis_falls_back_to_rectangular(cstm, cid):
    np.testing.assert_array_equal(transform_to_basic(cstm, cid, [0, 0, 4]), np.eye(3))


@pytest.mark.parametrize("cid", [1, 2, 3])
def test_orthonormal_and_vector_round_trip(cstm, cid):
    p = [0.3, -1.7, 2.2]
    ta = transform_to_basic(cstm, cid, p)
    np.testing.assert_allclose(ta.T @ ta, np.eye(3), atol=1e-14)
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(global_to_basic(cstm, cid, p, basic_to_global(cstm, cid, p, v)), v)


def test_unknown_cid(cstm):
    with pytest.raises(CstmError):
        transform_to_basic(cstm, 42, [0, 0, 0])


def test_from_points():
    cs = CoordSystem.from_points(4, 1, [1, 1, 1], [1, 1, 2], [2, 1, 1])
    np.testing.assert_allclose(cs.t, np.eye(3))
    cs = CoordSystem.from_points(5, 2, [0, 0, 0], [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(cs.t, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-15)
    with pytest.raises(ValueError):
        CoordSystem.from_points(6, 1, [0, 0, 0], [0, 0, 1], [0, 0, 2])


def test_cstm_word_round_trip(cstm):
    words = cstm.to_words()
    assert len(words) == 42
    again = Cstm.from_words(words)
    assert list(again) == [1, 2, 3]
    np.testing.assert_allclose(again[1].t, ROT_Z45)
    with pytest.raises(ValueError):
        Cstm([CoordSystem(0, 1)])
    with pytest.raises(ValueError):
        Cstm.from_words(words[:13])
