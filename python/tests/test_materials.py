import numpy as np
import pytest

from nastran95py.materials import (
    Mat1,
    Mat2,
    MaterialError,
    MaterialNotFoundError,
    MaterialTable,
    rotation_matrix,
)

STEEL = Mat1(1, e=1.0e7, g=3.8e6, nu=0.3, rho=0.1, alpha=1.2e-5, tref=70.0, ge=0.02)
ORTHO = Mat2(2, 1.0e7, 3.0e6, 1.0e5, 8.0e6, 2.0e5, 2.0e6, rho=0.2, a1=1e-5, a2=2e-5, a12=3e-6)


@pytest.fixture
def table():
    return MaterialTable([STEEL], [ORTHO])


def test_inflag1_passthrough(table):
    out = table.mat(1, 1)
    got = (out.e, out.g, out.nu, out.rho, out.alpha, out.tref)
    assert got == (1e7, 3.8e6, 0.3, 0.1, 1.2e-5, 70)


def test_inflag1_rejects_mat2(table):
    with pytest.raises(MaterialNotFoundError):
        table.mat1(2)


def test_mat1_to_plane_stress_g(table):
    out = table.mat2(1)
    g11 = 1.0e7 / (1 - 0.09)
    np.testing.assert_allclose(out.g, [[g11, 0.3 * g11, 0], [0.3 * g11, g11, 0], [0, 0, 3.8e6]])
    np.testing.assert_allclose(out.alpha, [1.2e-5, 1.2e-5, 0])
    assert out.source == "MAT1"
    assert len(out.to_words()) == 15


def test_mat1_ignores_angle(table):
    np.testing.assert_array_equal(table.mat2(1, 0.6, 0.8).g, table.mat2(1).g)


def test_mat2_zero_angle_unchanged(table):
    out = table.mat2(2, 0.0, 1.0)
    np.testing.assert_array_equal(out.g, ORTHO.g)
    np.testing.assert_array_equal(out.alpha, [1e-5, 2e-5, 3e-6])


def test_mat2_rotation_90_swaps_axes(table):
    out = table.mat2(2, 1.0, 0.0)
    assert out.g[0, 0] == pytest.approx(8.0e6)
    assert out.g[1, 1] == pytest.approx(1.0e7)
    assert out.g[0, 1] == pytest.approx(3.0e6)
    assert out.g[2, 2] == pytest.approx(2.0e6)
    np.testing.assert_allclose(out.alpha, [2e-5, 1e-5, -3e-6], atol=1e-20)


def test_mat2_rotation_is_symmetric_and_round_trips(table):
    th = np.deg2rad(37.0)
    rotated = table.mat2(2, np.sin(th), np.cos(th))
    np.testing.assert_allclose(rotated.g, rotated.g.T)
    back = MaterialTable(mat2=[Mat2(9, *rotated.g[np.triu_indices(3)])]).mat2(
        9, np.sin(-th), np.cos(-th)
    )
    np.testing.assert_allclose(back.g, ORTHO.g, rtol=1e-12, atol=1e-6)


def test_rotation_matrix_identity():
    np.testing.assert_array_equal(rotation_matrix(0.0, 1.0), np.eye(3))


def test_isotropic_mat2_invariant_under_rotation():
    e, nu = 1.0e7, 0.3
    iso = MaterialTable([Mat1(8, e, e / (2 * (1 + nu)), nu)]).mat2(8).g
    t = MaterialTable(mat2=[Mat2(5, *iso[np.triu_indices(3)])])
    np.testing.assert_allclose(t.mat2(5, 0.6, 0.8).g, iso, rtol=1e-12, atol=1e-6)


def test_bad_angle(table):
    with pytest.raises(MaterialError):
        table.mat2(2, 0.5, 0.5)


def test_transverse_shear_and_density(table):
    assert table.mat(1, 3) == (3.8e6, 0.0, 3.8e6)
    assert table.mat(2, 3) == (0.0, 0.0, 0.0)
    assert table.mat(1, 4) == 0.1
    assert table.mat(2, 4) == 0.2


def test_errors(table):
    with pytest.raises(MaterialNotFoundError):
        table.mat(99, 2)
    with pytest.raises(MaterialError):
        table.mat(1, 7)
    with pytest.raises(MaterialError):
        table.add(STEEL)


def test_mat1_searched_before_mat2():
    t = MaterialTable([Mat1(3, 1.0, 0.5, 0.0)], [Mat2(3, 9, 9, 9, 9, 9, 9)])
    assert t.mat2(3).source == "MAT1"
