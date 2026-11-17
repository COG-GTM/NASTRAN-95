from __future__ import annotations

import pytest

from nastran95py.materials import (
    Mat1,
    Mat2,
    MaterialError,
    MaterialTable,
    Matt1,
    TableM1,
    TableM2,
    TableM3,
    TableM4,
    interpolate,
)

# TABLEM1 100 of d01032a/d01033a/d01034a (E of MAT1 75 versus temperature).
D01032A_E = TableM1(
    100, (80.0, 150.0, 200.0, 250.0, 300.0), (10.4e6, 10.15e6, 9.84e6, 9.51e6, 9.15e6)
)


def test_interpolate_on_points_between_points_and_outside():
    x, y = D01032A_E.x, D01032A_E.y
    for xi, yi in zip(x, y, strict=True):
        assert interpolate(x, y, xi) == pytest.approx(yi)
    assert interpolate(x, y, 175.0) == pytest.approx((10.15e6 + 9.84e6) / 2)
    assert interpolate(x, y, 10.0) == pytest.approx(10.4e6 + 70.0 * 0.25e6 / 70.0)
    assert interpolate(x, y, 350.0) == pytest.approx(9.15e6 - 50.0 * 0.36e6 / 50.0)


def test_interpolate_decreasing_abscissae():
    x, y = (300.0, 200.0, 100.0), (3.0, 2.0, 0.0)
    assert interpolate(x, y, 150.0) == pytest.approx(1.0)
    assert interpolate(x, y, 400.0) == pytest.approx(4.0)
    assert interpolate(x, y, 0.0) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ((0.0, 10.0, 10.0, 20.0), (1.0, 2.0, 4.0, 5.0)),
        ((0.0, 5.0, 10.0, 10.0, 20.0), (0.0, 1.0, 2.0, 4.0, 5.0)),
    ],
)
def test_interpolate_step_returns_mean(x, y):
    assert interpolate(x, y, 10.0) == pytest.approx(3.0)
    assert interpolate(x, y, 15.0) == pytest.approx(4.5)


def test_tablem2_tablem3_scale_base_property():
    t2 = TableM2(2, 70.0, (0.0, 100.0), (1.0, 0.8))
    assert t2.evaluate(120.0, 1e7) == pytest.approx(0.9e7)
    t3 = TableM3(3, 20.0, 100.0, (0.0, 1.0), (1.0, 0.5))
    assert t3.evaluate(70.0, 2.0) == pytest.approx(1.5)
    assert t3.evaluate(220.0, 2.0) == pytest.approx(0.0)


def test_tablem4_polynomial_and_clamping():
    t = TableM4(4, 10.0, 2.0, 0.0, 30.0, (1.0, 0.5, 0.25))
    assert t.evaluate(14.0, 3.0) == pytest.approx(3.0 * (1.0 + 0.5 * 2 + 0.25 * 4))
    low = 3.0 * (1.0 + 0.5 * -5 + 0.25 * 25)
    high = 3.0 * (1.0 + 0.5 * 10 + 0.25 * 100)
    assert t.evaluate(-100.0, 3.0) == pytest.approx(low)
    assert t.evaluate(0.0, 3.0) == pytest.approx(low)
    assert t.evaluate(30.0, 3.0) == pytest.approx(high)
    assert t.evaluate(500.0, 3.0) == pytest.approx(high)


@pytest.mark.parametrize(
    "build",
    [
        lambda: TableM1(1, (0.0,), (1.0,)),
        lambda: TableM1(1, (0.0, 1.0), (1.0,)),
        lambda: TableM1(1, (0.0, 0.0, 1.0), (1.0, 2.0, 3.0)),
        lambda: TableM1(1, (0.0, 2.0, 1.0), (1.0, 2.0, 3.0)),
        lambda: TableM3(1, 0.0, 0.0, (0.0, 1.0), (1.0, 2.0)),
        lambda: TableM4(1, 0.0, 1.0, 0.0, 1.0, ()),
    ],
)
def test_invalid_tables(build):
    with pytest.raises(MaterialError):
        build()


def _d01032a() -> MaterialTable:
    mat1 = Mat1(75, 10.4e6, 3.9e6, 0.33, 2.5e-4, 1.28e-5, 70.0)
    return MaterialTable([mat1], matt1=[Matt1(75, e=100)], tables=[D01032A_E])


def test_matt1_replaces_only_referenced_properties():
    table = _d01032a()
    hot = table.mat1(75, temp=175.0)
    assert hot.e == pytest.approx(9.995e6)
    assert (hot.g, hot.nu, hot.rho) == (3.9e6, 0.33, 2.5e-4)
    assert table.mat1(75).e == 10.4e6
    assert table.mat(75, 4, temp=175.0) == pytest.approx(2.5e-4)


def test_matt1_flows_into_mat2_format_and_transverse_shear():
    mat1 = Mat1(1, 1e7, 4e6, 0.25)
    tables = [TableM2(10, 0.0, (0.0, 100.0), (1.0, 0.5)), TableM1(11, (0.0, 100.0), (0.2, 0.4))]
    table = MaterialTable([mat1], matt1=[Matt1(1, e=10, g=10, nu=11)], tables=tables)
    out = table.mat(1, 2, temp=50.0)
    g11 = 0.75e7 / (1.0 - 0.3**2)
    assert out.g[0, 0] == pytest.approx(g11)
    assert out.g[0, 1] == pytest.approx(0.3 * g11)
    assert out.g[2, 2] == pytest.approx(3e6)
    assert table.mat(1, 3, temp=50.0) == pytest.approx((3e6, 0.0, 3e6))


def test_mat2_ignores_temperature():
    m2 = Mat2(5, 1e7, 3e6, 0.0, 8e6, 0.0, 2e6)
    table = MaterialTable(mat2=[m2])
    assert (table.mat2(5, temp=500.0).to_words() == table.mat2(5).to_words()).all()


def test_matt1_validation():
    mat1 = Mat1(1, 1e7, 4e6, 0.25)
    t = TableM1(10, (0.0, 1.0), (1.0, 1.0))
    with pytest.raises(MaterialError, match="no matching MAT1"):
        MaterialTable([mat1], matt1=[Matt1(2, e=10)], tables=[t])
    with pytest.raises(MaterialError, match="undefined tables"):
        MaterialTable([mat1], matt1=[Matt1(1, e=11)], tables=[t])
    with pytest.raises(MaterialError, match="duplicate MATT1"):
        MaterialTable([mat1], matt1=[Matt1(1, e=10), Matt1(1, g=10)], tables=[t])
    with pytest.raises(MaterialError, match="duplicate TABLEM"):
        MaterialTable([mat1], tables=[t, t])
    assert Matt1(1, e=10, ss=10).table_ids() == {"e": 10, "ss": 10}
