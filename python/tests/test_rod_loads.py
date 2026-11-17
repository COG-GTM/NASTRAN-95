"""Rod/tube thermal and deformation loads (``nastran95py.elements.rod_loads``, port of ``ROD``).

Parity: the reference and vectorised paths against the Fortran ``ROD`` load
words (``rod_load_golden.json`` and the live ``build/rodload`` harness when
gfortran is available) at relative 1e-12; closed-form free thermal expansion
(load E*A*ALPHA*DT, and the stiffness solve elongates the rod by ALPHA*DT*L);
the ``DEFORM`` term E*A*DELTA/L**2 that ``ROD`` computes; and the d01161a rods
under its TEMP set as a smoke test.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import deformations, load_temperatures, read_ecpts
from nastran95py.coords import CoordSystem
from nastran95py.elements import (
    LAYOUTS,
    ElementData,
    GridPoint,
    RodGeometryError,
    RodLoadBatch,
    load_vector,
    rod_load,
    rod_loads,
    rodd,
    tube_area,
)
from nastran95py.materials import Mat1, MaterialTable
from nastran95py.reference.cards import read_deck
from nastran95py.reference.inventory import repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import rod_load_cases as LC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(mid, *vals) for mid, vals in C.MAT1])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)
CASES = LC.ROD_LOAD_CASES


def _orthonormal(t: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(t)
    return u @ vt


CSTM_EXACT = {
    cid: CoordSystem(cid, cs.ctype, cs.origin, _orthonormal(cs.t)) for cid, cs in CSTM.items()
}


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["TUBE" if case.element == "CTUBE" else "ROD"].parse(case.words)


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _python_loads(cases) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """SIL-ordered load vectors of ``cases`` from the reference and the batched path."""
    datas = [_data(c) for c in cases]
    ref = [
        rod_load(d, MATS, c.meta["temperature"], c.meta["deformation"], CSTM).vector(c.extra[5])
        for c, d in zip(cases, datas, strict=True)
    ]
    batch = RodLoadBatch.from_elements(
        datas,
        MATS,
        {
            d.eid: c.meta["temperature"]
            for c, d in zip(cases, datas, strict=True)
            if c.meta["temperature"] is not None
        },
        {d.eid: c.meta["deformation"] for c, d in zip(cases, datas, strict=True)},
    )
    forces = rod_loads(batch, CSTM)
    vec = [
        load_vector(batch.sil[i : i + 1], forces[i : i + 1], c.extra[5])
        for i, c in enumerate(cases)
    ]
    return ref, vec


def _compare(fortran: list[np.ndarray], label: str) -> None:
    ref, vec = _python_loads(CASES)
    for name, got in (("ref", ref), ("batch", vec)):
        worst_abs = worst_rel = 0.0
        for case, f, p in zip(CASES, fortran, got, strict=True):
            np.testing.assert_allclose(p, f, rtol=RTOL, atol=0.0, err_msg=f"{name} {case.name}")
            a, r = _err(p, f)
            worst_abs, worst_rel = max(worst_abs, a), max(worst_rel, r)
        record_parity(
            f"ROD loads {label} ({name})", sum(f.size for f in fortran), worst_abs, worst_rel,
            element="CROD/CONROD", quantity="load", source=harness_source(label),
        )  # fmt: skip


def test_rod_loads_match_fortran_golden():
    golden = json.loads(LC.GOLDEN.read_text())
    assert set(golden) == {c.name for c in CASES}
    words = [np.asarray(golden[c.name]["load"], np.float32).astype(float) for c in CASES]
    _compare(words, "golden")


@pytest.mark.fortran
def test_rod_loads_match_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    _compare(LC.run(), "live")


def _rod(cids=(0, 0), a=(0.0, 0.0, 0.0), b=(3.0, -4.0, 12.0), mid=1, area=0.8, element="ROD"):
    words = (
        [11, 1, 7, mid, area, 0.1, 0.0, 0.0]
        if element == "ROD"
        else [11, 1, 7, mid, 1.5, 0.125, 0.0]
    )
    words += [cids[0], *a, cids[1], *b, 0.0]
    return LAYOUTS[element].parse(words)


@pytest.mark.parametrize("cids", [(0, 0), (10, 20), (30, 41), (42, 0)])
def test_free_thermal_expansion_closed_form(cids):
    data = _rod(cids)
    mat = MATS.mat1(1)
    dt_ = 125.0
    length = 13.0
    load = rod_load(data, MATS, mat.tref + dt_, cstm=CSTM_EXACT, real_words=False)
    expect = mat.e * 0.8 * mat.alpha * dt_
    # Global loads back to basic: equal and opposite, magnitude E*A*ALPHA*DT along the axis.
    from nastran95py.coords import transforms_to_basic

    basic = [
        transforms_to_basic(CSTM_EXACT, g.cid, g.xyz)[0] @ f if g.cid else f
        for g, f in zip(data.grids, load.force, strict=True)
    ]
    axis = (data.grids[1].xyz - data.grids[0].xyz) / length
    np.testing.assert_allclose(basic[1], expect * axis, rtol=1e-12)
    np.testing.assert_allclose(basic[0], -expect * axis, rtol=1e-12)

    # Free expansion: fix grid A, load grid B; the rod lengthens by ALPHA*DT*L. K of the
    # free grid is rank one (axial), so take the minimum-norm solution.
    k = rodd(data, MATS, CSTM_EXACT, real_words=False).k()
    p = load.vector(12)
    u_b = np.linalg.pinv(k[6:9, 6:9], rcond=1e-10) @ p[6:9]
    g_b = data.grids[1]
    t_b = transforms_to_basic(CSTM_EXACT, g_b.cid, g_b.xyz)[0] if g_b.cid else np.eye(3)
    u_basic = t_b @ u_b
    np.testing.assert_allclose(u_basic, mat.alpha * dt_ * length * axis, rtol=1e-9, atol=1e-15)


def test_deform_load_is_e_a_delta_over_length_squared():
    data = _rod()
    mat = MATS.mat1(1)
    load = rod_load(data, MATS, deformation=0.003, real_words=False)
    axis = np.array([3.0, -4.0, 12.0]) / 13.0
    np.testing.assert_allclose(load.force[1], mat.e * 0.8 * 0.003 / 169.0 * axis, rtol=1e-12)
    np.testing.assert_allclose(load.force[0], -load.force[1], rtol=0)
    both = rod_load(data, MATS, mat.tref + 10.0, 0.003, real_words=False)
    therm = rod_load(data, MATS, mat.tref + 10.0, real_words=False)
    np.testing.assert_allclose(both.force, therm.force + load.force, rtol=1e-12)


def test_no_load_data_gives_zero_and_tref_cancels():
    data = _rod(mid=1)
    assert not rod_load(data, MATS).force.any()
    assert not rod_load(data, MATS, MATS.mat1(1).tref).force.any()


def test_tube_area_and_load():
    assert tube_area(1.5, 0.125, real_words=False) == pytest.approx(np.pi * 1.375 * 0.125)
    data = _rod(element="TUBE")
    mat = MATS.mat1(1)
    load = rod_load(data, MATS, mat.tref + 50.0, real_words=False)
    np.testing.assert_allclose(
        np.linalg.norm(load.force[1]), mat.e * np.pi * 1.375 * 0.125 * mat.alpha * 50.0, rtol=1e-12
    )


def test_degenerate_rod_raises():
    data = _rod(b=(0.0, 0.0, 0.0))
    with pytest.raises(RodGeometryError):
        rod_load(data, MATS, 100.0)
    with pytest.raises(RodGeometryError):
        rod_loads(RodLoadBatch.from_elements([data], MATS, {11: 100.0}))


def test_other_layouts_rejected():
    data = ElementData(
        "SHEAR",
        {"eid": 1, "mid": 1, "sil": (1, 7)},
        [GridPoint(0, (0, 0, 0)), GridPoint(0, (1, 0, 0))],
    )
    with pytest.raises(ValueError):
        rod_load(data, MATS, 1.0)


def test_deform_cards():
    deck = read_deck((repo_root() / "inp" / "d11011b.inp").read_text(errors="replace"))
    assert deformations(deck.bulk, 1102) == {10: pytest.approx(0.089045)}
    assert deformations(deck.bulk, 1) == {}


def test_d01161a_rods_under_temp_set_smoke():
    text = (repo_root() / "inp" / "d01161a.inp").read_text(errors="replace")
    model, ecpts = read_ecpts(text)
    rods = ecpts["ROD"]
    assert rods
    set_id = 3000  # TEMPD 3000 (also the TEMPERATURE(MATERIALS) set)
    assert set_id in model.default_temperatures
    temps = load_temperatures(model, rods, set_id)
    datas = [r.data for r in rods]
    mats = model.materials
    batch = RodLoadBatch.from_elements(datas, mats, temps)
    forces = rod_loads(batch, model.cstm)
    ref = np.array([rod_load(d, mats, temps[d.eid], cstm=model.cstm).force for d in datas])
    np.testing.assert_array_equal(forces, ref)
    for d, f in zip(datas, forces, strict=True):
        mat = mats.mat1(int(d["mid"]), d.temperature)
        expect = abs(mat.e * float(d["area"]) * mat.alpha * (temps[d.eid] - mat.tref))
        assert all(g.cid == 0 for g in d.grids)
        np.testing.assert_allclose(f[0], -f[1], rtol=0)
        np.testing.assert_allclose(np.linalg.norm(f[1]), expect, rtol=1e-6)
    # The deck's MAT1 cards have no ALPHA, so its TEMP loads vanish; enforce a DEFORM on
    # every rod to exercise the deck geometry with nonzero loads.
    deform = {d.eid: 0.001 for d in datas}
    forces = rod_loads(RodLoadBatch.from_elements(datas, mats, temps, deform), model.cstm)
    for d, f in zip(datas, forces, strict=True):
        mat = mats.mat1(int(d["mid"]), d.temperature)
        length2 = float(np.sum((d.grids[0].xyz - d.grids[1].xyz) ** 2))
        expect = mat.e * float(d["area"]) * 0.001 / length2
        np.testing.assert_allclose(np.linalg.norm(f[1]), expect, rtol=1e-6)
    assert np.abs(forces).max() > 0
