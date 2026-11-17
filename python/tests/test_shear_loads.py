"""CSHEAR thermal loads (``TSHEAR``) against the Fortran harness and closed forms.

Parity is checked on the SIL-ordered load words (``shear_load_golden.json`` and the
live ``build/tshearld`` harness when gfortran is available) for the reference and
batched ports, at a relative tolerance of 1e-12.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil

import numpy as np
import pytest

from nastran95py.bulk import load_grid_temperatures, read_ecpts
from nastran95py.coords import transform_to_basic
from nastran95py.elements import (
    LAYOUTS,
    ElementData,
    ShearGeometryError,
    ShearLoadBatch,
    load_vector,
    shear_load,
    shear_loads,
)
from nastran95py.materials import Mat1, MaterialTable
from nastran95py.reference.inventory import repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import runner as R
from tests.fortran_harness import shear_cases as SC
from tests.fortran_harness import shear_load_cases as LC
from tests.test_rod_loads import CSTM, CSTM_EXACT, MATS

RTOL = 1e-12
CASES = LC.SHEAR_LOAD_CASES
SMOKE_DECKS = ("d01011a", "d01012a", "d01013a", "d01014a")


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["SHEAR"].parse(case.words)


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _python_loads(cases) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load words at ``ILOAD..`` of ``cases`` from the reference and the batched path."""
    datas = [_data(c) for c in cases]
    ref = []
    for c, d in zip(cases, datas, strict=True):
        load = shear_load(d, MATS, c.meta["temperatures"], CSTM)
        ref.append(load_vector(np.array([load.sil]) - LC.SIL0, load.force[None], c.meta["nload"]))
    batch = ShearLoadBatch.from_elements(
        datas, MATS, {d.eid: c.meta["temperatures"] for c, d in zip(cases, datas, strict=True)}
    )
    forces = shear_loads(batch, CSTM)
    vec = [
        load_vector(batch.sil[i : i + 1] - LC.SIL0, forces[i : i + 1], c.meta["nload"])
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
            f"CSHEAR loads {label} ({name})", sum(f.size for f in fortran), worst_abs, worst_rel,
            element="CSHEAR", quantity="load", source=harness_source(label),
        )  # fmt: skip


def test_shear_loads_match_fortran_golden():
    golden = json.loads(LC.GOLDEN.read_text())
    assert set(golden) == {c.name for c in CASES}
    words = [np.asarray(golden[c.name]["load"], np.float32).astype(float) for c in CASES]
    assert sum(np.count_nonzero(w) for w in words) > 0
    _compare(words, "golden")


@pytest.mark.fortran
def test_shear_loads_match_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    _compare(LC.run(), "live")


def _panel(points, cids=(0, 0, 0, 0), mid=1, t=0.1, sils=(1, 7, 13, 19)) -> ElementData:
    grids = list(zip(cids, points, strict=True))
    return LAYOUTS["SHEAR"].parse(SC.shear_cstm_est(31, grids, mid, t, 0.0, sils))


_RECT = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 2.0, 0.0), (0.0, 2.0, 0.0)]


@pytest.mark.parametrize("real_words", [True, False])
def test_rectangle_closed_form(real_words):
    # Squared NORM lengths: AREA = 0.5*(2ab)**2*T/(2a**2) = b**2*T on the a-edges (1, 3)
    # and a**2*T on the b-edges (2, 4); each edge pushes its grids apart along the edge.
    a, b, t, dtemp = 5.0, 2.0, 0.1, 100.0
    mat = MATS.mat1(1)
    load = shear_load(_panel(_RECT, t=t), MATS, [mat.tref + dtemp] * 4, real_words=real_words)
    fa = mat.e * mat.alpha * dtemp * b * b * t
    fb = mat.e * mat.alpha * dtemp * a * a * t
    expect = np.array([[-fa, -fb, 0], [fa, -fb, 0], [fa, fb, 0], [-fa, fb, 0]])
    np.testing.assert_allclose(load.force, expect, rtol=1e-6 if real_words else 1e-14)


def test_edge_temperature_is_mean_of_its_grids():
    mat = MATS.mat1(1)
    temps = [mat.tref + d for d in (10.0, 30.0, -20.0, 60.0)]
    load = shear_load(_panel(_RECT), MATS, temps, real_words=False)
    unit = mat.e * mat.alpha * 0.1
    # Grid 1: edge 1 (grids 1-2, mean 20, area b**2 T) along -x, edge 4 (grids 4-1, 35, a**2 T) -y.
    np.testing.assert_allclose(load.force[0], [-unit * 4 * 20, -unit * 25 * 35, 0], rtol=1e-14)


BASIC_CASES = [c for c in CASES if all(g.cid == 0 for g in _data(c).grids)]


@pytest.mark.parametrize("case", BASIC_CASES, ids=lambda c: c.name)
def test_basic_system_loads_are_self_equilibrated(case):
    d = _data(case)
    f = shear_load(d, MATS, case.meta["temperatures"], real_words=False).force
    x = np.array([g.xyz for g in d.grids])
    scale = max(float(np.abs(f).max()), 1.0)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(np.cross(x, f).sum(axis=0), 0.0, atol=1e-11 * scale)


def test_tref_cancels_and_zero_thickness_gives_zero():
    mat = MATS.mat1(1)
    assert not shear_load(_panel(_RECT), MATS, [mat.tref] * 4).force.any()
    assert not shear_load(_panel(_RECT, t=0.0), MATS, [250.0] * 4).force.any()


@pytest.mark.parametrize("cids", [(10, 20, 30, 0), (40, 41, 42, 40), (21, 21, 10, 21)])
def test_global_components_rotate_basic_loads(cids):
    # With the grid point handed to BASGLB (fortran_pont=False) the global loads are the
    # basic loads in each grid's local frame; rectangular systems ignore PONT entirely.
    pts = [(0.3, 0.2, 0.1), (8.0, 0.4, 0.6), (9.0, 6.0, -0.4), (1.0, 5.0, 0.8)]
    temps = [80.0, 140.0, 20.0, 210.0]
    basic = shear_load(_panel(pts), MATS, temps, real_words=False).force
    d = _panel(pts, cids=cids)
    glob = shear_load(d, MATS, temps, CSTM_EXACT, real_words=False, fortran_pont=False).force
    for g, fb, fg in zip(d.grids, basic, glob, strict=True):
        if g.cid == 0:
            np.testing.assert_array_equal(fg, fb)
            continue
        t = transform_to_basic(CSTM_EXACT, g.cid, g.xyz)
        np.testing.assert_allclose(t @ fg, fb, atol=1e-9 * abs(fb).max())
    fp = shear_load(d, MATS, temps, CSTM_EXACT, real_words=False).force
    for g, a, b in zip(d.grids, fp, glob, strict=True):
        if g.cid == 0 or CSTM_EXACT[g.cid].ctype == 1:
            np.testing.assert_array_equal(a, b)


def test_fortran_pont_reads_trimex_words():
    # BASGLB gets ISILS(IB): the SIL, MID and THICK words that follow, read as REALs.
    from nastran95py.elements.shear_loads import _trimex_words

    words = _trimex_words(np.array([[1, 7, 13, 19]]), np.array([2]), np.array([0.25]))
    assert words.dtype == np.float32
    assert words[0, :5].view(np.int32).tolist() == [1, 7, 13, 19, 2]
    assert words[0, 5] == np.float32(0.25)


def test_batch_matches_reference_float64():
    datas = [_data(c) for c in CASES]
    temps = {d.eid: c.meta["temperatures"] for c, d in zip(CASES, datas, strict=True)}
    batch = ShearLoadBatch.from_elements(datas, MATS, temps)
    for fortran_pont in (True, False):
        got = shear_loads(batch, CSTM, real_words=False, fortran_pont=fortran_pont)
        ref = np.array(
            [shear_load(d, MATS, temps[d.eid], CSTM, False, fortran_pont).force for d in datas]
        )
        np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-9)


def test_errors():
    mat = MaterialTable([Mat1(1, 1.0e7, 4.0e6, 0.25, alpha=1e-5)])
    pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    with pytest.raises(ShearGeometryError, match="2026"):
        shear_load(_panel(pts), mat, [100.0] * 4)
    with pytest.raises(ShearGeometryError, match="2026"):
        shear_loads(ShearLoadBatch.from_elements([_panel(pts)], mat, {31: [100.0] * 4}))
    with pytest.raises(ValueError, match="four grid temperatures"):
        shear_load(_panel(_RECT), mat, [100.0] * 3)
    with pytest.raises(KeyError, match="31"):
        ShearLoadBatch.from_elements([_panel(_RECT)], mat, {})
    with pytest.raises(KeyError, match="coordinate system 10"):
        shear_load(_panel(_RECT, cids=(10, 0, 0, 0)), mat, [100.0] * 4)
    rod = LAYOUTS["ROD"].parse([11, 1, 7, 1, 0.8, 0.1, 0.0, 0.0, *[0, 0.0, 0.0, 0.0] * 2, 0.0])
    with pytest.raises(ValueError, match="SHEAR records"):
        shear_load(rod, mat, [100.0] * 4)


def _with_temp_set(text: str, set_id: int, grids: list[int]) -> str:
    """The deck with a TEMPD default and a few TEMP grid values for set ``set_id``."""
    cards = [f"TEMPD   {set_id:<8d}170.0"]
    for k, g in enumerate(grids):
        cards.append(f"TEMP    {set_id:<8d}{g:<8d}{75.0 + 37.5 * k:<8.2f}")
    return re.sub(r"(?im)^ENDDATA", "\n".join(cards) + "\nENDDATA", text, count=1)


@pytest.mark.parametrize("deck_id", SMOKE_DECKS)
def test_deck_cshears_under_temp_set_smoke(deck_id):
    text = (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")
    _, ecpts = read_ecpts(text)
    grids = sorted({g for r in ecpts["SHEAR"] for g in r.connection.grids})[:6]
    set_id = 9901
    model, ecpts = read_ecpts(_with_temp_set(text, set_id, grids))
    shears = ecpts["SHEAR"]
    assert shears
    temps = load_grid_temperatures(model, shears, set_id)
    assert {t for v in temps.values() for t in v} > {170.0}
    datas = [r.data for r in shears]
    # The decks' MAT1 cards have no ALPHA (their TEMP loads would vanish); give every
    # material one so the deck geometry produces nonzero loads.
    mats = MaterialTable(
        [
            dataclasses.replace(model.materials.mat1_at(mid, None), alpha=1.2e-5, tref=20.0)
            for mid in sorted({int(d["mid"]) for d in datas})
        ]
    )
    batch = ShearLoadBatch.from_elements(datas, mats, temps)
    forces = shear_loads(batch, model.cstm)
    ref = np.array([shear_load(d, mats, temps[d.eid], model.cstm).force for d in datas])
    np.testing.assert_array_equal(forces, ref)
    assert np.isfinite(forces).all() and np.abs(forces).max() > 0
    for d, f in zip(datas, forces, strict=True):
        if all(g.cid == 0 for g in d.grids):
            scale = float(np.abs(f).max())
            np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-5 * scale)
