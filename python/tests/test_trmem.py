"""CTRMEM stiffness, mass and thermal load (``nastran95py.elements.trmem``).

Parity: the reference and vectorised paths against the Fortran ``TRMEMD`` and
``TRIMEM`` dumps (basic-system triangles in ``harness_golden.json``,
offset/rotated/curvilinear systems, material angles and NSM in
``trmem_golden.json``, and the live harness when gfortran is available) at
relative 1e-12; the element-axis kernel against the closed-form CST; thermal
load against the stiffness times the free thermal expansion; and every CTRMEM
in the RF1 decks.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import read_ecpts
from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.elements import (
    LAYOUTS,
    TrmemBatch,
    TrmemGeometryError,
    expand_to_grid_dof,
    membrane_b_matrix,
    triangle_geometry,
    trimem,
    trmem_element_stiffness,
    trmem_matrices,
    trmem_properties,
    trmem_thermal_loads,
    trmemd,
)
from nastran95py.materials import Mat1, Mat2, MaterialTable, Matt1, TableM1
from nastran95py.reference.inventory import RF1_DECKS, repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R
from tests.fortran_harness import trmem_cases as TC

RTOL = 1e-12
MATS = MaterialTable([Mat1(m, *v) for m, v in C.MAT1] + [Mat2(m, *v) for m, v in C.MAT2])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)


def _orthonormal(t: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(t)
    return u @ vt


# The REAL CSTM words are not exactly orthonormal; closed-form checks need a true rotation.
CSTM_EXACT = {
    cid: CoordSystem(cid, cs.ctype, cs.origin, _orthonormal(cs.t)) for cid, cs in CSTM.items()
}
EMG_CASES = [c for c in C.ALL_CASES if c.routine == "TRMEMD"] + [
    c for c in TC.TRMEM_CASES if c.routine == "TRMEMD"
]
LOAD_CASES = [c for c in TC.TRMEM_CASES if c.routine == "TRIMEM"]
SMOKE_DECKS = ("d01011a", "d01012a", "d01161a")


def _golden() -> dict:
    data = json.loads(R.GOLDEN.read_text())
    data.update(json.loads(TC.GOLDEN.read_text()))
    return data


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case):
    return LAYOUTS["TRMEM"].parse(case.words)


class _Worst:
    def __init__(self) -> None:
        self.abs, self.rel, self.n = 0.0, 0.0, 0

    def add(self, a: float, r: float, n: int) -> None:
        self.abs, self.rel, self.n = max(self.abs, a), max(self.rel, r), self.n + n


def _compare_emg(cases, dumps: dict, label: str) -> None:
    datas = [_data(c) for c in cases]
    batch = trmem_matrices(TrmemBatch.from_elements(datas, MATS), CSTM)
    worst = {"ref": _Worst(), "batch": _Worst()}
    per_q = {"K": _Worst(), "M": _Worst()}
    for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
        out = trmemd(data, MATS, CSTM)
        k_ref, m_ref = dumps[case.name]
        assert tuple(out.sils) == tuple(sorted(data["sil"]))
        np.testing.assert_array_equal(batch.sils[i], out.sils)
        for q, mine, full, ref in (
            ("K", out.stiffness, batch.k[i], k_ref),
            ("M", out.mass, batch.m[i], m_ref),
        ):
            assert (mine.ndof, mine.icode, mine.dict2) == (ref["ndof"], ref["icode"], ref["dict2"])
            vals = np.asarray(ref["values"])
            ra, rr = _err(mine.values, vals)
            ba, br = _err(full, R._square({**ref, "values": vals}))
            assert rr <= RTOL and br <= RTOL, (case.name, ref["file"], rr, br)
            worst["ref"].add(ra, rr, vals.size)
            worst["batch"].add(ba, br, vals.size)
            per_q[q].add(max(ra, ba), max(rr, br), 2 * vals.size)
    for key, w in worst.items():
        record_parity(f"trmem K/M {key} vs {label}", w.n, w.abs, w.rel)
    for q, w in per_q.items():
        record_parity(
            f"trmem {q} vs {label}", w.n, w.abs, w.rel,
            element="CTRMEM", quantity=q, source=harness_source(label),
        )  # fmt: skip


def _compare_loads(cases, dumps: dict, label: str) -> None:
    datas = [_data(c) for c in cases]
    tbar = [c.extra[0] for c in cases]
    batch = trmem_thermal_loads(TrmemBatch.from_elements(datas, MATS), tbar, CSTM)
    worst = {"ref": _Worst(), "batch": _Worst()}
    for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
        # The harness prints REAL words with 9 digits: read back as float32 exactly.
        want = np.asarray(dumps[case.name], np.float32).astype(float)
        assert np.abs(want).max() > 0.0, case.name
        ref = trimem(data, MATS, case.extra[0], CSTM).add_to(np.zeros(TC.NPG))
        got = batch[i].add_to(np.zeros(TC.NPG))
        for key, vec in (("ref", ref), ("batch", got)):
            a, r = _err(vec, want)
            assert r <= RTOL, (case.name, key, r)
            worst[key].add(a, r, want.size)
    for key, w in worst.items():
        record_parity(
            f"trmem thermal load {key} vs {label}", w.n, w.abs, w.rel,
            element="CTRMEM", quantity="load", source=harness_source(label),
        )  # fmt: skip


def test_trmemd_matches_fortran_golden():
    gold = _golden()
    assert len(EMG_CASES) == 3 + len(TC.TRMEM_GEOMS) and len(LOAD_CASES) == 14
    assert {c.name for c in TC.TRMEM_CASES} <= set(gold)
    _compare_emg(EMG_CASES, {c.name: gold[c.name]["matrices"] for c in EMG_CASES}, "golden")


def test_trimem_matches_fortran_golden():
    gold = _golden()
    _compare_loads(
        LOAD_CASES, {c.name: gold[c.name]["words"]["LOAD"] for c in LOAD_CASES}, "golden"
    )


@pytest.mark.fortran
def test_trmem_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    live = {r.case.name: r for r in R.run(EMG_CASES + LOAD_CASES, cstm=TC.CSTM)}
    _compare_emg(EMG_CASES, {c.name: live[c.name].matrices for c in EMG_CASES}, "live harness")
    _compare_loads(
        LOAD_CASES, {c.name: live[c.name].words["LOAD"] for c in LOAD_CASES}, "live harness"
    )
    gold = _golden()
    errs = [
        _err(m["values"], g["values"])
        for c in EMG_CASES
        for m, g in zip(live[c.name].matrices, gold[c.name]["matrices"], strict=True)
    ] + [_err(live[c.name].words["LOAD"], gold[c.name]["words"]["LOAD"]) for c in LOAD_CASES]
    a, r = max(e[0] for e in errs), max(e[1] for e in errs)
    record_parity("trmem golden vs live harness", len(errs), a, r)
    assert r <= RTOL


def test_trmems_single_precision_close():
    """``TRMEMS`` (REAL arithmetic) matches the double kernel to float32 round-off."""
    gold = _golden()
    for case in (c for c in C.ALL_CASES if c.routine == "TRMEMS"):
        out = trmemd(_data(case), MATS)
        for mine, ref in zip((out.stiffness, out.mass), gold[case.name]["matrices"], strict=True):
            assert _err(mine.values, ref["values"])[1] < 1e-6, case.name


def _cst_closed_form(xy, g, t) -> np.ndarray:
    """Textbook constant-strain triangle ``t A B^T D B`` in element axes."""
    (x1, y1), (x2, y2), (x3, y3) = xy
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)
    bm = np.zeros((3, 6))
    bm[0, 0::2], bm[1, 1::2] = b, c
    bm[2, 0::2], bm[2, 1::2] = c, b
    return t * area * bm.T @ g @ bm


def test_element_stiffness_closed_form():
    g = MATS.mat2(3).g
    for xb, xc, yc in ((4.0, 1.0, 3.0), (2.0, 0.0, 5.0), (3.5, 4.2, 1.1), (1.0, -0.7, 0.4)):
        k = trmem_element_stiffness(xb, xc, yc, g, 0.15)
        want = _cst_closed_form(((0, 0), (xb, 0), (xc, yc)), g, 0.15)
        a, r = _err(k, want)
        assert r < 1e-14, (xb, xc, yc, r)
        assert np.allclose(k, k.T, rtol=0, atol=1e-14 * np.abs(k).max())
        b = membrane_b_matrix(xb, xc, yc)
        # rigid in-plane motions produce no strain
        for u in ([1, 0] * 3, [0, 1] * 3, [0, 0, 0, xb, -yc, xc]):
            np.testing.assert_allclose(b @ np.array(u, float), 0.0, atol=1e-14)


def test_global_stiffness_and_load_consistent():
    """K annihilates rigid motions, and the thermal load equals K times the free expansion."""
    for case in LOAD_CASES:
        d = _data(case)
        props = trmem_properties(d, MATS, real_words=False)
        out = trmemd(d, props, CSTM_EXACT, real_words=False)
        load = trimem(d, props, case.extra[0], CSTM_EXACT, real_words=False)
        order = np.argsort(d["sil"], kind="stable")
        xyz = np.array([g.xyz for g in d.grids])
        ta = transforms_to_basic(CSTM_EXACT, [g.cid for g in d.grids], xyz)
        geom = triangle_geometry(xyz, real_words=False)
        e = geom.e
        k_basic = np.zeros((9, 9))
        tb = np.zeros((9, 9))
        for a, gi in enumerate(order):
            tb[3 * a : 3 * a + 3, 3 * a : 3 * a + 3] = ta[gi]
        k_basic = tb @ out.k() @ tb.T
        rigid = [np.tile(v, 3) for v in np.eye(3)]
        rigid += [np.cross(w, xyz[order]).reshape(-1) for w in np.eye(3)]
        for u in rigid:
            assert np.abs(k_basic @ u).max() < 1e-12 * np.abs(k_basic).max() * np.abs(xyz).max()
        # free thermal strain (element axes) -> basic displacements of each grid
        eps = props.alpha * (case.extra[0] - props.tref)
        strain = np.array([[eps[0], eps[2] / 2], [eps[2] / 2, eps[1]]])
        local = (xyz - xyz[0]) @ e
        u_basic = np.array([e @ (strain @ p) for p in local])[order].reshape(-1)
        p_basic = np.zeros(9)
        for a, gi in enumerate(order):
            p_basic[3 * a : 3 * a + 3] = ta[gi] @ load.loads[gi]
        a, r = _err(k_basic @ u_basic, p_basic)
        assert r < 1e-12, (case.name, r)


def test_mass_closed_form_and_grid_expansion():
    d = _data(next(c for c in EMG_CASES if c.name == "CTRMEM-aniso30-TRMEMD"))
    out = trmemd(d, MATS, real_words=False)
    xyz = np.array([g.xyz for g in d.grids])
    area = 0.5 * np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
    total = (MATS.density(3) * d["t"] + d["nsm"]) * area
    assert np.isclose(np.trace(out.m()), 3 * total, rtol=1e-14)
    kg, mg = out.k_grid(), out.m_grid()
    assert kg.shape == mg.shape == (18, 18)
    tr = [6 * g + c for g in range(3) for c in range(3)]
    np.testing.assert_array_equal(kg[np.ix_(tr, tr)], out.k())
    assert not np.delete(np.delete(kg, tr, 0), tr, 1).any()
    np.testing.assert_array_equal(expand_to_grid_dof(out.m()[None])[0], mg)


def test_temperature_dependent_material():
    mats = MaterialTable(
        [Mat1(1, 1.0e7, 4.0e6, 0.25, 0.1, 1.0e-5)],
        matt1=[Matt1(1, e=7, g=7)],
        tables=[TableM1(7, (0.0, 200.0), (1.0, 0.5))],
    )
    d = _data(EMG_CASES[0])
    hot = type(d)("TRMEM", d.fields, d.grids, temperature=100.0)
    k_cold, k_hot = trmemd(d, mats).k(), trmemd(hot, mats).k()
    np.testing.assert_allclose(k_hot, 0.75 * k_cold, rtol=1e-6, atol=1e-9 * np.abs(k_cold).max())


def test_empty_batch():
    for real_words in (True, False):
        batch = TrmemBatch.from_elements([], MATS, real_words)
        out = trmem_matrices(batch, CSTM, real_words)
        assert out.k.shape == out.m.shape == (0, 9, 9) and out.sils.shape == (0, 3)
        assert trmem_thermal_loads(batch, 100.0, CSTM, real_words) == []


def test_degenerate_triangle_is_fatal():
    words = list(EMG_CASES[0].words)
    words[13:16] = words[9:12]  # grid B on top of grid A
    flat = LAYOUTS["TRMEM"].parse(words)
    collinear = list(EMG_CASES[0].words)
    collinear[17:20] = [2.0, 0.0, 0.0]
    line = LAYOUTS["TRMEM"].parse(collinear)
    for d in (flat, line):
        with pytest.raises(TrmemGeometryError):
            trmemd(d, MATS)
        with pytest.raises(TrmemGeometryError, match=str(d.eid)):
            trmem_matrices(TrmemBatch.from_elements([_data(EMG_CASES[1]), d], MATS))
        with pytest.raises(TrmemGeometryError):
            trimem(d, MATS, 100.0)


def _deck_trmems():
    for deck_id in sorted(RF1_DECKS):
        text = (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        if ecpts.get("TRMEM"):
            yield deck_id, model, [r.data for r in ecpts["TRMEM"]]


def test_every_deck_trmem_through_kernel():
    """Smoke test: every CTRMEM of the inventoried RF1 decks (incl. d01011a, d01012a, d01161a)."""
    seen: dict[str, int] = {}
    worst = _Worst()
    for deck_id, model, datas in _deck_trmems():
        batch = trmem_matrices(TrmemBatch.from_elements(datas, model.materials), model.cstm)
        tbar = [(d.temperature or 0.0) + 100.0 for d in datas]
        loads = trmem_thermal_loads(
            TrmemBatch.from_elements(datas, model.materials), tbar, model.cstm
        )
        for i, d in enumerate(datas):
            ref = trmemd(d, model.materials, model.cstm)
            a, r = _err(batch.k[i], ref.k())
            assert r <= 1e-15, (deck_id, d.eid)
            np.testing.assert_array_equal(batch.m[i], ref.m())
            k = batch.k[i]
            assert np.isfinite(k).all() and np.allclose(
                k, k.T, rtol=0, atol=1e-13 * np.abs(k).max()
            )
            assert np.linalg.eigvalsh(k).min() > -1e-9 * np.abs(k).max()
            assert np.linalg.matrix_rank(k, tol=1e-9 * np.abs(k).max()) == 3
            lref = trimem(d, model.materials, tbar[i], model.cstm)
            np.testing.assert_array_equal(loads[i].loads, lref.loads)
            # a uniform temperature load on a free triangle is self-equilibrated
            assert np.abs(loads[i].loads.sum(axis=0)).max() <= 1e-4 * np.abs(lref.loads).max()
            worst.add(a, r, k.size)
        seen[deck_id] = len(datas)
    assert set(SMOKE_DECKS) <= set(seen) and all(seen[d] > 0 for d in SMOKE_DECKS), seen
    record_parity(
        f"trmem deck smoke ({sum(seen.values())} CTRMEM): batch vs reference",
        worst.n,
        worst.abs,
        worst.rel,
    )
