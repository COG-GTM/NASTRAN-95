"""CQDMEM1 stiffness, mass, thermal load and stresses (``nastran95py.elements.qdmem1``).

Parity: ``QDMM1D`` (K, M), ``QDMM1`` (thermal load) and ``SQDM11``/``SQDM12``
(phase-1 words and stresses) against the Fortran dumps (basic-system cases in
``harness_golden.json``, warped/anisotropic/coordinate-system/branch cases in
``qdmem1_golden.json``, and the live harness when gfortran is available) at
relative 1e-12; physics checks (symmetry, rigid-body modes, constant-strain
patch, free thermal expansion); and the CQDMEM1 stress tables of d01012a,
d01013a and d01032a (MATT1 temperature-dependent E, 84 TEMP cards) at the
golden README tolerance.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import BulkModel, read_ecpts
from nastran95py.coords import CoordSystem
from nastran95py.elements import (
    LAYOUTS,
    QuadGeometryError,
    mean_plane,
    qdmem1_matrices,
    qdmem1_stress_table,
    qdmem1_stresses,
    qdmem1_thermal_loads,
    qdmm1,
    qdmm1d,
    qdmm1d_material_angle,
    qdmm1d_properties,
    sqdm11,
    sqdm12,
    trmem_properties,
)
from nastran95py.elements.qdmem1 import S4PISQ32
from nastran95py.elements.trmem import material_angle
from nastran95py.materials import Mat1, Mat2, MaterialTable, TableM1, TableM2, TableM3, TableM4
from nastran95py.reference.golden import GoldenTable, compare, load_golden, tables
from nastran95py.reference.inventory import RF1_DECKS, repo_root
from tests.conftest import record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import qdmem1_cases as QC
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(m, *v) for m, v in C.MAT1] + [Mat2(m, *v) for m, v in C.MAT2])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)

BASE = [c for c in C.ALL_CASES if c.element == "CQDMEM1"]
EMG_CASES = [c for c in BASE + QC.QDMEM1_CASES if c.routine == "QDMM1D"]
SINGLE_CASES = [c for c in BASE + QC.QDMEM1_CASES if c.routine == "QDMM1S"]
SDR_CASES = [c for c in BASE + QC.QDMEM1_CASES if c.kind == C.SDR2]
LOAD_CASES = QC.QDMEM1_LOAD_CASES


def _golden() -> dict:
    data = json.loads(R.GOLDEN.read_text())
    data.update(json.loads(QC.GOLDEN.read_text()))
    return data


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return max_abs, max_abs / scale if scale else max_abs


def _real(values) -> np.ndarray:
    """Harness REAL words (printed with 9 digits) read back as float32 exactly."""
    return np.asarray(values, np.float32).astype(float)


def _data(case: C.Case):
    return LAYOUTS["QDMEM1"].parse(case.words)


def _temperature(case: C.Case) -> float | None:
    return None if case.extra[5] == -1 else case.extra[6]


class _Worst:
    def __init__(self) -> None:
        self.abs, self.rel, self.n = 0.0, 0.0, 0

    def add(self, a: float, r: float, n: int) -> None:
        self.abs, self.rel, self.n = max(self.abs, a), max(self.rel, r), self.n + n


def _compare_emg(cases, dumps: dict, label: str) -> None:
    datas = [_data(c) for c in cases]
    batch = qdmem1_matrices(datas, MATS, CSTM)
    worst = {"ref": _Worst(), "batch": _Worst()}
    for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
        out = qdmm1d(data, MATS, CSTM)
        assert tuple(out.sils) == tuple(sorted(data["sil"]))
        np.testing.assert_array_equal(batch.sils[i], out.sils)
        for mine, full, ref in zip(
            (out.stiffness, out.mass), (batch.k[i], batch.m[i]), dumps[case.name], strict=True
        ):
            assert (mine.ndof, mine.icode, mine.dict2) == (ref["ndof"], ref["icode"], ref["dict2"])
            vals = np.asarray(ref["values"])
            ra, rr = _err(mine.values, vals)
            ba, br = _err(full, R._square({**ref, "values": vals}))
            assert rr <= RTOL and br <= RTOL, (case.name, ref["file"], rr, br)
            worst["ref"].add(ra, rr, vals.size)
            worst["batch"].add(ba, br, vals.size)
    for key, w in worst.items():
        record_parity(f"QDMM1D K/M {key} vs {label}", w.n, w.abs, w.rel)


def _compare_sdr(cases, dumps: dict, label: str) -> None:
    worst = {"phase1": _Worst(), "stress": _Worst()}
    for case in cases:
        data = _data(case)
        p1 = sqdm11(data, MATS, CSTM)
        want1 = _real(dumps[case.name]["PHASE1"])
        assert [R.real_word_as_int(w) for w in want1[:5]] == [data.eid, *data["sil"]]
        a, r = _err(p1.words()[5:], want1[5:])
        assert r <= RTOL, (case.name, "phase1", r)
        worst["phase1"].add(a, r, 40)
        st = sqdm12(p1, case.extra[9:], _temperature(case))
        want = _real(dumps[case.name]["STRESS"])
        assert R.real_word_as_int(want[0]) == data.eid
        # ANGLE comes from ATAN2 in REAL: libm atan2f may differ by an ulp across platforms.
        np.testing.assert_allclose(st.angle, want[4], rtol=2.5e-7, atol=1e-30)
        a, r = _err(np.delete(st.words(), 3), np.delete(want[1:], 3))
        assert r <= RTOL, (case.name, "stress", r)
        worst["stress"].add(a, r, 7)
        batch = qdmem1_stresses([data], MATS, case.extra[9:], CSTM, _temperature(case))
        np.testing.assert_array_equal(qdmem1_stress_table(batch)[0], st.row())
    for key, w in worst.items():
        record_parity(f"SQDM11/12 {key} vs {label}", w.n, w.abs, w.rel)


def _compare_loads(cases, dumps: dict, label: str) -> None:
    datas = [_data(c) for c in cases]
    tbar = [c.extra[0] for c in cases]
    batch = qdmem1_thermal_loads(datas, MATS, tbar, CSTM)
    worst = {"ref": _Worst(), "batch": _Worst()}
    for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
        want = _real(dumps[case.name])
        assert np.abs(want).max() > 0.0, case.name
        ref = qdmm1(data, MATS, case.extra[0], CSTM).add_to(np.zeros(QC.NPG))
        got = batch[i].add_to(np.zeros(QC.NPG))
        for key, vec in (("ref", ref), ("batch", got)):
            a, r = _err(vec, want)
            assert r <= RTOL, (case.name, key, r)
            worst[key].add(a, r, want.size)
    for key, w in worst.items():
        record_parity(f"QDMM1 thermal load {key} vs {label}", w.n, w.abs, w.rel)


# -- Fortran parity -----------------------------------------------------------------


def test_case_inventory():
    gold = _golden()
    assert {c.name for c in QC.QDMEM1_CASES + LOAD_CASES} <= set(gold)
    assert len(EMG_CASES) == 3 + len(QC.QDMEM1_GEOMS) == len(SINGLE_CASES)
    assert len(SDR_CASES) == 3 + len(QC.QDMEM1_GEOMS) + len(QC.THERMAL_STRESS)
    assert len(LOAD_CASES) == len(QC.QDMEM1_GEOMS) - 1  # "mixed" has three systems
    planar = qdmem1_matrices([_data(c) for c in EMG_CASES], MATS, CSTM).planar
    assert planar.any() and not planar.all()  # both QDMM1D branches


def test_qdmm1d_matches_fortran_golden():
    gold = _golden()
    _compare_emg(EMG_CASES, {c.name: gold[c.name]["matrices"] for c in EMG_CASES}, "golden")


def test_sqdm_matches_fortran_golden():
    gold = _golden()
    _compare_sdr(SDR_CASES, {c.name: gold[c.name]["words"] for c in SDR_CASES}, "golden")


def test_qdmm1_matches_fortran_golden():
    gold = _golden()
    _compare_loads(
        LOAD_CASES, {c.name: gold[c.name]["words"]["LOAD"] for c in LOAD_CASES}, "golden"
    )


@pytest.mark.fortran
def test_qdmem1_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    cases = EMG_CASES + SINGLE_CASES + SDR_CASES
    live = {r.case.name: r for r in R.run(cases, cstm=QC.CSTM)}
    live.update({r.case.name: r for r in R.run(LOAD_CASES, binary=QC.LOAD_BINARY, cstm=QC.CSTM)})
    assert not any(r.nogo for r in live.values())
    _compare_emg(EMG_CASES, {c.name: live[c.name].matrices for c in EMG_CASES}, "live harness")
    _compare_sdr(SDR_CASES, {c.name: live[c.name].words for c in SDR_CASES}, "live harness")
    _compare_loads(
        LOAD_CASES, {c.name: live[c.name].words["LOAD"] for c in LOAD_CASES}, "live harness"
    )
    gold = _golden()
    errs = [
        _err(m["values"], g["values"])
        for c in EMG_CASES + SINGLE_CASES
        for m, g in zip(live[c.name].matrices, gold[c.name]["matrices"], strict=True)
    ]
    errs += [
        _err(live[c.name].words[k], gold[c.name]["words"][k])
        for c in SDR_CASES
        for k in ("PHASE1", "STRESS")
    ]
    errs += [_err(live[c.name].words["LOAD"], gold[c.name]["words"]["LOAD"]) for c in LOAD_CASES]
    a, r = max(e[0] for e in errs), max(e[1] for e in errs)
    record_parity("qdmem1 golden vs live harness", len(errs), a, r)
    assert r <= RTOL


def test_qdmm1s_single_precision_close():
    """``QDMM1S`` (REAL arithmetic) matches the double kernel to float32 round-off."""
    gold = _golden()
    worst = _Worst()
    for case in SINGLE_CASES:
        out = qdmm1d(_data(case), MATS, CSTM)
        for mine, ref in zip((out.stiffness, out.mass), gold[case.name]["matrices"], strict=True):
            a, r = _err(mine.values, ref["values"])
            assert r < 1e-6, case.name
            worst.add(a, r, mine.values.size)
    record_parity("QDMM1S vs Python QDMM1D", worst.n, worst.abs, worst.rel)


def test_qdmm1d_material_angle_uses_four_pi_squared():
    """``QDMM1D`` reads DEGRA from ``/CONDAS/`` word 5 (4 pi^2), unlike ``SQDM11``."""
    for theta in (15.0, 30.0, -60.0):
        s, c = qdmm1d_material_angle(theta)
        want = float(np.float32(theta) * S4PISQ32)
        assert s == pytest.approx(np.sin(want), abs=1e-6)
        assert c == pytest.approx(np.cos(want), abs=1e-6)
        assert (s, c) != pytest.approx(material_angle(theta), abs=1e-3)
    assert qdmm1d_material_angle(0.0) == material_angle(0.0) == (0.0, 1.0)
    # With the textbook angle the anisotropic EMG case no longer matches the Fortran.
    gold = _golden()
    case = next(c for c in EMG_CASES if c.name == "CQDMEM1-aniso-QDMM1D")
    data = _data(case)
    textbook = qdmm1d(data, trmem_properties(data, MATS), CSTM)
    assert _err(textbook.stiffness.values, gold[case.name]["matrices"][0]["values"])[1] > 0.1
    # MAT1 does not depend on the angle.
    mat1 = next(c for c in EMG_CASES if c.name == "CQDMEM1-mat1-theta-QDMM1D")
    np.testing.assert_array_equal(
        qdmm1d_properties(_data(mat1), MATS).g, trmem_properties(_data(mat1), MATS).g
    )


# -- physics ------------------------------------------------------------------------


def _rigid_modes(xyz) -> list[np.ndarray]:
    modes = []
    for axis in range(3):
        t = np.zeros(3)
        t[axis] = 1.0
        modes.append(np.tile(t, len(xyz)))
    for axis in range(3):
        w = np.zeros(3)
        w[axis] = 1.0
        modes.append(np.concatenate([np.cross(w, p) for p in xyz]))
    return modes


@pytest.mark.parametrize("tag", ["square", "aniso30", "warp+", "warp-", "general3d", "trap"])
def test_stiffness_symmetric_psd_rigid_body(tag):
    case = next(c for c in EMG_CASES if c.name == f"CQDMEM1-{tag}-QDMM1D")
    data = _data(case)
    out = qdmm1d(data, MATS)
    order = np.argsort(data["sil"], kind="stable")
    xyz = np.asarray(case.grids, float)[order]
    k = out.k()
    scale = np.abs(k).max()
    assert np.abs(k - k.T).max() <= 1e-12 * scale
    assert np.linalg.eigvalsh(k).min() > -1e-9 * scale
    for v in _rigid_modes(xyz)[:3]:
        assert np.abs(k @ v).max() <= 1e-9 * scale
    in_plane = _rigid_modes(xyz)[3:]
    if out.planar and abs(xyz[:, 2]).max() == 0.0:
        assert np.abs(k @ in_plane[2]).max() <= 1e-9 * scale * np.abs(xyz).max()
    assert np.linalg.matrix_rank(k, tol=1e-9 * scale) == 5


def test_mass_sums_to_element_mass_and_grid_expansion():
    for case in EMG_CASES:
        data = _data(case)
        out = qdmm1d(data, MATS, CSTM)
        p = qdmm1d_properties(data, MATS)
        mp = mean_plane(np.asarray(case.grids, float))
        area = 0.5 * float((mp.x24 * mp.y3a - mp.x13 * mp.y4a)[0])
        total = (p.nsm + p.rho * p.t) * area
        assert np.trace(out.m()) == pytest.approx(3 * total, rel=1e-6), case.name
        kg, mg = out.k_grid(), out.m_grid()
        assert kg.shape == mg.shape == (24, 24)
        np.testing.assert_array_equal(kg[np.ix_([0, 1, 2], [6, 7, 8])], out.k()[0:3, 3:6])
        assert not kg[3:6].any() and not mg[:, 3:6].any()


@pytest.mark.parametrize("tag", ["square", "trap", "x24zero", "x13zero", "a2zero", "a3", "a3zero"])
def test_constant_strain_patch_stress(tag):
    """A uniform strain in the mean-plane axes recovers G * strain at (ETAS, XIS)."""
    case = next(c for c in SDR_CASES if c.name == f"CQDMEM1-{tag}-SQDM")
    data = _data(case)
    eps = np.array([1.0e-4, -4.0e-5, 6.0e-5])
    x = np.asarray([g.xyz for g in data.grids], float)
    mp = mean_plane(x, x14_from_x41=True)
    e = mp.e[0]
    grad = np.array([[eps[0], eps[2] / 2, 0.0], [eps[2] / 2, eps[1], 0.0], [0.0, 0.0, 0.0]])
    u = np.zeros(QC.NPG)
    for i in range(4):
        local = e.T @ (x[i] - x[0])
        u[data["sil"][i] - 1 : data["sil"][i] + 2] = e @ (grad @ local)
    p1 = sqdm11(data, MATS, CSTM, real_words=False)
    st = sqdm12(p1, u, real_words=False)
    g = trmem_properties(data, MATS, real_words=False).g
    np.testing.assert_allclose([st.sx, st.sy, st.sxy], g @ eps, rtol=1e-9, atol=1e-9 * 1e3)
    sx, sy, sxy = g @ eps
    tau = np.hypot((sx - sy) / 2, sxy)
    assert st.max_shear == pytest.approx(tau, rel=1e-9)
    assert st.major == pytest.approx((sx + sy) / 2 + tau, rel=1e-9)
    assert st.angle == pytest.approx(np.degrees(np.arctan2(2 * sxy, sx - sy)) / 2, abs=1e-6)


@pytest.mark.parametrize("tag", ["square", "a2zero", "a3"])
def test_free_thermal_expansion(tag):
    """Isotropic free expansion: stress free, and K u = P on a parallelogram."""
    grids, _, t, theta, nsm, sils, _ = QC.QDMEM1_GEOMS[tag]
    mats = MaterialTable([Mat1(9, 1.0e7, 1.0e7 / 2.6, 0.3, 0.0, 1.5e-5, 50.0)])
    est = QC.qdmem1_cstm_est(1, grids, 9, t, theta, nsm, sils)
    data = LAYOUTS["QDMEM1"].parse(est)
    tbar = 250.0
    x = np.asarray([p for _, p in grids], float)
    strain = 1.5e-5 * (tbar - 50.0)
    u = np.zeros(QC.NPG)
    for i, s in enumerate(sils):
        u[s - 1 : s + 2] = strain * (x[i] - x[0])
    st = sqdm12(sqdm11(data, mats, real_words=False), u, tbar, real_words=False)
    assert max(abs(st.sx), abs(st.sy), abs(st.sxy)) <= 1e-8 * 1.0e7 * strain
    if tag == "square":
        out = qdmm1d(data, mats, real_words=False)
        load = qdmm1(data, mats, tbar, real_words=False).add_to(np.zeros(QC.NPG))
        order = np.argsort(sils)
        idx = np.array([sils[i] - 1 + k for i in order for k in range(3)])
        np.testing.assert_allclose(out.k() @ u[idx], load[idx], rtol=0, atol=1e-9 * abs(load).max())
        assert np.abs(load.reshape(-1, 3).sum(axis=0)).max() <= 1e-9 * abs(load).max()


def test_geometry_errors():
    sq = [(0, p) for p in QC._SQ]
    bad = {
        "collinear": [(0, (0, 0, 0)), (0, (1, 0, 0)), (0, (2, 0, 0)), (0, (3, 0, 0))],
        "reentrant": [(0, (0, 0, 0)), (0, (4, 0, 0)), (0, (1, 1, 0)), (0, (0, 4, 0))],
        "bowtie": [sq[0], sq[1], sq[3], sq[2]],
    }
    for tag, grids in bad.items():
        data = LAYOUTS["QDMEM1"].parse(QC.qdmem1_cstm_est(5, grids, 1, 0.1, 0.0, 0.0, C.SILS))
        with pytest.raises(QuadGeometryError):
            qdmm1d(data, MATS)
        if tag == "collinear":
            with pytest.raises(QuadGeometryError):
                sqdm11(data, MATS)
            with pytest.raises(QuadGeometryError):
                qdmm1(data, MATS, 100.0)


def test_empty_batch_and_property_count():
    out = qdmem1_matrices([], MATS)
    assert out.k.shape == (0, 12, 12) and len(out) == 0
    assert qdmem1_thermal_loads([], MATS, 1.0) == []
    data = _data(EMG_CASES[0])
    with pytest.raises(ValueError):
        qdmem1_matrices([data, data], [qdmm1d_properties(data, MATS)])


def test_float64_path_close_to_real_emulation():
    for case in EMG_CASES:
        data = _data(case)
        a = qdmm1d(data, MATS, CSTM)
        b = qdmm1d(data, MATS, CSTM, real_words=False)
        # float64 also evaluates the 4 pi^2 material angle unrounded: ~1e-4 on MAT2.
        assert _err(b.k(), a.k())[1] < 1e-3, case.name
    for case in SDR_CASES:
        data = _data(case)
        a = sqdm12(sqdm11(data, MATS, CSTM), case.extra[9:], _temperature(case))
        b = sqdm12(
            sqdm11(data, MATS, CSTM, real_words=False),
            case.extra[9:],
            _temperature(case),
            real_words=False,
        )
        assert _err(b.words()[[0, 1, 2, 4, 5, 6]], a.words()[[0, 1, 2, 4, 5, 6]])[1] < 1e-3
    for case in LOAD_CASES:
        data = _data(case)
        a = qdmm1(data, MATS, case.extra[0], CSTM).loads
        b = qdmm1(data, MATS, case.extra[0], CSTM, real_words=False).loads
        assert _err(b, a)[1] < 1e-5, case.name


# -- bulk data: MATT1 / TABLEM ------------------------------------------------------


def test_bulk_reader_reads_matt1_and_tablem():
    text = (repo_root() / "inp" / "d01032a.inp").read_text(errors="replace")
    model, _ = read_ecpts(text)
    assert "MATT1" not in model.ignored and "TABLEM1" not in model.ignored
    e = model.materials.mat1(75, 238.75).e
    assert e == pytest.approx(9.84e6 + (9.51e6 - 9.84e6) * 38.75 / 50.0)
    assert model.materials.mat1(75).e == 10.4e6
    deck = "\n".join(
        [
            "ID T,T",
            "CEND",
            "BEGIN BULK",
            "MAT1    1       1.0+7           .3                                      +M1",
            "+M1     1.0+4   2.0+4   3.0+4",
            "MATT1   1       11      12      13      14      15              16      +MT",
            "+MT     17      18      19",
            "TABLEM1 11                                                              +T1",
            "+T1     0.      1.      100.    2.      ENDT",
            "TABLEM2 12      10.                                                     +T2",
            "+T2     0.      1.      100.    3.      ENDT",
            "TABLEM3 13      10.     20.                                             +T3",
            "+T3     0.      1.      10.     2.      ENDT",
            "TABLEM4 14      0.      1.      -10.    10.                             +T4",
            "+T4     1.      .5      ENDT",
            "TABLEM1 15                                                              +T5",
            "+T5     0.      1.      SKIP    SKIP    100.    2.      ENDT",
            "TABLEM1 16                                                              +T6",
            "+T6     0.      1.      100.    2.      ENDT",
            "TABLEM1 17                                                              +T7",
            "+T7     0.      1.      100.    2.      ENDT",
            "TABLEM1 18                                                              +T8",
            "+T8     0.      1.      100.    2.      ENDT",
            "TABLEM1 19                                                              +T9",
            "+T9     0.      1.      100.    2.      ENDT",
            "ENDDATA",
        ]
    )
    model = BulkModel.from_text(deck)
    mats = model.materials
    t = mats._tables
    assert isinstance(t[11], TableM1) and isinstance(t[12], TableM2)
    assert isinstance(t[13], TableM3) and isinstance(t[14], TableM4)
    assert t[15].x == (0.0, 100.0)
    matt1 = mats._matt1[1]
    assert (matt1.e, matt1.g, matt1.nu, matt1.rho, matt1.alpha) == (11, 12, 13, 14, 15)
    assert (matt1.ge, matt1.st, matt1.sc, matt1.ss) == (16, 17, 18, 19)
    m = mats.mat1(1, 50.0)
    assert m.e == pytest.approx(1.5)
    assert m.sigt == pytest.approx(1.5) and m.sigs == pytest.approx(1.5)


# -- listing parity -----------------------------------------------------------------


def _subset(table: GoldenTable, keep: list[int]) -> GoldenTable:
    return GoldenTable(table.key, table.columns, table.ids[keep], table.values[keep], table.meta)


def _solve_static(model, datas, temps) -> np.ndarray:
    """Assemble the CQDMEM1 K, the QDMM1 thermal loads and the SPCs, and solve K u = P."""
    n = max(model.sil.values()) + 5
    k = np.zeros((n, n))
    p = np.zeros(n)
    mats = qdmem1_matrices(datas, model.materials, model.cstm)
    for i in range(len(mats)):
        idx = np.array([s - 1 + c for s in mats.sils[i] for c in range(3)])
        k[np.ix_(idx, idx)] += mats.k[i]
    for load in qdmem1_thermal_loads(datas, model.materials, temps, model.cstm):
        load.add_to(p)
    fixed = {model.sil[g] - 1 + int(c) - 1 for g, gp in model.grids.items() for c in gp.ps}
    fixed |= {model.sil[s.g] - 1 + int(c) - 1 for s in model.spcs if s.sid == 1 for c in s.c}
    free = np.array(sorted(set(range(n)) - fixed))
    u = np.zeros(n)
    u[free] = np.linalg.solve(k[np.ix_(free, free)], p[free])
    return u, p


def _grid_rows(model, vec, ids) -> np.ndarray:
    return np.array([vec[model.sil[int(g)] - 1 : model.sil[int(g)] + 5] for g in ids])


def test_stress_tables_match_listings_from_printed_displacements():
    """d01012a and d01013a print every CQDMEM1 grid: feed the golden displacements."""
    for deck, n_tables in (("d01012a", 2), ("d01013a", 1)):
        text = (repo_root() / "inp" / f"{deck}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        records = {r.data.eid: r for r in ecpts["QDMEM1"]}
        golden = tables(load_golden(deck))
        cmps = []
        for key, table in golden.items():
            if not key.startswith("stress/CQDMEM1/"):
                continue
            _, _, subcase, occ = key.split("/")[:4]
            printed = golden[f"displacement/grid/{subcase}/{occ}"].by_id()
            assert all(g in printed for e in table.ids for g in records[int(e)].grid_ids)
            u = np.zeros(max(model.sil.values()) + 6)
            for gid, row in printed.items():
                u[model.sil[gid] - 1 : model.sil[gid] + 5] = np.nan_to_num(row)
            datas = [records[int(e)].data for e in table.ids]
            actual = qdmem1_stress_table(qdmem1_stresses(datas, model.materials, u, model.cstm))
            cmp = compare(table, actual)
            assert cmp.passed if hasattr(cmp, "passed") else cmp.failures == 0, (deck, key, cmp)
            cmps.append(cmp)
        assert len(cmps) == n_tables
        record_parity(
            f"CQDMEM1 stress {deck} vs listing",
            sum(c.n for c in cmps),
            max(c.max_abs for c in cmps),
            max(c.max_rel for c in cmps),
        )


def test_d01032a_thermal_matt1_matches_listing():
    """d01032a: QDMM1 loads vs OLOAD, then a static solve vs displacements and stresses.

    Its DISPLACEMENTS set covers four grid rows only, so no element has all four
    grids printed; the displacements come from assembling the ported K and
    thermal loads (E from MATT1/TABLEM1 at each element's mean TEMP) instead.
    """
    text = (repo_root() / "inp" / "d01032a.inp").read_text(errors="replace")
    model, ecpts = read_ecpts(text)
    datas = [r.data for r in ecpts["QDMEM1"]]
    assert len(datas) == 216 and len(model.temperatures[1]) == 247
    temps = [d.temperature for d in datas]
    assert min(temps) < 150.0 < 200.0 < max(temps)  # spans several TABLEM1 intervals
    golden = tables(load_golden("d01032a"))
    u, p = _solve_static(model, datas, temps)
    results = {}
    for kind in ("oload", "displacement"):
        table = golden[f"{kind}/grid/sc1/occ1"]
        results[kind] = compare(table, _grid_rows(model, p if kind == "oload" else u, table.ids))
    table = golden["stress/CQDMEM1/sc1/occ1"]
    by_eid = {d.eid: d for d in datas}
    sel = [by_eid[int(e)] for e in table.ids]
    stresses = qdmem1_stresses(sel, model.materials, u, model.cstm, [d.temperature for d in sel])
    results["stress"] = compare(table, qdmem1_stress_table(stresses))
    for kind, cmp in results.items():
        assert cmp.failures == 0 and cmp.n > 0, (kind, cmp)
        record_parity(f"CQDMEM1 d01032a {kind} vs listing", cmp.n, cmp.max_abs, cmp.max_rel)
    # Without MATT1 (constant E) the thermal loads miss the listing by ~8%.
    m1 = model.materials._mat1[75]
    flat = MaterialTable([m1])
    loads = qdmem1_thermal_loads(datas, flat, temps, model.cstm)
    pf = np.zeros_like(p)
    for load in loads:
        load.add_to(pf)
    assert (
        compare(
            golden["oload/grid/sc1/occ1"], _grid_rows(model, pf, golden["oload/grid/sc1/occ1"].ids)
        ).failures
        > 0
    )


def test_every_deck_qdmem1_through_kernel():
    seen: dict[str, int] = {}
    worst = _Worst()
    for deck_id in sorted(RF1_DECKS):
        text = (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        datas = [r.data for r in ecpts.get("QDMEM1", [])]
        if not datas:
            continue
        batch = qdmem1_matrices(datas, model.materials, model.cstm)
        for i, d in enumerate(datas):
            ref = qdmm1d(d, model.materials, model.cstm)
            a, r = _err(batch.k[i], ref.k())
            assert r <= 1e-15, (deck_id, d.eid)
            k = batch.k[i]
            assert np.allclose(k, k.T, rtol=0, atol=1e-12 * np.abs(k).max())
            assert np.linalg.matrix_rank(k, tol=1e-9 * np.abs(k).max()) == 5
            worst.add(a, r, k.size)
        seen[deck_id] = len(datas)
    assert set(seen) == {"d01012a", "d01013a", "d01032a"}, seen
    record_parity(
        f"qdmem1 deck smoke ({sum(seen.values())} CQDMEM1): batch vs reference",
        worst.n,
        worst.abs,
        worst.rel,
    )


def test_properties_replace_keeps_parity():
    case = EMG_CASES[0]
    data = _data(case)
    p = qdmm1d_properties(data, MATS)
    assert qdmm1d(data, dataclasses.replace(p)).k().tolist() == qdmm1d(data, MATS).k().tolist()
