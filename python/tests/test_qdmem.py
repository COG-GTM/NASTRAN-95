"""CQDMEM four-triangle quadrilateral (``nastran95py.elements.qdmem``).

Parity: stiffness (``KQDMEM``), lumped mass (``MASSTQ (1)``), thermal load
(``QDMEM``) and stress recovery (``SQDME1``/``STQME2 (2)``), per-element and
batched, against the Fortran dumps (basic-system cases of ``harness_golden.json``,
coordinate-system, warped, material-angle, thermal and zero-stress cases of
``qdmem_golden.json``, and the live harnesses when gfortran is available) at
relative 1e-12 (the principal angle to 2 ulp of ``atan2f``); rigid-body modes,
the thermal load against K times the free expansion, the constant-strain patch
and the mass in closed form; and the "STRESSES IN QUADRILATERAL MEMBRANES"
tables of d01011a and d01161a fed the golden displacements, at the golden README
tolerance.
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
    QdmemBatch,
    QdmemGeometryError,
    TrmemStressBatch,
    kqdmem_angles,
    mean_plane_angles,
    mean_plane_directions,
    qdmem,
    qdmem_load,
    qdmem_matrices,
    qdmem_phase1,
    qdmem_stress_matrices,
    qdmem_stresses,
    qdmem_thermal_loads,
    sqdme1,
    stqme2_qdmem,
)
from nastran95py.materials import Mat1, Mat2, MaterialTable
from nastran95py.reference.golden import compare, load_golden, tables, tolerances
from nastran95py.reference.inventory import RF1_DECKS, load_manifest, repo_root
from tests.conftest import record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import qdmem_cases as QC
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(m, *v) for m, v in C.MAT1] + [Mat2(m, *v) for m, v in C.MAT2])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)


def _orthonormal(t: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(t)
    return u @ vt


CSTM_EXACT = {
    cid: CoordSystem(cid, cs.ctype, cs.origin, _orthonormal(cs.t)) for cid, cs in CSTM.items()
}
BASIC = [c for c in C.ALL_CASES if c.element == "CQDMEM" and c.routine in ("KQDMEM", "SQDME1")]
STIFF = [c for c in BASIC if c.routine == "KQDMEM"] + [
    c for c in QC.HARNESS_CASES if c.routine == "KQDMEM"
]
STRESS = [c for c in BASIC if c.routine == "SQDME1"] + [
    c for c in QC.HARNESS_CASES if c.routine == "SQDME1"
]
MASS = [c for c in QC.AUX_CASES if c.routine == "MASSTQ"]
LOADS = [c for c in QC.AUX_CASES if c.routine == "QDMEM"]
LISTING_DECKS = {"d01011a", "d01161a"}
ANGLE, ANGLE_ULPS = 4, 2


def _f32(v) -> np.ndarray:
    return np.asarray(v, float).astype(np.float32).astype(float)


def _golden() -> dict[str, dict]:
    """Every case as ``{"parts": {(piv, j): 36 words}, "phase1"/"stress"/"load": words}``."""
    out: dict[str, dict] = {}
    basic = json.loads(R.GOLDEN.read_text())
    for c in BASIC:
        e = basic[c.name]
        out[c.name] = {k.lower(): _f32(v) for k, v in e.get("words", {}).items()}
        out[c.name]["parts"] = e.get("parts", {})
    for name, e in json.loads(QC.GOLDEN.read_text()).items():
        out[name] = {k: (v if k == "parts" else np.asarray(v)) for k, v in e.items()}
    for e in out.values():
        parts = e.get("parts", {})
        e["parts"] = {tuple(map(int, k.split(":"))): np.asarray(v) for k, v in parts.items()}
    return out


def _live() -> dict[str, dict]:
    R.build()
    results = R.run([c for c in BASIC], cstm=[]) + QC.run()
    gold = QC.to_golden(results)
    for e in gold.values():
        parts = e.get("parts", {})
        e["parts"] = {tuple(map(int, k.split(":"))): np.asarray(v) for k, v in parts.items()}
        for k in ("phase1", "stress", "load"):
            if k in e:
                e[k] = np.asarray(e[k])
    return gold


def _need_fortran():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case):
    return LAYOUTS["QDMEM"].parse(case.words)


class _Worst:
    def __init__(self) -> None:
        self.abs, self.rel, self.n = 0.0, 0.0, 0

    def add(self, a: float, r: float, n: int) -> None:
        self.abs, self.rel, self.n = max(self.abs, a), max(self.rel, r), self.n + n


def _assembled(parts: dict, sils) -> np.ndarray:
    """SMA1B/SMA2B partitions (pivot, column SIL) as the 12x12 translation matrix, ascending SIL."""
    order = sorted(sils)
    out = np.zeros((12, 12))
    for (piv, j), v in parts.items():
        r, c = order.index(piv), order.index(j)
        out[3 * r : 3 * r + 3, 3 * c : 3 * c + 3] = v.reshape(6, 6)[:3, :3]
    return out


def _compare_matrices(fortran: dict, label: str) -> None:
    for cases, key in ((STIFF, "k"), (MASS, "m")):
        datas = [_data(c) for c in cases]
        batch = qdmem_matrices(QdmemBatch.from_elements(datas), MATS, CSTM)
        worst = {"element": _Worst(), "batch": _Worst()}
        for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
            ref = _assembled(fortran[case.name]["parts"], data["sil"])
            assert len(fortran[case.name]["parts"]) == (16 if key == "k" else 4), case.name
            out = qdmem(data, MATS, CSTM)
            assert out.sils == tuple(sorted(data["sil"]))
            for path, mine in (("element", getattr(out, key)()), ("batch", batch.__dict__[key][i])):
                a, r = _err(mine, ref)
                assert r <= RTOL, (case.name, path, r)
                worst[path].add(a, r, ref.size)
        name = "KQDMEM K" if key == "k" else "MASSTQ M"
        for path, w in worst.items():
            record_parity(f"{name} {path} vs {label}", w.n, w.abs, w.rel)


def _compare_loads(fortran: dict, label: str) -> None:
    datas = [_data(c) for c in LOADS]
    tbar = [c.extra[0] for c in LOADS]
    batch = qdmem_thermal_loads(QdmemBatch.from_elements(datas), MATS, tbar, CSTM)
    worst = {"element": _Worst(), "batch": _Worst()}
    for i, (case, data) in enumerate(zip(LOADS, datas, strict=True)):
        want = _f32(fortran[case.name]["load"])
        assert np.abs(want).max() > 0.0, case.name
        ref = qdmem_load(data, MATS, case.extra[0], CSTM).add_to(np.zeros(QC.NPG))
        got = batch[i].add_to(np.zeros(QC.NPG))
        for path, vec in (("element", ref), ("batch", got)):
            a, r = _err(vec, want)
            assert r <= RTOL, (case.name, path, r)
            worst[path].add(a, r, want.size)
    for path, w in worst.items():
        record_parity(f"QDMEM thermal load {path} vs {label}", w.n, w.abs, w.rel)


def _load_args(case: C.Case):
    _, _, _, _, _, ldtemp, temp, _eldefm, ndisp, *disp = case.extra
    assert len(disp) == ndisp
    return disp, (None if ldtemp == -1 else temp)


def _angle_ok(got: float, exp: float) -> bool:
    """REAL ``ATAN2`` is libm's ``atan2f``, reproducible only to 1 ulp across libm versions."""
    return abs(got - exp) <= ANGLE_ULPS * float(np.spacing(np.float32(abs(exp))))


def _compare_stress(fortran: dict, label: str) -> None:
    datas = [_data(c) for c in STRESS]
    batch = qdmem_stress_matrices(datas, MATS, CSTM)
    worst = {k: _Worst() for k in ("phase1", "stress", "angle")}
    for i, (case, data) in enumerate(zip(STRESS, datas, strict=True)):
        disp, temp = _load_args(case)
        p1 = sqdme1(data, MATS, CSTM)
        one = TrmemStressBatch(
            **{f: getattr(batch, f)[i : i + 1] for f in batch.__dataclass_fields__}
        )
        paths = {
            "element": (p1.words(), stqme2_qdmem(p1, disp, temp)),
            "batch": (
                qdmem_phase1(batch, i).words(),
                qdmem_stresses(one, disp, None if temp is None else [temp]).stress_words(0),
            ),
        }
        exp1, exps = fortran[case.name]["phase1"], fortran[case.name]["stress"]
        for path, (w1, ws) in paths.items():
            for block, got, exp in (
                ("phase1", w1, exp1),
                ("stress", np.delete(ws, ANGLE), np.delete(exps, ANGLE)),
            ):
                np.testing.assert_allclose(got, exp, rtol=RTOL, atol=0.0, err_msg=f"{case.name}")
                worst[block].add(*_err(got, exp), got.size)
            assert _angle_ok(ws[ANGLE], exps[ANGLE]), (case.name, path, ws[ANGLE], exps[ANGLE])
            worst["angle"].add(*_err(ws[ANGLE : ANGLE + 1], exps[ANGLE : ANGLE + 1]), 1)
    for block, w in worst.items():
        record_parity(f"SQDME1/STQME2 {block} vs {label}", w.n, w.abs, w.rel)


def test_qdmem_matches_fortran_golden():
    gold = _golden()
    assert {c.name for c in QC.QDMEM_CASES} <= set(gold)
    assert len(STIFF) == 3 + len(QC.QDMEM_GEOMS) and len(STRESS) == 3 + 4 * len(QC.QDMEM_GEOMS)
    _compare_matrices(gold, "golden")
    _compare_loads(gold, "golden")
    _compare_stress(gold, "golden")


@pytest.mark.fortran
def test_qdmem_matches_live_harness():
    _need_fortran()
    live = _live()
    _compare_matrices(live, "live harness")
    _compare_loads(live, "live harness")
    _compare_stress(live, "live harness")
    # The checked-in golden file vs this machine's gfortran: exact but for the angle.
    gold = json.loads(QC.GOLDEN.read_text())
    errs = []
    for name, entry in gold.items():
        for k, v in entry.items():
            if k == "parts":
                for p, w in v.items():
                    errs.append(_err(live[name]["parts"][tuple(map(int, p.split(":")))], w))
            elif k in ("phase1", "load") or (k == "stress" and len(v)):
                a, b = np.asarray(live[name][k]), np.asarray(v)
                if k == "stress":
                    a, b = np.delete(a, ANGLE), np.delete(b, ANGLE)
                if b.size:
                    errs.append(_err(a, b))
    a, r = max(e[0] for e in errs), max(e[1] for e in errs)
    record_parity("qdmem_golden.json vs live harness", len(errs), a, r)
    assert r <= 1e-9  # newer gfortran/libm on CI: see the harness golden tolerance


def test_harness_cases_cover_thermal_and_zero_branches():
    gold = _golden()
    for case in STRESS:
        stress, p1 = gold[case.name]["stress"], gold[case.name]["phase1"]
        if case.name.endswith("-zero"):
            assert not stress[1:].any(), case.name
        elif case.name.endswith("-thermal-only"):
            want = -np.float32(p1[6:9]) * (np.float32(case.meta["temperature"]) - np.float32(p1[5]))
            np.testing.assert_array_equal(stress[1:4], want.astype(float))
            assert np.abs(stress[1:4]).max() > 0
        else:
            assert np.abs(stress[1:]).max() > 0, case.name


# -- closed-form and consistency checks -------------------------------------------


def _quad(xyz, mid=3, theta=30.0, cids=(0, 0, 0, 0), sils=(1, 7, 13, 19), t=0.1, nsm=0.0):
    words = QC.qdmem_cstm_est(31, list(zip(cids, xyz, strict=True)), mid, t, theta, nsm, sils)
    return LAYOUTS["QDMEM"].parse(words)


PLANE = ((0.0, 0.0, 0.0), (4.0, 0.5, 0.0), (4.5, 3.0, 0.0), (0.5, 2.5, 0.0))


def _rotated(xyz, cids):
    """``PLANE`` tilted out of the basic xy-plane, with grids in the given systems."""
    rot = RC._rot(25.0, -40.0, 15.0)
    return tuple(tuple(rot @ np.asarray(p) + (1.0, -2.0, 0.5)) for p in xyz), cids


def _global(data, vec_basic, cstm):
    """Basic grid vectors (connection order) -> global words in ascending-SIL order."""
    order = np.argsort(data["sil"], kind="stable")
    out = []
    for gi in order:
        gp = data.grids[gi]
        v = np.asarray(vec_basic[gi], float)
        if gp.cid:
            v = transforms_to_basic(cstm, gp.cid, gp.xyz)[0].T @ v
        out.append(v)
    return np.concatenate(out)


@pytest.mark.parametrize("cids", [(0, 0, 0, 0), (10, 20, 30, 0), (41, 42, 40, 21)])
@pytest.mark.parametrize("mid,theta", [(1, 0.0), (3, 30.0), (3, -70.0)])
def test_rigid_modes_and_thermal_load_equals_k_times_free_expansion(cids, mid, theta):
    xyz, cids = _rotated(PLANE, cids)
    d = _quad(xyz, mid, theta, cids)
    k = qdmem(d, MATS, CSTM_EXACT, real_words=False).k()
    x = np.asarray(xyz)
    assert np.allclose(k, k.T, rtol=0, atol=1e-12 * np.abs(k).max())
    # rigid translations and rotations (basic) are stress free
    for v in [*np.eye(3), *(np.cross(w, x - x[0]) for w in np.eye(3))]:
        field = np.broadcast_to(v, (4, 3)) if v.ndim == 1 else v
        u = _global(d, field, CSTM_EXACT)
        assert np.abs(k @ u).max() < 1e-12 * np.abs(k).max() * np.abs(x).max()
    assert np.linalg.matrix_rank(k, tol=1e-10 * np.abs(k).max()) == 5  # planar: 8 - 3 in-plane
    # thermal load = K u_free: the free thermal strain of the material, in the element frame
    tbar = 140.0
    load = qdmem_load(d, MATS, tbar, CSTM_EXACT, real_words=False)
    sin1, cos1 = kqdmem_angles(x, theta, real_words=False)
    m = MATS.mat2(mid, float(sin1[0, 0]), float(cos1[0, 0]))
    eps = m.alpha * (tbar - m.tref)
    i = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
    n = np.cross(i, x[3] - x[0])
    n /= np.linalg.norm(n)
    e = np.stack([i, np.cross(n, i)], axis=1)
    strain = np.array([[eps[0], eps[2] / 2], [eps[2] / 2, eps[1]]])
    u_free = _global(d, [e @ (strain @ (e.T @ (p - x[0]))) for p in x], CSTM_EXACT)
    order = np.argsort(d["sil"], kind="stable")
    p = np.concatenate([load.loads[gi] for gi in order])
    a, r = _err(k @ u_free, p)
    assert r < 1e-12, r
    # ... and the stresses of that free expansion vanish under the same load temperature
    u = np.zeros(24)
    for s, gi in zip(sorted(d["sil"]), order, strict=True):
        u[s - 1 : s + 2] = u_free[3 * list(order).index(gi) : 3 * list(order).index(gi) + 3]
    res = qdmem_stresses(qdmem_stress_matrices([d], MATS, CSTM_EXACT, False), u, tbar, False)
    sig = res.stress_table()[0, :3]
    scale = np.abs(m.g).max() * np.abs(eps).max()
    assert np.abs(sig).max() < 1e-10 * scale


@pytest.mark.parametrize("mid,theta", [(1, 0.0), (2, 45.0), (3, 30.0), (3, 120.0)])
def test_constant_strain_patch(mid, theta):
    """A uniform in-plane strain gives the same stress in all four sub-triangles: G eps."""
    xyz, cids = _rotated(PLANE, (10, 0, 30, 41))
    d = _quad(xyz, mid, theta, cids)
    x = np.asarray(xyz)
    u_dirs = mean_plane_directions(x, real_words=False)
    assert np.allclose(u_dirs[:, 0], [1.0, 0.0])
    i = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
    n = np.cross(x[2] - x[0], x[3] - x[1])
    n /= np.linalg.norm(n)
    e = np.stack([i, np.cross(n, i)], axis=1)
    eps = np.array([1.0e-4, -4.0e-5, 6.0e-5])
    strain = np.array([[eps[0], eps[2] / 2], [eps[2] / 2, eps[1]]])
    field = [e @ (strain @ (e.T @ (p - x[0]))) for p in x]
    u = np.zeros(24)
    for gi, s in enumerate(d["sil"]):
        gp = d.grids[gi]
        v = field[gi]
        if gp.cid:
            v = transforms_to_basic(CSTM_EXACT, gp.cid, gp.xyz)[0].T @ v
        u[s - 1 : s + 2] = v
    res = qdmem_stresses(qdmem_stress_matrices([d], MATS, CSTM_EXACT, False), u, None, False)
    s = np.sin(np.radians(theta)) if mid == 3 else 0.0
    c = np.cos(np.radians(theta)) if mid == 3 else 1.0
    want = MATS.mat2(mid, s, c).g @ eps
    np.testing.assert_allclose(res.stress_table()[0, :3], want, rtol=1e-10)
    sx, sy, txy, ang, s1, s2, tmax = res.stress_table()[0]
    rad = np.hypot((sx - sy) / 2, txy)
    np.testing.assert_allclose([s1, s2, tmax], [(sx + sy) / 2 + rad, (sx + sy) / 2 - rad, rad])
    np.testing.assert_allclose(ang, np.degrees(np.arctan2(2 * txy, sx - sy)) / 2)


def test_mass_closed_form():
    """``MASSTQ``: parallelogram -> each grid gets a quarter of the panel mass."""
    xyz = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (5.0, 3.0, 0.0), (1.0, 3.0, 0.0))
    d = _quad(xyz, 1, 0.0, t=0.2, nsm=0.01, sils=(19, 1, 13, 7))
    out = qdmem(d, MATS, real_words=False)
    area = 12.0
    each = (MATS.density(1) * 0.2 + 0.01) * area / 4.0
    np.testing.assert_allclose(np.diag(out.m()), each, rtol=1e-14)
    assert out.mass.dict2 == 2 and out.stiffness.ndof == 12
    kg, mg = out.k_grid(), out.m_grid()
    assert kg.shape == mg.shape == (24, 24)
    tr = [6 * g + c for g in range(4) for c in range(3)]
    np.testing.assert_array_equal(kg[np.ix_(tr, tr)], out.k())
    assert out.ge == 0.0


def test_kqdmem_and_mean_plane_angles_agree_on_flat_quads():
    for xyz in (PLANE, _rotated(PLANE, (0,) * 4)[0]):
        u = mean_plane_directions(xyz, real_words=False)
        for theta in (0.0, 30.0, -70.0):
            s, c = kqdmem_angles(xyz, theta, real_words=False)
            sm, cm = mean_plane_angles(u, [theta], real_words=False)
            np.testing.assert_allclose(c, cm, atol=1e-12)
            np.testing.assert_allclose(s, np.where(np.abs(sm) < 1e-6, 0.0, sm), atol=1e-12)


def test_degenerate_quads_are_fatal():
    flat = list(PLANE)
    flat[1] = flat[0]
    line = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    for xyz in (flat, line):
        d = _quad(xyz)
        with pytest.raises(QdmemGeometryError, match="31"):
            qdmem(d, MATS)
        with pytest.raises(QdmemGeometryError):
            sqdme1(d, MATS)
        with pytest.raises(QdmemGeometryError):
            qdmem_load(d, MATS, 100.0)


def test_collinear_subtriangle_is_a_quad_error():
    """Grids 1-3 collinear: the quad axes exist, but sub-triangle II (2, 3, 1) is degenerate."""
    d = _quad([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0), (1.0, 2.0, 0.0)])
    for call in (lambda: qdmem(d, MATS), lambda: sqdme1(d, MATS), lambda: qdmem_load(d, MATS, 1.0)):
        with pytest.raises(QdmemGeometryError, match="31"):
            call()


def test_empty_batch():
    batch = QdmemBatch.from_elements([])
    for real_words in (True, False):
        out = qdmem_matrices(batch, MATS, CSTM, real_words)
        assert out.k.shape == out.m.shape == (0, 12, 12) and out.sils.shape == (0, 4)
        assert qdmem_thermal_loads(batch, MATS, 100.0, CSTM, real_words) == []


# -- RF1 decks and listing parity ---------------------------------------------------


def _deck_qdmems():
    for deck_id in sorted(RF1_DECKS):
        text = (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        if ecpts.get("QDMEM"):
            yield deck_id, model, [r.data for r in ecpts["QDMEM"]]


def test_every_deck_qdmem_batch_matches_element_path():
    seen = {}
    for deck_id, model, datas in _deck_qdmems():
        batch = QdmemBatch.from_elements(datas)
        mats = qdmem_matrices(batch, model.materials, model.cstm)
        tbar = [d.temperature + 100.0 for d in datas]
        loads = qdmem_thermal_loads(batch, model.materials, tbar, model.cstm)
        p1 = qdmem_stress_matrices(batch, model.materials, model.cstm)
        u = np.random.default_rng(len(datas)).uniform(-1e-3, 1e-3, max(model.sil.values()) + 6)
        res = qdmem_stresses(p1, u, 321.0)
        for i, d in enumerate(datas):
            ref = qdmem(d, model.materials, model.cstm)
            np.testing.assert_array_equal(mats.k[i], ref.k())
            np.testing.assert_array_equal(mats.m[i], ref.m())
            k = ref.k()
            assert np.allclose(k, k.T, rtol=0, atol=1e-12 * np.abs(k).max())
            assert np.linalg.eigvalsh(k).min() > -1e-9 * np.abs(k).max()
            ld = qdmem_load(d, model.materials, tbar[i], model.cstm)
            np.testing.assert_array_equal(loads[i].loads, ld.loads)
            assert np.abs(ld.loads.sum(axis=0)).max() <= 1e-4 * np.abs(ld.loads).max()
            one = sqdme1(d, model.materials, model.cstm)
            np.testing.assert_array_equal(qdmem_phase1(p1, i).words(), one.words())
            np.testing.assert_array_equal(res.stress_words(i), stqme2_qdmem(one, u, 321.0))
        seen[deck_id] = len(datas)
    assert {"d01011a", "d01161a"} <= set(seen)


def _deck_parity():
    """Every CQDMEM stress table of every golden deck vs the port fed its displacements."""
    for deck in load_manifest()["decks"]:
        golden = tables(load_golden(deck["id"]))
        qd = {
            k: t
            for k, t in golden.items()
            if t.meta["kind"] == "stress" and t.meta["element_type"] == "CQDMEM"
        }
        if not qd:
            continue
        model, ecpts = read_ecpts((repo_root() / deck["inp"]).read_text(errors="replace"))
        records = {r.data.eid: r for r in ecpts["QDMEM"]}
        for key, table in qd.items():
            _, _, subcase, occ = key.split("/")[:4]
            printed = golden[f"displacement/grid/{subcase}/{occ}"].by_id()
            assert all(g in printed for e in table.ids for g in records[int(e)].grid_ids), key
            u = np.zeros(max(model.sil.values()) + 6)
            for gid, row in printed.items():
                u[model.sil[gid] - 1 : model.sil[gid] + 5] = np.nan_to_num(row)
            batch = qdmem_stress_matrices(
                [records[int(e)].data for e in table.ids], model.materials, model.cstm
            )
            actual = qdmem_stresses(batch, u).stress_table()
            yield deck["id"], key, table, actual, compare(table, actual)


def _worst_row(table, actual) -> str:
    """The entry with the largest relative error above the absolute floor (as ``compare``).

    Marked "(abs floor)" when it passes only because of the floor, not ``rtol``.
    """
    rtol, atol = tolerances(table)
    g = np.nan_to_num(table.values)
    err = np.abs(np.asarray(actual) - g)
    big = np.abs(g) > atol
    rel = np.divide(err, np.abs(g), out=np.zeros_like(err), where=big)
    r, c = np.unravel_index(np.argmax(rel), rel.shape)
    floor = bool(err[r, c] > rtol[c] * abs(g[r, c]))
    tag = " (abs floor)" if floor else ""
    return f"eid {int(table.ids[r])} {table.columns[c]} rel {rel[r, c]:.2e}{tag}"


def test_qdmem_tables_match_listings():
    by_deck: dict[str, list] = {}
    for deck, key, table, actual, cmp in _deck_parity():
        assert cmp.passed, (deck, key, cmp, _worst_row(table, actual))
        by_deck.setdefault(deck, []).append((cmp, _worst_row(table, actual)))
    for deck, items in sorted(by_deck.items()):
        cmps = [c for c, _ in items]
        worst = max(items, key=lambda it: it[0].max_rel)[1]
        record_parity(
            f"CQDMEM stress {deck} vs listing ({len(cmps)} tables; worst {worst})",
            sum(c.n for c in cmps),
            max(c.max_abs for c in cmps),
            max(c.max_rel for c in cmps),
        )
    assert set(by_deck) == LISTING_DECKS
    assert len(by_deck["d01011a"]) == 1 and len(by_deck["d01161a"]) == 3
