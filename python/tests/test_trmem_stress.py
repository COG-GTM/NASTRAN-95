"""CTRMEM stress recovery (``nastran95py.elements.trmem_stress``, port of ``STRME1``/``STQME2``).

Parity: the per-element and batched paths against the Fortran SDR2 words (the
basic-system STRME cases of ``harness_golden.json``, the coordinate-system,
material-angle, thermal and zero-stress cases of ``trmem_stress_golden.json``,
and the live harness when gfortran is available) at relative 1e-12; the
constant-strain patch and free thermal expansion in closed form; and the
"STRESSES IN TRIANGULAR MEMBRANES" tables of the COSMIC listings fed the golden
displacements, at the golden README tolerance.
"""

from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import read_ecpts
from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.elements import (
    LAYOUTS,
    TrmemGeometryError,
    TrmemStressBatch,
    membrane_stress_matrices,
    membrane_stresses,
    real_cstm,
    stqme2,
    strme1,
    transs,
    trmem_properties,
    trmem_stress_matrices,
    trmem_stresses,
)
from nastran95py.materials import Mat1, Mat2, MaterialTable
from nastran95py.reference.golden import compare, load_golden, tables
from nastran95py.reference.inventory import load_manifest, repo_root
from tests.conftest import record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R
from tests.fortran_harness import trmem_stress_cases as SC

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
BASIC_CASES = [c for c in C.ALL_CASES if c.routine == "STRME1"]
CASES = BASIC_CASES + SC.TRMEM_STRESS_CASES
LISTING_DECKS = {"d01011a", "d01012a", "d01013a", "d01014a", "d01161a"}


def _load(case: C.Case) -> tuple[list[float], float | None]:
    """Displacements and load temperature the harness passed to ``STQME2``."""
    _, _, _, _, _, ldtemp, temp, _eldefm, ndisp, *disp = case.extra
    assert len(disp) == ndisp
    return disp, (None if ldtemp == -1 else temp)


def _golden() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    basic = json.loads(R.GOLDEN.read_text())
    for c in BASIC_CASES:
        w = basic[c.name]["words"]
        out[c.name] = {k.lower(): np.asarray(w[k], np.float32).astype(float) for k in w}
    for name, entry in json.loads(SC.GOLDEN.read_text()).items():
        out[name] = {k: np.asarray(entry[k]) for k in ("phase1", "stress")}
    return out


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case):
    return LAYOUTS["TRMEM"].parse(case.words)


def _python_words(cases) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Phase-I and stress words of ``cases`` from the per-element and batched paths."""
    datas = [_data(c) for c in cases]
    batch = trmem_stress_matrices(datas, MATS, CSTM)
    out: dict[str, dict] = {"element": {}, "batch": {}}
    for i, (case, data) in enumerate(zip(cases, datas, strict=True)):
        disp, temp = _load(case)
        p1 = strme1(data, MATS, CSTM)
        out["element"][case.name] = {
            "phase1": p1.words(),
            "stress": stqme2(p1, disp, temp).stress_words(),
        }
        one = _slice(batch, i)
        res = trmem_stresses(one, disp, None if temp is None else [temp])
        out["batch"][case.name] = {"phase1": one.element(0).words(), "stress": res.stress_words(0)}
    return out


def _slice(batch: TrmemStressBatch, i: int) -> TrmemStressBatch:
    return TrmemStressBatch(**{f: getattr(batch, f)[i : i + 1] for f in batch.__dataclass_fields__})


ANGLE = 4
"""Index of the principal angle in the ``STRES`` words."""
ANGLE_ULPS = 2


def _angle_ok(got: float, exp: float) -> bool:
    """REAL ``ATAN2`` is libm's ``atan2f``, reproducible only to 1 ulp across libm versions."""
    return abs(got - exp) <= ANGLE_ULPS * float(np.spacing(np.float32(abs(exp))))


def _split(block: str, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Words compared at ``RTOL`` and the angle words (empty for phase I)."""
    if block != "stress":
        return words, words[:0]
    return np.delete(words, ANGLE), words[ANGLE : ANGLE + 1]


def _compare_words(got: dict, exp: dict, name: str, worst: dict) -> None:
    for block, g in got.items():
        e = exp[block]
        (g1, ga), (e1, ea) = _split(block, np.asarray(g)), _split(block, np.asarray(e))
        np.testing.assert_allclose(g1, e1, rtol=RTOL, atol=0.0, err_msg=f"{name} {block}")
        for x, y in zip(ga, ea, strict=True):
            assert _angle_ok(float(x), float(y)), (name, x, y)
        for key, a, b in ((block, g1, e1), ("angle", ga, ea)):
            if a.size:
                w = worst[key]
                ab, rel = _err(a, b)
                w[0], w[1], w[2] = max(w[0], ab), max(w[1], rel), w[2] + a.size


def _compare(fortran: dict[str, dict[str, np.ndarray]], label: str) -> None:
    for path, words in _python_words(CASES).items():
        worst = {k: [0.0, 0.0, 0] for k in ("phase1", "stress", "angle")}
        for case in CASES:
            _compare_words(words[case.name], fortran[case.name], f"{path} {case.name}", worst)
        for block, (a, r, n) in worst.items():
            record_parity(f"STRME1/STQME2 {block} {label} ({path})", n, a, r)


def test_trmem_stress_matches_fortran_golden():
    golden = _golden()
    assert set(golden) == {c.name for c in CASES}
    assert len(SC.TRMEM_STRESS_CASES) == 4 * 16
    _compare(golden, "golden")


@pytest.mark.fortran
def test_trmem_stress_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    live = {r.case.name: r for r in R.run(BASIC_CASES) + SC.run()}
    fortran = {
        name: {k.lower(): np.asarray(v, np.float32).astype(float) for k, v in r.words.items()}
        for name, r in live.items()
    }
    _compare(fortran, "live")
    # The checked-in golden file vs this machine's gfortran/libm: exact but for the angle.
    current = SC.to_golden([live[c.name] for c in SC.TRMEM_STRESS_CASES])
    golden = json.loads(SC.GOLDEN.read_text())
    assert set(current) == set(golden)
    worst = {k: [0.0, 0.0, 0] for k in ("phase1", "stress", "angle")}
    for name, entry in current.items():
        words = {k: entry[k] for k in ("phase1", "stress")}
        _compare_words(words, golden[name], f"live vs golden {name}", worst)
    for block, (a, r, n) in worst.items():
        record_parity(f"STRME1/STQME2 {block} live vs trmem_stress_golden.json", n, a, r)


def test_harness_cases_cover_thermal_and_zero_branches():
    golden = _golden()
    for case in SC.TRMEM_STRESS_CASES:
        p1 = golden[case.name]["phase1"]
        stress = golden[case.name]["stress"]
        tag = case.name.rsplit("-", 1)[-1]
        if case.name.endswith("-zero"):
            assert not stress[1:].any(), case.name  # ATAN2 bypass: angle 0
        elif case.name.endswith("-thermal-only"):
            # u = 0: sigma = -S_T (T - T_SUB_0)
            want = -np.float32(p1[6:9]) * (np.float32(case.meta["temperature"]) - np.float32(p1[5]))
            np.testing.assert_array_equal(stress[1:4], want.astype(float))
            assert np.abs(stress[1:4]).max() > 0, case.name
        else:
            assert tag in ("disp", "thermal") and np.abs(stress[1:]).max() > 0, case.name


def _tri(
    cids=(0, 0, 0),
    mid=3,
    theta=30.0,
    xyz=((1, -2, 0.5), (5, 1, 2.5), (2, 4, -1)),
    eid=21,
    sils=(1, 7, 13),
):
    words = [eid, *sils, theta, mid, 0.1, 0.0]
    for cid, p in zip(cids, xyz, strict=True):
        words += [cid, *map(float, p)]
    return LAYOUTS["TRMEM"].parse([*words, 0.0])


def _element_axes(xyz):
    x = np.asarray(xyz, float)
    i = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
    k = np.cross(i, x[2] - x[0])
    k /= np.linalg.norm(k)
    return i, np.cross(k, i)


def _global_u(data, field):
    """Global displacements (SIL-indexed, 18 words) of the basic displacement field."""
    u = np.zeros(18)
    for gp, sil in zip(data.grids, data["sil"], strict=True):
        v = field(np.asarray(gp.xyz, float))
        if gp.cid:
            v = transforms_to_basic(CSTM_EXACT, gp.cid, gp.xyz)[0].T @ v
        u[sil - 1 : sil + 2] = v
    return u


@pytest.mark.parametrize("cids", [(0, 0, 0), (10, 0, 20), (30, 41, 42), (21, 21, 40)])
@pytest.mark.parametrize("mid,theta", [(1, 0.0), (3, 30.0), (3, -75.0)])
def test_constant_strain_patch_closed_form(cids, mid, theta):
    data = _tri(cids, mid, theta)
    eps = np.array([[2e-4, 3e-5, 1e-5], [3e-5, -1e-4, -2e-5], [1e-5, -2e-5, 5e-5]])
    rigid = np.array([1e-3, -2e-3, 4e-4])
    u = _global_u(data, lambda p: rigid + eps @ p)
    p1 = strme1(data, MATS, CSTM_EXACT, real_words=False)
    s = stqme2(p1, u, real_words=False)
    i, j = _element_axes([g.xyz for g in data.grids])
    strain = np.array([i @ eps @ i, j @ eps @ j, 2 * i @ eps @ j])
    g = trmem_properties(data, MATS, real_words=False).g
    want = g @ strain
    assert (s.sigma_x, s.sigma_y, s.tau_xy) == pytest.approx(tuple(want), rel=1e-10)
    centre, rad = (want[0] + want[1]) / 2, math.hypot((want[0] - want[1]) / 2, want[2])
    assert s.major == pytest.approx(centre + rad, rel=1e-10)
    assert s.minor == pytest.approx(centre - rad, rel=1e-10)
    assert s.max_shear == pytest.approx(rad, rel=1e-10)
    angle = math.degrees(0.5 * math.atan2(2 * want[2], want[0] - want[1]))
    assert s.angle == pytest.approx(angle, rel=1e-8)
    # A rigid rotation is stress free.
    w = np.array([[0.0, -3e-4, 2e-4], [3e-4, 0.0, -1e-4], [-2e-4, 1e-4, 0.0]])
    r = stqme2(p1, _global_u(data, lambda p: w @ p), real_words=False)
    assert max(abs(r.sigma_x), abs(r.sigma_y), abs(r.tau_xy)) < 1e-9 * abs(want).max()


@pytest.mark.parametrize("cids", [(0, 0, 0), (10, 30, 41)])
def test_free_thermal_expansion_is_stress_free(cids):
    data = _tri(cids, mid=1, theta=0.0)
    mat = MATS.mat1(1)
    dt_ = 150.0
    u = _global_u(data, lambda p: mat.alpha * dt_ * p)
    p1 = strme1(data, MATS, CSTM_EXACT, real_words=False)
    e11 = mat.e / (1 - mat.nu**2)
    assert p1.st == pytest.approx([e11 * (1 + mat.nu) * mat.alpha] * 2 + [0.0])
    assert p1.tsub0 == mat.tref
    free = stqme2(p1, u, mat.tref + dt_, real_words=False)
    scale = e11 * (1 + mat.nu) * mat.alpha * dt_
    assert max(abs(free.sigma_x), abs(free.sigma_y), abs(free.tau_xy)) < 1e-9 * scale
    held = stqme2(p1, np.zeros(18), mat.tref + dt_, real_words=False)
    assert (held.sigma_x, held.sigma_y, held.tau_xy) == pytest.approx((-scale, -scale, 0.0))
    assert held.angle == 0.0 and held.max_shear == 0.0
    assert stqme2(p1, np.zeros(18), None).stress_words()[1:].tolist() == [0.0] * 7


def test_temperatures_by_eid_and_quad_grids():
    datas = [_tri(), _tri(mid=1, theta=0.0, eid=22, sils=(13, 1, 7))]
    batch = trmem_stress_matrices(datas, MATS)
    u = np.linspace(-1e-3, 1e-3, 18)
    res = trmem_stresses(batch, u, {22: 123.0})
    ref = [stqme2(strme1(datas[0], MATS), u), stqme2(strme1(datas[1], MATS), u, 123.0)]
    np.testing.assert_array_equal(res.stress_words(0), ref[0].stress_words())
    np.testing.assert_array_equal(res.stress_words(1), ref[1].stress_words())
    # NTYPE = 2 (four grids): a grid with S = 0 adds nothing.
    quad = TrmemStressBatch(
        batch.eid,
        np.concatenate([batch.sil, np.full((2, 1), 16)], axis=1),
        batch.tsub0,
        batch.st,
        np.concatenate([batch.s, np.zeros((2, 1, 3, 3))], axis=1),
    )
    np.testing.assert_array_equal(
        membrane_stresses(quad, np.r_[u, np.ones(3)], {22: 123.0}).stress_table(),
        res.stress_table(),
    )


def test_membrane_stress_matrices_entry():
    """``NTYPE = 1`` entry: given G, matches the CTRMEM phase I; TRANSS of rectangular = T."""
    data = _tri((10, 0, 30))
    p1 = strme1(data, MATS, CSTM)
    g = trmem_properties(data, MATS).g
    s = membrane_stress_matrices(
        [[gp.xyz for gp in data.grids]], [[gp.cid for gp in data.grids]], g, real_cstm(CSTM)
    )
    np.testing.assert_array_equal(s[0], p1.s)
    cs = real_cstm(CSTM)[10]
    np.testing.assert_array_equal(transs(cs, np.zeros((2, 3)))[1], cs.t.astype(np.float32))


def test_errors():
    with pytest.raises(TrmemGeometryError, match="30/31"):
        strme1(_tri(xyz=((1, 1, 1), (1, 1, 1), (2, 3, 4))), MATS)
    with pytest.raises(TrmemGeometryError, match="30/32"):
        trmem_stress_matrices([_tri(), _tri(xyz=((0, 0, 0), (1, 1, 1), (3, 3, 3)))], MATS)
    with pytest.raises(KeyError):
        strme1(_tri((99, 0, 0)), MATS, CSTM)


# -- listing parity -----------------------------------------------------------------


def _deck_parity():
    """Every CTRMEM stress table of every golden deck vs the port fed its displacements."""
    for deck in load_manifest()["decks"]:
        golden = tables(load_golden(deck["id"]))
        trmem_tables = {
            k: t
            for k, t in golden.items()
            if t.meta["kind"] == "stress" and t.meta["element_type"] == "CTRMEM"
        }
        if not trmem_tables:
            continue
        text = (repo_root() / deck["inp"]).read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        records = {r.data.eid: r for r in ecpts["TRMEM"]}
        for key, table in trmem_tables.items():
            _, _, subcase, occ = key.split("/")[:4]
            printed = golden[f"displacement/grid/{subcase}/{occ}"].by_id()
            assert all(g in printed for e in table.ids for g in records[int(e)].grid_ids), key
            u = np.zeros(max(model.sil.values()) + 6)
            for gid, row in printed.items():
                u[model.sil[gid] - 1 : model.sil[gid] + 5] = np.nan_to_num(row)
            datas = [records[int(e)].data for e in table.ids]
            batch = trmem_stress_matrices(datas, model.materials, model.cstm)
            actual = trmem_stresses(batch, u).stress_table()
            yield deck["id"], key, compare(table, actual)


def test_trmem_tables_match_listings():
    by_deck: dict[str, list] = {}
    for deck, key, cmp in _deck_parity():
        assert cmp.passed, (deck, key, cmp)
        by_deck.setdefault(deck, []).append(cmp)
    for deck, cmps in sorted(by_deck.items()):
        record_parity(
            f"CTRMEM stress {deck} vs listing ({len(cmps)} tables)",
            sum(c.n for c in cmps),
            max(c.max_abs for c in cmps),
            max(c.max_rel for c in cmps),
        )
    assert set(by_deck) == LISTING_DECKS
    # d01012a has two subcases; d01161a prints three fully stressed design iterations.
    assert {d: len(c) for d, c in by_deck.items()}["d01161a"] == 3
    assert sum(len(c) for c in by_deck.values()) == 8


def test_every_deck_trmem_batch_matches_element_path():
    """Smoke test: batched vs per-element phase I and II for every RF1 deck CTRMEM."""
    total = 0
    for deck in load_manifest()["decks"]:
        text = (repo_root() / deck["inp"]).read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        datas = [r.data for r in ecpts.get("TRMEM", [])]
        if not datas:
            continue
        batch = trmem_stress_matrices(datas, model.materials, model.cstm)
        u = np.random.default_rng(len(datas)).uniform(-1e-3, 1e-3, max(model.sil.values()) + 6)
        res = trmem_stresses(batch, u, 321.0)
        for i, d in enumerate(datas):
            p1 = strme1(d, model.materials, model.cstm)
            np.testing.assert_array_equal(batch.element(i).words(), p1.words())
            np.testing.assert_array_equal(res.stress_words(i), stqme2(p1, u, 321.0).stress_words())
        total += len(datas)
    assert total > 0
