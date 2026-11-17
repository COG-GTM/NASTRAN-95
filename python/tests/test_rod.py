"""CROD/CONROD stiffness and mass (``nastran95py.elements.rod``, port of ``RODD``).

Parity: the reference and vectorised paths against the Fortran ``RODD`` dumps
(basic-system rods in ``harness_golden.json``, offset/rotated/curvilinear
displacement systems in ``rod_golden.json``, and the live harness when gfortran
is available) at relative 1e-12; the closed-form EA/L, GJ/L and rod masses; the
KGGX matrix printed by ``t01241a``; and every CROD/CONROD in the RF1 decks.
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
    ElementData,
    RodBatch,
    RodGeometryError,
    RodProperties,
    assemble,
    rod_matrices,
    rodd,
)
from nastran95py.materials import Mat1, MaterialTable, Matt1, TableM1
from nastran95py.reference.golden import load_golden
from nastran95py.reference.inventory import RF1_DECKS, repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(mid, *vals) for mid, vals in C.MAT1])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
# CORD2-style systems stay unrounded float64; real_words=True rounds them as TRANSD reads REAL.
CSTM.update(RC.FROM_POINTS)


def _orthonormal(t: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(t)
    return u @ vt


# The REAL CSTM words are not exactly orthonormal; closed-form checks need a true rotation.
CSTM_EXACT = {
    cid: CoordSystem(cid, cs.ctype, cs.origin, _orthonormal(cs.t)) for cid, cs in CSTM.items()
}
CASES = [c for c in C.ALL_CASES if c.routine == "RODD"] + RC.ROD_CSTM_CASES


def _golden() -> dict:
    data = json.loads(R.GOLDEN.read_text())
    data.update(json.loads(RC.GOLDEN.read_text()))
    return data


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["ROD"].parse(case.words)


def _compare(cases, dumps: dict, label: str) -> None:
    """Reference and batched Python paths vs Fortran EMGOUT dumps of ``cases``."""
    worst = {"ref": [0.0, 0.0, 0], "batch": [0.0, 0.0, 0]}
    per_file = {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0]}
    for consistent in (False, True):
        group = [c for c in cases if (c.extra[0] > 0) == consistent]
        datas = [_data(c) for c in group]
        batch = rod_matrices(RodBatch.from_elements(datas, MATS), CSTM, consistent)
        for i, (case, data) in enumerate(zip(group, datas, strict=True)):
            out = rodd(data, MATS, CSTM, consistent)
            ref = dumps[case.name]
            assert len(ref) == 2, case.name
            for m, mine, full in zip(
                ref, (out.stiffness, out.mass), (batch.k[i], batch.m[i]), strict=True
            ):
                assert mine is not None, case.name
                assert (mine.ndof, mine.icode, mine.dict2) == (m["ndof"], m["icode"], m["dict2"])
                vals = np.asarray(m["values"])
                ra, rr = _err(mine.values, vals)
                expanded = R._square({**m, "values": vals})
                ba, br = _err(_reduce(full, m["icode"]), expanded)
                assert rr <= RTOL and br <= RTOL, (case.name, m["file"], rr, br)
                for key, (a, r) in (("ref", (ra, rr)), ("batch", (ba, br))):
                    w = worst[key]
                    w[0], w[1], w[2] = max(w[0], a), max(w[1], r), w[2] + vals.size
                    w = per_file[m["file"]]
                    w[0], w[1], w[2] = max(w[0], a), max(w[1], r), w[2] + vals.size
    for key, (a, r, n) in worst.items():
        record_parity(f"rod {key} vs {label}", n, a, r)
    for file, (a, r, n) in per_file.items():
        quantity = "K" if file == 1 else "M"
        record_parity(
            f"rod {quantity} vs {label}", n, a, r,
            element="CROD/CONROD", quantity=quantity, source=harness_source(label),
        )  # fmt: skip


def _reduce(full: np.ndarray, icode: int) -> np.ndarray:
    comps = {7: (0, 1, 2), 56: (3, 4, 5), 63: (0, 1, 2, 3, 4, 5)}[icode]
    idx = [*comps, *(c + 6 for c in comps)]
    return full[np.ix_(idx, idx)]


def _golden_dumps() -> dict:
    gold = _golden()
    return {c.name: gold[c.name]["matrices"] for c in CASES}


def test_rodd_matches_fortran_golden():
    names = {c.name for c in CASES}
    assert {c.name for c in RC.ROD_CSTM_CASES} <= set(json.loads(RC.GOLDEN.read_text()))
    assert len(names) == len(CASES) == 12 + 52
    _compare(CASES, _golden_dumps(), "Fortran RODD golden")


@pytest.mark.fortran
def test_rodd_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    live = {r.case.name: r.matrices for r in R.run(CASES, cstm=RC.CSTM)}
    _compare(CASES, live, "Fortran RODD harness")
    gold = _golden_dumps()
    errs = [
        _err(m["values"], g["values"])
        for name in live
        for m, g in zip(live[name], gold[name], strict=True)
    ]
    a, r = max(e[0] for e in errs), max(e[1] for e in errs)
    record_parity("rod_golden.json vs live RODD", len(errs), a, r)
    assert r <= RTOL


def _blockdiag(t_a, t_b) -> np.ndarray:
    out = np.zeros((12, 12))
    for k, t in enumerate((t_a, t_a, t_b, t_b)):
        out[3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = t
    return out


def _closed_form(props: RodProperties, xa, xb, consistent: bool):
    """Basic-system K and M of a rod from EA/L, GJ/L and (rho*A + nsm)*L."""
    d = np.asarray(xb, float) - np.asarray(xa, float)
    length = np.linalg.norm(d)
    nn = np.outer(d, d) / length**2
    pair = np.array([[1.0, -1.0], [-1.0, 1.0]])
    k = np.zeros((12, 12))
    k[np.ix_(C_T, C_T)] = np.kron(pair, props.e * props.area / length * nn)
    k[np.ix_(C_R, C_R)] = np.kron(pair, props.g * props.j / length * nn)
    mass = (props.rho * props.area + props.nsm) * length
    mm = np.array([[1 / 3, 1 / 6], [1 / 6, 1 / 3]]) if consistent else np.eye(2) / 2
    m = np.zeros((12, 12))
    m[np.ix_(C_T, C_T)] = np.kron(mass * mm, np.eye(3))
    return k, m


C_T = [0, 1, 2, 6, 7, 8]
C_R = [3, 4, 5, 9, 10, 11]


def _check_closed_form(datas, mats, cstm, label) -> None:
    worst = [0.0, 0.0, 0]
    for consistent in (False, True):
        batch = rod_matrices(
            RodBatch.from_elements(datas, mats), cstm, consistent, real_words=False
        )
        for i, d in enumerate(datas):
            ref = rodd(d, mats, cstm, consistent, real_words=False)
            np.testing.assert_allclose(
                batch.k[i], ref.k(), rtol=0, atol=1e-13 * np.abs(ref.k()).max()
            )
            np.testing.assert_allclose(
                batch.m[i], ref.m(), rtol=0, atol=1e-13 * np.abs(ref.m()).max(initial=1.0)
            )
            ga, gb = d.grids if d["sil"][0] < d["sil"][1] else d.grids[::-1]
            t = _blockdiag(*transforms_to_basic(cstm, [ga.cid, gb.cid], [ga.xyz, gb.xyz]))
            props = RodProperties(
                *(getattr(mats.mat1(d["mid"], d.temperature), f) for f in ("e", "g", "rho", "ge")),
                d["area"],
                d["j"],
                d["nsm"],
            )
            k_cf, m_cf = _closed_form(props, ga.xyz, gb.xyz, consistent)
            k_basic, m_basic = t @ batch.k[i] @ t.T, t @ batch.m[i] @ t.T
            assert np.allclose(batch.k[i], batch.k[i].T, rtol=0, atol=1e-14 * np.abs(k_cf).max())
            for got, want in ((k_basic, k_cf), (m_basic, m_cf)):
                if consistent and got is k_basic:
                    continue
                a, r = _err(got, want)
                assert r <= RTOL, (label, d.eid, r)
                worst = [max(worst[0], a), max(worst[1], r), worst[2] + 144]
    record_parity(label, worst[2], worst[0], worst[1])


def test_closed_form_ea_gj_and_mass():
    _check_closed_form(
        [_data(c) for c in CASES if c.extra[0] < 0], MATS, CSTM_EXACT, "rod K,M vs EA/L,GJ/L,mL"
    )


def test_crod_and_conrod_share_one_kernel():
    gold = _golden()
    for c in CASES:
        if c.element == "CROD":
            twin = gold[c.name.replace("CROD", "CONROD", 1)]["matrices"]
            for a, b in zip(gold[c.name]["matrices"], twin, strict=True):
                assert a["values"] == b["values"]
    deck = """ID ROD,TEST
APP DISPLACEMENT
SOL 1,0
CEND
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               3.      4.      12.
CROD    10      5       1       2
PROD    5       6       2.1     .35             .02
CONROD  11      1       2       6       2.1     .35             .02
MAT1    6       1.04+7  4.+6            .1
ENDDATA
"""
    model, ecpts = read_ecpts(deck)
    crod, conrod = (r.data for r in ecpts["ROD"])
    assert (crod.eid, conrod.eid) == (10, 11)
    for consistent in (False, True):
        a = rodd(crod, model.materials, model.cstm, consistent)
        b = rodd(conrod, model.materials, model.cstm, consistent)
        assert a.stiffness.icode == b.stiffness.icode == 63
        k_a, length = a.k(), 13.0
        f = np.float32
        assert np.isclose(np.trace(k_a[:3, :3]), float(f(1.04e7) * f(2.1)) / length, rtol=1e-15)
        assert np.isclose(np.trace(k_a[3:6, 3:6]), float(f(4e6) * f(0.35)) / length, rtol=1e-15)
        mass = float(f(0.1) * f(2.1) + f(0.02)) * length
        assert np.isclose(np.trace(a.m()), (2 if consistent else 3) * mass, rtol=1e-15)
        np.testing.assert_array_equal(k_a, b.k())
        np.testing.assert_array_equal(a.m(), b.m())
        assert a.mass.values.shape == b.mass.values.shape


def test_temperature_dependent_material():
    mats = MaterialTable(
        [Mat1(1, 1.0e7, 4.0e6, 0.25, 0.1)],
        matt1=[Matt1(1, e=7)],
        tables=[TableM1(7, (0.0, 200.0), (1.0, 0.5))],
    )
    case = next(c for c in CASES if c.name == "CROD-skew-RODD-m-1")
    base = _data(case)
    base.fields["mid"] = 1
    hot = ElementData("ROD", base.fields, base.grids, temperature=100.0)
    k_cold = rodd(base, mats).k()
    k_hot = rodd(hot, mats).k()
    np.testing.assert_allclose(k_hot[:3, :3], 0.75 * k_cold[:3, :3], rtol=1e-6)
    np.testing.assert_array_equal(k_hot[3:6, 3:6], k_cold[3:6, 3:6])


def test_illegal_geometry_is_fatal():
    case = CASES[0]
    words = list(case.words)
    words[12:15] = words[8:11]  # grid B on top of grid A
    zero = LAYOUTS["ROD"].parse(words)
    same = _data(case)
    same.fields["sil"] = (1, 1)
    for d in (zero, same):
        with pytest.raises(RodGeometryError):
            rodd(d, MATS)
        with pytest.raises(RodGeometryError, match=str(d.eid)):
            rod_matrices(RodBatch.from_elements([_data(CASES[1]), d], MATS))


def test_no_stiffness_or_mass_when_zero():
    d = _data(CASES[0])
    props = RodProperties(1e7, 4e6, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = rodd(d, props)
    assert out.stiffness is None and out.mass is None
    assert not out.k().any() and not out.m().any()


def test_t01241a_kggx():
    """Assembled CROD stiffness vs the KGGX matrix MATPRN prints for t01241a."""
    text = (repo_root() / "inp" / "t01241a.inp").read_text(errors="replace")
    model, ecpts = read_ecpts(text)
    datas = [r.data for r in ecpts["ROD"]]
    assert len(datas) == 3
    printed = next(m for m in load_golden("t01241a")["matrices"] if m["name"] == "KGGX")
    batch = rod_matrices(RodBatch.from_elements(datas, model.materials), model.cstm)
    kgg = assemble(list(zip(map(tuple, batch.sils), batch.k, strict=True)), printed["rows"])
    want = np.zeros_like(kgg)
    for r, c, v in printed["entries"]:
        want[r - 1, c - 1] = want[c - 1, r - 1] = v
    assert set(zip(*np.nonzero(kgg), strict=True)) == set(zip(*np.nonzero(want), strict=True))
    a, r = _err(kgg, want)
    nz = want != 0
    rel = float(np.max(np.abs(kgg[nz] - want[nz]) / np.abs(want[nz])))
    record_parity("rod KGGX vs t01241a listing", int(nz.sum()), a, rel)
    assert rel < 5e-7  # MATPRN prints 7 significant digits


def _deck_rods():
    for deck_id in sorted(RF1_DECKS):
        text = (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        if ecpts.get("ROD"):
            yield deck_id, model, [r.data for r in ecpts["ROD"]]


def test_every_deck_rod_through_kernel():
    """Smoke test: every CROD/CONROD of the inventoried RF1 decks."""
    total = 0
    worst = [0.0, 0.0]
    for deck_id, model, datas in _deck_rods():
        for consistent in (False, True):
            batch = rod_matrices(
                RodBatch.from_elements(datas, model.materials), model.cstm, consistent
            )
            for i, d in enumerate(datas):
                ref = rodd(d, model.materials, model.cstm, consistent)
                a, r = _err(batch.k[i], ref.k())
                assert r <= 1e-15, (deck_id, d.eid)
                np.testing.assert_array_equal(batch.m[i], ref.m())
                assert np.isfinite(batch.k[i]).all() and np.allclose(
                    batch.k[i], batch.k[i].T, rtol=0, atol=1e-14 * np.abs(batch.k[i]).max()
                )
                worst = [max(worst[0], a), max(worst[1], r)]
        _check_closed_form(datas, model.materials, model.cstm, f"rod {deck_id} vs closed form")
        total += len(datas)
    assert total > 0
    record_parity("rod deck smoke: batch vs reference", total, *worst)
