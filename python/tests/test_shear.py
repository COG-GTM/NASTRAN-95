"""CSHEAR stiffness and mass (``nastran95py.elements.shear``, port of ``SHEARD``).

Parity: the reference and vectorised paths against the Fortran ``SHEARD`` dumps
(basic-system panels in ``harness_golden.json``; flat, warped, skewed and
CSTM panels covering every flexibility branch in ``shear_golden.json``; the live
harness when gfortran is available) at relative 1e-12, and the panels ``SHEARD``
rejects raising :class:`ShearGeometryError` with the Fortran message. Also the
rigid-body null space, the lumped mass total, and every CSHEAR in the RF1 decks.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import read_ecpts
from nastran95py.coords import CoordSystem
from nastran95py.elements import (
    LAYOUTS,
    ElementData,
    ShearBatch,
    ShearGeometryError,
    ShearProperties,
    shear_matrices,
    sheard,
)
from nastran95py.elements.shear import (
    BRANCH_GENERAL,
    BRANCH_PARALLEL_12_34,
    BRANCH_PARALLEL_14_23,
    BRANCH_PARALLELOGRAM,
)
from nastran95py.materials import Mat1, MaterialTable, Matt1, TableM1
from nastran95py.reference.inventory import RF1_DECKS, repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R
from tests.fortran_harness import shear_cases as SC

RTOL = 1e-12
MATS = MaterialTable([Mat1(mid, *vals) for mid, vals in C.MAT1])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)
CASES = [c for c in C.ALL_CASES if c.routine == "SHEARD"] + SC.SHEAR_CASES
SMOKE_DECKS = ("d01011a", "d01012a", "d01013a", "d01014a")


def _golden() -> dict:
    data = json.loads(R.GOLDEN.read_text())
    data.update(json.loads(SC.GOLDEN.read_text()))
    return {c.name: data[c.name] for c in CASES}


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["SHEAR"].parse(case.words)


def _split(dumps: dict):
    ok = [c for c in CASES if "nogo" not in dumps[c.name]]
    bad = [c for c in CASES if "nogo" in dumps[c.name]]
    return ok, bad


def _expected_error(entry: dict) -> tuple[int, int | None]:
    (_, num, _), *_ = entry["mesage"]
    code = 2000 + num
    return code, (entry["nogo"] if code == 2027 else None)


def _compare(dumps: dict, label: str) -> None:
    """Reference and batched paths vs Fortran ``EMGOUT`` dumps, and the error exits."""
    ok, bad = _split(dumps)
    datas = [_data(c) for c in ok]
    batch = shear_matrices(ShearBatch.from_elements(datas, MATS), CSTM)
    worst = {"ref": [0.0, 0.0, 0], "batch": [0.0, 0.0, 0]}
    per_file = {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0]}
    for i, (case, data) in enumerate(zip(ok, datas, strict=True)):
        out = sheard(data, MATS, CSTM)
        ref = dumps[case.name]["matrices"]
        assert len(ref) == 2, case.name
        assert tuple(batch.sils[i]) == out.sils == tuple(sorted(case.meta.get("sils", C.SILS)))
        assert batch.branch[i] == out.branch
        for m, mine, full in zip(
            ref, (out.stiffness, out.mass), (batch.k[i], batch.m[i]), strict=True
        ):
            assert (mine.ndof, mine.icode, mine.dict2) == (m["ndof"], m["icode"], m["dict2"])
            vals = np.asarray(m["values"])
            ra, rr = _err(mine.values, vals)
            ba, br = _err(full, R._square({**m, "values": vals}))
            assert rr <= RTOL and br <= RTOL, (case.name, m["file"], rr, br)
            for key, (a, r) in (("ref", (ra, rr)), ("batch", (ba, br))):
                w = worst[key]
                w[0], w[1], w[2] = max(w[0], a), max(w[1], r), w[2] + vals.size
                w = per_file[m["file"]]
                w[0], w[1], w[2] = max(w[0], a), max(w[1], r), w[2] + vals.size
    for key, (a, r, n) in worst.items():
        record_parity(f"shear {key} vs {label}", n, a, r)
    for file, (a, r, n) in per_file.items():
        quantity = "K" if file == 1 else "M"
        record_parity(
            f"shear {quantity} vs {label}", n, a, r,
            element="CSHEAR", quantity=quantity, source=harness_source(label),
        )  # fmt: skip

    for case in bad:
        code, corner = _expected_error(dumps[case.name])
        data = _data(case)
        with pytest.raises(ShearGeometryError) as exc:
            sheard(data, MATS, CSTM)
        assert (exc.value.eid, exc.value.code, exc.value.corner) == (data.eid, code, corner)
        with pytest.raises(ShearGeometryError) as bexc:
            shear_matrices(ShearBatch.from_elements([_data(ok[0]), data], MATS), CSTM)
        assert str(bexc.value) == str(exc.value)


def test_cases_cover_every_branch_and_error():
    gold = _golden()
    ok, bad = _split(gold)
    assert len(CASES) == 3 + 28 and len(bad) == 8
    branches = {sheard(_data(c), MATS, CSTM).branch for c in ok}
    assert branches == {
        BRANCH_PARALLEL_12_34,
        BRANCH_PARALLEL_14_23,
        BRANCH_PARALLELOGRAM,
        BRANCH_GENERAL,
    }
    assert {_expected_error(gold[c.name]) for c in bad} == {
        (2026, None),
        (2027, 1),
        (2027, 2),
        (2027, 3),
        (2027, 4),
    }


def test_sheard_matches_fortran_golden():
    _compare(_golden(), "Fortran SHEARD golden")


@pytest.mark.fortran
def test_sheard_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    base = [c for c in CASES if c not in SC.SHEAR_CASES]
    live = R.to_golden(R.run(base))
    live.update(SC.to_golden(*SC.run()))
    _compare(live, "Fortran SHEARD harness")
    gold = _golden()
    errs = [
        _err(m["values"], g["values"])
        for name, entry in live.items()
        for m, g in zip(entry.get("matrices", []), gold[name].get("matrices", []), strict=True)
    ]
    a, r = max(e[0] for e in errs), max(e[1] for e in errs)
    record_parity("shear_golden.json vs live SHEARD", len(errs), a, r)
    assert r <= RTOL
    for name, entry in live.items():
        assert entry.get("mesage") == gold[name].get("mesage"), name


def test_error_messages_match_usrmsg():
    sq = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]
    flat = _data(
        C.Case(
            "x",
            "CSHEAR",
            "SHEARD",
            C.EMG,
            SC.shear_cstm_est(42, SC._basic(sq), 1, 0.0, 0.0, C.SILS),
            [],
        )
    )
    with pytest.raises(ShearGeometryError) as exc:
        sheard(flat, MATS)
    assert str(exc.value) == (
        "*** USER FATAL MESSAGE 2026,ELEMENT       42 GEOMETRY YIELDS UNREASONABLE MATRIX"
    )
    dart = [(0, 0, 0), (10, 0, 0), (3, 3, 0), (0, 10, 0)]
    bad = _data(
        C.Case(
            "x",
            "CSHEAR",
            "SHEARD",
            C.EMG,
            SC.shear_cstm_est(7, SC._basic(dart), 1, 0.1, 0.0, C.SILS),
            [],
        )
    )
    with pytest.raises(ShearGeometryError) as exc:
        sheard(bad, MATS)
    assert str(exc.value) == (
        "*** USER FATAL MESSAGE 2027,ELEMENT        7 HAS INTERIOR ANGLE GREATER THAN"
        " 180 DEG. AT GRID POINT        3"
    )


def _rigid_modes(xyz: np.ndarray) -> np.ndarray:
    """Translations and rotations of the four grids (basic system), 12x6."""
    modes = np.zeros((12, 6))
    for g, p in enumerate(xyz):
        modes[3 * g : 3 * g + 3, :3] = np.eye(3)
        x, y, z = p
        modes[3 * g : 3 * g + 3, 3:] = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
    return modes


def _check_physics(batch: ShearBatch, out, label: str) -> tuple[float, float]:
    """Basic-system panels: symmetric, rigid translations in the null space, mass total."""
    worst = [0.0, 0.0]
    for i in range(len(batch)):
        if (batch.cid[i] != 0).any():
            continue
        order = np.argsort(batch.sil[i], kind="stable")
        k = out.k[i]
        scale = np.abs(k).max()
        assert np.isfinite(k).all(), (label, batch.eid[i])
        np.testing.assert_allclose(k, k.T, rtol=0, atol=1e-13 * scale)
        modes = _rigid_modes(batch.xyz[i][order])
        assert np.abs(k @ modes[:, :3]).max() <= 1e-9 * scale, (label, batch.eid[i])
        p = batch.xyz[i]
        area = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[1]))
        mu = batch.rho[i] * batch.t[i] + batch.nsm[i]
        m = np.diag(out.m[i])
        flat = abs(np.dot(np.cross(p[1] - p[0], p[3] - p[0]), p[2] - p[0])) < 1e-12
        if flat:
            for c in range(3):
                a, r = _err(m[c::3].sum(), mu * area)
                worst = [max(worst[0], a), max(worst[1], r)]
                assert r < 1e-6, (label, batch.eid[i])  # REAL inputs
    return worst[0], worst[1]


def test_rigid_body_and_mass_total():
    ok, _ = _split(_golden())
    datas = [_data(c) for c in ok]
    batch = ShearBatch.from_elements(datas, MATS)
    out = shear_matrices(batch, CSTM, real_words=False)
    a, r = _check_physics(batch, out, "harness cases")
    record_parity("shear mass total vs rho*t*A (flat panels)", len(datas), a, r)


def test_float64_path_close_to_real_words():
    ok, _ = _split(_golden())
    datas = [_data(c) for c in ok]
    batch = ShearBatch.from_elements(datas, MATS)
    exact = shear_matrices(batch, CSTM, real_words=False)
    for i, d in enumerate(datas):
        ref = sheard(d, MATS, CSTM, real_words=False)
        np.testing.assert_allclose(exact.k[i], ref.k(), rtol=0, atol=1e-12 * np.abs(ref.k()).max())
        real = sheard(d, MATS, CSTM)
        np.testing.assert_allclose(real.k(), ref.k(), rtol=0, atol=1e-5 * np.abs(ref.k()).max())


def test_temperature_dependent_material():
    mats = MaterialTable(
        [Mat1(1, 1.0e7, 4.0e6, 0.25, 0.1)],
        matt1=[Matt1(1, g=7, rho=8)],
        tables=[TableM1(7, (0.0, 200.0), (1.0, 0.5)), TableM1(8, (0.0, 200.0), (1.0, 3.0))],
    )
    base = _data(SC.SHEAR_CASES[0])
    base.fields["mid"] = 1
    hot = ElementData("SHEAR", base.fields, base.grids, temperature=100.0)
    cold_out, hot_out = sheard(base, mats), sheard(hot, mats)
    np.testing.assert_allclose(hot_out.k(), 0.75 * cold_out.k(), rtol=1e-6)
    np.testing.assert_allclose(hot_out.m(), 2.0 * cold_out.m(), rtol=1e-6)


def test_properties_input_and_nsm_only_mass():
    data = _data(SC.SHEAR_CASES[0])
    props = ShearProperties(e=1e7, g=4e6, nu=0.25, rho=0.0, ge=0.02, t=0.1, nsm=0.5)
    out = sheard(data, props)
    assert out.ge == 0.02
    np.testing.assert_allclose(np.diag(out.m()), 0.5 * 16.0 / 4.0)


def _deck_shears():
    for deck_id in sorted(set(SMOKE_DECKS) | set(RF1_DECKS)):
        path = repo_root() / "inp" / f"{deck_id}.inp"
        if not path.exists():
            continue
        model, ecpts = read_ecpts(path.read_text(errors="replace"))
        if ecpts.get("SHEAR"):
            yield deck_id, model, [r.data for r in ecpts["SHEAR"]]


def test_every_deck_shear_through_kernel():
    """Smoke test: every CSHEAR of the inventoried decks (d01011a-d01014a)."""
    total, seen = 0, set()
    worst = [0.0, 0.0]
    for deck_id, model, datas in _deck_shears():
        seen.add(deck_id)
        batch = ShearBatch.from_elements(datas, model.materials)
        out = shear_matrices(batch, model.cstm)
        for i, d in enumerate(datas):
            ref = sheard(d, model.materials, model.cstm)
            a, r = _err(out.k[i], ref.k())
            assert r <= 1e-15, (deck_id, d.eid)
            np.testing.assert_array_equal(out.m[i], ref.m())
            worst = [max(worst[0], a), max(worst[1], r)]
        _check_physics(batch, out, deck_id)
        total += len(datas)
    assert set(SMOKE_DECKS) <= seen
    record_parity("shear deck smoke: batch vs reference", total, *worst)
