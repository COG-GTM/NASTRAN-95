"""CQDMEM2 stiffness, mass, thermal load and stress recovery (``nastran95py.elements.qdmem2``).

Parity: ``QDMM2D``/``QDMM2S`` (EMG), ``QDMM2`` (thermal load) and
``SQDM21``/``SQDM22`` (SDR2) against the Fortran dumps (basic-system cases in
``harness_golden.json``; warped, permuted, material-angle, NSM, offset/rotated/
curvilinear-system and load-temperature cases in ``qdmem2_golden.json``; and
the live harness when gfortran is available) at relative 1e-12; the CQDMEM2
stress tables of d01012a and d01014a fed the listing displacements, and those
of d01033a and d01034a fed the displacements of the solved thermal deck (whose
printed set does not cover the stressed elements), at the golden README
tolerance.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import read_ecpts
from nastran95py.bulk.ecpt import material_temperature_set
from nastran95py.bulk.loads import load_temperatures
from nastran95py.coords import CoordSystem
from nastran95py.elements import (
    LAYOUTS,
    Qdmem2Batch,
    Qdmem2Error,
    qdmem2_matrices,
    qdmem2_stress_matrices,
    qdmem2_stresses,
    qdmem2_thermal_loads,
    qdmm2,
    qdmm2d,
    sqdm21,
    sqdm22,
)
from nastran95py.elements.contract import ElementData
from nastran95py.materials import Mat1, Mat2, MaterialTable
from nastran95py.reference.cards import read_deck
from nastran95py.reference.golden import compare, load_golden, tables
from nastran95py.reference.inventory import repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import qdmem2_cases as QC
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(m, *v) for m, v in C.MAT1] + [Mat2(m, *v) for m, v in C.MAT2])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)
BASIC = [c for c in C.ALL_CASES if c.element == "CQDMEM2"]
CASES = BASIC + QC.QDMEM2_CASES
EMG = {p: [c for c in CASES if c.routine == f"QDMM2{p[0].upper()}"] for p in ("double", "single")}
SDR = [c for c in CASES if c.routine == "SQDM21"]
LOADS = QC.LOAD_CASES
LISTING_DECKS = ("d01012a", "d01014a", "d01033a", "d01034a")


def _golden() -> dict:
    data = {k: v for k, v in json.loads(R.GOLDEN.read_text()).items() if v["element"] == "CQDMEM2"}
    data.update(json.loads(QC.GOLDEN.read_text()))
    return data


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _real(v) -> np.ndarray:
    """Harness REAL words (9 digits) rounded back to the float32 the Fortran held."""
    return np.asarray(v, np.float32).astype(float)


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["QDMEM2"].parse(case.words)


def _batch(cases) -> Qdmem2Batch:
    return Qdmem2Batch.from_elements([_data(c) for c in cases], MATS)


def _sdr_load(case: C.Case):
    _, _, _, _, _, ldtemp, temp, _, ndisp, *disp = case.extra
    assert len(disp) == ndisp
    return np.asarray(disp, float), (None if ldtemp == -1 else temp)


class _Worst:
    def __init__(self):
        self.abs, self.rel, self.n = 0.0, 0.0, 0

    def add(self, got, exp, check=True):
        if check:
            np.testing.assert_allclose(got, exp, rtol=RTOL, atol=0.0)
        a, r = _err(got, exp)
        self.abs, self.rel, self.n = max(self.abs, a), max(self.rel, r), self.n + np.size(exp)


# -- Fortran parity ----------------------------------------------------------------


def _check_emg(golden: dict, label: str) -> None:
    for precision, cases in EMG.items():
        res = qdmem2_matrices(_batch(cases), CSTM, precision)
        wk, wm = _Worst(), _Worst()
        for i, case in enumerate(cases):
            mats = {m["file"]: m for m in golden[case.name]["matrices"]}
            rnd = _real if precision == "single" else (lambda v: np.asarray(v, float))
            el = res.element(i, case.words[0])
            assert mats[1]["dict2"] == 1 and mats[2]["dict2"] == 2
            wk.add(el.stiffness.values, rnd(mats[1]["values"]))
            wm.add(el.mass.values, rnd(mats[2]["values"]))
        for q, w in (("K", wk), ("M", wm)):
            record_parity(
                f"QDMM2{precision[0].upper()} {q} {label}", w.n, w.abs, w.rel,
                element="CQDMEM2", quantity=q, source=harness_source(label),
            )  # fmt: skip


ANGLE_ULPS = 2


def _check_angle(got: float, exp: float) -> None:
    """The principal angle is REAL ``ATAN2`` (libm ``atan2f``), whose last bit varies
    between C libraries: stored goldens allow ``ANGLE_ULPS`` float32 ulps on it, while
    the live harness (same libm as the port) must match exactly."""
    assert abs(got - exp) <= ANGLE_ULPS * abs(np.spacing(np.float32(exp))), (got, exp)


def _check_sdr(golden: dict, label: str) -> None:
    p1 = qdmem2_stress_matrices(_batch(SDR), CSTM)
    w = {k: _Worst() for k in ("phase1", "stress", "force", "angle")}
    for i, case in enumerate(SDR):
        words = golden[case.name]["words"]
        mine = p1.words(i)
        exp = _real(words["PHASE1"])
        assert [R.real_word_as_int(x) for x in exp[:5]] == case.words[:5]
        assert not exp[mine.size :].any()
        w["phase1"].add(mine[5:], exp[5 : mine.size])
        disp, temp = _sdr_load(case)
        one = qdmem2_stress_matrices(_batch([case]), CSTM)
        s = qdmem2_stresses(one, disp, temp)
        for block, got in (("stress", s.stress_words(0)), ("force", s.force_words(0))):
            exp = _real(words[block.upper()])
            assert R.real_word_as_int(exp[0]) == case.words[0]
            if block == "stress" and label != "live":
                got = got.copy()
                _check_angle(got[4], exp[4])
                w["angle"].add(got[4:5], exp[4:5], check=False)
                got[4] = exp[4]
            w[block].add(got[1:], exp[1:])
    for block, x in w.items():
        if x.n == 0:
            continue
        record_parity(
            f"SQDM2{1 if block == 'phase1' else 2} {block} {label}", x.n, x.abs, x.rel,
            element="CQDMEM2", quantity="stress", source=harness_source(label),
        )  # fmt: skip


def _check_loads(golden: dict, label: str) -> None:
    temps = [c.meta["temperature"] for c in LOADS]
    res = qdmem2_thermal_loads(_batch(LOADS), temps, CSTM)
    w = _Worst()
    for case, load in zip(LOADS, res, strict=True):
        exp = np.asarray(golden[case.name]["load"])
        w.add(load.add_to(np.zeros(exp.size)), exp)
    record_parity(
        f"QDMM2 thermal load {label}", w.n, w.abs, w.rel,
        element="CQDMEM2", quantity="load", source=harness_source(label),
    )  # fmt: skip


def test_matrices_match_fortran_golden():
    _check_emg(_golden(), "golden")


def test_stress_recovery_matches_fortran_golden():
    _check_sdr(_golden(), "golden")


def test_thermal_load_matches_fortran_golden():
    _check_loads(_golden(), "golden")


@pytest.fixture(scope="module")
def live() -> dict:
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    golden = R.to_golden(R.run(BASIC))
    golden.update(QC.run())
    return golden


@pytest.mark.fortran
def test_port_matches_live_harness(live):
    _check_emg(live, "live")
    _check_sdr(live, "live")
    _check_loads(live, "live")


@pytest.mark.fortran
def test_live_harness_reproduces_golden(live):
    # Newer gfortran builds reproduce the golden REAL words to ~1e-9 relative, not bit for bit.
    golden = json.loads(QC.GOLDEN.read_text())
    assert set(golden) == {c.name for c in QC.QDMEM2_CASES}
    for name, entry in golden.items():
        for key in ("matrices", "words", "load"):
            if key not in entry:
                continue
            if key == "matrices":
                pairs = [
                    (m["values"], r["values"])
                    for m, r in zip(live[name][key], entry[key], strict=True)
                ]
            elif key == "words":
                pairs = [(live[name][key][k], v) for k, v in entry[key].items()]
            else:
                pairs = [(live[name][key], entry[key])]
            for got, exp in pairs:
                np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-30, err_msg=name)


def test_cases_cover_every_branch():
    batch = _batch(EMG["double"])
    planar = qdmem2_matrices(batch, CSTM).planar
    assert planar.any() and (~planar).any()
    assert {0, 1, 2, 3} <= {int(CSTM[c].ctype) if c else 0 for c in batch.cid.ravel()}
    assert len({tuple(np.argsort(s)) for s in batch.sil}) > 3
    assert (batch.sinth != 0).any() and (batch.nsm != 0).any()
    assert any(_sdr_load(c)[1] is not None for c in SDR)


def test_single_element_wrappers_match_batch():
    case = next(c for c in QC.QDMEM2_CASES if c.name == "CQDMEM2-mixed-SQDM-temp")
    data, (disp, temp) = _data(case), _sdr_load(case)
    batch = _batch([case])
    k = qdmm2d(data, MATS, CSTM)
    np.testing.assert_array_equal(k.k(), qdmem2_matrices(batch, CSTM).k[0])
    p1 = sqdm21(data, MATS, CSTM)
    np.testing.assert_array_equal(p1.words(0), qdmem2_stress_matrices(batch, CSTM).words(0))
    np.testing.assert_array_equal(
        sqdm22(p1, disp, temp).stress, qdmem2_stresses(p1, disp, temp).stress
    )
    np.testing.assert_array_equal(
        qdmm2(data, MATS, 170.0, CSTM).loads, qdmem2_thermal_loads(batch, 170.0, CSTM)[0].loads
    )


# -- physics -----------------------------------------------------------------------


def _rigid(grids, sils_order) -> list[np.ndarray]:
    x = np.asarray(grids, float)[sils_order]
    modes = [np.tile(np.eye(3)[a], 4) for a in range(3)]
    modes += [np.cross(np.eye(3)[a], x).reshape(-1) for a in range(3)]
    return modes


@pytest.mark.parametrize(
    "case", [c for c in EMG["double"] if all(cid == 0 for cid in c.words[9:25:4])]
)
def test_stiffness_symmetric_and_rigid_body_free(case):
    el = qdmm2d(_data(case), MATS)
    k = el.k()
    scale = np.abs(k).max()
    assert np.abs(k - k.T).max() <= 1e-9 * scale
    order = np.argsort(case.words[1:5])
    grids = np.asarray([case.words[10 + 4 * g : 13 + 4 * g] for g in range(4)], float)
    # A warped (non-planar) CQDMEM2 is only checked against the three translations.
    modes = _rigid(grids, order) if el.planar else _rigid(grids, order)[:3]
    for v in modes:
        assert np.abs(k @ v).max() <= 1e-9 * scale * np.abs(v).max()
    ev = np.linalg.eigvalsh(0.5 * (k + k.T))
    assert ev.min() > -1e-9 * scale


def test_mass_is_total_mass_split_four_ways():
    case = next(c for c in QC.QDMEM2_CASES if c.name == "CQDMEM2-trap-aniso-60-QDMM2D")
    m = qdmm2d(_data(case), MATS).mass.values
    rho, t, nsm = C.MAT2[0][1][6], case.meta["t"], case.meta["nsm"]
    area = 0.5 * (8.0 + 3.0) * 3.0
    total = (rho * t + nsm) * area
    # Each grid gets (pivot triangle + whole quad) / 6; the four sum to the total mass.
    assert m.reshape(4, 3).sum(axis=0) == pytest.approx([total] * 3, rel=1e-6)
    assert len(set(np.round(m[::3], 12))) > 1


def test_constant_strain_patch_stress():
    """Basic patch cases (closed form checked in test_fortran_harness) via the wrappers."""
    for case in [c for c in SDR if c.name.endswith("-SQDM") and c in BASIC]:
        p1 = sqdm21(_data(case), MATS)
        disp, _ = _sdr_load(case)
        s = sqdm22(p1, disp).stress[0]
        ref = _real(_golden()[case.name]["words"]["STRESS"][1:8])
        _check_angle(s[3], ref[3])
        np.testing.assert_allclose(np.delete(s, 3), np.delete(ref, 3), rtol=RTOL)


def test_free_thermal_expansion_is_stress_free_and_balanced():
    """u = alpha * (T - TREF) * x: K u equals the thermal load and the stresses vanish."""
    grids = [(0, 0, 0), (4, 0, 0), (4.5, 3, 0), (0.5, 2.5, 0)]
    words = [901, 1, 7, 13, 19, 0.0, 1, 0.2, 0.0]
    for xyz in grids:
        words += [0, *map(float, xyz)]
    data = LAYOUTS["QDMEM2"].parse([*words, 0.0])
    mat = C.MAT1_BY_ID[1]
    alpha, tref, temp = mat[4], mat[5], 245.0
    u = np.zeros(24)
    for sil, xyz in zip((1, 7, 13, 19), grids, strict=True):
        u[sil - 1 : sil + 2] = alpha * (temp - tref) * np.asarray(xyz, float)
    k = qdmm2d(data, MATS).k()
    p = qdmm2(data, MATS, temp).add_to(np.zeros(24))
    ku = k @ np.concatenate([u[s - 1 : s + 2] for s in (1, 7, 13, 19)])
    pv = np.concatenate([p[s - 1 : s + 2] for s in (1, 7, 13, 19)])
    np.testing.assert_allclose(ku, pv, rtol=0, atol=1e-5 * np.abs(pv).max())
    s = sqdm22(sqdm21(data, MATS), u, temp).stress[0]
    e_alpha_dt = mat[0] * alpha * (temp - tref)
    assert np.abs(s[[0, 1, 2, 4, 5, 6]]).max() < 1e-5 * e_alpha_dt


def test_degenerate_geometry_raises():
    words = [902, 1, 7, 13, 19, 0.0, 1, 0.2, 0.0]
    for xyz in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]:
        words += [0, *map(float, xyz)]
    data = LAYOUTS["QDMEM2"].parse([*words, 0.0])
    for fn in (
        lambda: qdmm2d(data, MATS),
        lambda: sqdm21(data, MATS),
        lambda: qdmm2(data, MATS, 100.0),
    ):
        with pytest.raises(Qdmem2Error, match="902"):
            fn()


def test_bulk_reader_applies_matt1_tablem1():
    text = (repo_root() / "inp" / "d01033a.inp").read_text(errors="replace")
    model, _ = read_ecpts(text)
    assert "MATT1" not in model.ignored and "TABLEM1" not in model.ignored
    assert model.materials.mat1(75, 80.0).e == pytest.approx(10.4e6)
    assert model.materials.mat1(75, 225.0).e == pytest.approx(0.5 * (9.84e6 + 9.51e6))


# -- listing parity ------------------------------------------------------------------


def _solve_thermal_deck(model, recs, temp_set: int) -> np.ndarray:
    """Static solution of a CQDMEM2-only deck under its element thermal loads (SPC set 1)."""
    batch = Qdmem2Batch.from_elements([r.data for r in recs], model.materials)
    mats = qdmem2_matrices(batch, model.cstm)
    temps = load_temperatures(model, recs, temp_set)
    loads = qdmem2_thermal_loads(batch, [temps[r.data.eid] for r in recs], model.cstm)
    n = max(model.sil.values()) + 6
    k, p = np.zeros((n, n)), np.zeros(n)
    for i, load in enumerate(loads):
        idx = np.concatenate([np.arange(s - 1, s + 2) for s in mats.sils[i]])
        k[np.ix_(idx, idx)] += mats.k[i]
        load.add_to(p)
    free = np.zeros(n, bool)
    for gid in model.grids:
        s = model.sil[gid]
        free[s - 1 : s + 2] = True
    for gid in model.grids:  # GRDSET PS = 3456 in these decks
        free[model.sil[gid] + 1] = False
    for spc in model.spcs:
        if spc.sid == 1:
            for ch in spc.c:
                free[model.sil[spc.g] - 2 + int(ch)] = False
    u = np.zeros(n)
    u[free] = np.linalg.solve(k[np.ix_(free, free)], p[free])
    return u


def _deck_parity():
    for deck in LISTING_DECKS:
        text = (repo_root() / "inp" / f"{deck}.inp").read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        temp_set = material_temperature_set(read_deck(text).case_control)
        recs = ecpts["QDMEM2"]
        by_eid = {r.data.eid: r for r in recs}
        golden = tables(load_golden(deck))
        solved = _solve_thermal_deck(model, recs, temp_set) if temp_set else None
        if solved is not None:
            disp = golden["displacement/grid/sc1/occ1"]
            rows = np.array(
                [solved[model.sil[int(g)] - 1 : model.sil[int(g)] + 5] for g in disp.ids]
            )
            yield deck, f"{disp.key} (solved)", compare(disp, rows)
        for key, table in golden.items():
            if table.meta["kind"] != "stress" or table.meta["element_type"] != "CQDMEM2":
                continue
            _, _, subcase, occ = key.split("/")[:4]
            disp = golden[f"displacement/grid/{subcase}/{occ}"]
            ids = [int(e) for e in table.ids]
            if solved is None:
                printed = disp.by_id()
                assert all(g in printed for e in ids for g in by_eid[e].grid_ids), (deck, key)
                u = np.zeros(max(model.sil.values()) + 6)
                for gid, row in printed.items():
                    u[model.sil[gid] - 1 : model.sil[gid] + 5] = np.nan_to_num(row)
                temps = None
            else:
                u = solved
                lt = load_temperatures(model, [by_eid[e] for e in ids], temp_set)
                temps = [lt[e] for e in ids]
            batch = Qdmem2Batch.from_elements([by_eid[e].data for e in ids], model.materials)
            res = qdmem2_stresses(qdmem2_stress_matrices(batch, model.cstm), u, temps)
            yield deck, key, compare(table, res.stress_table())


def test_cqdmem2_stress_tables_match_listings():
    by_deck: dict[str, list] = {}
    for deck, key, cmp in _deck_parity():
        assert cmp.passed, (deck, key, cmp)
        by_deck.setdefault(deck, []).append((key, cmp))
    assert set(by_deck) == set(LISTING_DECKS)
    for deck, items in sorted(by_deck.items()):
        for kind in ("displacement", "stress"):
            cmps = [c for k, c in items if k.startswith(kind)]
            if cmps:
                record_parity(
                    f"CQDMEM2 {kind} {deck} vs listing",
                    sum(c.n for c in cmps),
                    max(c.max_abs for c in cmps),
                    max(c.max_rel for c in cmps),
                )
    # d01034a also prints its three SCAN selections of the same table.
    assert sum(1 for k, _ in by_deck["d01034a"] if k.startswith("stress")) == 4
