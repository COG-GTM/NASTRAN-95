"""CSHEAR stress recovery (``SPANL1``/``SPANL2``) vs the Fortran harness and the listings.

Harness parity covers the three basic-system SDR2 cases of ``harness_golden.json``
and the panels of ``shear_stress_golden.json`` (every flexibility branch, warped
and skewed panels, CSTM grids, margins of safety). Listing parity feeds the
golden displacements of d01011a-d01014a to the port and compares the
"STRESSES IN SHEAR PANELS" and shear-panel force tables.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from nastran95py.bulk import read_ecpts
from nastran95py.coords import CoordSystem
from nastran95py.elements import LAYOUTS, ShearGeometryError
from nastran95py.elements.rod import rodd
from nastran95py.elements.shear import sheard
from nastran95py.elements.shear_stress import (
    FORCE_COLUMNS,
    STRESS_COLUMNS,
    ShearStressBatch,
    ShearStressProperties,
    shear_stress_phase1,
    shear_stress_phase2,
    spanl1,
    spanl2,
)
from nastran95py.materials import Mat1, MaterialTable
from nastran95py.reference.golden import compare, load_golden, tables, tolerances
from nastran95py.reference.inventory import load_manifest, repo_root
from tests.conftest import harness_source, record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import runner as R
from tests.fortran_harness import shear_cases as SC
from tests.fortran_harness import shear_stress_cases as SS

RTOL = 1e-12
MATS = MaterialTable([Mat1(mid, *vals) for mid, vals in SS.MAT1])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)
BASIC_CASES = [c for c in C.ALL_CASES if c.routine == "SPANL1"]
CASES = BASIC_CASES + SS.SHEAR_STRESS_CASES
LISTING_DECKS = {"d01011a", "d01012a", "d01013a", "d01014a"}
BLOCKS = ("PHASE1", "STRESS", "FORCE")


def _golden() -> dict:
    data = json.loads(R.GOLDEN.read_text())
    data.update(json.loads(SS.GOLDEN.read_text()))
    return {c.name: {k: np.asarray(v) for k, v in data[c.name]["words"].items()} for c in CASES}


def _data(case: C.Case):
    return LAYOUTS["SHEAR"].parse(case.words)


def _disp(case: C.Case) -> np.ndarray:
    return np.asarray(case.extra[9 : 9 + case.extra[8]], float)


def _python_words(cases) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for c in cases:
        p1 = spanl1(_data(c), MATS, CSTM)
        p2 = spanl2(p1, _disp(c))
        out[c.name] = {
            "PHASE1": np.asarray(p1.words()),
            "STRESS": np.asarray(p2.stress_words()),
            "FORCE": np.asarray(p2.force_words()),
        }
    return out


def _err(a, b) -> tuple[float, float]:
    # The harness prints REAL words with nine digits: cast both sides back to REAL.
    a = np.asarray(a, float).astype(np.float32).astype(float)
    b = np.asarray(b, float).astype(np.float32).astype(float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _compare(fortran: dict, label: str) -> None:
    got = _python_words(CASES)
    for block in BLOCKS:
        worst = (0.0, 0.0)
        for c in CASES:
            err = _err(got[c.name][block], fortran[c.name][block])
            assert err[1] <= RTOL, (c.name, block, err)
            worst = tuple(map(max, worst, err))
        record_parity(
            f"SPANL1/SPANL2 {block} vs {label}", len(CASES), *worst,
            element="CSHEAR", quantity="stress", source=harness_source(label),
        )  # fmt: skip


def test_shear_stress_matches_fortran_golden():
    _compare(_golden(), "harness golden")


def test_shear_stress_matches_live_harness():
    if shutil.which(os.environ.get("FC", "gfortran")) is None:
        if os.environ.get("NASTRAN95PY_REQUIRE_FORTRAN") == "1":
            pytest.fail("gfortran is required (NASTRAN95PY_REQUIRE_FORTRAN=1)")
        pytest.skip("gfortran not available")
    R.build()
    live = {r.case.name: r.words for r in R.run(BASIC_CASES)}
    live.update({r.case.name: r.words for r in SS.run()})
    _compare(live, "live harness")


def test_harness_cases_cover_margins_and_branches():
    gold = _golden()
    margins = [np.float32(w["STRESS"][3]) for w in gold.values()]
    assert sum(m != SS_BLANK for m in margins) >= 5
    assert np.float32(gold["CSHEAR-general-ss-zero-SPANL"]["STRESS"][3]) == SS_BLANK
    branches = {sheard(_data(c), MATS, CSTM).branch for c in SS.SHEAR_STRESS_CASES}
    assert branches == {30, 40, 50, 70}


SS_BLANK = np.int32(1).view(np.float32)  # integer word 1: the margin is printed blank


def test_pure_shear_closed_form():
    """A rectangle in simple shear: every corner stress is ``G * gamma``."""
    gamma = 2.0e-4
    case = next(c for c in BASIC_CASES if c.name == "CSHEAR-rect-SPANL")
    pts, mid, t = C.SHEAR_GEOMS["rect"]
    data = _data(case)
    u = np.zeros(24)
    for i, (x, y, _) in enumerate(pts):
        u[6 * i : 6 * i + 2] = (0.5 * gamma * y, 0.5 * gamma * x)
    res = spanl2(spanl1(data, MATS, real_words=False), u, real_words=False)
    g = MATS.mat1(mid).g
    assert res.stress[0, 0] == pytest.approx(g * gamma, rel=1e-12)
    assert res.stress[0, 1] == pytest.approx(g * gamma, rel=1e-12)
    assert np.isnan(res.stress[0, 2])
    q = g * gamma * t
    np.testing.assert_allclose(res.forces[0, 9::2], q, rtol=1e-12)
    np.testing.assert_allclose(res.forces[0, 8::2], 0.0, atol=1e-9)


def test_margin_of_safety():
    c = next(c for c in SS.SHEAR_STRESS_CASES if c.name == "CSHEAR-trap34-ss-SPANL")
    res = spanl2(spanl1(_data(c), MATS, CSTM), _disp(c))
    sigs = np.float32(SS.SS_MAT[1][9])
    assert res.stress[0, 2] == pytest.approx(sigs / res.stress[0, 0] - 1.0, rel=1e-6)
    zero = spanl2(spanl1(_data(c), MATS, CSTM), np.zeros(24))
    assert np.isnan(zero.stress[0, 2]) and np.float32(zero.stress_words()[3]) == SS_BLANK


def test_double_precision_path_close_to_real_words():
    for c in SS.SHEAR_STRESS_CASES:
        p32 = spanl2(spanl1(_data(c), MATS, CSTM), _disp(c))
        p64 = spanl2(spanl1(_data(c), MATS, CSTM, real_words=False), _disp(c), real_words=False)
        np.testing.assert_allclose(p64.stress[:, :2], p32.stress[:, :2], rtol=1e-2, atol=1e-3)
        scale = np.max(np.abs(p32.forces))
        np.testing.assert_allclose(p64.forces, p32.forces, rtol=0, atol=1e-2 * scale)


def test_batch_matches_element_path():
    datas = [_data(c) for c in SS.SHEAR_STRESS_CASES]
    u = np.random.default_rng(7).uniform(-1e-3, 1e-3, 24)
    p1 = shear_stress_phase1(ShearStressBatch.from_elements(datas, MATS), CSTM)
    res = shear_stress_phase2(p1, u)
    for i, d in enumerate(datas):
        one = spanl1(d, MATS, CSTM)
        np.testing.assert_array_equal(p1.words(i), one.words())
        e = spanl2(one, u)
        np.testing.assert_array_equal(res.stress_words(i), e.stress_words())
        np.testing.assert_array_equal(res.force_words(i), e.force_words())


def test_properties_entry_and_missing_cstm():
    c = next(c for c in SS.SHEAR_STRESS_CASES if c.name == "CSHEAR-cstm-SPANL")
    m = MATS.mat1(1)
    props = ShearStressProperties(m.g, m.nu, m.sigs, c.meta["t"])
    np.testing.assert_array_equal(
        spanl1(_data(c), props, CSTM).words(), spanl1(_data(c), MATS, CSTM).words()
    )
    with pytest.raises(KeyError):
        spanl1(_data(c), MATS)


@pytest.mark.parametrize("tag", sorted(SC.SHEAR_ERROR_GEOMS))
def test_bad_geometry_matches_sheard(tag):
    grids, mid, t, nsm, sils = SC.SHEAR_ERROR_GEOMS[tag]
    data = LAYOUTS["SHEAR"].parse(SC.shear_cstm_est(1, grids, mid, t, nsm, sils))
    try:
        sheard(data, MATS)
    except ShearGeometryError as ref:
        if t == 0.0:  # SPANL1 has no thickness check: Z = inf, S = 0.
            pytest.skip("SHEARD-only check")
        with pytest.raises(ShearGeometryError) as err:
            spanl1(data, MATS)
        assert (err.value.code, err.value.corner) == (ref.code, ref.corner)
    else:
        spanl1(data, MATS)


# ----------------------------------------------------------------------------
# Listing parity
# ----------------------------------------------------------------------------


def _displacements(model, ecpts, printed) -> np.ndarray:
    """SIL-ordered displacements from the printed grids.

    The decks print only the upper-surface grids (``DISPLACEMENTS = 1``); the
    lower-surface grids have ``T1, T2`` (SPC1 12) and ``R1-R3`` (GRDSET 456)
    fixed, and their unloaded ``T3`` follows from the equilibrium rows of the
    CSHEAR and rod stiffness (``K_uu x_u = -K_uk u_k``).
    """
    n = max(model.sil.values()) + 6
    u, known = np.zeros(n), np.zeros(n, bool)
    for gid, row in printed.items():
        s = model.sil[gid] - 1
        u[s : s + 6], known[s : s + 6] = np.nan_to_num(row), True
    k = np.zeros((n, n))
    for r in ecpts["SHEAR"]:
        m = sheard(r.data, model.materials, model.cstm, real_words=False)
        idx = [s - 1 + c for s in m.sils for c in range(3)]
        k[np.ix_(idx, idx)] += m.k()
    for r in ecpts.get("ROD", []):
        m = rodd(r.data, model.materials, model.cstm, real_words=False)
        idx = [s - 1 + c for s in m.sils for c in range(6)]
        k[np.ix_(idx, idx)] += m.k()
    fixed = np.zeros(n, bool)
    for g in model.grids.values():
        for comp in g.ps:
            fixed[model.sil[g.gid] + int(comp) - 2] = True
    for spc in model.spcs:
        for comp in spc.c:
            fixed[model.sil[spc.g] + int(comp) - 2] = True
    unknown = ~known & ~fixed & (np.abs(np.diag(k)) > 0)
    assert all(known[model.sil[load.g] - 1] for load in model.loads)
    u[unknown] = np.linalg.solve(
        k[np.ix_(unknown, unknown)], -k[np.ix_(unknown, ~unknown)] @ u[~unknown]
    )
    return u


def _deck_parity():
    for deck in load_manifest()["decks"]:
        golden = tables(load_golden(deck["id"]))
        shear_tables = {
            k: t
            for k, t in golden.items()
            if t.meta["element_type"] == "CSHEAR" and t.meta["kind"] in ("stress", "force")
        }
        if not shear_tables:
            continue
        model, ecpts = read_ecpts((repo_root() / deck["inp"]).read_text(errors="replace"))
        records = {r.data.eid: r for r in ecpts["SHEAR"]}
        for key, table in shear_tables.items():
            kind, _, subcase, occ = key.split("/")[:4]
            u = _displacements(model, ecpts, golden[f"displacement/grid/{subcase}/{occ}"].by_id())
            datas = [records[int(e)].data for e in table.ids]
            p1 = shear_stress_phase1(
                ShearStressBatch.from_elements(datas, model.materials), model.cstm
            )
            res = shear_stress_phase2(p1, u)
            actual = res.stress if kind == "stress" else res.forces
            assert table.columns == (STRESS_COLUMNS if kind == "stress" else FORCE_COLUMNS)
            yield deck["id"], key, table, actual, compare(table, actual)


def _floor_rows(table, actual) -> list[tuple[float, int, str, float, float]]:
    """Entries that pass only on the absolute floor: (rel, eid, column, port, listing)."""
    rtol, _ = tolerances(table)
    gold = table.values
    with np.errstate(invalid="ignore"):
        err = np.abs(actual - gold)
        mask = ~np.isnan(gold) & (err > rtol * np.abs(gold))
    return [
        (
            float(err[i, j] / abs(gold[i, j])) if gold[i, j] else float("inf"),
            int(table.ids[i]),
            table.columns[j],
            float(actual[i, j]),
            float(gold[i, j]),
        )
        for i, j in np.argwhere(mask)
    ]


def test_shear_tables_match_listings():
    by_deck: dict[str, list] = {}
    floor: list = []
    for deck, key, table, actual, cmp in _deck_parity():
        assert cmp.passed, (deck, key, cmp)
        by_deck.setdefault(deck, []).append(cmp)
        floor += [(deck, key, *row) for row in _floor_rows(table, actual)]
    for deck, cmps in sorted(by_deck.items()):
        for c in cmps:
            record_parity(f"CSHEAR {c.key} {deck} vs listing", c.n, c.max_abs, c.max_rel)
    if floor:
        deck, key, rel, eid, col, got, exp = max(
            floor, key=lambda r: (r[6] != 0.0, r[2] if r[6] else abs(r[5]))
        )
        record_parity(
            f"CSHEAR worst abs-floor row: {deck} {key} EID {eid} {col} "
            f"(port {got:.6g}, listing {exp:.6g})",
            len(floor),
            abs(got - exp),
            rel,
        )
    assert set(by_deck) == LISTING_DECKS
    kinds = [c.key.split("/")[0] for cmps in by_deck.values() for c in cmps]
    assert kinds.count("stress") == 5 and kinds.count("force") == 1
