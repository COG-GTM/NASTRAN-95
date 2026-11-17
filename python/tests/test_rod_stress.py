"""Rod stress recovery (``nastran95py.elements.rod_stress``, port of ``SROD1``/``SROD2``).

Parity: the reference and vectorised paths against the Fortran SDR2 words
(the basic-system SROD cases of ``harness_golden.json``, the coordinate-system,
thermal, deformation and margin cases of ``rod_stress_golden.json``, and the
live harness when gfortran is available) at relative 1e-12; closed-form axial,
torsional, thermal and ``DEFORM`` stresses; every margin-of-safety branch; and
the rod stress/force tables of the COSMIC listings fed the golden displacements,
at the golden README tolerance.
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
    RodGeometryError,
    RodStressBatch,
    real_cstm,
    rod_stress_matrices,
    rod_stresses,
    srod1,
    srod2,
)
from nastran95py.elements.rod_stress import NO_MARGIN_WORD
from nastran95py.materials import Mat1, MaterialTable
from nastran95py.reference.golden import GoldenTable, compare, load_golden, tables
from nastran95py.reference.inventory import load_manifest, repo_root
from tests.conftest import record_parity
from tests.fortran_harness import cases as C
from tests.fortran_harness import rod_cases as RC
from tests.fortran_harness import rod_stress_cases as SC
from tests.fortran_harness import runner as R

RTOL = 1e-12
MATS = MaterialTable([Mat1(mid, *vals) for mid, vals in SC.MAT1])
CSTM = {cid: CoordSystem(cid, t, w[:3], np.reshape(w[3:], (3, 3))) for cid, t, w in RC.CSTM}
CSTM.update(RC.FROM_POINTS)
REAL_CSTM = real_cstm(CSTM)


def _orthonormal(t: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(t)
    return u @ vt


CSTM_EXACT = {
    cid: CoordSystem(cid, cs.ctype, cs.origin, _orthonormal(cs.t)) for cid, cs in CSTM.items()
}
BASIC_CASES = [c for c in C.ALL_CASES if c.routine == "SROD1"]
CASES = BASIC_CASES + SC.ROD_STRESS_CASES


def _load(case: C.Case) -> tuple[list[float], float | None, float]:
    """Displacements, load temperature and deformation the harness passed to ``SROD2``."""
    _, _, _, _, _, ldtemp, temp, eldefm, ndisp, *disp = case.extra
    assert len(disp) == ndisp
    return disp, (None if ldtemp == -1 else temp), eldefm


def _golden() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    basic = json.loads(R.GOLDEN.read_text())
    for c in BASIC_CASES:
        w = basic[c.name]["words"]
        out[c.name] = {k.lower(): np.asarray(w[k], np.float32).astype(float) for k in w}
    for name, entry in json.loads(SC.GOLDEN.read_text()).items():
        out[name] = {k: np.asarray(entry[k]) for k in ("phase1", "stress", "force")}
    return out


def _err(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b)))
    return max_abs, max_abs / scale if scale else max_abs


def _data(case: C.Case) -> ElementData:
    return LAYOUTS["ROD"].parse(case.words)


def _python_words(cases) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Phase-I, stress and force words of ``cases`` from the reference and batched paths."""
    datas = [_data(c) for c in cases]
    batch = rod_stress_matrices(datas, MATS, REAL_CSTM)
    loads = [_load(c) for c in cases]
    temps = np.array([np.nan if t is None else t for _, t, _ in loads])
    out: dict[str, dict] = {"ref": {}, "batch": {}}
    for i, (case, data, (disp, temp, eldefm)) in enumerate(zip(cases, datas, loads, strict=True)):
        p1 = srod1(data, MATS, REAL_CSTM)
        s = srod2(p1, disp, temp, eldefm)
        out["ref"][case.name] = {
            "phase1": p1.words(),
            "stress": s.stress_words(),
            "force": s.force_words(),
        }
        one = _slice(batch, i)
        r = rod_stresses(one, disp, temps[i : i + 1], eldefm)
        margins = [
            NO_MARGIN_WORD if np.isnan(m) else m for m in (r.axial_margin[0], r.torsional_margin[0])
        ]
        out["batch"][case.name] = {
            "phase1": batch.element(i).words(),
            "stress": np.array([s.stress_words()[0], r.sigma[0], margins[0], r.tau[0], margins[1]]),
            "force": np.array([s.force_words()[0], r.force[0], r.torque[0]]),
        }
    return out


def _slice(batch: RodStressBatch, i: int) -> RodStressBatch:
    return RodStressBatch(**{f: getattr(batch, f)[i : i + 1] for f in batch.__dataclass_fields__})


def _compare(fortran: dict[str, dict[str, np.ndarray]], label: str) -> None:
    mine = _python_words(CASES)
    for path, words in mine.items():
        worst = {k: [0.0, 0.0, 0] for k in ("phase1", "stress", "force")}
        for case in CASES:
            for block, got in words[case.name].items():
                exp = fortran[case.name][block]
                np.testing.assert_allclose(
                    got, exp, rtol=RTOL, atol=0.0, err_msg=f"{path} {case.name} {block}"
                )
                a, r = _err(got, exp)
                w = worst[block]
                w[0], w[1], w[2] = max(w[0], a), max(w[1], r), w[2] + exp.size
        for block, (a, r, n) in worst.items():
            record_parity(f"SROD {block} {label} ({path})", n, a, r)


def test_rod_stress_matches_fortran_golden():
    golden = _golden()
    assert set(golden) == {c.name for c in CASES}
    _compare(golden, "golden")


@pytest.mark.fortran
def test_rod_stress_matches_live_harness():
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
    golden = json.loads(SC.GOLDEN.read_text())
    assert SC.to_golden([live[c.name] for c in SC.ROD_STRESS_CASES]) == golden


def test_harness_cases_cover_every_branch():
    """Both stress signs and zero, every margin branch, thermal and DEFORM terms."""
    golden = _golden()
    seen = set()
    for case in SC.ROD_STRESS_CASES:
        _, sigma, ms_sigma, tau, ms_tau = golden[case.name]["stress"]
        mat = dict(SC.MAT1)[case.meta["mid"]]
        blank_s, blank_t = ms_sigma == NO_MARGIN_WORD, ms_tau == NO_MARGIN_WORD
        if sigma > 0:
            seen.add("tension-ms" if not blank_s else "tension-blank")
            assert blank_s == (mat[7] <= 0)
        elif sigma < 0:
            seen.add("compression-ms" if not blank_s else "compression-blank")
            assert blank_s == (mat[8] <= 0)
        else:
            seen.add("zero")
            assert blank_s
        seen.add("torsion-ms" if not blank_t else "torsion-blank")
        seen.update(k for k in ("temperature", "deformation") if case.meta[k])
    assert seen >= {
        "tension-ms",
        "tension-blank",
        "compression-ms",
        "compression-blank",
        "zero",
        "torsion-ms",
        "torsion-blank",
        "temperature",
        "deformation",
    }


def _rod(cids=(0, 0), a=(1.0, -2.0, 0.5), b=(4.0, 2.0, 12.5), mid=4, area=0.8, j=0.3, c=0.6):
    words = [11, 1, 7, mid, area, j, c, 0.0, cids[0], *a, cids[1], *b, 0.0]
    return LAYOUTS["ROD"].parse(words)


def _global(cid, xyz, v_basic):
    ta = transforms_to_basic(CSTM_EXACT, cid, np.asarray(xyz, float))[0]
    return ta.T @ v_basic


def _rigid_plus(data, stretch, twist, rigid=(0.0, 0.0, 0.0)):
    """Global displacements: rigid translation plus axial stretch and twist of grid B."""
    (ga, gb) = data.grids
    n = (gb.xyz - ga.xyz) / np.linalg.norm(gb.xyz - ga.xyz)
    u = np.zeros(12)
    t0 = np.asarray(rigid, float)
    u[0:3] = _global(ga.cid, ga.xyz, t0)
    u[6:9] = _global(gb.cid, gb.xyz, t0 + stretch * n)
    u[9:12] = _global(gb.cid, gb.xyz, twist * n)
    return u


@pytest.mark.parametrize("cids", [(0, 0), (10, 20), (30, 41), (42, 0), (40, 21)])
def test_closed_form_axial_and_torsion(cids):
    data = _rod(cids)
    mat = MATS.mat1(4)
    length = float(np.linalg.norm(data.grids[1].xyz - data.grids[0].xyz))
    u = _rigid_plus(data, 2.0e-3, 1.5e-3, rigid=(3e-3, -1e-3, 2e-3))
    s = srod2(srod1(data, MATS, CSTM_EXACT, real_words=False), u, real_words=False)
    sigma = mat.e * 2.0e-3 / length
    torque = mat.g * 0.3 * 1.5e-3 / length
    assert s.sigma == pytest.approx(sigma, rel=1e-12)
    assert s.force == pytest.approx(0.8 * sigma, rel=1e-12)
    assert s.torque == pytest.approx(torque, rel=1e-12)
    assert s.tau == pytest.approx(0.6 / 0.3 * torque, rel=1e-12)
    assert s.axial_margin == pytest.approx(mat.sigt / sigma - 1.0, rel=1e-12)
    assert s.torsional_margin == pytest.approx(mat.sigs / abs(s.tau) - 1.0, rel=1e-12)


@pytest.mark.parametrize("cids", [(0, 0), (10, 30), (41, 42)])
def test_free_thermal_expansion_is_stress_free(cids):
    data = _rod(cids)
    mat = MATS.mat1(4)
    length = float(np.linalg.norm(data.grids[1].xyz - data.grids[0].xyz))
    dt_ = 125.0
    u = _rigid_plus(data, mat.alpha * dt_ * length, 0.0)
    p1 = srod1(data, MATS, CSTM_EXACT, real_words=False)
    assert p1.st == pytest.approx(-mat.alpha * mat.e)
    free = srod2(p1, u, mat.tref + dt_, real_words=False)
    restrained = srod2(p1, np.zeros(12), mat.tref + dt_, real_words=False)
    scale = mat.e * mat.alpha * dt_
    assert abs(free.sigma) < 1e-12 * scale
    assert restrained.sigma == pytest.approx(-scale, rel=1e-12)
    assert srod2(p1, np.zeros(12), mat.tref, real_words=False).sigma == 0.0
    assert srod2(p1, np.zeros(12), None, real_words=False).sigma == 0.0


def test_deform_stress_is_minus_e_delta_over_length():
    data = _rod()
    mat = MATS.mat1(4)
    length = float(np.linalg.norm(data.grids[1].xyz - data.grids[0].xyz))
    p1 = srod1(data, MATS, real_words=False)
    s = srod2(p1, np.zeros(12), deformation=0.004, real_words=False)
    assert s.sigma == pytest.approx(-mat.e * 0.004 / length, rel=1e-12)
    # An enforced shortening of DELTA released by an equal stretch is stress free.
    u = _rigid_plus(data, -0.004, 0.0)
    assert abs(srod2(p1, u, deformation=-0.004, real_words=False).sigma) < 1e-9


@pytest.mark.parametrize(
    "mid,deformation,axial",
    [
        (4, -0.004, "tension"),  # ST > 0
        (4, 0.004, "compression"),  # SC > 0
        (6, -0.004, None),  # ST = 0: blank
        (5, 0.004, None),  # SC = 0: blank
        (4, 0.0, None),  # zero stress: blank
    ],
)
def test_axial_margin_branches(mid, deformation, axial):
    data = _rod(mid=mid)
    mat = MATS.mat1(mid)
    s = srod2(srod1(data, MATS, real_words=False), np.zeros(12), None, deformation, False)
    if axial == "tension":
        assert s.sigma > 0 and s.axial_margin == pytest.approx(mat.sigt / s.sigma - 1.0)
    elif axial == "compression":
        assert s.sigma < 0 and s.axial_margin == pytest.approx(-mat.sigc / s.sigma - 1.0)
    else:
        assert s.axial_margin is None
        assert s.stress_words()[2] == NO_MARGIN_WORD
    assert s.torsional_margin is None  # no twist


def test_torsional_margin_and_zero_j():
    u = np.zeros(12)
    u[9:12] = [1e-3, -2e-3, 4e-3]
    for mid, blank in ((4, False), (5, True), (6, False)):
        s = srod2(srod1(_rod(mid=mid), MATS), u)
        assert (s.torsional_margin is None) == blank
    no_j = srod1(_rod(j=0.0), MATS)
    assert no_j.c_over_j == 0.0 and not no_j.sar.any() and not no_j.sbr.any()
    s = srod2(no_j, u)
    assert s.tau == 0.0 and s.torque == 0.0 and s.torsional_margin is None


def test_crod_and_conrod_share_one_kernel():
    text = """ID T,T
CEND
BEGIN BULK
GRID    1               0.      0.      0.
GRID    2               3.      4.      12.
MAT1    4       1.04+7  4.+6            2.5-4   1.3-5   68.             +M
+M      4.2+4   3.6+4   2.5+4
CROD    11      5       1       2
PROD    5       4       .8      .3      .6
CONROD  12      1       2       4       .8      .3      .6
ENDDATA
"""
    model, ecpts = read_ecpts(text)
    datas = [r.data for r in ecpts["ROD"]]
    assert [d.eid for d in datas] == [11, 12]
    u = np.linspace(-1e-3, 2e-3, 12)
    crod, conrod = (srod2(srod1(d, model.materials), u) for d in datas)
    assert (crod.sigma, crod.tau, crod.force, crod.torque) == (
        conrod.sigma,
        conrod.tau,
        conrod.force,
        conrod.torque,
    )
    assert crod.axial_margin == conrod.axial_margin is not None


def test_errors():
    with pytest.raises(RodGeometryError):
        srod1(_rod(b=(1.0, -2.0, 0.5)), MATS)
    with pytest.raises(RodGeometryError):
        rod_stress_matrices([_rod(), _rod(b=(1.0, -2.0, 0.5))], MATS)
    with pytest.raises(KeyError):
        srod1(_rod(cids=(99, 0)), MATS, CSTM)
    with pytest.raises(KeyError):
        rod_stress_matrices([_rod(cids=(0, 99))], MATS, CSTM)


def test_batch_temperatures_and_deformations_by_eid():
    datas = [
        _rod(),
        LAYOUTS["ROD"].parse(
            [12, 7, 13, 6, 1.1, 0.2, 0.1, 0.0, 0, 0.0, 0.0, 0.0, 0, 2.0, 0.0, 0.0, 0.0]
        ),
    ]
    batch = rod_stress_matrices(datas, MATS)
    u = np.linspace(-1e-3, 1e-3, 20)
    res = rod_stresses(batch, u, {11: 150.0}, {12: 0.002})
    ref = [srod2(srod1(datas[0], MATS), u, 150.0), srod2(srod1(datas[1], MATS), u, None, 0.002)]
    np.testing.assert_array_equal(res.sigma, [r.sigma for r in ref])
    np.testing.assert_array_equal(res.force, [r.force for r in ref])


# -- listing parity -----------------------------------------------------------------

ROD_TYPES = ("CROD", "CONROD")


def _deck_parity():
    """Every rod stress/force table of every golden deck vs the port fed its displacements.

    Yields ``(deck, key, comparison, actual, table)`` for tables whose rods have
    both grids in the printed displacement set; ``(deck, key, None, None, table)``
    for tables that cannot be recomputed from the listing.
    """
    for deck in load_manifest()["decks"]:
        golden = tables(load_golden(deck["id"]))
        rod_tables = {
            k: t
            for k, t in golden.items()
            if t.meta["kind"] in ("stress", "force") and t.meta["element_type"] in ROD_TYPES
        }
        if not rod_tables:
            continue
        text = (repo_root() / deck["inp"]).read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        records = {r.data.eid: r for r in ecpts["ROD"]}
        for key, table in rod_tables.items():
            _, _, subcase, occ = key.split("/")[:4]
            disp = golden[f"displacement/grid/{subcase}/{occ}"]
            printed = disp.by_id()
            if not all(g in printed for e in table.ids for g in records[int(e)].grid_ids):
                yield deck["id"], key, None, None, table
                continue
            u = np.zeros(max(model.sil.values()) + 6)
            for gid, row in printed.items():
                u[model.sil[gid] - 1 : model.sil[gid] + 5] = np.nan_to_num(row)
            datas = [records[int(e)].data for e in table.ids]
            res = rod_stresses(rod_stress_matrices(datas, model.materials, model.cstm), u)
            actual = res.stress_table() if table.meta["kind"] == "stress" else res.force_table()
            yield deck["id"], key, compare(table, actual), actual, table


def _column_groups(table: GoldenTable) -> dict[str, list[str]]:
    margins = [c for c in table.columns if c.endswith("margin")]
    values = [c for c in table.columns if c not in margins]
    kind = table.meta["kind"]
    return {kind: values, "margins": margins} if margins else {kind: values}


def _columns(table: GoldenTable, cols: list[str]) -> GoldenTable:
    """``table`` restricted to ``cols`` (their absolute floors are unchanged)."""
    idx = [table.columns.index(c) for c in cols]
    return GoldenTable(table.key, tuple(cols), table.ids, table.values[:, idx], table.meta)


def test_rod_tables_match_listings():
    by_deck: dict[str, list] = {}
    by_group: dict[tuple[str, str], list] = {}
    skipped = []
    for deck, key, cmp, actual, table in _deck_parity():
        if cmp is None:
            skipped.append(f"{deck} {key}")
            continue
        assert cmp.passed, (deck, key, cmp)
        # Blank margins must stay blank; printed ones match to the 2 digits printed.
        assert np.array_equal(np.isnan(actual), np.isnan(table.values)), (deck, key)
        for col in ("axial_margin", "torsional_margin"):
            if col in table.columns:
                i = table.columns.index(col)
                for got, want in zip(actual[:, i], table.values[:, i], strict=True):
                    if not np.isnan(want):
                        assert f"{got:.1E}" == f"{want:.1E}", (deck, key, got, want)
        by_deck.setdefault(deck, []).append(cmp)
        for group, cols in _column_groups(table).items():
            sub = _columns(table, cols)
            c = compare(sub, actual[:, [table.columns.index(k) for k in cols]])
            by_group.setdefault((deck, group), []).append(c)
    for (deck, group), cmps in sorted(by_group.items()):
        record_parity(
            f"rod {group} {deck} vs listing",
            sum(c.n for c in cmps),
            max(c.max_abs for c in cmps),
            max(c.max_rel for c in cmps),
        )
    # d01161a prints CROD stresses (3 design iterations); d01013a prints CONROD forces.
    # Its CROD posts end on lower-surface grids outside DISPLACEMENTS = 1.
    assert set(by_deck) == {"d01161a", "d01013a"}
    assert sum(len(c) for c in by_deck.values()) == 4
    assert skipped == ["d01013a force/CROD/sc1/occ1"]


def test_requested_decks_rod_output():
    """d01011a and t01241a print no rod stress or force tables (t01241a's rods: KGGX only)."""
    for deck in ("d01011a", "t01241a"):
        keys = [t["key"] for t in load_golden(deck)["tables"]]
        assert not [k for k in keys if k.split("/")[1] in ROD_TYPES], deck
    assert any(m["name"] == "KGGX" for m in load_golden("t01241a")["matrices"])


def test_every_deck_rod_batch_matches_reference():
    """Smoke test: batched vs reference phase I and II for every RF1 deck rod."""
    total = 0
    for deck in load_manifest()["decks"]:
        text = (repo_root() / deck["inp"]).read_text(errors="replace")
        model, ecpts = read_ecpts(text)
        datas = [r.data for r in ecpts.get("ROD", [])]
        if not datas:
            continue
        batch = rod_stress_matrices(datas, model.materials, model.cstm)
        u = np.random.default_rng(len(datas)).uniform(-1e-3, 1e-3, max(model.sil.values()) + 6)
        res = rod_stresses(batch, u, 123.0, 1e-3)
        for i, d in enumerate(datas):
            p1 = srod1(d, model.materials, model.cstm)
            np.testing.assert_array_equal(batch.element(i).words(), p1.words())
            s = srod2(p1, u, 123.0, 1e-3)
            assert (res.sigma[i], res.tau[i], res.force[i]) == (s.sigma, s.tau, s.force)
        total += len(datas)
    assert total > 0
