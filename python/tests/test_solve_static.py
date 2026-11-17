"""``nastran95py.solve.static``: RF1 static solve smoke parity against the listings.

- t01241a: the assembled ``K_gg`` against the printed KGGX (the deck's DMAP ALTER
  stops before the solve; its only load sits on a DOF with no stiffness).
- d01011a: displacements, SPC forces and element stresses (one subcase).
- d01161a: displacements of both subcases and the subcase 10 stresses (the
  as-input analysis, first fully stressed design occurrence).
- d01032a: OLOAD and displacements of a TEMP-driven thermal subcase.

The full nine-deck comparison is a separate suite.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from nastran95py.reference.golden import compare, load_golden, tables
from nastran95py.reference.inventory import repo_root
from nastran95py.solve import (
    NotSupported,
    SingularStiffness,
    StaticModel,
    assemble_stiffness,
    solve_static,
)

from .conftest import record_parity


def _deck(deck_id: str) -> str:
    return (repo_root() / "inp" / f"{deck_id}.inp").read_text(errors="replace")


def _with_bulk(text: str, *cards: str, drop: tuple[str, ...] = ()) -> str:
    """Deck text with ``cards`` added before ENDDATA and lines starting with ``drop`` removed."""
    lines = [ln for ln in text.splitlines() if not ln.startswith(drop) or not drop]
    end = next(i for i, ln in enumerate(lines) if ln.startswith("ENDDATA"))
    return "\n".join([*lines[:end], *cards, *lines[end:]]) + "\n"


def _with_case(text: str, *lines: str) -> str:
    return text.replace("CEND", "\n".join(["CEND", *lines]), 1)


def _compare_listing(deck_id: str, solution, keys: list[str]) -> dict:
    golden = tables(load_golden(deck_id))
    out = {}
    for key in keys:
        table = golden[key]
        kind, what, sc = key.split("/")[:3]
        res = solution.subcases[int(sc[2:])]
        if what == "grid":
            vec = {"displacement": res.displacement, "spcforce": res.spc_force, "oload": res.load}
            actual = res.grid_rows(vec[kind], table.ids)
        else:
            src = res.stresses if kind == "stress" else res.forces
            actual = src[what].rows(int(e) for e in table.ids)
        cmp = compare(table, actual)
        assert cmp.n > 0 and cmp.failures == 0, (deck_id, key, cmp)
        out[key] = cmp
    return out


def _record(label: str, cmps) -> None:
    cmps = list(cmps)
    record_parity(
        label,
        sum(c.n for c in cmps),
        max(c.max_abs for c in cmps),
        max(c.max_rel for c in cmps),
    )


# -- listing parity ------------------------------------------------------------------


def test_t01241a_stiffness_matches_kggx():
    sm = StaticModel.from_text(_deck("t01241a"))
    k = assemble_stiffness(sm).toarray()
    (kggx,) = load_golden("t01241a")["matrices"]
    assert (kggx["rows"], kggx["cols"]) == k.shape == (24, 24)
    ref = np.zeros_like(k)
    for r, c, v in kggx["entries"]:
        ref[r - 1, c - 1] = v
    assert np.array_equal(k != 0.0, ref != 0.0)
    err = np.abs(k - ref)
    rel = err.max() / np.abs(ref).max()
    assert rel <= 1e-6, rel
    nz = ref != 0.0
    record_parity(
        "static t01241a K_gg vs KGGX", int(nz.sum()), err.max(), (err[nz] / abs(ref[nz])).max()
    )


def test_t01241a_load_on_unstiff_dof_is_rejected():
    # FORCE 1 is along Z at grid 4, but the three CRODs only carry axial X load.
    with pytest.raises(SingularStiffness, match=r"\(4, 3\)"):
        solve_static(_deck("t01241a"))


def test_d01011a_matches_listing():
    keys = [
        "displacement/grid/sc1/occ1",
        "spcforce/grid/sc1/occ1",
        "stress/CQDMEM/sc1/occ1",
        "stress/CSHEAR/sc1/occ1",
        "stress/CTRMEM/sc1/occ1",
    ]
    cmps = _compare_listing("d01011a", solve_static(_deck("d01011a")), keys)
    _record("static d01011a displacements", [cmps[keys[0]]])
    _record("static d01011a SPC forces", [cmps[keys[1]]])
    _record("static d01011a stresses", [cmps[k] for k in keys[2:]])


def test_d01161a_both_subcases_match_listing():
    sol = solve_static(_deck("d01161a"))
    assert sorted(sol.subcases) == [10, 12]
    assert sol.model.material_set == 3000 and sol.model.thermal_sets == {10: None, 12: None}
    disp = ["displacement/grid/sc10/occ1", "displacement/grid/sc12/occ1"]
    stress = ["stress/CQDMEM/sc10/occ1", "stress/CROD/sc10/occ1", "stress/CTRMEM/sc10/occ1"]
    cmps = _compare_listing("d01161a", sol, disp + stress)
    _record("static d01161a displacements", [cmps[k] for k in disp])
    _record("static d01161a stresses sc10", [cmps[k] for k in stress])


def test_d01032a_thermal_subcase_matches_listing():
    sol = solve_static(_deck("d01032a"))
    assert sol.model.thermal_sets == {1: 1} and sol.model.material_set == 1
    keys = ["oload/grid/sc1/occ1", "displacement/grid/sc1/occ1"]
    cmps = _compare_listing("d01032a", sol, keys)
    _record("static d01032a OLOAD+displacements", cmps.values())


# -- loads and constraints -----------------------------------------------------------


def test_spc_forces_balance_applied_loads():
    sol = solve_static(_deck("d01011a"))
    res = sol.subcases[1]
    k = sol.stiffness
    # K u = P + q everywhere; the reactions balance the 500 lb trailing-edge load.
    assert np.abs(k @ res.displacement - res.load - res.spc_force).max() < 1e-6
    tz = res.load[2::6].sum() + res.spc_force[2::6].sum()
    assert abs(tz) < 1e-6 and res.load[2::6].sum() == pytest.approx(-500.0)


def test_load_combination_card():
    text = _deck("d01011a")
    base = {sid: solve_static(re.sub(r"LOAD\s*=\s*1", f"LOAD = {sid}", text)) for sid in (1, 2)}
    combo = _with_bulk(re.sub(r"LOAD\s*=\s*1", "LOAD = 5", text), "LOAD,5,2.,1.,1,-.5,2")
    u = solve_static(combo).subcases[1].displacement
    expect = 2.0 * (base[1].subcases[1].displacement - 0.5 * base[2].subcases[1].displacement)
    assert np.allclose(u, expect, rtol=1e-12, atol=1e-15)


def _bar(
    *extra: str, ps: str = "456", spc: str = "SPC1,1,123,1", prod: str = "PROD,5,6,2.1"
) -> str:
    """t01241a's three CRODs (E 1.04e7, G 4e6, A 2.1) with replaced loads/constraints."""
    text = _deck("t01241a").replace("GRDSET                                                  456",
                                    f"GRDSET,,,,,,,{ps}")  # fmt: skip
    return _with_bulk(text, spc, prod, *extra, drop=("FORCE", "SPC1", "PROD"))


def test_axial_force_torque_and_load_coordinate_system():
    # CORD2R 1: local z = basic z, local x = basic y, so local -y is basic +x.
    text = _bar(
        "CORD2R,1,,0.,0.,0.,0.,0.,1.,+C1",
        "+C1,0.,1.,0.",
        "FORCE,1,4,1,100.,0.,-1.,0.",
        "MOMENT,1,4,,30.,1.,0.,0.",
        ps="56",
        spc="SPC1,1,1234,1",
        prod="PROD,5,6,2.1,0.5",
    )
    sol = solve_static(text)
    res = sol.subcases[1]
    sil4 = sol.model.model.sil[4]
    u4 = res.displacement[sil4 - 1 : sil4 + 5]
    assert u4[0] == pytest.approx(100.0 * 50.0 / (1.04e7 * 2.1), rel=1e-6)
    assert u4[3] == pytest.approx(30.0 * 50.0 / (4.0e6 * 0.5), rel=1e-6)
    assert np.allclose(u4[1:3], 0.0) and u4[4:] == pytest.approx([0.0, 0.0])
    # T2/T3 of grids 2-4 have no stiffness and no load: constrained, as GPSP does.
    assert set(res.singular) == {(g, c) for g in (2, 3, 4) for c in (2, 3)}
    sil1 = sol.model.model.sil[1]
    assert res.spc_force[sil1 - 1] == pytest.approx(-100.0)
    assert res.spc_force[sil1 + 2] == pytest.approx(-30.0)
    rod = res.forces["CROD"]
    assert rod.values[:, 0] == pytest.approx([100.0] * 3, rel=1e-6)
    assert rod.values[:, 1] == pytest.approx([30.0] * 3, rel=1e-6)


def test_enforced_spc_displacement():
    text = _bar("SPC,1,4,1,.01", spc="SPC1,1,123,1")
    text = text.replace("    LOAD = 1", "")
    sol = solve_static(text)
    res, sil = sol.subcases[1], sol.model.model.sil
    k = 1.04e7 * 2.1
    # Three springs in series (lengths 10, 20, 20) stretched 0.01 in total.
    force = 0.01 / (10.0 / k + 20.0 / k + 20.0 / k)
    assert res.displacement[sil[4] - 1] == pytest.approx(0.01)
    assert res.displacement[sil[2] - 1] == pytest.approx(force * 10.0 / k, rel=1e-9)
    assert res.spc_force[sil[4] - 1] == pytest.approx(force, rel=1e-9)


_ROD = """ID    SOLVE,TEST
APP   DISPLACEMENT
SOL   1,1
TIME  5
CEND
SPC = 1
LOAD = 1
BEGIN BULK
GRDSET,,,,,,,3456
GRID,1,,0.,0.,0.
GRID,2,,2.,3.,0.
MAT1,1,1.+7,,.3
CROD,5,15,1,2
PROD,15,1,.2
SPC1,1,12,1
FORCE,1,2,,1.,0.,1.,0.
{extra}
ENDDATA
"""


def test_diagonal_rod_mechanism_is_detected():
    # Grid 2 has only the rod's axial stiffness along (2, 3): K_ff is rank 1, but its
    # float32 assembly leaves a tiny nonzero pivot that a residual test accepts.
    with pytest.raises(SingularStiffness, match=r"mechanism\) at .*\[\(2, [12]\)\]"):
        solve_static(_ROD.format(extra=""))
    # Supporting grid 2 across the rod (SPC on T1) makes it determinate again.
    res = solve_static(_ROD.format(extra="SPC1,1,1,2")).subcases[1]
    axial = 1.0e7 * 0.2 / np.sqrt(13.0)
    # Load (0, 1) with T1 fixed: rod force N = 1 / sin, u_axial = N / k, u2 = u_axial / sin.
    sin = 3.0 / np.sqrt(13.0)
    assert res.displacement[7] == pytest.approx(1.0 / sin / axial / sin, rel=1e-6)


def test_floating_rod_pair_mechanism_is_detected():
    # Grids 3-4 are joined by an X rod only: K_ff = k [[1, -1], [-1, 1]], an exactly
    # zero pivot; both DOFs of the rigid-body mode are named.
    extra = "GRID,3,,5.,0.,0.\nGRID,4,,7.,0.,0.\nCROD,6,15,3,4\nSPC1,1,1,2\nSPC1,1,2,3,4"
    with pytest.raises(SingularStiffness, match=r"\(3, 1\), \(4, 1\)"):
        solve_static(_ROD.format(extra=extra))


# -- thermal loads through every element kernel ---------------------------------------

_PLATE = """ID    SOLVE,TEST
APP   DISPLACEMENT
SOL   1,1
TIME  5
CEND
TITLE = UNIFORM THERMAL EXPANSION
SPC = 1
TEMPERATURE(LOAD) = 7
BEGIN BULK
GRDSET,,,,,,,3456
GRID,1,,0.,0.,0.
GRID,2,,2.,0.,0.
GRID,3,,2.5,1.,0.
GRID,4,,0.,1.2,0.
MAT1,1,1.+7,,.3,.1,1.-5,50.
{elements}
TEMPD,7,150.
SPC1,1,12,1
SPC1,1,2,2
ENDDATA
"""

_ELEMENTS = {
    "CQDMEM": "CQDMEM,1,11,1,2,3,4\nPQDMEM,11,1,.1",
    "CQDMEM1": "CQDMEM1,2,12,1,2,3,4,30.\nPQDMEM1,12,1,.1",
    "CQDMEM2": "CQDMEM2,3,13,1,2,3,4,15.\nPQDMEM2,13,1,.1",
    "CTRMEM": "CTRMEM,4,14,1,2,3\nPTRMEM,14,1,.1",
    # Rods along grid 1-2 (X): T2 of grid 2 is supported, so they stay determinate.
    "CROD": "CROD,5,15,1,2\nPROD,15,1,.2",
    "CONROD": "CONROD,6,1,2,1,.3",
}


@pytest.mark.parametrize("card", sorted(_ELEMENTS))
def test_uniform_temperature_is_stress_free(card):
    """Statically determinate support + uniform DT: u = alpha DT x and zero stress."""
    sol = solve_static(_PLATE.format(elements=_ELEMENTS[card]))
    res = sol.subcases[1]
    p = res.load
    assert np.abs(p).max() > 1.0
    assert abs(p[0::6].sum()) < 1e-6 * np.abs(p).max()
    assert abs(p[1::6].sum()) < 1e-6 * np.abs(p).max()
    model = sol.model.model
    used = {g for r in next(iter(sol.model.ecpts.values())) for g in r.grid_ids}
    for gid in used:
        s = model.sil[gid]
        expect = 1.0e-5 * 100.0 * model.grids[gid].basic[:2]
        assert np.allclose(res.displacement[s - 1 : s + 1], expect, rtol=0, atol=1e-8), gid
    stress = res.stresses[card].values
    cols = [0] if card in ("CROD", "CONROD") else [0, 1, 2]
    assert np.abs(stress[:, cols]).max() < 1e-3 * 1.0e7 * 1.0e-5 * 100.0
    assert np.abs(res.spc_force).max() < 1e-6 * np.abs(p).max()


def test_shear_panel_thermal_load_is_self_equilibrated():
    text = _PLATE.format(elements="CSHEAR,7,17,1,2,3,4\nPSHEAR,17,1,.1\n" + _ELEMENTS["CQDMEM"])
    res = solve_static(text).subcases[1]
    assert set(res.stresses) == {"CSHEAR", "CQDMEM"} and "CSHEAR" in res.forces
    p = res.load
    assert abs(p[0::6].sum()) < 1e-6 * np.abs(p).max()
    assert abs(p[1::6].sum()) < 1e-6 * np.abs(p).max()
    assert np.isfinite(res.displacement).all()


# -- scope -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "card",
    ["MPC,3,16,3,1.,36,3,-1.", "OMIT,16,3", "ASET1,3,16", "SUPORT,16,3", "SPCD,1,16,3,.1",
     "DEFORM,1,60,.01", "PLOAD2,1,1.,1"],
)  # fmt: skip
def test_unsupported_bulk_cards_raise(card):
    with pytest.raises(NotSupported, match=card.split(",")[0]):
        solve_static(_with_bulk(_deck("d01011a"), card))


@pytest.mark.parametrize("line", ["MPC = 3", "DEFORM = 1", "SUBCOM 3"])
def test_unsupported_case_control_raises(line):
    with pytest.raises(NotSupported):
        StaticModel.from_text(_with_case(_deck("d01011a"), line))


def test_undefined_sets_raise():
    from nastran95py.bulk import BulkDataError

    with pytest.raises(BulkDataError, match="load set 9"):
        solve_static(re.sub(r"LOAD\s*=\s*1", "LOAD = 9", _deck("d01011a")))
    with pytest.raises(BulkDataError, match="SPC set 9"):
        solve_static(re.sub(r"SPC\s*=\s*1", "SPC = 9", _deck("d01011a")))
