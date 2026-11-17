"""``nastran95py.solve.static``: RF1 static solve parity against the golden listings.

Every table of every inventoried RF1 deck (``tests/reference/golden/``) is solved
and compared at the baseline tolerance (rtol 1e-4, atol 1e-4*max|column|, 5e-2 for
margins of safety): displacements, SPC forces, OLOAD, element forces and stresses.
A failure lists deck, subcase, grid/element id and component of every bad value.

- t01241a: no solution tables (its DMAP ALTER stops before the solve); the
  assembled ``K_gg`` is checked against the printed KGGX and the solve must raise
  ``SingularStiffness`` for the load on the unstiffened DOF (4, 3).
- d01161a: only the as-input analysis (fully stressed design occurrence 1); the
  resized-design occurrences 2 and 3 are out of scope.
- d01013a: six CROD post axial forces are one SROD2 REAL rounding step off and are
  strict xfails at the baseline tolerance, plus an exact one-step check (see
  ``D01013A_QUANTUM_ROWS``).
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from nastran95py.elements.rod_stress import rod_stress_matrices, rod_stresses
from nastran95py.reference.golden import compare, load_golden, tables, tolerances
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


DECKS = (
    "d01011a", "d01012a", "d01013a", "d01014a", "d01032a",
    "d01033a", "d01034a", "d01161a", "t01241a",
)  # fmt: skip
"""Every RF1 deck in the reference manifest (``tests/reference/golden/``)."""

KGGX_ONLY = {"t01241a"}
"""Decks whose listing has no solution tables (t01241a's ALTER exits before SSG3)."""

FIRST_OCCURRENCE_ONLY = {"d01161a"}
"""d01161a prints three fully stressed design iterations; only the as-input
analysis (``occ1``) is solved, POPT/PLIMIT resizing is out of scope."""

# d01013a vertical CROD posts whose listed axial force is one REAL rounding
# quantum away from ours. SROD2 forms SIGMA = TRANA + TRANB, two REAL dot products
# SAT.u_a and SBT.u_b of about 1e6 that cancel to a few psi, so the printed force
# moves in discrete REAL steps (0.525 lb for EID 63). Our
# float64 displacements are solver-independent at float32 resolution (dense LU,
# Cholesky and SuperLU with three orderings round to the same REAL vector; cond
# K_ff = 5.9e5), yet each row is reproduced exactly by moving one end's u3 by one
# float32 ulp (about 2e-8 relative, below the listing's 7 printed digits); for
# EID 63 the listing itself prints grid 14 T3 as -7.825613E-02, one ulp below our
# -7.825612E-02. Rounding every bulk REAL field to float32, and truncating instead
# of rounding the displacements, were both tried and do not move these rows.
D01013A_QUANTUM_ROWS = (63, 64, 65, 66, 77, 78)
D01013A_POST_FORCES = "force/CROD/sc1/occ1"

_SOLUTIONS: dict[str, object] = {}


def _solution(deck_id: str):
    if deck_id not in _SOLUTIONS:
        _SOLUTIONS[deck_id] = solve_static(_deck(deck_id))
    return _SOLUTIONS[deck_id]


def _listing_keys(deck_id: str) -> list[str]:
    keys = list(tables(load_golden(deck_id)))
    if deck_id in FIRST_OCCURRENCE_ONLY:
        keys = [k for k in keys if k.split("/")[3] == "occ1"]
    return keys


def _actual(deck_id: str, key: str, table) -> np.ndarray:
    kind, what, sc = key.split("/")[:3]
    res = _solution(deck_id).subcases[int(sc[2:])]
    if what == "grid":
        vec = {"displacement": res.displacement, "spcforce": res.spc_force, "oload": res.load}
        return np.asarray(res.grid_rows(vec[kind], table.ids), dtype=float)
    src = res.stresses if kind == "stress" else res.forces
    return np.asarray(src[what].rows(int(e) for e in table.ids), dtype=float)


def _mismatches(deck_id: str, table, actual: np.ndarray, skip=(), floor=None) -> list[str]:
    """One line per out-of-tolerance value: deck, subcase, grid/element id, component."""
    rtol, atol = tolerances(table)
    if floor is not None:
        atol = floor
    g = table.values
    err = np.abs(actual - g)
    bad = ~np.isnan(g) & ~(err <= atol + rtol * np.abs(g))
    what = table.meta["element_type"] or "grid"
    out = []
    for i, j in zip(*np.nonzero(bad), strict=True):
        if int(table.ids[i]) in skip:
            continue
        out.append(
            f"{deck_id} subcase {table.meta['subcase']} {table.meta['kind']} {what} "
            f"{int(table.ids[i])} {table.columns[j]}: golden {g[i, j]:.7g} actual "
            f"{actual[i, j]:.7g} (|err| {err[i, j]:.3g} > {atol[j] + rtol[j] * abs(g[i, j]):.3g})"
        )
    return out


def _without_rows(table, ids):
    """``table`` with the rows of ``ids`` blanked (NaN) so ``compare`` skips them."""
    vals = table.values.copy()
    vals[np.isin(table.ids, list(ids))] = np.nan
    return replace(table, values=vals)


_LISTING_CASES = [(d, k) for d in DECKS if d not in KGGX_ONLY for k in _listing_keys(d)]


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
        "static t01241a K_gg vs KGGX", int(nz.sum()), err.max(), (err[nz] / abs(ref[nz])).max(),
        deck="t01241a", table="kggx",
    )  # fmt: skip


def test_t01241a_load_on_unstiff_dof_is_rejected():
    # FORCE 1 is along Z at grid 4, but the three CRODs only carry axial X load.
    with pytest.raises(SingularStiffness, match=r"\(4, 3\)"):
        solve_static(_deck("t01241a"))


@pytest.mark.parametrize(
    ("deck_id", "key"), _LISTING_CASES, ids=[f"{d}:{k}" for d, k in _LISTING_CASES]
)
def test_reference_deck_matches_golden(deck_id, key):
    """Every golden table of every solvable RF1 deck at the baseline tolerance."""
    table = tables(load_golden(deck_id))[key]
    actual = _actual(deck_id, key, table)
    skip = D01013A_QUANTUM_ROWS if (deck_id, key) == ("d01013a", D01013A_POST_FORCES) else ()
    lines = _mismatches(deck_id, table, actual, skip)
    assert not lines, "\n".join([f"{len(lines)} value(s) out of tolerance:", *lines])
    cmp = compare(_without_rows(table, skip) if skip else table, actual)
    assert cmp.n > 0 and cmp.failures == 0, cmp
    record_parity(f"{deck_id} {key}", cmp.n, cmp.max_abs, cmp.max_rel, deck=deck_id, table=key)


def test_every_reference_deck_is_covered():
    assert {d for d, _ in _LISTING_CASES} | KGGX_ONLY == set(DECKS)
    assert {k.split("/")[0] for _, k in _LISTING_CASES} == {
        "displacement", "spcforce", "oload", "force", "stress",
    }  # fmt: skip
    for d in set(DECKS) - KGGX_ONLY:
        assert any(k.startswith("displacement/") for k in _listing_keys(d)), d
    assert tables(load_golden("t01241a")) == {}
    # d01161a: the resized-design occurrences 2 and 3 are parsed but not compared.
    assert {k.split("/")[3] for k in tables(load_golden("d01161a"))} == {"occ1", "occ2", "occ3"}
    assert {k.split("/")[3] for k in _listing_keys("d01161a")} == {"occ1"}


def _one_ulp_post_forces(deck_id: str, eids) -> dict[int, list[float]]:
    """SROD2 axial forces after moving one end's T3 by one float32 ulp (4 per rod)."""
    sol = _solution(deck_id)
    sm = sol.model
    batch = rod_stress_matrices(
        [r.data for r in sm.ecpts["ROD"]], sm.model.materials, sm.model.cstm
    )
    u = sol.subcases[1].displacement
    out: dict[int, list[float]] = {}
    for i, eid in enumerate(batch.eid):
        if int(eid) not in eids:
            continue
        out[int(eid)] = []
        for sil in batch.sil[i]:
            k = int(sil) + 1  # T3 of the end's first SIL
            for toward in (-np.inf, np.inf):
                up = u.copy()
                up[k] = float(np.nextafter(np.float32(u[k]), np.float32(toward)))
                out[int(eid)].append(float(rod_stresses(batch, up).force[i]))
    return out


_QUANTUM_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="d01013a CROD post: SROD2 REAL cancellation of TRANA+TRANB moves the axial "
    "force in REAL steps (|err| 0.027-1.75 lb, rel <= 7.7%); listing is one step "
    "off, from a 1-ulp (~2e-8 rel) difference in one end's u3",
)


@pytest.mark.parametrize(
    "eid", [pytest.param(e, marks=_QUANTUM_XFAIL) for e in D01013A_QUANTUM_ROWS]
)
def test_d01013a_post_axial_force_at_baseline_tolerance(eid):
    table = tables(load_golden("d01013a"))[D01013A_POST_FORCES]
    actual = _actual("d01013a", table.key, table)
    sel = table.ids == eid
    rows = replace(table, ids=table.ids[sel], values=table.values[sel])
    lines = _mismatches("d01013a", rows, actual[sel], floor=tolerances(table)[1])
    assert not lines, "\n".join(lines)


def test_d01013a_post_axial_forces_are_one_real_quantum_off():
    """The six rows the main suite skips are matched once one end's T3 moves 1 ulp."""
    table = tables(load_golden("d01013a"))[D01013A_POST_FORCES]
    actual = _actual("d01013a", table.key, table)
    rtol, atol = tolerances(table)
    nudged = _one_ulp_post_forces("d01013a", D01013A_QUANTUM_ROWS)
    rows = [int(np.flatnonzero(table.ids == e)[0]) for e in D01013A_QUANTUM_ROWS]
    for eid, i in zip(D01013A_QUANTUM_ROWS, rows, strict=True):
        gold = table.values[i, 0]
        hits = [f for f in nudged[eid] if abs(f - gold) <= atol[0] + rtol[0] * abs(gold)]
        assert hits, (eid, gold, actual[i, 0], nudged[eid])
        assert actual[i, 1] == table.values[i, 1] == 0.0
    gold = table.values[rows, 0]
    dev = np.abs(actual[rows, 0] - gold)
    record_parity(
        "d01013a CROD posts (1-ulp u3)", len(dev), dev.max(), (dev / np.abs(gold)).max(),
        deck="d01013a", table="quantum-rows",
    )  # fmt: skip


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
