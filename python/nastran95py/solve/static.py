"""Minimal Rigid Format 1 static solution.

What SMA1, GP4/SCE1, SSG1, SSG2/SSG3 and SDR1/SDR2 do for a RF1 run, without the
DMAP/GINO executive or grid point resequencing:

1. read the deck and build the ECPT records at the problem's material
   temperature set (``TEMPERATURE(MATERIAL|BOTH)``; none: plain MAT1);
2. assemble ``K_gg`` (SciPy sparse, float64) from the ported element kernels,
   which keep their REAL (float32) emulation internally;
3. constrain the permanent single-point constraints (GRID/GRDSET ``PS``) and the
   subcase's SPC/SPC1 set; DOFs with no stiffness at all and no load are
   constrained as GPSP does, a loaded one raises :class:`SingularStiffness`;
4. build the load vector from LOAD combinations of FORCE/MOMENT sets plus the
   element thermal loads of the subcase's ``TEMPERATURE(LOAD|BOTH)`` set
   (TEMP/TEMPD);
5. solve ``K_ff u_f = P_f - K_fs u_s`` with ``scipy.sparse.linalg.spsolve``;
6. recover the single-point forces ``q_s = K_sg u_g - P_s`` and the element
   stresses and forces.

The scope is the cards the RF1 reference decks use. MPC, OMIT/ASET, SUPORT,
SPCD, DEFORM and every card the bulk reader does not interpret raise
:class:`NotSupported`; PARAM, CNGRNT and the fully stressed design cards
POPT/PLIMIT are accepted and ignored (only the as-input analysis is solved).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from nastran95py.bulk import (
    BulkDataError,
    BulkModel,
    EcptRecord,
    build_ecpts,
    load_grid_temperatures,
    load_temperatures,
    material_temperature_set,
)
from nastran95py.bulk.model import GRID_DOF
from nastran95py.coords import transforms_to_basic
from nastran95py.elements import (
    Qdmem2Batch,
    QdmemBatch,
    RodBatch,
    RodLoadBatch,
    ShearBatch,
    ShearLoadBatch,
    TrmemBatch,
    load_vector,
    qdmem1_matrices,
    qdmem1_stress_table,
    qdmem1_stresses,
    qdmem1_thermal_loads,
    qdmem2_matrices,
    qdmem2_stress_matrices,
    qdmem2_stresses,
    qdmem2_thermal_loads,
    qdmem_matrices,
    qdmem_stress_matrices,
    qdmem_stresses,
    qdmem_thermal_loads,
    rod_loads,
    rod_matrices,
    rod_stress_matrices,
    rod_stresses,
    shear_loads,
    shear_matrices,
    trmem_matrices,
    trmem_stress_matrices,
    trmem_stresses,
    trmem_thermal_loads,
)
from nastran95py.elements.shear_stress import (
    ShearStressBatch,
    shear_stress_phase1,
    shear_stress_phase2,
)
from nastran95py.reference.cards import Card, Deck, read_deck
from nastran95py.reference.casecontrol import _ASSIGN, _logical_lines, parse_case_control

FloatArray = NDArray[np.float64]

ACCEPTED_CARDS = frozenset({"PARAM", "CNGRNT", "POPT", "PLIMIT", "LOAD"})
"""Bulk cards the bulk reader leaves uninterpreted that the static solve accepts."""

UNSUPPORTED_CASE_CONTROL = ("MPC", "DEFO", "SUBCOM", "SYMCOM", "REPCASE", "SYMSEQ")


class NotSupported(NotImplementedError):
    """Input outside the minimal RF1 static scope (MPC, OMIT/ASET, SUPORT, SPCD, ...)."""


class SingularStiffness(ValueError):
    """``K_ff`` is singular: a loaded DOF without stiffness, or a mechanism."""


@dataclass(frozen=True)
class ElementTable:
    """Recovered element stresses or forces of one connection card type, rows by EID."""

    card: str
    eid: NDArray[np.int64]
    values: FloatArray

    def rows(self, ids: Iterable[int]) -> FloatArray:
        index = {int(e): i for i, e in enumerate(self.eid)}
        return self.values[[index[int(e)] for e in ids]]


@dataclass(frozen=True)
class StaticModel:
    """A deck prepared for :func:`solve_static`: bulk model, ECPTs and case control."""

    deck: Deck
    model: BulkModel
    ecpts: dict[str, list[EcptRecord]]
    case_control: dict
    material_set: int | None
    thermal_sets: dict[int, int | None]
    """Subcase id -> TEMPERATURE(LOAD|BOTH) set (``None``: no thermal load)."""

    @classmethod
    def from_text(cls, text: str) -> StaticModel:
        deck = read_deck(text)
        _check_case_control(deck.case_control)
        model = BulkModel.from_deck(deck)
        unsupported = sorted(set(model.ignored) - ACCEPTED_CARDS)
        if unsupported:
            raise NotSupported(f"bulk data cards not supported by the static solve: {unsupported}")
        material_set = material_temperature_set(deck.case_control)
        return cls(
            deck,
            model,
            build_ecpts(model, material_set),
            parse_case_control(deck.case_control),
            material_set,
            _thermal_load_sets(deck.case_control),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> StaticModel:
        return cls.from_text(Path(path).read_text(errors="replace"))

    @property
    def ndof(self) -> int:
        m = self.model
        return max((s - 1 + (GRID_DOF if p in m.grids else 1) for p, s in m.sil.items()), default=0)

    def dof(self, point: int, component: int) -> int:
        """0-based g-set index of ``component`` (1-6 for grids, 0 for scalar points)."""
        m = self.model
        if point in m.grids:
            if not 1 <= component <= GRID_DOF:
                raise BulkDataError(f"grid {point}: illegal component {component}")
            return m.sil[point] - 1 + component - 1
        if point in m.spoints:
            if component not in (0, 1):
                raise BulkDataError(f"scalar point {point}: illegal component {component}")
            return m.sil[point] - 1
        raise BulkDataError(f"undefined grid or scalar point {point}")


@dataclass
class SubcaseResult:
    subcase: int
    load: FloatArray
    """``P_g``: applied plus thermal loads (SIL order, grid displacement axes)."""
    displacement: FloatArray
    """``u_g`` in each grid's displacement coordinate system."""
    spc_force: FloatArray
    """``q_g``: single-point forces of constraint, zero at the free DOFs."""
    stresses: dict[str, ElementTable] = field(default_factory=dict)
    forces: dict[str, ElementTable] = field(default_factory=dict)
    singular: tuple[tuple[int, int], ...] = ()
    """``(point, component)`` without stiffness that were constrained automatically."""
    sil: Mapping[int, int] = field(default_factory=dict, repr=False)

    def grid_rows(self, vector: FloatArray, ids: Iterable[int]) -> FloatArray:
        """``(n, 6)`` rows of a g-set vector for grid ids (``T1..R3``, as printed)."""
        return np.array([vector[self.sil[int(g)] - 1 : self.sil[int(g)] + 5] for g in ids])


@dataclass
class StaticSolution:
    model: StaticModel
    stiffness: sp.csr_matrix
    """``K_gg`` (SIL order, unconstrained)."""
    subcases: dict[int, SubcaseResult]


# -- input checks -----------------------------------------------------------------


def _case_lines(lines: list[str]) -> Iterable[str]:
    for line in _logical_lines(lines):
        head = line.split("$", 1)[0].strip().upper()
        if head.startswith("OUTPUT") and ("PLOT" in head or "XY" in head):
            return
        yield head


def _check_case_control(lines: list[str]) -> None:
    for head in _case_lines(lines):
        m = _ASSIGN.match(head)
        word = m[1] if m else head.split(" ", 1)[0]
        if any(word.startswith(w) for w in UNSUPPORTED_CASE_CONTROL):
            raise NotSupported(f"case control {head!r} is not supported by the static solve")


def _thermal_load_sets(lines: list[str]) -> dict[int, int | None]:
    """Subcase -> TEMPERATURE(LOAD|BOTH) set, read from the raw lines.

    ``parse_case_control`` keeps one TEMPERATURE entry per subcase, so a deck
    giving both TEMPERATURE(MATERIAL) and TEMPERATURE(LOAD) would lose one.
    """
    default: int | None = None
    current: int | None = None
    sets: dict[int, int | None] = {}
    for head in _case_lines(lines):
        if head.startswith("SUBCASE"):
            current = int(head.split()[1])
            sets[current] = default
            continue
        m = _ASSIGN.match(head)
        if not m or not m[1].startswith("TEMP"):
            continue
        kind = (m[3] or "BOTH").strip()
        if kind.startswith(("LOAD", "BOTH")):
            if current is None:
                default = int(m[4])
            else:
                sets[current] = int(m[4])
    return sets or {1: default}


# -- assembly ---------------------------------------------------------------------


def _element_stiffness(kind: str, datas: list, model: BulkModel):
    """``(sils, k, components per grid)`` of one ECPT layout, rows in ascending-SIL order."""
    mats, cstm = model.materials, model.cstm
    if kind == "ROD":
        m = rod_matrices(RodBatch.from_elements(datas, mats), cstm)
        return m.sils, m.k, GRID_DOF
    if kind == "SHEAR":
        m = shear_matrices(ShearBatch.from_elements(datas, mats), cstm)
    elif kind == "TRMEM":
        m = trmem_matrices(TrmemBatch.from_elements(datas, mats), cstm)
    elif kind == "QDMEM":
        m = qdmem_matrices(QdmemBatch.from_elements(datas), mats, cstm)
    elif kind == "QDMEM1":
        m = qdmem1_matrices(datas, mats, cstm)
    elif kind == "QDMEM2":
        m = qdmem2_matrices(Qdmem2Batch.from_elements(datas, mats), cstm)
    else:
        raise NotSupported(f"element layout {kind} has no stiffness kernel")
    return m.sils, m.k, 3


def assemble_stiffness(sm: StaticModel) -> sp.csr_matrix:
    """Global ``K_gg`` from every ECPT record (sparse, duplicates summed)."""
    n = sm.ndof
    rows, cols, vals = [], [], []
    for kind, recs in sm.ecpts.items():
        sils, k, nc = _element_stiffness(kind, [r.data for r in recs], sm.model)
        idx = (np.asarray(sils)[:, :, None] - 1 + np.arange(nc)).reshape(len(recs), -1)
        shape = k.shape
        rows.append(np.broadcast_to(idx[:, :, None], shape).ravel())
        cols.append(np.broadcast_to(idx[:, None, :], shape).ravel())
        vals.append(np.asarray(k, dtype=float).ravel())
    if not vals:
        return sp.csr_matrix((n, n))
    coo = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return coo.tocsr()


# -- loads ------------------------------------------------------------------------


def _load_combinations(cards: Iterable[Card]) -> dict[int, tuple[float, list[tuple[float, int]]]]:
    """LOAD cards: ``SID -> (S, [(Si, Li), ...])``."""
    out: dict[int, tuple[float, list[tuple[float, int]]]] = {}
    for c in cards:
        if c.name != "LOAD":
            continue
        sid, scale = c.get(1), c.get(2)
        if sid is None or scale is None:
            raise BulkDataError(f"LOAD (line {c.lineno}): SID and S are required")
        if sid in out:
            raise BulkDataError(f"duplicate LOAD set {sid}")
        terms = []
        for k in range(2, len(c.fields), 2):
            si, li = c.fields[k], c.fields[k + 1] if k + 1 < len(c.fields) else None
            if si is None and li is None:
                continue
            if si is None or li is None:
                raise BulkDataError(f"LOAD {sid}: incomplete scale/set pair")
            terms.append((float(si), int(li)))
        out[int(sid)] = (float(scale), terms)
    return out


def _point_loads(sm: StaticModel, sid: int, p: FloatArray, factor: float = 1.0) -> bool:
    """Add FORCE/MOMENT set ``sid`` (``F * N`` in CID, rotated to the grid's CD)."""
    model, found = sm.model, False
    for ld in model.loads:
        if ld.sid != sid:
            continue
        found = True
        if ld.g not in model.grids:
            raise BulkDataError(f"{ld.card} {sid}: {ld.g} is not a grid point")
        g = model.grids[ld.g]
        ta_c = transforms_to_basic(model.cstm, ld.cid, g.basic)[0]
        ta_d = transforms_to_basic(model.cstm, g.cd, g.basic)[0]
        off = sm.dof(ld.g, 1 if ld.card == "FORCE" else 4)
        p[off : off + 3] += factor * ld.scale * (ta_d.T @ (ta_c @ np.asarray(ld.n, float)))
    return found


def _applied_loads(sm: StaticModel, sid: int, p: FloatArray) -> None:
    combos = _load_combinations(sm.deck.bulk)
    if sid in combos:
        scale, terms = combos[sid]
        for si, li in terms:
            if li in combos:
                raise BulkDataError(f"LOAD {sid} references LOAD set {li}")
            if not _point_loads(sm, li, p, scale * si):
                raise BulkDataError(f"LOAD {sid}: load set {li} is not defined")
    elif not _point_loads(sm, sid, p):
        raise BulkDataError(f"load set {sid} is not defined")


def _add_grid_loads(p: FloatArray, sil: NDArray[np.int64], force: FloatArray) -> None:
    """Add ``(n, grids, 3)`` translational loads at SILs ``(n, grids)``."""
    for k in range(3):
        np.add.at(p, (np.asarray(sil) - 1 + k).ravel(), force[..., k].ravel())


def _thermal_loads(sm: StaticModel, set_id: int, p: FloatArray) -> dict[int, float]:
    """Add the element thermal loads of temperature set ``set_id``; return EID -> TI(1)."""
    model = sm.model
    mats, cstm = model.materials, model.cstm
    temps: dict[int, float] = {}
    for kind, recs in sm.ecpts.items():
        datas = [r.data for r in recs]
        lt = load_temperatures(model, recs, set_id)
        temps.update(lt)
        tb = [lt[d.eid] for d in datas]
        if kind == "ROD":
            b = RodLoadBatch.from_elements(datas, mats, temperatures=lt)
            p += load_vector(b.sil, rod_loads(b, cstm), p.shape[0])
            continue
        if kind == "SHEAR":
            gt = load_grid_temperatures(model, recs, set_id)
            sb = ShearLoadBatch.from_elements(datas, mats, gt)
            _add_grid_loads(p, sb.sil, shear_loads(sb, cstm))
            continue
        if kind == "TRMEM":
            loads = trmem_thermal_loads(TrmemBatch.from_elements(datas, mats), tb, cstm)
        elif kind == "QDMEM":
            loads = qdmem_thermal_loads(QdmemBatch.from_elements(datas), mats, tb, cstm)
        elif kind == "QDMEM1":
            loads = qdmem1_thermal_loads(datas, mats, tb, cstm)
        elif kind == "QDMEM2":
            loads = qdmem2_thermal_loads(Qdmem2Batch.from_elements(datas, mats), tb, cstm)
        else:
            raise NotSupported(f"element layout {kind} has no thermal load kernel")
        for load in loads:
            load.add_to(p)
    return temps


# -- constraints ------------------------------------------------------------------


def _constraints(sm: StaticModel, spc_set: int | None) -> tuple[NDArray[np.bool_], FloatArray]:
    """Constrained DOFs (``PS`` and SPC/SPC1 set ``spc_set``) and their enforced values."""
    n = sm.ndof
    fixed, us = np.zeros(n, bool), np.zeros(n)
    for g in sm.model.grids.values():
        for ch in g.ps:
            if ch != "0":
                fixed[sm.dof(g.gid, int(ch))] = True
    if spc_set is not None:
        entries = [s for s in sm.model.spcs if s.sid == spc_set]
        if not entries:
            raise BulkDataError(f"SPC set {spc_set} is not defined")
        for s in entries:
            for ch in s.c or "0":
                i = sm.dof(s.g, int(ch))
                fixed[i] = True
                us[i] = s.d
    return fixed, us


# -- recovery ---------------------------------------------------------------------


def _recover(
    sm: StaticModel, u: FloatArray, temps: Mapping[int, float] | None
) -> tuple[dict[str, ElementTable], dict[str, ElementTable]]:
    model = sm.model
    mats, cstm = model.materials, model.cstm
    stresses: dict[str, ElementTable] = {}
    forces: dict[str, ElementTable] = {}
    for kind, recs in sm.ecpts.items():
        datas = [r.data for r in recs]
        force = None
        if kind == "ROD":
            res = rod_stresses(rod_stress_matrices(datas, mats, cstm), u, temps)
            stress, force = res.stress_table(), res.force_table()
        elif kind == "SHEAR":
            p1 = shear_stress_phase1(ShearStressBatch.from_elements(datas, mats), cstm)
            sres = shear_stress_phase2(p1, u)
            stress, force = sres.stress, sres.forces
        elif kind == "TRMEM":
            stress = trmem_stresses(trmem_stress_matrices(datas, mats, cstm), u, temps)
            stress = stress.stress_table()
        elif kind == "QDMEM":
            stress = qdmem_stresses(qdmem_stress_matrices(datas, mats, cstm), u, temps)
            stress = stress.stress_table()
        elif kind == "QDMEM1":
            stress = qdmem1_stress_table(qdmem1_stresses(datas, mats, u, cstm, temps))
        elif kind == "QDMEM2":
            p2 = qdmem2_stress_matrices(Qdmem2Batch.from_elements(datas, mats), cstm)
            tb = None if temps is None else [temps.get(d.eid, np.nan) for d in datas]
            qres = qdmem2_stresses(p2, u, tb)
            stress = qres.stress_table()
            force = np.array([qres.force_words(i)[1:] for i in range(len(datas))])
        else:
            raise NotSupported(f"element layout {kind} has no stress recovery kernel")
        eids = np.array([d.eid for d in datas], dtype=np.int64)
        cards = np.array([r.card for r in recs])
        for card in dict.fromkeys(cards):
            sel = cards == card
            stresses[card] = ElementTable(card, eids[sel], np.asarray(stress)[sel])
            if force is not None:
                forces[card] = ElementTable(card, eids[sel], np.asarray(force)[sel])
    return stresses, forces


# -- solve ------------------------------------------------------------------------


def _point_component(sm: StaticModel, i: int) -> tuple[int, int]:
    best = max((s, p) for p, s in sm.model.sil.items() if s - 1 <= i)
    s, p = best
    return p, (i - (s - 1) + 1) if p in sm.model.grids else 0


def _solve_subcase(sm: StaticModel, k: sp.csr_matrix, subcase: dict) -> SubcaseResult:
    n = sm.ndof
    sel = subcase["selectors"]
    p = np.zeros(n)
    if "LOAD" in sel:
        _applied_loads(sm, sel["LOAD"]["set"], p)
    thermal = sm.thermal_sets.get(subcase["id"])
    temps = _thermal_loads(sm, thermal, p) if thermal is not None else None
    fixed, u = _constraints(sm, sel.get("SPC", {}).get("set"))

    empty = np.asarray(abs(k).sum(axis=1)).ravel() == 0.0
    unstiff = np.flatnonzero(empty & ~fixed)
    loaded = unstiff[p[unstiff] != 0.0]
    if loaded.size:
        where = [_point_component(sm, int(i)) for i in loaded]
        raise SingularStiffness(
            f"subcase {subcase['id']}: loads on DOFs without stiffness (point, component): {where}"
        )
    singular = tuple(_point_component(sm, int(i)) for i in unstiff)
    fixed[unstiff] = True

    free = np.flatnonzero(~fixed)
    if free.size:
        kc = k.tocsc()
        kff = kc[free][:, free]
        rhs = p[free] - kc[free][:, np.flatnonzero(fixed)] @ u[fixed]
        with np.errstate(all="ignore"):
            uf = np.atleast_1d(spsolve(kff.tocsc(), rhs))
        residual = kff @ uf - rhs
        scale = max(np.abs(rhs).max(), np.abs(kff @ uf).max(), 1e-300)
        if not np.isfinite(uf).all() or np.abs(residual).max() > 1e-6 * scale:
            raise SingularStiffness(f"subcase {subcase['id']}: K_ff is singular (mechanism)")
        u[free] = uf
    q = k @ u - p
    q[~fixed] = 0.0
    stresses, forces = _recover(sm, u, temps)
    return SubcaseResult(subcase["id"], p, u, q, stresses, forces, singular, dict(sm.model.sil))


def solve_static(
    source: StaticModel | str | Path, subcases: Iterable[int] | None = None
) -> StaticSolution:
    """Solve every SUBCASE of a RF1 deck (``source``: :class:`StaticModel`, deck text or path)."""
    if isinstance(source, StaticModel):
        sm = source
    elif isinstance(source, Path) or (isinstance(source, str) and "\n" not in source):
        sm = StaticModel.from_file(source)
    else:
        sm = StaticModel.from_text(source)
    k = assemble_stiffness(sm)
    wanted = None if subcases is None else set(subcases)
    results = {
        sc["id"]: _solve_subcase(sm, k, sc)
        for sc in sm.case_control["subcases"]
        if wanted is None or sc["id"] in wanted
    }
    return StaticSolution(sm, k, results)
