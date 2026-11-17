"""Typed bulk data model for the cards used by the RF1 element decks.

``BulkModel.from_deck`` turns the logical cards produced by
:func:`nastran95py.reference.cards.read_deck` (small, large and free field, with
continuations) into typed records, resolves coordinate systems and grid point
locations, and assigns scalar index numbers the way GP1 does. ECPT records are
built from it by :mod:`nastran95py.bulk.ecpt`.

Supported cards: GRID, GRDSET, SPOINT, SEQGP, CORD1R/C/S, CORD2R/C/S, CROD,
CONROD, PROD, CSHEAR, PSHEAR, CTRMEM, PTRMEM, CQDMEM, CQDMEM1, CQDMEM2, PQDMEM,
PQDMEM1, PQDMEM2, MAT1, MAT2, FORCE, MOMENT, SPC, SPC1, TEMP and TEMPD. Other
cards are kept in ``BulkModel.ignored`` (counted, not interpreted).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, CoordType, Cstm
from nastran95py.materials import Mat1, Mat2, MaterialTable
from nastran95py.reference.cards import Card, Deck, expand_ids, read_deck

FloatArray = NDArray[np.float64]

GRID_DOF = 6

_CORD_TYPES = {"R": CoordType.RECTANGULAR, "C": CoordType.CYLINDRICAL, "S": CoordType.SPHERICAL}

MEMBRANE_CARDS = {"CTRMEM": 3, "CQDMEM": 4, "CQDMEM1": 4, "CQDMEM2": 4}
PLATE_PROPERTY_CARDS = ("PSHEAR", "PTRMEM", "PQDMEM", "PQDMEM1", "PQDMEM2")
PROPERTY_FOR = {
    "CROD": "PROD",
    "CSHEAR": "PSHEAR",
    "CTRMEM": "PTRMEM",
    "CQDMEM": "PQDMEM",
    "CQDMEM1": "PQDMEM1",
    "CQDMEM2": "PQDMEM2",
}


class BulkDataError(ValueError):
    """Inconsistent or unsupported bulk data (the Fortran IFP/GP1 fatal messages)."""


@dataclass
class Grid:
    gid: int
    cp: int
    x: FloatArray
    cd: int
    ps: str = ""
    basic: FloatArray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class Connection:
    """One element connection card (CROD, CONROD, CSHEAR, CTRMEM, CQDMEMi)."""

    card: str
    eid: int
    pid: int | None
    grids: tuple[int, ...]
    theta: float = 0.0
    conrod: tuple[int, float, float, float, float] | None = None
    """CONROD (MID, A, J, C, NSM): the property lives on the connection card."""


@dataclass(frozen=True)
class RodProperty:
    pid: int
    mid: int
    a: float
    j: float = 0.0
    c: float = 0.0
    nsm: float = 0.0


@dataclass(frozen=True)
class PlateProperty:
    """PSHEAR, PTRMEM, PQDMEM, PQDMEM1, PQDMEM2: ``PID MID T NSM``."""

    card: str
    pid: int
    mid: int
    t: float
    nsm: float = 0.0


@dataclass(frozen=True)
class PointLoad:
    """FORCE or MOMENT: ``F * N`` at grid ``g`` in the directions of ``cid``."""

    card: str
    sid: int
    g: int
    cid: int
    scale: float
    n: tuple[float, float, float]


@dataclass(frozen=True)
class SinglePointConstraint:
    """One SPC or SPC1 constraint: components ``c`` of grid ``g`` set to ``d``."""

    card: str
    sid: int
    g: int
    c: str
    d: float = 0.0


@dataclass
class BulkModel:
    grids: dict[int, Grid] = field(default_factory=dict)
    spoints: set[int] = field(default_factory=set)
    sequence: dict[int, tuple[int, ...]] = field(default_factory=dict)
    cstm: Cstm = field(default_factory=Cstm)
    connections: list[Connection] = field(default_factory=list)
    rod_properties: dict[int, RodProperty] = field(default_factory=dict)
    plate_properties: dict[tuple[str, int], PlateProperty] = field(default_factory=dict)
    materials: MaterialTable = field(default_factory=MaterialTable)
    loads: list[PointLoad] = field(default_factory=list)
    spcs: list[SinglePointConstraint] = field(default_factory=list)
    temperatures: dict[int, dict[int, float]] = field(default_factory=dict)
    default_temperatures: dict[int, float] = field(default_factory=dict)
    ignored: Counter = field(default_factory=Counter)
    _sil: dict[int, int] | None = field(default=None, repr=False)

    @classmethod
    def from_text(cls, text: str) -> BulkModel:
        return cls.from_cards(read_deck(text).bulk)

    @classmethod
    def from_deck(cls, deck: Deck) -> BulkModel:
        return cls.from_cards(deck.bulk)

    @classmethod
    def from_cards(cls, cards: list[Card]) -> BulkModel:
        model = cls()
        grdset = next((c for c in cards if c.name == "GRDSET"), None)
        cords: list[Card] = []
        mat1: list[Mat1] = []
        mat2: list[Mat2] = []
        for c in cards:
            handler = _HANDLERS.get(c.name)
            if c.name.startswith("CORD") and c.name[-1] in _CORD_TYPES:
                cords.append(c)
            elif c.name == "GRID":
                model._add_grid(c, grdset)
            elif c.name == "MAT1":
                mat1.append(_mat1(c))
            elif c.name == "MAT2":
                mat2.append(_mat2(c))
            elif handler is not None:
                handler(model, c)
            elif c.name != "GRDSET":
                model.ignored[c.name] += 1
        model.materials = MaterialTable(mat1, mat2)
        model._resolve_geometry(cords)
        return model

    # -- cards -------------------------------------------------------------

    def _add_grid(self, c: Card, grdset: Card | None) -> None:
        gid = _req(c, 1)
        if gid in self.grids or gid in self.spoints:
            raise BulkDataError(f"duplicate grid/scalar point {gid}")
        dflt = (lambda i: grdset.get(i)) if grdset is not None else (lambda i: None)
        cp = c.get(2, dflt(2) or 0)
        cd = c.get(6, dflt(6) or 0)
        ps = c.get(7, dflt(7))
        x = np.array([float(c.get(i, 0.0)) for i in (3, 4, 5)])
        self.grids[gid] = Grid(gid, int(cp), x, int(cd), "" if ps is None else str(ps))

    def _resolve_geometry(self, cords: list[Card]) -> None:
        """Resolve CORD1x/CORD2x chains and grid basic coordinates (GP1 order-free)."""
        pending1: list[tuple[int, CoordType, tuple[int, int, int]]] = []
        pending2: list[tuple[int, CoordType, int, list[float]]] = []
        for c in cords:
            ctype = _CORD_TYPES[c.name[-1]]
            if c.name.startswith("CORD1"):
                for k in range(0, len(c.fields), 4):
                    if c.get(k + 1) is not None:
                        g = (_req(c, k + 2), _req(c, k + 3), _req(c, k + 4))
                        pending1.append((_req(c, k + 1), ctype, g))
            elif c.name.startswith("CORD2"):
                pts = [float(c.get(i, 0.0)) for i in range(3, 12)]
                pending2.append((_req(c, 1), ctype, int(c.get(2, 0)), pts))
            else:
                raise BulkDataError(f"unsupported coordinate card {c.name}")
        systems: dict[int, CoordSystem] = {}
        unresolved = {g.gid for g in self.grids.values()}
        while True:
            progress = False
            for gid in sorted(unresolved):
                g = self.grids[gid]
                if g.cp == 0 or g.cp in systems:
                    g.basic = point_to_basic(systems.get(g.cp), g.x)
                    unresolved.discard(gid)
                    progress = True
            for item in list(pending2):
                cid, ctype, rid, pts = item
                if rid == 0 or rid in systems:
                    ref = systems.get(rid)
                    a, b, cc = (point_to_basic(ref, pts[i : i + 3]) for i in (0, 3, 6))
                    systems[cid] = _new_system(systems, cid, ctype, a, b, cc)
                    pending2.remove(item)
                    progress = True
            for item in list(pending1):
                cid, ctype, gids = item
                missing = [g for g in gids if g not in self.grids]
                if missing:
                    raise BulkDataError(f"CORD1 {cid} references undefined grids {missing}")
                if not any(g in unresolved for g in gids):
                    a, b, cc = (self.grids[g].basic for g in gids)
                    systems[cid] = _new_system(systems, cid, ctype, a, b, cc)
                    pending1.remove(item)
                    progress = True
            if not progress:
                break
        if pending1 or pending2:
            ids = sorted([p[0] for p in pending1] + [p[0] for p in pending2])
            raise BulkDataError(f"coordinate systems {ids} cannot be resolved")
        if unresolved:
            cps = sorted({self.grids[g].cp for g in unresolved})
            raise BulkDataError(f"grids reference undefined coordinate systems {cps}")
        self.cstm = Cstm(systems.values())
        for g in self.grids.values():
            if g.cd and g.cd not in systems:
                raise BulkDataError(f"grid {g.gid}: undefined displacement system CD={g.cd}")

    # -- derived tables ----------------------------------------------------

    @property
    def sil(self) -> dict[int, int]:
        """Scalar index number of every grid/scalar point (GP1, with SEQGP resequencing)."""
        if self._sil is None:
            points = [(gid, GRID_DOF) for gid in self.grids] + [(s, 1) for s in self.spoints]
            unknown = set(self.sequence) - {p for p, _ in points}
            if unknown:
                raise BulkDataError(f"SEQGP references undefined points {sorted(unknown)}")
            points.sort(key=lambda p: self.sequence.get(p[0], (p[0],)))
            sil, nxt = {}, 1
            for pid, ndof in points:
                sil[pid] = nxt
                nxt += ndof
            self._sil = sil
        return self._sil

    def bgpdt(self, gid: int) -> tuple[int, FloatArray]:
        """BGPDT entry ``(CID, basic xyz)``; scalar points get CID -1 at the origin."""
        if gid in self.grids:
            g = self.grids[gid]
            return g.cd, g.basic
        if gid in self.spoints:
            return -1, np.zeros(3)
        raise BulkDataError(f"undefined grid point {gid}")

    def grid_temperature(self, set_id: int, gid: int) -> float:
        """TEMP value for ``gid`` in ``set_id``, else the TEMPD default of that set."""
        t = self.temperatures.get(set_id, {})
        if gid in t:
            return t[gid]
        if set_id in self.default_temperatures:
            return self.default_temperatures[set_id]
        if set_id not in self.temperatures:
            raise BulkDataError(f"temperature set {set_id} is not defined")
        raise BulkDataError(f"no temperature for grid {gid} in set {set_id} and no TEMPD")


def point_to_basic(cs: CoordSystem | None, local) -> FloatArray:
    """Location of ``local`` (coordinates in ``cs``, angles in degrees) in basic."""
    p = np.asarray(local, dtype=float).reshape(3)
    if cs is None:
        return p.copy()
    if cs.ctype == CoordType.CYLINDRICAL:
        r, th = p[0], np.radians(p[1])
        p = np.array([r * np.cos(th), r * np.sin(th), p[2]])
    elif cs.ctype == CoordType.SPHERICAL:
        r, th, ph = p[0], np.radians(p[1]), np.radians(p[2])
        p = r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return cs.origin + cs.t @ p


def _new_system(systems, cid, ctype, a, b, c) -> CoordSystem:
    if cid == 0 or cid in systems:
        raise BulkDataError(f"coordinate system {cid} is basic or defined twice")
    return CoordSystem.from_points(cid, ctype, a, b, c)


def _req(c: Card, i: int):
    v = c.get(i)
    if v is None:
        raise BulkDataError(f"{c.name} (line {c.lineno}): field {i + 1} is required")
    return v


def _real(c: Card, i: int, default: float = 0.0) -> float:
    v = c.get(i, default)
    if not isinstance(v, int | float):
        raise BulkDataError(f"{c.name} (line {c.lineno}): field {i + 1} is not numeric: {v!r}")
    return float(v)


def _mat1(c: Card) -> Mat1:
    """MAT1 with the User's Manual completion rule for blank E, G or NU."""
    e, g, nu = c.get(2), c.get(3), c.get(4)
    blank = [x is None for x in (e, g, nu)]
    if sum(blank) >= 2:
        e, g, nu = (0.0 if x is None else float(x) for x in (e, g, nu))
    elif e is None:
        e = 2.0 * (1.0 + nu) * g
    elif g is None:
        g = e / (2.0 * (1.0 + nu))
    elif nu is None:
        nu = e / (2.0 * g) - 1.0
    vals = [_real(c, i) for i in range(5, 12)]
    return Mat1(int(_req(c, 1)), float(e), float(g), float(nu), *vals)


def _mat2(c: Card) -> Mat2:
    return Mat2(int(_req(c, 1)), *(_real(c, i) for i in range(2, 17)))


def _crod(model: BulkModel, c: Card) -> None:
    for k in range(0, len(c.fields), 4):
        eid = c.get(k + 1)
        if eid is not None:
            g = (_req(c, k + 3), _req(c, k + 4))
            model.connections.append(Connection("CROD", eid, c.get(k + 2, eid), g))


def _conrod(model: BulkModel, c: Card) -> None:
    eid = _req(c, 1)
    prop = (int(_req(c, 4)), _real(c, 5), _real(c, 6), _real(c, 7), _real(c, 8))
    model.connections.append(Connection("CONROD", eid, None, (_req(c, 2), _req(c, 3)), conrod=prop))


def _plate_connection(model: BulkModel, c: Card) -> None:
    n = MEMBRANE_CARDS.get(c.name, 4)
    eid = _req(c, 1)
    grids = tuple(_req(c, 3 + i) for i in range(n))
    if len(set(grids)) != n:
        raise BulkDataError(f"{c.name} {eid}: repeated grid points {grids}")
    theta = _real(c, 3 + n) if c.name in MEMBRANE_CARDS else 0.0
    model.connections.append(Connection(c.name, eid, c.get(2, eid), grids, theta))


def _prod(model: BulkModel, c: Card) -> None:
    pid = _req(c, 1)
    if pid in model.rod_properties:
        raise BulkDataError(f"duplicate PROD {pid}")
    model.rod_properties[pid] = RodProperty(
        pid, int(_req(c, 2)), _real(c, 3), _real(c, 4), _real(c, 5), _real(c, 6)
    )


def _plate_property(model: BulkModel, c: Card) -> None:
    for k in range(0, len(c.fields), 4):
        pid = c.get(k + 1)
        if pid is None:
            continue
        if (c.name, pid) in model.plate_properties:
            raise BulkDataError(f"duplicate {c.name} {pid}")
        model.plate_properties[(c.name, pid)] = PlateProperty(
            c.name, pid, int(_req(c, k + 2)), _real(c, k + 3), _real(c, k + 4)
        )


def _point_load(model: BulkModel, c: Card) -> None:
    n = (_real(c, 5), _real(c, 6), _real(c, 7))
    model.loads.append(PointLoad(c.name, _req(c, 1), _req(c, 2), int(c.get(3, 0)), _real(c, 4), n))


def _spc(model: BulkModel, c: Card) -> None:
    sid = _req(c, 1)
    for k in (2, 5):
        g = c.get(k)
        if g is not None:
            model.spcs.append(
                SinglePointConstraint("SPC", sid, g, str(c.get(k + 1, "")), _real(c, k + 2))
            )


def _spc1(model: BulkModel, c: Card) -> None:
    sid, comp = _req(c, 1), str(_req(c, 2))
    for g in expand_ids(c.fields[2:]):
        model.spcs.append(SinglePointConstraint("SPC1", sid, g, comp))


def _temp(model: BulkModel, c: Card) -> None:
    t = model.temperatures.setdefault(_req(c, 1), {})
    for k in (2, 4, 6):
        g = c.get(k)
        if g is not None:
            if g in t:
                raise BulkDataError(f"TEMP set {c.get(1)}: grid {g} given twice")
            t[g] = _real(c, k + 1)


def _tempd(model: BulkModel, c: Card) -> None:
    for k in range(1, len(c.fields) + 1, 2):
        sid = c.get(k)
        if sid is not None:
            model.default_temperatures[sid] = _real(c, k + 1)


def _spoint(model: BulkModel, c: Card) -> None:
    for s in expand_ids(c.fields):
        if s in model.grids or s in model.spoints:
            raise BulkDataError(f"duplicate grid/scalar point {s}")
        model.spoints.add(s)


def _seqgp(model: BulkModel, c: Card) -> None:
    for k in range(1, len(c.fields) + 1, 2):
        pid = c.get(k)
        if pid is not None:
            raw = c.get(k + 1)
            key = tuple(int(p) for p in str(raw).split(".") if p != "")
            model.sequence[pid] = key


_HANDLERS = {
    "CROD": _crod,
    "CONROD": _conrod,
    "CSHEAR": _plate_connection,
    **dict.fromkeys(MEMBRANE_CARDS, _plate_connection),
    "PROD": _prod,
    **dict.fromkeys(PLATE_PROPERTY_CARDS, _plate_property),
    "FORCE": _point_load,
    "MOMENT": _point_load,
    "SPC": _spc,
    "SPC1": _spc1,
    "TEMP": _temp,
    "TEMPD": _tempd,
    "SPOINT": _spoint,
    "SEQGP": _seqgp,
}
