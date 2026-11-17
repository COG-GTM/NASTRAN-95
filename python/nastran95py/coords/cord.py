"""CORD1R/C/S and CORD2R/C/S cards to CSTM records (port of the CSTM pass of ``GP1``).

``GP1`` (``mis/gp1.f``) reads the coordinate system definitions (CSDT), then
repeatedly solves every system whose defining data is already in basic:

* a CORD2 with ``RID = 0``, or whose ``RID`` system is solved (``A``, ``B`` and
  ``C`` are converted from ``RID`` coordinates to basic first);
* a CORD1 whose three grid points are all located in basic. Grid points
  referenced by CORD1 cards that are defined in another system are converted
  to basic at the end of each pass, once that system is solved.

A pass that solves nothing while systems remain is an inconsistent definition.
Each solved system becomes a 14-word CSTM record: origin ``A`` and the
rotation ``T = [i j k]`` with ``k = (B - A)/|B - A|``, ``j = k x (C - A)``
normalised and ``i = j x k``.

``location_to_basic`` ports the ``GP1`` internal routines that convert a point
given in a system (``x, y, z``; ``r, theta, z``; ``rho, theta, phi`` with angles
in degrees) to basic; ``GP1`` also uses them to build the BGPDT, which
``grids_to_basic`` reproduces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from nastran95py.coords.transforms import CoordSystem, CoordType, Cstm, CstmError, FloatArray

# GP1 user fatal message 2xxx codes passed to MESAGE(-30, IERR, ...).
ERR_GRID_UNDEFINED = 3
ERR_RID_UNDEFINED = 4
ERR_INCONSISTENT = 5
ERR_CP_UNDEFINED = 6

_CORD_TYPES = {"R": CoordType.RECTANGULAR, "C": CoordType.CYLINDRICAL, "S": CoordType.SPHERICAL}


class CordError(ValueError):
    """Coordinate system definition error (``GP1`` ``MESAGE(-30, IERR, ...)``)."""

    def __init__(self, ierr: int, message: str):
        super().__init__(message)
        self.ierr = ierr


@dataclass(frozen=True)
class Cord1:
    """CORD1R/C/S: system defined by three grid points ``g1`` (origin), ``g2`` (z axis), ``g3``."""

    cid: int
    ctype: CoordType
    g1: int
    g2: int
    g3: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ctype", CoordType(self.ctype))
        if self.cid <= 0 or min(self.g1, self.g2, self.g3) <= 0:
            raise ValueError(f"CORD1 {self.cid}: CID and grid IDs must be positive")
        if len({self.g1, self.g2, self.g3}) != 3:
            raise ValueError(f"CORD1 {self.cid}: grid points must be distinct")


@dataclass(frozen=True)
class Cord2:
    """CORD2R/C/S: system defined by points ``a``, ``b``, ``c`` given in system ``rid``."""

    cid: int
    ctype: CoordType
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    c: tuple[float, float, float]
    rid: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ctype", CoordType(self.ctype))
        for name in ("a", "b", "c"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        if self.cid <= 0 or self.rid < 0:
            raise ValueError(f"CORD2 {self.cid}: CID must be positive and RID non-negative")
        if self.a == self.b or self.a == self.c or self.b == self.c:
            raise ValueError(f"CORD2 {self.cid}: points A, B, C must be distinct")


@dataclass(frozen=True)
class Grid:
    """Location part of a GRID card: ``xyz`` in system ``cp``."""

    gid: int
    cp: int
    xyz: tuple[float, float, float]


Cord = Cord1 | Cord2


def location_to_basic(cs: CoordSystem | None, xyz: ArrayLike) -> FloatArray:
    """Convert point(s) given in system ``cs`` to basic (``GP1`` statements 990-1010).

    ``xyz`` is ``(3,)`` or ``(n, 3)``: rectangular ``x, y, z``; cylindrical
    ``r, theta, z``; spherical ``rho, theta, phi`` with ``theta`` measured from
    the local z axis and ``phi`` from the local x axis, angles in degrees.
    ``cs=None`` means basic.
    """
    p = np.asarray(xyz, dtype=float)
    if cs is None:
        return p.copy()
    q = np.atleast_2d(p).copy()
    if cs.ctype == CoordType.CYLINDRICAL:
        r, th = q[:, 0].copy(), np.deg2rad(q[:, 1])
        q[:, 0], q[:, 1] = r * np.cos(th), r * np.sin(th)
    elif cs.ctype == CoordType.SPHERICAL:
        rho, th, ph = q[:, 0].copy(), np.deg2rad(q[:, 1]), np.deg2rad(q[:, 2])
        q[:, 0] = rho * np.sin(th) * np.cos(ph)
        q[:, 1] = rho * np.sin(th) * np.sin(ph)
        q[:, 2] = rho * np.cos(th)
    return (q @ cs.t.T + cs.origin).reshape(p.shape)


def build_cstm(cords: Iterable[Cord], grids: Mapping[int, Grid] | Iterable[Grid] = ()) -> Cstm:
    """Solve CORD1/CORD2 definitions into a :class:`Cstm`, in input (CSDT) order.

    ``grids`` supplies the GRID locations referenced by CORD1 cards. Raises
    :class:`CordError` with the ``GP1`` error code for undefined grid points or
    reference systems and for circular definitions.
    """
    csdt: list[Cord] = list(cords)
    defined: dict[int, Cord] = {}
    for cord in csdt:
        if cord.cid in defined:
            raise ValueError(f"duplicate coordinate system {cord.cid}")
        defined[cord.cid] = cord
    grid_map = dict(grids) if isinstance(grids, Mapping) else {g.gid: g for g in grids}

    # CSGP: grid points referenced by CORD1 entries, as (cp, xyz) until located in basic.
    csgp: dict[int, tuple[int, FloatArray]] = {}
    for cord in csdt:
        if isinstance(cord, Cord1):
            for gid in (cord.g1, cord.g2, cord.g3):
                if gid not in grid_map:
                    raise CordError(
                        ERR_GRID_UNDEFINED,
                        f"grid point {gid} on CORD1 {cord.cid} is not defined",
                    )
                g = grid_map[gid]
                csgp[gid] = (int(g.cp), np.asarray(g.xyz, dtype=float))

    solved: dict[int, CoordSystem] = {}
    while True:
        n_before = len(solved)
        for cord in csdt:
            if cord.cid in solved:
                continue
            if isinstance(cord, Cord1):
                located = [csgp[g] for g in (cord.g1, cord.g2, cord.g3)]
                if any(cp != 0 for cp, _ in located):
                    continue
                points = [xyz for _, xyz in located]
            elif cord.rid == 0:
                points = [np.asarray(p) for p in (cord.a, cord.b, cord.c)]
            else:
                if cord.rid not in defined:
                    raise CordError(
                        ERR_RID_UNDEFINED,
                        f"reference system {cord.rid} on CORD2 {cord.cid} is not defined",
                    )
                if cord.rid not in solved:
                    continue
                points = list(location_to_basic(solved[cord.rid], [cord.a, cord.b, cord.c]))
            solved[cord.cid] = CoordSystem.from_points(cord.cid, cord.ctype, *points)

        for gid, (cp, xyz) in csgp.items():
            if cp == 0:
                continue
            if cp not in defined:
                raise CordError(
                    ERR_CP_UNDEFINED,
                    f"coordinate system {cp} of grid point {gid} is not defined",
                )
            if cp in solved:
                csgp[gid] = (0, location_to_basic(solved[cp], xyz))

        if len(solved) == len(csdt):
            return Cstm(solved[cord.cid] for cord in csdt)
        if len(solved) == n_before:
            pending = sorted(set(defined) - set(solved))
            raise CordError(
                ERR_INCONSISTENT,
                f"inconsistent coordinate system definitions (unsolved: {pending})",
            )


def grids_to_basic(cstm: Mapping[int, CoordSystem], grids: Iterable[Grid]) -> dict[int, FloatArray]:
    """Basic location of each grid point (the BGPDT coordinates written by ``GP1``)."""
    out: dict[int, FloatArray] = {}
    for g in grids:
        if g.cp == 0:
            out[g.gid] = np.asarray(g.xyz, dtype=float)
            continue
        try:
            cs = cstm[g.cp]
        except KeyError:
            raise CstmError(g.cp) from None
        out[g.gid] = location_to_basic(cs, g.xyz)
    return out


def _num(v: Any) -> float:
    return 0.0 if v is None else float(v)


def _int(v: Any) -> int:
    return 0 if v is None else int(v)


def cords_from_cards(cards: Iterable[Any]) -> list[Cord]:
    """Build CORD1R/C/S and CORD2R/C/S entries from parsed bulk data cards.

    ``cards`` are objects with ``name`` and ``fields`` (e.g.
    :class:`nastran95py.reference.cards.Card`); other card names are ignored.
    A CORD1 card may define two systems; blank RID and coordinates are zero.
    """
    out: list[Cord] = []
    for card in cards:
        name = card.name.upper()
        if len(name) != 6 or not name.startswith("CORD") or name[5] not in _CORD_TYPES:
            continue
        ctype = _CORD_TYPES[name[5]]
        f: Sequence[Any] = list(card.fields) + [None] * 11
        if name[4] == "1":
            for k in (0, 4):
                if any(v is not None for v in f[k : k + 4]):
                    out.append(Cord1(_int(f[k]), ctype, *(_int(v) for v in f[k + 1 : k + 4])))
        elif name[4] == "2":
            vals = [_num(v) for v in f[2:11]]
            out.append(Cord2(_int(f[0]), ctype, vals[0:3], vals[3:6], vals[6:9], _int(f[1])))
    return out


def grids_from_cards(cards: Iterable[Any]) -> list[Grid]:
    """GRID locations (``ID, CP, X1, X2, X3``) from parsed bulk data cards.

    A blank GRID CP takes the CP of the deck's GRDSET card, as in IFS1P.
    """
    cards = list(cards)
    default_cp = 0
    for card in cards:
        if card.name.upper() == "GRDSET":
            default_cp = _int((list(card.fields) + [None] * 2)[1])
    out: list[Grid] = []
    for card in cards:
        if card.name.upper() == "GRID":
            f = list(card.fields) + [None] * 5
            cp = default_cp if f[1] is None else _int(f[1])
            out.append(Grid(_int(f[0]), cp, (_num(f[2]), _num(f[3]), _num(f[4]))))
    return out
