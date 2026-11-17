"""Coordinate system table (CSTM) and global-to-basic transforms.

A CSTM record is 14 words: ``CID, TYPE, X0, Y0, Z0, T11, T12, ..., T33`` where
``T`` (stored by rows) maps a vector from the local rectangular frame of the
system to basic. ``TYPE`` is 1 rectangular, 2 cylindrical, 3 spherical.

``transforms_to_basic`` ports the ``TRANSS``/``TRANSD`` entries of
``mis/pretrs.f`` / ``mis/pretrd.f`` and ``GBTRAN`` of ``mis/basglb.f``: given
coordinate system IDs and points in basic coordinates it returns the 3x3
matrices ``TA`` mapping a vector expressed in that system's global
(displacement) directions at the point to basic. ``basic_to_global`` and
``global_to_basic`` port the ``BASGLB`` and ``GLBBAS`` entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

CSTM_RECORD_WORDS = 14


class CstmError(KeyError):
    """Coordinate system not found in the CSTM (Fortran ``MESAGE(-30,25,...)``)."""


class CoordType(IntEnum):
    RECTANGULAR = 1
    CYLINDRICAL = 2
    SPHERICAL = 3


@dataclass(frozen=True)
class CoordSystem:
    """One CSTM record."""

    cid: int
    ctype: CoordType
    origin: FloatArray = field(default_factory=lambda: np.zeros(3))
    t: FloatArray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "ctype", CoordType(self.ctype))
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3, 3))

    @classmethod
    def from_points(
        cls, cid: int, ctype: CoordType | int, a: ArrayLike, b: ArrayLike, c: ArrayLike
    ) -> CoordSystem:
        """Build a system from CORD2R/C/S points given in basic coordinates.

        ``a`` is the origin, ``b`` lies on the local z axis and ``c`` in the
        local x-z plane.
        """
        pa, pb, pc = (np.asarray(p, dtype=float).reshape(3) for p in (a, b, c))
        ez = pb - pa
        nz = np.linalg.norm(ez)
        ey = np.cross(ez, pc - pa)
        ny = np.linalg.norm(ey)
        if nz == 0.0 or ny == 0.0:
            raise ValueError(f"coordinate system {cid}: points A, B, C are degenerate")
        ez /= nz
        ey /= ny
        ex = np.cross(ey, ez)
        return cls(cid, CoordType(ctype), pa, np.column_stack([ex, ey, ez]))

    @classmethod
    def from_record(cls, words: Sequence[float]) -> CoordSystem:
        if len(words) != CSTM_RECORD_WORDS:
            raise ValueError(f"CSTM record must have {CSTM_RECORD_WORDS} words")
        return cls(int(words[0]), CoordType(int(words[1])), words[2:5], words[5:14])

    def to_record(self) -> list[float]:
        return [self.cid, int(self.ctype), *self.origin.tolist(), *self.t.ravel().tolist()]


class Cstm(Mapping[int, CoordSystem]):
    """Coordinate system transformation matrix table keyed by CID."""

    def __init__(self, systems: Iterable[CoordSystem] = ()):
        self._systems: dict[int, CoordSystem] = {}
        for cs in systems:
            if cs.cid == 0:
                raise ValueError("CID 0 is the basic system and cannot be redefined")
            if cs.cid in self._systems:
                raise ValueError(f"duplicate coordinate system {cs.cid}")
            self._systems[cs.cid] = cs

    @classmethod
    def from_words(cls, words: Sequence[float]) -> Cstm:
        if len(words) % CSTM_RECORD_WORDS:
            raise ValueError("CSTM length is not a multiple of 14 words")
        return cls(
            CoordSystem.from_record(words[i : i + CSTM_RECORD_WORDS])
            for i in range(0, len(words), CSTM_RECORD_WORDS)
        )

    def to_words(self) -> list[float]:
        return [w for cs in self._systems.values() for w in cs.to_record()]

    def __getitem__(self, cid: int) -> CoordSystem:
        try:
            return self._systems[cid]
        except KeyError:
            raise CstmError(cid) from None

    def __iter__(self) -> Iterator[int]:
        return iter(self._systems)

    def __len__(self) -> int:
        return len(self._systems)


def transforms_to_basic(
    cstm: Mapping[int, CoordSystem],
    cid: ArrayLike,
    points: ArrayLike,
    real_offset: bool = False,
) -> FloatArray:
    """Vectorised ``TRANSS``/``TRANSD``/``GBTRAN``: ``TA`` at each of ``points``.

    ``points`` is ``(n, 3)`` (or ``(3,)``) in basic coordinates and ``cid`` a
    scalar or ``(n,)`` array of coordinate system IDs, so the grids of one
    element may use different systems. Returns ``(n, 3, 3)``; ``TA[k]`` maps a
    vector in the global directions of ``cid[k]`` at ``points[k]`` to basic.
    CID 0 gives the identity. For cylindrical and spherical systems the local
    axes are evaluated at the point; on the polar axis (``r == 0``) the
    rectangular matrix is returned, as in the Fortran.

    ``real_offset=True`` forms the point-minus-origin offset in REAL (float32)
    arithmetic, as ``TRANSD`` does, for bit-level parity with the Fortran.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[-1] != 3 or pts.ndim != 2:
        raise ValueError("points must have shape (3,) or (n, 3)")
    n = pts.shape[0]
    cids = np.broadcast_to(np.asarray(cid, dtype=np.int64), (n,))
    out = np.empty((n, 3, 3))
    for c in np.unique(cids):
        sel = cids == c
        if c == 0:
            out[sel] = np.eye(3)
            continue
        try:
            cs = cstm[int(c)]
        except KeyError:
            raise CstmError(int(c)) from None
        if cs.ctype == CoordType.RECTANGULAR:
            out[sel] = cs.t
            continue
        if real_offset:
            d = (pts[sel].astype(np.float32) - cs.origin.astype(np.float32)).astype(float)
        else:
            d = pts[sel] - cs.origin
        x, y, z = (d @ cs.t).T
        r = np.hypot(x, y)
        polar = r == 0.0
        rs = np.where(polar, 1.0, r)
        tl = np.zeros((x.size, 3, 3))
        if cs.ctype == CoordType.CYLINDRICAL:
            tl[:, 0, 0] = x / rs
            tl[:, 0, 1] = -y / rs
            tl[:, 1, 0] = y / rs
            tl[:, 1, 1] = x / rs
            tl[:, 2, 2] = 1.0
        else:
            xl = np.sqrt(x * x + y * y + z * z)
            xls = np.where(polar, 1.0, xl)
            tl[:, 0, 0] = x / xls
            tl[:, 0, 1] = x * z / (rs * xls)
            tl[:, 0, 2] = -y / rs
            tl[:, 1, 0] = y / xls
            tl[:, 1, 1] = y * z / (rs * xls)
            tl[:, 1, 2] = x / rs
            tl[:, 2, 0] = z / xls
            tl[:, 2, 1] = -r / xls
        tl[polar] = np.eye(3)
        out[sel] = cs.t @ tl
    return out


def transform_to_basic(cstm: Mapping[int, CoordSystem], cid: int, point: ArrayLike) -> FloatArray:
    """Port of ``TRANSS``/``TRANSD``: global-to-basic matrix ``TA`` at one ``point``.

    Single-point form of :func:`transforms_to_basic`.
    """
    return transforms_to_basic(cstm, cid, np.asarray(point, dtype=float).reshape(3))[0]


def global_to_basic(
    cstm: Mapping[int, CoordSystem], cid: ArrayLike, point: ArrayLike, vector: ArrayLike
) -> FloatArray:
    """Port of ``GLBBAS``: map vectors in the global directions of ``cid`` to basic.

    ``point`` and ``vector`` are ``(3,)`` or ``(n, 3)``; the result has the
    shape of ``vector``.
    """
    v = np.asarray(vector, dtype=float)
    ta = transforms_to_basic(cstm, cid, point)
    return np.einsum("nij,nj->ni", ta, np.atleast_2d(v)).reshape(v.shape)


def basic_to_global(
    cstm: Mapping[int, CoordSystem], cid: ArrayLike, point: ArrayLike, vector: ArrayLike
) -> FloatArray:
    """Port of ``BASGLB``: map basic vectors at ``point`` to the global directions of ``cid``.

    ``point`` and ``vector`` are ``(3,)`` or ``(n, 3)``; the result has the
    shape of ``vector``.
    """
    v = np.asarray(vector, dtype=float)
    ta = transforms_to_basic(cstm, cid, point)
    return np.einsum("nji,nj->ni", ta, np.atleast_2d(v)).reshape(v.shape)


# Fortran entry-point names of mis/basglb.f (argument order follows the Python API).
basglb = basic_to_global
glbbas = global_to_basic
gbtran = transforms_to_basic
