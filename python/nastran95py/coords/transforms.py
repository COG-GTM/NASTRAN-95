"""Coordinate system table (CSTM) and global-to-basic transforms.

A CSTM record is 14 words: ``CID, TYPE, X0, Y0, Z0, T11, T12, ..., T33`` where
``T`` (stored by rows) maps a vector from the local rectangular frame of the
system to basic. ``TYPE`` is 1 rectangular, 2 cylindrical, 3 spherical.

``transform_to_basic`` ports the ``TRANSS``/``TRANSD`` entries of
``mis/pretrs.f`` / ``mis/pretrd.f``: given a coordinate system ID and a point in
basic coordinates it returns the 3x3 matrix ``TA`` mapping a vector expressed in
that system's global (displacement) directions at the point to basic.
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


def transform_to_basic(cstm: Mapping[int, CoordSystem], cid: int, point: ArrayLike) -> FloatArray:
    """Port of ``TRANSS``/``TRANSD``: global-to-basic matrix ``TA`` at ``point``.

    ``point`` is in basic coordinates. CID 0 returns the identity. For
    cylindrical and spherical systems the local axes are evaluated at the point;
    on the polar axis (``r == 0``) the rectangular matrix is returned, as in the
    Fortran.
    """
    if cid == 0:
        return np.eye(3)
    try:
        cs = cstm[cid]
    except KeyError:
        raise CstmError(cid) from None
    if cs.ctype == CoordType.RECTANGULAR:
        return cs.t.copy()

    x, y, z = cs.t.T @ (np.asarray(point, dtype=float).reshape(3) - cs.origin)
    r = np.hypot(x, y)
    if r == 0.0:
        return cs.t.copy()
    if cs.ctype == CoordType.CYLINDRICAL:
        tl = np.array(
            [
                [x / r, -y / r, 0.0],
                [y / r, x / r, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    else:
        xl = np.sqrt(x * x + y * y + z * z)
        tl = np.array(
            [
                [x / xl, x * z / (r * xl), -y / r],
                [y / xl, y * z / (r * xl), x / r],
                [z / xl, -r / xl, 0.0],
            ]
        )
    return cs.t @ tl


def global_to_basic(
    cstm: Mapping[int, CoordSystem], cid: int, point: ArrayLike, vector: ArrayLike
) -> FloatArray:
    """Map a vector in the global directions of ``cid`` at ``point`` to basic."""
    return transform_to_basic(cstm, cid, point) @ np.asarray(vector, dtype=float)


def basic_to_global(
    cstm: Mapping[int, CoordSystem], cid: int, point: ArrayLike, vector: ArrayLike
) -> FloatArray:
    """Map a basic vector at ``point`` to the global directions of ``cid``."""
    return transform_to_basic(cstm, cid, point).T @ np.asarray(vector, dtype=float)
