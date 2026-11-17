"""Mean-plane geometry of the isoparametric quadrilateral membranes (CQDMEM1, CQDMEM2).

``QDMM1D``/``QDMM1S`` (EMG), ``QDMM1`` (EDTL thermal load) and ``SQDM11``
(SDR2 phase 1) share the same projection of a possibly warped quadrilateral
A, B, C, D onto its mean plane:

- ``k`` is the unit normal ``(C - A) x (D - B)``, ``i`` the projection of
  ``B - A`` on the plane and ``j = k x i``; ``E = [i j k]`` maps element axes
  to basic (``E(1), E(4), E(7)`` is ``i`` in the Fortran's row-major ``E(9)``);
- ``HH`` is minus half the offset of B from the plane through A (the measure of
  non-planarity, sign as corrected in the Fortran);
- in the mean plane A is the origin, B sits at ``(-X12, 0)``, C at
  ``(-X13, Y3A)`` and D at ``(-X14, Y4A)``;
- the 12x8 matrix ``B`` relates the mean-plane ``u, v`` of the four grids to
  the actual grid translations (it is the 8x8 identity on ``u, v`` plus ``HH``
  coupling terms into ``w``).

Every quantity is evaluated element-wise on ``(n,)`` arrays in the order of the
Fortran statements, in ``dtype`` (float32 for the REAL routines, float64 for
``QDMM1D``). The grid coordinates are REAL words and their differences are
taken in REAL arithmetic in every routine, including ``QDMM1D``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class QuadGeometryError(ValueError):
    """Degenerate or re-entrant quadrilateral (Fortran messages 30/31, 30/32, 30/26)."""


@dataclass(frozen=True)
class MeanPlane:
    """Mean-plane projection of ``n`` quadrilaterals (all fields ``(n,)`` unless noted)."""

    e: Array
    """``(n, 3, 3)``: ``E`` with columns ``i, j, k`` in basic."""
    hh: Array
    x12: Array
    x13: Array
    x14: Array
    x23: Array
    x24: Array
    x34: Array
    y3a: Array
    y4a: Array
    y34: Array
    magk: Array
    magi: Array
    magj: Array
    diag13: Array
    """``|C - A|`` (from the REAL differences, in ``dtype``)."""
    diag24: Array

    @property
    def lengths(self) -> tuple[Array, Array, Array, Array]:
        """Side lengths ``LA, LB, LC, LD`` in the mean plane."""
        la = np.abs(self.x12)
        lb = np.sqrt(self.x23 * self.x23 + self.y3a * self.y3a)
        lc = np.sqrt(self.x34 * self.x34 + self.y34 * self.y34)
        ld = np.sqrt(self.x14 * self.x14 + self.y4a * self.y4a)
        return la, lb, lc, ld


def mean_plane(
    xyz, dtype=np.float64, real_coords: bool = True, x14_from_x41: bool = False
) -> MeanPlane:
    """Project quadrilaterals with basic coordinates ``xyz`` (``(n, 4, 3)`` or ``(4, 3)``).

    ``x14_from_x41`` selects ``X14 = -(D - A).i`` and ``X24 = X14 - X12``
    (``QDMM1``, ``SQDM11``) instead of ``X24 = -(D - B).i`` and
    ``X14 = X12 + X24`` (``QDMM1D``/``QDMM1S``).
    """
    x = np.asarray(xyz, dtype=np.float32 if real_coords else np.float64)
    if x.ndim == 2:
        x = x[None]

    def diff(a: int, b: int):
        d = (x[:, a] - x[:, b]).astype(dtype)
        return d[:, 0], d[:, 1], d[:, 2]

    x21, y21, z21 = diff(1, 0)
    x31, y31, z31 = diff(2, 0)
    x41, y41, z41 = diff(3, 0)
    x42, y42, z42 = diff(3, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk1 = y31 * z42 - z31 * y42
        pk2 = z31 * x42 - x31 * z42
        pk3 = x31 * y42 - y31 * x42
        magk = np.sqrt(pk1 * pk1 + pk2 * pk2 + pk3 * pk3)
        pk1, pk2, pk3 = pk1 / magk, pk2 / magk, pk3 / magk
        hh = x21 * pk1 + y21 * pk2 + z21 * pk3
        pi1 = x21 - hh * pk1
        pi2 = y21 - hh * pk2
        pi3 = z21 - hh * pk3
        magi = np.sqrt(pi1 * pi1 + pi2 * pi2 + pi3 * pi3)
        pi1, pi2, pi3 = pi1 / magi, pi2 / magi, pi3 / magi
        hh = -hh / dtype(2.0)
        pj1 = pk2 * pi3 - pk3 * pi2
        pj2 = pk3 * pi1 - pk1 * pi3
        pj3 = pk1 * pi2 - pk2 * pi1
        magj = np.sqrt(pj1 * pj1 + pj2 * pj2 + pj3 * pj3)
        pj1, pj2, pj3 = pj1 / magj, pj2 / magj, pj3 / magj
    e = np.stack(
        [
            np.stack([pi1, pj1, pk1], -1),
            np.stack([pi2, pj2, pk2], -1),
            np.stack([pi3, pj3, pk3], -1),
        ],
        axis=1,
    )
    x12 = -(x21 * pi1 + y21 * pi2 + z21 * pi3)
    x13 = -(x31 * pi1 + y31 * pi2 + z31 * pi3)
    y3a = x31 * pj1 + y31 * pj2 + z31 * pj3
    y4a = x42 * pj1 + y42 * pj2 + z42 * pj3
    if x14_from_x41:
        x14 = -(x41 * pi1 + y41 * pi2 + z41 * pi3)
        x24 = x14 - x12
    else:
        x24 = -(x42 * pi1 + y42 * pi2 + z42 * pi3)
        x14 = x12 + x24
    return MeanPlane(
        e=e,
        hh=hh,
        x12=x12,
        x13=x13,
        x14=x14,
        x23=x13 - x12,
        x24=x24,
        x34=x14 - x13,
        y3a=y3a,
        y4a=y4a,
        y34=y3a - y4a,
        magk=magk,
        magi=magi,
        magj=magj,
        diag13=np.sqrt(x31 * x31 + y31 * y31 + z31 * z31),
        diag24=np.sqrt(x42 * x42 + y42 * y42 + z42 * z42),
    )


def mean_plane_b(mp: MeanPlane) -> Array:
    """The ``(n, 12, 8)`` matrix ``B`` (row-major ``B(96)`` of the Fortran).

    Rows are ``u, v, w`` of grids A-D, columns the mean-plane ``u, v`` of A-D.
    """
    dt = mp.hh.dtype.type
    hh = mp.hh
    la, lb, lc, ld = mp.lengths
    with np.errstate(divide="ignore", invalid="ignore"):
        cth1 = -mp.x14 / ld
        sth1 = mp.y4a / ld
        cth2 = mp.x23 / lb
        sth2 = mp.y3a / lb
        cth31 = mp.x34 / lc
        sth31 = -mp.y34 / lc
        cth41, sth41 = cth1, sth1
        cth32, sth32 = sth2, cth2
        cth42, sth42 = sth31, cth31
        dlt1 = cth31 * cth32 - sth31 * sth32
        dlt2 = cth42 * cth41 + sth41 * sth42
        ldd2 = ld * dlt2
        lbd1 = lb * dlt1
        lcd1 = lc * dlt1
        lcd2 = lc * dlt2
        b17 = -hh / la
        terms = {
            1: dt(1.0),
            10: dt(1.0),
            17: b17,
            18: -hh / (ld * sth1) + ((hh * cth1) / (la * sth1)),
            19: hh / la,
            20: (hh * cth2) / (la * sth2),
            23: (hh * cth42) / ldd2,
            24: (hh * sth42) / ldd2,
            27: dt(1.0),
            36: dt(1.0),
            41: -b17,
            42: (-hh * cth1) / (la * sth1),
            43: b17,
            44: ((-hh * cth2) / (la * sth2)) + (hh / (lb * sth2)),
            45: (-hh * sth31) / lbd1,
            46: (-hh * cth31) / lbd1,
            53: dt(1.0),
            62: dt(1.0),
            68: -hh / (lb * sth2),
            69: hh * ((sth31 / lbd1) + (cth32 / lcd1)),
            70: hh * ((cth31 / lbd1) + (sth32 / lcd1)),
            71: (-hh * sth41) / lcd2,
            72: (hh * cth41) / lcd2,
            79: dt(1.0),
            88: dt(1.0),
            90: hh / (ld * sth1),
            93: (-hh * cth32) / lcd1,
            94: (-hh * sth32) / lcd1,
            95: hh * ((-cth42 / ldd2) + (sth41 / lcd2)),
            96: hh * ((-sth42 / ldd2) - (cth41 / lcd2)),
        }
    b = np.zeros((hh.shape[0], 12, 8), dtype=hh.dtype)
    for idx, value in terms.items():
        b[:, (idx - 1) // 8, (idx - 1) % 8] = value
    return b
