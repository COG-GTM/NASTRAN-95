"""Rod element (CROD, CONROD) stiffness and mass: port of ``RODD`` (``mis/rodd.f``).

``RODD`` is the double-precision EMG routine (``RODS`` is its single-precision
twin, ``KROD`` the older SMA1 version). Both connection cards share one ECPT
layout (``EID, SIL1, SIL2, MID, A, J, C, NSM``, two BGPDT blocks, temperature),
so one kernel serves CROD (PROD property) and CONROD (inline properties).

For each rod with axis ``e = xB - xA`` and length ``L``:

- stiffness: axial ``KE = E*A/L`` on the translations and torsional
  ``TE = G*J/L`` on the rotations, ``h = TA^T e / L`` per grid with ``TA`` the
  grid's global-to-basic transform, so the 3x3 blocks are ``±S h_i h_j^T``;
- mass, ``ME = (RHO*A + NSM)*L/2``: lumped (``ME`` on each translation) or
  consistent (``2ME/3`` diagonal, ``2ME/6`` coupling, rotated to each grid's
  global system).

Rows and columns follow ascending SIL, as in the Fortran. Two paths are given:

- :func:`rodd`, a per-element reference that follows the Fortran statement by
  statement and returns the ``EMGOUT`` records (``RODD`` drops the rotations
  when ``J = 0`` and the translations when ``A = 0``);
- :func:`rod_matrices`, the vectorised kernel over a :class:`RodBatch`, which
  always returns full 12x12 stiffness and mass matrices.

With ``real_words=True`` (default) inputs are rounded to the 32-bit REAL words
the Fortran stores and the products ``RODD`` forms in REAL arithmetic
(``E*A``, ``G*J``, ``RHO*A+NSM``, grid coordinate differences and the
``TRANSD`` offset) are rounded to float32, which reproduces the Fortran to
float64 round-off. ``real_words=False`` evaluates everything in float64.

The heat-transfer branch of ``RODD`` (conductivity/capacity) is not ported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.elements.contract import ElementData
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

TRANSLATIONS = (0, 1, 2, 6, 7, 8)
"""Translational DOFs of the 12x12 rod matrices (grid 1: 0-2, grid 2: 6-8)."""

ICODE_TRANSLATION, ICODE_ROTATION, ICODE_ALL = 7, 56, 63


class RodGeometryError(ValueError):
    """Zero length or coincident SILs (Fortran user fatal message 3118)."""


@dataclass(frozen=True)
class EmgMatrix:
    """One ``EMGOUT`` record: the data words and their ``DICT`` description.

    ``values`` is the column-wise square matrix (``dict2 = 1``) or its diagonal
    (``dict2 = 2``); ``icode`` 7, 56 or 63 selects the translations, rotations
    or all six components of each grid.
    """

    values: FloatArray
    ndof: int
    icode: int
    dict2: int

    def square(self) -> FloatArray:
        if self.dict2 == 2:
            return np.diag(self.values)
        return self.values.reshape(self.ndof, self.ndof).T

    def full(self) -> FloatArray:
        """The record expanded to the 12x12 two-grid matrix."""
        if self.icode == ICODE_ALL:
            return self.square()
        comps = (0, 1, 2) if self.icode == ICODE_TRANSLATION else (3, 4, 5)
        idx = [*comps, *(c + 6 for c in comps)]
        out = np.zeros((12, 12))
        out[np.ix_(idx, idx)] = self.square()
        return out


@dataclass(frozen=True)
class RodMatrices:
    """``RODD`` output for one element; matrices are ``None`` when ``RODD`` skips them."""

    eid: int
    sils: tuple[int, int]
    """SILs in matrix order (ascending)."""
    stiffness: EmgMatrix | None
    mass: EmgMatrix | None
    ge: float

    def k(self) -> FloatArray:
        return self.stiffness.full() if self.stiffness else np.zeros((12, 12))

    def m(self) -> FloatArray:
        return self.mass.full() if self.mass else np.zeros((12, 12))


@dataclass(frozen=True)
class RodProperties:
    """Material and section values ``RODD`` uses: ``E, G, RHO, GE`` from ``MAT``, ``A, J, NSM``."""

    e: float
    g: float
    rho: float
    ge: float
    area: float
    j: float
    nsm: float


def _f32(x):
    return np.asarray(x, dtype=np.float32) if np.ndim(x) else np.float32(x)


def rod_properties(data: ElementData, materials: MaterialTable) -> RodProperties:
    """``CALL MAT`` with ``INFLAG = 1`` at the element temperature, plus the section words."""
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    return RodProperties(
        mat.e,
        mat.g,
        mat.rho,
        mat.ge,
        float(data["area"]),  # type: ignore[arg-type]
        float(data["j"]),  # type: ignore[arg-type]
        float(data["nsm"]),  # type: ignore[arg-type]
    )


def _check_rod(eid, sils, length) -> None:
    if sils[0] == sils[1] or not length > 0.0:
        raise RodGeometryError(f"rod element {eid} has illegal geometry or connections (3118)")


def _transd(cstm, cid: int, xyz, real_words: bool) -> FloatArray:
    """``TRANSD``: 3x3 global-to-basic matrix (row-major ``TA(9)`` as a 3x3 array)."""
    if cid == 0:
        return np.eye(3)
    return transforms_to_basic(cstm, cid, xyz, real_offset=real_words)[0]


def rodd(
    data: ElementData,
    materials: MaterialTable | RodProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    consistent: bool = False,
    real_words: bool = True,
) -> RodMatrices:
    """Per-element reference port of ``RODD`` (structural branch).

    ``consistent`` selects the consistent mass matrix (``ICMBAR > 0``);
    otherwise the lumped one. ``materials`` is a :class:`MaterialTable` (``MAT``
    is evaluated at the element temperature) or ready :class:`RodProperties`.
    """
    props = materials if isinstance(materials, RodProperties) else rod_properties(data, materials)
    cstm = {} if cstm is None else cstm
    eid = data.eid
    sil1, sil2 = (int(s) for s in data["sil"])  # type: ignore[union-attr]
    grid_a, grid_b = data.grids
    rnd = _f32 if real_words else float

    xa = [rnd(v) for v in grid_a.xyz]
    xb = [rnd(v) for v in grid_b.xyz]
    evect = [float(xb[i] - xa[i]) for i in range(3)]
    el = float(np.sqrt(evect[0] ** 2 + evect[1] ** 2 + evect[2] ** 2))
    _check_rod(eid, (sil1, sil2), el)

    e, g, rho, area, jfact, mu = (
        rnd(v) for v in (props.e, props.g, props.rho, props.area, props.j, props.nsm)
    )
    ke = float(e * area) / el
    me = float(rho * area + mu) * el / 2.0
    te = float(g * jfact) / el

    stiffness = None
    if ke != 0.0 or te != 0.0:
        ta = _transd(cstm, grid_a.cid, np.array(xa, dtype=float), real_words)
        tb = _transd(cstm, grid_b.cid, np.array(xb, dtype=float), real_words)
        ha = [sum(evect[i] * ta[i, j] for i in range(3)) / el for j in range(3)]
        hb = [sum(evect[i] * tb[i, j] for i in range(3)) / el for j in range(3)]
        if te == 0.0:
            icode, ndof = ICODE_TRANSLATION, 6
        elif ke == 0.0:
            icode, ndof = ICODE_ROTATION, 6
        else:
            icode, ndof = ICODE_ALL, 12
        ng, npart = ndof // 2, (ndof // 2) * ndof
        k = np.zeros(ndof * ndof)
        izero = 0
        for scale in (ke, te):
            if scale == 0.0:
                if ndof == 12:
                    izero = 39
                continue
            kha = [scale * h for h in ha]
            khb = [scale * h for h in hb]
            if sil2 < sil1:
                ibbz, iabz = izero, izero + ng
                ibaz = izero + npart
                iaaz = ibaz + ng
            else:
                iaaz, ibaz = izero, izero + ng
                iabz = izero + npart
                ibbz = iabz + ng
            for jj in range(3):
                for ii in range(3):
                    ij = ndof * jj + ii
                    k[ij + iaaz] = kha[ii] * ha[jj]
                    k[ij + ibaz] = -khb[ii] * ha[jj]
                    k[ij + iabz] = -kha[ii] * hb[jj]
                    k[ij + ibbz] = khb[ii] * hb[jj]
            if ndof == 12:
                izero = 39
        stiffness = EmgMatrix(k, ndof, icode, 1)

    mass = None
    if me != 0.0:
        if not consistent:
            mass = EmgMatrix(np.full(6, me), 6, ICODE_TRANSLATION, 2)
        else:
            me2 = 2.0 * me
            diag, off = np.eye(3) * (me2 / 3.0), np.eye(3) * (me2 / 6.0)
            mii, mjj, mij, mji = diag.copy(), diag.copy(), off.copy(), off.copy()
            gi, gj = (grid_a, grid_b) if sil2 > sil1 else (grid_b, grid_a)
            xi, xj = (xa, xb) if sil2 > sil1 else (xb, xa)
            if gi.cid != 0:
                t = _transd(cstm, gi.cid, np.array(xi, dtype=float), real_words)
                mii = (t.T @ mii) @ t
                mij = t.T @ off
                mji = off @ t
            if gj.cid != 0:
                t = _transd(cstm, gj.cid, np.array(xj, dtype=float), real_words)
                mjj = (t.T @ mjj) @ t
                mij, mji = mij @ t, t.T @ mji
            vals = np.zeros(36)
            for i in range(3):
                for blk, m in ((0, mii), (3, mij), (18, mji), (21, mjj)):
                    vals[blk + i] = m[0, i]
                    vals[blk + i + 6] = m[1, i]
                    vals[blk + i + 12] = m[2, i]
            mass = EmgMatrix(vals, 6, ICODE_TRANSLATION, 1)

    return RodMatrices(eid, (min(sil1, sil2), max(sil1, sil2)), stiffness, mass, props.ge)


@dataclass(frozen=True)
class RodBatch:
    """Rod elements as stacked arrays (``n`` elements)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 2)`` SILs in connection order."""
    cid: NDArray[np.int64]
    """``(n, 2)`` displacement coordinate systems of the grids."""
    xyz: FloatArray
    """``(n, 2, 3)`` basic grid coordinates."""
    e: FloatArray
    g: FloatArray
    rho: FloatArray
    ge: FloatArray
    area: FloatArray
    j: FloatArray
    nsm: FloatArray

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable | Sequence[RodProperties],
    ) -> RodBatch:
        if isinstance(materials, MaterialTable):
            props = [rod_properties(d, materials) for d in elements]
        else:
            props = list(materials)
        col = {
            name: np.array([getattr(p, name) for p in props], dtype=float)
            for name in ("e", "g", "rho", "ge", "area", "j", "nsm")
        }
        return cls(
            eid=np.array([d.eid for d in elements], dtype=np.int64),
            sil=np.array([d["sil"] for d in elements], dtype=np.int64).reshape(-1, 2),
            cid=np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(
                -1, 2
            ),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(
                -1, 2, 3
            ),
            **col,
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


@dataclass(frozen=True)
class RodBatchMatrices:
    """Batched rod matrices; rows/columns of each element in ascending-SIL order."""

    sils: NDArray[np.int64]
    """``(n, 2)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 12, 12)`` stiffness in each grid's global system."""
    m: FloatArray
    """``(n, 12, 12)`` mass (translations only)."""
    ge: FloatArray


def rod_matrices(
    batch: RodBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    consistent: bool = False,
    real_words: bool = True,
) -> RodBatchMatrices:
    """Vectorised ``RODD``: 12x12 stiffness and mass of every rod in ``batch``."""
    cstm = {} if cstm is None else cstm
    n = len(batch)
    dt = np.float32 if real_words else np.float64
    xyz = batch.xyz.astype(dt)
    evect = (xyz[:, 1] - xyz[:, 0]).astype(float)
    el = np.sqrt(evect[:, 0] ** 2 + evect[:, 1] ** 2 + evect[:, 2] ** 2)
    bad = (batch.sil[:, 0] == batch.sil[:, 1]) | ~(el > 0.0)
    if bad.any():
        raise RodGeometryError(
            f"rod elements {batch.eid[bad].tolist()} have illegal geometry or connections (3118)"
        )
    e, g, rho, area, jfact, mu = (
        getattr(batch, f).astype(dt) for f in ("e", "g", "rho", "area", "j", "nsm")
    )
    ke = (e * area).astype(float) / el
    me = (rho * area + mu).astype(float) * el / 2.0
    te = (g * jfact).astype(float) / el

    ta = transforms_to_basic(
        cstm, batch.cid.reshape(-1), xyz.reshape(-1, 3).astype(float), real_offset=real_words
    ).reshape(n, 2, 3, 3)
    h = np.einsum("ni,ngij->ngj", evect, ta) / el[:, None, None]
    swap = batch.sil[:, 1] < batch.sil[:, 0]
    order = np.where(swap[:, None], [1, 0], [0, 1])
    rows = np.arange(n)[:, None]
    h = h[rows, order]
    ta = ta[rows, order]

    k = np.zeros((n, 12, 12))
    for off, scale in ((0, ke), (3, te)):
        sh = scale[:, None, None] * h
        for a in range(2):
            for b in range(2):
                sign = 1.0 if a == b else -1.0
                blk = sign * sh[:, a, :, None] * h[:, b, None, :]
                k[:, 6 * a + off : 6 * a + off + 3, 6 * b + off : 6 * b + off + 3] = blk

    m = np.zeros((n, 12, 12))
    if consistent:
        me2 = 2.0 * me
        diag = (me2 / 3.0)[:, None, None] * np.eye(3)
        off = (me2 / 6.0)[:, None, None] * np.eye(3)
        ti, tj = ta[:, 0], ta[:, 1]
        tit, tjt = ti.transpose(0, 2, 1), tj.transpose(0, 2, 1)
        blocks = {
            (0, 0): (tit @ diag) @ ti,
            (0, 1): (tit @ off) @ tj,
            (1, 0): tjt @ (off @ ti),
            (1, 1): (tjt @ diag) @ tj,
        }
        # RODD stores each 3x3 product transposed into the column-wise record.
        for (a, b), blk in blocks.items():
            m[:, 6 * b : 6 * b + 3, 6 * a : 6 * a + 3] = blk.transpose(0, 2, 1)
    else:
        for i in TRANSLATIONS:
            m[:, i, i] = me
    sils = np.sort(batch.sil, axis=1)
    return RodBatchMatrices(sils, k, m, batch.ge.copy())


def assemble(elements: Sequence[tuple[tuple[int, int], FloatArray]], sil_size: int) -> FloatArray:
    """Add 12x12 two-grid matrices into a ``sil_size`` square matrix at their SILs."""
    out = np.zeros((sil_size, sil_size))
    for sils, mat in elements:
        idx = [s - 1 + c for s in sils for c in range(6)]
        out[np.ix_(idx, idx)] += mat
    return out
