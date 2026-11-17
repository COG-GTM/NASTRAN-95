"""Shear panel thermal loads: a port of ``TSHEAR`` (``mis/tshear.f``, called from ``EDTL``).

``TSHEAR`` loads each of the four panel edges as a pseudo-rod. Edge ``i`` runs from
grid ``ia = i`` to grid ``ib = i + 1`` (edge 4 closes back to grid 1) and carries the
mean temperature of its two grids, ``TBAR = (T(ia) + T(ib))/2 - TREF``. Its
extensional area is

    AREA = F * PA * T / (XL(i12) + XL(i12 + 2))

where ``i12`` is 1 for edges 1 and 3 and 2 for edges 2 and 4, ``F = 1`` (NASTRAN-95
``PSHEAR`` has no F1/F2 edge-effectiveness factors, and ``TSHEAR`` sets ``F12 = 1``),
``PA = 0.5*VKL`` and ``T`` is the panel thickness. The edge force ``E*AREA*ALPHA*TBAR``
acts along the unit edge vector ``(X(ia) - X(ib))/|X(ia) - X(ib)|``: ``+`` at grid ``ia``
and ``-`` at grid ``ib``, transformed to the grid's global system by ``BASGLB``.

Two Fortran values are kept as they are, for parity with the reference:

* ``NORM`` (``mis/mpyl.f``) returns the *squared* length of the vector it normalizes,
  so ``XL`` are squared edge lengths and ``VKL = |D1 x D2|**2`` is the squared norm of
  the diagonal cross product (twice the projected area, squared).
* ``BASGLB`` is passed ``ISILS(IB)`` as the point ``PONT`` (``CSID(1,IB)`` is only the
  coordinate system ID word). For cylindrical and spherical systems the transformation
  is therefore evaluated at the three ``/TRIMEX/`` words that follow ``ISILS(ib)``
  (SILs, MID and thickness) read as REALs, not at the grid. ``fortran_pont=False``
  evaluates it at the grid coordinates instead.

Where both edges of an opposite pair have zero length the Fortran divides 0/0; the
port raises :class:`ShearGeometryError` (2026) instead of returning NaN loads.

The routine has no deformation (``DEFORM``) input: its comment marks it ``NOT USED``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod_loads import _basglb, _basglb_batch, _system, load_vector
from nastran95py.elements.shear import ShearGeometryError
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

_NEXT = (1, 2, 3, 0)
_I12 = (0, 1, 0, 1)


@dataclass(frozen=True)
class ShearLoad:
    """Equivalent nodal loads of one shear panel, in connection order and global components."""

    eid: int
    sil: tuple[int, int, int, int]
    force: FloatArray
    """``(4, 3)`` translational loads at the four grids."""

    def vector(self, sil_size: int) -> FloatArray:
        """The loads added into a length-``sil_size`` load vector (``CORE`` in ``EDTL``)."""
        return load_vector(np.asarray([self.sil]), self.force[None], sil_size)


def _trimex_words(sil: NDArray[np.int64], mid: NDArray[np.int64], t: FloatArray) -> np.ndarray:
    """``(n, 6)`` float32 ``/TRIMEX/`` words 2-7 (``ISILS(4)``, ``MID``, ``THICK``) as REALs."""
    sil32 = np.ascontiguousarray(sil, dtype=np.int32).reshape(-1, 4).view(np.float32)
    mid32 = np.ascontiguousarray(mid, dtype=np.int32).reshape(-1, 1).view(np.float32)
    t32 = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([sil32, mid32, t32], axis=1)


def _pont(words: np.ndarray, xyz: np.ndarray, fortran_pont: bool, dt) -> np.ndarray:
    """``(n, 4, 3)`` points handed to ``BASGLB`` for each grid."""
    if not fortran_pont:
        return xyz.astype(dt)
    return np.stack([words[:, k : k + 3] for k in range(4)], axis=1).astype(dt)


def _check_denominator(eid: int, den) -> None:
    if not den > 0.0:
        raise ShearGeometryError(eid, 2026)


def shear_load(
    data: ElementData,
    materials: MaterialTable,
    temperatures: Sequence[float],
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
    fortran_pont: bool = True,
) -> ShearLoad:
    """Reference (statement-by-statement) port of ``TSHEAR`` for one CSHEAR record.

    ``temperatures`` are the four grid temperatures ``TI(2..5)`` that ``SSGETD``
    returns for the load set, in connection order.
    """
    dt = np.float32 if real_words else np.float64
    if data.element != "SHEAR":
        raise ValueError(f"TSHEAR loads are defined for SHEAR records, not {data.element}")
    if len(temperatures) != 4:
        raise ValueError("a shear panel needs four grid temperatures")
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    e1, alpha, to1 = dt(mat.e), dt(mat.alpha), dt(mat.tref)
    thick = dt(data["t"])  # type: ignore[arg-type]
    ti = [dt(x) for x in temperatures]
    sil = tuple(int(s) for s in data["sil"])  # type: ignore[union-attr]
    xyz = np.array([g.xyz for g in data.grids], dtype=float).astype(dt)
    words = _trimex_words(np.array(sil), np.array([int(data["mid"])]), np.array([thick]))  # type: ignore[arg-type]
    pont = _pont(words, xyz[None], fortran_pont, dt)[0]

    vec, xl = [], []
    for i in range(4):
        v = [xyz[i, j] - xyz[_NEXT[i], j] for j in range(3)]
        s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if s != 0.0:
            w = dt(1.0) / np.sqrt(s)
            v = [x * w for x in v]
        vec.append(v)
        xl.append(s)

    d1 = [xyz[2, j] - xyz[0, j] for j in range(3)]
    d2 = [xyz[3, j] - xyz[1, j] for j in range(3)]
    nrm = [
        d1[1] * d2[2] - d1[2] * d2[1],
        d1[2] * d2[0] - d1[0] * d2[2],
        d1[0] * d2[1] - d1[1] * d2[0],
    ]
    vkl = nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]
    pa = dt(0.5) * vkl
    f12 = dt(1.0)

    force = [[dt(0.0)] * 3 for _ in range(4)]
    for i in range(4):
        i12, ia, ib = _I12[i], i, _NEXT[i]
        tbar = (ti[ia] + ti[ib]) / dt(2.0) - to1
        den = xl[i12] + xl[i12 + 2]
        _check_denominator(data.eid, den)
        area = f12 * pa * thick / den
        vmag = e1 * area * alpha * tbar
        veca = [vmag * vec[i][j] for j in range(3)]
        vecb = [-x for x in veca]
        gb, ga = data.grids[ib], data.grids[ia]
        if gb.cid != 0:
            vecb = _basglb(vecb, list(pont[ib]), _system(cstm, gb.cid), dt)
        force[ib] = [force[ib][j] + vecb[j] for j in range(3)]
        if ga.cid != 0:
            veca = _basglb(veca, list(pont[ia]), _system(cstm, ga.cid), dt)
        force[ia] = [force[ia][j] + veca[j] for j in range(3)]
    return ShearLoad(data.eid, sil, np.array(force, dtype=float))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ShearLoadBatch:
    """Shear panel thermal load data as stacked arrays (``n`` elements)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 4)`` SILs in connection order."""
    cid: NDArray[np.int64]
    xyz: FloatArray
    """``(n, 4, 3)`` basic grid coordinates."""
    mid: NDArray[np.int64]
    t: FloatArray
    e: FloatArray
    alpha: FloatArray
    tref: FloatArray
    temperature: FloatArray
    """``(n, 4)`` load-set grid temperatures ``TI(2..5)``."""

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable,
        temperatures: Mapping[int, Sequence[float]],
    ) -> ShearLoadBatch:
        """Stack ``elements``; ``temperatures`` maps EID to the four grid temperatures."""
        missing = [d.eid for d in elements if d.eid not in temperatures]
        if missing:
            raise KeyError(f"no grid temperatures for shear panels {missing}")
        mats = [materials.mat1(int(d["mid"]), d.temperature) for d in elements]  # type: ignore[arg-type]
        return cls(
            eid=np.array([d.eid for d in elements], dtype=np.int64),
            sil=np.array([d["sil"] for d in elements], dtype=np.int64).reshape(-1, 4),
            cid=np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(
                -1, 4
            ),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(
                -1, 4, 3
            ),
            mid=np.array([d["mid"] for d in elements], dtype=np.int64),
            t=np.array([d["t"] for d in elements], dtype=float),
            e=np.array([m.e for m in mats], dtype=float),
            alpha=np.array([m.alpha for m in mats], dtype=float),
            tref=np.array([m.tref for m in mats], dtype=float),
            temperature=np.array([temperatures[d.eid] for d in elements], dtype=float).reshape(
                -1, 4
            ),
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


def _transform(
    vin: np.ndarray,
    pont: np.ndarray,
    cid: NDArray[np.int64],
    cstm: Mapping[int, CoordSystem] | None,
    dt,
) -> np.ndarray:
    out = vin.copy()
    for c in np.unique(cid):
        if c == 0:
            continue
        k = cid == c
        out[k] = _basglb_batch(vin[k], pont[k], _system(cstm, int(c)), dt)
    return out


def shear_loads(
    batch: ShearLoadBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
    fortran_pont: bool = True,
) -> FloatArray:
    """Vectorised ``TSHEAR``: ``(n, 4, 3)`` grid loads in global components."""
    dt = np.float32 if real_words else np.float64
    n = len(batch)
    xyz = batch.xyz.astype(dt)
    thick = batch.t.astype(dt)
    e1, alpha, to1 = batch.e.astype(dt), batch.alpha.astype(dt), batch.tref.astype(dt)
    ti = batch.temperature.astype(dt)
    pont = _pont(_trimex_words(batch.sil, batch.mid, batch.t), xyz, fortran_pont, dt)

    vec = xyz - xyz[:, list(_NEXT)]
    xl = vec[..., 0] * vec[..., 0] + vec[..., 1] * vec[..., 1] + vec[..., 2] * vec[..., 2]
    nz = xl != 0.0
    w = np.ones_like(xl)
    w[nz] = dt(1.0) / np.sqrt(xl[nz])
    vec = np.where(nz[..., None], vec * w[..., None], vec)

    d1 = xyz[:, 2] - xyz[:, 0]
    d2 = xyz[:, 3] - xyz[:, 1]
    nrm = np.stack(
        [
            d1[:, 1] * d2[:, 2] - d1[:, 2] * d2[:, 1],
            d1[:, 2] * d2[:, 0] - d1[:, 0] * d2[:, 2],
            d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0],
        ],
        axis=1,
    )
    vkl = nrm[:, 0] * nrm[:, 0] + nrm[:, 1] * nrm[:, 1] + nrm[:, 2] * nrm[:, 2]
    pa = dt(0.5) * vkl
    f12 = dt(1.0)

    force = np.zeros((n, 4, 3), dtype=dt)
    for i in range(4):
        i12, ia, ib = _I12[i], i, _NEXT[i]
        tbar = (ti[:, ia] + ti[:, ib]) / dt(2.0) - to1
        den = xl[:, i12] + xl[:, i12 + 2]
        bad = ~(den > 0.0)
        if bad.any():
            _check_denominator(int(batch.eid[bad][0]), den[bad][0])
        area = f12 * pa * thick / den
        vmag = e1 * area * alpha * tbar
        veca = vmag[:, None] * vec[:, i]
        vecb = -veca
        force[:, ib] = force[:, ib] + _transform(vecb, pont[:, ib], batch.cid[:, ib], cstm, dt)
        force[:, ia] = force[:, ia] + _transform(veca, pont[:, ia], batch.cid[:, ia], cstm, dt)
    return force.astype(float)
