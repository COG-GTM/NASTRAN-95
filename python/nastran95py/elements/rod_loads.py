"""Rod thermal and enforced-deformation loads: port of ``ROD`` (``mis/rod.f``).

``EDTL`` calls ``ROD`` for every ROD, CONROD and TUBE element of the static load
step with an element temperature (``TEMPERATURE(LOAD)``, via ``SSGETD``) or an
enforced deformation (``DEFORM``, via ``FEDT``). The equivalent nodal loads act
along the element axis, opposite at the two grids, in each grid's global
(displacement) coordinate system (``BASGLB``)::

    VMAG = E*A*(DELTA + ALPHA*XL*(TI - TREF)) / XL
    P_A  = +VMAG * (X_A - X_B) / L       P_B = -P_A

``XL`` is what ``CALL NORM (VECT, XL)`` returns: the *squared* length ``L**2``
(``NORM`` in ``mis/mpyl.f`` stores ``X.X`` without the square root). The thermal
term is therefore the expected ``E*A*ALPHA*(TI - TREF)``, and the deformation
term is ``E*A*DELTA/L**2``. The port keeps the Fortran value.

E and ALPHA come from ``MAT`` (``INFLAG = 1``) at the element temperature word
of the ECPT record, as in the stiffness kernel. TUBE elements use the mean
circumference area ``PI*(D - T)*T``.

With ``real_words=True`` (default) every quantity the Fortran holds in a REAL
word, and every operation on it, is evaluated in float32 in the Fortran order,
including the CSTM origin and rotation read by ``BASGLB``, so the Python loads
match the gfortran build bit for bit. ``real_words=False`` evaluates the same
expressions in float64 with unrounded coordinate systems.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import RodGeometryError
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

LOAD_ELEMENTS = ("ROD", "TUBE")
"""ECPT layouts ``ROD`` handles (CROD and CONROD both use ``ROD``)."""


@dataclass(frozen=True)
class RodLoad:
    """Equivalent nodal loads of one element, grid A then grid B, in global coordinates."""

    eid: int
    sil: tuple[int, int]
    force: FloatArray
    """``(2, 3)`` translational loads at the two grids, in connection order."""

    def vector(self, sil_size: int) -> FloatArray:
        """The loads added into a length-``sil_size`` load vector (``CORE`` in ``EDTL``)."""
        return load_vector(np.asarray([self.sil]), self.force[None], sil_size)


def tube_area(diameter: float, thickness: float, real_words: bool = True) -> float:
    """TUBE section area ``PI*(D - T)*T`` as ``ROD`` computes it."""
    dt = np.float32 if real_words else np.float64
    d, t = dt(diameter), dt(thickness)
    return float(dt(np.pi) * (d - t) * t)


def _area(data: ElementData, real_words: bool) -> float:
    if data.element == "TUBE":
        return tube_area(data["od"], data["t"], real_words)  # type: ignore[arg-type]
    if data.element == "ROD":
        return float(data["area"])  # type: ignore[arg-type]
    raise ValueError(f"ROD loads are defined for ROD and TUBE records, not {data.element}")


def _rotation(cs: CoordSystem, dt) -> tuple[np.ndarray, np.ndarray]:
    """CSTM origin and ``T`` (local -> basic) as ``BASGLB`` reads them."""
    return cs.origin.astype(dt), cs.t.astype(dt)


def _mpyl_vec(v, a, dt):
    """``MPYL (A, V, 3, 3, 1, C)``: ``C(L) = SUM_M V(M)*A(M,L)``, accumulated from zero."""
    out = []
    for col in range(3):
        c = dt(0.0)
        for m in range(3):
            c = c + v[m] * a[m][col]
        out.append(c)
    return out


def _basglb(vin, pont, cs: CoordSystem, dt):
    """``BASGLB``: basic vector ``vin`` at basic point ``pont`` to global components."""
    ro, ti = _rotation(cs, dt)
    ti = [[ti[i, j] for j in range(3)] for i in range(3)]
    tz = ti
    if int(cs.ctype) != 1:
        pont1 = [pont[i] - ro[i] for i in range(3)]
        v = _mpyl_vec(pont1, ti, dt)
        r = np.sqrt(v[0] * v[0] + v[1] * v[1])
        if r != 0.0:
            tl = [[dt(0.0)] * 3 for _ in range(3)]
            if int(cs.ctype) == 2:
                tl[2][2] = dt(1.0)
                tl[0][0] = v[0] / r
                tl[1][1] = tl[0][0]
                tl[1][0] = v[1] / r
                tl[0][1] = -tl[1][0]
            else:
                xl = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                xr, yr, zl = v[0] / r, v[1] / r, v[2] / xl
                tl[0][0] = v[0] / xl
                tl[1][0] = v[1] / xl
                tl[2][0] = zl
                tl[0][1] = xr * zl
                tl[1][1] = yr * zl
                tl[2][1] = -r / xl
                tl[0][2] = -yr
                tl[1][2] = xr
            # MPYL (TL, TI, 3, 3, 3, TZ): TZ(N,L) = SUM_M TI(N,M)*TL(M,L)
            tz = [_mpyl_vec(ti[n], tl, dt) for n in range(3)]
    return _mpyl_vec(vin, tz, dt)


def rod_load(
    data: ElementData,
    materials: MaterialTable,
    temperature: float | None = None,
    deformation: float = 0.0,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> RodLoad:
    """Reference (statement-by-statement) port of ``ROD`` for one ROD or TUBE record.

    ``temperature`` is the load-set element temperature ``TI(1)`` (``None``: no
    thermal load, ``TBAR = 0``); ``deformation`` the ``DEFORM`` value ``DELTA``.
    """
    dt = np.float32 if real_words else np.float64
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    e, alpha, tref = dt(mat.e), dt(mat.alpha), dt(mat.tref)
    area = dt(_area(data, real_words))
    (ga, gb) = data.grids
    xa, xb = ga.xyz.astype(dt), gb.xyz.astype(dt)
    sil = tuple(int(s) for s in data["sil"])  # type: ignore[union-attr]
    tbar = dt(dt(temperature) - tref) if temperature is not None else dt(0.0)
    delta = dt(deformation)

    vect = [xa[i] - xb[i] for i in range(3)]
    xl = vect[0] * vect[0] + vect[1] * vect[1] + vect[2] * vect[2]
    if sil[0] == sil[1] or not xl > 0.0:
        raise RodGeometryError(f"rod element {data.eid} has illegal geometry or connections")
    w = dt(1.0) / np.sqrt(xl)
    vect = [x * w for x in vect]
    vmag = e * area * (delta + alpha * xl * tbar) / xl
    vect = [-x * vmag for x in vect]
    force = [-x for x in vect]
    if gb.cid != 0:
        vect = _basglb(vect, xb, _system(cstm, gb.cid), dt)
    if ga.cid != 0:
        force = _basglb(force, xa, _system(cstm, ga.cid), dt)
    return RodLoad(data.eid, sil, np.array([force, vect], dtype=float))  # type: ignore[arg-type]


def _system(cstm: Mapping[int, CoordSystem] | None, cid: int) -> CoordSystem:
    if cstm is None or cid not in cstm:
        raise KeyError(f"coordinate system {cid} is not in the CSTM")
    return cstm[cid]


@dataclass(frozen=True)
class RodLoadBatch:
    """Rod and tube load data as stacked arrays (``n`` elements)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 2)`` SILs in connection order."""
    cid: NDArray[np.int64]
    xyz: FloatArray
    """``(n, 2, 3)`` basic grid coordinates."""
    e: FloatArray
    area: FloatArray
    alpha: FloatArray
    tref: FloatArray
    temperature: FloatArray
    """Load-set element temperature ``TI(1)``; NaN where the element has none."""
    deformation: FloatArray

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable,
        temperatures: Mapping[int, float] | None = None,
        deformations: Mapping[int, float] | None = None,
        real_words: bool = True,
    ) -> RodLoadBatch:
        """Stack ``elements``; ``temperatures``/``deformations`` map EID to ``TI(1)``/``DELTA``."""
        temperatures = temperatures or {}
        deformations = deformations or {}
        mats = [materials.mat1(int(d["mid"]), d.temperature) for d in elements]  # type: ignore[arg-type]
        return cls(
            eid=np.array([d.eid for d in elements], dtype=np.int64),
            sil=np.array([d["sil"] for d in elements], dtype=np.int64).reshape(-1, 2),
            cid=np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(
                -1, 2
            ),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(
                -1, 2, 3
            ),
            e=np.array([m.e for m in mats], dtype=float),
            area=np.array([_area(d, real_words) for d in elements], dtype=float),
            alpha=np.array([m.alpha for m in mats], dtype=float),
            tref=np.array([m.tref for m in mats], dtype=float),
            temperature=np.array([temperatures.get(d.eid, np.nan) for d in elements], float),
            deformation=np.array([deformations.get(d.eid, 0.0) for d in elements], float),
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


def _mpyl_rows(v: np.ndarray, a: np.ndarray, dt) -> np.ndarray:
    """Batched ``_mpyl_vec``: ``v`` is ``(m, 3)``, ``a`` is ``(m, 3, 3)`` or ``(3, 3)``."""
    a = np.broadcast_to(a, (v.shape[0], 3, 3))
    out = np.empty_like(v)
    for col in range(3):
        c = np.zeros(v.shape[0], dtype=dt)
        for m in range(3):
            c = c + v[:, m] * a[:, m, col]
        out[:, col] = c
    return out


def _basglb_batch(vin: np.ndarray, pont: np.ndarray, cs: CoordSystem, dt) -> np.ndarray:
    """``BASGLB`` for ``(m, 3)`` vectors at ``(m, 3)`` points of one coordinate system."""
    ro, ti = _rotation(cs, dt)
    m = vin.shape[0]
    tz = np.broadcast_to(ti, (m, 3, 3)).copy()
    if int(cs.ctype) != 1:
        v = _mpyl_rows(pont - ro, ti, dt)
        r = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
        k = r != 0.0
        if k.any():
            vk, rk = v[k], r[k]
            tl = np.zeros((vk.shape[0], 3, 3), dtype=dt)
            if int(cs.ctype) == 2:
                tl[:, 2, 2] = 1.0
                tl[:, 0, 0] = vk[:, 0] / rk
                tl[:, 1, 1] = tl[:, 0, 0]
                tl[:, 1, 0] = vk[:, 1] / rk
                tl[:, 0, 1] = -tl[:, 1, 0]
            else:
                xl = np.sqrt(vk[:, 0] * vk[:, 0] + vk[:, 1] * vk[:, 1] + vk[:, 2] * vk[:, 2])
                xr, yr, zl = vk[:, 0] / rk, vk[:, 1] / rk, vk[:, 2] / xl
                tl[:, 0, 0] = vk[:, 0] / xl
                tl[:, 1, 0] = vk[:, 1] / xl
                tl[:, 2, 0] = zl
                tl[:, 0, 1] = xr * zl
                tl[:, 1, 1] = yr * zl
                tl[:, 2, 1] = -rk / xl
                tl[:, 0, 2] = -yr
                tl[:, 1, 2] = xr
            tik = np.broadcast_to(ti, (vk.shape[0], 3, 3))
            tz[k] = np.stack([_mpyl_rows(tik[:, n, :], tl, dt) for n in range(3)], axis=1)
    return _mpyl_rows(vin, tz, dt)


def rod_loads(
    batch: RodLoadBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> FloatArray:
    """Vectorised ``ROD``: ``(n, 2, 3)`` global loads at grid A and grid B of each element."""
    dt = np.float32 if real_words else np.float64
    xyz = batch.xyz.astype(dt)
    e, area, alpha, tref = (getattr(batch, n).astype(dt) for n in ("e", "area", "alpha", "tref"))
    has_t = ~np.isnan(batch.temperature)
    tbar = np.where(has_t, np.nan_to_num(batch.temperature).astype(dt) - tref, dt(0.0)).astype(dt)
    delta = batch.deformation.astype(dt)

    vect = xyz[:, 0] - xyz[:, 1]
    xl = vect[:, 0] * vect[:, 0] + vect[:, 1] * vect[:, 1] + vect[:, 2] * vect[:, 2]
    bad = (batch.sil[:, 0] == batch.sil[:, 1]) | ~(xl > 0.0)
    if bad.any():
        raise RodGeometryError(
            f"rod element {int(batch.eid[bad][0])} has illegal geometry or connections"
        )
    w = dt(1.0) / np.sqrt(xl)
    vect = vect * w[:, None]
    vmag = e * area * (delta + alpha * xl * tbar) / xl
    vect = -vect * vmag[:, None]
    force = np.stack([-vect, vect], axis=1)
    for side in (0, 1):
        cids = batch.cid[:, side]
        for cid in np.unique(cids[cids != 0]):
            sel = cids == cid
            force[sel, side] = _basglb_batch(
                force[sel, side], xyz[sel, side], _system(cstm, int(cid)), dt
            )
    return force.astype(float)


def load_vector(sil: NDArray[np.int64], force: FloatArray, sil_size: int) -> FloatArray:
    """Add ``(n, 2, 3)`` grid loads into a SIL-ordered vector (translations at SIL..SIL+2)."""
    p = np.zeros(sil_size)
    sil = np.asarray(sil).reshape(-1, 2)
    force = np.asarray(force).reshape(-1, 2, 3)
    for k in range(3):
        np.add.at(p, (sil - 1 + k).ravel(), force[:, :, k].ravel())
    return p
