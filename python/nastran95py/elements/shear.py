"""Shear panel (CSHEAR) stiffness and mass: port of ``SHEARD`` (``mis/sheard.f``).

``SHEARD`` is the double-precision EMG routine (``SHEARS`` is its single-precision
twin). The ECPT is ``EID, SIL1-4, MID, T, NSM``, four BGPDT blocks and the
element temperature (PSHEAR has only ``MID, T, NSM`` in NASTRAN-95).

The panel may be warped: ``SHEARD`` projects it onto the mean plane normal to
``(x3 - x1) x (x4 - x2)``, with ``i`` along the projected edge 1-2. In that
plane it picks one of four flexibility formulas for ``Z`` depending on which
pairs of opposite edges are parallel (within 1%), and builds the 12x12
translational stiffness ``K_ij = a_i a_j / (2Z) (T_i^T s_i)(T_j^T s_j)^T``,
``s`` the unit diagonal directions and ``a`` the corner forces of a unit shear
flow. Rows and columns follow ascending SIL.

The mass is ``EMADTQ(6)``: per corner, one third of the mean of the two corner
triangle areas and the diagonal-product area, times ``RHO*T + NSM``, on its
three translations (``ISHEAR`` mass is diagonal only).

Two paths are given:

- :func:`sheard`, a per-element reference following the Fortran statement by
  statement and returning the ``EMGOUT`` records;
- :func:`shear_matrices`, the vectorised kernel over a :class:`ShearBatch`.

With ``real_words=True`` (default) the inputs are rounded to the REAL words the
Fortran stores, and every intermediate ``SHEARD``/``EMADTQ`` keeps in REAL
(grid coordinate differences, ``VKL``, ``PA``, ``V12DK``, ``VP12L``, ``VJL``,
``XQ``, the CSTM words ``TRANSD`` reads) is rounded to float32 at the same
point, which reproduces the Fortran to float64 round-off. ``real_words=False``
keeps everything in float64.

Bad geometry raises :class:`ShearGeometryError` with the text of user fatal
message 2026 (``MESAGE(30,26)``) or 2027 (``MESAGE(30,27)``, a non-convex
corner, numbered 1-4 as in the Fortran). The heat-transfer branch of
``EMADTQ`` is not ported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import ICODE_TRANSLATION, EmgMatrix, real_cstm
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

UFM = "*** USER FATAL MESSAGE"
C23 = 2.0 / 3.0
EP = 0.01

BRANCH_PARALLEL_12_34 = 30
"""Edge 3-4 parallel to edge 1-2 (Fortran label 30)."""
BRANCH_PARALLEL_14_23 = 40
"""Edge 1-4 parallel to edge 2-3 (label 40)."""
BRANCH_PARALLELOGRAM = 50
"""Both pairs parallel (label 50)."""
BRANCH_GENERAL = 70
"""No parallel edges (label 70)."""

# Corner triangles EMADTQ uses for each pivot (0-based grid indices P1, P2, P3).
_MASS_TRIANGLES = ((0, 1, 3), (0, 1, 2), (3, 1, 2), (0, 3, 2))


class ShearGeometryError(ValueError):
    """Bad CSHEAR geometry: user fatal message 2026 or 2027."""

    def __init__(self, eid: int, code: int, corner: int | None = None):
        self.eid, self.code, self.corner = eid, code, corner
        if code == 2026:
            text = f"{UFM} 2026,ELEMENT{eid:9d} GEOMETRY YIELDS UNREASONABLE MATRIX"
        else:
            text = (
                f"{UFM} 2027,ELEMENT{eid:9d} HAS INTERIOR ANGLE GREATER THAN"
                f" 180 DEG. AT GRID POINT{corner:9d}"
            )
        super().__init__(text)


@dataclass(frozen=True)
class ShearProperties:
    """Values ``SHEARD`` uses: ``E, G, NU, RHO, GE`` from ``MAT`` and ``T, NSM``."""

    e: float
    g: float
    nu: float
    rho: float
    ge: float
    t: float
    nsm: float


@dataclass(frozen=True)
class ShearMatrices:
    """``SHEARD`` output for one element (``EMGOUT`` records in ascending-SIL order)."""

    eid: int
    sils: tuple[int, int, int, int]
    stiffness: EmgMatrix
    mass: EmgMatrix
    ge: float
    branch: int

    def k(self) -> FloatArray:
        return self.stiffness.square()

    def m(self) -> FloatArray:
        return self.mass.square()


def shear_properties(data: ElementData, materials: MaterialTable) -> ShearProperties:
    """``CALL MAT`` (``MATFLG = 1``) at the element temperature, plus ``T`` and ``NSM``."""
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    rho = materials.density(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    return ShearProperties(
        mat.e,
        mat.g,
        mat.nu,
        rho,
        mat.ge,
        float(data["t"]),  # type: ignore[arg-type]
        float(data["nsm"]),  # type: ignore[arg-type]
    )


def _r(x: float, real_words: bool) -> float:
    return float(np.float32(x)) if real_words else float(x)


def _sil_order(sils: Sequence[int]) -> list[int]:
    """``IPART`` after the Fortran exchange sort: grids by ascending SIL (ties kept)."""
    return sorted(range(4), key=lambda i: (sils[i], i))


def _flexibility(eid, x2, x3, y3, x4, y4, pa, g, t, nuc, real_words=True) -> tuple[float, int]:
    """Convexity checks and the shear flexibility ``Z`` (statements 30-70)."""
    x1 = y1 = y2 = 0.0
    if y3 <= 0.0:
        raise ShearGeometryError(eid, 2027, 2)
    if y4 <= 0.0:
        raise ShearGeometryError(eid, 2027, 1)
    if x3 <= y3 * x4 / y4:
        raise ShearGeometryError(eid, 2027, 4)
    if x4 >= x2 - (x2 - x3) * y4 / y3:
        raise ShearGeometryError(eid, 2027, 3)
    cep1 = abs(y3 - y4)
    cepx = abs(x3 - x4)
    temp = x3 - x2
    cep2 = abs(y4 * temp - y3 * x4)
    cepy = abs(x4 * temp + y4 * y3)
    par12 = cep1 < EP * cepx
    par14 = cep2 < EP * cepy
    if par12 and not par14:
        temp = y3 * x4 - y4 * (x3 - x2)
        yp = x2 * y3 * y4 / temp
        p = [yp - y1, yp - y2, yp - y3, yp - y4]
        xp = x2 * y3 * x4 / temp
        sa = (x2 - xp) / yp
        c = (x1 - xp) / yp
        z = ((p[0] * p[1] * pa) / (p[2] * p[3] * 2.0 * g * t)) * (
            1.0 + C23 * nuc * (sa * sa + sa * c + c * c)
        )
        return z, BRANCH_PARALLEL_12_34
    if par14 and not par12:
        d = -0.5 * (x4 / y4 + (x3 - x2) / y3)
        xq = _r(x4 - y4 * (x3 - x4) / (y3 - y4), real_words)
        temp = 1.0 / math.sqrt(1.0 + d * d)
        p = [(xq - xi - d * yi) * temp for xi, yi in ((x1, y1), (x2, y2), (x3, y3), (x4, y4))]
        temp = xq - x4
        b = (temp * d + y4) / (temp - y4 * d)
        z = ((p[0] * p[1] * pa) / (p[2] * p[3] * 2.0 * g * t)) * (
            1.0 + C23 * nuc * (b * b + b * d + d * d)
        )
        return z, BRANCH_PARALLEL_14_23
    if par12 and par14:
        d = -0.5 * (x4 / y4 + (x3 - x2) / y3 + (y3 - y4) / (x3 - x4))
        z = pa / (2.0 * g * t) * (1.0 + 2.0 * (d * d) * nuc)
        return z, BRANCH_PARALLELOGRAM
    xq = _r(x4 - (x3 - x4) / (y3 - y4) * y4, real_words)
    temp = y3 * x4 - y4 * (x3 - x2)
    xp = x2 * y3 * x4 / temp
    yp = x2 * y3 * y4 / temp
    xl = math.sqrt((xq - xp) * (xq - xp) + yp * yp)
    d = (xq - xp) / yp
    temp = yp / xl
    p = [temp * (xq - xi - d * yi) for xi, yi in ((x1, y1), (x2, y2), (x3, y3), (x4, y4))]
    c = xl / p[0] - d
    b = xl / p[3] - c
    a = xl / p[1] - d
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d
    a3, b3, c3, d3 = a2 * a, b2 * b, c2 * c, d2 * d
    a4, b4, c4, d4 = a3 * a, b3 * b, c3 * c, d3 * d
    a5, b5, c5, d5 = a4 * a, b4 * b, c4 * c, d4 * d
    temp = 0.5 * p[0] * p[1] * p[2] * p[3] / (xl * xl)
    term = (a + b + C23 * (a3 + b3) + 0.2 * (a5 + b5)) * math.log(abs(a + b))
    term1 = (c + d + C23 * (c3 + d3) + 0.2 * (c5 + d5)) * math.log(abs(c + d))
    term2 = (b + c + C23 * (b3 + c3) + 0.2 * (b5 + c5)) * math.log(abs(b + c))
    term3 = (d + a + C23 * (d3 + a3) + 0.2 * (d5 + a5)) * math.log(abs(d + a))
    term4 = 0.1 * ((a2 - c2) * (b3 - d3) + (b2 - d2) * (a3 - c3))
    term5 = 0.2 * ((a - c) * (b4 - d4) + (b - d) * (a4 - c4))
    f = temp * (term + term1 - term2 - term3 + term4 - term5)
    z = p[0] * p[1] / (p[2] * p[3] * 2.0 * g * t) * (pa + 4.0 * nuc * (f - C23 * pa))
    return z, BRANCH_GENERAL


def _sub(a, b, real_words: bool) -> list[float]:
    return [_r(a[i] - b[i], real_words) for i in range(3)]


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _cross(a, b) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _stiffness(eid, xyz, cids, props, cstm, real_words) -> tuple[FloatArray, int]:
    """``KE`` (grid order, 12x12) and the flexibility branch."""
    g, t, nu = (_r(v, real_words) for v in (props.g, props.t, props.nu))
    if t * g == 0.0:
        raise ShearGeometryError(eid, 2026)
    nuc = 1.0 / (1.0 + nu)
    gp1, gp2, gp3, gp4 = xyz
    vd1 = _sub(gp3, gp1, real_words)
    vd2 = _sub(gp4, gp2, real_words)
    vkn = _cross(vd1, vd2)
    vkl = _r(_norm(vkn), real_words)
    if vkl == 0.0:
        raise ShearGeometryError(eid, 2026)
    vk = [v / vkl for v in vkn]
    pa = _r(vkl / 2.0, real_words)
    v12 = _sub(gp2, gp1, real_words)
    v41 = _sub(gp1, gp4, real_words)
    v12dk = _r(_dot(v12, vk), real_words)
    vp12 = [v12[i] - v12dk * vk[i] for i in range(3)]
    vp12l = _r(_norm(vp12), real_words)
    if vp12l == 0.0:
        raise ShearGeometryError(eid, 2026)
    vi = [v / vp12l for v in vp12]
    vj = _cross(vk, vi)
    vjl = _r(_norm(vj), real_words)
    if vjl == 0.0:
        raise ShearGeometryError(eid, 2026)
    vj = [v / vjl for v in vj]
    x2 = vp12l
    x3 = _dot(vi, vd1)
    y3 = _dot(vj, vd1)
    x4 = -vi[0] * v41[0] - vi[1] * v41[1] - vi[2] * v41[2]
    y4 = -vj[0] * v41[0] - vj[1] * v41[1] - vj[2] * v41[2]
    z, branch = _flexibility(eid, x2, x3, y3, x4, y4, pa, g, t, nuc, real_words)
    xl13 = math.sqrt(x3 * x3 + y3 * y3)
    xl24 = math.sqrt((x4 - x2) * (x4 - x2) + y4 * y4)
    su = [x3 / xl13, (x4 - x2) / xl24]
    sv = [y3 / xl13, y4 / xl24]
    su, sv = su + su, sv + sv
    temp = x4 * y3 - x3 * y4
    a1 = -0.5 * x2 * y4 * xl13 / temp
    a2 = 0.5 * x2 * y3 * xl24 / (temp - x2 * (y3 - y4))
    avec = [a1, a2, -a1, -a2]
    vec = []
    for i in range(4):
        v = np.array([vi[k] * su[i] + vj[k] * sv[i] for k in range(3)])
        if cids[i] != 0:
            ti = transforms_to_basic(cstm, cids[i], xyz[i], real_offset=real_words)[0]
            v = np.array([sum(ti[r, k] * v[r] for r in range(3)) for k in range(3)])
        vec.append(v)
    ke = np.zeros((12, 12))
    for i in range(4):
        con = avec[i] / (2.0 * z)
        for j in range(4):
            blk = np.outer(vec[i], vec[j])
            ke[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = con * blk * avec[j]
    return ke, branch


def emadtq_shear(xyz, props: ShearProperties, real_words: bool = True) -> FloatArray:
    """``EMADTQ(6)``: the 12 diagonal mass terms in grid order."""
    t, fmu, rho = (_r(v, real_words) for v in (props.t, props.nsm, props.rho))
    q1 = _sub(xyz[0], xyz[2], real_words)
    q2 = _sub(xyz[1], xyz[3], real_words)
    quad = _norm(_cross(q1, q2)) / 2.0
    out = np.zeros(12)
    for piv, (p1, p2, p3) in enumerate(_MASS_TRIANGLES):
        v1 = _sub(xyz[p3], xyz[p1], real_words)
        v2 = _sub(xyz[p3], xyz[p2], real_words)
        area = _norm(_cross(v1, v2)) / 2.0 + quad
        rhod = rho if t != 0.0 else 0.0
        term = (fmu + rhod * t) * area / 3.0 / 2.0
        out[3 * piv : 3 * piv + 3] = term
    return out


def _emg_records(ke: FloatArray, me: FloatArray, order: Sequence[int]):
    idx = [3 * g + c for g in order for c in range(3)]
    kout = ke[np.ix_(idx, idx)]
    stiffness = EmgMatrix(kout.T.reshape(-1).copy(), 12, ICODE_TRANSLATION, 1)
    mass = EmgMatrix(me[idx].copy(), 12, ICODE_TRANSLATION, 2)
    return stiffness, mass


def sheard(
    data: ElementData,
    materials: MaterialTable | ShearProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> ShearMatrices:
    """Per-element reference port of ``SHEARD`` (stiffness, then ``EMADTQ(6)`` mass).

    ``materials`` is a :class:`MaterialTable` (``MAT`` at the element
    temperature) or ready :class:`ShearProperties`. Raises
    :class:`ShearGeometryError` where the Fortran calls ``MESAGE(30,26|27)``.
    """
    props = (
        materials if isinstance(materials, ShearProperties) else shear_properties(data, materials)
    )
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    sils = tuple(int(s) for s in data["sil"])  # type: ignore[union-attr]
    xyz = [[_r(v, real_words) for v in g.xyz] for g in data.grids]
    cids = [int(g.cid) for g in data.grids]
    ke, branch = _stiffness(data.eid, xyz, cids, props, cstm, real_words)
    me = emadtq_shear(xyz, props, real_words)
    order = _sil_order(sils)
    stiffness, mass = _emg_records(ke, me, order)
    return ShearMatrices(
        data.eid,
        tuple(sils[i] for i in order),  # type: ignore[arg-type]
        stiffness,
        mass,
        props.ge,
        branch,
    )


@dataclass(frozen=True)
class ShearBatch:
    """Shear panels as stacked arrays (``n`` elements)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 4)`` SILs in connection order."""
    cid: NDArray[np.int64]
    """``(n, 4)`` displacement coordinate systems of the grids."""
    xyz: FloatArray
    """``(n, 4, 3)`` basic grid coordinates."""
    e: FloatArray
    g: FloatArray
    nu: FloatArray
    rho: FloatArray
    ge: FloatArray
    t: FloatArray
    nsm: FloatArray

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable | Sequence[ShearProperties],
    ) -> ShearBatch:
        if isinstance(materials, MaterialTable):
            props = [shear_properties(d, materials) for d in elements]
        else:
            props = list(materials)
        col = {
            name: np.array([getattr(p, name) for p in props], dtype=float)
            for name in ("e", "g", "nu", "rho", "ge", "t", "nsm")
        }
        return cls(
            eid=np.array([d.eid for d in elements], dtype=np.int64),
            sil=np.array([d["sil"] for d in elements], dtype=np.int64).reshape(-1, 4),
            cid=np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(
                -1, 4
            ),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(
                -1, 4, 3
            ),
            **col,
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


@dataclass(frozen=True)
class ShearBatchMatrices:
    """Batched shear panel matrices; rows/columns of each element in ascending-SIL order."""

    sils: NDArray[np.int64]
    """``(n, 4)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 12, 12)`` stiffness (translations of the four grids)."""
    m: FloatArray
    """``(n, 12, 12)`` diagonal lumped mass."""
    ge: FloatArray
    branch: NDArray[np.int64]
    """Fortran label of the flexibility formula used (30, 40, 50 or 70)."""


def _vdot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _vcross(a, b):
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _raise_first(eid, checks) -> None:
    """Raise for the first element (batch order) failing a check, checks in Fortran order."""
    n = eid.shape[0]
    first = np.full(n, -1)
    for k, (mask, _, _) in enumerate(checks):
        first = np.where((first < 0) & mask, k, first)
    bad = np.flatnonzero(first >= 0)
    if bad.size:
        i = bad[0]
        _, code, corner = checks[first[i]]
        raise ShearGeometryError(int(eid[i]), code, corner)


def shear_matrices(
    batch: ShearBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> ShearBatchMatrices:
    """Vectorised ``SHEARD``: 12x12 stiffness and lumped mass of every panel in ``batch``.

    Raises :class:`ShearGeometryError` for the first element (in batch order)
    the Fortran would reject.
    """
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    n = len(batch)

    def rnd(x):
        return np.asarray(x, dtype=np.float32).astype(float) if real_words else np.asarray(x)

    xyz = rnd(batch.xyz)
    g, t, nu, rho, fmu = (rnd(getattr(batch, f)) for f in ("g", "t", "nu", "rho", "nsm"))
    nuc = 1.0 / (1.0 + nu)
    gp1, gp2, gp3, gp4 = (xyz[:, i] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        vd1, vd2 = rnd(gp3 - gp1), rnd(gp4 - gp2)
        vkn = _vcross(vd1, vd2)
        vkl = rnd(np.sqrt(_vdot(vkn, vkn)))
        vk = vkn / vkl[:, None]
        pa = rnd(vkl / 2.0)
        v12, v41 = rnd(gp2 - gp1), rnd(gp1 - gp4)
        v12dk = rnd(_vdot(v12, vk))
        vp12 = v12 - v12dk[:, None] * vk
        vp12l = rnd(np.sqrt(_vdot(vp12, vp12)))
        vi = vp12 / vp12l[:, None]
        vj = _vcross(vk, vi)
        vjl = rnd(np.sqrt(_vdot(vj, vj)))
        vj = vj / vjl[:, None]
        x2 = vp12l
        x3, y3 = _vdot(vi, vd1), _vdot(vj, vd1)
        x4 = -vi[:, 0] * v41[:, 0] - vi[:, 1] * v41[:, 1] - vi[:, 2] * v41[:, 2]
        y4 = -vj[:, 0] * v41[:, 0] - vj[:, 1] * v41[:, 1] - vj[:, 2] * v41[:, 2]
        _raise_first(
            batch.eid,
            [
                (t * g == 0.0, 2026, None),
                (vkl == 0.0, 2026, None),
                (vp12l == 0.0, 2026, None),
                (vjl == 0.0, 2026, None),
                (y3 <= 0.0, 2027, 2),
                (y4 <= 0.0, 2027, 1),
                (x3 <= y3 * x4 / y4, 2027, 4),
                (x4 >= x2 - (x2 - x3) * y4 / y3, 2027, 3),
            ],
        )
        z, branch = _flexibility_batch(x2, x3, y3, x4, y4, pa, g, t, nuc, rnd)

        xl13 = np.sqrt(x3 * x3 + y3 * y3)
        xl24 = np.sqrt((x4 - x2) * (x4 - x2) + y4 * y4)
        su = np.stack([x3 / xl13, (x4 - x2) / xl24] * 2, axis=1)
        sv = np.stack([y3 / xl13, y4 / xl24] * 2, axis=1)
        temp = x4 * y3 - x3 * y4
        a1 = -0.5 * x2 * y4 * xl13 / temp
        a2 = 0.5 * x2 * y3 * xl24 / (temp - x2 * (y3 - y4))
    avec = np.stack([a1, a2, -a1, -a2], axis=1)
    vec = vi[:, None, :] * su[:, :, None] + vj[:, None, :] * sv[:, :, None]
    ti = transforms_to_basic(
        cstm, batch.cid.reshape(-1), xyz.reshape(-1, 3), real_offset=real_words
    ).reshape(n, 4, 3, 3)
    rot = batch.cid != 0
    tv = ti[..., 0, :] * vec[..., 0, None]
    tv = tv + ti[..., 1, :] * vec[..., 1, None]
    tv = tv + ti[..., 2, :] * vec[..., 2, None]
    vec = np.where(rot[..., None], tv, vec)

    con = avec / (2.0 * z)[:, None]
    blk = vec[:, :, None, :, None] * vec[:, None, :, None, :]
    blk = con[:, :, None, None, None] * blk * avec[:, None, :, None, None]
    ke = blk.transpose(0, 1, 3, 2, 4).reshape(n, 12, 12)

    q = _vcross(rnd(gp1 - gp3), rnd(gp2 - gp4))
    quad = np.sqrt(_vdot(q, q)) / 2.0
    rhod = np.where(t != 0.0, rho, 0.0)
    me = np.empty((n, 4))
    for piv, (p1, p2, p3) in enumerate(_MASS_TRIANGLES):
        c = _vcross(rnd(xyz[:, p3] - xyz[:, p1]), rnd(xyz[:, p3] - xyz[:, p2]))
        area = np.sqrt(_vdot(c, c)) / 2.0 + quad
        me[:, piv] = (fmu + rhod * t) * area / 3.0 / 2.0

    order = np.argsort(batch.sil, axis=1, kind="stable")
    idx = (3 * order[:, :, None] + np.arange(3)).reshape(n, 12)
    rows = np.arange(n)[:, None]
    k = ke[rows[:, :, None], idx[:, :, None], idx[:, None, :]]
    mdiag = np.repeat(me, 3, axis=1)[rows, idx]
    m = np.zeros((n, 12, 12))
    m[:, np.arange(12), np.arange(12)] = mdiag
    sils = np.take_along_axis(batch.sil, order, axis=1)
    return ShearBatchMatrices(sils, k, m, batch.ge.copy(), branch)


def _flexibility_batch(x2, x3, y3, x4, y4, pa, g, t, nuc, rnd):
    """Vectorised statements 30-70: ``Z`` and the branch label per element."""
    cep1 = np.abs(y3 - y4)
    cepx = np.abs(x3 - x4)
    temp = x3 - x2
    cep2 = np.abs(y4 * temp - y3 * x4)
    cepy = np.abs(x4 * temp + y4 * y3)
    par12 = cep1 < EP * cepx
    par14 = cep2 < EP * cepy
    branch = np.select(
        [par12 & par14, par12, par14],
        [BRANCH_PARALLELOGRAM, BRANCH_PARALLEL_12_34, BRANCH_PARALLEL_14_23],
        BRANCH_GENERAL,
    )
    den = 2.0 * g * t

    # 30: edge 3-4 parallel to 1-2.
    temp = y3 * x4 - y4 * (x3 - x2)
    yp = x2 * y3 * y4 / temp
    xp = x2 * y3 * x4 / temp
    sa = (x2 - xp) / yp
    c = (0.0 - xp) / yp
    z30 = ((yp * yp * pa) / ((yp - y3) * (yp - y4) * 2.0 * g * t)) * (
        1.0 + C23 * nuc * (sa * sa + sa * c + c * c)
    )

    # 40: edge 1-4 parallel to 2-3.
    d = -0.5 * (x4 / y4 + (x3 - x2) / y3)
    xq = rnd(x4 - y4 * (x3 - x4) / (y3 - y4))
    temp = 1.0 / np.sqrt(1.0 + d * d)
    p1 = (xq - 0.0 - d * 0.0) * temp
    p2 = (xq - x2 - d * 0.0) * temp
    p3 = (xq - x3 - d * y3) * temp
    p4 = (xq - x4 - d * y4) * temp
    temp = xq - x4
    b = (temp * d + y4) / (temp - y4 * d)
    z40 = ((p1 * p2 * pa) / (p3 * p4 * 2.0 * g * t)) * (1.0 + C23 * nuc * (b * b + b * d + d * d))

    # 50: parallelogram.
    d = -0.5 * (x4 / y4 + (x3 - x2) / y3 + (y3 - y4) / (x3 - x4))
    z50 = pa / den * (1.0 + 2.0 * (d * d) * nuc)

    # 70: general quadrilateral.
    xq = rnd(x4 - (x3 - x4) / (y3 - y4) * y4)
    temp = y3 * x4 - y4 * (x3 - x2)
    xp = x2 * y3 * x4 / temp
    yp = x2 * y3 * y4 / temp
    xl = np.sqrt((xq - xp) * (xq - xp) + yp * yp)
    d = (xq - xp) / yp
    temp = yp / xl
    p1 = temp * (xq - 0.0 - d * 0.0)
    p2 = temp * (xq - x2 - d * 0.0)
    p3 = temp * (xq - x3 - d * y3)
    p4 = temp * (xq - x4 - d * y4)
    c = xl / p1 - d
    b = xl / p4 - c
    a = xl / p2 - d
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d
    a3, b3, c3, d3 = a2 * a, b2 * b, c2 * c, d2 * d
    a4, b4, c4, d4 = a3 * a, b3 * b, c3 * c, d3 * d
    a5, b5, c5, d5 = a4 * a, b4 * b, c4 * c, d4 * d
    temp = 0.5 * p1 * p2 * p3 * p4 / (xl * xl)
    term = (a + b + C23 * (a3 + b3) + 0.2 * (a5 + b5)) * np.log(np.abs(a + b))
    term1 = (c + d + C23 * (c3 + d3) + 0.2 * (c5 + d5)) * np.log(np.abs(c + d))
    term2 = (b + c + C23 * (b3 + c3) + 0.2 * (b5 + c5)) * np.log(np.abs(b + c))
    term3 = (d + a + C23 * (d3 + a3) + 0.2 * (d5 + a5)) * np.log(np.abs(d + a))
    term4 = 0.1 * ((a2 - c2) * (b3 - d3) + (b2 - d2) * (a3 - c3))
    term5 = 0.2 * ((a - c) * (b4 - d4) + (b - d) * (a4 - c4))
    f = temp * (term + term1 - term2 - term3 + term4 - term5)
    z70 = p1 * p2 / (p3 * p4 * 2.0 * g * t) * (pa + 4.0 * nuc * (f - C23 * pa))

    z = np.select(
        [branch == BRANCH_PARALLEL_12_34, branch == BRANCH_PARALLEL_14_23, branch == 50],
        [z30, z40, z50],
        z70,
    )
    return z, branch.astype(np.int64)
