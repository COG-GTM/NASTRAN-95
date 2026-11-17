"""Triangular membrane (CTRMEM) stiffness, mass and thermal load.

Ports ``TRMEMD`` (``mis/trmemd.f``, the double-precision EMG routine; ``TRMEMS``
is its single-precision twin and ``KTRMEM`` the older SMA1 path) with its
stiffness kernel ``EKTRMD`` and the lumped mass of ``EMADTQ(4)``, plus the
thermal load ``TRIMEM`` (``mis/trimem.f``, called from ``EDTL``).

The constant-strain triangle with grids A, B, C has element axes ``i`` along
``B - A``, ``k`` normal to the plane and ``j = k x i``; in those axes B sits at
``(xb, 0)`` and C at ``(xc, yc)``. With the 3x2 strain-displacement block
``C_g`` of grid ``g`` and the in-plane basis ``E = [i j]`` (3x2):

- stiffness: ``K_gh = T_g^T E C_g^T G C_h E^T T_h * VOL``, ``VOL = xb*yc*t/2``,
  ``G`` the MAT2-format plane-stress matrix (MAT1 or anisotropic MAT2 rotated
  by the material angle THETA), ``T_g`` the grid's global-to-basic transform;
- lumped mass: ``(NSM + RHO*t) * AREA / 3`` on each translation;
- thermal load: ``T_g^T E C_g^T G alpha (TBAR - TREF) VOL`` on each grid.

The element-axis stiffness :func:`trmem_element_stiffness` (6x6 on
``u, v`` of the three grids) and :func:`membrane_b_matrix` are exposed for
the quadrilateral membranes built from overlapping triangles (CQDMEM).

Rows and columns of the EMG matrices follow ascending SIL, as in the Fortran;
the thermal load is added at each grid's SIL in connection order, as ``TRIMEM``
does. Two paths are given, as for the rod kernel:

- :func:`trmemd`, a per-element reference that follows ``EKTRMD``/``TRMEMD``
  statement by statement, and :func:`trmem_matrices`, the vectorised kernel
  over a :class:`TrmemBatch`;
- :func:`trmem_thermal_loads`, the vectorised ``TRIMEM`` that repeats its REAL
  operations in order (so it is exact to the bit), with :func:`trimem` its
  single-element form.

With ``real_words=True`` (default) the REAL words and REAL arithmetic of the
Fortran are emulated: inputs are rounded to float32, the material angle and the
``MAT`` (``INFLAG = 2``) plane-stress matrix and its rotation are evaluated in
float32, the CSTM is rounded as ``TRANSD`` reads it, and ``TRIMEM`` (entirely
single precision, including ``BASGLB``) is reproduced operation by operation
in float32. ``real_words=False`` evaluates everything in float64.

The heat-transfer branches of ``TRMEMD``/``EKTRMD`` are not ported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.coords.transforms import CoordType
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import ICODE_TRANSLATION, EmgMatrix, real_cstm
from nastran95py.materials import MaterialTable
from nastran95py.materials.mat import MaterialError

FloatArray = NDArray[np.float64]

f32 = np.float32
_PI32 = f32(4.0) * f32(math.atan(1.0))
DEGRA32 = f32(_PI32 / f32(180.0))
"""``CONSTS(4)``: degrees to radians as the REAL word ``/CONDAS/`` holds."""
_SIN_TOL32 = f32(1.0e-6)


class TrmemGeometryError(ValueError):
    """Degenerate triangle (Fortran messages 30/31, 30/32 and 30/26)."""


# ----------------------------------------------------------------------------
# Geometry and element-axis kernel (shared with CQDMEM)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TriangleGeometry:
    """Element axes of a membrane triangle: ``E = [i j]`` and the in-plane coordinates."""

    e: FloatArray
    """``(3, 2)`` basis: columns are the element ``i`` and ``j`` axes in basic."""
    xsubb: float
    """``x`` of grid B (``|B - A|``)."""
    xsubc: float
    ysubc: float

    @property
    def area(self) -> float:
        return self.xsubb * self.ysubc / 2.0


def _axes(d1, d2):
    """``EKTRMD`` element axes from ``B - A`` and ``C - A`` (float64 arrays ``(..., 3)``)."""
    xsubb = np.sqrt(d1[..., 0] ** 2 + d1[..., 1] ** 2 + d1[..., 2] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ei = d1 / xsubb[..., None]
        xsubc = ei[..., 0] * d2[..., 0] + ei[..., 1] * d2[..., 1] + ei[..., 2] * d2[..., 2]
        ek = np.stack(
            [
                ei[..., 1] * d2[..., 2] - ei[..., 2] * d2[..., 1],
                ei[..., 2] * d2[..., 0] - ei[..., 0] * d2[..., 2],
                ei[..., 0] * d2[..., 1] - ei[..., 1] * d2[..., 0],
            ],
            axis=-1,
        )
        ysubc = np.sqrt(ek[..., 0] ** 2 + ek[..., 1] ** 2 + ek[..., 2] ** 2)
        ek = ek / ysubc[..., None]
        ej = np.stack(
            [
                ei[..., 2] * ek[..., 1] - ei[..., 1] * ek[..., 2],
                ei[..., 0] * ek[..., 2] - ei[..., 2] * ek[..., 0],
                ei[..., 1] * ek[..., 0] - ei[..., 0] * ek[..., 1],
            ],
            axis=-1,
        )
        ej = ej / np.sqrt(ej[..., 0] ** 2 + ej[..., 1] ** 2 + ej[..., 2] ** 2)[..., None]
    return np.stack([ei, ej], axis=-1), xsubb, xsubc, ysubc


def triangle_geometry(xyz, real_words: bool = True, eid: int = 0) -> TriangleGeometry:
    """Element axes of the triangle with basic coordinates ``xyz`` (3x3, rows A, B, C).

    As ``EKTRMD``: the REAL coordinates are differenced in double precision.
    """
    x = np.asarray(xyz, dtype=np.float32 if real_words else float).astype(float)
    e, xsubb, xsubc, ysubc = _axes(x[1] - x[0], x[2] - x[0])
    _check_geometry(eid, float(xsubb), float(ysubc))
    return TriangleGeometry(e, float(xsubb), float(xsubc), float(ysubc))


def _check_geometry(eid, xsubb, ysubc) -> None:
    if not xsubb > 1.0e-6:
        raise TrmemGeometryError(f"triangle {eid}: grids A and B coincide (30/31)")
    if not ysubc > 1.0e-6:
        raise TrmemGeometryError(f"triangle {eid}: grids are collinear (30/32)")


def membrane_b_matrix(xsubb: float, xsubc: float, ysubc: float) -> FloatArray:
    """``EKTRMD`` strain-displacement matrix ``C``: ``(3, 6)``, columns ``u, v`` of A, B, C.

    Rows are ``eps_x, eps_y, gamma_xy`` in the element axes.
    """
    reelmu = 1.0 / xsubb
    flamda = 1.0 / ysubc
    delta = xsubc / xsubb - 1.0
    c10 = -flamda * reelmu * xsubc
    return np.array(
        [
            [-reelmu, 0.0, reelmu, 0.0, 0.0, 0.0],
            [0.0, flamda * delta, 0.0, c10, 0.0, flamda],
            [flamda * delta, -reelmu, c10, reelmu, flamda, 0.0],
        ]
    )


def trmem_element_stiffness(
    xsubb: float, xsubc: float, ysubc: float, g: FloatArray, t: float
) -> FloatArray:
    """Constant-strain membrane stiffness in element axes: ``VOL * C^T G C`` (6x6).

    DOFs are ``u, v`` of grids A, B, C in the element ``i, j`` axes, with A at
    the origin, B at ``(xsubb, 0)`` and C at ``(xsubc, ysubc)``; ``g`` is the
    3x3 plane-stress matrix in those axes and ``t`` the thickness.
    """
    c = membrane_b_matrix(xsubb, xsubc, ysubc)
    vol = xsubb * ysubc * t / 2.0
    return (c.T @ np.asarray(g, dtype=float) @ c) * vol


# ----------------------------------------------------------------------------
# Material (``MAT`` with INFLAG = 2) in REAL arithmetic
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrmemProperties:
    """Material and property values the CTRMEM routines use for one element."""

    g: FloatArray
    """3x3 plane-stress matrix in element axes (rotated by THETA)."""
    alpha: FloatArray
    """Thermal expansion ``(alpha_x, alpha_y, alpha_xy)`` in element axes."""
    tref: float
    rho: float
    ge: float
    t: float
    nsm: float


def material_angle(theta_deg: float, real_words: bool = True) -> tuple[float, float]:
    """``SINTH, COSTH`` of the material angle (``SINTH`` below 1e-6 is zeroed)."""
    if real_words:
        theta = f32(theta_deg) * DEGRA32
        s, c = f32(math.sin(float(theta))), f32(math.cos(float(theta)))
        if abs(s) < _SIN_TOL32:
            s = f32(0.0)
        return float(s), float(c)
    theta = math.radians(theta_deg)
    s, c = math.sin(theta), math.cos(theta)
    return (0.0 if abs(s) < 1.0e-6 else s), c


def _mm32(a, b):
    """``GMMATS`` product: float32, accumulated term by term from zero."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    out = np.zeros(a.shape[:-1] + b.shape[1:], np.float32)
    for k in range(a.shape[-1]):
        out = out + a[..., k, None] * b[k] if b.ndim == 2 else out + a[..., k] * b[k]
    return out


def mat2_real(
    materials: MaterialTable, mid: int, sinth: float, costh: float, temp: float | None = None
) -> tuple[FloatArray, FloatArray, float, float, float]:
    """``MAT`` (``INFLAG = 2``) in REAL arithmetic: ``(G, alpha, TREF, RHO, GE)``."""
    m = materials.mat2(mid, temp=temp)
    if m.source == "MAT1":
        m1 = materials.mat1(mid, temp)
        e, gx, nu, alpha = f32(m1.e), f32(m1.g), f32(m1.nu), f32(m1.alpha)
        g11 = e / (f32(1.0) - nu * nu)
        g12 = nu * g11
        g = np.array([[g11, g12, 0.0], [g12, g11, 0.0], [0.0, 0.0, gx]], np.float32)
        a = np.array([alpha, alpha, 0.0], np.float32)
    else:
        g = m.g.astype(np.float32)
        a = m.alpha.astype(np.float32)
        s, c = f32(sinth), f32(costh)
        if s != 0.0:
            if abs(s * s + c * c - f32(1.0)) > f32(1.0e-4):
                raise MaterialError(f"material {mid}: sin/cos of material angle not normalized")
            x1, x2, x3 = c * c, s * s, c * s
            x7 = f32(2.0) * -x3
            u = np.array([[x1, x2, x3], [x2, x1, -x3], [x7, -x7, x1 - x2]], np.float32)
            g = _mm32(u.T, _mm32(g, u))
            g = np.array(
                [
                    [g[0, 0], g[0, 1], g[0, 2]],
                    [g[0, 1], g[1, 1], g[1, 2]],
                    [g[0, 2], g[1, 2], g[2, 2]],
                ],
                np.float32,
            )
            ua = np.array([[x1, x2, -x3], [x2, x1, x3], [-x7, x7, x1 - x2]], np.float32)
            a = _mm32(ua, a)
    return g.astype(float), a.astype(float), float(f32(m.tref)), float(f32(m.rho)), float(f32(m.ge))


def trmem_properties(
    data: ElementData, materials: MaterialTable, real_words: bool = True
) -> TrmemProperties:
    """``CALL MAT`` at the element temperature and material angle, plus ``t`` and ``NSM``."""
    mid = int(data["mid"])  # type: ignore[arg-type]
    sinth, costh = material_angle(float(data["theta"]), real_words)  # type: ignore[arg-type]
    if real_words:
        g, alpha, tref, rho, ge = mat2_real(materials, mid, sinth, costh, data.temperature)
        rnd = lambda v: float(f32(v))  # noqa: E731
    else:
        m = materials.mat2(mid, sinth, costh, data.temperature)
        g, alpha, tref, rho, ge = m.g, m.alpha, m.tref, m.rho, m.ge
        rnd = float
    return TrmemProperties(
        np.asarray(g, float),
        np.asarray(alpha, float),
        tref,
        rho,
        ge,
        rnd(data["t"]),  # type: ignore[arg-type]
        rnd(data["nsm"]),  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------------
# Reference path: TRMEMD (EKTRMD + EMADTQ)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrmemMatrices:
    """``TRMEMD`` output for one element (translations of the three grids)."""

    eid: int
    sils: tuple[int, int, int]
    """SILs in matrix order (ascending)."""
    stiffness: EmgMatrix
    mass: EmgMatrix
    ge: float

    def k(self) -> FloatArray:
        """9x9 stiffness on the translations, ascending SIL, grid global axes."""
        return self.stiffness.square()

    def m(self) -> FloatArray:
        return self.mass.square()

    def k_grid(self) -> FloatArray:
        """Stiffness expanded to the 18 grid DOFs (zero rotations)."""
        return expand_to_grid_dof(self.k())

    def m_grid(self) -> FloatArray:
        return expand_to_grid_dof(self.m())


def expand_to_grid_dof(mat: FloatArray) -> FloatArray:
    """Translational ``(..., 3n, 3n)`` matrices expanded to six DOFs per grid."""
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[-1] // 3
    idx = np.array([6 * g + c for g in range(n) for c in range(3)])
    out = np.zeros((*mat.shape[:-2], 6 * n, 6 * n))
    out[..., idx[:, None], idx[None, :]] = mat
    return out


def _sil_order(sils) -> list[int]:
    """``TRMEMD`` IPART: grid indices in ascending-SIL order."""
    return sorted(range(3), key=lambda i: sils[i])


def _transd(cstm, cid: int, xyz, real_words: bool) -> FloatArray:
    if cid == 0:
        return np.eye(3)
    return transforms_to_basic(cstm, cid, xyz, real_offset=real_words)[0]


def trmemd(
    data: ElementData,
    materials: MaterialTable | TrmemProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemMatrices:
    """Per-element reference port of ``TRMEMD``: stiffness (``EKTRMD``) and lumped mass.

    ``materials`` is a :class:`MaterialTable` (``MAT`` is evaluated at the
    element temperature and material angle) or ready :class:`TrmemProperties`.
    """
    props = (
        materials
        if isinstance(materials, TrmemProperties)
        else trmem_properties(data, materials, real_words)
    )
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    eid = data.eid
    sils = tuple(int(s) for s in data["sil"])  # type: ignore[union-attr]
    dt = np.float32 if real_words else np.float64
    x = np.array([g.xyz for g in data.grids], dtype=dt)
    geom = triangle_geometry(x, real_words, eid)
    e = geom.e
    c = membrane_b_matrix(geom.xsubb, geom.xsubc, geom.ysubc)
    vol = geom.xsubb * geom.ysubc * props.t / 2.0
    g = props.g
    ti = [
        _transd(cstm, grd.cid, x[i].astype(float), real_words) for i, grd in enumerate(data.grids)
    ]

    kij = np.zeros((3, 3, 3, 3))
    for p in range(3):
        common = (e @ c[:, 2 * p : 2 * p + 2].T) @ g * vol
        if data.grids[p].cid != 0:
            common = ti[p].T @ common
        for i in range(3):
            right = c[:, 2 * i : 2 * i + 2] @ e.T
            if data.grids[i].cid != 0:
                right = right @ ti[i]
            kij[p, i] = common @ right

    order = _sil_order(sils)
    k = np.zeros((9, 9))
    for a, ii in enumerate(order):
        for b, jj in enumerate(order):
            k[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = kij[ii, jj]
    stiffness = EmgMatrix(k.reshape(-1), 9, ICODE_TRANSLATION, 1)

    term = (props.nsm + props.rho * props.t) * _emadtq_area(x) / 3.0
    mass = EmgMatrix(np.full(9, term), 9, ICODE_TRANSLATION, 2)
    return TrmemMatrices(eid, tuple(sils[i] for i in order), stiffness, mass, props.ge)  # type: ignore[arg-type]


def _emadtq_area(x):
    """``EMADTQ`` area: REAL differences ``C - A``, ``C - B``, double cross product."""
    v1 = (x[..., 2, :] - x[..., 0, :]).astype(float)
    v2 = (x[..., 2, :] - x[..., 1, :]).astype(float)
    cx = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
    cy = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
    cz = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return 0.0 + np.sqrt(cx**2 + cy**2 + cz**2) / 2.0


# ----------------------------------------------------------------------------
# Reference path: TRIMEM thermal load
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrmemLoad:
    """``TRIMEM`` thermal load of one element: a 3-vector per grid, connection order."""

    eid: int
    sils: tuple[int, int, int]
    loads: FloatArray
    """``(3, 3)``: row ``g`` is the load on grid ``g`` in its global axes."""

    def add_to(self, pg: FloatArray) -> FloatArray:
        """Add into a load vector indexed by SIL (``pg[sil - 1 + k]``), as ``TRIMEM`` does."""
        for sil, p in zip(self.sils, self.loads, strict=True):
            pg[sil - 1 : sil + 2] += p
        return pg


def _basglb32(cs: CoordSystem, v, pont):
    """``BASGLB`` in REAL arithmetic: basic vectors ``v (n, 3)`` at points to global."""
    v = np.asarray(v, np.float32)
    t = cs.t.astype(np.float32)
    if cs.ctype == CoordType.RECTANGULAR:
        return _mm32(v, t)
    p = np.asarray(pont, np.float32) - cs.origin.astype(np.float32)
    loc = _mm32(p, t)
    x, y = loc[:, 0], loc[:, 1]
    r = np.sqrt(x * x + y * y)
    n = v.shape[0]
    tl = np.zeros((n, 3, 3), np.float32)
    polar = r == 0.0
    rs = np.where(polar, f32(1.0), r)
    if cs.ctype == CoordType.CYLINDRICAL:
        tl[:, 2, 2] = 1.0
        tl[:, 0, 0] = x / rs
        tl[:, 1, 1] = tl[:, 0, 0]
        tl[:, 1, 0] = y / rs
        tl[:, 0, 1] = -tl[:, 1, 0]
    else:
        z = loc[:, 2]
        xl = np.sqrt(x * x + y * y + z * z)
        xls = np.where(polar, f32(1.0), xl)
        xr, yr, zl = x / rs, y / rs, z / xls
        tl[:, 0, 0] = x / xls
        tl[:, 0, 1] = xr * zl
        tl[:, 0, 2] = -yr
        tl[:, 1, 0] = y / xls
        tl[:, 1, 1] = yr * zl
        tl[:, 1, 2] = xr
        tl[:, 2, 0] = zl
        tl[:, 2, 1] = -r / xls
    # TZ(N,L) = sum_m TI(N,m) TL(m,L) with TI = T; VOUT(L) = sum_m v(m) TZ(m,L)
    tz = np.zeros((n, 3, 3), np.float32)
    for m in range(3):
        tz = tz + t[None, :, m, None] * tl[:, m, None, :]
    tz = np.where(polar[:, None, None], t[None], tz)
    out = np.zeros((n, 3), np.float32)
    for m in range(3):
        out = out + v[:, m, None] * tz[:, m, :]
    return out


def _trimem_axes32(x):
    """``TRIMEM`` element axes, ``C`` and ``VOL / t`` in REAL arithmetic (``x``: ``(n, 3, 3)``)."""
    one = f32(1.0)
    e1 = x[:, 1] - x[:, 0]
    xsubb = np.sqrt(e1[:, 0] ** 2 + e1[:, 1] ** 2 + e1[:, 2] ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ei = e1 / xsubb[:, None]
        d2 = x[:, 2] - x[:, 0]
        xsubc = ei[:, 0] * d2[:, 0] + ei[:, 1] * d2[:, 1] + ei[:, 2] * d2[:, 2]
        ek = np.stack(
            [
                ei[:, 1] * d2[:, 2] - ei[:, 2] * d2[:, 1],
                ei[:, 2] * d2[:, 0] - ei[:, 0] * d2[:, 2],
                ei[:, 0] * d2[:, 1] - ei[:, 1] * d2[:, 0],
            ],
            axis=1,
        )
        ysubc = np.sqrt(ek[:, 0] ** 2 + ek[:, 1] ** 2 + ek[:, 2] ** 2)
        ek = ek / ysubc[:, None]
        ej = np.stack(
            [
                ei[:, 2] * ek[:, 1] - ei[:, 1] * ek[:, 2],
                ei[:, 0] * ek[:, 2] - ei[:, 2] * ek[:, 0],
                ei[:, 1] * ek[:, 0] - ei[:, 0] * ek[:, 1],
            ],
            axis=1,
        )
        ej = ej / np.sqrt(ej[:, 0] ** 2 + ej[:, 1] ** 2 + ej[:, 2] ** 2)[:, None]
        reelmu = (1.0 / xsubb.astype(float)).astype(np.float32)
        flamda = (1.0 / ysubc.astype(float)).astype(np.float32)
        delta = xsubc / xsubb - one
    n = x.shape[0]
    c = np.zeros((n, 3, 3, 2), np.float32)  # grid, strain, (u, v)
    c[:, 0, 0, 0] = -reelmu
    c[:, 0, 1, 1] = flamda * delta
    c[:, 0, 2, 0] = c[:, 0, 1, 1]
    c[:, 0, 2, 1] = -reelmu
    c10 = -(flamda * reelmu * xsubc)
    c[:, 1, 0, 0] = reelmu
    c[:, 1, 1, 1] = c10
    c[:, 1, 2, 0] = c10
    c[:, 1, 2, 1] = reelmu
    c[:, 2, 1, 1] = flamda
    c[:, 2, 2, 0] = flamda
    return np.stack([ei, ej], axis=-1), c, xsubb, ysubc


def trimem(
    data: ElementData,
    materials: MaterialTable | TrmemProperties,
    tbar: float,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemLoad:
    """``TRIMEM`` for one element: thermal load at mean temperature ``tbar``."""
    props = [materials] if isinstance(materials, TrmemProperties) else materials
    batch = TrmemBatch.from_elements([data], props, real_words)
    return trmem_thermal_loads(batch, tbar, cstm, real_words)[0]


# ----------------------------------------------------------------------------
# Batched path
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrmemBatch:
    """CTRMEM elements as stacked arrays (``n`` elements)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 3)`` SILs in connection order."""
    cid: NDArray[np.int64]
    xyz: FloatArray
    """``(n, 3, 3)`` basic grid coordinates."""
    g: FloatArray
    """``(n, 3, 3)`` plane-stress matrices in element axes."""
    alpha: FloatArray
    tref: FloatArray
    rho: FloatArray
    ge: FloatArray
    t: FloatArray
    nsm: FloatArray

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable | Sequence[TrmemProperties],
        real_words: bool = True,
    ) -> TrmemBatch:
        if isinstance(materials, MaterialTable):
            props = [trmem_properties(d, materials, real_words) for d in elements]
        else:
            props = list(materials)
        col = {
            name: np.array([getattr(p, name) for p in props], dtype=float)
            for name in ("g", "alpha", "tref", "rho", "ge", "t", "nsm")
        }
        return cls(
            eid=np.array([d.eid for d in elements], dtype=np.int64),
            sil=np.array([d["sil"] for d in elements], dtype=np.int64).reshape(-1, 3),
            cid=np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(
                -1, 3
            ),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(
                -1, 3, 3
            ),
            **col,
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


@dataclass(frozen=True)
class TrmemBatchMatrices:
    """Batched CTRMEM matrices; rows/columns of each element in ascending-SIL order."""

    sils: NDArray[np.int64]
    """``(n, 3)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 9, 9)`` stiffness on the grid translations in global axes."""
    m: FloatArray
    """``(n, 9, 9)`` lumped mass."""
    ge: FloatArray


def _batch_geometry(batch: TrmemBatch, real_words: bool):
    dt = np.float32 if real_words else np.float64
    x = batch.xyz.astype(dt)
    xd = x.astype(float)
    e, xsubb, xsubc, ysubc = _axes(xd[:, 1] - xd[:, 0], xd[:, 2] - xd[:, 0])
    bad = ~(xsubb > 1.0e-6) | ~(ysubc > 1.0e-6)
    if bad.any():
        raise TrmemGeometryError(
            f"triangles {batch.eid[bad].tolist()} are degenerate (30/31, 30/32)"
        )
    return x, e, xsubb, xsubc, ysubc


def trmem_matrices(
    batch: TrmemBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemBatchMatrices:
    """Vectorised ``TRMEMD``: 9x9 stiffness and lumped mass of every CTRMEM in ``batch``."""
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    n = len(batch)
    x, e, xsubb, xsubc, ysubc = _batch_geometry(batch, real_words)
    reelmu, flamda = 1.0 / xsubb, 1.0 / ysubc
    delta = xsubc / xsubb - 1.0
    c10 = -flamda * reelmu * xsubc
    c = np.zeros((n, 3, 3, 2))
    c[:, 0, 0, 0] = -reelmu
    c[:, 0, 1, 1] = c[:, 0, 2, 0] = flamda * delta
    c[:, 0, 2, 1] = -reelmu
    c[:, 1, 0, 0] = c[:, 1, 2, 1] = reelmu
    c[:, 1, 1, 1] = c[:, 1, 2, 0] = c10
    c[:, 2, 1, 1] = c[:, 2, 2, 0] = flamda
    vol = xsubb * ysubc * batch.t / 2.0
    ta = transforms_to_basic(
        cstm, batch.cid.reshape(-1), x.reshape(-1, 3).astype(float), real_offset=real_words
    ).reshape(n, 3, 3, 3)
    # strain from the global translations of each grid: C_g E^T T_g  (n, grid, 3, 3)
    d = np.einsum("ngsa,nxa,ngxy->ngsy", c, e, ta)
    kij = np.einsum("nasx,nst,nbty->nabxy", d, batch.g, d) * vol[:, None, None, None, None]
    order = np.argsort(batch.sil, axis=1, kind="stable")
    rows = np.arange(n)[:, None, None]
    kij = kij[rows, order[:, :, None], order[:, None, :]]
    k = kij.transpose(0, 1, 3, 2, 4).reshape(n, 9, 9)
    term = (batch.nsm + batch.rho * batch.t) * _emadtq_area(x) / 3.0
    m = term[:, None, None] * np.eye(9)
    sils = np.take_along_axis(batch.sil, order, axis=1)
    return TrmemBatchMatrices(sils, k, m, batch.ge.copy())


def trmem_thermal_loads(
    batch: TrmemBatch,
    tbar: float | Sequence[float],
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> list[TrmemLoad]:
    """Vectorised ``TRIMEM``: thermal loads of every CTRMEM in ``batch`` at ``tbar``."""
    cstm = {} if cstm is None else cstm
    n = len(batch)
    tb = np.broadcast_to(np.asarray(tbar, dtype=float), (n,))
    if real_words:
        cstm = real_cstm(cstm)
        x = batch.xyz.astype(np.float32)
        e, c, xsubb, ysubc = _trimem_axes32(x)
        bad = ~(xsubb > f32(1.0e-6)) | ~(ysubc > f32(1.0e-6))
        if bad.any():
            raise TrmemGeometryError(
                f"triangles {batch.eid[bad].tolist()} are degenerate (30/31, 30/32)"
            )
        t = batch.t.astype(np.float32)
        vol = xsubb * ysubc * t / f32(2.0)
        temp = (tb.astype(np.float32) - batch.tref.astype(np.float32)) * vol
        tempar = batch.alpha.astype(np.float32) * temp[:, None]
        g = batch.g.astype(np.float32)
        s = np.zeros((n, 3), np.float32)
        for m in range(3):
            s = s + tempar[:, m, None] * g[:, m, :]
        loads = np.zeros((n, 3, 3), np.float32)
        for gi in range(3):
            uv = np.zeros((n, 2), np.float32)
            for m in range(3):
                uv = uv + c[:, gi, m, :] * s[:, m, None]
            p = np.zeros((n, 3), np.float32)
            for m in range(2):
                p = p + uv[:, m, None] * e[:, :, m]
            loads[:, gi] = p
        for gi in range(3):
            for cid in np.unique(batch.cid[:, gi]):
                if cid == 0:
                    continue
                sel = batch.cid[:, gi] == cid
                loads[sel, gi] = _basglb32(cstm[int(cid)], loads[sel, gi], x[sel, gi])
        loads = loads.astype(float)
    else:
        x, e, xsubb, xsubc, ysubc = _batch_geometry(batch, False)
        loads = np.zeros((n, 3, 3))
        for i in range(n):
            c = membrane_b_matrix(xsubb[i], xsubc[i], ysubc[i])
            vol = xsubb[i] * ysubc[i] * batch.t[i] / 2.0
            sig = batch.g[i] @ batch.alpha[i] * ((tb[i] - batch.tref[i]) * vol)
            uv = (c.T @ sig).reshape(3, 2)
            for gi in range(3):
                p = e[i] @ uv[gi]
                if batch.cid[i, gi] != 0:
                    ta = transforms_to_basic(cstm, batch.cid[i, gi], x[i, gi])[0]
                    p = ta.T @ p
                loads[i, gi] = p
    return [
        TrmemLoad(int(batch.eid[i]), tuple(int(s) for s in batch.sil[i]), loads[i])  # type: ignore[arg-type]
        for i in range(n)
    ]
