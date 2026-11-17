"""Isoparametric quadrilateral membrane CQDMEM1: stiffness, mass, thermal load and stresses.

Ports the four Fortran routines of the CQDMEM1 element (property PQDMEM1):

- ``QDMM1D`` (``mis/qdmm1d.f``, the double-precision EMG routine; ``QDMM1S``
  is its single-precision twin): the stiffness of the bilinear isoparametric
  quadrilateral in its mean plane, integrated with 2x2 Gauss points for the
  direct strains and one point for the shear, transformed to the actual grids
  through the warping matrix ``B`` and to each grid's global axes; and the
  lumped mass ``(NSM + RHO t)/12`` times the sum of three of the four corner
  triangle areas (twice) on the translations of each grid;
- ``QDMM1`` (``mis/qdmm1.f``, called from ``EDTL``): the thermal load
  ``E B C G alpha (TBAR - TREF)`` with the integrated strain-displacement
  matrix ``C`` (exact for a parallelogram), rotated to global by ``BASGLB``;
- ``SQDM11``/``SQDM12`` (SDR2 phases 1 and 2): the stress matrices
  ``S_g = G A B^T E^T T_g`` evaluated at the intersection of the diagonals
  ``(ETAS, XIS)``, then ``sigma = sum S_g u_g - G alpha (T - TREF)`` with the
  principal stresses, angle and maximum shear.

The mean-plane projection shared with CQDMEM2 lives in
:mod:`nastran95py.elements.quad_membrane`.

With ``real_words=True`` (default) the REAL words and REAL arithmetic of the
Fortran are emulated: grid coordinates and properties are float32, coordinate
differences are taken in float32 even in ``QDMM1D``, ``MAT`` (``INFLAG = 2``)
is evaluated in float32, the CSTM is rounded as ``TRANSD``/``TRANSS`` read it,
and ``QDMM1``, ``SQDM11`` and ``SQDM12`` (entirely single precision, with
``GMMATS`` accumulating term by term, ``SMMATS`` in double and ``BASGLB``/
``TRANSS`` in float32) are repeated operation by operation in float32.
``real_words=False`` evaluates everything in float64.

Rows and columns of the EMG matrices follow ascending SIL, as in the Fortran;
the thermal load and the stresses use the grids in connection order.

Not ported: the heat-transfer branch (``QDMM1D`` ignores CQDMEM1 in heat
formulations) and the old EMGOLD path (``QDMM1X``, ``KTRIQD``). ``QDMM1D``
calls ``MAT`` for the mass density before it stores the element temperature
in ``/MATIN/``, so a temperature-dependent density is read at the previous
element's temperature there; this port uses the element temperature.
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
from nastran95py.elements.quad_membrane import (
    MeanPlane,
    QuadGeometryError,
    mean_plane,
    mean_plane_b,
)
from nastran95py.elements.rod import ICODE_TRANSLATION, EmgMatrix, real_cstm
from nastran95py.elements.trmem import (
    TrmemProperties,
    _basglb32,
    expand_to_grid_dof,
    mat2_real,
    trmem_properties,
)
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]
Materials = MaterialTable | Sequence[TrmemProperties]

ETA01 = (0.211324865, 0.788675135)
"""Gauss points on ``[0, 1]`` as ``QDMM1D`` stores them."""
_F32 = np.float32
_PI32 = _F32(_F32(4.0) * _F32(math.atan(1.0)))
S4PISQ32 = _F32(_F32(4.0) * _F32(_PI32 * _PI32))
"""``CONSTS(5)`` (4 pi^2) as REAL: the factor ``QDMM1D``/``QDMM1S`` multiply THETA by."""
PHI_SCALE = 28.6478898
"""Half of 180/pi as ``SQDM12`` writes it: ``PHI = ATAN2(2 TXY, SX - SY) * PHI_SCALE``."""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _mm(a, b):
    """``GMMATS``/``GMMATD`` product ``a @ b``, accumulated term by term from zero."""
    lead = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    out = np.zeros((*lead, a.shape[-2], b.shape[-1]), a.dtype)
    for k in range(a.shape[-1]):
        out = out + a[..., :, k, None] * b[..., None, k, :]
    return out


def _mv(a, v):
    """``a @ v`` for ``a (..., r, c)``, ``v (..., c)``, accumulated term by term."""
    out = np.zeros(a.shape[:-1], a.dtype)
    for k in range(a.shape[-1]):
        out = out + a[..., k] * v[..., k, None]
    return out


def _block_e(e):
    """The ``(…, 12, 12)`` block-diagonal ``EE`` of four copies of ``E``."""
    ee = np.zeros(e.shape[:-2] + (12, 12), e.dtype)
    for g in range(4):
        ee[..., 3 * g : 3 * g + 3, 3 * g : 3 * g + 3] = e
    return ee


def qdmm1d_material_angle(theta_deg: float, real_words: bool = True) -> tuple[float, float]:
    """``SINTH, COSTH`` as ``QDMM1D``/``QDMM1S`` compute them.

    Both declare ``COMMON /CONDAS/ CONSTS(4), DEGRA``, so their ``DEGRA`` is the
    fifth word of ``/CONDAS/`` (``4 pi^2``), not degrees-to-radians: the material
    angle of the EMG stiffness is ``THETA * 4 pi^2`` radians. ``QDMM1`` and
    ``SQDM11`` use the correct ``CONSTS(4)``. Only anisotropic (MAT2) materials
    with a non-zero THETA are affected.
    """
    if real_words:
        theta = _F32(theta_deg) * S4PISQ32
        s, c = _F32(math.sin(float(theta))), _F32(math.cos(float(theta)))
        return (0.0 if abs(s) < _F32(1.0e-6) else float(s)), float(c)
    theta = theta_deg * 4.0 * math.pi**2
    s, c = math.sin(theta), math.cos(theta)
    return (0.0 if abs(s) < 1.0e-6 else s), c


def qdmm1d_properties(
    data: ElementData, materials: MaterialTable, real_words: bool = True
) -> TrmemProperties:
    """``CALL MAT`` as ``QDMM1D`` makes it (material angle from :func:`qdmm1d_material_angle`)."""
    mid = int(data["mid"])  # type: ignore[arg-type]
    sinth, costh = qdmm1d_material_angle(float(data["theta"]), real_words)  # type: ignore[arg-type]
    if real_words:
        g, alpha, tref, rho, ge = mat2_real(materials, mid, sinth, costh, data.temperature)
        rnd = lambda v: float(_F32(v))  # noqa: E731
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


def _properties(
    elements: Sequence[ElementData], materials: Materials, real_words: bool, emg: bool = False
):
    if isinstance(materials, MaterialTable):
        prop = qdmm1d_properties if emg else trmem_properties
        return [prop(d, materials, real_words) for d in elements]
    props = list(materials)
    if len(props) != len(elements):
        raise ValueError(f"{len(props)} property sets for {len(elements)} elements")
    return props


def _columns(elements: Sequence[ElementData]):
    n = len(elements)
    eid = np.array([d.eid for d in elements], dtype=np.int64)
    sil = np.array([d["sil"] for d in elements], dtype=np.int64).reshape(n, 4)
    cid = np.array([[g.cid for g in d.grids] for d in elements], dtype=np.int64).reshape(n, 4)
    xyz = np.array([[g.xyz for g in d.grids] for d in elements], dtype=float).reshape(n, 4, 3)
    return eid, sil, cid, xyz


def _fatal(eids, bad, what: str) -> None:
    if np.any(bad):
        raise QuadGeometryError(f"CQDMEM1 {np.asarray(eids)[bad].tolist()}: {what}")


def _check_projection(mp: MeanPlane, eids) -> None:
    _fatal(eids, ~(mp.magk > 1.0e-6), "diagonals are parallel (30/32)")
    _fatal(eids, ~(mp.magi > 1.0e-6), "grids A and B coincide (30/31)")


def _transs32(cs: CoordSystem, xyz) -> np.ndarray:
    """``TRANSS`` in REAL arithmetic: global-to-basic ``TA`` at basic point ``xyz``."""
    f32 = np.float32
    ke = cs.t.astype(f32)
    if cs.ctype == CoordType.RECTANGULAR:
        return ke
    xn = np.asarray(xyz, f32) - cs.origin.astype(f32)
    x, y, z = (ke[0, c] * xn[0] + ke[1, c] * xn[1] + ke[2, c] * xn[2] for c in range(3))
    r = np.sqrt(x * x + y * y)
    if r == f32(0.0):
        return ke
    if cs.ctype == CoordType.CYLINDRICAL:
        tl = np.array([[x / r, -y / r, 0.0], [y / r, x / r, 0.0], [0.0, 0.0, 1.0]], f32)
    else:
        xl = np.sqrt(x * x + y * y + z * z)
        tl = np.array(
            [
                [x / xl, (x * z) / (r * xl), -y / r],
                [y / xl, (y * z) / (r * xl), x / r],
                [z / xl, -r / xl, 0.0],
            ],
            f32,
        )
    return _mm(ke, tl)


def _transs(cstm, cid: int, xyz, real_words: bool) -> np.ndarray:
    if cid == 0:
        return np.eye(3, dtype=np.float32 if real_words else np.float64)
    if real_words:
        return _transs32(cstm[cid], xyz)
    return transforms_to_basic(cstm, cid, np.asarray(xyz, float))[0]


# ----------------------------------------------------------------------------
# QDMM1D: stiffness and lumped mass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem1Matrices:
    """``QDMM1D`` output for one element (translations of the four grids)."""

    eid: int
    sils: tuple[int, int, int, int]
    """SILs in matrix order (ascending)."""
    stiffness: EmgMatrix
    mass: EmgMatrix
    ge: float
    planar: bool

    def k(self) -> FloatArray:
        """12x12 stiffness on the translations, ascending SIL, grid global axes."""
        return self.stiffness.square()

    def m(self) -> FloatArray:
        return self.mass.square()

    def k_grid(self) -> FloatArray:
        """Stiffness expanded to the 24 grid DOFs (zero rotations)."""
        return expand_to_grid_dof(self.k())

    def m_grid(self) -> FloatArray:
        return expand_to_grid_dof(self.m())


@dataclass(frozen=True)
class Qdmem1BatchMatrices:
    """Batched CQDMEM1 matrices; rows/columns of each element in ascending-SIL order."""

    eid: NDArray[np.int64]
    sils: NDArray[np.int64]
    """``(n, 4)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 12, 12)`` stiffness on the grid translations in global axes."""
    m: FloatArray
    """``(n, 12, 12)`` lumped mass."""
    ge: FloatArray
    planar: NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.eid.shape[0])

    def element(self, i: int) -> Qdmem1Matrices:
        return Qdmem1Matrices(
            int(self.eid[i]),
            tuple(int(s) for s in self.sils[i]),  # type: ignore[arg-type]
            EmgMatrix(self.k[i].reshape(-1).copy(), 12, ICODE_TRANSLATION, 1),
            EmgMatrix(np.diag(self.m[i]).copy(), 12, ICODE_TRANSLATION, 2),
            float(self.ge[i]),
            bool(self.planar[i]),
        )


def _check_qdmm1d(mp: MeanPlane, eids) -> None:
    _check_projection(mp, eids)
    _fatal(eids, ~(mp.magj > 1.0e-6), "degenerate element axes (30/26)")
    with np.errstate(divide="ignore", invalid="ignore"):
        _fatal(eids, ~(mp.y3a > 0.0) | ~(mp.y4a > 0.0), "grids out of order (30/26)")
        temp = mp.x12 + mp.x23 * (mp.y4a / mp.y3a)
        ysub4 = (mp.y3a / mp.y4a) * mp.x14
        _fatal(
            eids, ~(mp.x13 < ysub4) | ~(mp.x14 > temp), "interior angle above 180 degrees (30/26)"
        )
    _fatal(eids, np.any([side == 0.0 for side in mp.lengths], axis=0), "zero-length side (30/26)")


def _mean_plane_stiffness(mp: MeanPlane, g, t):
    """``QDMM1D`` 8x8 mean-plane stiffness ``U`` (rows/columns ``u, v`` of A-D)."""
    n = t.shape[0]
    zero = np.zeros(n)
    y4a, x24, x14, x12, y3a = mp.y4a, mp.x24, mp.x14, mp.x12, mp.y3a
    x23, x34, y34, x13 = mp.x23, mp.x34, mp.y34, mp.x13
    cc = (g[:, 0, 0], g[:, 0, 1], g[:, 1, 1], g[:, 0, 2], g[:, 1, 2], zero)
    fact = g[:, 2, 2] * t / (x24 * y3a - x13 * y4a) * 2.0
    aq = {1: -y4a, 3: -x24, 5: -x24, 6: -y4a, 7: y4a, 9: x14, 11: x14, 12: y4a}
    aq.update({13: zero, 15: zero, 17: zero, 18: zero, 19: zero, 21: -x12, 23: -x12, 24: zero})
    bq = {1: y3a, 3: x23, 5: x23, 6: y3a, 7: -y4a, 9: -x14, 11: -x14, 12: -y4a}
    bq.update({13: y4a, 15: x14, 17: x14, 18: y4a, 19: -y3a, 21: -x23, 23: -x23, 24: -y3a})
    cq = {1: -y34, 3: x34, 5: x34, 6: -y34, 7: y34, 9: -x34, 11: -x34, 12: y34}
    cq.update({13: zero, 15: -x12, 17: -x12, 18: zero, 19: zero, 21: x12, 23: x12, 24: zero})

    def o(d, v, f, h, p, q, eta, tea):
        num = d + (v * tea) + (f * eta) + (h * tea * eta) + (p * tea * tea) + (q * eta * eta)
        den = (-y4a * x12) + (-y34 * x12 * eta) + ((-y4a * x23) + (y3a * x14)) * tea
        return num / den

    u = np.empty((n, 8, 8))
    for i in range(4):
        for k in range(2):
            k1, k2 = 6 * i + 4 * k + 1, 6 * i + 3 * k + 3
            for j in range(4):
                for el in range(2):
                    l1, l2 = 6 * j + 4 * el + 1, 6 * j + 3 * el + 3
                    ckl, ck3, cl3 = cc[k + el], cc[k + 3], cc[el + 3]

                    def coef(p, r, _k1=k1, _k2=k2, _l1=l1, _l2=l2, _a=ckl, _b=ck3, _c=cl3):
                        return (
                            _a * ((p[_k1] * r[_l1]) + (r[_k1] * p[_l1]))
                            + _b * ((p[_k1] * r[_l2]) + (r[_k1] * p[_l2]))
                            + _c * ((p[_k2] * r[_l1]) + (r[_k2] * p[_l1]))
                        )

                    d = ckl * aq[k1] * aq[l1] + ck3 * aq[k1] * aq[l2] + cl3 * aq[k2] * aq[l1]
                    v = coef(aq, bq)
                    f = coef(aq, cq)
                    h = coef(bq, cq)
                    p = ckl * bq[k1] * bq[l1] + ck3 * bq[k1] * bq[l2] + cl3 * bq[k2] * bq[l1]
                    q = ckl * cq[k1] * cq[l1] + ck3 * cq[k1] * cq[l2] + cl3 * cq[k2] * cq[l1]
                    s = zero
                    for eta in ETA01:
                        for tea in ETA01:
                            s = s + o(d, v, f, h, p, q, eta, tea)
                    s = s / 4.0 * t
                    s = s + fact * (aq[k2] + 0.5 * (bq[k2] + cq[k2])) * (
                        aq[l2] + 0.5 * (bq[l2] + cq[l2])
                    )
                    u[:, 2 * i + k, 2 * j + el] = s
    return u


def qdmem1_matrices(
    elements: Sequence[ElementData],
    materials: Materials,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> Qdmem1BatchMatrices:
    """Vectorised ``QDMM1D``: 12x12 stiffness and lumped mass of every CQDMEM1 in ``elements``.

    ``materials`` is a :class:`MaterialTable` (``MAT`` is evaluated at each
    element's temperature and at the material angle of
    :func:`qdmm1d_material_angle`) or one :class:`TrmemProperties` per element.
    """
    elements = list(elements)
    props = _properties(elements, materials, real_words, emg=True)
    n = len(elements)
    if n == 0:
        z = np.zeros((0, 12, 12))
        return Qdmem1BatchMatrices(
            np.zeros(0, np.int64), np.zeros((0, 4), np.int64), z, z.copy(), np.zeros(0),
            np.zeros(0, bool),
        )  # fmt: skip
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    eid, sil, cid, xyz = _columns(elements)
    if real_words:
        xyz = xyz.astype(np.float32).astype(float)
    mp = mean_plane(xyz, np.float64, real_words)
    _check_qdmm1d(mp, eid)
    g = np.array([p.g for p in props], dtype=float).reshape(n, 3, 3)
    t = np.array([p.t for p in props], dtype=float)

    h1 = (2.0 * mp.hh) / (mp.diag13 + mp.diag24)
    planar = ~(h1 > 1.0e-6)
    u = _mean_plane_stiffness(mp, g, t)
    uv = np.array([0, 1, 3, 4, 6, 7, 9, 10])
    tempar = np.zeros((n, 12, 12))
    tempar[:, uv[:, None], uv[None, :]] = u
    if not planar.all():
        b = mean_plane_b(mp)
        warped = np.einsum("nia,nab,njb->nij", b, u, b)
        tempar = np.where(planar[:, None, None], tempar, warped)

    ti = transforms_to_basic(
        cstm, cid.reshape(-1), xyz.reshape(-1, 3), real_offset=real_words
    ).reshape(n, 4, 3, 3)
    e = mp.e
    tie = np.einsum("ngba,nbc->ngac", ti, e)
    etj = np.einsum("nba,ngbc->ngac", e, ti)
    blocks = tempar.reshape(n, 4, 3, 4, 3).transpose(0, 1, 3, 2, 4)
    kij = np.einsum("npab,npqbc,nqcd->npqad", tie, blocks, etj)
    order = np.argsort(sil, axis=1, kind="stable")
    rows = np.arange(n)[:, None, None]
    kij = kij[rows, order[:, :, None], order[:, None, :]]
    k = kij.transpose(0, 1, 3, 2, 4).reshape(n, 12, 12)

    rho = np.array([p.rho for p in props], dtype=float)
    nsm = np.array([p.nsm for p in props], dtype=float)
    if real_words:
        lump = (nsm.astype(np.float32) + rho.astype(np.float32) * t.astype(np.float32)).astype(
            float
        )
    else:
        lump = nsm + rho * t
    fact = lump / 12.0
    at1 = -mp.x12 * mp.y4a
    at2 = -mp.x12 * mp.y3a
    at3 = -mp.x23 * mp.y4a + mp.x24 * mp.y3a
    at4 = -mp.x13 * mp.y4a + mp.x14 * mp.y3a
    mgg = np.stack(
        [
            (at4 + at1 + at2) * fact,
            (at1 + at2 + at3) * fact,
            (at2 + at3 + at4) * fact,
            (at3 + at4 + at1) * fact,
        ],
        axis=1,
    )
    mgg = np.take_along_axis(mgg, order, axis=1)
    m = np.zeros((n, 12, 12))
    diag = np.repeat(mgg, 3, axis=1)
    m[:, np.arange(12), np.arange(12)] = diag
    ge = np.array([p.ge for p in props], dtype=float)
    return Qdmem1BatchMatrices(eid, np.take_along_axis(sil, order, axis=1), k, m, ge, planar)


def qdmm1d(
    data: ElementData,
    materials: MaterialTable | TrmemProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> Qdmem1Matrices:
    """``QDMM1D`` for one element: stiffness and lumped mass."""
    props = [materials] if isinstance(materials, TrmemProperties) else materials
    return qdmem1_matrices([data], props, cstm, real_words).element(0)


# ----------------------------------------------------------------------------
# QDMM1: thermal load
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem1Load:
    """``QDMM1`` thermal load of one element: a 3-vector per grid, connection order."""

    eid: int
    sils: tuple[int, int, int, int]
    loads: FloatArray
    """``(4, 3)``: row ``g`` is the load on grid ``g`` in its global axes."""

    def add_to(self, pg: FloatArray) -> FloatArray:
        """Add into a load vector indexed by SIL (``pg[sil - 1 + k]``), as ``QDMM1`` does."""
        for sil, p in zip(self.sils, self.loads, strict=True):
            pg[sil - 1 : sil + 2] += p
        return pg


def qdmem1_thermal_loads(
    elements: Sequence[ElementData],
    materials: Materials,
    tbar: float | Sequence[float],
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> list[Qdmem1Load]:
    """Vectorised ``QDMM1``: thermal loads of every CQDMEM1 in ``elements`` at ``tbar``."""
    elements = list(elements)
    props = _properties(elements, materials, real_words)
    n = len(elements)
    if n == 0:
        return []
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    dt = np.float32 if real_words else np.float64
    eid, sil, cid, xyz = _columns(elements)
    x = xyz.astype(dt)
    mp = mean_plane(x, dt, real_words, x14_from_x41=True)
    _check_projection(mp, eid)
    b = mean_plane_b(mp)
    h = np.array([p.t for p in props]).astype(dt)
    two = dt(2.0)
    c = np.zeros((n, 8, 3), dt)
    for row, sign, col_a, col_b, va, vb in (
        (0, -1, 0, 2, mp.y4a, mp.x24),
        (1, -1, 1, 2, mp.x24, mp.y4a),
        (2, 1, 0, 2, mp.y3a, mp.x13),
        (3, 1, 1, 2, mp.x13, mp.y3a),
        (4, 1, 0, 2, mp.y4a, mp.x24),
        (5, 1, 1, 2, mp.x24, mp.y4a),
        (6, -1, 0, 2, mp.y3a, mp.x13),
        (7, -1, 1, 2, mp.x13, mp.y3a),
    ):
        pa, pb = h * va, h * vb
        c[:, row, col_a] = (-pa if sign < 0 else pa) / two
        c[:, row, col_b] = (-pb if sign < 0 else pb) / two
    g = np.array([p.g for p in props]).reshape(n, 3, 3).astype(dt)
    alpha = np.array([p.alpha for p in props]).reshape(n, 3).astype(dt)
    tref = np.array([p.tref for p in props]).astype(dt)
    temp = np.broadcast_to(np.asarray(tbar, dtype=float), (n,)).astype(dt) - tref
    tempar = alpha * temp[:, None]
    p12 = _mv(_block_e(mp.e), _mv(b, _mv(c, _mv(g, tempar))))
    loads = p12.reshape(n, 4, 3)
    for gi in range(4):
        for sys_id in np.unique(cid[:, gi]):
            if sys_id == 0:
                continue
            sel = cid[:, gi] == sys_id
            cs = cstm[int(sys_id)]
            if real_words:
                loads[sel, gi] = _basglb32(cs, loads[sel, gi], x[sel, gi])
            else:
                ta = transforms_to_basic(cstm, np.full(sel.sum(), sys_id), x[sel, gi])
                loads[sel, gi] = np.einsum("mab,ma->mb", ta, loads[sel, gi])
    loads = loads.astype(float)
    return [
        Qdmem1Load(int(eid[i]), tuple(int(s) for s in sil[i]), loads[i])  # type: ignore[arg-type]
        for i in range(n)
    ]


def qdmm1(
    data: ElementData,
    materials: MaterialTable | TrmemProperties,
    tbar: float,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> Qdmem1Load:
    """``QDMM1`` for one element: thermal load at mean temperature ``tbar``."""
    props = [materials] if isinstance(materials, TrmemProperties) else materials
    return qdmem1_thermal_loads([data], props, tbar, cstm, real_words)[0]


# ----------------------------------------------------------------------------
# SQDM11 / SQDM12: stress recovery
# ----------------------------------------------------------------------------


def diagonal_intersection(mp: MeanPlane, i: int = 0) -> tuple[float, float]:
    """``SQDM11`` ``(ETAS, XIS)``: isoparametric coordinates where the diagonals cross.

    Evaluated in the precision of ``mp`` with the 32-bit-word tolerances
    (``NBPW < 60``) of the Fortran, branch for branch.
    """
    dt = mp.x12.dtype.type
    x12, x13, x14 = mp.x12[i], mp.x13[i], mp.x14[i]
    x23, x24, x34 = mp.x23[i], mp.x24[i], mp.x34[i]
    y3a, y4a, y34 = mp.y3a[i], mp.y4a[i], mp.y34[i]
    zero, half, one, two, four = dt(0.0), dt(0.5), dt(1.0), dt(2.0), dt(4.0)
    tol = dt(1.0e-3) * (-x12)
    tol2 = dt(1.0e-3) * x12 * x12
    if not (abs(x34 + x12) > tol or abs(y34) > tol):
        return float(half), float(half)
    if abs(x24) < tol or abs(x13) < tol:
        if abs(x13) > tol:
            xstar = -x12
            ystar = (y3a / x13) * x12
        else:
            xstar = -x13
            ystar = (-y4a / x24) * x12
    else:
        xstar = (y4a * x13 * x12) / ((y3a * x24) - (y4a * x13))
        ystar = (-y4a / x24) * (xstar + x12)
    with np.errstate(divide="ignore", invalid="ignore"):
        if not abs(x34 + x12) < tol:
            c1 = y34 * xstar - ystar * (x34 + x12)
            a2 = -y4a * x23 + y3a * x14
            b2 = -y4a * x12 + c1
            if abs(a2) <= tol2:
                etas = (-x12 * ystar) / b2
            else:
                temp2 = np.sqrt(b2 * b2 - (four * a2 * x12 * ystar)) / (two * a2)
                temp1 = -b2 / (two * a2)
                etas = temp1 - temp2
                if etas <= zero or etas >= one:
                    etas = temp1 + temp2
            if not abs(y34) < tol:
                xis = (-c1 + ((y4a * x23) - (y3a * x14)) * etas) / (y34 * x12)
            else:
                xis = (xstar + (x14 * etas)) / ((etas * (x34 + x12)) - x12)
        else:
            a3 = -x14 * y34
            b3 = x12 * y4a - y34 * xstar
            if abs(a3) <= tol2:
                etas = (x12 * ystar) / b3
            else:
                temp2 = np.sqrt(b3 * b3 + (four * a3 * x12 * ystar)) / (two * a3)
                temp1 = -b3 / (two * a3)
                etas = temp1 - temp2
                if etas <= zero or etas >= one:
                    etas = temp1 + temp2
            xis = (ystar - (y4a * etas)) / (y34 * etas)
    return float(etas), float(xis)


@dataclass(frozen=True)
class Sqdm11Phase1:
    """``SQDM11`` phase-1 words (``PH1OUT``) for one element."""

    eid: int
    sils: tuple[int, int, int, int]
    tsub0: float
    st: FloatArray
    """``G alpha`` (3,): thermal stress per degree."""
    s: FloatArray
    """``(4, 3, 3)``: ``S_g`` relating grid ``g``'s global translations to the stresses."""
    etas: float
    xis: float

    def words(self) -> FloatArray:
        """The 45 ``PH1OUT`` words (the integer words as numbers)."""
        return np.array([self.eid, *self.sils, self.tsub0, *self.st, *self.s.reshape(-1)], float)


def sqdm11(
    data: ElementData,
    materials: MaterialTable | TrmemProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> Sqdm11Phase1:
    """``SQDM11``: stress matrices at the intersection of the diagonals."""
    props = (
        materials
        if isinstance(materials, TrmemProperties)
        else trmem_properties(data, materials, real_words)
    )
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    dt = np.float32 if real_words else np.float64
    x = np.array([gp.xyz for gp in data.grids], dtype=dt)
    mp = mean_plane(x, dt, real_words, x14_from_x41=True)
    _check_projection(mp, [data.eid])
    etas, xis = (dt(v) for v in diagonal_intersection(mp))
    y4a, y3a, y34 = mp.y4a[0], mp.y3a[0], mp.y34[0]
    x12, x14, x23, x24, x34 = mp.x12[0], mp.x14[0], mp.x23[0], mp.x24[0], mp.x34[0]
    aj = (-y4a * x12) + (-y34 * x12 * xis) + etas * ((-y4a * x23) + (y3a * x14))
    a_u = (
        (-y4a + (y3a * etas) - (y34 * xis)) / aj,
        (y4a - (y4a * etas) + (y34 * xis)) / aj,
        (y4a * etas) / aj,
        (-y3a * etas) / aj,
    )
    a_v = (
        (-x24 + (x23 * etas) + (x34 * xis)) / aj,
        (x14 - (x14 * etas) - (x34 * xis)) / aj,
        ((x14 * etas) - (x12 * xis)) / aj,
        (-x12 - (x23 * etas) + (x12 * xis)) / aj,
    )
    a = np.zeros((3, 8), dt)
    a[0, 0::2] = a_u
    a[1, 1::2] = a_v
    a[2, 0::2] = a_v
    a[2, 1::2] = a_u
    b = mean_plane_b(mp)[0]
    g = np.asarray(props.g).astype(dt)
    s = _mm(g, _mm(a, _mm(b.T.copy(), _block_e(mp.e[0]).T.copy())))
    s_g = np.stack(
        [
            _mm(s[:, 3 * i : 3 * i + 3], _transs(cstm, gp.cid, x[i], real_words).astype(dt))
            for i, gp in enumerate(data.grids)
        ]
    )
    st = _mv(g, np.asarray(props.alpha).astype(dt))
    return Sqdm11Phase1(
        data.eid,
        tuple(int(v) for v in data["sil"]),  # type: ignore[union-attr, arg-type]
        float(dt(props.tref)),
        st.astype(float),
        s_g.astype(float),
        float(etas),
        float(xis),
    )


@dataclass(frozen=True)
class Qdmem1Stress:
    """``SQDM12`` stresses of one element (element mean-plane axes)."""

    eid: int
    sx: float
    sy: float
    sxy: float
    angle: float
    major: float
    minor: float
    max_shear: float

    def words(self) -> FloatArray:
        """``STRES(2..8)`` as the Fortran stores them."""
        return np.array(
            [self.sx, self.sy, self.sxy, self.angle, self.major, self.minor, self.max_shear]
        )

    def row(self) -> FloatArray:
        """Listing order: NORMAL-X, NORMAL-Y, SHEAR-XY, ANGLE, MAJOR, MINOR, MAX SHEAR."""
        return self.words()


def sqdm12(
    phase1: Sqdm11Phase1,
    u: FloatArray,
    temperature: float | None = None,
    real_words: bool = True,
) -> Qdmem1Stress:
    """``SQDM12``: stresses from the displacement vector ``u`` (indexed by SIL).

    ``temperature`` is the element temperature of the thermal load set
    (``None`` when the subcase has none, ``LDTEMP = -1``).
    """
    dt = np.float32 if real_words else np.float64
    uw = np.asarray(u, dtype=dt)
    s = phase1.s.astype(dt)
    stress = np.zeros(3, dt)
    for i, sil in enumerate(phase1.sils):
        ui = uw[sil - 1 : sil + 2]
        vec = np.array([_dsum(s[i, j], ui) for j in range(3)], dtype=float).astype(dt)
        stress = stress + vec
    if temperature is not None:
        tem = dt(temperature) - dt(phase1.tsub0)
        stress = stress - phase1.st.astype(dt) * tem
    sx, sy, sxy = stress
    two = dt(2.0)
    temp = sx - sy
    half = temp / two
    tau = dt(np.sqrt(dt(half * half + sxy * sxy)))
    delta = (sx + sy) / two
    major, minor = delta + tau, delta - tau
    delta = two * sxy
    tiny = dt(1.0e-15)
    if abs(delta) < tiny and abs(temp) < tiny:
        angle = dt(0.0)
    elif abs(temp) < tiny:
        angle = dt(45.0)
    else:
        angle = dt(np.arctan2(delta, temp)) * dt(PHI_SCALE)
    return Qdmem1Stress(phase1.eid, *(float(v) for v in (sx, sy, sxy, angle, major, minor, tau)))


def _dsum(row, ui) -> float:
    """``SMMATS`` inner product: REAL words multiplied and summed in double precision."""
    acc = 0.0
    for a, b in zip(row, ui, strict=True):
        acc = acc + float(a) * float(b)
    return acc


def qdmem1_stresses(
    elements: Sequence[ElementData],
    materials: Materials,
    u: FloatArray,
    cstm: Mapping[int, CoordSystem] | None = None,
    temperatures: float | Sequence[float] | Mapping[int, float] | None = None,
    real_words: bool = True,
) -> list[Qdmem1Stress]:
    """``SQDM11`` + ``SQDM12`` for every element of ``elements`` from displacements ``u``.

    ``temperatures`` gives the thermal-load element temperatures (one value,
    one per element, or a mapping by element id); ``None`` for no thermal load.
    """
    elements = list(elements)
    props = _properties(elements, materials, real_words)
    out = []
    for i, (data, p) in enumerate(zip(elements, props, strict=True)):
        if temperatures is None:
            temp = None
        elif isinstance(temperatures, Mapping):
            temp = temperatures.get(data.eid)
        elif np.ndim(temperatures) == 0:
            temp = float(temperatures)  # type: ignore[arg-type]
        else:
            temp = float(temperatures[i])  # type: ignore[index]
        out.append(sqdm12(sqdm11(data, p, cstm, real_words), u, temp, real_words))
    return out


def stress_table(stresses: Sequence[Qdmem1Stress]) -> FloatArray:
    """``(n, 7)`` rows in listing column order (see :meth:`Qdmem1Stress.row`)."""
    return np.array([s.row() for s in stresses], dtype=float).reshape(-1, 7)
