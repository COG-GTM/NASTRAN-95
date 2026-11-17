"""Quadrilateral membrane CQDMEM2: stiffness, mass, thermal load and stress recovery.

Ports the PQDMEM2 routines:

- ``QDMM2D`` (``mis/qdmm2d.f``, EMG, double precision) and its single-precision
  twin ``QDMM2S`` with their helpers ``Q2BCD``/``Q2BCS`` (whole-element axes and
  sub-triangle coordinates), ``Q2TRMD``/``Q2TRMS`` (sub-triangle matrices),
  ``INVERD``/``INVERS`` and the lumped mass of ``EMADTQ(1)``/``EMASTQ(1)``;
- ``QDMM2`` (``mis/qdmm2.f``), the thermal load called from ``EDTL``;
- ``SQDM21``/``SQDM22`` (``mis/sqdm21.f``, ``mis/sqdm22.f``), SDR2 phase I and II.

The quadrilateral A, B, C, D is split into four triangles that share the mean
point 5 of the four grids. The element axes have ``k`` along ``(C - A) x
(D - B)``, ``i`` along ``B - A`` projected on the mean plane and ``j = k x i``;
the sub-triangle coordinates ``R`` are measured from A, shifted by half the
warp ``-h`` along ``k``. Each sub-triangle (``Q2TRM``) gives the constant-strain
membrane stiffness ``H_a^T G H_b * area * t`` with ``H = T_m C E``, ``T_m`` the
material-angle strain rotation. The centre point is condensed out:
``G_i = -K_55^-1 K_5i`` and ``K_ij += K_5i^T G_j`` (plus ``G_i^T K_5j +
G_i^T K_55 G_j`` for a planar element, whose out-of-plane centre equation is
replaced by the average of the corners). The result is transformed to the
global axes of each grid (``T_i^T E^T K E T_j``).

Every function reproduces the Fortran arithmetic: the REAL ECPT words are
rounded to float32, the material angle and ``MAT`` (``INFLAG = 2``) plane-stress
matrix are evaluated in float32 (``trmem.mat2_real``), the double-precision
routines work in float64 from those words and the single-precision ones
(``QDMM2S``, ``QDMM2``, ``SQDM21``, ``SQDM22``) repeat every REAL operation in
float32 in Fortran order (``GMMATS`` accumulates term by term from zero,
``INVERS`` pivots as Gauss-Jordan with full pivoting, ``SMMATS`` accumulates in
double). All kernels are vectorised over a :class:`Qdmem2Batch`; the
single-element functions are thin wrappers.

Rows and columns of the EMG matrices follow ascending SIL as in the Fortran.
The lumped-mass record is reordered exactly as ``QDMM2D`` does it (position of
connection grid ``i`` takes the term of the grid whose connection index is the
SIL rank of ``i``), which equals the ascending-SIL order whenever the
connection order already is ascending; the four terms only differ when the
pivot triangles have different areas.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.coords.transforms import CoordType
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import ICODE_TRANSLATION, EmgMatrix, real_cstm
from nastran95py.elements.trmem import DEGRA32, _basglb32, mat2_real
from nastran95py.io.words import bits_to_real, int_to_bits
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

f32 = np.float32
f64 = np.float64
SUB_TRIANGLES = ((0, 1), (1, 2), (2, 3), (3, 0))
"""``MAP``: corners (IA, IB) of the four sub-triangles; the third point is the centre."""
MASS_PIVOTS = ((0, 1, 3), (0, 1, 2), (3, 1, 2), (0, 3, 2))
"""``EMADTQ(1)`` pivot triangle of each grid (grid indices into A, B, C, D)."""
_EPSI = {f32: 1.0e-30, f64: 1.0e-36}
_PHI = f32(28.64788980)
PHASE1_WORDS = 206
"""Length of the ``SQDM21`` table (``/SDR2X5/`` words 101-306)."""
STRESS_COLUMNS = ("normal_x", "normal_y", "shear_xy", "angle", "major", "minor", "max_shear")


class Qdmem2Error(ValueError):
    """Illegal geometry or singular constraint (Fortran messages 3098, 3099, 3100, 3101)."""


# ----------------------------------------------------------------------------
# Fortran-order arithmetic helpers (all arrays carry a leading element axis)
# ----------------------------------------------------------------------------


def _dot(a, b):
    """``SADOTB``/``DADOTB``."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a, b):
    """``SAXB``/``DAXB``."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _load_atan2f():
    name = ctypes.util.find_library("m")
    try:
        fn = ctypes.CDLL(name).atan2f if name else None
    except (OSError, AttributeError):
        fn = None
    if fn is not None:
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float, ctypes.c_float]
    return fn


_LIBM_ATAN2F = _load_atan2f()


def _atan2f(y, x):
    """REAL ``ATAN2``: the C library's ``atan2f``, as gfortran calls it.

    Its last bit depends on the libm build, and NumPy's float32 SIMD kernel can differ
    from it by an ulp, so the libm routine is called directly when it can be found.
    """
    y, x = np.asarray(y, f32), np.asarray(x, f32)
    if _LIBM_ATAN2F is None:
        return np.arctan2(y.astype(f64), x.astype(f64)).astype(f32)
    out = [_LIBM_ATAN2F(float(a), float(b)) for a, b in zip(y.ravel(), x.ravel(), strict=True)]
    return np.asarray(out, f32).reshape(y.shape)


def _mm(a, b):
    """``GMMATS``/``GMMATD``: ``a @ b`` with the terms added in order to a zero start."""
    if a.dtype != b.dtype:
        raise TypeError(f"mixed precision product {a.dtype} x {b.dtype}")
    out = np.zeros(a.shape[:-1] + b.shape[-1:], dtype=a.dtype)
    for k in range(a.shape[-1]):
        out = out + a[..., :, k, None] * b[..., None, k, :]
    return out


def _mv(a, v):
    return _mm(a, v[..., :, None])[..., 0]


def _tr(a):
    return np.swapaxes(a, -1, -2)


def _smm(a, v):
    """``SMMATS``: REAL ``a @ v`` accumulated in double, rounded once."""
    acc = np.zeros(v.shape, f64)
    for k in range(a.shape[-1]):
        acc = acc + a[..., :, k].astype(f64) * v[..., k, None].astype(f64)
    return acc.astype(f32)


def _invers(a, dt):
    """``INVERS``/``INVERD`` on ``(n, 3, 3)`` matrices indexed as Fortran ``A(J, K)``.

    Gauss-Jordan with full pivoting (first largest ``|A(J, K)|`` in row-major
    search order) and the final column interchanges. Returns the inverse and a
    mask of the matrices the Fortran reports singular (``ISING = 2``).
    """
    a = np.array(a, dtype=dt)
    n = a.shape[0]
    rows = np.arange(n)
    eps = dt(_EPSI[dt])
    used = np.zeros((n, 3), bool)
    pr = np.zeros((n, 3), np.int64)
    pc = np.zeros((n, 3), np.int64)
    sing = np.zeros(n, bool)
    one, zero = dt(1.0), dt(0.0)
    for i in range(3):
        mag = np.where(used[:, :, None] | used[:, None, :], -1.0, np.abs(a).astype(f64))
        flat = mag.reshape(n, 9)
        idx = np.argmax(flat, axis=1)
        sing |= ~(flat[rows, idx] > 0.0)
        idx = np.where(sing, 4 * i, idx)
        ir, ic = idx // 3, idx % 3
        used[rows, ic] = True
        pr[:, i], pc[:, i] = ir, ic
        ra, rc = a[rows, ir].copy(), a[rows, ic].copy()
        a[rows, ir] = rc
        a[rows, ic] = ra
        pivot = a[rows, ic, ic].copy()
        sing |= np.abs(pivot) < eps
        pivot = np.where(sing, one, pivot)
        a[rows, ic, ic] = one
        a[rows, ic] = a[rows, ic] / pivot[:, None]
        for l1 in range(3):
            act = ic != l1
            t = a[rows, l1, ic].copy()
            a[rows, l1, ic] = np.where(act, zero, t)
            new = a[rows, l1] - a[rows, ic] * t[:, None]
            keep = ~act | (np.abs(t) < eps)
            a[rows, l1] = np.where(keep[:, None], a[rows, l1], new)
    for i in (2, 1, 0):
        jr, jc = pr[:, i], pc[:, i]
        ca, cb = a[rows, :, jr].copy(), a[rows, :, jc].copy()
        a[rows, :, jr] = cb
        a[rows, :, jc] = ca
    return a, sing


def _inverse_rowwise(m, dt):
    """``INVERS`` applied to a 3x3 that the caller stores by rows (as ``GMMATS`` reads it)."""
    inv, sing = _invers(_tr(m), dt)
    return _tr(inv), sing


def _transs32(cstm: Mapping[int, CoordSystem], cid, xyz) -> NDArray[np.float32]:
    """``TRANSS`` in REAL arithmetic: ``(n, 3, 3)`` global-to-basic at basic points ``xyz``."""
    cid = np.asarray(cid, np.int64)
    x32 = np.asarray(xyz, f32)
    out = np.zeros(cid.shape + (3, 3), f32)
    out[...] = np.eye(3, dtype=f32)
    for c in np.unique(cid):
        if c == 0:
            continue
        sel = cid == c
        cs = cstm[int(c)]
        ke = cs.t.astype(f32)
        if cs.ctype == CoordType.RECTANGULAR:
            out[sel] = ke
            continue
        xn = x32[sel] - cs.origin.astype(f32)
        x = ke[0, 0] * xn[:, 0] + ke[1, 0] * xn[:, 1] + ke[2, 0] * xn[:, 2]
        y = ke[0, 1] * xn[:, 0] + ke[1, 1] * xn[:, 1] + ke[2, 1] * xn[:, 2]
        z = ke[0, 2] * xn[:, 0] + ke[1, 2] * xn[:, 1] + ke[2, 2] * xn[:, 2]
        r = np.sqrt(x * x + y * y)
        polar = r == f32(0.0)
        rs = np.where(polar, f32(1.0), r)
        tl = np.zeros((x.size, 3, 3), f32)
        if cs.ctype == CoordType.CYLINDRICAL:
            tl[:, 0, 0] = x / rs
            tl[:, 0, 1] = -(y / rs)
            tl[:, 1, 0] = y / rs
            tl[:, 1, 1] = x / rs
            tl[:, 2, 2] = f32(1.0)
        else:
            xl = np.sqrt(x * x + y * y + z * z)
            xls = np.where(polar, f32(1.0), xl)
            tl[:, 0, 0] = x / xls
            tl[:, 0, 1] = (x * z) / (rs * xls)
            tl[:, 0, 2] = -(y / rs)
            tl[:, 1, 0] = y / xls
            tl[:, 1, 1] = (y * z) / (rs * xls)
            tl[:, 1, 2] = x / rs
            tl[:, 2, 0] = z / xls
            tl[:, 2, 1] = -(r / xls)
        ta = _mm(np.broadcast_to(ke, tl.shape).copy(), tl)
        out[sel] = np.where(polar[:, None, None], ke, ta)
    return out


def _sil_rank(sil) -> NDArray[np.int64]:
    """``IPART``: 0-based rank of each grid by SIL."""
    return np.argsort(np.argsort(sil, axis=1, kind="stable"), axis=1, kind="stable")


def _word(i: int) -> float:
    """An integer word as the REAL the Fortran tables hold (bit pattern)."""
    return bits_to_real(int_to_bits(int(i)))


# ----------------------------------------------------------------------------
# Element data
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem2Batch:
    """CQDMEM2 ECPT records and ``MAT`` outputs as stacked arrays (``n`` elements).

    Real words are stored as float64 holding the float32 values the Fortran sees.
    """

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 4)`` SILs in connection order."""
    cid: NDArray[np.int64]
    xyz: FloatArray
    """``(n, 4, 3)`` basic grid coordinates."""
    sinth: FloatArray
    costh: FloatArray
    """Sine and cosine of the material angle (REAL ``SIN``/``COS``)."""
    g: FloatArray
    """``(n, 3, 3)`` ``GSUBE``: plane-stress matrix in material axes (``MAT`` with THETA 0)."""
    alpha: FloatArray
    tref: FloatArray
    rho: FloatArray
    ge: FloatArray
    t: FloatArray
    nsm: FloatArray

    @classmethod
    def from_elements(
        cls, elements: Sequence[ElementData], materials: MaterialTable
    ) -> Qdmem2Batch:
        cols: dict[str, list] = {
            k: [] for k in ("sinth", "costh", "g", "alpha", "tref", "rho", "ge")
        }
        for d in elements:
            angl = f32(float(d["theta"])) * DEGRA32  # type: ignore[arg-type]
            cols["sinth"].append(float(f32(math.sin(float(angl)))))
            cols["costh"].append(float(f32(math.cos(float(angl)))))
            g, alpha, tref, rho, ge = mat2_real(materials, int(d["mid"]), 0.0, 1.0, d.temperature)  # type: ignore[arg-type]
            for k, v in zip(
                ("g", "alpha", "tref", "rho", "ge"), (g, alpha, tref, rho, ge), strict=True
            ):
                cols[k].append(v)
        rnd = lambda name: np.array([float(f32(float(d[name]))) for d in elements], float)  # noqa: E731
        n = len(elements)
        return cls(
            eid=np.array([d.eid for d in elements], np.int64),
            sil=np.array([d["sil"] for d in elements], np.int64).reshape(n, 4),
            cid=np.array([[gp.cid for gp in d.grids] for d in elements], np.int64).reshape(n, 4),
            xyz=np.array([[gp.xyz for gp in d.grids] for d in elements], f32)
            .astype(float)
            .reshape(n, 4, 3),
            sinth=np.array(cols["sinth"], float),
            costh=np.array(cols["costh"], float),
            g=np.array(cols["g"], float).reshape(n, 3, 3),
            alpha=np.array(cols["alpha"], float).reshape(n, 3),
            tref=np.array(cols["tref"], float),
            rho=np.array(cols["rho"], float),
            ge=np.array(cols["ge"], float),
            t=rnd("t"),
            nsm=rnd("nsm"),
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


def _raise_bad(batch: Qdmem2Batch, bad, code: int, what: str) -> None:
    if np.any(bad):
        raise Qdmem2Error(f"CQDMEM2 {batch.eid[bad].tolist()}: {what} ({code})")


# ----------------------------------------------------------------------------
# Q2BC / Q2TRM and the condensation shared by all paths
# ----------------------------------------------------------------------------


def _q2bc(x, dt):
    """``Q2BCD``/``Q2BCS``: sub-triangle coordinates ``(n, 5, 3)``, ``E`` (rows i, j, k), planar."""
    g1, g2, g3, g4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    d12, d13, d24 = g2 - g1, g3 - g1, g4 - g2
    with np.errstate(divide="ignore", invalid="ignore"):
        vec = _cross(d13, d24)
        mag = np.sqrt(_dot(vec, vec))
        iarea = dt(0.5) * mag
        bad = ~(mag > 0)
        ek = vec / mag[:, None]
        itwoh = _dot(d12, ek)
        vec = d12 - itwoh[:, None] * ek
        mag = np.sqrt(_dot(vec, vec))
        bad |= ~(mag > 0)
        ei = vec / mag[:, None]
        ej = _cross(ek, ei)
        e = np.stack([ei, ej, ek], axis=1)
        g5 = dt(0.25) * (g1 + g2 + g3 + g4)
        grid = np.stack([g1, g2, g3, g4, g5], axis=1)
        rmat = np.zeros_like(grid)
        rmat[:, 0, 2] = -itwoh / dt(2.0)
        for i in range(1, 5):
            rmat[:, i] = _mv(e, grid[:, i] - g1) + rmat[:, 0]
        half = itwoh / dt(2.0)
        planar = half * half / iarea <= dt(0.01)
    return rmat, e, planar & ~bad, bad


@dataclass
class _Tri:
    k: dict
    p: list
    s: list
    bad: NDArray[np.bool_]


def _q2trm(ra, rb, rc, sinth, costh, g, t, dt, iopt, alpha=None) -> _Tri:
    """``Q2TRMD``/``Q2TRMS``: one sub-triangle of every element (``IOPT`` 1, 2 or 3)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v12, v13 = rb - ra, rc - ra
        kv = _cross(v12, v13)
        mag = np.sqrt(_dot(kv, kv))
        bad = ~(mag > 0)
        kv = kv / mag[:, None]
        iarea = dt(0.5) * mag
        mag = np.sqrt(_dot(v12, v12))
        bad |= ~(mag > 0)
        iv = v12 / mag[:, None]
        xb = mag
        jv = _cross(kv, iv)
        mag = np.sqrt(iv[:, 0] * iv[:, 0] + iv[:, 1] * iv[:, 1])
        bad |= ~(mag > 0)
        ic = (iv[:, 0] * costh + iv[:, 1] * sinth) / mag
        is_ = (iv[:, 0] * sinth - iv[:, 1] * costh) / mag
        xc = _dot(iv, v13)
        yc = _dot(jv, v13)
        bad |= (xb == 0) | (yc == 0)
        ca1 = dt(-1.0) / xb
        i33 = dt(1.0) / yc
        ca3 = i33 * (xc / xb - dt(1.0))
        cb1 = -ca1
        cb3 = -(i33 * (xc / xb))
    n = ra.shape[0]
    zero = np.zeros(n, dt)
    bt = [
        np.stack([np.stack([c1, zero], -1), np.stack([zero, c3], -1), np.stack([c3, c1], -1)], 1)
        for c1, c3 in ((ca1, ca3), (cb1, cb3), (zero, i33))
    ]
    tm1, tm2, tm3 = ic * ic, is_ * is_, ic * is_
    tm6 = -tm3
    tm7 = dt(2.0) * tm6
    tm = np.stack(
        [
            np.stack([tm1, tm2, tm3], -1),
            np.stack([tm2, tm1, tm6], -1),
            np.stack([tm7, -tm7, tm1 - tm2], -1),
        ],
        1,
    )
    iareat = iarea * t
    e2 = np.stack([iv, jv], axis=1)
    hi = [_mm(_mm(tm, b), e2) for b in bt]
    hitge = [_mm(_tr(h), g) for h in hi]
    hia = [h * iareat[:, None, None] for h in hi]
    pairs = (
        ((2, 0), (2, 1), (2, 2))
        if iopt == 2
        else ((2, 0), (2, 1), (2, 2), (0, 0), (0, 1), (1, 0), (1, 1))
    )
    k = {(a, b): _mm(hitge[a], hia[b]) for a, b in pairs}
    p: list = []
    s: list = []
    if iopt == 2:
        alp = alpha * iareat[:, None]
        p = [_mv(hitge[a], alp) for a in range(3)]
    elif iopt == 3:
        s = [_mm(g, hi[a]) for a in range(3)]
        p = [_mv(hitge[a], alpha) * iareat[:, None] for a in range(3)]
    return _Tri(k, p, s, bad)


def _sum_triangles(rmat, sinth, costh, g, t, dt, iopt, alpha=None):
    """Sub-triangle loop: ``K1SUM`` (16 blocks), ``K5SUM`` (5), ``SISUM``, ``P(I)SUM``."""
    n = rmat.shape[0]
    k1 = np.zeros((n, 4, 4, 3, 3), dt)
    k5 = np.zeros((n, 5, 3, 3), dt)
    s5 = np.zeros((n, 5, 3, 3), dt)
    p5 = np.zeros((n, 5, 3), dt)
    bad = np.zeros(n, bool)
    for ia, ib in SUB_TRIANGLES:
        tri = _q2trm(rmat[:, ia], rmat[:, ib], rmat[:, 4], sinth, costh, g, t, dt, iopt, alpha)
        bad |= tri.bad
        pts = (ia, ib, 4)
        for q in range(3):
            k5[:, pts[q]] = k5[:, pts[q]] + tri.k[(2, q)]
        if iopt != 2:
            k1[:, ia, ia] = k1[:, ia, ia] + tri.k[(0, 0)]
            k1[:, ia, ib] = k1[:, ia, ib] + tri.k[(0, 1)]
            k1[:, ib, ia] = k1[:, ib, ia] + tri.k[(1, 0)]
            k1[:, ib, ib] = k1[:, ib, ib] + tri.k[(1, 1)]
        for q in range(3):
            if tri.p:
                p5[:, pts[q]] = p5[:, pts[q]] + tri.p[q]
            if tri.s:
                s5[:, pts[q]] = s5[:, pts[q]] + tri.s[q]
    return k1, k5, s5, p5, bad


def _planar_k5(k5, planar, dt):
    """Replace the out-of-plane centre equation of planar elements by the corner average."""
    out = k5.copy()
    out[planar, :, 2, :] = np.array([0.0, 0.0, -0.25], dt)
    out[planar, 4, 2, 2] = dt(1.0)
    return out


def _g_matrices(k5mod, dt):
    """``G_i = -K5MOD_55^-1 K5MOD_5i`` and the singular mask."""
    inv, sing = _inverse_rowwise(k5mod[:, 4], dt)
    inv = -inv
    return np.stack([_mm(inv, k5mod[:, i]) for i in range(4)], axis=1), sing


def _condense(k1, k5, gm, planar):
    """``K_ij = K1SUM_ij + K_5i^T G_j`` (+ planar terms), in place on ``k1``."""
    for i in range(4):
        itemp = _mm(_tr(gm[:, i]), k5[:, 4])
        for j in range(4):
            k1[:, i, j] = k1[:, i, j] + _mm(_tr(k5[:, i]), gm[:, j])
            if planar.any():
                jt = _mm(itemp, gm[:, j])
                kt = _mm(_tr(gm[:, i]), k5[:, j])
                k1[planar, i, j] = (k1[:, i, j] + kt + jt)[planar]
    return k1


# ----------------------------------------------------------------------------
# QDMM2D / QDMM2S: stiffness and lumped mass
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem2BatchMatrices:
    """Batched EMG output; rows and columns in ascending-SIL order."""

    sils: NDArray[np.int64]
    """``(n, 4)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 12, 12)`` stiffness on the translations in grid global axes (``KOUT`` by rows)."""
    m: FloatArray
    """``(n, 12)`` lumped mass record (diagonal, ``QDMM2D`` order)."""
    ge: FloatArray
    planar: NDArray[np.bool_]

    def element(self, i: int, eid: int = 0) -> Qdmem2Matrices:
        return Qdmem2Matrices(
            eid,
            tuple(int(s) for s in self.sils[i]),  # type: ignore[arg-type]
            EmgMatrix(self.k[i].reshape(-1), 12, ICODE_TRANSLATION, 1),
            EmgMatrix(self.m[i].copy(), 12, ICODE_TRANSLATION, 2),
            float(self.ge[i]),
            bool(self.planar[i]),
        )


@dataclass(frozen=True)
class Qdmem2Matrices:
    """``QDMM2D`` output for one element: the two ``EMGOUT`` records."""

    eid: int
    sils: tuple[int, int, int, int]
    stiffness: EmgMatrix
    """Data words as ``EMGOUT`` receives them (``KOUT``, by rows)."""
    mass: EmgMatrix
    ge: float
    planar: bool

    def k(self) -> FloatArray:
        """12x12 stiffness, ascending SIL, three translations per grid."""
        return self.stiffness.values.reshape(12, 12)

    def m(self) -> FloatArray:
        return np.diag(self.mass.values)


def _ta(batch: Qdmem2Batch, cstm, dt):
    if dt is f32:
        return _transs32(cstm, batch.cid, batch.xyz)
    n = len(batch)
    return transforms_to_basic(
        cstm, batch.cid.reshape(-1), batch.xyz.reshape(-1, 3), real_offset=True
    ).reshape(n, 4, 3, 3)


def _mass(batch: Qdmem2Batch, dt):
    """``EMADTQ(1)``/``EMASTQ(1)``: lumped translational mass of each grid, connection order."""
    x = batch.xyz.astype(f32)
    nsm, rho, t = (getattr(batch, k).astype(f32).astype(dt) for k in ("nsm", "rho", "t"))

    def half_area(v1, v2):
        c = _cross(v1.astype(dt), v2.astype(dt))
        return np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2 + c[:, 2] ** 2) / dt(2.0)

    whole = (x[:, 0] - x[:, 2], x[:, 1] - x[:, 3])
    out = np.zeros((len(batch), 4), dt)
    for g, (p1, p2, p3) in enumerate(MASS_PIVOTS):
        area = dt(0.0) + half_area(x[:, p3] - x[:, p1], x[:, p3] - x[:, p2])
        area = area + half_area(*whole)
        term = (nsm + rho * t) * area / dt(3.0)
        out[:, g] = term / dt(2.0)
    return out


def qdmem2_matrices(
    batch: Qdmem2Batch,
    cstm: Mapping[int, CoordSystem] | None = None,
    precision: str = "double",
) -> Qdmem2BatchMatrices:
    """Vectorised ``QDMM2D`` (``precision="single"``: ``QDMM2S``) stiffness and mass."""
    dt = {"double": f64, "single": f32}[precision]
    cstm = real_cstm({} if cstm is None else cstm)
    n = len(batch)
    x = batch.xyz.astype(f32).astype(dt)
    sinth, costh = batch.sinth.astype(f32).astype(dt), batch.costh.astype(f32).astype(dt)
    g = batch.g.astype(f32).astype(dt)
    t = batch.t.astype(f32).astype(dt)
    rmat, e, planar, bad = _q2bc(x, dt)
    _raise_bad(batch, bad, 3098, "illegal geometry")
    k1, k5, _, _, bad = _sum_triangles(rmat, sinth, costh, g, t, dt, 1)
    _raise_bad(batch, bad, 3098, "illegal geometry")
    gm, sing = _g_matrices(_planar_k5(k5, planar, dt), dt)
    _raise_bad(batch, sing, 3099, "singular constraint equation")
    ta = _ta(batch, cstm, dt).astype(dt)
    tmat = np.where((batch.cid != 0)[:, :, None, None], _mm(e[:, None], ta), e[:, None])
    _condense(k1, k5, gm, planar)
    for i in range(4):
        for j in range(4):
            k1[:, i, j] = _mm(_mm(_tr(tmat[:, i]), k1[:, i, j]), tmat[:, j])
    rank = _sil_rank(batch.sil)
    k = np.zeros((n, 12, 12))
    rows = np.arange(n)
    for i in range(4):
        for j in range(4):
            ri, rj = rank[:, i], rank[:, j]
            for a in range(3):
                for b in range(3):
                    k[rows, 3 * ri + a, 3 * rj + b] = k1[:, i, j, a, b]
    mass = _mass(batch, dt)
    m = np.repeat(np.take_along_axis(mass, rank, axis=1), 3, axis=1).astype(float)
    sils = np.take_along_axis(batch.sil, np.argsort(batch.sil, axis=1, kind="stable"), axis=1)
    return Qdmem2BatchMatrices(sils, k, m, batch.ge.astype(f32).astype(float), planar)


def qdmm2d(
    data: ElementData,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    precision: str = "double",
) -> Qdmem2Matrices:
    """``QDMM2D`` (or ``QDMM2S``) for one element."""
    batch = Qdmem2Batch.from_elements([data], materials)
    return qdmem2_matrices(batch, cstm, precision).element(0, data.eid)


# ----------------------------------------------------------------------------
# QDMM2: thermal load
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem2Load:
    """``QDMM2`` thermal load of one element: a 3-vector per grid in connection order."""

    eid: int
    sils: tuple[int, int, int, int]
    loads: FloatArray
    """``(4, 3)``: row ``g`` is the load on grid ``g`` in its global axes."""

    def add_to(self, pg: FloatArray) -> FloatArray:
        """Add into a load vector indexed by SIL (``pg[sil - 1 + k]``), as ``QDMM2`` does."""
        for sil, p in zip(self.sils, self.loads, strict=True):
            pg[sil - 1 : sil + 2] += p
        return pg


def qdmem2_thermal_loads(
    batch: Qdmem2Batch,
    temperature: float | Sequence[float],
    cstm: Mapping[int, CoordSystem] | None = None,
) -> list[Qdmem2Load]:
    """Vectorised ``QDMM2``: thermal load at element temperature ``TEMPS(1)`` (REAL arithmetic)."""
    dt = f32
    cstm = real_cstm({} if cstm is None else cstm)
    n = len(batch)
    temps = np.broadcast_to(np.asarray(temperature, float), (n,)).astype(f32)
    tbar = temps - batch.tref.astype(f32)
    alpha = batch.alpha.astype(f32) * tbar[:, None]
    x = batch.xyz.astype(f32)
    rmat, e, planar, bad = _q2bc(x, dt)
    _raise_bad(batch, bad, 3100, "illegal geometry")
    g, t = batch.g.astype(f32), batch.t.astype(f32)
    _, k5, _, p5, bad = _sum_triangles(
        rmat, batch.sinth.astype(f32), batch.costh.astype(f32), g, t, dt, 2, alpha
    )
    _raise_bad(batch, bad, 3100, "illegal geometry")
    k5 = _planar_k5(k5, planar, dt)
    inv, sing = _inverse_rowwise(k5[:, 4], dt)
    _raise_bad(batch, sing, 3100, "singular constraint equation")
    k55 = -inv
    loads = np.zeros((n, 4, 3), f32)
    for i in range(4):
        it9 = _mm(k55, k5[:, i])
        p5[:, i] = p5[:, i] + _mv(_tr(it9), p5[:, 4])
        loads[:, i] = _mv(_tr(e), p5[:, i])
        for c in np.unique(batch.cid[:, i]):
            if c == 0:
                continue
            sel = batch.cid[:, i] == c
            loads[sel, i] = _basglb32(cstm[int(c)], loads[sel, i], x[sel, i])
    return [
        Qdmem2Load(int(batch.eid[i]), tuple(int(s) for s in batch.sil[i]), loads[i].astype(float))  # type: ignore[arg-type]
        for i in range(n)
    ]


def qdmm2(
    data: ElementData,
    materials: MaterialTable,
    temperature: float,
    cstm: Mapping[int, CoordSystem] | None = None,
) -> Qdmem2Load:
    """``QDMM2`` for one element."""
    batch = Qdmem2Batch.from_elements([data], materials)
    return qdmem2_thermal_loads(batch, temperature, cstm)[0]


# ----------------------------------------------------------------------------
# SQDM21 / SQDM22: stress recovery
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Qdmem2StressMatrices:
    """``SQDM21`` tables of ``n`` elements (REAL values held as float64)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    t: FloatArray
    tref: FloatArray
    kij: FloatArray
    """``(n, 4, 4, 3, 3)`` corner-force matrices ``Q_i^-1 K_ij E T_j``."""
    sg: FloatArray
    """``(n, 4, 3, 3)`` stress matrices of the four grids."""
    pt: FloatArray
    """``(n, 4, 3)`` thermal corner-force vectors per unit ``TEMP - TSUB0``."""
    st: FloatArray
    """``(n, 3)`` ``GSUBE * ALPHA``: thermal stress per unit ``TEMP - TSUB0``."""
    rg: FloatArray
    """``(n, 4)`` side lengths (A-B, B-C, C-D, D-A)."""

    def __len__(self) -> int:
        return int(self.eid.shape[0])

    def words(self, i: int) -> FloatArray:
        """The 206-word ``/SDR2X5/`` table of element ``i`` (integer words as REAL bit patterns)."""
        return np.array(
            [
                _word(self.eid[i]),
                *(_word(s) for s in self.sil[i]),
                self.t[i],
                self.tref[i],
                *self.kij[i].reshape(-1),
                *self.sg[i].reshape(-1),
                *self.pt[i].reshape(-1),
                *self.st[i],
                *self.rg[i],
            ]
        )


def qdmem2_stress_matrices(
    batch: Qdmem2Batch, cstm: Mapping[int, CoordSystem] | None = None
) -> Qdmem2StressMatrices:
    """Vectorised ``SQDM21`` (REAL arithmetic)."""
    dt = f32
    cstm = real_cstm({} if cstm is None else cstm)
    x = batch.xyz.astype(f32)
    g, t, alps = batch.g.astype(f32), batch.t.astype(f32), batch.alpha.astype(f32)
    rmat, e, planar, bad = _q2bc(x, dt)
    _raise_bad(batch, bad, 3101, "bad geometry")
    k1, k5, s5, p5, bad = _sum_triangles(
        rmat, batch.sinth.astype(f32), batch.costh.astype(f32), g, t, dt, 3, alps
    )
    _raise_bad(batch, bad, 3101, "bad geometry")
    gm, sing = _g_matrices(_planar_k5(k5, planar, dt), dt)
    _raise_bad(batch, sing, 3101, "singularity")
    _condense(k1, k5, gm, planar)

    n = len(batch)
    dvec = np.zeros((n, 4, 3), dt)
    rg = np.zeros((n, 4), dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, (ia, ib) in enumerate(SUB_TRIANGLES):
            d = rmat[:, ib] - rmat[:, ia]
            rg[:, i] = np.sqrt(_dot(d, d))
            dvec[:, i] = d / rg[:, i, None]
        bad = ~(rg > 0).all(axis=1)
        q = np.zeros((n, 4, 3, 3), dt)
        for i in range(4):
            j = (i - 1) % 4
            kv = _cross(dvec[:, i], dvec[:, j])
            fmag = np.sqrt(_dot(kv, kv))
            bad |= ~(fmag > 0)
            q[:, i, :, 2] = -(kv / fmag[:, None])
            q[:, i, :, 0] = -dvec[:, i]
            q[:, i, :, 1] = dvec[:, j]
    _raise_bad(batch, bad, 3101, "bad geometry")
    qinv = np.zeros_like(q)
    for i in range(4):
        qinv[:, i], sing = _invers(q[:, i], dt)
        _raise_bad(batch, sing, 3101, "singularity")

    ta = _transs32(cstm, batch.cid, x)
    eti = np.where((batch.cid != 0)[:, :, None, None], _mm(e[:, None], ta), e[:, None])
    sg = np.zeros((n, 4, 3, 3), dt)
    pt = np.zeros((n, 4, 3), dt)
    for i in range(4):
        jt = _mm(s5[:, 4], gm[:, i])
        jt = dt(0.25) * (jt + s5[:, i])
        sg[:, i] = _mm(jt, eti[:, i])
        p = _mv(_tr(gm[:, i]), p5[:, 4])
        p5[:, i] = p + p5[:, i]
        pt[:, i] = _mv(qinv[:, i], p5[:, i])
    for i in range(4):
        for j in range(4):
            k1[:, i, j] = _mm(_mm(qinv[:, i], k1[:, i, j]), eti[:, j])
    st = _mv(g, alps)
    as64 = lambda a: a.astype(float)  # noqa: E731
    return Qdmem2StressMatrices(
        batch.eid.copy(),
        batch.sil.copy(),
        as64(t),
        batch.tref.astype(f32).astype(float),
        as64(k1),
        as64(sg),
        as64(pt),
        as64(st),
        as64(rg),
    )


def sqdm21(
    data: ElementData, materials: MaterialTable, cstm: Mapping[int, CoordSystem] | None = None
) -> Qdmem2StressMatrices:
    """``SQDM21`` for one element."""
    return qdmem2_stress_matrices(Qdmem2Batch.from_elements([data], materials), cstm)


@dataclass(frozen=True)
class Qdmem2Stresses:
    """``SQDM22`` results of ``n`` elements (REAL values held as float64)."""

    eid: NDArray[np.int64]
    stress: FloatArray
    """``(n, 7)``: ``STRESS_COLUMNS`` (sigma x, y, tau xy, angle, major, minor, max shear)."""
    corner_forces: FloatArray
    """``(n, 4, 3)``: force ``F(I, 1..3)`` at each grid along the element sides and normal."""
    kick: FloatArray
    """``(n, 4)`` shear flows ``Q1..Q4`` with the listing signs."""

    def stress_table(self) -> FloatArray:
        return self.stress.copy()

    def stress_words(self, i: int) -> FloatArray:
        return np.array([_word(self.eid[i]), *self.stress[i]])

    def force_words(self, i: int) -> FloatArray:
        f, q = self.corner_forces[i], self.kick[i]
        return np.array(
            [
                _word(self.eid[i]),
                f[0, 1],
                f[0, 0],
                f[1, 1],
                f[1, 0],
                f[2, 1],
                f[2, 0],
                f[3, 1],
                f[3, 0],
                f[0, 2],
                q[0],
                f[1, 2],
                q[1],
                f[2, 2],
                q[2],
                f[3, 2],
                q[3],
            ]
        )


def qdmem2_stresses(
    p1: Qdmem2StressMatrices,
    displacements: FloatArray,
    temperature: float | Sequence[float] | None = None,
) -> Qdmem2Stresses:
    """Vectorised ``SQDM22``: stresses and corner forces from a SIL-indexed displacement vector.

    ``displacements[sil - 1 + k]`` is component ``k`` of the grid with that SIL
    (global axes, REAL words); ``temperature`` is the element load temperature
    (``None`` or NaN: no thermal term, ``LDTEMP = -1``).
    """
    n = len(p1)
    u = np.asarray(displacements, float).astype(f32)
    ug = np.stack([u[p1.sil[:, g, None] - 1 + np.arange(3)] for g in range(4)], axis=1)
    if temperature is None:
        temps = np.full(n, np.nan)
    else:
        temps = np.broadcast_to(np.asarray(temperature, float), (n,))
    has_t = ~np.isnan(temps)
    tbar = np.where(has_t, temps, 0.0).astype(f32) - p1.tref.astype(f32)
    sg, kij = p1.sg.astype(f32), p1.kij.astype(f32)
    pt, st, rg = p1.pt.astype(f32), p1.st.astype(f32), p1.rg.astype(f32)
    sig = np.zeros((n, 3), f32)
    for i in range(4):
        sig = sig + _smm(sg[:, i], ug[:, i])
    sig = np.where(has_t[:, None], sig - st * tbar[:, None], sig)
    f = np.zeros((n, 4, 3), f32)
    for i in range(4):
        for j in range(4):
            f[:, i] = f[:, i] + _smm(kij[:, i, j], ug[:, j])
        f[:, i] = np.where(has_t[:, None], f[:, i] - pt[:, i] * tbar[:, None], f[:, i])
    with np.errstate(divide="ignore", invalid="ignore"):
        shears = np.stack(
            [(f[:, (i + 1) % 4, 1] - f[:, i, 0]) / rg[:, i] for i in range(4)], axis=1
        )
    kick = shears * np.array([-1.0, 1.0, -1.0, 1.0], f32)
    sx, sy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    two = f32(2.0)
    temp = sx - sy
    half = temp / two
    tau = np.sqrt(half * half + txy * txy)
    delta = (sx + sy) / two
    major, minor = delta + tau, delta - tau
    delta = two * txy
    tiny = f32(1.0e-15)
    with np.errstate(invalid="ignore"):
        phi = np.where(
            np.abs(temp) < tiny,
            np.where(np.abs(delta) < tiny, f32(45.0), f32(0.0)),
            _atan2f(delta, temp) * _PHI,
        ).astype(f32)
    stress = np.stack([sx, sy, txy, phi, major, minor, tau], axis=1).astype(float)
    return Qdmem2Stresses(p1.eid.copy(), stress, f.astype(float), kick.astype(float))


def sqdm22(
    p1: Qdmem2StressMatrices, displacements: FloatArray, temperature: float | None = None
) -> Qdmem2Stresses:
    """``SQDM22`` (alias of :func:`qdmem2_stresses` for symmetry with the Fortran names)."""
    return qdmem2_stresses(p1, displacements, temperature)
