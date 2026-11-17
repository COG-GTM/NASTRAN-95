"""Quadrilateral membrane (CQDMEM) from four overlapping CTRMEM sub-triangles.

CQDMEM grids 1-4 form the four sub-triangles ``(1, 2, 4)``, ``(2, 3, 1)``,
``(3, 4, 2)`` and ``(4, 1, 3)`` (the ``M`` map of the Fortran), each a
constant-strain CTRMEM. Ported paths:

- stiffness: EMG ``QDMEMS``/``QDMEMD`` are ``OLDEL2`` entries that send the
  element through ``EMGOLD`` to the pivot-point routine ``KQDMEM``
  (``mis/kqdmem.f``), which sums ``KTRMEM (1)`` over the sub-triangles at half
  the thickness (each grid lies in three of the four);
- lumped mass: ``EMGOLD`` calls ``MASSTQ (1)`` (``mis/masstq.f``): for pivot
  grid ``p``, ``(NSM + RHO t) (A_p + A_quad) / 6`` on each translation, with
  ``A_p`` the area of one sub-triangle through ``p`` and ``A_quad`` half the
  diagonal cross product;
- thermal load: ``QDMEM`` (``mis/qdmem.f``, from ``EDTL``) calls ``TRIMEM (1)``
  for each sub-triangle at half the thickness;
- stress recovery: ``SQDME1`` (``mis/sqdme1.f``) calls ``STRME1 (1)`` for each
  sub-triangle, rotates its stress matrices to the quadrilateral axes and
  averages them; phase II is ``STQME2 (2)``.

Each sub-triangle uses its own material angle. ``KQDMEM`` measures it from the
direction of the sub-triangle's first side against ``P = cos(THETA) i +
sin(THETA) j``, with ``i`` along grids 1-2 and ``k = i x (x4 - x1)``.
``SQDME1`` and ``QDMEM`` use the mean-plane axes (``k`` along the diagonal
cross product ``d13 x d24``, ``i`` the projection of side 1-2): with ``(u1,
u2)`` the in-plane direction of the sub-triangle's first side, ``sin =
sin(THETA) u1 - cos(THETA) u2`` and ``cos = cos(THETA) u1 + sin(THETA) u2``;
stresses are rotated back with the 3x3 stress transformation of ``(u1, u2)``.

The kernels are batched over :class:`QdmemBatch`; the per-element functions
(:func:`qdmem`, :func:`qdmem_load`, :func:`sqdme1`, :func:`stqme2_qdmem`) run
them on one element. They reuse the CTRMEM kernels: the ``KTRMEM`` stiffness
follows ``EKTRMD``/``TRMEMD`` (:mod:`nastran95py.elements.trmem`), the thermal
load is :func:`~nastran95py.elements.trmem.trmem_thermal_loads` and the stress
matrices :func:`~nastran95py.elements.trmem_stress.membrane_stress_matrices` /
:func:`~nastran95py.elements.trmem_stress.membrane_stresses` (``NTYPE = 2``).

With ``real_words=True`` (default) the REAL words and REAL steps of the Fortran
are emulated: the sub-triangle angles of ``KQDMEM``/``SQDME1``/``QDMEM``, the
``MASSTQ`` areas, ``MAT``, ``TRIMEM`` and ``STRME1``/``SQDME1`` in float32 (the
``KTRMEM`` products in double precision from REAL coordinates, as the Fortran);
``real_words=False`` evaluates everything in float64.

The heat-transfer branches (``HRING``/``HEAT``) are not ported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem, transforms_to_basic
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import ICODE_TRANSLATION, EmgMatrix, real_cstm
from nastran95py.elements.trmem import (
    DEGRA32,
    TrmemBatch,
    TrmemGeometryError,
    _axes,
    expand_to_grid_dof,
    mat2_real,
    trmem_thermal_loads,
)
from nastran95py.elements.trmem_stress import (
    MembraneStresses,
    TrmemStressBatch,
    _int_word,
    _mm,
    membrane_stress_matrices,
    membrane_stresses,
    thermal_stress_vector,
)
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

SUBTRIANGLES = np.array([[0, 1, 3], [1, 2, 0], [2, 3, 1], [3, 0, 2]])
"""Grids (0-based connection order) of sub-triangles I-IV, as the ``M`` map."""

_MASS_TRIANGLE = ((0, 1, 3), (0, 1, 2), (3, 1, 2), (0, 3, 2))
"""``MASSTQ`` pivot triangle ``(P1, P2, P3)`` for pivot grid 1-4 (area of ``(P3-P1) x (P3-P2)``)."""


class QdmemGeometryError(TrmemGeometryError):
    """Degenerate quadrilateral or sub-triangle (Fortran messages 30/26, 30/31, 30/32)."""


def _dtype(real_words: bool):
    return np.float32 if real_words else np.float64


# ----------------------------------------------------------------------------
# Batch and sub-triangle angles
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class QdmemBatch:
    """CQDMEM elements as stacked arrays (``n`` elements, connection order)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 4)`` SILs."""
    cid: NDArray[np.int64]
    """``(n, 4)`` displacement coordinate systems."""
    xyz: FloatArray
    """``(n, 4, 3)`` basic grid coordinates."""
    theta: FloatArray
    mid: NDArray[np.int64]
    t: FloatArray
    nsm: FloatArray
    temperature: FloatArray
    """Element temperature (ECPT word 26), the ``MAT`` lookup temperature; NaN: none."""

    @classmethod
    def from_elements(cls, elements: Sequence[ElementData]) -> QdmemBatch:
        n = len(elements)

        def col(name, dtype=float):
            return np.array([d[name] for d in elements], dtype=dtype).reshape(n)

        return cls(
            eid=np.array([d.eid for d in elements], np.int64),
            sil=np.array([d["sil"] for d in elements], np.int64).reshape(n, 4),
            cid=np.array([[g.cid for g in d.grids] for d in elements], np.int64).reshape(n, 4),
            xyz=np.array([[g.xyz for g in d.grids] for d in elements], float).reshape(n, 4, 3),
            theta=col("theta"),
            mid=col("mid", np.int64),
            t=col("t"),
            nsm=col("nsm"),
            temperature=np.array(
                [np.nan if d.temperature is None else d.temperature for d in elements], float
            ).reshape(n),
        )

    def __len__(self) -> int:
        return int(self.eid.shape[0])


def _trig(theta: np.ndarray, real_words: bool) -> tuple[np.ndarray, np.ndarray]:
    """``SIN``/``COS`` of ``THETA * DEGRA`` (REAL: the double result rounded once)."""
    if real_words:
        angl = theta.astype(np.float32) * DEGRA32
        a64 = angl.astype(np.float64)
        return np.sin(a64).astype(np.float32), np.cos(a64).astype(np.float32)
    angl = np.radians(theta)
    return np.sin(angl), np.cos(angl)


def _norm3(v: np.ndarray) -> np.ndarray:
    return np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _fail(eid, bad, what: str) -> None:
    if bad.any():
        raise QdmemGeometryError(f"quadrilaterals {np.asarray(eid)[bad].tolist()}: {what}")


def _as_quad_error(call, *args):
    """Run a CTRMEM kernel; its degenerate sub-triangles raise :class:`QdmemGeometryError`."""
    try:
        return call(*args)
    except QdmemGeometryError:
        raise
    except TrmemGeometryError as exc:
        raise QdmemGeometryError(f"sub-triangle: {exc}") from exc


def _zero_small_sin(sinth: np.ndarray, dt) -> np.ndarray:
    """``IF (ABS(SINTH) .LT. 1.0E-06) SINTH = 0.0`` of ``KTRMEM``/``TRIMEM``/``STRME1``."""
    return np.where(np.abs(sinth) < dt(1.0e-6), dt(0.0), sinth).astype(dt)


def kqdmem_angles(xyz, theta, real_words: bool = True, eid=None) -> tuple[FloatArray, FloatArray]:
    """``KQDMEM`` material angle of each sub-triangle: ``SINTH, COSTH`` ``(n, 4)``.

    Sub-triangle I takes ``THETA`` itself; the others the angle between their
    first side and ``P = cos(THETA) i + sin(THETA) j`` measured about ``k``.
    ``SINTH`` below 1e-6 is zeroed, as ``KQDMEM`` does before ``KTRMEM``.
    """
    dt = _dtype(real_words)
    x = np.asarray(xyz, float).astype(dt).reshape(-1, 4, 3)
    n = x.shape[0]
    eid = np.arange(n) if eid is None else np.asarray(eid)
    sinang, cosang = _trig(np.asarray(theta, float).reshape(n), real_words)
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = x[:, 1] - x[:, 0]
        vecl = _norm3(iv)
        _fail(eid, vecl == 0.0, "grids 1 and 2 coincide (30/26)")
        iv = iv / vecl[:, None]
        vk = _cross(iv, x[:, 3] - x[:, 0])
        vecl = _norm3(vk)
        _fail(eid, vecl == 0.0, "grid 4 lies on side 1-2 (30/26)")
        kv = vk / vecl[:, None]
        jv = _cross(kv, iv)
        pv = cosang[:, None] * iv + sinang[:, None] * jv
        sinth = np.zeros((n, 4), dt)
        costh = np.zeros((n, 4), dt)
        sinth[:, 0], costh[:, 0] = sinang, cosang
        for j in range(1, 4):
            a, b = SUBTRIANGLES[j, :2]
            v = x[:, b] - x[:, a]
            vecl = _norm3(v)
            _fail(eid, vecl == 0.0, f"sub-triangle {j + 1} side is degenerate (30/26)")
            u1 = _dot3(v, pv) / vecl
            u2 = _dot3(_cross(v, pv), kv) / vecl
            vecl = np.sqrt(u1 * u1 + u2 * u2)
            _fail(eid, vecl == 0.0, f"sub-triangle {j + 1} angle is undefined (30/26)")
            sinth[:, j], costh[:, j] = u2 / vecl, u1 / vecl
    return _zero_small_sin(sinth, dt).astype(float), costh.astype(float)


def mean_plane_directions(
    xyz, real_words: bool = True, eid=None, tol_k: float = 1.0e-6, tol_v: float = 1.0e-10
) -> FloatArray:
    """``SQDME1``/``QDMEM`` unit directions ``(u1, u2)`` of the sub-triangles' first sides.

    In the mean-plane axes of the quadrilateral: ``k`` along ``d13 x d24``, ``i``
    the side 1-2 less half its ``k`` component, ``j = k x i``. Returns ``(n, 4, 2)``
    (``(1, 0)`` for sub-triangle I). ``tol_k``/``tol_v`` are ``SQDME1``'s fatal
    thresholds (``QDMEM`` only rejects exact zeros: pass 0).
    """
    dt = _dtype(real_words)
    x = np.asarray(xyz, float).astype(dt).reshape(-1, 4, 3)
    n = x.shape[0]
    eid = np.arange(n) if eid is None else np.asarray(eid)

    def bad(v, tol):
        return ~(v > dt(tol)) if tol else v == 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = x[:, 2] - x[:, 0]
        d2 = x[:, 3] - x[:, 1]
        sides = [x[:, 1] - x[:, 0], x[:, 2] - x[:, 1], x[:, 3] - x[:, 2], x[:, 0] - x[:, 3]]
        kv = _cross(d1, d2)
        vecl = _norm3(kv)
        _fail(eid, bad(vecl, tol_k), "diagonals are parallel (30/26)")
        kv = kv / vecl[:, None]
        h = _dot3(sides[0], kv) / dt(2.0)
        iv = sides[0] - h[:, None] * kv
        vecl = _norm3(iv)
        _fail(eid, bad(vecl, tol_k), "side 1-2 is normal to the mean plane (30/26)")
        iv = iv / vecl[:, None]
        jv = _cross(kv, iv)
        jv = jv / _norm3(jv)[:, None]
        v = np.zeros((n, 4, 2), dt)
        v[:, 0, 0] = 1.0
        for j in range(1, 4):
            # GMMATS (R, 2, 3, 0, A, 3, 1, 0, V): rows I and J of R, summed from zero.
            v[:, j, 0] = _dot3(iv, sides[j])
            v[:, j, 1] = _dot3(jv, sides[j])
        vecl = np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])
        _fail(eid, bad(vecl, tol_v).any(axis=1), "a side has no in-plane length (30/26)")
        v = v / vecl[..., None]
    return v.astype(float)


def mean_plane_angles(u, theta, real_words: bool = True) -> tuple[FloatArray, FloatArray]:
    """Net ``SINTH, COSTH`` of the sub-triangles from their directions ``u`` (``SQDME1``/``QDMEM``).

    Not yet zeroed below 1e-6 (``STRME1``/``TRIMEM`` do that).
    """
    dt = _dtype(real_words)
    u = np.asarray(u, float).astype(dt)
    sinang, cosang = _trig(np.asarray(theta, float).reshape(-1), real_words)
    u1, u2 = u[..., 0], u[..., 1]
    sinth = sinang[:, None] * u1 - cosang[:, None] * u2
    costh = cosang[:, None] * u1 + sinang[:, None] * u2
    return sinth.astype(float), costh.astype(float)


def _materials(batch: QdmemBatch, sinth, costh, materials: MaterialTable, real_words: bool):
    """``MAT`` (``INFLAG = 2``) per sub-triangle: ``G (n, 4, 3, 3)``, alpha, TREF, RHO, GE."""
    n = len(batch)
    g = np.zeros((n, 4, 3, 3))
    alpha = np.zeros((n, 4, 3))
    scalars = np.zeros((3, n, 4))
    cache: dict = {}
    temps = [None if np.isnan(t) else float(t) for t in batch.temperature]
    for i in range(n):
        for j in range(4):
            key = (int(batch.mid[i]), float(sinth[i, j]), float(costh[i, j]), temps[i])
            if key not in cache:
                mid, s, c, temp = key
                if real_words:
                    cache[key] = mat2_real(materials, mid, s, c, temp)
                else:
                    m = materials.mat2(mid, s, c, temp)
                    cache[key] = (m.g, m.alpha, m.tref, m.rho, m.ge)
            gm, al, tref, rho, ge = cache[key]
            g[i, j], alpha[i, j] = gm, al
            scalars[:, i, j] = (tref, rho, ge)
    return g, alpha, scalars[0], scalars[1], scalars[2]


def _round(v, real_words: bool) -> FloatArray:
    return np.asarray(v, float).astype(_dtype(real_words)).astype(float)


# ----------------------------------------------------------------------------
# Stiffness (KQDMEM / KTRMEM (1)) and mass (MASSTQ (1))
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class QdmemBatchMatrices:
    """Batched CQDMEM matrices; rows/columns of each element in ascending-SIL order."""

    sils: NDArray[np.int64]
    """``(n, 4)`` SILs in matrix order."""
    k: FloatArray
    """``(n, 12, 12)`` stiffness on the grid translations in global axes."""
    m: FloatArray
    """``(n, 12, 12)`` lumped mass."""
    ge: FloatArray


@dataclass(frozen=True)
class QdmemMatrices:
    """CQDMEM stiffness and mass of one element (translations of the four grids)."""

    eid: int
    sils: tuple[int, int, int, int]
    """SILs in matrix order (ascending)."""
    stiffness: EmgMatrix
    mass: EmgMatrix
    ge: float

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


def subtriangle_stiffness(
    xyz, cid, g, t, cstm: Mapping[int, CoordSystem] | None = None, real_words: bool = True, eid=None
) -> FloatArray:
    """``KTRMEM (1)`` of membrane triangles: the 3x3 blocks ``K_pi`` in global axes.

    ``xyz``: ``(m, 3, 3)`` basic coordinates (REAL words with ``real_words``),
    ``cid``: ``(m, 3)``, ``g``: ``(m, 3, 3)`` plane-stress matrices for the
    triangle's material angle, ``t``: ``(m,)`` thickness. Returns ``(m, 3, 3, 3, 3)``
    (pivot grid, grid, rows, columns): ``T_p^T E C_p^T G C_i E^T T_i * VOL`` in
    double precision, in ``KTRMEM``'s product order. ``cstm`` should hold the REAL
    CSTM words with ``real_words`` (:func:`~nastran95py.elements.real_cstm`).
    """
    dt = _dtype(real_words)
    x = np.asarray(xyz, float).astype(dt).reshape(-1, 3, 3)
    m = x.shape[0]
    cid = np.asarray(cid, np.int64).reshape(m, 3)
    eid = np.arange(m) if eid is None else np.asarray(eid).reshape(m)
    xd = x.astype(float)
    e, xsubb, xsubc, ysubc = _axes(xd[:, 1] - xd[:, 0], xd[:, 2] - xd[:, 0])
    _fail(eid, ~(xsubb > 1.0e-6), "sub-triangle grids coincide (30/31)")
    _fail(eid, ~(ysubc > 1.0e-6), "sub-triangle grids are collinear (30/32)")
    reelmu, flamda = 1.0 / xsubb, 1.0 / ysubc
    delta = xsubc / xsubb - 1.0
    c10 = -flamda * reelmu * xsubc
    c = np.zeros((m, 3, 3, 2))  # grid, strain, (u, v)
    c[:, 0, 0, 0] = -reelmu
    c[:, 0, 1, 1] = c[:, 0, 2, 0] = flamda * delta
    c[:, 0, 2, 1] = -reelmu
    c[:, 1, 0, 0] = c[:, 1, 2, 1] = reelmu
    c[:, 1, 1, 1] = c[:, 1, 2, 0] = c10
    c[:, 2, 1, 1] = c[:, 2, 2, 0] = flamda
    vol = xsubb * ysubc * np.asarray(t, float).reshape(m) / 2.0
    ta = transforms_to_basic(
        {} if cstm is None else cstm, cid.reshape(-1), xd.reshape(-1, 3), real_offset=real_words
    ).reshape(m, 3, 3, 3)
    gm = np.asarray(g, float).reshape(m, 3, 3)
    # pivot side: (E C_p^T) G * VOL, then TI_p^T; other side: C_i E^T TI_i
    common = np.einsum("nxa,ngsa->ngxs", e, c)
    common = np.einsum("ngxs,nsr->ngxr", common, gm) * vol[:, None, None, None]
    common = np.einsum("ngyx,ngyr->ngxr", ta, common)
    right = np.einsum("ngsa,nxa->ngsx", c, e)
    right = np.einsum("ngsx,ngxy->ngsy", right, ta)
    return np.einsum("naxs,nbsy->nabxy", common, right)


def masstq_terms(batch: QdmemBatch, rho, real_words: bool = True) -> FloatArray:
    """``MASSTQ (1)`` lumped mass per grid (connection order): ``(n, 4)``.

    ``(NSM + RHO t) (A_p + A_quad) / 3 / 2`` with ``A_p`` the area of the pivot
    sub-triangle of :data:`_MASS_TRIANGLE` and ``A_quad = |d13 x d24| / 2``.
    """
    dt = _dtype(real_words)
    x = batch.xyz.astype(dt)
    two, three = dt(2.0), dt(3.0)
    quad = _norm3(_cross(x[:, 0] - x[:, 2], x[:, 1] - x[:, 3])) / two
    fmu = batch.nsm.astype(dt)
    t = batch.t.astype(dt)
    rt = np.where(t != 0.0, np.asarray(rho, float).astype(dt) * t, dt(0.0))
    out = np.zeros((len(batch), 4), dt)
    for p, (p1, p2, p3) in enumerate(_MASS_TRIANGLE):
        tri = _norm3(_cross(x[:, p3] - x[:, p1], x[:, p3] - x[:, p2])) / two
        area = tri + quad
        out[:, p] = (fmu + rt) * area / three / two
    return out.astype(float)


def qdmem_matrices(
    batch: QdmemBatch,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> QdmemBatchMatrices:
    """Vectorised ``KQDMEM`` + ``MASSTQ (1)``: 12x12 stiffness and lumped mass of every CQDMEM."""
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    n = len(batch)
    dt = _dtype(real_words)
    sinth, costh = kqdmem_angles(batch.xyz, batch.theta, real_words, batch.eid)
    g, _alpha, _tref, rho, ge = _materials(batch, sinth, costh, materials, real_words)
    t_half = (batch.t.astype(dt) / dt(2.0)).astype(float)
    xt = _round(batch.xyz, real_words)[:, SUBTRIANGLES]  # (n, 4, 3, 3)
    kt = subtriangle_stiffness(
        xt.reshape(-1, 3, 3),
        batch.cid[:, SUBTRIANGLES].reshape(-1, 3),
        g.reshape(-1, 3, 3),
        np.repeat(t_half, 4),
        cstm,
        real_words,
        np.repeat(batch.eid, 4),
    ).reshape(n, 4, 3, 3, 3, 3)
    k4 = np.zeros((n, 4, 4, 3, 3))
    for j in range(4):  # KSUM accumulates the triangles in order I-IV
        for a in range(3):
            for b in range(3):
                k4[:, SUBTRIANGLES[j, a], SUBTRIANGLES[j, b]] += kt[:, j, a, b]
    order = np.argsort(batch.sil, axis=1, kind="stable")
    rows = np.arange(n)[:, None, None]
    k4 = k4[rows, order[:, :, None], order[:, None, :]]
    k = k4.transpose(0, 1, 3, 2, 4).reshape(n, 12, 12)
    terms = np.take_along_axis(masstq_terms(batch, rho[:, 0], real_words), order, axis=1)
    m = np.zeros((n, 12, 12))
    idx = np.arange(12)
    m[:, idx, idx] = np.repeat(terms, 3, axis=1)
    sils = np.take_along_axis(batch.sil, order, axis=1)
    return QdmemBatchMatrices(sils, k, m, ge[:, 0].copy())


def qdmem(
    data: ElementData,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> QdmemMatrices:
    """CQDMEM stiffness (``QDMEMS``/``QDMEMD`` via ``KQDMEM``) and mass of one element."""
    out = qdmem_matrices(QdmemBatch.from_elements([data]), materials, cstm, real_words)
    k, m = out.k[0], out.m[0]
    return QdmemMatrices(
        data.eid,
        tuple(int(s) for s in out.sils[0]),  # type: ignore[arg-type]
        EmgMatrix(k.T.reshape(-1).copy(), 12, ICODE_TRANSLATION, 1),
        EmgMatrix(np.diag(m).copy(), 12, ICODE_TRANSLATION, 2),
        float(out.ge[0]),
    )


# ----------------------------------------------------------------------------
# Thermal load (QDMEM -> TRIMEM (1))
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class QdmemLoad:
    """``QDMEM`` thermal load of one element: a 3-vector per grid, connection order."""

    eid: int
    sils: tuple[int, int, int, int]
    loads: FloatArray
    """``(4, 3)``: row ``g`` is the load on grid ``g`` in its global axes."""

    def add_to(self, pg: FloatArray) -> FloatArray:
        """Add into a load vector indexed by SIL (``pg[sil - 1 + k]``)."""
        for sil, p in zip(self.sils, self.loads, strict=True):
            pg[sil - 1 : sil + 2] += p
        return pg


def qdmem_thermal_loads(
    batch: QdmemBatch,
    materials: MaterialTable,
    tbar: float | Sequence[float],
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> list[QdmemLoad]:
    """Vectorised ``QDMEM``: thermal loads of every CQDMEM in ``batch`` at temperature ``tbar``.

    The four ``TRIMEM (1)`` sub-triangle loads (half thickness, mean-plane
    material angles) are summed per grid in the Fortran order (REAL with
    ``real_words``).
    """
    n = len(batch)
    dt = _dtype(real_words)
    tb = np.broadcast_to(np.asarray(tbar, dtype=float), (n,))
    u = mean_plane_directions(batch.xyz, real_words, batch.eid, 0.0, 0.0)
    sinth, costh = mean_plane_angles(u, batch.theta, real_words)
    sinth = _zero_small_sin(sinth.astype(dt), dt).astype(float)
    g, alpha, tref, rho, ge = _materials(batch, sinth, costh, materials, real_words)
    t_half = (batch.t.astype(dt) / dt(2.0)).astype(float)
    sub = TrmemBatch(
        eid=np.repeat(batch.eid, 4),
        sil=batch.sil[:, SUBTRIANGLES].reshape(-1, 3),
        cid=batch.cid[:, SUBTRIANGLES].reshape(-1, 3),
        xyz=batch.xyz[:, SUBTRIANGLES].reshape(-1, 3, 3),
        g=g.reshape(-1, 3, 3),
        alpha=alpha.reshape(-1, 3),
        tref=tref.reshape(-1),
        rho=rho.reshape(-1),
        ge=ge.reshape(-1),
        t=np.repeat(t_half, 4),
        nsm=np.repeat(batch.nsm, 4),
    )
    loads = _as_quad_error(trmem_thermal_loads, sub, np.repeat(tb, 4), cstm, real_words)
    tri = np.array([ld.loads for ld in loads])
    tri = tri.reshape(n, 4, 3, 3).astype(dt)
    pg = np.zeros((n, 4, 3), dt)
    for j in range(4):
        for a in range(3):
            pg[:, SUBTRIANGLES[j, a]] = pg[:, SUBTRIANGLES[j, a]] + tri[:, j, a]
    return [
        QdmemLoad(int(batch.eid[i]), tuple(int(s) for s in batch.sil[i]), pg[i].astype(float))  # type: ignore[arg-type]
        for i in range(n)
    ]


def qdmem_load(
    data: ElementData,
    materials: MaterialTable,
    tbar: float,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> QdmemLoad:
    """``QDMEM`` for one element: thermal load at mean temperature ``tbar``."""
    batch = QdmemBatch.from_elements([data])
    return qdmem_thermal_loads(batch, materials, tbar, cstm, real_words)[0]


# ----------------------------------------------------------------------------
# Stress recovery (SQDME1 -> STRME1 (1), STQME2 (2))
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class QdmemStressMatrices:
    """``SQDME1`` phase-I output of one CQDMEM (``PH1OUT``, 45 words)."""

    eid: int
    sils: tuple[int, int, int, int]
    """SILs in connection order."""
    tsub0: float
    st: FloatArray
    """``(3,)`` averaged thermal stress vector."""
    s: FloatArray
    """``(4, 3, 3)``: averaged ``S_g`` of grids 1-4 (stress component, global translation)."""

    def words(self) -> FloatArray:
        """The 45 ``PH1OUT`` words; integer words as their REAL bit pattern."""
        head = [_int_word(self.eid), *(_int_word(s) for s in self.sils), self.tsub0]
        return np.array([*head, *self.st, *self.s.reshape(-1)])


def qdmem_phase1(batch: TrmemStressBatch, i: int) -> QdmemStressMatrices:
    """Element ``i`` of a CQDMEM phase-I batch."""
    return QdmemStressMatrices(
        int(batch.eid[i]),
        tuple(int(v) for v in batch.sil[i]),  # type: ignore[arg-type]
        float(batch.tsub0[i]),
        batch.st[i].copy(),
        batch.s[i].copy(),
    )


def stress_rotation(u, real_words: bool = True) -> FloatArray:
    """``SQDME1`` 3x3 stress transformation ``T`` of directions ``u`` (``(..., 2)``)."""
    dt = _dtype(real_words)
    u = np.asarray(u, float).astype(dt)
    u1, u2 = u[..., 0], u[..., 1]
    t1, t2, t7 = u1 * u1, u2 * u2, u1 * u2
    t3 = -(dt(2.0) * t7)
    t = np.stack([t1, t2, t3, t2, t1, -t3, t7, -t7, t1 - t2], axis=-1)
    return t.reshape(*u.shape[:-1], 3, 3).astype(float)


def qdmem_stress_matrices(
    elements: Sequence[ElementData] | QdmemBatch,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemStressBatch:
    """Vectorised ``SQDME1``: averaged sub-triangle stress matrices of every CQDMEM.

    Returns a :class:`~nastran95py.elements.TrmemStressBatch` with four grids
    per element (``s`` is ``(n, 4, 3, 3)``), ready for :func:`qdmem_stresses`;
    :func:`qdmem_phase1` extracts one element's ``PH1OUT`` words.
    """
    batch = elements if isinstance(elements, QdmemBatch) else QdmemBatch.from_elements(elements)
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    dt = _dtype(real_words)
    n = len(batch)
    u = mean_plane_directions(batch.xyz, real_words, batch.eid)
    sinth, costh = mean_plane_angles(u, batch.theta, real_words)
    sinth = _zero_small_sin(sinth.astype(dt), dt).astype(float)
    g, alpha, tref, _rho, _ge = _materials(batch, sinth, costh, materials, real_words)
    s = _as_quad_error(
        membrane_stress_matrices,
        _round(batch.xyz, real_words)[:, SUBTRIANGLES].reshape(-1, 3, 3),
        batch.cid[:, SUBTRIANGLES].reshape(-1, 3),
        g.reshape(-1, 3, 3),
        cstm,
        real_words,
        np.repeat(batch.eid, 4),
    ).reshape(n, 4, 3, 3, 3)
    st = thermal_stress_vector(g.reshape(-1, 3, 3), alpha.reshape(-1, 3), real_words)
    st = st.reshape(n, 4, 3, 1).astype(dt)
    tm = stress_rotation(u, real_words).astype(dt)
    total = np.zeros((n, 4, 3, 3), dt)
    tsum = np.zeros((n, 3, 1), dt)
    for j in range(4):
        for a in range(3):
            gi = SUBTRIANGLES[j, a]
            total[:, gi] = total[:, gi] + _mm(tm[:, j], s[:, j, a].astype(dt), dt)
        tsum = tsum + _mm(tm[:, j], st[:, j], dt)
    quarter = dt(0.25)
    return TrmemStressBatch(
        batch.eid.copy(),
        batch.sil.copy(),
        tref[:, 3].copy(),
        (tsum[:, :, 0] * quarter).astype(float),
        (quarter * total).astype(float),
    )


def sqdme1(
    data: ElementData,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> QdmemStressMatrices:
    """``SQDME1`` for one CQDMEM: averaged stress matrices, ``S_T`` and ``T_SUB_0``."""
    return qdmem_phase1(qdmem_stress_matrices([data], materials, cstm, real_words), 0)


def qdmem_stresses(
    batch: TrmemStressBatch, u, temperature=None, real_words: bool = True
) -> MembraneStresses:
    """Vectorised ``STQME2 (2)`` for CQDMEM elements (see ``membrane_stresses``)."""
    return membrane_stresses(batch, u, temperature, real_words)


def stqme2_qdmem(
    phase1: QdmemStressMatrices, u, temperature: float | None = None, real_words: bool = True
) -> FloatArray:
    """``STQME2 (2)`` for one CQDMEM: the 8 ``STRES`` words (element id as its REAL bit pattern)."""
    batch = TrmemStressBatch(
        np.array([phase1.eid], np.int64),
        np.array([phase1.sils], np.int64),
        np.array([phase1.tsub0]),
        phase1.st[None],
        phase1.s[None],
    )
    t = None if temperature is None else [temperature]
    return membrane_stresses(batch, u, t, real_words).stress_words(0)
