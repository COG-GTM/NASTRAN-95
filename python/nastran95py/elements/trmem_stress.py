"""Triangular membrane (CTRMEM) stress recovery: port of ``STRME1``/``STQME2``.

SDR2 recovers membrane stresses in two phases (``mis/strme1.f``, ``mis/stqme2.f``):

- phase I (``STRME1``) turns the ECPT record into stress matrices. With the
  element axes ``E = [i j]`` (3x2, ``i`` along ``B - A``, ``j`` in-plane), the
  3x2 strain-displacement block ``C_g`` of grid ``g`` (as in ``TRMEMD``), the
  plane-stress matrix ``G`` from ``MAT`` (``INFLAG = 2``, MAT1 or MAT2 rotated by
  the material angle, at the ECPT element temperature) and the ``TRANSS``
  global-to-basic matrix ``T_g`` of each grid::

      S_g = G C_g E^T T_g      (3x3, stress per unit global translation of grid g)
      S_T = G alpha            T_SUB_0 = TREF

- phase II (``STQME2``) applies the global displacements ``u`` (SIL-indexed)
  and, when the subcase has a load temperature, the thermal correction::

      [sigma_x, sigma_y, tau_xy] = sum_g S_g u_g - S_T (T_LOAD - T_SUB_0)

  then the principal stresses ``(sx + sy)/2 +- R``, the maximum shear
  ``R = sqrt(((sx - sy)/2)^2 + txy^2)`` and the principal angle
  ``atan2(2 txy, sx - sy)/2`` in degrees (0 when both arguments are below 1e-15).

``STRME1`` is also called with ``NTYPE = 1`` by ``SQDME1`` for the four
overlapping sub-triangles of CQDMEM, with the material angle set by the caller,
and ``STQME2`` is shared by CTRMEM (``NTYPE = 1``, 3 grids) and CQDMEM
(``NTYPE = 2``, 4 grids). :func:`membrane_stress_matrices` (phase I for given
``G``) and :func:`membrane_stresses` (phase II for any number of grids) are the
kernels exposed for that reuse; :func:`strme1`/:func:`stqme2` are the CTRMEM
per-element forms and :func:`trmem_stress_matrices`/:func:`trmem_stresses` the
batched ones over many elements. The scalar forms run the batched kernel on one
element, so both give identical words.

Both routines work in REAL (single precision); only ``SMMATS`` accumulates its
products in double precision. With ``real_words=True`` (default) every step is
evaluated in float32 in the Fortran order, including ``MAT``, ``TRANSS`` with
the REAL CSTM words it reads, ``GMMATS`` (accumulated term by term) and the
principal-stress formulas, so the results reproduce the gfortran build to the
last bit. ``real_words=False`` evaluates the same expressions in float64.

The ``NCHK`` precision check of ``STQME2`` (a diagnostic print) and the
``STRAIN`` option of ``STRME1`` (``G`` replaced by the identity) are not ported.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem
from nastran95py.coords.transforms import CoordType
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import real_cstm
from nastran95py.elements.trmem import TrmemGeometryError, trmem_properties
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

ANGLE_FACTOR = 28.6478898
"""``STQME2`` constant: degrees per radian over 2 (the half-angle of Mohr's circle)."""
NO_TEMPERATURE = np.nan
"""Load temperature of a subcase without TEMPERATURE(LOAD) (``LDTEMP = -1``)."""


def _load_atan2f():
    name = ctypes.util.find_library("m")
    try:
        fn = ctypes.CDLL(name).atan2f if name else None
    except (AttributeError, OSError):
        fn = None
    if fn is not None:
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float, ctypes.c_float]
    return fn


_LIBM_ATAN2F = _load_atan2f()


def _atan2f(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """REAL ``ATAN2``: the C library's ``atan2f``, which gfortran calls.

    NumPy's float32 ``arctan2`` is dispatched to CPU-dependent SIMD kernels and
    differs from libm by 1 ulp on some runners, so libm is called directly
    (falling back to NumPy where no libm is found).
    """
    y32, x32 = np.asarray(y, np.float32), np.asarray(x, np.float32)
    if _LIBM_ATAN2F is None:
        return np.arctan2(y32, x32)
    fn = _LIBM_ATAN2F
    return np.array([fn(float(a), float(b)) for a, b in zip(y32, x32, strict=True)], np.float32)


def _int_word(i: int) -> float:
    """The REAL reading of an integer word (as the harness prints ``PH1OUT``/``STRES``)."""
    return struct.unpack("<f", struct.pack("<i", int(i)))[0]


def _dtype(real_words: bool):
    return np.float32 if real_words else np.float64


# ----------------------------------------------------------------------------
# Phase I kernels
# ----------------------------------------------------------------------------


def _axes(x: np.ndarray, dt):
    """``STRME1`` element axes and ``C``: ``x`` is ``(n, 3, 3)`` (grid, xyz) in ``dt``.

    Returns ``E (n, 3, 2)``, ``C (n, grid, strain, uv)``, ``XSUBB``, ``YSUBC``.
    """
    one = dt(1.0)
    d1 = x[:, 1] - x[:, 0]
    xsubb = np.sqrt(d1[:, 0] * d1[:, 0] + d1[:, 1] * d1[:, 1] + d1[:, 2] * d1[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ei = d1 / xsubb[:, None]
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
        ysubc = np.sqrt(ek[:, 0] * ek[:, 0] + ek[:, 1] * ek[:, 1] + ek[:, 2] * ek[:, 2])
        ek = ek / ysubc[:, None]
        ej = np.stack(
            [
                ei[:, 2] * ek[:, 1] - ei[:, 1] * ek[:, 2],
                ei[:, 0] * ek[:, 2] - ei[:, 2] * ek[:, 0],
                ei[:, 1] * ek[:, 0] - ei[:, 0] * ek[:, 1],
            ],
            axis=1,
        )
        temp = np.sqrt(ej[:, 0] * ej[:, 0] + ej[:, 1] * ej[:, 1] + ej[:, 2] * ej[:, 2])
        ej = ej / temp[:, None]
        # REELMU = 1.0D0/XSUBB, FLAMDA = 1.0D0/YSUBC: double division stored as REAL.
        reelmu = (1.0 / xsubb.astype(np.float64)).astype(dt)
        flamda = (1.0 / ysubc.astype(np.float64)).astype(dt)
        delta = xsubc / xsubb - one
    n = x.shape[0]
    c = np.zeros((n, 3, 3, 2), dt)
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


def _mm(a: np.ndarray, b: np.ndarray, dt) -> np.ndarray:
    """``GMMATS`` batched ``(n, p, q) @ (n, q, r)``: accumulated term by term in ``dt``."""
    out = np.zeros((a.shape[0], a.shape[1], b.shape[2]), dt)
    for k in range(a.shape[2]):
        out = out + a[:, :, k, None] * b[:, None, k, :]
    return out


def _system(cstm: Mapping[int, CoordSystem], cid: int) -> CoordSystem:
    if cid not in cstm:
        raise KeyError(f"coordinate system {cid} is not in the CSTM")
    return cstm[cid]


def transs(cs: CoordSystem, pts: np.ndarray, real_words: bool = True) -> FloatArray:
    """``TRANSS`` (``mis/pretrs.f``) for ``(m, 3)`` basic points of one system: ``(m, 3, 3)``.

    The global-to-basic matrix ``TA`` (``v_basic = TA v_global``), evaluated in
    the working precision in the Fortran order from the CSTM words.
    """
    dt = _dtype(real_words)
    ke = np.asarray(cs.t).astype(dt)
    pts = np.asarray(pts).astype(dt).reshape(-1, 3)
    m = pts.shape[0]
    ta = np.broadcast_to(ke, (m, 3, 3)).copy()
    if int(cs.ctype) == CoordType.RECTANGULAR:
        return ta
    xn = pts - np.asarray(cs.origin).astype(dt)
    x = ke[0, 0] * xn[:, 0] + ke[1, 0] * xn[:, 1] + ke[2, 0] * xn[:, 2]
    y = ke[0, 1] * xn[:, 0] + ke[1, 1] * xn[:, 1] + ke[2, 1] * xn[:, 2]
    z = ke[0, 2] * xn[:, 0] + ke[1, 2] * xn[:, 1] + ke[2, 2] * xn[:, 2]
    r = np.sqrt(x * x + y * y)
    k = r != 0.0
    if not k.any():
        return ta
    x, y, z, r = x[k], y[k], z[k], r[k]
    tl = np.zeros((x.shape[0], 3, 3), dt)
    if int(cs.ctype) == CoordType.CYLINDRICAL:
        tl[:, 0, 0] = x / r
        tl[:, 0, 1] = -y / r
        tl[:, 1, 0] = -tl[:, 0, 1]
        tl[:, 1, 1] = tl[:, 0, 0]
        tl[:, 2, 2] = dt(1.0)
    else:
        xl = np.sqrt(x * x + y * y + z * z)
        tl[:, 0, 0] = x / xl
        tl[:, 0, 1] = (x * z) / (r * xl)
        tl[:, 0, 2] = -y / r
        tl[:, 1, 0] = y / xl
        tl[:, 1, 1] = (y * z) / (r * xl)
        tl[:, 1, 2] = x / r
        tl[:, 2, 0] = z / xl
        tl[:, 2, 1] = -r / xl
    ta[k] = _mm(np.broadcast_to(ke, tl.shape), tl, dt)
    return ta


def membrane_stress_matrices(
    xyz,
    cid,
    g,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
    eid=None,
) -> FloatArray:
    """``STRME1`` stress matrices ``S_g = G C_g E^T T_g`` of membrane triangles.

    ``xyz``: ``(n, 3, 3)`` basic coordinates of grids A, B, C; ``cid``: ``(n, 3)``
    displacement systems; ``g``: ``(n, 3, 3)`` plane-stress matrices (as ``MAT``
    returns them for the material angle). Returns ``(n, 3, 3, 3)``: grid, stress
    component, global translation. This is the ``NTYPE = 1`` entry ``SQDME1``
    uses for the CQDMEM sub-triangles. With ``real_words=True`` ``cstm`` should
    hold the REAL CSTM words (:func:`nastran95py.elements.real_cstm`).
    """
    dt = _dtype(real_words)
    x = np.asarray(xyz, float).astype(dt).reshape(-1, 3, 3)
    cid = np.asarray(cid, np.int64).reshape(-1, 3)
    n = x.shape[0]
    eid = np.arange(n) if eid is None else np.asarray(eid).reshape(-1)
    e, c, xsubb, ysubc = _axes(x, dt)
    tiny = dt(1.0e-6)
    for bad, msg in ((~(xsubb > tiny), "grids A and B coincide (30/31)"), (~(ysubc > tiny), None)):
        if bad.any():
            what = msg or "grids are collinear (30/32)"
            raise TrmemGeometryError(f"triangles {eid[bad].tolist()}: {what}")
    gm = np.asarray(g, float).astype(dt).reshape(n, 3, 3)
    et = np.ascontiguousarray(np.swapaxes(e, 1, 2))  # (n, 2, 3) = E^T
    s = np.zeros((n, 3, 3, 3), dt)
    cstm = {} if cstm is None else cstm
    for gi in range(3):
        tempar = _mm(_mm(gm, c[:, gi], dt), et, dt)
        s[:, gi] = tempar
        for c_id in np.unique(cid[:, gi]):
            if c_id == 0:
                continue
            sel = cid[:, gi] == c_id
            ti = transs(_system(cstm, int(c_id)), x[sel, gi], real_words)
            s[sel, gi] = _mm(tempar[sel], ti, dt)
    return s.astype(float)


def thermal_stress_vector(g, alpha, real_words: bool = True) -> FloatArray:
    """``S_T = G alpha`` (``GMMATS``), ``(n, 3)``."""
    dt = _dtype(real_words)
    gm = np.asarray(g, float).astype(dt).reshape(-1, 3, 3)
    a = np.asarray(alpha, float).astype(dt).reshape(-1, 3, 1)
    return _mm(gm, a, dt)[:, :, 0].astype(float)


# ----------------------------------------------------------------------------
# Phase I: CTRMEM
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrmemStressMatrices:
    """``STRME1`` phase-I output of one CTRMEM (``PH1OUT``)."""

    eid: int
    sils: tuple[int, int, int]
    """SILs in connection order."""
    tsub0: float
    st: FloatArray
    """``(3,)`` thermal stress vector ``G alpha``."""
    s: FloatArray
    """``(3, 3, 3)``: ``S_g`` of grids A, B, C (stress component, global translation)."""

    def words(self) -> FloatArray:
        """The 36 ``PH1OUT`` words; integer words as their REAL bit pattern."""
        head = [_int_word(self.eid), *(_int_word(s) for s in self.sils), 0.0, self.tsub0]
        return np.array([*head, *self.st, *self.s.reshape(-1)])


@dataclass(frozen=True)
class TrmemStressBatch:
    """Phase-I output of ``n`` membrane elements with ``k`` grids each."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, k)`` SILs in connection order."""
    tsub0: FloatArray
    st: FloatArray
    """``(n, 3)``."""
    s: FloatArray
    """``(n, k, 3, 3)``."""

    def __len__(self) -> int:
        return int(self.eid.shape[0])

    def element(self, i: int) -> TrmemStressMatrices:
        return TrmemStressMatrices(
            int(self.eid[i]),
            tuple(int(v) for v in self.sil[i]),  # type: ignore[arg-type]
            float(self.tsub0[i]),
            self.st[i].copy(),
            self.s[i].copy(),
        )


def trmem_stress_matrices(
    elements: Sequence[ElementData],
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemStressBatch:
    """Vectorised ``STRME1`` (``NTYPE = 0``) for every CTRMEM in ``elements``."""
    cstm = {} if cstm is None else cstm
    if real_words:
        cstm = real_cstm(cstm)
    props = [trmem_properties(d, materials, real_words) for d in elements]
    n = len(elements)
    eid = np.array([d.eid for d in elements], np.int64)
    g = np.array([p.g for p in props], float).reshape(n, 3, 3)
    s = membrane_stress_matrices(
        np.array([[gp.xyz for gp in d.grids] for d in elements], float).reshape(n, 3, 3),
        np.array([[gp.cid for gp in d.grids] for d in elements], np.int64).reshape(n, 3),
        g,
        cstm,
        real_words,
        eid,
    )
    alpha = np.array([p.alpha for p in props], float).reshape(n, 3)
    return TrmemStressBatch(
        eid,
        np.array([d["sil"] for d in elements], np.int64).reshape(n, 3),
        np.array([p.tref for p in props], float),
        thermal_stress_vector(g, alpha, real_words),
        s,
    )


def strme1(
    data: ElementData,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> TrmemStressMatrices:
    """``STRME1`` for one CTRMEM: stress matrices, ``S_T`` and ``T_SUB_0``."""
    return trmem_stress_matrices([data], materials, cstm, real_words).element(0)


# ----------------------------------------------------------------------------
# Phase II
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MembraneStresses:
    """``STQME2`` stresses of ``n`` elements (element axes)."""

    eid: NDArray[np.int64]
    sigma_x: FloatArray
    sigma_y: FloatArray
    tau_xy: FloatArray
    angle: FloatArray
    """Principal angle in degrees."""
    major: FloatArray
    minor: FloatArray
    max_shear: FloatArray

    def __len__(self) -> int:
        return int(self.eid.shape[0])

    def stress_table(self) -> FloatArray:
        """``(n, 7)`` rows as the listing prints them (``normal_x`` ... ``max_shear``)."""
        return np.stack(
            [
                self.sigma_x,
                self.sigma_y,
                self.tau_xy,
                self.angle,
                self.major,
                self.minor,
                self.max_shear,
            ],
            axis=1,
        )

    def stress_words(self, i: int) -> FloatArray:
        """The 8 ``STRES`` words of element ``i`` (element id as its REAL bit pattern)."""
        return np.array([_int_word(self.eid[i]), *self.stress_table()[i]])


def _temperatures(temperature, eids: np.ndarray) -> np.ndarray:
    """Per-element load temperature; NaN (or ``None``) means no thermal correction."""
    if temperature is None:
        return np.full(eids.shape[0], np.nan)
    if isinstance(temperature, Mapping):
        return np.array([temperature.get(int(e), np.nan) for e in eids], float)
    return np.broadcast_to(np.asarray(temperature, float), eids.shape).astype(float)


def membrane_stresses(
    batch: TrmemStressBatch,
    u,
    temperature=None,
    real_words: bool = True,
) -> MembraneStresses:
    """``STQME2`` for every element of ``batch`` (3 grids: CTRMEM, 4: CQDMEM).

    ``u`` is the global displacement vector indexed by SIL (``u[sil - 1]`` is the
    first translation). ``temperature`` is the load temperature: ``None`` (no
    TEMPERATURE(LOAD), ``LDTEMP = -1``), one value, one per element, or a
    mapping from element id (elements missing from it, or NaN, get none).
    """
    dt = _dtype(real_words)
    n, k = batch.sil.shape
    uu = np.asarray(u, float).astype(dt)
    idx = batch.sil[:, :, None] - 1 + np.arange(3)
    ug = uu[idx].astype(np.float64)  # (n, k, 3)
    s64 = batch.s.astype(dt).astype(np.float64)
    stress = np.zeros((n, 3), dt)
    for gi in range(k):
        # SMMATS: double-precision products and sums, rounded to REAL.
        vec = np.zeros((n, 3))
        for m in range(3):
            vec = vec + s64[:, gi, :, m] * ug[:, gi, m, None]
        stress = stress + vec.astype(dt)
    temps = _temperatures(temperature, batch.eid)
    hot = ~np.isnan(temps)
    if hot.any():
        tem = temps[hot].astype(dt) - batch.tsub0[hot].astype(dt)
        stress[hot] = stress[hot] - batch.st[hot].astype(dt) * tem[:, None]
    sx, sy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    two = dt(2.0)
    temp = sx - sy
    half = temp / two
    rad = np.sqrt(half * half + txy * txy)
    centre = (sx + sy) / two
    delta = two * txy
    tiny = dt(1.0e-15)
    flat = (np.abs(delta) < tiny) & (np.abs(temp) < tiny)
    phi = _atan2f(delta, temp) if real_words else np.arctan2(delta, temp)
    angle = np.where(flat, dt(0.0), phi * dt(ANGLE_FACTOR)).astype(dt)
    return MembraneStresses(
        batch.eid.copy(),
        *(v.astype(float) for v in (sx, sy, txy, angle, centre + rad, centre - rad, rad)),
    )


def trmem_stresses(
    batch: TrmemStressBatch, u, temperature=None, real_words: bool = True
) -> MembraneStresses:
    """Vectorised ``STQME2`` (``NTYPE = 1``) for CTRMEM elements; see :func:`membrane_stresses`."""
    return membrane_stresses(batch, u, temperature, real_words)


@dataclass(frozen=True)
class TrmemStress:
    """``STQME2`` stresses of one CTRMEM."""

    eid: int
    sigma_x: float
    sigma_y: float
    tau_xy: float
    angle: float
    major: float
    minor: float
    max_shear: float

    def stress_words(self) -> FloatArray:
        """The 8 ``STRES`` words (element id as its REAL bit pattern)."""
        vals = (self.sigma_x, self.sigma_y, self.tau_xy, self.angle)
        rest = (self.major, self.minor, self.max_shear)
        return np.array([_int_word(self.eid), *vals, *rest])


def stqme2(
    phase1: TrmemStressMatrices,
    u,
    temperature: float | None = None,
    real_words: bool = True,
) -> TrmemStress:
    """``STQME2`` (``NTYPE = 1``) for one CTRMEM; ``temperature`` is the load temperature."""
    batch = TrmemStressBatch(
        np.array([phase1.eid], np.int64),
        np.array([phase1.sils], np.int64),
        np.array([phase1.tsub0]),
        phase1.st[None],
        phase1.s[None],
    )
    t = None if temperature is None else [temperature]
    r = membrane_stresses(batch, u, t, real_words)
    return TrmemStress(int(r.eid[0]), *(float(v) for v in r.stress_table()[0]))
