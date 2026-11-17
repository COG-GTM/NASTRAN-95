"""CSHEAR stress recovery: SDR2 phase I ``SPANL1(4)`` and phase II ``SPANL2(4)``.

``SPANL1`` (``mis/spanl1.f``) repeats the ``SHEARD`` panel geometry in REAL
arithmetic: the mean plane through the diagonals (``VKN``), the projected
corner coordinates, the warped-panel flexibility ``Z`` (labels 30/40/50/70:
edge 3-4 or 1-4 parallel, parallelogram, general quadrilateral, the last with
``ALOG`` terms), the effective-extension areas ``AVEC`` and the stress matrices

    S_i = -AVEC(i) / (2 Z T) * (u_i VI + v_i VJ) [T_i]

that turn each corner displacement into the reference shear stress ``S1BAR``.
It also keeps the corner-stress ratios ``P2/P1, P1P2/P3^2, P1P2/P4^2``, the
allowable ``SIGS``, and the edge/kick-load geometry ``RQ(4), RK(4)``.

``SPANL2`` (``mis/spanl2.f``) sums ``S_i u_i`` (``SMMATS``, double accumulation
rounded to REAL), forms the four corner stresses, the average and the maximum
absolute shear stress, the margin of safety ``SIGS/TAUMAX - 1`` (an integer
word ``1``, printed blank, when ``SIGS <= 0`` or ``TAUMAX = 0``), the edge shear
flows ``Q1..Q4``, the corner forces ``F1..F8`` and the kick loads ``RK1..RK4``.

With ``real_words=True`` (default) every step is evaluated in float32 in the
Fortran expression order, the CSTM words are rounded to REAL and ``TRANSS`` is
evaluated in REAL. ``real_words=False`` evaluates the same expressions in
float64. :func:`spanl1`/:func:`spanl2` are the per-element reference paths;
:func:`shear_stress_phase1`/:func:`shear_stress_phase2` the batched paths.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nastran95py.coords import CoordSystem
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import real_cstm
from nastran95py.elements.shear import ShearGeometryError
from nastran95py.elements.trmem_stress import transs
from nastran95py.materials import MaterialTable

__all__ = [
    "FORCE_COLUMNS",
    "STRESS_COLUMNS",
    "ShearStressBatch",
    "ShearStressForces",
    "ShearStressPhase1",
    "ShearStressProperties",
    "shear_stress_phase1",
    "shear_stress_phase2",
    "shear_stress_properties",
    "spanl1",
    "spanl2",
]

FloatArray = NDArray[np.float64]

STRESS_COLUMNS = ("max_shear", "avg_shear", "margin")
FORCE_COLUMNS = (
    "f1_from4",
    "f1_from2",
    "f2_from1",
    "f2_from3",
    "f3_from2",
    "f3_from4",
    "f4_from3",
    "f4_from1",
    "kick1",
    "shear12",
    "kick2",
    "shear23",
    "kick3",
    "shear34",
    "kick4",
    "shear41",
)


def _int_word(i: int) -> float:
    """The REAL reading of an integer word (as the harness prints ``PH1OUT``/``STRES``)."""
    return struct.unpack("<f", struct.pack("<i", int(i)))[0]


def _dtype(real_words: bool):
    return np.float32 if real_words else np.float64


@dataclass(frozen=True)
class ShearStressProperties:
    """``MAT`` (``MATFLG = 1``) values ``SPANL1`` uses, plus the panel thickness."""

    g: float
    nu: float
    sigs: float
    t: float


def shear_stress_properties(data: ElementData, materials: MaterialTable) -> ShearStressProperties:
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    return ShearStressProperties(mat.g, mat.nu, mat.sigs, float(data["t"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ShearStressBatch:
    """Column arrays of ``n`` CSHEAR elements (EST order, basic coordinates)."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]  # (n, 4)
    cid: NDArray[np.int64]  # (n, 4)
    xyz: FloatArray  # (n, 4, 3)
    g: FloatArray
    nu: FloatArray
    sigs: FloatArray
    t: FloatArray

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[ElementData],
        materials: MaterialTable | Sequence[ShearStressProperties],
    ) -> ShearStressBatch:
        if isinstance(materials, MaterialTable):
            props = [shear_stress_properties(d, materials) for d in elements]
        else:
            props = list(materials)
        col = {
            name: np.array([getattr(p, name) for p in props], dtype=float)
            for name in ("g", "nu", "sigs", "t")
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
class ShearStressPhase1:
    """``SPANL1`` output (``PH1OUT``) for ``n`` elements.

    ``s`` is ``(n, 4, 3)`` (corner, global dof); ``out`` is ``(n, 15)``:
    ``AVEC(1:2), T, RATIO(3), SIGS, RQ(4), RK(4)``.
    """

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    s: FloatArray
    out: FloatArray

    def words(self, i: int = 0) -> list[float]:
        """The 33 ``PH1OUT`` words of element ``i`` (the last one is unused, zero)."""
        ints = [_int_word(self.eid[i]), *(_int_word(v) for v in self.sil[i])]
        s = [float(v) for v in self.s[i].ravel()]
        return [*ints, *s, *(float(v) for v in self.out[i]), 0.0]

    def __len__(self) -> int:
        return int(self.eid.shape[0])


@dataclass(frozen=True)
class ShearStressForces:
    """``SPANL2`` output: ``stress (n, 3)`` (``max_shear, avg_shear, margin``; a
    blank margin is NaN) and ``forces (n, 16)`` in :data:`FORCE_COLUMNS` order."""

    eid: NDArray[np.int64]
    stress: FloatArray
    forces: FloatArray

    def stress_words(self, i: int = 0) -> list[float]:
        m = self.stress[i, 2]
        margin = _int_word(1) if np.isnan(m) else float(m)
        return [_int_word(self.eid[i]), float(self.stress[i, 0]), float(self.stress[i, 1]), margin]

    def force_words(self, i: int = 0) -> list[float]:
        return [_int_word(self.eid[i]), *(float(v) for v in self.forces[i])]


# ----------------------------------------------------------------------------
# Phase I
# ----------------------------------------------------------------------------


def _mm(a: np.ndarray, b: np.ndarray, dt) -> np.ndarray:
    """``GMMATS`` batched ``(n, p, q) @ (n, q, r)``: accumulated term by term in ``dt``."""
    out = np.zeros((a.shape[0], a.shape[1], b.shape[2]), dt)
    for k in range(a.shape[2]):
        out = out + a[:, :, k, None] * b[:, None, k, :]
    return out


def _dot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _cross(a, b):
    return np.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        axis=1,
    )


def _flexibility(x2, x3, y3, x4, y4, pa, g, t, nu, dt):
    """``Z`` and corner distances ``P(4)`` (labels 30/40/50/70 of ``SPANL1``)."""
    n = x2.shape[0]
    one, two, half = dt(1.0), dt(2.0), dt(0.5)
    zero = np.zeros(n, dt)
    xs, ys = (zero, x2, x3, x4), (zero, zero, y3, y4)
    p = np.ones((n, 4), dt)
    z = np.zeros(n, dt)
    temp = x3 - x2
    ep = dt(0.01)
    c34 = np.abs(y3 - y4) < np.abs(x3 - x4) * ep
    c14 = np.abs(y4 * temp - y3 * x4) < np.abs(x4 * temp + y4 * y3) * ep
    b30, b40, b50 = c34 & ~c14, ~c34 & c14, c34 & c14
    b70 = ~(c34 | c14)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # 30: edge 3-4 parallel to edge 1-2.
        tm = y3 * x4 - y4 * (x3 - x2)
        yp = x2 * y3 * y4 / tm
        p30 = np.stack([yp - ys[i] for i in range(4)], axis=1)
        xp = x2 * y3 * x4 / tm
        sa = (x2 - xp) / yp
        c = (zero - xp) / yp
        z30 = (p30[:, 0] * p30[:, 1] * pa / (p30[:, 2] * p30[:, 3] * two * g * t)) * (
            one + two / (dt(3.0) + dt(3.0) * nu) * (sa * sa + sa * c + c * c)
        )
        # 40: edge 1-4 parallel to edge 2-3.
        d = -half * (x4 / y4 + (x3 - x2) / y3)
        xq = x4 - y4 * (x3 - x4) / (y3 - y4)
        tm = one / np.sqrt(one + d * d)
        p40 = np.stack([(xq - xs[i] - d * ys[i]) * tm for i in range(4)], axis=1)
        tm = xq - x4
        b = (tm * d + y4) / (tm - y4 * d)
        z40 = (p40[:, 0] * p40[:, 1] * pa / (p40[:, 2] * p40[:, 3] * two * g * t)) * (
            one + two / (dt(3.0) + dt(3.0) * nu) * (b * b + b * d + d * d)
        )
        # 50: parallelogram.
        d = -half * (x4 / y4 + (x3 - x2) / y3 + (y3 - y4) / (x3 - x4))
        z50 = pa / (two * g * t) * (one + two * (d * d) / (one + nu))
        # 70: general quadrilateral.
        xq = x4 - (x3 - x4) / (y3 - y4) * y4
        tm = y3 * x4 - y4 * (x3 - x2)
        xp = x2 * y3 * x4 / tm
        yp = x2 * y3 * y4 / tm
        xl = np.sqrt((xq - xp) * (xq - xp) + yp * yp)
        d = (xq - xp) / yp
        tm = yp / xl
        p70 = np.stack([tm * (xq - xs[i] - d * ys[i]) for i in range(4)], axis=1)
        p1, p2, p3, p4 = (p70[:, i] for i in range(4))
        c = xl / p1 - d
        b = xl / p4 - c
        a = xl / p2 - d
        a2 = a * a
        b2 = b * b
        c2 = c * c
        d2 = d * d
        a3 = a2 * a
        b3 = b2 * b
        c3 = c2 * c
        d3 = d2 * d
        a4 = a3 * a
        b4 = b3 * b
        c4 = c3 * c
        d4 = d3 * d
        a5 = a4 * a
        b5 = b4 * b
        c5 = c4 * c
        d5 = d4 * d
        tm = half * p1 * p2 * p3 * p4 / (xl * xl)
        three, fifth, tenth = dt(3.0), dt(0.2), dt(0.1)
        term = a + b + two * (a3 + b3) / three + fifth * (a5 + b5)
        term1 = c + d + two * (c3 + d3) / three + fifth * (c5 + d5)
        term2 = b + c + two * (b3 + c3) / three + fifth * (b5 + c5)
        term3 = d + a + two * (d3 + a3) / three + fifth * (d5 + a5)
        term = term * np.log(np.abs(a + b))
        term1 = term1 * np.log(np.abs(c + d))
        term2 = term2 * np.log(np.abs(b + c))
        term3 = term3 * np.log(np.abs(d + a))
        term4 = tenth * ((a2 - c2) * (b3 - d3) + (b2 - d2) * (a3 - c3))
        term5 = fifth * ((a - c) * (b4 - d4) + (b - d) * (a4 - c4))
        f = tm * (term + term1 - term2 - term3 + term4 - term5)
        z70 = (
            p1 * p2 / (p3 * p4 * two * g * t) * (pa + dt(4.0) / (one + nu) * (f - two * pa / three))
        )
    for mask, zz, pp in ((b30, z30, p30), (b40, z40, p40), (b50, z50, None), (b70, z70, p70)):
        z[mask] = zz[mask]
        if pp is not None:
            p[mask] = pp[mask]
    return z, p


def shear_stress_phase1(
    batch: ShearStressBatch,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> ShearStressPhase1:
    """Batched ``SPANL1(4)``. Raises :class:`ShearGeometryError` (2026/2027)."""
    dt = _dtype(real_words)
    n = len(batch)
    one, two, half = dt(1.0), dt(2.0), dt(0.5)
    gp = batch.xyz.astype(dt)
    g, nu, t, sigs = (np.asarray(v).astype(dt) for v in (batch.g, batch.nu, batch.t, batch.sigs))

    def _fail(mask, code, corner=None):
        if mask.any():
            raise ShearGeometryError(int(batch.eid[np.argmax(mask)]), code, corner)

    with np.errstate(divide="ignore", invalid="ignore"):
        vd1 = gp[:, 2] - gp[:, 0]
        vd2 = gp[:, 3] - gp[:, 1]
        vkn = _cross(vd1, vd2)
        vkl = np.sqrt(vkn[:, 0] * vkn[:, 0] + vkn[:, 1] * vkn[:, 1] + vkn[:, 2] * vkn[:, 2])
        _fail(vkl == 0.0, 2026)
        vk = vkn / vkl[:, None]
        pa = half * vkl
        v12 = gp[:, 1] - gp[:, 0]
        v41 = gp[:, 0] - gp[:, 3]
        v12dk = _dot(v12, vk)
        vp12 = v12 - v12dk[:, None] * vk
        vp12l = np.sqrt(_dot(vp12, vp12))
        _fail(vp12l == 0.0, 2026)
        vi = vp12 / vp12l[:, None]
        vj = _cross(vk, vi)
        vjl = np.sqrt(_dot(vj, vj))
        _fail(vjl == 0.0, 2026)
        vj = vj / vjl[:, None]
        x2 = vp12l
        x3 = _dot(vi, vd1)
        y3 = _dot(vj, vd1)
        x4 = -vi[:, 0] * v41[:, 0] - vi[:, 1] * v41[:, 1] - vi[:, 2] * v41[:, 2]
        y4 = -vj[:, 0] * v41[:, 0] - vj[:, 1] * v41[:, 1] - vj[:, 2] * v41[:, 2]
        _fail(y3 <= 0.0, 2027, 2)
        _fail(x3 <= y3 * x4 / y4, 2027, 4)
        _fail(y4 <= 0.0, 2027, 1)
        _fail(x4 >= x2 - (x2 - x3) * y4 / y3, 2027, 3)

        z, p = _flexibility(x2, x3, y3, x4, y4, pa, g, t, nu, dt)

        xl13 = np.sqrt(x3 * x3 + y3 * y3)
        xl24 = np.sqrt((x4 - x2) * (x4 - x2) + y4 * y4)
        su = np.stack([x3 / xl13, (x4 - x2) / xl24], axis=1)
        sv = np.stack([y3 / xl13, y4 / xl24], axis=1)
        su = np.concatenate([su, su], axis=1)
        sv = np.concatenate([sv, sv], axis=1)
        tm = x4 * y3 - x3 * y4
        avec1 = -(half * x2 * y4 * xl13 / tm)
        avec2 = half * x2 * y3 * xl24 / (tm - x2 * (y3 - y4))
        avec = np.stack([avec1, avec2, -avec1, -avec2], axis=1)
        spcon = -(one / (two * z * t))

        vleft = su[:, :, None] * vi[:, None, :] + sv[:, :, None] * vj[:, None, :]  # (n, 4, 3)
        if np.any(batch.cid != 0):
            if cstm is None:
                raise KeyError("CSHEAR grids reference a coordinate system but no CSTM was given")
            cs_map = real_cstm(cstm) if real_words else cstm
            for cid in np.unique(batch.cid[batch.cid != 0]):
                if int(cid) not in cs_map:
                    raise KeyError(f"coordinate system {int(cid)} is not in the CSTM")
                e_i, c_i = np.nonzero(batch.cid == cid)
                ti = transs(cs_map[int(cid)], gp[e_i, c_i], real_words).astype(dt)
                vleft[e_i, c_i] = _mm(vleft[e_i, c_i][:, None, :], ti, dt)[:, 0, :]
        s = spcon[:, None, None] * vleft * avec[:, :, None]

        p1, p2, p3, p4 = (p[:, i] for i in range(4))
        out = np.stack(
            [
                avec1,
                avec2,
                t,
                p2 / p1,
                p1 * p2 / (p3 * p3),
                p1 * p2 / (p4 * p4),
                sigs,
                p1 / p3 * t,
                p1 * p2 / (p3 * p4) * t,
                p2 / p4 * t,
                -v12dk / two,
                x2 / two,
                np.sqrt((x3 - x2) * (x3 - x2) + y3 * y3) / two,
                np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)) / two,
                np.sqrt(x4 * x4 + y4 * y4) / two,
            ],
            axis=1,
        )
    assert out.shape == (n, 15)
    return ShearStressPhase1(batch.eid.copy(), batch.sil.copy(), s.astype(float), out.astype(float))


def spanl1(
    data: ElementData,
    materials: MaterialTable | ShearStressProperties,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> ShearStressPhase1:
    """``SPANL1(4)`` for one CSHEAR element (EST layout of ``mis/spanl1.f``)."""
    props = materials if isinstance(materials, ShearStressProperties) else None
    batch = ShearStressBatch.from_elements([data], [props] if props else materials)  # type: ignore[arg-type]
    return shear_stress_phase1(batch, cstm, real_words)


# ----------------------------------------------------------------------------
# Phase II
# ----------------------------------------------------------------------------


def shear_stress_phase2(
    phase1: ShearStressPhase1, u: ArrayLike, real_words: bool = True
) -> ShearStressForces:
    """Batched ``SPANL2(4)``: ``u`` is the SIL-ordered global displacement vector."""
    dt = _dtype(real_words)
    u = np.asarray(u, dtype=float)
    if real_words:
        u = u.astype(np.float32).astype(float)
    s = phase1.s.astype(dt).astype(float)
    out = phase1.out.astype(dt)
    idx = phase1.sil[:, :, None] - 1 + np.arange(3)
    ug = u[idx]  # (n, 4, 3)
    n = len(phase1)
    s1bar = np.zeros(n, dt)
    for i in range(4):
        acc = np.zeros(n)
        for k in range(3):
            acc = acc + s[:, i, k] * ug[:, i, k]
        s1bar = s1bar + acc.astype(dt)
    t, r1, r2, r3, sigs = out[:, 2], out[:, 3], out[:, 4], out[:, 5], out[:, 6]
    rq = out[:, 7:11]
    rk = out[:, 11:15]
    one = dt(1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.stack([r1 * s1bar, s1bar / r1, r2 * s1bar, r3 * s1bar], axis=1)
        tauavg = dt(0.25) * (tau[:, 0] + tau[:, 1] + tau[:, 2] + tau[:, 3])
        taumax = np.abs(tau).max(axis=1)
        ok = (sigs > 0.0) & (taumax != 0.0)
        margin = np.where(ok, sigs / taumax - one, np.nan)
        rq4 = rq[:, 3]
        q1 = s1bar * t / np.sqrt(one + (rq4 / rk[:, 0]) * (rq4 / rk[:, 0]))
        q2 = s1bar * rq[:, 0] / np.sqrt(one + (rq4 / rk[:, 1]) * (rq4 / rk[:, 1]))
        q3 = s1bar * rq[:, 1] / np.sqrt(one + (rq4 / rk[:, 2]) * (rq4 / rk[:, 2]))
        q4 = s1bar * rq[:, 2] / np.sqrt(one + (rq4 / rk[:, 3]) * (rq4 / rk[:, 3]))
        k1 = -(q1 + q4) * rq4
        k2 = -(q1 + q2) * rq4
        k3 = -(q2 + q3) * rq4
        k4 = -(q3 + q4) * rq4
        f1 = q4 * rk[:, 3]
        f2 = q1 * rk[:, 0]
        f5 = q2 * rk[:, 1]
        f6 = q3 * rk[:, 2]
    forces = np.stack([f1, f2, -f2, -f5, f5, f6, -f6, -f1, k1, q1, k2, q2, k3, q3, k4, q4], axis=1)
    stress = np.stack([taumax.astype(float), tauavg.astype(float), margin.astype(float)], axis=1)
    return ShearStressForces(phase1.eid.copy(), stress, forces.astype(float))


def spanl2(phase1: ShearStressPhase1, u: ArrayLike, real_words: bool = True) -> ShearStressForces:
    """``SPANL2(4)`` (alias of the batched path; one row per phase-I element)."""
    return shear_stress_phase2(phase1, u, real_words)
