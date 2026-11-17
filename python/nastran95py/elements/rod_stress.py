"""Rod stress recovery (CROD, CONROD): port of ``SROD1``/``SROD2`` (``mis/srod1.f``, ``srod2.f``).

SDR2 recovers rod stresses in two phases:

- phase I (``SROD1``) turns the ECPT record into stress matrices. With
  ``n = (xA - xB)/L`` (the Fortran's ``XN``, pointing from B to A),
  ``SAT = TA^T n E/L``, ``SBT = -TB^T n E/L`` (axial, per unit global
  translation) and ``SAR = TA^T n G J/L / L``... precisely ``SAR = TA^T n G*J/L``,
  ``SBR = -TB^T n G*J/L`` (torque per unit global rotation), where ``TA``/``TB``
  are the ``TRANSS`` global-to-basic matrices of the two grids. It also keeps
  ``ST = -ALPHA*E``, ``SDELTA = -E/L``, the area, ``C/J`` (``FJOVRC``), ``TREF``
  and the MAT1 limits ``ST, SC, SS``, all from ``MAT`` at the ECPT element
  temperature;
- phase II (``SROD2``) applies the global displacements ``u`` (SIL-indexed)::

    SIGMA  = SAT.uA + SBT.uB + SDELTA*ELDEFM [+ ST*(TEMPLD - TREF)]
    TORQUE = SAR.thetaA + SBR.thetaB
    P      = A*SIGMA                TAU = (C/J)*TORQUE

  with the thermal term only when the subcase has a load temperature set, and
  ``ELDEFM`` the element's enforced deformation (``DEFORM``). Margins of safety:
  ``ST/SIGMA - 1`` in tension, ``-SC/SIGMA - 1`` in compression and
  ``SS/|TAU| - 1`` in torsion; the Fortran stores the integer 1 instead (printed
  blank) when the limit is not positive or the stress is zero. Here a blank
  margin is ``None`` (reference path) or NaN (batched path).

The ``NCHK`` precision check of ``SROD2`` (a diagnostic print) is not ported.

``SROD1``/``SROD2`` work in REAL (single precision); only ``SMMATS`` accumulates
its dot products in double precision. With ``real_words=True`` (default) every
step is evaluated in float32 in the Fortran order, including ``TRANSS`` and the
REAL CSTM words it reads, so the results reproduce the gfortran build to the
last bit. ``real_words=False`` evaluates the same expressions in float64.

Two paths are given: :func:`srod1`/:func:`srod2`, a per-element reference that
follows the Fortran statement by statement, and :func:`rod_stress_matrices`/
:func:`rod_stresses`, the vectorised kernel over many elements.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nastran95py.coords import CoordSystem
from nastran95py.elements.contract import ElementData
from nastran95py.elements.rod import RodGeometryError
from nastran95py.materials import MaterialTable

FloatArray = NDArray[np.float64]

NO_MARGIN_WORD = struct.unpack("<f", struct.pack("<i", 1))[0]
"""The REAL reading of the integer 1 ``SROD2`` stores for a blank margin."""


def _int_word(i: int) -> float:
    return struct.unpack("<f", struct.pack("<i", int(i)))[0]


@dataclass(frozen=True)
class RodStressMatrices:
    """``SROD1`` output: the 23 phase-I words (``/SDR2X5/`` words 101-123)."""

    eid: int
    sil: tuple[int, int]
    """SILs of grids A and B, in connection order."""
    sat: FloatArray
    sbt: FloatArray
    sar: FloatArray
    sbr: FloatArray
    st: float
    """``-ALPHA*E``: thermal stress per degree."""
    sdelta: float
    """``-E/L``: stress per unit enforced deformation."""
    area: float
    c_over_j: float
    """``C/J`` (``FJOVRC``; 0 when ``J = 0``)."""
    tref: float
    sigma_t: float
    sigma_c: float
    sigma_s: float

    def words(self) -> FloatArray:
        """The phase-I block as REAL readings (EID and SILs as their bit patterns)."""
        return np.array(
            [
                _int_word(self.eid),
                _int_word(self.sil[0]),
                _int_word(self.sil[1]),
                *self.sat,
                *self.sbt,
                *self.sar,
                *self.sbr,
                self.st,
                self.sdelta,
                self.area,
                self.c_over_j,
                self.tref,
                self.sigma_t,
                self.sigma_c,
                self.sigma_s,
            ]
        )


@dataclass(frozen=True)
class RodStress:
    """``SROD2`` output for one element (stress block and force block)."""

    eid: int
    sigma: float
    """Axial stress (tension positive)."""
    axial_margin: float | None
    tau: float
    """Torsional stress ``(C/J)*TORQUE``."""
    torsional_margin: float | None
    force: float
    """Axial force ``A*SIGMA``."""
    torque: float

    def stress_words(self) -> FloatArray:
        """``JSELID, SIGMA, SMSIG, TAU, SMTAU`` as REAL readings."""
        return np.array(
            [
                _int_word(self.eid),
                self.sigma,
                NO_MARGIN_WORD if self.axial_margin is None else self.axial_margin,
                self.tau,
                NO_MARGIN_WORD if self.torsional_margin is None else self.torsional_margin,
            ]
        )

    def force_words(self) -> FloatArray:
        """``JFELID, P, TORQUE`` as REAL readings."""
        return np.array([_int_word(self.eid), self.force, self.torque])


def _system(cstm: Mapping[int, CoordSystem] | None, cid: int) -> CoordSystem:
    if cstm is None or cid not in cstm:
        raise KeyError(f"coordinate system {cid} is not in the CSTM")
    return cstm[cid]


def _transs(cs: CoordSystem, pts: np.ndarray, dt) -> np.ndarray:
    """``TRANSS`` for ``(m, 3)`` basic points of one system: ``(m, 3, 3)`` ``TA`` in ``dt``.

    Follows ``mis/pretrs.f`` operation by operation (``GMMATS`` accumulates in
    the working precision), reading the CSTM origin and rotation as ``dt`` words.
    """
    ke = cs.t.astype(dt)
    m = pts.shape[0]
    ta = np.broadcast_to(ke, (m, 3, 3)).copy()
    if int(cs.ctype) == 1:
        return ta
    ro = cs.origin.astype(dt)
    xn = pts.astype(dt) - ro
    x = ke[0, 0] * xn[:, 0] + ke[1, 0] * xn[:, 1] + ke[2, 0] * xn[:, 2]
    y = ke[0, 1] * xn[:, 0] + ke[1, 1] * xn[:, 1] + ke[2, 1] * xn[:, 2]
    z = ke[0, 2] * xn[:, 0] + ke[1, 2] * xn[:, 1] + ke[2, 2] * xn[:, 2]
    r = np.sqrt(x * x + y * y)
    k = r != 0.0
    if not k.any():
        return ta
    x, y, z, r = x[k], y[k], z[k], r[k]
    tl = np.zeros((x.shape[0], 3, 3), dtype=dt)
    if int(cs.ctype) == 2:
        tl[:, 0, 0] = x / r
        tl[:, 0, 1] = -y / r
        tl[:, 1, 0] = -tl[:, 0, 1]
        tl[:, 1, 1] = tl[:, 0, 0]
        tl[:, 2, 2] = 1.0
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
    out = np.zeros_like(tl)
    for i in range(3):
        for j in range(3):
            c = np.zeros(tl.shape[0], dtype=dt)
            for kk in range(3):
                c = c + ke[i, kk] * tl[:, kk, j]
            out[:, i, j] = c
    ta[k] = out
    return ta


def _to_global(xn: np.ndarray, ta: np.ndarray, dt) -> np.ndarray:
    """``GMMATS (XN,3,1,1, TA,3,3,0, XN(4))``: ``TA^T XN`` accumulated in ``dt``."""
    out = np.empty_like(xn)
    for j in range(3):
        c = np.zeros(xn.shape[0], dtype=dt)
        for kk in range(3):
            c = c + xn[:, kk] * ta[:, kk, j]
        out[:, j] = c
    return out


def _dot(a: np.ndarray, u: np.ndarray, dt) -> np.ndarray:
    """``SMMATS (A,3,1,1, U,3,1,0, C, E)``: double-precision dot product, rounded to ``dt``."""
    a64, u64 = a.astype(dt).astype(np.float64), u.astype(dt).astype(np.float64)
    c = np.zeros(a.shape[0])
    for kk in range(3):
        c = c + a64[:, kk] * u64[:, kk]
    return c.astype(dt)


@dataclass(frozen=True)
class RodStressBatch:
    """Phase-I stress matrices of ``n`` rods as stacked arrays."""

    eid: NDArray[np.int64]
    sil: NDArray[np.int64]
    """``(n, 2)`` SILs in connection order."""
    sat: FloatArray
    sbt: FloatArray
    sar: FloatArray
    sbr: FloatArray
    """``(n, 3)`` each."""
    st: FloatArray
    sdelta: FloatArray
    area: FloatArray
    c_over_j: FloatArray
    tref: FloatArray
    sigma_t: FloatArray
    sigma_c: FloatArray
    sigma_s: FloatArray

    def __len__(self) -> int:
        return int(self.eid.shape[0])

    def element(self, i: int) -> RodStressMatrices:
        return RodStressMatrices(
            int(self.eid[i]),
            (int(self.sil[i, 0]), int(self.sil[i, 1])),
            self.sat[i].copy(),
            self.sbt[i].copy(),
            self.sar[i].copy(),
            self.sbr[i].copy(),
            *(
                float(getattr(self, f)[i])
                for f in (
                    "st",
                    "sdelta",
                    "area",
                    "c_over_j",
                    "tref",
                    "sigma_t",
                    "sigma_c",
                    "sigma_s",
                )
            ),
        )


def rod_stress_matrices(
    elements: Sequence[ElementData],
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> RodStressBatch:
    """Vectorised ``SROD1`` over ROD records (CROD and CONROD)."""
    dt = np.float32 if real_words else np.float64
    n = len(elements)
    mats = [materials.mat1(int(d["mid"]), d.temperature) for d in elements]  # type: ignore[arg-type]

    def col(values) -> np.ndarray:
        return np.array(list(values), dtype=float).astype(dt)

    e = col(m.e for m in mats)
    g = col(m.g for m in mats)
    alpha = col(m.alpha for m in mats)
    area = col(d["area"] for d in elements)
    jt = col(d["j"] for d in elements)
    c = col(d["c"] for d in elements)
    sil = np.array([d["sil"] for d in elements], dtype=np.int64).reshape(n, 2)
    cid = np.array([[gp.cid for gp in d.grids] for d in elements], dtype=np.int64).reshape(n, 2)
    xyz = np.array([[gp.xyz for gp in d.grids] for d in elements], dtype=float).reshape(n, 2, 3)
    xyz = xyz.astype(dt)

    xn = xyz[:, 0] - xyz[:, 1]
    xl = np.sqrt(xn[:, 0] * xn[:, 0] + xn[:, 1] * xn[:, 1] + xn[:, 2] * xn[:, 2])
    bad = ~(xl > 0.0) | (sil[:, 0] == sil[:, 1])
    if bad.any():
        raise RodGeometryError(
            f"rod elements {np.asarray([d.eid for d in elements])[bad].tolist()} "
            "have illegal geometry or connections"
        )
    xn = xn / xl[:, None]
    eoverl = e / xl
    gcovrl = g * jt / xl

    dirs = []
    for side in (0, 1):
        v = xn.copy()
        cids = cid[:, side]
        for c_id in np.unique(cids[cids != 0]):
            sel = cids == c_id
            ta = _transs(_system(cstm, int(c_id)), xyz[sel, side], dt)
            v[sel] = _to_global(xn[sel], ta, dt)
        dirs.append(v)
    (va, vb) = dirs
    with np.errstate(divide="ignore", invalid="ignore"):
        c_over_j = np.where(jt != 0.0, c / np.where(jt != 0.0, jt, dt(1.0)), dt(0.0)).astype(dt)

    def out(values) -> FloatArray:
        return np.asarray(values, dtype=dt).astype(float)

    return RodStressBatch(
        eid=np.array([d.eid for d in elements], dtype=np.int64),
        sil=sil,
        sat=out(va * eoverl[:, None]),
        sbt=out(-(vb * eoverl[:, None])),
        sar=out(va * gcovrl[:, None]),
        sbr=out(-(vb * gcovrl[:, None])),
        st=out(-(alpha * e)),
        sdelta=out(-eoverl),
        area=out(area),
        c_over_j=out(c_over_j),
        tref=out(col(m.tref for m in mats)),
        sigma_t=out(col(m.sigt for m in mats)),
        sigma_c=out(col(m.sigc for m in mats)),
        sigma_s=out(col(m.sigs for m in mats)),
    )


def srod1(
    data: ElementData,
    materials: MaterialTable,
    cstm: Mapping[int, CoordSystem] | None = None,
    real_words: bool = True,
) -> RodStressMatrices:
    """Per-element reference port of ``SROD1`` (phase I) for one ROD record."""
    dt = np.float32 if real_words else np.float64
    mat = materials.mat1(int(data["mid"]), data.temperature)  # type: ignore[arg-type]
    e, g, alpha = dt(mat.e), dt(mat.g), dt(mat.alpha)
    area, jt, c = (dt(float(data[k])) for k in ("area", "j", "c"))  # type: ignore[arg-type]
    sil = tuple(int(s) for s in data["sil"])  # type: ignore[union-attr]
    ga, gb = data.grids
    xa, xb = ga.xyz.astype(dt), gb.xyz.astype(dt)

    # Each REAL operation is rounded explicitly so the result does not depend on
    # NumPy's scalar promotion rules (NumPy 1.x promotes some float32 scalar ops).
    xn = [dt(xa[i] - xb[i]) for i in range(3)]
    xl = dt(dt(dt(xn[0] * xn[0]) + dt(xn[1] * xn[1])) + dt(xn[2] * xn[2]))
    xl = dt(np.sqrt(xl))
    if not xl > 0.0 or sil[0] == sil[1]:
        raise RodGeometryError(f"rod element {data.eid} has illegal geometry or connections")
    xn = [dt(x / xl) for x in xn]
    eoverl = dt(e / xl)
    gcovrl = dt(dt(g * jt) / xl)

    def global_dir(gp, xyz) -> list:
        if gp.cid == 0:
            return xn
        ti = _transs(_system(cstm, gp.cid), xyz[None, :], dt)[0]
        out = []
        for j in range(3):
            s = dt(0.0)
            for k in range(3):
                s = dt(s + dt(xn[k] * ti[k, j]))
            out.append(s)
        return out

    na, nb = global_dir(ga, xa), global_dir(gb, xb)
    f64 = lambda vals: np.array(vals, dtype=dt).astype(float)  # noqa: E731
    return RodStressMatrices(
        eid=data.eid,
        sil=sil,  # type: ignore[arg-type]
        sat=f64([dt(v * eoverl) for v in na]),
        sbt=f64([-dt(v * eoverl) for v in nb]),
        sar=f64([dt(v * gcovrl) for v in na]),
        sbr=f64([-dt(v * gcovrl) for v in nb]),
        st=float(-dt(alpha * e)),
        sdelta=float(-eoverl),
        area=float(area),
        c_over_j=float(dt(c / jt)) if jt != 0.0 else 0.0,
        tref=float(dt(mat.tref)),
        sigma_t=float(dt(mat.sigt)),
        sigma_c=float(dt(mat.sigc)),
        sigma_s=float(dt(mat.sigs)),
    )


def _margins(sigma, tau, st, sc, ss, dt) -> tuple[float | None, float | None]:
    one = dt(1.0)
    ms_sigma = None
    if sigma > 0.0:
        if st > 0.0:
            ms_sigma = float(dt(dt(st / sigma) - one))
    elif sigma != 0.0 and sc > 0.0:
        ms_sigma = float(dt(dt(-sc / sigma) - one))
    ms_tau = None
    if ss > 0.0 and tau != 0.0:
        ms_tau = float(dt(dt(ss / abs(tau)) - one))
    return ms_sigma, ms_tau


def srod2(
    phase1: RodStressMatrices,
    displacements: Sequence[float] | NDArray[np.float64],
    temperature: float | None = None,
    deformation: float = 0.0,
    real_words: bool = True,
) -> RodStress:
    """Per-element reference port of ``SROD2`` (phase II).

    ``displacements`` is the SIL-ordered global displacement vector (SIL ``s``
    at index ``s - 1``); ``temperature`` the subcase load temperature of the
    element (``TEMPLD``; ``None`` when the subcase has no thermal load) and
    ``deformation`` its enforced deformation (``ELDEFM``).
    """
    dt = np.float32 if real_words else np.float64
    u = np.asarray(displacements, dtype=float)
    p = phase1
    ia, ib = p.sil[0] - 1, p.sil[1] - 1

    def dot(a, k):
        return _dot(np.asarray(a)[None, :], u[k : k + 3][None, :], dt)[0]

    sat, sbt, sar, sbr = (
        np.asarray(v, dtype=float).astype(dt) for v in (p.sat, p.sbt, p.sar, p.sbr)
    )
    trana, tranb = dot(sat, ia), dot(sbt, ib)
    sigma = dt(dt(trana + tranb) + dt(dt(p.sdelta) * dt(deformation)))
    if temperature is not None:
        sigma = dt(sigma + dt(dt(p.st) * dt(dt(temperature) - dt(p.tref))))
    rota, rotb = dot(sar, ia + 3), dot(sbr, ib + 3)
    torque = dt(rota + rotb)
    force = dt(dt(p.area) * sigma)
    tau = dt(dt(p.c_over_j) * torque)
    ms_sigma, ms_tau = _margins(sigma, tau, dt(p.sigma_t), dt(p.sigma_c), dt(p.sigma_s), dt)
    return RodStress(p.eid, float(sigma), ms_sigma, float(tau), ms_tau, float(force), float(torque))


@dataclass(frozen=True)
class RodStressResults:
    """Vectorised ``SROD2`` output; blank margins are NaN."""

    eid: NDArray[np.int64]
    sigma: FloatArray
    axial_margin: FloatArray
    tau: FloatArray
    torsional_margin: FloatArray
    force: FloatArray
    torque: FloatArray

    def stress_table(self) -> FloatArray:
        """``(n, 4)`` rows ``axial, axial_margin, torsional, torsional_margin`` (stress print)."""
        return np.column_stack([self.sigma, self.axial_margin, self.tau, self.torsional_margin])

    def force_table(self) -> FloatArray:
        """``(n, 2)`` rows ``axial force, torque`` (force print)."""
        return np.column_stack([self.force, self.torque])


def _per_element(values, eids: np.ndarray, default: float) -> np.ndarray:
    if values is None:
        return np.full(eids.shape[0], default)
    if isinstance(values, Mapping):
        return np.array([values.get(int(e), default) for e in eids], dtype=float)
    return np.broadcast_to(np.asarray(values, dtype=float), eids.shape).astype(float)


def rod_stresses(
    batch: RodStressBatch,
    displacements: Sequence[float] | NDArray[np.float64],
    temperatures: Mapping[int, float] | float | NDArray[np.float64] | None = None,
    deformations: Mapping[int, float] | float | NDArray[np.float64] | None = None,
    real_words: bool = True,
) -> RodStressResults:
    """Vectorised ``SROD2`` for every rod of ``batch``.

    ``temperatures``/``deformations`` are per element (mapping EID -> value, a
    scalar or an ``(n,)`` array). A NaN or missing temperature means no thermal
    load for that element; ``None`` means none at all.
    """
    dt = np.float32 if real_words else np.float64
    u = np.asarray(displacements, dtype=float)
    ia, ib = batch.sil[:, 0] - 1, batch.sil[:, 1] - 1
    trans = np.arange(3)

    def dot(a, base):
        return _dot(a, u[base[:, None] + trans], dt)

    temp = _per_element(temperatures, batch.eid, np.nan)
    has_t = ~np.isnan(temp)
    delta = _per_element(deformations, batch.eid, 0.0).astype(dt)
    sdelta, st, tref = (getattr(batch, f).astype(dt) for f in ("sdelta", "st", "tref"))

    sigma = dot(batch.sat, ia) + dot(batch.sbt, ib) + sdelta * delta
    thermal = st * (np.nan_to_num(temp).astype(dt) - tref)
    sigma = np.where(has_t, sigma + thermal, sigma).astype(dt)
    torque = dot(batch.sar, ia + 3) + dot(batch.sbr, ib + 3)
    force = batch.area.astype(dt) * sigma
    tau = batch.c_over_j.astype(dt) * torque

    sig_t, sig_c, sig_s = (getattr(batch, f).astype(dt) for f in ("sigma_t", "sigma_c", "sigma_s"))
    one = dt(1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tension = (sigma > 0.0) & (sig_t > 0.0)
        compression = (sigma < 0.0) & (sig_c > 0.0)
        ms_sigma = np.full(len(batch), np.nan)
        ms_sigma[tension] = (sig_t[tension] / sigma[tension] - one).astype(float)
        ms_sigma[compression] = (-sig_c[compression] / sigma[compression] - one).astype(float)
        torsion = (sig_s > 0.0) & (tau != 0.0)
        ms_tau = np.full(len(batch), np.nan)
        ms_tau[torsion] = (sig_s[torsion] / np.abs(tau[torsion]) - one).astype(float)
    return RodStressResults(
        batch.eid.copy(),
        sigma.astype(float),
        ms_sigma,
        tau.astype(float),
        ms_tau,
        force.astype(float),
        torque.astype(float),
    )
