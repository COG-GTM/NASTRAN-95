"""CQDMEM1 isoparametric quadrilateral membrane element.

Port of NASTRAN-95 mis/qdmm1d.f (double precision stiffness, EMG; helpers mis/qdmm1x.f) and mis/sqdm11.f, mis/sqdm12.f (SDR2 stresses).

The element is the NASTRAN-95 isoparametric membrane on the mean plane of the four grids:
the normal stiffness is integrated with 2x2 Gauss points and the shear term is evaluated
once at the centroid (mean of the corner shear strains). A warped element is joined to
its grids through the mean-plane transformation B (rigid offsets HH = -h/2 normal to the
mean plane). Stresses are recovered in single precision at the intersection of the
diagonals (ETAS, XIS), in the element system whose x axis follows side 1-2.
"""
import numpy as np

from ..model import Mat1

CARDS = ("CQDMEM1",)
STRESSES = "STRESSES IN QUADRILATERAL MEMBRANES (CQDMEM1)"

GAUSS = (0.211324865, 0.788675135)
NBPW = 32  # gfortran build: SYSTEM(40), bits per word (mds/btstrp.f, MACH = 7)


def _r(x):
    """ECPT, MATOUT and the SDR2 working storage are REAL*4."""
    return float(np.float32(x))


def _material(model, elem):
    """MAT, INFLAG = 2 (mis/premat.f). premat rotates G by THETA (U' G U) only for MAT2
    data; a MAT1 returns the isotropic G before that branch, so THETA leaves it unchanged."""
    prop = model.props[elem.pid]
    mat = model.mats[prop["mid"]]
    if not isinstance(mat, Mat1):
        raise NotImplementedError(f"CQDMEM1 {elem.id}: only MAT1 is parsed (THETA rotates MAT2 data)")
    e, nu, g = _r(mat.e), _r(mat.nu), _r(mat.g)
    g11 = _r(e / _r(1.0 - _r(nu * nu)))
    g = np.array([[g11, _r(nu * g11), 0.0], [_r(nu * g11), g11, 0.0], [0.0, 0.0, g]])
    return g, _r(prop["t"])


def _gmmats(a, b):
    """mis/gmmats.f: A * B accumulated term by term in REAL*4."""
    c = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for k in range(a.shape[1]):
        c += np.outer(a[:, k], b[k, :]).astype(np.float32)
    return c


def _geometry(model, elem, dt=np.float64):
    """Mean plane of qdmm1d.f (dt float64) or sqdm11.f (dt float32, all REAL*4)."""
    x = [np.array(model.grids[gid].xyz, np.float32).astype(dt) for gid in elem.grids]
    d21, d31, d41, d42 = x[1] - x[0], x[2] - x[0], x[3] - x[0], x[3] - x[1]
    k = np.cross(d31, d42)
    magk = np.linalg.norm(k)
    if magk <= 1.0e-6:
        raise ValueError(f"CQDMEM1 {elem.id}: diagonals are parallel")
    k = k / magk
    hh = d21 @ k
    i = d21 - hh * k
    i = i / np.linalg.norm(i)
    hh = -hh / 2.0
    j = np.cross(k, i)
    j = j / np.linalg.norm(j)
    e = np.column_stack((i, j, k))  # E(1..9) of qdmm1d.f, stored by rows
    g = {"hh": hh, "e": e, "x12": -(d21 @ i), "x13": -(d31 @ i), "x14": -(d41 @ i),
         "y3a": d31 @ j, "y4a": d42 @ j}
    g["x24"] = g["x14"] - g["x12"]
    g["x23"] = g["x13"] - g["x12"]
    g["x34"] = g["x14"] - g["x13"]
    g["y34"] = g["y3a"] - g["y4a"]
    if g["y3a"] <= 0.0 or g["y4a"] <= 0.0:
        raise ValueError(f"CQDMEM1 {elem.id}: bad geometry (qdmm1d.f error 26)")
    temp = np.linalg.norm(d31)
    ysub4 = np.linalg.norm(d42)
    g["planar"] = 2.0 * hh / (temp + ysub4) <= 1.0e-6
    return g


def _b_matrix(g):
    """(12 x 8) mean plane -> actual grid transformation, B(1..96) by rows."""
    hh, x12, x14, x23, x34 = g["hh"], g["x12"], g["x14"], g["x23"], g["x34"]
    y3a, y4a, y34 = g["y3a"], g["y4a"], g["y34"]
    la = abs(x12)
    lb = np.hypot(x23, y3a)
    lc = np.hypot(x34, y34)
    ld = np.hypot(x14, y4a)
    cth1, sth1 = -x14 / ld, y4a / ld
    cth2, sth2 = x23 / lb, y3a / lb
    cth31, sth31 = x34 / lc, -y34 / lc
    cth41, sth41 = cth1, sth1
    cth32, sth32 = sth2, cth2
    cth42, sth42 = sth31, cth31
    dlt1 = cth31 * cth32 - sth31 * sth32
    dlt2 = cth42 * cth41 + sth41 * sth42
    ldd2, lbd1, lcd1, lcd2 = ld * dlt2, lb * dlt1, lc * dlt1, lc * dlt2
    b = np.zeros(97, type(hh))
    b[1] = b[10] = b[27] = b[36] = b[53] = b[62] = b[79] = b[88] = 1.0
    b[17] = -hh / la
    b[18] = -hh / (ld * sth1) + (hh * cth1) / (la * sth1)
    b[19] = hh / la
    b[20] = (hh * cth2) / (la * sth2)
    b[23] = (hh * cth42) / ldd2
    b[24] = (hh * sth42) / ldd2
    b[41] = -b[17]
    b[42] = (-hh * cth1) / (la * sth1)
    b[43] = b[17]
    b[44] = (-hh * cth2) / (la * sth2) + hh / (lb * sth2)
    b[45] = (-hh * sth31) / lbd1
    b[46] = (-hh * cth31) / lbd1
    b[68] = -hh / (lb * sth2)
    b[69] = hh * (sth31 / lbd1 + cth32 / lcd1)
    b[70] = hh * (cth31 / lbd1 + sth32 / lcd1)
    b[71] = (-hh * sth41) / lcd2
    b[72] = (hh * cth41) / lcd2
    b[90] = hh / (ld * sth1)
    b[93] = (-hh * cth32) / lcd1
    b[94] = (-hh * sth32) / lcd1
    b[95] = hh * (-cth42 / ldd2 + sth41 / lcd2)
    b[96] = hh * (-sth42 / ldd2 - cth41 / lcd2)
    return b[1:].reshape(12, 8)


def _mean_plane_stiffness(g, gm, t):
    x12, x13, x14, x23, x24, x34 = (g[k] for k in ("x12", "x13", "x14", "x23", "x24", "x34"))
    y3a, y4a, y34 = g["y3a"], g["y4a"], g["y34"]
    c = [None, gm[0, 0], gm[0, 1], gm[1, 1], gm[0, 2], gm[1, 2], 0.0]
    fact = gm[2, 2] * t / (x24 * y3a - x13 * y4a) * 2.0
    aq, bq, cq = np.zeros(25), np.zeros(25), np.zeros(25)
    for arr, vals in (
        (aq, {1: -y4a, 3: -x24, 5: -x24, 6: -y4a, 7: y4a, 9: x14, 11: x14, 12: y4a,
              21: -x12, 23: -x12}),
        (bq, {1: y3a, 3: x23, 5: x23, 6: y3a, 7: -y4a, 9: -x14, 11: -x14, 12: -y4a,
              13: y4a, 15: x14, 17: x14, 18: y4a, 19: -y3a, 21: -x23, 23: -x23, 24: -y3a}),
        (cq, {1: -y34, 3: x34, 5: x34, 6: -y34, 7: y34, 9: -x34, 11: -x34, 12: y34,
              15: -x12, 17: -x12, 21: x12, 23: x12})):
        for idx, v in vals.items():
            arr[idx] = v

    def o(d, v, f, h, p, q, eta, tea):
        return (d + v * tea + f * eta + h * tea * eta + p * tea * tea + q * eta * eta) / (
            -y4a * x12 - y34 * x12 * eta + (-y4a * x23 + y3a * x14) * tea)

    u = np.zeros((8, 8))
    for i in range(4):
        for k in range(2):
            k1, k2 = 6 * i + 4 * k + 1, 6 * i + 3 * k + 3
            for jj in range(4):
                for l in range(2):
                    l1, l2 = 6 * jj + 4 * l + 1, 6 * jj + 3 * l + 3
                    cc = (c[k + l + 1], c[k + 4], c[l + 4])

                    def term(x, y, cc=cc, k1=k1, k2=k2, l1=l1, l2=l2):
                        return cc[0] * x[k1] * y[l1] + cc[1] * x[k1] * y[l2] + cc[2] * x[k2] * y[l1]

                    d = term(aq, aq)
                    v = term(aq, bq) + term(bq, aq)
                    f = term(aq, cq) + term(cq, aq)
                    h = term(bq, cq) + term(cq, bq)
                    p = term(bq, bq)
                    q = term(cq, cq)
                    s = sum(o(d, v, f, h, p, q, ea, eb) for ea in GAUSS for eb in GAUSS)
                    s = s / 4.0 * t
                    s += fact * (aq[k2] + 0.5 * (bq[k2] + cq[k2])) * (aq[l2] + 0.5 * (bq[l2] + cq[l2]))
                    u[2 * i + k, 2 * jj + l] = s
    return u


def stiffness(model, elem):
    g = _geometry(model, elem)
    gm, t = _material(model, elem)
    u = _mean_plane_stiffness(g, gm, t)
    if g["planar"]:
        idx = [0, 1, 3, 4, 6, 7, 9, 10]
        k12 = np.zeros((12, 12))
        k12[np.ix_(idx, idx)] = u
    else:
        b = _b_matrix(g)
        k12 = b @ u @ b.T
    e = g["e"]
    ke = np.zeros((24, 24))
    for a in range(4):
        for c in range(4):
            ke[6 * a:6 * a + 3, 6 * c:6 * c + 3] = e @ k12[3 * a:3 * a + 3, 3 * c:3 * c + 3] @ e.T
    return ke


def _diagonal_intersection(g):
    """ETAS, XIS of mis/sqdm11.f."""
    x12, x13, x14, x23, x24, x34 = (g[k] for k in ("x12", "x13", "x14", "x23", "x24", "x34"))
    y3a, y4a, y34 = g["y3a"], g["y4a"], g["y34"]
    tol = (1.0e-5 if NBPW >= 60 else 1.0e-3) * (-x12)
    tol2 = (1.0e-5 if NBPW >= 60 else 1.0e-3) * x12 * x12
    if abs(x34 + x12) <= tol and abs(y34) <= tol:
        return 0.5, 0.5
    if not (abs(x24) < tol or abs(x13) < tol):
        xstar = (y4a * x13 * x12) / (y3a * x24 - y4a * x13)
        ystar = (-y4a / x24) * (xstar + x12)
    elif abs(x13) <= tol:
        xstar = -x13
        ystar = (-y4a / x24) * x12
    else:
        xstar = -x12
        ystar = (y3a / x13) * x12
    if abs(x34 + x12) >= tol:
        c1 = y34 * xstar - ystar * (x34 + x12)
        a2 = -y4a * x23 + y3a * x14
        b2 = -y4a * x12 + c1
        if abs(a2) > tol2:
            temp2 = np.sqrt(b2 ** 2 - 4.0 * a2 * x12 * ystar) / (2.0 * a2)
            temp1 = -b2 / (2.0 * a2)
            etas = temp1 - temp2
            if etas <= 0.0 or etas >= 1.0:
                etas = temp1 + temp2
        else:
            etas = (-x12 * ystar) / b2
        if abs(y34) >= tol:
            xis = (-c1 + (y4a * x23 - y3a * x14) * etas) / (y34 * x12)
        else:
            xis = (xstar + x14 * etas) / (etas * (x34 + x12) - x12)
        return etas, xis
    a3 = -x14 * y34
    b3 = x12 * y4a - y34 * xstar
    if abs(a3) > tol2:
        temp2 = np.sqrt(b3 ** 2 + 4.0 * a3 * x12 * ystar) / (2.0 * a3)
        temp1 = -b3 / (2.0 * a3)
        etas = temp1 - temp2
        if etas <= 0.0 or etas >= 1.0:
            etas = temp1 + temp2
    else:
        etas = (x12 * ystar) / b3
    return etas, (ystar - y4a * etas) / (y34 * etas)


def _stress_matrix(model, elem):
    """S (3 x 12) of mis/sqdm11.f: basic grid translations -> element stresses."""
    g = _geometry(model, elem, np.float32)
    gm = _material(model, elem)[0].astype(np.float32)
    x12, x14, x23, x24, x34 = (g[k] for k in ("x12", "x14", "x23", "x24", "x34"))
    y3a, y4a, y34 = g["y3a"], g["y4a"], g["y34"]
    etas, xis = _diagonal_intersection(g)
    aj = -y4a * x12 - y34 * x12 * xis + etas * (-y4a * x23 + y3a * x14)
    a = np.zeros(25, np.float32)
    a[1] = a[18] = (-y4a + y3a * etas - y34 * xis) / aj
    a[3] = a[20] = (y4a - y4a * etas + y34 * xis) / aj
    a[5] = a[22] = (y4a * etas) / aj
    a[7] = a[24] = (-y3a * etas) / aj
    a[10] = a[17] = (-x24 + x23 * etas + x34 * xis) / aj
    a[12] = a[19] = (x14 - x14 * etas - x34 * xis) / aj
    a[14] = a[21] = (x14 * etas - x12 * xis) / aj
    a[16] = a[23] = (-x12 - x23 * etas + x12 * xis) / aj
    ee = np.kron(np.eye(4, dtype=np.float32), g["e"])  # EE(144); GMMATS(B,12,8,1, EE,12,12,1) is B' EE'
    s = _gmmats(_b_matrix(g).T, ee.T)
    s = _gmmats(a[1:].reshape(3, 8), s)
    return _gmmats(gm, s)


def recover(model, elem, disp):
    s = _stress_matrix(model, elem)
    u = np.concatenate([np.asarray(disp[gid], dtype=float)[:3] for gid in elem.grids])
    sx, sy, sxy = s.astype(np.float64) @ u
    temp = sx - sy
    tau = np.sqrt((temp / 2.0) ** 2 + sxy ** 2)
    mean = (sx + sy) / 2.0
    delta = 2.0 * sxy
    if abs(delta) < 1.0e-15 and abs(temp) < 1.0e-15:
        phi = 0.0
    elif abs(temp) < 1.0e-15:
        phi = 45.0
    else:
        phi = np.arctan2(delta, temp) * 28.6478898
    return {STRESSES: {elem.id: [sx, sy, sxy, phi, mean + tau, mean - tau, tau]}}
