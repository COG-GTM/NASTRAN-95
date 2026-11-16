"""CQDMEM2 quadrilateral membrane element (four overlapping triangles).

Port of NASTRAN-95 mis/qdmm2d.f (double precision stiffness, EMG) and mis/sqdm21.f, mis/sqdm22.f (SDR2 stresses).
Helpers: mis/q2bcd.f / q2bcs.f (element axes, corner and centre coordinates) and
mis/q2trmd.f / q2trms.f (sub-triangle partitions). The quad is split into the triangles
(1,2,5), (2,3,5), (3,4,5), (4,1,5) about the mean point 5, whose in-plane DOFs are condensed
out. Each 3x3 partition acts on the three translations of a grid in element coordinates;
the rotations carry no stiffness.
"""
import numpy as np

CARDS = ("CQDMEM2",)
FORCES = "FORCES ACTING ON QDMEM2 ELEMENTS (CQDMEM2)"
STRESSES = "STRESSES ACTING IN QDMEM2 ELEMENTS (CQDMEM2)"

SIDES = ((0, 1), (1, 2), (2, 3), (3, 0))  # sub-triangles (ia, ib, 5) and the quad's sides


def _material(model, elem):
    prop = model.props[elem.pid]
    mat = model.mats[prop["mid"]]
    g11 = mat.e / (1.0 - mat.nu ** 2)
    gsube = np.array([[g11, mat.nu * g11, 0.0], [mat.nu * g11, g11, 0.0], [0.0, 0.0, mat.g]])
    return gsube, prop["t"]


def _geometry(model, elem):
    """Q2BCD: element axes E (rows x, y, z in basic), RMAT (5 points in element coords), PLANAR."""
    g = np.array([model.grids[i].xyz for i in elem.grids], dtype=float)
    d12, d13, d24 = g[1] - g[0], g[2] - g[0], g[3] - g[1]
    vec = np.cross(d13, d24)
    mag = np.linalg.norm(vec)
    if mag <= 0.0:
        raise ValueError(f"CQDMEM2 {elem.id}: illegal geometry")
    area = 0.5 * mag
    ez = vec / mag
    twoh = d12 @ ez
    vec = d12 - twoh * ez
    ex = vec / np.linalg.norm(vec)
    e = np.array([ex, np.cross(ez, ex), ez])
    pts = np.vstack([g, g.mean(axis=0)])
    rmat = (pts - g[0]) @ e.T
    rmat[:, 2] -= twoh / 2.0
    rmat[0] = (0.0, 0.0, -twoh / 2.0)
    planar = (twoh / 2.0) ** 2 / area <= 0.01
    return e, rmat, planar


def _triangle(ra, rb, rc, sinth, costh, gsube, t):
    """Q2TRMD/Q2TRMS: K partitions {(i, j): 3x3} for i, j in A,B,C (0, 1, 2) and S_i = G H_i."""
    v12, v13 = rb - ra, rc - ra
    kvec = np.cross(v12, v13)
    mag = np.linalg.norm(kvec)
    kvec /= mag
    area = 0.5 * mag
    xb = np.linalg.norm(v12)
    ivec = v12 / xb
    jvec = np.cross(kvec, ivec)
    m = np.hypot(ivec[0], ivec[1])
    ic = (ivec[0] * costh + ivec[1] * sinth) / m
    is_ = (ivec[0] * sinth - ivec[1] * costh) / m
    xc, yc = ivec @ v13, jvec @ v13
    i33 = 1.0 / yc
    ca = np.array([[-1.0 / xb, 0.0, i33 * (xc / xb - 1.0)], [0.0, i33 * (xc / xb - 1.0), -1.0 / xb]])
    cb = np.array([[1.0 / xb, 0.0, -i33 * xc / xb], [0.0, -i33 * xc / xb, 1.0 / xb]])
    cc = np.array([[0.0, 0.0, i33], [0.0, i33, 0.0]])
    tm = np.array([[ic * ic, is_ * is_, ic * is_],
                   [is_ * is_, ic * ic, -ic * is_],
                   [-2.0 * ic * is_, 2.0 * ic * is_, ic * ic - is_ * is_]])
    e2 = np.array([ivec, jvec])
    h = [tm @ c.T @ e2 for c in (ca, cb, cc)]
    at = area * t
    k = {(i, j): h[i].T @ gsube @ h[j] * at for i in range(3) for j in range(3)}
    return k, [gsube @ hi for hi in h]


def _condensed(model, elem):
    """Sub-triangle sums and condensation of point 5: K_ij (element coords), G_i, SISUM."""
    gsube, t = _material(model, elem)
    e, rmat, planar = _geometry(model, elem)
    ang = np.radians(elem.data.get("theta", 0.0))
    sinth, costh = np.sin(ang), np.cos(ang)
    k1 = np.zeros((4, 4, 3, 3))
    k5 = np.zeros((5, 3, 3))  # K_5i, i = 1..5
    si = np.zeros((5, 3, 3))
    for ia, ib in SIDES:
        k, s = _triangle(rmat[ia], rmat[ib], rmat[4], sinth, costh, gsube, t)
        for n, p in enumerate((ia, ib, 4)):
            k5[p] += k[2, n]
            si[p] += s[n]
        k1[ia, ia] += k[0, 0]
        k1[ia, ib] += k[0, 1]
        k1[ib, ia] += k[1, 0]
        k1[ib, ib] += k[1, 1]
    k5mod = k5.copy()
    if planar:
        k5mod[:, 2] = (0.0, 0.0, -0.25)
        k5mod[4, 2] = (0.0, 0.0, 1.0)
    inv = -np.linalg.inv(k5mod[4])
    g = [inv @ k5mod[i] for i in range(4)]
    k = k1.copy()
    for i in range(4):
        for j in range(4):
            k[i, j] += k5[i].T @ g[j]
            if planar:
                k[i, j] += g[i].T @ k5[4] @ g[j] + g[i].T @ k5[j]
    return e, rmat, k, g, si


def stiffness(model, elem):
    e, _, k, _, _ = _condensed(model, elem)
    ke = np.zeros((24, 24))
    for i in range(4):
        for j in range(4):
            ke[6 * i:6 * i + 3, 6 * j:6 * j + 3] = e.T @ k[i, j] @ e
    return ke


def recover(model, elem, disp):
    e, rmat, k, g, si = _condensed(model, elem)
    u = [np.asarray(disp[gid][:3], dtype=float) for gid in elem.grids]
    sig = sum(0.25 * (si[i] + si[4] @ g[i]) @ e @ u[i] for i in range(4))

    d = np.array([rmat[b] - rmat[a] for a, b in SIDES])
    rg = np.linalg.norm(d, axis=1)
    d /= rg[:, None]
    f = np.zeros((4, 3))
    for i in range(4):
        prev = (i - 1) % 4
        n = np.cross(d[i], d[prev])
        q = np.column_stack([-d[i], d[prev], -n / np.linalg.norm(n)])
        qinv = np.linalg.inv(q)
        f[i] = sum(qinv @ k[i, j] @ e @ u[j] for j in range(4))
    shear = [(f[(i + 1) % 4, 1] - f[i, 0]) / rg[i] for i in range(4)]

    sx, sy, sxy = sig
    diff = sx - sy
    tau = np.sqrt((diff / 2.0) ** 2 + sxy ** 2)
    mean = (sx + sy) / 2.0
    if abs(diff) >= 1.0e-15:
        angle = np.degrees(np.arctan2(2.0 * sxy, diff)) / 2.0
    elif abs(2.0 * sxy) >= 1.0e-15:
        angle = 0.0
    else:
        angle = 45.0
    forces = [f[0, 1], f[0, 0], f[1, 1], f[1, 0], f[2, 1], f[2, 0], f[3, 1], f[3, 0],
              f[0, 2], -shear[0], f[1, 2], shear[1], f[2, 2], -shear[2], f[3, 2], shear[3]]
    return {FORCES: {elem.id: forces},
            STRESSES: {elem.id: [sx, sy, sxy, angle, mean + tau, mean - tau, tau]}}
