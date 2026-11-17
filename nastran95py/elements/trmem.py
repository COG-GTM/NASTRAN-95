"""CTRMEM constant-strain triangular membrane element.

Port of NASTRAN-95 mis/trmemd.f with mis/ektrmd.f (double precision stiffness, EMG),
mis/strme1.f (SDR2 phase 1 stress matrices) and mis/stqme2.f (SDR2 phase 2 stresses,
NTYPE=1). Material via mis/premat.f, ENTRY MAT with INFLAG=2.
"""
import numpy as np

from ..model import Mat1

CARDS = ("CTRMEM",)
STRESSES = "STRESSES IN TRIANGULAR MEMBRANES (CTRMEM)"
DEG_HALF = 28.6478898  # stqme2.f: degrees per radian / 2


def _geometry(model, elem):
    """Element axes E (3x2, columns i and j), area and the C_a, C_b, C_c strain matrices
    (each 3x2: rows eps_x, eps_y, gamma_xy; columns u, v in element axes)."""
    xa, xb, xc = (model.grids[g].xyz for g in elem.grids)
    i = xb - xa
    xsubb = np.linalg.norm(i)
    if xsubb <= 1e-6:
        raise ValueError(f"CTRMEM {elem.id}: grids A and B coincide")
    i = i / xsubb
    rc = xc - xa
    xsubc = i @ rc
    k = np.cross(i, rc)
    ysubc = np.linalg.norm(k)
    if ysubc <= 1e-6:
        raise ValueError(f"CTRMEM {elem.id}: grids are collinear")
    k = k / ysubc
    j = np.cross(k, i)
    j = j / np.linalg.norm(j)
    mu, lam = 1.0 / xsubb, 1.0 / ysubc
    delta = xsubc / xsubb - 1.0
    c = np.array([
        [[-mu, 0.0], [0.0, lam * delta], [lam * delta, -mu]],
        [[mu, 0.0], [0.0, -lam * mu * xsubc], [-lam * mu * xsubc, mu]],
        [[0.0, 0.0], [0.0, lam], [lam, 0.0]],
    ])
    return np.column_stack((i, j)), xsubb * ysubc / 2.0, c


def _material(model, elem):
    """Plane-stress G as ENTRY MAT returns it for INFLAG=2 with SINTH/COSTH from THETA.
    premat.f rotates only MAT2 data (U^T G U); a MAT1 G is isotropic and returned as is,
    so THETA does not change a MAT1 CTRMEM, and stresses print in element axes."""
    mat = model.mats[model.props[elem.pid]["mid"]]
    if not isinstance(mat, Mat1):
        raise NotImplementedError(f"CTRMEM {elem.id}: only MAT1 materials are ported")
    g11 = mat.e / (1.0 - mat.nu ** 2)
    return np.array([[g11, mat.nu * g11, 0.0], [mat.nu * g11, g11, 0.0], [0.0, 0.0, mat.g]])


def stiffness(model, elem):
    e, area, c = _geometry(model, elem)
    g = _material(model, elem)
    vol = area * model.props[elem.pid]["t"]
    ke = np.zeros((18, 18))
    for a in range(3):
        left = vol * e @ c[a].T @ g
        for b in range(3):
            ke[6 * a:6 * a + 3, 6 * b:6 * b + 3] = left @ c[b] @ e.T
    return ke


def recover(model, elem, disp):
    e, _, c = _geometry(model, elem)
    g = _material(model, elem)
    sx, sy, sxy = sum(g @ c[a] @ e.T @ np.asarray(disp[gid][:3], dtype=float)
                      for a, gid in enumerate(elem.grids))
    temp = sx - sy
    shear = np.sqrt((temp / 2.0) ** 2 + sxy ** 2)
    mean = (sx + sy) / 2.0
    two_sxy = 2.0 * sxy
    if abs(two_sxy) < 1e-15 and abs(temp) < 1e-15:
        angle = 0.0
    else:
        angle = np.arctan2(two_sxy, temp) * DEG_HALF
    return {STRESSES: {elem.id: [sx, sy, sxy, angle, mean + shear, mean - shear, shear]}}
