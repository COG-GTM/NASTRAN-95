"""CROD and CONROD axial/torsion rod elements.

Port of NASTRAN-95 mis/rodd.f (double precision stiffness, EMG) and mis/srod1.f, mis/srod2.f (SDR2 force and stress recovery); PROD via mis/rod.f.
"""
import numpy as np

CARDS = ("CROD", "CONROD")
FORCES = "FORCES IN ROD ELEMENTS ({})"
STRESSES = "STRESSES IN ROD ELEMENTS ({})"


def _section(model, elem):
    prop = elem.data if elem.type == "CONROD" else model.props[elem.pid]
    return model.mats[prop["mid"]], prop["a"], prop["j"], prop["c"]


def _axis(model, elem):
    d = model.grids[elem.grids[1]].xyz - model.grids[elem.grids[0]].xyz
    length = np.linalg.norm(d)
    return d / length, length


def stiffness(model, elem):
    mat, a, j, _ = _section(model, elem)
    e, length = _axis(model, elem)
    ee = np.outer(e, e)
    ke = np.zeros((12, 12))
    for off, k in ((0, mat.e * a / length), (3, mat.g * j / length)):
        for r, c, s in ((0, 0, 1), (0, 6, -1), (6, 0, -1), (6, 6, 1)):
            ke[off + r:off + r + 3, off + c:off + c + 3] = s * k * ee
    return ke


def recover(model, elem, disp):
    mat, a, j, c = _section(model, elem)
    e, length = _axis(model, elem)
    u1, u2 = disp[elem.grids[0]], disp[elem.grids[1]]
    axial = mat.e * a / length * e @ (u2[:3] - u1[:3])
    torque = mat.g * j / length * e @ (u2[3:] - u1[3:])
    stress = axial / a if a else 0.0
    tstress = c * torque / j if j else 0.0
    return {FORCES.format(elem.type): {elem.id: [axial, torque]},
            STRESSES.format(elem.type): {elem.id: [stress, tstress]}}
