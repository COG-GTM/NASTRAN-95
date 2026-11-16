"""Rigid Format 1 linear static solution: assemble, constrain, solve, recover.

Mirrors the NASTRAN-95 static path for the ported element group: EMG/EMA element
stiffness assembly into KGG, GP4 single-point constraint partitioning (u_s = 0),
SSG1 static loads, SSG2/SSG3 reduction and solution, then SDR1 SPC forces and SDR2
element force/stress recovery. Displacement degrees of freedom with no stiffness
(the grid point singularities GPSP would find) are held at zero, as NASTRAN does.
"""
from dataclasses import dataclass

import numpy as np

from .elements import REGISTRY

DISPLACEMENT = "DISPLACEMENT VECTOR"
SPCFORCE = "FORCES OF SINGLE-POINT CONSTRAINT"


@dataclass
class Result:
    disp: dict  # grid id -> length-6 ndarray
    spcforce: dict  # grid id -> length-6 ndarray, grids carrying SPC/SPC1 constraints
    tables: dict  # {title: {id: [values]}}, the listing the Python port prints


def _comps(s):
    return [int(ch) - 1 for ch in str(s) if ch.isdigit() and ch != "0"]


def assemble(model):
    order = sorted(model.grids)
    index = {g: 6 * i for i, g in enumerate(order)}
    k = np.zeros((6 * len(order), 6 * len(order)))
    for elem in model.elements:
        ke = REGISTRY[elem.type].stiffness(model, elem)
        dofs = np.concatenate([np.arange(index[g], index[g] + 6) for g in elem.grids])
        k[np.ix_(dofs, dofs)] += ke
    return order, index, k


def load_vector(model, sid, order, index):
    p = np.zeros(6 * len(order))
    for gid, cid, f, n, kind in model.forces.get(sid, []):
        if cid:
            raise NotImplementedError("loads in a local coordinate system are not ported")
        off = index[gid] + (3 if kind == "MOMENT" else 0)
        p[off:off + 3] += f * n
    return p


def solve(model):
    order, index, k = assemble(model)
    p = load_vector(model, int(model.case.get("LOAD", 0)), order, index)
    spc = np.zeros(len(p), bool)
    user_spc = np.zeros(len(p), bool)
    for g in model.grids.values():
        for c in _comps(g.ps):
            spc[index[g.id] + c] = True
    for gid, comps in model.spcs.get(int(model.case.get("SPC", 0)), []):
        for c in _comps(comps):
            spc[index[gid] + c] = user_spc[index[gid] + c] = True
    free = ~spc & (np.abs(np.diag(k)) > 0.0)
    u = np.zeros(len(p))
    u[free] = np.linalg.solve(k[np.ix_(free, free)], p[free])
    q = np.where(spc, k @ u - p, 0.0)
    disp = {g: u[index[g]:index[g] + 6] for g in order}
    spcforce = {g: q[index[g]:index[g] + 6] for g in order if user_spc[index[g]:index[g] + 6].any()}
    tables = {DISPLACEMENT: {g: list(v) for g, v in disp.items()},
              SPCFORCE: {g: list(v) for g, v in spcforce.items()}}
    tables.update(recover(model, disp))
    return Result(disp, spcforce, tables)


def recover(model, disp):
    tables = {}
    for elem in model.elements:
        for title, rows in REGISTRY[elem.type].recover(model, elem, disp).items():
            tables.setdefault(title, {}).update(rows)
    return tables
