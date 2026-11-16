"""CQDMEM2 quadrilateral membrane element (four overlapping triangles).

Port of NASTRAN-95 mis/qdmm2d.f (double precision stiffness, EMG) and mis/sqdm21.f, mis/sqdm22.f (SDR2 stresses).
"""
CARDS = ("CQDMEM2",)


def stiffness(model, elem):
    raise NotImplementedError("CQDMEM2 stiffness not ported yet")


def recover(model, elem, disp):
    raise NotImplementedError("CQDMEM2 stress/force recovery not ported yet")
