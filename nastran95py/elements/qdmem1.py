"""CQDMEM1 isoparametric quadrilateral membrane element.

Port of NASTRAN-95 mis/qdmm1d.f (double precision stiffness, EMG; helpers mis/qdmm1x.f) and mis/sqdm11.f, mis/sqdm12.f (SDR2 stresses).
"""
CARDS = ("CQDMEM1",)


def stiffness(model, elem):
    raise NotImplementedError("CQDMEM1 stiffness not ported yet")


def recover(model, elem, disp):
    raise NotImplementedError("CQDMEM1 stress/force recovery not ported yet")
