"""CSHEAR shear panel element.

Port of NASTRAN-95 mis/sheard.f (double precision stiffness, EMG) and mis/spanl1.f, mis/spanl2.f (SDR2 forces, kick loads and stresses).
"""
CARDS = ("CSHEAR",)


def stiffness(model, elem):
    raise NotImplementedError("CSHEAR stiffness not ported yet")


def recover(model, elem, disp):
    raise NotImplementedError("CSHEAR stress/force recovery not ported yet")
