"""CROD and CONROD axial/torsion rod elements.

Port of NASTRAN-95 mis/rodd.f (double precision stiffness, EMG) and mis/srod1.f, mis/srod2.f (SDR2 force and stress recovery); PROD via mis/rod.f.
"""
CARDS = ("CROD", "CONROD")


def stiffness(model, elem):
    raise NotImplementedError("CROD/CONROD stiffness not ported yet")


def recover(model, elem, disp):
    raise NotImplementedError("CROD/CONROD stress/force recovery not ported yet")
