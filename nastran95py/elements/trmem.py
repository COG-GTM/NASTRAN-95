"""CTRMEM constant-strain triangular membrane element.

Port of NASTRAN-95 mis/trmemd.f (double precision stiffness, EMG) and mis/strme1.f plus the shared SDR2 membrane stress output.
"""
CARDS = ("CTRMEM",)


def stiffness(model, elem):
    raise NotImplementedError("CTRMEM stiffness not ported yet")


def recover(model, elem, disp):
    raise NotImplementedError("CTRMEM stress/force recovery not ported yet")
