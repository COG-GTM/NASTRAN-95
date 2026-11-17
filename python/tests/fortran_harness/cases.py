"""Hand-built element cases for the Fortran harness.

Every case is an EST/ECPT record laid out exactly as the Fortran routine reads
it (``int`` entries are integer words, ``float`` entries are REAL words), plus
the driver parameters for its path. All coordinates are basic (CID 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EMG, SMA1, SDR2 = 1, 2, 3

# MAT1: E G NU RHO ALPHA TREF GE ST SC SS
MAT1 = [
    (1, (3.0e7, 3.0e7 / 2.6, 0.3, 7.3e-4, 6.5e-6, 70.0, 0.0, 0.0, 0.0, 0.0)),
    (2, (1.0e7, 1.0e7 / 2.66, 0.33, 2.5e-4, 1.2e-5, 0.0, 0.0, 0.0, 0.0, 0.0)),
]
# MAT2: G11 G12 G13 G22 G23 G33 RHO A1 A2 A12 TREF GE ST SC SS
MAT2 = [
    (
        3,
        (
            2.0e7,
            5.0e6,
            1.0e6,
            1.5e7,
            8.0e5,
            6.0e6,
            1.5e-4,
            1e-5,
            2e-5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
]
MAT1_BY_ID = {mid: vals for mid, vals in MAT1}

SILS = (1, 7, 13, 19)

# Element type code, SDR2 word counts (NWDSA, NWDSTR, NWDFOR) from GPTA1 (bd/gptabd.f).
SDR_WORDS = {
    "CROD": (1, 23, 5, 3),
    "CONROD": (10, 23, 5, 3),
    "CSHEAR": (4, 33, 4, 17),
    "CTRMEM": (9, 36, 8, 0),
    "CQDMEM": (16, 45, 8, 0),
    "CQDMEM1": (62, 45, 8, 0),
    "CQDMEM2": (63, 250, 8, 17),
}


@dataclass
class Case:
    name: str
    element: str
    routine: str
    kind: int
    words: list
    extra: list
    grids: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _grid_words(grids):
    words: list = []
    for xyz in grids:
        words += [0, *map(float, xyz)]
    return words


def rod_est(eid, grids, mid, area, j, c=0.0, nsm=0.0):
    return [eid, SILS[0], SILS[1], mid, area, j, c, nsm, *_grid_words(grids), 0.0]


def shear_est(eid, grids, mid, t, nsm=0.0):
    return [eid, *SILS, mid, t, nsm, *_grid_words(grids), 0.0]


def trmem_est(eid, grids, mid, t, theta=0.0, nsm=0.0):
    return [eid, *SILS[:3], theta, mid, t, nsm, *_grid_words(grids), 0.0]


def qdmem_est(eid, grids, mid, t, theta=0.0, nsm=0.0):
    return [eid, *SILS, theta, mid, t, nsm, *_grid_words(grids), 0.0]


def quad1_membrane_est(eid, grids, mid, t, theta=0.0, nsm=0.0):
    # CQUAD1 ECPT with no bending material (I = 0): KTRIQD(3) only calls KQDMEM.
    return [eid, *SILS, theta, mid, t, 0, 0.0, 0, 0.0, nsm, 0.0, 0.0, *_grid_words(grids), 0.0]


ROD_GEOMS = {
    "x": ([(0, 0, 0), (10, 0, 0)], 1, 2.0, 0.5, 0.1, 0.01),
    "skew": ([(1, 2, 3), (4, -2, 8)], 2, 0.75, 0.2, 0.05, 0.0),
    "z": ([(0, 0, 0), (0, 0, 25)], 1, 1.5, 0.0, 0.0, 0.002),
}
SHEAR_GEOMS = {
    "rect": ([(0, 0, 0), (10, 0, 0), (10, 5, 0), (0, 5, 0)], 1, 0.1),
    "skew3d": ([(0, 0, 0), (8, 0, 4), (9, 6, 4.5), (1, 5, 0.5)], 2, 0.05),
    "trap": ([(0, 0, 0), (6, 0, 0), (5, 4, 0), (1, 4, 0)], 1, 0.08),
}
TRI_GEOMS = {
    "plane": ([(0, 0, 0), (4, 0, 0), (1, 3, 0)], 1, 0.2, 0.0),
    "aniso3d": ([(1, 1, 1), (5, 2, 1), (2, 6, 3)], 3, 0.1, 30.0),
    "right": ([(0, 0, 0), (2, 0, 0), (0, 5, 0)], 2, 0.05, 0.0),
}
QUAD_GEOMS = {
    "rect": ([(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], 1, 0.2, 0.0),
    "skew3d": ([(0, 0, 0), (6, 0, 3), (7, 4, 3.5), (1, 5, 0.5)], 2, 0.1, 0.0),
    "aniso": ([(0, 0, 0), (5, 0, 0), (5, 2, 0), (0, 2, 0)], 3, 0.15, 15.0),
}

# Uniform strain state used by the SDR2 patch tests.
STRAIN = {"ex": 1.0e-4, "ey": -4.0e-5, "gxy": 6.0e-5}


def patch_displacements(grids, ex=STRAIN["ex"], ey=STRAIN["ey"], gxy=STRAIN["gxy"]):
    disp = []
    for x, y, _ in grids:
        disp += [ex * x + 0.5 * gxy * y, ey * y + 0.5 * gxy * x, 0.0, 0.0, 0.0, 0.0]
    return disp


def rod_displacements(grids, stretch=2.0e-3, twist=1.0e-3):
    (a, b) = grids
    axis = [bi - ai for ai, bi in zip(a, b, strict=True)]
    length = math.sqrt(sum(v * v for v in axis))
    n = [v / length for v in axis]
    return [0.0] * 6 + [
        stretch * n[0],
        stretch * n[1],
        stretch * n[2],
        twist * n[0],
        twist * n[1],
        twist * n[2],
    ]


def _sdr(name, element, phase1, phase2, est, grids, disp, meta):
    eltype, nwdsa, nwdstr, nwdfor = SDR_WORDS[element]
    extra = [phase2, eltype, nwdsa, nwdstr, nwdfor, -1, 0.0, 0.0, len(disp), *disp]
    return Case(name, element, phase1, SDR2, est, extra, grids, meta)


def _build() -> list[Case]:
    out: list[Case] = []
    eid = 100
    for elem in ("CROD", "CONROD"):
        for tag, (grids, mid, area, j, c, nsm) in ROD_GEOMS.items():
            eid += 1
            est = rod_est(eid, grids, mid, area, j, c, nsm)
            meta = {"mid": mid, "area": area, "j": j, "c": c, "nsm": nsm}
            for routine in ("RODS", "RODD"):
                for icmbar in (-1, 1):
                    out.append(
                        Case(
                            f"{elem}-{tag}-{routine}-m{icmbar:+d}",
                            elem,
                            routine,
                            EMG,
                            est,
                            [icmbar, 2],
                            grids,
                            meta,
                        )
                    )
            out.append(
                Case(f"{elem}-{tag}-KROD", elem, "KROD", SMA1, est, [2, *SILS[:2]], grids, meta)
            )
            out.append(
                _sdr(
                    f"{elem}-{tag}-SROD",
                    elem,
                    "SROD1",
                    "SROD2",
                    est,
                    grids,
                    rod_displacements(grids),
                    meta,
                )
            )
    for tag, (grids, mid, t) in SHEAR_GEOMS.items():
        eid += 1
        est = shear_est(eid, grids, mid, t)
        meta = {"mid": mid, "t": t}
        for routine in ("SHEARS", "SHEARD"):
            out.append(
                Case(f"CSHEAR-{tag}-{routine}", "CSHEAR", routine, EMG, est, [-1, 4], grids, meta)
            )
        out.append(
            _sdr(
                f"CSHEAR-{tag}-SPANL",
                "CSHEAR",
                "SPANL1",
                "SPANL2",
                est,
                grids,
                patch_displacements(grids, 0.0, 0.0, 1.0e-4),
                meta,
            )
        )
    for tag, (grids, mid, t, theta) in TRI_GEOMS.items():
        eid += 1
        est = trmem_est(eid, grids, mid, t, theta)
        meta = {"mid": mid, "t": t, "theta": theta}
        for routine in ("TRMEMS", "TRMEMD"):
            out.append(
                Case(f"CTRMEM-{tag}-{routine}", "CTRMEM", routine, EMG, est, [-1, 3], grids, meta)
            )
        out.append(
            Case(f"CTRMEM-{tag}-KTRMEM", "CTRMEM", "KTRMEM", SMA1, est, [3, *SILS[:3]], grids, meta)
        )
        out.append(
            _sdr(
                f"CTRMEM-{tag}-STRME",
                "CTRMEM",
                "STRME1",
                "STQME2",
                est,
                grids,
                patch_displacements(grids),
                meta,
            )
        )
    for tag, (grids, mid, t, theta) in QUAD_GEOMS.items():
        eid += 1
        est = qdmem_est(eid, grids, mid, t, theta)
        meta = {"mid": mid, "t": t, "theta": theta}
        disp = patch_displacements(grids)
        out.append(
            Case(f"CQDMEM-{tag}-KQDMEM", "CQDMEM", "KQDMEM", SMA1, est, [4, *SILS], grids, meta)
        )
        out.append(
            Case(
                f"CQDMEM-{tag}-KTRIQD3",
                "CQDMEM",
                "KTRIQD3",
                SMA1,
                quad1_membrane_est(eid, grids, mid, t, theta),
                [4, *SILS],
                grids,
                meta,
            )
        )
        out.append(
            _sdr(f"CQDMEM-{tag}-SQDME", "CQDMEM", "SQDME1", "STQME2", est, grids, disp, meta)
        )
        for elem, (s, d), (p1, p2) in (
            ("CQDMEM1", ("QDMM1S", "QDMM1D"), ("SQDM11", "SQDM12")),
            ("CQDMEM2", ("QDMM2S", "QDMM2D"), ("SQDM21", "SQDM22")),
        ):
            for routine in (s, d):
                out.append(
                    Case(f"{elem}-{tag}-{routine}", elem, routine, EMG, est, [-1, 4], grids, meta)
                )
            out.append(_sdr(f"{elem}-{tag}-{p1[:4]}", elem, p1, p2, est, grids, disp, meta))
    return out


ALL_CASES = _build()
