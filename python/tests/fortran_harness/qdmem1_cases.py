"""CQDMEM1 harness cases: QDMM1D/QDMM1S, SQDM11/SQDM12 and the QDMM1 thermal load.

Complements the three basic-system CQDMEM1 cases in :mod:`cases` (whose golden
dumps stay untouched) with warped elements (both signs of the non-planarity),
material angles, non-structural mass, permuted SIL orders, grids in the
rectangular, cylindrical and spherical systems of :mod:`rod_cases`, every
branch of the ``SQDM11`` diagonal-intersection search, thermal stresses, and
the ``QDMM1`` thermal load (``build/qdmm1load``; at most one non-basic system
per element, preloaded into ``BASGLB``'s cache).

``python -m tests.fortran_harness.qdmem1_cases`` reruns these cases and rewrites
``golden/qdmem1_golden.json``.
"""

from __future__ import annotations

import json
import math
import sys

from . import cases as C
from . import rod_cases as RC
from . import runner as R

GOLDEN = R.HERE / "golden" / "qdmem1_golden.json"
LOAD_BINARY = R.HERE / "build" / "qdmm1load"
CSTM = RC.CSTM
QDMM1 = 5
NPG = 24
SDR = ("SQDM12", 62, 45, 8, 0)

_SQ = ((0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0))


def _basic(*pts):
    return tuple((0, p) for p in pts)


# tag: ((cid, xyz) x 4), mid, t, theta, nsm, SILs, element temperature
QDMEM1_GEOMS = {
    "square": (_basic(*_SQ), 1, 0.2, 0.0, 0.0, (1, 7, 13, 19), 0.0),
    "aniso30": (_basic((0, 0, 0), (5, 1, 0), (6, 4, 1), (1, 3, 1)), 3, 0.1, 30.0, 0.01,
                (1, 7, 13, 19), 0.0),
    "aniso-60": (_basic((1, 1, 1), (5, 2, 1), (5, 6, 3), (0, 4, 2)), 3, 0.15, -60.0, 0.0,
                 (13, 1, 19, 7), 0.0),
    "mat1-theta": (_basic((0, 0, 0), (4, 1, 0), (5, 5, 2), (1, 3, 2)), 2, 0.12, 25.0, 0.003,
                   (19, 13, 7, 1), 0.0),
    "warp+": (_basic((0, 0, 0), (4, 0, 0.5), (4, 3, 0), (0, 3, 0.5)), 1, 0.2, 0.0, 0.0,
              (1, 7, 13, 19), 0.0),
    "warp-": (_basic((0, 0, 0), (4, 0, -0.5), (4, 3, 0), (0, 3, -0.5)), 1, 0.2, 0.0, 0.0,
              (1, 7, 13, 19), 0.0),
    "warp-aniso": (_basic((0, 0, 0), (5, 0.5, 0.8), (6, 4, 0.1), (1, 3.5, 0.9)), 3, 0.08, 45.0,
                   0.002, (7, 19, 1, 13), 0.0),
    # SQDM11 (ETAS, XIS) branches: trapezoid (Y34 = 0), D above B (X24 = 0), C above A
    # (X13 = 0), BC parallel to AD (A2 = 0), X34 + X12 = 0 (A3 quadratic) and with D
    # above A (A3 = 0).
    "trap": (_basic((0, 0, 0), (6, 0, 0), (5, 4, 0), (1, 4, 0)), 1, 0.08, 0.0, 0.0,
             (1, 7, 13, 19), 0.0),
    "x24zero": (_basic((0, 0, 0), (4, 0, 0), (6, 2, 0), (4, 4, 0)), 2, 0.1, 0.0, 0.0,
                (1, 7, 13, 19), 0.0),
    "x13zero": (_basic((0, 0, 0), (4, 0, 0), (0, 3, 0), (-3, 2, 0)), 3, 0.1, 10.0, 0.0,
                (1, 7, 13, 19), 0.0),
    "a2zero": (_basic((0, 0, 0), (4, 0, 0), (5, 3, 0), (2, 6, 0)), 1, 0.1, 0.0, 0.0,
               (1, 7, 13, 19), 0.0),
    "a3": (_basic((0, 0, 0), (4, 0, 0), (5, 3, 0), (1, 5, 0)), 2, 0.1, 0.0, 0.0,
           (1, 7, 13, 19), 0.0),
    "a3zero": (_basic((0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 5, 0)), 1, 0.1, 0.0, 0.0,
               (1, 7, 13, 19), 0.0),
    "general3d": (_basic((0.2, 0.1, 0.0), (5.3, 0.7, 1.1), (6.1, 4.4, 2.3), (0.9, 3.2, 0.8)), 3,
                  0.11, 20.0, 0.0, (1, 7, 13, 19), 0.0),
    "rect10": (((10, (0, 0, 0)), (10, (4, 1, -1)), (10, (5, 4, 0)), (10, (1, 3, 1))), 1, 0.2, 0.0,
             0.01, (1, 7, 13, 19), 0.0),
    "rect-aniso": (((10, (1, 2, 0)), (0, (5, 1, 2)), (10, (6, 5, 2)), (0, (2, 6, 1))), 3, 0.1,
                   15.0, 0.0, (19, 13, 7, 1), 0.0),
    "cyl": (((20, (1, 2, 3)), (0, (5, 1, 4)), (20, (6, 5, 5)), (20, (2, 6, 4))), 2, 0.3, 0.0,
            0.003, (1, 7, 13, 19), 0.0),
    "cyl-axis": (((21, (0, 0, 5)), (21, (3, 0, 5)), (0, (3, 2, 6)), (21, (0, 3, 6))), 1, 0.3, 0.0,
                 0.0, (7, 1, 19, 13), 0.0),
    "sph": (((30, (2, 1, -1)), (30, (6, 2, 0)), (30, (7, 6, 1)), (30, (3, 5, 0))), 3, 0.25, 40.0,
            0.0, (1, 13, 7, 19), 0.0),
    "fp-rect": (((40, (0.1, 0.2, 0.3)), (40, (4.3, 0.9, 1.1)), (0, (5.2, 4.1, 1.7)),
                 (40, (0.7, 3.3, 0.6))), 1, 0.2, 0.0, 0.01, (1, 7, 13, 19), 0.0),
    "fp-cyl": (((41, (1.7, 2.3, 3.1)), (41, (6.1, 2.9, 3.3)), (41, (6.9, 6.2, 4.2)),
                (41, (2.2, 5.7, 3.9))), 3, 0.09, 60.0, 0.0, (7, 13, 1, 19), 0.0),
    "fp-sph": (((42, (2.3, 1.9, -1.3)), (0, (6.7, 2.1, -0.9)), (42, (7.1, 6.3, 0.4)),
                (42, (2.9, 5.1, -0.2))), 2, 0.4, 0.0, 0.004, (13, 7, 1, 19), 0.0),
    # Mixed systems: QDMM1D/SQDM11 only (BASGLB caches a single system).
    "mixed": (((10, (1, 1, 1)), (30, (6, 1, 2)), (20, (7, 5, 3)), (0, (2, 5, 2))), 3, 0.1, 30.0,
              0.02, (1, 7, 13, 19), 0.0),
}  # fmt: skip
TBAR = {1: 170.0, 2: -35.5, 3: 212.25}
ELTEMP = {1: 95.0, 2: 12.5, 3: -40.0}
THERMAL_STRESS = ("square", "aniso30", "warp+", "general3d", "cyl", "sph", "mixed")


def qdmem1_cstm_est(eid, grids, mid, t, theta, nsm, sils, eltemp=0.0) -> list:
    words: list = [eid, *sils, float(theta), mid, float(t), float(nsm)]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, float(eltemp)]


def displacements(seed: int, ndisp: int = NPG) -> list[float]:
    """A deterministic, non-rigid displacement vector (indexed by SIL)."""
    return [1.0e-3 * math.sin(1.7 * k + 0.3 * seed) + 2.0e-4 * math.cos(0.9 * k * seed) for k in
            range(ndisp)]  # fmt: skip


def _single_system(grids) -> bool:
    return len({cid for cid, _ in grids if cid}) <= 1


def _cases() -> tuple[list[C.Case], list[C.Case]]:
    cases: list[C.Case] = []
    loads: list[C.Case] = []
    eid = 900
    for tag, (grids, mid, t, theta, nsm, sils, eltemp) in QDMEM1_GEOMS.items():
        eid += 1
        est = qdmem1_cstm_est(eid, grids, mid, t, theta, nsm, sils, eltemp)
        meta = {"mid": mid, "t": t, "theta": theta, "nsm": nsm, "sils": sils}
        xyz = [p for _, p in grids]
        for routine in ("QDMM1S", "QDMM1D"):
            cases.append(
                C.Case(
                    f"CQDMEM1-{tag}-{routine}", "CQDMEM1", routine, C.EMG, est, [-1, 4], xyz, meta
                )
            )
        disp = displacements(eid)
        cases.append(
            C.Case(f"CQDMEM1-{tag}-SQDM", "CQDMEM1", "SQDM11", C.SDR2, est,
                   [*SDR, -1, 0.0, 0.0, len(disp), *disp], xyz, meta)
        )  # fmt: skip
        if tag in THERMAL_STRESS:
            temp = TBAR[mid]
            hot = qdmem1_cstm_est(eid, grids, mid, t, theta, nsm, sils, ELTEMP[mid])
            cases.append(
                C.Case(f"CQDMEM1-{tag}-SQDM-T", "CQDMEM1", "SQDM11", C.SDR2, hot,
                       [*SDR, 0, temp, 0.0, len(disp), *disp], xyz, {**meta, "temp": temp})
            )  # fmt: skip
        if _single_system(grids):
            tbar = TBAR[mid]
            loads.append(
                C.Case(f"CQDMEM1-{tag}-QDMM1", "CQDMEM1", "QDMM1", QDMM1, est, [tbar, NPG], xyz,
                       {**meta, "tbar": tbar})
            )  # fmt: skip
    return cases, loads


QDMEM1_CASES, QDMEM1_LOAD_CASES = _cases()


def run() -> list[R.CaseResult]:
    return R.run(QDMEM1_CASES, cstm=CSTM) + R.run(QDMEM1_LOAD_CASES, binary=LOAD_BINARY, cstm=CSTM)


def main() -> int:
    R.build()
    golden = R.to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
