"""RODD harness cases with grids in offset/rotated coordinate systems.

Complements the basic-system rod cases in :mod:`cases` (whose golden dumps
stay untouched) with rectangular, cylindrical and spherical displacement
systems, reversed SIL order and a torsion-only rod, for the Python
``nastran95py.elements.rod`` parity tests.

``python -m tests.fortran_harness.rod_cases`` reruns these cases and rewrites
``golden/rod_golden.json``.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from nastran95py.coords import CoordSystem

from . import cases as C
from . import runner as R

GOLDEN = R.HERE / "golden" / "rod_golden.json"


def _rot(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (
        f(math.radians(a)) for a in (ax, ay, az) for f in (math.cos, math.sin)
    )
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _cs(cid: int, ctype: int, origin, t) -> tuple[int, int, list[float]]:
    """CSTM record (origin, then T row-wise) rounded to the REAL words the harness reads."""
    words = np.concatenate([np.asarray(origin, float), np.asarray(t, float).reshape(9)])
    return cid, ctype, [float(np.float32(w)) for w in words]


# (CID, TYPE 1/2/3 = rectangular/cylindrical/spherical, origin, T: local -> basic)
CSTM = [
    _cs(10, 1, (1.0, -2.0, 3.0), _rot(20.0, -35.0, 50.0)),
    _cs(20, 2, (0.5, 0.25, -1.0), _rot(-15.0, 25.0, 40.0)),
    _cs(21, 2, (0.0, 0.0, 0.0), _rot(0.0, 0.0, 30.0)),
    _cs(30, 3, (-2.0, 1.0, 0.5), _rot(10.0, 60.0, -20.0)),
]

# Systems defined by three points (as CORD2R/C/S) and kept in float64: the Python
# side uses them unrounded, the harness gets their REAL words.
FROM_POINTS = {
    40: CoordSystem.from_points(40, 1, (0.3, -1.1, 2.7), (1.9, 0.4, 5.3), (4.1, -2.2, 1.7)),
    41: CoordSystem.from_points(41, 2, (-0.7, 0.9, 1.3), (0.6, 2.1, 4.9), (3.3, -1.4, 0.2)),
    42: CoordSystem.from_points(42, 3, (1.1, 0.7, -0.9), (-1.3, 3.1, 2.3), (2.9, 4.7, -3.1)),
}
CSTM += [_cs(cid, int(cs.ctype), cs.origin, cs.t) for cid, cs in FROM_POINTS.items()]

# tag: ((cidA, xyzA), (cidB, xyzB)), mid, A, J, C, NSM, (SIL1, SIL2)
ROD_CSTM_GEOMS = {
    "rect": (((10, (0.0, 0.0, 0.0)), (10, (4.0, 3.0, -2.0))), 1, 2.0, 0.5, 0.1, 0.01, (1, 7)),
    "cyl": (((20, (1.0, 2.0, 3.0)), (0, (4.0, -2.0, 8.0))), 2, 0.75, 0.2, 0.05, 0.003, (1, 7)),
    "sph": (((30, (2.0, 1.0, -1.0)), (30, (-3.0, 4.0, 6.0))), 1, 1.25, 0.4, 0.0, 0.0, (1, 7)),
    "mixed": (((10, (1.0, 1.0, 1.0)), (30, (6.0, -1.0, 4.0))), 2, 0.5, 0.1, 0.0, 0.02, (1, 7)),
    "axis": (((21, (0.0, 0.0, 5.0)), (21, (3.0, 4.0, 9.0))), 1, 1.0, 0.25, 0.0, 0.0, (1, 7)),
    "mixed-rev": (((10, (1.0, 1.0, 1.0)), (30, (6.0, -1.0, 4.0))), 2, 0.5, 0.1, 0.0, 0.02, (7, 1)),
    "skew-rev": (((0, (1.0, 2.0, 3.0)), (20, (4.0, -2.0, 8.0))), 2, 0.75, 0.2, 0.05, 0.0, (13, 7)),
    "torsion": (((10, (0.0, 0.0, 0.0)), (20, (4.0, 3.0, -2.0))), 1, 0.0, 0.3, 0.0, 0.004, (1, 7)),
    "fp-rect": (((40, (0.1, 0.2, 0.3)), (40, (4.3, -2.9, 6.1))), 1, 2.0, 0.5, 0.0, 0.01, (1, 7)),
    "fp-cyl": (((41, (1.7, 2.3, 3.1)), (41, (-4.1, 0.9, 7.3))), 2, 0.75, 0.2, 0.0, 0.003, (1, 7)),
    "fp-sph": (((42, (2.3, 1.9, -1.3)), (42, (-3.7, 4.1, 6.9))), 1, 1.25, 0.4, 0.0, 0.0, (7, 1)),
    "fp-mixed": (((40, (1.3, 1.1, 0.7)), (42, (6.1, -1.7, 4.3))), 2, 0.5, 0.1, 0.0, 0.02, (1, 7)),
    "axial": (((0, (0.0, 0.0, 0.0)), (10, (0.0, 0.0, 25.0))), 1, 1.5, 0.0, 0.0, 0.002, (7, 1)),
}


def rod_cstm_est(eid, grids, mid, area, j, c, nsm, sils) -> list:
    words: list = [eid, *sils, mid, area, j, c, nsm]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, 0.0]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 500
    for elem in ("CROD", "CONROD"):
        for tag, (grids, mid, area, j, c, nsm, sils) in ROD_CSTM_GEOMS.items():
            eid += 1
            est = rod_cstm_est(eid, grids, mid, area, j, c, nsm, sils)
            meta = {"mid": mid, "area": area, "j": j, "c": c, "nsm": nsm, "sils": sils}
            for icmbar in (-1, 1):
                out.append(
                    C.Case(
                        f"{elem}-{tag}-RODD-m{icmbar:+d}",
                        elem,
                        "RODD",
                        C.EMG,
                        est,
                        [icmbar, 2],
                        [xyz for _, xyz in grids],
                        meta,
                    )
                )
    return out


ROD_CSTM_CASES = _cases()


def run() -> list[R.CaseResult]:
    return R.run(ROD_CSTM_CASES, cstm=CSTM)


def main() -> int:
    R.build()
    golden = R.to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
