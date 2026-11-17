"""SHEARD harness cases: flat, warped and skewed panels, every flexibility branch.

Complements the three basic-system shear cases in :mod:`cases` with all four
``Z`` formulas, warped and arbitrarily oriented panels, grids in the rod
cases' rectangular/cylindrical/spherical systems, reversed SIL order,
non-structural mass, and the panels ``SHEARD`` rejects (``MESAGE(30,26)`` bad
geometry, ``MESAGE(30,27)`` non-convex corner). Only ``SHEARD`` is run: the
single-precision ``SHEARS`` differs from it by up to ~3e-5.

``python -m tests.fortran_harness.shear_cases`` reruns these cases and rewrites
``golden/shear_golden.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import runner as R

GOLDEN = R.HERE / "golden" / "shear_golden.json"
CSTM = RC.CSTM

_ROT = RC._rot(25.0, -40.0, 65.0)


def _rotated(points, shift=(1.5, -2.0, 0.75)):
    return [tuple(float(v) for v in _ROT @ np.asarray(p, float) + shift) for p in points]


def _basic(points):
    return [(0, tuple(map(float, p))) for p in points]


# tag: (((cid, xyz) x 4), mid, T, NSM, SILs)
_D = (1, 7, 13, 19)
SHEAR_CSTM_GEOMS = {
    # Both pairs of edges parallel (label 50).
    "square": (_basic([(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]), 1, 0.1, 0.0, _D),
    "parallelogram": (_basic([(0, 0, 0), (6, 0, 0), (8, 3, 0), (2, 3, 0)]), 2, 0.05, 0.0, _D),
    "near-parallel": (
        _basic([(0, 0, 0), (10, 0, 0), (10.02, 5.03, 0), (0.01, 5, 0)]),
        1,
        0.08,
        0.0,
        _D,
    ),
    # Edge 3-4 parallel to 1-2 (label 30) and edge 1-4 parallel to 2-3 (label 40).
    "trap34": (_basic([(0, 0, 0), (8, 0, 0), (6, 3, 0), (1, 3, 0)]), 1, 0.1, 0.002, _D),
    "trap14": (_basic([(0, 0, 0), (5, 0, 0), (6, 4, 0), (0.5, 2, 0)]), 2, 0.07, 0.0, _D),
    # General quadrilaterals (label 70).
    "general": (_basic([(0, 0, 0), (7, 0.5, 0), (6, 5, 0), (-1, 4, 0)]), 1, 0.12, 0.0, _D),
    "kite": (_basic([(0, 0, 0), (5, -2, 0), (9, 0, 0), (5, 3, 0)]), 2, 0.06, 0.001, _D),
    # Warped panels: the corners are off the mean plane.
    "warped": (_basic([(0, 0, 0), (8, 0, 0.6), (9, 6, -0.4), (1, 5, 0.8)]), 1, 0.1, 0.0, _D),
    "warped-rect": (
        _basic([(0, 0, 0.3), (10, 0, -0.3), (10, 5, 0.3), (0, 5, -0.3)]),
        2,
        0.05,
        0.0,
        _D,
    ),
    "warped-trap": (
        _basic([(0, 0, 0), (8, 0, 0.5), (6, 3, -0.5), (1, 3, 0.5)]),
        1,
        0.09,
        0.003,
        _D,
    ),
    # Skewed panels in an arbitrarily oriented plane.
    "skew-general": (
        _basic(_rotated([(0, 0, 0), (7, 0.5, 0), (6, 5, 0), (-1, 4, 0)])),
        1,
        0.1,
        0.0,
        _D,
    ),
    "skew-trap14": (
        _basic(_rotated([(0, 0, 0), (5, 0, 0), (6, 4, 0), (0.5, 2, 0)])),
        2,
        0.04,
        0.0,
        _D,
    ),
    "skew-warped": (
        _basic(_rotated([(0, 0, 0), (8, 0, 0.6), (9, 6, -0.4), (1, 5, 0.8)])),
        1,
        0.15,
        0.002,
        _D,
    ),
    # Reversed / shuffled SIL order.
    "general-rev": (
        _basic([(0, 0, 0), (7, 0.5, 0), (6, 5, 0), (-1, 4, 0)]),
        1,
        0.12,
        0.004,
        (19, 13, 7, 1),
    ),
    "warped-shuffled": (
        _basic([(0, 0, 0), (8, 0, 0.6), (9, 6, -0.4), (1, 5, 0.8)]),
        2,
        0.1,
        0.0,
        (13, 1, 19, 7),
    ),
    # Grids in rectangular, cylindrical and spherical displacement systems.
    "cstm": (
        [(10, (0.0, 0.0, 0.0)), (20, (8.0, 0.0, 0.6)), (30, (9.0, 6.0, -0.4)), (0, (1, 5, 0.8))],
        1,
        0.1,
        0.001,
        _D,
    ),
    "cstm-fp": (
        [(40, (0.1, 0.2, 0.3)), (41, (7.1, 0.7, 0.2)), (42, (6.3, 5.1, -0.1)), (40, (-1, 4, 0))],
        2,
        0.08,
        0.0,
        (7, 19, 1, 13),
    ),
    "cstm-axis": (
        [(21, (0.0, 0.0, 0.0)), (21, (4.0, 0.0, 0.0)), (10, (4.0, 4.0, 0.0)), (21, (0, 4, 0))],
        1,
        0.1,
        0.0,
        _D,
    ),
}

# Panels SHEARD rejects, plus two darts its projected-corner tests accept.
_SQ = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]
SHEAR_ERROR_GEOMS = {
    "zero-t": (_basic(_SQ), 1, 0.0, 0.0, _D),
    "collinear": (_basic([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]), 1, 0.1, 0.0, _D),
    "coincident-12": (_basic([(0, 0, 0), (0, 0, 0), (5, 5, 0), (0, 5, 0)]), 1, 0.1, 0.0, _D),
    "reentrant-1": (_basic([(3, 3, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]), 1, 0.1, 0.0, _D),
    "reentrant-2": (_basic([(0, 0, 0), (7, 3, 0), (10, 10, 0), (0, 10, 0)]), 1, 0.1, 0.0, _D),
    "reentrant-3": (_basic([(0, 0, 0), (10, 0, 0), (3, 3, 0), (0, 10, 0)]), 1, 0.1, 0.0, _D),
    "reentrant-4": (_basic([(0, 0, 0), (10, 0, 0), (10, 10, 0), (7, 3, 0)]), 1, 0.1, 0.0, _D),
    "corner3-below-12": (_basic([(0, 0, 0), (10, 0, 0), (5, -1, 0), (0, 10, 0)]), 1, 0.1, 0.0, _D),
    "corner4-below-12": (_basic([(0, 0, 0), (10, 0, 0), (10, 10, 0), (5, -1, 0)]), 2, 0.1, 0.0, _D),
    "bowtie": (_basic([(0, 0, 0), (10, 0, 0), (0, 5, 0), (10, 5, 0)]), 1, 0.1, 0.0, _D),
}


def shear_cstm_est(eid, grids, mid, t, nsm, sils) -> list:
    words: list = [eid, *sils, mid, t, nsm]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, 0.0]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 700
    for group in (SHEAR_CSTM_GEOMS, SHEAR_ERROR_GEOMS):
        for tag, (grids, mid, t, nsm, sils) in group.items():
            eid += 1
            meta = {"mid": mid, "t": t, "nsm": nsm, "sils": sils}
            out.append(
                C.Case(
                    f"CSHEAR-{tag}-SHEARD",
                    "CSHEAR",
                    "SHEARD",
                    C.EMG,
                    shear_cstm_est(eid, grids, mid, t, nsm, sils),
                    [-1, 4],
                    [xyz for _, xyz in grids],
                    meta,
                )
            )
    return out


SHEAR_CASES = _cases()


def _messages(stdout: str) -> list[dict]:
    """Per case: the ``MESAGE`` numbers and the ``IECPT(2)`` word printed with ``NOGO``."""
    out: list[dict] = []
    for line in stdout.splitlines():
        head = line.split()
        if not head:
            continue
        if head[0] == "CASE":
            out.append({})
        elif head[0] == "MESAGE":
            out[-1].setdefault("mesage", []).append([int(head[1]), int(head[2]), int(head[3])])
        elif head[0] == "NOGO":
            out[-1]["nogo"] = int(head[1])
    return out


def run() -> tuple[list[R.CaseResult], list[dict]]:
    proc = subprocess.run(
        [str(R.BINARY)],
        input=R.stdin_for(SHEAR_CASES, CSTM),
        capture_output=True,
        text=True,
        check=True,
    )
    results = R.parse(proc.stdout, SHEAR_CASES)
    msgs = _messages(proc.stdout)
    if not len(results) == len(msgs) == len(SHEAR_CASES):
        raise RuntimeError(f"harness produced {len(results)} of {len(SHEAR_CASES)} cases")
    return results, msgs


def to_golden(results: list[R.CaseResult], msgs: list[dict]) -> dict:
    golden = R.to_golden(results)
    for r, m in zip(results, msgs, strict=True):
        golden[r.case.name].update(m)
    return golden


def main() -> int:
    R.build()
    golden = to_golden(*run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
