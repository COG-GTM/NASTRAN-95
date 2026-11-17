"""CTRMEM harness cases: ``TRMEMD`` in offset/rotated systems and the ``TRIMEM`` thermal load.

Complements the basic-system CTRMEM cases in :mod:`cases` (whose golden dumps
stay untouched) with grids in the rectangular, cylindrical and spherical
displacement systems of :mod:`rod_cases`, permuted SIL orders, material angles
and non-structural mass for the Python ``nastran95py.elements.trmem`` parity
tests. ``TRIMEM`` cases use at most one non-basic system per element (the
driver preloads that system into ``BASGLB``'s cache instead of a CSTM file).

``python -m tests.fortran_harness.trmem_cases`` reruns these cases and rewrites
``golden/trmem_golden.json``.
"""

from __future__ import annotations

import json
import sys

from . import cases as C
from . import rod_cases as RC
from . import runner as R

GOLDEN = R.HERE / "golden" / "trmem_golden.json"
CSTM = RC.CSTM
TRIMEM = 4

# tag: ((cid, xyz) x 3), mid, t, theta, nsm, (SIL1, SIL2, SIL3)
TRMEM_GEOMS = {
    "plane": (((0, (0, 0, 0)), (0, (4, 0, 0)), (0, (1, 3, 0))), 1, 0.2, 0.0, 0.0, (1, 7, 13)),
    "aniso30": (((0, (1, 1, 1)), (0, (5, 2, 1)), (0, (2, 6, 3))), 3, 0.1, 30.0, 0.01, (1, 7, 13)),
    "aniso-45": (
        ((0, (0, 0, 0)), (0, (3, 1, 2)), (0, (-1, 4, 1))),
        3,
        0.15,
        -45.0,
        0.0,
        (7, 13, 1),
    ),
    "aniso90": (
        ((0, (0.5, 0, 0)), (0, (3, 0, 1)), (0, (1, 2, 4))),
        3,
        0.08,
        90.0,
        0.002,
        (13, 1, 7),
    ),
    "aniso180": (((0, (0, 0, 0)), (0, (2, 0, 0)), (0, (0, 5, 0))), 3, 0.05, 180.0, 0.0, (1, 7, 13)),
    "mat1-theta": (
        ((0, (0, 0, 0)), (0, (4, 1, 0)), (0, (1, 3, 2))),
        2,
        0.12,
        25.0,
        0.003,
        (1, 7, 13),
    ),
    "rect": (((10, (0, 0, 0)), (10, (4, 3, -2)), (10, (1, 5, 2))), 1, 0.2, 0.0, 0.01, (1, 7, 13)),
    "rect-aniso": (
        ((10, (1, 2, 0)), (0, (5, 1, 2)), (10, (2, 6, 1))),
        3,
        0.1,
        15.0,
        0.0,
        (13, 7, 1),
    ),
    "cyl": (((20, (1, 2, 3)), (0, (4, -2, 8)), (20, (6, 3, 1))), 2, 0.75, 0.0, 0.003, (1, 7, 13)),
    "cyl-axis": (((21, (0, 0, 5)), (21, (3, 4, 9)), (0, (-2, 1, 6))), 1, 0.3, 0.0, 0.0, (7, 1, 13)),
    "sph": (((30, (2, 1, -1)), (30, (-3, 4, 6)), (30, (1, -2, 5))), 3, 0.25, 40.0, 0.0, (1, 13, 7)),
    "fp-rect": (
        ((40, (0.1, 0.2, 0.3)), (40, (4.3, -2.9, 6.1)), (0, (2, 3, 1))),
        1,
        0.2,
        0.0,
        0.01,
        (1, 7, 13),
    ),
    "fp-cyl": (
        ((41, (1.7, 2.3, 3.1)), (41, (-4.1, 0.9, 7.3)), (41, (0.3, -1.2, 2.2))),
        3,
        0.09,
        60.0,
        0.0,
        (7, 13, 1),
    ),
    "fp-sph": (
        ((42, (2.3, 1.9, -1.3)), (0, (-3.7, 4.1, 6.9)), (42, (1.1, -2.1, 0.4))),
        2,
        0.4,
        0.0,
        0.004,
        (13, 7, 1),
    ),
    # Mixed systems: TRANSD (stiffness) only, BASGLB caches a single system.
    "mixed": (((10, (1, 1, 1)), (30, (6, -1, 4)), (20, (2, 5, 3))), 3, 0.1, 30.0, 0.02, (1, 7, 13)),
    "fp-mixed": (
        ((40, (1.3, 1.1, 0.7)), (42, (6.1, -1.7, 4.3)), (0, (2.2, 4.4, 1.5))),
        1,
        0.15,
        0.0,
        0.0,
        (13, 1, 7),
    ),
}
TBAR = {1: 170.0, 2: -35.5, 3: 212.25}
NPG = 15


def trmem_cstm_est(eid, grids, mid, t, theta, nsm, sils) -> list:
    words: list = [eid, *sils, float(theta), mid, float(t), float(nsm)]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, 0.0]


def _single_system(grids) -> bool:
    return len({cid for cid, _ in grids if cid}) <= 1


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 700
    for tag, (grids, mid, t, theta, nsm, sils) in TRMEM_GEOMS.items():
        eid += 1
        est = trmem_cstm_est(eid, grids, mid, t, theta, nsm, sils)
        meta = {"mid": mid, "t": t, "theta": theta, "nsm": nsm, "sils": sils}
        xyz = [p for _, p in grids]
        out.append(
            C.Case(f"CTRMEM-{tag}-TRMEMD", "CTRMEM", "TRMEMD", C.EMG, est, [-1, 3], xyz, meta)
        )
        if _single_system(grids):
            tbar = TBAR[mid]
            out.append(
                C.Case(
                    f"CTRMEM-{tag}-TRIMEM",
                    "CTRMEM",
                    "TRIMEM",
                    TRIMEM,
                    est,
                    [tbar, NPG],
                    xyz,
                    {**meta, "tbar": tbar},
                )
            )
    return out


TRMEM_CASES = _cases()


def run() -> list[R.CaseResult]:
    return R.run(TRMEM_CASES, cstm=CSTM)


def main() -> int:
    R.build()
    golden = R.to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
