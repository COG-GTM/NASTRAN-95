"""CQDMEM2 harness cases: ``QDMM2D/S`` and ``SQDM21/22`` in offset/rotated systems, ``QDMM2``.

Complements the basic-system CQDMEM2 cases in :mod:`cases` (whose golden dumps
stay untouched) with warped (non-planar) quadrilaterals, permuted SIL orders,
material angles, non-structural mass, grids in the rectangular, cylindrical and
spherical displacement systems of :mod:`rod_cases`, SDR2 stress recovery with
and without a load temperature, and the ``QDMM2`` thermal load (``build/qdmm2load``,
``mis/qdmm2.f`` with the ``BASGLB`` CSTM read served by ``loadstubs.f``).

``python -m tests.fortran_harness.qdmem2_cases`` rebuilds the harness, reruns
these cases and rewrites ``golden/qdmem2_golden.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import rod_load_cases as RLC
from . import runner as R

GOLDEN = R.HERE / "golden" / "qdmem2_golden.json"
LOAD_BINARY = R.HERE / "build" / "qdmm2load"
CSTM = RC.CSTM
QDMM2 = 5
NDISP = 24

# tag: ((cid, xyz) x 4), mid, t, theta, nsm, (SIL A..D)
QDMEM2_GEOMS = {
    "plane-nsm": (
        ((0, (0, 0, 0)), (0, (4, 0, 0)), (0, (4.5, 3, 0)), (0, (0.5, 2.5, 0))),
        1,
        0.2,
        0.0,
        0.01,
        (1, 7, 13, 19),
    ),
    "warped": (
        ((0, (0, 0, 0)), (0, (4, 0, 1)), (0, (4, 3, 0)), (0, (0, 3, 1))),
        1,
        0.1,
        0.0,
        0.0,
        (1, 7, 13, 19),
    ),
    "warped-aniso30": (
        ((0, (1, 1, 0)), (0, (6, 0.5, 1.5)), (0, (6.5, 5, 0.2)), (0, (0.5, 4, 1.2))),
        3,
        0.15,
        30.0,
        0.002,
        (13, 1, 19, 7),
    ),
    "mat1-theta": (
        ((0, (0, 0, 0)), (0, (5, 1, 0)), (0, (4, 4, 0.5)), (0, (-1, 3, 0.25))),
        2,
        0.12,
        25.0,
        0.003,
        (7, 13, 19, 1),
    ),
    "trap-aniso-60": (
        ((0, (0, 0, 0)), (0, (8, 0, 0)), (0, (5, 3, 0)), (0, (2, 3, 0))),
        3,
        0.08,
        -60.0,
        0.004,
        (19, 13, 7, 1),
    ),
    "rect-cs": (
        ((10, (0, 0, 0)), (10, (4, 1, -1)), (10, (5, 5, 0)), (10, (1, 4, 1))),
        1,
        0.2,
        0.0,
        0.01,
        (1, 7, 13, 19),
    ),
    "cyl": (
        ((20, (1, 2, 3)), (0, (5, 1, 4)), (20, (6, 5, 3.5)), (0, (2, 6, 2.5))),
        2,
        0.25,
        10.0,
        0.0,
        (1, 13, 7, 19),
    ),
    "cyl-axis": (
        ((21, (0, 0, 2)), (21, (3, 0, 2.5)), (21, (3, 4, 2)), (0, (0, 4, 2.5))),
        1,
        0.3,
        0.0,
        0.0,
        (7, 1, 19, 13),
    ),
    "sph": (
        ((30, (2, 1, -1)), (30, (6, 0, 1)), (30, (7, 4, 0)), (30, (3, 5, -2))),
        3,
        0.2,
        45.0,
        0.001,
        (1, 7, 13, 19),
    ),
    "fp-mixed": (
        ((40, (0.3, 0.2, 0.1)), (41, (4.3, -0.9, 1.1)), (42, (5.1, 3.7, 0.6)), (0, (1, 3.2, -0.4))),
        3,
        0.11,
        75.0,
        0.0,
        (19, 1, 13, 7),
    ),
    "mixed": (
        ((10, (1, 1, 1)), (20, (6, 0, 2)), (30, (7, 5, 1.5)), (0, (2, 6, 0.5))),
        2,
        0.14,
        -20.0,
        0.02,
        (13, 19, 1, 7),
    ),
}
TEMPERATURES = {1: 170.0, 2: -35.5, 3: 212.25}


def qdmem2_cstm_est(eid, grids, mid, t, theta, nsm, sils) -> list:
    words: list = [eid, *sils, float(theta), mid, float(t), float(nsm)]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, 0.0]


def displacements(seed: int) -> list[float]:
    """A deterministic, non-uniform displacement vector over SILs 1-24."""
    rng = np.random.default_rng(seed)
    return [float(np.float32(v)) for v in rng.uniform(-1.0e-3, 1.0e-3, NDISP)]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eltype, nwdsa, nwdstr, nwdfor = C.SDR_WORDS["CQDMEM2"]
    eid = 800
    for tag, (grids, mid, t, theta, nsm, sils) in QDMEM2_GEOMS.items():
        eid += 1
        est = qdmem2_cstm_est(eid, grids, mid, t, theta, nsm, sils)
        temp = TEMPERATURES[mid]
        meta = {"mid": mid, "t": t, "theta": theta, "nsm": nsm, "sils": sils}
        xyz = [p for _, p in grids]
        for routine in ("QDMM2S", "QDMM2D"):
            out.append(
                C.Case(
                    f"CQDMEM2-{tag}-{routine}", "CQDMEM2", routine, C.EMG, est, [-1, 4], xyz, meta
                )
            )
        disp = displacements(eid)
        for label, ldtemp, tval in (("SQDM", -1, 0.0), ("SQDM-temp", 1, temp)):
            extra = ["SQDM22", eltype, nwdsa, nwdstr, nwdfor, ldtemp, tval, 0.0, NDISP, *disp]
            out.append(
                C.Case(
                    f"CQDMEM2-{tag}-{label}",
                    "CQDMEM2",
                    "SQDM21",
                    C.SDR2,
                    est,
                    extra,
                    xyz,
                    {**meta, "temperature": tval if ldtemp == 1 else None, "disp": disp},
                )
            )
        out.append(
            C.Case(
                f"CQDMEM2-{tag}-QDMM2",
                "CQDMEM2",
                "QDMM2",
                QDMM2,
                est,
                [temp, NDISP],
                xyz,
                {**meta, "temperature": temp},
            )
        )
    return out


QDMEM2_CASES = _cases()
HARNESS_CASES = [c for c in QDMEM2_CASES if c.kind != QDMM2]
LOAD_CASES = [c for c in QDMEM2_CASES if c.kind == QDMM2]


def run_loads(cases: list[C.Case] | None = None, binary=LOAD_BINARY) -> list[np.ndarray]:
    cases = LOAD_CASES if cases is None else cases
    proc = subprocess.run(
        [str(binary)], input=R.stdin_for(cases, CSTM), capture_output=True, text=True, check=True
    )
    loads = RLC.parse(proc.stdout)
    if len(loads) != len(cases):
        raise RuntimeError(f"harness produced {len(loads)} of {len(cases)} cases:\n{proc.stdout}")
    return loads


def run() -> dict:
    """Golden entries of every case: harness dumps plus ``{"load": [...]}`` for ``QDMM2``."""
    golden = R.to_golden(R.run(HARNESS_CASES, cstm=CSTM))
    for c, v in zip(LOAD_CASES, run_loads(), strict=True):
        golden[c.name] = {
            "element": c.element,
            "routine": c.routine,
            "kind": c.kind,
            "load": v.tolist(),
        }
    return golden


def main() -> int:
    R.build()
    golden = run()
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
