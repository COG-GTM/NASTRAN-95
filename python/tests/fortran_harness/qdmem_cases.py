"""CQDMEM harness cases: stiffness, mass, thermal load and stress recovery in coordinate systems.

Complements the basic-system CQDMEM cases in :mod:`cases` (``KQDMEM`` and
``SQDME1`` with the uniform-strain patch, whose golden dumps stay untouched)
with flat, warped and skewed quadrilaterals whose grids sit in the
rectangular, cylindrical and spherical displacement systems of
:mod:`rod_cases`, permuted SIL orders, MAT1 and anisotropic MAT2 at several
material angles and non-structural mass:

- ``KQDMEM`` (SMA1, one call per pivot; ``QDMEMS``/``QDMEMD`` reach it through
  ``EMGOLD``) and ``SQDME1``/``STQME2`` with general displacements, a load
  temperature, a thermal-only and a zero-stress case, on ``build/harness``;
- the thermal load ``QDMEM`` (four ``TRIMEM (1)`` calls) and the lumped mass
  ``MASSTQ (1)`` on ``build/qdmemx`` (``qdmemx.f``), whose ``BASGLB`` reads the
  CSTM through the ``CSTMRD`` stub, so mixed systems are allowed.

``python -m tests.fortran_harness.qdmem_cases`` rebuilds the harnesses, reruns
these cases and rewrites ``golden/qdmem_golden.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import runner as R
from . import trmem_stress_cases as SC

GOLDEN = R.HERE / "golden" / "qdmem_golden.json"
BINARY = R.HERE / "build" / "qdmemx"
CSTM = RC.CSTM
QDMEM_LOAD, MASSTQ = 4, 5
NDISP = 24
NPG = 24

# tag: ((cid, xyz) x 4), mid, t, theta, nsm, (SIL1..SIL4)
QDMEM_GEOMS = {
    "plane": (
        ((0, (0, 0, 0)), (0, (4, 0, 0)), (0, (4.5, 3, 0)), (0, (0.5, 2.5, 0))),
        1,
        0.2,
        0.0,
        0.0,
        (1, 7, 13, 19),
    ),
    "warped30": (
        ((0, (1, 1, 1)), (0, (5, 2, 1.4)), (0, (5.5, 6, 2.6)), (0, (1.5, 5, 0.8))),
        3,
        0.1,
        30.0,
        0.01,
        (19, 1, 13, 7),
    ),
    "skew-45": (
        ((0, (0, 0, 0)), (0, (3, 1, 2)), (0, (2, 5, 3)), (0, (-1, 4, 1))),
        3,
        0.15,
        -45.0,
        0.0,
        (7, 13, 19, 1),
    ),
    "trap90": (
        ((0, (0, 0, 0)), (0, (6, 0, 0)), (0, (5, 4, 0.3)), (0, (1, 4, -0.3))),
        3,
        0.08,
        90.0,
        0.002,
        (13, 19, 1, 7),
    ),
    "mat1-theta": (
        ((0, (0, 0, 0)), (0, (4, 1, 0)), (0, (4.5, 4, 1)), (0, (1, 3, 2))),
        2,
        0.12,
        25.0,
        0.003,
        (1, 13, 7, 19),
    ),
    "rect-sys": (
        ((10, (0, 0, 0)), (10, (4, 3, -2)), (10, (5, 7, 1)), (10, (1, 5, 2))),
        1,
        0.2,
        0.0,
        0.01,
        (1, 7, 13, 19),
    ),
    "rect-aniso": (
        ((10, (1, 2, 0)), (0, (5, 1, 2)), (10, (6, 5, 2.5)), (0, (2, 6, 1))),
        3,
        0.1,
        15.0,
        0.0,
        (19, 13, 7, 1),
    ),
    "cyl": (
        ((20, (1, 2, 3)), (20, (4, -2, 8)), (20, (8, 1, 6)), (20, (6, 3, 1))),
        2,
        0.75,
        0.0,
        0.003,
        (1, 7, 13, 19),
    ),
    "cyl-axis": (
        ((21, (0, 0, 5)), (21, (3, 4, 9)), (0, (1, 6, 10)), (21, (-2, 1, 6))),
        1,
        0.3,
        10.0,
        0.0,
        (7, 1, 19, 13),
    ),
    "sph": (
        ((30, (2, 1, -1)), (30, (-3, 4, 6)), (30, (0, 3, 9)), (30, (1, -2, 5))),
        3,
        0.25,
        40.0,
        0.0,
        (1, 19, 7, 13),
    ),
    "fp-rect": (
        ((40, (0.1, 0.2, 0.3)), (40, (4.3, -2.9, 6.1)), (0, (5.1, 1.7, 5.3)), (40, (2, 3, 1))),
        1,
        0.2,
        0.0,
        0.01,
        (1, 7, 13, 19),
    ),
    "fp-cyl": (
        (
            (41, (1.7, 2.3, 3.1)),
            (41, (-4.1, 0.9, 7.3)),
            (41, (-3.2, -2.6, 5.9)),
            (41, (0.3, -1.2, 2.2)),
        ),
        3,
        0.09,
        60.0,
        0.0,
        (7, 13, 1, 19),
    ),
    "fp-sph": (
        (
            (42, (2.3, 1.9, -1.3)),
            (0, (-3.7, 4.1, 6.9)),
            (42, (-1.9, -0.7, 5.3)),
            (42, (1.1, -2.1, 0.4)),
        ),
        2,
        0.4,
        -20.0,
        0.004,
        (13, 7, 19, 1),
    ),
    "mixed": (
        ((10, (1, 1, 1)), (30, (6, -1, 4)), (20, (7, 4, 5)), (40, (2, 5, 3))),
        3,
        0.1,
        30.0,
        0.02,
        (1, 7, 13, 19),
    ),
}
TBAR = {1: 170.0, 2: -35.5, 3: 212.25}


def qdmem_cstm_est(eid, grids, mid, t, theta, nsm, sils, temp=0.0) -> list:
    words: list = [eid, *sils, float(theta), mid, float(t), float(nsm)]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, float(temp)]


def _meta(mid, t, theta, nsm, sils, **kw) -> dict:
    return {"mid": mid, "t": t, "theta": theta, "nsm": nsm, "sils": sils, **kw}


def _cases() -> tuple[list[C.Case], list[C.Case]]:
    """``(harness cases, qdmemx cases)``."""
    harness: list[C.Case] = []
    aux: list[C.Case] = []
    eltype, nwdsa, nwdstr, nwdfor = C.SDR_WORDS["CQDMEM"]
    eid = 1600
    for tag, (grids, mid, t, theta, nsm, sils) in QDMEM_GEOMS.items():
        eid += 1
        est = qdmem_cstm_est(eid, grids, mid, t, theta, nsm, sils)
        xyz = [p for _, p in grids]
        meta = _meta(mid, t, theta, nsm, sils)
        harness.append(
            C.Case(f"CQDMEM-{tag}-KQDMEM", "CQDMEM", "KQDMEM", C.SMA1, est, [4, *sils], xyz, meta)
        )
        aux.append(
            C.Case(f"CQDMEM-{tag}-MASSTQ", "CQDMEM", "MASSTQ", MASSTQ, est, [4, *sils], xyz, meta)
        )
        tbar = TBAR[mid]
        aux.append(
            C.Case(
                f"CQDMEM-{tag}-QDMEM",
                "CQDMEM",
                "QDMEM",
                QDMEM_LOAD,
                est,
                [tbar, NPG],
                xyz,
                {**meta, "tbar": tbar},
            )
        )
        for load, (ldtemp, temp, moving) in SC.LOADS.items():
            sid = eid + 100 * (1 + list(SC.LOADS).index(load))
            words = qdmem_cstm_est(sid, grids, mid, t, theta, nsm, sils)
            disp = SC.displacements(sid, NDISP) if moving else [0.0] * NDISP
            extra = ["STQME2", eltype, nwdsa, nwdstr, nwdfor, ldtemp, temp, 0.0, NDISP, *disp]
            smeta = {
                **meta,
                "temperature": None if ldtemp == -1 else temp,
                "displacements": disp,
            }
            harness.append(
                C.Case(f"CQDMEM-{tag}-{load}", "CQDMEM", "SQDME1", C.SDR2, words, extra, xyz, smeta)
            )
    return harness, aux


HARNESS_CASES, AUX_CASES = _cases()
QDMEM_CASES = HARNESS_CASES + AUX_CASES


def run_aux(cases: list[C.Case] | None = None, binary=BINARY) -> list[R.CaseResult]:
    cases = AUX_CASES if cases is None else cases
    proc = subprocess.run(
        [str(binary)],
        input=R.stdin_for(cases, CSTM),
        capture_output=True,
        text=True,
        check=True,
    )
    results = R.parse(proc.stdout, cases)
    if len(results) != len(cases):
        raise RuntimeError(f"qdmemx produced {len(results)} of {len(cases)} cases:\n{proc.stdout}")
    return results


def run() -> list[R.CaseResult]:
    return R.run(HARNESS_CASES, cstm=CSTM) + run_aux()


def _f32(values) -> list[float]:
    # 9 significant digits round-trip the REAL words once rounded back to float32.
    return np.asarray(values, dtype=float).astype(np.float32).astype(float).tolist()


def to_golden(results: list[R.CaseResult]) -> dict:
    out = {}
    for r in results:
        entry: dict = {"element": r.case.element, "routine": r.case.routine}
        if r.parts:
            entry["parts"] = {f"{p}:{j}": v.tolist() for (p, j), v in r.parts.items()}
        for k, v in r.words.items():
            entry[k.lower()] = _f32(v)
        out[r.case.name] = entry
    return out


def main() -> int:
    R.build()
    golden = to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
