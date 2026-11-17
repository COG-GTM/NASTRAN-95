"""SROD1/SROD2 harness cases: rod stress recovery with limits, temperatures and deformations.

The basic-system SDR2 rod cases in :mod:`cases` use MAT1 entries without stress
limits, no load temperature and no enforced deformation. These cases add
CROD/CONROD records in the basic system and in the offset, rotated, cylindrical
and spherical displacement systems of :mod:`rod_cases`, general displacement
vectors, materials with tension/compression/shear limits (so every margin of
safety branch runs), a subcase load temperature (``TEMPLD``), an enforced
deformation (``ELDEFM``) and zero-stress states.

``python -m tests.fortran_harness.rod_stress_cases`` rebuilds the harness,
reruns these cases and rewrites ``golden/rod_stress_golden.json``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import runner as R

GOLDEN = R.HERE / "golden" / "rod_stress_golden.json"

# MAT1: E G NU RHO ALPHA TREF GE ST SC SS (MID 1, 2 from cases.MAT1 have no limits)
MAT1 = [
    *C.MAT1,
    (4, (1.04e7, 4.0e6, 0.3, 2.5e-4, 1.3e-5, 68.0, 0.0, 4.2e4, 3.6e4, 2.5e4)),
    (5, (2.9e7, 1.1e7, 0.32, 7.3e-4, 6.5e-6, -10.0, 0.0, 6.0e4, 0.0, 0.0)),
    (6, (1.6e7, 6.2e6, 0.29, 1.6e-4, 5.0e-6, 300.0, 0.0, 0.0, 7.5e4, 4.4e4)),
]

BASIC_GEOMS = {
    f"basic-{tag}": (tuple((0, xyz) for xyz in grids), mid, area, j, c, nsm, C.SILS[:2])
    for tag, (grids, mid, area, j, c, nsm) in C.ROD_GEOMS.items()
}
GEOMS = {**BASIC_GEOMS, **RC.ROD_CSTM_GEOMS}

# tag: (MID or None for the geometry's, LDTEMP, TEMP, ELDEFM, displacements?)
LOADS = {
    "disp": (None, -1, 0.0, 0.0, True),
    "limits": (4, -1, 0.0, 0.0, True),
    "thermal": (4, 0, 170.0, 0.0, True),
    "deform": (5, -1, 0.0, 0.003, True),
    "both": (6, 0, -45.5, -0.0125, True),
    "tension": (4, -1, 0.0, -0.004, False),
    "compression": (4, -1, 0.0, 0.004, False),
    "zero": (4, -1, 0.0, 0.0, False),
}


def displacements(seed: int, ndisp: int, scale: float = 1.0e-3) -> list[float]:
    """Deterministic global displacement vector (translations and rotations)."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in scale * rng.uniform(-1.0, 1.0, ndisp)]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 900
    for elem in ("CROD", "CONROD"):
        eltype, nwdsa, nwdstr, nwdfor = C.SDR_WORDS[elem]
        for tag, (grids, gmid, area, j, c, nsm, sils) in GEOMS.items():
            for load, (mid, ldtemp, temp, eldefm, moving) in LOADS.items():
                eid += 1
                mid = gmid if mid is None else mid
                est = RC.rod_cstm_est(eid, grids, mid, area, j, c, nsm, sils)
                ndisp = max(sils) + 5
                disp = displacements(eid, ndisp) if moving else [0.0] * ndisp
                extra = [
                    "SROD2",
                    eltype,
                    nwdsa,
                    nwdstr,
                    nwdfor,
                    ldtemp,
                    temp,
                    eldefm,
                    ndisp,
                    *disp,
                ]
                meta = {
                    "mid": mid,
                    "temperature": None if ldtemp == -1 else temp,
                    "deformation": eldefm,
                    "displacements": disp,
                }
                out.append(
                    C.Case(
                        f"{elem}-{tag}-{load}",
                        elem,
                        "SROD1",
                        C.SDR2,
                        est,
                        extra,
                        [xyz for _, xyz in grids],
                        meta,
                    )
                )
    return out


ROD_STRESS_CASES = _cases()


def _f32(values) -> list[float]:
    # 9 significant digits round-trip the REAL words once rounded back to float32.
    return np.asarray(values, dtype=float).astype(np.float32).astype(float).tolist()


def run() -> list[R.CaseResult]:
    return R.run(ROD_STRESS_CASES, cstm=RC.CSTM, mat1=MAT1)


def to_golden(results: list[R.CaseResult]) -> dict:
    return {
        r.case.name: {
            "element": r.case.element,
            **{k.lower(): _f32(r.words[k]) for k in ("PHASE1", "STRESS", "FORCE")},
        }
        for r in results
    }


def main() -> int:
    R.build()
    golden = to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
