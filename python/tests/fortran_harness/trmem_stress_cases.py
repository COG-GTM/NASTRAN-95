"""STRME1/STQME2 harness cases: CTRMEM stress recovery in coordinate systems, with temperatures.

The basic-system CTRMEM SDR2 cases in :mod:`cases` apply a uniform strain with
no load temperature. These cases reuse the CTRMEM geometries of
:mod:`trmem_cases` (grids in the rectangular, cylindrical and spherical
displacement systems of :mod:`rod_cases`, permuted SIL orders, MAT1 and
anisotropic MAT2 at several material angles) with general displacement
vectors, a subcase load temperature (``LDTEMP``/``TEMP``: the thermal
correction ``- S_T (T - T_SUB_0)``), a thermal-only case and a zero-stress case
(the ``ATAN2`` bypass of ``STQME2``).

``python -m tests.fortran_harness.trmem_stress_cases`` rebuilds the harness,
reruns these cases and rewrites ``golden/trmem_stress_golden.json``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import runner as R
from . import trmem_cases as TC

GOLDEN = R.HERE / "golden" / "trmem_stress_golden.json"
CSTM = RC.CSTM
NDISP = max(C.SILS[:3]) + 5

# tag: (LDTEMP, TEMP, displacements?)
LOADS = {
    "disp": (-1, 0.0, True),
    "thermal": (0, 245.5, True),
    "thermal-only": (0, -60.25, False),
    "zero": (-1, 0.0, False),
}


def displacements(seed: int, ndisp: int = NDISP, scale: float = 1.0e-3) -> list[float]:
    """Deterministic global displacement vector (translations and rotations)."""
    rng = np.random.default_rng(seed)
    return [float(v) for v in scale * rng.uniform(-1.0, 1.0, ndisp)]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eltype, nwdsa, nwdstr, nwdfor = C.SDR_WORDS["CTRMEM"]
    eid = 1500
    for tag, (grids, mid, t, theta, nsm, sils) in TC.TRMEM_GEOMS.items():
        for load, (ldtemp, temp, moving) in LOADS.items():
            eid += 1
            est = TC.trmem_cstm_est(eid, grids, mid, t, theta, nsm, sils)
            disp = displacements(eid) if moving else [0.0] * NDISP
            extra = ["STQME2", eltype, nwdsa, nwdstr, nwdfor, ldtemp, temp, 0.0, NDISP, *disp]
            meta = {
                "mid": mid,
                "theta": theta,
                "sils": sils,
                "temperature": None if ldtemp == -1 else temp,
                "displacements": disp,
            }
            xyz = [p for _, p in grids]
            out.append(
                C.Case(f"CTRMEM-{tag}-{load}", "CTRMEM", "STRME1", C.SDR2, est, extra, xyz, meta)
            )
    return out


TRMEM_STRESS_CASES = _cases()


def _f32(values) -> list[float]:
    # 9 significant digits round-trip the REAL words once rounded back to float32.
    return np.asarray(values, dtype=float).astype(np.float32).astype(float).tolist()


def run() -> list[R.CaseResult]:
    return R.run(TRMEM_STRESS_CASES, cstm=CSTM)


def to_golden(results: list[R.CaseResult]) -> dict:
    return {
        r.case.name: {
            "element": r.case.element,
            **{k.lower(): _f32(r.words[k]) for k in ("PHASE1", "STRESS")},
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
