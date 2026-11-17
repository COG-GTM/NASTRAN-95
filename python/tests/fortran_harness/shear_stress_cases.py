"""SPANL1/SPANL2 harness cases: CSHEAR stress recovery on every SHEARD panel.

Reuses the panels of :mod:`shear_cases` (all four flexibility branches, warped
and skewed panels, shuffled SILs, grids in rectangular/cylindrical/spherical
systems) with general displacements, so the ``AVEC``, ``Z``, ``TRANSS`` and
force/kick-load paths of ``SPANL1(4)``/``SPANL2(4)`` are all exercised. A
material with a shear allowable (``SS``) covers the margin of safety; a
zero-displacement case covers the blank margin (``TAUMAX = 0``).

``python -m tests.fortran_harness.shear_stress_cases`` reruns these cases and
rewrites ``golden/shear_stress_golden.json``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from . import cases as C
from . import runner as R
from . import shear_cases as SC

GOLDEN = R.HERE / "golden" / "shear_stress_golden.json"
CSTM = SC.CSTM

# MAT1 with a shear allowable, only passed to the harness by this module.
SS_MAT = (5, (2.0e7, 2.0e7 / 2.6, 0.3, 5.0e-4, 1.0e-5, 0.0, 0.0, 4.0e4, 3.5e4, 2.5e4))
MAT1 = [*C.MAT1, SS_MAT]
NDISP = 24


def displacements(seed: int, scale: float = 1.0e-3) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in scale * rng.uniform(-1.0, 1.0, NDISP)]


def _case(name, eid, grids, mid, t, sils, disp) -> C.Case:
    est = SC.shear_cstm_est(eid, grids, mid, t, 0.0, sils)
    eltype, nwdsa, nwdstr, nwdfor = C.SDR_WORDS["CSHEAR"]
    extra = ["SPANL2", eltype, nwdsa, nwdstr, nwdfor, -1, 0.0, 0.0, len(disp), *disp]
    meta = {"mid": mid, "t": t, "sils": sils}
    return C.Case(name, "CSHEAR", "SPANL1", C.SDR2, est, extra, [x for _, x in grids], meta)


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 900
    for seed, (tag, (grids, mid, t, _nsm, sils)) in enumerate(SC.SHEAR_CSTM_GEOMS.items()):
        eid += 1
        out.append(_case(f"CSHEAR-{tag}-SPANL", eid, grids, mid, t, sils, displacements(seed)))
        if seed % 3 == 0:
            eid += 1
            out.append(
                _case(f"CSHEAR-{tag}-ss-SPANL", eid, grids, SS_MAT[0], t, sils, displacements(seed))
            )
    grids, _, t, _, sils = SC.SHEAR_CSTM_GEOMS["general"]
    eid += 1
    out.append(_case("CSHEAR-general-ss-zero-SPANL", eid, grids, SS_MAT[0], t, sils, [0.0] * NDISP))
    return out


SHEAR_STRESS_CASES = _cases()


def run() -> list[R.CaseResult]:
    saved = list(C.MAT1)
    C.MAT1[:] = MAT1
    try:
        return R.run(SHEAR_STRESS_CASES, cstm=CSTM)
    finally:
        C.MAT1[:] = saved


def main() -> int:
    R.build()
    golden = R.to_golden(run())
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
