"""CSHEAR thermal load harness cases (``mis/tshear.f`` via ``build/tshearld``).

The SHEARD geometries of :mod:`shear_cases` (flat, warped, skewed, shuffled SILs,
rectangular/cylindrical/spherical displacement systems and the panels SHEARD
rejects) under a uniform and a non-uniform grid temperature field. ``TSHEAR`` adds
its loads at ``PG(ISILS)`` and hands ``ISILS(IB)`` to ``BASGLB`` as the point, so the
record's SILs are offset to start at the harness load area ``ILOAD``; the Python
side uses the same SIL words and subtracts the offset for the load vector.

``python -m tests.fortran_harness.shear_load_cases`` rebuilds the harness, reruns
these cases and rewrites ``golden/shear_load_golden.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys

from . import cases as C
from . import rod_load_cases as LC
from . import runner as R
from . import shear_cases as SC

BINARY = R.HERE / "build" / "tshearld"
GOLDEN = R.HERE / "golden" / "shear_load_golden.json"
LOAD = 4
SIL0 = 290000  # ILOAD - 1 in tshearld.f
# tag: grid temperatures T1..T4 (ITEMP = 1)
TEMPS = {
    "uniform": (170.0, 170.0, 170.0, 170.0),
    "gradient": (70.0, 120.5, -30.25, 210.0),
}
GEOMS = {**SC.SHEAR_CSTM_GEOMS, **SC.SHEAR_ERROR_GEOMS}


def est(eid, grids, mid, t, nsm, sils, temps) -> list:
    words = SC.shear_cstm_est(eid, grids, mid, t, nsm, [SIL0 + s for s in sils])
    words[-1] = sum(temps) / 4.0  # ECPT element temperature (MAT lookup)
    return words


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 900
    for tag, (grids, mid, t, nsm, sils) in GEOMS.items():
        for load, temps in TEMPS.items():
            eid += 1
            nload = max(sils) + 5
            out.append(
                C.Case(
                    f"CSHEAR-{tag}-{load}",
                    "CSHEAR",
                    "TSHEAR",
                    LOAD,
                    est(eid, grids, mid, t, nsm, sils, temps),
                    [1, *temps, nload],
                    [xyz for _, xyz in grids],
                    {"temperatures": temps, "nload": nload},
                )
            )
    return out


SHEAR_LOAD_CASES = _cases()


def run(cases: list[C.Case] | None = None, binary=BINARY):
    cases = SHEAR_LOAD_CASES if cases is None else cases
    proc = subprocess.run(
        [str(binary)],
        input=R.stdin_for(cases, SC.CSTM),
        capture_output=True,
        text=True,
        check=True,
    )
    loads = LC.parse(proc.stdout)
    if len(loads) != len(cases):
        raise RuntimeError(f"harness produced {len(loads)} of {len(cases)} cases:\n{proc.stdout}")
    return loads


def main() -> int:
    R.build()
    golden = {
        c.name: {"element": c.element, "load": v.tolist()}
        for c, v in zip(SHEAR_LOAD_CASES, run(), strict=True)
    }
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
