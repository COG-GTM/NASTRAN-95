"""ROD thermal/deformation load harness cases (``mis/rod.f`` via ``build/rodload``).

CROD, CONROD and TUBE records in the basic system and in the offset, rotated,
cylindrical and spherical displacement systems of :mod:`rod_cases`, with a
load-set element temperature, an enforced deformation, or both. The harness
prints the SIL-ordered load words ``ROD`` adds into ``CORE``.

``python -m tests.fortran_harness.rod_load_cases`` rebuilds the harness, reruns
these cases and rewrites ``golden/rod_load_golden.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from . import cases as C
from . import rod_cases as RC
from . import runner as R

BINARY = R.HERE / "build" / "rodload"
GOLDEN = R.HERE / "golden" / "rod_load_golden.json"
LOAD = 4

# GPTA1 element type codes; ROD only tests ELTYPE = 3 (TUBE).
ELTYPE = {"CROD": 1, "CTUBE": 3, "CONROD": 10}
# tag: (ITEMP, TI(1), IDEFT, DELTA)
LOADS = {
    "temp": (1, 170.0, 0, 0.0),
    "deform": (0, 0.0, 1, 0.003),
    "both": (1, -45.5, 1, -0.0125),
}
BASIC_GEOMS = {
    "basic-x": (((0, (0.0, 0.0, 0.0)), (0, (12.0, 0.0, 0.0))), 1, 0.8, 0.0, 0.0, 0.0, (1, 7)),
    "basic-skew": (((0, (1.0, -2.0, 0.5)), (0, (-3.0, 4.5, 7.25))), 2, 1.3, 0.2, 0.0, 0.0, (7, 1)),
}
GEOMS = {**BASIC_GEOMS, **RC.ROD_CSTM_GEOMS}
TUBE_SECTION = (1.5, 0.125, 0.004)  # OD, T, NSM


def est(element: str, eid: int, grids, mid, area, j, c, nsm, sils) -> list:
    if element == "CTUBE":
        words: list = [eid, *sils, mid, *TUBE_SECTION]
    else:
        words = [eid, *sils, mid, area, j, c, nsm]
    for cid, xyz in grids:
        words += [cid, *map(float, xyz)]
    return [*words, 0.0]


def _cases() -> list[C.Case]:
    out: list[C.Case] = []
    eid = 700
    for element in ("CROD", "CONROD", "CTUBE"):
        for tag, (grids, mid, area, j, c, nsm, sils) in GEOMS.items():
            if element == "CTUBE" and tag not in ("basic-skew", "rect", "cyl", "sph", "fp-mixed"):
                continue
            for load, (itemp, ti, ideft, delta) in LOADS.items():
                eid += 1
                nload = max(sils) + 5
                out.append(
                    C.Case(
                        f"{element}-{tag}-{load}",
                        element,
                        "ROD",
                        LOAD,
                        est(element, eid, grids, mid, area, j, c, nsm, sils),
                        [ELTYPE[element], itemp, ti, ideft, delta, nload],
                        [xyz for _, xyz in grids],
                        {"temperature": ti if itemp else None, "deformation": delta},
                    )
                )
    return out


ROD_LOAD_CASES = _cases()


def parse(text: str) -> list[np.ndarray]:
    """``LOAD n`` word blocks, one per case."""
    out: list[np.ndarray] = []
    vals: list[float] | None = None
    for line in text.splitlines():
        tag = line.split()[0] if line.strip() else ""
        if tag == "LOAD":
            vals = []
        elif tag == "END":
            # 9 significant digits round-trip the REAL words once rounded back to float32.
            out.append(np.array(vals).astype(np.float32).astype(float))
            vals = None
        elif vals is not None:
            vals += [float(x) for x in line.split()]
    return out


def run(cases: list[C.Case] | None = None, binary=BINARY) -> list[np.ndarray]:
    cases = ROD_LOAD_CASES if cases is None else cases
    proc = subprocess.run(
        [str(binary)],
        input=R.stdin_for(cases, RC.CSTM),
        capture_output=True,
        text=True,
        check=True,
    )
    loads = parse(proc.stdout)
    if len(loads) != len(cases):
        raise RuntimeError(f"harness produced {len(loads)} of {len(cases)} cases:\n{proc.stdout}")
    return loads


def main() -> int:
    R.build()
    golden = {
        c.name: {"element": c.element, "load": v.tolist()}
        for c, v in zip(ROD_LOAD_CASES, run(), strict=True)
    }
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {len(golden)} cases to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
