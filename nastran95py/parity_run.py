"""Run the Python port on a deck and check it against the gfortran NASTRAN-95 listing.

    python3 -m nastran95py.parity_run tests/decks/d01013a_all.inp tests/golden/d01013a_all.out
"""
import re
import sys

import numpy as np

from . import f06, parity
from .elements import REGISTRY
from .model import build_model
from .static import DISPLACEMENT, SPCFORCE, solve

IN_SCOPE = re.compile(r"\((%s)\)$" % "|".join(REGISTRY))


def tables_in_scope(golden):
    return [t for t in golden if t in (DISPLACEMENT, SPCFORCE) or IN_SCOPE.search(t)]


def _tolerance(model, golden, result):
    disp = {g: np.array([v for v, _ in row]) for g, row in golden[DISPLACEMENT].items()}
    du = parity.displacement_uncertainty(golden, result.disp)
    for g, u in result.disp.items():  # grids the deck does not print: Python values, f32 rounding
        if g not in disp:
            disp[g] = np.asarray(u, dtype=float)
            du[g] = parity.F32_EPS * np.abs(disp[g])
    return parity.recovery_tolerance(model, disp, du)


def parity_problems(model, golden, result, titles=None):
    return parity.compare(golden, result.tables, titles or tables_in_scope(golden),
                          _tolerance(model, golden, result))


def main(argv):
    deck, listing = argv
    with open(deck, errors="replace") as f:
        model = build_model(f.read())
    result = solve(model)
    golden = f06.parse_file(listing)
    tol = _tolerance(model, golden, result)
    failed = 0
    for title in tables_in_scope(golden):
        problems = parity.compare(golden, result.tables, [title], tol)
        n = sum(len(row) for row in golden[title].values())
        print(f"{'PASS' if not problems else 'FAIL'}  {title:<52} {n:5d} values")
        for p in problems[:3]:
            print("      ", p)
        failed += bool(problems)
    print(f"{'PARITY' if not failed else 'MISMATCH'}: {deck} vs gfortran ({listing}), "
          f"rtol={parity.RTOL:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
