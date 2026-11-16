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


NOISY = 1e-2


def _tolerance(model, golden, result):
    """The Fortran solves in double precision (SSG3) and recovers from the REAL*4 rounding of
    that solution, which the Python solve reproduces (the displacement table checks it), so
    each component is uncertain by 2**-24 of its value."""
    disp = {g: np.asarray(u, dtype=float) for g, u in result.disp.items()}
    du = {g: parity.F32_EPS * np.abs(u) for g, u in disp.items()}
    return parity.recovery_tolerance(model, disp, du)


def noisy_values(golden, tol, title):
    """Printed values whose REAL*4 recovery spread exceeds NOISY of their size: the 1995
    Fortran itself only knows these to a digit or two, so parity on them is weak."""
    return sum(1 for i, row in tol.get(title, {}).items()
               for t, (g, _) in zip(row, golden[title].get(i, [])) if t > NOISY * abs(g))


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
        noisy = noisy_values(golden, tol, title)
        note = f"  ({noisy} REAL*4-noise-limited)" if noisy else ""
        print(f"{'PASS' if not problems else 'FAIL'}  {title:<52} {n:5d} values{note}")
        for p in problems[:3]:
            print("      ", p)
        failed += bool(problems)
    print(f"{'PARITY' if not failed else 'MISMATCH'}: {deck} vs gfortran ({listing}), "
          f"rtol={parity.RTOL:g}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
