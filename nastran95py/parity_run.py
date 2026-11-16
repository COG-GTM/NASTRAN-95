"""Run the Python port on a deck and check it against the gfortran NASTRAN-95 listing.

    python3 -m nastran95py.parity_run tests/decks/d01013a_all.inp tests/golden/d01013a_all.out
"""
import re
import sys

from . import f06, parity
from .elements import REGISTRY
from .model import build_model
from .static import DISPLACEMENT, SPCFORCE, solve

IN_SCOPE = re.compile(r"\((%s)\)$" % "|".join(REGISTRY))


def tables_in_scope(golden):
    return [t for t in golden if t in (DISPLACEMENT, SPCFORCE) or IN_SCOPE.search(t)]


def main(argv):
    deck, listing = argv
    with open(deck, errors="replace") as f:
        result = solve(build_model(f.read()))
    golden = f06.parse_file(listing)
    failed = 0
    for title in tables_in_scope(golden):
        problems = parity.compare(golden, result.tables, [title])
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
