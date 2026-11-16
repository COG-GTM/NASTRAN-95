#!/usr/bin/env python3
"""Compare scripts/run.sh output (out/<deck>.out) with NASA's reference output (demoout/<deck>.out).

Run-dependent text is masked before comparing: page-header dates, the restart/checkpoint date
stamp, CPU/elapsed timing, GINO timing constants and the round-off residual (EPSILON SUB E). Numeric tokens then match when they agree to
within one unit in the last printed digit, or a relative 1e-4 (the 1995 reference ran on a Sun
SPARC; this build runs on x86-64, so the last digits of decomposed solutions can differ).
Values below 1e-8 in magnitude on both sides are treated as round-off zeros.

    scripts/compare_demoout.py d01011a d01101a ...   # prints MATCH / DIFF per deck
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+|[-+]\d{2,3})?")
MASKS = [
    (re.compile(r"\d+/\d+/(?:\d\d|\*\*), +\d+,"), "DATE, TIME,"),
    (re.compile(r"[A-Z]{3} +\d+, +(?:\d{2,4}|\*\*)"), "DATE"),
]
SKIP = re.compile(
    r"TIME CONSTANTS|CPU|ELAPSED|SECONDS|TIME ESTIMATE|TIME TO GO|DIAG 48|NASTRAN RELEASE NEWS|"
    r"MODULES AND NEW BULKDATA|PAGE +\d+$|^ +DATE: |END TIME:|WALL CLOCK|EPSILON SUB E"
)


def lines(path):
    out = []
    for raw in path.read_text(errors="replace").replace("\r", "").splitlines():
        line = raw.rstrip()
        if not line.strip() or SKIP.search(line):
            continue
        for rx, rep in MASKS:
            line = rx.sub(rep, line)
        out.append(" ".join(line.split()))
    return out


def split_exp(token):
    t = token.upper().replace("D", "E")
    t = re.sub(r"(?<=\d)([-+]\d{2,3})$", r"E\1", t)
    mant, _, exp = t.partition("E")
    return mant, int(exp) if exp else 0


def tol(token):
    mant, exp = split_exp(token)
    digits = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (exp - digits)


def value(token):
    mant, exp = split_exp(token)
    return float(mant) * 10.0 ** exp


def same(a, b):
    if a == b:
        return True
    na, nb = NUM.findall(a), NUM.findall(b)
    if len(na) != len(nb) or NUM.sub("#", a).split() != NUM.sub("#", b).split():
        return False
    for x, y in zip(na, nb):
        if x != y and max(abs(value(x)), abs(value(y))) >= 1e-8 and abs(value(x) - value(y)) > max(tol(y), 1e-4 * abs(value(y))) * 1.0001:
            return False
    return True


def compare(deck):
    ref = lines(ROOT / "demoout" / f"{deck}.out")
    got = lines(ROOT / "out" / f"{deck}.out")
    bad = [i for i, (a, b) in enumerate(zip(got, ref)) if not same(a, b)]
    return ref, got, bad


def main(decks):
    worst = 0
    for deck in decks:
        deck = Path(deck).stem
        ref, got, bad = compare(deck)
        ok = not bad and len(ref) == len(got)
        print(f"{'MATCH' if ok else 'DIFF '} {deck}: {len(got)}/{len(ref)} lines, {len(bad)} differing")
        if not ok:
            worst = 1
            for i in bad[:2]:
                print(f"    ref: {ref[i][:110]}\n    got: {got[i][:110]}")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
