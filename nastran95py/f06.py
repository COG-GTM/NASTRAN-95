"""Parse NASTRAN printed output (the .out / F06 listing) into tables.

A table is keyed by its spaced-out title as one string, e.g. 'DISPLACEMENT VECTOR' or
'STRESSES IN QUADRILATERAL MEMBRANES (CQDMEM1)'. Each table maps a grid or element id to
a list of (value, ulp) pairs: ulp is one unit in the last printed digit of that value.
"""
import re

_TITLE = re.compile(r"^(?:[A-Z0-9\-()] ){3,}[A-Z0-9\-()]?")
_INT = re.compile(r"^[+-]?\d+$")
_NUM = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)(E[+-]\d+)?$")


def title_of(line):
    s = line[1:].strip()
    if not _TITLE.match(s):
        return None
    words = re.split(r" {2,}", s)
    return " ".join(w.replace(" ", "") for w in words)


def ulp(token):
    m = re.match(r"^[+-]?(\d*)\.(\d*)(?:E([+-]\d+))?$", token)
    if float(token) == 0.0:
        return 0.0
    return 10.0 ** (int(m.group(3) or 0) - len(m.group(2)))


def parse(text):
    tables, current, last = {}, None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        title = title_of(line)
        if title:
            current = tables.setdefault(title, {})
            last = None
            continue
        if current is None or line.startswith("1"):
            continue
        toks = [t for t in line[1:].split() if t not in ("G", "S")]
        if not toks or not all(_INT.match(t) or _NUM.match(t) for t in toks):
            continue
        if not _INT.match(toks[0]):
            if last is not None:
                last.extend((float(t), ulp(t)) for t in toks)
            continue
        for t in toks:
            if _INT.match(t):
                last = current.setdefault(int(t), [])
                last.clear()
            else:
                last.append((float(t), ulp(t)))
    return {k: v for k, v in tables.items() if v}


def parse_file(path):
    with open(path, errors="replace") as f:
        return parse(f.read())
