"""Numerical parity between the Python port and the gfortran NASTRAN-95 output.

A Python value v matches a printed Fortran value g when
    |v - g| <= max(RTOL * |g|, ulp(g), ATOL_FRAC * max|table|)
RTOL = 1e-5 (the Fortran prints 7 significant digits, 1e-6 relative; this allows 10x that),
ulp(g) is one unit in g's last printed digit (covers angles printed to 4 decimals), and
ATOL_FRAC * max|table| (1e-6 of the largest magnitude in the table) treats round-off
residues such as a printed 3.48E-10 against an exact 0.0 as zero.
"""
RTOL = 1e-5
ATOL_FRAC = 1e-6


def compare(golden, result, tables=None):
    """Return a list of mismatch strings (empty means parity). golden: f06.parse output;
    result: {title: {id: [float, ...]}} from the Python port."""
    problems = []
    for title in tables or golden:
        if title not in golden:
            problems.append(f"{title}: missing from Fortran output")
            continue
        gt = golden[title]
        pt = result.get(title, {})
        scale = max((abs(v) for row in gt.values() for v, _ in row), default=0.0)
        for eid, grow in gt.items():
            prow = pt.get(eid)
            if prow is None:
                problems.append(f"{title} {eid}: missing from Python result")
                continue
            if len(prow) != len(grow):
                problems.append(f"{title} {eid}: {len(prow)} values, Fortran has {len(grow)}")
                continue
            for k, ((g, u), v) in enumerate(zip(grow, prow)):
                tol = max(RTOL * abs(g), u, ATOL_FRAC * scale)
                if abs(v - g) > tol:
                    problems.append(f"{title} {eid}[{k}]: python {v:.7E} fortran {g:.7E}")
    return problems
