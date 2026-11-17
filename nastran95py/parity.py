"""Numerical parity between the Python port and the gfortran NASTRAN-95 output.

A Python value v matches a printed Fortran value g when
    |v - g| <= max(RTOL * |g|, ulp(g), ATOL_FRAC * max|table|)
RTOL = 1e-5 (the Fortran prints 7 significant digits, 1e-6 relative; this allows 10x that),
ulp(g) is one unit in g's last printed digit (covers angles printed to 4 decimals), and
ATOL_FRAC * max|table| (1e-6 of the largest magnitude in the table) treats round-off
residues such as a printed 3.48E-10 against an exact 0.0 as zero.

Element forces and stresses get one more allowance, `recovery_tolerance`: the Fortran
recovers them in single precision (REAL*4 SDR2 routines such as mis/srod2.f) from a
single-precision displacement vector, and the parity tests can only see that vector to its
7 printed digits. For short, stiff elements (the 0.8 in. CROD posts joining the delta
wing's skins) the recovered force is a small difference of two large, nearly equal
displacements, so those digits are all the Fortran itself has. Every displacement
component u_j carries an uncertainty du_j = max(ulp_j / 2, 2**-24 |u_j|) (plus any
Python-vs-Fortran displacement difference in the full solution), and since recovery is
linear in u, sum_j |d(output)/du_j| du_j bounds how far that moves each output.
"""
import numpy as np

RTOL = 1e-5
ATOL_FRAC = 1e-6


F32_EPS = 2.0 ** -24


def displacement_uncertainty(golden, python_disp=None):
    """{grid: du} from the printed Fortran displacements (ulp/2, single-precision rounding)
    and, if given, the Python solution's distance from them."""
    du = {}
    for g, row in golden["DISPLACEMENT VECTOR"].items():
        vals = np.array([v for v, _ in row])
        d = np.maximum(np.array([u for _, u in row]) / 2.0, F32_EPS * np.abs(vals))
        if python_disp is not None and g in python_disp:
            d = d + np.abs(np.asarray(python_disp[g]) - vals)
        du[g] = d
    return du


def recovery_tolerance(model, disp, du):
    """{title: {id: [tol, ...]}}: how far displacement uncertainty du can move each element
    output. Element recovery is linear in the displacements, so perturbing one component at
    a time by its du gives the exact first-order bound."""
    from .elements import REGISTRY
    tol = {}
    for elem in model.elements:
        rec = REGISTRY[elem.type].recover
        base = rec(model, elem, disp)
        acc = {t: {i: np.zeros(len(v)) for i, v in rows.items()} for t, rows in base.items()}
        for g in elem.grids:
            for c in range(6):
                if not du[g][c]:
                    continue
                pert = dict(disp)
                pert[g] = np.array(disp[g], dtype=float)
                pert[g][c] += du[g][c]
                for t, rows in rec(model, elem, pert).items():
                    for i, v in rows.items():
                        acc[t][i] += np.abs(np.array(v) - np.array(base[t][i]))
        for t, rows in acc.items():
            tol.setdefault(t, {}).update({i: list(v) for i, v in rows.items()})
    return tol


def compare(golden, result, tables=None, extra_tol=None):
    """Return a list of mismatch strings (empty means parity). golden: f06.parse output;
    result: {title: {id: [float, ...]}} from the Python port; extra_tol: optional
    {title: {id: [tol, ...]}} added to the base tolerance (see recovery_tolerance)."""
    problems = []
    for title in tables or golden:
        if title not in golden:
            problems.append(f"{title}: missing from Fortran output")
            continue
        gt = golden[title]
        pt = result.get(title, {})
        et = (extra_tol or {}).get(title, {})
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
                if eid in et:
                    tol += et[eid][k]
                if abs(v - g) > tol:
                    problems.append(f"{title} {eid}[{k}]: python {v:.7E} fortran {g:.7E}")
    return problems
