"""Parity of MATT1 + TABLEM1-4 temperature dependence against ``mis/premat.f``."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from nastran95py.materials import Mat1, MaterialTable, Matt1, TableM1, TableM2, TableM3, TableM4

from .conftest import record_parity

pytestmark = pytest.mark.fortran

f32 = np.float32


def _r(x: float) -> float:
    return float(f32(x))


def _rs(values) -> tuple[float, ...]:
    return tuple(_r(v) for v in values)


# d01032a/d01033a/d01034a: MAT1 75 with E from TABLEM1 100. d01161a: MAT1 2 with
# E from TABLEM1 222 (a constant table) at TEMPD 80. The rest exercise every
# table type, decreasing abscissae, extrapolation and several properties at once.
MAT1S = [
    Mat1(75, _r(10.4e6), _r(3.9e6), _r(0.33), _r(2.5e-4), _r(1.28e-5), 70.0),
    Mat1(2, 12500.0, _r(12500.0 / 2.6), _r(0.3), 0.0, _r(6.5e-6), 80.0),
    Mat1(10, 3.0e7, _r(1.15e7), _r(0.3), _r(7.3e-4), _r(6.5e-6), 70.0, _r(0.02), 6e4, 5e4, 3e4),
    Mat1(11, _r(6.9e10), _r(2.6e10), _r(0.33), 2700.0, _r(2.3e-5), 20.0),
    Mat1(12, 1.0e7, _r(3.8e6), _r(0.3), _r(0.1), _r(1.2e-5), 70.0),
]
TABLES = [
    TableM1(
        100, (80.0, 150.0, 200.0, 250.0, 300.0), _rs((10.4e6, 10.15e6, 9.84e6, 9.51e6, 9.15e6))
    ),
    TableM1(222, (1.0, 10.0), (12500.0, 12500.0)),
    TableM1(301, (600.0, 400.0, 200.0, 0.0), _rs((0.21, 0.27, 0.3, 0.31))),
    TableM2(302, 70.0, (0.0, 100.0, 300.0, 700.0), _rs((1.0, 0.97, 0.88, 0.6))),
    TableM2(303, 70.0, (-100.0, 0.0, 500.0), _rs((1.05, 1.0, 1.4))),
    TableM3(304, 20.0, 100.0, (0.0, 1.0, 2.5, 4.0), _rs((1.0, 0.95, 0.8, 0.55))),
    TableM3(305, 20.0, -50.0, (-4.0, -2.0, 0.0), _rs((1.3, 1.1, 1.0))),
    TableM4(306, 70.0, 100.0, -50.0, 800.0, _rs((1.0, -0.02, -3.5e-3, 1.0e-4))),
    TableM4(307, 20.0, 200.0, 0.0, 600.0, _rs((1.0, 0.1, -0.05))),
]
MATT1S = [
    Matt1(75, e=100),
    Matt1(2, e=222),
    Matt1(10, e=302, g=303, nu=301, rho=306, alpha=304, ge=307, st=305, sc=302, ss=306),
    Matt1(11, e=306, g=304, alpha=303),
    Matt1(12, nu=307, rho=301),
]
TEMPS = [
    -300.0, -40.0, 0.0, 20.0, 70.0, 80.0, 115.0, 150.0, 199.0, 200.0,
    250.0, 300.0, 333.3, 400.0, 600.0, 800.0, 950.0,
]  # fmt: skip


def _run(binary, cases) -> list[np.ndarray]:
    fmt = " ".join
    lines = [str(len(MAT1S))]
    for m in MAT1S:
        vals = (m.e, m.g, m.nu, m.rho, m.alpha, m.tref, m.ge, m.st, m.sc, m.ss)
        lines.append(fmt([str(m.mid), *(f"{v:.9e}" for v in vals)]))
    lines += ["0", str(len(MATT1S))]
    for t in MATT1S:
        ids = (t.e, t.g, t.nu, t.rho, t.alpha, 0, t.ge, t.st, t.sc, t.ss)
        lines.append(fmt(str(v) for v in (t.mid, *ids)))
    lines.append(str(len(TABLES)))
    for t in sorted(TABLES, key=lambda t: t.type_code):
        if isinstance(t, TableM4):
            params, values = (t.x1, t.x2, t.x3, t.x4), t.a
        else:
            params = (getattr(t, "x1", 0.0), getattr(t, "x2", 0.0), 0.0, 0.0)
            values = tuple(v for xy in zip(t.x, t.y, strict=True) for v in xy)
        lines.append(fmt([str(t.type_code), str(t.tid), *(f"{v:.9e}" for v in params)]))
        lines[-1] += f" {len(values)}"
        lines.append(fmt(f"{v:.9e}" for v in values))
    lines.append(str(len(cases)))
    lines += [
        fmt([str(i), str(m), f"{tp:.9e}", f"{s:.9e}", f"{c:.9e}"]) for i, m, tp, s, c in cases
    ]
    proc = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    return [np.array([float(v) for v in ln.split()]) for ln in proc.stdout.splitlines()]


def _cases():
    cases = []
    for m in MAT1S:
        for tp in TEMPS:
            cases.append((1, m.mid, tp, 0.0, 1.0))
            cases.append((3, m.mid, tp, 0.0, 1.0))
            cases.append((4, m.mid, tp, 0.0, 1.0))
            t = np.deg2rad(30.0)
            cases.append((2, m.mid, tp, _r(np.sin(t)), _r(np.cos(t))))
    return cases


def test_matt1_matches_fortran(fortran_kernels):
    binary = fortran_kernels.parent / "kernels_temp"
    table = MaterialTable(MAT1S, matt1=MATT1S, tables=TABLES)
    cases = _cases()
    out = _run(binary, cases)
    assert len(out) == len(cases)
    py_all, ft_all, sc_all = [], [], []
    for (inflag, mid, tp, s, c), words in zip(cases, out, strict=True):
        if inflag == 1:
            o = table.mat1(mid, temp=tp)
            py = np.array([o.e, o.g, o.nu, o.rho, o.alpha, o.tref, o.ge, o.sigt, o.sigc, o.sigs])
            ft = words[:10]
            scale = np.abs(ft)
        elif inflag == 2:
            py, ft = table.mat2(mid, s, c, temp=tp).to_words(), words[:15]
            scale = np.abs(ft).copy()
            scale[:6] = np.abs(ft[:6]).max()
            scale[7:10] = np.abs(ft[7:10]).max() or 1.0
        elif inflag == 3:
            py, ft = np.array(table.transverse_shear(mid, temp=tp)), words[15:18]
            scale = np.abs(ft)
        else:
            py, ft = np.array([table.density(mid, temp=tp)]), words[:1]
            scale = np.abs(ft)
        py_all.append(py)
        ft_all.append(ft)
        sc_all.append(scale)
    py, ft, sc = (np.concatenate(a) for a in (py_all, ft_all, sc_all))
    abs_err = np.abs(py - ft)
    max_abs = float(abs_err.max())
    max_rel = float((abs_err / np.where(sc == 0.0, 1.0, sc)).max())
    record_parity(f"MATT1+TABLEM1-4 ({len(cases)} calls)", py.size, max_abs, max_rel)
    assert max_rel < 1e-5
