"""Parity of the shared kernels against the unmodified Fortran (mis/premat.f, mis/pretr*.f)."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from nastran95py.coords import CoordSystem, Cstm, transform_to_basic
from nastran95py.materials import Mat1, Mat2, MaterialTable

from .conftest import record_parity

pytestmark = pytest.mark.fortran

f32 = np.float32


def _r(x: float) -> float:
    """Round to single precision so both sides see identical inputs."""
    return float(f32(x))


def _fmt(values) -> str:
    return " ".join(f"{v:.9e}" if isinstance(v, float) else str(v) for v in values)


RNG = np.random.default_rng(1995)

MAT1S = [
    Mat1(1, 1.0e7, 3.8e6, _r(0.3), _r(0.1), _r(1.2e-5), 70.0, _r(0.02), 3e4, 2.5e4, 1.5e4),
    Mat1(4, 3.0e7, _r(3.0e7 / 2.6), _r(0.3), _r(7.3e-4), _r(6.5e-6), 0.0, 0.0, 0.0, 0.0, 0.0),
    Mat1(7, _r(6.9e10), _r(2.6e10), _r(0.33), 2700.0, _r(2.3e-5), 20.0, 0.0, 0.0, 0.0, 0.0),
]
MAT2S = [
    Mat2(
        2, 1e7, 3e6, 1e5, 8e6, 2e5, 2e6, _r(0.2), _r(1e-5), _r(2e-5), _r(3e-6), 0, _r(0.01), 1, 2, 3
    ),
    Mat2(
        5,
        _r(2.1e7),
        _r(4.3e5),
        0.0,
        _r(1.4e6),
        0.0,
        _r(6.5e5),
        _r(1.5e-4),
        _r(-3e-7),
        _r(1.6e-5),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
]


def _cstm() -> Cstm:
    systems = []
    for cid, ctype in [(10, 1), (11, 2), (12, 3), (13, 2), (14, 3)]:
        a = RNG.normal(size=3)
        b = a + RNG.normal(size=3)
        c = a + RNG.normal(size=3)
        cs = CoordSystem.from_points(cid, ctype, a, b, c)
        systems.append(
            CoordSystem(cid, ctype, [_r(v) for v in cs.origin], [_r(v) for v in cs.t.ravel()])
        )
    return Cstm(systems)


def _run(binary, cstm, cases) -> list[np.ndarray]:
    lines = [str(len(MAT1S))]
    for m in MAT1S:
        lines.append(_fmt([m.mid, m.e, m.g, m.nu, m.rho, m.alpha, m.tref, m.ge, m.st, m.sc, m.ss]))
    lines.append(str(len(MAT2S)))
    for m in MAT2S:
        lines.append(
            _fmt(
                [
                    m.mid,
                    m.g11,
                    m.g12,
                    m.g13,
                    m.g22,
                    m.g23,
                    m.g33,
                    m.rho,
                    m.a1,
                    m.a2,
                    m.a12,
                    m.tref,
                    m.ge,
                    m.st,
                    m.sc,
                    m.ss,
                ]
            )
        )
    lines.append(str(len(cstm)))
    for cid in cstm:
        rec = cstm[cid].to_record()
        lines.append(_fmt([int(rec[0]), int(rec[1]), *map(float, rec[2:])]))
    lines.append(str(len(cases)))
    lines += [_fmt(c) for c in cases]
    proc = subprocess.run(
        [str(binary)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True
    )
    return [np.array([float(v) for v in ln.split()]) for ln in proc.stdout.splitlines()]


def _errors(py: np.ndarray, ft: np.ndarray, scale: np.ndarray) -> tuple[float, float]:
    abs_err = np.abs(py - ft)
    rel = abs_err / np.where(scale == 0.0, 1.0, scale)
    return float(abs_err.max()), float(rel.max())


def _material_cases():
    angles = [0.0, 30.0, 45.0, 90.0, 135.0, -60.0, *RNG.uniform(-180, 180, 6)]
    cases = []
    for m in MAT1S:
        cases.append((1, 1, m.mid, 0.0, 1.0))
        cases.append((1, 3, m.mid, 0.0, 1.0))
        cases.append((1, 4, m.mid, 0.0, 1.0))
    for mid in [m.mid for m in MAT1S + MAT2S]:
        for th in angles:
            t = np.deg2rad(th)
            cases.append((1, 2, mid, _r(np.sin(t)), _r(np.cos(t))))
    for m in MAT2S:
        cases.append((1, 3, m.mid, 0.0, 1.0))
        cases.append((1, 4, m.mid, 0.0, 1.0))
    return cases


def test_mat_matches_fortran(fortran_kernels):
    table = MaterialTable(MAT1S, MAT2S)
    cases = _material_cases()
    out = _run(fortran_kernels, _cstm(), cases)
    assert len(out) == len(cases)
    py_all, ft_all, sc_all = [], [], []
    for (_, inflag, mid, s, c), words in zip(cases, out, strict=True):
        if inflag == 1:
            o = table.mat1(mid)
            py = np.array([o.e, o.g, o.nu, o.rho, o.alpha, o.tref, o.ge, o.sigt, o.sigc, o.sigs])
            ft = words[:10]
            scale = np.abs(ft)
        elif inflag == 2:
            o = table.mat2(mid, s, c)
            py, ft = o.to_words(), words[:15]
            # Scale G terms by the largest stiffness term so near-zero couplings
            # are judged against single-precision round-off of the matrix.
            gscale = np.abs(ft[:6]).max()
            ascale = np.abs(ft[7:10]).max() or 1.0
            scale = np.abs(ft).copy()
            scale[:6] = gscale
            scale[7:10] = ascale
        elif inflag == 3:
            py, ft = np.array(table.transverse_shear(mid)), words[15:18]
            scale = np.abs(ft)
        else:
            py, ft = np.array([table.density(mid)]), words[:1]
            scale = np.abs(ft)
        py_all.append(py)
        ft_all.append(ft)
        sc_all.append(scale)
    py, ft, sc = (np.concatenate(a) for a in (py_all, ft_all, sc_all))
    max_abs, max_rel = _errors(py, ft, sc)
    record_parity(f"MAT inflag 1-4 ({len(cases)} calls)", py.size, max_abs, max_rel)
    assert max_rel < 1e-5


# TRANSD returns a DOUBLE PRECISION matrix but forms the point-minus-origin offset
# from REAL words, so both entries agree with the double-precision port to ~1e-7.
@pytest.mark.parametrize("kind,label", [(2, "TRANSS"), (3, "TRANSD")])
def test_transforms_match_fortran(fortran_kernels, kind, label):
    cstm = _cstm()
    points = [[_r(v) for v in RNG.uniform(-5, 5, 3)] for _ in range(8)]
    cases = [(kind, cid, *p) for cid in [0, *cstm] for p in points]
    origin = cstm[13].origin
    cases.append((kind, 13, *[float(v) for v in origin]))  # on the polar axis
    out = _run(fortran_kernels, cstm, cases)
    py = np.concatenate([transform_to_basic(cstm, c[1], c[2:]).ravel() for c in cases])
    ft = np.concatenate(out)
    max_abs, max_rel = _errors(py, ft, np.ones_like(ft))
    record_parity(f"{label} ({len(cases)} calls)", py.size, max_abs, max_rel)
    assert max_abs < 1e-6
