# nastran95py

Python port of the NASTRAN-95 element routines (RF1 element library). Each
kernel mirrors a Fortran routine in `../mis/` and is validated numerically
against the compiled Fortran.

**Scope.** This is a library of element-level routines that take the same
inputs as the Fortran (MPT material records, CSTM coordinate records, EST/ECPT
element records). It is *not* a replacement for the GINO/DMAP executive: there
is no bulk-data deck processing, DMAP sequencing or matrix assembly here.

| Subpackage | Fortran counterpart |
|---|---|
| `nastran95py.materials` | `MAT` in `mis/premat.f` (MAT1 isotropic, MAT2 anisotropic; `inflag` 1-4) |
| `nastran95py.coords` | `TRANSS`/`TRANSD` (`mis/pretrs.f`, `mis/pretrd.f`), CSTM 14-word records |
| `nastran95py.elements` | EST/ECPT record contract (`ElementData`, `EcptLayout`; ROD, BAR layouts) |
| `nastran95py.io` | 32-bit INTEGER/REAL word reinterpretation, Fortran `D` exponents |

## Setup

```bash
cd python
python -m pip install -e ".[dev]"   # Python 3.11+, NumPy, pytest, ruff
```

## Tests and parity

```bash
ruff check . && ruff format --check .
python -m pytest                      # unit tests + Fortran parity
make -C tests/fortran_kernels         # optional: build the kernel harness by hand
```

Parity tests (`tests/test_fortran_parity.py`, marker `fortran`) build
`tests/fortran_kernels/` with gfortran: the unmodified `premat.f`, `pretrs.f`,
`pretrd.f`, `gmmats.f` and `gmmatd.f` linked against small GINO stubs and a
driver. pytest feeds the same inputs to Python and Fortran and prints a
"Fortran parity" table with the max absolute and relative error per kernel.
Without gfortran these tests skip; set `NASTRAN95PY_REQUIRE_FORTRAN=1` (as CI
does) to make that a failure. Run only parity with `python -m pytest -m fortran`.

The element-level harness lives in `tests/fortran_harness/`; CI runs
`make -C python/tests/fortran_harness` whenever that directory exists.
