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

## Reference baseline

`nastran95py.reference` inventories the Rigid Format 1 demo decks in `../inp/`
and indexes their COSMIC listings in `../demoout/` (the Fortran reference, since
`../bin/nastrn.exe` is a SPARC build). Its deck reader covers only what the
inventory needs. Generated outputs:

- `tests/reference/manifest.json`: machine-readable manifest (elements,
  properties, materials, loads, subcases/output requests, listing section page and
  line ranges, printed matrices, consistency checks). Load it with
  `nastran95py.reference.inventory.load_manifest()`.
- `tests/reference/DECKS.md`: the same inventory as a readable catalogue.

```bash
python -m nastran95py.reference.inventory          # regenerate
python -m nastran95py.reference.inventory --check  # fail if stale (CI)
```

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
It links the unmodified RF1 element routines (EMG `rods/rodd`, `shears/sheard`,
`trmems/trmemd`, `qdmm1s/d`, `qdmm2s/d`; SMA1 `krod`, `ktrmem`, `kqdmem`,
`ktriqd`; SDR2 `srod1/2`, `spanl1/2`, `strme1`, `sqdme1`, `sqdm11/12/21/22`,
`stqme2`) and dumps K, M and stress/force words for the hand-built cases in
`tests/fortran_harness/cases.py` (`tests/test_fortran_harness.py`);
`python -m tests.fortran_harness.runner` regenerates `golden/harness_golden.json`.
