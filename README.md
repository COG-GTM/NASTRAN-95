# NASTRAN-95

NASTRAN has been released under the
[NASA Open Source Agreement version 1.3](NASA%20Open%20Source%20Agreement-NASTRAN%2095.doc).

NASTRAN is the NASA Structural Analysis System, a finite element analysis program (FEA) completed in the early 1970's. It was the first of its kind and opened the door to computer-aided engineering. Subsections of a design can be modeled and then larger groupings of these elements can again be modeled. NASTRAN can handle elastic stability analysis, complex eigenvalues for vibration and dynamic stability analysis, dynamic response for transient and steady state loads, and random excitation, and static response to concentrated and distributed loads, thermal expansion, and enforced deformations.

NOTE: There is no technical support available for this software.

## What is in this repository

- The COSMIC NASTRAN-95 Fortran 77 source as released by NASA, with its DMAP Rigid Formats, ALTER packages, demonstration decks and reference listings.
- [`python/`](python/README.md): `nastran95py`, a Python/NumPy port of the Rigid Format 1 (static analysis) element routines and a minimal RF1 static solve, tested against the Fortran source and the COSMIC listings.

How the parts fit together is described in [docs/architecture.md](docs/architecture.md), with a diagram in [docs/diagrams/](docs/diagrams/architecture.svg).

## Layout

| Path | Contents |
|---|---|
| `mis/` | Fortran routines: preface, DMAP executive, functional modules, element routines |
| `bd/` | `BLOCK DATA` tables |
| `mds/` | Machine-dependent GINO / database I/O layer |
| `rf/` | Rigid Formats (DMAP solution sequences: `DISP*`, `HEAT*`, `AERO*`) |
| `alt/` | COSMIC DMAP ALTER packages (hydroelastic, DDAM, cyclic forced vibration, delamination buckling) |
| `bin/` | `nastran` csh driver, `nastrn.f`, `linknas`, `nastlib.a` and 32-bit SPARC Solaris executables |
| `inp/` | 132 demonstration and test decks with `.txt` descriptions |
| `demoout/` | COSMIC output listings for the decks in `inp/` |
| `um/` | User's Manual text and the `nasthelp` viewer |
| `utility/` | `ff` (free-field converter) and `nastplot` (plot translator) |
| `python/` | `nastran95py` package, tests and gfortran parity harnesses |
| `docs/` | Architecture document and diagram |

## Running the Fortran program

`bin/nastran <problem>` writes a run script `<problem>.cmd` for `<problem>.inp` and offers to run it; the run writes `<problem>.out`, `.log`, `.dic`, `.plt` and `.pch`. Before use, edit the `rfdir`, `nasexec` and `naschk` paths at the top of `bin/nastran` (they point at `/usr/users/bob/nast95`). The checked-in `nastrn.exe` and `chkfil.exe` are 32-bit SPARC Solaris binaries and `bin/linknas` links with Sun `f77`, so they do not run on x86-64 Linux; there `demoout/` is the reference output.

## Python package

Requires Python 3.11+ (and gfortran for the parity harnesses):

```sh
cd python
python -m pip install -e '.[dev]'
ruff check . && ruff format --check .
python -m pytest
```

See [python/README.md](python/README.md) for the API, the RF1 static solve and how the golden files are regenerated.
