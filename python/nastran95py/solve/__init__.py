"""Minimal Rigid Format 1 static solution (assembly, constraints, loads, recovery)."""

from nastran95py.solve.static import (
    ElementTable,
    NotSupported,
    SingularStiffness,
    StaticModel,
    StaticSolution,
    SubcaseResult,
    assemble_stiffness,
    solve_static,
)

__all__ = [
    "ElementTable",
    "NotSupported",
    "SingularStiffness",
    "StaticModel",
    "StaticSolution",
    "SubcaseResult",
    "assemble_stiffness",
    "solve_static",
]
