"""Material evaluation (port of the Fortran ``MAT`` entry of ``mis/premat.f``).

MAT1/MAT2 plus MATT1 temperature dependence through TABLEM1-4.
"""

from nastran95py.materials.mat import (
    Mat1,
    Mat1Out,
    Mat2,
    Mat2Out,
    MaterialError,
    MaterialNotFoundError,
    MaterialTable,
    Matt1,
    rotation_matrix,
)
from nastran95py.materials.tables import TableM, TableM1, TableM2, TableM3, TableM4, interpolate

__all__ = [
    "Mat1",
    "Mat1Out",
    "Mat2",
    "Mat2Out",
    "MaterialError",
    "MaterialNotFoundError",
    "MaterialTable",
    "Matt1",
    "TableM",
    "TableM1",
    "TableM2",
    "TableM3",
    "TableM4",
    "interpolate",
    "rotation_matrix",
]
