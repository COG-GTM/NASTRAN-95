"""Element library. Each module exposes

    CARDS: tuple of bulk data element names it handles
    stiffness(model, elem) -> (6n x 6n) ndarray in basic coordinates, DOF order
        [T1 T2 T3 R1 R2 R3] for each of elem.grids in turn
    recover(model, elem, disp) -> {table title: {elem.id: [printed values]}}
        disp maps grid id -> length-6 basic displacement; titles and value order follow
        the gfortran NASTRAN-95 listing exactly (see nastran95py/f06.py)
"""
from . import qdmem1, qdmem2, rod, shear, trmem

REGISTRY = {card: mod for mod in (rod, shear, trmem, qdmem1, qdmem2) for card in mod.CARDS}
