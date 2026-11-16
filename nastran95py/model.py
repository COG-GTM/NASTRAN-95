"""Finite element model built from the bulk data cards the port supports."""
from dataclasses import dataclass, field

import numpy as np

from .bulk import case_control, read_deck

SUPPORTED_ELEMENTS = ("CROD", "CONROD", "CSHEAR", "CTRMEM", "CQDMEM1", "CQDMEM2")


@dataclass
class Grid:
    id: int
    cp: int
    xyz: np.ndarray
    cd: int
    ps: str


@dataclass
class Mat1:
    id: int
    e: float
    g: float
    nu: float
    rho: float = 0.0


@dataclass
class Element:
    type: str
    id: int
    pid: int | None
    grids: list
    data: dict = field(default_factory=dict)  # card-specific fields (theta, mid, area, ...)


@dataclass
class Model:
    grids: dict = field(default_factory=dict)
    mats: dict = field(default_factory=dict)
    props: dict = field(default_factory=dict)  # pid -> {'type', 'mid', 't' or 'a', 'j', 'c', 'nsm'}
    elements: list = field(default_factory=list)
    forces: dict = field(default_factory=dict)  # sid -> [(gid, cid, f, (n1, n2, n3), kind)]
    spcs: dict = field(default_factory=dict)  # sid -> [(gid, comps str)]
    loads: dict = field(default_factory=dict)  # LOAD sid -> (S, [(Si, Li)])
    params: dict = field(default_factory=dict)
    case: dict = field(default_factory=dict)


def _f(card, i, default=None):
    v = card[i] if i < len(card) else None
    return default if v is None else v


def _mat1(card):
    e, g, nu = _f(card, 2), _f(card, 3), _f(card, 4)
    if nu is None and e and g:
        nu = e / (2.0 * g) - 1.0
    if g is None and e is not None and nu is not None:
        g = e / (2.0 * (1.0 + nu))
    if e is None and g is not None and nu is not None:
        e = 2.0 * (1.0 + nu) * g
    return Mat1(card[1], float(e or 0.0), float(g or 0.0), float(nu or 0.0), float(_f(card, 5, 0.0)))


def build_model(text):
    deck = read_deck(text)
    m = Model(case=case_control(deck.case))
    grdset = {"cp": 0, "cd": 0, "ps": ""}
    grid_cards = []
    for c in deck.cards:
        name = c[0]
        if name == "GRDSET":
            grdset = {"cp": _f(c, 2, 0), "cd": _f(c, 6, 0), "ps": str(_f(c, 7, ""))}
        elif name == "GRID":
            grid_cards.append(c)
        elif name == "MAT1":
            m.mats[c[1]] = _mat1(c)
        elif name in ("PSHEAR", "PTRMEM", "PQDMEM1", "PQDMEM2"):
            m.props[c[1]] = {"type": name, "mid": c[2], "t": float(c[3]), "nsm": float(_f(c, 4, 0.0))}
        elif name == "PROD":
            m.props[c[1]] = {"type": name, "mid": c[2], "a": float(c[3]), "j": float(_f(c, 4, 0.0)),
                             "c": float(_f(c, 5, 0.0)), "nsm": float(_f(c, 6, 0.0))}
        elif name == "CROD":  # up to two elements per card: EID PID G1 G2
            for k in (1, 5):
                if _f(c, k) is not None:
                    m.elements.append(Element("CROD", c[k], c[k + 1], [c[k + 2], c[k + 3]]))
        elif name == "CONROD":
            m.elements.append(Element("CONROD", c[1], None, [c[2], c[3]], {
                "mid": c[4], "a": float(_f(c, 5, 0.0)), "j": float(_f(c, 6, 0.0)),
                "c": float(_f(c, 7, 0.0)), "nsm": float(_f(c, 8, 0.0))}))
        elif name == "CSHEAR":
            m.elements.append(Element(name, c[1], c[2], list(c[3:7])))
        elif name == "CTRMEM":
            m.elements.append(Element(name, c[1], c[2], list(c[3:6]), {"theta": float(_f(c, 6, 0.0))}))
        elif name in ("CQDMEM1", "CQDMEM2"):
            m.elements.append(Element(name, c[1], c[2], list(c[3:7]), {"theta": float(_f(c, 7, 0.0))}))
        elif name in ("FORCE", "MOMENT"):
            n = np.array([float(_f(c, i, 0.0)) for i in (5, 6, 7)])
            m.forces.setdefault(c[1], []).append((c[2], _f(c, 3, 0), float(_f(c, 4, 0.0)), n, name))
        elif name == "LOAD":
            rest = [v for v in c[3:] if v is not None]
            m.loads[c[1]] = (float(c[2]), [(float(rest[i]), rest[i + 1]) for i in range(0, len(rest) - 1, 2)])
        elif name == "SPC1":
            m.spcs.setdefault(c[1], []).extend((g, str(c[2])) for g in _thru(c[3:]))
        elif name == "SPC":
            for k in (2, 5):
                if _f(c, k) is not None:
                    m.spcs.setdefault(c[1], []).append((c[k], str(c[k + 1])))
        elif name == "PARAM":
            m.params[c[1]] = _f(c, 2)
        elif name.startswith("C") and name not in SUPPORTED_ELEMENTS:
            raise NotImplementedError(f"element card {name} is outside the ported element group")
    for c in grid_cards:
        cp, cd = _f(c, 2, grdset["cp"]), _f(c, 6, grdset["cd"])
        if cp or cd:
            raise NotImplementedError(f"GRID {c[1]}: only the basic coordinate system is ported")
        m.grids[c[1]] = Grid(c[1], 0, np.array([float(_f(c, i, 0.0)) for i in (3, 4, 5)]), 0,
                             str(_f(c, 7, grdset["ps"])))
    return m


def _thru(fields):
    vals = [f for f in fields if f is not None]
    if len(vals) == 3 and vals[1] == "THRU":
        return list(range(vals[0], vals[2] + 1))
    return vals
