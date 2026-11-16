"""CSHEAR shear panel element.

Port of NASTRAN-95 mis/sheard.f (double precision stiffness, EMG) and mis/spanl1.f, mis/spanl2.f (SDR2 forces, kick loads and stresses).
"""
import numpy as np

CARDS = ("CSHEAR",)
FORCES = "FORCES ACTING ON SHEAR PANEL ELEMENTS (CSHEAR)"
STRESSES = "STRESSES IN SHEAR PANELS (CSHEAR)"
EP = 0.01


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _panel(model, elem, real=np.float64):
    """Geometry and flexibility of SHEARD (double) or SPANL1 (REAL).

    Written statement by statement in SPANL1's evaluation order: with real=np.float32 it
    reproduces the single precision Z, whose general-quadrilateral log terms cancel to
    about 1e-3 of their size, so their rounding shows in the printed forces and stresses.
    """
    prop = model.props[elem.pid]
    mat = model.mats[prop["mid"]]
    t, g, nu = real(prop["t"]), real(mat.g), real(mat.nu)
    gp1, gp2, gp3, gp4 = ([real(c) for c in model.grids[gid].xyz] for gid in elem.grids)
    if t * g == 0.0:
        raise ValueError(f"CSHEAR {elem.id}: zero thickness or shear modulus")
    sqrt = np.sqrt
    vd1 = [gp3[i] - gp1[i] for i in range(3)]
    vd2 = [gp4[i] - gp2[i] for i in range(3)]
    vkn = _cross(vd1, vd2)
    vkl = sqrt(vkn[0] ** 2 + vkn[1] ** 2 + vkn[2] ** 2)
    vk = [c / vkl for c in vkn]
    pa = 0.5 * vkl
    v12 = [gp2[i] - gp1[i] for i in range(3)]
    v41 = [gp1[i] - gp4[i] for i in range(3)]
    v12dk = _dot(v12, vk)
    vp12 = [v12[i] - v12dk * vk[i] for i in range(3)]
    x2 = sqrt(vp12[0] ** 2 + vp12[1] ** 2 + vp12[2] ** 2)
    vi = [c / x2 for c in vp12]
    vj = _cross(vk, vi)
    vjl = sqrt(vj[0] ** 2 + vj[1] ** 2 + vj[2] ** 2)
    vj = [c / vjl for c in vj]
    x1 = y1 = y2 = real(0.0)
    x3, y3 = _dot(vi, vd1), _dot(vj, vd1)
    x4 = -vi[0] * v41[0] - vi[1] * v41[1] - vi[2] * v41[2]
    y4 = -vj[0] * v41[0] - vj[1] * v41[1] - vj[2] * v41[2]
    if y3 <= 0.0 or x3 <= y3 * x4 / y4 or y4 <= 0.0 or x4 >= x2 - (x2 - x3) * y4 / y3:
        raise ValueError(f"CSHEAR {elem.id}: interior angle not less than 180 degrees")

    xs, ys = (x1, x2, x3, x4), (y1, y2, y3, y4)
    temp = x3 - x2
    par34 = abs(y3 - y4) < abs(x3 - x4) * EP
    par23 = abs(y4 * temp - y3 * x4) < abs(x4 * temp + y4 * y3) * EP
    if par34 and not par23:  # side 3-4 parallel to side 1-2
        temp = y3 * x4 - y4 * (x3 - x2)
        yp = x2 * y3 * y4 / temp
        p = [yp - y for y in ys]
        xp = x2 * y3 * x4 / temp
        sa, c = (x2 - xp) / yp, (x1 - xp) / yp
        z = (p[0] * p[1] * pa) / (p[2] * p[3] * 2.0 * g * t) * (
            1.0 + 2.0 / (3.0 + 3.0 * nu) * (sa ** 2 + sa * c + c ** 2))
    elif par23 and not par34:  # side 1-4 parallel to side 2-3
        d = -0.5 * (x4 / y4 + (x3 - x2) / y3)
        xq = x4 - y4 * (x3 - x4) / (y3 - y4)
        temp = 1.0 / sqrt(1.0 + d ** 2)
        p = [(xq - x - d * y) * temp for x, y in zip(xs, ys)]
        temp = xq - x4
        b = (temp * d + y4) / (temp - y4 * d)
        z = (p[0] * p[1] * pa) / (p[2] * p[3] * 2.0 * g * t) * (
            1.0 + 2.0 / (3.0 + 3.0 * nu) * (b ** 2 + b * d + d ** 2))
    elif par34 and par23:  # parallelogram
        p = [real(1.0)] * 4
        d = -0.5 * (x4 / y4 + (x3 - x2) / y3 + (y3 - y4) / (x3 - x4))
        z = pa / (2.0 * g * t) * (1.0 + 2.0 * d ** 2 / (1.0 + nu))
    else:
        xq = x4 - (x3 - x4) / (y3 - y4) * y4
        temp = y3 * x4 - y4 * (x3 - x2)
        xp = x2 * y3 * x4 / temp
        yp = x2 * y3 * y4 / temp
        xl = sqrt((xq - xp) ** 2 + yp ** 2)
        d = (xq - xp) / yp
        temp = yp / xl
        p = [temp * (xq - x - d * y) for x, y in zip(xs, ys)]
        c = xl / p[0] - d
        b = xl / p[3] - c
        a = xl / p[1] - d

        def pw(u):
            u2 = u * u
            u3 = u2 * u
            u4 = u3 * u
            return u2, u3, u4, u4 * u

        (a2, a3, a4, a5), (b2, b3, b4, b5), (c2, c3, c4, c5), (d2, d3, d4, d5) = map(pw, (a, b, c, d))

        def term(u, v, u3, v3, u5, v5):
            return (u + v + 2.0 * (u3 + v3) / 3.0 + 0.2 * (u5 + v5)) * np.log(abs(u + v))

        temp = 0.5 * p[0] * p[1] * p[2] * p[3] / xl ** 2
        term0 = term(a, b, a3, b3, a5, b5)
        term1 = term(c, d, c3, d3, c5, d5)
        term2 = term(b, c, b3, c3, b5, c5)
        term3 = term(d, a, d3, a3, d5, a5)
        term4 = 0.1 * ((a2 - c2) * (b3 - d3) + (b2 - d2) * (a3 - c3))
        term5 = 0.2 * ((a - c) * (b4 - d4) + (b - d) * (a4 - c4))
        f = temp * (term0 + term1 - term2 - term3 + term4 - term5)
        z = p[0] * p[1] / (p[2] * p[3] * 2.0 * g * t) * (pa + 4.0 / (1.0 + nu) * (f - 2.0 * pa / 3.0))

    xl13 = sqrt(x3 ** 2 + y3 ** 2)
    xl24 = sqrt((x4 - x2) ** 2 + y4 ** 2)
    u = (x3 / xl13, (x4 - x2) / xl24) * 2
    v = (y3 / xl13, y4 / xl24) * 2
    temp = x4 * y3 - x3 * y4
    a1 = -(0.5 * x2 * y4 * xl13 / temp)
    a2 = 0.5 * x2 * y3 * xl24 / (temp - x2 * (y3 - y4))
    avec = (a1, a2, -a1, -a2)
    # vleft[i] = u_i VI + v_i VJ, the basic-coordinate direction of corner i's shear flow
    vleft = [[u[i] * vi[k] + v[i] * vj[k] for k in range(3)] for i in range(4)]
    side = {"x2": x2, "x3": x3, "y3": y3, "x4": x4, "y4": y4, "v12dk": v12dk}
    return t, z, p, avec, vleft, side


def stiffness(model, elem):
    _, z, _, avec, vleft, _ = _panel(model, elem)
    w6 = np.zeros((4, 6))
    w6[:, :3] = np.array(vleft) * np.array(avec)[:, None]
    w6 = w6.ravel()
    return np.outer(w6, w6) / (2.0 * z)


def recover(model, elem, disp):
    # SPANL1 / SPANL2 run in single precision (SMMATS accumulates each corner in double)
    real = np.float32
    t, z, p, avec, vleft, s = _panel(model, elem, real)
    spcon = -1.0 / (2.0 * z * t)
    s1bar = real(0.0)
    for i, gid in enumerate(elem.grids):
        si = [float(spcon * vleft[i][k] * avec[i]) for k in range(3)]
        s1bar += real(np.dot(si, np.asarray(disp[gid], dtype=real)[:3].astype(float)))

    ratio = (p[1] / p[0], p[0] * p[1] / p[2] ** 2, p[0] * p[1] / p[3] ** 2)
    tau = (ratio[0] * s1bar, s1bar / ratio[0], ratio[1] * s1bar, ratio[2] * s1bar)
    tauavg = 0.25 * (tau[0] + tau[1] + tau[2] + tau[3])
    taumax = max(abs(x) for x in tau)
    # SPANL2 prints a safety margin only for a material shear allowable (MAT1 SS), which
    # the model does not read, so the column is blank as in the listing

    rq = (p[0] / p[2] * t, p[0] * p[1] / (p[2] * p[3]) * t, p[1] / p[3] * t, -s["v12dk"] / 2.0)
    x2, x3, y3, x4, y4 = s["x2"], s["x3"], s["y3"], s["x4"], s["y4"]
    rk = (x2 / 2.0, np.sqrt((x3 - x2) ** 2 + y3 ** 2) / 2.0,
          np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2) / 2.0, np.sqrt(x4 ** 2 + y4 ** 2) / 2.0)
    scale = (t,) + rq[:3]
    q = [s1bar * scale[i] / np.sqrt(1.0 + (rq[3] / rk[i]) ** 2) for i in range(4)]
    kick = [-(q[0] + q[3]) * rq[3], -(q[0] + q[1]) * rq[3], -(q[1] + q[2]) * rq[3], -(q[2] + q[3]) * rq[3]]
    f1, f2, f5, f6 = q[3] * rk[3], q[0] * rk[0], q[1] * rk[1], q[2] * rk[2]
    forces = [f1, f2, -f2, -f5, f5, f6, -f6, -f1]
    for k, qq in zip(kick, q):
        forces += [k, qq]
    return {FORCES: {elem.id: [float(x) for x in forces]},
            STRESSES: {elem.id: [float(taumax), float(tauavg)]}}
