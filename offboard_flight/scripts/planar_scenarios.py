#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Occupancy grids for exercising the planar planner off-vehicle.

Geometry defaults match the real map this stack publishes -- 144 x 104 cells at
0.05 m, i.e. a 7.2 x 5.2 m arena -- so a solve here costs what a solve costs on
the Jetson, and the safety radius means the same thing relative to the free
space as it does in the flying area.

`load_grid_npy` / `save_grid_npy` round-trip a real captured /grid_map, so the
demo can run against recorded perception output rather than these synthetic
stand-ins. See demo_planar.py --grid.
"""

import numpy as np

from planar_map import PlanarOccupancy

# The live pipeline's grid: depth_to_grid.py at 0.05 m over the arena.
RES = 0.05
WIDTH = 144
HEIGHT = 104
ORIGIN = (-3.6, -2.6)

FREE = 0
UNKNOWN = -1
OCCUPIED = 100


def blank(width=WIDTH, height=HEIGHT, fill=FREE):
    return np.full((height, width), fill, dtype=np.int16)


def _world_mesh(values, res=RES, origin=ORIGIN):
    h, w = values.shape
    ys, xs = np.mgrid[0:h, 0:w]
    return (origin[0] + (xs + 0.5) * res, origin[1] + (ys + 0.5) * res)


def _wrap(values, res=RES, origin=ORIGIN, unknown_unsafe=False, **kw):
    return PlanarOccupancy.from_values(values, res, origin,
                                       unknown_unsafe=unknown_unsafe,
                                       frame_id="vicon/world", **kw)


def open_arena(**kw):
    """Nothing but free space."""
    return _wrap(blank(), **kw)


def pillar(px=0.5, py=0.0, radius=0.30, **kw):
    """One circular pillar. The canonical nonconvex case: safe left, safe
    right, unsafe straight through the average of the two."""
    v = blank()
    wx, wy = _world_mesh(v)
    v[(wx - px) ** 2 + (wy - py) ** 2 <= radius * radius] = OCCUPIED
    return _wrap(v, **kw)


def wall(x=0.5, thickness=0.10, gap_centre=None, gap_half_width=0.5, **kw):
    """Vertical wall, optionally with a gap to fly through."""
    v = blank()
    wx, wy = _world_mesh(v)
    m = np.abs(wx - x) <= 0.5 * thickness
    if gap_centre is not None:
        m &= np.abs(wy - gap_centre) > gap_half_width
    v[m] = OCCUPIED
    return _wrap(v, **kw)


def slalom(xs=(-0.8, 0.4, 1.6), ys=(0.7, -0.7, 0.7), radius=0.28, **kw):
    """A few staggered pillars -- forces a committed left/right sequence."""
    v = blank()
    wx, wy = _world_mesh(v)
    for px, py in zip(xs, ys):
        v[(wx - px) ** 2 + (wy - py) ** 2 <= radius * radius] = OCCUPIED
    return _wrap(v, **kw)


def corridor(half_width=0.9, **kw):
    """Two long walls with a gap between them."""
    v = blank()
    wx, wy = _world_mesh(v)
    v[np.abs(wy) >= half_width] = OCCUPIED
    return _wrap(v, **kw)


def partially_observed(fraction=0.55, seed=0, **kw):
    """Free space in front, unknown behind -- what the ZED actually gives you.

    Roughly 60-70% of the live grid is unobserved at any moment. With unknown
    treated as unsafe (which it must be), that is the single biggest constraint
    on where the planner may fly, and it is worth seeing in a scenario rather
    than discovering on the vehicle.
    """
    v = blank(fill=UNKNOWN)
    wx, wy = _world_mesh(v)
    rng = np.random.RandomState(seed)
    # a forward-facing observed wedge from the origin
    ang = np.arctan2(wy, wx)
    rad = np.hypot(wx, wy)
    seen = (np.abs(ang) < np.radians(55.0)) & (rad < 4.0)
    v[seen] = FREE
    v[seen & (rng.rand(*v.shape) < 0.004)] = OCCUPIED
    return _wrap(v, unknown_unsafe=True, **kw)


# Gap widths are chosen against the DEFAULT r_safe of 0.59 m (see
# planar_safety.safe_radius). A gap is passable only if its half-width exceeds
# r_safe, so 0.75 m clears with 0.16 m to spare and 0.45 m cannot be flown at
# all. Both are worth having: "narrow" is not a broken scenario, it is what a
# 0.31 m quadrotor with 0.18 m of depth uncertainty is actually not allowed to
# attempt, and the planner stopping short of it is the correct outcome.
SCENARIOS = {
    "open": open_arena,
    "pillar": pillar,
    "wall": lambda **kw: wall(gap_centre=None, **kw),
    "gap": lambda **kw: wall(gap_centre=0.8, gap_half_width=0.75, **kw),
    "narrow": lambda **kw: wall(gap_centre=0.8, gap_half_width=0.45, **kw),
    "slalom": slalom,
    "corridor": corridor,
    "partial": partially_observed,
}


def save_grid_npy(path, occ):
    """Persist an occupancy set plus its geometry, for offline replay."""
    np.savez(path, unsafe=occ.unsafe, res=occ.res, origin=np.asarray(occ.origin),
             frame_id=occ.frame_id)


def load_grid_npy(path):
    """Load a grid saved by save_grid_npy (or capture_grid.py)."""
    z = np.load(path, allow_pickle=False)
    return PlanarOccupancy(z["unsafe"], float(z["res"]),
                           tuple(z["origin"].tolist()),
                           frame_id=str(z["frame_id"]) if "frame_id" in z
                           else "map")


def ascii_map(occ, traj=None, goal=None, start=None, cols=72, rows=26):
    """Coarse text rendering, so the demo shows something without a GUI."""
    xmin, ymin, xmax, ymax = occ.bounds
    out = [[" "] * cols for _ in range(rows)]

    def cell(x, y):
        cx = int((x - xmin) / max(xmax - xmin, 1e-9) * (cols - 1))
        cy = int((y - ymin) / max(ymax - ymin, 1e-9) * (rows - 1))
        if 0 <= cx < cols and 0 <= cy < rows:
            return cx, cy
        return None

    ys, xs = np.nonzero(occ.unsafe)
    wx, wy = occ.cell_to_world(xs, ys)
    for x, y in zip(wx, wy):
        c = cell(x, y)
        if c:
            out[c[1]][c[0]] = "#"

    if traj is not None:
        for x, y in np.asarray(traj)[:, :2]:
            c = cell(x, y)
            if c and out[c[1]][c[0]] == " ":
                out[c[1]][c[0]] = "."
    for pt, ch in ((start, "S"), (goal, "G")):
        if pt is not None:
            c = cell(pt[0], pt[1])
            if c:
                out[c[1]][c[0]] = ch

    # row 0 is ymin, so print top-down
    border = "+" + "-" * cols + "+"
    lines = [border]
    for r in reversed(out):
        lines.append("|" + "".join(r) + "|")
    lines.append(border)
    return "\n".join(lines)
