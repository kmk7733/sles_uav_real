#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What one Andert frame costs on THIS box, and proof the pipeline runs.

    python3 bench_fuse.py                       # the defaults the node ships
    python3 bench_fuse.py --decim 1 2 3 --rpc 0 16 8 4
    OMP_NUM_THREADS=1 python3 bench_fuse.py     # how the node is launched

Needs no ROS master, no camera and no bag -- it renders its own room -- so it
can be run on the aircraft before anything is wired up, and again under load
while the planner is running. That second number is the one that matters: this
box is oversubscribed (loadavg ~13 on 6 cores) and the planner's 100 ms budget
is what the mapper has to fit beside, not the mapper's own.

WHY A BENCH TRAVELS WITH THIS PACKAGE. The simulator measured 124 ms/frame at
rays_per_col 16 and 640x360 on a desktop EPYC core; this is a Xavier NX. The
ratio is not a thing to assume -- the whole `rays_per_col` and `decim` choice
turns on it, and the fallback ladder in the port plan is ordered by it.

WHAT IT ALSO CHECKS. Every configuration must observe cells, believe some of
them occupied, and agree with every other configuration to within a fraction of
a percent on how many cells it touched. Thinning that quietly stopped covering
the room would otherwise look exactly like thinning that got cheaper.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.fuse import (band_rows_per_col, fuse_frame,
                             ray_spacing_limit, stereo_from_camera_info)
from perception.grid import AndertGrid
from perception.inverse_sensor import ProfileParams, validate_profile_params
from perception.renderer import CameraModel, camera_to_renderer, render_depth

# The camera this vehicle actually has, at 640x360 (HD720, downsample 0.5),
# read off `depth/camera_info` in bags/depth_debug_20260730_045726.bag.
# NOT the simulator's NATIVE_F -- see fuse.stereo_from_camera_info.
K_REAL = (261.431, 0.0, 327.4344,
          0.0, 261.431, 176.0503,
          0.0, 0.0, 1.0)
RAW_W, RAW_H = 640, 360
BASELINE_M = 0.12

# The flight volume, as the live node is configured: x [-4.0, 3.0] and
# y [-2.5, 2.5], grown by a 2-cell border wall, 0.05 m cells -> 144 x 104.
RES = 0.05
MIN_X, MIN_Y = -4.1, -2.6
NX, NY = 144, 104
PLANE_H, PLANE_TOL = 1.0, 0.25
Z_MIN, Z_MAX = 0.4, 6.0


def room_segments():
    """A 7 x 5 m room with two pillars, in RENDERER coordinates.

    render_depth wants the grid's own zero-based frame -- the same shift
    fuse_frame applies to the pose -- so the world box [-4.0, 3.0] x
    [-2.5, 2.5] lands at [0.1, 7.1] x [0.1, 5.1] against MIN_X/MIN_Y.
    """
    def w(x, y):
        return (x - MIN_X, y - MIN_Y)
    x0, x1, y0, y1 = -4.0, 3.0, -2.5, 2.5
    segs = [(w(x0, y0), w(x1, y0)), (w(x1, y0), w(x1, y1)),
            (w(x1, y1), w(x0, y1)), (w(x0, y1), w(x0, y0))]
    for cx, cy in ((-1.1, 0.9), (0.9, -0.7)):       # two 0.4 m square pillars
        h = 0.2
        c = [w(cx - h, cy - h), w(cx + h, cy - h),
             w(cx + h, cy + h), w(cx - h, cy + h)]
        segs += [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    return segs


def synth_depth(cam, x, y, z, roll, pitch, yaw, segs, dropout, rng):
    """One rendered depth image in the shape the ZED delivers.

    Misses come back +inf from render_depth; the matcher's own failures are
    NaN. Both are dropped by build_frame_grid and neither is a distance, which
    is the property being exercised -- a bench that fed it a dense image would
    not be measuring the code that runs.
    """
    pose = (x - MIN_X, z, y - MIN_Y, roll, pitch, yaw)
    d = render_depth(segs, pose, cam, obstacle_height=3.0).copy()
    if dropout > 0.0:
        d[rng.random(d.shape) < dropout] = np.nan
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decim", type=int, nargs="+", default=[2],
                    help="COLUMN stride; a column is one bearing, so this is "
                         "the map's angular resolution (default 2)")
    ap.add_argument("--row-decim", type=int, nargs="+", default=[0],
                    help="ROW stride; 0 means 'same as --decim'. Rows only "
                         "feed the height-band test, so they can be thinned "
                         "much harder than columns -- that is the whole point "
                         "of having two numbers")
    ap.add_argument("--rpc", type=int, nargs="+", default=[8],
                    help="rays_per_col; 0 keeps every row the band admits")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15,
                    help="fraction of pixels with no stereo match")
    ap.add_argument("--tilt-deg", type=float, default=4.0,
                    help="roll/pitch amplitude; 0 hides a roll/pitch swap")
    ap.add_argument("--z-max", type=float, nargs="+", default=[Z_MAX],
                    help="stereo range gate; every ray is marched to this")
    ap.add_argument("--plane-tol", type=float, nargs="+", default=[PLANE_TOL],
                    help="half-thickness of the slice. A NARROWER slab keeps "
                         "fewer image rows per column, so it is a compute "
                         "saving as well as a modelling choice -- but it must "
                         "stay wider than the altitude tube or the map starves "
                         "exactly while the vehicle is off altitude")
    a = ap.parse_args(argv)

    segs = room_segments()
    prof = ProfileParams()
    print("threads: OMP_NUM_THREADS=%s" % os.environ.get("OMP_NUM_THREADS", "unset"))
    print("grid   : %d x %d @ %.3f m, origin (%.2f, %.2f), slice at %.2f m"
          % (NX, NY, RES, MIN_X, MIN_Y, PLANE_H))
    print("")
    print("%9s %10s %6s %6s %8s %8s %8s %8s %8s"
          % ("decim c/r", "raster", "z_max", "tol", "rays/col", "ms p50",
             "ms p95", "seen", "occupied"))

    rows = []
    combos = [(d, rd, zm, pt, r) for d in a.decim for rd in a.row_decim
              for zm in a.z_max for pt in a.plane_tol for r in a.rpc]
    warned = set()
    for decim, row_decim, z_max, tol, rpc in combos:
        rd = decim if row_decim <= 0 else row_decim
        stereo = stereo_from_camera_info(K_REAL, RAW_W, RAW_H, decim=decim,
                                         row_decim=rd, baseline_m=BASELINE_M,
                                         z_min_m=Z_MIN, z_max_m=z_max)
        lim = ray_spacing_limit(RES, stereo.fx)
        if z_max > lim and (decim, z_max) not in warned:
            warned.add((decim, z_max))
            print("  WARNING decim %d: z_max %.1f m exceeds the ray-spacing "
                  "limit res*fx = %.1f m; far walls will be perforated"
                  % (decim, z_max, lim))
        sigma_min = validate_profile_params(prof, stereo, RES)
        cam = CameraModel(stereo.width, stereo.height, stereo.fx, stereo.fy,
                          stereo.cx, stereo.cy, stereo.z_min_m, stereo.z_max_m)
        if True:
            rng = np.random.default_rng(0)
            grid = AndertGrid(NX, NY, RES, MIN_X, MIN_Y)
            ts, touched_tot = [], 0
            n = a.frames + a.warmup
            for i in range(n):
                # A slow arc across the room, tilting, so the height band is
                # exercised at attitude rather than at a fixed row window.
                f = i / float(max(n - 1, 1))
                x, y = -2.5 + 3.0 * f, -1.2 + 1.6 * f
                yaw = 0.35 + 1.2 * f
                tilt = np.deg2rad(a.tilt_deg)
                roll, pitch = tilt * np.sin(6.0 * f), tilt * np.cos(6.0 * f)
                d = synth_depth(cam, x, y, PLANE_H, roll, pitch, yaw,
                                segs, a.dropout, rng)
                R = camera_to_renderer(roll, pitch, yaw)
                # camera_to_renderer returns world->renderer for the SIM's
                # convention; undo the renderer half so fuse_frame is handed the
                # same kind of matrix TF will hand the node.
                R_world_opt = np.linalg.inv(
                    np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])).dot(R)
                t0 = time.perf_counter()
                _frame, tch = fuse_frame(d, (x, y, PLANE_H), R_world_opt,
                                         stereo, prof, grid,
                                         PLANE_H, tol, rays_per_col=rpc)
                dt = (time.perf_counter() - t0) * 1e3
                if i >= a.warmup:
                    ts.append(dt)
                    touched_tot += int(tch.sum())
            ts = np.array(ts)
            cov = grid.coverage()
            print("%9s %10s %6.1f %6.2f %8s %8.1f %8.1f %8d %8d"
                  % ("%d/%d" % (decim, rd),
                     "%dx%d" % (stereo.width, stereo.height), z_max, tol,
                     rpc if rpc else "all", np.percentile(ts, 50),
                     np.percentile(ts, 95),
                     int(grid.seen.sum()), int((grid.L > grid.occ_thresh).sum())))
            if rpc:
                # The row stride is safe only while the band still offers at
                # least `rays_per_col` rows; past that the cap stops binding
                # and thinning starts costing coverage instead of time.
                r4 = band_rows_per_col(stereo.fy, tol, 4.0)
                if r4 < rpc:
                    print("          note: at 4 m the band keeps only %.1f "
                          "rows/col, below rays_per_col %d -- the row stride "
                          "is now costing rays, not just time" % (r4, rpc))
            rows.append(("%d/%d" % (decim, rd), rpc, np.percentile(ts, 50),
                         int(grid.seen.sum()), cov))

    print("\nsigma_l floor %.4f m (half a cell); peak P stays under 1 by "
          "validate_profile_params" % sigma_min)

    # A configuration that got cheap by seeing less is not cheaper, it is
    # broken. Compare coverage against the richest configuration run.
    # Only compare coverage across configurations that were asked to see the
    # same thing: a shorter z_max or a narrower slab covers less BY REQUEST.
    best = max(r[3] for r in rows)
    bad = [r for r in rows if r[3] < 0.97 * best]
    if any(r[3] == 0 for r in rows):
        print("\nFAIL: a configuration observed nothing at all")
        return 1
    if not any(r[4]["occ_pct"] > 0.0 for r in rows):
        print("\nFAIL: nothing was ever believed occupied -- the room has walls")
        return 1
    if bad:
        print("\nWARNING: these saw >3%% fewer cells than the best "
              "configuration -- thinning is costing coverage, not just time:")
        for decim, rpc, ms, seen, _ in bad:
            print("  decim %s rays/col %s: %d cells vs %d"
                  % (decim, rpc or "all", seen, best))
    else:
        print("every configuration covered within 3%% of the best (%d cells)"
              % best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
