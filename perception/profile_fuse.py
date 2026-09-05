#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where the frame time actually goes, on a REAL captured frame.

    python3 capture_frame.py --out /tmp/frame.npz     # with the stack running
    OMP_NUM_THREADS=1 python3 profile_fuse.py /tmp/frame.npz

Elimination had gone as far as it could: halving the image rows moved the node's
frame time by 0%, halving rays_per_col by 12%, and neither the ray/box clip nor
batching `np.maximum.at` moved it either. That pattern says the cost is in
something none of those knobs touch, and the way to find out which is to look
rather than to reason.
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.fuse import fuse_frame, stereo_from_camera_info
from perception.grid import AndertGrid
from perception.inverse_sensor import ProfileParams

RES, MIN_X, MIN_Y, NX, NY = 0.05, -4.1, -2.6, 144, 104


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--row-decim", type=int, default=4)
    ap.add_argument("--rpc", type=int, default=4)
    ap.add_argument("--plane-height", type=float, default=1.0)
    ap.add_argument("--plane-tol", type=float, default=0.25)
    ap.add_argument("--z-max", type=float, default=6.0)
    ap.add_argument("--repeat", type=int, default=8)
    a = ap.parse_args()

    d = np.load(a.frame, allow_pickle=True)
    depth_full = d["depth"]
    st = stereo_from_camera_info(d["K"], int(d["width"]), int(d["height"]),
                                 decim=a.decim, row_decim=a.row_decim,
                                 z_min_m=0.4, z_max_m=a.z_max)
    depth = depth_full[::a.row_decim, ::a.decim]
    cam, R = d["cam"], d["R"]
    prof = ProfileParams()
    print("OMP_NUM_THREADS=%s  raster %dx%d  fx %.1f fy %.1f  rays/col %d"
          % (os.environ.get("OMP_NUM_THREADS", "unset"), st.width, st.height,
             st.fx, st.fy, a.rpc))
    finite = np.isfinite(depth)
    print("depth %.1f%% finite, median %.2f m, p95 %.2f m"
          % (100.0 * finite.mean(),
             float(np.median(depth[finite])) if finite.any() else -1,
             float(np.percentile(depth[finite], 95)) if finite.any() else -1))

    def once():
        g = AndertGrid(NX, NY, RES, MIN_X, MIN_Y)
        fuse_frame(depth, cam, R, st, prof, g, a.plane_height, a.plane_tol,
                   rays_per_col=a.rpc)

    once()
    ts = []
    for _ in range(a.repeat):
        t0 = time.perf_counter(); once(); ts.append((time.perf_counter()-t0)*1e3)
    print("\nwall clock: p50 %.1f ms  p95 %.1f ms  (n=%d)"
          % (np.percentile(ts, 50), np.percentile(ts, 95), len(ts)))

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(max(3, a.repeat // 2)):
        once()
    pr.disable()
    print("\ncumulative time, top 18 (profiler overhead inflates the total; the")
    print("SHARES are what to read, not the absolute numbers)")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
    return 0


if __name__ == "__main__":
    sys.exit(main())
