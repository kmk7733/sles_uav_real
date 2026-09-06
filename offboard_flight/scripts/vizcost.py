#!/usr/bin/env python3
"""What does each per-tick publisher actually cost, on this box, on this grid?

No ROS master, no node: builds a grid the size the vehicle flies and times the
work each _publish_* does, minus the publish() call itself. That is the part
that runs INSIDE plan_once, on the solve's own thread.
"""
import os, sys, time
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, "/home/rogx/catkin_ws/src")

W, H, RES = 144, 104, 0.05          # the flown grid
R_EFF, R_VIZ = 0.51 - 0.31, 0.31    # r_safe - r_quad, and r_quad
N_NODES, N_VIZ = 31, 30             # horizon+1, viz_rollouts

rng = np.random.RandomState(0)
occupied = rng.rand(H, W) < 0.04
unknown = rng.rand(H, W) < 0.55
unsafe = occupied | unknown

def timeit(fn, reps=30):
    for _ in range(3):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter()-t0)*1e3)
    a = np.array(t)
    return float(np.median(a)), float(np.percentile(a, 95))

def inflated():
    # what _publish_inflated does: EDT on the OCCUPIED set, threshold, and then
    # _publish_outer_ring's second threshold on the same field
    d_occ = distance_transform_edt(~occupied, sampling=RES)
    inner = d_occ < R_EFF
    grown = inner | unknown
    ring = (d_occ < (R_EFF + R_VIZ)) & ~inner
    out = np.full((H, W), -1, dtype=np.int8)
    out[grown] = 100
    out2 = np.full((H, W), -1, dtype=np.int8)
    out2[ring] = 100
    return out.ravel().tolist(), out2.ravel().tolist()

def rollouts():
    X = rng.randn(N_VIZ, N_NODES, 6)
    arr = []
    for i in range(N_VIZ):
        pts = [(float(X[i, k, 0]), float(X[i, k, 1]), 1.0)
               for k in range(N_NODES)]
        arr.append(pts)
    return arr

def path():
    return [(float(i), float(i), 1.0) for i in range(N_NODES)]

def goal():
    return (2.0, 0.0, 1.0)

print("grid %dx%d @ %.2f m, %d viz rollouts x %d nodes\n" % (W, H, RES, N_VIZ, N_NODES))
print("  %-34s %8s %8s" % ("per plan tick", "p50 ms", "p95 ms"))
print("  %-34s %8s %8s" % ("-"*34, "-"*8, "-"*8))
tot = 0.0
for name, fn in (("~inflated + ~inflated_outer", inflated),
                 ("~rollouts (MarkerArray)", rollouts),
                 ("~nominal_path", path),
                 ("~goal_marker", goal)):
    p50, p95 = timeit(fn)
    tot += p50
    print("  %-34s %8.2f %8.2f" % (name, p50, p95))
print("  %-34s %8.2f" % ("TOTAL", tot))
print("\n  against a %d ms budget at plan_rate 5" % 200)
