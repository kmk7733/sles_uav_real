#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is r_quad counted twice in the collision check?

Part 1 answers it with a synthetic map where the true distance is known by
construction: one occupied cell, one query point a measured distance away. If
clearance() came back reduced by r_quad, the number would be short by 0.31 m.

Part 2 then measures, on the LIVE map, where the conservatism actually comes
from -- because "the vehicle behaved timidly" has several possible causes and
r_quad is only one of them.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius


def part1():
    print("=" * 70)
    print(" PART 1 -- synthetic map, true distance known by construction")
    print("=" * 70)
    res = 0.05
    H = W = 81
    unsafe = np.zeros((H, W), dtype=bool)
    unsafe[40, 40] = True                      # one occupied cell
    # origin chosen so cell (40,40) has its CENTRE at world (0,0)
    origin = (-(40 + 0.5) * res, -(40 + 0.5) * res)
    occ = PlanarOccupancy(unsafe, res, origin)

    r_safe = safe_radius(0.31, 0.18, 0.05, 0.05)
    val = PlanarSafetyValidator(occ, r_safe)

    print("  one occupied cell, centre at world (0.00, 0.00)")
    print("  r_safe = %.3f   edt_margin = %.3f m (= res/2)"
          % (r_safe, occ.edt_margin))
    print("")
    print("   true dist   clearance()   clearance+margin   gate(>=r_safe)")
    for d in (0.20, 0.31, 0.59, 0.60, 0.90, 1.00):
        c = float(occ.clearance(np.array([d]), np.array([0.0]))[0])
        ok = bool(val.nodes_safe(np.array([[[d, 0.0]]]))[0])
        print("   %6.2f m    %7.3f       %7.3f            %s"
              % (d, c, c + occ.edt_margin, "PASS" if ok else "reject"))

    print("")
    print("  READ: clearance() == true distance - edt_margin. r_quad does NOT")
    print("  appear. If it were double-counted, the 1.00 m row would read")
    print("  ~0.69 and the gate would first PASS somewhere near 0.90 m.")
    c = float(occ.clearance(np.array([1.0]), np.array([0.0]))[0])
    print("  verdict: %s"
          % ("NO double counting (clearance tracks true distance)"
             if abs(c - (1.0 - occ.edt_margin)) < 1e-6
             else "UNEXPECTED -- clearance %.3f at true 1.000" % c))

    # Where does the gate first open?
    ds = np.arange(0.30, 1.20, 0.005)
    cl = occ.clearance(ds, np.zeros_like(ds))
    first = ds[np.argmax(cl >= r_safe)]
    print("  gate first opens at true distance %.3f m  (= r_safe + edt_margin"
          " = %.3f)" % (first, r_safe + occ.edt_margin))


def part2():
    print("")
    print("=" * 70)
    print(" PART 2 -- live map: where the conservatism actually comes from")
    print("=" * 70)
    msg = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    r_safe = safe_radius(0.31, 0.18, 0.05, 0.05)
    d_infl = r_safe + 0.35

    for uu in (True, False):
        occ = PlanarOccupancy.from_occupancy_grid_msg(msg, occ_thresh=50,
                                                      unknown_unsafe=uu)
        ys, xs = np.mgrid[0:occ.H, 0:occ.W]
        wx = occ.origin[0] + (xs + 0.5) * occ.res
        wy = occ.origin[1] + (ys + 0.5) * occ.res
        cl = occ.clearance(wx, wy)
        n = cl.size
        print("")
        print("  unknown_unsafe = %s   (unsafe cells %d of %d)"
              % (uu, int(occ.unsafe.sum()), n))
        print("    clearance >= r_safe   %.3f : %6d cells (%4.1f%%)  <- flyable"
              % (r_safe, int((cl >= r_safe).sum()), 100.0*(cl >= r_safe).sum()/n))
        print("    clearance >= d_infl   %.3f : %6d cells (%4.1f%%)  <- cost-free"
              % (d_infl, int((cl >= d_infl).sum()), 100.0*(cl >= d_infl).sum()/n))
        print("    median clearance            : %.3f m" % float(np.median(cl)))

    print("")
    print("  READ: the gap between those two rows is the SOFT cost. Cells")
    print("  between r_safe and d_influence are legal but penalised, and with")
    print("  w_obs = 20 that is what makes the plan hug open space.")


if __name__ == "__main__":
    rospy.init_node("diag_rsafe", anonymous=True, disable_signals=True)
    part1()
    try:
        part2()
    except Exception as e:
        print("\n  (part 2 skipped: %s)" % e)
