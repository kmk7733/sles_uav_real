#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the Euclidean goal cost point at the gap, or away from it?

geodesic.py exists for exactly one failure: pressed against something with the
goal behind it, ||p - goal|| increases in every direction that actually helps,
so the planner sits still. This asks whether the CURRENT map is that case.

For a ring of candidate steps around the vehicle it prints, for each:
    d_eu    Euclidean distance to the goal      (what the cost uses now)
    d_geo   geodesic distance through free space (what it used before)
and marks which directions each metric calls an improvement. If Euclidean says
"every useful direction is worse" while geodesic says otherwise, the switch to
Euclidean is what stalled it -- not the sampler and not r_safe.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from planar_map import PlanarOccupancy
from planar_safety import safe_radius
from geodesic import CostToGo


def main():
    rospy.init_node("diag_cost", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                         rospy.get_param("~r_perc", 0.10),
                         rospy.get_param("~r_track", 0.05),
                         rospy.get_param("~d_clr", 0.05))
    step = float(rospy.get_param("~step", 0.60))

    g = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(g, occ_thresh=50,
                                                  unknown_unsafe=True)
    p = rospy.wait_for_message("/robot/pose_world", PoseStamped, timeout=10.0)
    x0, y0 = p.pose.position.x, p.pose.position.y
    occ.clear_disc(x0, y0, r_safe + 0.05)

    ctg = CostToGo(occ, goal, r_safe, unknown_free=True)
    here_eu = float(np.linalg.norm(np.array([x0, y0]) - goal))
    here_geo = float(ctg.query(np.array([[x0, y0]]))[0])

    print("\nvehicle (%.2f, %.2f)   goal (%.2f, %.2f)   r_safe %.3f  step %.2f m"
          % (x0, y0, goal[0], goal[1], r_safe, step))
    print("at the vehicle:  d_eu = %.3f    d_geo = %.3f" % (here_eu, here_geo))
    print("%s" % ctg.describe())
    print("")
    print("  bearing      point        clear   d_eu    dEU     d_geo   dGEO")
    print("  " + "-" * 62)

    best_eu = best_geo = None
    for deg in range(0, 360, 30):
        th = np.radians(deg)
        px, py = x0 + step*np.cos(th), y0 + step*np.sin(th)
        clr = float(occ.clearance(np.array([px]), np.array([py]))[0])
        eu = float(np.linalg.norm(np.array([px, py]) - goal))
        ge = float(ctg.query(np.array([[px, py]]))[0])
        d_eu, d_ge = eu - here_eu, ge - here_geo
        ok = clr >= r_safe
        if ok:
            if best_eu is None or d_eu < best_eu[1]: best_eu = (deg, d_eu)
            if best_geo is None or d_ge < best_geo[1]: best_geo = (deg, d_ge)
        print("  %4d deg  (%6.2f,%6.2f)  %5.2f%s  %6.2f %+6.2f  %6.2f %+6.2f%s"
              % (deg, px, py, clr, " " if ok else "X",
                 eu, d_eu, ge, d_ge,
                 "  <- both improve" if (ok and d_eu < 0 and d_ge < 0) else
                 ("  <- GEO only" if (ok and d_ge < 0 <= d_eu) else "")))

    print("")
    if best_eu:
        print("  best VALID direction by Euclidean : %3d deg  (%+.2f m)"
              % best_eu)
    if best_geo:
        print("  best VALID direction by geodesic  : %3d deg  (%+.2f m)"
              % best_geo)
    if best_eu and best_geo and best_eu[1] >= 0 > best_geo[1]:
        print("\n  => EUCLIDEAN IS STUCK: no valid direction reduces it, so the")
        print("     weighted mean has nothing to pull it anywhere. Geodesic")
        print("     still has a downhill direction. This is the local minimum")
        print("     geodesic.py was written to remove.")
    elif best_eu and best_eu[1] < 0:
        print("\n  => Euclidean still has a downhill valid direction; the stall")
        print("     is NOT a goal-metric local minimum.")


if __name__ == "__main__":
    main()
