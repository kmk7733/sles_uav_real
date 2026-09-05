#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which cost term decides the direction?

The goal term wants +x. The obstacle term wants clearance >= d_influence, and
d_influence = r_safe + 0.35 is larger than the clearance available anywhere
near the gap. If the obstacle term outweighs the goal term, MPPI's weighted
mean prefers open space BEHIND the vehicle over progress toward the goal, and
the plan stays local while the rollouts fan out everywhere.

This evaluates the ACTUAL cost of a straight, feasible 2 s move in each
direction, term by term, using the same weights the node uses.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius


def main():
    rospy.init_node("diag_terms", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                         rospy.get_param("~r_perc", 0.10),
                         rospy.get_param("~r_track", 0.05),
                         rospy.get_param("~d_clr", 0.05))
    w_goal = float(rospy.get_param("~w_goal", 1.0))
    w_term_pos = float(rospy.get_param("~w_term_pos", 10.0))
    w_term_vel = float(rospy.get_param("~w_term_vel", 2.0))
    w_obs = float(rospy.get_param("~w_obs", 20.0))
    d_infl = rospy.get_param("~d_influence", None)
    d_infl = (r_safe + 0.35) if d_infl is None else float(d_infl)
    N, dt, v = 20, 0.1, float(rospy.get_param("~v_probe", 0.5))

    g = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(g, occ_thresh=50,
                                                  unknown_unsafe=True)
    p = rospy.wait_for_message("/robot/pose_world", PoseStamped, timeout=10.0)
    x0, y0 = p.pose.position.x, p.pose.position.y
    occ.clear_disc(x0, y0, r_safe + 0.05)
    val = PlanarSafetyValidator(occ, r_safe)

    print("\nvehicle (%.2f, %.2f)  goal (%.2f, %.2f)" % (x0, y0, goal[0], goal[1]))
    print("r_safe %.3f   d_influence %.3f   w_obs %.1f  w_goal %.1f  "
          "w_term_pos %.1f" % (r_safe, d_infl, w_obs, w_goal, w_term_pos))
    print("probe: straight line, %.1f m/s for %.1f s = %.2f m, arriving stopped"
          % (v, N*dt, v*N*dt))
    print("")
    print("  bearing   J_goal   J_term   J_obs   J_TOTAL   valid   min clr")
    print("  " + "-" * 60)

    rows = []
    for deg in range(0, 360, 30):
        th = np.radians(deg)
        s = v * dt * np.arange(N + 1)                 # arc length per node
        P = np.stack([x0 + s*np.cos(th), y0 + s*np.sin(th)], axis=1)
        d = np.linalg.norm(P - goal[None, :], axis=1)
        J_goal = w_goal * d[:-1].sum()
        J_term = w_term_pos * d[-1]                   # arrives stopped -> no vel term
        clr = val.clearance(P)
        J_obs = w_obs * np.square(np.maximum(0.0, d_infl - clr)).sum()
        ok = bool((clr >= r_safe).all())
        tot = J_goal + J_term + J_obs
        rows.append((deg, tot, ok))
        print("  %4d    %8.1f %8.1f %7.1f  %8.1f    %s  %6.2f"
              % (deg, J_goal, J_term, J_obs, tot,
                 "yes" if ok else "NO ", clr.min()))

    okrows = [r for r in rows if r[2]]
    if okrows:
        best = min(okrows, key=lambda r: r[1])
        print("")
        print("  cheapest VALID direction: %d deg  (J=%.1f)" % (best[0], best[1]))
        fwd = [r for r in okrows if r[0] in (0, 30, 330)]
        back = [r for r in okrows if r[0] in (150, 180, 210)]
        if fwd and back:
            bf = min(fwd, key=lambda r: r[1]); bb = min(back, key=lambda r: r[1])
            print("  best forward  %3d deg J=%.1f" % (bf[0], bf[1]))
            print("  best backward %3d deg J=%.1f" % (bb[0], bb[1]))
            print("  forward is %s by %.1f"
                  % ("CHEAPER" if bf[1] < bb[1] else "MORE EXPENSIVE",
                     abs(bf[1]-bb[1])))

    # how much of the total is the obstacle term at the best forward heading?
    print("")
    print("  NOTE: J_obs is zero only where clearance >= %.2f m. Nothing near"
          % d_infl)
    print("  the gap has that much room, so every useful direction pays it.")


if __name__ == "__main__":
    main()
