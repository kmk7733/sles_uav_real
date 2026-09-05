#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What is actually in the map around the vehicle?

The reach toward the goal stops at the same distance whether or not unknown is
treated as unsafe, so the thing blocking the plan is a MEASURED obstacle. This
prints where those cells are, so a real wall can be told apart from the floor
being painted as one -- the failure the bring-up checklist warns about.
"""

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from planar_map import PlanarOccupancy
from planar_safety import safe_radius


def main():
    rospy.init_node("diag_map", anonymous=True, disable_signals=True)
    goal = np.array([3.0, 0.0])
    r_safe = safe_radius(0.31, 0.18, 0.05, 0.05)

    msg = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(msg, occ_thresh=50,
                                                  unknown_unsafe=True)
    try:
        pose = rospy.wait_for_message("/robot/pose_world", PoseStamped,
                                      timeout=5.0)
        x0, y0 = pose.pose.position.x, pose.pose.position.y
    except Exception:
        x0, y0 = 0.0, 0.0

    print("\nvehicle at (%.2f, %.2f)   r_safe %.2f m" % (x0, y0, r_safe))
    print("grid %s" % occ.describe())

    # Clearance straight toward the goal.
    d = goal - np.array([x0, y0])
    d = d / np.linalg.norm(d)
    print("\nclearance along the ray toward the goal:")
    print("   s[m]  clear[m]  verdict")
    for s in np.arange(0.0, 2.6, 0.2):
        wx, wy = x0 + d[0] * s, y0 + d[1] * s
        c = float(occ.clearance(np.array([wx]), np.array([wy]))[0])
        print("  %5.2f   %6.2f    %s" % (s, c, "ok" if c >= r_safe else "BLOCKED"))

    # Nearest measured obstacle cells.
    ys, xs = np.nonzero(occ.occupied)
    if ys.size:
        wx = occ.origin[0] + (xs + 0.5) * occ.res
        wy = occ.origin[1] + (ys + 0.5) * occ.res
        dist = np.hypot(wx - x0, wy - y0)
        o = np.argsort(dist)[:10]
        print("\n10 nearest OCCUPIED cells:")
        for i in o:
            print("   %.2f m  at (%6.2f, %6.2f)" % (dist[i], wx[i], wy[i]))
        print("\noccupied-cell spread: x [%.2f, %.2f]  y [%.2f, %.2f]"
              % (wx.min(), wx.max(), wy.min(), wy.max()))

    # ASCII view, 0.2 m per character, centred on the vehicle.
    print("\nmap around the vehicle  ('#' occupied, '.' free, '?' unknown,"
          " 'V' vehicle, 'G' goal dir)   1 char = 0.2 m")
    step = 0.2
    for j in range(12, -13, -1):
        row = ""
        for i in range(-14, 26):
            wx, wy = x0 + i * step, y0 + j * step
            ix = int(np.floor((wx - occ.origin[0]) / occ.res))
            iy = int(np.floor((wy - occ.origin[1]) / occ.res))
            if i == 0 and j == 0:
                row += "V"
            elif not (0 <= ix < occ.W and 0 <= iy < occ.H):
                row += " "
            elif occ.occupied[iy, ix]:
                row += "#"
            elif occ.unknown[iy, ix]:
                row += "?"
            else:
                row += "."
        print("  " + row)


if __name__ == "__main__":
    main()
