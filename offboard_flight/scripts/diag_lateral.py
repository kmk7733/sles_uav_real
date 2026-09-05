#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the vehicle boxed in, and if so by what -- a real obstacle or the map's
own synthetic border?

Answers two specific objections:
  1. "it is ~90 cm clear, that is plenty"  -> what is the nearest occupied cell,
     and is it a REAL detection or the border wall depth_to_grid draws at the
     grid limit (border_wall_enable=true, thickness 0.1 m)?
  2. "the plane has full freedom, it should just go sideways" -> the free set is
     eroded by r_safe and then flood-filled from the vehicle. If the reachable
     component is a small pocket, no lateral trajectory exists either, and the
     planner is not choosing to creep -- there is nowhere to go.
"""

import numpy as np
import rospy
from scipy.ndimage import distance_transform_edt, label

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from planar_map import PlanarOccupancy
from planar_safety import safe_radius


def main():
    rospy.init_node("diag_lateral", anonymous=True, disable_signals=True)
    r_safe = safe_radius(0.31, 0.18, 0.05, 0.05)
    goal = np.array([3.0, 0.0])

    msg = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(msg, occ_thresh=50,
                                                  unknown_unsafe=True)
    try:
        pose = rospy.wait_for_message("/robot/pose_world", PoseStamped,
                                      timeout=5.0)
        x0, y0 = pose.pose.position.x, pose.pose.position.y
    except Exception:
        x0, y0 = 0.0, 0.0
    occ.clear_disc(x0, y0, r_safe + 0.05)

    res = occ.res
    xmin, ymin, xmax, ymax = occ.bounds
    print("\nvehicle (%.2f, %.2f)   r_safe %.2f   grid x[%.2f,%.2f] y[%.2f,%.2f]"
          % (x0, y0, r_safe, xmin, xmax, ymin, ymax))

    # ---- 1. what is the nearest occupied cell, and is it the border wall?
    ys, xs = np.nonzero(occ.occupied)
    wx = occ.origin[0] + (xs + 0.5) * res
    wy = occ.origin[1] + (ys + 0.5) * res
    dist = np.hypot(wx - x0, wy - y0)

    # depth_to_grid draws a synthetic wall of `thickness` at the grid limit.
    thick = 0.10
    is_border = ((wx <= xmin + thick) | (wx >= xmax - thick) |
                 (wy <= ymin + thick) | (wy >= ymax - thick))

    k = int(np.argmin(dist))
    print("\nnearest OCCUPIED cell overall : %.2f m at (%.2f, %.2f)  %s"
          % (dist[k], wx[k], wy[k],
             "SYNTHETIC BORDER WALL" if is_border[k] else "real detection"))
    if (~is_border).any():
        j = int(np.argmin(np.where(is_border, np.inf, dist)))
        print("nearest REAL (non-border) cell: %.2f m at (%.2f, %.2f)"
              % (dist[j], wx[j], wy[j]))
    print("occupied cells: %d total, %d of them are border wall (%.0f%%)"
          % (is_border.size, int(is_border.sum()),
             100.0 * is_border.sum() / max(is_border.size, 1)))

    # ---- 2. where can the vehicle actually GO?
    def reachable(occupied, unknown, label_txt, unknown_blocks=True):
        blocked = occupied | unknown if unknown_blocks else occupied
        free = distance_transform_edt(~blocked) * res >= r_safe
        lab, n = label(free)
        ix = int(np.floor((x0 - occ.origin[0]) / res))
        iy = int(np.floor((y0 - occ.origin[1]) / res))
        cid = lab[iy, ix]
        if cid == 0:
            print("\n%-28s vehicle cell is NOT in the free set at all"
                  % label_txt)
            return
        comp = lab == cid
        cy, cx = np.nonzero(comp)
        ccx = occ.origin[0] + (cx + 0.5) * res
        ccy = occ.origin[1] + (cy + 0.5) * res
        d = np.hypot(ccx - x0, ccy - y0)
        # how far toward the goal can the reachable component take us?
        u = (goal - np.array([x0, y0]))
        u = u / np.linalg.norm(u)
        proj = (ccx - x0) * u[0] + (ccy - y0) * u[1]
        print("\n%-28s reachable cells %d (%.1f%% of grid), of %d components"
              % (label_txt, int(comp.sum()), 100.0 * comp.sum() / comp.size, n))
        print("%-28s max radius from vehicle %.2f m"
              % ("", float(d.max())))
        print("%-28s extent x[%.2f, %.2f]  y[%.2f, %.2f]"
              % ("", ccx.min(), ccx.max(), ccy.min(), ccy.max()))
        print("%-28s furthest progress toward goal %.2f m  (lateral spread %.2f m)"
              % ("", float(proj.max()), float(ccy.max() - ccy.min())))

    reachable(occ.occupied, occ.unknown, "as flown (unknown blocks):")
    reachable(occ.occupied, occ.unknown, "if unknown were free:",
              unknown_blocks=False)

    # same, but with the synthetic border wall removed
    occ_nb = occ.occupied.copy()
    bys, bxs = np.nonzero(occ_nb)
    bwx = occ.origin[0] + (bxs + 0.5) * res
    bwy = occ.origin[1] + (bys + 0.5) * res
    bmask = ((bwx <= xmin + thick) | (bwx >= xmax - thick) |
             (bwy <= ymin + thick) | (bwy >= ymax - thick))
    occ_nb[bys[bmask], bxs[bmask]] = False
    reachable(occ_nb, occ.unknown, "no border wall:")

    # ---- 3. how much clearance would be needed to open things up?
    print("\nclearance the vehicle actually has: %.2f m"
          % float(occ.clearance(np.array([x0]), np.array([y0]))[0]))
    for r in (0.59, 0.50, 0.45, 0.40, 0.35, 0.30):
        free = distance_transform_edt(~(occ.occupied | occ.unknown)) * res >= r
        lab, _ = label(free)
        ix = int(np.floor((x0 - occ.origin[0]) / res))
        iy = int(np.floor((y0 - occ.origin[1]) / res))
        cid = lab[iy, ix]
        if cid == 0:
            print("  r_safe %.2f -> vehicle not in free set" % r)
            continue
        comp = lab == cid
        cy, cx = np.nonzero(comp)
        ccx = occ.origin[0] + (cx + 0.5) * res
        u = (goal - np.array([x0, y0])); u = u / np.linalg.norm(u)
        ccy = occ.origin[1] + (cy + 0.5) * res
        proj = (ccx - x0) * u[0] + (ccy - y0) * u[1]
        print("  r_safe %.2f -> reachable %5d cells, goal progress %.2f m"
              % (r, int(comp.sum()), float(proj.max())))


if __name__ == "__main__":
    main()
