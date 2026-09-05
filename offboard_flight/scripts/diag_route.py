#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where does the open route actually go, and how tight is it?

The flood fill says the goal side is reachable at r_safe, but a straight probe
toward the goal is rejected at every r_safe down to 0.41. Both can be true: the
route exists and is CURVED. This traces it and reports the bearing MPPI would
have to commit to, plus the narrowest point along the way -- which is what
decides whether random acceleration noise can thread it.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from planar_safety import safe_radius

_NB = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
       (-1,-1,2**0.5),(-1,1,2**0.5),(1,-1,2**0.5),(1,1,2**0.5)]


def main():
    rospy.init_node("diag_route", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                         rospy.get_param("~r_perc", 0.10),
                         rospy.get_param("~r_track", 0.05),
                         rospy.get_param("~d_clr", 0.05))

    g = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    p = rospy.wait_for_message("/robot/pose_world", PoseStamped, timeout=10.0)
    H, W, res = g.info.height, g.info.width, g.info.resolution
    ox, oy = g.info.origin.position.x, g.info.origin.position.y
    v = np.asarray(g.data, dtype=np.int16).reshape(H, W)
    occ = v >= 50
    unk = v < 0
    x0, y0 = p.pose.position.x, p.pose.position.y

    clr = distance_transform_edt(~(occ | unk)) * res
    free = clr >= r_safe
    # assert the footprint, as the node does
    yy, xx = np.mgrid[0:H, 0:W]
    wx = ox + (xx + 0.5) * res
    wy = oy + (yy + 0.5) * res
    free |= np.hypot(wx - x0, wy - y0) <= r_safe + 0.05

    ys, xs = np.nonzero(free)
    idx = -np.ones((H, W), dtype=np.int64)
    idx[ys, xs] = np.arange(ys.size)
    rows, cols, vals = [], [], []
    for dy, dx, w in _NB:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        ok &= free[np.clip(y2, 0, H-1), np.clip(x2, 0, W-1)]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[y2[ok], x2[ok]])
        vals.append(np.full(int(ok.sum()), w * res))
    G = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(ys.size, ys.size)).tocsr()

    ix0 = int((x0-ox)/res); iy0 = int((y0-oy)/res)
    src = idx[iy0, ix0]
    print("\nvehicle (%.2f, %.2f)  goal (%.2f, %.2f)  r_safe %.3f"
          % (x0, y0, goal[0], goal[1], r_safe))
    if src < 0:
        print("  vehicle is not in the free set"); return
    d, pred = dijkstra(G, directed=False, indices=src, return_predecessors=True)

    # nearest reachable cell to the goal
    gx = ox + (xs + 0.5) * res
    gy = oy + (ys + 0.5) * res
    reach = np.isfinite(d)
    if not reach.any():
        print("  nothing reachable"); return
    dist_to_goal = np.hypot(gx - goal[0], gy - goal[1])
    dist_to_goal[~reach] = np.inf
    tgt = int(np.argmin(dist_to_goal))
    print("  closest reachable point to the goal: (%.2f, %.2f), %.2f m short,"
          " path length %.2f m" % (gx[tgt], gy[tgt], dist_to_goal[tgt], d[tgt]))

    # walk the path back
    path = []
    k = tgt
    while k >= 0 and k != src:
        path.append(k); k = pred[k]
    path.append(src); path.reverse()
    P = np.stack([gx[path], gy[path]], axis=1)
    Wd = clr[ys[path], xs[path]]

    print("")
    print("  route waypoints every ~0.5 m (bearing = heading to take):")
    print("    s[m]   point           clearance   bearing")
    acc = 0.0; last = P[0]; shown = 0
    for i in range(1, len(P)):
        acc += float(np.linalg.norm(P[i] - P[i-1]))
        if acc >= 0.5 * (shown + 1) or i == len(P)-1:
            hd = np.degrees(np.arctan2(P[i][1]-last[1], P[i][0]-last[0])) % 360
            print("    %5.2f  (%6.2f,%6.2f)    %5.2f      %5.1f deg"
                  % (acc, P[i][0], P[i][1], Wd[i], hd))
            last = P[i]; shown += 1

    j = int(np.argmin(Wd))
    print("")
    print("  NARROWEST point on the route: clearance %.2f m at (%.2f, %.2f),"
          % (Wd[j], P[j][0], P[j][1]))
    print("    %.2f m along the path.  Usable half-width above r_safe: %.2f m"
          % (float(np.linalg.norm(np.diff(P[:j+1], axis=0), axis=1).sum()),
             Wd[j] - r_safe))
    print("    A corridor this tight is why random acceleration noise rarely")
    print("    threads it: the sample must commit to the turn AND stay inside")
    print("    %.0f cm of lateral slack for the whole 2 s horizon."
          % (100 * (Wd[j] - r_safe)))

    # initial bearing over the first metre
    m = np.argmax(np.linalg.norm(P - P[0], axis=1) >= 1.0)
    if m:
        hd0 = np.degrees(np.arctan2(P[m][1]-P[0][1], P[m][0]-P[0][0])) % 360
        print("")
        print("  bearing over the first 1.0 m of the route: %.0f deg" % hd0)
        print("  (a straight probe at 0 deg pinches to 0.40 m -- that is why it"
              " fails)")


if __name__ == "__main__":
    main()
