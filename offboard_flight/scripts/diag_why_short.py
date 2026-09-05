#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why is the nominal plan short and barely moving?

Prints, for one solve on the live map:
  n_valid   how many of the K sampled sequences survived the validator. If this
            is a handful, the plan is not "choosing" to creep -- almost every
            candidate was rejected and the weighted mean is over the few that
            were left.
  arc/disp  how far the plan actually travels vs how far it displaces.
  reach     how far observed-free space extends from the start toward the goal,
            which is the ceiling on how far any ACCEPTED plan can go, because
            the validator refuses to enter unknown.
"""

import numpy as np
import rospy
from scipy.ndimage import distance_transform_edt

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

from planar_types import PlanarState
from planar_dynamics import PlanarDynamics, PlanarLimits
from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarCostWeights
from frontier import FrontierMPPI


def main():
    rospy.init_node("diag_why_short", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(0.31, 0.18, 0.05, 0.05)

    msg = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    uu = bool(rospy.get_param("~unknown_unsafe", True))
    occ = PlanarOccupancy.from_occupancy_grid_msg(msg, occ_thresh=50,
                                                  unknown_unsafe=uu)
    print("\n--- unknown_unsafe = %s ---" % uu)
    try:
        pose = rospy.wait_for_message("/robot/pose_world", PoseStamped,
                                      timeout=5.0)
        x0, y0 = pose.pose.position.x, pose.pose.position.y
    except Exception:
        x0, y0 = 0.0, 0.0

    occ.clear_disc(x0, y0, r_safe + 0.05)

    limits = PlanarLimits(v_max=1.0, a_max=2.5, omega_max=1.5, alpha_max=3.0,
                          tilt_max=0.5236, j_max=8.0)
    w = PlanarCostWeights(w_goal=1.0, w_term_pos=10.0, w_term_vel=2.0,
                          w_obs=20.0, d_influence=None, w_yaw=0.0)
    dyn = PlanarDynamics(limits, dt=0.1)
    val = PlanarSafetyValidator(occ, r_safe)
    pl = FrontierMPPI(dyn, val, weights=w, use_geodesic=True, w_frontier=0.0,
                      horizon=20, num_samples=192, temperature=1.0, seed=0,
                      goal_tol=0.25)

    st = PlanarState(x=x0, y=y0, psi=0.0, vx=0.0, vy=0.0, omega=0.0)
    res = pl.plan(st, goal, a_prev=None, n_viz=0)

    p = res.reference.p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    disp = float(np.linalg.norm(p[-1] - p[0]))

    print("")
    print("start (%.2f, %.2f)   goal (%.2f, %.2f)   dist %.2f m"
          % (x0, y0, goal[0], goal[1], np.linalg.norm(goal - p[0])))
    print("status   %s   reason=%s" % (res.status, res.reason))
    print("n_valid  %d / %d sampled sequences survived the validator"
          % (res.n_valid, res.n_samples))
    print("plan     arc %.2f m   displacement %.2f m   over %.1f s"
          % (float(seg.sum()), disp, 20 * 0.1))
    print("         mean speed %.2f m/s   (v_max %.2f)"
          % (float(seg.sum()) / 2.0, limits.v_max))
    print("         terminal |v| %.3f m/s"
          % float(np.linalg.norm(res.reference.v[-1])))

    # How far can ANY accepted plan go? Observed-free means clear of measured
    # obstacles by r_safe AND not unknown, since the validator gates on unsafe.
    unsafe = occ.unsafe
    free = distance_transform_edt(~unsafe) * occ.res >= r_safe
    print("")
    print("map      occupied %d   unknown %d   of %d cells"
          % (int(occ.occupied.sum()), int(occ.unknown.sum()), occ.H * occ.W))
    print("         cells the validator would accept: %d (%.1f%%)"
          % (int(free.sum()), 100.0 * free.sum() / free.size))

    # Ray from start toward goal: where does acceptable space stop?
    d = goal - np.array([x0, y0])
    d = d / max(np.linalg.norm(d), 1e-9)
    reach = 0.0
    for s in np.arange(0.0, 7.0, occ.res):
        wx, wy = x0 + d[0] * s, y0 + d[1] * s
        ix = int(np.floor((wx - occ.origin[0]) / occ.res))
        iy = int(np.floor((wy - occ.origin[1]) / occ.res))
        if not (0 <= ix < occ.W and 0 <= iy < occ.H) or not free[iy, ix]:
            break
        reach = s
    print("         straight toward the goal, acceptable space ends at %.2f m"
          % reach)
    print("")
    print("=> the plan cannot exceed that reach: the validator rejects any")
    print("   trajectory entering unknown, so a 2 s / %.1f m capable horizon"
          % (limits.v_max * 2.0))
    print("   is clipped to whatever the camera has actually observed.")


if __name__ == "__main__":
    main()
