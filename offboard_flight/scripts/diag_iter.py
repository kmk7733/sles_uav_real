#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the nominal build up over successive solves, or reset every tick?

One MPPI solve from U_nom = 0 cannot reach far: the samples are zero-mean noise
around "do nothing", so the cloud is centred on no motion and the best any
sample manages here is ~1.3 m of 2.0 m possible. That is not a fault -- MPPI is
supposed to bootstrap, each solve warm-starting from the last so the nominal
accumulates a sustained acceleration.

This runs successive solves from a FIXED state, exactly what the node does while
hovering, and prints what the accepted plan does each time. If the terminal
progress grows, the loop is healthy and the vehicle is simply slow to commit. If
it stays flat, the warm start is not accumulating and THAT is the bug.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from planar_types import PlanarState
from planar_dynamics import PlanarDynamics, PlanarLimits
from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarCostWeights
from frontier import FrontierMPPI


def main():
    rospy.init_node("diag_iter", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                         rospy.get_param("~r_perc", 0.10),
                         rospy.get_param("~r_track", 0.05),
                         rospy.get_param("~d_clr", 0.05))
    n_iter = int(rospy.get_param("~n_iter", 25))
    advance = bool(rospy.get_param("~advance", False))

    g = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(g, occ_thresh=50,
                                                  unknown_unsafe=True)
    p = rospy.wait_for_message("/robot/pose_world", PoseStamped, timeout=10.0)
    x0, y0 = p.pose.position.x, p.pose.position.y
    occ.clear_disc(x0, y0, r_safe + 0.05)

    limits = PlanarLimits(v_max=1.0, a_max=2.5, omega_max=1.5, alpha_max=3.0,
                          tilt_max=0.5236, j_max=8.0)
    Ra = float(rospy.get_param("~r_dnu_a", 0.0))
    w = PlanarCostWeights(w_goal=1.0, w_term_pos=10.0, w_term_vel=2.0,
                          w_obs=20.0, d_influence=None, w_yaw=0.0,
                          R_dnu=(Ra, Ra, 0.2 * Ra))
    dyn = PlanarDynamics(limits, dt=0.1)
    val = PlanarSafetyValidator(occ, r_safe)
    pl = FrontierMPPI(dyn, val, weights=w, use_geodesic=False, w_frontier=0.0,
                      horizon=20, num_samples=int(rospy.get_param('~num_samples',192)), temperature=float(rospy.get_param('~temperature',1.0)), seed=0,
                      goal_tol=0.25,
                      sigma=(float(rospy.get_param('~sigma',1.2)),
                             float(rospy.get_param('~sigma',1.2)), 1.5))

    u = goal - np.array([x0, y0]); u /= np.linalg.norm(u)
    st = PlanarState(x=x0, y=y0, psi=0.0, vx=0.0, vy=0.0, omega=0.0)
    a_prev = None

    print("\nvehicle (%.2f, %.2f)  goal (%.2f, %.2f)  r_safe %.3f"
          % (x0, y0, goal[0], goal[1], r_safe))
    print("state is %s between solves\n"
          % ("ADVANCED along the plan" if advance else "HELD FIXED (hover)"))
    print("  temperature = %.3f   R_dnu_a = %.3g   sigma = %s   K = %d"
          % (pl.temperature, Ra, pl.sigma[:2], pl.K))
    print("  iter  status     valid   position        travelled  clr   |v|"
          "   plan_end")
    print("  " + "-" * 74)
    p_start = np.array([x0, y0])

    for i in range(n_iter):
        res = pl.plan(st, goal, a_prev=a_prev, n_viz=0)
        if not res.ok:
            print("  %4d  %-11s %5d   ---   (%s)"
                  % (i, res.status, res.n_valid, res.reason))
            continue
        ref = res.reference
        prog = float((ref.p[-1] - np.array([st.x, st.y])) @ u)
        tv = float(np.linalg.norm(ref.v[-1]))
        un = float(np.linalg.norm(pl.U_nom[0, :2]))
        cur = np.array([st.x, st.y])
        clr_now = float(val.clearance(cur[None, :])[0])
        spd = float(np.hypot(st.vx, st.vy))
        print("  %4d  %-10s %5d  (%6.2f,%6.2f)  %6.2f m  %5.2f %5.2f  %+7.3f m"
              % (i, str(res.status), res.n_valid, st.x, st.y,
                 float((cur - p_start) @ u), clr_now, spd, prog))
        a_prev = ref.a[0].copy()
        if advance:
            # follow the plan one control step, as the vehicle would
            st = PlanarState(x=ref.p[1][0], y=ref.p[1][1], psi=ref.psi[1],
                             vx=ref.v[1][0], vy=ref.v[1][1],
                             omega=ref.psi_dot[1])

    print("")
    print("  SUMMARY: final progress %+.3f m   final |U_nom[0]| %.3f"
          % (prog, un))
    print("  READ: |U_nom[0]| is the acceleration the nominal commits to.")
    print("  If it stays near zero while progress stays near zero, successive")
    print("  solves are not accumulating and the planner can never build speed.")


if __name__ == "__main__":
    main()
