#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which stage kills the samples, and does any survivor go forward?

plan() rejects in three stages:
    dyn.states_ok(X)              velocity / heading-rate / finite
    dyn.inputs_ok(U, a_prev)      accel disc / alpha box / jerk chain
    validator.nodes_safe(P)       clearance >= r_safe

Only the intersection survives, and X_viz shows a uniform subsample of THAT --
not the high-weight ones. This replicates the pipeline exactly and reports the
kill count per stage, plus where the survivors actually end up, so "it only
plans locally" can be attributed rather than guessed at.
"""

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from planar_types import PlanarState, S_POS, S_VEL
from planar_dynamics import PlanarDynamics, PlanarLimits
from planar_map import PlanarOccupancy
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarCostWeights
from frontier import FrontierMPPI


def main():
    rospy.init_node("diag_reject", anonymous=True, disable_signals=True)
    goal = np.array([float(rospy.get_param("~goal_x", 3.0)),
                     float(rospy.get_param("~goal_y", 0.0))])
    r_safe = safe_radius(rospy.get_param("~r_quad", 0.31),
                         rospy.get_param("~r_perc", 0.10),
                         rospy.get_param("~r_track", 0.05),
                         rospy.get_param("~d_clr", 0.05))
    K = int(rospy.get_param("~num_samples", 192))

    g = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=20.0)
    occ = PlanarOccupancy.from_occupancy_grid_msg(g, occ_thresh=50,
                                                  unknown_unsafe=True)
    p = rospy.wait_for_message("/robot/pose_world", PoseStamped, timeout=10.0)
    x0, y0 = p.pose.position.x, p.pose.position.y
    # Override to probe a specific place -- the cost landscape is position
    # dependent and measuring it at the START says nothing about why the
    # vehicle halts 0.4 m later.
    x0 = float(rospy.get_param("~x0", x0))
    y0 = float(rospy.get_param("~y0", y0))
    occ.clear_disc(x0, y0, r_safe + 0.05)

    limits = PlanarLimits(v_max=1.0, a_max=2.5, omega_max=1.5, alpha_max=3.0,
                          tilt_max=0.5236, j_max=8.0)
    w = PlanarCostWeights(w_goal=1.0, w_term_pos=10.0, w_term_vel=2.0,
                          w_obs=20.0, d_influence=None, w_yaw=0.0)
    dyn = PlanarDynamics(limits, dt=0.1)
    val = PlanarSafetyValidator(occ, r_safe)
    pl = FrontierMPPI(dyn, val, weights=w, use_geodesic=False, w_frontier=0.0,
                      horizon=20, num_samples=K, temperature=1.0, seed=0,
                      goal_tol=0.25)

    st = PlanarState(x=x0, y=y0, psi=0.0, vx=0.0, vy=0.0, omega=0.0)
    xi0 = st.to_array()

    # --- replicate plan()'s sampling exactly -----------------------------
    pl.warm_start(1)
    U_nom = pl.U_nom
    noise = pl.rng.randn(pl.K, pl.N, 3) * pl.sigma[None, None, :]
    U = U_nom[None, :, :] + noise
    U[0] = U_nom
    U = dyn.clip_inputs(U, a_prev=None)
    X = dyn.rollout(xi0, U)

    s_ok = dyn.states_ok(X)
    i_ok = dyn.inputs_ok(U, a_prev=None)
    c_ok = val.nodes_safe(X[..., S_POS])
    valid = s_ok & i_ok & c_ok

    sp = np.linalg.norm(X[..., S_VEL], axis=-1).max(axis=1)
    om = np.abs(X[..., 5]).max(axis=1)

    print("\nvehicle (%.2f, %.2f)  goal (%.2f, %.2f)  r_safe %.3f  K=%d"
          % (x0, y0, goal[0], goal[1], r_safe, K))
    print("|U_nom| first accel = %.3f   (warm start carried over)"
          % float(np.linalg.norm(U_nom[0, :2])))
    print("")
    print("  stage                      pass    fail")
    print("  states_ok (v/omega)      %6d  %6d" % (s_ok.sum(), (~s_ok).sum()))
    print("     of which |v|>v_max    %6s  %6d" % ("", int((sp > 1.0).sum())))
    print("     of which |om|>om_max  %6s  %6d" % ("", int((om > 1.5).sum())))
    print("  inputs_ok (a/jerk)       %6d  %6d" % (i_ok.sum(), (~i_ok).sum()))
    print("  nodes_safe (clearance)   %6d  %6d" % (c_ok.sum(), (~c_ok).sum()))
    print("  ---------------------------------------")
    print("  ALL THREE (valid)        %6d  %6d" % (valid.sum(), (~valid).sum()))

    if valid.sum() == 0:
        print("\n  no survivors"); return

    # where do survivors end up?
    term = X[valid][:, -1, 0:2]
    dvec = term - np.array([x0, y0])
    disp = np.linalg.norm(dvec, axis=1)
    brg = np.degrees(np.arctan2(dvec[:, 1], dvec[:, 0])) % 360
    u = goal - np.array([x0, y0]); u /= np.linalg.norm(u)
    prog = dvec @ u

    print("")
    print("  SURVIVORS: terminal displacement")
    print("    max %.2f m   mean %.2f m   (2 s at v_max would be 2.0 m)"
          % (disp.max(), disp.mean()))
    print("    progress toward goal: max %+.2f m   mean %+.2f m"
          % (prog.max(), prog.mean()))
    print("    heading toward goal (|bearing|<60 deg): %d of %d"
          % (int(((brg < 60) | (brg > 300)).sum()), valid.sum()))
    print("    heading away      (120..240 deg)      : %d of %d"
          % (int(((brg > 120) & (brg < 240)).sum()), valid.sum()))

    # ---- WHICH COST TERM DRIVES THE WEIGHTING? -------------------------
    # MPPI weights on the SPREAD across samples, not on absolute cost. The term
    # with the largest std is the one actually choosing the direction.
    P = X[..., 0:2]
    d = np.linalg.norm(P - goal[None, None, :], axis=-1)
    J_goal = 1.0 * d[:, :-1].sum(axis=1)
    J_term = 10.0 * d[:, -1]
    J_vel  = 2.0 * np.square(X[:, -1, 2:4]).sum(axis=1)
    cl = val.clearance(P)
    d_infl = r_safe + 0.35
    J_obs  = 20.0 * np.square(np.maximum(0.0, d_infl - cl)).sum(axis=1)
    print("")
    print("  COST TERM SPREAD over the %d VALID samples (d_influence %.2f):"
          % (int(valid.sum()), d_infl))
    print("    term        mean      std     range")
    for lab, T in (("goal(run) ", J_goal), ("goal(term)", J_term),
                   ("term vel  ", J_vel),  ("OBSTACLE  ", J_obs)):
        t = T[valid]
        print("    %s %8.1f %8.1f  %8.1f" % (lab, t.mean(), t.std(), t.max()-t.min()))
    # THE TERM OMITTED EARLIER: input-change cost. R_dnu penalises |nu_k -
    # nu_{k-1}|^2, and the sampler's own white noise IS a large input change.
    R = np.array([1.0, 1.0, 0.2])
    nu_prev = np.zeros((U.shape[0], 1, 3))
    chain = np.concatenate([nu_prev, U], axis=1)
    dUc = np.diff(chain, axis=1)
    J_slew = (R[None, None, :] * np.square(dUc)).sum(axis=(1, 2))
    t = J_slew[valid]
    print("    %s %8.1f %8.1f  %8.1f" % ("SLEW R_dnu", t.mean(), t.std(), t.max()-t.min()))
    print("    %s %8.1f %8s  %8s" % ("  sample 0 (= U_nom, unperturbed)",
                                     J_slew[0], "", ""))

    tot_goal = (J_goal+J_term)[valid]
    print("")
    print("    goal terms combined  std %.1f" % tot_goal.std())
    print("    obstacle term        std %.1f" % J_obs[valid].std())
    r = J_obs[valid].std()/max(tot_goal.std(),1e-9)
    print("    slew term            std %.1f" % J_slew[valid].std())
    print("    obstacle/goal spread ratio = %.2f" % r)
    print("    SLEW/goal spread ratio     = %.2f"
          % (J_slew[valid].std()/max(tot_goal.std(),1e-9)))
    Jtot = (J_goal+J_term+J_vel+J_obs+J_slew)
    k = int(np.argmin(np.where(valid, Jtot, np.inf)))
    print("")
    print("    cheapest VALID sample is index %d %s" % (k, "(= the UNPERTURBED NOMINAL)" if k==0 else ""))
    print("      its slew %.1f vs mean %.1f  |  its goal %.1f vs mean %.1f"
          % (J_slew[k], J_slew[valid].mean(), (J_goal+J_term)[k], tot_goal.mean()))
    if r > 1.0:
        print("    => the OBSTACLE term dominates the weighting: MPPI is")
        print("       choosing clearance, not progress. Staying put and going")
        print("       backward both score better than threading the gap.")

    # same for ALL samples, to separate "not sampled" from "sampled and killed"
    termA = X[:, -1, 0:2] - np.array([x0, y0])
    progA = termA @ u
    print("")
    print("  ALL SAMPLES (before rejection):")
    print("    progress toward goal: max %+.2f m   mean %+.2f m"
          % (progA.max(), progA.mean()))
    fwd = progA > 0.3
    print("    samples making >0.3 m forward progress: %d   of those valid: %d"
          % (int(fwd.sum()), int((fwd & valid).sum())))


if __name__ == "__main__":
    main()
