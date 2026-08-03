#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Demonstration of the planar MPPI planner against a 2D occupancy map.

No vehicle, no ROS needed for the default path.

    python3 demo_planar.py                       # slalom scenario
    python3 demo_planar.py --scenario pillar     # one of the built-ins
    python3 demo_planar.py --list                # what is available
    python3 demo_planar.py --grid captured.npz   # a REAL captured /grid_map
    python3 demo_planar.py --capture captured.npz   # capture one (needs ROS)

The synthetic scenarios use the live grid's geometry (144 x 104 at 0.05 m over
a 7.2 x 5.2 m arena), so the timings and the safety radius mean the same thing
here as they do on the Jetson. To drive it with genuine perception output,
--capture one frame off /grid_map and replay it with --grid.

What it prints, per replan: the status (which branch of the safety chain ran),
the solve time, the distance to goal and the worst clearance along the plan.
Then the flown path as ASCII, and the first few nodes of the fixed-altitude 3D
reference exactly as the low-level controller would receive them.
"""

import argparse
import sys
import time

import numpy as np

from planar_types import PlanarState, PlannerStatus
from planar_dynamics import PlanarLimits, PlanarDynamics
from planar_safety import PlanarSafetyValidator, safe_radius
from planar_mppi import PlanarMPPI, PlanarCostWeights
import planar_scenarios as sc


def capture_grid(path, topic="/grid_map", timeout=15.0):
    """Save one live OccupancyGrid to an .npz for offline replay.

    Imported lazily so nothing else in this file depends on ROS.
    """
    import rospy
    from nav_msgs.msg import OccupancyGrid
    from planar_map import PlanarOccupancy

    rospy.init_node("planar_grid_capture", anonymous=True)
    print("waiting up to %.0fs for %s ..." % (timeout, topic))
    msg = rospy.wait_for_message(topic, OccupancyGrid, timeout=timeout)
    occ = PlanarOccupancy.from_occupancy_grid_msg(msg, unknown_unsafe=True)
    sc.save_grid_npy(path, occ)
    print("saved %s\n  %s" % (path, occ.describe()))


def run(occ, start, goal, args):
    limits = PlanarLimits(v_max=args.v_max, a_max=args.a_max,
                          omega_max=1.5, alpha_max=3.0,
                          tilt_max=np.radians(args.tilt_deg), j_max=args.j_max)
    r_safe = safe_radius(args.r_quad, args.r_perc, args.r_track, args.d_clr)

    dyn = PlanarDynamics(limits, dt=args.dt)
    val = PlanarSafetyValidator(occ, r_safe)
    weights = PlanarCostWeights(w_yaw=args.w_yaw, yaw_mode=args.yaw_mode)
    planner = PlanarMPPI(dyn, val, weights=weights, horizon=args.horizon,
                         num_samples=args.samples, seed=args.seed)

    # The vehicle's own footprint is frequently unknown-and-therefore-unsafe
    # (the ZED cannot see underneath itself). Without clearing it the very first
    # node collides and nothing is ever feasible.
    occ.clear_disc(start.x, start.y, r_safe + 0.05)

    print(planner.describe())
    print("  start (%.2f, %.2f)  goal (%.2f, %.2f)  z0 = %.2f m"
          % (start.x, start.y, goal[0], goal[1], args.z0))
    print("  r_safe = %.3f = r_Q %.2f + r_perc %.2f + r_track %.2f + d_clr %.2f"
          % (r_safe, args.r_quad, args.r_perc, args.r_track, args.d_clr))
    print()

    audit = PlanarSafetyValidator(occ, r_safe)      # independent of the planner
    s = start
    a_prev = np.zeros(2)
    traj = [[s.x, s.y]]
    counts = {}
    times = []
    violations = 0
    last = None
    reached_at = None

    print("  %4s %-11s %8s %8s %9s %8s" %
          ("step", "status", "solve", "d_goal", "min_clr", "speed"))
    for k in range(args.steps):
        t0 = time.time()
        res = planner.plan(s, goal, a_prev=a_prev)
        ms = (time.time() - t0) * 1000.0
        times.append(ms)
        counts[res.status] = counts.get(res.status, 0) + 1

        if res.reference is None:
            print("  %4d %-11s %8.1f   -- no trajectory: %s"
                  % (k, res.status, ms, res.reason))
            break

        if not audit.path_safe(res.reference.p):
            violations += 1

        d_goal = float(np.linalg.norm(np.array([s.x, s.y]) - goal))
        min_clr = float(audit.clearance(res.reference.p).min())
        if k % args.every == 0 or res.status != PlannerStatus.WEIGHTED:
            print("  %4d %-11s %6.1fms %7.2fm %8.2fm %7.2f%s"
                  % (k, res.status, ms, d_goal, min_clr, s.speed,
                     "  <- " + res.reason if res.reason else ""))

        last = res
        pt = res.reference.node(1)
        s = PlanarState(pt.p[0], pt.p[1], pt.v[0], pt.v[1], pt.psi, pt.psi_dot)
        a_prev = res.reference.a[0]
        traj.append([s.x, s.y])

        if reached_at is None and planner.at_goal(s, goal):
            reached_at = k

    traj = np.array(traj)
    print()
    print(sc.ascii_map(occ, traj=traj, goal=goal, start=(start.x, start.y)))
    print("  '#' unsafe (occupied or unknown)   '.' flown path   S start   G goal")
    print()

    print("RESULT")
    print("  replans          %d" % sum(counts.values()))
    print("  status           %s"
          % ", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())))
    print("  solve time       mean %.1f ms, max %.1f ms  (10 Hz budget = 100 ms)"
          % (np.mean(times), np.max(times)))
    print("  path length      %.2f m"
          % float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()))
    print("  final distance   %.2f m" % float(np.linalg.norm(traj[-1] - goal)))
    if reached_at is not None:
        print("  reached goal     step %d (%.1f s of flight)"
              % (reached_at, reached_at * args.dt))
    else:
        print("  reached goal     NO (within %.2f m tolerance)" % planner.goal_tol)
    print("  swept violations %d   <- must be 0" % violations)

    if last is not None and last.reference is not None:
        lifted = last.reference.lift(args.z0)
        print()
        print("FIXED-ALTITUDE 3D REFERENCE (last plan, first 5 of %d nodes)"
              % lifted.n_nodes)
        print("  this is what goes to the PX4 / geometric controller")
        print("  %4s %-22s %-22s %-22s %8s %9s"
              % ("k", "p_d [m]", "v_d [m/s]", "a_d [m/s^2]", "psi [rad]",
                 "psi_dot"))
        for i in range(min(5, lifted.n_nodes)):
            n = lifted.node(i)
            print("  %4d (%6.2f,%6.2f,%6.2f) (%6.2f,%6.2f,%6.2f) "
                  "(%6.2f,%6.2f,%6.2f) %8.3f %9.3f"
                  % (i, n.p[0], n.p[1], n.p[2], n.v[0], n.v[1], n.v[2],
                     n.a[0], n.a[1], n.a[2], n.psi, n.psi_dot))
        print("  vz and az are identically zero by construction: altitude is "
              "held by the\n  low-level controller, and was never a planning "
              "variable.")

    return violations == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="slalom",
                    help="built-in map: " + ", ".join(sorted(sc.SCENARIOS)))
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--grid", help="replay a captured grid (.npz)")
    ap.add_argument("--capture", help="capture one /grid_map to this .npz (needs ROS)")
    ap.add_argument("--topic", default="/grid_map")

    ap.add_argument("--start", type=float, nargs=2, default=[-2.0, 0.0])
    ap.add_argument("--goal", type=float, nargs=2, default=[2.5, 0.0])
    ap.add_argument("--z0", type=float, default=1.0, help="hold altitude [m]")
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--every", type=int, default=20, help="print every Nth step")

    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--v-max", type=float, default=1.5)
    ap.add_argument("--a-max", type=float, default=2.5)
    ap.add_argument("--j-max", type=float, default=8.0)
    ap.add_argument("--tilt-deg", type=float, default=30.0)

    ap.add_argument("--r-quad", type=float, default=0.31)
    ap.add_argument("--r-perc", type=float, default=0.18)
    ap.add_argument("--r-track", type=float, default=0.05)
    ap.add_argument("--d-clr", type=float, default=0.05)

    ap.add_argument("--w-yaw", type=float, default=0.0)
    ap.add_argument("--yaw-mode", default="velocity",
                    choices=("velocity", "goal", "hold"))
    args = ap.parse_args(argv)

    if args.list:
        print("scenarios: " + ", ".join(sorted(sc.SCENARIOS)))
        return 0
    if args.capture:
        capture_grid(args.capture, args.topic)
        return 0

    if args.grid:
        occ = sc.load_grid_npy(args.grid)
        print("replaying captured grid %s" % args.grid)
    else:
        if args.scenario not in sc.SCENARIOS:
            print("unknown scenario %r; choose from %s"
                  % (args.scenario, ", ".join(sorted(sc.SCENARIOS))))
            return 2
        occ = sc.SCENARIOS[args.scenario]()
        print("scenario: %s" % args.scenario)
    print("  map: %s\n" % occ.describe())

    start = PlanarState(args.start[0], args.start[1], 0.0, 0.0, 0.0, 0.0)
    ok = run(occ, start, np.asarray(args.goal, dtype=np.float64), args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
