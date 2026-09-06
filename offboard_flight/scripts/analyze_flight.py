#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read a flight bag and say what happened.

    python3 analyze_flight.py ~/bags/flight_20260906_141500_light.bag
    python3 analyze_flight.py FLIGHT.bag --geometry obstacles.yaml
    python3 analyze_flight.py FLIGHT.bag --template > obstacles.yaml

WHAT THIS IS FOR. r_safe = 0.51 m is the number the whole safety argument
rests on and it has never been checked against hardware -- only against the
occupancy grid, which is the planner's BELIEF. Vicon carries the obstacles as
subjects (pillar1, pillar2, ..., wall1, ...), so the true distance is
recoverable, and "did the vehicle ever come closer to a real pillar than the
radius it was planning with" becomes a measurement rather than an assumption.

CENTRE DISTANCE IS NOT CLEARANCE. Vicon gives an obstacle's pose, not its
extent. Without geometry this reports centre-to-centre distance and says so;
with --geometry it subtracts the surface and reports true clearance. It will
not guess a radius, because a made-up radius turns a safety measurement into
a number that merely looks like one.
"""

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import rosbag

TEMPLATE = """\
# Obstacle geometry, in the Vicon world frame. Names are Vicon SUBJECT names,
# so /vicon/pillar1/pillar1 is `pillar1`. Sizes are the PHYSICAL extent; the
# analysis subtracts them from the centre distance to get surface clearance.
#
#   disc:  radius [m]                      -- pillars
#   box:   sx, sy [m], rotated by the subject's own Vicon yaw   -- walls
#
# MEASURE THESE. A guessed radius makes the clearance number wrong in the
# direction that matters.
pillar1: {shape: disc, radius: 0.15}
pillar2: {shape: disc, radius: 0.15}
wall1:   {shape: box,  sx: 2.00, sy: 0.10}
"""

# Vicon runs at 100+ Hz, so a gap shorter than this is a timestamp artefact
# rather than a measurement.
MIN_DT = 0.005

NS = "rogx2"
T_STATE = "/%s/mission_node/state" % NS
T_STATUS = "/%s/planar_planner_node/status" % NS
T_CONFIG = "/%s/planar_planner_node/config" % NS
T_ARRIVE = "/goal_arrive_tf"
T_WORLD = "/robot/pose_world"


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def surface_distance(p, obs_xy, obs_yaw, geom):
    """Distance from the point p to the obstacle's SURFACE, in metres.

    Negative means inside it. Returns the centre distance when no geometry is
    known, and the caller is responsible for saying which it printed.
    """
    d = float(np.hypot(p[0] - obs_xy[0], p[1] - obs_xy[1]))
    if geom is None:
        return d, False
    if geom["shape"] == "disc":
        return d - float(geom["radius"]), True
    # box: into the obstacle's own frame, then the standard rounded-box
    # distance. Exact outside the box, which is where the vehicle is.
    c, s = math.cos(-obs_yaw), math.sin(-obs_yaw)
    dx, dy = p[0] - obs_xy[0], p[1] - obs_xy[1]
    lx, ly = c * dx - s * dy, s * dx + c * dy
    qx = abs(lx) - float(geom["sx"]) / 2.0
    qy = abs(ly) - float(geom["sy"]) / 2.0
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    return outside + min(max(qx, qy), 0.0), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", nargs="?")
    ap.add_argument("--geometry", help="YAML: obstacle name -> shape and size")
    ap.add_argument("--template", action="store_true",
                    help="print a geometry file to fill in, and exit")
    ap.add_argument("--vehicle", default="ROGX2",
                    help="Vicon subject name of the aircraft")
    args = ap.parse_args()

    if args.template:
        sys.stdout.write(TEMPLATE)
        return 0
    if not args.bag:
        ap.error("give a bag (or --template)")

    geom = {}
    if args.geometry:
        import yaml
        with open(args.geometry) as f:
            geom = yaml.safe_load(f) or {}

    bag = rosbag.Bag(args.bag)
    info = bag.get_type_and_topic_info()[1]
    vicon = sorted(t for t in info if t.startswith("/vicon/"))

    veh_topic = "/vicon/%s/%s" % (args.vehicle, args.vehicle)
    obs_topics = [t for t in vicon if t != veh_topic]

    # ------------------------------------------------------------- read once
    traj = []                       # (t, x, y, z) from Vicon, ground truth
    obs = defaultdict(list)         # name -> (t, x, y, yaw)
    states, status, arrive = [], [], []
    cfg = None
    want = set([T_STATE, T_STATUS, T_CONFIG, T_ARRIVE] + vicon)
    for topic, msg, t in bag.read_messages(topics=list(want)):
        # THE MESSAGE'S OWN STAMP, not the bag receipt time. Receipt times
        # bunch: several Vicon frames can land in the same millisecond while
        # the recorder catches up, and dividing a real displacement by that
        # gap is what produced a 152 m/s "max speed" on the first bag this
        # was run against. Falls back to receipt time for anything unstamped.
        h = getattr(msg, "header", None)
        ts = h.stamp.to_sec() if h is not None and h.stamp.to_sec() > 0 \
            else t.to_sec()
        if topic == T_CONFIG and cfg is None:
            cfg = json.loads(msg.data)
        elif topic == T_STATE:
            states.append((ts, msg.data))
        elif topic == T_STATUS:
            status.append((ts, msg.data))
        elif topic == T_ARRIVE:
            arrive.append((ts, msg.data))
        elif topic == veh_topic:
            tr = msg.transform.translation
            traj.append((ts, tr.x, tr.y, tr.z))
        elif topic in obs_topics:
            tr, ro = msg.transform.translation, msg.transform.rotation
            obs[topic.split("/")[2]].append((ts, tr.x, tr.y, yaw_of(ro)))
    bag.close()

    print("=" * 72)
    print("BAG  %s" % args.bag)
    print("=" * 72)

    # ------------------------------------------------------------ what flew
    if cfg is None:
        print("\nNO ~config IN THIS BAG -- which planner flew it is not "
              "recorded. Bags from before that topic existed cannot be "
              "attributed; do not compare them against ones that can.")
    else:
        print("\nWHAT FLEW")
        print("  producer   %s (%s, %s)" % (cfg["producer"],
                                            cfg["planner_class"],
                                            cfg["dynamics"]))
        L, M, W = cfg["limits"], cfg["mppi"], cfg["weights"]
        print("  limits     v %.2f  a %.2f  omega %.3f  j %.2f"
              % (L["v_max"], L["a_max"], L["omega_max"], L["j_max"]))
        print("  mppi       K %d  N %d  dt %.2f  sigma (%.4f, %.4f, %.4f)"
              % (M["num_samples"], M["horizon"], M["dt"], *cfg["sigma"]))
        print("  cost       w_obs %.1f  d_infl %.2f  w_front %.1f  "
              "geodesic %s  R_dnu %s"
              % (W["w_obs"], W["d_influence"], M["w_frontier"],
                 M["use_geodesic"], W["R_dnu"]))
        print("  r_safe     %.3f  (goal %.2f, %.2f)"
              % (cfg["safety"]["r_safe"], *cfg["goal"]))
        print("  git        %s%s" % (cfg["git"],
                                     "  PLANNER TREE WAS DIRTY"
                                     if cfg.get("planner_dirty") else ""))

    # -------------------------------------------------------------- timeline
    print("\nTIMELINE")
    if not states:
        print("  no %s in this bag" % T_STATE)
    else:
        t0 = states[0][0]
        for i, (ts, name) in enumerate(states):
            end = states[i + 1][0] if i + 1 < len(states) else \
                (traj[-1][0] if traj else ts)
            print("  %7.1f s  %-8s  %5.1f s" % (ts - t0, name, end - ts))
    first_arrive = next((ts for ts, v in arrive if v), None)
    if first_arrive and states:
        print("  goal_arrive_tf first True at %+.1f s"
              % (first_arrive - states[0][0]))
    elif arrive:
        print("  goal_arrive_tf NEVER went True")

    # ------------------------------------------------------------ trajectory
    if traj:
        a = np.array(traj)
        t, p = a[:, 0], a[:, 1:3]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        dt = np.diff(t)
        # DISCARD short gaps, do not clamp them. Clamping dt to a floor keeps
        # the sample and reports a speed computed from a gap that was never
        # measured; dropping it loses one interval out of thousands.
        ok = dt >= MIN_DT
        v = step[ok] / dt[ok]
        dropped = int((~ok).sum())
        print("\nTRAJECTORY (Vicon, ground truth)")
        print("  duration   %.1f s     path %.2f m" % (t[-1] - t[0],
                                                       step.sum()))
        if v.size:
            print("  speed      mean %.3f  p95 %.3f  max %.3f m/s%s"
                  % (v.mean(), np.percentile(v, 95), v.max(),
                     "   [%d intervals under %.0f ms dropped]"
                     % (dropped, 1000 * MIN_DT) if dropped else ""))
        else:
            print("  speed      no usable intervals")
        print("  height     min %.2f  max %.2f m" % (a[:, 3].min(),
                                                     a[:, 3].max()))
        if cfg:
            g = np.array(cfg["goal"])
            print("  final      (%.2f, %.2f), %.2f m from the goal"
                  % (p[-1, 0], p[-1, 1], np.linalg.norm(p[-1] - g)))
        if cfg and v.size and v.max() > cfg["limits"]["v_max"] * 1.15:
            print("  NOTE speed exceeded the planned v_max %.2f by %.0f%% -- "
                  "z_vel is the margin that covers this"
                  % (cfg["limits"]["v_max"],
                     100 * (v.max() / cfg["limits"]["v_max"] - 1)))

    # ------------------------------------------------------------- clearance
    print("\nCLEARANCE TO VICON OBSTACLES")
    if not obs:
        print("  no obstacle subjects in this bag (only %s). Either none were"
              % (args.vehicle,))
        print("  in Tracker, or vicon_bridge was not running.")
    elif not traj:
        print("  no %s track, cannot measure" % veh_topic)
    else:
        a = np.array(traj)
        worst = None
        known_any = False
        for name in sorted(obs):
            o = np.array(obs[name])
            # obstacles are static; take the median pose and note any drift
            ox, oy, oyaw = np.median(o[:, 1]), np.median(o[:, 2]), \
                np.median(o[:, 3])
            drift = float(np.max(np.hypot(o[:, 1] - ox, o[:, 2] - oy)))
            g = geom.get(name)
            ds = [surface_distance((x, y), (ox, oy), oyaw, g)
                  for _, x, y, _ in a]
            d = np.array([v for v, _ in ds])
            known = ds[0][1]
            known_any |= known
            i = int(d.argmin())
            kind = "surface" if known else "CENTRE"
            print("  %-10s at (%6.2f, %6.2f)  min %s %.3f m  at t+%.1f s%s"
                  % (name, ox, oy, kind, d[i], a[i, 0] - a[0, 0],
                     "  [moved %.2f m]" % drift if drift > 0.05 else ""))
            if known and (worst is None or d[i] < worst[1]):
                worst = (name, d[i])

        if not known_any:
            print("\n  These are CENTRE-TO-CENTRE distances. Pass --geometry")
            print("  to subtract the obstacles' extent and get real clearance")
            print("  (%s --template > obstacles.yaml)." % sys.argv[0])
        elif worst and cfg:
            r_safe = cfg["safety"]["r_safe"]
            r_quad = cfg["safety"]["r_quad"]
            print("\n  worst surface clearance %.3f m (%s)" % (worst[1],
                                                               worst[0]))
            if worst[1] < r_quad:
                print("  *** BELOW r_quad %.2f -- the airframe disc "
                      "intersected an obstacle" % r_quad)
            elif worst[1] < r_safe:
                print("  BELOW r_safe %.2f but clear of the airframe disc "
                      "%.2f. The gate is on the BELIEF map, so this means the"
                      % (r_safe, r_quad))
                print("  grid did not see the obstacle where Vicon says it "
                      "is -- a perception result, not a planner one.")
            else:
                print("  clear of r_safe %.2f, margin %.3f m"
                      % (r_safe, worst[1] - r_safe))

    # -------------------------------------------------------- planner health
    print("\nPLANNER")
    if not status:
        print("  no %s in this bag" % T_STATUS)
    else:
        solve, valid, kinds = [], [], defaultdict(int)
        for _, s in status:
            kinds[s.split()[0]] += 1
            for tok in s.split():
                if tok.startswith("solve=") and tok.endswith("ms"):
                    solve.append(float(tok[6:-2]))
                elif tok.startswith("valid="):
                    n, k = tok[6:].split("/")
                    valid.append(float(n) / float(k))
        if solve:
            s = np.array(solve)
            print("  solve      mean %.0f  p95 %.0f  max %.0f ms  (%d ticks)"
                  % (s.mean(), np.percentile(s, 95), s.max(), len(s)))
            if cfg:
                budget = 1000.0 / cfg["rates"]["plan_rate"]
                over = 100.0 * (s > budget).mean()
                print("  budget     %.0f ms at plan_rate %.1f Hz -- %.0f%% of "
                      "ticks over" % (budget, cfg["rates"]["plan_rate"], over))
        if valid:
            w = np.array(valid)
            print("  valid      mean %.0f%%  min %.0f%%"
                  % (100 * w.mean(), 100 * w.min()))
        print("  statuses   %s" % ", ".join(
            "%s x%d" % kv for kv in sorted(kinds.items(),
                                           key=lambda kv: -kv[1])))
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
