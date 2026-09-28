#!/usr/bin/env python3
"""Judge a fly_haa / fly_hpa / fly_desimplex run with the simulator's rule. Offline; no FCU.

  judge    RUN_DIR [...]   one verdict per run (written to RUN_DIR/judgement.json)
  summary  RUN_DIR [...]   table per method x goal index

THE RULE IS experiments/haa_vs_hpa.py WITH --stop-speed 0.15 (planar_sim/harness.py
run()): the episode succeeds at the first instant with t > 1.0 s, distance to the
goal <= GOAL_TOL_M and speed < STOP_SPEED_M_S, and fails on a collision first, or
at the time limit. On the vehicle:

  t = 0        the guarded mission entering MISSION (the first fresh planner
               command after the takeoff hover; before that no planner flew)
  distance     Vicon subject position (ground truth; the goal is given in the
               Vicon world) to the goal
  speed        PX4 local_position/velocity_local, horizontal (operator decision)
  collision    Vicon subject within VEHICLE_RADIUS_M of a pillar rectangle of the
               run's map.yaml (StopBeforeCollision's own geometry)
  time limit   MISSION_TIMEOUT_S, the simulator comparison's episode limit

Not in the simulator: a StopBeforeCollision STOP (collision_stop.jsonl) ends the
planner's flight, so it is its own failure reason, "stop_before_collision". The
remaining failure reasons (stalled / wandered / timeout) use haa_vs_hpa's tests.
"""
import argparse
import bisect
import json
import math
import os
import sys

GOAL_TOL_M = 0.25            # hpa.goal_tol == haa.goal_tol (sim_test40)
STOP_SPEED_M_S = 0.15        # common simulator stop speed (operator decision)
MIN_T_S = 1.0                # harness.run: t > 1.0
MISSION_TIMEOUT_S = 120.0    # operator decision: same as the simulator comparison
VEHICLE_RADIUS_M = 0.31      # fly_helper.VEHICLE_RADIUS_M (propeller-tip disc)
VELOCITY_MAX_GAP_S = 0.1     # a PX4 velocity sample further than this from a Vicon sample is not used
VEHICLE_TOPIC = "/vicon/ROGX2/ROGX2"
VELOCITY_TOPIC = "/rogx2/mavros/local_position/velocity_local"
STATE_TOPIC = "/rogx2/guarded_mission_node/state"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def judge_series(goal, pillars, vicon, velocity, states, guard_stops):
    """The verdict from plain series (testable without a bag).

    goal         (x, y) Vicon world
    pillars      [{"name", "center", "size", "yaw"}]
    vicon        [(t, x, y)] sorted
    velocity     [(t, vx, vy)] sorted, PX4 local ENU
    states       [(t, state_string)] sorted
    guard_stops  [(t, reason)] StopBeforeCollision STOP records
    """
    from collision_stop_core import point_rect_distance
    t0 = next((t for t, s in states if s == "MISSION"), None)
    if t0 is None:
        return dict(success=False, reason="no_mission", t0=None)
    t_limit = t0 + MISSION_TIMEOUT_S
    stop = next(((t, r) for t, r in guard_stops if t >= t0), None)
    vt = [v[0] for v in velocity]

    def speed_at(t):
        i = bisect.bisect_left(vt, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(vt)), key=lambda j: abs(vt[j] - t), default=None)
        if best is None or abs(vt[best] - t) > VELOCITY_MAX_GAP_S:
            return None
        return math.hypot(velocity[best][1], velocity[best][2])

    rows, collision, success = [], None, None
    min_margin = (float("inf"), None)
    for t, x, y in vicon:
        if t < t0:
            continue
        if t > t_limit or (stop is not None and t >= stop[0]):
            break
        for p in pillars:
            m = point_rect_distance((x, y), p["center"], p["size"], p["yaw"]) - VEHICLE_RADIUS_M
            if m < min_margin[0]:
                min_margin = (m, p["name"])
        if min_margin[0] < 0.0:
            collision = dict(t=t - t0, x=x, y=y, pillar=min_margin[1], margin_m=min_margin[0])
            break
        d = math.hypot(x - goal[0], y - goal[1])
        v = speed_at(t)
        rows.append((t - t0, d, v))
        # harness.run order: collision first, then the goal test.
        if d <= GOAL_TOL_M and v is not None and v < STOP_SPEED_M_S and t - t0 > MIN_T_S:
            success = dict(t=t - t0, d_goal=d, speed=v)
            break
    out = dict(t0=t0, goal=list(goal), rule=dict(goal_tol_m=GOAL_TOL_M, stop_speed_m_s=STOP_SPEED_M_S,
                                                 timeout_s=MISSION_TIMEOUT_S, vehicle_radius_m=VEHICLE_RADIUS_M,
                                                 distance="vicon", speed="px4_velocity_local"),
               min_pillar_margin_m=None if min_margin[1] is None else round(min_margin[0], 3),
               nearest_pillar=min_margin[1],
               stop_before_collision=None if stop is None else dict(t=stop[0] - t0, reason=stop[1]))
    if not rows and success is None and collision is None:
        out.update(success=False, reason="no_data_after_mission")
        return out
    last = rows[-1] if rows else (collision["t"], None, None)
    out.update(t_end=round(success["t"] if success else last[0], 3),
               d_goal_end=None if last[1] is None else round(last[1], 3),
               speed_end=None if last[2] is None else round(last[2], 3))
    if collision:
        out.update(success=False, reason="collision", collision=collision)
    elif success:
        out.update(success=True, reason="", t_end=round(success["t"], 3), d_goal_end=round(success["d_goal"], 3),
                   speed_end=round(success["speed"], 3))
    elif stop is not None:
        out.update(success=False, reason="stop_before_collision")
    else:
        # haa_vs_hpa.py: tail over the last 3 s; closing = got closer once, then left.
        t_last = rows[-1][0]
        tail = [r[2] for r in rows if r[0] >= t_last - 3.0 and r[2] is not None]
        d_min = min(r[1] for r in rows)
        if tail and sum(tail) / len(tail) < 0.02:
            why = "stalled"
        elif rows[-1][1] > d_min + 0.30:
            why = "wandered"
        else:
            why = "timeout" if t_last >= MISSION_TIMEOUT_S - 0.5 else "ended_before_timeout"
        out.update(success=False, reason=why, d_goal_min=round(d_min, 3))
    return out


def read_run(run):
    """Series from RUN_DIR/flight.bag, goal_world.txt, map.yaml and collision_stop.jsonl."""
    import rosbag
    import yaml
    goal = [float(v) for v in open(os.path.join(run, "goal_world.txt")).read().split()[:2]]
    pillars = yaml.safe_load(open(os.path.join(run, "map.yaml")))["pillars"]
    bags = sorted(f for f in os.listdir(run) if f.startswith("flight") and f.endswith(".bag"))
    if not bags:
        raise FileNotFoundError("no flight*.bag in " + run)
    vicon, velocity, states = [], [], []
    for name in bags:
        with rosbag.Bag(os.path.join(run, name)) as bag:
            for topic, msg, t in bag.read_messages(topics=[VEHICLE_TOPIC, VELOCITY_TOPIC, STATE_TOPIC]):
                if topic == VEHICLE_TOPIC:
                    tr = msg.transform.translation
                    vicon.append((msg.header.stamp.to_sec(), tr.x, tr.y))
                elif topic == VELOCITY_TOPIC:
                    lin = msg.twist.linear
                    velocity.append((msg.header.stamp.to_sec(), lin.x, lin.y))
                else:
                    states.append((t.to_sec(), msg.data))
    stops = []
    path = os.path.join(run, "collision_stop.jsonl")
    if os.path.exists(path):
        was_stop = False
        for line in open(path):
            rec = json.loads(line)
            is_stop = rec.get("state") == "STOP"
            if is_stop and not was_stop:      # first record of each STOP episode
                stops.append((float(rec["stamp"]), "%s:%s" % (rec.get("cause", ""), rec.get("reason", ""))))
            was_stop = is_stop
    return goal, pillars, sorted(vicon), sorted(velocity), sorted(states), stops


def _meta(run):
    base = os.path.basename(os.path.normpath(run))
    index = None
    p = os.path.join(run, "goal_index.json")
    if os.path.exists(p):
        index = json.load(open(p)).get("index")
    return dict(run=base, mode=base.split("_")[0], goal_index=index)


def cmd_judge(args):
    rc = 0
    for run in args.runs:
        out = dict(_meta(run), **judge_series(*read_run(run)))
        path = os.path.join(run, "judgement.json")
        if os.path.exists(path) and not args.overwrite:
            print("exists (use --overwrite): " + path)
        else:
            with open(path, "w") as f:
                json.dump(out, f, indent=1)
        print(json.dumps({k: out.get(k) for k in ("run", "mode", "goal_index", "success", "reason", "t_end",
                                                    "d_goal_end", "speed_end", "min_pillar_margin_m")}))
    return rc


def cmd_summary(args):
    rows = []
    for run in args.runs:
        p = os.path.join(run, "judgement.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    print("%-10s %5s %4s %8s  %s" % ("mode", "goal", "ok", "t_end s", "reason / run"))
    for r in sorted(rows, key=lambda r: (r.get("goal_index") or 0, r["mode"], r["run"])):
        print("%-10s %5s %4s %8s  %s %s" % (r["mode"], r.get("goal_index"), "yes" if r["success"] else "no",
                                           r.get("t_end"), r.get("reason") or "", r["run"]))
    for mode in sorted({r["mode"] for r in rows}):
        m = [r for r in rows if r["mode"] == mode]
        ok = [r for r in m if r["success"]]
        med = sorted(r["t_end"] for r in ok)
        print("%-10s success %d/%d  median t_end %s" % (
            mode, len(ok), len(m), "%.1f s" % med[len(med) // 2] if med else "-"))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("judge")
    a.add_argument("runs", nargs="+")
    a.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("summary")
    a.add_argument("runs", nargs="+")
    args = p.parse_args()
    return {"judge": cmd_judge, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
