#!/usr/bin/env python3
"""Helpers for fly_haa.sh / fly_hpa.sh / fly_desimplex.sh. No FCU command here.

  inputs      preflight of the inputs the selected mode and StopBeforeCollision read
  profile     write the ENFORCE CollisionStopGuard profile for one session
  wait-hover  block until the guarded mission reports a settled hover; print its altitude
  frame       goal/altitude in the frames each planner expects (from the shared alignment)
  make-goals  write the fixed comparison goal list (seeded, reproducible)
  goal        one goal of that list by index, screened against today's pillars and /grid_map
  goals-check screen the whole list against today's pillars and /grid_map

StopBeforeCollision numbers and where each one comes from are in GUARD_LIMITS
below and in guarded_mission.rogx.yaml.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import uuid

# ---------------------------------------------------------------- guard values
# Legacy = offboard_flight vicon_safety_supervisor.py (hardware/collision_stop/
# reference/vicon_traj), calibrated for 0.5 m/s: stops ~0.10 m before a pillar.
VEHICLE_RADIUS_M = 0.31          # legacy --drone-radius (propeller tip disc)
CENTER_OFFSET_M = [0.0, 0.0, 0.0]  # legacy margins are from the Vicon subject position
HORIZON_S = 0.5                  # legacy --lookahead
STOP_MARGIN_M = 0.10             # legacy --d-stop
GUARD_LIMITS = {
    "rate_hz": 20.0,                    # legacy --rate (calibration was taken at 20 Hz)
    "vicon_max_age_s": 0.5,             # legacy --vicon-timeout help text (0.5 s; code default 1.0)
    "max_future_s": 0.05,               # mission_node setpoint future tolerance (-0.05 s)
    "execution_max_age_s": 0.5,         # mission publishes every 20 Hz tick; = mission sp_timeout
    "nominal_max_age_s": 0.5,           # mission_node ~sp_timeout
    "velocity_max_gap_s": 0.1,          # legacy --vel-window (Vicon differencing window)
    "max_subject_skew_s": 0.1,          # = velocity window; skew is padded into the margin anyway
    "max_vicon_speed_m_s": 3.0,         # legacy --vel-sane
    "max_obstacle_yaw_rate_rad_s": 5.0,  # static pillars; only catches marker swaps (offline diagnostic value)
}
VEHICLE_TOPIC = "/vicon/ROGX2/ROGX2"
VICON_FRAME = "vicon/world"
EXECUTION_TOPIC = "/rogx2/commander/collision_stop_execution"


# ---------------------------------------------------------------- goals
# One list for all three methods, so HAA / HPA / DeSimplex fly the SAME goals.
# Region: operator decision (Vicon world). Rejection: the /grid_map border wall
# (depth_to_grid_andert border_wall, measured 2026-09-27: occupied cells up to
# y = -2.475 -> face y = -2.45; right-hand face x = 3.00) must be at least
# GOAL_CLEARANCE_M away, or the goal sits inside the planners' r_safe inflation.
GOALS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_goals.json")
GOAL_SEED = 20260927
GOAL_COUNT = 20                    # 1..15 the planned runs, 16..20 spares in order (same draw sequence)
GOAL_PRIMARY = 15
GOAL_REGION = {"x": [2.1, 2.4], "y": [-2.0, 0.5]}
R_SAFE_M = 0.51                    # planar_producer_config.sim_test40.json safety.r_safe (and HAA r_safe)
GRID_RES_M = 0.05                  # /grid_map resolution: one cell of discretisation margin
GOAL_CLEARANCE_M = R_SAFE_M + GRID_RES_M
BORDER_WALL_FACES = {"y_min": -2.45, "x_max": 3.00}


def sample_goals(seed=GOAL_SEED, count=GOAL_COUNT):
    rng = random.Random(seed)       # Mersenne Twister: identical on every Python 3
    goals, rejected = [], 0
    while len(goals) < count:
        x = rng.uniform(*GOAL_REGION["x"])
        y = rng.uniform(*GOAL_REGION["y"])
        wall = min(y - BORDER_WALL_FACES["y_min"], BORDER_WALL_FACES["x_max"] - x)
        if wall < GOAL_CLEARANCE_M:
            rejected += 1
            continue
        goals.append([round(x, 3), round(y, 3)])
    return goals, rejected


def cmd_make_goals(args):
    goals, rejected = sample_goals()
    doc = {"schema": 1, "frame": VICON_FRAME, "seed": GOAL_SEED, "count": GOAL_COUNT, "primary": GOAL_PRIMARY,
           "spare_rule": "a primary goal that fails goals-check on the day's map is flown by NO method; "
                         "the next unused spare (16, 17, ...) replaces it for ALL three",
           "region": GOAL_REGION,
           "clearance_required_m": GOAL_CLEARANCE_M, "border_wall_faces": BORDER_WALL_FACES,
           "rejected_near_border_wall": rejected,
           "generator": "fly_helper.py make-goals (random.Random(seed).uniform x then y; reject near border wall)",
           "goals": [{"index": i + 1, "x": g[0], "y": g[1]} for i, g in enumerate(goals)]}
    with open(args.out, "x") as f:
        json.dump(doc, f, indent=1)
    print(args.out)
    return 0


def load_goals(path=GOALS_FILE):
    doc = json.load(open(path))
    goals, _ = sample_goals(doc["seed"], doc["count"])
    stored = [[g["x"], g["y"]] for g in doc["goals"]]
    if stored != goals:
        raise ValueError("%s does not match its own seed; refusing an edited goal list" % path)
    return doc


def _pillar_clearances(goal, map_path):
    import yaml
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from collision_stop_core import point_rect_distance
    out = {}
    for p in yaml.safe_load(open(map_path))["pillars"]:
        out[p["name"]] = point_rect_distance(goal, p["center"], p["size"], p["yaw"])
    return out


def _grid_clearance(goal, timeout=5.0):
    """Distance from the goal to the nearest OCCUPIED /grid_map cell edge (unknown is not an obstacle here)."""
    import numpy as np
    import rospy
    from nav_msgs.msg import OccupancyGrid
    rospy.init_node("fly_goal_check", anonymous=True, disable_rosout=True)
    m = rospy.wait_for_message("/grid_map", OccupancyGrid, timeout=timeout)
    if m.header.frame_id.lstrip("/") != VICON_FRAME:
        raise ValueError("/grid_map frame %s is not %s" % (m.header.frame_id, VICON_FRAME))
    r = m.info.resolution
    g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    ys, xs = np.nonzero(g >= 50)
    if not len(xs):
        return float("inf")
    cx = m.info.origin.position.x + (xs + .5) * r
    cy = m.info.origin.position.y + (ys + .5) * r
    return float(np.min(np.hypot(cx - goal[0], cy - goal[1]))) - r / 2


def screen_goal(goal, map_path, use_grid=True):
    pillars = _pillar_clearances(goal, map_path)
    entry = {"x": goal[0], "y": goal[1],
             "pillar_clearance_m": round(min(pillars.values()), 3) if pillars else None,
             "nearest_pillar": min(pillars, key=pillars.get) if pillars else None}
    ok = not pillars or min(pillars.values()) >= GOAL_CLEARANCE_M
    if use_grid:
        entry["grid_clearance_m"] = round(_grid_clearance(goal), 3)
        ok = ok and entry["grid_clearance_m"] >= GOAL_CLEARANCE_M
    entry["ok"] = bool(ok)
    return entry


def cmd_goal(args):
    doc = load_goals(args.goals)
    if not 1 <= args.index <= len(doc["goals"]):
        print(json.dumps(dict(ok=False, reason="GOAL_INDEX must be 1..%d" % len(doc["goals"]))))
        return 1
    g = doc["goals"][args.index - 1]
    entry = screen_goal([g["x"], g["y"]], args.map, use_grid=not args.no_grid)
    entry.update(index=args.index, seed=doc["seed"], required_m=GOAL_CLEARANCE_M)
    print(json.dumps(entry))
    return 0 if entry["ok"] else 1


def cmd_goals_check(args):
    doc = load_goals(args.goals)
    rows = [dict(index=g["index"], **screen_goal([g["x"], g["y"]], args.map, use_grid=not args.no_grid))
            for g in doc["goals"]]
    for r in rows:
        print("%2d  (%.3f, %6.3f)  pillar %s  grid %s  %s" % (
            r["index"], r["x"], r["y"], r["pillar_clearance_m"], r.get("grid_clearance_m", "-"),
            "ok" if r["ok"] else "TOO CLOSE (< %.2f m)" % GOAL_CLEARANCE_M))
    bad = [r["index"] for r in rows if not r["ok"]]
    print(json.dumps(dict(ok=not bad, too_close=bad, required_m=GOAL_CLEARANCE_M)))
    return 0 if not bad else 1


def cmd_profile(args):
    import yaml
    doc = yaml.safe_load(open(args.map))
    names = [p["name"] for p in doc["pillars"]]
    profile = {
        "schema": 1, "enabled": True, "session_id": args.session, "mode": "enforce",
        "vicon_frame_id": VICON_FRAME, "map_path": args.map, "log_path": args.log,
        "vehicle": {"topic": VEHICLE_TOPIC, "center_offset_subject_m": CENTER_OFFSET_M,
                    "radius_m": VEHICLE_RADIUS_M},
        "obstacle_topics": {name: "/vicon/%s/%s" % (name, name) for name in names},
        "topics": {"nominal": "commander/set_pose", "safe": "commander/set_pose_safe",
                   "status": "commander/collision_stop_status", "execution": "commander/collision_stop_execution"},
        "nominal": {"coordinate_frame": 1, "frame_id": "fcu_local", "frame_policy": "epoch_tagged"},
        "limits": GUARD_LIMITS,
        "collision": {"horizon_s": HORIZON_S, "stop_margin_m": STOP_MARGIN_M},
    }
    with open(args.out, "x") as f:
        json.dump(profile, f, indent=2)
    print(args.out)


def _collect(topics, seconds):
    import rospy
    got = {}

    def cb(name):
        def f(msg):
            got.setdefault(name, []).append((rospy.get_time(), msg))
        return f
    subs = [rospy.Subscriber(topic, kind, cb(name), queue_size=5) for name, (topic, kind) in topics.items()]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(.05)
    for s in subs:
        s.unregister()
    return got


def cmd_inputs(args):
    import rospy
    import yaml
    from geometry_msgs.msg import TransformStamped, TwistStamped
    from mavros_msgs.msg import State
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    rospy.init_node("fly_mode_preflight", anonymous=True, disable_rosout=True)
    topics = {"fcu_state": ("/rogx2/mavros/state", State), "alignment": ("/robot/frame_alignment", String),
              "vicon_vehicle": (VEHICLE_TOPIC, TransformStamped)}
    for p in yaml.safe_load(open(args.map))["pillars"]:
        topics["vicon_" + p["name"]] = ("/vicon/%s/%s" % (p["name"], p["name"]), TransformStamped)
    if args.mode in ("hpa", "desimplex"):
        topics.update(depth=("/rogx2/zed2i/zed_node/depth/depth_registered", Image),
                      camera_info=("/rogx2/zed2i/zed_node/depth/camera_info", CameraInfo),
                      odom=("/rogx2/mavros/local_position/odom", Odometry),
                      velocity_local=("/rogx2/mavros/local_position/velocity_local", TwistStamped))
    if args.mode in ("haa", "desimplex"):
        topics["grid"] = ("/grid_map", OccupancyGrid)
    got = _collect(topics, args.seconds)
    report, ok = {}, True
    for name in topics:
        rows = got.get(name, [])
        entry = {"messages": len(rows)}
        if not rows:
            entry["error"] = "missing"
        else:
            received, msg = rows[-1]
            entry["receipt_age_s"] = round(rospy.get_time() - received, 3)
            limit = 2.0 if name in ("fcu_state", "grid") else 0.5
            if entry["receipt_age_s"] > limit:
                entry["error"] = "stale receipt"
            header = getattr(msg, "header", None)
            if header is not None and header.stamp.to_sec() > 0 and name != "grid":
                age = rospy.get_time() - header.stamp.to_sec()
                entry["header_age_s"] = round(age, 3)
                # StopBeforeCollision rejects Vicon stamps more than 0.05 s in the future.
                if name.startswith("vicon") and not -GUARD_LIMITS["max_future_s"] <= age <= GUARD_LIMITS["vicon_max_age_s"]:
                    entry["error"] = "Vicon header clock outside [-0.05, 0.5] s of this computer"
            if name == "fcu_state" and not msg.connected:
                entry["error"] = "FCU not connected"
            if name == "alignment":
                status = json.loads(msg.data)
                entry["valid"] = bool(status.get("valid"))
                if not entry["valid"]:
                    entry["error"] = "alignment not valid: " + str(status.get("reason"))
        ok = ok and "error" not in entry
        report[name] = entry
    print(json.dumps(dict(mode=args.mode, ok=ok, topics=report), indent=1))
    return 0 if ok else 1


def cmd_wait_hover(args):
    import rospy
    from std_msgs.msg import String
    rospy.init_node("fly_wait_hover", anonymous=True, disable_rosout=True)
    latest = {}

    def cb(msg):
        latest["doc"] = json.loads(msg.data)
    sub = rospy.Subscriber(EXECUTION_TOPIC, String, cb, queue_size=1)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and not rospy.is_shutdown():
            doc = latest.get("doc")
            if doc and doc.get("phase") not in (None, "IDLE"):
                print(json.dumps(dict(ok=False, reason="stop latched: " + str(doc.get("reason")))))
                return 1
            if doc and doc.get("state") in ("DONE", "PILOT", "LAND", "DISARM"):
                print(json.dumps(dict(ok=False, reason="mission state " + doc["state"])))
                return 1
            if doc and doc.get("hover_settled") and doc.get("z_want_local") is not None:
                print(json.dumps(dict(ok=True, z_want_local=doc["z_want_local"], state=doc["state"])))
                return 0
            time.sleep(.1)
    finally:
        sub.unregister()
    print(json.dumps(dict(ok=False, reason="no settled hover before timeout")))
    return 1


def cmd_frame(args):
    import numpy as np
    import rospy
    from std_msgs.msg import String
    sys.path.insert(0, args.common)
    from ekf_alignment import SharedFrameAlignment
    rospy.init_node("fly_frame", anonymous=True, disable_rosout=True)
    world = rospy.get_param("/robot/world_frame", VICON_FRAME)
    align = SharedFrameAlignment(expected_world=world)
    msg = rospy.wait_for_message("/robot/frame_alignment", String, timeout=5.)
    align.update_status(msg.data, now=rospy.get_time())
    if not align.is_ready(now=rospy.get_time(), max_age=0.5):
        print(json.dumps(dict(ok=False, reason=align.reason)))
        return 1
    goal_world = np.array([args.goal_x, args.goal_y, 0.0])
    # The operator gives the goal in the Vicon world; HPA/DeSimplex take PX4 local.
    goal_local = align.world_to_local(goal_world)
    # HAA (planar_planner_node) lifts to ~z0 in the WORLD frame, then maps world->local.
    z_world = float(align.local_to_world(np.array([0., 0., args.z_local]))[2])
    print(json.dumps(dict(ok=True, epoch=align.epoch, world_frame=world, z_local=args.z_local,
                          z_world=z_world, goal_world=[args.goal_x, args.goal_y],
                          goal_local=[float(goal_local[0]), float(goal_local[1]), args.z_local])))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("--mode", choices=("haa", "hpa", "desimplex"), required=True)
    a.add_argument("--map", required=True)
    a.add_argument("--seconds", type=float, default=3.)
    a = sub.add_parser("profile")
    a.add_argument("--map", required=True)
    a.add_argument("--session", default=None)
    a.add_argument("--log", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("make-goals")
    a.add_argument("--out", required=True)
    for name in ("goal", "goals-check"):
        a = sub.add_parser(name)
        if name == "goal":
            a.add_argument("--index", type=int, required=True)
        a.add_argument("--map", required=True)
        a.add_argument("--goals", default=GOALS_FILE)
        a.add_argument("--no-grid", action="store_true", help="pillars only (offline)")
    a = sub.add_parser("wait-hover")
    a.add_argument("--timeout", type=float, default=90.)
    a = sub.add_parser("frame")
    a.add_argument("--goal-x", type=float, required=True)
    a.add_argument("--goal-y", type=float, required=True)
    a.add_argument("--z-local", type=float, required=True)
    a.add_argument("--common", required=True, help="directory containing ekf_alignment.py")
    a = sub.add_parser("session")
    args = p.parse_args()
    if args.cmd == "session":
        print(uuid.uuid4().hex)
        return 0
    if args.cmd == "profile" and args.session is None:
        p.error("--session required")
    return {"inputs": cmd_inputs, "profile": cmd_profile, "wait-hover": cmd_wait_hover,
            "frame": cmd_frame, "make-goals": cmd_make_goals, "goal": cmd_goal,
            "goals-check": cmd_goals_check}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
