#!/usr/bin/env python3
"""Helpers for fly_haa.sh / fly_hpa.sh / fly_desimplex.sh. No FCU command here.

  inputs      preflight of the inputs the selected mode and StopBeforeCollision read
  profile     write the ENFORCE CollisionStopGuard profile for one session
  wait-hover  block until the guarded mission reports a settled hover; print its altitude
  frame       goal/altitude in the frames each planner expects (from the shared alignment)
  make-goals  write the fixed comparison goal list (seeded, reproducible)
  goal        one goal of that list by index, screened against today's Vicon obstacle map
  goals-check screen the whole list against today's Vicon obstacle map
  capture-map capture today's Vicon obstacle layout into a map.yaml (no trajectory)
  status      one-shot readout of the mission execution, StopBeforeCollision and FCU state

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


def screen_goal(goal, map_path):
    """Clearance to the Vicon obstacles of today's map only (operator decision 2026-09-28).

    The live /grid_map is NOT used: what it shows near the goal depends on what
    the camera happened to see from the start pose (unknown yesterday, noisy
    5 m depth today), so the same goal passed one day and failed the next. The
    screen has to give all three methods the same answer.
    """
    pillars = _pillar_clearances(goal, map_path)
    entry = {"x": goal[0], "y": goal[1],
             "pillar_clearance_m": round(min(pillars.values()), 3) if pillars else None,
             "nearest_pillar": min(pillars, key=pillars.get) if pillars else None}
    entry["ok"] = bool(not pillars or min(pillars.values()) >= GOAL_CLEARANCE_M)
    return entry


def cmd_goal(args):
    doc = load_goals(args.goals)
    if not 1 <= args.index <= len(doc["goals"]):
        print(json.dumps(dict(ok=False, reason="GOAL_INDEX must be 1..%d" % len(doc["goals"]))))
        return 1
    g = doc["goals"][args.index - 1]
    entry = screen_goal([g["x"], g["y"]], args.map)
    entry.update(index=args.index, seed=doc["seed"], required_m=GOAL_CLEARANCE_M)
    print(json.dumps(entry))
    return 0 if entry["ok"] else 1


def cmd_goals_check(args):
    doc = load_goals(args.goals)
    rows = [dict(index=g["index"], **screen_goal([g["x"], g["y"]], args.map))
            for g in doc["goals"]]
    for r in rows:
        print("%2d  (%.3f, %6.3f)  obstacle %s (%s)  %s" % (
            r["index"], r["x"], r["y"], r["pillar_clearance_m"], r["nearest_pillar"],
            "ok" if r["ok"] else "TOO CLOSE (< %.2f m)" % GOAL_CLEARANCE_M))
    bad = [r["index"] for r in rows if not r["ok"]]
    print(json.dumps(dict(ok=not bad, too_close=bad, required_m=GOAL_CLEARANCE_M)))
    return 0 if not bad else 1


# ---------------------------------------------------------------- map capture
# The obstacle half of the GCS tool vicon_traj/draw_trajectory.py (ViconCapture,
# convex_hull, min_area_rect), without the trajectory it also draws: fly_*.sh
# reads only `pillars` (StopBeforeCollision footprints, goal screening,
# fly_judge). Same fields and the same fit, so either tool's map.yaml works.

def _convex_hull(points):
    """Andrew monotone chain (draw_trajectory.convex_hull)."""
    import numpy as np
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        out = []
        for p in ps:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out
    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def min_area_rect(points):
    """Minimum-area enclosing rectangle (draw_trajectory.min_area_rect): (center, size, yaw)."""
    import numpy as np
    hull = _convex_hull(points)
    if len(hull) < 3:
        lo, hi = points.min(axis=0), points.max(axis=0)
        return (lo + hi) / 2.0, (hi - lo), 0.0
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        theta = math.atan2(edge[1], edge[0])
        c, s = math.cos(-theta), math.sin(-theta)
        local = points @ np.array([[c, -s], [s, c]]).T
        lo, hi = local.min(axis=0), local.max(axis=0)
        size = hi - lo
        area = size[0] * size[1]
        if best is None or area < best[0]:
            mid = (lo + hi) / 2.0
            inv = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            best = (area, inv @ mid, size, theta)
    _, center, size, yaw = best
    if size[1] > size[0]:
        size = size[::-1]
        yaw += math.pi / 2.0
    yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
    return center, size, yaw


def build_layout(segments, markers, duration, drone_subject="ROGX2", prefixes=("wall", "pillar"),
                 min_samples=10):
    """draw_trajectory.ViconCapture._build: {subject/segment: [(x,y,z,yaw)]}, {subject: {marker: [(x,y,z)]}}."""
    import datetime
    import numpy as np
    pillars, drone, skipped, ignored = [], None, [], []
    for subject in sorted(set(k.split("/")[0] for k in segments) | set(markers)):
        is_drone = subject == drone_subject
        if not is_drone and not subject.lower().startswith(prefixes):
            ignored.append(subject)
            continue
        samples = [s for k in segments if k.split("/")[0] == subject for s in segments[k]]
        mk = markers.get(subject, {})
        if len(samples) < min_samples and not mk:
            skipped.append({"name": subject, "n_samples": len(samples)})
            continue
        entry = {"name": subject, "n_samples": int(len(samples))}
        if len(samples) >= min_samples:
            arr = np.array(samples)
            entry["body_position"] = [float(v) for v in arr[:, :3].mean(axis=0)]
            entry["body_yaw"] = float(math.atan2(np.sin(arr[:, 3]).mean(), np.cos(arr[:, 3]).mean()))
        else:
            entry["body_position"] = None
            entry["note"] = "rigid-body segment never published (Tracker object fit failing); footprint from markers"
        if mk:
            names = sorted(mk)
            pts, jitter = [], []
            for name in names:
                smp = np.array(mk[name])
                med = np.median(smp, axis=0)
                pts.append(med)
                jitter.append(float(np.max(np.linalg.norm(smp - med, axis=1))))
            allpts = np.array(pts)
            center, size, yaw = min_area_rect(allpts[:, :2])
            entry.update({"center": [float(center[0]), float(center[1])], "size": [float(size[0]), float(size[1])],
                          "yaw": float(yaw), "z_min": float(allpts[:, 2].min()), "z_max": float(allpts[:, 2].max()),
                          "n_markers": int(len(names)), "marker_names": names,
                          "marker_positions": [[float(v) for v in p] for p in allpts],
                          "marker_max_jitter_m": float(max(jitter)), "footprint_source": "markers"})
        else:
            entry["footprint_source"] = "none"
        if is_drone:
            drone = entry
        else:
            pillars.append(entry)
    return {"frame": "vicon_world", "captured_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "duration_s": duration, "pillars": pillars, "drone": drone, "obstacle_prefixes": list(prefixes),
            "ignored": ignored, "skipped": skipped, "source": "fly_helper.py capture-map"}


def layout_problems(doc):
    """What would make StopBeforeCollision refuse this map (collision_stop_core._obstacles)."""
    bad = []
    if not doc["pillars"]:
        bad.append("no wall*/pillar* obstacle captured")
    for p in doc["pillars"]:
        if "center" not in p:
            bad.append("%s: no markers seen, no footprint" % p["name"])
        if p.get("marker_max_jitter_m", 0.0) > 0.05:
            bad.append("%s: a marker moved %.2f m during capture (ghost/swapped label?)"
                       % (p["name"], p["marker_max_jitter_m"]))
    for s in doc["skipped"]:
        bad.append("%s: occluded / too few samples (%d)" % (s["name"], s["n_samples"]))
    return bad


def cmd_status(args):
    """Read the JSON status topics as messages (rostopic echo wraps long strings)."""
    import rospy
    from mavros_msgs.msg import State
    from std_msgs.msg import Bool, String
    rospy.init_node("fly_status", anonymous=True, disable_rosout=True)

    def get(topic, typ):
        try:
            return rospy.wait_for_message(topic, typ, timeout=args.timeout)
        except rospy.ROSException:
            return None
    ex = get(EXECUTION_TOPIC, String)
    if ex is None:
        print("execution: unavailable (guarded mission not running?)")
    else:
        d = json.loads(ex.data)
        print("execution: state=%s phase=%s hover_settled=%s z_want_local=%s reason=%s" % (
            d.get("state"), d.get("phase"), d.get("hover_settled"), d.get("z_want_local"), d.get("reason") or ""))
    g = get("/rogx2/commander/collision_stop_status", String)
    if g is None:
        print("guard    : unavailable (StopBeforeCollision not running?)")
    else:
        d = json.loads(g.data)
        risks = d.get("risks") or []
        near = min(risks, key=lambda r: r["margin_m"]) if risks else None
        print("guard    : %s ready=%s takeoff_ready=%s reason=%s%s" % (
            d["state"], d["ready"], d.get("takeoff_ready"), d.get("reason") or "-",
            "" if near is None else "  nearest %s %.2f m" % (near["obstacle"], near["margin_m"])))
    m = get("/rogx2/mavros/state", State)
    print("mav      : " + ("unavailable" if m is None else "connected=%s armed=%s mode=%s" % (m.connected, m.armed, m.mode)))
    a = get("/goal_arrive_tf", Bool)
    print("arrived  : " + ("-" if a is None else str(a.data)))
    return 0


def cmd_capture_map(args):
    import rospy
    import yaml
    from geometry_msgs.msg import TransformStamped
    from vicon_bridge.msg import Markers
    if os.path.exists(args.out):
        print("refusing to overwrite " + args.out)
        return 1
    rospy.init_node("fly_capture_map", anonymous=True, disable_rosout=True)
    topics = [(n, n.split("/")[2], n.split("/")[3]) for n, t in rospy.get_published_topics()
              if t == "geometry_msgs/TransformStamped" and n.startswith("/vicon/") and len(n.split("/")) == 4]
    if not topics:
        print("no /vicon/<subject>/<segment> topics -- is start_test_grid.sh vicon running?")
        return 1
    segments, markers, subs = {}, {}, []

    def seg_cb(msg, key):
        t, q = msg.transform.translation, msg.transform.rotation
        segments[key].append((t.x, t.y, t.z, math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))))

    def marker_cb(msg):
        for m in msg.markers:
            if m.occluded or not m.subject_name or not m.marker_name:
                continue
            # Millimetres on this topic (draw_trajectory._marker_cb).
            markers.setdefault(m.subject_name, {}).setdefault(m.marker_name, []).append(
                (m.translation.x / 1000.0, m.translation.y / 1000.0, m.translation.z / 1000.0))
    for topic, subject, segment in sorted(topics):
        key = "%s/%s" % (subject, segment)
        segments[key] = []
        subs.append(rospy.Subscriber(topic, TransformStamped, seg_cb, callback_args=key, queue_size=200))
    subs.append(rospy.Subscriber("/vicon/markers", Markers, marker_cb, queue_size=200))
    rospy.sleep(args.seconds)
    for s in subs:
        s.unregister()
    doc = build_layout(segments, markers, args.seconds)
    for p in doc["pillars"]:
        if "center" in p:
            print("  %-10s center=(%6.3f,%6.3f)  %.3f x %.3f m  yaw=%6.1f deg  markers=%d"
                  % (p["name"], p["center"][0], p["center"][1], p["size"][0], p["size"][1],
                     math.degrees(p["yaw"]), p["n_markers"]))
    if doc["drone"] and doc["drone"].get("body_position"):
        print("  drone      at (%.3f, %.3f)" % tuple(doc["drone"]["body_position"][:2]))
    if doc["ignored"]:
        print("  ignored (not wall*/pillar*): " + ", ".join(doc["ignored"]))
    bad = layout_problems(doc)
    if bad:
        print("NOT saved:\n  " + "\n  ".join(bad))
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "x") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    print("saved %s (%d obstacles)" % (args.out, len(doc["pillars"])))
    return 0


def cmd_profile(args):
    import yaml
    doc = yaml.safe_load(open(args.map))
    profile = {
        "schema": 1, "enabled": True, "session_id": args.session, "mode": "enforce",
        "vicon_frame_id": VICON_FRAME, "map_path": args.map, "log_path": args.log,
        "vehicle": {"topic": VEHICLE_TOPIC, "center_offset_subject_m": CENTER_OFFSET_M,
                    "radius_m": VEHICLE_RADIUS_M},
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
    a = sub.add_parser("status")
    a.add_argument("--timeout", type=float, default=3.0)
    a = sub.add_parser("capture-map")
    a.add_argument("--out", required=True, help="e.g. /home/rogx/traj/fly_YYYYMMDD_HHMM/map.yaml")
    a.add_argument("--seconds", type=float, default=5.0, help="Vicon averaging window (draw_trajectory default 5)")
    a = sub.add_parser("make-goals")
    a.add_argument("--out", required=True)
    for name in ("goal", "goals-check"):
        a = sub.add_parser(name)
        if name == "goal":
            a.add_argument("--index", type=int, required=True)
        a.add_argument("--map", required=True)
        a.add_argument("--goals", default=GOALS_FILE)
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
            "goals-check": cmd_goals_check, "capture-map": cmd_capture_map,
            "status": cmd_status}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
