"""Vicon-only emergency gate, independent of the planar/DeSimplex planner.

No ROS, PX4 state, controller, mode service or estimator fallback lives here.
The result is a latched request; the separately selected mission executor owns
the frozen *local* hold and landing. All operational thresholds are mandatory.
The constant-velocity swept footprint is a configured risk heuristic, not a
validated stopping-distance model or a collision-free guarantee.
"""

import copy
import math
import re
import uuid


ACTIVE_STATES = frozenset(("STREAM", "CLIMB", "MISSION", "HOLD", "STOP_HOLD",
                           "STOP_AWAIT_LOCAL_POSE", "AUTO_LAND_REQUESTED"))
PASSIVE_STATES = frozenset(("LAND", "DISARM", "AUTO_LAND", "PILOT", "DONE"))
EXECUTION_STATES = ACTIVE_STATES | PASSIVE_STATES | frozenset(("WAIT",))
FORWARD_STATES = frozenset(("WAIT", "STREAM", "CLIMB", "MISSION", "HOLD"))


def _number(value, name, minimum=0.0, positive=False):
    if isinstance(value, (bool, str)):
        raise ValueError("%s must be a finite number" % name)
    try:
        v = float(value)
    except (ValueError, TypeError):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(v) or v < minimum or (positive and v == 0):
        raise ValueError("invalid %s" % name)
    return v


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("%s requires %d finite numbers" % (name, size))
    return tuple(_number(x, name, minimum=-float("inf")) for x in value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be explicit and nonempty" % name)
    return value


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("%s requires exactly: %s" % (name, ", ".join(keys)))


def validate_profile(profile):
    p = copy.deepcopy(profile)
    _keys(p, ("schema", "enabled", "session_id", "mode", "vicon_frame_id",
              "vehicle", "map_path", "topics", "nominal",
              "limits", "collision", "log_path"), "profile")
    if p["schema"] != 1 or p["enabled"] is not True:
        raise ValueError("profile must explicitly enable schema 1")
    _text(p["session_id"], "session_id")
    if p["mode"] not in ("shadow", "enforce"):
        raise ValueError("mode must be shadow or enforce")
    for key in ("vicon_frame_id", "map_path", "log_path"):
        _text(p[key], key)
    _keys(p["vehicle"], ("topic", "center_offset_subject_m", "radius_m"), "vehicle")
    _text(p["vehicle"]["topic"], "vehicle.topic")
    p["vehicle"]["center_offset_subject_m"] = _vector(
        p["vehicle"]["center_offset_subject_m"], 3, "vehicle.center_offset_subject_m")
    p["vehicle"]["radius_m"] = _number(p["vehicle"]["radius_m"], "vehicle.radius_m", positive=True)
    _keys(p["topics"], ("nominal", "safe", "status", "execution"), "topics")
    for key, topic in p["topics"].items():
        _text(topic, "topics." + key)
    if len(set(p["topics"].values())) != 4:
        raise ValueError("command/status/execution topics must be distinct")
    _keys(p["nominal"], ("coordinate_frame", "frame_id", "frame_policy"), "nominal")
    if type(p["nominal"]["coordinate_frame"]) is not int or p["nominal"]["coordinate_frame"] != 1:
        raise ValueError("only explicit MAVROS local coordinate_frame=1 is supported")
    _text(p["nominal"]["frame_id"], "nominal.frame_id")
    if p["nominal"]["frame_policy"] not in ("exact", "epoch_tagged"):
        raise ValueError("nominal.frame_policy must be exact or epoch_tagged")
    limit_names = ("rate_hz", "vicon_max_age_s", "max_future_s", "execution_max_age_s",
                   "nominal_max_age_s", "velocity_max_gap_s", "max_subject_skew_s",
                   "max_vicon_speed_m_s")
    _keys(p["limits"], limit_names, "limits")
    for key in limit_names:
        p["limits"][key] = _number(p["limits"][key], "limits." + key,
                                   positive=key not in ("max_future_s", "max_subject_skew_s"))
    _keys(p["collision"], ("horizon_s", "stop_margin_m"), "collision")
    for key in p["collision"]:
        p["collision"][key] = _number(p["collision"][key], "collision." + key,
                                      positive=key == "horizon_s")
    return p


def _rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])


def _quat(q):
    q = _vector(q, 4, "quaternion")
    norm = math.sqrt(sum(x * x for x in q))
    if abs(norm - 1.0) > 0.01:
        raise ValueError("invalid unit quaternion")
    return tuple(x / norm for x in q)


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotate3(v, q):
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty,
            v[1] + w * ty + z * tx - x * tz,
            v[2] + w * tz + x * ty - y * tx)


def point_rect_distance(point, center, size, yaw):
    p = _rotate((point[0] - center[0], point[1] - center[1]), -yaw)
    dx, dy = abs(p[0]) - size[0] / 2, abs(p[1]) - size[1] / 2
    return math.hypot(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0)


def _point_segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm2 = dx * dx + dy * dy
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / norm2)) if norm2 else 0
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def segment_rect_distance(start, end, center, size, yaw):
    """Exact nonnegative segment-to-filled-rectangle distance (no sampling)."""
    a = _rotate((start[0] - center[0], start[1] - center[1]), -yaw)
    b = _rotate((end[0] - center[0], end[1] - center[1]), -yaw)
    half = (size[0] / 2, size[1] / 2)
    low, high = 0.0, 1.0
    for i in range(2):
        d = b[i] - a[i]
        if d == 0:
            if abs(a[i]) > half[i]:
                low, high = 1.0, 0.0
                break
        else:
            t1, t2 = (-half[i] - a[i]) / d, (half[i] - a[i]) / d
            low, high = max(low, min(t1, t2)), min(high, max(t1, t2))
    if low <= high:
        return 0.0
    distances = [max(0.0, point_rect_distance(p, (0, 0), size, 0)) for p in (a, b)]
    for x in (-half[0], half[0]):
        for y in (-half[1], half[1]):
            distances.append(_point_segment_distance((x, y), a, b))
    return min(distances)


class CollisionStopCore(object):
    """Timestamped Vicon input and independent, terminal stop decision.

    Positions/velocities never come from PX4. The obstacles are STATIC: their
    footprints are the map.yaml captured before the flight and are never
    re-read from Vicon (operator decision 2026-09-28 -- a Vicon orientation
    glitch on a pillar that does not move stopped a flight). Only the vehicle
    is a live Vicon input. Vertical obstacle extents are
    deliberately NOT used to waive a stop: all pillars are treated as infinite
    vertical extrusions, a conservative planar contract. Subject histories need
    two increasing header stamps; there is no arrival-time fallback or coast.
    """

    def __init__(self, profile, map_doc):
        self.config = validate_profile(profile)
        self.limits = self.config["limits"]
        self.obstacles = self._obstacles(map_doc)
        self.samples = {}
        self.errors = {}
        self.execution = None
        self.execution_record = None
        self.execution_error = None
        self.nominal = None
        self.nominal_error = None
        self.last_now = None
        self.seq = 0
        self.ever_active = False
        self.stop = None

    def _obstacles(self, doc):
        if not isinstance(doc, dict) or not isinstance(doc.get("pillars"), list) or not doc["pillars"]:
            raise ValueError("map must contain nonempty marker-fitted pillars")
        out = {}
        for item in doc["pillars"]:
            if not isinstance(item, dict):
                raise ValueError("every map pillar must be an object")
            name = _text(item.get("name"), "pillar.name")
            if name == "vehicle":
                raise ValueError("map pillar name 'vehicle' is reserved")
            if name in out:
                raise ValueError("duplicate map pillar " + name)
            center = _vector(item.get("center"), 2, name + ".center")
            size = _vector(item.get("size"), 2, name + ".size")
            if min(size) <= 0:
                raise ValueError("pillar size must be positive")
            yaw = _number(item.get("yaw"), name + ".yaw", minimum=-float("inf"))
            out[name] = {"size": size, "center": center, "yaw": yaw}
        return out

    def update_vicon(self, name, stamp, frame_id, position, quaternion):
        if name != "vehicle":
            raise ValueError("only the vehicle is a Vicon input; obstacles come from the map")
        try:
            stamp = _number(stamp, "Vicon header stamp", positive=True)
            if frame_id != self.config["vicon_frame_id"]:
                raise ValueError("frame_mismatch")
            position, q = _vector(position, 3, "Vicon position"), _quat(quaternion)
            prev = self.samples.get(name)
            if prev is not None and stamp <= prev["stamp"]:
                raise ValueError("timestamp_not_increasing")
            yaw = _yaw(q)
            delta = _rotate3(self.config["vehicle"]["center_offset_subject_m"], q)
            center = tuple(position[i] + delta[i] for i in range(3))
            velocity, omega = None, None
            if prev is not None:
                dt = stamp - prev["stamp"]
                if dt <= self.limits["velocity_max_gap_s"]:
                    velocity = tuple((center[i] - prev["center"][i]) / dt for i in range(2))
                    omega = math.atan2(math.sin(yaw - prev["yaw"]), math.cos(yaw - prev["yaw"])) / dt
                    if math.hypot(*velocity) > self.limits["max_vicon_speed_m_s"]:
                        raise ValueError("position_jump_or_speed_limit")
            self.samples[name] = {"stamp": stamp, "center": center, "yaw": yaw,
                                  "velocity": velocity, "omega": omega}
            self.errors.pop(name, None)
            return True
        except (ValueError, TypeError, OverflowError) as exc:
            # Clear the history. A reset/reordered sample cannot produce a huge
            # derivative or be silently accepted as a new healthy stream.
            self.samples.pop(name, None)
            self.errors[name] = str(exc)
            if self._monitoring():
                self._stop("vicon_%s:%s" % (name, exc), "input_fault")
            return False

    def update_execution(self, value):
        try:
            if not isinstance(value, dict) or value.get("schema") != 1:
                raise ValueError("execution_schema")
            if value.get("session_id") != self.config["session_id"]:
                raise ValueError("execution_session_mismatch")
            if value.get("state") not in EXECUTION_STATES:
                raise ValueError("execution_state")
            stamp = _number(value.get("stamp"), "execution stamp", positive=True)
            seq = value.get("seq")
            if type(seq) is not int or seq < 1:
                raise ValueError("execution_seq")
            if self.execution is not None and stamp <= self.execution["stamp"]:
                raise ValueError("execution_timestamp_not_increasing")
            if self.execution is not None and seq <= self.execution["seq"]:
                raise ValueError("execution_sequence_not_increasing")
            if self.ever_active and value["state"] == "WAIT":
                raise ValueError("execution_restarted_same_session")
            self.execution = {"stamp": stamp, "state": value["state"], "seq": seq}
            # Keep the complete ACK/phase/mode/hold-anchor record for audit.
            # JSON callbacks guarantee serialisable primitives; direct callers
            # should supply the same wire dictionary.
            self.execution_record = copy.deepcopy(value)
            self.execution_error = None
            self.ever_active = self.ever_active or value["state"] in ACTIVE_STATES
            return True
        except (ValueError, TypeError) as exc:
            self.execution_error = str(exc)
            if self._monitoring():
                self._stop(str(exc), "input_fault")
            return False

    def update_nominal(self, value):
        """Store message metadata; the wrapper retains the original ROS message."""
        try:
            stamp = _number(value.get("stamp"), "nominal stamp", positive=True)
            if self.nominal is not None and stamp <= self.nominal["stamp"]:
                raise ValueError("nominal_timestamp_not_increasing")
            cfg = self.config["nominal"]
            if value.get("coordinate_frame") != cfg["coordinate_frame"]:
                raise ValueError("nominal_coordinate_frame")
            frame = value.get("frame_id")
            if cfg["frame_policy"] == "exact":
                valid_frame = frame == cfg["frame_id"]
            else:
                valid_frame = isinstance(frame, str) and re.match(
                    r"^" + re.escape(cfg["frame_id"]) + r"/epoch/[A-Za-z0-9_-]+$", frame) is not None
            if not valid_frame:
                raise ValueError("nominal_frame_id")
            mask = value.get("type_mask")
            if type(mask) is not int or not 0 <= mask < 4096:
                raise ValueError("nominal_type_mask")
            fields = _vector(value.get("fields"), 11, "nominal fields")
            # FORCE with all acceleration ignored is legacy harmless metadata;
            # active force commands are outside this acceleration/position contract.
            if mask & 512 and mask & 448 != 448:
                raise ValueError("nominal_active_force")
            if mask & 511 == 511:
                raise ValueError("nominal_no_translation_command")
            self.nominal = {"stamp": stamp, "frame_id": frame, "type_mask": mask,
                            "coordinate_frame": cfg["coordinate_frame"], "fields": list(fields)}
            self.nominal_error = None
            return True
        except (ValueError, TypeError, AttributeError) as exc:
            self.nominal = None
            self.nominal_error = str(exc)
            if self._monitoring():
                self._stop(str(exc), "input_fault")
            return False

    def _stop(self, reason, kind):
        if self.stop is None:
            self.stop = {"intervention_id": str(uuid.uuid4()), "reason": reason,
                         "cause": kind, "failure": True}

    def _monitoring(self):
        state = self.execution["state"] if self.execution else None
        return self.ever_active and state not in PASSIVE_STATES

    def _age_error(self, age, limit, name):
        if age < -self.limits["max_future_s"]:
            return name + "_future_stamp"
        if age > limit:
            return name + "_stale"
        return None

    def evaluate(self, now):
        now = _number(now, "ROS time", positive=True)
        state = self.execution["state"] if self.execution else None
        if self.last_now is not None and now < self.last_now and state not in PASSIVE_STATES:
            self._stop("ros_time_backwards", "input_fault")
        self.last_now = now
        self.seq += 1
        active = state in ACTIVE_STATES
        # A lost executor heartbeat does not revoke monitoring authority. An
        # explicit observed PILOT/DONE/AUTO_LAND status does stop forwarding.
        monitor = self._monitoring()
        gt_errors = list(self.errors.values())
        errors = []
        if self.execution_error:
            errors.append(self.execution_error)
        execution_age = now - self.execution["stamp"] if self.execution else None
        if execution_age is None:
            errors.append("execution_missing")
        else:
            err = self._age_error(execution_age, self.limits["execution_max_age_s"], "execution")
            if err:
                errors.append(err)
        ages = {}
        for name in ["vehicle"]:
            item = self.samples.get(name)
            if item is None:
                gt_errors.append("vicon_%s_missing" % name)
                continue
            age = now - item["stamp"]
            ages[name] = age
            err = self._age_error(age, self.limits["vicon_max_age_s"], "vicon_" + name)
            if err:
                gt_errors.append(err)
            if item["velocity"] is None:
                gt_errors.append("vicon_%s_velocity_unready" % name)
        if ages and max(ages.values()) - min(ages.values()) > self.limits["max_subject_skew_s"]:
            gt_errors.append("vicon_subject_timestamp_skew")
        errors.extend(gt_errors)
        nominal_age = now - self.nominal["stamp"] if self.nominal else None
        nominal_error = self.nominal_error
        if nominal_age is None:
            nominal_error = nominal_error or "nominal_missing"
        else:
            nominal_error = nominal_error or self._age_error(
                nominal_age, self.limits["nominal_max_age_s"], "nominal")
        # During climb/stop hold, mission owns a command independent of the
        # producer. Missing nominal inhibits forwarding; only MISSION requires
        # it as an ongoing safety contract.
        if state == "MISSION" and nominal_error:
            errors.append(nominal_error)
        # Shadow remains useful with FCU/mission off. Geometry depends solely
        # on valid Vicon, not a synthetic execution state or nominal producer.
        margins = self._risk(now) if not gt_errors else []
        risk = min(margins, key=lambda r: min(r["margin_m"], r["projected_margin_m"])) if margins else None
        if monitor and errors:
            self._stop(errors[0], "input_fault")
        elif monitor and risk and min(risk["margin_m"], risk["projected_margin_m"]) <= self.config["collision"]["stop_margin_m"]:
            self._stop("collision_proximity:" + risk["obstacle"], "collision")
        healthy = not errors and self.stop is None
        ready = healthy and state in FORWARD_STATES and not nominal_error
        # Takeoff precedes the planner: the executor may start the takeoff on
        # a healthy, monitoring guard without any nominal command yet.
        takeoff_ready = healthy and state in FORWARD_STATES
        # A dangerous initial placement must not be advertised as ready even
        # before mission begins. It does not fabricate a flight failure.
        if risk and min(risk["margin_m"], risk["projected_margin_m"]) <= self.config["collision"]["stop_margin_m"]:
            ready = takeoff_ready = False
        result = {"schema": 1, "role": "CollisionStopGuard", "session_id": self.config["session_id"],
                  "seq": self.seq, "stamp": now, "mode": self.config["mode"],
                  "state": "STOP" if self.stop else ("NOMINAL" if active and ready else "STANDBY"),
                  "ready": ready, "takeoff_ready": takeoff_ready, "intervention_id": "", "reason": errors[0] if errors else (nominal_error or ""),
                  "cause": "input_fault" if errors else "", "failure": False,
                  "execution_state": state, "execution_age_s": execution_age,
                  "execution": copy.deepcopy(self.execution_record),
                  "vicon_ages_s": ages, "nominal_age_s": nominal_age,
                  "nominal": copy.deepcopy(self.nominal), "risks": margins,
                  "risk_source": "vicon_gt_only", "allow_nominal": ready,
                  "would_stop": bool(risk and min(risk["margin_m"], risk["projected_margin_m"]) <= self.config["collision"]["stop_margin_m"])}
        if self.stop:
            result.update(self.stop)
            result["ready"] = result["allow_nominal"] = result["takeoff_ready"] = False
        return result

    def _risk(self, now):
        drone = self.samples["vehicle"]
        p, v = drone["center"], drone["velocity"]
        horizon = self.config["collision"]["horizon_s"]
        radius = self.config["vehicle"]["radius_m"]
        drone_age = max(0.0, now - drone["stamp"])
        # A valid but old vehicle sample must not silently shorten lookahead:
        # inflate for the motion between its stamp and decision time. Obstacles
        # are static map footprints (no age, no velocity, no rotation).
        age_pad = drone_age * math.hypot(*v)
        end = (p[0] + v[0] * horizon, p[1] + v[1] * horizon)
        result = []
        for name, obs in sorted(self.obstacles.items()):
            current = point_rect_distance(p, obs["center"], obs["size"], obs["yaw"]) - radius - age_pad
            projected = segment_rect_distance(p, end, obs["center"], obs["size"], obs["yaw"]) - radius - age_pad
            result.append({"obstacle": name, "margin_m": current,
                           "projected_margin_m": projected, "sample_age_padding_m": age_pad,
                           "horizon_s": horizon,
                           "effective_lookahead_from_oldest_stamp_s": horizon + drone_age,
                           "vehicle_velocity_vicon_xy": list(v), "obstacle_source": "map",
                           "vehicle_stamp": drone["stamp"]})
        return result
