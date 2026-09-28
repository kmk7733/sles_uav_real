"""Pure stop/landing contract. No ROS, mode requests, or vehicle side effects.

Collision risk belongs to CollisionStopGuard. These classes only validate
its per-run status stream and execute a terminal local hold/landing decision.
All operational timing and stop thresholds are supplied explicitly by launch.
"""
import json
import math


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def positive(value, name, allow_zero=False):
    if not finite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError("%s must be a finite %s number" % (name, "nonnegative" if allow_zero else "positive"))
    return float(value)


class GuardSession(object):
    """Reject stale, replayed, wrong-run and shadow guard messages."""

    def __init__(self, session_id, timeout, future_tolerance):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("an explicit, unique session_id is required")
        self.session_id = session_id
        self.timeout = positive(timeout, "guard_timeout")
        self.future_tolerance = positive(future_tolerance, "future_tolerance", True)
        self.latest = None
        self.received = None
        self.error = "no_guard_status"

    def update(self, text, now):
        try:
            doc = json.loads(text)
            if not isinstance(doc, dict) or type(doc.get("schema")) is not int or doc.get("schema") != 1:
                raise ValueError("bad_schema")
            if doc.get("session_id") != self.session_id:
                raise ValueError("wrong_session")
            if doc.get("mode") != "enforce":
                raise ValueError("guard_not_enforcing")
            stamp, seq = doc.get("stamp"), doc.get("seq")
            if not finite(stamp) or stamp <= 0 or not finite(now):
                raise ValueError("invalid_stamp")
            if type(seq) is not int or seq < 0:
                raise ValueError("invalid_sequence")
            if not -self.future_tolerance <= now - stamp <= self.timeout:
                raise ValueError("stale_or_future_guard")
            if self.latest is not None and (seq <= self.latest["seq"] or stamp < self.latest["stamp"]):
                raise ValueError("guard_time_or_sequence_reversal")
            if self.received is not None and now < self.received:
                raise ValueError("local_clock_reversal")
            if doc.get("state") not in ("STANDBY", "NOMINAL", "STOP"):
                raise ValueError("invalid_guard_state")
            if type(doc.get("ready")) is not bool or type(doc.get("failure")) is not bool:
                raise ValueError("invalid_guard_flags")
            if not isinstance(doc.get("reason"), str):
                raise ValueError("invalid_guard_reason")
            if doc["state"] == "STOP" and not isinstance(doc.get("intervention_id"), str):
                raise ValueError("missing_intervention_id")
            if doc["state"] == "STOP" and not doc["intervention_id"].strip():
                raise ValueError("missing_intervention_id")
        except (ValueError, TypeError) as exc:
            self.error = str(exc)
            return False
        self.latest, self.received, self.error = doc, now, ""
        return True

    def current(self, now):
        return (not self.error and self.latest is not None and finite(now) and
                0 <= now - self.received <= self.timeout and
                -self.future_tolerance <= now - self.latest["stamp"] <= self.timeout)

    def ready(self, now):
        return (self.current(now) and self.latest["ready"] and
                self.latest["state"] in ("STANDBY", "NOMINAL"))

    def takeoff_ready(self, now):
        """Healthy Vicon monitoring; no nominal command is required before takeoff."""
        return (self.current(now) and self.latest.get("takeoff_ready") is True and
                self.latest["state"] in ("STANDBY", "NOMINAL"))


class MissionFailsafe(object):
    """Execution faults stay distinct from a Vicon collision intervention."""

    @staticmethod
    def event(reason, session_id):
        return {"intervention_id": "mission_failsafe:" + session_id,
                "reason": reason, "failure": True, "source": "mission_failsafe"}


class MissionHoldExecutor(object):
    """Terminal hold then requested AUTO.LAND; mode telemetry confirms it.

    ``capture`` receives a validated PX4 local pose, never a Vicon position.
    An estimator reset clears the frozen target and requires a post-reset pose.
    No service response, inferred ground height, or timer declares landing done.
    """

    POLICIES = ("settled_velocity",)

    def __init__(self, policy, retry_interval, max_attempts, **policy_values):
        if policy not in self.POLICIES:
            raise ValueError("stop_policy must be explicitly selected")
        self.policy = policy
        self.retry_interval = positive(retry_interval, "land_retry_interval")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("land_max_attempts must be positive integer")
        self.max_attempts = max_attempts
        self.speed_threshold = positive(policy_values.get("speed_threshold"), "speed_threshold")
        self.settle_seconds = positive(policy_values.get("settle_seconds"), "settle_seconds")
        self.velocity_timeout = positive(policy_values.get("velocity_timeout"), "velocity_timeout")
        self.phase = "IDLE"
        self.event = None
        self.anchor = None
        self.min_pose_stamp = None
        self.hold_since = self.settled_since = None
        self.holds_emitted = 0
        self.attempts = 0
        self.last_attempt = None
        self.land_issued = False
        self.reason = ""
        self.velocity_stamp = None
        self.velocity_value = None
        self.generation = 0

    @property
    def latched(self):
        return self.event is not None

    def stop(self, event):
        if self.latched:
            return False
        if not event.get("intervention_id"):
            raise ValueError("stop event needs intervention_id")
        self.event = dict(event)
        self.phase, self.reason = "AWAIT_LOCAL_POSE", event["reason"]
        return True

    def reset(self, stamp):
        if not self.latched or self.phase in ("AUTO_LAND", "DONE", "PILOT"):
            return
        self.anchor = None
        self.hold_since = self.settled_since = None
        self.velocity_stamp = self.velocity_value = None
        self.holds_emitted = 0
        self.generation += 1
        self.min_pose_stamp = stamp if self.min_pose_stamp is None else max(stamp, self.min_pose_stamp)
        self.phase = "AWAIT_LOCAL_POSE"
        self.reason = "estimator_reset_waiting_for_fresh_local_pose"

    def capture(self, anchor, pose_stamp):
        if (not self.latched or self.anchor is not None or
                self.phase != "AWAIT_LOCAL_POSE"):
            return False
        if len(anchor) != 4 or not all(finite(v) for v in anchor) or not finite(pose_stamp):
            return False
        if self.min_pose_stamp is not None and pose_stamp < self.min_pose_stamp:
            return False
        self.anchor, self.phase = tuple(anchor), "HOLD"
        return True

    def emitted_hold(self, now):
        if self.anchor is None or self.phase not in ("HOLD", "AUTO_LAND_REQUESTED"):
            raise ValueError("no valid hold is available")
        self.holds_emitted += 1
        if self.hold_since is None:
            self.hold_since = now

    def observe_velocity(self, velocity, stamp, now):
        """Track dwell across distinct post-hold sensor samples, not loop ticks."""
        fresh = (velocity is not None and len(velocity) == 3 and
                 all(finite(v) for v in velocity) and finite(stamp) and
                 0 <= now - stamp <= self.velocity_timeout and
                 self.hold_since is not None and stamp >= self.hold_since)
        slow = fresh and math.sqrt(sum(v * v for v in velocity)) <= self.speed_threshold
        if not slow:
            self.settled_since = None
            self.velocity_value = None
            self.reason = "waiting_for_fresh_settled_px4_velocity"
            return
        if self.velocity_stamp is not None:
            if stamp < self.velocity_stamp:
                self.settled_since = None
                self.velocity_value = None
                return
            if stamp == self.velocity_stamp:
                return
            if stamp - self.velocity_stamp > self.velocity_timeout:
                self.settled_since = None
        self.velocity_stamp, self.velocity_value = stamp, tuple(velocity)
        if self.settled_since is None:
            self.settled_since = stamp

    def should_request_land(self, now, velocity=None, velocity_stamp=None):
        self.observe_velocity(velocity, velocity_stamp, now)
        if self.phase not in ("HOLD", "AUTO_LAND_REQUESTED") or self.holds_emitted == 0:
            return False
        if self.attempts >= self.max_attempts:
            self.reason = "auto_land_attempts_exhausted_hold_retained"
            return False
        if self.last_attempt is not None and now - self.last_attempt < self.retry_interval:
            return False
        return (self.velocity_value is not None and self.settled_since is not None and
                self.velocity_stamp - self.settled_since >= self.settle_seconds)

    def requested_land(self, now):
        self.attempts += 1
        self.last_attempt, self.land_issued = now, True
        self.phase = "AUTO_LAND_REQUESTED"

    def observe_mode(self, armed, mode, armed_once, offboard_once):
        if not self.latched or self.phase in ("DONE", "PILOT"):
            return
        if armed_once and not armed:
            self.phase, self.anchor = "DONE", None
        elif mode == "AUTO.LAND" and self.land_issued:
            self.phase, self.anchor = "AUTO_LAND", None
        elif offboard_once and mode != "OFFBOARD":
            self.phase, self.anchor = "PILOT", None
        elif self.phase == "AUTO_LAND" and mode != "AUTO.LAND":
            # Never reclaim authority even if the pilot subsequently uses OFFBOARD.
            self.phase, self.anchor = "PILOT", None
