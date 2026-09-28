#!/usr/bin/env python3
"""Opt-in mission executor with terminal CollisionStopGuard intervention.

The original mission_node.py remains unchanged. This executable inherits its
normal mission, geofence and alignment behavior. It must be launched separately
with an explicit guard session, safety timings and settled-velocity thresholds.
It never starts a mission automatically. No stop path arms or disarms the FCU.

Order (differs from mission_node on purpose): ~start takes off to
home_z + ~takeoff_height with NO planner running; after the existing hover
settle, the first fresh planner command on the guarded safe topic enters
MISSION automatically (the unchanged CLIMB rule). The execution status reports
the hover altitude so the launcher can start the planner at that same height.
"""
import json
import math
import os
import sys
import threading

import rospy
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import String

from mission_node import MissionNode, yaw_of
from mission_stop_contract import GuardSession, MissionFailsafe, MissionHoldExecutor, positive


ACTIVE = ("STREAM", "CLIMB", "MISSION", "HOLD", "LAND", "DISARM")
TERMINAL = ("DONE", "PILOT")


def make_hold_target(anchor, stamp, local_frame):
    """MAVROS accepts local ENU values for FRAME_LOCAL_NED (enum 1).

    Position and zero velocity are explicit; acceleration and yaw rate ignored.
    FORCE is deliberately absent. The target is frozen in local coordinates.
    """
    out = PositionTarget()
    out.header.stamp, out.header.frame_id = stamp, local_frame
    out.coordinate_frame = 1
    out.type_mask = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 11)
    out.position.x, out.position.y, out.position.z, out.yaw = anchor
    out.velocity.x = out.velocity.y = out.velocity.z = 0.0
    return out


class GuardedMissionNode(MissionNode):
    def __init__(self):
        # Base subscribers can call overridden callbacks during initialization.
        self.stop_executor = None
        self.guard = None
        self.velocity = None
        self.velocity_stamp = self.velocity_received = None
        self.mav_received = None
        self._land_worker = None
        self._execution_seq = 0
        self._last_loop_time = None
        super(GuardedMissionNode, self).__init__()
        if self.start_req:
            raise ValueError("guarded mission requires ~auto_start:=false")
        if not self.require_frame_alignment:
            raise ValueError("guarded mission requires ~require_frame_alignment:=true")
        P = rospy.get_param
        self.pose_timeout = positive(P("~pose_timeout"), "pose_timeout")
        self.alignment_timeout = positive(P("~alignment_timeout"), "alignment_timeout")
        self.sp_timeout = positive(P("~sp_timeout"), "sp_timeout")
        self.guard = GuardSession(P("~session_id"), P("~guard_timeout"),
                                  P("~future_tolerance"))
        self.mav_timeout = positive(P("~mav_timeout"), "mav_timeout")
        # A value must be provided explicitly, even though one policy is supported.
        self.stop_executor = MissionHoldExecutor(
            P("~stop_policy"), P("~land_retry_interval"), P("~land_max_attempts"),
            speed_threshold=P("~stop_speed_threshold"),
            settle_seconds=P("~stop_settle_seconds"),
            velocity_timeout=P("~stop_velocity_timeout"))
        nominal = rospy.resolve_name(P("~nominal_topic", "commander/set_pose"))
        raw = rospy.resolve_name("mavros/setpoint_raw/local")
        safe = rospy.resolve_name(self.sp_topic)
        if safe != rospy.resolve_name(P("~safe_topic", "commander/set_pose_safe")):
            raise ValueError("~sp_topic must be the explicitly gated safe topic")
        guard_topic = P("~guard_topic", "commander/collision_stop_status")
        execution_topic = P("~execution_topic", "commander/collision_stop_execution")
        diagnostic_topics = [rospy.resolve_name(guard_topic), rospy.resolve_name(execution_topic)]
        if len(set([safe, nominal, raw] + diagnostic_topics)) != 5:
            raise ValueError("nominal, safe, FCU raw, guard and execution topics must resolve separately")
        if any("mavros" in topic.split("/") for topic in diagnostic_topics):
            raise ValueError("guard/execution diagnostics must not use MAVROS command paths")
        self.execution_pub = rospy.Publisher(
            execution_topic,
            String, queue_size=1, latch=True)
        rospy.Subscriber(guard_topic,
                         String, self._guard_cb, queue_size=10)
        rospy.Subscriber("mavros/local_position/velocity_local", TwistStamped,
                         self._velocity_cb, queue_size=10)

    def _cb(self, field, stamp=None):
        parent_cb = super(GuardedMissionNode, self)._cb(field, stamp)

        def callback(msg):
            with self.lock:
                parent_cb(msg)
                if field == "mav":
                    self.mav_received = rospy.get_time()
                    if self.stop_executor is not None and self.stop_executor.latched:
                        self._observe_stop_mode_locked()
        return callback

    def _velocity_cb(self, msg):
        with self.lock:
            stamp = msg.header.stamp.to_sec()
            value = (msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
            if (not all(math.isfinite(v) for v in value + (stamp,)) or stamp <= 0 or
                    (self.velocity_stamp is not None and stamp <= self.velocity_stamp)):
                self.velocity = None
                if self.stop_executor is not None:
                    self.stop_executor.settled_since = None
                return
            self.velocity, self.velocity_stamp = value, stamp
            self.velocity_received = rospy.get_time()
            if self.stop_executor is not None and self.stop_executor.latched:
                self.stop_executor.observe_velocity(value, stamp, self.velocity_received)

    def _guard_cb(self, msg):
        with self.lock:
            if self.guard is None:
                return
            if not self.guard.update(msg.data, rospy.get_time()):
                # Do not let a subsequent good heartbeat erase an observed
                # protocol/clock fault between control-loop ticks.
                if self.state in ACTIVE:
                    self._fail_locked("invalid_guard_status:" + self.guard.error)
                return
            doc = self.guard.latest
            if doc["state"] == "STOP":
                event = dict(doc)
                event["source"] = "collision_stop_guard"
                self._latch_stop_locked(event)

    def _latch_stop_locked(self, event):
        if self.state in TERMINAL or not self.stop_executor.stop(event):
            return
        self.sp = self.cmd = self.frozen = None
        self.sp_epoch = None
        self.t_sp = 0.0
        self.t_arrived = None
        self.start_req = self.land_req = False
        self._enter("STOP_AWAIT_LOCAL_POSE", "-- " + event["reason"])

    def _fail_locked(self, reason):
        self._latch_stop_locked(MissionFailsafe.event(reason, self.guard.session_id))

    def _cb_sp(self, msg):
        with self.lock:
            if self.stop_executor is not None and self.stop_executor.latched:
                return
            if self.guard is None or not self._guard_current_locked(rospy.get_time()):
                return
            super(GuardedMissionNode, self)._cb_sp(msg)

    def _cb_alignment(self, msg):
        with self.lock:
            old_epoch, was_ready = self.alignment.epoch, self.alignment.ready
            super(GuardedMissionNode, self)._cb_alignment(msg)
            changed = old_epoch != self.alignment.epoch or (was_ready and not self.alignment.ready)
            if changed and self.stop_executor is not None and self.stop_executor.latched:
                stamp = self.alignment.valid_from if self.alignment.ready else self.alignment.stamp
                self.stop_executor.reset(stamp if stamp is not None else rospy.get_time())
                self.cmd = self.frozen = self.velocity = None
                self.velocity_stamp = self.velocity_received = None

    def _freeze(self, pose, why):
        # Preserve intentional goal HOLD. Operational failures use the new
        # terminal stop contract instead of returning to a nominal mission.
        if self.stop_executor is None or "GOAL REACHED" in why:
            return super(GuardedMissionNode, self)._freeze(pose, why)
        self._fail_locked(why)

    def _mav_fresh_locked(self, now):
        return (self.mav is not None and self.mav.connected and
                self.mav_received is not None and
                0 <= now - self.mav_received <= self.mav_timeout)

    def _guard_current_locked(self, now):
        return (self.guard is not None and self.guard.current(now) and
                self.guard.latest["state"] != "STOP")

    def _send(self, *args, **kwargs):
        with self.lock:
            if self.stop_executor is not None and self.stop_executor.latched:
                return False
            if not self._guard_current_locked(rospy.get_time()):
                return False
            if not self._mav_fresh_locked(rospy.get_time()):
                return False
            return super(GuardedMissionNode, self)._send(*args, **kwargs)

    def _request(self, srv, args, what):
        with self.lock:
            if self.stop_executor.latched or not self._guard_current_locked(rospy.get_time()):
                return
            if not self._mav_fresh_locked(rospy.get_time()):
                return
        return super(GuardedMissionNode, self)._request(srv, args, what)

    def _wait(self, pose, mav, sp, sp_age, _arr):
        # Takeoff precedes the planner: no nominal command is required here.
        with self.lock:
            now = rospy.get_time()
            if (self.state != "WAIT" or self.stop_executor.latched or
                    not self.guard.takeoff_ready(now) or not self._frame_guard_locked(now)):
                return
            if not mav.connected:
                rospy.logwarn_throttle(2.0, "[guarded mission] no FCU link")
            elif not self.start_req:
                rospy.loginfo_throttle(5.0, "[guarded mission] READY -- rosservice call %s/start",
                                       rospy.get_name())
            else:
                p = pose.pose.position
                self.home = (p.x, p.y, p.z, yaw_of(pose))
                self.active_epoch = self.alignment.epoch
                rospy.loginfo("[guarded mission] home (%.2f, %.2f, %.2f)", p.x, p.y, p.z)
                self._enter("STREAM")

    def _stream(self, pose, mav, sp, sp_age, _arr):
        # mission_node._stream with the altitude from ~takeoff_height only:
        # no planner exists yet, and the launcher passes this same height on.
        if not self._send(*self.home, expected_state="STREAM"):
            return
        if self.elapsed < 1.0:
            return          # PX4 accepts OFFBOARD only once setpoints stream
        if mav.mode != "OFFBOARD":
            if self._every(1.0):
                self._request(self.set_mode, dict(custom_mode="OFFBOARD"), "OFFBOARD")
        elif not mav.armed:
            if self._every(1.0):
                self._request(self.arming, (True,), "ARM")
        else:
            with self.lock:
                if self.state != "STREAM" or not self._frame_guard_locked(rospy.get_time()):
                    return
            self.z_want = self.home[2] + self.z_takeoff
            self.z0 = pose.pose.position.z
            self._enter("CLIMB", "-- armed in OFFBOARD, to z=%.2f" % self.z_want)

    def _observe_stop_mode_locked(self):
        mav, stop = self.mav, self.stop_executor
        if mav is None:
            return
        stop.observe_mode(mav.armed, mav.mode, self.armed_once, self.offboard_once)
        if stop.phase in ("AUTO_LAND", "DONE", "PILOT"):
            self.cmd = self.frozen = None
            if self.state != stop.phase:
                self._enter(stop.phase, "-- stop execution observed FCU mode=" + mav.mode)

    def _publish_execution_locked(self, now):
        stop = self.stop_executor
        self._execution_seq += 1
        event = stop.event or {}
        doc = {"schema": 1, "session_id": self.guard.session_id,
               "seq": self._execution_seq, "stamp": now,
               "state": self.state, "phase": stop.phase,
               "intervention_id": event.get("intervention_id", ""),
               "reason": stop.reason, "failure": bool(event.get("failure", False)),
               "source": event.get("source", ""),
               "observed_mode": self.mav.mode if self.mav is not None else "",
               "armed": bool(self.mav is not None and self.mav.armed),
               "ready": self.guard.ready(now) and self._mav_fresh_locked(now),
               "hold_emitted": stop.holds_emitted,
               "z_want_local": getattr(self, "z_want", None) if self.state != "WAIT" else None,
               "hover_settled": bool(self.state == "CLIMB" and getattr(self, "t_level", None) is not None and
                                     now - self.t_level >= self.settle),
               "planner_fresh": bool(getattr(self, "sp", None) is not None and
                                     now - self.t_sp <= self.sp_timeout),
               "land_attempts": stop.attempts,
               "hold_anchor_local": stop.anchor}
        self.execution_pub.publish(String(data=json.dumps(doc, allow_nan=False)))

    def _request_auto_land_async_locked(self, now):
        if self._land_worker is not None and self._land_worker.is_alive():
            return

        generation = self.stop_executor.generation

        def request():
            with self.lock:
                t = rospy.get_time()
                stop = self.stop_executor
                self._observe_stop_mode_locked()
                if (stop.generation != generation or
                        stop.phase not in ("HOLD", "AUTO_LAND_REQUESTED") or
                        self.state in TERMINAL or not self._mav_fresh_locked(t) or
                        not self.mav.armed or self.mav.mode != "OFFBOARD" or
                        not self._fresh_local_pose_locked(t) or
                        not self.alignment.is_ready(now=t, max_age=self.alignment_timeout) or
                        not stop.should_request_land(t, self.velocity, self.velocity_stamp)):
                    return
                stop.requested_land(t)
                self._enter("AUTO_LAND_REQUESTED", "-- awaiting FCU mode confirmation")
            # Do not hold the mission lock during an RPC. An in-flight RPC cannot
            # be recalled, but no later request is made after observed takeover.
            try:
                response = self.set_mode(custom_mode="AUTO.LAND")
                if not getattr(response, "mode_sent", False):
                    rospy.logwarn("[guarded mission] AUTO.LAND service rejected; hold retained")
            except rospy.ServiceException as exc:
                rospy.logerr("[guarded mission] AUTO.LAND service error: %s", exc)

        self._land_worker = threading.Thread(target=request)
        self._land_worker.daemon = True
        self._land_worker.start()

    def _stop_tick_locked(self, now):
        stop = self.stop_executor
        self._observe_stop_mode_locked()
        if stop.phase in ("AUTO_LAND", "DONE", "PILOT"):
            return
        if not self._mav_fresh_locked(now):
            self.cmd = None
            stop.reason = "fcu_state_unavailable_no_command"
            return
        if not self.mav.armed:
            # A stop before arming is terminal and must not initiate takeoff.
            stop.phase = "DONE"
            self._enter("DONE", "-- intervention while disarmed")
            return
        if self.mav.mode != "OFFBOARD":
            stop.phase = "PILOT"
            self._enter("PILOT", "-- no OFFBOARD authority for hold")
            return
        pose_ok = (self._fresh_local_pose_locked(now) and
                   self.alignment.is_ready(now=now, max_age=self.alignment_timeout))
        if not pose_ok:
            if stop.anchor is not None:
                stop.reset(now)
            self.cmd = None
            stop.reason = "local_pose_or_alignment_unavailable_no_command"
            return
        pose = self.pose
        p, q = pose.pose.position, pose.pose.orientation
        quat = (q.x, q.y, q.z, q.w)
        norm = sum(v * v for v in quat)
        if not all(math.isfinite(v) for v in quat) or abs(norm - 1.0) > 1e-3:
            self.cmd = None
            stop.reason = "invalid_local_quaternion_no_command"
            return
        stop.capture((p.x, p.y, p.z, yaw_of(pose)), pose.header.stamp.to_sec())
        if stop.anchor is None:
            return
        if self.state not in ("STOP_HOLD", "AUTO_LAND_REQUESTED"):
            self._enter("STOP_HOLD", "-- frozen local position and zero velocity")
        self.cmd = make_hold_target(stop.anchor, rospy.Time.now(), self.alignment.local_frame)
        self.pub.publish(self.cmd)
        stop.emitted_hold(now)
        velocity = self.velocity
        if (self.velocity_received is None or
                not 0 <= now - self.velocity_received <= stop.velocity_timeout):
            velocity = None
        if stop.should_request_land(now, velocity, self.velocity_stamp):
            self._request_auto_land_async_locked(now)

    def _tick_locked(self, now):
        if self._last_loop_time is not None and now < self._last_loop_time:
            if self.state in ACTIVE or self.stop_executor.latched:
                self._fail_locked("ros_clock_reversal")
                self.stop_executor.reset(now)
            self.sp = self.cmd = self.velocity = None
        self._last_loop_time = now
        mav, pose = self.mav, self.pose
        if mav is not None and mav.armed:
            self.armed_once = True
        if self.stop_executor.latched:
            self._stop_tick_locked(now)
            return
        # Manual mode/disarm priority precedes watchdog or guard fault handling.
        if mav is not None and self.state not in ("WAIT", "DONE", "PILOT"):
            if self.armed_once and not mav.armed:
                self._enter("DONE", "-- disarmed")
            elif mav.mode == "OFFBOARD":
                self.offboard_once = True
            elif self.offboard_once and self.state != "DISARM":
                self._enter("PILOT", "-- RC/failsafe mode=" + mav.mode)
        if self.state in TERMINAL:
            return
        if self.state in ACTIVE:
            if not self._guard_current_locked(now):
                self._fail_locked("guard_unavailable_or_not_ready")
            elif not self._mav_fresh_locked(now):
                self._fail_locked("fcu_state_unavailable")
            elif not self._fresh_local_pose_locked(now):
                self._fail_locked("px4_local_pose_stale")
            if self.stop_executor.latched:
                self._stop_tick_locked(now)
                return
        frame_ready = self._frame_guard_locked(now)
        if self.stop_executor.latched:
            self._stop_tick_locked(now)
            return
        if not frame_ready or pose is None or mav is None:
            return
        if self.state != "WAIT" and not self._fresh_local_pose_locked(now):
            return
        if self.land_req and self.state in ("STREAM", "CLIMB", "MISSION", "HOLD"):
            self._begin_land(pose, "-- ~land requested")
        elif self.state in ("CLIMB", "MISSION") and math.hypot(
                pose.pose.position.x - self.home[0], pose.pose.position.y - self.home[1]) > self.fence_r:
            self._freeze(pose, "-- geofence")
        if self.stop_executor.latched:
            self._stop_tick_locked(now)
            return
        sp_age = now - self.t_sp
        arrived_for = now - self.t_arrived if self.t_arrived else 0.0
        getattr(self, "_" + self.state.lower())(pose, mav, self.sp, sp_age, arrived_for)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        self.z_want = self.z_takeoff
        while not rospy.is_shutdown():
            with self.lock:
                now = rospy.get_time()
                self._tick_locked(now)
                self._publish_execution_locked(now)
            rate.sleep()


def assert_sole_writer_before_init():
    """Refuse graph errors, an existing mission, or any raw-setpoint writer.

    This runs before base init_node can register a conflicting node name. ROS1
    cannot enforce exclusive publishers atomically, so launch remains exclusive.
    """
    import rosgraph
    remappings = rosgraph.names.load_mappings(sys.argv)
    namespace = remappings.get("__ns", os.environ.get("ROS_NAMESPACE", "/"))
    namespace = "/" + namespace.strip("/") + "/"
    namespace = namespace.replace("//", "/")
    basename = remappings.get("__name", "guarded_mission_node")
    if basename.strip("/").split("/")[-1] == "mission_node":
        raise RuntimeError("guarded node must not use the existing mission_node name")
    caller = rosgraph.names.resolve_name(basename, namespace)
    master = rosgraph.Master("/guarded_mission_preflight")
    publishers, subscribers, services = master.getSystemState()
    raw = rosgraph.names.resolve_name(remappings.get("mavros/setpoint_raw/local", "mavros/setpoint_raw/local"), namespace)
    conflicts = set()
    for topic, nodes in publishers:
        if topic == raw:
            conflicts.update(nodes)
    nodes = set(n for _, names in publishers + subscribers + services for n in names)
    for name in nodes:
        if name in (caller, rosgraph.names.resolve_name("mission_node", namespace)):
            conflicts.add(name)
    if conflicts:
        raise RuntimeError("existing mission/raw setpoint writers: " + ", ".join(sorted(conflicts)))
    if "__name" not in remappings:
        sys.argv.append("__name:=guarded_mission_node")


def main():
    # Any inability to inspect the graph is an error, never permission to run.
    assert_sole_writer_before_init()
    node = GuardedMissionNode()
    # Recheck the resolved raw topic after all ROS remappings have been applied.
    import rosgraph
    publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
    writers = dict(publishers).get(rospy.resolve_name("mavros/setpoint_raw/local"), [])
    if any(name != rospy.get_name() for name in writers):
        raise RuntimeError("another FCU setpoint writer is active: " + repr(writers))
    node.run()


if __name__ == "__main__":
    main()
