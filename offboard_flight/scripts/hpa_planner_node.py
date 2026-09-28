#!/usr/bin/env python3
"""Separate V4 HPA node: CUDA inference and existing-controller reference output.

Default output is a diagnostic PositionTarget on /hpa_shadow/position_target.
--controller-output selects the nominal CollisionStopGuard input, never MAVROS.
No arm, mode, mission-start or other FCU service client is created here.
"""
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "hardware/hpa_shadow", ROOT / "hardware/hpa_today",
                  ROOT / "hardware/ekf_state/common", ROOT / "common",
                  Path(__file__).parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ekf_alignment import SharedFrameAlignment
from hpa_controller_bridge import (make_target, make_hold_target, validate_output_topic, check_output_graph,
                                   validate_arrived_topic, check_arrived_graph)
from hpa_depth_adapter import odom_pose, latest_causal, planner_state, InputRejected
from shadow_core import Rejected, check_age
from reference_lifecycle import warmup_runtime
from ros_shadow import ReferenceShadow
from shadow_node import parser as shadow_parser
from shadow_core import quaternion_rpy, verify_bundle


def parser():
    result = shadow_parser()
    result.description = __doc__
    result.add_argument("--controller-output", action="store_true")
    result.add_argument("--output-topic", help="default: diagnostic preview; controller mode: /rogx2/commander/set_pose")
    result.add_argument("--arrived-topic", help="default: preview /hpa_shadow/goal_arrive_tf; controller mode: existing /goal_arrive_tf")
    result.add_argument("--guard-node", default="/rogx2/collision_stop_guard")
    result.add_argument("--observer-node", action="append", default=[],
                        help="explicit passive nominal subscriber (repeatable): recorder/diagnostic only; never mission or command relay")
    result.add_argument("--z-local", type=float, required=True, help="fixed altitude in PX4 local ENU metres")
    result.add_argument("--max-setpoint-step", type=float, default=1.2, help="existing mission XY positional leash, metres")
    result.add_argument("--pub-rate", type=float, default=50.)
    result.add_argument("--alignment-topic", default="/robot/frame_alignment")
    result.add_argument("--alignment-max-age", type=float, default=.5)
    result.add_argument("--world-frame", default="vicon/world", help="existing shared alignment's world frame; never an HPA state input")
    result.add_argument("--local-frame", default="fcu_local")
    # Simulator ReferenceHolder / planar_planner_node plan_timeout: follow a
    # trajectory 0.5 s from its observation (depth) time, then hold.
    result.add_argument("--reference-max-duration", type=float, default=.5,
                        help="trajectory validity from its depth anchor, s (default 0.5 = simulator plan_timeout)")
    result.add_argument("--max-tick-lag", type=float, default=1.0,
                        help="10 Hz tick engine: stop when a tick is this far behind (s). Preview-only "
                             "diagnostics may raise it to measure steady-state compute; controller output may not")
    result.add_argument("--ready-file", type=Path)
    result.add_argument("--summary-file", type=Path)
    # A controller run has no wall-clock end; preview/replay runs default to 60 s.
    result.set_defaults(duration=None)
    return result


def validate_args(args):
    if not (math.isfinite(args.max_tick_lag) and args.max_tick_lag > 0):
        raise ValueError("max-tick-lag must be finite and positive")
    if args.controller_output and args.max_tick_lag > 1.0:
        raise ValueError("controller output keeps the 1 s tick-lag limit")
    if args.controller_output:
        # None, not inf: the value is logged as strict JSON.
        if args.duration is not None:
            raise ValueError("controller output has no --duration; it ends on operator stop, ROS shutdown or fault")
    elif args.duration is None:
        args.duration = 60.
    if args.duration is not None and not (math.isfinite(args.duration) and args.duration > 0):
        raise ValueError("duration must be finite and positive")
    for key in ("rate", "pose_wait", "sensor_max_gap", "max_age", "completion_max_age", "state_max_age",
                "heartbeat", "position_jump_margin", "position_speed_bound", "yaw_jump_margin",
                "yaw_rate_bound", "pub_rate", "max_setpoint_step", "alignment_max_age"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(key + " must be finite and positive")
    if not math.isfinite(args.z_local) or not math.isfinite(args.future_tolerance) or args.future_tolerance < 0:
        raise ValueError("finite z-local and nonnegative future-tolerance required")
    if args.reference_max_duration is not None and (not math.isfinite(args.reference_max_duration) or
                                                    not 0 < args.reference_max_duration <= 1.):
        raise ValueError("reference-max-duration must be in (0,1] seconds")
    if not 1 <= args.threads <= 4 or args.rate > 200 or args.pub_rate > 200:
        raise ValueError("threads must be 1..4 and rates at most 200 Hz")
    if args.device == "cpu" and not args.cpu_test:
        raise ValueError("CPU requires --cpu-test; no CUDA fallback")
    if args.goal_local is not None and not all(math.isfinite(v) for v in args.goal_local):
        raise ValueError("finite PX4 local ENU goal required")
    if not all((args.depth_frame, args.pose_frame, args.velocity_frame, args.body_frame, args.local_frame)):
        raise ValueError("explicit nonempty sensor/local frames required")
    if args.controller_output:
        if args.device != "cuda" or args.goal_current_position or not args.mount_confirmed:
            raise ValueError("controller output requires CUDA, explicit --goal-local and --mount-confirmed")
        if not args.alignment_topic:
            raise ValueError("controller output requires existing shared alignment")
    if args.output_topic is None:
        args.output_topic = "/rogx2/commander/set_pose" if args.controller_output else "/hpa_shadow/position_target"
    if args.arrived_topic is None:
        args.arrived_topic = "/goal_arrive_tf" if args.controller_output else "/hpa_shadow/goal_arrive_tf"


class HPAPlannerNode(ReferenceShadow):
    def __init__(self, args, rospy, bundle, manifest, model_info, log, controller):
        self.alignment = SharedFrameAlignment(expected_world=args.world_frame, expected_local=args.local_frame)
        self.bound_alignment_epoch = None
        self.target_pub = None
        self.target_count = 0
        self.output_last_reason = None
        self.controller = controller
        self.arrived_pub = None
        self.bool_type = None
        self.arrival = None
        # (xy, yaw, anchor) of the last sampled reference: the simulator's
        # ReferenceHolder holds it with zero feedforward once the plan is stale.
        self.reference_hold = None
        # 10 Hz algorithm tick (simulator plan rate) for hpa.commit > 1.
        self.last_tick_time = None
        self.ticks_done = 0
        super().__init__(args, rospy, bundle, manifest, model_info, log)
        self.bridge_ready_file = args.ready_file
        self.output_requires_depth = False
        # Same test as planar_planner_node / PlanarMPPI.at_goal: XY distance
        # <= goal_tol, latched, no speed/z/dwell (mission_node owns the dwell).
        # goal_tol is the producer config value, not a new constant. A
        # diagnostic start-position goal would latch immediately, so arrival
        # is only evaluated for an explicit --goal-local.
        self.goal_tol = self.producer_goal_tol()
        self.arrival_enabled = not args.goal_current_position
        self.emit("goal_arrival_contract", enabled=self.arrival_enabled, goal_tol_m=self.goal_tol,
                  topic=args.arrived_topic, test="PX4 local XY distance <= goal_tol at output tick; latched",
                  after_arrival="stop inference; hold arrival XY/yaw at z-local with zero v/a/yaw-rate "
                                "(planar_planner_node hold contract)")

    last_tick_time = None
    ticks_done = 0
    TICK_S = 0.1          # simulator plan_hz 10: one producer call per 0.1 s of state time
    MAX_TICK_LAG_S = 1.0  # a tick this far behind can no longer be flown: stop the session

    @property
    def commit(self):
        return int(self.producer_config.get("hpa", {}).get("commit", 1))

    def build_assembly(self, build_producer):
        # commit > 1 is served by the 10 Hz tick engine below (simulator semantics).
        return build_producer("hpa", self.producer_config, occupancy=None, goal=np.zeros(2),
                              action_provider=lambda state, goal: self.current_chunk)

    def plan_reference(self, xi, a_prev, stamp, epoch):
        result = self._plan_tick(xi, a_prev, stamp, epoch)
        # The producer was called: this state time is a tick, whatever the result.
        self.last_tick_time = stamp
        self.ticks_done += 1
        return result

    def _plan_tick(self, xi, a_prev, stamp, epoch):
        return self.assembly.producer.plan(xi, goal=self.goal[:2], a_prev=a_prev), None

    def hpa_producer(self):
        return None if self.assembly is None else self.assembly.producer

    def tick_needs_observation(self):
        """True when the NEXT producer call will ask the policy (commit refresh).

        Mirrors planner/hpa/commit.py exactly: a fresh answer when there is no
        cached result or the call counter reached `commit`.
        """
        producer = self.hpa_producer()
        if producer is None:
            return True
        st = getattr(producer, "_commit_state", None)
        return st is None or st["res"] is None or st["i"] >= self.commit

    def producer_goal_tol(self):
        return float(self.assembly.hpa.goal_tol)

    def extra_subscribers(self, rospy):
        """ROS inputs beyond the HPA contract (none here); unregistered at exit."""
        return []

    def snapshot(self):
        result = super().snapshot()
        if self.bridge_ready_file is not None and not self.bridge_ready_file.exists():
            with self.bridge_ready_file.open("x") as stream:
                json.dump(dict(pid=os.getpid(), wall_time=time.time(), device=self.args.device,
                               cuda_ready=self.args.device == "cuda", subscriptions_ready=True,
                               output_topic=self.args.output_topic, controller_output=self.args.controller_output,
                               fcu_service_clients=0), stream)
        return result

    def emit(self, event, **fields):
        if event == "session_start":
            fields["safety"] = ("separate nominal reference publisher; CollisionStopGuard and guarded mission required"
                                if self.args.controller_output else "diagnostic PositionTarget publisher only")
            fields["fcu_service_clients"] = 0
        super().emit(event, **fields)

    def alignment_callback(self, message):
        with self.lock:
            now = self.rospy.Time.now().to_sec()
            self.alignment.update_status(message.data, now=now)
            if self.bound_alignment_epoch is not None and (
                    not self.alignment.ready or self.alignment.epoch != self.bound_alignment_epoch):
                self.invalidate_locked("shared alignment invalidated/epoch changed; restart with a new session")
            elif self.alignment.ready:
                self.bound_alignment_epoch = self.alignment.epoch

    def _frame_locked(self, now, anchor):
        if self.alignment.is_ready(now=now, max_age=self.args.alignment_max_age):
            if self.alignment.valid_from is None or anchor < self.alignment.valid_from:
                return None, "reference predates current alignment epoch"
            return self.alignment.local_frame + "/epoch/" + self.alignment.epoch, None
        if self.bound_alignment_epoch is not None:
            self.invalidate_locked("shared alignment heartbeat stale; restart with a new session")
            return None, self.fatal
        if self.args.controller_output:
            return None, "waiting for existing shared alignment"
        return "PX4_local_ENU_preview", None

    def infer_once(self, consumed):
        # planar_planner_node stops solving once arrived; so does HPA.
        with self.lock:
            if self.arrival is not None:
                return consumed
        if self.commit <= 1:
            return super().infer_once(consumed)
        return self._tick_once(consumed)

    def _tick_once(self, consumed):
        """One simulator plan tick per 0.1 s of state time (hpa.commit > 1).

        Refresh ticks (the commit wrapper will ask the policy) run the full V4
        path on the first depth frame stamped at least 0.1 s after the previous
        tick, so the policy's observation, state and anchor coincide. Committed
        ticks run exactly 0.1 s after the previous one on the PX4 state at that
        time: no new observation, the wrapper serves the committed chunk shifted
        by the call count -- the simulator's 10 Hz call semantics.
        """
        if self.tick_needs_observation():
            with self.lock:
                depth = self.latest_depth
            if depth is None or depth[0] == consumed:
                return consumed
            if self.last_tick_time is not None and depth[0] < self.last_tick_time + self.TICK_S - 1e-6:
                return consumed      # wait for the first frame at or after the next tick time
            return super().infer_once(consumed)
        self._committed_tick(self.last_tick_time + self.TICK_S)
        return consumed

    def _committed_tick(self, tau):
        now = self.rospy.Time.now().to_sec()
        if now < tau:
            return
        if now - tau > getattr(self.args, "max_tick_lag", self.MAX_TICK_LAG_S):
            with self.lock:
                self.invalidate_locked("planner cannot keep the 10 Hz tick (%.2f s behind)" % (now - tau))
            raise RuntimeError(self.fatal)
        with self.lock:
            if self.fatal:
                raise RuntimeError(self.fatal)
            poses, velocities = list(self.buffers["pose"]), list(self.buffers["velocity"])
            state_record, epoch = self.state, self.epoch
        # Causal PX4 state at tau (no network input here): wait until a pose
        # stamped after tau has arrived so no older sample is still in flight.
        if not any(r[0] >= tau for r in poses) and now - tau < self.args.sensor_max_gap + self.TICK_S:
            return
        started = time.perf_counter()
        try:
            if state_record is None or not state_record[1].connected:
                raise Rejected("FCU disconnected")
            odom = latest_causal(poses, tau, self.args.sensor_max_gap, "odom angular/pose")
            velocity = latest_causal(velocities, tau, self.args.sensor_max_gap, "velocity_local.linear")
            p, q = odom_pose(odom[1])
            rpy = quaternion_rpy(q)
            linear, angular = velocity[1].twist.linear, odom[1].twist.twist.angular
            v = np.array([linear.x, linear.y, linear.z])
            w = np.array([angular.x, angular.y, angular.z])
            xi = planner_state(p, v, rpy, w)
        except (Rejected, InputRejected, ValueError) as error:
            # No causal state for this tick time: the call cannot be made.
            with self.lock:
                self.invalidate_locked("committed tick input unavailable at %.3f: %s" % (tau, error))
            raise RuntimeError(self.fatal)
        a_prev, a_prev_source = self.previous_acceleration(tau, epoch)
        result, planner_log = self.plan_reference(xi, a_prev, tau, epoch)
        ref = result.reference
        finished = time.perf_counter()
        with self.lock:
            if self.fatal or epoch != self.epoch:
                raise RuntimeError(self.fatal or "estimator epoch invalidated during tick")
            now_complete = self.rospy.Time.now().to_sec()
            accepted, discard_reason = ref is not None, (None if ref is not None else "producer returned no reference")
            if accepted:
                try:
                    check_age(tau, now_complete, self.args.completion_max_age, self.args.future_tolerance)
                    if self.state is None or not self.state[1].connected:
                        raise Rejected("FCU disconnected or state unavailable at completion")
                except Rejected as error:
                    accepted, discard_reason = False, str(error) + " at completion"
                    self.reject(discard_reason)
            fields = dict(accepted=accepted, discard_reason=discard_reason, anchor_stamp=tau,
                          estimator_epoch=epoch, tick_kind="committed", planner=planner_log,
                          goal_local_enu=self.goal.tolist(), position_enu=p.tolist(), rpy_rad=rpy.tolist(),
                          velocity_enu=v.tolist(), angular_body_flu=w.tolist(),
                          input_stamps=dict(pose=odom[0], velocity=velocity[0]),
                          input_age_complete_ms=dict(pose=1000 * (now_complete - odom[0]),
                                                     velocity=1000 * (now_complete - velocity[0])),
                          timing=dict(pipeline_ms=1000 * (finished - started),
                                      reference_ms=1000 * (finished - started),
                                      tick_age_complete_ms=1000 * (now_complete - tau)))
            if ref is not None:
                fields["integrated_reference"] = dict(
                    frame="PX4 local ENU", anchor_stamp=tau, node_dt_s=float(ref.dt), p=ref.p.tolist(),
                    v=ref.v.tolist(), a=ref.a.tolist(), psi=ref.psi.tolist(), psi_dot=ref.psi_dot.tolist(),
                    a_prev_source=a_prev_source, a_prev_used=np.asarray(a_prev).tolist(), status=str(result.status))
            self.emit("prediction", **fields)
            self.counts["committed_ticks"] += 1
            self.counts["accepted" if accepted else "discarded_after_inference"] += 1

    def depth_scan_validated(self, anchor_stamp, valid):
        # An inference already in flight at arrival is unused; its late
        # (possibly failed) projection must not freeze the output depth gate.
        with self.lock:
            if self.arrival is None and (self.depth_validation is None or anchor_stamp >= self.depth_validation[0]):
                self.depth_validation = (anchor_stamp, bool(valid))

    def _arrival_locked(self, now):
        """Latch arrival from the freshly validated PX4 pose; return it or None."""
        if self.arrival is not None or not self.arrival_enabled:
            return self.arrival
        stamp, pose = self.buffers["pose"][-1][:2]
        position, quaternion = odom_pose(pose)
        distance = math.hypot(position[0] - self.goal[0], position[1] - self.goal[1])
        # Like planar_planner_node, judge arrival only on a state that is usable
        # in the current alignment epoch; a latch that could never be
        # published would silence the node for the rest of the session.
        if distance <= self.goal_tol and self._frame_locked(now, stamp)[1] is None:
            yaw = float(quaternion_rpy(quaternion)[2])
            self.arrival = dict(xy=[float(position[0]), float(position[1])], yaw=yaw, pose_stamp=stamp)
            self.emit("goal_reached", distance_m=distance, goal_tol_m=self.goal_tol, hold_xy=self.arrival["xy"],
                      hold_yaw=yaw, hold_z_local=self.args.z_local, pose_stamp=stamp, sample_stamp=now,
                      goal_local_enu=self.goal.tolist())
        return self.arrival

    def _publish_hold_locked(self, now, xy, yaw, anchor_stamp, kind, arrived):
        """Zero-feedforward hold at a latched XY/yaw (arrival or stale reference)."""
        frame, reason = self._frame_locked(now, anchor_stamp)
        if reason:
            return reason
        try:
            position, _ = odom_pose(self.buffers["pose"][-1][1])
            target = make_hold_target(self.controller, xy, yaw, self.args.z_local,
                                      position[:2], self.args.max_setpoint_step, self.rospy.Time.from_sec(now), frame)
            self.target_pub.publish(target)
            self.arrived_pub.publish(self.bool_type(data=arrived))
            self.target_count += 1
            self.counts[kind + "_hold_targets"] += 1
            self.output_last_reason = None
            self.emit("controller_hold_target", kind=kind, sample_stamp=now, estimator_epoch=self.epoch,
                      frame_id=frame, coordinate_frame=target.coordinate_frame, type_mask=target.type_mask,
                      position=[target.position.x, target.position.y, target.position.z],
                      velocity=[target.velocity.x, target.velocity.y, target.velocity.z],
                      acceleration=[target.acceleration_or_force.x, target.acceleration_or_force.y,
                                    target.acceleration_or_force.z], yaw=target.yaw, yaw_rate=target.yaw_rate,
                      hold_xy=[float(xy[0]), float(xy[1])], current_xy=[float(position[0]), float(position[1])],
                      max_setpoint_step=self.args.max_setpoint_step,
                      topic=self.args.output_topic, arrived_topic=self.args.arrived_topic, arrived=arrived,
                      controller_output=self.args.controller_output)
        except (ValueError, TypeError, IndexError, AttributeError) as error:
            self.invalidate_locked("controller hold serialization: " + str(error))
        return None

    def publish_target(self, timer_event=None):
        # Reset callbacks and final prediction acceptance use the same lock.
        with self.lock:
            now = self.rospy.Time.now().to_sec()
            reason = self.output_block_reason_locked(now)
            if reason is None:
                try:
                    arrival = self._arrival_locked(now)
                except (ValueError, TypeError, IndexError, AttributeError) as error:
                    self.invalidate_locked("goal arrival check: " + str(error))
                    arrival, reason = None, self.fatal
                if arrival is not None:
                    reason = self._publish_hold_locked(now, arrival["xy"], arrival["yaw"], arrival["pose_stamp"],
                                                       "arrival", True)
                    if reason is None:
                        return
            sample = None if reason else self.lifecycle.sample(now, self.epoch)
            if sample is None and reason is None and self.reference_hold is not None and not self.fatal:
                # Stale or no newer trajectory: hold the last commanded point
                # with zero feedforward (simulator ReferenceHolder), never go silent.
                xy, yaw, anchor = self.reference_hold
                reason = self._publish_hold_locked(now, xy, yaw, anchor, "reference_expired", False)
                if reason is None:
                    return
            if sample is None:
                reason = reason or self.lifecycle.reason or "no reference available"
            else:
                frame, reason = self._frame_locked(now, sample["anchor_stamp"])
            if reason:
                if reason != self.output_last_reason:
                    self.emit("controller_output_blocked", reason=reason, sample_stamp=now)
                    self.output_last_reason = reason
                return
            try:
                position, _ = odom_pose(self.buffers["pose"][-1][1])
                stamp = self.rospy.Time.from_sec(now)
                target = make_target(self.controller, sample, self.args.z_local, position[:2],
                                     self.args.max_setpoint_step, stamp, frame)
                self.target_pub.publish(target)
                self.arrived_pub.publish(self.bool_type(data=False))
                self.target_count += 1
                self.reference_hold = ([float(sample["p"][0]), float(sample["p"][1])], float(sample["psi"]),
                                       sample["anchor_stamp"])
                self.output_last_reason = None
                self.emit("controller_reference_target", anchor_stamp=sample["anchor_stamp"], sample_stamp=now,
                          reference_end_stamp=sample["reference_end_stamp"], estimator_epoch=self.epoch,
                          frame_id=frame, coordinate_frame=target.coordinate_frame, type_mask=target.type_mask,
                          position=[target.position.x, target.position.y, target.position.z],
                          velocity=[target.velocity.x, target.velocity.y, target.velocity.z],
                          acceleration=[target.acceleration_or_force.x, target.acceleration_or_force.y,
                                        target.acceleration_or_force.z], yaw=target.yaw, yaw_rate=target.yaw_rate,
                          topic=self.args.output_topic, controller_output=self.args.controller_output,
                          direct_fcu_publications=0, direct_fcu_service_calls=0,
                          downstream_execution="not observed by this node")
            except (ValueError, TypeError, IndexError, AttributeError) as error:
                self.invalidate_locked("controller reference serialization: " + str(error))


def main(argv=None, node_class=HPAPlannerNode, node_name="hpa_planner_node", make_parser=parser,
         check_args=validate_args):
    import rosgraph
    import rospy
    from mavros_msgs.msg import PositionTarget
    from std_msgs.msg import Bool, String
    args = make_parser().parse_args(argv if argv is not None else rospy.myargv()[1:])
    check_args(args)
    bundle, manifest, model_info = verify_bundle(args.bundle)
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5.)
    try:
        rosgraph.Master("/hpa_planner_probe").getPid()
    finally:
        socket.setdefaulttimeout(old_timeout)
    rospy.init_node(node_name, anonymous=True, disable_rosout=True)
    if rospy.get_param("/use_sim_time", False):
        raise ValueError("requires wall ROS time; replay uses constant-shifted timestamps")
    args.output_topic = validate_output_topic(rospy.resolve_name(args.output_topic), args.controller_output)
    args.arrived_topic = validate_arrived_topic(rospy.resolve_name(args.arrived_topic), args.controller_output)
    args.guard_node = rospy.resolve_name(args.guard_node)
    args.observer_node = [rospy.resolve_name(name) for name in args.observer_node]
    master = rosgraph.Master(rospy.get_name())
    check_output_graph(master.getSystemState(), args.output_topic, args.controller_output, args.guard_node,
                           observer_nodes=args.observer_node)
    check_arrived_graph(master.getSystemState(), args.arrived_topic)
    # On ROGX this resolves the unchanged offboard_flight/scripts implementation.
    from guidance_library import Controller
    args.log.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.ready_file, args.summary_file):
        if path is not None and path.exists():
            raise ValueError("refusing to overwrite " + str(path))
    failure, node = None, None
    with args.log.open("x", buffering=1) as log:
        node = node_class(args, rospy, bundle, manifest, model_info, log, Controller())
        node.emit("model_warmup", **warmup_runtime(node.runtime))
        check_output_graph(master.getSystemState(), args.output_topic, args.controller_output, args.guard_node,
                           observer_nodes=args.observer_node)
        check_arrived_graph(master.getSystemState(), args.arrived_topic)
        node.target_pub = rospy.Publisher(args.output_topic, PositionTarget, queue_size=1)
        # Same type/queue/latch as planar_planner_node's goal-arrived publisher.
        node.bool_type = Bool
        node.arrived_pub = rospy.Publisher(args.arrived_topic, Bool, queue_size=1, latch=True)
        alignment_sub = rospy.Subscriber(args.alignment_topic, String, node.alignment_callback, queue_size=10)
        extra_subs = node.extra_subscribers(rospy)
        timer = rospy.Timer(rospy.Duration(1. / args.pub_rate), node.publish_target)
        try:
            node.run()
        except Exception as error:
            failure = str(error)
        finally:
            timer.shutdown()
            timer.join()
            alignment_sub.unregister()
            for sub in extra_subs:
                sub.unregister()
            node.target_pub.unregister()
            node.arrived_pub.unregister()
            summary = dict(success=failure is None and node.target_count > 0, failure=failure,
                           accepted_predictions=node.counts.get("accepted", 0), published_targets=node.target_count,
                           counters=dict(node.counts), preview=not args.controller_output,
                           output_topic=args.output_topic, target_rate_hz=args.pub_rate,
                           bound_alignment_epoch=node.bound_alignment_epoch, fcu_service_calls=0,
                           arrived_topic=args.arrived_topic, goal_reached=node.arrival is not None,
                           arrival=node.arrival, arrival_hold_targets=node.counts.get("arrival_hold_targets", 0),
                           scope="reference serialization/nominal publication; no flight or stop validation")
            node.emit("controller_summary", **summary)
            if args.summary_file is not None:
                with args.summary_file.open("x") as stream:
                    json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
