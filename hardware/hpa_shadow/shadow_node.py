#!/usr/bin/env python3
"""Subscriber-only ROS 1 HPA shadow inference. No publishers or service clients."""
import argparse
from collections import Counter, deque
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "offboard_flight" / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from shadow_core import (Rejected, camera_calibration_record, camera_intrinsics, check_age, decode_depth,
                         percentiles, pose_discontinuity,
                         quaternion_rpy, stamp_seconds, verify_bundle)
from hpa_depth_adapter import (InputPending, InputRejected, InputExpired, TOPIC_FIELDS,
                               align_inputs, odom_pose, planner_state, quantize_scan)
from planner.hpa.v4 import PinnedV4Runtime


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path, help="new JSONL file; existing files are never overwritten")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--cpu-test", action="store_true", help="explicit CPU offline/diagnostic comparison; never a fallback")
    p.add_argument("--mount-confirmed", action="store_true", help="operator verified the complete training optical-to-body transform")
    goal = p.add_mutually_exclusive_group(required=True)
    goal.add_argument("--goal-local", type=float, nargs=3, metavar=("X", "Y", "Z"))
    goal.add_argument("--goal-current-position", action="store_true", help="explicitly latch first valid synchronized PX4 position as a diagnostic goal")
    p.add_argument("--depth-frame", required=True, help="observed and verified depth optical frame_id")
    p.add_argument("--pose-frame", default="odom")
    p.add_argument("--velocity-frame", default="base_link", help="verify actual MAVROS header; velocity_local.linear remains ENU even when MAVROS labels it base_link")
    p.add_argument("--body-frame", default="base_link")
    p.add_argument("--depth-topic", default="/rogx2/zed2i/zed_node/depth/depth_registered")
    p.add_argument("--camera-info-topic", default="/rogx2/zed2i/zed_node/depth/camera_info")
    p.add_argument("--pose-topic", default="/rogx2/mavros/local_position/odom")
    p.add_argument("--velocity-topic", default="/rogx2/mavros/local_position/velocity_local")
    p.add_argument("--state-topic", default="/rogx2/mavros/state")
    p.add_argument("--invalidation-topic", help="optional std_msgs/Empty from an independently verified estimator reset monitor; any message stops the session")
    p.add_argument("--duration", type=float, default=60.)
    p.add_argument("--rate", type=float, default=100., help="maximum polling rate; not assumed depth rate or model action interval")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--pose-policy", choices=("interpolate", "causal"), default="interpolate")
    p.add_argument("--pose-wait", type=float, default=.04, help="maximum future-odom wait after depth receipt, seconds")
    p.add_argument("--sensor-max-gap", type=float, default=.10, help="operational original-sample age/bracket gap at anchor; not a training threshold")
    p.add_argument("--camera-max-gap", type=float, default=.25,
                   help="legacy argument; ignored for fixed session calibration")
    p.add_argument("--scan-precision", choices=("training-fp16", "float32"), default="training-fp16")
    p.add_argument("--compare-scan-precision", action="store_true", help="extra inference of alternate scan precision; logged separately and adds workload")
    p.add_argument("--goal-source", default="explicit CLI navigation goal in PX4 local ENU")
    p.add_argument("--producer-config", type=Path, default=ROOT / "offboard_flight/scripts/planar_producer_config.example.json")
    p.add_argument("--max-age", type=float, default=.25, help="maximum input age at start and for current sensor health, seconds")
    p.add_argument("--completion-max-age", type=float, default=1.0,
                   help="maximum age of depth/PX4 samples when inference completes, seconds; does not extend the trajectory")
    p.add_argument("--future-tolerance", type=float, default=.01)
    p.add_argument("--state-max-age", type=float, default=2.)
    p.add_argument("--heartbeat", type=float, default=5.)
    p.add_argument("--position-jump-margin", type=float, default=.5)
    p.add_argument("--position-speed-bound", type=float, default=5.)
    p.add_argument("--yaw-jump-margin", type=float, default=.5)
    p.add_argument("--yaw-rate-bound", type=float, default=3.)
    p.add_argument("--workload-label", default="unverified", help="record independently observed concurrent sensor/MPPI/mapping workload")
    return p


class Shadow:
    def __init__(self, args, rospy, bundle, manifest, model_info, log):
        self.args, self.rospy, self.log = args, rospy, log
        self.lock = threading.Lock()
        self.buffers = {name: deque(maxlen=128) for name in ("camera_info", "pose", "velocity")}
        self.latest_depth = None
        self.state = None
        self.stream_last = {}
        self.last_pose = None
        self.fatal = None
        self.epoch = 0
        self.counts = Counter()
        self.goal = None if args.goal_current_position else np.array(args.goal_local, dtype=float)
        self.timings = {key: deque(maxlen=10000) for key in
                        ("preprocess_ms", "model_transfer_ms", "pipeline_ms", "depth_age_start_ms", "depth_age_complete_ms", "callback_to_complete_ms", "pose_wait_ms", "input_arrival_wait_ms", "reference_ms", "precision_compare_ms")}
        self.last_rejection_log = {}
        self.loaded_at = time.monotonic()
        self.emit("session_start", args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  pid=os.getpid(), host=socket.gethostname(), model=model_info, bundle_hashes=manifest,
                  safety="subscriber-only; no publishers, no service clients; no flight-safety validation",
                  reset_limitation="Pose/frame/timestamp discontinuities are heuristics; small or smooth PX4 EKF resets are unobservable in these messages. Use a verified invalidation source before any future control integration.",
                  action_contract={"shape": [10, 3], "components": ["ax_m_s2", "ay_m_s2", "yaw_acceleration_rad_s2"],
                                   "node_dt_s": .1, "frame": "fixed body FLU at depth anchor for all 10 nodes"},
                  input_topic_fields=TOPIC_FIELDS,
                  subscribed_topics={"depth": args.depth_topic, "camera_info": args.camera_info_topic,
                                     "pose_and_angular": args.pose_topic, "linear_velocity": args.velocity_topic,
                                     "fcu_state": args.state_topic},
                  training_gap_thresholds="not supplied; operational thresholds are separate CLI settings",
                  synchronization="velocity_local.linear and odom.twist.angular are original latest stamp<=depth; pose policy explicit",
                  camera_policy="first valid CameraInfo cached for the session; fixed calibration, no timestamp freshness requirement",
                  goal_source=("diagnostic first aligned PX4 position" if args.goal_current_position else args.goal_source))
        import torch
        self.torch = torch
        torch.set_num_threads(args.threads)
        self.runtime = PinnedV4Runtime(bundle, device=args.device, allow_cpu_for_tests=args.cpu_test)
        from planar_producer_factory import build_producer
        self.producer_config = json.loads(args.producer_config.read_text())
        self.current_chunk = None
        self.assembly = self.build_assembly(build_producer)
        self.emit("model_loaded", torch=torch.__version__, cuda=torch.version.cuda,
                  device=args.device, threads=args.threads, elapsed_ms=1000*(time.monotonic()-self.loaded_at),
                  runtime=self.runtime.metadata,
                  producer=self.assembly.effective_config if self.assembly is not None else None,
                  mount={"translation_body_flu_m": [.096, .060, .082],
                         "rotation_optical_to_body_flu": [[0,0,1],[-1,0,0],[0,-1,0]],
                         "confirmed_by_operator": args.mount_confirmed})

    def build_assembly(self, build_producer):
        assembly = build_producer("hpa", self.producer_config, occupancy=None, goal=np.zeros(2),
                                  action_provider=lambda state, goal: self.current_chunk)
        if assembly.hpa.commit != 1:
            raise Rejected("depth-driven shadow reconstructs each observed chunk; configure hpa.commit=1")
        return assembly

    def plan_reference(self, xi, a_prev, stamp, epoch):
        """One algorithm tick on this observation; returns (MPPIResult, log fields)."""
        return self.assembly.producer.plan(xi, goal=self.goal[:2], a_prev=a_prev), None

    def emit(self, event, **fields):
        self.log.write(json.dumps(dict(event=event, wall_unix_s=time.time(), **fields), allow_nan=False) + "\n")
        self.log.flush()

    def reject(self, reason):
        self.counts["reject:" + reason] += 1
        now = time.monotonic()
        if now - self.last_rejection_log.get(reason, -math.inf) >= 1.:
            self.emit("input_rejected", reason=reason, count=self.counts["reject:" + reason])
            self.last_rejection_log[reason] = now

    def invalidate_locked(self, reason):
        if self.fatal is None:
            self.fatal = reason
            self.epoch += 1
            self.latest_depth = None
            for buffer in self.buffers.values():
                buffer.clear()

    def callback(self, name, message):
        received = time.monotonic()
        with self.lock:
            self.counts["received:" + name] += 1
            if self.fatal is not None:
                return
            if name == "invalidation":
                self.invalidate_locked("external estimator invalidation")
                return
            if name == "state" and not message.connected:
                self.invalidate_locked("FCU disconnected")
                return
            try:
                if name == "camera_info":
                    if not self.buffers[name]:
                        record = camera_calibration_record(message, received, self.args.depth_frame)
                        self.buffers[name].append(record)
                        self.emit("camera_calibration_cached", source_header_stamp=record[0],
                                  camera_K=list(message.K), width=message.width, height=message.height,
                                  policy="fixed for session; header stamp is provenance, not dynamic input age")
                    return
                stamp = stamp_seconds(message)
                previous_stamp = self.stream_last.get(name)
                if previous_stamp is not None and stamp < previous_stamp:
                    self.invalidate_locked(name + " timestamp reversal")
                    return
                self.stream_last[name] = stamp
                frames = {"depth": self.args.depth_frame, "camera_info": self.args.depth_frame,
                          "pose": self.args.pose_frame, "velocity": self.args.velocity_frame}
                if name in frames and message.header.frame_id != frames[name]:
                    self.invalidate_locked(name + " frame mismatch/discontinuity: " + repr(message.header.frame_id))
                    return
                if name == "pose":
                    if message.child_frame_id != self.args.body_frame:
                        self.invalidate_locked("odom child_frame_id mismatch: " + repr(message.child_frame_id))
                        return
                    p, q = odom_pose(message)
                    rpy = quaternion_rpy(q)
                    if not np.isfinite(p).all():
                        raise Rejected("nonfinite PX4 position")
                    current = (stamp, p, rpy[2])
                    reason = pose_discontinuity(self.last_pose, current, self.args.position_jump_margin,
                                                self.args.position_speed_bound, self.args.yaw_jump_margin, self.args.yaw_rate_bound)
                    if reason:
                        self.invalidate_locked(reason)
                        return
                    self.last_pose = current
                if previous_stamp == stamp:
                    self.counts["duplicate:" + name] += 1
                    return
                record = (stamp, message, received)
                if name == "depth":
                    if self.latest_depth is not None:
                        self.counts["depth_slot_replacements"] += 1
                    self.latest_depth = record
                elif name == "state":
                    self.state = record
                else:
                    self.buffers[name].append(record)
            except (ValueError, TypeError, AttributeError) as error:
                self.counts["callback_reject:" + name + ":" + str(error)] += 1

    def snapshot(self):
        with self.lock:
            if self.fatal is not None:
                raise RuntimeError(self.fatal)
            return (self.latest_depth, {name: tuple(records) for name, records in self.buffers.items()}, self.state, self.epoch)

    def previous_acceleration(self, anchor_stamp, epoch):
        """Standalone shadow reconstructs each chunk independently by default."""
        return np.zeros(2), "zero for independent shadow reconstruction; no tracked FCU command"

    def depth_scan_validated(self, anchor_stamp, valid):
        """Optional output-adapter hook; standalone shadow has no output timer."""
        pass

    def infer_once(self, consumed):
        depth_record, buffers, state_record, epoch = self.snapshot()
        if depth_record is None or depth_record[0] == consumed:
            return consumed
        stamp, depth_msg, received = depth_record
        now = self.rospy.Time.now().to_sec()
        try:
            age_start = check_age(stamp, now, self.args.max_age, self.args.future_tolerance)
            if state_record is None:
                raise Rejected("missing MAVROS FCU state")
            check_age(state_record[0], now, self.args.state_max_age, self.args.future_tolerance)
            if not state_record[1].connected:
                raise Rejected("FCU disconnected")
            aligned = align_inputs(stamp, buffers, policy=self.args.pose_policy,
                                   pose_wait_s=self.args.pose_wait, waited_s=time.monotonic()-received,
                                   depth_received=received, max_gap=self.args.sensor_max_gap,
                                   camera_max_gap=self.args.camera_max_gap)
            records = aligned["records"]
            for name, (source_stamp, _, _) in records.items():
                if name != "camera_info":
                    check_age(source_stamp, now, self.args.max_age, self.args.future_tolerance)
        except InputPending:
            self.counts["waiting_pose_bracket"] += 1
            return consumed
        except InputExpired as error:
            self.reject(str(error))
            return stamp  # This anchor cannot be revived by a later sample.
        except (Rejected, InputRejected) as error:
            self.reject(str(error))
            return consumed
        # One attempt per synchronized anchor; chunks are never queued or dispatched.
        consumed = stamp
        started = time.perf_counter()
        scan_validated = False
        try:
            depth = decode_depth(depth_msg)
            info = records["camera_info"][1]
            k = camera_intrinsics(info, depth_msg, self.args.depth_frame)
            p, q, rpy = aligned["position"], aligned["quaternion"], aligned["rpy"]
            v, w = aligned["velocity"], aligned["angular"]
            if self.goal is None:
                self.goal = p.copy()
                self.emit("diagnostic_goal_latched", goal_local_enu=self.goal.tolist(), stamp=stamp)
            raw_scan = self.runtime.make_scan(depth, rpy[:2], k)
            self.depth_scan_validated(stamp, True)
            scan_validated = True
            scan, quantization_delta = quantize_scan(raw_scan, self.args.scan_precision)
            encoded = self.runtime.encode_px4_state(p, q, v, w, self.goal)
            preprocessed = time.perf_counter()
            model_start = time.perf_counter()
            actions = self.runtime.infer(scan, encoded)
            model_finished = time.perf_counter()
            if actions.shape != (10, 3) or not np.isfinite(actions).all():
                raise Rejected("nonfinite or invalid output chunk")
            from planner.hpa import BodyActionChunk
            self.current_chunk = BodyActionChunk(actions, float(rpy[2]))
            xi = planner_state(p, v, rpy, w)
            reference_start = time.perf_counter()
            # Independent shadow reconstruction, not an assertion about actual
            # tracked acceleration. The ROS controller connection is separate.
            a_prev, a_prev_source = self.previous_acceleration(stamp, epoch)
            result, planner_log = self.plan_reference(xi, a_prev, stamp, epoch)
            ref = result.reference
            if ref is None or not all(np.isfinite(a).all() for a in (ref.p, ref.v, ref.a, ref.psi, ref.psi_dot)):
                raise Rejected("invalid/nonfinite integrated shadow reference")
            reference_done = time.perf_counter()
            compare = None
            compare_start = time.perf_counter()
            if self.args.compare_scan_precision:
                alternate_mode = "float32" if self.args.scan_precision == "training-fp16" else "training-fp16"
                alternate_scan, _ = quantize_scan(raw_scan, alternate_mode)
                alternate_actions = self.runtime.infer(alternate_scan, encoded)
                compare = dict(alternate_mode=alternate_mode,
                               max_abs_action_delta=float(np.max(np.abs(actions-alternate_actions))))
            finished = time.perf_counter()
            # Serialize the final guard and log emission with invalidation. A
            # reset/disconnect cannot advance the epoch between acceptance and
            # emitting an accepted prediction. Model work stays outside the lock.
            with self.lock:
                fatal, epoch_now, current_state = self.fatal, self.epoch, self.state
                if fatal or epoch != epoch_now:
                    raise RuntimeError(fatal or "estimator epoch invalidated during inference")
                now_complete = self.rospy.Time.now().to_sec()
                complete_age = now_complete - stamp
                timing = dict(preprocess_ms=1000*(preprocessed-started), model_transfer_ms=1000*(model_finished-model_start),
                              pipeline_ms=1000*(finished-started), depth_age_start_ms=1000*age_start,
                              depth_age_complete_ms=1000*complete_age, callback_to_complete_ms=1000*(time.monotonic()-received),
                              pose_wait_ms=1000*max(0., aligned["alignment"].get("future_pose_receipt_offset_s") or 0.),
                              input_arrival_wait_ms=1000*max(0., max(r[2] for r in records.values())-received),
                              reference_ms=1000*(reference_done-reference_start),
                              precision_compare_ms=(1000*(finished-compare_start) if compare is not None else 0.))
                accepted, discard_reason = True, None
                try:
                    if now_complete < now:
                        raise RuntimeError("ROS clock reversed during inference")
                    if current_state is None or not current_state[1].connected:
                        raise RuntimeError("FCU disconnected or state unavailable at completion")
                    check_age(stamp, now_complete, self.args.completion_max_age, self.args.future_tolerance)
                    for name, (source_stamp, _, _) in records.items():
                        if name != "camera_info":
                            check_age(source_stamp, now_complete, self.args.completion_max_age, self.args.future_tolerance)
                    check_age(current_state[0], now_complete, self.args.state_max_age, self.args.future_tolerance)
                except Rejected as error:
                    accepted, discard_reason = False, str(error) + " at completion"
                    self.reject(discard_reason)
                self.emit("prediction", accepted=accepted, discard_reason=discard_reason, anchor_stamp=stamp,
                          estimator_epoch=epoch_now,
                          alignment=aligned["alignment"],
                          input_age_start_ms={name: 1000*(now-record[0]) for name, record in records.items()
                                              if name != "camera_info"},
                          input_age_complete_ms={name: 1000*(now_complete-record[0]) for name, record in records.items()
                                                 if name != "camera_info"},
                          scan_precision=self.args.scan_precision, scan_quantization_max_abs=quantization_delta,
                          precision_comparison=compare,
                          action_shape=list(actions.shape), action_min=actions.min(axis=0).tolist(), action_max=actions.max(axis=0).tolist(),
                          integrated_reference=dict(frame="PX4 local ENU", anchor_stamp=stamp, node_dt_s=float(ref.dt),
                              p=ref.p.tolist(), v=ref.v.tolist(), a=ref.a.tolist(), psi=ref.psi.tolist(), psi_dot=ref.psi_dot.tolist(),
                              a_prev_source=a_prev_source, a_prev_used=np.asarray(a_prev).tolist(), status=result.status),
                          planner=planner_log,
                          input_stamps={name: record[0] for name, record in records.items()},
                          sync_span_ms=1000*(max([stamp]+[r[0] for name, r in records.items() if name != "camera_info"])
                                             -min([stamp]+[r[0] for name, r in records.items() if name != "camera_info"])),
                          frames={"depth": depth_msg.header.frame_id, **{name: record[1].header.frame_id for name, record in records.items()}},
                          camera_K=k.tolist(), camera_distortion_model=info.distortion_model,
                          camera_D=list(info.D), camera_R=list(info.R), camera_P=list(info.P),
                          depth_encoding=depth_msg.encoding, depth_step=depth_msg.step,
                          position_enu=p.tolist(), quaternion_xyzw=q.tolist(), rpy_rad=rpy.tolist(),
                          velocity_enu=v.tolist(), angular_body_flu=w.tolist(), goal_local_enu=self.goal.tolist(),
                          state_unstandardized=encoded.tolist(), valid_beam_fraction=float(scan[1].mean()),
                          anchor_yaw_rad=float(rpy[2]), action_fixed_anchor_body_flu=actions.tolist(), timing=timing,
                          fcu={"connected": current_state[1].connected, "armed": current_state[1].armed, "mode": current_state[1].mode})
                for key, value in timing.items():
                    self.timings[key].append(value)
                self.counts["inferences"] += 1
                self.counts["accepted" if accepted else "discarded_after_inference"] += 1
        except Rejected as error:
            if not scan_validated:
                self.depth_scan_validated(stamp, False)
            self.reject(str(error))
        except ValueError as error:
            if not scan_validated:
                self.depth_scan_validated(stamp, False)
            self.reject("preprocessing/model contract: " + str(error))
        return consumed

    def run(self):
        from geometry_msgs.msg import TwistStamped
        from nav_msgs.msg import Odometry
        from mavros_msgs.msg import State
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Empty
        specs = [("depth", self.args.depth_topic, Image), ("camera_info", self.args.camera_info_topic, CameraInfo),
                 ("pose", self.args.pose_topic, Odometry), ("velocity", self.args.velocity_topic, TwistStamped),
                 ("state", self.args.state_topic, State)]
        if self.args.invalidation_topic:
            specs.append(("invalidation", self.args.invalidation_topic, Empty))
        subscribers = [self.rospy.Subscriber(topic, kind, lambda message, name=name: self.callback(name, message),
                                            queue_size=128 if name in ("pose", "velocity") else 1,
                                            buff_size=2**21 if name == "depth" else 65536)
                       for name, topic, kind in specs]
        start = time.monotonic()
        # None = no wall-clock end (controller output runs until stopped).
        limit = math.inf if self.args.duration is None else self.args.duration
        heartbeat, consumed, previous_clock = start, None, None
        reason = "duration complete"
        failure = None
        try:
            while not self.rospy.is_shutdown() and time.monotonic()-start < limit:
                cycle = time.monotonic()
                now = self.rospy.Time.now().to_sec()
                if previous_clock is not None and now < previous_clock:
                    raise RuntimeError("ROS clock reversal")
                previous_clock = now
                self.snapshot()  # Check fatal guard even when no new depth arrives.
                consumed = self.infer_once(consumed)
                if time.monotonic() >= heartbeat:
                    with self.lock:
                        counts, latest = dict(self.counts), dict(self.stream_last)
                    self.emit("heartbeat", elapsed_s=time.monotonic()-start, counters=counts,
                              last_input_stamp=latest, timing_window={key: percentiles(values) for key, values in self.timings.items()})
                    heartbeat = time.monotonic() + self.args.heartbeat
                time.sleep(max(0., 1./self.args.rate-(time.monotonic()-cycle)))
            if self.rospy.is_shutdown():
                reason = "ROS shutdown or operator interrupt"
        except KeyboardInterrupt:
            reason = "operator interrupt"
        except Exception as error:
            reason = str(error)
            failure = reason
            self.emit("session_invalidated", reason=reason)
            raise
        finally:
            for subscriber in subscribers:
                subscriber.unregister()
            with self.lock:
                counts = dict(self.counts)
                final_fatal = self.fatal
            success = failure is None and final_fatal is None and counts.get("accepted", 0) > 0
            self.emit("summary", reason=reason, success=success,
                      success_criterion="at least one accepted prediction and no session invalidation; inference-only",
                      elapsed_s=time.monotonic()-start, counters=counts,
                      latency_unit="milliseconds", timing_window_capacity=10000,
                      timing={key: percentiles(values) for key, values in self.timings.items()},
                      workload_label=self.args.workload_label,
                      interpretation="Inference-only evidence; first model invocation included; workload label requires independent verification. No control or flight validation.")
        if not success:
            raise RuntimeError(final_fatal or "Session ended with zero accepted predictions")


def pose_arrays(message):
    return odom_pose(message)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.mount_confirmed:
        p.error("--mount-confirmed is required after verifying translation [0.096,0.060,0.082] m and optical x=-body y, optical y=-body z, optical z=body x; do not guess")
    for key in ("duration", "rate", "pose_wait", "sensor_max_gap", "camera_max_gap", "max_age", "completion_max_age", "state_max_age", "heartbeat", "position_jump_margin", "position_speed_bound", "yaw_jump_margin", "yaw_rate_bound"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error("--" + key.replace("_", "-") + " must be finite and positive")
    if args.threads < 1 or args.threads > 4 or args.rate > 200 or args.future_tolerance < 0 or not math.isfinite(args.future_tolerance):
        p.error("require threads 1..4, poll rate <=200 Hz and finite future-tolerance >=0")
    if args.goal_local is not None and not np.isfinite(args.goal_local).all():
        p.error("goal-local must contain finite PX4 local ENU coordinates")
    if any(not value for value in (args.depth_frame, args.pose_frame, args.velocity_frame, args.body_frame)):
        p.error("expected frame names cannot be empty")
    if args.device == "cpu" and not args.cpu_test:
        p.error("CPU requires --cpu-test; deployed CUDA never falls back")
    bundle, manifest, model_info = verify_bundle(args.bundle)
    # Read-only master availability probe: never let init_node wait indefinitely.
    import rosgraph
    import rospy
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5.)
    try:
        rosgraph.Master("/hpa_shadow_probe").getPid()
    except Exception as error:
        raise RuntimeError("ROS master unavailable: " + str(error))
    finally:
        socket.setdefaulttimeout(old_timeout)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x", buffering=1) as log:
        try:
            shadow = Shadow(args, rospy, bundle, manifest, model_info, log)
            rospy.init_node("hpa_shadow", anonymous=True, disable_rosout=True)
            if rospy.get_param("/use_sim_time", False):
                raise Rejected("real-aircraft shadow requires wall ROS time; /use_sim_time is enabled")
            shadow.run()
        except Exception as error:
            log.write(json.dumps(dict(event="fatal", wall_unix_s=time.time(), error=str(error)), allow_nan=False) + "\n")
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("HPA shadow stopped: " + str(error), file=sys.stderr)
        sys.exit(1)
