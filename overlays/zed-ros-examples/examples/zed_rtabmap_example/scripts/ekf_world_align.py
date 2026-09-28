#!/usr/bin/env python3
"""Own a single, frozen world <- EKF local transform and validity heartbeat.

Only startup alignment observes Vicon. Operational body state is always EKF.
No identity fallback and no mid-flight automatic re-alignment are provided.
After a valid epoch is lost, restart the grid stack on the ground to realign.
"""

from collections import deque
import json
import math
from pathlib import Path
import sys
import threading
import uuid

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from mavros_msgs.msg import State
from std_msgs.msg import Empty, String

try:
    import ekf_alignment as alignment
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    import ekf_alignment as alignment


class EkfWorldAlign:
    def __init__(self):
        self._lock = threading.RLock()
        self.mode = rospy.get_param("~mode", "vicon")
        if self.mode not in ("vicon", "novicon"):
            raise ValueError("mode must be vicon or novicon")
        self.world_frame = rospy.get_param("~world_frame", "vicon/world" if self.mode == "vicon" else "plan_world")
        self.local_frame = rospy.get_param("~local_frame", "fcu_local")
        if not self.world_frame or not self.local_frame or self.world_frame == self.local_frame:
            raise ValueError("world and local frames must be distinct nonempty names")
        self.topic = rospy.get_param("~alignment_topic", "/robot/frame_alignment")
        self.n_samples = int(rospy.get_param("~n_samples", 25))
        self.pair_max_age = float(rospy.get_param("~pair_max_age", 0.05))
        self.source_timeout = float(rospy.get_param("~source_timeout", 0.5))
        self.state_timeout = float(rospy.get_param("~state_timeout", 2.0))
        self.position_span = float(rospy.get_param("~stationary_position_span", 0.05))
        self.yaw_span = float(rospy.get_param("~stationary_yaw_span", math.radians(5.0)))
        hz = float(rospy.get_param("~publish_hz", 10.0))
        if self.n_samples < 2 or min(hz, self.pair_max_age, self.source_timeout, self.state_timeout, self.position_span, self.yaw_span) <= 0:
            raise ValueError("alignment sample counts and tolerances must be positive")
        self.epoch = uuid.uuid4().hex
        self.valid = False
        self.ever_valid = False
        self.latched_invalid = False
        self.reason = "waiting_for_fcu"
        self.valid_from = None
        self.yaw = 0.0
        self.translation = np.zeros(3, dtype=np.float64)
        self._last_clock = None
        self._state = None
        self._state_received = None
        self._pose_received = None
        self._last_pose_stamp = None
        self._last_local_sample_stamp = None
        self._last_sample_received = None
        self._last_vicon_stamp = None
        self._local = deque(maxlen=100)
        self._pending_vicon = deque(maxlen=100)
        self._samples = deque(maxlen=self.n_samples)
        self._mount_translation, self._mount_quaternion = self._load_mount()
        rospy.set_param("/robot/world_frame", self.world_frame)
        rospy.set_param("/robot/local_frame", self.local_frame)
        self.publisher = rospy.Publisher(self.topic, String, queue_size=1, latch=True)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._subscriptions = [
            rospy.Subscriber(rospy.get_param("~local_pose_topic", "/rogx2/mavros/local_position/pose"),
                             PoseStamped, self._on_pose, queue_size=100),
            rospy.Subscriber(rospy.get_param("~fcu_state_topic", "/rogx2/mavros/state"),
                             State, self._on_state, queue_size=10),
            rospy.Subscriber(self.topic + "/invalidate", Empty, self._on_invalidate, queue_size=1),
        ]
        if self.mode == "vicon":
            self._subscriptions.append(rospy.Subscriber(
                rospy.get_param("~vicon_topic", "/vicon/ROGX2/ROGX2"),
                TransformStamped, self._on_vicon, queue_size=100))
        self._publish(rospy.Time.now().to_sec())
        self.timer = rospy.Timer(rospy.Duration(1.0/hz), self._on_timer, reset=True)
        rospy.loginfo("ekf_world_align: %s, waiting for %d stationary EKF%s samples; %s <- %s, epoch=%s",
                      self.mode, self.n_samples, "+Vicon paired" if self.mode == "vicon" else "",
                      self.world_frame, self.local_frame, self.epoch)

    def _load_mount(self):
        """T_subject_body: preserve existing calibrated rotation; explicit xyz.

        mount_q_xyzw is the old calibration's body->subject mount rotation;
        its inverse gives subject->body. The subject-coordinate translation is
        supplied independently, including when a rotation calibration is used.
        """
        translation = np.array([float(rospy.get_param("~subject_base_"+axis, 0.0)) for axis in "xyz"])
        path = rospy.get_param("~mount_calib", "") if self.mode == "vicon" else ""
        if path:
            with open(path) as stream:
                data = json.load(stream)
            q = alignment.normalized_quaternion(data["mount_q_xyzw"])
            q = q*np.array([-1.0, -1.0, -1.0, 1.0])
            rospy.loginfo("ekf_world_align: subject->body rotation from %s; explicit subject-frame xyz=%s", path, translation)
        else:
            r, p, y = [float(rospy.get_param("~subject_base_"+axis, 0.0)) for axis in ("roll", "pitch", "yaw")]
            qx = np.array([math.sin(r/2.0), 0.0, 0.0, math.cos(r/2.0)])
            qy = np.array([0.0, math.sin(p/2.0), 0.0, math.cos(p/2.0)])
            q = alignment.quaternion_multiply(alignment.yaw_quaternion(y), alignment.quaternion_multiply(qy, qx))
        if not np.all(np.isfinite(translation)):
            raise ValueError("subject->body translation must be finite")
        return translation, q

    def _observe_clock(self, now):
        if self._last_clock is not None and now < self._last_clock:
            self._invalidate("ros_clock_moved_backwards")
            self._local.clear()
            self._pending_vicon.clear()
            self._samples.clear()
        self._last_clock = now

    def _invalidate(self, reason):
        self.valid = False
        self._samples.clear()
        if self.ever_valid:
            self.latched_invalid = True
        if self.reason != reason:
            if self.latched_invalid:
                rospy.logerr("ekf_world_align: alignment invalid: %s; restart grid stack on ground to realign", reason)
            self.reason = reason

    def _on_invalidate(self, _message):
        with self._lock:
            self.latched_invalid = True
            self._invalidate("explicit_invalidation")
            self._publish(rospy.Time.now().to_sec())

    def _on_state(self, message):
        with self._lock:
            now = rospy.Time.now().to_sec()
            self._observe_clock(now)
            self._state, self._state_received = message, now
            if not message.connected:
                self._invalidate("fcu_disconnected")
                self._publish(now)

    def _available(self, now):
        if self.latched_invalid:
            return False
        if self._state is None or not self._state.connected:
            self._invalidate("waiting_for_fcu")
            return False
        if not 0.0 <= now-self._state_received <= self.state_timeout:
            self._invalidate("fcu_state_stale")
            return False
        if not self.ever_valid and self._state.armed:
            self._invalidate("startup_requires_disarmed_fcu")
            return False
        return True

    @staticmethod
    def _read_pose(position, quaternion):
        p = np.array([position.x, position.y, position.z], dtype=np.float64)
        q = alignment.normalized_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        if not np.all(np.isfinite(p)):
            raise ValueError("nonfinite position")
        return p, q

    def _stamp_valid(self, stamp, now):
        return stamp > 0.0 and -0.1 <= now-stamp <= self.source_timeout

    def _on_pose(self, message):
        with self._lock:
            now = rospy.Time.now().to_sec()
            self._observe_clock(now)
            stamp = message.header.stamp.to_sec()
            if not self._stamp_valid(stamp, now):
                return
            if self._last_pose_stamp is not None and stamp < self._last_pose_stamp:
                self._invalidate("ekf_pose_stamp_moved_backwards")
                self._publish(now)
                return
            if stamp == self._last_pose_stamp:
                return
            try:
                p, q = self._read_pose(message.pose.position, message.pose.orientation)
            except ValueError:
                self._invalidate("invalid_ekf_pose")
                self._publish(now)
                return
            self._last_pose_stamp, self._pose_received = stamp, now
            self._local.append((stamp, p, q))
            if self.ever_valid or not self._available(now):
                return
            if self.mode == "novicon":
                self._append_sample((p, q, None, None), now)
                self._try_estimate(now)
            else:
                self._pair_pending(now)

    def _on_vicon(self, message):
        with self._lock:
            now = rospy.Time.now().to_sec()
            self._observe_clock(now)
            if self.ever_valid or not self._available(now):
                return
            stamp = message.header.stamp.to_sec()
            if not self._stamp_valid(stamp, now) or (self._last_vicon_stamp is not None and stamp <= self._last_vicon_stamp):
                return
            try:
                p, q = self._read_pose(message.transform.translation, message.transform.rotation)
                p = p+alignment.quaternion_rotation(q).dot(self._mount_translation)
                q = alignment.quaternion_multiply(q, self._mount_quaternion)
            except ValueError:
                return
            self._last_vicon_stamp = stamp
            self._pending_vicon.append((stamp, p, q))
            self._pair_pending(now)

    def _pair_pending(self, now):
        while self._pending_vicon and self._local:
            stamp, p_world, q_world = self._pending_vicon[0]
            if now-stamp > self.source_timeout:
                self._pending_vicon.popleft()
                continue
            candidates = [pose for pose in self._local
                          if self._last_local_sample_stamp is None or pose[0] > self._last_local_sample_stamp]
            if not candidates:
                return
            local = min(candidates, key=lambda pose: abs(pose[0]-stamp))
            if abs(local[0]-stamp) > self.pair_max_age:
                if local[0] > stamp:
                    self._pending_vicon.popleft()
                    continue
                return
            self._pending_vicon.popleft()
            self._last_local_sample_stamp = local[0]
            self._append_sample((local[1], local[2], p_world, q_world), now)
            self._try_estimate(now)
            if self.ever_valid:
                return

    def _append_sample(self, sample, now):
        if self._last_sample_received is not None and now-self._last_sample_received > self.source_timeout:
            self._samples.clear()
        self._last_sample_received = now
        self._samples.append(sample)

    def _stationary(self, positions, quaternions):
        p = np.asarray(positions)
        span = np.linalg.norm(np.ptp(p, axis=0))
        yaws = np.unwrap([alignment.yaw_from_quaternion(q) for q in quaternions])
        return span <= self.position_span and float(np.ptp(yaws)) <= self.yaw_span

    def _try_estimate(self, now):
        self.reason = "collecting_stationary_samples"
        if len(self._samples) < self.n_samples:
            return
        local_p, local_q, world_p, world_q = zip(*self._samples)
        if not self._stationary(local_p, local_q) or (self.mode == "vicon" and not self._stationary(world_p, world_q)):
            self.reason = "waiting_for_stationary_body"
            return
        try:
            self.yaw, self.translation = alignment.estimate_alignment(
                local_p, local_q, world_p if self.mode == "vicon" else None,
                world_q if self.mode == "vicon" else None)
        except ValueError as exc:
            self.reason = "initial_alignment_rejected: " + str(exc)
            return
        self.valid = self.ever_valid = True
        self.valid_from = now
        self.reason = "ready"
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.from_sec(now)
        transform.header.frame_id, transform.child_frame_id = self.world_frame, self.local_frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = self.translation
        q = alignment.yaw_quaternion(self.yaw)
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q
        self.broadcaster.sendTransform(transform)
        self._publish(now)
        self._local.clear()
        self._pending_vicon.clear()
        self._samples.clear()
        rospy.loginfo("ekf_world_align: READY %s <- %s yaw_deg=%.6f translation=%s epoch=%s valid_from=%.9f",
                      self.world_frame, self.local_frame, math.degrees(self.yaw), self.translation, self.epoch, self.valid_from)

    def _publish(self, now):
        status = dict(valid=self.valid, epoch=self.epoch, world_frame=self.world_frame,
                      local_frame=self.local_frame, yaw=float(self.yaw),
                      translation=self.translation.tolist(), stamp=now,
                      valid_from=self.valid_from, epoch_started_at=self.valid_from,
                      reason=self.reason)
        self.publisher.publish(String(data=json.dumps(status, separators=(",", ":"), allow_nan=False)))

    def _on_timer(self, _event):
        with self._lock:
            now = rospy.Time.now().to_sec()
            self._observe_clock(now)
            if self._available(now) and self.ever_valid:
                if self._pose_received is None or not 0.0 <= now-self._pose_received <= self.source_timeout:
                    self._invalidate("ekf_pose_stale")
                elif self._last_pose_stamp is None or not self._stamp_valid(self._last_pose_stamp, now):
                    self._invalidate("ekf_pose_source_stamp_stale")
            self._publish(now)


def main():
    rospy.init_node("ekf_world_align")
    EkfWorldAlign()
    rospy.spin()


if __name__ == "__main__":
    main()
