"""ROS-independent, float64 fixed world <- FCU-local alignment contract.

The JSON heartbeat owns validity. A latched TF is never evidence of validity.
Vectors are already expressed in MAVROS local ENU: these helpers only apply
the fixed local/world yaw alignment, not another body/ENU or ENU/NED change.
"""

import copy
import json
import math
import time

import numpy as np


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotation_z(yaw):
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def normalized_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("expected a finite quaternion in xyzw order")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return q / norm


def quaternion_multiply(left, right):
    x, y, z, w = normalized_quaternion(left)
    a, b, c, d = normalized_quaternion(right)
    return normalized_quaternion([
        w*a + x*d + y*c - z*b,
        w*b - x*c + y*d + z*a,
        w*c + x*b - y*a + z*d,
        w*d - x*a - y*b - z*c,
    ])


def yaw_quaternion(yaw):
    return np.array([0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0)], dtype=np.float64)


def quaternion_rotation(q):
    x, y, z, w = normalized_quaternion(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def yaw_from_quaternion(q):
    x, y, z, w = normalized_quaternion(q)
    return math.atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))


def estimate_alignment(local_positions, local_quaternions,
                       world_positions=None, world_quaternions=None):
    """Estimate one yaw and xyz translation from simultaneous stationary poses.

    No world observations means novicon: the mean initial body position and
    heading define world zero. Stationarity and sample timing are caller checks.
    """
    local_positions = np.asarray(local_positions, dtype=np.float64)
    if local_positions.ndim != 2 or local_positions.shape[1:] != (3,):
        raise ValueError("local positions must have shape (n,3)")
    if not len(local_positions) or not np.all(np.isfinite(local_positions)):
        raise ValueError("nonempty finite local positions required")
    local_yaws = np.array([yaw_from_quaternion(q) for q in local_quaternions])
    if len(local_yaws) != len(local_positions):
        raise ValueError("pose count mismatch")
    if world_positions is None and world_quaternions is None:
        differences = -local_yaws
        world_positions = np.zeros_like(local_positions)
    elif world_positions is not None and world_quaternions is not None:
        world_positions = np.asarray(world_positions, dtype=np.float64)
        if world_positions.shape != local_positions.shape or not np.all(np.isfinite(world_positions)):
            raise ValueError("world positions must match local positions")
        world_yaws = np.array([yaw_from_quaternion(q) for q in world_quaternions])
        if world_yaws.shape != local_yaws.shape:
            raise ValueError("pose count mismatch")
        differences = world_yaws - local_yaws
    else:
        raise ValueError("world positions and quaternions must be supplied together")
    sine, cosine = np.mean(np.sin(differences)), np.mean(np.cos(differences))
    if math.hypot(sine, cosine) < 1e-6:
        raise ValueError("ambiguous initial heading")
    yaw = math.atan2(sine, cosine)
    translation = np.mean(world_positions-local_positions.dot(rotation_z(yaw).T), axis=0)
    return yaw, translation


class SharedFrameAlignment:
    """Validated immutable-per-epoch transform with independently timed heartbeat.

    ``ready`` means valid transform. Consumers use ``is_ready(now,max_age)``
    immediately before use to also enforce heartbeat freshness. ``now`` is the
    receipt clock, normally ROS seconds in ROS nodes (monotonic by default).
    A previously valid epoch cannot be revived after explicit invalidation.
    """

    def __init__(self, expected_world=None, expected_local="fcu_local"):
        self.expected_world = expected_world
        self.expected_local = expected_local
        self.world_frame = None
        self.local_frame = None
        self.epoch = None
        self.valid = False
        self.reason = "waiting_for_alignment"
        self.yaw = 0.0
        self.translation = np.zeros(3, dtype=np.float64)
        self.stamp = None
        self.epoch_started_at = None
        self.changed_epoch = False
        self._received = None
        self._shared_stamp_clock = False
        self._ever_valid = False
        self._closed_epochs = set()

    @property
    def ready(self):
        return bool(self.valid and self.epoch is not None)

    @property
    def valid_from(self):
        return self.epoch_started_at

    def snapshot(self):
        return copy.deepcopy(self)

    def age(self, now=None):
        if self._received is None:
            return float("inf")
        now = time.monotonic() if now is None else float(now)
        elapsed = now-self._received
        return elapsed if elapsed >= 0.0 else float("inf")

    def is_ready(self, now=None, max_age=0.5):
        now = time.monotonic() if now is None else float(now)
        if not self.ready or self.age(now) > float(max_age):
            return False
        # Explicit receipt clocks must use the status stamp's time base (ROS in
        # ROS consumers). This also rejects a stale latched valid message when
        # the owner has died and a new subscriber has only just received it.
        if self._shared_stamp_clock:
            return self.stamp is not None and 0.0 <= now-self.stamp <= float(max_age)
        return True

    def _invalidate(self, reason):
        if self._ever_valid and self.epoch is not None:
            self._closed_epochs.add(self.epoch)
        self.valid = False
        self.reason = reason

    def update_status(self, data, now=None):
        """Read std_msgs/String.data or a dict; return whether epoch changed.

        Malformed messages invalidate instead of leaving a formerly valid TF
        usable. Repeated initial ``valid:false`` messages may become valid once;
        after a valid epoch is invalidated only a new epoch can become ready.
        """
        self.changed_epoch = False
        shared_stamp_clock = now is not None
        now = time.monotonic() if now is None else float(now)
        try:
            status = json.loads(data) if isinstance(data, str) else data
            if not isinstance(status, dict):
                raise ValueError("status is not an object")
            epoch = status["epoch"]
            valid = status["valid"]
            world, local = status["world_frame"], status["local_frame"]
            yaw, stamp = float(status["yaw"]), float(status["stamp"])
            translation = np.asarray(status["translation"], dtype=np.float64)
            started = status.get("valid_from", status.get("epoch_started_at"))
            started = float(started) if started is not None else None
            if not isinstance(epoch, str) or not epoch or type(valid) is not bool:
                raise ValueError("invalid epoch or validity")
            if not isinstance(world, str) or not isinstance(local, str) or not world or not local or world == local:
                raise ValueError("invalid frame names")
            if self.expected_world is not None and world != self.expected_world:
                raise ValueError("unexpected world frame")
            if self.expected_local is not None and local != self.expected_local:
                raise ValueError("unexpected local frame")
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError("invalid translation")
            if not all(math.isfinite(x) for x in (yaw, stamp, now)):
                raise ValueError("invalid timestamp or yaw")
            if started is not None and (not math.isfinite(started) or started > stamp):
                raise ValueError("invalid epoch start")
            if valid and started is None:
                raise ValueError("valid alignment requires valid_from")
            if epoch in self._closed_epochs and (valid or epoch != self.epoch):
                raise ValueError("closed epoch")
            if epoch == self.epoch and self._ever_valid:
                if yaw != self.yaw or not np.array_equal(translation, self.translation):
                    raise ValueError("transform changed within epoch")
                if world != self.world_frame or local != self.local_frame:
                    raise ValueError("frames changed within epoch")
                if self.stamp is not None and stamp < self.stamp:
                    raise ValueError("alignment clock moved backwards")
                if self.epoch_started_at is not None and started != self.epoch_started_at:
                    raise ValueError("epoch start changed")
            changed = epoch != self.epoch
            if changed:
                if self._ever_valid and self.epoch is not None:
                    self._closed_epochs.add(self.epoch)
                self._ever_valid = False
            self.changed_epoch = changed
            self.epoch, self.world_frame, self.local_frame = epoch, world, local
            self.yaw, self.translation = yaw, translation.copy()
            self.stamp, self.epoch_started_at, self._received = stamp, started, now
            self._shared_stamp_clock = shared_stamp_clock
            self.valid = valid
            self.reason = str(status.get("reason", "ready" if valid else "invalid"))
            if valid:
                self._ever_valid = True
            elif self._ever_valid:
                self._closed_epochs.add(epoch)
            return changed
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._invalidate("invalid_alignment_status: " + str(exc))
            return False

    def _rotate(self, vector, inverse=False):
        if not self.ready:
            raise RuntimeError("frame alignment unavailable: " + self.reason)
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim == 0 or vector.shape[-1] not in (2, 3):
            raise ValueError("vector must have last dimension 2 or 3")
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector must be finite")
        rotation = rotation_z(-self.yaw if inverse else self.yaw)
        n = vector.shape[-1]
        return vector.dot(rotation[:n, :n].T)

    def rotate_to_world(self, vector):
        return self._rotate(vector)

    def rotate_to_local(self, vector):
        return self._rotate(vector, inverse=True)

    def local_to_world(self, position, yaw=None):
        position = np.asarray(position, dtype=np.float64)
        result = self.rotate_to_world(position) + self.translation[:position.shape[-1]]
        return result if yaw is None else (result, float(wrap_angle(yaw+self.yaw)))

    def world_to_local(self, position, yaw=None):
        position = np.asarray(position, dtype=np.float64)
        result = self.rotate_to_local(position-self.translation[:position.shape[-1]])
        return result if yaw is None else (result, float(wrap_angle(yaw-self.yaw)))

    def orientation_to_world(self, quaternion):
        if not self.ready:
            raise RuntimeError("frame alignment unavailable: " + self.reason)
        return quaternion_multiply(yaw_quaternion(self.yaw), quaternion)

    def orientation_to_local(self, quaternion):
        if not self.ready:
            raise RuntimeError("frame alignment unavailable: " + self.reason)
        return quaternion_multiply(yaw_quaternion(-self.yaw), quaternion)
