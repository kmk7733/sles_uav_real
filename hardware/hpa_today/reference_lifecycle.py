"""Reference ownership and sampling; no ROS/FCU interface.

Input age is checked when accepting a newly computed result. An accepted
trajectory keeps its original depth timestamp. With reference_max_duration
configured it is sampled for exactly that long from the anchor, clamping past
its last node (simulator ReferenceHolder stale_after), whatever its node count
(HPA 11 nodes, HAA N+1, DeSimplex recovery/bridge); otherwise until its horizon. Callers must
check current sensor/FCU health before publishing each sample; this class does
not extend any controller command timeout. Sampling reuses the simulator class.
Previous acceleration keeps the existing planner's first-node convention.
"""
import math
import threading
import numpy as np
from planner.types import PlanarReferenceSequence


class ReferenceLifecycle:
    def __init__(self, input_max_age=1.0, future_tolerance=.01,
                 reference_max_duration=None, *, max_age=None):
        # Compatibility for older diagnostics: max_age ONLY limits acceptance.
        if max_age is not None:
            input_max_age = max_age
        if not math.isfinite(input_max_age) or input_max_age <= 0:
            raise ValueError("positive input_max_age required")
        if not math.isfinite(future_tolerance) or future_tolerance < 0:
            raise ValueError("nonnegative future_tolerance required")
        if reference_max_duration is not None and (
                not math.isfinite(reference_max_duration) or reference_max_duration <= 0):
            raise ValueError("positive reference_max_duration required when configured")
        self.input_max_age = input_max_age
        self.reference_max_duration = reference_max_duration
        self.future_tolerance = future_tolerance
        self.lock = threading.RLock()
        self.active = None
        self.last_anchor = None
        self.last_clock = None
        self.reason = "no accepted reference"
        self.retired = False

    def invalidate(self, reason):
        with self.lock:
            self.active = None
            self.reason = str(reason)
            self.retired = True

    def accept(self, event, now):
        with self.lock:
            if self.retired:
                return False
            if not event.get("accepted"):
                return False
            t0, epoch = float(event["anchor_stamp"]), int(event["estimator_epoch"])
            if (not math.isfinite(t0) or not math.isfinite(now) or t0 <= 0
                    or now-t0 > self.input_max_age or now-t0 < -self.future_tolerance):
                self.reason = "result stale/future at reference acceptance"
                return False
            if self.last_anchor is not None and t0 <= self.last_anchor:
                self.reason = "non-increasing reference anchor"
                return False
            raw = event["integrated_reference"]
            if raw["frame"] != "PX4 local ENU" or raw["anchor_stamp"] != t0:
                raise ValueError("reference frame/anchor mismatch")
            dt = float(raw.get("node_dt_s", raw.get("dt", 0.)))
            if dt != .1:
                raise ValueError("V4 reference interval must be 0.1s")
            fields = {key: np.array(raw[key], dtype=float, copy=True)
                      for key in ("p", "v", "a", "psi", "psi_dot")}
            nodes = fields["p"].shape[0] if fields["p"].ndim == 2 else 0
            if nodes < 2:
                raise ValueError("invalid reference p")
            for key, arr in fields.items():
                expected = (nodes, 2) if key in ("p", "v", "a") else (nodes,)
                if arr.shape != expected or not np.isfinite(arr).all():
                    raise ValueError("invalid reference " + key)
                arr.setflags(write=False)
            ref = PlanarReferenceSequence(dt=dt, **fields)
            goal = np.asarray(event["goal_local_enu"], dtype=float)
            if goal.shape != (3,) or not np.isfinite(goal).all():
                raise ValueError("invalid snapshot goal")
            if self.active and (epoch != self.active["epoch"] or not np.array_equal(goal, self.active["goal"])):
                self.invalidate("epoch/goal changed; new session required")
                return False
            duration = ref.duration if self.reference_max_duration is None else self.reference_max_duration
            if now-t0 >= duration:
                self.reason = "reference horizon expired at acceptance"
                return False
            self.active = dict(ref=ref, t0=t0, accepted_at=now, duration=duration,
                               epoch=epoch, goal=goal.copy())
            self.last_anchor = t0
            self.reason = None
            return True

    def previous_acceleration(self, anchor, epoch):
        # Simulator harness and planar_planner_node keep the last accepted
        # plan's a[0] until a reset, however old; no expiry here.
        with self.lock:
            a = self.active
            if not self.retired and a and a["epoch"] == epoch and anchor >= a["t0"]:
                return a["ref"].a[0].copy(), "last accepted reference.a[0]; simulator/planar_planner_node convention"
            return np.zeros(2), "zero before the first accepted reference or after a reset"

    def sample(self, now, epoch):
        with self.lock:
            if not math.isfinite(now):
                self.invalidate("nonfinite sampling clock")
                return None
            if self.last_clock is not None and now < self.last_clock:
                self.invalidate("sampling clock reversal")
                return None
            self.last_clock = now
            a = self.active
            if self.retired or a is None:
                return None
            if a["epoch"] != epoch:
                self.invalidate("estimator epoch changed")
                return None
            age = now-a["t0"]
            if age < 0 or age >= a["duration"]:
                self.reason = "reference not started/expired"
                return None
            point = a["ref"].sample(age)
            self.reason = None
            return dict(anchor_stamp=a["t0"], sample_stamp=now, reference_age_s=age,
                        accepted_at=a["accepted_at"], reference_end_stamp=a["t0"]+a["duration"],
                        estimator_epoch=epoch, frame="PX4 local ENU", goal_local_enu=a["goal"].tolist(),
                        **{key: getattr(point, key).tolist() for key in ("p", "v", "a")},
                        psi=point.psi, psi_dot=point.psi_dot,
                        purpose="planar reference shadow preview only; no controller/FCU command")


def warmup_runtime(runtime, iterations=3):
    """Synthetic initialization only. Every output is discarded before readiness."""
    import time
    if iterations < 1:
        raise ValueError("warmup must precede readiness")
    scan = np.ones((2, 512), dtype=np.float32)
    state = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = runtime.infer(scan, state)
        if output.shape != (10, 3) or not np.isfinite(output).all():
            raise ValueError("warmup output contract failed")
        times.append(1000*(time.perf_counter()-start))
    return dict(iterations=iterations, timing_ms=times,
                input="synthetic initialization only", outputs_discarded=True)
