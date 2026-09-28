"""Planar reference to the existing controller serializer; no ROS lifecycle.

The reference is already in PX4 local ENU. Goal arrival reuses the existing
planar_planner_node hold (latched position, zero velocity/acceleration/yaw-rate).
Only the existing fixed-altitude
lift and positional leash are applied here; HPA acceleration is never treated
as a velocity command and no world/local transform is applied a second time.
"""
import math

import numpy as np


def reference_values(sample, z_local, current_xy, max_step):
    """Return Controller.construct_target_full arguments without mutating input."""
    if sample.get("frame") != "PX4 local ENU":
        raise ValueError("reference must already be PX4 local ENU")
    stamp = float(sample["sample_stamp"])
    anchor = float(sample["anchor_stamp"])
    end = float(sample["reference_end_stamp"])
    if not all(math.isfinite(v) for v in (stamp, anchor, end, z_local, max_step)):
        raise ValueError("nonfinite reference time or altitude/leash")
    if not 0 < anchor <= stamp < end or max_step <= 0:
        raise ValueError("reference expired/not started or invalid leash")
    fields = {}
    for name in ("p", "v", "a"):
        value = np.asarray(sample[name], dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("invalid planar reference " + name)
        fields[name] = value.copy()
    current = np.asarray(current_xy, dtype=float)
    if current.shape != (2,) or not np.isfinite(current).all():
        raise ValueError("invalid current PX4 position")
    yaw, yaw_rate = float(sample["psi"]), float(sample["psi_dot"])
    if not math.isfinite(yaw) or not math.isfinite(yaw_rate):
        raise ValueError("nonfinite reference yaw/yaw-rate")
    delta = fields["p"] - current
    distance = float(np.linalg.norm(delta))
    if distance > max_step:
        fields["p"] = current + delta * (max_step / distance)
    return (np.r_[fields["p"], float(z_local)], np.r_[fields["v"], 0.],
            np.r_[fields["a"], 0.], yaw, yaw_rate)


def make_target(controller, sample, z_local, current_xy, max_step, stamp, frame_id):
    """Reuse the real controller's full position/velocity/acceleration serializer."""
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("explicit output frame required")
    values = reference_values(sample, z_local, current_xy, max_step)
    target = controller.construct_target_full(*values)
    if target.coordinate_frame != 1 or target.type_mask != 0:
        raise ValueError("controller serializer changed local ENU acceleration contract")
    target.header.stamp = stamp
    target.header.frame_id = frame_id
    return target


def hold_values(hold_xy, hold_yaw, z_local, current_xy, max_step):
    """Arrival hold as in planar_planner_node: latched p, zero v/a/yaw-rate."""
    hold = np.asarray(hold_xy, dtype=float)
    current = np.asarray(current_xy, dtype=float)
    if hold.shape != (2,) or current.shape != (2,) or not np.isfinite(hold).all() or not np.isfinite(current).all():
        raise ValueError("invalid hold or current PX4 position")
    if not all(math.isfinite(v) for v in (hold_yaw, z_local, max_step)) or max_step <= 0:
        raise ValueError("nonfinite hold yaw/altitude or invalid leash")
    delta = hold - current
    distance = float(np.linalg.norm(delta))
    if distance > max_step:
        hold = current + delta * (max_step / distance)
    return np.r_[hold, float(z_local)], np.zeros(3), np.zeros(3), float(hold_yaw), 0.


def make_hold_target(controller, hold_xy, hold_yaw, z_local, current_xy, max_step, stamp, frame_id):
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("explicit output frame required")
    target = controller.construct_target_full(*hold_values(hold_xy, hold_yaw, z_local, current_xy, max_step))
    if target.coordinate_frame != 1 or target.type_mask != 0:
        raise ValueError("controller serializer changed local ENU acceleration contract")
    target.header.stamp = stamp
    target.header.frame_id = frame_id
    return target


def validate_arrived_topic(topic, controller_output):
    """Controller mode feeds the mission's goal-arrived input; preview never does."""
    if not isinstance(topic, str) or not topic.startswith("/"):
        raise ValueError("resolved absolute goal-arrived topic required")
    if "mavros" in topic.split("/"):
        raise ValueError("goal-arrived topic cannot be an FCU topic")
    if not controller_output and not (topic.startswith("/hpa_shadow/") or topic.startswith("/hpa_validation/")):
        raise ValueError("preview goal-arrived topic must remain under /hpa_shadow/ or /hpa_validation/")
    return topic


def check_arrived_graph(system_state, arrived_topic, own_node=None):
    """A second goal-arrived writer (e.g. planar_planner_node) could land the mission."""
    publishers, _, _ = system_state
    if any(writer != own_node for writer in dict(publishers).get(arrived_topic, [])):
        raise ValueError("existing goal-arrived publisher: " + arrived_topic)


def validate_output_topic(topic, controller_output):
    """Preview destinations cannot be mistaken for a commander/FCU channel."""
    if not isinstance(topic, str) or not topic.startswith("/"):
        raise ValueError("resolved absolute output topic required")
    parts = topic.split("/")
    if "mavros" in parts or topic.endswith("/setpoint_raw/local"):
        raise ValueError("HPA node cannot publish to FCU topics")
    if controller_output:
        if not topic.endswith("/commander/set_pose"):
            raise ValueError("controller output must use nominal commander/set_pose")
    elif not (topic.startswith("/hpa_shadow/") or topic.startswith("/hpa_validation/")):
        raise ValueError("preview output must remain under /hpa_shadow/ or /hpa_validation/")
    return topic


def check_output_graph(system_state, output_topic, controller_output, guard_node,
                       own_node=None, observer_nodes=()):
    """Refuse another writer or an unapproved nominal subscriber at startup.

    observer_nodes is an explicit operator allowlist of passive recorders or
    diagnostics only. ROS graph metadata cannot prove that a subscriber is
    passive; never put a mission executor or command relay in this list.
    """
    publishers, subscribers, _ = system_state
    writers = dict(publishers).get(output_topic, [])
    if any(writer != own_node for writer in writers):
        raise ValueError("existing output publisher: " + output_topic)
    if controller_output:
        consumers = set(dict(subscribers).get(output_topic, []))
        allowed = {guard_node} | set(observer_nodes)
        if guard_node not in consumers or consumers - allowed:
            raise ValueError("nominal output requires CollisionStopGuard and only explicitly approved passive observers")
