"""Offline contract checks. Numerical fixtures are NOT a flight profile."""
import copy
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "offboard_flight" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from collision_stop_core import CollisionStopCore, segment_rect_distance, point_rect_distance


@pytest.fixture
def profile():
    return {
        "schema": 1, "enabled": True, "session_id": "offline-test-only", "mode": "shadow",
        "vicon_frame_id": "vicon/world", "map_path": "/fixture/map.yaml", "log_path": "/fixture/test.jsonl",
        "vehicle": {"topic": "/vicon/drone/drone", "center_offset_subject_m": [0, 0, 0], "radius_m": 0.2},
        "obstacle_topics": {"wall": "/vicon/wall/wall"},
        "topics": {"nominal": "commander/set_pose", "safe": "commander/set_pose_safe",
                   "status": "commander/collision_stop_status", "execution": "commander/collision_stop_execution"},
        "nominal": {"coordinate_frame": 1, "frame_id": "fcu_local", "frame_policy": "epoch_tagged"},
        "limits": {"rate_hz": 20, "vicon_max_age_s": 0.3, "max_future_s": 0.01,
                   "execution_max_age_s": 0.3, "nominal_max_age_s": 0.3,
                   "velocity_max_gap_s": 0.2, "max_subject_skew_s": 0.1,
                   "max_vicon_speed_m_s": 10, "max_obstacle_yaw_rate_rad_s": 5},
        "collision": {"horizon_s": 0.5, "stop_margin_m": 0.1},
    }


@pytest.fixture
def map_doc():
    return {"pillars": [{"name": "wall", "center": [2, 0], "size": [0.4, 1], "yaw": 0,
                         "body_position": [2, 0, 0], "body_yaw": 0}]}


def execution(stamp=1.0, state="MISSION", seq=1):
    return {"schema": 1, "session_id": "offline-test-only", "stamp": stamp, "seq": seq, "state": state}


def nominal(stamp=1.0):
    return {"stamp": stamp, "frame_id": "fcu_local/epoch/4ac73cef16bb491a9d6f8925bd0472ee",
            "coordinate_frame": 1, "type_mask": 0, "fields": [0.0] * 11}


def healthy(core, x=0.0, vx=0.0, state="MISSION"):
    for stamp, pos in ((0.9, x - vx * 0.1), (1.0, x)):
        core.update_vicon("vehicle", stamp, "vicon/world", [pos, 0, 1], [0, 0, 0, 1])
        core.update_vicon("wall", stamp, "vicon/world", [2, 0, 0], [0, 0, 0, 1])
    core.update_execution(execution(state=state))
    core.update_nominal(nominal())


def test_tunneling_exact_segment_and_rotated_box():
    assert segment_rect_distance((-10, 0), (10, 0), (0, 0), (0.001, 0.1), 0.3) == 0
    assert segment_rect_distance((-2, 2), (2, 2), (0, 0), (2, 2), 0) == pytest.approx(1)
    assert segment_rect_distance((2, 2), (2, 2), (0, 0), (2, 2), 0) == pytest.approx(math.sqrt(2))
    assert point_rect_distance((0, 0), (0, 0), (2, 4), 0) == -1


def test_nominal_and_startup_wait_are_ready_without_px4(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, state="WAIT")
    result = core.evaluate(1.0)
    assert result["state"] == "STANDBY" and result["ready"] and result["allow_nominal"]
    assert result["risk_source"] == "vicon_gt_only"
    assert core.update_execution(execution(1.01, "MISSION", 2))
    result = core.evaluate(1.01)
    assert result["state"] == "NOMINAL" and result["failure"] is False
    assert result["role"] == "CollisionStopGuard"
    json.dumps(result, allow_nan=False)


def test_collision_latches_failure_and_never_resumes(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, x=1.3, vx=1)
    first = core.evaluate(1.0)
    assert first["state"] == "STOP" and first["cause"] == "collision" and first["failure"]
    assert first["intervention_id"] and not first["allow_nominal"]
    core.update_vicon("vehicle", 1.1, "vicon/world", [0.4, 0, 1], [0, 0, 0, 1])
    later = core.evaluate(1.1)
    assert later["state"] == "STOP" and later["intervention_id"] == first["intervention_id"]
    assert later["reason"] == first["reason"]


def test_unsafe_initial_placement_blocks_start_without_fake_failure(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, x=1.7, state="WAIT")
    result = core.evaluate(1.0)
    assert result["state"] == "STANDBY" and not result["ready"] and not result["failure"]


@pytest.mark.parametrize("subject", ["vehicle", "wall"])
def test_every_vicon_subject_must_be_fresh(profile, map_doc, subject):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.samples[subject]["stamp"] = 0.5
    result = core.evaluate(1.0)
    assert result["state"] == "STOP" and result["cause"] == "input_fault"
    assert "stale" in result["reason"]


@pytest.mark.parametrize("stamp,frame,position,quaternion,reason", [
    (0.99, "vicon/world", [0, 0, 1], [0, 0, 0, 1], "timestamp_not_increasing"),
    (1.01, "wrong", [0, 0, 1], [0, 0, 0, 1], "frame_mismatch"),
    (1.01, "vicon/world", [float("nan"), 0, 1], [0, 0, 0, 1], "Vicon position"),
    (1.01, "vicon/world", [0, 0, 1], [0, 0, 0, 0], "unit quaternion"),
    (1.01, "vicon/world", [100, 0, 1], [0, 0, 0, 1], "position_jump"),
])
def test_bad_vicon_latches_immediately(profile, map_doc, stamp, frame, position, quaternion, reason):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    assert not core.update_vicon("vehicle", stamp, frame, position, quaternion)
    result = core.evaluate(1.02)
    assert result["state"] == "STOP" and reason in result["reason"]


def test_backward_ros_clock_blocks(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.evaluate(1.0)
    assert core.evaluate(0.999)["reason"] == "ros_time_backwards"


def test_future_vicon_and_inter_subject_skew_block(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.samples["wall"]["stamp"] = 1.02
    assert "future_stamp" in core.evaluate(1.0)["reason"]
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.samples["wall"]["stamp"] = 0.89
    assert core.evaluate(1.0)["reason"] == "vicon_subject_timestamp_skew"


def test_velocity_gap_needs_two_new_samples_not_coasting(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, state="WAIT")
    core.update_vicon("vehicle", 1.25, "vicon/world", [0, 0, 1], [0, 0, 0, 1])
    result = core.evaluate(1.25)
    assert not result["ready"] and "velocity_unready" in result["reason"]


def test_relative_obstacle_velocity_can_trigger_for_stationary_drone(profile, map_doc):
    profile["collision"]["horizon_s"] = 1.0
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.samples["wall"]["center"] = (1.2, 0, 0)
    core.samples["wall"]["velocity"] = (-1, 0)
    result = core.evaluate(1.0)
    assert result["cause"] == "collision"
    assert result["risks"][0]["obstacle_velocity_vicon_xy"] == [-1, 0]


def test_age_padding_prevents_stale_valid_samples_shortening_horizon(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, x=0.0, vx=1.0)
    first = core.evaluate(1.0)["risks"][0]
    second = core.evaluate(1.1)["risks"][0]
    assert second["sample_age_padding_m"] == pytest.approx(0.1)
    assert second["projected_margin_m"] == pytest.approx(first["projected_margin_m"] - 0.1)
    assert second["effective_lookahead_from_oldest_stamp_s"] == pytest.approx(0.6)


def test_rigid_body_offset_rotates_fitted_rectangle(profile, map_doc):
    # Capture subject (1,0), fitted box (2,0): offset must rotate with subject.
    map_doc["pillars"][0]["body_position"] = [1, 0, 0]
    core = CollisionStopCore(profile, map_doc)
    q = [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    core.update_vicon("wall", 1.0, "vicon/world", [5, 6, 0], q)
    assert core.samples["wall"]["center"] == pytest.approx((5, 7, 0))
    assert core.samples["wall"]["yaw"] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("fault", ["missing", "wrong_session", "replay", "restart", "stale"])
def test_executor_protocol_failure_is_terminal_when_active(profile, map_doc, fault):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    value = execution(1.01, "MISSION", 2)
    if fault == "missing":
        value = None
    elif fault == "wrong_session":
        value["session_id"] = "another-run"
    elif fault == "replay":
        value["seq"] = 1
    elif fault == "restart":
        value["state"] = "WAIT"
    if fault != "stale":
        core.update_execution(value)
    result = core.evaluate(1.31 if fault == "stale" else 1.01)
    assert result["state"] == "STOP" and result["cause"] == "input_fault"


@pytest.mark.parametrize("state", ["PILOT", "DONE", "AUTO_LAND", "LAND", "DISARM"])
def test_terminal_mission_states_disable_forwarding(profile, map_doc, state):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.update_execution(execution(1.01, state, 2))
    result = core.evaluate(1.01)
    assert not result["allow_nominal"]
    assert result["state"] != "STOP"
    core.update_vicon("vehicle", 0.99, "vicon/world", [0, 0, 1], [0, 0, 0, 1])
    core.update_nominal({})
    assert core.evaluate(1.02)["failure"] is False


def test_shadow_without_executor_still_logs_vicon_geometry(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    for stamp in (0.9, 1.0):
        core.update_vicon("vehicle", stamp, "vicon/world", [1.7, 0, 1], [0, 0, 0, 1])
        core.update_vicon("wall", stamp, "vicon/world", [2, 0, 0], [0, 0, 0, 1])
    result = core.evaluate(1.0)
    assert result["execution"] is None and result["risks"] and result["would_stop"]
    assert result["state"] == "STANDBY" and not result["ready"]
    assert not result["failure"] and not result["allow_nominal"]


def test_execution_ack_and_hold_anchor_survive_logging(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    value = execution(1.01, "STOP_HOLD", 2)
    value.update({"intervention_id": "collision-ack", "phase": "STOP_HOLD",
                  "hold_anchor_local": [1, 2, 3, 0.1], "land_attempts": 0,
                  "observed_mode": "OFFBOARD", "armed": True})
    core.update_execution(value)
    assert core.evaluate(1.01)["execution"] == value


@pytest.mark.parametrize("state", ["STOP_HOLD", "STOP_AWAIT_LOCAL_POSE", "AUTO_LAND_REQUESTED"])
def test_mission_initiated_stop_phases_never_forward_nominal(profile, map_doc, state):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    assert core.update_execution(execution(1.01, state, 2))
    result = core.evaluate(1.01)
    assert not result["allow_nominal"] and not result["ready"]


def test_clock_reset_after_done_does_not_rewrite_trial_as_failure(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    core.evaluate(1.0)
    core.update_execution(execution(1.01, "DONE", 2))
    result = core.evaluate(0.99)
    assert not result["failure"] and not result["allow_nominal"]


@pytest.mark.parametrize("change", [
    {"frame_id": "fcu_local/epoch/"}, {"coordinate_frame": 8},
    {"fields": [float("inf")] * 11}, {"stamp": 0.9}, {"type_mask": 512},
])
def test_invalid_nominal_never_passes(profile, map_doc, change):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    value = nominal(1.01)
    value.update(change)
    assert not core.update_nominal(value)
    assert core.evaluate(1.01)["state"] == "STOP"


def test_epoch_suffix_and_all_nominal_fields_preserved(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core)
    value = nominal(1.01)
    value["frame_id"] = "fcu_local/epoch/epoch-a"
    value["fields"][0] = 3.1
    assert core.update_nominal(value)
    assert core.evaluate(1.01)["nominal"] == value


@pytest.mark.parametrize("fault", ["empty", "missing_geometry", "missing_capture", "unmapped", "reserved", "not_object"])
def test_map_fails_closed_no_silent_omission(profile, map_doc, fault):
    if fault == "empty":
        map_doc["pillars"] = []
    elif fault == "missing_geometry":
        del map_doc["pillars"][0]["size"]
    elif fault == "missing_capture":
        del map_doc["pillars"][0]["body_position"]
    elif fault == "unmapped":
        profile["obstacle_topics"] = {"different": "/vicon/other/other"}
    elif fault == "reserved":
        map_doc["pillars"][0]["name"] = "vehicle"
    else:
        map_doc["pillars"] = [None]
    with pytest.raises(ValueError):
        CollisionStopCore(profile, map_doc)


def test_disabled_example_cannot_run(map_doc):
    path = Path(__file__).resolve().parents[1] / "collision_stop_profile.example.json"
    with path.open() as stream:
        disabled = json.load(stream)
    with pytest.raises(ValueError, match="explicitly enable"):
        CollisionStopCore(disabled, map_doc)


def test_takeoff_ready_needs_healthy_vicon_but_no_nominal(profile, map_doc):
    """Takeoff precedes the planner: no nominal yet, but monitoring is healthy."""
    core = CollisionStopCore(profile, map_doc)
    for stamp in (0.9, 1.0):
        core.update_vicon("vehicle", stamp, "vicon/world", [0, 0, 1], [0, 0, 0, 1])
        core.update_vicon("wall", stamp, "vicon/world", [2, 0, 0], [0, 0, 0, 1])
    core.update_execution(execution(state="WAIT"))
    result = core.evaluate(1.0)
    assert result["takeoff_ready"] and not result["ready"] and not result["allow_nominal"]
    assert core.update_execution(execution(1.01, "CLIMB", 2))
    result = core.evaluate(1.01)
    assert result["state"] == "STANDBY" and result["takeoff_ready"] and not result["failure"]


def test_takeoff_ready_false_on_unsafe_placement_or_missing_vicon(profile, map_doc):
    core = CollisionStopCore(profile, map_doc)
    healthy(core, x=1.7, state="WAIT")
    assert not core.evaluate(1.0)["takeoff_ready"]
    core = CollisionStopCore(profile, map_doc)
    core.update_execution(execution(state="WAIT"))
    assert not core.evaluate(1.0)["takeoff_ready"]
