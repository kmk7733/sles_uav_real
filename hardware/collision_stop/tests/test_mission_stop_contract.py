"""Pure stop-policy tests; numbers are fixtures, not aircraft calibration."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "offboard_flight/scripts/mission_stop_contract.py"
SPEC = importlib.util.spec_from_file_location("stop_contract_test", PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def status(**kwargs):
    doc = dict(schema=1, session_id="test-run", stamp=100.0, seq=1,
               mode="enforce", state="NOMINAL", ready=True,
               intervention_id="", reason="ready", failure=False)
    doc.update(kwargs)
    return json.dumps(doc)


def executor():
    return contract.MissionHoldExecutor("settled_velocity", 1.0, 2,
        speed_threshold=0.2, settle_seconds=0.3, velocity_timeout=0.2)


def held():
    obj = executor()
    assert obj.stop(dict(intervention_id="collision-a", reason="gt_collision", failure=True))
    assert obj.capture((1.0, 2.0, 3.0, 0.4), 100.0)
    obj.emitted_hold(100.0)
    return obj


def settle(obj, start=100.0):
    for offset in (0.0, 0.1, 0.2, 0.31):
        t = start + offset
        result = obj.should_request_land(t, (0.0, 0.0, 0.0), t)
    return result


@pytest.mark.parametrize("override", [
    dict(mode="shadow"), dict(session_id="wrong"), dict(stamp=99.0),
    dict(stamp=100.2), dict(stamp=float("nan")), dict(seq=True),
    dict(schema=True), dict(ready=1), dict(failure=1), dict(state="unknown"),
    dict(state="STOP", intervention_id=""), dict(reason=None),
])
def test_bad_guard_status_cannot_enable_start(override):
    guard = contract.GuardSession("test-run", 0.5, 0.01)
    assert not guard.update(status(**override), 100.0)
    assert not guard.ready(100.0)


def test_invalid_or_replayed_guard_invalidates_prior_ready():
    guard = contract.GuardSession("test-run", 0.5, 0.01)
    assert guard.update(status(), 100.0)
    assert guard.ready(100.0)
    assert not guard.update(status(seq=1), 100.1)
    assert not guard.current(100.1)
    assert guard.update(status(seq=2, stamp=100.1), 100.1)
    assert guard.ready(100.1)
    assert not guard.current(100.7)
    assert not guard.current(99.9)


def test_guard_stop_is_current_but_never_ready():
    guard = contract.GuardSession("test-run", 0.5, 0.01)
    assert guard.update(status(state="STOP", intervention_id="stop-a", ready=False), 100)
    assert guard.current(100)
    assert not guard.ready(100)


def test_config_requires_policy_and_explicit_numeric_contract():
    with pytest.raises(ValueError):
        contract.MissionHoldExecutor(None, 1.0, 2)
    with pytest.raises(ValueError):
        contract.MissionHoldExecutor("settled_velocity", 1.0, 2)
    with pytest.raises(ValueError):
        contract.MissionHoldExecutor("fixed_hold", 1.0, 2)


def test_stop_latches_once_and_capture_does_not_chase_vehicle():
    obj = held()
    assert not obj.stop(dict(intervention_id="second", reason="other"))
    assert not obj.capture((8, 9, 10, 1), 100.2)
    assert obj.anchor == (1.0, 2.0, 3.0, 0.4)
    assert obj.event["intervention_id"] == "collision-a"


def test_no_hold_no_landing_and_prehold_low_velocity_cannot_count():
    obj = executor()
    assert not obj.should_request_land(100, (0, 0, 0), 100)
    obj = held()
    assert not obj.should_request_land(100.1, (0, 0, 0), 99.99)
    assert obj.settled_since is None
    assert settle(obj, 100.1)


def test_repeated_single_velocity_sample_never_proves_dwell():
    obj = held()
    obj.velocity_timeout = 2.0
    for now in (100.0, 100.1, 100.3, 100.8):
        assert not obj.should_request_land(now, (0, 0, 0), 100.0)


def test_vertical_motion_and_stale_samples_break_settle():
    obj = held()
    assert settle(obj)
    assert not obj.should_request_land(100.4, (0, 0, 0.3), 100.4)
    assert obj.settled_since is None
    assert not obj.should_request_land(100.5, (0, 0, 0), 100.0)
    assert settle(obj, 100.6)


def test_high_velocity_between_loop_ticks_breaks_dwell_even_during_retry():
    obj = held()
    assert settle(obj)
    obj.requested_land(100.31)
    obj.observe_velocity((1, 0, 0), 100.4, 100.4)
    assert obj.settled_since is None
    obj.observe_velocity((0, 0, 0), 100.41, 100.41)
    assert obj.settled_since == 100.41
    assert not obj.should_request_land(100.5, (0, 0, 0), 100.5)


def test_missing_velocity_gap_breaks_dwell():
    obj = held()
    assert not obj.should_request_land(100, (0, 0, 0), 100)
    assert not obj.should_request_land(100.1, (0, 0, 0), 100.1)
    assert not obj.should_request_land(100.5, (0, 0, 0), 100.5)
    assert obj.settled_since == 100.5


def test_request_does_not_confirm_mode_and_retries_are_bounded():
    obj = held()
    assert settle(obj)
    obj.requested_land(100.31)
    assert obj.phase == "AUTO_LAND_REQUESTED"
    assert not obj.should_request_land(100.4, (0, 0, 0), 100.4)
    for t in (100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.32):
        eligible = obj.should_request_land(t, (0, 0, 0), t)
    assert eligible
    obj.requested_land(101.32)
    assert not obj.should_request_land(103, (0, 0, 0), 103)
    assert obj.anchor is not None


def test_auto_land_confirmed_only_from_fcu_and_never_reclaims_authority():
    obj = held()
    obj.requested_land(100)
    obj.observe_mode(True, "AUTO.LAND", True, True)
    assert obj.phase == "AUTO_LAND" and obj.anchor is None
    obj.observe_mode(True, "OFFBOARD", True, True)
    assert obj.phase == "PILOT"
    obj.observe_mode(True, "AUTO.LAND", True, True)
    assert obj.phase == "PILOT"


@pytest.mark.parametrize("mode", ["POSCTL", "AUTO.RTL", "AUTO.LAND", "MANUAL"])
def test_unsolicited_mode_change_yields_to_pilot_or_failsafe(mode):
    obj = held()
    obj.observe_mode(True, mode, True, True)
    assert obj.phase == "PILOT" and obj.anchor is None


def test_disarm_precedes_mode_change_without_issuing_disarm():
    obj = held()
    obj.observe_mode(False, "MANUAL", True, True)
    assert obj.phase == "DONE"


def test_reset_retires_anchor_dwell_and_queued_worker_generation():
    obj = held()
    assert settle(obj)
    old_generation = obj.generation
    obj.reset(101.0)
    assert obj.phase == "AWAIT_LOCAL_POSE" and obj.anchor is None
    assert obj.generation != old_generation
    assert obj.settled_since is None
    assert not obj.capture((1, 2, 3, 0), 100.9)
    assert obj.capture((8, 9, 3, 0), 101.1)
    assert not obj.should_request_land(101.1, (0, 0, 0), 101.1)
    obj.emitted_hold(101.1)
    assert settle(obj, 101.1)


def test_mission_failsafe_event_is_not_forged_collision():
    event = contract.MissionFailsafe.event("guard_timeout", "run")
    assert event["source"] == "mission_failsafe"
    assert event["failure"] is True
    assert event["reason"] == "guard_timeout"


def test_takeoff_ready_is_separate_from_nominal_ready():
    guard = contract.GuardSession("test-run", 0.5, 0.01)
    assert guard.update(status(state="STANDBY", ready=False, takeoff_ready=True), 100.0)
    assert guard.takeoff_ready(100.0) and not guard.ready(100.0)
    assert guard.update(status(seq=2, stamp=100.1, state="STANDBY", ready=False), 100.1)
    assert not guard.takeoff_ready(100.1)          # older guard without the field never enables takeoff
    assert guard.update(status(seq=3, stamp=100.2, state="STOP", intervention_id="s",
                               ready=False, takeoff_ready=True), 100.2)
    assert not guard.takeoff_ready(100.2)
