"""ROS-free wrapper integration tests. No FCU or ROS service is contacted."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Obj(object):
    def __init__(self, **values):
        self.__dict__.update(values)


class Stamp(object):
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


class Target(object):
    def __init__(self):
        self.header = Obj(stamp=Stamp(0), frame_id="")
        self.position = Obj(x=0, y=0, z=0)
        self.velocity = Obj(x=0, y=0, z=0)
        self.acceleration_or_force = Obj(x=0, y=0, z=0)
        self.yaw = self.yaw_rate = 0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def wrapper(monkeypatch):
    now = [100.0]
    ros = types.ModuleType("rospy")
    ros.get_time = lambda: now[0]
    ros.Time = Obj(now=lambda: Stamp(now[0]))
    ros.ServiceException = RuntimeError
    for name in ("loginfo", "logwarn", "logerr", "logwarn_throttle", "logerr_throttle", "loginfo_throttle"):
        setattr(ros, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "rospy", ros)
    for name, members in {
        "geometry_msgs.msg": ["PoseStamped", "TwistStamped"],
        "mavros_msgs.msg": ["PositionTarget", "State", "ExtendedState"],
        "mavros_msgs.srv": ["CommandBool", "SetMode"],
        "std_msgs.msg": ["String", "Bool"],
        "std_srvs.srv": ["Trigger", "TriggerResponse"],
        "guidance_library": ["Controller"],
    }.items():
        module = types.ModuleType(name)
        for member in members:
            setattr(module, member, Obj)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["mavros_msgs.msg"].PositionTarget = Target
    sys.modules["mavros_msgs.msg"].ExtendedState = Obj(LANDED_STATE_ON_GROUND=1)
    monkeypatch.syspath_prepend(str(ROOT / "hardware/ekf_state/common"))
    base = load("stop_test_mission_node", ROOT / "hardware/ekf_state/offboard_flight/scripts/mission_node.py")
    monkeypatch.setitem(sys.modules, "mission_node", base)
    contract = load("stop_test_contract", ROOT / "offboard_flight/scripts/mission_stop_contract.py")
    monkeypatch.setitem(sys.modules, "mission_stop_contract", contract)
    module = load("stop_test_wrapper", ROOT / "offboard_flight/scripts/guarded_mission_node.py")
    return module, now


def guard_json(seq=1, stamp=100.0, **values):
    doc = dict(schema=1, session_id="test", seq=seq, stamp=stamp,
               mode="enforce", state="NOMINAL", ready=True,
               failure=False, reason="ready", intervention_id="")
    doc.update(values)
    return Obj(data=json.dumps(doc))


def pose(stamp=100.0, x=2.0):
    return Obj(header=Obj(stamp=Stamp(stamp), frame_id="map"),
               pose=Obj(position=Obj(x=x, y=3.0, z=1.0),
                        orientation=Obj(x=0., y=0., z=0., w=1.)))


def make_node(wrapper):
    mod, now = wrapper
    node = mod.GuardedMissionNode.__new__(mod.GuardedMissionNode)
    node.lock = threading.RLock()
    node.guard = mod.GuardSession("test", 0.5, 0.01)
    node.guard.update(guard_json().data, now[0])
    node.stop_executor = mod.MissionHoldExecutor("settled_velocity", 1.0, 2,
        speed_threshold=0.2, settle_seconds=0.3, velocity_timeout=0.2)
    node.state, node.t_state = "MISSION", 99.0
    node.pose, node.t_pose = pose(), 100.0
    node.pose_timeout = node.mav_timeout = node.sp_timeout = node.alignment_timeout = 0.5
    node.mav = Obj(connected=True, armed=True, mode="OFFBOARD")
    node.mav_received = 100.0
    node.armed_once = node.offboard_once = True
    node.require_frame_alignment = True
    from ekf_alignment import SharedFrameAlignment
    node.alignment = SharedFrameAlignment(expected_world="vicon/world")
    node.alignment.update_status(json.dumps(dict(valid=True, epoch="e1", world_frame="vicon/world",
        local_frame="fcu_local", yaw=0.0, translation=[0., 0., 0.], stamp=100.0,
        valid_from=99.0, reason="ready")), now=100.0)
    node.active_epoch = node.sp_epoch = "e1"
    node._alignment_stop_pending = node._alignment_fault_active = False
    node._hold_pose_min_stamp = None
    node.velocity = node.velocity_stamp = node.velocity_received = None
    node._land_worker = None
    node._execution_seq = 0
    node._last_loop_time = None
    node.sp = node.cmd = node.frozen = None
    node.start_req = node.land_req = False
    node.t_sp = 100.0
    node.t_arrived = None
    node.commands, node.executions, node.calls = [], [], []
    node.pub = Obj(publish=node.commands.append)
    node.pub_state = Obj(publish=lambda msg: None)
    node.execution_pub = Obj(publish=lambda msg: node.executions.append(json.loads(msg.data)))
    node.set_mode = lambda **kwargs: (node.calls.append(kwargs) or Obj(mode_sent=True))
    node.home = (2., 3., 0., 0.)
    node.fence_r = 6.
    return node


def stop(node):
    node._guard_cb(guard_json(seq=2, state="STOP", ready=False,
        intervention_id="collision-1", failure=True, reason="gt_collision"))


def test_hold_has_frozen_position_zero_velocity_yaw_and_no_force(wrapper):
    node = make_node(wrapper)
    stop(node)
    node._stop_tick_locked(100.)
    out = node.commands[-1]
    assert out.coordinate_frame == 1
    assert out.header.frame_id == "fcu_local"
    assert out.type_mask == 2496
    assert out.type_mask & (1 << 9) == 0
    assert (out.position.x, out.position.y, out.position.z) == (2., 3., 1.)
    assert (out.velocity.x, out.velocity.y, out.velocity.z) == (0., 0., 0.)
    node.pose = pose(x=10.)
    node._stop_tick_locked(100.1)
    assert node.commands[-1].position.x == 2.
    assert not node.calls
    assert node.sp is None and node.t_arrived is None


def test_stale_pose_never_replays_nominal_or_old_hold(wrapper):
    node = make_node(wrapper)
    stop(node)
    node._stop_tick_locked(100.)
    assert len(node.commands) == 1
    node.mav_received = 101.
    node._stop_tick_locked(101.)
    assert len(node.commands) == 1 and node.cmd is None
    assert node.stop_executor.anchor is None


def test_disarmed_stop_does_not_arm_or_publish(wrapper):
    node = make_node(wrapper)
    node.armed_once = node.offboard_once = False
    node.mav.armed = False
    node.mav.mode = "MANUAL"
    stop(node)
    node._stop_tick_locked(100.)
    assert node.state == "DONE"
    assert not node.commands and not node.calls


def test_manual_takeover_prevents_hold_and_landing(wrapper):
    node = make_node(wrapper)
    stop(node)
    node.mav.mode = "POSCTL"
    node._stop_tick_locked(100.)
    assert node.state == "PILOT"
    assert not node.commands and not node.calls


def test_watchdog_fault_is_distinct_and_wait_heartbeat_has_no_deadlock(wrapper):
    node = make_node(wrapper)
    node.state = "WAIT"
    node.guard = wrapper[0].GuardSession("test", 0.5, 0.01)
    node._tick_locked(100.)
    node._publish_execution_locked(100.)
    assert node.executions[-1]["state"] == "WAIT"
    assert not node.executions[-1]["ready"]
    node.state = "MISSION"
    node._tick_locked(100.)
    assert node.stop_executor.event["source"] == "mission_failsafe"
    assert node.stop_executor.event["reason"] == "guard_unavailable_or_not_ready"


@pytest.mark.parametrize("state", ["STREAM", "CLIMB", "HOLD", "LAND", "DISARM"])
def test_current_guard_without_nominal_readiness_preserves_existing_phase(wrapper, state):
    node = make_node(wrapper)
    node.state = state
    node.guard.update(guard_json(seq=2, ready=False).data, 100.)
    called = []
    setattr(node, "_" + state.lower(), lambda *args: called.append(state))
    node._tick_locked(100.)
    assert called == [state]
    assert not node.stop_executor.latched


def test_stop_ignores_subsequent_nominal_goal_arrive_and_planner_commands(wrapper):
    node = make_node(wrapper)
    stop(node)
    node._cb_sp(Target())
    node.guard.update(guard_json(seq=3).data, 100.)
    node._cb_sp(Target())
    assert node.sp is None and node.stop_executor.latched


def test_pose_reset_recaptures_only_fresh_new_epoch_pose(wrapper):
    node = make_node(wrapper)
    stop(node)
    node._stop_tick_locked(100.)
    wrapper[1][0] = 100.1
    node._cb_alignment(Obj(data=json.dumps(dict(valid=True, epoch="e2", world_frame="vicon/world",
        local_frame="fcu_local", yaw=0., translation=[0., 0., 0.], stamp=100.1,
        valid_from=100.1, reason="reset"))))
    assert node.stop_executor.anchor is None
    node._stop_tick_locked(100.1)
    assert len(node.commands) == 1
    node.pose, node.t_pose = pose(100.2, x=8.), 100.2
    node._stop_tick_locked(100.2)
    assert node.commands[-1].position.x == 8.


def test_auto_land_rpc_does_not_block_hold_or_claim_mode_confirmation(wrapper):
    mod, now = wrapper
    node = make_node(wrapper)
    stop(node)
    node._stop_tick_locked(100.)
    started, finish = threading.Event(), threading.Event()
    def blocked_service(**kwargs):
        node.calls.append(kwargs)
        started.set()
        assert finish.wait(2)
        return Obj(mode_sent=True)
    node.set_mode = blocked_service
    for t in (100., 100.1, 100.2, 100.31):
        now[0] = t
        node.velocity = (0., 0., 0.)
        node.velocity_stamp = node.velocity_received = t
        with node.lock:
            node._stop_tick_locked(t)
    assert started.wait(2)
    count = len(node.commands)
    with node.lock:
        node._stop_tick_locked(100.32)
    assert len(node.commands) == count + 1
    assert node.stop_executor.phase == "AUTO_LAND_REQUESTED"
    finish.set()
    node._land_worker.join(2)
    assert node.stop_executor.phase == "AUTO_LAND_REQUESTED"
    node._cb("mav")(Obj(connected=True, armed=True, mode="AUTO.LAND"))
    assert node.state == "AUTO_LAND"
    count = len(node.commands)
    node._stop_tick_locked(100.32)
    assert len(node.commands) == count
    assert node.calls == [dict(custom_mode="AUTO.LAND")]


def test_queued_land_worker_rechecks_pilot_before_service(wrapper):
    node = make_node(wrapper)
    stop(node)
    node._stop_tick_locked(100.)
    with node.lock:
        node._request_auto_land_async_locked(100.)
        node.mav.mode = "POSCTL"
    node._land_worker.join(2)
    assert not node.calls and node.state == "PILOT"


def test_actual_guard_executor_protocol_roundtrip_all_phases(wrapper):
    helpers = load("stop_guard_integration_helpers",
                   ROOT / "hardware/collision_stop/tests/test_collision_stop_core.py")
    profile = helpers.profile.__wrapped__()
    profile["mode"], profile["session_id"] = "enforce", "test"
    map_doc = helpers.map_doc.__wrapped__()
    def core_at(x=0., vx=0.):
        core = helpers.CollisionStopCore(profile, map_doc)
        for t, px in ((99.9, x - vx * .1), (100., x)):
            assert core.update_vicon("vehicle", t, "vicon/world", [px, 0, 1], [0, 0, 0, 1])
            assert core.update_vicon("wall", t, "vicon/world", [2, 0, 0], [0, 0, 0, 1])
        core.update_nominal(helpers.nominal(100.))
        return core
    core = core_at()
    node = make_node(wrapper)
    node.guard = wrapper[0].GuardSession("test", .5, .01)
    for i, state in enumerate(("WAIT", "STREAM", "CLIMB", "MISSION", "HOLD", "LAND", "DISARM", "DONE", "PILOT")):
        t = 100. + i * .001
        node.state = state
        node._publish_execution_locked(t)
        assert core.update_execution(node.executions[-1]), state
        result = core.evaluate(t)
        assert result["state"] != "STOP", (state, result)
        node._guard_cb(Obj(data=json.dumps(result)))
        assert node.guard.current(100.)
        assert not node.stop_executor.latched
    # A real geometric collision result must latch and ACK the exact guard id.
    core = core_at(1.3, 1.)
    node = make_node(wrapper)
    node.guard = wrapper[0].GuardSession("test", .5, .01)
    node._publish_execution_locked(100.)
    assert core.update_execution(node.executions[-1])
    result = core.evaluate(100.)
    assert result["state"] == "STOP" and result["cause"] == "collision"
    node._guard_cb(Obj(data=json.dumps(result)))
    assert node.state == "STOP_AWAIT_LOCAL_POSE"
    for i, state in enumerate(("STOP_AWAIT_LOCAL_POSE", "STOP_HOLD", "AUTO_LAND_REQUESTED", "AUTO_LAND", "DONE")):
        t = 100. + (i + 1) * .001
        node.state = state
        node._publish_execution_locked(t)
        ack = node.executions[-1]
        assert ack["intervention_id"] == result["intervention_id"]
        assert core.update_execution(ack), state
        later = core.evaluate(t)
        assert later["state"] == "STOP"
        assert later["intervention_id"] == result["intervention_id"]


def test_invalid_guard_between_ticks_cannot_be_erased_by_next_good_status(wrapper):
    node = make_node(wrapper)
    node._guard_cb(Obj(data="invalid-json"))
    assert node.stop_executor.latched
    assert node.stop_executor.event["source"] == "mission_failsafe"
    node._guard_cb(guard_json(seq=3))
    assert node.guard.current(100.)
    assert node.state == "STOP_AWAIT_LOCAL_POSE"
    assert node.sp is None


def test_takeoff_starts_without_planner_on_healthy_guard(wrapper):
    node = make_node(wrapper)
    node.state, node.start_req, node.sp = "WAIT", True, None
    node._enter = lambda state, why="": setattr(node, "state", state)
    node.guard.update(guard_json(seq=2, state="STANDBY", ready=False, takeoff_ready=False).data, 100.)
    node._wait(pose(), node.mav, None, 99., 0.)
    assert node.state == "WAIT"
    node.guard.update(guard_json(seq=3, state="STANDBY", ready=False, takeoff_ready=True).data, 100.)
    node._wait(pose(), node.mav, None, 99., 0.)
    assert node.state == "STREAM" and node.home == (2.0, 3.0, 1.0, 0.0)


def test_climb_altitude_is_takeoff_height_never_planner_z(wrapper):
    node = make_node(wrapper)
    node.state, node.t_state = "STREAM", 98.
    node.home, node.z_takeoff = (2., 3., 0.2, 0.), 1.0
    node._send = lambda *a, **k: True
    node._enter = lambda state, why="": setattr(node, "state", state)
    planner = Target()
    planner.position.z = 5.0
    node._stream(pose(), node.mav, planner, 0.0, 0.)
    assert node.state == "CLIMB" and node.z_want == pytest.approx(1.2)


def test_execution_reports_hover_altitude_and_planner_presence(wrapper):
    node = make_node(wrapper)
    node.state, node.z_want, node.settle, node.t_level = "CLIMB", 1.2, 3.0, 96.0
    node.sp, node.t_sp = None, 0.
    node._publish_execution_locked(100.)
    doc = node.executions[-1]
    assert doc["z_want_local"] == 1.2 and doc["hover_settled"] and not doc["planner_fresh"]
