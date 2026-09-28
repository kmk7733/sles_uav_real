"""ROS-free adapter tests, including publisher authority and wire preservation."""
import copy
import io
import json
from types import SimpleNamespace as NS

import pytest

from test_collision_stop_core import profile, map_doc, healthy, execution
from collision_stop_guard import (CollisionStopGuard, check_resolved_topics,
                                 check_graph_state, _preinit_graph_check)


class Stamp(object):
    def __init__(self, sec):
        self.sec = sec

    def to_sec(self):
        return self.sec


class String(object):
    def __init__(self, data):
        self.data = data


class FakeRos(object):
    def __init__(self, remaps=None):
        self.remaps = remaps or {}
        self.publishers = {}
        self.subscribers = []
        self.now = 1.0
        self.Time = NS(now=lambda: Stamp(self.now))

    def resolve_name(self, name):
        return self.remaps.get(name, "/rogx2/" + name.lstrip("/"))

    def Publisher(self, topic, typ, **kwargs):
        messages = []
        self.publishers[topic] = messages
        return NS(publish=lambda msg: messages.append(copy.deepcopy(msg)))

    def Subscriber(self, topic, typ, callback, **kwargs):
        self.subscribers.append((topic, callback, kwargs))


def message(stamp=1.0):
    vec = lambda x, y, z: NS(x=x, y=y, z=z)
    return NS(header=NS(stamp=Stamp(stamp), frame_id="fcu_local/epoch/abcdef123"),
              coordinate_frame=1, type_mask=0, position=vec(1, 2, 3),
              velocity=vec(0.1, 0.2, 0.3), acceleration_or_force=vec(0.4, 0.5, 0.6),
              yaw=0.7, yaw_rate=0.8)


def build(profile, map_doc, ros=None):
    ros = ros or FakeRos()
    stream = io.StringIO()
    guard = CollisionStopGuard(ros, {"String": String, "PositionTarget": object,
                                    "TransformStamped": object}, profile, map_doc, stream)
    healthy(guard.core)
    guard._nominal_cb(message(1.001))
    ros.now = 1.01
    return ros, guard, stream


def test_shadow_advertises_diagnostics_only_no_px4_subscription(profile, map_doc):
    ros, guard, stream = build(profile, map_doc)
    result = guard.tick()
    assert result["ready"] and result["mode"] == "shadow"
    assert list(ros.publishers) == [ros.resolve_name(profile["topics"]["status"])]
    assert not any("mavros" in topic for topic, _, _ in ros.subscribers)
    assert guard.safe_pub is None
    assert json.loads(stream.getvalue())["risk_source"] == "vicon_gt_only"
    # FakeRos has no service API: constructing/ticking this node cannot change
    # modes, arm, publish raw controls or accidentally instantiate a service.


def test_gate_preserves_original_header_mask_and_all_fields(profile, map_doc):
    profile["mode"] = "enforce"
    ros, guard, stream = build(profile, map_doc)
    guard.tick()
    outputs = ros.publishers[ros.resolve_name(profile["topics"]["safe"])]
    out = outputs[0]
    assert out.header.stamp.to_sec() == 1.001
    assert out.header.frame_id == "fcu_local/epoch/abcdef123"
    assert (out.type_mask, out.coordinate_frame) == (0, 1)
    assert (out.position.x, out.velocity.y, out.acceleration_or_force.z, out.yaw, out.yaw_rate) == (1, 0.2, 0.6, 0.7, 0.8)
    ros.now = 1.02
    guard.tick()
    assert outputs[-1].header.stamp.to_sec() == 1.001  # no stale command rejuvenation
    guard.core.samples["vehicle"]["center"] = (1.8, 0, 1)
    assert guard.tick()["state"] == "STOP"
    assert len(outputs) == 2


def test_stale_nominal_suppressed_and_latched(profile, map_doc):
    profile["mode"] = "enforce"
    ros, guard, _ = build(profile, map_doc)
    guard.core.nominal["stamp"] = 0.5
    assert guard.tick()["state"] == "STOP"
    assert not ros.publishers[ros.resolve_name(profile["topics"]["safe"])]


@pytest.mark.parametrize("remaps", [
    {"commander/set_pose_safe": "/rogx2/mavros/setpoint_raw/local"},
    {"commander/set_pose_safe": "/rogx2/commander/set_pose"},
    {"commander/collision_stop_status": "/rogx2/commander/collision_stop_execution"},
    {"/vicon/wall/wall": "/rogx2/vicon/drone/drone"},
])
def test_unsafe_topic_remaps_refused_before_advertising(profile, map_doc, remaps):
    ros = FakeRos(remaps)
    with pytest.raises(ValueError):
        build(profile, map_doc, ros)
    assert not ros.publishers


def test_execution_ack_is_in_jsonl(profile, map_doc):
    ros, guard, stream = build(profile, map_doc)
    value = execution(1.005, "STOP_HOLD", 2)
    value.update(phase="STOP_HOLD", intervention_id="abc", observed_mode="OFFBOARD",
                 land_attempts=0, hold_anchor_local=[1, 2, 3, 0])
    guard._execution_cb(String(json.dumps(value)))
    guard.tick()
    assert json.loads(stream.getvalue())["execution"] == value


def test_log_failure_never_forwards_unlogged_command(profile, map_doc):
    profile["mode"] = "enforce"
    ros, guard, stream = build(profile, map_doc)
    stream.close()
    with pytest.raises(ValueError):
        guard.tick()
    assert not ros.publishers[ros.resolve_name(profile["topics"]["safe"])]


def test_existing_legacy_safe_writer_and_same_name_refused(profile):
    topics = {k: "/" + v for k, v in profile["topics"].items()}
    with pytest.raises(RuntimeError, match="existing publisher"):
        check_graph_state(([(topics["safe"], ["/vicon_safety_supervisor"])], [], []),
                          "/collision_stop_guard", topics, "enforce")
    with pytest.raises(RuntimeError, match="already owns"):
        check_graph_state(([], [("/unrelated", ["/collision_stop_guard"])], []),
                          "/collision_stop_guard", topics, "enforce")
    # Shadow owns no safe publisher, and does not evict the operational guard.
    check_graph_state(([(topics["safe"], ["/vicon_safety_supervisor"])], [], []),
                      "/collision_stop_guard_shadow", topics, "shadow")


def test_preinit_resolves_remapped_node_name_without_init(profile):
    def resolve(name, namespace):
        return name if name.startswith("/") else namespace + name
    fake_graph = NS(names=NS(load_mappings=lambda argv: {"__ns": "/rogx2", "__name": "guard"},
                             resolve_name=resolve),
                    Master=lambda name: NS(getSystemState=lambda: ([], [("/x", ["/rogx2/guard"])], [])))
    with pytest.raises(RuntimeError, match="already owns"):
        _preinit_graph_check(fake_graph, profile, ["unused"])


def test_master_unavailable_is_not_permission_to_publish(profile):
    def fail():
        raise RuntimeError("master unreachable")
    fake_graph = NS(names=NS(load_mappings=lambda argv: {},
                             resolve_name=lambda name, ns: ns + name.lstrip("/")),
                    Master=lambda name: NS(getSystemState=fail))
    with pytest.raises(RuntimeError, match="master unreachable"):
        _preinit_graph_check(fake_graph, profile, ["unused"])


@pytest.mark.parametrize("from_env", [True, False])
def test_relative_namespace_normalized_before_graph_query(profile, monkeypatch, from_env):
    monkeypatch.setenv("ROS_NAMESPACE", "rogx2")
    seen_callers = []

    def master(name):
        seen_callers.append(name)
        return NS(getSystemState=lambda: ([], [], []))

    fake_graph = NS(names=NS(load_mappings=lambda argv: {} if from_env else {"__ns": "rogx2"},
                             resolve_name=lambda name, ns: name if name.startswith("/") else ns + name),
                    Master=master)
    default_name, node_name = _preinit_graph_check(fake_graph, profile, ["unused"])
    assert node_name == "/rogx2/collision_stop_guard_shadow"
    assert seen_callers == ["/rogx2/collision_stop_guard_shadow_read_only_preflight"]
