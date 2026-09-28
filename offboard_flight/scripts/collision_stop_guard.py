#!/usr/bin/env python3
"""Explicitly selected ROS adapter for the Vicon CollisionStopGuard.

Run with --profile PATH. No command publisher exists in a shadow profile.
Enforcement additionally requires --enforce; no model or mode is inferred.
This node cannot hold, change PX4 modes, arm or land. Its only command output
is the unchanged nominal PositionTarget when the independent gate permits it.
"""

import argparse
import copy
import hashlib
import json
import os
import threading

from collision_stop_core import CollisionStopCore, validate_profile


def nominal_metadata(msg):
    return {"stamp": msg.header.stamp.to_sec(), "frame_id": msg.header.frame_id,
            "coordinate_frame": int(msg.coordinate_frame), "type_mask": int(msg.type_mask),
            "fields": [msg.position.x, msg.position.y, msg.position.z,
                       msg.velocity.x, msg.velocity.y, msg.velocity.z,
                       msg.acceleration_or_force.x, msg.acceleration_or_force.y,
                       msg.acceleration_or_force.z, msg.yaw, msg.yaw_rate]}


def check_resolved_topics(ros, config):
    """Reject aliases/remaps that could turn the nominal gate into an FCU writer."""
    resolved = {key: ros.resolve_name(value) for key, value in config["topics"].items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("resolved nominal/safe/status/execution topics must be distinct")
    sensor_topics = [config["vehicle"]["topic"]] + list(config["obstacle_topics"].values())
    sensor_names = [ros.resolve_name(topic) for topic in sensor_topics]
    if len(set(sensor_names)) != len(sensor_names) or set(sensor_names) & set(resolved.values()):
        raise ValueError("resolved Vicon subjects must be distinct from each other and gate topics")
    # Both a relative MAVROS remap and any namespace ending in a MAVROS command
    # topic are forbidden; guard status is not a substitute command channel.
    for name in resolved.values():
        if "/mavros/" in name or name.endswith("/setpoint_raw/local"):
            raise ValueError("CollisionStopGuard topics cannot be MAVROS command/state topics")
    return resolved


def check_graph_state(system_state, node_name, topics, mode, before_init=True):
    """Refuse an existing gate/writer; never evict it by ROS name registration."""
    publishers, subscribers, services = system_state
    if before_init:
        nodes = set(n for _, owners in publishers + subscribers + services for n in owners)
        if node_name in nodes:
            raise RuntimeError("an existing ROS node already owns " + node_name)
    outputs = [topics["status"]] + ([topics["safe"]] if mode == "enforce" else [])
    for topic, owners in publishers:
        if topic in outputs and any(before_init or n != node_name for n in owners):
            raise RuntimeError("existing publisher on guard output %s: %s" % (topic, owners))


def _preinit_graph_check(rosgraph, profile, argv):
    mappings = rosgraph.names.load_mappings(argv)
    namespace = mappings.get("__ns", os.environ.get("ROS_NAMESPACE", "/"))
    namespace = "/" + namespace.strip("/")
    if namespace != "/":
        namespace += "/"
    default_name = "collision_stop_guard_shadow" if profile["mode"] == "shadow" else "collision_stop_guard"
    node_name = rosgraph.names.resolve_name(mappings.get("__name", default_name), namespace)
    resolved_remaps = {}
    for source, target in mappings.items():
        if not source.startswith("_"):
            resolved_remaps[rosgraph.names.resolve_name(source, namespace)] = rosgraph.names.resolve_name(target, namespace)

    class Resolver(object):
        @staticmethod
        def resolve_name(name):
            if name.startswith("~"):
                raise ValueError("shared guard topics must be explicit global or relative names, not private names")
            resolved = rosgraph.names.resolve_name(name, namespace)
            return resolved_remaps.get(resolved, resolved)

    topics = check_resolved_topics(Resolver, profile)
    master = rosgraph.Master(node_name + "_read_only_preflight")
    check_graph_state(master.getSystemState(), node_name, topics, profile["mode"])
    return default_name, node_name


class CollisionStopGuard(object):
    def __init__(self, ros, message_types, profile, map_doc, log_stream):
        # message_types is just an injectable message-type dictionary; no ROS
        # imports are necessary when running the protocol/wiring unit tests.
        self.ros = ros
        self.types = message_types
        self.core = CollisionStopCore(profile, map_doc)
        self.profile = self.core.config
        self.lock = threading.RLock()
        self.nominal_msg = None
        self.log_stream = log_stream
        self.topics = check_resolved_topics(ros, self.profile)
        self.status_pub = ros.Publisher(self.topics["status"], message_types["String"], queue_size=1)
        self.safe_pub = None
        if self.profile["mode"] == "enforce":
            self.safe_pub = ros.Publisher(self.topics["safe"], message_types["PositionTarget"], queue_size=1)
        ros.Subscriber(self.topics["nominal"], message_types["PositionTarget"], self._nominal_cb, queue_size=1)
        ros.Subscriber(self.topics["execution"], message_types["String"], self._execution_cb, queue_size=1)
        subjects = [("vehicle", self.profile["vehicle"]["topic"])] + sorted(self.profile["obstacle_topics"].items())
        for name, topic in subjects:
            ros.Subscriber(topic, message_types["TransformStamped"], self._vicon_cb,
                           callback_args=name, queue_size=10)

    def _nominal_cb(self, msg):
        with self.lock:
            valid = self.core.update_nominal(nominal_metadata(msg))
            self.nominal_msg = copy.deepcopy(msg) if valid else None

    def _execution_cb(self, msg):
        try:
            value = json.loads(msg.data)
        except (ValueError, TypeError):
            value = None
        with self.lock:
            self.core.update_execution(value)

    def _vicon_cb(self, msg, name):
        t, q = msg.transform.translation, msg.transform.rotation
        with self.lock:
            self.core.update_vicon(name, msg.header.stamp.to_sec(), msg.header.frame_id,
                                   (t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def tick(self):
        with self.lock:
            result = self.core.evaluate(self.ros.Time.now().to_sec())
            # Persist the decision before forwarding a command. An I/O failure
            # exits the node without forwarding; executor heartbeat timeout
            # retains its independent authority to hold.
            self.log_stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            self.log_stream.flush()
            self.status_pub.publish(self.types["String"](data=json.dumps(result, allow_nan=False)))
            if self.safe_pub is not None and result["allow_nominal"] and self.nominal_msg is not None:
                # Never turn an old producer command into a fresh one. Exact
                # header stamp, frame/epoch, mask and numeric payload survive.
                self.safe_pub.publish(copy.deepcopy(self.nominal_msg))
            return result

    def run(self):
        rate = self.ros.Rate(self.profile["limits"]["rate_hz"])
        while not self.ros.is_shutdown():
            self.tick()
            rate.sleep()


def main(argv=None):
    import sys
    import rospy
    import rosgraph
    from geometry_msgs.msg import TransformStamped
    from mavros_msgs.msg import PositionTarget
    from std_msgs.msg import String
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="explicit enabled JSON profile")
    parser.add_argument("--enforce", action="store_true", help="also require profile.mode=enforce")
    command_argv = sys.argv if argv is None else argv
    args = parser.parse_args(rospy.myargv(argv=command_argv)[1:])
    with open(os.path.expanduser(args.profile)) as stream:
        profile = validate_profile(json.load(stream))
    if (profile["mode"] == "enforce") != args.enforce:
        parser.error("enforcement requires BOTH profile.mode=enforce and --enforce")
    map_path = os.path.expanduser(profile["map_path"])
    log_path = os.path.expanduser(profile["log_path"])
    if not os.path.isabs(map_path) or not os.path.isabs(log_path):
        parser.error("map_path and log_path must be absolute")
    with open(map_path, "rb") as stream:
        map_bytes = stream.read()
    map_doc = yaml.safe_load(map_bytes)
    CollisionStopCore(profile, map_doc)  # refuse invalid geometry before ROS advertisements
    default_name, node_name = _preinit_graph_check(rosgraph, profile, command_argv)
    rospy.init_node(default_name, argv=command_argv, anonymous=False)
    topics = check_resolved_topics(rospy, profile)
    # Recheck after registration, before any publisher is advertised. This
    # narrows the startup race and refuses existing legacy safe-topic writers.
    check_graph_state(rosgraph.Master(rospy.get_name()).getSystemState(),
                      rospy.get_name(), topics, profile["mode"], before_init=False)
    # Exclusive-create prevents silently mixing/replacing evidence from another
    # session. The log's parent directory must already exist.
    with open(log_path, "x") as stream:
        stream.write(json.dumps({"record": "configuration", "profile": profile,
                                 "map_sha256": hashlib.sha256(map_bytes).hexdigest(),
                                 "risk_contract": "Vicon-only, planar infinite pillars, constant velocity"},
                                sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        CollisionStopGuard(rospy, {"String": String, "TransformStamped": TransformStamped,
                                  "PositionTarget": PositionTarget}, profile, map_doc, stream).run()


if __name__ == "__main__":
    main()
