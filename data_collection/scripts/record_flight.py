#!/usr/bin/env python3
"""Record a human-piloted drone flight into a rosbag for HPA planner training.

This is the PRIMARY, low-overhead collection tool intended to run on the
onboard / remote flight computer. It wraps `rosbag record` with the topic list
from config.yaml, writing each flight to its own timestamped session folder
together with a metadata.json describing what was captured.

Why rosbag (and not a live per-frame writer): rosbag uses a dedicated buffered
writer and is the most reliable way to capture high-rate stereo + depth without
dropping frames on the flight computer. Turning bags into a synchronized
(image -> x,y,z,yaw) training set is done afterwards, off the drone, with
`extract_flight_dataset.py`.

Usage:
    # start recording (Ctrl+C to stop and finalize)
    python3 record_flight.py --note "indoor_forest_run1"

    # just print the topics that WOULD be recorded, then exit
    python3 record_flight.py --dry-run

    # verify every configured topic is actually being published right now
    python3 record_flight.py --check
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime

import yaml


# Where the label pose is read from, keyed by `label_source` in config.yaml.
# Each value is a `{ns}`-templated topic resolved against the drone namespace.
_LABEL_TOPICS = {
    "mavros_odom": "{ns}/mavros/local_position/odom",   # EKF2 fused, ENU
    "zed_odom": "{ns}/zed2i/zed_node/odom",             # ZED visual-inertial
}


class Config:
    """Minimal, self-contained replacement for `dds_common.Config`.

    Loads config.yaml, resolves the `{ns}` template in every topic against the
    drone namespace, and exposes the handful of attributes/methods that
    record_flight.py relies on.
    """

    def __init__(self, path=None):
        if path is None:
            # Default to config.yaml sitting next to this script.
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        self.path = os.path.expanduser(path)
        with open(self.path) as f:
            self._raw = yaml.safe_load(f) or {}

        # Namespace, normalized to a single leading slash and no trailing slash
        # so that "{ns}/mavros/..." expands to an absolute topic like
        # "/rogx2/mavros/...".
        drone_ns = str(self._raw.get("drone_ns", "")).strip().strip("/")
        self.ns = ("/" + drone_ns) if drone_ns else ""

        self.output_dir = os.path.expanduser(str(self._raw.get("output_dir", "~/drone_data")))
        self.rosbag = self._raw.get("rosbag", {}) or {}
        self.extraction = self._raw.get("extraction", {}) or {}

        self.state_topics = self._resolve_list(self._raw.get("state_topics", []))
        self.rc_topics = self._resolve_list(self._raw.get("rc_topics", []))
        self.sensor_topics = self._resolve_list(self._raw.get("sensor_topics", []))
        self.tf_topics = self._resolve_list(self._raw.get("tf_topics", []))

        self.label_source = str(self._raw.get("label_source", "mavros_odom"))
        self.label_topic = self._resolve_label_topic()

    def _resolve(self, topic):
        """Substitute the `{ns}` placeholder in a single topic string."""
        return str(topic).format(ns=self.ns)

    def _resolve_list(self, topics):
        return [self._resolve(t) for t in (topics or [])]

    def _resolve_label_topic(self):
        if self.label_source in _LABEL_TOPICS:
            return self._resolve(_LABEL_TOPICS[self.label_source])
        if self.label_source == "tf":
            tf = self._raw.get("tf_label", {}) or {}
            return "tf:{parent}->{child}".format(
                parent=tf.get("parent_frame", "odom"),
                child=tf.get("child_frame", "base_link"),
            )
        return ""

    def all_record_topics(self):
        """Every topic to hand to `rosbag record`, de-duplicated, order-preserved."""
        seen = set()
        out = []
        for t in (self.state_topics + self.rc_topics + self.sensor_topics + self.tf_topics):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def check_topics(cfg):
    """Report which configured topics are currently live. Requires a ROS master."""
    try:
        import rospy  # noqa: F401
        from rostopic import get_topic_type
    except Exception as e:
        print(f"[check] Could not import ROS python API: {e}")
        return False

    print("[check] Probing configured topics against the ROS master...\n")
    all_ok = True
    for t in cfg.all_record_topics():
        ttype, _, _ = get_topic_type(t)
        if ttype is None:
            print(f"  MISSING  {t}")
            all_ok = False
        else:
            print(f"  ok       {t}   ({ttype})")
    print()
    if all_ok:
        print("[check] All configured topics are being published. ✅")
    else:
        print("[check] Some topics are missing — check the ZED/mavros nodes are up, "
              "or edit config.yaml. ⚠")
    return all_ok


def build_rosbag_cmd(cfg, bag_prefix):
    topics = cfg.all_record_topics()
    cmd = ["rosbag", "record", "-O", bag_prefix]

    comp = str(cfg.rosbag.get("compression", "lz4")).lower()
    if comp in ("lz4", "bz2"):
        cmd.append(f"--{comp}")

    split_mb = int(cfg.rosbag.get("split_size_mb", 0) or 0)
    if split_mb > 0:
        cmd += ["--split", f"--size={split_mb}"]

    buf_mb = int(cfg.rosbag.get("buffer_size_mb", 0) or 0)
    if buf_mb > 0:
        cmd += ["-b", str(buf_mb)]

    cmd += topics
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Record a drone flight to a rosbag.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--note", default="", help="Free-text label added to the folder name and metadata")
    parser.add_argument("--dry-run", action="store_true", help="Print the rosbag command and exit")
    parser.add_argument("--check", action="store_true", help="Check that configured topics are live, then exit")
    args = parser.parse_args()

    cfg = Config(args.config)

    if args.check:
        sys.exit(0 if check_topics(cfg) else 1)

    # --- session folder -------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    note = "_" + args.note.strip().replace(" ", "-") if args.note.strip() else ""
    session = f"session_{ts}{note}"
    session_dir = os.path.join(cfg.output_dir, session)
    os.makedirs(session_dir, exist_ok=True)
    bag_prefix = os.path.join(session_dir, f"flight_{ts}")

    topics = cfg.all_record_topics()
    cmd = build_rosbag_cmd(cfg, bag_prefix)

    # --- metadata sidecar -----------------------------------------------------
    metadata = {
        "session": session,
        "created": ts,
        "note": args.note,
        "drone_ns": cfg.ns,
        "label_source": cfg.label_source,
        "label_topic": cfg.label_topic,
        "state_topics": cfg.state_topics,
        "rc_topics": cfg.rc_topics,
        "sensor_topics": cfg.sensor_topics,
        "tf_topics": cfg.tf_topics,
        "rosbag_options": cfg.rosbag,
        "extraction_defaults": cfg.extraction,
        "rosbag_command": " ".join(shlex.quote(c) for c in cmd),
    }
    meta_path = os.path.join(session_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print(f"Drone flight recorder")
    print("=" * 70)
    print(f"Session folder : {session_dir}")
    print(f"Metadata       : {meta_path}")
    print(f"Label source   : {cfg.label_source}  ({cfg.label_topic})")
    print(f"Recording {len(topics)} topics:")
    for t in topics:
        print(f"    {t}")
    print("-" * 70)
    print("rosbag command:")
    print("  " + metadata["rosbag_command"])
    print("=" * 70)

    if args.dry_run:
        print("[dry-run] Not starting rosbag.")
        return

    print("Recording... press Ctrl+C to stop and finalize the bag.\n")

    # Run rosbag record as a child. Forward Ctrl+C so rosbag closes the bag
    # cleanly (rosbag installs its own SIGINT handler that flushes the buffer).
    proc = subprocess.Popen(cmd)

    def _forward(signum, frame):
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    ret = proc.wait()
    print(f"\n[done] rosbag exited with code {ret}.")
    print(f"[done] Data in: {session_dir}")
    print("[next] Extract a training set with:")
    print(f"    python3 extract_flight_dataset.py {session_dir}")


if __name__ == "__main__":
    main()
