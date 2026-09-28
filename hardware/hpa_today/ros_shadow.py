#!/usr/bin/env python3
"""HPA CUDA + reference lifecycle on an isolated ROS replay graph only."""
import argparse
import json
import math
import numpy as np
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT/'offboard_flight/scripts', ROOT/'hardware/hpa_shadow', Path(__file__).parent):
    sys.path.insert(0, str(directory))
from shadow_node import Shadow, parser as shadow_parser
from shadow_core import Rejected, check_age, quaternion_rpy, verify_bundle
from hpa_depth_adapter import odom_pose
from reference_lifecycle import ReferenceLifecycle, warmup_runtime

PREFIX = '/hpa_validation'


def require_isolated_master():
    master = urlparse(os.environ.get('ROS_MASTER_URI', ''))
    if master.scheme != 'http' or master.hostname not in ('127.0.0.1', 'localhost') or master.port in (None, 11311):
        raise ValueError('explicit loopback non-11311 ROS_MASTER_URI required; no live graph allowed')


class ReferenceShadow(Shadow):
    # The controller node follows an accepted trajectory like the simulator's
    # ReferenceHolder and does not stop on the newest depth's age at output.
    output_requires_depth = True

    def __init__(self, *args, **kwargs):
        self.lifecycle = ReferenceLifecycle(
            input_max_age=args[0].completion_max_age,
            future_tolerance=args[0].future_tolerance,
            reference_max_duration=getattr(args[0], 'reference_max_duration', None))
        self.output_clock = None
        self.depth_validation = None
        self.reference_pub = None
        self.ready_file = None
        self.preview_count = 0
        self.reference_count = 0
        super().__init__(*args, **kwargs)

    def previous_acceleration(self, anchor_stamp, epoch):
        return self.lifecycle.previous_acceleration(anchor_stamp, epoch)

    def depth_scan_validated(self, anchor_stamp, valid):
        # Projection is already done by the inference worker. Do not decode or
        # reproject the full depth image in the higher-rate output timer.
        with self.lock:
            if self.depth_validation is None or anchor_stamp >= self.depth_validation[0]:
                self.depth_validation = (anchor_stamp, bool(valid))

    def output_block_reason_locked(self, now):
        """Current input health, checked under self.lock before every output.

        Sensor timeouts concern the latest received measurements, independently
        of the older depth anchor of an accepted trajectory. A known-invalid
        processed depth frame blocks output until a later projection succeeds.
        Pending depth is not processed a second time here. Camera calibration
        is fixed per session and does not have an age timeout.
        """
        if self.fatal:
            return self.fatal
        if not math.isfinite(now) or now <= 0:
            self.invalidate_locked('invalid output ROS clock')
            return self.fatal
        if self.output_clock is not None and now < self.output_clock:
            self.invalidate_locked('output ROS clock reversal')
            return self.fatal
        self.output_clock = now
        try:
            if self.state is None or not self.state[1].connected:
                raise Rejected('FCU disconnected or state unavailable at output')
            check_age(self.state[0], now, self.args.state_max_age, self.args.future_tolerance)
            if self.output_requires_depth:
                if self.latest_depth is None:
                    raise Rejected('missing depth at output')
                check_age(self.latest_depth[0], now, self.args.max_age, self.args.future_tolerance)
                if self.depth_validation is None or not self.depth_validation[1]:
                    raise Rejected('no valid processed depth at output')
            records = {}
            for name in ('pose', 'velocity'):
                if not self.buffers[name]:
                    raise Rejected('missing ' + name + ' at output')
                records[name] = self.buffers[name][-1]
                check_age(records[name][0], now, self.args.max_age, self.args.future_tolerance)
            pose = records['pose'][1]
            position, quaternion = odom_pose(pose)
            quaternion_rpy(quaternion)
            linear = records['velocity'][1].twist.linear
            values = list(position) + [linear.x, linear.y, linear.z, pose.twist.twist.angular.z]
            if not np.isfinite(np.asarray(values, dtype=float)).all():
                raise Rejected('nonfinite current PX4 state at output')
        except (Rejected, ValueError, TypeError, AttributeError) as error:
            return str(error)
        return None

    def invalidate_locked(self, reason):
        self.lifecycle.invalidate(reason)
        return super().invalidate_locked(reason)

    def emit(self, event, **fields):
        if event == 'prediction':
            accepted = self.lifecycle.accept(dict(event=event, **fields), self.rospy.Time.now().to_sec())
            fields['reference_lifecycle_accepted'] = accepted
            if accepted and self.reference_pub is not None:
                self.reference_count += 1
                payload = dict(anchor_stamp=fields['anchor_stamp'], estimator_epoch=fields['estimator_epoch'],
                               goal_local_enu=fields['goal_local_enu'], reference=fields['integrated_reference'],
                               purpose='diagnostic reference only; no commander/FCU consumer')
                from std_msgs.msg import String
                self.reference_pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        super().emit(event, **fields)

    def snapshot(self):
        result = super().snapshot()
        # This is called by the base poll loop after subscriptions exist.
        if self.ready_file is not None and not self.ready_file.exists():
            with self.ready_file.open('x') as f:
                json.dump(dict(pid=os.getpid(), wall_time=time.time(), cuda_ready=True,
                               subscriptions_ready=True, namespace=PREFIX,
                               no_fcu_interface=True), f)
        return result

    def preview_once(self, timer_event=None):
        # Dedicated timer: inference does not set the reference sampling rate.
        # Serialize enqueueing with reset/disconnect invalidation, matching
        # the base class's final prediction acceptance lock ordering.
        with self.lock:
            now = self.rospy.Time.now().to_sec()
            reason = self.output_block_reason_locked(now)
            if reason is not None:
                self.reject('reference output blocked: ' + reason)
                return
            preview = self.lifecycle.sample(now, self.epoch)
            if preview is not None:
                from std_msgs.msg import String
                self.preview_pub.publish(String(data=json.dumps(preview, allow_nan=False)))
                self.preview_count += 1
                super().emit('reference_preview', **preview)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=ROOT/'deploy/rogx_hpa_v4')
    p.add_argument('--config', type=Path, default=ROOT/'offboard_flight/scripts/planar_producer_config.example.json')
    p.add_argument('--duration', type=float, default=100.)
    p.add_argument('--reference-max-duration', type=float, default=None,
                   help='optional trajectory duration from depth anchor; default is the model horizon (1s), independent of input max-age')
    p.add_argument('--workload-label', default='isolated recorded-input ROS transport; concurrent load unverified')
    args = p.parse_args()
    require_isolated_master()
    if not 1 <= args.duration <= 3600:
        p.error('duration must be 1..3600 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import rospy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TwistStamped
    from mavros_msgs.msg import State
    rospy.init_node('hpa_reference_shadow', anonymous=True, disable_rosout=True)
    if rospy.get_param('/use_sim_time', False):
        raise ValueError('this transport replay uses constant-shifted wall timestamps, not /clock')
    base = shadow_parser().parse_args(['--bundle', str(args.bundle), '--log', str(args.output_dir/'shadow.jsonl'),
        '--goal-current-position', '--depth-frame', 'zed2i_left_camera_optical_frame', '--device', 'cuda',
        '--duration', str(args.duration), '--producer-config', str(args.config), '--workload-label', args.workload_label])
    # Recorded-input diagnostics use the training transform as an assumption,
    # never an operator confirmation of physical mounting on today's aircraft.
    base.reference_max_duration = args.reference_max_duration
    base.depth_topic = PREFIX+'/input/depth'
    base.camera_info_topic = PREFIX+'/input/camera_info'
    base.pose_topic = PREFIX+'/input/odom'
    base.velocity_topic = PREFIX+'/input/velocity_local'
    base.state_topic = PREFIX+'/input/fcu_state_recorded'
    base.invalidation_topic = PREFIX+'/input/invalidation'
    directory, manifest, info = verify_bundle(args.bundle)
    received = {'reference': 0, 'preview': 0}
    def observed(name):
        def callback(message):
            payload = json.loads(message.data)
            if 'anchor_stamp' not in payload:
                raise ValueError('unstamped diagnostic reference')
            received[name] += 1
        return callback
    monitors = [rospy.Subscriber(PREFIX+'/output/reference', String, observed('reference'), queue_size=10),
                rospy.Subscriber(PREFIX+'/output/preview', String, observed('preview'), queue_size=10)]
    with base.log.open('x', buffering=1) as log:
        shadow = ReferenceShadow(base, rospy, directory, manifest, info, log)
        shadow.emit('model_warmup', **warmup_runtime(shadow.runtime))
        shadow.reference_pub = rospy.Publisher(PREFIX+'/output/reference', String, queue_size=1)
        shadow.preview_pub = rospy.Publisher(PREFIX+'/output/preview', String, queue_size=1)
        shadow.ready_file = args.output_dir/'ready.json'
        preview_timer = rospy.Timer(rospy.Duration(.02), shadow.preview_once)
        failure = None
        try:
            shadow.run()
        except Exception as error:
            failure = str(error)
        finally:
            preview_timer.shutdown()
            # shutdown only sets rospy.Timer's flag. Let an in-flight callback
            # finish before counting messages, unregistering or closing its log.
            preview_timer.join()
            summary = dict(success=failure is None and received['reference'] > 0 and received['preview'] > 0,
                           failure=failure, counters=dict(shadow.counts), published_references=shadow.reference_count,
                           published_previews=shadow.preview_count, tcp_ros_received=received,
                           preview_timer_target_hz=50., control_outputs=0, fcu_service_calls=0,
                           scope='recorded-input ROS shadow only')
            with (args.output_dir/'transport_summary.json').open('x') as f:
                json.dump(summary, f, indent=2)
            for sub in monitors:
                sub.unregister()
            shadow.reference_pub.unregister()
            shadow.preview_pub.unregister()
    print(json.dumps(summary, indent=2))
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
