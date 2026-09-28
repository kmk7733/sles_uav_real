#!/usr/bin/env python3
"""Express MAVROS EKF body pose in the shared, fixed planning world.

The input topic defines the local ENU contract: MAVROS headers may say odom
or map, neither of which is the ZED tracking map. Alignment comes only from
ekf_world_align; an old TF or ZED pose is never a fallback.
"""
import threading

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String

from ekf_alignment import SharedFrameAlignment


class PoseToWorld:
    def __init__(self):
        self.world_frame = rospy.get_param('~world_frame', 'plan_world').lstrip('/')
        self.local_frame = rospy.get_param('~local_frame', 'fcu_local').lstrip('/')
        self.alignment_max_age = float(rospy.get_param('~alignment_max_age', 0.5))
        self.pose_max_age = float(rospy.get_param('~pose_max_age', 0.25))
        self._lock = threading.RLock()
        self.alignment = SharedFrameAlignment(self.world_frame, self.local_frame)
        self.pub = rospy.Publisher('~pose_out', PoseStamped, queue_size=10)
        self.pub_epoch = rospy.Publisher(
            rospy.get_param('~pose_epoch_out', '/robot/pose_world_epoch'),
            TransformStamped, queue_size=10)
        self.sub_alignment = rospy.Subscriber(
            rospy.get_param('~alignment_topic', '/robot/frame_alignment'),
            String, self.cb_alignment, queue_size=10)
        self.sub_pose = rospy.Subscriber('~pose_in', PoseStamped, self.cb,
                                         queue_size=10)

    def cb_alignment(self, msg):
        with self._lock:
            self.alignment.update_status(msg.data, rospy.Time.now().to_sec())

    def cb(self, msg):
        now = rospy.Time.now().to_sec()
        stamp = msg.header.stamp.to_sec()
        with self._lock:
            a = self.alignment
            if not a.is_ready(now, self.alignment_max_age):
                rospy.logwarn_throttle(5.0, 'pose_to_world: waiting for valid EKF alignment')
                return
            if (stamp <= 0.0 or stamp < a.valid_from or
                    stamp > now + 0.05 or now - stamp > self.pose_max_age):
                return
            p, q = msg.pose.position, msg.pose.orientation
            try:
                xyz = a.local_to_world([p.x, p.y, p.z])
                quat = a.orientation_to_world([q.x, q.y, q.z, q.w])
            except ValueError as exc:
                rospy.logwarn_throttle(5.0, 'pose_to_world: invalid EKF pose (%s)', exc)
                return
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = self.world_frame
            out.pose.position.x, out.pose.position.y, out.pose.position.z = xyz
            (out.pose.orientation.x, out.pose.orientation.y,
             out.pose.orientation.z, out.pose.orientation.w) = quat
            tagged = TransformStamped()
            tagged.header.stamp = msg.header.stamp
            tagged.header.frame_id = self.world_frame
            # This is a topic payload, NOT a TF broadcast. The child field
            # makes pose+epoch atomic without changing /robot/pose_world or
            # relying on Header.seq (which rospy overwrites at publication).
            tagged.child_frame_id = 'ekf_body/epoch/' + a.epoch
            (tagged.transform.translation.x, tagged.transform.translation.y,
             tagged.transform.translation.z) = xyz
            (tagged.transform.rotation.x, tagged.transform.rotation.y,
             tagged.transform.rotation.z, tagged.transform.rotation.w) = quat
            # Serialize validity/epoch changes with publication. The original
            # measurement stamp and EKF roll/pitch are preserved.
            self.pub_epoch.publish(tagged)
            self.pub.publish(out)


if __name__ == '__main__':
    rospy.init_node('pose_to_world')
    PoseToWorld()
    rospy.spin()
