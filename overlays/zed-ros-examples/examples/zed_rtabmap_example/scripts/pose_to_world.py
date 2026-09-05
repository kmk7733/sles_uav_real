#!/usr/bin/env python3
"""
Republish a PoseStamped into the world frame of the grid map.

Bridges the localization source (ZED pose in 'map', later mavros
local_position) to consumers that expect poses in the same frame as
/grid_map (e.g. mpc_node, which uses its pose topic verbatim). Uses TF, so
it works through the static vicon/world->map latched by vicon_map_align.

  ~pose_in   (geometry_msgs/PoseStamped)  source pose
  ~pose_out  (geometry_msgs/PoseStamped)  same pose expressed in ~world_frame
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped


class PoseToWorld:
    def __init__(self):
        self.world_frame = rospy.get_param('~world_frame', 'vicon/world')
        # tf2 rejects leading '/' in lookups; messages keep the name verbatim
        self.world_frame_tf = self.world_frame.lstrip('/')
        # override for sources with a wrong/empty frame_id (e.g. mavros also
        # stamps 'map' but means the FCU ENU frame, not the ZED map frame)
        self.pose_frame = rospy.get_param('~pose_frame', '')
        self.buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.buf)
        self.pub = rospy.Publisher('~pose_out', PoseStamped, queue_size=10)
        rospy.Subscriber('~pose_in', PoseStamped, self.cb, queue_size=10)

    def cb(self, msg):
        src = (self.pose_frame or msg.header.frame_id).lstrip('/')
        if src == self.world_frame_tf:
            msg.header.frame_id = self.world_frame
            self.pub.publish(msg)
            return
        try:
            # Time(0): the source->world chain is static after alignment
            tfm = self.buf.lookup_transform(self.world_frame_tf, src,
                                            rospy.Time(0), rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, 'pose_to_world: no TF %s->%s (%s)',
                                   src, self.world_frame, e)
            return
        msg.header.frame_id = src
        out = tf2_geometry_msgs.do_transform_pose(msg, tfm)
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.world_frame
        self.pub.publish(out)


if __name__ == '__main__':
    rospy.init_node('pose_to_world')
    PoseToWorld()
    rospy.spin()
