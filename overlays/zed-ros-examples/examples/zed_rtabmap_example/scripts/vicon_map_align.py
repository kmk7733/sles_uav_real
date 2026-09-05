#!/usr/bin/env python3
"""
One-shot alignment of the ZED tracking frame ('map') into the Vicon world frame.

At startup:
  1. Averages ~n_samples of the robot pose from Vicon (T_world_subject).
  2. Reads the current ZED pose from TF (T_map_base) -- identity fallback if the
     ZED just started at this pose.
  3. Computes T_world_map = T_world_subject * T_subject_base * inv(T_map_base)
     and latches it as a STATIC TF  world -> map.

After that the Vicon is not used again: ZED positional tracking (map->base_link)
carries localization, so anything expressed in 'map' (grid, trajectories) is now
also available in the Vicon world frame through this fixed transform.

yaw_only (default true): both vicon/world and the ZED map frame are gravity
aligned (z up), so the true offset is translation + yaw. Projecting to yaw also
makes the alignment immune to a subject frame whose z-axis was defined pointing
down (observed on ROGX2: roll ~ -177 deg) -- the subject x-axis heading is
unaffected by that flip. Requires subject x-axis == robot forward.
"""

import json
import math
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import (quaternion_matrix, quaternion_from_matrix,
                                euler_matrix, euler_from_matrix)


def tf_to_matrix(t, q):
    m = quaternion_matrix([q.x, q.y, q.z, q.w])
    m[0:3, 3] = [t.x, t.y, t.z]
    return m


def main():
    rospy.init_node('vicon_map_align')

    vicon_topic = rospy.get_param('~vicon_topic', '/vicon/ROGX2/ROGX2')
    # slash-less (tf2 convention); vicon_rogx.launch broadcasts the same name
    world_frame = rospy.get_param('~world_frame', 'vicon/world')
    map_frame   = rospy.get_param('~map_frame', 'map')
    base_frame  = rospy.get_param('~base_frame', 'base_link')
    n_samples   = int(rospy.get_param('~n_samples', 25))
    timeout     = float(rospy.get_param('~vicon_timeout', 30.0))
    yaw_only    = bool(rospy.get_param('~yaw_only', True))
    # subject->base_link offset. Preferred source: a mount_calib.json with
    # 'mount_q_xyzw' = rotation body->subject (R_world_subject = R_world_body *
    # R_mount), so T_subject_base rotation = inv(mount_q). Fallback: the
    # ~subject_base_* params (rad/m).
    mount_calib = rospy.get_param('~mount_calib', '')
    sb_rpy = [float(rospy.get_param('~subject_base_' + k, 0.0))
              for k in ('roll', 'pitch', 'yaw')]
    sb_xyz = [float(rospy.get_param('~subject_base_' + k, 0.0))
              for k in ('x', 'y', 'z')]
    T_sb = None
    if mount_calib:
        with open(mount_calib) as f:
            calib = json.load(f)
        T_sb = np.linalg.inv(quaternion_matrix(calib['mount_q_xyzw']))
        rospy.loginfo('vicon_map_align: mount calib %s (n=%s, residual %.1f '
                      'deg): subject->base rpy_deg=[%.1f %.1f %.1f]',
                      mount_calib, calib.get('n', '?'),
                      calib.get('attitude_residual_deg_mean', float('nan')),
                      *[math.degrees(a) for a in euler_from_matrix(T_sb)])

    # --- 1. average Vicon samples -------------------------------------------
    rospy.loginfo('vicon_map_align: waiting for %d samples on %s ...',
                  n_samples, vicon_topic)
    trans, quats = [], []
    deadline = rospy.Time.now() + rospy.Duration(timeout)

    def cb(msg):
        t, q = msg.transform.translation, msg.transform.rotation
        trans.append([t.x, t.y, t.z])
        quats.append([q.x, q.y, q.z, q.w])

    sub = rospy.Subscriber(vicon_topic, TransformStamped, cb, queue_size=50)
    while len(trans) < n_samples and not rospy.is_shutdown():
        if rospy.Time.now() > deadline:
            rospy.logfatal('vicon_map_align: no Vicon data on %s within %.0fs',
                           vicon_topic, timeout)
            return
        rospy.sleep(0.05)
    sub.unregister()

    t_avg = np.mean(np.array(trans), axis=0)
    qs = np.array(quats)
    qs[np.dot(qs, qs[0]) < 0] *= -1.0          # hemisphere-align before mean
    q_avg = np.mean(qs, axis=0)
    q_avg /= np.linalg.norm(q_avg)

    T_ws = quaternion_matrix(q_avg)             # world -> subject
    T_ws[0:3, 3] = t_avg

    # --- 2. current ZED pose (map -> base_link) -----------------------------
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    T_mb = np.eye(4)
    try:
        tfm = buf.lookup_transform(map_frame, base_frame, rospy.Time(0),
                                   rospy.Duration(5.0))
        T_mb = tf_to_matrix(tfm.transform.translation, tfm.transform.rotation)
        rospy.loginfo('vicon_map_align: ZED pose in map: [%.3f %.3f %.3f]',
                      *T_mb[0:3, 3])
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logwarn('vicon_map_align: no %s->%s TF (%s) -- assuming robot '
                      'is at the ZED tracking origin', map_frame, base_frame, e)

    # --- 3. compose and latch ----------------------------------------------
    if T_sb is None:
        T_sb = euler_matrix(*sb_rpy)            # subject -> base_link
        T_sb[0:3, 3] = sb_xyz
    T_wb = T_ws @ T_sb                          # world -> base_link
    # sanity: if the robot is sitting level, world->base roll/pitch ~ 0
    rb, pb, yb = euler_from_matrix(T_wb)
    rospy.loginfo('vicon_map_align: robot in world: rpy_deg=[%.1f %.1f %.1f]',
                  math.degrees(rb), math.degrees(pb), math.degrees(yb))
    if max(abs(rb), abs(pb)) > math.radians(8.0):
        rospy.logwarn('vicon_map_align: world->base roll/pitch > 8 deg -- '
                      'robot tilted, or mount calibration is off')
    T_wm = T_wb @ np.linalg.inv(T_mb)           # world -> map

    if yaw_only:
        yaw = math.atan2(T_wm[1, 0], T_wm[0, 0])   # heading of map x-axis
        R = euler_matrix(0.0, 0.0, yaw)
        R[0:3, 3] = T_wm[0:3, 3]
        T_wm = R

    q = quaternion_from_matrix(T_wm)
    out = TransformStamped()
    out.header.stamp = rospy.Time.now()
    out.header.frame_id = world_frame
    out.child_frame_id = map_frame
    out.transform.translation.x, out.transform.translation.y, \
        out.transform.translation.z = T_wm[0:3, 3]
    out.transform.rotation.x, out.transform.rotation.y, \
        out.transform.rotation.z, out.transform.rotation.w = q

    rospy.loginfo('vicon_map_align: broadcasting %s -> %s: t=[%.3f %.3f %.3f] '
                  'yaw=%.1f deg%s', world_frame, map_frame,
                  T_wm[0, 3], T_wm[1, 3], T_wm[2, 3],
                  math.degrees(math.atan2(T_wm[1, 0], T_wm[0, 0])),
                  ' (yaw-only)' if yaw_only else '')
    # broadcast on /tf (not /tf_static): latched tf_static messages don't
    # reliably reach foxglove_bridge clients across restarts; a periodic
    # re-send always does. The transform itself stays constant.
    bcast = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(float(rospy.get_param('~broadcast_hz', 10.0)))
    while not rospy.is_shutdown():
        out.header.stamp = rospy.Time.now()
        bcast.sendTransform(out)
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break


if __name__ == '__main__':
    main()
