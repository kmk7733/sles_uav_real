#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mission supervisor: takeoff -> hover -> MPPI to the goal -> land -> disarm.

    /rogx2/commander/set_pose  (planar_planner_node, FCU frame)
                |
                |  forwarded ONLY in the MISSION state
                v
    mission_node  ---->  mavros/setpoint_raw/local  ---->  PX4

WHY THIS REPLACES setpoint_buffer.py
setpoint_buffer.py is the only other writer of mavros/setpoint_raw/local, and
two writers on that topic is two people flying the same aircraft. It also owns
the takeoff itself, which is the wrong place for it: the takeoff has to finish
before the planner may command anything, and the buffer has no way to know
whether the planner is even alive. So this node takes over the whole stream and
setpoint_buffer.py MUST NOT run alongside it -- startup refuses if it does.

    Terminal A:  ROS_NAMESPACE=rogx2 python3 planar_planner_node.py   (dry_run:=false)
    Terminal B:  ROS_NAMESPACE=rogx2 python3 mission_node.py
    then:        rosservice call /rogx2/mission_node/start

STATES
    WAIT     have pose + FCU link; waiting for the operator's start call
    STREAM   publish the current position at pub_rate, then request OFFBOARD
             and ARM. PX4 refuses OFFBOARD unless setpoints are ALREADY
             streaming, which is what this state exists for.
    TAKEOFF  ramp z from the ground to takeoff_height at takeoff_speed, x/y and
             yaw frozen
    HOVER    hold; hand the altitude over to the planner's own z; settle
    MISSION  forward the planner's setpoints verbatim (bounded, see _guard)
    HOLD     freeze where we are. Reached by the ~hold service, by a fence
             violation, or by mission_timeout. Never lands by itself.
    LAND     descend at landing_speed with x/y frozen
    DISARM   call arming(False) until /mavros/state says disarmed
    DONE     stop publishing
    PILOT    the RC took the aircraft out of OFFBOARD -- stop publishing and
             stay stopped. Never fight or silently resume behind the pilot.

TRANSITIONS READ /mavros/state, NOT THE SERVICE RETURN VALUE
FlightModes.arm()/offboard() in guidance_library.py are written as
`if self.armService(True): return True` -- a service response OBJECT, which is
truthy whether or not .success is set, so both always return True. Every
arm/offboard/disarm check here is against the /mavros/state feedback instead.

LANDING follows path_generation.py: a constant-rate descent in OFFBOARD (ours,
not AUTO.LAND, so the profile is one we chose) followed by disarm. Two
differences, both deliberate:
  - x/y are held by POSITION rather than left to a zero velocity command, which
    drifts with estimator bias. ~land_mode:=velocity restores the original.
  - the descent distance comes from the measured ground z, not from the assumed
    takeoff height, and touchdown is confirmed from mavros/extended_state
    before disarming rather than from a stopwatch alone.
"""

import copy
import sys
import threading

import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State, ExtendedState
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from guidance_library import Controller


WAIT, STREAM, TAKEOFF, HOVER, MISSION, HOLD, LAND, DISARM, DONE, PILOT = (
    "WAIT", "STREAM", "TAKEOFF", "HOVER", "MISSION", "HOLD", "LAND", "DISARM",
    "DONE", "PILOT")


def yaw_of(pose):
    q = pose.pose.orientation
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


class MissionNode(object):

    def __init__(self):
        rospy.init_node("mission_node")
        self.controller = Controller()
        self.lock = threading.Lock()

        # ------------------------------------------------------------ params
        self.pub_rate = rospy.get_param("~pub_rate", 20.0)
        self.takeoff_height = rospy.get_param("~takeoff_height", 1.0)
        self.takeoff_speed = rospy.get_param("~takeoff_speed", 0.3)
        self.hover_tol = rospy.get_param("~hover_tol", 0.15)
        self.hover_settle = rospy.get_param("~hover_settle", 3.0)
        # Handing the altitude over. The planner commands z0 in vicon/world and
        # converts it through the world->FCU alignment, so its z lands a few
        # centimetres away from our FCU-frame takeoff_height (the Vicon floor
        # is not the EKF2 origin -- measured ~0.14 m here). Stepping into that
        # difference at MISSION entry is a needless vertical transient, so
        # HOVER slews onto the planner's z first, at z_slew m/s.
        self.z_slew = rospy.get_param("~z_slew", 0.15)

        self.sp_topic = rospy.get_param("~sp_topic", "commander/set_pose")
        self.sp_timeout = rospy.get_param("~sp_timeout", 0.5)
        self.require_planner = rospy.get_param("~require_planner", True)
        self.arrived_topic = rospy.get_param("~arrived_topic",
                                             "/goal_arrive_tf")
        self.arrive_hold = rospy.get_param("~arrive_hold", 1.0)
        self.mission_timeout = rospy.get_param("~mission_timeout", 120.0)

        self.land_on_arrive = rospy.get_param("~land_on_arrive", True)
        self.land_mode = rospy.get_param("~land_mode", "position")
        self.landing_speed = rospy.get_param("~landing_speed", 0.4)
        # How far BELOW the recorded ground the position setpoint is allowed to
        # go. PX4's land detector needs the thrust to fall off, which only
        # happens once the setpoint is pushing into the floor. Bounded, because
        # this is the one command in the file that is deliberately infeasible.
        self.land_push = rospy.get_param("~land_push", 0.20)
        self.land_touch = rospy.get_param("~land_touch", 0.12)
        self.land_timeout = rospy.get_param("~land_timeout", 10.0)
        self.disarm_after_land = rospy.get_param("~disarm_after_land", True)

        # Guards on anything forwarded from the planner. The planner has its
        # own ~max_setpoint_step, but this node is the last thing between a
        # planner bug and the motors, so it does not take that on trust.
        self.max_step = rospy.get_param("~max_step", 1.2)
        self.fence_r = rospy.get_param("~fence_r", 6.0)
        self.fence_z = rospy.get_param("~fence_z", 2.0)
        self.pose_timeout = rospy.get_param("~pose_timeout", 0.5)

        self.auto_start = rospy.get_param("~auto_start", False)

        # ------------------------------------------------------------- state
        self.fcu_pose = None
        self.fcu_stamp = 0.0
        self.mav_state = None
        self.landed_state = None
        self.sp = None                  # last planner setpoint
        self.sp_stamp = 0.0
        self.arrived = False
        self.arrived_since = None

        self.state = WAIT
        self.t_state = rospy.get_time()
        self.start_req = self.auto_start
        self.hold_req = False
        self.land_req = False
        self.resume_req = False

        self.home = None                # (x, y, z_ground, yaw) at STREAM entry
        self.was_offboard = False
        self.was_armed = False
        self.cmd = None                 # last PositionTarget we published

        # ----------------------------------------------------------- ROS I/O
        rospy.Subscriber("mavros/local_position/pose", PoseStamped,
                         self._pose_cb, queue_size=1)
        rospy.Subscriber("mavros/state", State, self._state_cb, queue_size=1)
        rospy.Subscriber("mavros/extended_state", ExtendedState,
                         self._ext_cb, queue_size=1)
        rospy.Subscriber(self.sp_topic, PositionTarget, self._sp_cb,
                         queue_size=1)
        rospy.Subscriber(self.arrived_topic, Bool, self._arrived_cb,
                         queue_size=1)

        self.pub_sp = rospy.Publisher("mavros/setpoint_raw/local",
                                      PositionTarget, queue_size=10)
        self.pub_state = rospy.Publisher("~state", String, queue_size=1,
                                         latch=True)

        self.srv_arm = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.srv_mode = rospy.ServiceProxy("mavros/set_mode", SetMode)

        rospy.Service("~start", Trigger, self._srv_start)
        rospy.Service("~hold", Trigger, self._srv_hold)
        rospy.Service("~resume", Trigger, self._srv_resume)
        rospy.Service("~land", Trigger, self._srv_land)

        rospy.loginfo("[mission] takeoff %.2f m @ %.2f m/s, land %s @ %.2f m/s,"
                      " goal-arrival from %s", self.takeoff_height,
                      self.takeoff_speed, self.land_mode, self.landing_speed,
                      self.arrived_topic)
        if not self.land_on_arrive:
            rospy.logwarn("[mission] ~land_on_arrive is FALSE -- the mission "
                          "ends in HOLD and you land it by hand")

    # ------------------------------------------------------------- callbacks

    def _pose_cb(self, msg):
        with self.lock:
            self.fcu_pose = msg
            self.fcu_stamp = rospy.get_time()

    def _state_cb(self, msg):
        with self.lock:
            self.mav_state = msg

    def _ext_cb(self, msg):
        with self.lock:
            self.landed_state = msg.landed_state

    def _sp_cb(self, msg):
        with self.lock:
            self.sp = msg
            self.sp_stamp = rospy.get_time()

    def _arrived_cb(self, msg):
        with self.lock:
            if msg.data:
                # Not `and not self.arrived`: the flag is latched and the
                # planner republishes True every tick, so an edge test would
                # never re-arm the timer after MISSION entry clears it, and the
                # vehicle would hover over the goal forever.
                if self.arrived_since is None:
                    self.arrived_since = rospy.get_time()
            else:
                self.arrived_since = None
            self.arrived = bool(msg.data)

    # -------------------------------------------------------------- services

    def _srv_start(self, _req):
        if self.state != WAIT:
            return TriggerResponse(False, "already started (%s)" % self.state)
        self.start_req = True
        return TriggerResponse(True, "starting")

    def _srv_hold(self, _req):
        self.hold_req = True
        return TriggerResponse(True, "holding")

    def _srv_resume(self, _req):
        if self.state != HOLD:
            return TriggerResponse(False, "not holding (%s)" % self.state)
        self.resume_req = True
        return TriggerResponse(True, "resuming mission")

    def _srv_land(self, _req):
        if self.state in (WAIT, DONE, PILOT, DISARM):
            return TriggerResponse(False, "cannot land from %s" % self.state)
        self.land_req = True
        return TriggerResponse(True, "landing")

    # ---------------------------------------------------------------- helpers

    def _enter(self, state, why=""):
        if state == self.state:
            return
        rospy.loginfo("[mission] %s -> %s %s", self.state, state, why)
        self.state = state
        self.t_state = rospy.get_time()
        self.pub_state.publish(String(data=state))

    @property
    def elapsed(self):
        return rospy.get_time() - self.t_state

    def _snapshot(self):
        with self.lock:
            pose, age = self.fcu_pose, rospy.get_time() - self.fcu_stamp
            st, sp = self.mav_state, self.sp
            sp_age = rospy.get_time() - self.sp_stamp
        return pose, age, st, sp, sp_age

    def _hold_here(self, pose):
        return (pose.pose.position.x, pose.pose.position.y,
                pose.pose.position.z, yaw_of(pose))

    def _send_pos(self, x, y, z, yaw):
        t = self.controller.construct_target(x, y, z, yaw)
        self.cmd = t
        self.pub_sp.publish(t)

    def _send_vel(self, vx, vy, vz, yaw):
        t = self.controller.construct_target_velocity(vx, vy, vz, yaw)
        self.cmd = t
        self.pub_sp.publish(t)

    def _guard(self, sp, pose):
        """Bound a planner setpoint before it reaches PX4.

        Returns a PositionTarget, or None if it is unusable. Rejecting is safe
        here: the caller falls back to holding, which is what the vehicle
        should do when it stops believing the planner.
        """
        p = sp.position
        if not all(np.isfinite([p.x, p.y, p.z])):
            rospy.logerr_throttle(1.0, "[mission] non-finite setpoint dropped")
            return None
        # A COPY. self.sp still points at the message the subscriber handed us,
        # and the planner may publish slower than this loop runs, so clamping
        # in place would clamp the SAME setpoint again on the next tick and
        # walk it back toward the vehicle a bit at a time until it stalls.
        out = copy.deepcopy(sp)
        px, py = pose.pose.position.x, pose.pose.position.y
        d = float(np.hypot(p.x - px, p.y - py))
        if d > self.max_step:
            f = self.max_step / d
            rospy.logwarn_throttle(1.0, "[mission] setpoint %.2f m away -> "
                                        "clamped to %.2f", d, self.max_step)
            out.position.x = px + (p.x - px) * f
            out.position.y = py + (p.y - py) * f
        # The planner is fixed-altitude by construction and never plans in z,
        # so a z that wanders is a symptom, not a command to follow.
        out.position.z = float(np.clip(p.z, self.hover_z - 0.5,
                                       self.hover_z + 0.5))
        out.header.stamp = rospy.Time.now()
        return out

    def _fence_ok(self, pose):
        x0, y0, z0, _ = self.home
        dx = pose.pose.position.x - x0
        dy = pose.pose.position.y - y0
        dz = pose.pose.position.z - z0
        if float(np.hypot(dx, dy)) > self.fence_r:
            rospy.logerr("[mission] FENCE: %.2f m from home > %.2f",
                         float(np.hypot(dx, dy)), self.fence_r)
            return False
        if dz > self.fence_z:
            rospy.logerr("[mission] FENCE: %.2f m above home > %.2f", dz,
                         self.fence_z)
            return False
        return True

    def _touched_down(self, pose):
        with self.lock:
            ls = self.landed_state
        if ls == ExtendedState.LANDED_STATE_ON_GROUND:
            return True
        # Fallback for a stack that does not publish extended_state: back on
        # the recorded ground height, and only after the nominal descent time.
        return (pose.pose.position.z - self.home[2] < self.land_touch
                and self.elapsed > self.land_nominal)

    # ------------------------------------------------------------------- run

    def run(self):
        rate = rospy.Rate(self.pub_rate)
        self.hover_z = self.takeoff_height
        self.land_nominal = 0.0
        self.pub_state.publish(String(data=self.state))

        while not rospy.is_shutdown():
            pose, pose_age, st, sp, sp_age = self._snapshot()

            # ---------------------------------------------- global overrides
            if st is not None and st.armed:
                self.was_armed = True

            # Disarm is checked FIRST. PX4 commonly drops out of OFFBOARD in
            # the same breath as disarming, and reading that as an RC takeover
            # would report PILOT for what is actually a completed landing.
            # Gated on was_ARMED, not was_offboard: OFFBOARD is accepted
            # BEFORE the arm request in STREAM, so keying this off the mode
            # alone declares the flight over one tick after it is allowed to
            # begin.
            if (self.was_armed and st is not None and not st.armed
                    and self.state not in (DONE, WAIT, PILOT)):
                self._enter(DONE, "-- disarmed")
            elif st is not None and st.mode == "OFFBOARD":
                self.was_offboard = True
            elif self.was_offboard and self.state not in (PILOT, DONE):
                # The RC took it. Stop commanding, and do not come back on our
                # own if OFFBOARD returns -- restart the node deliberately.
                self._enter(PILOT, "-- RC took over (mode=%s), setpoint "
                                   "stream stopped" %
                            (st.mode if st else "?"))

            if self.state in (DONE, PILOT):
                rate.sleep()
                continue

            if pose is None:
                rospy.logwarn_throttle(2.0, "[mission] waiting for "
                                            "mavros/local_position/pose")
                rate.sleep()
                continue

            if pose_age > self.pose_timeout and self.state not in (WAIT,):
                # Keep the stream alive with the last command -- dropping it
                # would trip PX4's offboard failsafe -- but do not integrate
                # anything new on a stale estimate.
                rospy.logerr_throttle(1.0, "[mission] pose stale %.2f s -- "
                                           "repeating last setpoint", pose_age)
                if self.cmd is not None:
                    self.cmd.header.stamp = rospy.Time.now()
                    self.pub_sp.publish(self.cmd)
                rate.sleep()
                continue

            if self.land_req and self.state not in (LAND, DISARM):
                self._begin_land(pose, "-- ~land requested")
            elif self.hold_req and self.state in (TAKEOFF, HOVER, MISSION):
                self.hold_req = False
                self.hold_pose = self._hold_here(pose)
                self._enter(HOLD, "-- ~hold requested")

            if self.state in (TAKEOFF, HOVER, MISSION) and \
                    not self._fence_ok(pose):
                self.hold_pose = self._hold_here(pose)
                self._enter(HOLD, "-- geofence")

            getattr(self, "_do_" + self.state.lower())(pose, st, sp, sp_age)
            rate.sleep()

    # ----------------------------------------------------------- state bodies

    def _do_wait(self, pose, st, sp, sp_age):
        if st is None or not st.connected:
            rospy.logwarn_throttle(2.0, "[mission] waiting for the FCU link")
            return
        if not self.start_req:
            rospy.loginfo_throttle(5.0, "[mission] READY -- call "
                                        "rosservice call %s/start",
                                   rospy.get_name())
            return
        if self.require_planner and sp_age > self.sp_timeout:
            rospy.logwarn_throttle(2.0, "[mission] no planner setpoints on %s "
                                        "-- is planar_planner_node running "
                                        "with _dry_run:=false?", self.sp_topic)
            return
        self.home = self._hold_here(pose)
        rospy.loginfo("[mission] home (%.2f, %.2f, %.2f) yaw %.1f deg",
                      self.home[0], self.home[1], self.home[2],
                      np.degrees(self.home[3]))
        self._enter(STREAM)

    def _do_stream(self, pose, st, sp, sp_age):
        x, y, z, yaw = self.home
        self._send_pos(x, y, z, yaw)

        # PX4 will not accept OFFBOARD until it has seen a setpoint stream for
        # a moment, so the first second here is pure streaming.
        if self.elapsed < 1.0:
            return

        if st.mode != "OFFBOARD":
            if int(self.elapsed * self.pub_rate) % int(self.pub_rate) == 0:
                try:
                    self.srv_mode(custom_mode="OFFBOARD")
                except rospy.ServiceException as e:
                    rospy.logerr_throttle(2.0, "[mission] set_mode: %s", e)
            rospy.loginfo_throttle(2.0, "[mission] requesting OFFBOARD "
                                        "(mode=%s)", st.mode)
            return

        if not st.armed:
            if int(self.elapsed * self.pub_rate) % int(self.pub_rate) == 0:
                try:
                    self.srv_arm(True)
                except rospy.ServiceException as e:
                    rospy.logerr_throttle(2.0, "[mission] arming: %s", e)
            rospy.loginfo_throttle(2.0, "[mission] OFFBOARD accepted, "
                                        "requesting ARM")
            return

        # Both confirmed by /mavros/state, not by a service return value.
        self.takeoff_t0 = rospy.get_time()
        self.takeoff_z0 = pose.pose.position.z
        self._enter(TAKEOFF, "-- ARMED in OFFBOARD")

    def _do_takeoff(self, pose, st, sp, sp_age):
        x, y, _, yaw = self.home
        dz = self.takeoff_height - (self.takeoff_z0 - self.home[2])
        dt = max(dz, 0.0) / max(self.takeoff_speed, 1e-3)
        f = 1.0 if dt <= 1e-3 else min(self.elapsed / dt, 1.0)
        z = self.takeoff_z0 + f * dz
        self._send_pos(x, y, z, yaw)

        if f >= 1.0 and self.elapsed > dt + 0.5:
            self.hover_z = self.home[2] + self.takeoff_height
            self.settled_since = None
            self._enter(HOVER)

    def _do_hover(self, pose, st, sp, sp_age):
        x, y, _, yaw = self.home
        target = self.home[2] + self.takeoff_height

        # Slew onto the planner's altitude so MISSION entry is not a step.
        if sp is not None and sp_age <= self.sp_timeout:
            want = float(sp.position.z)
            if abs(want - target) < 0.5:
                step = self.z_slew / self.pub_rate
                target = self.hover_z + float(np.clip(want - self.hover_z,
                                                      -step, step))
        self.hover_z = target
        self._send_pos(x, y, self.hover_z, yaw)

        err = abs(pose.pose.position.z - self.hover_z)
        now = rospy.get_time()
        if err > self.hover_tol:
            self.settled_since = None
        elif self.settled_since is None:
            self.settled_since = now

        if self.settled_since is None or now - self.settled_since < \
                self.hover_settle:
            rospy.loginfo_throttle(2.0, "[mission] hover z=%.2f (want %.2f, "
                                        "err %.02f)", pose.pose.position.z,
                                   self.hover_z, err)
            return

        if sp_age > self.sp_timeout:
            rospy.logwarn_throttle(2.0, "[mission] hovering, waiting for the "
                                        "planner on %s (last %.1f s ago)",
                                   self.sp_topic, sp_age)
            return

        with self.lock:
            self.arrived_since = None   # ignore a latch from before takeoff
        self._enter(MISSION, "-- hover settled, planner live")

    def _do_mission(self, pose, st, sp, sp_age):
        if sp_age > self.sp_timeout:
            rospy.logerr_throttle(1.0, "[mission] planner silent %.2f s -- "
                                       "holding", sp_age)
            self.hold_pose = self._hold_here(pose)
            self._enter(HOLD, "-- planner silent")
            return

        out = self._guard(sp, pose)
        if out is None:
            self.hold_pose = self._hold_here(pose)
            self._enter(HOLD, "-- unusable setpoint")
            return
        self.cmd = out
        self.pub_sp.publish(out)

        with self.lock:
            arrived, since = self.arrived, self.arrived_since
        if arrived and since is not None and \
                rospy.get_time() - since >= self.arrive_hold:
            if self.land_on_arrive:
                self._begin_land(pose, "-- GOAL REACHED")
            else:
                self.hold_pose = self._hold_here(pose)
                self._enter(HOLD, "-- GOAL REACHED (~land_on_arrive false)")
            return

        if self.elapsed > self.mission_timeout:
            self.hold_pose = self._hold_here(pose)
            self._enter(HOLD, "-- mission_timeout %.0f s" %
                        self.mission_timeout)

    def _do_hold(self, pose, st, sp, sp_age):
        x, y, z, yaw = self.hold_pose
        self._send_pos(x, y, z, yaw)
        if self.resume_req:
            self.resume_req = False
            if sp_age <= self.sp_timeout:
                self._enter(MISSION, "-- resumed")
            else:
                rospy.logwarn("[mission] resume refused: planner silent")
        rospy.loginfo_throttle(5.0, "[mission] HOLD at (%.2f, %.2f, %.2f) -- "
                                    "%s/land or %s/resume", x, y, z,
                               rospy.get_name(), rospy.get_name())

    def _begin_land(self, pose, why):
        self.land_req = False
        self.land_pose = self._hold_here(pose)
        drop = max(self.land_pose[2] - self.home[2], 0.0)
        self.land_nominal = drop / max(self.landing_speed, 1e-3)
        rospy.loginfo("[mission] landing from z=%.2f (%.2f above home), "
                      "nominal %.1f s", self.land_pose[2], drop,
                      self.land_nominal)
        self._enter(LAND, why)

    def _do_land(self, pose, st, sp, sp_age):
        x, y, z0, yaw = self.land_pose
        if self.land_mode == "velocity":
            # path_generation.py's descent, kept available: a pure velocity
            # command, so x/y are not held and will drift with estimator bias.
            self._send_vel(0.0, 0.0, -self.landing_speed, yaw)
        else:
            z = max(z0 - self.landing_speed * self.elapsed,
                    self.home[2] - self.land_push)
            self._send_pos(x, y, z, yaw)

        with self.lock:
            have_ext = self.landed_state is not None
        if not have_ext:
            rospy.logwarn_throttle(2.0, "[mission] no mavros/extended_state -- "
                                        "touchdown will be judged from the "
                                        "recorded ground height and the clock")

        if self._touched_down(pose):
            if not self.disarm_after_land:
                self._enter(DONE, "-- touchdown (~disarm_after_land false)")
                return
            self.disarm_tries = 0
            self._enter(DISARM, "-- touchdown")
            return

        if self.elapsed > self.land_nominal + self.land_timeout:
            # No touchdown confirmation. Disarming now could be a disarm in the
            # air, which is the one thing this node must never do on a guess.
            rospy.logerr("[mission] no touchdown after %.1f s -- HOLDING. "
                         "Take it with the RC.", self.elapsed)
            self.hold_pose = self._hold_here(pose)
            self._enter(HOLD, "-- landing did not confirm")

    def _do_disarm(self, pose, st, sp, sp_age):
        if st is not None and not st.armed:
            self._enter(DONE, "-- disarmed after %d request(s)" %
                        self.disarm_tries)
            return
        # Keep the stream alive while asking, so PX4 does not drop OFFBOARD
        # between attempts and hand the aircraft to a failsafe instead.
        x, y, _, yaw = self.land_pose
        self._send_pos(x, y, self.home[2] - self.land_push, yaw)
        if int(self.elapsed * self.pub_rate) % max(int(self.pub_rate / 2), 1):
            return
        self.disarm_tries += 1
        try:
            self.srv_arm(False)
        except rospy.ServiceException as e:
            rospy.logerr_throttle(2.0, "[mission] disarm: %s", e)
        if self.elapsed > 8.0:
            rospy.logerr("[mission] still armed after %.0f s -- disarm it on "
                         "the RC", self.elapsed)

    def _do_done(self, pose, st, sp, sp_age):
        rospy.loginfo_throttle(10.0, "[mission] DONE -- no setpoints are "
                                     "being published")

    def _do_pilot(self, pose, st, sp, sp_age):
        rospy.logwarn_throttle(5.0, "[mission] PILOT IN COMMAND -- restart "
                                    "this node to fly again")


def _refuse_if_buffer_running():
    """Two writers on mavros/setpoint_raw/local is two pilots."""
    try:
        import rosnode
        names = rosnode.get_node_names()
    except Exception:
        return
    clash = [n for n in names if "setpoint_buffer" in n]
    if clash:
        rospy.logfatal("[mission] %s is running and also publishes "
                       "mavros/setpoint_raw/local. Kill it first.",
                       ", ".join(clash))
        sys.exit(1)


if __name__ == "__main__":
    node = MissionNode()
    _refuse_if_buffer_running()
    node.run()
